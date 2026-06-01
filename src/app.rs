use crate::{
    config::Config,
    db::Database,
    design_types::*,
    embed::{
        self, Embedder, EmbeddingBatch, FastEmbedBgeM3Embedder, LocalHashEmbedder, SparseVector,
    },
    graph::{Graph, GraphStore},
    model::{
        MemoryRow, MemoryType, RecallHit, RelationKind, ScoreComponents, clamp_unit,
        epoch_to_rfc3339, json_object_or_empty, normalize_tags, now_epoch, opt_epoch_to_rfc3339,
        parse_time, validate_content,
    },
};
use anyhow::{Context, bail};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Serialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::Arc,
    time::Instant,
};
use time::OffsetDateTime;
use unicode_segmentation::UnicodeSegmentation;
use uuid::Uuid;

const TEMPORAL_ENTITY_WINDOW_SECONDS: i64 = 7 * 86_400;
const TEMPORAL_EDGE_METADATA: &str = r#"{"derived":"temporal_entity_overlap","window_days":7}"#;
pub struct AgskMem {
    pub config: Config,
    db: Database,
    embedder: Arc<dyn Embedder>,
    graph: GraphStore,
}

#[derive(Debug, Serialize)]
pub struct StoreResult {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResult {
    pub deleted: usize,
    pub dry_run: bool,
    pub confirmation_token: Option<String>,
    pub ids: Vec<String>,
}

fn build_embedder(config: &Config) -> anyhow::Result<Arc<dyn Embedder>> {
    match config.embed.provider.trim().to_ascii_lowercase().as_str() {
        "local" | "local-hash" => Ok(Arc::new(LocalHashEmbedder::new(
            config.embed.model.clone(),
            config.embed.dims,
        ))),
        "fastembed" | "fastembed-bgem3" | "bge-m3" | "bgem3" => {
            Ok(Arc::new(FastEmbedBgeM3Embedder::new(
                config.embed.model.clone(),
                config.embed.dims,
                config.embed.cache_dir.clone(),
            )?))
        }
        other => bail!("unsupported embed.provider {other}"),
    }
}
impl AgskMem {
    pub fn open(config: Config) -> anyhow::Result<Self> {
        let db = Database::open(config.db.path.clone())?;
        let embedder = build_embedder(&config)?;
        let app = Self {
            config,
            db,
            embedder,
            graph: GraphStore::default(),
        };
        app.ensure_embedding_meta()?;
        app.repair_index()?;
        Ok(app)
    }

    pub fn db_path(&self) -> &Path {
        self.db.path()
    }

    fn ensure_embedding_meta(&self) -> anyhow::Result<()> {
        let conn = self.db.writer()?;
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_model', ?)",
            [self.embedder.model()],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_dims', ?)",
            [self.embedder.dims().to_string()],
        )?;
        Ok(())
    }

    pub fn store_memory(&self, args: StoreMemoryArgs) -> anyhow::Result<StoreResult> {
        let mut items = args.memories;
        if let Some(content) = non_empty_string(args.content) {
            items.push(StoreOneArgs {
                content,
                tags: args.tags,
                importance: args.importance,
                confidence: args.confidence,
                metadata: args.metadata,
                memory_type: args.memory_type,
                source: non_empty_string(args.source),
                summary: non_empty_string(args.summary),
                timestamp: non_empty_string(args.timestamp),
                t_valid: non_empty_string(args.t_valid),
                t_invalid: non_empty_string(args.t_invalid),
                id: non_empty_string(args.id),
            });
        }
        if items.is_empty() {
            bail!("store_memory requires content or memories");
        }
        let now = now_epoch();
        let mut conn = self.db.writer()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            let id = non_empty_string(item.id).unwrap_or_else(|| Uuid::now_v7().to_string());
            let mut metadata = metadata_map(item.metadata)?;
            let content = govern_content(item.content, &self.config, &mut metadata)?;
            let classification = classify_memory(
                &tx,
                &content,
                item.memory_type.as_deref(),
                item.confidence,
                &mut metadata,
            )?;
            insert_memory(
                &tx,
                &*self.embedder,
                InsertMemory {
                    id: id.clone(),
                    content,
                    summary: non_empty_string(item.summary),
                    memory_type: classification.memory_type,
                    tags: normalize_tags(&item.tags),
                    importance: clamp_unit(item.importance, 0.5, "importance")?,
                    confidence: classification.confidence,
                    metadata: serde_json::to_string(&Value::Object(metadata))?,
                    source: non_empty_string(item.source),
                    created_at: parse_time(non_empty_string(item.timestamp).as_deref())?
                        .unwrap_or(now),
                    t_valid: parse_time(non_empty_string(item.t_valid).as_deref())?,
                    t_invalid: parse_time(non_empty_string(item.t_invalid).as_deref())?,
                },
            )?;
            ids.push(id);
        }
        tx.commit()?;
        drop(conn);
        self.repair_index()?;
        Ok(StoreResult { ids })
    }

    pub fn update_memory(&self, args: UpdateMemoryArgs) -> anyhow::Result<Value> {
        if args.memory_id.is_empty() {
            bail!("memory_id is required");
        }
        let mut conn = self.db.writer()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let existing = tx
            .query_row(
                "SELECT content, summary, type, confidence, metadata FROM memory WHERE id = ?",
                [&args.memory_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, f64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((
            existing_content,
            existing_summary,
            existing_type,
            existing_confidence,
            existing_metadata,
        )) = existing
        else {
            bail!("memory {} not found", args.memory_id);
        };
        let now = now_epoch();
        let mut metadata = metadata_map(Some(serde_json::from_str(&existing_metadata)?))?;
        if let Some(patch) = args.metadata {
            merge_metadata(&mut metadata, metadata_map(Some(patch))?);
        }

        let mut content = existing_content;
        let mut summary = existing_summary;
        let mut content_changed = false;
        let mut summary_changed = false;
        if let Some(new_content) = args.content {
            content = govern_content(new_content, &self.config, &mut metadata)?;
            content_changed = true;
        }
        if let Some(new_summary) = args.summary {
            summary = non_empty_string(Some(new_summary));
            summary_changed = true;
        }

        let explicit_type = args.memory_type.as_deref();
        let mut memory_type = MemoryType::from_i64(existing_type)?;
        let mut confidence = existing_confidence;
        if explicit_type.is_some() || (content_changed && !classification_locked(&metadata)) {
            let classification =
                classify_memory(&tx, &content, explicit_type, args.confidence, &mut metadata)?;
            memory_type = classification.memory_type;
            confidence = classification.confidence;
        } else if let Some(v) = args.confidence {
            confidence = clamp_unit(Some(v), 0.9, "confidence")?;
        }

        let metadata_text = serde_json::to_string(&Value::Object(metadata))?;
        tx.execute(
            "UPDATE memory SET content = ?, summary = ?, type = ?, confidence = ?, metadata = ?, updated_at = ? WHERE id = ?",
            params![content, summary, memory_type as i64, confidence, metadata_text, now, args.memory_id],
        )?;
        if content_changed || summary_changed {
            let text = embedding_text(&content, summary.as_deref());
            upsert_embedding(&tx, &*self.embedder, &args.memory_id, &text, now)?;
            re_enrich_memory(&tx, &args.memory_id, &content, confidence)?;
        }
        if let Some(v) = args.importance {
            tx.execute(
                "UPDATE memory SET importance = ?, updated_at = ? WHERE id = ?",
                params![clamp_unit(Some(v), 0.5, "importance")?, now, args.memory_id],
            )?;
        }
        if let Some(v) = args.relevance {
            tx.execute(
                "UPDATE memory SET relevance = ?, updated_at = ? WHERE id = ?",
                params![clamp_unit(Some(v), 0.5, "relevance")?, now, args.memory_id],
            )?;
        }
        if let Some(v) = args.reliability {
            tx.execute(
                "UPDATE memory SET reliability = ?, updated_at = ? WHERE id = ?",
                params![
                    clamp_unit(Some(v), 0.7, "reliability")?,
                    now,
                    args.memory_id
                ],
            )?;
        }
        if let Some(source) = args.source {
            tx.execute(
                "UPDATE memory SET source = ?, updated_at = ? WHERE id = ?",
                params![source, now, args.memory_id],
            )?;
        }
        if let Some(value) = args.t_valid {
            tx.execute(
                "UPDATE memory SET t_valid = ?, updated_at = ? WHERE id = ?",
                params![parse_time(Some(&value))?, now, args.memory_id],
            )?;
        }
        if let Some(value) = args.t_invalid {
            tx.execute(
                "UPDATE memory SET t_invalid = ?, updated_at = ? WHERE id = ?",
                params![parse_time(Some(&value))?, now, args.memory_id],
            )?;
        }
        if let Some(value) = args.archived {
            tx.execute(
                "UPDATE memory SET archived = ?, updated_at = ? WHERE id = ?",
                params![if value { 1 } else { 0 }, now, args.memory_id],
            )?;
        }
        if let Some(value) = args.protected {
            tx.execute(
                "UPDATE memory SET protected = ?, updated_at = ? WHERE id = ?",
                params![if value { 1 } else { 0 }, now, args.memory_id],
            )?;
        }
        if let Some(tags) = args.tags {
            tx.execute("DELETE FROM tag WHERE memory_id = ?", [&args.memory_id])?;
            tx.execute(
                "DELETE FROM tag_prefix WHERE memory_id = ?",
                [&args.memory_id],
            )?;
            for tag in normalize_tags(&tags) {
                insert_tag(&tx, &args.memory_id, &tag)?;
            }
            re_enrich_memory(&tx, &args.memory_id, &content, confidence)?;
        }
        tx.commit()?;
        drop(conn);
        self.repair_index()?;
        Ok(json!({"updated": true, "id": args.memory_id}))
    }

    pub fn delete_memory(&self, args: DeleteMemoryArgs) -> anyhow::Result<DeleteResult> {
        let memory_id = non_empty_string(args.memory_id);
        let ids = if let Some(id) = memory_id.as_ref() {
            vec![id.clone()]
        } else if !args.tags.is_empty() {
            self.ids_for_tags(&normalize_tags(&args.tags))?
        } else {
            bail!("delete_memory requires memory_id or tags");
        };
        if memory_id.is_none() {
            let token = delete_confirmation_token(&ids);
            if args.dry_run {
                return Ok(DeleteResult {
                    deleted: ids.len(),
                    dry_run: true,
                    confirmation_token: Some(token),
                    ids,
                });
            }
            if non_empty_string(args.confirmation_token).as_deref() != Some(token.as_str()) {
                bail!("bulk tag delete requires dry_run first and matching confirmation_token");
            }
        }
        let mut conn = self.db.writer()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut deleted = 0;
        for id in &ids {
            deleted += tx.execute("DELETE FROM memory WHERE id = ?", [id])?;
        }
        tx.commit()?;
        drop(conn);
        self.repair_index()?;
        Ok(DeleteResult {
            deleted,
            dry_run: false,
            confirmation_token: None,
            ids,
        })
    }

    fn ids_for_tags(&self, tags: &[String]) -> anyhow::Result<Vec<String>> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.db.read_connection()?;
        let mut sql = "SELECT memory_id FROM tag WHERE tag IN (".to_string();
        sql.push_str(&vec!["?"; tags.len()].join(","));
        sql.push_str(") GROUP BY memory_id HAVING COUNT(DISTINCT tag) = ? ORDER BY memory_id");
        let mut stmt = conn.prepare(&sql)?;
        let mut params: Vec<&dyn rusqlite::ToSql> =
            tags.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        let count = tags.len() as i64;
        params.push(&count);
        let rows = stmt.query_map(params.as_slice(), |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn associate_memories(&self, args: AssociateArgs) -> anyhow::Result<Value> {
        let kind = RelationKind::parse_authorable(&args.relation_type)?;
        self.upsert_edge(args, kind, true)
    }

    pub fn import_relation(&self, args: AssociateArgs) -> anyhow::Result<Value> {
        let kind = RelationKind::parse(&args.relation_type)?;
        self.upsert_edge(args, kind, false)
    }

    fn upsert_edge(
        &self,
        args: AssociateArgs,
        kind: RelationKind,
        rebuild_graph: bool,
    ) -> anyhow::Result<Value> {
        let strength = clamp_unit(args.strength, 0.5, "strength")?;
        let confidence = clamp_unit(args.confidence, 0.5, "confidence")?;
        let metadata = json_object_or_empty(args.metadata)?;
        let now = now_epoch();
        let mut conn = self.db.writer()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        assert_memory_exists(&tx, &args.memory1_id)?;
        assert_memory_exists(&tx, &args.memory2_id)?;
        tx.execute("INSERT INTO edge(src, dst, kind, strength, confidence, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(src, dst, kind) DO UPDATE SET strength=excluded.strength, confidence=excluded.confidence, metadata=excluded.metadata, updated_at=excluded.updated_at", params![args.memory1_id, args.memory2_id, kind as i64, strength, confidence, metadata, now, now])?;
        tx.commit()?;
        drop(conn);
        if rebuild_graph {
            self.repair_index()?;
        }
        Ok(
            json!({"associated": true, "src": args.memory1_id, "dst": args.memory2_id, "type": kind.as_str()}),
        )
    }

    pub fn recall_memory(&self, mut args: RecallArgs) -> anyhow::Result<Value> {
        let start = Instant::now();
        if let Some(id) = non_empty_string(args.memory_id.take()) {
            let conn = self.db.read_connection()?;
            let Some(row) =
                fetch_memory_row(&conn, &id, args.current_only.unwrap_or(true), now_epoch())?
            else {
                return Ok(json!({"results": []}));
            };
            self.touch(std::slice::from_ref(&id))?;
            return Ok(json!({"results": [row]}));
        }
        if args.query.is_none() && !args.queries.is_empty() {
            args.query = Some(args.queries.join(" "));
        }
        let query = args.query.clone().unwrap_or_default();
        let tags = normalize_tags(&args.tags);
        let exclude_tags = normalize_tags(&args.exclude_tags);
        let limit = args.limit.unwrap_or(10).clamp(1, 200);
        let offset = args.offset.or(args.cursor).unwrap_or(0);
        if query.trim().is_empty() && !tags.is_empty() {
            let page = self.enumerate_tags(
                &tags,
                &exclude_tags,
                offset,
                limit,
                args.current_only.unwrap_or(true),
                parse_time(args.as_of.as_deref())?.unwrap_or_else(now_epoch),
            )?;
            return Ok(
                json!({"results": page.0, "offset": offset, "limit": limit, "has_more": page.1}),
            );
        }
        let hits = self.rank_recall(&query, &args, limit + offset)?;
        let results: Vec<_> = hits.into_iter().skip(offset).take(limit).collect();
        let ids: Vec<String> = results.iter().map(|h| h.memory.id.clone()).collect();
        self.touch(&ids)?;
        self.record_metric(
            "recall_memory",
            start.elapsed().as_millis() as i64,
            ids.len(),
        )?;
        Ok(json!({"results": results, "offset": offset, "limit": limit}))
    }

    fn rank_recall(
        &self,
        query: &str,
        args: &RecallArgs,
        limit: usize,
    ) -> anyhow::Result<Vec<RecallHit>> {
        let conn = self.db.read_connection()?;
        let now = parse_time(args.as_of.as_deref())?.unwrap_or_else(now_epoch);
        let current_only = args.current_only.unwrap_or(true);
        let tags = normalize_tags(&args.tags);
        let exclude_tags = normalize_tags(&args.exclude_tags);
        let mut context_tags = normalize_tags(&args.context_tags);
        enrich_context_tags(args, &mut context_tags);
        let decomposed = if args.auto_decompose {
            decompose_query(query)
        } else {
            Vec::new()
        };
        let mut query_batch = self.embedder.embed_for_recall(&[query])?;
        let query_vec = query_batch.dense.pop().unwrap_or_default();
        let query_sparse = query_batch.sparse.pop().unwrap_or_default();
        let query_colbert = query_batch.colbert.pop().unwrap_or_default();
        let query_embedding = QueryEmbedding {
            dense: &query_vec,
            sparse: &query_sparse,
            colbert: &query_colbert,
        };

        let mut candidate_ids = HashSet::new();
        let sparse_scores =
            sparse_candidates(&conn, &query_sparse, self.config.recall.per_source_limit)?;
        for id in sparse_scores.keys() {
            candidate_ids.insert(id.clone());
        }
        for id in vector_candidates(
            &conn,
            &query_vec,
            self.embedder.model(),
            self.config.recall.per_source_limit,
        )? {
            candidate_ids.insert(id);
        }
        for id in fts_candidates(&conn, query, self.config.recall.per_source_limit)? {
            candidate_ids.insert(id);
        }
        for id in exact_phrase_candidates(&conn, query, self.config.recall.per_source_limit)? {
            candidate_ids.insert(id);
        }
        for id in entity_candidates(&conn, query, self.config.recall.per_source_limit)? {
            candidate_ids.insert(id);
        }
        for id in fallback_candidates(&conn, self.config.recall.per_source_limit)? {
            candidate_ids.insert(id);
        }
        for term in &decomposed {
            for id in fts_candidates(&conn, term, self.config.recall.per_source_limit / 2)? {
                candidate_ids.insert(id);
            }
            for id in exact_phrase_candidates(&conn, term, self.config.recall.per_source_limit / 2)?
            {
                candidate_ids.insert(id);
            }
            for id in entity_candidates(&conn, term, self.config.recall.per_source_limit / 2)? {
                candidate_ids.insert(id);
            }
        }
        for id in &args.priority_ids {
            candidate_ids.insert(id.clone());
        }

        let graph = self.graph.load();
        let (time_query_start, time_query_end) = time_query_bounds(args.time_query.as_deref())?;
        let start_bound = args.start.as_deref().or(time_query_start.as_deref());
        let end_bound = args.end.as_deref().or(time_query_end.as_deref());
        let filters = FilterSpec {
            tags: &tags,
            exclude_tags: &exclude_tags,
            current_only,
            as_of: now,
            start: start_bound,
            end: end_bound,
            time_range: args.time_range.as_ref(),
        };
        let mut seed_scores = HashMap::new();
        let mut raw = Vec::new();
        for id in candidate_ids {
            if !passes_filters(&conn, &id, &filters)? {
                continue;
            }
            let Some(memory) = fetch_memory_row(&conn, &id, false, now)? else {
                continue;
            };
            let components = compute_components(
                &conn,
                &memory,
                query,
                &query_embedding,
                sparse_scores.get(&id).copied(),
                &context_tags,
            )?;
            let score = score_components(&self.config.recall.weights, &components);
            if score > 0.0 || args.priority_ids.contains(&id) {
                seed_scores.insert(id.clone(), score.max(0.01) as f32);
                raw.push((id, memory, components, score));
            }
        }
        let ppr = if args.expand_relations || args.expand_entities {
            graph.forward_push(
                &seed_scores,
                self.config.ppr.alpha,
                self.config.ppr.epsilon,
                self.config.ppr.max_pushes,
            )
        } else {
            HashMap::new()
        };
        for id in ppr.keys() {
            if raw.iter().any(|(candidate, _, _, _)| candidate == id) {
                continue;
            }
            let allowed = !args.expand_respect_tags || passes_filters(&conn, id, &filters)?;
            if !allowed {
                continue;
            }
            if let Some(memory) = fetch_memory_row(&conn, id, false, now)? {
                if args
                    .expand_min_importance
                    .is_some_and(|floor| memory.importance < floor)
                {
                    continue;
                }
                let mut components = compute_components(
                    &conn,
                    &memory,
                    query,
                    &query_embedding,
                    sparse_scores.get(id).copied(),
                    &context_tags,
                )?;
                components.ppr = f64::from(*ppr.get(id).unwrap_or(&0.0));
                let score = score_components(&self.config.recall.weights, &components);
                raw.push((id.clone(), memory, components, score));
            }
        }
        let mut hits = Vec::new();
        for (id, memory, mut components, _) in raw {
            components.ppr = f64::from(*ppr.get(&id).unwrap_or(&0.0));
            let score = score_components(&self.config.recall.weights, &components);
            hits.push(RecallHit {
                memory,
                score,
                components,
                state_replacements: active_replacements(&conn, &id, now)?,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.memory.updated_at.cmp(&a.memory.updated_at))
        });
        apply_score_floor(&mut hits, args, self.config.recall.adaptive_floor);
        sort_hits(&mut hits, args, query);
        if score_sort(args) && (limit > 10 || query.unicode_words().count() <= 2) {
            apply_mmr(&mut hits, limit, self.config.recall.mmr_lambda);
        }
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn enumerate_tags(
        &self,
        tags: &[String],
        exclude_tags: &[String],
        offset: usize,
        limit: usize,
        current_only: bool,
        as_of: i64,
    ) -> anyhow::Result<(Vec<MemoryRow>, bool)> {
        let conn = self.db.read_connection()?;
        let filters = FilterSpec {
            tags,
            exclude_tags,
            current_only,
            as_of,
            start: None,
            end: None,
            time_range: None,
        };
        let ids = self.ids_for_tags(tags)?;
        let mut rows = Vec::new();
        for id in ids {
            if !passes_filters(&conn, &id, &filters)? {
                continue;
            }
            if let Some(row) = fetch_memory_row(&conn, &id, false, as_of)? {
                rows.push(row);
            }
        }
        rows.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        let has_more = rows.len() > offset + limit;
        Ok((
            rows.into_iter().skip(offset).take(limit).collect(),
            has_more,
        ))
    }

    pub fn startup_recall(&self, limit: usize) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let mut stmt = conn.prepare("SELECT id FROM memory WHERE archived = 0 ORDER BY importance DESC, updated_at DESC LIMIT ?")?;
        let ids = stmt
            .query_map([limit as i64], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let mut rows = Vec::new();
        for id in ids {
            if let Some(row) = fetch_memory_row(&conn, &id, true, now_epoch())? {
                rows.push(row);
            }
        }
        Ok(json!({"results": rows}))
    }

    pub fn check_database_health(&self) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let tables = [
            "memory",
            "tag",
            "tag_prefix",
            "entity",
            "memory_entity",
            "edge",
            "statement",
            "embedding",
            "embedding_job",
            "embedding_sparse",
            "embedding_colbert",
            "enrichment_job",
            "classification_cache",
        ];
        let mut counts = serde_json::Map::new();
        for table in tables {
            counts.insert(
                table.to_string(),
                json!(
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| row
                        .get::<_, i64>(0))?
                ),
            );
        }
        let missing_embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM memory m LEFT JOIN embedding e ON e.memory_id = m.id WHERE e.memory_id IS NULL", [], |row| row.get(0))?;
        let missing_sparse_embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM memory m LEFT JOIN embedding_sparse e ON e.memory_id = m.id WHERE e.memory_id IS NULL", [], |row| row.get(0))?;
        let missing_colbert_embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM memory m LEFT JOIN embedding_colbert e ON e.memory_id = m.id WHERE e.memory_id IS NULL", [], |row| row.get(0))?;
        let pending_jobs: i64 =
            conn.query_row("SELECT COUNT(*) FROM embedding_job", [], |row| row.get(0))?;
        let graph = self.graph.load().stats();
        Ok(
            json!({"ok": true, "path": self.db_path(), "counts": counts, "missing_embeddings": missing_embeddings, "missing_sparse_embeddings": missing_sparse_embeddings, "missing_colbert_embeddings": missing_colbert_embeddings, "pending_jobs": pending_jobs, "graph": graph}),
        )
    }

    pub fn analyze_memories(&self) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let by_type = query_pairs(&conn, "SELECT type, COUNT(*) FROM memory GROUP BY type")?;
        let top_tags = query_tag_counts(&conn)?;
        Ok(json!({"by_type": by_type, "top_tags": top_tags, "graph": self.graph.load().stats()}))
    }

    pub fn relation_types(&self) -> Value {
        let authorable = [
            RelationKind::RelatesTo,
            RelationKind::LeadsTo,
            RelationKind::OccurredBefore,
            RelationKind::PrefersOver,
            RelationKind::Exemplifies,
            RelationKind::Contradicts,
            RelationKind::Reinforces,
            RelationKind::InvalidatedBy,
            RelationKind::EvolvedInto,
            RelationKind::DerivedFrom,
            RelationKind::PartOf,
        ];
        let system = [
            RelationKind::SimilarTo,
            RelationKind::PrecededBy,
            RelationKind::Discovered,
            RelationKind::ExtractedFrom,
        ];
        json!({"authorable": authorable.map(|k| k.as_str()), "system": system.map(|k| k.as_str())})
    }

    pub fn memory_types(&self) -> Value {
        let types = [
            MemoryType::Decision,
            MemoryType::Pattern,
            MemoryType::Preference,
            MemoryType::Style,
            MemoryType::Habit,
            MemoryType::Insight,
            MemoryType::Context,
            MemoryType::Statement,
        ];
        json!(types.map(|t| t.as_str()))
    }

    pub fn graph_stats(&self) -> Value {
        json!(self.graph.load().stats())
    }
    pub fn graph_neighbors(&self, id: &str) -> Value {
        json!({"memory_id": id, "neighbors": self.graph.load().neighbors(id)})
    }
    pub fn graph_snapshot(&self) -> Value {
        let graph = self.graph.load();
        let edges: Vec<_> = graph
            .node_to_id
            .iter()
            .take(1000)
            .map(|id| json!({"id": id, "neighbors": graph.neighbors(id)}))
            .collect();
        json!({"stats": graph.stats(), "nodes": edges})
    }

    pub fn get_related_memories(&self, args: RelatedArgs) -> anyhow::Result<Value> {
        let graph = self.graph.load();
        let seed = HashMap::from([(args.memory_id.clone(), 1.0_f32)]);
        let ppr = graph.forward_push(
            &seed,
            self.config.ppr.alpha,
            self.config.ppr.epsilon,
            self.config.ppr.max_pushes,
        );
        let conn = self.db.read_connection()?;
        let limit = args.limit.unwrap_or(10);
        let mut ranked: Vec<_> = ppr
            .into_iter()
            .filter(|(id, _)| id != &args.memory_id)
            .collect();
        ranked.sort_by(|(id_a, score_a), (id_b, score_b)| {
            score_b.total_cmp(score_a).then_with(|| id_a.cmp(id_b))
        });
        let mut rows = Vec::new();
        for (id, score) in ranked.into_iter().take(limit) {
            if let Some(memory) = fetch_memory_row(&conn, &id, true, now_epoch())? {
                rows.push(json!({"score": score, "memory": memory}));
            }
        }
        Ok(json!({"results": rows}))
    }

    pub fn trace_recall(&self, args: RecallArgs) -> anyhow::Result<Value> {
        let value = self.recall_memory(args)?;
        Ok(json!({"trace": value}))
    }

    pub fn enrichment_status(&self) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let pending: i64 =
            conn.query_row("SELECT COUNT(*) FROM enrichment_job", [], |row| row.get(0))?;
        let failures: i64 = conn.query_row(
            "SELECT COUNT(*) FROM enrichment_job WHERE last_error IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        let last_error: Option<String> = conn
            .query_row(
                "SELECT last_error FROM enrichment_job WHERE last_error IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(json!({
            "worker_running": false,
            "queue_depth": pending,
            "pending_ids_count": pending,
            "inflight_ids_count": 0,
            "successes": 0,
            "failures": failures,
            "last_error": last_error,
            "last_run_at": null,
        }))
    }

    pub fn enrichment_reprocess(&self, args: EnrichmentReprocessArgs) -> anyhow::Result<Value> {
        let mut writer = self.db.writer()?;
        let tx = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut queued = 0;
        let mut not_found = 0;
        let mut skipped = 0;
        let now = now_epoch();
        for id in args.ids {
            let exists: Option<i64> = tx
                .query_row("SELECT 1 FROM memory WHERE id = ?", [&id], |row| row.get(0))
                .optional()?;
            if exists.is_none() {
                not_found += 1;
                continue;
            }
            if !args.forced {
                let already: Option<i64> = tx
                    .query_row(
                        "SELECT 1 FROM enrichment_job WHERE memory_id = ?",
                        [&id],
                        |row| row.get(0),
                    )
                    .optional()?;
                if already.is_some() {
                    skipped += 1;
                    continue;
                }
            }
            tx.execute(
                "INSERT INTO enrichment_job(memory_id, reason, forced, attempts, next_retry_at, last_error, updated_at) VALUES (?, 'reprocess', ?, 0, ?, NULL, ?) ON CONFLICT(memory_id) DO UPDATE SET reason=excluded.reason, forced=excluded.forced, next_retry_at=excluded.next_retry_at, updated_at=excluded.updated_at",
                params![id, if args.forced { 1 } else { 0 }, now, now],
            )?;
            queued += 1;
        }
        tx.commit()?;
        Ok(json!({"queued": queued, "skipped": skipped, "not_found": not_found}))
    }

    pub fn consolidate_status(&self) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let last_recall: Option<i64> =
            conn.query_row("SELECT MAX(created_at) FROM recall_metric", [], |row| {
                row.get::<_, Option<i64>>(0)
            })?;
        Ok(json!({
            "scheduler_running": false,
            "last_decay_at": null,
            "last_forget_at": null,
            "last_creative_at": null,
            "last_cluster_at": null,
            "last_recall_metric_at": last_recall.and_then(|ts| epoch_to_rfc3339(ts).ok()),
        }))
    }
    pub fn repair_index(&self) -> anyhow::Result<()> {
        let conn = self.db.writer()?;
        conn.execute_batch("INSERT INTO memory_fts(memory_fts) VALUES('rebuild'); INSERT INTO statement_fts(statement_fts) VALUES('rebuild'); DELETE FROM tag_prefix;")?;
        let tags = {
            let mut tag_stmt = conn.prepare("SELECT memory_id, tag FROM tag")?;
            tag_stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?
        };
        for (memory_id, tag) in tags {
            for prefix in tag_prefixes(&tag) {
                conn.execute(
                    "INSERT OR IGNORE INTO tag_prefix(memory_id, prefix) VALUES (?, ?)",
                    params![memory_id, prefix],
                )?;
            }
        }
        conn.execute(
            "DELETE FROM entity WHERE id NOT IN (SELECT DISTINCT entity_id FROM memory_entity)",
            [],
        )?;
        rebuild_temporal_edges(&conn)?;
        let mut stmt =
            conn.prepare("SELECT src, dst, kind, strength, confidence FROM edge ORDER BY src")?;
        let edges = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    RelationKind::from_i64(row.get::<_, i64>(2)?)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(e.into()))?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, f32>(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);
        self.graph.publish(Graph::from_edges(edges));
        Ok(())
    }

    pub fn reembed(&self, args: ReembedArgs) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let ids = if let Some(id) = non_empty_string(args.memory_id) {
            vec![id]
        } else if !args.tags.is_empty() {
            self.ids_for_tags(&normalize_tags(&args.tags))?
        } else {
            conn.prepare("SELECT id FROM memory")?
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        drop(conn);
        let mut writer = self.db.writer()?;
        let tx = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = now_epoch();
        for id in &ids {
            let (content, summary): (String, Option<String>) = tx.query_row(
                "SELECT content, summary FROM memory WHERE id = ?",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            upsert_embedding(
                &tx,
                &*self.embedder,
                id,
                &embedding_text(&content, summary.as_deref()),
                now,
            )?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_model', ?)",
            [self.embedder.model()],
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_dims', ?)",
            [self.embedder.dims().to_string()],
        )?;
        tx.commit()?;
        Ok(json!({"reembedded": ids.len()}))
    }

    pub fn consolidate(&self, args: ConsolidateArgs) -> anyhow::Result<Value> {
        let mode = args.mode.unwrap_or_else(|| "all".to_string());
        let mut out = serde_json::Map::new();
        if mode == "all" || mode == "decay" {
            out.insert("decay".to_string(), self.decay(args.dry_run)?);
        }
        if mode == "all" || mode == "forget" {
            out.insert("forget".to_string(), self.forget(args.dry_run)?);
        }
        if mode == "all" || mode == "creative" {
            out.insert("creative".to_string(), self.creative(args.dry_run)?);
        }
        if mode == "all" || mode == "cluster" {
            out.insert("cluster".to_string(), self.cluster(args.dry_run)?);
        }
        Ok(Value::Object(out))
    }

    fn decay(&self, dry_run: bool) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let graph = self.graph.load();
        let now = now_epoch();
        let mut stmt = conn.prepare("SELECT id, importance, confidence, created_at, last_accessed FROM memory WHERE archived = 0")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        let mut updates = Vec::with_capacity(rows.len());
        for (id, importance, confidence, created_at, last_accessed) in rows {
            let age_days = ((now - created_at).max(0) as f64) / 86_400.0;
            let access_factor = last_accessed
                .map(|ts| 1.0 + (30.0 / (1.0 + ((now - ts).max(0) as f64 / 86_400.0))))
                .unwrap_or(1.0);
            let degree = graph
                .id_to_node
                .get(&id)
                .map(|n| graph.out_degree(*n))
                .unwrap_or(0) as f64;
            let relevance = (-self.config.decay.base * age_days).exp()
                * access_factor
                * degree.ln_1p().max(1.0)
                * (0.5 + importance)
                * (0.7 + 0.3 * confidence);
            let relevance = relevance
                .max(importance * self.config.decay.floor_factor)
                .min(1.0);
            updates.push((id, relevance));
        }
        if !dry_run {
            let mut writer = self.db.writer()?;
            let tx = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            for (id, relevance) in &updates {
                tx.execute(
                    "UPDATE memory SET relevance = ? WHERE id = ?",
                    params![relevance, id],
                )?;
            }
            tx.commit()?;
        }
        Ok(json!({"updated": updates.len(), "dry_run": dry_run}))
    }

    fn forget(&self, dry_run: bool) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let cutoff = now_epoch() - self.config.decay.grace_days * 86_400;
        let mut stmt = conn.prepare("SELECT id, relevance FROM memory WHERE protected = 0 AND importance < 0.8 AND COALESCE(last_accessed, created_at) < ?")?;
        let rows = stmt
            .query_map([cutoff], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        let archive: Vec<_> = rows
            .iter()
            .filter(|(_, relevance)| *relevance < self.config.decay.archive_threshold)
            .map(|(id, _)| id.clone())
            .collect();
        let delete: Vec<_> = rows
            .iter()
            .filter(|(_, relevance)| *relevance < self.config.decay.delete_threshold)
            .map(|(id, _)| id.clone())
            .collect();
        if !dry_run {
            let mut writer = self.db.writer()?;
            let tx = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            for id in &archive {
                tx.execute("UPDATE memory SET archived = 1 WHERE id = ?", [id])?;
            }
            for id in &delete {
                tx.execute("DELETE FROM memory WHERE id = ?", [id])?;
            }
            tx.commit()?;
            drop(writer);
            self.repair_index()?;
        }
        Ok(json!({"archived": archive.len(), "deleted": delete.len(), "dry_run": dry_run}))
    }

    fn creative(&self, dry_run: bool) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let mut stmt = conn.prepare("SELECT a.id, b.id FROM memory a JOIN memory b ON a.id < b.id WHERE a.archived=0 AND b.archived=0 AND a.relevance > 0.7 AND b.relevance > 0.7 LIMIT 50")?;
        let pairs = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        let mut created = 0;
        if !dry_run {
            let mut writer = self.db.writer()?;
            let tx = writer.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let now = now_epoch();
            for (a, b) in &pairs {
                created += tx.execute("INSERT OR IGNORE INTO edge(src, dst, kind, strength, confidence, metadata, created_at, updated_at) VALUES (?, ?, ?, 0.45, 0.5, '{}', ?, ?)", params![a, b, RelationKind::Discovered as i64, now, now])?;
            }
            tx.commit()?;
            drop(writer);
            self.repair_index()?;
        }
        Ok(json!({"candidate_pairs": pairs.len(), "created": created, "dry_run": dry_run}))
    }

    fn cluster(&self, dry_run: bool) -> anyhow::Result<Value> {
        let conn = self.db.read_connection()?;
        let similar: i64 = conn.query_row(
            "SELECT COUNT(*) FROM edge WHERE kind = ?",
            [RelationKind::SimilarTo as i64],
            |row| row.get(0),
        )?;
        Ok(json!({"similar_edges": similar, "meta_patterns_created": 0, "dry_run": dry_run}))
    }

    pub fn export_backup(&self, dest: &Path) -> anyhow::Result<Value> {
        self.db.backup_to(dest)?;
        let manifest = dest.with_extension("json");
        fs::write(
            &manifest,
            serde_json::to_vec_pretty(
                &json!({"format": "agskmem-sqlite-backup", "created_at": epoch_to_rfc3339(now_epoch())?, "source": self.db_path()}),
            )?,
        )?;
        Ok(json!({"backup": dest, "manifest": manifest}))
    }

    pub fn import_backup(&self, src: &Path) -> anyhow::Result<Value> {
        self.db.restore_from(src)?;
        self.repair_index()?;
        Ok(json!({"imported": src}))
    }

    fn touch(&self, ids: &[String]) -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.db.writer()?;
        let now = now_epoch();
        for id in ids {
            conn.execute(
                "UPDATE memory SET last_accessed = ? WHERE id = ?",
                params![now, id],
            )?;
        }
        Ok(())
    }

    fn record_metric(&self, tool: &str, dur_ms: i64, candidates: usize) -> anyhow::Result<()> {
        let conn = self.db.writer()?;
        conn.execute(
            "INSERT INTO recall_metric(tool, dur_ms, candidates, created_at) VALUES (?, ?, ?, ?)",
            params![tool, dur_ms, candidates as i64, now_epoch()],
        )?;
        conn.execute("DELETE FROM recall_metric WHERE id NOT IN (SELECT id FROM recall_metric ORDER BY id DESC LIMIT 1000)", [])?;
        Ok(())
    }
}

struct InsertMemory {
    id: String,
    content: String,
    summary: Option<String>,
    memory_type: MemoryType,
    tags: Vec<String>,
    importance: f64,
    confidence: f64,
    metadata: String,
    source: Option<String>,
    created_at: i64,
    t_valid: Option<i64>,
    t_invalid: Option<i64>,
}

fn insert_memory(
    tx: &Transaction<'_>,
    embedder: &dyn Embedder,
    item: InsertMemory,
) -> anyhow::Result<()> {
    validate_content(&item.content, usize::MAX)?;
    tx.execute("INSERT INTO memory(id, content, summary, type, importance, confidence, metadata, source, created_at, updated_at, t_valid, t_invalid) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params![item.id, item.content, item.summary, item.memory_type as i64, item.importance, item.confidence, item.metadata, item.source, item.created_at, item.created_at, item.t_valid, item.t_invalid])?;
    for tag in &item.tags {
        insert_tag(tx, &item.id, tag)?;
    }
    upsert_embedding(
        tx,
        embedder,
        &item.id,
        &embedding_text(&item.content, item.summary.as_deref()),
        item.created_at,
    )?;
    enrich_memory(tx, &item.id, &item.content, item.confidence)?;
    derive_temporal_edges(tx, &item.id, item.created_at)?;
    Ok(())
}

fn upsert_embedding(
    tx: &Transaction<'_>,
    embedder: &dyn Embedder,
    id: &str,
    content: &str,
    now: i64,
) -> anyhow::Result<()> {
    let batch = embedder.embed_for_store(&[content])?;
    persist_embedding_batch(tx, embedder, id, batch, now)?;
    tx.execute("DELETE FROM embedding_job WHERE memory_id = ?", [id])?;
    Ok(())
}

fn persist_embedding_batch(
    tx: &Transaction<'_>,
    embedder: &dyn Embedder,
    id: &str,
    mut batch: EmbeddingBatch,
    now: i64,
) -> anyhow::Result<()> {
    let vec = batch
        .dense
        .pop()
        .context("embedder returned no dense vector")?;
    if vec.len() != embedder.dims() {
        bail!(
            "embedder returned {} dims, expected {}",
            vec.len(),
            embedder.dims()
        );
    }
    let sparse = batch
        .sparse
        .pop()
        .context("embedder returned no sparse vector")?;
    if sparse.indices.len() != sparse.values.len() {
        bail!(
            "embedder returned sparse vector with {} indices and {} values",
            sparse.indices.len(),
            sparse.values.len()
        );
    }
    let colbert = batch
        .colbert
        .pop()
        .context("embedder returned no ColBERT vectors")?;
    let blob = embed::vector_to_blob(&vec);
    tx.execute("INSERT INTO embedding(memory_id, model, dims, norm, vec, created_at) VALUES (?, ?, ?, 1.0, ?, ?) ON CONFLICT(memory_id) DO UPDATE SET model=excluded.model, dims=excluded.dims, norm=excluded.norm, vec=excluded.vec, created_at=excluded.created_at", params![id, embedder.model(), embedder.dims() as i64, blob, now])?;
    tx.execute("DELETE FROM embedding_sparse WHERE memory_id = ?", [id])?;
    for (token_id, weight) in sparse.indices.iter().zip(sparse.values.iter()) {
        let token_id = i64::try_from(*token_id).context("sparse token id exceeds sqlite i64")?;
        tx.execute(
            "INSERT INTO embedding_sparse(memory_id, token_id, weight) VALUES (?, ?, ?)",
            params![id, token_id, *weight as f64],
        )?;
    }
    tx.execute("DELETE FROM embedding_colbert WHERE memory_id = ?", [id])?;
    for (token_index, vector) in colbert.iter().enumerate() {
        if vector.len() != embedder.dims() {
            bail!(
                "embedder returned ColBERT vector {} with {} dims, expected {}",
                token_index,
                vector.len(),
                embedder.dims()
            );
        }
        let token_index =
            i64::try_from(token_index).context("ColBERT token index exceeds sqlite i64")?;
        tx.execute(
            "INSERT INTO embedding_colbert(memory_id, token_index, vec) VALUES (?, ?, ?)",
            params![id, token_index, embed::vector_to_blob(vector)],
        )?;
    }
    Ok(())
}
fn enrich_memory(
    tx: &Transaction<'_>,
    id: &str,
    content: &str,
    confidence: f64,
) -> anyhow::Result<()> {
    let mut entity_tags = Vec::new();
    for (kind, slug, label) in extract_entities(content) {
        tx.execute(
            "INSERT OR IGNORE INTO entity(kind, slug, label, quality) VALUES (?, ?, ?, 1.0)",
            params![kind, slug, label],
        )?;
        let entity_id: i64 = tx.query_row(
            "SELECT id FROM entity WHERE kind = ? AND slug = ?",
            params![kind, slug],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO memory_entity(memory_id, entity_id, role) VALUES (?, ?, 2)",
            params![id, entity_id],
        )?;
        entity_tags.push(format!("entity:{}:{}", entity_kind_tag(kind), slug));
    }
    for tag in entity_tags {
        insert_tag(tx, id, &tag)?;
    }
    for statement in extract_statements(content) {
        let sid = Uuid::now_v7().to_string();
        tx.execute("INSERT INTO statement(id, memory_id, content, confidence, reliability, created_at) VALUES (?, ?, ?, ?, 0.7, ?)", params![sid, id, statement, confidence, now_epoch()])?;
    }
    Ok(())
}

fn derive_temporal_edges(tx: &Transaction<'_>, id: &str, created_at: i64) -> anyhow::Result<usize> {
    let peers = {
        let mut stmt = tx.prepare(
            "SELECT DISTINCT other.memory_id, m.created_at
             FROM memory_entity this
             JOIN memory_entity other ON other.entity_id = this.entity_id AND other.memory_id <> this.memory_id
             JOIN memory m ON m.id = other.memory_id
             WHERE this.memory_id = ? AND ABS(m.created_at - ?) <= ?",
        )?;
        stmt.query_map(
            params![id, created_at, TEMPORAL_ENTITY_WINDOW_SECONDS],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )?
        .collect::<Result<Vec<_>, _>>()?
    };
    let mut created = 0;
    let now = now_epoch();
    for (peer_id, peer_created_at) in peers {
        let (src, dst) = if created_at > peer_created_at
            || (created_at == peer_created_at && id > peer_id.as_str())
        {
            (id, peer_id.as_str())
        } else {
            (peer_id.as_str(), id)
        };
        created += tx.execute(
            "INSERT OR IGNORE INTO edge(src, dst, kind, strength, confidence, metadata, created_at, updated_at) VALUES (?, ?, ?, 0.35, 0.8, ?, ?, ?)",
            params![src, dst, RelationKind::PrecededBy as i64, TEMPORAL_EDGE_METADATA, now, now],
        )?;
    }
    Ok(created)
}

fn rebuild_temporal_edges(conn: &Connection) -> anyhow::Result<usize> {
    conn.execute(
        "DELETE FROM edge WHERE kind = ?",
        [RelationKind::PrecededBy as i64],
    )?;
    let pairs = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT newer.memory_id, older.memory_id
             FROM memory_entity newer
             JOIN memory mn ON mn.id = newer.memory_id
             JOIN memory_entity older ON older.entity_id = newer.entity_id AND older.memory_id <> newer.memory_id
             JOIN memory mo ON mo.id = older.memory_id
             WHERE ABS(mn.created_at - mo.created_at) <= ?
               AND (mn.created_at > mo.created_at OR (mn.created_at = mo.created_at AND newer.memory_id > older.memory_id))",
        )?;
        stmt.query_map([TEMPORAL_ENTITY_WINDOW_SECONDS], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?
    };
    let now = now_epoch();
    let mut created = 0;
    for (src, dst) in pairs {
        created += conn.execute(
            "INSERT OR IGNORE INTO edge(src, dst, kind, strength, confidence, metadata, created_at, updated_at) VALUES (?, ?, ?, 0.35, 0.8, ?, ?, ?)",
            params![src, dst, RelationKind::PrecededBy as i64, TEMPORAL_EDGE_METADATA, now, now],
        )?;
    }
    Ok(created)
}

fn extract_entities(content: &str) -> Vec<(i64, String, String)> {
    let words: Vec<&str> = content.unicode_words().collect();
    let mut out = Vec::new();
    for pair in words.windows(2) {
        if is_capitalized(pair[0]) && is_capitalized(pair[1]) {
            let label = format!(
                "{} {}",
                pair[0].trim_matches('\''),
                pair[1].trim_matches('\'')
            );
            let slug = label.to_ascii_lowercase().replace(' ', "-");
            out.push((0, slug, label));
        }
    }
    for word in words {
        if word.contains('/') || word.contains('.') && word.len() > 3 {
            out.push((6, word.to_ascii_lowercase(), word.to_string()));
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    out
}

fn is_capitalized(word: &str) -> bool {
    let mut chars = word.chars();
    matches!(chars.next(), Some(c) if c.is_uppercase()) && chars.any(|c| c.is_lowercase())
}

fn extract_statements(content: &str) -> Vec<String> {
    content
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.len() >= 8)
        .take(8)
        .map(ToOwned::to_owned)
        .collect()
}

fn assert_memory_exists(tx: &Transaction<'_>, id: &str) -> anyhow::Result<()> {
    let exists: Option<i64> = tx
        .query_row("SELECT 1 FROM memory WHERE id = ?", [id], |row| row.get(0))
        .optional()?;
    if exists.is_none() {
        bail!("memory {id} not found");
    }
    Ok(())
}

fn fetch_memory_row(
    conn: &Connection,
    id: &str,
    current_only: bool,
    as_of: i64,
) -> anyhow::Result<Option<MemoryRow>> {
    if current_only && !is_active(conn, id, as_of)? {
        return Ok(None);
    }
    let mut stmt = conn.prepare("SELECT id, content, summary, type, importance, confidence, relevance, reliability, metadata, source, created_at, updated_at, last_accessed, t_valid, t_invalid, archived, protected FROM memory WHERE id = ?")?;
    let row = stmt
        .query_row([id], |row| {
            let metadata_text: String = row.get(8)?;
            let type_code: i64 = row.get(3)?;
            let created_at: i64 = row.get(10)?;
            let updated_at: i64 = row.get(11)?;
            let last_accessed: Option<i64> = row.get(12)?;
            let t_valid: Option<i64> = row.get(13)?;
            let t_invalid: Option<i64> = row.get(14)?;
            let archived: i64 = row.get(15)?;
            let protected: i64 = row.get(16)?;
            let metadata =
                serde_json::from_str(&metadata_text).unwrap_or(Value::Object(Default::default()));
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                type_code,
                row.get::<_, f64>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, f64>(7)?,
                metadata,
                row.get::<_, Option<String>>(9)?,
                created_at,
                updated_at,
                last_accessed,
                t_valid,
                t_invalid,
                archived != 0,
                protected != 0,
            ))
        })
        .optional()?;
    let Some((
        id,
        content,
        summary,
        type_code,
        importance,
        confidence,
        relevance,
        reliability,
        metadata,
        source,
        created_at,
        updated_at,
        last_accessed,
        t_valid,
        t_invalid,
        archived,
        protected,
    )) = row
    else {
        return Ok(None);
    };
    let tags = tags_for(conn, &id)?;
    let state = state_for(conn, &id, archived, t_valid, t_invalid, as_of)?;
    Ok(Some(MemoryRow {
        id,
        content,
        summary,
        memory_type: MemoryType::from_i64(type_code)?.as_str().to_string(),
        importance,
        confidence,
        relevance,
        reliability,
        metadata,
        source,
        tags,
        created_at: epoch_to_rfc3339(created_at)?,
        updated_at: epoch_to_rfc3339(updated_at)?,
        last_accessed: opt_epoch_to_rfc3339(last_accessed)?,
        t_valid: opt_epoch_to_rfc3339(t_valid)?,
        t_invalid: opt_epoch_to_rfc3339(t_invalid)?,
        archived,
        protected,
        state,
    }))
}

fn tags_for(conn: &Connection, id: &str) -> anyhow::Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT tag FROM tag WHERE memory_id = ? ORDER BY tag")?;
    stmt.query_map([id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn state_for(
    conn: &Connection,
    id: &str,
    archived: bool,
    t_valid: Option<i64>,
    t_invalid: Option<i64>,
    as_of: i64,
) -> anyhow::Result<String> {
    if archived {
        return Ok("archived".to_string());
    }
    if t_valid.is_some_and(|t| t > as_of) {
        return Ok("future".to_string());
    }
    if t_invalid.is_some_and(|t| t <= as_of) {
        return Ok("expired".to_string());
    }
    if !active_replacements(conn, id, as_of)?.is_empty() {
        return Ok("superseded".to_string());
    }
    Ok("active".to_string())
}

fn is_active(conn: &Connection, id: &str, as_of: i64) -> anyhow::Result<bool> {
    let row: Option<(i64, Option<i64>, Option<i64>)> = conn
        .query_row(
            "SELECT archived, t_valid, t_invalid FROM memory WHERE id = ?",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((archived, t_valid, t_invalid)) = row else {
        return Ok(false);
    };
    Ok(archived == 0
        && t_valid.is_none_or(|t| t <= as_of)
        && t_invalid.is_none_or(|t| t > as_of)
        && active_replacements(conn, id, as_of)?.is_empty())
}

fn active_replacements(conn: &Connection, id: &str, as_of: i64) -> anyhow::Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT dst FROM edge WHERE src = ? AND kind IN (?, ?)")?;
    let candidates = stmt
        .query_map(
            params![
                id,
                RelationKind::InvalidatedBy as i64,
                RelationKind::EvolvedInto as i64
            ],
            |row| row.get::<_, String>(0),
        )?
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Vec::new();
    for candidate in candidates {
        if candidate != id && is_state_active_without_supersession(conn, &candidate, as_of)? {
            out.push(candidate);
        }
    }
    Ok(out)
}

fn is_state_active_without_supersession(
    conn: &Connection,
    id: &str,
    as_of: i64,
) -> anyhow::Result<bool> {
    let row: Option<(i64, Option<i64>, Option<i64>)> = conn
        .query_row(
            "SELECT archived, t_valid, t_invalid FROM memory WHERE id = ?",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    Ok(row.is_some_and(|(archived, t_valid, t_invalid)| {
        archived == 0 && t_valid.is_none_or(|t| t <= as_of) && t_invalid.is_none_or(|t| t > as_of)
    }))
}

struct FilterSpec<'a> {
    tags: &'a [String],
    exclude_tags: &'a [String],
    current_only: bool,
    as_of: i64,
    start: Option<&'a str>,
    end: Option<&'a str>,
    time_range: Option<&'a TimeRange>,
}

struct QueryEmbedding<'a> {
    dense: &'a [f32],
    sparse: &'a SparseVector,
    colbert: &'a [Vec<f32>],
}

fn passes_filters(conn: &Connection, id: &str, filters: &FilterSpec<'_>) -> anyhow::Result<bool> {
    if filters.current_only && !is_active(conn, id, filters.as_of)? {
        return Ok(false);
    }
    let row_tags = tags_for(conn, id)?;
    if !filters
        .tags
        .iter()
        .all(|tag| row_tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
    {
        return Ok(false);
    }
    if filters
        .exclude_tags
        .iter()
        .any(|tag| row_tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
    {
        return Ok(false);
    }
    let mut range_start = parse_time(filters.start)?;
    let mut range_end = parse_time(filters.end)?;
    if let Some(range) = filters.time_range {
        range_start = range_start.or(parse_time(range.start.as_deref())?);
        range_end = range_end.or(parse_time(range.end.as_deref())?);
    }
    if range_start.is_some() || range_end.is_some() {
        let created: i64 =
            conn.query_row("SELECT created_at FROM memory WHERE id = ?", [id], |row| {
                row.get(0)
            })?;
        if range_start.is_some_and(|s| created < s) || range_end.is_some_and(|e| created > e) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn vector_candidates(
    conn: &Connection,
    query_vec: &[f32],
    model: &str,
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    if query_vec.is_empty() || query_vec.iter().all(|v| *v == 0.0) || limit == 0 {
        return Ok(Vec::new());
    }
    let query_blob = embed::vector_to_blob(query_vec);
    let mut stmt = conn.prepare(
        "SELECT memory_id FROM (
            SELECT memory_id, cosine(?1, vec) AS score
            FROM embedding
            WHERE model = ?2 AND dims = ?3
        )
        WHERE score > 0.0
        ORDER BY score DESC, memory_id
        LIMIT ?4",
    )?;
    stmt.query_map(
        params![query_blob, model, query_vec.len() as i64, limit as i64],
        |row| row.get::<_, String>(0),
    )?
    .collect::<Result<Vec<_>, _>>()
    .map_err(Into::into)
}

fn sparse_candidates(
    conn: &Connection,
    query_sparse: &SparseVector,
    limit: usize,
) -> anyhow::Result<HashMap<String, f64>> {
    if query_sparse.indices.is_empty() || limit == 0 {
        return Ok(HashMap::new());
    }
    let mut scores = HashMap::<String, f64>::new();
    let mut stmt =
        conn.prepare("SELECT memory_id, weight FROM embedding_sparse WHERE token_id = ?")?;
    for (token_id, query_weight) in query_sparse.indices.iter().zip(query_sparse.values.iter()) {
        let token_id =
            i64::try_from(*token_id).context("sparse query token id exceeds sqlite i64")?;
        let rows = stmt.query_map([token_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        for row in rows {
            let (id, doc_weight) = row?;
            *scores.entry(id).or_default() += f64::from(*query_weight) * doc_weight;
        }
    }
    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|(id_a, score_a), (id_b, score_b)| {
        score_b.total_cmp(score_a).then_with(|| id_a.cmp(id_b))
    });
    ranked.truncate(limit);
    Ok(ranked
        .into_iter()
        .map(|(id, score)| (id, score.clamp(0.0, 1.0)))
        .collect())
}

fn fts_candidates(conn: &Connection, query: &str, limit: usize) -> anyhow::Result<Vec<String>> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare("SELECT memory.id FROM memory_fts JOIN memory ON memory.rowid = memory_fts.rowid WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts) LIMIT ?")?;
    stmt.query_map(params![fts, limit as i64], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn exact_phrase_candidates(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    let phrase = query.trim();
    if phrase.len() < 3 {
        return Ok(Vec::new());
    }
    let like = format!("%{}%", phrase.replace('%', "\\%").replace('_', "\\_"));
    let mut stmt = conn.prepare("SELECT id FROM memory WHERE content LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' ORDER BY updated_at DESC LIMIT ?")?;
    stmt.query_map(params![like, like, limit as i64], |row| {
        row.get::<_, String>(0)
    })?
    .collect::<Result<Vec<_>, _>>()
    .map_err(Into::into)
}

fn entity_candidates(conn: &Connection, query: &str, limit: usize) -> anyhow::Result<Vec<String>> {
    let entities = extract_entities(query);
    if entities.is_empty() {
        return Ok(Vec::new());
    }
    let slugs: Vec<String> = entities.into_iter().map(|(_, slug, _)| slug).collect();
    let mut out = Vec::new();
    let mut stmt = conn.prepare("SELECT DISTINCT memory_id FROM memory_entity me JOIN entity e ON e.id = me.entity_id WHERE e.slug = ? LIMIT ?")?;
    for slug in slugs {
        for id in stmt.query_map(params![slug, limit as i64], |row| row.get::<_, String>(0))? {
            out.push(id?);
        }
    }
    out.sort();
    out.dedup();
    out.truncate(limit);
    Ok(out)
}

fn fallback_candidates(conn: &Connection, limit: usize) -> anyhow::Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT id FROM memory WHERE archived = 0 ORDER BY relevance DESC, created_at DESC LIMIT ?",
    )?;
    stmt.query_map([limit as i64], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn compute_components(
    conn: &Connection,
    memory: &MemoryRow,
    query: &str,
    query_embedding: &QueryEmbedding<'_>,
    sparse_candidate_score: Option<f64>,
    context_tags: &[String],
) -> anyhow::Result<ScoreComponents> {
    let keyword = lexical_score(query, &memory.content);
    let exact_phrase = if !query.trim().is_empty()
        && memory
            .content
            .to_ascii_lowercase()
            .contains(&query.to_ascii_lowercase())
    {
        1.0
    } else {
        0.0
    };
    let vector = embedding_for(conn, &memory.id)?
        .map(|v| f64::from(embed::cosine(query_embedding.dense, &v)).max(0.0))
        .unwrap_or(0.0);
    let sparse = match sparse_candidate_score {
        Some(score) => score,
        None => sparse_score_for(conn, &memory.id, query_embedding.sparse)?,
    };
    let colbert = colbert_for(conn, &memory.id)?
        .map(|doc| colbert_score(query_embedding.colbert, &doc))
        .unwrap_or(0.0);
    let tag_overlap = if context_tags.is_empty() {
        0.0
    } else {
        context_tags
            .iter()
            .filter(|tag| memory.tags.iter().any(|t| t == *tag))
            .count() as f64
            / context_tags.len() as f64
    };
    let context_bonus = (tag_overlap * 0.10).min(0.10);
    Ok(ScoreComponents {
        vector,
        sparse,
        colbert,
        keyword,
        ppr: 0.0,
        tag_overlap,
        exact_phrase,
        importance: memory.importance,
        recency: recency_score(&memory.updated_at),
        confidence: memory.confidence,
        reliability: memory.reliability,
        context_bonus,
    })
}

fn embedding_for(conn: &Connection, id: &str) -> anyhow::Result<Option<Vec<f32>>> {
    let blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vec FROM embedding WHERE memory_id = ?",
            [id],
            |row| row.get(0),
        )
        .optional()?;
    blob.map(|b| embed::blob_to_vector(&b)).transpose()
}

fn sparse_score_for(
    conn: &Connection,
    id: &str,
    query_sparse: &SparseVector,
) -> anyhow::Result<f64> {
    if query_sparse.indices.is_empty() {
        return Ok(0.0);
    }
    let mut score = 0.0;
    let mut stmt =
        conn.prepare("SELECT weight FROM embedding_sparse WHERE memory_id = ? AND token_id = ?")?;
    for (token_id, query_weight) in query_sparse.indices.iter().zip(query_sparse.values.iter()) {
        let token_id =
            i64::try_from(*token_id).context("sparse query token id exceeds sqlite i64")?;
        let doc_weight: Option<f64> = stmt
            .query_row(params![id, token_id], |row| row.get(0))
            .optional()?;
        if let Some(doc_weight) = doc_weight {
            score += f64::from(*query_weight) * doc_weight;
        }
    }
    Ok(score.clamp(0.0, 1.0))
}

fn colbert_for(conn: &Connection, id: &str) -> anyhow::Result<Option<Vec<Vec<f32>>>> {
    let mut stmt =
        conn.prepare("SELECT vec FROM embedding_colbert WHERE memory_id = ? ORDER BY token_index")?;
    let rows = stmt
        .query_map([id], |row| row.get::<_, Vec<u8>>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Ok(None);
    }
    rows.into_iter()
        .map(|blob| embed::blob_to_vector(&blob))
        .collect::<anyhow::Result<Vec<_>>>()
        .map(Some)
}

fn colbert_score(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f64 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for query_vector in query {
        let best = doc
            .iter()
            .map(|doc_vector| embed::cosine(query_vector, doc_vector))
            .fold(0.0_f32, f32::max);
        total += best;
    }
    f64::from(total / query.len() as f32).clamp(0.0, 1.0)
}

fn lexical_score(query: &str, content: &str) -> f64 {
    let q: Vec<String> = query
        .unicode_words()
        .map(|w| w.to_ascii_lowercase())
        .collect();
    if q.is_empty() {
        return 0.0;
    }
    let content_lc = content.to_ascii_lowercase();
    let hits = q.iter().filter(|w| content_lc.contains(w.as_str())).count();
    hits as f64 / q.len() as f64
}

fn recency_score(updated_at: &str) -> f64 {
    let Ok(Some(epoch)) = parse_time(Some(updated_at)) else {
        return 0.0;
    };
    let days = (now_epoch() - epoch).max(0) as f64 / 86_400.0;
    (1.0 / (1.0 + days / 30.0)).clamp(0.0, 1.0)
}

fn score_components(weights: &crate::config::RecallWeights, c: &ScoreComponents) -> f64 {
    weights.vector * c.vector
        + weights.sparse * c.sparse
        + weights.colbert * c.colbert
        + weights.keyword * c.keyword
        + weights.ppr * c.ppr
        + weights.tag_overlap * c.tag_overlap
        + weights.exact_phrase * c.exact_phrase
        + weights.importance * c.importance
        + weights.recency * c.recency
        + weights.confidence * c.confidence
        + weights.reliability * c.reliability
        + c.context_bonus
}

fn apply_score_floor(hits: &mut Vec<RecallHit>, args: &RecallArgs, default_adaptive: bool) {
    if !score_sort(args) {
        return;
    }
    if let Some(floor) = args.min_score {
        hits.retain(|hit| hit.score >= floor);
    }
    if !args.adaptive_floor.unwrap_or(default_adaptive) || hits.len() < 4 {
        return;
    }
    let Some(max_score) = hits.first().map(|hit| hit.score) else {
        return;
    };
    if max_score <= 0.0 {
        return;
    }
    let floor = max_score * 0.25;
    hits.retain(|hit| {
        hit.score >= floor || args.priority_ids.iter().any(|id| id == &hit.memory.id)
    });
}

fn sort_hits(hits: &mut [RecallHit], args: &RecallArgs, query: &str) {
    match recall_sort(args, query) {
        RecallSort::Score => hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.memory.updated_at.cmp(&a.memory.updated_at))
        }),
        RecallSort::TimeDesc => hits.sort_by(|a, b| b.memory.created_at.cmp(&a.memory.created_at)),
        RecallSort::TimeAsc => hits.sort_by(|a, b| a.memory.created_at.cmp(&b.memory.created_at)),
        RecallSort::UpdatedDesc => {
            hits.sort_by(|a, b| b.memory.updated_at.cmp(&a.memory.updated_at))
        }
        RecallSort::UpdatedAsc => {
            hits.sort_by(|a, b| a.memory.updated_at.cmp(&b.memory.updated_at))
        }
    }
}

fn score_sort(args: &RecallArgs) -> bool {
    matches!(recall_sort(args, ""), RecallSort::Score)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecallSort {
    Score,
    TimeDesc,
    TimeAsc,
    UpdatedDesc,
    UpdatedAsc,
}

fn recall_sort(args: &RecallArgs, query: &str) -> RecallSort {
    let value = args.sort.as_deref().or(args.order_by.as_deref());
    match value {
        Some("time_desc") => RecallSort::TimeDesc,
        Some("time_asc") => RecallSort::TimeAsc,
        Some("updated_desc") => RecallSort::UpdatedDesc,
        Some("updated_asc") => RecallSort::UpdatedAsc,
        Some("score") => RecallSort::Score,
        _ if query.trim().is_empty()
            && (args.start.is_some() || args.end.is_some() || args.time_query.is_some()) =>
        {
            RecallSort::TimeDesc
        }
        _ => RecallSort::Score,
    }
}

fn enrich_context_tags(args: &RecallArgs, out: &mut Vec<String>) {
    if let Some(language) = args
        .language
        .as_deref()
        .or_else(|| args.active_path.as_deref().and_then(language_from_path))
    {
        out.push(format!("language:{language}"));
    }
    if let Some(context) = args.context.as_deref() {
        for word in context.unicode_words().take(4) {
            out.push(word.to_ascii_lowercase());
        }
    }
    for ty in &args.context_types {
        out.push(format!("type:{}", ty.to_ascii_lowercase()));
    }
    out.sort_unstable();
    out.dedup();
}

fn language_from_path(path: &str) -> Option<&'static str> {
    let ext = Path::new(path).extension()?.to_str()?;
    match ext {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" => Some("javascript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cc" | "cpp" | "hpp" => Some("cpp"),
        _ => None,
    }
}

fn decompose_query(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for (_, slug, label) in extract_entities(query) {
        terms.push(label);
        terms.push(slug.replace('-', " "));
    }
    for word in query.unicode_words() {
        let lower = word.to_ascii_lowercase();
        if lower.len() >= 4 && !is_stop_word(&lower) {
            terms.push(lower);
        }
    }
    terms.sort_unstable();
    terms.dedup();
    terms.truncate(8);
    terms
}

fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "about"
            | "from"
            | "with"
            | "that"
            | "this"
            | "what"
            | "when"
            | "where"
            | "which"
            | "have"
            | "will"
            | "should"
    )
}

fn time_query_bounds(query: Option<&str>) -> anyhow::Result<(Option<String>, Option<String>)> {
    let Some(query) = query.map(str::trim).filter(|q| !q.is_empty()) else {
        return Ok((None, None));
    };
    let now = OffsetDateTime::now_utc();
    let day = 86_400;
    let seconds_since_midnight = i64::from(now.time().hour()) * 3600
        + i64::from(now.time().minute()) * 60
        + i64::from(now.time().second());
    let (start, end) = match query.to_ascii_lowercase().as_str() {
        "today" => (
            now.unix_timestamp() - seconds_since_midnight,
            now.unix_timestamp(),
        ),
        "yesterday" => {
            let end = now.unix_timestamp() - seconds_since_midnight;
            (end - day, end)
        }
        "last week" => (now.unix_timestamp() - 7 * day, now.unix_timestamp()),
        "last month" => (now.unix_timestamp() - 30 * day, now.unix_timestamp()),
        "last year" => (now.unix_timestamp() - 365 * day, now.unix_timestamp()),
        other => {
            if let Some(days) = other
                .strip_prefix("last ")
                .and_then(|s| s.strip_suffix(" days"))
                .and_then(|n| n.parse::<i64>().ok())
            {
                (
                    now.unix_timestamp() - days.max(0) * day,
                    now.unix_timestamp(),
                )
            } else if let Some(start) = parse_time(Some(query))? {
                (start, now.unix_timestamp())
            } else {
                bail!("unsupported time_query {query}");
            }
        }
    };
    Ok((Some(start.to_string()), Some(end.to_string())))
}
fn apply_mmr(hits: &mut Vec<RecallHit>, limit: usize, lambda: f64) {
    let mut remaining = std::mem::take(hits);
    let mut selected: Vec<RecallHit> = Vec::new();
    while !remaining.is_empty() && selected.len() < limit {
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, hit) in remaining.iter().enumerate() {
            let overlap = selected
                .iter()
                .map(|s| lexical_score(&s.memory.content, &hit.memory.content))
                .fold(0.0_f64, f64::max);
            let mmr = lambda * hit.score - (1.0 - lambda) * overlap;
            if mmr > best_score {
                best = i;
                best_score = mmr;
            }
        }
        selected.push(remaining.swap_remove(best));
    }
    selected.extend(remaining);
    *hits = selected;
}

fn fts_query(query: &str) -> String {
    query
        .unicode_words()
        .take(16)
        .map(|w| format!("\"{}\"", w.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn non_empty_string(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

#[derive(Debug, Clone, Copy)]
struct Classification {
    memory_type: MemoryType,
    confidence: f64,
}

fn metadata_map(value: Option<Value>) -> anyhow::Result<Map<String, Value>> {
    match value.unwrap_or(Value::Object(Map::new())) {
        Value::Object(map) => Ok(map),
        _ => bail!("metadata must be a JSON object"),
    }
}

fn merge_metadata(dst: &mut Map<String, Value>, src: Map<String, Value>) {
    for (key, value) in src {
        dst.insert(key, value);
    }
}

fn govern_content(
    content: String,
    config: &Config,
    metadata: &mut Map<String, Value>,
) -> anyhow::Result<String> {
    validate_content(&content, config.content.hard_limit_bytes)?;
    if content.len() <= config.content.soft_limit_bytes {
        return Ok(content);
    }
    if config.content.auto_summarize && config.classification.provider == "local" {
        let summary = deterministic_summary(&content, config.content.summary_target_chars);
        metadata.insert("original_content".to_string(), Value::String(content));
        metadata.insert("was_summarized".to_string(), Value::Bool(true));
        metadata.insert(
            "original_length".to_string(),
            Value::from(
                metadata
                    .get("original_content")
                    .and_then(Value::as_str)
                    .map(str::len)
                    .unwrap_or(0),
            ),
        );
        metadata.insert("summarized_length".to_string(), Value::from(summary.len()));
        metadata.insert(
            "summary_model".to_string(),
            Value::String("local-deterministic-v1".to_string()),
        );
        metadata.insert("summary_created_at".to_string(), Value::from(now_epoch()));
        return Ok(summary);
    }
    metadata.insert("was_summarized".to_string(), Value::Bool(false));
    metadata.insert(
        "summarization_skipped".to_string(),
        Value::String(if config.content.auto_summarize {
            "provider unavailable".to_string()
        } else {
            "disabled".to_string()
        }),
    );
    metadata.insert("original_length".to_string(), Value::from(content.len()));
    Ok(content)
}

fn deterministic_summary(content: &str, target_chars: usize) -> String {
    let target = target_chars.max(1);
    let mut out = String::new();
    for sentence in content.split_inclusive(['.', '!', '?']) {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }
        let add_len = trimmed.len() + usize::from(!out.is_empty());
        if !out.is_empty() && out.len() + add_len > target {
            break;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(trimmed);
        if out.len() >= target {
            break;
        }
    }
    if out.is_empty() {
        for (idx, ch) in content.char_indices() {
            if idx >= target {
                break;
            }
            out.push(ch);
        }
    }
    if out.len() > target {
        truncate_to_char_boundary(&mut out, target);
    }
    out
}

fn truncate_to_char_boundary(value: &mut String, max_len: usize) {
    if value.len() <= max_len {
        return;
    }
    let mut end = max_len;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
}

fn classify_memory(
    tx: &Transaction<'_>,
    content: &str,
    explicit_type: Option<&str>,
    explicit_confidence: Option<f64>,
    metadata: &mut Map<String, Value>,
) -> anyhow::Result<Classification> {
    if let Some(value) = explicit_type {
        let memory_type = MemoryType::parse(Some(value))?;
        let confidence = clamp_unit(explicit_confidence, 0.9, "confidence")?;
        metadata.insert(
            "classification".to_string(),
            json!({"method": "explicit", "confidence": confidence, "reasons": ["caller supplied type"]}),
        );
        return Ok(Classification {
            memory_type,
            confidence,
        });
    }

    let hash = content_hash(content);
    if let Some((code, confidence, meta)) = tx
        .query_row(
            "SELECT type, confidence, metadata FROM classification_cache WHERE content_hash = ?",
            [&hash],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?
    {
        if let Ok(Value::Object(class_meta)) = serde_json::from_str::<Value>(&meta) {
            metadata.insert("classification".to_string(), Value::Object(class_meta));
        }
        return Ok(Classification {
            memory_type: MemoryType::from_i64(code)?,
            confidence: clamp_unit(
                Some(explicit_confidence.unwrap_or(confidence)),
                0.9,
                "confidence",
            )?,
        });
    }

    let (memory_type, reason) = deterministic_classification(content);
    let confidence = clamp_unit(explicit_confidence, 0.82, "confidence")?;
    let meta = json!({
        "method": "local-deterministic-v1",
        "model": "rules",
        "confidence": confidence,
        "reasons": [reason],
    });
    metadata.insert("classification".to_string(), meta.clone());
    tx.execute(
        "INSERT OR REPLACE INTO classification_cache(content_hash, type, confidence, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        params![hash, memory_type as i64, confidence, serde_json::to_string(&meta)?, now_epoch()],
    )?;
    Ok(Classification {
        memory_type,
        confidence,
    })
}

fn deterministic_classification(content: &str) -> (MemoryType, &'static str) {
    let lower = content.to_ascii_lowercase();
    if lower.contains("decision:") || lower.contains("decided ") || lower.contains(" chose ") {
        (MemoryType::Decision, "decision marker")
    } else if lower.starts_with("pattern:") || lower.contains(" pattern ") {
        (MemoryType::Pattern, "pattern marker")
    } else if lower.contains("prefer")
        || lower.contains("preference")
        || lower.contains("do not ")
        || lower.contains("don't ")
        || lower.contains("correction")
    {
        (MemoryType::Preference, "preference/correction marker")
    } else if lower.contains("style guide")
        || lower.contains("coding style")
        || lower.contains("formatting")
    {
        (MemoryType::Style, "style vocabulary")
    } else if lower.contains("habit") || lower.contains("usually ") || lower.contains("always ") {
        (MemoryType::Habit, "habit phrase")
    } else if lower.contains("insight")
        || lower.contains("learned ")
        || lower.contains("root cause")
    {
        (MemoryType::Insight, "insight marker")
    } else {
        (MemoryType::Context, "default context")
    }
}

fn classification_locked(metadata: &Map<String, Value>) -> bool {
    metadata
        .get("classification")
        .and_then(Value::as_object)
        .and_then(|m| m.get("locked"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

fn embedding_text<'a>(content: &'a str, summary: Option<&'a str>) -> String {
    match summary {
        Some(summary) if !summary.trim().is_empty() => {
            let mut text = String::with_capacity(content.len() + summary.len() + 1);
            text.push_str(content);
            text.push(' ');
            text.push_str(summary);
            text
        }
        _ => content.to_string(),
    }
}

fn insert_tag(tx: &Transaction<'_>, memory_id: &str, tag: &str) -> anyhow::Result<()> {
    tx.execute(
        "INSERT OR IGNORE INTO tag(memory_id, tag) VALUES (?, ?)",
        params![memory_id, tag],
    )?;
    for prefix in tag_prefixes(tag) {
        tx.execute(
            "INSERT OR IGNORE INTO tag_prefix(memory_id, prefix) VALUES (?, ?)",
            params![memory_id, prefix],
        )?;
    }
    Ok(())
}

fn tag_prefixes(tag: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(tag.len());
    let mut end = 0;
    for ch in tag.chars() {
        end += ch.len_utf8();
        out.push(tag[..end].to_string());
    }
    out
}

fn re_enrich_memory(
    tx: &Transaction<'_>,
    id: &str,
    content: &str,
    confidence: f64,
) -> anyhow::Result<()> {
    let created_at = tx.query_row("SELECT created_at FROM memory WHERE id = ?", [id], |row| {
        row.get::<_, i64>(0)
    })?;
    tx.execute("DELETE FROM memory_entity WHERE memory_id = ?", [id])?;
    tx.execute("DELETE FROM statement WHERE memory_id = ?", [id])?;
    tx.execute(
        "DELETE FROM edge WHERE kind = ? AND (src = ? OR dst = ?)",
        params![RelationKind::PrecededBy as i64, id, id],
    )?;
    tx.execute(
        "DELETE FROM tag WHERE memory_id = ? AND tag LIKE 'entity:%'",
        [id],
    )?;
    tx.execute("DELETE FROM tag_prefix WHERE memory_id = ?", [id])?;
    enrich_memory(tx, id, content, confidence)?;
    derive_temporal_edges(tx, id, created_at)?;
    Ok(())
}

fn entity_kind_tag(kind: i64) -> &'static str {
    match kind {
        0 => "people",
        1 => "orgs",
        2 => "places",
        3 => "projects",
        4 => "tech",
        5 => "concepts",
        6 => "paths",
        _ => "other",
    }
}

fn delete_confirmation_token(ids: &[String]) -> String {
    let mut ids = ids.to_vec();
    ids.sort();
    let mut hasher = sha2::Sha256::new();
    for id in ids {
        sha2::Digest::update(&mut hasher, id.as_bytes());
    }
    format!(
        "delete-{}",
        &hex::encode(sha2::Digest::finalize(hasher))[..16]
    )
}

fn query_pairs(conn: &Connection, sql: &str) -> anyhow::Result<Value> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut map = serde_json::Map::new();
    for (k, v) in rows {
        map.insert(MemoryType::from_i64(k)?.as_str().to_string(), json!(v));
    }
    Ok(Value::Object(map))
}

fn query_tag_counts(conn: &Connection) -> anyhow::Result<Value> {
    let mut stmt = conn.prepare(
        "SELECT tag, COUNT(*) FROM tag GROUP BY tag ORDER BY COUNT(*) DESC, tag LIMIT 50",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(json!({"tag": row.get::<_, String>(0)?, "count": row.get::<_, i64>(1)?}))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!(rows))
}
