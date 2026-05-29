# agskmem design

agskmem is a single Rust MCP server backed by one user-global SQLite database. It replaces the old multi-process memory stack with one binary, one database file, and no required daemon, container, npm bridge, graph server, or vector server.

The core intent is boring local reliability:

- SQLite is the source of truth.
- FTS rows, embeddings, tag prefixes, entities, statements, recall metrics, temporal edges, and the in-memory graph are derived state and must be repairable.
- Coding agents never pass raw embedding vectors; agskmem generates vectors server-side.
- Memory writes should store durable user corrections, finalized decisions, and user-articulated patterns, not transient session summaries or speculation.
- Existing facts should be corrected with `update_memory`; duplicates should be merged or superseded rather than accumulated.
- Connections should be explicit and meaningful: causal, preference, provenance, invalidation, example, and containment links are preferred over vague hubs.

## Runtime shape

- `agskmem serve` runs a newline-delimited stdio MCP JSON-RPC server. `serve --http` is rejected in this build.
- MCP `initialize` advertises protocol `2024-11-05`, tool capability, server info, and the server instructions.
- `agskmem install <client>` writes or prints client configuration for supported agents. Client-specific installers refuse conflicting `agskmem` entries unless `--force` is supplied.
- JSON-client installs write `mcpServers.agskmem` with stdio transport, `agskmem serve`, a 30-second timeout, and `enabled: true`.
- TOML-client installs append `[mcp_servers.agskmem]`.
- `agskmem export` and `agskmem import` use SQLite online backup. Export also writes a JSON manifest next to the backup.
- `agskmem import-jsonl` imports memory rows from JSONL, skipping blank or content-less entries.
- `agskmem import-automem` imports legacy JSON array or JSONL exports, updates existing rows by id, imports relations, ignores external vector stores, and repairs indexes when edges are added.
- `agskmem reembed` and `agskmem repair` rebuild derived indexes from source tables.

The default database path is `${AGSKMEM_DB:-${XDG_DATA_HOME:-~/.local/share}/agskmem/agskmem.sqlite3}`. The default config path is `${XDG_CONFIG_HOME:-~/.config}/agskmem/config.toml` when the platform reports a config directory.

Config is loaded as defaults, optional TOML file overrides, then `AGSKMEM_DB`, CLI `--db`, and `AGSKMEM_LOG_LEVEL`. Validation rejects invalid embedding dimensions, recall weights that do not sum to 1, zero hard content limit, and soft content limits above hard limits.

## Agent memory policy

The MCP server instructions and installed policy are part of the design:

- Use `startup_recall` at the start of coding sessions.
- During long sessions, revisit memory context roughly every five user-assistant rounds with `recall_memory` or `startup_recall` as appropriate.
- Use `recall_memory` before explicit memory questions and before decisions that may depend on prior preferences, corrections, project decisions, or patterns.
- Treat `tags` as hard filters and `context_tags` as soft boosts.
- Use canonical project slug tags for project-specific memories.
- Store stable corrections, finalized decisions, and user-articulated patterns.
- Do not store transient session summaries or speculative notes.
- Use `associate_memories` for durable causal, preference, provenance, invalidation, and example links.
- Use `update_memory` for corrections to an existing fact.
- Bulk tag deletes require `delete_memory` dry-run first and then the returned `confirmation_token`.

## Configuration defaults

Default embeddings use local BGE-M3 through fastembed; classification remains local:

- `embed.provider = "fastembed-bgem3"`, `embed.model = "BGEM3Q"`, `embed.dims = 1024`.
- BGE-M3 uses fastembed 5.14's `Bgem3Embedding` over `gpahal/bge-m3-onnx-int8`; agskmem stores dense, sparse, and ColBERT outputs. Dense vectors remain in `embedding`, sparse token weights in `embedding_sparse`, and ColBERT token vectors in `embedding_colbert`.
- `classification.provider = "local"` with a deterministic local classifier and `classification_cache`.
- `content.soft_limit_bytes = 500`, `content.hard_limit_bytes = 2000`, `content.auto_summarize = true`, `content.summary_target_chars = 300`.
- Recall weights are vector `0.20`, sparse `0.15`, ColBERT `0.20`, keyword `0.15`, PPR `0.10`, tag overlap `0.05`, exact phrase `0.03`, importance `0.04`, recency `0.04`, confidence `0.02`, reliability `0.02`.
- Recall uses `mmr_lambda = 0.7`, `per_source_limit = 200`, and `adaptive_floor = true`.
- PPR uses `alpha = 0.15`, `epsilon = 1e-4`, `max_pushes = 50000`, CSR rebuild threshold `1024`, and CSR rebuild interval `60s`.
- Decay uses base `0.005`, floor factor `0.10`, archive threshold `0.05`, delete threshold `0.01`, and grace window `30` days.

## Storage

The migrations create:

- `meta`: schema/app/embedding metadata.
- `schema_history`: forward-only migration records with SHA-256 migration hashes; a hash mismatch aborts startup.
- `memory`: canonical records with UUID/string ids, content, summary, type, importance, confidence, relevance, reliability, metadata, source, timestamps, validity windows, archived flag, and protected flag.
- `memory_fts`: external-content FTS5 index over memory content and summaries.
- `tag`: normalized lower-case exact tags.
- `tag_prefix`: derived prefixes for prefix-style tag lookup support.
- `entity` and `memory_entity`: deterministic entity mentions.
- `edge`: typed relationship graph edges with strength, confidence, metadata, and timestamps.
- `statement` and `statement_fts`: extracted factual statements with provenance to source memories.
- `embedding`: normalized little-endian `f32` vector blobs with model, dimension count, norm, and creation time.
- `embedding_sparse`: BGE-M3 sparse lexical token weights by memory and token id.
- `embedding_colbert`: BGE-M3 ColBERT token vectors by memory and token index.
- `embedding_job`: retry queue for future embedders that cannot encode immediately.
- `enrichment_job`: explicit re-enrichment queue.
- `classification_cache`: content-hash keyed local classification cache.
- `recall_metric`: recent recall timing/candidate metrics.

Writers use a single mutex-protected SQLite writer. Migrations and canonical write tools use `BEGIN IMMEDIATE`; connections enable WAL, foreign keys, a busy timeout, in-memory temp store, and mmap on the writer. Mutating tools rebuild or republish derived graph/index state when needed.

## Embeddings and content governance

`FastEmbedBgeM3Embedder` is the default embedder. It calls fastembed's BGE-M3 joint dense/sparse/ColBERT model in batches capped at 8 and stores all three outputs: dense 1024-dimensional vectors, sparse token weights, and ColBERT token vectors. `LocalHashEmbedder` remains available only as an explicit `embed.provider = "local"` fallback for tests and offline recovery. The blob codec rejects malformed `f32` blobs and dimension mismatches.

Store/update content passes through governance:

- Content above `hard_limit_bytes` is rejected.
- Content above `soft_limit_bytes` is deterministically summarized when local auto-summarization is enabled.
- Summarization metadata records original content, lengths, summary model, and creation time; skipped summarization is recorded.
- Classification is explicit type first, then cache, then deterministic local marker rules. Classification metadata is stored with the memory.

Memory types are `Decision`, `Pattern`, `Preference`, `Style`, `Habit`, `Insight`, `Context`, and `Statement`; `Memory` parses as `Context`.

## Write path

`store_memory` accepts top-level single-memory fields through the public MCP schema so tool logs show visible content. The internal Rust argument type also supports a batch `memories` vector for non-schema callers. Blank optional strings are treated as absent.

For each stored memory agskmem:

1. chooses the provided id or a UUIDv7 id,
2. applies content governance,
3. classifies the memory,
4. normalizes and inserts tags,
5. stores the memory and embedding,
6. extracts entities and statements,
7. derives temporal `PRECEDED_BY` edges from shared entities within a seven-day window,
8. commits the transaction,
9. rebuilds derived indexes and republishes the graph.

`update_memory` patches only provided fields. Content or summary changes re-embed and re-enrich the row. Tag changes replace tags, rebuild tag prefixes, and re-enrich. Metadata patches merge into existing metadata. Updates can set importance, confidence, relevance, reliability, source, validity windows, archived, and protected.

`delete_memory` deletes by id immediately. Bulk delete by tags is guarded by a dry-run confirmation token computed from the target ids. Deletion repairs indexes and graph state.

`associate_memories` upserts only authorable relation kinds and requires both endpoints to exist.

## Recall

`recall_memory` has three effective modes:

1. ID fetch by `memory_id`.
2. Exact tag enumeration when no query is provided and tags are present.
3. Ranked search.

Ranked search combines candidate sources:

- dense embedding cosine search,
- sparse token-weight search,
- FTS5 keyword matches,
- exact phrase matches,
- entity matches,
- recent/high-relevance fallback rows,
- decomposed terms when `auto_decompose` is true,
- caller `priority_ids`.

Hard filters are applied before scoring: all requested tags, excluded tags, current-state semantics, `as_of`, `start`, `end`, `time_query`, and `time_range`. `queries[]` joins into a single query string. `context`, `language`, `active_path`, `context_types`, and `context_tags` enrich soft context tags used only for scoring.

Score components are computed from the candidate content and state, not from the retrieval path: dense vector cosine, sparse lexical score, ColBERT late-interaction score, lexical keyword overlap, PPR, tag overlap, exact phrase, importance, recency, confidence, reliability, and context bonus. Configured weights produce the final score. `priority_ids` bypass score-floor pruning. Hits can be sorted by score or time; broad score-sorted result sets use MMR to reduce near-duplicates. Returned memories are touched by updating `last_accessed`.

Graph expansion is enabled by `expand_relations` or `expand_entities`; both currently seed the same PPR path. `expand_respect_tags` makes expanded nodes obey the original filters. `expand_min_importance` prunes low-importance expanded nodes.

Compatibility fields accepted by the MCP schema but not yet used for distinct behavior include `tag_mode`, `tag_match`, `relation_limit`, `expansion_limit`, and `expand_min_strength`. Current tag filtering is exact all-tag matching.

`startup_recall` returns current, non-archived memories ordered by importance and update time. `trace_recall` runs normal recall and preserves score components in the compact MCP response.

## Current-state semantics

The default is `current_only=true`:

- Archived memories are hidden.
- Future `t_valid` memories are hidden.
- Expired `t_invalid` memories are hidden.
- Memories with active outgoing `INVALIDATED_BY` or `EVOLVED_INTO` replacements are hidden.
- `CONTRADICTS` does not suppress by itself.

`current_only=false` can return non-active rows with state labels: `archived`, `future`, `expired`, `superseded`, or `active`.

## Graph

Edges are stored in SQLite and loaded into an in-memory CSR cache. Readers use an `ArcSwap<Graph>` pointer, so recall reads do not take a graph lock. `repair_index` rebuilds the CSR from `edge` and atomically publishes the new graph.

The CSR stores id/node maps, row pointers, destination node indexes, relation kind, strength, and confidence. Adjacency lists are deterministic. Graph expansion uses a single-threaded forward-push Personalized PageRank implementation. Seed scores are normalized, edge mass is multiplied by relation default weight, clamped strength, and clamped confidence, and high-degree nodes naturally split mass across outgoing edges.

Authorable relation kinds are:

- `RELATES_TO`
- `LEADS_TO`
- `OCCURRED_BEFORE`
- `PREFERS_OVER`
- `EXEMPLIFIES`
- `CONTRADICTS`
- `REINFORCES`
- `INVALIDATED_BY`
- `EVOLVED_INTO`
- `DERIVED_FROM`
- `PART_OF`

System-managed relation kinds are:

- `SIMILAR_TO`
- `PRECEDED_BY`
- `DISCOVERED`
- `EXTRACTED_FROM`

Graph tools expose direct neighbors, graph stats, a bounded snapshot, graph-only related-memory lookup, and PPR components through trace recall.

## Enrichment and consolidation

Store/update performs cheap deterministic enrichment inline:

- capitalized full-name-like entity mentions,
- path-like entity mentions,
- `entity:<kind>:<slug>` tags,
- up to eight sentence-split statements,
- statement rows with source-memory provenance,
- temporal `PRECEDED_BY` edges for shared entities in a seven-day window.

`repair_index` rebuilds memory FTS, statement FTS, tag prefixes, prunes orphan entities, rebuilds temporal edges, and republishes the graph.

Administrative consolidation supports explicit dry-run or mutating modes:

- `decay`: recomputes relevance from age, access recency, graph degree, importance, and confidence.
- `forget`: archives or deletes unprotected, low-importance rows outside the grace window when relevance crosses configured thresholds.
- `creative`: creates conservative `DISCOVERED` edges among high-relevance active memory pairs.
- `cluster`: currently reports existing `SIMILAR_TO` edge count and does not create meta-patterns.
- `all`: runs all modes.

`consolidate_status` and `enrichment_status` expose persisted queue/metric state. They currently report no live background scheduler or worker. `enrichment_reprocess` queues selected ids for explicit reprocessing and can force replacement of existing queue entries.

## MCP tool surface

The server exposes AutoMem-compatible tool names:

- Writes: `store_memory`, `update_memory`, `delete_memory`, `associate_memories`.
- Reads: `recall_memory`, `startup_recall`, `get_related_memories`, `graph_snapshot`, `graph_neighbors`, `graph_stats`, `trace_recall`.
- Introspection: `check_database_health`, `analyze_memories`, `relation_types`, `memory_types`.
- Administration: `consolidate`, `consolidate_status`, `enrichment_status`, `enrichment_reprocess`, `repair_index`, `reembed`, `export_backup`, `import_backup`.

MCP recall-like responses are compact by default: noisy metadata, raw score components, and transport-only fields are omitted unless tracing/debug output is requested. Human-readable `results_text` is returned for recall-style tools.

## Memory stewardship intent

The memory graph should stay useful over time:

- Normalize taxonomy and tags instead of allowing parallel synonyms.
- Prefer stable project namespace tags such as `agskmem`, `moirai`, `opengrid`, and `capture_rs`.
- Treat generated entity tags as recall aids, not canonical project tags.
- Merge duplicate preferences into one stronger canonical memory.
- Archive or delete weak transient notes that are not reusable for future decisions.
- Use high-confidence explicit edges over many vague `RELATES_TO` links.
- Link preferences to concrete examples with `EXEMPLIFIES`.
- Link decisions to implementation outcomes with `DERIVED_FROM`.
- Link superseded memories with `INVALIDATED_BY` or `EVOLVED_INTO`.
- Avoid generic hubs unless the memory is truly foundational.

## Security and privacy

agskmem keeps all data local. No telemetry leaves the machine. Provider keys, if non-local providers are added, are read from config/env and must never be logged. The active implementation does not require provider credentials.

Backups are SQLite backups; they contain the same private memory content as the live database and should be protected accordingly.
