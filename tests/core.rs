use agskmem::design_types::*;
use agskmem::{AgskMem, Config};
use serde_json::{Value, json};
use tempfile::TempDir;

fn app() -> (TempDir, AgskMem) {
    let tmp = TempDir::new().expect("tempdir");
    let mut cfg = Config::default();
    cfg.db.path = tmp.path().join("agskmem.sqlite3");
    (tmp, AgskMem::open(cfg).expect("open app"))
}

fn assert_mcp_memory_is_compact(memory: &Value) {
    for field in [
        "metadata",
        "source",
        "created_at",
        "updated_at",
        "last_accessed",
        "relevance",
        "reliability",
        "archived",
        "protected",
        "components",
    ] {
        assert!(
            memory.get(field).is_none(),
            "{field} leaked into MCP recall output"
        );
    }
    assert!(memory.get("id").and_then(Value::as_str).is_some());
    assert!(memory.get("content").and_then(Value::as_str).is_some());
    assert!(memory.get("type").and_then(Value::as_str).is_some());
    assert!(memory.get("tags").and_then(Value::as_array).is_some());
}

#[test]
fn store_recall_and_tag_enumeration_are_end_to_end() {
    let (_tmp, app) = app();
    let stored = app
        .store_memory(StoreMemoryArgs {
            content: Some("Alice Smith prefers Rust for local memory tooling.".to_string()),
            tags: vec!["ProjectX".to_string(), "Preference".to_string()],
            memory_type: Some("Preference".to_string()),
            importance: Some(0.8),
            confidence: Some(0.9),
            metadata: Some(json!({"source":"test"})),
            ..Default::default()
        })
        .expect("store");
    assert_eq!(stored.ids.len(), 1);

    let recall = app
        .recall_memory(RecallArgs {
            query: Some("Alice Rust memory".to_string()),
            limit: Some(5),
            ..Default::default()
        })
        .expect("recall");
    let results = recall
        .get("results")
        .and_then(|v| v.as_array())
        .expect("results array");
    assert_eq!(
        results[0].get("id").and_then(|v| v.as_str()),
        Some(stored.ids[0].as_str())
    );

    let page = app
        .recall_memory(RecallArgs {
            tags: vec!["projectx".to_string()],
            limit: Some(10),
            ..Default::default()
        })
        .expect("tag page");
    assert_eq!(page["results"].as_array().expect("results").len(), 1);
    assert_eq!(page["results"][0]["metadata"]["source"], "test");
}

#[test]
fn mcp_recall_outputs_are_compact_by_default() {
    let (_tmp, app) = app();
    app.store_memory(StoreMemoryArgs {
        content: Some("Compact MCP recall should preserve the useful memory text.".to_string()),
        tags: vec!["compact-mcp".to_string()],
        memory_type: Some("Context".to_string()),
        importance: Some(0.7),
        confidence: Some(0.8),
        metadata: Some(json!({
            "enrichment": {
                "forced": false,
                "patterns_detected": [{"type": "Context", "similar_memories": 8}],
                "semantic_neighbors": ["unneeded context noise"],
                "temporal_links": 5
            },
            "entities": {"organizations": ["NoisyCo"]},
            "source": "test"
        })),
        source: Some("migration".to_string()),
        summary: Some("Compact MCP recall.".to_string()),
        ..Default::default()
    })
    .expect("store");

    let recall = agskmem::mcp::call_tool(
        &app,
        "recall_memory",
        json!({"query": "compact MCP recall", "limit": 5}),
    )
    .expect("mcp recall");
    let recall_memory = recall["results"]
        .as_array()
        .expect("recall results")
        .first()
        .expect("recall result");
    assert_mcp_memory_is_compact(recall_memory);
    assert!(recall_memory.get("score").is_some());
    assert_eq!(recall_memory["summary"], "Compact MCP recall.");

    let startup =
        agskmem::mcp::call_tool(&app, "startup_recall", json!({"limit": 5})).expect("mcp startup");
    let startup_memory = startup["results"]
        .as_array()
        .expect("startup results")
        .first()
        .expect("startup result");
    assert_mcp_memory_is_compact(startup_memory);
    assert!(startup_memory.get("score").is_none());

    let trace = agskmem::mcp::call_tool(
        &app,
        "trace_recall",
        json!({"query": "compact MCP recall", "limit": 5}),
    )
    .expect("mcp trace");
    let trace_memory = trace["trace"]["results"]
        .as_array()
        .expect("trace results")
        .first()
        .expect("trace result");
    assert!(trace_memory.get("metadata").is_none());
    assert!(trace_memory.get("components").is_some());
}

#[test]
fn empty_optional_ids_are_treated_as_absent() {
    let (_tmp, app) = app();
    let stored = app
        .store_memory(StoreMemoryArgs {
            id: Some(String::new()),
            content: Some("Blank transport ids should not become memory ids.".to_string()),
            tags: vec!["blank-id".to_string()],
            ..Default::default()
        })
        .expect("store");

    assert_eq!(stored.ids.len(), 1);
    assert!(!stored.ids[0].is_empty());

    let page = app
        .recall_memory(RecallArgs {
            memory_id: Some(String::new()),
            tags: vec!["blank-id".to_string()],
            limit: Some(10),
            ..Default::default()
        })
        .expect("tag recall");
    let results = page["results"].as_array().expect("results");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"].as_str(), Some(stored.ids[0].as_str()));
}

#[test]
fn superseded_memories_are_hidden_by_default() {
    let (_tmp, app) = app();
    let old = app
        .store_memory(StoreMemoryArgs {
            content: Some("The deployment target is blue.".to_string()),
            ..Default::default()
        })
        .expect("store old")
        .ids[0]
        .clone();
    let new = app
        .store_memory(StoreMemoryArgs {
            content: Some("The deployment target is green.".to_string()),
            ..Default::default()
        })
        .expect("store new")
        .ids[0]
        .clone();
    app.associate_memories(AssociateArgs {
        memory1_id: old.clone(),
        memory2_id: new.clone(),
        relation_type: "INVALIDATED_BY".to_string(),
        strength: Some(0.9),
        confidence: Some(0.9),
        metadata: None,
    })
    .expect("associate");

    let current = app
        .recall_memory(RecallArgs {
            query: Some("deployment target".to_string()),
            limit: Some(10),
            ..Default::default()
        })
        .expect("current recall");
    let ids: Vec<_> = current["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|v| v["id"].as_str().expect("id").to_string())
        .collect();
    assert!(ids.contains(&new));
    assert!(!ids.contains(&old));

    let historical = app
        .recall_memory(RecallArgs {
            query: Some("deployment target".to_string()),
            current_only: Some(false),
            limit: Some(10),
            ..Default::default()
        })
        .expect("historical recall");
    let ids: Vec<_> = historical["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|v| v["id"].as_str().expect("id").to_string())
        .collect();
    assert!(ids.contains(&old));
}

#[test]
fn bulk_delete_requires_dry_run_confirmation() {
    let (_tmp, app) = app();
    app.store_memory(StoreMemoryArgs {
        content: Some("Delete me one".to_string()),
        tags: vec!["deleteme".to_string()],
        ..Default::default()
    })
    .expect("store 1");
    app.store_memory(StoreMemoryArgs {
        content: Some("Delete me two".to_string()),
        tags: vec!["deleteme".to_string()],
        ..Default::default()
    })
    .expect("store 2");

    let err = app
        .delete_memory(DeleteMemoryArgs {
            memory_id: Some(String::new()),
            tags: vec!["deleteme".to_string()],
            ..Default::default()
        })
        .expect_err("confirmation required");
    assert!(err.to_string().contains("confirmation_token"));

    let dry = app
        .delete_memory(DeleteMemoryArgs {
            memory_id: Some(String::new()),
            tags: vec!["deleteme".to_string()],
            dry_run: true,
            ..Default::default()
        })
        .expect("dry run");
    assert_eq!(dry.deleted, 2);
    assert!(dry.dry_run);
    assert_eq!(
        app.recall_memory(RecallArgs {
            tags: vec!["deleteme".to_string()],
            limit: Some(10),
            ..Default::default()
        })
        .expect("after dry run")["results"]
            .as_array()
            .expect("results")
            .len(),
        2
    );
    let token = dry.confirmation_token.expect("token");
    let deleted = app
        .delete_memory(DeleteMemoryArgs {
            memory_id: Some(String::new()),
            tags: vec!["deleteme".to_string()],
            confirmation_token: Some(token),
            ..Default::default()
        })
        .expect("delete");
    assert_eq!(deleted.deleted, 2);
}

#[test]
fn graph_related_uses_associated_edges() {
    let (_tmp, app) = app();
    let a = app
        .store_memory(StoreMemoryArgs {
            content: Some("Pattern: prefer boring reliable systems.".to_string()),
            ..Default::default()
        })
        .expect("store a")
        .ids[0]
        .clone();
    let b = app
        .store_memory(StoreMemoryArgs {
            content: Some("Example: SQLite is the source of truth.".to_string()),
            ..Default::default()
        })
        .expect("store b")
        .ids[0]
        .clone();
    app.associate_memories(AssociateArgs {
        memory1_id: a.clone(),
        memory2_id: b.clone(),
        relation_type: "EXEMPLIFIES".to_string(),
        strength: Some(1.0),
        confidence: Some(1.0),
        metadata: None,
    })
    .expect("associate");

    let related = app
        .get_related_memories(RelatedArgs {
            memory_id: a.clone(),
            limit: Some(5),
        })
        .expect("related");
    let results = related["results"].as_array().expect("results");
    assert!(results.iter().any(|row| row["memory"]["id"] == b));
    let mcp_related = agskmem::mcp::call_tool(
        &app,
        "get_related_memories",
        json!({"memory_id": a, "limit": 5}),
    )
    .expect("mcp related");
    let mcp_results = mcp_related["results"].as_array().expect("mcp results");
    let mcp_memory = mcp_results
        .iter()
        .find(|row| row["memory"]["id"] == b)
        .expect("mcp related memory");
    assert_mcp_memory_is_compact(&mcp_memory["memory"]);
    assert_eq!(app.graph_stats()["directed_edges"], 1);
}

#[test]
fn temporal_edges_are_derived_from_shared_entities() {
    let (_tmp, app) = app();
    let older = app
        .store_memory(StoreMemoryArgs {
            content: Some("Ada Lovelace designed reliable analytical notes.".to_string()),
            timestamp: Some("2026-05-20T00:00:00Z".to_string()),
            ..Default::default()
        })
        .expect("store older")
        .ids[0]
        .clone();
    let newer = app
        .store_memory(StoreMemoryArgs {
            content: Some("Ada Lovelace documented reliable analytical engines.".to_string()),
            timestamp: Some("2026-05-21T00:00:00Z".to_string()),
            ..Default::default()
        })
        .expect("store newer")
        .ids[0]
        .clone();
    let outside_window = app
        .store_memory(StoreMemoryArgs {
            content: Some("Ada Lovelace reviewed later analytical engines.".to_string()),
            timestamp: Some("2026-06-15T00:00:00Z".to_string()),
            ..Default::default()
        })
        .expect("store outside window")
        .ids[0]
        .clone();

    let neighbors = app.graph_neighbors(&newer);
    let rows = neighbors["neighbors"].as_array().expect("neighbors");
    assert!(
        rows.iter()
            .any(|row| row["id"] == older && row["kind"] == "PRECEDED_BY")
    );
    assert!(!rows.iter().any(|row| row["id"] == outside_window));
    assert_eq!(app.graph_stats()["directed_edges"], 1);

    app.repair_index().expect("repair");
    assert_eq!(app.graph_stats()["directed_edges"], 1);

    app.update_memory(UpdateMemoryArgs {
        memory_id: newer.clone(),
        content: Some("Grace Hopper documented reliable compilers.".to_string()),
        ..Default::default()
    })
    .expect("remove shared entity");
    let after_update = app.graph_neighbors(&newer);
    assert!(
        !after_update["neighbors"]
            .as_array()
            .expect("neighbors")
            .iter()
            .any(|row| row["id"] == older && row["kind"] == "PRECEDED_BY")
    );
}

#[test]
fn content_governance_and_classification_are_applied() {
    let (_tmp, app) = app();
    let long = format!("Decision: {}.", "store compact memories ".repeat(40));
    let id = app
        .store_memory(StoreMemoryArgs {
            content: Some(long),
            metadata: Some(json!({"caller": "test"})),
            ..Default::default()
        })
        .expect("store governed")
        .ids[0]
        .clone();

    let row = app
        .recall_memory(RecallArgs {
            memory_id: Some(id),
            current_only: Some(false),
            ..Default::default()
        })
        .expect("fetch");
    let memory = &row["results"][0];
    assert_eq!(memory["type"], "Decision");
    assert_eq!(memory["metadata"]["caller"], "test");
    assert_eq!(memory["metadata"]["was_summarized"], true);
    assert!(memory["metadata"]["original_content"].as_str().is_some());
    assert!(memory["content"].as_str().expect("content").len() <= 300);
}

#[test]
fn hard_limit_rejects_agent_content() {
    let (_tmp, app) = app();
    let too_large = "x".repeat(app.config.content.hard_limit_bytes + 1);
    let err = app
        .store_memory(StoreMemoryArgs {
            content: Some(too_large),
            ..Default::default()
        })
        .expect_err("hard limit error");
    assert!(err.to_string().contains("hard limit"));
}

#[test]
fn update_reclassifies_unlocked_content_but_preserves_locked_classification() {
    let (_tmp, app) = app();
    let id = app
        .store_memory(StoreMemoryArgs {
            content: Some("Initial context fact.".to_string()),
            ..Default::default()
        })
        .expect("store")
        .ids[0]
        .clone();

    app.update_memory(UpdateMemoryArgs {
        memory_id: id.clone(),
        content: Some("Pattern: use boring reliable primitives.".to_string()),
        ..Default::default()
    })
    .expect("update pattern");
    let row = app
        .recall_memory(RecallArgs {
            memory_id: Some(id.clone()),
            current_only: Some(false),
            ..Default::default()
        })
        .expect("fetch pattern");
    assert_eq!(row["results"][0]["type"], "Pattern");

    app.update_memory(UpdateMemoryArgs {
        memory_id: id.clone(),
        metadata: Some(json!({"classification": {"locked": true}})),
        ..Default::default()
    })
    .expect("lock");
    app.update_memory(UpdateMemoryArgs {
        memory_id: id.clone(),
        content: Some("Decision: change the implementation.".to_string()),
        ..Default::default()
    })
    .expect("locked update");
    let locked = app
        .recall_memory(RecallArgs {
            memory_id: Some(id),
            current_only: Some(false),
            ..Default::default()
        })
        .expect("fetch locked");
    assert_eq!(locked["results"][0]["type"], "Pattern");
}

#[test]
fn enrichment_tables_status_and_prefixes_are_wired() {
    let (_tmp, app) = app();
    let id = app
        .store_memory(StoreMemoryArgs {
            content: Some("Ada Lovelace preferred reliable analytical engines.".to_string()),
            tags: vec!["project:analytical-engine".to_string()],
            ..Default::default()
        })
        .expect("store")
        .ids[0]
        .clone();

    let health = app.check_database_health().expect("health");
    assert!(health["counts"]["enrichment_job"].as_i64().is_some());
    assert!(health["counts"]["classification_cache"].as_i64().is_some());
    assert!(health["counts"]["tag_prefix"].as_i64().is_some());

    let status = agskmem::mcp::call_tool(&app, "enrichment_status", json!({})).expect("status");
    assert_eq!(status["queue_depth"], 0);
    let queued = agskmem::mcp::call_tool(
        &app,
        "enrichment_reprocess",
        json!({"ids": [id], "forced": true}),
    )
    .expect("queue");
    assert_eq!(queued["queued"], 1);
    let consolidate =
        agskmem::mcp::call_tool(&app, "consolidate_status", json!({})).expect("consolidate status");
    assert_eq!(consolidate["scheduler_running"], false);
}
