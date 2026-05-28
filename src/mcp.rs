use crate::{AgskMem, design_types::*};
use anyhow::{Context, anyhow};
use serde::de::{DeserializeOwned, IntoDeserializer};
use serde_json::{Value, json};
use std::{fmt::Write as _, sync::Arc};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub async fn serve_stdio(app: Arc<AgskMem>) -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut lines = BufReader::new(stdin).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                write_json(&mut stdout, &json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":err.to_string()}})).await?;
                continue;
            }
        };
        if request.get("id").is_none() {
            let _ = handle_notification(app.clone(), &request).await;
            continue;
        }
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or(Value::Null);
        let response = match handle_request(app.clone(), method, params).await {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err(err) => {
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32603,"message":err.to_string()}})
            }
        };
        write_json(&mut stdout, &response).await?;
    }
    Ok(())
}

async fn write_json(stdout: &mut tokio::io::Stdout, value: &Value) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    stdout.write_all(&bytes).await?;
    stdout.flush().await?;
    Ok(())
}

async fn handle_notification(_app: Arc<AgskMem>, request: &Value) -> anyhow::Result<()> {
    match request.get("method").and_then(Value::as_str).unwrap_or("") {
        "notifications/initialized" | "$/cancelRequest" => Ok(()),
        _ => Ok(()),
    }
}

async fn handle_request(app: Arc<AgskMem>, method: &str, params: Value) -> anyhow::Result<Value> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "agskmem", "version": env!("CARGO_PKG_VERSION")},
            "instructions": SERVER_INSTRUCTIONS
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_specs()})),
        "tools/call" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .context("tools/call missing name")?;
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));
            let result = call_tool(&app, name, args)?;
            let text = render_tool_result(name, &result)?;
            Ok(
                json!({"content":[{"type":"text","text":text}],"structuredContent":result,"isError":false}),
            )
        }
        _ => anyhow::bail!("unsupported method {method}"),
    }
}
#[derive(serde::Deserialize)]
struct StartupRecallArgs {
    #[serde(default)]
    limit: Option<u64>,
}

fn parse_tool_args<T>(tool: &str, args: Value) -> anyhow::Result<T>
where
    T: DeserializeOwned,
{
    let deserializer = args.into_deserializer();
    serde_path_to_error::deserialize(deserializer).map_err(|err| {
        let path = err.path().to_string();
        if path == "." {
            anyhow!("{tool} arguments: {}", err.inner())
        } else {
            anyhow!("{tool}.{path}: {}", err.inner())
        }
    })
}

pub fn call_tool(app: &AgskMem, name: &str, args: Value) -> anyhow::Result<Value> {
    match name {
        "store_memory" => {
            Ok(json!(app.store_memory(parse_tool_args::<
                StoreMemoryArgs,
            >(
                "store_memory", args,
            )?)?))
        }
        "update_memory" => {
            app.update_memory(parse_tool_args::<UpdateMemoryArgs>("update_memory", args)?)
        }
        "delete_memory" => Ok(json!(app.delete_memory(parse_tool_args::<
            DeleteMemoryArgs,
        >(
            "delete_memory", args,
        )?)?)),
        "associate_memories" => app.associate_memories(parse_tool_args::<AssociateArgs>(
            "associate_memories",
            args,
        )?),
        "recall_memory" => {
            let args = parse_tool_args::<RecallArgs>("recall_memory", args)?;
            let state_debug = args.state_debug;
            let prompt = recall_prompt(&args, args.limit.unwrap_or(10));
            app.recall_memory(args)
                .map(|value| compact_recall_response(value, state_debug, false, Some(prompt)))
        }
        "get_related_memories" => {
            let args = parse_tool_args::<RelatedArgs>("get_related_memories", args)?;
            let limit = args.limit.unwrap_or(20).clamp(1, 200);
            let prompt = format!(
                "get_related_memories memory_id={}; limit={limit}",
                quote_for_prompt(&args.memory_id)
            );
            app.get_related_memories(args)
                .map(|value| compact_related_response(value, Some(prompt)))
        }
        "graph_snapshot" => Ok(app.graph_snapshot()),
        "graph_neighbors" => Ok(app.graph_neighbors(
            parse_tool_args::<GraphNeighborsArgs>("graph_neighbors", args)?
                .memory_id
                .as_str(),
        )),
        "graph_stats" => Ok(app.graph_stats()),
        "trace_recall" => {
            let args = parse_tool_args::<RecallArgs>("trace_recall", args)?;
            let state_debug = args.state_debug;
            let prompt = recall_prompt(&args, args.limit.unwrap_or(10));
            app.trace_recall(args)
                .map(|value| compact_trace_response(value, state_debug, prompt))
        }
        "check_database_health" => app.check_database_health(),
        "analyze_memories" => app.analyze_memories(),
        "relation_types" => Ok(app.relation_types()),
        "memory_types" => Ok(app.memory_types()),
        "startup_recall" => {
            let args = parse_tool_args::<StartupRecallArgs>("startup_recall", args)?;
            let limit = args.limit.unwrap_or(20);
            Ok(app.startup_recall(limit as usize).map(|value| {
                compact_recall_response(
                    value,
                    false,
                    false,
                    Some(format!("startup_recall limit={limit}")),
                )
            })?)
        }
        "consolidate" => app.consolidate(parse_tool_args::<ConsolidateArgs>("consolidate", args)?),
        "consolidate_status" => app.consolidate_status(),
        "enrichment_status" => app.enrichment_status(),
        "enrichment_reprocess" => app.enrichment_reprocess(parse_tool_args::<
            EnrichmentReprocessArgs,
        >("enrichment_reprocess", args)?),
        "repair_index" => {
            app.repair_index()?;
            Ok(json!({"repaired": true}))
        }
        "reembed" => app.reembed(parse_tool_args::<ReembedArgs>("reembed", args)?),
        "export_backup" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .context("export_backup.path required")?;
            app.export_backup(std::path::Path::new(path))
        }
        "import_backup" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .context("import_backup.path required")?;
            app.import_backup(std::path::Path::new(path))
        }
        _ => anyhow::bail!("unknown tool {name}"),
    }
}
fn render_tool_result(name: &str, result: &Value) -> anyhow::Result<String> {
    if matches!(
        name,
        "recall_memory" | "startup_recall" | "get_related_memories"
    ) && let Some(text) = result.get("results_text").and_then(Value::as_str)
    {
        return Ok(format_prompted_text(result.get("prompt"), text));
    }
    if name == "trace_recall"
        && let Some(trace) = result.get("trace")
        && let Some(text) = trace.get("results_text").and_then(Value::as_str)
    {
        return Ok(format_prompted_text(trace.get("prompt"), text));
    }
    Ok(serde_json::to_string_pretty(result)?)
}

fn format_prompted_text(prompt: Option<&Value>, text: &str) -> String {
    if let Some(prompt) = prompt.and_then(Value::as_str) {
        format!("{prompt}\n\n{text}")
    } else {
        text.to_string()
    }
}

fn compact_trace_response(value: Value, state_debug: bool, prompt: String) -> Value {
    match value {
        Value::Object(mut object) => {
            let Some(trace) = object.remove("trace") else {
                return Value::Object(object);
            };
            json!({"trace": compact_recall_response(trace, state_debug, true, Some(prompt))})
        }
        other => other,
    }
}

fn compact_related_response(value: Value, prompt: Option<String>) -> Value {
    match value {
        Value::Object(mut object) => {
            let results = match object.remove("results") {
                Some(Value::Array(results)) => results
                    .into_iter()
                    .map(|row| match row {
                        Value::Object(mut row) => {
                            let mut compact = serde_json::Map::new();
                            move_field(&mut row, &mut compact, "score");
                            if let Some(memory) = row.remove("memory") {
                                compact.insert(
                                    "memory".to_string(),
                                    compact_memory(memory, false, false),
                                );
                            }
                            Value::Object(compact)
                        }
                        other => other,
                    })
                    .collect::<Vec<_>>(),
                Some(other) => return Value::Object(object_with_value(object, "results", other)),
                None => Vec::new(),
            };
            let mut compact = serde_json::Map::new();
            if let Some(prompt) = prompt {
                compact.insert("prompt".to_string(), Value::String(prompt));
            }
            compact.insert(
                "results_text".to_string(),
                Value::String(format_related_results(&results)),
            );
            compact.insert("result_count".to_string(), json!(results.len()));
            compact.extend(object);
            compact.insert("results".to_string(), Value::Array(results));
            Value::Object(compact)
        }
        other => other,
    }
}

fn compact_recall_response(
    value: Value,
    state_debug: bool,
    include_components: bool,
    prompt: Option<String>,
) -> Value {
    match value {
        Value::Object(mut object) => {
            let results = match object.remove("results") {
                Some(Value::Array(results)) => results
                    .into_iter()
                    .map(|result| compact_memory(result, state_debug, include_components))
                    .collect::<Vec<_>>(),
                Some(other) => return Value::Object(object_with_value(object, "results", other)),
                None => Vec::new(),
            };

            let mut compact = serde_json::Map::new();
            if let Some(prompt) = prompt {
                compact.insert("prompt".to_string(), Value::String(prompt));
            }
            compact.insert(
                "results_text".to_string(),
                Value::String(format_recall_results(&results)),
            );
            compact.insert("result_count".to_string(), json!(results.len()));
            move_field(&mut object, &mut compact, "has_more");
            move_field(&mut object, &mut compact, "limit");
            move_field(&mut object, &mut compact, "offset");
            move_field(&mut object, &mut compact, "cursor");
            compact.extend(object);
            compact.insert("results".to_string(), Value::Array(results));
            Value::Object(compact)
        }
        other => other,
    }
}
fn object_with_value(
    mut object: serde_json::Map<String, Value>,
    key: &str,
    value: Value,
) -> serde_json::Map<String, Value> {
    object.insert(key.to_string(), value);
    object
}

fn recall_prompt(args: &RecallArgs, default_limit: usize) -> String {
    let mut parts = Vec::new();

    if let Some(memory_id) = non_blank(args.memory_id.as_deref()) {
        parts.push(format!("memory_id={}", quote_for_prompt(memory_id)));
    } else {
        if let Some(query) = non_blank(args.query.as_deref()) {
            parts.push(format!("query={}", quote_for_prompt(query)));
        } else if !args.queries.is_empty() {
            let effective_query = args.queries.join(" ");
            parts.push(format!("queries={}", json!(&args.queries)));
            parts.push(format!(
                "effective_query={}",
                quote_for_prompt(effective_query.trim())
            ));
        } else if args.tags.is_empty() {
            parts.push("query=<empty: fallback ranking>".to_string());
        }

        if !args.tags.is_empty() {
            parts.push(format!("tags={}", json!(&args.tags)));
        }
        if !args.exclude_tags.is_empty() {
            parts.push(format!("exclude_tags={}", json!(&args.exclude_tags)));
        }
        if !args.context_tags.is_empty() {
            parts.push(format!("context_tags={}", json!(&args.context_tags)));
        }
        if let Some(context) = non_blank(args.context.as_deref()) {
            parts.push(format!("context={}", quote_for_prompt(context)));
        }
        if let Some(language) = non_blank(args.language.as_deref()) {
            parts.push(format!("language={}", quote_for_prompt(language)));
        }
        if let Some(active_path) = non_blank(args.active_path.as_deref()) {
            parts.push(format!("active_path={}", quote_for_prompt(active_path)));
        }
        if args.expand_relations {
            parts.push("expand_relations=true".to_string());
        }
        if args.expand_entities {
            parts.push("expand_entities=true".to_string());
        }
        if args.auto_decompose {
            parts.push("auto_decompose=true".to_string());
        }
        if let Some(sort) = non_blank(args.sort.as_deref().or(args.order_by.as_deref())) {
            parts.push(format!("sort={}", quote_for_prompt(sort)));
        }
    }

    parts.push(format!("limit={}", default_limit.clamp(1, 200)));
    let offset = args.offset.or(args.cursor).unwrap_or(0);
    if offset != 0 {
        parts.push(format!("offset={offset}"));
    }
    if args.current_only == Some(false) {
        parts.push("current_only=false".to_string());
    }
    if let Some(as_of) = non_blank(args.as_of.as_deref()) {
        parts.push(format!("as_of={}", quote_for_prompt(as_of)));
    }
    if let Some(time_query) = non_blank(args.time_query.as_deref()) {
        parts.push(format!("time_query={}", quote_for_prompt(time_query)));
    }

    format!("recall_memory {}", parts.join("; "))
}

fn non_blank(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn quote_for_prompt(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a string cannot fail")
}
fn format_related_results(results: &[Value]) -> String {
    let mut memories = Vec::with_capacity(results.len());
    for result in results {
        let Some(row) = result.as_object() else {
            memories.push(result.clone());
            continue;
        };
        let Some(Value::Object(memory)) = row.get("memory") else {
            memories.push(result.clone());
            continue;
        };
        let mut memory = memory.clone();
        if let Some(score) = row.get("score") {
            memory.insert("score".to_string(), score.clone());
        }
        memories.push(Value::Object(memory));
    }
    format_recall_results(&memories)
}

fn format_recall_results(results: &[Value]) -> String {
    if results.is_empty() {
        return "No memories matched.".to_string();
    }

    let mut text = String::new();
    for (index, result) in results.iter().enumerate() {
        if index != 0 {
            text.push('\n');
        }
        let Some(memory) = result.as_object() else {
            writeln!(&mut text, "{}. {}", index + 1, result)
                .expect("writing to String cannot fail");
            continue;
        };

        let content = string_field(memory, "content").unwrap_or("<missing content>");
        let summary = string_field(memory, "summary").filter(|summary| *summary != content);
        let title = summary.unwrap_or(content);
        writeln!(
            &mut text,
            "{}. {}",
            index + 1,
            truncate_for_display(title, 240)
        )
        .expect("writing to String cannot fail");

        write!(&mut text, "   ").expect("writing to String cannot fail");
        write_scalar(&mut text, memory, "id");
        write_scalar(&mut text, memory, "type");
        write_number(&mut text, memory, "score");
        write_number(&mut text, memory, "importance");
        write_number(&mut text, memory, "confidence");
        text.push('\n');

        if summary.is_some() {
            writeln!(
                &mut text,
                "   content: {}",
                truncate_for_display(content, 360)
            )
            .expect("writing to String cannot fail");
        }
        if let Some(tags) = memory
            .get("tags")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
        {
            write!(&mut text, "   tags: ").expect("writing to String cannot fail");
            for (tag_index, tag) in tags.iter().filter_map(Value::as_str).enumerate() {
                if tag_index != 0 {
                    text.push_str(", ");
                }
                text.push_str(tag);
            }
            text.push('\n');
        }
    }
    while text.ends_with('\n') {
        text.pop();
    }
    text
}

fn string_field<'a>(object: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    object.get(key).and_then(Value::as_str)
}

fn write_scalar(text: &mut String, object: &serde_json::Map<String, Value>, key: &str) {
    if let Some(value) = string_field(object, key) {
        write!(text, "{key}={value} ").expect("writing to String cannot fail");
    }
}

fn write_number(text: &mut String, object: &serde_json::Map<String, Value>, key: &str) {
    if let Some(value) = object.get(key).and_then(Value::as_f64) {
        write!(text, "{key}={value:.3} ").expect("writing to String cannot fail");
    }
}

fn truncate_for_display(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let mut truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        truncated.push('…');
    }
    truncated
}

fn compact_memory(value: Value, state_debug: bool, include_components: bool) -> Value {
    let Value::Object(mut memory) = value else {
        return value;
    };

    let mut compact = serde_json::Map::new();
    move_field(&mut memory, &mut compact, "id");
    move_field(&mut memory, &mut compact, "content");
    move_non_null_field(&mut memory, &mut compact, "summary");
    move_field(&mut memory, &mut compact, "type");
    move_field(&mut memory, &mut compact, "importance");
    move_field(&mut memory, &mut compact, "confidence");
    move_field(&mut memory, &mut compact, "tags");
    move_field(&mut memory, &mut compact, "score");

    if include_components {
        move_field(&mut memory, &mut compact, "components");
    }

    if state_debug {
        move_non_null_field(&mut memory, &mut compact, "state");
        move_non_null_field(&mut memory, &mut compact, "t_valid");
        move_non_null_field(&mut memory, &mut compact, "t_invalid");
        move_true_field(&mut memory, &mut compact, "archived");
        move_true_field(&mut memory, &mut compact, "protected");
        move_non_empty_array_field(&mut memory, &mut compact, "state_replacements");
    } else if let Some(state) = memory.remove("state").filter(|state| state != "active") {
        compact.insert("state".to_string(), state);
    }

    Value::Object(compact)
}

fn move_field(
    from: &mut serde_json::Map<String, Value>,
    to: &mut serde_json::Map<String, Value>,
    field: &str,
) {
    if let Some(value) = from.remove(field) {
        to.insert(field.to_string(), value);
    }
}

fn move_non_null_field(
    from: &mut serde_json::Map<String, Value>,
    to: &mut serde_json::Map<String, Value>,
    field: &str,
) {
    if let Some(value) = from.remove(field).filter(|value| !value.is_null()) {
        to.insert(field.to_string(), value);
    }
}

fn move_true_field(
    from: &mut serde_json::Map<String, Value>,
    to: &mut serde_json::Map<String, Value>,
    field: &str,
) {
    if let Some(value) = from
        .remove(field)
        .filter(|value| value.as_bool() == Some(true))
    {
        to.insert(field.to_string(), value);
    }
}

fn move_non_empty_array_field(
    from: &mut serde_json::Map<String, Value>,
    to: &mut serde_json::Map<String, Value>,
    field: &str,
) {
    if let Some(value) = from
        .remove(field)
        .filter(|value| value.as_array().is_some_and(|array| !array.is_empty()))
    {
        to.insert(field.to_string(), value);
    }
}

const SERVER_INSTRUCTIONS: &str = "agskmem is the local, user-global memory store backed by SQLite. Use startup_recall at the start of a coding session. Use recall_memory before answering explicit memory questions and before decisions that may depend on prior user preferences, project decisions, corrections, or patterns. Treat tags as hard filters and context_tags as soft boosts. Store only stable user corrections, finalized decisions, and user-articulated patterns; do not store transient session summaries or speculative notes. Never send embedding vectors: agskmem generates embeddings server-side. Use associate_memories for durable causal/preference/provenance links. Use update_memory for corrections to an existing fact instead of duplicating it. Bulk delete by tag requires delete_memory dry_run first and then the returned confirmation_token.";

fn tool_specs() -> Vec<Value> {
    [
        (
            "store_memory",
            "Store one memory with visible top-level content; embeddings are generated server-side.",
        ),
        (
            "update_memory",
            "Patch an existing memory and re-embed when content changes.",
        ),
        (
            "delete_memory",
            "Delete one memory, or bulk delete by tag using dry-run confirmation.",
        ),
        (
            "associate_memories",
            "Create or update an authorable relationship edge.",
        ),
        (
            "recall_memory",
            "Fetch by id, enumerate tags, or ranked search with optional graph expansion.",
        ),
        (
            "get_related_memories",
            "Graph-only personalized PageRank neighborhood.",
        ),
        ("graph_snapshot", "Read graph cache snapshot."),
        (
            "graph_neighbors",
            "Read direct graph neighbors for a memory.",
        ),
        ("graph_stats", "Read graph cache statistics."),
        (
            "trace_recall",
            "Return recall results with score components.",
        ),
        (
            "check_database_health",
            "Database, FTS, embedding, and graph health.",
        ),
        ("analyze_memories", "Aggregate memory counts and top tags."),
        (
            "relation_types",
            "List authorable and system relation kinds.",
        ),
        ("memory_types", "List accepted memory types."),
        (
            "startup_recall",
            "Cheap recent-and-important startup slice.",
        ),
        (
            "consolidate",
            "Run decay, forget, creative, cluster, or all.",
        ),
        (
            "consolidate_status",
            "Read consolidation scheduler/history status.",
        ),
        (
            "enrichment_status",
            "Read enrichment queue and worker status.",
        ),
        (
            "enrichment_reprocess",
            "Queue existing memories for explicit enrichment reprocessing.",
        ),
        ("repair_index", "Rebuild FTS and the in-memory CSR graph."),
        ("reembed", "Rebuild embeddings for all or a filter."),
        (
            "export_backup",
            "Create SQLite online backup plus manifest.",
        ),
        ("import_backup", "Restore from SQLite online backup."),
    ]
    .into_iter()
    .map(|(name, description)| {
        json!({
            "name": name,
            "description": description,
            "inputSchema": tool_schema(name)
        })
    })
    .collect()
}

fn store_memory_schema() -> Value {
    json!({
        "type": "object",
        "description": "Store one memory with top-level fields so MCP tool logs show the content being written.",
        "required": ["content"],
        "properties": {
            "content": {"type": "string", "minLength": 1, "description": "Memory text to store."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Hard-filter tags; normalized lower-case and deduplicated."},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "metadata": {"type": "object"},
            "type": {"type": "string", "enum": ["Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context", "Statement"]},
            "summary": {"type": "string"},
            "source": {"type": "string"},
            "timestamp": {"type": "string", "description": "RFC3339 or epoch seconds."},
            "t_valid": {"type": "string", "description": "RFC3339 or epoch seconds."},
            "t_invalid": {"type": "string", "description": "RFC3339 or epoch seconds."},
            "id": {"type": "string", "description": "Optional caller-provided memory id."}
        },
        "additionalProperties": false
    })
}

fn tool_schema(name: &str) -> Value {
    match name {
        "store_memory" => store_memory_schema(),
        "recall_memory" | "trace_recall" => json!({
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "query": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Hard include filter: result must have all tags."},
                "context": {"type": "string"},
                "language": {"type": "string"},
                "active_path": {"type": "string"},
                "context_types": {"type": "array", "items": {"type": "string"}},
                "tag_mode": {"type": "string", "enum": ["any", "all"]},
                "tag_match": {"type": "string", "enum": ["exact", "prefix"]},
                "time_query": {"type": "string"},
                "sort": {"type": "string", "enum": ["score", "time_desc", "time_asc", "updated_desc", "updated_asc"]},
                "order_by": {"type": "string", "enum": ["score", "time_desc", "time_asc", "updated_desc", "updated_asc"]},
                "relation_limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "expansion_limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "expand_min_importance": {"type": "number", "minimum": 0, "maximum": 1},
                "expand_min_strength": {"type": "number", "minimum": 0, "maximum": 1},
                "auto_decompose": {"type": "boolean"},
                "min_score": {"type": "number"},
                "adaptive_floor": {"type": "boolean"},
                "exclude_tags": {"type": "array", "items": {"type": "string"}},
                "context_tags": {"type": "array", "items": {"type": "string"}, "description": "Soft scoring boosts, not hard filters."},
                "priority_ids": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "offset": {"type": "integer", "minimum": 0},
                "cursor": {"type": "integer", "minimum": 0},
                "current_only": {"type": "boolean", "default": true},
                "state_debug": {"type": "boolean"},
                "expand_relations": {"type": "boolean"},
                "expand_entities": {"type": "boolean"},
                "expand_respect_tags": {"type": "boolean"},
                "as_of": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"}
            },
            "additionalProperties": false
        }),
        "update_memory" => json!({
            "type": "object",
            "required": ["memory_id"],
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string"},
                "summary": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string", "enum": ["Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context", "Statement"]},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                "reliability": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"},
                "source": {"type": "string"},
                "t_valid": {"type": "string"},
                "t_invalid": {"type": "string"},
                "archived": {"type": "boolean"},
                "protected": {"type": "boolean"}
            },
            "additionalProperties": false
        }),
        "delete_memory" => json!({
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "dry_run": {"type": "boolean", "description": "Required first for bulk tag deletion."},
                "confirmation_token": {"type": "string", "description": "Token returned by dry_run for bulk tag deletion."}
            },
            "additionalProperties": false
        }),
        "associate_memories" => json!({
            "type": "object",
            "required": ["memory1_id", "memory2_id", "type"],
            "properties": {
                "memory1_id": {"type": "string"},
                "memory2_id": {"type": "string"},
                "type": {"type": "string", "enum": ["RELATES_TO", "LEADS_TO", "OCCURRED_BEFORE", "PREFERS_OVER", "EXEMPLIFIES", "CONTRADICTS", "REINFORCES", "INVALIDATED_BY", "EVOLVED_INTO", "DERIVED_FROM", "PART_OF"]},
                "strength": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            },
            "additionalProperties": false
        }),
        "get_related_memories" | "graph_neighbors" => json!({
            "type": "object",
            "required": ["memory_id"],
            "properties": {
                "memory_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200}
            },
            "additionalProperties": false
        }),
        "startup_recall" => json!({
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}},
            "additionalProperties": false
        }),
        "consolidate" => json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["decay", "cluster", "creative", "forget", "all"]},
                "dry_run": {"type": "boolean"}
            },
            "additionalProperties": false
        }),
        "enrichment_reprocess" => json!({
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}},
                "forced": {"type": "boolean"}
            },
            "additionalProperties": false
        }),
        "reembed" => json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "memory_id": {"type": "string"}
            },
            "additionalProperties": false
        }),
        "export_backup" | "import_backup" => json!({
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": false
        }),
        _ => json!({"type": "object", "additionalProperties": false}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_memory_schema_keeps_content_visible() {
        let schema = tool_schema("store_memory");

        assert_eq!(schema["required"][0], "content");
        assert_eq!(schema["properties"]["content"]["type"], "string");
        assert_eq!(schema["properties"]["content"]["minLength"], 1);
        assert_eq!(schema["properties"]["memories"], Value::Null);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn recall_tool_text_uses_human_display() {
        let result = json!({
            "prompt": "recall_memory query=\"visible\"; limit=1",
            "results_text": "1. Visible memory\n   id=mem-1 type=Context"
        });

        let text = render_tool_result("recall_memory", &result).expect("render recall text");

        assert_eq!(
            text,
            "recall_memory query=\"visible\"; limit=1\n\n1. Visible memory\n   id=mem-1 type=Context"
        );
    }

    #[test]
    fn store_memory_deserialization_errors_include_nested_path() {
        let err = parse_tool_args::<StoreMemoryArgs>(
            "store_memory",
            json!({"memories": [{"tags": ["visible"], "type": "Preference"}]}),
        )
        .expect_err("batch item without content must fail");
        let message = err.to_string();

        assert!(message.contains("store_memory.memories[0]"), "{message}");
        assert!(message.contains("missing field `content`"), "{message}");
    }
}
