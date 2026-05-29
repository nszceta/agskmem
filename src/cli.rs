use crate::{AgskMem, Config, design_types::*, mcp};
use anyhow::{Context, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    sync::Arc,
};

#[derive(Debug, Parser)]
#[command(
    name = "agskmem",
    version,
    about = "SQLite-backed local memory MCP server"
)]
pub struct Args {
    #[arg(long, env = "AGSKMEM_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, env = "AGSKMEM_DB")]
    pub db: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Serve {
        #[arg(long)]
        http: bool,
        #[arg(long, default_value = "127.0.0.1:7337")]
        addr: String,
    },
    Install {
        client: Client,
        #[arg(long)]
        force: bool,
    },
    Export {
        path: PathBuf,
    },
    Import {
        path: PathBuf,
    },
    ImportJsonl {
        path: PathBuf,
    },
    ImportAutomem {
        #[arg(long)]
        falkordb: PathBuf,
        #[arg(long)]
        qdrant: Option<PathBuf>,
        #[arg(long)]
        allow_extra_kinds: bool,
    },
    Reembed,
    Repair,
    Tool {
        name: String,
        #[arg(default_value = "{}")]
        args: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Client {
    ClaudeCode,
    Cursor,
    Codex,
    Omp,
    Pi,
    Generic,
}

pub async fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = Config::load(args.config, args.db)?;
    let app = Arc::new(AgskMem::open(cfg)?);
    match args.command.unwrap_or(Command::Serve {
        http: false,
        addr: "127.0.0.1:7337".to_string(),
    }) {
        Command::Serve { http, addr } => {
            if http {
                bail!(
                    "HTTP transport is not enabled in this build; use stdio or place behind a stdio MCP client (requested addr {addr})"
                );
            }
            mcp::serve_stdio(app).await
        }
        Command::Install { client, force } => install(client, force),
        Command::Export { path } => print_json(app.export_backup(&path)?),
        Command::Import { path } => print_json(app.import_backup(&path)?),
        Command::ImportJsonl { path } => print_json(import_jsonl(&app, path)?),
        Command::ImportAutomem {
            falkordb,
            qdrant,
            allow_extra_kinds,
        } => print_json(import_automem(&app, falkordb, qdrant, allow_extra_kinds)?),
        Command::Reembed => print_json(app.reembed(ReembedArgs::default())?),
        Command::Repair => {
            app.repair_index()?;
            print_json(json!({"repaired": true}))
        }
        Command::Tool { name, args } => {
            let value: Value = serde_json::from_str(&args).context("parsing tool args JSON")?;
            print_json(mcp::call_tool(&app, &name, value)?)
        }
    }
}

fn print_json(value: Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn install(client: Client, force: bool) -> anyhow::Result<()> {
    match client {
        Client::Generic => print_json(
            json!({"mcpServers":{"agskmem":{"command":"agskmem","args":["serve"]}},"policy":"Recall at session start; store stable corrections, decisions, and named patterns; never send embedding vectors."}),
        ),
        Client::ClaudeCode => {
            let home = dirs::home_dir().context("home directory not found")?;
            let settings = home.join(".claude/settings.json");
            let policy = home.join(".claude/agents/memory-policy.md");
            merge_json_server(&settings, force)?;
            write_policy(&policy)?;
            print_json(json!({"updated": [settings, policy]}))
        }
        Client::Cursor => {
            let home = dirs::home_dir().context("home directory not found")?;
            let mcp_path = home.join(".cursor/mcp.json");
            merge_json_server(&mcp_path, force)?;
            let rules = std::env::current_dir()?.join("rules/agskmem.mdc");
            write_policy(&rules)?;
            print_json(json!({"updated": [mcp_path, rules]}))
        }
        Client::Codex => {
            let home = dirs::home_dir().context("home directory not found")?;
            let path = home.join(".codex/config.toml");
            merge_toml_server(&path, force)?;
            print_json(json!({"updated": [path]}))
        }
        Client::Omp | Client::Pi => {
            let home = dirs::home_dir().context("home directory not found")?;
            let path = home.join(match client {
                Client::Omp => ".omp/agent/mcp.json",
                Client::Pi => ".pi/agent/mcp.json",
                _ => unreachable!(),
            });
            merge_json_server(&path, force)?;
            print_json(json!({"updated": [path]}))
        }
    }
}

fn merge_json_server(path: &PathBuf, force: bool) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root: Value = if path.exists() {
        serde_json::from_str(&fs::read_to_string(path)?)?
    } else {
        json!({})
    };
    let servers = root
        .as_object_mut()
        .context("config root must be object")?
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    let map = servers
        .as_object_mut()
        .context("mcpServers must be object")?;
    let new_entry =
        json!({"type":"stdio","command":"agskmem","args":["serve"],"timeout":30000,"enabled":true});
    if let Some(existing) = map.get("agskmem")
        && existing != &new_entry
        && !force
    {
        bail!(
            "conflicting agskmem entry in {}; rerun with --force",
            path.display()
        );
    }
    map.insert("agskmem".to_string(), new_entry);
    fs::write(path, serde_json::to_vec_pretty(&root)?)?;
    Ok(())
}

fn merge_toml_server(path: &PathBuf, force: bool) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = if path.exists() {
        fs::read_to_string(path)?
    } else {
        String::new()
    };
    if text.contains("[mcp_servers.agskmem]") && !force {
        bail!(
            "conflicting agskmem entry in {}; rerun with --force",
            path.display()
        );
    }
    let mut out = text;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("\n[mcp_servers.agskmem]\ncommand = \"agskmem\"\nargs = [\"serve\"]\n");
    fs::write(path, out)?;
    Ok(())
}

const AGSKMEM_POLICY: &str = "## agskmem memory policy\n\n- agskmem is the active local user-global memory store. Prefer agskmem MCP tools over legacy automem tools.\n- At session start, use startup_recall for recent/important memory context when available.\n- During long sessions, revisit memory context roughly every five user-assistant rounds with recall_memory or startup_recall as appropriate.\n- Before answering explicit memory questions or making decisions that may depend on prior user preferences, corrections, project decisions, or patterns, use recall_memory.\n- Store only stable user corrections, finalized decisions, and user-articulated patterns; do not store transient session summaries or speculative notes.\n- Use update_memory when correcting an existing fact instead of duplicating it.\n- Use associate_memories for durable causal, preference, provenance, invalidation, and example links.\n- Tags are hard filters; context_tags are soft boosts. Project-specific entries should include the canonical project slug tag.\n- Never send embedding vectors; agskmem generates embeddings server-side.\n";

fn write_policy(path: &PathBuf) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, AGSKMEM_POLICY)?;
    Ok(())
}

fn import_jsonl(app: &AgskMem, path: PathBuf) -> anyhow::Result<Value> {
    let file = fs::File::open(&path)?;
    let reader = io::BufReader::new(file);
    let mut imported = 0;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        let mut args: StoreMemoryArgs = serde_json::from_value(value)?;
        if args.content.is_none() && args.memories.is_empty() {
            continue;
        }
        imported += app.store_memory(std::mem::take(&mut args))?.ids.len();
    }
    Ok(json!({"imported": imported, "path": path}))
}

fn import_automem(
    app: &AgskMem,
    falkordb: PathBuf,
    qdrant: Option<PathBuf>,
    allow_extra_kinds: bool,
) -> anyhow::Result<Value> {
    let text = fs::read_to_string(&falkordb)?;
    let mut created = 0;
    let mut updated = 0;
    let mut edges = 0;
    let values: Vec<Value> = if text.trim_start().starts_with('[') {
        serde_json::from_str(&text)?
    } else {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?
    };
    let mut source_id_map = std::collections::HashMap::new();
    for value in &values {
        if value.get("src").is_some() && value.get("dst").is_some() {
            continue;
        }
        let content = value
            .get("content")
            .or_else(|| value.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if content.is_empty() {
            continue;
        }
        let Some(source_id) = value.get("id").and_then(Value::as_str) else {
            continue;
        };
        let id = source_id.to_string();
        source_id_map.insert(source_id.to_string(), id.clone());
        let tags = json_string_array(value.get("tags"));
        let memory_type = if json_string_array(value.get("labels"))
            .iter()
            .any(|label| label == "Pattern")
        {
            Some("Pattern".to_string())
        } else {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        };
        let summary = value
            .get("summary")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let timestamp = value
            .get("timestamp")
            .or_else(|| value.get("created_at"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let t_valid = value
            .get("t_valid")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let t_invalid = value
            .get("t_invalid")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let metadata = value.get("metadata").cloned();
        if memory_exists(app, &id)? {
            app.update_memory(UpdateMemoryArgs {
                memory_id: id,
                content: Some(content.to_string()),
                summary,
                tags: Some(tags),
                memory_type,
                importance: value.get("importance").and_then(Value::as_f64),
                confidence: value.get("confidence").and_then(Value::as_f64),
                relevance: value
                    .get("relevance")
                    .or_else(|| value.get("relevance_score"))
                    .and_then(Value::as_f64),
                metadata,
                t_valid,
                t_invalid,
                ..Default::default()
            })?;
            updated += 1;
        } else {
            app.store_memory(StoreMemoryArgs {
                id: Some(id),
                content: Some(content.to_string()),
                tags,
                importance: value.get("importance").and_then(Value::as_f64),
                confidence: value.get("confidence").and_then(Value::as_f64),
                metadata,
                memory_type,
                summary,
                timestamp,
                t_valid,
                t_invalid,
                source: Some("falkordb".to_string()),
                ..Default::default()
            })?;
            created += 1;
        }
    }
    for value in &values {
        let Some(src) = value.get("src").and_then(Value::as_str) else {
            continue;
        };
        let Some(dst) = value.get("dst").and_then(Value::as_str) else {
            continue;
        };
        let kind = value
            .get("kind")
            .or_else(|| value.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("RELATES_TO");
        if !allow_extra_kinds && crate::model::RelationKind::parse(kind).is_err() {
            bail!("unsupported relation kind {kind}");
        }
        let src = source_id_map
            .get(src)
            .cloned()
            .unwrap_or_else(|| src.to_string());
        let dst = source_id_map
            .get(dst)
            .cloned()
            .unwrap_or_else(|| dst.to_string());
        app.import_relation(AssociateArgs {
            memory1_id: src,
            memory2_id: dst,
            relation_type: if crate::model::RelationKind::parse(kind).is_ok() {
                kind.to_string()
            } else {
                "RELATES_TO".to_string()
            },
            strength: value.get("strength").and_then(Value::as_f64),
            confidence: value.get("confidence").and_then(Value::as_f64),
            metadata: value.get("metadata").cloned(),
        })?;
        edges += 1;
    }
    if edges > 0 {
        app.repair_index()?;
    }
    if let Some(path) = qdrant {
        let _ = fs::File::open(path)?;
    }
    Ok(
        json!({"created": created, "updated": updated, "edges": edges, "vectors": "local embeddings are generated inside agskmem; qdrant import is optional and reembed remains available"}),
    )
}

fn memory_exists(app: &AgskMem, id: &str) -> anyhow::Result<bool> {
    let result = app.recall_memory(RecallArgs {
        memory_id: Some(id.to_string()),
        current_only: Some(false),
        ..Default::default()
    })?;
    Ok(result
        .get("results")
        .and_then(Value::as_array)
        .is_some_and(|results| !results.is_empty()))
}

fn json_string_array(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        Some(Value::String(raw)) => raw
            .trim_matches(['[', ']'])
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

pub fn write_stdout_line(value: &Value) -> anyhow::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{}", serde_json::to_string(value)?)?;
    Ok(())
}
