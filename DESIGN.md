# agskmem design

agskmem is a single Rust MCP server backed by one user-global SQLite database. It replaces the old multi-process memory stack with one binary, one database file, and no required daemon, container, npm bridge, graph server, or vector server.

## Runtime shape

- `agskmem serve` runs a stdio MCP JSON-RPC server.
- `agskmem install <client>` writes idempotent client configuration for supported agents.
- `agskmem export` and `agskmem import` use SQLite online backup.
- `agskmem reembed` and `agskmem repair` rebuild derived indexes from source tables.
- The default database path is `${AGSKMEM_DB:-${XDG_DATA_HOME:-~/.local/share}/agskmem/agskmem.sqlite3}`.
- The default config path is `${XDG_CONFIG_HOME:-~/.config}/agskmem/config.toml`.

SQLite is the source of truth. FTS rows, embeddings, statements, and the in-memory graph are all rebuildable derived state.

## Storage

The initial migration creates:

- `memory`: canonical memory records, UUID string ids, timestamps as epoch seconds, JSON metadata as text.
- `memory_fts`: external-content FTS5 index over memory content and summaries.
- `tag`: normalized lower-case tags with exact enumeration indexes.
- `entity` and `memory_entity`: deterministic entity mentions for entity-driven recall.
- `edge`: typed relationship graph edges.
- `statement`: extracted factual statement rows with provenance to a source memory.
- `statement_fts`: external-content FTS5 index over statements.
- `embedding`: normalized little-endian `f32` vector blobs generated inside agskmem.
- `embedding_job`: retry queue for embedders that cannot encode immediately.
- `schema_history`: forward-only migration history with SHA-256 migration hashes.

Writers use `BEGIN IMMEDIATE`. Connections enable WAL, foreign keys, busy timeout, in-memory temp store, and mmap. Every write tool commits or rolls back as one transaction.

## Embeddings

Coding agents never pass raw embeddings. agskmem generates vectors server-side. The v1 implementation ships a deterministic local hash embedder so lexical and vector recall work offline without provider credentials. The storage format is already compatible with replacement provider implementations: a model name, dimension count, norm, and normalized `f32` blob per memory.

## Recall

`recall_memory` has three modes:

1. ID fetch.
2. Exact tag enumeration with `offset`/`limit` pagination.
3. Ranked search.

Ranked search combines independent candidate sources:

- FTS5 keyword matches.
- Exact phrase matches.
- Entity matches.
- Recent/high-relevance fallback rows.
- Caller `priority_ids`.

Hard filters are applied before scoring: tags, excluded tags, active/current-state semantics, validity windows, and time ranges. Score components are computed from the candidate content and state, not from the retrieval path: vector cosine, lexical keyword overlap, PPR, tag overlap, exact phrase, importance, recency, confidence, reliability, and a small context bonus.

Broad result sets are reranked with MMR to reduce near-duplicate output. Returned memories are touched in one batched update of `last_accessed`.

## Current-state semantics

The default is `current_only=true`:

- Archived memories are hidden.
- Future `t_valid` and expired `t_invalid` memories are hidden.
- Memories with active outgoing `INVALIDATED_BY` or `EVOLVED_INTO` replacements are hidden.
- `CONTRADICTS` does not suppress by itself.
- `current_only=false` returns rows with explicit `state` labels.

## Graph

Edges are stored in SQLite and loaded into an in-memory CSR cache. Readers use an `ArcSwap<Graph>` pointer, so recall reads do not take a graph lock. `repair_index` rebuilds the CSR from `edge` ordered by source and atomically publishes the new graph.

Graph expansion uses a deterministic single-threaded forward-push Personalized PageRank implementation. Edge mass is normalized per source, which naturally penalizes hubs because high-degree nodes split their mass across more outgoing edges.

Graph tools expose:

- direct neighbors,
- graph stats,
- a bounded snapshot,
- graph-only related-memory lookup,
- PPR components through `trace_recall`.

## Enrichment and consolidation

Store/update performs cheap deterministic enrichment inline:

- canonical-ish full-name and path/entity mentions,
- statement extraction from sentence boundaries,
- statement provenance to source memory.

Administrative consolidation supports explicit dry-run or mutating modes:

- `decay`: updates relevance from age, access, degree, importance, and confidence.
- `forget`: archives or deletes low-relevance unprotected rows outside the grace window.
- `creative`: creates conservative `DISCOVERED` edges among highly relevant memories.
- `cluster`: reports current similarity cluster inputs.

## MCP tool surface

The server exposes the AutoMem-compatible tool names:

- Writes: `store_memory`, `update_memory`, `delete_memory`, `associate_memories`.
- Reads: `recall_memory`, `get_related_memories`, `graph_snapshot`, `graph_neighbors`, `graph_stats`, `trace_recall`.
- Admin/introspection: `check_database_health`, `analyze_memories`, `relation_types`, `memory_types`, `startup_recall`, `consolidate`, `export_backup`, `import_backup`, `repair_index`, `reembed`.

Bulk tag deletes require a dry-run token before mutation.

## Security and privacy

agskmem keeps all data local. No telemetry leaves the machine. Provider keys, when provider implementations are added, are read from config/env and must never be logged. The installer merges client configs and refuses conflicting `agskmem` entries unless `--force` is supplied.
