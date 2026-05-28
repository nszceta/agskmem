CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS schema_history(version INTEGER PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL, applied_at INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS memory(
  rowid INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  summary TEXT,
  type INTEGER NOT NULL,
  importance REAL NOT NULL,
  confidence REAL NOT NULL,
  relevance REAL NOT NULL DEFAULT 0.5,
  reliability REAL NOT NULL DEFAULT 0.7,
  metadata TEXT NOT NULL DEFAULT '{}',
  source TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_accessed INTEGER,
  t_valid INTEGER,
  t_invalid INTEGER,
  archived INTEGER NOT NULL DEFAULT 0,
  protected INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  content, summary,
  content='memory', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, content, summary) VALUES (new.rowid, new.content, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, summary) VALUES('delete', old.rowid, old.content, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE OF content, summary ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, summary) VALUES('delete', old.rowid, old.content, old.summary);
  INSERT INTO memory_fts(rowid, content, summary) VALUES (new.rowid, new.content, new.summary);
END;

CREATE TABLE IF NOT EXISTS tag(
  memory_id TEXT NOT NULL,
  tag TEXT NOT NULL COLLATE NOCASE,
  PRIMARY KEY(memory_id, tag),
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity(
  id INTEGER PRIMARY KEY,
  kind INTEGER NOT NULL,
  slug TEXT NOT NULL,
  label TEXT NOT NULL,
  quality REAL NOT NULL DEFAULT 1.0,
  UNIQUE(kind, slug)
);
CREATE TABLE IF NOT EXISTS memory_entity(
  memory_id TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  role INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(memory_id, entity_id, role),
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE,
  FOREIGN KEY(entity_id) REFERENCES entity(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edge(
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  kind INTEGER NOT NULL,
  strength REAL NOT NULL DEFAULT 0.5,
  confidence REAL NOT NULL DEFAULT 0.5,
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(src, dst, kind),
  FOREIGN KEY(src) REFERENCES memory(id) ON DELETE CASCADE,
  FOREIGN KEY(dst) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS statement(
  rowid INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  memory_id TEXT NOT NULL,
  content TEXT NOT NULL,
  subject TEXT,
  predicate TEXT,
  object TEXT,
  confidence REAL NOT NULL,
  reliability REAL NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);
CREATE VIRTUAL TABLE IF NOT EXISTS statement_fts USING fts5(
  content, content='statement', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS statement_ai AFTER INSERT ON statement BEGIN
  INSERT INTO statement_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS statement_ad AFTER DELETE ON statement BEGIN
  INSERT INTO statement_fts(statement_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS statement_au AFTER UPDATE OF content ON statement BEGIN
  INSERT INTO statement_fts(statement_fts, rowid, content) VALUES('delete', old.rowid, old.content);
  INSERT INTO statement_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS embedding(
  memory_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  norm REAL NOT NULL,
  vec BLOB NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embedding_job(
  memory_id TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_retry_at INTEGER NOT NULL,
  last_error TEXT,
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recall_metric(
  id INTEGER PRIMARY KEY,
  tool TEXT NOT NULL,
  dur_ms INTEGER NOT NULL,
  candidates INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS tag_tag_memory_idx ON tag(tag, memory_id);
CREATE INDEX IF NOT EXISTS edge_src_kind_idx ON edge(src, kind);
CREATE INDEX IF NOT EXISTS edge_dst_kind_idx ON edge(dst, kind);
CREATE INDEX IF NOT EXISTS memory_type_created_idx ON memory(type, created_at DESC);
CREATE INDEX IF NOT EXISTS memory_archived_relevance_idx ON memory(archived, relevance DESC);
CREATE INDEX IF NOT EXISTS memory_updated_idx ON memory(updated_at DESC);
CREATE INDEX IF NOT EXISTS memory_t_invalid_idx ON memory(t_invalid) WHERE t_invalid IS NOT NULL;
CREATE INDEX IF NOT EXISTS memory_entity_entity_idx ON memory_entity(entity_id);

INSERT OR IGNORE INTO meta(key, value) VALUES
 ('schema_version', '1'),
 ('embedding_model', 'local-hash-v1'),
 ('embedding_dims', '128'),
 ('created_at', CAST(strftime('%s','now') AS TEXT)),
 ('app_version', '0.1.0');
