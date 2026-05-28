CREATE TABLE IF NOT EXISTS tag_prefix(
  memory_id TEXT NOT NULL,
  prefix TEXT NOT NULL COLLATE NOCASE,
  PRIMARY KEY(memory_id, prefix),
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enrichment_job(
  memory_id TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  forced INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_retry_at INTEGER NOT NULL,
  last_error TEXT,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classification_cache(
  content_hash TEXT PRIMARY KEY,
  type INTEGER NOT NULL,
  confidence REAL NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS tag_prefix_prefix_memory_idx ON tag_prefix(prefix, memory_id);
