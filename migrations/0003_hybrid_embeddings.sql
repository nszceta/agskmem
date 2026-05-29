CREATE TABLE IF NOT EXISTS embedding_sparse(
  memory_id TEXT NOT NULL,
  token_id INTEGER NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY(memory_id, token_id),
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embedding_colbert(
  memory_id TEXT NOT NULL,
  token_index INTEGER NOT NULL,
  vec BLOB NOT NULL,
  PRIMARY KEY(memory_id, token_index),
  FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS embedding_sparse_token_memory_idx ON embedding_sparse(token_id, memory_id);
