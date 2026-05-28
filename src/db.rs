use crate::{embed, model::now_epoch};
use anyhow::{Context, bail};
use rusqlite::{Connection, OpenFlags, functions::FunctionFlags};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

pub struct Database {
    path: PathBuf,
    writer: Mutex<Connection>,
}

impl Database {
    pub fn open(path: PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening sqlite {}", path.display()))?;
        configure(&conn)?;
        register_functions(&conn)?;
        let db = Self {
            path,
            writer: Mutex::new(conn),
        };
        db.migrate()?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn writer(&self) -> anyhow::Result<std::sync::MutexGuard<'_, Connection>> {
        self.writer
            .lock()
            .map_err(|_| anyhow::anyhow!("writer connection mutex poisoned"))
    }

    pub fn read_connection(&self) -> anyhow::Result<Connection> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(&self.path, flags)
            .with_context(|| format!("opening readonly sqlite {}", self.path.display()))?;
        configure_readonly(&conn)?;
        register_functions(&conn)?;
        Ok(conn)
    }

    pub fn migrate(&self) -> anyhow::Result<()> {
        let conn = self.writer()?;
        conn.execute_batch("BEGIN IMMEDIATE;")?;
        let result = (|| -> anyhow::Result<()> {
            apply_migration(
                &conn,
                1,
                "0001_init.sql",
                include_str!("../migrations/0001_init.sql"),
            )?;
            apply_migration(
                &conn,
                2,
                "0002_governance_enrichment.sql",
                include_str!("../migrations/0002_governance_enrichment.sql"),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT;")?;
                Ok(())
            }
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK;");
                Err(err)
            }
        }
    }

    pub fn backup_to(&self, dest: &Path) -> anyhow::Result<()> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let src = self.read_connection()?;
        let mut dst = Connection::open(dest)?;
        let backup = rusqlite::backup::Backup::new(&src, &mut dst)?;
        backup.step(-1)?;
        Ok(())
    }

    pub fn restore_from(&self, src_path: &Path) -> anyhow::Result<()> {
        let src = Connection::open(src_path)?;
        let mut writer = self.writer()?;
        let backup = rusqlite::backup::Backup::new(&src, &mut writer)?;
        backup.step(-1)?;
        Ok(())
    }
}

fn apply_migration(conn: &Connection, version: i64, name: &str, sql: &str) -> anyhow::Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_history(version INTEGER PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL, applied_at INTEGER NOT NULL);")?;
    let existing: Option<String> = conn
        .query_row(
            "SELECT sha256 FROM schema_history WHERE version = ?",
            [version],
            |row| row.get(0),
        )
        .optional()?;
    let hash = hex::encode(Sha256::digest(sql.as_bytes()));
    if let Some(existing) = existing {
        if existing != hash {
            bail!("migration {version} hash mismatch: database has {existing}, binary has {hash}");
        }
        return Ok(());
    }
    conn.execute_batch(sql)
        .with_context(|| format!("applying migration {name}"))?;
    conn.execute(
        "INSERT INTO schema_history(version, name, sha256, applied_at) VALUES (?, ?, ?, ?)",
        rusqlite::params![version, name, hash, now_epoch()],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        [version.to_string()],
    )?;
    Ok(())
}

pub fn configure(conn: &Connection) -> anyhow::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "busy_timeout", 5000_i64)?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "mmap_size", 268_435_456_i64)?;
    Ok(())
}

pub fn configure_readonly(conn: &Connection) -> anyhow::Result<()> {
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "busy_timeout", 5000_i64)?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    Ok(())
}

pub fn register_functions(conn: &Connection) -> anyhow::Result<()> {
    conn.create_scalar_function(
        "now",
        0,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |_ctx| Ok(now_epoch()),
    )?;
    conn.create_scalar_function(
        "cosine",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let a: Vec<u8> = ctx.get(0)?;
            let b: Vec<u8> = ctx.get(1)?;
            embed::cosine_from_blobs(&a, &b)
                .map_err(|e| rusqlite::Error::UserFunctionError(e.into()))
        },
    )?;
    Ok(())
}

trait OptionalRow<T> {
    fn optional(self) -> rusqlite::Result<Option<T>>;
}

impl<T> OptionalRow<T> for rusqlite::Result<T> {
    fn optional(self) -> rusqlite::Result<Option<T>> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
