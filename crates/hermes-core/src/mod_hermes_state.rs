//! SQLite State Store for Hermes Agent — native Rust port of `hermes_state.py`.
//!
//! Provides persistent session storage with FTS5 full-text search, replacing
//! the per-session JSONL file approach. Stores session metadata, full message
//! history, and model configuration for CLI and gateway sessions.
//!
//! Key design decisions (mirrors the Python original):
//! - WAL mode for concurrent readers + one writer (gateway multi-platform)
//! - FTS5 virtual table for fast text search across all session messages
//! - Compression-triggered session splitting via parent_session_id chains
//! - Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
//!
//! Threading: the Python class held a single connection guarded by a
//! `threading.Lock`. Here we wrap the `rusqlite::Connection` in a `Mutex`,
//! giving the same single-writer / serialized-access semantics.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use log::{debug, info, warn};
use rusqlite::types::ValueRef;
use rusqlite::{params_from_iter, Connection, OpenFlags, ToSql};
use serde_json::{json, Map, Value};

// Cross-reference: the canonical hermes home directory lives in
// `crate::mod_hermes_constants`. Sanitisation of replayed conversation content
// lives in `crate::memory::sanitize_context` (the module wired into lib.rs;
// `ag_memory_manager` carries an identical copy).
use crate::memory::sanitize_context;
use crate::mod_hermes_constants::get_hermes_home;

/// Current schema version. Matches `SCHEMA_VERSION` in the Python source.
pub const SCHEMA_VERSION: i64 = 11;

/// Maximum length for session titles (chars).
pub const MAX_TITLE_LENGTH: usize = 100;

// ── Write-contention tuning (see Python docstring) ──
const WRITE_MAX_RETRIES: u32 = 15;
const WRITE_RETRY_MIN_MS: u64 = 20;
const WRITE_RETRY_MAX_MS: u64 = 150;
const CHECKPOINT_EVERY_N_WRITES: u64 = 50;

/// Sentinel prefix used to distinguish JSON-encoded structured content
/// (multimodal messages) from plain string content. The NUL byte cannot
/// appear in normal text so this never collides with real user content.
const CONTENT_JSON_PREFIX: &str = "\u{0}json:";

pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"#;

pub const FTS_SQL: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"#;

pub const FTS_TRIGRAM_SQL: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"#;

/// Errors surfaced by [`SessionDB`].
#[derive(Debug)]
pub enum StateError {
    /// Underlying SQLite error.
    Sqlite(rusqlite::Error),
    /// Validation failure (e.g. title too long, title already in use).
    Value(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::Sqlite(e) => write!(f, "sqlite error: {e}"),
            StateError::Value(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StateError {}

impl From<rusqlite::Error> for StateError {
    fn from(e: rusqlite::Error) -> Self {
        StateError::Sqlite(e)
    }
}

pub type Result<T> = std::result::Result<T, StateError>;

/// Wall-clock seconds as a float, matching Python's `time.time()`.
fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Default path to the state database: `<hermes_home>/state.db`.
pub fn default_db_path() -> PathBuf {
    get_hermes_home().join("state.db")
}

/// Returns true if `err` looks like a SQLite "database is locked"/"busy" error.
fn is_lock_error(err: &rusqlite::Error) -> bool {
    if let rusqlite::Error::SqliteFailure(e, _) = err {
        if matches!(
            e.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        ) {
            return true;
        }
    }
    let msg = err.to_string().to_lowercase();
    msg.contains("locked") || msg.contains("busy")
}

/// Convert one SQLite row into a serde_json object keyed by column name,
/// matching Python's `dict(sqlite3.Row)` shape (REAL→float, INTEGER→int,
/// TEXT→string, NULL→null, BLOB→null).
fn row_to_object(row: &rusqlite::Row<'_>, names: &[String]) -> Map<String, Value> {
    let mut m = Map::new();
    for (i, name) in names.iter().enumerate() {
        let v: Value = match row.get_ref(i).unwrap_or(ValueRef::Null) {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(n) => Value::from(n),
            ValueRef::Real(f) => Value::from(f),
            ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(_) => Value::Null,
        };
        m.insert(name.clone(), v);
    }
    m
}

/// Escape SQL LIKE wildcards in a literal the way the Python code does:
/// backslash → `\\`, `%` → `\%`, `_` → `\_` (used with `ESCAPE '\'`).
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// A tiny xorshift PRNG seeded per-call so we don't pull in the `rand` crate.
/// Produces a jitter in milliseconds in `[WRITE_RETRY_MIN_MS, WRITE_RETRY_MAX_MS]`.
fn retry_jitter_ms() -> u64 {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
        | 1;
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let span = WRITE_RETRY_MAX_MS - WRITE_RETRY_MIN_MS + 1;
    WRITE_RETRY_MIN_MS + (x % span)
}

/// SQLite-backed session storage with FTS5 search.
///
/// Thread-safe: all access funnels through a `Mutex<Connection>`, reproducing
/// the Python `threading.Lock`-guarded single-connection model.
pub struct SessionDB {
    pub db_path: PathBuf,
    conn: Mutex<Connection>,
    write_count: Mutex<u64>,
}

impl SessionDB {
    /// Open (creating if needed) the state database at the default path.
    pub fn new() -> Result<Self> {
        Self::open(default_db_path())
    }

    /// Open (creating if needed) the state database at `db_path`.
    pub fn open<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let conn = Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_URI
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        // Short busy timeout — application-level retry with jitter handles
        // contention rather than sitting in SQLite's internal busy handler.
        conn.busy_timeout(std::time::Duration::from_millis(1000))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        let db = SessionDB {
            db_path,
            conn: Mutex::new(conn),
            write_count: Mutex::new(0),
        };
        db.init_schema()?;
        Ok(db)
    }

    // ── Core write helper ──

    /// Execute a write transaction with BEGIN IMMEDIATE + jitter retry.
    ///
    /// `fn` receives the connection and should perform DML. On
    /// `database is locked`, the lock is released, we sleep a random
    /// 20–150ms, and retry — breaking SQLite's deterministic backoff convoy.
    fn execute_write<T, F>(&self, mut f: F) -> Result<T>
    where
        F: FnMut(&Connection) -> Result<T>,
    {
        let mut last_err: Option<rusqlite::Error> = None;
        for attempt in 0..WRITE_MAX_RETRIES {
            let outcome: std::result::Result<T, rusqlite::Error> = {
                let conn = self.conn.lock().unwrap();
                match conn.execute_batch("BEGIN IMMEDIATE") {
                    Ok(()) => {
                        let res = f(&conn);
                        match res {
                            Ok(v) => match conn.execute_batch("COMMIT") {
                                Ok(()) => Ok(v),
                                Err(e) => {
                                    let _ = conn.execute_batch("ROLLBACK");
                                    Err(e)
                                }
                            },
                            Err(e) => {
                                let _ = conn.execute_batch("ROLLBACK");
                                // Propagate application errors immediately
                                // (e.g. ValueError) without retrying.
                                return Err(e);
                            }
                        }
                    }
                    Err(e) => Err(e),
                }
            };

            match outcome {
                Ok(v) => {
                    // Success — periodic best-effort checkpoint.
                    let mut wc = self.write_count.lock().unwrap();
                    *wc += 1;
                    let do_checkpoint = *wc % CHECKPOINT_EVERY_N_WRITES == 0;
                    drop(wc);
                    if do_checkpoint {
                        self.try_wal_checkpoint();
                    }
                    return Ok(v);
                }
                Err(e) => {
                    if is_lock_error(&e) {
                        if attempt < WRITE_MAX_RETRIES - 1 {
                            last_err = Some(e);
                            std::thread::sleep(std::time::Duration::from_millis(retry_jitter_ms()));
                            continue;
                        }
                    }
                    return Err(StateError::Sqlite(e));
                }
            }
        }
        Err(StateError::Sqlite(last_err.unwrap_or_else(|| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                Some("database is locked after max retries".into()),
            )
        })))
    }

    /// Best-effort PASSIVE WAL checkpoint. Never blocks, never raises.
    fn try_wal_checkpoint(&self) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                let busy: i64 = row.get(0).unwrap_or(0);
                let log: i64 = row.get(1).unwrap_or(0);
                let ckpt: i64 = row.get(2).unwrap_or(0);
                if log > 0 {
                    debug!("WAL checkpoint: {ckpt}/{log} pages checkpointed (busy={busy})");
                }
                Ok(())
            });
        }
    }

    /// Close the connection (best-effort PASSIVE checkpoint first). Idempotent.
    pub fn close(self) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)");
        }
        // Connection drops when `self` is consumed.
    }

    // ── Schema ──

    /// Extract expected columns per table from `SCHEMA_SQL` using an in-memory
    /// SQLite DB, exactly as the Python `_parse_schema_columns` does. Returns
    /// `Vec<(table, Vec<(col_name, type_expr)>)>` preserving declaration order.
    fn parse_schema_columns(schema_sql: &str) -> Result<Vec<(String, Vec<(String, String)>)>> {
        let ref_conn = Connection::open_in_memory()?;
        ref_conn.execute_batch(schema_sql)?;

        let tables: Vec<String> = {
            let mut stmt = ref_conn.prepare(
                "SELECT name FROM sqlite_master \
                 WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            )?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let mut out: Vec<(String, Vec<(String, String)>)> = Vec::new();
        for tbl in tables {
            let mut cols: Vec<(String, String)> = Vec::new();
            let mut stmt = ref_conn.prepare(&format!("PRAGMA table_info(\"{}\")", tbl))?;
            let rows = stmt.query_map([], |row| {
                // (cid, name, type, notnull, dflt_value, pk)
                let name: String = row.get(1)?;
                let col_type: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
                let notnull: i64 = row.get(3)?;
                let default: Option<String> = match row.get_ref(4)? {
                    ValueRef::Null => None,
                    ValueRef::Integer(n) => Some(n.to_string()),
                    ValueRef::Real(f) => Some(f.to_string()),
                    ValueRef::Text(t) => Some(String::from_utf8_lossy(t).into_owned()),
                    ValueRef::Blob(_) => None,
                };
                let pk: i64 = row.get(5)?;
                Ok((name, col_type, notnull, default, pk))
            })?;
            for r in rows {
                let (name, col_type, notnull, default, pk) = r?;
                let mut parts: Vec<String> = Vec::new();
                if !col_type.is_empty() {
                    parts.push(col_type);
                }
                if notnull != 0 && pk == 0 {
                    parts.push("NOT NULL".to_string());
                }
                if let Some(d) = default {
                    parts.push(format!("DEFAULT {d}"));
                }
                cols.push((name, parts.join(" ")));
            }
            out.push((tbl, cols));
        }
        Ok(out)
    }

    /// Ensure live tables have every column declared in `SCHEMA_SQL`.
    fn reconcile_columns(conn: &Connection) -> Result<()> {
        let expected = Self::parse_schema_columns(SCHEMA_SQL)?;
        for (table_name, declared_cols) in expected {
            let live_cols: std::collections::HashSet<String> = {
                let stmt = conn.prepare(&format!("PRAGMA table_info(\"{}\")", table_name));
                match stmt {
                    Err(_) => continue,
                    Ok(mut stmt) => {
                        let rows = stmt.query_map([], |r| r.get::<_, String>(1));
                        match rows {
                            Err(_) => continue,
                            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                        }
                    }
                }
            };
            for (col_name, col_type) in declared_cols {
                if !live_cols.contains(&col_name) {
                    let safe_name = col_name.replace('"', "\"\"");
                    let sql = format!(
                        "ALTER TABLE \"{}\" ADD COLUMN \"{}\" {}",
                        table_name, safe_name, col_type
                    );
                    if let Err(exc) = conn.execute(&sql, []) {
                        debug!("reconcile {table_name}.{col_name}: {exc}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Create tables and FTS if missing, reconcile columns, run version-gated
    /// data migrations. Mirrors Python `_init_schema`.
    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute_batch(SCHEMA_SQL)?;

        Self::reconcile_columns(&conn)?;

        // ── Schema version bookkeeping ──
        let existing: Option<i64> = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .ok();

        match existing {
            None => {
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [SCHEMA_VERSION],
                )?;
            }
            Some(current_version) => {
                if current_version < 10 {
                    // v10: trigram FTS5 table + one-time backfill.
                    let fts_trigram_exists = conn
                        .execute_batch("SELECT * FROM messages_fts_trigram LIMIT 0")
                        .is_ok();
                    if !fts_trigram_exists {
                        conn.execute_batch(FTS_TRIGRAM_SQL)?;
                        conn.execute(
                            "INSERT INTO messages_fts_trigram(rowid, content) \
                             SELECT id, content FROM messages WHERE content IS NOT NULL",
                            [],
                        )?;
                    }
                }
                if current_version < 11 {
                    // v11: drop old FTS tables/triggers and rebuild + backfill.
                    for trig in [
                        "messages_fts_insert",
                        "messages_fts_delete",
                        "messages_fts_update",
                        "messages_fts_trigram_insert",
                        "messages_fts_trigram_delete",
                        "messages_fts_trigram_update",
                    ] {
                        let _ = conn.execute_batch(&format!("DROP TRIGGER IF EXISTS {trig}"));
                    }
                    for tbl in ["messages_fts", "messages_fts_trigram"] {
                        let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {tbl}"));
                    }
                    conn.execute_batch(FTS_SQL)?;
                    conn.execute_batch(FTS_TRIGRAM_SQL)?;
                    conn.execute(
                        "INSERT INTO messages_fts(rowid, content) \
                         SELECT id, \
                         COALESCE(content, '') || ' ' || \
                         COALESCE(tool_name, '') || ' ' || \
                         COALESCE(tool_calls, '') \
                         FROM messages",
                        [],
                    )?;
                    conn.execute(
                        "INSERT INTO messages_fts_trigram(rowid, content) \
                         SELECT id, \
                         COALESCE(content, '') || ' ' || \
                         COALESCE(tool_name, '') || ' ' || \
                         COALESCE(tool_calls, '') \
                         FROM messages",
                        [],
                    )?;
                }
                if current_version < SCHEMA_VERSION {
                    conn.execute("UPDATE schema_version SET version = ?", [SCHEMA_VERSION])?;
                }
            }
        }

        // Unique title index — always ensure it exists.
        let _ = conn.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique \
             ON sessions(title) WHERE title IS NOT NULL",
        );

        // FTS5 setup (separate; CREATE VIRTUAL TABLE can't reliably be in batch
        // with IF NOT EXISTS).
        if conn
            .execute_batch("SELECT * FROM messages_fts LIMIT 0")
            .is_err()
        {
            conn.execute_batch(FTS_SQL)?;
        }
        if conn
            .execute_batch("SELECT * FROM messages_fts_trigram LIMIT 0")
            .is_err()
        {
            conn.execute_batch(FTS_TRIGRAM_SQL)?;
        }

        Ok(())
    }

    // =====================================================================
    // Session lifecycle
    // =====================================================================

    /// Shared INSERT OR IGNORE for session rows.
    #[allow(clippy::too_many_arguments)]
    fn insert_session_row(
        &self,
        session_id: &str,
        source: &str,
        model: Option<&str>,
        model_config: Option<&Value>,
        system_prompt: Option<&str>,
        user_id: Option<&str>,
        parent_session_id: Option<&str>,
    ) -> Result<()> {
        let model_config_json = model_config
            .filter(|v| !v.is_null())
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        let ts = now_ts();
        self.execute_write(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config, \
                 system_prompt, parent_session_id, started_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rusqlite::params![
                    session_id,
                    source,
                    user_id,
                    model,
                    model_config_json,
                    system_prompt,
                    parent_session_id,
                    ts,
                ],
            )?;
            Ok(())
        })
    }

    /// Create a new session record. Returns the session_id.
    pub fn create_session(
        &self,
        session_id: &str,
        source: &str,
        model: Option<&str>,
        model_config: Option<&Value>,
        system_prompt: Option<&str>,
        user_id: Option<&str>,
        parent_session_id: Option<&str>,
    ) -> Result<String> {
        self.insert_session_row(
            session_id,
            source,
            model,
            model_config,
            system_prompt,
            user_id,
            parent_session_id,
        )?;
        Ok(session_id.to_string())
    }

    /// Ensure a session row exists (INSERT OR IGNORE). Returns the session_id.
    pub fn ensure_session(
        &self,
        session_id: &str,
        source: &str,
        model: Option<&str>,
        model_config: Option<&Value>,
        system_prompt: Option<&str>,
        user_id: Option<&str>,
        parent_session_id: Option<&str>,
    ) -> Result<String> {
        self.insert_session_row(
            session_id,
            source,
            model,
            model_config,
            system_prompt,
            user_id,
            parent_session_id,
        )?;
        Ok(session_id.to_string())
    }

    /// Mark a session as ended. No-ops when already ended (first reason wins).
    pub fn end_session(&self, session_id: &str, end_reason: &str) -> Result<()> {
        let ts = now_ts();
        self.execute_write(|conn| {
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? \
                 WHERE id = ? AND ended_at IS NULL",
                rusqlite::params![ts, end_reason, session_id],
            )?;
            Ok(())
        })
    }

    /// Clear ended_at/end_reason so a session can be resumed.
    pub fn reopen_session(&self, session_id: &str) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                [session_id],
            )?;
            Ok(())
        })
    }

    /// Store the full assembled system prompt snapshot.
    pub fn update_system_prompt(&self, session_id: &str, system_prompt: &str) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                rusqlite::params![system_prompt, session_id],
            )?;
            Ok(())
        })
    }

    /// Token / cost / billing counter update. See [`TokenUpdate`] for fields.
    pub fn update_token_counts(&self, session_id: &str, upd: &TokenUpdate) -> Result<()> {
        let sql = if upd.absolute {
            "UPDATE sessions SET \
               input_tokens = ?, \
               output_tokens = ?, \
               cache_read_tokens = ?, \
               cache_write_tokens = ?, \
               reasoning_tokens = ?, \
               estimated_cost_usd = COALESCE(?, 0), \
               actual_cost_usd = CASE WHEN ? IS NULL THEN actual_cost_usd ELSE ? END, \
               cost_status = COALESCE(?, cost_status), \
               cost_source = COALESCE(?, cost_source), \
               pricing_version = COALESCE(?, pricing_version), \
               billing_provider = COALESCE(billing_provider, ?), \
               billing_base_url = COALESCE(billing_base_url, ?), \
               billing_mode = COALESCE(billing_mode, ?), \
               model = COALESCE(model, ?), \
               api_call_count = ? \
               WHERE id = ?"
        } else {
            "UPDATE sessions SET \
               input_tokens = input_tokens + ?, \
               output_tokens = output_tokens + ?, \
               cache_read_tokens = cache_read_tokens + ?, \
               cache_write_tokens = cache_write_tokens + ?, \
               reasoning_tokens = reasoning_tokens + ?, \
               estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0), \
               actual_cost_usd = CASE WHEN ? IS NULL THEN actual_cost_usd ELSE COALESCE(actual_cost_usd, 0) + ? END, \
               cost_status = COALESCE(?, cost_status), \
               cost_source = COALESCE(?, cost_source), \
               pricing_version = COALESCE(?, pricing_version), \
               billing_provider = COALESCE(billing_provider, ?), \
               billing_base_url = COALESCE(billing_base_url, ?), \
               billing_mode = COALESCE(billing_mode, ?), \
               model = COALESCE(model, ?), \
               api_call_count = COALESCE(api_call_count, 0) + ? \
               WHERE id = ?"
        };
        self.execute_write(|conn| {
            conn.execute(
                sql,
                rusqlite::params![
                    upd.input_tokens,
                    upd.output_tokens,
                    upd.cache_read_tokens,
                    upd.cache_write_tokens,
                    upd.reasoning_tokens,
                    upd.estimated_cost_usd,
                    upd.actual_cost_usd,
                    upd.actual_cost_usd,
                    upd.cost_status,
                    upd.cost_source,
                    upd.pricing_version,
                    upd.billing_provider,
                    upd.billing_base_url,
                    upd.billing_mode,
                    upd.model,
                    upd.api_call_count,
                    session_id,
                ],
            )?;
            Ok(())
        })
    }

    /// Remove empty TUI ghost sessions (no messages, no title, >24hr old).
    /// Returns the number removed. `sessions_dir`, if provided, gets its
    /// on-disk transcript files cleaned up too.
    pub fn prune_empty_ghost_sessions(&self, sessions_dir: Option<&Path>) -> Result<usize> {
        let cutoff = now_ts() - 86400.0;
        let removed_ids: Vec<String> = self.execute_write(|conn| {
            let ids: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT id FROM sessions \
                     WHERE source = 'tui' \
                       AND title IS NULL \
                       AND ended_at IS NOT NULL \
                       AND started_at < ? \
                       AND NOT EXISTS ( \
                           SELECT 1 FROM messages WHERE messages.session_id = sessions.id \
                       )",
                )?;
                let rows = stmt.query_map([cutoff], |r| r.get::<_, String>(0))?;
                rows.filter_map(|r| r.ok()).collect()
            };
            if !ids.is_empty() {
                let placeholders = vec!["?"; ids.len()].join(",");
                conn.execute(
                    &format!("DELETE FROM sessions WHERE id IN ({placeholders})"),
                    params_from_iter(ids.iter()),
                )?;
            }
            Ok(ids)
        })?;
        if let Some(dir) = sessions_dir {
            for sid in &removed_ids {
                Self::remove_session_files(Some(dir), sid);
            }
        }
        Ok(removed_ids.len())
    }

    /// Mark orphaned compression continuation sessions as ended. Returns count.
    pub fn finalize_orphaned_compression_sessions(&self) -> Result<usize> {
        let cutoff = now_ts() - 604800.0;
        self.execute_write(|conn| {
            let now = now_ts();
            let n = conn.execute(
                "UPDATE sessions \
                 SET ended_at = ?, end_reason = 'orphaned_compression' \
                 WHERE api_call_count = 0 \
                   AND end_reason IS NULL \
                   AND ended_at IS NULL \
                   AND started_at < ? \
                   AND parent_session_id IS NOT NULL \
                   AND EXISTS ( \
                       SELECT 1 FROM sessions p \
                       WHERE p.id = sessions.parent_session_id \
                         AND p.end_reason = 'compression' \
                         AND p.ended_at IS NOT NULL \
                   ) \
                   AND EXISTS ( \
                       SELECT 1 FROM messages m WHERE m.session_id = sessions.id \
                   )",
                rusqlite::params![now, cutoff],
            )?;
            Ok(n)
        })
    }

    /// Get a session by ID as a JSON object, or None.
    pub fn get_session(&self, session_id: &str) -> Result<Option<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT * FROM sessions WHERE id = ?")?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = stmt.query([session_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_object(row, &names)))
        } else {
            Ok(None)
        }
    }

    /// Resolve an exact or uniquely prefixed session ID to the full ID.
    pub fn resolve_session_id(&self, session_id_or_prefix: &str) -> Result<Option<String>> {
        if let Some(exact) = self.get_session(session_id_or_prefix)? {
            if let Some(Value::String(id)) = exact.get("id") {
                return Ok(Some(id.clone()));
            }
        }
        let escaped = escape_like(session_id_or_prefix);
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
        )?;
        let matches: Vec<String> = stmt
            .query_map([format!("{escaped}%")], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        if matches.len() == 1 {
            Ok(Some(matches[0].clone()))
        } else {
            Ok(None)
        }
    }

    /// Validate and sanitize a session title (see Python `sanitize_title`).
    /// Returns Ok(None) for empty/blank, Err on over-length.
    pub fn sanitize_title(title: Option<&str>) -> Result<Option<String>> {
        let title = match title {
            None => return Ok(None),
            Some(t) if t.is_empty() => return Ok(None),
            Some(t) => t,
        };

        // Remove ASCII control chars 0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f, 0x7f
        // but keep \t \n \r so the whitespace-collapse step normalizes them.
        let mut cleaned: String = title
            .chars()
            .filter(|&c| {
                let cp = c as u32;
                let ascii_ctrl = (0x00..=0x08).contains(&cp)
                    || cp == 0x0b
                    || cp == 0x0c
                    || (0x0e..=0x1f).contains(&cp)
                    || cp == 0x7f;
                !ascii_ctrl
            })
            // Remove problematic Unicode control chars.
            .filter(|&c| {
                let cp = c as u32;
                let bad = (0x200b..=0x200f).contains(&cp)
                    || (0x2028..=0x202e).contains(&cp)
                    || (0x2060..=0x2069).contains(&cp)
                    || cp == 0xfeff
                    || cp == 0xfffc
                    || (0xfff9..=0xfffb).contains(&cp);
                !bad
            })
            .collect();

        // Collapse internal whitespace runs to single spaces and trim.
        let collapsed = {
            let mut out = String::with_capacity(cleaned.len());
            let mut prev_ws = false;
            for c in cleaned.chars() {
                if c.is_whitespace() {
                    if !prev_ws {
                        out.push(' ');
                    }
                    prev_ws = true;
                } else {
                    out.push(c);
                    prev_ws = false;
                }
            }
            out.trim().to_string()
        };
        cleaned = collapsed;

        if cleaned.is_empty() {
            return Ok(None);
        }
        // Python `len()` counts Unicode code points.
        let char_len = cleaned.chars().count();
        if char_len > MAX_TITLE_LENGTH {
            return Err(StateError::Value(format!(
                "Title too long ({char_len} chars, max {MAX_TITLE_LENGTH})"
            )));
        }
        Ok(Some(cleaned))
    }

    /// Set or update a session's title. Returns true if the session was found.
    /// Err on uniqueness conflict or validation failure.
    pub fn set_session_title(&self, session_id: &str, title: Option<&str>) -> Result<bool> {
        let title = Self::sanitize_title(title)?;
        let session_id_owned = session_id.to_string();
        let rowcount = self.execute_write(move |conn| {
            if let Some(ref t) = title {
                let conflict: Option<String> = conn
                    .query_row(
                        "SELECT id FROM sessions WHERE title = ? AND id != ?",
                        rusqlite::params![t, session_id_owned],
                        |r| r.get(0),
                    )
                    .ok();
                if let Some(other) = conflict {
                    return Err(StateError::Value(format!(
                        "Title '{t}' is already in use by session {other}"
                    )));
                }
            }
            let n = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                rusqlite::params![title, session_id_owned],
            )?;
            Ok(n)
        })?;
        Ok(rowcount > 0)
    }

    /// Get the title for a session, or None.
    pub fn get_session_title(&self, session_id: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let r: Option<Option<String>> = conn
            .query_row(
                "SELECT title FROM sessions WHERE id = ?",
                [session_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .ok();
        Ok(r.flatten())
    }

    /// Look up a session by exact title. Returns session dict or None.
    pub fn get_session_by_title(&self, title: &str) -> Result<Option<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT * FROM sessions WHERE title = ?")?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = stmt.query([title])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_object(row, &names)))
        } else {
            Ok(None)
        }
    }

    /// Resolve a title to a session ID, preferring the latest in a lineage.
    pub fn resolve_session_by_title(&self, title: &str) -> Result<Option<String>> {
        let exact = self.get_session_by_title(title)?;
        let escaped = escape_like(title);
        let numbered_first: Option<String> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT id, title, started_at FROM sessions \
                 WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
            )?;
            let mut rows = stmt.query([format!("{escaped} #%")])?;
            if let Some(row) = rows.next()? {
                Some(row.get::<_, String>(0)?)
            } else {
                None
            }
        };
        if let Some(id) = numbered_first {
            Ok(Some(id))
        } else if let Some(s) = exact {
            if let Some(Value::String(id)) = s.get("id") {
                Ok(Some(id.clone()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Generate the next title in a lineage ("my session" → "my session #2").
    pub fn get_next_title_in_lineage(&self, base_title: &str) -> Result<String> {
        // Strip existing #N suffix to find the true base.
        let suffix_re = regex::Regex::new(r"^(.*?) #(\d+)$").unwrap();
        let base = if let Some(c) = suffix_re.captures(base_title) {
            c.get(1).unwrap().as_str().to_string()
        } else {
            base_title.to_string()
        };

        let escaped = escape_like(&base);
        let existing: Vec<String> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
            )?;
            let rows = stmt.query_map(
                rusqlite::params![base, format!("{escaped} #%")],
                |r| r.get::<_, Option<String>>(0),
            )?;
            rows.filter_map(|r| r.ok().flatten()).collect()
        };

        if existing.is_empty() {
            return Ok(base);
        }

        let num_re = regex::Regex::new(r"^.* #(\d+)$").unwrap();
        let mut max_num: i64 = 1;
        for t in &existing {
            if let Some(c) = num_re.captures(t) {
                if let Ok(n) = c.get(1).unwrap().as_str().parse::<i64>() {
                    max_num = max_num.max(n);
                }
            }
        }
        Ok(format!("{base} #{}", max_num + 1))
    }

    /// Walk the compression-continuation chain forward and return the tip.
    pub fn get_compression_tip(&self, session_id: &str) -> Result<String> {
        let mut current = session_id.to_string();
        let conn = self.conn.lock().unwrap();
        for _ in 0..100 {
            let next: Option<String> = conn
                .query_row(
                    "SELECT id FROM sessions \
                     WHERE parent_session_id = ? \
                       AND started_at >= ( \
                           SELECT ended_at FROM sessions \
                           WHERE id = ? AND end_reason = 'compression' \
                       ) \
                     ORDER BY started_at DESC LIMIT 1",
                    rusqlite::params![current, current],
                    |r| r.get(0),
                )
                .ok();
            match next {
                None => return Ok(current),
                Some(id) => current = id,
            }
        }
        Ok(current)
    }

    /// List sessions with preview + last_active. See Python `list_sessions_rich`.
    #[allow(clippy::too_many_arguments)]
    pub fn list_sessions_rich(
        &self,
        source: Option<&str>,
        exclude_sources: Option<&[String]>,
        limit: i64,
        offset: i64,
        include_children: bool,
        project_compression_tips: bool,
        order_by_last_active: bool,
    ) -> Result<Vec<Map<String, Value>>> {
        let mut where_clauses: Vec<String> = Vec::new();
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();

        if !include_children {
            where_clauses.push(
                "(s.parent_session_id IS NULL \
                 OR EXISTS (SELECT 1 FROM sessions p \
                            WHERE p.id = s.parent_session_id \
                            AND p.end_reason = 'branched' \
                            AND s.started_at >= p.ended_at))"
                    .to_string(),
            );
        }
        if let Some(src) = source {
            where_clauses.push("s.source = ?".to_string());
            params.push(Box::new(src.to_string()));
        }
        if let Some(excl) = exclude_sources {
            if !excl.is_empty() {
                let placeholders = vec!["?"; excl.len()].join(",");
                where_clauses.push(format!("s.source NOT IN ({placeholders})"));
                for s in excl {
                    params.push(Box::new(s.clone()));
                }
            }
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query: String;
        if order_by_last_active {
            query = format!(
                "WITH RECURSIVE chain(root_id, cur_id) AS ( \
                    SELECT s.id, s.id FROM sessions s {where_sql} \
                    UNION ALL \
                    SELECT c.root_id, child.id \
                    FROM chain c \
                    JOIN sessions parent ON parent.id = c.cur_id \
                    JOIN sessions child ON child.parent_session_id = c.cur_id \
                    WHERE parent.end_reason = 'compression' \
                      AND child.started_at >= parent.ended_at \
                ), \
                chain_max AS ( \
                    SELECT root_id, MAX(COALESCE( \
                            (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = cur_id), \
                            (SELECT started_at FROM sessions ss WHERE ss.id = cur_id) \
                        )) AS effective_last_active \
                    FROM chain GROUP BY root_id \
                ) \
                SELECT s.*, \
                    COALESCE( \
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63) \
                         FROM messages m \
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL \
                         ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw, \
                    COALESCE( \
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id), \
                        s.started_at) AS last_active, \
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active \
                FROM sessions s \
                LEFT JOIN chain_max cm ON cm.root_id = s.id \
                {where_sql} \
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC \
                LIMIT ? OFFSET ?"
            );
            // WHERE params apply twice (CTE seed + outer select).
            let mut doubled: Vec<Box<dyn ToSql>> = Vec::new();
            for p in &params {
                doubled.push(Box::new(sql_clone(p.as_ref())));
            }
            params.append(&mut doubled);
            params.push(Box::new(limit));
            params.push(Box::new(offset));
        } else {
            query = format!(
                "SELECT s.*, \
                    COALESCE( \
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63) \
                         FROM messages m \
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL \
                         ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw, \
                    COALESCE( \
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id), \
                        s.started_at) AS last_active \
                FROM sessions s \
                {where_sql} \
                ORDER BY s.started_at DESC \
                LIMIT ? OFFSET ?"
            );
            params.push(Box::new(limit));
            params.push(Box::new(offset));
        }

        let mut sessions: Vec<Map<String, Value>> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn.prepare(&query)?;
            let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
            let param_refs: Vec<&dyn ToSql> = params.iter().map(|b| b.as_ref()).collect();
            let mut rows = stmt.query(param_refs.as_slice())?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(row_to_object(row, &names));
            }
            out
        };

        for s in sessions.iter_mut() {
            let raw = s
                .remove("_preview_raw")
                .and_then(|v| v.as_str().map(|x| x.to_string()))
                .unwrap_or_default();
            let raw = raw.trim().to_string();
            apply_preview(s, &raw);
            s.remove("_effective_last_active");
        }

        if project_compression_tips && !include_children {
            let mut projected: Vec<Map<String, Value>> = Vec::new();
            for s in sessions.into_iter() {
                let is_compression = s.get("end_reason").and_then(|v| v.as_str())
                    == Some("compression");
                if !is_compression {
                    projected.push(s);
                    continue;
                }
                let sid = match s.get("id").and_then(|v| v.as_str()) {
                    Some(id) => id.to_string(),
                    None => {
                        projected.push(s);
                        continue;
                    }
                };
                let tip_id = self.get_compression_tip(&sid)?;
                if tip_id == sid {
                    projected.push(s);
                    continue;
                }
                let tip_row = self.get_session_rich_row(&tip_id)?;
                match tip_row {
                    None => projected.push(s),
                    Some(tip) => {
                        let mut merged = s.clone();
                        for key in [
                            "id",
                            "ended_at",
                            "end_reason",
                            "message_count",
                            "tool_call_count",
                            "title",
                            "last_active",
                            "preview",
                            "model",
                            "system_prompt",
                        ] {
                            if let Some(v) = tip.get(key) {
                                merged.insert(key.to_string(), v.clone());
                            }
                        }
                        merged.insert("_lineage_root_id".to_string(), Value::String(sid));
                        projected.push(merged);
                    }
                }
            }
            sessions = projected;
        }

        Ok(sessions)
    }

    /// Fetch a single session with enriched columns (preview + last_active).
    fn get_session_rich_row(&self, session_id: &str) -> Result<Option<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT s.*, \
                COALESCE( \
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63) \
                     FROM messages m \
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL \
                     ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw, \
                COALESCE( \
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id), \
                    s.started_at) AS last_active \
            FROM sessions s WHERE s.id = ?",
        )?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = stmt.query([session_id])?;
        if let Some(row) = rows.next()? {
            let mut s = row_to_object(row, &names);
            let raw = s
                .remove("_preview_raw")
                .and_then(|v| v.as_str().map(|x| x.to_string()))
                .unwrap_or_default();
            apply_preview(&mut s, raw.trim());
            Ok(Some(s))
        } else {
            Ok(None)
        }
    }

    // =====================================================================
    // Message storage
    // =====================================================================

    /// Serialize structured (array/object) message content for sqlite. Scalars
    /// pass through unchanged; arrays/objects get a sentinel-prefixed JSON
    /// string. Returns None when input is JSON null.
    fn encode_content(content: &Value) -> Option<String> {
        match content {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            other => {
                let json = serde_json::to_string(other).unwrap_or_else(|_| other.to_string());
                Some(format!("{CONTENT_JSON_PREFIX}{json}"))
            }
        }
    }

    /// Reverse [`Self::encode_content`]; returns scalars unchanged.
    fn decode_content(content: Option<&str>) -> Value {
        match content {
            None => Value::Null,
            Some(s) => {
                if let Some(rest) = s.strip_prefix(CONTENT_JSON_PREFIX) {
                    match serde_json::from_str::<Value>(rest) {
                        Ok(v) => v,
                        Err(_) => {
                            warn!(
                                "Failed to decode JSON-encoded message content; \
                                 returning raw string"
                            );
                            Value::String(s.to_string())
                        }
                    }
                } else {
                    Value::String(s.to_string())
                }
            }
        }
    }

    /// Append a message to a session. Returns the message row ID. See
    /// [`MessageInput`] for fields. Also increments message_count and
    /// tool_call_count.
    pub fn append_message(&self, session_id: &str, msg: &MessageInput) -> Result<i64> {
        let reasoning_details_json = json_if_truthy(&msg.reasoning_details);
        let codex_items_json = json_if_truthy(&msg.codex_reasoning_items);
        let codex_message_items_json = json_if_truthy(&msg.codex_message_items);
        let tool_calls_json = json_if_truthy(&msg.tool_calls);
        let stored_content = Self::encode_content(&msg.content);

        let num_tool_calls: i64 = match &msg.tool_calls {
            Value::Null => 0,
            Value::Array(a) => a.len() as i64,
            _ => 1,
        };

        let ts = now_ts();
        self.execute_write(|conn| {
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_call_id, \
                 tool_calls, tool_name, timestamp, token_count, finish_reason, \
                 reasoning, reasoning_content, reasoning_details, codex_reasoning_items, \
                 codex_message_items) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rusqlite::params![
                    session_id,
                    msg.role,
                    stored_content,
                    msg.tool_call_id,
                    tool_calls_json,
                    msg.tool_name,
                    ts,
                    msg.token_count,
                    msg.finish_reason,
                    msg.reasoning,
                    msg.reasoning_content,
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                ],
            )?;
            let msg_id = conn.last_insert_rowid();
            if num_tool_calls > 0 {
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1, \
                     tool_call_count = tool_call_count + ? WHERE id = ?",
                    rusqlite::params![num_tool_calls, session_id],
                )?;
            } else {
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    [session_id],
                )?;
            }
            Ok(msg_id)
        })
    }

    /// Atomically replace every message for a session (for /retry, /undo,
    /// /compress transcript rewrites). `messages` items are JSON objects with
    /// keys matching the Python `dict` shape.
    pub fn replace_messages(&self, session_id: &str, messages: &[Value]) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                [session_id],
            )?;

            let mut now_ts_local = now_ts();
            let mut total_messages: i64 = 0;
            let mut total_tool_calls: i64 = 0;

            for msg in messages {
                let role = msg
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let is_assistant = role == "assistant";
                let tool_calls = msg.get("tool_calls").cloned().unwrap_or(Value::Null);

                let reasoning_details = if is_assistant {
                    msg.get("reasoning_details").cloned().unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let codex_reasoning_items = if is_assistant {
                    msg.get("codex_reasoning_items")
                        .cloned()
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let codex_message_items = if is_assistant {
                    msg.get("codex_message_items")
                        .cloned()
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };

                let reasoning_details_json = json_if_truthy(&reasoning_details);
                let codex_items_json = json_if_truthy(&codex_reasoning_items);
                let codex_message_items_json = json_if_truthy(&codex_message_items);
                let tool_calls_json = json_if_truthy(&tool_calls);

                let stored_content =
                    Self::encode_content(msg.get("content").unwrap_or(&Value::Null));
                let tool_call_id = str_opt(msg.get("tool_call_id"));
                let tool_name = str_opt(msg.get("tool_name"));
                let token_count = int_opt(msg.get("token_count"));
                let finish_reason = str_opt(msg.get("finish_reason"));
                let reasoning = if is_assistant {
                    str_opt(msg.get("reasoning"))
                } else {
                    None
                };
                let reasoning_content = if is_assistant {
                    str_opt(msg.get("reasoning_content"))
                } else {
                    None
                };

                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tool_call_id, \
                     tool_calls, tool_name, timestamp, token_count, finish_reason, \
                     reasoning, reasoning_content, reasoning_details, codex_reasoning_items, \
                     codex_message_items) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rusqlite::params![
                        session_id,
                        role,
                        stored_content,
                        tool_call_id,
                        tool_calls_json,
                        tool_name,
                        now_ts_local,
                        token_count,
                        finish_reason,
                        reasoning,
                        reasoning_content,
                        reasoning_details_json,
                        codex_items_json,
                        codex_message_items_json,
                    ],
                )?;
                total_messages += 1;
                match &tool_calls {
                    Value::Null => {}
                    Value::Array(a) => total_tool_calls += a.len() as i64,
                    _ => total_tool_calls += 1,
                }
                now_ts_local += 1e-6;
            }

            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                rusqlite::params![total_messages, total_tool_calls, session_id],
            )?;
            Ok(())
        })
    }

    /// Load all messages for a session as JSON objects, ordered by timestamp.
    /// `content` is decoded and `tool_calls` deserialized.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
        )?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = stmt.query([session_id])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let mut msg = row_to_object(row, &names);
            if msg.contains_key("content") {
                let raw = msg.get("content").and_then(|v| v.as_str());
                let decoded = Self::decode_content(raw);
                msg.insert("content".to_string(), decoded);
            }
            if let Some(tc) = msg.get("tool_calls").and_then(|v| v.as_str()) {
                if !tc.is_empty() {
                    match serde_json::from_str::<Value>(tc) {
                        Ok(v) => {
                            msg.insert("tool_calls".to_string(), v);
                        }
                        Err(_) => {
                            warn!(
                                "Failed to deserialize tool_calls in get_messages, \
                                 falling back to []"
                            );
                            msg.insert("tool_calls".to_string(), Value::Array(vec![]));
                        }
                    }
                }
            }
            result.push(msg);
        }
        Ok(result)
    }

    /// Redirect a resume target to the descendant session that holds messages.
    pub fn resolve_resume_session_id(&self, session_id: &str) -> Result<String> {
        if session_id.is_empty() {
            return Ok(session_id.to_string());
        }
        let conn = self.conn.lock().unwrap();
        // If this session already has messages, nothing to redirect.
        let has_own: bool = conn
            .query_row(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                [session_id],
                |_| Ok(()),
            )
            .is_ok();
        if has_own {
            return Ok(session_id.to_string());
        }

        let mut current = session_id.to_string();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(current.clone());
        for _ in 0..32 {
            let child_id: Option<String> = conn
                .query_row(
                    "SELECT id FROM sessions WHERE parent_session_id = ? \
                     ORDER BY started_at DESC, id DESC LIMIT 1",
                    [&current],
                    |r| r.get(0),
                )
                .ok();
            let child_id = match child_id {
                None => return Ok(session_id.to_string()),
                Some(c) => c,
            };
            if child_id.is_empty() || seen.contains(&child_id) {
                return Ok(session_id.to_string());
            }
            seen.insert(child_id.clone());
            let child_has: bool = conn
                .query_row(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    [&child_id],
                    |_| Ok(()),
                )
                .is_ok();
            if child_has {
                return Ok(child_id);
            }
            current = child_id;
        }
        Ok(session_id.to_string())
    }

    /// Load messages in OpenAI conversation format (role + content dicts).
    pub fn get_messages_as_conversation(
        &self,
        session_id: &str,
        include_ancestors: bool,
    ) -> Result<Vec<Map<String, Value>>> {
        let session_ids: Vec<String> = if include_ancestors {
            self.session_lineage_root_to_tip(session_id)?
        } else {
            vec![session_id.to_string()]
        };

        let rows: Vec<RawConvRow> = {
            let conn = self.conn.lock().unwrap();
            let placeholders = vec!["?"; session_ids.len()].join(",");
            let sql = format!(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, \
                 finish_reason, reasoning, reasoning_content, reasoning_details, \
                 codex_reasoning_items, codex_message_items \
                 FROM messages WHERE session_id IN ({placeholders}) ORDER BY timestamp, id"
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut q = stmt.query(params_from_iter(session_ids.iter()))?;
            let mut out = Vec::new();
            while let Some(row) = q.next()? {
                out.push(RawConvRow {
                    role: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    content: row.get::<_, Option<String>>(1)?,
                    tool_call_id: row.get::<_, Option<String>>(2)?,
                    tool_calls: row.get::<_, Option<String>>(3)?,
                    tool_name: row.get::<_, Option<String>>(4)?,
                    finish_reason: row.get::<_, Option<String>>(5)?,
                    reasoning: row.get::<_, Option<String>>(6)?,
                    reasoning_content: row.get::<_, Option<String>>(7)?,
                    reasoning_details: row.get::<_, Option<String>>(8)?,
                    codex_reasoning_items: row.get::<_, Option<String>>(9)?,
                    codex_message_items: row.get::<_, Option<String>>(10)?,
                });
            }
            out
        };

        let mut messages: Vec<Map<String, Value>> = Vec::new();
        for row in rows {
            let mut content = Self::decode_content(row.content.as_deref());
            if (row.role == "user" || row.role == "assistant") && content.is_string() {
                let s = content.as_str().unwrap();
                content = Value::String(sanitize_context(s).trim().to_string());
            }
            let mut msg = Map::new();
            msg.insert("role".to_string(), Value::String(row.role.clone()));
            msg.insert("content".to_string(), content);
            if let Some(tci) = row.tool_call_id.as_deref() {
                if !tci.is_empty() {
                    msg.insert("tool_call_id".to_string(), Value::String(tci.to_string()));
                }
            }
            if let Some(tn) = row.tool_name.as_deref() {
                if !tn.is_empty() {
                    msg.insert("tool_name".to_string(), Value::String(tn.to_string()));
                }
            }
            if let Some(tc) = row.tool_calls.as_deref() {
                if !tc.is_empty() {
                    match serde_json::from_str::<Value>(tc) {
                        Ok(v) => {
                            msg.insert("tool_calls".to_string(), v);
                        }
                        Err(_) => {
                            warn!(
                                "Failed to deserialize tool_calls in conversation replay, \
                                 falling back to []"
                            );
                            msg.insert("tool_calls".to_string(), Value::Array(vec![]));
                        }
                    }
                }
            }
            if row.role == "assistant" {
                if let Some(fr) = row.finish_reason.as_deref() {
                    if !fr.is_empty() {
                        msg.insert("finish_reason".to_string(), Value::String(fr.to_string()));
                    }
                }
                if let Some(r) = row.reasoning.as_deref() {
                    if !r.is_empty() {
                        msg.insert("reasoning".to_string(), Value::String(r.to_string()));
                    }
                }
                // reasoning_content uses "is not None" semantics — present even
                // when empty string.
                if let Some(rc) = row.reasoning_content.as_ref() {
                    msg.insert(
                        "reasoning_content".to_string(),
                        Value::String(rc.clone()),
                    );
                }
                Self::insert_json_field(&mut msg, "reasoning_details", &row.reasoning_details);
                Self::insert_json_field(
                    &mut msg,
                    "codex_reasoning_items",
                    &row.codex_reasoning_items,
                );
                Self::insert_json_field(
                    &mut msg,
                    "codex_message_items",
                    &row.codex_message_items,
                );
            }
            if include_ancestors && Self::is_duplicate_replayed_user_message(&messages, &msg) {
                continue;
            }
            messages.push(msg);
        }
        Ok(messages)
    }

    fn insert_json_field(msg: &mut Map<String, Value>, key: &str, raw: &Option<String>) {
        if let Some(s) = raw.as_deref() {
            if !s.is_empty() {
                match serde_json::from_str::<Value>(s) {
                    Ok(v) => {
                        msg.insert(key.to_string(), v);
                    }
                    Err(_) => {
                        warn!("Failed to deserialize {key}, falling back to None");
                        msg.insert(key.to_string(), Value::Null);
                    }
                }
            }
        }
    }

    fn session_lineage_root_to_tip(&self, session_id: &str) -> Result<Vec<String>> {
        if session_id.is_empty() {
            return Ok(vec![session_id.to_string()]);
        }
        let mut chain: Vec<String> = Vec::new();
        let mut current = session_id.to_string();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        {
            let conn = self.conn.lock().unwrap();
            for _ in 0..100 {
                if current.is_empty() || seen.contains(&current) {
                    break;
                }
                seen.insert(current.clone());
                chain.push(current.clone());
                let parent: Option<Option<String>> = conn
                    .query_row(
                        "SELECT parent_session_id FROM sessions WHERE id = ?",
                        [&current],
                        |r| r.get::<_, Option<String>>(0),
                    )
                    .ok();
                match parent {
                    None => break, // row missing
                    Some(p) => {
                        current = p.unwrap_or_default();
                    }
                }
            }
        }
        chain.reverse();
        if chain.is_empty() {
            Ok(vec![session_id.to_string()])
        } else {
            Ok(chain)
        }
    }

    fn is_duplicate_replayed_user_message(
        messages: &[Map<String, Value>],
        msg: &Map<String, Value>,
    ) -> bool {
        if msg.get("role").and_then(|v| v.as_str()) != Some("user") {
            return false;
        }
        let content = match msg.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c,
            _ => return false,
        };
        for prev in messages.iter().rev() {
            let role = prev.get("role").and_then(|v| v.as_str());
            if role == Some("user")
                && prev.get("content").and_then(|v| v.as_str()) == Some(content)
            {
                return true;
            }
            if role == Some("assistant") {
                let has_content = prev
                    .get("content")
                    .map(|v| !v.is_null() && v.as_str() != Some(""))
                    .unwrap_or(false);
                let has_tc = prev
                    .get("tool_calls")
                    .map(|v| !v.is_null())
                    .unwrap_or(false);
                if has_content || has_tc {
                    return false;
                }
            }
        }
        false
    }

    // =====================================================================
    // Search
    // =====================================================================

    /// Sanitize user input for safe use in FTS5 MATCH queries.
    pub fn sanitize_fts5_query(query: &str) -> String {
        // Step 1: extract balanced double-quoted phrases via placeholders.
        let quoted_re = regex::Regex::new(r#""[^"]*""#).unwrap();
        let mut quoted_parts: Vec<String> = Vec::new();
        let mut sanitized = quoted_re
            .replace_all(query, |caps: &regex::Captures| {
                quoted_parts.push(caps[0].to_string());
                format!("\u{0}Q{}\u{0}", quoted_parts.len() - 1)
            })
            .into_owned();

        // Step 2: strip remaining FTS5-special characters.
        let special_re = regex::Regex::new(r#"[+{}()"^]"#).unwrap();
        sanitized = special_re.replace_all(&sanitized, " ").into_owned();

        // Step 3: collapse repeated *, drop leading *.
        let star_re = regex::Regex::new(r"\*+").unwrap();
        sanitized = star_re.replace_all(&sanitized, "*").into_owned();
        let lead_star_re = regex::Regex::new(r"(^|\s)\*").unwrap();
        sanitized = lead_star_re.replace_all(&sanitized, "$1").into_owned();

        // Step 4: remove dangling boolean operators at start/end.
        let lead_bool = regex::Regex::new(r"(?i)^(AND|OR|NOT)\b\s*").unwrap();
        sanitized = lead_bool.replace(sanitized.trim(), "").into_owned();
        let trail_bool = regex::Regex::new(r"(?i)\s+(AND|OR|NOT)\s*$").unwrap();
        sanitized = trail_bool.replace(sanitized.trim(), "").into_owned();

        // Step 5: wrap unquoted dotted/hyphenated/underscored terms in quotes.
        let dotted_re = regex::Regex::new(r"\b(\w+(?:[._-]\w+)+)\b").unwrap();
        sanitized = dotted_re.replace_all(&sanitized, "\"$1\"").into_owned();

        // Step 6: restore preserved quoted phrases.
        for (i, quoted) in quoted_parts.iter().enumerate() {
            sanitized = sanitized.replace(&format!("\u{0}Q{i}\u{0}"), quoted);
        }

        sanitized.trim().to_string()
    }

    fn is_cjk_codepoint(cp: u32) -> bool {
        (0x4E00..=0x9FFF).contains(&cp)
            || (0x3400..=0x4DBF).contains(&cp)
            || (0x20000..=0x2A6DF).contains(&cp)
            || (0x3000..=0x303F).contains(&cp)
            || (0x3040..=0x309F).contains(&cp)
            || (0x30A0..=0x30FF).contains(&cp)
            || (0xAC00..=0xD7AF).contains(&cp)
    }

    fn contains_cjk(text: &str) -> bool {
        text.chars().any(|c| Self::is_cjk_codepoint(c as u32))
    }

    fn count_cjk(text: &str) -> usize {
        text.chars().filter(|c| Self::is_cjk_codepoint(*c as u32)).count()
    }

    /// Full-text search across session messages using FTS5. Returns matching
    /// messages with metadata, snippet, and surrounding context.
    pub fn search_messages(
        &self,
        query: &str,
        source_filter: Option<&[String]>,
        exclude_sources: Option<&[String]>,
        role_filter: Option<&[String]>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Map<String, Value>>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let query = Self::sanitize_fts5_query(query);
        if query.is_empty() {
            return Ok(vec![]);
        }

        let build_filters = |params: &mut Vec<Box<dyn ToSql>>| -> Vec<String> {
            let mut clauses: Vec<String> = Vec::new();
            if let Some(sf) = source_filter {
                let ph = vec!["?"; sf.len()].join(",");
                clauses.push(format!("s.source IN ({ph})"));
                for s in sf {
                    params.push(Box::new(s.clone()));
                }
            }
            if let Some(ex) = exclude_sources {
                let ph = vec!["?"; ex.len()].join(",");
                clauses.push(format!("s.source NOT IN ({ph})"));
                for s in ex {
                    params.push(Box::new(s.clone()));
                }
            }
            if let Some(rf) = role_filter {
                if !rf.is_empty() {
                    let ph = vec!["?"; rf.len()].join(",");
                    clauses.push(format!("m.role IN ({ph})"));
                    for s in rf {
                        params.push(Box::new(s.clone()));
                    }
                }
            }
            clauses
        };

        let is_cjk = Self::contains_cjk(&query);

        let mut matches: Vec<Map<String, Value>> = if is_cjk {
            let raw_query = query.trim_matches('"').trim().to_string();
            let cjk_count = Self::count_cjk(&raw_query);
            if cjk_count >= 3 {
                // Trigram FTS5 path.
                let tokens: Vec<&str> = raw_query.split_whitespace().collect();
                let mut parts: Vec<String> = Vec::new();
                for tok in tokens {
                    let upper = tok.to_uppercase();
                    if upper == "AND" || upper == "OR" || upper == "NOT" {
                        parts.push(tok.to_string());
                    } else {
                        parts.push(format!("\"{}\"", tok.replace('"', "\"\"")));
                    }
                }
                let trigram_query = parts.join(" ");
                let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(trigram_query)];
                let mut where_clauses = vec!["messages_fts_trigram MATCH ?".to_string()];
                where_clauses.extend(build_filters(&mut params));
                params.push(Box::new(limit));
                params.push(Box::new(offset));
                let sql = format!(
                    "SELECT m.id, m.session_id, m.role, \
                     snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet, \
                     m.content, m.timestamp, m.tool_name, \
                     s.source, s.model, s.started_at AS session_started \
                     FROM messages_fts_trigram \
                     JOIN messages m ON m.id = messages_fts_trigram.rowid \
                     JOIN sessions s ON s.id = m.session_id \
                     WHERE {} ORDER BY rank LIMIT ? OFFSET ?",
                    where_clauses.join(" AND ")
                );
                self.run_search_query(&sql, &params).unwrap_or_default()
            } else {
                // Short CJK query (1-2 chars) — LIKE substring search.
                let escaped = escape_like(&raw_query);
                let mut params: Vec<Box<dyn ToSql>> = Vec::new();
                // instr() parameter goes first.
                params.push(Box::new(raw_query.clone()));
                let like_pat = format!("%{escaped}%");
                params.push(Box::new(like_pat.clone()));
                params.push(Box::new(like_pat.clone()));
                params.push(Box::new(like_pat.clone()));
                let mut where_clauses = vec![
                    "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' \
                     OR m.tool_calls LIKE ? ESCAPE '\\')"
                        .to_string(),
                ];
                where_clauses.extend(build_filters(&mut params));
                params.push(Box::new(limit));
                params.push(Box::new(offset));
                let sql = format!(
                    "SELECT m.id, m.session_id, m.role, \
                     substr(m.content, max(1, instr(m.content, ?) - 40), 120) AS snippet, \
                     m.content, m.timestamp, m.tool_name, \
                     s.source, s.model, s.started_at AS session_started \
                     FROM messages m JOIN sessions s ON s.id = m.session_id \
                     WHERE {} ORDER BY m.timestamp DESC LIMIT ? OFFSET ?",
                    where_clauses.join(" AND ")
                );
                self.run_search_query(&sql, &params).unwrap_or_default()
            }
        } else {
            let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(query.clone())];
            let mut where_clauses = vec!["messages_fts MATCH ?".to_string()];
            where_clauses.extend(build_filters(&mut params));
            params.push(Box::new(limit));
            params.push(Box::new(offset));
            let sql = format!(
                "SELECT m.id, m.session_id, m.role, \
                 snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet, \
                 m.content, m.timestamp, m.tool_name, \
                 s.source, s.model, s.started_at AS session_started \
                 FROM messages_fts \
                 JOIN messages m ON m.id = messages_fts.rowid \
                 JOIN sessions s ON s.id = m.session_id \
                 WHERE {} ORDER BY rank LIMIT ? OFFSET ?",
                where_clauses.join(" AND ")
            );
            // FTS5 syntax error despite sanitization — return empty.
            self.run_search_query(&sql, &params).unwrap_or_default()
        };

        // Add surrounding context (1 message before + after each match).
        for m in matches.iter_mut() {
            let mid = m.get("id").and_then(|v| v.as_i64());
            let ctx = match mid {
                Some(id) => self.fetch_match_context(id).unwrap_or_default(),
                None => vec![],
            };
            m.insert("context".to_string(), Value::Array(ctx));
        }

        // Remove full content (snippet is enough).
        for m in matches.iter_mut() {
            m.remove("content");
        }

        Ok(matches)
    }

    fn run_search_query(
        &self,
        sql: &str,
        params: &[Box<dyn ToSql>],
    ) -> Result<Vec<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(sql)?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|b| b.as_ref()).collect();
        let mut rows = stmt.query(param_refs.as_slice())?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_object(row, &names));
        }
        Ok(out)
    }

    fn fetch_match_context(&self, match_id: i64) -> Result<Vec<Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "WITH target AS ( \
                 SELECT session_id, timestamp, id FROM messages WHERE id = ? \
             ) \
             SELECT role, content FROM ( \
                 SELECT m.id, m.timestamp, m.role, m.content FROM messages m \
                 JOIN target t ON t.session_id = m.session_id \
                 WHERE (m.timestamp < t.timestamp) \
                    OR (m.timestamp = t.timestamp AND m.id < t.id) \
                 ORDER BY m.timestamp DESC, m.id DESC LIMIT 1 \
             ) \
             UNION ALL \
             SELECT role, content FROM messages WHERE id = ? \
             UNION ALL \
             SELECT role, content FROM ( \
                 SELECT m.id, m.timestamp, m.role, m.content FROM messages m \
                 JOIN target t ON t.session_id = m.session_id \
                 WHERE (m.timestamp > t.timestamp) \
                    OR (m.timestamp = t.timestamp AND m.id > t.id) \
                 ORDER BY m.timestamp ASC, m.id ASC LIMIT 1 \
             )",
        )?;
        let mut rows = stmt.query(rusqlite::params![match_id, match_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let role: String = row.get::<_, Option<String>>(0)?.unwrap_or_default();
            let raw: Option<String> = row.get::<_, Option<String>>(1)?;
            let decoded = Self::decode_content(raw.as_deref());
            let preview = match &decoded {
                Value::Array(parts) => {
                    let texts: Vec<String> = parts
                        .iter()
                        .filter_map(|p| {
                            if p.get("type").and_then(|v| v.as_str()) == Some("text") {
                                p.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                        .filter(|t| !t.is_empty())
                        .collect();
                    let joined = texts.join(" ").trim().to_string();
                    if joined.is_empty() {
                        "[multimodal content]".to_string()
                    } else {
                        joined
                    }
                }
                Value::String(s) => s.clone(),
                _ => String::new(),
            };
            let truncated: String = preview.chars().take(200).collect();
            out.push(json!({"role": role, "content": truncated}));
        }
        Ok(out)
    }

    /// List sessions, optionally filtered by source, enriched with last_active.
    pub fn search_sessions(
        &self,
        source: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Map<String, Value>>> {
        let select = "SELECT s.*, COALESCE(m.last_active, s.started_at) AS last_active \
             FROM sessions s \
             LEFT JOIN (SELECT session_id, MAX(timestamp) AS last_active \
             FROM messages GROUP BY session_id) m ON m.session_id = s.id ";
        let conn = self.conn.lock().unwrap();
        let (sql, params): (String, Vec<Box<dyn ToSql>>) = match source {
            Some(src) => (
                format!(
                    "{select}WHERE s.source = ? \
                     ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?"
                ),
                vec![Box::new(src.to_string()), Box::new(limit), Box::new(offset)],
            ),
            None => (
                format!(
                    "{select}ORDER BY last_active DESC, s.started_at DESC, s.id DESC \
                     LIMIT ? OFFSET ?"
                ),
                vec![Box::new(limit), Box::new(offset)],
            ),
        };
        let mut stmt = conn.prepare(&sql)?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let param_refs: Vec<&dyn ToSql> = params.iter().map(|b| b.as_ref()).collect();
        let mut rows = stmt.query(param_refs.as_slice())?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_object(row, &names));
        }
        Ok(out)
    }

    // =====================================================================
    // Utility
    // =====================================================================

    /// Count sessions, optionally filtered by source.
    pub fn session_count(&self, source: Option<&str>) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let n = match source {
            Some(src) => conn.query_row(
                "SELECT COUNT(*) FROM sessions WHERE source = ?",
                [src],
                |r| r.get(0),
            )?,
            None => conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?,
        };
        Ok(n)
    }

    /// Count messages, optionally for a specific session.
    pub fn message_count(&self, session_id: Option<&str>) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let n = match session_id {
            Some(sid) => conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                [sid],
                |r| r.get(0),
            )?,
            None => conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?,
        };
        Ok(n)
    }

    // =====================================================================
    // Export and cleanup
    // =====================================================================

    /// Export a single session with all its messages as a dict.
    pub fn export_session(&self, session_id: &str) -> Result<Option<Map<String, Value>>> {
        let session = match self.get_session(session_id)? {
            None => return Ok(None),
            Some(s) => s,
        };
        let messages = self.get_messages(session_id)?;
        let mut out = session;
        out.insert(
            "messages".to_string(),
            Value::Array(messages.into_iter().map(Value::Object).collect()),
        );
        Ok(Some(out))
    }

    /// Export all sessions (with messages).
    pub fn export_all(&self, source: Option<&str>) -> Result<Vec<Map<String, Value>>> {
        let sessions = self.search_sessions(source, 100000, 0)?;
        let mut results = Vec::new();
        for session in sessions {
            let sid = session
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let messages = self.get_messages(&sid)?;
            let mut out = session;
            out.insert(
                "messages".to_string(),
                Value::Array(messages.into_iter().map(Value::Object).collect()),
            );
            results.push(out);
        }
        Ok(results)
    }

    /// Delete all messages for a session and reset its counters.
    pub fn clear_messages(&self, session_id: &str) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                [session_id],
            )?;
            Ok(())
        })
    }

    /// Remove on-disk transcript files for a session (best-effort).
    fn remove_session_files(sessions_dir: Option<&Path>, session_id: &str) {
        let dir = match sessions_dir {
            None => return,
            Some(d) => d,
        };
        for suffix in [".json", ".jsonl"] {
            let p = dir.join(format!("{session_id}{suffix}"));
            let _ = std::fs::remove_file(&p);
        }
        // request_dump_{session_id}_*.json
        let prefix = format!("request_dump_{session_id}_");
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&prefix) && name.ends_with(".json") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Delete a session and all its messages. Child sessions are orphaned
    /// (parent_session_id set NULL). Returns true if found and deleted.
    pub fn delete_session(
        &self,
        session_id: &str,
        sessions_dir: Option<&Path>,
    ) -> Result<bool> {
        let deleted = self.execute_write(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM sessions WHERE id = ?",
                [session_id],
                |r| r.get(0),
            )?;
            if count == 0 {
                return Ok(false);
            }
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?",
                [session_id],
            )?;
            conn.execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
            conn.execute("DELETE FROM sessions WHERE id = ?", [session_id])?;
            Ok(true)
        })?;
        if deleted {
            Self::remove_session_files(sessions_dir, session_id);
        }
        Ok(deleted)
    }

    /// Delete ended sessions older than N days. Returns count deleted.
    pub fn prune_sessions(
        &self,
        older_than_days: i64,
        source: Option<&str>,
        sessions_dir: Option<&Path>,
    ) -> Result<usize> {
        let cutoff = now_ts() - (older_than_days as f64 * 86400.0);
        let source_owned = source.map(|s| s.to_string());
        let removed_ids: Vec<String> = self.execute_write(move |conn| {
            let session_ids: Vec<String> = match &source_owned {
                Some(src) => {
                    let mut stmt = conn.prepare(
                        "SELECT id FROM sessions \
                         WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?",
                    )?;
                    let rows = stmt.query_map(
                        rusqlite::params![cutoff, src],
                        |r| r.get::<_, String>(0),
                    )?;
                    rows.filter_map(|r| r.ok()).collect()
                }
                None => {
                    let mut stmt = conn.prepare(
                        "SELECT id FROM sessions WHERE started_at < ? AND ended_at IS NOT NULL",
                    )?;
                    let rows = stmt.query_map([cutoff], |r| r.get::<_, String>(0))?;
                    rows.filter_map(|r| r.ok()).collect()
                }
            };
            if session_ids.is_empty() {
                return Ok(Vec::new());
            }
            let placeholders = vec!["?"; session_ids.len()].join(",");
            conn.execute(
                &format!(
                    "UPDATE sessions SET parent_session_id = NULL \
                     WHERE parent_session_id IN ({placeholders})"
                ),
                params_from_iter(session_ids.iter()),
            )?;
            for sid in &session_ids {
                conn.execute("DELETE FROM messages WHERE session_id = ?", [sid])?;
                conn.execute("DELETE FROM sessions WHERE id = ?", [sid])?;
            }
            Ok(session_ids)
        })?;
        for sid in &removed_ids {
            Self::remove_session_files(sessions_dir, sid);
        }
        Ok(removed_ids.len())
    }

    // ── Meta key/value ──

    /// Read a value from the state_meta key/value store.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let r: Option<Option<String>> = conn
            .query_row("SELECT value FROM state_meta WHERE key = ?", [key], |row| {
                row.get::<_, Option<String>>(0)
            })
            .ok();
        Ok(r.flatten())
    }

    /// Write a value to the state_meta key/value store.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![key, value],
            )?;
            Ok(())
        })
    }

    // ── Telegram topic mode ──

    /// Create Telegram DM topic-mode tables on explicit /topic opt-in, applying
    /// the v1→v2 FK-cascade rebuild migration.
    pub fn apply_telegram_topic_migration(&self) -> Result<()> {
        self.execute_write(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode ( \
                    chat_id TEXT PRIMARY KEY, \
                    user_id TEXT NOT NULL, \
                    enabled INTEGER NOT NULL DEFAULT 1, \
                    activated_at REAL NOT NULL, \
                    updated_at REAL NOT NULL, \
                    has_topics_enabled INTEGER, \
                    allows_users_to_create_topics INTEGER, \
                    capability_checked_at REAL, \
                    intro_message_id TEXT, \
                    pinned_message_id TEXT \
                ); \
                CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings ( \
                    chat_id TEXT NOT NULL, \
                    thread_id TEXT NOT NULL, \
                    user_id TEXT NOT NULL, \
                    session_key TEXT NOT NULL, \
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, \
                    managed_mode TEXT NOT NULL DEFAULT 'auto', \
                    linked_at REAL NOT NULL, \
                    updated_at REAL NOT NULL, \
                    PRIMARY KEY (chat_id, thread_id) \
                ); \
                CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session \
                ON telegram_dm_topic_bindings(session_id); \
                CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user \
                ON telegram_dm_topic_bindings(user_id, chat_id);",
            )?;

            let current: Option<String> = conn
                .query_row(
                    "SELECT value FROM state_meta WHERE key = ?",
                    ["telegram_dm_topic_schema_version"],
                    |r| r.get::<_, Option<String>>(0),
                )
                .ok()
                .flatten();
            let current_version: i64 = current
                .as_deref()
                .filter(|s| s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            if current_version < 2 {
                let needs_rebuild = {
                    let mut stmt =
                        conn.prepare("PRAGMA foreign_key_list('telegram_dm_topic_bindings')")?;
                    let rows = stmt.query_map([], |row| {
                        // (id, seq, table, from, to, on_update, on_delete, match)
                        let table: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
                        let on_delete: String =
                            row.get::<_, Option<String>>(6)?.unwrap_or_default();
                        Ok((table, on_delete))
                    })?;
                    let mut needs = false;
                    for r in rows {
                        let (table, on_delete) = r?;
                        if table == "sessions" && on_delete != "CASCADE" {
                            needs = true;
                        }
                    }
                    needs
                };
                if needs_rebuild {
                    conn.execute_batch(
                        "CREATE TABLE telegram_dm_topic_bindings_new ( \
                            chat_id TEXT NOT NULL, \
                            thread_id TEXT NOT NULL, \
                            user_id TEXT NOT NULL, \
                            session_key TEXT NOT NULL, \
                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, \
                            managed_mode TEXT NOT NULL DEFAULT 'auto', \
                            linked_at REAL NOT NULL, \
                            updated_at REAL NOT NULL, \
                            PRIMARY KEY (chat_id, thread_id) \
                        ); \
                        INSERT INTO telegram_dm_topic_bindings_new \
                            SELECT chat_id, thread_id, user_id, session_key, \
                                   session_id, managed_mode, linked_at, updated_at \
                            FROM telegram_dm_topic_bindings; \
                        DROP TABLE telegram_dm_topic_bindings; \
                        ALTER TABLE telegram_dm_topic_bindings_new \
                            RENAME TO telegram_dm_topic_bindings; \
                        CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session \
                            ON telegram_dm_topic_bindings(session_id); \
                        CREATE INDEX idx_telegram_dm_topic_bindings_user \
                            ON telegram_dm_topic_bindings(user_id, chat_id);",
                    )?;
                }
            }

            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params!["telegram_dm_topic_schema_version", "2"],
            )?;
            Ok(())
        })
    }

    /// Enable Telegram DM topic mode for one private chat/user.
    pub fn enable_telegram_topic_mode(
        &self,
        chat_id: &str,
        user_id: &str,
        has_topics_enabled: Option<bool>,
        allows_users_to_create_topics: Option<bool>,
    ) -> Result<()> {
        self.apply_telegram_topic_migration()?;
        let now = now_ts();
        let hte = has_topics_enabled.map(|b| if b { 1i64 } else { 0i64 });
        let auct = allows_users_to_create_topics.map(|b| if b { 1i64 } else { 0i64 });
        self.execute_write(|conn| {
            conn.execute(
                "INSERT INTO telegram_dm_topic_mode ( \
                    chat_id, user_id, enabled, activated_at, updated_at, \
                    has_topics_enabled, allows_users_to_create_topics, \
                    capability_checked_at \
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?) \
                ON CONFLICT(chat_id) DO UPDATE SET \
                    user_id = excluded.user_id, \
                    enabled = 1, \
                    updated_at = excluded.updated_at, \
                    has_topics_enabled = excluded.has_topics_enabled, \
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics, \
                    capability_checked_at = excluded.capability_checked_at",
                rusqlite::params![chat_id, user_id, now, now, hte, auct, now],
            )?;
            Ok(())
        })
    }

    /// Disable Telegram DM topic mode for one private chat. No-op if tables
    /// don't exist.
    pub fn disable_telegram_topic_mode(
        &self,
        chat_id: &str,
        clear_bindings: bool,
    ) -> Result<()> {
        let chat_id = chat_id.to_string();
        self.execute_write(move |conn| {
            let res = conn.execute(
                "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? WHERE chat_id = ?",
                rusqlite::params![now_ts(), chat_id],
            );
            if res.is_err() {
                // Tables don't exist yet — nothing to disable.
                return Ok(());
            }
            if clear_bindings {
                let _ = conn.execute(
                    "DELETE FROM telegram_dm_topic_bindings WHERE chat_id = ?",
                    [&chat_id],
                );
            }
            Ok(())
        })
    }

    /// Return whether topic mode is enabled for this chat/user.
    pub fn is_telegram_topic_mode_enabled(&self, chat_id: &str, user_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let row: Option<i64> = conn
            .query_row(
                "SELECT enabled FROM telegram_dm_topic_mode WHERE chat_id = ? AND user_id = ?",
                rusqlite::params![chat_id, user_id],
                |r| r.get(0),
            )
            .ok();
        Ok(row.map(|e| e != 0).unwrap_or(false))
    }

    /// Return the session binding for a Telegram DM topic, if present.
    pub fn get_telegram_topic_binding(
        &self,
        chat_id: &str,
        thread_id: &str,
    ) -> Result<Option<Map<String, Value>>> {
        let conn = self.conn.lock().unwrap();
        let stmt = conn.prepare(
            "SELECT * FROM telegram_dm_topic_bindings WHERE chat_id = ? AND thread_id = ?",
        );
        let mut stmt = match stmt {
            Err(_) => return Ok(None),
            Ok(s) => s,
        };
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = match stmt.query(rusqlite::params![chat_id, thread_id]) {
            Err(_) => return Ok(None),
            Ok(r) => r,
        };
        match rows.next()? {
            Some(row) => Ok(Some(row_to_object(row, &names))),
            None => Ok(None),
        }
    }

    /// Bind one Telegram DM topic thread to one Hermes session. Err if the
    /// session is already linked to a *different* topic.
    pub fn bind_telegram_topic(
        &self,
        chat_id: &str,
        thread_id: &str,
        user_id: &str,
        session_key: &str,
        session_id: &str,
        managed_mode: &str,
    ) -> Result<()> {
        self.apply_telegram_topic_migration()?;
        let now = now_ts();
        let chat_id = chat_id.to_string();
        let thread_id = thread_id.to_string();
        let user_id = user_id.to_string();
        let session_key = session_key.to_string();
        let session_id = session_id.to_string();
        let managed_mode = managed_mode.to_string();
        self.execute_write(move |conn| {
            let existing: Option<(String, String)> = conn
                .query_row(
                    "SELECT chat_id, thread_id FROM telegram_dm_topic_bindings \
                     WHERE session_id = ?",
                    [&session_id],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .ok();
            if let Some((linked_chat, linked_thread)) = existing {
                if linked_chat != chat_id || linked_thread != thread_id {
                    return Err(StateError::Value(
                        "session is already linked to another Telegram topic".to_string(),
                    ));
                }
            }
            conn.execute(
                "INSERT INTO telegram_dm_topic_bindings ( \
                    chat_id, thread_id, user_id, session_key, session_id, \
                    managed_mode, linked_at, updated_at \
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET \
                    user_id = excluded.user_id, \
                    session_key = excluded.session_key, \
                    session_id = excluded.session_id, \
                    managed_mode = excluded.managed_mode, \
                    updated_at = excluded.updated_at",
                rusqlite::params![
                    chat_id,
                    thread_id,
                    user_id,
                    session_key,
                    session_id,
                    managed_mode,
                    now,
                    now,
                ],
            )?;
            Ok(())
        })
    }

    /// Return True if a session is already bound to any Telegram DM topic.
    /// Read-only: does NOT trigger the migration.
    pub fn is_telegram_session_linked_to_topic(&self, session_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let found = conn
            .query_row(
                "SELECT 1 FROM telegram_dm_topic_bindings WHERE session_id = ? LIMIT 1",
                [session_id],
                |_| Ok(()),
            )
            .is_ok();
        Ok(found)
    }

    /// List previous Telegram sessions for this user not bound to a topic.
    /// Read-only: does NOT trigger the migration. Falls back to a simpler
    /// query if the bindings table is absent.
    pub fn list_unlinked_telegram_sessions_for_user(
        &self,
        _chat_id: &str,
        user_id: &str,
        limit: i64,
    ) -> Result<Vec<Map<String, Value>>> {
        let with_bindings = "SELECT s.*, \
                COALESCE( \
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63) \
                     FROM messages m \
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL \
                     ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw, \
                COALESCE( \
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id), \
                    s.started_at) AS last_active \
            FROM sessions s \
            WHERE s.source = 'telegram' AND s.user_id = ? \
              AND NOT EXISTS ( \
                  SELECT 1 FROM telegram_dm_topic_bindings b WHERE b.session_id = s.id \
              ) \
            ORDER BY last_active DESC, s.started_at DESC LIMIT ?";
        let without_bindings = "SELECT s.*, \
                COALESCE( \
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63) \
                     FROM messages m \
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL \
                     ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw, \
                COALESCE( \
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id), \
                    s.started_at) AS last_active \
            FROM sessions s \
            WHERE s.source = 'telegram' AND s.user_id = ? \
            ORDER BY last_active DESC, s.started_at DESC LIMIT ?";

        let conn = self.conn.lock().unwrap();
        let mut sessions: Vec<Map<String, Value>> = Vec::new();
        let mut collect = |sql: &str| -> std::result::Result<bool, rusqlite::Error> {
            let mut stmt = conn.prepare(sql)?;
            let names: Vec<String> =
                stmt.column_names().iter().map(|s| s.to_string()).collect();
            let mut rows = stmt.query(rusqlite::params![user_id, limit])?;
            sessions.clear();
            while let Some(row) = rows.next()? {
                sessions.push(row_to_object(row, &names));
            }
            Ok(true)
        };
        if collect(with_bindings).is_err() {
            // telegram_dm_topic_bindings doesn't exist — fall back.
            let _ = collect(without_bindings);
        }

        for s in sessions.iter_mut() {
            let raw = s
                .remove("_preview_raw")
                .and_then(|v| v.as_str().map(|x| x.to_string()))
                .unwrap_or_default();
            let raw = raw.trim().to_string();
            // Note: matches Python's simpler preview (no "..." trailing logic
            // difference — uses 60-char cut with ellipsis when longer).
            apply_preview(s, &raw);
        }
        Ok(sessions)
    }

    // ── Space reclamation ──

    /// Run VACUUM to reclaim disk space (best-effort WAL checkpoint first).
    pub fn vacuum(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
        conn.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Idempotent auto-maintenance: prune old sessions + optional VACUUM.
    /// Never raises; on failure sets `error` in the returned object.
    pub fn maybe_auto_prune_and_vacuum(
        &self,
        retention_days: i64,
        min_interval_hours: i64,
        vacuum: bool,
        sessions_dir: Option<&Path>,
    ) -> Map<String, Value> {
        let mut result = Map::new();
        result.insert("skipped".to_string(), Value::Bool(false));
        result.insert("pruned".to_string(), Value::from(0i64));
        result.insert("vacuumed".to_string(), Value::Bool(false));

        // Closure returns (skipped, pruned, vacuumed); never mutates `result`
        // directly so we avoid a closure/borrow conflict.
        let run = || -> Result<(bool, usize, bool)> {
            let last_raw = self.get_meta("last_auto_prune")?;
            let now = now_ts();
            if let Some(raw) = last_raw {
                if let Ok(last_ts) = raw.parse::<f64>() {
                    if now - last_ts < min_interval_hours as f64 * 3600.0 {
                        return Ok((true, 0, false));
                    }
                }
            }

            let pruned = self.prune_sessions(retention_days, None, sessions_dir)?;

            let mut vacuumed = false;
            if vacuum && pruned > 0 {
                match self.vacuum() {
                    Ok(()) => vacuumed = true,
                    Err(exc) => warn!("state.db VACUUM failed: {exc}"),
                }
            }

            self.set_meta("last_auto_prune", &now.to_string())?;

            if pruned > 0 {
                info!(
                    "state.db auto-maintenance: pruned {pruned} session(s) older than \
                     {retention_days} days{}",
                    if vacuumed { " + VACUUM" } else { "" }
                );
            }
            Ok((false, pruned, vacuumed))
        };

        match run() {
            Ok((skipped, pruned, vacuumed)) => {
                result.insert("skipped".to_string(), Value::Bool(skipped));
                result.insert("pruned".to_string(), Value::from(pruned as i64));
                result.insert("vacuumed".to_string(), Value::Bool(vacuumed));
            }
            Err(exc) => {
                warn!("state.db auto-maintenance failed: {exc}");
                result.insert("error".to_string(), Value::String(exc.to_string()));
            }
        }
        result
    }
}

/// Raw row fetched by `get_messages_as_conversation`.
struct RawConvRow {
    role: String,
    content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<String>,
    tool_name: Option<String>,
    finish_reason: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    reasoning_details: Option<String>,
    codex_reasoning_items: Option<String>,
    codex_message_items: Option<String>,
}

/// Field bundle for [`SessionDB::append_message`]. All optional fields default
/// to None/null. `content` and `tool_calls` accept any JSON value.
#[derive(Default)]
pub struct MessageInput {
    pub role: String,
    pub content: Value,
    pub tool_name: Option<String>,
    pub tool_calls: Value,
    pub tool_call_id: Option<String>,
    pub token_count: Option<i64>,
    pub finish_reason: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Value,
    pub codex_reasoning_items: Value,
    pub codex_message_items: Value,
}

impl MessageInput {
    /// Construct a minimal text message with the given role + string content.
    pub fn text(role: &str, content: &str) -> Self {
        MessageInput {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            ..Default::default()
        }
    }
}

/// Field bundle for [`SessionDB::update_token_counts`].
#[derive(Default, Clone)]
pub struct TokenUpdate {
    pub input_tokens: i64,
    pub output_tokens: i64,
    /// Backfilled into `sessions.model` only if not already set (COALESCE).
    pub model: Option<String>,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub reasoning_tokens: i64,
    pub estimated_cost_usd: Option<f64>,
    pub actual_cost_usd: Option<f64>,
    pub cost_status: Option<String>,
    pub cost_source: Option<String>,
    pub pricing_version: Option<String>,
    pub billing_provider: Option<String>,
    pub billing_base_url: Option<String>,
    pub billing_mode: Option<String>,
    pub api_call_count: i64,
    pub absolute: bool,
}

// ── module-private helpers ──

/// Serialize a JSON value to a string only if it is "truthy" in the Python
/// sense (non-null, non-empty list/dict/string, non-zero number). Mirrors the
/// `json.dumps(x) if x else None` idiom used throughout the Python source.
fn json_if_truthy(v: &Value) -> Option<String> {
    let truthy = match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    };
    if truthy {
        Some(serde_json::to_string(v).unwrap_or_default())
    } else {
        None
    }
}

/// Apply the preview-string logic shared by list/rich helpers: take up to 60
/// chars, append "..." when the raw was longer than 60.
fn apply_preview(s: &mut Map<String, Value>, raw: &str) {
    if raw.is_empty() {
        s.insert("preview".to_string(), Value::String(String::new()));
        return;
    }
    let chars: Vec<char> = raw.chars().collect();
    let text: String = chars.iter().take(60).collect();
    let preview = if chars.len() > 60 {
        format!("{text}...")
    } else {
        text
    };
    s.insert("preview".to_string(), Value::String(preview));
}

/// Extract an owned String from an optional JSON value if it is a string.
fn str_opt(v: Option<&Value>) -> Option<String> {
    v.and_then(|v| v.as_str().map(|s| s.to_string()))
}

/// Extract an i64 from an optional JSON value if it is an integer-like number.
fn int_opt(v: Option<&Value>) -> Option<i64> {
    v.and_then(|v| v.as_i64())
}

/// Clone a `ToSql` trait object into an owned `Box<dyn ToSql>` by snapshotting
/// its `ToSqlOutput`. Used to duplicate WHERE params for the recursive-CTE
/// query in `list_sessions_rich`.
fn sql_clone(v: &dyn ToSql) -> ClonedSql {
    use rusqlite::types::{ToSqlOutput, Value as RV, ValueRef as RVR};
    let out = v.to_sql();
    match out {
        Ok(ToSqlOutput::Borrowed(r)) => match r {
            RVR::Null => ClonedSql(RV::Null),
            RVR::Integer(i) => ClonedSql(RV::Integer(i)),
            RVR::Real(f) => ClonedSql(RV::Real(f)),
            RVR::Text(t) => ClonedSql(RV::Text(String::from_utf8_lossy(t).into_owned())),
            RVR::Blob(b) => ClonedSql(RV::Blob(b.to_vec())),
        },
        Ok(ToSqlOutput::Owned(v)) => ClonedSql(v),
        _ => ClonedSql(RV::Null),
    }
}

/// Owned SQL value wrapper so it can live in a `Vec<Box<dyn ToSql>>`.
struct ClonedSql(rusqlite::types::Value);

impl ToSql for ClonedSql {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(rusqlite::types::ToSqlOutput::Borrowed(
            (&self.0).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mkdb() -> SessionDB {
        let dir = std::env::temp_dir();
        let unique = format!(
            "hermes_state_test_{}_{}.db",
            std::process::id(),
            now_ts().to_bits()
        );
        let path = dir.join(unique);
        let _ = std::fs::remove_file(&path);
        SessionDB::open(path).unwrap()
    }

    #[test]
    fn schema_version_initialized() {
        let db = mkdb();
        let conn = db.conn.lock().unwrap();
        let v: i64 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn create_and_get_session() {
        let db = mkdb();
        db.create_session("sess1", "cli", Some("gpt"), None, None, None, None)
            .unwrap();
        let s = db.get_session("sess1").unwrap().unwrap();
        assert_eq!(s.get("id").unwrap(), "sess1");
        assert_eq!(s.get("source").unwrap(), "cli");
        assert_eq!(s.get("model").unwrap(), "gpt");
        assert_eq!(s.get("message_count").unwrap(), 0);
        assert!(db.get_session("missing").unwrap().is_none());
    }

    #[test]
    fn append_message_increments_counts() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.append_message("s", &MessageInput::text("user", "hello"))
            .unwrap();
        let mut tool_msg = MessageInput::text("assistant", "");
        tool_msg.tool_calls = json!([{"id": "1"}, {"id": "2"}]);
        db.append_message("s", &tool_msg).unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("message_count").unwrap(), 2);
        assert_eq!(s.get("tool_call_count").unwrap(), 2);
        assert_eq!(db.message_count(Some("s")).unwrap(), 2);
    }

    #[test]
    fn content_encode_decode_multimodal() {
        let parts = json!([{"type": "text", "text": "hi"}, {"type": "image_url"}]);
        let enc = SessionDB::encode_content(&parts).unwrap();
        assert!(enc.starts_with(CONTENT_JSON_PREFIX));
        let dec = SessionDB::decode_content(Some(&enc));
        assert_eq!(dec, parts);
        // Plain string passes through.
        assert_eq!(
            SessionDB::decode_content(Some("plain")),
            Value::String("plain".to_string())
        );
    }

    #[test]
    fn get_messages_roundtrip() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        let mut m = MessageInput::text("assistant", "answer");
        m.tool_calls = json!([{"id": "abc", "fn": "x"}]);
        db.append_message("s", &m).unwrap();
        let msgs = db.get_messages("s").unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].get("content").unwrap(), "answer");
        assert!(msgs[0].get("tool_calls").unwrap().is_array());
    }

    #[test]
    fn sanitize_title_rules() {
        assert_eq!(SessionDB::sanitize_title(None).unwrap(), None);
        assert_eq!(SessionDB::sanitize_title(Some("   ")).unwrap(), None);
        assert_eq!(
            SessionDB::sanitize_title(Some("  a\t\nb  ")).unwrap(),
            Some("a b".to_string())
        );
        // Control + zero-width chars stripped.
        assert_eq!(
            SessionDB::sanitize_title(Some("a\u{200b}b")).unwrap(),
            Some("ab".to_string())
        );
        let long = "x".repeat(MAX_TITLE_LENGTH + 1);
        assert!(SessionDB::sanitize_title(Some(&long)).is_err());
    }

    #[test]
    fn title_uniqueness_and_lineage() {
        let db = mkdb();
        db.create_session("a", "cli", None, None, None, None, None)
            .unwrap();
        db.create_session("b", "cli", None, None, None, None, None)
            .unwrap();
        assert!(db.set_session_title("a", Some("Proj")).unwrap());
        // Conflict on duplicate title.
        assert!(db.set_session_title("b", Some("Proj")).is_err());
        // Next in lineage.
        assert_eq!(db.get_next_title_in_lineage("Proj").unwrap(), "Proj #2");
    }

    #[test]
    fn fts5_sanitize() {
        // Hyphenated term gets quoted.
        assert_eq!(SessionDB::sanitize_fts5_query("chat-send"), "\"chat-send\"");
        // Dangling boolean removed.
        assert_eq!(SessionDB::sanitize_fts5_query("hello AND"), "hello");
        // Special chars stripped.
        let out = SessionDB::sanitize_fts5_query("foo(bar)");
        assert!(!out.contains('(') && !out.contains(')'));
    }

    #[test]
    fn search_messages_basic() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.append_message("s", &MessageInput::text("user", "docker deployment guide"))
            .unwrap();
        db.append_message("s", &MessageInput::text("assistant", "use kubernetes"))
            .unwrap();
        let results = db
            .search_messages("docker", None, None, None, 20, 0)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].get("snippet").is_some());
        // content removed from results.
        assert!(results[0].get("content").is_none());
    }

    #[test]
    fn replace_messages_resets_counts() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.append_message("s", &MessageInput::text("user", "one"))
            .unwrap();
        let new = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b", "tool_calls": [{"id": "1"}]}),
        ];
        db.replace_messages("s", &new).unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("message_count").unwrap(), 2);
        assert_eq!(s.get("tool_call_count").unwrap(), 1);
    }

    #[test]
    fn meta_kv_roundtrip() {
        let db = mkdb();
        assert_eq!(db.get_meta("k").unwrap(), None);
        db.set_meta("k", "v").unwrap();
        assert_eq!(db.get_meta("k").unwrap(), Some("v".to_string()));
        db.set_meta("k", "v2").unwrap();
        assert_eq!(db.get_meta("k").unwrap(), Some("v2".to_string()));
    }

    #[test]
    fn end_and_reopen_session() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.end_session("s", "done").unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("end_reason").unwrap(), "done");
        // First end_reason wins.
        db.end_session("s", "other").unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("end_reason").unwrap(), "done");
        db.reopen_session("s").unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert!(s.get("end_reason").unwrap().is_null());
    }

    #[test]
    fn delete_orphans_children() {
        let db = mkdb();
        db.create_session("parent", "cli", None, None, None, None, None)
            .unwrap();
        db.create_session("child", "cli", None, None, None, None, Some("parent"))
            .unwrap();
        assert!(db.delete_session("parent", None).unwrap());
        let child = db.get_session("child").unwrap().unwrap();
        assert!(child.get("parent_session_id").unwrap().is_null());
        assert!(!db.delete_session("nope", None).unwrap());
    }

    #[test]
    fn resolve_session_id_prefix() {
        let db = mkdb();
        db.create_session("abcdef", "cli", None, None, None, None, None)
            .unwrap();
        assert_eq!(
            db.resolve_session_id("abc").unwrap(),
            Some("abcdef".to_string())
        );
        assert_eq!(db.resolve_session_id("abcdef").unwrap(), Some("abcdef".to_string()));
        // Ambiguous.
        db.create_session("abcxyz", "cli", None, None, None, None, None)
            .unwrap();
        assert_eq!(db.resolve_session_id("abc").unwrap(), None);
    }

    #[test]
    fn token_counts_increment_and_absolute() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        let mut upd = TokenUpdate {
            input_tokens: 10,
            output_tokens: 5,
            api_call_count: 1,
            ..Default::default()
        };
        db.update_token_counts("s", &upd).unwrap();
        db.update_token_counts("s", &upd).unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("input_tokens").unwrap(), 20);
        assert_eq!(s.get("api_call_count").unwrap(), 2);
        // Absolute set.
        upd.absolute = true;
        upd.input_tokens = 100;
        upd.api_call_count = 7;
        db.update_token_counts("s", &upd).unwrap();
        let s = db.get_session("s").unwrap().unwrap();
        assert_eq!(s.get("input_tokens").unwrap(), 100);
        assert_eq!(s.get("api_call_count").unwrap(), 7);
    }

    #[test]
    fn telegram_topic_enable_disable() {
        let db = mkdb();
        db.create_session("s", "telegram", None, None, None, Some("u1"), None)
            .unwrap();
        assert!(!db.is_telegram_topic_mode_enabled("c1", "u1").unwrap());
        db.enable_telegram_topic_mode("c1", "u1", Some(true), Some(false))
            .unwrap();
        assert!(db.is_telegram_topic_mode_enabled("c1", "u1").unwrap());
        db.bind_telegram_topic("c1", "t1", "u1", "key", "s", "auto")
            .unwrap();
        assert!(db.is_telegram_session_linked_to_topic("s").unwrap());
        // Binding same session to another topic fails.
        assert!(db.bind_telegram_topic("c1", "t2", "u1", "key", "s", "auto").is_err());
        let binding = db.get_telegram_topic_binding("c1", "t1").unwrap().unwrap();
        assert_eq!(binding.get("session_id").unwrap(), "s");
        db.disable_telegram_topic_mode("c1", true).unwrap();
        assert!(!db.is_telegram_topic_mode_enabled("c1", "u1").unwrap());
    }

    #[test]
    fn list_sessions_rich_preview() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.append_message("s", &MessageInput::text("user", "first user message here"))
            .unwrap();
        let rows = db
            .list_sessions_rich(None, None, 20, 0, false, true, false)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("preview").unwrap(), "first user message here");
        assert!(rows[0].get("last_active").is_some());
    }

    #[test]
    fn conversation_dedup_and_sanitize() {
        let db = mkdb();
        db.create_session("s", "cli", None, None, None, None, None)
            .unwrap();
        db.append_message("s", &MessageInput::text("user", "hi there"))
            .unwrap();
        let convo = db.get_messages_as_conversation("s", false).unwrap();
        assert_eq!(convo.len(), 1);
        assert_eq!(convo[0].get("role").unwrap(), "user");
        assert_eq!(convo[0].get("content").unwrap(), "hi there");
    }
}
