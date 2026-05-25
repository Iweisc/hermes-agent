use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::Serialize;
use serde_json::Value;

use crate::{HermesContext, HermesError};

const SCHEMA_VERSION: i64 = 11;
const CONTENT_JSON_PREFIX: &str = "__json__:";
const MAX_TITLE_LENGTH: usize = 100;

const FTS_SQL: &str = r#"
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

const SCHEMA_SQL: &str = r#"
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

#[derive(Debug, Clone)]
pub struct SessionCreate {
    pub id: String,
    pub source: String,
    pub user_id: Option<String>,
    pub model: Option<String>,
    pub model_config: Option<Value>,
    pub system_prompt: Option<String>,
    pub parent_session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionRecord {
    pub id: String,
    pub source: String,
    pub user_id: Option<String>,
    pub model: Option<String>,
    pub model_config: Option<Value>,
    pub system_prompt: Option<String>,
    pub parent_session_id: Option<String>,
    pub started_at: f64,
    pub ended_at: Option<f64>,
    pub end_reason: Option<String>,
    pub message_count: i64,
    pub tool_call_count: i64,
    pub title: Option<String>,
    pub api_call_count: i64,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub source: String,
    pub model: Option<String>,
    pub title: Option<String>,
    pub started_at: f64,
    pub ended_at: Option<f64>,
    pub message_count: i64,
    pub last_active: Option<f64>,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct SessionSearchRow {
    pub id: String,
    pub source: String,
    pub model: Option<String>,
    pub title: Option<String>,
    pub started_at: f64,
    pub ended_at: Option<f64>,
    pub end_reason: Option<String>,
    pub message_count: i64,
    pub tool_call_count: i64,
    pub last_active: f64,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct MessageAppend {
    pub role: String,
    pub content: Option<Value>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Value>,
    pub tool_name: Option<String>,
    pub token_count: Option<i64>,
    pub finish_reason: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<Value>,
    pub codex_reasoning_items: Option<Value>,
    pub codex_message_items: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageRecord {
    pub id: i64,
    pub session_id: String,
    pub role: String,
    pub content: Option<Value>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Value>,
    pub tool_name: Option<String>,
    pub timestamp: f64,
    pub token_count: Option<i64>,
    pub finish_reason: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<Value>,
    pub codex_reasoning_items: Option<Value>,
    pub codex_message_items: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchContextMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct MessageSearchRow {
    pub id: i64,
    pub session_id: String,
    pub role: String,
    pub snippet: String,
    pub timestamp: f64,
    pub tool_name: Option<String>,
    pub source: String,
    pub model: Option<String>,
    pub session_started: f64,
    pub context: Vec<SearchContextMessage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportedSession {
    #[serde(flatten)]
    pub session: SessionRecord,
    pub messages: Vec<MessageRecord>,
}

pub struct SessionStore {
    path: PathBuf,
    connection: Connection,
}

impl HermesContext {
    pub fn state_db_path(&self) -> PathBuf {
        self.hermes_home().join("state.db")
    }

    pub fn open_session_store(&self) -> Result<SessionStore, HermesError> {
        SessionStore::open(self.state_db_path())
    }
}

impl SessionStore {
    pub fn open(path: PathBuf) -> Result<Self, HermesError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| HermesError::Io {
                action: "creating",
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let connection = Connection::open(&path).map_err(|source| HermesError::State {
            action: "opening state database",
            detail: source.to_string(),
        })?;
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(state_err("enabling WAL mode"))?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(state_err("enabling foreign keys"))?;
        connection
            .execute_batch(SCHEMA_SQL)
            .map_err(state_err("initializing schema"))?;
        connection
            .execute_batch(FTS_SQL)
            .map_err(state_err("initializing FTS schema"))?;
        connection
            .execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                [SCHEMA_VERSION],
            )
            .map_err(state_err("seeding schema version"))?;
        connection
            .execute("UPDATE schema_version SET version = ?", [SCHEMA_VERSION])
            .map_err(state_err("updating schema version"))?;
        connection
            .execute(
                "INSERT OR REPLACE INTO messages_fts(rowid, content)
                 SELECT id,
                        COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '')
                 FROM messages",
                [],
            )
            .map_err(state_err("backfilling FTS index"))?;

        Ok(Self { path, connection })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn session_count(&self) -> Result<i64, HermesError> {
        self.connection
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .map_err(state_err("counting sessions"))
    }

    pub fn create_session(&self, create: &SessionCreate) -> Result<String, HermesError> {
        let model_config = create
            .model_config
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing session model_config",
                detail: source.to_string(),
            })?;
        self.connection
            .execute(
                "INSERT OR IGNORE INTO sessions (id, source, user_id, model, model_config, system_prompt, parent_session_id, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    create.id,
                    create.source,
                    create.user_id,
                    create.model,
                    model_config,
                    create.system_prompt,
                    create.parent_session_id,
                    now_ts(),
                ],
            )
            .map_err(state_err("creating session"))?;
        Ok(create.id.clone())
    }

    pub fn end_session(&self, session_id: &str, end_reason: &str) -> Result<(), HermesError> {
        self.connection
            .execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                params![now_ts(), end_reason, session_id],
            )
            .map_err(state_err("ending session"))?;
        Ok(())
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>, HermesError> {
        self.connection
            .query_row(
                "SELECT id, source, user_id, model, model_config, system_prompt, parent_session_id, started_at, ended_at, end_reason, message_count, tool_call_count, title, api_call_count FROM sessions WHERE id = ?",
                [session_id],
                |row| map_session_record(row),
            )
            .optional()
            .map_err(state_err("loading session"))
    }

    pub fn update_session_runtime(
        &self,
        session_id: &str,
        model: Option<&str>,
        model_config: Option<&Value>,
    ) -> Result<bool, HermesError> {
        let model_config = model_config
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing session model_config",
                detail: source.to_string(),
            })?;
        let rowcount = self
            .connection
            .execute(
                "UPDATE sessions SET model = ?, model_config = ? WHERE id = ?",
                params![model, model_config, session_id],
            )
            .map_err(state_err("updating session runtime"))?;
        Ok(rowcount > 0)
    }

    pub fn resolve_session_id(
        &self,
        session_id_or_prefix: &str,
    ) -> Result<Option<String>, HermesError> {
        if let Some(exact) = self.get_session(session_id_or_prefix)? {
            return Ok(Some(exact.id));
        }

        let escaped = escape_like(session_id_or_prefix);
        let mut statement = self
            .connection
            .prepare(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
            )
            .map_err(state_err("preparing session prefix query"))?;
        let rows = statement
            .query_map([format!("{escaped}%")], |row| row.get::<_, String>(0))
            .map_err(state_err("querying session prefix"))?;
        let matches = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting session prefix matches"))?;
        Ok((matches.len() == 1).then(|| matches[0].clone()))
    }

    pub fn sanitize_title(title: Option<&str>) -> Result<Option<String>, HermesError> {
        let Some(title) = title else {
            return Ok(None);
        };
        if title.is_empty() {
            return Ok(None);
        }

        let mut filtered = String::new();
        for ch in title.chars() {
            let cp = ch as u32;
            let remove_ascii_control = matches!(cp, 0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x7F);
            let remove_problematic_unicode = matches!(
                cp,
                0x200B..=0x200F
                    | 0x2028..=0x202E
                    | 0x2060..=0x2069
                    | 0xFEFF
                    | 0xFFFC
                    | 0xFFF9..=0xFFFB
            );
            if !(remove_ascii_control || remove_problematic_unicode) {
                filtered.push(ch);
            }
        }

        let collapsed = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            return Ok(None);
        }
        if collapsed.chars().count() > MAX_TITLE_LENGTH {
            return Err(HermesError::State {
                action: "validating session title",
                detail: format!(
                    "Title too long ({} chars, max {MAX_TITLE_LENGTH})",
                    collapsed.chars().count()
                ),
            });
        }
        Ok(Some(collapsed))
    }

    pub fn set_session_title(&self, session_id: &str, title: &str) -> Result<bool, HermesError> {
        let title = Self::sanitize_title(Some(title))?;
        if let Some(ref title) = title {
            let conflict: Option<String> = self
                .connection
                .query_row(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    params![title, session_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(state_err("checking title uniqueness"))?;
            if let Some(conflict_id) = conflict {
                return Err(HermesError::State {
                    action: "setting session title",
                    detail: format!("Title '{title}' is already in use by session {conflict_id}"),
                });
            }
        }

        let rowcount = self
            .connection
            .execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                params![title, session_id],
            )
            .map_err(state_err("updating session title"))?;
        Ok(rowcount > 0)
    }

    pub fn get_session_title(&self, session_id: &str) -> Result<Option<String>, HermesError> {
        self.connection
            .query_row(
                "SELECT title FROM sessions WHERE id = ?",
                [session_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(state_err("loading session title"))
    }

    pub fn get_session_by_title(&self, title: &str) -> Result<Option<SessionRecord>, HermesError> {
        self.connection
            .query_row(
                "SELECT id, source, user_id, model, model_config, system_prompt, parent_session_id, started_at, ended_at, end_reason, message_count, tool_call_count, title, api_call_count FROM sessions WHERE title = ?",
                [title],
                |row| map_session_record(row),
            )
            .optional()
            .map_err(state_err("loading session by title"))
    }

    pub fn resolve_session_by_title(&self, title: &str) -> Result<Option<String>, HermesError> {
        let exact = self.get_session_by_title(title)?;
        let escaped = escape_like(title);
        let mut statement = self
            .connection
            .prepare(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
            )
            .map_err(state_err("preparing numbered title query"))?;
        let numbered = statement
            .query_map([format!("{escaped} #%")], |row| row.get::<_, String>(0))
            .map_err(state_err("querying numbered titles"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting numbered titles"))?;

        if let Some(latest) = numbered.first() {
            Ok(Some(latest.clone()))
        } else {
            Ok(exact.map(|row| row.id))
        }
    }

    pub fn get_next_title_in_lineage(&self, base_title: &str) -> Result<String, HermesError> {
        let base = strip_title_suffix(base_title);
        let escaped = escape_like(&base);
        let mut statement = self
            .connection
            .prepare("SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'")
            .map_err(state_err("preparing lineage title query"))?;
        let existing = statement
            .query_map(params![base, format!("{escaped} #%")], |row| {
                row.get::<_, Option<String>>(0)
            })
            .map_err(state_err("querying lineage titles"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting lineage titles"))?;

        if existing.is_empty() {
            return Ok(base);
        }

        let mut max_num = 1_u64;
        for title in existing.into_iter().flatten() {
            if let Some(num) = parse_title_suffix(&title) {
                max_num = max_num.max(num);
            }
        }
        Ok(format!("{base} #{}", max_num + 1))
    }

    pub fn list_sessions(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SessionSummary>, HermesError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, source, model, title, started_at, ended_at, message_count FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
            )
            .map_err(state_err("preparing session list"))?;
        let rows = statement
            .query_map(params![limit.max(0), offset.max(0)], |row| {
                Ok(SessionSummary {
                    id: row.get(0)?,
                    source: row.get(1)?,
                    model: row.get(2)?,
                    title: row.get(3)?,
                    started_at: row.get(4)?,
                    ended_at: row.get(5)?,
                    message_count: row.get(6)?,
                    last_active: None,
                    preview: String::new(),
                })
            })
            .map_err(state_err("listing sessions"))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting session list"))
    }

    pub fn search_sessions(
        &self,
        source: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SessionSearchRow>, HermesError> {
        let sql = if source.is_some() {
            "SELECT s.id, s.source, s.model, s.title, s.started_at, s.ended_at, s.end_reason, s.message_count, s.tool_call_count,
                    COALESCE(m.last_active, s.started_at) AS last_active,
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(msg.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages msg
                         WHERE msg.session_id = s.id AND msg.role = 'user' AND msg.content IS NOT NULL
                         ORDER BY msg.timestamp, msg.id LIMIT 1),
                        ''
                    ) AS preview
             FROM sessions s
             LEFT JOIN (
                 SELECT session_id, MAX(timestamp) AS last_active
                 FROM messages
                 GROUP BY session_id
             ) m ON m.session_id = s.id
             WHERE s.source = ?
             ORDER BY last_active DESC, s.started_at DESC, s.id DESC
             LIMIT ? OFFSET ?"
        } else {
            "SELECT s.id, s.source, s.model, s.title, s.started_at, s.ended_at, s.end_reason, s.message_count, s.tool_call_count,
                    COALESCE(m.last_active, s.started_at) AS last_active,
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(msg.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages msg
                         WHERE msg.session_id = s.id AND msg.role = 'user' AND msg.content IS NOT NULL
                         ORDER BY msg.timestamp, msg.id LIMIT 1),
                        ''
                    ) AS preview
             FROM sessions s
             LEFT JOIN (
                 SELECT session_id, MAX(timestamp) AS last_active
                 FROM messages
                 GROUP BY session_id
             ) m ON m.session_id = s.id
             ORDER BY last_active DESC, s.started_at DESC, s.id DESC
             LIMIT ? OFFSET ?"
        };
        let mut statement = self
            .connection
            .prepare(sql)
            .map_err(state_err("preparing session search"))?;
        let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SessionSearchRow> {
            let raw_preview: String = row.get(10)?;
            Ok(SessionSearchRow {
                id: row.get(0)?,
                source: row.get(1)?,
                model: row.get(2)?,
                title: row.get(3)?,
                started_at: row.get(4)?,
                ended_at: row.get(5)?,
                end_reason: row.get(6)?,
                message_count: row.get(7)?,
                tool_call_count: row.get(8)?,
                last_active: row.get(9)?,
                preview: format_preview(&raw_preview),
            })
        };
        let rows = if let Some(source) = source {
            statement.query_map(params![source, limit.max(0), offset.max(0)], mapper)
        } else {
            statement.query_map(params![limit.max(0), offset.max(0)], mapper)
        }
        .map_err(state_err("querying session search"))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting session search"))
    }

    pub fn append_message(
        &self,
        session_id: &str,
        message: &MessageAppend,
    ) -> Result<i64, HermesError> {
        let content = message.content.as_ref().map(encode_content).transpose()?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing tool_calls",
                detail: source.to_string(),
            })?;
        let reasoning_details = message
            .reasoning_details
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing reasoning_details",
                detail: source.to_string(),
            })?;
        let codex_reasoning_items = message
            .codex_reasoning_items
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing codex_reasoning_items",
                detail: source.to_string(),
            })?;
        let codex_message_items = message
            .codex_message_items
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|source| HermesError::State {
                action: "serializing codex_message_items",
                detail: source.to_string(),
            })?;
        let tx = self
            .connection
            .unchecked_transaction()
            .map_err(state_err("opening message transaction"))?;
        tx.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_count, finish_reason, reasoning, reasoning_content, reasoning_details, codex_reasoning_items, codex_message_items) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                session_id,
                message.role,
                content,
                message.tool_call_id,
                tool_calls,
                message.tool_name,
                now_ts(),
                message.token_count,
                message.finish_reason,
                message.reasoning,
                message.reasoning_content,
                reasoning_details,
                codex_reasoning_items,
                codex_message_items,
            ],
        )
        .map_err(state_err("inserting message"))?;
        let message_id = tx.last_insert_rowid();
        let tool_call_count = message.tool_calls.as_ref().map_or(0_i64, count_tool_calls);
        if tool_call_count > 0 {
            tx.execute(
                "UPDATE sessions SET message_count = message_count + 1, tool_call_count = tool_call_count + ? WHERE id = ?",
                params![tool_call_count, session_id],
            )
            .map_err(state_err("updating message counters"))?;
        } else {
            tx.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                params![session_id],
            )
            .map_err(state_err("updating message count"))?;
        }
        tx.commit()
            .map_err(state_err("committing message transaction"))?;
        Ok(message_id)
    }

    pub fn get_messages(&self, session_id: &str) -> Result<Vec<MessageRecord>, HermesError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_count, finish_reason, reasoning, reasoning_content, reasoning_details, codex_reasoning_items, codex_message_items FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            )
            .map_err(state_err("preparing message query"))?;
        let rows = statement
            .query_map([session_id], |row| map_message_record(row))
            .map_err(state_err("loading messages"))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting messages"))
    }

    pub fn search_messages(
        &self,
        query: &str,
        source_filter: Option<&[String]>,
        exclude_sources: Option<&[String]>,
        role_filter: Option<&[String]>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MessageSearchRow>, HermesError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let query = sanitize_fts5_query(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let mut where_clauses = vec![String::from("messages_fts MATCH ?")];
        let mut sql_params = vec![SqlValue::Text(query)];

        if let Some(source_filter) = source_filter.filter(|items| !items.is_empty()) {
            where_clauses.push(format!(
                "s.source IN ({})",
                vec!["?"; source_filter.len()].join(",")
            ));
            sql_params.extend(source_filter.iter().cloned().map(SqlValue::Text));
        }
        if let Some(exclude_sources) = exclude_sources.filter(|items| !items.is_empty()) {
            where_clauses.push(format!(
                "s.source NOT IN ({})",
                vec!["?"; exclude_sources.len()].join(",")
            ));
            sql_params.extend(exclude_sources.iter().cloned().map(SqlValue::Text));
        }
        if let Some(role_filter) = role_filter.filter(|items| !items.is_empty()) {
            where_clauses.push(format!(
                "m.role IN ({})",
                vec!["?"; role_filter.len()].join(",")
            ));
            sql_params.extend(role_filter.iter().cloned().map(SqlValue::Text));
        }

        sql_params.push(SqlValue::Integer(limit.max(0)));
        sql_params.push(SqlValue::Integer(offset.max(0)));

        let sql = format!(
            "SELECT
                 m.id,
                 m.session_id,
                 m.role,
                 snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                 m.timestamp,
                 m.tool_name,
                 s.source,
                 s.model,
                 s.started_at AS session_started
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             JOIN sessions s ON s.id = m.session_id
             WHERE {}
             ORDER BY rank
             LIMIT ? OFFSET ?",
            where_clauses.join(" AND ")
        );

        let mut statement = self
            .connection
            .prepare(&sql)
            .map_err(state_err("preparing message search"))?;
        let rows = statement
            .query_map(params_from_iter(sql_params), |row| {
                Ok(MessageSearchRow {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    snippet: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    timestamp: row.get(4)?,
                    tool_name: row.get(5)?,
                    source: row.get(6)?,
                    model: row.get(7)?,
                    session_started: row.get(8)?,
                    context: Vec::new(),
                })
            })
            .map_err(state_err("querying message search"))?;
        let mut results = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting message search"))?;

        for result in &mut results {
            result.context = self.message_search_context(result.id)?;
        }
        Ok(results)
    }

    pub fn message_count(&self, session_id: Option<&str>) -> Result<i64, HermesError> {
        match session_id {
            Some(session_id) => self
                .connection
                .query_row(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    [session_id],
                    |row| row.get(0),
                )
                .map_err(state_err("counting session messages")),
            None => self
                .connection
                .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
                .map_err(state_err("counting messages")),
        }
    }

    pub fn session_counts_by_source(&self) -> Result<Vec<(String, i64)>, HermesError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT source, COUNT(*) AS session_count
                 FROM sessions
                 GROUP BY source
                 ORDER BY source",
            )
            .map_err(state_err("preparing session counts by source"))?;
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(state_err("querying session counts by source"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting session counts by source"))
    }

    pub fn export_session(&self, session_id: &str) -> Result<Option<ExportedSession>, HermesError> {
        let Some(session) = self.get_session(session_id)? else {
            return Ok(None);
        };
        let messages = self.get_messages(session_id)?;
        Ok(Some(ExportedSession { session, messages }))
    }

    pub fn export_all(&self, source: Option<&str>) -> Result<Vec<ExportedSession>, HermesError> {
        let ids = if let Some(source) = source {
            let mut statement = self
                .connection
                .prepare("SELECT id FROM sessions WHERE source = ? ORDER BY started_at DESC")
                .map_err(state_err("preparing source export query"))?;
            statement
                .query_map([source], |row| row.get::<_, String>(0))
                .map_err(state_err("querying source export ids"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(state_err("collecting source export ids"))?
        } else {
            let mut statement = self
                .connection
                .prepare("SELECT id FROM sessions ORDER BY started_at DESC")
                .map_err(state_err("preparing export query"))?;
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(state_err("querying export ids"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(state_err("collecting export ids"))?
        };

        let mut exports = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(exported) = self.export_session(&id)? {
                exports.push(exported);
            }
        }
        Ok(exports)
    }

    pub fn prune_sessions(
        &self,
        older_than_days: u64,
        source: Option<&str>,
    ) -> Result<Vec<String>, HermesError> {
        let cutoff = now_ts() - (older_than_days as f64 * 86_400.0);
        let tx = self
            .connection
            .unchecked_transaction()
            .map_err(state_err("opening prune sessions transaction"))?;
        let ids = if let Some(source) = source {
            let mut statement = tx
                .prepare(
                    "SELECT id
                     FROM sessions
                     WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?",
                )
                .map_err(state_err("preparing source prune query"))?;
            statement
                .query_map(params![cutoff, source], |row| row.get::<_, String>(0))
                .map_err(state_err("querying source prune ids"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(state_err("collecting source prune ids"))?
        } else {
            let mut statement = tx
                .prepare(
                    "SELECT id
                     FROM sessions
                     WHERE started_at < ? AND ended_at IS NOT NULL",
                )
                .map_err(state_err("preparing prune query"))?;
            statement
                .query_map([cutoff], |row| row.get::<_, String>(0))
                .map_err(state_err("querying prune ids"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(state_err("collecting prune ids"))?
        };

        if ids.is_empty() {
            tx.rollback()
                .map_err(state_err("rolling back empty prune transaction"))?;
            return Ok(Vec::new());
        }

        let placeholders = vec!["?"; ids.len()].join(",");
        tx.execute(
            &format!(
                "UPDATE sessions
                 SET parent_session_id = NULL
                 WHERE parent_session_id IN ({placeholders})"
            ),
            params_from_iter(ids.iter()),
        )
        .map_err(state_err("orphaning pruned child sessions"))?;

        for session_id in &ids {
            tx.execute("DELETE FROM messages WHERE session_id = ?", [session_id])
                .map_err(state_err("deleting pruned session messages"))?;
            tx.execute("DELETE FROM sessions WHERE id = ?", [session_id])
                .map_err(state_err("deleting pruned sessions"))?;
        }

        tx.commit()
            .map_err(state_err("committing prune sessions transaction"))?;
        Ok(ids)
    }

    pub fn clear_messages(&self, session_id: &str) -> Result<(), HermesError> {
        let tx = self
            .connection
            .unchecked_transaction()
            .map_err(state_err("opening clear messages transaction"))?;
        tx.execute("DELETE FROM messages WHERE session_id = ?", [session_id])
            .map_err(state_err("deleting messages"))?;
        tx.execute(
            "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
            [session_id],
        )
        .map_err(state_err("resetting message counters"))?;
        tx.commit()
            .map_err(state_err("committing clear messages transaction"))?;
        Ok(())
    }

    pub fn trim_last_exchange(&self, session_id: &str) -> Result<usize, HermesError> {
        let tx = self
            .connection
            .unchecked_transaction()
            .map_err(state_err("opening undo transaction"))?;
        let mut statement = tx
            .prepare(
                "SELECT id, role, tool_calls
                 FROM messages
                 WHERE session_id = ?
                 ORDER BY timestamp DESC, id DESC",
            )
            .map_err(state_err("preparing undo message query"))?;
        let rows = statement
            .query_map([session_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(state_err("querying undo messages"))?;

        let mut removed_ids = Vec::new();
        let mut removed_tool_calls = 0_i64;
        for row in rows {
            let (message_id, role, tool_calls) = row.map_err(state_err("collecting undo rows"))?;
            if matches!(role.as_str(), "assistant" | "tool") {
                removed_ids.push(message_id);
                removed_tool_calls += parse_tool_call_count(tool_calls.as_deref())?;
                continue;
            }
            if role == "user" {
                removed_ids.push(message_id);
                removed_tool_calls += parse_tool_call_count(tool_calls.as_deref())?;
            }
            break;
        }

        drop(statement);

        if removed_ids.is_empty() {
            tx.rollback()
                .map_err(state_err("rolling back empty undo transaction"))?;
            return Ok(0);
        }

        for message_id in &removed_ids {
            tx.execute("DELETE FROM messages WHERE id = ?", [message_id])
                .map_err(state_err("deleting undone messages"))?;
        }
        tx.execute(
            "UPDATE sessions
             SET message_count = MAX(message_count - ?, 0),
                 tool_call_count = MAX(tool_call_count - ?, 0)
             WHERE id = ?",
            params![removed_ids.len() as i64, removed_tool_calls, session_id],
        )
        .map_err(state_err("updating undo counters"))?;
        tx.commit()
            .map_err(state_err("committing undo transaction"))?;
        Ok(removed_ids.len())
    }

    pub fn delete_session(&self, session_id: &str) -> Result<bool, HermesError> {
        let tx = self
            .connection
            .unchecked_transaction()
            .map_err(state_err("opening delete session transaction"))?;
        let exists: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE id = ?",
                [session_id],
                |row| row.get(0),
            )
            .map_err(state_err("checking session existence"))?;
        if exists == 0 {
            tx.rollback()
                .map_err(state_err("rolling back delete session"))?;
            return Ok(false);
        }
        tx.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?",
            [session_id],
        )
        .map_err(state_err("orphaning child sessions"))?;
        tx.execute("DELETE FROM messages WHERE session_id = ?", [session_id])
            .map_err(state_err("deleting session messages"))?;
        tx.execute("DELETE FROM sessions WHERE id = ?", [session_id])
            .map_err(state_err("deleting session"))?;
        tx.commit()
            .map_err(state_err("committing delete session"))?;
        Ok(true)
    }

    fn message_search_context(
        &self,
        message_id: i64,
    ) -> Result<Vec<SearchContextMessage>, HermesError> {
        let mut statement = self
            .connection
            .prepare(
                "WITH target AS (
                     SELECT session_id, timestamp, id
                     FROM messages
                     WHERE id = ?
                 )
                 SELECT role, content
                 FROM (
                     SELECT m.id, m.timestamp, m.role, m.content
                     FROM messages m
                     JOIN target t ON t.session_id = m.session_id
                     WHERE (m.timestamp < t.timestamp)
                        OR (m.timestamp = t.timestamp AND m.id < t.id)
                     ORDER BY m.timestamp DESC, m.id DESC
                     LIMIT 1
                 )
                 UNION ALL
                 SELECT role, content
                 FROM messages
                 WHERE id = ?
                 UNION ALL
                 SELECT role, content
                 FROM (
                     SELECT m.id, m.timestamp, m.role, m.content
                     FROM messages m
                     JOIN target t ON t.session_id = m.session_id
                     WHERE (m.timestamp > t.timestamp)
                        OR (m.timestamp = t.timestamp AND m.id > t.id)
                     ORDER BY m.timestamp ASC, m.id ASC
                     LIMIT 1
                 )",
            )
            .map_err(state_err("preparing message context query"))?;
        let rows = statement
            .query_map(params![message_id, message_id], |row| {
                let raw = row.get::<_, Option<String>>(1)?.unwrap_or_default();
                Ok(SearchContextMessage {
                    role: row.get(0)?,
                    content: content_preview(&raw),
                })
            })
            .map_err(state_err("querying message context"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(state_err("collecting message context"))
    }
}

fn map_session_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRecord> {
    Ok(SessionRecord {
        id: row.get(0)?,
        source: row.get(1)?,
        user_id: row.get(2)?,
        model: row.get(3)?,
        model_config: row
            .get::<_, Option<String>>(4)?
            .and_then(|value| serde_json::from_str(&value).ok()),
        system_prompt: row.get(5)?,
        parent_session_id: row.get(6)?,
        started_at: row.get(7)?,
        ended_at: row.get(8)?,
        end_reason: row.get(9)?,
        message_count: row.get(10)?,
        tool_call_count: row.get(11)?,
        title: row.get(12)?,
        api_call_count: row.get(13)?,
    })
}

fn map_message_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRecord> {
    Ok(MessageRecord {
        id: row.get(0)?,
        session_id: row.get(1)?,
        role: row.get(2)?,
        content: row
            .get::<_, Option<String>>(3)?
            .and_then(|value| decode_content(&value).ok()),
        tool_call_id: row.get(4)?,
        tool_calls: row
            .get::<_, Option<String>>(5)?
            .and_then(|value| serde_json::from_str(&value).ok()),
        tool_name: row.get(6)?,
        timestamp: row.get(7)?,
        token_count: row.get(8)?,
        finish_reason: row.get(9)?,
        reasoning: row.get(10)?,
        reasoning_content: row.get(11)?,
        reasoning_details: row
            .get::<_, Option<String>>(12)?
            .and_then(|value| serde_json::from_str(&value).ok()),
        codex_reasoning_items: row
            .get::<_, Option<String>>(13)?
            .and_then(|value| serde_json::from_str(&value).ok()),
        codex_message_items: row
            .get::<_, Option<String>>(14)?
            .and_then(|value| serde_json::from_str(&value).ok()),
    })
}

fn encode_content(value: &Value) -> Result<String, HermesError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        _ => serde_json::to_string(value)
            .map(|json| format!("{CONTENT_JSON_PREFIX}{json}"))
            .map_err(|source| HermesError::State {
                action: "serializing message content",
                detail: source.to_string(),
            }),
    }
}

fn decode_content(value: &str) -> Result<Value, HermesError> {
    if let Some(json) = value.strip_prefix(CONTENT_JSON_PREFIX) {
        return serde_json::from_str(json).map_err(|source| HermesError::State {
            action: "decoding message content",
            detail: source.to_string(),
        });
    }
    Ok(Value::String(value.to_string()))
}

fn count_tool_calls(value: &Value) -> i64 {
    match value {
        Value::Array(items) => items.len() as i64,
        Value::Null => 0,
        _ => 1,
    }
}

fn parse_tool_call_count(raw: Option<&str>) -> Result<i64, HermesError> {
    let Some(raw) = raw else {
        return Ok(0);
    };
    let parsed = serde_json::from_str::<Value>(raw).map_err(|source| HermesError::State {
        action: "parsing stored tool_calls",
        detail: source.to_string(),
    })?;
    Ok(count_tool_calls(&parsed))
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn strip_title_suffix(title: &str) -> String {
    parse_title_suffix(title)
        .and_then(|_| title.rsplit_once(" #").map(|(base, _)| base.to_string()))
        .unwrap_or_else(|| title.to_string())
}

fn parse_title_suffix(title: &str) -> Option<u64> {
    let (_base, suffix) = title.rsplit_once(" #")?;
    (!suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
        .then(|| suffix.parse::<u64>().ok())?
}

fn format_preview(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let text: String = trimmed.chars().take(60).collect();
    if trimmed.chars().count() > 60 {
        format!("{text}...")
    } else {
        text
    }
}

fn content_preview(raw: &str) -> String {
    match decode_content(raw) {
        Ok(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.as_object()
                        .filter(|obj| obj.get("type").and_then(Value::as_str) == Some("text"))
                        .and_then(|obj| obj.get("text").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join(" ");
            if text.trim().is_empty() {
                String::from("[multimodal content]")
            } else {
                text.chars().take(200).collect()
            }
        }
        Ok(Value::String(text)) => text.chars().take(200).collect(),
        Ok(other) => other.to_string().chars().take(200).collect(),
        Err(_) => raw.chars().take(200).collect(),
    }
}

fn sanitize_fts5_query(query: &str) -> String {
    let mut sanitized = query
        .chars()
        .map(|ch| match ch {
            '+' | '{' | '}' | '(' | ')' | '^' => ' ',
            _ => ch,
        })
        .collect::<String>();
    while sanitized.contains("***") {
        sanitized = sanitized.replace("***", "*");
    }
    while sanitized.contains("**") {
        sanitized = sanitized.replace("**", "*");
    }
    sanitized = sanitized
        .split_whitespace()
        .map(|token| {
            let token = token.trim_matches('"');
            if token.contains(['.', '_', '-']) {
                format!("\"{token}\"")
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    sanitized = sanitized.trim().to_string();
    for op in ["AND", "OR", "NOT"] {
        if sanitized
            .to_ascii_uppercase()
            .strip_prefix(&format!("{op} "))
            .is_some()
        {
            sanitized = sanitized[op.len()..].trim().to_string();
        }
        if sanitized.to_ascii_uppercase().ends_with(&format!(" {op}")) {
            let new_len = sanitized.len().saturating_sub(op.len());
            sanitized = sanitized[..new_len].trim().to_string();
        }
    }
    sanitized
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn state_err(action: &'static str) -> impl FnOnce(rusqlite::Error) -> HermesError {
    move |source| HermesError::State {
        action,
        detail: source.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HermesContext;
    use tempfile::TempDir;

    fn test_store() -> (TempDir, SessionStore) {
        let temp = TempDir::new().expect("tempdir");
        let ctx = HermesContext::new(temp.path().join("home"));
        let store = ctx.open_session_store().expect("open store");
        (temp, store)
    }

    #[test]
    fn session_store_creates_and_lists_sessions() {
        let (_temp, store) = test_store();
        let created = store
            .create_session(&SessionCreate {
                id: String::from("sess-1"),
                source: String::from("cli"),
                user_id: Some(String::from("user-1")),
                model: Some(String::from("gpt-test")),
                model_config: None,
                system_prompt: Some(String::from("system")),
                parent_session_id: None,
            })
            .expect("create session");
        assert_eq!(created, "sess-1");
        assert_eq!(store.session_count().expect("session count"), 1);
        let listed = store.list_sessions(10, 0).expect("list sessions");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "sess-1");
    }

    #[test]
    fn append_message_round_trips_structured_content() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-1"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session");
        store
            .append_message(
                "sess-1",
                &MessageAppend {
                    role: String::from("assistant"),
                    content: Some(serde_json::json!([
                        {"type": "text", "text": "hello"}
                    ])),
                    tool_call_id: None,
                    tool_calls: Some(serde_json::json!([{"name": "search"}])),
                    tool_name: None,
                    token_count: Some(12),
                    finish_reason: Some(String::from("stop")),
                    reasoning: Some(String::from("brief")),
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .expect("append message");

        let session = store
            .get_session("sess-1")
            .expect("get session")
            .expect("session");
        assert_eq!(session.message_count, 1);
        assert_eq!(session.tool_call_count, 1);

        let messages = store.get_messages("sess-1").expect("get messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content,
            Some(serde_json::json!([{"type": "text", "text": "hello"}]))
        );
        assert_eq!(
            messages[0].tool_calls,
            Some(serde_json::json!([{"name": "search"}]))
        );
    }

    #[test]
    fn session_titles_are_sanitized_and_unique() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-1"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session 1");
        store
            .create_session(&SessionCreate {
                id: String::from("sess-2"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session 2");

        let updated = store
            .set_session_title("sess-1", "  Hello\tworld\u{200b}\n")
            .expect("set title");
        assert!(updated);
        assert_eq!(
            store.get_session_title("sess-1").expect("get title"),
            Some(String::from("Hello world"))
        );
        let err = store
            .set_session_title("sess-2", "Hello world")
            .expect_err("duplicate title should fail");
        assert!(err.to_string().contains("already in use"));
    }

    #[test]
    fn resolve_session_ids_and_lineage_titles_follow_python_rules() {
        let (_temp, store) = test_store();
        for id in ["abc123", "abc999", "other1"] {
            store
                .create_session(&SessionCreate {
                    id: id.to_string(),
                    source: String::from("cli"),
                    user_id: None,
                    model: None,
                    model_config: None,
                    system_prompt: None,
                    parent_session_id: None,
                })
                .expect("create session");
        }
        store
            .set_session_title("abc123", "Project")
            .expect("title 1");
        store
            .set_session_title("abc999", "Project #2")
            .expect("title 2");

        assert_eq!(
            store.resolve_session_id("other1").expect("exact"),
            Some(String::from("other1"))
        );
        assert_eq!(store.resolve_session_id("abc").expect("prefix"), None);
        assert_eq!(
            store
                .resolve_session_by_title("Project")
                .expect("resolve by title"),
            Some(String::from("abc999"))
        );
        assert_eq!(
            store
                .get_next_title_in_lineage("Project")
                .expect("next lineage title"),
            "Project #3"
        );
    }

    #[test]
    fn search_sessions_returns_last_active_and_preview() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-a"),
                source: String::from("cli"),
                user_id: None,
                model: Some(String::from("model-a")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session");
        store
            .append_message(
                "sess-a",
                &MessageAppend {
                    role: String::from("user"),
                    content: Some(serde_json::json!("This is a fairly long user message that should become a preview in the search results.")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .expect("append message");

        let results = store
            .search_sessions(Some("cli"), 10, 0)
            .expect("search sessions");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "sess-a");
        assert!(results[0].last_active >= results[0].started_at);
        assert!(
            results[0]
                .preview
                .starts_with("This is a fairly long user message")
        );
    }

    #[test]
    fn search_messages_and_export_helpers_work() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-search"),
                source: String::from("cli"),
                user_id: None,
                model: Some(String::from("model-a")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session");
        store
            .append_message(
                "sess-search",
                &MessageAppend {
                    role: String::from("user"),
                    content: Some(serde_json::json!("first needle message")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .expect("append first");
        store
            .append_message(
                "sess-search",
                &MessageAppend {
                    role: String::from("assistant"),
                    content: Some(serde_json::json!([
                        {"type": "text", "text": "needle answer"},
                        {"type": "image_url", "image_url": {"url": "https://example.com"}}
                    ])),
                    tool_call_id: None,
                    tool_calls: Some(serde_json::json!([{"name": "search"}])),
                    tool_name: Some(String::from("search")),
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .expect("append second");

        let matches = store
            .search_messages("needle", Some(&[String::from("cli")]), None, None, 10, 0)
            .expect("search messages");
        assert!(!matches.is_empty());
        assert!(matches[0].snippet.contains("needle"));
        assert!(!matches[0].context.is_empty());

        let exported = store
            .export_session("sess-search")
            .expect("export session")
            .expect("session export");
        assert_eq!(exported.messages.len(), 2);
        assert_eq!(
            store
                .message_count(Some("sess-search"))
                .expect("message count"),
            2
        );

        let exported_all = store.export_all(Some("cli")).expect("export all");
        assert_eq!(exported_all.len(), 1);

        store.clear_messages("sess-search").expect("clear messages");
        assert_eq!(
            store
                .message_count(Some("sess-search"))
                .expect("message count"),
            0
        );
        let session = store
            .get_session("sess-search")
            .expect("get session")
            .expect("session");
        assert_eq!(session.message_count, 0);
        assert_eq!(session.tool_call_count, 0);
    }

    #[test]
    fn trim_last_exchange_removes_trailing_assistant_tool_and_user_messages() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-undo"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session");
        for message in [
            MessageAppend {
                role: String::from("user"),
                content: Some(serde_json::json!("keep me")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            MessageAppend {
                role: String::from("assistant"),
                content: Some(serde_json::json!("prior answer")),
                tool_call_id: None,
                tool_calls: Some(serde_json::json!([{"name": "search"}])),
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            MessageAppend {
                role: String::from("user"),
                content: Some(serde_json::json!("remove me")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            MessageAppend {
                role: String::from("assistant"),
                content: Some(serde_json::json!("thinking")),
                tool_call_id: None,
                tool_calls: Some(serde_json::json!([{"name": "plan"}, {"name": "search"}])),
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            MessageAppend {
                role: String::from("tool"),
                content: Some(serde_json::json!("tool output")),
                tool_call_id: Some(String::from("call-1")),
                tool_calls: None,
                tool_name: Some(String::from("search")),
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ] {
            store
                .append_message("sess-undo", &message)
                .expect("append message");
        }

        let removed = store
            .trim_last_exchange("sess-undo")
            .expect("trim last exchange");
        assert_eq!(removed, 3);

        let messages = store.get_messages("sess-undo").expect("get messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, Some(serde_json::json!("keep me")));
        assert_eq!(messages[1].content, Some(serde_json::json!("prior answer")));

        let session = store
            .get_session("sess-undo")
            .expect("get session")
            .expect("session");
        assert_eq!(session.message_count, 2);
        assert_eq!(session.tool_call_count, 1);
    }

    #[test]
    fn trim_last_exchange_removes_single_trailing_user_message() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("sess-undo-user"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create session");
        store
            .append_message(
                "sess-undo-user",
                &MessageAppend {
                    role: String::from("user"),
                    content: Some(serde_json::json!("question")),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .expect("append message");

        let removed = store
            .trim_last_exchange("sess-undo-user")
            .expect("trim last exchange");
        assert_eq!(removed, 1);
        assert!(
            store
                .get_messages("sess-undo-user")
                .expect("get messages")
                .is_empty()
        );
    }

    #[test]
    fn prune_sessions_and_source_counts_work() {
        let (_temp, store) = test_store();
        for (id, source) in [
            ("sess-old", "cli"),
            ("sess-new", "cli"),
            ("sess-tool", "tool"),
        ] {
            store
                .create_session(&SessionCreate {
                    id: id.to_string(),
                    source: source.to_string(),
                    user_id: None,
                    model: None,
                    model_config: None,
                    system_prompt: None,
                    parent_session_id: None,
                })
                .expect("create session");
        }

        store
            .connection
            .execute(
                "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'done' WHERE id = ?",
                params![1.0_f64, 2.0_f64, "sess-old"],
            )
            .expect("age old session");
        store
            .connection
            .execute(
                "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'done' WHERE id = ?",
                params![1.0_f64, 2.0_f64, "sess-tool"],
            )
            .expect("age tool session");

        let counts = store.session_counts_by_source().expect("counts by source");
        assert_eq!(
            counts,
            vec![(String::from("cli"), 2), (String::from("tool"), 1)]
        );

        let removed = store
            .prune_sessions(90, Some("tool"))
            .expect("prune tool sessions");
        assert_eq!(removed, vec![String::from("sess-tool")]);
        assert!(
            store
                .get_session("sess-tool")
                .expect("load pruned session")
                .is_none()
        );
        assert_eq!(store.session_count().expect("session count"), 2);
    }

    #[test]
    fn delete_session_orphans_children_and_removes_rows() {
        let (_temp, store) = test_store();
        store
            .create_session(&SessionCreate {
                id: String::from("parent"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .expect("create parent");
        store
            .create_session(&SessionCreate {
                id: String::from("child"),
                source: String::from("cli"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: Some(String::from("parent")),
            })
            .expect("create child");

        let deleted = store.delete_session("parent").expect("delete parent");
        assert!(deleted);
        assert!(store.get_session("parent").expect("get parent").is_none());
        let child = store
            .get_session("child")
            .expect("get child")
            .expect("child");
        assert_eq!(child.parent_session_id, None);
    }
}
