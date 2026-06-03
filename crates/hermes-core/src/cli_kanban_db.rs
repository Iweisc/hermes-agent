//! SQLite-backed Kanban board for multi-profile, multi-project collaboration.
//!
//! Native Rust port of `hermes_cli/kanban_db.py`. Reproduces the board /
//! path resolution, schema + migrations, task lifecycle (create / claim /
//! complete / block / archive), runs (attempt history), comments / events,
//! dependency resolution, the dispatcher one-shot pass, the worker-context
//! builder, stats, notification subscriptions, retention GC, and the runs
//! accessors.
//!
//! Concurrency strategy mirrors the Python module: WAL mode + `BEGIN
//! IMMEDIATE` write transactions + compare-and-swap (CAS) updates on
//! `tasks.status` / `tasks.claim_lock`. SQLite serializes writers via its
//! WAL lock, so at most one claimer can win any given task; losers observe
//! zero affected rows and move on.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use regex::Regex;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Row, ToSql};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const VALID_STATUSES: &[&str] = &[
    "triage", "todo", "ready", "running", "blocked", "done", "archived",
];
pub const VALID_WORKSPACE_KINDS: &[&str] = &["scratch", "worktree", "dir"];

/// A running task's claim is valid for 15 minutes; after that the next
/// dispatcher tick reclaims it.
pub const DEFAULT_CLAIM_TTL_SECONDS: i64 = 15 * 60;

// Worker-context caps so build_worker_context() stays bounded.
const CTX_MAX_PRIOR_ATTEMPTS: usize = 10;
const CTX_MAX_COMMENTS: usize = 30;
const CTX_MAX_FIELD_BYTES: usize = 4 * 1024;
const CTX_MAX_BODY_BYTES: usize = 8 * 1024;
const CTX_MAX_COMMENT_BYTES: usize = 2 * 1024;

pub const DEFAULT_BOARD: &str = "default";

/// After this many consecutive non-success outcomes the dispatcher gives up.
pub const DEFAULT_FAILURE_LIMIT: i64 = 5;
/// Legacy alias.
pub const DEFAULT_SPAWN_FAILURE_LIMIT: i64 = DEFAULT_FAILURE_LIMIT;
/// Max bytes to keep in a single worker log file before rotation.
pub const DEFAULT_LOG_ROTATE_BYTES: u64 = 2 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum KanbanError {
    /// Recoverable user error (Python: `ValueError`).
    Value(String),
    /// Operation refused due to state (Python: `RuntimeError`).
    Runtime(String),
    /// Completion blocked: phantom `created_cards` (Python:
    /// `HallucinatedCardsError`).
    HallucinatedCards {
        phantom: Vec<String>,
        completing_task_id: String,
    },
    /// Underlying SQLite / IO failure.
    Sqlite(String),
}

impl std::fmt::Display for KanbanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KanbanError::Value(m) => write!(f, "{m}"),
            KanbanError::Runtime(m) => write!(f, "{m}"),
            KanbanError::Sqlite(m) => write!(f, "{m}"),
            KanbanError::HallucinatedCards { phantom, .. } => write!(
                f,
                "completion blocked: claimed created_cards that do not exist \
                 or were not created by this worker: {}",
                phantom.join(", ")
            ),
        }
    }
}

impl std::error::Error for KanbanError {}

impl From<rusqlite::Error> for KanbanError {
    fn from(e: rusqlite::Error) -> Self {
        KanbanError::Sqlite(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, KanbanError>;

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Hermes root resolution
// ---------------------------------------------------------------------------

fn expanduser(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    } else if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(p)
}

/// Mirror of `hermes_constants.get_default_hermes_root()`.
///
/// When `HERMES_HOME` points at `<root>/profiles/<name>` it returns
/// `<root>`; otherwise returns `HERMES_HOME` directly. Falls back to
/// `~/.hermes`.
pub fn get_default_hermes_root() -> PathBuf {
    if let Ok(home) = env::var("HERMES_HOME") {
        let home = home.trim();
        if !home.is_empty() {
            let path = expanduser(home);
            // If HERMES_HOME is <root>/profiles/<name>, collapse to <root>.
            if let Some(parent) = path.parent() {
                if parent.file_name().and_then(|s| s.to_str()) == Some("profiles") {
                    if let Some(root) = parent.parent() {
                        return root.to_path_buf();
                    }
                }
            }
            return path;
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
}

// ---------------------------------------------------------------------------
// Board slug + paths
// ---------------------------------------------------------------------------

fn board_slug_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-z0-9][a-z0-9\-_]{0,63}$").unwrap())
}

/// Lowercase + strip a slug; validate; return `None` for empty.
pub fn normalize_board_slug(slug: Option<&str>) -> Result<Option<String>> {
    let slug = match slug {
        None => return Ok(None),
        Some(s) => s,
    };
    let s = slug.trim().to_lowercase();
    if s.is_empty() {
        return Ok(None);
    }
    if !board_slug_re().is_match(&s) {
        return Err(KanbanError::Value(format!(
            "invalid board slug {slug:?}: must be 1-64 chars, lowercase \
             alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )));
    }
    Ok(Some(s))
}

/// Return the shared Hermes root that anchors the kanban board.
pub fn kanban_home() -> PathBuf {
    if let Ok(over) = env::var("HERMES_KANBAN_HOME") {
        let over = over.trim();
        if !over.is_empty() {
            return expanduser(over);
        }
    }
    get_default_hermes_root()
}

/// `<root>/kanban/boards` — the parent of non-default board dirs.
pub fn boards_root() -> PathBuf {
    kanban_home().join("kanban").join("boards")
}

/// Path to `<root>/kanban/current`.
pub fn current_board_path() -> PathBuf {
    kanban_home().join("kanban").join("current")
}

/// Return the active board slug, honouring the resolution chain.
pub fn get_current_board() -> String {
    if let Ok(envv) = env::var("HERMES_KANBAN_BOARD") {
        let envv = envv.trim();
        if !envv.is_empty() {
            if let Ok(Some(normed)) = normalize_board_slug(Some(envv)) {
                return normed;
            }
        }
    }
    let f = current_board_path();
    if f.exists() {
        if let Ok(val) = fs::read_to_string(&f) {
            let val = val.trim();
            if !val.is_empty() {
                if let Ok(Some(normed)) = normalize_board_slug(Some(val)) {
                    if board_exists(Some(&normed)) {
                        return normed;
                    }
                }
            }
        }
    }
    DEFAULT_BOARD.to_string()
}

/// Persist `slug` as the active board. Returns the file written.
pub fn set_current_board(slug: &str) -> Result<PathBuf> {
    let normed = normalize_board_slug(Some(slug))?
        .ok_or_else(|| KanbanError::Value("board slug is required".into()))?;
    let path = current_board_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    }
    fs::write(&path, format!("{normed}\n")).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    Ok(path)
}

/// Remove `<root>/kanban/current` so the active board reverts to default.
pub fn clear_current_board() {
    let _ = fs::remove_file(current_board_path());
}

/// On-disk directory for `board`.
pub fn board_dir(board: Option<&str>) -> PathBuf {
    let slug = normalize_board_slug(board)
        .ok()
        .flatten()
        .unwrap_or_else(|| DEFAULT_BOARD.to_string());
    boards_root().join(slug)
}

/// True if the board has a DB or a metadata dir on disk (default always exists).
pub fn board_exists(board: Option<&str>) -> bool {
    let slug = normalize_board_slug(board)
        .ok()
        .flatten()
        .unwrap_or_else(|| DEFAULT_BOARD.to_string());
    if slug == DEFAULT_BOARD {
        return true;
    }
    let d = board_dir(Some(&slug));
    d.is_dir() || d.join("kanban.db").exists()
}

/// Path to the `kanban.db` for `board`.
pub fn kanban_db_path(board: Option<&str>) -> PathBuf {
    if let Ok(over) = env::var("HERMES_KANBAN_DB") {
        let over = over.trim();
        if !over.is_empty() {
            return expanduser(over);
        }
    }
    let slug = match normalize_board_slug(board).ok().flatten() {
        Some(s) => s,
        None => get_current_board(),
    };
    if slug == DEFAULT_BOARD {
        return kanban_home().join("kanban.db");
    }
    board_dir(Some(&slug)).join("kanban.db")
}

/// Directory under which `scratch` workspaces are created.
pub fn workspaces_root(board: Option<&str>) -> PathBuf {
    if let Ok(over) = env::var("HERMES_KANBAN_WORKSPACES_ROOT") {
        let over = over.trim();
        if !over.is_empty() {
            return expanduser(over);
        }
    }
    let slug = match normalize_board_slug(board).ok().flatten() {
        Some(s) => s,
        None => get_current_board(),
    };
    if slug == DEFAULT_BOARD {
        return kanban_home().join("kanban").join("workspaces");
    }
    board_dir(Some(&slug)).join("workspaces")
}

/// Directory under which per-task worker logs are written.
pub fn worker_logs_dir(board: Option<&str>) -> PathBuf {
    let slug = match normalize_board_slug(board).ok().flatten() {
        Some(s) => s,
        None => get_current_board(),
    };
    if slug == DEFAULT_BOARD {
        return kanban_home().join("kanban").join("logs");
    }
    board_dir(Some(&slug)).join("logs")
}

/// Path to `board.json` for `board`.
pub fn board_metadata_path(board: Option<&str>) -> PathBuf {
    board_dir(board).join("board.json")
}

/// Turn a slug into a reasonable default display name.
fn default_board_display_name(slug: &str) -> String {
    let pretty: Vec<String> = slug
        .replace('_', "-")
        .split('-')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect();
    if pretty.is_empty() {
        slug.to_string()
    } else {
        pretty.join(" ")
    }
}

/// Return `board.json` contents (or synthesized defaults). Never fails.
pub fn read_board_metadata(board: Option<&str>) -> Map<String, Value> {
    let slug = normalize_board_slug(board)
        .ok()
        .flatten()
        .unwrap_or_else(|| DEFAULT_BOARD.to_string());
    let mut meta = Map::new();
    meta.insert("slug".into(), Value::String(slug.clone()));
    meta.insert(
        "name".into(),
        Value::String(default_board_display_name(&slug)),
    );
    meta.insert("description".into(), Value::String(String::new()));
    meta.insert("icon".into(), Value::String(String::new()));
    meta.insert("color".into(), Value::String(String::new()));
    meta.insert("created_at".into(), Value::Null);
    meta.insert("archived".into(), Value::Bool(false));

    let p = board_metadata_path(Some(&slug));
    if p.exists() {
        if let Ok(raw) = fs::read_to_string(&p) {
            if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(&raw) {
                for (k, v) in obj {
                    meta.insert(k, v);
                }
                // Trust the filesystem for the slug.
                meta.insert("slug".into(), Value::String(slug.clone()));
            }
        }
    }
    meta.insert(
        "db_path".into(),
        Value::String(kanban_db_path(Some(&slug)).to_string_lossy().into_owned()),
    );
    meta
}

/// Create / update `board.json` for `board`.
pub fn write_board_metadata(
    board: Option<&str>,
    name: Option<&str>,
    description: Option<&str>,
    icon: Option<&str>,
    color: Option<&str>,
    archived: Option<bool>,
) -> Result<Map<String, Value>> {
    let slug = normalize_board_slug(board)
        .ok()
        .flatten()
        .unwrap_or_else(|| DEFAULT_BOARD.to_string());
    let mut meta = read_board_metadata(Some(&slug));
    meta.remove("db_path");
    if let Some(name) = name {
        let n = name.trim();
        let val = if n.is_empty() {
            default_board_display_name(&slug)
        } else {
            n.to_string()
        };
        meta.insert("name".into(), Value::String(val));
    }
    if let Some(d) = description {
        meta.insert("description".into(), Value::String(d.to_string()));
    }
    if let Some(i) = icon {
        meta.insert("icon".into(), Value::String(i.to_string()));
    }
    if let Some(c) = color {
        meta.insert("color".into(), Value::String(c.to_string()));
    }
    if let Some(a) = archived {
        meta.insert("archived".into(), Value::Bool(a));
    }
    let has_created = matches!(meta.get("created_at"), Some(v) if !v.is_null() && v.as_i64() != Some(0));
    if !has_created {
        meta.insert("created_at".into(), Value::from(now_secs()));
    }
    let path = board_metadata_path(Some(&slug));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    }
    let body = serde_json::to_string_pretty(&Value::Object(meta.clone()))
        .map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    fs::write(&path, format!("{body}\n")).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    meta.insert(
        "db_path".into(),
        Value::String(kanban_db_path(Some(&slug)).to_string_lossy().into_owned()),
    );
    Ok(meta)
}

/// Create a new board directory + DB + metadata. Idempotent.
pub fn create_board(
    slug: &str,
    name: Option<&str>,
    description: Option<&str>,
    icon: Option<&str>,
    color: Option<&str>,
) -> Result<Map<String, Value>> {
    let normed = normalize_board_slug(Some(slug))?
        .ok_or_else(|| KanbanError::Value("board slug is required".into()))?;
    let meta = write_board_metadata(Some(&normed), name, description, icon, color, None)?;
    init_db(None, Some(&normed))?;
    Ok(meta)
}

/// Enumerate all boards that exist on disk. Default is always first.
pub fn list_boards(include_archived: bool) -> Vec<Map<String, Value>> {
    let mut entries: Vec<Map<String, Value>> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    entries.push(read_board_metadata(Some(DEFAULT_BOARD)));
    seen.insert(DEFAULT_BOARD.to_string());

    let root = boards_root();
    if root.is_dir() {
        let mut children: Vec<PathBuf> = match fs::read_dir(&root) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(_) => Vec::new(),
        };
        children.sort_by_key(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase())
                .unwrap_or_default()
        });
        for child in children {
            if !child.is_dir() {
                continue;
            }
            let slug = match child.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let normed = match normalize_board_slug(Some(&slug)) {
                Ok(Some(n)) => n,
                _ => continue,
            };
            if seen.contains(&normed) {
                continue;
            }
            let has_db = child.join("kanban.db").exists();
            let has_meta = child.join("board.json").exists();
            if !(has_db || has_meta) {
                continue;
            }
            let meta = read_board_metadata(Some(&normed));
            if meta.get("archived").and_then(|v| v.as_bool()) == Some(true) && !include_archived {
                continue;
            }
            entries.push(meta);
            seen.insert(normed);
        }
    }
    entries
}

/// Remove or archive a board. `default` cannot be removed.
pub fn remove_board(slug: &str, archive: bool) -> Result<Map<String, Value>> {
    let normed = normalize_board_slug(Some(slug))?
        .ok_or_else(|| KanbanError::Value("board slug is required".into()))?;
    if normed == DEFAULT_BOARD {
        return Err(KanbanError::Value(
            "the 'default' board cannot be removed".into(),
        ));
    }
    let d = board_dir(Some(&normed));
    if !d.exists() {
        return Err(KanbanError::Value(format!("board {normed:?} does not exist")));
    }
    if get_current_board() == normed {
        clear_current_board();
    }
    let mut out = Map::new();
    out.insert("slug".into(), Value::String(normed.clone()));
    if archive {
        let archive_root = boards_root().join("_archived");
        fs::create_dir_all(&archive_root).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
        let ts = now_secs();
        let mut target = archive_root.join(format!("{normed}-{ts}"));
        let mut suffix = 1;
        while target.exists() {
            target = archive_root.join(format!("{normed}-{ts}-{suffix}"));
            suffix += 1;
        }
        fs::rename(&d, &target).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
        out.insert("action".into(), Value::String("archived".into()));
        out.insert(
            "new_path".into(),
            Value::String(target.to_string_lossy().into_owned()),
        );
    } else {
        fs::remove_dir_all(&d).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
        out.insert("action".into(), Value::String("deleted".into()));
        out.insert("new_path".into(), Value::String(String::new()));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Data classes
// ---------------------------------------------------------------------------

/// In-memory view of a row from the `tasks` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub body: Option<String>,
    pub assignee: Option<String>,
    pub status: String,
    pub priority: i64,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub workspace_kind: String,
    pub workspace_path: Option<String>,
    pub claim_lock: Option<String>,
    pub claim_expires: Option<i64>,
    pub tenant: Option<String>,
    pub result: Option<String>,
    pub idempotency_key: Option<String>,
    pub consecutive_failures: i64,
    pub worker_pid: Option<i64>,
    pub last_failure_error: Option<String>,
    pub max_runtime_seconds: Option<i64>,
    pub last_heartbeat_at: Option<i64>,
    pub current_run_id: Option<i64>,
    pub workflow_template_id: Option<String>,
    pub current_step_key: Option<String>,
    pub skills: Option<Vec<String>>,
}

fn has_col(row: &Row, name: &str) -> bool {
    row.as_ref().column_index(name).is_ok()
}

fn opt_str(row: &Row, name: &str) -> Option<String> {
    if has_col(row, name) {
        row.get::<_, Option<String>>(name).unwrap_or(None)
    } else {
        None
    }
}

fn opt_i64(row: &Row, name: &str) -> Option<i64> {
    if has_col(row, name) {
        row.get::<_, Option<i64>>(name).unwrap_or(None)
    } else {
        None
    }
}

impl Task {
    pub fn from_row(row: &Row) -> Task {
        let skills = if has_col(row, "skills") {
            match row.get::<_, Option<String>>("skills").unwrap_or(None) {
                Some(s) if !s.is_empty() => match serde_json::from_str::<Value>(&s) {
                    Ok(Value::Array(arr)) => Some(
                        arr.into_iter()
                            .filter_map(|v| match v {
                                Value::String(x) if !x.is_empty() => Some(x),
                                Value::String(_) => None,
                                Value::Null => None,
                                other if !other.is_null() => Some(other.to_string()),
                                _ => None,
                            })
                            .collect::<Vec<String>>(),
                    ),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };
        let consecutive_failures = if has_col(row, "consecutive_failures") {
            row.get::<_, Option<i64>>("consecutive_failures")
                .unwrap_or(None)
                .unwrap_or(0)
        } else {
            opt_i64(row, "spawn_failures").unwrap_or(0)
        };
        let last_failure_error = if has_col(row, "last_failure_error") {
            row.get::<_, Option<String>>("last_failure_error")
                .unwrap_or(None)
        } else {
            opt_str(row, "last_spawn_error")
        };
        Task {
            id: row.get("id").unwrap_or_default(),
            title: row.get("title").unwrap_or_default(),
            body: row.get("body").unwrap_or(None),
            assignee: row.get("assignee").unwrap_or(None),
            status: row.get("status").unwrap_or_default(),
            priority: row.get::<_, Option<i64>>("priority").unwrap_or(None).unwrap_or(0),
            created_by: row.get("created_by").unwrap_or(None),
            created_at: row.get("created_at").unwrap_or(0),
            started_at: row.get("started_at").unwrap_or(None),
            completed_at: row.get("completed_at").unwrap_or(None),
            workspace_kind: row.get("workspace_kind").unwrap_or_default(),
            workspace_path: row.get("workspace_path").unwrap_or(None),
            claim_lock: row.get("claim_lock").unwrap_or(None),
            claim_expires: row.get("claim_expires").unwrap_or(None),
            tenant: opt_str(row, "tenant"),
            result: opt_str(row, "result"),
            idempotency_key: opt_str(row, "idempotency_key"),
            consecutive_failures,
            worker_pid: opt_i64(row, "worker_pid"),
            last_failure_error,
            max_runtime_seconds: opt_i64(row, "max_runtime_seconds"),
            last_heartbeat_at: opt_i64(row, "last_heartbeat_at"),
            current_run_id: opt_i64(row, "current_run_id"),
            workflow_template_id: opt_str(row, "workflow_template_id"),
            current_step_key: opt_str(row, "current_step_key"),
            skills,
        }
    }
}

/// In-memory view of a `task_runs` row.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub id: i64,
    pub task_id: String,
    pub profile: Option<String>,
    pub step_key: Option<String>,
    pub status: String,
    pub claim_lock: Option<String>,
    pub claim_expires: Option<i64>,
    pub worker_pid: Option<i64>,
    pub max_runtime_seconds: Option<i64>,
    pub last_heartbeat_at: Option<i64>,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub outcome: Option<String>,
    pub summary: Option<String>,
    pub metadata: Option<Value>,
    pub error: Option<String>,
}

impl Run {
    pub fn from_row(row: &Row) -> Run {
        let metadata = match row.get::<_, Option<String>>("metadata").unwrap_or(None) {
            Some(s) if !s.is_empty() => serde_json::from_str::<Value>(&s).ok(),
            _ => None,
        };
        Run {
            id: row.get("id").unwrap_or(0),
            task_id: row.get("task_id").unwrap_or_default(),
            profile: row.get("profile").unwrap_or(None),
            step_key: row.get("step_key").unwrap_or(None),
            status: row.get("status").unwrap_or_default(),
            claim_lock: row.get("claim_lock").unwrap_or(None),
            claim_expires: row.get("claim_expires").unwrap_or(None),
            worker_pid: row.get("worker_pid").unwrap_or(None),
            max_runtime_seconds: row.get("max_runtime_seconds").unwrap_or(None),
            last_heartbeat_at: row.get("last_heartbeat_at").unwrap_or(None),
            started_at: row.get("started_at").unwrap_or(0),
            ended_at: row.get("ended_at").unwrap_or(None),
            outcome: row.get("outcome").unwrap_or(None),
            summary: row.get("summary").unwrap_or(None),
            metadata,
            error: row.get("error").unwrap_or(None),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Comment {
    pub id: i64,
    pub task_id: String,
    pub author: String,
    pub body: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub id: i64,
    pub task_id: String,
    pub kind: String,
    pub payload: Option<Value>,
    pub created_at: i64,
    pub run_id: Option<i64>,
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    current_run_id       INTEGER,
    workflow_template_id TEXT,
    current_step_key     TEXT,
    skills               TEXT
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_tenant          ON tasks(tenant);
CREATE INDEX IF NOT EXISTS idx_tasks_idempotency     ON tasks(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run            ON task_events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"#;

// ---------------------------------------------------------------------------
// Connection helpers
// ---------------------------------------------------------------------------

fn initialized_paths() -> &'static std::sync::Mutex<HashSet<String>> {
    use std::sync::OnceLock;
    static SET: OnceLock<std::sync::Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| std::sync::Mutex::new(HashSet::new()))
}

fn resolved_key(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Open (and initialize if needed) the kanban DB.
pub fn connect(db_path: Option<&Path>, board: Option<&str>) -> Result<Connection> {
    let path: PathBuf = match db_path {
        Some(p) => p.to_path_buf(),
        None => kanban_db_path(board),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    }
    let resolved = resolved_key(&path);
    let needs_init = {
        let set = initialized_paths().lock().unwrap();
        !set.contains(&resolved)
    };
    let conn = Connection::open(&path)?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    if needs_init {
        conn.execute_batch(SCHEMA_SQL)?;
        migrate_add_optional_columns(&conn)?;
        initialized_paths().lock().unwrap().insert(resolved);
    }
    Ok(conn)
}

/// Create the schema if it doesn't exist; return the path used. Always
/// re-runs the migration pass (clears the per-path init cache first).
pub fn init_db(db_path: Option<&Path>, board: Option<&str>) -> Result<PathBuf> {
    let path: PathBuf = match db_path {
        Some(p) => p.to_path_buf(),
        None => kanban_db_path(board),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    }
    let resolved = resolved_key(&path);
    initialized_paths().lock().unwrap().remove(&resolved);
    let _conn = connect(Some(&path), None)?;
    Ok(path)
}

fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let cols: rusqlite::Result<HashSet<String>> = stmt
        .query_map([], |r| r.get::<_, String>("name"))?
        .collect();
    Ok(cols?)
}

/// Add columns introduced after v1 release to legacy DBs.
pub fn migrate_add_optional_columns(conn: &Connection) -> Result<()> {
    let cols = table_columns(conn, "tasks")?;
    if !cols.contains("tenant") {
        conn.execute("ALTER TABLE tasks ADD COLUMN tenant TEXT", [])?;
    }
    if !cols.contains("result") {
        conn.execute("ALTER TABLE tasks ADD COLUMN result TEXT", [])?;
    }
    if !cols.contains("idempotency_key") {
        conn.execute("ALTER TABLE tasks ADD COLUMN idempotency_key TEXT", [])?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)",
            [],
        )?;
    }
    if !cols.contains("consecutive_failures") {
        if cols.contains("spawn_failures") {
            conn.execute(
                "ALTER TABLE tasks RENAME COLUMN spawn_failures TO consecutive_failures",
                [],
            )?;
        } else {
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN consecutive_failures INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
    }
    if !cols.contains("worker_pid") {
        conn.execute("ALTER TABLE tasks ADD COLUMN worker_pid INTEGER", [])?;
    }
    if !cols.contains("last_failure_error") {
        if cols.contains("last_spawn_error") {
            conn.execute(
                "ALTER TABLE tasks RENAME COLUMN last_spawn_error TO last_failure_error",
                [],
            )?;
        } else {
            conn.execute("ALTER TABLE tasks ADD COLUMN last_failure_error TEXT", [])?;
        }
    }
    if !cols.contains("max_runtime_seconds") {
        conn.execute("ALTER TABLE tasks ADD COLUMN max_runtime_seconds INTEGER", [])?;
    }
    if !cols.contains("last_heartbeat_at") {
        conn.execute("ALTER TABLE tasks ADD COLUMN last_heartbeat_at INTEGER", [])?;
    }
    if !cols.contains("current_run_id") {
        conn.execute("ALTER TABLE tasks ADD COLUMN current_run_id INTEGER", [])?;
    }
    if !cols.contains("workflow_template_id") {
        conn.execute("ALTER TABLE tasks ADD COLUMN workflow_template_id TEXT", [])?;
    }
    if !cols.contains("current_step_key") {
        conn.execute("ALTER TABLE tasks ADD COLUMN current_step_key TEXT", [])?;
    }
    if !cols.contains("skills") {
        conn.execute("ALTER TABLE tasks ADD COLUMN skills TEXT", [])?;
    }

    let ev_cols = table_columns(conn, "task_events")?;
    if !ev_cols.contains("run_id") {
        conn.execute("ALTER TABLE task_events ADD COLUMN run_id INTEGER", [])?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run ON task_events(run_id, id)",
            [],
        )?;
    }

    // One-shot backfill: running tasks predating runs get a synthesized run.
    let runs_exist: bool = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'",
            [],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if runs_exist {
        backfill_inflight_runs(conn)?;
    }

    // One-shot event-kind rename pass.
    let renames: &[(&str, &str)] = &[
        ("ready", "promoted"),
        ("priority", "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    ];
    for (old, new) in renames {
        conn.execute(
            "UPDATE task_events SET kind = ? WHERE kind = ?",
            params![new, old],
        )?;
    }
    Ok(())
}

fn backfill_inflight_runs(conn: &Connection) -> Result<()> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let res = (|| -> Result<()> {
        let inflight: Vec<(String, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<i64>)> = {
            let mut stmt = conn.prepare(
                "SELECT id, assignee, claim_lock, claim_expires, worker_pid, \
                 max_runtime_seconds, last_heartbeat_at, started_at \
                 FROM tasks WHERE status = 'running' AND current_run_id IS NULL",
            )?;
            let rows: rusqlite::Result<Vec<_>> = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<i64>>(3)?,
                        r.get::<_, Option<i64>>(4)?,
                        r.get::<_, Option<i64>>(5)?,
                        r.get::<_, Option<i64>>(6)?,
                        r.get::<_, Option<i64>>(7)?,
                    ))
                })?
                .collect();
            rows?
        };
        for (id, assignee, claim_lock, claim_expires, worker_pid, max_rt, last_hb, started_at) in
            inflight
        {
            let started = started_at.unwrap_or_else(now_secs);
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, \
                 worker_pid, max_runtime_seconds, last_heartbeat_at, started_at) \
                 VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)",
                params![id, assignee, claim_lock, claim_expires, worker_pid, max_rt, last_hb, started],
            )?;
            let run_id = conn.last_insert_rowid();
            let upd = conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ? AND current_run_id IS NULL",
                params![run_id, id],
            )?;
            if upd != 1 {
                conn.execute(
                    "UPDATE task_runs SET status = 'reclaimed', outcome = 'reclaimed', ended_at = ? \
                     WHERE id = ?",
                    params![now_secs(), run_id],
                )?;
            }
        }
        Ok(())
    })();
    match res {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Run `f` inside a `BEGIN IMMEDIATE` write transaction, committing on Ok
/// and rolling back on Err. Mirrors the `write_txn` context manager.
pub fn write_txn<T, F>(conn: &Connection, f: F) -> Result<T>
where
    F: FnOnce(&Connection) -> Result<T>,
{
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match f(conn) {
        Ok(v) => {
            conn.execute_batch("COMMIT")?;
            Ok(v)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

fn new_task_id() -> String {
    let mut bytes = [0u8; 4];
    getrandom_bytes(&mut bytes);
    format!("t_{:02x}{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2], bytes[3])
}

fn getrandom_bytes(buf: &mut [u8]) {
    if getrandom::fill(buf).is_err() {
        // Fallback: derive from time + a counter (non-cryptographic).
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((n >> (8 * i)) & 0xff) as u8;
        }
    }
}

fn claimer_id() -> String {
    let host = hostname().unwrap_or_else(|| "unknown".to_string());
    let host = if host.is_empty() {
        "unknown".to_string()
    } else {
        host
    };
    format!("{host}:{}", std::process::id())
}

fn hostname() -> Option<String> {
    if let Ok(h) = env::var("HOSTNAME") {
        if !h.is_empty() {
            return Some(h);
        }
    }
    // POSIX gethostname.
    let mut buf = vec![0u8; 256];
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        return String::from_utf8(buf[..end].to_vec()).ok();
    }
    None
}

// ---------------------------------------------------------------------------
// Task creation / mutation
// ---------------------------------------------------------------------------

/// Lowercase-assignee normalization (dashboard/CLI parity). The Python
/// version delegates to `normalize_profile_name`, which strips + lowercases.
pub fn canonical_assignee(assignee: Option<&str>) -> Option<String> {
    assignee.map(|a| a.trim().to_lowercase())
}

/// Parameters for [`create_task`].
#[derive(Debug, Clone, Default)]
pub struct CreateTask<'a> {
    pub title: &'a str,
    pub body: Option<&'a str>,
    pub assignee: Option<&'a str>,
    pub created_by: Option<&'a str>,
    pub workspace_kind: &'a str,
    pub workspace_path: Option<&'a str>,
    pub tenant: Option<&'a str>,
    pub priority: i64,
    pub parents: Vec<String>,
    pub triage: bool,
    pub idempotency_key: Option<&'a str>,
    pub max_runtime_seconds: Option<i64>,
    pub skills: Option<Vec<String>>,
}

impl<'a> CreateTask<'a> {
    pub fn new(title: &'a str) -> Self {
        CreateTask {
            title,
            workspace_kind: "scratch",
            ..Default::default()
        }
    }
}

/// Create a new task and optionally link it under parent tasks.
pub fn create_task(conn: &Connection, opts: CreateTask<'_>) -> Result<String> {
    let assignee = canonical_assignee(opts.assignee);
    if opts.title.trim().is_empty() {
        return Err(KanbanError::Value("title is required".into()));
    }
    if !VALID_WORKSPACE_KINDS.contains(&opts.workspace_kind) {
        let mut sorted: Vec<&str> = VALID_WORKSPACE_KINDS.to_vec();
        sorted.sort();
        return Err(KanbanError::Value(format!(
            "workspace_kind must be one of {sorted:?}, got {:?}",
            opts.workspace_kind
        )));
    }
    let parents: Vec<String> = opts
        .parents
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect();

    // Normalise + validate skills.
    let skills_list: Option<Vec<String>> = match &opts.skills {
        None => None,
        Some(skills) => {
            let mut cleaned: Vec<String> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for s in skills {
                if s.is_empty() {
                    continue;
                }
                let name = s.trim().to_string();
                if name.is_empty() {
                    continue;
                }
                if name.contains(',') {
                    return Err(KanbanError::Value(format!(
                        "skill name cannot contain comma: {name:?} \
                         (pass a list of separate names instead of a comma-joined string)"
                    )));
                }
                if seen.contains(&name) {
                    continue;
                }
                seen.insert(name.clone());
                cleaned.push(name);
            }
            Some(cleaned)
        }
    };

    // Idempotency check before the write txn.
    if let Some(key) = opts.idempotency_key {
        if !key.is_empty() {
            let existing: Option<String> = conn
                .query_row(
                    "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' \
                     ORDER BY created_at DESC LIMIT 1",
                    params![key],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(id) = existing {
                return Ok(id);
            }
        }
    }

    let now = now_secs();
    for attempt in 0..2 {
        let task_id = new_task_id();
        let res = write_txn(conn, |conn| {
            let initial_status: String = if opts.triage {
                "triage".to_string()
            } else {
                let mut status = "ready".to_string();
                if !parents.is_empty() {
                    let missing = find_missing_parents(conn, &parents)?;
                    if !missing.is_empty() {
                        return Err(KanbanError::Value(format!(
                            "unknown parent task(s): {}",
                            missing.join(", ")
                        )));
                    }
                    let any_unfinished = parents_any_not_done(conn, &parents)?;
                    if any_unfinished {
                        status = "todo".to_string();
                    }
                }
                status
            };
            if opts.triage && !parents.is_empty() {
                let missing = find_missing_parents(conn, &parents)?;
                if !missing.is_empty() {
                    return Err(KanbanError::Value(format!(
                        "unknown parent task(s): {}",
                        missing.join(", ")
                    )));
                }
            }
            conn.execute(
                "INSERT INTO tasks ( \
                    id, title, body, assignee, status, priority, \
                    created_by, created_at, workspace_kind, workspace_path, \
                    tenant, idempotency_key, max_runtime_seconds, skills \
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    task_id,
                    opts.title.trim(),
                    opts.body,
                    assignee,
                    initial_status,
                    opts.priority,
                    opts.created_by,
                    now,
                    opts.workspace_kind,
                    opts.workspace_path,
                    opts.tenant,
                    opts.idempotency_key,
                    opts.max_runtime_seconds.filter(|&v| v != 0),
                    skills_list
                        .as_ref()
                        .map(|s| serde_json::to_string(s).unwrap_or_else(|_| "[]".into())),
                ],
            )?;
            for pid in &parents {
                conn.execute(
                    "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                    params![pid, task_id],
                )?;
            }
            let payload = json!({
                "assignee": assignee,
                "status": initial_status,
                "parents": parents,
                "tenant": opts.tenant,
                "skills": skills_list.as_ref().filter(|s| !s.is_empty()),
            });
            append_event(conn, &task_id, "created", Some(&payload), None)?;
            Ok(())
        });
        match res {
            Ok(()) => return Ok(task_id),
            Err(KanbanError::Sqlite(msg)) if msg.contains("UNIQUE") || msg.contains("constraint") => {
                if attempt == 1 {
                    return Err(KanbanError::Sqlite(msg));
                }
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(KanbanError::Runtime("unreachable".into()))
}

fn find_missing_parents(conn: &Connection, parents: &[String]) -> Result<Vec<String>> {
    if parents.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = vec!["?"; parents.len()].join(",");
    let sql = format!("SELECT id FROM tasks WHERE id IN ({placeholders})");
    let mut stmt = conn.prepare(&sql)?;
    let present: HashSet<String> = stmt
        .query_map(params_from_iter(parents.iter()), |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(parents
        .iter()
        .filter(|p| !present.contains(*p))
        .cloned()
        .collect())
}

fn parents_any_not_done(conn: &Connection, parents: &[String]) -> Result<bool> {
    let placeholders = vec!["?"; parents.len()].join(",");
    let sql = format!("SELECT status FROM tasks WHERE id IN ({placeholders})");
    let mut stmt = conn.prepare(&sql)?;
    let statuses: Vec<String> = stmt
        .query_map(params_from_iter(parents.iter()), |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(statuses.iter().any(|s| s != "done"))
}

pub fn get_task(conn: &Connection, task_id: &str) -> Result<Option<Task>> {
    let task = conn
        .query_row("SELECT * FROM tasks WHERE id = ?", params![task_id], |r| {
            Ok(Task::from_row(r))
        })
        .optional()?;
    Ok(task)
}

/// Filters for [`list_tasks`].
#[derive(Debug, Clone, Default)]
pub struct ListTasksFilter<'a> {
    pub assignee: Option<&'a str>,
    pub status: Option<&'a str>,
    pub tenant: Option<&'a str>,
    pub include_archived: bool,
    pub limit: Option<i64>,
}

pub fn list_tasks(conn: &Connection, filter: ListTasksFilter<'_>) -> Result<Vec<Task>> {
    let mut query = String::from("SELECT * FROM tasks WHERE 1=1");
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    if let Some(a) = filter.assignee {
        query.push_str(" AND assignee = ?");
        params.push(Box::new(canonical_assignee(Some(a))));
    }
    if let Some(s) = filter.status {
        if !VALID_STATUSES.contains(&s) {
            let mut sorted: Vec<&str> = VALID_STATUSES.to_vec();
            sorted.sort();
            return Err(KanbanError::Value(format!(
                "status must be one of {sorted:?}"
            )));
        }
        query.push_str(" AND status = ?");
        params.push(Box::new(s.to_string()));
    }
    if let Some(t) = filter.tenant {
        query.push_str(" AND tenant = ?");
        params.push(Box::new(t.to_string()));
    }
    if !filter.include_archived && filter.status != Some("archived") {
        query.push_str(" AND status != 'archived'");
    }
    query.push_str(" ORDER BY priority DESC, created_at ASC");
    if let Some(limit) = filter.limit {
        if limit != 0 {
            query.push_str(&format!(" LIMIT {limit}"));
        }
    }
    let mut stmt = conn.prepare(&query)?;
    let rows: Vec<Task> = stmt
        .query_map(params_from_iter(params.iter().map(|b| b.as_ref())), |r| {
            Ok(Task::from_row(r))
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// Assign or reassign a task. Refuses to reassign a running (claimed) task.
pub fn assign_task(conn: &Connection, task_id: &str, profile: Option<&str>) -> Result<bool> {
    let profile = canonical_assignee(profile);
    write_txn(conn, |conn| {
        let row: Option<(String, Option<String>)> = conn
            .query_row(
                "SELECT status, claim_lock FROM tasks WHERE id = ?",
                params![task_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (status, claim_lock) = match row {
            None => return Ok(false),
            Some(r) => r,
        };
        if claim_lock.is_some() && status == "running" {
            return Err(KanbanError::Runtime(format!(
                "cannot reassign {task_id}: currently running (claimed). \
                 Wait for completion or reclaim the stale lock first."
            )));
        }
        conn.execute(
            "UPDATE tasks SET assignee = ? WHERE id = ?",
            params![profile, task_id],
        )?;
        append_event(
            conn,
            task_id,
            "assigned",
            Some(&json!({ "assignee": profile })),
            None,
        )?;
        Ok(true)
    })
}

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

pub fn link_tasks(conn: &Connection, parent_id: &str, child_id: &str) -> Result<()> {
    if parent_id == child_id {
        return Err(KanbanError::Value("a task cannot depend on itself".into()));
    }
    write_txn(conn, |conn| {
        let missing = find_missing_parents(conn, &[parent_id.to_string(), child_id.to_string()])?;
        if !missing.is_empty() {
            return Err(KanbanError::Value(format!(
                "unknown task(s): {}",
                missing.join(", ")
            )));
        }
        if would_cycle(conn, parent_id, child_id)? {
            return Err(KanbanError::Value(format!(
                "linking {parent_id} -> {child_id} would create a cycle"
            )));
        }
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            params![parent_id, child_id],
        )?;
        let parent_status: String = conn.query_row(
            "SELECT status FROM tasks WHERE id = ?",
            params![parent_id],
            |r| r.get(0),
        )?;
        if parent_status != "done" {
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                params![child_id],
            )?;
        }
        append_event(
            conn,
            child_id,
            "linked",
            Some(&json!({ "parent": parent_id, "child": child_id })),
            None,
        )?;
        Ok(())
    })
}

fn would_cycle(conn: &Connection, parent_id: &str, child_id: &str) -> Result<bool> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = vec![child_id.to_string()];
    while let Some(node) = stack.pop() {
        if node == parent_id {
            return Ok(true);
        }
        if seen.contains(&node) {
            continue;
        }
        seen.insert(node.clone());
        let mut stmt = conn.prepare("SELECT child_id FROM task_links WHERE parent_id = ?")?;
        let kids: Vec<String> = stmt
            .query_map(params![node], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        stack.extend(kids);
    }
    Ok(false)
}

pub fn unlink_tasks(conn: &Connection, parent_id: &str, child_id: &str) -> Result<bool> {
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            params![parent_id, child_id],
        )?;
        if affected > 0 {
            append_event(
                conn,
                child_id,
                "unlinked",
                Some(&json!({ "parent": parent_id, "child": child_id })),
                None,
            )?;
        }
        Ok(affected > 0)
    })
}

pub fn parent_ids(conn: &Connection, task_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
    )?;
    let ids: Vec<String> = stmt
        .query_map(params![task_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(ids)
}

pub fn child_ids(conn: &Connection, task_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
    )?;
    let ids: Vec<String> = stmt
        .query_map(params![task_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(ids)
}

/// Return `(parent_id, result)` for every done parent of `task_id`.
pub fn parent_results(conn: &Connection, task_id: &str) -> Result<Vec<(String, Option<String>)>> {
    let mut stmt = conn.prepare(
        "SELECT t.id AS id, t.result AS result FROM tasks t \
         JOIN task_links l ON l.parent_id = t.id \
         WHERE l.child_id = ? AND t.status = 'done' ORDER BY t.completed_at ASC",
    )?;
    let rows: Vec<(String, Option<String>)> = stmt
        .query_map(params![task_id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Comments & events
// ---------------------------------------------------------------------------

pub fn add_comment(conn: &Connection, task_id: &str, author: &str, body: &str) -> Result<i64> {
    if body.trim().is_empty() {
        return Err(KanbanError::Value("comment body is required".into()));
    }
    if author.trim().is_empty() {
        return Err(KanbanError::Value("comment author is required".into()));
    }
    let now = now_secs();
    write_txn(conn, |conn| {
        let exists: Option<i64> = conn
            .query_row("SELECT 1 FROM tasks WHERE id = ?", params![task_id], |r| {
                r.get(0)
            })
            .optional()?;
        if exists.is_none() {
            return Err(KanbanError::Value(format!("unknown task {task_id}")));
        }
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            params![task_id, author.trim(), body.trim(), now],
        )?;
        let id = conn.last_insert_rowid();
        append_event(
            conn,
            task_id,
            "commented",
            Some(&json!({ "author": author, "len": body.chars().count() })),
            None,
        )?;
        Ok(id)
    })
}

pub fn list_comments(conn: &Connection, task_id: &str) -> Result<Vec<Comment>> {
    let mut stmt = conn.prepare(
        "SELECT * FROM task_comments WHERE task_id = ? ORDER BY created_at ASC",
    )?;
    let rows: Vec<Comment> = stmt
        .query_map(params![task_id], |r| {
            Ok(Comment {
                id: r.get("id")?,
                task_id: r.get("task_id")?,
                author: r.get("author")?,
                body: r.get("body")?,
                created_at: r.get("created_at")?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

pub fn list_events(conn: &Connection, task_id: &str) -> Result<Vec<Event>> {
    let mut stmt = conn.prepare(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC, id ASC",
    )?;
    let rows: Vec<Event> = stmt
        .query_map(params![task_id], |r| Ok(event_from_row(r)))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

fn event_from_row(r: &Row) -> Event {
    let payload = match r.get::<_, Option<String>>("payload").unwrap_or(None) {
        Some(s) if !s.is_empty() => serde_json::from_str::<Value>(&s).ok(),
        _ => None,
    };
    Event {
        id: r.get("id").unwrap_or(0),
        task_id: r.get("task_id").unwrap_or_default(),
        kind: r.get("kind").unwrap_or_default(),
        payload,
        created_at: r.get("created_at").unwrap_or(0),
        run_id: if has_col(r, "run_id") {
            r.get::<_, Option<i64>>("run_id").unwrap_or(None)
        } else {
            None
        },
    }
}

/// Record an event row. Called from within an already-open txn.
fn append_event(
    conn: &Connection,
    task_id: &str,
    kind: &str,
    payload: Option<&Value>,
    run_id: Option<i64>,
) -> Result<()> {
    let now = now_secs();
    // Python: `json.dumps(payload) if payload else None` — falsy payloads
    // (None, empty dict) serialize to NULL.
    let pl: Option<String> = match payload {
        Some(p) if !value_is_falsy(p) => Some(serde_json::to_string(p).unwrap_or_default()),
        _ => None,
    };
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        params![task_id, run_id, kind, pl, now],
    )?;
    Ok(())
}

fn value_is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Object(o) => o.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// Close the currently-active run for `task_id` and clear the pointer.
fn end_run(
    conn: &Connection,
    task_id: &str,
    outcome: &str,
    status: Option<&str>,
    summary: Option<&str>,
    error: Option<&str>,
    metadata: Option<&Value>,
) -> Result<Option<i64>> {
    let now = now_secs();
    let current: Option<Option<i64>> = conn
        .query_row(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            params![task_id],
            |r| r.get(0),
        )
        .optional()?;
    let run_id = match current {
        Some(Some(id)) if id != 0 => id,
        _ => return Ok(None),
    };
    let meta_str: Option<String> = metadata
        .filter(|m| !value_is_falsy(m))
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "UPDATE task_runs SET status = ?, outcome = ?, summary = ?, error = ?, metadata = ?, \
         ended_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL \
         WHERE id = ? AND ended_at IS NULL",
        params![
            status.unwrap_or(outcome),
            outcome,
            summary,
            error,
            meta_str,
            now,
            run_id
        ],
    )?;
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?",
        params![task_id],
    )?;
    Ok(Some(run_id))
}

fn current_run_id(conn: &Connection, task_id: &str) -> Result<Option<i64>> {
    let r: Option<Option<i64>> = conn
        .query_row(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            params![task_id],
            |r| r.get(0),
        )
        .optional()?;
    Ok(match r {
        Some(Some(id)) if id != 0 => Some(id),
        _ => None,
    })
}

/// Insert a zero-duration, already-closed run row.
fn synthesize_ended_run(
    conn: &Connection,
    task_id: &str,
    outcome: &str,
    summary: Option<&str>,
    error: Option<&str>,
    metadata: Option<&Value>,
) -> Result<i64> {
    let now = now_secs();
    let trow: Option<(Option<String>, Option<String>)> = conn
        .query_row(
            "SELECT assignee, current_step_key FROM tasks WHERE id = ?",
            params![task_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let (profile, step_key) = trow.unwrap_or((None, None));
    let meta_str: Option<String> = metadata
        .filter(|m| !value_is_falsy(m))
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "INSERT INTO task_runs ( \
            task_id, profile, step_key, status, outcome, summary, error, metadata, \
            started_at, ended_at \
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            task_id, profile, step_key, outcome, outcome, summary, error, meta_str, now, now
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

// ---------------------------------------------------------------------------
// Dependency resolution (todo -> ready)
// ---------------------------------------------------------------------------

/// Promote `todo` tasks to `ready` when all parents are `done`. Returns the
/// number of tasks promoted.
pub fn recompute_ready(conn: &Connection) -> Result<usize> {
    write_txn(conn, |conn| {
        let todo_ids: Vec<String> = {
            let mut stmt = conn.prepare("SELECT id FROM tasks WHERE status = 'todo'")?;
            stmt.query_map([], |r| r.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect()
        };
        let mut promoted = 0usize;
        for task_id in todo_ids {
            let all_done: bool = {
                let mut stmt = conn.prepare(
                    "SELECT t.status FROM tasks t JOIN task_links l ON l.parent_id = t.id \
                     WHERE l.child_id = ?",
                )?;
                let statuses: Vec<String> = stmt
                    .query_map(params![task_id], |r| r.get::<_, String>(0))?
                    .filter_map(|r| r.ok())
                    .collect();
                statuses.iter().all(|s| s == "done")
            };
            if all_done {
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                    params![task_id],
                )?;
                append_event(conn, &task_id, "promoted", None, None)?;
                promoted += 1;
            }
        }
        Ok(promoted)
    })
}

// ---------------------------------------------------------------------------
// Claim / complete / block
// ---------------------------------------------------------------------------

/// Atomically transition `ready -> running`. Returns the claimed task or
/// `None` if it was already claimed / not ready.
pub fn claim_task(
    conn: &Connection,
    task_id: &str,
    ttl_seconds: i64,
    claimer: Option<&str>,
) -> Result<Option<Task>> {
    let now = now_secs();
    let lock = claimer.map(|c| c.to_string()).unwrap_or_else(claimer_id);
    let expires = now + ttl_seconds;
    write_txn(conn, |conn| {
        let stale: Option<Option<i64>> = conn
            .query_row(
                "SELECT current_run_id FROM tasks WHERE id = ? AND status = 'ready'",
                params![task_id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(Some(rid)) = stale {
            if rid != 0 {
                conn.execute(
                    "UPDATE task_runs SET status = 'reclaimed', outcome = 'reclaimed', \
                     summary = COALESCE(summary, 'invariant recovery on re-claim'), \
                     ended_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL \
                     WHERE id = ? AND ended_at IS NULL",
                    params![now, rid],
                )?;
            }
        }
        let affected = conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, \
             started_at = COALESCE(started_at, ?) \
             WHERE id = ? AND status = 'ready' AND claim_lock IS NULL",
            params![lock, expires, now, task_id],
        )?;
        if affected != 1 {
            return Ok(None);
        }
        let trow: Option<(Option<String>, Option<i64>, Option<String>)> = conn
            .query_row(
                "SELECT assignee, max_runtime_seconds, current_step_key FROM tasks WHERE id = ?",
                params![task_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let (assignee, max_rt, step_key) = trow.unwrap_or((None, None, None));
        conn.execute(
            "INSERT INTO task_runs ( \
                task_id, profile, step_key, status, claim_lock, claim_expires, \
                max_runtime_seconds, started_at \
             ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
            params![task_id, assignee, step_key, lock, expires, max_rt, now],
        )?;
        let run_id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            params![run_id, task_id],
        )?;
        append_event(
            conn,
            task_id,
            "claimed",
            Some(&json!({ "lock": lock, "expires": expires, "run_id": run_id })),
            Some(run_id),
        )?;
        get_task(conn, task_id)
    })
}

/// Extend a running claim. Returns true if we still own it.
pub fn heartbeat_claim(
    conn: &Connection,
    task_id: &str,
    ttl_seconds: i64,
    claimer: Option<&str>,
) -> Result<bool> {
    let expires = now_secs() + ttl_seconds;
    let lock = claimer.map(|c| c.to_string()).unwrap_or_else(claimer_id);
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ? AND status = 'running' AND claim_lock = ?",
            params![expires, task_id, lock],
        )?;
        if affected == 1 {
            if let Some(run_id) = current_run_id(conn, task_id)? {
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    params![expires, run_id],
                )?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    })
}

/// Reset any `running` task whose claim has expired. Returns count reclaimed.
pub fn release_stale_claims(conn: &Connection) -> Result<usize> {
    let now = now_secs();
    write_txn(conn, |conn| {
        let stale: Vec<(String, Option<String>)> = {
            let mut stmt = conn.prepare(
                "SELECT id, claim_lock FROM tasks WHERE status = 'running' \
                 AND claim_expires IS NOT NULL AND claim_expires < ?",
            )?;
            stmt.query_map(params![now], |r| Ok((r.get(0)?, r.get(1)?)))?
                .filter_map(|r| r.ok())
                .collect()
        };
        let mut reclaimed = 0usize;
        for (id, claim_lock) in stale {
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, \
                 worker_pid = NULL WHERE id = ? AND status = 'running'",
                params![id],
            )?;
            let run_id = end_run(
                conn,
                &id,
                "reclaimed",
                Some("reclaimed"),
                None,
                Some(&format!("stale_lock={}", claim_lock.clone().unwrap_or_default())),
                None,
            )?;
            append_event(
                conn,
                &id,
                "reclaimed",
                Some(&json!({ "stale_lock": claim_lock })),
                run_id,
            )?;
            reclaimed += 1;
        }
        Ok(reclaimed)
    })
}

/// Operator-driven reclaim: release the claim and reset to `ready`.
pub fn reclaim_task(conn: &Connection, task_id: &str, reason: Option<&str>) -> Result<bool> {
    let did = write_txn(conn, |conn| {
        let row: Option<(String, Option<String>, Option<i64>)> = conn
            .query_row(
                "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
                params![task_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let (status, claim_lock, worker_pid) = match row {
            None => return Ok(false),
            Some(r) => r,
        };
        if status != "running" && claim_lock.is_none() {
            return Ok(false);
        }
        let prev_lock = claim_lock.clone();
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, \
             worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready', 'blocked')",
            params![task_id],
        )?;
        let err_text = match reason {
            Some(r) => format!("manual_reclaim: {r}"),
            None => format!("manual_reclaim lock={}", prev_lock.clone().unwrap_or_default()),
        };
        let run_id = end_run(
            conn,
            task_id,
            "reclaimed",
            Some("reclaimed"),
            None,
            Some(&err_text),
            None,
        )?;
        append_event(
            conn,
            task_id,
            "reclaimed",
            Some(&json!({
                "manual": true,
                "reason": reason,
                "prev_lock": prev_lock,
                "prev_pid": worker_pid,
            })),
            run_id,
        )?;
        Ok(true)
    })?;
    if did {
        clear_failure_counter(conn, task_id)?;
    }
    Ok(did)
}

/// Reassign a task, optionally reclaiming a stuck running worker first.
pub fn reassign_task(
    conn: &Connection,
    task_id: &str,
    profile: Option<&str>,
    reclaim_first: bool,
    reason: Option<&str>,
) -> Result<bool> {
    if reclaim_first {
        reclaim_task(conn, task_id, Some(reason.unwrap_or("reassign")))?;
    }
    match assign_task(conn, task_id, profile) {
        Ok(v) => Ok(v),
        Err(KanbanError::Runtime(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Partition `claimed_ids` into (verified, phantom). Never mutates.
fn verify_created_cards(
    conn: &Connection,
    completing_task_id: &str,
    claimed_ids: &[String],
) -> Result<(Vec<String>, Vec<String>)> {
    let claimed: Vec<String> = claimed_ids
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if claimed.is_empty() {
        return Ok((vec![], vec![]));
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut ordered: Vec<String> = Vec::new();
    for cid in &claimed {
        if !seen.contains(cid) {
            seen.insert(cid.clone());
            ordered.push(cid.clone());
        }
    }

    let completing_assignee: Option<Option<String>> = conn
        .query_row(
            "SELECT assignee FROM tasks WHERE id = ?",
            params![completing_task_id],
            |r| r.get(0),
        )
        .optional()?;
    let completing_assignee = match completing_assignee {
        None => return Ok((vec![], ordered)),
        Some(a) => a,
    };

    let placeholders = vec!["?"; ordered.len()].join(",");
    let sql = format!("SELECT id, created_by FROM tasks WHERE id IN ({placeholders})");
    let found: BTreeMap<String, Option<String>> = {
        let mut stmt = conn.prepare(&sql)?;
        stmt.query_map(params_from_iter(ordered.iter()), |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    let linked_children: HashSet<String> =
        child_ids(conn, completing_task_id)?.into_iter().collect();

    let mut verified: Vec<String> = Vec::new();
    let mut phantom: Vec<String> = Vec::new();
    for cid in &ordered {
        match found.get(cid) {
            None => phantom.push(cid.clone()),
            Some(created_by) => {
                let cb = created_by.clone();
                if completing_assignee.is_some()
                    && cb.as_deref() == completing_assignee.as_deref()
                {
                    verified.push(cid.clone());
                } else if cb.as_deref() == Some(completing_task_id) {
                    verified.push(cid.clone());
                } else if linked_children.contains(cid) {
                    verified.push(cid.clone());
                } else {
                    phantom.push(cid.clone());
                }
            }
        }
    }
    Ok((verified, phantom))
}

fn task_id_prose_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bt_[a-f0-9]{8,}\b").unwrap())
}

/// Regex-scan free-form text for `t_<hex>` references; return non-existent ids.
fn scan_prose_for_phantom_ids(conn: &Connection, text: &str) -> Result<Vec<String>> {
    if text.is_empty() {
        return Ok(vec![]);
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique: Vec<String> = Vec::new();
    for m in task_id_prose_re().find_iter(text) {
        let s = m.as_str().to_string();
        if !seen.contains(&s) {
            seen.insert(s.clone());
            unique.push(s);
        }
    }
    if unique.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = vec!["?"; unique.len()].join(",");
    let sql = format!("SELECT id FROM tasks WHERE id IN ({placeholders})");
    let existing: HashSet<String> = {
        let mut stmt = conn.prepare(&sql)?;
        stmt.query_map(params_from_iter(unique.iter()), |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect()
    };
    Ok(unique.into_iter().filter(|m| !existing.contains(m)).collect())
}

fn first_line_capped(text: &str, cap: usize) -> String {
    let line = text.trim().lines().next().unwrap_or("");
    line.chars().take(cap).collect()
}

/// Options for [`complete_task`].
#[derive(Debug, Clone, Default)]
pub struct CompleteOpts<'a> {
    pub result: Option<&'a str>,
    pub summary: Option<&'a str>,
    pub metadata: Option<Value>,
    pub created_cards: Option<Vec<String>>,
    pub expected_run_id: Option<i64>,
}

/// Transition `running|ready|blocked -> done` and record `result`.
pub fn complete_task(conn: &Connection, task_id: &str, opts: CompleteOpts<'_>) -> Result<bool> {
    let now = now_secs();

    let verified_cards: Vec<String> = match &opts.created_cards {
        Some(cards) if !cards.is_empty() => {
            let (verified, phantom) = verify_created_cards(conn, task_id, cards)?;
            if !phantom.is_empty() {
                let preview = opts
                    .summary
                    .or(opts.result)
                    .map(|s| first_line_capped(s, 200))
                    .filter(|s| !s.is_empty());
                write_txn(conn, |conn| {
                    append_event(
                        conn,
                        task_id,
                        "completion_blocked_hallucination",
                        Some(&json!({
                            "phantom_cards": phantom,
                            "verified_cards": verified,
                            "summary_preview": preview,
                        })),
                        None,
                    )
                })?;
                return Err(KanbanError::HallucinatedCards {
                    phantom,
                    completing_task_id: task_id.to_string(),
                });
            }
            verified
        }
        _ => vec![],
    };

    let summary_for_run = opts.summary.or(opts.result);
    let run_id = write_txn(conn, |conn| {
        let affected = match opts.expected_run_id {
            None => conn.execute(
                "UPDATE tasks SET status = 'done', result = ?, completed_at = ?, \
                 claim_lock = NULL, claim_expires = NULL, worker_pid = NULL \
                 WHERE id = ? AND status IN ('running', 'ready', 'blocked')",
                params![opts.result, now, task_id],
            )?,
            Some(rid) => conn.execute(
                "UPDATE tasks SET status = 'done', result = ?, completed_at = ?, \
                 claim_lock = NULL, claim_expires = NULL, worker_pid = NULL \
                 WHERE id = ? AND status IN ('running', 'ready', 'blocked') AND current_run_id = ?",
                params![opts.result, now, task_id, rid],
            )?,
        };
        if affected != 1 {
            return Ok(None);
        }
        let mut run_id = end_run(
            conn,
            task_id,
            "completed",
            Some("done"),
            summary_for_run,
            None,
            opts.metadata.as_ref(),
        )?;
        if run_id.is_none()
            && (opts.summary.is_some() || opts.metadata.is_some() || opts.result.is_some())
        {
            run_id = Some(synthesize_ended_run(
                conn,
                task_id,
                "completed",
                summary_for_run,
                None,
                opts.metadata.as_ref(),
            )?);
        }
        let ev_summary = summary_for_run
            .map(|s| first_line_capped(s, 400))
            .filter(|s| !s.is_empty());
        let mut completed_payload = Map::new();
        completed_payload.insert(
            "result_len".into(),
            Value::from(opts.result.map(|r| r.chars().count()).unwrap_or(0)),
        );
        completed_payload.insert(
            "summary".into(),
            ev_summary.map(Value::String).unwrap_or(Value::Null),
        );
        if !verified_cards.is_empty() {
            completed_payload.insert(
                "verified_cards".into(),
                Value::Array(verified_cards.iter().map(|c| Value::String(c.clone())).collect()),
            );
        }
        append_event(
            conn,
            task_id,
            "completed",
            Some(&Value::Object(completed_payload)),
            run_id,
        )?;
        Ok(Some(run_id))
    })?;

    let run_id = match run_id {
        None => return Ok(false),
        Some(r) => r,
    };

    // Prose-scan summary + result for phantom t_<hex> references (advisory).
    let scan_text = [opts.summary, opts.result]
        .into_iter()
        .flatten()
        .collect::<Vec<&str>>()
        .join(" ");
    if !scan_text.is_empty() {
        let mut phantom_refs = scan_prose_for_phantom_ids(conn, &scan_text)?;
        let verified_set: HashSet<&String> = verified_cards.iter().collect();
        phantom_refs.retain(|p| !verified_set.contains(p));
        if !phantom_refs.is_empty() {
            write_txn(conn, |conn| {
                append_event(
                    conn,
                    task_id,
                    "suspected_hallucinated_references",
                    Some(&json!({
                        "phantom_refs": phantom_refs,
                        "source": "completion_summary",
                    })),
                    run_id,
                )
            })?;
        }
    }

    clear_failure_counter(conn, task_id)?;
    recompute_ready(conn)?;
    Ok(true)
}

/// Backfill the user-visible result for an already completed task.
pub fn edit_completed_task_result(
    conn: &Connection,
    task_id: &str,
    result: &str,
    summary: Option<&str>,
    metadata: Option<Value>,
) -> Result<bool> {
    let handoff_summary = summary.unwrap_or(result);
    write_txn(conn, |conn| {
        let status: Option<String> = conn
            .query_row("SELECT status FROM tasks WHERE id = ?", params![task_id], |r| {
                r.get(0)
            })
            .optional()?;
        if status.as_deref() != Some("done") {
            return Ok(false);
        }
        conn.execute(
            "UPDATE tasks SET result = ? WHERE id = ?",
            params![result, task_id],
        )?;
        let existing_run: Option<i64> = conn
            .query_row(
                "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'completed' \
                 ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC LIMIT 1",
                params![task_id],
                |r| r.get(0),
            )
            .optional()?;
        let run_id = match existing_run {
            Some(rid) => {
                conn.execute(
                    "UPDATE task_runs SET summary = ? WHERE id = ?",
                    params![handoff_summary, rid],
                )?;
                if let Some(m) = &metadata {
                    conn.execute(
                        "UPDATE task_runs SET metadata = ? WHERE id = ?",
                        params![serde_json::to_string(m).unwrap_or_default(), rid],
                    )?;
                }
                rid
            }
            None => synthesize_ended_run(
                conn,
                task_id,
                "completed",
                Some(handoff_summary),
                None,
                metadata.as_ref(),
            )?,
        };
        let ev_summary = first_line_capped(handoff_summary, 400);
        let mut fields = vec![
            Value::String("result".into()),
            Value::String("summary".into()),
        ];
        if metadata.is_some() {
            fields.push(Value::String("metadata".into()));
        }
        append_event(
            conn,
            task_id,
            "edited",
            Some(&json!({
                "fields": fields,
                "result_len": result.chars().count(),
                "summary": if ev_summary.is_empty() { Value::Null } else { Value::String(ev_summary) },
            })),
            Some(run_id),
        )?;
        Ok(true)
    })
}

/// Transition `running|ready -> blocked`.
pub fn block_task(
    conn: &Connection,
    task_id: &str,
    reason: Option<&str>,
    expected_run_id: Option<i64>,
) -> Result<bool> {
    write_txn(conn, |conn| {
        let affected = match expected_run_id {
            None => conn.execute(
                "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, \
                 worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready')",
                params![task_id],
            )?,
            Some(rid) => conn.execute(
                "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, \
                 worker_pid = NULL WHERE id = ? AND status IN ('running', 'ready') \
                 AND current_run_id = ?",
                params![task_id, rid],
            )?,
        };
        if affected != 1 {
            return Ok(false);
        }
        let mut run_id = end_run(conn, task_id, "blocked", Some("blocked"), reason, None, None)?;
        if run_id.is_none() {
            if let Some(r) = reason {
                if !r.is_empty() {
                    run_id = Some(synthesize_ended_run(
                        conn, task_id, "blocked", Some(r), None, None,
                    )?);
                }
            }
        }
        append_event(
            conn,
            task_id,
            "blocked",
            Some(&json!({ "reason": reason })),
            run_id,
        )?;
        Ok(true)
    })
}

/// Transition `blocked -> ready`.
pub fn unblock_task(conn: &Connection, task_id: &str) -> Result<bool> {
    let now = now_secs();
    write_txn(conn, |conn| {
        let stale: Option<Option<i64>> = conn
            .query_row(
                "SELECT current_run_id FROM tasks WHERE id = ? AND status = 'blocked'",
                params![task_id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(Some(rid)) = stale {
            if rid != 0 {
                conn.execute(
                    "UPDATE task_runs SET status = 'reclaimed', outcome = 'reclaimed', \
                     summary = COALESCE(summary, 'invariant recovery on unblock'), ended_at = ?, \
                     claim_lock = NULL, claim_expires = NULL, worker_pid = NULL \
                     WHERE id = ? AND ended_at IS NULL",
                    params![now, rid],
                )?;
            }
        }
        let affected = conn.execute(
            "UPDATE tasks SET status = 'ready', current_run_id = NULL \
             WHERE id = ? AND status = 'blocked'",
            params![task_id],
        )?;
        if affected != 1 {
            return Ok(false);
        }
        append_event(conn, task_id, "unblocked", None, None)?;
        Ok(true)
    })
}

pub fn archive_task(conn: &Connection, task_id: &str) -> Result<bool> {
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "UPDATE tasks SET status = 'archived', claim_lock = NULL, claim_expires = NULL, \
             worker_pid = NULL WHERE id = ? AND status != 'archived'",
            params![task_id],
        )?;
        if affected != 1 {
            return Ok(false);
        }
        let run_id = end_run(
            conn,
            task_id,
            "reclaimed",
            Some("reclaimed"),
            Some("task archived with run still active"),
            None,
            None,
        )?;
        append_event(conn, task_id, "archived", None, run_id)?;
        Ok(true)
    })
}

// ---------------------------------------------------------------------------
// Workspace resolution
// ---------------------------------------------------------------------------

/// Resolve (and create if needed) the workspace for a task.
pub fn resolve_workspace(task: &Task, board: Option<&str>) -> Result<PathBuf> {
    let kind = if task.workspace_kind.is_empty() {
        "scratch"
    } else {
        task.workspace_kind.as_str()
    };
    match kind {
        "scratch" => {
            let p = if let Some(wp) = task.workspace_path.as_deref().filter(|s| !s.is_empty()) {
                let p = expanduser(wp);
                if !p.is_absolute() {
                    return Err(KanbanError::Value(format!(
                        "task {} has non-absolute workspace_path {:?}; \
                         workspace paths must be absolute",
                        task.id, wp
                    )));
                }
                p
            } else {
                workspaces_root(board).join(&task.id)
            };
            fs::create_dir_all(&p).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
            Ok(p)
        }
        "dir" => {
            let wp = task
                .workspace_path
                .as_deref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    KanbanError::Value(format!(
                        "task {} has workspace_kind=dir but no workspace_path",
                        task.id
                    ))
                })?;
            let p = expanduser(wp);
            if !p.is_absolute() {
                return Err(KanbanError::Value(format!(
                    "task {} has non-absolute workspace_path {:?}; use an absolute path \
                     (relative paths are ambiguous against the dispatcher's CWD)",
                    task.id, wp
                )));
            }
            fs::create_dir_all(&p).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
            Ok(p)
        }
        "worktree" => {
            match task.workspace_path.as_deref().filter(|s| !s.is_empty()) {
                None => {
                    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    Ok(cwd.join(".worktrees").join(&task.id))
                }
                Some(wp) => {
                    let p = expanduser(wp);
                    if !p.is_absolute() {
                        return Err(KanbanError::Value(format!(
                            "task {} has non-absolute worktree path {:?}; use an absolute path",
                            task.id, wp
                        )));
                    }
                    Ok(p)
                }
            }
        }
        other => Err(KanbanError::Value(format!("unknown workspace_kind: {other}"))),
    }
}

pub fn set_workspace_path(conn: &Connection, task_id: &str, path: &str) -> Result<()> {
    write_txn(conn, |conn| {
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            params![path, task_id],
        )?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Dispatcher (one-shot pass)
// ---------------------------------------------------------------------------

/// Outcome of a single dispatch pass.
#[derive(Debug, Clone, Default)]
pub struct DispatchResult {
    pub reclaimed: usize,
    pub promoted: usize,
    /// `(task_id, assignee, workspace_path)` triples.
    pub spawned: Vec<(String, String, String)>,
    pub skipped_unassigned: Vec<String>,
    pub skipped_nonspawnable: Vec<String>,
    pub crashed: Vec<String>,
    pub auto_blocked: Vec<String>,
    pub timed_out: Vec<String>,
}

/// Return true if `pid` is still running on this host (POSIX, with zombie
/// detection on Linux/macOS).
pub fn pid_alive(pid: Option<i64>) -> bool {
    let pid = match pid {
        Some(p) if p > 0 => p,
        _ => return false,
    };
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EPERM) => return true, // exists, can't signal
            _ => return false,                // ESRCH or other → dead
        }
    }
    // kill(0) succeeded — probe for zombie state.
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string(format!("/proc/{pid}/status")) {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("State:") {
                    if rest.contains('Z') {
                        return false;
                    }
                    break;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
        {
            if !out.status.success() {
                return false;
            }
            if String::from_utf8_lossy(&out.stdout).trim().contains('Z') {
                return false;
            }
        }
    }
    true
}

/// Record a heartbeat event + touch `last_heartbeat_at`.
pub fn heartbeat_worker(
    conn: &Connection,
    task_id: &str,
    note: Option<&str>,
    expected_run_id: Option<i64>,
) -> Result<bool> {
    let now = now_secs();
    write_txn(conn, |conn| {
        let affected = match expected_run_id {
            None => conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running'",
                params![now, task_id],
            )?,
            Some(rid) => conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running' \
                 AND current_run_id = ?",
                params![now, task_id, rid],
            )?,
        };
        if affected != 1 {
            return Ok(false);
        }
        let run_id = match expected_run_id {
            Some(rid) => Some(rid),
            None => current_run_id(conn, task_id)?,
        };
        if let Some(rid) = run_id {
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                params![now, rid],
            )?;
        }
        let payload = note.map(|n| json!({ "note": n }));
        append_event(conn, task_id, "heartbeat", payload.as_ref(), run_id)?;
        Ok(true)
    })
}

/// Set or clear the per-task max_runtime_seconds. Returns true on success.
pub fn set_max_runtime(conn: &Connection, task_id: &str, seconds: Option<i64>) -> Result<bool> {
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "UPDATE tasks SET max_runtime_seconds = ? WHERE id = ?",
            params![seconds, task_id],
        )?;
        Ok(affected == 1)
    })
}

/// Reclaim `running` tasks whose worker PID is no longer alive (host-local).
pub fn detect_crashed_workers(conn: &Connection) -> Result<Vec<String>> {
    let mut crashed: Vec<String> = Vec::new();
    let mut crash_details: Vec<(String, i64, String)> = Vec::new();
    let host_prefix = format!("{}:", claimer_id().split(':').next().unwrap_or(""));
    write_txn(conn, |conn| {
        let rows: Vec<(String, Option<i64>, Option<String>)> = {
            let mut stmt = conn.prepare(
                "SELECT id, worker_pid, claim_lock FROM tasks \
                 WHERE status = 'running' AND worker_pid IS NOT NULL",
            )?;
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                .filter_map(|r| r.ok())
                .collect()
        };
        for (id, worker_pid, claim_lock) in rows {
            let lock = claim_lock.clone().unwrap_or_default();
            if !lock.starts_with(&host_prefix) {
                continue;
            }
            if pid_alive(worker_pid) {
                continue;
            }
            let pid = worker_pid.unwrap_or(0);
            let affected = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, \
                 worker_pid = NULL WHERE id = ? AND status = 'running'",
                params![id],
            )?;
            if affected == 1 {
                let run_id = end_run(
                    conn,
                    &id,
                    "crashed",
                    Some("crashed"),
                    None,
                    Some(&format!("pid {pid} not alive")),
                    Some(&json!({ "pid": pid, "claimer": claim_lock })),
                )?;
                append_event(
                    conn,
                    &id,
                    "crashed",
                    Some(&json!({ "pid": pid, "claimer": claim_lock })),
                    run_id,
                )?;
                crashed.push(id.clone());
                crash_details.push((id, pid, lock));
            }
        }
        Ok(())
    })?;
    for (tid, pid, claimer) in crash_details {
        record_task_failure(
            conn,
            &tid,
            &format!("pid {pid} not alive"),
            "crashed",
            None,
            false,
            false,
            Some(json!({ "pid": pid, "claimer": claimer })),
        )?;
    }
    Ok(crashed)
}

/// Terminate workers whose `max_runtime_seconds` elapsed (SIGTERM → SIGKILL).
/// Returns ids that were timed out. `signal_fn` is a test hook.
pub fn enforce_max_runtime(
    conn: &Connection,
    signal_fn: Option<&dyn Fn(i64, i32)>,
) -> Result<Vec<String>> {
    let mut timed_out: Vec<String> = Vec::new();
    let now = now_secs();
    let host_prefix = format!("{}:", claimer_id().split(':').next().unwrap_or(""));

    let rows: Vec<(String, Option<i64>, Option<i64>, Option<i64>, Option<String>)> = {
        let mut stmt = conn.prepare(
            "SELECT t.id, t.worker_pid, COALESCE(r.started_at, t.started_at) AS active_started_at, \
             t.max_runtime_seconds, t.claim_lock FROM tasks t \
             LEFT JOIN task_runs r ON r.id = t.current_run_id \
             WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL \
             AND COALESCE(r.started_at, t.started_at) IS NOT NULL AND t.worker_pid IS NOT NULL",
        )?;
        stmt.query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    for (tid, worker_pid, active_started_at, max_rt, claim_lock) in rows {
        let lock = claim_lock.unwrap_or_default();
        if !lock.starts_with(&host_prefix) {
            continue;
        }
        let started = match active_started_at {
            Some(s) => s,
            None => continue,
        };
        let limit = max_rt.unwrap_or(0);
        let elapsed = now - started;
        if elapsed < limit {
            continue;
        }
        let pid = worker_pid.unwrap_or(0);

        let mut killed = false;
        let send = |sig: i32| {
            if let Some(f) = signal_fn {
                f(pid, sig);
            } else {
                unsafe {
                    libc::kill(pid as libc::pid_t, sig);
                }
            }
        };
        send(libc::SIGTERM);
        for _ in 0..10 {
            if !pid_alive(Some(pid)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        if pid_alive(Some(pid)) {
            send(libc::SIGKILL);
            killed = true;
        }

        let updated = write_txn(conn, |conn| {
            let affected = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, \
                 worker_pid = NULL, last_heartbeat_at = NULL WHERE id = ? AND status = 'running'",
                params![tid],
            )?;
            if affected == 1 {
                let payload = json!({
                    "pid": pid,
                    "elapsed_seconds": elapsed,
                    "limit_seconds": limit,
                    "sigkill": killed,
                });
                let run_id = end_run(
                    conn,
                    &tid,
                    "timed_out",
                    Some("timed_out"),
                    None,
                    Some(&format!("elapsed {elapsed}s > limit {limit}s")),
                    Some(&payload),
                )?;
                append_event(conn, &tid, "timed_out", Some(&payload), run_id)?;
                timed_out.push(tid.clone());
                Ok(true)
            } else {
                Ok(false)
            }
        })?;
        if updated {
            record_task_failure(
                conn,
                &tid,
                &format!("elapsed {elapsed}s > limit {limit}s"),
                "timed_out",
                None,
                false,
                false,
                Some(json!({ "pid": pid, "sigkill": killed })),
            )?;
        }
    }
    Ok(timed_out)
}

fn truncate_500(s: &str) -> String {
    s.chars().take(500).collect()
}

/// Record a non-success outcome and maybe trip the circuit breaker. Returns
/// true when the task was auto-blocked.
#[allow(clippy::too_many_arguments)]
pub fn record_task_failure(
    conn: &Connection,
    task_id: &str,
    error: &str,
    outcome: &str,
    failure_limit: Option<i64>,
    release_claim: bool,
    end_run_flag: bool,
    event_payload_extra: Option<Value>,
) -> Result<bool> {
    let failure_limit = failure_limit.unwrap_or(DEFAULT_FAILURE_LIMIT);
    let err500 = truncate_500(error);
    write_txn(conn, |conn| {
        let row: Option<(i64, String)> = conn
            .query_row(
                "SELECT consecutive_failures, status FROM tasks WHERE id = ?",
                params![task_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (prev_failures, _status) = match row {
            None => return Ok(false),
            Some(r) => r,
        };
        let failures = prev_failures + 1;

        if failures >= failure_limit {
            if release_claim {
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, \
                     worker_pid = NULL, consecutive_failures = ?, last_failure_error = ? \
                     WHERE id = ? AND status IN ('running', 'ready')",
                    params![failures, err500, task_id],
                )?;
            } else {
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', consecutive_failures = ?, \
                     last_failure_error = ? WHERE id = ? AND status IN ('ready', 'running')",
                    params![failures, err500, task_id],
                )?;
            }
            let mut run_id = None;
            if end_run_flag {
                run_id = end_run(
                    conn,
                    task_id,
                    "gave_up",
                    Some("gave_up"),
                    None,
                    Some(&err500),
                    Some(&json!({ "failures": failures, "trigger_outcome": outcome })),
                )?;
            }
            let mut payload = Map::new();
            payload.insert("failures".into(), Value::from(failures));
            payload.insert("error".into(), Value::String(err500.clone()));
            payload.insert("trigger_outcome".into(), Value::String(outcome.into()));
            if let Some(Value::Object(extra)) = &event_payload_extra {
                for (k, v) in extra {
                    payload.insert(k.clone(), v.clone());
                }
            }
            append_event(conn, task_id, "gave_up", Some(&Value::Object(payload)), run_id)?;
            Ok(true)
        } else {
            if release_claim {
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, \
                     worker_pid = NULL, consecutive_failures = ?, last_failure_error = ? \
                     WHERE id = ? AND status = 'running'",
                    params![failures, err500, task_id],
                )?;
            } else {
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, last_failure_error = ? WHERE id = ?",
                    params![failures, err500, task_id],
                )?;
            }
            if end_run_flag {
                let run_id = end_run(
                    conn,
                    task_id,
                    outcome,
                    Some(outcome),
                    None,
                    Some(&err500),
                    Some(&json!({ "failures": failures })),
                )?;
                append_event(
                    conn,
                    task_id,
                    outcome,
                    Some(&json!({ "error": err500, "failures": failures })),
                    run_id,
                )?;
            }
            Ok(false)
        }
    })
}

/// Spawn-failure path: convenience wrapper over [`record_task_failure`].
pub fn record_spawn_failure(
    conn: &Connection,
    task_id: &str,
    error: &str,
    failure_limit: Option<i64>,
) -> Result<bool> {
    record_task_failure(
        conn,
        task_id,
        error,
        "spawn_failed",
        failure_limit,
        true,
        true,
        None,
    )
}

fn set_worker_pid(conn: &Connection, task_id: &str, pid: i64) -> Result<()> {
    write_txn(conn, |conn| {
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            params![pid, task_id],
        )?;
        let run_id = current_run_id(conn, task_id)?;
        if let Some(rid) = run_id {
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                params![pid, rid],
            )?;
        }
        append_event(conn, task_id, "spawned", Some(&json!({ "pid": pid })), run_id)?;
        Ok(())
    })
}

/// Reset the unified consecutive-failures counter.
pub fn clear_failure_counter(conn: &Connection, task_id: &str) -> Result<()> {
    write_txn(conn, |conn| {
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, last_failure_error = NULL WHERE id = ?",
            params![task_id],
        )?;
        Ok(())
    })
}

/// True iff there is at least one ready+assigned+unclaimed task. (The Python
/// version further filters by `profile_exists`; that hook is provided as the
/// optional `profile_exists` callback.)
pub fn has_spawnable_ready(
    conn: &Connection,
    profile_exists: Option<&dyn Fn(&str) -> bool>,
) -> Result<bool> {
    let assignees: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT assignee FROM tasks WHERE status = 'ready' \
             AND assignee IS NOT NULL AND claim_lock IS NULL",
        )?;
        stmt.query_map([], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect()
    };
    if assignees.is_empty() {
        return Ok(false);
    }
    match profile_exists {
        None => Ok(true),
        Some(check) => Ok(assignees.iter().any(|a| check(a))),
    }
}

/// Options for [`dispatch_once`].
pub struct DispatchOpts<'a> {
    pub ttl_seconds: i64,
    pub dry_run: bool,
    pub max_spawn: Option<usize>,
    pub failure_limit: i64,
    pub board: Option<&'a str>,
    /// Profile-existence check (Python's `profile_exists`). `None` disables
    /// the non-spawnable skip.
    pub profile_exists: Option<&'a dyn Fn(&str) -> bool>,
    /// Spawn callback: `(task, workspace, board) -> Option<pid>`. `None`
    /// uses [`default_spawn`].
    pub spawn_fn: Option<&'a dyn Fn(&Task, &str, Option<&str>) -> Result<Option<i64>>>,
}

impl Default for DispatchOpts<'_> {
    fn default() -> Self {
        DispatchOpts {
            ttl_seconds: DEFAULT_CLAIM_TTL_SECONDS,
            dry_run: false,
            max_spawn: None,
            failure_limit: DEFAULT_SPAWN_FAILURE_LIMIT,
            board: None,
            profile_exists: None,
            spawn_fn: None,
        }
    }
}

/// Run one dispatcher tick.
pub fn dispatch_once(conn: &Connection, opts: DispatchOpts<'_>) -> Result<DispatchResult> {
    let mut result = DispatchResult {
        reclaimed: release_stale_claims(conn)?,
        crashed: detect_crashed_workers(conn)?,
        timed_out: enforce_max_runtime(conn, None)?,
        promoted: recompute_ready(conn)?,
        ..Default::default()
    };

    let ready_rows: Vec<(String, Option<String>)> = {
        let mut stmt = conn.prepare(
            "SELECT id, assignee FROM tasks WHERE status = 'ready' AND claim_lock IS NULL \
             ORDER BY priority DESC, created_at ASC",
        )?;
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect()
    };

    let mut spawned = 0usize;
    for (id, assignee) in ready_rows {
        if let Some(cap) = opts.max_spawn {
            if spawned >= cap {
                break;
            }
        }
        let assignee = match assignee.filter(|a| !a.is_empty()) {
            None => {
                result.skipped_unassigned.push(id);
                continue;
            }
            Some(a) => a,
        };
        if let Some(check) = opts.profile_exists {
            if !check(&assignee) {
                result.skipped_nonspawnable.push(id);
                continue;
            }
        }
        if opts.dry_run {
            result.spawned.push((id, assignee, String::new()));
            continue;
        }
        let claimed = match claim_task(conn, &id, opts.ttl_seconds, None)? {
            None => continue,
            Some(t) => t,
        };
        let workspace = match resolve_workspace(&claimed, opts.board) {
            Ok(w) => w,
            Err(exc) => {
                let auto = record_spawn_failure(
                    conn,
                    &claimed.id,
                    &format!("workspace: {exc}"),
                    Some(opts.failure_limit),
                )?;
                if auto {
                    result.auto_blocked.push(claimed.id);
                }
                continue;
            }
        };
        let workspace_str = workspace.to_string_lossy().into_owned();
        set_workspace_path(conn, &claimed.id, &workspace_str)?;
        let spawn_res = match opts.spawn_fn {
            Some(f) => f(&claimed, &workspace_str, opts.board),
            None => default_spawn(&claimed, &workspace_str, opts.board),
        };
        match spawn_res {
            Ok(pid) => {
                if let Some(pid) = pid {
                    if pid != 0 {
                        set_worker_pid(conn, &claimed.id, pid)?;
                    }
                }
                result.spawned.push((
                    claimed.id.clone(),
                    claimed.assignee.clone().unwrap_or_default(),
                    workspace_str,
                ));
                spawned += 1;
            }
            Err(exc) => {
                let auto = record_spawn_failure(
                    conn,
                    &claimed.id,
                    &exc.to_string(),
                    Some(opts.failure_limit),
                )?;
                if auto {
                    result.auto_blocked.push(claimed.id);
                }
            }
        }
    }
    Ok(result)
}

/// Rotate `<log>` to `<log>.1` if it exceeds `max_bytes`.
pub fn rotate_worker_log(log_path: &Path, max_bytes: u64) {
    let meta = match fs::metadata(log_path) {
        Ok(m) => m,
        Err(_) => return,
    };
    if meta.len() <= max_bytes {
        return;
    }
    let rotated = {
        let mut s = log_path.as_os_str().to_os_string();
        s.push(".1");
        PathBuf::from(s)
    };
    let _ = fs::remove_file(&rotated);
    let _ = fs::rename(log_path, &rotated);
}

/// Fire-and-forget `hermes -p <profile> chat -q ...` subprocess. Returns the
/// child PID. Mirrors the Python `_default_spawn`.
pub fn default_spawn(task: &Task, workspace: &str, board: Option<&str>) -> Result<Option<i64>> {
    use std::process::{Command, Stdio};

    let assignee = task
        .assignee
        .as_deref()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| KanbanError::Value(format!("task {} has no assignee", task.id)))?;
    let profile_arg = assignee.trim().to_lowercase();

    let prompt = format!("work kanban task {}", task.id);

    let log_dir = worker_logs_dir(board);
    fs::create_dir_all(&log_dir).map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    let log_path = log_dir.join(format!("{}.log", task.id));
    rotate_worker_log(&log_path, DEFAULT_LOG_ROTATE_BYTES);
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| KanbanError::Sqlite(e.to_string()))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| KanbanError::Sqlite(e.to_string()))?;

    let mut cmd = Command::new("hermes");
    cmd.arg("-p")
        .arg(&profile_arg)
        .arg("--skills")
        .arg("kanban-worker");
    if let Some(skills) = &task.skills {
        for sk in skills {
            if !sk.is_empty() && sk != "kanban-worker" {
                cmd.arg("--skills").arg(sk);
            }
        }
    }
    cmd.arg("chat").arg("-q").arg(&prompt);

    if Path::new(workspace).is_dir() {
        cmd.current_dir(workspace);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));

    if let Some(tenant) = task.tenant.as_deref().filter(|t| !t.is_empty()) {
        cmd.env("HERMES_TENANT", tenant);
    }
    cmd.env("HERMES_KANBAN_TASK", &task.id);
    cmd.env("HERMES_KANBAN_WORKSPACE", workspace);
    if let Some(rid) = task.current_run_id {
        cmd.env("HERMES_KANBAN_RUN_ID", rid.to_string());
    }
    if let Some(lock) = task.claim_lock.as_deref().filter(|l| !l.is_empty()) {
        cmd.env("HERMES_KANBAN_CLAIM_LOCK", lock);
    }
    cmd.env(
        "HERMES_KANBAN_DB",
        kanban_db_path(board).to_string_lossy().into_owned(),
    );
    cmd.env(
        "HERMES_KANBAN_WORKSPACES_ROOT",
        workspaces_root(board).to_string_lossy().into_owned(),
    );
    let resolved_board = normalize_board_slug(board)
        .ok()
        .flatten()
        .unwrap_or_else(get_current_board);
    cmd.env("HERMES_KANBAN_BOARD", &resolved_board);
    cmd.env("HERMES_PROFILE", &profile_arg);

    // start_new_session=True equivalent: detach into its own session.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    match cmd.spawn() {
        Ok(child) => Ok(Some(child.id() as i64)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(KanbanError::Runtime(
            "`hermes` executable not found on PATH. Install Hermes Agent or activate \
             its venv before running the kanban dispatcher."
                .into(),
        )),
        Err(e) => Err(KanbanError::Sqlite(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Worker context builder
// ---------------------------------------------------------------------------

fn cap(s: Option<&str>, limit: usize) -> String {
    let s = match s {
        None => return String::new(),
        Some(s) => s,
    };
    let s = s.trim();
    let count = s.chars().count();
    if count <= limit {
        return s.to_string();
    }
    let head: String = s.chars().take(limit).collect();
    format!("{head}… [truncated, {} chars omitted]", count - limit)
}

fn fmt_local(ts: i64) -> String {
    match Local.timestamp_opt(ts, 0).single() {
        Some(dt) => dt.format("%Y-%m-%d %H:%M").to_string(),
        None => ts.to_string(),
    }
}

/// Return the full text a worker should read to understand its task.
pub fn build_worker_context(conn: &Connection, task_id: &str) -> Result<String> {
    let task = get_task(conn, task_id)?
        .ok_or_else(|| KanbanError::Value(format!("unknown task {task_id}")))?;

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("# Kanban task {}: {}", task.id, task.title));
    lines.push(String::new());
    lines.push(format!(
        "Assignee: {}",
        task.assignee.clone().unwrap_or_else(|| "(unassigned)".into())
    ));
    lines.push(format!("Status:   {}", task.status));
    if let Some(t) = task.tenant.as_deref().filter(|t| !t.is_empty()) {
        lines.push(format!("Tenant:   {t}"));
    }
    lines.push(format!(
        "Workspace: {} @ {}",
        task.workspace_kind,
        task.workspace_path.clone().unwrap_or_else(|| "(unresolved)".into())
    ));
    lines.push(String::new());

    if let Some(body) = task.body.as_deref().filter(|b| !b.trim().is_empty()) {
        lines.push("## Body".into());
        lines.push(cap(Some(body), CTX_MAX_BODY_BYTES));
        lines.push(String::new());
    }

    // Prior attempts.
    let all_prior: Vec<Run> = list_runs(conn, task_id, true)?
        .into_iter()
        .filter(|r| r.ended_at.is_some())
        .collect();
    let (omitted, shown, first_shown_idx): (usize, Vec<Run>, usize) =
        if all_prior.len() > CTX_MAX_PRIOR_ATTEMPTS {
            let omitted = all_prior.len() - CTX_MAX_PRIOR_ATTEMPTS;
            let shown = all_prior[all_prior.len() - CTX_MAX_PRIOR_ATTEMPTS..].to_vec();
            (omitted, shown, omitted + 1)
        } else {
            (0, all_prior, 1)
        };
    if !shown.is_empty() {
        lines.push("## Prior attempts on this task".into());
        if omitted > 0 {
            lines.push(format!(
                "_({omitted} earlier attempt{} omitted; showing most recent {})_",
                if omitted != 1 { "s" } else { "" },
                shown.len()
            ));
        }
        for (offset, run) in shown.iter().enumerate() {
            let idx = first_shown_idx + offset;
            let ts = fmt_local(run.started_at);
            let profile = run.profile.clone().unwrap_or_else(|| "(unknown)".into());
            let outcome = run.outcome.clone().unwrap_or_else(|| run.status.clone());
            lines.push(format!("### Attempt {idx} — {outcome} ({profile}, {ts})"));
            if let Some(s) = run.summary.as_deref().filter(|s| !s.trim().is_empty()) {
                lines.push(cap(Some(s), CTX_MAX_FIELD_BYTES));
            }
            if let Some(e) = run.error.as_deref().filter(|e| !e.trim().is_empty()) {
                lines.push(format!("_error_: {}", cap(Some(e), CTX_MAX_FIELD_BYTES)));
            }
            if let Some(meta) = &run.metadata {
                if let Ok(meta_str) = serde_json::to_string(&sorted_json(meta)) {
                    lines.push(format!("_metadata_: `{}`", cap(Some(&meta_str), CTX_MAX_FIELD_BYTES)));
                }
            }
            lines.push(String::new());
        }
    }

    // Parents.
    let parent_id_list = parent_ids(conn, task_id)?;
    if !parent_id_list.is_empty() {
        let mut wrote_header = false;
        for pid in &parent_id_list {
            let pt = match get_task(conn, pid)? {
                Some(t) if t.status == "done" => t,
                _ => continue,
            };
            let mut runs: Vec<Run> = list_runs(conn, pid, true)?
                .into_iter()
                .filter(|r| r.outcome.as_deref() == Some("completed"))
                .collect();
            runs.sort_by(|a, b| b.started_at.cmp(&a.started_at));
            let run = runs.into_iter().next();

            if !wrote_header {
                lines.push("## Parent task results".into());
                wrote_header = true;
            }
            lines.push(format!("### {pid}"));

            let mut body_lines: Vec<String> = Vec::new();
            if let Some(r) = &run {
                if let Some(s) = r.summary.as_deref().filter(|s| !s.trim().is_empty()) {
                    body_lines.push(cap(Some(s), CTX_MAX_FIELD_BYTES));
                } else if let Some(res) = pt.result.as_deref() {
                    body_lines.push(cap(Some(res), CTX_MAX_FIELD_BYTES));
                } else {
                    body_lines.push("(no result recorded)".into());
                }
            } else if let Some(res) = pt.result.as_deref() {
                body_lines.push(cap(Some(res), CTX_MAX_FIELD_BYTES));
            } else {
                body_lines.push("(no result recorded)".into());
            }
            if let Some(r) = &run {
                if let Some(meta) = &r.metadata {
                    if let Ok(meta_str) = serde_json::to_string(&sorted_json(meta)) {
                        body_lines.push(format!(
                            "_metadata_: `{}`",
                            cap(Some(&meta_str), CTX_MAX_FIELD_BYTES)
                        ));
                    }
                }
            }
            lines.extend(body_lines);
            lines.push(String::new());
        }
    }

    // Cross-task role history.
    if let Some(assignee) = task.assignee.as_deref().filter(|a| !a.is_empty()) {
        let role_rows: Vec<(String, String, Option<String>, i64)> = {
            let mut stmt = conn.prepare(
                "SELECT t.id, t.title, r.summary, r.ended_at FROM task_runs r \
                 JOIN tasks t ON r.task_id = t.id WHERE r.profile = ? AND r.task_id != ? \
                 AND r.outcome = 'completed' ORDER BY r.ended_at DESC LIMIT 5",
            )?;
            stmt.query_map(params![assignee, task_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get::<_, Option<i64>>(3)?.unwrap_or(0)))
            })?
            .filter_map(|r| r.ok())
            .collect()
        };
        if !role_rows.is_empty() {
            lines.push(format!("## Recent work by @{assignee}"));
            for (id, title, summary, ended_at) in role_rows {
                let ts = fmt_local(ended_at);
                let first = summary
                    .as_deref()
                    .map(|s| s.trim().lines().next().unwrap_or("").chars().take(200).collect::<String>())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "(no summary)".into());
                lines.push(format!("- {id} — {title} ({ts}): {first}"));
            }
            lines.push(String::new());
        }
    }

    // Comments.
    let all_comments = list_comments(conn, task_id)?;
    let (omitted_c, shown_c): (usize, Vec<Comment>) = if all_comments.len() > CTX_MAX_COMMENTS {
        let omitted = all_comments.len() - CTX_MAX_COMMENTS;
        (omitted, all_comments[all_comments.len() - CTX_MAX_COMMENTS..].to_vec())
    } else {
        (0, all_comments)
    };
    if !shown_c.is_empty() {
        lines.push("## Comment thread".into());
        if omitted_c > 0 {
            lines.push(format!(
                "_({omitted_c} earlier comment{} omitted; showing most recent {})_",
                if omitted_c != 1 { "s" } else { "" },
                shown_c.len()
            ));
        }
        for c in &shown_c {
            let ts = fmt_local(c.created_at);
            lines.push(format!("**{}** ({ts}):", c.author));
            lines.push(cap(Some(&c.body), CTX_MAX_COMMENT_BYTES));
            lines.push(String::new());
        }
    }

    Ok(format!("{}\n", lines.join("\n").trim_end()))
}

/// Recursively sort object keys to match Python's `json.dumps(sort_keys=True)`.
fn sorted_json(v: &Value) -> Value {
    match v {
        Value::Object(o) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, val) in o {
                sorted.insert(k.clone(), sorted_json(val));
            }
            let mut map = Map::new();
            for (k, val) in sorted {
                map.insert(k, val);
            }
            Value::Object(map)
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted_json).collect()),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Stats + SLA helpers
// ---------------------------------------------------------------------------

/// Per-status + per-assignee counts + oldest ready age.
pub fn board_stats(conn: &Connection) -> Result<Value> {
    let mut by_status: BTreeMap<String, i64> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT status, COUNT(*) AS n FROM tasks WHERE status != 'archived' GROUP BY status",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        for r in rows.flatten() {
            by_status.insert(r.0, r.1);
        }
    }

    let mut by_assignee: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT assignee, status, COUNT(*) AS n FROM tasks \
             WHERE status != 'archived' AND assignee IS NOT NULL GROUP BY assignee, status",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;
        for r in rows.flatten() {
            by_assignee.entry(r.0).or_default().insert(r.1, r.2);
        }
    }

    let oldest: Option<i64> = conn
        .query_row(
            "SELECT MIN(created_at) AS ts FROM tasks WHERE status = 'ready'",
            [],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()?
        .flatten();
    let now = now_secs();
    let oldest_ready_age = oldest.map(|ts| now - ts);

    Ok(json!({
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }))
}

/// Age metrics for a single task (seconds or null).
pub fn task_age(task: &Task) -> Value {
    let now = now_secs();
    let age_since_created = if task.created_at != 0 {
        Some(now - task.created_at)
    } else {
        None
    };
    let age_since_started = task.started_at.map(|s| now - s);
    let time_to_complete = task.completed_at.map(|c| {
        let base = task.started_at.unwrap_or(task.created_at);
        c - base
    });
    json!({
        "created_age_seconds": age_since_created,
        "started_age_seconds": age_since_started,
        "time_to_complete_seconds": time_to_complete,
    })
}

// ---------------------------------------------------------------------------
// Notification subscriptions
// ---------------------------------------------------------------------------

pub fn add_notify_sub(
    conn: &Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
) -> Result<()> {
    let now = now_secs();
    write_txn(conn, |conn| {
        conn.execute(
            "INSERT OR IGNORE INTO kanban_notify_subs \
             (task_id, platform, chat_id, thread_id, user_id, created_at) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![task_id, platform, chat_id, thread_id.unwrap_or(""), user_id, now],
        )?;
        Ok(())
    })
}

pub fn list_notify_subs(conn: &Connection, task_id: Option<&str>) -> Result<Vec<Map<String, Value>>> {
    let collect = |stmt: &mut rusqlite::Statement, p: &[&dyn ToSql]| -> Result<Vec<Map<String, Value>>> {
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let rows = stmt.query_map(p, |r| {
            let mut m = Map::new();
            for (i, name) in names.iter().enumerate() {
                let v: Value = match r.get_ref(i)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(n) => Value::from(n),
                    rusqlite::types::ValueRef::Real(f) => Value::from(f),
                    rusqlite::types::ValueRef::Text(t) => {
                        Value::String(String::from_utf8_lossy(t).into_owned())
                    }
                    rusqlite::types::ValueRef::Blob(_) => Value::Null,
                };
                m.insert(name.clone(), v);
            }
            Ok(m)
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    };
    match task_id {
        Some(tid) => {
            let mut stmt = conn.prepare("SELECT * FROM kanban_notify_subs WHERE task_id = ?")?;
            collect(&mut stmt, &[&tid])
        }
        None => {
            let mut stmt = conn.prepare("SELECT * FROM kanban_notify_subs")?;
            collect(&mut stmt, &[])
        }
    }
}

pub fn remove_notify_sub(
    conn: &Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
) -> Result<bool> {
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id = ? AND platform = ? \
             AND chat_id = ? AND thread_id = ?",
            params![task_id, platform, chat_id, thread_id.unwrap_or("")],
        )?;
        Ok(affected > 0)
    })
}

/// Return `(new_cursor, events)` for a subscription (events with id > cursor).
pub fn unseen_events_for_sub(
    conn: &Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    kinds: Option<&[String]>,
) -> Result<(i64, Vec<Event>)> {
    let cursor: Option<i64> = conn
        .query_row(
            "SELECT last_event_id FROM kanban_notify_subs WHERE task_id = ? AND platform = ? \
             AND chat_id = ? AND thread_id = ?",
            params![task_id, platform, chat_id, thread_id.unwrap_or("")],
            |r| r.get(0),
        )
        .optional()?;
    let cursor = match cursor {
        None => return Ok((0, vec![])),
        Some(c) => c,
    };
    let kind_list: Option<Vec<String>> = kinds
        .filter(|k| !k.is_empty())
        .map(|k| k.to_vec());
    let mut q = String::from("SELECT * FROM task_events WHERE task_id = ? AND id > ? ");
    if let Some(kl) = &kind_list {
        q.push_str(&format!("AND kind IN ({}) ", vec!["?"; kl.len()].join(",")));
    }
    q.push_str("ORDER BY id ASC");

    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(task_id.to_string()), Box::new(cursor)];
    if let Some(kl) = &kind_list {
        for k in kl {
            params.push(Box::new(k.clone()));
        }
    }
    let mut stmt = conn.prepare(&q)?;
    let events: Vec<Event> = stmt
        .query_map(params_from_iter(params.iter().map(|b| b.as_ref())), |r| {
            Ok(event_from_row(r))
        })?
        .filter_map(|r| r.ok())
        .collect();
    let mut max_id = cursor;
    for e in &events {
        if e.id > max_id {
            max_id = e.id;
        }
    }
    Ok((max_id, events))
}

pub fn advance_notify_cursor(
    conn: &Connection,
    task_id: &str,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    new_cursor: i64,
) -> Result<()> {
    write_txn(conn, |conn| {
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? WHERE task_id = ? AND platform = ? \
             AND chat_id = ? AND thread_id = ?",
            params![new_cursor, task_id, platform, chat_id, thread_id.unwrap_or("")],
        )?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Retention + GC
// ---------------------------------------------------------------------------

/// Delete `task_events` older than `older_than_seconds` for terminal tasks.
pub fn gc_events(conn: &Connection, older_than_seconds: i64) -> Result<usize> {
    let cutoff = now_secs() - older_than_seconds;
    write_txn(conn, |conn| {
        let affected = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN \
             (SELECT id FROM tasks WHERE status IN ('done', 'archived'))",
            params![cutoff],
        )?;
        Ok(affected)
    })
}

/// Delete worker log files older than `older_than_seconds`. Returns count.
pub fn gc_worker_logs(older_than_seconds: i64, board: Option<&str>) -> usize {
    let log_dir = worker_logs_dir(board);
    if !log_dir.exists() {
        return 0;
    }
    let cutoff = now_secs() - older_than_seconds;
    let mut removed = 0usize;
    if let Ok(rd) = fs::read_dir(&log_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if let Ok(meta) = p.metadata() {
                if meta.is_file() {
                    if let Ok(modified) = meta.modified() {
                        let mtime = modified
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(i64::MAX);
                        if mtime < cutoff && fs::remove_file(&p).is_ok() {
                            removed += 1;
                        }
                    }
                }
            }
        }
    }
    removed
}

// ---------------------------------------------------------------------------
// Worker log accessor
// ---------------------------------------------------------------------------

pub fn worker_log_path(task_id: &str, board: Option<&str>) -> PathBuf {
    worker_logs_dir(board).join(format!("{task_id}.log"))
}

/// Read the worker log for `task_id`. Returns None if missing. With
/// `tail_bytes`, only the last N bytes are returned (skipping a partial
/// leading line where possible).
pub fn read_worker_log(
    task_id: &str,
    tail_bytes: Option<u64>,
    board: Option<&str>,
) -> Option<String> {
    let path = worker_log_path(task_id, board);
    if !path.exists() {
        return None;
    }
    match tail_bytes {
        None => fs::read(&path)
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned()),
        Some(tail) => {
            let mut f = fs::File::open(&path).ok()?;
            let size = f.metadata().ok()?.len();
            if size > tail {
                f.seek(SeekFrom::Start(size - tail)).ok()?;
                // Skip a partial line if we tailed mid-line, unless the whole
                // window is one giant line.
                let probe = f.stream_position().ok()?;
                let mut partial = Vec::new();
                read_one_line(&mut f, &mut partial).ok()?;
                let pos = f.stream_position().ok()?;
                if !partial.ends_with(b"\n") && pos >= size {
                    f.seek(SeekFrom::Start(probe)).ok()?;
                }
            }
            let mut data = Vec::new();
            f.read_to_end(&mut data).ok()?;
            Some(String::from_utf8_lossy(&data).into_owned())
        }
    }
}

fn read_one_line(f: &mut fs::File, buf: &mut Vec<u8>) -> std::io::Result<()> {
    let mut byte = [0u8; 1];
    loop {
        let n = f.read(&mut byte)?;
        if n == 0 {
            break;
        }
        buf.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Assignee enumeration
// ---------------------------------------------------------------------------

/// Return the set of assignee/profile names discovered on disk.
pub fn list_profiles_on_disk() -> Vec<String> {
    let default_root = get_default_hermes_root();
    let profiles_dir = default_root.join("profiles");

    let mut names: HashSet<String> = HashSet::new();
    if default_root.exists() {
        names.insert("default".to_string());
    }
    if profiles_dir.is_dir() {
        if let Ok(rd) = fs::read_dir(&profiles_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() && p.join("config.yaml").is_file() {
                    if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                        names.insert(name.to_string());
                    }
                }
            }
        }
    }
    let mut sorted: Vec<String> = names.into_iter().collect();
    sorted.sort();
    sorted
}

/// Return every assignee name known to the board or on disk.
pub fn known_assignees(conn: &Connection) -> Result<Vec<Value>> {
    let on_disk: HashSet<String> = list_profiles_on_disk().into_iter().collect();

    let mut counts: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT assignee, status, COUNT(*) AS n FROM tasks \
             WHERE status != 'archived' AND assignee IS NOT NULL GROUP BY assignee, status",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;
        for r in rows.flatten() {
            counts.entry(r.0).or_default().insert(r.1, r.2);
        }
    }

    let mut all_names: HashSet<String> = on_disk.clone();
    for k in counts.keys() {
        all_names.insert(k.clone());
    }
    let mut names: Vec<String> = all_names.into_iter().collect();
    names.sort();

    Ok(names
        .into_iter()
        .map(|name| {
            let c = counts.get(&name).cloned().unwrap_or_default();
            json!({
                "name": name,
                "on_disk": on_disk.contains(&name),
                "counts": c,
            })
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Runs (attempt history)
// ---------------------------------------------------------------------------

/// Return all runs for `task_id` in start order.
pub fn list_runs(conn: &Connection, task_id: &str, include_active: bool) -> Result<Vec<Run>> {
    let mut q = String::from("SELECT * FROM task_runs WHERE task_id = ?");
    if !include_active {
        q.push_str(" AND ended_at IS NOT NULL");
    }
    q.push_str(" ORDER BY started_at ASC, id ASC");
    let mut stmt = conn.prepare(&q)?;
    let rows: Vec<Run> = stmt
        .query_map(params![task_id], |r| Ok(Run::from_row(r)))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

pub fn get_run(conn: &Connection, run_id: i64) -> Result<Option<Run>> {
    let run = conn
        .query_row("SELECT * FROM task_runs WHERE id = ?", params![run_id], |r| {
            Ok(Run::from_row(r))
        })
        .optional()?;
    Ok(run)
}

/// Return the currently-open run for `task_id` (`ended_at IS NULL`).
pub fn active_run(conn: &Connection, task_id: &str) -> Result<Option<Run>> {
    let run = conn
        .query_row(
            "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NULL \
             ORDER BY started_at DESC LIMIT 1",
            params![task_id],
            |r| Ok(Run::from_row(r)),
        )
        .optional()?;
    Ok(run)
}

/// Return the most recent run regardless of outcome.
pub fn latest_run(conn: &Connection, task_id: &str) -> Result<Option<Run>> {
    let run = conn
        .query_row(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY started_at DESC, id DESC LIMIT 1",
            params![task_id],
            |r| Ok(Run::from_row(r)),
        )
        .optional()?;
    Ok(run)
}

/// Return the latest non-null `task_runs.summary` for `task_id`.
pub fn latest_summary(conn: &Connection, task_id: &str) -> Result<Option<String>> {
    let s: Option<Option<String>> = conn
        .query_row(
            "SELECT summary FROM task_runs WHERE task_id = ? AND summary IS NOT NULL \
             AND summary != '' ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
            params![task_id],
            |r| r.get(0),
        )
        .optional()?;
    Ok(s.flatten())
}

/// Batch-fetch latest non-null summaries for a list of task ids.
pub fn latest_summaries(
    conn: &Connection,
    task_ids: &[String],
) -> Result<BTreeMap<String, String>> {
    if task_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let placeholders = vec!["?"; task_ids.len()].join(",");
    let sql = format!(
        "SELECT task_id, summary FROM ( \
            SELECT task_id, summary, ROW_NUMBER() OVER ( \
                PARTITION BY task_id ORDER BY COALESCE(ended_at, started_at) DESC, id DESC \
            ) AS rn FROM task_runs WHERE task_id IN ({placeholders}) \
            AND summary IS NOT NULL AND summary != '' \
        ) WHERE rn = 1"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows: BTreeMap<String, String> = stmt
        .query_map(params_from_iter(task_ids.iter()), |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SCHEMA_SQL).unwrap();
        conn
    }

    #[test]
    fn test_normalize_board_slug() {
        assert_eq!(normalize_board_slug(None).unwrap(), None);
        assert_eq!(normalize_board_slug(Some("  ")).unwrap(), None);
        assert_eq!(
            normalize_board_slug(Some("Atm10-Server")).unwrap(),
            Some("atm10-server".into())
        );
        assert!(normalize_board_slug(Some("../etc")).is_err());
        assert!(normalize_board_slug(Some("-leading")).is_err());
    }

    #[test]
    fn test_default_board_display_name() {
        assert_eq!(default_board_display_name("atm10-server"), "Atm10 Server");
        assert_eq!(default_board_display_name("hermes_agent"), "Hermes Agent");
    }

    #[test]
    fn test_create_and_get_task() {
        let conn = mem_conn();
        let id = create_task(
            &conn,
            CreateTask {
                title: "  hello  ",
                assignee: Some("Alice"),
                ..CreateTask::new("hello")
            },
        )
        .unwrap();
        assert!(id.starts_with("t_"));
        let task = get_task(&conn, &id).unwrap().unwrap();
        assert_eq!(task.title, "hello");
        assert_eq!(task.assignee.as_deref(), Some("alice"));
        assert_eq!(task.status, "ready");
    }

    #[test]
    fn test_create_task_requires_title() {
        let conn = mem_conn();
        let err = create_task(&conn, CreateTask::new("   ")).unwrap_err();
        assert!(matches!(err, KanbanError::Value(_)));
    }

    #[test]
    fn test_idempotency_key() {
        let conn = mem_conn();
        let id1 = create_task(
            &conn,
            CreateTask {
                idempotency_key: Some("dup"),
                ..CreateTask::new("t1")
            },
        )
        .unwrap();
        let id2 = create_task(
            &conn,
            CreateTask {
                idempotency_key: Some("dup"),
                ..CreateTask::new("t2")
            },
        )
        .unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_parent_child_status() {
        let conn = mem_conn();
        let parent = create_task(&conn, CreateTask::new("parent")).unwrap();
        let child = create_task(
            &conn,
            CreateTask {
                parents: vec![parent.clone()],
                ..CreateTask::new("child")
            },
        )
        .unwrap();
        // Parent not done -> child is todo.
        assert_eq!(get_task(&conn, &child).unwrap().unwrap().status, "todo");
        // Complete parent then recompute.
        claim_task(&conn, &parent, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        complete_task(&conn, &parent, CompleteOpts::default()).unwrap();
        assert_eq!(get_task(&conn, &child).unwrap().unwrap().status, "ready");
    }

    #[test]
    fn test_claim_is_atomic() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        let first = claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        assert!(first.is_some());
        let second = claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:2")).unwrap();
        assert!(second.is_none());
        let task = get_task(&conn, &id).unwrap().unwrap();
        assert_eq!(task.status, "running");
        assert!(task.current_run_id.is_some());
    }

    #[test]
    fn test_complete_creates_run_and_event() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        let ok = complete_task(
            &conn,
            &id,
            CompleteOpts {
                summary: Some("did the thing"),
                ..CompleteOpts::default()
            },
        )
        .unwrap();
        assert!(ok);
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "done");
        assert_eq!(latest_summary(&conn, &id).unwrap().as_deref(), Some("did the thing"));
        let events = list_events(&conn, &id).unwrap();
        assert!(events.iter().any(|e| e.kind == "completed"));
    }

    #[test]
    fn test_block_and_unblock() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        assert!(block_task(&conn, &id, Some("waiting"), None).unwrap());
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "blocked");
        assert!(unblock_task(&conn, &id).unwrap());
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "ready");
    }

    #[test]
    fn test_release_stale_claims() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        // Claim with a TTL in the past.
        claim_task(&conn, &id, -100, Some("h:1")).unwrap();
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "running");
        let n = release_stale_claims(&conn).unwrap();
        assert_eq!(n, 1);
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "ready");
    }

    #[test]
    fn test_cycle_detection() {
        let conn = mem_conn();
        let a = create_task(&conn, CreateTask::new("a")).unwrap();
        let b = create_task(&conn, CreateTask::new("b")).unwrap();
        link_tasks(&conn, &a, &b).unwrap();
        let err = link_tasks(&conn, &b, &a).unwrap_err();
        assert!(matches!(err, KanbanError::Value(_)));
    }

    #[test]
    fn test_hallucinated_cards() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        let err = complete_task(
            &conn,
            &id,
            CompleteOpts {
                created_cards: Some(vec!["t_deadbeef".into()]),
                ..CompleteOpts::default()
            },
        )
        .unwrap_err();
        match err {
            KanbanError::HallucinatedCards { phantom, .. } => {
                assert_eq!(phantom, vec!["t_deadbeef".to_string()]);
            }
            other => panic!("expected HallucinatedCards, got {other:?}"),
        }
        // Task still running (not mutated).
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "running");
    }

    #[test]
    fn test_record_task_failure_trips_breaker() {
        let conn = mem_conn();
        let id = create_task(
            &conn,
            CreateTask {
                assignee: Some("a"),
                ..CreateTask::new("t")
            },
        )
        .unwrap();
        let mut blocked = false;
        for _ in 0..DEFAULT_FAILURE_LIMIT {
            claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
            blocked = record_spawn_failure(&conn, &id, "boom", None).unwrap();
        }
        assert!(blocked);
        assert_eq!(get_task(&conn, &id).unwrap().unwrap().status, "blocked");
    }

    #[test]
    fn test_comments_and_context() {
        let conn = mem_conn();
        let id = create_task(
            &conn,
            CreateTask {
                body: Some("opening post"),
                ..CreateTask::new("Build it")
            },
        )
        .unwrap();
        add_comment(&conn, &id, "alice", "first comment").unwrap();
        let ctx = build_worker_context(&conn, &id).unwrap();
        assert!(ctx.contains("# Kanban task"));
        assert!(ctx.contains("## Body"));
        assert!(ctx.contains("opening post"));
        assert!(ctx.contains("## Comment thread"));
        assert!(ctx.contains("first comment"));
        assert!(ctx.ends_with('\n'));
    }

    #[test]
    fn test_board_stats() {
        let conn = mem_conn();
        create_task(&conn, CreateTask::new("a")).unwrap();
        create_task(&conn, CreateTask::new("b")).unwrap();
        let stats = board_stats(&conn).unwrap();
        assert_eq!(stats["by_status"]["ready"], 2);
    }

    #[test]
    fn test_notify_subs_flow() {
        let conn = mem_conn();
        let id = create_task(&conn, CreateTask::new("t")).unwrap();
        add_notify_sub(&conn, &id, "telegram", "chat1", None, Some("u1")).unwrap();
        claim_task(&conn, &id, DEFAULT_CLAIM_TTL_SECONDS, Some("h:1")).unwrap();
        complete_task(&conn, &id, CompleteOpts::default()).unwrap();
        let (cursor, events) =
            unseen_events_for_sub(&conn, &id, "telegram", "chat1", None, Some(&["completed".into()]))
                .unwrap();
        assert!(events.iter().any(|e| e.kind == "completed"));
        assert!(cursor > 0);
        advance_notify_cursor(&conn, &id, "telegram", "chat1", None, cursor).unwrap();
        let (_c2, events2) =
            unseen_events_for_sub(&conn, &id, "telegram", "chat1", None, None).unwrap();
        assert!(events2.is_empty());
    }

    #[test]
    fn test_dispatch_dry_run() {
        let conn = mem_conn();
        create_task(
            &conn,
            CreateTask {
                assignee: Some("alice"),
                ..CreateTask::new("t")
            },
        )
        .unwrap();
        create_task(&conn, CreateTask::new("unassigned")).unwrap();
        let res = dispatch_once(
            &conn,
            DispatchOpts {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res.spawned.len(), 1);
        assert_eq!(res.skipped_unassigned.len(), 1);
    }

    #[test]
    fn test_dispatch_custom_spawn_fn() {
        let conn = mem_conn();
        create_task(
            &conn,
            CreateTask {
                assignee: Some("alice"),
                ..CreateTask::new("t")
            },
        )
        .unwrap();
        let spawn = |_t: &Task, _ws: &str, _b: Option<&str>| -> Result<Option<i64>> { Ok(None) };
        let res = dispatch_once(
            &conn,
            DispatchOpts {
                spawn_fn: Some(&spawn),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res.spawned.len(), 1);
    }

    #[test]
    fn test_skills_roundtrip() {
        let conn = mem_conn();
        let id = create_task(
            &conn,
            CreateTask {
                skills: Some(vec!["translation".into(), "translation".into(), "  ".into()]),
                ..CreateTask::new("t")
            },
        )
        .unwrap();
        let task = get_task(&conn, &id).unwrap().unwrap();
        assert_eq!(task.skills, Some(vec!["translation".to_string()]));
    }

    #[test]
    fn test_skills_comma_rejected() {
        let conn = mem_conn();
        let err = create_task(
            &conn,
            CreateTask {
                skills: Some(vec!["a,b".into()]),
                ..CreateTask::new("t")
            },
        )
        .unwrap_err();
        assert!(matches!(err, KanbanError::Value(_)));
    }

    #[test]
    fn test_migrate_legacy_columns() {
        let conn = Connection::open_in_memory().unwrap();
        // Minimal legacy tasks table with the old column names.
        conn.execute_batch(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, \
             assignee TEXT, status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT, \
             created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER, \
             workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT, \
             claim_lock TEXT, claim_expires INTEGER, spawn_failures INTEGER NOT NULL DEFAULT 3, \
             last_spawn_error TEXT); \
             CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, \
             kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL); \
             CREATE TABLE task_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, \
             profile TEXT, status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER, \
             worker_pid INTEGER, max_runtime_seconds INTEGER, last_heartbeat_at INTEGER, \
             started_at INTEGER NOT NULL, ended_at INTEGER, outcome TEXT, summary TEXT, \
             metadata TEXT, error TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind, spawn_failures) \
             VALUES ('t_x', 'legacy', 'ready', 1, 'scratch', 7)",
            [],
        )
        .unwrap();
        migrate_add_optional_columns(&conn).unwrap();
        let task = get_task(&conn, "t_x").unwrap().unwrap();
        assert_eq!(task.consecutive_failures, 7);
        let cols = table_columns(&conn, "tasks").unwrap();
        assert!(cols.contains("consecutive_failures"));
        assert!(cols.contains("skills"));
    }

    #[test]
    fn test_event_kind_renames() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SCHEMA_SQL).unwrap();
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) VALUES ('t', 'ready', 1)",
            [],
        )
        .unwrap();
        migrate_add_optional_columns(&conn).unwrap();
        let kind: String = conn
            .query_row("SELECT kind FROM task_events LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(kind, "promoted");
    }

    #[test]
    fn test_cap_truncation() {
        let long = "a".repeat(50);
        let out = cap(Some(&long), 10);
        assert!(out.starts_with("aaaaaaaaaa…"));
        assert!(out.contains("40 chars omitted"));
    }

    #[test]
    fn test_sorted_json() {
        let v = json!({ "b": 1, "a": { "z": 2, "y": 3 } });
        let s = serde_json::to_string(&sorted_json(&v)).unwrap();
        assert_eq!(s, r#"{"a":{"y":3,"z":2},"b":1}"#);
    }
}
