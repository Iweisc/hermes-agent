//! Hermes MCP Server — expose messaging conversations as MCP tools.
//!
//! Native Rust port of `mcp_serve.py`.
//!
//! The Python original starts a stdio MCP server (via the `mcp` FastMCP SDK)
//! that lets any MCP client list conversations, read message history, send
//! messages, poll for live events, and manage approval requests across all
//! connected platforms.
//!
//! There is no first-class Rust FastMCP SDK available in this crate set, so the
//! transport/registration layer (`FastMCP`, `run_stdio_async`) is *not*
//! reproduced here. Instead this module ports the load-bearing logic 1:1:
//!
//! * the filesystem helpers (`_get_sessions_dir`, `_load_sessions_index`,
//!   `_load_channel_directory`),
//! * content/attachment extraction (`_extract_message_content`,
//!   `_extract_attachments`),
//! * the [`EventBridge`] background poller with its in-memory event queue and
//!   waiter support, and
//! * the ten MCP tool handlers (`conversations_list`, `conversation_get`,
//!   `messages_read`, `attachments_fetch`, `events_poll`, `events_wait`,
//!   `messages_send`, `channels_list`, `permissions_list_open`,
//!   `permissions_respond`) as plain functions that return the exact same JSON
//!   strings the Python tools returned.
//!
//! A caller wiring up an actual MCP server registers each function as a tool;
//! the surface (names, args, JSON shapes) matches OpenClaw's 9-tool channel
//! bridge plus the Hermes-specific `channels_list`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::mod_hermes_constants::get_hermes_home;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of events retained in the in-memory queue.
pub const QUEUE_LIMIT: usize = 1000;
/// Seconds between DB polls (200 ms), matching `POLL_INTERVAL` in Python.
pub const POLL_INTERVAL_SECS: f64 = 0.2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the sessions directory using HERMES_HOME.
pub fn get_sessions_dir() -> PathBuf {
    get_hermes_home().join("sessions")
}

/// Load the gateway `sessions.json` index directly.
///
/// Returns a map of `session_key -> entry` (each entry an arbitrary JSON
/// object) with platform routing info. Mirrors `_load_sessions_index`: a
/// missing file or any read/parse error yields an empty map.
pub fn load_sessions_index() -> Map<String, Value> {
    let sessions_file = get_sessions_dir().join("sessions.json");
    if !sessions_file.exists() {
        return Map::new();
    }
    match std::fs::read_to_string(&sessions_file) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            Ok(_) => Map::new(),
            Err(e) => {
                log::debug!("Failed to load sessions.json: {e}");
                Map::new()
            }
        },
        Err(e) => {
            log::debug!("Failed to load sessions.json: {e}");
            Map::new()
        }
    }
}

/// Load the cached channel directory for available targets.
///
/// Mirrors `_load_channel_directory`: missing file or any error yields an empty
/// map.
pub fn load_channel_directory() -> Map<String, Value> {
    let directory_file = get_hermes_home().join("channel_directory.json");
    if !directory_file.exists() {
        return Map::new();
    }
    match std::fs::read_to_string(&directory_file) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            Ok(_) => Map::new(),
            Err(e) => {
                log::debug!("Failed to load channel_directory.json: {e}");
                Map::new()
            }
        },
        Err(e) => {
            log::debug!("Failed to load channel_directory.json: {e}");
            Map::new()
        }
    }
}

/// Best-effort string lookup on a JSON object, returning "" if absent or
/// non-string. Mirrors Python's `d.get(key, "")` where the result is treated as
/// a string.
fn get_str(obj: &Value, key: &str) -> String {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_default()
}

/// Like [`get_str`] but for an object value, returning `&Value::Null` when
/// missing so chained lookups stay safe (mirrors `d.get(k, {})`).
fn get_obj<'a>(obj: &'a Value, key: &str) -> &'a Value {
    obj.get(key).unwrap_or(&Value::Null)
}

/// Extract text content from a message, handling multi-part content.
///
/// Mirrors `_extract_message_content`. If `content` is a list, joins the
/// `text` fields of `{"type": "text"}` blocks with newlines. Otherwise returns
/// the string form (empty for null/missing).
pub fn extract_message_content(msg: &Value) -> String {
    let content = msg.get("content").unwrap_or(&Value::Null);
    if let Value::Array(parts) = content {
        let text_parts: Vec<String> = parts
            .iter()
            .filter_map(|p| {
                if let Value::Object(map) = p {
                    if map.get("type").and_then(Value::as_str) == Some("text") {
                        return Some(
                            map.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                }
                None
            })
            .collect();
        return text_parts.join("\n");
    }
    // `str(content) if content else ""`
    match content {
        Value::Null => String::new(),
        Value::String(s) => {
            if s.is_empty() {
                String::new()
            } else {
                s.clone()
            }
        }
        Value::Bool(false) => String::new(),
        Value::Number(n) if n.as_f64() == Some(0.0) => String::new(),
        other => other.to_string(),
    }
}

/// Truncate a `&str` to at most `max` characters (not bytes), mirroring
/// Python's `s[:max]` semantics on Unicode strings.
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Extract non-text attachments from a message.
///
/// Mirrors `_extract_attachments`: multi-part image/file content blocks,
/// `MEDIA:` tags in text, image URLs, and file references.
pub fn extract_attachments(msg: &Value) -> Vec<Value> {
    let mut attachments: Vec<Value> = Vec::new();
    let content = msg.get("content").unwrap_or(&Value::Null);

    if let Value::Array(parts) = content {
        for part in parts {
            let map = match part {
                Value::Object(m) => m,
                _ => continue,
            };
            let ptype = map.get("type").and_then(Value::as_str).unwrap_or("");
            if ptype == "image_url" {
                // url = part["image_url"]["url"] if image_url is a dict else ""
                let url = match map.get("image_url") {
                    Some(Value::Object(iu)) => {
                        iu.get("url").and_then(Value::as_str).unwrap_or("").to_string()
                    }
                    _ => String::new(),
                };
                if !url.is_empty() {
                    attachments.push(json!({"type": "image", "url": url}));
                }
            } else if ptype == "image" {
                // url = part.get("url", part.get("source", {}).get("url", ""))
                let url = match map.get("url").and_then(Value::as_str) {
                    Some(u) => u.to_string(),
                    None => match map.get("source") {
                        Some(Value::Object(src)) => {
                            src.get("url").and_then(Value::as_str).unwrap_or("").to_string()
                        }
                        _ => String::new(),
                    },
                };
                if !url.is_empty() {
                    attachments.push(json!({"type": "image", "url": url}));
                }
            } else if ptype != "text" {
                // Unknown non-text content type
                attachments.push(json!({"type": ptype, "data": part}));
            }
        }
    }

    // MEDIA: tags in text content
    let text = extract_message_content(msg);
    if !text.is_empty() {
        // re.compile(r'MEDIA:\s*(\S+)')
        let re = regex::Regex::new(r"MEDIA:\s*(\S+)").expect("static regex");
        for caps in re.captures_iter(&text) {
            let path = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            attachments.push(json!({"type": "media", "path": path}));
        }
    }

    attachments
}

// ---------------------------------------------------------------------------
// Message source abstraction
// ---------------------------------------------------------------------------

/// Abstraction over the message store used by the bridge and tool handlers.
///
/// The Python original calls `SessionDB().get_messages(session_id)` returning a
/// list of message dicts. We take a trait so callers can wire in the native
/// [`crate::state::SessionStore`] (or a test double) without this module
/// depending on the exact open/connect lifecycle.
pub trait MessageSource: Send + Sync {
    /// Return the messages for a session as JSON objects, newest-last, each
    /// containing at least `id`, `role`, `content`, `timestamp`. An error
    /// causes callers to skip the session (mirrors the bare `except` guards in
    /// Python).
    fn get_messages(&self, session_id: &str) -> Result<Vec<Value>, String>;
}

/// Convert a [`crate::state::MessageRecord`] into the loosely-typed JSON dict
/// shape the Python code worked with.
pub fn message_record_to_value(rec: &crate::state::MessageRecord) -> Value {
    json!({
        "id": rec.id,
        "session_id": rec.session_id,
        "role": rec.role,
        "content": rec.content.clone().unwrap_or(Value::Null),
        "timestamp": rec.timestamp,
        "tool_name": rec.tool_name.clone(),
    })
}

/// Adapter exposing a [`crate::state::SessionStore`] as a [`MessageSource`].
pub struct SessionStoreSource {
    store: crate::state::SessionStore,
}

impl SessionStoreSource {
    pub fn new(store: crate::state::SessionStore) -> Self {
        Self { store }
    }
}

impl MessageSource for SessionStoreSource {
    fn get_messages(&self, session_id: &str) -> Result<Vec<Value>, String> {
        self.store
            .get_messages(session_id)
            .map(|recs| recs.iter().map(message_record_to_value).collect())
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// EventBridge — polls the message store for new messages, maintains queue
// ---------------------------------------------------------------------------

/// An event in the bridge's in-memory queue.
///
/// Mirrors the Python `QueueEvent` dataclass. `data` holds the event-type
/// specific payload that gets flattened into the serialized event.
#[derive(Debug, Clone)]
pub struct QueueEvent {
    pub cursor: i64,
    /// "message", "approval_requested", or "approval_resolved".
    pub event_type: String,
    pub session_key: String,
    pub data: Map<String, Value>,
}

impl QueueEvent {
    pub fn new(event_type: impl Into<String>, session_key: impl Into<String>, data: Map<String, Value>) -> Self {
        Self {
            cursor: 0,
            event_type: event_type.into(),
            session_key: session_key.into(),
            data,
        }
    }

    /// Serialize as `{"cursor", "type", "session_key", ...data}` — matching the
    /// dict-unpacking the Python tools did when emitting events.
    fn to_value(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("cursor".into(), json!(self.cursor));
        obj.insert("type".into(), json!(self.event_type));
        obj.insert("session_key".into(), json!(self.session_key));
        for (k, v) in &self.data {
            obj.insert(k.clone(), v.clone());
        }
        Value::Object(obj)
    }
}

/// Mutable interior state of the [`EventBridge`], guarded by a single mutex.
struct BridgeInner {
    queue: Vec<QueueEvent>,
    cursor: i64,
    last_poll_timestamps: HashMap<String, f64>,
    pending_approvals: HashMap<String, Value>,
    sessions_json_mtime: f64,
    state_db_mtime: f64,
    cached_sessions_index: Map<String, Value>,
}

impl BridgeInner {
    fn new() -> Self {
        Self {
            queue: Vec::new(),
            cursor: 0,
            last_poll_timestamps: HashMap::new(),
            pending_approvals: HashMap::new(),
            sessions_json_mtime: 0.0,
            state_db_mtime: 0.0,
            cached_sessions_index: Map::new(),
        }
    }
}

/// Background poller that watches the message store for new messages and
/// maintains an in-memory event queue with waiter support.
///
/// Hermes equivalent of OpenClaw's WebSocket gateway bridge: instead of
/// WebSocket events, it polls for changes (here via mtime checks plus the
/// injected [`MessageSource`]).
pub struct EventBridge {
    inner: Mutex<BridgeInner>,
    /// Wakes any waiters in [`EventBridge::wait_for_event`] when a new event is
    /// enqueued (analogue of the Python `threading.Event`).
    new_event: Condvar,
    running: Mutex<bool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Default for EventBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBridge {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BridgeInner::new()),
            new_event: Condvar::new(),
            running: Mutex::new(false),
            thread: Mutex::new(None),
        }
    }

    /// Start the background polling thread, using the given message source.
    ///
    /// Mirrors `EventBridge.start` + `_poll_loop`: if the source is `None` the
    /// loop logs a warning and exits, leaving event polling disabled.
    pub fn start(self: &Arc<Self>, source: Option<Arc<dyn MessageSource>>) {
        {
            let mut running = self.running.lock().unwrap();
            if *running {
                return;
            }
            *running = true;
        }

        let bridge = Arc::clone(self);
        let handle = std::thread::spawn(move || {
            bridge.poll_loop(source);
        });
        *self.thread.lock().unwrap() = Some(handle);
        log::debug!("EventBridge started");
    }

    /// Stop the background polling thread.
    pub fn stop(&self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = false;
        }
        // Wake any waiters.
        self.new_event.notify_all();
        if let Some(handle) = self.thread.lock().unwrap().take() {
            let _ = handle.join();
        }
        log::debug!("EventBridge stopped");
    }

    fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Return events since `after_cursor`, optionally filtered by `session_key`.
    pub fn poll_events(
        &self,
        after_cursor: i64,
        session_key: Option<&str>,
        limit: usize,
    ) -> Value {
        let inner = self.inner.lock().unwrap();
        let events: Vec<&QueueEvent> = inner
            .queue
            .iter()
            .filter(|e| {
                e.cursor > after_cursor
                    && session_key.map_or(true, |sk| e.session_key == sk)
            })
            .take(limit)
            .collect();

        let next_cursor = events.last().map(|e| e.cursor).unwrap_or(after_cursor);
        let serialized: Vec<Value> = events.iter().map(|e| e.to_value()).collect();
        json!({
            "events": serialized,
            "next_cursor": next_cursor,
        })
    }

    /// Block until a matching event arrives or `timeout_ms` expires.
    pub fn wait_for_event(
        &self,
        after_cursor: i64,
        session_key: Option<&str>,
        timeout_ms: u64,
    ) -> Option<Value> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);

        loop {
            if Instant::now() >= deadline {
                break;
            }
            {
                let inner = self.inner.lock().unwrap();
                for e in &inner.queue {
                    if e.cursor > after_cursor
                        && session_key.map_or(true, |sk| e.session_key == sk)
                    {
                        return Some(e.to_value());
                    }
                }
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let poll = Duration::from_secs_f64(POLL_INTERVAL_SECS);
            let wait = remaining.min(poll);
            let guard = self.inner.lock().unwrap();
            let _ = self.new_event.wait_timeout(guard, wait).unwrap();
        }

        None
    }

    /// List approval requests observed during this bridge session, sorted by
    /// `created_at`.
    pub fn list_pending_approvals(&self) -> Vec<Value> {
        let inner = self.inner.lock().unwrap();
        let mut approvals: Vec<Value> = inner.pending_approvals.values().cloned().collect();
        approvals.sort_by(|a, b| {
            let ka = a.get("created_at").and_then(Value::as_str).unwrap_or("");
            let kb = b.get("created_at").and_then(Value::as_str).unwrap_or("");
            ka.cmp(kb)
        });
        approvals
    }

    /// Resolve a pending approval (best-effort without gateway IPC).
    pub fn respond_to_approval(&self, approval_id: &str, decision: &str) -> Value {
        let approval = {
            let mut inner = self.inner.lock().unwrap();
            inner.pending_approvals.remove(approval_id)
        };

        let approval = match approval {
            Some(a) => a,
            None => return json!({"error": format!("Approval not found: {approval_id}")}),
        };

        let session_key = get_str(&approval, "session_key");
        let mut data = Map::new();
        data.insert("approval_id".into(), json!(approval_id));
        data.insert("decision".into(), json!(decision));
        self.enqueue(QueueEvent::new("approval_resolved", session_key, data));

        json!({"resolved": true, "approval_id": approval_id, "decision": decision})
    }

    /// Record an approval request so [`list_pending_approvals`] can surface it.
    ///
    /// Not present as a standalone method in Python (approvals were populated
    /// from observed events), but exposed here so a gateway integration can
    /// feed approvals in. `approval` should contain at least `id`,
    /// `session_key`, and `created_at`.
    pub fn add_pending_approval(&self, approval_id: impl Into<String>, approval: Value) {
        let id = approval_id.into();
        let session_key = get_str(&approval, "session_key");
        {
            let mut inner = self.inner.lock().unwrap();
            inner.pending_approvals.insert(id.clone(), approval.clone());
        }
        let mut data = Map::new();
        data.insert("approval".into(), approval);
        self.enqueue(QueueEvent::new("approval_requested", session_key, data));
        // Re-store after enqueue clears nothing; keep id referenced.
        let _ = id;
    }

    /// Add an event to the queue and wake any waiters.
    fn enqueue(&self, mut event: QueueEvent) {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.cursor += 1;
            event.cursor = inner.cursor;
            inner.queue.push(event);
            while inner.queue.len() > QUEUE_LIMIT {
                inner.queue.remove(0);
            }
        }
        self.new_event.notify_all();
    }

    /// Background loop: poll the message source for new messages.
    fn poll_loop(self: Arc<Self>, source: Option<Arc<dyn MessageSource>>) {
        let source = match source {
            Some(s) => s,
            None => {
                log::warn!("EventBridge: SessionDB unavailable, event polling disabled");
                return;
            }
        };

        while self.is_running() {
            if let Err(e) = self.poll_once(source.as_ref()) {
                log::debug!("EventBridge poll error: {e}");
            }
            std::thread::sleep(Duration::from_secs_f64(POLL_INTERVAL_SECS));
        }
    }

    /// Check for new messages across all sessions.
    ///
    /// Uses mtime checks on `sessions.json` and `state.db` to skip work when
    /// nothing has changed. Mirrors `_poll_once`.
    fn poll_once(&self, source: &dyn MessageSource) -> Result<(), String> {
        let sessions_file = get_sessions_dir().join("sessions.json");
        let sj_mtime = file_mtime(&sessions_file);

        {
            let mut inner = self.inner.lock().unwrap();
            if sj_mtime != inner.sessions_json_mtime {
                inner.sessions_json_mtime = sj_mtime;
                drop(inner);
                let idx = load_sessions_index();
                self.inner.lock().unwrap().cached_sessions_index = idx;
            }
        }

        let db_file = get_hermes_home().join("state.db");
        let db_mtime = file_mtime(&db_file);

        // Snapshot state under the lock and decide whether to bail.
        let entries = {
            let mut inner = self.inner.lock().unwrap();
            if db_mtime == inner.state_db_mtime && sj_mtime == inner.sessions_json_mtime {
                // Nothing changed since last poll — skip entirely.
                return Ok(());
            }
            inner.state_db_mtime = db_mtime;
            inner.cached_sessions_index.clone()
        };

        for (session_key, entry) in entries.iter() {
            let session_id = get_str(entry, "session_id");
            if session_id.is_empty() {
                continue;
            }

            let last_seen = {
                let inner = self.inner.lock().unwrap();
                *inner.last_poll_timestamps.get(session_key).unwrap_or(&0.0)
            };

            let messages = match source.get_messages(&session_id) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if messages.is_empty() {
                continue;
            }

            // Find messages newer than our last seen timestamp.
            let mut new_messages: Vec<&Value> = Vec::new();
            for msg in &messages {
                let ts = ts_float(msg.get("timestamp").unwrap_or(&Value::Null));
                let role = get_str(msg, "role");
                if role != "user" && role != "assistant" {
                    continue;
                }
                if ts > last_seen {
                    new_messages.push(msg);
                }
            }

            for msg in new_messages {
                let content = extract_message_content(msg);
                if content.is_empty() {
                    continue;
                }
                let mut data = Map::new();
                data.insert("role".into(), json!(get_str(msg, "role")));
                data.insert("content".into(), json!(truncate_chars(&content, 500)));
                data.insert("timestamp".into(), json!(value_to_str(msg.get("timestamp"))));
                data.insert("message_id".into(), json!(value_to_str(msg.get("id"))));
                self.enqueue(QueueEvent::new("message", session_key.clone(), data));
            }

            // Update last seen to the most recent message timestamp.
            let latest = messages
                .iter()
                .map(|m| ts_float(m.get("timestamp").unwrap_or(&Value::Null)))
                .fold(f64::NEG_INFINITY, f64::max);
            if latest.is_finite() && latest > last_seen {
                self.inner
                    .lock()
                    .unwrap()
                    .last_poll_timestamps
                    .insert(session_key.clone(), latest);
            }
        }

        Ok(())
    }
}

/// File mtime as seconds since the epoch, or 0.0 when missing/unreadable.
/// Mirrors `path.stat().st_mtime if path.exists() else 0.0` with `OSError`
/// swallowed.
fn file_mtime(path: &std::path::Path) -> f64 {
    if !path.exists() {
        return 0.0;
    }
    match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(t) => t
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        Err(_) => 0.0,
    }
}

/// Normalize a timestamp JSON value to f64. Mirrors the nested `_ts_float`
/// helper: ints/floats pass through, numeric strings parse, otherwise an ISO
/// string is parsed to an epoch timestamp, falling back to 0.0.
fn ts_float(ts: &Value) -> f64 {
    match ts {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) if !s.is_empty() => {
            if let Ok(f) = s.parse::<f64>() {
                return f;
            }
            // ISO string — parse to epoch.
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
                return dt.timestamp() as f64;
            }
            if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
                return naive.and_utc().timestamp() as f64;
            }
            if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
                return naive.and_utc().timestamp() as f64;
            }
            0.0
        }
        _ => 0.0,
    }
}

/// Render a JSON value as Python's `str(...)` would for the limited shapes seen
/// in message dicts (numbers, strings). Mirrors `str(msg.get(...))`.
fn value_to_str(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => {
            // Python int formatting: 5.0 -> "5.0", 5 -> "5". serde keeps ints
            // and floats distinct, so to_string is faithful enough here.
            n.to_string()
        }
        Some(other) => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// MCP tool handlers
//
// Each function reproduces a FastMCP `@mcp.tool()` from the Python module and
// returns the exact same JSON string. `json::to_string_pretty` matches Python's
// `json.dumps(..., indent=2)` (2-space indent).
// ---------------------------------------------------------------------------

fn dumps_pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| "{}".to_string())
}

fn dumps_compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
}

/// `conversations_list` — list active messaging conversations.
pub fn conversations_list(
    platform: Option<&str>,
    limit: usize,
    search: Option<&str>,
) -> String {
    let entries = load_sessions_index();
    let mut conversations: Vec<Map<String, Value>> = Vec::new();

    for (key, entry) in entries.iter() {
        let origin = get_obj(entry, "origin");
        // entry.get("platform") or origin.get("platform", "")
        let entry_platform = {
            let p = get_str(entry, "platform");
            if p.is_empty() {
                get_str(origin, "platform")
            } else {
                p
            }
        };

        if let Some(plat) = platform {
            if entry_platform.to_lowercase() != plat.to_lowercase() {
                continue;
            }
        }

        let display_name = get_str(entry, "display_name");
        let chat_name = get_str(origin, "chat_name");
        if let Some(s) = search {
            let search_lower = s.to_lowercase();
            if !display_name.to_lowercase().contains(&search_lower)
                && !chat_name.to_lowercase().contains(&search_lower)
                && !key.to_lowercase().contains(&search_lower)
            {
                continue;
            }
        }

        // chat_type: entry.get("chat_type", origin.get("chat_type", ""))
        let chat_type = match entry.get("chat_type") {
            Some(v) if !v.is_null() => value_or_empty_str(v),
            _ => get_str(origin, "chat_type"),
        };

        let mut c = Map::new();
        c.insert("session_key".into(), json!(key));
        c.insert("session_id".into(), json!(get_str(entry, "session_id")));
        c.insert("platform".into(), json!(entry_platform));
        c.insert("chat_type".into(), json!(chat_type));
        c.insert("display_name".into(), json!(display_name));
        c.insert("chat_name".into(), json!(chat_name));
        c.insert("user_name".into(), json!(get_str(origin, "user_name")));
        c.insert("updated_at".into(), json!(get_str(entry, "updated_at")));
        conversations.push(c);
    }

    // sort by updated_at descending, then take limit.
    conversations.sort_by(|a, b| {
        let ka = a.get("updated_at").and_then(Value::as_str).unwrap_or("");
        let kb = b.get("updated_at").and_then(Value::as_str).unwrap_or("");
        kb.cmp(ka)
    });
    conversations.truncate(limit);

    let out = json!({
        "count": conversations.len(),
        "conversations": conversations.into_iter().map(Value::Object).collect::<Vec<_>>(),
    });
    dumps_pretty(&out)
}

/// `conversation_get` — detailed info about one conversation by session key.
pub fn conversation_get(session_key: &str) -> String {
    let entries = load_sessions_index();
    let entry = match entries.get(session_key) {
        Some(e) => e,
        None => {
            return dumps_compact(&json!({
                "error": format!("Conversation not found: {session_key}")
            }))
        }
    };

    let origin = get_obj(entry, "origin");
    let platform = {
        let p = get_str(entry, "platform");
        if p.is_empty() {
            get_str(origin, "platform")
        } else {
            p
        }
    };
    let chat_type = match entry.get("chat_type") {
        Some(v) if !v.is_null() => value_or_empty_str(v),
        _ => get_str(origin, "chat_type"),
    };

    let out = json!({
        "session_key": session_key,
        "session_id": get_str(entry, "session_id"),
        "platform": platform,
        "chat_type": chat_type,
        "display_name": get_str(entry, "display_name"),
        "user_name": get_str(origin, "user_name"),
        "chat_name": get_str(origin, "chat_name"),
        "chat_id": get_str(origin, "chat_id"),
        // thread_id has no default in Python (None when absent)
        "thread_id": origin.get("thread_id").cloned().unwrap_or(Value::Null),
        "updated_at": get_str(entry, "updated_at"),
        "created_at": get_str(entry, "created_at"),
        "input_tokens": entry.get("input_tokens").cloned().unwrap_or(json!(0)),
        "output_tokens": entry.get("output_tokens").cloned().unwrap_or(json!(0)),
        "total_tokens": entry.get("total_tokens").cloned().unwrap_or(json!(0)),
    });
    dumps_pretty(&out)
}

/// `messages_read` — read recent messages from a conversation.
pub fn messages_read(source: &dyn MessageSource, session_key: &str, limit: usize) -> String {
    let entries = load_sessions_index();
    let entry = match entries.get(session_key) {
        Some(e) => e,
        None => {
            return dumps_compact(&json!({
                "error": format!("Conversation not found: {session_key}")
            }))
        }
    };

    let session_id = get_str(entry, "session_id");
    if session_id.is_empty() {
        return dumps_compact(&json!({"error": "No session ID for this conversation"}));
    }

    let all_messages = match source.get_messages(&session_id) {
        Ok(m) => m,
        Err(e) => {
            return dumps_compact(&json!({
                "error": format!("Failed to read messages: {e}")
            }))
        }
    };

    let mut filtered: Vec<Value> = Vec::new();
    for msg in &all_messages {
        let role = get_str(msg, "role");
        if role == "user" || role == "assistant" {
            let content = extract_message_content(msg);
            if !content.is_empty() {
                filtered.push(json!({
                    "id": value_to_str(msg.get("id")),
                    "role": role,
                    "content": truncate_chars(&content, 2000),
                    "timestamp": msg.get("timestamp").cloned().unwrap_or(json!("")),
                }));
            }
        }
    }

    let total = filtered.len();
    // messages = filtered[-limit:]
    let messages: Vec<Value> = if limit >= filtered.len() {
        filtered
    } else {
        filtered.split_off(filtered.len() - limit)
    };

    let out = json!({
        "session_key": session_key,
        "count": messages.len(),
        "total_in_session": total,
        "messages": messages,
    });
    dumps_pretty(&out)
}

/// `attachments_fetch` — list non-text attachments for a message.
pub fn attachments_fetch(
    source: &dyn MessageSource,
    session_key: &str,
    message_id: &str,
) -> String {
    let entries = load_sessions_index();
    let entry = match entries.get(session_key) {
        Some(e) => e,
        None => {
            return dumps_compact(&json!({
                "error": format!("Conversation not found: {session_key}")
            }))
        }
    };

    let session_id = get_str(entry, "session_id");
    if session_id.is_empty() {
        return dumps_compact(&json!({"error": "No session ID for this conversation"}));
    }

    let all_messages = match source.get_messages(&session_id) {
        Ok(m) => m,
        Err(e) => {
            return dumps_compact(&json!({
                "error": format!("Failed to read messages: {e}")
            }))
        }
    };

    let target_msg = all_messages
        .iter()
        .find(|msg| value_to_str(msg.get("id")) == message_id);

    let target_msg = match target_msg {
        Some(m) => m,
        None => {
            return dumps_compact(&json!({
                "error": format!("Message not found: {message_id}")
            }))
        }
    };

    let attachments = extract_attachments(target_msg);
    let out = json!({
        "message_id": message_id,
        "count": attachments.len(),
        "attachments": attachments,
    });
    dumps_pretty(&out)
}

/// `events_poll` — poll for new conversation events since a cursor position.
pub fn events_poll(
    bridge: &EventBridge,
    after_cursor: i64,
    session_key: Option<&str>,
    limit: usize,
) -> String {
    let result = bridge.poll_events(after_cursor, session_key, limit);
    dumps_pretty(&result)
}

/// `events_wait` — wait for the next conversation event (long-poll).
pub fn events_wait(
    bridge: &EventBridge,
    after_cursor: i64,
    session_key: Option<&str>,
    timeout_ms: u64,
) -> String {
    // Cap at 5 minutes (matches min(timeout_ms, 300000)).
    let event = bridge.wait_for_event(after_cursor, session_key, timeout_ms.min(300_000));
    match event {
        Some(e) => dumps_pretty(&json!({"event": e})),
        None => dumps_pretty(&json!({"event": Value::Null, "reason": "timeout"})),
    }
}

/// Result of attempting a send via the injected sender (see [`messages_send`]).
pub type SendFn<'a> = dyn Fn(&str, &str) -> Result<String, SendError> + 'a;

/// Failure modes for the send-message path, mirroring the Python `ImportError`
/// vs generic exception branches.
pub enum SendError {
    /// Equivalent to `ImportError` — the send tool is unavailable.
    Unavailable,
    /// Equivalent to a generic exception during send.
    Failed(String),
}

/// `messages_send` — send a message to a platform conversation.
///
/// The Python tool imported `tools.send_message_tool.send_message_tool` and
/// invoked it with `{"action": "send", "target", "message"}`. Here the sender
/// is injected so callers can wire in [`crate::send_message`] (or any other
/// transport) without this module owning the runtime.
pub fn messages_send(send: &SendFn<'_>, target: &str, message: &str) -> String {
    if target.is_empty() || message.is_empty() {
        return dumps_compact(&json!({"error": "Both target and message are required"}));
    }

    match send(target, message) {
        Ok(result_str) => result_str,
        Err(SendError::Unavailable) => {
            dumps_compact(&json!({"error": "Send message tool not available"}))
        }
        Err(SendError::Failed(e)) => dumps_compact(&json!({"error": format!("Send failed: {e}")})),
    }
}

/// `channels_list` — list available messaging channels and targets.
pub fn channels_list(platform: Option<&str>) -> String {
    let directory = load_channel_directory();

    if directory.is_empty() {
        // Fall back to deriving targets from the sessions index.
        let entries = load_sessions_index();
        let mut targets: Vec<Value> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (_key, entry) in entries.iter() {
            let origin = get_obj(entry, "origin");
            let p = {
                let pv = get_str(entry, "platform");
                if pv.is_empty() {
                    get_str(origin, "platform")
                } else {
                    pv
                }
            };
            let chat_id = get_str(origin, "chat_id");
            if p.is_empty() || chat_id.is_empty() {
                continue;
            }
            if let Some(plat) = platform {
                if p.to_lowercase() != plat.to_lowercase() {
                    continue;
                }
            }
            let target_str = format!("{p}:{chat_id}");
            if seen.contains(&target_str) {
                continue;
            }
            seen.insert(target_str.clone());

            // name: entry.get("display_name") or origin.get("chat_name", "")
            let name = {
                let dn = get_str(entry, "display_name");
                if dn.is_empty() {
                    get_str(origin, "chat_name")
                } else {
                    dn
                }
            };
            let chat_type = match entry.get("chat_type") {
                Some(v) if !v.is_null() => value_or_empty_str(v),
                _ => get_str(origin, "chat_type"),
            };
            targets.push(json!({
                "target": target_str,
                "platform": p,
                "name": name,
                "chat_type": chat_type,
            }));
        }
        return dumps_pretty(&json!({"count": targets.len(), "channels": targets}));
    }

    let mut channels: Vec<Value> = Vec::new();
    for (plat, entries_list) in directory.iter() {
        if let Some(filter) = platform {
            if plat.to_lowercase() != filter.to_lowercase() {
                continue;
            }
        }
        if let Value::Array(list) = entries_list {
            for ch in list {
                if let Value::Object(_) = ch {
                    // chat_id = ch.get("id", ch.get("chat_id", ""))
                    let chat_id = match ch.get("id") {
                        Some(v) if !v.is_null() => value_or_empty_str(v),
                        _ => get_str(ch, "chat_id"),
                    };
                    let target = if chat_id.is_empty() {
                        plat.clone()
                    } else {
                        format!("{plat}:{chat_id}")
                    };
                    // name = ch.get("name", ch.get("display_name", ""))
                    let name = match ch.get("name") {
                        Some(v) if !v.is_null() => value_or_empty_str(v),
                        _ => get_str(ch, "display_name"),
                    };
                    channels.push(json!({
                        "target": target,
                        "platform": plat,
                        "name": name,
                        "chat_type": get_str(ch, "type"),
                    }));
                }
            }
        }
    }

    dumps_pretty(&json!({"count": channels.len(), "channels": channels}))
}

/// `permissions_list_open` — list pending approval requests observed this
/// session.
pub fn permissions_list_open(bridge: &EventBridge) -> String {
    let approvals = bridge.list_pending_approvals();
    dumps_pretty(&json!({
        "count": approvals.len(),
        "approvals": approvals,
    }))
}

/// `permissions_respond` — respond to a pending approval request.
pub fn permissions_respond(bridge: &EventBridge, id: &str, decision: &str) -> String {
    if decision != "allow-once" && decision != "allow-always" && decision != "deny" {
        return dumps_compact(&json!({
            "error": format!(
                "Invalid decision: {decision}. Must be allow-once, allow-always, or deny"
            )
        }));
    }
    let result = bridge.respond_to_approval(id, decision);
    dumps_pretty(&result)
}

/// String form of a JSON value used where the Python code interpolated a
/// dict-get result that may be a string or number into a string field.
fn value_or_empty_str(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => {
            // Python str(True) == "True"
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_content_plain_string() {
        let msg = json!({"content": "hello"});
        assert_eq!(extract_message_content(&msg), "hello");
    }

    #[test]
    fn extract_content_empty_and_null() {
        assert_eq!(extract_message_content(&json!({"content": ""})), "");
        assert_eq!(extract_message_content(&json!({})), "");
        assert_eq!(extract_message_content(&json!({"content": Value::Null})), "");
    }

    #[test]
    fn extract_content_multipart_text() {
        let msg = json!({
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
                {"type": "text", "text": "line2"},
            ]
        });
        assert_eq!(extract_message_content(&msg), "line1\nline2");
    }

    #[test]
    fn attachments_image_url_and_media_tag() {
        let msg = json!({
            "content": [
                {"type": "text", "text": "see MEDIA: /tmp/a.png here"},
                {"type": "image_url", "image_url": {"url": "http://img"}},
                {"type": "weird", "blah": 1},
            ]
        });
        let atts = extract_attachments(&msg);
        // image_url block, weird block, then MEDIA tag from text
        assert_eq!(atts.len(), 3);
        assert_eq!(atts[0], json!({"type": "image", "url": "http://img"}));
        assert_eq!(atts[1]["type"], json!("weird"));
        assert_eq!(atts[2], json!({"type": "media", "path": "/tmp/a.png"}));
    }

    #[test]
    fn attachments_image_source_url() {
        let msg = json!({
            "content": [
                {"type": "image", "source": {"url": "http://src"}},
            ]
        });
        let atts = extract_attachments(&msg);
        assert_eq!(atts, vec![json!({"type": "image", "url": "http://src"})]);
    }

    #[test]
    fn ts_float_variants() {
        assert_eq!(ts_float(&json!(12.5)), 12.5);
        assert_eq!(ts_float(&json!("8")), 8.0);
        assert_eq!(ts_float(&json!("not-a-time")), 0.0);
        assert_eq!(ts_float(&Value::Null), 0.0);
        let iso = ts_float(&json!("2021-01-01T00:00:00+00:00"));
        assert_eq!(iso, 1609459200.0);
    }

    struct FakeSource(Vec<Value>);
    impl MessageSource for FakeSource {
        fn get_messages(&self, _session_id: &str) -> Result<Vec<Value>, String> {
            Ok(self.0.clone())
        }
    }

    fn write_sessions_index(home: &std::path::Path, body: &str) {
        let sessions = home.join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join("sessions.json"), body).unwrap();
    }

    #[test]
    fn conversation_get_not_found() {
        let tmp = std::env::temp_dir().join(format!("hermes_mcp_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        write_sessions_index(&tmp, "{}");
        let out = conversation_get("nope");
        assert!(out.contains("Conversation not found: nope"));
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn conversations_list_filters_and_sorts() {
        let tmp = std::env::temp_dir().join(format!("hermes_mcp_list_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let index = json!({
            "a": {"platform": "telegram", "display_name": "Alice", "updated_at": "2021-01-01", "session_id": "s1"},
            "b": {"platform": "discord", "display_name": "Bob", "updated_at": "2022-01-01", "session_id": "s2"},
        });
        write_sessions_index(&tmp, &index.to_string());

        let out = conversations_list(Some("telegram"), 50, None);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["count"], json!(1));
        assert_eq!(parsed["conversations"][0]["display_name"], json!("Alice"));

        // No filter: sorted by updated_at desc -> Bob first.
        let out2 = conversations_list(None, 50, None);
        let parsed2: Value = serde_json::from_str(&out2).unwrap();
        assert_eq!(parsed2["count"], json!(2));
        assert_eq!(parsed2["conversations"][0]["display_name"], json!("Bob"));

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn messages_read_filters_roles_and_truncates() {
        let tmp = std::env::temp_dir().join(format!("hermes_mcp_read_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let index = json!({"k": {"session_id": "sid"}});
        write_sessions_index(&tmp, &index.to_string());

        let big = "x".repeat(3000);
        let source = FakeSource(vec![
            json!({"id": 1, "role": "user", "content": "hi", "timestamp": 1.0}),
            json!({"id": 2, "role": "system", "content": "ignore me", "timestamp": 2.0}),
            json!({"id": 3, "role": "assistant", "content": big, "timestamp": 3.0}),
        ]);
        let out = messages_read(&source, "k", 50);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["count"], json!(2));
        assert_eq!(parsed["total_in_session"], json!(2));
        let content = parsed["messages"][1]["content"].as_str().unwrap();
        assert_eq!(content.chars().count(), 2000);

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn permissions_respond_invalid() {
        let bridge = EventBridge::new();
        let out = permissions_respond(&bridge, "x", "maybe");
        assert!(out.contains("Invalid decision: maybe"));
    }

    #[test]
    fn permissions_respond_not_found() {
        let bridge = EventBridge::new();
        let out = permissions_respond(&bridge, "missing", "deny");
        assert!(out.contains("Approval not found: missing"));
    }

    #[test]
    fn approval_roundtrip() {
        let bridge = EventBridge::new();
        bridge.add_pending_approval(
            "ap1",
            json!({"id": "ap1", "session_key": "sk", "created_at": "2021"}),
        );
        let listed = bridge.list_pending_approvals();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["id"], json!("ap1"));

        let resp = bridge.respond_to_approval("ap1", "allow-once");
        assert_eq!(resp["resolved"], json!(true));
        assert_eq!(bridge.list_pending_approvals().len(), 0);
    }

    #[test]
    fn poll_events_cursor_and_filter() {
        let bridge = EventBridge::new();
        let mut d1 = Map::new();
        d1.insert("content".into(), json!("a"));
        bridge.enqueue(QueueEvent::new("message", "s1", d1));
        let mut d2 = Map::new();
        d2.insert("content".into(), json!("b"));
        bridge.enqueue(QueueEvent::new("message", "s2", d2));

        let all = bridge.poll_events(0, None, 20);
        assert_eq!(all["events"].as_array().unwrap().len(), 2);
        assert_eq!(all["next_cursor"], json!(2));

        let filtered = bridge.poll_events(0, Some("s2"), 20);
        assert_eq!(filtered["events"].as_array().unwrap().len(), 1);
        assert_eq!(filtered["events"][0]["content"], json!("b"));
        assert_eq!(filtered["events"][0]["type"], json!("message"));

        let after = bridge.poll_events(1, None, 20);
        assert_eq!(after["events"].as_array().unwrap().len(), 1);
        assert_eq!(after["events"][0]["cursor"], json!(2));
    }

    #[test]
    fn channels_list_from_directory() {
        let tmp = std::env::temp_dir().join(format!("hermes_mcp_chan_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let dir = json!({
            "telegram": [{"id": "123", "name": "General", "type": "group"}],
            "discord": [{"chat_id": "456", "display_name": "dev"}],
        });
        std::fs::write(tmp.join("channel_directory.json"), dir.to_string()).unwrap();

        let out = channels_list(Some("telegram"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["count"], json!(1));
        assert_eq!(parsed["channels"][0]["target"], json!("telegram:123"));
        assert_eq!(parsed["channels"][0]["name"], json!("General"));

        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn messages_send_validation() {
        let send = |_t: &str, _m: &str| -> Result<String, SendError> { Ok("ok".into()) };
        assert!(messages_send(&send, "", "hi").contains("required"));
        assert!(messages_send(&send, "t:1", "").contains("required"));
        assert_eq!(messages_send(&send, "t:1", "hi"), "ok");

        let bad = |_t: &str, _m: &str| -> Result<String, SendError> {
            Err(SendError::Failed("boom".into()))
        };
        assert!(messages_send(&bad, "t:1", "hi").contains("Send failed: boom"));
    }

    #[test]
    fn events_wait_timeout_returns_none_payload() {
        let bridge = EventBridge::new();
        let out = events_wait(&bridge, 0, None, 10);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["event"], Value::Null);
        assert_eq!(parsed["reason"], json!("timeout"));
    }
}
