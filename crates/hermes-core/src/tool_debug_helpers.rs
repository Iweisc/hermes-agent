//! Shared debug session infrastructure for Hermes tools.
//!
//! Port of `tools/debug_helpers.py`. Replaces the identical
//! `DEBUG_MODE` / `_log_debug_call` / `_save_debug_log` /
//! `get_debug_session_info` boilerplate previously duplicated across
//! `web_tools`, `vision_tools`, `mixture_of_agents_tool`, and
//! `image_generation_tool`.
//!
//! Usage in a tool module:
//!
//! ```ignore
//! use crate::tool_debug_helpers::DebugSession;
//!
//! let mut debug = DebugSession::new("web_tools", "WEB_TOOLS_DEBUG");
//!
//! // Log a call (no-op when debug mode is off)
//! debug.log_call("web_search", serde_json::json!({"query": q, "results": n}));
//!
//! // Save the debug log (no-op when debug mode is off)
//! debug.save();
//!
//! // Expose debug info to external callers
//! let info = debug.get_session_info();
//! ```

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Map, Value};

use crate::mod_hermes_constants::get_hermes_home;

/// Return an ISO-8601 (`isoformat()`-style) timestamp for "now" in local time,
/// matching Python's `datetime.datetime.now().isoformat()` shape
/// (e.g. `2026-06-03T12:34:56.123456`).
fn now_isoformat() -> String {
    // Python's datetime.now().isoformat() emits microsecond precision with no
    // timezone suffix. chrono's "%.6f" gives the fractional seconds with a
    // leading dot.
    chrono::Local::now()
        .naive_local()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

/// Per-tool debug session that records tool calls to a JSON log file.
///
/// Activated by a tool-specific environment variable (e.g.
/// `WEB_TOOLS_DEBUG=true`). When disabled, all methods are cheap no-ops.
pub struct DebugSession {
    pub tool_name: String,
    enabled: bool,
    session_id: String,
    log_dir: PathBuf,
    calls: Vec<Value>,
    start_time: String,
}

impl DebugSession {
    /// Create a new debug session for `tool_name`, activated when the
    /// environment variable `env_var` is set to the string `"true"`
    /// (case-insensitive). Mirrors the Python constructor: the session id and
    /// start time are only populated when enabled, and the logs directory is
    /// created eagerly when enabled.
    pub fn new(tool_name: &str, env_var: &str) -> Self {
        let enabled = std::env::var(env_var)
            .unwrap_or_else(|_| "false".to_string())
            .to_lowercase()
            == "true";

        let session_id = if enabled {
            uuid_v4()
        } else {
            String::new()
        };

        let log_dir = get_hermes_home().join("logs");

        let start_time = if enabled {
            now_isoformat()
        } else {
            String::new()
        };

        if enabled {
            if let Err(e) = fs::create_dir_all(&log_dir) {
                log::error!(
                    "Error creating {} debug log dir: {}",
                    tool_name,
                    e
                );
            }
            log::debug!(
                "{} debug mode enabled - Session ID: {}",
                tool_name,
                session_id
            );
        }

        Self {
            tool_name: tool_name.to_string(),
            enabled,
            session_id,
            log_dir,
            calls: Vec::new(),
            start_time,
        }
    }

    /// Whether the debug session is active. Mirrors the Python `active`
    /// property.
    pub fn active(&self) -> bool {
        self.enabled
    }

    /// Append a tool-call entry to the in-memory log.
    ///
    /// `call_data` is merged into the entry alongside the `timestamp` and
    /// `tool_name` keys, matching the Python `{**call_data}` spread. To match
    /// Python semantics exactly, if `call_data` contains a `timestamp` or
    /// `tool_name` key it overrides the injected one (the spread comes last in
    /// the Python dict literal).
    pub fn log_call(&mut self, call_name: &str, call_data: Value) {
        if !self.enabled {
            return;
        }

        let mut entry: Map<String, Value> = Map::new();
        entry.insert("timestamp".to_string(), json!(now_isoformat()));
        entry.insert("tool_name".to_string(), json!(call_name));

        if let Value::Object(map) = call_data {
            for (k, v) in map {
                entry.insert(k, v);
            }
        }

        self.calls.push(Value::Object(entry));
    }

    /// Flush the in-memory log to a JSON file in the logs directory.
    /// No-op when debug mode is off. Errors are logged, not propagated,
    /// matching the Python `try/except` behavior.
    pub fn save(&self) {
        if !self.enabled {
            return;
        }

        let filename = format!("{}_debug_{}.json", self.tool_name, self.session_id);
        let filepath = self.log_dir.join(&filename);

        let payload = json!({
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": now_isoformat(),
            "debug_enabled": true,
            "total_calls": self.calls.len(),
            "tool_calls": self.calls,
        });

        // serde_json::to_string_pretty uses 2-space indentation, matching
        // json.dump(..., indent=2). ensure_ascii=False is the default for
        // serde_json (non-ASCII is emitted verbatim).
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => match fs::write(&filepath, text) {
                Ok(()) => log::debug!(
                    "{} debug log saved: {}",
                    self.tool_name,
                    filepath.display()
                ),
                Err(e) => log::error!(
                    "Error saving {} debug log: {}",
                    self.tool_name,
                    e
                ),
            },
            Err(e) => log::error!(
                "Error saving {} debug log: {}",
                self.tool_name,
                e
            ),
        }
    }

    /// Return a summary value suitable for returning from
    /// `get_debug_session_info()`. When disabled, the nullable fields are
    /// JSON `null` to match the Python `None` values.
    pub fn get_session_info(&self) -> Value {
        if !self.enabled {
            return json!({
                "enabled": false,
                "session_id": Value::Null,
                "log_path": Value::Null,
                "total_calls": 0,
            });
        }

        let log_path = self
            .log_dir
            .join(format!("{}_debug_{}.json", self.tool_name, self.session_id));

        json!({
            "enabled": true,
            "session_id": self.session_id,
            "log_path": log_path.to_string_lossy(),
            "total_calls": self.calls.len(),
        })
    }
}

/// Generate a random UUID v4 string in canonical hyphenated form, mirroring
/// Python's `str(uuid.uuid4())`. Implemented without an external crate using a
/// best-effort entropy source so the module stays dependency-light.
fn uuid_v4() -> String {
    let mut bytes = random_16_bytes();

    // Set version (4) and variant (RFC 4122) bits, as uuid.uuid4() does.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

/// Best-effort 16 bytes of entropy. Combines process/thread/time-derived
/// values through a simple xorshift mixer. This is sufficient for log-file
/// session identifiers (which is the only consumer).
fn random_16_bytes() -> [u8; 16] {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let pid = std::process::id() as u64;

    // Use the address of a stack local as an additional entropy source
    // (ASLR / stack-layout variability between processes and calls).
    let stack_marker = &nanos as *const u64 as u64;

    let mut state = nanos
        ^ pid.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ stack_marker.rotate_left(17);
    if state == 0 {
        state = 0x1234_5678_9ABC_DEF0;
    }

    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(8) {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let val = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let vb = val.to_le_bytes();
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = vb[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_env_var(suffix: &str) -> String {
        format!("HERMES_TEST_DEBUG_{}_{}", std::process::id(), suffix)
    }

    #[test]
    fn disabled_session_is_noop() {
        let var = unique_env_var("DISABLED");
        unsafe {
            std::env::remove_var(&var);
        }
        let mut sess = DebugSession::new("web_tools", &var);
        assert!(!sess.active());
        // log_call is a no-op
        sess.log_call("web_search", json!({"query": "x"}));
        sess.save(); // must not panic / write

        let info = sess.get_session_info();
        assert_eq!(info["enabled"], json!(false));
        assert_eq!(info["session_id"], Value::Null);
        assert_eq!(info["log_path"], Value::Null);
        assert_eq!(info["total_calls"], json!(0));
    }

    #[test]
    fn enabled_session_records_and_saves() {
        let var = unique_env_var("ENABLED");
        let tmp = std::env::temp_dir().join(format!(
            "hermes_dbg_test_{}_{}",
            std::process::id(),
            "enabled"
        ));
        unsafe {
            std::env::set_var(&var, "TRUE");
            std::env::set_var("HERMES_HOME", &tmp);
        }

        let mut sess = DebugSession::new("vision_tools", &var);
        assert!(sess.active());
        assert!(!sess.session_id.is_empty());

        sess.log_call("describe", json!({"model": "x", "count": 2}));
        sess.log_call("describe", json!({"model": "y"}));

        let info = sess.get_session_info();
        assert_eq!(info["enabled"], json!(true));
        assert_eq!(info["total_calls"], json!(2));

        sess.save();

        let saved = tmp
            .join("logs")
            .join(format!("vision_tools_debug_{}.json", sess.session_id));
        let text = std::fs::read_to_string(&saved).expect("log file written");
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["debug_enabled"], json!(true));
        assert_eq!(parsed["total_calls"], json!(2));
        assert_eq!(parsed["session_id"], json!(sess.session_id));
        assert_eq!(parsed["tool_calls"].as_array().unwrap().len(), 2);
        // injected keys present
        assert_eq!(parsed["tool_calls"][0]["tool_name"], json!("describe"));
        assert!(parsed["tool_calls"][0]["timestamp"].is_string());
        assert_eq!(parsed["tool_calls"][0]["model"], json!("x"));
        assert_eq!(parsed["tool_calls"][0]["count"], json!(2));

        // cleanup
        unsafe {
            std::env::remove_var(&var);
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn call_data_can_override_injected_keys() {
        // Matches Python {**call_data} spreading last: a tool_name inside
        // call_data overrides the call_name argument.
        let var = unique_env_var("OVERRIDE");
        let tmp = std::env::temp_dir().join(format!(
            "hermes_dbg_test_{}_{}",
            std::process::id(),
            "override"
        ));
        unsafe {
            std::env::set_var(&var, "true");
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let mut sess = DebugSession::new("moa", &var);
        sess.log_call("orig", json!({"tool_name": "overridden", "x": 1}));
        let info = sess.get_session_info();
        assert_eq!(info["total_calls"], json!(1));
        assert_eq!(sess.calls[0]["tool_name"], json!("overridden"));
        assert_eq!(sess.calls[0]["x"], json!(1));

        unsafe {
            std::env::remove_var(&var);
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn uuid_v4_shape() {
        let id = uuid_v4();
        assert_eq!(id.len(), 36);
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 12 - 8); // 4
        assert_eq!(parts[4].len(), 12);
        // version nibble
        assert_eq!(&parts[2][0..1], "4");
        // variant nibble is one of 8,9,a,b
        let v = &parts[3][0..1];
        assert!(["8", "9", "a", "b"].contains(&v), "variant was {}", v);
        // two distinct calls differ
        assert_ne!(uuid_v4(), uuid_v4());
    }
}
