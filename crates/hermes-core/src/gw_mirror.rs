//! Session mirroring for cross-platform message delivery.
//!
//! When a message is sent to a platform (via send_message or cron delivery),
//! this module appends a "delivery-mirror" record to the target session's
//! transcript so the receiving-side agent has context about what was sent.
//!
//! Standalone — works from CLI, cron, and gateway contexts without needing
//! the full SessionStore machinery.
//!
//! Port of `gateway/mirror.py`.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;

use chrono::Local;
use serde_json::{Map, Value};

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_hermes_state::{MessageInput, SessionDB};

/// Directory holding per-session transcripts and the sessions index.
pub fn sessions_dir() -> PathBuf {
    get_hermes_home().join("sessions")
}

/// Path to the sessions index JSON file (`sessions.json`).
pub fn sessions_index_path() -> PathBuf {
    sessions_dir().join("sessions.json")
}

/// Append a delivery-mirror message to the target session's transcript.
///
/// Finds the gateway session that matches the given platform + chat_id,
/// then writes a mirror entry to both the JSONL transcript and SQLite DB.
///
/// Returns `true` if mirrored successfully, `false` if no matching session
/// or on error. All errors are swallowed — this is never fatal.
pub fn mirror_to_session(
    platform: &str,
    chat_id: &str,
    message_text: &str,
    source_label: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
) -> bool {
    let session_id = match find_session_id(platform, chat_id, thread_id, user_id) {
        Some(sid) => sid,
        None => {
            log::debug!(
                "Mirror: no session found for {}:{}:{}:{}",
                platform,
                chat_id,
                thread_id.unwrap_or(""),
                user_id.unwrap_or(""),
            );
            return false;
        }
    };

    // Build the mirror message object preserving the exact JSON shape the
    // Python source wrote to the JSONL transcript.
    let mut mirror_msg = Map::new();
    mirror_msg.insert("role".to_string(), Value::String("assistant".to_string()));
    mirror_msg.insert(
        "content".to_string(),
        Value::String(message_text.to_string()),
    );
    mirror_msg.insert(
        "timestamp".to_string(),
        // datetime.now().isoformat() — local naive time, microsecond precision.
        Value::String(Local::now().naive_local().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()),
    );
    mirror_msg.insert("mirror".to_string(), Value::Bool(true));
    mirror_msg.insert(
        "mirror_source".to_string(),
        Value::String(source_label.to_string()),
    );

    let mirror_msg = Value::Object(mirror_msg);

    append_to_jsonl(&session_id, &mirror_msg);
    append_to_sqlite(&session_id, &mirror_msg);

    log::debug!("Mirror: wrote to session {} (from {})", session_id, source_label);
    true
}

/// String coercion that mirrors Python's `str(x or "")` for an optional JSON
/// value: `None`/`null`/falsy → "", strings keep their raw value, numbers and
/// bools are stringified.
fn str_or_empty(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => {
            // Python str(True) == "True"; but origin fields are never bools in
            // practice. Treat falsy bool like Python `x or ""`.
            if *b {
                "True".to_string()
            } else {
                String::new()
            }
        }
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                // Python str(float) — fall back to default formatting.
                f.to_string()
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Like [`str_or_empty`] but does NOT collapse falsy values — mirrors plain
/// `str(origin.get("chat_id", ""))` where the default is `""` and any present
/// value is stringified as-is.
fn str_default_empty(v: Option<&Value>) -> String {
    match v {
        None => String::new(),
        Some(Value::Null) => "None".to_string(), // str(None) == "None"
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                f.to_string()
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Find the active `session_id` for a platform + chat_id pair.
///
/// Scans `sessions.json` entries and matches where `origin.chat_id == chat_id`
/// on the right platform. DM session keys don't embed the chat_id
/// (e.g. `"agent:main:telegram:dm"`), so we check the origin dict.
///
/// When `user_id` is provided, prefer exact sender matches. If multiple
/// same-chat candidates exist and none matches the user, return `None` instead
/// of guessing and contaminating another participant's session.
pub fn find_session_id(
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
) -> Option<String> {
    let index_path = sessions_index_path();
    if !index_path.exists() {
        return None;
    }

    let raw = std::fs::read_to_string(&index_path).ok()?;
    let data: Value = serde_json::from_str(&raw).ok()?;
    let obj = data.as_object()?;

    let platform_lower = platform.to_lowercase();
    let mut candidates: Vec<&Value> = Vec::new();

    for (_key, entry) in obj.iter() {
        let empty = Value::Object(Map::new());
        let origin = entry.get("origin").filter(|v| v.is_object()).unwrap_or(&empty);

        // (origin.get("platform") or entry.get("platform", "")).lower()
        let entry_platform = {
            let from_origin = origin.get("platform").and_then(|v| v.as_str());
            match from_origin {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => entry
                    .get("platform")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            }
        }
        .to_lowercase();

        if entry_platform != platform_lower {
            continue;
        }

        let origin_chat_id = str_default_empty(origin.get("chat_id"));
        if origin_chat_id == chat_id {
            if let Some(tid) = thread_id {
                // str(origin_thread_id or "") != str(thread_id)
                let origin_thread_id = str_or_empty(origin.get("thread_id"));
                if origin_thread_id != tid {
                    continue;
                }
            }
            candidates.push(entry);
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // user_id matching logic.
    if let Some(uid) = user_id.filter(|u| !u.is_empty()) {
        let exact_user_matches: Vec<&Value> = candidates
            .iter()
            .copied()
            .filter(|entry| {
                let origin = entry.get("origin").filter(|v| v.is_object());
                let entry_uid = str_or_empty(origin.and_then(|o| o.get("user_id")));
                entry_uid == uid
            })
            .collect();

        if !exact_user_matches.is_empty() {
            candidates = exact_user_matches;
        } else if candidates.len() > 1 {
            return None;
        }
    } else if candidates.len() > 1 {
        // No user_id given: if candidates span more than one distinct user,
        // refuse to guess.
        let mut distinct: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in &candidates {
            let origin = entry.get("origin").filter(|v| v.is_object());
            let uid = str_or_empty(origin.and_then(|o| o.get("user_id")));
            let trimmed = uid.trim();
            if !trimmed.is_empty() {
                distinct.insert(trimmed.to_string());
            }
        }
        if distinct.len() > 1 {
            return None;
        }
    }

    // max(candidates, key=lambda e: e.get("updated_at", "")).
    // Python's max returns the first maximal element on ties.
    let mut best: Option<&Value> = None;
    let mut best_key: String = String::new();
    for entry in &candidates {
        let key = entry
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if best.is_none() || key > best_key {
            best = Some(entry);
            best_key = key;
        }
    }

    best.and_then(|e| e.get("session_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Append a message to the JSONL transcript file.
fn append_to_jsonl(session_id: &str, message: &Value) {
    let transcript_path = sessions_dir().join(format!("{session_id}.jsonl"));
    // json.dumps(message, ensure_ascii=False) + "\n"
    let line = match serde_json::to_string(message) {
        Ok(s) => s,
        Err(e) => {
            log::debug!("Mirror JSONL write failed: {}", e);
            return;
        }
    };
    let result = (|| -> std::io::Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&transcript_path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    })();
    if let Err(e) = result {
        log::debug!("Mirror JSONL write failed: {}", e);
    }
}

/// Append a message to the SQLite session database.
fn append_to_sqlite(session_id: &str, message: &Value) {
    let result = (|| -> Result<(), String> {
        let db = SessionDB::new().map_err(|e| e.to_string())?;
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("assistant");
        let content = message
            .get("content")
            .cloned()
            .unwrap_or(Value::Null);
        let msg = MessageInput {
            role: role.to_string(),
            content,
            ..Default::default()
        };
        db.append_message(session_id, &msg).map_err(|e| e.to_string())?;
        db.close();
        Ok(())
    })();
    if let Err(e) = result {
        log::debug!("Mirror SQLite write failed: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate the HERMES_HOME env var.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write_index(dir: &std::path::Path, json: &str) {
        std::fs::create_dir_all(dir.join("sessions")).unwrap();
        std::fs::write(dir.join("sessions").join("sessions.json"), json).unwrap();
    }

    fn with_home<F: FnOnce()>(dir: &std::path::Path, f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("HERMES_HOME", dir);
        }
        f();
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }

    #[test]
    fn no_index_returns_none() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        with_home(&tmp, || {
            assert_eq!(find_session_id("telegram", "123", None, None), None);
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn matches_origin_chat_id() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "agent:main:telegram:dm": {
                "session_id": "sess-1",
                "updated_at": "2024-01-01T00:00:00",
                "origin": {"platform": "telegram", "chat_id": "123"}
            }
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            assert_eq!(
                find_session_id("telegram", "123", None, None),
                Some("sess-1".to_string())
            );
            // platform mismatch
            assert_eq!(find_session_id("slack", "123", None, None), None);
            // chat mismatch
            assert_eq!(find_session_id("telegram", "999", None, None), None);
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn numeric_chat_id_coerced() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "k": {
                "session_id": "sess-num",
                "updated_at": "2024",
                "origin": {"platform": "Telegram", "chat_id": 123}
            }
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            // platform compared case-insensitively; numeric chat_id stringified
            assert_eq!(
                find_session_id("telegram", "123", None, None),
                Some("sess-num".to_string())
            );
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn picks_latest_updated_at() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "a": {"session_id": "old", "updated_at": "2024-01-01", "origin": {"platform": "tg", "chat_id": "5", "user_id": "u"}},
            "b": {"session_id": "new", "updated_at": "2024-06-01", "origin": {"platform": "tg", "chat_id": "5", "user_id": "u"}}
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            assert_eq!(
                find_session_id("tg", "5", None, None),
                Some("new".to_string())
            );
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ambiguous_users_without_user_id_returns_none() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t4_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "a": {"session_id": "s-alice", "updated_at": "2024-01-01", "origin": {"platform": "tg", "chat_id": "5", "user_id": "alice"}},
            "b": {"session_id": "s-bob", "updated_at": "2024-06-01", "origin": {"platform": "tg", "chat_id": "5", "user_id": "bob"}}
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            // Distinct users, no user_id supplied → refuse to guess.
            assert_eq!(find_session_id("tg", "5", None, None), None);
            // Exact user match wins.
            assert_eq!(
                find_session_id("tg", "5", None, Some("bob")),
                Some("s-bob".to_string())
            );
            // user_id given but no match, multiple candidates → None.
            assert_eq!(find_session_id("tg", "5", None, Some("carol")), None);
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn thread_id_filter() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t5_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "a": {"session_id": "thread-1", "updated_at": "2024", "origin": {"platform": "tg", "chat_id": "5", "thread_id": "T1"}}
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            assert_eq!(
                find_session_id("tg", "5", Some("T1"), None),
                Some("thread-1".to_string())
            );
            // Wrong thread → no match.
            assert_eq!(find_session_id("tg", "5", Some("T2"), None), None);
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn single_candidate_no_user_id_ok() {
        let tmp = std::env::temp_dir().join(format!("hermes_mirror_t6_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let index = r#"{
            "a": {"session_id": "only", "updated_at": "2024", "origin": {"platform": "tg", "chat_id": "5", "user_id": "alice"}}
        }"#;
        write_index(&tmp, index);
        with_home(&tmp, || {
            // Single candidate → returned even without a user_id.
            assert_eq!(
                find_session_id("tg", "5", None, None),
                Some("only".to_string())
            );
        });
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
