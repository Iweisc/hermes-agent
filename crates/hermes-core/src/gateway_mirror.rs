//! Session mirroring for cross-platform message delivery.
//!
//! Faithful port of `gateway/mirror.py`: when a message is delivered to a
//! platform, append a "delivery-mirror" assistant record to the matching
//! session's transcript (JSONL) and SQLite DB so the receiving-side agent has
//! context. Standalone — works from CLI, cron, and gateway contexts. Never
//! fatal: all errors resolve to `false`.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::state::{MessageAppend, SessionStore};

fn sessions_dir(hermes_home: &Path) -> PathBuf {
    hermes_home.join("sessions")
}

/// Append a delivery-mirror message to the target session's transcript.
///
/// Finds the gateway session matching `platform` + `chat_id` (+ optional
/// `thread_id`/`user_id`) in `<hermes_home>/sessions/sessions.json`, then writes
/// a mirror entry to both the JSONL transcript and the SQLite DB. Returns
/// `true` on success, `false` when no session matches or on any error.
///
/// `timestamp` is the ISO-8601 string for the mirror record (callers pass it so
/// the function does not read the clock itself).
pub fn mirror_to_session(
    hermes_home: &Path,
    platform: &str,
    chat_id: &str,
    message_text: &str,
    source_label: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
    timestamp: &str,
) -> bool {
    let Some(session_id) = find_session_id(hermes_home, platform, chat_id, thread_id, user_id)
    else {
        return false;
    };

    let mirror_msg = json!({
        "role": "assistant",
        "content": message_text,
        "timestamp": timestamp,
        "mirror": true,
        "mirror_source": source_label,
    });

    append_to_jsonl(hermes_home, &session_id, &mirror_msg);
    append_to_sqlite(hermes_home, &session_id, message_text);
    true
}

/// Find the active `session_id` for a platform + chat_id pair by scanning the
/// `sessions.json` index. Port of `_find_session_id`, including the
/// thread/user disambiguation rules.
pub fn find_session_id(
    hermes_home: &Path,
    platform: &str,
    chat_id: &str,
    thread_id: Option<&str>,
    user_id: Option<&str>,
) -> Option<String> {
    let index_path = sessions_dir(hermes_home).join("sessions.json");
    if !index_path.exists() {
        return None;
    }
    let data: Value = std::fs::read_to_string(&index_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())?;
    let entries = data.as_object()?;

    let platform_lower = platform.to_lowercase();
    let mut candidates: Vec<&Value> = Vec::new();

    for entry in entries.values() {
        let origin = entry.get("origin").and_then(Value::as_object);
        let entry_platform = origin
            .and_then(|o| o.get("platform"))
            .and_then(Value::as_str)
            .or_else(|| entry.get("platform").and_then(Value::as_str))
            .unwrap_or("")
            .to_lowercase();
        if entry_platform != platform_lower {
            continue;
        }
        let origin_chat_id = origin
            .and_then(|o| o.get("chat_id"))
            .map(value_to_string)
            .unwrap_or_default();
        if origin_chat_id != chat_id {
            continue;
        }
        if let Some(thread_id) = thread_id {
            let origin_thread_id = origin
                .and_then(|o| o.get("thread_id"))
                .map(value_to_string)
                .unwrap_or_default();
            if origin_thread_id != thread_id {
                continue;
            }
        }
        candidates.push(entry);
    }

    if candidates.is_empty() {
        return None;
    }

    let origin_user_id = |entry: &Value| -> String {
        entry
            .get("origin")
            .and_then(Value::as_object)
            .and_then(|o| o.get("user_id"))
            .map(value_to_string)
            .unwrap_or_default()
    };

    if let Some(user_id) = user_id {
        let exact: Vec<&Value> = candidates
            .iter()
            .copied()
            .filter(|entry| origin_user_id(entry) == user_id)
            .collect();
        if !exact.is_empty() {
            candidates = exact;
        } else if candidates.len() > 1 {
            return None;
        }
    } else if candidates.len() > 1 {
        let distinct: std::collections::BTreeSet<String> = candidates
            .iter()
            .map(|entry| origin_user_id(entry).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if distinct.len() > 1 {
            return None;
        }
    }

    // Pick the most-recently-updated candidate (max by updated_at string).
    let best = candidates.into_iter().max_by(|a, b| {
        let ua = a.get("updated_at").and_then(Value::as_str).unwrap_or("");
        let ub = b.get("updated_at").and_then(Value::as_str).unwrap_or("");
        ua.cmp(ub)
    })?;
    best.get("session_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Value-to-string mirroring Python `str(x or "")` for scalar JSON values used
/// as identifiers (string as-is, number stringified, null/absent -> "").
fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Append a mirror message to the JSONL transcript file (best-effort).
fn append_to_jsonl(hermes_home: &Path, session_id: &str, message: &Value) {
    use std::io::Write;
    let path = sessions_dir(hermes_home).join(format!("{session_id}.jsonl"));
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(message) {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Append a mirror message to the SQLite session DB (best-effort).
fn append_to_sqlite(hermes_home: &Path, session_id: &str, content: &str) {
    let Ok(store) = SessionStore::open(hermes_home.join("state.db")) else {
        return;
    };
    let message = MessageAppend {
        role: "assistant".to_string(),
        content: Some(Value::String(content.to_string())),
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
    };
    let _ = store.append_message(session_id, &message);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_index(home: &Path, json_text: &str) {
        let dir = sessions_dir(home);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("sessions.json"), json_text).unwrap();
    }

    #[test]
    fn find_matches_platform_and_chat_id() {
        let temp = TempDir::new().unwrap();
        write_index(
            temp.path(),
            r#"{
              "k1": {"session_id": "s1", "updated_at": "2026-01-01",
                     "origin": {"platform": "telegram", "chat_id": "100"}},
              "k2": {"session_id": "s2", "updated_at": "2026-02-01",
                     "origin": {"platform": "telegram", "chat_id": "100"}},
              "k3": {"session_id": "s3", "updated_at": "2026-03-01",
                     "origin": {"platform": "discord", "chat_id": "100"}}
            }"#,
        );
        // Two telegram/100 candidates with no distinct user_ids -> newest wins.
        assert_eq!(
            find_session_id(temp.path(), "telegram", "100", None, None).as_deref(),
            Some("s2")
        );
        // Wrong platform/chat -> None
        assert!(find_session_id(temp.path(), "telegram", "999", None, None).is_none());
    }

    #[test]
    fn distinct_users_without_user_id_is_ambiguous() {
        let temp = TempDir::new().unwrap();
        write_index(
            temp.path(),
            r#"{
              "k1": {"session_id": "s1", "updated_at": "2026-01-01",
                     "origin": {"platform": "telegram", "chat_id": "100", "user_id": "u1"}},
              "k2": {"session_id": "s2", "updated_at": "2026-02-01",
                     "origin": {"platform": "telegram", "chat_id": "100", "user_id": "u2"}}
            }"#,
        );
        // ambiguous (distinct users) without user_id -> None
        assert!(find_session_id(temp.path(), "telegram", "100", None, None).is_none());
        // exact user match -> that session
        assert_eq!(
            find_session_id(temp.path(), "telegram", "100", None, Some("u2")).as_deref(),
            Some("s2")
        );
        // unknown user among multiple -> None
        assert!(find_session_id(temp.path(), "telegram", "100", None, Some("u9")).is_none());
    }

    #[test]
    fn thread_id_filter() {
        let temp = TempDir::new().unwrap();
        write_index(
            temp.path(),
            r#"{
              "k1": {"session_id": "s1", "updated_at": "2026-01-01",
                     "origin": {"platform": "telegram", "chat_id": "100", "thread_id": "7"}}
            }"#,
        );
        assert_eq!(
            find_session_id(temp.path(), "telegram", "100", Some("7"), None).as_deref(),
            Some("s1")
        );
        assert!(find_session_id(temp.path(), "telegram", "100", Some("8"), None).is_none());
    }

    #[test]
    fn mirror_writes_jsonl_when_session_found() {
        let temp = TempDir::new().unwrap();
        write_index(
            temp.path(),
            r#"{"k1": {"session_id": "s1", "updated_at": "2026-01-01",
                      "origin": {"platform": "telegram", "chat_id": "100"}}}"#,
        );
        let ok = mirror_to_session(
            temp.path(),
            "telegram",
            "100",
            "hello there",
            "cron",
            None,
            None,
            "2026-05-31T00:00:00",
        );
        assert!(ok);
        let jsonl = fs::read_to_string(sessions_dir(temp.path()).join("s1.jsonl")).unwrap();
        let line: Value = serde_json::from_str(jsonl.trim()).unwrap();
        assert_eq!(line["role"], "assistant");
        assert_eq!(line["content"], "hello there");
        assert_eq!(line["mirror"], true);
        assert_eq!(line["mirror_source"], "cron");

        // No matching session -> false, no write.
        assert!(!mirror_to_session(
            temp.path(), "telegram", "999", "x", "cli", None, None, "t"
        ));
    }
}
