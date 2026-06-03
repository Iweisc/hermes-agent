//! Generic slash-command confirmation primitive (gateway-side).
//!
//! Slash commands that have a non-destructive but expensive side effect worth
//! surfacing to the user (currently only `/reload-mcp`, which invalidates the
//! provider prompt cache) route through this module.
//!
//! Two delivery paths:
//!
//!   1. Button UI — adapters that override `send_slash_confirm` render three
//!      inline buttons (Approve Once / Always Approve / Cancel). The button
//!      callback calls [`resolve`].
//!
//!   2. Text fallback — adapters without button UIs get a plain text prompt.
//!      Users reply with `/approve`, `/always`, or `/cancel`; the gateway's
//!      `_handle_message` intercepts those replies and calls [`resolve`]
//!      directly.
//!
//! State is stored module-level (like `tools.approval`) so platform adapters
//! can resolve callbacks without needing a backreference to the gateway
//! runner instance.
//!
//! This is a faithful native Rust port of `tools/slash_confirm.py`. The Python
//! handler was an `async` callable; here it is modeled as a boxed synchronous
//! closure `Box<dyn FnOnce(&str) -> Option<String> + Send>`. The single async
//! point in the original (`await handler(choice)`) collapses to a direct call
//! because the Rust callers driving this are synchronous (reqwest::blocking
//! gateway runtime). Any panic inside the handler is caught and surfaced as an
//! error string, mirroring the Python `try/except` behavior.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default timeout — a pending confirm older than this is discarded when the
/// next message arrives for the same session. Buttons work up until the
/// adapter drops the callback_data (Telegram: ~48h; Discord: ephemeral;
/// Slack: 3s ack + long-lived actions).
pub const DEFAULT_TIMEOUT_SECONDS: f64 = 300.0;

/// The choice a user makes when resolving a confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Approve this single invocation.
    Once,
    /// Approve and remember for future invocations.
    Always,
    /// Cancel without running.
    Cancel,
}

impl Choice {
    /// Parse a choice from its string form (`"once"`, `"always"`, `"cancel"`).
    pub fn from_str(s: &str) -> Option<Choice> {
        match s {
            "once" => Some(Choice::Once),
            "always" => Some(Choice::Always),
            "cancel" => Some(Choice::Cancel),
            _ => None,
        }
    }

    /// The canonical string form, matching the Python protocol values.
    pub fn as_str(&self) -> &'static str {
        match self {
            Choice::Once => "once",
            Choice::Always => "always",
            Choice::Cancel => "cancel",
        }
    }
}

/// Handler invoked when a confirmation resolves. Receives the chosen string
/// (`"once"` / `"always"` / `"cancel"`) and returns an optional follow-up
/// message to send to the user.
pub type Handler = Box<dyn FnOnce(&str) -> Option<String> + Send>;

/// A pending slash-command confirmation entry.
struct PendingEntry {
    confirm_id: String,
    command: String,
    handler: Handler,
    created_at: f64,
}

/// A snapshot view of a pending entry, returned by [`get_pending`]. The
/// handler itself is non-cloneable, so it is intentionally excluded — this
/// mirrors the Python `get_pending` returning a shallow `dict(entry)` copy
/// callers only inspect for metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingInfo {
    pub confirm_id: String,
    pub command: String,
    pub created_at_millis: u64,
}

fn registry() -> &'static Mutex<HashMap<String, PendingEntry>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, PendingEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Current wall-clock time in seconds since the Unix epoch (matches Python's
/// `time.time()`).
fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Register a pending slash-command confirmation.
///
/// Overwrites any prior pending confirm for the same `session_key` — the user
/// invoking a new confirmable command supersedes the stale one.
pub fn register(session_key: &str, confirm_id: &str, command: &str, handler: Handler) {
    register_at(session_key, confirm_id, command, handler, now_seconds());
}

/// Like [`register`] but with an explicit creation timestamp (seconds since
/// the Unix epoch). Primarily for tests.
pub fn register_at(
    session_key: &str,
    confirm_id: &str,
    command: &str,
    handler: Handler,
    created_at: f64,
) {
    let entry = PendingEntry {
        confirm_id: confirm_id.to_string(),
        command: command.to_string(),
        handler,
        created_at,
    };
    let mut map = registry().lock().expect("slash_confirm registry poisoned");
    map.insert(session_key.to_string(), entry);
}

/// Return metadata for the pending confirm for a session, or `None`.
pub fn get_pending(session_key: &str) -> Option<PendingInfo> {
    let map = registry().lock().expect("slash_confirm registry poisoned");
    map.get(session_key).map(|e| PendingInfo {
        confirm_id: e.confirm_id.clone(),
        command: e.command.clone(),
        created_at_millis: (e.created_at * 1000.0) as u64,
    })
}

/// Drop the pending confirm for `session_key` without running it.
pub fn clear(session_key: &str) {
    let mut map = registry().lock().expect("slash_confirm registry poisoned");
    map.remove(session_key);
}

/// Drop the pending confirm if older than `timeout` seconds.
///
/// Returns `true` if an entry was dropped.
pub fn clear_if_stale(session_key: &str, timeout: f64) -> bool {
    clear_if_stale_at(session_key, timeout, now_seconds())
}

/// Like [`clear_if_stale`] but with an explicit "now" timestamp. For tests.
pub fn clear_if_stale_at(session_key: &str, timeout: f64, now: f64) -> bool {
    let mut map = registry().lock().expect("slash_confirm registry poisoned");
    let stale = match map.get(session_key) {
        None => return false,
        Some(entry) => now - entry.created_at > timeout,
    };
    if stale {
        map.remove(session_key);
        return true;
    }
    false
}

/// Resolve a pending confirm.
///
/// `choice` must be one of `"once"`, `"always"`, or `"cancel"`. Returns the
/// handler's output string (to be sent as a follow-up message), or `None` if
/// the confirm was stale, already resolved, or the `confirm_id` doesn't match.
///
/// The entry is popped *before* the handler runs to prevent duplicate
/// callbacks (e.g. a button double-click) from running it twice — matching the
/// Python implementation exactly.
pub fn resolve(session_key: &str, confirm_id: &str, choice: &str, timeout: f64) -> Option<String> {
    resolve_at(session_key, confirm_id, choice, timeout, now_seconds())
}

/// Like [`resolve`] but with an explicit "now" timestamp. For tests.
pub fn resolve_at(
    session_key: &str,
    confirm_id: &str,
    choice: &str,
    timeout: f64,
    now: f64,
) -> Option<String> {
    // Extract handler + metadata under the lock, then drop the lock before
    // running the (potentially slow) handler — mirroring the Python `with
    // _lock:` block scope.
    let (handler, command, stale) = {
        let mut map = registry().lock().expect("slash_confirm registry poisoned");
        let entry = map.get(session_key)?;
        if entry.confirm_id != confirm_id {
            // Stale confirm_id — superseded by a newer prompt on this session.
            return None;
        }
        // Pop before we run the handler to prevent duplicate callbacks from
        // running it twice.
        let entry = map.remove(session_key)?;
        let stale = now - entry.created_at > timeout;
        (entry.handler, entry.command, stale)
    };

    if stale {
        return None;
    }

    // Run the handler, catching panics the way Python caught exceptions.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(choice)));
    match result {
        Ok(out) => out,
        Err(err) => {
            let msg = panic_message(&err);
            log::error!("Slash-confirm handler for /{command} raised: {msg}");
            Some(format!("\u{274c} Error handling confirmation: {msg}"))
        }
    }
}

fn panic_message(err: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    // Each test uses a unique session_key so the shared global registry does
    // not cross-contaminate tests run in parallel.
    fn key(name: &str) -> String {
        format!("test::{name}")
    }

    #[test]
    fn choice_roundtrip() {
        assert_eq!(Choice::from_str("once"), Some(Choice::Once));
        assert_eq!(Choice::from_str("always"), Some(Choice::Always));
        assert_eq!(Choice::from_str("cancel"), Some(Choice::Cancel));
        assert_eq!(Choice::from_str("nope"), None);
        assert_eq!(Choice::Once.as_str(), "once");
        assert_eq!(Choice::Always.as_str(), "always");
        assert_eq!(Choice::Cancel.as_str(), "cancel");
    }

    #[test]
    fn register_and_get_pending() {
        let k = key("get_pending");
        register_at(&k, "cid-1", "reload-mcp", Box::new(|_| None), 100.0);
        let info = get_pending(&k).expect("should be pending");
        assert_eq!(info.confirm_id, "cid-1");
        assert_eq!(info.command, "reload-mcp");
        assert_eq!(info.created_at_millis, 100_000);
        // Unknown session is None.
        assert!(get_pending(&key("nonexistent_xyz")).is_none());
        clear(&k);
    }

    #[test]
    fn register_overwrites_prior() {
        let k = key("overwrite");
        register_at(&k, "cid-1", "reload-mcp", Box::new(|_| None), 100.0);
        register_at(&k, "cid-2", "reload-mcp", Box::new(|_| None), 200.0);
        let info = get_pending(&k).unwrap();
        assert_eq!(info.confirm_id, "cid-2");
        clear(&k);
    }

    #[test]
    fn clear_drops_entry() {
        let k = key("clear");
        register_at(&k, "cid", "cmd", Box::new(|_| None), 0.0);
        assert!(get_pending(&k).is_some());
        clear(&k);
        assert!(get_pending(&k).is_none());
        // Idempotent.
        clear(&k);
    }

    #[test]
    fn clear_if_stale_behavior() {
        let k = key("stale");
        // Missing entry -> false.
        assert!(!clear_if_stale_at(&key("stale_missing"), 300.0, 1000.0));

        register_at(&k, "cid", "cmd", Box::new(|_| None), 100.0);
        // Not yet stale (now - created = 200 <= 300).
        assert!(!clear_if_stale_at(&k, 300.0, 300.0));
        assert!(get_pending(&k).is_some());
        // Now stale (now - created = 500 > 300).
        assert!(clear_if_stale_at(&k, 300.0, 600.0));
        assert!(get_pending(&k).is_none());
    }

    #[test]
    fn resolve_runs_handler_and_returns_output() {
        let k = key("resolve_ok");
        register_at(
            &k,
            "cid",
            "reload-mcp",
            Box::new(|choice| Some(format!("did:{choice}"))),
            100.0,
        );
        let out = resolve_at(&k, "cid", "always", 300.0, 150.0);
        assert_eq!(out.as_deref(), Some("did:always"));
        // Entry was popped.
        assert!(get_pending(&k).is_none());
    }

    #[test]
    fn resolve_handler_returning_none() {
        let k = key("resolve_none");
        register_at(&k, "cid", "cmd", Box::new(|_| None), 100.0);
        assert_eq!(resolve_at(&k, "cid", "once", 300.0, 150.0), None);
    }

    #[test]
    fn resolve_missing_session() {
        assert_eq!(
            resolve_at(&key("resolve_missing_xyz"), "cid", "once", 300.0, 150.0),
            None
        );
    }

    #[test]
    fn resolve_wrong_confirm_id_keeps_entry() {
        let k = key("resolve_wrongid");
        register_at(&k, "cid-real", "cmd", Box::new(|_| Some("ran".into())), 100.0);
        // Wrong confirm_id -> None and entry preserved.
        assert_eq!(resolve_at(&k, "cid-other", "once", 300.0, 150.0), None);
        assert!(get_pending(&k).is_some());
        clear(&k);
    }

    #[test]
    fn resolve_stale_does_not_run_handler() {
        let k = key("resolve_stale");
        let ran = Arc::new(AtomicU32::new(0));
        let ran2 = ran.clone();
        register_at(
            &k,
            "cid",
            "cmd",
            Box::new(move |_| {
                ran2.fetch_add(1, Ordering::SeqCst);
                Some("ran".into())
            }),
            100.0,
        );
        // now - created = 500 > 300 -> stale, returns None, handler not run,
        // but entry is still popped.
        assert_eq!(resolve_at(&k, "cid", "once", 300.0, 600.0), None);
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        assert!(get_pending(&k).is_none());
    }

    #[test]
    fn resolve_pops_before_running_no_double_run() {
        let k = key("resolve_double");
        register_at(&k, "cid", "cmd", Box::new(|_| Some("first".into())), 100.0);
        assert_eq!(
            resolve_at(&k, "cid", "once", 300.0, 150.0).as_deref(),
            Some("first")
        );
        // Second resolve finds nothing.
        assert_eq!(resolve_at(&k, "cid", "once", 300.0, 150.0), None);
    }

    #[test]
    fn resolve_handler_panic_is_caught() {
        let k = key("resolve_panic");
        register_at(
            &k,
            "cid",
            "reload-mcp",
            Box::new(|_| panic!("boom")),
            100.0,
        );
        let out = resolve_at(&k, "cid", "once", 300.0, 150.0).expect("error message");
        assert!(out.contains("Error handling confirmation"));
        assert!(out.contains("boom"));
        // Entry popped even on panic.
        assert!(get_pending(&k).is_none());
    }
}
