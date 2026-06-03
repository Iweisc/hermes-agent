//! Session-scoped context variables for the Hermes gateway.
//!
//! This is a native Rust port of `gateway/session_context.py`.
//!
//! The Python version replaced the previous `os.environ`-based session state
//! (`HERMES_SESSION_PLATFORM`, `HERMES_SESSION_CHAT_ID`, etc.) with Python's
//! `contextvars.ContextVar`, whose values are *task-local*: each `asyncio`
//! task gets its own copy so concurrent messages never interfere.
//!
//! # Rust model
//!
//! The closest idiomatic equivalent to Python's `contextvars.ContextVar` for
//! async code is `tokio::task_local!`. However, `task_local!` only exposes a
//! value while inside a `scope(...)` future and cannot be mutated freely the
//! way Python's `var.set()` can. To faithfully reproduce the Python API
//! (`set_session_vars` returning tokens, `clear_session_vars(tokens)`, and
//! `get_session_env(name, default)` with the `_UNSET`-sentinel fallback to the
//! process environment), we store a per-task mutable map inside a single
//! task-local `RefCell`.
//!
//! Each task that wants session context runs its work inside
//! [`with_session_scope`] (analogous to entering an `asyncio` task with fresh
//! contextvars). Within that scope `set_session_vars` / `clear_session_vars` /
//! `get_session_env` operate on the task-local map. Outside any scope (CLI,
//! cron scheduler, tests), `get_session_env` falls back to `std::env`.
//!
//! # Semantics reproduced from Python
//!
//! There are three distinguishable states for each variable, mirroring the
//! Python `_UNSET` sentinel:
//!
//! 1. **Never set in this context** -> [`SessionValue::Unset`]. `get_session_env`
//!    falls back to the process environment, then to `default`.
//! 2. **Explicitly set** (even to `""`) via `set_session_vars` ->
//!    [`SessionValue::Set`]. That value is returned with **no** environment
//!    fallback.
//! 3. **Explicitly cleared** via `clear_session_vars` -> `SessionValue::Set("")`.
//!    Returns `""`, never falling back to a (potentially stale) environment.

use std::cell::RefCell;
use std::collections::HashMap;

/// The legacy `HERMES_SESSION_*` / `HERMES_CRON_AUTO_DELIVER_*` variable names
/// that the gateway tracks per task.
pub const VAR_NAMES: &[&str] = &[
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_USER_ID",
    "HERMES_SESSION_USER_NAME",
    "HERMES_SESSION_KEY",
    "HERMES_CRON_AUTO_DELIVER_PLATFORM",
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
];

/// The seven session variable names set together by [`set_session_vars`] /
/// reset by [`clear_session_vars`], in positional order.
pub const SESSION_VAR_NAMES: &[&str] = &[
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_USER_ID",
    "HERMES_SESSION_USER_NAME",
    "HERMES_SESSION_KEY",
];

/// Tri-state value for a session context variable, mirroring Python's `_UNSET`
/// sentinel vs. an explicitly-set string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionValue {
    /// Never set in this context — fall back to the process environment.
    Unset,
    /// Explicitly set (possibly to `""`) — return as-is, no fallback.
    Set(String),
}

impl SessionValue {
    /// Returns `true` when this value is the `_UNSET` sentinel.
    pub fn is_unset(&self) -> bool {
        matches!(self, SessionValue::Unset)
    }
}

/// A reset token returned by [`set_session_vars`], analogous to a
/// `contextvars.Token`.
///
/// The token records the variable name and the value it held *before* the set.
/// It is accepted by [`clear_session_vars`] for API compatibility with callers
/// that save the return value of `set_session_vars`, but — exactly as in the
/// Python implementation — `clear_session_vars` does **not** use the token to
/// restore the previous value; it sets each variable to `""` so the
/// "explicitly cleared" state is distinguishable from "never set".
#[derive(Debug, Clone)]
pub struct Token {
    /// The variable name this token resets.
    pub name: &'static str,
    /// The value the variable held before the corresponding `set`.
    pub old_value: SessionValue,
}

thread_local! {
    /// Per-task (per-thread, when used with `with_session_scope`) session map.
    ///
    /// `None` means "no session scope is active" — analogous to running with
    /// the bare contextvar defaults (all `_UNSET`). `Some(map)` is an active
    /// scope; absent keys in the map are also treated as `_UNSET`.
    static SESSION_MAP: RefCell<Option<HashMap<&'static str, SessionValue>>> =
        const { RefCell::new(None) };
}

/// Resolve a runtime variable name to the canonical `'static` name used as the
/// map key, so lookups and inserts share identical keys.
fn canonical_name(name: &str) -> Option<&'static str> {
    VAR_NAMES.iter().copied().find(|&n| n == name)
}

/// Run `f` inside a fresh session scope, mirroring the per-task isolation of
/// `contextvars.ContextVar`.
///
/// Any modifications made by [`set_session_vars`] / [`clear_session_vars`]
/// inside `f` are confined to this scope and discarded when it returns. Nested
/// scopes are supported: the previous scope (if any) is restored on exit.
pub fn with_session_scope<R>(f: impl FnOnce() -> R) -> R {
    let previous = SESSION_MAP.with(|cell| cell.replace(Some(HashMap::new())));
    let result = f();
    SESSION_MAP.with(|cell| {
        *cell.borrow_mut() = previous;
    });
    result
}

/// Returns `true` if a session scope is currently active on this task/thread.
pub fn in_session_scope() -> bool {
    SESSION_MAP.with(|cell| cell.borrow().is_some())
}

/// Set all seven session context variables and return reset tokens.
///
/// Mirrors Python `set_session_vars(...)`. Call [`clear_session_vars`] with the
/// returned tokens (typically in a cleanup/`finally` equivalent) to mark the
/// variables as explicitly cleared.
///
/// If no session scope is active, one is implicitly created for the current
/// thread (matching Python's behaviour where `var.set()` always succeeds and
/// affects the current context). Returns one [`Token`] per variable, in the
/// same positional order as the arguments.
#[allow(clippy::too_many_arguments)]
pub fn set_session_vars(
    platform: &str,
    chat_id: &str,
    chat_name: &str,
    thread_id: &str,
    user_id: &str,
    user_name: &str,
    session_key: &str,
) -> Vec<Token> {
    let values = [
        platform, chat_id, chat_name, thread_id, user_id, user_name, session_key,
    ];

    SESSION_MAP.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(HashMap::new());
        }
        let map = borrow.as_mut().expect("session map present");

        let mut tokens = Vec::with_capacity(SESSION_VAR_NAMES.len());
        for (&name, &value) in SESSION_VAR_NAMES.iter().zip(values.iter()) {
            let old_value = map.get(name).cloned().unwrap_or(SessionValue::Unset);
            tokens.push(Token { name, old_value });
            map.insert(name, SessionValue::Set(value.to_string()));
        }
        tokens
    })
}

/// Mark the seven session context variables as explicitly cleared.
///
/// Mirrors Python `clear_session_vars(tokens)`: each variable is set to `""`
/// (state [`SessionValue::Set`]`("")`) so that [`get_session_env`] returns an
/// empty string instead of falling back to a stale process-environment value.
///
/// The `tokens` argument is accepted for API compatibility but, exactly as in
/// the Python version, is **not** used to restore previous values.
pub fn clear_session_vars(_tokens: &[Token]) {
    SESSION_MAP.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(HashMap::new());
        }
        let map = borrow.as_mut().expect("session map present");
        for &name in SESSION_VAR_NAMES {
            map.insert(name, SessionValue::Set(String::new()));
        }
    });
}

/// Read a session context variable by its legacy `HERMES_SESSION_*` name.
///
/// Drop-in replacement for Python `os.getenv("HERMES_SESSION_*", default)`,
/// faithfully reproducing the resolution order:
///
/// 1. **Context variable** (set via [`set_session_vars`] / [`clear_session_vars`]).
///    If explicitly set — even to `""` — that value is returned with **no**
///    environment fallback.
/// 2. **Process environment** (`std::env`), only when the variable was never
///    set in this context (`_UNSET`) — i.e. CLI, cron scheduler, and tests.
/// 3. **`default`**.
pub fn get_session_env(name: &str, default: &str) -> String {
    if let Some(canon) = canonical_name(name) {
        let value = SESSION_MAP.with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|map| map.get(canon).cloned())
        });
        // `None` (no scope, or key absent) == Python `_UNSET` -> fall through.
        if let Some(SessionValue::Set(v)) = value {
            return v;
        }
    }
    // Fall back to the process environment for CLI, cron, and test compat.
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Convenience: read the raw [`SessionValue`] for a variable in the current
/// scope without applying the environment fallback. Returns
/// [`SessionValue::Unset`] when unknown, outside a scope, or never set.
pub fn get_session_value(name: &str) -> SessionValue {
    if let Some(canon) = canonical_name(name) {
        return SESSION_MAP.with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|map| map.get(canon).cloned())
                .unwrap_or(SessionValue::Unset)
        });
    }
    SessionValue::Unset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_then_get_returns_set_value() {
        with_session_scope(|| {
            set_session_vars("whatsapp", "chat-1", "Chat One", "thr-9", "u-1", "Alice", "key-x");
            assert_eq!(get_session_env("HERMES_SESSION_PLATFORM", "def"), "whatsapp");
            assert_eq!(get_session_env("HERMES_SESSION_CHAT_ID", "def"), "chat-1");
            assert_eq!(get_session_env("HERMES_SESSION_CHAT_NAME", "def"), "Chat One");
            assert_eq!(get_session_env("HERMES_SESSION_THREAD_ID", "def"), "thr-9");
            assert_eq!(get_session_env("HERMES_SESSION_USER_ID", "def"), "u-1");
            assert_eq!(get_session_env("HERMES_SESSION_USER_NAME", "def"), "Alice");
            assert_eq!(get_session_env("HERMES_SESSION_KEY", "def"), "key-x");
        });
    }

    #[test]
    fn explicit_empty_set_does_not_fall_back_to_env() {
        let key = "HERMES_SESSION_PLATFORM";
        unsafe {
            std::env::set_var(key, "from-env");
        }
        with_session_scope(|| {
            set_session_vars("", "", "", "", "", "", "");
            // Explicitly set to "" -> return "", NOT the env value.
            assert_eq!(get_session_env(key, "def"), "");
        });
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn unset_falls_back_to_env_then_default() {
        let key = "HERMES_SESSION_USER_ID";
        unsafe {
            std::env::remove_var(key);
        }
        // No scope active at all -> _UNSET -> env missing -> default.
        assert_eq!(get_session_env(key, "fallback"), "fallback");

        unsafe {
            std::env::set_var(key, "env-user");
        }
        assert_eq!(get_session_env(key, "fallback"), "env-user");

        // Inside a scope but variable never set -> still falls back to env.
        with_session_scope(|| {
            assert_eq!(get_session_env(key, "fallback"), "env-user");
        });
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn clear_sets_empty_and_blocks_env_fallback() {
        let key = "HERMES_SESSION_CHAT_ID";
        unsafe {
            std::env::set_var(key, "stale-env");
        }
        with_session_scope(|| {
            let tokens = set_session_vars("p", "live-chat", "", "", "", "", "");
            assert_eq!(get_session_env(key, "def"), "live-chat");
            clear_session_vars(&tokens);
            // Cleared -> "" returned, no env fallback.
            assert_eq!(get_session_env(key, "def"), "");
        });
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn cron_vars_not_touched_by_session_setters() {
        let key = "HERMES_CRON_AUTO_DELIVER_PLATFORM";
        unsafe {
            std::env::set_var(key, "cron-env");
        }
        with_session_scope(|| {
            // set_session_vars only sets the 7 session vars, not cron vars.
            set_session_vars("p", "c", "", "", "", "", "");
            // Cron var remains _UNSET -> falls back to env.
            assert_eq!(get_session_env(key, "def"), "cron-env");
            assert!(get_session_value(key).is_unset());
        });
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn tokens_record_previous_values() {
        with_session_scope(|| {
            let tokens = set_session_vars("a", "b", "c", "d", "e", "f", "g");
            assert_eq!(tokens.len(), 7);
            // First set: previous values are all Unset.
            for t in &tokens {
                assert_eq!(t.old_value, SessionValue::Unset);
            }
            let tokens2 = set_session_vars("a2", "b2", "c2", "d2", "e2", "f2", "g2");
            assert_eq!(tokens2[0].name, "HERMES_SESSION_PLATFORM");
            assert_eq!(tokens2[0].old_value, SessionValue::Set("a".to_string()));
        });
    }

    #[test]
    fn scopes_are_isolated() {
        with_session_scope(|| {
            set_session_vars("outer", "", "", "", "", "", "");
            assert_eq!(get_session_env("HERMES_SESSION_PLATFORM", ""), "outer");
            with_session_scope(|| {
                set_session_vars("inner", "", "", "", "", "", "");
                assert_eq!(get_session_env("HERMES_SESSION_PLATFORM", ""), "inner");
            });
            // Inner scope restored on exit.
            assert_eq!(get_session_env("HERMES_SESSION_PLATFORM", ""), "outer");
        });
    }

    #[test]
    fn unknown_name_falls_back_to_env() {
        let key = "HERMES_NOT_A_SESSION_VAR";
        unsafe {
            std::env::set_var(key, "xyz");
        }
        with_session_scope(|| {
            set_session_vars("p", "", "", "", "", "", "");
            assert_eq!(get_session_env(key, "def"), "xyz");
        });
        unsafe {
            std::env::remove_var(key);
        }
    }
}
