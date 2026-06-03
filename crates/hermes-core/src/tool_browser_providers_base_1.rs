//! Abstract base for cloud browser providers.
//!
//! Port of `tools/browser_providers/base.py`.
//!
//! The Python module defines an `abc.ABC` with five abstract methods. In Rust
//! the natural analogue is a trait, [`CloudBrowserProvider`], which concrete
//! provider modules (Browserbase, Steel, etc.) implement. The trait is
//! registered in the browser tool's provider registry; the user selects a
//! provider via `hermes setup` / `hermes tools` and the choice is persisted as
//! `config["browser"]["cloud_provider"]`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Metadata returned by [`CloudBrowserProvider::create_session`].
///
/// Mirrors the dict contract documented in the Python `create_session`
/// docstring. `bb_session_id` is a legacy key name kept for backward compat
/// with the rest of `browser_tool.py` — it holds the provider's session ID
/// regardless of which provider is in use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique name for `agent-browser --session`.
    pub session_name: String,
    /// Provider session ID (for close/cleanup). Legacy key name.
    pub bb_session_id: String,
    /// CDP websocket URL.
    pub cdp_url: String,
    /// Feature flags that were enabled.
    pub features: HashMap<String, Value>,
}

impl SessionMetadata {
    /// Construct metadata with no enabled features.
    pub fn new(
        session_name: impl Into<String>,
        bb_session_id: impl Into<String>,
        cdp_url: impl Into<String>,
    ) -> Self {
        Self {
            session_name: session_name.into(),
            bb_session_id: bb_session_id.into(),
            cdp_url: cdp_url.into(),
            features: HashMap::new(),
        }
    }

    /// Render the metadata as a `serde_json::Value` dict matching the exact
    /// shape produced by the Python implementation:
    ///
    /// ```text
    /// {
    ///     "session_name": str,
    ///     "bb_session_id": str,
    ///     "cdp_url": str,
    ///     "features": dict,
    /// }
    /// ```
    pub fn to_value(&self) -> Value {
        let mut features = serde_json::Map::new();
        for (k, v) in &self.features {
            features.insert(k.clone(), v.clone());
        }
        let mut map = serde_json::Map::new();
        map.insert("session_name".into(), Value::String(self.session_name.clone()));
        map.insert(
            "bb_session_id".into(),
            Value::String(self.bb_session_id.clone()),
        );
        map.insert("cdp_url".into(), Value::String(self.cdp_url.clone()));
        map.insert("features".into(), Value::Object(features));
        Value::Object(map)
    }
}

/// Interface for cloud browser backends (Browserbase, Steel, etc.).
///
/// Implementations live in sibling modules and are registered in the browser
/// tool's provider registry. Each abstract method from the Python
/// `CloudBrowserProvider` ABC maps to a required trait method here.
pub trait CloudBrowserProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    ///
    /// Called at tool-registration time (`check_browser_requirements`) to gate
    /// availability. Must be cheap — no network calls.
    fn is_configured(&self) -> bool;

    /// Create a cloud browser session and return session metadata.
    ///
    /// Returns a [`SessionMetadata`] (equivalent to the documented dict with at
    /// least `session_name`, `bb_session_id`, `cdp_url`, and `features`). On
    /// failure, returns an error string rather than panicking.
    fn create_session(&self, task_id: &str) -> Result<SessionMetadata, String>;

    /// Release / terminate a cloud session by its provider session ID.
    ///
    /// Returns `true` on success, `false` on failure. Should not raise/panic.
    fn close_session(&self, session_id: &str) -> bool;

    /// Best-effort session teardown during process exit.
    ///
    /// Called from atexit / signal handlers. Must tolerate missing credentials,
    /// network errors, etc. — log and move on (never panic).
    fn emergency_cleanup(&self, session_id: &str);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeProvider {
        configured: bool,
        closed: std::cell::RefCell<Vec<String>>,
        cleaned: std::cell::RefCell<Vec<String>>,
    }

    impl FakeProvider {
        fn new(configured: bool) -> Self {
            Self {
                configured,
                closed: std::cell::RefCell::new(Vec::new()),
                cleaned: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    impl CloudBrowserProvider for FakeProvider {
        fn provider_name(&self) -> String {
            "fake".to_string()
        }

        fn is_configured(&self) -> bool {
            self.configured
        }

        fn create_session(&self, task_id: &str) -> Result<SessionMetadata, String> {
            if !self.configured {
                return Err("not configured".to_string());
            }
            let mut meta = SessionMetadata::new(
                format!("session-{task_id}"),
                format!("sid-{task_id}"),
                "wss://example/cdp".to_string(),
            );
            meta.features
                .insert("stealth".to_string(), Value::Bool(true));
            Ok(meta)
        }

        fn close_session(&self, session_id: &str) -> bool {
            self.closed.borrow_mut().push(session_id.to_string());
            self.configured
        }

        fn emergency_cleanup(&self, session_id: &str) {
            self.cleaned.borrow_mut().push(session_id.to_string());
        }
    }

    #[test]
    fn provider_name_and_configured() {
        let p = FakeProvider::new(true);
        assert_eq!(p.provider_name(), "fake");
        assert!(p.is_configured());
        assert!(!FakeProvider::new(false).is_configured());
    }

    #[test]
    fn create_session_returns_metadata() {
        let p = FakeProvider::new(true);
        let meta = p.create_session("abc").expect("session");
        assert_eq!(meta.session_name, "session-abc");
        assert_eq!(meta.bb_session_id, "sid-abc");
        assert_eq!(meta.cdp_url, "wss://example/cdp");
        assert_eq!(meta.features.get("stealth"), Some(&Value::Bool(true)));
    }

    #[test]
    fn create_session_errors_when_unconfigured() {
        let p = FakeProvider::new(false);
        assert!(p.create_session("abc").is_err());
    }

    #[test]
    fn to_value_shape_matches_python_dict() {
        let mut meta = SessionMetadata::new("s", "id", "url");
        meta.features
            .insert("f".to_string(), Value::String("on".to_string()));
        let v = meta.to_value();
        assert_eq!(v["session_name"], Value::String("s".into()));
        assert_eq!(v["bb_session_id"], Value::String("id".into()));
        assert_eq!(v["cdp_url"], Value::String("url".into()));
        assert_eq!(v["features"]["f"], Value::String("on".into()));
        // Exactly the four documented keys.
        assert_eq!(v.as_object().unwrap().len(), 4);
    }

    #[test]
    fn close_and_cleanup_record_session_ids() {
        let p = FakeProvider::new(true);
        assert!(p.close_session("sess1"));
        p.emergency_cleanup("sess1");
        assert_eq!(p.closed.borrow().as_slice(), ["sess1"]);
        assert_eq!(p.cleaned.borrow().as_slice(), ["sess1"]);

        let q = FakeProvider::new(false);
        assert!(!q.close_session("sess2"));
    }
}
