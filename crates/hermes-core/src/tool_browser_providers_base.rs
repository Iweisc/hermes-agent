//! Abstract base interface for cloud browser providers.
//!
//! Port of `tools/browser_providers/base.py`.
//!
//! Defines the [`CloudBrowserProvider`] trait — the Rust equivalent of the
//! Python `CloudBrowserProvider` ABC. Concrete implementations (Browserbase,
//! Steel, etc.) live in sibling modules and are registered in the browser
//! tool's provider registry. The user selects a provider via `hermes setup` /
//! `hermes tools`; the choice is persisted as
//! `config["browser"]["cloud_provider"]`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Metadata describing a freshly created cloud browser session.
///
/// Mirrors the dict returned by the Python `create_session`, which is
/// documented to contain at least the four keys below. `bb_session_id` is a
/// legacy key name kept for backward compatibility with the rest of
/// `browser_tool.py` — it holds the provider's session ID regardless of which
/// provider is in use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique name for `agent-browser --session`.
    pub session_name: String,
    /// Provider session ID (for close/cleanup). Legacy key name.
    pub bb_session_id: String,
    /// CDP websocket URL.
    pub cdp_url: String,
    /// Feature flags that were enabled.
    pub features: BTreeMap<String, Value>,
}

impl SessionMetadata {
    /// Construct a new session-metadata record.
    pub fn new(
        session_name: impl Into<String>,
        bb_session_id: impl Into<String>,
        cdp_url: impl Into<String>,
        features: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            session_name: session_name.into(),
            bb_session_id: bb_session_id.into(),
            cdp_url: cdp_url.into(),
            features,
        }
    }

    /// Serialize to a `serde_json::Value` object, matching the shape of the
    /// dict returned by the Python implementation.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "session_name": self.session_name,
            "bb_session_id": self.bb_session_id,
            "cdp_url": self.cdp_url,
            "features": self.features,
        })
    }
}

/// Interface for cloud browser backends (Browserbase, Steel, etc.).
///
/// Implementations live in sibling modules and are registered in the browser
/// tool's provider registry. The user selects a provider via `hermes setup` /
/// `hermes tools`; the choice is persisted as
/// `config["browser"]["cloud_provider"]`.
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
    /// The returned [`SessionMetadata`] carries at least `session_name`,
    /// `bb_session_id`, `cdp_url`, and `features` — matching the dict contract
    /// documented in the Python base class.
    fn create_session(&self, task_id: &str) -> SessionMetadata;

    /// Release / terminate a cloud session by its provider session ID.
    ///
    /// Returns `true` on success, `false` on failure. Should not panic.
    fn close_session(&self, session_id: &str) -> bool;

    /// Best-effort session teardown during process exit.
    ///
    /// Called from atexit / signal handlers. Must tolerate missing
    /// credentials, network errors, etc. — log and move on.
    fn emergency_cleanup(&self, session_id: &str);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal stub implementation used to exercise the trait contract.
    struct StubProvider {
        configured: bool,
        cleaned: std::cell::RefCell<Vec<String>>,
    }

    impl CloudBrowserProvider for StubProvider {
        fn provider_name(&self) -> String {
            "stub".to_string()
        }

        fn is_configured(&self) -> bool {
            self.configured
        }

        fn create_session(&self, task_id: &str) -> SessionMetadata {
            let mut features = BTreeMap::new();
            features.insert("stealth".to_string(), Value::Bool(true));
            SessionMetadata::new(
                format!("session-{task_id}"),
                format!("prov-{task_id}"),
                "wss://example.test/cdp".to_string(),
                features,
            )
        }

        fn close_session(&self, session_id: &str) -> bool {
            !session_id.is_empty()
        }

        fn emergency_cleanup(&self, session_id: &str) {
            self.cleaned.borrow_mut().push(session_id.to_string());
        }
    }

    fn stub(configured: bool) -> StubProvider {
        StubProvider {
            configured,
            cleaned: std::cell::RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn provider_name_and_configured() {
        let p = stub(true);
        assert_eq!(p.provider_name(), "stub");
        assert!(p.is_configured());
        assert!(!stub(false).is_configured());
    }

    #[test]
    fn create_session_has_required_keys() {
        let p = stub(true);
        let meta = p.create_session("abc");
        assert_eq!(meta.session_name, "session-abc");
        assert_eq!(meta.bb_session_id, "prov-abc");
        assert_eq!(meta.cdp_url, "wss://example.test/cdp");
        assert_eq!(meta.features.get("stealth"), Some(&Value::Bool(true)));
    }

    #[test]
    fn create_session_json_shape() {
        let p = stub(true);
        let json = p.create_session("xyz").to_json();
        assert!(json.get("session_name").is_some());
        assert!(json.get("bb_session_id").is_some());
        assert!(json.get("cdp_url").is_some());
        assert!(json.get("features").is_some());
        assert_eq!(json["bb_session_id"], Value::String("prov-xyz".into()));
    }

    #[test]
    fn close_session_truthiness() {
        let p = stub(true);
        assert!(p.close_session("prov-1"));
        assert!(!p.close_session(""));
    }

    #[test]
    fn emergency_cleanup_records() {
        let p = stub(true);
        p.emergency_cleanup("prov-1");
        p.emergency_cleanup("prov-2");
        assert_eq!(*p.cleaned.borrow(), vec!["prov-1", "prov-2"]);
    }

    #[test]
    fn metadata_roundtrips_through_serde() {
        let mut features = BTreeMap::new();
        features.insert("proxy".to_string(), Value::String("us".into()));
        let meta = SessionMetadata::new("s", "id", "wss://x", features);
        let s = serde_json::to_string(&meta).unwrap();
        let back: SessionMetadata = serde_json::from_str(&s).unwrap();
        assert_eq!(meta, back);
    }
}
