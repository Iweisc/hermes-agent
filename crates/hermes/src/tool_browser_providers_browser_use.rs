//! Browser Use cloud browser provider.
//!
//! Port of `tools/browser_providers/browser_use.py`.
//!
//! Implements the [`CloudBrowserProvider`](crate::tool_browser_providers_base::CloudBrowserProvider)
//! contract for the Browser Use (<https://browser-use.com>) cloud browser
//! backend. Sessions can be created either with a direct `BROWSER_USE_API_KEY`
//! credential or routed through a managed Nous tool gateway.
//!
//! # Dependencies not yet ported
//!
//! The Python module imports three helpers that do not have stable native Rust
//! equivalents wired up yet:
//!
//! * `resolve_managed_tool_gateway(vendor)` — returns a managed gateway config.
//! * `managed_nous_tools_enabled()` — whether managed Nous tools are enabled.
//! * `prefers_gateway(section)` — whether a config section prefers the gateway.
//!
//! These are injected into [`BrowserUseProvider`] as boxed closures
//! ([`GatewayResolver`], [`BoolHook`]) so the provider can be constructed and
//! tested in isolation. The default constructor wires in conservative stubs
//! (`prefers_gateway -> false`, `managed_nous_tools_enabled -> false`,
//! `resolve_managed_tool_gateway -> None`) so direct-API-key mode works exactly
//! like the Python implementation. When the gateway resolver is ported it can
//! be supplied via [`BrowserUseProvider::with_hooks`].

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// NOTE: the `CloudBrowserProvider` trait and `SessionMetadata` are defined in
// `hermes-core/src/tool_browser_providers_base.rs`, but that module is declared
// private (`mod`, not `pub mod`) in `hermes-core/src/lib.rs`, so it is not
// reachable from this (`hermes`) crate. To keep this module self-contained and
// compilable on its own — integration is wired up separately — the base types
// are mirrored locally below. They are intentionally identical to the ported
// base module so a future integration step can re-export instead.

/// Metadata describing a freshly created cloud browser session.
///
/// Mirror of `SessionMetadata` from `tool_browser_providers_base`. `bb_session_id`
/// is a legacy key name kept for backward compatibility with the rest of
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

    /// Serialize to a `serde_json::Value` object matching the Python dict shape.
    pub fn to_json(&self) -> Value {
        json!({
            "session_name": self.session_name,
            "bb_session_id": self.bb_session_id,
            "cdp_url": self.cdp_url,
            "features": self.features,
        })
    }
}

/// Interface for cloud browser backends. Mirror of the trait from
/// `tool_browser_providers_base`.
pub trait CloudBrowserProvider {
    /// Short, human-readable name shown in logs and diagnostics.
    fn provider_name(&self) -> String;

    /// Return `true` when all required env vars / credentials are present.
    /// Must be cheap — no network calls.
    fn is_configured(&self) -> bool;

    /// Create a cloud browser session and return session metadata.
    fn create_session(&self, task_id: &str) -> SessionMetadata;

    /// Release / terminate a cloud session by its provider session ID.
    fn close_session(&self, session_id: &str) -> bool;

    /// Best-effort session teardown during process exit.
    fn emergency_cleanup(&self, session_id: &str);
}

/// Base URL for the direct Browser Use v3 API.
pub const BASE_URL: &str = "https://api.browser-use.com/api/v3";

/// Default session timeout (minutes) used for gateway-backed sessions so that
/// billing authorization does not default to a long Browser Use timeout when
/// Hermes only needs a task-scoped ephemeral browser.
pub const DEFAULT_MANAGED_TIMEOUT_MINUTES: i64 = 5;

/// Default proxy country code applied to gateway-backed sessions.
pub const DEFAULT_MANAGED_PROXY_COUNTRY_CODE: &str = "us";

/// Resolved configuration describing how to talk to Browser Use.
///
/// Mirrors the dict returned by Python `_get_config_or_none`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserUseConfig {
    /// API key (direct credential or managed Nous user token).
    pub api_key: String,
    /// Base URL for requests (trailing slash stripped in managed mode).
    pub base_url: String,
    /// Whether the request is routed through the managed gateway.
    pub managed_mode: bool,
}

/// Minimal mirror of `ManagedToolGatewayConfig` from
/// `tools/managed_tool_gateway.py`.
///
/// A native `resolve_managed_tool_gateway` is not yet ported; this captures the
/// only two fields this module needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedGateway {
    /// Gateway origin (may carry a trailing slash; stripped when building URLs).
    pub gateway_origin: String,
    /// Nous user token used as the API key in managed mode.
    pub nous_user_token: String,
}

/// Closure type for the (not-yet-ported) `resolve_managed_tool_gateway`.
pub type GatewayResolver = Box<dyn Fn(&str) -> Option<ManagedGateway> + Send + Sync>;

/// Closure type for the boolean helper hooks
/// (`managed_nous_tools_enabled`, `prefers_gateway`).
pub type BoolHook = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// Reason a session-create response should *not* clear a pending idempotency
/// key. Returned from [`should_preserve_pending_create_key`].
///
/// Network responses are summarised into [`HttpStatus`] so this module's logic
/// can be unit-tested without performing real HTTP requests.
#[derive(Debug, Clone)]
pub struct HttpStatus {
    /// HTTP status code.
    pub status_code: u16,
    /// Raw response body text.
    pub body: String,
}

impl HttpStatus {
    /// Whether the status code is in the 2xx success range (mirrors
    /// `requests.Response.ok`).
    pub fn ok(&self) -> bool {
        (200..400).contains(&self.status_code)
    }

    /// Parse the body as JSON, returning `None` on failure (mirrors a
    /// guarded `response.json()`).
    pub fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }
}

/// Decide whether a pending idempotency key should be preserved across a failed
/// managed `create_session` attempt.
///
/// Port of `_should_preserve_pending_create_key`:
/// * 5xx responses preserve the key (transient server error — retry safely).
/// * a 409 whose error message contains `"already in progress"` preserves it.
/// * everything else does not.
pub fn should_preserve_pending_create_key(response: &HttpStatus) -> bool {
    if response.status_code >= 500 {
        return true;
    }

    if response.status_code != 409 {
        return false;
    }

    let payload = match response.json() {
        Some(p) => p,
        None => return false,
    };

    if !payload.is_object() {
        return false;
    }

    let error = match payload.get("error") {
        Some(e) if e.is_object() => e,
        _ => return false,
    };

    let message = error
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_lowercase();

    message.contains("already in progress")
}

/// Browser Use (<https://browser-use.com>) cloud browser backend.
///
/// Holds the per-task pending idempotency-key registry plus the injectable
/// gateway/helper hooks described in the module docs.
pub struct BrowserUseProvider {
    /// `task_id -> idempotency key` for in-flight managed session creates.
    pending_create_keys: Mutex<BTreeMap<String, String>>,
    /// `resolve_managed_tool_gateway` stand-in.
    resolve_gateway: GatewayResolver,
    /// `managed_nous_tools_enabled` stand-in.
    managed_nous_tools_enabled: BoolHook,
    /// `prefers_gateway` stand-in.
    prefers_gateway: BoolHook,
}

impl Default for BrowserUseProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserUseProvider {
    /// Construct a provider with conservative default hooks: no gateway, no
    /// managed Nous tools. This reproduces direct-`BROWSER_USE_API_KEY` mode.
    pub fn new() -> Self {
        Self::with_hooks(
            Box::new(|_vendor| None),
            Box::new(|_| false),
            Box::new(|_| false),
        )
    }

    /// Construct a provider wiring in the (eventually ported) gateway resolver
    /// and helper booleans.
    ///
    /// * `resolve_gateway` mirrors `resolve_managed_tool_gateway(vendor)`.
    /// * `managed_nous_tools_enabled` mirrors `managed_nous_tools_enabled()`
    ///   (the argument is the unused tool name, always `""`).
    /// * `prefers_gateway` mirrors `prefers_gateway(section)`.
    pub fn with_hooks(
        resolve_gateway: GatewayResolver,
        managed_nous_tools_enabled: BoolHook,
        prefers_gateway: BoolHook,
    ) -> Self {
        Self {
            pending_create_keys: Mutex::new(BTreeMap::new()),
            resolve_gateway,
            managed_nous_tools_enabled,
            prefers_gateway,
        }
    }

    // ------------------------------------------------------------------
    // Pending idempotency key registry
    // ------------------------------------------------------------------

    /// Get the existing pending idempotency key for `task_id`, or create a new
    /// one. Port of `_get_or_create_pending_create_key`.
    pub fn get_or_create_pending_create_key(&self, task_id: &str) -> String {
        let mut keys = self.pending_create_keys.lock().unwrap();
        if let Some(existing) = keys.get(task_id) {
            if !existing.is_empty() {
                return existing.clone();
            }
        }
        let created = format!("browser-use-session-create:{}", new_uuid_hex());
        keys.insert(task_id.to_string(), created.clone());
        created
    }

    /// Remove any pending idempotency key for `task_id`. Port of
    /// `_clear_pending_create_key`.
    pub fn clear_pending_create_key(&self, task_id: &str) {
        let mut keys = self.pending_create_keys.lock().unwrap();
        keys.remove(task_id);
    }

    // ------------------------------------------------------------------
    // Config resolution (direct API key OR managed Nous gateway)
    // ------------------------------------------------------------------

    /// Resolve config from the environment / gateway, or `None` when neither a
    /// direct credential nor a managed gateway is available. Port of
    /// `_get_config_or_none`.
    pub fn get_config_or_none(&self) -> Option<BrowserUseConfig> {
        let api_key = std::env::var("BROWSER_USE_API_KEY").ok().filter(|k| !k.is_empty());

        if let Some(api_key) = api_key {
            if !(self.prefers_gateway)("browser") {
                return Some(BrowserUseConfig {
                    api_key,
                    base_url: BASE_URL.to_string(),
                    managed_mode: false,
                });
            }
        }

        let managed = (self.resolve_gateway)("browser-use")?;

        Some(BrowserUseConfig {
            api_key: managed.nous_user_token,
            base_url: managed.gateway_origin.trim_end_matches('/').to_string(),
            managed_mode: true,
        })
    }

    /// Resolve config or return a descriptive error string. Port of
    /// `_get_config` (which raises `ValueError`).
    pub fn get_config(&self) -> Result<BrowserUseConfig, String> {
        match self.get_config_or_none() {
            Some(c) => Ok(c),
            None => {
                let message = if (self.managed_nous_tools_enabled)("") {
                    "Browser Use requires either a direct BROWSER_USE_API_KEY \
                     credential or a managed Browser Use gateway configuration."
                } else {
                    "Browser Use requires a direct BROWSER_USE_API_KEY credential."
                };
                Err(message.to_string())
            }
        }
    }

    // ------------------------------------------------------------------
    // Request construction
    // ------------------------------------------------------------------

    /// Build the request headers. Port of `_headers`.
    pub fn headers(&self, config: &BrowserUseConfig) -> BTreeMap<String, String> {
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert(
            "X-Browser-Use-API-Key".to_string(),
            config.api_key.clone(),
        );
        headers
    }

    /// Build the JSON payload for a `POST /browsers` create request. Port of the
    /// inline payload construction in `create_session`: managed sessions carry a
    /// short timeout + proxy country; direct sessions send an empty object.
    pub fn create_payload(managed_mode: bool) -> Value {
        if managed_mode {
            json!({
                "timeout": DEFAULT_MANAGED_TIMEOUT_MINUTES,
                "proxyCountryCode": DEFAULT_MANAGED_PROXY_COUNTRY_CODE,
            })
        } else {
            json!({})
        }
    }

    /// Build the session metadata returned to callers given a successful create
    /// response body, the (lowercased) response headers, and the task id.
    ///
    /// Factored out of `create_session` so the parsing logic can be unit-tested
    /// without a live HTTP round-trip.
    ///
    /// Returns `(SessionMetadata, external_call_id)`. Errors if the response
    /// body lacks the required `id` field (mirrors the Python `session_data["id"]`
    /// `KeyError`).
    pub fn build_session_metadata(
        session_data: &Value,
        response_headers: &BTreeMap<String, String>,
        task_id: &str,
        managed_mode: bool,
    ) -> Result<(SessionMetadata, Option<String>), String> {
        let id = session_data
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| session_data.get("id").map(value_to_string))
            .ok_or_else(|| "Browser Use create response missing 'id'".to_string())?;

        let session_name = format!("hermes_{}_{}", task_id, &new_uuid_hex()[..8]);

        let external_call_id = if managed_mode {
            response_headers
                .get("x-external-call-id")
                .cloned()
        } else {
            None
        };

        let cdp_url = session_data
            .get("cdpUrl")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                session_data
                    .get("connectUrl")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or("")
            .to_string();

        let mut features = BTreeMap::new();
        features.insert("browser_use".to_string(), Value::Bool(true));

        Ok((
            SessionMetadata::new(session_name, id, cdp_url, features),
            external_call_id,
        ))
    }
}

impl CloudBrowserProvider for BrowserUseProvider {
    fn provider_name(&self) -> String {
        "Browser Use".to_string()
    }

    fn is_configured(&self) -> bool {
        self.get_config_or_none().is_some()
    }

    /// Create a cloud browser session.
    ///
    /// Port of `create_session`. Performs a blocking `POST /browsers`. On any
    /// error (missing config, HTTP failure, malformed response) this panics —
    /// the Python equivalent raises `ValueError` / `RuntimeError`, and the trait
    /// signature is infallible. Prefer [`BrowserUseProvider::try_create_session`]
    /// for a non-panicking path.
    fn create_session(&self, task_id: &str) -> SessionMetadata {
        self.try_create_session(task_id)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    fn close_session(&self, session_id: &str) -> bool {
        let config = match self.get_config() {
            Ok(c) => c,
            Err(_) => {
                log::warn!(
                    "Cannot close Browser Use session {session_id} — missing credentials"
                );
                return false;
            }
        };

        let client = match build_client(Duration::from_secs(10)) {
            Ok(c) => c,
            Err(e) => {
                log::error!("Exception closing Browser Use session {session_id}: {e}");
                return false;
            }
        };

        let url = format!("{}/browsers/{}", config.base_url, session_id);
        let mut req = client.patch(&url).json(&json!({"action": "stop"}));
        for (k, v) in self.headers(&config) {
            req = req.header(k, v);
        }

        match req.send() {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if matches!(status, 200 | 201 | 204) {
                    log::debug!("Successfully closed Browser Use session {session_id}");
                    true
                } else {
                    let body = resp.text().unwrap_or_default();
                    let snippet: String = body.chars().take(200).collect();
                    log::warn!(
                        "Failed to close Browser Use session {session_id}: HTTP {status} - {snippet}"
                    );
                    false
                }
            }
            Err(e) => {
                log::error!("Exception closing Browser Use session {session_id}: {e}");
                false
            }
        }
    }

    fn emergency_cleanup(&self, session_id: &str) {
        let config = match self.get_config_or_none() {
            Some(c) => c,
            None => {
                log::warn!(
                    "Cannot emergency-cleanup Browser Use session {session_id} — missing credentials"
                );
                return;
            }
        };

        let client = match build_client(Duration::from_secs(5)) {
            Ok(c) => c,
            Err(e) => {
                log::debug!("Emergency cleanup failed for Browser Use session {session_id}: {e}");
                return;
            }
        };

        let url = format!("{}/browsers/{}", config.base_url, session_id);
        let mut req = client.patch(&url).json(&json!({"action": "stop"}));
        for (k, v) in self.headers(&config) {
            req = req.header(k, v);
        }

        if let Err(e) = req.send() {
            log::debug!("Emergency cleanup failed for Browser Use session {session_id}: {e}");
        }
    }
}

impl BrowserUseProvider {
    /// Non-panicking variant of [`create_session`](CloudBrowserProvider::create_session).
    ///
    /// Performs the blocking `POST /browsers`, manages the managed-mode pending
    /// idempotency key exactly like the Python implementation, and returns the
    /// resulting [`SessionMetadata`].
    pub fn try_create_session(&self, task_id: &str) -> Result<SessionMetadata, String> {
        let config = self.get_config()?;
        let managed_mode = config.managed_mode;

        let mut headers = self.headers(&config);
        if managed_mode {
            headers.insert(
                "X-Idempotency-Key".to_string(),
                self.get_or_create_pending_create_key(task_id),
            );
        }

        let payload = Self::create_payload(managed_mode);

        let client = build_client(Duration::from_secs(30))
            .map_err(|e| format!("Failed to create Browser Use session: {e}"))?;

        let url = format!("{}/browsers", config.base_url);
        let mut req = client.post(&url).json(&payload);
        for (k, v) in &headers {
            req = req.header(k, v);
        }

        let resp = req
            .send()
            .map_err(|e| format!("Failed to create Browser Use session: {e}"))?;

        let status_code = resp.status().as_u16();
        let resp_headers: BTreeMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_lowercase(),
                    v.to_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        let body = resp.text().unwrap_or_default();

        let status = HttpStatus {
            status_code,
            body: body.clone(),
        };

        if !status.ok() {
            if managed_mode && !should_preserve_pending_create_key(&status) {
                self.clear_pending_create_key(task_id);
            }
            return Err(format!(
                "Failed to create Browser Use session: {status_code} {body}"
            ));
        }

        let session_data: Value = serde_json::from_str(&body)
            .map_err(|e| format!("Failed to parse Browser Use create response: {e}"))?;

        if managed_mode {
            self.clear_pending_create_key(task_id);
        }

        let (meta, _external_call_id) =
            Self::build_session_metadata(&session_data, &resp_headers, task_id, managed_mode)?;

        log::info!("Created Browser Use session {}", meta.session_name);

        Ok(meta)
    }
}

/// Build a blocking reqwest client with the given timeout.
fn build_client(timeout: Duration) -> Result<reqwest::blocking::Client, reqwest::Error> {
    reqwest::blocking::Client::builder().timeout(timeout).build()
}

/// Generate a hex UUID (32 lowercase hex chars), matching `uuid.uuid4().hex`.
///
/// No `uuid` crate is in the allowed set, so this composes 128 bits of v4-style
/// randomness from the OS-time-seeded source available in std plus process
/// entropy. The exact value is never compared against Python output; only its
/// shape (32 hex chars) matters.
fn new_uuid_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Mix in a thread-local counter and address entropy for uniqueness.
    let counter = COUNTER.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    });
    let pid = std::process::id() as u128;
    let stack_addr = &nanos as *const _ as usize as u128;

    let mut state = nanos
        ^ (counter as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ pid.wrapping_shl(64)
        ^ stack_addr.wrapping_mul(0xBF58_476D_1CE4_E5B9);

    let mut out = String::with_capacity(32);
    for _ in 0..32 {
        // xorshift-style mixing producing one nibble at a time.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let nibble = (state & 0xF) as u8;
        out.push(char::from_digit(nibble as u32, 16).unwrap());
    }
    out
}

thread_local! {
    static COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Coerce a JSON value to a string the way Python's `str()` would for the `id`
/// field (used as a fallback when `id` is non-string).
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserve_key_on_5xx() {
        let r = HttpStatus {
            status_code: 503,
            body: String::new(),
        };
        assert!(should_preserve_pending_create_key(&r));
    }

    #[test]
    fn preserve_key_on_409_already_in_progress() {
        let r = HttpStatus {
            status_code: 409,
            body: r#"{"error": {"message": "A create is Already In Progress for task"}}"#
                .to_string(),
        };
        assert!(should_preserve_pending_create_key(&r));
    }

    #[test]
    fn no_preserve_on_409_other_message() {
        let r = HttpStatus {
            status_code: 409,
            body: r#"{"error": {"message": "quota exceeded"}}"#.to_string(),
        };
        assert!(!should_preserve_pending_create_key(&r));
    }

    #[test]
    fn no_preserve_on_409_bad_json() {
        let r = HttpStatus {
            status_code: 409,
            body: "not json".to_string(),
        };
        assert!(!should_preserve_pending_create_key(&r));
    }

    #[test]
    fn no_preserve_on_409_non_dict_error() {
        let r = HttpStatus {
            status_code: 409,
            body: r#"{"error": "scalar"}"#.to_string(),
        };
        assert!(!should_preserve_pending_create_key(&r));
    }

    #[test]
    fn no_preserve_on_4xx_non_409() {
        let r = HttpStatus {
            status_code: 404,
            body: String::new(),
        };
        assert!(!should_preserve_pending_create_key(&r));
    }

    #[test]
    fn pending_key_is_stable_per_task() {
        let p = BrowserUseProvider::new();
        let k1 = p.get_or_create_pending_create_key("task-1");
        let k2 = p.get_or_create_pending_create_key("task-1");
        assert_eq!(k1, k2);
        assert!(k1.starts_with("browser-use-session-create:"));
        let other = p.get_or_create_pending_create_key("task-2");
        assert_ne!(k1, other);
        p.clear_pending_create_key("task-1");
        let k3 = p.get_or_create_pending_create_key("task-1");
        assert_ne!(k1, k3);
    }

    #[test]
    fn direct_config_when_api_key_set() {
        let p = BrowserUseProvider::with_hooks(
            Box::new(|_| None),
            Box::new(|_| false),
            Box::new(|_| false),
        );
        let prev = std::env::var("BROWSER_USE_API_KEY").ok();
        unsafe {
            std::env::set_var("BROWSER_USE_API_KEY", "k-direct");
        }
        let cfg = p.get_config_or_none().expect("config");
        assert_eq!(cfg.api_key, "k-direct");
        assert_eq!(cfg.base_url, BASE_URL);
        assert!(!cfg.managed_mode);
        assert!(p.is_configured());
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BROWSER_USE_API_KEY", v),
                None => std::env::remove_var("BROWSER_USE_API_KEY"),
            }
        }
    }

    #[test]
    fn gateway_config_when_prefers_gateway() {
        let p = BrowserUseProvider::with_hooks(
            Box::new(|vendor| {
                assert_eq!(vendor, "browser-use");
                Some(ManagedGateway {
                    gateway_origin: "https://gw.example.com/".to_string(),
                    nous_user_token: "nous-token".to_string(),
                })
            }),
            Box::new(|_| true),
            Box::new(|section| section == "browser"),
        );
        let prev = std::env::var("BROWSER_USE_API_KEY").ok();
        unsafe {
            std::env::set_var("BROWSER_USE_API_KEY", "k-direct");
        }
        // prefers_gateway("browser") -> true, so we fall through to gateway.
        let cfg = p.get_config_or_none().expect("config");
        assert_eq!(cfg.api_key, "nous-token");
        assert_eq!(cfg.base_url, "https://gw.example.com"); // trailing slash stripped
        assert!(cfg.managed_mode);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BROWSER_USE_API_KEY", v),
                None => std::env::remove_var("BROWSER_USE_API_KEY"),
            }
        }
    }

    #[test]
    fn no_config_error_message_varies() {
        let plain = BrowserUseProvider::with_hooks(
            Box::new(|_| None),
            Box::new(|_| false),
            Box::new(|_| false),
        );
        let prev = std::env::var("BROWSER_USE_API_KEY").ok();
        unsafe {
            std::env::remove_var("BROWSER_USE_API_KEY");
        }
        let err = plain.get_config().unwrap_err();
        assert!(err.contains("requires a direct BROWSER_USE_API_KEY credential"));
        assert!(!err.contains("managed"));

        let managed = BrowserUseProvider::with_hooks(
            Box::new(|_| None),
            Box::new(|_| true),
            Box::new(|_| false),
        );
        let err2 = managed.get_config().unwrap_err();
        assert!(err2.contains("managed Browser Use gateway configuration"));
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("BROWSER_USE_API_KEY", v);
            }
        }
    }

    #[test]
    fn headers_include_api_key() {
        let p = BrowserUseProvider::new();
        let cfg = BrowserUseConfig {
            api_key: "abc".to_string(),
            base_url: BASE_URL.to_string(),
            managed_mode: false,
        };
        let h = p.headers(&cfg);
        assert_eq!(h.get("Content-Type").unwrap(), "application/json");
        assert_eq!(h.get("X-Browser-Use-API-Key").unwrap(), "abc");
    }

    #[test]
    fn create_payload_shapes() {
        let direct = BrowserUseProvider::create_payload(false);
        assert_eq!(direct, json!({}));

        let managed = BrowserUseProvider::create_payload(true);
        assert_eq!(managed["timeout"], json!(DEFAULT_MANAGED_TIMEOUT_MINUTES));
        assert_eq!(
            managed["proxyCountryCode"],
            json!(DEFAULT_MANAGED_PROXY_COUNTRY_CODE)
        );
    }

    #[test]
    fn build_metadata_prefers_cdp_url() {
        let data = json!({"id": "sess-1", "cdpUrl": "wss://cdp", "connectUrl": "wss://connect"});
        let (meta, ext) =
            BrowserUseProvider::build_session_metadata(&data, &BTreeMap::new(), "task-9", false)
                .unwrap();
        assert_eq!(meta.bb_session_id, "sess-1");
        assert_eq!(meta.cdp_url, "wss://cdp");
        assert!(meta.session_name.starts_with("hermes_task-9_"));
        assert_eq!(meta.features.get("browser_use"), Some(&Value::Bool(true)));
        assert!(ext.is_none());
    }

    #[test]
    fn build_metadata_falls_back_to_connect_url() {
        let data = json!({"id": "sess-2", "connectUrl": "wss://connect"});
        let (meta, _) =
            BrowserUseProvider::build_session_metadata(&data, &BTreeMap::new(), "t", false)
                .unwrap();
        assert_eq!(meta.cdp_url, "wss://connect");
    }

    #[test]
    fn build_metadata_empty_cdp_when_missing() {
        let data = json!({"id": "sess-3"});
        let (meta, _) =
            BrowserUseProvider::build_session_metadata(&data, &BTreeMap::new(), "t", false)
                .unwrap();
        assert_eq!(meta.cdp_url, "");
    }

    #[test]
    fn build_metadata_external_call_id_only_in_managed_mode() {
        let mut headers = BTreeMap::new();
        headers.insert("x-external-call-id".to_string(), "call-123".to_string());
        let data = json!({"id": "sess-4", "cdpUrl": "wss://cdp"});

        let (_, ext_managed) =
            BrowserUseProvider::build_session_metadata(&data, &headers, "t", true).unwrap();
        assert_eq!(ext_managed.as_deref(), Some("call-123"));

        let (_, ext_direct) =
            BrowserUseProvider::build_session_metadata(&data, &headers, "t", false).unwrap();
        assert!(ext_direct.is_none());
    }

    #[test]
    fn build_metadata_errors_without_id() {
        let data = json!({"cdpUrl": "wss://cdp"});
        let err = BrowserUseProvider::build_session_metadata(&data, &BTreeMap::new(), "t", false)
            .unwrap_err();
        assert!(err.contains("missing 'id'"));
    }

    #[test]
    fn provider_name_is_browser_use() {
        assert_eq!(BrowserUseProvider::new().provider_name(), "Browser Use");
    }

    #[test]
    fn uuid_hex_is_32_hex_chars() {
        let h = new_uuid_hex();
        assert_eq!(h.len(), 32);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        // Reasonable uniqueness expectation.
        assert_ne!(new_uuid_hex(), new_uuid_hex());
    }

    #[test]
    fn http_status_ok_and_json() {
        let s = HttpStatus {
            status_code: 200,
            body: r#"{"id":"x"}"#.to_string(),
        };
        assert!(s.ok());
        assert_eq!(s.json().unwrap()["id"], json!("x"));

        let bad = HttpStatus {
            status_code: 500,
            body: "oops".to_string(),
        };
        assert!(!bad.ok());
        assert!(bad.json().is_none());
    }
}
