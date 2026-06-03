//! Browserbase cloud browser provider (direct credentials only).
//!
//! Native Rust port of `tools/browser_providers/browserbase.py`.
//!
//! This provider requires direct `BROWSERBASE_API_KEY` and
//! `BROWSERBASE_PROJECT_ID` credentials. Managed Nous gateway support has been
//! removed — the Nous subscription now routes through Browser Use instead.
//!
//! Faithful behaviours preserved:
//!   * Env-var knobs: `BROWSERBASE_PROXIES` (default on, "false" disables),
//!     `BROWSERBASE_ADVANCED_STEALTH` (default off, "true" enables),
//!     `BROWSERBASE_KEEP_ALIVE` (default on, "false" disables),
//!     `BROWSERBASE_SESSION_TIMEOUT` (positive int ms), `BROWSERBASE_BASE_URL`
//!     (default `https://api.browserbase.com`, trailing slashes stripped).
//!   * Session-config JSON body construction matching the Python original.
//!   * 402 fallback flow: retry without keepAlive, then without proxies.
//!   * Feature-flag bookkeeping (`basic_stealth` always true).
//!   * Session name format `hermes_{task_id}_{8 hex chars}`.
//!   * close_session / emergency_cleanup REST request construction.
//!
//! The HTTP surface is abstracted behind the [`HttpClient`] trait so the
//! provider logic is testable without a live API. A `reqwest::blocking`-backed
//! implementation, [`ReqwestClient`], reproduces the real request construction
//! and response parsing.

use serde_json::{json, Value};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// Resolved Browserbase credentials + base URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserbaseConfig {
    pub api_key: String,
    pub project_id: String,
    pub base_url: String,
}

/// Read configuration from the process environment, returning `None` when
/// either required credential is absent (mirrors `_get_config_or_none`).
pub fn get_config_or_none() -> Option<BrowserbaseConfig> {
    let api_key = std::env::var("BROWSERBASE_API_KEY").ok().filter(|s| !s.is_empty());
    let project_id = std::env::var("BROWSERBASE_PROJECT_ID").ok().filter(|s| !s.is_empty());
    match (api_key, project_id) {
        (Some(api_key), Some(project_id)) => {
            let base_url = std::env::var("BROWSERBASE_BASE_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "https://api.browserbase.com".to_string());
            Some(BrowserbaseConfig {
                api_key,
                project_id,
                base_url: base_url.trim_end_matches('/').to_string(),
            })
        }
        _ => None,
    }
}

/// Resolve configuration, erroring when credentials are missing (mirrors
/// `_get_config`, which raises `ValueError`).
pub fn get_config() -> Result<BrowserbaseConfig, BrowserbaseError> {
    get_config_or_none().ok_or(BrowserbaseError::MissingCredentials)
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum BrowserbaseError {
    /// Credentials missing — corresponds to the Python `ValueError`.
    MissingCredentials,
    /// Session creation failed: HTTP status + response body text.
    CreateFailed { status: u16, body: String },
    /// A missing expected field in the JSON response (e.g. `id`/`connectUrl`).
    MissingField(&'static str),
    /// Transport-level failure.
    Transport(String),
}

impl std::fmt::Display for BrowserbaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrowserbaseError::MissingCredentials => write!(
                f,
                "Browserbase requires BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables."
            ),
            BrowserbaseError::CreateFailed { status, body } => {
                write!(f, "Failed to create Browserbase session: {} {}", status, body)
            }
            BrowserbaseError::MissingField(name) => {
                write!(f, "Browserbase response missing field: {}", name)
            }
            BrowserbaseError::Transport(msg) => write!(f, "Browserbase transport error: {}", msg),
        }
    }
}

impl std::error::Error for BrowserbaseError {}

// ---------------------------------------------------------------------------
// HTTP abstraction
// ---------------------------------------------------------------------------

/// A minimal HTTP response: status code + body text.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    /// `requests.Response.ok`: True for 2xx/3xx (status < 400).
    pub fn ok(&self) -> bool {
        self.status < 400
    }

    /// Parse the body as JSON (mirrors `response.json()`).
    pub fn json(&self) -> Result<Value, BrowserbaseError> {
        serde_json::from_str(&self.body)
            .map_err(|e| BrowserbaseError::Transport(format!("invalid JSON response: {}", e)))
    }
}

/// Abstraction over the POST requests this provider makes, so the session
/// logic can be unit-tested without network access.
pub trait HttpClient {
    /// Perform a POST with JSON body + headers, returning status & body text.
    fn post_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &Value,
        timeout_secs: u64,
    ) -> Result<HttpResponse, BrowserbaseError>;
}

// ---------------------------------------------------------------------------
// reqwest-backed implementation
// ---------------------------------------------------------------------------

/// `reqwest::blocking`-backed [`HttpClient`].
#[derive(Debug, Default, Clone)]
pub struct ReqwestClient;

impl HttpClient for ReqwestClient {
    fn post_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &Value,
        timeout_secs: u64,
    ) -> Result<HttpResponse, BrowserbaseError> {
        let client = reqwest::blocking::Client::new();
        let mut req = client
            .post(url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .json(body);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = req
            .send()
            .map_err(|e| BrowserbaseError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .map_err(|e| BrowserbaseError::Transport(e.to_string()))?;
        Ok(HttpResponse { status, body: text })
    }
}

// ---------------------------------------------------------------------------
// feature flags
// ---------------------------------------------------------------------------

/// The five feature flags reported back from `create_session`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Features {
    pub basic_stealth: bool,
    pub proxies: bool,
    pub advanced_stealth: bool,
    pub keep_alive: bool,
    pub custom_timeout: bool,
}

impl Default for Features {
    fn default() -> Self {
        // Mirrors the initial `features_enabled` dict (basic_stealth True).
        Features {
            basic_stealth: true,
            proxies: false,
            advanced_stealth: false,
            keep_alive: false,
            custom_timeout: false,
        }
    }
}

impl Features {
    /// Ordered (insertion-order in Python) list of enabled feature names.
    pub fn enabled_names(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.basic_stealth {
            out.push("basic_stealth");
        }
        if self.proxies {
            out.push("proxies");
        }
        if self.advanced_stealth {
            out.push("advanced_stealth");
        }
        if self.keep_alive {
            out.push("keep_alive");
        }
        if self.custom_timeout {
            out.push("custom_timeout");
        }
        out
    }

    /// Comma-joined feature string used in the log line.
    pub fn feature_string(&self) -> String {
        self.enabled_names().join(", ")
    }

    /// Convert to a JSON object for inclusion in `create_session`'s return value.
    pub fn to_json(&self) -> Value {
        json!({
            "basic_stealth": self.basic_stealth,
            "proxies": self.proxies,
            "advanced_stealth": self.advanced_stealth,
            "keep_alive": self.keep_alive,
            "custom_timeout": self.custom_timeout,
        })
    }
}

// ---------------------------------------------------------------------------
// env knobs
// ---------------------------------------------------------------------------

/// The four env-var knobs parsed at the top of `create_session`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionKnobs {
    pub enable_proxies: bool,
    pub enable_advanced_stealth: bool,
    pub enable_keep_alive: bool,
    /// Raw `BROWSERBASE_SESSION_TIMEOUT` string, if set & non-empty.
    pub custom_timeout_ms: Option<String>,
}

impl SessionKnobs {
    /// Read knobs from the process environment.
    pub fn from_env() -> Self {
        let getenv = |k: &str| std::env::var(k).ok();
        SessionKnobs {
            // default "true"; only the literal lowercase "false" disables.
            enable_proxies: getenv("BROWSERBASE_PROXIES")
                .unwrap_or_else(|| "true".to_string())
                .to_lowercase()
                != "false",
            // default "false"; only "true" enables.
            enable_advanced_stealth: getenv("BROWSERBASE_ADVANCED_STEALTH")
                .unwrap_or_else(|| "false".to_string())
                .to_lowercase()
                == "true",
            enable_keep_alive: getenv("BROWSERBASE_KEEP_ALIVE")
                .unwrap_or_else(|| "true".to_string())
                .to_lowercase()
                != "false",
            // Python: `os.environ.get(...)` -> None if unset; empty string is
            // falsy in the `if custom_timeout_ms:` guards, so treat "" as None.
            custom_timeout_ms: getenv("BROWSERBASE_SESSION_TIMEOUT").filter(|s| !s.is_empty()),
        }
    }
}

// ---------------------------------------------------------------------------
// session config builder
// ---------------------------------------------------------------------------

/// Build the session-config JSON body for `POST /v1/sessions`.
///
/// Uses a `BTreeMap` purely as an intermediate; the resulting `Value` object
/// carries the keys Browserbase expects: `projectId`, optional `keepAlive`,
/// `timeout`, `proxies`, `browserSettings`.
///
/// Returns the body plus a flag indicating whether a valid `timeout` key was
/// actually inserted (needed for the `custom_timeout` feature bookkeeping).
pub fn build_session_config(
    config: &BrowserbaseConfig,
    knobs: &SessionKnobs,
) -> (Value, bool) {
    let mut map: BTreeMap<&str, Value> = BTreeMap::new();
    map.insert("projectId", Value::String(config.project_id.clone()));

    if knobs.enable_keep_alive {
        map.insert("keepAlive", Value::Bool(true));
    }

    let mut timeout_set = false;
    if let Some(raw) = &knobs.custom_timeout_ms {
        match raw.parse::<i64>() {
            Ok(timeout_val) if timeout_val > 0 => {
                map.insert("timeout", json!(timeout_val));
                timeout_set = true;
            }
            Ok(_) => {
                // non-positive: parsed fine but not applied (matches Python).
            }
            Err(_) => {
                log::warn!("Invalid BROWSERBASE_SESSION_TIMEOUT value: {}", raw);
            }
        }
    }

    if knobs.enable_proxies {
        map.insert("proxies", Value::Bool(true));
    }

    if knobs.enable_advanced_stealth {
        map.insert("browserSettings", json!({"advancedStealth": true}));
    }

    let obj: serde_json::Map<String, Value> =
        map.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    (Value::Object(obj), timeout_set)
}

// ---------------------------------------------------------------------------
// session name
// ---------------------------------------------------------------------------

/// Generate a session name `hermes_{task_id}_{8 lowercase hex chars}`.
///
/// Mirrors `f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"`.
pub fn make_session_name(task_id: &str) -> String {
    let suffix = random_hex8();
    format!("hermes_{}_{}", task_id, suffix)
}

/// 8 lowercase hex characters from a random UUID's first 4 bytes.
fn random_hex8() -> String {
    // 4 random bytes -> 8 hex chars, matching uuid4().hex[:8].
    let bytes: [u8; 4] = {
        // Use a quick mix of time + a stack address for entropy without an
        // extra crate. This only needs to be unique-ish per session.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut seed = nanos as u64
            ^ ((&nanos as *const u128 as u64).rotate_left(17))
            ^ (std::process::id() as u64).rotate_left(33);
        // xorshift64
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let v = next();
        [(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, v as u8]
    };
    let mut s = String::with_capacity(8);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ---------------------------------------------------------------------------
// session result
// ---------------------------------------------------------------------------

/// Result of a successful `create_session` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionResult {
    pub session_name: String,
    pub bb_session_id: String,
    pub cdp_url: String,
    pub features: Features,
}

impl SessionResult {
    /// JSON-object form matching the Python dict return value.
    pub fn to_json(&self) -> Value {
        json!({
            "session_name": self.session_name,
            "bb_session_id": self.bb_session_id,
            "cdp_url": self.cdp_url,
            "features": self.features.to_json(),
        })
    }
}

// ---------------------------------------------------------------------------
// provider
// ---------------------------------------------------------------------------

/// Browserbase cloud browser backend.
pub struct BrowserbaseProvider<C: HttpClient = ReqwestClient> {
    client: C,
}

impl Default for BrowserbaseProvider<ReqwestClient> {
    fn default() -> Self {
        BrowserbaseProvider {
            client: ReqwestClient,
        }
    }
}

impl BrowserbaseProvider<ReqwestClient> {
    /// Construct a provider using the default `reqwest::blocking` client.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: HttpClient> BrowserbaseProvider<C> {
    /// Construct with a custom HTTP client (used in tests).
    pub fn with_client(client: C) -> Self {
        BrowserbaseProvider { client }
    }

    pub fn provider_name(&self) -> &'static str {
        "Browserbase"
    }

    /// `is_configured`: credentials present in the environment.
    pub fn is_configured(&self) -> bool {
        get_config_or_none().is_some()
    }

    /// Create a Browserbase session, reproducing the 402 fallback flow.
    pub fn create_session(&self, task_id: &str) -> Result<SessionResult, BrowserbaseError> {
        let config = get_config()?;
        let knobs = SessionKnobs::from_env();

        let (mut session_config, timeout_set) = build_session_config(&config, &knobs);

        let headers: [(&str, &str); 2] = [
            ("Content-Type", "application/json"),
            ("X-BB-API-Key", config.api_key.as_str()),
        ];
        let url = format!("{}/v1/sessions", config.base_url);

        let mut response = self.client.post_json(&url, &headers, &session_config, 30)?;

        let mut proxies_fallback = false;
        let mut keepalive_fallback = false;

        // Handle 402 — paid features unavailable.
        if response.status == 402 {
            if knobs.enable_keep_alive {
                keepalive_fallback = true;
                log::warn!(
                    "keepAlive may require paid plan (402), retrying without it. \
                     Sessions may timeout during long operations."
                );
                if let Value::Object(m) = &mut session_config {
                    m.remove("keepAlive");
                }
                response = self.client.post_json(&url, &headers, &session_config, 30)?;
            }

            if response.status == 402 && knobs.enable_proxies {
                proxies_fallback = true;
                log::warn!(
                    "Proxies unavailable (402), retrying without proxies. \
                     Bot detection may be less effective."
                );
                if let Value::Object(m) = &mut session_config {
                    m.remove("proxies");
                }
                response = self.client.post_json(&url, &headers, &session_config, 30)?;
            }
        }

        if !response.ok() {
            return Err(BrowserbaseError::CreateFailed {
                status: response.status,
                body: response.body.clone(),
            });
        }

        let session_data = response.json()?;
        let session_name = make_session_name(task_id);

        let mut features = Features::default();
        if knobs.enable_proxies && !proxies_fallback {
            features.proxies = true;
        }
        if knobs.enable_advanced_stealth {
            features.advanced_stealth = true;
        }
        if knobs.enable_keep_alive && !keepalive_fallback {
            features.keep_alive = true;
        }
        // Python: `if custom_timeout_ms and "timeout" in session_config`. Note
        // `timeout` is never popped, so its presence == it was inserted.
        if knobs.custom_timeout_ms.is_some() && timeout_set {
            features.custom_timeout = true;
        }

        log::info!(
            "Created Browserbase session {} with features: {}",
            session_name,
            features.feature_string()
        );

        let bb_session_id = session_data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or(BrowserbaseError::MissingField("id"))?
            .to_string();
        let cdp_url = session_data
            .get("connectUrl")
            .and_then(|v| v.as_str())
            .ok_or(BrowserbaseError::MissingField("connectUrl"))?
            .to_string();

        Ok(SessionResult {
            session_name,
            bb_session_id,
            cdp_url,
            features,
        })
    }

    /// Close a session via `REQUEST_RELEASE`. Returns `true` on success.
    ///
    /// Mirrors `close_session`: missing credentials -> warn + `false`; any
    /// transport exception -> error log + `false`.
    pub fn close_session(&self, session_id: &str) -> bool {
        let config = match get_config() {
            Ok(c) => c,
            Err(_) => {
                log::warn!(
                    "Cannot close Browserbase session {} — missing credentials",
                    session_id
                );
                return false;
            }
        };

        let url = format!("{}/v1/sessions/{}", config.base_url, session_id);
        let headers: [(&str, &str); 2] = [
            ("X-BB-API-Key", config.api_key.as_str()),
            ("Content-Type", "application/json"),
        ];
        let body = json!({
            "projectId": config.project_id,
            "status": "REQUEST_RELEASE",
        });

        match self.client.post_json(&url, &headers, &body, 10) {
            Ok(resp) => {
                if matches!(resp.status, 200 | 201 | 204) {
                    log::debug!("Successfully closed Browserbase session {}", session_id);
                    true
                } else {
                    let snippet: String = resp.body.chars().take(200).collect();
                    log::warn!(
                        "Failed to close session {}: HTTP {} - {}",
                        session_id,
                        resp.status,
                        snippet
                    );
                    false
                }
            }
            Err(e) => {
                log::error!("Exception closing Browserbase session {}: {}", session_id, e);
                false
            }
        }
    }

    /// Best-effort session release used on shutdown. Never errors.
    ///
    /// Mirrors `emergency_cleanup`: missing credentials -> warn + return; any
    /// failure -> debug log + return.
    pub fn emergency_cleanup(&self, session_id: &str) {
        let config = match get_config_or_none() {
            Some(c) => c,
            None => {
                log::warn!(
                    "Cannot emergency-cleanup Browserbase session {} — missing credentials",
                    session_id
                );
                return;
            }
        };

        let url = format!("{}/v1/sessions/{}", config.base_url, session_id);
        let headers: [(&str, &str); 2] = [
            ("X-BB-API-Key", config.api_key.as_str()),
            ("Content-Type", "application/json"),
        ];
        let body = json!({
            "projectId": config.project_id,
            "status": "REQUEST_RELEASE",
        });

        if let Err(e) = self.client.post_json(&url, &headers, &body, 5) {
            log::debug!(
                "Emergency cleanup failed for Browserbase session {}: {}",
                session_id,
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::{Mutex, MutexGuard};

    // Env access serialization: env-var-mutating tests must hold this lock.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        _g: MutexGuard<'static, ()>,
        keys: Vec<&'static str>,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: Vec<&'static str>) -> Self {
            let g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let saved = keys
                .iter()
                .map(|k| (*k, std::env::var(*k).ok()))
                .collect::<Vec<_>>();
            // start clean
            for k in &keys {
                unsafe { std::env::remove_var(k) };
            }
            EnvGuard { _g: g, keys, saved }
        }
        fn set(&self, k: &str, v: &str) {
            unsafe { std::env::set_var(k, v) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => unsafe { std::env::set_var(k, val) },
                    None => unsafe { std::env::remove_var(k) },
                }
            }
            let _ = &self.keys;
        }
    }

    const ALL_ENV: &[&str] = &[
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSERBASE_BASE_URL",
        "BROWSERBASE_PROXIES",
        "BROWSERBASE_ADVANCED_STEALTH",
        "BROWSERBASE_KEEP_ALIVE",
        "BROWSERBASE_SESSION_TIMEOUT",
    ];

    /// Mock client: scripted responses + recorded requests.
    struct MockClient {
        responses: RefCell<Vec<HttpResponse>>,
        calls: RefCell<Vec<(String, Value, u64)>>,
    }

    impl MockClient {
        fn new(responses: Vec<HttpResponse>) -> Self {
            MockClient {
                responses: RefCell::new(responses),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl HttpClient for MockClient {
        fn post_json(
            &self,
            url: &str,
            _headers: &[(&str, &str)],
            body: &Value,
            timeout_secs: u64,
        ) -> Result<HttpResponse, BrowserbaseError> {
            self.calls
                .borrow_mut()
                .push((url.to_string(), body.clone(), timeout_secs));
            let mut r = self.responses.borrow_mut();
            if r.is_empty() {
                Ok(HttpResponse {
                    status: 200,
                    body: "{}".to_string(),
                })
            } else {
                Ok(r.remove(0))
            }
        }
    }

    fn ok_session() -> HttpResponse {
        HttpResponse {
            status: 200,
            body: r#"{"id":"sess-123","connectUrl":"wss://cdp.example/abc"}"#.to_string(),
        }
    }

    #[test]
    fn config_none_without_creds() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        assert!(get_config_or_none().is_none());
        assert!(matches!(get_config(), Err(BrowserbaseError::MissingCredentials)));
    }

    #[test]
    fn config_default_base_url_and_trim() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "p");
        let c = get_config_or_none().unwrap();
        assert_eq!(c.base_url, "https://api.browserbase.com");

        e.set("BROWSERBASE_BASE_URL", "https://custom.example///");
        let c2 = get_config_or_none().unwrap();
        assert_eq!(c2.base_url, "https://custom.example");
    }

    #[test]
    fn knobs_defaults() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let k = SessionKnobs::from_env();
        assert!(k.enable_proxies);
        assert!(!k.enable_advanced_stealth);
        assert!(k.enable_keep_alive);
        assert!(k.custom_timeout_ms.is_none());
    }

    #[test]
    fn knobs_overrides() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_PROXIES", "FALSE");
        e.set("BROWSERBASE_ADVANCED_STEALTH", "True");
        e.set("BROWSERBASE_KEEP_ALIVE", "false");
        e.set("BROWSERBASE_SESSION_TIMEOUT", "5000");
        let k = SessionKnobs::from_env();
        assert!(!k.enable_proxies);
        assert!(k.enable_advanced_stealth);
        assert!(!k.enable_keep_alive);
        assert_eq!(k.custom_timeout_ms.as_deref(), Some("5000"));
    }

    #[test]
    fn build_config_full() {
        let cfg = BrowserbaseConfig {
            api_key: "k".into(),
            project_id: "proj".into(),
            base_url: "https://api.browserbase.com".into(),
        };
        let knobs = SessionKnobs {
            enable_proxies: true,
            enable_advanced_stealth: true,
            enable_keep_alive: true,
            custom_timeout_ms: Some("7000".into()),
        };
        let (body, timeout_set) = build_session_config(&cfg, &knobs);
        assert!(timeout_set);
        assert_eq!(body["projectId"], json!("proj"));
        assert_eq!(body["keepAlive"], json!(true));
        assert_eq!(body["timeout"], json!(7000));
        assert_eq!(body["proxies"], json!(true));
        assert_eq!(body["browserSettings"], json!({"advancedStealth": true}));
    }

    #[test]
    fn build_config_invalid_timeout_omitted() {
        let cfg = BrowserbaseConfig {
            api_key: "k".into(),
            project_id: "proj".into(),
            base_url: "u".into(),
        };
        // non-numeric
        let knobs = SessionKnobs {
            enable_proxies: false,
            enable_advanced_stealth: false,
            enable_keep_alive: false,
            custom_timeout_ms: Some("abc".into()),
        };
        let (body, timeout_set) = build_session_config(&cfg, &knobs);
        assert!(!timeout_set);
        assert!(body.get("timeout").is_none());
        assert!(body.get("keepAlive").is_none());
        assert!(body.get("proxies").is_none());

        // zero/negative: parses but not applied
        let knobs2 = SessionKnobs {
            custom_timeout_ms: Some("0".into()),
            ..knobs.clone()
        };
        let (body2, ts2) = build_session_config(&cfg, &knobs2);
        assert!(!ts2);
        assert!(body2.get("timeout").is_none());
    }

    #[test]
    fn features_string_order() {
        let f = Features {
            basic_stealth: true,
            proxies: true,
            advanced_stealth: false,
            keep_alive: true,
            custom_timeout: false,
        };
        assert_eq!(f.feature_string(), "basic_stealth, proxies, keep_alive");
    }

    #[test]
    fn session_name_format() {
        let name = make_session_name("task42");
        assert!(name.starts_with("hermes_task42_"));
        let hex = name.rsplit('_').next().unwrap();
        assert_eq!(hex.len(), 8);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn create_session_success() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![ok_session()]));
        let res = provider.create_session("t1").unwrap();
        assert_eq!(res.bb_session_id, "sess-123");
        assert_eq!(res.cdp_url, "wss://cdp.example/abc");
        // defaults: proxies on, keep_alive on, advanced off, timeout off
        assert!(res.features.proxies);
        assert!(res.features.keep_alive);
        assert!(res.features.basic_stealth);
        assert!(!res.features.advanced_stealth);
        assert!(!res.features.custom_timeout);
    }

    #[test]
    fn create_session_402_drops_keepalive_then_proxies() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        // first 402, second (no keepAlive) 402, third (no proxies) ok
        let resp402 = HttpResponse {
            status: 402,
            body: "payment required".into(),
        };
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![
            resp402.clone(),
            resp402,
            ok_session(),
        ]));
        let res = provider.create_session("t").unwrap();
        // both fallbacks triggered -> proxies + keep_alive disabled
        assert!(!res.features.proxies);
        assert!(!res.features.keep_alive);
    }

    #[test]
    fn create_session_402_keepalive_only_recovers() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        let resp402 = HttpResponse {
            status: 402,
            body: "x".into(),
        };
        // first 402, second (no keepAlive) ok -> proxies stay enabled
        let provider =
            BrowserbaseProvider::with_client(MockClient::new(vec![resp402, ok_session()]));
        let res = provider.create_session("t").unwrap();
        assert!(res.features.proxies);
        assert!(!res.features.keep_alive);
    }

    #[test]
    fn create_session_failure_propagates_body() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 500,
            body: "boom".into(),
        }]));
        match provider.create_session("t") {
            Err(BrowserbaseError::CreateFailed { status, body }) => {
                assert_eq!(status, 500);
                assert_eq!(body, "boom");
            }
            other => panic!("expected CreateFailed, got {:?}", other),
        }
    }

    #[test]
    fn close_session_success_codes() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        for code in [200u16, 201, 204] {
            let provider = BrowserbaseProvider::with_client(MockClient::new(vec![HttpResponse {
                status: code,
                body: "".into(),
            }]));
            assert!(provider.close_session("sid"), "code {}", code);
        }
        // failure code
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 404,
            body: "nope".into(),
        }]));
        assert!(!provider.close_session("sid"));
    }

    #[test]
    fn close_session_no_creds() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![]));
        assert!(!provider.close_session("sid"));
        // no HTTP call made
        assert!(provider.client.calls.borrow().is_empty());
    }

    #[test]
    fn close_session_url_and_body() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        e.set("BROWSERBASE_BASE_URL", "https://api.browserbase.com/");
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 204,
            body: "".into(),
        }]));
        assert!(provider.close_session("sid-9"));
        let calls = provider.client.calls.borrow();
        assert_eq!(calls[0].0, "https://api.browserbase.com/v1/sessions/sid-9");
        assert_eq!(calls[0].1["status"], json!("REQUEST_RELEASE"));
        assert_eq!(calls[0].1["projectId"], json!("proj"));
        assert_eq!(calls[0].2, 10);
    }

    #[test]
    fn emergency_cleanup_no_creds_noop() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![]));
        provider.emergency_cleanup("sid");
        assert!(provider.client.calls.borrow().is_empty());
    }

    #[test]
    fn emergency_cleanup_makes_request() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("BROWSERBASE_API_KEY", "k");
        e.set("BROWSERBASE_PROJECT_ID", "proj");
        let provider = BrowserbaseProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 200,
            body: "".into(),
        }]));
        provider.emergency_cleanup("sid");
        let calls = provider.client.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, 5);
    }
}
