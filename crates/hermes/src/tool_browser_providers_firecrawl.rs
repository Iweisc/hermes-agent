//! Firecrawl cloud browser provider.
//!
//! Native Rust port of `tools/browser_providers/firecrawl.py`.
//!
//! Firecrawl (<https://firecrawl.dev>) cloud browser backend. Sessions are
//! created/destroyed via the `/v2/browser` REST endpoints.
//!
//! Faithful behaviours preserved:
//!   * `is_configured`: True iff `FIRECRAWL_API_KEY` env var is set & non-empty.
//!   * `_api_url`: `FIRECRAWL_API_URL` env override, default
//!     `https://api.firecrawl.dev`.
//!   * `_headers`: requires `FIRECRAWL_API_KEY`, otherwise raises a
//!     `ValueError`-equivalent ([`FirecrawlError::MissingCredentials`]); sets
//!     `Content-Type: application/json` + `Authorization: Bearer <key>`.
//!   * `create_session`: `FIRECRAWL_BROWSER_TTL` env (default `300`, parsed as
//!     int) → POST `/v2/browser` with `{"ttl": ttl}`, timeout 30s. Non-ok
//!     response → error carrying status + body. Returns `session_name`
//!     (`hermes_{task_id}_{8 hex}`), `bb_session_id` (`data["id"]`), `cdp_url`
//!     (`data["cdpUrl"]`), and `features` (`{"firecrawl": true}`).
//!   * `close_session`: DELETE `/v2/browser/{id}`, timeout 10s. Returns `true`
//!     for status in {200,201,204}; logs warn + `false` otherwise; transport
//!     error → error log + `false`.
//!   * `emergency_cleanup`: best-effort DELETE `/v2/browser/{id}`, timeout 5s.
//!     Missing credentials → warn; other errors → debug log. Never raises.
//!
//! The HTTP surface is abstracted behind the [`HttpClient`] trait so the
//! provider logic is testable without a live API. A `reqwest::blocking`-backed
//! implementation, [`ReqwestClient`], reproduces the real request construction
//! and response parsing.

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

/// Default API base URL (`_BASE_URL` in the Python original).
pub const BASE_URL: &str = "https://api.firecrawl.dev";

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum FirecrawlError {
    /// `FIRECRAWL_API_KEY` missing — corresponds to the Python `ValueError`.
    MissingCredentials,
    /// Session creation failed: HTTP status + response body text.
    CreateFailed { status: u16, body: String },
    /// A missing expected field in the JSON response (e.g. `id`/`cdpUrl`).
    MissingField(&'static str),
    /// Transport-level failure (mirrors a `requests` exception / bad JSON).
    Transport(String),
}

impl std::fmt::Display for FirecrawlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FirecrawlError::MissingCredentials => write!(
                f,
                "FIRECRAWL_API_KEY environment variable is required. \
                 Get your key at https://firecrawl.dev"
            ),
            FirecrawlError::CreateFailed { status, body } => {
                write!(
                    f,
                    "Failed to create Firecrawl browser session: {} {}",
                    status, body
                )
            }
            FirecrawlError::MissingField(name) => {
                write!(f, "Firecrawl response missing field: {}", name)
            }
            FirecrawlError::Transport(msg) => write!(f, "Firecrawl transport error: {}", msg),
        }
    }
}

impl std::error::Error for FirecrawlError {}

// ---------------------------------------------------------------------------
// configuration helpers
// ---------------------------------------------------------------------------

/// Resolve the API base URL (`_api_url`).
///
/// `FIRECRAWL_API_URL` env override; default [`BASE_URL`].
///
/// Note: Python uses `os.environ.get("FIRECRAWL_API_URL", _BASE_URL)`, which
/// returns the env value even if it is the empty string. We reproduce that
/// exactly: an explicitly-set empty `FIRECRAWL_API_URL` yields `""`.
pub fn api_url() -> String {
    match std::env::var("FIRECRAWL_API_URL") {
        Ok(v) => v,
        Err(_) => BASE_URL.to_string(),
    }
}

/// Read the configured API key, if present and non-empty.
pub fn api_key() -> Option<String> {
    std::env::var("FIRECRAWL_API_KEY").ok().filter(|s| !s.is_empty())
}

/// Build the request headers (`_headers`), erroring when the key is missing.
///
/// Returns `(name, value)` pairs in the same order as the Python dict:
/// `Content-Type` then `Authorization`.
pub fn build_headers() -> Result<Vec<(String, String)>, FirecrawlError> {
    let key = api_key().ok_or(FirecrawlError::MissingCredentials)?;
    Ok(vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Authorization".to_string(), format!("Bearer {}", key)),
    ])
}

/// Read `FIRECRAWL_BROWSER_TTL` (default `300`), parsed as an integer.
///
/// Python does `int(os.environ.get("FIRECRAWL_BROWSER_TTL", "300"))`, which
/// raises `ValueError` on a non-integer value. We mirror that by returning an
/// error; the default `300` is used when the var is unset.
pub fn browser_ttl() -> Result<i64, FirecrawlError> {
    let raw = std::env::var("FIRECRAWL_BROWSER_TTL").unwrap_or_else(|_| "300".to_string());
    let trimmed = raw.trim();
    trimmed.parse::<i64>().map_err(|_| {
        FirecrawlError::Transport(format!(
            "invalid literal for int() with base 10: {:?}",
            raw
        ))
    })
}

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
    pub fn json(&self) -> Result<Value, FirecrawlError> {
        serde_json::from_str(&self.body)
            .map_err(|e| FirecrawlError::Transport(format!("invalid JSON response: {}", e)))
    }
}

/// Abstraction over the HTTP requests this provider makes, so the session
/// logic can be unit-tested without network access.
pub trait HttpClient {
    /// Perform a POST with a JSON body + headers, returning status & body text.
    fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        timeout_secs: u64,
    ) -> Result<HttpResponse, FirecrawlError>;

    /// Perform a DELETE with headers, returning status & body text.
    fn delete(
        &self,
        url: &str,
        headers: &[(String, String)],
        timeout_secs: u64,
    ) -> Result<HttpResponse, FirecrawlError>;
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
        headers: &[(String, String)],
        body: &Value,
        timeout_secs: u64,
    ) -> Result<HttpResponse, FirecrawlError> {
        let client = reqwest::blocking::Client::new();
        let mut req = client
            .post(url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .json(body);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req
            .send()
            .map_err(|e| FirecrawlError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .map_err(|e| FirecrawlError::Transport(e.to_string()))?;
        Ok(HttpResponse { status, body: text })
    }

    fn delete(
        &self,
        url: &str,
        headers: &[(String, String)],
        timeout_secs: u64,
    ) -> Result<HttpResponse, FirecrawlError> {
        let client = reqwest::blocking::Client::new();
        let mut req = client
            .delete(url)
            .timeout(std::time::Duration::from_secs(timeout_secs));
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req
            .send()
            .map_err(|e| FirecrawlError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .map_err(|e| FirecrawlError::Transport(e.to_string()))?;
        Ok(HttpResponse { status, body: text })
    }
}

// ---------------------------------------------------------------------------
// session name
// ---------------------------------------------------------------------------

/// Generate a session name `hermes_{task_id}_{8 lowercase hex chars}`.
///
/// Mirrors `f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"`.
pub fn make_session_name(task_id: &str) -> String {
    format!("hermes_{}_{}", task_id, random_hex8())
}

/// 8 lowercase hex characters from random bytes, matching `uuid4().hex[:8]`.
fn random_hex8() -> String {
    let bytes: [u8; 4] = {
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
    /// Provider session ID. Named `bb_session_id` for backward compat with the
    /// rest of `browser_tool` (legacy key shared across providers).
    pub bb_session_id: String,
    pub cdp_url: String,
}

impl SessionResult {
    /// JSON-object form matching the Python dict return value.
    pub fn to_json(&self) -> Value {
        json!({
            "session_name": self.session_name,
            "bb_session_id": self.bb_session_id,
            "cdp_url": self.cdp_url,
            "features": {"firecrawl": true},
        })
    }
}

// ---------------------------------------------------------------------------
// provider
// ---------------------------------------------------------------------------

/// Firecrawl cloud browser backend.
pub struct FirecrawlProvider<C: HttpClient = ReqwestClient> {
    client: C,
}

impl Default for FirecrawlProvider<ReqwestClient> {
    fn default() -> Self {
        FirecrawlProvider {
            client: ReqwestClient,
        }
    }
}

impl FirecrawlProvider<ReqwestClient> {
    /// Construct a provider using the default `reqwest::blocking` client.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: HttpClient> FirecrawlProvider<C> {
    /// Construct with a custom HTTP client (used in tests).
    pub fn with_client(client: C) -> Self {
        FirecrawlProvider { client }
    }

    /// `provider_name`.
    pub fn provider_name(&self) -> &'static str {
        "Firecrawl"
    }

    /// `is_configured`: `FIRECRAWL_API_KEY` present & non-empty.
    pub fn is_configured(&self) -> bool {
        api_key().is_some()
    }

    /// Create a Firecrawl browser session.
    pub fn create_session(&self, task_id: &str) -> Result<SessionResult, FirecrawlError> {
        let ttl = browser_ttl()?;
        let body = json!({ "ttl": ttl });
        let headers = build_headers()?;
        let url = format!("{}/v2/browser", api_url());

        let response = self.client.post_json(&url, &headers, &body, 30)?;

        if !response.ok() {
            return Err(FirecrawlError::CreateFailed {
                status: response.status,
                body: response.body.clone(),
            });
        }

        let data = response.json()?;
        let session_name = make_session_name(task_id);

        log::info!("Created Firecrawl browser session {}", session_name);

        let bb_session_id = data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or(FirecrawlError::MissingField("id"))?
            .to_string();
        let cdp_url = data
            .get("cdpUrl")
            .and_then(|v| v.as_str())
            .ok_or(FirecrawlError::MissingField("cdpUrl"))?
            .to_string();

        Ok(SessionResult {
            session_name,
            bb_session_id,
            cdp_url,
        })
    }

    /// Close a session via DELETE. Returns `true` on success.
    ///
    /// Mirrors `close_session`: success codes {200,201,204} → `true`; other
    /// codes → warn + `false`; any exception (incl. missing credentials) →
    /// error log + `false`.
    pub fn close_session(&self, session_id: &str) -> bool {
        let headers = match build_headers() {
            Ok(h) => h,
            Err(e) => {
                // Python wraps the whole body in `try/except Exception`, so a
                // missing-key ValueError lands here as an error log + False.
                log::error!("Exception closing Firecrawl session {}: {}", session_id, e);
                return false;
            }
        };
        let url = format!("{}/v2/browser/{}", api_url(), session_id);

        match self.client.delete(&url, &headers, 10) {
            Ok(resp) => {
                if matches!(resp.status, 200 | 201 | 204) {
                    log::debug!("Successfully closed Firecrawl session {}", session_id);
                    true
                } else {
                    let snippet: String = resp.body.chars().take(200).collect();
                    log::warn!(
                        "Failed to close Firecrawl session {}: HTTP {} - {}",
                        session_id,
                        resp.status,
                        snippet
                    );
                    false
                }
            }
            Err(e) => {
                log::error!("Exception closing Firecrawl session {}: {}", session_id, e);
                false
            }
        }
    }

    /// Best-effort session teardown used on shutdown. Never raises.
    ///
    /// Mirrors `emergency_cleanup`: missing credentials → warn; any other
    /// failure → debug log.
    pub fn emergency_cleanup(&self, session_id: &str) {
        let headers = match build_headers() {
            Ok(h) => h,
            Err(_) => {
                // Python: `except ValueError:` → warn.
                log::warn!(
                    "Cannot emergency-cleanup Firecrawl session {} — missing credentials",
                    session_id
                );
                return;
            }
        };
        let url = format!("{}/v2/browser/{}", api_url(), session_id);

        if let Err(e) = self.client.delete(&url, &headers, 5) {
            log::debug!(
                "Emergency cleanup failed for Firecrawl session {}: {}",
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
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_BROWSER_TTL",
    ];

    #[derive(Debug, Clone)]
    enum Call {
        Post { url: String, body: Value, timeout: u64 },
        Delete { url: String, timeout: u64 },
    }

    /// Mock client: scripted responses + recorded requests.
    struct MockClient {
        responses: RefCell<Vec<HttpResponse>>,
        calls: RefCell<Vec<Call>>,
    }

    impl MockClient {
        fn new(responses: Vec<HttpResponse>) -> Self {
            MockClient {
                responses: RefCell::new(responses),
                calls: RefCell::new(Vec::new()),
            }
        }
        fn next_response(&self) -> HttpResponse {
            let mut r = self.responses.borrow_mut();
            if r.is_empty() {
                HttpResponse {
                    status: 200,
                    body: "{}".to_string(),
                }
            } else {
                r.remove(0)
            }
        }
    }

    impl HttpClient for MockClient {
        fn post_json(
            &self,
            url: &str,
            _headers: &[(String, String)],
            body: &Value,
            timeout_secs: u64,
        ) -> Result<HttpResponse, FirecrawlError> {
            self.calls.borrow_mut().push(Call::Post {
                url: url.to_string(),
                body: body.clone(),
                timeout: timeout_secs,
            });
            Ok(self.next_response())
        }

        fn delete(
            &self,
            url: &str,
            _headers: &[(String, String)],
            timeout_secs: u64,
        ) -> Result<HttpResponse, FirecrawlError> {
            self.calls.borrow_mut().push(Call::Delete {
                url: url.to_string(),
                timeout: timeout_secs,
            });
            Ok(self.next_response())
        }
    }

    fn ok_session() -> HttpResponse {
        HttpResponse {
            status: 200,
            body: r#"{"id":"sess-123","cdpUrl":"wss://cdp.example/abc"}"#.to_string(),
        }
    }

    #[test]
    fn api_url_default_and_override() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        assert_eq!(api_url(), BASE_URL);
        e.set("FIRECRAWL_API_URL", "https://custom.example");
        assert_eq!(api_url(), "https://custom.example");
    }

    #[test]
    fn is_configured_reflects_key() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![]));
        assert!(!provider.is_configured());
        e.set("FIRECRAWL_API_KEY", "");
        assert!(!provider.is_configured());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        assert!(provider.is_configured());
    }

    #[test]
    fn headers_require_key() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        assert!(matches!(
            build_headers(),
            Err(FirecrawlError::MissingCredentials)
        ));
        e.set("FIRECRAWL_API_KEY", "fc-key");
        let h = build_headers().unwrap();
        assert_eq!(h[0], ("Content-Type".into(), "application/json".into()));
        assert_eq!(
            h[1],
            ("Authorization".into(), "Bearer fc-key".into())
        );
    }

    #[test]
    fn ttl_default_and_parse() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        assert_eq!(browser_ttl().unwrap(), 300);
        e.set("FIRECRAWL_BROWSER_TTL", "600");
        assert_eq!(browser_ttl().unwrap(), 600);
        e.set("FIRECRAWL_BROWSER_TTL", "not-an-int");
        assert!(matches!(browser_ttl(), Err(FirecrawlError::Transport(_))));
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
        e.set("FIRECRAWL_API_KEY", "fc-key");
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![ok_session()]));
        let res = provider.create_session("t1").unwrap();
        assert_eq!(res.bb_session_id, "sess-123");
        assert_eq!(res.cdp_url, "wss://cdp.example/abc");
        assert!(res.session_name.starts_with("hermes_t1_"));
        // Verify request shape.
        let calls = provider.client.calls.borrow();
        assert_eq!(calls.len(), 1);
        match &calls[0] {
            Call::Post { url, body, timeout } => {
                assert_eq!(url, "https://api.firecrawl.dev/v2/browser");
                assert_eq!(body["ttl"], json!(300));
                assert_eq!(*timeout, 30);
            }
            other => panic!("expected POST, got {:?}", other),
        }
        // Returned JSON shape.
        let j = res.to_json();
        assert_eq!(j["features"], json!({"firecrawl": true}));
        assert_eq!(j["bb_session_id"], json!("sess-123"));
    }

    #[test]
    fn create_session_uses_custom_ttl_and_url() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        e.set("FIRECRAWL_API_URL", "https://eu.firecrawl.dev");
        e.set("FIRECRAWL_BROWSER_TTL", "900");
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![ok_session()]));
        provider.create_session("x").unwrap();
        let calls = provider.client.calls.borrow();
        match &calls[0] {
            Call::Post { url, body, .. } => {
                assert_eq!(url, "https://eu.firecrawl.dev/v2/browser");
                assert_eq!(body["ttl"], json!(900));
            }
            other => panic!("expected POST, got {:?}", other),
        }
    }

    #[test]
    fn create_session_missing_key_errors() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![]));
        assert!(matches!(
            provider.create_session("t"),
            Err(FirecrawlError::MissingCredentials)
        ));
        // No HTTP call made (headers fail before request).
        assert!(provider.client.calls.borrow().is_empty());
    }

    #[test]
    fn create_session_failure_propagates_body() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 500,
            body: "boom".into(),
        }]));
        match provider.create_session("t") {
            Err(FirecrawlError::CreateFailed { status, body }) => {
                assert_eq!(status, 500);
                assert_eq!(body, "boom");
            }
            other => panic!("expected CreateFailed, got {:?}", other),
        }
    }

    #[test]
    fn close_session_success_codes() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        for code in [200u16, 201, 204] {
            let provider = FirecrawlProvider::with_client(MockClient::new(vec![HttpResponse {
                status: code,
                body: "".into(),
            }]));
            assert!(provider.close_session("sid"), "code {}", code);
            let calls = provider.client.calls.borrow();
            match &calls[0] {
                Call::Delete { url, timeout } => {
                    assert_eq!(url, "https://api.firecrawl.dev/v2/browser/sid");
                    assert_eq!(*timeout, 10);
                }
                other => panic!("expected DELETE, got {:?}", other),
            }
        }
    }

    #[test]
    fn close_session_failure_code() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 404,
            body: "nope".into(),
        }]));
        assert!(!provider.close_session("sid"));
    }

    #[test]
    fn close_session_no_creds_false_no_request() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![]));
        assert!(!provider.close_session("sid"));
        assert!(provider.client.calls.borrow().is_empty());
    }

    #[test]
    fn emergency_cleanup_makes_request() {
        let e = EnvGuard::new(ALL_ENV.to_vec());
        e.set("FIRECRAWL_API_KEY", "fc-key");
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![HttpResponse {
            status: 200,
            body: "".into(),
        }]));
        provider.emergency_cleanup("sid");
        let calls = provider.client.calls.borrow();
        assert_eq!(calls.len(), 1);
        match &calls[0] {
            Call::Delete { url, timeout } => {
                assert_eq!(url, "https://api.firecrawl.dev/v2/browser/sid");
                assert_eq!(*timeout, 5);
            }
            other => panic!("expected DELETE, got {:?}", other),
        }
    }

    #[test]
    fn emergency_cleanup_no_creds_noop() {
        let _e = EnvGuard::new(ALL_ENV.to_vec());
        let provider = FirecrawlProvider::with_client(MockClient::new(vec![]));
        provider.emergency_cleanup("sid");
        assert!(provider.client.calls.borrow().is_empty());
    }
}
