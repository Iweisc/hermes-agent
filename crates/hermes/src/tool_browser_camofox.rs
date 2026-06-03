//! Camofox browser backend — local anti-detection browser via REST API.
//!
//! Camofox-browser is a self-hosted Node.js server wrapping Camoufox (Firefox
//! fork with C++ fingerprint spoofing). It exposes a REST API that maps 1:1 to
//! our browser tool interface: accessibility snapshots with element refs,
//! click/type/scroll by ref, screenshots, etc.
//!
//! When `CAMOFOX_URL` is set (e.g. `http://localhost:9377`), the browser tools
//! route through this module instead of the `agent-browser` CLI.
//!
//! Setup:
//! ```text
//! # Option 1: npm
//! git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
//! npm install && npm start   # downloads Camoufox (~300MB) on first run
//!
//! # Option 2: Docker
//! docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser
//! ```
//!
//! Then set `CAMOFOX_URL=http://localhost:9377` in `~/.hermes/.env`.
//!
//! This is a faithful port of `tools/browser_camofox.py`. It reproduces the
//! REST request construction and JSON response shapes exactly. Where the Python
//! relied on cross-module helpers (config loading, snapshot summarization,
//! redaction, the vision LLM, identity derivation), this module references the
//! ported equivalents via `crate::` paths where available and otherwise accepts
//! injectable closures / falls back to local minimal implementations so that it
//! does not block on un-ported dependencies.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Cross-module dependencies (ported equivalents).
// ---------------------------------------------------------------------------
//
// `get_camofox_identity` lives in the already-ported state module. We reference
// it directly. `tool_error` mirrors `tools.registry.tool_error`.

use hermes_core::tool_browser_camofox_state::get_camofox_identity;
use hermes_core::tool_registry::tool_error as registry_tool_error;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Seconds per HTTP request (default).
const DEFAULT_TIMEOUT: u64 = 30;
/// Camofox paginates snapshots at this character limit. Retained for parity
/// with the Python constant `_SNAPSHOT_MAX_CHARS`.
#[allow(dead_code)]
const SNAPSHOT_MAX_CHARS: usize = 80_000;
/// Snapshot summarization threshold, mirroring
/// `tools.browser_tool.SNAPSHOT_SUMMARIZE_THRESHOLD`.
const SNAPSHOT_SUMMARIZE_THRESHOLD: usize = 8000;

/// Cached VNC URL discovered from the `/health` response (probed once).
static VNC_STATE: Mutex<VncState> = Mutex::new(VncState {
    vnc_url: None,
    checked: false,
});

struct VncState {
    vnc_url: Option<String>,
    checked: bool,
}

/// Return the configured Camofox server URL, or empty string.
///
/// Mirrors `os.getenv("CAMOFOX_URL", "").rstrip("/")`.
pub fn get_camofox_url() -> String {
    std::env::var("CAMOFOX_URL")
        .unwrap_or_default()
        .trim_end_matches('/')
        .to_string()
}

/// True when the Camofox backend is configured and no CDP override is active.
///
/// When the user has explicitly connected to a live Chrome instance via
/// `/browser connect` (which sets `BROWSER_CDP_URL`), the CDP connection takes
/// priority over Camofox so the browser tools operate on the real browser
/// instead of being silently routed to the Camofox backend.
pub fn is_camofox_mode() -> bool {
    if !std::env::var("BROWSER_CDP_URL")
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        return false;
    }
    !get_camofox_url().is_empty()
}

fn http_client(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

/// Verify the Camofox server is reachable.
///
/// On the first successful probe, parses the `/health` response for a `vncPort`
/// in the valid `1..=65535` range and caches a derived VNC URL.
pub fn check_camofox_available() -> bool {
    let url = get_camofox_url();
    if url.is_empty() {
        return false;
    }
    let client = http_client(5);
    let resp = match client.get(format!("{url}/health")).send() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let ok = resp.status().as_u16() == 200;
    if ok {
        let mut state = VNC_STATE.lock().unwrap();
        if !state.checked {
            if let Ok(data) = resp.json::<Value>() {
                if let Some(vnc_port) = data.get("vncPort").and_then(|v| v.as_i64()) {
                    if (1..=65535).contains(&vnc_port) {
                        let host = url::Url::parse(&url)
                            .ok()
                            .and_then(|p| p.host_str().map(|h| h.to_string()))
                            .unwrap_or_else(|| "localhost".to_string());
                        state.vnc_url = Some(format!("http://{host}:{vnc_port}"));
                    }
                }
            }
            state.checked = true;
        }
    }
    ok
}

/// Return the VNC URL if the Camofox server exposes one, or `None`.
///
/// Probes `/health` once if not yet checked.
pub fn get_vnc_url() -> Option<String> {
    let checked = { VNC_STATE.lock().unwrap().checked };
    if !checked {
        check_camofox_available();
    }
    VNC_STATE.lock().unwrap().vnc_url.clone()
}

/// Return whether Hermes-managed persistence is enabled for Camofox.
///
/// When enabled, sessions use a stable profile-scoped userId so the Camofox
/// server can map it to a persistent browser profile directory. When disabled
/// (the default), each session gets a random userId (ephemeral).
///
/// Controlled by `browser.camofox.managed_persistence` in `config.yaml`.
pub fn managed_persistence_enabled() -> bool {
    // `load_config` is in the ported config module; treat any failure as
    // "disabled" exactly like the Python `except Exception` path.
    let cfg = hermes_core::cli_config::load_config();
    cfg.get("browser")
        .and_then(|b| b.get("camofox"))
        .and_then(|c| c.get("managed_persistence"))
        .map(value_truthy)
        .unwrap_or(false)
}

/// Python `bool(...)` semantics for a JSON value.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/// In-memory session state, mirroring the Python session dict.
#[derive(Clone, Debug)]
pub struct Session {
    pub user_id: String,
    pub tab_id: Option<String>,
    pub session_key: String,
    pub managed: bool,
}

/// Maps `task_id -> Session`.
static SESSIONS: Mutex<Option<HashMap<String, Session>>> = Mutex::new(None);

fn with_sessions<R>(f: impl FnOnce(&mut HashMap<String, Session>) -> R) -> R {
    let mut guard = SESSIONS.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    f(guard.as_mut().unwrap())
}

/// Generate a random ephemeral userId suffix (10 hex chars), like
/// `uuid.uuid4().hex[:10]` in the Python source.
fn random_user_suffix() -> String {
    // Build a 16-byte random value and hex-encode the first 5 bytes (10 chars).
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Mix in the thread id and an atomic counter for additional entropy.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = format!("{:?}", std::thread::current().id());
    let seed = format!("{nanos}-{c}-{tid}");
    let digest = md5::compute(seed.as_bytes());
    let hex = format!("{digest:x}");
    hex[..10].to_string()
}

/// Get or create a Camofox session for the given task.
///
/// When managed persistence is enabled, uses a deterministic userId derived
/// from the Hermes profile so the Camofox server can map it to the same
/// persistent browser profile across restarts.
pub fn get_session(task_id: Option<&str>) -> Session {
    let task_id = task_id.filter(|t| !t.is_empty()).unwrap_or("default");
    with_sessions(|sessions| {
        if let Some(existing) = sessions.get(task_id) {
            return existing.clone();
        }
        let session = if managed_persistence_enabled() {
            let identity = get_camofox_identity(Some(task_id));
            Session {
                user_id: identity.get("user_id").cloned().unwrap_or_default(),
                tab_id: None,
                session_key: identity.get("session_key").cloned().unwrap_or_default(),
                managed: true,
            }
        } else {
            let key_part: String = task_id.chars().take(16).collect();
            Session {
                user_id: format!("hermes_{}", random_user_suffix()),
                tab_id: None,
                session_key: format!("task_{key_part}"),
                managed: false,
            }
        };
        sessions.insert(task_id.to_string(), session.clone());
        session
    })
}

/// Update the cached `tab_id` for a task's session.
fn set_session_tab(task_id: &str, tab_id: Option<String>) {
    with_sessions(|sessions| {
        if let Some(s) = sessions.get_mut(task_id) {
            s.tab_id = tab_id;
        }
    });
}

/// Ensure a tab exists for the session, creating one if needed.
///
/// Returns the (possibly updated) session. Network errors propagate as
/// `reqwest::Error` so callers can map them to tool errors.
pub fn ensure_tab(task_id: Option<&str>, url: &str) -> Result<Session, reqwest::Error> {
    let key = task_id.filter(|t| !t.is_empty()).unwrap_or("default").to_string();
    let session = get_session(Some(&key));
    if session.tab_id.is_some() {
        return Ok(session);
    }
    let base = get_camofox_url();
    let client = http_client(DEFAULT_TIMEOUT);
    let resp = client
        .post(format!("{base}/tabs"))
        .json(&json!({
            "userId": session.user_id,
            "sessionKey": session.session_key,
            "url": url,
        }))
        .send()?
        .error_for_status()?;
    let data: Value = resp.json()?;
    let tab_id = data
        .get("tabId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    set_session_tab(&key, tab_id);
    Ok(get_session(Some(&key)))
}

/// Remove and return session info.
pub fn drop_session(task_id: Option<&str>) -> Option<Session> {
    let task_id = task_id.filter(|t| !t.is_empty()).unwrap_or("default");
    with_sessions(|sessions| sessions.remove(task_id))
}

/// Release the in-memory session without destroying the server-side context.
///
/// When managed persistence is enabled the browser profile (and its cookies)
/// must survive across agent tasks. This drops only the local tracking entry
/// and returns `true`. When managed persistence is *not* enabled it does
/// nothing and returns `false` so the caller can fall back to [`camofox_close`].
pub fn camofox_soft_cleanup(task_id: Option<&str>) -> bool {
    if managed_persistence_enabled() {
        drop_session(task_id);
        log::debug!(
            "Camofox soft cleanup for task {} (managed persistence)",
            task_id.unwrap_or("default")
        );
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// POST JSON to Camofox and return parsed response.
fn post(path: &str, body: &Value, timeout: u64) -> Result<Value, reqwest::Error> {
    let url = format!("{}{}", get_camofox_url(), path);
    let client = http_client(timeout);
    let resp = client.post(url).json(body).send()?.error_for_status()?;
    resp.json()
}

/// GET from Camofox (with optional query params) and return parsed response.
fn get(path: &str, params: &[(&str, &str)], timeout: u64) -> Result<Value, reqwest::Error> {
    let url = format!("{}{}", get_camofox_url(), path);
    let client = http_client(timeout);
    let resp = client.get(url).query(params).send()?.error_for_status()?;
    resp.json()
}

/// GET from Camofox and return raw bytes (for binary data such as screenshots).
fn get_raw(path: &str, params: &[(&str, &str)], timeout: u64) -> Result<Vec<u8>, reqwest::Error> {
    let url = format!("{}{}", get_camofox_url(), path);
    let client = http_client(timeout);
    let resp = client.get(url).query(params).send()?.error_for_status()?;
    Ok(resp.bytes()?.to_vec())
}

/// DELETE to Camofox and return parsed response.
fn delete(path: &str, body: Option<&Value>, timeout: u64) -> Result<Value, reqwest::Error> {
    let url = format!("{}{}", get_camofox_url(), path);
    let client = http_client(timeout);
    let mut req = client.delete(url);
    if let Some(b) = body {
        req = req.json(b);
    }
    let resp = req.send()?.error_for_status()?;
    resp.json()
}

// ---------------------------------------------------------------------------
// Error / result helpers
// ---------------------------------------------------------------------------

/// `tools.registry.tool_error(message, success=False)` — returns a JSON object
/// `{"error": <message>, "success": false}`.
fn tool_error(message: &str) -> String {
    registry_tool_error(message, Some(json!({ "success": false })))
}

/// Detect whether a `reqwest::Error` is a connection-level failure (the Python
/// code distinguishes `requests.ConnectionError` from `requests.HTTPError`).
fn is_connection_error(e: &reqwest::Error) -> bool {
    e.is_connect() || e.is_timeout() || (e.is_request() && e.status().is_none())
}

// ---------------------------------------------------------------------------
// Snapshot summarization hook
// ---------------------------------------------------------------------------

/// Truncate a snapshot to the default budget. Mirrors
/// `tools.browser_tool._truncate_snapshot`.
fn truncate_snapshot(snapshot_text: &str) -> String {
    crate::tool_browser_tool::truncate_snapshot_default(snapshot_text)
}

/// Apply the snapshot summarization logic used by the main browser tool.
///
/// When a `user_task` is supplied and an extraction `llm` closure is provided,
/// `extract_relevant_content` is used; otherwise the snapshot is truncated. The
/// closure mirrors the auxiliary-client delegation in the ported browser tool.
fn summarize_snapshot<F>(snapshot: &str, user_task: Option<&str>, llm: Option<F>) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    if snapshot.len() > SNAPSHOT_SUMMARIZE_THRESHOLD {
        if user_task.is_some() {
            crate::tool_browser_tool::extract_relevant_content(snapshot, user_task, llm)
        } else {
            truncate_snapshot(snapshot)
        }
    } else {
        snapshot.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

/// Navigate to a URL via Camofox.
///
/// Returns a JSON result string identical in shape to the Python implementation
/// (`success`, `url`, `title`, optional `vnc_url`/`vnc_hint`, optional
/// `snapshot`/`element_count`).
pub fn camofox_navigate(url: &str, task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let (data, active_session): (Value, Session) = if session.tab_id.is_none() {
        // Create tab with the target URL directly.
        match ensure_tab(task_id, url) {
            Ok(s) => (json!({ "ok": true, "url": url }), s),
            Err(e) => return navigate_error(&e),
        }
    } else {
        let tab_id = session.tab_id.clone().unwrap();
        match post(
            &format!("/tabs/{tab_id}/navigate"),
            &json!({ "userId": session.user_id, "url": url }),
            60,
        ) {
            Ok(d) => (d, session.clone()),
            Err(e) => return navigate_error(&e),
        }
    };

    let mut result = Map::new();
    result.insert("success".to_string(), Value::Bool(true));
    result.insert(
        "url".to_string(),
        Value::String(
            data.get("url")
                .and_then(|v| v.as_str())
                .unwrap_or(url)
                .to_string(),
        ),
    );
    result.insert(
        "title".to_string(),
        Value::String(
            data.get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        ),
    );

    if let Some(vnc) = get_vnc_url() {
        result.insert("vnc_url".to_string(), Value::String(vnc));
        result.insert(
            "vnc_hint".to_string(),
            Value::String(
                "Browser is visible via VNC. Share this link with the user so they \
                 can watch the browser live."
                    .to_string(),
            ),
        );
    }

    // Auto-take a compact snapshot so the model can act immediately. Failures
    // here are swallowed: navigation already succeeded, the snapshot is a bonus.
    if let Some(tab_id) = active_session.tab_id.as_deref() {
        if let Ok(snap_data) = get(
            &format!("/tabs/{tab_id}/snapshot"),
            &[("userId", active_session.user_id.as_str())],
            DEFAULT_TIMEOUT,
        ) {
            let mut snapshot_text = snap_data
                .get("snapshot")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if snapshot_text.len() > SNAPSHOT_SUMMARIZE_THRESHOLD {
                snapshot_text = truncate_snapshot(&snapshot_text);
            }
            result.insert("snapshot".to_string(), Value::String(snapshot_text));
            result.insert(
                "element_count".to_string(),
                snap_data
                    .get("refsCount")
                    .cloned()
                    .unwrap_or(Value::Number(0.into())),
            );
        }
    }

    Value::Object(result).to_string()
}

fn navigate_error(e: &reqwest::Error) -> String {
    if is_connection_error(e) {
        json!({
            "success": false,
            "error": format!(
                "Cannot connect to Camofox at {}. Is the server running? Start with: \
                 npm start (in camofox-browser dir) or: docker run -p 9377:9377 \
                 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser",
                get_camofox_url()
            ),
        })
        .to_string()
    } else if e.status().is_some() {
        tool_error(&format!("Navigation failed: {e}"))
    } else {
        tool_error(&e.to_string())
    }
}

/// Get an accessibility tree snapshot from Camofox.
///
/// `llm` is an optional extraction closure used for relevance-based
/// summarization when `user_task` is supplied; pass `None::<fn(&str) ->
/// Option<String>>` to fall back to plain truncation.
pub fn camofox_snapshot<F>(
    _full: bool,
    task_id: Option<&str>,
    user_task: Option<&str>,
    llm: Option<F>,
) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };

    let data = match get(
        &format!("/tabs/{tab_id}/snapshot"),
        &[("userId", session.user_id.as_str())],
        DEFAULT_TIMEOUT,
    ) {
        Ok(d) => d,
        Err(e) => return tool_error(&e.to_string()),
    };

    let snapshot = data
        .get("snapshot")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let refs_count = data
        .get("refsCount")
        .cloned()
        .unwrap_or(Value::Number(0.into()));

    let snapshot = summarize_snapshot(&snapshot, user_task, llm);

    json!({
        "success": true,
        "snapshot": snapshot,
        "element_count": refs_count,
    })
    .to_string()
}

/// Click an element by ref via Camofox.
///
/// The leading `@` (our tool convention) is stripped before sending.
pub fn camofox_click(ref_: &str, task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };
    let clean_ref = ref_.trim_start_matches('@');
    match post(
        &format!("/tabs/{tab_id}/click"),
        &json!({ "userId": session.user_id, "ref": clean_ref }),
        DEFAULT_TIMEOUT,
    ) {
        Ok(data) => json!({
            "success": true,
            "clicked": clean_ref,
            "url": data.get("url").and_then(|v| v.as_str()).unwrap_or(""),
        })
        .to_string(),
        Err(e) => tool_error(&e.to_string()),
    }
}

/// Type text into an element by ref via Camofox.
pub fn camofox_type(ref_: &str, text: &str, task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };
    let clean_ref = ref_.trim_start_matches('@');
    match post(
        &format!("/tabs/{tab_id}/type"),
        &json!({ "userId": session.user_id, "ref": clean_ref, "text": text }),
        DEFAULT_TIMEOUT,
    ) {
        Ok(_) => json!({
            "success": true,
            "typed": text,
            "element": clean_ref,
        })
        .to_string(),
        Err(e) => tool_error(&e.to_string()),
    }
}

/// Scroll the page via Camofox.
pub fn camofox_scroll(direction: &str, task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };
    match post(
        &format!("/tabs/{tab_id}/scroll"),
        &json!({ "userId": session.user_id, "direction": direction }),
        DEFAULT_TIMEOUT,
    ) {
        Ok(_) => json!({ "success": true, "scrolled": direction }).to_string(),
        Err(e) => tool_error(&e.to_string()),
    }
}

/// Navigate back via Camofox.
pub fn camofox_back(task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };
    match post(
        &format!("/tabs/{tab_id}/back"),
        &json!({ "userId": session.user_id }),
        DEFAULT_TIMEOUT,
    ) {
        Ok(data) => json!({
            "success": true,
            "url": data.get("url").and_then(|v| v.as_str()).unwrap_or(""),
        })
        .to_string(),
        Err(e) => tool_error(&e.to_string()),
    }
}

/// Press a keyboard key via Camofox.
pub fn camofox_press(key: &str, task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };
    match post(
        &format!("/tabs/{tab_id}/press"),
        &json!({ "userId": session.user_id, "key": key }),
        DEFAULT_TIMEOUT,
    ) {
        Ok(_) => json!({ "success": true, "pressed": key }).to_string(),
        Err(e) => tool_error(&e.to_string()),
    }
}

/// Close the browser session via Camofox.
///
/// Always reports `success: true`; a server-side failure is surfaced as a
/// non-fatal `warning` field, matching the Python behavior.
pub fn camofox_close(task_id: Option<&str>) -> String {
    let session = match drop_session(task_id) {
        Some(s) => s,
        None => return json!({ "success": true, "closed": true }).to_string(),
    };
    match delete(
        &format!("/sessions/{}", session.user_id),
        None,
        DEFAULT_TIMEOUT,
    ) {
        Ok(_) => json!({ "success": true, "closed": true }).to_string(),
        Err(e) => json!({
            "success": true,
            "closed": true,
            "warning": e.to_string(),
        })
        .to_string(),
    }
}

/// Get images on the current page via Camofox.
///
/// Camofox does not expose a dedicated `/images` endpoint, so image info is
/// extracted from the accessibility tree snapshot. Mirrors the Python regex
/// parsing exactly: `img` entries (optionally prefixed `- `) carry an alt text
/// in double quotes, and the following line may carry a `/url:` source.
pub fn camofox_get_images(task_id: Option<&str>) -> String {
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };

    let data = match get(
        &format!("/tabs/{tab_id}/snapshot"),
        &[("userId", session.user_id.as_str())],
        DEFAULT_TIMEOUT,
    ) {
        Ok(d) => d,
        Err(e) => return tool_error(&e.to_string()),
    };
    let snapshot = data
        .get("snapshot")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let images = parse_images_from_snapshot(&snapshot);
    let count = images.len();
    json!({
        "success": true,
        "images": images,
        "count": count,
    })
    .to_string()
}

/// Parse `img` elements out of an accessibility-tree snapshot, returning a list
/// of `{ "src": ..., "alt": ... }` objects. Extracted as a free function so it
/// can be unit-tested without a live server.
fn parse_images_from_snapshot(snapshot: &str) -> Vec<Value> {
    use regex::Regex;
    let alt_re = Regex::new(r#"img\s+"([^"]*)""#).unwrap();
    let url_re = Regex::new(r"/url:\s*(\S+)").unwrap();

    let lines: Vec<&str> = snapshot.split('\n').collect();
    let mut images = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let stripped = line.trim();
        if stripped.starts_with("- img ") || stripped.starts_with("img ") {
            let alt = alt_re
                .captures(stripped)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let mut src = String::new();
            if let Some(next) = lines.get(i + 1) {
                if let Some(c) = url_re.captures(next.trim()) {
                    if let Some(m) = c.get(1) {
                        src = m.as_str().to_string();
                    }
                }
            }
            if !alt.is_empty() || !src.is_empty() {
                images.push(json!({ "src": src, "alt": alt }));
            }
        }
    }
    images
}

/// Outcome of capturing a screenshot for [`camofox_vision`].
///
/// The Python `camofox_vision` couples three concerns: fetching a PNG, saving
/// it to the Hermes cache, and calling the auxiliary vision LLM. Because the
/// vision LLM is supplied by a separate ported module, this function takes the
/// LLM call as a closure (`vision_llm`) so the request/response plumbing can be
/// tested and reused independently.
pub struct VisionPrep {
    /// Filesystem path the PNG was written to.
    pub screenshot_path: String,
    /// Base64-encoded PNG bytes (ready for a `data:image/png;base64,...` URL).
    pub image_b64: String,
    /// Annotation context (redacted accessibility-tree snippet), already
    /// prefixed exactly as the Python builds it.
    pub annotation_context: String,
}

/// Take a screenshot and analyze it with a vision LLM via Camofox.
///
/// `vision_llm` receives `(vision_prompt, image_b64, temperature, timeout)` and
/// must return the raw analysis text (it is redacted here before being placed
/// in the result), mirroring `agent.auxiliary_client.call_llm(task="vision")`.
/// Pass `None` to skip the LLM call (the screenshot is still saved and its path
/// returned with an empty analysis).
pub fn camofox_vision<F>(
    question: &str,
    annotate: bool,
    task_id: Option<&str>,
    vision_llm: Option<F>,
) -> String
where
    F: FnOnce(&str, &str, f64, f64) -> Option<String>,
{
    let session = get_session(task_id);
    let tab_id = match session.tab_id.as_deref() {
        Some(t) => t,
        None => return tool_error("No browser session. Call browser_navigate first."),
    };

    // Get screenshot as binary PNG.
    let png = match get_raw(
        &format!("/tabs/{tab_id}/screenshot"),
        &[("userId", session.user_id.as_str())],
        DEFAULT_TIMEOUT,
    ) {
        Ok(b) => b,
        Err(e) => return tool_error(&e.to_string()),
    };

    // Save screenshot to cache.
    let screenshots_dir = hermes_core::mod_hermes_constants::get_hermes_home().join("browser_screenshots");
    if let Err(e) = std::fs::create_dir_all(&screenshots_dir) {
        return tool_error(&e.to_string());
    }
    let screenshot_path = screenshots_dir
        .join(format!("browser_screenshot_{}.png", &random_user_suffix()[..8]))
        .to_string_lossy()
        .to_string();
    if let Err(e) = std::fs::write(&screenshot_path, &png) {
        return tool_error(&e.to_string());
    }

    // Encode for the vision LLM.
    use base64::Engine;
    let img_b64 = base64::engine::general_purpose::STANDARD.encode(&png);

    // Also get an annotated snapshot if requested.
    let mut annotation_context = String::new();
    if annotate {
        if let Ok(snap_data) = get(
            &format!("/tabs/{tab_id}/snapshot"),
            &[("userId", session.user_id.as_str())],
            DEFAULT_TIMEOUT,
        ) {
            let snap = snap_data
                .get("snapshot")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let snippet: String = snap.chars().take(3000).collect();
            annotation_context =
                format!("\n\nAccessibility tree (element refs for interaction):\n{snippet}");
        }
    }

    // Redact secrets from annotation context before sending to the vision LLM.
    annotation_context =
        hermes_core::agent_redact::redact_sensitive_text(&annotation_context, false, false);

    let vision_prompt = format!(
        "Analyze this browser screenshot and answer: {question}{annotation_context}"
    );

    // Vision config defaults mirror `auxiliary.vision.{timeout,temperature}`.
    let cfg = hermes_core::cli_config::load_config();
    let vision_timeout = cfg
        .get("auxiliary")
        .and_then(|a| a.get("vision"))
        .and_then(|v| v.get("timeout"))
        .and_then(|t| t.as_f64())
        .unwrap_or(120.0);
    let vision_temperature = cfg
        .get("auxiliary")
        .and_then(|a| a.get("vision"))
        .and_then(|v| v.get("temperature"))
        .and_then(|t| t.as_f64())
        .unwrap_or(0.1);

    let analysis_raw = match vision_llm {
        Some(f) => f(&vision_prompt, &img_b64, vision_temperature, vision_timeout)
            .unwrap_or_default(),
        None => String::new(),
    };
    let analysis = hermes_core::agent_redact::redact_sensitive_text(analysis_raw.trim(), false, false);

    let _ = VisionPrep {
        screenshot_path: screenshot_path.clone(),
        image_b64: img_b64,
        annotation_context,
    };

    json!({
        "success": true,
        "analysis": analysis,
        "screenshot_path": screenshot_path,
    })
    .to_string()
}

/// Get console output — limited support in Camofox.
///
/// Camofox does not expose browser console logs via its REST API, so this
/// returns an empty result with an explanatory note (matching the Python).
pub fn camofox_console(_clear: bool, _task_id: Option<&str>) -> String {
    json!({
        "success": true,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. \
                 Use browser_snapshot or browser_vision to inspect page state.",
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_camofox_url_strips_trailing_slashes() {
        unsafe {
            std::env::set_var("CAMOFOX_URL", "http://localhost:9377///");
        }
        assert_eq!(get_camofox_url(), "http://localhost:9377");
        unsafe {
            std::env::remove_var("CAMOFOX_URL");
        }
        assert_eq!(get_camofox_url(), "");
    }

    #[test]
    fn is_camofox_mode_respects_cdp_override() {
        unsafe {
            std::env::set_var("CAMOFOX_URL", "http://localhost:9377");
            std::env::remove_var("BROWSER_CDP_URL");
        }
        assert!(is_camofox_mode());

        unsafe {
            std::env::set_var("BROWSER_CDP_URL", "http://localhost:9222");
        }
        assert!(!is_camofox_mode());

        unsafe {
            std::env::set_var("BROWSER_CDP_URL", "   ");
        }
        assert!(is_camofox_mode());

        unsafe {
            std::env::remove_var("BROWSER_CDP_URL");
            std::env::remove_var("CAMOFOX_URL");
        }
        assert!(!is_camofox_mode());
    }

    #[test]
    fn value_truthy_matches_python_bool() {
        assert!(!value_truthy(&Value::Null));
        assert!(!value_truthy(&Value::Bool(false)));
        assert!(value_truthy(&Value::Bool(true)));
        assert!(!value_truthy(&json!(0)));
        assert!(value_truthy(&json!(1)));
        assert!(!value_truthy(&json!("")));
        assert!(value_truthy(&json!("x")));
    }

    #[test]
    fn tool_error_shape_matches_python() {
        let s = tool_error("boom");
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["error"], json!("boom"));
        assert_eq!(v["success"], json!(false));
    }

    #[test]
    fn random_user_suffix_is_10_hex_chars() {
        let s = random_user_suffix();
        assert_eq!(s.len(), 10);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
        // Two consecutive calls should differ (counter + time entropy).
        assert_ne!(random_user_suffix(), random_user_suffix());
    }

    #[test]
    fn console_returns_empty_with_note() {
        let s = camofox_console(false, Some("t1"));
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["total_messages"], json!(0));
        assert_eq!(v["total_errors"], json!(0));
        assert!(v["console_messages"].as_array().unwrap().is_empty());
        assert!(v["note"].as_str().unwrap().contains("not available"));
    }

    #[test]
    fn parse_images_extracts_alt_and_url() {
        let snapshot = "\
- generic
  - img \"Logo\"
    - /url: https://example.com/logo.png
  - img \"\"
  - img \"NoUrl\"
  - button \"Click\"
img \"BareImg\"
  - /url: https://example.com/bare.png
";
        let images = parse_images_from_snapshot(snapshot);
        // "Logo" with url, "NoUrl" with empty src, "BareImg" with url.
        // The empty-alt + no-url img is skipped (both empty).
        assert_eq!(images.len(), 3);
        assert_eq!(images[0]["alt"], json!("Logo"));
        assert_eq!(images[0]["src"], json!("https://example.com/logo.png"));
        assert_eq!(images[1]["alt"], json!("NoUrl"));
        assert_eq!(images[1]["src"], json!(""));
        assert_eq!(images[2]["alt"], json!("BareImg"));
        assert_eq!(images[2]["src"], json!("https://example.com/bare.png"));
    }

    #[test]
    fn parse_images_empty_when_none() {
        let snapshot = "- generic\n  - button \"Click\"\n  - text \"hello\"";
        assert!(parse_images_from_snapshot(snapshot).is_empty());
    }

    #[test]
    fn summarize_snapshot_short_passthrough() {
        let short = "small snapshot";
        let out = summarize_snapshot(short, None, None::<fn(&str) -> Option<String>>);
        assert_eq!(out, short);
    }

    #[test]
    fn summarize_snapshot_long_truncates_without_task() {
        let long = "a".repeat(SNAPSHOT_SUMMARIZE_THRESHOLD + 500);
        let out = summarize_snapshot(&long, None, None::<fn(&str) -> Option<String>>);
        assert!(out.len() <= long.len());
    }

    #[test]
    fn drop_and_get_session_roundtrip() {
        // No managed persistence config in test env -> ephemeral session.
        unsafe {
            std::env::remove_var("CAMOFOX_URL");
        }
        let s = get_session(Some("test-task-roundtrip-xyz"));
        assert!(s.user_id.starts_with("hermes_"));
        assert!(s.session_key.starts_with("task_"));
        assert!(s.tab_id.is_none());
        let dropped = drop_session(Some("test-task-roundtrip-xyz"));
        assert!(dropped.is_some());
        // Second drop returns None.
        assert!(drop_session(Some("test-task-roundtrip-xyz")).is_none());
    }
}
