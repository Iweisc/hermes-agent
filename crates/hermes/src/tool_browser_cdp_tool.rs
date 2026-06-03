//! Raw Chrome DevTools Protocol (CDP) passthrough tool.
//!
//! Native Rust port of `tools/browser_cdp_tool.py`.
//!
//! Exposes a single tool, `browser_cdp`, that sends arbitrary CDP commands to
//! the browser's DevTools WebSocket endpoint. Works when a CDP URL is
//! configured — either via `/browser connect` (sets `BROWSER_CDP_URL`) or
//! `browser.cdp_url` in `config.yaml` — or when a CDP-backed cloud provider
//! session is active.
//!
//! This is the escape hatch for browser operations not covered by the main
//! browser tool surface (`browser_navigate`, `browser_click`,
//! `browser_console`, etc.) — handling native dialogs, iframe-scoped
//! evaluation, cookie/network control, low-level tab management, etc.
//!
//! Method reference: <https://chromedevtools.github.io/devtools-protocol/>

use std::net::TcpStream;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};
use url::Url;

pub const CDP_DOCS_URL: &str = "https://chromedevtools.github.io/devtools-protocol/";

const DEFAULT_CDP_TIMEOUT_SECS: f64 = 30.0;
const MIN_CDP_TIMEOUT_SECS: f64 = 1.0;
const MAX_CDP_TIMEOUT_SECS: f64 = 300.0;

// ---------------------------------------------------------------------------
// Error helper (mirrors tools.registry.tool_error with optional extra keys)
// ---------------------------------------------------------------------------

/// Build a `{"error": ...}` JSON payload, optionally merging extra context
/// keys (e.g. `method`, `cdp_docs`) the way the Python `tool_error(...)` does
/// via keyword arguments.
pub fn tool_error_with(message: impl Into<String>, extra: &[(&str, Value)]) -> String {
    let mut object = Map::new();
    object.insert("error".to_string(), Value::String(message.into()));
    for (key, value) in extra {
        object.insert((*key).to_string(), value.clone());
    }
    Value::Object(object).to_string()
}

/// Simple `{"error": ...}` payload.
pub fn tool_error(message: impl Into<String>) -> String {
    tool_error_with(message, &[])
}

// ---------------------------------------------------------------------------
// Endpoint resolution
// ---------------------------------------------------------------------------

/// Resolve the CDP override the same way the rest of the browser surface does:
///
/// 1. `BROWSER_CDP_URL` env var (live override from `/browser connect`)
/// 2. `browser.cdp_url` in `config.yaml` (under `hermes_home`)
///
/// Returns the trimmed string, or empty string when unavailable. This mirrors
/// `tools.browser_tool._get_cdp_override()`.
pub fn resolve_cdp_endpoint(hermes_home: &std::path::Path) -> String {
    if let Ok(value) = std::env::var("BROWSER_CDP_URL") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    cdp_url_from_config(hermes_home).unwrap_or_default()
}

fn cdp_url_from_config(hermes_home: &std::path::Path) -> Option<String> {
    use serde_yaml::Value as YamlValue;
    let path = hermes_home.join("config.yaml");
    let contents = std::fs::read_to_string(path).ok()?;
    let parsed = serde_yaml::from_str::<YamlValue>(&contents).ok()?;
    parsed
        .as_mapping()?
        .get(YamlValue::String("browser".to_string()))?
        .as_mapping()?
        .get(YamlValue::String("cdp_url".to_string()))?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

// ---------------------------------------------------------------------------
// Core CDP call (stateless, fresh connection — mirrors `_cdp_call`)
// ---------------------------------------------------------------------------

/// Distinguish timeout from other errors so callers can format messages the
/// way the Python handler does (`asyncio.TimeoutError`, `TimeoutError`,
/// `RuntimeError`, `WebSocketException`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdpError {
    /// Timed out attaching / waiting for a response.
    Timeout(String),
    /// CDP protocol-level error (`Target.attachToTarget failed`, `CDP error`).
    Runtime(String),
    /// Transport / WebSocket error (connect, send, read failures).
    WebSocket(String),
}

impl CdpError {
    pub fn message(&self) -> &str {
        match self {
            CdpError::Timeout(m) | CdpError::Runtime(m) | CdpError::WebSocket(m) => m,
        }
    }
}

/// Make a single CDP call, optionally attaching to a target first.
///
/// When `target_id` is provided, calls `Target.attachToTarget` with
/// `flatten=true` to multiplex a page-level session over the same
/// browser-level WebSocket, then sends `method` with that `sessionId`. When
/// `target_id` is `None`, `method` is sent at browser level.
pub fn cdp_call(
    ws_url: &str,
    method: &str,
    params: Value,
    target_id: Option<&str>,
    timeout: f64,
) -> Result<Value, CdpError> {
    let _ = Url::parse(ws_url)
        .map_err(|error| CdpError::WebSocket(format!("Invalid CDP endpoint: {error}")))?;

    let (mut socket, _) = connect(ws_url)
        .map_err(|error| CdpError::WebSocket(format!("Connecting to CDP endpoint failed: {error}")))?;
    let socket_timeout = Duration::from_secs_f64(timeout.max(MIN_CDP_TIMEOUT_SECS));
    set_cdp_timeouts(&mut socket, socket_timeout);

    let mut next_id: i64 = 1;
    let mut session_id: Option<String> = None;

    // --- Step 1: attach to target if requested ---
    if let Some(target_id) = target_id {
        let attach_id = next_id;
        next_id += 1;
        socket
            .send(Message::Text(
                json!({
                    "id": attach_id,
                    "method": "Target.attachToTarget",
                    "params": {"targetId": target_id, "flatten": true},
                })
                .to_string()
                .into(),
            ))
            .map_err(|error| {
                CdpError::WebSocket(format!("Sending Target.attachToTarget failed: {error}"))
            })?;

        let deadline = Instant::now() + socket_timeout;
        loop {
            if Instant::now() >= deadline {
                return Err(CdpError::Timeout(format!(
                    "Timed out attaching to target {target_id}"
                )));
            }
            let message = read_cdp_message(&mut socket)?;
            if message.get("id").and_then(Value::as_i64) != Some(attach_id) {
                // Ignore events (messages without matching "id") while waiting.
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(CdpError::Runtime(format!(
                    "Target.attachToTarget failed: {error}"
                )));
            }
            let Some(value) = message
                .get("result")
                .and_then(|result| result.get("sessionId"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            else {
                return Err(CdpError::Runtime(
                    "Target.attachToTarget did not return a sessionId".to_string(),
                ));
            };
            session_id = Some(value.to_string());
            break;
        }
    }

    // --- Step 2: dispatch the real method ---
    let call_id = next_id;
    let mut request = json!({
        "id": call_id,
        "method": method,
        "params": params,
    });
    if let Some(session_id) = session_id
        && let Some(object) = request.as_object_mut()
    {
        object.insert("sessionId".to_string(), Value::String(session_id));
    }
    socket
        .send(Message::Text(request.to_string().into()))
        .map_err(|error| CdpError::WebSocket(format!("Sending CDP method {method} failed: {error}")))?;

    let deadline = Instant::now() + socket_timeout;
    loop {
        if Instant::now() >= deadline {
            return Err(CdpError::Timeout(format!(
                "Timed out waiting for response to {method}"
            )));
        }
        let message = read_cdp_message(&mut socket)?;
        if message.get("id").and_then(Value::as_i64) != Some(call_id) {
            // Ignore events / out-of-order responses.
            continue;
        }
        if let Some(error) = message.get("error") {
            return Err(CdpError::Runtime(format!("CDP error: {error}")));
        }
        return Ok(message.get("result").cloned().unwrap_or(Value::Null));
    }
}

fn read_cdp_message(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>) -> Result<Value, CdpError> {
    loop {
        let message = socket
            .read()
            .map_err(|error| CdpError::WebSocket(format!("Reading CDP response failed: {error}")))?;
        match message {
            Message::Text(text) => {
                return serde_json::from_str(text.as_ref()).map_err(|error| {
                    CdpError::WebSocket(format!("Invalid JSON from CDP endpoint: {error}"))
                });
            }
            Message::Binary(bytes) => {
                let text = String::from_utf8(bytes.to_vec()).map_err(|error| {
                    CdpError::WebSocket(format!("Invalid binary CDP frame: {error}"))
                })?;
                return serde_json::from_str(&text).map_err(|error| {
                    CdpError::WebSocket(format!("Invalid JSON from CDP endpoint: {error}"))
                });
            }
            Message::Ping(payload) => {
                socket.send(Message::Pong(payload)).map_err(|error| {
                    CdpError::WebSocket(format!("Responding to CDP ping failed: {error}"))
                })?;
            }
            Message::Close(_) => {
                return Err(CdpError::WebSocket(
                    "CDP connection closed before a response arrived".to_string(),
                ));
            }
            _ => {}
        }
    }
}

fn set_cdp_timeouts(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>, timeout: Duration) {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }
        MaybeTlsStream::Rustls(stream) => {
            let tcp = stream.get_mut();
            let _ = tcp.set_read_timeout(Some(timeout));
            let _ = tcp.set_write_timeout(Some(timeout));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Supervisor (frame_id / OOPIF) routing
// ---------------------------------------------------------------------------

/// A supervisor-routed CDP dispatcher. The integration layer supplies this so
/// `browser_cdp` can run `method` against an OOPIF's live CDP session over the
/// supervisor's already-connected WebSocket (the way Python dispatches via
/// `supervisor._cdp(...)` on the supervisor loop).
///
/// Implementations receive the resolved child `session_id` and the call
/// parameters and must return the raw CDP response message (the object that
/// contains a `"result"` field), or an error string.
pub trait SupervisorCdpRouter {
    /// Look up the frame in the supervisor for `task_id` and dispatch `method`
    /// against its dedicated CDP session.
    ///
    /// Returns:
    /// - `Ok(Some(result_message))` — dispatched; `result_message` is the raw
    ///   CDP response (an object with a `"result"` key).
    /// - `Ok(None)` — the supervisor exists but cannot route (caller should
    ///   surface a not-found / not-OOPIF error itself); generally unused since
    ///   routers return their own error strings.
    /// - `Err(error)` — a fully formed `tool_error` JSON string to return
    ///   directly to the caller.
    fn route(
        &self,
        task_id: &str,
        frame_id: &str,
        method: &str,
        params: Option<Value>,
        timeout: f64,
    ) -> Result<RouteOutcome, String>;
}

/// Outcome of a supervisor-routed CDP call.
#[derive(Debug, Clone)]
pub enum RouteOutcome {
    /// Successfully dispatched; carries `(session_id, result_value)` where
    /// `result_value` is the contents of the CDP response `"result"` field.
    Dispatched { session_id: String, result: Value },
    /// A `tool_error(...)` JSON string the handler should return verbatim
    /// (e.g. supervisor not attached, frame not found, frame not an OOPIF).
    Error(String),
}

/// Default router used when integration has not installed a real one.
///
/// It performs the same frame lookup the Python code does (against the
/// supervisor registry snapshot), and returns precise, faithful error strings
/// for the not-available / not-attached / not-found / not-OOPIF cases. When a
/// dedicated OOPIF session *is* found but no live-loop dispatch is wired, it
/// returns a defensive error rather than silently mis-routing.
pub struct DefaultSupervisorRouter;

impl SupervisorCdpRouter for DefaultSupervisorRouter {
    fn route(
        &self,
        task_id: &str,
        frame_id: &str,
        _method: &str,
        _params: Option<Value>,
        _timeout: f64,
    ) -> Result<RouteOutcome, String> {
        match lookup_oopif_session(task_id, frame_id) {
            FrameLookup::SupervisorUnavailable(message) => Ok(RouteOutcome::Error(message)),
            FrameLookup::NotAttached(message) => Ok(RouteOutcome::Error(message)),
            FrameLookup::NotFound(message) => Ok(RouteOutcome::Error(message)),
            FrameLookup::NotOopif(message) => Ok(RouteOutcome::Error(message)),
            FrameLookup::Found { .. } => Ok(RouteOutcome::Error(tool_error_with(
                "CDP supervisor live-session routing is not wired into this build. \
                 The frame was located but no dispatcher is installed.",
                &[("cdp_docs", Value::String(CDP_DOCS_URL.to_string()))],
            ))),
        }
    }
}

/// Result of resolving a `frame_id` to an OOPIF CDP session via the supervisor
/// registry snapshot. Each non-`Found` variant carries a ready-to-return
/// `tool_error(...)` JSON string mirroring the Python messages.
pub enum FrameLookup {
    SupervisorUnavailable(String),
    NotAttached(String),
    NotFound(String),
    NotOopif(String),
    Found { session_id: String },
}

/// Locate `frame_id`'s OOPIF session in the supervisor for `task_id`.
///
/// Mirrors `_browser_cdp_via_supervisor` lookup logic: searches the snapshot's
/// `frame_tree.top` then `frame_tree.children[]` for a matching `frame_id`,
/// and reads its `session_id`. A missing `session_id` means the frame is not
/// an out-of-process iframe.
pub fn lookup_oopif_session(task_id: &str, frame_id: &str) -> FrameLookup {
    let supervisor = match crate::tool_browser_supervisor::supervisor_registry().get(task_id) {
        Some(supervisor) => supervisor,
        None => {
            return FrameLookup::NotAttached(tool_error(format!(
                "No CDP supervisor is attached for task={task_id:?}. Call \
                 browser_navigate or /browser connect first so the supervisor \
                 can attach. Once attached, browser_snapshot will populate \
                 frame_tree with frame_ids you can pass here."
            )));
        }
    };

    let snapshot = supervisor.snapshot();
    let frame_tree = &snapshot.frame_tree;

    let mut frame_info: Option<Value> = None;
    if let Some(top) = frame_tree.get("top")
        && top.get("frame_id").and_then(Value::as_str) == Some(frame_id)
    {
        frame_info = Some(top.clone());
    }
    if frame_info.is_none()
        && let Some(children) = frame_tree.get("children").and_then(Value::as_array)
    {
        for child in children {
            if child.get("frame_id").and_then(Value::as_str) == Some(frame_id) {
                frame_info = Some(child.clone());
                break;
            }
        }
    }

    let frame_info = match frame_info {
        Some(info) => info,
        None => {
            return FrameLookup::NotFound(tool_error(format!(
                "frame_id {frame_id:?} not found in supervisor state. \
                 Call browser_snapshot to see current frame_tree."
            )));
        }
    };

    let child_sid = frame_info
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());

    match child_sid {
        Some(session_id) => FrameLookup::Found {
            session_id: session_id.to_string(),
        },
        None => FrameLookup::NotOopif(tool_error(format!(
            "frame_id {frame_id:?} is not an out-of-process iframe (no \
             dedicated CDP session). For same-origin iframes, use \
             `browser_cdp(method='Runtime.evaluate', params={{'expression': \
             \"document.querySelector('iframe').contentDocument.title\"}})` \
             at the top-level page instead."
        ))),
    }
}

/// Route a CDP call through the live supervisor session for an OOPIF frame.
///
/// Mirrors `_browser_cdp_via_supervisor`: resolves the frame's child session
/// and, on success, returns the success payload
/// `{"success": true, "method", "frame_id", "session_id", "result"}`.
pub fn browser_cdp_via_supervisor(
    router: &dyn SupervisorCdpRouter,
    task_id: &str,
    frame_id: &str,
    method: &str,
    params: Option<Value>,
    timeout: f64,
) -> String {
    match router.route(task_id, frame_id, method, params, timeout) {
        Ok(RouteOutcome::Error(error)) => error,
        Ok(RouteOutcome::Dispatched { session_id, result }) => {
            let payload = json!({
                "success": true,
                "method": method,
                "frame_id": frame_id,
                "session_id": session_id,
                "result": result,
            });
            payload.to_string()
        }
        Err(error) => error,
    }
}

// ---------------------------------------------------------------------------
// Public tool function (mirrors `browser_cdp`)
// ---------------------------------------------------------------------------

/// Send a raw CDP command. See [`CDP_DOCS_URL`] for method documentation.
///
/// - `method`: CDP method name, e.g. `"Target.getTargets"`.
/// - `params`: method-specific parameters; defaults to `{}`.
/// - `target_id`: optional target/tab ID for page-level methods.
/// - `frame_id`: optional OOPIF iframe id; routes via the supervisor.
/// - `timeout`: seconds to wait (clamped to [1, 300]).
/// - `task_id`: supervisor task id (defaults to `"default"` when `frame_id`
///   is set).
/// - `hermes_home`: base dir used to resolve `config.yaml`.
///
/// Returns a JSON string: `{"success": true, "method", "result"}` on success,
/// or `{"error": ...}` on failure.
#[allow(clippy::too_many_arguments)]
pub fn browser_cdp(
    method: &str,
    params: Option<Value>,
    target_id: Option<&str>,
    frame_id: Option<&str>,
    timeout: Option<f64>,
    task_id: Option<&str>,
    hermes_home: &std::path::Path,
    router: &dyn SupervisorCdpRouter,
) -> String {
    // --- Route iframe-scoped calls through the supervisor ---------------
    if let Some(frame_id) = frame_id.filter(|value| !value.is_empty()) {
        return browser_cdp_via_supervisor(
            router,
            task_id.filter(|value| !value.is_empty()).unwrap_or("default"),
            frame_id,
            method,
            params,
            timeout.unwrap_or(DEFAULT_CDP_TIMEOUT_SECS),
        );
    }

    if method.is_empty() {
        return tool_error_with(
            "'method' is required (e.g. 'Target.getTargets')",
            &[("cdp_docs", Value::String(CDP_DOCS_URL.to_string()))],
        );
    }

    let endpoint = resolve_cdp_endpoint(hermes_home);
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return tool_error_with(
            "No CDP endpoint is available. Run '/browser connect' to attach \
             to a running Chrome, or set 'browser.cdp_url' in config.yaml. \
             The Camofox backend is REST-only and does not expose CDP.",
            &[("cdp_docs", Value::String(CDP_DOCS_URL.to_string()))],
        );
    }

    if !endpoint.starts_with("ws://") && !endpoint.starts_with("wss://") {
        return tool_error(format!(
            "CDP endpoint is not a WebSocket URL: {endpoint:?}. \
             Expected ws://... or wss://... — the /browser connect \
             resolver should have rewritten this. Check that Chrome is \
             actually listening on the debug port."
        ));
    }

    // `params` must be an object/dict when provided.
    let call_params = match params {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::Object(map)) => Value::Object(map),
        Some(other) => {
            return tool_error(format!(
                "'params' must be an object/dict, got {}",
                json_type_name(&other)
            ));
        }
    };

    // Clamp timeout to [1, 300]; non-positive / unset falls back to default.
    let safe_timeout = match timeout {
        Some(value) if value.is_finite() && value > 0.0 => value,
        _ => DEFAULT_CDP_TIMEOUT_SECS,
    };
    let safe_timeout = safe_timeout.clamp(MIN_CDP_TIMEOUT_SECS, MAX_CDP_TIMEOUT_SECS);

    let result = match cdp_call(endpoint, method, call_params, target_id, safe_timeout) {
        Ok(value) => value,
        Err(CdpError::Timeout(message)) => {
            return tool_error_with(message, &[("method", Value::String(method.to_string()))]);
        }
        Err(CdpError::Runtime(message)) => {
            return tool_error_with(message, &[("method", Value::String(method.to_string()))]);
        }
        Err(CdpError::WebSocket(message)) => {
            return tool_error_with(
                format!(
                    "WebSocket error talking to CDP at {endpoint}: {message}. The \
                     browser may have disconnected — try '/browser connect' again."
                ),
                &[("method", Value::String(method.to_string()))],
            );
        }
    };

    let mut payload = json!({
        "success": true,
        "method": method,
        "result": result,
    });
    if let Some(target_id) = target_id.filter(|value| !value.is_empty())
        && let Some(object) = payload.as_object_mut()
    {
        object.insert("target_id".to_string(), Value::String(target_id.to_string()));
    }
    payload.to_string()
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Convenience wrapper that pulls `method`/`params`/`target_id`/`frame_id`/
/// `timeout` out of a tool-args JSON object the way the Python registry
/// handler does, and dispatches through [`browser_cdp`] with the default
/// supervisor router.
pub fn handle_browser_cdp(args: &Value, task_id: Option<&str>, hermes_home: &std::path::Path) -> String {
    let method = args.get("method").and_then(Value::as_str).unwrap_or("");
    let params = args.get("params").cloned();
    let target_id = args.get("target_id").and_then(Value::as_str);
    let frame_id = args.get("frame_id").and_then(Value::as_str);
    let timeout = args.get("timeout").and_then(Value::as_f64);
    browser_cdp(
        method,
        params,
        target_id,
        frame_id,
        timeout,
        task_id,
        hermes_home,
        &DefaultSupervisorRouter,
    )
}

// ---------------------------------------------------------------------------
// Availability check (mirrors `_browser_cdp_check`)
// ---------------------------------------------------------------------------

/// The tool is only offered when a static CDP URL is reachable right now —
/// `BROWSER_CDP_URL` env var or `browser.cdp_url` in `config.yaml`.
pub fn browser_cdp_check(hermes_home: &std::path::Path) -> bool {
    !resolve_cdp_endpoint(hermes_home).trim().is_empty()
}

// ---------------------------------------------------------------------------
// Tool schema (mirrors `BROWSER_CDP_SCHEMA`)
// ---------------------------------------------------------------------------

/// JSON schema for the `browser_cdp` tool, matching the Python definition.
pub fn browser_cdp_schema() -> Value {
    let description = format!(
        "Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for \
browser operations not covered by browser_navigate, browser_click, \
browser_console, etc.\n\n\
**Requires a reachable CDP endpoint.** Available when the user has \
run '/browser connect' to attach to a running Chrome, or when \
'browser.cdp_url' is set in config.yaml. Not currently wired up for \
cloud backends (Browserbase, Browser Use, Firecrawl) — those expose \
CDP per session but live-session routing is a follow-up. Camofox is \
REST-only and will never support CDP. If the tool is in your toolset \
at all, a CDP endpoint is already reachable.\n\n\
**CDP method reference:** {CDP_DOCS_URL} — use web_extract on a \
method's URL (e.g. '/tot/Page/#method-handleJavaScriptDialog') \
to look up parameters and return shape.\n\n\
**Common patterns:**\n\
- List tabs: method='Target.getTargets', params={{}}\n\
- Handle a native JS dialog: method='Page.handleJavaScriptDialog', \
params={{'accept': true, 'promptText': ''}}, target_id=<tabId>\n\
- Get all cookies: method='Network.getAllCookies', params={{}}\n\
- Eval in a specific tab: method='Runtime.evaluate', \
params={{'expression': '...', 'returnByValue': true}}, \
target_id=<tabId>\n\
- Set viewport for a tab: method='Emulation.setDeviceMetricsOverride', \
params={{'width': 1280, 'height': 720, 'deviceScaleFactor': 1, \
'mobile': false}}, target_id=<tabId>\n\n\
**Usage rules:**\n\
- Browser-level methods (Target.*, Browser.*, Storage.*): omit \
target_id and frame_id.\n\
- Page-level methods (Page.*, Runtime.*, DOM.*, Emulation.*, \
Network.* scoped to a tab): pass target_id from Target.getTargets.\n\
- **Cross-origin iframe scope** (Runtime.evaluate inside an OOPIF, \
Page.* targeting a frame target, etc.): pass frame_id from the \
browser_snapshot frame_tree output. This routes through the CDP \
supervisor's live connection — the only reliable way on \
Browserbase where stateless CDP calls hit signed-URL expiry.\n\
- Each stateless call (without frame_id) is independent — sessions \
and event subscriptions do not persist between calls. For stateful \
workflows, prefer the dedicated browser tools or use frame_id \
routing."
    );

    json!({
        "name": "browser_cdp",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "CDP method name, e.g. 'Target.getTargets', 'Runtime.evaluate', 'Page.handleJavaScriptDialog'."
                },
                "params": {
                    "type": "object",
                    "description": "Method-specific parameters as a JSON object. Omit or pass {} for methods that take no parameters.",
                    "properties": {},
                    "additionalProperties": true
                },
                "target_id": {
                    "type": "string",
                    "description": "Optional. Target/tab ID from Target.getTargets result (each entry's 'targetId'). Use for page-level methods at the top-level tab scope. Mutually exclusive with frame_id."
                },
                "frame_id": {
                    "type": "string",
                    "description": "Optional. Out-of-process iframe (OOPIF) frame_id from browser_snapshot.frame_tree.children[] where is_oopif=true. When set, routes the call through the CDP supervisor's live session for that iframe. Essential for Runtime.evaluate inside cross-origin iframes, especially on Browserbase where fresh per-call CDP connections can't keep up with signed URL rotation. For same-origin iframes, use parent contentWindow/contentDocument from Runtime.evaluate at the top-level page instead."
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 30, max 300).",
                    "default": 30
                }
            },
            "required": ["method"]
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    // Serialize tests that mutate BROWSER_CDP_URL.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn parse(json_str: &str) -> Value {
        serde_json::from_str(json_str).expect("valid JSON")
    }

    #[test]
    fn empty_method_errors() {
        let _guard = env_guard();
        let dir = TempDir::new().unwrap();
        let out = browser_cdp(
            "",
            None,
            None,
            None,
            None,
            None,
            dir.path(),
            &DefaultSupervisorRouter,
        );
        let value = parse(&out);
        assert!(
            value["error"]
                .as_str()
                .unwrap()
                .contains("'method' is required")
        );
        assert_eq!(value["cdp_docs"], CDP_DOCS_URL);
    }

    #[test]
    fn no_endpoint_errors() {
        let _guard = env_guard();
        let prev = std::env::var_os("BROWSER_CDP_URL");
        unsafe {
            std::env::remove_var("BROWSER_CDP_URL");
        }
        let dir = TempDir::new().unwrap();
        let out = browser_cdp(
            "Target.getTargets",
            None,
            None,
            None,
            None,
            None,
            dir.path(),
            &DefaultSupervisorRouter,
        );
        if let Some(value) = prev {
            unsafe {
                std::env::set_var("BROWSER_CDP_URL", value);
            }
        }
        let value = parse(&out);
        assert!(
            value["error"]
                .as_str()
                .unwrap()
                .contains("No CDP endpoint is available")
        );
        assert_eq!(value["cdp_docs"], CDP_DOCS_URL);
    }

    #[test]
    fn non_websocket_endpoint_errors() {
        let _guard = env_guard();
        let prev = std::env::var_os("BROWSER_CDP_URL");
        unsafe {
            std::env::set_var("BROWSER_CDP_URL", "http://localhost:9222");
        }
        let dir = TempDir::new().unwrap();
        let out = browser_cdp(
            "Target.getTargets",
            None,
            None,
            None,
            None,
            None,
            dir.path(),
            &DefaultSupervisorRouter,
        );
        match prev {
            Some(value) => unsafe { std::env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { std::env::remove_var("BROWSER_CDP_URL") },
        }
        let value = parse(&out);
        assert!(
            value["error"]
                .as_str()
                .unwrap()
                .contains("not a WebSocket URL")
        );
    }

    #[test]
    fn non_object_params_errors() {
        let _guard = env_guard();
        let prev = std::env::var_os("BROWSER_CDP_URL");
        unsafe {
            std::env::set_var("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/browser/x");
        }
        let dir = TempDir::new().unwrap();
        let out = browser_cdp(
            "Target.getTargets",
            Some(json!("not-an-object")),
            None,
            None,
            None,
            None,
            dir.path(),
            &DefaultSupervisorRouter,
        );
        match prev {
            Some(value) => unsafe { std::env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { std::env::remove_var("BROWSER_CDP_URL") },
        }
        let value = parse(&out);
        assert!(value["error"].as_str().unwrap().contains("must be an object"));
        assert!(value["error"].as_str().unwrap().contains("str"));
    }

    #[test]
    fn config_yaml_endpoint_resolves() {
        let _guard = env_guard();
        let prev = std::env::var_os("BROWSER_CDP_URL");
        unsafe {
            std::env::remove_var("BROWSER_CDP_URL");
        }
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("config.yaml"),
            "browser:\n  cdp_url: ws://127.0.0.1:9222/devtools/browser/abc\n",
        )
        .unwrap();
        let resolved = resolve_cdp_endpoint(dir.path());
        if let Some(value) = prev {
            unsafe {
                std::env::set_var("BROWSER_CDP_URL", value);
            }
        }
        assert_eq!(resolved, "ws://127.0.0.1:9222/devtools/browser/abc");
    }

    #[test]
    fn env_overrides_config() {
        let _guard = env_guard();
        let prev = std::env::var_os("BROWSER_CDP_URL");
        unsafe {
            std::env::set_var("BROWSER_CDP_URL", "  ws://env-wins/x  ");
        }
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("config.yaml"),
            "browser:\n  cdp_url: ws://config-loses/y\n",
        )
        .unwrap();
        let resolved = resolve_cdp_endpoint(dir.path());
        match prev {
            Some(value) => unsafe { std::env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { std::env::remove_var("BROWSER_CDP_URL") },
        }
        // Env var is trimmed.
        assert_eq!(resolved, "ws://env-wins/x");
    }

    #[test]
    fn frame_id_routes_to_supervisor_not_attached() {
        let _guard = env_guard();
        // No supervisor registered for this random task id -> NotAttached.
        let dir = TempDir::new().unwrap();
        let out = browser_cdp(
            "Runtime.evaluate",
            None,
            None,
            Some("frame-xyz"),
            None,
            Some("nonexistent-task-12345"),
            dir.path(),
            &DefaultSupervisorRouter,
        );
        let value = parse(&out);
        let error = value["error"].as_str().unwrap();
        assert!(error.contains("No CDP supervisor is attached"));
        assert!(error.contains("nonexistent-task-12345"));
    }

    #[test]
    fn timeout_clamped_and_default() {
        // Indirectly verify clamp via the public constants and logic.
        let clamp = |t: f64| t.clamp(MIN_CDP_TIMEOUT_SECS, MAX_CDP_TIMEOUT_SECS);
        assert_eq!(clamp(0.5), 1.0);
        assert_eq!(clamp(500.0), 300.0);
        assert_eq!(clamp(30.0), 30.0);
    }

    #[test]
    fn schema_shape() {
        let schema = browser_cdp_schema();
        assert_eq!(schema["name"], "browser_cdp");
        assert_eq!(schema["parameters"]["required"][0], "method");
        assert_eq!(schema["parameters"]["properties"]["timeout"]["default"], 30);
        assert!(
            schema["description"]
                .as_str()
                .unwrap()
                .contains(CDP_DOCS_URL)
        );
    }

    #[test]
    fn cdp_error_message_accessor() {
        assert_eq!(CdpError::Timeout("t".into()).message(), "t");
        assert_eq!(CdpError::Runtime("r".into()).message(), "r");
        assert_eq!(CdpError::WebSocket("w".into()).message(), "w");
    }
}
