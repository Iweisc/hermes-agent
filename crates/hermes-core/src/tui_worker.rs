//! Internal worker entrypoint for the `tui_gateway` JSON-RPC server.
//!
//! Faithful native Rust port of `tui_gateway/worker.py`.
//!
//! The worker is a thin, restricted JSON-RPC loop: it reads newline-delimited
//! JSON-RPC request frames off an input stream, validates that the requested
//! `method` is in a small allow-list of methods that are safe to run on the
//! internal worker, and dispatches each allowed request to a handler. Replies
//! are written back through a [`Transport`] (the gateway's `write_json` sink).
//!
//! ## Relationship to the gateway
//!
//! In Python this module imports `tui_gateway.server.{dispatch, write_json}`
//! and `tui_gateway.transport.TeeTransport`. In this crate the transport layer
//! already lives in [`crate::tui_transport`] (`Transport`, `TeeTransport`,
//! `StdioTransport`, `current_transport`). The `dispatch` and `write_json`
//! functions live in the (not-yet-ported) gateway server module, so this port
//! takes the dispatcher as an injected closure and resolves the write sink the
//! same way `server.write_json` does:
//!
//! ```text
//! (current_transport() or _stdio_transport).write(obj)
//! ```
//!
//! ## Behaviour parity with `worker.py`
//!
//! * `_ALLOWED_METHODS` — reproduced exactly as [`ALLOWED_METHODS`] /
//!   [`is_allowed_method`].
//! * Blank lines (after trimming) are skipped.
//! * A line that is not valid JSON produces a `-32700` "parse error" reply with
//!   `id: null` and the loop continues — unless the write reports the peer is
//!   gone, in which case the worker exits cleanly (`sys.exit(0)`).
//! * A request whose `method` is not in the allow-list produces a `-32601`
//!   `internal worker method not allowed: <method>` reply, echoing the request
//!   `id`; same peer-gone exit semantics.
//! * Otherwise `dispatch(req)` runs; if it returns a response and the write
//!   reports the peer is gone, the worker exits cleanly.
//!
//! The sidecar publisher install (`HERMES_TUI_SIDECAR_URL` →
//! `TeeTransport(stdio, WsPublisherTransport(url))`) and MCP discovery
//! (`_discover_mcp`) are startup side effects. The network publisher transport
//! is not portable inline here, so [`build_stdio_transport`] takes an optional
//! sidecar transport to tee onto, mirroring the wiring exactly.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::tui_transport::{
    current_transport, StdioTransport, TeeTransport, Transport, WriteResult,
};

// ---------------------------------------------------------------------------
// Allowed methods (mirror of `_ALLOWED_METHODS`)
// ---------------------------------------------------------------------------

/// The frozen set of JSON-RPC methods the internal worker is allowed to run.
///
/// Mirrors `_ALLOWED_METHODS` from `worker.py` exactly. Any method not in this
/// list is rejected with a `-32601` error before it ever reaches the
/// dispatcher.
pub static ALLOWED_METHODS: &[&str] = &[
    "agents.list",
    "browser.manage",
    "cli.exec",
    "clarify.respond",
    "config.get",
    "config.set",
    "cron.manage",
    "delegation.pause",
    "delegation.status",
    "image.attach",
    "model.disconnect",
    "model.options",
    "model.save_key",
    "process.stop",
    "prompt.submit",
    "reload.env",
    "reload.mcp",
    "rollback.diff",
    "rollback.list",
    "rollback.restore",
    "secret.respond",
    "session.close",
    "session.compress",
    "session.interrupt",
    "session.resume",
    "session.steer",
    "shell.exec",
    "skills.manage",
    "skills.reload",
    "subagent.interrupt",
    "sudo.respond",
    "terminal.resize",
    "tools.configure",
    "tools.list",
    "tools.show",
    "toolsets.list",
    "voice.record",
    "voice.toggle",
    "voice.tts",
];

/// Returns `true` when `method` is in the worker allow-list.
///
/// Mirrors `method not in _ALLOWED_METHODS` (negated). The empty string — the
/// Python default for a missing `method` key — is not in the set, so it is
/// rejected just like in `worker.py`.
pub fn is_allowed_method(method: &str) -> bool {
    ALLOWED_METHODS.contains(&method)
}

// ---------------------------------------------------------------------------
// write_json (mirror of server.write_json resolution)
// ---------------------------------------------------------------------------

/// Resolve the active write sink and emit one JSON frame.
///
/// Mirrors `server.write_json`:
///
/// ```text
/// return (current_transport() or _stdio_transport).write(obj)
/// ```
///
/// `fallback` is the module-level stdio transport (`_stdio_transport`) used
/// when nothing is bound on the current-transport slot.
pub fn write_json(obj: &Value, fallback: &Arc<dyn Transport>) -> WriteResult {
    match current_transport() {
        Some(t) => t.write(obj),
        None => fallback.write(obj),
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC error frames (mirror of the inline dicts in worker.py)
// ---------------------------------------------------------------------------

/// Build the `-32700` parse-error frame (`id: null`).
///
/// Mirrors the dict written when `json.loads(line)` raises `JSONDecodeError`.
pub fn parse_error_frame() -> Value {
    json!({
        "jsonrpc": "2.0",
        "error": {"code": -32700, "message": "parse error"},
        "id": Value::Null,
    })
}

/// Build the `-32601` method-not-allowed frame, echoing the request `id`.
///
/// Mirrors the dict written when `method not in _ALLOWED_METHODS`. `req_id` is
/// `req.get("id")` (may be `Value::Null`).
pub fn method_not_allowed_frame(method: &str, req_id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": format!("internal worker method not allowed: {method}"),
        },
        "id": req_id,
    })
}

// ---------------------------------------------------------------------------
// Startup wiring (mirror of _install_sidecar_publisher)
// ---------------------------------------------------------------------------

/// Build the module-level stdio transport, optionally teed onto a sidecar
/// publisher transport.
///
/// Mirrors `_install_sidecar_publisher`:
///
/// ```text
/// url = os.environ.get("HERMES_TUI_SIDECAR_URL")
/// if not url: return
/// server._stdio_transport = TeeTransport(server._stdio_transport, WsPublisherTransport(url))
/// ```
///
/// The `WsPublisherTransport` construction is not portable inline (it needs the
/// gateway's WebSocket publisher), so the caller passes the already-constructed
/// sidecar transport. When `HERMES_TUI_SIDECAR_URL` is unset/empty the sidecar
/// is ignored and the bare stdio transport is returned.
pub fn build_stdio_transport(sidecar: Option<Arc<dyn Transport>>) -> Arc<dyn Transport> {
    let base: Arc<dyn Transport> = Arc::new(StdioTransport::real_stdout());
    let url = std::env::var("HERMES_TUI_SIDECAR_URL").unwrap_or_default();
    if url.is_empty() {
        return base;
    }
    match sidecar {
        Some(secondary) => Arc::new(TeeTransport::new(base, vec![secondary])),
        None => base,
    }
}

// ---------------------------------------------------------------------------
// Main loop (mirror of main())
// ---------------------------------------------------------------------------

/// Outcome of processing a single request line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineOutcome {
    /// The line was processed (or skipped); keep reading.
    Continue,
    /// A write reported the peer is gone; the worker should exit cleanly
    /// (`sys.exit(0)`).
    Exit,
}

/// Process exactly one raw input line, mirroring one iteration of the
/// `for raw in sys.stdin` loop body in `worker.py`.
///
/// * Trims the line; blank → [`LineOutcome::Continue`] (skipped).
/// * Invalid JSON → write [`parse_error_frame`]; exit on peer-gone.
/// * `method` not in allow-list → write [`method_not_allowed_frame`]; exit on
///   peer-gone.
/// * Otherwise call `dispatch(&req)`; if it returns `Some(resp)` write it and
///   exit on peer-gone.
///
/// `fallback` is the module-level stdio transport. `dispatch` is the gateway
/// dispatcher (`server.dispatch`), injected here.
///
/// Returns `Err` only on a *real* (non-peer-gone) transport error, matching the
/// Python split where peer-gone produces a clean exit but other errors
/// propagate to the crash log.
pub fn process_line<F>(
    raw: &str,
    fallback: &Arc<dyn Transport>,
    dispatch: &F,
) -> Result<LineOutcome, crate::tui_transport::TransportError>
where
    F: Fn(&Value) -> Option<Value>,
{
    let line = raw.trim();
    if line.is_empty() {
        return Ok(LineOutcome::Continue);
    }

    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            // Mirror `if not write_json(parse_error): sys.exit(0)`.
            if !write_json(&parse_error_frame(), fallback)? {
                return Ok(LineOutcome::Exit);
            }
            return Ok(LineOutcome::Continue);
        }
    };

    // `req.get("method", "")` — missing/non-string method becomes "".
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    if !is_allowed_method(method) {
        let req_id = req.get("id").cloned().unwrap_or(Value::Null);
        let frame = method_not_allowed_frame(method, req_id);
        if !write_json(&frame, fallback)? {
            return Ok(LineOutcome::Exit);
        }
        return Ok(LineOutcome::Continue);
    }

    // `resp = dispatch(req)`
    if let Some(resp) = dispatch(&req) {
        // `if resp is not None and not write_json(resp): sys.exit(0)`
        if !write_json(&resp, fallback)? {
            return Ok(LineOutcome::Exit);
        }
    }
    Ok(LineOutcome::Continue)
}

/// Run the worker loop over `lines`, dispatching allowed requests.
///
/// Faithful port of `main()`'s body (`for raw in sys.stdin: ...`), minus the
/// startup side effects (`_install_sidecar_publisher` / `_discover_mcp`), which
/// the caller wires explicitly via [`build_stdio_transport`] and the
/// dispatcher.
///
/// `lines` yields the raw input lines (e.g. `BufRead::lines()` results,
/// unwrapped). `fallback` is the module-level stdio transport. `dispatch` is
/// the gateway dispatcher.
///
/// Returns `Ok(())` on a clean shutdown (input exhausted or peer gone) and
/// `Err` on a real transport error.
pub fn run_worker<I, F>(
    lines: I,
    fallback: &Arc<dyn Transport>,
    dispatch: &F,
) -> Result<(), crate::tui_transport::TransportError>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
    F: Fn(&Value) -> Option<Value>,
{
    for raw in lines {
        match process_line(raw.as_ref(), fallback, dispatch)? {
            LineOutcome::Continue => {}
            LineOutcome::Exit => return Ok(()),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records every frame written, and can be configured to report the peer is
    /// gone (returns `Ok(false)`) after a given number of writes.
    struct RecordingTransport {
        frames: Mutex<Vec<Value>>,
        peer_gone_after: Option<usize>,
    }

    impl RecordingTransport {
        fn new() -> Self {
            RecordingTransport {
                frames: Mutex::new(Vec::new()),
                peer_gone_after: None,
            }
        }
        fn peer_gone() -> Self {
            RecordingTransport {
                frames: Mutex::new(Vec::new()),
                peer_gone_after: Some(0),
            }
        }
        fn count(&self) -> usize {
            self.frames.lock().unwrap().len()
        }
        fn last(&self) -> Option<Value> {
            self.frames.lock().unwrap().last().cloned()
        }
    }

    impl Transport for RecordingTransport {
        fn write(&self, obj: &Value) -> WriteResult {
            let mut frames = self.frames.lock().unwrap();
            let alive = match self.peer_gone_after {
                Some(n) => frames.len() < n,
                None => true,
            };
            frames.push(obj.clone());
            Ok(alive)
        }
    }

    fn noop_dispatch(_req: &Value) -> Option<Value> {
        None
    }

    #[test]
    fn allowed_methods_match_python_set() {
        // Spot-check representative entries and the count (39 methods).
        assert_eq!(ALLOWED_METHODS.len(), 39);
        assert!(is_allowed_method("prompt.submit"));
        assert!(is_allowed_method("voice.tts"));
        assert!(is_allowed_method("agents.list"));
        assert!(!is_allowed_method("not.a.method"));
        // Empty string (missing `method`) is rejected.
        assert!(!is_allowed_method(""));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let out = process_line("   \t  ", &t, &noop_dispatch).unwrap();
        assert_eq!(out, LineOutcome::Continue);
        // No frame should have been written.
        assert_eq!(rec.count(), 0);
    }

    #[test]
    fn invalid_json_writes_parse_error() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let out = process_line("{not json", &t, &noop_dispatch).unwrap();
        assert_eq!(out, LineOutcome::Continue);
        assert_eq!(rec.count(), 1);
        let frame = rec.last().unwrap();
        assert_eq!(frame["error"]["code"], json!(-32700));
        assert_eq!(frame["error"]["message"], json!("parse error"));
        assert_eq!(frame["id"], Value::Null);
        assert_eq!(frame["jsonrpc"], json!("2.0"));
    }

    #[test]
    fn invalid_json_peer_gone_exits() {
        let t: Arc<dyn Transport> = Arc::new(RecordingTransport::peer_gone());
        let out = process_line("garbage", &t, &noop_dispatch).unwrap();
        assert_eq!(out, LineOutcome::Exit);
    }

    #[test]
    fn disallowed_method_writes_32601_and_echoes_id() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let line = r#"{"jsonrpc":"2.0","method":"danger.run","id":42}"#;
        let out = process_line(line, &t, &noop_dispatch).unwrap();
        assert_eq!(out, LineOutcome::Continue);
        let frame = rec.last().unwrap();
        assert_eq!(frame["error"]["code"], json!(-32601));
        assert_eq!(
            frame["error"]["message"],
            json!("internal worker method not allowed: danger.run")
        );
        assert_eq!(frame["id"], json!(42));
    }

    #[test]
    fn missing_method_is_rejected_with_empty_name() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let line = r#"{"jsonrpc":"2.0","id":7}"#;
        process_line(line, &t, &noop_dispatch).unwrap();
        let frame = rec.last().unwrap();
        assert_eq!(
            frame["error"]["message"],
            json!("internal worker method not allowed: ")
        );
        assert_eq!(frame["id"], json!(7));
    }

    #[test]
    fn allowed_method_dispatches_and_writes_response() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let dispatch = |req: &Value| -> Option<Value> {
            Some(json!({
                "jsonrpc": "2.0",
                "result": "ok",
                "id": req.get("id").cloned().unwrap_or(Value::Null),
            }))
        };
        let line = r#"{"jsonrpc":"2.0","method":"tools.list","id":1}"#;
        let out = process_line(line, &t, &dispatch).unwrap();
        assert_eq!(out, LineOutcome::Continue);
        let frame = rec.last().unwrap();
        assert_eq!(frame["result"], json!("ok"));
        assert_eq!(frame["id"], json!(1));
    }

    #[test]
    fn allowed_method_with_none_response_writes_nothing() {
        let rec = Arc::new(RecordingTransport::new());
        let t: Arc<dyn Transport> = rec.clone();
        let line = r#"{"jsonrpc":"2.0","method":"session.interrupt","id":1}"#;
        let out = process_line(line, &t, &noop_dispatch).unwrap();
        assert_eq!(out, LineOutcome::Continue);
        assert_eq!(rec.count(), 0);
    }

    #[test]
    fn run_worker_processes_multiple_lines_and_exits_on_peer_gone() {
        let rec = Arc::new(RecordingTransport::peer_gone());
        let t: Arc<dyn Transport> = rec.clone();
        let lines = vec![
            "   ",                               // skipped
            r#"{"method":"bad.method","id":1}"#, // triggers write -> peer gone -> exit
            r#"{"method":"tools.list","id":2}"#, // never reached
        ];
        run_worker(lines, &t, &noop_dispatch).unwrap();
        // Exactly one write attempted before exiting.
        assert_eq!(rec.count(), 1);
    }

    #[test]
    fn write_json_prefers_bound_transport() {
        let fallback = Arc::new(RecordingTransport::new());
        let fb: Arc<dyn Transport> = fallback.clone();
        let bound = Arc::new(RecordingTransport::new());
        let bound_dyn: Arc<dyn Transport> = bound.clone();

        let token = crate::tui_transport::bind_transport(Some(bound_dyn));
        let _ = write_json(&json!({"hello": "world"}), &fb);
        crate::tui_transport::reset_transport(token);

        assert_eq!(bound.count(), 1);
        assert_eq!(fallback.count(), 0);
    }

    #[test]
    fn build_stdio_transport_without_sidecar_url() {
        unsafe {
            std::env::remove_var("HERMES_TUI_SIDECAR_URL");
        }
        // Should return a bare stdio transport without panicking.
        let _t = build_stdio_transport(None);
    }
}
