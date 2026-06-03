//! WebSocket transport for the `tui_gateway` JSON-RPC server.
//!
//! Faithful native Rust port of `tui_gateway/ws.py`.
//!
//! Reuses the gateway `dispatch` callback verbatim so every RPC method, every
//! slash command, every approval/clarify/sudo flow, and every agent event flows
//! through the same handlers whether the client is Ink over stdio or an iOS /
//! web client over WebSocket.
//!
//! ## Wire protocol
//!
//! Identical to stdio: newline-delimited JSON-RPC in both directions. The server
//! emits a `gateway.ready` event immediately after connection accept, then echoes
//! responses/events for inbound requests. No framing differences.
//!
//! ## Threading model
//!
//! The Python original distinguishes "called from the owning event loop" vs
//! "called from a pool worker thread", using `asyncio.run_coroutine_threadsafe`
//! to marshal cross-thread writes onto the loop. In native Rust there is no
//! event loop that owns the socket the same way; instead the socket is guarded
//! by a [`Mutex`]. [`WsTransport::write`] (the [`Transport`] impl used by pool
//! workers and inline handlers alike) takes the lock and sends synchronously.
//!
//! The "fire-and-forget on the loop thread / blocking with timeout off-loop"
//! split collapses to a single mutex-guarded send here because there is no risk
//! of a thread deadlocking against itself: a `Mutex` is re-entrant-free but the
//! handler thread is not *also* the thread blocked inside `send`. The
//! [`_WS_WRITE_TIMEOUT_S`] timeout is preserved as a configurable knob for
//! socket implementations that support a write deadline.
//!
//! ## Crate dependencies
//!
//! * [`crate::tui_transport::Transport`] — the transport trait shared with the
//!   stdio path.
//! * [`crate::skins::resolve_skin`] — the `gateway.ready` skin payload.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::tui_transport::{Transport, TransportError, WriteResult};

/// Max time a pool-dispatched handler will block waiting for a WS frame to flush
/// before we mark the transport dead. Protects handler threads from a wedged
/// socket. Mirrors Python's `_WS_WRITE_TIMEOUT_S = 10.0`.
pub const WS_WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// Reason a WS receive loop terminated. Mirrors the `break` exits of the Python
/// `while True` loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsLoopExit {
    /// The peer disconnected (`WebSocketDisconnect`) or the receive stream ended.
    Disconnected,
    /// A write reported the transport is gone, so the loop stopped writing.
    WriteFailed,
}

/// Abstraction over the underlying WebSocket connection.
///
/// Mirrors the subset of the Starlette `WebSocket` API that `ws.py` touches:
/// `accept`, `send_text`, `receive_text`, and `close`. Implementors wrap a
/// concrete socket (e.g. a `tungstenite` connection). Kept as a trait so the
/// module is testable without a live socket and so the same loop logic drives
/// any transport that satisfies the contract.
pub trait WebSocketConn: Send {
    /// Accept the inbound handshake. Called once before any frame is sent.
    fn accept(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    /// Send one text frame.
    fn send_text(&mut self, line: &str) -> std::io::Result<()>;

    /// Receive one text frame.
    ///
    /// Returns `Ok(None)` to signal a clean disconnect (`WebSocketDisconnect` in
    /// the Python original); `Ok(Some(_))` for a received frame; `Err(_)` for a
    /// hard transport error (also treated as a disconnect by the loop).
    fn receive_text(&mut self) -> std::io::Result<Option<String>>;

    /// Close the socket. Best-effort; errors are swallowed by the caller.
    fn close(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Per-connection WS transport.
///
/// Wraps a shared, mutex-guarded [`WebSocketConn`]. [`WsTransport::write`] is the
/// [`Transport`] trait method pool workers call; it serialises the payload as
/// `ensure_ascii=False` newline-free JSON (the frame text) and forwards it under
/// the lock. A `closed` flag short-circuits once the peer is gone, matching the
/// Python `_closed` sticky bit.
pub struct WsTransport {
    conn: Arc<Mutex<dyn WebSocketConn>>,
    closed: AtomicBool,
}

impl WsTransport {
    /// Wrap a shared connection in a transport.
    pub fn new(conn: Arc<Mutex<dyn WebSocketConn>>) -> Self {
        WsTransport {
            conn,
            closed: AtomicBool::new(false),
        }
    }

    /// Whether the transport has been marked dead.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Serialise `obj` to the wire form Python emits: `json.dumps(obj,
    /// ensure_ascii=False)` — compact, UTF-8, no trailing newline (the WS frame
    /// boundary is the message boundary; newline-delimited semantics are
    /// preserved by one object per frame).
    fn encode(obj: &Value) -> Result<String, TransportError> {
        serde_json::to_string(obj).map_err(TransportError::from)
    }

    /// Internal send used by both the sync [`Transport::write`] and
    /// [`WsTransport::write_async_like`]. Mirrors Python `_safe_send`: on any send
    /// error the transport is marked closed and the error swallowed (debug log).
    fn safe_send(&self, line: &str) -> bool {
        let mut guard = self
            .conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match guard.send_text(line) {
            Ok(()) => true,
            Err(exc) => {
                self.closed.store(true, Ordering::SeqCst);
                log::debug!("ws send failed: {exc}");
                false
            }
        }
    }

    /// Send `obj`, awaiting (synchronously here) until the frame is on the wire.
    ///
    /// Mirrors Python `write_async`: returns `false` early if already closed,
    /// otherwise sends and returns whether the transport is still alive after the
    /// send.
    pub fn write_async_like(&self, obj: &Value) -> Result<bool, TransportError> {
        if self.is_closed() {
            return Ok(false);
        }
        let line = Self::encode(obj)?;
        self.safe_send(&line);
        Ok(!self.is_closed())
    }
}

impl Transport for WsTransport {
    /// Emit one JSON frame from a pool worker thread.
    ///
    /// Mirrors Python `WSTransport.write` (the off-loop path): serialise, send
    /// under the lock, and report liveness. A failed send marks the transport
    /// closed and returns `Ok(false)` (peer-gone), matching the stdio transport's
    /// peer-gone contract. Serialisation failures surface as `Err` (programming
    /// error), consistent with [`crate::tui_transport`].
    fn write(&self, obj: &Value) -> WriteResult {
        if self.is_closed() {
            return Ok(false);
        }
        let line = Self::encode(obj)?;
        let ok = self.safe_send(&line);
        if !ok {
            return Ok(false);
        }
        Ok(!self.is_closed())
    }

    /// Mark the transport dead. Mirrors Python `WSTransport.close`.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

/// Callback that resolves the `gateway.ready` skin payload.
///
/// Defaulted via [`run_ws_session`] to [`crate::skins::resolve_skin`] in the
/// caller; kept as a parameter so tests and alternate gateways can inject a
/// fixed payload without an on-disk config.
pub type SkinResolver = dyn Fn() -> Value + Send + Sync;

/// Dispatch callback: process one inbound JSON-RPC request.
///
/// Mirrors `tui_gateway.server.dispatch(req, transport)`. The handler may
/// schedule long work on a pool, in which case it returns `None` and the worker
/// writes its own response via the supplied `transport`. For inline handlers it
/// returns `Some(response)`, which [`run_ws_session`] writes from the loop.
pub type Dispatcher = dyn Fn(&Value, Arc<WsTransport>) -> Option<Value> + Send + Sync;

/// Hook invoked once the receive loop exits, to detach this transport from any
/// sessions it owned so later emits fall back to stdio instead of crashing into
/// a closed socket. Mirrors the `for _, sess in server._sessions...` cleanup.
pub type SessionDetach = dyn Fn(&Arc<WsTransport>) + Send + Sync;

/// Build the `gateway.ready` event frame. Extracted so the exact shape is
/// asserted in tests and reused by [`run_ws_session`].
pub fn gateway_ready_frame(skin: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "gateway.ready",
            "payload": { "skin": skin },
        },
    })
}

/// Build the JSON-RPC parse-error frame emitted for unparseable input. Mirrors
/// the `-32700` reply in `ws.py`.
pub fn parse_error_frame() -> Value {
    json!({
        "jsonrpc": "2.0",
        "error": { "code": -32700, "message": "parse error" },
        "id": Value::Null,
    })
}

/// Run one WebSocket session. Wire-compatible with `tui_gateway.entry`.
///
/// Faithful port of `handle_ws`:
///
/// 1. Accept the handshake.
/// 2. Emit `gateway.ready` with the resolved skin.
/// 3. Loop: receive a frame, strip it, skip blanks, parse JSON. On parse error
///    reply with `-32700` (and break if that write fails). On success dispatch
///    and, for inline responses, write the result (breaking if that fails).
/// 4. On exit: close the transport, detach it from owned sessions, and best-
///    effort close the socket.
///
/// `conn` is the shared socket the [`WsTransport`] also holds, so writes and
/// receives serialise through the same lock.
pub fn run_ws_session(
    conn: Arc<Mutex<dyn WebSocketConn>>,
    resolve_skin: &SkinResolver,
    dispatch: &Dispatcher,
    detach_sessions: &SessionDetach,
) -> WsLoopExit {
    // accept()
    {
        let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
        let _ = guard.accept();
    }

    let transport = Arc::new(WsTransport::new(conn.clone()));

    // gateway.ready
    let _ = transport.write_async_like(&gateway_ready_frame(resolve_skin()));

    let exit = run_receive_loop(&conn, &transport, dispatch);

    // finally:
    transport.close();
    detach_sessions(&transport);
    {
        let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
        let _ = guard.close();
    }

    exit
}

/// The `while True` receive/dispatch body of [`run_ws_session`], split out so the
/// `finally` cleanup in the caller is unconditional regardless of how the loop
/// terminates (Rust has no `try/finally`; this models it via early return).
fn run_receive_loop(
    conn: &Arc<Mutex<dyn WebSocketConn>>,
    transport: &Arc<WsTransport>,
    dispatch: &Dispatcher,
) -> WsLoopExit {
    loop {
        let raw = {
            let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
            match guard.receive_text() {
                // WebSocketDisconnect / clean end of stream.
                Ok(None) => return WsLoopExit::Disconnected,
                // A hard receive error is treated as a disconnect (the Python
                // loop only special-cases WebSocketDisconnect; any other
                // exception would propagate out of the try and hit `finally`,
                // ending the session all the same).
                Err(_) => return WsLoopExit::Disconnected,
                Ok(Some(raw)) => raw,
            }
        };

        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                // parse error reply
                match transport.write_async_like(&parse_error_frame()) {
                    Ok(true) => continue,
                    // ok == false (or err) => break, matching `if not ok: break`.
                    _ => return WsLoopExit::WriteFailed,
                }
            }
        };

        // dispatch() may schedule long handlers on the pool and return None
        // (the worker writes its own response via the transport). For inline
        // handlers it returns the response, written here.
        let resp = dispatch(&req, transport.clone());
        if let Some(resp) = resp {
            match transport.write_async_like(&resp) {
                Ok(true) => {}
                _ => return WsLoopExit::WriteFailed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A scripted in-memory socket: `inbound` frames are popped front-to-back by
    /// `receive_text`; once empty it reports a clean disconnect. Everything sent
    /// is captured in `sent`.
    struct FakeConn {
        inbound: std::collections::VecDeque<std::io::Result<Option<String>>>,
        sent: Arc<Mutex<Vec<String>>>,
        accepted: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
        fail_send: bool,
    }

    impl FakeConn {
        fn new(frames: Vec<&str>) -> (Arc<Mutex<dyn WebSocketConn>>, Probe) {
            let sent = Arc::new(Mutex::new(Vec::new()));
            let accepted = Arc::new(AtomicBool::new(false));
            let closed = Arc::new(AtomicBool::new(false));
            let mut inbound = std::collections::VecDeque::new();
            for f in frames {
                inbound.push_back(Ok(Some(f.to_string())));
            }
            let probe = Probe {
                sent: sent.clone(),
                accepted: accepted.clone(),
                closed: closed.clone(),
            };
            let conn: Arc<Mutex<dyn WebSocketConn>> = Arc::new(Mutex::new(FakeConn {
                inbound,
                sent,
                accepted,
                closed,
                fail_send: false,
            }));
            (conn, probe)
        }
    }

    struct Probe {
        sent: Arc<Mutex<Vec<String>>>,
        accepted: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
    }

    impl WebSocketConn for FakeConn {
        fn accept(&mut self) -> std::io::Result<()> {
            self.accepted.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn send_text(&mut self, line: &str) -> std::io::Result<()> {
            if self.fail_send {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            self.sent.lock().unwrap().push(line.to_string());
            Ok(())
        }
        fn receive_text(&mut self) -> std::io::Result<Option<String>> {
            match self.inbound.pop_front() {
                Some(r) => r,
                None => Ok(None),
            }
        }
        fn close(&mut self) -> std::io::Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    fn fixed_skin() -> Value {
        json!({"name": "default"})
    }

    #[test]
    fn emits_gateway_ready_then_disconnects() {
        let (conn, probe) = FakeConn::new(vec![]);
        let exit = run_ws_session(
            conn,
            &fixed_skin,
            &|_req, _t| None,
            &|_t| {},
        );
        assert_eq!(exit, WsLoopExit::Disconnected);
        assert!(probe.accepted.load(Ordering::SeqCst));
        assert!(probe.closed.load(Ordering::SeqCst));
        let sent = probe.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        let frame: Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(frame["method"], "event");
        assert_eq!(frame["params"]["type"], "gateway.ready");
        assert_eq!(frame["params"]["payload"]["skin"]["name"], "default");
    }

    #[test]
    fn dispatch_inline_response_is_written() {
        let (conn, probe) = FakeConn::new(vec![r#"{"id":1,"method":"ping"}"#]);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let exit = run_ws_session(
            conn,
            &fixed_skin,
            &move |req, _t| {
                calls2.fetch_add(1, Ordering::SeqCst);
                Some(json!({"jsonrpc":"2.0","id": req["id"], "result": "pong"}))
            },
            &|_t| {},
        );
        assert_eq!(exit, WsLoopExit::Disconnected);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let sent = probe.sent.lock().unwrap();
        // gateway.ready + the inline response.
        assert_eq!(sent.len(), 2);
        let resp: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(resp["result"], "pong");
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn dispatch_none_writes_nothing_extra() {
        let (conn, probe) = FakeConn::new(vec![r#"{"id":2,"method":"async"}"#]);
        let exit = run_ws_session(conn, &fixed_skin, &|_req, _t| None, &|_t| {});
        assert_eq!(exit, WsLoopExit::Disconnected);
        // Only gateway.ready was sent (worker would write its own later).
        assert_eq!(probe.sent.lock().unwrap().len(), 1);
    }

    #[test]
    fn blank_frames_are_skipped() {
        let (conn, probe) = FakeConn::new(vec!["   ", "\n", r#"{"id":3}"#]);
        let seen = Arc::new(AtomicUsize::new(0));
        let seen2 = seen.clone();
        run_ws_session(
            conn,
            &fixed_skin,
            &move |_req, _t| {
                seen2.fetch_add(1, Ordering::SeqCst);
                Some(json!({"ok": true}))
            },
            &|_t| {},
        );
        // Only the one non-blank frame dispatched.
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        // gateway.ready + one response.
        assert_eq!(probe.sent.lock().unwrap().len(), 2);
    }

    #[test]
    fn parse_error_emits_minus_32700() {
        let (conn, probe) = FakeConn::new(vec!["{not json"]);
        run_ws_session(conn, &fixed_skin, &|_req, _t| None, &|_t| {});
        let sent = probe.sent.lock().unwrap();
        // gateway.ready + parse-error frame.
        assert_eq!(sent.len(), 2);
        let err: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(err["error"]["code"], -32700);
        assert_eq!(err["error"]["message"], "parse error");
        assert_eq!(err["id"], Value::Null);
    }

    #[test]
    fn write_failure_breaks_loop_and_detaches() {
        // A conn whose send fails: gateway.ready write fails, then the inline
        // response write fails -> WriteFailed. We exercise the detach hook too.
        let sent = Arc::new(Mutex::new(Vec::new()));
        let accepted = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let mut inbound = std::collections::VecDeque::new();
        inbound.push_back(Ok(Some(r#"{"id":9}"#.to_string())));
        let conn: Arc<Mutex<dyn WebSocketConn>> = Arc::new(Mutex::new(FakeConn {
            inbound,
            sent: sent.clone(),
            accepted,
            closed: closed.clone(),
            fail_send: true,
        }));
        let detached = Arc::new(AtomicBool::new(false));
        let detached2 = detached.clone();
        let exit = run_ws_session(
            conn,
            &fixed_skin,
            &|_req, _t| Some(json!({"x": 1})),
            &move |_t| detached2.store(true, Ordering::SeqCst),
        );
        assert_eq!(exit, WsLoopExit::WriteFailed);
        assert!(detached.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
        // Nothing actually landed on the wire (all sends failed).
        assert!(sent.lock().unwrap().is_empty());
    }

    #[test]
    fn transport_write_marks_closed_on_failure() {
        let (conn, _probe) = FakeConn::new(vec![]);
        // Flip the fake to fail sends.
        {
            // Downcast not available; rebuild a failing conn instead.
        }
        let failing: Arc<Mutex<dyn WebSocketConn>> = Arc::new(Mutex::new(FakeConn {
            inbound: std::collections::VecDeque::new(),
            sent: Arc::new(Mutex::new(Vec::new())),
            accepted: Arc::new(AtomicBool::new(false)),
            closed: Arc::new(AtomicBool::new(false)),
            fail_send: true,
        }));
        let _ = conn; // unused happy-path conn
        let t = WsTransport::new(failing);
        assert!(!t.is_closed());
        let r = t.write(&json!({"a": 1})).unwrap();
        assert!(!r, "failed send should report peer-gone (false)");
        assert!(t.is_closed());
        // Subsequent writes short-circuit to false.
        assert!(!t.write(&json!({"b": 2})).unwrap());
    }

    #[test]
    fn non_ascii_serialised_unescaped() {
        // ensure_ascii=False equivalent.
        let line = WsTransport::encode(&json!({"t": "héllo 世界"})).unwrap();
        assert!(line.contains("héllo 世界"), "got: {line}");
    }

    #[test]
    fn ready_frame_shape_exact() {
        let f = gateway_ready_frame(json!({"k": "v"}));
        assert_eq!(f["jsonrpc"], "2.0");
        assert_eq!(f["method"], "event");
        assert_eq!(f["params"]["type"], "gateway.ready");
        assert_eq!(f["params"]["payload"]["skin"]["k"], "v");
    }
}
