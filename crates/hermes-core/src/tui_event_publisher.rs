//! Best-effort WebSocket publisher transport for the PTY-side gateway.
//!
//! The dashboard's `/api/pty` spawns `hermes --tui` as a child process, which
//! spawns its own `tui_gateway.entry`. Tool/reasoning/status events fire on
//! *that* gateway's transport — three processes removed from the dashboard
//! server itself. To surface them in the dashboard sidebar (`/api/events`),
//! the PTY-side gateway opens a back-WS to the dashboard at startup and
//! mirrors every emit through this transport.
//!
//! Wire protocol: newline-framed JSON dicts (the same shape the dispatcher
//! already passes to `write`). No JSON-RPC envelope here — the dashboard's
//! `/api/pub` endpoint just rebroadcasts the bytes verbatim to subscribers.
//!
//! Failure mode: silent. The agent loop must never block waiting for the
//! sidecar to drain. A dead WS short-circuits all subsequent writes. Actual
//! `send` calls run on a daemon thread so the tee transport's `write` returns
//! after enqueueing (best-effort; drop when the queue is full).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::Value;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message as WsMessage, WebSocket, client::IntoClientRequest, connect as ws_connect};

/// Mirror of the Python `_QUEUE_MAX = 256`.
const QUEUE_MAX: usize = 256;

/// Default connect timeout, matching the Python `connect_timeout: float = 2.0`.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

type Ws = WebSocket<MaybeTlsStream<std::net::TcpStream>>;

/// Items that travel down the drain channel.
///
/// `Line(String)` mirrors a JSON-encoded payload to send; `Stop` is the
/// sentinel that mirrors the Python `_DRAIN_STOP` object.
enum DrainItem {
    Line(String),
    Stop,
}

/// Best-effort WebSocket publisher transport.
///
/// Construct with [`WsPublisherTransport::connect`]. The transport spawns a
/// background drain thread that owns the live WebSocket; [`write`] simply
/// enqueues a JSON line and returns immediately.
///
/// [`write`]: WsPublisherTransport::write
pub struct WsPublisherTransport {
    url: String,
    dead: Arc<AtomicBool>,
    /// `Some` only when a live connection + drain worker exist.
    sender: Option<SyncSender<DrainItem>>,
    worker: Option<JoinHandle<()>>,
}

impl WsPublisherTransport {
    /// Open a back-WS to `url`, mirroring the Python constructor.
    ///
    /// On any connection failure the returned transport is "dead": [`write`]
    /// returns `false` for every call. This never errors — the failure mode is
    /// silent, exactly as in Python.
    ///
    /// [`write`]: WsPublisherTransport::write
    pub fn connect(url: impl Into<String>) -> Self {
        Self::connect_with_timeout(url, DEFAULT_CONNECT_TIMEOUT)
    }

    /// As [`connect`](Self::connect), but with an explicit connect timeout.
    ///
    /// Note: `tungstenite::connect` does not expose a fine-grained open
    /// timeout the way `websockets.sync.client.connect(open_timeout=...)` does;
    /// we apply the timeout to the underlying TCP read/write where the stream
    /// allows it. The behavioural contract (dead-on-failure) is preserved.
    pub fn connect_with_timeout(url: impl Into<String>, connect_timeout: Duration) -> Self {
        let url = url.into();
        let dead = Arc::new(AtomicBool::new(false));

        let request = match url.as_str().into_client_request() {
            Ok(req) => req,
            Err(err) => {
                log::debug!("event publisher connect failed: {err}");
                dead.store(true, Ordering::SeqCst);
                return Self {
                    url,
                    dead,
                    sender: None,
                    worker: None,
                };
            }
        };

        let ws = match ws_connect(request) {
            Ok((ws, _resp)) => ws,
            Err(err) => {
                log::debug!("event publisher connect failed: {err}");
                dead.store(true, Ordering::SeqCst);
                return Self {
                    url,
                    dead,
                    sender: None,
                    worker: None,
                };
            }
        };

        // Apply the connect timeout to the live socket where reachable so a
        // wedged peer cannot block the drain thread forever.
        apply_timeout(&ws, connect_timeout);

        let (sender, receiver) = sync_channel::<DrainItem>(QUEUE_MAX);
        let drain_dead = Arc::clone(&dead);
        let ws_cell = Arc::new(Mutex::new(Some(ws)));
        let worker = std::thread::Builder::new()
            .name("hermes-ws-pub".to_string())
            .spawn(move || drain(receiver, ws_cell, drain_dead))
            .ok();

        // If the thread failed to spawn, behave as dead (no worker to drain).
        let (sender, worker) = match worker {
            Some(handle) => (Some(sender), Some(handle)),
            None => {
                dead.store(true, Ordering::SeqCst);
                (None, None)
            }
        };

        Self {
            url,
            dead,
            sender,
            worker,
        }
    }

    /// The dashboard URL this transport targets.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Whether this transport has been marked dead (connect failed, a send
    /// failed, or `close` was called).
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    /// Enqueue `obj` (serialised as a single JSON line) for best-effort
    /// delivery. Returns `true` when enqueued, `false` when the transport is
    /// dead, has no worker, or the queue is full.
    ///
    /// Mirrors `json.dumps(obj, ensure_ascii=False)` — `serde_json` already
    /// emits non-ASCII UTF-8 verbatim, matching `ensure_ascii=False`.
    pub fn write(&self, obj: &Value) -> bool {
        if self.dead.load(Ordering::SeqCst) {
            return false;
        }
        let sender = match self.sender.as_ref() {
            Some(s) => s,
            None => return false,
        };

        let line = match serde_json::to_string(obj) {
            Ok(line) => line,
            Err(_) => return false,
        };

        match sender.try_send(DrainItem::Line(line)) {
            Ok(()) => true,
            // Full or disconnected: drop, like Python's `queue.Full` branch.
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Mark dead, signal the drain worker to stop, join it (best-effort, 3s),
    /// and close the WebSocket. Idempotent.
    pub fn close(&mut self) {
        self.dead.store(true, Ordering::SeqCst);

        if let Some(sender) = self.sender.as_ref() {
            // Best-effort: if the queue is wedged, the daemon thread will be
            // torn down with the process.
            let _ = sender.try_send(DrainItem::Stop);
        }
        // Drop the sender so the receiver also observes a disconnect, which
        // lets a blocked `recv` in the drain loop wake up and exit.
        self.sender = None;

        if let Some(worker) = self.worker.take() {
            join_with_timeout(worker, Duration::from_secs(3));
        }
        // The drain thread owns and closes the WebSocket on exit.
    }
}

impl Drop for WsPublisherTransport {
    fn drop(&mut self) {
        self.close();
    }
}

/// Apply a read/write timeout to the underlying TCP stream when accessible.
fn apply_timeout(ws: &Ws, timeout: Duration) {
    let stream = match ws.get_ref() {
        MaybeTlsStream::Plain(s) => Some(s),
        #[allow(unreachable_patterns)]
        _ => None,
    };
    if let Some(s) = stream {
        let _ = s.set_read_timeout(Some(timeout));
        let _ = s.set_write_timeout(Some(timeout));
    }
}

/// Drain loop: mirrors the Python `_drain` daemon thread.
///
/// Pulls lines off the channel and sends them as WebSocket text frames. On the
/// first send error it marks the transport dead and drops the socket, which
/// short-circuits all subsequent writes.
fn drain(receiver: Receiver<DrainItem>, ws_cell: Arc<Mutex<Option<Ws>>>, dead: Arc<AtomicBool>) {
    loop {
        let item = match receiver.recv() {
            Ok(item) => item,
            // All senders dropped: treat like the `_DRAIN_STOP` sentinel.
            Err(_) => break,
        };

        let line = match item {
            DrainItem::Stop => break,
            DrainItem::Line(line) => line,
        };

        // `self._ws is None` short-circuit.
        {
            let guard = ws_cell.lock().unwrap();
            if guard.is_none() {
                continue;
            }
        }

        let send_result = {
            let mut guard = ws_cell.lock().unwrap();
            match guard.as_mut() {
                Some(ws) => Some(ws.send(WsMessage::Text(line.into()))),
                None => None,
            }
        };

        match send_result {
            Some(Ok(())) => {}
            Some(Err(err)) => {
                log::debug!("event publisher write failed: {err}");
                dead.store(true, Ordering::SeqCst);
                *ws_cell.lock().unwrap() = None;
            }
            None => {}
        }
    }

    // On exit, close the socket if still present (mirrors `close`'s WS teardown
    // now that the worker, rather than the caller, owns the socket).
    if let Some(mut ws) = ws_cell.lock().unwrap().take() {
        let _ = ws.close(None);
        // Drive the close handshake briefly; ignore all errors.
        let _ = ws.flush();
    }
}

/// Join a worker thread, giving up after `timeout`.
///
/// `std::thread::JoinHandle` has no timed join, so we poll `is_finished`.
fn join_with_timeout(handle: JoinHandle<()>, timeout: Duration) {
    let start = std::time::Instant::now();
    let poll = Duration::from_millis(10);
    while !handle.is_finished() {
        if start.elapsed() >= timeout {
            // Detach: let it be torn down with the process.
            return;
        }
        std::thread::sleep(poll);
    }
    let _ = handle.join();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::net::TcpListener;

    /// A transport that fails to connect must be dead and reject writes.
    #[test]
    fn dead_transport_rejects_writes() {
        // Port 1 is privileged/unbindable for clients in practice; an
        // unroutable address yields a connect failure quickly.
        let t = WsPublisherTransport::connect_with_timeout(
            "ws://127.0.0.1:1/",
            Duration::from_millis(200),
        );
        assert!(t.is_dead());
        assert!(!t.write(&json!({"type": "tool"})));
    }

    /// A malformed URL must produce a dead transport, not a panic.
    #[test]
    fn bad_url_is_dead() {
        let t = WsPublisherTransport::connect("not a url at all");
        assert!(t.is_dead());
        assert!(!t.write(&json!({"a": 1})));
        assert_eq!(t.url(), "not a url at all");
    }

    /// `close` is idempotent on a dead transport.
    #[test]
    fn close_idempotent() {
        let mut t = WsPublisherTransport::connect("ws://127.0.0.1:1/");
        t.close();
        t.close();
        assert!(t.is_dead());
    }

    /// Spin up a real WS server, connect, write a frame, and assert it arrives.
    #[test]
    fn round_trip_delivers_json_line() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("ws://{}/", addr);

        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut ws = tungstenite::accept(stream).unwrap();
            match ws.read() {
                Ok(WsMessage::Text(t)) => {
                    let _ = tx.send(t.to_string());
                }
                _ => {
                    let _ = tx.send(String::new());
                }
            }
        });

        let t = WsPublisherTransport::connect(&url);
        assert!(!t.is_dead());
        assert!(t.write(&json!({"type": "status", "msg": "héllo"})));

        let received = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let parsed: Value = serde_json::from_str(&received).unwrap();
        assert_eq!(parsed["type"], "status");
        // ensure_ascii=False parity: non-ASCII preserved verbatim.
        assert_eq!(parsed["msg"], "héllo");
        assert!(received.contains("héllo"));

        drop(t);
        let _ = server.join();
    }
}
