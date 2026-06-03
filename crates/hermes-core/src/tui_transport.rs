//! Transport abstraction for the `tui_gateway` JSON-RPC server.
//!
//! Faithful native Rust port of `tui_gateway/transport.py`.
//!
//! Historically the gateway wrote every JSON frame directly to real stdout.
//! This module decouples the I/O sink from the handler logic so the same
//! dispatcher can be driven over stdio (`tui_gateway.entry`) or WebSocket
//! (`tui_gateway.ws`) without duplicating code.
//!
//! A [`Transport`] is anything that can accept a JSON-serialisable value and
//! forward it to its peer. The active transport for the current request is
//! tracked in a task/thread-local "current transport" slot so handlers route
//! their writes to the right peer.
//!
//! ## Backward compatibility
//!
//! The gateway's `write_json` still works without any transport bound. When
//! nothing is on the current-transport slot and no session-level transport is
//! found, it falls back to the module-level [`StdioTransport`], which wraps the
//! original real-stdout + lock pair. Tests that monkey-patch the stream
//! continue to work because the stdio transport resolves the stream lazily
//! through a callback (`stream_getter`).

use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use serde_json::Value;

// ---------------------------------------------------------------------------
// Peer-gone errno classification
// ---------------------------------------------------------------------------

/// Errno values that mean "the peer is gone" rather than "the host has a real
/// I/O problem". Anything outside this set should be treated as a real error
/// (re-raised / propagated) so it surfaces in the crash log instead of looking
/// like a clean disconnect.
///
/// Mirrors Python's `_PEER_GONE_ERRNOS`:
///   * `EPIPE`      — write to closed pipe (POSIX)
///   * `ECONNRESET` — peer reset the connection
///   * `EBADF`      — fd closed under us
///   * `ESHUTDOWN`  — transport endpoint shut down
///
/// The Python code also folds in the Windows `WSAECONNRESET` / `WSAESHUTDOWN`
/// mappings. On the platforms this crate targets those map onto the POSIX
/// constants above (Rust's `std::io::ErrorKind` already normalises the common
/// cases), so we classify by the POSIX errno set plus `ErrorKind`.
pub fn peer_gone_errnos() -> &'static [i32] {
    // libc constants; computed once.
    &PEER_GONE_ERRNOS
}

static PEER_GONE_ERRNOS: [i32; 4] = [
    libc::EPIPE,
    libc::ECONNRESET,
    libc::EBADF,
    libc::ESHUTDOWN,
];

/// Returns `true` when this errno means "peer gone" (clean disconnect) rather
/// than a real host I/O problem (ENOSPC, EACCES, ...).
pub fn is_peer_gone_errno(errno: i32) -> bool {
    PEER_GONE_ERRNOS.contains(&errno)
}

/// Classify an [`io::Error`] as a peer-gone condition.
///
/// Returns `true` for the POSIX errnos in [`peer_gone_errnos`] as well as the
/// matching [`io::ErrorKind`] variants (`BrokenPipe`, `ConnectionReset`), which
/// is how Rust surfaces `EPIPE` / `ECONNRESET` on most platforms.
pub fn is_peer_gone(err: &io::Error) -> bool {
    match err.kind() {
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset => return true,
        _ => {}
    }
    if let Some(code) = err.raw_os_error() {
        return is_peer_gone_errno(code);
    }
    false
}

// ---------------------------------------------------------------------------
// Flush knob
// ---------------------------------------------------------------------------

/// Reads the `HERMES_TUI_GATEWAY_NO_FLUSH` environment variable and reports
/// whether flushing should be disabled.
///
/// Optional knob: when true, [`StdioTransport`] does not flush after writing.
/// Use this on environments where a half-closed pipe (TUI Node parent quit
/// while the gateway is still emitting events) makes flush block long enough to
/// starve the rest of the worker pool.
///
/// Truthy values (case-insensitive, trimmed): `1`, `true`, `yes`, `on`.
/// Default stays off so the existing flush-after-write behaviour is unchanged.
pub fn disable_flush_from_env() -> bool {
    match std::env::var("HERMES_TUI_GATEWAY_NO_FLUSH") {
        Ok(v) => is_truthy_flag(&v),
        Err(_) => false,
    }
}

fn is_truthy_flag(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Outcome of a transport write/flush attempt.
///
/// This is the Rust analogue of the Python `bool` + exception split:
///
/// * `Ok(true)`  — frame written successfully.
/// * `Ok(false)` — the peer is gone (clean disconnect). This is the
///   dispatcher's "broken stdout pipe" signal; `entry.py` exits cleanly when
///   `write_json` reports `False`.
/// * `Err(_)`    — a real error (programming bug, host I/O problem like
///   `ENOSPC`, non-serialisable payload). The caller must propagate this so the
///   crash log records it instead of treating it as a clean disconnect.
pub type WriteResult = Result<bool, TransportError>;

/// Error type for real (non-peer-gone) transport failures.
#[derive(Debug)]
pub enum TransportError {
    /// Underlying serialization failed (programming error — analogous to a
    /// non-JSON-safe payload raising in `json.dumps`).
    Serialize(serde_json::Error),
    /// A real host I/O error (NOT a peer-gone errno).
    Io(io::Error),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Serialize(e) => write!(f, "transport serialize error: {e}"),
            TransportError::Io(e) => write!(f, "transport io error: {e}"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransportError::Serialize(e) => Some(e),
            TransportError::Io(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for TransportError {
    fn from(e: serde_json::Error) -> Self {
        TransportError::Serialize(e)
    }
}

/// Minimal interface every transport implements.
///
/// Mirrors the Python `Transport` protocol: a `write` that emits one JSON frame
/// and reports whether the peer is gone, plus a `close` to release resources.
pub trait Transport: Send + Sync {
    /// Emit one JSON frame.
    ///
    /// Returns `Ok(false)` when the peer is gone, `Ok(true)` on success, and
    /// `Err(_)` for real errors that must surface (see [`WriteResult`]).
    fn write(&self, obj: &Value) -> WriteResult;

    /// Release any resources owned by this transport.
    fn close(&self) {}
}

// ---------------------------------------------------------------------------
// Current-transport slot (contextvars analogue)
// ---------------------------------------------------------------------------

thread_local! {
    static CURRENT_TRANSPORT: RefCell<Option<Arc<dyn Transport>>> = const { RefCell::new(None) };
}

/// Opaque token returned by [`bind_transport`] and consumed by
/// [`reset_transport`] to restore the previous binding (analogue of the
/// `contextvars` token).
#[must_use = "the token must be passed to reset_transport to restore the previous binding"]
pub struct TransportToken {
    previous: Option<Arc<dyn Transport>>,
}

/// Return the transport bound for the current context (thread), if any.
pub fn current_transport() -> Option<Arc<dyn Transport>> {
    CURRENT_TRANSPORT.with(|slot| slot.borrow().clone())
}

/// Bind `transport` for the current context. Returns a token for
/// [`reset_transport`].
pub fn bind_transport(transport: Option<Arc<dyn Transport>>) -> TransportToken {
    CURRENT_TRANSPORT.with(|slot| {
        let previous = slot.borrow_mut().replace_with_option(transport);
        TransportToken { previous }
    })
}

/// Restore the transport binding captured by [`bind_transport`].
pub fn reset_transport(token: TransportToken) {
    CURRENT_TRANSPORT.with(|slot| {
        *slot.borrow_mut() = token.previous;
    });
}

// Small helper to mimic `Option::replace` for an `Option<T>` held behind a
// mutable borrow, returning the old value.
trait ReplaceWithOption<T> {
    fn replace_with_option(&mut self, value: Option<T>) -> Option<T>;
}

impl<T> ReplaceWithOption<T> for Option<T> {
    fn replace_with_option(&mut self, value: Option<T>) -> Option<T> {
        std::mem::replace(self, value)
    }
}

// ---------------------------------------------------------------------------
// StdioTransport
// ---------------------------------------------------------------------------

/// A writable byte sink resolved lazily through a callback.
///
/// The Python original resolves `sys.stdout` via a callable so runtime
/// monkey-patches of the underlying stream continue to work. Here the sink is
/// any `Write` returned by the getter, locked for the duration of one frame.
pub type StreamGetter = Arc<dyn Fn() -> Arc<Mutex<dyn Write + Send>> + Send + Sync>;

/// Writes JSON frames to a stream (usually real stdout).
///
/// The stream is resolved via a callable so runtime swaps of the underlying
/// stream continue to work — this preserves the behaviour the existing test
/// suite relies on.
pub struct StdioTransport {
    stream_getter: StreamGetter,
    disable_flush: bool,
}

impl StdioTransport {
    /// Construct a transport that resolves its sink lazily through
    /// `stream_getter`. The `HERMES_TUI_GATEWAY_NO_FLUSH` env knob is read once
    /// at construction time (matching the Python module-load semantics).
    pub fn new(stream_getter: StreamGetter) -> Self {
        StdioTransport {
            stream_getter,
            disable_flush: disable_flush_from_env(),
        }
    }

    /// Like [`StdioTransport::new`] but with an explicit `disable_flush`
    /// override, useful for tests.
    pub fn with_flush_setting(stream_getter: StreamGetter, disable_flush: bool) -> Self {
        StdioTransport {
            stream_getter,
            disable_flush,
        }
    }

    /// Construct an `StdioTransport` writing to the process's real stdout.
    pub fn real_stdout() -> Self {
        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(io::stdout()));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        StdioTransport::new(getter)
    }
}

impl Transport for StdioTransport {
    fn write(&self, obj: &Value) -> WriteResult {
        // Serialization is OUTSIDE the lock so a large payload can't block
        // other threads emitting their own frames. A non-serialisable payload
        // is a programming error: surface it (Err) instead of taking the
        // peer-gone (false) path. `ensure_ascii=False` in Python => serde_json
        // already emits UTF-8 / non-escaped unicode by default.
        let mut line = serde_json::to_string(obj)?;
        line.push('\n');
        let bytes = line.as_bytes();

        // Hold the lock for the write (and flush) of this one frame.
        let stream = (self.stream_getter)();
        let mut guard = stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        match guard.write_all(bytes) {
            Ok(()) => {}
            Err(e) => {
                if is_peer_gone(&e) {
                    log::debug!("StdioTransport write peer gone: {e}");
                    return Ok(false);
                }
                // Real host problem (ENOSPC, EACCES, ...) — propagate.
                return Err(TransportError::Io(e));
            }
        }

        // A flush that errors with a peer-gone errno means the dispatcher
        // should exit cleanly. `disable_flush` is the "skip flush entirely"
        // escape hatch for half-closed pipes that hang on flush.
        if !self.disable_flush {
            match guard.flush() {
                Ok(()) => {}
                Err(e) => {
                    if is_peer_gone(&e) {
                        log::debug!("StdioTransport flush peer gone: {e}");
                        return Ok(false);
                    }
                    return Err(TransportError::Io(e));
                }
            }
        }

        Ok(true)
    }

    fn close(&self) {}
}

// ---------------------------------------------------------------------------
// TeeTransport
// ---------------------------------------------------------------------------

/// Mirrors writes to one primary plus N best-effort secondaries.
///
/// The primary's return value (and errors) determine the result — secondaries
/// swallow failures so a wedged sidecar never stalls the main IO path. Used by
/// the PTY child so every dispatcher emit lands on stdio (Ink) AND on a back-WS
/// feeding the dashboard sidebar.
pub struct TeeTransport {
    primary: Arc<dyn Transport>,
    secondaries: Vec<Arc<dyn Transport>>,
}

impl TeeTransport {
    /// Build a tee over a `primary` and zero or more best-effort
    /// `secondaries`.
    pub fn new(primary: Arc<dyn Transport>, secondaries: Vec<Arc<dyn Transport>>) -> Self {
        TeeTransport {
            primary,
            secondaries,
        }
    }
}

impl Transport for TeeTransport {
    fn write(&self, obj: &Value) -> WriteResult {
        // Primary first so a slow sidecar (WS publisher) never delays Ink/stdio.
        let ok = self.primary.write(obj);
        for sec in &self.secondaries {
            // Secondaries swallow everything (peer-gone AND real errors) so a
            // wedged sidecar never stalls the main path.
            let _ = sec.write(obj);
        }
        ok
    }

    fn close(&self) {
        // Close primary first; secondaries best-effort regardless of outcome.
        // (Rust has no exceptions, so "finally" is just unconditional ordering.)
        self.primary.close();
        for sec in &self.secondaries {
            sec.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn truthy_flag_parsing() {
        for v in ["1", "true", "TRUE", "Yes", " on ", "On"] {
            assert!(is_truthy_flag(v), "{v:?} should be truthy");
        }
        for v in ["0", "false", "no", "off", "", "  "] {
            assert!(!is_truthy_flag(v), "{v:?} should be falsey");
        }
    }

    #[test]
    fn peer_gone_classification() {
        assert!(is_peer_gone_errno(libc::EPIPE));
        assert!(is_peer_gone_errno(libc::ECONNRESET));
        assert!(is_peer_gone_errno(libc::EBADF));
        assert!(is_peer_gone_errno(libc::ESHUTDOWN));
        // Real host problems are NOT peer-gone.
        assert!(!is_peer_gone_errno(libc::ENOSPC));
        assert!(!is_peer_gone_errno(libc::EACCES));

        let bp = io::Error::from(io::ErrorKind::BrokenPipe);
        assert!(is_peer_gone(&bp));
        let cr = io::Error::from(io::ErrorKind::ConnectionReset);
        assert!(is_peer_gone(&cr));
        let enospc = io::Error::from_raw_os_error(libc::ENOSPC);
        assert!(!is_peer_gone(&enospc));
    }

    #[test]
    fn stdio_write_appends_newline_and_serializes() {
        let buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(SharedBuf(buf.clone())));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        let t = StdioTransport::with_flush_setting(getter, false);

        let ok = t.write(&json!({"method": "gateway.ready"})).unwrap();
        assert!(ok);
        let written = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert_eq!(written, "{\"method\":\"gateway.ready\"}\n");
    }

    #[test]
    fn stdio_write_non_ascii_unescaped() {
        let buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(SharedBuf(buf.clone())));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        let t = StdioTransport::with_flush_setting(getter, true);

        t.write(&json!({"text": "héllo 世界"})).unwrap();
        let written = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(written.contains("héllo 世界"), "got: {written}");
    }

    // A `Write` sink backed by a shared byte buffer so tests can inspect the
    // bytes written after the transport drops its handle.
    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl Write for SharedBuf {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // A sink that always returns a peer-gone error on write.
    struct PeerGoneSink;
    impl Write for PeerGoneSink {
        fn write(&mut self, _b: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // A sink that errors with a real host problem.
    struct EnospcSink;
    impl Write for EnospcSink {
        fn write(&mut self, _b: &[u8]) -> io::Result<usize> {
            Err(io::Error::from_raw_os_error(libc::ENOSPC))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn stdio_write_peer_gone_returns_false() {
        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(PeerGoneSink));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        let t = StdioTransport::with_flush_setting(getter, false);
        let r = t.write(&json!({"a": 1})).unwrap();
        assert!(!r, "peer-gone write should report false");
    }

    #[test]
    fn stdio_write_real_error_propagates() {
        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(EnospcSink));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        let t = StdioTransport::with_flush_setting(getter, false);
        let err = t.write(&json!({"a": 1})).unwrap_err();
        match err {
            TransportError::Io(e) => assert_eq!(e.raw_os_error(), Some(libc::ENOSPC)),
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    // Counting transport for tee tests.
    struct CountingTransport {
        count: Arc<AtomicUsize>,
        result: WriteResultKind,
        closed: Arc<AtomicUsize>,
    }
    #[derive(Clone, Copy)]
    enum WriteResultKind {
        Ok,
        PeerGone,
    }
    impl Transport for CountingTransport {
        fn write(&self, _obj: &Value) -> WriteResult {
            self.count.fetch_add(1, Ordering::SeqCst);
            match self.result {
                WriteResultKind::Ok => Ok(true),
                WriteResultKind::PeerGone => Ok(false),
            }
        }
        fn close(&self) {
            self.closed.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn tee_returns_primary_result_and_mirrors_secondaries() {
        let pc = Arc::new(AtomicUsize::new(0));
        let pclose = Arc::new(AtomicUsize::new(0));
        let sc = Arc::new(AtomicUsize::new(0));
        let sclose = Arc::new(AtomicUsize::new(0));

        let primary: Arc<dyn Transport> = Arc::new(CountingTransport {
            count: pc.clone(),
            result: WriteResultKind::PeerGone,
            closed: pclose.clone(),
        });
        let secondary: Arc<dyn Transport> = Arc::new(CountingTransport {
            count: sc.clone(),
            result: WriteResultKind::Ok,
            closed: sclose.clone(),
        });
        let tee = TeeTransport::new(primary, vec![secondary]);

        // Primary reports peer-gone => tee reports peer-gone (false).
        let r = tee.write(&json!({})).unwrap();
        assert!(!r);
        assert_eq!(pc.load(Ordering::SeqCst), 1);
        assert_eq!(sc.load(Ordering::SeqCst), 1);

        tee.close();
        assert_eq!(pclose.load(Ordering::SeqCst), 1);
        assert_eq!(sclose.load(Ordering::SeqCst), 1);
    }

    // Secondary that errors hard — tee must swallow it.
    struct ErroringTransport;
    impl Transport for ErroringTransport {
        fn write(&self, _obj: &Value) -> WriteResult {
            Err(TransportError::Io(io::Error::from_raw_os_error(libc::ENOSPC)))
        }
    }

    #[test]
    fn tee_swallows_secondary_errors() {
        let pc = Arc::new(AtomicUsize::new(0));
        let pclose = Arc::new(AtomicUsize::new(0));
        let primary: Arc<dyn Transport> = Arc::new(CountingTransport {
            count: pc.clone(),
            result: WriteResultKind::Ok,
            closed: pclose.clone(),
        });
        let secondary: Arc<dyn Transport> = Arc::new(ErroringTransport);
        let tee = TeeTransport::new(primary, vec![secondary]);

        // Secondary errors hard, but tee returns the primary's Ok(true).
        let r = tee.write(&json!({"x": 1})).unwrap();
        assert!(r);
    }

    #[test]
    fn bind_and_reset_transport_roundtrip() {
        assert!(current_transport().is_none());

        let sink: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(Vec::<u8>::new()));
        let getter: StreamGetter = Arc::new(move || sink.clone());
        let t: Arc<dyn Transport> = Arc::new(StdioTransport::new(getter));

        let token = bind_transport(Some(t.clone()));
        assert!(current_transport().is_some());

        // Nested bind then reset restores.
        let inner_token = bind_transport(None);
        assert!(current_transport().is_none());
        reset_transport(inner_token);
        assert!(current_transport().is_some());

        reset_transport(token);
        assert!(current_transport().is_none());
    }
}
