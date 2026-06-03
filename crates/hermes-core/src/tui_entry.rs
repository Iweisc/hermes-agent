//! TUI gateway stdio entrypoint.
//!
//! Faithful native Rust port of `tui_gateway/entry.py`.
//!
//! This module drives the JSON-RPC gateway over stdio: it emits a
//! `gateway.ready` event, then reads newline-delimited JSON-RPC requests from
//! stdin, dispatches each one, and writes any response back to stdout. Three
//! shutdown paths (startup-write fail, parse-error-response write fail,
//! dispatch-response write fail) plus stdin EOF all collapse into a clean exit
//! and a crash-log breadcrumb.
//!
//! ## Differences from the Python original
//!
//! * **`sys.path` shadowing guard** (Python lines 7-12) is a Python-import
//!   concern with no analogue in a compiled Rust binary; it is intentionally
//!   omitted.
//! * **Signal handling.** Python installs handlers for `SIGPIPE` (ignore),
//!   `SIGTERM`/`SIGHUP` (log + grace + hard-exit) and `SIGINT` (ignore). The
//!   equivalents are provided by [`install_signal_handlers`] using `libc`, but
//!   the diagnostic stack-dump that Python emits via `traceback` has no safe
//!   Rust equivalent inside an async-signal context, so the handler records a
//!   one-line breadcrumb instead. See [`shutdown_grace_seconds`].
//! * **Dependency injection.** `tui_gateway.server`'s `dispatch`, `write_json`
//!   and `resolve_skin` are not ported yet. They are taken as callbacks on
//!   [`Deps`] so this module can be wired up once the server lands. A minimal
//!   [`default_resolve_skin`] (returns `{}`) and a stdout-backed
//!   [`stdio_write_json`] are provided so the entrypoint is usable standalone.
//! * **MCP tool discovery** (Python lines 188-201) is a Python-import side
//!   effect (`tools.mcp_tool.discover_mcp_tools`); it is represented by the
//!   optional [`Deps::discover_mcp_tools`] hook and the config probe in
//!   [`has_mcp_servers`].

use std::io::{BufRead, Write};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::tui_transport::{StdioTransport, Transport, WriteResult};

/// Crash-log path: `<hermes_home>/logs/tui_gateway_crash.log`.
///
/// Mirrors `tui_gateway.server._CRASH_LOG`, which is
/// `os.path.join(_hermes_home, "logs", "tui_gateway_crash.log")`.
pub fn crash_log_path() -> std::path::PathBuf {
    crate::mod_hermes_constants::get_hermes_home()
        .join("logs")
        .join("tui_gateway_crash.log")
}

/// Default shutdown grace, in seconds.
///
/// Mirrors Python's `_DEFAULT_SHUTDOWN_GRACE_S = 1.0`.
pub const DEFAULT_SHUTDOWN_GRACE_S: f64 = 1.0;

/// Resolve the orderly-shutdown grace window from
/// `HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S`, falling back to
/// [`DEFAULT_SHUTDOWN_GRACE_S`].
///
/// Faithful port of `_shutdown_grace_seconds`: blank → default; unparseable →
/// default; non-positive → default.
pub fn shutdown_grace_seconds() -> f64 {
    let raw = std::env::var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S")
        .unwrap_or_default()
        .trim()
        .to_string();
    if raw.is_empty() {
        return DEFAULT_SHUTDOWN_GRACE_S;
    }
    match raw.parse::<f64>() {
        Ok(value) if value > 0.0 => value,
        _ => DEFAULT_SHUTDOWN_GRACE_S,
    }
}

/// Append a crash-log entry recording why the gateway subprocess is shutting
/// down, and echo a one-line summary to stderr.
///
/// Faithful port of `_log_exit`: best-effort file append (errors swallowed),
/// always followed by `[gateway-exit] <reason>` on stderr.
pub fn log_exit(reason: &str) {
    let path = crash_log_path();
    // Best-effort: makedirs + append; swallow any error (Python `except: pass`).
    let _ = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let ts = now_timestamp();
        writeln!(f, "\n=== gateway exit · {ts} · reason={reason} ===")?;
        Ok(())
    })();
    eprintln!("[gateway-exit] {reason}");
    let _ = std::io::stderr().flush();
}

/// Append a crash-log entry recording WHICH termination signal hit the
/// process, then echo a one-line summary to stderr.
///
/// Port of the file-logging portion of `_log_signal`. The Python original also
/// dumps every live thread's stack via `traceback`; that has no
/// async-signal-safe Rust analogue, so this records the signal name only. The
/// grace-window + hard-exit semantics live in [`install_signal_handlers`].
pub fn log_signal(signum: i32) {
    let name = signal_name(signum);
    let path = crash_log_path();
    let _ = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let ts = now_timestamp();
        writeln!(f, "\n=== {name} received · {ts} ===")?;
        Ok(())
    })();
    eprintln!("[gateway-signal] {name}");
    let _ = std::io::stderr().flush();
}

/// Map a signal number to its name, matching Python's lookup table.
fn signal_name(signum: i32) -> String {
    match signum {
        libc::SIGPIPE => "SIGPIPE".to_string(),
        libc::SIGTERM => "SIGTERM".to_string(),
        libc::SIGHUP => "SIGHUP".to_string(),
        other => format!("signal {other}"),
    }
}

/// Local timestamp in `%Y-%m-%d %H:%M:%S`, matching Python's
/// `time.strftime('%Y-%m-%d %H:%M:%S')`.
fn now_timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

// ---------------------------------------------------------------------------
// Signal handlers
// ---------------------------------------------------------------------------

/// Install the gateway's termination-signal handlers.
///
/// Mirrors the Python install block:
///
/// ```text
/// signal.signal(signal.SIGPIPE, signal.SIG_IGN)
/// signal.signal(signal.SIGTERM, _log_signal)
/// signal.signal(signal.SIGHUP,  _log_signal)
/// signal.signal(signal.SIGINT,  signal.SIG_IGN)
/// ```
///
/// * `SIGPIPE` and `SIGINT` are ignored (`SIG_IGN`). Ignoring `SIGPIPE` lets a
///   broken-pipe write surface as an `EPIPE` error the transport handles
///   cleanly, instead of the kernel killing the process silently.
/// * `SIGTERM` and `SIGHUP` route to a handler that records a crash-log
///   breadcrumb, arms a daemon timer for the configured grace window, then
///   exits. The grace window comes from [`shutdown_grace_seconds`].
///
/// # Safety
/// Installs process-global signal dispositions via `libc::signal`. The handler
/// runs in an async-signal context; it spawns a watchdog thread that calls
/// `_exit(0)` after the grace window so a wedged worker can never strand the
/// process, then triggers `std::process::exit(0)`.
pub fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        libc::signal(libc::SIGINT, libc::SIG_IGN);
        libc::signal(libc::SIGTERM, handle_term_signal as libc::sighandler_t);
        libc::signal(libc::SIGHUP, handle_term_signal as libc::sighandler_t);
    }
}

/// C-ABI signal handler for SIGTERM/SIGHUP.
///
/// Async-signal-safety note: file I/O and thread spawning are not strictly
/// async-signal-safe, mirroring the latitude Python's pure-Python handler
/// takes. The grace timer guarantees forward progress: even if the orderly
/// exit wedges, `_exit(0)` fires after the configured window.
extern "C" fn handle_term_signal(signum: libc::c_int) {
    log_signal(signum);

    // Daemon watchdog: hard `_exit(0)` after the grace window so a wedged
    // write/flush can never strand the process. Mirrors Python's
    // `threading.Timer(_shutdown_grace_seconds(), _hard_exit)`.
    let grace = shutdown_grace_seconds();
    std::thread::spawn(move || {
        let millis = (grace * 1000.0) as u64;
        std::thread::sleep(std::time::Duration::from_millis(millis));
        // `os._exit(0)` analogue: skip destructors/atexit, break any deadlock.
        unsafe { libc::_exit(0) };
    });

    // Orderly exit on the main path (analogue of `sys.exit(0)` re-raised so the
    // interpreter unwinds and runs finalisers within the grace window).
    std::process::exit(0);
}

// ---------------------------------------------------------------------------
// Dependency injection
// ---------------------------------------------------------------------------

/// Pluggable dependencies for [`run`].
///
/// The Python entrypoint imports `dispatch`, `write_json` and `resolve_skin`
/// directly from `tui_gateway.server`. Until that module is ported, callers
/// supply them here. `discover_mcp_tools` is the optional MCP discovery hook.
pub struct Deps {
    /// Dispatch one parsed JSON-RPC request to a response. `None` means
    /// "notification — no response" (the Python `dispatch` returns `None`).
    pub dispatch: Box<dyn Fn(&Value) -> Option<Value> + Send + Sync>,
    /// Emit one JSON frame to the peer. Mirrors `server.write_json`: returns
    /// `false` when the peer is gone (clean disconnect → exit), `true` on
    /// success.
    pub write_json: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    /// Resolve the active skin for the `gateway.ready` payload. Mirrors
    /// `server.resolve_skin`.
    pub resolve_skin: Box<dyn Fn() -> Value + Send + Sync>,
    /// Optional MCP tool discovery hook. Invoked at startup only when
    /// [`has_mcp_servers`] (or the supplied predicate) reports MCP work to do.
    /// Best-effort: errors are swallowed.
    pub discover_mcp_tools: Option<Box<dyn Fn() + Send + Sync>>,
}

impl Deps {
    /// Build [`Deps`] wired to a stdout-backed transport, the default
    /// (empty) skin resolver, and the supplied dispatcher. No MCP discovery.
    pub fn with_dispatch<F>(dispatch: F) -> Deps
    where
        F: Fn(&Value) -> Option<Value> + Send + Sync + 'static,
    {
        let transport: Arc<dyn Transport> = Arc::new(StdioTransport::real_stdout());
        let wt = transport.clone();
        Deps {
            dispatch: Box::new(dispatch),
            write_json: Box::new(move |obj| transport_write(&wt, obj)),
            resolve_skin: Box::new(default_resolve_skin),
            discover_mcp_tools: None,
        }
    }
}

/// Emit one JSON frame via a [`Transport`], collapsing the [`WriteResult`] into
/// the Python `write_json` boolean contract.
///
/// `Ok(true)` → `true`; `Ok(false)` (peer gone) → `false`; `Err(_)` (a real
/// I/O / serialization error) is logged and treated as a hard write failure
/// (`false`) so the caller takes the clean-exit path rather than panicking.
pub fn transport_write(transport: &Arc<dyn Transport>, obj: &Value) -> bool {
    match transport.write(obj) {
        Ok(ok) => ok,
        Err(e) => {
            log::error!("tui_entry write_json transport error: {e}");
            false
        }
    }
}

/// Default skin resolver: returns an empty JSON object.
///
/// Matches the Python fallback (`resolve_skin` returns `{}` on any error). The
/// full skin resolution lives in `tui_gateway.server.resolve_skin`, which is
/// not ported here.
pub fn default_resolve_skin() -> Value {
    json!({})
}

/// Write a JSON frame to the process's real stdout as a newline-terminated
/// line, returning `false` if the pipe is gone (broken-pipe / peer gone).
///
/// Convenience for callers that want the simplest possible `write_json` without
/// constructing a [`Transport`].
pub fn stdio_write_json(obj: &Value) -> bool {
    let line = match serde_json::to_string(obj) {
        Ok(s) => s,
        Err(e) => {
            log::error!("tui_entry stdio_write_json serialize error: {e}");
            return false;
        }
    };
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    match handle
        .write_all(line.as_bytes())
        .and_then(|_| handle.write_all(b"\n"))
        .and_then(|_| handle.flush())
    {
        Ok(()) => true,
        Err(e) => {
            if is_peer_gone(&e) {
                false
            } else {
                log::error!("tui_entry stdio_write_json io error: {e}");
                false
            }
        }
    }
}

/// Classify an I/O error as a clean "peer gone" disconnect (broken pipe,
/// connection reset, bad fd, transport shut down) versus a real host error.
fn is_peer_gone(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    if matches!(
        e.kind(),
        ErrorKind::BrokenPipe | ErrorKind::ConnectionReset
    ) {
        return true;
    }
    if let Some(code) = e.raw_os_error() {
        return code == libc::EPIPE
            || code == libc::ECONNRESET
            || code == libc::EBADF
            || code == libc::ESHUTDOWN;
    }
    false
}

// ---------------------------------------------------------------------------
// MCP server probe
// ---------------------------------------------------------------------------

/// Decide whether MCP tool discovery should run at startup.
///
/// Faithful port of the cold-start guard in `main`: read the raw config and
/// return `true` iff `mcp_servers` is a non-empty mapping. The Python code is
/// conservative on *any* error reading the config — it falls back to `true`
/// (run discovery and let it handle its own errors). This mirrors that: a
/// `None`/error config yields `true`.
pub fn has_mcp_servers(raw_config: Option<&Value>) -> bool {
    match raw_config {
        None => true, // conservative fallback (Python `except: _has_mcp_servers = True`)
        Some(cfg) => match cfg.get("mcp_servers") {
            Some(Value::Object(map)) => !map.is_empty(),
            _ => false,
        },
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Outcome of [`run`], capturing which shutdown path the loop took. Useful for
/// tests; the binary entrypoint discards it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Startup `gateway.ready` write failed (broken stdout pipe before first
    /// event).
    StartupWriteFailed,
    /// A parse-error response write failed.
    ParseErrorResponseWriteFailed,
    /// A dispatch response write failed; carries the offending method name (or
    /// `None`, matching `req.get("method")`).
    ResponseWriteFailed(Option<String>),
    /// stdin reached EOF — the TUI closed the command pipe.
    StdinEof,
}

/// Run the gateway stdio loop against an arbitrary reader (stdin in
/// production).
///
/// Faithful port of `main`'s body after sidecar/MCP setup:
///
/// 1. Emit the `gateway.ready` event with the resolved skin. On write failure,
///    log `"startup write failed ..."` and return [`RunOutcome::StartupWriteFailed`].
/// 2. For each non-blank input line: parse JSON. On a parse error, write a
///    JSON-RPC `-32700` "parse error" response; if that write fails, log and
///    return. Otherwise dispatch the request and, if a response is produced,
///    write it; on write failure, log with the method name and return.
/// 3. On EOF, log `"stdin EOF ..."` and return [`RunOutcome::StdinEof`].
///
/// The MCP discovery hook (`deps.discover_mcp_tools`) is invoked first when
/// `run_mcp_discovery` is `true`, mirroring `main`'s `if _has_mcp_servers:`
/// branch; pass [`has_mcp_servers`] to compute that flag.
pub fn run<R: BufRead>(reader: R, deps: &Deps, run_mcp_discovery: bool) -> RunOutcome {
    // MCP tool discovery (Python lines 188-201). Best-effort: swallow errors.
    if run_mcp_discovery {
        if let Some(discover) = &deps.discover_mcp_tools {
            // Mirrors the Python `try: discover_mcp_tools() except Exception: pass`.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| discover()));
        }
    }

    // gateway.ready
    let ready = json!({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "gateway.ready",
            "payload": {"skin": (deps.resolve_skin)()},
        },
    });
    if !(deps.write_json)(&ready) {
        log_exit("startup write failed (broken stdout pipe before first event)");
        return RunOutcome::StartupWriteFailed;
    }

    for raw in reader.lines() {
        let raw = match raw {
            Ok(s) => s,
            // A read error on stdin behaves like EOF for the loop's purposes.
            Err(_) => break,
        };
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                let parse_err = json!({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "parse error"},
                    "id": Value::Null,
                });
                if !(deps.write_json)(&parse_err) {
                    log_exit("parse-error-response write failed (broken stdout pipe)");
                    return RunOutcome::ParseErrorResponseWriteFailed;
                }
                continue;
            }
        };

        // `method = req.get("method") if isinstance(req, dict) else None`
        let method: Option<String> = req
            .as_object()
            .and_then(|m| m.get("method"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(resp) = (deps.dispatch)(&req) {
            if !(deps.write_json)(&resp) {
                // Python: f"response write failed for method={method!r} ..."
                log_exit(&format!(
                    "response write failed for method={} (broken stdout pipe)",
                    py_repr_opt_str(&method)
                ));
                return RunOutcome::ResponseWriteFailed(method);
            }
        }
    }

    log_exit("stdin EOF (TUI closed the command pipe)");
    RunOutcome::StdinEof
}

/// Format an `Option<String>` the way Python's `{!r}` does for `Optional[str]`:
/// `None` → `None`, `Some("x")` → `'x'`.
fn py_repr_opt_str(value: &Option<String>) -> String {
    match value {
        None => "None".to_string(),
        Some(s) => format!("'{s}'"),
    }
}

/// Process entrypoint: install signal handlers and run the stdio loop against
/// real stdin, consuming the resolved dependencies.
///
/// Mirrors `main` end-to-end (minus the Python-only `sys.path` guard). The MCP
/// discovery flag is computed from `raw_config` via [`has_mcp_servers`].
pub fn main(deps: &Deps, raw_config: Option<&Value>) -> RunOutcome {
    install_signal_handlers();
    let run_mcp = has_mcp_servers(raw_config);
    let stdin = std::io::stdin();
    let locked = stdin.lock();
    run(locked, deps, run_mcp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Build a `Deps` whose `write_json` records every frame into `sink` and
    /// can be forced to report "peer gone" after the Nth write.
    fn recording_deps(
        sink: Arc<Mutex<Vec<Value>>>,
        fail_after: Option<usize>,
        dispatch: impl Fn(&Value) -> Option<Value> + Send + Sync + 'static,
    ) -> Deps {
        let count = Arc::new(AtomicUsize::new(0));
        let s = sink.clone();
        Deps {
            dispatch: Box::new(dispatch),
            write_json: Box::new(move |obj| {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if let Some(limit) = fail_after {
                    if n >= limit {
                        return false;
                    }
                }
                s.lock().unwrap().push(obj.clone());
                true
            }),
            resolve_skin: Box::new(|| json!({"name": "default"})),
            discover_mcp_tools: None,
        }
    }

    #[test]
    fn shutdown_grace_default_when_unset() {
        unsafe { std::env::remove_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S") };
        assert_eq!(shutdown_grace_seconds(), DEFAULT_SHUTDOWN_GRACE_S);
    }

    #[test]
    fn shutdown_grace_parses_positive() {
        unsafe { std::env::set_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", " 2.5 ") };
        assert_eq!(shutdown_grace_seconds(), 2.5);
        unsafe { std::env::remove_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S") };
    }

    #[test]
    fn shutdown_grace_rejects_nonpositive_and_garbage() {
        unsafe { std::env::set_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "0") };
        assert_eq!(shutdown_grace_seconds(), DEFAULT_SHUTDOWN_GRACE_S);
        unsafe { std::env::set_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "-3") };
        assert_eq!(shutdown_grace_seconds(), DEFAULT_SHUTDOWN_GRACE_S);
        unsafe { std::env::set_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "abc") };
        assert_eq!(shutdown_grace_seconds(), DEFAULT_SHUTDOWN_GRACE_S);
        unsafe { std::env::remove_var("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S") };
    }

    #[test]
    fn has_mcp_servers_logic() {
        assert!(has_mcp_servers(None), "missing config is conservative true");
        assert!(!has_mcp_servers(Some(&json!({}))));
        assert!(!has_mcp_servers(Some(&json!({"mcp_servers": {}}))));
        assert!(!has_mcp_servers(Some(&json!({"mcp_servers": []}))));
        assert!(!has_mcp_servers(Some(&json!({"mcp_servers": null}))));
        assert!(has_mcp_servers(Some(&json!({"mcp_servers": {"a": 1}}))));
    }

    #[test]
    fn py_repr_matches_python() {
        assert_eq!(py_repr_opt_str(&None), "None");
        assert_eq!(py_repr_opt_str(&Some("chat.send".to_string())), "'chat.send'");
    }

    #[test]
    fn emits_ready_then_dispatches_and_stops_at_eof() {
        let sink = Arc::new(Mutex::new(Vec::new()));
        let deps = recording_deps(sink.clone(), None, |req| {
            // echo the id back as a result
            let id = req.get("id").cloned().unwrap_or(Value::Null);
            Some(json!({"jsonrpc": "2.0", "result": "ok", "id": id}))
        });
        let input = "{\"jsonrpc\":\"2.0\",\"method\":\"ping\",\"id\":1}\n\n{\"jsonrpc\":\"2.0\",\"method\":\"ping\",\"id\":2}\n";
        let outcome = run(Cursor::new(input), &deps, false);
        assert_eq!(outcome, RunOutcome::StdinEof);
        let frames = sink.lock().unwrap();
        // ready + 2 responses
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0]["params"]["type"], json!("gateway.ready"));
        assert_eq!(frames[0]["params"]["payload"]["skin"], json!({"name": "default"}));
        assert_eq!(frames[1]["id"], json!(1));
        assert_eq!(frames[2]["id"], json!(2));
    }

    #[test]
    fn parse_error_emits_minus_32700() {
        let sink = Arc::new(Mutex::new(Vec::new()));
        let deps = recording_deps(sink.clone(), None, |_| Some(json!({"ok": true})));
        let outcome = run(Cursor::new("not json\n"), &deps, false);
        assert_eq!(outcome, RunOutcome::StdinEof);
        let frames = sink.lock().unwrap();
        // ready + parse error
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1]["error"]["code"], json!(-32700));
        assert_eq!(frames[1]["error"]["message"], json!("parse error"));
        assert_eq!(frames[1]["id"], Value::Null);
    }

    #[test]
    fn startup_write_failure_short_circuits() {
        let sink = Arc::new(Mutex::new(Vec::new()));
        // fail on the very first write (the ready frame)
        let deps = recording_deps(sink.clone(), Some(0), |_| None);
        let outcome = run(Cursor::new("{\"method\":\"x\"}\n"), &deps, false);
        assert_eq!(outcome, RunOutcome::StartupWriteFailed);
        assert!(sink.lock().unwrap().is_empty());
    }

    #[test]
    fn response_write_failure_reports_method() {
        let sink = Arc::new(Mutex::new(Vec::new()));
        // ready write (#0) succeeds; the response write (#1) fails.
        let deps = recording_deps(sink.clone(), Some(1), |_| Some(json!({"result": 1})));
        let outcome = run(
            Cursor::new("{\"jsonrpc\":\"2.0\",\"method\":\"chat.send\",\"id\":7}\n"),
            &deps,
            false,
        );
        assert_eq!(
            outcome,
            RunOutcome::ResponseWriteFailed(Some("chat.send".to_string()))
        );
    }

    #[test]
    fn notification_dispatch_writes_nothing() {
        let sink = Arc::new(Mutex::new(Vec::new()));
        // dispatch returns None → no response frame
        let deps = recording_deps(sink.clone(), None, |_| None);
        let outcome = run(Cursor::new("{\"method\":\"notify\"}\n"), &deps, false);
        assert_eq!(outcome, RunOutcome::StdinEof);
        // only the ready frame
        assert_eq!(sink.lock().unwrap().len(), 1);
    }

    #[test]
    fn mcp_discovery_hook_runs_when_enabled() {
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        let sink = Arc::new(Mutex::new(Vec::new()));
        let mut deps = recording_deps(sink, None, |_| None);
        deps.discover_mcp_tools = Some(Box::new(move || {
            r.fetch_add(1, Ordering::SeqCst);
        }));
        let _ = run(Cursor::new(""), &deps, true);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mcp_discovery_hook_skipped_when_disabled() {
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        let sink = Arc::new(Mutex::new(Vec::new()));
        let mut deps = recording_deps(sink, None, |_| None);
        deps.discover_mcp_tools = Some(Box::new(move || {
            r.fetch_add(1, Ordering::SeqCst);
        }));
        let _ = run(Cursor::new(""), &deps, false);
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn signal_name_table() {
        assert_eq!(signal_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(signal_name(libc::SIGHUP), "SIGHUP");
        assert_eq!(signal_name(libc::SIGPIPE), "SIGPIPE");
        assert_eq!(signal_name(99999), "signal 99999");
    }
}
