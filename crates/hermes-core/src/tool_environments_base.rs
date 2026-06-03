//! Base class for all Hermes execution environment backends.
//!
//! Faithful native Rust port of `tools/environments/base.py`.
//!
//! Unified spawn-per-call model: every command spawns a fresh `bash -c`
//! process.  A session snapshot (env vars, functions, aliases) is captured
//! once at init and re-sourced before each command.  CWD persists via in-band
//! stdout markers (remote) or a temp file (local).
//!
//! The Python module mixes pure string/transform logic (command wrapping, CWD
//! marker extraction, stdin heredoc embedding, snapshot bootstrap script
//! generation, JSON store helpers, activity throttling) with a concrete
//! process-polling loop.  This port reproduces the pure logic faithfully and
//! exposes a [`BaseEnvironment`] trait plus a default [`SnapshotState`]
//! command-wrapping helper that any backend can reuse.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::Value;

/// Opt-in debug tracing for the interrupt/activity/poll machinery. Set
/// `HERMES_DEBUG_INTERRUPT=1` to enable.  Mirrors the Python module-level
/// `_DEBUG_INTERRUPT`.
pub fn debug_interrupt_enabled() -> bool {
    std::env::var_os("HERMES_DEBUG_INTERRUPT")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Activity callback (thread-local in Python; here a process-global slot keyed
// per-thread is overkill — we expose set/get + the throttled touch helper).
// ---------------------------------------------------------------------------

type ActivityCb = Box<dyn Fn(&str) + Send + 'static>;

thread_local! {
    static ACTIVITY_CALLBACK: std::cell::RefCell<Option<ActivityCb>> =
        const { std::cell::RefCell::new(None) };
}

/// Register a callback that the wait loop fires periodically. Mirrors
/// `set_activity_callback`.  Passing `None` clears it.
pub fn set_activity_callback(cb: Option<ActivityCb>) {
    ACTIVITY_CALLBACK.with(|slot| {
        *slot.borrow_mut() = cb;
    });
}

/// Returns whether an activity callback is currently registered on this thread.
pub fn has_activity_callback() -> bool {
    ACTIVITY_CALLBACK.with(|slot| slot.borrow().is_some())
}

fn fire_activity_callback(msg: &str) {
    ACTIVITY_CALLBACK.with(|slot| {
        if let Some(cb) = slot.borrow().as_ref() {
            cb(msg);
        }
    });
}

/// Mutable state for [`touch_activity_if_due`].  Mirrors the Python `state`
/// dict with `last_touch`, `start` (monotonic [`Instant`]s) and an optional
/// `interval` override (default 10.0 s).
pub struct ActivityState {
    pub last_touch: Instant,
    pub start: Instant,
    pub interval: f64,
}

impl ActivityState {
    /// Create a state with both timestamps set to now and the default 10 s
    /// cadence.
    pub fn new() -> Self {
        let now = Instant::now();
        ActivityState {
            last_touch: now,
            start: now,
            interval: 10.0,
        }
    }
}

impl Default for ActivityState {
    fn default() -> Self {
        Self::new()
    }
}

/// Fire the activity callback at most once every `state.interval` seconds.
///
/// Faithful port of `touch_activity_if_due`. Swallows all callback errors
/// (the Rust callback can't "raise" but the throttle semantics match).
pub fn touch_activity_if_due(state: &mut ActivityState, label: &str) {
    let now = Instant::now();
    let interval = state.interval;
    if now.duration_since(state.last_touch).as_secs_f64() < interval {
        return;
    }
    state.last_touch = now;
    let elapsed = now.duration_since(state.start).as_secs() as i64;
    fire_activity_callback(&format!("{label} ({elapsed}s elapsed)"));
}

// ---------------------------------------------------------------------------
// Sandbox dir
// ---------------------------------------------------------------------------

/// Return the host-side root for all sandbox storage.
///
/// Configurable via `TERMINAL_SANDBOX_DIR`. Defaults to
/// `{hermes_home}/sandboxes/`.  Faithful port of `get_sandbox_dir`; the
/// hermes-home base is taken as a parameter so this module does not hard-depend
/// on the constants crate (callers usually pass
/// `crate::mod_hermes_constants::get_hermes_home()`).
pub fn get_sandbox_dir(hermes_home: &Path) -> std::io::Result<PathBuf> {
    let p = match std::env::var_os("TERMINAL_SANDBOX_DIR") {
        Some(custom) if !custom.is_empty() => PathBuf::from(custom),
        _ => hermes_home.join("sandboxes"),
    };
    fs::create_dir_all(&p)?;
    Ok(p)
}

// ---------------------------------------------------------------------------
// JSON store helpers
// ---------------------------------------------------------------------------

/// Load a JSON file as an object, returning `{}` on any error. Faithful port of
/// `_load_json_store`.
pub fn load_json_store(path: &Path) -> serde_json::Map<String, Value> {
    if path.exists() {
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) {
                return map;
            }
        }
    }
    serde_json::Map::new()
}

/// Write `data` as pretty-printed (2-space indent) JSON to `path`. Faithful
/// port of `_save_json_store`.
pub fn save_json_store(
    path: &Path,
    data: &serde_json::Map<String, Value>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(&Value::Object(data.clone()))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(path, text)
}

/// Return `(mtime, size)` for cache comparison, or `None` if unreadable.
/// Faithful port of `_file_mtime_key`. The mtime is seconds-since-epoch as
/// `f64` to mirror Python's `st_mtime`.
pub fn file_mtime_key(host_path: &str) -> Option<(f64, u64)> {
    let meta = fs::metadata(host_path).ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Some((mtime, size))
}

// ---------------------------------------------------------------------------
// CWD marker
// ---------------------------------------------------------------------------

/// Build the in-band CWD marker for a session. Faithful port of `_cwd_marker`.
pub fn cwd_marker(session_id: &str) -> String {
    format!("__HERMES_CWD_{session_id}__")
}

// ---------------------------------------------------------------------------
// shlex.quote equivalent
// ---------------------------------------------------------------------------

/// Shell-quote a string the way Python's `shlex.quote` does: if it's safe
/// (matches `[A-Za-z0-9_@%+=:,./-]+` and non-empty), return as-is; otherwise
/// wrap in single quotes, escaping embedded single quotes as `'"'"'`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    });
    if safe {
        return s.to_string();
    }
    // Wrap in single quotes; replace each ' with '"'"'
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

// ---------------------------------------------------------------------------
// Result of a command execution.
// ---------------------------------------------------------------------------

/// Mirrors the Python `{"output": str, "returncode": int}` dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    pub output: String,
    pub returncode: i32,
}

impl ExecResult {
    pub fn new(output: impl Into<String>, returncode: i32) -> Self {
        ExecResult {
            output: output.into(),
            returncode,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot / command-wrapping state.
// ---------------------------------------------------------------------------

/// Holds the per-session identifiers and paths used to wrap commands. Mirrors
/// the instance attributes (`_session_id`, `_snapshot_path`, `_cwd_file`,
/// `_cwd_marker`, `_snapshot_ready`) that `BaseEnvironment.__init__` computes.
#[derive(Debug, Clone)]
pub struct SnapshotState {
    pub session_id: String,
    pub snapshot_path: String,
    pub cwd_file: String,
    pub cwd_marker: String,
    pub snapshot_ready: bool,
}

impl SnapshotState {
    /// Construct the session state from a temp dir, replicating
    /// `BaseEnvironment.__init__`'s path computation. `session_id` is the
    /// 12-hex-char uuid token; pass [`new_session_id`] to generate one.
    pub fn new(temp_dir: &str, session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        // temp_dir = self.get_temp_dir().rstrip("/") or "/"
        let trimmed = temp_dir.trim_end_matches('/');
        let temp_dir = if trimmed.is_empty() { "/" } else { trimmed };
        // f"{temp_dir}/..." — note when temp_dir == "/" this yields "//hermes-..."
        // exactly as the Python f-string does.
        SnapshotState {
            snapshot_path: format!("{temp_dir}/hermes-snap-{session_id}.sh"),
            cwd_file: format!("{temp_dir}/hermes-cwd-{session_id}.txt"),
            cwd_marker: cwd_marker(&session_id),
            session_id,
            snapshot_ready: false,
        }
    }

    /// Build the init-session bootstrap script. Faithful port of the
    /// `bootstrap` string assembled in `init_session`. `cwd` is the configured
    /// working directory (quoted via [`shlex_quote`]).
    pub fn bootstrap_script(&self, cwd: &str) -> String {
        let quoted_cwd = shlex_quote(cwd);
        let snap = &self.snapshot_path;
        let cwd_file = &self.cwd_file;
        let marker = &self.cwd_marker;
        format!(
            "export -p > {snap}\n\
             declare -f | grep -vE '^_[^_]' >> {snap}\n\
             alias -p >> {snap}\n\
             echo 'shopt -s expand_aliases' >> {snap}\n\
             echo 'set +e' >> {snap}\n\
             echo 'set +u' >> {snap}\n\
             builtin cd {quoted_cwd} 2>/dev/null || true\n\
             pwd -P > {cwd_file} 2>/dev/null || true\n\
             printf '\\n{marker}%s{marker}\\n' \"$(pwd -P)\"\n"
        )
    }

    /// Quote a `cd` target while preserving `~` expansion. Faithful port of
    /// `_quote_cwd_for_cd`.
    pub fn quote_cwd_for_cd(cwd: &str) -> String {
        if cwd == "~" {
            return cwd.to_string();
        }
        if cwd == "~/" {
            return "$HOME".to_string();
        }
        if let Some(rest) = cwd.strip_prefix("~/") {
            return format!("$HOME/{}", shlex_quote(rest));
        }
        shlex_quote(cwd)
    }

    /// Build the full bash script that sources the snapshot, cd's, runs the
    /// command, re-dumps env vars, and emits CWD markers. Faithful port of
    /// `_wrap_command`.
    pub fn wrap_command(&self, command: &str, cwd: &str) -> String {
        let escaped = command.replace('\'', "'\\''");
        let mut parts: Vec<String> = Vec::new();

        if self.snapshot_ready {
            parts.push(format!(
                "source {} >/dev/null 2>&1 || true",
                self.snapshot_path
            ));
        }

        let quoted_cwd = Self::quote_cwd_for_cd(cwd);
        parts.push(format!("builtin cd -- {quoted_cwd} || exit 126"));

        parts.push(format!("eval '{escaped}'"));
        parts.push("__hermes_ec=$?".to_string());

        if self.snapshot_ready {
            parts.push(format!(
                "export -p > {} 2>/dev/null || true",
                self.snapshot_path
            ));
        }

        parts.push(format!("pwd -P > {} 2>/dev/null || true", self.cwd_file));
        parts.push(format!(
            "printf '\\n{marker}%s{marker}\\n' \"$(pwd -P)\"",
            marker = self.cwd_marker
        ));
        parts.push("exit $__hermes_ec".to_string());

        parts.join("\n")
    }

    /// Parse the `__HERMES_CWD_{session}__` marker from output, updating `cwd`
    /// (returned) and stripping the marker (plus the injected leading newline)
    /// from `output`. Faithful port of `_extract_cwd_from_output`.
    ///
    /// Returns the (possibly updated) cwd and the cleaned output.
    pub fn extract_cwd_from_output(&self, output: &str, cwd: &str) -> (String, String) {
        let marker = &self.cwd_marker;
        let mlen = marker.len();

        let last = match output.rfind(marker.as_str()) {
            Some(i) => i,
            None => return (cwd.to_string(), output.to_string()),
        };

        // search_start = max(0, last - 4096)
        let search_start = last.saturating_sub(4096);
        // first = output.rfind(marker, search_start, last)
        let first = match output[search_start..last].rfind(marker.as_str()) {
            Some(rel) => search_start + rel,
            None => return (cwd.to_string(), output.to_string()),
        };
        if first == last {
            return (cwd.to_string(), output.to_string());
        }

        let mut new_cwd = cwd.to_string();
        let cwd_path = output[first + mlen..last].trim();
        if !cwd_path.is_empty() {
            new_cwd = cwd_path.to_string();
        }

        // line_start = output.rfind("\n", 0, first); if -1 -> first
        let line_start = output[..first].rfind('\n').unwrap_or(first);
        // line_end = output.find("\n", last + len(marker))
        let after = last + mlen;
        let line_end = match output[after..].find('\n') {
            Some(rel) => after + rel + 1, // line_end + 1
            None => output.len(),
        };

        let cleaned = format!("{}{}", &output[..line_start], &output[line_end..]);
        (new_cwd, cleaned)
    }
}

/// Generate a 12-hex-char session id, mirroring `uuid.uuid4().hex[:12]`.
pub fn new_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Build 16 pseudo-random-ish bytes; we only need 12 hex chars (6 bytes).
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mut x = nanos ^ (pid << 64) ^ (nanos << 17);
    // xorshift-ish mixing
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    let hex = format!("{x:032x}");
    hex[..12].to_string()
}

/// Append `stdin_data` as a shell heredoc to `command`. Faithful port of
/// `_embed_stdin_heredoc`. The delimiter token uses a fresh session id.
pub fn embed_stdin_heredoc(command: &str, stdin_data: &str) -> String {
    let delimiter = format!("HERMES_STDIN_{}", new_session_id());
    format!("{command} << '{delimiter}'\n{stdin_data}\n{delimiter}")
}

// ---------------------------------------------------------------------------
// BaseEnvironment trait
// ---------------------------------------------------------------------------

/// How stdin is delivered to the spawned bash process. Mirrors `_stdin_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdinMode {
    /// Pipe stdin to the process (default).
    Pipe,
    /// Embed stdin as a heredoc appended to the command (Modal, Daytona).
    Heredoc,
}

impl Default for StdinMode {
    fn default() -> Self {
        StdinMode::Pipe
    }
}

/// Common interface for all Hermes backends. Mirrors the abstract surface of
/// the Python `BaseEnvironment`. Concrete backends implement [`run_bash`] and
/// [`cleanup`]; the trait supplies the shared `execute` flow built on top of
/// the pure [`SnapshotState`] helpers.
///
/// [`run_bash`]: BaseEnvironment::run_bash
/// [`cleanup`]: BaseEnvironment::cleanup
pub trait BaseEnvironment {
    /// The spawned process handle type.
    type Handle;

    /// Configured snapshot/session state (mutable so `execute` and
    /// `init_session` can flip `snapshot_ready` and stash cwd).
    fn snapshot(&self) -> &SnapshotState;
    fn snapshot_mut(&mut self) -> &mut SnapshotState;

    /// Current working directory (`self.cwd`).
    fn cwd(&self) -> &str;
    fn set_cwd(&mut self, cwd: String);

    /// Default timeout (`self.timeout`).
    fn timeout(&self) -> i32;

    /// Stdin embedding mode (`_stdin_mode`). Defaults to [`StdinMode::Pipe`].
    fn stdin_mode(&self) -> StdinMode {
        StdinMode::Pipe
    }

    /// Snapshot-creation timeout (`_snapshot_timeout`). Defaults to 30.
    fn snapshot_timeout(&self) -> i32 {
        30
    }

    /// Backend temp directory (`get_temp_dir`). Defaults to `/tmp`.
    fn temp_dir(&self) -> String {
        "/tmp".to_string()
    }

    /// Spawn a bash process to run `cmd_string`. Must be implemented by every
    /// backend. Mirrors `_run_bash`.
    fn run_bash(
        &mut self,
        cmd_string: &str,
        login: bool,
        timeout: i32,
        stdin_data: Option<&str>,
    ) -> std::io::Result<Self::Handle>;

    /// Release backend resources. Mirrors `cleanup`.
    fn cleanup(&mut self);

    /// Hook called before each command execution. Mirrors `_before_execute`.
    /// Default is a no-op.
    fn before_execute(&mut self) {}

    /// Transform sudo commands if a SUDO_PASSWORD is available. Mirrors
    /// `_prepare_command`, which delegates to
    /// `tools.terminal_tool._transform_sudo_command`. Default passes through
    /// unchanged (no sudo password); backends/callers should override to wire
    /// `crate::...::transform_sudo_command`.
    fn prepare_command(&self, command: &str) -> (String, Option<String>) {
        (command.to_string(), None)
    }

    /// Rewrite the `A && B &` subshell-wait trap. Mirrors the call to
    /// `tools.terminal_tool._rewrite_compound_background`. Default passes
    /// through unchanged; callers should override to wire
    /// `crate::tool_terminal_tool::rewrite_compound_background` (in the
    /// `hermes` crate).
    fn rewrite_compound_background(&self, command: &str) -> String {
        command.to_string()
    }

    /// Alias for [`cleanup`]. Mirrors `stop`.
    ///
    /// [`cleanup`]: BaseEnvironment::cleanup
    fn stop(&mut self) {
        self.cleanup();
    }
}

/// Plan describing exactly how `execute()` will dispatch a command, with the
/// pure transforms already applied. Backends can compute this, then hand the
/// pieces to their own process-polling implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutePlan {
    /// The fully wrapped bash script (post sudo/compound rewrite + wrap).
    pub wrapped: String,
    /// Effective timeout used for the run.
    pub effective_timeout: i32,
    /// Effective cwd used for the run.
    pub effective_cwd: String,
    /// Stdin to feed to the process, if any (None when embedded as heredoc).
    pub effective_stdin: Option<String>,
    /// Whether to spawn a login shell (true when snapshot is not ready).
    pub login: bool,
}

/// Build the [`ExecutePlan`] for a command, replicating the pure portion of
/// `BaseEnvironment.execute` up to (but not including) the `_run_bash` call.
///
/// This factors out the transform pipeline so it can be unit-tested and reused
/// without spawning processes. `before_execute` and `prepare_command` /
/// `rewrite_compound_background` are taken from the trait object.
pub fn build_execute_plan<E: BaseEnvironment + ?Sized>(
    env: &E,
    command: &str,
    cwd: &str,
    timeout: Option<i32>,
    stdin_data: Option<&str>,
) -> ExecutePlan {
    let (exec_command, sudo_stdin) = env.prepare_command(command);
    let exec_command = env.rewrite_compound_background(&exec_command);

    // effective_timeout = timeout or self.timeout  (Python truthiness: 0 -> falls back)
    let effective_timeout = match timeout {
        Some(t) if t != 0 => t,
        _ => env.timeout(),
    };
    // effective_cwd = cwd or self.cwd
    let effective_cwd = if cwd.is_empty() {
        env.cwd().to_string()
    } else {
        cwd.to_string()
    };

    // Merge sudo stdin with caller stdin.
    let mut effective_stdin: Option<String> = match (&sudo_stdin, stdin_data) {
        (Some(s), Some(d)) => Some(format!("{s}{d}")),
        (Some(s), None) => Some(s.clone()),
        (None, d) => d.map(|s| s.to_string()),
    };

    let mut exec_command = exec_command;
    // Embed stdin as heredoc for backends that need it. Python truthiness:
    // empty string stdin is falsy, so an empty stdin is NOT embedded.
    let stdin_truthy = effective_stdin.as_deref().map(|s| !s.is_empty()).unwrap_or(false);
    if stdin_truthy && env.stdin_mode() == StdinMode::Heredoc {
        let stdin = effective_stdin.take().unwrap();
        exec_command = embed_stdin_heredoc(&exec_command, &stdin);
        effective_stdin = None;
    }

    let wrapped = env.snapshot().wrap_command(&exec_command, &effective_cwd);
    let login = !env.snapshot().snapshot_ready;

    ExecutePlan {
        wrapped,
        effective_timeout,
        effective_cwd,
        effective_stdin,
        login,
    }
}

// ---------------------------------------------------------------------------
// Wait-loop outcome formatting (pure helpers extracted from _wait_for_process)
// ---------------------------------------------------------------------------

/// Build the interrupted-command result. Mirrors the interrupt branch:
/// `{"output": "".join(chunks) + "\n[Command interrupted]", "returncode": 130}`.
pub fn interrupted_result(output_chunks: &str) -> ExecResult {
    ExecResult::new(format!("{output_chunks}\n[Command interrupted]"), 130)
}

/// Build the timed-out command result. Mirrors the timeout branch: if partial
/// output exists, append `\n[Command timed out after {timeout}s]`; otherwise
/// emit the message with the leading newline stripped. Returncode 124.
pub fn timeout_result(partial: &str, timeout: i32) -> ExecResult {
    let timeout_msg = format!("\n[Command timed out after {timeout}s]");
    let output = if !partial.is_empty() {
        format!("{partial}{timeout_msg}")
    } else {
        timeout_msg.trim_start().to_string()
    };
    ExecResult::new(output, 124)
}

// ---------------------------------------------------------------------------
// A simple JSON-store with last-mtime-cache key tracking, useful to backends.
// (Process-global guard so concurrent flushes don't interleave.)
// ---------------------------------------------------------------------------

static STORE_LOCK: Mutex<()> = Mutex::new(());

/// Atomically read-modify-write a JSON object store under a process lock.
/// Helper for backends that persist small key/value maps (mirrors the
/// load/modify/save pattern around `_load_json_store`/`_save_json_store`).
pub fn with_json_store<F, R>(path: &Path, f: F) -> std::io::Result<R>
where
    F: FnOnce(&mut serde_json::Map<String, Value>) -> R,
{
    let _g = STORE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let mut map = load_json_store(path);
    let r = f(&mut map);
    save_json_store(path, &map)?;
    Ok(r)
}

/// Convenience: convert a [`BTreeMap`] of strings into a JSON object map,
/// useful when building env snapshots for stores.
pub fn string_map_to_json(m: &BTreeMap<String, String>) -> serde_json::Map<String, Value> {
    m.iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shlex_quote_basic() {
        assert_eq!(shlex_quote("foo"), "foo");
        assert_eq!(shlex_quote("/a/b-c.d"), "/a/b-c.d");
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn quote_cwd_for_cd_cases() {
        assert_eq!(SnapshotState::quote_cwd_for_cd("~"), "~");
        assert_eq!(SnapshotState::quote_cwd_for_cd("~/"), "$HOME");
        assert_eq!(SnapshotState::quote_cwd_for_cd("~/a b"), "$HOME/'a b'");
        assert_eq!(SnapshotState::quote_cwd_for_cd("~/proj"), "$HOME/proj");
        assert_eq!(SnapshotState::quote_cwd_for_cd("/abs/path"), "/abs/path");
        assert_eq!(SnapshotState::quote_cwd_for_cd("with space"), "'with space'");
    }

    #[test]
    fn snapshot_paths() {
        let s = SnapshotState::new("/tmp", "abc123def456");
        assert_eq!(s.snapshot_path, "/tmp/hermes-snap-abc123def456.sh");
        assert_eq!(s.cwd_file, "/tmp/hermes-cwd-abc123def456.txt");
        assert_eq!(s.cwd_marker, "__HERMES_CWD_abc123def456__");
        assert!(!s.snapshot_ready);
    }

    #[test]
    fn snapshot_trailing_slash_and_empty() {
        let s = SnapshotState::new("/tmp/", "s");
        assert_eq!(s.snapshot_path, "/tmp/hermes-snap-s.sh");
        // get_temp_dir() rstrip("/") or "/" -> "/" -> "//hermes-..."
        let root = SnapshotState::new("/", "s");
        assert_eq!(root.snapshot_path, "//hermes-snap-s.sh");
        let empty = SnapshotState::new("", "s");
        assert_eq!(empty.snapshot_path, "//hermes-snap-s.sh");
    }

    #[test]
    fn wrap_command_with_snapshot() {
        let mut s = SnapshotState::new("/tmp", "sess");
        s.snapshot_ready = true;
        let w = s.wrap_command("echo hi", "/work");
        assert!(w.contains("source /tmp/hermes-snap-sess.sh >/dev/null 2>&1 || true"));
        assert!(w.contains("builtin cd -- /work || exit 126"));
        assert!(w.contains("eval 'echo hi'"));
        assert!(w.contains("__hermes_ec=$?"));
        assert!(w.contains("export -p > /tmp/hermes-snap-sess.sh 2>/dev/null || true"));
        assert!(w.contains("pwd -P > /tmp/hermes-cwd-sess.txt 2>/dev/null || true"));
        assert!(w.contains("__HERMES_CWD_sess__"));
        assert!(w.trim_end().ends_with("exit $__hermes_ec"));
    }

    #[test]
    fn wrap_command_without_snapshot_escapes_quotes() {
        let s = SnapshotState::new("/tmp", "sess");
        let w = s.wrap_command("echo 'hi'", "~/p");
        // No source line when snapshot not ready.
        assert!(!w.contains("source /tmp/hermes-snap"));
        // No export re-dump line either.
        assert!(!w.contains("export -p > /tmp/hermes-snap"));
        // Single quotes escaped via '\'' .
        assert!(w.contains(r"eval 'echo '\''hi'\'''"));
        assert!(w.contains("builtin cd -- $HOME/p || exit 126"));
    }

    #[test]
    fn extract_cwd_updates_and_strips() {
        let s = SnapshotState::new("/tmp", "sess");
        let m = &s.cwd_marker;
        // Emulate the printf '\n{m}%s{m}\n' output appended after command output.
        let output = format!("hello world\n{m}/new/dir{m}\n");
        let (cwd, cleaned) = s.extract_cwd_from_output(&output, "/old");
        assert_eq!(cwd, "/new/dir");
        assert_eq!(cleaned, "hello world");
    }

    #[test]
    fn extract_cwd_no_marker_noop() {
        let s = SnapshotState::new("/tmp", "sess");
        let (cwd, cleaned) = s.extract_cwd_from_output("plain output", "/old");
        assert_eq!(cwd, "/old");
        assert_eq!(cleaned, "plain output");
    }

    #[test]
    fn extract_cwd_empty_path_keeps_cwd() {
        let s = SnapshotState::new("/tmp", "sess");
        let m = &s.cwd_marker;
        // Empty path between markers -> cwd unchanged but markers still stripped.
        let output = format!("out\n{m}{m}\n");
        let (cwd, cleaned) = s.extract_cwd_from_output(&output, "/old");
        assert_eq!(cwd, "/old");
        assert_eq!(cleaned, "out");
    }

    #[test]
    fn embed_stdin_heredoc_shape() {
        let h = embed_stdin_heredoc("cat", "line1\nline2");
        assert!(h.starts_with("cat << 'HERMES_STDIN_"));
        assert!(h.contains("\nline1\nline2\n"));
        // delimiter appears twice (open + close)
        let count = h.matches("HERMES_STDIN_").count();
        assert_eq!(count, 2);
    }

    #[test]
    fn interrupted_and_timeout_results() {
        assert_eq!(
            interrupted_result("abc"),
            ExecResult::new("abc\n[Command interrupted]", 130)
        );
        assert_eq!(
            timeout_result("part", 30),
            ExecResult::new("part\n[Command timed out after 30s]", 124)
        );
        // No partial -> message lstripped.
        assert_eq!(
            timeout_result("", 5),
            ExecResult::new("[Command timed out after 5s]", 124)
        );
    }

    #[test]
    fn json_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("hermes-jstest-{}", new_session_id()));
        let path = dir.join("store.json");
        let mut m = serde_json::Map::new();
        m.insert("k".to_string(), Value::String("v".to_string()));
        save_json_store(&path, &m).unwrap();
        let loaded = load_json_store(&path);
        assert_eq!(loaded.get("k").and_then(|v| v.as_str()), Some("v"));
        // Missing file -> empty.
        let missing = load_json_store(&dir.join("nope.json"));
        assert!(missing.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn with_json_store_modifies() {
        let dir = std::env::temp_dir().join(format!("hermes-jstest2-{}", new_session_id()));
        let path = dir.join("s.json");
        with_json_store(&path, |m| {
            m.insert("a".into(), Value::from(1));
        })
        .unwrap();
        let loaded = load_json_store(&path);
        assert_eq!(loaded.get("a").and_then(|v| v.as_i64()), Some(1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_id_is_12_hex() {
        let id = new_session_id();
        assert_eq!(id.len(), 12);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sandbox_dir_custom_env() {
        let tmp = std::env::temp_dir().join(format!("hermes-sbx-{}", new_session_id()));
        unsafe {
            std::env::set_var("TERMINAL_SANDBOX_DIR", &tmp);
        }
        let p = get_sandbox_dir(Path::new("/unused")).unwrap();
        assert_eq!(p, tmp);
        assert!(p.is_dir());
        unsafe {
            std::env::remove_var("TERMINAL_SANDBOX_DIR");
        }
        // Default path uses provided hermes_home/sandboxes.
        let home = std::env::temp_dir().join(format!("hermes-home-{}", new_session_id()));
        let p2 = get_sandbox_dir(&home).unwrap();
        assert_eq!(p2, home.join("sandboxes"));
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn activity_touch_throttles() {
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let f2 = fired.clone();
        set_activity_callback(Some(Box::new(move |_m: &str| {
            f2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })));
        let mut state = ActivityState::new();
        // Just created -> not due (interval 10s).
        touch_activity_if_due(&mut state, "lbl");
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 0);
        // Force it due.
        state.last_touch = Instant::now() - std::time::Duration::from_secs(20);
        touch_activity_if_due(&mut state, "lbl");
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 1);
        set_activity_callback(None);
        assert!(!has_activity_callback());
    }

    // --- build_execute_plan tests via a stub backend --------------------

    struct StubEnv {
        snap: SnapshotState,
        cwd: String,
        timeout: i32,
        mode: StdinMode,
    }

    impl BaseEnvironment for StubEnv {
        type Handle = ();
        fn snapshot(&self) -> &SnapshotState {
            &self.snap
        }
        fn snapshot_mut(&mut self) -> &mut SnapshotState {
            &mut self.snap
        }
        fn cwd(&self) -> &str {
            &self.cwd
        }
        fn set_cwd(&mut self, cwd: String) {
            self.cwd = cwd;
        }
        fn timeout(&self) -> i32 {
            self.timeout
        }
        fn stdin_mode(&self) -> StdinMode {
            self.mode
        }
        fn run_bash(
            &mut self,
            _c: &str,
            _l: bool,
            _t: i32,
            _s: Option<&str>,
        ) -> std::io::Result<()> {
            Ok(())
        }
        fn cleanup(&mut self) {}
    }

    fn stub(mode: StdinMode, ready: bool) -> StubEnv {
        let mut snap = SnapshotState::new("/tmp", "sess");
        snap.snapshot_ready = ready;
        StubEnv {
            snap,
            cwd: "/base".into(),
            timeout: 120,
            mode,
        }
    }

    #[test]
    fn plan_defaults_and_login() {
        let env = stub(StdinMode::Pipe, false);
        let plan = build_execute_plan(&env, "ls", "", None, None);
        assert_eq!(plan.effective_timeout, 120);
        assert_eq!(plan.effective_cwd, "/base");
        assert!(plan.login); // snapshot not ready -> login shell
        assert_eq!(plan.effective_stdin, None);
        assert!(plan.wrapped.contains("eval 'ls'"));
    }

    #[test]
    fn plan_timeout_zero_falls_back() {
        let env = stub(StdinMode::Pipe, true);
        let plan = build_execute_plan(&env, "x", "/c", Some(0), None);
        assert_eq!(plan.effective_timeout, 120); // 0 is falsy -> self.timeout
        assert_eq!(plan.effective_cwd, "/c");
        assert!(!plan.login);
    }

    #[test]
    fn plan_heredoc_embeds_stdin() {
        let env = stub(StdinMode::Heredoc, true);
        let plan = build_execute_plan(&env, "cat", "", None, Some("data"));
        // stdin embedded -> effective_stdin None, command wrapped with heredoc.
        assert_eq!(plan.effective_stdin, None);
        assert!(plan.wrapped.contains("HERMES_STDIN_"));
        assert!(plan.wrapped.contains("\ndata\n"));
    }

    #[test]
    fn plan_empty_stdin_not_embedded() {
        let env = stub(StdinMode::Heredoc, true);
        let plan = build_execute_plan(&env, "cat", "", None, Some(""));
        // Empty stdin is falsy in Python -> not embedded; passed through as Some("").
        assert_eq!(plan.effective_stdin.as_deref(), Some(""));
        assert!(!plan.wrapped.contains("HERMES_STDIN_"));
    }

    #[test]
    fn plan_pipe_keeps_stdin() {
        let env = stub(StdinMode::Pipe, true);
        let plan = build_execute_plan(&env, "cat", "", None, Some("data"));
        assert_eq!(plan.effective_stdin.as_deref(), Some("data"));
        assert!(!plan.wrapped.contains("HERMES_STDIN_"));
    }
}
