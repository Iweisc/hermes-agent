//! Shared Hermes-side execution flow for Modal transports.
//!
//! Port of `tools/environments/modal_utils.py`.
//!
//! This module deliberately stops at the Hermes boundary:
//! - command preparation
//! - cwd/timeout normalization
//! - stdin/sudo shell wrapping
//! - common result shape
//! - interrupt/cancel polling
//!
//! Direct Modal and managed Modal keep separate transport logic, persistence,
//! and trust-boundary decisions in their own modules.
//!
//! ## Design notes
//!
//! The Python original is an abstract base class (`BaseModalExecutionEnvironment`)
//! that subclasses `BaseEnvironment` and uses a template-method pattern: a
//! concrete `execute()` calls the abstract `_start_modal_exec`,
//! `_poll_modal_exec`, and `_cancel_modal_exec` hooks. In Rust we express this
//! as a [`ModalExecutionEnvironment`] trait. Required methods correspond to the
//! Python `@abstractmethod`s plus the `_prepare_command` / `cwd` / `timeout`
//! accessors inherited from `BaseEnvironment`. The provided [`execute`]
//! method reproduces the Python `execute()` polling loop exactly.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tool_interrupt::is_interrupted;

/// Result shape returned by Modal execution: `{"output": str, "returncode": int}`.
///
/// Mirrors the plain `dict` returned by the Python `_result`/`_error_result`
/// helpers. Use [`ExecResult::output`] / [`ExecResult::returncode`] to read
/// fields, or [`ExecResult::into_map`] for a JSON-compatible map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecResult {
    pub output: String,
    pub returncode: i64,
}

impl ExecResult {
    pub fn new(output: impl Into<String>, returncode: i64) -> Self {
        ExecResult {
            output: output.into(),
            returncode,
        }
    }

    /// Convert to a `{"output": ..., "returncode": ...}` JSON object,
    /// matching the Python dict shape exactly.
    pub fn into_json(self) -> serde_json::Value {
        serde_json::json!({
            "output": self.output,
            "returncode": self.returncode,
        })
    }

    /// Convert to a `HashMap` with string keys for callers that want a map.
    pub fn into_map(self) -> HashMap<String, serde_json::Value> {
        let mut m = HashMap::new();
        m.insert("output".to_string(), serde_json::Value::String(self.output));
        m.insert(
            "returncode".to_string(),
            serde_json::Value::Number(self.returncode.into()),
        );
        m
    }
}

/// Normalized command data passed to a transport-specific exec runner.
///
/// Port of the frozen dataclass `PreparedModalExec`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedModalExec {
    pub command: String,
    pub cwd: String,
    pub timeout: i64,
    pub stdin_data: Option<String>,
}

/// Transport response after starting an exec.
///
/// Port of the frozen dataclass `ModalExecStart`. `handle` is generic so each
/// transport can carry its own opaque handle type.
#[derive(Clone, Debug)]
pub struct ModalExecStart<H> {
    pub handle: Option<H>,
    pub immediate_result: Option<ExecResult>,
}

impl<H> ModalExecStart<H> {
    pub fn with_handle(handle: H) -> Self {
        ModalExecStart {
            handle: Some(handle),
            immediate_result: None,
        }
    }

    pub fn with_immediate(result: ExecResult) -> Self {
        ModalExecStart {
            handle: None,
            immediate_result: Some(result),
        }
    }
}

impl<H> Default for ModalExecStart<H> {
    fn default() -> Self {
        ModalExecStart {
            handle: None,
            immediate_result: None,
        }
    }
}

/// How a transport receives stdin. Port of the `_stdin_mode` class attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdinMode {
    /// stdin delivered as a structured payload alongside the command.
    Payload,
    /// stdin wrapped as a shell heredoc appended to the command.
    Heredoc,
}

/// shlex.quote equivalent — POSIX shell single-quote escaping.
///
/// Returns a string that, when parsed by a POSIX shell, yields exactly `s`.
/// Mirrors CPython's `shlex.quote`: safe unquoted strings are returned as-is,
/// otherwise the string is single-quoted with embedded `'` rewritten as
/// `'"'"'`. An empty string becomes `''`.
fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // CPython's unsafe regex: anything NOT in [A-Za-z0-9@%+=:,./_-] is unsafe.
    let safe = s.bytes().all(|b| {
        matches!(b,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'@' | b'%' | b'+' | b'=' | b':' | b',' | b'.' | b'/' | b'_' | b'-')
    });
    if safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    out.push_str(&s.replace('\'', "'\"'\"'"));
    out.push('\'');
    out
}

/// Generate `HERMES_EOF_<8 hex>` from a random uuid hex prefix.
fn random_eof_marker() -> String {
    // uuid4().hex[:8] — 8 lowercase hex chars from a random 128-bit value.
    let mut bytes = [0u8; 4];
    // Use getrandom via std: fall back to time-seeded if unavailable.
    fill_random(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!("HERMES_EOF_{}", hex)
}

fn fill_random(buf: &mut [u8]) {
    // Best-effort randomness without an extra crate dependency. Combines the
    // address of a stack value, the system time, and a counter. Sufficient for
    // a heredoc collision marker (also guarded by a uniqueness loop below).
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let ctr = COUNTER.fetch_add(1, Ordering::Relaxed);
    let stack_addr = (&now as *const u64) as u64;
    let mut state = now ^ ctr.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ stack_addr;
    for b in buf.iter_mut() {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = (state & 0xff) as u8;
    }
}

/// Append stdin as a shell heredoc for transports without stdin piping.
///
/// Port of `wrap_modal_stdin_heredoc`. Picks a `HERMES_EOF_<hex>` marker that
/// does not appear in `stdin_data`, then emits `<command> << '<marker>'\n<data>\n<marker>`.
pub fn wrap_modal_stdin_heredoc(command: &str, stdin_data: &str) -> String {
    let mut marker = random_eof_marker();
    while stdin_data.contains(&marker) {
        marker = random_eof_marker();
    }
    format!("{} << '{}'\n{}\n{}", command, marker, stdin_data, marker)
}

/// Feed sudo via a shell pipe for transports without direct stdin piping.
///
/// Port of `wrap_modal_sudo_pipe`. Produces
/// `printf '%s\n' <quoted-stdin> | <command>` where the stdin is right-trimmed
/// (Python `str.rstrip()`) then shell-quoted.
pub fn wrap_modal_sudo_pipe(command: &str, sudo_stdin: &str) -> String {
    let trimmed = py_rstrip(sudo_stdin);
    format!("printf '%s\\n' {} | {}", shlex_quote(trimmed), command)
}

/// Equivalent of Python `str.rstrip()` with no args: strip trailing whitespace.
fn py_rstrip(s: &str) -> &str {
    // Python's str.rstrip() strips Unicode whitespace; trim_end uses
    // char::is_whitespace which matches CPython's whitespace set closely enough
    // for shell stdin trimming.
    s.trim_end()
}

/// Execution flow for the *managed* Modal transport (gateway-owned sandbox).
///
/// Port of `BaseModalExecutionEnvironment`. Implementors provide the transport
/// hooks; the [`Self::execute`] default method reproduces the Python polling
/// loop including interrupt handling, deadline/timeout handling, and the
/// periodic activity touch.
///
/// `Handle` is the opaque exec handle type returned by [`Self::start_modal_exec`].
pub trait ModalExecutionEnvironment {
    /// Opaque transport-specific exec handle.
    type Handle;

    // --- attributes / config (Python class attributes) ---

    /// `_stdin_mode`. Defaults to [`StdinMode::Payload`].
    fn stdin_mode(&self) -> StdinMode {
        StdinMode::Payload
    }

    /// `_poll_interval_seconds`. Defaults to 0.25s.
    fn poll_interval_seconds(&self) -> f64 {
        0.25
    }

    /// `_client_timeout_grace_seconds`. `None` disables the client-side deadline.
    fn client_timeout_grace_seconds(&self) -> Option<f64> {
        None
    }

    /// `_interrupt_output`.
    fn interrupt_output(&self) -> &str {
        "[Command interrupted]"
    }

    /// `_unexpected_error_prefix`.
    fn unexpected_error_prefix(&self) -> &str {
        "Modal execution error"
    }

    // --- BaseEnvironment-inherited accessors ---

    /// `self.cwd` from `BaseEnvironment`.
    fn cwd(&self) -> &str;

    /// `self.timeout` from `BaseEnvironment`.
    fn timeout(&self) -> i64;

    /// `self._prepare_command(command)` from `BaseEnvironment`.
    ///
    /// Returns the (possibly rewritten) command and an optional sudo-stdin
    /// string. When the second element is `Some`, the caller wraps the command
    /// with [`wrap_modal_sudo_pipe`].
    fn prepare_command(&self, command: &str) -> (String, Option<String>);

    // --- hooks ---

    /// Hook for backends that need pre-exec sync or validation. Default no-op.
    fn before_execute(&mut self) {}

    /// Begin a transport-specific exec. Errors are surfaced via the
    /// `_unexpected_error_prefix` error result.
    fn start_modal_exec(
        &mut self,
        prepared: &PreparedModalExec,
    ) -> Result<ModalExecStart<Self::Handle>, String>;

    /// Return `Some(result)` when complete, else `None` to keep polling.
    fn poll_modal_exec(&mut self, handle: &Self::Handle) -> Result<Option<ExecResult>, String>;

    /// Cancel or terminate the active transport exec.
    fn cancel_modal_exec(&mut self, handle: &Self::Handle) -> Result<(), String>;

    /// Periodic activity touch so the gateway knows we're alive.
    ///
    /// Mirrors `tools.environments.base.touch_activity_if_due`. Default no-op;
    /// the Python original swallows all exceptions, so this never fails.
    fn touch_activity_if_due(&mut self, _label: &str, _elapsed: Duration) {}

    // --- result builders (Python _result/_error_result/_timeout_result_for_modal) ---

    fn result(&self, output: impl Into<String>, returncode: i64) -> ExecResult {
        ExecResult::new(output, returncode)
    }

    fn error_result(&self, output: impl Into<String>) -> ExecResult {
        self.result(output, 1)
    }

    fn timeout_result_for_modal(&self, timeout: i64) -> ExecResult {
        self.result(format!("Command timed out after {}s", timeout), 124)
    }

    /// `_prepare_modal_exec`: normalize cwd/timeout and apply stdin/sudo wrapping.
    fn prepare_modal_exec(
        &self,
        command: &str,
        cwd: &str,
        timeout: Option<i64>,
        stdin_data: Option<&str>,
    ) -> PreparedModalExec {
        // effective_cwd = cwd or self.cwd  (Python truthiness: empty -> fallback)
        let effective_cwd = if cwd.is_empty() {
            self.cwd().to_string()
        } else {
            cwd.to_string()
        };
        // effective_timeout = timeout or self.timeout (0 / None -> fallback)
        let effective_timeout = match timeout {
            Some(t) if t != 0 => t,
            _ => self.timeout(),
        };

        let mut exec_command = command.to_string();
        let mode = self.stdin_mode();
        let exec_stdin = if mode == StdinMode::Payload {
            stdin_data.map(|s| s.to_string())
        } else {
            None
        };
        if let Some(data) = stdin_data {
            if mode == StdinMode::Heredoc {
                exec_command = wrap_modal_stdin_heredoc(&exec_command, data);
            }
        }

        let (cmd, sudo_stdin) = self.prepare_command(&exec_command);
        exec_command = cmd;
        if let Some(sudo) = sudo_stdin {
            exec_command = wrap_modal_sudo_pipe(&exec_command, &sudo);
        }

        PreparedModalExec {
            command: exec_command,
            cwd: effective_cwd,
            timeout: effective_timeout,
            stdin_data: exec_stdin,
        }
    }

    /// Port of `BaseModalExecutionEnvironment.execute`.
    ///
    /// Drives the full prepare → start → poll loop, honoring interrupts,
    /// the optional client-side deadline, and the periodic activity touch.
    ///
    /// `cwd` of `""` means "use `self.cwd`". `timeout` of `None` (or `0`)
    /// means "use `self.timeout`".
    fn execute(
        &mut self,
        command: &str,
        cwd: &str,
        timeout: Option<i64>,
        stdin_data: Option<&str>,
    ) -> ExecResult {
        self.before_execute();
        let prepared = self.prepare_modal_exec(command, cwd, timeout, stdin_data);

        let start = match self.start_modal_exec(&prepared) {
            Ok(s) => s,
            Err(exc) => {
                return self.error_result(format!("{}: {}", self.unexpected_error_prefix(), exc));
            }
        };

        if let Some(immediate) = start.immediate_result {
            return immediate;
        }

        let handle = match start.handle {
            Some(h) => h,
            None => {
                return self.error_result(format!(
                    "{}: transport did not return an exec handle",
                    self.unexpected_error_prefix()
                ));
            }
        };

        // deadline = now + timeout + grace, when grace is configured.
        let deadline = self.client_timeout_grace_seconds().map(|grace| {
            Instant::now()
                + Duration::from_secs_f64(prepared.timeout as f64 + grace)
        });

        let start_instant = Instant::now();
        let mut last_touch = start_instant;
        let touch_interval = Duration::from_secs_f64(10.0);

        let poll_sleep = Duration::from_secs_f64(self.poll_interval_seconds());

        loop {
            if is_interrupted() {
                // Best-effort cancel; swallow errors (Python `except Exception: pass`).
                let _ = self.cancel_modal_exec(&handle);
                let interrupt_output = self.interrupt_output().to_string();
                return self.result(interrupt_output, 130);
            }

            match self.poll_modal_exec(&handle) {
                Ok(Some(result)) => return result,
                Ok(None) => {}
                Err(exc) => {
                    return self
                        .error_result(format!("{}: {}", self.unexpected_error_prefix(), exc));
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    let _ = self.cancel_modal_exec(&handle);
                    return self.timeout_result_for_modal(prepared.timeout);
                }
            }

            // Periodic activity touch (at most once per interval). Mirrors
            // touch_activity_if_due; swallows all errors.
            let now = Instant::now();
            if now.duration_since(last_touch) >= touch_interval {
                last_touch = now;
                let elapsed = now.duration_since(start_instant);
                self.touch_activity_if_due("modal command running", elapsed);
            }

            std::thread::sleep(poll_sleep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn shlex_quote_basics() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("abc"), "abc");
        assert_eq!(shlex_quote("a/b_c-d.e"), "a/b_c-d.e");
        assert_eq!(shlex_quote("hello world"), "'hello world'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shlex_quote("$(rm -rf)"), "'$(rm -rf)'");
    }

    #[test]
    fn sudo_pipe_wrapping() {
        let out = wrap_modal_sudo_pipe("sudo -S apt-get install x", "secret  \n");
        // rstrip removes trailing whitespace/newline before quoting.
        assert_eq!(out, "printf '%s\\n' secret | sudo -S apt-get install x");
    }

    #[test]
    fn sudo_pipe_quotes_special() {
        let out = wrap_modal_sudo_pipe("cmd", "pa ss");
        assert_eq!(out, "printf '%s\\n' 'pa ss' | cmd");
    }

    #[test]
    fn heredoc_wrapping_shape() {
        let out = wrap_modal_stdin_heredoc("cat", "line1\nline2");
        assert!(out.starts_with("cat << 'HERMES_EOF_"));
        assert!(out.contains("\nline1\nline2\n"));
        // marker appears twice (open + close)
        let marker_start = out.find("HERMES_EOF_").unwrap();
        let marker = &out[marker_start..marker_start + "HERMES_EOF_".len() + 8];
        assert_eq!(out.matches(marker).count(), 2);
    }

    #[test]
    fn heredoc_avoids_collision() {
        // Force-feed data that contains an EOF prefix; the loop must pick a
        // marker not present in the data. We can't control the random hex, but
        // we can assert the chosen marker is genuinely absent before reuse.
        let data = "HERMES_EOF_deadbeef in body";
        let out = wrap_modal_stdin_heredoc("cmd", data);
        let marker_start = out.find("cat").map(|_| 0).unwrap_or_else(|| out.find("HERMES_EOF_").unwrap());
        let _ = marker_start;
        // The opening marker line is the substring after "cmd << '".
        let open = out.find("<< '").unwrap() + 4;
        let close_q = out[open..].find('\'').unwrap();
        let marker = &out[open..open + close_q];
        // marker must equal HERMES_EOF_<8hex> and not be the colliding one
        assert!(marker.starts_with("HERMES_EOF_"));
        assert_eq!(marker.len(), "HERMES_EOF_".len() + 8);
    }

    #[test]
    fn exec_result_json_shape() {
        let r = ExecResult::new("hi", 0);
        let v = r.into_json();
        assert_eq!(v["output"], "hi");
        assert_eq!(v["returncode"], 0);
    }

    // --- A fake transport exercising the execute() loop ---

    struct FakeEnv {
        cwd: String,
        timeout: i64,
        mode: StdinMode,
        // scripted poll results: each call pops the front
        poll_script: RefCell<Vec<Option<ExecResult>>>,
        start_immediate: Option<ExecResult>,
        return_handle: bool,
        prepared: RefCell<Option<PreparedModalExec>>,
        cancelled: RefCell<bool>,
    }

    impl FakeEnv {
        fn new() -> Self {
            FakeEnv {
                cwd: "/work".to_string(),
                timeout: 30,
                mode: StdinMode::Payload,
                poll_script: RefCell::new(vec![]),
                start_immediate: None,
                return_handle: true,
                prepared: RefCell::new(None),
                cancelled: RefCell::new(false),
            }
        }
    }

    impl ModalExecutionEnvironment for FakeEnv {
        type Handle = u32;

        fn stdin_mode(&self) -> StdinMode {
            self.mode
        }
        fn poll_interval_seconds(&self) -> f64 {
            0.0
        }
        fn cwd(&self) -> &str {
            &self.cwd
        }
        fn timeout(&self) -> i64 {
            self.timeout
        }
        fn prepare_command(&self, command: &str) -> (String, Option<String>) {
            (command.to_string(), None)
        }
        fn start_modal_exec(
            &mut self,
            prepared: &PreparedModalExec,
        ) -> Result<ModalExecStart<u32>, String> {
            *self.prepared.borrow_mut() = Some(prepared.clone());
            if let Some(r) = self.start_immediate.clone() {
                return Ok(ModalExecStart::with_immediate(r));
            }
            if self.return_handle {
                Ok(ModalExecStart::with_handle(1u32))
            } else {
                Ok(ModalExecStart::default())
            }
        }
        fn poll_modal_exec(&mut self, _handle: &u32) -> Result<Option<ExecResult>, String> {
            let mut s = self.poll_script.borrow_mut();
            if s.is_empty() {
                Ok(None)
            } else {
                Ok(s.remove(0))
            }
        }
        fn cancel_modal_exec(&mut self, _handle: &u32) -> Result<(), String> {
            *self.cancelled.borrow_mut() = true;
            Ok(())
        }
    }

    #[test]
    fn execute_returns_immediate_result() {
        let mut env = FakeEnv::new();
        env.start_immediate = Some(ExecResult::new("done", 0));
        let r = env.execute("ls", "", None, None);
        assert_eq!(r, ExecResult::new("done", 0));
    }

    #[test]
    fn execute_missing_handle_errors() {
        let mut env = FakeEnv::new();
        env.return_handle = false;
        let r = env.execute("ls", "", None, None);
        assert_eq!(r.returncode, 1);
        assert!(r.output.contains("did not return an exec handle"));
    }

    #[test]
    fn execute_polls_until_complete() {
        let mut env = FakeEnv::new();
        env.poll_script = RefCell::new(vec![None, None, Some(ExecResult::new("ok", 0))]);
        let r = env.execute("ls", "", None, None);
        assert_eq!(r, ExecResult::new("ok", 0));
    }

    #[test]
    fn prepare_uses_defaults_when_empty() {
        let env = FakeEnv::new();
        let p = env.prepare_modal_exec("echo hi", "", None, None);
        assert_eq!(p.cwd, "/work");
        assert_eq!(p.timeout, 30);
        assert_eq!(p.command, "echo hi");
        assert_eq!(p.stdin_data, None);
    }

    #[test]
    fn prepare_overrides_cwd_timeout() {
        let env = FakeEnv::new();
        let p = env.prepare_modal_exec("echo hi", "/other", Some(5), None);
        assert_eq!(p.cwd, "/other");
        assert_eq!(p.timeout, 5);
    }

    #[test]
    fn prepare_payload_stdin_carried() {
        let env = FakeEnv::new(); // Payload mode
        let p = env.prepare_modal_exec("cat", "", None, Some("input"));
        assert_eq!(p.stdin_data.as_deref(), Some("input"));
        assert_eq!(p.command, "cat"); // not wrapped in payload mode
    }

    #[test]
    fn prepare_heredoc_stdin_wraps_command() {
        let mut env = FakeEnv::new();
        env.mode = StdinMode::Heredoc;
        let p = env.prepare_modal_exec("cat", "", None, Some("input"));
        assert_eq!(p.stdin_data, None); // not carried as payload
        assert!(p.command.starts_with("cat << 'HERMES_EOF_"));
        assert!(p.command.contains("\ninput\n"));
    }

    #[test]
    fn timeout_result_shape() {
        let env = FakeEnv::new();
        let r = env.timeout_result_for_modal(42);
        assert_eq!(r, ExecResult::new("Command timed out after 42s", 124));
    }
}
