//! PTY bridge for the `hermes dashboard` chat tab.
//!
//! Wraps a child process behind a pseudo-terminal so its ANSI output can be
//! streamed to a browser-side terminal emulator (xterm.js) and typed
//! keystrokes can be fed back in. The only caller today is the `/api/pty`
//! WebSocket endpoint in the web server.
//!
//! This is a native Rust port of `hermes_cli/pty_bridge.py`.
//!
//! Design constraints:
//!
//! * **POSIX-only.** Hermes Agent supports Windows exclusively via WSL, which
//!   exposes a native POSIX PTY via `openpty(3)`. On native Windows there is
//!   no PTY; [`PtyUnavailableError`] is raised with a user-readable
//!   platform message so the dashboard can render a banner instead of
//!   crashing.
//! * **Byte-safe I/O.** Reads and writes go through the PTY master fd
//!   directly — streaming ANSI is inherently byte-oriented and UTF-8
//!   boundaries may land mid-read, so we never decode here.
//!
//! Unlike the Python implementation, which depends on the pure-Python
//! `ptyprocess` package, this port talks directly to `libc` (`openpty`,
//! `fork`, `execvp`, `select`, `read`, `write`, `ioctl(TIOCSWINSZ)`).

use std::collections::HashMap;
use std::ffi::CString;
use std::time::{Duration, Instant};

/// Error raised when a PTY cannot be created on this platform.
///
/// Today this means native Windows (no ConPTY bindings) or an exec failure
/// while spawning the child. The dashboard surfaces the message to the user
/// as a chat-tab banner.
#[derive(Debug, Clone)]
pub struct PtyUnavailableError(pub String);

impl std::fmt::Display for PtyUnavailableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PtyUnavailableError {}

/// Errors that can occur while spawning a [`PtyBridge`].
#[derive(Debug)]
pub enum SpawnError {
    /// The platform cannot host a PTY at all.
    Unavailable(PtyUnavailableError),
    /// An ordinary OS error (missing binary, bad cwd, openpty/fork failure).
    Os(std::io::Error),
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::Unavailable(e) => write!(f, "{e}"),
            SpawnError::Os(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SpawnError {}

impl From<PtyUnavailableError> for SpawnError {
    fn from(e: PtyUnavailableError) -> Self {
        SpawnError::Unavailable(e)
    }
}

impl From<std::io::Error> for SpawnError {
    fn from(e: std::io::Error) -> Self {
        SpawnError::Os(e)
    }
}

/// True at compile time on non-Windows targets (matching the Python guard
/// `not sys.platform.startswith("win")`).
const PTY_AVAILABLE: bool = cfg!(not(target_os = "windows"));

/// Thin wrapper around a forked child running behind a PTY, for byte streaming.
///
/// Not thread-safe in the sense that the [`PtyBridge`] value must not be
/// shared across threads without external synchronisation. As in the Python
/// original, the kernel PTY is the real synchronisation point: a reader may
/// `read` on one thread while a writer `write`s on another, since both operate
/// directly on the master fd.
pub struct PtyBridge {
    fd: i32,
    pid: i32,
    closed: bool,
}

impl PtyBridge {
    /// True if a PTY can be spawned on this platform.
    pub fn is_available() -> bool {
        PTY_AVAILABLE
    }

    /// Process id of the spawned child.
    pub fn pid(&self) -> i32 {
        self.pid
    }

    /// Spawn `argv` behind a new PTY and return a bridge.
    ///
    /// Returns [`SpawnError::Unavailable`] if the platform can't host a PTY,
    /// and [`SpawnError::Os`] for ordinary exec failures (missing binary,
    /// bad cwd, etc.).
    ///
    /// `env`, when `Some`, fully replaces the child environment; when `None`
    /// the parent environment is copied. In either case `TERM` is backfilled
    /// to `xterm-256color` if missing or blank, matching the Python source.
    #[cfg(not(target_os = "windows"))]
    pub fn spawn(
        argv: &[String],
        cwd: Option<&str>,
        env: Option<&HashMap<String, String>>,
        cols: u16,
        rows: u16,
    ) -> Result<PtyBridge, SpawnError> {
        if argv.is_empty() {
            return Err(SpawnError::Os(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "argv must not be empty",
            )));
        }

        // Build the environment. PTY-hosted programs expect TERM to describe
        // the terminal type; CI often runs without TERM, which makes simple
        // probes like `tput cols` fail. Preserve explicit caller overrides but
        // backfill a sensible default when TERM is missing or blank.
        let mut spawn_env: HashMap<String, String> = match env {
            None => std::env::vars().collect(),
            Some(e) => e.clone(),
        };
        if spawn_env
            .get("TERM")
            .map(|v| v.is_empty())
            .unwrap_or(true)
        {
            spawn_env.insert("TERM".to_string(), "xterm-256color".to_string());
        }

        // Open the PTY pair.
        let mut master_fd: libc::c_int = -1;
        let mut slave_fd: libc::c_int = -1;
        let mut winsize = libc::winsize {
            ws_row: rows.max(1),
            ws_col: cols.max(1),
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let rc = unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut winsize,
            )
        };
        if rc != 0 {
            return Err(SpawnError::Os(std::io::Error::last_os_error()));
        }

        // Pre-compute everything we need post-fork (the child must avoid
        // non-async-signal-safe allocation where possible; we precompute).
        let argv_c: Vec<CString> = argv
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                SpawnError::Os(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "argv contained NUL byte",
                ))
            })?;
        let mut argv_ptrs: Vec<*const libc::c_char> =
            argv_c.iter().map(|c| c.as_ptr()).collect();
        argv_ptrs.push(std::ptr::null());

        // Pre-build NUL-terminated key/value CStrings so the child can call
        // `setenv` without allocating. `execvpe` is not portable across all
        // libc targets (notably macOS/BSD), so we apply the environment with
        // `setenv` in the child and then `execvp`.
        let env_kv: Vec<(CString, CString)> = spawn_env
            .iter()
            .map(|(k, v)| {
                Ok((
                    CString::new(k.as_bytes())?,
                    CString::new(v.as_bytes())?,
                ))
            })
            .collect::<Result<Vec<_>, std::ffi::NulError>>()
            .map_err(|_| {
                SpawnError::Os(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "env contained NUL byte",
                ))
            })?;

        let cwd_c: Option<CString> = match cwd {
            Some(d) => Some(CString::new(d.as_bytes()).map_err(|_| {
                SpawnError::Os(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "cwd contained NUL byte",
                ))
            })?),
            None => None,
        };

        let pid = unsafe { libc::fork() };
        if pid < 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(master_fd);
                libc::close(slave_fd);
            }
            return Err(SpawnError::Os(err));
        }

        if pid == 0 {
            // -- Child --
            unsafe {
                // New session so the slave becomes the controlling terminal.
                libc::setsid();
                // Make the slave the controlling tty. The request argument
                // type differs per platform (c_ulong on BSD/macOS, c_int on
                // some others); cast to whatever `ioctl` expects.
                libc::ioctl(slave_fd, libc::TIOCSCTTY as _, 0);

                // Wire stdio to the slave.
                libc::dup2(slave_fd, 0);
                libc::dup2(slave_fd, 1);
                libc::dup2(slave_fd, 2);
                if slave_fd > 2 {
                    libc::close(slave_fd);
                }
                libc::close(master_fd);

                if let Some(ref d) = cwd_c {
                    if libc::chdir(d.as_ptr()) != 0 {
                        libc::_exit(127);
                    }
                }

                // Apply the (already fully resolved) environment. We clear the
                // inherited environment first so that the child's environment
                // matches `spawn_env` exactly, mirroring Python's `env=`
                // semantics of full replacement.
                #[cfg(not(target_os = "macos"))]
                {
                    libc::clearenv();
                }
                #[cfg(target_os = "macos")]
                {
                    // macOS has no clearenv(); empty out the environ vector by
                    // unsetting each existing entry instead. `environ` is not
                    // surfaced by libc on macOS, so declare it directly.
                    unsafe extern "C" {
                        static mut environ: *mut *mut libc::c_char;
                    }
                    loop {
                        let envp = environ;
                        if envp.is_null() || (*envp).is_null() {
                            break;
                        }
                        let entry = std::ffi::CStr::from_ptr(*envp);
                        let mut removed = false;
                        if let Some(eq) =
                            entry.to_bytes().iter().position(|&b| b == b'=')
                        {
                            let name = &entry.to_bytes()[..eq];
                            if let Ok(name_c) = CString::new(name) {
                                // unsetenv compacts environ in place; restart
                                // from the top on the next loop iteration.
                                libc::unsetenv(name_c.as_ptr());
                                removed = true;
                            }
                        }
                        if !removed {
                            break;
                        }
                    }
                }
                for (k, v) in &env_kv {
                    libc::setenv(k.as_ptr(), v.as_ptr(), 1);
                }

                libc::execvp(argv_ptrs[0], argv_ptrs.as_ptr());
                // execvp only returns on failure.
                libc::_exit(127);
            }
        }

        // -- Parent --
        unsafe {
            libc::close(slave_fd);
        }

        Ok(PtyBridge {
            fd: master_fd,
            pid,
            closed: false,
        })
    }

    /// Windows stub: always unavailable, matching the Python platform guard.
    #[cfg(target_os = "windows")]
    pub fn spawn(
        _argv: &[String],
        _cwd: Option<&str>,
        _env: Option<&HashMap<String, String>>,
        _cols: u16,
        _rows: u16,
    ) -> Result<PtyBridge, SpawnError> {
        Err(SpawnError::Unavailable(PtyUnavailableError(
            "Pseudo-terminals are unavailable on this platform. \
             Hermes Agent supports Windows only via WSL."
                .to_string(),
        )))
    }

    /// True if the child is still running.
    pub fn is_alive(&self) -> bool {
        if self.closed {
            return false;
        }
        self.proc_is_alive()
    }

    /// Internal: non-reaping liveness probe via `waitpid(WNOHANG)`.
    #[cfg(not(target_os = "windows"))]
    fn proc_is_alive(&self) -> bool {
        let mut status: libc::c_int = 0;
        let rc = unsafe { libc::waitpid(self.pid, &mut status, libc::WNOHANG) };
        // rc == 0  -> still running
        // rc == pid -> just exited (reaped)
        // rc < 0   -> error (already reaped / no such child)
        rc == 0
    }

    #[cfg(target_os = "windows")]
    fn proc_is_alive(&self) -> bool {
        false
    }

    /// Read up to 64 KiB of raw bytes from the PTY master.
    ///
    /// Returns:
    /// * `Some(bytes)` with one or more bytes of child output;
    /// * `Some(empty)` — no data available within `timeout`;
    /// * `None` — child has exited and the master fd is at EOF (or the
    ///   bridge is closed / the select failed).
    ///
    /// Never blocks longer than `timeout`. Safe to call after [`close`](Self::close);
    /// returns `None` in that case.
    #[cfg(not(target_os = "windows"))]
    pub fn read(&self, timeout: Duration) -> Option<Vec<u8>> {
        if self.closed {
            return None;
        }

        // select() with the configured timeout.
        let mut readfds: libc::fd_set = unsafe { std::mem::zeroed() };
        unsafe {
            libc::FD_ZERO(&mut readfds);
            libc::FD_SET(self.fd, &mut readfds);
        }
        let mut tv = libc::timeval {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_usec: timeout.subsec_micros() as libc::suseconds_t,
        };
        let rc = unsafe {
            libc::select(
                self.fd + 1,
                &mut readfds,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut tv,
            )
        };
        if rc < 0 {
            // Treat select errors like the Python (OSError, ValueError) branch.
            return None;
        }
        let readable = unsafe { libc::FD_ISSET(self.fd, &readfds) };
        if rc == 0 || !readable {
            return Some(Vec::new());
        }

        let mut buf = vec![0u8; 65536];
        let n = unsafe {
            libc::read(
                self.fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            match err.raw_os_error() {
                // EIO on Linux = slave side closed. EBADF = already closed.
                Some(e) if e == libc::EIO || e == libc::EBADF => return None,
                _ => return None,
            }
        }
        if n == 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(buf)
    }

    #[cfg(target_os = "windows")]
    pub fn read(&self, _timeout: Duration) -> Option<Vec<u8>> {
        None
    }

    /// Write raw bytes to the PTY master (i.e. the child's stdin).
    ///
    /// Loops until the buffer is drained, tolerating short writes. Silently
    /// returns on EIO/EBADF/EPIPE, matching the Python original.
    #[cfg(not(target_os = "windows"))]
    pub fn write(&self, data: &[u8]) {
        if self.closed || data.is_empty() {
            return;
        }
        let mut offset = 0usize;
        while offset < data.len() {
            let n = unsafe {
                libc::write(
                    self.fd,
                    data[offset..].as_ptr() as *const libc::c_void,
                    data.len() - offset,
                )
            };
            if n < 0 {
                let err = std::io::Error::last_os_error();
                match err.raw_os_error() {
                    Some(e)
                        if e == libc::EIO
                            || e == libc::EBADF
                            || e == libc::EPIPE =>
                    {
                        return;
                    }
                    _ => return,
                }
            }
            if n <= 0 {
                return;
            }
            offset += n as usize;
        }
    }

    #[cfg(target_os = "windows")]
    pub fn write(&self, _data: &[u8]) {}

    /// Forward a terminal resize to the child via `TIOCSWINSZ`.
    #[cfg(not(target_os = "windows"))]
    pub fn resize(&self, cols: u16, rows: u16) {
        if self.closed {
            return;
        }
        let winsize = libc::winsize {
            ws_row: rows.max(1),
            ws_col: cols.max(1),
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // Errors are intentionally ignored, as in the Python original.
        unsafe {
            libc::ioctl(self.fd, libc::TIOCSWINSZ, &winsize);
        }
    }

    #[cfg(target_os = "windows")]
    pub fn resize(&self, _cols: u16, _rows: u16) {}

    /// Terminate the child (SIGHUP -> SIGTERM -> SIGKILL, 0.5s grace each)
    /// and close the master fd.
    ///
    /// Idempotent. Reaping the child is important so we don't leak zombies
    /// across the lifetime of the dashboard process.
    #[cfg(not(target_os = "windows"))]
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;

        // SIGHUP is the conventional "your terminal went away" signal. We
        // escalate if the child ignores it.
        for sig in [libc::SIGHUP, libc::SIGTERM, libc::SIGKILL] {
            if !self.proc_is_alive() {
                break;
            }
            unsafe {
                libc::kill(self.pid, sig);
            }
            let deadline = Instant::now() + Duration::from_millis(500);
            while self.proc_is_alive() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        // Reap (in case it exited) and close the master fd.
        let mut status: libc::c_int = 0;
        unsafe {
            libc::waitpid(self.pid, &mut status, libc::WNOHANG);
            libc::close(self.fd);
        }
    }

    #[cfg(target_os = "windows")]
    pub fn close(&mut self) {
        self.closed = true;
    }
}

impl Drop for PtyBridge {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod tests {
    use super::*;

    #[test]
    fn is_available_is_true_on_posix() {
        assert!(PtyBridge::is_available());
    }

    #[test]
    fn spawn_empty_argv_errors() {
        let argv: Vec<String> = vec![];
        let res = PtyBridge::spawn(&argv, None, None, 80, 24);
        assert!(matches!(res, Err(SpawnError::Os(_))));
    }

    #[test]
    fn spawn_and_read_echo_output() {
        let argv = vec![
            "/bin/echo".to_string(),
            "hello-pty".to_string(),
        ];
        let mut bridge =
            PtyBridge::spawn(&argv, None, None, 80, 24).expect("spawn echo");
        assert!(bridge.pid() > 0);

        // Drain output until EOF (None) or we collect the marker.
        let mut collected = Vec::new();
        for _ in 0..200 {
            match bridge.read(Duration::from_millis(100)) {
                Some(b) if !b.is_empty() => collected.extend_from_slice(&b),
                Some(_) => {}
                None => break,
            }
            if String::from_utf8_lossy(&collected).contains("hello-pty") {
                break;
            }
        }
        let text = String::from_utf8_lossy(&collected);
        assert!(
            text.contains("hello-pty"),
            "expected echo output, got: {text:?}"
        );
        bridge.close();
        // Idempotent.
        bridge.close();
    }

    #[test]
    fn read_after_close_is_none() {
        let argv = vec!["/bin/cat".to_string()];
        let mut bridge =
            PtyBridge::spawn(&argv, None, None, 80, 24).expect("spawn cat");
        bridge.close();
        assert!(bridge.read(Duration::from_millis(10)).is_none());
        assert!(!bridge.is_alive());
    }

    #[test]
    fn write_then_read_roundtrip_through_cat() {
        // `cat` echoes stdin; the PTY also echoes input by default, so we
        // should see our payload come back at least once.
        let argv = vec!["/bin/cat".to_string()];
        let bridge =
            PtyBridge::spawn(&argv, None, None, 80, 24).expect("spawn cat");
        bridge.write(b"ping\n");

        let mut collected = Vec::new();
        for _ in 0..50 {
            if let Some(b) = bridge.read(Duration::from_millis(100)) {
                collected.extend_from_slice(&b);
                if String::from_utf8_lossy(&collected).contains("ping") {
                    break;
                }
            }
        }
        assert!(String::from_utf8_lossy(&collected).contains("ping"));
    }

    #[test]
    fn term_backfilled_when_env_provided_without_term() {
        // Spawn `env` with a TERM-less environment and confirm the backfill
        // surfaces in the child's view of the environment.
        let mut env: HashMap<String, String> = HashMap::new();
        env.insert("PATH".to_string(), "/usr/bin:/bin".to_string());
        let argv = vec!["/usr/bin/env".to_string()];
        let bridge =
            PtyBridge::spawn(&argv, None, Some(&env), 80, 24).expect("spawn env");

        let mut collected = Vec::new();
        for _ in 0..100 {
            match bridge.read(Duration::from_millis(100)) {
                Some(b) if !b.is_empty() => collected.extend_from_slice(&b),
                Some(_) => {}
                None => break,
            }
        }
        let text = String::from_utf8_lossy(&collected);
        assert!(
            text.contains("TERM=xterm-256color"),
            "expected backfilled TERM, got: {text:?}"
        );
    }

    #[test]
    fn resize_on_live_child_does_not_panic() {
        let argv = vec!["/bin/cat".to_string()];
        let bridge =
            PtyBridge::spawn(&argv, None, None, 80, 24).expect("spawn cat");
        bridge.resize(120, 40);
        // 0 dims are clamped to 1 internally; should not panic.
        bridge.resize(0, 0);
    }
}
