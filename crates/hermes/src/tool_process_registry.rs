//! Process Registry — in-memory registry for managed background processes.
//!
//! Native Rust port of `tools/process_registry.py`.
//!
//! Tracks processes spawned via `terminal(background=true)`, providing:
//!   - Output buffering (rolling 200KB window)
//!   - Status polling and log retrieval
//!   - Blocking wait with interrupt support
//!   - Process killing
//!   - Crash recovery via JSON checkpoint file
//!   - Session-scoped tracking for gateway reset protection
//!
//! Background processes execute through the environment interface. For the
//! local backend the command runs on the host via the user's login shell; for
//! sandbox backends an [`EnvExecutor`] runs the command inside the sandbox.
//!
//! Differences from the Python original that are unavoidable in Rust:
//!   - PTY mode (`ptyprocess`/`winpty`) is not backed by an allowed crate, so
//!     `spawn_local(use_pty=true)` transparently falls back to the standard
//!     pipe-based `Popen` path, exactly as the Python code does when
//!     `ptyprocess` is not installed.
//!   - The orphaned-pipe non-blocking drain in `_reconcile_local_exit` is
//!     approximated: when the direct child has exited we flip the session to
//!     `exited` and rely on the reader thread reaping remaining bytes.

use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Write};
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use hermes_core::tool_ansi_strip::strip_ansi;

// ---------------------------------------------------------------------------
// Constants (mirror the Python module-level constants)
// ---------------------------------------------------------------------------

/// 200KB rolling output buffer.
pub const MAX_OUTPUT_CHARS: usize = 200_000;
/// Keep finished processes for 30 minutes.
pub const FINISHED_TTL_SECONDS: f64 = 1800.0;
/// Max concurrent tracked processes (LRU pruning).
pub const MAX_PROCESSES: usize = 64;

/// Minimum spacing between consecutive watch matches.
pub const WATCH_MIN_INTERVAL_SECONDS: f64 = 15.0;
/// Strikes in a row → disable watch + promote to notify_on_complete.
pub const WATCH_STRIKE_LIMIT: u32 = 3;

/// Global circuit breaker — across all sessions.
pub const WATCH_GLOBAL_MAX_PER_WINDOW: u32 = 15;
pub const WATCH_GLOBAL_WINDOW_SECONDS: f64 = 10.0;
pub const WATCH_GLOBAL_COOLDOWN_SECONDS: f64 = 30.0;

const SHELL_NOISE_SUBSTRINGS: &[&str] = &[
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
];

const IS_WINDOWS: bool = cfg!(windows);

// ---------------------------------------------------------------------------
// format_uptime_short
// ---------------------------------------------------------------------------

/// Faithful port of `format_uptime_short`.
pub fn format_uptime_short(seconds: i64) -> String {
    let s = seconds.max(0);
    if s < 60 {
        return format!("{s}s");
    }
    let (mins, secs) = (s / 60, s % 60);
    if mins < 60 {
        return format!("{mins}m {secs}s");
    }
    let (hours, mins) = (mins / 60, mins % 60);
    format!("{hours}h {mins}m")
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Environment executor abstraction (for non-local sandbox backends)
// ---------------------------------------------------------------------------

/// Minimal abstraction over a sandbox environment's `execute()` interface.
///
/// The Python code calls `env.execute(command, timeout=...)` and reads the
/// `"output"` key from the returned dict. Sandbox backends implement this
/// trait so the registry can spawn/poll/kill inside the sandbox.
pub trait EnvExecutor: Send + Sync {
    /// Execute `command` in the sandbox, returning its captured output.
    fn execute(&self, command: &str, timeout: u64) -> std::io::Result<String>;

    /// Return the writable sandbox temp dir, if the backend exposes one.
    fn get_temp_dir(&self) -> Option<String> {
        None
    }
}

// ---------------------------------------------------------------------------
// ProcessSession
// ---------------------------------------------------------------------------

/// The mutable per-session state guarded by the session lock.
#[derive(Default)]
struct SessionInner {
    pid: Option<i32>,
    cwd: Option<String>,
    started_at: f64,
    exited: bool,
    exit_code: Option<i32>,
    output_buffer: String,
    max_output_chars: usize,
    detached: bool,
    pid_scope: String,

    watcher_platform: String,
    watcher_chat_id: String,
    watcher_user_id: String,
    watcher_user_name: String,
    watcher_thread_id: String,
    watcher_interval: i64,
    notify_on_complete: bool,
    watch_patterns: Vec<String>,

    watch_hits: u64,
    watch_suppressed: u64,
    watch_disabled: bool,
    watch_last_emit_at: f64,
    watch_cooldown_until: f64,
    watch_strike_candidate: bool,
    watch_consecutive_strikes: u32,

    // Live OS handles (local Popen). None for env/PTY/detached sessions.
    child: Option<Child>,
    // Captured stdin handle for write/submit/close.
    stdin: Option<std::process::ChildStdin>,
}

/// A tracked background process with output buffering.
///
/// The session is shared between the registry, reader/poller threads, and the
/// gateway. The interior state is guarded by `inner`.
pub struct ProcessSession {
    /// Unique session ID ("proc_xxxxxxxxxxxx").
    pub id: String,
    /// Original command string.
    pub command: String,
    /// Task/sandbox isolation key.
    pub task_id: String,
    /// Gateway session key (for reset protection).
    pub session_key: String,
    /// Reference to the environment object (sandbox backends only).
    env_ref: Option<Arc<dyn EnvExecutor>>,
    inner: Mutex<SessionInner>,
}

impl ProcessSession {
    fn new(id: String, command: String) -> Arc<Self> {
        Arc::new(ProcessSession {
            id,
            command,
            task_id: String::new(),
            session_key: String::new(),
            env_ref: None,
            inner: Mutex::new(SessionInner {
                max_output_chars: MAX_OUTPUT_CHARS,
                pid_scope: "host".to_string(),
                ..Default::default()
            }),
        })
    }

    /// Whether the process has finished.
    pub fn exited(&self) -> bool {
        self.inner.lock().unwrap().exited
    }

    /// Exit code (None if still running).
    pub fn exit_code(&self) -> Option<i32> {
        self.inner.lock().unwrap().exit_code
    }

    /// OS process ID.
    pub fn pid(&self) -> Option<i32> {
        self.inner.lock().unwrap().pid
    }

    /// True if recovered from crash (no pipe).
    pub fn detached(&self) -> bool {
        self.inner.lock().unwrap().detached
    }

    /// Snapshot of the current rolling output buffer.
    pub fn output_buffer(&self) -> String {
        self.inner.lock().unwrap().output_buffer.clone()
    }
}

fn append_buffer(inner: &mut SessionInner, chunk: &str) {
    inner.output_buffer.push_str(chunk);
    if inner.output_buffer.chars().count() > inner.max_output_chars {
        // Keep the last max_output_chars characters (char-aware to match
        // Python string slicing semantics on the rolling window).
        let total = inner.output_buffer.chars().count();
        let skip = total - inner.max_output_chars;
        inner.output_buffer = inner.output_buffer.chars().skip(skip).collect();
    }
}

/// Tail the last `n` characters of `s` (char-aware), like Python `s[-n:]`.
fn char_tail(s: &str, n: usize) -> String {
    let total = s.chars().count();
    if total <= n {
        return s.to_string();
    }
    s.chars().skip(total - n).collect()
}

// ---------------------------------------------------------------------------
// Completion / notification queue
// ---------------------------------------------------------------------------

/// A unified queue for all background-process events. Completion notifications
/// (`notify_on_complete`) and watch-pattern matches both land here, tagged by
/// the `"type"` field — matching the Python `completion_queue` payloads.
#[derive(Default)]
pub struct CompletionQueue {
    items: Mutex<std::collections::VecDeque<Value>>,
}

impl CompletionQueue {
    fn new() -> Self {
        CompletionQueue::default()
    }

    /// Enqueue an event.
    pub fn put(&self, item: Value) {
        self.items.lock().unwrap().push_back(item);
    }

    /// Pop the next event, FIFO.
    pub fn get_nowait(&self) -> Option<Value> {
        self.items.lock().unwrap().pop_front()
    }

    /// Number of queued events.
    pub fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain all queued events.
    pub fn drain(&self) -> Vec<Value> {
        let mut guard = self.items.lock().unwrap();
        guard.drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// Checkpoint entry
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default)]
struct CheckpointEntry {
    session_id: String,
    command: String,
    pid: Option<i32>,
    #[serde(default = "default_host")]
    pid_scope: String,
    cwd: Option<String>,
    #[serde(default)]
    started_at: f64,
    #[serde(default)]
    task_id: String,
    #[serde(default)]
    session_key: String,
    #[serde(default)]
    watcher_platform: String,
    #[serde(default)]
    watcher_chat_id: String,
    #[serde(default)]
    watcher_user_id: String,
    #[serde(default)]
    watcher_user_name: String,
    #[serde(default)]
    watcher_thread_id: String,
    #[serde(default)]
    watcher_interval: i64,
    #[serde(default)]
    notify_on_complete: bool,
    #[serde(default)]
    watch_patterns: Vec<String>,
}

fn default_host() -> String {
    "host".to_string()
}

fn checkpoint_path() -> std::path::PathBuf {
    hermes_core::mod_hermes_constants::get_hermes_home().join("processes.json")
}

// ---------------------------------------------------------------------------
// ProcessRegistry
// ---------------------------------------------------------------------------

#[derive(Default)]
struct GlobalWatchState {
    window_start: f64,
    window_hits: u32,
    tripped_until: f64,
    suppressed_during_trip: u32,
}

struct RegistryInner {
    running: BTreeMap<String, Arc<ProcessSession>>,
    finished: BTreeMap<String, Arc<ProcessSession>>,
    completion_consumed: HashSet<String>,
    pending_watchers: Vec<Value>,
}

/// In-memory registry of running and finished background processes.
///
/// Thread-safe. Accessed from executor threads, reader/poller threads, and the
/// gateway loop.
pub struct ProcessRegistry {
    inner: Mutex<RegistryInner>,
    /// Notification queue drained by CLI/gateway after each agent turn.
    pub completion_queue: Arc<CompletionQueue>,
    global_watch: Mutex<GlobalWatchState>,
    /// Condvar to wake waiters (used by `wait()`).
    wait_cv: Condvar,
    wait_mutex: Mutex<()>,
}

impl ProcessRegistry {
    /// Construct a fresh, empty registry.
    pub fn new() -> Self {
        ProcessRegistry {
            inner: Mutex::new(RegistryInner {
                running: BTreeMap::new(),
                finished: BTreeMap::new(),
                completion_consumed: HashSet::new(),
                pending_watchers: Vec::new(),
            }),
            completion_queue: Arc::new(CompletionQueue::new()),
            global_watch: Mutex::new(GlobalWatchState::default()),
            wait_cv: Condvar::new(),
            wait_mutex: Mutex::new(()),
        }
    }

    /// Snapshot of `pending_watchers` (gateway reads these after recovery).
    pub fn pending_watchers(&self) -> Vec<Value> {
        self.inner.lock().unwrap().pending_watchers.clone()
    }

    fn notify_waiters(&self) {
        let _g = self.wait_mutex.lock().unwrap();
        self.wait_cv.notify_all();
    }

    // ----- Shell noise -----

    fn clean_shell_noise(text: &str) -> String {
        let mut lines: Vec<&str> = text.split('\n').collect();
        while !lines.is_empty()
            && SHELL_NOISE_SUBSTRINGS
                .iter()
                .any(|noise| lines[0].contains(noise))
        {
            lines.remove(0);
        }
        lines.join("\n")
    }

    // ----- Watch patterns -----

    fn check_watch_patterns(&self, session: &Arc<ProcessSession>, new_text: &str) {
        {
            let inner = session.inner.lock().unwrap();
            if inner.watch_patterns.is_empty() || inner.watch_disabled {
                return;
            }
            // Suppress-after-exit: late chunks after exit are post-exit noise.
            if inner.exited {
                return;
            }
        }

        // Scan new text line-by-line for pattern matches.
        let patterns: Vec<String> = session.inner.lock().unwrap().watch_patterns.clone();
        let mut matched_lines: Vec<String> = Vec::new();
        let mut matched_pattern: Option<String> = None;
        for line in new_text.split('\n') {
            // splitlines() in Python also splits on \r etc., but in practice
            // the reader feeds \n-delimited chunks; \n split is faithful here.
            for pat in &patterns {
                if line.contains(pat.as_str()) {
                    matched_lines.push(line.trim_end().to_string());
                    if matched_pattern.is_none() {
                        matched_pattern = Some(pat.clone());
                    }
                    break;
                }
            }
        }

        if matched_lines.is_empty() {
            return;
        }

        let now = now_unix();
        let mut should_disable = false;
        let return_early;
        let suppressed_for_emit;
        {
            let mut inner = session.inner.lock().unwrap();
            if inner.watch_cooldown_until != 0.0 && now < inner.watch_cooldown_until {
                // Case 1: inside cooldown — count one strike per window, drop.
                inner.watch_suppressed += matched_lines.len() as u64;
                if !inner.watch_strike_candidate {
                    inner.watch_strike_candidate = true;
                    inner.watch_consecutive_strikes += 1;
                    if inner.watch_consecutive_strikes >= WATCH_STRIKE_LIMIT {
                        inner.watch_disabled = true;
                        inner.notify_on_complete = true;
                        should_disable = true;
                    }
                }
                return_early = true;
                suppressed_for_emit = 0;
            } else {
                // Case 2: cooldown expired.
                if inner.watch_cooldown_until != 0.0 && !inner.watch_strike_candidate {
                    inner.watch_consecutive_strikes = 0;
                }
                inner.watch_strike_candidate = false;

                inner.watch_last_emit_at = now;
                inner.watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS;
                inner.watch_hits += 1;
                suppressed_for_emit = inner.watch_suppressed;
                inner.watch_suppressed = 0;
                return_early = false;
            }
        }

        if return_early {
            if should_disable {
                let (sk, cmd, plat, chat, uid, uname, tid, suppressed) = {
                    let inner = session.inner.lock().unwrap();
                    (
                        session.session_key.clone(),
                        session.command.clone(),
                        inner.watcher_platform.clone(),
                        inner.watcher_chat_id.clone(),
                        inner.watcher_user_id.clone(),
                        inner.watcher_user_name.clone(),
                        inner.watcher_thread_id.clone(),
                        inner.watch_suppressed,
                    )
                };
                self.completion_queue.put(json!({
                    "session_id": session.id,
                    "session_key": sk,
                    "command": cmd,
                    "type": "watch_disabled",
                    "suppressed": suppressed,
                    "platform": plat,
                    "chat_id": chat,
                    "user_id": uid,
                    "user_name": uname,
                    "thread_id": tid,
                    "message": format!(
                        "Watch patterns disabled for process {} — {} consecutive \
                         rate-limit windows triggered (min spacing {}s). Falling \
                         back to notify_on_complete semantics; you'll get exactly \
                         one notification when the process exits.",
                        session.id, WATCH_STRIKE_LIMIT, WATCH_MIN_INTERVAL_SECONDS as i64
                    ),
                }));
            }
            return;
        }

        // Trim matched output.
        let mut output = matched_lines
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if output.chars().count() > 2000 {
            output = format!("{}\n...(truncated)", char_head(&output, 2000));
        }

        // Global circuit breaker.
        if !self.global_watch_admit(now) {
            return;
        }

        let (plat, chat, uid, uname, tid) = {
            let inner = session.inner.lock().unwrap();
            (
                inner.watcher_platform.clone(),
                inner.watcher_chat_id.clone(),
                inner.watcher_user_id.clone(),
                inner.watcher_user_name.clone(),
                inner.watcher_thread_id.clone(),
            )
        };
        self.completion_queue.put(json!({
            "session_id": session.id,
            "session_key": session.session_key,
            "command": session.command,
            "type": "watch_match",
            "pattern": matched_pattern,
            "output": output,
            "suppressed": suppressed_for_emit,
            "platform": plat,
            "chat_id": chat,
            "user_id": uid,
            "user_name": uname,
            "thread_id": tid,
        }));
    }

    fn global_watch_admit(&self, now: f64) -> bool {
        let mut release_msg: Option<Value> = None;
        let mut trip_now = false;
        let admit;
        {
            let mut g = self.global_watch.lock().unwrap();

            // Handle cooldown expiry first.
            if g.tripped_until != 0.0 && now >= g.tripped_until {
                let suppressed = g.suppressed_during_trip;
                g.tripped_until = 0.0;
                g.suppressed_during_trip = 0;
                g.window_start = now;
                g.window_hits = 0;
                if suppressed > 0 {
                    release_msg = Some(json!({
                        "session_id": "",
                        "session_key": "",
                        "command": "",
                        "type": "watch_overflow_released",
                        "suppressed": suppressed,
                        "message": format!(
                            "Watch-pattern notifications resumed. {} match event(s) \
                             were suppressed during the flood.",
                            suppressed
                        ),
                        "platform": "",
                        "chat_id": "",
                        "user_id": "",
                        "user_name": "",
                        "thread_id": "",
                    }));
                }
            }

            // Still in cooldown — drop and count.
            if g.tripped_until != 0.0 && now < g.tripped_until {
                g.suppressed_during_trip += 1;
                admit = false;
            } else {
                // Slide the window.
                if now - g.window_start >= WATCH_GLOBAL_WINDOW_SECONDS {
                    g.window_start = now;
                    g.window_hits = 0;
                }

                if g.window_hits >= WATCH_GLOBAL_MAX_PER_WINDOW {
                    g.tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS;
                    g.suppressed_during_trip += 1;
                    trip_now = true;
                    admit = false;
                } else {
                    g.window_hits += 1;
                    admit = true;
                }
            }
        }

        if let Some(msg) = release_msg {
            self.completion_queue.put(msg);
        }
        if trip_now {
            self.completion_queue.put(json!({
                "session_id": "",
                "session_key": "",
                "command": "",
                "type": "watch_overflow_tripped",
                "message": format!(
                    "Watch-pattern overflow: >{} notifications in {}s across all \
                     processes. Suppressing further watch_match events for {}s.",
                    WATCH_GLOBAL_MAX_PER_WINDOW,
                    WATCH_GLOBAL_WINDOW_SECONDS as i64,
                    WATCH_GLOBAL_COOLDOWN_SECONDS as i64
                ),
                "platform": "",
                "chat_id": "",
                "user_id": "",
                "user_name": "",
                "thread_id": "",
            }));
        }
        admit
    }

    // ----- PID liveness -----

    fn is_host_pid_alive(pid: Option<i32>) -> bool {
        let pid = match pid {
            Some(p) if p != 0 => p,
            _ => return false,
        };
        #[cfg(unix)]
        {
            // SAFETY: kill(pid, 0) is the POSIX existence check; sends no signal.
            let rc = unsafe { libc::kill(pid, 0) };
            if rc == 0 {
                return true;
            }
            matches!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::EPERM)
            )
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            false
        }
    }

    fn refresh_detached_session(&self, session: &Arc<ProcessSession>) -> bool {
        // Returns true if it transitioned to finished here.
        {
            let inner = session.inner.lock().unwrap();
            if inner.exited || !inner.detached || inner.pid_scope != "host" {
                return false;
            }
            if Self::is_host_pid_alive(inner.pid) {
                return false;
            }
        }
        let transitioned = {
            let mut inner = session.inner.lock().unwrap();
            if inner.exited {
                false
            } else {
                inner.exited = true;
                inner.exit_code = None;
                true
            }
        };
        if transitioned {
            self.move_to_finished(session);
        }
        true
    }

    fn terminate_host_pid(pid: i32) {
        #[cfg(unix)]
        {
            // SAFETY: best-effort group/proc termination of a host-visible PID.
            unsafe {
                let pgid = libc::getpgid(pid);
                if pgid > 0 && libc::kill(-pgid, libc::SIGTERM) == 0 {
                    return;
                }
                libc::kill(pid, libc::SIGTERM);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
        }
    }

    // ----- Spawn -----

    fn env_temp_dir(env: &Arc<dyn EnvExecutor>) -> String {
        if let Some(temp_dir) = env.get_temp_dir() {
            if temp_dir.starts_with('/') {
                let trimmed = temp_dir.trim_end_matches('/');
                return if trimmed.is_empty() {
                    "/".to_string()
                } else {
                    trimmed.to_string()
                };
            }
        }
        "/tmp".to_string()
    }

    /// Spawn a background process locally (TERMINAL_ENV=local).
    ///
    /// `use_pty` is accepted for API parity but always falls back to the
    /// standard pipe-based path (no PTY crate is available).
    pub fn spawn_local(
        self: &Arc<Self>,
        command: &str,
        cwd: Option<&str>,
        task_id: &str,
        session_key: &str,
        env_vars: Option<&BTreeMap<String, String>>,
        _use_pty: bool,
    ) -> Arc<ProcessSession> {
        let resolved_cwd = resolve_safe_cwd(cwd);
        let session = ProcessSession::new(new_session_id(), command.to_string());
        // Fill identity fields (they are immutable so set them on a fresh
        // Arc before sharing — done via Arc::get_mut while uniquely owned).
        let session = {
            let mut s = session;
            {
                let m = Arc::get_mut(&mut s).expect("freshly created session is unique");
                m.task_id = task_id.to_string();
                m.session_key = session_key.to_string();
            }
            s
        };
        {
            let mut inner = session.inner.lock().unwrap();
            inner.cwd = Some(resolved_cwd.clone());
            inner.started_at = now_unix();
        }

        let user_shell = find_shell();
        let mut cmd = std::process::Command::new(&user_shell);
        cmd.arg("-lic")
            .arg(format!("set +m; {command}"))
            .current_dir(&resolved_cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()) // merged below via reader
            .stdin(Stdio::piped());

        // _sanitize_subprocess_env: start from current env + overlay env_vars,
        // force PYTHONUNBUFFERED=1.
        if let Some(vars) = env_vars {
            for (k, v) in vars {
                cmd.env(k, v);
            }
        }
        cmd.env("PYTHONUNBUFFERED", "1");

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: setsid in the child before exec creates a dedicated
            // process group so kill requests can target the whole subtree.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }

        match cmd.spawn() {
            Ok(mut child) => {
                let pid = child.id() as i32;
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let stdin = child.stdin.take();
                {
                    let mut inner = session.inner.lock().unwrap();
                    inner.pid = Some(pid);
                    inner.child = Some(child);
                    inner.stdin = stdin;
                }
                let reg = Arc::clone(self);
                let sess = Arc::clone(&session);
                thread::Builder::new()
                    .name(format!("proc-reader-{}", session.id))
                    .spawn(move || reg.reader_loop(sess, stdout, stderr))
                    .ok();
            }
            Err(e) => {
                let mut inner = session.inner.lock().unwrap();
                inner.exited = true;
                inner.exit_code = Some(-1);
                inner.output_buffer = format!("Failed to start: {e}");
            }
        }

        {
            let mut inner = self.inner.lock().unwrap();
            self.prune_if_needed(&mut inner);
            inner.running.insert(session.id.clone(), Arc::clone(&session));
        }
        self.write_checkpoint();
        session
    }

    /// Spawn a background process through a non-local environment backend.
    pub fn spawn_via_env(
        self: &Arc<Self>,
        env: Arc<dyn EnvExecutor>,
        command: &str,
        cwd: Option<&str>,
        task_id: &str,
        session_key: &str,
        timeout: u64,
    ) -> Arc<ProcessSession> {
        let mut session = ProcessSession::new(new_session_id(), command.to_string());
        {
            let m = Arc::get_mut(&mut session).expect("freshly created session is unique");
            m.task_id = task_id.to_string();
            m.session_key = session_key.to_string();
            m.env_ref = Some(Arc::clone(&env));
        }
        {
            let mut inner = session.inner.lock().unwrap();
            inner.cwd = cwd.map(str::to_string);
            inner.started_at = now_unix();
            inner.pid_scope = "sandbox".to_string();
        }

        let temp_dir = Self::env_temp_dir(&env);
        let log_path = format!("{temp_dir}/hermes_bg_{}.log", session.id);
        let pid_path = format!("{temp_dir}/hermes_bg_{}.pid", session.id);
        let exit_path = format!("{temp_dir}/hermes_bg_{}.exit", session.id);
        let bg_command = format!(
            "mkdir -p {q_temp} && ( nohup bash -lc {q_cmd} > {q_log} 2>&1; \
             rc=$?; printf '%s\\n' \"$rc\" > {q_exit} ) & echo $! > {q_pid} && cat {q_pid}",
            q_temp = shell_quote(&temp_dir),
            q_cmd = shell_quote(command),
            q_log = shell_quote(&log_path),
            q_exit = shell_quote(&exit_path),
            q_pid = shell_quote(&pid_path),
        );

        let mut failed = false;
        match env.execute(&bg_command, timeout) {
            Ok(output) => {
                let output = output.trim();
                for line in output.lines() {
                    let line = line.trim();
                    if !line.is_empty() && line.chars().all(|c| c.is_ascii_digit()) {
                        if let Ok(p) = line.parse::<i32>() {
                            session.inner.lock().unwrap().pid = Some(p);
                        }
                        break;
                    }
                }
            }
            Err(e) => {
                let mut inner = session.inner.lock().unwrap();
                inner.exited = true;
                inner.exit_code = Some(-1);
                inner.output_buffer = format!("Failed to start: {e}");
                failed = true;
            }
        }

        if !failed {
            let reg = Arc::clone(self);
            let sess = Arc::clone(&session);
            let env2 = Arc::clone(&env);
            thread::Builder::new()
                .name(format!("proc-poller-{}", session.id))
                .spawn(move || reg.env_poller_loop(sess, env2, log_path, pid_path, exit_path))
                .ok();
        }

        {
            let mut inner = self.inner.lock().unwrap();
            self.prune_if_needed(&mut inner);
            inner.running.insert(session.id.clone(), Arc::clone(&session));
        }
        self.write_checkpoint();
        session
    }

    // ----- Reader / Poller threads -----

    fn reader_loop(
        self: Arc<Self>,
        session: Arc<ProcessSession>,
        stdout: Option<std::process::ChildStdout>,
        stderr: Option<std::process::ChildStderr>,
    ) {
        // Merge stderr into stdout-equivalent ordering by reading stdout first
        // then draining stderr. Python merges via stderr=STDOUT; we approximate
        // by reading stdout (primary) on this thread and spawning a stderr
        // drain thread that appends to the same buffer.
        if let Some(mut err) = stderr {
            let sess = Arc::clone(&session);
            let reg = Arc::clone(&self);
            thread::Builder::new()
                .name(format!("proc-reader-err-{}", session.id))
                .spawn(move || {
                    let mut buf = [0u8; 4096];
                    loop {
                        match err.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                let chunk = String::from_utf8_lossy(&buf[..n]).into_owned();
                                {
                                    let mut inner = sess.inner.lock().unwrap();
                                    append_buffer(&mut inner, &chunk);
                                }
                                reg.check_watch_patterns(&sess, &chunk);
                            }
                        }
                    }
                })
                .ok();
        }

        let mut first_chunk = true;
        if let Some(mut out) = stdout {
            let mut buf = [0u8; 4096];
            loop {
                match out.read(&mut buf) {
                    Ok(0) => break,
                    Err(_) => break,
                    Ok(n) => {
                        let mut chunk = String::from_utf8_lossy(&buf[..n]).into_owned();
                        if first_chunk {
                            chunk = Self::clean_shell_noise(&chunk);
                            first_chunk = false;
                        }
                        {
                            let mut inner = session.inner.lock().unwrap();
                            append_buffer(&mut inner, &chunk);
                        }
                        self.check_watch_patterns(&session, &chunk);
                    }
                }
            }
        }

        // Reap the child.
        let rc = {
            let mut inner = session.inner.lock().unwrap();
            let mut code = None;
            if let Some(child) = inner.child.as_mut() {
                match child.wait() {
                    Ok(status) => code = exit_code_of(&status),
                    Err(_) => code = None,
                }
            }
            inner.exited = true;
            inner.exit_code = code;
            code
        };
        let _ = rc;
        self.move_to_finished(&session);
    }

    fn env_poller_loop(
        self: Arc<Self>,
        session: Arc<ProcessSession>,
        env: Arc<dyn EnvExecutor>,
        log_path: String,
        pid_path: String,
        exit_path: String,
    ) {
        let q_log = shell_quote(&log_path);
        let q_pid = shell_quote(&pid_path);
        let q_exit = shell_quote(&exit_path);
        let mut prev_output_len: usize = 0;

        loop {
            if session.inner.lock().unwrap().exited {
                return;
            }
            thread::sleep(Duration::from_secs(2));

            // Read new output from the log file.
            match env.execute(&format!("cat {q_log} 2>/dev/null"), 10) {
                Ok(new_output) => {
                    if !new_output.is_empty() {
                        let chars: Vec<char> = new_output.chars().collect();
                        let delta: String = if chars.len() > prev_output_len {
                            chars[prev_output_len..].iter().collect()
                        } else {
                            String::new()
                        };
                        prev_output_len = chars.len();
                        {
                            let mut inner = session.inner.lock().unwrap();
                            inner.output_buffer = new_output.clone();
                            if inner.output_buffer.chars().count() > inner.max_output_chars {
                                inner.output_buffer =
                                    char_tail(&inner.output_buffer, inner.max_output_chars);
                            }
                        }
                        if !delta.is_empty() {
                            self.check_watch_patterns(&session, &delta);
                        }
                    }

                    // Check if process is still running.
                    let check = env.execute(
                        &format!("kill -0 \"$(cat {q_pid} 2>/dev/null)\" 2>/dev/null; echo $?"),
                        5,
                    );
                    match check {
                        Ok(check_output) => {
                            let check_output = check_output.trim();
                            let last = check_output
                                .lines()
                                .last()
                                .map(str::trim)
                                .unwrap_or("");
                            if !check_output.is_empty() && last != "0" {
                                // Process exited — read exit code.
                                let exit_str = env
                                    .execute(&format!("cat {q_exit} 2>/dev/null"), 5)
                                    .unwrap_or_default();
                                let code = exit_str
                                    .trim()
                                    .lines()
                                    .last()
                                    .and_then(|s| s.trim().parse::<i32>().ok())
                                    .unwrap_or(-1);
                                {
                                    let mut inner = session.inner.lock().unwrap();
                                    inner.exit_code = Some(code);
                                    inner.exited = true;
                                }
                                self.move_to_finished(&session);
                                return;
                            }
                        }
                        Err(_) => {
                            self.finish_with_error(&session);
                            return;
                        }
                    }
                }
                Err(_) => {
                    self.finish_with_error(&session);
                    return;
                }
            }
        }
    }

    fn finish_with_error(&self, session: &Arc<ProcessSession>) {
        {
            let mut inner = session.inner.lock().unwrap();
            inner.exited = true;
            inner.exit_code = Some(-1);
        }
        self.move_to_finished(session);
    }

    fn move_to_finished(&self, session: &Arc<ProcessSession>) {
        let was_running;
        {
            let mut inner = self.inner.lock().unwrap();
            was_running = inner.running.remove(&session.id).is_some();
            inner
                .finished
                .insert(session.id.clone(), Arc::clone(session));
        }
        self.write_checkpoint();
        self.notify_waiters();

        if was_running {
            let (notify, exit_code, buffer) = {
                let inner = session.inner.lock().unwrap();
                (
                    inner.notify_on_complete,
                    inner.exit_code,
                    inner.output_buffer.clone(),
                )
            };
            if notify {
                let output_tail = if buffer.is_empty() {
                    String::new()
                } else {
                    strip_ansi(&char_tail(&buffer, 2000)).into_owned()
                };
                self.completion_queue.put(json!({
                    "type": "completion",
                    "session_id": session.id,
                    "command": session.command,
                    "exit_code": exit_code,
                    "output": output_tail,
                }));
            }
        }
    }

    // ----- Query methods -----

    /// Check if a completion was already consumed via wait/poll/log.
    pub fn is_completion_consumed(&self, session_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .completion_consumed
            .contains(session_id)
    }

    fn mark_consumed(&self, session_id: &str) {
        self.inner
            .lock()
            .unwrap()
            .completion_consumed
            .insert(session_id.to_string());
    }

    /// Get a session by ID (running or finished), refreshing detached state.
    pub fn get(&self, session_id: &str) -> Option<Arc<ProcessSession>> {
        let session = {
            let inner = self.inner.lock().unwrap();
            inner
                .running
                .get(session_id)
                .cloned()
                .or_else(|| inner.finished.get(session_id).cloned())
        };
        if let Some(s) = &session {
            self.refresh_detached_session(s);
        }
        session
    }

    fn reconcile_local_exit(&self, session: &Arc<ProcessSession>) {
        let rc = {
            let mut inner = session.inner.lock().unwrap();
            if inner.exited {
                return;
            }
            let child = match inner.child.as_mut() {
                Some(c) => c,
                None => return,
            };
            match child.try_wait() {
                Ok(Some(status)) => exit_code_of(&status),
                Ok(None) => return, // still running — reader block is legitimate
                Err(_) => return,
            }
        };

        // Direct child exited. Best-effort drain is handled by the reader
        // thread; flip the session to exited so poll/wait don't hang on an
        // orphaned pipe (issue #17327).
        {
            let mut inner = session.inner.lock().unwrap();
            if inner.exited {
                return;
            }
            inner.exited = true;
            inner.exit_code = rc;
        }
        self.move_to_finished(session);
    }

    /// Check status and get new output for a background process.
    pub fn poll(&self, session_id: &str) -> Value {
        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };
        self.reconcile_local_exit(&session);

        let (output_preview, exited, exit_code, pid, started_at, detached) = {
            let inner = session.inner.lock().unwrap();
            let preview = if inner.output_buffer.is_empty() {
                String::new()
            } else {
                strip_ansi(&char_tail(&inner.output_buffer, 1000)).into_owned()
            };
            (
                preview,
                inner.exited,
                inner.exit_code,
                inner.pid,
                inner.started_at,
                inner.detached,
            )
        };

        let mut result = Map::new();
        result.insert("session_id".into(), json!(session.id));
        result.insert("command".into(), json!(session.command));
        result.insert(
            "status".into(),
            json!(if exited { "exited" } else { "running" }),
        );
        result.insert("pid".into(), json!(pid));
        result.insert(
            "uptime_seconds".into(),
            json!((now_unix() - started_at) as i64),
        );
        result.insert("output_preview".into(), json!(output_preview));
        if exited {
            result.insert("exit_code".into(), json!(exit_code));
            self.mark_consumed(session_id);
        }
        if detached {
            result.insert("detached".into(), json!(true));
            result.insert(
                "note".into(),
                json!("Process recovered after restart -- output history unavailable"),
            );
        }
        Value::Object(result)
    }

    /// Read the full output log with optional pagination by lines.
    pub fn read_log(&self, session_id: &str, offset: usize, limit: i64) -> Value {
        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };

        let (full_output, exited) = {
            let inner = session.inner.lock().unwrap();
            (strip_ansi(&inner.output_buffer).into_owned(), inner.exited)
        };

        // Python splitlines() drops a trailing empty element; emulate that.
        let lines = splitlines(&full_output);
        let total_lines = lines.len();

        let selected: Vec<&str> = if offset == 0 && limit > 0 {
            let take = limit as usize;
            if total_lines > take {
                lines[total_lines - take..].to_vec()
            } else {
                lines.clone()
            }
        } else {
            let start = offset.min(total_lines);
            let end = if limit > 0 {
                (offset + limit as usize).min(total_lines)
            } else {
                start
            };
            lines[start..end].to_vec()
        };

        let mut result = Map::new();
        result.insert("session_id".into(), json!(session.id));
        result.insert(
            "status".into(),
            json!(if exited { "exited" } else { "running" }),
        );
        result.insert("output".into(), json!(selected.join("\n")));
        result.insert("total_lines".into(), json!(total_lines));
        result.insert(
            "showing".into(),
            json!(format!("{} lines", selected.len())),
        );
        if exited {
            self.mark_consumed(session_id);
        }
        Value::Object(result)
    }

    /// Block until a process exits, times out, or is interrupted.
    pub fn wait(&self, session_id: &str, timeout: Option<i64>) -> Value {
        let default_timeout = std::env::var("TERMINAL_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(180);
        let max_timeout = default_timeout;
        let requested_timeout = timeout;
        let mut timeout_note: Option<String> = None;

        let effective_timeout = match requested_timeout {
            Some(req) if req > max_timeout => {
                timeout_note = Some(format!(
                    "Requested wait of {req}s was clamped to configured limit of {max_timeout}s"
                ));
                max_timeout
            }
            Some(req) if req > 0 => req,
            _ => max_timeout,
        };

        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };

        let deadline = Instant::now() + Duration::from_secs(effective_timeout.max(0) as u64);

        while Instant::now() < deadline {
            self.refresh_detached_session(&session);
            self.reconcile_local_exit(&session);

            let (exited, exit_code, buffer) = {
                let inner = session.inner.lock().unwrap();
                (inner.exited, inner.exit_code, inner.output_buffer.clone())
            };

            if exited {
                self.mark_consumed(session_id);
                let mut result = Map::new();
                result.insert("status".into(), json!("exited"));
                result.insert("exit_code".into(), json!(exit_code));
                result.insert(
                    "output".into(),
                    json!(strip_ansi(&char_tail(&buffer, 2000)).into_owned()),
                );
                if let Some(note) = &timeout_note {
                    result.insert("timeout_note".into(), json!(note));
                }
                return Value::Object(result);
            }

            if hermes_core::tool_interrupt::is_interrupted() {
                let mut result = Map::new();
                result.insert("status".into(), json!("interrupted"));
                result.insert(
                    "output".into(),
                    json!(strip_ansi(&char_tail(&buffer, 1000)).into_owned()),
                );
                result.insert(
                    "note".into(),
                    json!("User sent a new message -- wait interrupted"),
                );
                if let Some(note) = &timeout_note {
                    result.insert("timeout_note".into(), json!(note));
                }
                return Value::Object(result);
            }

            thread::sleep(Duration::from_secs(1));
        }

        let buffer = session.inner.lock().unwrap().output_buffer.clone();
        let mut result = Map::new();
        result.insert("status".into(), json!("timeout"));
        result.insert(
            "output".into(),
            json!(strip_ansi(&char_tail(&buffer, 1000)).into_owned()),
        );
        result.insert(
            "timeout_note".into(),
            json!(timeout_note.unwrap_or_else(|| format!(
                "Waited {effective_timeout}s, process still running"
            ))),
        );
        Value::Object(result)
    }

    /// Kill a background process.
    pub fn kill_process(&self, session_id: &str) -> Value {
        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };

        if session.exited() {
            return json!({
                "status": "already_exited",
                "exit_code": session.exit_code(),
            });
        }

        // Determine kill strategy.
        let (has_child, has_env_pid, detached_host_pid, child_pid) = {
            let inner = session.inner.lock().unwrap();
            (
                inner.child.is_some(),
                session.env_ref.is_some() && inner.pid.is_some(),
                inner.detached && inner.pid_scope == "host" && inner.pid.is_some(),
                inner.pid,
            )
        };

        if has_child {
            // Local process — kill the process group.
            #[cfg(unix)]
            {
                if let Some(pid) = child_pid {
                    // SAFETY: signal the process group created via setsid.
                    let killed = unsafe {
                        let pgid = libc::getpgid(pid);
                        pgid > 0 && libc::kill(-pgid, libc::SIGTERM) == 0
                    };
                    if !killed {
                        let mut inner = session.inner.lock().unwrap();
                        if let Some(child) = inner.child.as_mut() {
                            let _ = child.kill();
                        }
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let mut inner = session.inner.lock().unwrap();
                if let Some(child) = inner.child.as_mut() {
                    let _ = child.kill();
                }
            }
        } else if has_env_pid {
            if let (Some(env), Some(pid)) = (&session.env_ref, child_pid) {
                let _ = env.execute(&format!("kill {pid} 2>/dev/null"), 5);
            }
        } else if detached_host_pid {
            if !Self::is_host_pid_alive(child_pid) {
                {
                    let mut inner = session.inner.lock().unwrap();
                    inner.exited = true;
                    inner.exit_code = None;
                }
                self.move_to_finished(&session);
                return json!({
                    "status": "already_exited",
                    "exit_code": Value::Null,
                });
            }
            if let Some(pid) = child_pid {
                Self::terminate_host_pid(pid);
            }
        } else {
            return json!({
                "status": "error",
                "error": "Recovered process cannot be killed after restart because \
                          its original runtime handle is no longer available",
            });
        }

        {
            let mut inner = session.inner.lock().unwrap();
            inner.exited = true;
            inner.exit_code = Some(-15); // SIGTERM
        }
        self.move_to_finished(&session);
        self.write_checkpoint();
        json!({"status": "killed", "session_id": session.id})
    }

    /// Send raw data to a running process's stdin (no newline appended).
    pub fn write_stdin(&self, session_id: &str, data: &str) -> Value {
        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };
        if session.exited() {
            return json!({"status": "already_exited", "error": "Process has already finished"});
        }

        let mut inner = session.inner.lock().unwrap();
        match inner.stdin.as_mut() {
            Some(stdin) => match stdin
                .write_all(data.as_bytes())
                .and_then(|_| stdin.flush())
            {
                Ok(_) => json!({"status": "ok", "bytes_written": data.len()}),
                Err(e) => json!({"status": "error", "error": e.to_string()}),
            },
            None => json!({
                "status": "error",
                "error": "Process stdin not available (non-local backend or stdin closed)",
            }),
        }
    }

    /// Send data + newline to a running process's stdin (like pressing Enter).
    pub fn submit_stdin(&self, session_id: &str, data: &str) -> Value {
        self.write_stdin(session_id, &format!("{data}\n"))
    }

    /// Close a running process's stdin / send EOF without killing the process.
    pub fn close_stdin(&self, session_id: &str) -> Value {
        let session = match self.get(session_id) {
            Some(s) => s,
            None => {
                return json!({
                    "status": "not_found",
                    "error": format!("No process with ID {session_id}"),
                })
            }
        };
        if session.exited() {
            return json!({"status": "already_exited", "error": "Process has already finished"});
        }

        let mut inner = session.inner.lock().unwrap();
        if inner.stdin.is_some() {
            inner.stdin = None; // dropping the handle closes the pipe (EOF)
            json!({"status": "ok", "message": "stdin closed"})
        } else {
            json!({
                "status": "error",
                "error": "Process stdin not available (non-local backend or stdin closed)",
            })
        }
    }

    /// List all running and recently-finished processes.
    pub fn list_sessions(&self, task_id: Option<&str>) -> Vec<Value> {
        let all_sessions: Vec<Arc<ProcessSession>> = {
            let inner = self.inner.lock().unwrap();
            inner
                .running
                .values()
                .cloned()
                .chain(inner.finished.values().cloned())
                .collect()
        };
        for s in &all_sessions {
            self.refresh_detached_session(s);
        }

        let mut result = Vec::new();
        for s in &all_sessions {
            if let Some(tid) = task_id {
                if s.task_id != tid {
                    continue;
                }
            }
            let inner = s.inner.lock().unwrap();
            let mut entry = Map::new();
            entry.insert("session_id".into(), json!(s.id));
            entry.insert("command".into(), json!(char_head(&s.command, 200)));
            entry.insert("cwd".into(), json!(inner.cwd));
            entry.insert("pid".into(), json!(inner.pid));
            entry.insert(
                "started_at".into(),
                json!(format_local_iso(inner.started_at)),
            );
            entry.insert(
                "uptime_seconds".into(),
                json!((now_unix() - inner.started_at) as i64),
            );
            entry.insert(
                "status".into(),
                json!(if inner.exited { "exited" } else { "running" }),
            );
            entry.insert(
                "output_preview".into(),
                json!(if inner.output_buffer.is_empty() {
                    String::new()
                } else {
                    char_tail(&inner.output_buffer, 200)
                }),
            );
            if inner.exited {
                entry.insert("exit_code".into(), json!(inner.exit_code));
            }
            if inner.detached {
                entry.insert("detached".into(), json!(true));
            }
            result.push(Value::Object(entry));
        }
        result
    }

    // ----- Session/task queries (gateway) -----

    /// Whether there are active (running) processes for a task_id.
    pub fn has_active_processes(&self, task_id: &str) -> bool {
        let sessions: Vec<Arc<ProcessSession>> =
            self.inner.lock().unwrap().running.values().cloned().collect();
        for s in &sessions {
            self.refresh_detached_session(s);
        }
        let inner = self.inner.lock().unwrap();
        inner.running.values().any(|s| {
            let i = s.inner.lock().unwrap();
            s.task_id == task_id && !i.exited
        })
    }

    /// Whether there are active processes for a gateway session key.
    pub fn has_active_for_session(&self, session_key: &str) -> bool {
        let sessions: Vec<Arc<ProcessSession>> =
            self.inner.lock().unwrap().running.values().cloned().collect();
        for s in &sessions {
            self.refresh_detached_session(s);
        }
        let inner = self.inner.lock().unwrap();
        inner.running.values().any(|s| {
            let i = s.inner.lock().unwrap();
            s.session_key == session_key && !i.exited
        })
    }

    /// Kill all running processes, optionally filtered by task_id. Returns count killed.
    pub fn kill_all(&self, task_id: Option<&str>) -> usize {
        let targets: Vec<String> = {
            let inner = self.inner.lock().unwrap();
            inner
                .running
                .values()
                .filter(|s| {
                    let i = s.inner.lock().unwrap();
                    (task_id.is_none() || Some(s.task_id.as_str()) == task_id) && !i.exited
                })
                .map(|s| s.id.clone())
                .collect()
        };
        let mut killed = 0;
        for sid in targets {
            let result = self.kill_process(&sid);
            if let Some(status) = result.get("status").and_then(Value::as_str) {
                if status == "killed" || status == "already_exited" {
                    killed += 1;
                }
            }
        }
        killed
    }

    // ----- Cleanup / pruning -----

    fn prune_if_needed(&self, inner: &mut RegistryInner) {
        let now = now_unix();
        let expired: Vec<String> = inner
            .finished
            .iter()
            .filter(|(_, s)| (now - s.inner.lock().unwrap().started_at) > FINISHED_TTL_SECONDS)
            .map(|(sid, _)| sid.clone())
            .collect();
        for sid in expired {
            inner.finished.remove(&sid);
            inner.completion_consumed.remove(&sid);
        }

        let total = inner.running.len() + inner.finished.len();
        if total >= MAX_PROCESSES && !inner.finished.is_empty() {
            let oldest_id = inner
                .finished
                .iter()
                .min_by(|a, b| {
                    let sa = a.1.inner.lock().unwrap().started_at;
                    let sb = b.1.inner.lock().unwrap().started_at;
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(sid, _)| sid.clone());
            if let Some(oldest_id) = oldest_id {
                inner.finished.remove(&oldest_id);
                inner.completion_consumed.remove(&oldest_id);
            }
        }

        let tracked: HashSet<String> = inner
            .running
            .keys()
            .chain(inner.finished.keys())
            .cloned()
            .collect();
        let stale: Vec<String> = inner
            .completion_consumed
            .difference(&tracked)
            .cloned()
            .collect();
        for sid in stale {
            inner.completion_consumed.remove(&sid);
        }
    }

    // ----- Checkpoint (crash recovery) -----

    fn write_checkpoint(&self) {
        let entries: Vec<CheckpointEntry> = {
            let inner = self.inner.lock().unwrap();
            inner
                .running
                .values()
                .filter_map(|s| {
                    let i = s.inner.lock().unwrap();
                    if i.exited {
                        return None;
                    }
                    Some(CheckpointEntry {
                        session_id: s.id.clone(),
                        command: s.command.clone(),
                        pid: i.pid,
                        pid_scope: i.pid_scope.clone(),
                        cwd: i.cwd.clone(),
                        started_at: i.started_at,
                        task_id: s.task_id.clone(),
                        session_key: s.session_key.clone(),
                        watcher_platform: i.watcher_platform.clone(),
                        watcher_chat_id: i.watcher_chat_id.clone(),
                        watcher_user_id: i.watcher_user_id.clone(),
                        watcher_user_name: i.watcher_user_name.clone(),
                        watcher_thread_id: i.watcher_thread_id.clone(),
                        watcher_interval: i.watcher_interval,
                        notify_on_complete: i.notify_on_complete,
                        watch_patterns: i.watch_patterns.clone(),
                    })
                })
                .collect()
        };

        let value = match serde_json::to_value(&entries) {
            Ok(v) => v,
            Err(_) => return,
        };
        let _ = hermes_core::mod_utils::atomic_json_write(&checkpoint_path(), &value, 0);
    }

    /// On gateway startup, probe PIDs from the checkpoint file.
    ///
    /// Returns the number of processes recovered as detached.
    pub fn recover_from_checkpoint(&self) -> usize {
        let path = checkpoint_path();
        if !path.exists() {
            return 0;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return 0,
        };
        let entries: Vec<CheckpointEntry> = match serde_json::from_str(&text) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut recovered = 0;
        for entry in entries {
            let pid = match entry.pid {
                Some(p) if p != 0 => p,
                _ => continue,
            };
            if entry.pid_scope != "host" {
                continue;
            }
            if !Self::is_host_pid_alive(Some(pid)) {
                continue;
            }

            let mut session = ProcessSession::new(entry.session_id.clone(), entry.command.clone());
            {
                let m = Arc::get_mut(&mut session).expect("freshly created session is unique");
                m.task_id = entry.task_id.clone();
                m.session_key = entry.session_key.clone();
            }
            {
                let mut inner = session.inner.lock().unwrap();
                inner.pid = Some(pid);
                inner.pid_scope = entry.pid_scope.clone();
                inner.cwd = entry.cwd.clone();
                inner.started_at = if entry.started_at != 0.0 {
                    entry.started_at
                } else {
                    now_unix()
                };
                inner.detached = true;
                inner.watcher_platform = entry.watcher_platform.clone();
                inner.watcher_chat_id = entry.watcher_chat_id.clone();
                inner.watcher_user_id = entry.watcher_user_id.clone();
                inner.watcher_user_name = entry.watcher_user_name.clone();
                inner.watcher_thread_id = entry.watcher_thread_id.clone();
                inner.watcher_interval = entry.watcher_interval;
                inner.notify_on_complete = entry.notify_on_complete;
                inner.watch_patterns = entry.watch_patterns.clone();
            }

            {
                let mut inner = self.inner.lock().unwrap();
                inner
                    .running
                    .insert(session.id.clone(), Arc::clone(&session));
            }
            recovered += 1;

            if entry.watcher_interval > 0 {
                let mut inner = self.inner.lock().unwrap();
                inner.pending_watchers.push(json!({
                    "session_id": session.id,
                    "check_interval": entry.watcher_interval,
                    "session_key": session.session_key,
                    "platform": entry.watcher_platform,
                    "chat_id": entry.watcher_chat_id,
                    "user_id": entry.watcher_user_id,
                    "user_name": entry.watcher_user_name,
                    "thread_id": entry.watcher_thread_id,
                    "notify_on_complete": entry.notify_on_complete,
                }));
            }
        }

        self.write_checkpoint();
        recovered
    }
}

impl Default for ProcessRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Module-level singleton
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

static PROCESS_REGISTRY: OnceLock<Arc<ProcessRegistry>> = OnceLock::new();

/// The module-level singleton registry (mirrors Python `process_registry`).
pub fn process_registry() -> Arc<ProcessRegistry> {
    PROCESS_REGISTRY
        .get_or_init(|| Arc::new(ProcessRegistry::new()))
        .clone()
}

// ---------------------------------------------------------------------------
// Tool schema + handler
// ---------------------------------------------------------------------------

/// The `process` tool JSON schema (faithful port of `PROCESS_SCHEMA`).
pub fn process_schema() -> Value {
    json!({
        "name": "process",
        "description":
            "Manage background processes started with terminal(background=true). \
             Actions: 'list' (show all), 'poll' (check status + new output), \
             'log' (full output with pagination), 'wait' (block until done or timeout), \
             'kill' (terminate), 'write' (send raw stdin data without newline), \
             'submit' (send data + Enter, for answering prompts), 'close' (close stdin/send EOF).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"],
                    "description": "Action to perform on background processes"
                },
                "session_id": {
                    "type": "string",
                    "description": "Process session ID (from terminal background output). Required for all actions except 'list'."
                },
                "data": {
                    "type": "string",
                    "description": "Text to send to process stdin (for 'write' and 'submit' actions)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to block for 'wait' action. Returns partial output on timeout.",
                    "minimum": 1
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset for 'log' action (default: last 200 lines)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return for 'log' action",
                    "minimum": 1
                }
            },
            "required": ["action"]
        }
    })
}

/// Handler for the `process` tool (faithful port of `_handle_process`).
///
/// `task_id` is the optional kwarg the Python handler reads from `**kw`.
pub fn handle_process(args: &Value, task_id: Option<&str>) -> String {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("");
    // Coerce session_id to string — some models send it as an integer.
    let session_id = match args.get("session_id") {
        Some(Value::Null) | None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    };

    let reg = process_registry();

    if action == "list" {
        return json!({"processes": reg.list_sessions(task_id)}).to_string();
    }

    if matches!(
        action,
        "poll" | "log" | "wait" | "kill" | "write" | "submit" | "close"
    ) {
        if session_id.is_empty() {
            return tool_error(&format!("session_id is required for {action}"));
        }
        return match action {
            "poll" => reg.poll(&session_id).to_string(),
            "log" => {
                let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
                let limit = args.get("limit").and_then(Value::as_i64).unwrap_or(200);
                reg.read_log(&session_id, offset, limit).to_string()
            }
            "wait" => {
                let timeout = args.get("timeout").and_then(Value::as_i64);
                reg.wait(&session_id, timeout).to_string()
            }
            "kill" => reg.kill_process(&session_id).to_string(),
            "write" => {
                let data = arg_str(args, "data");
                reg.write_stdin(&session_id, &data).to_string()
            }
            "submit" => {
                let data = arg_str(args, "data");
                reg.submit_stdin(&session_id, &data).to_string()
            }
            "close" => reg.close_stdin(&session_id).to_string(),
            _ => unreachable!(),
        };
    }

    tool_error(&format!(
        "Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close"
    ))
}

fn arg_str(args: &Value, key: &str) -> String {
    match args.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Build a `{"error": ...}` JSON string (mirrors `tools.registry.tool_error`).
fn tool_error(message: &str) -> String {
    json!({"error": message}).to_string()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a `proc_<12-hex>` session id.
fn new_session_id() -> String {
    // Combine a high-resolution timestamp with a monotonic counter and fold to
    // 12 hex chars, matching the `proc_<hex>` shape `uuid4().hex[:12]` produces.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mixed = (nanos as u64) ^ (counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    format!("proc_{:012x}", mixed & 0xFFFF_FFFF_FFFF)
}

/// Resolve the working directory like `_resolve_safe_cwd(cwd or os.getcwd())`.
fn resolve_safe_cwd(cwd: Option<&str>) -> String {
    let candidate = match cwd {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string()),
    };
    // Mirror the safety check: fall back to cwd if the path doesn't exist.
    if std::path::Path::new(&candidate).is_dir() {
        candidate
    } else {
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or(candidate)
    }
}

/// Resolve the user's login shell (mirror of `_find_shell`).
fn find_shell() -> String {
    if let Ok(shell) = std::env::var("SHELL") {
        if !shell.trim().is_empty() {
            return shell;
        }
    }
    "/bin/bash".to_string()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn char_head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python `str.splitlines()` — splits on `\n`/`\r\n`/`\r` and drops the
/// trailing empty element. Used for line pagination so totals match Python.
fn splitlines(s: &str) -> Vec<&str> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let bytes = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&s[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&s[start..i]);
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    i += 2;
                } else {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// Format `time.strftime("%Y-%m-%dT%H:%M:%S", localtime(ts))`.
fn format_local_iso(ts: f64) -> String {
    use chrono::{Local, TimeZone};
    match Local.timestamp_opt(ts as i64, 0).single() {
        Some(dt) => dt.format("%Y-%m-%dT%H:%M:%S").to_string(),
        None => String::new(),
    }
}

fn exit_code_of(status: &std::process::ExitStatus) -> Option<i32> {
    if let Some(code) = status.code() {
        return Some(code);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            // Python's Popen.returncode is -signal for signal terminations.
            return Some(-sig);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_uptime_short() {
        assert_eq!(format_uptime_short(-5), "0s");
        assert_eq!(format_uptime_short(0), "0s");
        assert_eq!(format_uptime_short(45), "45s");
        assert_eq!(format_uptime_short(90), "1m 30s");
        assert_eq!(format_uptime_short(3661), "1h 1m");
        assert_eq!(format_uptime_short(7325), "2h 2m");
    }

    #[test]
    fn test_clean_shell_noise() {
        let text = "bash: no job control in this shell\nreal output\nmore";
        assert_eq!(
            ProcessRegistry::clean_shell_noise(text),
            "real output\nmore"
        );
        // Leading noise lines (multiple) all stripped.
        let text2 = "cannot set terminal process group\nbash: no job control in this shell\nok";
        assert_eq!(ProcessRegistry::clean_shell_noise(text2), "ok");
        // No noise — unchanged.
        assert_eq!(ProcessRegistry::clean_shell_noise("clean\nlines"), "clean\nlines");
    }

    #[test]
    fn test_session_id_shape() {
        let id = new_session_id();
        assert!(id.starts_with("proc_"));
        assert_eq!(id.len(), 5 + 12);
        assert!(id[5..].chars().all(|c| c.is_ascii_hexdigit()));
        // Two ids should differ.
        assert_ne!(new_session_id(), new_session_id());
    }

    #[test]
    fn test_char_tail_and_head() {
        assert_eq!(char_tail("abcdef", 3), "def");
        assert_eq!(char_tail("ab", 5), "ab");
        assert_eq!(char_head("abcdef", 3), "abc");
        assert_eq!(char_head("ab", 5), "ab");
        // Unicode-aware.
        assert_eq!(char_tail("héllo", 3), "llo");
    }

    #[test]
    fn test_splitlines() {
        assert_eq!(splitlines("a\nb\nc"), vec!["a", "b", "c"]);
        // Trailing newline drops the empty element (Python semantics).
        assert_eq!(splitlines("a\nb\n"), vec!["a", "b"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("a\r\nb"), vec!["a", "b"]);
        assert_eq!(splitlines("a\rb"), vec!["a", "b"]);
    }

    #[test]
    fn test_shell_quote() {
        assert_eq!(shell_quote("simple"), "'simple'");
        assert_eq!(shell_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn test_append_buffer_trims_rolling_window() {
        let mut inner = SessionInner {
            max_output_chars: 5,
            ..Default::default()
        };
        append_buffer(&mut inner, "abc");
        append_buffer(&mut inner, "defgh");
        // "abcdefgh" trimmed to last 5 chars.
        assert_eq!(inner.output_buffer, "defgh");
    }

    #[test]
    fn test_poll_not_found() {
        let reg = Arc::new(ProcessRegistry::new());
        let result = reg.poll("proc_doesnotexist");
        assert_eq!(result["status"], "not_found");
    }

    #[test]
    fn test_wait_not_found() {
        let reg = Arc::new(ProcessRegistry::new());
        let result = reg.wait("proc_nope", Some(1));
        assert_eq!(result["status"], "not_found");
    }

    #[test]
    fn test_kill_not_found() {
        let reg = Arc::new(ProcessRegistry::new());
        let result = reg.kill_process("proc_nope");
        assert_eq!(result["status"], "not_found");
    }

    #[test]
    fn test_handle_process_list_empty() {
        let out = handle_process(&json!({"action": "list"}), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["processes"].is_array());
    }

    #[test]
    fn test_handle_process_requires_session_id() {
        let out = handle_process(&json!({"action": "poll"}), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("session_id is required"));
    }

    #[test]
    fn test_handle_process_unknown_action() {
        let out = handle_process(&json!({"action": "frobnicate"}), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("Unknown process action"));
    }

    #[test]
    fn test_handle_process_session_id_coercion() {
        // Integer session_id is coerced to string then looked up (not found).
        let out = handle_process(&json!({"action": "poll", "session_id": 12345}), None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["status"], "not_found");
        assert!(v["error"].as_str().unwrap().contains("12345"));
    }

    #[test]
    fn test_process_schema_shape() {
        let schema = process_schema();
        assert_eq!(schema["name"], "process");
        let enum_vals = schema["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(enum_vals.len(), 8);
        assert!(enum_vals.contains(&json!("kill")));
    }

    #[test]
    fn test_global_watch_admit_caps() {
        let reg = Arc::new(ProcessRegistry::new());
        let now = 1000.0;
        // First WATCH_GLOBAL_MAX_PER_WINDOW are admitted.
        for _ in 0..WATCH_GLOBAL_MAX_PER_WINDOW {
            assert!(reg.global_watch_admit(now));
        }
        // The next one trips the breaker (dropped).
        assert!(!reg.global_watch_admit(now));
        // Still tripped within cooldown.
        assert!(!reg.global_watch_admit(now + 1.0));
    }

    #[test]
    fn test_recover_no_checkpoint() {
        // Point HERMES_HOME at an empty temp dir.
        let dir = std::env::temp_dir().join(format!("hermes_proc_test_{}", new_session_id()));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: edition 2024 requires unsafe for env mutation in tests.
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        let reg = Arc::new(ProcessRegistry::new());
        assert_eq!(reg.recover_from_checkpoint(), 0);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
