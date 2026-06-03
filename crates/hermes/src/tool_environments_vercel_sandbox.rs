//! Vercel Sandbox execution environment.
//!
//! Native Rust port of `tools/environments/vercel_sandbox.py`.
//!
//! The Python original drives the Vercel Python SDK (`vercel.sandbox`) to run
//! commands in cloud sandboxes through Hermes' shared `BaseEnvironment` shell
//! contract. When persistence is enabled, task-scoped snapshot metadata is
//! stored under `HERMES_HOME/vercel_sandbox_snapshots.json` and new sandboxes
//! are restored from those snapshots on later task reuse.
//!
//! Faithful behaviours preserved:
//!   * Transient-error classification and bounded retry with linear backoff
//!     (`_is_transient_vercel_error` / `_retry_vercel_call`).
//!   * Snapshot store load/store/delete keyed by `task_id`
//!     (`_get_snapshot_id` / `_store_snapshot` / `_delete_snapshot`).
//!   * `disk` validation (only `0` or the default container disk allowed) and
//!     resource sizing (`vcpus = floor(cpu)`, `memory_mb = memory`), and the
//!     `max(timeout, MIN_SANDBOX_TIMEOUT)` sandbox-timeout floor.
//!   * Terminal-state detection (`ABORTED` / `FAILED` / `STOPPED`).
//!   * `cwd` rewrite when requested cwd is `~`, empty, or the default Vercel
//!     cwd; remote `$HOME` detection.
//!   * Result text/returncode extraction (`_extract_result_output` /
//!     `_extract_result_returncode`) and snapshot-id extraction across the
//!     `snapshot_id` / `snapshotId` / `id` key variants.
//!   * File-sync shell-string builders (rm -f / tar) matching the `file_sync`
//!     helpers, including a PID-suffixed remote temp tar path.
//!
//! The Vercel SDK surface is abstracted behind the [`VercelSdk`] trait so the
//! environment logic is testable without a live API.

use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Constants (mirror the module-level Python constants)
// ---------------------------------------------------------------------------

/// Default working directory inside a Vercel sandbox.
pub const DEFAULT_VERCEL_CWD: &str = "/vercel/sandbox";

/// Default container disk size in MiB (the only configurable value Vercel
/// accepts besides `0`).
pub const DEFAULT_CONTAINER_DISK_MB: i64 = 51200;

const CREATE_RETRY_ATTEMPTS: u32 = 3;
const WRITE_RETRY_ATTEMPTS: u32 = 3;

/// HTTP status codes treated as transient (retryable).
pub const TRANSIENT_STATUS_CODES: &[u16] = &[408, 425, 429, 500, 502, 503, 504];

/// Linear backoff step: 100ms multiplied by the attempt number.
const RETRY_BACKOFF_STEP: Duration = Duration::from_millis(100);

/// Minimum sandbox lifetime (5 minutes).
pub const MIN_SANDBOX_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Floor for the "wait for running" timeout.
pub const MIN_RUNNING_WAIT: Duration = Duration::from_secs(1);

/// Default "wait for running" timeout.
pub const RUNNING_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll interval while waiting for the sandbox to reach `RUNNING`.
pub const RUNNING_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Blocking-stop timeout.
pub const STOP_TIMEOUT: Duration = Duration::from_secs(15);

/// Poll interval while stopping.
pub const STOP_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Snapshot-store file name under `HERMES_HOME`.
pub const SNAPSHOT_STORE_NAME: &str = "vercel_sandbox_snapshots.json";

// ---------------------------------------------------------------------------
// shell quoting + file_sync shell-string helpers (mirror file_sync.py)
// ---------------------------------------------------------------------------

/// Shell-quote a string the way Python's `shlex.quote` does: if it's safe
/// (matches `[A-Za-z0-9_@%+=:,./-]+` and non-empty), return as-is; otherwise
/// wrap in single quotes, escaping embedded single quotes as `'"'"'`.
///
/// Inlined faithful copy of `hermes_core::tool_environments_base::shlex_quote`
/// (the source module is not exported across crate boundaries).
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

/// Build a shell `rm -f` command for a batch of remote paths.
///
/// Mirrors `tools.environments.file_sync.quoted_rm_command`.
pub fn quoted_rm_command(remote_paths: &[String]) -> String {
    let mut s = String::from("rm -f ");
    s.push_str(
        &remote_paths
            .iter()
            .map(|p| shlex_quote(p))
            .collect::<Vec<_>>()
            .join(" "),
    );
    s
}

// ---------------------------------------------------------------------------
// Sandbox status (mirrors vercel.sandbox.SandboxStatus)
// ---------------------------------------------------------------------------

/// Lifecycle state of a Vercel sandbox. The string values mirror the SDK's
/// enum member casing as observed over the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SandboxStatus {
    Pending,
    Running,
    Stopping,
    Stopped,
    Failed,
    Aborted,
    Unknown,
}

impl SandboxStatus {
    /// Parse an API status string (case-insensitive).
    pub fn from_api(s: &str) -> SandboxStatus {
        match s.to_ascii_lowercase().as_str() {
            "pending" => SandboxStatus::Pending,
            "running" => SandboxStatus::Running,
            "stopping" => SandboxStatus::Stopping,
            "stopped" => SandboxStatus::Stopped,
            "failed" => SandboxStatus::Failed,
            "aborted" => SandboxStatus::Aborted,
            _ => SandboxStatus::Unknown,
        }
    }

    /// Whether this status is one of the terminal states the Python code treats
    /// as unrecoverable: `ABORTED`, `FAILED`, `STOPPED`. Mirrors
    /// `_terminal_sandbox_states`.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            SandboxStatus::Aborted | SandboxStatus::Failed | SandboxStatus::Stopped
        )
    }
}

// ---------------------------------------------------------------------------
// Resources / create params (mirror _SandboxCreateParams + Resources)
// ---------------------------------------------------------------------------

/// Compute resource request mirroring `vercel.sandbox.Resources`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resources {
    pub vcpus: Option<i64>,
    pub memory: Option<i64>,
}

/// Parameters for creating a sandbox. Mirrors `_SandboxCreateParams`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCreateParams {
    /// Sandbox lifetime.
    pub timeout: Duration,
    /// Optional runtime identifier (e.g. `node22`).
    pub runtime: Option<String>,
    /// Optional resource request; `None` if neither vcpus nor memory is set.
    pub resources: Option<Resources>,
}

/// Validate `disk` and build [`SandboxCreateParams`] from raw CLI-style inputs.
///
/// Faithful port of `_build_create_params`:
///   * `disk` must be `0` or [`DEFAULT_CONTAINER_DISK_MB`]; otherwise an error.
///   * sandbox timeout = `max(timeout_secs.max(0), MIN_SANDBOX_TIMEOUT)`.
///   * `vcpus = floor(cpu)` when `cpu > 0`, else `None`.
///   * `memory_mb = memory` when `memory > 0`, else `None`.
///   * `resources` is `Some` only when at least one of the two is set.
pub fn build_create_params(
    runtime: Option<String>,
    timeout_secs: i64,
    cpu: f64,
    memory: i64,
    disk: i64,
) -> Result<SandboxCreateParams, String> {
    if disk != 0 && disk != DEFAULT_CONTAINER_DISK_MB {
        return Err(
            "Vercel Sandbox does not support configurable container_disk. \
             Use the default shared setting."
                .to_string(),
        );
    }

    let base_timeout = Duration::from_secs(timeout_secs.max(0) as u64);
    let sandbox_timeout = base_timeout.max(MIN_SANDBOX_TIMEOUT);

    let vcpus = if cpu > 0.0 {
        Some(cpu.floor() as i64)
    } else {
        None
    };
    let memory_mb = if memory > 0 { Some(memory) } else { None };
    let resources = if vcpus.is_some() || memory_mb.is_some() {
        Some(Resources {
            vcpus,
            memory: memory_mb,
        })
    } else {
        None
    };

    Ok(SandboxCreateParams {
        timeout: sandbox_timeout,
        runtime,
        resources,
    })
}

// ---------------------------------------------------------------------------
// Result extraction (mirror _coerce_text / _extract_result_output / returncode)
// ---------------------------------------------------------------------------

/// Result of a `run_command` call. Mirrors the SDK command-result object: the
/// output text and an exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub output: String,
    pub exit_code: i32,
}

impl CommandResult {
    pub fn new(output: impl Into<String>, exit_code: i32) -> Self {
        CommandResult {
            output: output.into(),
            exit_code,
        }
    }
}

/// Coerce an optional output value to text, mirroring `_coerce_text` (None ->
/// "", bytes decoded lossily, everything else stringified).
pub fn coerce_text(value: Option<&str>) -> String {
    value.unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Snapshot-id extraction (mirror _extract_snapshot_id)
// ---------------------------------------------------------------------------

/// Extract a snapshot id from a JSON-ish snapshot object, checking the
/// `snapshot_id`, `snapshotId`, and `id` keys in order. Returns the first
/// non-empty string. Mirrors `_extract_snapshot_id`.
pub fn extract_snapshot_id(snapshot: &serde_json::Value) -> Option<String> {
    for key in ["snapshot_id", "snapshotId", "id"] {
        if let Some(s) = snapshot.get(key).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Transient-error classification (mirror _is_transient_vercel_error)
// ---------------------------------------------------------------------------

/// A classified error from a Vercel SDK call. The `status_code` and `name`
/// reproduce the attributes Python's `_extract_status_code` /
/// `_is_transient_vercel_error` inspect; `network` marks
/// httpx network/protocol/read errors.
#[derive(Debug, Clone)]
pub struct VercelError {
    pub message: String,
    pub status_code: Option<u16>,
    /// Lowercased type name, inspected for `ratelimit` / `servererror`.
    pub name: String,
    /// True for httpx NetworkError / ProtocolError / ReadError equivalents.
    pub network: bool,
}

impl VercelError {
    pub fn message(msg: impl Into<String>) -> Self {
        VercelError {
            message: msg.into(),
            status_code: None,
            name: String::new(),
            network: false,
        }
    }

    pub fn status(msg: impl Into<String>, code: u16) -> Self {
        VercelError {
            message: msg.into(),
            status_code: Some(code),
            name: String::new(),
            network: false,
        }
    }

    pub fn network(msg: impl Into<String>) -> Self {
        VercelError {
            message: msg.into(),
            status_code: None,
            name: String::new(),
            network: true,
        }
    }
}

impl std::fmt::Display for VercelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for VercelError {}

/// Whether an error should be retried. Mirrors `_is_transient_vercel_error`:
/// transient if the status code is in [`TRANSIENT_STATUS_CODES`], or it is a
/// network/protocol/read error, or its (lowercased) type name contains
/// `ratelimit` or `servererror`.
pub fn is_transient_vercel_error(err: &VercelError) -> bool {
    if let Some(code) = err.status_code {
        if TRANSIENT_STATUS_CODES.contains(&code) {
            return true;
        }
    }
    if err.network {
        return true;
    }
    let name = err.name.to_ascii_lowercase();
    name.contains("ratelimit") || name.contains("servererror")
}

/// Run `callback` up to `attempts` times, retrying only transient errors with
/// linear backoff (`RETRY_BACKOFF_STEP * attempt`). Mirrors
/// `_retry_vercel_call`. `sleep` is injected for testability; pass
/// `std::thread::sleep` in production.
pub fn retry_vercel_call<T, F, S>(
    label: &str,
    mut callback: F,
    attempts: u32,
    mut sleep: S,
) -> Result<T, VercelError>
where
    F: FnMut() -> Result<T, VercelError>,
    S: FnMut(Duration),
{
    let backoff = RETRY_BACKOFF_STEP;
    let mut attempt: u32 = 1;
    loop {
        match callback() {
            Ok(v) => return Ok(v),
            Err(exc) => {
                if attempt >= attempts || !is_transient_vercel_error(&exc) {
                    return Err(exc);
                }
                log::warn!(
                    "Vercel: {label} failed ({exc}); retrying {attempt}/{attempts}"
                );
                sleep(backoff * attempt);
                attempt += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot store (mirror _load/_save/_get/_store/_delete snapshot helpers)
// ---------------------------------------------------------------------------

/// Load a JSON file as an object, returning `{}` on any error. Inlined faithful
/// copy of `hermes_core::tool_environments_base::load_json_store`.
fn load_json_store(path: &std::path::Path) -> serde_json::Map<String, serde_json::Value> {
    if path.exists() {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(serde_json::Value::Object(map)) =
                serde_json::from_str::<serde_json::Value>(&text)
            {
                return map;
            }
        }
    }
    serde_json::Map::new()
}

/// Write `data` as pretty-printed (2-space indent) JSON to `path`. Inlined
/// faithful copy of `hermes_core::tool_environments_base::save_json_store`.
fn save_json_store(
    path: &std::path::Path,
    data: &serde_json::Map<String, serde_json::Value>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(data.clone()))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, text)
}

static STORE_LOCK: Mutex<()> = Mutex::new(());

/// Atomically read-modify-write a JSON object store under a process lock.
/// Inlined faithful copy of `hermes_core::tool_environments_base::with_json_store`.
fn with_json_store<F, R>(path: &std::path::Path, f: F) -> std::io::Result<R>
where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>) -> R,
{
    let _g = STORE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let mut map = load_json_store(path);
    let r = f(&mut map);
    save_json_store(path, &map)?;
    Ok(r)
}

/// Path to the snapshot store under `hermes_home`. Mirrors
/// `_snapshot_store_path`.
pub fn snapshot_store_path(hermes_home: &std::path::Path) -> std::path::PathBuf {
    hermes_home.join(SNAPSHOT_STORE_NAME)
}

/// Read the snapshot id stored for `task_id`, or `None`. Mirrors
/// `_get_snapshot_id`: empty task ids and empty/non-string values yield `None`.
pub fn get_snapshot_id(store_path: &std::path::Path, task_id: &str) -> Option<String> {
    if task_id.is_empty() {
        return None;
    }
    let map = load_json_store(store_path);
    match map.get(task_id).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => Some(s.to_string()),
        _ => None,
    }
}

/// Store `snapshot_id` for `task_id`. No-op when either is empty. Mirrors
/// `_store_snapshot`.
pub fn store_snapshot(
    store_path: &std::path::Path,
    task_id: &str,
    snapshot_id: &str,
) -> std::io::Result<()> {
    if task_id.is_empty() || snapshot_id.is_empty() {
        return Ok(());
    }
    with_json_store(store_path, |map| {
        map.insert(
            task_id.to_string(),
            serde_json::Value::String(snapshot_id.to_string()),
        );
    })
}

/// Delete the stored snapshot for `task_id`. When `snapshot_id` is `Some` the
/// deletion only happens if the stored value matches. Mirrors
/// `_delete_snapshot`.
pub fn delete_snapshot(
    store_path: &std::path::Path,
    task_id: &str,
    snapshot_id: Option<&str>,
) -> std::io::Result<()> {
    if task_id.is_empty() {
        return Ok(());
    }
    with_json_store(store_path, |map| {
        let existing = map.get(task_id).cloned();
        let existing = match existing {
            None | Some(serde_json::Value::Null) => return,
            Some(v) => v,
        };
        if let Some(want) = snapshot_id {
            if existing.as_str() != Some(want) {
                return;
            }
        }
        map.remove(task_id);
    })
}

// ---------------------------------------------------------------------------
// remote-home / workspace helpers
// ---------------------------------------------------------------------------

/// The `.hermes` base directory under a remote home. Mirrors the inline
/// computation in `_configure_attached_sandbox` / `_vercel_bulk_download`:
/// `/.hermes` when home is `/`, otherwise `{home.rstrip('/')}/.hermes`.
pub fn remote_hermes_base(remote_home: &str) -> String {
    if remote_home == "/" {
        "/.hermes".to_string()
    } else {
        format!("{}/.hermes", remote_home.trim_end_matches('/'))
    }
}

/// Decide the effective `cwd` from the requested cwd, mirroring the branch in
/// `_configure_attached_sandbox`:
///   * `~`            -> remote home
///   * `""` or default Vercel cwd -> workspace root
///   * anything else  -> as requested
pub fn resolve_cwd(requested_cwd: &str, remote_home: &str, workspace_root: &str) -> String {
    if requested_cwd == "~" {
        remote_home.to_string()
    } else if requested_cwd.is_empty() || requested_cwd == DEFAULT_VERCEL_CWD {
        workspace_root.to_string()
    } else {
        requested_cwd.to_string()
    }
}

/// Normalise a detected workspace cwd: keep it only if absolute, else fall back
/// to the default. Mirrors `_detect_workspace_root`.
pub fn normalize_workspace_root(cwd: &str) -> String {
    if cwd.starts_with('/') {
        cwd.to_string()
    } else {
        DEFAULT_VERCEL_CWD.to_string()
    }
}

// ---------------------------------------------------------------------------
// Vercel SDK surface (abstracted for testability)
// ---------------------------------------------------------------------------

/// The slice of the Vercel sandbox SDK the environment needs. Each sandbox the
/// SDK returns is identified by a string handle; the implementation maps that
/// handle back to its live SDK object.
pub trait VercelSdk: Send + Sync {
    /// `Sandbox.create(...)` — fresh sandbox. Returns the sandbox handle.
    fn create(&self, params: &SandboxCreateParams) -> Result<String, VercelError>;

    /// `Sandbox.create(..., source={"type":"snapshot","snapshot_id":id})`.
    fn create_from_snapshot(
        &self,
        params: &SandboxCreateParams,
        snapshot_id: &str,
    ) -> Result<String, VercelError>;

    /// Current `sandbox.status`; `None` mirrors the SDK returning `None`.
    fn status(&self, handle: &str) -> Result<Option<SandboxStatus>, VercelError>;

    /// `sandbox.wait_for_status(RUNNING, timeout, poll_interval)`.
    fn wait_for_running(
        &self,
        handle: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<(), VercelError>;

    /// `sandbox.refresh()`.
    fn refresh(&self, handle: &str) -> Result<(), VercelError>;

    /// `sandbox.sandbox.cwd` — the SDK-reported workspace cwd.
    fn workspace_cwd(&self, handle: &str) -> Result<String, VercelError>;

    /// `sandbox.run_command(program, args, cwd=...)`.
    fn run_command(
        &self,
        handle: &str,
        program: &str,
        args: &[String],
        cwd: &str,
    ) -> Result<CommandResult, VercelError>;

    /// `sandbox.write_files([{path, content}, ...])`.
    fn write_files(&self, handle: &str, files: &[WriteFile]) -> Result<(), VercelError>;

    /// `sandbox.download_file(remote_path, dest)`.
    fn download_file(&self, handle: &str, remote_path: &str, dest: &str)
        -> Result<(), VercelError>;

    /// `sandbox.snapshot()` — returns a snapshot object (as JSON) or error.
    fn snapshot(&self, handle: &str) -> Result<serde_json::Value, VercelError>;

    /// `sandbox.stop(blocking=True, timeout, poll_interval)`.
    fn stop(&self, handle: &str, timeout: Duration, poll_interval: Duration);

    /// `sandbox.client.close()`.
    fn close_client(&self, handle: &str);
}

/// A file to upload, mirroring the SDK `WriteFile` TypedDict
/// (`{"path": ..., "content": bytes}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteFile {
    pub path: String,
    pub content: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Configuration for [`VercelSandboxEnvironment`], mirroring the keyword
/// arguments of the Python `__init__`.
#[derive(Debug, Clone)]
pub struct VercelConfig {
    pub runtime: Option<String>,
    pub cwd: String,
    pub timeout: i64,
    pub cpu: f64,
    pub memory: i64,
    pub disk: i64,
    pub persistent_filesystem: bool,
    pub task_id: String,
}

impl Default for VercelConfig {
    fn default() -> Self {
        VercelConfig {
            runtime: None,
            cwd: DEFAULT_VERCEL_CWD.to_string(),
            timeout: 60,
            cpu: 1.0,
            memory: 5120,
            disk: DEFAULT_CONTAINER_DISK_MB,
            persistent_filesystem: true,
            task_id: "default".to_string(),
        }
    }
}

/// Mutable runtime state guarded by the environment lock.
#[derive(Debug, Default)]
struct VercelState {
    sandbox: Option<String>,
    workspace_root: String,
    remote_home: String,
    cwd: String,
}

/// Vercel cloud sandbox execution backend. Faithful port of
/// `VercelSandboxEnvironment`.
pub struct VercelSandboxEnvironment {
    sdk: Arc<dyn VercelSdk>,
    store_path: std::path::PathBuf,
    persistent: bool,
    task_id: String,
    requested_cwd: String,
    create_params: SandboxCreateParams,
    state: Mutex<VercelState>,
}

/// Stdin handling mode for the base `execute()` contract. Vercel uses
/// `heredoc`, mirroring `_stdin_mode = "heredoc"`.
pub const STDIN_MODE: &str = "heredoc";

impl VercelSandboxEnvironment {
    /// Build the environment and create the initial sandbox. Faithful port of
    /// `__init__` up to (and including) sandbox creation and cwd configuration.
    /// File-sync `sync(force=True)` and `init_session()` are driven by callers
    /// (see [`Self::configure_attached`]).
    ///
    /// `sleep` is injected for retry backoff; pass `std::thread::sleep` in
    /// production.
    pub fn new<S>(
        sdk: Arc<dyn VercelSdk>,
        store_path: std::path::PathBuf,
        cfg: VercelConfig,
        sleep: S,
    ) -> Result<Self, VercelError>
    where
        S: FnMut(Duration) + Clone,
    {
        let create_params = build_create_params(
            cfg.runtime.clone().filter(|r| !r.is_empty()),
            cfg.timeout,
            cfg.cpu,
            cfg.memory,
            cfg.disk,
        )
        .map_err(VercelError::message)?;

        let env = VercelSandboxEnvironment {
            sdk,
            store_path,
            persistent: cfg.persistent_filesystem,
            task_id: cfg.task_id.clone(),
            requested_cwd: cfg.cwd.clone(),
            create_params,
            state: Mutex::new(VercelState::default()),
        };

        let handle = env.create_sandbox(sleep.clone())?;
        {
            let mut st = env.state.lock().unwrap();
            st.sandbox = Some(handle);
        }
        env.configure_attached(&cfg.cwd)?;
        Ok(env)
    }

    /// Create a sandbox, restoring from a stored snapshot when persistent and
    /// one exists, falling back to a fresh sandbox on restore failure. Faithful
    /// port of `_create_sandbox`.
    pub fn create_sandbox<S>(&self, sleep: S) -> Result<String, VercelError>
    where
        S: FnMut(Duration) + Clone,
    {
        let snapshot_id = if self.persistent {
            get_snapshot_id(&self.store_path, &self.task_id)
        } else {
            None
        };

        if let Some(snapshot_id) = snapshot_id {
            let params = self.create_params.clone();
            let sdk = self.sdk.clone();
            let snap = snapshot_id.clone();
            let restore = retry_vercel_call(
                "sandbox restore",
                || sdk.create_from_snapshot(&params, &snap),
                CREATE_RETRY_ATTEMPTS,
                sleep.clone(),
            );
            match restore {
                Ok(handle) => return Ok(handle),
                Err(exc) => {
                    log::warn!(
                        "Vercel: failed to restore snapshot {} for task {}; \
                         falling back to a fresh sandbox: {}",
                        snapshot_id,
                        self.task_id,
                        exc
                    );
                    let _ = delete_snapshot(&self.store_path, &self.task_id, Some(&snapshot_id));
                }
            }
        }

        let params = self.create_params.clone();
        let sdk = self.sdk.clone();
        retry_vercel_call(
            "sandbox create",
            || sdk.create(&params),
            CREATE_RETRY_ATTEMPTS,
            sleep,
        )
    }

    /// Wait for running, detect workspace root + remote home, and resolve the
    /// effective cwd. Faithful port of `_configure_attached_sandbox`
    /// (the FileSyncManager wiring is left to the integrating caller, which has
    /// access to the upload/delete/bulk closures via the `vercel_*` methods).
    pub fn configure_attached(&self, requested_cwd: &str) -> Result<(), VercelError> {
        let handle = self.require_sandbox()?;
        self.wait_for_running(&handle, RUNNING_WAIT_TIMEOUT)?;

        let raw_cwd = self.sdk.workspace_cwd(&handle)?;
        let workspace_root = normalize_workspace_root(&raw_cwd);
        let remote_home = self.detect_remote_home(&handle, &workspace_root);
        let cwd = resolve_cwd(requested_cwd, &remote_home, &workspace_root);

        let mut st = self.state.lock().unwrap();
        st.workspace_root = workspace_root;
        st.remote_home = remote_home;
        st.cwd = cwd;
        Ok(())
    }

    /// Detect the remote `$HOME` via `sh -lc 'printf %s "$HOME"'`, falling back
    /// to the workspace root on error or a non-absolute value. Faithful port of
    /// `_detect_remote_home`.
    pub fn detect_remote_home(&self, handle: &str, workspace_root: &str) -> String {
        let args = vec!["-lc".to_string(), "printf %s \"$HOME\"".to_string()];
        match self.sdk.run_command(handle, "sh", &args, workspace_root) {
            Ok(result) => {
                let home = result.output.trim();
                if home.starts_with('/') {
                    home.to_string()
                } else {
                    workspace_root.to_string()
                }
            }
            Err(exc) => {
                log::debug!(
                    "Vercel: home detection failed for task {}: {}",
                    self.task_id,
                    exc
                );
                workspace_root.to_string()
            }
        }
    }

    /// Wait for the sandbox to reach `RUNNING`. Faithful port of
    /// `_wait_for_running`: returns immediately when status is `None` or
    /// already running, raises on terminal states, otherwise polls.
    pub fn wait_for_running(&self, handle: &str, timeout: Duration) -> Result<(), VercelError> {
        let status = self.sdk.status(handle)?;
        match status {
            None | Some(SandboxStatus::Running) => return Ok(()),
            Some(s) if s.is_terminal() => {
                return Err(VercelError::message(format!(
                    "Sandbox entered terminal state: {s:?}"
                )));
            }
            _ => {}
        }

        let wait_timeout = timeout.max(MIN_RUNNING_WAIT);
        match self
            .sdk
            .wait_for_running(handle, wait_timeout, RUNNING_WAIT_POLL_INTERVAL)
        {
            Ok(()) => Ok(()),
            Err(_timeout_err) => {
                let status = self.sdk.status(handle).ok().flatten();
                match status {
                    Some(s) if s.is_terminal() => Err(VercelError::message(format!(
                        "Sandbox entered terminal state: {s:?}"
                    ))),
                    other => Err(VercelError::message(format!(
                        "Sandbox did not reach running state (last status: {other:?})"
                    ))),
                }
            }
        }
    }

    /// Capture a filesystem snapshot and persist its id. Faithful port of
    /// `_snapshot_sandbox`: no-op (returns `None`) when not persistent or no
    /// task id; logs and returns `None` on failure or missing id.
    pub fn snapshot_sandbox(&self, handle: &str) -> Option<String> {
        if !self.persistent || self.task_id.is_empty() {
            return None;
        }
        let snapshot = match self.sdk.snapshot(handle) {
            Ok(s) => s,
            Err(exc) => {
                log::warn!(
                    "Vercel: filesystem snapshot failed for task {}: {}",
                    self.task_id,
                    exc
                );
                return None;
            }
        };
        let snapshot_id = match extract_snapshot_id(&snapshot) {
            Some(id) => id,
            None => {
                log::warn!(
                    "Vercel: filesystem snapshot for task {} did not return a snapshot id",
                    self.task_id
                );
                return None;
            }
        };
        let _ = store_snapshot(&self.store_path, &self.task_id, &snapshot_id);
        log::info!(
            "Vercel: saved filesystem snapshot {} for task {}",
            snapshot_id,
            self.task_id
        );
        Some(snapshot_id)
    }

    /// Ensure the sandbox exists and is running, recreating on refresh failure
    /// or terminal state. Faithful port of `_ensure_sandbox_ready`.
    pub fn ensure_sandbox_ready<S>(&self, sleep: S) -> Result<(), VercelError>
    where
        S: FnMut(Duration) + Clone,
    {
        let (handle, requested_cwd) = {
            let st = self.state.lock().unwrap();
            let cwd = if !st.cwd.is_empty() {
                st.cwd.clone()
            } else if !self.requested_cwd.is_empty() {
                self.requested_cwd.clone()
            } else {
                DEFAULT_VERCEL_CWD.to_string()
            };
            (st.sandbox.clone(), cwd)
        };

        let handle = match handle {
            None => {
                let h = self.create_sandbox(sleep)?;
                self.state.lock().unwrap().sandbox = Some(h);
                return self.configure_attached(&requested_cwd);
            }
            Some(h) => h,
        };

        if let Err(exc) = self.sdk.refresh(&handle) {
            log::warn!(
                "Vercel: sandbox refresh failed for task {}: {}; recreating",
                self.task_id,
                exc
            );
            self.sdk.close_client(&handle);
            let h = self.create_sandbox(sleep)?;
            self.state.lock().unwrap().sandbox = Some(h);
            return self.configure_attached(&requested_cwd);
        }

        let status = self.sdk.status(&handle).ok().flatten();
        if let Some(s) = status {
            if s.is_terminal() {
                log::warn!(
                    "Vercel: sandbox entered state {:?} for task {}; recreating",
                    s,
                    self.task_id
                );
                self.sdk.close_client(&handle);
                let h = self.create_sandbox(sleep)?;
                self.state.lock().unwrap().sandbox = Some(h);
                return self.configure_attached(&requested_cwd);
            }
        }

        self.wait_for_running(&handle, RUNNING_WAIT_TIMEOUT)
    }

    /// Upload a single file. Mirrors `_vercel_upload`.
    pub fn vercel_upload<S>(
        &self,
        host_path: &str,
        remote_path: &str,
        sleep: S,
    ) -> Result<(), VercelError>
    where
        S: FnMut(Duration) + Clone,
    {
        self.vercel_bulk_upload(&[(host_path.to_string(), remote_path.to_string())], sleep)
    }

    /// Batch-upload files via `write_files`, reading each host file into bytes.
    /// Faithful port of `_vercel_bulk_upload`.
    pub fn vercel_bulk_upload<S>(
        &self,
        files: &[(String, String)],
        sleep: S,
    ) -> Result<(), VercelError>
    where
        S: FnMut(Duration) + Clone,
    {
        if files.is_empty() {
            return Ok(());
        }
        let mut payload: Vec<WriteFile> = Vec::with_capacity(files.len());
        for (host_path, remote_path) in files {
            let content = std::fs::read(host_path)
                .map_err(|e| VercelError::message(format!("read {host_path}: {e}")))?;
            payload.push(WriteFile {
                path: remote_path.clone(),
                content,
            });
        }
        let handle = self.require_sandbox()?;
        let sdk = self.sdk.clone();
        retry_vercel_call(
            "write_files",
            || sdk.write_files(&handle, &payload),
            WRITE_RETRY_ATTEMPTS,
            sleep,
        )
    }

    /// Delete remote files via `bash -lc 'rm -f ...'`. Faithful port of
    /// `_vercel_delete`: raises if the command exits non-zero.
    pub fn vercel_delete(&self, remote_paths: &[String]) -> Result<(), VercelError> {
        if remote_paths.is_empty() {
            return Ok(());
        }
        let handle = self.require_sandbox()?;
        let workspace_root = self.workspace_root();
        let args = vec!["-lc".to_string(), quoted_rm_command(remote_paths)];
        let result = self
            .sdk
            .run_command(&handle, "bash", &args, &workspace_root)?;
        if result.exit_code != 0 {
            return Err(VercelError::message(format!(
                "Vercel delete failed: {}",
                result.output.trim()
            )));
        }
        Ok(())
    }

    /// Tar `.hermes` on the remote, download it, then remove the remote tar.
    /// Faithful port of `_vercel_bulk_download`.
    pub fn vercel_bulk_download(&self, dest_tar_path: &str) -> Result<(), VercelError> {
        let remote_home = self.remote_home();
        let remote_hermes = remote_hermes_base(&remote_home);
        let archive_member = remote_hermes.trim_start_matches('/').to_string();
        let remote_tar = format!("/tmp/.hermes_sync.{}.tar", std::process::id());
        let handle = self.require_sandbox()?;
        let workspace_root = self.workspace_root();

        let tar_cmd = format!(
            "tar cf {} -C / {}",
            shlex_quote(&remote_tar),
            shlex_quote(&archive_member)
        );
        let result = (|| -> Result<(), VercelError> {
            let args = vec!["-lc".to_string(), tar_cmd];
            let result = self
                .sdk
                .run_command(&handle, "bash", &args, &workspace_root)?;
            if result.exit_code != 0 {
                return Err(VercelError::message(format!(
                    "Vercel bulk download failed: {}",
                    result.output.trim()
                )));
            }
            self.sdk.download_file(&handle, &remote_tar, dest_tar_path)
        })();

        // finally: best-effort cleanup of the remote tar.
        let cleanup_args = vec![
            "-lc".to_string(),
            format!("rm -f {}", shlex_quote(&remote_tar)),
        ];
        let _ = self
            .sdk
            .run_command(&handle, "bash", &cleanup_args, &workspace_root);

        result
    }

    /// Run a bash command. Mirrors `_run_bash`'s `exec_fn`: wraps the command
    /// in `bash -c` (or `bash -lc` when `login`) and execs it against the
    /// workspace root. `timeout` and `stdin_data` are intentionally discarded
    /// (the base class enforces timeout via cancel and embeds stdin as a
    /// heredoc before this call). Returns `(output, returncode)`.
    pub fn run_bash(&self, cmd_string: &str, login: bool) -> Result<(String, i32), VercelError> {
        let handle = self.require_sandbox()?;
        let workspace_root = self.workspace_root();
        let flag = if login { "-lc" } else { "-c" };
        let args = vec![flag.to_string(), cmd_string.to_string()];
        let result = self
            .sdk
            .run_command(&handle, "bash", &args, &workspace_root)?;
        Ok((result.output, result.exit_code))
    }

    /// Stop the current sandbox (the `cancel_fn` used by `_run_bash`). Mirrors
    /// the locked `_stop_sandbox(sandbox)` call.
    pub fn cancel(&self) {
        let handle = { self.state.lock().unwrap().sandbox.clone() };
        if let Some(h) = handle {
            self.sdk.stop(&h, STOP_TIMEOUT, STOP_POLL_INTERVAL);
        }
    }

    /// Tear down: optionally sync back, snapshot, stop, and close the client.
    /// Faithful port of `cleanup`. `sync_back` runs while the sandbox/sync
    /// manager are still present; failures are swallowed with a warning.
    pub fn cleanup<F>(&self, sync_back: F)
    where
        F: FnOnce() -> Result<(), VercelError>,
    {
        let handle = {
            let mut st = self.state.lock().unwrap();
            let h = st.sandbox.clone();
            if h.is_some() {
                if let Err(exc) = sync_back() {
                    log::warn!(
                        "Vercel: sync_back failed for task {}: {}",
                        self.task_id,
                        exc
                    );
                }
            }
            st.sandbox = None;
            h
        };

        let handle = match handle {
            None => return,
            Some(h) => h,
        };

        // Snapshot (best-effort), then always stop + close to avoid leaks.
        let _ = self.snapshot_sandbox(&handle);
        self.sdk.stop(&handle, STOP_TIMEOUT, STOP_POLL_INTERVAL);
        self.sdk.close_client(&handle);
    }

    // -- accessors -----------------------------------------------------------

    fn require_sandbox(&self) -> Result<String, VercelError> {
        self.state
            .lock()
            .unwrap()
            .sandbox
            .clone()
            .ok_or_else(|| VercelError::message("Vercel sandbox is not attached"))
    }

    pub fn workspace_root(&self) -> String {
        self.state.lock().unwrap().workspace_root.clone()
    }

    pub fn remote_home(&self) -> String {
        self.state.lock().unwrap().remote_home.clone()
    }

    pub fn cwd(&self) -> String {
        self.state.lock().unwrap().cwd.clone()
    }

    pub fn create_params(&self) -> &SandboxCreateParams {
        &self.create_params
    }

    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn sandbox_handle(&self) -> Option<String> {
        self.state.lock().unwrap().sandbox.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    fn no_sleep(_: Duration) {}

    #[test]
    fn test_sandbox_status_parse_and_terminal() {
        assert_eq!(SandboxStatus::from_api("running"), SandboxStatus::Running);
        assert_eq!(SandboxStatus::from_api("RUNNING"), SandboxStatus::Running);
        assert_eq!(SandboxStatus::from_api("aborted"), SandboxStatus::Aborted);
        assert_eq!(SandboxStatus::from_api("weird"), SandboxStatus::Unknown);
        assert!(SandboxStatus::Aborted.is_terminal());
        assert!(SandboxStatus::Failed.is_terminal());
        assert!(SandboxStatus::Stopped.is_terminal());
        assert!(!SandboxStatus::Running.is_terminal());
        assert!(!SandboxStatus::Pending.is_terminal());
    }

    #[test]
    fn test_build_create_params_disk_validation() {
        assert!(build_create_params(None, 60, 1.0, 5120, 0).is_ok());
        assert!(build_create_params(None, 60, 1.0, 5120, DEFAULT_CONTAINER_DISK_MB).is_ok());
        assert!(build_create_params(None, 60, 1.0, 5120, 1234).is_err());
    }

    #[test]
    fn test_build_create_params_resources_and_timeout() {
        // cpu floored, memory passed through; timeout floored to 5 min.
        let p = build_create_params(Some("node22".into()), 60, 2.7, 5120, 0).unwrap();
        assert_eq!(p.timeout, MIN_SANDBOX_TIMEOUT);
        assert_eq!(p.runtime.as_deref(), Some("node22"));
        assert_eq!(
            p.resources,
            Some(Resources {
                vcpus: Some(2),
                memory: Some(5120)
            })
        );

        // larger explicit timeout retained.
        let p2 = build_create_params(None, 600, 0.0, 0, 0).unwrap();
        assert_eq!(p2.timeout, Duration::from_secs(600));
        // cpu<=0 and memory<=0 -> no resources.
        assert_eq!(p2.resources, None);

        // only memory.
        let p3 = build_create_params(None, 60, 0.0, 2048, 0).unwrap();
        assert_eq!(
            p3.resources,
            Some(Resources {
                vcpus: None,
                memory: Some(2048)
            })
        );
    }

    #[test]
    fn test_transient_classification() {
        for code in TRANSIENT_STATUS_CODES {
            assert!(is_transient_vercel_error(&VercelError::status("x", *code)));
        }
        assert!(!is_transient_vercel_error(&VercelError::status("x", 404)));
        assert!(is_transient_vercel_error(&VercelError::network("net")));
        let mut e = VercelError::message("boom");
        e.name = "RateLimitError".into();
        assert!(is_transient_vercel_error(&e));
        e.name = "InternalServerError".into();
        assert!(is_transient_vercel_error(&e));
        e.name = "ValueError".into();
        assert!(!is_transient_vercel_error(&e));
    }

    #[test]
    fn test_retry_retries_transient_then_succeeds() {
        let calls = StdMutex::new(0);
        let res: Result<i32, VercelError> = retry_vercel_call(
            "x",
            || {
                let mut n = calls.lock().unwrap();
                *n += 1;
                if *n < 3 {
                    Err(VercelError::status("rl", 429))
                } else {
                    Ok(42)
                }
            },
            3,
            no_sleep,
        );
        assert_eq!(res.unwrap(), 42);
        assert_eq!(*calls.lock().unwrap(), 3);
    }

    #[test]
    fn test_retry_does_not_retry_permanent() {
        let calls = StdMutex::new(0);
        let res: Result<i32, VercelError> = retry_vercel_call(
            "x",
            || {
                *calls.lock().unwrap() += 1;
                Err(VercelError::status("nope", 404))
            },
            3,
            no_sleep,
        );
        assert!(res.is_err());
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn test_extract_snapshot_id_variants() {
        let v = serde_json::json!({"snapshotId": "abc"});
        assert_eq!(extract_snapshot_id(&v).as_deref(), Some("abc"));
        let v = serde_json::json!({"id": "xyz"});
        assert_eq!(extract_snapshot_id(&v).as_deref(), Some("xyz"));
        let v = serde_json::json!({"snapshot_id": "first", "id": "second"});
        assert_eq!(extract_snapshot_id(&v).as_deref(), Some("first"));
        let v = serde_json::json!({"snapshot_id": ""});
        assert_eq!(extract_snapshot_id(&v), None);
        let v = serde_json::json!({"other": 1});
        assert_eq!(extract_snapshot_id(&v), None);
    }

    #[test]
    fn test_snapshot_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vercel_snap_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = snapshot_store_path(&dir);
        let _ = std::fs::remove_file(&path);

        assert_eq!(get_snapshot_id(&path, "task1"), None);
        assert_eq!(get_snapshot_id(&path, ""), None);

        store_snapshot(&path, "task1", "snap-1").unwrap();
        assert_eq!(get_snapshot_id(&path, "task1").as_deref(), Some("snap-1"));

        // empty ids are no-ops.
        store_snapshot(&path, "", "snap-x").unwrap();
        store_snapshot(&path, "task2", "").unwrap();
        assert_eq!(get_snapshot_id(&path, "task2"), None);

        // delete with mismatched id is a no-op.
        delete_snapshot(&path, "task1", Some("other")).unwrap();
        assert_eq!(get_snapshot_id(&path, "task1").as_deref(), Some("snap-1"));

        // matching delete clears it.
        delete_snapshot(&path, "task1", Some("snap-1")).unwrap();
        assert_eq!(get_snapshot_id(&path, "task1"), None);

        // unconditional delete.
        store_snapshot(&path, "task3", "snap-3").unwrap();
        delete_snapshot(&path, "task3", None).unwrap();
        assert_eq!(get_snapshot_id(&path, "task3"), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_remote_hermes_base_and_cwd_resolution() {
        assert_eq!(remote_hermes_base("/"), "/.hermes");
        assert_eq!(remote_hermes_base("/home/user"), "/home/user/.hermes");
        assert_eq!(remote_hermes_base("/home/user/"), "/home/user/.hermes");

        assert_eq!(resolve_cwd("~", "/home/u", "/vercel/sandbox"), "/home/u");
        assert_eq!(resolve_cwd("", "/home/u", "/vercel/sandbox"), "/vercel/sandbox");
        assert_eq!(
            resolve_cwd(DEFAULT_VERCEL_CWD, "/home/u", "/work"),
            "/work"
        );
        assert_eq!(resolve_cwd("/custom", "/home/u", "/work"), "/custom");

        assert_eq!(normalize_workspace_root("/abs"), "/abs");
        assert_eq!(normalize_workspace_root("rel"), DEFAULT_VERCEL_CWD);
    }

    #[test]
    fn test_quoted_rm_command() {
        let paths = vec!["/a/b".to_string(), "/c d/e".to_string()];
        assert_eq!(quoted_rm_command(&paths), "rm -f /a/b '/c d/e'");
    }

    // -- mock SDK + environment-level tests ---------------------------------

    #[derive(Default)]
    struct MockState {
        calls: Vec<String>,
        status: HashMap<String, SandboxStatus>,
        next_handle: usize,
        snapshot: Option<serde_json::Value>,
        snapshot_err: bool,
        workspace_cwd: String,
        home_output: Option<String>,
    }

    struct MockSdk {
        st: StdMutex<MockState>,
    }

    impl MockSdk {
        fn new() -> Self {
            MockSdk {
                st: StdMutex::new(MockState {
                    workspace_cwd: "/vercel/sandbox".to_string(),
                    home_output: Some("/root".to_string()),
                    ..Default::default()
                }),
            }
        }
        fn push(&self, s: &str) {
            self.st.lock().unwrap().calls.push(s.to_string());
        }
        fn calls(&self) -> Vec<String> {
            self.st.lock().unwrap().calls.clone()
        }
    }

    impl VercelSdk for MockSdk {
        fn create(&self, _p: &SandboxCreateParams) -> Result<String, VercelError> {
            let mut st = self.st.lock().unwrap();
            st.next_handle += 1;
            let h = format!("sb-{}", st.next_handle);
            st.status.insert(h.clone(), SandboxStatus::Running);
            st.calls.push(format!("create:{h}"));
            Ok(h)
        }
        fn create_from_snapshot(
            &self,
            _p: &SandboxCreateParams,
            snapshot_id: &str,
        ) -> Result<String, VercelError> {
            self.push(&format!("restore:{snapshot_id}"));
            let mut st = self.st.lock().unwrap();
            st.next_handle += 1;
            let h = format!("sb-{}", st.next_handle);
            st.status.insert(h.clone(), SandboxStatus::Running);
            Ok(h)
        }
        fn status(&self, handle: &str) -> Result<Option<SandboxStatus>, VercelError> {
            Ok(self.st.lock().unwrap().status.get(handle).copied())
        }
        fn wait_for_running(
            &self,
            handle: &str,
            _t: Duration,
            _p: Duration,
        ) -> Result<(), VercelError> {
            self.push(&format!("wait:{handle}"));
            Ok(())
        }
        fn refresh(&self, handle: &str) -> Result<(), VercelError> {
            self.push(&format!("refresh:{handle}"));
            Ok(())
        }
        fn workspace_cwd(&self, _handle: &str) -> Result<String, VercelError> {
            Ok(self.st.lock().unwrap().workspace_cwd.clone())
        }
        fn run_command(
            &self,
            handle: &str,
            program: &str,
            args: &[String],
            cwd: &str,
        ) -> Result<CommandResult, VercelError> {
            self.push(&format!("run:{handle}:{program}:{}:{cwd}", args.join(",")));
            // emulate $HOME detection
            if program == "sh" && args.len() == 2 && args[1].contains("$HOME") {
                let home = self.st.lock().unwrap().home_output.clone();
                return Ok(CommandResult::new(home.unwrap_or_default(), 0));
            }
            Ok(CommandResult::new("", 0))
        }
        fn write_files(&self, handle: &str, files: &[WriteFile]) -> Result<(), VercelError> {
            self.push(&format!("write_files:{handle}:{}", files.len()));
            Ok(())
        }
        fn download_file(
            &self,
            handle: &str,
            remote_path: &str,
            dest: &str,
        ) -> Result<(), VercelError> {
            self.push(&format!("download:{handle}:{remote_path}:{dest}"));
            Ok(())
        }
        fn snapshot(&self, handle: &str) -> Result<serde_json::Value, VercelError> {
            self.push(&format!("snapshot:{handle}"));
            let st = self.st.lock().unwrap();
            if st.snapshot_err {
                return Err(VercelError::message("snap failed"));
            }
            Ok(st
                .snapshot
                .clone()
                .unwrap_or_else(|| serde_json::json!({"snapshot_id": "snap-x"})))
        }
        fn stop(&self, handle: &str, _t: Duration, _p: Duration) {
            self.push(&format!("stop:{handle}"));
        }
        fn close_client(&self, handle: &str) {
            self.push(&format!("close:{handle}"));
        }
    }

    fn temp_store() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vercel_env_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        snapshot_store_path(&dir)
    }

    #[test]
    fn test_new_creates_and_configures() {
        let sdk = Arc::new(MockSdk::new());
        let store = temp_store();
        let env = VercelSandboxEnvironment::new(
            sdk.clone(),
            store,
            VercelConfig {
                persistent_filesystem: false,
                task_id: "t1".into(),
                ..Default::default()
            },
            no_sleep,
        )
        .unwrap();

        // workspace root from sdk cwd, home detected via run_command.
        assert_eq!(env.workspace_root(), "/vercel/sandbox");
        assert_eq!(env.remote_home(), "/root");
        // default cwd resolves to workspace root.
        assert_eq!(env.cwd(), "/vercel/sandbox");

        let calls = sdk.calls();
        assert!(calls.iter().any(|c| c.starts_with("create:")));
        assert!(calls.iter().any(|c| c.starts_with("wait:")));
        assert!(calls.iter().any(|c| c.contains(":sh:")));
    }

    #[test]
    fn test_run_bash_wraps_login_and_plain() {
        let sdk = Arc::new(MockSdk::new());
        let store = temp_store();
        let env = VercelSandboxEnvironment::new(
            sdk.clone(),
            store,
            VercelConfig {
                persistent_filesystem: false,
                ..Default::default()
            },
            no_sleep,
        )
        .unwrap();

        let (_out, rc) = env.run_bash("echo hi", false).unwrap();
        assert_eq!(rc, 0);
        let (_out, _rc) = env.run_bash("echo hi", true).unwrap();

        let calls = sdk.calls();
        assert!(calls
            .iter()
            .any(|c| c.contains(":bash:-c,echo hi:/vercel/sandbox")));
        assert!(calls
            .iter()
            .any(|c| c.contains(":bash:-lc,echo hi:/vercel/sandbox")));
    }

    #[test]
    fn test_snapshot_on_cleanup_persists_id() {
        let sdk = Arc::new(MockSdk::new());
        let store = temp_store();
        let env = VercelSandboxEnvironment::new(
            sdk.clone(),
            store.clone(),
            VercelConfig {
                persistent_filesystem: true,
                task_id: "task-snap".into(),
                ..Default::default()
            },
            no_sleep,
        )
        .unwrap();

        env.cleanup(|| Ok(()));

        // snapshot id stored, sandbox stopped + closed.
        assert_eq!(
            get_snapshot_id(&store, "task-snap").as_deref(),
            Some("snap-x")
        );
        let calls = sdk.calls();
        assert!(calls.iter().any(|c| c.starts_with("snapshot:")));
        assert!(calls.iter().any(|c| c.starts_with("stop:")));
        assert!(calls.iter().any(|c| c.starts_with("close:")));
    }

    #[test]
    fn test_restore_from_existing_snapshot() {
        let sdk = Arc::new(MockSdk::new());
        let store = temp_store();
        store_snapshot(&store, "task-r", "snap-prev").unwrap();

        let _env = VercelSandboxEnvironment::new(
            sdk.clone(),
            store,
            VercelConfig {
                persistent_filesystem: true,
                task_id: "task-r".into(),
                ..Default::default()
            },
            no_sleep,
        )
        .unwrap();

        let calls = sdk.calls();
        assert!(calls.iter().any(|c| c == "restore:snap-prev"));
        // no fresh create path should have been needed.
        assert!(!calls.iter().any(|c| c.starts_with("create:")));
    }

    #[test]
    fn test_ensure_ready_recreates_on_terminal() {
        let sdk = Arc::new(MockSdk::new());
        let store = temp_store();
        let env = VercelSandboxEnvironment::new(
            sdk.clone(),
            store,
            VercelConfig {
                persistent_filesystem: false,
                ..Default::default()
            },
            no_sleep,
        )
        .unwrap();

        // force the current sandbox into a terminal state.
        let handle = env.sandbox_handle().unwrap();
        sdk.st
            .lock()
            .unwrap()
            .status
            .insert(handle.clone(), SandboxStatus::Aborted);

        env.ensure_sandbox_ready(no_sleep).unwrap();

        let calls = sdk.calls();
        assert!(calls.iter().any(|c| c == format!("close:{handle}")));
        // a new sandbox was created.
        assert!(calls.iter().filter(|c| c.starts_with("create:")).count() >= 2);
    }
}
