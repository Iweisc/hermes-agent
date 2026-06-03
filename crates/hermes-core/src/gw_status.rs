//! Gateway runtime status helpers.
//!
//! Native Rust port of `gateway/status.py`.
//!
//! Provides PID-file based detection of whether the gateway daemon is running,
//! cross-process runtime locking, machine-local scoped identity locks, and the
//! `--replace` / planned-stop takeover markers.
//!
//! The PID file lives at `{HERMES_HOME}/gateway.pid`. HERMES_HOME defaults to
//! `~/.hermes` but can be overridden via the environment variable. This means
//! separate HERMES_HOME directories naturally get separate PID files.
//!
//! Behavioural notes vs. the Python original:
//! * This is a POSIX-only port. Windows-specific code paths (`msvcrt` locks,
//!   `taskkill`) are not reproduced — the gateway runs on POSIX. The
//!   `_IS_WINDOWS` branches collapse to their POSIX equivalents.
//! * Process start time is read from `/proc/<pid>/stat` field 22 exactly as the
//!   Python original (returns `None` on platforms without `/proc`, e.g. macOS,
//!   matching the Python behaviour where the same read fails).
//! * The cross-process runtime lock uses `flock(LOCK_EX | LOCK_NB)` via `libc`,
//!   mirroring `fcntl.flock`. The process-owned lock handle is kept in a global
//!   `Mutex<Option<...>>` like the Python module-level `_gateway_lock_handle`.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::atomic_json_write;

pub const GATEWAY_KIND: &str = "hermes-gateway";
const RUNTIME_STATUS_FILE: &str = "gateway_state.json";
const LOCKS_DIRNAME: &str = "gateway-locks";
const GATEWAY_LOCK_FILENAME: &str = "gateway.lock";

const TAKEOVER_MARKER_FILENAME: &str = ".gateway-takeover.json";
const TAKEOVER_MARKER_TTL_S: i64 = 60;
const PLANNED_STOP_MARKER_FILENAME: &str = ".gateway-planned-stop.json";
const PLANNED_STOP_MARKER_TTL_S: i64 = 60;

/// Live handle for the process-owned gateway runtime lock. Holding the `File`
/// keeps the `flock` claimed; dropping it (or explicitly releasing) frees it.
static GATEWAY_LOCK_HANDLE: Mutex<Option<File>> = Mutex::new(None);

// ── Path helpers ──────────────────────────────────────────────────────

/// Return the path to the gateway PID file, respecting HERMES_HOME.
pub fn get_pid_path() -> PathBuf {
    get_hermes_home().join("gateway.pid")
}

/// Return the path to the runtime gateway lock file.
///
/// When `pid_path` is supplied the lock sits beside it; otherwise it lives in
/// HERMES_HOME.
pub fn get_gateway_lock_path(pid_path: Option<&Path>) -> PathBuf {
    match pid_path {
        Some(p) => with_name(p, GATEWAY_LOCK_FILENAME),
        None => get_hermes_home().join(GATEWAY_LOCK_FILENAME),
    }
}

/// Return the persisted runtime health/status file path.
pub fn get_runtime_status_path() -> PathBuf {
    with_name(&get_pid_path(), RUNTIME_STATUS_FILE)
}

/// Return the machine-local directory for token-scoped gateway locks.
pub fn get_lock_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("HERMES_GATEWAY_LOCK_DIR") {
        if !override_dir.is_empty() {
            return PathBuf::from(override_dir);
        }
    }
    let state_home = match std::env::var("XDG_STATE_HOME") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => home_dir().join(".local").join("state"),
    };
    state_home.join("hermes").join(LOCKS_DIRNAME)
}

fn get_takeover_marker_path() -> PathBuf {
    get_hermes_home().join(TAKEOVER_MARKER_FILENAME)
}

fn get_planned_stop_marker_path() -> PathBuf {
    get_hermes_home().join(PLANNED_STOP_MARKER_FILENAME)
}

/// Replicate `Path.with_name(name)`: replace the final path component.
fn with_name(path: &Path, name: &str) -> PathBuf {
    match path.parent() {
        Some(parent) => parent.join(name),
        None => PathBuf::from(name),
    }
}

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

fn utc_now_iso() -> String {
    // Python's datetime.now(timezone.utc).isoformat() yields e.g.
    // "2026-06-03T12:34:56.789012+00:00" with microsecond precision.
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

// ── Process inspection ────────────────────────────────────────────────

/// Return the kernel start time for a process when available.
///
/// Field 22 in `/proc/<pid>/stat` is process start time (clock ticks). Returns
/// `None` when `/proc` is unavailable (e.g. macOS) or parsing fails, matching
/// the Python original.
pub fn get_process_start_time(pid: i64) -> Option<i64> {
    let stat_path = format!("/proc/{pid}/stat");
    let raw = fs::read_to_string(stat_path).ok()?;
    // The comm field (field 2) may contain spaces inside parentheses; field 22
    // is well past it and the Python original simply split on whitespace and
    // indexed [21], so we do the same.
    let fields: Vec<&str> = raw.split_whitespace().collect();
    fields.get(21).and_then(|s| s.parse::<i64>().ok())
}

/// Return the process command line as a space-separated string.
fn read_process_cmdline(pid: i64) -> Option<String> {
    let cmdline_path = format!("/proc/{pid}/cmdline");
    let raw = fs::read(cmdline_path).ok()?;
    if raw.is_empty() {
        return None;
    }
    let replaced: Vec<u8> = raw
        .into_iter()
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    let s = String::from_utf8_lossy(&replaced).trim().to_string();
    Some(s)
}

const GATEWAY_CMDLINE_PATTERNS: &[&str] = &[
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "hermes gateway",
    "hermes-gateway",
    "gateway/run.py",
];

const GATEWAY_RECORD_PATTERNS: &[&str] = &[
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "hermes gateway",
    "gateway/run.py",
];

/// Return true when the live PID still looks like the Hermes gateway.
pub fn looks_like_gateway_process(pid: i64) -> bool {
    match read_process_cmdline(pid) {
        Some(cmdline) if !cmdline.is_empty() => GATEWAY_CMDLINE_PATTERNS
            .iter()
            .any(|p| cmdline.contains(p)),
        _ => false,
    }
}

/// Validate gateway identity from PID-file metadata when cmdline is unavailable.
pub fn record_looks_like_gateway(record: &Value) -> bool {
    if record.get("kind").and_then(|v| v.as_str()) != Some(GATEWAY_KIND) {
        return false;
    }
    let argv = match record.get("argv").and_then(|v| v.as_array()) {
        Some(a) if !a.is_empty() => a,
        _ => return false,
    };
    let cmdline = argv
        .iter()
        .map(value_to_string_part)
        .collect::<Vec<_>>()
        .join(" ");
    GATEWAY_RECORD_PATTERNS.iter().any(|p| cmdline.contains(p))
}

/// Mirror `str(part)` for argv entries (which are normally JSON strings).
fn value_to_string_part(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Return whether a PID is alive (mirrors `os.kill(pid, 0)`).
///
/// Returns `Ok(())` if the process exists and is signalable, or maps the errno
/// to a [`KillError`] for the caller to distinguish existence vs. permission.
fn pid_exists(pid: i64) -> Result<(), KillError> {
    // SAFETY: kill with signal 0 performs an existence/permission check only.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return Ok(());
    }
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    match errno {
        libc::ESRCH => Err(KillError::NoSuchProcess),
        libc::EPERM => Err(KillError::PermissionDenied),
        _ => Err(KillError::Other),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KillError {
    NoSuchProcess,
    PermissionDenied,
    Other,
}

/// Terminate a PID. POSIX uses SIGTERM/SIGKILL depending on `force`.
pub fn terminate_pid(pid: i64, force: bool) -> std::io::Result<()> {
    let sig = if force { libc::SIGKILL } else { libc::SIGTERM };
    // SAFETY: standard signal delivery.
    let rc = unsafe { libc::kill(pid as libc::pid_t, sig) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn current_pid() -> i64 {
    std::process::id() as i64
}

fn current_argv() -> Vec<String> {
    std::env::args().collect()
}

// ── PID / status records ──────────────────────────────────────────────

fn build_pid_record() -> Value {
    json!({
        "pid": current_pid(),
        "kind": GATEWAY_KIND,
        "argv": current_argv(),
        "start_time": get_process_start_time(current_pid()),
    })
}

fn build_runtime_status_record() -> Value {
    let mut payload = build_pid_record();
    let obj = payload.as_object_mut().unwrap();
    obj.insert("gateway_state".into(), json!("starting"));
    obj.insert("exit_reason".into(), Value::Null);
    obj.insert("restart_requested".into(), json!(false));
    obj.insert("active_agents".into(), json!(0));
    obj.insert("platforms".into(), json!({}));
    obj.insert("updated_at".into(), json!(utc_now_iso()));
    payload
}

/// Read a JSON object from `path`, returning `None` on any failure or when the
/// payload is not a JSON object.
fn read_json_file(path: &Path) -> Option<Value> {
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let payload: Value = serde_json::from_str(raw).ok()?;
    if payload.is_object() {
        Some(payload)
    } else {
        None
    }
}

fn write_json_file(path: &Path, payload: &Value) -> std::io::Result<()> {
    // Python writes compact JSON (indent=None, separators=(",", ":")).
    atomic_json_write(path, payload, 0)
}

/// Read a PID record, accepting either a JSON object, a bare JSON int, or a raw
/// integer string (mirrors `_read_pid_record`).
fn read_pid_record(pid_path: &Path) -> Option<Value> {
    if !pid_path.exists() {
        return None;
    }
    let raw = fs::read_to_string(pid_path).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(map)) => Some(Value::Object(map)),
        Ok(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            Some(json!({ "pid": n }))
        }
        Ok(_) => None,
        Err(_) => {
            // Not valid JSON: try a bare integer string.
            raw.parse::<i64>().ok().map(|p| json!({ "pid": p }))
        }
    }
}

fn read_gateway_lock_record(lock_path: &Path) -> Option<Value> {
    read_pid_record(lock_path)
}

/// Extract an integer PID from a record (mirrors `_pid_from_record`).
fn pid_from_record(record: &Option<Value>) -> Option<i64> {
    let record = record.as_ref()?;
    coerce_pid(record.get("pid")?)
}

/// Coerce a JSON value to an integer PID like Python's `int(record["pid"])`.
fn coerce_pid(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// Delete a stale gateway PID file (and its sibling lock metadata).
fn cleanup_invalid_pid_path(pid_path: &Path, cleanup_stale: bool) {
    if !cleanup_stale {
        return;
    }
    let _ = fs::remove_file(pid_path);
    let _ = fs::remove_file(get_gateway_lock_path(Some(pid_path)));
}

// ── Runtime lock (flock-based) ────────────────────────────────────────

fn try_acquire_file_lock(handle: &File) -> bool {
    // SAFETY: flock on a valid fd.
    let rc = unsafe { libc::flock(handle.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    rc == 0
}

fn release_file_lock(handle: &File) {
    // SAFETY: flock on a valid fd.
    unsafe {
        libc::flock(handle.as_raw_fd(), libc::LOCK_UN);
    }
}

fn write_gateway_lock_record(handle: &mut File) -> std::io::Result<()> {
    handle.seek(SeekFrom::Start(0))?;
    handle.set_len(0)?;
    let record = build_pid_record();
    let serialized = serde_json::to_string(&record)?;
    handle.write_all(serialized.as_bytes())?;
    handle.flush()?;
    handle.sync_all().ok();
    Ok(())
}

/// Claim the cross-process runtime lock for the gateway.
///
/// Unlike the PID file, the lock is owned by the live process itself. If the
/// process dies abruptly, the OS releases the lock automatically.
pub fn acquire_gateway_runtime_lock() -> bool {
    let mut guard = GATEWAY_LOCK_HANDLE.lock().unwrap();
    if guard.is_some() {
        return true;
    }

    let path = get_gateway_lock_path(None);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut handle = match OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&path)
    {
        Ok(h) => h,
        Err(_) => return false,
    };
    if !try_acquire_file_lock(&handle) {
        return false;
    }
    if write_gateway_lock_record(&mut handle).is_err() {
        // Best-effort: keep the lock held even if record write failed, matching
        // Python which would raise — but here we still hold the lock.
    }
    *guard = Some(handle);
    true
}

/// Release the gateway runtime lock when owned by this process.
pub fn release_gateway_runtime_lock() {
    let mut guard = GATEWAY_LOCK_HANDLE.lock().unwrap();
    if let Some(handle) = guard.take() {
        release_file_lock(&handle);
        // Drop closes the file.
        drop(handle);
    }
}

/// Return true when some process currently owns the gateway runtime lock.
pub fn is_gateway_runtime_lock_active(lock_path: Option<&Path>) -> bool {
    let default_path = get_gateway_lock_path(None);
    let resolved_lock_path = lock_path.map(|p| p.to_path_buf()).unwrap_or_else(|| default_path.clone());

    {
        let guard = GATEWAY_LOCK_HANDLE.lock().unwrap();
        if guard.is_some() && resolved_lock_path == default_path {
            return true;
        }
    }

    if !resolved_lock_path.exists() {
        return false;
    }

    let handle = match OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&resolved_lock_path)
    {
        Ok(h) => h,
        Err(_) => return false,
    };
    if try_acquire_file_lock(&handle) {
        release_file_lock(&handle);
        false
    } else {
        true
    }
}

// ── PID file ──────────────────────────────────────────────────────────

/// Error returned by [`write_pid_file`] when another process already owns it.
#[derive(Debug)]
pub enum WritePidError {
    /// The PID file already exists (another gateway is racing us).
    AlreadyExists,
    /// Some other I/O error occurred.
    Io(std::io::Error),
}

impl std::fmt::Display for WritePidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WritePidError::AlreadyExists => write!(f, "gateway PID file already exists"),
            WritePidError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WritePidError {}

/// Write the current process PID and metadata to the gateway PID file.
///
/// Uses atomic create-new (`O_CREAT | O_EXCL`) so that concurrent `--replace`
/// invocations race: exactly one process wins and the rest get
/// [`WritePidError::AlreadyExists`].
pub fn write_pid_file() -> Result<(), WritePidError> {
    let path = get_pid_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(WritePidError::Io)?;
    }
    let record = serde_json::to_string(&build_pid_record())
        .map_err(|e| WritePidError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;

    let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(WritePidError::AlreadyExists);
        }
        Err(e) => return Err(WritePidError::Io(e)),
    };
    if let Err(e) = file.write_all(record.as_bytes()) {
        let _ = fs::remove_file(&path);
        return Err(WritePidError::Io(e));
    }
    Ok(())
}

/// Remove the gateway PID file, but only if it belongs to this process.
///
/// During `--replace` handoffs the old process's exit handler can fire after
/// the new process has written its own PID file; blindly removing it would
/// delete the new record.
pub fn remove_pid_file() {
    let path = get_pid_path();
    if let Some(record) = read_json_file(&path) {
        let file_pid = record.get("pid").and_then(coerce_pid);
        if let Some(fp) = file_pid {
            if fp != current_pid() {
                // PID file belongs to a different process — leave it alone.
                return;
            }
        }
    }
    let _ = fs::remove_file(&path);
}

// ── Runtime status ────────────────────────────────────────────────────

/// Optional field update. `None` means "leave unchanged" (Python's `_UNSET`);
/// `Some(value)` sets it.
#[derive(Default)]
pub struct RuntimeStatusUpdate {
    pub gateway_state: Option<Value>,
    pub exit_reason: Option<Value>,
    pub restart_requested: Option<bool>,
    pub active_agents: Option<i64>,
    pub platform: Option<String>,
    pub platform_state: Option<Value>,
    pub error_code: Option<Value>,
    pub error_message: Option<Value>,
}

/// Persist gateway runtime health information for diagnostics/status.
pub fn write_runtime_status(update: &RuntimeStatusUpdate) -> std::io::Result<()> {
    let path = get_runtime_status_path();
    let mut payload = read_json_file(&path).unwrap_or_else(build_runtime_status_record);
    {
        let obj = payload.as_object_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "status payload not an object")
        })?;
        if !obj.contains_key("platforms") {
            obj.insert("platforms".into(), json!({}));
        }
        if !obj.contains_key("kind") {
            obj.insert("kind".into(), json!(GATEWAY_KIND));
        }
        obj.insert("pid".into(), json!(current_pid()));
        obj.insert(
            "start_time".into(),
            match get_process_start_time(current_pid()) {
                Some(t) => json!(t),
                None => Value::Null,
            },
        );
        obj.insert("updated_at".into(), json!(utc_now_iso()));

        if let Some(gs) = &update.gateway_state {
            obj.insert("gateway_state".into(), gs.clone());
        }
        if let Some(er) = &update.exit_reason {
            obj.insert("exit_reason".into(), er.clone());
        }
        if let Some(rr) = update.restart_requested {
            obj.insert("restart_requested".into(), json!(rr));
        }
        if let Some(aa) = update.active_agents {
            obj.insert("active_agents".into(), json!(aa.max(0)));
        }

        if let Some(platform) = &update.platform {
            let platforms = obj
                .get_mut("platforms")
                .and_then(|v| v.as_object_mut())
                .expect("platforms is an object");
            let mut platform_payload = platforms
                .get(platform)
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            if let Some(ps) = &update.platform_state {
                platform_payload.insert("state".into(), ps.clone());
            }
            if let Some(ec) = &update.error_code {
                platform_payload.insert("error_code".into(), ec.clone());
            }
            if let Some(em) = &update.error_message {
                platform_payload.insert("error_message".into(), em.clone());
            }
            platform_payload.insert("updated_at".into(), json!(utc_now_iso()));
            platforms.insert(platform.clone(), Value::Object(platform_payload));
        }
    }
    write_json_file(&path, &payload)
}

/// Read the persisted gateway runtime health/status information.
pub fn read_runtime_status() -> Option<Value> {
    read_json_file(&get_runtime_status_path())
}

// ── Scoped locks ──────────────────────────────────────────────────────

fn scope_hash(identity: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(identity.as_bytes());
    let digest = hasher.finalize();
    let hex = hex_encode(&digest);
    hex[..16].to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn get_scope_lock_path(scope: &str, identity: &str) -> PathBuf {
    get_lock_dir().join(format!("{scope}-{}.lock", scope_hash(identity)))
}

/// Check whether a process appears stopped (SIGTSTP / trace-stop) via
/// `/proc/<pid>/status` State field. Stopped processes still respond to
/// `kill(pid, 0)` but are not actually running.
fn process_is_stopped(pid: i64) -> bool {
    let status_path = format!("/proc/{pid}/status");
    if let Ok(content) = fs::read_to_string(status_path) {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("State:") {
                if let Some(state) = rest.split_whitespace().next() {
                    return state == "T" || state == "t";
                }
                break;
            }
        }
    }
    false
}

/// Acquire a machine-local lock keyed by scope + identity.
///
/// Returns `(acquired, existing_record)`. When acquisition fails because a live
/// owner holds the lock, `existing_record` carries that owner's record.
pub fn acquire_scoped_lock(
    scope: &str,
    identity: &str,
    metadata: Option<Value>,
) -> (bool, Option<Value>) {
    let lock_path = get_scope_lock_path(scope, identity);
    if let Some(parent) = lock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut record = build_pid_record();
    {
        let obj = record.as_object_mut().unwrap();
        obj.insert("scope".into(), json!(scope));
        obj.insert("identity_hash".into(), json!(scope_hash(identity)));
        obj.insert("metadata".into(), metadata.unwrap_or_else(|| json!({})));
        obj.insert("updated_at".into(), json!(utc_now_iso()));
    }
    let record_start_time = record.get("start_time").cloned().unwrap_or(Value::Null);

    let existing = read_json_file(&lock_path);
    if existing.is_none() && lock_path.exists() {
        // Lock file exists but is empty or invalid JSON — treat as stale.
        let _ = fs::remove_file(&lock_path);
    }

    if let Some(existing) = existing {
        let existing_pid = existing.get("pid").and_then(coerce_pid);

        if existing_pid == Some(current_pid())
            && existing.get("start_time").cloned().unwrap_or(Value::Null) == record_start_time
        {
            let _ = write_json_file(&lock_path, &record);
            return (true, Some(existing));
        }

        let mut stale = existing_pid.is_none();
        if !stale {
            let pid = existing_pid.unwrap();
            match pid_exists(pid) {
                Err(KillError::NoSuchProcess)
                | Err(KillError::PermissionDenied)
                | Err(KillError::Other) => {
                    stale = true;
                }
                Ok(()) => {
                    let recorded_start = existing.get("start_time").and_then(|v| v.as_i64());
                    let current_start = get_process_start_time(pid);
                    if let (Some(rs), Some(cs)) = (recorded_start, current_start) {
                        if cs != rs {
                            stale = true;
                        }
                    }
                    if !stale && process_is_stopped(pid) {
                        stale = true;
                    }
                }
            }
        }

        if stale {
            let _ = fs::remove_file(&lock_path);
        } else {
            return (false, Some(existing));
        }
    }

    // Atomic create-new; if it already exists, someone else won the race.
    match OpenOptions::new().write(true).create_new(true).open(&lock_path) {
        Ok(mut handle) => {
            let serialized = serde_json::to_string(&record).unwrap_or_default();
            if handle.write_all(serialized.as_bytes()).is_err() {
                let _ = fs::remove_file(&lock_path);
                return (false, None);
            }
            (true, None)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            (false, read_json_file(&lock_path))
        }
        Err(_) => (false, None),
    }
}

/// Release a previously-acquired scope lock when owned by this process.
pub fn release_scoped_lock(scope: &str, identity: &str) {
    let lock_path = get_scope_lock_path(scope, identity);
    let existing = match read_json_file(&lock_path) {
        Some(e) => e,
        None => return,
    };
    let existing_pid = existing.get("pid").and_then(Value::as_i64);
    if existing_pid != Some(current_pid()) {
        return;
    }
    let existing_start = existing.get("start_time").and_then(Value::as_i64);
    if existing_start != get_process_start_time(current_pid()) {
        return;
    }
    let _ = fs::remove_file(&lock_path);
}

/// Remove scoped lock files in the lock directory.
///
/// When `owner_pid` is provided, only lock records belonging to that gateway
/// process are removed; `owner_start_time` further narrows the match to protect
/// against PID reuse. When `owner_pid` is `None`, removes every scoped lock.
/// Returns the number of lock files removed.
pub fn release_all_scoped_locks(
    owner_pid: Option<i64>,
    owner_start_time: Option<i64>,
) -> usize {
    let lock_dir = get_lock_dir();
    let mut removed = 0;
    if !lock_dir.exists() {
        return removed;
    }
    let entries = match fs::read_dir(&lock_dir) {
        Ok(e) => e,
        Err(_) => return removed,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lock") {
            continue;
        }
        if let Some(owner) = owner_pid {
            let record = match read_json_file(&path) {
                Some(r) if r.is_object() => r,
                _ => continue,
            };
            let record_pid = match record.get("pid").and_then(coerce_pid) {
                Some(p) => p,
                None => continue,
            };
            if record_pid != owner {
                continue;
            }
            if let Some(ost) = owner_start_time {
                let rec_start = record.get("start_time").and_then(Value::as_i64);
                if rec_start != Some(ost) {
                    continue;
                }
            }
        }
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

// ── Takeover / planned-stop markers ───────────────────────────────────

/// Return true when an ISO timestamp is older than `ttl_s` seconds (or
/// unparseable, matching the Python original which treats bad input as stale).
fn marker_is_stale(written_at: &str, ttl_s: i64) -> bool {
    match DateTime::parse_from_rfc3339(written_at) {
        Ok(dt) => {
            let age = Utc::now().signed_duration_since(dt.with_timezone(&Utc));
            age.num_seconds() > ttl_s
        }
        Err(_) => true,
    }
}

/// Check & unlink a PID marker if it names the current process.
fn consume_pid_marker_for_self(
    path: &Path,
    pid_field: &str,
    start_time_field: &str,
    ttl_s: i64,
) -> bool {
    let record = match read_json_file(path) {
        Some(r) => r,
        None => return false,
    };

    let target_pid = record.get(pid_field).and_then(coerce_pid);
    let target_start_time = record.get(start_time_field).cloned();
    let written_at = record
        .get("written_at")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let target_pid = match target_pid {
        Some(p) => p,
        None => {
            let _ = fs::remove_file(path);
            return false;
        }
    };

    if marker_is_stale(&written_at, ttl_s) {
        let _ = fs::remove_file(path);
        return false;
    }

    let our_pid = current_pid();
    let our_start_time = get_process_start_time(our_pid);
    let target_start_i64 = target_start_time.as_ref().and_then(Value::as_i64);
    let matches = target_pid == our_pid
        && target_start_i64.is_some()
        && our_start_time.is_some()
        && target_start_i64 == our_start_time;

    let _ = fs::remove_file(path);
    matches
}

/// Record that `target_pid` is being replaced by the current process.
///
/// Returns true on successful write, false on any failure (best-effort signal).
pub fn write_takeover_marker(target_pid: i64) -> bool {
    let target_start_time = get_process_start_time(target_pid);
    let record = json!({
        "target_pid": target_pid,
        "target_start_time": target_start_time,
        "replacer_pid": current_pid(),
        "written_at": utc_now_iso(),
    });
    write_json_file(&get_takeover_marker_path(), &record).is_ok()
}

/// Check & unlink the takeover marker if it names the current process.
///
/// Returns true only when a valid (non-stale) marker names this PID + start
/// time, indicating the current SIGTERM is a planned `--replace` takeover.
pub fn consume_takeover_marker_for_self() -> bool {
    consume_pid_marker_for_self(
        &get_takeover_marker_path(),
        "target_pid",
        "target_start_time",
        TAKEOVER_MARKER_TTL_S,
    )
}

/// Remove the takeover marker unconditionally. Safe to call repeatedly.
pub fn clear_takeover_marker() {
    let _ = fs::remove_file(get_takeover_marker_path());
}

/// Record that `target_pid` is being stopped intentionally.
pub fn write_planned_stop_marker(target_pid: i64) -> bool {
    let target_start_time = get_process_start_time(target_pid);
    let record = json!({
        "target_pid": target_pid,
        "target_start_time": target_start_time,
        "stopper_pid": current_pid(),
        "written_at": utc_now_iso(),
    });
    write_json_file(&get_planned_stop_marker_path(), &record).is_ok()
}

/// Return true when the current process is being intentionally stopped.
pub fn consume_planned_stop_marker_for_self() -> bool {
    consume_pid_marker_for_self(
        &get_planned_stop_marker_path(),
        "target_pid",
        "target_start_time",
        PLANNED_STOP_MARKER_TTL_S,
    )
}

/// Remove the planned-stop marker unconditionally.
pub fn clear_planned_stop_marker() {
    let _ = fs::remove_file(get_planned_stop_marker_path());
}

// ── Running-PID detection ─────────────────────────────────────────────

/// Return the PID of a running gateway instance, or `None`.
///
/// Checks the PID file, verifies the runtime lock is held, and confirms the
/// process is actually alive and looks like a gateway. Cleans up stale PID
/// files automatically when `cleanup_stale` is true.
pub fn get_running_pid(pid_path: Option<&Path>, cleanup_stale: bool) -> Option<i64> {
    let default_pid_path = get_pid_path();
    let resolved_pid_path = pid_path.map(|p| p.to_path_buf()).unwrap_or(default_pid_path);
    let resolved_lock_path = get_gateway_lock_path(Some(&resolved_pid_path));

    let lock_active = is_gateway_runtime_lock_active(Some(&resolved_lock_path));
    if !lock_active {
        cleanup_invalid_pid_path(&resolved_pid_path, cleanup_stale);
        return None;
    }

    let primary_record = read_pid_record(&resolved_pid_path);
    let fallback_record = read_gateway_lock_record(&resolved_lock_path);

    for record in [&primary_record, &fallback_record] {
        let pid = match pid_from_record(record) {
            Some(p) => p,
            None => continue,
        };
        let record_val = record.as_ref().unwrap();

        match pid_exists(pid) {
            Err(KillError::NoSuchProcess) | Err(KillError::Other) => continue,
            Err(KillError::PermissionDenied) => {
                // Process exists but belongs to another user/scope. With the
                // runtime lock still held, prefer keeping it visible.
                if record_looks_like_gateway(record_val) {
                    return Some(pid);
                }
                continue;
            }
            Ok(()) => {}
        }

        let recorded_start = record_val.get("start_time").and_then(Value::as_i64);
        let current_start = get_process_start_time(pid);
        if let (Some(rs), Some(cs)) = (recorded_start, current_start) {
            if cs != rs {
                continue;
            }
        }

        if looks_like_gateway_process(pid) || record_looks_like_gateway(record_val) {
            return Some(pid);
        }
    }

    cleanup_invalid_pid_path(&resolved_pid_path, cleanup_stale);
    None
}

/// Check if the gateway daemon is currently running.
pub fn is_gateway_running(pid_path: Option<&Path>, cleanup_stale: bool) -> bool {
    get_running_pid(pid_path, cleanup_stale).is_some()
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests that mutate HERMES_HOME to avoid env-var races.
    static ENV_GUARD: StdMutex<()> = StdMutex::new(());

    fn temp_home() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hermes-gwstatus-test-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn with_name_replaces_final_component() {
        let p = Path::new("/a/b/gateway.pid");
        assert_eq!(with_name(p, "gateway.lock"), PathBuf::from("/a/b/gateway.lock"));
    }

    #[test]
    fn scope_hash_is_16_hex_chars_of_sha256() {
        let h = scope_hash("hello");
        assert_eq!(h.len(), 16);
        // SHA256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e...
        assert_eq!(h, "2cf24dba5fb0a30e");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn record_looks_like_gateway_validates_kind_and_argv() {
        let good = json!({
            "kind": GATEWAY_KIND,
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
        });
        assert!(record_looks_like_gateway(&good));

        let wrong_kind = json!({
            "kind": "other",
            "argv": ["hermes", "gateway"],
        });
        assert!(!record_looks_like_gateway(&wrong_kind));

        let no_match = json!({
            "kind": GATEWAY_KIND,
            "argv": ["hermes", "tui"],
        });
        assert!(!record_looks_like_gateway(&no_match));

        let empty_argv = json!({ "kind": GATEWAY_KIND, "argv": [] });
        assert!(!record_looks_like_gateway(&empty_argv));
    }

    #[test]
    fn record_looks_like_gateway_joins_argv_for_split_pattern() {
        let r = json!({
            "kind": GATEWAY_KIND,
            "argv": ["hermes", "gateway", "--replace"],
        });
        // "hermes gateway" appears across joined argv.
        assert!(record_looks_like_gateway(&r));
    }

    #[test]
    fn coerce_pid_handles_int_string_and_float() {
        assert_eq!(coerce_pid(&json!(123)), Some(123));
        assert_eq!(coerce_pid(&json!("456")), Some(456));
        assert_eq!(coerce_pid(&json!(789.0)), Some(789));
        assert_eq!(coerce_pid(&json!("notanint")), None);
        assert_eq!(coerce_pid(&json!(null)), None);
    }

    #[test]
    fn read_pid_record_parses_object_int_and_raw() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();

        let obj_path = dir.join("obj.pid");
        fs::write(&obj_path, r#"{"pid": 42, "kind": "hermes-gateway"}"#).unwrap();
        let rec = read_pid_record(&obj_path).unwrap();
        assert_eq!(rec.get("pid").and_then(Value::as_i64), Some(42));

        let int_path = dir.join("int.pid");
        fs::write(&int_path, "777").unwrap();
        let rec = read_pid_record(&int_path).unwrap();
        assert_eq!(rec.get("pid").and_then(Value::as_i64), Some(777));

        let bare_int_json = dir.join("bare.pid");
        fs::write(&bare_int_json, " 555 ").unwrap();
        let rec = read_pid_record(&bare_int_json).unwrap();
        assert_eq!(rec.get("pid").and_then(Value::as_i64), Some(555));

        let empty_path = dir.join("empty.pid");
        fs::write(&empty_path, "   ").unwrap();
        assert!(read_pid_record(&empty_path).is_none());

        let missing = dir.join("missing.pid");
        assert!(read_pid_record(&missing).is_none());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_json_file_rejects_non_objects() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();

        let arr = dir.join("arr.json");
        fs::write(&arr, "[1,2,3]").unwrap();
        assert!(read_json_file(&arr).is_none());

        let obj = dir.join("obj.json");
        fs::write(&obj, r#"{"a":1}"#).unwrap();
        assert!(read_json_file(&obj).is_some());

        let bad = dir.join("bad.json");
        fs::write(&bad, "{not json").unwrap();
        assert!(read_json_file(&bad).is_none());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn marker_is_stale_logic() {
        // Unparseable -> stale.
        assert!(marker_is_stale("not-a-date", 60));
        // Fresh -> not stale.
        let now = utc_now_iso();
        assert!(!marker_is_stale(&now, 60));
        // Old -> stale.
        let old = (Utc::now() - chrono::Duration::seconds(120)).to_rfc3339();
        assert!(marker_is_stale(&old, 60));
    }

    #[test]
    fn write_pid_file_is_exclusive() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        let _ = fs::remove_file(get_pid_path());
        write_pid_file().unwrap();
        // Second write must fail with AlreadyExists.
        match write_pid_file() {
            Err(WritePidError::AlreadyExists) => {}
            other => panic!("expected AlreadyExists, got {other:?}"),
        }

        // The written record should be our PID.
        let rec = read_pid_record(&get_pid_path()).unwrap();
        assert_eq!(rec.get("pid").and_then(Value::as_i64), Some(current_pid()));

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn remove_pid_file_only_removes_own() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        // Foreign PID file should be left alone.
        let path = get_pid_path();
        fs::write(&path, r#"{"pid": 999999, "kind": "hermes-gateway"}"#).unwrap();
        remove_pid_file();
        assert!(path.exists(), "foreign PID file should not be removed");

        // Own PID file should be removed.
        fs::write(
            &path,
            serde_json::to_string(&build_pid_record()).unwrap(),
        )
        .unwrap();
        remove_pid_file();
        assert!(!path.exists(), "own PID file should be removed");

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runtime_status_roundtrip_and_platform_merge() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }
        let _ = fs::remove_file(get_runtime_status_path());

        write_runtime_status(&RuntimeStatusUpdate {
            gateway_state: Some(json!("running")),
            active_agents: Some(3),
            ..Default::default()
        })
        .unwrap();

        let status = read_runtime_status().unwrap();
        assert_eq!(status.get("gateway_state").and_then(Value::as_str), Some("running"));
        assert_eq!(status.get("active_agents").and_then(Value::as_i64), Some(3));
        assert_eq!(status.get("kind").and_then(Value::as_str), Some(GATEWAY_KIND));

        // active_agents is clamped to >= 0.
        write_runtime_status(&RuntimeStatusUpdate {
            active_agents: Some(-5),
            ..Default::default()
        })
        .unwrap();
        let status = read_runtime_status().unwrap();
        assert_eq!(status.get("active_agents").and_then(Value::as_i64), Some(0));
        // gateway_state preserved across the partial update.
        assert_eq!(status.get("gateway_state").and_then(Value::as_str), Some("running"));

        // Platform nested update.
        write_runtime_status(&RuntimeStatusUpdate {
            platform: Some("telegram".into()),
            platform_state: Some(json!("connected")),
            error_code: Some(json!("E1")),
            ..Default::default()
        })
        .unwrap();
        let status = read_runtime_status().unwrap();
        let plat = status
            .get("platforms")
            .and_then(|p| p.get("telegram"))
            .unwrap();
        assert_eq!(plat.get("state").and_then(Value::as_str), Some("connected"));
        assert_eq!(plat.get("error_code").and_then(Value::as_str), Some("E1"));
        assert!(plat.get("updated_at").is_some());

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scoped_lock_acquire_release_cycle() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_GATEWAY_LOCK_DIR", &dir); }

        let (acquired, prev) = acquire_scoped_lock("telegram", "bot-token-abc", None);
        assert!(acquired);
        assert!(prev.is_none());

        // Re-acquire by same process/start_time refreshes and returns existing.
        let (acquired2, prev2) = acquire_scoped_lock("telegram", "bot-token-abc", None);
        assert!(acquired2);
        assert!(prev2.is_some());

        let lock_path = get_scope_lock_path("telegram", "bot-token-abc");
        assert!(lock_path.exists());

        release_scoped_lock("telegram", "bot-token-abc");
        // Release only succeeds if start_time matched; on /proc-less platforms
        // both sides are None so equality holds and the file is removed. On
        // Linux they also match (same live process). Either way, removed.
        assert!(!lock_path.exists());

        unsafe { std::env::remove_var("HERMES_GATEWAY_LOCK_DIR"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn release_all_scoped_locks_filters_by_owner() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_GATEWAY_LOCK_DIR", &dir); }

        // Lock owned by our PID.
        acquire_scoped_lock("slack", "id-1", None);
        // A foreign lock file written directly.
        let foreign = get_scope_lock_path("slack", "id-2");
        fs::write(
            &foreign,
            r#"{"pid": 424242, "kind": "hermes-gateway"}"#,
        )
        .unwrap();

        // Remove only those owned by a non-existent PID -> removes nothing of ours.
        let removed = release_all_scoped_locks(Some(424242), None);
        assert_eq!(removed, 1);
        assert!(!foreign.exists());
        assert!(get_scope_lock_path("slack", "id-1").exists());

        // Remove all.
        let removed_all = release_all_scoped_locks(None, None);
        assert!(removed_all >= 1);

        unsafe { std::env::remove_var("HERMES_GATEWAY_LOCK_DIR"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runtime_lock_acquire_and_active_detection() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        // Ensure clean global state.
        release_gateway_runtime_lock();

        assert!(acquire_gateway_runtime_lock());
        // Idempotent.
        assert!(acquire_gateway_runtime_lock());
        // We hold it -> active for the default path.
        assert!(is_gateway_runtime_lock_active(None));

        release_gateway_runtime_lock();
        // After release, no live owner -> inactive.
        assert!(!is_gateway_runtime_lock_active(None));

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn takeover_marker_consume_for_self() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }

        // Marker naming a foreign PID should not match and is unlinked.
        let path = get_takeover_marker_path();
        fs::write(
            &path,
            serde_json::to_string(&json!({
                "target_pid": 123456,
                "target_start_time": 99,
                "replacer_pid": 1,
                "written_at": utc_now_iso(),
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(!consume_takeover_marker_for_self());
        assert!(!path.exists(), "marker unlinked after consume");

        // Stale marker (old timestamp) -> false + unlinked.
        fs::write(
            &path,
            serde_json::to_string(&json!({
                "target_pid": current_pid(),
                "target_start_time": get_process_start_time(current_pid()),
                "written_at": (Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(!consume_takeover_marker_for_self());
        assert!(!path.exists());

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clear_markers_is_safe_when_absent() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }
        clear_takeover_marker();
        clear_planned_stop_marker();
        clear_takeover_marker();
        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_running_pid_none_without_lock() {
        let _g = ENV_GUARD.lock().unwrap();
        let dir = temp_home();
        unsafe { std::env::set_var("HERMES_HOME", &dir); }
        release_gateway_runtime_lock();

        // No lock file -> not active -> None, and PID file cleaned up.
        let pid_path = get_pid_path();
        fs::write(&pid_path, r#"{"pid": 999999}"#).unwrap();
        assert_eq!(get_running_pid(None, true), None);
        assert!(!pid_path.exists(), "stale PID file cleaned up");

        unsafe { std::env::remove_var("HERMES_HOME"); }
        fs::remove_dir_all(&dir).ok();
    }
}
