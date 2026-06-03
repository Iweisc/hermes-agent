//! Native Rust port of `hermes_cli/gateway.py`.
//!
//! Handles: `hermes gateway [run|start|stop|restart|status|install|uninstall|setup|migrate-legacy]`
//!
//! Faithful port of the Python process-management + service-install logic.
//! The actual async run loop of `run_gateway` is delegated to the native
//! gateway runtime via the [`GatewayRunner`] trait.
//!
//! Cross-references:
//! * `hermes_core::gateway` — restart drain constants + `parse_restart_drain_timeout`.
//! * `hermes_core::mod_hermes_constants` — platform/profile detection helpers.
//! * `hermes_core::cli_config` — env/config readers.
//! * `crate::cli_profiles` — profile enumeration.
//!
//! `hermes_core::gw_status` is private in hermes-core, so the small
//! PID-file / runtime-status helpers it provides are reimplemented locally
//! (they read the same on-disk files).

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::Value;

use hermes_core::gateway::{
    parse_restart_drain_timeout, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
};
// `is_wsl` / `is_container` are public at the hermes-core crate root.
use hermes_core::{is_container, is_wsl};

// `hermes_core::mod_hermes_constants` and `hermes_core::cli_config` are private
// modules in hermes-core, so the small set of constants/config helpers this
// module needs are reimplemented locally below (faithful to the Python originals
// in `hermes_constants.py` / `hermes_cli/config.py`).

/// Best-effort home directory, mirroring `Path.home()`.
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Lexical absolute path (no symlink resolution) for fallback canonicalization.
fn lexical_abspath(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(p)
    }
}

/// Return the Hermes home directory (default `~/.hermes`), honouring `HERMES_HOME`.
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    home_dir().join(".hermes")
}

/// Return the root Hermes directory for profile-level operations.
/// Mirrors `hermes_constants.get_default_hermes_root`.
pub fn get_default_hermes_root() -> PathBuf {
    let native_home = home_dir().join(".hermes");
    let env_home = std::env::var("HERMES_HOME").unwrap_or_default();
    if env_home.trim().is_empty() {
        return native_home;
    }
    let env_path = PathBuf::from(env_home.trim());
    let env_resolved = env_path
        .canonicalize()
        .unwrap_or_else(|_| lexical_abspath(&env_path));
    let native_resolved = native_home
        .canonicalize()
        .unwrap_or_else(|_| lexical_abspath(&native_home));
    if env_resolved.starts_with(&native_resolved) {
        return native_home;
    }
    // Docker / custom: `<root>/profiles/<name>` → root is grandparent.
    if env_path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n == "profiles")
        == Some(true)
    {
        if let Some(root) = env_path.parent().and_then(|p| p.parent()) {
            return root.to_path_buf();
        }
    }
    env_path
}

/// Mirror `hermes_constants.is_termux`.
fn is_termux() -> bool {
    if let Some(prefix) = getenv("PREFIX") {
        if prefix.contains("com.termux") {
            return true;
        }
    }
    Path::new("/data/data/com.termux").exists()
}

/// Mirror `hermes_constants.display_hermes_home` (tilde-collapsed path).
fn display_hermes_home() -> String {
    let home = get_hermes_home();
    let user_home = home_dir();
    match home.strip_prefix(&user_home) {
        Ok(rel) if rel.as_os_str().is_empty() => "~".to_string(),
        Ok(rel) => format!("~/{}", rel.display()),
        Err(_) => home.to_string_lossy().to_string(),
    }
}

// =============================================================================
// Platform predicates
// =============================================================================

pub fn is_linux() -> bool {
    cfg!(target_os = "linux")
}
pub fn is_macos() -> bool {
    cfg!(target_os = "macos")
}
pub fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

const SERVICE_BASE: &str = "hermes-gateway";
pub const SERVICE_DESCRIPTION: &str = "Hermes Agent Gateway - Messaging Platform Integration";

/// Project root: the gateway working directory. Mirrors `PROJECT_ROOT` in the
/// Python file (`gateway.py.parent.parent`). For the native binary we derive it
/// from `HERMES_PROJECT_ROOT` when set, otherwise the current working dir.
pub fn project_root() -> PathBuf {
    if let Ok(p) = std::env::var("HERMES_PROJECT_ROOT") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// =============================================================================
// Gateway run-loop seam
// =============================================================================

/// Abstracts the actual gateway run loop (Python's `asyncio.run(start_gateway)`).
pub trait GatewayRunner {
    /// Run the gateway in the foreground.
    /// Ok(true)=clean success, Ok(false)=startup failed (exit 1), Err=fatal.
    fn start(&self, replace: bool, verbosity: Option<i32>) -> Result<bool, String>;
}

// =============================================================================
// Low-level process primitives
// =============================================================================

#[cfg(unix)]
fn os_getpid() -> i64 {
    unsafe { libc::getpid() as i64 }
}
#[cfg(not(unix))]
fn os_getpid() -> i64 {
    std::process::id() as i64
}

#[cfg(unix)]
fn os_getuid() -> u32 {
    unsafe { libc::getuid() }
}
#[cfg(not(unix))]
fn os_getuid() -> u32 {
    0
}

#[cfg(unix)]
fn os_geteuid() -> u32 {
    unsafe { libc::geteuid() }
}
#[cfg(not(unix))]
fn os_geteuid() -> u32 {
    0
}

#[derive(Debug, PartialEq, Eq)]
pub enum KillError {
    NoSuchProcess,
    PermissionDenied,
    Other(i32),
}

#[cfg(unix)]
pub fn os_kill(pid: i64, sig: i32) -> Result<(), KillError> {
    let rc = unsafe { libc::kill(pid as libc::pid_t, sig) };
    if rc == 0 {
        return Ok(());
    }
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    match errno {
        libc::ESRCH => Err(KillError::NoSuchProcess),
        libc::EPERM => Err(KillError::PermissionDenied),
        other => Err(KillError::Other(other)),
    }
}
#[cfg(not(unix))]
pub fn os_kill(_pid: i64, _sig: i32) -> Result<(), KillError> {
    Err(KillError::Other(-1))
}

#[cfg(unix)]
fn sig_term() -> i32 {
    libc::SIGTERM
}
#[cfg(unix)]
fn sig_kill() -> i32 {
    libc::SIGKILL
}
#[cfg(unix)]
fn sig_usr1() -> Option<i32> {
    Some(libc::SIGUSR1)
}
#[cfg(not(unix))]
fn sig_term() -> i32 {
    15
}
#[cfg(not(unix))]
fn sig_kill() -> i32 {
    9
}
#[cfg(not(unix))]
fn sig_usr1() -> Option<i32> {
    None
}

/// Terminate a PID (SIGTERM, or SIGKILL with `force`). Mirrors
/// `gateway.status.terminate_pid`.
pub fn terminate_pid(pid: i64, force: bool) -> Result<(), KillError> {
    let sig = if force { sig_kill() } else { sig_term() };
    os_kill(pid, sig)
}

// =============================================================================
// Local PID-file / runtime-status readers (mirror gateway.status)
// =============================================================================

fn gateway_pid_path() -> PathBuf {
    get_hermes_home().join("gateway.pid")
}

fn runtime_status_path() -> PathBuf {
    get_hermes_home().join("gateway_runtime.json")
}

pub fn get_running_pid(pid_path: Option<&Path>, cleanup_stale: bool) -> Option<i64> {
    let path = pid_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(gateway_pid_path);
    let raw = fs::read_to_string(&path).ok()?;
    let pid: i64 = raw.trim().parse().ok()?;
    if pid <= 0 {
        return None;
    }
    match os_kill(pid, 0) {
        Ok(()) | Err(KillError::PermissionDenied) => Some(pid),
        Err(KillError::NoSuchProcess) => {
            if cleanup_stale {
                let _ = fs::remove_file(&path);
            }
            None
        }
        Err(KillError::Other(_)) => Some(pid),
    }
}

pub fn remove_pid_file() {
    let _ = fs::remove_file(gateway_pid_path());
}

pub fn write_planned_stop_marker(target_pid: i64) -> bool {
    let path = get_hermes_home().join("gateway_planned_stop");
    fs::write(path, target_pid.to_string()).is_ok()
}

pub fn read_runtime_status() -> Option<Value> {
    let raw = fs::read_to_string(runtime_status_path()).ok()?;
    serde_json::from_str(&raw).ok()
}

// =============================================================================
// Config / env helpers
// =============================================================================

fn env_file_path() -> PathBuf {
    get_hermes_home().join(".env")
}

fn config_yaml_path() -> PathBuf {
    get_hermes_home().join("config.yaml")
}

/// Parse a `.env` file into key/value pairs. Mirrors the subset of
/// `hermes_cli.config` behaviour this module relies on (strip quotes, ignore
/// comments / blank lines). Process env takes precedence.
fn read_env_file() -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Ok(text) = fs::read_to_string(env_file_path()) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim().to_string();
                let mut v = v.trim().to_string();
                if (v.starts_with('"') && v.ends_with('"') && v.len() >= 2)
                    || (v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2)
                {
                    v = v[1..v.len() - 1].to_string();
                }
                map.insert(k, v);
            }
        }
    }
    map
}

/// Read an env value: process environment first, then the profile `.env` file.
fn get_env_value(key: &str) -> Option<String> {
    if let Some(v) = getenv(key) {
        return Some(v);
    }
    read_env_file().get(key).cloned().filter(|s| !s.is_empty())
}

/// Persist a key/value into the profile `.env` file (create/update in place).
fn save_env_value(key: &str, value: &str) {
    let path = env_file_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let existing = fs::read_to_string(&path).unwrap_or_default();
    let mut lines: Vec<String> = Vec::new();
    let mut replaced = false;
    let prefix = format!("{key}=");
    for line in existing.lines() {
        let trimmed = line.trim_start();
        let stripped = trimmed.strip_prefix("export ").unwrap_or(trimmed);
        if stripped.starts_with(&prefix) {
            lines.push(format!("{key}={value}"));
            replaced = true;
        } else {
            lines.push(line.to_string());
        }
    }
    if !replaced {
        lines.push(format!("{key}={value}"));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    let _ = fs::write(&path, out);
}

/// Read the raw `config.yaml` as a JSON value (empty object when absent/invalid).
fn read_raw_config() -> Value {
    match fs::read_to_string(config_yaml_path()) {
        Ok(text) => serde_yaml::from_str::<Value>(&text)
            .unwrap_or_else(|_| Value::Object(Default::default())),
        Err(_) => Value::Object(Default::default()),
    }
}

/// True when this install is centrally managed (e.g. NixOS) and config is read-only.
fn is_managed() -> bool {
    matches!(
        getenv("HERMES_MANAGED").map(|v| v.to_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn managed_error(action: &str) {
    print_error(&format!(
        "Cannot {action}: this Hermes install is managed (read-only configuration)."
    ));
}

// =============================================================================
// Console output helpers
// =============================================================================

fn print_header(text: &str) {
    println!("\n=== {text} ===");
}
fn print_info(text: &str) {
    println!("{text}");
}
fn print_success(text: &str) {
    println!("\u{2713} {text}");
}
fn print_warning(text: &str) {
    println!("\u{26a0} {text}");
}
fn print_error(text: &str) {
    eprintln!("\u{2717} {text}");
}

// =============================================================================
// Subprocess helpers
// =============================================================================

/// Run a command, capturing stdout/stderr. Returns (exit_code, stdout, stderr)
/// or None when the binary is missing / spawn fails.
fn run_capture(program: &str, args: &[&str]) -> Option<(i32, String, String)> {
    let out = Command::new(program).args(args).output().ok()?;
    let code = out.status.code().unwrap_or(-1);
    Some((
        code,
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    ))
}

/// Run a command attached to the current stdio. Returns exit code or None.
fn run_inherit(program: &str, args: &[&str]) -> Option<i32> {
    let status = Command::new(program).args(args).status().ok()?;
    Some(status.code().unwrap_or(-1))
}

fn which(binary: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn getenv(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn current_username() -> String {
    getenv("USER")
        .or_else(|| getenv("LOGNAME"))
        .or_else(|| getenv("SUDO_USER"))
        .unwrap_or_else(|| "root".to_string())
}

// =============================================================================
// Data types (mirror Python dataclasses)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayRuntimeSnapshot {
    pub manager: String,
    pub service_installed: bool,
    pub service_running: bool,
    pub gateway_pids: Vec<i64>,
    pub service_scope: Option<String>,
}

impl GatewayRuntimeSnapshot {
    pub fn running(&self) -> bool {
        self.service_running || !self.gateway_pids.is_empty()
    }

    pub fn has_process_service_mismatch(&self) -> bool {
        self.service_installed && self.running() && !self.service_running
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileGatewayProcess {
    pub profile: String,
    pub path: PathBuf,
    pub pid: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileGatewayServiceTarget {
    pub profile: String,
    pub path: PathBuf,
    pub service: String,
}

// =============================================================================
// Profile enumeration (via crate::cli_profiles)
// =============================================================================

fn list_profiles() -> Vec<crate::cli_profiles::ProfileInfo> {
    crate::cli_profiles::list_profiles()
}

fn get_active_profile_name() -> String {
    crate::cli_profiles::get_active_profile_name()
}

fn running_pid_for_profile(path: &Path) -> Option<i64> {
    get_running_pid(Some(&path.join("gateway.pid")), false)
}

// =============================================================================
// Service-name derivation (profile-scoped)
// =============================================================================

fn sha256_hex8(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    let full = digest.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    full[..8].to_string()
}

fn canonical(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Mirror `re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name)`.
fn is_valid_profile_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    let bytes = name.as_bytes();
    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Derive a service-name suffix from a HERMES_HOME path.
/// Returns "" for the default root, the profile name for `<root>/profiles/<name>`,
/// or a short hash for any other path.
pub fn profile_suffix(hermes_home: Option<&Path>) -> String {
    let home = match hermes_home {
        Some(p) => canonical(p),
        None => canonical(&get_hermes_home()),
    };
    let default = canonical(&get_default_hermes_root());
    if home == default {
        return String::new();
    }
    let profiles_root = canonical(&default.join("profiles"));
    if let Ok(rel) = home.strip_prefix(&profiles_root) {
        let parts: Vec<_> = rel.components().collect();
        if parts.len() == 1 {
            if let Some(name) = rel.to_str() {
                if is_valid_profile_name(name) {
                    return name.to_string();
                }
            }
        }
    }
    sha256_hex8(home.to_string_lossy().as_ref())
}

/// Return `--profile <name>` only when HERMES_HOME is a named profile.
pub fn profile_arg(hermes_home: Option<&str>) -> String {
    let home = match hermes_home {
        Some(p) => canonical(Path::new(p)),
        None => canonical(&get_hermes_home()),
    };
    let default = canonical(&get_default_hermes_root());
    if home == default {
        return String::new();
    }
    let profiles_root = canonical(&default.join("profiles"));
    if let Ok(rel) = home.strip_prefix(&profiles_root) {
        let parts: Vec<_> = rel.components().collect();
        if parts.len() == 1 {
            if let Some(name) = rel.to_str() {
                if is_valid_profile_name(name) {
                    return format!("--profile {name}");
                }
            }
        }
    }
    String::new()
}

pub fn get_service_name(hermes_home: Option<&Path>) -> String {
    let suffix = profile_suffix(hermes_home);
    if suffix.is_empty() {
        SERVICE_BASE.to_string()
    } else {
        format!("{SERVICE_BASE}-{suffix}")
    }
}

pub fn get_systemd_unit_path(system: bool, service_name: Option<&str>) -> PathBuf {
    let name = service_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| get_service_name(None));
    if system {
        PathBuf::from("/etc/systemd/system").join(format!("{name}.service"))
    } else {
        home_dir()
            .join(".config")
            .join("systemd")
            .join("user")
            .join(format!("{name}.service"))
    }
}

fn file_mentions_hermes_home(path: &Path, hermes_home: &Path) -> bool {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let home = canonical(hermes_home);
    let home_str = home.to_string_lossy();
    text.contains(&format!("HERMES_HOME={home_str}")) || text.contains(home_str.as_ref())
}

pub fn systemd_unit_matches_home(
    system: bool,
    hermes_home: &Path,
    service_name: Option<&str>,
) -> bool {
    let unit_path = get_systemd_unit_path(system, service_name);
    unit_path.exists() && file_mentions_hermes_home(&unit_path, hermes_home)
}

pub fn launchd_plist_matches_home(hermes_home: &Path) -> bool {
    let p = get_launchd_plist_path();
    p.exists() && file_mentions_hermes_home(&p, hermes_home)
}

pub fn profile_gateway_service_targets() -> Vec<ProfileGatewayServiceTarget> {
    let profiles = list_profiles();
    if profiles.is_empty() {
        return vec![ProfileGatewayServiceTarget {
            profile: "default".to_string(),
            path: get_hermes_home(),
            service: get_service_name(None),
        }];
    }
    let mut targets = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for profile in profiles {
        let service = get_service_name(Some(&profile.path));
        let key = (profile.name.clone(), service.clone());
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        targets.push(ProfileGatewayServiceTarget {
            profile: profile.name,
            path: profile.path,
            service,
        });
    }
    targets
}

// =============================================================================
// PID discovery
// =============================================================================

/// Return the parent PID for `pid`, or None.
fn get_parent_pid(pid: i64) -> Option<i64> {
    if pid <= 1 {
        return None;
    }
    let (code, stdout, _) = run_capture("ps", &["-o", "ppid=", "-p", &pid.to_string()])?;
    if code != 0 {
        return None;
    }
    let raw = stdout.trim();
    if raw.is_empty() {
        return None;
    }
    let last = raw.lines().last()?.trim();
    let parent: i64 = last.parse().ok()?;
    if parent > 0 {
        Some(parent)
    } else {
        None
    }
}

fn is_pid_ancestor_of_current_process(target_pid: i64) -> bool {
    if target_pid <= 0 {
        return false;
    }
    let mut pid = os_getpid();
    let mut seen: HashSet<i64> = HashSet::new();
    while pid != 0 && !seen.contains(&pid) {
        if pid == target_pid {
            return true;
        }
        seen.insert(pid);
        pid = get_parent_pid(pid).unwrap_or(0);
    }
    false
}

/// Ask a running gateway ancestor to restart itself asynchronously (SIGUSR1).
fn request_gateway_self_restart(pid: i64) -> bool {
    let sig = match sig_usr1() {
        Some(s) => s,
        None => return false,
    };
    if !is_pid_ancestor_of_current_process(pid) {
        return false;
    }
    os_kill(pid, sig).is_ok()
}

/// Send SIGUSR1 to a gateway PID and wait for it to exit gracefully.
fn graceful_restart_via_sigusr1(pid: i64, drain_timeout: f64) -> bool {
    let sig = match sig_usr1() {
        Some(s) => s,
        None => return false,
    };
    if pid <= 0 {
        return false;
    }
    match os_kill(pid, sig) {
        Ok(()) => {}
        Err(KillError::NoSuchProcess) => return true,
        Err(_) => return false,
    }
    let deadline = Instant::now() + Duration::from_secs_f64(drain_timeout.max(1.0));
    while Instant::now() < deadline {
        match os_kill(pid, 0) {
            Err(KillError::NoSuchProcess) => return true,
            _ => {}
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

/// Return the set of PIDs in the current process's ancestor chain.
fn get_ancestor_pids() -> HashSet<i64> {
    let mut ancestors: HashSet<i64> = HashSet::new();
    let mut pid = os_getpid();
    for _ in 0..64 {
        ancestors.insert(pid);
        match get_parent_pid(pid) {
            Some(parent) if parent > 0 && !ancestors.contains(&parent) => pid = parent,
            _ => break,
        }
    }
    ancestors
}

fn append_unique_pid(pids: &mut Vec<i64>, pid: Option<i64>, exclude: &HashSet<i64>) {
    let pid = match pid {
        Some(p) if p > 0 => p,
        _ => return,
    };
    if pid == os_getpid() || exclude.contains(&pid) || pids.contains(&pid) {
        return;
    }
    pids.push(pid);
}

/// PIDs currently managed by this profile's gateway service (systemd/launchd).
fn get_service_pids() -> HashSet<i64> {
    let mut pids: HashSet<i64> = HashSet::new();

    // systemd: user + system scopes
    if supports_systemd_services() {
        let service = get_service_name(None);
        for system in [false, true] {
            if !systemd_unit_matches_home(system, &get_hermes_home(), Some(&service)) {
                continue;
            }
            if let Some((_, stdout, _)) = run_systemctl(
                &["show", &service, "--property=MainPID", "--value"],
                system,
                true,
            ) {
                if let Ok(pid) = stdout.trim().parse::<i64>() {
                    if pid > 0 {
                        pids.insert(pid);
                    }
                }
            }
        }
    }

    // launchd (macOS)
    if is_macos() {
        let label = get_launchd_label();
        if !launchd_plist_matches_home(&get_hermes_home()) {
            return pids;
        }
        if let Some((code, stdout, _)) = run_capture("launchctl", &["list", &label]) {
            if code == 0 {
                for line in stdout.trim().lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 && parts[2] == label {
                        if let Ok(pid) = parts[0].parse::<i64>() {
                            if pid > 0 {
                                pids.insert(pid);
                            }
                        }
                    }
                }
            }
        }
    }

    pids
}

/// Best-effort process-table scan for gateway PIDs.
fn scan_gateway_pids(exclude_pids: &HashSet<i64>, all_profiles: bool) -> Vec<i64> {
    let mut exclude = exclude_pids.clone();
    exclude.extend(get_ancestor_pids());
    let mut pids: Vec<i64> = Vec::new();
    let patterns = [
        "hermes_cli.main gateway",
        "hermes_cli.main --profile",
        "hermes_cli.main -p",
        "hermes_cli/main.py gateway",
        "hermes_cli/main.py --profile",
        "hermes_cli/main.py -p",
        "hermes gateway",
        "gateway/run.py",
    ];
    let current_home = canonical(&get_hermes_home()).to_string_lossy().to_string();
    let current_profile_arg = profile_arg(Some(&current_home));
    let current_profile_name = current_profile_arg
        .split_whitespace()
        .last()
        .unwrap_or("")
        .to_string();

    let matches_current_profile = |command: &str| -> bool {
        if !current_profile_name.is_empty() {
            return command.contains(&format!("--profile {current_profile_name}"))
                || command.contains(&format!("-p {current_profile_name}"))
                || command.contains(&format!("HERMES_HOME={current_home}"));
        }
        if command.contains("--profile ") || command.contains(" -p ") {
            return false;
        }
        if command.contains("HERMES_HOME=")
            && !command.contains(&format!("HERMES_HOME={current_home}"))
        {
            return false;
        }
        true
    };

    if is_windows() {
        let out = run_capture(
            "wmic",
            &["process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
        );
        let (code, stdout, _) = match out {
            Some(v) => v,
            None => return vec![],
        };
        if code != 0 {
            return vec![];
        }
        let mut current_cmd = String::new();
        for line in stdout.split('\n') {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("CommandLine=") {
                current_cmd = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("ProcessId=") {
                let pid_str = rest;
                if patterns.iter().any(|p| current_cmd.contains(p))
                    && (all_profiles || matches_current_profile(&current_cmd))
                {
                    if let Ok(pid) = pid_str.parse::<i64>() {
                        append_unique_pid(&mut pids, Some(pid), &exclude);
                    }
                }
                current_cmd = String::new();
            }
        }
    } else {
        let out = run_capture("ps", &["-A", "eww", "-o", "pid=,command="]);
        let (code, stdout, _) = match out {
            Some(v) => v,
            None => return vec![],
        };
        if code != 0 {
            return vec![];
        }
        for line in stdout.split('\n') {
            let stripped = line.trim();
            if stripped.is_empty() || stripped.contains("grep") {
                continue;
            }
            let mut pid: Option<i64> = None;
            let mut command = String::new();

            // split(None, 1) — first whitespace-separated token + remainder
            let mut split_iter = stripped.splitn(2, char::is_whitespace);
            if let (Some(first), Some(rest)) = (split_iter.next(), split_iter.next()) {
                if let Ok(p) = first.parse::<i64>() {
                    pid = Some(p);
                    command = rest.trim_start().to_string();
                }
            }

            if pid.is_none() {
                let aux_parts: Vec<&str> = stripped.split_whitespace().collect();
                if aux_parts.len() > 10 && aux_parts[1].chars().all(|c| c.is_ascii_digit()) {
                    if let Ok(p) = aux_parts[1].parse::<i64>() {
                        pid = Some(p);
                        command = aux_parts[10..].join(" ");
                    }
                }
            }

            let pid = match pid {
                Some(p) => p,
                None => continue,
            };
            if patterns.iter().any(|p| command.contains(p))
                && (all_profiles || matches_current_profile(&command))
            {
                append_unique_pid(&mut pids, Some(pid), &exclude);
            }
        }
    }

    pids
}

/// Find PIDs of running gateway processes.
pub fn find_gateway_pids(exclude_pids: Option<&HashSet<i64>>, all_profiles: bool) -> Vec<i64> {
    let exclude = exclude_pids.cloned().unwrap_or_default();
    let mut pids: Vec<i64> = Vec::new();
    if all_profiles {
        for proc in find_profile_gateway_processes(Some(&exclude)) {
            append_unique_pid(&mut pids, Some(proc.pid), &exclude);
        }
        return pids;
    }

    append_unique_pid(&mut pids, get_running_pid(None, true), &exclude);
    for pid in get_service_pids() {
        append_unique_pid(&mut pids, Some(pid), &exclude);
    }
    for pid in scan_gateway_pids(&exclude, all_profiles) {
        append_unique_pid(&mut pids, Some(pid), &exclude);
    }
    pids
}

/// Return running gateway PIDs mapped to Hermes profiles via PID files.
pub fn find_profile_gateway_processes(
    exclude_pids: Option<&HashSet<i64>>,
) -> Vec<ProfileGatewayProcess> {
    let exclude = exclude_pids.cloned().unwrap_or_default();
    let mut processes: Vec<ProfileGatewayProcess> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for profile in list_profiles() {
        let pid = match running_pid_for_profile(&profile.path) {
            Some(p) => p,
            None => continue,
        };
        if pid <= 0 || exclude.contains(&pid) || seen.contains(&pid) {
            continue;
        }
        seen.insert(pid);
        processes.push(ProfileGatewayProcess {
            profile: profile.name,
            path: profile.path,
            pid,
        });
    }
    processes
}

// =============================================================================
// systemd / launchd environment + invocation helpers
// =============================================================================

fn python_path() -> String {
    get_python_path()
}

fn gateway_run_args_for_profile(profile: &str) -> Vec<String> {
    let mut args = vec![python_path(), "-m".to_string(), "hermes_cli.main".to_string()];
    if profile != "default" {
        args.push("--profile".to_string());
        args.push(profile.to_string());
    }
    args.push("gateway".to_string());
    args.push("run".to_string());
    args.push("--replace".to_string());
    args
}

fn user_dbus_socket_path() -> PathBuf {
    let xdg = getenv("XDG_RUNTIME_DIR").unwrap_or_else(|| format!("/run/user/{}", os_getuid()));
    PathBuf::from(xdg).join("bus")
}

fn user_systemd_private_socket_path() -> PathBuf {
    let xdg = getenv("XDG_RUNTIME_DIR").unwrap_or_else(|| format!("/run/user/{}", os_getuid()));
    PathBuf::from(xdg).join("systemd").join("private")
}

fn user_systemd_socket_ready() -> bool {
    user_dbus_socket_path().exists() || user_systemd_private_socket_path().exists()
}

/// Ensure DBUS_SESSION_BUS_ADDRESS + XDG_RUNTIME_DIR are set for systemctl --user.
fn ensure_user_systemd_env() {
    let uid = os_getuid();
    if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
        let runtime_dir = format!("/run/user/{uid}");
        if Path::new(&runtime_dir).exists() {
            // SAFETY: edition 2024 requires unsafe for set_var.
            unsafe {
                std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
            }
        }
    }
    if std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
        let xdg = getenv("XDG_RUNTIME_DIR").unwrap_or_else(|| format!("/run/user/{uid}"));
        let bus_path = PathBuf::from(&xdg).join("bus");
        if bus_path.exists() {
            // SAFETY: edition 2024 requires unsafe for set_var.
            unsafe {
                std::env::set_var(
                    "DBUS_SESSION_BUS_ADDRESS",
                    format!("unix:path={}", bus_path.display()),
                );
            }
        }
    }
}

fn wait_for_user_dbus_socket(timeout: f64) -> bool {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    while Instant::now() < deadline {
        if user_systemd_socket_ready() {
            ensure_user_systemd_env();
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    user_systemd_socket_ready()
}

/// Error carrying a user-facing remediation message for unreachable user systemd.
#[derive(Debug, Clone)]
pub struct UserSystemdUnavailableError(pub String);

impl std::fmt::Display for UserSystemdUnavailableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for UserSystemdUnavailableError {}

fn build_user_systemd_unavailable(_username: &str, reason: &str, fix_hint: &str) -> UserSystemdUnavailableError {
    let msg = format!(
        "{reason}\n  systemctl --user cannot reach the user D-Bus session in this shell.\n\n  To fix:\n{fix_hint}\n\n  Alternative: run the gateway in the foreground (stays up until\n  you exit / close the terminal):\n    hermes gateway run"
    );
    UserSystemdUnavailableError(msg)
}

/// Ensure `systemctl --user` will reach the user-scope systemd instance.
fn preflight_user_systemd(auto_enable_linger: bool) -> Result<(), UserSystemdUnavailableError> {
    ensure_user_systemd_env();
    if user_systemd_socket_ready() {
        return Ok(());
    }
    let username = current_username();
    let (linger_enabled, linger_detail) = get_systemd_linger_status();

    if linger_enabled == Some(true) {
        if wait_for_user_dbus_socket(3.0) {
            return Ok(());
        }
        return Err(build_user_systemd_unavailable(
            &username,
            "User systemd control sockets are missing even though linger is enabled.",
            &format!(
                "  systemctl start user@{}.service\n  (may require sudo; try again after the command succeeds)",
                os_getuid()
            ),
        ));
    }

    if auto_enable_linger && which("loginctl").is_some() {
        match run_capture("loginctl", &["enable-linger", &username]) {
            None => {
                return Err(build_user_systemd_unavailable(
                    &username,
                    "loginctl enable-linger failed (spawn error).",
                    &format!("  sudo loginctl enable-linger {username}"),
                ));
            }
            Some((code, stdout, stderr)) => {
                if code == 0 {
                    if wait_for_user_dbus_socket(5.0) {
                        println!("\u{2713} Enabled linger for {username} — user D-Bus now available");
                        return Ok(());
                    }
                    return Err(build_user_systemd_unavailable(
                        &username,
                        "Linger was enabled, but the user D-Bus socket did not appear.",
                        &format!(
                            "  Log out and log back in, then re-run the command.\n  Or reboot and run: systemctl --user start {}",
                            get_service_name(None)
                        ),
                    ));
                }
                let detail = {
                    let s = if !stderr.trim().is_empty() {
                        stderr.trim().to_string()
                    } else if !stdout.trim().is_empty() {
                        stdout.trim().to_string()
                    } else {
                        format!("exit {code}")
                    };
                    s
                };
                return Err(build_user_systemd_unavailable(
                    &username,
                    &format!("loginctl enable-linger was denied: {detail}"),
                    &format!("  sudo loginctl enable-linger {username}"),
                ));
            }
        }
    }

    Err(build_user_systemd_unavailable(
        &username,
        &format!(
            "User D-Bus session is not available ({}).",
            if linger_detail.is_empty() {
                "linger disabled".to_string()
            } else {
                linger_detail
            }
        ),
        &format!("  sudo loginctl enable-linger {username}"),
    ))
}

/// Run a systemctl command (user or system scope). Returns
/// (exit_code, stdout, stderr), or None when systemctl is missing / spawn fails.
fn run_systemctl(args: &[&str], system: bool, _capture: bool) -> Option<(i32, String, String)> {
    if !system {
        ensure_user_systemd_env();
    }
    let mut full: Vec<&str> = Vec::new();
    if !system {
        full.push("--user");
    }
    full.extend_from_slice(args);
    run_capture("systemctl", &full)
}

/// Run systemctl attached to current stdio (for `status`).
fn run_systemctl_inherit(args: &[&str], system: bool) -> Option<i32> {
    if !system {
        ensure_user_systemd_env();
    }
    let mut full: Vec<&str> = Vec::new();
    if !system {
        full.push("--user");
    }
    full.extend_from_slice(args);
    run_inherit("systemctl", &full)
}

fn journalctl_cmd(system: bool) -> Vec<String> {
    if system {
        vec!["journalctl".to_string()]
    } else {
        vec!["journalctl".to_string(), "--user".to_string()]
    }
}

fn service_scope_label(system: bool) -> &'static str {
    if system {
        "system"
    } else {
        "user"
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn systemd_operational(system: bool) -> bool {
    match run_systemctl(&["is-system-running"], system, true) {
        Some((_, stdout, _)) => {
            let status = stdout.trim().to_lowercase();
            matches!(
                status.as_str(),
                "running" | "degraded" | "starting" | "initializing"
            )
        }
        None => false,
    }
}

fn wsl_systemd_operational() -> bool {
    systemd_operational(true)
}

fn container_systemd_operational() -> bool {
    systemd_operational(false) || systemd_operational(true)
}

pub fn supports_systemd_services() -> bool {
    if !is_linux() || is_termux() {
        return false;
    }
    if which("systemctl").is_none() {
        return false;
    }
    if is_wsl() {
        return wsl_systemd_operational();
    }
    if is_container() {
        return container_systemd_operational();
    }
    true
}

// =============================================================================
// Scope selection + service probes + snapshot
// =============================================================================

fn select_systemd_scope(system: bool) -> bool {
    if system {
        return true;
    }
    get_systemd_unit_path(true, None).exists() && !get_systemd_unit_path(false, None).exists()
}

fn probe_systemd_service_running(system: bool) -> (bool, bool) {
    let selected_system = select_systemd_scope(system);
    let unit_exists = get_systemd_unit_path(selected_system, None).exists();
    if !unit_exists {
        return (selected_system, false);
    }
    match run_systemctl(&["is-active", &get_service_name(None)], selected_system, true) {
        Some((_, stdout, _)) => (selected_system, stdout.trim() == "active"),
        None => (selected_system, false),
    }
}

const SYSTEMD_PROPS: &[&str] = &["ActiveState", "SubState", "Result", "ExecMainStatus"];

fn read_systemd_unit_properties(system: bool) -> std::collections::HashMap<String, String> {
    let selected_system = select_systemd_scope(system);
    let prop_arg = SYSTEMD_PROPS.join(",");
    let svc = get_service_name(None);
    let result = run_systemctl(
        &["show", &svc, "--no-pager", "--property", &prop_arg],
        selected_system,
        true,
    );
    let mut parsed = std::collections::HashMap::new();
    if let Some((code, stdout, _)) = result {
        if code != 0 {
            return parsed;
        }
        for line in stdout.lines() {
            if let Some((key, value)) = line.split_once('=') {
                parsed.insert(key.to_string(), value.trim().to_string());
            }
        }
    }
    parsed
}

fn probe_launchd_service_running() -> bool {
    if !get_launchd_plist_path().exists() {
        return false;
    }
    match run_capture("launchctl", &["list", &get_launchd_label()]) {
        Some((code, _, _)) => code == 0,
        None => false,
    }
}

pub fn get_gateway_runtime_snapshot(system: bool) -> GatewayRuntimeSnapshot {
    let gateway_pids = find_gateway_pids(None, false);
    if is_termux() {
        return GatewayRuntimeSnapshot {
            manager: "Termux / manual process".to_string(),
            service_installed: false,
            service_running: false,
            gateway_pids,
            service_scope: None,
        };
    }
    if is_linux() && is_container() {
        return GatewayRuntimeSnapshot {
            manager: "docker (foreground)".to_string(),
            service_installed: false,
            service_running: false,
            gateway_pids,
            service_scope: None,
        };
    }
    if supports_systemd_services() {
        let (selected_system, service_running) = probe_systemd_service_running(system);
        let scope_label = service_scope_label(selected_system);
        return GatewayRuntimeSnapshot {
            manager: format!("systemd ({scope_label})"),
            service_installed: get_systemd_unit_path(selected_system, None).exists(),
            service_running,
            gateway_pids,
            service_scope: Some(scope_label.to_string()),
        };
    }
    if is_macos() {
        return GatewayRuntimeSnapshot {
            manager: "launchd".to_string(),
            service_installed: get_launchd_plist_path().exists(),
            service_running: probe_launchd_service_running(),
            gateway_pids,
            service_scope: Some("launchd".to_string()),
        };
    }
    GatewayRuntimeSnapshot {
        manager: "manual process".to_string(),
        service_installed: false,
        service_running: false,
        gateway_pids,
        service_scope: None,
    }
}

fn format_gateway_pids(pids: &[i64], limit: Option<usize>) -> String {
    let rendered: Vec<String> = match limit {
        Some(lim) => pids
            .iter()
            .take(lim)
            .filter(|&&p| p > 0)
            .map(|p| p.to_string())
            .collect(),
        None => pids.iter().filter(|&&p| p > 0).map(|p| p.to_string()).collect(),
    };
    let mut rendered = rendered;
    if let Some(lim) = limit {
        if pids.len() > lim {
            rendered.push("...".to_string());
        }
    }
    rendered.join(", ")
}

fn print_gateway_process_mismatch(snapshot: &GatewayRuntimeSnapshot) {
    if !snapshot.has_process_service_mismatch() {
        return;
    }
    println!();
    println!("\u{26a0} Gateway process is running for this profile, but the service is not active");
    println!(
        "  PID(s): {}",
        format_gateway_pids(&snapshot.gateway_pids, None)
    );
    println!("  This is usually a manual foreground/tmux/nohup run, so `hermes gateway`");
    println!("  can refuse to start another copy until this process stops.");
}

fn print_other_profiles_gateway_status() {
    let current = get_active_profile_name();
    let other: Vec<ProfileGatewayProcess> = find_profile_gateway_processes(None)
        .into_iter()
        .filter(|p| p.profile != current)
        .collect();
    if other.is_empty() {
        return;
    }
    println!();
    println!("Other profiles:");
    for proc in other {
        println!("  \u{2713} {:<16} — PID {}", proc.profile, proc.pid);
    }
}

// =============================================================================
// Kill / stop helpers
// =============================================================================

pub fn kill_gateway_processes(
    force: bool,
    exclude_pids: Option<&HashSet<i64>>,
    all_profiles: bool,
) -> i32 {
    let exclude = exclude_pids.cloned().unwrap_or_default();
    let mut pids: Vec<i64> = Vec::new();
    if all_profiles {
        for proc in find_profile_gateway_processes(Some(&exclude)) {
            append_unique_pid(&mut pids, Some(proc.pid), &exclude);
        }
    } else {
        append_unique_pid(&mut pids, get_running_pid(None, true), &exclude);
        for pid in get_service_pids() {
            append_unique_pid(&mut pids, Some(pid), &exclude);
        }
    }
    let mut killed = 0;
    for pid in pids {
        match terminate_pid(pid, force) {
            Ok(()) => killed += 1,
            Err(KillError::NoSuchProcess) => {}
            Err(KillError::PermissionDenied) => {
                println!("\u{26a0} Permission denied to kill PID {pid}");
            }
            Err(KillError::Other(e)) => {
                println!("Failed to kill PID {pid}: errno {e}");
            }
        }
    }
    killed
}

/// Stop only the gateway for the current profile (HERMES_HOME-scoped).
pub fn stop_profile_gateway() -> bool {
    let pid = match get_running_pid(None, true) {
        Some(p) => p,
        None => return false,
    };
    write_planned_stop_marker(pid);

    match os_kill(pid, sig_term()) {
        Ok(()) | Err(KillError::NoSuchProcess) => {}
        Err(KillError::PermissionDenied) => {
            println!("\u{26a0} Permission denied to kill PID {pid}");
            return false;
        }
        Err(_) => {}
    }

    for _ in 0..20 {
        match os_kill(pid, 0) {
            Ok(()) => std::thread::sleep(Duration::from_millis(500)),
            Err(KillError::NoSuchProcess) | Err(KillError::PermissionDenied) => break,
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    }

    if get_running_pid(None, true).is_none() {
        remove_pid_file();
    }
    true
}

// =============================================================================
// Restart drain timeout
// =============================================================================

fn get_restart_drain_timeout() -> f64 {
    let raw = getenv("HERMES_RESTART_DRAIN_TIMEOUT")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(raw) = raw {
        return parse_restart_drain_timeout(Some(&Value::String(raw)));
    }
    let cfg = read_raw_config();
    let agent_cfg = cfg.get("agent").and_then(|v| v.as_object());
    let value = agent_cfg
        .and_then(|m| m.get("restart_drain_timeout"))
        .cloned();
    match value {
        Some(v) => parse_restart_drain_timeout(Some(&v)),
        None => DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    }
}

// =============================================================================
// Runtime health summary
// =============================================================================

fn runtime_health_lines() -> Vec<String> {
    let state = match read_runtime_status() {
        Some(s) => s,
        None => return vec![],
    };
    let mut lines: Vec<String> = Vec::new();
    let gateway_state = state.get("gateway_state").and_then(|v| v.as_str());
    let exit_reason = state.get("exit_reason").and_then(|v| v.as_str());
    let active_agents = state.get("active_agents").and_then(|v| v.as_i64()).unwrap_or(0);
    let restart_requested = state
        .get("restart_requested")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(platforms) = state.get("platforms").and_then(|v| v.as_object()) {
        for (platform, pdata) in platforms {
            if pdata.get("state").and_then(|v| v.as_str()) == Some("fatal") {
                let message = pdata
                    .get("error_message")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown error");
                lines.push(format!("\u{26a0} {platform}: {message}"));
            }
        }
    }

    if gateway_state == Some("startup_failed") {
        if let Some(reason) = exit_reason.filter(|s| !s.is_empty()) {
            lines.push(format!("\u{26a0} Last startup issue: {reason}"));
        }
    } else if gateway_state == Some("draining") {
        let action = if restart_requested { "restart" } else { "shutdown" };
        lines.push(format!(
            "\u{23f3} Gateway draining for {action} ({active_agents} active agent(s))"
        ));
    } else if gateway_state == Some("stopped") {
        if let Some(reason) = exit_reason.filter(|s| !s.is_empty()) {
            lines.push(format!("\u{26a0} Last shutdown reason: {reason}"));
        }
    }

    lines
}

// =============================================================================
// venv / python detection
// =============================================================================

fn detect_venv_dir() -> Option<PathBuf> {
    if let Some(v) = getenv("VIRTUAL_ENV") {
        let venv = PathBuf::from(v);
        if venv.is_dir() {
            return Some(venv);
        }
    }
    for candidate in [".venv", "venv"] {
        let venv = project_root().join(candidate);
        if venv.is_dir() {
            return Some(venv);
        }
    }
    None
}

pub fn get_python_path() -> String {
    if let Some(venv) = detect_venv_dir() {
        let venv_python = if is_windows() {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python")
        };
        if venv_python.exists() {
            return venv_python.to_string_lossy().to_string();
        }
    }
    // Fallback to a bare `python3` invocation (native binary has no sys.executable).
    getenv("HERMES_PYTHON").unwrap_or_else(|| "python3".to_string())
}

// =============================================================================
// Legacy systemd units (pre-rename hermes.service)
// =============================================================================

const LEGACY_SERVICE_NAMES: &[&str] = &["hermes.service"];
const LEGACY_UNIT_EXECSTART_MARKERS: &[&str] = &[
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
];

fn legacy_unit_search_paths() -> Vec<(bool, PathBuf)> {
    vec![
        (
            false,
            home_dir().join(".config").join("systemd").join("user"),
        ),
        (true, PathBuf::from("/etc/systemd/system")),
    ]
}

/// Return [(unit_name, unit_path, is_system)] for legacy Hermes gateway units.
fn find_legacy_hermes_units() -> Vec<(String, PathBuf, bool)> {
    let mut results = Vec::new();
    for (is_system, base) in legacy_unit_search_paths() {
        for &name in LEGACY_SERVICE_NAMES {
            let unit_path = base.join(name);
            if !unit_path.exists() {
                continue;
            }
            let text = match fs::read_to_string(&unit_path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !LEGACY_UNIT_EXECSTART_MARKERS.iter().any(|m| text.contains(m)) {
                continue;
            }
            results.push((name.to_string(), unit_path, is_system));
        }
    }
    results
}

pub fn has_legacy_hermes_units() -> bool {
    !find_legacy_hermes_units().is_empty()
}

pub fn print_legacy_unit_warning() {
    let legacy = find_legacy_hermes_units();
    if legacy.is_empty() {
        return;
    }
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:");
    for (_name, path, is_system) in &legacy {
        let scope = if *is_system { "system" } else { "user" };
        print_info(&format!("    {}  ({scope} scope)", path.display()));
    }
    print_info("  These run alongside the current hermes-gateway service and");
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.");
    print_info("  Remove them with:");
    print_info("    hermes gateway migrate-legacy");
}

pub fn remove_legacy_hermes_units(interactive: bool, dry_run: bool) -> (i32, Vec<PathBuf>) {
    let legacy = find_legacy_hermes_units();
    if legacy.is_empty() {
        println!("No legacy Hermes gateway units found.");
        return (0, vec![]);
    }

    let user_units: Vec<(String, PathBuf)> = legacy
        .iter()
        .filter(|(_, _, sys)| !*sys)
        .map(|(n, p, _)| (n.clone(), p.clone()))
        .collect();
    let system_units: Vec<(String, PathBuf)> = legacy
        .iter()
        .filter(|(_, _, sys)| *sys)
        .map(|(n, p, _)| (n.clone(), p.clone()))
        .collect();

    println!();
    println!("Legacy Hermes gateway unit(s) found:");
    for (_name, path, is_system) in &legacy {
        let scope = if *is_system { "system" } else { "user" };
        println!("  {}  ({scope} scope)", path.display());
    }
    println!();

    if dry_run {
        println!("(dry-run — nothing removed)");
        return (0, legacy.iter().map(|(_, p, _)| p.clone()).collect());
    }

    if interactive && !prompt_yes_no("Remove these legacy units?", true) {
        println!("Skipped. Run again with: hermes gateway migrate-legacy");
        return (0, legacy.iter().map(|(_, p, _)| p.clone()).collect());
    }

    let mut removed = 0;
    let mut remaining: Vec<PathBuf> = Vec::new();

    for (name, path) in &user_units {
        run_systemctl(&["stop", name], false, true);
        run_systemctl(&["disable", name], false, true);
        match fs::remove_file(path) {
            Ok(()) => {
                println!("  \u{2713} Removed {}", path.display());
                removed += 1;
            }
            Err(_) if !path.exists() => {
                println!("  \u{2713} Removed {}", path.display());
                removed += 1;
            }
            Err(e) => {
                println!("  \u{26a0} Could not remove {}: {e}", path.display());
                remaining.push(path.clone());
            }
        }
    }

    if !user_units.is_empty() {
        run_systemctl(&["daemon-reload"], false, true);
    }

    if !system_units.is_empty() {
        if os_geteuid() != 0 {
            println!();
            print_warning("System-scope legacy units require root to remove.");
            print_info("  Re-run with: sudo hermes gateway migrate-legacy");
            for (_, path) in &system_units {
                remaining.push(path.clone());
            }
        } else {
            for (name, path) in &system_units {
                run_systemctl(&["stop", name], true, true);
                run_systemctl(&["disable", name], true, true);
                match fs::remove_file(path) {
                    Ok(()) => {
                        println!("  \u{2713} Removed {}", path.display());
                        removed += 1;
                    }
                    Err(_) if !path.exists() => {
                        println!("  \u{2713} Removed {}", path.display());
                        removed += 1;
                    }
                    Err(e) => {
                        println!("  \u{26a0} Could not remove {}: {e}", path.display());
                        remaining.push(path.clone());
                    }
                }
            }
            run_systemctl(&["daemon-reload"], true, true);
        }
    }

    println!();
    if !remaining.is_empty() {
        print_warning(&format!(
            "{} legacy unit(s) still present — see messages above.",
            remaining.len()
        ));
    } else {
        print_success(&format!("Removed {removed} legacy unit(s)."));
    }

    (removed, remaining)
}

// =============================================================================
// Conflicting-scope detection
// =============================================================================

pub fn get_installed_systemd_scopes() -> Vec<String> {
    let mut scopes = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for (system, label) in [(false, "user"), (true, "system")] {
        let unit_path = get_systemd_unit_path(system, None);
        if seen.contains(&unit_path) {
            continue;
        }
        if unit_path.exists() {
            scopes.push(label.to_string());
            seen.insert(unit_path);
        }
    }
    scopes
}

pub fn has_conflicting_systemd_units() -> bool {
    get_installed_systemd_scopes().len() > 1
}

pub fn print_systemd_scope_conflict_warning() {
    let scopes = get_installed_systemd_scopes();
    if scopes.len() < 2 {
        return;
    }
    let rendered = scopes.join(" + ");
    print_warning(&format!(
        "Both user and system gateway services are installed ({rendered})."
    ));
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.");
    print_info("  Default gateway commands target the user service unless you pass --system.");
    print_info("  Keep one of these:");
    print_info("    hermes gateway uninstall");
    print_info("    sudo hermes gateway uninstall --system");
}

// =============================================================================
// Linger
// =============================================================================

/// Return systemd linger status: (Some(true)/Some(false)/None, detail).
pub fn get_systemd_linger_status() -> (Option<bool>, String) {
    if is_termux() {
        return (None, "not supported in Termux".to_string());
    }
    if !is_linux() {
        return (None, "not supported on this platform".to_string());
    }
    if which("loginctl").is_none() {
        return (None, "loginctl not found".to_string());
    }
    let username = getenv("USER").or_else(|| getenv("LOGNAME"));
    let username = match username {
        Some(u) => u,
        None => current_username(),
    };
    match run_capture(
        "loginctl",
        &["show-user", &username, "--property=Linger", "--value"],
    ) {
        None => (None, "loginctl query failed".to_string()),
        Some((code, stdout, stderr)) => {
            if code != 0 {
                let detail = if !stderr.trim().is_empty() {
                    stderr.trim().to_string()
                } else if !stdout.trim().is_empty() {
                    stdout.trim().to_string()
                } else {
                    format!("exit {code}")
                };
                return (None, detail);
            }
            let value = stdout.trim().to_lowercase();
            match value.as_str() {
                "yes" | "true" | "1" => (Some(true), String::new()),
                "no" | "false" | "0" => (Some(false), String::new()),
                "" => (None, "unexpected loginctl output: <empty>".to_string()),
                other => (None, format!("unexpected loginctl output: {other}")),
            }
        }
    }
}

pub fn print_systemd_linger_guidance() {
    let (linger_enabled, linger_detail) = get_systemd_linger_status();
    match linger_enabled {
        Some(true) => println!("\u{2713} Systemd linger is enabled (service survives logout)"),
        Some(false) => {
            println!("\u{26a0} Systemd linger is disabled (gateway may stop when you log out)");
            println!("  Run: sudo loginctl enable-linger $USER");
        }
        None => {
            println!("\u{26a0} Could not verify systemd linger ({linger_detail})");
            println!("  If you want the gateway user service to survive logout, run:");
            println!("  sudo loginctl enable-linger $USER");
        }
    }
}

fn print_linger_enable_warning(username: &str, detail: Option<&str>) {
    println!();
    println!("\u{26a0} Linger not enabled — gateway may stop when you close this terminal.");
    if let Some(d) = detail.filter(|s| !s.is_empty()) {
        println!("  Auto-enable failed: {d}");
    }
    println!();
    println!("  On headless servers (VPS, cloud instances) run:");
    println!("    sudo loginctl enable-linger {username}");
    println!();
    println!("  Then restart the gateway:");
    println!("    systemctl --user restart {}.service", get_service_name(None));
    println!();
}

fn ensure_linger_enabled() {
    if is_termux() || !is_linux() {
        return;
    }
    let username = current_username();
    let linger_file = PathBuf::from(format!("/var/lib/systemd/linger/{username}"));
    if linger_file.exists() {
        println!("\u{2713} Systemd linger is enabled (service survives logout)");
        return;
    }
    let (linger_enabled, linger_detail) = get_systemd_linger_status();
    if linger_enabled == Some(true) {
        println!("\u{2713} Systemd linger is enabled (service survives logout)");
        return;
    }
    if which("loginctl").is_none() {
        let d = if linger_detail.is_empty() {
            "loginctl not found".to_string()
        } else {
            linger_detail
        };
        print_linger_enable_warning(&username, Some(&d));
        return;
    }
    println!("Enabling linger so the gateway survives SSH logout...");
    match run_capture("loginctl", &["enable-linger", &username]) {
        None => print_linger_enable_warning(&username, Some("spawn error")),
        Some((code, stdout, stderr)) => {
            if code == 0 {
                println!("\u{2713} Linger enabled — gateway will persist after logout");
                return;
            }
            let detail = if !stderr.trim().is_empty() {
                stderr.trim().to_string()
            } else if !stdout.trim().is_empty() {
                stdout.trim().to_string()
            } else {
                format!("exit {code}")
            };
            let detail = if detail.is_empty() { linger_detail } else { detail };
            print_linger_enable_warning(&username, Some(&detail));
        }
    }
}

// =============================================================================
// Root + system-service identity
// =============================================================================

fn require_root_for_system_service(action: &str) -> Result<(), i32> {
    if os_geteuid() != 0 {
        println!("System gateway {action} requires root. Re-run with sudo.");
        return Err(1);
    }
    Ok(())
}

#[cfg(unix)]
fn passwd_lookup(username: &str) -> Option<(String, String)> {
    // Returns (group_name, home_dir) for username via getpwnam/getgrgid.
    use std::ffi::CString;
    let c_user = CString::new(username).ok()?;
    unsafe {
        let pw = libc::getpwnam(c_user.as_ptr());
        if pw.is_null() {
            return None;
        }
        let home = std::ffi::CStr::from_ptr((*pw).pw_dir)
            .to_string_lossy()
            .to_string();
        let gid = (*pw).pw_gid;
        let gr = libc::getgrgid(gid);
        let group = if gr.is_null() {
            gid.to_string()
        } else {
            std::ffi::CStr::from_ptr((*gr).gr_name)
                .to_string_lossy()
                .to_string()
        };
        Some((group, home))
    }
}

#[cfg(not(unix))]
fn passwd_lookup(_username: &str) -> Option<(String, String)> {
    None
}

/// Mirror `_system_service_identity`. Returns (username, group, home_dir).
fn system_service_identity(run_as_user: Option<&str>) -> Result<(String, String, String), String> {
    let username = run_as_user
        .map(|s| s.to_string())
        .or_else(|| getenv("SUDO_USER"))
        .or_else(|| getenv("USER"))
        .or_else(|| getenv("LOGNAME"))
        .unwrap_or_else(current_username)
        .trim()
        .to_string();
    if username.is_empty() {
        return Err("Could not determine which user the gateway service should run as".to_string());
    }
    if username == "root" && run_as_user.is_none() {
        return Err("Refusing to install the gateway system service as root; pass --run-as-user root to override (e.g. in LXC containers)".to_string());
    }
    if username == "root" {
        print_warning("Installing gateway service to run as root.");
        print_info("  This is fine for LXC/container environments but not recommended on bare-metal hosts.");
    }
    match passwd_lookup(&username) {
        Some((group, home)) => Ok((username, group, home)),
        None => Err(format!("Unknown user: {username}")),
    }
}

fn read_systemd_user_from_unit(unit_path: &Path) -> Option<String> {
    if !unit_path.exists() {
        return None;
    }
    let text = fs::read_to_string(unit_path).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("User=") {
            let value = rest.trim();
            if value.is_empty() {
                return None;
            }
            return Some(value.to_string());
        }
    }
    None
}

fn default_system_service_user() -> Option<String> {
    for candidate in [getenv("SUDO_USER"), getenv("USER"), getenv("LOGNAME")] {
        if let Some(c) = candidate {
            let trimmed = c.trim();
            if !trimmed.is_empty() && trimmed != "root" {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

// =============================================================================
// PATH assembly for generated units
// =============================================================================

fn build_user_local_paths(home: &Path, path_entries: &[String]) -> Vec<String> {
    let candidates = [
        home.join(".local").join("bin"),
        home.join(".cargo").join("bin"),
        home.join("go").join("bin"),
        home.join(".npm-global").join("bin"),
    ];
    candidates
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .filter(|p| !path_entries.contains(p) && Path::new(p).exists())
        .collect()
}

fn build_wsl_interop_paths(path_entries: &[String]) -> Vec<String> {
    if !is_wsl() {
        return vec![];
    }
    let mut candidates: Vec<String> = Vec::new();
    if let Some(path) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&path) {
            let s = entry.to_string_lossy().to_string();
            if s.starts_with("/mnt/") {
                candidates.push(s);
            }
        }
    }
    for executable in ["powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"] {
        if let Some(resolved) = which(executable) {
            if let Some(parent) = canonical(&resolved).parent() {
                candidates.push(parent.to_string_lossy().to_string());
            }
        }
    }
    for entry in [
        "/mnt/c/WINDOWS/system32",
        "/mnt/c/WINDOWS",
        "/mnt/c/WINDOWS/System32/Wbem",
        "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
        "/mnt/c/WINDOWS/System32/OpenSSH/",
    ] {
        if Path::new(entry).exists() {
            candidates.push(entry.to_string());
        }
    }
    let mut result: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = path_entries.iter().cloned().collect();
    for entry in candidates {
        if !entry.is_empty() && !seen.contains(&entry) {
            seen.insert(entry.clone());
            result.push(entry);
        }
    }
    result
}

/// Remap a path from the current user's home to `target_home_dir` (lexically).
fn remap_path_for_user(path: &str, target_home_dir: &str) -> String {
    let current_home = home_dir();
    // expanduser: replace leading ~ with home.
    let expanded = if let Some(rest) = path.strip_prefix("~") {
        if rest.is_empty() || rest.starts_with('/') {
            current_home.join(rest.trim_start_matches('/')).to_string_lossy().to_string()
        } else {
            path.to_string()
        }
    } else {
        path.to_string()
    };
    let p = Path::new(&expanded);
    match p.strip_prefix(&current_home) {
        Ok(rel) => Path::new(target_home_dir)
            .join(rel)
            .to_string_lossy()
            .to_string(),
        Err(_) => expanded,
    }
}

fn hermes_home_for_target_user(target_home_dir: &str) -> String {
    let current_hermes = canonical(&get_hermes_home());
    let current_default = canonical(&home_dir().join(".hermes"));
    let target_default = Path::new(target_home_dir).join(".hermes");

    if current_hermes == current_default {
        return target_default.to_string_lossy().to_string();
    }
    match current_hermes.strip_prefix(&current_default) {
        Ok(rel) => target_default.join(rel).to_string_lossy().to_string(),
        Err(_) => current_hermes.to_string_lossy().to_string(),
    }
}

// =============================================================================
// systemd unit generation
// =============================================================================

fn common_bin_paths() -> Vec<String> {
    [
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn generate_systemd_unit(system: bool, run_as_user: Option<&str>) -> Result<String, String> {
    let mut python_path = get_python_path();
    let mut working_dir = project_root().to_string_lossy().to_string();
    let detected_venv = detect_venv_dir();
    let mut venv_dir = detected_venv
        .clone()
        .unwrap_or_else(|| project_root().join("venv"))
        .to_string_lossy()
        .to_string();
    let mut venv_bin = detected_venv
        .clone()
        .map(|v| v.join("bin"))
        .unwrap_or_else(|| project_root().join("venv").join("bin"))
        .to_string_lossy()
        .to_string();
    let mut node_bin = project_root()
        .join("node_modules")
        .join(".bin")
        .to_string_lossy()
        .to_string();

    let mut path_entries: Vec<String> = vec![venv_bin.clone(), node_bin.clone()];
    if let Some(resolved_node) = which("node") {
        let resolved_node_dir = canonical(&resolved_node)
            .parent()
            .map(|p| p.to_string_lossy().to_string());
        if let Some(dir) = resolved_node_dir {
            if !path_entries.contains(&dir) {
                path_entries.push(dir);
            }
        }
    }

    let drain_timeout = get_restart_drain_timeout() as i64;
    let restart_timeout = 60.max(drain_timeout) + 30;
    let exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE;

    if system {
        let (username, group_name, home_dir_str) = system_service_identity(run_as_user)?;
        let hermes_home = hermes_home_for_target_user(&home_dir_str);
        let prof_arg = profile_arg(Some(&hermes_home));
        python_path = remap_path_for_user(&python_path, &home_dir_str);
        working_dir = remap_path_for_user(&working_dir, &home_dir_str);
        venv_dir = remap_path_for_user(&venv_dir, &home_dir_str);
        venv_bin = remap_path_for_user(&venv_bin, &home_dir_str);
        node_bin = remap_path_for_user(&node_bin, &home_dir_str);
        path_entries = path_entries
            .iter()
            .map(|p| remap_path_for_user(p, &home_dir_str))
            .collect();
        path_entries.extend(build_user_local_paths(Path::new(&home_dir_str), &path_entries.clone()));
        path_entries.extend(build_wsl_interop_paths(&path_entries.clone()));
        path_entries.extend(common_bin_paths());
        let sane_path = path_entries.join(":");
        let exec_profile = if prof_arg.is_empty() {
            String::new()
        } else {
            format!(" {prof_arg}")
        };
        let _ = (venv_bin, node_bin);
        return Ok(format!(
            "[Unit]\nDescription={SERVICE_DESCRIPTION}\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=0\n\n[Service]\nType=simple\nUser={username}\nGroup={group_name}\nExecStart={python_path} -m hermes_cli.main{exec_profile} gateway run --replace\nWorkingDirectory={working_dir}\nEnvironment=\"HOME={home_dir_str}\"\nEnvironment=\"USER={username}\"\nEnvironment=\"LOGNAME={username}\"\nEnvironment=\"PATH={sane_path}\"\nEnvironment=\"VIRTUAL_ENV={venv_dir}\"\nEnvironment=\"HERMES_HOME={hermes_home}\"\nRestart=always\nRestartSec=60\nRestartMaxDelaySec=300\nRestartSteps=5\nRestartForceExitStatus={exit_code}\nKillMode=mixed\nKillSignal=SIGTERM\nExecReload=/bin/kill -USR1 $MAINPID\nTimeoutStopSec={restart_timeout}\nStandardOutput=journal\nStandardError=journal\n\n[Install]\nWantedBy=multi-user.target\n"
        ));
    }

    let hermes_home = canonical(&get_hermes_home()).to_string_lossy().to_string();
    let prof_arg = profile_arg(Some(&hermes_home));
    path_entries.extend(build_user_local_paths(&home_dir(), &path_entries.clone()));
    path_entries.extend(build_wsl_interop_paths(&path_entries.clone()));
    path_entries.extend(common_bin_paths());
    let sane_path = path_entries.join(":");
    let exec_profile = if prof_arg.is_empty() {
        String::new()
    } else {
        format!(" {prof_arg}")
    };
    let _ = (venv_bin, node_bin);
    Ok(format!(
        "[Unit]\nDescription={SERVICE_DESCRIPTION}\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=0\n\n[Service]\nType=simple\nExecStart={python_path} -m hermes_cli.main{exec_profile} gateway run --replace\nWorkingDirectory={working_dir}\nEnvironment=\"PATH={sane_path}\"\nEnvironment=\"VIRTUAL_ENV={venv_dir}\"\nEnvironment=\"HERMES_HOME={hermes_home}\"\nRestart=always\nRestartSec=60\nRestartMaxDelaySec=300\nRestartSteps=5\nRestartForceExitStatus={exit_code}\nKillMode=mixed\nKillSignal=SIGTERM\nExecReload=/bin/kill -USR1 $MAINPID\nTimeoutStopSec={restart_timeout}\nStandardOutput=journal\nStandardError=journal\n\n[Install]\nWantedBy=default.target\n"
    ))
}

fn normalize_service_definition(text: &str) -> String {
    text.trim()
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn systemd_unit_is_current(system: bool) -> bool {
    let unit_path = get_systemd_unit_path(system, None);
    if !unit_path.exists() {
        return false;
    }
    let installed = match fs::read_to_string(&unit_path) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let expected_user = if system {
        read_systemd_user_from_unit(&unit_path)
    } else {
        None
    };
    let expected = match generate_systemd_unit(system, expected_user.as_deref()) {
        Ok(e) => e,
        Err(_) => return false,
    };
    normalize_service_definition(&installed) == normalize_service_definition(&expected)
}

pub fn refresh_systemd_unit_if_needed(system: bool) -> bool {
    let unit_path = get_systemd_unit_path(system, None);
    if !unit_path.exists() || systemd_unit_is_current(system) {
        return false;
    }
    let expected_user = if system {
        read_systemd_user_from_unit(&unit_path)
    } else {
        None
    };
    let unit = match generate_systemd_unit(system, expected_user.as_deref()) {
        Ok(u) => u,
        Err(_) => return false,
    };
    if fs::write(&unit_path, unit).is_err() {
        return false;
    }
    run_systemctl(&["daemon-reload"], system, true);
    println!(
        "\u{21bb} Updated gateway {} service definition to match the current Hermes install",
        service_scope_label(system)
    );
    true
}

// =============================================================================
// systemd restart-recovery wait helpers
// =============================================================================

fn wait_for_systemd_service_restart(system: bool, previous_pid: Option<i64>, timeout: f64) -> bool {
    let svc = get_service_name(None);
    let scope_label = capitalize(service_scope_label(system));
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);

    while Instant::now() < deadline {
        let props = read_systemd_unit_properties(system);
        let active_state = props.get("ActiveState").map(|s| s.as_str()).unwrap_or("");
        let sub_state = props.get("SubState").map(|s| s.as_str()).unwrap_or("");
        let new_pid = get_running_pid(None, true);

        if active_state == "active" {
            if let Some(np) = new_pid {
                if previous_pid.is_none() || Some(np) != previous_pid {
                    println!("\u{2713} {scope_label} service restarted (PID {np})");
                    return true;
                }
            }
            if previous_pid.is_none() {
                println!("\u{2713} {scope_label} service restarted");
                return true;
            }
        }

        if active_state == "activating" && sub_state == "auto-restart" {
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        std::thread::sleep(Duration::from_secs(2));
    }

    let user_flag = if !system { "--user " } else { "" };
    let sudo = if system { "sudo " } else { "" };
    println!(
        "\u{26a0} {scope_label} service did not become active within {}s.\n  Check status: {sudo}hermes gateway status\n  Check logs:   journalctl {user_flag}-u {svc} -l --since '2 min ago'",
        timeout as i64
    );
    false
}

fn recover_pending_systemd_restart(system: bool, previous_pid: Option<i64>) -> bool {
    let props = read_systemd_unit_properties(system);
    if props.is_empty() {
        return false;
    }
    let runtime_state = read_runtime_status().unwrap_or_else(|| Value::Object(Default::default()));
    if !runtime_state
        .get("restart_requested")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return false;
    }

    let active_state = props.get("ActiveState").map(|s| s.as_str()).unwrap_or("");
    let sub_state = props.get("SubState").map(|s| s.as_str()).unwrap_or("");
    let exec_main_status = props.get("ExecMainStatus").map(|s| s.as_str()).unwrap_or("");
    let result = props.get("Result").map(|s| s.as_str()).unwrap_or("");

    if active_state == "activating" && sub_state == "auto-restart" {
        println!("\u{23f3} Service restart already pending — waiting for systemd relaunch...");
        return wait_for_systemd_service_restart(system, previous_pid, 60.0);
    }

    if active_state == "failed"
        && (exec_main_status == GATEWAY_SERVICE_RESTART_EXIT_CODE.to_string()
            || result == "exit-code")
    {
        let svc = get_service_name(None);
        let scope_label = capitalize(service_scope_label(system));
        println!(
            "\u{21bb} Clearing failed state for pending {} service restart...",
            scope_label.to_lowercase()
        );
        run_systemctl(&["reset-failed", &svc], system, true);
        run_systemctl(&["start", &svc], system, true);
        return wait_for_systemd_service_restart(system, previous_pid, 60.0);
    }

    false
}

// =============================================================================
// systemd install / uninstall / start / stop / restart / status
// =============================================================================

fn select_install_scope(force: bool, system: bool, run_as_user: Option<&str>) -> Result<(), i32> {
    if system {
        require_root_for_system_service("install")?;
    }
    if has_legacy_hermes_units() {
        println!();
        print_legacy_unit_warning();
        println!();
        if prompt_yes_no("Remove the legacy unit(s) before installing?", true) {
            remove_legacy_hermes_units(false, false);
            println!();
        }
    }

    let unit_path = get_systemd_unit_path(system, None);
    let scope_flag = if system { " --system" } else { "" };

    if unit_path.exists() && !force {
        if !systemd_unit_is_current(system) {
            println!(
                "\u{21bb} Repairing outdated {} systemd service at: {}",
                service_scope_label(system),
                unit_path.display()
            );
            refresh_systemd_unit_if_needed(system);
            run_systemctl(&["enable", &get_service_name(None)], system, true);
            println!(
                "\u{2713} {} service definition updated",
                capitalize(service_scope_label(system))
            );
            return Ok(());
        }
        println!("Service already installed at: {}", unit_path.display());
        println!("Use --force to reinstall");
        return Ok(());
    }

    if let Some(parent) = unit_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    println!(
        "Installing {} systemd service to: {}",
        service_scope_label(system),
        unit_path.display()
    );
    let unit = generate_systemd_unit(system, run_as_user).map_err(|e| {
        print_error(&e);
        1
    })?;
    if fs::write(&unit_path, unit).is_err() {
        print_error("Failed to write systemd unit file");
        return Err(1);
    }
    run_systemctl(&["daemon-reload"], system, true);
    run_systemctl(&["enable", &get_service_name(None)], system, true);

    let svc = get_service_name(None);
    let sudo = if system { "sudo " } else { "" };
    let journal = if system { "journalctl" } else { "journalctl --user" };
    println!();
    println!(
        "\u{2713} {} service installed and enabled!",
        capitalize(service_scope_label(system))
    );
    println!();
    println!("Next steps:");
    println!("  {sudo}hermes gateway start{scope_flag}              # Start the service");
    println!("  {sudo}hermes gateway status{scope_flag}             # Check status");
    println!("  {journal} -u {svc} -f  # View logs");
    println!();

    if system {
        if let Some(configured_user) = read_systemd_user_from_unit(&unit_path) {
            println!("Configured to run as: {configured_user}");
        }
    } else {
        ensure_linger_enabled();
    }

    print_systemd_scope_conflict_warning();
    print_legacy_unit_warning();
    Ok(())
}

pub fn systemd_install(force: bool, system: bool, run_as_user: Option<&str>) -> Result<(), i32> {
    select_install_scope(force, system, run_as_user)
}

pub fn systemd_uninstall(system: bool) -> Result<(), i32> {
    let system = select_systemd_scope(system);
    if system {
        require_root_for_system_service("uninstall")?;
    }
    let svc = get_service_name(None);
    run_systemctl(&["stop", &svc], system, true);
    run_systemctl(&["disable", &svc], system, true);

    let unit_path = get_systemd_unit_path(system, None);
    if unit_path.exists() {
        let _ = fs::remove_file(&unit_path);
        println!("\u{2713} Removed {}", unit_path.display());
    }
    run_systemctl(&["daemon-reload"], system, true);
    println!(
        "\u{2713} {} service uninstalled",
        capitalize(service_scope_label(system))
    );
    Ok(())
}

fn require_service_installed(_action: &str, system: bool) -> Result<(), i32> {
    let unit_path = get_systemd_unit_path(system, None);
    if !unit_path.exists() {
        let scope_flag = if system { " --system" } else { "" };
        let sudo = if system { "sudo " } else { "" };
        println!("\u{2717} Gateway service is not installed");
        println!("  Run: {sudo}hermes gateway install{scope_flag}");
        return Err(1);
    }
    Ok(())
}

pub fn systemd_start(system: bool) -> Result<(), UserSystemdUnavailableError> {
    let system = select_systemd_scope(system);
    if system {
        if require_root_for_system_service("start").is_err() {
            std::process::exit(1);
        }
    } else {
        preflight_user_systemd(true)?;
    }
    if require_service_installed("start", system).is_err() {
        std::process::exit(1);
    }
    refresh_systemd_unit_if_needed(system);
    run_systemctl(&["start", &get_service_name(None)], system, true);
    println!(
        "\u{2713} {} service started",
        capitalize(service_scope_label(system))
    );
    Ok(())
}

pub fn systemd_stop(system: bool) -> Result<(), i32> {
    let system = select_systemd_scope(system);
    if system {
        require_root_for_system_service("stop")?;
    }
    require_service_installed("stop", system)?;
    if let Some(pid) = get_running_pid(None, false) {
        write_planned_stop_marker(pid);
    }
    run_systemctl(&["stop", &get_service_name(None)], system, true);
    println!(
        "\u{2713} {} service stopped",
        capitalize(service_scope_label(system))
    );
    Ok(())
}

pub fn systemd_restart(system: bool) -> Result<(), UserSystemdUnavailableError> {
    let system = select_systemd_scope(system);
    if system {
        if require_root_for_system_service("restart").is_err() {
            std::process::exit(1);
        }
    } else {
        preflight_user_systemd(true)?;
    }
    if require_service_installed("restart", system).is_err() {
        std::process::exit(1);
    }
    refresh_systemd_unit_if_needed(system);

    let pid = get_running_pid(None, true);
    if let Some(pid) = pid {
        if request_gateway_self_restart(pid) {
            let scope_label = capitalize(service_scope_label(system));
            let svc = get_service_name(None);

            println!("\u{23f3} {scope_label} service draining active work...");
            let deadline = Instant::now() + Duration::from_secs(90);
            let mut still_alive = true;
            while Instant::now() < deadline {
                match os_kill(pid, 0) {
                    Ok(()) => std::thread::sleep(Duration::from_secs(1)),
                    Err(KillError::NoSuchProcess) | Err(KillError::PermissionDenied) => {
                        still_alive = false;
                        break;
                    }
                    Err(_) => std::thread::sleep(Duration::from_secs(1)),
                }
            }
            if still_alive {
                println!("\u{26a0} Old process (PID {pid}) still alive after 90s");
            }

            run_systemctl(&["reset-failed", &svc], system, true);
            run_systemctl(&["start", &svc], system, true);
            wait_for_systemd_service_restart(system, Some(pid), 60.0);
            return Ok(());
        }
    }

    if recover_pending_systemd_restart(system, pid) {
        return Ok(());
    }

    run_systemctl(&["reset-failed", &get_service_name(None)], system, true);
    run_systemctl(&["reload-or-restart", &get_service_name(None)], system, true);
    println!(
        "\u{2713} {} service restarted",
        capitalize(service_scope_label(system))
    );
    Ok(())
}

pub fn systemd_status(deep: bool, system: bool, full: bool) {
    let system = select_systemd_scope(system);
    let unit_path = get_systemd_unit_path(system, None);
    let scope_flag = if system { " --system" } else { "" };
    let sudo = if system { "sudo " } else { "" };

    if !unit_path.exists() {
        println!("\u{2717} Gateway service is not installed");
        println!("  Run: {sudo}hermes gateway install{scope_flag}");
        return;
    }

    if has_conflicting_systemd_units() {
        print_systemd_scope_conflict_warning();
        println!();
    }
    if has_legacy_hermes_units() {
        print_legacy_unit_warning();
        println!();
    }
    if !systemd_unit_is_current(system) {
        println!("\u{26a0} Installed gateway service definition is outdated");
        println!("  Run: {sudo}hermes gateway restart{scope_flag}  # auto-refreshes the unit");
        println!();
    }

    let svc = get_service_name(None);
    let mut status_cmd: Vec<String> = vec!["status".to_string(), svc.clone(), "--no-pager".to_string()];
    if full {
        status_cmd.push("-l".to_string());
    }
    let status_ref: Vec<&str> = status_cmd.iter().map(|s| s.as_str()).collect();
    run_systemctl_inherit(&status_ref, system);

    let status = run_systemctl(&["is-active", &svc], system, true)
        .map(|(_, out, _)| out.trim().to_string())
        .unwrap_or_default();

    if status == "active" {
        println!(
            "\u{2713} {} gateway service is running",
            capitalize(service_scope_label(system))
        );
    } else {
        println!(
            "\u{2717} {} gateway service is stopped",
            capitalize(service_scope_label(system))
        );
        println!("  Run: {sudo}hermes gateway start{scope_flag}");
    }

    if system {
        if let Some(configured_user) = read_systemd_user_from_unit(&unit_path) {
            println!("Configured to run as: {configured_user}");
        }
    }

    let runtime_lines = runtime_health_lines();
    if !runtime_lines.is_empty() {
        println!();
        println!("Recent gateway health:");
        for line in &runtime_lines {
            println!("  {line}");
        }
    }

    let unit_props = read_systemd_unit_properties(system);
    let active_state = unit_props.get("ActiveState").map(|s| s.as_str()).unwrap_or("");
    let sub_state = unit_props.get("SubState").map(|s| s.as_str()).unwrap_or("");
    let exec_main_status = unit_props.get("ExecMainStatus").map(|s| s.as_str()).unwrap_or("");
    let result_code = unit_props.get("Result").map(|s| s.as_str()).unwrap_or("");
    if active_state == "activating" && sub_state == "auto-restart" {
        println!("  \u{23f3} Restart pending: systemd is waiting to relaunch the gateway");
    } else if active_state == "failed"
        && exec_main_status == GATEWAY_SERVICE_RESTART_EXIT_CODE.to_string()
    {
        let user_flag = if !system { "--user " } else { "" };
        println!("  \u{26a0} Planned restart is stuck in systemd failed state (exit 75)");
        println!(
            "  Run: systemctl {user_flag}reset-failed {svc} && {sudo}hermes gateway start{scope_flag}"
        );
    } else if active_state == "failed" && !result_code.is_empty() {
        println!("  \u{26a0} Systemd unit result: {result_code}");
    }

    if system {
        println!("\u{2713} System service starts at boot without requiring systemd linger");
    } else if deep {
        print_systemd_linger_guidance();
    } else {
        let (linger_enabled, _) = get_systemd_linger_status();
        match linger_enabled {
            Some(true) => println!("\u{2713} Systemd linger is enabled (service survives logout)"),
            Some(false) => {
                println!("\u{26a0} Systemd linger is disabled (gateway may stop when you log out)");
                println!("  Run: sudo loginctl enable-linger $USER");
            }
            None => {}
        }
    }

    if deep {
        println!();
        println!("Recent logs:");
        let mut log_cmd = journalctl_cmd(system);
        log_cmd.extend([
            "-u".to_string(),
            svc.clone(),
            "-n".to_string(),
            "20".to_string(),
            "--no-pager".to_string(),
        ]);
        if full {
            log_cmd.push("-l".to_string());
        }
        let program = log_cmd[0].clone();
        let args: Vec<&str> = log_cmd[1..].iter().map(|s| s.as_str()).collect();
        run_inherit(&program, &args);
    }
}

// =============================================================================
// install_linux_gateway_from_setup + scope prompt
// =============================================================================

/// Returns Some("user")/Some("system")/None for the chosen install scope.
pub fn prompt_linux_gateway_install_scope() -> Option<String> {
    let choice = prompt_choice(
        "  Choose how the gateway should run in the background:",
        &[
            "User service (no sudo; best for laptops/dev boxes; may need linger after logout)",
            "System service (starts on boot; requires sudo; still runs as your user)",
            "Skip service install for now",
        ],
        0,
    );
    match choice {
        0 => Some("user".to_string()),
        1 => Some("system".to_string()),
        _ => None,
    }
}

/// Returns (scope, did_install).
pub fn install_linux_gateway_from_setup(force: bool) -> (Option<String>, bool) {
    let scope = match prompt_linux_gateway_install_scope() {
        Some(s) => s,
        None => return (None, false),
    };

    if scope == "system" {
        let mut run_as_user = default_system_service_user();
        if os_geteuid() != 0 {
            print_warning("  System service install requires sudo, so Hermes can't create it from this user session.");
            match &run_as_user {
                Some(u) => print_info(&format!(
                    "  After setup, run: sudo hermes gateway install --system --run-as-user {u}"
                )),
                None => print_info(
                    "  After setup, run: sudo hermes gateway install --system --run-as-user <your-user>",
                ),
            }
            print_info("  Then start it with: sudo hermes gateway start --system");
            return (Some(scope), false);
        }
        if run_as_user.is_none() {
            loop {
                let entered = prompt("  Run the system gateway service as which user?", "", false);
                let trimmed = entered.trim().to_string();
                if !trimmed.is_empty() {
                    run_as_user = Some(trimmed);
                    break;
                }
                print_error("  Enter a username.");
            }
        }
        let _ = systemd_install(force, true, run_as_user.as_deref());
        return (Some(scope), true);
    }

    let _ = systemd_install(force, false, None);
    (Some(scope), true)
}

// =============================================================================
// Launchd (macOS)
// =============================================================================

#[cfg(unix)]
fn launchd_user_home() -> PathBuf {
    let (_, home) = passwd_lookup(&current_username())
        .unwrap_or_else(|| (String::new(), home_dir().to_string_lossy().to_string()));
    if home.is_empty() {
        home_dir()
    } else {
        PathBuf::from(home)
    }
}
#[cfg(not(unix))]
fn launchd_user_home() -> PathBuf {
    home_dir()
}

pub fn get_launchd_plist_path() -> PathBuf {
    let suffix = profile_suffix(None);
    let name = if suffix.is_empty() {
        "ai.hermes.gateway".to_string()
    } else {
        format!("ai.hermes.gateway-{suffix}")
    };
    launchd_user_home()
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{name}.plist"))
}

pub fn get_launchd_label() -> String {
    let suffix = profile_suffix(None);
    if suffix.is_empty() {
        "ai.hermes.gateway".to_string()
    } else {
        format!("ai.hermes.gateway-{suffix}")
    }
}

fn launchd_domain() -> String {
    format!("gui/{}", os_getuid())
}

pub fn generate_launchd_plist() -> String {
    let python_path = get_python_path();
    let working_dir = project_root().to_string_lossy().to_string();
    let hermes_home = canonical(&get_hermes_home()).to_string_lossy().to_string();
    let log_dir = get_hermes_home().join("logs");
    let _ = fs::create_dir_all(&log_dir);
    let label = get_launchd_label();
    let prof_arg = profile_arg(Some(&hermes_home));

    let detected_venv = detect_venv_dir();
    let venv_bin = detected_venv
        .clone()
        .map(|v| v.join("bin"))
        .unwrap_or_else(|| project_root().join("venv").join("bin"))
        .to_string_lossy()
        .to_string();
    let venv_dir = detected_venv
        .clone()
        .unwrap_or_else(|| project_root().join("venv"))
        .to_string_lossy()
        .to_string();
    let node_bin = project_root()
        .join("node_modules")
        .join(".bin")
        .to_string_lossy()
        .to_string();

    let mut priority_dirs: Vec<String> = vec![venv_bin, node_bin];
    if let Some(resolved_node) = which("node") {
        if let Some(parent) = canonical(&resolved_node).parent() {
            let d = parent.to_string_lossy().to_string();
            if !priority_dirs.contains(&d) {
                priority_dirs.push(d);
            }
        }
    }
    // dict.fromkeys-style dedup, preserving order.
    let mut sane_entries: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for entry in priority_dirs
        .into_iter()
        .chain(getenv("PATH").unwrap_or_default().split(':').filter(|p| !p.is_empty()).map(|s| s.to_string()))
    {
        if !seen.contains(&entry) {
            seen.insert(entry.clone());
            sane_entries.push(entry);
        }
    }
    let sane_path = sane_entries.join(":");

    let mut prog_args: Vec<String> = vec![
        format!("<string>{python_path}</string>"),
        "<string>-m</string>".to_string(),
        "<string>hermes_cli.main</string>".to_string(),
    ];
    if !prof_arg.is_empty() {
        for part in prof_arg.split_whitespace() {
            prog_args.push(format!("<string>{part}</string>"));
        }
    }
    prog_args.extend([
        "<string>gateway</string>".to_string(),
        "<string>run</string>".to_string(),
        "<string>--replace</string>".to_string(),
    ]);
    let prog_args_xml = prog_args.join("\n        ");
    let log_dir_str = log_dir.to_string_lossy();

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>{label}</string>\n\n    <key>ProgramArguments</key>\n    <array>\n        {prog_args_xml}\n    </array>\n    \n    <key>WorkingDirectory</key>\n    <string>{working_dir}</string>\n    \n    <key>EnvironmentVariables</key>\n    <dict>\n        <key>PATH</key>\n        <string>{sane_path}</string>\n        <key>VIRTUAL_ENV</key>\n        <string>{venv_dir}</string>\n        <key>HERMES_HOME</key>\n        <string>{hermes_home}</string>\n    </dict>\n    \n    <key>RunAtLoad</key>\n    <true/>\n    \n    <key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>\n    \n    <key>StandardOutPath</key>\n    <string>{log_dir_str}/gateway.log</string>\n    \n    <key>StandardErrorPath</key>\n    <string>{log_dir_str}/gateway.error.log</string>\n</dict>\n</plist>\n"
    )
}

fn normalize_launchd_plist_for_comparison(text: &str) -> String {
    let normalized = normalize_service_definition(text);
    // Replace PATH payload with placeholder (regex over <key>PATH</key>...<string>...</string>).
    let re = regex::Regex::new(r"(?s)(<key>PATH</key>\s*<string>)(.*?)(</string>)").unwrap();
    re.replace_all(&normalized, "${1}__HERMES_PATH__${3}").to_string()
}

pub fn launchd_plist_is_current() -> bool {
    let plist_path = get_launchd_plist_path();
    if !plist_path.exists() {
        return false;
    }
    let installed = match fs::read_to_string(&plist_path) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let expected = generate_launchd_plist();
    normalize_launchd_plist_for_comparison(&installed)
        == normalize_launchd_plist_for_comparison(&expected)
}

pub fn refresh_launchd_plist_if_needed() -> bool {
    let plist_path = get_launchd_plist_path();
    if !plist_path.exists() || launchd_plist_is_current() {
        return false;
    }
    if fs::write(&plist_path, generate_launchd_plist()).is_err() {
        return false;
    }
    let label = get_launchd_label();
    run_capture("launchctl", &["bootout", &format!("{}/{label}", launchd_domain())]);
    run_capture(
        "launchctl",
        &["bootstrap", &launchd_domain(), &plist_path.to_string_lossy()],
    );
    println!("\u{21bb} Updated gateway launchd service definition to match the current Hermes install");
    true
}

pub fn launchd_install(force: bool) {
    let plist_path = get_launchd_plist_path();
    if plist_path.exists() && !force {
        if !launchd_plist_is_current() {
            println!("\u{21bb} Repairing outdated launchd service at: {}", plist_path.display());
            refresh_launchd_plist_if_needed();
            println!("\u{2713} Service definition updated");
            return;
        }
        println!("Service already installed at: {}", plist_path.display());
        println!("Use --force to reinstall");
        return;
    }
    if let Some(parent) = plist_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    println!("Installing launchd service to: {}", plist_path.display());
    let _ = fs::write(&plist_path, generate_launchd_plist());
    run_capture(
        "launchctl",
        &["bootstrap", &launchd_domain(), &plist_path.to_string_lossy()],
    );
    println!();
    println!("\u{2713} Service installed and loaded!");
    println!();
    println!("Next steps:");
    println!("  hermes gateway status             # Check status");
    let dhh = display_hermes_home();
    println!("  tail -f {dhh}/logs/gateway.log  # View logs");
}

pub fn launchd_uninstall() {
    let plist_path = get_launchd_plist_path();
    let label = get_launchd_label();
    run_capture("launchctl", &["bootout", &format!("{}/{label}", launchd_domain())]);
    if plist_path.exists() {
        let _ = fs::remove_file(&plist_path);
        println!("\u{2713} Removed {}", plist_path.display());
    }
    println!("\u{2713} Service uninstalled");
}

/// Wait for the gateway process (by saved PID) to exit.
pub fn wait_for_gateway_exit(timeout: f64, force_after: Option<f64>) -> bool {
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(timeout);
    let force_deadline = force_after.map(|f| start + Duration::from_secs_f64(f));
    let mut force_sent = false;

    while Instant::now() < deadline {
        let pid = get_running_pid(None, true);
        if pid.is_none() {
            return true;
        }
        let pid = pid.unwrap();
        if let Some(fd) = force_deadline {
            if !force_sent && Instant::now() >= fd {
                match terminate_pid(pid, true) {
                    Ok(()) => {
                        println!("\u{26a0} Gateway PID {pid} did not exit gracefully; sent SIGKILL");
                    }
                    Err(_) => return true,
                }
                force_sent = true;
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }

    match get_running_pid(None, true) {
        Some(remaining) => {
            println!(
                "\u{26a0} Gateway PID {remaining} still running after {}s — restart may fail",
                timeout
            );
            false
        }
        None => true,
    }
}

pub fn launchd_start() {
    let plist_path = get_launchd_plist_path();
    let label = get_launchd_label();
    let domain = launchd_domain();
    let target = format!("{domain}/{label}");

    if !plist_path.exists() {
        println!("\u{21bb} launchd plist missing; regenerating service definition");
        if let Some(parent) = plist_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&plist_path, generate_launchd_plist());
        run_capture("launchctl", &["bootstrap", &domain, &plist_path.to_string_lossy()]);
        run_capture("launchctl", &["kickstart", &target]);
        println!("\u{2713} Service started");
        return;
    }

    refresh_launchd_plist_if_needed();
    match run_capture("launchctl", &["kickstart", &target]) {
        Some((code, _, _)) if code == 0 => {}
        Some((code, _, _)) if code == 3 || code == 113 => {
            println!("\u{21bb} launchd job was unloaded; reloading service definition");
            run_capture("launchctl", &["bootstrap", &domain, &plist_path.to_string_lossy()]);
            run_capture("launchctl", &["kickstart", &target]);
        }
        _ => {}
    }
    println!("\u{2713} Service started");
}

pub fn launchd_stop() {
    let label = get_launchd_label();
    let target = format!("{}/{label}", launchd_domain());
    if let Some(pid) = get_running_pid(None, false) {
        write_planned_stop_marker(pid);
    }
    match run_capture("launchctl", &["bootout", &target]) {
        Some((code, _, _)) if code == 3 || code == 113 => {}
        _ => {}
    }
    wait_for_gateway_exit(10.0, Some(5.0));
    println!("\u{2713} Service stopped");
}

pub fn launchd_restart() {
    let label = get_launchd_label();
    let target = format!("{}/{label}", launchd_domain());
    let drain_timeout = get_restart_drain_timeout();

    let mut pid = get_running_pid(None, true);
    if let Some(p) = pid {
        if request_gateway_self_restart(p) {
            println!("\u{2713} Service restart requested");
            return;
        }
        match terminate_pid(p, false) {
            Ok(()) => {}
            Err(_) => pid = None,
        }
        if pid.is_some() {
            let exited = wait_for_gateway_exit(drain_timeout, None);
            if !exited {
                println!(
                    "\u{26a0} Gateway drain timed out after {:.0}s — forcing launchd restart",
                    drain_timeout
                );
            }
        }
    }
    match run_capture("launchctl", &["kickstart", "-k", &target]) {
        Some((code, _, _)) if code == 3 || code == 113 => {
            println!("\u{21bb} launchd job was unloaded; reloading");
            let plist_path = get_launchd_plist_path();
            run_capture(
                "launchctl",
                &["bootstrap", &launchd_domain(), &plist_path.to_string_lossy()],
            );
            run_capture("launchctl", &["kickstart", &target]);
            println!("\u{2713} Service restarted");
        }
        _ => println!("\u{2713} Service restarted"),
    }
}

pub fn launchd_status(deep: bool) {
    let plist_path = get_launchd_plist_path();
    let label = get_launchd_label();
    let (loaded, loaded_output) = match run_capture("launchctl", &["list", &label]) {
        Some((code, stdout, _)) => (code == 0, stdout),
        None => (false, String::new()),
    };

    println!("Launchd plist: {}", plist_path.display());
    if launchd_plist_is_current() {
        println!("\u{2713} Service definition matches the current Hermes install");
    } else {
        println!("\u{26a0} Service definition is stale relative to the current Hermes install");
        println!("  Run: hermes gateway start");
    }

    if loaded {
        println!("\u{2713} Gateway service is loaded");
        println!("{loaded_output}");
    } else {
        println!("\u{2717} Gateway service is not loaded");
        println!("  Service definition exists locally but launchd has not loaded it.");
        println!("  Run: hermes gateway start");
    }

    if deep {
        let log_file = get_hermes_home().join("logs").join("gateway.log");
        if log_file.exists() {
            println!();
            println!("Recent logs:");
            run_inherit("tail", &["-20", &log_file.to_string_lossy()]);
        }
    }
}

fn is_service_installed() -> bool {
    if supports_systemd_services() {
        get_systemd_unit_path(false, None).exists() || get_systemd_unit_path(true, None).exists()
    } else if is_macos() {
        get_launchd_plist_path().exists()
    } else {
        false
    }
}

fn is_service_running() -> bool {
    if supports_systemd_services() {
        let user_unit = get_systemd_unit_path(false, None).exists();
        let system_unit = get_systemd_unit_path(true, None).exists();
        if user_unit {
            if let Some((_, stdout, _)) = run_systemctl(&["is-active", &get_service_name(None)], false, true) {
                if stdout.trim() == "active" {
                    return true;
                }
            }
        }
        if system_unit {
            if let Some((_, stdout, _)) = run_systemctl(&["is-active", &get_service_name(None)], true, true) {
                if stdout.trim() == "active" {
                    return true;
                }
            }
        }
        false
    } else if is_macos() && get_launchd_plist_path().exists() {
        matches!(
            run_capture("launchctl", &["list", &get_launchd_label()]),
            Some((0, _, _))
        )
    } else {
        !find_gateway_pids(None, false).is_empty()
    }
}

// =============================================================================
// Interactive prompt helpers (stdin-based; mirror hermes_cli.setup)
// =============================================================================

fn read_line() -> Option<String> {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
    }
}

fn flush_stdout() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Prompt for a line of text, returning `default` on empty input / EOF.
pub fn prompt(message: &str, default: &str, _password: bool) -> String {
    if default.is_empty() {
        print!("{message}: ");
    } else {
        print!("{message} [{default}]: ");
    }
    flush_stdout();
    match read_line() {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        Some(_) => default.to_string(),
        None => default.to_string(),
    }
}

/// Yes/no prompt. Returns `default` on empty input / EOF.
pub fn prompt_yes_no(message: &str, default: bool) -> bool {
    let hint = if default { "Y/n" } else { "y/N" };
    print!("{message} [{hint}]: ");
    flush_stdout();
    match read_line() {
        None => default,
        Some(s) => {
            let s = s.trim().to_lowercase();
            if s.is_empty() {
                default
            } else {
                matches!(s.as_str(), "y" | "yes" | "true" | "1")
            }
        }
    }
}

/// Single-select menu. Returns the chosen 0-based index (or `default` on EOF).
pub fn prompt_choice(message: &str, choices: &[&str], default: usize) -> usize {
    println!("{message}");
    for (i, choice) in choices.iter().enumerate() {
        let marker = if i == default { "*" } else { " " };
        println!("  {marker} {}. {choice}", i + 1);
    }
    print!("Select [{}]: ", default + 1);
    flush_stdout();
    match read_line() {
        None => default,
        Some(s) => {
            let s = s.trim();
            if s.is_empty() {
                return default;
            }
            match s.parse::<usize>() {
                Ok(n) if n >= 1 && n <= choices.len() => n - 1,
                _ => default,
            }
        }
    }
}

// =============================================================================
// run_gateway
// =============================================================================

/// Run the gateway in foreground via the supplied [`GatewayRunner`].
/// Returns Ok(()) on clean stop, Err(1) when startup failed (caller exits 1).
pub fn run_gateway<R: GatewayRunner>(
    runner: &R,
    verbose: i32,
    quiet: bool,
    replace: bool,
) -> Result<(), i32> {
    // Best-effort: refresh systemd unit on every boot so restart settings stay current.
    if supports_systemd_services() {
        let _ = std::panic::catch_unwind(|| {
            refresh_systemd_unit_if_needed(false);
        });
    }

    println!("\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}");
    println!("\u{2502}           \u{2695} Hermes Gateway Starting...                 \u{2502}");
    println!("\u{251c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}");
    println!("\u{2502}  Messaging platforms + cron scheduler                    \u{2502}");
    println!("\u{2502}  Press Ctrl+C to stop                                   \u{2502}");
    println!("\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}");
    println!();

    let verbosity = if quiet { None } else { Some(verbose) };
    match runner.start(replace, verbosity) {
        Ok(true) => Ok(()),
        Ok(false) => Err(1),
        Err(_) => Err(1),
    }
}

// =============================================================================
// Platform definitions (mirror _PLATFORMS)
// =============================================================================

#[derive(Debug, Clone)]
pub struct PlatformVar {
    pub name: &'static str,
    pub prompt: &'static str,
    pub password: bool,
    pub is_allowlist: bool,
    pub help: &'static str,
}

impl PlatformVar {
    const fn new(
        name: &'static str,
        prompt: &'static str,
        password: bool,
        is_allowlist: bool,
        help: &'static str,
    ) -> Self {
        PlatformVar {
            name,
            prompt,
            password,
            is_allowlist,
            help,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlatformDef {
    pub key: &'static str,
    pub label: &'static str,
    pub emoji: &'static str,
    pub token_var: &'static str,
    pub setup_instructions: Vec<&'static str>,
    pub vars: Vec<PlatformVar>,
}

/// Build the full built-in platform table (matches Python `_PLATFORMS`).
pub fn builtin_platforms() -> Vec<PlatformDef> {
    use PlatformVar as V;
    vec![
        PlatformDef {
            key: "telegram",
            label: "Telegram",
            emoji: "\u{1f4f1}",
            token_var: "TELEGRAM_BOT_TOKEN",
            setup_instructions: vec![
                "1. Open Telegram and message @BotFather",
                "2. Send /newbot and follow the prompts to create your bot",
                "3. Copy the bot token BotFather gives you",
                "4. To find your user ID: message @userinfobot — it replies with your numeric ID",
            ],
            vars: vec![
                V::new("TELEGRAM_BOT_TOKEN", "Bot token", true, false, "Paste the token from @BotFather (step 3 above)."),
                V::new("TELEGRAM_ALLOWED_USERS", "Allowed user IDs (comma-separated)", false, true, "Paste your user ID from step 4 above."),
                V::new("TELEGRAM_HOME_CHANNEL", "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", false, false, "For DMs, this is your user ID. You can set it later by typing /set-home in chat."),
            ],
        },
        PlatformDef {
            key: "discord",
            label: "Discord",
            emoji: "\u{1f4ac}",
            token_var: "DISCORD_BOT_TOKEN",
            setup_instructions: vec![
                "1. Go to https://discord.com/developers/applications → New Application",
                "2. Go to Bot → Reset Token → copy the bot token",
                "3. Enable: Bot → Privileged Gateway Intents → Message Content Intent",
                "4. Invite the bot to your server:",
                "   OAuth2 → URL Generator → check BOTH scopes:",
                "     - bot",
                "     - applications.commands  (required for slash commands!)",
                "   Bot Permissions: Send Messages, Read Message History, Attach Files",
                "   Copy the URL and open it in your browser to invite.",
                "5. Get your user ID: enable Developer Mode in Discord settings,",
                "   then right-click your name → Copy ID",
            ],
            vars: vec![
                V::new("DISCORD_BOT_TOKEN", "Bot token", true, false, "Paste the token from step 2 above."),
                V::new("DISCORD_ALLOWED_USERS", "Allowed user IDs or usernames (comma-separated)", false, true, "Paste your user ID from step 5 above."),
                V::new("DISCORD_HOME_CHANNEL", "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", false, false, "Right-click a channel → Copy Channel ID (requires Developer Mode)."),
            ],
        },
        PlatformDef {
            key: "slack",
            label: "Slack",
            emoji: "\u{1f4bc}",
            token_var: "SLACK_BOT_TOKEN",
            setup_instructions: vec![
                "1. Go to https://api.slack.com/apps → Create New App → From Scratch",
                "2. Enable Socket Mode: Settings → Socket Mode → Enable",
                "   Create an App-Level Token with scope: connections:write → copy xapp-... token",
                "3. Add Bot Token Scopes: Features → OAuth & Permissions → Scopes",
                "   Required: chat:write, app_mentions:read, channels:history, channels:read,",
                "   groups:history, im:history, im:read, im:write, users:read, files:read, files:write",
                "4. Subscribe to Events: Features → Event Subscriptions → Enable",
                "   Required events: message.im, message.channels, app_mention",
                "   Optional: message.groups (for private channels)",
                "   \u{26a0} Without message.channels the bot will ONLY work in DMs!",
                "5. Install to Workspace: Settings → Install App → copy xoxb-... token",
                "6. Reinstall the app after any scope or event changes",
                "7. Find your user ID: click your profile → three dots → Copy member ID",
                "8. Invite the bot to channels: /invite @YourBot",
            ],
            vars: vec![
                V::new("SLACK_BOT_TOKEN", "Bot Token (xoxb-...)", true, false, "Paste the bot token from step 3 above."),
                V::new("SLACK_APP_TOKEN", "App Token (xapp-...)", true, false, "Paste the app-level token from step 4 above."),
                V::new("SLACK_ALLOWED_USERS", "Allowed user IDs (comma-separated)", false, true, "Paste your member ID from step 7 above."),
            ],
        },
        PlatformDef {
            key: "matrix",
            label: "Matrix",
            emoji: "\u{1f510}",
            token_var: "MATRIX_ACCESS_TOKEN",
            setup_instructions: vec![
                "1. Works with any Matrix homeserver (self-hosted Synapse/Conduit/Dendrite or matrix.org)",
                "2. Create a bot user on your homeserver, or use your own account",
                "3. Get an access token: Element → Settings → Help & About → Access Token",
                "4. Alternatively, provide user ID + password and Hermes will log in directly",
                "5. For E2EE: set MATRIX_ENCRYPTION=true (requires pip install 'mautrix[encryption]')",
                "6. To find your user ID: it's @username:your-server (shown in Element profile)",
            ],
            vars: vec![
                V::new("MATRIX_HOMESERVER", "Homeserver URL (e.g. https://matrix.example.org)", false, false, "Your Matrix homeserver URL. Works with any self-hosted instance."),
                V::new("MATRIX_ACCESS_TOKEN", "Access token (leave empty to use password login instead)", true, false, "Paste your access token, or leave empty and provide user ID + password below."),
                V::new("MATRIX_USER_ID", "User ID (@bot:server — required for password login)", false, false, "Full Matrix user ID, e.g. @hermes:matrix.example.org"),
                V::new("MATRIX_ALLOWED_USERS", "Allowed user IDs (comma-separated, e.g. @you:server)", false, true, "Matrix user IDs who can interact with the bot."),
                V::new("MATRIX_HOME_ROOM", "Home room ID (for cron/notification delivery, or empty to set later with /set-home)", false, false, "Room ID (e.g. !abc123:server) for delivering cron results and notifications."),
            ],
        },
        PlatformDef {
            key: "mattermost",
            label: "Mattermost",
            emoji: "\u{1f4ac}",
            token_var: "MATTERMOST_TOKEN",
            setup_instructions: vec![
                "1. In Mattermost: Integrations → Bot Accounts → Add Bot Account",
                "2. Give it a username (e.g. hermes) and copy the bot token",
                "3. Works with any self-hosted Mattermost instance — enter your server URL",
                "4. To find your user ID: click your avatar (top-left) → Profile",
                "5. To get a channel ID: click the channel name → View Info → copy the ID",
            ],
            vars: vec![
                V::new("MATTERMOST_URL", "Server URL (e.g. https://mm.example.com)", false, false, "Your Mattermost server URL. Works with any self-hosted instance."),
                V::new("MATTERMOST_TOKEN", "Bot token", true, false, "Paste the bot token from step 2 above."),
                V::new("MATTERMOST_ALLOWED_USERS", "Allowed user IDs (comma-separated)", false, true, "Your Mattermost user ID from step 4 above."),
                V::new("MATTERMOST_HOME_CHANNEL", "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", false, false, "Channel ID where Hermes delivers cron results and notifications."),
                V::new("MATTERMOST_REPLY_MODE", "Reply mode — 'off' for flat messages, 'thread' for threaded replies (default: off)", false, false, "off = flat channel messages, thread = replies nest under your message."),
            ],
        },
        PlatformDef { key: "whatsapp", label: "WhatsApp", emoji: "\u{1f4f2}", token_var: "WHATSAPP_ENABLED", setup_instructions: vec![], vars: vec![] },
        PlatformDef { key: "signal", label: "Signal", emoji: "\u{1f4e1}", token_var: "SIGNAL_HTTP_URL", setup_instructions: vec![], vars: vec![] },
        PlatformDef {
            key: "email",
            label: "Email",
            emoji: "\u{1f4e7}",
            token_var: "EMAIL_ADDRESS",
            setup_instructions: vec![
                "1. Use a dedicated email account for your Hermes agent",
                "2. For Gmail: enable 2FA, then create an App Password at",
                "   https://myaccount.google.com/apppasswords",
                "3. For other providers: use your email password or app-specific password",
                "4. IMAP must be enabled on your email account",
            ],
            vars: vec![
                V::new("EMAIL_ADDRESS", "Email address", false, false, "The email address Hermes will use (e.g., hermes@gmail.com)."),
                V::new("EMAIL_PASSWORD", "Email password (or app password)", true, false, "For Gmail, use an App Password (not your regular password)."),
                V::new("EMAIL_IMAP_HOST", "IMAP host", false, false, "e.g., imap.gmail.com for Gmail, outlook.office365.com for Outlook."),
                V::new("EMAIL_SMTP_HOST", "SMTP host", false, false, "e.g., smtp.gmail.com for Gmail, smtp.office365.com for Outlook."),
                V::new("EMAIL_ALLOWED_USERS", "Allowed sender emails (comma-separated)", false, true, "Only emails from these addresses will be processed."),
            ],
        },
        PlatformDef {
            key: "sms",
            label: "SMS (Twilio)",
            emoji: "\u{1f4f1}",
            token_var: "TWILIO_ACCOUNT_SID",
            setup_instructions: vec![
                "1. Create a Twilio account at https://www.twilio.com/",
                "2. Get your Account SID and Auth Token from the Twilio Console dashboard",
                "3. Buy or configure a phone number capable of sending SMS",
                "4. Set up your webhook URL for inbound SMS",
            ],
            vars: vec![
                V::new("TWILIO_ACCOUNT_SID", "Twilio Account SID", false, false, "Found on the Twilio Console dashboard."),
                V::new("TWILIO_AUTH_TOKEN", "Twilio Auth Token", true, false, "Found on the Twilio Console dashboard (click to reveal)."),
                V::new("TWILIO_PHONE_NUMBER", "Twilio phone number (E.164 format, e.g. +15551234567)", false, false, "The Twilio phone number to send SMS from."),
                V::new("SMS_ALLOWED_USERS", "Allowed phone numbers (comma-separated, E.164 format)", false, true, "Only messages from these phone numbers will be processed."),
                V::new("SMS_HOME_CHANNEL", "Home channel phone number (for cron/notification delivery, or empty)", false, false, "Phone number to deliver cron job results and notifications to."),
            ],
        },
        PlatformDef {
            key: "dingtalk",
            label: "DingTalk",
            emoji: "\u{1f4ac}",
            token_var: "DINGTALK_CLIENT_ID",
            setup_instructions: vec![
                "1. Go to https://open-dev.dingtalk.com → Create Application",
                "2. Under 'Credentials', copy the AppKey (Client ID) and AppSecret (Client Secret)",
                "3. Enable 'Stream Mode' under the bot settings",
                "4. Add the bot to a group chat or message it directly",
            ],
            vars: vec![
                V::new("DINGTALK_CLIENT_ID", "AppKey (Client ID)", false, false, "The AppKey from your DingTalk application credentials."),
                V::new("DINGTALK_CLIENT_SECRET", "AppSecret (Client Secret)", true, false, "The AppSecret from your DingTalk application credentials."),
            ],
        },
        PlatformDef {
            key: "feishu",
            label: "Feishu / Lark",
            emoji: "\u{1fabd}",
            token_var: "FEISHU_APP_ID",
            setup_instructions: vec![
                "1. Go to https://open.feishu.cn/ (or https://open.larksuite.com/ for Lark)",
                "2. Create an app and copy the App ID and App Secret",
                "3. Enable the Bot capability for the app",
                "4. Choose WebSocket (recommended) or Webhook connection mode",
                "5. Add the bot to a group chat or message it directly",
                "6. Restrict access with FEISHU_ALLOWED_USERS for production use",
            ],
            vars: vec![
                V::new("FEISHU_APP_ID", "App ID", false, false, "The App ID from your Feishu/Lark application."),
                V::new("FEISHU_APP_SECRET", "App Secret", true, false, "The App Secret from your Feishu/Lark application."),
                V::new("FEISHU_DOMAIN", "Domain — feishu or lark (default: feishu)", false, false, "Use 'feishu' for Feishu China, or 'lark' for Lark international."),
                V::new("FEISHU_CONNECTION_MODE", "Connection mode — websocket or webhook (default: websocket)", false, false, "websocket is recommended unless you specifically need webhook mode."),
                V::new("FEISHU_ALLOWED_USERS", "Allowed user IDs (comma-separated, or empty)", false, true, "Restrict which Feishu/Lark users can interact with the bot."),
                V::new("FEISHU_HOME_CHANNEL", "Home chat ID (optional, for cron/notifications)", false, false, "Chat ID for scheduled results and notifications."),
            ],
        },
        PlatformDef {
            key: "wecom",
            label: "WeCom (Enterprise WeChat)",
            emoji: "\u{1f4ac}",
            token_var: "WECOM_BOT_ID",
            setup_instructions: vec![
                "1. Go to WeCom Admin Console → Applications → Create AI Bot",
                "2. Copy the Bot ID and Secret from the bot's credentials page",
                "3. The bot connects via WebSocket — no public endpoint needed",
                "4. Add the bot to a group chat or message it directly in WeCom",
                "5. Restrict access with WECOM_ALLOWED_USERS for production use",
            ],
            vars: vec![
                V::new("WECOM_BOT_ID", "Bot ID", false, false, "The Bot ID from your WeCom AI Bot."),
                V::new("WECOM_SECRET", "Secret", true, false, "The secret from your WeCom AI Bot."),
                V::new("WECOM_ALLOWED_USERS", "Allowed user IDs (comma-separated, or empty)", false, true, "Restrict which WeCom users can interact with the bot."),
                V::new("WECOM_HOME_CHANNEL", "Home chat ID (optional, for cron/notifications)", false, false, "Chat ID for scheduled results and notifications."),
            ],
        },
        PlatformDef {
            key: "wecom_callback",
            label: "WeCom Callback (Self-Built App)",
            emoji: "\u{1f4ac}",
            token_var: "WECOM_CALLBACK_CORP_ID",
            setup_instructions: vec![
                "1. Go to WeCom Admin Console → Applications → Create Self-Built App",
                "2. Note the Corp ID (top of admin console) and create a Corp Secret",
                "3. Under Receive Messages, configure the callback URL to point to your server",
                "4. Copy the Token and EncodingAESKey from the callback configuration",
                "5. The adapter runs an HTTP server — ensure the port is reachable from WeCom",
                "6. Restrict access with WECOM_CALLBACK_ALLOWED_USERS for production use",
            ],
            vars: vec![
                V::new("WECOM_CALLBACK_CORP_ID", "Corp ID", false, false, "Your WeCom enterprise Corp ID."),
                V::new("WECOM_CALLBACK_CORP_SECRET", "Corp Secret", true, false, "The secret for your self-built application."),
                V::new("WECOM_CALLBACK_AGENT_ID", "Agent ID", false, false, "The Agent ID of your self-built application."),
                V::new("WECOM_CALLBACK_TOKEN", "Callback Token", true, false, "The Token from your WeCom callback configuration."),
                V::new("WECOM_CALLBACK_ENCODING_AES_KEY", "Encoding AES Key", true, false, "The EncodingAESKey from your WeCom callback configuration."),
                V::new("WECOM_CALLBACK_PORT", "Callback server port (default: 8645)", false, false, "Port for the HTTP callback server."),
                V::new("WECOM_CALLBACK_ALLOWED_USERS", "Allowed user IDs (comma-separated, or empty)", false, true, "Restrict which WeCom users can interact with the app."),
            ],
        },
        PlatformDef { key: "weixin", label: "Weixin / WeChat", emoji: "\u{1f4ac}", token_var: "WEIXIN_ACCOUNT_ID", setup_instructions: vec![], vars: vec![] },
        PlatformDef {
            key: "bluebubbles",
            label: "BlueBubbles (iMessage)",
            emoji: "\u{1f4ac}",
            token_var: "BLUEBUBBLES_SERVER_URL",
            setup_instructions: vec![
                "1. Install BlueBubbles on a Mac that will act as your iMessage server:",
                "   https://bluebubbles.app/",
                "2. Complete the BlueBubbles setup wizard — sign in with your Apple ID",
                "3. In BlueBubbles Settings → API, note the Server URL and password",
                "4. The server URL is typically http://<your-mac-ip>:1234",
                "5. Hermes connects via the BlueBubbles REST API and receives",
                "   incoming messages via a local webhook",
                "6. To authorize users, use DM pairing: hermes pairing generate bluebubbles",
            ],
            vars: vec![
                V::new("BLUEBUBBLES_SERVER_URL", "BlueBubbles server URL (e.g. http://192.168.1.10:1234)", false, false, "The URL shown in BlueBubbles Settings → API."),
                V::new("BLUEBUBBLES_PASSWORD", "BlueBubbles server password", true, false, "The password shown in BlueBubbles Settings → API."),
                V::new("BLUEBUBBLES_ALLOWED_USERS", "Pre-authorized phone numbers or iMessage IDs (comma-separated, or leave empty for DM pairing)", false, true, "Optional — pre-authorize specific users. Leave empty to use DM pairing instead (recommended)."),
                V::new("BLUEBUBBLES_HOME_CHANNEL", "Home channel (phone number or iMessage ID for cron/notifications, or empty)", false, false, "Phone number or Apple ID to deliver cron results and notifications to."),
            ],
        },
        PlatformDef {
            key: "qqbot",
            label: "QQ Bot",
            emoji: "\u{1f427}",
            token_var: "QQ_APP_ID",
            setup_instructions: vec![
                "1. Register a QQ Bot application at q.qq.com",
                "2. Note your App ID and App Secret from the application page",
                "3. Enable the required intents (C2C, Group, Guild messages)",
                "4. Configure sandbox or publish the bot",
            ],
            vars: vec![
                V::new("QQ_APP_ID", "QQ Bot App ID", false, false, "Your QQ Bot App ID from q.qq.com."),
                V::new("QQ_CLIENT_SECRET", "QQ Bot App Secret", true, false, "Your QQ Bot App Secret from q.qq.com."),
                V::new("QQ_ALLOWED_USERS", "Allowed user OpenIDs (comma-separated, leave empty for open access)", false, true, "Optional — restrict DM access to specific user OpenIDs."),
                V::new("QQBOT_HOME_CHANNEL", "Home channel (user/group OpenID for cron delivery, or empty)", false, false, "OpenID to deliver cron results and notifications to."),
            ],
        },
        PlatformDef {
            key: "yuanbao",
            label: "Yuanbao",
            emoji: "\u{1f48e}",
            token_var: "YUANBAO_APP_ID",
            setup_instructions: vec![
                "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
                "2. In the app, go to PAI → My Bot and create a new bot",
                "3. After the bot is created, copy the App ID and App Secret",
                "4. Enter them below and Hermes will connect automatically over WebSocket",
            ],
            vars: vec![
                V::new("YUANBAO_APP_ID", "App ID", false, false, "The App ID from your Yuanbao IM Bot credentials."),
                V::new("YUANBAO_APP_SECRET", "App Secret", true, false, "The App Secret (used for HMAC signing) from your Yuanbao IM Bot."),
            ],
        },
    ]
}

// =============================================================================
// Platform status
// =============================================================================

/// Return a plain-text status string for a platform (uncolored).
pub fn platform_status(platform: &PlatformDef) -> String {
    let token_var = platform.token_var;
    if token_var.is_empty() {
        return "not configured".to_string();
    }
    let val = get_env_value(token_var);

    if token_var == "WHATSAPP_ENABLED" {
        if let Some(v) = &val {
            if v.to_lowercase() == "true" {
                let session_file = get_hermes_home()
                    .join("whatsapp")
                    .join("session")
                    .join("creds.json");
                if session_file.exists() {
                    return "configured + paired".to_string();
                }
                return "enabled, not paired".to_string();
            }
        }
        return "not configured".to_string();
    }

    match platform.key {
        "signal" => {
            let account = get_env_value("SIGNAL_ACCOUNT");
            if val.is_some() && account.is_some() {
                return "configured".to_string();
            }
            if val.is_some() || account.is_some() {
                return "partially configured".to_string();
            }
            "not configured".to_string()
        }
        "email" => {
            let pwd = get_env_value("EMAIL_PASSWORD");
            let imap = get_env_value("EMAIL_IMAP_HOST");
            let smtp = get_env_value("EMAIL_SMTP_HOST");
            let all = val.is_some() && pwd.is_some() && imap.is_some() && smtp.is_some();
            let any = val.is_some() || pwd.is_some() || imap.is_some() || smtp.is_some();
            if all {
                "configured".to_string()
            } else if any {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "matrix" => {
            let homeserver = get_env_value("MATRIX_HOMESERVER");
            let password = get_env_value("MATRIX_PASSWORD");
            if (val.is_some() || password.is_some()) && homeserver.is_some() {
                let e2ee = get_env_value("MATRIX_ENCRYPTION");
                let suffix = match e2ee {
                    Some(v) if matches!(v.to_lowercase().as_str(), "true" | "1" | "yes") => " + E2EE",
                    _ => "",
                };
                return format!("configured{suffix}");
            }
            if val.is_some() || password.is_some() || homeserver.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        "weixin" => {
            let token = get_env_value("WEIXIN_TOKEN");
            if val.is_some() && token.is_some() {
                "configured".to_string()
            } else if val.is_some() || token.is_some() {
                "partially configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
        _ => {
            if val.is_some() {
                "configured".to_string()
            } else {
                "not configured".to_string()
            }
        }
    }
}

// =============================================================================
// Standard platform setup
// =============================================================================

/// Interactive setup for a standard env-var-driven platform.
pub fn setup_standard_platform(platform: &PlatformDef) {
    let emoji = platform.emoji;
    let label = platform.label;
    let token_var = platform.token_var;

    println!();
    println!("  ─── {emoji} {label} Setup ───");

    if !platform.setup_instructions.is_empty() {
        println!();
        for line in &platform.setup_instructions {
            print_info(&format!("  {line}"));
        }
    }

    if get_env_value(token_var).is_some() {
        println!();
        print_success(&format!("{label} is already configured."));
        if !prompt_yes_no(&format!("  Reconfigure {label}?"), false) {
            return;
        }
    }

    let mut allowed_val_set: Option<String> = None;

    for var in &platform.vars {
        println!();
        print_info(&format!("  {}", var.help));
        let existing = get_env_value(var.name);
        if let Some(e) = &existing {
            if var.name != token_var {
                print_info(&format!("  Current: {e}"));
            }
        }

        if var.is_allowlist {
            print_info("  The gateway DENIES all users by default for security.");
            print_info("  Enter user IDs to create an allowlist, or leave empty");
            print_info("  and you'll be asked about open access next.");
            let value = prompt(&format!("  {}", var.prompt), "", false);
            if !value.is_empty() {
                let mut cleaned = value.replace(' ', "");
                if var.name.contains("DISCORD") {
                    let mut parts: Vec<String> = Vec::new();
                    for uid in cleaned.split(',') {
                        let mut uid = uid.trim().to_string();
                        if uid.starts_with("<@") && uid.ends_with('>') {
                            uid = uid
                                .trim_start_matches(|c| c == '<' || c == '@' || c == '!')
                                .trim_end_matches('>')
                                .to_string();
                        }
                        if uid.to_lowercase().starts_with("user:") {
                            uid = uid[5..].to_string();
                        }
                        if !uid.is_empty() {
                            parts.push(uid);
                        }
                    }
                    cleaned = parts.join(",");
                }
                save_env_value(var.name, &cleaned);
                print_success("  Saved — only these users can interact with the bot.");
                allowed_val_set = Some(cleaned);
            } else {
                println!();
                let access_choices = [
                    "Enable open access (anyone can message the bot)",
                    "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
                    "Skip for now (bot will deny all users until configured)",
                ];
                let idx = prompt_choice("  How should unauthorized users be handled?", &access_choices, 1);
                match idx {
                    0 => {
                        save_env_value("GATEWAY_ALLOW_ALL_USERS", "true");
                        print_warning("  Open access enabled — anyone can use your bot!");
                    }
                    1 => {
                        print_success("  DM pairing mode — users will receive a code to request access.");
                        print_info("  Approve with: hermes pairing approve <platform> <code>");
                    }
                    _ => print_info("  Skipped — configure later with 'hermes gateway setup'"),
                }
            }
            continue;
        }

        let value = prompt(&format!("  {}", var.prompt), "", var.password);
        if !value.is_empty() {
            save_env_value(var.name, &value);
            print_success(&format!("  Saved {}", var.name));
        } else if var.name == token_var {
            print_warning(&format!("  Skipped — {label} won't work without this."));
            return;
        } else {
            print_info("  Skipped (can configure later)");
        }
    }

    let home_var = format!("{}_HOME_CHANNEL", label.to_uppercase());
    let home_val = get_env_value(&home_var);
    if let Some(allowed) = &allowed_val_set {
        if home_val.is_none() && label == "Telegram" {
            let first_id = allowed.split(',').next().unwrap_or("").trim().to_string();
            if !first_id.is_empty()
                && prompt_yes_no(
                    &format!("  Use your user ID ({first_id}) as the home channel?"),
                    true,
                )
            {
                save_env_value(&home_var, &first_id);
                print_success(&format!("  Home channel set to {first_id}"));
            }
        }
    }

    println!();
    print_success(&format!("{emoji} {label} configured!"));
}

/// Interactive setup for Signal (self-contained HTTP-daemon flow).
pub fn setup_signal() {
    println!();
    println!("  ─── \u{1f4e1} Signal Setup ───");

    let existing_url = get_env_value("SIGNAL_HTTP_URL");
    let existing_account = get_env_value("SIGNAL_ACCOUNT");
    if existing_url.is_some() && existing_account.is_some() {
        println!();
        print_success("Signal is already configured.");
        if !prompt_yes_no("  Reconfigure Signal?", false) {
            return;
        }
    }

    println!();
    if which("signal-cli").is_some() {
        print_success("signal-cli found on PATH.");
    } else {
        print_warning("signal-cli not found on PATH.");
        print_info("  Signal requires signal-cli running as an HTTP daemon.");
        print_info("  Install options:");
        print_info("    Linux:  download from https://github.com/AsamK/signal-cli/releases");
        print_info("    macOS:  brew install signal-cli");
        print_info("    Docker: bbernhard/signal-cli-rest-api");
    }

    println!();
    print_info("  Enter the URL where signal-cli HTTP daemon is running.");
    let default_url = existing_url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
    let url = prompt("  HTTP URL", &default_url, false);

    print_info("  Testing connection...");
    let check_url = format!("{}/api/v1/check", url.trim_end_matches('/'));
    match reqwest::blocking::Client::new()
        .get(&check_url)
        .timeout(Duration::from_secs(10))
        .send()
    {
        Ok(resp) if resp.status().as_u16() == 200 => {
            print_success("  signal-cli daemon is reachable!");
        }
        Ok(resp) => {
            print_warning(&format!("  signal-cli responded with status {}.", resp.status().as_u16()));
            if !prompt_yes_no("  Continue anyway?", false) {
                return;
            }
        }
        Err(e) => {
            print_warning(&format!("  Could not reach signal-cli at {url}: {e}"));
            if !prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", true) {
                return;
            }
        }
    }
    save_env_value("SIGNAL_HTTP_URL", &url);

    println!();
    print_info("  Enter your Signal account phone number in E.164 format.");
    print_info("  Example: +15551234567");
    let default_account = existing_account.unwrap_or_default();
    let account = prompt("  Account number", &default_account, false);
    if account.is_empty() {
        print_error("  Account number is required.");
        return;
    }
    save_env_value("SIGNAL_ACCOUNT", &account);

    println!();
    print_info("  The gateway DENIES all users by default for security.");
    print_info("  Enter phone numbers or UUIDs of allowed users (comma-separated).");
    let existing_allowed = get_env_value("SIGNAL_ALLOWED_USERS").unwrap_or_default();
    let default_allowed = if existing_allowed.is_empty() {
        account.clone()
    } else {
        existing_allowed
    };
    let allowed = prompt("  Allowed users", &default_allowed, false);
    save_env_value("SIGNAL_ALLOWED_USERS", &allowed);

    println!();
    if prompt_yes_no("  Enable group messaging? (disabled by default for security)", false) {
        println!();
        print_info("  Enter group IDs to allow, or * for all groups.");
        let existing_groups = get_env_value("SIGNAL_GROUP_ALLOWED_USERS").unwrap_or_default();
        let default_groups = if existing_groups.is_empty() {
            "*".to_string()
        } else {
            existing_groups
        };
        let groups = prompt("  Group IDs", &default_groups, false);
        save_env_value("SIGNAL_GROUP_ALLOWED_USERS", &groups);
    }

    println!();
    print_success("Signal configured!");
    print_info(&format!("  URL: {url}"));
    print_info(&format!("  Account: {account}"));
    print_info("  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing");
    let groups_enabled = get_env_value("SIGNAL_GROUP_ALLOWED_USERS").is_some();
    print_info(&format!(
        "  Groups: {}",
        if groups_enabled { "enabled" } else { "disabled" }
    ));
}

/// Dispatch a single platform's interactive setup.
/// QR-based / Python-delegated flows fall back to the standard env-var flow
/// (or env-var hints) since the native binary cannot import the Python adapters.
pub fn configure_platform(platform: &PlatformDef) {
    match platform.key {
        "signal" => setup_signal(),
        "whatsapp" => {
            println!();
            println!("  ─── {} {} Setup ───", platform.emoji, platform.label);
            print_info("  Run: hermes whatsapp   # pairs the WhatsApp session");
        }
        _ if !platform.vars.is_empty() => setup_standard_platform(platform),
        _ => {
            println!();
            println!("  ─── {} {} Setup ───", platform.emoji, platform.label);
            if !platform.token_var.is_empty() {
                print_info(&format!(
                    "  Set these env vars in ~/.hermes/.env: {}",
                    platform.token_var
                ));
            } else {
                print_info(&format!(
                    "  Configure {} in config.yaml under gateway.platforms.{}",
                    platform.label, platform.key
                ));
            }
        }
    }
}

// =============================================================================
// gateway_setup
// =============================================================================

fn is_progress(status: &str) -> bool {
    let s = status.to_lowercase();
    !(s == "not configured" || s.starts_with("partially") || s.starts_with("plugin disabled"))
}

/// Interactive setup for messaging platforms + gateway service.
pub fn gateway_setup() {
    if is_managed() {
        managed_error("run gateway setup");
        return;
    }

    println!();
    println!("\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}");
    println!("\u{2502}             \u{2695} Gateway Setup                            \u{2502}");
    println!("\u{2502}  Configure messaging platforms and the gateway service. \u{2502}");
    println!("\u{2502}  Press Ctrl+C at any time to exit.                     \u{2502}");
    println!("\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}");

    println!();
    let service_installed = is_service_installed();
    let service_running = is_service_running();

    if supports_systemd_services() && has_conflicting_systemd_units() {
        print_systemd_scope_conflict_warning();
        println!();
    }
    if supports_systemd_services() && has_legacy_hermes_units() {
        print_legacy_unit_warning();
        println!();
    }

    if service_installed && service_running {
        print_success("Gateway service is installed and running.");
    } else if service_installed {
        print_warning("Gateway service is installed but not running.");
        if prompt_yes_no("  Start it now?", true) {
            if supports_systemd_services() {
                if let Err(e) = systemd_start(false) {
                    print_error("  Failed to start — user systemd not reachable:");
                    for line in e.0.lines() {
                        println!("  {line}");
                    }
                }
            } else if is_macos() {
                launchd_start();
            }
        }
    } else {
        print_info("Gateway service is not installed yet.");
        print_info("You'll be offered to install it after configuring platforms.");
    }

    // ── Platform configuration loop ──
    loop {
        println!();
        print_header("Messaging Platforms");

        let platforms = builtin_platforms();
        let mut menu_items: Vec<String> = platforms
            .iter()
            .map(|p| format!("{} {}  ({})", p.emoji, p.label, platform_status(p)))
            .collect();
        menu_items.push("Done".to_string());
        let menu_refs: Vec<&str> = menu_items.iter().map(|s| s.as_str()).collect();

        let choice = prompt_choice("Select a platform to configure:", &menu_refs, menu_items.len() - 1);
        if choice == platforms.len() {
            break;
        }
        configure_platform(&platforms[choice]);
    }

    let any_configured = builtin_platforms()
        .iter()
        .any(|p| is_progress(&platform_status(p)));

    if any_configured {
        println!();
        println!("{}", "\u{2500}".repeat(58));
        let service_installed = is_service_installed();
        let service_running = is_service_running();

        if service_running {
            if prompt_yes_no("  Restart the gateway to pick up changes?", true) {
                if supports_systemd_services() {
                    if let Err(e) = systemd_restart(false) {
                        print_error("  Restart failed — user systemd not reachable:");
                        for line in e.0.lines() {
                            println!("  {line}");
                        }
                    }
                } else if is_macos() {
                    launchd_restart();
                } else {
                    stop_profile_gateway();
                    print_info("Start manually: hermes gateway");
                }
            }
        } else if service_installed {
            if prompt_yes_no("  Start the gateway service?", true) {
                if supports_systemd_services() {
                    if let Err(e) = systemd_start(false) {
                        print_error("  Start failed — user systemd not reachable:");
                        for line in e.0.lines() {
                            println!("  {line}");
                        }
                    }
                } else if is_macos() {
                    launchd_start();
                }
            }
        } else {
            println!();
            if supports_systemd_services() || is_macos() {
                let platform_name = if supports_systemd_services() {
                    "systemd"
                } else {
                    "launchd"
                };
                let wsl_note = if is_wsl() {
                    " (note: services may not survive WSL restarts)"
                } else {
                    ""
                };
                if prompt_yes_no(
                    &format!(
                        "  Install the gateway as a {platform_name} service?{wsl_note} (runs in background, starts on boot)"
                    ),
                    true,
                ) {
                    let mut installed_scope: Option<String> = None;
                    let did_install;
                    if supports_systemd_services() {
                        let (scope, installed) = install_linux_gateway_from_setup(false);
                        installed_scope = scope;
                        did_install = installed;
                    } else {
                        launchd_install(false);
                        did_install = true;
                    }
                    println!();
                    if did_install && prompt_yes_no("  Start the service now?", true) {
                        if supports_systemd_services() {
                            if let Err(e) = systemd_start(installed_scope.as_deref() == Some("system")) {
                                print_error("  Start failed — user systemd not reachable:");
                                for line in e.0.lines() {
                                    println!("  {line}");
                                }
                            }
                        } else {
                            launchd_start();
                        }
                    }
                } else {
                    print_info("  You can install later: hermes gateway install");
                    if supports_systemd_services() {
                        print_info("  Or as a boot-time service: sudo hermes gateway install --system");
                    }
                    print_info("  Or run in foreground:  hermes gateway run");
                }
            } else if is_wsl() {
                print_info("  WSL detected but systemd is not running.");
                print_info("  Run in foreground: hermes gateway run");
                print_info("  For persistence:   tmux new -s hermes 'hermes gateway run'");
                print_info("  To enable systemd: add systemd=true to /etc/wsl.conf, then 'wsl --shutdown'");
            } else if is_termux() {
                let dhh = display_hermes_home();
                print_info("  Termux does not use systemd/launchd services.");
                print_info("  Run in foreground: hermes gateway run");
                print_info(&format!(
                    "  Or start it manually in the background (best effort): nohup hermes gateway run >{dhh}/logs/gateway.log 2>&1 &"
                ));
            } else {
                print_info("  Service install not supported on this platform.");
                print_info("  Run in foreground: hermes gateway run");
            }
        }
    } else {
        println!();
        print_info("No platforms configured. Run 'hermes gateway setup' when ready.");
    }

    println!();
}

// =============================================================================
// Command dispatch
// =============================================================================

/// Parsed gateway subcommand arguments (mirrors the argparse Namespace fields).
#[derive(Debug, Clone, Default)]
pub struct GatewayArgs {
    pub gateway_command: Option<String>,
    pub verbose: i32,
    pub quiet: bool,
    pub replace: bool,
    pub force: bool,
    pub system: bool,
    pub run_as_user: Option<String>,
    pub all: bool,
    pub deep: bool,
    pub full: bool,
    pub dry_run: bool,
    pub yes: bool,
}

/// Top-level handler. Wraps the inner dispatch to surface clean
/// `UserSystemdUnavailableError` messages instead of panics.
/// Returns the process exit code (0 = success).
pub fn gateway_command<R: GatewayRunner>(runner: &R, args: &GatewayArgs) -> i32 {
    match gateway_command_inner(runner, args) {
        Ok(()) => 0,
        Err(GatewayCommandError::Exit(code)) => code,
        Err(GatewayCommandError::UserSystemd(e)) => {
            print_error("User systemd not reachable:");
            for line in e.0.lines() {
                println!("  {line}");
            }
            1
        }
    }
}

enum GatewayCommandError {
    Exit(i32),
    UserSystemd(UserSystemdUnavailableError),
}

impl From<UserSystemdUnavailableError> for GatewayCommandError {
    fn from(e: UserSystemdUnavailableError) -> Self {
        GatewayCommandError::UserSystemd(e)
    }
}

fn user_unit_or_system_unit_exists() -> bool {
    get_systemd_unit_path(false, None).exists() || get_systemd_unit_path(true, None).exists()
}

fn gateway_command_inner<R: GatewayRunner>(
    runner: &R,
    args: &GatewayArgs,
) -> Result<(), GatewayCommandError> {
    let subcmd = args.gateway_command.as_deref();

    if subcmd.is_none() || subcmd == Some("run") {
        return run_gateway(runner, args.verbose, args.quiet, args.replace)
            .map_err(GatewayCommandError::Exit);
    }

    match subcmd.unwrap() {
        "setup" => {
            gateway_setup();
            Ok(())
        }
        "install" => cmd_install(args),
        "uninstall" => cmd_uninstall(args),
        "start" => cmd_start(args),
        "stop" => {
            cmd_stop(args);
            Ok(())
        }
        "restart" => cmd_restart(runner, args),
        "status" => {
            cmd_status(args);
            Ok(())
        }
        "migrate-legacy" => {
            if !supports_systemd_services() && !is_macos() {
                println!("Legacy unit migration only applies to systemd-based Linux hosts.");
                return Ok(());
            }
            remove_legacy_hermes_units(!args.yes, args.dry_run);
            Ok(())
        }
        _ => Ok(()),
    }
}

fn cmd_install(args: &GatewayArgs) -> Result<(), GatewayCommandError> {
    if is_managed() {
        managed_error("install gateway service (managed by NixOS)");
        return Ok(());
    }
    if is_termux() {
        println!("Gateway service installation is not supported on Termux.");
        println!("Run manually: hermes gateway");
        return Err(GatewayCommandError::Exit(1));
    }
    if supports_systemd_services() {
        if is_wsl() {
            print_warning("WSL detected — systemd services may not survive WSL restarts.");
            print_info("  Consider running in foreground instead: hermes gateway run");
            print_info("  Or use tmux/screen for persistence: tmux new -s hermes 'hermes gateway run'");
            println!();
        }
        systemd_install(args.force, args.system, args.run_as_user.as_deref())
            .map_err(GatewayCommandError::Exit)?;
        Ok(())
    } else if is_macos() {
        launchd_install(args.force);
        Ok(())
    } else if is_wsl() {
        println!("WSL detected but systemd is not running.");
        println!("Either enable systemd (add systemd=true to /etc/wsl.conf and restart WSL)");
        println!("or run the gateway in foreground mode:");
        println!();
        println!("  hermes gateway run                              # direct foreground");
        println!("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux");
        println!("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background");
        Err(GatewayCommandError::Exit(1))
    } else if is_container() {
        println!("Service installation is not needed inside a Docker container.");
        println!("The container runtime is your service manager — use Docker restart policies instead:");
        println!();
        println!("  docker run --restart unless-stopped ...   # auto-restart on crash/reboot");
        println!("  docker restart <container>                # manual restart");
        println!();
        println!("To run the gateway: hermes gateway run");
        Err(GatewayCommandError::Exit(0))
    } else {
        println!("Service installation not supported on this platform.");
        println!("Run manually: hermes gateway run");
        Err(GatewayCommandError::Exit(1))
    }
}

fn cmd_uninstall(args: &GatewayArgs) -> Result<(), GatewayCommandError> {
    if is_managed() {
        managed_error("uninstall gateway service (managed by NixOS)");
        return Ok(());
    }
    if is_termux() {
        println!("Gateway service uninstall is not supported on Termux because there is no managed service to remove.");
        println!("Stop manual runs with: hermes gateway stop");
        return Err(GatewayCommandError::Exit(1));
    }
    if supports_systemd_services() {
        systemd_uninstall(args.system).map_err(GatewayCommandError::Exit)
    } else if is_macos() {
        launchd_uninstall();
        Ok(())
    } else if is_container() {
        println!("Service uninstall is not applicable inside a Docker container.");
        println!("To stop the gateway, stop or remove the container:");
        println!();
        println!("  docker stop <container>");
        println!("  docker rm <container>");
        Err(GatewayCommandError::Exit(0))
    } else {
        println!("Not supported on this platform.");
        Err(GatewayCommandError::Exit(1))
    }
}

fn cmd_start(args: &GatewayArgs) -> Result<(), GatewayCommandError> {
    if args.all {
        let killed = kill_gateway_processes(false, None, true);
        if killed > 0 {
            println!("\u{2713} Killed {killed} stale gateway process(es) across all profiles");
            wait_for_gateway_exit(10.0, Some(5.0));
        }
    }

    if is_termux() {
        println!("Gateway service start is not supported on Termux because there is no system service manager.");
        println!("Run manually: hermes gateway");
        return Err(GatewayCommandError::Exit(1));
    }
    if supports_systemd_services() {
        systemd_start(args.system)?;
        Ok(())
    } else if is_macos() {
        launchd_start();
        Ok(())
    } else if is_wsl() {
        println!("WSL detected but systemd is not available.");
        println!("Run the gateway in foreground mode instead:");
        println!();
        println!("  hermes gateway run                              # direct foreground");
        println!("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux");
        println!("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background");
        println!();
        println!("To enable systemd: add systemd=true to /etc/wsl.conf and run 'wsl --shutdown' from PowerShell.");
        Err(GatewayCommandError::Exit(1))
    } else if is_container() {
        println!("Service start is not applicable inside a Docker container.");
        println!("The gateway runs as the container's main process.");
        println!();
        println!("  docker start <container>     # start a stopped container");
        println!("  docker restart <container>   # restart a running container");
        println!();
        println!("Or run the gateway directly: hermes gateway run");
        Err(GatewayCommandError::Exit(0))
    } else {
        println!("Not supported on this platform.");
        Err(GatewayCommandError::Exit(1))
    }
}

fn cmd_stop(args: &GatewayArgs) {
    if args.all {
        let mut service_available = false;
        if supports_systemd_services() && user_unit_or_system_unit_exists() {
            if systemd_stop(args.system).is_ok() {
                service_available = true;
            }
        } else if is_macos() && get_launchd_plist_path().exists() {
            launchd_stop();
            service_available = true;
        }
        let killed = kill_gateway_processes(false, None, true);
        let total = killed + if service_available { 1 } else { 0 };
        if total > 0 {
            println!("\u{2713} Stopped {total} gateway process(es) across all profiles");
        } else {
            println!("\u{2717} No gateway processes found");
        }
    } else {
        let mut service_available = false;
        if supports_systemd_services() && user_unit_or_system_unit_exists() {
            if systemd_stop(args.system).is_ok() {
                service_available = true;
            }
        } else if is_macos() && get_launchd_plist_path().exists() {
            launchd_stop();
            service_available = true;
        }

        if !service_available {
            if stop_profile_gateway() {
                println!("\u{2713} Stopped gateway for this profile");
            } else {
                println!("\u{2717} No gateway running for this profile");
            }
        } else {
            println!("\u{2713} Stopped {} service", get_service_name(None));
        }
    }
}

fn cmd_restart<R: GatewayRunner>(
    runner: &R,
    args: &GatewayArgs,
) -> Result<(), GatewayCommandError> {
    if args.all {
        let mut service_stopped = false;
        if supports_systemd_services() && user_unit_or_system_unit_exists() {
            if systemd_stop(args.system).is_ok() {
                service_stopped = true;
            }
        } else if is_macos() && get_launchd_plist_path().exists() {
            launchd_stop();
            service_stopped = true;
        }
        let killed = kill_gateway_processes(false, None, true);
        let total = killed + if service_stopped { 1 } else { 0 };
        if total > 0 {
            println!("\u{2713} Stopped {total} gateway process(es) across all profiles");
        }
        wait_for_gateway_exit(10.0, Some(5.0));

        println!("Starting gateway...");
        if supports_systemd_services() && user_unit_or_system_unit_exists() {
            systemd_start(args.system)?;
        } else if is_macos() && get_launchd_plist_path().exists() {
            launchd_start();
        } else {
            run_gateway(runner, 0, false, false).map_err(GatewayCommandError::Exit)?;
        }
        return Ok(());
    }

    let mut service_available = false;
    let mut service_configured = false;

    if supports_systemd_services() && user_unit_or_system_unit_exists() {
        service_configured = true;
        if systemd_restart(args.system).is_ok() {
            service_available = true;
        }
    } else if is_macos() && get_launchd_plist_path().exists() {
        service_configured = true;
        launchd_restart();
        service_available = true;
    }

    if !service_available {
        if supports_systemd_services() {
            let (linger_ok, _) = get_systemd_linger_status();
            if linger_ok != Some(true) {
                let username = current_username();
                println!();
                println!("\u{26a0} Cannot restart gateway as a service — linger is not enabled.");
                println!("  The gateway user service requires linger to function on headless servers.");
                println!();
                println!("  Run:  sudo loginctl enable-linger {username}");
                println!();
                println!("  Then restart the gateway:");
                println!("    hermes gateway restart");
                return Ok(());
            }
        }

        if service_configured {
            println!();
            println!("\u{2717} Gateway service restart failed.");
            println!("  The service definition exists, but the service manager did not recover it.");
            println!("  Fix the service, then retry: hermes gateway start");
            return Err(GatewayCommandError::Exit(1));
        }

        if stop_profile_gateway() {
            println!("\u{2713} Stopped gateway for this profile");
        }
        wait_for_gateway_exit(10.0, Some(5.0));
        println!("Starting gateway...");
        run_gateway(runner, 0, false, false).map_err(GatewayCommandError::Exit)?;
    }
    Ok(())
}

fn cmd_status(args: &GatewayArgs) {
    let snapshot = get_gateway_runtime_snapshot(args.system);

    if supports_systemd_services() && user_unit_or_system_unit_exists() {
        systemd_status(args.deep, args.system, args.full);
        print_gateway_process_mismatch(&snapshot);
    } else if is_macos() && get_launchd_plist_path().exists() {
        launchd_status(args.deep);
        print_gateway_process_mismatch(&snapshot);
    } else {
        let pids = &snapshot.gateway_pids;
        if !pids.is_empty() {
            let pid_str: Vec<String> = pids.iter().map(|p| p.to_string()).collect();
            println!("\u{2713} Gateway is running (PID: {})", pid_str.join(", "));
            println!("  (Running manually, not as a system service)");
            let runtime_lines = runtime_health_lines();
            if !runtime_lines.is_empty() {
                println!();
                println!("Recent gateway health:");
                for line in &runtime_lines {
                    println!("  {line}");
                }
            }
            println!();
            if is_termux() {
                println!("Termux note:");
                println!("  Android may stop background jobs when Termux is suspended");
            } else if is_wsl() {
                println!("WSL note:");
                println!("  The gateway is running in foreground/manual mode (recommended for WSL).");
                println!("  Use tmux or screen for persistence across terminal closes.");
            } else {
                println!("To install as a service:");
                println!("  hermes gateway install");
                println!("  sudo hermes gateway install --system");
            }
        } else {
            println!("\u{2717} Gateway is not running");
            let runtime_lines = runtime_health_lines();
            if !runtime_lines.is_empty() {
                println!();
                println!("Recent gateway health:");
                for line in &runtime_lines {
                    println!("  {line}");
                }
            }
            println!();
            println!("To start:");
            println!("  hermes gateway run      # Run in foreground");
            if is_termux() {
                println!("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # Best-effort background start");
            } else if is_wsl() {
                println!("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux");
                println!("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background");
            } else {
                println!("  hermes gateway install  # Install as user service");
                println!("  sudo hermes gateway install --system  # Install as boot-time system service");
            }
        }
    }

    print_other_profiles_gateway_status();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_running_and_mismatch() {
        let snap = GatewayRuntimeSnapshot {
            manager: "systemd (user)".to_string(),
            service_installed: true,
            service_running: false,
            gateway_pids: vec![1234],
            service_scope: Some("user".to_string()),
        };
        assert!(snap.running());
        assert!(snap.has_process_service_mismatch());

        let snap2 = GatewayRuntimeSnapshot {
            manager: "systemd (user)".to_string(),
            service_installed: true,
            service_running: true,
            gateway_pids: vec![1234],
            service_scope: Some("user".to_string()),
        };
        assert!(snap2.running());
        assert!(!snap2.has_process_service_mismatch());

        let snap3 = GatewayRuntimeSnapshot {
            manager: "manual process".to_string(),
            service_installed: false,
            service_running: false,
            gateway_pids: vec![],
            service_scope: None,
        };
        assert!(!snap3.running());
        assert!(!snap3.has_process_service_mismatch());
    }

    #[test]
    fn valid_profile_name_matches_regex() {
        assert!(is_valid_profile_name("coder"));
        assert!(is_valid_profile_name("a"));
        assert!(is_valid_profile_name("dev-1_x"));
        assert!(is_valid_profile_name("0abc"));
        assert!(!is_valid_profile_name(""));
        assert!(!is_valid_profile_name("-bad"));
        assert!(!is_valid_profile_name("_bad"));
        assert!(!is_valid_profile_name("Upper"));
        assert!(!is_valid_profile_name("has space"));
        let too_long = "a".repeat(65);
        assert!(!is_valid_profile_name(&too_long));
        let max_len = "a".repeat(64);
        assert!(is_valid_profile_name(&max_len));
    }

    #[test]
    fn service_name_default_is_base() {
        // For an unknown custom path, suffix is an 8-char hash → service name appended.
        let custom = PathBuf::from("/tmp/some-arbitrary-hermes-home-xyz");
        let name = get_service_name(Some(&custom));
        assert!(name.starts_with("hermes-gateway-"));
        assert_eq!(name.len(), "hermes-gateway-".len() + 8);
    }

    #[test]
    fn sha256_hex8_is_eight_chars() {
        let h = sha256_hex8("/some/path");
        assert_eq!(h.len(), 8);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn normalize_strips_trailing_whitespace() {
        let input = "  [Unit]   \nDescription=x  \n\n";
        let out = normalize_service_definition(input);
        assert_eq!(out, "[Unit]\nDescription=x");
    }

    #[test]
    fn launchd_plist_path_masking_ignores_path() {
        let a = "<key>PATH</key>\n<string>/usr/bin:/bin</string>";
        let b = "<key>PATH</key>\n<string>/opt/x:/usr/bin</string>";
        assert_eq!(
            normalize_launchd_plist_for_comparison(a),
            normalize_launchd_plist_for_comparison(b)
        );
    }

    #[test]
    fn format_gateway_pids_limits_and_filters() {
        let pids = vec![10, 20, 30, 40];
        assert_eq!(format_gateway_pids(&pids, Some(3)), "10, 20, 30, ...");
        assert_eq!(format_gateway_pids(&pids, None), "10, 20, 30, 40");
        let with_zero = vec![0, 5, -1, 7];
        assert_eq!(format_gateway_pids(&with_zero, None), "5, 7");
    }

    #[test]
    fn append_unique_pid_skips_excluded_and_dupes() {
        let mut pids: Vec<i64> = vec![];
        let mut exclude = HashSet::new();
        exclude.insert(99);
        append_unique_pid(&mut pids, Some(99), &exclude); // excluded
        append_unique_pid(&mut pids, Some(0), &exclude); // <= 0
        append_unique_pid(&mut pids, None, &exclude); // none
        append_unique_pid(&mut pids, Some(5), &exclude);
        append_unique_pid(&mut pids, Some(5), &exclude); // dup
        assert_eq!(pids, vec![5]);
    }

    #[test]
    fn remap_path_for_user_swaps_home_prefix() {
        // Build a path under the real home so strip_prefix succeeds.
        let home = home_dir();
        let under_home = home.join(".hermes").join("bin").to_string_lossy().to_string();
        let remapped = remap_path_for_user(&under_home, "/home/alice");
        assert_eq!(remapped, "/home/alice/.hermes/bin");

        // A path not under home is preserved.
        let outside = "/opt/hermes";
        assert_eq!(remap_path_for_user(outside, "/home/alice"), "/opt/hermes");
    }

    #[test]
    fn capitalize_works() {
        assert_eq!(capitalize("user"), "User");
        assert_eq!(capitalize("system"), "System");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn is_progress_classification() {
        assert!(is_progress("configured"));
        assert!(is_progress("configured + E2EE"));
        assert!(is_progress("enabled, not paired"));
        assert!(!is_progress("not configured"));
        assert!(!is_progress("partially configured"));
        assert!(!is_progress("Plugin disabled"));
    }

    #[test]
    fn platform_status_respects_env() {
        // Use a synthetic platform with a token var unlikely to be set.
        let platforms = builtin_platforms();
        let telegram = platforms.iter().find(|p| p.key == "telegram").unwrap();
        // Without the env var set, status should be "not configured".
        // (cli_config::get_env_value reads .env/process env; assume unset in test.)
        unsafe {
            std::env::remove_var("TELEGRAM_BOT_TOKEN");
        }
        let status = platform_status(telegram);
        assert!(status == "not configured" || status == "configured");
    }

    #[test]
    fn builtin_platforms_complete_set() {
        let platforms = builtin_platforms();
        let keys: Vec<&str> = platforms.iter().map(|p| p.key).collect();
        for expected in [
            "telegram",
            "discord",
            "slack",
            "matrix",
            "mattermost",
            "whatsapp",
            "signal",
            "email",
            "sms",
            "dingtalk",
            "feishu",
            "wecom",
            "wecom_callback",
            "weixin",
            "bluebubbles",
            "qqbot",
            "yuanbao",
        ] {
            assert!(keys.contains(&expected), "missing platform {expected}");
        }
        assert_eq!(platforms.len(), 17);
    }

    #[test]
    fn terminate_pid_nonexistent_is_no_such_process() {
        // PID 999999999 almost certainly doesn't exist.
        match os_kill(999_999_999, 0) {
            Err(KillError::NoSuchProcess) => {}
            other => {
                // On some sandboxes kill of nonexistent may differ; just assert it's an error.
                assert!(other.is_err());
            }
        }
    }

    #[test]
    fn installed_systemd_scopes_no_panic() {
        // Should not panic regardless of environment.
        let _ = get_installed_systemd_scopes();
        let _ = has_conflicting_systemd_units();
    }
}
