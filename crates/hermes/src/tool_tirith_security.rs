//! Tirith pre-exec security scanning wrapper.
//!
//! Native Rust port of `tools/tirith_security.py`.
//!
//! Runs the `tirith` binary as a subprocess to scan commands for content-level
//! threats (homograph URLs, pipe-to-interpreter, terminal injection, etc.).
//!
//! Exit code is the verdict source of truth:
//!   0 = allow, 1 = block, 2 = warn
//!
//! JSON stdout enriches findings/summary but never overrides the verdict.
//! Operational failures (spawn error, timeout, unknown exit code) respect the
//! `fail_open` config setting. Programming errors propagate.
//!
//! Auto-install: if tirith is not found on PATH or at the configured path, it is
//! automatically downloaded from GitHub releases to `$HERMES_HOME/bin/tirith`.
//! The download always verifies SHA-256 checksums. When cosign is available on
//! PATH, provenance verification (GitHub Actions workflow signature) is also
//! performed. If cosign is not installed, the download proceeds with SHA-256
//! verification only — still secure via HTTPS + checksum, just without supply
//! chain provenance proof. Installation runs in a background thread so startup
//! never blocks.
//!
//! Behavioural notes vs. the Python original:
//! * Config access goes through [`hermes_core::cli_config::load_config`] and
//!   [`hermes_core::mod_hermes_constants::get_hermes_home`]; if `load_config`
//!   fails for any reason the defaults (plus env overrides) are used, matching
//!   the Python `try/except` fallback.
//! * The background install thread is a `std::thread` spawned daemon-style (it
//!   is not joined). Process-lifetime resolution caching uses a global `Mutex`
//!   over an enum that mirrors the Python `_resolved_path`/`_INSTALL_FAILED`
//!   sentinel tri-state.

use std::fs;
use std::io::{Read, Write as _};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};

const REPO: &str = "sheeki03/tirith";

// Cosign provenance verification — pinned to the specific release workflow.
fn cosign_identity_regexp() -> String {
    format!(
        "^https://github.com/{REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
    )
}
const COSIGN_ISSUER: &str = "https://token.actions.githubusercontent.com";

const MARKER_TTL: u64 = 86400; // 24 hours
const MAX_FINDINGS: usize = 50;
const MAX_SUMMARY_LEN: usize = 500;

// ---------------------------------------------------------------------------
// Resolution state (mirrors Python module globals)
// ---------------------------------------------------------------------------

/// Tri-state for `_resolved_path`:
/// * `NotTried` — not yet resolved (Python `None`)
/// * `Resolved(path)` — successfully resolved (Python `str`)
/// * `InstallFailed` — tried and failed (Python `_INSTALL_FAILED` sentinel)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedState {
    NotTried,
    Resolved(String),
    InstallFailed,
}

struct InstallState {
    resolved: ResolvedState,
    failure_reason: String,
    /// True while a background install thread is running.
    thread_alive: bool,
}

fn state() -> &'static Mutex<InstallState> {
    static STATE: OnceLock<Mutex<InstallState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(InstallState {
            resolved: ResolvedState::NotTried,
            failure_reason: String::new(),
            thread_alive: false,
        })
    })
}

/// Test-only: reset the global resolution state.
#[cfg(test)]
pub fn reset_state_for_test() {
    let mut s = state().lock().unwrap();
    s.resolved = ResolvedState::NotTried;
    s.failure_reason = String::new();
    s.thread_alive = false;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Resolved security configuration for tirith.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecurityConfig {
    pub tirith_enabled: bool,
    pub tirith_path: String,
    pub tirith_timeout: u64,
    pub tirith_fail_open: bool,
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(val) => matches!(val.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => default,
    }
}

fn env_int(key: &str, default: u64) -> u64 {
    match std::env::var(key) {
        Ok(val) => val.trim().parse::<u64>().unwrap_or(default),
        Err(_) => default,
    }
}

/// Read the `security` mapping from `~/.hermes/config.yaml`.
///
/// Returns `Value::Null` if the file is missing, empty, unparseable, or has no
/// `security` section — mirroring the Python `load_config()["security"]` access
/// guarded by a try/except.
fn load_security_section() -> serde_yaml::Value {
    let config_path = get_hermes_home().join("config.yaml");
    let text = match fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(_) => return serde_yaml::Value::Null,
    };
    if text.trim().is_empty() {
        return serde_yaml::Value::Null;
    }
    let cfg: serde_yaml::Value = match serde_yaml::from_str(&text) {
        Ok(v) => v,
        Err(_) => return serde_yaml::Value::Null,
    };
    cfg.get("security").cloned().unwrap_or(serde_yaml::Value::Null)
}

/// Load security settings from config.yaml, with env var overrides.
pub fn load_security_config() -> SecurityConfig {
    let default_enabled = true;
    let default_path = "tirith".to_string();
    let default_timeout: u64 = 5;
    let default_fail_open = true;

    // try { from hermes_cli.config import load_config; ...security... } except {}
    // Read the `security` section from ~/.hermes/config.yaml directly (mirrors
    // hermes_cli.config.load_config -> cfg["security"]). Any failure (missing
    // file, parse error) falls through to defaults + env overrides, matching the
    // Python try/except fallback.
    let sec: serde_yaml::Value = std::panic::catch_unwind(load_security_section)
        .unwrap_or(serde_yaml::Value::Null);

    let cfg_enabled = sec
        .get("tirith_enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(default_enabled);
    let cfg_path = sec
        .get("tirith_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_path.clone());
    let cfg_timeout = sec
        .get("tirith_timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(default_timeout);
    let cfg_fail_open = sec
        .get("tirith_fail_open")
        .and_then(|v| v.as_bool())
        .unwrap_or(default_fail_open);

    // os.getenv("TIRITH_BIN", cfg_path) — env wins only if set.
    let tirith_path = std::env::var("TIRITH_BIN").unwrap_or(cfg_path);

    SecurityConfig {
        tirith_enabled: env_bool("TIRITH_ENABLED", cfg_enabled),
        tirith_path,
        tirith_timeout: env_int("TIRITH_TIMEOUT", cfg_timeout),
        tirith_fail_open: env_bool("TIRITH_FAIL_OPEN", cfg_fail_open),
    }
}

// ---------------------------------------------------------------------------
// Filesystem / PATH helpers
// ---------------------------------------------------------------------------

/// Return the Hermes home directory (default: `~/.hermes`).
///
/// Mirrors `hermes_constants.get_hermes_home`: honours `HERMES_HOME` when set
/// and non-empty, otherwise falls back to `~/.hermes`.
fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
}

fn failure_marker_path() -> PathBuf {
    get_hermes_home().join(".tirith-install-failed")
}

fn hermes_bin_dir() -> PathBuf {
    let d = get_hermes_home().join("bin");
    let _ = fs::create_dir_all(&d);
    d
}

/// Expand a leading `~` to the home directory (mirrors os.path.expanduser).
fn expanduser(path: &str) -> String {
    if path == "~" {
        if let Some(h) = dirs::home_dir() {
            return h.to_string_lossy().to_string();
        }
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

/// True if `path` is a regular file and executable by the current user.
fn is_file_and_executable(path: &str) -> bool {
    let p = Path::new(path);
    match fs::metadata(p) {
        Ok(md) => md.is_file() && (md.permissions().mode() & 0o111) != 0,
        Err(_) => false,
    }
}

/// Locate an executable on PATH, mirroring `shutil.which`.
///
/// If `name` contains a path separator it is checked directly (Python's
/// `shutil.which` returns the path as-is when it points at an executable file).
fn which(name: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }
    if name.contains('/') {
        if is_file_and_executable(name) {
            return Some(name.to_string());
        }
        return None;
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        let s = candidate.to_string_lossy().to_string();
        if is_file_and_executable(&s) {
            return Some(s);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Disk failure marker
// ---------------------------------------------------------------------------

/// Read the failure reason from the disk marker.
///
/// Returns the reason string, or None if the marker doesn't exist or is older
/// than `MARKER_TTL`.
fn read_failure_reason() -> Option<String> {
    let p = failure_marker_path();
    let md = fs::metadata(&p).ok()?;
    let mtime = md.modified().ok()?;
    let mtime_secs = mtime.duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    if (now - mtime_secs) >= MARKER_TTL as f64 {
        return None;
    }
    let contents = fs::read_to_string(&p).ok()?;
    Some(contents.trim().to_string())
}

/// Check if a recent install failure was persisted to disk.
///
/// Returns False (allowing retry) when:
/// - No marker exists
/// - Marker is older than `MARKER_TTL` (24h)
/// - Marker reason is 'cosign_missing' and cosign is now on PATH
fn is_install_failed_on_disk() -> bool {
    let reason = match read_failure_reason() {
        Some(r) => r,
        None => return false,
    };
    if reason == "cosign_missing" && which("cosign").is_some() {
        clear_install_failed();
        return false;
    }
    true
}

/// Persist install failure to disk to avoid retry on next process.
fn mark_install_failed(reason: &str) {
    let p = failure_marker_path();
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = fs::File::create(&p) {
        let _ = f.write_all(reason.as_bytes());
    }
}

/// Remove the failure marker after successful install.
fn clear_install_failed() {
    let _ = fs::remove_file(failure_marker_path());
}

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------

/// Return the Rust target triple for the current platform, or None.
fn detect_target() -> Option<String> {
    // platform.system() / platform.machine() mapped via cfg.
    // Android (Termux) is ABI-compatible with Linux — reuse Linux binaries.
    let plat = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_os = "linux") || cfg!(target_os = "android") {
        "unknown-linux-gnu"
    } else {
        return None;
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        return None;
    };

    Some(format!("{arch}-{plat}"))
}

// ---------------------------------------------------------------------------
// Download + verification
// ---------------------------------------------------------------------------

/// Download a URL to a local file (mirrors `_download_file`).
fn download_file(url: &str, dest: &Path, timeout_secs: u64) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client.get(url);
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        if !token.is_empty() {
            req = req.header("Authorization", format!("token {token}"));
        }
    }
    let resp = req.send().map_err(|e| e.to_string())?;
    let resp = resp.error_for_status().map_err(|e| e.to_string())?;
    let bytes = resp.bytes().map_err(|e| e.to_string())?;
    let mut f = fs::File::create(dest).map_err(|e| e.to_string())?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(())
}

/// Cosign provenance result.
///
/// * `Some(true)`  — cosign verified successfully
/// * `Some(false)` — cosign found but verification failed
/// * `None`        — cosign not available (not on PATH, or execution failed)
fn verify_cosign(checksums_path: &Path, sig_path: &Path, cert_path: &Path) -> Option<bool> {
    let cosign = match which("cosign") {
        Some(c) => c,
        None => {
            log::info!("cosign not found on PATH");
            return None;
        }
    };

    let output = Command::new(&cosign)
        .arg("verify-blob")
        .arg("--certificate")
        .arg(cert_path)
        .arg("--signature")
        .arg(sig_path)
        .arg("--certificate-identity-regexp")
        .arg(cosign_identity_regexp())
        .arg("--certificate-oidc-issuer")
        .arg(COSIGN_ISSUER)
        .arg(checksums_path)
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                log::info!("cosign provenance verification passed");
                Some(true)
            } else {
                let code = out.status.code().unwrap_or(-1);
                let stderr = String::from_utf8_lossy(&out.stderr);
                log::warn!(
                    "cosign verification failed (exit {}): {}",
                    code,
                    stderr.trim()
                );
                Some(false)
            }
        }
        Err(exc) => {
            log::warn!("cosign execution failed: {exc}");
            None
        }
    }
}

/// Verify SHA-256 of the archive against checksums.txt.
fn verify_checksum(archive_path: &Path, checksums_path: &Path, archive_name: &str) -> bool {
    let contents = match fs::read_to_string(checksums_path) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("No checksum entry for {archive_name}");
            return false;
        }
    };

    let mut expected: Option<String> = None;
    for line in contents.lines() {
        // Format: "<hash>  <filename>"
        let trimmed = line.trim();
        if let Some((hash, fname)) = trimmed.split_once("  ") {
            if fname == archive_name {
                expected = Some(hash.to_string());
                break;
            }
        }
    }
    let expected = match expected {
        Some(e) if !e.is_empty() => e,
        _ => {
            log::warn!("No checksum entry for {archive_name}");
            return false;
        }
    };

    let mut f = match fs::File::open(archive_path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(_) => return false,
        }
    }
    let actual = hex_encode(&hasher.finalize());
    if actual != expected {
        log::warn!("Checksum mismatch: expected {expected}, got {actual}");
        return false;
    }
    true
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

/// Download and install tirith to `$HERMES_HOME/bin/tirith`.
///
/// Verifies provenance via cosign and SHA-256 checksum. Returns
/// `(installed_path, failure_reason)`. On success `failure_reason` is "".
fn install_tirith(log_failures: bool) -> (Option<String>, String) {
    macro_rules! log_fail {
        ($($arg:tt)*) => {
            if log_failures { log::warn!($($arg)*); } else { log::debug!($($arg)*); }
        };
    }

    let target = match detect_target() {
        Some(t) => t,
        None => {
            log::info!("tirith auto-install: unsupported platform");
            return (None, "unsupported_platform".to_string());
        }
    };

    let archive_name = format!("tirith-{target}.tar.gz");
    let base_url = format!("https://github.com/{REPO}/releases/latest/download");

    let tmpdir = match tempfile::Builder::new().prefix("tirith-install-").tempdir() {
        Ok(d) => d,
        Err(_) => return (None, "download_failed".to_string()),
    };
    let tmp = tmpdir.path();

    let archive_path = tmp.join(&archive_name);
    let checksums_path = tmp.join("checksums.txt");
    let sig_path = tmp.join("checksums.txt.sig");
    let cert_path = tmp.join("checksums.txt.pem");

    log::info!("tirith not found — downloading latest release for {target}...");

    if let Err(exc) = download_file(&format!("{base_url}/{archive_name}"), &archive_path, 10) {
        log_fail!("tirith download failed: {exc}");
        return (None, "download_failed".to_string());
    }
    if let Err(exc) = download_file(&format!("{base_url}/checksums.txt"), &checksums_path, 10) {
        log_fail!("tirith download failed: {exc}");
        return (None, "download_failed".to_string());
    }

    // Cosign provenance verification — preferred but not mandatory.
    let mut cosign_verified = false;
    if which("cosign").is_some() {
        let sig_ok = download_file(&format!("{base_url}/checksums.txt.sig"), &sig_path, 10);
        let cert_ok = download_file(&format!("{base_url}/checksums.txt.pem"), &cert_path, 10);
        match (sig_ok, cert_ok) {
            (Ok(()), Ok(())) => match verify_cosign(&checksums_path, &sig_path, &cert_path) {
                Some(true) => cosign_verified = true,
                Some(false) => {
                    // Verification explicitly rejected — abort, the release
                    // may have been tampered with.
                    log_fail!("tirith install aborted: cosign provenance verification failed");
                    return (None, "cosign_verification_failed".to_string());
                }
                None => {
                    // execution failure (timeout/OSError) — proceed with
                    // SHA-256 only since cosign itself is broken.
                    log::info!("cosign execution failed, proceeding with SHA-256 only");
                }
            },
            (Err(exc), _) | (Ok(()), Err(exc)) => {
                log::info!("cosign artifacts unavailable ({exc}), proceeding with SHA-256 only");
            }
        }
    } else {
        log::info!(
            "cosign not on PATH — installing tirith with SHA-256 verification only \
             (install cosign for full supply chain verification)"
        );
    }

    if !verify_checksum(&archive_path, &checksums_path, &archive_name) {
        return (None, "checksum_failed".to_string());
    }

    // Extract only the tirith binary (safety: reject paths with ..)
    let extracted = tmp.join("tirith");
    let mut found_binary = false;
    {
        let file = match fs::File::open(&archive_path) {
            Ok(f) => f,
            Err(_) => return (None, "checksum_failed".to_string()),
        };
        let gz = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(gz);
        let entries = match tar.entries() {
            Ok(e) => e,
            Err(_) => {
                log_fail!("tirith binary not found in archive");
                return (None, "binary_not_in_archive".to_string());
            }
        };
        for entry in entries {
            let mut entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = match entry.path() {
                Ok(p) => p.to_string_lossy().to_string(),
                Err(_) => continue,
            };
            if name == "tirith" || name.ends_with("/tirith") {
                if name.contains("..") {
                    continue;
                }
                if entry.unpack(&extracted).is_ok() {
                    found_binary = true;
                }
                break;
            }
        }
    }
    if !found_binary {
        log_fail!("tirith binary not found in archive");
        return (None, "binary_not_in_archive".to_string());
    }

    let dest = hermes_bin_dir().join("tirith");
    // shutil.move with copy fallback. fs::rename fails across devices; fall
    // back to copy. If the copy fails, clean up partial dest.
    if fs::rename(&extracted, &dest).is_err() {
        if fs::copy(&extracted, &dest).is_err() {
            let _ = fs::remove_file(&dest);
            return (None, "cross_device_copy_failed".to_string());
        }
    }

    // chmod: add user/group/other execute bits to current mode.
    if let Ok(md) = fs::metadata(&dest) {
        let mode = md.permissions().mode() | 0o111;
        let _ = fs::set_permissions(&dest, fs::Permissions::from_mode(mode));
    }

    let verification = if cosign_verified {
        "cosign + SHA-256"
    } else {
        "SHA-256 only"
    };
    let dest_str = dest.to_string_lossy().to_string();
    log::info!("tirith installed to {dest_str} ({verification})");
    (Some(dest_str), String::new())
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Return True if the user explicitly configured a non-default tirith path.
fn is_explicit_path(configured_path: &str) -> bool {
    configured_path != "tirith"
}

/// Resolve the tirith binary path, auto-installing if necessary.
///
/// Mirrors `_resolve_tirith_path`. Returns the path to use (which may be the
/// configured path even when resolution failed, matching the Python contract
/// where the caller's spawn step fails-open).
pub fn resolve_tirith_path(configured_path: &str) -> String {
    let expanded = expanduser(configured_path);
    let explicit = is_explicit_path(configured_path);

    let mut s = state().lock().unwrap();

    // Fast path: successfully resolved on a previous call.
    if let ResolvedState::Resolved(p) = &s.resolved {
        return p.clone();
    }

    let mut install_failed = matches!(s.resolved, ResolvedState::InstallFailed);

    // Explicit path: check it and stop. Never auto-download a replacement.
    if explicit {
        if is_file_and_executable(&expanded) {
            s.resolved = ResolvedState::Resolved(expanded.clone());
            return expanded;
        }
        if let Some(found) = which(&expanded) {
            s.resolved = ResolvedState::Resolved(found.clone());
            return found;
        }
        log::warn!("Configured tirith path {configured_path:?} not found; scanning disabled");
        s.resolved = ResolvedState::InstallFailed;
        s.failure_reason = "explicit_path_missing".to_string();
        return expanded;
    }

    // Default "tirith" — always re-run cheap local checks so a manual install
    // is picked up even after a previous network failure.
    if let Some(found) = which("tirith") {
        s.resolved = ResolvedState::Resolved(found.clone());
        s.failure_reason = String::new();
        clear_install_failed();
        return found;
    }

    let hermes_bin = hermes_bin_dir().join("tirith");
    let hermes_bin_str = hermes_bin.to_string_lossy().to_string();
    if is_file_and_executable(&hermes_bin_str) {
        s.resolved = ResolvedState::Resolved(hermes_bin_str.clone());
        s.failure_reason = String::new();
        clear_install_failed();
        return hermes_bin_str;
    }

    // Local checks failed. If a previous install attempt already failed, skip
    // the network retry — UNLESS the failure was "cosign_missing" and cosign is
    // now available.
    if install_failed {
        if s.failure_reason == "cosign_missing" && which("cosign").is_some() {
            s.resolved = ResolvedState::NotTried;
            s.failure_reason = String::new();
            clear_install_failed();
            install_failed = false;
        } else {
            return expanded;
        }
    }
    let _ = install_failed;

    // If a background install thread is running, don't start a parallel one.
    if s.thread_alive {
        return expanded;
    }

    // Check disk failure marker before attempting network download.
    let disk_reason = read_failure_reason();
    if let Some(reason) = &disk_reason {
        if is_install_failed_on_disk() {
            s.resolved = ResolvedState::InstallFailed;
            s.failure_reason = reason.clone();
            return expanded;
        }
    }

    // Run the (blocking) install. Drop the lock while doing network IO so other
    // callers see the thread_alive flag and don't pile on.
    s.thread_alive = true;
    drop(s);

    let (installed, reason) = install_tirith(true);

    let mut s = state().lock().unwrap();
    s.thread_alive = false;
    if let Some(installed) = installed {
        s.resolved = ResolvedState::Resolved(installed.clone());
        s.failure_reason = String::new();
        clear_install_failed();
        installed
    } else {
        s.resolved = ResolvedState::InstallFailed;
        s.failure_reason = reason.clone();
        mark_install_failed(&reason);
        expanded
    }
}

/// Background thread target: download and install tirith.
fn background_install(log_failures: bool) {
    {
        let mut s = state().lock().unwrap();
        // Double-check after acquiring lock (another thread may have resolved).
        if !matches!(s.resolved, ResolvedState::NotTried) {
            s.thread_alive = false;
            return;
        }

        // Re-check local paths (may have been installed by another process).
        if let Some(found) = which("tirith") {
            s.resolved = ResolvedState::Resolved(found);
            s.failure_reason = String::new();
            s.thread_alive = false;
            return;
        }
        let hermes_bin = hermes_bin_dir().join("tirith");
        let hermes_bin_str = hermes_bin.to_string_lossy().to_string();
        if is_file_and_executable(&hermes_bin_str) {
            s.resolved = ResolvedState::Resolved(hermes_bin_str);
            s.failure_reason = String::new();
            s.thread_alive = false;
            return;
        }
    }

    let (installed, reason) = install_tirith(log_failures);

    let mut s = state().lock().unwrap();
    s.thread_alive = false;
    if let Some(installed) = installed {
        s.resolved = ResolvedState::Resolved(installed);
        s.failure_reason = String::new();
        clear_install_failed();
    } else {
        s.resolved = ResolvedState::InstallFailed;
        s.failure_reason = reason.clone();
        mark_install_failed(&reason);
    }
}

/// Ensure tirith is available, downloading in background if needed.
///
/// Quick PATH/local checks are synchronous; network download runs in a daemon
/// thread so startup never blocks. Safe to call multiple times. Returns the
/// resolved path immediately if available, or None.
pub fn ensure_installed(log_failures: bool) -> Option<String> {
    let cfg = load_security_config();
    if !cfg.tirith_enabled {
        return None;
    }

    let mut s = state().lock().unwrap();

    // Already resolved from a previous call.
    if let ResolvedState::Resolved(path) = &s.resolved {
        let path = path.clone();
        if is_file_and_executable(&path) {
            return Some(path);
        }
        return None;
    }

    let configured_path = &cfg.tirith_path;
    let explicit = is_explicit_path(configured_path);
    let expanded = expanduser(configured_path);

    // Explicit path: synchronous check only, no download.
    if explicit {
        if is_file_and_executable(&expanded) {
            s.resolved = ResolvedState::Resolved(expanded.clone());
            return Some(expanded);
        }
        if let Some(found) = which(&expanded) {
            s.resolved = ResolvedState::Resolved(found.clone());
            return Some(found);
        }
        s.resolved = ResolvedState::InstallFailed;
        s.failure_reason = "explicit_path_missing".to_string();
        return None;
    }

    // Default "tirith" — quick local checks first (no network).
    if let Some(found) = which("tirith") {
        s.resolved = ResolvedState::Resolved(found.clone());
        s.failure_reason = String::new();
        clear_install_failed();
        return Some(found);
    }
    let hermes_bin = hermes_bin_dir().join("tirith");
    let hermes_bin_str = hermes_bin.to_string_lossy().to_string();
    if is_file_and_executable(&hermes_bin_str) {
        s.resolved = ResolvedState::Resolved(hermes_bin_str.clone());
        s.failure_reason = String::new();
        clear_install_failed();
        return Some(hermes_bin_str);
    }

    // If previously failed in-memory, check if the cause is now resolved.
    if matches!(s.resolved, ResolvedState::InstallFailed) {
        if s.failure_reason == "cosign_missing" && which("cosign").is_some() {
            s.resolved = ResolvedState::NotTried;
            s.failure_reason = String::new();
            clear_install_failed();
        } else {
            return None;
        }
    }

    // Check disk failure marker (skip network attempt for 24h).
    let disk_reason = read_failure_reason();
    if let Some(reason) = &disk_reason {
        if is_install_failed_on_disk() {
            s.resolved = ResolvedState::InstallFailed;
            s.failure_reason = reason.clone();
            return None;
        }
    }

    // Need to download — launch background thread so startup doesn't block.
    if !s.thread_alive {
        s.thread_alive = true;
        drop(s);
        thread::Builder::new()
            .name("tirith-install".to_string())
            .spawn(move || background_install(log_failures))
            .ok();
    }

    None // Not available yet; commands will fail-open until ready.
}

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/// Result of a security scan.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SecurityResult {
    pub action: String,
    pub findings: Vec<Value>,
    pub summary: String,
}

impl SecurityResult {
    fn new(action: &str, findings: Vec<Value>, summary: &str) -> Self {
        SecurityResult {
            action: action.to_string(),
            findings,
            summary: summary.to_string(),
        }
    }
}

/// Run tirith security scan on a command.
///
/// Exit code determines action (0=allow, 1=block, 2=warn). JSON enriches
/// findings/summary. Spawn failures and timeouts respect fail_open config.
pub fn check_command_security(command: &str) -> SecurityResult {
    let cfg = load_security_config();

    if !cfg.tirith_enabled {
        return SecurityResult::new("allow", vec![], "");
    }

    let tirith_path = if std::env::var("HERMES_GATEWAY_SESSION").is_ok()
        || std::env::var("HERMES_EXEC_ASK").is_ok()
    {
        ensure_installed(false)
    } else {
        Some(resolve_tirith_path(&cfg.tirith_path))
    };
    let timeout = cfg.tirith_timeout;
    let fail_open = cfg.tirith_fail_open;

    let tirith_path = match tirith_path {
        Some(p) => p,
        None => {
            log::warn!("tirith path resolved to None; scanning disabled");
            if fail_open {
                return SecurityResult::new("allow", vec![], "tirith path unavailable");
            }
            return SecurityResult::new(
                "block",
                vec![],
                "tirith path unavailable (fail-closed)",
            );
        }
    };

    // Spawn tirith with a timeout. subprocess.run(..., timeout=) maps to a
    // wait loop on the child since std has no built-in timeout.
    let spawn = Command::new(&tirith_path)
        .args([
            "check",
            "--json",
            "--non-interactive",
            "--shell",
            "posix",
            "--",
            command,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match spawn {
        Ok(c) => c,
        Err(exc) => {
            // Covers FileNotFoundError, PermissionError, exec format error.
            log::warn!("tirith spawn failed: {exc}");
            if fail_open {
                return SecurityResult::new(
                    "allow",
                    vec![],
                    &format!("tirith unavailable: {exc}"),
                );
            }
            return SecurityResult::new(
                "block",
                vec![],
                &format!("tirith spawn failed (fail-closed): {exc}"),
            );
        }
    };

    let deadline = std::time::Instant::now() + Duration::from_secs(timeout);
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    log::warn!("tirith timed out after {timeout}s");
                    if fail_open {
                        return SecurityResult::new(
                            "allow",
                            vec![],
                            &format!("tirith timed out ({timeout}s)"),
                        );
                    }
                    return SecurityResult::new(
                        "block",
                        vec![],
                        "tirith timed out (fail-closed)",
                    );
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(exc) => {
                log::warn!("tirith spawn failed: {exc}");
                if fail_open {
                    return SecurityResult::new(
                        "allow",
                        vec![],
                        &format!("tirith unavailable: {exc}"),
                    );
                }
                return SecurityResult::new(
                    "block",
                    vec![],
                    &format!("tirith spawn failed (fail-closed): {exc}"),
                );
            }
        }
    };

    // Collect stdout.
    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }

    // Map exit code to action.
    let exit_code = exit_status.code().unwrap_or(-1);
    let action = match exit_code {
        0 => "allow",
        1 => "block",
        2 => "warn",
        _ => {
            // Unknown exit code — respect fail_open.
            log::warn!("tirith returned unexpected exit code {exit_code}");
            if fail_open {
                return SecurityResult::new(
                    "allow",
                    vec![],
                    &format!("tirith exit code {exit_code} (fail-open)"),
                );
            }
            return SecurityResult::new(
                "block",
                vec![],
                &format!("tirith exit code {exit_code} (fail-closed)"),
            );
        }
    };

    // Parse JSON for enrichment (never overrides the exit code verdict).
    parse_enrichment(&stdout, action)
}

/// Parse the JSON stdout for findings/summary enrichment. Pure function so it
/// can be unit-tested directly.
pub fn parse_enrichment(stdout: &str, action: &str) -> SecurityResult {
    let mut findings: Vec<Value> = vec![];
    let mut summary = String::new();

    let parsed: Result<Value, _> = if stdout.trim().is_empty() {
        Ok(Value::Object(serde_json::Map::new()))
    } else {
        serde_json::from_str(stdout)
    };

    match parsed {
        Ok(data) => {
            if let Some(raw) = data.get("findings").and_then(|v| v.as_array()) {
                findings = raw.iter().take(MAX_FINDINGS).cloned().collect();
            }
            let raw_summary = data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            summary = truncate_chars(raw_summary, MAX_SUMMARY_LEN);
        }
        Err(_) => {
            // JSON parse failure degrades findings/summary, not the verdict.
            log::debug!("tirith JSON parse failed, using exit code only");
            if action == "block" {
                summary = "security issue detected (details unavailable)".to_string();
            } else if action == "warn" {
                summary = "security warning detected (details unavailable)".to_string();
            }
        }
    }

    SecurityResult::new(action, findings, &summary)
}

/// Truncate to at most `max` characters (matching Python `[:n]` on a str,
/// which slices by Unicode code points).
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_bool() {
        unsafe {
            std::env::set_var("TIRITH_TEST_BOOL", "yes");
        }
        assert!(env_bool("TIRITH_TEST_BOOL", false));
        unsafe {
            std::env::set_var("TIRITH_TEST_BOOL", "TRUE");
        }
        assert!(env_bool("TIRITH_TEST_BOOL", false));
        unsafe {
            std::env::set_var("TIRITH_TEST_BOOL", "0");
        }
        assert!(!env_bool("TIRITH_TEST_BOOL", true));
        unsafe {
            std::env::set_var("TIRITH_TEST_BOOL", "garbage");
        }
        assert!(!env_bool("TIRITH_TEST_BOOL", true));
        unsafe {
            std::env::remove_var("TIRITH_TEST_BOOL");
        }
        assert!(env_bool("TIRITH_TEST_BOOL", true));
        assert!(!env_bool("TIRITH_TEST_BOOL", false));
    }

    #[test]
    fn test_env_int() {
        unsafe {
            std::env::set_var("TIRITH_TEST_INT", "42");
        }
        assert_eq!(env_int("TIRITH_TEST_INT", 5), 42);
        unsafe {
            std::env::set_var("TIRITH_TEST_INT", "notanumber");
        }
        assert_eq!(env_int("TIRITH_TEST_INT", 5), 5);
        unsafe {
            std::env::remove_var("TIRITH_TEST_INT");
        }
        assert_eq!(env_int("TIRITH_TEST_INT", 7), 7);
    }

    #[test]
    fn test_is_explicit_path() {
        assert!(!is_explicit_path("tirith"));
        assert!(is_explicit_path("/usr/local/bin/tirith"));
        assert!(is_explicit_path("mytirith"));
    }

    #[test]
    fn test_detect_target_known_or_none() {
        // On supported CI platforms this returns Some; on others None. Just
        // assert it doesn't panic and the format is sane when present.
        if let Some(t) = detect_target() {
            assert!(t.contains('-'));
            assert!(t.starts_with("x86_64") || t.starts_with("aarch64"));
        }
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0x10]), "00ff10");
        assert_eq!(hex_encode(&[]), "");
    }

    #[test]
    fn test_truncate_chars() {
        assert_eq!(truncate_chars("hello", 3), "hel");
        assert_eq!(truncate_chars("hello", 10), "hello");
        // Unicode code points, not bytes.
        assert_eq!(truncate_chars("héllo", 2), "hé");
    }

    #[test]
    fn test_verify_checksum_matches() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("tirith-x.tar.gz");
        fs::write(&archive, b"hello world").unwrap();
        // sha256("hello world")
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        let checksums = dir.path().join("checksums.txt");
        fs::write(&checksums, format!("{expected}  tirith-x.tar.gz\n")).unwrap();
        assert!(verify_checksum(&archive, &checksums, "tirith-x.tar.gz"));
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("tirith-x.tar.gz");
        fs::write(&archive, b"hello world").unwrap();
        let checksums = dir.path().join("checksums.txt");
        fs::write(&checksums, "deadbeef  tirith-x.tar.gz\n").unwrap();
        assert!(!verify_checksum(&archive, &checksums, "tirith-x.tar.gz"));
    }

    #[test]
    fn test_verify_checksum_no_entry() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("tirith-x.tar.gz");
        fs::write(&archive, b"data").unwrap();
        let checksums = dir.path().join("checksums.txt");
        fs::write(&checksums, "abc  other-file.tar.gz\n").unwrap();
        assert!(!verify_checksum(&archive, &checksums, "tirith-x.tar.gz"));
    }

    #[test]
    fn test_parse_enrichment_valid_json() {
        let stdout = r#"{"findings": [{"id": "X"}], "summary": "danger"}"#;
        let r = parse_enrichment(stdout, "block");
        assert_eq!(r.action, "block");
        assert_eq!(r.findings.len(), 1);
        assert_eq!(r.summary, "danger");
    }

    #[test]
    fn test_parse_enrichment_empty_stdout() {
        let r = parse_enrichment("   ", "allow");
        assert_eq!(r.action, "allow");
        assert!(r.findings.is_empty());
        assert_eq!(r.summary, "");
    }

    #[test]
    fn test_parse_enrichment_invalid_json_block() {
        let r = parse_enrichment("not json{", "block");
        assert_eq!(r.action, "block");
        assert!(r.findings.is_empty());
        assert_eq!(r.summary, "security issue detected (details unavailable)");
    }

    #[test]
    fn test_parse_enrichment_invalid_json_warn() {
        let r = parse_enrichment("<<<", "warn");
        assert_eq!(r.summary, "security warning detected (details unavailable)");
    }

    #[test]
    fn test_parse_enrichment_invalid_json_allow() {
        let r = parse_enrichment("<<<", "allow");
        assert_eq!(r.summary, "");
    }

    #[test]
    fn test_parse_enrichment_findings_capped() {
        let many: Vec<Value> = (0..100).map(|i| serde_json::json!({"i": i})).collect();
        let data = serde_json::json!({"findings": many, "summary": ""});
        let stdout = serde_json::to_string(&data).unwrap();
        let r = parse_enrichment(&stdout, "warn");
        assert_eq!(r.findings.len(), MAX_FINDINGS);
    }

    #[test]
    fn test_parse_enrichment_summary_capped() {
        let long = "a".repeat(1000);
        let data = serde_json::json!({"findings": [], "summary": long});
        let stdout = serde_json::to_string(&data).unwrap();
        let r = parse_enrichment(&stdout, "allow");
        assert_eq!(r.summary.chars().count(), MAX_SUMMARY_LEN);
    }

    #[test]
    fn test_parse_enrichment_null_summary() {
        // summary present but null -> Python `(data.get("summary","") or "")`
        let data = serde_json::json!({"findings": [], "summary": null});
        let stdout = serde_json::to_string(&data).unwrap();
        let r = parse_enrichment(&stdout, "allow");
        assert_eq!(r.summary, "");
    }

    #[test]
    fn test_which_absolute_nonexistent() {
        assert!(which("/nonexistent/path/to/tirith-xyz").is_none());
    }

    #[test]
    fn test_check_disabled_returns_allow() {
        reset_state_for_test();
        unsafe {
            std::env::set_var("TIRITH_ENABLED", "0");
        }
        let r = check_command_security("rm -rf /");
        unsafe {
            std::env::remove_var("TIRITH_ENABLED");
        }
        assert_eq!(r.action, "allow");
        assert!(r.findings.is_empty());
        assert_eq!(r.summary, "");
    }

    #[test]
    fn test_explicit_missing_path_fail_open() {
        reset_state_for_test();
        unsafe {
            std::env::set_var("TIRITH_ENABLED", "1");
            std::env::set_var("TIRITH_BIN", "/nonexistent/tirith-binary-xyz");
            std::env::set_var("TIRITH_FAIL_OPEN", "1");
            std::env::remove_var("HERMES_GATEWAY_SESSION");
            std::env::remove_var("HERMES_EXEC_ASK");
        }
        let r = check_command_security("echo hi");
        unsafe {
            std::env::remove_var("TIRITH_ENABLED");
            std::env::remove_var("TIRITH_BIN");
            std::env::remove_var("TIRITH_FAIL_OPEN");
        }
        // Resolution returns the (nonexistent) expanded path; spawn fails ->
        // fail-open -> allow.
        assert_eq!(r.action, "allow");
    }
}
