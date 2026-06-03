//! Profile management for multiple isolated Hermes instances.
//!
//! Native Rust port of `hermes_cli/profiles.py`.
//!
//! Each profile is a fully independent `HERMES_HOME` directory with its own
//! `config.yaml`, `.env`, memory, sessions, skills, gateway, cron, and logs.
//! Profiles live under `~/.hermes/profiles/<name>/` by default. The "default"
//! profile is `~/.hermes` itself — backward compatible, zero migration needed.
//!
//! Behavioural notes vs. the Python original:
//! * Path/home resolution reads `HERMES_HOME` and `Path.home()` exactly like
//!   the Python `hermes_constants` helpers. `_get_default_hermes_home` mirrors
//!   `get_default_hermes_root()` (the resolved root, accounting for Docker and
//!   profile-mode layouts).
//! * `seed_profile_skills` shells out to the Python project just like the
//!   original (it depends on `tools.skills_sync`, which is still Python).
//! * Diagnostic/UX `print(...)` calls are reproduced verbatim on stdout.
//! * Archive export/import use `flate2`+`tar` (gztar) and reproduce the same
//!   exclusion rules, credential stripping, and path-safety checks.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use regex::Regex;
use std::sync::OnceLock;
use tar::{Archive, Builder, EntryType, Header};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `^[a-z0-9][a-z0-9_-]{0,63}$`
fn profile_id_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-z0-9][a-z0-9_-]{0,63}$").unwrap())
}

/// Directories bootstrapped inside every new profile.
pub const PROFILE_DIRS: &[&str] = &[
    "memories",
    "sessions",
    "skills",
    "skins",
    "logs",
    "plans",
    "workspace",
    "cron",
    // Per-profile HOME for subprocesses: isolates system tool configs (git,
    // ssh, gh, npm …) so credentials don't bleed between profiles.
    "home",
];

/// Files copied during `--clone` (if they exist in the source).
pub const CLONE_CONFIG_FILES: &[&str] = &["config.yaml", ".env", "SOUL.md"];

/// Subdirectory files copied during `--clone` (path relative to profile root).
pub const CLONE_SUBDIR_FILES: &[&str] = &["memories/MEMORY.md", "memories/USER.md"];

/// Runtime files stripped after `--clone-all` (shouldn't carry over).
pub const CLONE_ALL_STRIP: &[&str] = &["gateway.pid", "gateway_state.json", "processes.json"];

/// Directories/files to exclude when exporting the default (`~/.hermes`) profile.
pub const DEFAULT_EXPORT_EXCLUDE_ROOT: &[&str] = &[
    // Infrastructure
    "hermes-agent",
    ".worktrees",
    "profiles",
    "bin",
    "node_modules",
    // Databases & runtime state
    "state.db",
    "state.db-shm",
    "state.db-wal",
    "hermes_state.db",
    "response_store.db",
    "response_store.db-shm",
    "response_store.db-wal",
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
    "auth.json",
    ".env",
    "auth.lock",
    "active_profile",
    ".update_check",
    "errors.log",
    ".hermes_history",
    // Caches (regenerated on use)
    "image_cache",
    "audio_cache",
    "document_cache",
    "browser_screenshots",
    "checkpoints",
    "sandboxes",
    "logs",
];

/// Names that cannot be used as profile aliases.
pub const RESERVED_NAMES: &[&str] = &["hermes", "default", "test", "tmp", "root", "sudo"];

/// Hermes subcommands that cannot be used as profile names/aliases.
pub const HERMES_SUBCOMMANDS: &[&str] = &[
    "chat", "model", "gateway", "setup", "whatsapp", "login", "logout", "status", "cron", "doctor",
    "dump", "config", "pairing", "skills", "tools", "mcp", "sessions", "insights", "version",
    "update", "uninstall", "profile", "plugins", "honcho", "acp",
];

/// Error type for profile operations. Variants carry the human-readable message
/// the Python original would have raised or printed.
#[derive(Debug)]
pub enum ProfileError {
    /// `ValueError`
    Value(String),
    /// `FileExistsError`
    Exists(String),
    /// `FileNotFoundError`
    NotFound(String),
    /// I/O failure
    Io(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::Value(m)
            | ProfileError::Exists(m)
            | ProfileError::NotFound(m)
            | ProfileError::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ProfileError {}

impl From<io::Error> for ProfileError {
    fn from(err: io::Error) -> Self {
        ProfileError::Io(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ProfileError>;

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Lexically resolve a path to absolute form (no filesystem access).
/// Mirrors enough of `Path.resolve()` for comparison purposes.
fn lexical_abspath(path: &Path) -> PathBuf {
    let base = if path.is_absolute() {
        PathBuf::new()
    } else {
        env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };
    let mut out = base;
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir => {
                out = PathBuf::from("/");
            }
            Component::Prefix(p) => {
                out = PathBuf::from(p.as_os_str());
            }
            Component::Normal(c) => out.push(c),
        }
    }
    out
}

fn resolve(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| lexical_abspath(path))
}

/// Return the default (pre-profile) `HERMES_HOME` path.
///
/// Mirrors `hermes_constants.get_default_hermes_root()`:
/// * Standard deployments: `~/.hermes`.
/// * Docker / custom deployments where `HERMES_HOME` points outside `~/.hermes`:
///   returns `HERMES_HOME` directly (or its profile-root grandparent).
pub fn get_default_hermes_home() -> PathBuf {
    let native_home = home_dir().join(".hermes");
    let env_home = env::var("HERMES_HOME").unwrap_or_default();
    if env_home.trim().is_empty() {
        return native_home;
    }
    let env_path = PathBuf::from(env_home.trim());

    let env_resolved = resolve(&env_path);
    let native_resolved = resolve(&native_home);
    if env_resolved.starts_with(&native_resolved) {
        return native_home;
    }

    // Docker / custom deployment. `<root>/profiles/<name>` → root is grandparent.
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

/// Return the active `HERMES_HOME` (default `~/.hermes`), mirroring
/// `hermes_constants.get_hermes_home()`.
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    home_dir().join(".hermes")
}

/// Return the directory where named profiles are stored.
pub fn get_profiles_root() -> PathBuf {
    get_default_hermes_home().join("profiles")
}

/// Return the path to the sticky `active_profile` file.
pub fn get_active_profile_path() -> PathBuf {
    get_default_hermes_home().join("active_profile")
}

/// Return the directory for wrapper scripts (`~/.local/bin`).
pub fn get_wrapper_dir() -> PathBuf {
    home_dir().join(".local").join("bin")
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Return the canonical profile id used on disk and in CLI `-p` argv.
pub fn normalize_profile_name(name: &str) -> Result<String> {
    let stripped = name.trim();
    if stripped.is_empty() {
        return Err(ProfileError::Value("profile name cannot be empty".to_string()));
    }
    if stripped.eq_ignore_ascii_case("default") {
        return Ok("default".to_string());
    }
    Ok(stripped.to_lowercase())
}

/// Raise `ValueError` if *name* is not a valid profile identifier.
///
/// Validates the input as-given — strict lowercase match. Callers that accept
/// mixed-case input should call [`normalize_profile_name`] first.
pub fn validate_profile_name(name: &str) -> Result<()> {
    if name == "default" {
        return Ok(());
    }
    if profile_id_re().is_match(name) {
        return Ok(());
    }
    Err(ProfileError::Value(format!(
        "Invalid profile name '{name}'. Must match [a-z0-9][a-z0-9_-]{{0,63}}"
    )))
}

/// Resolve a profile name to its `HERMES_HOME` directory.
pub fn get_profile_dir(name: &str) -> Result<PathBuf> {
    let canon = normalize_profile_name(name)?;
    if canon == "default" {
        return Ok(get_default_hermes_home());
    }
    Ok(get_profiles_root().join(canon))
}

/// Check whether a profile directory exists.
pub fn profile_exists(name: &str) -> bool {
    let canon = match normalize_profile_name(name) {
        Ok(value) => value,
        Err(_) => return false,
    };
    if canon == "default" {
        return true;
    }
    get_profile_dir(&canon).map(|p| p.is_dir()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Alias / wrapper script management
// ---------------------------------------------------------------------------

/// Return the path of `name` if found on `PATH` (mirrors `which`).
fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Return a human-readable collision message, or `None` if the name is safe.
///
/// Checks: reserved names, hermes subcommands, existing binaries on PATH.
pub fn check_alias_collision(name: &str) -> Result<Option<String>> {
    let canon = normalize_profile_name(name)?;
    if RESERVED_NAMES.contains(&canon.as_str()) {
        return Ok(Some(format!("'{canon}' is a reserved name")));
    }
    if HERMES_SUBCOMMANDS.contains(&canon.as_str()) {
        return Ok(Some(format!("'{canon}' conflicts with a hermes subcommand")));
    }

    let wrapper_dir = get_wrapper_dir();
    if let Some(existing) = which_on_path(&canon) {
        // Allow overwriting our own wrappers.
        if existing == wrapper_dir.join(&canon) {
            if let Ok(content) = fs::read_to_string(&existing) {
                if content.contains("hermes -p") {
                    return Ok(None);
                }
            }
        }
        return Ok(Some(format!(
            "'{canon}' conflicts with an existing command ({})",
            existing.display()
        )));
    }
    Ok(None)
}

/// Check if `~/.local/bin` is in PATH.
pub fn is_wrapper_dir_in_path() -> bool {
    let target = get_wrapper_dir();
    env::split_paths(&env::var_os("PATH").unwrap_or_default()).any(|p| p == target)
}

/// Create a shell wrapper script at `~/.local/bin/<name>`.
///
/// Returns the path to the created wrapper, or `None` if creation failed.
pub fn create_wrapper_script(name: &str) -> Option<PathBuf> {
    let canon = normalize_profile_name(name).ok()?;
    let wrapper_dir = get_wrapper_dir();
    if let Err(e) = fs::create_dir_all(&wrapper_dir) {
        println!("⚠ Could not create {}: {e}", wrapper_dir.display());
        return None;
    }

    let wrapper_path = wrapper_dir.join(&canon);
    match fs::write(
        &wrapper_path,
        format!("#!/bin/sh\nexec hermes -p {canon} \"$@\"\n"),
    ) {
        Ok(()) => {}
        Err(e) => {
            println!("⚠ Could not create wrapper at {}: {e}", wrapper_path.display());
            return None;
        }
    }
    #[cfg(unix)]
    {
        match fs::metadata(&wrapper_path) {
            Ok(meta) => {
                let mut perms = meta.permissions();
                // S_IEXEC | S_IXGRP | S_IXOTH = 0o111
                perms.set_mode(perms.mode() | 0o111);
                if let Err(e) = fs::set_permissions(&wrapper_path, perms) {
                    println!("⚠ Could not create wrapper at {}: {e}", wrapper_path.display());
                    return None;
                }
            }
            Err(e) => {
                println!("⚠ Could not create wrapper at {}: {e}", wrapper_path.display());
                return None;
            }
        }
    }
    Some(wrapper_path)
}

/// Remove the wrapper script for a profile. Returns `true` if removed.
pub fn remove_wrapper_script(name: &str) -> bool {
    let canon = match normalize_profile_name(name) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let wrapper_path = get_wrapper_dir().join(canon);
    if wrapper_path.exists() {
        // Verify it's our wrapper before removing.
        if let Ok(content) = fs::read_to_string(&wrapper_path) {
            if content.contains("hermes -p") && fs::remove_file(&wrapper_path).is_ok() {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// ProfileInfo
// ---------------------------------------------------------------------------

/// Summary information about a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileInfo {
    pub name: String,
    pub path: PathBuf,
    pub is_default: bool,
    pub gateway_running: bool,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub has_env: bool,
    pub skill_count: usize,
    pub alias_path: Option<PathBuf>,
}

/// Read model/provider from a profile's `config.yaml`. Returns `(model, provider)`.
pub fn read_config_model(profile_dir: &Path) -> (Option<String>, Option<String>) {
    let config_path = profile_dir.join("config.yaml");
    if !config_path.exists() {
        return (None, None);
    }
    let text = match fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(_) => return (None, None),
    };
    let cfg: serde_yaml::Value = match serde_yaml::from_str(&text) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let model_cfg = match cfg.as_mapping().and_then(|m| {
        m.get(serde_yaml::Value::String("model".to_string()))
    }) {
        Some(v) => v,
        None => return (None, None),
    };

    if let Some(s) = model_cfg.as_str() {
        return (Some(s.to_string()), None);
    }
    if let Some(map) = model_cfg.as_mapping() {
        let default = map
            .get(serde_yaml::Value::String("default".to_string()))
            .or_else(|| map.get(serde_yaml::Value::String("model".to_string())))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let provider = map
            .get(serde_yaml::Value::String("provider".to_string()))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        return (default, provider);
    }
    (None, None)
}

/// Return `true` if the gateway PID file points to a live process.
pub fn check_gateway_running(profile_dir: &Path) -> bool {
    let pid_file = profile_dir.join("gateway.pid");
    let raw = match fs::read_to_string(&pid_file) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let trimmed = raw.trim();
    let pid: Option<i64> = if trimmed.starts_with('{') {
        serde_json::from_str::<serde_json::Value>(trimmed)
            .ok()
            .and_then(|v| v.get("pid").and_then(|p| p.as_i64()))
    } else {
        trimmed.parse::<i64>().ok()
    };
    match pid {
        Some(pid) => process_running(pid),
        None => false,
    }
}

fn process_running(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Count installed skills in a profile (recursively, skipping `.hub`/`.git`).
pub fn count_skills(profile_dir: &Path) -> usize {
    let skills_dir = profile_dir.join("skills");
    if !skills_dir.is_dir() {
        return 0;
    }
    let mut count = 0;
    walk_skill_markdowns(&skills_dir, &mut count);
    count
}

fn walk_skill_markdowns(root: &Path, count: &mut usize) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            // Python filters on "/.hub/" and "/.git/" substrings of the path.
            if name == ".hub" || name == ".git" {
                continue;
            }
            walk_skill_markdowns(&path, count);
        } else if name == "SKILL.md" {
            *count += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// CRUD operations
// ---------------------------------------------------------------------------

/// Return info for all profiles, including the default.
pub fn list_profiles() -> Vec<ProfileInfo> {
    let mut profiles = Vec::new();
    let wrapper_dir = get_wrapper_dir();

    let default_home = get_default_hermes_home();
    if default_home.is_dir() {
        let (model, provider) = read_config_model(&default_home);
        profiles.push(ProfileInfo {
            name: "default".to_string(),
            path: default_home.clone(),
            is_default: true,
            gateway_running: check_gateway_running(&default_home),
            model,
            provider,
            has_env: default_home.join(".env").exists(),
            skill_count: count_skills(&default_home),
            alias_path: None,
        });
    }

    let profiles_root = get_profiles_root();
    if profiles_root.is_dir() {
        let mut entries: Vec<PathBuf> = match fs::read_dir(&profiles_root) {
            Ok(rd) => rd.flatten().map(|e| e.path()).collect(),
            Err(_) => Vec::new(),
        };
        entries.sort();
        for entry in entries {
            if !entry.is_dir() {
                continue;
            }
            let name = match entry.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if !profile_id_re().is_match(&name) {
                continue;
            }
            let (model, provider) = read_config_model(&entry);
            let alias_path = wrapper_dir.join(&name);
            profiles.push(ProfileInfo {
                name,
                path: entry.clone(),
                is_default: false,
                gateway_running: check_gateway_running(&entry),
                model,
                provider,
                has_env: entry.join(".env").exists(),
                skill_count: count_skills(&entry),
                alias_path: alias_path.exists().then_some(alias_path),
            });
        }
    }

    profiles
}

/// Create a new profile directory.
///
/// * `clone_from` — source profile to clone from. If `None` and
///   `clone_config`/`clone_all` is set, defaults to the active profile.
/// * `clone_all` — full copytree of the source (all state).
/// * `clone_config` — copy config files, installed skills, and identity files.
/// * `_no_alias` — accepted for API parity; wrapper creation is performed by
///   the caller (matching the Python original, which also defers alias creation
///   to the CLI command layer).
pub fn create_profile(
    name: &str,
    clone_from: Option<&str>,
    clone_all: bool,
    clone_config: bool,
    _no_alias: bool,
) -> Result<PathBuf> {
    let canon = normalize_profile_name(name)?;
    validate_profile_name(&canon)?;

    if canon == "default" {
        return Err(ProfileError::Value(
            "Cannot create a profile named 'default' — it is the built-in profile (~/.hermes)."
                .to_string(),
        ));
    }

    let profile_dir = get_profile_dir(&canon)?;
    if profile_dir.exists() {
        return Err(ProfileError::Exists(format!(
            "Profile '{canon}' already exists at {}",
            profile_dir.display()
        )));
    }

    // Resolve clone source.
    let mut source_dir: Option<PathBuf> = None;
    let mut clone_label = String::from("active");
    if clone_from.is_some() || clone_all || clone_config {
        let dir = match clone_from {
            None => get_hermes_home(),
            Some(raw) => {
                let from = normalize_profile_name(raw)?;
                validate_profile_name(&from)?;
                clone_label = from.clone();
                get_profile_dir(&from)?
            }
        };
        if !dir.is_dir() {
            return Err(ProfileError::NotFound(format!(
                "Source profile '{clone_label}' does not exist at {}",
                dir.display()
            )));
        }
        source_dir = Some(dir);
    }

    if clone_all {
        if let Some(source) = &source_dir {
            clone_all_copytree(source, &profile_dir)?;
            for stale in CLONE_ALL_STRIP {
                let _ = fs::remove_file(profile_dir.join(stale));
            }
        }
    } else {
        // Bootstrap directory structure.
        fs::create_dir_all(&profile_dir)?;
        for subdir in PROFILE_DIRS {
            fs::create_dir_all(profile_dir.join(subdir))?;
        }

        if let Some(source) = &source_dir {
            for filename in CLONE_CONFIG_FILES {
                let src = source.join(filename);
                if src.exists() {
                    fs::copy(&src, profile_dir.join(filename))?;
                }
            }

            // Clone installed skills from the source profile (dirs_exist_ok).
            let source_skills = source.join("skills");
            if source_skills.is_dir() {
                copy_dir_recursive(&source_skills, &profile_dir.join("skills"), &|_, _| false)?;
            }

            // Clone memory and other subdirectory files.
            for relpath in CLONE_SUBDIR_FILES {
                let src = source.join(relpath);
                if src.exists() {
                    let dst = profile_dir.join(relpath);
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::copy(&src, &dst)?;
                }
            }
        }
    }

    // Seed a default SOUL.md so the user has a file to customize immediately.
    // Best-effort: the canonical content lives in Python (`default_soul.py`);
    // we honour `HERMES_DEFAULT_SOUL` if provided so the native path stays in
    // step, otherwise leave it for the Python layer / clone source.
    let soul_path = profile_dir.join("SOUL.md");
    if !soul_path.exists() {
        if let Some(default_soul) = env::var_os("HERMES_DEFAULT_SOUL") {
            let _ = fs::write(&soul_path, default_soul.to_string_lossy().as_bytes());
        }
    }

    Ok(profile_dir)
}

/// Copytree for `--clone-all`, ignoring `profiles/` at the root of the source.
fn clone_all_copytree(source: &Path, dest: &Path) -> Result<()> {
    let source_resolved = resolve(source);
    copy_dir_recursive(source, dest, &move |path, depth| {
        // Ignore `profiles` only at the source root (depth 1 of children).
        depth == 1
            && path.file_name().map(|n| n == "profiles") == Some(true)
            && path
                .parent()
                .map(|p| resolve(p) == source_resolved)
                .unwrap_or(false)
    })
}

/// Seed bundled skills into a profile via subprocess.
///
/// Mirrors the Python original, which uses a subprocess because `sync_skills()`
/// caches `HERMES_HOME` at module level. Returns the parsed sync result JSON, or
/// `None` on failure. Honours `HERMES_PROFILE_PYTHON` (defaults to `python3`).
pub fn seed_profile_skills(profile_dir: &Path, quiet: bool) -> Option<serde_json::Value> {
    let python = env::var("HERMES_PROFILE_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let project_root = env::var_os("HERMES_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let mut command = std::process::Command::new(&python);
    command
        .arg("-c")
        .arg(
            "import json; from tools.skills_sync import sync_skills; \
             r = sync_skills(quiet=True); print(json.dumps(r))",
        )
        .env("HERMES_HOME", profile_dir)
        .current_dir(&project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let output = match command.output() {
        Ok(o) => o,
        Err(e) => {
            if !quiet {
                println!("⚠ Skill seeding failed: {e}");
            }
            return None;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    if output.status.success() && !stdout.is_empty() {
        return serde_json::from_str(stdout).ok();
    }
    if !quiet {
        let code = output.status.code().unwrap_or(-1);
        println!("⚠ Skill seeding returned exit code {code}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        if !stderr.is_empty() {
            let snippet: String = stderr.chars().take(200).collect();
            println!("  {snippet}");
        }
    }
    None
}

/// Delete a profile, its wrapper script, and its gateway service.
///
/// Prints the same summary/confirmation flow as the Python original. When `yes`
/// is `false`, reads a confirmation line from stdin. Returns the removed path.
pub fn delete_profile(name: &str, yes: bool) -> Result<PathBuf> {
    let canon = normalize_profile_name(name)?;
    validate_profile_name(&canon)?;

    if canon == "default" {
        return Err(ProfileError::Value(
            "Cannot delete the default profile (~/.hermes).\nTo remove everything, use: hermes uninstall"
                .to_string(),
        ));
    }

    let profile_dir = get_profile_dir(&canon)?;
    if !profile_dir.is_dir() {
        return Err(ProfileError::NotFound(format!(
            "Profile '{canon}' does not exist."
        )));
    }

    let (model, provider) = read_config_model(&profile_dir);
    let gw_running = check_gateway_running(&profile_dir);
    let skill_count = count_skills(&profile_dir);

    println!("\nProfile: {canon}");
    println!("Path:    {}", profile_dir.display());
    if let Some(model) = &model {
        match &provider {
            Some(p) if !p.is_empty() => println!("Model:   {model} ({p})"),
            _ => println!("Model:   {model}"),
        }
    }
    if skill_count != 0 {
        println!("Skills:  {skill_count}");
    }

    let mut items = vec!["All config, API keys, memories, sessions, skills, cron jobs".to_string()];

    let wrapper_path = get_wrapper_dir().join(&canon);
    let has_wrapper = wrapper_path.exists();
    if has_wrapper {
        items.push(format!("Command alias ({})", wrapper_path.display()));
    }

    println!("\nThis will permanently delete:");
    for item in &items {
        println!("  • {item}");
    }
    if gw_running {
        println!("  ⚠ Gateway is running — it will be stopped.");
    }

    if !yes {
        println!();
        print!("Type '{canon}' to confirm: ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("\nCancelled.");
            return Ok(profile_dir);
        }
        if input.trim() != canon {
            println!("Cancelled.");
            return Ok(profile_dir);
        }
    }

    // 1. Disable service (prevents auto-restart).
    cleanup_gateway_service(&canon, &profile_dir);

    // 2. Stop running gateway.
    if gw_running {
        stop_gateway_process(&profile_dir);
    }

    // 3. Remove wrapper script.
    if has_wrapper && remove_wrapper_script(&canon) {
        println!("✓ Removed {}", wrapper_path.display());
    }

    // 4. Remove profile directory.
    match fs::remove_dir_all(&profile_dir) {
        Ok(()) => println!("✓ Removed {}", profile_dir.display()),
        Err(e) => println!("⚠ Could not remove {}: {e}", profile_dir.display()),
    }

    // 5. Clear active_profile if it pointed to this profile.
    if get_active_profile() == canon {
        let _ = set_active_profile("default");
        println!("✓ Active profile reset to default");
    }

    println!("\nProfile '{canon}' deleted.");
    Ok(profile_dir)
}

/// Disable and remove the systemd/launchd service for a profile.
///
/// The full service-name derivation lives in the gateway module; this performs
/// the platform-specific teardown using the conventional names, mirroring the
/// Python original's best-effort behaviour. Honours `HERMES_PROFILE_PYTHON`-style
/// stubs in tests by gracefully ignoring missing tools.
fn cleanup_gateway_service(name: &str, profile_dir: &Path) {
    // Profile suffix mirrors the gateway's per-profile service naming: the
    // default profile has no suffix; named profiles append `-<name>`.
    let canon = name;
    let svc_name = if canon == "default" {
        "hermes-gateway".to_string()
    } else {
        format!("hermes-gateway-{canon}")
    };
    let _ = profile_dir; // profile_dir is used by the gateway module proper.

    if cfg!(target_os = "linux") {
        let svc_file = home_dir()
            .join(".config")
            .join("systemd")
            .join("user")
            .join(format!("{svc_name}.service"));
        if svc_file.exists() {
            let _ = run_quiet("systemctl", &["--user", "disable", &svc_name]);
            let _ = run_quiet("systemctl", &["--user", "stop", &svc_name]);
            let _ = fs::remove_file(&svc_file);
            let _ = run_quiet("systemctl", &["--user", "daemon-reload"]);
            println!("✓ Service {svc_name} removed");
        }
    } else if cfg!(target_os = "macos") {
        let plist_path = home_dir()
            .join("Library")
            .join("LaunchAgents")
            .join(format!("com.{svc_name}.plist"));
        if plist_path.exists() {
            let _ = run_quiet("launchctl", &["unload", &plist_path.to_string_lossy()]);
            let _ = fs::remove_file(&plist_path);
            println!("✓ Launchd service removed");
        }
    }
}

fn run_quiet(program: &str, args: &[&str]) -> std::result::Result<(), io::Error> {
    std::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|_| ())
}

/// Stop a running gateway process via its PID file (SIGTERM, then SIGKILL).
fn stop_gateway_process(profile_dir: &Path) {
    let pid_file = profile_dir.join("gateway.pid");
    if !pid_file.exists() {
        return;
    }

    let raw = match fs::read_to_string(&pid_file) {
        Ok(s) => s.trim().to_string(),
        Err(_) => return,
    };
    let pid: Option<i64> = if raw.starts_with('{') {
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|v| v.get("pid").and_then(|p| p.as_i64()))
    } else {
        raw.parse::<i64>().ok()
    };
    let pid = match pid {
        Some(p) => p,
        None => return,
    };

    #[cfg(unix)]
    {
        unsafe {
            if libc::kill(pid as libc::pid_t, libc::SIGTERM) != 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::ESRCH)
                    || err.raw_os_error() == Some(libc::EPERM)
                {
                    println!("✓ Gateway already stopped");
                    return;
                }
                println!("⚠ Could not stop gateway: {err}");
                return;
            }
        }
        // Wait up to 10s for graceful shutdown.
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let alive = unsafe { libc::kill(pid as libc::pid_t, 0) == 0 };
            if !alive {
                println!("✓ Gateway stopped (PID {pid})");
                return;
            }
        }
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
        println!("✓ Gateway force-stopped (PID {pid})");
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        println!("✓ Gateway already stopped");
    }
}

// ---------------------------------------------------------------------------
// Active profile (sticky default)
// ---------------------------------------------------------------------------

/// Read the sticky active profile name (`"default"` if unset/empty).
pub fn get_active_profile() -> String {
    let path = get_active_profile_path();
    match fs::read_to_string(&path) {
        Ok(s) => {
            let name = s.trim();
            if name.is_empty() {
                "default".to_string()
            } else {
                name.to_string()
            }
        }
        Err(_) => "default".to_string(),
    }
}

/// Set the sticky active profile. Use `"default"` to clear.
pub fn set_active_profile(name: &str) -> Result<()> {
    let canon = normalize_profile_name(name)?;
    validate_profile_name(&canon)?;
    if canon != "default" && !profile_exists(&canon) {
        return Err(ProfileError::NotFound(format!(
            "Profile '{canon}' does not exist. Create it with: hermes profile create {canon}"
        )));
    }

    let path = get_active_profile_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if canon == "default" {
        let _ = fs::remove_file(&path);
    } else {
        // Atomic write via a sibling .tmp file.
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, format!("{canon}\n"))?;
        fs::rename(&tmp, &path)?;
    }
    Ok(())
}

/// Infer the current profile name from `HERMES_HOME`.
///
/// `"default"` if `HERMES_HOME` is unset or points to `~/.hermes`; the profile
/// name if it points into `~/.hermes/profiles/<name>`; `"custom"` otherwise.
pub fn get_active_profile_name() -> String {
    let hermes_home = get_hermes_home();
    let resolved = resolve(&hermes_home);

    let default_resolved = resolve(&get_default_hermes_home());
    if resolved == default_resolved {
        return "default".to_string();
    }

    let profiles_root = resolve(&get_profiles_root());
    if let Ok(rel) = resolved.strip_prefix(&profiles_root) {
        let parts: Vec<&str> = rel
            .components()
            .filter_map(|c| match c {
                Component::Normal(s) => s.to_str(),
                _ => None,
            })
            .collect();
        if parts.len() == 1 && profile_id_re().is_match(parts[0]) {
            return parts[0].to_string();
        }
    }

    "custom".to_string()
}

// ---------------------------------------------------------------------------
// Export / Import
// ---------------------------------------------------------------------------

/// Decide whether an export entry should be ignored.
///
/// Universal exclusions at any depth: `__pycache__`, `*.sock`, `*.tmp`,
/// `package.json`, `package-lock.json`. Root-level (`is_root`) exclusions add
/// [`DEFAULT_EXPORT_EXCLUDE_ROOT`] (only for the default profile export).
fn default_export_ignore(name: &str, is_root: bool, default_profile_root: bool) -> bool {
    if name == "__pycache__" || name.ends_with(".sock") || name.ends_with(".tmp") {
        return true;
    }
    if name == "package.json" || name == "package-lock.json" {
        return true;
    }
    if is_root && default_profile_root && DEFAULT_EXPORT_EXCLUDE_ROOT.contains(&name) {
        return true;
    }
    false
}

/// Export a profile to a `.tar.gz` archive. Returns the output file path.
pub fn export_profile(name: &str, output_path: &str) -> Result<PathBuf> {
    let canon = normalize_profile_name(name)?;
    validate_profile_name(&canon)?;
    let profile_dir = get_profile_dir(&canon)?;
    if !profile_dir.is_dir() {
        return Err(ProfileError::NotFound(format!(
            "Profile '{canon}' does not exist."
        )));
    }

    // `make_archive` wants the base name without extension; `.gztar` re-appends
    // `.tar.gz`. Reproduce that exactly: strip a `.tar.gz`/`.tgz` suffix, then
    // append `.tar.gz`.
    let base = output_path
        .strip_suffix(".tar.gz")
        .or_else(|| output_path.strip_suffix(".tgz"))
        .unwrap_or(output_path);
    let final_output = PathBuf::from(format!("{base}.tar.gz"));

    if let Some(parent) = final_output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let default_profile = canon == "default";
    // Default profile archives root at `default/`; named profiles at `<canon>/`.
    let root_name = if default_profile { "default" } else { &canon };

    let file = fs::File::create(&final_output)?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut builder = Builder::new(encoder);

    append_dir_entry(&mut builder, Path::new(root_name), &profile_dir)?;
    walk_export_entries(
        &mut builder,
        &profile_dir,
        Path::new(root_name),
        0,
        default_profile,
    )?;
    builder.finish().map_err(ProfileError::from)?;

    Ok(final_output)
}

fn walk_export_entries(
    builder: &mut Builder<GzEncoder<fs::File>>,
    source: &Path,
    archive_root: &Path,
    depth: usize,
    default_profile: bool,
) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(source)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_root = depth == 0;
        // Named-profile credential stripping at root.
        if !default_profile && is_root && (name == "auth.json" || name == ".env") {
            continue;
        }
        if default_export_ignore(&name, is_root, default_profile) {
            continue;
        }
        let archive_path = archive_root.join(entry.file_name());
        if path.is_dir() {
            append_dir_entry(builder, &archive_path, &path)?;
            walk_export_entries(builder, &path, &archive_path, depth + 1, default_profile)?;
        } else if path.is_file() {
            builder
                .append_path_with_name(&path, &archive_path)
                .map_err(ProfileError::from)?;
        }
    }
    Ok(())
}

fn append_dir_entry(
    builder: &mut Builder<GzEncoder<fs::File>>,
    archive_path: &Path,
    source: &Path,
) -> Result<()> {
    let metadata = fs::metadata(source)?;
    let mut header = Header::new_gnu();
    header
        .set_path(archive_path)
        .map_err(ProfileError::from)?;
    header.set_entry_type(EntryType::Directory);
    header.set_size(0);
    #[cfg(unix)]
    header.set_mode(metadata.permissions().mode());
    #[cfg(not(unix))]
    {
        let _ = metadata;
        header.set_mode(0o755);
    }
    header.set_cksum();
    builder
        .append(&header, io::empty())
        .map_err(ProfileError::from)?;
    Ok(())
}

/// Return safe path parts for a profile archive member, rejecting escapes.
fn normalize_profile_archive_parts(member_name: &str) -> Result<Vec<String>> {
    let normalized = member_name.replace('\\', "/");
    let path = Path::new(&normalized);

    // Reject absolute / drive-prefixed / windows-absolute paths.
    let is_windows_absolute = {
        // e.g. "C:\\foo" or "C:/foo" after replacing backslashes.
        let bytes = member_name.as_bytes();
        (bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic())
            || member_name.starts_with('\\')
    };
    if normalized.is_empty() || path.is_absolute() || is_windows_absolute {
        return Err(ProfileError::Value(format!(
            "Unsafe archive member path: {member_name}"
        )));
    }

    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => {
                let value = value.to_string_lossy();
                if value.is_empty() || value == "." {
                    continue;
                }
                if value == ".." {
                    return Err(ProfileError::Value(format!(
                        "Unsafe archive member path: {member_name}"
                    )));
                }
                parts.push(value.to_string());
            }
            Component::CurDir => {}
            _ => {
                return Err(ProfileError::Value(format!(
                    "Unsafe archive member path: {member_name}"
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(ProfileError::Value(format!(
            "Unsafe archive member path: {member_name}"
        )));
    }
    Ok(parts)
}

/// Extract a profile archive without allowing path escapes or links.
fn safe_extract_profile_archive(archive_bytes: &[u8], destination: &Path) -> Result<()> {
    let cursor = io::Cursor::new(archive_bytes);
    let decoder = GzDecoder::new(cursor);
    let mut archive = Archive::new(decoder);
    for entry in archive.entries().map_err(ProfileError::from)? {
        let mut entry = entry.map_err(ProfileError::from)?;
        let raw = entry.path().map_err(ProfileError::from)?;
        let member_name = raw.to_string_lossy().to_string();
        let parts = normalize_profile_archive_parts(&member_name)?;
        let target = parts
            .iter()
            .fold(destination.to_path_buf(), |acc, part| acc.join(part));

        let entry_type = entry.header().entry_type();
        if entry_type.is_dir() {
            fs::create_dir_all(&target)?;
            continue;
        }
        if !entry_type.is_file() {
            return Err(ProfileError::Value(format!(
                "Unsupported archive member type: {member_name}"
            )));
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).map_err(ProfileError::from)?;
        fs::write(&target, &buf)?;

        #[cfg(unix)]
        if let Ok(mode) = entry.header().mode() {
            let _ = fs::set_permissions(&target, fs::Permissions::from_mode(mode & 0o777));
        }
    }
    Ok(())
}

/// Return the archive's top-level directory names.
fn inspect_profile_archive_roots(archive_bytes: &[u8]) -> Result<BTreeSet<String>> {
    let cursor = io::Cursor::new(archive_bytes);
    let decoder = GzDecoder::new(cursor);
    let mut archive = Archive::new(decoder);
    let mut top_dirs = BTreeSet::new();
    let mut dir_roots = BTreeSet::new();
    for entry in archive.entries().map_err(ProfileError::from)? {
        let entry = entry.map_err(ProfileError::from)?;
        let raw = entry.path().map_err(ProfileError::from)?;
        let member_name = raw.to_string_lossy().to_string();
        let parts = normalize_profile_archive_parts(&member_name)?;
        let is_dir = entry.header().entry_type().is_dir();
        if parts.len() > 1 || is_dir {
            top_dirs.insert(parts[0].clone());
        }
        if is_dir {
            dir_roots.insert(parts[0].clone());
        }
    }
    if top_dirs.is_empty() {
        return Ok(dir_roots);
    }
    Ok(top_dirs)
}

/// Import a profile from a `.tar.gz` archive.
///
/// If `name` is not given, infers it from the archive's top-level directory.
/// Returns the imported profile directory.
pub fn import_profile(archive_path: &str, name: Option<&str>) -> Result<PathBuf> {
    let archive = PathBuf::from(archive_path);
    if !archive.exists() {
        return Err(ProfileError::NotFound(format!(
            "Archive not found: {}",
            archive.display()
        )));
    }

    let bytes = fs::read(&archive)?;
    let top_dirs = inspect_profile_archive_roots(&bytes)?;
    let archive_root: Option<String> = if top_dirs.len() == 1 {
        top_dirs.iter().next().cloned()
    } else {
        None
    };

    let inferred_name = name
        .map(str::to_string)
        .or_else(|| archive_root.clone());
    let inferred_name = match inferred_name {
        Some(n) => n,
        None => {
            return Err(ProfileError::Value(
                "Cannot determine profile name from archive. Specify it explicitly: \
                 hermes profile import <archive> --name <name>"
                    .to_string(),
            ));
        }
    };
    let archive_root = match archive_root {
        Some(r) => r,
        None => {
            return Err(ProfileError::Value(
                "Profile archive must contain exactly one top-level directory.".to_string(),
            ));
        }
    };

    let canon = normalize_profile_name(&inferred_name)?;
    validate_profile_name(&canon)?;
    if canon == "default" {
        return Err(ProfileError::Value(
            "Cannot import as 'default' — that is the built-in root profile (~/.hermes). \
             Specify a different name: hermes profile import <archive> --name <name>"
                .to_string(),
        ));
    }

    let profile_dir = get_profile_dir(&canon)?;
    if profile_dir.exists() {
        return Err(ProfileError::Exists(format!(
            "Profile '{canon}' already exists at {}",
            profile_dir.display()
        )));
    }

    let profiles_root = get_profiles_root();
    fs::create_dir_all(&profiles_root)?;

    let staging = tempfile::Builder::new()
        .prefix("hermes_profile_import_")
        .tempdir()?;
    let staging_root = staging.path();
    safe_extract_profile_archive(&bytes, staging_root)?;

    let extracted = staging_root.join(&archive_root);
    if !extracted.is_dir() {
        return Err(ProfileError::Value(format!(
            "Profile archive root is missing or invalid: {archive_root}"
        )));
    }

    let final_source = if archive_root != canon {
        let renamed = staging_root.join(&canon);
        fs::rename(&extracted, &renamed)?;
        renamed
    } else {
        extracted
    };

    move_path(&final_source, &profile_dir)?;
    Ok(profile_dir)
}

/// Move `source` to `dest`, falling back to copy+remove across filesystems.
fn move_path(source: &Path, dest: &Path) -> Result<()> {
    if fs::rename(source, dest).is_ok() {
        return Ok(());
    }
    if source.is_dir() {
        copy_dir_recursive(source, dest, &|_, _| false)?;
        fs::remove_dir_all(source)?;
    } else {
        fs::copy(source, dest)?;
        fs::remove_file(source)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rename
// ---------------------------------------------------------------------------

/// Rename Honcho host blocks for a renamed profile without changing peers.
fn migrate_honcho_profile_host(old_name: &str, new_name: &str, new_dir: &Path) {
    let old_host = format!("hermes.{old_name}");
    let new_host = format!("hermes.{new_name}");

    let candidates = [
        new_dir.join("honcho.json"),
        get_default_hermes_home().join("honcho.json"),
        home_dir().join(".honcho").join("config.json"),
    ];

    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for path in candidates {
        let resolved = resolve(&path);
        if seen.contains(&resolved) || !path.is_file() {
            continue;
        }
        seen.insert(resolved);

        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut data: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let hosts = match data.get_mut("hosts").and_then(|h| h.as_object_mut()) {
            Some(h) => h,
            None => continue,
        };
        if !hosts.contains_key(&old_host) {
            continue;
        }
        if hosts.contains_key(&new_host) {
            println!(
                "⚠ Honcho host block not migrated: {new_host} already exists in {}",
                path.display()
            );
            continue;
        }

        let mut block = hosts.remove(&old_host).unwrap();
        if let Some(obj) = block.as_object_mut() {
            if !obj.contains_key("aiPeer") {
                let bare = old_host
                    .split_once('.')
                    .map(|(_, rest)| rest.to_string())
                    .unwrap_or_else(|| old_host.clone());
                obj.insert("aiPeer".to_string(), serde_json::Value::String(bare));
            }
        }
        hosts.insert(new_host.clone(), block);

        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        let serialized = match serde_json::to_string_pretty(&data) {
            Ok(s) => format!("{s}\n"),
            Err(_) => continue,
        };
        if fs::write(&tmp, serialized).is_err() {
            let _ = fs::remove_file(&tmp);
            continue;
        }
        if fs::rename(&tmp, &path).is_err() {
            let _ = fs::remove_file(&tmp);
            continue;
        }

        println!("✓ Honcho host updated: {old_host} → {new_host}");
    }
}

/// Rename a profile: directory, wrapper script, service, active_profile.
/// Returns the new profile directory.
pub fn rename_profile(old_name: &str, new_name: &str) -> Result<PathBuf> {
    let old_canon = normalize_profile_name(old_name)?;
    let new_canon = normalize_profile_name(new_name)?;
    validate_profile_name(&old_canon)?;
    validate_profile_name(&new_canon)?;

    if old_canon == "default" {
        return Err(ProfileError::Value(
            "Cannot rename the default profile.".to_string(),
        ));
    }
    if new_canon == "default" {
        return Err(ProfileError::Value(
            "Cannot rename to 'default' — it is reserved.".to_string(),
        ));
    }

    let old_dir = get_profile_dir(&old_canon)?;
    let new_dir = get_profile_dir(&new_canon)?;

    if !old_dir.is_dir() {
        return Err(ProfileError::NotFound(format!(
            "Profile '{old_canon}' does not exist."
        )));
    }
    if new_dir.exists() {
        return Err(ProfileError::Exists(format!(
            "Profile '{new_canon}' already exists."
        )));
    }

    // 1. Stop gateway if running.
    if check_gateway_running(&old_dir) {
        cleanup_gateway_service(&old_canon, &old_dir);
        stop_gateway_process(&old_dir);
    }

    // 2. Rename directory.
    let old_label = old_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| old_canon.clone());
    let new_label = new_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| new_canon.clone());
    fs::rename(&old_dir, &new_dir)?;
    println!("✓ Renamed {old_label} → {new_label}");

    // 3. Update profile-scoped Honcho host blocks.
    migrate_honcho_profile_host(&old_canon, &new_canon, &new_dir);

    // 4. Update wrapper script.
    remove_wrapper_script(&old_canon);
    match check_alias_collision(&new_canon)? {
        None => {
            create_wrapper_script(&new_canon);
            println!("✓ Alias updated: {new_canon}");
        }
        Some(collision) => {
            println!("⚠ Cannot create alias '{new_canon}' — {collision}");
        }
    }

    // 5. Update active_profile if it pointed to old name.
    if get_active_profile() == old_canon {
        let _ = set_active_profile(&new_canon);
        println!("✓ Active profile updated: {new_canon}");
    }

    Ok(new_dir)
}

// ---------------------------------------------------------------------------
// Recursive copy helper (shutil.copytree analogue)
// ---------------------------------------------------------------------------

/// Copy `source` into `destination` recursively, with `dirs_exist_ok=True`
/// semantics. `skip(path, depth)` decides whether to ignore an entry (depth 0
/// is the top-level call on `source`).
fn copy_dir_recursive(
    source: &Path,
    destination: &Path,
    skip: &dyn Fn(&Path, usize) -> bool,
) -> Result<()> {
    fn walk(
        source: &Path,
        destination: &Path,
        depth: usize,
        skip: &dyn Fn(&Path, usize) -> bool,
    ) -> Result<()> {
        if skip(source, depth) {
            return Ok(());
        }
        fs::create_dir_all(destination)?;
        let mut entries: Vec<_> = fs::read_dir(source)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let src = entry.path();
            let dest = destination.join(entry.file_name());
            if skip(&src, depth + 1) {
                continue;
            }
            if src.is_dir() {
                walk(&src, &dest, depth + 1, skip)?;
            } else {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&src, &dest)?;
            }
        }
        Ok(())
    }
    walk(source, destination, 0, skip)
}

// ---------------------------------------------------------------------------
// Tab completion
// ---------------------------------------------------------------------------

/// Generate a bash completion script for hermes profile names.
pub fn generate_bash_completion() -> String {
    r#"# Hermes Agent profile completion
# Add to ~/.bashrc: eval "$(hermes completion bash)"

_hermes_profiles() {
    local profiles_dir="$HOME/.hermes/profiles"
    local profiles="default"
    if [ -d "$profiles_dir" ]; then
        profiles="$profiles $(ls "$profiles_dir" 2>/dev/null)"
    fi
    echo "$profiles"
}

_hermes_completion() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Complete profile names after -p / --profile
    if [[ "$prev" == "-p" || "$prev" == "--profile" ]]; then
        COMPREPLY=($(compgen -W "$(_hermes_profiles)" -- "$cur"))
        return
    fi

    # Complete profile subcommands
    if [[ "${COMP_WORDS[1]}" == "profile" ]]; then
        case "$prev" in
            profile)
                COMPREPLY=($(compgen -W "list use create delete show alias rename export import" -- "$cur"))
                return
                ;;
            use|delete|show|alias|rename|export)
                COMPREPLY=($(compgen -W "$(_hermes_profiles)" -- "$cur"))
                return
                ;;
        esac
    fi

    # Top-level subcommands
    if [[ "$COMP_CWORD" == 1 ]]; then
        local commands="chat model gateway setup status cron doctor dump config skills tools mcp sessions profile update version"
        COMPREPLY=($(compgen -W "$commands" -- "$cur"))
    fi
}

complete -F _hermes_completion hermes
"#
    .to_string()
}

/// Generate a zsh completion script for hermes profile names.
pub fn generate_zsh_completion() -> String {
    r#"#compdef hermes
# Hermes Agent profile completion
# Add to ~/.zshrc: eval "$(hermes completion zsh)"

_hermes() {
    local -a profiles
    profiles=(default)
    if [[ -d "$HOME/.hermes/profiles" ]]; then
        profiles+=("${(@f)$(ls $HOME/.hermes/profiles 2>/dev/null)}")
    fi

    _arguments \
        '-p[Profile name]:profile:($profiles)' \
        '--profile[Profile name]:profile:($profiles)' \
        '1:command:(chat model gateway setup status cron doctor dump config skills tools mcp sessions profile update version)' \
        '*::arg:->args'

    case $words[1] in
        profile)
            _arguments '1:action:(list use create delete show alias rename export import)' \
                        '2:profile:($profiles)'
            ;;
    esac
}

_hermes "$@"
"#
    .to_string()
}

// ---------------------------------------------------------------------------
// Profile env resolution
// ---------------------------------------------------------------------------

/// Resolve a profile name to a `HERMES_HOME` path string.
///
/// Called early in the CLI entry point, before any hermes modules are imported,
/// to set the `HERMES_HOME` environment variable.
pub fn resolve_profile_env(profile_name: &str) -> Result<String> {
    let canon = normalize_profile_name(profile_name)?;
    validate_profile_name(&canon)?;
    let profile_dir = get_profile_dir(&canon)?;

    if canon != "default" && !profile_dir.is_dir() {
        return Err(ProfileError::NotFound(format!(
            "Profile '{canon}' does not exist. Create it with: hermes profile create {canon}"
        )));
    }

    Ok(profile_dir.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    // Serialize tests that mutate process-global env (HERMES_HOME / HOME).
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        home: Option<std::ffi::OsString>,
        hermes_home: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(home: &Path) -> Self {
            let guard = EnvGuard {
                home: env::var_os("HOME"),
                hermes_home: env::var_os("HERMES_HOME"),
            };
            unsafe {
                unsafe { env::set_var("HOME", home); }
                unsafe { env::remove_var("HERMES_HOME"); }
            }
            guard
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.home {
                    Some(v) => env::set_var("HOME", v),
                    None => env::remove_var("HOME"),
                }
                match &self.hermes_home {
                    Some(v) => env::set_var("HERMES_HOME", v),
                    None => env::remove_var("HERMES_HOME"),
                }
            }
        }
    }

    #[test]
    fn normalize_and_validate_names() {
        assert_eq!(normalize_profile_name("Default").unwrap(), "default");
        assert_eq!(normalize_profile_name("  Coder ").unwrap(), "coder");
        assert!(normalize_profile_name("   ").is_err());

        assert!(validate_profile_name("default").is_ok());
        assert!(validate_profile_name("coder-1").is_ok());
        assert!(validate_profile_name("9lives").is_ok());
        assert!(validate_profile_name("-bad").is_err());
        assert!(validate_profile_name("Bad").is_err());
        assert!(validate_profile_name("").is_err());
    }

    #[test]
    fn create_profile_default_is_rejected() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());
        let err = create_profile("default", None, false, false, false).unwrap_err();
        assert!(matches!(err, ProfileError::Value(_)));
    }

    #[test]
    fn create_profile_bootstraps_directories() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let dir = create_profile("coder", None, false, false, true).unwrap();
        assert!(dir.is_dir());
        for sub in PROFILE_DIRS {
            assert!(dir.join(sub).is_dir(), "missing {sub}");
        }
        // Re-creating must fail.
        let err = create_profile("coder", None, false, false, true).unwrap_err();
        assert!(matches!(err, ProfileError::Exists(_)));
    }

    #[test]
    fn create_profile_clone_copies_config_skills_and_memory() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let source = create_profile("source", None, false, false, true).unwrap();
        fs::write(source.join("config.yaml"), "model:\n  default: test-model\n").unwrap();
        fs::write(source.join(".env"), "OPENAI_API_KEY=test-key\n").unwrap();
        fs::write(source.join("SOUL.md"), "custom soul").unwrap();
        fs::create_dir_all(source.join("skills").join("team").join("demo")).unwrap();
        fs::write(
            source.join("skills").join("team").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();
        fs::write(source.join("memories").join("MEMORY.md"), "memory").unwrap();
        fs::write(source.join("memories").join("USER.md"), "user").unwrap();

        let dest = create_profile("clone", Some("source"), false, true, true).unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("config.yaml")).unwrap(),
            "model:\n  default: test-model\n"
        );
        assert_eq!(
            fs::read_to_string(dest.join(".env")).unwrap(),
            "OPENAI_API_KEY=test-key\n"
        );
        assert_eq!(fs::read_to_string(dest.join("SOUL.md")).unwrap(), "custom soul");
        assert!(dest.join("skills").join("team").join("demo").join("SKILL.md").exists());
        assert_eq!(
            fs::read_to_string(dest.join("memories").join("MEMORY.md")).unwrap(),
            "memory"
        );
        assert_eq!(
            fs::read_to_string(dest.join("memories").join("USER.md")).unwrap(),
            "user"
        );
    }

    #[test]
    fn clone_all_skips_nested_profiles_and_runtime_files() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let default_root = get_default_hermes_home();
        fs::create_dir_all(default_root.join("workspace")).unwrap();
        fs::write(default_root.join("workspace").join("note.txt"), "hello").unwrap();
        fs::write(default_root.join("gateway.pid"), "123").unwrap();
        fs::write(default_root.join("processes.json"), "{}").unwrap();
        fs::create_dir_all(default_root.join("profiles").join("other")).unwrap();
        fs::write(
            default_root.join("profiles").join("other").join("marker.txt"),
            "ignore me",
        )
        .unwrap();

        let dest = create_profile("mirror", None, true, false, true).unwrap();
        assert!(dest.join("workspace").join("note.txt").exists());
        assert!(!dest.join("gateway.pid").exists());
        assert!(!dest.join("processes.json").exists());
        assert!(!dest.join("profiles").exists());
    }

    #[test]
    fn read_config_model_handles_string_and_mapping() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path();
        fs::write(dir.join("config.yaml"), "model: gpt-4o\n").unwrap();
        assert_eq!(read_config_model(dir), (Some("gpt-4o".to_string()), None));

        fs::write(
            dir.join("config.yaml"),
            "model:\n  default: claude\n  provider: anthropic\n",
        )
        .unwrap();
        assert_eq!(
            read_config_model(dir),
            (Some("claude".to_string()), Some("anthropic".to_string()))
        );
    }

    #[test]
    fn count_skills_skips_hub_and_git() {
        let temp = TempDir::new().unwrap();
        let skills = temp.path().join("skills");
        fs::create_dir_all(skills.join("a")).unwrap();
        fs::write(skills.join("a").join("SKILL.md"), "x").unwrap();
        fs::create_dir_all(skills.join(".hub").join("b")).unwrap();
        fs::write(skills.join(".hub").join("b").join("SKILL.md"), "x").unwrap();
        fs::create_dir_all(skills.join(".git").join("c")).unwrap();
        fs::write(skills.join(".git").join("c").join("SKILL.md"), "x").unwrap();
        assert_eq!(count_skills(temp.path()), 1);
    }

    #[test]
    fn alias_create_and_remove_roundtrip() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let path = create_wrapper_script("coder").unwrap();
        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("hermes -p coder"));
        assert!(remove_wrapper_script("coder"));
        assert!(!path.exists());
    }

    #[test]
    fn check_alias_collision_rejects_reserved_and_subcommands() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        assert!(
            check_alias_collision("hermes")
                .unwrap()
                .unwrap()
                .contains("reserved")
        );
        assert!(
            check_alias_collision("chat")
                .unwrap()
                .unwrap()
                .contains("conflicts with a hermes subcommand")
        );
    }

    #[test]
    fn active_profile_get_set_clear() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        assert_eq!(get_active_profile(), "default");
        create_profile("coder", None, false, false, true).unwrap();
        set_active_profile("coder").unwrap();
        assert_eq!(get_active_profile(), "coder");
        set_active_profile("default").unwrap();
        assert_eq!(get_active_profile(), "default");
        assert!(!get_active_profile_path().exists());
    }

    #[test]
    fn set_active_profile_requires_existing() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());
        let err = set_active_profile("ghost").unwrap_err();
        assert!(matches!(err, ProfileError::NotFound(_)));
    }

    #[test]
    fn export_and_import_round_trips_without_credentials() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let source = create_profile("coder", None, false, false, true).unwrap();
        fs::write(source.join("config.yaml"), "model:\n  default: imported\n").unwrap();
        fs::write(source.join(".env"), "SECRET=1\n").unwrap();
        fs::write(source.join("auth.json"), "{\"token\":1}\n").unwrap();
        fs::create_dir_all(source.join("skills").join("demo")).unwrap();
        fs::write(
            source.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();

        let archive = temp.path().join("coder.tar.gz");
        let out = export_profile("coder", archive.to_str().unwrap()).unwrap();
        assert_eq!(out, archive);

        let imported = import_profile(archive.to_str().unwrap(), Some("builder")).unwrap();
        assert_eq!(imported, get_profile_dir("builder").unwrap());
        assert_eq!(
            fs::read_to_string(imported.join("config.yaml")).unwrap(),
            "model:\n  default: imported\n"
        );
        assert!(imported.join("skills").join("demo").join("SKILL.md").exists());
        assert!(!imported.join(".env").exists());
        assert!(!imported.join("auth.json").exists());
    }

    #[test]
    fn import_rejects_default_name() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let source = create_profile("coder", None, false, false, true).unwrap();
        fs::write(source.join("config.yaml"), "model: x\n").unwrap();
        let archive = temp.path().join("coder.tar.gz");
        export_profile("coder", archive.to_str().unwrap()).unwrap();
        let err = import_profile(archive.to_str().unwrap(), Some("default")).unwrap_err();
        assert!(matches!(err, ProfileError::Value(_)));
    }

    #[test]
    fn normalize_archive_parts_rejects_escapes() {
        assert!(normalize_profile_archive_parts("../etc/passwd").is_err());
        assert!(normalize_profile_archive_parts("/abs/path").is_err());
        assert!(normalize_profile_archive_parts("C:\\windows").is_err());
        assert_eq!(
            normalize_profile_archive_parts("coder/config.yaml").unwrap(),
            vec!["coder".to_string(), "config.yaml".to_string()]
        );
    }

    #[test]
    fn rename_moves_directory_and_alias() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let old_dir = create_profile("coder", None, false, false, true).unwrap();
        create_wrapper_script("coder").unwrap();
        set_active_profile("coder").unwrap();

        let new_dir = rename_profile("coder", "builder").unwrap();
        assert!(!old_dir.exists());
        assert!(new_dir.exists());
        assert!(!get_wrapper_dir().join("coder").exists());
        assert!(get_wrapper_dir().join("builder").exists());
        assert_eq!(get_active_profile(), "builder");
    }

    #[test]
    fn list_profiles_includes_default_and_named() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        fs::create_dir_all(get_default_hermes_home()).unwrap();
        let coder = create_profile("coder", None, false, false, true).unwrap();
        fs::write(
            coder.join("config.yaml"),
            "model:\n  default: gpt-4.1-mini\n  provider: openai\n",
        )
        .unwrap();
        fs::write(coder.join("skills").join("SKILL.md"), "x").unwrap();
        create_wrapper_script("coder").unwrap();

        let rows = list_profiles();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "default");
        assert!(rows[0].is_default);
        assert_eq!(rows[1].name, "coder");
        assert_eq!(rows[1].model.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(rows[1].provider.as_deref(), Some("openai"));
        assert_eq!(rows[1].skill_count, 1);
        assert!(rows[1].alias_path.is_some());
    }

    #[test]
    fn completion_scripts_contain_expected_markers() {
        assert!(generate_bash_completion().contains("complete -F _hermes_completion hermes"));
        assert!(generate_zsh_completion().contains("#compdef hermes"));
    }

    #[test]
    fn resolve_profile_env_returns_path_or_errors() {
        let _g = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let _env = EnvGuard::set(temp.path());

        let dir = create_profile("coder", None, false, false, true).unwrap();
        assert_eq!(
            resolve_profile_env("coder").unwrap(),
            dir.to_string_lossy().to_string()
        );
        assert!(resolve_profile_env("ghost").is_err());
        // default always resolves.
        assert_eq!(
            resolve_profile_env("default").unwrap(),
            get_default_hermes_home().to_string_lossy().to_string()
        );
    }
}
