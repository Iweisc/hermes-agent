//! Checkpoint Manager — Transparent filesystem snapshots via a single shared
//! shadow git store.
//!
//! Native Rust port of `tools/checkpoint_manager.py`.
//!
//! Creates automatic snapshots of working directories before file-mutating
//! operations, triggered once per conversation turn, and provides rollback to
//! any previous checkpoint.
//!
//! This is NOT a tool — the LLM never sees it. It is transparent
//! infrastructure controlled by the `checkpoints` config flag.
//!
//! # Storage layout (single shared store, git objects deduplicated across projects)
//!
//! ```text
//! <base>/                                  (e.g. ~/.hermes/checkpoints)
//!     store/                               — single bare-ish git repo
//!         HEAD, config, objects/           — standard git internals (shared)
//!         refs/hermes/<hash16>             — per-project branch tip
//!         indexes/<hash16>                 — per-project git index
//!         projects/<hash16>.json           — {workdir, created_at, last_touch}
//!         info/exclude                     — default excludes (shared)
//!     .last_prune                          — auto-prune idempotency marker
//!     legacy-<timestamp>/                  — archived pre-v2 per-project shadow repos
//! ```
//!
//! The shadow store uses `GIT_DIR` + `GIT_WORK_TREE` + `GIT_INDEX_FILE` so no
//! git state leaks into the user's project directory.

use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const STORE_DIRNAME: &str = "store";
pub const REFS_PREFIX: &str = "refs/hermes";
pub const INDEXES_DIRNAME: &str = "indexes";
pub const PROJECTS_DIRNAME: &str = "projects";
pub const LEGACY_PREFIX: &str = "legacy-";
pub const PRUNE_MARKER_NAME: &str = ".last_prune";

/// Max files to snapshot — skip huge directories to avoid slowdowns.
pub const MAX_FILES: usize = 50_000;

/// Default excludes written to `info/exclude`.
pub const DEFAULT_EXCLUDES: &[&str] = &[
    // Dependency / build output
    "node_modules/",
    "dist/",
    "build/",
    "target/",
    "out/",
    ".next/",
    ".nuxt/",
    // Caches
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "coverage/",
    ".coverage",
    // Virtualenvs
    ".venv/",
    "venv/",
    "env/",
    // VCS
    ".git/",
    ".hg/",
    ".svn/",
    // Worktrees
    ".worktrees/",
    // Native / compiled binaries
    "*.so",
    "*.dylib",
    "*.dll",
    "*.o",
    "*.a",
    "*.jar",
    "*.class",
    "*.exe",
    "*.obj",
    // Media / large binaries
    "*.mp4",
    "*.mov",
    "*.mkv",
    "*.webm",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.iso",
    // Secrets
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    // OS junk
    ".DS_Store",
    "Thumbs.db",
    // Logs
    "*.log",
];

/// Git subprocess timeout (seconds), clamped to [10, 60] and read from
/// `HERMES_CHECKPOINT_TIMEOUT`.
pub fn git_timeout_secs() -> u64 {
    let raw = std::env::var("HERMES_CHECKPOINT_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(30);
    raw.clamp(10, 60) as u64
}

fn commit_hash_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[0-9a-fA-F]{4,64}$").unwrap())
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

/// Validate a commit hash to prevent git argument injection.
///
/// Returns an error string if invalid, `None` if valid. Values starting with
/// `-` would be interpreted as git flags instead of revision specifiers.
pub fn validate_commit_hash(commit_hash: &str) -> Option<String> {
    if commit_hash.trim().is_empty() {
        return Some("Empty commit hash".to_string());
    }
    if commit_hash.starts_with('-') {
        return Some(format!(
            "Invalid commit hash (must not start with '-'): {commit_hash:?}"
        ));
    }
    if !commit_hash_re().is_match(commit_hash) {
        return Some(format!(
            "Invalid commit hash (expected 4-64 hex characters): {commit_hash:?}"
        ));
    }
    None
}

/// Validate a file path to prevent path traversal outside the working directory.
///
/// Returns an error string if invalid, `None` if valid.
pub fn validate_file_path(file_path: &str, working_dir: &str) -> Option<String> {
    if file_path.trim().is_empty() {
        return Some("Empty file path".to_string());
    }
    if Path::new(file_path).is_absolute() {
        return Some(format!(
            "File path must be relative, got absolute path: {file_path:?}"
        ));
    }
    let abs_workdir = normalize_path(working_dir);
    let resolved = resolve_path(&abs_workdir.join(file_path));
    if !resolved.starts_with(&abs_workdir) {
        return Some(format!(
            "File path escapes the working directory via traversal: {file_path:?}"
        ));
    }
    None
}

// ---------------------------------------------------------------------------
// Path / hash helpers
// ---------------------------------------------------------------------------

/// Expand a leading `~` to the user's home directory.
fn expanduser(path_value: &str) -> PathBuf {
    if path_value == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path_value.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path_value)
}

/// Lexically resolve a path: make absolute (against cwd) and collapse `.`/`..`
/// without requiring the path to exist (mirrors `Path.resolve()`'s behaviour
/// for non-existent paths well enough for our checks).
fn resolve_path(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    let mut out: Vec<std::ffi::OsString> = Vec::new();
    let mut prefix: Option<std::ffi::OsString> = None;
    for comp in abs.components() {
        use std::path::Component;
        match comp {
            Component::Prefix(p) => prefix = Some(p.as_os_str().to_os_string()),
            Component::RootDir => {}
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(seg) => out.push(seg.to_os_string()),
        }
    }
    let mut result = PathBuf::new();
    if let Some(p) = prefix {
        result.push(p);
    }
    result.push(std::path::MAIN_SEPARATOR.to_string());
    for seg in out {
        result.push(seg);
    }
    result
}

/// Return a canonical absolute path for checkpoint operations.
pub fn normalize_path(path_value: &str) -> PathBuf {
    let expanded = expanduser(path_value);
    let expanded_str = expanded.to_string_lossy().into_owned();
    // Prefer real canonicalization when the path exists (resolves symlinks),
    // otherwise fall back to a lexical resolve.
    match fs::canonicalize(&expanded) {
        Ok(p) => p,
        Err(_) => resolve_path(Path::new(&expanded_str)),
    }
}

/// Deterministic per-project hash: `sha256(abs_path)[:16]`.
pub fn project_hash(working_dir: &str) -> String {
    use sha2::{Digest, Sha256};
    let abs_path = normalize_path(working_dir);
    let mut hasher = Sha256::new();
    hasher.update(abs_path.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    hex[..16].to_string()
}

/// Return the single shared shadow store path.
pub fn store_path(base: &Path) -> PathBuf {
    base.join(STORE_DIRNAME)
}

fn index_path(store: &Path, dir_hash: &str) -> PathBuf {
    store.join(INDEXES_DIRNAME).join(dir_hash)
}

fn ref_name(dir_hash: &str) -> String {
    format!("{REFS_PREFIX}/{dir_hash}")
}

fn project_meta_path(store: &Path, dir_hash: &str) -> PathBuf {
    store.join(PROJECTS_DIRNAME).join(format!("{dir_hash}.json"))
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Git env + runner
// ---------------------------------------------------------------------------

/// Result of running a git command: `(ok, stdout, stderr)`.
pub type GitResult = (bool, String, String);

fn devnull() -> &'static str {
    if cfg!(windows) {
        "NUL"
    } else {
        "/dev/null"
    }
}

/// Apply the shared-store isolation env to a `Command`.
fn apply_git_env(cmd: &mut Command, store: &Path, working_dir: &Path, index_file: Option<&Path>) {
    cmd.env("GIT_DIR", store);
    cmd.env("GIT_WORK_TREE", working_dir);
    cmd.env_remove("GIT_NAMESPACE");
    cmd.env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES");
    match index_file {
        Some(idx) => {
            cmd.env("GIT_INDEX_FILE", idx);
        }
        None => {
            cmd.env_remove("GIT_INDEX_FILE");
        }
    }
    cmd.env("GIT_CONFIG_GLOBAL", devnull());
    cmd.env("GIT_CONFIG_SYSTEM", devnull());
    cmd.env("GIT_CONFIG_NOSYSTEM", "1");
}

/// Run a git command against the shared store. Returns `(ok, stdout, stderr)`.
///
/// `allowed_returncodes` suppresses error logging for known/expected non-zero
/// exits while preserving the normal `ok = (returncode == 0)` contract.
pub fn run_git<S: AsRef<OsStr>>(
    args: &[S],
    store: &Path,
    working_dir: &str,
    timeout: u64,
    allowed_returncodes: &HashSet<i32>,
    index_file: Option<&Path>,
) -> GitResult {
    let normalized_working_dir = normalize_path(working_dir);

    if !normalized_working_dir.exists() {
        let msg = format!(
            "working directory not found: {}",
            normalized_working_dir.display()
        );
        log::error!("Git command skipped: git {} ({msg})", join_args(args));
        return (false, String::new(), msg);
    }
    if !normalized_working_dir.is_dir() {
        let msg = format!(
            "working directory is not a directory: {}",
            normalized_working_dir.display()
        );
        log::error!("Git command skipped: git {} ({msg})", join_args(args));
        return (false, String::new(), msg);
    }

    let mut cmd = Command::new("git");
    cmd.args(args);
    cmd.current_dir(&normalized_working_dir);
    apply_git_env(&mut cmd, store, &normalized_working_dir, index_file);

    // Spawn + wait with a timeout.
    let _ = timeout; // timeout enforced via wait_timeout below
    match run_with_timeout(cmd, Duration::from_secs(timeout)) {
        RunOutcome::Completed { code, stdout, stderr } => {
            let ok = code == Some(0);
            let stdout = stdout.trim_end_matches(['\n', '\r', ' ', '\t']).to_string();
            let stderr = stderr.trim_end_matches(['\n', '\r', ' ', '\t']).to_string();
            let stdout = stdout.trim().to_string();
            let stderr = stderr.trim().to_string();
            if !ok {
                let rc = code.unwrap_or(-1);
                if !allowed_returncodes.contains(&rc) {
                    log::error!(
                        "Git command failed: git {} (rc={rc}) stderr={stderr}",
                        join_args(args)
                    );
                }
            }
            (ok, stdout, stderr)
        }
        RunOutcome::Timeout => {
            let msg = format!("git timed out after {timeout}s: git {}", join_args(args));
            log::error!("{msg}");
            (false, String::new(), msg)
        }
        RunOutcome::NotFound => {
            log::error!("Git executable not found: git {}", join_args(args));
            (false, String::new(), "git not found".to_string())
        }
        RunOutcome::Error(e) => {
            log::error!("Unexpected git error running git {}: {e}", join_args(args));
            (false, String::new(), e)
        }
    }
}

fn join_args<S: AsRef<OsStr>>(args: &[S]) -> String {
    args.iter()
        .map(|a| a.as_ref().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

enum RunOutcome {
    Completed {
        code: Option<i32>,
        stdout: String,
        stderr: String,
    },
    Timeout,
    NotFound,
    Error(String),
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> RunOutcome {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::thread;

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return RunOutcome::NotFound;
            }
            return RunOutcome::Error(e.to_string());
        }
    };

    // Drain stdout/stderr on threads to avoid deadlock on full pipes.
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let (out_tx, out_rx) = mpsc::channel();
    let (err_tx, err_rx) = mpsc::channel();
    let out_handle = thread::spawn(move || {
        let mut buf = String::new();
        if let Some(p) = stdout_pipe.as_mut() {
            let _ = p.read_to_string(&mut buf);
        }
        let _ = out_tx.send(buf);
    });
    let err_handle = thread::spawn(move || {
        let mut buf = String::new();
        if let Some(p) = stderr_pipe.as_mut() {
            let _ = p.read_to_string(&mut buf);
        }
        let _ = err_tx.send(buf);
    });

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = out_handle.join();
                let _ = err_handle.join();
                let stdout = out_rx.recv().unwrap_or_default();
                let stderr = err_rx.recv().unwrap_or_default();
                return RunOutcome::Completed {
                    code: status.code(),
                    stdout,
                    stderr,
                };
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return RunOutcome::Timeout;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return RunOutcome::Error(e.to_string()),
        }
    }
}

/// Convenience: run git with default timeout and no allowed return codes.
fn git(args: &[&str], store: &Path, working_dir: &str) -> GitResult {
    run_git(args, store, working_dir, git_timeout_secs(), &HashSet::new(), None)
}

// ---------------------------------------------------------------------------
// Directory size / count helpers
// ---------------------------------------------------------------------------

/// Quick file count estimate (stops early if over `MAX_FILES`).
pub fn dir_file_count(path: &Path) -> usize {
    let mut count = 0usize;
    let mut stack: Vec<PathBuf> = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            count += 1;
            if count > MAX_FILES {
                return count;
            }
            let p = entry.path();
            // rglob("*") counts dirs and files; recurse into dirs.
            let is_dir = entry
                .file_type()
                .map(|t| t.is_dir())
                .unwrap_or_else(|_| p.is_dir());
            if is_dir {
                stack.push(p);
            }
        }
    }
    count
}

/// Best-effort recursive size in bytes. Returns 0 on error.
pub fn dir_size_bytes(path: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack: Vec<PathBuf> = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(t) if t.is_file() => {
                    if let Ok(meta) = entry.metadata() {
                        total += meta.len();
                    }
                }
                _ => {
                    // symlink or other — try a plain metadata (no follow)
                    if let Ok(meta) = fs::symlink_metadata(&p) {
                        if meta.is_file() {
                            total += meta.len();
                        } else if meta.is_dir() {
                            stack.push(p);
                        }
                    }
                }
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Store initialisation + legacy migration
// ---------------------------------------------------------------------------

/// Move pre-v2 per-project shadow repos into a `legacy-<ts>/` dir.
///
/// Returns the legacy-archive path, or `None` if nothing to migrate.
pub fn migrate_legacy_store(base: &Path) -> Option<PathBuf> {
    if !base.exists() {
        return None;
    }
    let mut legacy_root: Option<PathBuf> = None;
    let entries = fs::read_dir(base).ok()?;
    let children: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    for child in children {
        let name = match child.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if name == STORE_DIRNAME || name == PRUNE_MARKER_NAME || name.starts_with(LEGACY_PREFIX) {
            continue;
        }
        if legacy_root.is_none() {
            let stamp = strftime_compact(now_secs());
            let root = base.join(format!("{LEGACY_PREFIX}{stamp}"));
            if let Err(exc) = fs::create_dir_all(&root) {
                log::warn!("Could not create legacy archive dir: {exc}");
                return None;
            }
            legacy_root = Some(root);
        }
        let dest = legacy_root.as_ref().unwrap().join(&name);
        if let Err(exc) = move_path(&child, &dest) {
            log::warn!("Could not archive legacy checkpoint {}: {exc}", child.display());
        }
    }
    if let Some(ref root) = legacy_root {
        log::info!(
            "Migrated pre-v2 checkpoint repos to {}. Clear with `hermes checkpoints clear-legacy` when safe.",
            root.display()
        );
    }
    legacy_root
}

/// `%Y%m%d-%H%M%S` of a unix timestamp (UTC).
fn strftime_compact(ts: f64) -> String {
    use chrono::{TimeZone, Utc};
    let dt = Utc
        .timestamp_opt(ts as i64, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap());
    dt.format("%Y%m%d-%H%M%S").to_string()
}

/// Move a path, falling back to recursive copy + remove across filesystems.
fn move_path(src: &Path, dest: &Path) -> std::io::Result<()> {
    match fs::rename(src, dest) {
        Ok(()) => Ok(()),
        Err(_) => {
            if src.is_dir() {
                copy_dir_recursive(src, dest)?;
                fs::remove_dir_all(src)?;
            } else {
                fs::copy(src, dest)?;
                fs::remove_file(src)?;
            }
            Ok(())
        }
    }
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Initialise the shared shadow store if needed. Returns error string or `None`.
pub fn init_store(store: &Path, working_dir: &str) -> Option<String> {
    let base = match store.parent() {
        Some(p) => p.to_path_buf(),
        None => return Some("Invalid store path".to_string()),
    };

    if !store.exists() {
        if let Err(exc) = fs::create_dir_all(&base) {
            return Some(format!("Could not create checkpoint base: {exc}"));
        }
        migrate_legacy_store(&base);
    }

    if store.join("HEAD").exists() {
        return None;
    }

    if let Err(exc) = fs::create_dir_all(store) {
        return Some(format!("Could not create checkpoint base: {exc}"));
    }
    let _ = fs::create_dir_all(store.join(INDEXES_DIRNAME));
    let _ = fs::create_dir_all(store.join(PROJECTS_DIRNAME));

    // `git init --bare` rejects GIT_WORK_TREE, so use a raw command with only
    // config-isolation env vars.
    let mut init_cmd = Command::new("git");
    init_cmd.args(["init", "--bare"]);
    init_cmd.arg(store);
    for k in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ] {
        init_cmd.env_remove(k);
    }
    init_cmd.env("GIT_CONFIG_GLOBAL", devnull());
    init_cmd.env("GIT_CONFIG_SYSTEM", devnull());
    init_cmd.env("GIT_CONFIG_NOSYSTEM", "1");

    match run_with_timeout(init_cmd, Duration::from_secs(git_timeout_secs())) {
        RunOutcome::Completed { code, stderr, .. } => {
            if code != Some(0) {
                return Some(format!("Shadow store init failed: {}", stderr.trim()));
            }
        }
        RunOutcome::Timeout => {
            return Some("Shadow store init failed: timed out".to_string());
        }
        RunOutcome::NotFound => {
            return Some("Shadow store init failed: git not found".to_string());
        }
        RunOutcome::Error(e) => {
            return Some(format!("Shadow store init failed: {e}"));
        }
    }

    let cfg_wd = base.to_string_lossy().into_owned();
    git(&["config", "user.email", "hermes@local"], store, &cfg_wd);
    git(&["config", "user.name", "Hermes Checkpoint"], store, &cfg_wd);
    git(&["config", "commit.gpgsign", "false"], store, &cfg_wd);
    git(&["config", "tag.gpgSign", "false"], store, &cfg_wd);
    git(&["config", "gc.auto", "0"], store, &cfg_wd);

    let info_dir = store.join("info");
    let _ = fs::create_dir_all(&info_dir);
    let mut excludes = DEFAULT_EXCLUDES.join("\n");
    excludes.push('\n');
    let _ = fs::write(info_dir.join("exclude"), excludes);

    let _ = working_dir;
    log::debug!("Initialised checkpoint store at {}", store.display());
    None
}

/// Create or update `projects/<hash>.json` with workdir + timestamps.
pub fn register_project(store: &Path, working_dir: &str) {
    let dir_hash = project_hash(working_dir);
    let meta_path = project_meta_path(store, &dir_hash);
    let now = now_secs();
    let mut created_at = now;
    if meta_path.exists() {
        if let Ok(text) = fs::read_to_string(&meta_path) {
            if let Ok(Value::Object(existing)) = serde_json::from_str::<Value>(&text) {
                if let Some(c) = existing.get("created_at").and_then(|v| v.as_f64()) {
                    created_at = c;
                }
            }
        }
    }
    let meta = json!({
        "workdir": normalize_path(working_dir).to_string_lossy(),
        "created_at": created_at,
        "last_touch": now,
    });
    if let Some(parent) = meta_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(exc) = fs::write(&meta_path, serde_json::to_string(&meta).unwrap_or_default()) {
        log::debug!("Could not write project metadata {}: {exc}", meta_path.display());
    }
}

/// Update `last_touch` for a project, preserving `created_at`.
pub fn touch_project(store: &Path, working_dir: &str) {
    let dir_hash = project_hash(working_dir);
    let meta_path = project_meta_path(store, &dir_hash);
    if !meta_path.exists() {
        register_project(store, working_dir);
        return;
    }
    let mut meta: Map<String, Value> = fs::read_to_string(&meta_path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    let now = now_secs();
    meta.insert(
        "workdir".to_string(),
        Value::String(normalize_path(working_dir).to_string_lossy().into_owned()),
    );
    meta.insert("last_touch".to_string(), json!(now));
    meta.entry("created_at".to_string()).or_insert(json!(now));
    if let Err(exc) = fs::write(
        &meta_path,
        serde_json::to_string(&Value::Object(meta)).unwrap_or_default(),
    ) {
        log::debug!("Could not update project metadata {}: {exc}", meta_path.display());
    }
}

/// Return all registered projects under the store. Each entry carries a
/// `_hash` field with the project hash (its filename stem).
pub fn list_projects(store: &Path) -> Vec<Map<String, Value>> {
    let projects_dir = store.join(PROJECTS_DIRNAME);
    if !projects_dir.exists() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let entries = match fs::read_dir(&projects_dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let dir_hash = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let meta = match fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            Some(Value::Object(m)) => m,
            _ => continue,
        };
        let mut meta = meta;
        meta.insert("_hash".to_string(), Value::String(dir_hash));
        out.push(meta);
    }
    out
}

/// Backwards-compatible initialiser. Initialises the store, registers the
/// project, and writes a `HERMES_WORKDIR` compat marker.
pub fn init_shadow_repo(shadow_repo: &Path, working_dir: &str) -> Option<String> {
    if let Some(err) = init_store(shadow_repo, working_dir) {
        return Some(err);
    }
    register_project(shadow_repo, working_dir);
    let marker = shadow_repo.join("HERMES_WORKDIR");
    let _ = fs::write(
        &marker,
        format!("{}\n", normalize_path(working_dir).to_string_lossy()),
    );
    None
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages automatic filesystem checkpoints.
///
/// Call [`CheckpointManager::new_turn`] at the start of each conversation turn
/// and [`CheckpointManager::ensure_checkpoint`] before any file-mutating tool
/// call. The manager deduplicates so at most one snapshot is taken per
/// directory per turn.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    pub enabled: bool,
    pub max_snapshots: usize,
    pub max_total_size_mb: u64,
    pub max_file_size_mb: u64,
    /// Checkpoint base directory (e.g. `~/.hermes/checkpoints`).
    pub checkpoint_base: PathBuf,
    checkpointed_dirs: HashSet<String>,
    git_available: Option<bool>,
}

impl CheckpointManager {
    /// Construct a manager. `checkpoint_base` is the directory that holds the
    /// shared `store/` (in Python this is the module-level `CHECKPOINT_BASE`).
    pub fn new(
        checkpoint_base: PathBuf,
        enabled: bool,
        max_snapshots: i64,
        max_total_size_mb: i64,
        max_file_size_mb: i64,
    ) -> Self {
        CheckpointManager {
            enabled,
            max_snapshots: max_snapshots.max(1) as usize,
            max_total_size_mb: max_total_size_mb.max(0) as u64,
            max_file_size_mb: max_file_size_mb.max(0) as u64,
            checkpoint_base,
            checkpointed_dirs: HashSet::new(),
            git_available: None,
        }
    }

    fn store(&self) -> PathBuf {
        store_path(&self.checkpoint_base)
    }

    // ------------------------------------------------------------------
    // Turn lifecycle
    // ------------------------------------------------------------------

    /// Reset per-turn dedup. Call at the start of each agent iteration.
    pub fn new_turn(&mut self) {
        self.checkpointed_dirs.clear();
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Take a checkpoint if enabled and not already done this turn.
    ///
    /// Returns `true` if a checkpoint was taken, `false` otherwise. Never
    /// panics — all errors are silently logged.
    pub fn ensure_checkpoint(&mut self, working_dir: &str, reason: &str) -> bool {
        if !self.enabled {
            return false;
        }

        if self.git_available.is_none() {
            let available = which_git();
            self.git_available = Some(available);
            if !available {
                log::debug!("Checkpoints disabled: git not found");
            }
        }
        if self.git_available != Some(true) {
            return false;
        }

        let abs_dir = normalize_path(working_dir).to_string_lossy().into_owned();

        // Skip root, home, and other overly broad directories.
        let home = dirs::home_dir().map(|h| h.to_string_lossy().into_owned());
        if abs_dir == "/" || home.as_deref() == Some(abs_dir.as_str()) {
            log::debug!("Checkpoint skipped: directory too broad ({abs_dir})");
            return false;
        }

        if self.checkpointed_dirs.contains(&abs_dir) {
            return false;
        }
        self.checkpointed_dirs.insert(abs_dir.clone());

        self.take(&abs_dir, reason)
    }

    /// List available checkpoints for a directory (most recent first).
    pub fn list_checkpoints(&self, working_dir: &str) -> Vec<Map<String, Value>> {
        let abs_dir = normalize_path(working_dir).to_string_lossy().into_owned();
        let store = self.store();

        if !store.join("HEAD").exists() {
            return Vec::new();
        }

        let ref_n = ref_name(&project_hash(&abs_dir));
        let allowed: HashSet<i32> = [128, 129].into_iter().collect();
        let (ok, stdout, _) = run_git(
            &[
                "log",
                &ref_n,
                "--format=%H|%h|%aI|%s",
                "-n",
                &self.max_snapshots.to_string(),
            ],
            &store,
            &abs_dir,
            git_timeout_secs(),
            &allowed,
            None,
        );

        if !ok || stdout.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::new();
        for line in stdout.lines() {
            let parts: Vec<&str> = line.splitn(4, '|').collect();
            if parts.len() != 4 {
                continue;
            }
            let mut entry = Map::new();
            entry.insert("hash".to_string(), Value::String(parts[0].to_string()));
            entry.insert("short_hash".to_string(), Value::String(parts[1].to_string()));
            entry.insert("timestamp".to_string(), Value::String(parts[2].to_string()));
            entry.insert("reason".to_string(), Value::String(parts[3].to_string()));
            entry.insert("files_changed".to_string(), json!(0));
            entry.insert("insertions".to_string(), json!(0));
            entry.insert("deletions".to_string(), json!(0));

            let parent_spec = format!("{}~1", parts[0]);
            let (stat_ok, stat_out, _) = run_git(
                &["diff", "--shortstat", &parent_spec, parts[0]],
                &store,
                &abs_dir,
                git_timeout_secs(),
                &allowed,
                None,
            );
            if stat_ok && !stat_out.is_empty() {
                Self::parse_shortstat(&stat_out, &mut entry);
            }
            results.push(entry);
        }
        results
    }

    /// Parse git `--shortstat` output into an entry map.
    pub fn parse_shortstat(stat_line: &str, entry: &mut Map<String, Value>) {
        use std::sync::OnceLock;
        static FILE_RE: OnceLock<Regex> = OnceLock::new();
        static INS_RE: OnceLock<Regex> = OnceLock::new();
        static DEL_RE: OnceLock<Regex> = OnceLock::new();
        let file_re = FILE_RE.get_or_init(|| Regex::new(r"(\d+) file").unwrap());
        let ins_re = INS_RE.get_or_init(|| Regex::new(r"(\d+) insertion").unwrap());
        let del_re = DEL_RE.get_or_init(|| Regex::new(r"(\d+) deletion").unwrap());

        if let Some(c) = file_re.captures(stat_line) {
            if let Ok(n) = c[1].parse::<i64>() {
                entry.insert("files_changed".to_string(), json!(n));
            }
        }
        if let Some(c) = ins_re.captures(stat_line) {
            if let Ok(n) = c[1].parse::<i64>() {
                entry.insert("insertions".to_string(), json!(n));
            }
        }
        if let Some(c) = del_re.captures(stat_line) {
            if let Ok(n) = c[1].parse::<i64>() {
                entry.insert("deletions".to_string(), json!(n));
            }
        }
    }

    /// Show diff between a checkpoint and the current working tree.
    pub fn diff(&self, working_dir: &str, commit_hash: &str) -> Value {
        if let Some(hash_err) = validate_commit_hash(commit_hash) {
            return json!({"success": false, "error": hash_err});
        }

        let abs_dir = normalize_path(working_dir).to_string_lossy().into_owned();
        let store = self.store();

        if !store.join("HEAD").exists() {
            return json!({"success": false, "error": "No checkpoints exist for this directory"});
        }

        let (ok, _, _) = git(&["cat-file", "-t", commit_hash], &store, &abs_dir);
        if !ok {
            return json!({"success": false, "error": format!("Checkpoint '{commit_hash}' not found")});
        }

        let dir_hash = project_hash(&abs_dir);
        let index_file = index_path(&store, &dir_hash);

        // Stage current state into the per-project index to compare.
        run_git(
            &["add", "-A"],
            &store,
            &abs_dir,
            git_timeout_secs() * 2,
            &HashSet::new(),
            Some(&index_file),
        );

        let (ok_stat, stat_out, _) = run_git(
            &["diff", "--stat", commit_hash, "--cached"],
            &store,
            &abs_dir,
            git_timeout_secs(),
            &HashSet::new(),
            Some(&index_file),
        );
        let (ok_diff, diff_out, _) = run_git(
            &["diff", commit_hash, "--cached", "--no-color"],
            &store,
            &abs_dir,
            git_timeout_secs(),
            &HashSet::new(),
            Some(&index_file),
        );

        // Reset staged tree back to the project's last checkpoint.
        let ref_n = ref_name(&dir_hash);
        run_git(
            &["read-tree", &ref_n],
            &store,
            &abs_dir,
            git_timeout_secs(),
            &[128].into_iter().collect(),
            Some(&index_file),
        );

        if !ok_stat && !ok_diff {
            return json!({"success": false, "error": "Could not generate diff"});
        }

        json!({
            "success": true,
            "stat": if ok_stat { stat_out } else { String::new() },
            "diff": if ok_diff { diff_out } else { String::new() },
        })
    }

    /// Restore files to a checkpoint state.
    pub fn restore(&self, working_dir: &str, commit_hash: &str, file_path: Option<&str>) -> Value {
        if let Some(hash_err) = validate_commit_hash(commit_hash) {
            return json!({"success": false, "error": hash_err});
        }

        let abs_dir = normalize_path(working_dir).to_string_lossy().into_owned();

        if let Some(fp) = file_path {
            if let Some(path_err) = validate_file_path(fp, &abs_dir) {
                return json!({"success": false, "error": path_err});
            }
        }

        let store = self.store();

        if !store.join("HEAD").exists() {
            return json!({"success": false, "error": "No checkpoints exist for this directory"});
        }

        let (ok, _, err) = git(&["cat-file", "-t", commit_hash], &store, &abs_dir);
        if !ok {
            return json!({
                "success": false,
                "error": format!("Checkpoint '{commit_hash}' not found"),
                "debug": if err.is_empty() { Value::Null } else { Value::String(err) },
            });
        }

        // Take a pre-rollback snapshot so you can undo the undo.
        let short = &commit_hash[..commit_hash.len().min(8)];
        self.take(&abs_dir, &format!("pre-rollback snapshot (restoring to {short})"));

        let dir_hash = project_hash(&abs_dir);
        let index_file = index_path(&store, &dir_hash);

        let restore_target = file_path.unwrap_or(".");
        let (ok, _stdout, err) = run_git(
            &["checkout", commit_hash, "--", restore_target],
            &store,
            &abs_dir,
            git_timeout_secs() * 2,
            &HashSet::new(),
            Some(&index_file),
        );

        if !ok {
            return json!({
                "success": false,
                "error": format!("Restore failed: {err}"),
                "debug": if err.is_empty() { Value::Null } else { Value::String(err) },
            });
        }

        let (ok2, reason_out, _) = git(&["log", "--format=%s", "-1", commit_hash], &store, &abs_dir);
        let reason = if ok2 { reason_out } else { "unknown".to_string() };

        let mut result = Map::new();
        result.insert("success".to_string(), json!(true));
        result.insert("restored_to".to_string(), Value::String(short.to_string()));
        result.insert("reason".to_string(), Value::String(reason));
        result.insert("directory".to_string(), Value::String(abs_dir));
        if let Some(fp) = file_path {
            result.insert("file".to_string(), Value::String(fp.to_string()));
        }
        Value::Object(result)
    }

    /// Resolve a file path to its working directory for checkpointing.
    pub fn get_working_dir_for_path(&self, file_path: &str) -> String {
        let path = normalize_path(file_path);
        let candidate = if path.is_dir() {
            path.clone()
        } else {
            path.parent().map(|p| p.to_path_buf()).unwrap_or(path.clone())
        };

        let markers = [
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "Makefile",
            "pom.xml",
            ".hg",
            "Gemfile",
        ];
        let mut check = candidate.clone();
        loop {
            let parent = match check.parent() {
                Some(p) => p.to_path_buf(),
                None => break,
            };
            if parent == check {
                break;
            }
            if markers.iter().any(|m| check.join(m).exists()) {
                return check.to_string_lossy().into_owned();
            }
            check = parent;
        }

        candidate.to_string_lossy().into_owned()
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    /// Take a snapshot. Returns `true` on success.
    pub fn take(&self, working_dir: &str, reason: &str) -> bool {
        let store = self.store();

        if let Some(err) = init_store(&store, working_dir) {
            log::debug!("Checkpoint store init failed: {err}");
            return false;
        }

        touch_project(&store, working_dir);

        if dir_file_count(&normalize_path(working_dir)) > MAX_FILES {
            log::debug!("Checkpoint skipped: >{MAX_FILES} files in {working_dir}");
            return false;
        }

        let dir_hash = project_hash(working_dir);
        let index_file = index_path(&store, &dir_hash);
        let ref_n = ref_name(&dir_hash);

        let allowed128: HashSet<i32> = [128].into_iter().collect();

        // Seed the per-project index from the last checkpoint, if any.
        if index_file.exists() {
            let (ok_ref, ref_commit, _) = run_git(
                &["rev-parse", "--verify", &format!("{ref_n}^{{commit}}")],
                &store,
                working_dir,
                git_timeout_secs(),
                &allowed128,
                None,
            );
            if ok_ref && !ref_commit.is_empty() {
                run_git(
                    &["read-tree", &ref_commit],
                    &store,
                    working_dir,
                    git_timeout_secs(),
                    &allowed128,
                    Some(&index_file),
                );
            } else {
                let _ = fs::remove_file(&index_file);
            }
        } else if let Some(parent) = index_file.parent() {
            let _ = fs::create_dir_all(parent);
        }

        // Stage with per-project index.
        let (ok, _, err) = run_git(
            &["add", "-A"],
            &store,
            working_dir,
            git_timeout_secs() * 2,
            &HashSet::new(),
            Some(&index_file),
        );
        if !ok {
            log::debug!("Checkpoint git-add failed: {err}");
            return false;
        }

        if self.max_file_size_mb > 0 {
            self.drop_oversize_from_index(&store, working_dir, &index_file);
        }

        // Compare against the current ref tip.
        let (ok_ref, ref_commit, _) = run_git(
            &["rev-parse", "--verify", &format!("{ref_n}^{{commit}}")],
            &store,
            working_dir,
            git_timeout_secs(),
            &allowed128,
            None,
        );
        let has_ref = ok_ref && !ref_commit.is_empty();

        if has_ref {
            let (ok_diff, _, _) = run_git(
                &["diff-index", "--cached", "--quiet", &ref_commit],
                &store,
                working_dir,
                git_timeout_secs(),
                &[1].into_iter().collect(),
                Some(&index_file),
            );
            if ok_diff {
                log::debug!("Checkpoint skipped: no changes in {working_dir}");
                return false;
            }
        } else {
            let (ok_ls, ls_out, _) = run_git(
                &["ls-files", "--cached"],
                &store,
                working_dir,
                git_timeout_secs(),
                &HashSet::new(),
                Some(&index_file),
            );
            if ok_ls && ls_out.trim().is_empty() {
                log::debug!("Checkpoint skipped: empty tree in {working_dir}");
                return false;
            }
        }

        // Write tree from per-project index.
        let (ok_tree, tree_sha, err) = run_git(
            &["write-tree"],
            &store,
            working_dir,
            git_timeout_secs(),
            &HashSet::new(),
            Some(&index_file),
        );
        if !ok_tree || tree_sha.is_empty() {
            log::debug!("Checkpoint write-tree failed: {err}");
            return false;
        }

        // Build commit (parent = current ref tip, if any).
        let (ok_commit, new_sha, err) = if has_ref {
            run_git(
                &["commit-tree", &tree_sha, "-p", &ref_commit, "-m", reason, "--no-gpg-sign"],
                &store,
                working_dir,
                git_timeout_secs(),
                &HashSet::new(),
                Some(&index_file),
            )
        } else {
            run_git(
                &["commit-tree", &tree_sha, "-m", reason, "--no-gpg-sign"],
                &store,
                working_dir,
                git_timeout_secs(),
                &HashSet::new(),
                Some(&index_file),
            )
        };
        if !ok_commit || new_sha.is_empty() {
            log::debug!("Checkpoint commit-tree failed: {err}");
            return false;
        }

        // Update the per-project ref.
        let (ok_update, _, err) = if has_ref {
            git(&["update-ref", &ref_n, &new_sha, &ref_commit], &store, working_dir)
        } else {
            git(&["update-ref", &ref_n, &new_sha], &store, working_dir)
        };
        if !ok_update {
            log::debug!("Checkpoint update-ref failed: {err}");
            return false;
        }

        let short = &new_sha[..new_sha.len().min(8)];
        log::debug!("Checkpoint taken in {working_dir}: {reason} ({short})");

        // Real pruning — drop old commits beyond max_snapshots.
        self.prune(&store, working_dir, &ref_n);

        // Enforce global size cap.
        self.enforce_size_cap(&store);

        true
    }

    /// Remove any staged file larger than `max_file_size_mb` from the index.
    fn drop_oversize_from_index(&self, store: &Path, working_dir: &str, index_file: &Path) {
        let cap = self.max_file_size_mb * 1024 * 1024;
        if cap == 0 {
            return;
        }
        let (ok, stdout, _) = run_git(
            &["ls-files", "--cached", "-z"],
            store,
            working_dir,
            git_timeout_secs(),
            &HashSet::new(),
            Some(index_file),
        );
        if !ok || stdout.is_empty() {
            return;
        }
        let paths: Vec<&str> = stdout.split('\u{0}').filter(|p| !p.is_empty()).collect();
        let abs_workdir = normalize_path(working_dir);
        let mut oversize: Vec<String> = Vec::new();
        for rel in paths {
            match fs::metadata(abs_workdir.join(rel)) {
                Ok(meta) => {
                    if meta.len() > cap {
                        oversize.push(rel.to_string());
                    }
                }
                Err(_) => continue,
            }
        }
        if oversize.is_empty() {
            return;
        }
        log::debug!(
            "Checkpoint: dropping {} oversize file(s) (>{} MB) from index",
            oversize.len(),
            self.max_file_size_mb
        );
        const BATCH: usize = 200;
        let allowed128: HashSet<i32> = [128].into_iter().collect();
        for chunk in oversize.chunks(BATCH) {
            let mut args: Vec<String> =
                vec!["rm".into(), "--cached".into(), "--quiet".into(), "--".into()];
            args.extend(chunk.iter().cloned());
            run_git(
                &args,
                store,
                working_dir,
                git_timeout_secs(),
                &allowed128,
                Some(index_file),
            );
        }
    }

    /// Keep only the last `max_snapshots` commits on the per-project ref.
    fn prune(&self, store: &Path, working_dir: &str, ref_n: &str) {
        let allowed128: HashSet<i32> = [128].into_iter().collect();
        let (ok, stdout, _) = run_git(
            &["rev-list", "--count", ref_n],
            store,
            working_dir,
            git_timeout_secs(),
            &allowed128,
            None,
        );
        if !ok {
            return;
        }
        let count = match stdout.trim().parse::<usize>() {
            Ok(n) => n,
            Err(_) => return,
        };
        if count <= self.max_snapshots {
            return;
        }

        let (ok_list, list_out, _) = git(&["rev-list", "--reverse", ref_n], store, working_dir);
        if !ok_list || list_out.is_empty() {
            return;
        }
        let commits: Vec<&str> = list_out.lines().collect();
        let keep_from = commits.len().saturating_sub(self.max_snapshots);
        let keep = &commits[keep_from..];

        let mut new_parent: Option<String> = None;
        for sha in keep {
            let (ok_tree, tree_sha, _) =
                git(&["rev-parse", &format!("{sha}^{{tree}}")], store, working_dir);
            if !ok_tree || tree_sha.is_empty() {
                return;
            }
            let (ok_msg, msg, _) = git(&["log", "--format=%s", "-1", sha], store, working_dir);
            let commit_msg = if ok_msg && !msg.is_empty() { msg } else { "checkpoint".to_string() };
            let (ok_commit, new_sha, _) = if let Some(p) = &new_parent {
                git(
                    &["commit-tree", &tree_sha, "-p", p, "-m", &commit_msg, "--no-gpg-sign"],
                    store,
                    working_dir,
                )
            } else {
                git(
                    &["commit-tree", &tree_sha, "-m", &commit_msg, "--no-gpg-sign"],
                    store,
                    working_dir,
                )
            };
            if !ok_commit || new_sha.is_empty() {
                return;
            }
            new_parent = Some(new_sha);
        }

        let new_parent = match new_parent {
            Some(p) => p,
            None => return,
        };
        git(&["update-ref", ref_n, &new_parent], store, working_dir);

        git(&["reflog", "expire", "--expire=now", "--all"], store, working_dir);
        run_git(
            &["gc", "--prune=now", "--quiet"],
            store,
            working_dir,
            git_timeout_secs() * 3,
            &HashSet::new(),
            None,
        );
    }

    /// If total store size exceeds `max_total_size_mb`, drop oldest checkpoints
    /// across ALL projects until under the cap.
    fn enforce_size_cap(&self, store: &Path) {
        if self.max_total_size_mb == 0 {
            return;
        }
        let cap_bytes = self.max_total_size_mb * 1024 * 1024;
        let mut size = dir_size_bytes(store);
        if size <= cap_bytes {
            return;
        }
        log::info!(
            "Checkpoint store exceeded {} MB (actual {} MB) — pruning oldest",
            self.max_total_size_mb,
            size / (1024 * 1024)
        );

        let base = store.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let base_str = base.to_string_lossy().into_owned();
        let allowed128: HashSet<i32> = [128].into_iter().collect();

        let (ok, stdout, _) = run_git(
            &["for-each-ref", "--format=%(refname)", REFS_PREFIX],
            store,
            &base_str,
            git_timeout_secs(),
            &allowed128,
            None,
        );
        if !ok || stdout.is_empty() {
            return;
        }
        let refs: Vec<String> = stdout
            .lines()
            .filter(|r| !r.trim().is_empty())
            .map(|r| r.to_string())
            .collect();

        for _ in 0..20 {
            size = dir_size_bytes(store);
            if size <= cap_bytes {
                break;
            }
            let mut any_dropped = false;
            for ref_n in &refs {
                let dropped = drop_oldest_commit(store, &base_str, ref_n, &allowed128);
                if dropped {
                    any_dropped = true;
                }
            }
            if !any_dropped {
                break;
            }
        }

        git(&["reflog", "expire", "--expire=now", "--all"], store, &base_str);
        run_git(
            &["gc", "--prune=now", "--quiet"],
            store,
            &base_str,
            git_timeout_secs() * 3,
            &HashSet::new(),
            None,
        );
    }
}

/// Drop the oldest commit on a ref, rebuilding a linear chain. Returns `true`
/// if the ref was rewritten.
fn drop_oldest_commit(
    store: &Path,
    working_dir: &str,
    ref_n: &str,
    allowed128: &HashSet<i32>,
) -> bool {
    let (ok_count, count_out, _) = run_git(
        &["rev-list", "--count", ref_n],
        store,
        working_dir,
        git_timeout_secs(),
        allowed128,
        None,
    );
    let count = if ok_count {
        count_out.trim().parse::<usize>().unwrap_or(0)
    } else {
        0
    };
    if count <= 1 {
        return false;
    }
    let (ok_list, list_out, _) = git(&["rev-list", "--reverse", ref_n], store, working_dir);
    if !ok_list || list_out.is_empty() {
        return false;
    }
    let commits: Vec<&str> = list_out.lines().collect();
    let keep = &commits[1..]; // drop oldest

    let mut new_parent: Option<String> = None;
    for sha in keep {
        let (ok_tree, tree_sha, _) =
            git(&["rev-parse", &format!("{sha}^{{tree}}")], store, working_dir);
        if !ok_tree || tree_sha.is_empty() {
            return false;
        }
        let (ok_msg, msg, _) = git(&["log", "--format=%s", "-1", sha], store, working_dir);
        let commit_msg = if ok_msg && !msg.is_empty() { msg } else { "checkpoint".to_string() };
        let (ok_cm, new_sha, _) = if let Some(p) = &new_parent {
            git(
                &["commit-tree", &tree_sha, "-p", p, "-m", &commit_msg, "--no-gpg-sign"],
                store,
                working_dir,
            )
        } else {
            git(
                &["commit-tree", &tree_sha, "-m", &commit_msg, "--no-gpg-sign"],
                store,
                working_dir,
            )
        };
        if !ok_cm || new_sha.is_empty() {
            return false;
        }
        new_parent = Some(new_sha);
    }
    match new_parent {
        Some(p) => {
            git(&["update-ref", ref_n, &p], store, working_dir);
            true
        }
        None => false,
    }
}

fn which_git() -> bool {
    // Mirror shutil.which("git"): check PATH for an executable named git.
    if let Ok(path) = std::env::var("PATH") {
        let exe_names: &[&str] = if cfg!(windows) {
            &["git.exe", "git.cmd", "git"]
        } else {
            &["git"]
        };
        for dir in std::env::split_paths(&path) {
            for name in exe_names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return true;
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Display formatting
// ---------------------------------------------------------------------------

/// Format checkpoint list for display to user.
pub fn format_checkpoint_list(checkpoints: &[Map<String, Value>], directory: &str) -> String {
    if checkpoints.is_empty() {
        return format!("No checkpoints found for {directory}");
    }

    let mut lines: Vec<String> = vec![format!("\u{1F4F8} Checkpoints for {directory}:\n")];
    for (i, cp) in checkpoints.iter().enumerate() {
        let idx = i + 1;
        let raw_ts = cp.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        let ts = if raw_ts.contains('T') {
            // ts.split("T")[1].split("+")[0].split("-")[0][:5]
            let after_t = raw_ts.splitn(2, 'T').nth(1).unwrap_or("");
            let no_plus = after_t.splitn(2, '+').next().unwrap_or("");
            let no_minus = no_plus.splitn(2, '-').next().unwrap_or("");
            let clock: String = no_minus.chars().take(5).collect();
            let date = raw_ts.splitn(2, 'T').next().unwrap_or("");
            format!("{date} {clock}")
        } else {
            raw_ts.to_string()
        };

        let files = cp.get("files_changed").and_then(|v| v.as_i64()).unwrap_or(0);
        let ins = cp.get("insertions").and_then(|v| v.as_i64()).unwrap_or(0);
        let dele = cp.get("deletions").and_then(|v| v.as_i64()).unwrap_or(0);
        let stat = if files != 0 {
            let plural = if files != 1 { "s" } else { "" };
            format!("  ({files} file{plural}, +{ins}/-{dele})")
        } else {
            String::new()
        };

        let short_hash = cp.get("short_hash").and_then(|v| v.as_str()).unwrap_or("");
        let reason = cp.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        lines.push(format!("  {idx}. {short_hash}  {ts}  {reason}{stat}"));
    }

    lines.push("\n  /rollback <N>             restore to checkpoint N".to_string());
    lines.push("  /rollback diff <N>        preview changes since checkpoint N".to_string());
    lines.push("  /rollback <N> <file>      restore a single file from checkpoint N".to_string());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Auto-maintenance
// ---------------------------------------------------------------------------

/// Counts returned by [`prune_checkpoints`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneResult {
    pub scanned: u64,
    pub deleted_orphan: u64,
    pub deleted_stale: u64,
    pub errors: u64,
    pub bytes_freed: u64,
}

impl PruneResult {
    pub fn to_json(&self) -> Value {
        json!({
            "scanned": self.scanned,
            "deleted_orphan": self.deleted_orphan,
            "deleted_stale": self.deleted_stale,
            "errors": self.errors,
            "bytes_freed": self.bytes_freed,
        })
    }
}

fn delete_ref(store: &Path, ref_n: &str) -> bool {
    let base = store.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let (ok, _, _) = run_git(
        &["update-ref", "-d", ref_n],
        store,
        &base.to_string_lossy(),
        git_timeout_secs(),
        &[128].into_iter().collect(),
        None,
    );
    ok
}

fn path_mtime(path: &Path) -> Option<f64> {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
}

/// Newest mtime found anywhere under `path` (recursive). Returns 0.0 if none.
fn newest_mtime(path: &Path) -> f64 {
    let mut newest = 0.0f64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(mt) = path_mtime(&p) {
                if mt > newest {
                    newest = mt;
                }
            }
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            }
        }
    }
    newest
}

/// Delete stale/orphan checkpoints and reclaim store space. Never panics.
pub fn prune_checkpoints(
    retention_days: i64,
    delete_orphans: bool,
    checkpoint_base: &Path,
    max_total_size_mb: u64,
) -> PruneResult {
    let base = checkpoint_base;
    let mut result = PruneResult::default();
    if !base.exists() {
        return result;
    }

    let size_before = dir_size_bytes(base);

    let cutoff = if retention_days > 0 {
        now_secs() - (retention_days as f64) * 86400.0
    } else {
        0.0
    };

    // --- Legacy pre-v2 per-project shadow repos ---
    let children: Vec<PathBuf> = fs::read_dir(base)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    for child in &children {
        if !child.is_dir() {
            continue;
        }
        let name = child.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == STORE_DIRNAME {
            continue;
        }
        if name.starts_with(LEGACY_PREFIX) {
            if retention_days <= 0 {
                continue;
            }
            let m = match path_mtime(child) {
                Some(m) => m,
                None => continue,
            };
            if m >= cutoff {
                continue;
            }
            let size = dir_size_bytes(child);
            match fs::remove_dir_all(child) {
                Ok(()) => {
                    result.bytes_freed += size;
                    result.deleted_stale += 1;
                }
                Err(exc) => {
                    result.errors += 1;
                    log::warn!("Failed to delete legacy archive {}: {exc}", child.display());
                }
            }
            continue;
        }
        if !child.join("HEAD").exists() {
            continue;
        }
        result.scanned += 1;
        let mut reason: Option<&str> = None;
        if delete_orphans {
            let mut workdir: Option<String> = None;
            let wd_marker = child.join("HERMES_WORKDIR");
            if wd_marker.exists() {
                if let Ok(text) = fs::read_to_string(&wd_marker) {
                    workdir = Some(text.trim().to_string());
                }
            }
            let exists = workdir
                .as_ref()
                .map(|w| Path::new(w).exists())
                .unwrap_or(false);
            if workdir.is_none() || !exists {
                reason = Some("orphan");
            }
        }
        if reason.is_none() && retention_days > 0 {
            let newest = newest_mtime(child);
            if newest > 0.0 && newest < cutoff {
                reason = Some("stale");
            }
        }
        let reason = match reason {
            Some(r) => r,
            None => continue,
        };
        let size = dir_size_bytes(child);
        match fs::remove_dir_all(child) {
            Ok(()) => {
                result.bytes_freed += size;
                if reason == "orphan" {
                    result.deleted_orphan += 1;
                } else {
                    result.deleted_stale += 1;
                }
            }
            Err(exc) => {
                result.errors += 1;
                log::warn!("Failed to prune checkpoint repo {name}: {exc}");
            }
        }
    }

    // --- v2 shared store: per-project ref pruning via metadata ---
    let store = store_path(base);
    if store.join("HEAD").exists() {
        let base_str = base.to_string_lossy().into_owned();
        for meta in list_projects(&store) {
            let dir_hash = meta.get("_hash").and_then(|v| v.as_str()).unwrap_or("");
            let workdir = meta.get("workdir").and_then(|v| v.as_str()).unwrap_or("");
            if dir_hash.is_empty() {
                continue;
            }
            result.scanned += 1;
            let mut reason: Option<&str> = None;
            if delete_orphans && (workdir.is_empty() || !Path::new(workdir).exists()) {
                reason = Some("orphan");
            } else if retention_days > 0 {
                let last_touch = meta.get("last_touch").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if last_touch > 0.0 && last_touch < cutoff {
                    reason = Some("stale");
                }
            }
            let reason = match reason {
                Some(r) => r,
                None => continue,
            };
            let ref_n = ref_name(dir_hash);
            delete_ref(&store, &ref_n);
            let idx = index_path(&store, dir_hash);
            if idx.exists() {
                let _ = fs::remove_file(&idx);
            }
            let mp = project_meta_path(&store, dir_hash);
            if mp.exists() {
                let _ = fs::remove_file(&mp);
            }
            if reason == "orphan" {
                result.deleted_orphan += 1;
            } else {
                result.deleted_stale += 1;
            }
        }

        git(&["reflog", "expire", "--expire=now", "--all"], &store, &base_str);
        run_git(
            &["gc", "--prune=now", "--quiet"],
            &store,
            &base_str,
            git_timeout_secs() * 3,
            &HashSet::new(),
            None,
        );

        // Size-cap pass.
        if max_total_size_mb > 0 {
            let cap_bytes = max_total_size_mb * 1024 * 1024;
            let allowed128: HashSet<i32> = [128].into_iter().collect();
            for _ in 0..20 {
                let size = dir_size_bytes(&store);
                if size <= cap_bytes {
                    break;
                }
                let (ok, stdout, _) = run_git(
                    &["for-each-ref", "--format=%(refname)", REFS_PREFIX],
                    &store,
                    &base_str,
                    git_timeout_secs(),
                    &allowed128,
                    None,
                );
                let refs: Vec<String> = if ok {
                    stdout
                        .lines()
                        .filter(|r| !r.trim().is_empty())
                        .map(|r| r.to_string())
                        .collect()
                } else {
                    Vec::new()
                };
                if refs.is_empty() {
                    break;
                }
                let mut any_drop = false;
                for ref_n in &refs {
                    if drop_oldest_commit(&store, &base_str, ref_n, &allowed128) {
                        any_drop = true;
                    }
                }
                if !any_drop {
                    break;
                }
            }
            git(&["reflog", "expire", "--expire=now", "--all"], &store, &base_str);
            run_git(
                &["gc", "--prune=now", "--quiet"],
                &store,
                &base_str,
                git_timeout_secs() * 3,
                &HashSet::new(),
                None,
            );
        }
    }

    let size_after = dir_size_bytes(base);
    let delta = size_before.saturating_sub(size_after);
    if delta > result.bytes_freed {
        result.bytes_freed = delta;
    }

    result
}

/// Outcome of [`maybe_auto_prune_checkpoints`].
#[derive(Debug, Clone)]
pub struct AutoPruneOutcome {
    pub skipped: bool,
    pub result: Option<PruneResult>,
    pub error: Option<String>,
}

impl AutoPruneOutcome {
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("skipped".to_string(), json!(self.skipped));
        if let Some(r) = &self.result {
            m.insert("result".to_string(), r.to_json());
        }
        if let Some(e) = &self.error {
            m.insert("error".to_string(), Value::String(e.clone()));
        }
        Value::Object(m)
    }
}

/// Idempotent wrapper around [`prune_checkpoints`] for startup hooks.
///
/// Writes `<base>/.last_prune` on completion so subsequent calls within
/// `min_interval_hours` short-circuit.
pub fn maybe_auto_prune_checkpoints(
    retention_days: i64,
    min_interval_hours: i64,
    delete_orphans: bool,
    checkpoint_base: &Path,
    max_total_size_mb: u64,
) -> AutoPruneOutcome {
    let base = checkpoint_base;
    let mut out = AutoPruneOutcome {
        skipped: false,
        result: None,
        error: None,
    };

    if !base.exists() {
        out.result = Some(PruneResult::default());
        return out;
    }

    let marker = base.join(PRUNE_MARKER_NAME);
    let now = now_secs();
    if marker.exists() {
        if let Ok(text) = fs::read_to_string(&marker) {
            if let Ok(last_ts) = text.trim().parse::<f64>() {
                if now - last_ts < (min_interval_hours as f64) * 3600.0 {
                    out.skipped = true;
                    return out;
                }
            }
        }
    }

    let result = prune_checkpoints(retention_days, delete_orphans, base, max_total_size_mb);

    if let Err(exc) = fs::write(&marker, now.to_string()) {
        log::debug!("Could not write checkpoint prune marker: {exc}");
    }

    let total = result.deleted_orphan + result.deleted_stale;
    if total > 0 {
        log::info!(
            "checkpoint auto-maintenance: pruned {total} entry(ies) ({} orphan, {} stale), reclaimed {:.1} MB",
            result.deleted_orphan,
            result.deleted_stale,
            result.bytes_freed as f64 / (1024.0 * 1024.0)
        );
    }
    out.result = Some(result);
    out
}

// ---------------------------------------------------------------------------
// Public helpers for `hermes checkpoints` CLI
// ---------------------------------------------------------------------------

/// Return a summary of the shadow store.
pub fn store_status(checkpoint_base: &Path) -> Value {
    let base = checkpoint_base;
    let mut out = Map::new();
    out.insert("base".to_string(), Value::String(base.to_string_lossy().into_owned()));
    out.insert("store_size_bytes".to_string(), json!(0));
    out.insert("legacy_size_bytes".to_string(), json!(0));
    out.insert("total_size_bytes".to_string(), json!(0));
    out.insert("project_count".to_string(), json!(0));
    out.insert("projects".to_string(), json!([]));
    out.insert("legacy_archives".to_string(), json!([]));

    if !base.exists() {
        return Value::Object(out);
    }

    let store = store_path(base);
    let mut projects: Vec<Value> = Vec::new();
    if store.exists() {
        out.insert("store_size_bytes".to_string(), json!(dir_size_bytes(&store)));
        if store.join("HEAD").exists() {
            let base_str = base.to_string_lossy().into_owned();
            let allowed128: HashSet<i32> = [128].into_iter().collect();
            for meta in list_projects(&store) {
                let dir_hash = meta.get("_hash").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let workdir = meta.get("workdir").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let ref_n = ref_name(&dir_hash);
                let (ok, count_out, _) = run_git(
                    &["rev-list", "--count", &ref_n],
                    &store,
                    &base_str,
                    git_timeout_secs(),
                    &allowed128,
                    None,
                );
                let commits = if ok {
                    count_out.trim().parse::<i64>().unwrap_or(0)
                } else {
                    0
                };
                let exists = !workdir.is_empty() && Path::new(&workdir).exists();
                projects.push(json!({
                    "hash": dir_hash,
                    "workdir": workdir,
                    "exists": exists,
                    "created_at": meta.get("created_at").cloned().unwrap_or(Value::Null),
                    "last_touch": meta.get("last_touch").cloned().unwrap_or(Value::Null),
                    "commits": commits,
                }));
            }
        }
    }
    let project_count = projects.len();
    out.insert("projects".to_string(), Value::Array(projects));
    out.insert("project_count".to_string(), json!(project_count));

    let mut legacy_size: u64 = 0;
    let mut legacy_archives: Vec<Value> = Vec::new();
    if let Ok(rd) = fs::read_dir(base) {
        // Sorted for deterministic output (BTreeMap keyed by name).
        let mut by_name: BTreeMap<String, PathBuf> = BTreeMap::new();
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(LEGACY_PREFIX) {
                        by_name.insert(name.to_string(), p);
                    }
                }
            }
        }
        for (name, p) in by_name {
            let size = dir_size_bytes(&p);
            legacy_size += size;
            let mt = path_mtime(&p).unwrap_or(0.0);
            legacy_archives.push(json!({
                "name": name,
                "size_bytes": size,
                "mtime": mt,
            }));
        }
    }
    out.insert("legacy_size_bytes".to_string(), json!(legacy_size));
    out.insert("legacy_archives".to_string(), Value::Array(legacy_archives));

    out.insert("total_size_bytes".to_string(), json!(dir_size_bytes(base)));
    Value::Object(out)
}

/// Result of [`clear_all`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClearResult {
    pub bytes_freed: u64,
    pub deleted: u64,
}

/// Nuke the entire checkpoint base (store + legacy). Irreversible.
///
/// Returns `{bytes_freed, deleted}` where `deleted` is 0 or 1.
pub fn clear_all(checkpoint_base: &Path) -> ClearResult {
    let base = checkpoint_base;
    let mut out = ClearResult { bytes_freed: 0, deleted: 0 };
    if !base.exists() {
        return out;
    }
    let size = dir_size_bytes(base);
    match fs::remove_dir_all(base) {
        Ok(()) => {
            out.bytes_freed = size;
            out.deleted = 1;
        }
        Err(exc) => {
            log::warn!("Could not clear checkpoint base {}: {exc}", base.display());
        }
    }
    out
}

/// Delete all `legacy-*` archive directories.
///
/// Returns `{bytes_freed, deleted}` where `deleted` is a count.
pub fn clear_legacy(checkpoint_base: &Path) -> ClearResult {
    let base = checkpoint_base;
    let mut out = ClearResult { bytes_freed: 0, deleted: 0 };
    if !base.exists() {
        return out;
    }
    let children: Vec<PathBuf> = fs::read_dir(base)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    for child in children {
        let name = child.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !child.is_dir() || !name.starts_with(LEGACY_PREFIX) {
            continue;
        }
        let size = dir_size_bytes(&child);
        match fs::remove_dir_all(&child) {
            Ok(()) => {
                out.bytes_freed += size;
                out.deleted += 1;
            }
            Err(exc) => {
                log::warn!("Could not delete legacy archive {}: {exc}", child.display());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn git_present() -> bool {
        which_git()
    }

    #[test]
    fn validate_commit_hash_cases() {
        assert!(validate_commit_hash("").is_some());
        assert!(validate_commit_hash("   ").is_some());
        assert!(validate_commit_hash("-p").is_some());
        assert!(validate_commit_hash("--patch").is_some());
        assert!(validate_commit_hash("xyz").is_some()); // non-hex
        assert!(validate_commit_hash("abc").is_some()); // too short (3)
        assert!(validate_commit_hash("abcd").is_none()); // 4 hex ok
        assert!(validate_commit_hash("DEADBEEF").is_none());
        let forty = "a".repeat(40);
        assert!(validate_commit_hash(&forty).is_none());
        let sixtyfive = "a".repeat(65);
        assert!(validate_commit_hash(&sixtyfive).is_some());
    }

    #[test]
    fn validate_file_path_cases() {
        let wd = std::env::temp_dir();
        let wd = wd.to_string_lossy();
        assert!(validate_file_path("", &wd).is_some());
        assert!(validate_file_path("/etc/passwd", &wd).is_some()); // absolute
        assert!(validate_file_path("../../etc/passwd", &wd).is_some()); // traversal
        assert!(validate_file_path("src/main.rs", &wd).is_none());
        assert!(validate_file_path("a/b/c.txt", &wd).is_none());
    }

    #[test]
    fn project_hash_is_deterministic_16_hex() {
        let h1 = project_hash("/tmp/some/project");
        let h2 = project_hash("/tmp/some/project");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn ref_and_path_helpers() {
        let store = PathBuf::from("/base/store");
        assert_eq!(ref_name("deadbeef"), "refs/hermes/deadbeef");
        assert_eq!(
            index_path(&store, "deadbeef"),
            PathBuf::from("/base/store/indexes/deadbeef")
        );
        assert_eq!(
            project_meta_path(&store, "deadbeef"),
            PathBuf::from("/base/store/projects/deadbeef.json")
        );
        assert_eq!(store_path(Path::new("/base")), PathBuf::from("/base/store"));
    }

    #[test]
    fn git_timeout_clamped() {
        // Can't reliably set env in parallel tests; just check default range.
        let t = git_timeout_secs();
        assert!((10..=60).contains(&t));
    }

    #[test]
    fn parse_shortstat_extracts_numbers() {
        let mut entry = Map::new();
        CheckpointManager::parse_shortstat(
            " 3 files changed, 10 insertions(+), 4 deletions(-)",
            &mut entry,
        );
        assert_eq!(entry.get("files_changed").unwrap().as_i64(), Some(3));
        assert_eq!(entry.get("insertions").unwrap().as_i64(), Some(10));
        assert_eq!(entry.get("deletions").unwrap().as_i64(), Some(4));

        let mut entry2 = Map::new();
        CheckpointManager::parse_shortstat(" 1 file changed, 2 insertions(+)", &mut entry2);
        assert_eq!(entry2.get("files_changed").unwrap().as_i64(), Some(1));
        assert_eq!(entry2.get("insertions").unwrap().as_i64(), Some(2));
        // deletions not present in input -> stays whatever default (unset here)
        assert!(entry2.get("deletions").is_none());
    }

    #[test]
    fn dir_file_count_and_size() {
        let tmp = std::env::temp_dir().join(format!("ckpt_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("sub")).unwrap();
        fs::write(tmp.join("a.txt"), b"hello").unwrap();
        fs::write(tmp.join("sub/b.txt"), b"world!!").unwrap();

        // 2 files + 1 dir = 3 entries counted by rglob-style walk.
        assert_eq!(dir_file_count(&tmp), 3);
        assert_eq!(dir_size_bytes(&tmp), 5 + 7);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn format_checkpoint_list_empty_and_populated() {
        assert_eq!(
            format_checkpoint_list(&[], "/proj"),
            "No checkpoints found for /proj"
        );

        let mut cp = Map::new();
        cp.insert("short_hash".to_string(), json!("abc1234"));
        cp.insert("timestamp".to_string(), json!("2026-06-03T14:30:55+00:00"));
        cp.insert("reason".to_string(), json!("auto"));
        cp.insert("files_changed".to_string(), json!(2));
        cp.insert("insertions".to_string(), json!(5));
        cp.insert("deletions".to_string(), json!(1));
        let out = format_checkpoint_list(&[cp], "/proj");
        assert!(out.contains("Checkpoints for /proj"));
        assert!(out.contains("abc1234"));
        assert!(out.contains("2026-06-03 14:30"));
        assert!(out.contains("(2 files, +5/-1)"));
        assert!(out.contains("/rollback <N>"));
    }

    #[test]
    fn get_working_dir_for_path_finds_marker() {
        let tmp = std::env::temp_dir().join(format!("ckpt_wd_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("nested/deep")).unwrap();
        fs::write(tmp.join("Cargo.toml"), b"[package]").unwrap();
        fs::write(tmp.join("nested/deep/file.rs"), b"fn main() {}").unwrap();

        let mgr = CheckpointManager::new(tmp.join("ckpt"), false, 20, 500, 10);
        let file = tmp.join("nested/deep/file.rs");
        let wd = mgr.get_working_dir_for_path(&file.to_string_lossy());
        // Should walk up to the dir containing Cargo.toml.
        let canon = normalize_path(&tmp.to_string_lossy());
        assert_eq!(wd, canon.to_string_lossy());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn store_status_empty_base() {
        let tmp = std::env::temp_dir().join(format!("ckpt_status_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let status = store_status(&tmp);
        assert_eq!(status["project_count"].as_u64(), Some(0));
        assert_eq!(status["total_size_bytes"].as_u64(), Some(0));
        assert_eq!(status["projects"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn maybe_auto_prune_skips_within_interval() {
        let tmp = std::env::temp_dir().join(format!("ckpt_prune_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // Write a fresh marker -> should skip.
        fs::write(tmp.join(PRUNE_MARKER_NAME), now_secs().to_string()).unwrap();
        let out = maybe_auto_prune_checkpoints(7, 24, true, &tmp, 0);
        assert!(out.skipped);

        // Old marker -> should run.
        let old = now_secs() - 1_000_000.0;
        fs::write(tmp.join(PRUNE_MARKER_NAME), old.to_string()).unwrap();
        let out2 = maybe_auto_prune_checkpoints(7, 24, true, &tmp, 0);
        assert!(!out2.skipped);
        assert!(out2.result.is_some());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clear_legacy_removes_only_legacy_dirs() {
        let tmp = std::env::temp_dir().join(format!("ckpt_legacy_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("legacy-20200101-000000")).unwrap();
        fs::write(tmp.join("legacy-20200101-000000/x"), b"data").unwrap();
        fs::create_dir_all(tmp.join("store")).unwrap();
        fs::write(tmp.join("store/y"), b"keep").unwrap();

        let res = clear_legacy(&tmp);
        assert_eq!(res.deleted, 1);
        assert!(res.bytes_freed >= 4);
        assert!(!tmp.join("legacy-20200101-000000").exists());
        assert!(tmp.join("store").exists());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn end_to_end_checkpoint_and_restore() {
        if !git_present() {
            eprintln!("skipping: git not available");
            return;
        }
        let root = std::env::temp_dir().join(format!("ckpt_e2e_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let project = root.join("proj");
        let base = root.join("checkpoints");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("file.txt"), b"version one\n").unwrap();

        let mut mgr = CheckpointManager::new(base.clone(), true, 20, 500, 10);
        mgr.new_turn();
        let pdir = project.to_string_lossy().into_owned();
        let took = mgr.ensure_checkpoint(&pdir, "first snapshot");
        assert!(took, "first checkpoint should be taken");

        // Dedup within the same turn.
        assert!(!mgr.ensure_checkpoint(&pdir, "dup"));

        // New turn, modify file, snapshot again.
        mgr.new_turn();
        fs::write(project.join("file.txt"), b"version two\n").unwrap();
        let took2 = mgr.ensure_checkpoint(&pdir, "second snapshot");
        assert!(took2, "second checkpoint should be taken");

        let checkpoints = mgr.list_checkpoints(&pdir);
        assert_eq!(checkpoints.len(), 2);
        // Most recent first.
        assert_eq!(
            checkpoints[0].get("reason").unwrap().as_str(),
            Some("second snapshot")
        );
        let oldest_hash = checkpoints[1].get("hash").unwrap().as_str().unwrap().to_string();

        // Diff current tree vs oldest checkpoint should mention file.txt.
        let d = mgr.diff(&pdir, &oldest_hash);
        assert_eq!(d["success"], json!(true));

        // Restore to the oldest checkpoint.
        let r = mgr.restore(&pdir, &oldest_hash, None);
        assert_eq!(r["success"], json!(true), "restore failed: {r:?}");
        let content = fs::read_to_string(project.join("file.txt")).unwrap();
        assert_eq!(content, "version one\n");

        // store_status reflects one project.
        let status = store_status(&base);
        assert!(status["project_count"].as_u64().unwrap() >= 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_commit_hash_errors() {
        if !git_present() {
            return;
        }
        let root = std::env::temp_dir().join(format!("ckpt_unknown_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let project = root.join("proj");
        let base = root.join("checkpoints");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("a.txt"), b"x").unwrap();

        let mut mgr = CheckpointManager::new(base, true, 20, 500, 10);
        let pdir = project.to_string_lossy().into_owned();
        mgr.ensure_checkpoint(&pdir, "init");

        // Valid-shaped but non-existent hash.
        let r = mgr.restore(&pdir, "abcdef12", None);
        assert_eq!(r["success"], json!(false));
        let d = mgr.diff(&pdir, "abcdef12");
        assert_eq!(d["success"], json!(false));

        let _ = fs::remove_dir_all(&root);
    }
}
