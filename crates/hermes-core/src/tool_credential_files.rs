//! File passthrough registry for remote terminal backends.
//!
//! Remote backends (Docker, Modal, SSH) create sandboxes with no host files.
//! This module ensures that credential files, skill directories, and host-side
//! cache directories (documents, images, audio, screenshots) are mounted or
//! synced into those sandboxes so the agent can access them.
//!
//! **Credentials and skills** — session-scoped registry fed by skill
//! declarations (`required_credential_files`) and user config
//! (`terminal.credential_files`).
//!
//! **Cache directories** — gateway-cached uploads, browser screenshots, TTS
//! audio, and processed images. Mounted read-only so the remote terminal can
//! reference files the host side created (e.g. `unzip` an uploaded archive).
//!
//! This is a faithful native Rust port of `tools/credential_files.py`.
//!
//! ## Session scoping
//!
//! The Python original backs the skill-registered credential map with a
//! `ContextVar`, which is per-async-context / per-session. Rust does not have
//! a direct equivalent of `ContextVar` here, so we use a thread-local map. The
//! gateway pipeline runs each session on its own task/thread boundary, so a
//! thread-local prevents cross-session bleed the same way the original
//! `ContextVar` did. The behaviour-visible semantics (get-or-create empty map,
//! `clear()` reset) are preserved.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

use crate::mod_hermes_constants::{get_config_path, get_hermes_dir, get_hermes_home};
use crate::tool_path_security::validate_within_dir;

/// A host/container mount entry. Mirrors the Python dicts that use the keys
/// `host_path` and `container_path`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mount {
    pub host_path: String,
    pub container_path: String,
}

thread_local! {
    /// Session-scoped map of `container_path -> host_path`, mirroring the
    /// Python `ContextVar[Dict[str, str]]`. Insertion order is not relied upon
    /// in the original (it is a plain dict; iteration order is whatever), so a
    /// `BTreeMap` (deterministic by container path) is used here for stable,
    /// reproducible output.
    static REGISTERED_FILES: RefCell<BTreeMap<String, String>> =
        RefCell::new(BTreeMap::new());
}

/// Process-wide cache of the config-based credential file list (loaded once).
/// Mirrors the Python module-level `_config_files` global.
static CONFIG_FILES: OnceLock<Vec<Mount>> = OnceLock::new();

/// Resolve HERMES_HOME (equivalent to `hermes_constants.get_hermes_home()`).
fn resolve_hermes_home() -> PathBuf {
    get_hermes_home()
}

/// Strip a single trailing `/` from a container base, matching Python's
/// `container_base.rstrip('/')` — except Python's `rstrip` strips *all*
/// trailing slashes, so we replicate that.
fn rstrip_slashes(s: &str) -> &str {
    s.trim_end_matches('/')
}

/// Register a single credential file for mounting into remote sandboxes.
///
/// `relative_path` is relative to `HERMES_HOME` (e.g. `google_token.json`).
/// Returns true if the file exists on the host and was registered.
///
/// Security: rejects absolute paths and path traversal sequences (`..`).
/// The resolved host path must remain inside HERMES_HOME so that a malicious
/// skill cannot declare `required_credential_files: ['../../.ssh/id_rsa']`
/// and exfiltrate sensitive host files into a container sandbox.
pub fn register_credential_file(relative_path: &str, container_base: &str) -> bool {
    let hermes_home = resolve_hermes_home();

    // Reject absolute paths — they bypass the HERMES_HOME sandbox entirely.
    if Path::new(relative_path).is_absolute() {
        log::warn!(
            "credential_files: rejected absolute path {:?} (must be relative to HERMES_HOME)",
            relative_path
        );
        return false;
    }

    let host_path = hermes_home.join(relative_path);

    // Resolve symlinks and normalise `..` before the containment check so that
    // traversal like `../.ssh/id_rsa` cannot escape HERMES_HOME.
    if let Some(containment_error) = validate_within_dir(&host_path, &hermes_home) {
        log::warn!(
            "credential_files: rejected path traversal {:?} ({})",
            relative_path,
            containment_error
        );
        return false;
    }

    let resolved = crate::tool_path_security::resolve_path(&host_path);
    if !resolved.is_file() {
        log::debug!("credential_files: skipping {} (not found)", resolved.display());
        return false;
    }

    let container_path = format!("{}/{}", rstrip_slashes(container_base), relative_path);
    let host_str = resolved.to_string_lossy().to_string();
    REGISTERED_FILES.with(|m| {
        m.borrow_mut().insert(container_path.clone(), host_str.clone());
    });
    log::debug!(
        "credential_files: registered {} -> {}",
        host_str,
        container_path
    );
    true
}

/// Default container base used throughout the module (`/root/.hermes`).
pub const DEFAULT_CONTAINER_BASE: &str = "/root/.hermes";

/// A skill-frontmatter credential entry, which is either a bare relative path
/// string or a mapping with a `path` (or `name`) key.
#[derive(Debug, Clone)]
pub enum CredentialEntry {
    Str(String),
    Map(BTreeMap<String, String>),
}

impl CredentialEntry {
    /// Extract the relative path for this entry, replicating the Python logic:
    /// strings are stripped; dicts use `path` then `name`, stripped. Returns an
    /// empty string when nothing usable is present.
    fn rel_path(&self) -> Option<String> {
        match self {
            CredentialEntry::Str(s) => Some(s.trim().to_string()),
            CredentialEntry::Map(m) => {
                let v = m
                    .get("path")
                    .filter(|s| !s.is_empty())
                    .or_else(|| m.get("name"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                Some(v)
            }
        }
    }
}

/// Register multiple credential files from skill frontmatter entries.
///
/// Each entry is either a string (relative path) or a map with a `path` key.
/// Returns the list of relative paths that were NOT found on the host
/// (i.e. missing files).
pub fn register_credential_files(entries: &[CredentialEntry], container_base: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for entry in entries {
        let rel_path = match entry.rel_path() {
            Some(p) => p,
            None => continue,
        };
        if rel_path.is_empty() {
            continue;
        }
        if !register_credential_file(&rel_path, container_base) {
            missing.push(rel_path);
        }
    }
    missing
}

/// Parse a `serde_yaml::Value` into a list of [`CredentialEntry`], used when
/// the caller wants to feed raw YAML frontmatter directly. Non-string /
/// non-mapping items are skipped (matching the Python `continue`).
pub fn entries_from_value(value: &Value) -> Vec<CredentialEntry> {
    let mut out = Vec::new();
    if let Value::Sequence(seq) = value {
        for item in seq {
            match item {
                Value::String(s) => out.push(CredentialEntry::Str(s.clone())),
                Value::Mapping(map) => {
                    let mut m = BTreeMap::new();
                    for (k, v) in map {
                        if let (Value::String(k), Value::String(v)) = (k, v) {
                            m.insert(k.clone(), v.clone());
                        }
                    }
                    out.push(CredentialEntry::Map(m));
                }
                _ => continue,
            }
        }
    }
    out
}

/// Load `terminal.credential_files` from config.yaml (cached process-wide).
fn load_config_files() -> &'static Vec<Mount> {
    CONFIG_FILES.get_or_init(compute_config_files)
}

fn compute_config_files() -> Vec<Mount> {
    let mut result: Vec<Mount> = Vec::new();
    let hermes_home = resolve_hermes_home();

    let cfg = read_raw_config_value();
    let cred_files = cfg_get(cfg.as_ref(), &["terminal", "credential_files"]);

    if let Some(Value::Sequence(items)) = cred_files {
        for item in items {
            let s = match item {
                Value::String(s) => s,
                _ => continue,
            };
            let rel = s.trim();
            if rel.is_empty() {
                continue;
            }
            if Path::new(rel).is_absolute() {
                log::warn!("credential_files: rejected absolute config path {:?}", rel);
                continue;
            }
            let host_path = hermes_home.join(rel);
            if let Some(containment_error) = validate_within_dir(&host_path, &hermes_home) {
                log::warn!(
                    "credential_files: rejected config path traversal {:?} ({})",
                    rel,
                    containment_error
                );
                continue;
            }
            let resolved_path = crate::tool_path_security::resolve_path(&host_path);
            if resolved_path.is_file() {
                let container_path = format!("/root/.hermes/{}", rel);
                result.push(Mount {
                    host_path: resolved_path.to_string_lossy().to_string(),
                    container_path,
                });
            }
        }
    }

    result
}

/// Read the raw config.yaml as a `serde_yaml::Value`, returning None on any
/// error (mirrors the Python try/except around `read_raw_config`). We read the
/// file directly here so the function is self-contained and testable; the
/// path comes from `get_config_path()`.
fn read_raw_config_value() -> Option<Value> {
    let path = get_config_path();
    let text = std::fs::read_to_string(&path).ok()?;
    serde_yaml::from_str::<Value>(&text).ok()
}

/// Nested-key lookup over a YAML mapping, equivalent to
/// `hermes_cli.config.cfg_get(cfg, *keys)` (and the ported
/// `crate::cli_config::cfg_get`). Kept local so this module does not depend on
/// a not-yet-wired `cli_config` module declaration.
fn cfg_get<'a>(cfg: Option<&'a Value>, keys: &[&str]) -> Option<&'a Value> {
    let mut node = match cfg {
        Some(c) if c.as_mapping().is_some() => c,
        _ => return None,
    };
    for key in keys {
        let map = node.as_mapping()?;
        let key_v = Value::String((*key).to_string());
        match map.get(&key_v) {
            Some(v) => node = v,
            None => return None,
        }
    }
    Some(node)
}

/// Return all credential files that should be mounted into remote sandboxes.
///
/// Each item has `host_path` and `container_path`. Combines skill-registered
/// files and user config.
pub fn get_credential_file_mounts() -> Vec<Mount> {
    // container_path -> host_path
    let mut mounts: BTreeMap<String, String> = BTreeMap::new();

    // Skill-registered files. Re-check existence (file may have been deleted
    // since registration).
    REGISTERED_FILES.with(|m| {
        for (container_path, host_path) in m.borrow().iter() {
            if Path::new(host_path).is_file() {
                mounts.insert(container_path.clone(), host_path.clone());
            }
        }
    });

    // Config-based files.
    for entry in load_config_files() {
        let cp = &entry.container_path;
        if !mounts.contains_key(cp) && Path::new(&entry.host_path).is_file() {
            mounts.insert(cp.clone(), entry.host_path.clone());
        }
    }

    mounts
        .into_iter()
        .map(|(container_path, host_path)| Mount {
            host_path,
            container_path,
        })
        .collect()
}

/// Resolve the configured external skill directories. Mirrors
/// `agent.skill_utils.get_external_skills_dirs`. The ImportError fallback in
/// Python becomes an empty list here when inputs cannot be derived.
fn external_skills_dirs(hermes_home: &Path) -> Vec<PathBuf> {
    let config_path = get_config_path();
    let local_skills = hermes_home.join("skills");
    crate::ag_skill_utils::get_external_skills_dirs(&config_path, hermes_home, &local_skills)
}

/// Return mount info for all skill directories (local + external).
///
/// Skills may include `scripts/`, `templates/`, and `references/`
/// subdirectories that the agent needs to execute inside remote sandboxes.
///
/// **Security:** Bind mounts follow symlinks, so a malicious symlink inside
/// the skills tree could expose arbitrary host files to the container. When
/// symlinks are detected, this function creates a sanitized copy (regular
/// files only) in a temp directory and returns that path instead. When no
/// symlinks are present (the common case), the original directory is returned
/// directly with zero overhead.
///
/// The local skills dir mounts at `<container_base>/skills`, external dirs at
/// `<container_base>/external_skills/<index>`.
pub fn get_skills_directory_mount(container_base: &str) -> Vec<Mount> {
    let mut mounts = Vec::new();
    let hermes_home = resolve_hermes_home();
    let skills_dir = hermes_home.join("skills");
    if skills_dir.is_dir() {
        let host_path = safe_skills_path(&skills_dir);
        mounts.push(Mount {
            host_path,
            container_path: format!("{}/skills", rstrip_slashes(container_base)),
        });
    }

    for (idx, ext_dir) in external_skills_dirs(&hermes_home).into_iter().enumerate() {
        if ext_dir.is_dir() {
            let host_path = safe_skills_path(&ext_dir);
            mounts.push(Mount {
                host_path,
                container_path: format!(
                    "{}/external_skills/{}",
                    rstrip_slashes(container_base),
                    idx
                ),
            });
        }
    }

    mounts
}

thread_local! {
    /// Holds the path of the last symlink-safe temp copy, reused across calls
    /// to avoid accumulation (mirrors the Python module-level
    /// `_safe_skills_tempdir`).
    static SAFE_SKILLS_TEMPDIR: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Recursively walk `dir`, returning every entry (files, dirs, symlinks) as
/// `(path, is_symlink)` so callers can replicate Python's `rglob("*")` which
/// yields all descendants and does *not* follow symlinks while walking.
fn rglob_all(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_collect(dir, &mut out);
    out
}

fn walk_collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        out.push(path.clone());
        // Only descend into real directories, not symlinked directories
        // (Python's rglob does not traverse symlinked dirs).
        let meta = std::fs::symlink_metadata(&path);
        let is_symlink = meta.as_ref().map(|m| m.file_type().is_symlink()).unwrap_or(false);
        if !is_symlink && path.is_dir() {
            walk_collect(&path, out);
        }
    }
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

/// Return `skills_dir` (as a string) if symlink-free, else a sanitized temp
/// copy containing regular files only.
fn safe_skills_path(skills_dir: &Path) -> String {
    let all = rglob_all(skills_dir);
    let symlinks: Vec<&PathBuf> = all.iter().filter(|p| is_symlink(p)).collect();
    if symlinks.is_empty() {
        return skills_dir.to_string_lossy().to_string();
    }

    for link in &symlinks {
        let target = std::fs::read_link(link)
            .map(|t| t.to_string_lossy().to_string())
            .unwrap_or_default();
        log::warn!(
            "credential_files: skipping symlink in skills dir: {} -> {}",
            link.display(),
            target
        );
    }

    // Reuse the same temp dir across calls to avoid accumulation.
    SAFE_SKILLS_TEMPDIR.with(|cell| {
        if let Some(prev) = cell.borrow().as_ref() {
            if prev.is_dir() {
                let _ = std::fs::remove_dir_all(prev);
            }
        }
    });

    let safe_dir = match make_temp_dir("hermes-skills-safe-") {
        Some(d) => d,
        None => {
            // On failure, fall back to the original path (best effort).
            return skills_dir.to_string_lossy().to_string();
        }
    };
    SAFE_SKILLS_TEMPDIR.with(|cell| {
        *cell.borrow_mut() = Some(safe_dir.clone());
    });

    for item in &all {
        if is_symlink(item) {
            continue;
        }
        let rel = match item.strip_prefix(skills_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let target = safe_dir.join(rel);
        if item.is_dir() {
            let _ = std::fs::create_dir_all(&target);
        } else if item.is_file() {
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::copy(item, &target);
        }
    }

    log::info!(
        "credential_files: created symlink-safe skills copy at {}",
        safe_dir.display()
    );
    safe_dir.to_string_lossy().to_string()
}

/// Create a unique temp directory with the given prefix, analogous to
/// `tempfile.mkdtemp(prefix=...)`.
fn make_temp_dir(prefix: &str) -> Option<PathBuf> {
    let base = std::env::temp_dir();
    for _ in 0..1000 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let candidate = base.join(format!("{}{}-{}", prefix, pid, nanos));
        if std::fs::create_dir(&candidate).is_ok() {
            return Some(candidate);
        }
    }
    None
}

/// Yield individual `(host_path, container_path)` entries for skills files.
///
/// Includes both the local skills dir and any external dirs configured via
/// `skills.external_dirs`. Skips symlinks entirely. Preferred for backends
/// that upload files individually (Daytona, Modal) rather than mounting a
/// directory.
pub fn iter_skills_files(container_base: &str) -> Vec<Mount> {
    let mut result = Vec::new();
    let hermes_home = resolve_hermes_home();

    let skills_dir = hermes_home.join("skills");
    if skills_dir.is_dir() {
        let container_root = format!("{}/skills", rstrip_slashes(container_base));
        for item in rglob_all(&skills_dir) {
            if is_symlink(&item) || !item.is_file() {
                continue;
            }
            if let Ok(rel) = item.strip_prefix(&skills_dir) {
                result.push(Mount {
                    host_path: item.to_string_lossy().to_string(),
                    container_path: format!("{}/{}", container_root, rel.to_string_lossy()),
                });
            }
        }
    }

    for (idx, ext_dir) in external_skills_dirs(&hermes_home).into_iter().enumerate() {
        if !ext_dir.is_dir() {
            continue;
        }
        let container_root = format!("{}/external_skills/{}", rstrip_slashes(container_base), idx);
        for item in rglob_all(&ext_dir) {
            if is_symlink(&item) || !item.is_file() {
                continue;
            }
            if let Ok(rel) = item.strip_prefix(&ext_dir) {
                result.push(Mount {
                    host_path: item.to_string_lossy().to_string(),
                    container_path: format!("{}/{}", container_root, rel.to_string_lossy()),
                });
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Cache directory mounts (documents, images, audio, screenshots)
// ---------------------------------------------------------------------------

/// The four cache subdirectories that should be mirrored into remote backends.
/// Each tuple is `(new_subpath, old_name)` matching
/// `hermes_constants.get_hermes_dir()`.
pub const CACHE_DIRS: [(&str, &str); 4] = [
    ("cache/documents", "document_cache"),
    ("cache/images", "image_cache"),
    ("cache/audio", "audio_cache"),
    ("cache/screenshots", "browser_screenshots"),
];

/// Return mount entries for each cache directory that exists on disk.
///
/// Used by Docker to create bind mounts. Each entry has `host_path` and
/// `container_path`. The host path is resolved via `get_hermes_dir()` for
/// backward compatibility with old directory layouts.
pub fn get_cache_directory_mounts(container_base: &str) -> Vec<Mount> {
    let mut mounts = Vec::new();
    for (new_subpath, old_name) in CACHE_DIRS.iter() {
        let host_dir = get_hermes_dir(new_subpath, old_name);
        if host_dir.is_dir() {
            // Always map to the *new* container layout regardless of host layout.
            let container_path = format!("{}/{}", rstrip_slashes(container_base), new_subpath);
            mounts.push(Mount {
                host_path: host_dir.to_string_lossy().to_string(),
                container_path,
            });
        }
    }
    mounts
}

/// Return individual `(host_path, container_path)` entries for cache files.
///
/// Used by Modal to upload files individually and resync before each command.
/// Skips symlinks. The container paths use the new `cache/<subdir>` layout.
pub fn iter_cache_files(container_base: &str) -> Vec<Mount> {
    let mut result = Vec::new();
    for (new_subpath, old_name) in CACHE_DIRS.iter() {
        let host_dir = get_hermes_dir(new_subpath, old_name);
        if !host_dir.is_dir() {
            continue;
        }
        let container_root = format!("{}/{}", rstrip_slashes(container_base), new_subpath);
        for item in rglob_all(&host_dir) {
            if is_symlink(&item) || !item.is_file() {
                continue;
            }
            if let Ok(rel) = item.strip_prefix(&host_dir) {
                result.push(Mount {
                    host_path: item.to_string_lossy().to_string(),
                    container_path: format!("{}/{}", container_root, rel.to_string_lossy()),
                });
            }
        }
    }
    result
}

/// Reset the skill-scoped registry (e.g. on session reset).
pub fn clear_credential_files() {
    REGISTERED_FILES.with(|m| m.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    // Serialize env-mutating tests since HERMES_HOME is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn setup_home(tmp: &Path) {
        unsafe {
            std::env::set_var("HERMES_HOME", tmp);
        }
    }

    #[test]
    fn rstrip_slashes_trims_trailing() {
        assert_eq!(rstrip_slashes("/root/.hermes/"), "/root/.hermes");
        assert_eq!(rstrip_slashes("/root/.hermes///"), "/root/.hermes");
        assert_eq!(rstrip_slashes("/root/.hermes"), "/root/.hermes");
    }

    #[test]
    fn register_rejects_absolute_path() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        assert!(!register_credential_file("/etc/passwd", DEFAULT_CONTAINER_BASE));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn register_rejects_traversal() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        // Even if the file existed outside, containment must reject it.
        assert!(!register_credential_file("../../etc/passwd", DEFAULT_CONTAINER_BASE));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn register_missing_file_returns_false() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        assert!(!register_credential_file("nope.json", DEFAULT_CONTAINER_BASE));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn register_existing_file_succeeds_and_mounts() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        fs::write(tmp.join("google_token.json"), b"{}").unwrap();
        assert!(register_credential_file("google_token.json", DEFAULT_CONTAINER_BASE));

        let mounts = get_credential_file_mounts();
        assert!(mounts
            .iter()
            .any(|m| m.container_path == "/root/.hermes/google_token.json"));
        clear_credential_files();
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn register_uses_container_base_and_strips_slash() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        fs::write(tmp.join("tok.json"), b"x").unwrap();
        assert!(register_credential_file("tok.json", "/custom/base/"));
        let mounts = get_credential_file_mounts();
        assert!(mounts
            .iter()
            .any(|m| m.container_path == "/custom/base/tok.json"));
        clear_credential_files();
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn register_credential_files_collects_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        fs::write(tmp.join("present.json"), b"x").unwrap();

        let mut map = BTreeMap::new();
        map.insert("path".to_string(), "viamap.json".to_string());

        let entries = vec![
            CredentialEntry::Str("present.json".to_string()),
            CredentialEntry::Str("absent.json".to_string()),
            CredentialEntry::Map(map),
            CredentialEntry::Str("   ".to_string()), // empty after strip -> skipped
        ];
        let missing = register_credential_files(&entries, DEFAULT_CONTAINER_BASE);
        assert!(missing.contains(&"absent.json".to_string()));
        assert!(missing.contains(&"viamap.json".to_string()));
        assert!(!missing.contains(&"present.json".to_string()));
        clear_credential_files();
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn entries_from_value_parses_strings_and_maps() {
        let yaml = "- a.json\n- path: b.json\n- 42\n";
        let v: Value = serde_yaml::from_str(yaml).unwrap();
        let entries = entries_from_value(&v);
        // The integer entry is skipped.
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].rel_path().unwrap(), "a.json");
        assert_eq!(entries[1].rel_path().unwrap(), "b.json");
    }

    #[test]
    fn cache_mounts_detect_existing_dirs() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        fs::create_dir_all(tmp.join("cache/images")).unwrap();
        fs::write(tmp.join("cache/images/a.png"), b"img").unwrap();

        let mounts = get_cache_directory_mounts(DEFAULT_CONTAINER_BASE);
        assert!(mounts
            .iter()
            .any(|m| m.container_path == "/root/.hermes/cache/images"));

        let files = iter_cache_files(DEFAULT_CONTAINER_BASE);
        assert!(files
            .iter()
            .any(|m| m.container_path == "/root/.hermes/cache/images/a.png"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn iter_skills_files_skips_symlinks_and_dirs() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        let skills = tmp.join("skills");
        fs::create_dir_all(skills.join("mySkill/scripts")).unwrap();
        fs::write(skills.join("mySkill/SKILL.md"), b"# skill").unwrap();
        fs::write(skills.join("mySkill/scripts/run.sh"), b"echo hi").unwrap();

        let files = iter_skills_files(DEFAULT_CONTAINER_BASE);
        assert!(files
            .iter()
            .any(|m| m.container_path == "/root/.hermes/skills/mySkill/SKILL.md"));
        assert!(files
            .iter()
            .any(|m| m.container_path == "/root/.hermes/skills/mySkill/scripts/run.sh"));
        // Directories are not emitted as files.
        assert!(!files
            .iter()
            .any(|m| m.container_path == "/root/.hermes/skills/mySkill"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clear_resets_registry() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = make_temp_dir("hermes-cf-test-").unwrap();
        setup_home(&tmp);
        clear_credential_files();
        fs::write(tmp.join("c.json"), b"x").unwrap();
        assert!(register_credential_file("c.json", DEFAULT_CONTAINER_BASE));
        assert!(!get_credential_file_mounts().is_empty());
        clear_credential_files();
        assert!(get_credential_file_mounts().is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }
}
