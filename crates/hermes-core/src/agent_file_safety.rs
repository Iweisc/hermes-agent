//! Shared file safety rules used by both tools and ACP shims.
//!
//! Native Rust port of `agent/file_safety.py`. Reproduces the write-denylist,
//! denied-prefix, safe-write-root, and internal-cache read-block semantics.
//!
//! Path-resolution semantics mirror Python's `os.path` helpers:
//! * [`expand_user`] mirrors `os.path.expanduser` (leading `~` / `~/...`).
//! * [`abspath`] mirrors `os.path.abspath` (lexical absolutisation + `.`/`..`
//!   collapse, *without* resolving symlinks).
//! * [`realpath`] mirrors `os.path.realpath` (symlink resolution that still
//!   yields a path for non-existent targets by resolving the longest existing
//!   prefix and re-appending the remainder).

use std::collections::HashSet;
use std::env;
use std::path::{Component, Path, PathBuf};

/// Resolve the active HERMES_HOME (profile-aware) without circular imports.
///
/// Mirrors `_hermes_home_path()` -> `get_hermes_home()`: honour `HERMES_HOME`
/// when set & non-empty, otherwise fall back to `~/.hermes`.
pub fn hermes_home_path() -> PathBuf {
    if let Ok(val) = env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    home_dir().join(".hermes")
}

/// Best-effort home directory, mirroring `os.path.expanduser("~")`.
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Mirror `os.path.expanduser`: expand a leading `~` (current user only).
///
/// `~` and `~/...` expand to the home directory. `~user` forms are left
/// untouched (Python would look them up; we conservatively pass through, which
/// keeps the denylist comparisons safe). Everything else is returned verbatim.
pub fn expand_user(path: &str) -> String {
    if path == "~" {
        return home_dir().to_string_lossy().into_owned();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        let mut home = home_dir();
        home.push(rest);
        return home.to_string_lossy().into_owned();
    }
    path.to_string()
}

/// Lexically collapse `.` / `..` / repeated separators in an absolute path.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out: Vec<Component> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop a preceding *normal* component; never pop past root.
                match out.last() {
                    Some(Component::Normal(_)) => {
                        out.pop();
                    }
                    Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                    _ => out.push(comp),
                }
            }
            other => out.push(other),
        }
    }
    if out.is_empty() {
        return PathBuf::from(".");
    }
    let mut buf = PathBuf::new();
    for comp in out {
        buf.push(comp.as_os_str());
    }
    buf
}

/// Mirror `os.path.abspath`: make absolute (relative to CWD) and collapse
/// `.`/`..` lexically, without resolving symlinks.
pub fn abspath(path: &str) -> PathBuf {
    let p = Path::new(path);
    let absolute = if p.is_absolute() {
        p.to_path_buf()
    } else {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        cwd.join(p)
    };
    lexical_normalize(&absolute)
}

/// Mirror `os.path.realpath`: resolve symlinks while still returning a path for
/// non-existent targets.
///
/// We first make the input absolute (lexically), then canonicalise the longest
/// existing prefix and re-append the non-existent remainder. If nothing in the
/// chain exists we fall back to the lexically-absolutised path. This matches
/// Python's behaviour where `realpath` of a missing file returns the
/// absolutised path with any existing symlink ancestors resolved.
pub fn realpath(path: &str) -> PathBuf {
    let absolute = abspath(path);
    if let Ok(canon) = absolute.canonicalize() {
        return canon;
    }

    // Walk from the deepest existing ancestor.
    let mut remainder: Vec<std::ffi::OsString> = Vec::new();
    let mut current = absolute.as_path();
    loop {
        if let Ok(canon) = current.canonicalize() {
            let mut resolved = canon;
            for part in remainder.iter().rev() {
                resolved.push(part);
            }
            return resolved;
        }
        match current.parent() {
            Some(parent) => {
                if let Some(name) = current.file_name() {
                    remainder.push(name.to_os_string());
                }
                current = parent;
            }
            None => break,
        }
    }
    absolute
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Return exact sensitive paths that must never be written.
///
/// Mirrors `build_write_denied_paths(home)`. `home` is the realpath-resolved
/// home directory (as `is_write_denied` computes it).
pub fn build_write_denied_paths(home: &str) -> HashSet<String> {
    let hermes_home = hermes_home_path();
    let raw: Vec<String> = vec![
        join(home, &[".ssh", "authorized_keys"]),
        join(home, &[".ssh", "id_rsa"]),
        join(home, &[".ssh", "id_ed25519"]),
        join(home, &[".ssh", "config"]),
        path_string(&hermes_home.join(".env")),
        join(home, &[".bashrc"]),
        join(home, &[".zshrc"]),
        join(home, &[".profile"]),
        join(home, &[".bash_profile"]),
        join(home, &[".zprofile"]),
        join(home, &[".netrc"]),
        join(home, &[".pgpass"]),
        join(home, &[".npmrc"]),
        join(home, &[".pypirc"]),
        "/etc/sudoers".to_string(),
        "/etc/passwd".to_string(),
        "/etc/shadow".to_string(),
    ];
    raw.iter().map(|p| path_string(&realpath(p))).collect()
}

/// Return sensitive directory prefixes (each terminated with a path separator)
/// that must never be written. Mirrors `build_write_denied_prefixes(home)`.
pub fn build_write_denied_prefixes(home: &str) -> Vec<String> {
    let raw: Vec<String> = vec![
        join(home, &[".ssh"]),
        join(home, &[".aws"]),
        join(home, &[".gnupg"]),
        join(home, &[".kube"]),
        "/etc/sudoers.d".to_string(),
        "/etc/systemd".to_string(),
        join(home, &[".docker"]),
        join(home, &[".azure"]),
        join(home, &[".config", "gh"]),
    ];
    let sep = std::path::MAIN_SEPARATOR.to_string();
    raw.iter()
        .map(|p| format!("{}{}", path_string(&realpath(p)), sep))
        .collect()
}

/// Join `home` with additional path segments (mirrors `os.path.join`).
fn join(home: &str, parts: &[&str]) -> String {
    let mut buf = PathBuf::from(home);
    for part in parts {
        buf.push(part);
    }
    path_string(&buf)
}

/// Return the resolved `HERMES_WRITE_SAFE_ROOT` path, or `None` if unset.
///
/// Mirrors `get_safe_write_root()`.
pub fn get_safe_write_root() -> Option<String> {
    let root = env::var("HERMES_WRITE_SAFE_ROOT").unwrap_or_default();
    if root.is_empty() {
        return None;
    }
    let expanded = expand_user(&root);
    Some(path_string(&realpath(&expanded)))
}

/// Return `true` if `path` is blocked by the write denylist or safe root.
///
/// Mirrors `is_write_denied(path)`.
pub fn is_write_denied(path: &str) -> bool {
    let home = path_string(&realpath(&expand_user("~")));
    let expanded = path_string(&abspath(&expand_user(path)));
    let resolved = path_string(&realpath(&expanded));

    let mut candidates: HashSet<String> = HashSet::new();
    candidates.insert(expanded.clone());
    candidates.insert(resolved.clone());

    let denied = build_write_denied_paths(&home);
    if candidates.iter().any(|c| denied.contains(c)) {
        return true;
    }

    for prefix in build_write_denied_prefixes(&home) {
        for candidate in &candidates {
            if candidate.starts_with(&prefix) {
                return true;
            }
        }
    }

    if let Some(safe_root) = get_safe_write_root() {
        let sep = std::path::MAIN_SEPARATOR.to_string();
        let within = resolved == safe_root || resolved.starts_with(&format!("{safe_root}{sep}"));
        if !within {
            return true;
        }
    }

    false
}

/// Return an error message when a read targets internal Hermes cache files.
///
/// Mirrors `get_read_block_error(path)`. Returns `Some(message)` when `path`
/// resolves to (or under) the internal skills hub cache, else `None`.
pub fn get_read_block_error(path: &str) -> Option<String> {
    let resolved = realpath(&expand_user(path));
    let hermes_home = realpath(&path_string(&hermes_home_path()));

    let blocked_dirs = [
        hermes_home
            .join("skills")
            .join(".hub")
            .join("index-cache"),
        hermes_home.join("skills").join(".hub"),
    ];

    for blocked in &blocked_dirs {
        if resolved == *blocked || resolved.starts_with(blocked) {
            return Some(format!(
                "Access denied: {path} is an internal Hermes cache file \
                 and cannot be read directly to prevent prompt injection. \
                 Use the skills_list or skill_view tools instead."
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    // Env vars are process-global; serialise tests that mutate them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let prev = env::var(key).ok();
            // SAFETY: env mutation is serialised via ENV_LOCK in these tests.
            unsafe { env::set_var(key, val) };
            EnvGuard { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = env::var(key).ok();
            // SAFETY: env mutation is serialised via ENV_LOCK in these tests.
            unsafe { env::remove_var(key) };
            EnvGuard { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: env mutation is serialised via ENV_LOCK in these tests.
            unsafe {
                match &self.prev {
                    Some(v) => env::set_var(self.key, v),
                    None => env::remove_var(self.key),
                }
            }
        }
    }

    fn home() -> String {
        path_string(&realpath(&expand_user("~")))
    }

    #[test]
    fn expand_user_handles_tilde() {
        let h = home_dir();
        assert_eq!(expand_user("~"), h.to_string_lossy());
        assert_eq!(
            expand_user("~/foo/bar"),
            path_string(&h.join("foo").join("bar"))
        );
        assert_eq!(expand_user("/abs/path"), "/abs/path");
        // ~user form is passed through.
        assert_eq!(expand_user("~root/x"), "~root/x");
    }

    #[test]
    fn abspath_collapses_dot_segments() {
        assert_eq!(abspath("/a/b/../c"), PathBuf::from("/a/c"));
        assert_eq!(abspath("/a/./b"), PathBuf::from("/a/b"));
        assert_eq!(abspath("/a/b/.."), PathBuf::from("/a"));
        // Cannot escape root.
        assert_eq!(abspath("/../../x"), PathBuf::from("/x"));
    }

    #[test]
    fn exact_denied_paths_blocked() {
        let _g = lock_env();
        let _hh = EnvGuard::unset("HERMES_HOME");
        let _sr = EnvGuard::unset("HERMES_WRITE_SAFE_ROOT");

        for rel in [
            ".ssh/id_rsa",
            ".ssh/authorized_keys",
            ".ssh/config",
            ".bashrc",
            ".zshrc",
            ".netrc",
            ".npmrc",
        ] {
            assert!(
                is_write_denied(&format!("~/{rel}")),
                "expected ~/{rel} to be denied"
            );
        }
        assert!(is_write_denied("/etc/passwd"));
        assert!(is_write_denied("/etc/shadow"));
        assert!(is_write_denied("/etc/sudoers"));
    }

    #[test]
    fn denied_prefixes_block_subpaths() {
        let _g = lock_env();
        let _hh = EnvGuard::unset("HERMES_HOME");
        let _sr = EnvGuard::unset("HERMES_WRITE_SAFE_ROOT");

        assert!(is_write_denied("~/.ssh/some_new_key"));
        assert!(is_write_denied("~/.aws/credentials"));
        assert!(is_write_denied("~/.gnupg/secring.gpg"));
        assert!(is_write_denied("~/.kube/config"));
        assert!(is_write_denied("~/.config/gh/hosts.yml"));
        assert!(is_write_denied("/etc/systemd/system/foo.service"));
        assert!(is_write_denied("/etc/sudoers.d/custom"));
    }

    #[test]
    fn ordinary_paths_allowed_without_safe_root() {
        let _g = lock_env();
        let _hh = EnvGuard::unset("HERMES_HOME");
        let _sr = EnvGuard::unset("HERMES_WRITE_SAFE_ROOT");

        let h = home();
        assert!(!is_write_denied(&format!("{h}/projects/file.txt")));
        assert!(!is_write_denied("/tmp/whatever.txt"));
    }

    #[test]
    fn safe_root_blocks_outside_paths() {
        let _g = lock_env();
        let _hh = EnvGuard::unset("HERMES_HOME");

        let dir = std::env::temp_dir();
        let safe = dir.join("hermes_safe_root_test");
        std::fs::create_dir_all(&safe).unwrap();
        let safe_str = path_string(&realpath(&path_string(&safe)));
        let _sr = EnvGuard::set("HERMES_WRITE_SAFE_ROOT", &safe_str);

        // A path inside the safe root is allowed (and not otherwise denied).
        let inside = format!("{safe_str}/ok.txt");
        assert!(!is_write_denied(&inside), "inside safe root should be allowed");

        // The safe root itself is allowed.
        assert!(!is_write_denied(&safe_str));

        // Anything outside the safe root is denied.
        assert!(is_write_denied("/tmp/definitely_outside_safe_root.txt"));
    }

    #[test]
    fn safe_root_unset_returns_none() {
        let _g = lock_env();
        let _sr = EnvGuard::unset("HERMES_WRITE_SAFE_ROOT");
        assert_eq!(get_safe_write_root(), None);

        let _sr2 = EnvGuard::set("HERMES_WRITE_SAFE_ROOT", "");
        assert_eq!(get_safe_write_root(), None);
    }

    #[test]
    fn read_block_targets_hub_cache() {
        let _g = lock_env();
        let dir = std::env::temp_dir().join("hermes_home_read_block_test");
        std::fs::create_dir_all(dir.join("skills").join(".hub").join("index-cache")).unwrap();
        let hh = path_string(&dir);
        let _hh = EnvGuard::set("HERMES_HOME", &hh);

        let blocked = format!("{hh}/skills/.hub/index-cache/foo.json");
        let msg = get_read_block_error(&blocked);
        assert!(msg.is_some(), "expected hub cache read to be blocked");
        let msg = msg.unwrap();
        assert!(msg.contains("Access denied"));
        assert!(msg.contains("skills_list"));

        // A path under .hub but outside index-cache is still blocked by the
        // broader .hub rule.
        let blocked2 = format!("{hh}/skills/.hub/manifest.json");
        assert!(get_read_block_error(&blocked2).is_some());

        // An unrelated path is not blocked.
        let allowed = format!("{hh}/skills/productivity/readme.md");
        assert_eq!(get_read_block_error(&allowed), None);
    }

    #[test]
    fn read_block_allows_normal_paths() {
        let _g = lock_env();
        let _hh = EnvGuard::unset("HERMES_HOME");
        assert_eq!(get_read_block_error("/tmp/some_file.txt"), None);
    }
}
