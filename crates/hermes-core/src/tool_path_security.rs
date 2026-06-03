//! Shared path validation helpers for tool implementations.
//!
//! Native Rust port of `tools/path_security.py`.
//!
//! Extracts the `resolve() + relative_to()` and `..` traversal check
//! patterns previously duplicated across `skill_manager_tool`, `skills_tool`,
//! `skills_hub`, `cronjob_tools`, and `credential_files`.

use std::path::{Component, Path, PathBuf};

/// Ensure `path` resolves to a location within `root`.
///
/// Returns an error message string if validation fails, or `None` if the
/// path is safe. Mirrors Python's `Path.resolve()` which follows symlinks
/// and normalizes `..` components.
///
/// In Python this relied on `Path.resolve()` raising `OSError` and
/// `Path.relative_to()` raising `ValueError`. Here we canonicalize both
/// paths (following symlinks) and verify `root` is a prefix of the resolved
/// path. When canonicalization fails (e.g. a nonexistent component) we fall
/// back to a lexical normalization so behavior stays well-defined even for
/// paths that do not yet exist on disk — matching the intent of the original
/// "does this escape the allowed directory" check.
///
/// # Example
///
/// ```
/// use std::path::Path;
/// use hermes_core::tool_path_security::validate_within_dir;
///
/// let err = validate_within_dir(Path::new("/tmp"), Path::new("/tmp"));
/// assert!(err.is_none());
/// ```
pub fn validate_within_dir(path: &Path, root: &Path) -> Option<String> {
    let resolved = resolve_path(path);
    let root_resolved = resolve_path(root);

    if resolved.starts_with(&root_resolved) {
        None
    } else {
        Some(format!(
            "Path escapes allowed directory: {} is not within {}",
            resolved.display(),
            root_resolved.display()
        ))
    }
}

/// Resolve a path the way Python's `Path.resolve()` does: follow symlinks and
/// normalize `..`/`.` components, returning an absolute path.
///
/// Falls back to lexical normalization (against the current working directory
/// when relative) if the path — or a parent of it — does not exist on disk.
pub fn resolve_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    lexical_normalize(path)
}

/// Lexically normalize a path without touching the filesystem.
///
/// Resolves `.` and `..` purely textually and makes the result absolute by
/// prepending the current working directory when the input is relative.
pub fn lexical_normalize(path: &Path) -> PathBuf {
    let base: PathBuf = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };

    let combined = if base.as_os_str().is_empty() {
        path.to_path_buf()
    } else {
        base.join(path)
    };

    let mut out: Vec<Component> = Vec::new();
    for comp in combined.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                match out.last() {
                    Some(Component::Normal(_)) => {
                        out.pop();
                    }
                    Some(Component::RootDir) | Some(Component::Prefix(_)) => {
                        // Cannot go above the root.
                    }
                    _ => {
                        out.push(comp);
                    }
                }
            }
            other => out.push(other),
        }
    }

    let mut result = PathBuf::new();
    for comp in out {
        result.push(comp.as_os_str());
    }
    if result.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        result
    }
}

/// Return `true` if `path_str` contains `..` traversal components.
///
/// Quick check for obvious traversal attempts before doing full resolution.
/// Mirrors Python's `".." in Path(path_str).parts`.
pub fn has_traversal_component(path_str: &str) -> bool {
    Path::new(path_str)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn traversal_detected() {
        assert!(has_traversal_component("../etc/passwd"));
        assert!(has_traversal_component("foo/../bar"));
        assert!(has_traversal_component("a/b/.."));
    }

    #[test]
    fn no_traversal() {
        assert!(!has_traversal_component("foo/bar"));
        assert!(!has_traversal_component("./foo/bar"));
        assert!(!has_traversal_component("/abs/path"));
        // ".." only matches as a whole component, not a substring.
        assert!(!has_traversal_component("foo/..bar/baz"));
        assert!(!has_traversal_component("foo/bar.."));
    }

    #[test]
    fn lexical_normalize_resolves_dotdot() {
        let n = lexical_normalize(Path::new("/a/b/../c"));
        assert_eq!(n, PathBuf::from("/a/c"));
    }

    #[test]
    fn lexical_normalize_strips_curdir() {
        let n = lexical_normalize(Path::new("/a/./b/./c"));
        assert_eq!(n, PathBuf::from("/a/b/c"));
    }

    #[test]
    fn lexical_normalize_cannot_escape_root() {
        let n = lexical_normalize(Path::new("/../../etc"));
        assert_eq!(n, PathBuf::from("/etc"));
    }

    #[test]
    fn within_dir_ok() {
        let tmp = std::env::temp_dir();
        let root = tmp.join("hermes_path_sec_test_ok");
        let sub = root.join("inner");
        let _ = fs::create_dir_all(&sub);

        assert!(validate_within_dir(&sub, &root).is_none());
        assert!(validate_within_dir(&root, &root).is_none());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn escapes_dir_detected() {
        let tmp = std::env::temp_dir();
        let root = tmp.join("hermes_path_sec_test_escape");
        let _ = fs::create_dir_all(&root);

        let outside = root.join("..").join("somewhere_else");
        let err = validate_within_dir(&outside, &root);
        assert!(err.is_some(), "expected escape to be detected");
        assert!(err.unwrap().contains("escapes allowed directory"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn nonexistent_within_dir_ok() {
        let tmp = std::env::temp_dir();
        let root = tmp.join("hermes_path_sec_test_nonexist");
        let _ = fs::create_dir_all(&root);

        // A path that does not exist yet but is lexically inside root.
        let candidate = root.join("not_created_yet").join("file.txt");
        assert!(validate_within_dir(&candidate, &root).is_none());

        let _ = fs::remove_dir_all(&root);
    }
}
