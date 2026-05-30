use std::path::{Path, PathBuf};

pub fn resolve_repo_python(project_root: &Path, override_env_var: Option<&str>) -> Option<PathBuf> {
    if let Some(env_var) = override_env_var {
        if let Some(value) = std::env::var(env_var).ok() {
            if !value.trim().is_empty() {
                return Some(PathBuf::from(value.trim()));
            }
        }
    }

    let candidates = [
        project_root.join(".venv").join(python_bin_name()),
        project_root.join("venv").join(python_bin_name()),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(".hermes")
            .join("hermes-agent")
            .join("venv")
            .join(python_bin_name()),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    which_on_path("python3").or_else(|| which_on_path("python"))
}

pub fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn python_bin_name() -> &'static str {
    #[cfg(windows)]
    {
        "Scripts/python.exe"
    }
    #[cfg(not(windows))]
    {
        "bin/python"
    }
}

/// Check whether a top-level Python module/package is importable for the given
/// interpreter, without spawning Python in the common case.
///
/// This replaces the `python -c "import <mod>"` / `find_spec` availability
/// probes used for optional third-party dependencies (mautrix, honcho, mem0,
/// …). It resolves the interpreter's `site-packages` directories from the
/// filesystem and looks for a matching package directory, single-file module,
/// or `*.dist-info`/`*.egg-info` distribution marker.
///
/// If no `site-packages` can be located (e.g. an unusual interpreter layout),
/// it falls back to actually running the interpreter so behavior never
/// regresses relative to the previous subprocess probe.
///
/// Note: unlike `python -c "import X"`, this does not see modules importable
/// only via a custom `PYTHONPATH` when site-packages is otherwise present.
/// These probes target optional pip-installed dependencies (mautrix, honcho,
/// mem0, …), which always land in site-packages, so that case does not arise
/// in practice.
pub fn python_module_installed(python: &Path, module: &str) -> bool {
    let module = module.trim();
    if module.is_empty() {
        return false;
    }
    // Only the top-level name matters for an availability probe.
    let top = module.split('.').next().unwrap_or(module);

    let site_dirs = site_packages_dirs(python);
    if site_dirs.is_empty() {
        return python_import_probe(python, module);
    }
    for dir in &site_dirs {
        if site_packages_has_module(dir, top) {
            return true;
        }
    }
    false
}

/// Locate `site-packages` (and `dist-packages`) directories for `python` by
/// walking the interpreter's `lib*/python*/` layout. Returns an empty vec when
/// the interpreter prefix can't be determined from the binary path.
fn site_packages_dirs(python: &Path) -> Vec<PathBuf> {
    // python is typically <prefix>/bin/python(.exe) or <prefix>/Scripts/python.exe.
    let mut prefixes: Vec<PathBuf> = Vec::new();
    if let Some(bin_dir) = python.parent() {
        if let Some(prefix) = bin_dir.parent() {
            prefixes.push(prefix.to_path_buf());
        }
        // Also consider the bin dir's own parent-less case (rare).
    }

    let mut result = Vec::new();
    for prefix in prefixes {
        // Windows venvs: <prefix>/Lib/site-packages
        let win = prefix.join("Lib").join("site-packages");
        if win.is_dir() {
            result.push(win);
        }
        // POSIX: <prefix>/lib/pythonX.Y/site-packages (and dist-packages)
        for lib in ["lib", "lib64"] {
            let lib_dir = prefix.join(lib);
            let Ok(entries) = std::fs::read_dir(&lib_dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                if !name.starts_with("python") {
                    continue;
                }
                for leaf in ["site-packages", "dist-packages"] {
                    let candidate = entry.path().join(leaf);
                    if candidate.is_dir() {
                        result.push(candidate);
                    }
                }
            }
        }
    }
    result
}

/// Return true when `site_packages` contains `module` as an importable package
/// directory, a single-file module, or a distribution metadata marker.
fn site_packages_has_module(site_packages: &Path, module: &str) -> bool {
    // Regular package: <module>/__init__.py(.pyc) or a namespace package dir.
    let pkg = site_packages.join(module);
    if pkg.join("__init__.py").is_file()
        || pkg.join("__init__.pyc").is_file()
        || pkg.is_dir()
    {
        return true;
    }
    // Single-file module: <module>.py / .pyc / compiled extension.
    for ext in ["py", "pyc", "so", "pyd"] {
        if site_packages.join(format!("{module}.{ext}")).is_file() {
            return true;
        }
    }
    // Distribution markers: <module>-<ver>.dist-info / .egg-info. The
    // distribution name uses '-' where the project name had '-'/'_'; normalize
    // both sides so e.g. "hindsight_client" matches "hindsight-client-*.dist-info".
    let normalized = module.replace('_', "-").to_lowercase();
    if let Ok(entries) = std::fs::read_dir(site_packages) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let lower = name.to_lowercase();
            if !(lower.ends_with(".dist-info") || lower.ends_with(".egg-info")) {
                continue;
            }
            // Strip the trailing "-<version>.dist-info"/".egg-info" and compare
            // the distribution name, normalizing separators. The version is the
            // LAST hyphen-delimited segment, so split on the final '-' — the
            // distribution name itself may contain hyphens (e.g.
            // "hindsight-client-1.0.0.dist-info" -> "hindsight-client").
            let stem = lower
                .trim_end_matches(".dist-info")
                .trim_end_matches(".egg-info");
            let dist_name = stem.rsplit_once('-').map(|(name, _ver)| name).unwrap_or(stem);
            let dist_norm = dist_name.replace('_', "-");
            if dist_norm == normalized || dist_name == module.to_lowercase() {
                return true;
            }
        }
    }
    false
}

/// Last-resort probe: actually run `python -c "import <module>"`.
fn python_import_probe(python: &Path, module: &str) -> bool {
    std::process::Command::new(python)
        .arg("-c")
        .arg(format!("import {module}"))
        .output()
        .ok()
        .is_some_and(|output| output.status.success())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-pybridge-{label}-{unique}"))
    }

    #[test]
    fn project_root_contains_python_cli() {
        let root = project_root();
        assert!(root.join("hermes_cli").join("main.py").exists());
    }

    #[test]
    fn resolve_repo_python_prefers_local_venv() {
        let root = temp_path("venv");
        let python = root.join(".venv").join(python_bin_name());
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(&python, b"").unwrap();
        let resolved = resolve_repo_python(&root, None).unwrap();
        assert_eq!(resolved, python);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn python_module_installed_detects_site_packages_layouts() {
        // Build a fake POSIX venv: <prefix>/bin/python and
        // <prefix>/lib/python3.12/site-packages/...
        let prefix = temp_path("venv-probe");
        let python = prefix.join("bin").join("python");
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(&python, b"").unwrap();
        let site = prefix.join("lib").join("python3.12").join("site-packages");
        fs::create_dir_all(&site).unwrap();

        // Package with __init__.py
        fs::create_dir_all(site.join("mautrix")).unwrap();
        fs::write(site.join("mautrix").join("__init__.py"), b"").unwrap();
        // Single-file module
        fs::write(site.join("singlemod.py"), b"").unwrap();
        // dist-info marker with a MULTI-DASH distribution name and no package
        // dir (forces the dist-info branch): hindsight_client maps to the
        // distribution "hindsight-client", filed as hindsight-client-<ver>.dist-info.
        fs::create_dir_all(site.join("hindsight-client-1.2.0.dist-info")).unwrap();

        assert!(python_module_installed(&python, "mautrix"));
        assert!(python_module_installed(&python, "singlemod"));
        assert!(python_module_installed(&python, "hindsight_client"));
        // The version is the LAST hyphen segment: must not match just the
        // first token ("hindsight") of a multi-dash distribution name.
        assert!(!python_module_installed(&python, "hindsight"));
        // Submodule path resolves on the top-level name.
        assert!(python_module_installed(&python, "mautrix.client"));
        // Absent package.
        assert!(!python_module_installed(&python, "definitely_absent_pkg"));

        let _ = fs::remove_dir_all(prefix);
    }
}
