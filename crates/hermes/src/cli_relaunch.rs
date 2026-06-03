//! Unified self-relaunch for the Hermes CLI.
//!
//! Native Rust port of `hermes_cli/relaunch.py`.
//!
//! Preserves critical flags (`--tui`, `--dev`, `--profile`, `--model`, etc.)
//! across process replacement so that `hermes sessions browse` or a post-setup
//! relaunch doesn't silently drop the user's UI mode or other preferences.
//!
//! Also works when `hermes` is not on PATH (e.g. `nix run` or `python -m`).
//!
//! The flag-inheritance metadata is introspected from [`crate::cli__parser`]
//! exactly as the Python builder introspected the live argparse parser: it
//! walks the top-level parser flags then the `chat` subparser flags, expands
//! each into `(option_string, takes_value)` pairs (long then short), dedups by
//! `(option, takes_value)`, and finally appends `PRE_ARGPARSE_INHERITED_FLAGS`.

use std::path::Path;

use crate::cli__parser::{
    chat_inherited_flags, inherited_flags, InheritedFlag, PRE_ARGPARSE_INHERITED_FLAGS,
};

/// Build the `(option_string, takes_value)` table of flags that must survive a
/// self-relaunch, by introspecting the real parser used by `hermes` itself.
///
/// A flag participates if its parser entry carries `inherit_on_relaunch = True`
/// (in this port: appears in [`inherited_flags`] / [`chat_inherited_flags`]).
///
/// Mirrors `relaunch._build_inherited_flag_table`. Both the top-level parser
/// and the `chat` parser are scanned, in that order; for each flag, every
/// option string (long, then short) is emitted, with `(opt, takes_value)`
/// deduplicated across the whole table. `PRE_ARGPARSE_INHERITED_FLAGS` is
/// appended last.
pub fn build_inherited_flag_table() -> Vec<(String, bool)> {
    let mut table: Vec<(String, bool)> = Vec::new();
    let mut seen: std::collections::HashSet<(String, bool)> = std::collections::HashSet::new();

    // The Python source iterates `(parser, chat_parser)` and, within each, the
    // parser's `_actions` in registration order. For each action it walks
    // `action.option_strings`. argparse stores option strings in the order the
    // user passed them to `add_argument` (short forms typically before long in
    // the Python registrations, but the dedup by `(opt, takes_value)` makes the
    // emitted set order-stable regardless). Here we emit long then short, which
    // produces the same final set; dedup keys on the exact tuple.
    let push = |flag: &InheritedFlag, table: &mut Vec<(String, bool)>, seen: &mut std::collections::HashSet<(String, bool)>| {
        let takes_value = flag.takes_value;
        let mut opts: Vec<&str> = vec![flag.long];
        if let Some(short) = flag.short {
            opts.push(short);
        }
        for opt in opts {
            let key = (opt.to_string(), takes_value);
            if !seen.contains(&key) {
                seen.insert(key.clone());
                table.push(key);
            }
        }
    };

    for flag in inherited_flags() {
        push(&flag, &mut table, &mut seen);
    }
    for flag in chat_inherited_flags() {
        push(&flag, &mut table, &mut seen);
    }

    for &(flag, takes_value) in PRE_ARGPARSE_INHERITED_FLAGS {
        let key = (flag.to_string(), takes_value);
        // Python extends unconditionally (no dedup against `seen`).
        table.push(key);
    }

    table
}

/// Pull out flags that should carry over into a self-relaunched hermes.
///
/// Mirrors `relaunch._extract_inherited_flags`.
pub fn extract_inherited_flags(argv: &[String], table: &[(String, bool)]) -> Vec<String> {
    let mut flags: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < argv.len() {
        let arg = &argv[i];

        if arg.contains('=') {
            let key = arg.splitn(2, '=').next().unwrap_or("");
            for (flag, _) in table {
                if key == flag.as_str() {
                    flags.push(arg.clone());
                    break;
                }
            }
            i += 1;
            continue;
        }

        for (flag, takes_value) in table {
            if arg == flag {
                flags.push(arg.clone());
                if *takes_value && i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                    flags.push(argv[i + 1].clone());
                    i += 1;
                }
                break;
            }
        }
        i += 1;
    }
    flags
}

/// True if `path` exists, is a regular file, and is executable by the current
/// process. Equivalent to `os.path.isfile(p) and os.access(p, os.X_OK)`.
fn is_executable_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    access_x_ok(path)
}

/// Equivalent of `os.access(path, os.X_OK)`.
#[cfg(unix)]
fn access_x_ok(path: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = match CString::new(path.as_os_str().as_bytes()) {
        Ok(c) => c,
        Err(_) => return false,
    };
    // libc::X_OK == 1
    unsafe { libc::access(c.as_ptr(), libc::X_OK) == 0 }
}

#[cfg(not(unix))]
fn access_x_ok(path: &Path) -> bool {
    // On non-unix, `os.access(p, X_OK)` is approximated by file existence; the
    // Python tooling for this module is unix-first.
    path.is_file()
}

/// Look up an executable named `name` on `PATH`. Equivalent to
/// `shutil.which(name)`.
fn which(name: &str) -> Option<String> {
    // If the name contains a path separator, `shutil.which` checks it directly.
    if name.contains('/') {
        let p = Path::new(name);
        if is_executable_file(p) {
            return Some(name.to_string());
        }
        return None;
    }

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if is_executable_file(&candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

/// Find the hermes entry point.
///
/// Priority:
///   1. `argv[0]` if it resolves to a real executable.
///   2. `which("hermes")` on PATH.
///   3. `None` -> caller should fall back to `python -m hermes_cli.main`.
///
/// Mirrors `relaunch.resolve_hermes_bin`. `argv0` is passed explicitly (the
/// Python code read `sys.argv[0]`); callers should pass the program's argv[0].
pub fn resolve_hermes_bin(argv0: &str) -> Option<String> {
    let argv0_path = Path::new(argv0);

    // Absolute path to an executable (covers nix store, venv wrappers, etc.)
    if argv0_path.is_absolute() && argv0_path.is_file() && access_x_ok(argv0_path) {
        return Some(argv0.to_string());
    }

    // Relative path — resolve against CWD.
    if !argv0.starts_with('-') && argv0_path.is_file() {
        if let Ok(abs) = std::fs::canonicalize(argv0_path) {
            if access_x_ok(&abs) {
                return Some(abs.to_string_lossy().into_owned());
            }
        } else {
            // canonicalize may fail; fall back to an absolute join with CWD.
            if let Ok(cwd) = std::env::current_dir() {
                let abs = cwd.join(argv0_path);
                if access_x_ok(&abs) {
                    return Some(abs.to_string_lossy().into_owned());
                }
            }
        }
    }

    // PATH lookup.
    if let Some(path_bin) = which("hermes") {
        return Some(path_bin);
    }

    None
}

/// Construct an argv list for replacing the current process with hermes.
///
/// Mirrors `relaunch.build_relaunch_argv`.
///
/// Args:
/// - `extra_args`: Arguments to append (e.g. `["--resume", id]`).
/// - `preserve_inherited`: Whether to carry over UI / behaviour flags tagged
///   `inherit_on_relaunch`.
/// - `original_argv`: The original argv (without argv[0]) to scan for flags.
///   In the Python code this defaulted to `sys.argv[1:]`; here the caller
///   passes it explicitly.
/// - `argv0`: The current process's argv[0], used to resolve the hermes binary.
/// - `python_executable`: Path to the Python interpreter, used for the
///   `python -m hermes_cli.main` fallback (Python's `sys.executable`).
pub fn build_relaunch_argv(
    extra_args: &[String],
    preserve_inherited: bool,
    original_argv: &[String],
    argv0: &str,
    python_executable: &str,
) -> Vec<String> {
    let bin_path = resolve_hermes_bin(argv0);

    let mut argv: Vec<String> = match bin_path {
        Some(p) => vec![p],
        None => vec![
            python_executable.to_string(),
            "-m".to_string(),
            "hermes_cli.main".to_string(),
        ],
    };

    if preserve_inherited {
        let table = build_inherited_flag_table();
        argv.extend(extract_inherited_flags(original_argv, &table));
    }

    argv.extend(extra_args.iter().cloned());
    argv
}

/// Replace the current process with a fresh hermes invocation.
///
/// Mirrors `relaunch.relaunch`, which calls `os.execvp(new_argv[0], new_argv)`.
///
/// On success this never returns (the current process image is replaced). On
/// failure it returns the underlying I/O error.
#[cfg(unix)]
pub fn relaunch(
    extra_args: &[String],
    preserve_inherited: bool,
    original_argv: &[String],
    argv0: &str,
    python_executable: &str,
) -> std::io::Error {
    let new_argv = build_relaunch_argv(
        extra_args,
        preserve_inherited,
        original_argv,
        argv0,
        python_executable,
    );

    // `os.execvp` searches PATH for argv[0] when it has no slash. `exec::execvp`
    // is unavailable without an extra crate, so replicate the semantics with
    // libc directly.
    exec_replace(&new_argv)
}

/// Perform an `execvp`-style process replacement. Returns the I/O error if the
/// exec call fails (it does not return on success).
#[cfg(unix)]
fn exec_replace(argv: &[String]) -> std::io::Error {
    use std::ffi::CString;

    if argv.is_empty() {
        return std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty argv");
    }

    let prog = match CString::new(argv[0].as_str()) {
        Ok(c) => c,
        Err(e) => return std::io::Error::new(std::io::ErrorKind::InvalidInput, e),
    };

    let mut c_args: Vec<CString> = Vec::with_capacity(argv.len());
    for a in argv {
        match CString::new(a.as_str()) {
            Ok(c) => c_args.push(c),
            Err(e) => return std::io::Error::new(std::io::ErrorKind::InvalidInput, e),
        }
    }

    let mut ptrs: Vec<*const libc::c_char> = c_args.iter().map(|c| c.as_ptr()).collect();
    ptrs.push(std::ptr::null());

    unsafe {
        // execvp searches PATH when prog has no slash, matching os.execvp.
        libc::execvp(prog.as_ptr(), ptrs.as_ptr());
    }

    // Only reached on failure.
    std::io::Error::last_os_error()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Vec<(String, bool)> {
        build_inherited_flag_table()
    }

    fn has(table: &[(String, bool)], flag: &str, takes_value: bool) -> bool {
        table.iter().any(|(f, tv)| f == flag && *tv == takes_value)
    }

    #[test]
    fn table_includes_core_flags() {
        let t = table();
        assert!(has(&t, "--model", true));
        assert!(has(&t, "-m", true));
        assert!(has(&t, "--tui", false));
        assert!(has(&t, "--dev", false));
        assert!(has(&t, "--skills", true));
        assert!(has(&t, "-s", true));
        // PRE_ARGPARSE entries appended last.
        assert!(has(&t, "--profile", true));
        assert!(has(&t, "-p", true));
    }

    #[test]
    fn table_dedup_no_duplicate_keys() {
        let t = table();
        // `--model`/`-m` appear in both top-level and chat parsers; the dedup
        // on (opt, takes_value) means only one entry each (excluding the
        // unconditionally-appended PRE_ARGPARSE entries, which differ in flag).
        let model_count = t.iter().filter(|(f, _)| f == "--model").count();
        assert_eq!(model_count, 1);
        let m_count = t.iter().filter(|(f, _)| f == "-m").count();
        assert_eq!(m_count, 1);
    }

    #[test]
    fn extract_store_true_switch() {
        let t = table();
        let argv = vec!["chat".to_string(), "--tui".to_string(), "--dev".to_string()];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(got, vec!["--tui".to_string(), "--dev".to_string()]);
    }

    #[test]
    fn extract_flag_with_value() {
        let t = table();
        let argv = vec![
            "--model".to_string(),
            "gpt-4".to_string(),
            "ignored".to_string(),
        ];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(got, vec!["--model".to_string(), "gpt-4".to_string()]);
    }

    #[test]
    fn extract_equals_form() {
        let t = table();
        let argv = vec!["--model=gpt-4".to_string(), "--profile=work".to_string()];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(
            got,
            vec!["--model=gpt-4".to_string(), "--profile=work".to_string()]
        );
    }

    #[test]
    fn extract_value_flag_with_dash_next_does_not_consume() {
        // `--model --tui`: the value-taking flag does not eat a following
        // option-looking token (mirrors `not argv[i+1].startswith('-')`).
        let t = table();
        let argv = vec!["--model".to_string(), "--tui".to_string()];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(got, vec!["--model".to_string(), "--tui".to_string()]);
    }

    #[test]
    fn extract_ignores_unknown_flags() {
        let t = table();
        let argv = vec![
            "--not-a-flag".to_string(),
            "value".to_string(),
            "--tui".to_string(),
        ];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(got, vec!["--tui".to_string()]);
    }

    #[test]
    fn extract_short_skills_value() {
        let t = table();
        let argv = vec!["-s".to_string(), "skill-a,skill-b".to_string()];
        let got = extract_inherited_flags(&argv, &t);
        assert_eq!(got, vec!["-s".to_string(), "skill-a,skill-b".to_string()]);
    }

    #[test]
    fn build_relaunch_argv_falls_back_to_python() {
        // argv0 that won't resolve, and `hermes` unlikely on PATH inside the
        // sandbox PATH we set below.
        let original = vec!["--tui".to_string(), "--model".to_string(), "gpt-4".to_string()];
        let extra = vec!["--resume".to_string(), "abc".to_string()];

        let saved_path = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("PATH", "/nonexistent-hermes-dir");
        }

        let argv = build_relaunch_argv(
            &extra,
            true,
            &original,
            "-",          // argv0 starting with '-' => not a file path
            "/usr/bin/python3",
        );

        unsafe {
            match saved_path {
                Some(p) => std::env::set_var("PATH", p),
                None => std::env::remove_var("PATH"),
            }
        }

        assert_eq!(argv[0], "/usr/bin/python3");
        assert_eq!(argv[1], "-m");
        assert_eq!(argv[2], "hermes_cli.main");
        // inherited flags preserved
        assert!(argv.contains(&"--tui".to_string()));
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"gpt-4".to_string()));
        // extra args appended last
        assert_eq!(&argv[argv.len() - 2..], &["--resume".to_string(), "abc".to_string()][..]);
    }

    #[test]
    fn build_relaunch_argv_resolves_absolute_executable() {
        // /bin/sh is virtually always an executable file on unix.
        #[cfg(unix)]
        {
            let bin = if Path::new("/bin/sh").exists() {
                "/bin/sh"
            } else {
                "/usr/bin/env"
            };
            let argv = build_relaunch_argv(
                &[],
                false,
                &[],
                bin,
                "/usr/bin/python3",
            );
            assert_eq!(argv[0], bin);
            assert_eq!(argv.len(), 1);
        }
    }

    #[test]
    fn build_relaunch_argv_no_preserve_skips_flags() {
        let original = vec!["--tui".to_string()];
        let argv = build_relaunch_argv(
            &["--resume".to_string()],
            false,
            &original,
            "-",
            "/usr/bin/python3",
        );
        assert!(!argv.contains(&"--tui".to_string()));
        assert!(argv.contains(&"--resume".to_string()));
    }
}
