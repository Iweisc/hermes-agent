//! Shared helpers for attaching Hermes to a local Chrome CDP port.
//!
//! Faithful native Rust port of `hermes_cli/browser_connect.py`.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use crate::mod_hermes_constants::get_hermes_home;

pub const DEFAULT_BROWSER_CDP_PORT: u16 = 9222;

/// `http://127.0.0.1:9222` — the default CDP URL.
pub fn default_browser_cdp_url() -> String {
    format!("http://127.0.0.1:{DEFAULT_BROWSER_CDP_PORT}")
}

const DARWIN_APPS: &[&str] = &[
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
];

/// Windows install path fragments, joined under each base directory.
const WINDOWS_INSTALL_PARTS: &[&[&str]] = &[
    &["Google", "Chrome", "Application", "chrome.exe"],
    &["Chromium", "Application", "chrome.exe"],
    &["Chromium", "Application", "chromium.exe"],
    &["BraveSoftware", "Brave-Browser", "Application", "brave.exe"],
    &["Microsoft", "Edge", "Application", "msedge.exe"],
];

const LINUX_BIN_NAMES: &[&str] = &[
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    "brave-browser",
    "microsoft-edge",
];

const WINDOWS_BIN_NAMES: &[&str] = &[
    "chrome.exe",
    "msedge.exe",
    "brave.exe",
    "chromium.exe",
    "chrome",
    "msedge",
    "brave",
    "chromium",
];

/// Normalise a path the way Python's `os.path.normcase(os.path.normpath(path))`
/// would for the purpose of de-duplication. On non-Windows platforms `normcase`
/// is a no-op (case-sensitive); on Windows it lowercases and swaps `/` for `\`.
fn normalize_for_seen(path: &str) -> String {
    let normpath = Path::new(path)
        .components()
        .collect::<PathBuf>()
        .to_string_lossy()
        .into_owned();
    if cfg!(windows) {
        normpath.to_lowercase().replace('/', "\\")
    } else {
        normpath
    }
}

/// Mirror of `shutil.which`: search `PATH` for an executable named `name`.
fn which(name: &str) -> Option<String> {
    // If the name contains a path separator, Python's which checks it directly.
    if name.contains(std::path::MAIN_SEPARATOR) || name.contains('/') {
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

#[cfg(unix)]
fn is_executable_file(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(p) {
        Ok(md) => md.is_file() && (md.permissions().mode() & 0o111 != 0),
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable_file(p: &Path) -> bool {
    p.is_file()
}

/// Return the list of Chrome-family executable paths usable for remote
/// debugging, for the given `system` (`"Darwin"`, `"Windows"`, or other →
/// treated as Linux). Order and de-duplication match the Python original;
/// only paths that currently exist as files are returned.
pub fn get_chrome_debug_candidates(system: &str) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut add = |path: Option<&str>| {
        let path = match path {
            Some(p) if !p.is_empty() => p,
            _ => return,
        };
        let normalized = normalize_for_seen(path);
        if seen.contains(&normalized) || !Path::new(path).is_file() {
            return;
        }
        candidates.push(path.to_string());
        seen.insert(normalized);
    };

    let add_install_paths = |add: &mut dyn FnMut(Option<&str>), bases: &[Option<String>]| {
        for base in bases.iter().flatten() {
            for parts in WINDOWS_INSTALL_PARTS {
                let mut joined = PathBuf::from(base);
                for part in *parts {
                    joined.push(part);
                }
                add(Some(joined.to_string_lossy().as_ref()));
            }
        }
    };

    if system == "Darwin" {
        for app in DARWIN_APPS {
            add(Some(app));
        }
        return candidates;
    }

    if system == "Windows" {
        for name in WINDOWS_BIN_NAMES {
            add(which(name).as_deref());
        }
        let bases = [
            std::env::var("ProgramFiles").ok(),
            std::env::var("ProgramFiles(x86)").ok(),
            std::env::var("LOCALAPPDATA").ok(),
        ];
        add_install_paths(&mut add, &bases);
        return candidates;
    }

    for name in LINUX_BIN_NAMES {
        add(which(name).as_deref());
    }
    let bases = [
        Some("/mnt/c/Program Files".to_string()),
        Some("/mnt/c/Program Files (x86)".to_string()),
    ];
    add_install_paths(&mut add, &bases);
    candidates
}

/// The dedicated user-data directory used for the debug Chrome profile.
pub fn chrome_debug_data_dir() -> String {
    get_hermes_home()
        .join("chrome-debug")
        .to_string_lossy()
        .into_owned()
}

fn chrome_debug_args(port: u16) -> Vec<String> {
    vec![
        format!("--remote-debugging-port={port}"),
        format!("--user-data-dir={}", chrome_debug_data_dir()),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
    ]
}

/// Quote a single argument the way POSIX `shlex.quote` would.
fn shlex_quote(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    // Safe set matches shlex._find_unsafe (anything not in this set is unsafe).
    let safe = arg.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    });
    if safe {
        return arg.to_string();
    }
    // Wrap in single quotes, escaping embedded single quotes.
    format!("'{}'", arg.replace('\'', "'\"'\"'"))
}

/// `shlex.join`: join arguments with spaces, quoting each.
fn shlex_join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| shlex_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `subprocess.list2cmdline`: Windows command-line quoting rules.
fn list2cmdline(argv: &[String]) -> String {
    let mut result = String::new();
    for (i, arg) in argv.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        let needquote = arg.is_empty() || arg.contains(' ') || arg.contains('\t');
        if needquote {
            result.push('"');
        }
        let mut bs_buf: Vec<char> = Vec::new();
        for c in arg.chars() {
            if c == '\\' {
                bs_buf.push(c);
            } else if c == '"' {
                // Double preceding backslashes, then escape the quote.
                for _ in 0..(bs_buf.len() * 2) {
                    result.push('\\');
                }
                bs_buf.clear();
                result.push_str("\\\"");
            } else {
                for bc in bs_buf.drain(..) {
                    result.push(bc);
                }
                result.push(c);
            }
        }
        // Append remaining backslashes.
        for bc in &bs_buf {
            result.push(*bc);
        }
        if needquote {
            for _ in 0..bs_buf.len() {
                result.push('\\');
            }
            result.push('"');
        }
    }
    result
}

/// Build a human-runnable command string to launch Chrome with remote
/// debugging, or `None` when no launch strategy is available.
pub fn manual_chrome_debug_command(port: u16, system: Option<&str>) -> Option<String> {
    let system = system
        .map(|s| s.to_string())
        .unwrap_or_else(platform_system);
    let candidates = get_chrome_debug_candidates(&system);

    if !candidates.is_empty() {
        let mut argv = vec![candidates[0].clone()];
        argv.extend(chrome_debug_args(port));
        return Some(if system == "Windows" {
            list2cmdline(&argv)
        } else {
            shlex_join(&argv)
        });
    }

    if system == "Darwin" {
        let data_dir = chrome_debug_data_dir();
        return Some(format!(
            "open -a \"Google Chrome\" --args --remote-debugging-port={port} \
             --user-data-dir=\"{data_dir}\" --no-first-run --no-default-browser-check"
        ));
    }

    None
}

/// Return the value `platform.system()` would yield for the current OS.
pub fn platform_system() -> String {
    if cfg!(target_os = "macos") {
        "Darwin".to_string()
    } else if cfg!(target_os = "windows") {
        "Windows".to_string()
    } else if cfg!(target_os = "linux") {
        "Linux".to_string()
    } else {
        std::env::consts::OS.to_string()
    }
}

/// Attempt to spawn a detached Chrome process with remote debugging enabled.
/// Returns `true` on a successful spawn, `false` if there is no candidate
/// browser or the spawn fails.
pub fn try_launch_chrome_debug(port: u16, system: Option<&str>) -> bool {
    let system = system
        .map(|s| s.to_string())
        .unwrap_or_else(platform_system);
    let candidates = get_chrome_debug_candidates(&system);
    if candidates.is_empty() {
        return false;
    }

    if std::fs::create_dir_all(chrome_debug_data_dir()).is_err() {
        // Python's makedirs(exist_ok=True) would raise on real failure, which is
        // not caught here (it sits outside the try). Mirror that by bailing.
        return false;
    }

    let mut cmd = std::process::Command::new(&candidates[0]);
    cmd.args(chrome_debug_args(port))
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    apply_detach(&mut cmd, &system);

    cmd.spawn().is_ok()
}

#[cfg(unix)]
fn apply_detach(cmd: &mut std::process::Command, _system: &str) {
    use std::os::unix::process::CommandExt;
    // Equivalent to subprocess start_new_session=True.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(windows)]
fn apply_detach(cmd: &mut std::process::Command, _system: &str) {
    use std::os::windows::process::CommandExt;
    // DETACHED_PROCESS (0x08) | CREATE_NEW_PROCESS_GROUP (0x200)
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(any(unix, windows)))]
fn apply_detach(_cmd: &mut std::process::Command, _system: &str) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_url_uses_default_port() {
        assert_eq!(default_browser_cdp_url(), "http://127.0.0.1:9222");
        assert_eq!(DEFAULT_BROWSER_CDP_PORT, 9222);
    }

    #[test]
    fn darwin_candidates_skip_missing_files() {
        // On CI these app bundles will not exist; the function must still run
        // and only return real files.
        let cands = get_chrome_debug_candidates("Darwin");
        for c in &cands {
            assert!(Path::new(c).is_file(), "returned non-file candidate: {c}");
        }
    }

    #[test]
    fn unknown_system_treated_as_linux() {
        // Should not panic and should only contain existing files.
        let cands = get_chrome_debug_candidates("Plan9");
        for c in &cands {
            assert!(Path::new(c).is_file());
        }
    }

    #[test]
    fn chrome_debug_args_shape() {
        let args = chrome_debug_args(9333);
        assert_eq!(args[0], "--remote-debugging-port=9333");
        assert!(args[1].starts_with("--user-data-dir="));
        assert!(args[1].ends_with("chrome-debug"));
        assert_eq!(args[2], "--no-first-run");
        assert_eq!(args[3], "--no-default-browser-check");
    }

    #[test]
    fn shlex_quote_basics() {
        assert_eq!(shlex_quote("simple"), "simple");
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("/usr/bin/chrome"), "/usr/bin/chrome");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn shlex_join_quotes_spaces() {
        let argv = vec![
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".to_string(),
            "--remote-debugging-port=9222".to_string(),
        ];
        let joined = shlex_join(&argv);
        assert!(joined.contains("'/Applications/Google Chrome.app"));
        assert!(joined.contains("--remote-debugging-port=9222"));
    }

    #[test]
    fn list2cmdline_quotes_spaces_and_escapes() {
        let argv = vec![
            "C:\\Program Files\\chrome.exe".to_string(),
            "--foo".to_string(),
        ];
        let line = list2cmdline(&argv);
        assert_eq!(line, "\"C:\\Program Files\\chrome.exe\" --foo");
    }

    #[test]
    fn list2cmdline_escapes_embedded_quote() {
        let argv = vec!["a\"b".to_string()];
        // matches CPython list2cmdline behaviour
        assert_eq!(list2cmdline(&argv), "a\\\"b");
    }

    #[test]
    fn manual_command_darwin_fallback_when_no_candidates() {
        // On Linux/CI the Darwin app bundles don't exist, so this exercises the
        // `open -a` fallback branch deterministically.
        let cmd = manual_chrome_debug_command(9222, Some("Darwin"));
        if get_chrome_debug_candidates("Darwin").is_empty() {
            let cmd = cmd.expect("darwin fallback should produce a command");
            assert!(cmd.starts_with("open -a \"Google Chrome\" --args"));
            assert!(cmd.contains("--remote-debugging-port=9222"));
            assert!(cmd.contains("--no-default-browser-check"));
        }
    }

    #[test]
    fn manual_command_none_for_linux_without_browser() {
        let cands = get_chrome_debug_candidates("Linux");
        let cmd = manual_chrome_debug_command(9222, Some("Linux"));
        if cands.is_empty() {
            assert!(cmd.is_none());
        } else {
            assert!(cmd.is_some());
        }
    }

    #[test]
    fn platform_system_nonempty() {
        assert!(!platform_system().is_empty());
    }

    #[test]
    fn launch_returns_false_without_candidates() {
        // Force a system with no real candidates by checking the empty case.
        if get_chrome_debug_candidates("Linux").is_empty() {
            assert!(!try_launch_chrome_debug(9222, Some("Linux")));
        }
    }
}
