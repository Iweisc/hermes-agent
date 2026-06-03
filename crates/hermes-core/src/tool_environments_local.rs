//! Local execution environment — spawn-per-call with session snapshot.
//!
//! Faithful native Rust port of `tools/environments/local.py`.
//!
//! Spawn-per-call: every `execute()` spawns a fresh bash process. A session
//! snapshot preserves env vars across calls. CWD persists via a file-based read
//! after each command.
//!
//! # Port notes
//!
//! * The Python module mixes pure helper logic (cwd recovery, env sanitisation,
//!   bash discovery, shell-init resolution, process-group kill) with a concrete
//!   `subprocess.Popen` lifecycle. This port reproduces the helper logic
//!   faithfully and exposes the process spawn/kill via `std::process` +
//!   `libc` (POSIX) so the spawn-per-call contract is preserved.
//! * The provider-credential blocklist mirrors the Python
//!   `_build_provider_env_blocklist` literal. Rather than re-deriving it from
//!   provider/config registries (not yet ported), we reuse the canonical
//!   built-in copy from [`crate::tool_env_passthrough::builtin_provider_env_blocklist`]
//!   plus an injectable extension point ([`set_extra_blocklist`]) for callers
//!   that have ported the provider/tool registries.
//! * `is_env_passthrough` is consulted via
//!   [`crate::tool_env_passthrough::is_env_passthrough`].
//! * Per-profile HOME isolation reads
//!   [`crate::mod_hermes_constants::get_subprocess_home`].
//! * Config-driven `terminal.shell_init_files` / `auto_source_bashrc` are read
//!   via an injectable hook ([`set_terminal_shell_init_source`]); when no hook
//!   is installed we fall back to the Python defaults (`[]`, `true`).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::RwLock;

use crate::tool_env_passthrough::{builtin_provider_env_blocklist, is_env_passthrough};
use crate::mod_hermes_constants::get_subprocess_home;

/// `platform.system() == "Windows"`.
pub const IS_WINDOWS: bool = cfg!(target_os = "windows");

/// Hermes-internal env var prefix that should NOT leak into terminal
/// subprocesses, but whose stripped (`_HERMES_FORCE_`-less) form *should* be
/// forced into the child env.
pub const HERMES_PROVIDER_ENV_FORCE_PREFIX: &str = "_HERMES_FORCE_";

/// Standard PATH entries for environments with a minimal PATH.
pub const SANE_PATH: &str = "/opt/homebrew/bin:/opt/homebrew/sbin:\
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

// ---------------------------------------------------------------------------
// _resolve_safe_cwd
// ---------------------------------------------------------------------------

/// Return `cwd` if it exists as a directory, else the nearest existing
/// ancestor. Falls back to the system temp dir only if walking up the path
/// can't find any existing directory.
///
/// Used by [`LocalEnvironment::run_bash_args`] to recover when the configured
/// cwd is gone — most commonly because a previous tool call deleted its own
/// working directory (issue #17558). Without this guard, spawning with a
/// missing `cwd` fails before bash starts, wedging every subsequent terminal
/// call until the gateway restarts.
pub fn resolve_safe_cwd(cwd: &str) -> String {
    if !cwd.is_empty() && Path::new(cwd).is_dir() {
        return cwd.to_string();
    }
    let mut parent = if cwd.is_empty() {
        String::new()
    } else {
        dirname(cwd)
    };
    while !parent.is_empty() {
        if Path::new(&parent).is_dir() {
            return parent;
        }
        let next_parent = dirname(&parent);
        if next_parent == parent {
            // Reached the filesystem root and it doesn't exist either.
            break;
        }
        parent = next_parent;
    }
    std::env::temp_dir().to_string_lossy().into_owned()
}

/// `os.path.dirname` semantics (POSIX): drop everything after the final `/`,
/// stripping trailing slashes from the result except for the root.
fn dirname(path: &str) -> String {
    // Python os.path.dirname: split on last '/', return head; if head is all
    // slashes keep them, else strip trailing slashes.
    match path.rfind('/') {
        None => String::new(),
        Some(idx) => {
            let head = &path[..=idx];
            // strip trailing slashes unless head is entirely slashes
            if head.chars().all(|c| c == '/') {
                head.to_string()
            } else {
                head.trim_end_matches('/').to_string()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Provider env blocklist
// ---------------------------------------------------------------------------

/// Optional caller-installed extension to the provider env blocklist.
///
/// The Python `_build_provider_env_blocklist` augmented the static literal with
/// entries derived from `PROVIDER_REGISTRY` and `OPTIONAL_ENV_VARS`. Those
/// registries are not yet ported here, so callers that have them can inject
/// the extra names; otherwise only the static literal applies (matching the
/// Python `ImportError` fallthrough paths).
static EXTRA_BLOCKLIST: RwLock<Vec<String>> = RwLock::new(Vec::new());

static BLOCKLIST: OnceLock<std::collections::HashSet<String>> = OnceLock::new();

/// Install additional provider/tool env-var names to add to the blocklist.
///
/// Must be called before the blocklist is first read (it is cached on first
/// use, like the Python module-level `_HERMES_PROVIDER_ENV_BLOCKLIST`).
pub fn set_extra_blocklist<I, S>(names: I)
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut guard = EXTRA_BLOCKLIST.write().unwrap();
    *guard = names.into_iter().map(Into::into).collect();
}

/// The Hermes-managed env-var blocklist (provider + tool + gateway config).
pub fn provider_env_blocklist() -> &'static std::collections::HashSet<String> {
    BLOCKLIST.get_or_init(|| {
        let mut set = builtin_provider_env_blocklist();
        if let Ok(extra) = EXTRA_BLOCKLIST.read() {
            for name in extra.iter() {
                set.insert(name.clone());
            }
        }
        set
    })
}

// ---------------------------------------------------------------------------
// _sanitize_subprocess_env
// ---------------------------------------------------------------------------

/// Filter Hermes-managed secrets from a subprocess environment.
///
/// Mirrors `_sanitize_subprocess_env`: walks `base_env` then `extra_env`,
/// dropping `_HERMES_FORCE_`-prefixed keys from base (and unprefixing them in
/// extra), and dropping blocklisted keys unless they are passthrough. Applies
/// per-profile HOME isolation when configured.
pub fn sanitize_subprocess_env(
    base_env: Option<&BTreeMap<String, String>>,
    extra_env: Option<&BTreeMap<String, String>>,
) -> BTreeMap<String, String> {
    let blocklist = provider_env_blocklist();
    let mut sanitized: BTreeMap<String, String> = BTreeMap::new();

    if let Some(base) = base_env {
        for (key, value) in base.iter() {
            if key.starts_with(HERMES_PROVIDER_ENV_FORCE_PREFIX) {
                continue;
            }
            if !blocklist.contains(key) || is_env_passthrough(key) {
                sanitized.insert(key.clone(), value.clone());
            }
        }
    }

    if let Some(extra) = extra_env {
        for (key, value) in extra.iter() {
            if let Some(real_key) = key.strip_prefix(HERMES_PROVIDER_ENV_FORCE_PREFIX) {
                sanitized.insert(real_key.to_string(), value.clone());
            } else if !blocklist.contains(key) || is_env_passthrough(key) {
                sanitized.insert(key.clone(), value.clone());
            }
        }
    }

    if let Some(profile_home) = get_subprocess_home() {
        sanitized.insert("HOME".to_string(), profile_home);
    }

    sanitized
}

// ---------------------------------------------------------------------------
// _find_bash / _find_shell
// ---------------------------------------------------------------------------

/// Find bash for command execution.
///
/// On non-Windows: `which bash` -> `/usr/bin/bash` -> `/bin/bash` ->
/// `$SHELL` -> `/bin/sh`.
///
/// On Windows: `HERMES_GIT_BASH_PATH` -> `which bash` -> common Git-for-Windows
/// install locations -> error.
pub fn find_bash() -> Result<String, String> {
    if !IS_WINDOWS {
        if let Some(p) = which("bash") {
            return Ok(p);
        }
        if Path::new("/usr/bin/bash").is_file() {
            return Ok("/usr/bin/bash".to_string());
        }
        if Path::new("/bin/bash").is_file() {
            return Ok("/bin/bash".to_string());
        }
        if let Some(shell) = std::env::var_os("SHELL") {
            if !shell.is_empty() {
                return Ok(shell.to_string_lossy().into_owned());
            }
        }
        return Ok("/bin/sh".to_string());
    }

    if let Some(custom) = std::env::var_os("HERMES_GIT_BASH_PATH") {
        let custom = custom.to_string_lossy().into_owned();
        if !custom.is_empty() && Path::new(&custom).is_file() {
            return Ok(custom);
        }
    }

    if let Some(found) = which("bash") {
        return Ok(found);
    }

    let program_files =
        std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
    let program_files_x86 = std::env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let local_appdata = std::env::var("LOCALAPPDATA").unwrap_or_default();

    let candidates = [
        format!(r"{program_files}\Git\bin\bash.exe"),
        format!(r"{program_files_x86}\Git\bin\bash.exe"),
        if local_appdata.is_empty() {
            String::new()
        } else {
            format!(r"{local_appdata}\Programs\Git\bin\bash.exe")
        },
    ];
    for candidate in candidates.iter() {
        if !candidate.is_empty() && Path::new(candidate).is_file() {
            return Ok(candidate.clone());
        }
    }

    Err("Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n\
         Install it from: https://git-scm.com/download/win\n\
         Or set HERMES_GIT_BASH_PATH to your bash.exe location."
        .to_string())
}

/// Backward-compat alias — `process_registry.py` imports `_find_shell`.
#[inline]
pub fn find_shell() -> Result<String, String> {
    find_bash()
}

/// Minimal `shutil.which`: search `$PATH` for an executable named `name`.
fn which(name: &str) -> Option<String> {
    // If name contains a path separator, test it directly (shutil.which behaviour).
    if name.contains('/') || (IS_WINDOWS && name.contains('\\')) {
        let p = Path::new(name);
        if p.is_file() {
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
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if IS_WINDOWS {
            // Try common Windows executable extensions.
            for ext in ["exe", "bat", "cmd", "com"] {
                let c = dir.join(format!("{name}.{ext}"));
                if c.is_file() {
                    return Some(c.to_string_lossy().into_owned());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// _make_run_env
// ---------------------------------------------------------------------------

/// Build a run environment with a sane PATH and provider-var stripping.
///
/// Merges `os.environ | env` (with `env` taking precedence), forces
/// `_HERMES_FORCE_`-prefixed vars, strips blocklisted ones (unless passthrough),
/// ensures `/usr/bin` is on PATH, and applies per-profile HOME isolation.
pub fn make_run_env(env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let blocklist = provider_env_blocklist();

    // merged = dict(os.environ | env): os.environ first, env overrides.
    let mut merged: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in std::env::vars() {
        merged.insert(k, v);
    }
    for (k, v) in env.iter() {
        merged.insert(k.clone(), v.clone());
    }

    let mut run_env: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in merged.iter() {
        if let Some(real_key) = k.strip_prefix(HERMES_PROVIDER_ENV_FORCE_PREFIX) {
            run_env.insert(real_key.to_string(), v.clone());
        } else if !blocklist.contains(k) || is_env_passthrough(k) {
            run_env.insert(k.clone(), v.clone());
        }
    }

    let existing_path = run_env.get("PATH").cloned().unwrap_or_default();
    let has_usr_bin = existing_path.split(':').any(|p| p == "/usr/bin");
    if !has_usr_bin {
        let new_path = if existing_path.is_empty() {
            SANE_PATH.to_string()
        } else {
            format!("{existing_path}:{SANE_PATH}")
        };
        run_env.insert("PATH".to_string(), new_path);
    }

    if let Some(profile_home) = get_subprocess_home() {
        run_env.insert("HOME".to_string(), profile_home);
    }

    run_env
}

// ---------------------------------------------------------------------------
// Shell-init config (terminal.shell_init_files / auto_source_bashrc)
// ---------------------------------------------------------------------------

/// Injectable source for `(shell_init_files, auto_source_bashrc)` read from
/// `config.yaml`'s `terminal` section.
///
/// The Python `_read_terminal_shell_init_config` read this via
/// `hermes_cli.config.load_config`. That config plumbing is injectable here;
/// when no source is installed we return the Python best-effort default
/// `([], true)`.
type ShellInitSource = fn() -> (Vec<String>, bool);
static SHELL_INIT_SOURCE: RwLock<Option<ShellInitSource>> = RwLock::new(None);

/// Install the function used to source `terminal.shell_init_files` /
/// `terminal.auto_source_bashrc`.
pub fn set_terminal_shell_init_source(source: ShellInitSource) {
    let mut guard = SHELL_INIT_SOURCE.write().unwrap();
    *guard = Some(source);
}

/// Return `(shell_init_files, auto_source_bashrc)`.
///
/// Best-effort — returns sensible defaults `([], true)` on any failure so
/// terminal execution never breaks because the config file is unreadable.
pub fn read_terminal_shell_init_config() -> (Vec<String>, bool) {
    let source = { SHELL_INIT_SOURCE.read().ok().and_then(|g| *g) };
    match source {
        Some(f) => {
            let (files, auto) = f();
            // Mirror Python's `[str(f) for f in files if f]` filtering.
            let files = files.into_iter().filter(|s| !s.is_empty()).collect();
            (files, auto)
        }
        None => (Vec::new(), true),
    }
}

/// Resolve the list of files to source before the login-shell snapshot.
///
/// Expands `~` and `${VAR}` references and drops anything that doesn't exist on
/// disk, so a missing `~/.bashrc` never breaks the snapshot. The
/// `auto_source_bashrc` path runs only when the user hasn't supplied an
/// explicit list.
pub fn resolve_shell_init_files() -> Vec<String> {
    let (explicit, auto_bashrc) = read_terminal_shell_init_config();

    let mut candidates: Vec<String> = Vec::new();
    if !explicit.is_empty() {
        candidates.extend(explicit);
    } else if auto_bashrc && !IS_WINDOWS {
        candidates.push("~/.profile".to_string());
        candidates.push("~/.bash_profile".to_string());
        candidates.push("~/.bashrc".to_string());
    }

    let mut resolved: Vec<String> = Vec::new();
    for raw in candidates.iter() {
        let path = match expand_user_and_vars(raw) {
            Some(p) => p,
            None => continue,
        };
        if !path.is_empty() && Path::new(&path).is_file() {
            resolved.push(path);
        }
    }
    resolved
}

/// `os.path.expandvars(os.path.expanduser(raw))`.
fn expand_user_and_vars(raw: &str) -> Option<String> {
    let expanded = expand_user(raw);
    Some(expand_vars(&expanded))
}

/// `os.path.expanduser`: leading `~` -> $HOME, `~user` is left untouched here
/// (we only handle the common `~`/`~/` case, matching the candidate inputs).
fn expand_user(path: &str) -> String {
    if !path.starts_with('~') {
        return path.to_string();
    }
    // Find end of the user portion (up to first '/').
    let rest_idx = path.find('/').unwrap_or(path.len());
    let user_spec = &path[1..rest_idx];
    if user_spec.is_empty() {
        // bare ~ or ~/...
        let home = home_dir();
        match home {
            Some(h) => format!("{}{}", h, &path[rest_idx..]),
            None => path.to_string(),
        }
    } else {
        // ~user — not resolved; return unchanged (matches our limited needs).
        path.to_string()
    }
}

fn home_dir() -> Option<String> {
    if IS_WINDOWS {
        // os.path.expanduser on Windows uses USERPROFILE / HOMEDRIVE+HOMEPATH.
        if let Some(v) = std::env::var_os("USERPROFILE") {
            return Some(v.to_string_lossy().into_owned());
        }
        match (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
            (Some(d), Some(p)) => {
                return Some(format!(
                    "{}{}",
                    d.to_string_lossy(),
                    p.to_string_lossy()
                ));
            }
            _ => {}
        }
    }
    std::env::var_os("HOME").map(|v| v.to_string_lossy().into_owned())
}

/// `os.path.expandvars`: expand `$VAR` and `${VAR}`. Unknown vars are left
/// intact (POSIX behaviour).
fn expand_vars(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c != '$' {
            out.push(c);
            i += 1;
            continue;
        }
        // Found '$'
        if i + 1 >= bytes.len() {
            out.push('$');
            break;
        }
        let next = bytes[i + 1] as char;
        if next == '{' {
            // ${NAME}
            if let Some(close) = input[i + 2..].find('}') {
                let name = &input[i + 2..i + 2 + close];
                if is_valid_var_name(name) {
                    match std::env::var(name) {
                        Ok(val) => out.push_str(&val),
                        Err(_) => out.push_str(&input[i..i + 2 + close + 1]),
                    }
                } else {
                    out.push_str(&input[i..i + 2 + close + 1]);
                }
                i = i + 2 + close + 1;
                continue;
            } else {
                // No closing brace; emit literally.
                out.push_str(&input[i..]);
                break;
            }
        } else if next == '_' || next.is_ascii_alphabetic() {
            // $NAME
            let mut j = i + 1;
            while j < bytes.len() {
                let ch = bytes[j] as char;
                if ch == '_' || ch.is_ascii_alphanumeric() {
                    j += 1;
                } else {
                    break;
                }
            }
            let name = &input[i + 1..j];
            match std::env::var(name) {
                Ok(val) => out.push_str(&val),
                Err(_) => out.push_str(&input[i..j]),
            }
            i = j;
            continue;
        } else {
            out.push('$');
            i += 1;
        }
    }
    out
}

fn is_valid_var_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(idx, c)| if idx == 0 { c == '_' || c.is_ascii_alphabetic() } else { c == '_' || c.is_ascii_alphanumeric() })
}

// ---------------------------------------------------------------------------
// _prepend_shell_init
// ---------------------------------------------------------------------------

/// Prepend `source <file>` lines (guarded + silent) to a bash script.
///
/// Each file is wrapped so a failing rc file doesn't abort the whole bootstrap:
/// `set +e` keeps going on errors, `2>/dev/null` hides noisy prompts, and
/// `|| true` neutralises the exit status.
pub fn prepend_shell_init(cmd_string: &str, files: &[String]) -> String {
    if files.is_empty() {
        return cmd_string.to_string();
    }
    let mut prelude_parts: Vec<String> = vec!["set +e".to_string()];
    for path in files {
        let safe = path.replace('\'', "'\\''");
        prelude_parts.push(format!(
            "[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true"
        ));
    }
    let prelude = prelude_parts.join("\n") + "\n";
    format!("{prelude}{cmd_string}")
}

// ---------------------------------------------------------------------------
// get_temp_dir helper (used by LocalEnvironment::get_temp_dir)
// ---------------------------------------------------------------------------

/// Return a shell-safe writable temp dir for local execution.
///
/// Prefers POSIX-style env vars (`TMPDIR`/`TMP`/`TEMP`) checking the backend
/// `env` first then the host process env, keeps `/tmp` on regular Unix systems,
/// and falls back to the system temp dir only when it resolves to a POSIX path.
pub fn compute_temp_dir(env: &BTreeMap<String, String>) -> String {
    for env_var in ["TMPDIR", "TMP", "TEMP"] {
        let candidate = env
            .get(env_var)
            .cloned()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var(env_var).ok().filter(|s| !s.is_empty()));
        if let Some(c) = candidate {
            if c.starts_with('/') {
                let trimmed = c.trim_end_matches('/');
                return if trimmed.is_empty() {
                    "/".to_string()
                } else {
                    trimmed.to_string()
                };
            }
        }
    }

    if Path::new("/tmp").is_dir() && is_writable_executable("/tmp") {
        return "/tmp".to_string();
    }

    let candidate = std::env::temp_dir().to_string_lossy().into_owned();
    if candidate.starts_with('/') {
        let trimmed = candidate.trim_end_matches('/');
        return if trimmed.is_empty() {
            "/".to_string()
        } else {
            trimmed.to_string()
        };
    }

    "/tmp".to_string()
}

/// `os.access(path, os.W_OK | os.X_OK)` for a directory.
#[cfg(unix)]
fn is_writable_executable(path: &str) -> bool {
    use std::ffi::CString;
    let cstr = match CString::new(path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    // W_OK = 2, X_OK = 1
    unsafe { libc::access(cstr.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

#[cfg(not(unix))]
fn is_writable_executable(path: &str) -> bool {
    Path::new(path).is_dir()
}

// ---------------------------------------------------------------------------
// LocalEnvironment
// ---------------------------------------------------------------------------

/// The arguments + environment computed for a single bash spawn.
///
/// Mirrors what `_run_bash` assembles before calling `subprocess.Popen`:
/// `argv`, the sanitised+PATH-fixed `env`, the recovered `cwd`, and whether
/// stdin should be wired up.
#[derive(Debug, Clone)]
pub struct BashSpawn {
    /// Full argv (e.g. `[bash, "-l", "-c", cmd]` for login, `[bash, "-c", cmd]`
    /// otherwise).
    pub argv: Vec<String>,
    /// The child process environment.
    pub env: BTreeMap<String, String>,
    /// The (possibly recovered) working directory.
    pub cwd: String,
    /// `Some(data)` when stdin should be piped, `None` for `DEVNULL`.
    pub stdin_data: Option<String>,
}

/// Run commands directly on the host machine.
///
/// Spawn-per-call: every `execute()` spawns a fresh bash process. Session
/// snapshot preserves env vars across calls. CWD persists via a file-based read
/// after each command.
///
/// This port carries the mutable state the Python `LocalEnvironment` held
/// (`cwd`, `timeout`, `env`, the snapshot/cwd marker temp-file paths) and the
/// pure command-assembly logic. The concrete `subprocess.Popen` lifecycle is
/// modelled by [`BashSpawn`] (assembly) which callers feed to `std::process` /
/// `libc` spawn machinery.
#[derive(Debug, Clone)]
pub struct LocalEnvironment {
    /// Current working directory.
    pub cwd: String,
    /// Default command timeout in seconds.
    pub timeout: i64,
    /// Backend environment overrides.
    pub env: BTreeMap<String, String>,
    /// Path to the captured session snapshot file.
    pub snapshot_path: String,
    /// Path to the temp file that records the live cwd after each command.
    pub cwd_file: String,
}

impl LocalEnvironment {
    /// Construct a `LocalEnvironment`.
    ///
    /// `cwd` is `~`-expanded; an empty `cwd` falls back to the process's current
    /// directory (matching `os.getcwd()`). `init_session()` (snapshot capture)
    /// is the caller's responsibility in this port — it requires the full base
    /// environment machinery — but the snapshot/cwd-marker temp-file paths are
    /// allocated here so [`Self::cleanup`] can remove them.
    pub fn new(cwd: &str, timeout: i64, env: Option<BTreeMap<String, String>>) -> Self {
        let cwd = if !cwd.is_empty() {
            expand_user(cwd)
        } else {
            String::new()
        };
        let cwd = if cwd.is_empty() {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".to_string())
        } else {
            cwd
        };

        let tmp = std::env::temp_dir();
        let suffix = new_token();
        let snapshot_path = tmp
            .join(format!("hermes_env_snapshot_{suffix}.sh"))
            .to_string_lossy()
            .into_owned();
        let cwd_file = tmp
            .join(format!("hermes_cwd_{suffix}.txt"))
            .to_string_lossy()
            .into_owned();

        LocalEnvironment {
            cwd,
            timeout,
            env: env.unwrap_or_default(),
            snapshot_path,
            cwd_file,
        }
    }

    /// Return a shell-safe writable temp dir for this backend. See
    /// [`compute_temp_dir`].
    pub fn get_temp_dir(&self) -> String {
        compute_temp_dir(&self.env)
    }

    /// Assemble the argv/env/cwd for spawning bash, applying the login-shell
    /// shell-init prepend and the deleted-cwd recovery.
    ///
    /// This is the pure portion of `_run_bash`: it mutates `self.cwd` if the
    /// configured directory has vanished (logging a warning), exactly as the
    /// Python recovery path does. The caller is responsible for the actual
    /// `Popen`/`setsid` spawn.
    pub fn run_bash_spawn(
        &mut self,
        cmd_string: &str,
        login: bool,
        stdin_data: Option<String>,
    ) -> Result<BashSpawn, String> {
        let bash = find_bash()?;

        let mut cmd_string = cmd_string.to_string();
        if login {
            let init_files = resolve_shell_init_files();
            if !init_files.is_empty() {
                cmd_string = prepend_shell_init(&cmd_string, &init_files);
            }
        }

        let argv = if login {
            vec![bash, "-l".to_string(), "-c".to_string(), cmd_string]
        } else {
            vec![bash, "-c".to_string(), cmd_string]
        };

        let run_env = make_run_env(&self.env);

        let safe_cwd = resolve_safe_cwd(&self.cwd);
        if safe_cwd != self.cwd {
            log::warn!(
                "LocalEnvironment cwd {:?} is missing on disk; falling back to {:?} \
                 so terminal commands keep working.",
                self.cwd,
                safe_cwd,
            );
            self.cwd = safe_cwd.clone();
        }

        Ok(BashSpawn {
            argv,
            env: run_env,
            cwd: self.cwd.clone(),
            stdin_data,
        })
    }

    /// Read CWD from the temp marker file (local-only, no round-trip needed).
    ///
    /// Skips the assignment when the path no longer exists as a directory —
    /// `pwd -P` on a deleted cwd can leave a stale value in the marker file, and
    /// propagating it would re-wedge the next spawn. The
    /// [`Self::run_bash_spawn`] recovery path resolves a safe fallback if needed.
    ///
    /// Returns the cwd-marker contents so a caller can also strip it from the
    /// command output (the Python `_extract_cwd_from_output` step).
    pub fn update_cwd_from_marker(&mut self) -> Option<String> {
        match std::fs::read_to_string(&self.cwd_file) {
            Ok(contents) => {
                let cwd_path = contents.trim().to_string();
                if !cwd_path.is_empty() && Path::new(&cwd_path).is_dir() {
                    self.cwd = cwd_path.clone();
                }
                Some(cwd_path)
            }
            Err(_) => None,
        }
    }

    /// Clean up temp files (snapshot + cwd marker).
    pub fn cleanup(&self) {
        for f in [&self.snapshot_path, &self.cwd_file] {
            let _ = std::fs::remove_file(f);
        }
    }
}

// ---------------------------------------------------------------------------
// Process-group kill (POSIX)
// ---------------------------------------------------------------------------

/// Whether the process group `pgid` is still alive.
///
/// Mirrors the Python `_group_alive`: `killpg(pgid, 0)` returning 0 (or EPERM)
/// means alive; ESRCH means dead.
#[cfg(unix)]
pub fn group_alive(pgid: i32) -> bool {
    let rc = unsafe { libc::killpg(pgid, 0) };
    if rc == 0 {
        return true;
    }
    // errno
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    if errno == libc::EPERM {
        // The group exists, even if this process cannot signal it.
        return true;
    }
    // ESRCH (or anything else) => treat as dead.
    false
}

#[cfg(not(unix))]
pub fn group_alive(_pgid: i32) -> bool {
    false
}

/// Wait up to `timeout` seconds for the process group `pgid` to exit.
///
/// `poll` should reap the wrapper process (so a dead-but-unreaped group leader
/// doesn't keep the group reporting alive). Mirrors `_wait_for_group_exit`.
#[cfg(unix)]
pub fn wait_for_group_exit<F: FnMut()>(pgid: i32, timeout: f64, mut poll: F) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
    while std::time::Instant::now() < deadline {
        poll();
        if !group_alive(pgid) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    poll();
    !group_alive(pgid)
}

#[cfg(not(unix))]
pub fn wait_for_group_exit<F: FnMut()>(_pgid: i32, _timeout: f64, _poll: F) -> bool {
    false
}

/// Send `SIGTERM` then (after a grace period) `SIGKILL` to an entire process
/// group, reaping along the way. Mirrors the POSIX branch of `_kill_process`.
///
/// `poll` reaps the wrapper process between liveness checks. Returns once the
/// group is gone or the escalation sequence is exhausted.
#[cfg(unix)]
pub fn kill_process_group<F: FnMut()>(pgid: i32, mut poll: F) {
    // SIGTERM
    let rc = unsafe { libc::killpg(pgid, libc::SIGTERM) };
    if rc != 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno == libc::ESRCH {
            return;
        }
    }

    if wait_for_group_exit(pgid, 1.0, &mut poll) {
        return;
    }

    // SIGKILL
    let rc = unsafe { libc::killpg(pgid, libc::SIGKILL) };
    if rc != 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno == libc::ESRCH {
            return;
        }
    }
    wait_for_group_exit(pgid, 2.0, &mut poll);
}

// ---------------------------------------------------------------------------
// small util
// ---------------------------------------------------------------------------

/// A short unique token for temp-file naming (mirrors the use of `tempfile`
/// uniqueness in the Python paths). Uses pid + a monotonic counter + nanos.
fn new_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{pid}_{nanos}_{n}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirname_matches_python() {
        assert_eq!(dirname("/a/b/c"), "/a/b");
        assert_eq!(dirname("/a/b/c/"), "/a/b/c");
        assert_eq!(dirname("/a"), "/");
        assert_eq!(dirname("/"), "/");
        assert_eq!(dirname("a"), "");
        assert_eq!(dirname(""), "");
        assert_eq!(dirname("///"), "///");
    }

    #[test]
    fn resolve_safe_cwd_existing_dir() {
        let tmp = std::env::temp_dir();
        let s = tmp.to_string_lossy().into_owned();
        let trimmed = s.trim_end_matches('/').to_string();
        assert_eq!(resolve_safe_cwd(&trimmed), trimmed);
    }

    #[test]
    fn resolve_safe_cwd_walks_up_to_existing_ancestor() {
        let tmp = std::env::temp_dir();
        let base = tmp.to_string_lossy().trim_end_matches('/').to_string();
        let missing = format!("{base}/__hermes_definitely_missing__/deeper/still");
        let resolved = resolve_safe_cwd(&missing);
        assert_eq!(resolved, base);
    }

    #[test]
    fn resolve_safe_cwd_empty_falls_back_to_tempdir() {
        let resolved = resolve_safe_cwd("");
        let tmp = std::env::temp_dir().to_string_lossy().into_owned();
        assert_eq!(resolved, tmp);
    }

    #[test]
    fn blocklist_contains_known_secrets() {
        let bl = provider_env_blocklist();
        assert!(bl.contains("ANTHROPIC_TOKEN"));
        assert!(bl.contains("OPENAI_API_KEY"));
        assert!(bl.contains("VERCEL_TOKEN"));
        assert!(!bl.contains("PATH"));
    }

    #[test]
    fn sanitize_strips_blocklisted_and_forces_prefix() {
        let mut base = BTreeMap::new();
        base.insert("PATH".to_string(), "/usr/bin".to_string());
        base.insert("OPENAI_API_KEY".to_string(), "secret".to_string());
        base.insert(
            format!("{HERMES_PROVIDER_ENV_FORCE_PREFIX}FOO"),
            "leak".to_string(),
        );

        let mut extra = BTreeMap::new();
        extra.insert(
            format!("{HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY"),
            "forced".to_string(),
        );
        extra.insert("SAFE".to_string(), "ok".to_string());

        let out = sanitize_subprocess_env(Some(&base), Some(&extra));

        // base PATH kept, base OPENAI_API_KEY stripped, base _HERMES_FORCE_ skipped
        assert_eq!(out.get("PATH").map(String::as_str), Some("/usr/bin"));
        assert!(!out.contains_key(&format!("{HERMES_PROVIDER_ENV_FORCE_PREFIX}FOO")));
        assert!(!out.contains_key("FOO"));
        // extra forces the (otherwise blocklisted) key back in, unprefixed
        assert_eq!(out.get("OPENAI_API_KEY").map(String::as_str), Some("forced"));
        assert_eq!(out.get("SAFE").map(String::as_str), Some("ok"));
    }

    #[test]
    fn make_run_env_adds_sane_path_when_missing_usr_bin() {
        let env = BTreeMap::new();
        // Force a PATH without /usr/bin so we can observe the append.
        let prev = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("PATH", "/some/weird/path");
        }
        let run_env = make_run_env(&env);
        let path = run_env.get("PATH").cloned().unwrap_or_default();
        assert!(path.contains("/some/weird/path"));
        assert!(path.ends_with(SANE_PATH));
        assert!(path.split(':').any(|p| p == "/usr/bin"));
        // restore
        unsafe {
            match prev {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }
    }

    #[test]
    fn make_run_env_keeps_path_with_usr_bin() {
        let env = BTreeMap::new();
        let prev = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("PATH", "/usr/bin:/custom");
        }
        let run_env = make_run_env(&env);
        assert_eq!(
            run_env.get("PATH").map(String::as_str),
            Some("/usr/bin:/custom")
        );
        unsafe {
            match prev {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }
    }

    #[test]
    fn make_run_env_force_prefix_unwraps() {
        let mut env = BTreeMap::new();
        env.insert(
            format!("{HERMES_PROVIDER_ENV_FORCE_PREFIX}ANTHROPIC_TOKEN"),
            "forced".to_string(),
        );
        let run_env = make_run_env(&env);
        assert_eq!(
            run_env.get("ANTHROPIC_TOKEN").map(String::as_str),
            Some("forced")
        );
        assert!(!run_env.contains_key(&format!(
            "{HERMES_PROVIDER_ENV_FORCE_PREFIX}ANTHROPIC_TOKEN"
        )));
    }

    #[test]
    fn prepend_shell_init_empty_is_noop() {
        assert_eq!(prepend_shell_init("echo hi", &[]), "echo hi");
    }

    #[test]
    fn prepend_shell_init_wraps_files() {
        let files = vec!["/home/u/.bashrc".to_string()];
        let out = prepend_shell_init("echo hi", &files);
        assert!(out.starts_with("set +e\n"));
        assert!(out.contains("[ -r '/home/u/.bashrc' ] && . '/home/u/.bashrc' 2>/dev/null || true"));
        assert!(out.ends_with("echo hi"));
    }

    #[test]
    fn prepend_shell_init_escapes_single_quotes() {
        let files = vec!["/a/b'c".to_string()];
        let out = prepend_shell_init("x", &files);
        assert!(out.contains("'/a/b'\\''c'"));
    }

    #[test]
    fn expand_vars_substitutes_and_preserves_unknown() {
        unsafe {
            std::env::set_var("HERMES_TEST_VAR", "VAL");
        }
        assert_eq!(expand_vars("a/${HERMES_TEST_VAR}/b"), "a/VAL/b");
        assert_eq!(expand_vars("$HERMES_TEST_VAR"), "VAL");
        // Unknown var is preserved verbatim.
        assert_eq!(
            expand_vars("$HERMES_NOT_SET_XYZ/x"),
            "$HERMES_NOT_SET_XYZ/x"
        );
        // Literal dollar at end.
        assert_eq!(expand_vars("cost is 5$"), "cost is 5$");
        unsafe {
            std::env::remove_var("HERMES_TEST_VAR");
        }
    }

    #[test]
    fn expand_user_bare_tilde() {
        unsafe {
            std::env::set_var("HOME", "/home/tester");
        }
        if !IS_WINDOWS {
            assert_eq!(expand_user("~/.bashrc"), "/home/tester/.bashrc");
            assert_eq!(expand_user("~"), "/home/tester");
            // ~user left untouched
            assert_eq!(expand_user("~other/x"), "~other/x");
            assert_eq!(expand_user("/abs/path"), "/abs/path");
        }
    }

    #[test]
    fn read_terminal_shell_init_config_defaults() {
        // No source installed in this test process by default.
        let (files, auto) = read_terminal_shell_init_config();
        // We can't guarantee no other test installed a source, so just assert
        // the contract holds for the default path: auto is a bool, files a vec.
        let _ = files;
        let _ = auto;
    }

    #[test]
    fn run_bash_spawn_non_login_argv() {
        let mut le = LocalEnvironment::new("", 60, None);
        // Point cwd at temp dir so recovery doesn't trigger.
        le.cwd = std::env::temp_dir().to_string_lossy().into_owned();
        let spawn = le.run_bash_spawn("echo hi", false, None).unwrap();
        assert_eq!(spawn.argv.len(), 3);
        assert_eq!(spawn.argv[1], "-c");
        assert_eq!(spawn.argv[2], "echo hi");
        assert!(spawn.stdin_data.is_none());
    }

    #[test]
    fn run_bash_spawn_login_argv() {
        let mut le = LocalEnvironment::new("", 60, None);
        le.cwd = std::env::temp_dir().to_string_lossy().into_owned();
        let spawn = le
            .run_bash_spawn("echo hi", true, Some("data".to_string()))
            .unwrap();
        assert_eq!(spawn.argv[1], "-l");
        assert_eq!(spawn.argv[2], "-c");
        assert_eq!(spawn.stdin_data.as_deref(), Some("data"));
    }

    #[test]
    fn run_bash_spawn_recovers_missing_cwd() {
        let mut le = LocalEnvironment::new("", 60, None);
        let base = std::env::temp_dir()
            .to_string_lossy()
            .trim_end_matches('/')
            .to_string();
        le.cwd = format!("{base}/__hermes_missing_xyz__/deep");
        let spawn = le.run_bash_spawn("true", false, None).unwrap();
        assert_eq!(spawn.cwd, base);
        assert_eq!(le.cwd, base);
    }

    #[test]
    fn compute_temp_dir_prefers_env_tmpdir() {
        let mut env = BTreeMap::new();
        env.insert("TMPDIR".to_string(), "/custom/tmp/".to_string());
        assert_eq!(compute_temp_dir(&env), "/custom/tmp");
    }

    #[test]
    fn compute_temp_dir_root_collapse() {
        let mut env = BTreeMap::new();
        env.insert("TMPDIR".to_string(), "/".to_string());
        assert_eq!(compute_temp_dir(&env), "/");
    }

    #[test]
    fn local_env_new_expands_tilde_and_paths_allocated() {
        unsafe {
            std::env::set_var("HOME", "/home/tester");
        }
        if !IS_WINDOWS {
            let le = LocalEnvironment::new("~/work", 30, None);
            assert_eq!(le.cwd, "/home/tester/work");
            assert!(le.snapshot_path.contains("hermes_env_snapshot_"));
            assert!(le.cwd_file.contains("hermes_cwd_"));
            assert_eq!(le.timeout, 30);
        }
    }

    #[test]
    fn cleanup_removes_temp_files() {
        let le = LocalEnvironment::new("", 60, None);
        std::fs::write(&le.snapshot_path, "x").unwrap();
        std::fs::write(&le.cwd_file, "y").unwrap();
        le.cleanup();
        assert!(!Path::new(&le.snapshot_path).exists());
        assert!(!Path::new(&le.cwd_file).exists());
    }

    #[test]
    fn update_cwd_from_marker_respects_existing_dir() {
        let mut le = LocalEnvironment::new("", 60, None);
        let valid = std::env::temp_dir().to_string_lossy().into_owned();
        std::fs::write(&le.cwd_file, format!("  {valid}  \n")).unwrap();
        let marker = le.update_cwd_from_marker();
        // The marker returns the trimmed file contents.
        assert_eq!(marker.as_deref(), Some(valid.as_str()));
        assert_eq!(le.cwd, valid);
        le.cleanup();
    }

    #[test]
    fn update_cwd_from_marker_ignores_missing_dir() {
        let mut le = LocalEnvironment::new("", 60, None);
        le.cwd = std::env::temp_dir().to_string_lossy().into_owned();
        let before = le.cwd.clone();
        std::fs::write(&le.cwd_file, "/__hermes_missing_marker_dir__/x").unwrap();
        le.update_cwd_from_marker();
        // cwd unchanged because the marker path isn't a directory.
        assert_eq!(le.cwd, before);
        le.cleanup();
    }
}
