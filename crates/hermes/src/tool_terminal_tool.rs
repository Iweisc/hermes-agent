//! Terminal tool module.
//!
//! Native Rust port of `tools/terminal_tool.py`. The Python module dispatches
//! shell commands to one of several execution backends (local, docker,
//! singularity, ssh, modal, daytona, vercel_sandbox), manages environment
//! lifecycle/cleanup, and applies a thick layer of command pre-processing and
//! result post-processing.
//!
//! What is ported here, faithfully and idiomatically:
//!   - Config parsing from environment variables (`_get_env_config`,
//!     `_parse_env_var`, `_safe_parse_import_env`).
//!   - Vercel Sandbox requirement validation (`_check_vercel_sandbox_requirements`,
//!     `_is_supported_vercel_runtime`).
//!   - The shell command transforms: sudo rewriting
//!     (`_rewrite_real_sudo_invocations`, `_transform_sudo_command`), compound
//!     background rewrite (`_rewrite_compound_background`), shell tokenizer
//!     (`_read_shell_token`), env-assignment detection (`_looks_like_env_assignment`).
//!   - Workdir validation (`_validate_workdir`).
//!   - Exit code interpretation (`_interpret_exit_code`).
//!   - Foreground/background guidance heuristics
//!     (`_foreground_background_guidance`, `_looks_like_help_or_version_command`,
//!     `_command_requires_pipe_stdin`).
//!   - Notification flag conflict resolution
//!     (`_resolve_notification_flag_conflict`).
//!   - Container task id resolution + per-task override registry.
//!   - The interactive-sudo password cache, scoped like the Python original.
//!   - Modal backend state resolution wrapper.
//!   - The tool description constant and JSON schema.
//!
//! Cross-cutting subsystems that the Python file imports (the concrete
//! environment classes, the process registry, the approval guard, plugin hooks,
//! the cleanup thread machinery) are not reproduced wholesale. Where their
//! decision logic is self-contained it is mirrored; where it requires un-ported
//! runtime state it is exposed as a trait/parameter so callers can wire it in.
//! `crate::tool_tool_backend_helpers` is reused for modal-mode coercion and
//! backend-state resolution.

use std::collections::HashMap;
use std::env;
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::tool_tool_backend_helpers::{
    coerce_modal_mode, has_direct_modal_credentials, resolve_modal_backend_state,
    ModalBackendState,
};

// ===========================================================================
// Module-level constants and import-time env parsing
// ===========================================================================

/// Default container/sandbox image with Python + Node for max compatibility.
pub const DEFAULT_IMAGE: &str = "nikolaik/python-nodejs:python3.11-nodejs20";

const VERCEL_SANDBOX_DEFAULT_CWD: &str = "/vercel/sandbox";
const SUPPORTED_VERCEL_RUNTIMES: [&str; 3] = ["node24", "node22", "python3.13"];

/// Parse a module-level numeric env var without panicking on malformed input.
///
/// Mirrors `_safe_parse_import_env`: returns `default` when the var is unset or
/// empty, attempts the conversion otherwise, and falls back to `default` (with
/// a warning) on parse failure.
fn safe_parse_import_env_i64(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(raw) if !raw.is_empty() => match raw.trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => {
                log::warn!(
                    "Invalid value for {name}: {raw:?} (expected integer). Falling back to {default:?}.",
                );
                default
            }
        },
        _ => default,
    }
}

fn safe_parse_import_env_f64(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(raw) if !raw.is_empty() => match raw.trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                log::warn!(
                    "Invalid value for {name}: {raw:?} (expected number). Falling back to {default:?}.",
                );
                default
            }
        },
        _ => default,
    }
}

/// Hard cap on foreground timeout; override via `TERMINAL_MAX_FOREGROUND_TIMEOUT`.
pub fn foreground_max_timeout() -> i64 {
    safe_parse_import_env_i64("TERMINAL_MAX_FOREGROUND_TIMEOUT", 600)
}

/// Disk usage warning threshold in GB; override via `TERMINAL_DISK_WARNING_GB`.
pub fn disk_usage_warning_threshold_gb() -> f64 {
    safe_parse_import_env_f64("TERMINAL_DISK_WARNING_GB", 500.0)
}

// ===========================================================================
// Vercel Sandbox requirement validation
// ===========================================================================

/// Mirror of `_is_supported_vercel_runtime`: empty runtime is allowed (means
/// "use the backend default"); otherwise it must be one of the supported set.
pub fn is_supported_vercel_runtime(runtime: &str) -> bool {
    runtime.is_empty() || SUPPORTED_VERCEL_RUNTIMES.contains(&runtime)
}

/// Result of validating Vercel Sandbox backend requirements.
///
/// The Python `_check_vercel_sandbox_requirements` returns a bool and logs the
/// reason. Here we return both so callers can surface the message; `ok` matches
/// the Python return value exactly. `vercel_available` is supplied by the
/// caller because Python probes `importlib.util.find_spec("vercel")` which has
/// no native analogue.
pub struct VercelCheck {
    pub ok: bool,
    pub error: Option<String>,
}

/// Faithful port of `_check_vercel_sandbox_requirements`.
///
/// `vercel_available` stands in for the `importlib.util.find_spec("vercel")`
/// probe (Python: the `vercel` package must be importable). Auth env vars are
/// read directly from the process environment, matching the Python `os.getenv`
/// usage.
pub fn check_vercel_sandbox_requirements(
    vercel_runtime: &str,
    container_disk: i64,
    vercel_available: bool,
) -> VercelCheck {
    let runtime = vercel_runtime.trim();
    if !is_supported_vercel_runtime(runtime) {
        let supported = SUPPORTED_VERCEL_RUNTIMES.join(", ");
        let msg = format!(
            "Vercel Sandbox runtime {runtime:?} is not supported. Set TERMINAL_VERCEL_RUNTIME to one of: {supported}."
        );
        log::error!("{msg}");
        return VercelCheck { ok: false, error: Some(msg) };
    }

    if container_disk != 0 && container_disk != 51200 {
        let msg = format!(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK={container_disk}. Use the default shared setting (51200 MB)."
        );
        log::error!("{msg}");
        return VercelCheck { ok: false, error: Some(msg) };
    }

    if !vercel_available {
        let msg =
            "vercel is required for the Vercel Sandbox terminal backend: pip install vercel".to_string();
        log::error!("{msg}");
        return VercelCheck { ok: false, error: Some(msg) };
    }

    let has_oidc = !env::var("VERCEL_OIDC_TOKEN").unwrap_or_default().is_empty();
    let has_token = !env::var("VERCEL_TOKEN").unwrap_or_default().is_empty();
    let has_project = !env::var("VERCEL_PROJECT_ID").unwrap_or_default().is_empty();
    let has_team = !env::var("VERCEL_TEAM_ID").unwrap_or_default().is_empty();

    if has_oidc {
        return VercelCheck { ok: true, error: None };
    }

    if has_token || has_project || has_team {
        if has_token && has_project && has_team {
            return VercelCheck { ok: true, error: None };
        }
        let msg = "Vercel Sandbox backend selected with token auth, but VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all be set together. VERCEL_OIDC_TOKEN is supported for one-off local development only.".to_string();
        log::error!("{msg}");
        return VercelCheck { ok: false, error: Some(msg) };
    }

    let msg = "Vercel Sandbox backend selected but no supported auth configuration was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID for normal use. VERCEL_OIDC_TOKEN is supported for one-off local development only.".to_string();
    log::error!("{msg}");
    VercelCheck { ok: false, error: Some(msg) }
}

// ===========================================================================
// Interactive sudo password cache
// ===========================================================================

static SUDO_PASSWORD_CACHE: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

/// Resolve the cache scope for interactive sudo passwords.
///
/// Mirrors `_get_sudo_password_cache_scope`, but only the session-key and
/// thread-id branches are reproducible natively (the Python callback-identity
/// branch depends on registered Python callbacks). When `HERMES_SESSION_KEY` is
/// set, the scope is `session:<key>`; otherwise it falls back to the current
/// thread id, matching the Python `thread:<ident>` form.
fn sudo_password_cache_scope() -> String {
    let session_key = env::var("HERMES_SESSION_KEY").unwrap_or_default();
    if !session_key.is_empty() {
        return format!("session:{session_key}");
    }
    // Python uses threading.get_ident(); a stable per-thread token suffices.
    format!("thread:{:?}", std::thread::current().id())
}

/// Return the cached sudo password for the current scope (empty if none).
pub fn get_cached_sudo_password() -> String {
    let scope = sudo_password_cache_scope();
    let guard = SUDO_PASSWORD_CACHE.lock().unwrap();
    guard
        .as_ref()
        .and_then(|m| m.get(&scope).cloned())
        .unwrap_or_default()
}

/// Persist a sudo password for the current scope. Empty clears the entry.
pub fn set_cached_sudo_password(password: &str) {
    let scope = sudo_password_cache_scope();
    let mut guard = SUDO_PASSWORD_CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if password.is_empty() {
        map.remove(&scope);
    } else {
        map.insert(scope, password.to_string());
    }
}

/// Clear all cached sudo passwords (test/teardown helper).
pub fn reset_cached_sudo_passwords() {
    let mut guard = SUDO_PASSWORD_CACHE.lock().unwrap();
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
}

// ===========================================================================
// Workdir validation
// ===========================================================================

/// Return true if `ch` is in the workdir allowlist (matches the Python
/// `_WORKDIR_SAFE_RE` character class `[A-Za-z0-9/\\:_\-.~ +@=,]`).
fn is_workdir_safe_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || matches!(
            ch,
            '/' | '\\' | ':' | '_' | '-' | '.' | '~' | ' ' | '+' | '@' | '=' | ','
        )
}

/// Reject workdir values that don't look like a filesystem path.
///
/// Faithful port of `_validate_workdir`: empty is safe (returns `None`); any
/// disallowed character yields an error naming the first offender.
pub fn validate_workdir(workdir: &str) -> Option<String> {
    if workdir.is_empty() {
        return None;
    }
    if workdir.chars().all(is_workdir_safe_char) {
        return None;
    }
    for ch in workdir.chars() {
        if !is_workdir_safe_char(ch) {
            // Python uses repr(ch); reproduce the single-quoted form for ASCII.
            return Some(format!(
                "Blocked: workdir contains disallowed character '{ch}'. Use a simple filesystem path without shell metacharacters."
            ));
        }
    }
    Some("Blocked: workdir contains disallowed characters.".to_string())
}

// ===========================================================================
// Shell tokenizer + sudo / background rewriting
// ===========================================================================

/// Read one shell token preserving quotes/escapes, starting at byte `start`.
///
/// Faithful port of `_read_shell_token`. Operates on a `&[char]` view so we can
/// index by character position exactly as the Python string indexing does, then
/// the caller maps positions back. Returns `(token_string, next_index)`.
fn read_shell_token(command: &[char], start: usize) -> (String, usize) {
    let n = command.len();
    let mut i = start;

    while i < n {
        let ch = command[i];
        if ch.is_whitespace() || matches!(ch, ';' | '|' | '&' | '(' | ')') {
            break;
        }
        if ch == '\'' {
            i += 1;
            while i < n && command[i] != '\'' {
                i += 1;
            }
            if i < n {
                i += 1;
            }
            continue;
        }
        if ch == '"' {
            i += 1;
            while i < n {
                let inner = command[i];
                if inner == '\\' && i + 1 < n {
                    i += 2;
                    continue;
                }
                if inner == '"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if ch == '\\' && i + 1 < n {
            i += 2;
            continue;
        }
        i += 1;
    }

    (command[start..i].iter().collect(), i)
}

/// Return true when `token` is a leading shell environment assignment
/// (`NAME=value` with a valid identifier name). Mirrors
/// `_looks_like_env_assignment`.
fn looks_like_env_assignment(token: &str) -> bool {
    if !token.contains('=') || token.starts_with('=') {
        return false;
    }
    let name = token.splitn(2, '=').next().unwrap_or("");
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Rewrite only real unquoted `sudo` command words to `sudo -S -p ''`.
///
/// Faithful port of `_rewrite_real_sudo_invocations`. Returns the rewritten
/// command and whether any real sudo invocation was found.
pub fn rewrite_real_sudo_invocations(command: &str) -> (String, bool) {
    let chars: Vec<char> = command.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0usize;
    let mut command_start = true;
    let mut found = false;

    let starts_with = |idx: usize, pat: &str| -> bool {
        let p: Vec<char> = pat.chars().collect();
        if idx + p.len() > n {
            return false;
        }
        chars[idx..idx + p.len()] == p[..]
    };

    while i < n {
        let ch = chars[i];

        if ch.is_whitespace() {
            out.push(ch);
            if ch == '\n' {
                command_start = true;
            }
            i += 1;
            continue;
        }

        if ch == '#' && command_start {
            // command.find("\n", i)
            let rest: String = chars[i..].iter().collect();
            match rest.find('\n') {
                None => {
                    out.push_str(&rest);
                    break;
                }
                Some(rel) => {
                    let comment_end = i + rest[..rel].chars().count();
                    let segment: String = chars[i..comment_end].iter().collect();
                    out.push_str(&segment);
                    i = comment_end;
                    continue;
                }
            }
        }

        if starts_with(i, "&&") || starts_with(i, "||") || starts_with(i, ";;") {
            out.push(chars[i]);
            out.push(chars[i + 1]);
            i += 2;
            command_start = true;
            continue;
        }

        if matches!(ch, ';' | '|' | '&' | '(') {
            out.push(ch);
            i += 1;
            command_start = true;
            continue;
        }

        if ch == ')' {
            out.push(ch);
            i += 1;
            command_start = false;
            continue;
        }

        let (token, next_i) = read_shell_token(&chars, i);
        if command_start && token == "sudo" {
            out.push_str("sudo -S -p ''");
            found = true;
        } else {
            out.push_str(&token);
        }

        if command_start && looks_like_env_assignment(&token) {
            command_start = true;
        } else {
            command_start = false;
        }
        i = next_i;
    }

    (out, found)
}

/// Wrap `A && B &` (or `A || B &`) into `A && { B & }` at depth 0.
///
/// Faithful port of `_rewrite_compound_background`. Handles redirects,
/// quoted strings, parenthesised subshells and brace groups; leaves simple
/// `cmd &` alone.
pub fn rewrite_compound_background(command: &str) -> String {
    let chars: Vec<char> = command.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut paren_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut last_chain_op_end: i64 = -1;
    let mut rewrites: Vec<(usize, usize)> = Vec::new(); // (chain_op_end, amp_pos)

    let starts_with = |idx: usize, pat: &str| -> bool {
        let p: Vec<char> = pat.chars().collect();
        if idx + p.len() > n {
            return false;
        }
        chars[idx..idx + p.len()] == p[..]
    };

    while i < n {
        let ch = chars[i];

        if ch == '\n' && paren_depth == 0 && brace_depth == 0 {
            last_chain_op_end = -1;
            i += 1;
            continue;
        }

        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch == '#' {
            // find next newline
            let mut nl = None;
            for j in i..n {
                if chars[j] == '\n' {
                    nl = Some(j);
                    break;
                }
            }
            match nl {
                None => break,
                Some(j) => {
                    i = j;
                    continue;
                }
            }
        }

        if ch == '\\' && i + 1 < n {
            i += 2;
            continue;
        }

        if ch == '\'' || ch == '"' {
            let (_, next_i) = read_shell_token(&chars, i);
            i = next_i.max(i + 1);
            continue;
        }

        if ch == '(' {
            paren_depth += 1;
            i += 1;
            continue;
        }
        if ch == ')' {
            paren_depth = (paren_depth - 1).max(0);
            i += 1;
            continue;
        }

        if ch == '{' && i + 1 < n && (chars[i + 1].is_whitespace() || chars[i + 1] == '\n') {
            brace_depth += 1;
            i += 1;
            continue;
        }
        if ch == '}' && brace_depth > 0 {
            brace_depth -= 1;
            last_chain_op_end = -1;
            i += 1;
            continue;
        }

        if paren_depth > 0 || brace_depth > 0 {
            i += 1;
            continue;
        }

        if starts_with(i, "&&") || starts_with(i, "||") {
            last_chain_op_end = (i + 2) as i64;
            i += 2;
            continue;
        }

        if ch == ';' {
            last_chain_op_end = -1;
            i += 1;
            continue;
        }

        if ch == '|' {
            last_chain_op_end = -1;
            i += 1;
            continue;
        }

        if ch == '&' {
            if i + 1 < n && chars[i + 1] == '>' {
                i += 2;
                continue;
            }
            // `>&` / `<&` fd target — look back past whitespace
            let mut j = i as i64 - 1;
            while j >= 0 && chars[j as usize].is_whitespace() {
                j -= 1;
            }
            if j >= 0 && matches!(chars[j as usize], '<' | '>') {
                i += 1;
                continue;
            }
            if last_chain_op_end >= 0 {
                rewrites.push((last_chain_op_end as usize, i));
            }
            last_chain_op_end = -1;
            i += 1;
            continue;
        }

        let (_, next_i) = read_shell_token(&chars, i);
        i = next_i.max(i + 1);
    }

    if rewrites.is_empty() {
        return command.to_string();
    }

    // Apply rewrites back-to-front; operate on the char vector.
    let mut result = chars;
    for (chain_end, amp_pos) in rewrites.into_iter().rev() {
        let mut insert_pos = chain_end;
        while insert_pos < amp_pos && result[insert_pos].is_whitespace() {
            insert_pos += 1;
        }
        let prefix: Vec<char> = result[..insert_pos].to_vec();
        let middle: Vec<char> = result[insert_pos..amp_pos].to_vec();
        let suffix: Vec<char> = result[amp_pos + 1..].to_vec();

        let mut rebuilt: Vec<char> = prefix;
        rebuilt.extend("{ ".chars());
        rebuilt.extend(middle);
        rebuilt.extend("& }".chars());
        rebuilt.extend(suffix);
        result = rebuilt;
    }

    result.into_iter().collect()
}

// ===========================================================================
// sudo transform
// ===========================================================================

/// Result of `transform_sudo_command`: the (possibly rewritten) command and the
/// stdin string to feed to sudo, mirroring the Python tuple.
pub struct SudoTransform {
    pub command: Option<String>,
    pub sudo_stdin: Option<String>,
}

/// Inputs required to reproduce `_transform_sudo_command` without consulting
/// live runtime state that has no native analogue.
///
/// - `sudo_nopasswd_works`: result of the `sudo -n true` probe (Python
///   `_sudo_nopasswd_works`, only meaningful for the local backend).
/// - `interactive`: whether `HERMES_INTERACTIVE` is set.
/// - `prompt_password`: optional closure used in interactive mode to read the
///   password (stands in for `_prompt_for_sudo_password`). When `None`, the
///   interactive prompt is skipped.
pub struct SudoTransformCtx<'a> {
    pub sudo_nopasswd_works: bool,
    pub interactive: bool,
    pub prompt_password: Option<&'a dyn Fn() -> String>,
}

/// Faithful port of `_transform_sudo_command`.
///
/// `SUDO_PASSWORD` is read from the environment; the interactive prompt and the
/// `sudo -n true` probe are supplied through `ctx` so this function has no
/// side effects beyond updating the password cache (matching Python).
pub fn transform_sudo_command(command: Option<&str>, ctx: &SudoTransformCtx<'_>) -> SudoTransform {
    let command = match command {
        None => return SudoTransform { command: None, sudo_stdin: None },
        Some(c) => c,
    };
    let (transformed, has_real_sudo) = rewrite_real_sudo_invocations(command);
    if !has_real_sudo {
        return SudoTransform { command: Some(command.to_string()), sudo_stdin: None };
    }

    let has_configured_password = env::var_os("SUDO_PASSWORD").is_some();
    let mut sudo_password = if has_configured_password {
        env::var("SUDO_PASSWORD").unwrap_or_default()
    } else {
        get_cached_sudo_password()
    };

    if !has_configured_password && sudo_password.is_empty() && ctx.sudo_nopasswd_works {
        return SudoTransform { command: Some(command.to_string()), sudo_stdin: None };
    }

    if !has_configured_password && sudo_password.is_empty() && ctx.interactive {
        if let Some(prompt) = ctx.prompt_password {
            sudo_password = prompt();
            if !sudo_password.is_empty() {
                set_cached_sudo_password(&sudo_password);
            }
        }
    }

    if has_configured_password || !sudo_password.is_empty() {
        let mut stdin = sudo_password.clone();
        stdin.push('\n');
        return SudoTransform { command: Some(transformed), sudo_stdin: Some(stdin) };
    }

    SudoTransform { command: Some(command.to_string()), sudo_stdin: None }
}

// ===========================================================================
// Exit code interpretation
// ===========================================================================

/// Return a human-readable note when a non-zero exit code is non-erroneous.
///
/// Faithful port of `_interpret_exit_code`. Returns `None` for exit 0 or
/// genuinely erroneous codes.
pub fn interpret_exit_code(command: &str, exit_code: i64) -> Option<String> {
    if exit_code == 0 {
        return None;
    }

    // Split on shell operators `||`, `&&`, `|`, `;` and take the last segment.
    let last_segment = split_last_segment(command);

    // Base command: first word, skipping VAR=val assignments, stripping path.
    let mut base_cmd = String::new();
    for w in last_segment.split_whitespace() {
        if w.contains('=') && !w.starts_with('-') {
            continue;
        }
        base_cmd = w.rsplit('/').next().unwrap_or(w).to_string();
        break;
    }

    if base_cmd.is_empty() {
        return None;
    }

    let note: Option<&str> = match base_cmd.as_str() {
        "grep" | "egrep" | "fgrep" | "rg" | "ag" | "ack" if exit_code == 1 => {
            Some("No matches found (not an error)")
        }
        "diff" | "colordiff" if exit_code == 1 => Some("Files differ (expected, not an error)"),
        "find" if exit_code == 1 => {
            Some("Some directories were inaccessible (partial results may still be valid)")
        }
        "test" | "[" if exit_code == 1 => {
            Some("Condition evaluated to false (expected, not an error)")
        }
        "curl" => match exit_code {
            6 => Some("Could not resolve host"),
            7 => Some("Failed to connect to host"),
            22 => Some("HTTP response code indicated error (e.g. 404, 500)"),
            28 => Some("Operation timed out"),
            _ => None,
        },
        "git" if exit_code == 1 => Some(
            "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)",
        ),
        _ => None,
    };

    note.map(|s| s.to_string())
}

/// Mirror of `re.split(r'\s*(?:\|\||&&|[|;])\s*', command)` then `.strip()` on
/// the last element. Splits on the shell operators `||`, `&&`, `|`, `;` with
/// surrounding whitespace consumed.
fn split_last_segment(command: &str) -> String {
    let chars: Vec<char> = command.chars().collect();
    let n = chars.len();
    let mut last_start = 0usize;
    let mut i = 0usize;

    let is_op_at = |idx: usize| -> Option<usize> {
        // Return op length if a delimiter starts at idx.
        if idx + 1 < n && chars[idx] == '|' && chars[idx + 1] == '|' {
            return Some(2);
        }
        if idx + 1 < n && chars[idx] == '&' && chars[idx + 1] == '&' {
            return Some(2);
        }
        if chars[idx] == '|' || chars[idx] == ';' {
            return Some(1);
        }
        None
    };

    while i < n {
        if let Some(op_len) = is_op_at(i) {
            // The regex consumes \s* before and after the operator; the start of
            // the following segment is after trailing whitespace.
            let mut after = i + op_len;
            while after < n && chars[after].is_whitespace() {
                after += 1;
            }
            last_start = after;
            i += op_len;
            continue;
        }
        i += 1;
    }

    let segment: String = chars[last_start..].iter().collect();
    segment.trim().to_string()
}

// ===========================================================================
// PTY / foreground guidance heuristics
// ===========================================================================

/// Return true when PTY mode would break stdin-driven commands.
///
/// Faithful port of `_command_requires_pipe_stdin`.
pub fn command_requires_pipe_stdin(command: &str) -> bool {
    let normalized = normalize_ws_lower(command);
    normalized.starts_with("gh auth login") && normalized.contains("--with-token")
}

/// Collapse whitespace runs to single spaces and lowercase, matching the Python
/// idiom `" ".join(command.lower().split())`.
fn normalize_ws_lower(command: &str) -> String {
    command
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Return true for informational invocations that should never be blocked.
///
/// Faithful port of `_looks_like_help_or_version_command`.
pub fn looks_like_help_or_version_command(command: &str) -> bool {
    let normalized = normalize_ws_lower(command);
    normalized.contains(" --help")
        || normalized.ends_with(" -h")
        || normalized.contains(" --version")
        || normalized.ends_with(" -v")
}

/// Suggest background mode when a foreground command looks long-lived.
///
/// Faithful port of `_foreground_background_guidance`. Returns the guidance
/// string when one applies, else `None`.
pub fn foreground_background_guidance(command: &str) -> Option<String> {
    if looks_like_help_or_version_command(command) {
        return None;
    }

    if shell_level_background(command) {
        return Some(
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). Use terminal(background=true) so Hermes can track the process, then run readiness checks and tests in separate commands.".to_string(),
        );
    }

    if inline_or_trailing_background_amp(command) {
        return Some(
            "Foreground command uses '&' backgrounding. Use terminal(background=true) for long-lived processes, then run health checks and tests in follow-up terminal calls.".to_string(),
        );
    }

    if matches_long_lived_foreground(command) {
        return Some(
            "This foreground command appears to start a long-lived server/watch process. Run it with background=true, verify readiness (health endpoint/log signal), then execute tests in a separate command.".to_string(),
        );
    }

    None
}

/// Match `\b(?:nohup|disown|setsid)\b` case-insensitively.
fn shell_level_background(command: &str) -> bool {
    let lower = command.to_lowercase();
    for kw in ["nohup", "disown", "setsid"] {
        if word_boundary_contains(&lower, kw) {
            return true;
        }
    }
    false
}

/// Match either `\s&\s` (inline) or `\s&\s*(?:#.*)?$` (trailing) backgrounding.
fn inline_or_trailing_background_amp(command: &str) -> bool {
    let chars: Vec<char> = command.chars().collect();
    let n = chars.len();
    for i in 0..n {
        if chars[i] != '&' {
            continue;
        }
        // require a whitespace char immediately before the &
        if i == 0 || !chars[i - 1].is_whitespace() {
            continue;
        }
        // inline: \s&\s -> a whitespace immediately after
        if i + 1 < n && chars[i + 1].is_whitespace() {
            return true;
        }
        // trailing: \s&\s*(?:#.*)?$ -> after the &, only whitespace then
        // optional comment then end.
        let mut j = i + 1;
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        if j == n {
            return true;
        }
        if chars[j] == '#' {
            // `#.*` to end (any chars, no newline requirement in Python's `$`
            // default mode — but commands here are single-line typically).
            return true;
        }
    }
    false
}

/// Match the `_LONG_LIVED_FOREGROUND_PATTERNS` set against the command.
fn matches_long_lived_foreground(command: &str) -> bool {
    let lower = command.to_lowercase();

    // npm|pnpm|yarn|bun (run )?(dev|start|serve|watch)
    if long_lived_pkg_run(&lower) {
        return true;
    }
    // docker compose up
    if contains_words(&lower, &["docker", "compose", "up"]) {
        return true;
    }
    // next dev
    if contains_words(&lower, &["next", "dev"]) {
        return true;
    }
    // vite (\s|$) — \bvite followed by whitespace or end
    if word_boundary_vite(&lower) {
        return true;
    }
    // nodemon
    if word_boundary_contains(&lower, "nodemon") {
        return true;
    }
    // uvicorn
    if word_boundary_contains(&lower, "uvicorn") {
        return true;
    }
    // gunicorn
    if word_boundary_contains(&lower, "gunicorn") {
        return true;
    }
    // python(3)? -m http.server
    if long_lived_http_server(&lower) {
        return true;
    }
    false
}

fn long_lived_pkg_run(lower: &str) -> bool {
    // \b(npm|pnpm|yarn|bun)\s+(run\s+)?(dev|start|serve|watch)\b
    let pkgs = ["npm", "pnpm", "yarn", "bun"];
    let toks: Vec<&str> = lower.split_whitespace().collect();
    for i in 0..toks.len() {
        if !pkgs.contains(&toks[i]) {
            continue;
        }
        let mut j = i + 1;
        if j < toks.len() && toks[j] == "run" {
            j += 1;
        }
        if j < toks.len() && matches!(toks[j], "dev" | "start" | "serve" | "watch") {
            return true;
        }
    }
    false
}

fn long_lived_http_server(lower: &str) -> bool {
    // \bpython(3)?\s+-m\s+http\.server\b
    let toks: Vec<&str> = lower.split_whitespace().collect();
    for i in 0..toks.len() {
        if (toks[i] == "python" || toks[i] == "python3")
            && i + 2 < toks.len()
            && toks[i + 1] == "-m"
            && (toks[i + 2] == "http.server" || toks[i + 2].starts_with("http.server"))
        {
            return true;
        }
    }
    false
}

/// `\bword\b` containment using ASCII word-character boundaries.
fn word_boundary_contains(haystack: &str, word: &str) -> bool {
    let hc: Vec<char> = haystack.chars().collect();
    let wc: Vec<char> = word.chars().collect();
    if wc.is_empty() {
        return false;
    }
    let n = hc.len();
    let m = wc.len();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut i = 0;
    while i + m <= n {
        if hc[i..i + m] == wc[..] {
            let before_ok = i == 0 || !is_word(hc[i - 1]);
            let after_ok = i + m == n || !is_word(hc[i + m]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// `\bvite(?:\s|$)` — `vite` at a word boundary followed by whitespace or end.
fn word_boundary_vite(haystack: &str) -> bool {
    let hc: Vec<char> = haystack.chars().collect();
    let wc: Vec<char> = "vite".chars().collect();
    let n = hc.len();
    let m = wc.len();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut i = 0;
    while i + m <= n {
        if hc[i..i + m] == wc[..] {
            let before_ok = i == 0 || !is_word(hc[i - 1]);
            let after_ok = i + m == n || hc[i + m].is_whitespace();
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// All of `words` appear in order, each at a word boundary, as a contiguous
/// whitespace-separated run (matching `\bA\s+B\s+C\b`).
fn contains_words(lower: &str, words: &[&str]) -> bool {
    let toks: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    if toks.len() < words.len() {
        return false;
    }
    for start in 0..=toks.len() - words.len() {
        if (0..words.len()).all(|k| toks[start + k] == words[k]) {
            return true;
        }
    }
    false
}

// ===========================================================================
// Notification flag conflict resolution
// ===========================================================================

/// Decide what to do when both `notify_on_complete` and `watch_patterns` are
/// set. Faithful port of `_resolve_notification_flag_conflict`.
///
/// Returns `(watch_patterns_to_use, conflict_note)`. `conflict_note` is empty
/// when there is no conflict. When a conflict exists, watch patterns are dropped
/// (`None`).
pub fn resolve_notification_flag_conflict(
    notify_on_complete: bool,
    watch_patterns: Option<Vec<String>>,
    background: bool,
) -> (Option<Vec<String>>, String) {
    let has_patterns = watch_patterns.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
    if background && notify_on_complete && has_patterns {
        return (
            None,
            "watch_patterns ignored because notify_on_complete=True; these two flags produce duplicate notifications when combined".to_string(),
        );
    }
    (watch_patterns, String::new())
}

// ===========================================================================
// Per-task env overrides + container task id resolution
// ===========================================================================

static TASK_ENV_OVERRIDES: Mutex<Option<HashMap<String, Value>>> = Mutex::new(None);

/// Register environment overrides for a specific task/rollout. Mirrors
/// `register_task_env_overrides`. `overrides` should be a JSON object.
pub fn register_task_env_overrides(task_id: &str, overrides: Value) {
    let mut guard = TASK_ENV_OVERRIDES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(task_id.to_string(), overrides);
}

/// Clear environment overrides for a task after rollout completes. Mirrors
/// `clear_task_env_overrides`.
pub fn clear_task_env_overrides(task_id: &str) {
    let mut guard = TASK_ENV_OVERRIDES.lock().unwrap();
    if let Some(map) = guard.as_mut() {
        map.remove(task_id);
    }
}

/// Return a clone of the registered overrides for `task_id`, if any.
pub fn get_task_env_overrides(task_id: &str) -> Option<Value> {
    let guard = TASK_ENV_OVERRIDES.lock().unwrap();
    guard.as_ref().and_then(|m| m.get(task_id).cloned())
}

/// Map a tool-call `task_id` to the container/sandbox key. Faithful port of
/// `_resolve_container_task_id`: a registered override keeps the task id
/// isolated; everything else collapses to `"default"`.
pub fn resolve_container_task_id(task_id: Option<&str>) -> String {
    if let Some(id) = task_id {
        if !id.is_empty() {
            let guard = TASK_ENV_OVERRIDES.lock().unwrap();
            if guard.as_ref().map(|m| m.contains_key(id)).unwrap_or(false) {
                return id.to_string();
            }
        }
    }
    "default".to_string()
}

// ===========================================================================
// Environment configuration parsing
// ===========================================================================

/// Resolved terminal environment configuration. Mirror of the dict returned by
/// `_get_env_config`. Numeric fields use the same types/defaults as Python.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvConfig {
    pub env_type: String,
    pub modal_mode: String,
    pub docker_image: String,
    pub docker_forward_env: Value,
    pub singularity_image: String,
    pub modal_image: String,
    pub daytona_image: String,
    pub vercel_runtime: String,
    pub cwd: String,
    pub host_cwd: Option<String>,
    pub docker_mount_cwd_to_workspace: bool,
    pub timeout: i64,
    pub lifetime_seconds: i64,
    pub ssh_host: String,
    pub ssh_user: String,
    pub ssh_port: i64,
    pub ssh_key: String,
    pub ssh_persistent: bool,
    pub local_persistent: bool,
    pub container_cpu: f64,
    pub container_memory: i64,
    pub container_disk: i64,
    pub container_persistent: bool,
    pub docker_volumes: Value,
    pub docker_run_as_host_user: bool,
}

/// Error raised by config parsing for a malformed env var. Mirrors the
/// `ValueError` raised by `_parse_env_var`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvConfigError(pub String);

impl std::fmt::Display for EnvConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EnvConfigError {}

fn parse_env_i64(name: &str, default: &str) -> Result<i64, EnvConfigError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    raw.trim().parse::<i64>().map_err(|_| {
        EnvConfigError(format!(
            "Invalid value for {name}: {raw:?} (expected integer). Check ~/.hermes/.env or environment variables."
        ))
    })
}

fn parse_env_f64(name: &str, default: &str) -> Result<f64, EnvConfigError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    raw.trim().parse::<f64>().map_err(|_| {
        EnvConfigError(format!(
            "Invalid value for {name}: {raw:?} (expected number). Check ~/.hermes/.env or environment variables."
        ))
    })
}

fn parse_env_json(name: &str, default: &str) -> Result<Value, EnvConfigError> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    serde_json::from_str(&raw).map_err(|_| {
        EnvConfigError(format!(
            "Invalid value for {name}: {raw:?} (expected valid JSON). Check ~/.hermes/.env or environment variables."
        ))
    })
}

/// Truthy parse matching `... .lower() in ("true", "1", "yes")`.
fn env_truthy(name: &str, default: &str) -> bool {
    let raw = env::var(name).unwrap_or_else(|_| default.to_string());
    matches!(raw.to_lowercase().as_str(), "true" | "1" | "yes")
}

/// Expand a leading `~` to the home directory, matching `os.path.expanduser`
/// for the common `~`/`~/...` cases.
fn expanduser(path: &str) -> String {
    if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().to_string();
        }
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return format!("{}/{}", home.to_string_lossy(), rest);
        }
    }
    path.to_string()
}

/// Get terminal environment configuration. Faithful port of `_get_env_config`.
///
/// `cwd_provider` supplies the host current directory (Python `os.getcwd()`).
/// Defaulting to `std::env::current_dir` keeps behaviour identical; it is a
/// parameter so tests can pin it.
pub fn get_env_config() -> Result<EnvConfig, EnvConfigError> {
    let cwd_now = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    get_env_config_with_cwd(&cwd_now)
}

/// Variant of [`get_env_config`] with an injectable host current directory.
pub fn get_env_config_with_cwd(host_getcwd: &str) -> Result<EnvConfig, EnvConfigError> {
    let env_type = env::var("TERMINAL_ENV").unwrap_or_else(|_| "local".to_string());

    let mount_docker_cwd =
        env_truthy("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false");

    let default_cwd = match env_type.as_str() {
        "local" => host_getcwd.to_string(),
        "ssh" => "~".to_string(),
        "vercel_sandbox" => VERCEL_SANDBOX_DEFAULT_CWD.to_string(),
        _ => "/root".to_string(),
    };

    let mut cwd = env::var("TERMINAL_CWD").unwrap_or_else(|_| default_cwd.clone());
    if !cwd.is_empty() {
        cwd = expanduser(&cwd);
    }

    let mut host_cwd: Option<String> = None;
    let host_prefixes = ["/Users/", "/home/", "C:\\", "C:/"];

    if env_type == "docker" && mount_docker_cwd {
        let docker_cwd_source = match env::var("TERMINAL_CWD") {
            Ok(v) if !v.is_empty() => v,
            _ => host_getcwd.to_string(),
        };
        let candidate = abspath(&expanduser(&docker_cwd_source));
        let starts_host = host_prefixes.iter().any(|p| candidate.starts_with(p));
        let is_abs = candidate.starts_with('/') || is_windows_abs(&candidate);
        let is_dir = std::path::Path::new(&candidate).is_dir();
        let not_sandbox_root =
            !candidate.starts_with("/workspace") && !candidate.starts_with("/root");
        if starts_host || (is_abs && is_dir && not_sandbox_root) {
            host_cwd = Some(candidate);
            cwd = "/workspace".to_string();
        }
    } else if matches!(
        env_type.as_str(),
        "modal" | "docker" | "singularity" | "daytona" | "vercel_sandbox"
    ) && !cwd.is_empty()
    {
        let is_host_path = host_prefixes.iter().any(|p| cwd.starts_with(p));
        let is_relative = !(cwd.starts_with('/') || is_windows_abs(&cwd));
        if (is_host_path || is_relative) && cwd != default_cwd {
            log::info!(
                "Ignoring TERMINAL_CWD={cwd:?} for {env_type} backend (host/relative path won't work in sandbox). Using {default_cwd:?} instead.",
            );
            cwd = default_cwd.clone();
        }
    }

    let ssh_persistent_default = env::var("TERMINAL_PERSISTENT_SHELL").unwrap_or_else(|_| "true".to_string());
    let ssh_persistent_raw =
        env::var("TERMINAL_SSH_PERSISTENT").unwrap_or(ssh_persistent_default);
    let ssh_persistent = matches!(ssh_persistent_raw.to_lowercase().as_str(), "true" | "1" | "yes");

    Ok(EnvConfig {
        env_type: env_type.clone(),
        modal_mode: coerce_modal_mode(env::var("TERMINAL_MODAL_MODE").ok().as_deref().or(Some("auto"))),
        docker_image: env::var("TERMINAL_DOCKER_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_string()),
        docker_forward_env: parse_env_json("TERMINAL_DOCKER_FORWARD_ENV", "[]")?,
        singularity_image: env::var("TERMINAL_SINGULARITY_IMAGE")
            .unwrap_or_else(|_| format!("docker://{DEFAULT_IMAGE}")),
        modal_image: env::var("TERMINAL_MODAL_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_string()),
        daytona_image: env::var("TERMINAL_DAYTONA_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_string()),
        vercel_runtime: env::var("TERMINAL_VERCEL_RUNTIME").unwrap_or_default().trim().to_string(),
        cwd,
        host_cwd,
        docker_mount_cwd_to_workspace: mount_docker_cwd,
        timeout: parse_env_i64("TERMINAL_TIMEOUT", "180")?,
        lifetime_seconds: parse_env_i64("TERMINAL_LIFETIME_SECONDS", "300")?,
        ssh_host: env::var("TERMINAL_SSH_HOST").unwrap_or_default(),
        ssh_user: env::var("TERMINAL_SSH_USER").unwrap_or_default(),
        ssh_port: parse_env_i64("TERMINAL_SSH_PORT", "22")?,
        ssh_key: env::var("TERMINAL_SSH_KEY").unwrap_or_default(),
        ssh_persistent,
        local_persistent: env_truthy("TERMINAL_LOCAL_PERSISTENT", "false"),
        container_cpu: parse_env_f64("TERMINAL_CONTAINER_CPU", "1")?,
        container_memory: parse_env_i64("TERMINAL_CONTAINER_MEMORY", "5120")?,
        container_disk: parse_env_i64("TERMINAL_CONTAINER_DISK", "51200")?,
        container_persistent: env_truthy("TERMINAL_CONTAINER_PERSISTENT", "true"),
        docker_volumes: parse_env_json("TERMINAL_DOCKER_VOLUMES", "[]")?,
        docker_run_as_host_user: env_truthy("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false"),
    })
}

fn is_windows_abs(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Approximate `os.path.abspath`: absolute paths returned as-is, relative paths
/// joined to the current directory. (Used only for the docker mount candidate.)
fn abspath(path: &str) -> String {
    if path.starts_with('/') || is_windows_abs(path) {
        return path.to_string();
    }
    match env::current_dir() {
        Ok(cwd) => cwd.join(path).to_string_lossy().to_string(),
        Err(_) => path.to_string(),
    }
}

// ===========================================================================
// Modal backend state wrapper
// ===========================================================================

/// Resolve direct vs managed Modal backend selection.
///
/// Mirror of `_get_modal_backend_state`. Delegates to
/// [`crate::tool_tool_backend_helpers::resolve_modal_backend_state`], supplying
/// the direct-credentials probe natively. `managed_ready` (Python
/// `is_managed_tool_gateway_ready("modal")`) and `managed_nous_tools_enabled`
/// are passed in because they consult un-ported subscription state.
pub fn get_modal_backend_state(
    modal_mode: Option<&str>,
    managed_ready: bool,
    managed_nous_tools_enabled: bool,
) -> ModalBackendState {
    resolve_modal_backend_state(
        modal_mode,
        has_direct_modal_credentials(),
        managed_ready,
        managed_nous_tools_enabled,
    )
}

// ===========================================================================
// Output truncation (foreground result post-processing)
// ===========================================================================

/// Truncate output keeping head + tail, mirroring the inline truncation in
/// `terminal_tool`. `max_chars` corresponds to `get_max_bytes()`.
///
/// Operates on character counts to match Python's `len(str)` / slicing
/// semantics (Python strings are sequences of code points).
pub fn truncate_output(output: &str, max_chars: usize) -> String {
    let chars: Vec<char> = output.chars().collect();
    if chars.len() <= max_chars {
        return output.to_string();
    }
    let head_chars = (max_chars as f64 * 0.4) as usize;
    let tail_chars = max_chars - head_chars;
    let total = chars.len();
    let omitted = total - head_chars - tail_chars;
    let truncated_notice = format!(
        "\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted out of {total} total] ...\n\n"
    );
    let head: String = chars[..head_chars].iter().collect();
    let tail: String = chars[total - tail_chars..].iter().collect();
    format!("{head}{truncated_notice}{tail}")
}

// ===========================================================================
// sudo failure messaging
// ===========================================================================

/// Append a helpful tip when sudo fails in a gateway (messaging) context.
///
/// Faithful port of `_handle_sudo_failure`. `is_gateway` corresponds to
/// `os.getenv("HERMES_GATEWAY_SESSION")` being set; `hermes_home_display` is the
/// value of `display_hermes_home()` (Python imports it lazily). When not in a
/// gateway context, the output is returned unchanged.
pub fn handle_sudo_failure(output: &str, is_gateway: bool, hermes_home_display: &str) -> String {
    if !is_gateway {
        return output.to_string();
    }
    let sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ];
    for failure in sudo_failures {
        if output.contains(failure) {
            return format!(
                "{output}\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {hermes_home_display}/.env on the agent machine."
            );
        }
    }
    output.to_string()
}

// ===========================================================================
// Tool description + schema
// ===========================================================================

/// Tool description shown to the model. Verbatim port of
/// `TERMINAL_TOOL_DESCRIPTION`.
pub const TERMINAL_TOOL_DESCRIPTION: &str = "Execute shell commands on a Linux environment. Filesystem usually persists between calls.\n\nDo NOT use cat/head/tail to read files — use read_file instead.\nDo NOT use grep/rg/find to search — use search_files instead.\nDo NOT use ls to list directories — use search_files(target='files') instead.\nDo NOT use sed/awk to edit files — use patch instead.\nDo NOT use echo/cat heredoc to create files — use write_file instead.\nReserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.\n\nForeground (default): Commands return INSTANTLY when done, even if the timeout is high. Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. Prefer foreground for short commands.\nBackground: Set background=true to get a session_id. Two patterns:\n  (1) Long-lived processes that never exit (servers, watchers).\n  (2) Long-running tasks with notify_on_complete=true — you can keep working on other things and the system auto-notifies you when the task finishes. Great for test suites, builds, deployments, or anything that takes more than a minute.\nFor servers/watchers, do NOT use shell-level background wrappers (nohup/disown/setsid/trailing '&') in foreground mode. Use background=true so Hermes can track lifecycle and output.\nAfter starting a server, verify readiness with a health check or log signal, then run tests in a separate terminal() call. Avoid blind sleep loops.\nUse process(action=\"poll\") for progress checks, process(action=\"wait\") to block until done.\nWorking directory: Use 'workdir' for per-command cwd.\nPTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).\n\nDo NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page.\n";

/// Build the terminal tool JSON schema. Faithful port of `TERMINAL_SCHEMA`,
/// with the `foreground max` substitutions resolved at call time.
pub fn terminal_schema() -> Value {
    let fg_max = foreground_max_timeout();
    json!({
        "name": "terminal",
        "description": TERMINAL_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute on the VM"
                },
                "background": {
                    "type": "boolean",
                    "description": "Run the command in the background. Two patterns: (1) Long-lived processes that never exit (servers, watchers). (2) Long-running tasks paired with notify_on_complete=true — you can keep working and get notified when the task finishes. For short commands, prefer foreground with a generous timeout instead.",
                    "default": false
                },
                "timeout": {
                    "type": "integer",
                    "description": format!("Max seconds to wait (default: 180, foreground max: {fg_max}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. Foreground timeout above {fg_max}s is rejected; use background=true for longer commands."),
                    "minimum": 1
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for this command (absolute path). Defaults to the session working directory."
                },
                "pty": {
                    "type": "boolean",
                    "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                    "default": false
                },
                "notify_on_complete": {
                    "type": "boolean",
                    "description": "When true (and background=true), you'll be automatically notified exactly once when the process finishes. **This is the right choice for almost every long-running task** — tests, builds, deployments, multi-item batch jobs, anything that takes over a minute and has a defined end. Use this and keep working on other things; the system notifies you on exit. MUTUALLY EXCLUSIVE with watch_patterns — when both are set, watch_patterns is dropped.",
                    "default": false
                },
                "watch_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Strings to watch for in background process output. HARD RATE LIMIT: at most 1 notification per 15 seconds per process — matches arriving inside the cooldown are dropped. After 3 consecutive 15-second windows with dropped matches, watch_patterns is automatically disabled for that process and promoted to notify_on_complete behavior (one notification on exit, no more mid-process spam). USE ONLY for truly rare, one-shot mid-process signals on LONG-LIVED processes that will never exit on their own — e.g. ['Application startup complete'] on a server so you know when to hit its endpoint, or ['migration done'] on a daemon. DO NOT use for: (1) end-of-run markers like 'DONE'/'PASS' — use notify_on_complete instead; (2) error patterns like 'ERROR'/'Traceback' in loops or multi-item batch jobs — they fire on every iteration and you'll hit the strike limit fast; (3) anything you'd ever combine with notify_on_complete. When in doubt, choose notify_on_complete. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both."
                }
            },
            "required": ["command"]
        }
    })
}

/// A `Value` JSON error string matching `json.dumps({...}, ensure_ascii=False)`.
fn json_err(map: Value) -> String {
    serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
}

/// Build the early-return error JSON for a non-string command. Mirrors the
/// `not isinstance(command, str)` branch of `terminal_tool`.
pub fn invalid_command_error(type_name: &str) -> String {
    json_err(json!({
        "output": "",
        "exit_code": -1,
        "error": format!("Invalid command: expected string, got {type_name}"),
        "status": "error",
    }))
}

/// Build the foreground-timeout-exceeded error JSON. Mirrors the
/// `timeout > FOREGROUND_MAX_TIMEOUT` branch.
pub fn foreground_timeout_exceeded_error(timeout: i64) -> String {
    let fg_max = foreground_max_timeout();
    json_err(json!({
        "error": format!(
            "Foreground timeout {timeout}s exceeds the maximum of {fg_max}s. Use background=true with notify_on_complete=true for long-running commands."
        ),
    }))
}

/// Build the foreground/background guidance error JSON.
pub fn guidance_error(guidance: &str) -> String {
    json_err(json!({
        "output": "",
        "exit_code": -1,
        "error": guidance,
        "status": "error",
    }))
}

/// Build the blocked-workdir error JSON.
pub fn workdir_blocked_error(message: &str) -> String {
    json_err(json!({
        "output": "",
        "exit_code": -1,
        "error": message,
        "status": "blocked",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_runtime() {
        assert!(is_supported_vercel_runtime(""));
        assert!(is_supported_vercel_runtime("node24"));
        assert!(is_supported_vercel_runtime("python3.13"));
        assert!(!is_supported_vercel_runtime("python3.11"));
    }

    #[test]
    fn validate_workdir_allowlist() {
        assert_eq!(validate_workdir(""), None);
        assert_eq!(validate_workdir("/home/user/project-1"), None);
        assert_eq!(validate_workdir("C:\\Users\\me"), None);
        assert_eq!(validate_workdir("~/dir with space"), None);
        let err = validate_workdir("/tmp; rm -rf /").unwrap();
        assert!(err.contains("disallowed character ';'"), "got: {err}");
        let err = validate_workdir("/a$(b)").unwrap();
        assert!(err.contains("disallowed character '$'"), "got: {err}");
    }

    #[test]
    fn env_assignment_detection() {
        assert!(looks_like_env_assignment("FOO=bar"));
        assert!(looks_like_env_assignment("_X=1"));
        assert!(!looks_like_env_assignment("=bar"));
        assert!(!looks_like_env_assignment("foo"));
        assert!(!looks_like_env_assignment("1FOO=bar"));
        assert!(!looks_like_env_assignment("FO-O=bar"));
    }

    #[test]
    fn rewrite_sudo_basic() {
        let (out, found) = rewrite_real_sudo_invocations("sudo apt update");
        assert!(found);
        assert_eq!(out, "sudo -S -p '' apt update");
    }

    #[test]
    fn rewrite_sudo_after_chain() {
        let (out, found) = rewrite_real_sudo_invocations("cd /tmp && sudo make install");
        assert!(found);
        assert_eq!(out, "cd /tmp && sudo -S -p '' make install");
    }

    #[test]
    fn rewrite_sudo_ignores_text_mentions() {
        // "sudo" inside a quoted string / not at command start should be left.
        let (out, found) = rewrite_real_sudo_invocations("echo 'use sudo here'");
        assert!(!found);
        assert_eq!(out, "echo 'use sudo here'");
    }

    #[test]
    fn rewrite_sudo_after_env_assignment() {
        let (out, found) = rewrite_real_sudo_invocations("FOO=bar sudo run");
        assert!(found);
        assert_eq!(out, "FOO=bar sudo -S -p '' run");
    }

    #[test]
    fn rewrite_sudo_in_comment() {
        let (out, found) = rewrite_real_sudo_invocations("# sudo not run\nls");
        assert!(!found);
        assert_eq!(out, "# sudo not run\nls");
    }

    #[test]
    fn compound_background_rewrite() {
        let out = rewrite_compound_background("make && python3 -m http.server &");
        assert_eq!(out, "make && { python3 -m http.server & }");
    }

    #[test]
    fn compound_background_simple_amp_untouched() {
        let out = rewrite_compound_background("sleep 5 &");
        assert_eq!(out, "sleep 5 &");
    }

    #[test]
    fn compound_background_or_chain() {
        let out = rewrite_compound_background("a || b &");
        assert_eq!(out, "a || { b & }");
    }

    #[test]
    fn compound_background_redirect_amp_not_rewritten() {
        // `&>` redirect should not be treated as backgrounding.
        let out = rewrite_compound_background("a && b &> log");
        assert_eq!(out, "a && b &> log");
    }

    #[test]
    fn interpret_exit_codes() {
        assert_eq!(interpret_exit_code("ls", 0), None);
        assert_eq!(
            interpret_exit_code("grep foo file", 1).as_deref(),
            Some("No matches found (not an error)")
        );
        assert_eq!(
            interpret_exit_code("cat a | rg foo", 1).as_deref(),
            Some("No matches found (not an error)")
        );
        assert_eq!(
            interpret_exit_code("/usr/bin/diff a b", 1).as_deref(),
            Some("Files differ (expected, not an error)")
        );
        assert_eq!(
            interpret_exit_code("curl http://x", 6).as_deref(),
            Some("Could not resolve host")
        );
        assert_eq!(interpret_exit_code("grep foo", 2), None);
        assert_eq!(
            interpret_exit_code("FOO=bar git diff", 1).as_deref(),
            Some("Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)")
        );
    }

    #[test]
    fn help_version_detection() {
        assert!(looks_like_help_or_version_command("foo --help"));
        assert!(looks_like_help_or_version_command("foo -h"));
        assert!(looks_like_help_or_version_command("foo --version"));
        assert!(looks_like_help_or_version_command("foo -v"));
        assert!(!looks_like_help_or_version_command("foo bar"));
    }

    #[test]
    fn pipe_stdin_detection() {
        assert!(command_requires_pipe_stdin("gh auth login --with-token"));
        assert!(command_requires_pipe_stdin("gh   auth   login --with-token < tok"));
        assert!(!command_requires_pipe_stdin("gh auth login"));
        assert!(!command_requires_pipe_stdin("echo --with-token"));
    }

    #[test]
    fn foreground_guidance() {
        assert!(foreground_background_guidance("nohup server").unwrap().contains("nohup"));
        assert!(foreground_background_guidance("python app.py &").unwrap().contains("backgrounding"));
        assert!(foreground_background_guidance("server start & ").unwrap().contains("backgrounding"));
        assert!(foreground_background_guidance("npm run dev").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("npm start").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("docker compose up").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("uvicorn main:app").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("python3 -m http.server 8000").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("vite").unwrap().contains("long-lived"));
        assert!(foreground_background_guidance("ls -la").is_none());
        // help/version short-circuits even for long-lived-looking commands
        assert!(foreground_background_guidance("vite --help").is_none());
    }

    #[test]
    fn notification_conflict() {
        let (wp, note) = resolve_notification_flag_conflict(
            true,
            Some(vec!["x".to_string()]),
            true,
        );
        assert!(wp.is_none());
        assert!(note.contains("watch_patterns ignored"));

        let (wp, note) =
            resolve_notification_flag_conflict(false, Some(vec!["x".to_string()]), true);
        assert_eq!(wp, Some(vec!["x".to_string()]));
        assert_eq!(note, "");

        // background false -> no conflict even if both set
        let (wp, note) =
            resolve_notification_flag_conflict(true, Some(vec!["x".to_string()]), false);
        assert!(wp.is_some());
        assert_eq!(note, "");
    }

    #[test]
    fn task_id_resolution() {
        assert_eq!(resolve_container_task_id(None), "default");
        assert_eq!(resolve_container_task_id(Some("subagent-1")), "default");
        register_task_env_overrides("rl-task", json!({"docker_image": "x"}));
        assert_eq!(resolve_container_task_id(Some("rl-task")), "rl-task");
        clear_task_env_overrides("rl-task");
        assert_eq!(resolve_container_task_id(Some("rl-task")), "default");
    }

    #[test]
    fn truncate_keeps_head_tail() {
        let s: String = std::iter::repeat('a').take(100).collect();
        let out = truncate_output(&s, 40);
        assert!(out.contains("OUTPUT TRUNCATED"));
        assert!(out.contains("60 chars omitted out of 100 total"));
        assert!(out.starts_with("aaaaaaaaaaaaaaaa")); // 16 head chars (40*0.4)
        // unchanged when within limit
        assert_eq!(truncate_output("short", 40), "short");
    }

    #[test]
    fn sudo_failure_messaging() {
        let out = "sudo: a password is required";
        let enhanced = handle_sudo_failure(out, true, "/home/u/.hermes");
        assert!(enhanced.contains("add SUDO_PASSWORD to /home/u/.hermes/.env"));
        // non-gateway: unchanged
        assert_eq!(handle_sudo_failure(out, false, "/x"), out);
        // gateway but no sudo failure: unchanged
        assert_eq!(handle_sudo_failure("all good", true, "/x"), "all good");
    }

    #[test]
    fn vercel_disk_validation() {
        let r = check_vercel_sandbox_requirements("node24", 12345, true);
        assert!(!r.ok);
        assert!(r.error.unwrap().contains("does not support custom"));
        // default disk + missing package
        let r = check_vercel_sandbox_requirements("node24", 51200, false);
        assert!(!r.ok);
        assert!(r.error.unwrap().contains("pip install vercel"));
    }

    #[test]
    fn sudo_password_cache_scope_roundtrip() {
        reset_cached_sudo_passwords();
        unsafe {
            env::set_var("HERMES_SESSION_KEY", "sess-1");
        }
        set_cached_sudo_password("secret");
        assert_eq!(get_cached_sudo_password(), "secret");
        set_cached_sudo_password("");
        assert_eq!(get_cached_sudo_password(), "");
        unsafe {
            env::remove_var("HERMES_SESSION_KEY");
        }
        reset_cached_sudo_passwords();
    }

    #[test]
    fn transform_sudo_with_configured_password() {
        unsafe {
            env::set_var("SUDO_PASSWORD", "hunter2");
        }
        let ctx = SudoTransformCtx {
            sudo_nopasswd_works: false,
            interactive: false,
            prompt_password: None,
        };
        let r = transform_sudo_command(Some("sudo ls"), &ctx);
        assert_eq!(r.command.as_deref(), Some("sudo -S -p '' ls"));
        assert_eq!(r.sudo_stdin.as_deref(), Some("hunter2\n"));
        unsafe {
            env::remove_var("SUDO_PASSWORD");
        }
    }

    #[test]
    fn transform_sudo_no_sudo() {
        let ctx = SudoTransformCtx {
            sudo_nopasswd_works: false,
            interactive: false,
            prompt_password: None,
        };
        let r = transform_sudo_command(Some("ls -la"), &ctx);
        assert_eq!(r.command.as_deref(), Some("ls -la"));
        assert!(r.sudo_stdin.is_none());
    }

    #[test]
    fn transform_sudo_nopasswd_works_returns_original() {
        reset_cached_sudo_passwords();
        // ensure no configured/cached password
        unsafe {
            env::remove_var("SUDO_PASSWORD");
        }
        let ctx = SudoTransformCtx {
            sudo_nopasswd_works: true,
            interactive: false,
            prompt_password: None,
        };
        let r = transform_sudo_command(Some("sudo ls"), &ctx);
        // original command unchanged, no stdin
        assert_eq!(r.command.as_deref(), Some("sudo ls"));
        assert!(r.sudo_stdin.is_none());
    }

    #[test]
    fn schema_has_required_command() {
        let s = terminal_schema();
        assert_eq!(s["name"], "terminal");
        assert_eq!(s["parameters"]["required"][0], "command");
        assert_eq!(s["parameters"]["properties"]["background"]["default"], false);
    }

    #[test]
    fn config_defaults_local() {
        // Pin all relevant env vars to defaults for determinism.
        unsafe {
            env::remove_var("TERMINAL_ENV");
            env::remove_var("TERMINAL_CWD");
            env::remove_var("TERMINAL_TIMEOUT");
            env::remove_var("TERMINAL_DOCKER_FORWARD_ENV");
            env::remove_var("TERMINAL_DOCKER_VOLUMES");
            env::remove_var("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE");
        }
        let cfg = get_env_config_with_cwd("/work/here").unwrap();
        assert_eq!(cfg.env_type, "local");
        assert_eq!(cfg.cwd, "/work/here");
        assert_eq!(cfg.timeout, 180);
        assert_eq!(cfg.lifetime_seconds, 300);
        assert_eq!(cfg.docker_image, DEFAULT_IMAGE);
        assert_eq!(cfg.singularity_image, format!("docker://{DEFAULT_IMAGE}"));
        assert_eq!(cfg.container_disk, 51200);
        assert!(cfg.container_persistent);
    }

    #[test]
    fn config_rejects_bad_timeout() {
        unsafe {
            env::set_var("TERMINAL_TIMEOUT", "5m");
        }
        let err = get_env_config_with_cwd("/x").unwrap_err();
        assert!(err.0.contains("Invalid value for TERMINAL_TIMEOUT"));
        unsafe {
            env::remove_var("TERMINAL_TIMEOUT");
        }
    }

    #[test]
    fn config_docker_host_cwd_ignored() {
        unsafe {
            env::set_var("TERMINAL_ENV", "docker");
            env::set_var("TERMINAL_CWD", "/Users/me/project");
            env::remove_var("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE");
        }
        let cfg = get_env_config_with_cwd("/x").unwrap();
        // host path discarded -> falls back to default /root
        assert_eq!(cfg.cwd, "/root");
        unsafe {
            env::remove_var("TERMINAL_ENV");
            env::remove_var("TERMINAL_CWD");
        }
    }
}
