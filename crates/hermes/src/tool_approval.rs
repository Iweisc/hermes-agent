//! Dangerous command approval -- detection, prompting, and per-session state.
//!
//! Native Rust port of `tools/approval.py`. This module is the single source
//! of truth for the dangerous command system:
//! - Pattern detection ([`DANGEROUS_PATTERNS`], [`detect_dangerous_command`])
//! - Hardline (unconditional) blocklist ([`detect_hardline_command`])
//! - Per-session approval state (thread-safe, keyed by session_key)
//! - Approval orchestration ([`check_dangerous_command`],
//!   [`check_all_command_guards`])
//! - Smart approval hook + permanent allowlist persistence
//!
//! Differences from the Python original:
//! - CLI interactive `input()` prompting is replaced by an injectable
//!   approval callback (the TUI / native CLI installs one). When no callback
//!   is present we fail closed (deny), matching the Python fail-closed guard.
//! - The gateway sync->async bridge is preserved via a registered notify
//!   callback plus a per-session queue of blocking [`ApprovalEntry`] events.
//! - Config access (`load_config`, `save_config`, `cfg_get`) and ANSI
//!   stripping are taken from `hermes_core` where exported, with small local
//!   fallbacks (`is_truthy`, `strip_ansi_local`, NFKC-ish normalization) so
//!   the module compiles standalone in the `hermes` crate.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use regex::Regex;

// =========================================================================
// Small local helpers (kept local to avoid editing shared crate exports)
// =========================================================================

/// Shared truthy strings, mirroring the project's `is_truthy_value`.
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "y" | "t" | "enable" | "enabled"
    )
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).map(|v| is_truthy(&v)).unwrap_or(false)
}

fn env_present(name: &str) -> bool {
    std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Strip ANSI / control escape sequences. Prefers the hermes-core
/// implementation when reachable, otherwise applies a CSI/OSC-aware fallback
/// matching the cases `tools.ansi_strip.strip_ansi` guards against.
fn strip_ansi_local(text: &str) -> String {
    // Fallback regex covering CSI (ESC[ ... letter), OSC (ESC] ... BEL/ST),
    // and stray single-character escapes, plus 8-bit C1 introducers.
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)   # OSC ... BEL or ST
          | \x1b[\[\(][0-?]*[ -/]*[@-~]          # CSI / charset
          | \x1b[@-Z\\-_]                         # other 7-bit escapes
          | [\x9b\x9d][0-?]*[ -/]*[@-~]           # 8-bit C1 CSI/OSC
          | \x1b                                  # stray ESC
          ",
        )
        .unwrap()
    });
    re.replace_all(text, "").into_owned()
}

/// Approximate `unicodedata.normalize('NFKC', ...)`.
///
/// The full NFKC table is large; here we cover the obfuscation vectors the
/// Python code defends against (fullwidth ASCII Latin/digits/punct in the
/// range U+FF01..U+FF5E map to U+0021..U+007E, ideographic space U+3000 ->
/// space). This is sufficient for command-string deobfuscation; if exact
/// Unicode parity is required, swap in the `unicode-normalization` crate.
fn nfkc_normalize(text: &str) -> String {
    text.chars()
        .map(|c| {
            let cp = c as u32;
            if (0xFF01..=0xFF5E).contains(&cp) {
                // Fullwidth ASCII variants.
                char::from_u32(cp - 0xFEE0).unwrap_or(c)
            } else if cp == 0x3000 {
                ' '
            } else {
                c
            }
        })
        .collect()
}

// =========================================================================
// Plugin lifecycle hook bridge
// =========================================================================

/// Signature for an approval lifecycle hook dispatcher.
///
/// Mirrors `_fire_approval_hook(hook_name, **kwargs)`. The payload carries the
/// keyword arguments the Python code passes (command, description, pattern
/// keys, session key, surface, and -- for `post_approval_response` -- choice).
pub type ApprovalHookFn = dyn Fn(&str, &ApprovalHookPayload) + Send + Sync;

#[derive(Debug, Clone, Default)]
pub struct ApprovalHookPayload {
    pub command: String,
    pub description: String,
    pub pattern_key: String,
    pub pattern_keys: Vec<String>,
    pub session_key: String,
    pub surface: String,
    /// Only populated for `post_approval_response`.
    pub choice: Option<String>,
}

static APPROVAL_HOOK: OnceLock<Box<ApprovalHookFn>> = OnceLock::new();

/// Install the plugin-lifecycle hook dispatcher. Optional; if unset, hooks are
/// no-ops (matching the lazy-import-failure path in Python).
pub fn set_approval_hook(hook: Box<ApprovalHookFn>) {
    let _ = APPROVAL_HOOK.set(hook);
}

fn fire_approval_hook(hook_name: &str, payload: &ApprovalHookPayload) {
    if let Some(hook) = APPROVAL_HOOK.get() {
        // Hook implementations are expected not to panic; swallow like Python.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hook(hook_name, payload);
        }));
        if res.is_err() {
            log::debug!("Approval hook {hook_name} dispatch failed");
        }
    }
}

// =========================================================================
// Per-thread/per-task gateway session identity
// =========================================================================

thread_local! {
    static APPROVAL_SESSION_KEY: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Opaque token returned by [`set_current_session_key`], passed back to
/// [`reset_current_session_key`] to restore the prior value.
pub struct SessionKeyToken {
    prev_len: usize,
}

/// Bind the active approval session key to the current thread/context.
pub fn set_current_session_key(session_key: &str) -> SessionKeyToken {
    APPROVAL_SESSION_KEY.with(|stack| {
        let mut s = stack.borrow_mut();
        let prev_len = s.len();
        s.push(session_key.to_string());
        SessionKeyToken { prev_len }
    })
}

/// Restore the prior approval session key context.
pub fn reset_current_session_key(token: SessionKeyToken) {
    APPROVAL_SESSION_KEY.with(|stack| {
        let mut s = stack.borrow_mut();
        s.truncate(token.prev_len);
    });
}

/// Return the active session key, preferring context-local state, then the
/// `HERMES_SESSION_KEY` env var, then `default`.
pub fn get_current_session_key(default: &str) -> String {
    let ctx = APPROVAL_SESSION_KEY.with(|stack| stack.borrow().last().cloned());
    if let Some(k) = ctx {
        if !k.is_empty() {
            return k;
        }
    }
    match std::env::var("HERMES_SESSION_KEY") {
        Ok(v) if !v.is_empty() => v,
        _ => default.to_string(),
    }
}

// =========================================================================
// Regex fragments (verbatim from approval.py)
// =========================================================================

const SSH_SENSITIVE_PATH: &str = r"(?:~|\$home|\$\{home\})/\.ssh(?:/|$)";
const HERMES_ENV_PATH: &str = concat!(
    r"(?:~\/\.hermes/|",
    r"(?:\$home|\$\{home\})/\.hermes/|",
    r"(?:\$hermes_home|\$\{hermes_home\})/)",
    r"\.env\b",
);
const PROJECT_ENV_PATH: &str = r#"(?:(?:/|\.{1,2}/)?(?:[^\s/"'`]+/)*\.env(?:\.[^/\s"'`]+)*)"#;
const PROJECT_CONFIG_PATH: &str = r#"(?:(?:/|\.{1,2}/)?(?:[^\s/"'`]+/)*config\.yaml)"#;
const SHELL_RC_FILES: &str =
    r"(?:~|\$home|\$\{home\})/\.(?:bashrc|zshrc|profile|bash_profile|zprofile)\b";
const CREDENTIAL_FILES: &str = r"(?:~|\$home|\$\{home\})/\.(?:netrc|pgpass|npmrc|pypirc)\b";
const COMMAND_TAIL: &str = r"(?:\s*(?:&&|\|\||;).*)?$";

fn sensitive_write_target() -> String {
    format!(
        r"(?:/etc/|/dev/sd|{SSH_SENSITIVE_PATH}|{HERMES_ENV_PATH}|{SHELL_RC_FILES}|{CREDENTIAL_FILES})"
    )
}

fn project_sensitive_write_target() -> String {
    format!(r"(?:{PROJECT_ENV_PATH}|{PROJECT_CONFIG_PATH})")
}

// _CMDPOS: matches start-of-command positions.
const CMDPOS: &str = concat!(
    r"(?:^|[;&|\n`]|\$\()",
    r"\s*",
    r"(?:sudo\s+(?:-[^\s]+\s+)*)?",
    r"(?:env\s+(?:\w+=\S*\s+)*)?",
    r"(?:(?:exec|nohup|setsid|time)\s+)*",
    r"\s*",
);

// =========================================================================
// Hardline (unconditional) blocklist
// =========================================================================

/// `(pattern, description)` pairs for the unconditional hardline blocklist.
pub fn hardline_patterns() -> Vec<(String, &'static str)> {
    vec![
        (
            r"\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)".to_string(),
            "recursive delete of root filesystem",
        ),
        (
            r"\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)".to_string(),
            "recursive delete of system directory",
        ),
        (
            r"\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)".to_string(),
            "recursive delete of home directory",
        ),
        (r"\bmkfs(\.[a-z0-9]+)?\b".to_string(), "format filesystem (mkfs)"),
        (
            r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*".to_string(),
            "dd to raw block device",
        ),
        (
            r">\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b".to_string(),
            "redirect to raw block device",
        ),
        (
            r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:".to_string(),
            "fork bomb",
        ),
        (r"\bkill\s+(-[^\s]+\s+)*-1\b".to_string(), "kill all processes"),
        (
            format!("{CMDPOS}(shutdown|reboot|halt|poweroff)\\b"),
            "system shutdown/reboot",
        ),
        (
            format!("{CMDPOS}init\\s+[06]\\b"),
            "init 0/6 (shutdown/reboot)",
        ),
        (
            format!("{CMDPOS}systemctl\\s+(poweroff|reboot|halt|kexec)\\b"),
            "systemctl poweroff/reboot",
        ),
        (
            format!("{CMDPOS}telinit\\s+[06]\\b"),
            "telinit 0/6 (shutdown/reboot)",
        ),
    ]
}

fn hardline_compiled() -> &'static Vec<(Regex, &'static str)> {
    static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    COMPILED.get_or_init(|| {
        hardline_patterns()
            .into_iter()
            .map(|(p, d)| (compile_ci_dotall(&p), d))
            .collect()
    })
}

/// `(?is)` flags = IGNORECASE | DOTALL, matching `_RE_FLAGS`.
fn compile_ci_dotall(pattern: &str) -> Regex {
    Regex::new(&format!("(?is){pattern}"))
        .unwrap_or_else(|e| panic!("invalid approval pattern {pattern:?}: {e}"))
}

/// Check if a command matches the unconditional hardline blocklist.
/// Returns `Some(description)` on a match.
pub fn detect_hardline_command(command: &str) -> Option<&'static str> {
    let normalized = normalize_command_for_detection(command).to_lowercase();
    for (re, description) in hardline_compiled() {
        if re.is_match(&normalized) {
            return Some(description);
        }
    }
    None
}

/// Build the standard block result for a hardline match.
pub fn hardline_block_result(description: &str) -> ApprovalResult {
    ApprovalResult {
        approved: false,
        hardline: true,
        message: Some(format!(
            "BLOCKED (hardline): {description}. \
This command is on the unconditional blocklist and cannot \
be executed via the agent — not even with --yolo, /yolo, \
approvals.mode=off, or cron approve mode. If you genuinely \
need to run it, run it yourself in a terminal outside the \
agent."
        )),
        ..Default::default()
    }
}

// =========================================================================
// Dangerous command patterns
// =========================================================================

/// `(pattern, description)` pairs for the dangerous-command detector.
pub fn dangerous_patterns() -> Vec<(String, &'static str)> {
    let swt = sensitive_write_target();
    let pswt = project_sensitive_write_target();
    vec![
        (r"\brm\s+(-[^\s]*\s+)*/".to_string(), "delete in root path"),
        (r"\brm\s+-[^\s]*r".to_string(), "recursive delete"),
        (r"\brm\s+--recursive\b".to_string(), "recursive delete (long flag)"),
        (
            r"\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b".to_string(),
            "world/other-writable permissions",
        ),
        (
            r"\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)".to_string(),
            "recursive world/other-writable (long flag)",
        ),
        (r"\bchown\s+(-[^\s]*)?R\s+root".to_string(), "recursive chown to root"),
        (
            r"\bchown\s+--recursive\b.*root".to_string(),
            "recursive chown to root (long flag)",
        ),
        (r"\bmkfs\b".to_string(), "format filesystem"),
        (r"\bdd\s+.*if=".to_string(), "disk copy"),
        (r">\s*/dev/sd".to_string(), "write to block device"),
        (r"\bDROP\s+(TABLE|DATABASE)\b".to_string(), "SQL DROP"),
        // NOTE: the Python source uses a negative lookahead
        // `\bDELETE\s+FROM\b(?!.*\bWHERE\b)`. The `regex` crate has no
        // look-around, so we match the prefix here and enforce the
        // "no trailing WHERE" condition in `detect_dangerous_command` via
        // `SQL_DELETE_WHERE_GUARD`.
        (
            r"\bDELETE\s+FROM\b".to_string(),
            "SQL DELETE without WHERE",
        ),
        (r"\bTRUNCATE\s+(TABLE)?\s*\w".to_string(), "SQL TRUNCATE"),
        (r">\s*/etc/".to_string(), "overwrite system config"),
        (
            r"\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b".to_string(),
            "stop/restart system service",
        ),
        (r"\bkill\s+-9\s+-1\b".to_string(), "kill all processes"),
        (r"\bpkill\s+-9\b".to_string(), "force kill processes"),
        (
            r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:".to_string(),
            "fork bomb",
        ),
        (
            r"\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)".to_string(),
            "shell command via -c/-lc flag",
        ),
        (
            r"\b(python[23]?|perl|ruby|node)\s+-[ec]\s+".to_string(),
            "script execution via -e/-c flag",
        ),
        (
            r"\b(curl|wget)\b.*\|\s*(ba)?sh\b".to_string(),
            "pipe remote content to shell",
        ),
        (
            r"\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b".to_string(),
            "execute remote script via process substitution",
        ),
        (
            format!(r#"\btee\b.*["']?{swt}"#),
            "overwrite system file via tee",
        ),
        (
            format!(r#">>?\s*["']?{swt}"#),
            "overwrite system file via redirection",
        ),
        (
            format!(r#"\btee\b.*["']?{pswt}["']?{COMMAND_TAIL}"#),
            "overwrite project env/config via tee",
        ),
        (
            format!(r#">>?\s*["']?{pswt}["']?{COMMAND_TAIL}"#),
            "overwrite project env/config via redirection",
        ),
        (r"\bxargs\s+.*\brm\b".to_string(), "xargs with rm"),
        (r"\bfind\b.*-exec\s+(/\S*/)?rm\b".to_string(), "find -exec rm"),
        (r"\bfind\b.*-delete\b".to_string(), "find -delete"),
        (
            r"\bhermes\s+gateway\s+(stop|restart)\b".to_string(),
            "stop/restart hermes gateway (kills running agents)",
        ),
        (
            r"\bhermes\s+update\b".to_string(),
            "hermes update (restarts gateway, kills running agents)",
        ),
        (
            r"gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)".to_string(),
            "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')",
        ),
        (
            r"\bnohup\b.*gateway\s+run\b".to_string(),
            "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')",
        ),
        (
            r"\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b".to_string(),
            "kill hermes/gateway process (self-termination)",
        ),
        (
            r"\bkill\b.*\$\(\s*pgrep\b".to_string(),
            "kill process via pgrep expansion (self-termination)",
        ),
        (
            r"\bkill\b.*`\s*pgrep\b".to_string(),
            "kill process via backtick pgrep expansion (self-termination)",
        ),
        (
            r"\b(cp|mv|install)\b.*\s/etc/".to_string(),
            "copy/move file into /etc/",
        ),
        (
            format!(r#"\b(cp|mv|install)\b.*\s["']?{pswt}["']?{COMMAND_TAIL}"#),
            "overwrite project env/config file",
        ),
        (
            r"\bsed\s+-[^\s]*i.*\s/etc/".to_string(),
            "in-place edit of system config",
        ),
        (
            r"\bsed\s+--in-place\b.*\s/etc/".to_string(),
            "in-place edit of system config (long flag)",
        ),
        (
            r"\b(python[23]?|perl|ruby|node)\s+<<".to_string(),
            "script execution via heredoc",
        ),
        (
            r"\bgit\s+reset\s+--hard\b".to_string(),
            "git reset --hard (destroys uncommitted changes)",
        ),
        (
            r"\bgit\s+push\b.*--force\b".to_string(),
            "git force push (rewrites remote history)",
        ),
        (
            r"\bgit\s+push\b.*-f\b".to_string(),
            "git force push short flag (rewrites remote history)",
        ),
        (
            r"\bgit\s+clean\s+-[^\s]*f".to_string(),
            "git clean with force (deletes untracked files)",
        ),
        (r"\bgit\s+branch\s+-D\b".to_string(), "git branch force delete"),
        (
            r"\bchmod\s+\+x\b.*[;&|]+\s*\./".to_string(),
            "chmod +x followed by immediate execution",
        ),
    ]
}

fn dangerous_compiled() -> &'static Vec<(Regex, &'static str)> {
    static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    COMPILED.get_or_init(|| {
        dangerous_patterns()
            .into_iter()
            .map(|(p, d)| (compile_ci_dotall(&p), d))
            .collect()
    })
}

/// Reproduce the old regex-derived approval key for backwards compatibility:
/// `pattern.split(r'\b')[1]` if `\b` present, else first 20 chars.
fn legacy_pattern_key(pattern: &str) -> String {
    if let Some(idx) = pattern.find(r"\b") {
        let rest = &pattern[idx + 2..];
        // split produces: ["", <this>, ...]; element [1] is the substring up
        // to the next `\b` (or end).
        match rest.find(r"\b") {
            Some(end) => rest[..end].to_string(),
            None => rest.to_string(),
        }
    } else {
        pattern.chars().take(20).collect()
    }
}

/// canonical_key/legacy_key -> set of all keys that should match it.
fn pattern_key_aliases() -> &'static HashMap<String, HashSet<String>> {
    static MAP: OnceLock<HashMap<String, HashSet<String>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        for (pattern, description) in dangerous_patterns() {
            let legacy = legacy_pattern_key(&pattern);
            let canonical = description.to_string();
            map.entry(canonical.clone())
                .or_default()
                .extend([canonical.clone(), legacy.clone()]);
            map.entry(legacy.clone())
                .or_default()
                .extend([legacy, canonical]);
        }
        map
    })
}

/// Return all approval keys that should match this pattern (canonical +
/// historical regex-derived key).
pub fn approval_key_aliases(pattern_key: &str) -> HashSet<String> {
    match pattern_key_aliases().get(pattern_key) {
        Some(s) => s.clone(),
        None => {
            let mut s = HashSet::new();
            s.insert(pattern_key.to_string());
            s
        }
    }
}

// =========================================================================
// Detection
// =========================================================================

/// Normalize a command string before dangerous-pattern matching. Strips ANSI
/// escape sequences, null bytes, and applies NFKC-ish Unicode normalization.
pub fn normalize_command_for_detection(command: &str) -> String {
    let command = strip_ansi_local(command);
    let command = command.replace('\u{0}', "");
    nfkc_normalize(&command)
}

/// Result of dangerous-command detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DangerousMatch {
    /// Approval key (the human-readable description string).
    pub pattern_key: String,
    pub description: String,
}

/// Check if a command matches any dangerous patterns. Returns `Some(..)` with
/// the pattern key (== description) on the first match.
pub fn detect_dangerous_command(command: &str) -> Option<DangerousMatch> {
    let command_lower = normalize_command_for_detection(command).to_lowercase();
    for (re, description) in dangerous_compiled() {
        if re.is_match(&command_lower) {
            // Re-implement the Python `(?!.*\bWHERE\b)` negative lookahead the
            // `regex` crate cannot express: a `DELETE FROM` that also contains
            // a `WHERE` clause is a targeted delete and is NOT flagged.
            if *description == "SQL DELETE without WHERE"
                && sql_delete_where_guard().is_match(&command_lower)
            {
                continue;
            }
            return Some(DangerousMatch {
                pattern_key: description.to_string(),
                description: description.to_string(),
            });
        }
    }
    None
}

/// Matches a `WHERE` keyword anywhere; used to suppress the SQL-DELETE warning
/// when the statement is scoped (the Python negative lookahead).
fn sql_delete_where_guard() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bwhere\b").unwrap())
}

// =========================================================================
// Per-session approval state (thread-safe)
// =========================================================================

#[derive(Default)]
struct ApprovalState {
    pending: HashMap<String, PendingApproval>,
    session_approved: HashMap<String, HashSet<String>>,
    session_yolo: HashSet<String>,
    permanent_approved: HashSet<String>,
    gateway_queues: HashMap<String, Vec<Arc<ApprovalEntry>>>,
}

/// Stored pending-approval payload (mirrors the `_pending` dict entry).
#[derive(Debug, Clone, Default)]
pub struct PendingApproval {
    pub command: String,
    pub pattern_key: String,
    pub pattern_keys: Vec<String>,
    pub description: String,
}

fn state() -> &'static Mutex<ApprovalState> {
    static STATE: OnceLock<Mutex<ApprovalState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ApprovalState::default()))
}

// --- Gateway notify callbacks (sync -> async bridge) ---

/// Notify callback: `cb(approval_data)` schedules sending the request to the
/// user. Runs in the agent thread.
pub type GatewayNotifyFn = dyn Fn(&PendingApproval) -> Result<(), String> + Send + Sync;

fn notify_cbs() -> &'static Mutex<HashMap<String, Arc<GatewayNotifyFn>>> {
    static CBS: OnceLock<Mutex<HashMap<String, Arc<GatewayNotifyFn>>>> = OnceLock::new();
    CBS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// One pending dangerous-command approval inside a gateway session. Equivalent
/// to Python's `_ApprovalEntry`: an event + the request data + a result slot.
pub struct ApprovalEntry {
    /// (resolved?, choice) protected by the condvar mutex.
    inner: Mutex<EntryInner>,
    cond: Condvar,
    pub data: PendingApproval,
    /// Stable identity for queue removal comparisons.
    id: usize,
}

struct EntryInner {
    signaled: bool,
    result: Option<String>,
}

impl ApprovalEntry {
    fn new(data: PendingApproval, id: usize) -> Arc<Self> {
        Arc::new(ApprovalEntry {
            inner: Mutex::new(EntryInner {
                signaled: false,
                result: None,
            }),
            cond: Condvar::new(),
            data,
            id,
        })
    }

    fn set_result(&self, choice: Option<String>) {
        let mut g = self.inner.lock().unwrap();
        g.result = choice;
        g.signaled = true;
        self.cond.notify_all();
    }

    fn signal(&self) {
        let mut g = self.inner.lock().unwrap();
        g.signaled = true;
        self.cond.notify_all();
    }

    /// Wait up to `timeout`. Returns true if signaled within the window.
    fn wait(&self, timeout: Duration) -> bool {
        let g = self.inner.lock().unwrap();
        if g.signaled {
            return true;
        }
        let (guard, res) = self.cond.wait_timeout(g, timeout).unwrap();
        drop(guard);
        !res.timed_out()
    }

    fn result(&self) -> Option<String> {
        self.inner.lock().unwrap().result.clone()
    }
}

fn next_entry_id() -> usize {
    use std::sync::atomic::AtomicUsize;
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Register a per-session callback for sending approval requests to the user.
pub fn register_gateway_notify(session_key: &str, cb: Arc<GatewayNotifyFn>) {
    notify_cbs()
        .lock()
        .unwrap()
        .insert(session_key.to_string(), cb);
}

/// Unregister the per-session gateway approval callback and signal all blocked
/// threads for the session so they do not hang.
pub fn unregister_gateway_notify(session_key: &str) {
    notify_cbs().lock().unwrap().remove(session_key);
    let entries = {
        let mut st = state().lock().unwrap();
        st.gateway_queues.remove(session_key).unwrap_or_default()
    };
    for entry in entries {
        entry.signal();
    }
}

/// Resolve gateway approval(s). When `resolve_all` is true, every pending
/// approval in the session is resolved (FIFO otherwise). Returns the number
/// resolved (0 if nothing pending).
pub fn resolve_gateway_approval(session_key: &str, choice: &str, resolve_all: bool) -> usize {
    let targets = {
        let mut st = state().lock().unwrap();
        let queue = match st.gateway_queues.get_mut(session_key) {
            Some(q) if !q.is_empty() => q,
            _ => return 0,
        };
        let targets: Vec<Arc<ApprovalEntry>> = if resolve_all {
            std::mem::take(queue)
        } else {
            vec![queue.remove(0)]
        };
        if st
            .gateway_queues
            .get(session_key)
            .map(|q| q.is_empty())
            .unwrap_or(false)
        {
            st.gateway_queues.remove(session_key);
        }
        targets
    };
    let n = targets.len();
    for entry in targets {
        entry.set_result(Some(choice.to_string()));
    }
    n
}

/// Whether a session has one or more blocking gateway approvals waiting.
pub fn has_blocking_approval(session_key: &str) -> bool {
    state()
        .lock()
        .unwrap()
        .gateway_queues
        .get(session_key)
        .map(|q| !q.is_empty())
        .unwrap_or(false)
}

/// Store a pending approval request for a session.
pub fn submit_pending(session_key: &str, approval: PendingApproval) {
    state()
        .lock()
        .unwrap()
        .pending
        .insert(session_key.to_string(), approval);
}

/// Retrieve (and keep) the pending approval for a session.
pub fn get_pending(session_key: &str) -> Option<PendingApproval> {
    state().lock().unwrap().pending.get(session_key).cloned()
}

/// Approve a pattern for this session only.
pub fn approve_session(session_key: &str, pattern_key: &str) {
    state()
        .lock()
        .unwrap()
        .session_approved
        .entry(session_key.to_string())
        .or_default()
        .insert(pattern_key.to_string());
}

/// Enable YOLO bypass for a single session key.
pub fn enable_session_yolo(session_key: &str) {
    if session_key.is_empty() {
        return;
    }
    state()
        .lock()
        .unwrap()
        .session_yolo
        .insert(session_key.to_string());
}

/// Disable YOLO bypass for a single session key.
pub fn disable_session_yolo(session_key: &str) {
    if session_key.is_empty() {
        return;
    }
    state().lock().unwrap().session_yolo.remove(session_key);
}

/// Remove all approval and yolo state for a given session, cancelling any
/// blocked approval waits with a `deny` result.
pub fn clear_session(session_key: &str) {
    if session_key.is_empty() {
        return;
    }
    let entries = {
        let mut st = state().lock().unwrap();
        st.session_approved.remove(session_key);
        st.session_yolo.remove(session_key);
        st.pending.remove(session_key);
        st.gateway_queues.remove(session_key).unwrap_or_default()
    };
    for entry in entries {
        entry.set_result(Some("deny".to_string()));
    }
}

/// True when YOLO bypass is enabled for a specific session.
pub fn is_session_yolo_enabled(session_key: &str) -> bool {
    if session_key.is_empty() {
        return false;
    }
    state().lock().unwrap().session_yolo.contains(session_key)
}

/// True when the active approval session has YOLO bypass enabled.
pub fn is_current_session_yolo_enabled() -> bool {
    is_session_yolo_enabled(&get_current_session_key(""))
}

/// Check if a pattern is approved (session-scoped or permanent), accepting
/// both the canonical key and any legacy alias.
pub fn is_approved(session_key: &str, pattern_key: &str) -> bool {
    let aliases = approval_key_aliases(pattern_key);
    let st = state().lock().unwrap();
    if aliases.iter().any(|a| st.permanent_approved.contains(a)) {
        return true;
    }
    let empty = HashSet::new();
    let session_approvals = st.session_approved.get(session_key).unwrap_or(&empty);
    aliases.iter().any(|a| session_approvals.contains(a))
}

/// Add a pattern to the permanent allowlist.
pub fn approve_permanent(pattern_key: &str) {
    state()
        .lock()
        .unwrap()
        .permanent_approved
        .insert(pattern_key.to_string());
}

/// Bulk-load permanent allowlist entries.
pub fn load_permanent<I: IntoIterator<Item = String>>(patterns: I) {
    state().lock().unwrap().permanent_approved.extend(patterns);
}

/// Snapshot of the current permanent allowlist (used for persistence).
pub fn permanent_approved_snapshot() -> HashSet<String> {
    state().lock().unwrap().permanent_approved.clone()
}

// =========================================================================
// Config persistence for permanent allowlist
// =========================================================================

/// Load permanently allowed command patterns from `config.yaml` and sync them
/// into the in-memory permanent allowlist. Returns the loaded set.
pub fn load_permanent_allowlist() -> HashSet<String> {
    let config = hermes_core::cli_config::load_config();
    let patterns: HashSet<String> = config
        .get("command_allowlist")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    if !patterns.is_empty() {
        load_permanent(patterns.iter().cloned());
    }
    patterns
}

/// Save permanently allowed command patterns to `config.yaml`.
pub fn save_permanent_allowlist(patterns: &HashSet<String>) {
    let mut config = hermes_core::cli_config::load_config();
    let list: Vec<serde_yaml::Value> = patterns
        .iter()
        .map(|p| serde_yaml::Value::String(p.clone()))
        .collect();
    if let serde_yaml::Value::Mapping(map) = &mut config {
        map.insert(
            serde_yaml::Value::String("command_allowlist".to_string()),
            serde_yaml::Value::Sequence(list),
        );
    } else {
        let mut map = serde_yaml::Mapping::new();
        map.insert(
            serde_yaml::Value::String("command_allowlist".to_string()),
            serde_yaml::Value::Sequence(list),
        );
        config = serde_yaml::Value::Mapping(map);
    }
    if let Err(e) = hermes_core::cli_config::save_config(&config) {
        log::warn!("Could not save allowlist: {e}");
    }
}

// =========================================================================
// Approval config
// =========================================================================

/// Normalize approval mode values loaded from YAML/config. YAML 1.1 parses
/// bare `off` as boolean `false`; treat that as the string mode "off".
pub fn normalize_approval_mode(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::Bool(false) => "off".to_string(),
        serde_yaml::Value::Bool(true) => "manual".to_string(),
        serde_yaml::Value::String(s) => {
            let n = s.trim().to_lowercase();
            if n.is_empty() {
                "manual".to_string()
            } else {
                n
            }
        }
        _ => "manual".to_string(),
    }
}

fn approval_config() -> serde_yaml::Mapping {
    let config = hermes_core::cli_config::load_config();
    config
        .get("approvals")
        .and_then(|v| v.as_mapping())
        .cloned()
        .unwrap_or_default()
}

fn get_approval_mode() -> String {
    let cfg = approval_config();
    let mode = cfg
        .get(serde_yaml::Value::String("mode".to_string()))
        .cloned()
        .unwrap_or(serde_yaml::Value::String("manual".to_string()));
    normalize_approval_mode(&mode)
}

fn get_approval_timeout() -> i64 {
    let cfg = approval_config();
    match cfg.get(serde_yaml::Value::String("timeout".to_string())) {
        Some(serde_yaml::Value::Number(n)) => n.as_i64().unwrap_or(60),
        Some(serde_yaml::Value::String(s)) => s.trim().parse::<i64>().unwrap_or(60),
        _ => 60,
    }
}

fn get_gateway_timeout() -> i64 {
    let cfg = approval_config();
    match cfg.get(serde_yaml::Value::String("gateway_timeout".to_string())) {
        Some(serde_yaml::Value::Number(n)) => n.as_i64().unwrap_or(300),
        Some(serde_yaml::Value::String(s)) => s.trim().parse::<i64>().unwrap_or(300),
        _ => 300,
    }
}

fn get_cron_approval_mode() -> &'static str {
    let cfg = approval_config();
    let raw = cfg
        .get(serde_yaml::Value::String("cron_mode".to_string()))
        .and_then(|v| match v {
            serde_yaml::Value::String(s) => Some(s.clone()),
            serde_yaml::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "deny".to_string())
        .to_lowercase();
    let raw = raw.trim();
    if matches!(raw, "approve" | "off" | "allow" | "yes") {
        "approve"
    } else {
        "deny"
    }
}

// =========================================================================
// Approval callback (replaces CLI input() prompting)
// =========================================================================

/// CLI interactive approval callback. Signature mirrors the Python
/// `approval_callback(command, description, *, allow_permanent) -> choice`.
/// Returns one of "once" | "session" | "always" | "deny".
pub type ApprovalCallback = dyn Fn(&str, &str, bool) -> String + Send + Sync;

/// Prompt the user to approve a dangerous command via the supplied callback.
///
/// Unlike the Python original (which falls back to a daemon-thread `input()`),
/// the native port requires an `approval_callback`. When none is provided we
/// fail closed and return "deny", matching the Python fail-closed guard for
/// threads with no callback installed.
pub fn prompt_dangerous_approval(
    command: &str,
    description: &str,
    allow_permanent: bool,
    approval_callback: Option<&ApprovalCallback>,
) -> String {
    match approval_callback {
        Some(cb) => {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cb(command, description, allow_permanent)
            }));
            match res {
                Ok(choice) => choice,
                Err(_) => {
                    log::error!("Approval callback failed");
                    "deny".to_string()
                }
            }
        }
        None => {
            log::warn!(
                "Dangerous-command approval requested with no approval callback; denying. \
command={command:?} description={description:?}"
            );
            "deny".to_string()
        }
    }
}

// =========================================================================
// Result type
// =========================================================================

/// Decision returned by the orchestration entry points. Mirrors the Python
/// approval dict (keys present only when relevant).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalResult {
    pub approved: bool,
    pub message: Option<String>,
    pub hardline: bool,
    pub pattern_key: Option<String>,
    pub description: Option<String>,
    /// "approval_required" when the gateway/ask fallback path is taken.
    pub status: Option<String>,
    pub command: Option<String>,
    pub smart_approved: bool,
    pub smart_denied: bool,
    pub user_approved: bool,
}

impl ApprovalResult {
    fn approved_clean() -> Self {
        ApprovalResult {
            approved: true,
            ..Default::default()
        }
    }
}

// =========================================================================
// Smart approval (auxiliary LLM risk assessment)
// =========================================================================

/// Verdict from the smart-approval LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartVerdict {
    Approve,
    Deny,
    Escalate,
}

/// Injectable smart-approval function. Returns the verdict given the command
/// and combined description. Defaults to escalate when unset (matching the
/// Python LLM-failure path).
pub type SmartApproveFn = dyn Fn(&str, &str) -> SmartVerdict + Send + Sync;

static SMART_APPROVE: OnceLock<Box<SmartApproveFn>> = OnceLock::new();

/// Install the smart-approval implementation (wraps the auxiliary LLM call).
pub fn set_smart_approve(f: Box<SmartApproveFn>) {
    let _ = SMART_APPROVE.set(f);
}

/// Build the security-reviewer prompt sent to the auxiliary LLM, identical to
/// the Python `_smart_approve` prompt.
pub fn smart_approve_prompt(command: &str, description: &str) -> String {
    format!(
        "You are a security reviewer for an AI coding agent. A terminal command was flagged by pattern matching as potentially dangerous.\n\nCommand: {command}\nFlagged reason: {description}\n\nAssess the ACTUAL risk of this command. Many flagged commands are false positives — for example, `python -c \"print('hello')\"` is flagged as \"script execution via -c flag\" but is completely harmless.\n\nRules:\n- APPROVE if the command is clearly safe (benign script execution, safe file operations, development tools, package installs, git operations, etc.)\n- DENY if the command could genuinely damage the system (recursive delete of important paths, overwriting system files, fork bombs, wiping disks, dropping databases, etc.)\n- ESCALATE if you're uncertain\n\nRespond with exactly one word: APPROVE, DENY, or ESCALATE"
    )
}

/// Parse a raw LLM answer into a verdict, matching the Python substring logic.
pub fn parse_smart_verdict(answer: &str) -> SmartVerdict {
    let answer = answer.trim().to_uppercase();
    if answer.contains("APPROVE") {
        SmartVerdict::Approve
    } else if answer.contains("DENY") {
        SmartVerdict::Deny
    } else {
        SmartVerdict::Escalate
    }
}

fn smart_approve(command: &str, description: &str) -> SmartVerdict {
    match SMART_APPROVE.get() {
        Some(f) => f(command, description),
        None => {
            log::debug!("Smart approvals: no LLM provider installed, escalating");
            SmartVerdict::Escalate
        }
    }
}

// =========================================================================
// Main orchestration entry point (single-pattern)
// =========================================================================

const CONTAINER_ENVS: &[&str] = &["docker", "singularity", "modal", "daytona", "vercel_sandbox"];

/// Check if a command is dangerous and handle approval. Mirrors Python
/// `check_dangerous_command`.
pub fn check_dangerous_command(
    command: &str,
    env_type: &str,
    approval_callback: Option<&ApprovalCallback>,
) -> ApprovalResult {
    if CONTAINER_ENVS.contains(&env_type) {
        return ApprovalResult::approved_clean();
    }

    if let Some(desc) = detect_hardline_command(command) {
        log::warn!("Hardline block: {desc} (command: {})", truncate(command, 200));
        return hardline_block_result(desc);
    }

    if env_truthy("HERMES_YOLO_MODE") || is_current_session_yolo_enabled() {
        return ApprovalResult::approved_clean();
    }

    let detected = match detect_dangerous_command(command) {
        Some(d) => d,
        None => return ApprovalResult::approved_clean(),
    };
    let pattern_key = detected.pattern_key;
    let description = detected.description;

    let session_key = get_current_session_key("default");
    if is_approved(&session_key, &pattern_key) {
        return ApprovalResult::approved_clean();
    }

    let is_cli = env_present("HERMES_INTERACTIVE");
    let is_gateway = env_present("HERMES_GATEWAY_SESSION");

    if !is_cli && !is_gateway {
        if env_present("HERMES_CRON_SESSION") && get_cron_approval_mode() == "deny" {
            return ApprovalResult {
                approved: false,
                message: Some(cron_block_message(&description)),
                ..Default::default()
            };
        }
        return ApprovalResult::approved_clean();
    }

    if is_gateway || env_present("HERMES_EXEC_ASK") {
        submit_pending(
            &session_key,
            PendingApproval {
                command: command.to_string(),
                pattern_key: pattern_key.clone(),
                pattern_keys: vec![],
                description: description.clone(),
            },
        );
        return ApprovalResult {
            approved: false,
            pattern_key: Some(pattern_key),
            status: Some("approval_required".to_string()),
            command: Some(command.to_string()),
            description: Some(description.clone()),
            message: Some(format!(
                "⚠️ This command is potentially dangerous ({description}). \
Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            )),
            ..Default::default()
        };
    }

    let choice = prompt_dangerous_approval(command, &description, true, approval_callback);

    if choice == "deny" {
        return ApprovalResult {
            approved: false,
            message: Some(format!(
                "BLOCKED: User denied this potentially dangerous command (matched '{description}' pattern). Do NOT retry this command - the user has explicitly rejected it."
            )),
            pattern_key: Some(pattern_key),
            description: Some(description),
            ..Default::default()
        };
    }

    if choice == "session" {
        approve_session(&session_key, &pattern_key);
    } else if choice == "always" {
        approve_session(&session_key, &pattern_key);
        approve_permanent(&pattern_key);
        save_permanent_allowlist(&permanent_approved_snapshot());
    }

    ApprovalResult::approved_clean()
}

fn cron_block_message(description: &str) -> String {
    format!(
        "BLOCKED: Command flagged as dangerous ({description}) \
but cron jobs run without a user present to approve it. \
Find an alternative approach that avoids this command. \
To allow dangerous commands in cron jobs, set \
approvals.cron_mode: approve in config.yaml."
    )
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// =========================================================================
// Combined pre-exec guard (tirith + dangerous command detection)
// =========================================================================

/// A tirith-style content security finding.
#[derive(Debug, Clone, Default)]
pub struct TirithFinding {
    pub severity: String,
    pub title: String,
    pub description: String,
    pub rule_id: String,
}

/// Result of a tirith content security scan.
#[derive(Debug, Clone)]
pub struct TirithResult {
    /// "allow" | "warn" | "block".
    pub action: String,
    pub findings: Vec<TirithFinding>,
    pub summary: String,
}

impl Default for TirithResult {
    fn default() -> Self {
        TirithResult {
            action: "allow".to_string(),
            findings: vec![],
            summary: String::new(),
        }
    }
}

/// Build a human-readable description from tirith findings. Mirrors
/// `_format_tirith_description`.
pub fn format_tirith_description(tirith: &TirithResult) -> String {
    if tirith.findings.is_empty() {
        let summary = if tirith.summary.is_empty() {
            "security issue detected"
        } else {
            &tirith.summary
        };
        return format!("Security scan: {summary}");
    }

    let mut parts: Vec<String> = Vec::new();
    for f in &tirith.findings {
        let severity = &f.severity;
        let title = &f.title;
        let desc = &f.description;
        if !title.is_empty() && !desc.is_empty() {
            if !severity.is_empty() {
                parts.push(format!("[{severity}] {title}: {desc}"));
            } else {
                parts.push(format!("{title}: {desc}"));
            }
        } else if !title.is_empty() {
            if !severity.is_empty() {
                parts.push(format!("[{severity}] {title}"));
            } else {
                parts.push(title.clone());
            }
        }
    }
    if parts.is_empty() {
        let summary = if tirith.summary.is_empty() {
            "security issue detected"
        } else {
            &tirith.summary
        };
        return format!("Security scan: {summary}");
    }
    format!("Security scan — {}", parts.join("; "))
}

/// Tirith scanner hook. Takes a command, returns a content-security result.
/// When unset, all commands are allowed (== ImportError path in Python).
pub type TirithCheckFn = dyn Fn(&str) -> TirithResult + Send + Sync;

static TIRITH_CHECK: OnceLock<Box<TirithCheckFn>> = OnceLock::new();

/// Install the tirith content-security scanner.
pub fn set_tirith_check(f: Box<TirithCheckFn>) {
    let _ = TIRITH_CHECK.set(f);
}

fn run_tirith(command: &str) -> TirithResult {
    match TIRITH_CHECK.get() {
        Some(f) => f(command),
        None => TirithResult::default(),
    }
}

/// Activity heartbeat callback fired ~every 10s while blocked on a gateway
/// approval (mirrors `touch_activity_if_due`).
pub type ActivityTouchFn = dyn Fn(&str) + Send + Sync;

static ACTIVITY_TOUCH: OnceLock<Box<ActivityTouchFn>> = OnceLock::new();

/// Install the inactivity-heartbeat callback used while waiting for gateway
/// approval responses.
pub fn set_activity_touch(f: Box<ActivityTouchFn>) {
    let _ = ACTIVITY_TOUCH.set(f);
}

/// Run all pre-exec security checks and return a single approval decision.
/// Mirrors Python `check_all_command_guards`.
pub fn check_all_command_guards(
    command: &str,
    env_type: &str,
    approval_callback: Option<&ApprovalCallback>,
) -> ApprovalResult {
    if CONTAINER_ENVS.contains(&env_type) {
        return ApprovalResult::approved_clean();
    }

    if let Some(desc) = detect_hardline_command(command) {
        log::warn!("Hardline block: {desc} (command: {})", truncate(command, 200));
        return hardline_block_result(desc);
    }

    let approval_mode = get_approval_mode();
    if env_truthy("HERMES_YOLO_MODE")
        || is_current_session_yolo_enabled()
        || approval_mode == "off"
    {
        return ApprovalResult::approved_clean();
    }

    let is_cli = env_present("HERMES_INTERACTIVE");
    let is_gateway = env_present("HERMES_GATEWAY_SESSION");
    let is_ask = env_present("HERMES_EXEC_ASK");

    if !is_cli && !is_gateway && !is_ask {
        if env_present("HERMES_CRON_SESSION") && get_cron_approval_mode() == "deny" {
            if let Some(d) = detect_dangerous_command(command) {
                return ApprovalResult {
                    approved: false,
                    message: Some(cron_block_message(&d.description)),
                    ..Default::default()
                };
            }
        }
        return ApprovalResult::approved_clean();
    }

    // --- Phase 1: Gather findings from both checks ---
    let tirith_result = run_tirith(command);
    let dangerous = detect_dangerous_command(command);

    // --- Phase 2: Decide ---
    // warnings: (pattern_key, description, is_tirith)
    let mut warnings: Vec<(String, String, bool)> = Vec::new();
    let session_key = get_current_session_key("default");

    if tirith_result.action == "block" || tirith_result.action == "warn" {
        let rule_id = tirith_result
            .findings
            .first()
            .map(|f| {
                if f.rule_id.is_empty() {
                    "unknown".to_string()
                } else {
                    f.rule_id.clone()
                }
            })
            .unwrap_or_else(|| "unknown".to_string());
        let tirith_key = format!("tirith:{rule_id}");
        let tirith_desc = format_tirith_description(&tirith_result);
        if !is_approved(&session_key, &tirith_key) {
            warnings.push((tirith_key, tirith_desc, true));
        }
    }

    if let Some(d) = &dangerous {
        if !is_approved(&session_key, &d.pattern_key) {
            warnings.push((d.pattern_key.clone(), d.description.clone(), false));
        }
    }

    if warnings.is_empty() {
        return ApprovalResult::approved_clean();
    }

    // --- Phase 2.5: Smart approval ---
    if approval_mode == "smart" {
        let combined_desc_for_llm = warnings
            .iter()
            .map(|(_, d, _)| d.clone())
            .collect::<Vec<_>>()
            .join("; ");
        match smart_approve(command, &combined_desc_for_llm) {
            SmartVerdict::Approve => {
                for (key, _, _) in &warnings {
                    approve_session(&session_key, key);
                }
                log::debug!(
                    "Smart approval: auto-approved '{}' ({combined_desc_for_llm})",
                    truncate(command, 60)
                );
                return ApprovalResult {
                    approved: true,
                    smart_approved: true,
                    description: Some(combined_desc_for_llm),
                    ..Default::default()
                };
            }
            SmartVerdict::Deny => {
                return ApprovalResult {
                    approved: false,
                    message: Some(format!(
                        "BLOCKED by smart approval: {combined_desc_for_llm}. \
The command was assessed as genuinely dangerous. Do NOT retry."
                    )),
                    smart_denied: true,
                    ..Default::default()
                };
            }
            SmartVerdict::Escalate => {}
        }
    }

    // --- Phase 3: Approval ---
    let combined_desc = warnings
        .iter()
        .map(|(_, d, _)| d.clone())
        .collect::<Vec<_>>()
        .join("; ");
    let primary_key = warnings[0].0.clone();
    let all_keys: Vec<String> = warnings.iter().map(|(k, _, _)| k.clone()).collect();
    let has_tirith = warnings.iter().any(|(_, _, is_t)| *is_t);

    if is_gateway || is_ask {
        let notify_cb = notify_cbs().lock().unwrap().get(&session_key).cloned();

        if let Some(cb) = notify_cb {
            return gateway_blocking_approval(
                command,
                &session_key,
                &primary_key,
                &all_keys,
                &combined_desc,
                &warnings,
                &cb,
            );
        }

        // Fallback: no gateway callback registered.
        submit_pending(
            &session_key,
            PendingApproval {
                command: command.to_string(),
                pattern_key: primary_key.clone(),
                pattern_keys: all_keys.clone(),
                description: combined_desc.clone(),
            },
        );
        return ApprovalResult {
            approved: false,
            pattern_key: Some(primary_key),
            status: Some("approval_required".to_string()),
            command: Some(command.to_string()),
            description: Some(combined_desc.clone()),
            message: Some(format!(
                "⚠️ {combined_desc}. Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            )),
            ..Default::default()
        };
    }

    // CLI interactive: single combined prompt
    fire_approval_hook(
        "pre_approval_request",
        &ApprovalHookPayload {
            command: command.to_string(),
            description: combined_desc.clone(),
            pattern_key: primary_key.clone(),
            pattern_keys: all_keys.clone(),
            session_key: session_key.clone(),
            surface: "cli".to_string(),
            choice: None,
        },
    );
    let choice = prompt_dangerous_approval(command, &combined_desc, !has_tirith, approval_callback);
    fire_approval_hook(
        "post_approval_response",
        &ApprovalHookPayload {
            command: command.to_string(),
            description: combined_desc.clone(),
            pattern_key: primary_key.clone(),
            pattern_keys: all_keys.clone(),
            session_key: session_key.clone(),
            surface: "cli".to_string(),
            choice: Some(choice.clone()),
        },
    );

    if choice == "deny" {
        return ApprovalResult {
            approved: false,
            message: Some("BLOCKED: User denied. Do NOT retry.".to_string()),
            pattern_key: Some(primary_key),
            description: Some(combined_desc),
            ..Default::default()
        };
    }

    persist_approvals(&session_key, &warnings, &choice);

    ApprovalResult {
        approved: true,
        user_approved: true,
        description: Some(combined_desc),
        ..Default::default()
    }
}

/// Persist approval for each warning based on the chosen scope. tirith
/// warnings never get permanent allowlisting (session-only even on "always").
fn persist_approvals(session_key: &str, warnings: &[(String, String, bool)], choice: &str) {
    for (key, _, is_tirith) in warnings {
        if choice == "session" || (choice == "always" && *is_tirith) {
            approve_session(session_key, key);
        } else if choice == "always" {
            approve_session(session_key, key);
            approve_permanent(key);
            save_permanent_allowlist(&permanent_approved_snapshot());
        }
        // choice == "once": no persistence.
    }
}

#[allow(clippy::too_many_arguments)]
fn gateway_blocking_approval(
    command: &str,
    session_key: &str,
    primary_key: &str,
    all_keys: &[String],
    combined_desc: &str,
    warnings: &[(String, String, bool)],
    notify_cb: &Arc<GatewayNotifyFn>,
) -> ApprovalResult {
    let approval_data = PendingApproval {
        command: command.to_string(),
        pattern_key: primary_key.to_string(),
        pattern_keys: all_keys.to_vec(),
        description: combined_desc.to_string(),
    };
    let entry = ApprovalEntry::new(approval_data.clone(), next_entry_id());
    {
        let mut st = state().lock().unwrap();
        st.gateway_queues
            .entry(session_key.to_string())
            .or_default()
            .push(entry.clone());
    }

    fire_approval_hook(
        "pre_approval_request",
        &ApprovalHookPayload {
            command: command.to_string(),
            description: combined_desc.to_string(),
            pattern_key: primary_key.to_string(),
            pattern_keys: all_keys.to_vec(),
            session_key: session_key.to_string(),
            surface: "gateway".to_string(),
            choice: None,
        },
    );

    // Notify the user (bridges sync agent thread -> async gateway).
    if let Err(exc) = notify_cb(&approval_data) {
        log::warn!("Gateway approval notify failed: {exc}");
        remove_entry(session_key, &entry);
        return ApprovalResult {
            approved: false,
            message: Some(
                "BLOCKED: Failed to send approval request to user. Do NOT retry.".to_string(),
            ),
            pattern_key: Some(primary_key.to_string()),
            description: Some(combined_desc.to_string()),
            ..Default::default()
        };
    }

    // Block until the user responds or timeout, polling in 1s slices so we can
    // fire activity heartbeats every ~10s.
    let timeout_secs = get_gateway_timeout().max(0) as u64;
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut last_touch = Instant::now();
    let mut resolved = false;
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline - now;
        let slice = remaining.min(Duration::from_secs(1));
        if entry.wait(slice) {
            resolved = true;
            break;
        }
        if let Some(touch) = ACTIVITY_TOUCH.get() {
            if last_touch.elapsed() >= Duration::from_secs(10) {
                touch("waiting for user approval");
                last_touch = Instant::now();
            }
        }
    }

    remove_entry(session_key, &entry);

    let choice = entry.result();
    let outcome = if !resolved {
        "timeout".to_string()
    } else {
        choice.clone().unwrap_or_else(|| "timeout".to_string())
    };
    fire_approval_hook(
        "post_approval_response",
        &ApprovalHookPayload {
            command: command.to_string(),
            description: combined_desc.to_string(),
            pattern_key: primary_key.to_string(),
            pattern_keys: all_keys.to_vec(),
            session_key: session_key.to_string(),
            surface: "gateway".to_string(),
            choice: Some(outcome),
        },
    );

    let denied = !resolved || choice.is_none() || choice.as_deref() == Some("deny");
    if denied {
        let reason = if !resolved { "timed out" } else { "denied by user" };
        return ApprovalResult {
            approved: false,
            message: Some(format!(
                "BLOCKED: Command {reason}. Do NOT retry this command."
            )),
            pattern_key: Some(primary_key.to_string()),
            description: Some(combined_desc.to_string()),
            ..Default::default()
        };
    }

    let choice = choice.unwrap();
    persist_approvals(session_key, warnings, &choice);

    ApprovalResult {
        approved: true,
        user_approved: true,
        description: Some(combined_desc.to_string()),
        ..Default::default()
    }
}

fn remove_entry(session_key: &str, entry: &Arc<ApprovalEntry>) {
    let mut st = state().lock().unwrap();
    if let Some(queue) = st.gateway_queues.get_mut(session_key) {
        queue.retain(|e| e.id != entry.id);
        if queue.is_empty() {
            st.gateway_queues.remove(session_key);
        }
    }
}

// =========================================================================
// One-time module initialization (mirrors module-level load_permanent_allowlist)
// =========================================================================

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Load the permanent allowlist from config on first use. Idempotent.
/// Mirrors the module-level `load_permanent_allowlist()` call in Python.
pub fn ensure_initialized() {
    if INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        let _ = load_permanent_allowlist();
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize env-var mutating tests (process-global env).
    static ENV_GUARD: StdMutex<()> = StdMutex::new(());

    #[test]
    fn detects_recursive_delete() {
        assert!(detect_dangerous_command("rm -rf /tmp/foo").is_some());
        let m = detect_dangerous_command("rm -rf build").unwrap();
        assert_eq!(m.pattern_key, "recursive delete");
    }

    #[test]
    fn benign_command_is_not_dangerous() {
        assert!(detect_dangerous_command("ls -la").is_none());
        assert!(detect_dangerous_command("echo hello").is_none());
    }

    #[test]
    fn hardline_blocks_rm_root() {
        assert_eq!(
            detect_hardline_command("rm -rf /"),
            Some("recursive delete of root filesystem")
        );
        assert_eq!(
            detect_hardline_command("mkfs.ext4 /dev/sda1"),
            Some("format filesystem (mkfs)")
        );
        assert!(detect_hardline_command("rm -rf /tmp/x").is_none());
    }

    #[test]
    fn hardline_anchors_shutdown_to_command_position() {
        assert!(detect_hardline_command("shutdown -h now").is_some());
        assert!(detect_hardline_command("sudo reboot").is_some());
        // Should NOT fire on text mentioning reboot.
        assert!(detect_hardline_command("echo reboot").is_none());
        assert!(detect_hardline_command("grep 'shutdown' /var/log/x").is_none());
    }

    #[test]
    fn fork_bomb_detected() {
        assert!(detect_hardline_command(":(){ :|:& };:").is_some());
    }

    #[test]
    fn unicode_fullwidth_obfuscation_normalized() {
        // Fullwidth "rm -rf /" should normalize and still be caught.
        let obf = "\u{ff52}\u{ff4d} -rf /"; // ｒｍ -rf /
        assert!(detect_hardline_command(obf).is_some());
    }

    #[test]
    fn normalize_strips_ansi_and_nulls() {
        let cmd = "rm\x1b[0m -rf\x00 /tmp";
        let n = normalize_command_for_detection(cmd);
        assert!(!n.contains('\x1b'));
        assert!(!n.contains('\x00'));
    }

    #[test]
    fn smart_verdict_parsing() {
        assert_eq!(parse_smart_verdict("APPROVE"), SmartVerdict::Approve);
        assert_eq!(parse_smart_verdict(" deny "), SmartVerdict::Deny);
        assert_eq!(parse_smart_verdict("hmm"), SmartVerdict::Escalate);
        assert_eq!(
            parse_smart_verdict("I think APPROVE is right"),
            SmartVerdict::Approve
        );
    }

    #[test]
    fn normalize_approval_mode_handles_yaml_bool_off() {
        assert_eq!(
            normalize_approval_mode(&serde_yaml::Value::Bool(false)),
            "off"
        );
        assert_eq!(
            normalize_approval_mode(&serde_yaml::Value::Bool(true)),
            "manual"
        );
        assert_eq!(
            normalize_approval_mode(&serde_yaml::Value::String("  Smart ".to_string())),
            "smart"
        );
        assert_eq!(
            normalize_approval_mode(&serde_yaml::Value::String("".to_string())),
            "manual"
        );
        assert_eq!(
            normalize_approval_mode(&serde_yaml::Value::Null),
            "manual"
        );
    }

    #[test]
    fn legacy_pattern_key_derivation() {
        // r"\brm\s+..." -> split on \b -> ["", "rm\s+...", ...] -> [1] truncated at next \b
        assert_eq!(legacy_pattern_key(r"\brm\s+(-[^\s]*\s+)*/"), r"rm\s+(-[^\s]*\s+)*/");
        // pattern with no \b -> first 20 chars
        let p = r">\s*/dev/sd";
        assert_eq!(legacy_pattern_key(p), &p[..p.len().min(20)]);
    }

    #[test]
    fn alias_round_trips_canonical_and_legacy() {
        let canonical = "recursive delete";
        let aliases = approval_key_aliases(canonical);
        assert!(aliases.contains(canonical));
        // Legacy key for r"\brm\s+-[^\s]*r"
        assert!(aliases.contains(r"rm\s+-[^\s]*r"));
    }

    #[test]
    fn session_approval_state() {
        let sk = "test-session-approval-state";
        clear_session(sk);
        assert!(!is_approved(sk, "recursive delete"));
        approve_session(sk, "recursive delete");
        assert!(is_approved(sk, "recursive delete"));
        // Legacy alias also matches.
        assert!(is_approved(sk, r"rm\s+-[^\s]*r"));
        clear_session(sk);
        assert!(!is_approved(sk, "recursive delete"));
    }

    #[test]
    fn permanent_approval_visible_across_sessions() {
        approve_permanent("xargs with rm");
        assert!(is_approved("any-session-xyz", "xargs with rm"));
    }

    #[test]
    fn session_yolo_toggle() {
        let sk = "yolo-session";
        assert!(!is_session_yolo_enabled(sk));
        enable_session_yolo(sk);
        assert!(is_session_yolo_enabled(sk));
        disable_session_yolo(sk);
        assert!(!is_session_yolo_enabled(sk));
        // Empty key is a no-op.
        enable_session_yolo("");
        assert!(!is_session_yolo_enabled(""));
    }

    #[test]
    fn container_envs_bypass_checks() {
        for env in CONTAINER_ENVS {
            let r = check_dangerous_command("rm -rf /tmp/x", env, None);
            assert!(r.approved, "env {env} should bypass");
        }
        // Even hardline rm -rf / is allowed in containers (matches Python: the
        // container check returns before the hardline check).
        let r = check_dangerous_command("rm -rf /", "docker", None);
        assert!(r.approved);
    }

    #[test]
    fn hardline_blocks_before_yolo() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe {
            std::env::set_var("HERMES_YOLO_MODE", "1");
        }
        let r = check_dangerous_command("rm -rf /", "local", None);
        assert!(!r.approved);
        assert!(r.hardline);
        unsafe {
            std::env::remove_var("HERMES_YOLO_MODE");
        }
    }

    #[test]
    fn yolo_bypasses_dangerous_but_not_hardline() {
        let _g = ENV_GUARD.lock().unwrap();
        unsafe {
            std::env::set_var("HERMES_YOLO_MODE", "yes");
        }
        let r = check_dangerous_command("git reset --hard", "local", None);
        assert!(r.approved);
        unsafe {
            std::env::remove_var("HERMES_YOLO_MODE");
        }
    }

    #[test]
    fn no_callback_denies_in_cli() {
        let _g = ENV_GUARD.lock().unwrap();
        let sk = "no-callback-deny";
        clear_session(sk);
        let _t = set_current_session_key(sk);
        unsafe {
            std::env::set_var("HERMES_INTERACTIVE", "1");
            std::env::remove_var("HERMES_GATEWAY_SESSION");
            std::env::remove_var("HERMES_YOLO_MODE");
        }
        let r = check_dangerous_command("git reset --hard", "local", None);
        assert!(!r.approved);
        let msg = r.message.unwrap();
        assert!(msg.contains("BLOCKED"));
        unsafe {
            std::env::remove_var("HERMES_INTERACTIVE");
        }
        reset_current_session_key(_t);
        clear_session(sk);
    }

    #[test]
    fn callback_session_approval_persists() {
        let _g = ENV_GUARD.lock().unwrap();
        let sk = "cb-session-approve";
        clear_session(sk);
        let _t = set_current_session_key(sk);
        unsafe {
            std::env::set_var("HERMES_INTERACTIVE", "1");
            std::env::remove_var("HERMES_GATEWAY_SESSION");
            std::env::remove_var("HERMES_YOLO_MODE");
        }
        let cb: Box<ApprovalCallback> = Box::new(|_c, _d, _ap| "session".to_string());
        let r = check_dangerous_command("git reset --hard", "local", Some(cb.as_ref()));
        assert!(r.approved);
        // Subsequent identical command should be auto-approved (session).
        let r2 = check_dangerous_command("git reset --hard", "local", None);
        assert!(r2.approved);
        unsafe {
            std::env::remove_var("HERMES_INTERACTIVE");
        }
        reset_current_session_key(_t);
        clear_session(sk);
    }

    #[test]
    fn gateway_resolve_unblocks_waiter() {
        let sk = "gw-resolve-test";
        clear_session(sk);
        let entry = ApprovalEntry::new(
            PendingApproval {
                command: "rm -rf x".to_string(),
                ..Default::default()
            },
            next_entry_id(),
        );
        {
            let mut st = state().lock().unwrap();
            st.gateway_queues
                .entry(sk.to_string())
                .or_default()
                .push(entry.clone());
        }
        assert!(has_blocking_approval(sk));
        let n = resolve_gateway_approval(sk, "once", false);
        assert_eq!(n, 1);
        assert!(entry.wait(Duration::from_millis(10)));
        assert_eq!(entry.result().as_deref(), Some("once"));
        assert!(!has_blocking_approval(sk));
    }

    #[test]
    fn tirith_description_formatting() {
        let res = TirithResult {
            action: "warn".to_string(),
            findings: vec![TirithFinding {
                severity: "high".to_string(),
                title: "Secret leak".to_string(),
                description: "exposes token".to_string(),
                rule_id: "secret-1".to_string(),
            }],
            summary: String::new(),
        };
        assert_eq!(
            format_tirith_description(&res),
            "Security scan — [high] Secret leak: exposes token"
        );
        let empty = TirithResult {
            action: "warn".to_string(),
            findings: vec![],
            summary: "weird thing".to_string(),
        };
        assert_eq!(
            format_tirith_description(&empty),
            "Security scan: weird thing"
        );
    }

    #[test]
    fn session_key_context_stack() {
        let t1 = set_current_session_key("alpha");
        assert_eq!(get_current_session_key("default"), "alpha");
        let t2 = set_current_session_key("beta");
        assert_eq!(get_current_session_key("default"), "beta");
        reset_current_session_key(t2);
        assert_eq!(get_current_session_key("default"), "alpha");
        reset_current_session_key(t1);
    }
}
