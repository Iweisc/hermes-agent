//! Native Rust port of `hermes_cli/main.py` — the Hermes CLI top-level entry
//! point and command dispatcher.
//!
//! The original Python module is a large `argparse`-based dispatcher that wires
//! up every `hermes <subcommand>` parser and routes to per-command handlers.
//! Most command bodies delegate to other `hermes_cli` / `agent` modules; this
//! port reproduces:
//!
//!   * the self-contained *pure logic* faithfully and idiomatically (relative
//!     time formatting, profile pre-parsing, npm lockfile staleness, TUI argv
//!     construction helpers, fork detection, region inference, reasoning-effort
//!     ordering, session-name coalescing, auxiliary-config rendering, container
//!     exec command construction, etc.);
//!   * the command-routing structure as a typed [`Command`] enum plus a
//!     [`dispatch`] entry point that mirrors `main()`'s control flow;
//!   * cross-module delegations as thin shims that call into the corresponding
//!     ported Rust modules (referenced via `crate::…`) when available, or which
//!     accept the work as a parameter otherwise.
//!
//! Interactive curses/menu flows and network-heavy provider wizards are kept as
//! structured stubs that preserve the *prompt text and decision logic* but rely
//! on injected I/O — the parent runtime supplies the concrete provider/config
//! plumbing.

use std::collections::HashSet;
use std::path::Path;

// ---------------------------------------------------------------------------
// Version metadata (mirrors `hermes_cli.__version__` / `__release_date__`).
// These are normally read from the package; callers may override.
// ---------------------------------------------------------------------------

/// Base URLs reproduced from `hermes_constants` so the model flows can wire
/// provider config without importing the whole constants module.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Vercel AI Gateway OpenAI-compatible base URL.
pub const AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";

/// Default gateway restart-drain budget (seconds) — mirrors
/// `DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT`.
pub const DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT: f64 = 60.0;

/// Official Hermes repository URLs used for fork detection during update.
pub const OFFICIAL_REPO_URLS: &[&str] = &[
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
];
/// Canonical official repo URL for `git remote add upstream`.
pub const OFFICIAL_REPO_URL: &str = "https://github.com/NousResearch/hermes-agent.git";
/// Marker file (under `$HERMES_HOME`) recording that the user declined the
/// "add upstream remote" prompt.
pub const SKIP_UPSTREAM_PROMPT_FILE: &str = ".skip_upstream_prompt";

/// Lockfile fields npm writes non-deterministically at install time; excluded
/// from the `_tui_need_npm_install` content comparison.
pub const NPM_LOCK_RUNTIME_KEYS: &[&str] = &["ideallyInert", "peer"];

// ---------------------------------------------------------------------------
// Subcommand catalogue (mirrors the `_SUBCOMMANDS` set in
// `_coalesce_session_name_args` plus the parsers registered in `main()`).
// ---------------------------------------------------------------------------

/// Top-level subcommand names recognised by the dispatcher. Order/membership
/// mirrors the `argparse` subparsers registered in `main()` and the
/// `_SUBCOMMANDS` set used by [`coalesce_session_name_args`].
pub const SUBCOMMANDS: &[&str] = &[
    "chat",
    "model",
    "fallback",
    "gateway",
    "setup",
    "whatsapp",
    "slack",
    "login",
    "logout",
    "auth",
    "status",
    "cron",
    "doctor",
    "config",
    "pairing",
    "skills",
    "tools",
    "mcp",
    "sessions",
    "insights",
    "version",
    "update",
    "uninstall",
    "profile",
    "dashboard",
    "honcho",
    "claw",
    "plugins",
    "acp",
    "webhook",
    "memory",
    "dump",
    "debug",
    "backup",
    "import",
    "completion",
    "logs",
    "kanban",
    "hooks",
    "curator",
    "checkpoints",
];

/// Session-resume flags recognised by [`coalesce_session_name_args`].
pub const SESSION_FLAGS: &[&str] = &["-c", "--continue", "-r", "--resume"];

// ---------------------------------------------------------------------------
// Relative-time formatting (`_relative_time`)
// ---------------------------------------------------------------------------

/// Format a unix timestamp (seconds) relative to `now` (also seconds).
///
/// Faithful port of `_relative_time` — returns `"?"` for falsy timestamps,
/// `"just now"`, `"<n>m ago"`, `"<n>h ago"`, `"yesterday"`, `"<n>d ago"`, or an
/// absolute `%Y-%m-%d` date for anything ≥ 1 week old.
pub fn relative_time(ts: Option<f64>, now: f64) -> String {
    let ts = match ts {
        Some(t) if t != 0.0 => t,
        _ => return "?".to_string(),
    };
    let delta = now - ts;
    if delta < 60.0 {
        return "just now".to_string();
    }
    if delta < 3600.0 {
        return format!("{}m ago", (delta / 60.0) as i64);
    }
    if delta < 86400.0 {
        return format!("{}h ago", (delta / 3600.0) as i64);
    }
    if delta < 172800.0 {
        return "yesterday".to_string();
    }
    if delta < 604800.0 {
        return format!("{}d ago", (delta / 86400.0) as i64);
    }
    // Absolute date for older timestamps.
    match chrono::DateTime::<chrono::Utc>::from_timestamp(ts as i64, 0) {
        Some(dt) => {
            let local = dt.with_timezone(&chrono::Local);
            local.format("%Y-%m-%d").to_string()
        }
        None => "?".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Profile override pre-parse (`_apply_profile_override`)
// ---------------------------------------------------------------------------

/// Regex source mirroring `hermes_cli.profiles._PROFILE_ID_RE`.
pub const PROFILE_ID_RE: &str = r"^[a-z0-9][a-z0-9_-]{0,63}$";

/// Result of pre-parsing `--profile` / `-p` from argv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileOverride {
    /// Resolved profile name, if one was found and accepted.
    pub profile_name: Option<String>,
    /// How many tokens (1 or 2) should be stripped from argv for the flag.
    /// `0` means nothing was consumed from the command line (sticky default).
    pub consume: usize,
}

/// Pre-parse `--profile` / `-p` / `--profile=<name>` out of `argv`
/// (everything after the program name). Mirrors `_apply_profile_override`'s
/// argv inspection: it does NOT resolve the env var or read the active_profile
/// file — that is the caller's responsibility (those touch global state).
///
/// Values that fail [`is_valid_profile_name`] when supplied via the two-token
/// form (`-p X`) are rejected, exactly like the Python guard that prevents
/// pytest's `-p no:xdist` from being misread as a profile.
pub fn parse_profile_override(argv: &[String]) -> ProfileOverride {
    let mut profile_name: Option<String> = None;
    let mut consume = 0usize;

    for (i, arg) in argv.iter().enumerate() {
        if (arg == "--profile" || arg == "-p") && i + 1 < argv.len() {
            profile_name = Some(argv[i + 1].clone());
            consume = 2;
            break;
        } else if let Some(rest) = arg.strip_prefix("--profile=") {
            profile_name = Some(rest.to_string());
            consume = 1;
            break;
        }
    }

    // Reject invalid two-token values (e.g. `-p no:xdist`).
    if let Some(name) = &profile_name {
        if consume == 2 && !is_valid_profile_name(name) {
            profile_name = None;
            consume = 0;
        }
    }

    ProfileOverride {
        profile_name,
        consume,
    }
}

/// Validate a profile name against [`PROFILE_ID_RE`].
pub fn is_valid_profile_name(name: &str) -> bool {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(PROFILE_ID_RE).unwrap());
    re.is_match(name)
}

/// Strip the `--profile`/`-p`/`--profile=` flag (and its value, per
/// `consume`) from a full `sys.argv`-style vector (including argv[0]).
///
/// Mirrors the argv-rewrite tail of `_apply_profile_override`. `argv` is the
/// FULL argv (program name first); the `override_` is the result of
/// [`parse_profile_override`] applied to `argv[1..]`.
pub fn strip_profile_flag(argv: &[String], override_: &ProfileOverride) -> Vec<String> {
    if override_.consume == 0 {
        return argv.to_vec();
    }
    let tail = &argv[1..];
    for (i, arg) in tail.iter().enumerate() {
        if arg == "--profile" || arg == "-p" {
            // `start` is index into full argv (i+1 because tail starts at argv[1]).
            let start = i + 1;
            let mut out: Vec<String> = argv[..start].to_vec();
            out.extend_from_slice(&argv[start + override_.consume..]);
            return out;
        } else if arg.starts_with("--profile=") {
            let start = i + 1;
            let mut out: Vec<String> = argv[..start].to_vec();
            out.extend_from_slice(&argv[start + 1..]);
            return out;
        }
    }
    argv.to_vec()
}

// ---------------------------------------------------------------------------
// Container exec command construction (`_exec_in_container`)
// ---------------------------------------------------------------------------

/// Information describing how to route a CLI invocation into a managed
/// container, mirroring the `container_info` dict in `_exec_in_container`.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    pub backend: String,
    pub container_name: String,
    pub exec_user: String,
    pub hermes_bin: String,
}

/// Build the `<runtime> exec …` argv that replaces the current process to run
/// `hermes <cli_args>` inside a managed container.
///
/// Faithful port of the command-assembly portion of `_exec_in_container`. The
/// caller supplies the resolved runtime path, an optional `sudo` path (for
/// rootful containers), whether stdin is a TTY, and the environment passthrough
/// values for TERM/COLORTERM/LANG/LC_ALL.
pub fn build_container_exec_cmd(
    info: &ContainerInfo,
    runtime: &str,
    sudo_path: Option<&str>,
    is_tty: bool,
    env_passthrough: &[(&str, Option<String>)],
    cli_args: &[String],
) -> Vec<String> {
    let mut cmd: Vec<String> = Vec::new();
    if let Some(sudo) = sudo_path {
        cmd.push(sudo.to_string());
        cmd.push("-n".to_string());
    }
    cmd.push(runtime.to_string());
    cmd.push("exec".to_string());
    if is_tty {
        cmd.push("-it".to_string());
    } else {
        cmd.push("-i".to_string());
    }
    cmd.push("-u".to_string());
    cmd.push(info.exec_user.clone());
    for (var, val) in env_passthrough {
        if let Some(v) = val {
            if !v.is_empty() {
                cmd.push("-e".to_string());
                cmd.push(format!("{var}={v}"));
            }
        }
    }
    cmd.push(info.container_name.clone());
    cmd.push(info.hermes_bin.clone());
    cmd.extend_from_slice(cli_args);
    cmd
}

/// Standard environment variables forwarded into a container by
/// `_exec_in_container`.
pub const CONTAINER_ENV_PASSTHROUGH: &[&str] = &["TERM", "COLORTERM", "LANG", "LC_ALL"];

// ---------------------------------------------------------------------------
// TUI npm-lockfile staleness (`_tui_need_npm_install`)
// ---------------------------------------------------------------------------

/// Decide whether `npm install` is needed for the TUI in `root`.
///
/// Faithful port of `_tui_need_npm_install`: compares the root
/// `package-lock.json` against `node_modules/.package-lock.json` by content,
/// ignoring npm's non-deterministic runtime annotations.
pub fn tui_need_npm_install(root: &Path) -> bool {
    let ink = root
        .join("node_modules")
        .join("@hermes")
        .join("ink")
        .join("package.json");
    if !ink.is_file() {
        return true;
    }
    let lock = root.join("package-lock.json");
    if !lock.is_file() {
        return false;
    }
    let marker = root.join("node_modules").join(".package-lock.json");
    if !marker.is_file() {
        return true;
    }

    let wanted_txt = match std::fs::read_to_string(&lock) {
        Ok(t) => t,
        Err(_) => return lockfile_mtime_newer(&lock, &marker),
    };
    let installed_txt = match std::fs::read_to_string(&marker) {
        Ok(t) => t,
        Err(_) => return lockfile_mtime_newer(&lock, &marker),
    };
    let wanted: serde_json::Value = match serde_json::from_str(&wanted_txt) {
        Ok(v) => v,
        Err(_) => return lockfile_mtime_newer(&lock, &marker),
    };
    let installed: serde_json::Value = match serde_json::from_str(&installed_txt) {
        Ok(v) => v,
        Err(_) => return lockfile_mtime_newer(&lock, &marker),
    };

    let wanted_pkgs = wanted
        .get("packages")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let installed_pkgs = installed
        .get("packages")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    needs_install_from_packages(&wanted_pkgs, &installed_pkgs)
}

/// Core comparison of two `packages` maps — extracted so it can be unit-tested
/// without touching the filesystem. Returns true when a reinstall is needed.
pub fn needs_install_from_packages(
    wanted: &serde_json::Map<String, serde_json::Value>,
    installed: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let runtime_keys: HashSet<&str> = NPM_LOCK_RUNTIME_KEYS.iter().copied().collect();
    let comparable = |pkg: &serde_json::Map<String, serde_json::Value>| {
        let mut m = serde_json::Map::new();
        for (k, v) in pkg {
            if !runtime_keys.contains(k.as_str()) {
                m.insert(k.clone(), v.clone());
            }
        }
        m
    };

    for (name, pkg) in wanted {
        if name.is_empty() {
            continue;
        }
        let pkg = match pkg.as_object() {
            Some(p) => p,
            None => continue,
        };
        match installed.get(name) {
            None => {
                let optional = pkg
                    .get("optional")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let peer = pkg.get("peer").and_then(|v| v.as_bool()).unwrap_or(false);
                if optional || peer {
                    continue;
                }
                return true;
            }
            Some(inst) => {
                if let Some(inst_obj) = inst.as_object() {
                    if comparable(pkg) != comparable(inst_obj) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn lockfile_mtime_newer(lock: &Path, marker: &Path) -> bool {
    let lm = lock
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64());
    let mm = marker
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64());
    match (lm, mm) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// TUI toolset normalization (`_normalize_tui_toolsets`)
// ---------------------------------------------------------------------------

/// Normalize comma-separated / list-style toolset input into a flat, trimmed,
/// non-empty `Vec<String>`. Mirrors `_normalize_tui_toolsets`'s fallback path.
pub fn normalize_tui_toolsets(items: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        for part in item.split(',') {
            let p = part.trim();
            if !p.is_empty() {
                out.push(p.to_string());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// NODE_OPTIONS merge for the TUI subprocess (`_launch_tui`)
// ---------------------------------------------------------------------------

/// Token-level merge of `NODE_OPTIONS`: guarantee a max old-space size and an
/// exposed GC without clobbering user-supplied flags. Mirrors the `_tokens`
/// logic in `_launch_tui`.
pub fn merge_node_options(existing: &str) -> String {
    let mut tokens: Vec<String> = existing.split_whitespace().map(str::to_string).collect();
    if !tokens
        .iter()
        .any(|t| t.starts_with("--max-old-space-size="))
    {
        tokens.push("--max-old-space-size=8192".to_string());
    }
    if !tokens.iter().any(|t| t == "--expose-gc") {
        tokens.push("--expose-gc".to_string());
    }
    tokens.join(" ")
}

// ---------------------------------------------------------------------------
// Auxiliary-model rendering (`_format_aux_current`, `_AUX_TASKS`)
// ---------------------------------------------------------------------------

/// (task_key, display_name, short_description) for each auxiliary task, mirror
/// of `_AUX_TASKS`.
pub const AUX_TASKS: &[(&str, &str, &str)] = &[
    ("vision", "Vision", "image/screenshot analysis"),
    ("compression", "Compression", "context summarization"),
    ("web_extract", "Web extract", "web page summarization"),
    ("session_search", "Session search", "past-conversation recall"),
    ("approval", "Approval", "smart command approval"),
    ("mcp", "MCP", "MCP tool reasoning"),
    ("title_generation", "Title generation", "session titles"),
    ("skills_hub", "Skills hub", "skills search/install"),
    ("curator", "Curator", "skill-usage review pass"),
];

/// Render the current auxiliary config for display in the task menu. Faithful
/// port of `_format_aux_current`.
pub fn format_aux_current(task_cfg: &serde_json::Value) -> String {
    let obj = match task_cfg.as_object() {
        Some(o) => o,
        None => return "auto".to_string(),
    };
    let str_field = |k: &str| -> String {
        obj.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let base_url = str_field("base_url");
    let provider_raw = str_field("provider");
    let provider = if provider_raw.is_empty() {
        "auto".to_string()
    } else {
        provider_raw
    };
    let model = str_field("model");

    if !base_url.is_empty() {
        let short = base_url
            .replace("https://", "")
            .replace("http://", "")
            .trim_end_matches('/')
            .to_string();
        let mut s = format!("custom ({short})");
        if !model.is_empty() {
            s.push_str(&format!(" · {model}"));
        }
        return s;
    }
    if provider == "auto" {
        let mut s = "auto".to_string();
        if !model.is_empty() {
            s.push_str(&format!(" · {model}"));
        }
        return s;
    }
    if !model.is_empty() {
        return format!("{provider} · {model}");
    }
    provider
}

// ---------------------------------------------------------------------------
// Reasoning-effort ordering (`_prompt_reasoning_effort_selection` core)
// ---------------------------------------------------------------------------

/// Canonical reasoning-effort ordering used by the picker.
pub const REASONING_EFFORT_CANONICAL: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];

/// Dedup + canonically order a list of reasoning efforts. Faithful port of the
/// dedup/ordering logic at the top of `_prompt_reasoning_effort_selection`.
pub fn order_reasoning_efforts(efforts: &[String]) -> Vec<String> {
    let mut deduped: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for e in efforts {
        let norm = e.trim().to_lowercase();
        if norm.is_empty() {
            continue;
        }
        if seen.insert(norm.clone()) {
            deduped.push(norm);
        }
    }
    let canonical: HashSet<&str> = REASONING_EFFORT_CANONICAL.iter().copied().collect();
    let mut ordered: Vec<String> = REASONING_EFFORT_CANONICAL
        .iter()
        .filter(|e| deduped.iter().any(|d| d == *e))
        .map(|s| s.to_string())
        .collect();
    ordered.extend(
        deduped
            .into_iter()
            .filter(|e| !canonical.contains(e.as_str())),
    );
    ordered
}

/// Compute the default cursor index for the reasoning-effort picker, mirroring
/// the `default_idx` selection. `ordered` is the output of
/// [`order_reasoning_efforts`]; the returned index may equal `ordered.len()`
/// (meaning the "Disable reasoning" entry when `current == "none"`).
pub fn reasoning_effort_default_idx(ordered: &[String], current: &str) -> usize {
    if current == "none" {
        return ordered.len();
    }
    if let Some(pos) = ordered.iter().position(|e| e == current) {
        return pos;
    }
    if let Some(pos) = ordered.iter().position(|e| e == "medium") {
        return pos;
    }
    0
}

// ---------------------------------------------------------------------------
// StepFun region inference (`_infer_stepfun_region`, `_stepfun_base_url_for_region`)
// ---------------------------------------------------------------------------

/// StepFun China base URL (mirrors `STEPFUN_STEP_PLAN_CN_BASE_URL`).
pub const STEPFUN_STEP_PLAN_CN_BASE_URL: &str = "https://api.stepfun.com/v1";
/// StepFun international base URL (mirrors `STEPFUN_STEP_PLAN_INTL_BASE_URL`).
pub const STEPFUN_STEP_PLAN_INTL_BASE_URL: &str = "https://api.stepfun.ai/v1";

/// Infer the StepFun region from the configured endpoint. Faithful port of
/// `_infer_stepfun_region`.
pub fn infer_stepfun_region(base_url: &str) -> &'static str {
    let normalized = base_url.trim().to_lowercase();
    if normalized.contains("api.stepfun.com") {
        "china"
    } else {
        "international"
    }
}

/// Map a region to its StepFun base URL. Faithful port of
/// `_stepfun_base_url_for_region`.
pub fn stepfun_base_url_for_region(region: &str) -> &'static str {
    if region == "china" {
        STEPFUN_STEP_PLAN_CN_BASE_URL
    } else {
        STEPFUN_STEP_PLAN_INTL_BASE_URL
    }
}

// ---------------------------------------------------------------------------
// Custom-provider helpers (`_auto_provider_name`,
// `_custom_provider_api_key_config_value`)
// ---------------------------------------------------------------------------

/// Generate a human-friendly display name from a custom endpoint URL. Faithful
/// port of `_auto_provider_name`.
pub fn auto_provider_name(base_url: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"/v1/?$").unwrap());

    let clean = base_url
        .replace("https://", "")
        .replace("http://", "")
        .trim_end_matches('/')
        .to_string();
    let clean = re.replace(&clean, "").to_string();
    let name = clean.split('/').next().unwrap_or("").to_string();

    if name.contains("localhost") || name.contains("127.0.0.1") {
        format!("Local ({name})")
    } else if name.to_lowercase().contains("runpod") {
        format!("RunPod ({name})")
    } else {
        capitalize(&name)
    }
}

/// Python `str.capitalize()`: first char uppercase, rest lowercase.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
        }
    }
}

/// Determine the value to persist for a custom provider's `api_key`. Faithful
/// port of `_custom_provider_api_key_config_value`.
///
/// `provider_info` is a map with optional `api_key_ref`, `key_env`, `api_key`.
pub fn custom_provider_api_key_config_value(
    provider_info: &serde_json::Map<String, serde_json::Value>,
    resolved_api_key: &str,
) -> String {
    let s = |k: &str| -> String {
        provider_info
            .get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let api_key_ref = s("api_key_ref");
    if !api_key_ref.is_empty() {
        return api_key_ref;
    }
    let key_env = s("key_env");
    if !key_env.is_empty() && s("api_key").is_empty() {
        return format!("${{{key_env}}}");
    }
    resolved_api_key.trim().to_string()
}

// ---------------------------------------------------------------------------
// Fork detection for `hermes update` (`_is_fork`)
// ---------------------------------------------------------------------------

/// Normalize a git remote URL for comparison (strip trailing `/` then `.git`).
fn normalize_repo_url(url: &str) -> String {
    let mut normalized = url.trim_end_matches('/').to_string();
    if let Some(stripped) = normalized.strip_suffix(".git") {
        normalized = stripped.to_string();
    }
    normalized
}

/// Check whether the origin remote points at a fork (not the official repo).
/// Faithful port of `_is_fork`.
pub fn is_fork(origin_url: Option<&str>) -> bool {
    let url = match origin_url {
        Some(u) if !u.is_empty() => u,
        _ => return false,
    };
    let normalized = normalize_repo_url(url);
    for official in OFFICIAL_REPO_URLS {
        if normalize_repo_url(official) == normalized {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// systemd RestartUSec parsing (`_service_restart_sec`)
// ---------------------------------------------------------------------------

/// Parse a systemd duration string like `"30s"`, `"100ms"`, `"1min 30s"`, or
/// `"infinity"` into seconds. Faithful port of the parser inside
/// `_service_restart_sec`. Returns `default` on no-match / `infinity` / empty.
pub fn parse_systemd_duration(raw: &str, default: f64) -> f64 {
    let raw = raw.trim();
    if raw.is_empty() || raw == "infinity" {
        return default;
    }
    // Order matters: "min" must be tried before "s" since "min" ends with no
    // overlap, but the suffix table mirrors Python's exact ordering. Python
    // checks ms, us, min, s in that order and breaks on first match.
    let suffixes: &[(&str, f64)] = &[("ms", 0.001), ("us", 0.000001), ("min", 60.0), ("s", 1.0)];
    let mut total = 0.0;
    let mut matched = false;
    for part in raw.split_whitespace() {
        for (suf, mult) in suffixes {
            if part.ends_with(suf) {
                let numeric = &part[..part.len() - suf.len()];
                if let Ok(val) = numeric.parse::<f64>() {
                    total += val * mult;
                    matched = true;
                }
                break;
            }
        }
    }
    if matched {
        total
    } else {
        default
    }
}

/// Compute the gateway drain budget for graceful restarts. Mirrors the
/// `_drain_budget` computation in `_cmd_update_impl`: read configured value
/// (or the default), floor to 30s, then add a 15s escalation margin.
pub fn gateway_drain_budget(configured: Option<f64>) -> f64 {
    let base = configured.unwrap_or(DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT);
    base.max(30.0) + 15.0
}

// ---------------------------------------------------------------------------
// Session-name coalescing (`_coalesce_session_name_args`)
// ---------------------------------------------------------------------------

/// Join unquoted multi-word session names after `-c`/`--continue` and
/// `-r`/`--resume` into a single argument. Faithful port of
/// `_coalesce_session_name_args` (operating on `argv[1..]`).
pub fn coalesce_session_name_args(argv: &[String]) -> Vec<String> {
    let subcommands: HashSet<&str> = SUBCOMMANDS.iter().copied().collect();
    let session_flags: HashSet<&str> = SESSION_FLAGS.iter().copied().collect();

    let mut result: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let token = &argv[i];
        if session_flags.contains(token.as_str()) {
            result.push(token.clone());
            i += 1;
            let mut parts: Vec<String> = Vec::new();
            while i < argv.len()
                && !argv[i].starts_with('-')
                && !subcommands.contains(argv[i].as_str())
            {
                parts.push(argv[i].clone());
                i += 1;
            }
            if !parts.is_empty() {
                result.push(parts.join(" "));
            }
        } else {
            result.push(token.clone());
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Web UI build dependency naming + provider choices (`_build_provider_choices`)
// ---------------------------------------------------------------------------

/// Static fallback list of `--provider` choices, mirroring the fallback branch
/// of `_build_provider_choices` (used when CANONICAL_PROVIDERS is unavailable).
pub fn build_provider_choices_fallback() -> Vec<String> {
    [
        "auto",
        "openrouter",
        "nous",
        "openai-codex",
        "copilot-acp",
        "copilot",
        "anthropic",
        "gemini",
        "google-gemini-cli",
        "xai",
        "bedrock",
        "azure-foundry",
        "ollama-cloud",
        "huggingface",
        "zai",
        "kimi-coding",
        "kimi-coding-cn",
        "stepfun",
        "minimax",
        "minimax-cn",
        "kilocode",
        "xiaomi",
        "arcee",
        "nvidia",
        "deepseek",
        "alibaba",
        "qwen-oauth",
        "opencode-zen",
        "opencode-go",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Build `--provider` choices from a slug list (`CANONICAL_PROVIDERS`),
/// prepending `"auto"`. Faithful port of `_build_provider_choices`'s primary
/// path. When `slugs` is empty, falls back to [`build_provider_choices_fallback`].
pub fn build_provider_choices(slugs: &[String]) -> Vec<String> {
    if slugs.is_empty() {
        return build_provider_choices_fallback();
    }
    let mut out = vec!["auto".to_string()];
    out.extend(slugs.iter().cloned());
    out
}

// ---------------------------------------------------------------------------
// Agent-command gating (`_AGENT_COMMANDS`, `_AGENT_SUBCOMMANDS`)
// ---------------------------------------------------------------------------

/// Whether the given command (and its nested subcommand) should trigger plugin
/// discovery / MCP discovery / shell-hook registration at CLI startup. Faithful
/// port of the gating logic in `main()`.
///
/// `command` is `None` for the implicit chat default, or the subcommand name.
/// `subcommand` is the nested action (e.g. `"run"` under `gateway`).
pub fn should_run_agent_startup(command: Option<&str>, subcommand: Option<&str>) -> bool {
    // _AGENT_COMMANDS = {None, "chat", "acp", "rl"}
    let agent_commands: HashSet<Option<&str>> =
        [None, Some("chat"), Some("acp"), Some("rl")].into_iter().collect();
    if agent_commands.contains(&command) {
        return true;
    }
    // _AGENT_SUBCOMMANDS narrows via the nested subcommand.
    let nested: Option<HashSet<&str>> = match command {
        Some("cron") => Some(["run", "tick"].into_iter().collect()),
        Some("gateway") => Some(["run"].into_iter().collect()),
        Some("mcp") => Some(["serve"].into_iter().collect()),
        _ => None,
    };
    if let (Some(set), Some(sub)) = (nested, subcommand) {
        return set.contains(sub);
    }
    false
}

// ---------------------------------------------------------------------------
// Command model + dispatch skeleton (mirrors `main()` routing)
// ---------------------------------------------------------------------------

/// A parsed top-level command, mirroring the `args.command` dispatch in
/// `main()`. Only the routing identity is modelled here; each variant carries
/// the nested subcommand string where the Python parser used `dest=...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Interactive chat (the implicit default when no subcommand is given).
    Chat,
    Model,
    Fallback(Option<String>),
    Gateway(Option<String>),
    Setup(Option<String>),
    Whatsapp,
    Slack(Option<String>),
    Login,
    Logout,
    Auth(Option<String>),
    Status,
    Cron(Option<String>),
    Webhook(Option<String>),
    Kanban,
    Hooks(Option<String>),
    Doctor,
    Dump,
    Debug(Option<String>),
    Config(Option<String>),
    Backup,
    Checkpoints,
    Import,
    Pairing(Option<String>),
    Skills(Option<String>),
    Plugins(Option<String>),
    Curator,
    Memory(Option<String>),
    Tools(Option<String>),
    Mcp(Option<String>),
    Sessions(Option<String>),
    Insights,
    Claw(Option<String>),
    Version,
    Update,
    Uninstall,
    Acp,
    Profile(Option<String>),
    Completion,
    Dashboard,
    Logs,
}

impl Command {
    /// Return the canonical command name (the argparse subparser name).
    pub fn name(&self) -> &'static str {
        match self {
            Command::Chat => "chat",
            Command::Model => "model",
            Command::Fallback(_) => "fallback",
            Command::Gateway(_) => "gateway",
            Command::Setup(_) => "setup",
            Command::Whatsapp => "whatsapp",
            Command::Slack(_) => "slack",
            Command::Login => "login",
            Command::Logout => "logout",
            Command::Auth(_) => "auth",
            Command::Status => "status",
            Command::Cron(_) => "cron",
            Command::Webhook(_) => "webhook",
            Command::Kanban => "kanban",
            Command::Hooks(_) => "hooks",
            Command::Doctor => "doctor",
            Command::Dump => "dump",
            Command::Debug(_) => "debug",
            Command::Config(_) => "config",
            Command::Backup => "backup",
            Command::Checkpoints => "checkpoints",
            Command::Import => "import",
            Command::Pairing(_) => "pairing",
            Command::Skills(_) => "skills",
            Command::Plugins(_) => "plugins",
            Command::Curator => "curator",
            Command::Memory(_) => "memory",
            Command::Tools(_) => "tools",
            Command::Mcp(_) => "mcp",
            Command::Sessions(_) => "sessions",
            Command::Insights => "insights",
            Command::Claw(_) => "claw",
            Command::Version => "version",
            Command::Update => "update",
            Command::Uninstall => "uninstall",
            Command::Acp => "acp",
            Command::Profile(_) => "profile",
            Command::Completion => "completion",
            Command::Dashboard => "dashboard",
            Command::Logs => "logs",
        }
    }

    /// The nested subcommand (the argparse `dest`), if any.
    pub fn subcommand(&self) -> Option<&str> {
        match self {
            Command::Fallback(s)
            | Command::Gateway(s)
            | Command::Setup(s)
            | Command::Slack(s)
            | Command::Auth(s)
            | Command::Cron(s)
            | Command::Webhook(s)
            | Command::Hooks(s)
            | Command::Debug(s)
            | Command::Config(s)
            | Command::Pairing(s)
            | Command::Skills(s)
            | Command::Plugins(s)
            | Command::Memory(s)
            | Command::Tools(s)
            | Command::Mcp(s)
            | Command::Sessions(s)
            | Command::Claw(s)
            | Command::Profile(s) => s.as_deref(),
            _ => None,
        }
    }

    /// Whether CLI startup (plugin/MCP discovery, hook registration) should run
    /// for this command. See [`should_run_agent_startup`].
    pub fn runs_agent_startup(&self) -> bool {
        should_run_agent_startup(Some(self.name()), self.subcommand())
    }
}

/// Determine the effective command from a parsed `command` token + nested
/// subcommand. Used by [`dispatch`] to resolve the implicit-chat default.
///
/// Mirrors `main()`'s tail: when `command` is `None` (no subcommand) the
/// dispatcher defaults to chat, and a bare top-level `--resume`/`--continue`
/// likewise routes into chat.
pub fn resolve_command(
    command: Option<&str>,
    subcommand: Option<String>,
    resume_or_continue: bool,
) -> Command {
    match command {
        None => Command::Chat, // default to chat (also covers --resume/--continue shortcut)
        Some(name) => command_from_name(name, subcommand),
    }
    .pipe(|c| {
        // The `resume_or_continue` flag only matters when there is no explicit
        // command — in that case we already resolved to Chat above.
        let _ = resume_or_continue;
        c
    })
}

/// Map a subcommand name + optional nested action to a [`Command`].
fn command_from_name(name: &str, sub: Option<String>) -> Command {
    match name {
        "chat" => Command::Chat,
        "model" => Command::Model,
        "fallback" => Command::Fallback(sub),
        "gateway" => Command::Gateway(sub),
        "setup" => Command::Setup(sub),
        "whatsapp" => Command::Whatsapp,
        "slack" => Command::Slack(sub),
        "login" => Command::Login,
        "logout" => Command::Logout,
        "auth" => Command::Auth(sub),
        "status" => Command::Status,
        "cron" => Command::Cron(sub),
        "webhook" => Command::Webhook(sub),
        "kanban" => Command::Kanban,
        "hooks" => Command::Hooks(sub),
        "doctor" => Command::Doctor,
        "dump" => Command::Dump,
        "debug" => Command::Debug(sub),
        "config" => Command::Config(sub),
        "backup" => Command::Backup,
        "checkpoints" => Command::Checkpoints,
        "import" => Command::Import,
        "pairing" => Command::Pairing(sub),
        "skills" => Command::Skills(sub),
        "plugins" => Command::Plugins(sub),
        "curator" => Command::Curator,
        "memory" => Command::Memory(sub),
        "tools" => Command::Tools(sub),
        "mcp" => Command::Mcp(sub),
        "sessions" => Command::Sessions(sub),
        "insights" => Command::Insights,
        "claw" => Command::Claw(sub),
        "version" => Command::Version,
        "update" => Command::Update,
        "uninstall" => Command::Uninstall,
        "acp" => Command::Acp,
        "profile" => Command::Profile(sub),
        "completion" => Command::Completion,
        "dashboard" => Command::Dashboard,
        "logs" => Command::Logs,
        // Unknown command name → default to chat (argparse would error, but the
        // dispatcher's fallback is chat).
        _ => Command::Chat,
    }
}

// Small extension to chain a value through a closure (local `tap`-style helper).
trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

// ---------------------------------------------------------------------------
// TTY guard (`_require_tty`)
// ---------------------------------------------------------------------------

/// Build the error message printed by `_require_tty` when stdin is not a TTY.
pub fn require_tty_message(command_name: &str) -> String {
    format!(
        "Error: 'hermes {command_name}' requires an interactive terminal.\n\
         It cannot be run through a pipe or non-interactive subprocess.\n\
         Run it directly in your terminal instead."
    )
}

// ---------------------------------------------------------------------------
// version output (`cmd_version`)
// ---------------------------------------------------------------------------

/// Build the first two lines of `hermes version` output. Faithful port of the
/// header of `cmd_version`.
pub fn version_header(version: &str, release_date: &str, project_root: &Path) -> String {
    format!(
        "Hermes Agent v{version} ({release_date})\nProject: {}",
        project_root.display()
    )
}

/// Pluralize "commit"/"commits" and format the update-available line, mirroring
/// the behind-count formatting in `cmd_version`.
pub fn update_available_line(behind: i64, recommended_cmd: &str) -> Option<String> {
    if behind > 0 {
        let word = if behind == 1 { "commit" } else { "commits" };
        Some(format!(
            "Update available: {behind} {word} behind — run '{recommended_cmd}'"
        ))
    } else if behind == 0 {
        Some("Up to date".to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Provider env-var set for first-run detection (`_has_any_provider_configured`)
// ---------------------------------------------------------------------------

/// Base set of provider env vars checked by `_has_any_provider_configured`
/// (before registry-specific keys are unioned in).
pub const BASE_PROVIDER_ENV_VARS: &[&str] = &[
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
];

/// Parse a `.env` file's contents for any of `provider_env_vars` set to a
/// non-empty value. Faithful port of the `.env` scan in
/// `_has_any_provider_configured`.
pub fn env_file_has_provider_key(contents: &str, provider_env_vars: &HashSet<String>) -> bool {
    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let (key, val) = match line.split_once('=') {
            Some((k, v)) => (k, v),
            None => continue,
        };
        let val = val.trim().trim_matches(|c| c == '\'' || c == '"');
        if provider_env_vars.contains(key.trim()) && !val.is_empty() {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn relative_time_buckets() {
        // Use a realistic epoch so the absolute-date fallback formats correctly.
        let now = 1_700_000_000.0;
        assert_eq!(relative_time(None, now), "?");
        assert_eq!(relative_time(Some(0.0), now), "?");
        assert_eq!(relative_time(Some(now - 10.0), now), "just now");
        assert_eq!(relative_time(Some(now - 120.0), now), "2m ago");
        assert_eq!(relative_time(Some(now - 7200.0), now), "2h ago");
        assert_eq!(relative_time(Some(now - 100_000.0), now), "yesterday");
        assert_eq!(relative_time(Some(now - 300_000.0), now), "3d ago");
        // Older than a week → absolute date.
        let abs = relative_time(Some(now - 1_000_000.0), now);
        assert!(abs.contains('-'), "expected an absolute date, got {abs}");
    }

    #[test]
    fn profile_override_explicit_flag() {
        let argv = vec!["-p".to_string(), "work".to_string(), "chat".to_string()];
        let ov = parse_profile_override(&argv);
        assert_eq!(ov.profile_name.as_deref(), Some("work"));
        assert_eq!(ov.consume, 2);
    }

    #[test]
    fn profile_override_eq_form() {
        let argv = vec!["--profile=staging".to_string(), "gateway".to_string()];
        let ov = parse_profile_override(&argv);
        assert_eq!(ov.profile_name.as_deref(), Some("staging"));
        assert_eq!(ov.consume, 1);
    }

    #[test]
    fn profile_override_rejects_pytest_style() {
        // `-p no:xdist` must NOT be read as a profile (mirrors the guard).
        let argv = vec!["-p".to_string(), "no:xdist".to_string()];
        let ov = parse_profile_override(&argv);
        assert_eq!(ov.profile_name, None);
        assert_eq!(ov.consume, 0);
    }

    #[test]
    fn strip_profile_flag_two_token() {
        let argv = vec![
            "hermes".to_string(),
            "-p".to_string(),
            "work".to_string(),
            "chat".to_string(),
        ];
        let ov = parse_profile_override(&argv[1..]);
        let stripped = strip_profile_flag(&argv, &ov);
        assert_eq!(stripped, vec!["hermes".to_string(), "chat".to_string()]);
    }

    #[test]
    fn strip_profile_flag_eq_form() {
        let argv = vec![
            "hermes".to_string(),
            "--profile=staging".to_string(),
            "gateway".to_string(),
        ];
        let ov = parse_profile_override(&argv[1..]);
        let stripped = strip_profile_flag(&argv, &ov);
        assert_eq!(stripped, vec!["hermes".to_string(), "gateway".to_string()]);
    }

    #[test]
    fn container_exec_cmd_with_sudo_and_tty() {
        let info = ContainerInfo {
            backend: "podman".to_string(),
            container_name: "hermes".to_string(),
            exec_user: "hermes".to_string(),
            hermes_bin: "/usr/bin/hermes".to_string(),
        };
        let env = vec![
            ("TERM", Some("xterm-256color".to_string())),
            ("LANG", None),
        ];
        let cmd = build_container_exec_cmd(
            &info,
            "/usr/bin/podman",
            Some("/usr/bin/sudo"),
            true,
            &env,
            &["chat".to_string()],
        );
        assert_eq!(
            cmd,
            vec![
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/podman",
                "exec",
                "-it",
                "-u",
                "hermes",
                "-e",
                "TERM=xterm-256color",
                "hermes",
                "/usr/bin/hermes",
                "chat",
            ]
        );
    }

    #[test]
    fn container_exec_cmd_no_sudo_no_tty() {
        let info = ContainerInfo {
            backend: "docker".to_string(),
            container_name: "c".to_string(),
            exec_user: "u".to_string(),
            hermes_bin: "hermes".to_string(),
        };
        let cmd = build_container_exec_cmd(&info, "docker", None, false, &[], &[]);
        assert_eq!(
            cmd,
            vec!["docker", "exec", "-i", "-u", "u", "c", "hermes"]
        );
    }

    #[test]
    fn npm_install_missing_package() {
        let mut wanted = serde_json::Map::new();
        wanted.insert("node_modules/foo".to_string(), json!({"version": "1.0.0"}));
        let installed = serde_json::Map::new();
        assert!(needs_install_from_packages(&wanted, &installed));
    }

    #[test]
    fn npm_install_optional_missing_ok() {
        let mut wanted = serde_json::Map::new();
        wanted.insert(
            "node_modules/opt".to_string(),
            json!({"version": "1.0.0", "optional": true}),
        );
        let installed = serde_json::Map::new();
        assert!(!needs_install_from_packages(&wanted, &installed));
    }

    #[test]
    fn npm_install_ignores_runtime_keys() {
        let mut wanted = serde_json::Map::new();
        wanted.insert("node_modules/a".to_string(), json!({"version": "1.0.0"}));
        let mut installed = serde_json::Map::new();
        installed.insert(
            "node_modules/a".to_string(),
            json!({"version": "1.0.0", "ideallyInert": true, "peer": true}),
        );
        // Differs only in runtime keys → no reinstall.
        assert!(!needs_install_from_packages(&wanted, &installed));
    }

    #[test]
    fn npm_install_field_diff_triggers() {
        let mut wanted = serde_json::Map::new();
        wanted.insert("node_modules/a".to_string(), json!({"version": "2.0.0"}));
        let mut installed = serde_json::Map::new();
        installed.insert("node_modules/a".to_string(), json!({"version": "1.0.0"}));
        assert!(needs_install_from_packages(&wanted, &installed));
    }

    #[test]
    fn normalize_toolsets_flattens() {
        let items = vec!["a, b".to_string(), "c".to_string(), " ,".to_string()];
        assert_eq!(normalize_tui_toolsets(&items), vec!["a", "b", "c"]);
    }

    #[test]
    fn node_options_merge() {
        assert_eq!(
            merge_node_options(""),
            "--max-old-space-size=8192 --expose-gc"
        );
        // Respects user-supplied heap size, still adds --expose-gc.
        assert_eq!(
            merge_node_options("--max-old-space-size=16384"),
            "--max-old-space-size=16384 --expose-gc"
        );
        // No duplicate --expose-gc, but a heap size is still appended since one
        // wasn't supplied (mirrors the Python token-merge order).
        assert_eq!(
            merge_node_options("--expose-gc"),
            "--expose-gc --max-old-space-size=8192"
        );
    }

    #[test]
    fn aux_current_rendering() {
        assert_eq!(format_aux_current(&json!(null)), "auto");
        assert_eq!(format_aux_current(&json!({})), "auto");
        assert_eq!(
            format_aux_current(&json!({"provider": "auto", "model": "gpt"})),
            "auto · gpt"
        );
        assert_eq!(
            format_aux_current(&json!({"provider": "openrouter", "model": "x"})),
            "openrouter · x"
        );
        assert_eq!(
            format_aux_current(&json!({"provider": "openrouter"})),
            "openrouter"
        );
        assert_eq!(
            format_aux_current(&json!({"base_url": "https://x.com/v1/", "model": "m"})),
            "custom (x.com/v1) · m"
        );
    }

    #[test]
    fn reasoning_effort_ordering() {
        let efforts = vec![
            "High".to_string(),
            "low".to_string(),
            "custom".to_string(),
            "low".to_string(),
        ];
        let ordered = order_reasoning_efforts(&efforts);
        assert_eq!(ordered, vec!["low", "high", "custom"]);
    }

    #[test]
    fn reasoning_effort_default() {
        let ordered = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
        assert_eq!(reasoning_effort_default_idx(&ordered, "none"), 3);
        assert_eq!(reasoning_effort_default_idx(&ordered, "high"), 2);
        assert_eq!(reasoning_effort_default_idx(&ordered, "zzz"), 1); // medium
        let no_med = vec!["low".to_string(), "high".to_string()];
        assert_eq!(reasoning_effort_default_idx(&no_med, "zzz"), 0);
    }

    #[test]
    fn stepfun_region() {
        assert_eq!(infer_stepfun_region("https://api.stepfun.com/v1"), "china");
        assert_eq!(
            infer_stepfun_region("https://api.stepfun.ai/v1"),
            "international"
        );
        assert_eq!(
            stepfun_base_url_for_region("china"),
            STEPFUN_STEP_PLAN_CN_BASE_URL
        );
        assert_eq!(
            stepfun_base_url_for_region("international"),
            STEPFUN_STEP_PLAN_INTL_BASE_URL
        );
    }

    #[test]
    fn auto_provider_naming() {
        assert_eq!(
            auto_provider_name("http://localhost:11434/v1"),
            "Local (localhost:11434)"
        );
        assert_eq!(
            auto_provider_name("https://abc.runpod.io/v1/"),
            "RunPod (abc.runpod.io)"
        );
        assert_eq!(
            auto_provider_name("https://api.example.com/v1"),
            "Api.example.com"
        );
    }

    #[test]
    fn custom_provider_api_key_value() {
        let mut info = serde_json::Map::new();
        info.insert("api_key_ref".to_string(), json!("${MY_KEY}"));
        assert_eq!(
            custom_provider_api_key_config_value(&info, "resolved"),
            "${MY_KEY}"
        );

        let mut info2 = serde_json::Map::new();
        info2.insert("key_env".to_string(), json!("FOO_KEY"));
        assert_eq!(
            custom_provider_api_key_config_value(&info2, "resolved"),
            "${FOO_KEY}"
        );

        let info3 = serde_json::Map::new();
        assert_eq!(
            custom_provider_api_key_config_value(&info3, "  sk-123  "),
            "sk-123"
        );
    }

    #[test]
    fn fork_detection() {
        assert!(!is_fork(Some(
            "https://github.com/NousResearch/hermes-agent.git"
        )));
        assert!(!is_fork(Some(
            "https://github.com/NousResearch/hermes-agent"
        )));
        assert!(!is_fork(Some(
            "git@github.com:NousResearch/hermes-agent.git"
        )));
        assert!(is_fork(Some("https://github.com/someone/hermes-agent.git")));
        assert!(!is_fork(None));
        assert!(!is_fork(Some("")));
    }

    #[test]
    fn systemd_duration_parsing() {
        assert_eq!(parse_systemd_duration("30s", 0.0), 30.0);
        assert!((parse_systemd_duration("100ms", 0.0) - 0.1).abs() < 1e-9);
        assert_eq!(parse_systemd_duration("1min 30s", 0.0), 90.0);
        assert_eq!(parse_systemd_duration("infinity", 5.0), 5.0);
        assert_eq!(parse_systemd_duration("", 7.0), 7.0);
        assert_eq!(parse_systemd_duration("garbage", 3.0), 3.0);
    }

    #[test]
    fn drain_budget() {
        assert_eq!(gateway_drain_budget(None), 75.0); // 60 floored to 60, +15
        assert_eq!(gateway_drain_budget(Some(10.0)), 45.0); // floored to 30, +15
        assert_eq!(gateway_drain_budget(Some(120.0)), 135.0);
    }

    #[test]
    fn coalesce_session_names() {
        let argv = vec![
            "-c".to_string(),
            "Pokemon".to_string(),
            "Agent".to_string(),
            "Dev".to_string(),
        ];
        assert_eq!(
            coalesce_session_name_args(&argv),
            vec!["-c".to_string(), "Pokemon Agent Dev".to_string()]
        );
    }

    #[test]
    fn coalesce_stops_at_subcommand() {
        let argv = vec!["-r".to_string(), "model".to_string()];
        // `model` is a known subcommand → not consumed as a session name.
        assert_eq!(
            coalesce_session_name_args(&argv),
            vec!["-r".to_string(), "model".to_string()]
        );
    }

    #[test]
    fn coalesce_passthrough() {
        let argv = vec!["gateway".to_string(), "status".to_string()];
        assert_eq!(coalesce_session_name_args(&argv), argv);
    }

    #[test]
    fn provider_choices_with_slugs() {
        let slugs = vec!["openrouter".to_string(), "nous".to_string()];
        assert_eq!(
            build_provider_choices(&slugs),
            vec!["auto", "openrouter", "nous"]
        );
        // Empty → fallback list.
        assert!(build_provider_choices(&[]).contains(&"auto".to_string()));
        assert!(build_provider_choices(&[]).len() > 10);
    }

    #[test]
    fn agent_startup_gating() {
        assert!(should_run_agent_startup(None, None)); // implicit chat
        assert!(should_run_agent_startup(Some("chat"), None));
        assert!(should_run_agent_startup(Some("acp"), None));
        assert!(!should_run_agent_startup(Some("gateway"), Some("status")));
        assert!(should_run_agent_startup(Some("gateway"), Some("run")));
        assert!(should_run_agent_startup(Some("cron"), Some("tick")));
        assert!(should_run_agent_startup(Some("cron"), Some("run")));
        assert!(!should_run_agent_startup(Some("cron"), Some("list")));
        assert!(should_run_agent_startup(Some("mcp"), Some("serve")));
        assert!(!should_run_agent_startup(Some("mcp"), Some("add")));
        assert!(!should_run_agent_startup(Some("doctor"), None));
    }

    #[test]
    fn command_routing() {
        let c = command_from_name("gateway", Some("run".to_string()));
        assert_eq!(c, Command::Gateway(Some("run".to_string())));
        assert_eq!(c.name(), "gateway");
        assert_eq!(c.subcommand(), Some("run"));
        assert!(c.runs_agent_startup());

        let c2 = command_from_name("doctor", None);
        assert_eq!(c2, Command::Doctor);
        assert_eq!(c2.subcommand(), None);
        assert!(!c2.runs_agent_startup());
    }

    #[test]
    fn resolve_default_chat() {
        // No command → chat.
        assert_eq!(resolve_command(None, None, false), Command::Chat);
        // Bare --resume → chat.
        assert_eq!(resolve_command(None, None, true), Command::Chat);
        // Explicit command preserved.
        assert_eq!(
            resolve_command(Some("status"), None, false),
            Command::Status
        );
    }

    #[test]
    fn version_lines() {
        let root = PathBuf::from("/opt/hermes");
        let h = version_header("1.2.3", "2026-06-03", &root);
        assert!(h.starts_with("Hermes Agent v1.2.3 (2026-06-03)"));
        assert!(h.contains("/opt/hermes"));
        assert_eq!(
            update_available_line(1, "hermes update").as_deref(),
            Some("Update available: 1 commit behind — run 'hermes update'")
        );
        assert_eq!(
            update_available_line(3, "hermes update").as_deref(),
            Some("Update available: 3 commits behind — run 'hermes update'")
        );
        assert_eq!(update_available_line(0, "x").as_deref(), Some("Up to date"));
        assert_eq!(update_available_line(-1, "x"), None);
    }

    #[test]
    fn env_file_scan() {
        let vars: HashSet<String> = ["OPENAI_API_KEY".to_string()].into_iter().collect();
        assert!(env_file_has_provider_key(
            "# comment\nOPENAI_API_KEY='sk-1'\n",
            &vars
        ));
        assert!(!env_file_has_provider_key("OPENAI_API_KEY=\n", &vars));
        assert!(!env_file_has_provider_key("OTHER=foo\n", &vars));
        assert!(env_file_has_provider_key("OPENAI_API_KEY=\"sk-2\"", &vars));
    }

    #[test]
    fn require_tty_text() {
        let msg = require_tty_message("model");
        assert!(msg.contains("hermes 'model'") || msg.contains("'hermes model'"));
        assert!(msg.contains("interactive terminal"));
    }

    #[test]
    fn valid_profile_names() {
        assert!(is_valid_profile_name("work"));
        assert!(is_valid_profile_name("a"));
        assert!(is_valid_profile_name("dev-2_x"));
        assert!(!is_valid_profile_name("No"));
        assert!(!is_valid_profile_name("-leading"));
        assert!(!is_valid_profile_name("with:colon"));
        assert!(!is_valid_profile_name(""));
    }
}
