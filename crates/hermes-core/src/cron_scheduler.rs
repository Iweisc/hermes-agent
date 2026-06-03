//! Cron job scheduler — executes due jobs.
//!
//! Native Rust port of `cron/scheduler.py`.
//!
//! Provides [`tick`] which checks for due jobs and runs them. The gateway calls
//! this every 60 seconds from a background thread. A file-based lock
//! (`~/.hermes/cron/.tick.lock`) ensures only one tick runs at a time if
//! multiple processes overlap.
//!
//! The heavy LLM execution path in the original Python (`AIAgent` construction,
//! provider routing, the inactivity-timeout polling loop) is intentionally
//! abstracted behind the [`AgentRunner`] seam: this keeps the *scheduling*,
//! *delivery-target resolution*, *wake-gate*, *script execution* and
//! *prompt-building* logic — which is where all the subtle behavior lives —
//! fully native and unit-testable, while letting a caller plug in the concrete
//! agent runtime (see `crate::cronjob` for the in-tree agent execution).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use chrono::Local;
use serde_json::Value;

use crate::agent_redact::redact_sensitive_text;
use crate::mod_hermes_constants::get_hermes_home;

/// Sentinel: when a cron agent has nothing new to report, it can start its
/// response with this marker to suppress delivery. Output is still saved
/// locally for audit.
pub const SILENT_MARKER: &str = "[SILENT]";

const DEFAULT_SCRIPT_TIMEOUT: u64 = 120; // seconds
const MAX_CONTEXT_CHARS: usize = 8000;

/// Valid delivery platforms — used to validate user-supplied platform names in
/// cron delivery targets, preventing env-var enumeration via crafted names.
pub const KNOWN_DELIVERY_PLATFORMS: &[&str] = &[
    "telegram",
    "discord",
    "slack",
    "whatsapp",
    "signal",
    "matrix",
    "mattermost",
    "homeassistant",
    "dingtalk",
    "feishu",
    "wecom",
    "wecom_callback",
    "weixin",
    "sms",
    "email",
    "webhook",
    "bluebubbles",
    "qqbot",
    "yuanbao",
];

/// Platforms that support a configured cron/notification home target, mapped to
/// the environment variable used by gateway setup/runtime config.
///
/// Order matters: the deliver=origin fallback iterates these in declaration
/// order and uses the first platform with a configured home channel, matching
/// the insertion order of the Python dict.
pub const HOME_TARGET_ENV_VARS: &[(&str, &str)] = &[
    ("matrix", "MATRIX_HOME_ROOM"),
    ("telegram", "TELEGRAM_HOME_CHANNEL"),
    ("discord", "DISCORD_HOME_CHANNEL"),
    ("slack", "SLACK_HOME_CHANNEL"),
    ("signal", "SIGNAL_HOME_CHANNEL"),
    ("mattermost", "MATTERMOST_HOME_CHANNEL"),
    ("sms", "SMS_HOME_CHANNEL"),
    ("email", "EMAIL_HOME_ADDRESS"),
    ("dingtalk", "DINGTALK_HOME_CHANNEL"),
    ("feishu", "FEISHU_HOME_CHANNEL"),
    ("wecom", "WECOM_HOME_CHANNEL"),
    ("weixin", "WEIXIN_HOME_CHANNEL"),
    ("bluebubbles", "BLUEBUBBLES_HOME_CHANNEL"),
    ("qqbot", "QQBOT_HOME_CHANNEL"),
];

/// Legacy env-var names kept for back-compat. Maps the current primary env var
/// to the previous name; [`get_home_target_chat_id`] falls back to the legacy
/// name if the primary is unset.
pub const LEGACY_HOME_TARGET_ENV_VARS: &[(&str, &str)] = &[("QQBOT_HOME_CHANNEL", "QQ_HOME_CHANNEL")];

const VIDEO_EXTS: &[&str] = &[".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"];
const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".webp", ".gif"];

/// A resolved concrete delivery target for a cron job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryTarget {
    pub platform: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
}

/// Result of executing a cron job (mirrors the Python 4-tuple
/// `(success, full_output_doc, final_response, error_message)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunJobResult {
    pub success: bool,
    pub output_doc: String,
    pub final_response: String,
    pub error: Option<String>,
}

/// Outcome of running a job's pre-run / data-collection script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptResult {
    pub success: bool,
    pub output: String,
}

// ---------------------------------------------------------------------------
// Hermes home / lock paths
// ---------------------------------------------------------------------------

/// Resolve Hermes home dynamically (defers to `crate::mod_hermes_constants`).
fn hermes_home() -> PathBuf {
    get_hermes_home()
}

/// Resolve cron lock paths at call time so profile/env changes are honored.
/// Returns `(lock_dir, lock_file)`.
pub fn get_lock_paths() -> (PathBuf, PathBuf) {
    let lock_dir = hermes_home().join("cron");
    let lock_file = lock_dir.join(".tick.lock");
    (lock_dir, lock_file)
}

// ---------------------------------------------------------------------------
// Origin / home-target resolution
// ---------------------------------------------------------------------------

/// Extract origin info from a job, preserving any extra routing metadata.
///
/// Treats non-dict origins (free-form provenance strings, ints, lists from
/// migration scripts or hand-edited jobs.json) as missing instead of crashing.
/// Only returns the origin when it is an object with both a truthy `platform`
/// and `chat_id`.
pub fn resolve_origin(job: &Value) -> Option<Value> {
    let origin = job.get("origin")?;
    if !origin.is_object() {
        return None;
    }
    let platform_ok = origin
        .get("platform")
        .map(value_is_truthy)
        .unwrap_or(false);
    let chat_ok = origin.get("chat_id").map(value_is_truthy).unwrap_or(false);
    if platform_ok && chat_ok {
        Some(origin.clone())
    } else {
        None
    }
}

/// Python truthiness for JSON values used in origin checks: non-empty string,
/// non-zero number, non-null, non-empty container, `true`.
fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Coerce a JSON value to the string Python's `str(...)` would produce for the
/// shapes seen in origin chat_ids (strings unquoted; ints/floats stringified).
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read an env var, trimming nothing (mirrors `os.getenv(env_var, "")`).
fn env_or_empty(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

fn home_env_var_for(platform_name: &str) -> Option<&'static str> {
    let lower = platform_name.to_lowercase();
    HOME_TARGET_ENV_VARS
        .iter()
        .find(|(p, _)| *p == lower)
        .map(|(_, env)| *env)
}

fn legacy_for(env_var: &str) -> Option<&'static str> {
    LEGACY_HOME_TARGET_ENV_VARS
        .iter()
        .find(|(primary, _)| *primary == env_var)
        .map(|(_, legacy)| *legacy)
}

/// Return the configured home target chat/room ID for a delivery platform.
pub fn get_home_target_chat_id(platform_name: &str) -> String {
    let env_var = match home_env_var_for(platform_name) {
        Some(e) => e,
        None => return String::new(),
    };
    let mut value = env_or_empty(env_var);
    if value.is_empty() {
        if let Some(legacy) = legacy_for(env_var) {
            value = env_or_empty(legacy);
        }
    }
    value
}

/// Return the optional thread/topic ID for a platform home target.
pub fn get_home_target_thread_id(platform_name: &str) -> Option<String> {
    let env_var = home_env_var_for(platform_name)?;
    let mut value = env_or_empty(&format!("{env_var}_THREAD_ID"))
        .trim()
        .to_string();
    if value.is_empty() {
        if let Some(legacy) = legacy_for(env_var) {
            value = env_or_empty(&format!("{legacy}_THREAD_ID"))
                .trim()
                .to_string();
        }
    }
    if value.is_empty() { None } else { Some(value) }
}

// ---------------------------------------------------------------------------
// Deliver value normalization + target resolution
// ---------------------------------------------------------------------------

/// Normalize a stored/submitted `deliver` value to its canonical string form.
///
/// `deliver` is contractually a string (`"local"`, `"origin"`, `"telegram"`,
/// `"telegram:-1001:17"`, or comma-separated combinations). Historically some
/// callers stored a list/tuple like `["telegram"]`. Flatten arrays into a
/// comma-separated string so both forms work. Returns `"local"` for anything
/// falsy.
pub fn normalize_deliver_value(deliver: &Value) -> String {
    match deliver {
        Value::Null => "local".to_string(),
        Value::String(s) if s.is_empty() => "local".to_string(),
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(|p| value_to_str(p).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.is_empty() {
                "local".to_string()
            } else {
                parts.join(",")
            }
        }
        other => value_to_str(other),
    }
}

/// A parsed `platform:rest` target reference seam.
///
/// In the Python this is `tools.send_message_tool._parse_target_ref`, which can
/// split things like `telegram:-1001:17` into `(chat_id, thread_id, explicit)`.
/// We accept an optional resolver so a caller can wire the real parser; without
/// one we apply the documented default (whole `rest` is the chat_id, no thread,
/// not explicit).
pub trait TargetRefParser {
    /// Returns `(chat_id, thread_id, is_explicit)`.
    fn parse(&self, platform_key: &str, rest: &str) -> (String, Option<String>, bool);
}

/// Default parser: treats `rest` as a bare chat_id (mirrors the Python
/// `else: chat_id, thread_id = rest, None` branch when not explicit).
pub struct DefaultTargetRefParser;

impl TargetRefParser for DefaultTargetRefParser {
    fn parse(&self, _platform_key: &str, rest: &str) -> (String, Option<String>, bool) {
        (rest.to_string(), None, false)
    }
}

/// Resolve one concrete auto-delivery target for a cron job.
pub fn resolve_single_delivery_target(
    job: &Value,
    deliver_value: &str,
    parser: &dyn TargetRefParser,
) -> Option<DeliveryTarget> {
    let origin = resolve_origin(job);

    if deliver_value == "local" {
        return None;
    }

    if deliver_value == "origin" {
        if let Some(origin) = &origin {
            return Some(DeliveryTarget {
                platform: value_to_str(origin.get("platform").unwrap_or(&Value::Null)),
                chat_id: value_to_str(origin.get("chat_id").unwrap_or(&Value::Null)),
                thread_id: origin
                    .get("thread_id")
                    .filter(|v| !v.is_null())
                    .map(value_to_str),
            });
        }
        // Origin missing — try each platform's home channel as a fallback.
        for (platform_name, _) in HOME_TARGET_ENV_VARS {
            let chat_id = get_home_target_chat_id(platform_name);
            if !chat_id.is_empty() {
                log::info!(
                    "Job '{}' has deliver=origin but no origin; falling back to {} home channel",
                    job_label(job),
                    platform_name,
                );
                return Some(DeliveryTarget {
                    platform: (*platform_name).to_string(),
                    chat_id,
                    thread_id: get_home_target_thread_id(platform_name),
                });
            }
        }
        return None;
    }

    if let Some((platform_name, rest)) = deliver_value.split_once(':') {
        let platform_key = platform_name.to_lowercase();
        let (parsed_chat_id, parsed_thread_id, is_explicit) = parser.parse(&platform_key, rest);
        let (chat_id, thread_id) = if is_explicit {
            (parsed_chat_id, parsed_thread_id)
        } else {
            (rest.to_string(), None)
        };
        // NOTE: the Python additionally resolves human-friendly labels via
        // gateway.channel_directory.resolve_channel_name; that lookup is a
        // best-effort enrichment wrapped in try/except and is omitted here as
        // it depends on gateway runtime state not available in this module.
        return Some(DeliveryTarget {
            platform: platform_name.to_string(),
            chat_id,
            thread_id,
        });
    }

    let platform_name = deliver_value;
    if let Some(origin) = &origin {
        if value_to_str(origin.get("platform").unwrap_or(&Value::Null)) == platform_name {
            return Some(DeliveryTarget {
                platform: platform_name.to_string(),
                chat_id: value_to_str(origin.get("chat_id").unwrap_or(&Value::Null)),
                thread_id: origin
                    .get("thread_id")
                    .filter(|v| !v.is_null())
                    .map(value_to_str),
            });
        }
    }

    if !KNOWN_DELIVERY_PLATFORMS.contains(&platform_name.to_lowercase().as_str()) {
        return None;
    }
    let chat_id = get_home_target_chat_id(platform_name);
    if chat_id.is_empty() {
        return None;
    }

    Some(DeliveryTarget {
        platform: platform_name.to_string(),
        chat_id,
        thread_id: get_home_target_thread_id(platform_name),
    })
}

/// Resolve all concrete auto-delivery targets for a cron job (supports
/// comma-separated `deliver`).
pub fn resolve_delivery_targets(job: &Value) -> Vec<DeliveryTarget> {
    resolve_delivery_targets_with(job, &DefaultTargetRefParser)
}

/// Like [`resolve_delivery_targets`] but with an injectable target-ref parser.
pub fn resolve_delivery_targets_with(
    job: &Value,
    parser: &dyn TargetRefParser,
) -> Vec<DeliveryTarget> {
    let deliver = normalize_deliver_value(job.get("deliver").unwrap_or(&Value::Null));
    if deliver == "local" {
        return Vec::new();
    }
    let mut seen: BTreeSet<(String, String, Option<String>)> = BTreeSet::new();
    let mut targets = Vec::new();
    for part in deliver.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(target) = resolve_single_delivery_target(job, part, parser) {
            let key = (
                target.platform.to_lowercase(),
                target.chat_id.clone(),
                target.thread_id.clone(),
            );
            if seen.insert(key) {
                targets.push(target);
            }
        }
    }
    targets
}

/// Resolve the concrete auto-delivery target for a cron job, if any.
pub fn resolve_delivery_target(job: &Value) -> Option<DeliveryTarget> {
    resolve_delivery_targets(job).into_iter().next()
}

// ---------------------------------------------------------------------------
// Wake gate
// ---------------------------------------------------------------------------

/// Parse the last non-empty stdout line of a cron job's pre-check script as a
/// wake gate.
///
/// If the last stdout line is JSON like `{"wakeAgent": false}`, the agent is
/// skipped (returns `false`). Any other output (non-JSON, missing flag, gate
/// absent, or `wakeAgent: true`) means wake the agent normally (`true`).
pub fn parse_wake_gate(script_output: &str) -> bool {
    if script_output.is_empty() {
        return true;
    }
    let last_line = match script_output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .next_back()
    {
        Some(l) => l.trim(),
        None => return true,
    };
    let gate: Value = match serde_json::from_str(last_line) {
        Ok(v) => v,
        Err(_) => return true,
    };
    if !gate.is_object() {
        return true;
    }
    // `gate.get("wakeAgent", True) is not False` — only an explicit `false`
    // suppresses; anything else (missing, true, non-bool) wakes.
    !matches!(gate.get("wakeAgent"), Some(Value::Bool(false)))
}

// ---------------------------------------------------------------------------
// Script timeout + execution
// ---------------------------------------------------------------------------

/// Resolve cron pre-run script timeout from env/config with a safe default.
///
/// Precedence: `HERMES_CRON_SCRIPT_TIMEOUT` env (positive int) > config
/// `cron.script_timeout_seconds` (positive int) > default 120s. The Python
/// module-level `_SCRIPT_TIMEOUT` monkeypatch hook has no native analogue.
pub fn get_script_timeout(config: Option<&Value>) -> u64 {
    let env_value = env_or_empty("HERMES_CRON_SCRIPT_TIMEOUT")
        .trim()
        .to_string();
    if !env_value.is_empty() {
        match env_value.parse::<f64>() {
            Ok(f) => {
                let t = f as i64;
                if t > 0 {
                    return t as u64;
                }
            }
            Err(_) => {
                log::warn!(
                    "Invalid HERMES_CRON_SCRIPT_TIMEOUT={env_value:?}; using config/default"
                );
            }
        }
    }

    if let Some(cfg) = config {
        if let Some(configured) = cfg
            .get("cron")
            .and_then(|c| c.get("script_timeout_seconds"))
        {
            if let Some(t) = coerce_positive_int(configured) {
                return t;
            }
        }
    }

    DEFAULT_SCRIPT_TIMEOUT
}

fn coerce_positive_int(v: &Value) -> Option<u64> {
    let f = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    let t = f as i64;
    if t > 0 { Some(t as u64) } else { None }
}

/// Execute a cron job's data-collection script and capture its output.
///
/// Scripts MUST reside within `HERMES_HOME/scripts/`. Both relative and
/// absolute paths are resolved and validated against this directory to prevent
/// arbitrary script execution via path traversal or absolute path injection.
///
/// Interpreter selection by extension: `.sh`/`.bash` run with `/bin/bash`,
/// everything else with `python3` (the original uses `sys.executable`).
///
/// Returns [`ScriptResult`]; on failure the `output` contains the error message
/// so the LLM can report the problem to the user.
pub fn run_job_script(script_path: &str, config: Option<&Value>) -> ScriptResult {
    let scripts_dir = hermes_home().join("scripts");
    if let Err(e) = std::fs::create_dir_all(&scripts_dir) {
        return ScriptResult {
            success: false,
            output: format!("Script execution failed: {e}"),
        };
    }
    let scripts_dir_resolved = scripts_dir.canonicalize().unwrap_or(scripts_dir.clone());

    let raw = expanduser(script_path);
    let candidate = if raw.is_absolute() {
        raw
    } else {
        scripts_dir.join(&raw)
    };
    // Resolve; if the file does not exist canonicalize fails — fall back to a
    // lexically-normalized form so the traversal guard and existence checks
    // still produce sensible messages.
    let path = candidate
        .canonicalize()
        .unwrap_or_else(|_| lexically_normalize(&candidate));

    // Guard against path traversal, absolute path injection, and symlink escape.
    if !path.starts_with(&scripts_dir_resolved) {
        return ScriptResult {
            success: false,
            output: format!(
                "Blocked: script path resolves outside the scripts directory ({}): {:?}",
                scripts_dir_resolved.display(),
                script_path
            ),
        };
    }

    if !path.exists() {
        return ScriptResult {
            success: false,
            output: format!("Script not found: {}", path.display()),
        };
    }
    if !path.is_file() {
        return ScriptResult {
            success: false,
            output: format!("Script path is not a file: {}", path.display()),
        };
    }

    let script_timeout = get_script_timeout(config);

    let suffix = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let mut command = if suffix == ".sh" || suffix == ".bash" {
        let mut c = Command::new("/bin/bash");
        c.arg(&path);
        c
    } else {
        let mut c = Command::new("python3");
        c.arg(&path);
        c
    };
    if let Some(parent) = path.parent() {
        command.current_dir(parent);
    }

    match run_with_timeout(command, Duration::from_secs(script_timeout)) {
        Ok(Some(out)) => {
            let mut stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let mut stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();

            // Redact secrets from both before any return path.
            stdout = redact_sensitive_text(&stdout, false, false);
            stderr = redact_sensitive_text(&stderr, false, false);

            let code = out.status.code();
            if code != Some(0) {
                let mut parts = vec![format!(
                    "Script exited with code {}",
                    code.map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".to_string())
                )];
                if !stderr.is_empty() {
                    parts.push(format!("stderr:\n{stderr}"));
                }
                if !stdout.is_empty() {
                    parts.push(format!("stdout:\n{stdout}"));
                }
                return ScriptResult {
                    success: false,
                    output: parts.join("\n"),
                };
            }
            ScriptResult {
                success: true,
                output: stdout,
            }
        }
        Ok(None) => ScriptResult {
            success: false,
            output: format!("Script timed out after {script_timeout}s: {}", path.display()),
        },
        Err(e) => ScriptResult {
            success: false,
            output: format!("Script execution failed: {e}"),
        },
    }
}

/// Run a command, killing it (returning `Ok(None)`) if it exceeds `timeout`.
fn run_with_timeout(
    mut command: Command,
    timeout: Duration,
) -> std::io::Result<Option<std::process::Output>> {
    use std::process::Stdio;
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait()? {
            Some(_) => {
                let out = child.wait_with_output()?;
                return Ok(Some(out));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Expand a leading `~` to the user's home directory (mirrors
/// `Path(...).expanduser()`).
fn expanduser(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    } else if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(p)
}

/// Lexically normalize a path (resolve `.`/`..` without touching the
/// filesystem) for use when `canonicalize` fails on a non-existent target.
fn lexically_normalize(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Prompt building
// ---------------------------------------------------------------------------

const CRON_HINT: &str = "[IMPORTANT: You are running as a scheduled cron job. \
DELIVERY: Your final response will be automatically delivered \
to the user — do NOT use send_message or try to deliver \
the output yourself. Just produce your report/output as your \
final response and the system handles the rest. \
SILENT: If there is genuinely nothing new to report, respond \
with exactly \"[SILENT]\" (nothing else) to suppress delivery. \
Never combine [SILENT] with content — either report your \
findings normally, or say [SILENT] and nothing more.]\n\n";

/// A loaded skill: `(content, found)` — mirrors the relevant fields of
/// `tools.skills_tool.skill_view`'s JSON result.
pub trait SkillLoader {
    /// Returns `Ok(content)` if the skill loaded, `Err(error_message)` otherwise.
    fn load(&self, skill_name: &str) -> Result<String, String>;
    /// Bump usage statistics; best-effort.
    fn bump_use(&self, _skill_name: &str) {}
}

/// Build the effective prompt for a cron job, optionally loading one or more
/// skills first.
///
/// `prerun_script` carries an already-executed `(success, stdout)` so the
/// script is not re-run when the caller has done a wake-gate check. When
/// `None`, the configured script (if any) runs inline.
///
/// `skill_loader` is the seam for skill content; when `None`, skills are not
/// loaded (the prompt is returned with script/context injections + cron hint).
pub fn build_job_prompt(
    job: &Value,
    prerun_script: Option<&ScriptResult>,
    config: Option<&Value>,
    skill_loader: Option<&dyn SkillLoader>,
) -> String {
    let mut prompt = job
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Run data-collection script if configured, inject output as context.
    if let Some(script_path) = job.get("script").and_then(|v| v.as_str()) {
        if !script_path.is_empty() {
            let result = match prerun_script {
                Some(r) => r.clone(),
                None => run_job_script(script_path, config),
            };
            if result.success {
                if !result.output.is_empty() {
                    prompt = format!(
                        "## Script Output\nThe following data was collected by a pre-run script. \
Use it as context for your analysis.\n\n```\n{}\n```\n\n{}",
                        result.output, prompt
                    );
                } else {
                    prompt = format!(
                        "## Script Output\nThe pre-run script completed successfully but produced no output.\n\n{prompt}"
                    );
                }
            } else {
                prompt = format!(
                    "## Script Error\nThe data-collection script failed. Report this to the user.\n\n```\n{}\n```\n\n{}",
                    result.output, prompt
                );
            }
        }
    }

    // Inject output from referenced cron jobs as context.
    let context_from = collect_context_from(job);
    let output_dir = hermes_home().join("cron").join("output");
    for source_job_id in context_from {
        // Guard against path traversal — valid job IDs are hex strings.
        if source_job_id.is_empty()
            || !source_job_id.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
        {
            log::warn!("context_from: skipping invalid job_id {source_job_id:?}");
            continue;
        }
        let job_output_dir = output_dir.join(&source_job_id);
        if !job_output_dir.exists() {
            continue;
        }
        let latest = match latest_md_output(&job_output_dir) {
            Some(s) => s,
            None => continue,
        };
        let mut latest_output = latest.trim().to_string();
        if latest_output.len() > MAX_CONTEXT_CHARS {
            latest_output = format!(
                "{}\n\n[... output truncated ...]",
                &latest_output[..byte_floor_boundary(&latest_output, MAX_CONTEXT_CHARS)]
            );
        }
        if latest_output.is_empty() {
            continue;
        }
        prompt = format!(
            "## Output from job '{source_job_id}'\nThe following is the most recent output from a preceding \
cron job. Use it as context for your analysis.\n\n```\n{latest_output}\n```\n\n{prompt}"
        );
    }

    // Always prepend cron execution guidance.
    prompt = format!("{CRON_HINT}{prompt}");

    // Resolve the skill list: explicit `skills`, else legacy `skill`.
    let mut skill_names: Vec<String> = match job.get("skills") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(value_to_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::Null) | None => {
            let legacy = job.get("skill").and_then(|v| v.as_str()).unwrap_or("");
            if legacy.is_empty() {
                Vec::new()
            } else {
                vec![legacy.to_string()]
            }
        }
        // Non-list, non-null skills value: iterate as Python would over the
        // truthy value (best-effort string coercion of a scalar).
        Some(other) => {
            let s = value_to_str(other);
            let t = s.trim().to_string();
            if t.is_empty() { Vec::new() } else { vec![t] }
        }
    };
    skill_names.retain(|s| !s.is_empty());

    if skill_names.is_empty() {
        return prompt;
    }

    let loader = match skill_loader {
        Some(l) => l,
        // No loader available: cannot load skills, so return the prompt as-is
        // (the script/context/hint injections are preserved).
        None => return prompt,
    };

    let mut parts: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for skill_name in &skill_names {
        match loader.load(skill_name) {
            Ok(content) => {
                loader.bump_use(skill_name);
                let content = content.trim().to_string();
                if !parts.is_empty() {
                    parts.push(String::new());
                }
                parts.push(format!(
                    "[IMPORTANT: The user has invoked the \"{skill_name}\" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]"
                ));
                parts.push(String::new());
                parts.push(content);
            }
            Err(error) => {
                log::warn!(
                    "Cron job '{}': skill not found, skipping — {}",
                    job_label(job),
                    error
                );
                skipped.push(skill_name.clone());
            }
        }
    }

    if !skipped.is_empty() {
        let joined = skipped.join(", ");
        let notice = format!(
            "[IMPORTANT: The following skill(s) were listed for this job but could not be found \
and were skipped: {joined}. \
Start your response with a brief notice so the user is aware, e.g.: \
'⚠️ Skill(s) not found and skipped: {joined}']"
        );
        parts.insert(0, notice);
    }

    if !prompt.is_empty() {
        parts.push(String::new());
        parts.push(format!(
            "The user has provided the following instruction alongside the skill invocation: {prompt}"
        ));
    }
    parts.join("\n")
}

fn collect_context_from(job: &Value) -> Vec<String> {
    match job.get("context_from") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr.iter().map(value_to_str).collect(),
        _ => Vec::new(),
    }
}

/// Return the contents of the most-recently-modified `*.md` file in `dir`.
fn latest_md_output(dir: &Path) -> Option<String> {
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) == Some("md") {
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((mtime, path))
            } else {
                None
            }
        })
        .collect();
    if entries.is_empty() {
        return None;
    }
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    std::fs::read_to_string(&entries[0].1).ok()
}

/// Largest byte index `<= n` that lies on a char boundary of `s`.
fn byte_floor_boundary(s: &str, n: usize) -> usize {
    if n >= s.len() {
        return s.len();
    }
    let mut idx = n;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn job_label(job: &Value) -> String {
    if let Some(name) = job.get("name").and_then(|v| v.as_str()) {
        if !name.is_empty() {
            return name.to_string();
        }
    }
    job.get("id")
        .map(value_to_str)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "?".to_string())
}

// ---------------------------------------------------------------------------
// no_agent run path + agent seam
// ---------------------------------------------------------------------------

/// Seam for executing the LLM agent portion of a cron job.
///
/// The Python builds an `AIAgent` with provider routing, runs it with an
/// inactivity-timeout polling loop, and returns the final response. That
/// machinery lives in `crate::cronjob` (the in-tree agent runtime); callers
/// pass an implementor here so the scheduler stays runtime-agnostic.
pub trait AgentRunner {
    /// Run the agent for `job` with the fully-built `prompt`.
    ///
    /// Returns the agent's `final_response` on success, or an error string.
    fn run(&self, job: &Value, prompt: &str) -> Result<String, String>;
}

/// Run the `no_agent` short-circuit path: the script IS the job, no LLM.
///
/// Semantics (matching Python `run_job`):
/// * script stdout (trimmed) → delivered verbatim as the final message
/// * empty stdout → silent run (no delivery, success=true)
/// * non-zero exit / timeout → delivered as an error alert, success=false
/// * `wakeAgent=false` gate → treated like empty stdout (silent)
///
/// Honours the optional per-job `workdir` as the subprocess cwd by `cd`-ing the
/// process for the duration (matching Python's `os.chdir`), restoring it after.
pub fn run_no_agent_job(job: &Value, config: Option<&Value>) -> RunJobResult {
    let job_id = job.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let job_name = job.get("name").and_then(|v| v.as_str()).unwrap_or("");

    let script_path = job
        .get("script")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let script_path = match script_path {
        Some(p) => p,
        None => {
            let err = "no_agent=True but no script is set for this job".to_string();
            log::error!("Job '{job_id}': {err}");
            return RunJobResult {
                success: false,
                output_doc: String::new(),
                final_response: String::new(),
                error: Some(err),
            };
        }
    };

    // Apply workdir if configured (subprocess cwd).
    let workdir = job
        .get("workdir")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let prior_cwd = match workdir {
        Some(wd) if Path::new(wd).is_dir() => {
            let prior = std::env::current_dir().ok();
            if std::env::set_current_dir(wd).is_ok() {
                prior
            } else {
                None
            }
        }
        _ => None,
    };

    let result = run_job_script(script_path, config);

    if let Some(prior) = prior_cwd {
        let _ = std::env::set_current_dir(prior);
    }

    let now_iso = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    if !result.success {
        let output = result.output;
        let alert = format!(
            "⚠ Cron watchdog '{job_name}' script failed\n\n{output}\n\nTime: {now_iso}"
        );
        let doc = format!(
            "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {now_iso}\n**Mode:** no_agent (script)\n**Status:** script failed\n\n{output}\n"
        );
        return RunJobResult {
            success: false,
            output_doc: doc,
            final_response: alert,
            error: Some(output),
        };
    }

    if !parse_wake_gate(&result.output) {
        log::info!("Job '{job_id}' (no_agent): wakeAgent=false gate — silent run");
        let doc = format!(
            "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {now_iso}\n**Mode:** no_agent (script)\n**Status:** silent (wakeAgent=false)\n"
        );
        return RunJobResult {
            success: true,
            output_doc: doc,
            final_response: SILENT_MARKER.to_string(),
            error: None,
        };
    }

    if result.output.trim().is_empty() {
        log::info!("Job '{job_id}' (no_agent): empty stdout — silent run");
        let doc = format!(
            "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {now_iso}\n**Mode:** no_agent (script)\n**Status:** silent (empty output)\n"
        );
        return RunJobResult {
            success: true,
            output_doc: doc,
            final_response: SILENT_MARKER.to_string(),
            error: None,
        };
    }

    let output = result.output;
    let doc = format!(
        "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {now_iso}\n**Mode:** no_agent (script)\n\n---\n\n{output}\n"
    );
    RunJobResult {
        success: true,
        output_doc: doc,
        final_response: output,
        error: None,
    }
}

/// Execute a single cron job (native analogue of Python `run_job`).
///
/// `no_agent` jobs are handled entirely by [`run_no_agent_job`]. Otherwise the
/// prompt is built (running the pre-check script once, honouring the wake gate)
/// and dispatched to the supplied [`AgentRunner`], wrapping the response in the
/// standard output document.
pub fn run_job(
    job: &Value,
    config: Option<&Value>,
    skill_loader: Option<&dyn SkillLoader>,
    agent: &dyn AgentRunner,
) -> RunJobResult {
    let job_id = job.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let job_name = job.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let schedule_display = job
        .get("schedule_display")
        .and_then(|v| v.as_str())
        .unwrap_or("N/A");

    if job.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false) {
        return run_no_agent_job(job, config);
    }

    // Wake-gate: run the pre-check script BEFORE building the prompt so a
    // `{"wakeAgent": false}` response can short-circuit the agent run.
    let mut prerun: Option<ScriptResult> = None;
    if let Some(script_path) = job.get("script").and_then(|v| v.as_str()) {
        if !script_path.is_empty() {
            let r = run_job_script(script_path, config);
            if r.success && !parse_wake_gate(&r.output) {
                log::info!(
                    "Job '{job_name}' (ID: {job_id}): wakeAgent=false, skipping agent run"
                );
                let now_iso = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                let doc = format!(
                    "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {now_iso}\n\nScript gate returned `wakeAgent=false` — agent skipped.\n"
                );
                return RunJobResult {
                    success: true,
                    output_doc: doc,
                    final_response: SILENT_MARKER.to_string(),
                    error: None,
                };
            }
            prerun = Some(r);
        }
    }

    let prompt = build_job_prompt(job, prerun.as_ref(), config, skill_loader);

    log::info!("Running job '{job_name}' (ID: {job_id})");

    match agent.run(job, &prompt) {
        Ok(final_response) => {
            // Strip leaked placeholder upstream may inject on empty completions.
            let final_response = if final_response.trim() == "(No response generated)" {
                String::new()
            } else {
                final_response
            };
            let logged_response = if final_response.is_empty() {
                "(No response generated)"
            } else {
                &final_response
            };
            let run_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let output = format!(
                "# Cron Job: {job_name}\n\n**Job ID:** {job_id}\n**Run Time:** {run_time}\n**Schedule:** {schedule_display}\n\n## Prompt\n\n{prompt}\n\n## Response\n\n{logged_response}\n"
            );
            log::info!("Job '{job_name}' completed successfully");
            RunJobResult {
                success: true,
                output_doc: output,
                final_response,
                error: None,
            }
        }
        Err(error_msg) => {
            log::error!("Job '{job_name}' failed: {error_msg}");
            let run_time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let output = format!(
                "# Cron Job: {job_name} (FAILED)\n\n**Job ID:** {job_id}\n**Run Time:** {run_time}\n**Schedule:** {schedule_display}\n\n## Prompt\n\n{prompt}\n\n## Error\n\n```\n{error_msg}\n```\n"
            );
            RunJobResult {
                success: false,
                output_doc: output,
                final_response: String::new(),
                error: Some(error_msg),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Delivery dispatch helpers (pure parts)
// ---------------------------------------------------------------------------

/// Determine whether a file extension routes as video.
pub fn is_video_ext(ext: &str) -> bool {
    VIDEO_EXTS.contains(&ext.to_lowercase().as_str())
}

/// Determine whether a file extension routes as an image.
pub fn is_image_ext(ext: &str) -> bool {
    IMAGE_EXTS.contains(&ext.to_lowercase().as_str())
}

/// Wrap job output with the standard cron delivery header/footer.
///
/// Wrapping is on by default; pass `wrap_response = false` (resolved from
/// `cron.wrap_response`) for clean output. Mirrors the Python `_deliver_result`
/// wrapping block.
pub fn wrap_delivery_content(job: &Value, content: &str, wrap_response: bool) -> String {
    if !wrap_response {
        return content.to_string();
    }
    let job_id = job.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let task_name = job
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(job_id);
    format!(
        "Cronjob Response: {task_name}\n(job_id: {job_id})\n-------------\n\n{content}\n\nTo stop or manage this job, send me a new message (e.g. \"stop reminder {task_name}\")."
    )
}

/// Resolve the `cron.wrap_response` config flag (default `true`).
pub fn resolve_wrap_response(config: Option<&Value>) -> bool {
    config
        .and_then(|c| c.get("cron"))
        .and_then(|c| c.get("wrap_response"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

/// Decide whether a job's `final_response` should suppress delivery.
///
/// Mirrors the tick `should_deliver` gate: an empty response never delivers,
/// and a successful run whose response contains `[SILENT]` (case-insensitive,
/// after trimming) is suppressed.
pub fn should_suppress_delivery(success: bool, final_response: &str) -> bool {
    if final_response.is_empty() {
        return true;
    }
    if success && final_response.trim().to_uppercase().contains(SILENT_MARKER) {
        return true;
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

    // Guards against env-var test interleaving (tests touch process-global env).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn normalize_deliver_handles_all_shapes() {
        assert_eq!(normalize_deliver_value(&Value::Null), "local");
        assert_eq!(normalize_deliver_value(&json!("")), "local");
        assert_eq!(normalize_deliver_value(&json!("telegram")), "telegram");
        assert_eq!(
            normalize_deliver_value(&json!(["telegram"])),
            "telegram"
        );
        assert_eq!(
            normalize_deliver_value(&json!(["telegram", " discord "])),
            "telegram,discord"
        );
        assert_eq!(normalize_deliver_value(&json!([])), "local");
        assert_eq!(normalize_deliver_value(&json!(["", "  "])), "local");
    }

    #[test]
    fn resolve_origin_rejects_non_dict_and_partial() {
        assert_eq!(resolve_origin(&json!({})), None);
        assert_eq!(
            resolve_origin(&json!({"origin": "some-string"})),
            None
        );
        assert_eq!(
            resolve_origin(&json!({"origin": {"platform": "telegram"}})),
            None
        );
        let job = json!({"origin": {"platform": "telegram", "chat_id": 123}});
        let got = resolve_origin(&job).unwrap();
        assert_eq!(got.get("platform").unwrap(), "telegram");
    }

    #[test]
    fn local_deliver_yields_no_targets() {
        let job = json!({"deliver": "local"});
        assert!(resolve_delivery_targets(&job).is_empty());
        let job = json!({});
        assert!(resolve_delivery_targets(&job).is_empty());
    }

    #[test]
    fn origin_deliver_uses_origin() {
        let job = json!({
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": -1001, "thread_id": 17}
        });
        let target = resolve_delivery_target(&job).unwrap();
        assert_eq!(target.platform, "telegram");
        assert_eq!(target.chat_id, "-1001");
        assert_eq!(target.thread_id.as_deref(), Some("17"));
    }

    #[test]
    fn explicit_platform_colon_target() {
        let job = json!({"deliver": "telegram:-1001:17"});
        let target = resolve_delivery_target(&job).unwrap();
        assert_eq!(target.platform, "telegram");
        // Default parser treats the whole rest as chat_id.
        assert_eq!(target.chat_id, "-1001:17");
        assert_eq!(target.thread_id, None);
    }

    #[test]
    fn comma_separated_dedups() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("TELEGRAM_HOME_CHANNEL", "999");
        }
        let job = json!({"deliver": "telegram,telegram"});
        let targets = resolve_delivery_targets(&job);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].chat_id, "999");
        unsafe {
            std::env::remove_var("TELEGRAM_HOME_CHANNEL");
        }
    }

    #[test]
    fn unknown_platform_rejected() {
        let _g = ENV_LOCK.lock().unwrap();
        let job = json!({"deliver": "bogusplatform"});
        assert!(resolve_delivery_target(&job).is_none());
    }

    #[test]
    fn home_target_legacy_fallback() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("QQBOT_HOME_CHANNEL");
            std::env::set_var("QQ_HOME_CHANNEL", "legacy-room");
        }
        assert_eq!(get_home_target_chat_id("qqbot"), "legacy-room");
        unsafe {
            std::env::remove_var("QQ_HOME_CHANNEL");
        }
    }

    #[test]
    fn wake_gate_parsing() {
        assert!(parse_wake_gate(""));
        assert!(parse_wake_gate("hello\nworld"));
        assert!(parse_wake_gate("not json {"));
        assert!(parse_wake_gate("{\"wakeAgent\": true}"));
        assert!(parse_wake_gate("{\"other\": 1}"));
        assert!(!parse_wake_gate("{\"wakeAgent\": false}"));
        // last non-empty line is the gate
        assert!(!parse_wake_gate("noise\n\n{\"wakeAgent\": false}\n\n"));
        assert!(parse_wake_gate("{\"wakeAgent\": false}\ntrailing"));
    }

    #[test]
    fn script_timeout_env_and_config() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("HERMES_CRON_SCRIPT_TIMEOUT");
        }
        assert_eq!(get_script_timeout(None), DEFAULT_SCRIPT_TIMEOUT);

        let cfg = json!({"cron": {"script_timeout_seconds": 45}});
        assert_eq!(get_script_timeout(Some(&cfg)), 45);
        let cfg = json!({"cron": {"script_timeout_seconds": "30"}});
        assert_eq!(get_script_timeout(Some(&cfg)), 30);
        // non-positive ignored
        let cfg = json!({"cron": {"script_timeout_seconds": 0}});
        assert_eq!(get_script_timeout(Some(&cfg)), DEFAULT_SCRIPT_TIMEOUT);

        unsafe {
            std::env::set_var("HERMES_CRON_SCRIPT_TIMEOUT", "77");
        }
        assert_eq!(get_script_timeout(Some(&cfg)), 77);
        unsafe {
            std::env::remove_var("HERMES_CRON_SCRIPT_TIMEOUT");
        }
    }

    #[test]
    fn build_prompt_prepends_hint_and_script_output() {
        let job = json!({"prompt": "Summarize the weather."});
        let prompt = build_job_prompt(&job, None, None, None);
        assert!(prompt.starts_with(CRON_HINT));
        assert!(prompt.contains("Summarize the weather."));

        // With a successful prerun script, output is injected before the prompt.
        let job = json!({"prompt": "do it", "script": "x.py"});
        let prerun = ScriptResult {
            success: true,
            output: "DATA=1".to_string(),
        };
        let prompt = build_job_prompt(&job, Some(&prerun), None, None);
        assert!(prompt.contains("## Script Output"));
        assert!(prompt.contains("DATA=1"));
        // hint comes first overall
        assert!(prompt.starts_with(CRON_HINT));
    }

    #[test]
    fn build_prompt_script_error_branch() {
        let job = json!({"prompt": "p", "script": "x.py"});
        let prerun = ScriptResult {
            success: false,
            output: "boom".to_string(),
        };
        let prompt = build_job_prompt(&job, Some(&prerun), None, None);
        assert!(prompt.contains("## Script Error"));
        assert!(prompt.contains("boom"));
    }

    struct StubSkills;
    impl SkillLoader for StubSkills {
        fn load(&self, name: &str) -> Result<String, String> {
            if name == "good" {
                Ok("SKILL BODY".to_string())
            } else {
                Err(format!("no skill {name}"))
            }
        }
    }

    #[test]
    fn build_prompt_with_skills() {
        let job = json!({"prompt": "instr", "skills": ["good", "missing"]});
        let prompt = build_job_prompt(&job, None, None, Some(&StubSkills));
        assert!(prompt.contains("SKILL BODY"));
        assert!(prompt.contains("could not be found"));
        assert!(prompt.contains("missing"));
        assert!(prompt.contains("instruction alongside the skill invocation: "));
    }

    #[test]
    fn build_prompt_legacy_skill_field() {
        let job = json!({"prompt": "instr", "skill": "good"});
        let prompt = build_job_prompt(&job, None, None, Some(&StubSkills));
        assert!(prompt.contains("SKILL BODY"));
    }

    #[test]
    fn suppress_delivery_logic() {
        assert!(should_suppress_delivery(true, ""));
        assert!(should_suppress_delivery(true, "  [silent] "));
        assert!(should_suppress_delivery(true, "[SILENT]"));
        assert!(!should_suppress_delivery(true, "real content"));
        // failed jobs with content still deliver
        assert!(!should_suppress_delivery(false, "error report"));
    }

    #[test]
    fn wrap_delivery_content_formats() {
        let job = json!({"id": "abc123", "name": "Weather"});
        let wrapped = wrap_delivery_content(&job, "sunny", true);
        assert!(wrapped.starts_with("Cronjob Response: Weather"));
        assert!(wrapped.contains("(job_id: abc123)"));
        assert!(wrapped.contains("sunny"));
        assert!(wrapped.contains("stop reminder Weather"));

        let plain = wrap_delivery_content(&job, "sunny", false);
        assert_eq!(plain, "sunny");
    }

    #[test]
    fn ext_routing() {
        assert!(is_video_ext(".MP4"));
        assert!(is_image_ext(".png"));
        assert!(!is_video_ext(".png"));
        assert!(!is_image_ext(".mp4"));
    }

    #[test]
    fn no_agent_requires_script() {
        let job = json!({"id": "j1", "name": "watchdog", "no_agent": true});
        let r = run_no_agent_job(&job, None);
        assert!(!r.success);
        assert_eq!(
            r.error.as_deref(),
            Some("no_agent=True but no script is set for this job")
        );
    }

    struct StubAgent(&'static str);
    impl AgentRunner for StubAgent {
        fn run(&self, _job: &Value, _prompt: &str) -> Result<String, String> {
            Ok(self.0.to_string())
        }
    }

    #[test]
    fn run_job_agent_success_doc() {
        let job = json!({"id": "j1", "name": "Daily", "prompt": "p", "schedule_display": "every day"});
        let r = run_job(&job, None, None, &StubAgent("the report"));
        assert!(r.success);
        assert_eq!(r.final_response, "the report");
        assert!(r.output_doc.contains("# Cron Job: Daily"));
        assert!(r.output_doc.contains("**Schedule:** every day"));
        assert!(r.output_doc.contains("## Response"));
        assert!(r.output_doc.contains("the report"));
    }

    #[test]
    fn run_job_empty_response_uses_placeholder_in_doc() {
        let job = json!({"id": "j1", "name": "Daily", "prompt": "p"});
        let r = run_job(&job, None, None, &StubAgent("(No response generated)"));
        assert!(r.success);
        assert_eq!(r.final_response, "");
        assert!(r.output_doc.contains("(No response generated)"));
    }

    struct FailAgent;
    impl AgentRunner for FailAgent {
        fn run(&self, _job: &Value, _prompt: &str) -> Result<String, String> {
            Err("RuntimeError: kaboom".to_string())
        }
    }

    #[test]
    fn run_job_agent_failure_doc() {
        let job = json!({"id": "j1", "name": "Daily", "prompt": "p"});
        let r = run_job(&job, None, None, &FailAgent);
        assert!(!r.success);
        assert_eq!(r.final_response, "");
        assert_eq!(r.error.as_deref(), Some("RuntimeError: kaboom"));
        assert!(r.output_doc.contains("(FAILED)"));
        assert!(r.output_doc.contains("## Error"));
    }

    #[test]
    fn lock_paths_under_cron() {
        let (dir, file) = get_lock_paths();
        assert!(dir.ends_with("cron"));
        assert!(file.ends_with(".tick.lock"));
    }
}
