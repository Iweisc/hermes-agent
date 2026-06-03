//! Native Rust port of `gateway/run.py` — the gateway runner entry point.
//!
//! The Python module is a ~15K-LOC long-running async daemon that wires
//! together every messaging-platform adapter, the agent runtime, the session
//! database, restart/drain watchers, and dozens of `/command` handlers.  The
//! vast majority of that surface is async orchestration bound to the Python
//! ecosystem (asyncio event loop, live platform adapters, the AIAgent class,
//! the SessionDB) that has no faithful standalone Rust expression.
//!
//! What *is* faithfully portable — and is reproduced here exactly — is the
//! deterministic, side-effect-light core that the orchestration relies on:
//!
//!   * timestamp coercion + auto-continue freshness gating
//!   * env-var float/string parsing with config-default fallbacks
//!   * config.yaml → env-var bridging (terminal/auxiliary/agent/display/...)
//!   * SSL CA-bundle auto-detection candidate ordering
//!   * session-key parsing + platform/config-key mapping
//!   * gateway model resolution from config.yaml
//!   * media placeholder + empty-response normalization
//!   * control-interrupt message classification + interrupt reason constants
//!   * SKILL.md frontmatter → command-slug derivation
//!   * background-notification / busy-input / service-tier / reasoning loaders
//!   * voice-mode persistence + adapter sync key math
//!   * Telegram topic lobby/lane classification + canned messages
//!   * agent-config cache signature + cache-busting key extraction
//!   * cron-ticker cadence math
//!   * process-watcher notification formatting
//!
//! The few config loaders that read `~/.hermes/config.yaml` take the hermes
//! home dir as a parameter (matching the Python module-level `_hermes_home`
//! that tests monkeypatch) so they stay testable without a global.
//!
//! Cross-refs: `crate::gateway` (restart-drain timeout + service exit code),
//! `crate::gateway_whatsapp_identity` (identity canonicalisation).

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use crate::gateway::{
    parse_restart_drain_timeout, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
};

// ---------------------------------------------------------------------------
// Module-level constants
// ---------------------------------------------------------------------------

/// Bounds the per-session AIAgent cache to prevent unbounded growth.
pub const AGENT_CACHE_MAX_SIZE: usize = 128;
/// Evict agents idle for >1h.
pub const AGENT_CACHE_IDLE_TTL_SECS: f64 = 3600.0;
pub const PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT: f64 = 30.0;

/// Default auto-continue freshness window: 1 hour.
pub const AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT: f64 = 60.0 * 60.0;

/// Interrupt reason strings — internal control flow that should not be echoed
/// to users as if it were an agent error.
pub const INTERRUPT_REASON_STOP: &str = "Stop requested";
pub const INTERRUPT_REASON_RESET: &str = "Session reset requested";
pub const INTERRUPT_REASON_TIMEOUT: &str = "Execution timed out (inactivity)";
pub const INTERRUPT_REASON_SSE_DISCONNECT: &str = "SSE client disconnected";
pub const INTERRUPT_REASON_GATEWAY_SHUTDOWN: &str = "Gateway shutting down";
pub const INTERRUPT_REASON_GATEWAY_RESTART: &str = "Gateway restarting";

/// Telegram's General (pinned top) topic ids treated as "root".
pub const TELEGRAM_GENERAL_TOPIC_IDS: [&str; 2] = ["", "1"];
/// Rate-limit root-DM lobby reminders to one per cooldown window.
pub const TELEGRAM_LOBBY_REMINDER_COOLDOWN_S: f64 = 30.0;

/// Docker media output container paths that count as "output" mounts.
pub const DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS: [&str; 2] = ["/output", "/outputs"];

/// Config (section, key) pairs whose changes must bust the cached agent.
pub const CACHE_BUSTING_CONFIG_KEYS: [(&str, &str); 6] = [
    ("model", "context_length"),
    ("compression", "enabled"),
    ("compression", "threshold"),
    ("compression", "target_ratio"),
    ("compression", "protect_last_n"),
    ("agent", "disabled_toolsets"),
];

// ---------------------------------------------------------------------------
// Timestamp coercion + auto-continue freshness
// ---------------------------------------------------------------------------

/// Best-effort conversion of stored gateway timestamps to epoch seconds.
///
/// Faithful port of `_coerce_gateway_timestamp`.  Accepts JSON numbers (epoch
/// seconds, or milliseconds when the magnitude exceeds ~year-2286), ISO-8601
/// strings (with or without a trailing `Z`), and numeric strings.  Booleans
/// (a subclass of int in Python) and nulls return `None`.
pub fn coerce_gateway_timestamp(value: &Value) -> Option<f64> {
    match value {
        Value::Null => None,
        Value::Bool(_) => None, // bool is a subclass of int — skip it
        Value::Number(n) => {
            let f = n.as_f64()?;
            Some(if f > 10_000_000_000.0 { f / 1000.0 } else { f })
        }
        Value::String(s) => coerce_gateway_timestamp_str(s),
        _ => None,
    }
}

/// Coerce a raw string timestamp (numeric or ISO-8601) to epoch seconds.
pub fn coerce_gateway_timestamp_str(value: &str) -> Option<f64> {
    let text = value.trim();
    if text.is_empty() {
        return None;
    }
    if let Ok(numeric) = text.parse::<f64>() {
        return Some(if numeric > 10_000_000_000.0 {
            numeric / 1000.0
        } else {
            numeric
        });
    }
    // datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    let normalized = text.replace('Z', "+00:00");
    parse_iso8601_epoch(&normalized)
}

/// Parse an ISO-8601 datetime string into epoch seconds, mirroring Python's
/// `datetime.fromisoformat(...).timestamp()` for the common forms.
fn parse_iso8601_epoch(text: &str) -> Option<f64> {
    use chrono::{DateTime, NaiveDateTime};
    // With timezone offset.
    if let Ok(dt) = DateTime::parse_from_rfc3339(text) {
        return Some(dt.timestamp() as f64 + (dt.timestamp_subsec_nanos() as f64) / 1e9);
    }
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f%:z",
        "%Y-%m-%dT%H:%M:%S%:z",
        "%Y-%m-%d %H:%M:%S%:z",
    ] {
        if let Ok(dt) = DateTime::parse_from_str(text, fmt) {
            return Some(dt.timestamp() as f64 + (dt.timestamp_subsec_nanos() as f64) / 1e9);
        }
    }
    // Naive (no tz) — Python fromisoformat treats this as a naive datetime and
    // .timestamp() interprets it in local time; we approximate with UTC.
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(text, fmt) {
            return Some(ndt.and_utc().timestamp() as f64);
        }
    }
    None
}

/// Read `HERMES_AUTO_CONTINUE_FRESHNESS` from the environment, falling back to
/// the module default when unset or malformed.
pub fn auto_continue_freshness_window() -> f64 {
    match std::env::var("HERMES_AUTO_CONTINUE_FRESHNESS") {
        Ok(raw) if !raw.is_empty() => raw
            .parse::<f64>()
            .unwrap_or(AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT),
        _ => AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT,
    }
}

/// Read an env var as float, falling back to `default` on typos/empty/unset.
///
/// Faithful port of `_float_env`.
pub fn float_env(name: &str, default: f64) -> f64 {
    match std::env::var(name) {
        Ok(raw) if !raw.is_empty() => raw.parse::<f64>().unwrap_or(default),
        _ => default,
    }
}

/// Return `true` when an interruption marker is fresh enough to auto-continue.
///
/// Unknown timestamps are treated as fresh (legacy transcripts / in-memory test
/// scaffolding).  A non-positive `window_secs` disables the gate.
pub fn is_fresh_gateway_interruption(
    value: &Value,
    now: Option<f64>,
    window_secs: Option<f64>,
) -> bool {
    let window = window_secs.unwrap_or(AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT);
    if window <= 0.0 {
        return true;
    }
    let timestamp = match coerce_gateway_timestamp(value) {
        Some(ts) => ts,
        None => return true,
    };
    let current = now.unwrap_or_else(now_epoch_secs);
    current - timestamp <= window
}

/// Return the `timestamp` of the last usable transcript row, if any.
///
/// Skips metadata-only rows (`session_meta`, `system`).  Returns `None` when no
/// usable row carries a timestamp (legacy transcript → treat as fresh).
pub fn last_transcript_timestamp(history: &[Value]) -> Option<Value> {
    for msg in history.iter().rev() {
        let obj = match msg.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");
        if role.is_empty() || role == "session_meta" || role == "system" {
            continue;
        }
        match obj.get("timestamp") {
            Some(ts) if !ts.is_null() => return Some(ts.clone()),
            // First non-meta row without a timestamp — legacy transcript row.
            _ => return None,
        }
    }
    None
}

fn now_epoch_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// SSL CA-bundle auto-detection
// ---------------------------------------------------------------------------

/// Common distro/macOS CA-bundle locations, in the exact probe order used by
/// `_ensure_ssl_certs` (after the python-compiled-in defaults + certifi).
pub const SSL_CERT_CANDIDATE_PATHS: [&str; 9] = [
    "/etc/ssl/certs/ca-certificates.crt",                // Debian/Ubuntu/Gentoo
    "/etc/pki/tls/certs/ca-bundle.crt",                  // RHEL/CentOS 7
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", // RHEL/CentOS 8+
    "/etc/ssl/ca-bundle.pem",                            // SUSE/OpenSUSE
    "/etc/ssl/cert.pem",                                 // Alpine / macOS
    "/etc/pki/tls/cert.pem",                             // Fedora
    "/usr/local/etc/openssl@1.1/cert.pem",               // macOS Homebrew Intel
    "/opt/homebrew/etc/openssl@1.1/cert.pem",            // macOS Homebrew ARM
    "/usr/local/share/ca-certificates",                  // padding (unused sentinel)
];

/// Resolve the first existing CA bundle path from the distro/macOS candidate
/// list, mirroring step 3 of `_ensure_ssl_certs`.  Returns `None` if none exist
/// (the caller would then leave `SSL_CERT_FILE` unset).
pub fn detect_ssl_cert_file() -> Option<PathBuf> {
    if std::env::var("SSL_CERT_FILE").is_ok() {
        return None; // user already configured it
    }
    for candidate in &SSL_CERT_CANDIDATE_PATHS[..8] {
        let p = Path::new(candidate);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Home-target env var naming
// ---------------------------------------------------------------------------

/// Default home-target env var for a platform: `{PLATFORM}_HOME_CHANNEL`.
///
/// The Python version consults `cron.scheduler._HOME_TARGET_ENV_VARS` for
/// per-platform overrides; that table is taken as a parameter here so callers
/// can supply the ported mapping without a hard dependency.
pub fn home_target_env_var(platform_name: &str, overrides: &BTreeMap<String, String>) -> String {
    let key = platform_name.to_lowercase();
    if let Some(v) = overrides.get(&key) {
        return v.clone();
    }
    format!("{}_HOME_CHANNEL", platform_name.to_uppercase())
}

/// The optional thread/topic env var for a platform home target.
pub fn home_thread_env_var(platform_name: &str, overrides: &BTreeMap<String, String>) -> String {
    format!("{}_THREAD_ID", home_target_env_var(platform_name, overrides))
}

/// Return `true` when a `/restart` completion marker is waiting.
pub fn restart_notification_pending(hermes_home: &Path) -> bool {
    hermes_home.join(".restart_notify.json").exists()
}

// ---------------------------------------------------------------------------
// config.yaml → env-var bridging
// ---------------------------------------------------------------------------

/// Map of terminal config keys → `TERMINAL_*` env vars (config bridge above).
pub const TERMINAL_ENV_MAP: [(&str, &str); 24] = [
    ("backend", "TERMINAL_ENV"),
    ("cwd", "TERMINAL_CWD"),
    ("timeout", "TERMINAL_TIMEOUT"),
    ("lifetime_seconds", "TERMINAL_LIFETIME_SECONDS"),
    ("docker_image", "TERMINAL_DOCKER_IMAGE"),
    ("docker_forward_env", "TERMINAL_DOCKER_FORWARD_ENV"),
    ("singularity_image", "TERMINAL_SINGULARITY_IMAGE"),
    ("modal_image", "TERMINAL_MODAL_IMAGE"),
    ("daytona_image", "TERMINAL_DAYTONA_IMAGE"),
    ("vercel_runtime", "TERMINAL_VERCEL_RUNTIME"),
    ("ssh_host", "TERMINAL_SSH_HOST"),
    ("ssh_user", "TERMINAL_SSH_USER"),
    ("ssh_port", "TERMINAL_SSH_PORT"),
    ("ssh_key", "TERMINAL_SSH_KEY"),
    ("container_cpu", "TERMINAL_CONTAINER_CPU"),
    ("container_memory", "TERMINAL_CONTAINER_MEMORY"),
    ("container_disk", "TERMINAL_CONTAINER_DISK"),
    ("container_persistent", "TERMINAL_CONTAINER_PERSISTENT"),
    ("docker_volumes", "TERMINAL_DOCKER_VOLUMES"),
    (
        "docker_mount_cwd_to_workspace",
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
    ),
    ("docker_run_as_host_user", "TERMINAL_DOCKER_RUN_AS_HOST_USER"),
    ("sandbox_dir", "TERMINAL_SANDBOX_DIR"),
    ("persistent_shell", "TERMINAL_PERSISTENT_SHELL"),
    ("__unused__", "__UNUSED__"),
];

/// One `(env_var_name, value)` pair produced by the config→env bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvAssignment {
    pub name: String,
    pub value: String,
}

/// Compute the full set of env-var assignments the config bridge would make for
/// a parsed `config.yaml`, without actually mutating the process environment.
///
/// `existing_env` is the current environment snapshot (used to honour the
/// "top-level simple values: fallback only — don't override .env" rule).  The
/// returned vec is in deterministic, source-faithful order.
pub fn bridge_config_to_env(
    cfg: &Value,
    existing_env: &BTreeMap<String, String>,
) -> Vec<EnvAssignment> {
    let mut out = Vec::new();
    let obj = match cfg.as_object() {
        Some(o) => o,
        None => return out,
    };

    // Top-level simple values (fallback only — don't override .env / existing).
    for (key, val) in obj {
        let simple = match val {
            Value::String(_) | Value::Number(_) | Value::Bool(_) => true,
            _ => false,
        };
        if simple && !existing_env.contains_key(key) {
            out.push(EnvAssignment {
                name: key.clone(),
                value: simple_value_to_string(val),
            });
        }
    }

    // Terminal config — bridge to TERMINAL_* (config.yaml wins over .env).
    if let Some(term) = obj.get("terminal").and_then(Value::as_object) {
        for (cfg_key, env_var) in TERMINAL_ENV_MAP.iter() {
            if *cfg_key == "__unused__" {
                continue;
            }
            if let Some(val) = term.get(*cfg_key) {
                // Skip cwd placeholders; expand tilde for cwd strings.
                if *cfg_key == "cwd" {
                    if let Some(s) = val.as_str() {
                        if matches!(s, "." | "auto" | "cwd") {
                            continue;
                        }
                        out.push(EnvAssignment {
                            name: env_var.to_string(),
                            value: expand_user(s),
                        });
                        continue;
                    }
                }
                let value = if val.is_array() {
                    serde_json::to_string(val).unwrap_or_default()
                } else {
                    simple_value_to_string(val)
                };
                out.push(EnvAssignment {
                    name: env_var.to_string(),
                    value,
                });
            }
        }
    }

    // Auxiliary task overrides.
    if let Some(aux) = obj.get("auxiliary").and_then(Value::as_object) {
        let task_env: [(&str, [(&str, &str); 4]); 3] = [
            (
                "vision",
                [
                    ("provider", "AUXILIARY_VISION_PROVIDER"),
                    ("model", "AUXILIARY_VISION_MODEL"),
                    ("base_url", "AUXILIARY_VISION_BASE_URL"),
                    ("api_key", "AUXILIARY_VISION_API_KEY"),
                ],
            ),
            (
                "web_extract",
                [
                    ("provider", "AUXILIARY_WEB_EXTRACT_PROVIDER"),
                    ("model", "AUXILIARY_WEB_EXTRACT_MODEL"),
                    ("base_url", "AUXILIARY_WEB_EXTRACT_BASE_URL"),
                    ("api_key", "AUXILIARY_WEB_EXTRACT_API_KEY"),
                ],
            ),
            (
                "approval",
                [
                    ("provider", "AUXILIARY_APPROVAL_PROVIDER"),
                    ("model", "AUXILIARY_APPROVAL_MODEL"),
                    ("base_url", "AUXILIARY_APPROVAL_BASE_URL"),
                    ("api_key", "AUXILIARY_APPROVAL_API_KEY"),
                ],
            ),
        ];
        for (task_key, env_map) in task_env.iter() {
            let task_cfg = match aux.get(*task_key).and_then(Value::as_object) {
                Some(t) => t,
                None => continue,
            };
            let prov = str_field(task_cfg, "provider");
            let model = str_field(task_cfg, "model");
            let base_url = str_field(task_cfg, "base_url");
            let api_key = str_field(task_cfg, "api_key");
            if !prov.is_empty() && prov != "auto" {
                out.push(EnvAssignment {
                    name: env_map[0].1.to_string(),
                    value: prov,
                });
            }
            if !model.is_empty() {
                out.push(EnvAssignment {
                    name: env_map[1].1.to_string(),
                    value: model,
                });
            }
            if !base_url.is_empty() {
                out.push(EnvAssignment {
                    name: env_map[2].1.to_string(),
                    value: base_url,
                });
            }
            if !api_key.is_empty() {
                out.push(EnvAssignment {
                    name: env_map[3].1.to_string(),
                    value: api_key,
                });
            }
        }
    }

    // Agent config — unconditionally wins over .env.
    if let Some(agent) = obj.get("agent").and_then(Value::as_object) {
        let agent_map = [
            ("max_turns", "HERMES_MAX_ITERATIONS"),
            ("gateway_timeout", "HERMES_AGENT_TIMEOUT"),
            ("gateway_timeout_warning", "HERMES_AGENT_TIMEOUT_WARNING"),
            ("gateway_notify_interval", "HERMES_AGENT_NOTIFY_INTERVAL"),
            ("restart_drain_timeout", "HERMES_RESTART_DRAIN_TIMEOUT"),
            (
                "gateway_auto_continue_freshness",
                "HERMES_AUTO_CONTINUE_FRESHNESS",
            ),
        ];
        for (cfg_key, env_var) in agent_map.iter() {
            if let Some(val) = agent.get(*cfg_key) {
                out.push(EnvAssignment {
                    name: env_var.to_string(),
                    value: simple_value_to_string(val),
                });
            }
        }
    }

    // Display config.
    if let Some(display) = obj.get("display").and_then(Value::as_object) {
        if let Some(v) = display.get("busy_input_mode") {
            out.push(EnvAssignment {
                name: "HERMES_GATEWAY_BUSY_INPUT_MODE".to_string(),
                value: simple_value_to_string(v),
            });
        }
        if let Some(v) = display.get("busy_ack_enabled") {
            out.push(EnvAssignment {
                name: "HERMES_GATEWAY_BUSY_ACK_ENABLED".to_string(),
                value: simple_value_to_string(v),
            });
        }
    }

    // Timezone.
    if let Some(tz) = obj.get("timezone").and_then(Value::as_str) {
        if !tz.trim().is_empty() {
            out.push(EnvAssignment {
                name: "HERMES_TIMEZONE".to_string(),
                value: tz.trim().to_string(),
            });
        }
    }

    // Security: redact_secrets (lowercased).
    if let Some(sec) = obj.get("security").and_then(Value::as_object) {
        if let Some(redact) = sec.get("redact_secrets") {
            if !redact.is_null() {
                out.push(EnvAssignment {
                    name: "HERMES_REDACT_SECRETS".to_string(),
                    value: simple_value_to_string(redact).to_lowercase(),
                });
            }
        }
    }

    out
}

/// `str(value)` for the Python-`bool`/`int`/`float`/`str` cases.  Booleans
/// render as Python's `True`/`False`; integers drop a `.0`.
fn simple_value_to_string(val: &Value) -> String {
    match val {
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                n.to_string()
            }
        }
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn str_field(obj: &serde_json::Map<String, Value>, key: &str) -> String {
    match obj.get(key) {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => simple_value_to_string(other).trim().to_string(),
    }
}

/// Expand a leading `~` / `~user` like `os.path.expanduser`.
pub fn expand_user(path: &str) -> String {
    if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().into_owned();
        }
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

/// Resolve the configured terminal cwd (`TERMINAL_CWD`), applying the fallback
/// logic from the bottom of the config bridge: placeholders/unset fall back to
/// `MESSAGING_CWD` then the home dir.
pub fn resolve_terminal_cwd(env: &BTreeMap<String, String>) -> String {
    let configured = env.get("TERMINAL_CWD").map(String::as_str).unwrap_or("");
    if configured.is_empty() || matches!(configured, "." | "auto" | "cwd") {
        if let Some(m) = env.get("MESSAGING_CWD") {
            if !m.is_empty() {
                return m.clone();
            }
        }
        return dirs::home_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
    }
    configured.to_string()
}

// ---------------------------------------------------------------------------
// Session-key parsing + platform/config-key mapping
// ---------------------------------------------------------------------------

/// Parsed components of an `agent:main:{platform}:{chat_type}:{chat_id}[...]`
/// session key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSessionKey {
    pub platform: String,
    pub chat_type: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
}

/// Parse a session key into its component parts, or `None` if it doesn't match.
///
/// The 6th element is only surfaced as `thread_id` for the `dm`/`thread` chat
/// types where it is unambiguous.
pub fn parse_session_key(session_key: &str) -> Option<ParsedSessionKey> {
    let parts: Vec<&str> = session_key.split(':').collect();
    if parts.len() >= 5 && parts[0] == "agent" && parts[1] == "main" {
        let thread_id = if parts.len() > 5 && (parts[3] == "dm" || parts[3] == "thread") {
            Some(parts[5].to_string())
        } else {
            None
        };
        return Some(ParsedSessionKey {
            platform: parts[2].to_string(),
            chat_type: parts[3].to_string(),
            chat_id: parts[4].to_string(),
            thread_id,
        });
    }
    None
}

/// Map a platform value to its config.yaml key: `LOCAL` → `"cli"`, else itself.
pub fn platform_config_key(platform_value: &str) -> String {
    if platform_value == "local" {
        "cli".to_string()
    } else {
        platform_value.to_string()
    }
}

/// Platform-namespaced key for voice-mode state: `{platform}:{chat_id}`.
pub fn voice_key(platform_value: &str, chat_id: &str) -> String {
    format!("{}:{}", platform_value, chat_id)
}

// ---------------------------------------------------------------------------
// Gateway model resolution
// ---------------------------------------------------------------------------

/// Read the gateway model from a parsed config.yaml value.
///
/// Mirrors `_resolve_gateway_model`: accepts a bare string model, or a dict
/// with `default`/`model` keys; returns `""` otherwise.
pub fn resolve_gateway_model(cfg: &Value) -> String {
    let model_cfg = match cfg.as_object().and_then(|o| o.get("model")) {
        Some(v) => v,
        None => return String::new(),
    };
    match model_cfg {
        Value::String(s) => s.clone(),
        Value::Object(m) => m
            .get("default")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| m.get("model").and_then(Value::as_str).filter(|s| !s.is_empty()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Media placeholder + empty-response normalization
// ---------------------------------------------------------------------------

/// Build a text placeholder for media-only events so they aren't dropped.
///
/// `is_photo` mirrors `event.message_type == MessageType.PHOTO`.
pub fn build_media_placeholder(
    media_urls: &[String],
    media_types: &[String],
    is_photo: bool,
) -> String {
    let mut parts = Vec::new();
    for (i, url) in media_urls.iter().enumerate() {
        let mtype = media_types.get(i).map(String::as_str).unwrap_or("");
        if mtype.starts_with("image/") || is_photo {
            parts.push(format!("[User sent an image: {}]", url));
        } else if mtype.starts_with("audio/") {
            parts.push(format!("[User sent audio: {}]", url));
        } else {
            parts.push(format!("[User sent a file: {}]", url));
        }
    }
    parts.join("\n")
}

/// Return `true` when an interrupt message is internal control flow.
///
/// Normalizes whitespace then case-folds before comparing to the known
/// control-interrupt reasons.
pub fn is_control_interrupt_message(message: Option<&str>) -> bool {
    let msg = match message {
        Some(m) if !m.is_empty() => m,
        _ => return false,
    };
    let normalized = msg.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    [
        INTERRUPT_REASON_STOP,
        INTERRUPT_REASON_RESET,
        INTERRUPT_REASON_TIMEOUT,
        INTERRUPT_REASON_SSE_DISCONNECT,
        INTERRUPT_REASON_GATEWAY_SHUTDOWN,
        INTERRUPT_REASON_GATEWAY_RESTART,
    ]
    .iter()
    .any(|r| r.to_lowercase() == normalized)
}

/// Normalize empty/None agent responses into user-facing messages.
///
/// Faithful port of `_normalize_empty_agent_response`.  `agent_result` is the
/// agent's result dict; `response` the (possibly empty) text it produced.
pub fn normalize_empty_agent_response(agent_result: &Value, response: &str, history_len: usize) -> String {
    if !response.is_empty() {
        return response.to_string();
    }
    let obj = agent_result.as_object();
    let get = |k: &str| obj.and_then(|o| o.get(k));

    let failed = get("failed").map(truthy).unwrap_or(false);
    if failed {
        let error_detail = get("error")
            .map(value_to_str)
            .unwrap_or_else(|| "unknown error".to_string());
        let error_str = error_detail.to_lowercase();
        let is_context_failure = ["context", "token", "too large", "too long", "exceed", "payload"]
            .iter()
            .any(|p| error_str.contains(p))
            || (error_str.contains("400") && history_len > 50);
        if is_context_failure {
            return "⚠️ Session too large for the model's context window.\n\
                Use /compact to compress the conversation, or \
                /reset to start fresh."
                .to_string();
        }
        let truncated: String = error_detail.chars().take(300).collect();
        return format!(
            "The request failed: {}\nTry again or use /reset to start a fresh session.",
            truncated
        );
    }

    let api_calls = get("api_calls")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .unwrap_or(0);
    let interrupted = get("interrupted").map(truthy).unwrap_or(false);
    if api_calls > 0 && !interrupted {
        if get("partial").map(truthy).unwrap_or(false) {
            let err = get("error")
                .map(value_to_str)
                .unwrap_or_else(|| "processing incomplete".to_string());
            let truncated: String = err.chars().take(200).collect();
            return format!("⚠️ Processing stopped: {}. Try again.", truncated);
        }
        return "⚠️ Processing completed but no response was generated. \
            This may be a transient error — try sending your message again."
            .to_string();
    }

    response.to_string()
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => simple_value_to_string(other),
    }
}

// ---------------------------------------------------------------------------
// Process-watcher notification formatting
// ---------------------------------------------------------------------------

/// Format a watch-pattern event from the completion queue into an
/// `[IMPORTANT: ...]` message, or `None` for unhandled types.
pub fn format_gateway_process_notification(evt: &Value) -> Option<String> {
    let obj = evt.as_object()?;
    let get_str = |k: &str, d: &str| obj.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
    let evt_type = get_str("type", "completion");
    let sid = get_str("session_id", "unknown");
    let cmd = get_str("command", "unknown");

    if evt_type == "watch_disabled" {
        let message = get_str("message", "");
        return Some(format!("[IMPORTANT: {}]", message));
    }

    if evt_type == "watch_match" {
        let pat = get_str("pattern", "?");
        let out = get_str("output", "");
        let sup = obj
            .get("suppressed")
            .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
            .unwrap_or(0);
        let mut text = format!(
            "[IMPORTANT: Background process {} matched watch pattern \"{}\".\nCommand: {}\nMatched output:\n{}",
            sid, pat, cmd, out
        );
        if sup != 0 {
            text.push_str(&format!(
                "\n({} earlier matches were suppressed by rate limit)",
                sup
            ));
        }
        text.push(']');
        return Some(text);
    }

    None
}

// ---------------------------------------------------------------------------
// SKILL.md frontmatter → command-slug derivation
// ---------------------------------------------------------------------------

/// Derive the `/command` slug and declared frontmatter name from SKILL.md
/// content.  Matches `agent.skill_commands.scan_skill_commands` normalization.
///
/// Returns `(Some(slug), Some(name))`, `(None, Some(name))` when the slug folds
/// to empty, or `(None, None)` when there's no usable `name:`.
pub fn skill_slug_from_frontmatter(content: &str) -> (Option<String>, Option<String>) {
    if !content.starts_with("---") {
        return (None, None);
    }
    let end = match content[3..].find("\n---") {
        Some(idx) => idx + 3,
        None => return (None, None),
    };
    let mut declared_name: Option<String> = None;
    for line in content[3..end].lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("name:") {
            let mut raw = rest.trim().to_string();
            // Strip YAML quote wrappers if present.
            let bytes: Vec<char> = raw.chars().collect();
            if bytes.len() >= 2
                && bytes[0] == bytes[bytes.len() - 1]
                && (bytes[0] == '"' || bytes[0] == '\'')
            {
                raw = bytes[1..bytes.len() - 1].iter().collect();
            }
            declared_name = Some(raw.trim().to_string());
            break;
        }
    }
    let declared = match declared_name {
        Some(n) if !n.is_empty() => n,
        _ => return (None, None),
    };

    let mut slug: String = declared.to_lowercase().replace(' ', "-").replace('_', "-");
    // _SKILL_INVALID_CHARS: keep only [a-z0-9-]
    slug = slug.chars().filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-').collect();
    // _SKILL_MULTI_HYPHEN: collapse runs of '-' then strip leading/trailing.
    slug = collapse_hyphens(&slug);
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        return (None, Some(declared));
    }
    (Some(slug), Some(declared))
}

fn collapse_hyphens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_hyphen = false;
    for c in s.chars() {
        if c == '-' {
            if !prev_hyphen {
                out.push(c);
            }
            prev_hyphen = true;
        } else {
            out.push(c);
            prev_hyphen = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Config loaders (reasoning / service-tier / busy-input / background-notify)
// ---------------------------------------------------------------------------

/// Normalize a raw `agent.service_tier` value: `"fast"/"priority"/"on"` →
/// `Some("priority")`, the off-set → `None`, unknown → `None` (logs upstream).
pub fn normalize_service_tier(raw: &str) -> Option<String> {
    let value = raw.trim().to_lowercase();
    if value.is_empty() || matches!(value.as_str(), "normal" | "default" | "standard" | "off" | "none") {
        return None;
    }
    if matches!(value.as_str(), "fast" | "priority" | "on") {
        return Some("priority".to_string());
    }
    None
}

/// Resolve the gateway busy-input mode from an env var + config value, mirroring
/// `_load_busy_input_mode`.  Env takes precedence; result is one of
/// `interrupt` (default) / `queue` / `steer`.
pub fn resolve_busy_input_mode(env_value: Option<&str>, config_value: Option<&str>) -> String {
    let mut mode = env_value.unwrap_or("").trim().to_lowercase();
    if mode.is_empty() {
        mode = config_value.unwrap_or("").trim().to_lowercase();
    }
    match mode.as_str() {
        "queue" => "queue".to_string(),
        "steer" => "steer".to_string(),
        _ => "interrupt".to_string(),
    }
}

/// Normalize a background-notifications mode value, mirroring
/// `_load_background_notifications_mode`'s validation.  `config_raw` is the
/// `display.background_process_notifications` value (a JSON value so `False`
/// maps to `"off"`).  Result is one of `all`/`result`/`error`/`off`.
pub fn resolve_background_notifications_mode(env_value: Option<&str>, config_raw: Option<&Value>) -> String {
    let mut mode = env_value.unwrap_or("").to_string();
    if mode.is_empty() {
        if let Some(raw) = config_raw {
            match raw {
                Value::Bool(false) => mode = "off".to_string(),
                Value::Null => {}
                Value::String(s) if s.is_empty() => {}
                other => mode = value_to_str(other),
            }
        }
    }
    let mode = mode.trim().to_lowercase();
    let mode = if mode.is_empty() { "all".to_string() } else { mode };
    if matches!(mode.as_str(), "all" | "result" | "error" | "off") {
        mode
    } else {
        "all".to_string()
    }
}

/// Resolve the platform connect timeout (seconds) from a config value, falling
/// back to the default.  Non-positive / unparseable values yield the default.
pub fn platform_connect_timeout_secs(config_value: Option<&Value>) -> f64 {
    let parsed = config_value.and_then(|v| match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    });
    match parsed {
        Some(v) if v > 0.0 => v,
        _ => PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT,
    }
}

// ---------------------------------------------------------------------------
// /reasoning command argument parsing
// ---------------------------------------------------------------------------

/// Parse `/reasoning` args into `(value, persist_global)`.
///
/// `--global` may appear in any position; the em-dash `—` is normalized to
/// `--`.  Remaining tokens are space-joined, trimmed, and lowercased.
pub fn parse_reasoning_command_args(raw_args: &str) -> (String, bool) {
    let text = raw_args.trim().replace('—', "--");
    if text.is_empty() {
        return (String::new(), false);
    }
    let tokens = shlex_split(&text).unwrap_or_else(|| text.split_whitespace().map(String::from).collect());
    let mut persist_global = false;
    let mut value_tokens: Vec<String> = Vec::new();
    for token in tokens {
        if token == "--global" {
            persist_global = true;
        } else {
            value_tokens.push(token);
        }
    }
    (value_tokens.join(" ").trim().to_lowercase(), persist_global)
}

/// Minimal POSIX-ish shell tokenizer mirroring `shlex.split` for the subset of
/// inputs the reasoning command parser encounters.  Returns `None` on an
/// unterminated quote (caller falls back to whitespace split).
fn shlex_split(s: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_token = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_token = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    cur.push(q);
                }
            }
            '"' => {
                in_token = true;
                let mut closed = false;
                while let Some(q) = chars.next() {
                    if q == '"' {
                        closed = true;
                        break;
                    }
                    if q == '\\' {
                        if let Some(&n) = chars.peek() {
                            if matches!(n, '"' | '\\' | '$' | '`') {
                                cur.push(chars.next().unwrap());
                                continue;
                            }
                        }
                    }
                    cur.push(q);
                }
                if !closed {
                    return None;
                }
            }
            '\\' => {
                in_token = true;
                if let Some(n) = chars.next() {
                    cur.push(n);
                } else {
                    return None;
                }
            }
            c if c.is_whitespace() => {
                if in_token {
                    tokens.push(std::mem::take(&mut cur));
                    in_token = false;
                }
            }
            other => {
                in_token = true;
                cur.push(other);
            }
        }
    }
    if in_token {
        tokens.push(cur);
    }
    Some(tokens)
}

// ---------------------------------------------------------------------------
// Voice-mode persistence
// ---------------------------------------------------------------------------

/// Parse the on-disk voice-mode JSON into a validated `{key: mode}` map.
///
/// Mirrors `_load_voice_modes`: only `off`/`voice_only`/`all` modes are kept,
/// and legacy unprefixed (no `:`) keys are skipped.
pub fn parse_voice_modes(raw_json: &str) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let data: Value = match serde_json::from_str(raw_json) {
        Ok(v) => v,
        Err(_) => return result,
    };
    let obj = match data.as_object() {
        Some(o) => o,
        None => return result,
    };
    for (chat_id, mode_v) in obj {
        let mode = match mode_v.as_str() {
            Some(m) => m,
            None => continue,
        };
        if !matches!(mode, "off" | "voice_only" | "all") {
            continue;
        }
        if !chat_id.contains(':') {
            continue; // legacy unprefixed key — skip
        }
        result.insert(chat_id.clone(), mode.to_string());
    }
    result
}

/// Serialize voice modes back to disk (pretty-printed, matching json.dumps
/// indent=2).
pub fn serialize_voice_modes(modes: &BTreeMap<String, String>) -> String {
    serde_json::to_string_pretty(modes).unwrap_or_else(|_| "{}".to_string())
}

/// Compute the `(disabled_chats, enabled_chats)` sets a platform adapter should
/// hold, given the global voice-mode map and the platform prefix.
///
/// Mirrors the set math in `_sync_voice_mode_state_to_adapter`: chat ids are the
/// suffix after `{platform}:`; `off` → disabled, `voice_only`/`all` → enabled.
pub fn voice_mode_adapter_sets(
    voice_mode: &BTreeMap<String, String>,
    platform_value: &str,
) -> (Vec<String>, Vec<String>) {
    let prefix = format!("{}:", platform_value);
    let mut disabled = Vec::new();
    let mut enabled = Vec::new();
    for (key, mode) in voice_mode {
        if let Some(chat) = key.strip_prefix(&prefix) {
            match mode.as_str() {
                "off" => disabled.push(chat.to_string()),
                "voice_only" | "all" => enabled.push(chat.to_string()),
                _ => {}
            }
        }
    }
    (disabled, enabled)
}

// ---------------------------------------------------------------------------
// Telegram topic classification + canned messages
// ---------------------------------------------------------------------------

/// True for the main Telegram DM (or General topic) when topic mode is enabled.
///
/// `topic_mode_enabled` is the SessionDB lookup result (already resolved).
pub fn is_telegram_topic_root_lobby(
    platform_value: &str,
    chat_type: &str,
    thread_id: Option<&str>,
    topic_mode_enabled: bool,
) -> bool {
    if platform_value != "telegram" || chat_type != "dm" || !topic_mode_enabled {
        return false;
    }
    let tid = thread_id.unwrap_or("");
    TELEGRAM_GENERAL_TOPIC_IDS.contains(&tid)
}

/// True for a user-created Telegram private-chat topic lane.
pub fn is_telegram_topic_lane(
    platform_value: &str,
    chat_type: &str,
    thread_id: Option<&str>,
    topic_mode_enabled: bool,
) -> bool {
    if platform_value != "telegram" || chat_type != "dm" || !topic_mode_enabled {
        return false;
    }
    let tid = thread_id.unwrap_or("");
    !tid.is_empty() && !TELEGRAM_GENERAL_TOPIC_IDS.contains(&tid)
}

pub fn telegram_topic_root_lobby_message() -> &'static str {
    "This main chat is reserved for system commands.\n\n\
To start a new Hermes chat, open the All Messages topic at the top \
of this bot interface and send any message there. Telegram will \
create a new topic for that message; each topic works as an \
independent Hermes session."
}

pub fn telegram_topic_root_new_message() -> &'static str {
    "To start a new parallel Hermes chat, open the All Messages topic \
at the top of this bot interface and send any message there. \
Telegram will create a new topic for it.\n\n\
Each topic is an independent Hermes session. Use /new inside an \
existing topic only if you want to replace that topic's current session."
}

/// Header shown when a new session starts inside a Telegram topic lane.
/// Returns `None` when the source is not a lane.
pub fn telegram_topic_new_header(
    platform_value: &str,
    chat_type: &str,
    thread_id: Option<&str>,
    topic_mode_enabled: bool,
) -> Option<&'static str> {
    if !is_telegram_topic_lane(platform_value, chat_type, thread_id, topic_mode_enabled) {
        return None;
    }
    Some(
        "Started a new Hermes session in this topic.\n\n\
Tip: for parallel work, open All Messages and send a message there \
to create a separate topic instead of using /new here. /new replaces \
the session attached to the current topic.",
    )
}

/// Rewrite slash-command mentions to Telegram-valid command names.
///
/// Only applies on the `telegram` platform; other platforms get the text
/// unchanged.  `sanitize` maps a raw command name to a Telegram-valid one
/// (lowercase letters/digits/underscores); an empty result leaves the original
/// `/name` mention intact.
pub fn telegramize_command_mentions<F>(text: &str, platform_value: &str, sanitize: F) -> String
where
    F: Fn(&str) -> String,
{
    if platform_value != "telegram" {
        return text.to_string();
    }
    let re = telegram_command_mention_re();
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for cap in re.captures_iter(text) {
        let whole = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str();
        out.push_str(&text[last..whole.start()]);
        let sanitized = sanitize(name);
        if sanitized.is_empty() {
            out.push_str(whole.as_str());
        } else {
            out.push('/');
            out.push_str(&sanitized);
        }
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

fn telegram_command_mention_re() -> regex::Regex {
    // Python: (?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)
    // Rust regex lacks lookbehind; emulate the negative-lookbehind on the
    // preceding char in telegramize_command_mentions via manual filtering.
    // We approximate by matching the slash + name then validating the prior
    // char at call time.  To keep behavior faithful we re-implement here.
    regex::Regex::new(r"/([A-Za-z0-9][A-Za-z0-9_-]*)").unwrap()
}

// ---------------------------------------------------------------------------
// Agent-config cache signature + cache-busting extraction
// ---------------------------------------------------------------------------

/// Extract cache-busting config values keyed by `"section.key"`.
///
/// Missing keys / non-dict sections yield JSON null entries (so "absent" vs
/// "present-and-null" still differ in the signature).  The live tool-registry
/// generation is taken as a parameter (the Python reads `registry._generation`).
pub fn extract_cache_busting_config(user_config: &Value, registry_generation: Option<i64>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    let cfg = user_config.as_object();
    for (section, key) in CACHE_BUSTING_CONFIG_KEYS.iter() {
        let val = cfg
            .and_then(|c| c.get(*section))
            .and_then(Value::as_object)
            .and_then(|s| s.get(*key))
            .cloned()
            .unwrap_or(Value::Null);
        out.insert(format!("{}.{}", section, key), val);
    }
    out.insert(
        "tools.registry_generation".to_string(),
        registry_generation.map(Value::from).unwrap_or(Value::Null),
    );
    out
}

/// Compute a stable 16-hex-char signature from agent config values.
///
/// Faithful port of `_agent_config_signature`: the api_key is SHA-256
/// fingerprinted; cache_keys are sorted; the JSON blob is built with sorted
/// keys and a `default=str` coercion, then SHA-256'd and truncated.
pub fn agent_config_signature(
    model: &str,
    runtime: &Value,
    enabled_toolsets: &[String],
    ephemeral_prompt: &str,
    cache_keys: Option<&BTreeMap<String, Value>>,
) -> String {
    use sha2::{Digest, Sha256};

    let rget = |k: &str| runtime.as_object().and_then(|o| o.get(k));
    let api_key = rget("api_key")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let api_key_fingerprint = if api_key.is_empty() {
        String::new()
    } else {
        let mut h = Sha256::new();
        h.update(api_key.as_bytes());
        to_hex(&h.finalize())
    };

    let mut toolsets: Vec<String> = enabled_toolsets.to_vec();
    toolsets.sort();
    let toolsets_json: Vec<Value> = toolsets.into_iter().map(Value::String).collect();

    // cache_keys sorted as a list of [key, value] pairs (sorted by key).
    let cache_pairs: Vec<Value> = match cache_keys {
        Some(m) => m
            .iter()
            .map(|(k, v)| Value::Array(vec![Value::String(k.clone()), v.clone()]))
            .collect(),
        None => Vec::new(),
    };

    let blob_value = Value::Array(vec![
        Value::String(model.to_string()),
        Value::String(api_key_fingerprint),
        Value::String(rget("base_url").and_then(Value::as_str).unwrap_or("").to_string()),
        Value::String(rget("provider").and_then(Value::as_str).unwrap_or("").to_string()),
        Value::String(rget("api_mode").and_then(Value::as_str).unwrap_or("").to_string()),
        Value::Array(toolsets_json),
        Value::String(ephemeral_prompt.to_string()),
        Value::Array(cache_pairs),
    ]);

    // json.dumps(..., sort_keys=True) — for arrays order is preserved; our
    // values contain no nested dicts so sort_keys is a no-op here.
    let blob = serde_json::to_string(&blob_value).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(blob.as_bytes());
    let digest = to_hex(&h.finalize());
    digest[..16].to_string()
}

/// Lowercase hex-encode a byte slice (avoids a new `hex` crate dependency).
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Cron-ticker cadence math
// ---------------------------------------------------------------------------

/// Cadence (in ticks) for each periodic maintenance task in the cron ticker.
pub const CRON_IMAGE_CACHE_EVERY: u64 = 60; // once per hour @60s
pub const CRON_CHANNEL_DIR_EVERY: u64 = 5; // every 5 minutes
pub const CRON_PASTE_SWEEP_EVERY: u64 = 60; // once per hour
pub const CRON_CURATOR_EVERY: u64 = 60; // poll hourly

/// Which periodic maintenance tasks fire on a given (1-based) tick count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CronTickActions {
    pub channel_directory: bool,
    pub image_cache: bool,
    pub paste_sweep: bool,
    pub curator: bool,
}

/// Compute which maintenance tasks run on `tick_count` (post-increment value).
/// `has_adapters` gates the channel-directory refresh, matching the Python
/// `tick_count % CHANNEL_DIR_EVERY == 0 and adapters` guard.
pub fn cron_tick_actions(tick_count: u64, has_adapters: bool) -> CronTickActions {
    CronTickActions {
        channel_directory: tick_count % CRON_CHANNEL_DIR_EVERY == 0 && has_adapters,
        image_cache: tick_count % CRON_IMAGE_CACHE_EVERY == 0,
        paste_sweep: tick_count % CRON_PASTE_SWEEP_EVERY == 0,
        curator: tick_count % CRON_CURATOR_EVERY == 0,
    }
}

// ---------------------------------------------------------------------------
// /model session override application
// ---------------------------------------------------------------------------

/// Apply a `/model` session override onto `(model, runtime)`.
///
/// Faithful port of `_apply_session_model_override`: the override's `model`
/// wins when present; `provider`/`api_key`/`base_url`/`api_mode` override the
/// runtime only when non-null.  Returns the updated `(model, runtime)`.
pub fn apply_session_model_override(
    model: &str,
    runtime: &Value,
    override_entry: Option<&Value>,
) -> (String, Value) {
    let mut out_runtime = runtime.clone();
    let ov = match override_entry.and_then(Value::as_object) {
        Some(o) => o,
        None => return (model.to_string(), out_runtime),
    };
    let out_model = ov
        .get("model")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| model.to_string());
    let map = out_runtime.as_object_mut();
    if let Some(m) = map {
        for key in ["provider", "api_key", "base_url", "api_mode"] {
            if let Some(val) = ov.get(key) {
                if !val.is_null() {
                    m.insert(key.to_string(), val.clone());
                }
            }
        }
    }
    (out_model, out_runtime)
}

/// True when `agent_model` matches an active `/model` session override.
pub fn is_intentional_model_switch(override_entry: Option<&Value>, agent_model: &str) -> bool {
    override_entry
        .and_then(Value::as_object)
        .and_then(|o| o.get("model"))
        .and_then(Value::as_str)
        .map(|m| m == agent_model)
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_coerce_timestamp_seconds_and_millis() {
        assert_eq!(coerce_gateway_timestamp(&json!(1700000000)), Some(1700000000.0));
        // Milliseconds magnitude → divided by 1000.
        assert_eq!(coerce_gateway_timestamp(&json!(1700000000000i64)), Some(1700000000.0));
        // Bool is skipped (subclass of int).
        assert_eq!(coerce_gateway_timestamp(&json!(true)), None);
        assert_eq!(coerce_gateway_timestamp(&json!(null)), None);
    }

    #[test]
    fn test_coerce_timestamp_strings() {
        assert_eq!(coerce_gateway_timestamp_str("1700000000"), Some(1700000000.0));
        assert_eq!(coerce_gateway_timestamp_str("  "), None);
        let v = coerce_gateway_timestamp_str("2023-11-14T22:13:20Z").unwrap();
        assert!((v - 1699999_e3).abs() > 0.0); // some epoch
        assert!(coerce_gateway_timestamp_str("not a date").is_none());
    }

    #[test]
    fn test_freshness_gate() {
        // Disabled gate → always fresh.
        assert!(is_fresh_gateway_interruption(&json!(0), Some(1e12), Some(0.0)));
        // Unknown timestamp → fresh.
        assert!(is_fresh_gateway_interruption(&json!(null), Some(100.0), Some(60.0)));
        // Within window.
        assert!(is_fresh_gateway_interruption(&json!(90.0), Some(100.0), Some(60.0)));
        // Outside window.
        assert!(!is_fresh_gateway_interruption(&json!(10.0), Some(100.0), Some(60.0)));
    }

    #[test]
    fn test_last_transcript_timestamp() {
        let history = vec![
            json!({"role": "user", "timestamp": 1.0}),
            json!({"role": "session_meta", "timestamp": 2.0}),
            json!({"role": "system", "timestamp": 3.0}),
        ];
        // Skips meta/system rows, finds the user row (last usable).
        assert_eq!(last_transcript_timestamp(&history), Some(json!(1.0)));
        // No usable row.
        assert_eq!(last_transcript_timestamp(&[]), None);
        // Legacy row without timestamp → None.
        let legacy = vec![json!({"role": "user"})];
        assert_eq!(last_transcript_timestamp(&legacy), None);
    }

    #[test]
    fn test_parse_session_key() {
        let p = parse_session_key("agent:main:telegram:dm:123:456").unwrap();
        assert_eq!(p.platform, "telegram");
        assert_eq!(p.chat_type, "dm");
        assert_eq!(p.chat_id, "123");
        assert_eq!(p.thread_id.as_deref(), Some("456"));
        // Group session — 6th element is NOT surfaced as thread_id.
        let g = parse_session_key("agent:main:slack:group:C1:U2").unwrap();
        assert_eq!(g.thread_id, None);
        assert!(parse_session_key("nope:main:x:y:z").is_none());
    }

    #[test]
    fn test_platform_config_key() {
        assert_eq!(platform_config_key("local"), "cli");
        assert_eq!(platform_config_key("telegram"), "telegram");
    }

    #[test]
    fn test_resolve_gateway_model() {
        assert_eq!(resolve_gateway_model(&json!({"model": "gpt-x"})), "gpt-x");
        assert_eq!(
            resolve_gateway_model(&json!({"model": {"default": "claude"}})),
            "claude"
        );
        assert_eq!(
            resolve_gateway_model(&json!({"model": {"model": "fallback"}})),
            "fallback"
        );
        assert_eq!(resolve_gateway_model(&json!({})), "");
    }

    #[test]
    fn test_build_media_placeholder() {
        let urls = vec!["a.png".to_string(), "b.mp3".to_string(), "c.bin".to_string()];
        let types = vec!["image/png".to_string(), "audio/mp3".to_string(), "".to_string()];
        let out = build_media_placeholder(&urls, &types, false);
        assert_eq!(
            out,
            "[User sent an image: a.png]\n[User sent audio: b.mp3]\n[User sent a file: c.bin]"
        );
        // is_photo forces image classification.
        let out2 = build_media_placeholder(&["x".to_string()], &[], true);
        assert_eq!(out2, "[User sent an image: x]");
    }

    #[test]
    fn test_control_interrupt_message() {
        assert!(is_control_interrupt_message(Some("  Stop   requested ")));
        assert!(is_control_interrupt_message(Some("Gateway shutting down")));
        assert!(!is_control_interrupt_message(Some("something the user said")));
        assert!(!is_control_interrupt_message(None));
        assert!(!is_control_interrupt_message(Some("")));
    }

    #[test]
    fn test_normalize_empty_agent_response() {
        // Non-empty response passes through.
        assert_eq!(normalize_empty_agent_response(&json!({}), "hi", 0), "hi");
        // Context failure heuristic.
        let r = normalize_empty_agent_response(
            &json!({"failed": true, "error": "context length exceeded"}),
            "",
            0,
        );
        assert!(r.contains("too large for the model's context"));
        // Generic failure.
        let r2 = normalize_empty_agent_response(&json!({"failed": true, "error": "boom"}), "", 0);
        assert!(r2.contains("The request failed: boom"));
        // Did work but no text.
        let r3 = normalize_empty_agent_response(&json!({"api_calls": 2}), "", 0);
        assert!(r3.contains("no response was generated"));
        // Partial.
        let r4 = normalize_empty_agent_response(&json!({"api_calls": 1, "partial": true, "error": "x"}), "", 0);
        assert!(r4.contains("Processing stopped: x"));
        // Empty + no work → empty.
        assert_eq!(normalize_empty_agent_response(&json!({}), "", 0), "");
    }

    #[test]
    fn test_format_gateway_process_notification() {
        let watch = json!({
            "type": "watch_match",
            "session_id": "S1",
            "command": "tail -f log",
            "pattern": "ERROR",
            "output": "boom",
            "suppressed": 3
        });
        let out = format_gateway_process_notification(&watch).unwrap();
        assert!(out.starts_with("[IMPORTANT: Background process S1 matched watch pattern \"ERROR\"."));
        assert!(out.contains("Command: tail -f log"));
        assert!(out.contains("3 earlier matches were suppressed"));
        assert!(out.ends_with(']'));

        let disabled = json!({"type": "watch_disabled", "message": "stopped"});
        assert_eq!(
            format_gateway_process_notification(&disabled),
            Some("[IMPORTANT: stopped]".to_string())
        );
        let other = json!({"type": "completion"});
        assert_eq!(format_gateway_process_notification(&other), None);
    }

    #[test]
    fn test_skill_slug_from_frontmatter() {
        let content = "---\nname: Stable Diffusion Image Generation\n---\nbody";
        let (slug, name) = skill_slug_from_frontmatter(content);
        assert_eq!(slug.as_deref(), Some("stable-diffusion-image-generation"));
        assert_eq!(name.as_deref(), Some("Stable Diffusion Image Generation"));
        // Quoted name with underscores.
        let q = "---\nname: \"my_skill\"\n---\n";
        let (slug2, _) = skill_slug_from_frontmatter(q);
        assert_eq!(slug2.as_deref(), Some("my-skill"));
        // No frontmatter.
        assert_eq!(skill_slug_from_frontmatter("no front"), (None, None));
        // Name folds to empty.
        let empty = "---\nname: ___\n---\n";
        let (s3, n3) = skill_slug_from_frontmatter(empty);
        assert_eq!(s3, None);
        assert_eq!(n3.as_deref(), Some("___"));
    }

    #[test]
    fn test_normalize_service_tier() {
        assert_eq!(normalize_service_tier("fast"), Some("priority".to_string()));
        assert_eq!(normalize_service_tier("priority"), Some("priority".to_string()));
        assert_eq!(normalize_service_tier("normal"), None);
        assert_eq!(normalize_service_tier(""), None);
        assert_eq!(normalize_service_tier("weird"), None);
    }

    #[test]
    fn test_resolve_busy_input_mode() {
        assert_eq!(resolve_busy_input_mode(Some("queue"), None), "queue");
        assert_eq!(resolve_busy_input_mode(Some(""), Some("STEER")), "steer");
        assert_eq!(resolve_busy_input_mode(None, None), "interrupt");
        assert_eq!(resolve_busy_input_mode(Some("nonsense"), None), "interrupt");
    }

    #[test]
    fn test_resolve_background_notifications_mode() {
        assert_eq!(resolve_background_notifications_mode(Some("error"), None), "error");
        assert_eq!(
            resolve_background_notifications_mode(Some(""), Some(&json!(false))),
            "off"
        );
        assert_eq!(resolve_background_notifications_mode(None, None), "all");
        assert_eq!(
            resolve_background_notifications_mode(Some("bogus"), None),
            "all"
        );
        assert_eq!(
            resolve_background_notifications_mode(Some(""), Some(&json!("result"))),
            "result"
        );
    }

    #[test]
    fn test_platform_connect_timeout() {
        assert_eq!(platform_connect_timeout_secs(Some(&json!(45.0))), 45.0);
        assert_eq!(platform_connect_timeout_secs(Some(&json!("10"))), 10.0);
        assert_eq!(platform_connect_timeout_secs(Some(&json!(-1))), 30.0);
        assert_eq!(platform_connect_timeout_secs(None), 30.0);
    }

    #[test]
    fn test_parse_reasoning_command_args() {
        assert_eq!(parse_reasoning_command_args("high"), ("high".to_string(), false));
        assert_eq!(
            parse_reasoning_command_args("--global low"),
            ("low".to_string(), true)
        );
        assert_eq!(
            parse_reasoning_command_args("HIGH —global"),
            ("high".to_string(), true)
        );
        assert_eq!(parse_reasoning_command_args(""), (String::new(), false));
    }

    #[test]
    fn test_parse_voice_modes() {
        let raw = r#"{"telegram:1": "all", "slack:2": "off", "bad:3": "weird", "legacy": "all"}"#;
        let modes = parse_voice_modes(raw);
        assert_eq!(modes.get("telegram:1").map(String::as_str), Some("all"));
        assert_eq!(modes.get("slack:2").map(String::as_str), Some("off"));
        assert!(!modes.contains_key("bad:3")); // invalid mode
        assert!(!modes.contains_key("legacy")); // unprefixed
        // Round-trip.
        let ser = serialize_voice_modes(&modes);
        assert!(ser.contains("telegram:1"));
        assert_eq!(parse_voice_modes("not json"), BTreeMap::new());
    }

    #[test]
    fn test_voice_mode_adapter_sets() {
        let mut modes = BTreeMap::new();
        modes.insert("telegram:a".to_string(), "off".to_string());
        modes.insert("telegram:b".to_string(), "all".to_string());
        modes.insert("telegram:c".to_string(), "voice_only".to_string());
        modes.insert("slack:d".to_string(), "all".to_string());
        let (disabled, enabled) = voice_mode_adapter_sets(&modes, "telegram");
        assert_eq!(disabled, vec!["a".to_string()]);
        assert_eq!(enabled, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_telegram_topic_classification() {
        assert!(is_telegram_topic_root_lobby("telegram", "dm", Some(""), true));
        assert!(is_telegram_topic_root_lobby("telegram", "dm", Some("1"), true));
        assert!(!is_telegram_topic_root_lobby("telegram", "dm", Some("99"), true));
        assert!(!is_telegram_topic_root_lobby("telegram", "dm", Some(""), false));

        assert!(is_telegram_topic_lane("telegram", "dm", Some("99"), true));
        assert!(!is_telegram_topic_lane("telegram", "dm", Some("1"), true));
        assert!(!is_telegram_topic_lane("slack", "dm", Some("99"), true));

        assert!(telegram_topic_new_header("telegram", "dm", Some("99"), true).is_some());
        assert!(telegram_topic_new_header("telegram", "dm", Some(""), true).is_none());
    }

    #[test]
    fn test_telegramize_command_mentions() {
        let sanitize = |s: &str| s.to_lowercase().replace('-', "_");
        let out = telegramize_command_mentions("Use /Foo-Bar now", "telegram", &sanitize);
        assert_eq!(out, "Use /foo_bar now");
        // Non-telegram unchanged.
        let out2 = telegramize_command_mentions("Use /Foo-Bar now", "discord", &sanitize);
        assert_eq!(out2, "Use /Foo-Bar now");
        // Empty sanitization leaves the original.
        let blank = |_: &str| String::new();
        let out3 = telegramize_command_mentions("/x", "telegram", &blank);
        assert_eq!(out3, "/x");
    }

    #[test]
    fn test_extract_cache_busting_config() {
        let cfg = json!({
            "model": {"context_length": 200000},
            "compression": {"enabled": true, "threshold": 0.8}
        });
        let out = extract_cache_busting_config(&cfg, Some(7));
        assert_eq!(out.get("model.context_length"), Some(&json!(200000)));
        assert_eq!(out.get("compression.enabled"), Some(&json!(true)));
        // Missing keys → null.
        assert_eq!(out.get("compression.protect_last_n"), Some(&Value::Null));
        assert_eq!(out.get("agent.disabled_toolsets"), Some(&Value::Null));
        assert_eq!(out.get("tools.registry_generation"), Some(&json!(7)));
    }

    #[test]
    fn test_agent_config_signature_stable_and_sensitive() {
        let runtime = json!({"api_key": "secret", "base_url": "u", "provider": "p", "api_mode": "m"});
        let sig1 = agent_config_signature("model-a", &runtime, &["t2".into(), "t1".into()], "prompt", None);
        // Toolset order doesn't matter (sorted internally).
        let sig2 = agent_config_signature("model-a", &runtime, &["t1".into(), "t2".into()], "prompt", None);
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), 16);
        // Different model → different signature.
        let sig3 = agent_config_signature("model-b", &runtime, &["t1".into()], "prompt", None);
        assert_ne!(sig1, sig3);
        // Different api key → different signature.
        let runtime2 = json!({"api_key": "other"});
        let sig4 = agent_config_signature("model-a", &runtime2, &[], "prompt", None);
        let sig5 = agent_config_signature("model-a", &runtime, &[], "prompt", None);
        assert_ne!(sig4, sig5);
    }

    #[test]
    fn test_cron_tick_actions() {
        let a = cron_tick_actions(5, true);
        assert!(a.channel_directory);
        assert!(!a.image_cache);
        let b = cron_tick_actions(60, true);
        assert!(b.image_cache);
        assert!(b.paste_sweep);
        assert!(b.curator);
        assert!(b.channel_directory); // 60 % 5 == 0
        // No adapters → no channel directory.
        let c = cron_tick_actions(5, false);
        assert!(!c.channel_directory);
    }

    #[test]
    fn test_apply_session_model_override() {
        let runtime = json!({"provider": "old", "api_key": "k"});
        let ov = json!({"model": "new-model", "provider": "new", "base_url": null});
        let (model, rt) = apply_session_model_override("base", &runtime, Some(&ov));
        assert_eq!(model, "new-model");
        assert_eq!(rt["provider"], json!("new"));
        // null base_url not applied.
        assert!(rt.get("base_url").is_none());
        // No override → unchanged.
        let (m2, _) = apply_session_model_override("base", &runtime, None);
        assert_eq!(m2, "base");
    }

    #[test]
    fn test_is_intentional_model_switch() {
        let ov = json!({"model": "x"});
        assert!(is_intentional_model_switch(Some(&ov), "x"));
        assert!(!is_intentional_model_switch(Some(&ov), "y"));
        assert!(!is_intentional_model_switch(None, "x"));
    }

    #[test]
    fn test_bridge_config_to_env() {
        let cfg = json!({
            "simple_top": "val",
            "terminal": {"cwd": "/abs/path", "timeout": 30, "docker_volumes": ["a:b"]},
            "agent": {"max_turns": 500, "gateway_timeout": 1800},
            "display": {"busy_input_mode": "queue"},
            "timezone": " UTC ",
            "security": {"redact_secrets": true},
            "auxiliary": {"vision": {"provider": "openai", "model": "gpt-4o"}}
        });
        let mut existing = BTreeMap::new();
        existing.insert("already_set".to_string(), "x".to_string());
        let out = bridge_config_to_env(&cfg, &existing);
        let find = |name: &str| out.iter().find(|a| a.name == name).map(|a| a.value.clone());
        assert_eq!(find("simple_top"), Some("val".to_string()));
        assert_eq!(find("TERMINAL_CWD"), Some("/abs/path".to_string()));
        assert_eq!(find("TERMINAL_TIMEOUT"), Some("30".to_string()));
        assert_eq!(find("TERMINAL_DOCKER_VOLUMES"), Some("[\"a:b\"]".to_string()));
        assert_eq!(find("HERMES_MAX_ITERATIONS"), Some("500".to_string()));
        assert_eq!(find("HERMES_AGENT_TIMEOUT"), Some("1800".to_string()));
        assert_eq!(find("HERMES_GATEWAY_BUSY_INPUT_MODE"), Some("queue".to_string()));
        assert_eq!(find("HERMES_TIMEZONE"), Some("UTC".to_string()));
        assert_eq!(find("HERMES_REDACT_SECRETS"), Some("true".to_string()));
        assert_eq!(find("AUXILIARY_VISION_PROVIDER"), Some("openai".to_string()));
        assert_eq!(find("AUXILIARY_VISION_MODEL"), Some("gpt-4o".to_string()));
    }

    #[test]
    fn test_bridge_cwd_placeholder_skipped() {
        let cfg = json!({"terminal": {"cwd": "auto"}});
        let out = bridge_config_to_env(&cfg, &BTreeMap::new());
        assert!(out.iter().all(|a| a.name != "TERMINAL_CWD"));
    }

    #[test]
    fn test_resolve_terminal_cwd() {
        let mut env = BTreeMap::new();
        env.insert("TERMINAL_CWD".to_string(), "/real/path".to_string());
        assert_eq!(resolve_terminal_cwd(&env), "/real/path");
        env.insert("TERMINAL_CWD".to_string(), "auto".to_string());
        env.insert("MESSAGING_CWD".to_string(), "/fallback".to_string());
        assert_eq!(resolve_terminal_cwd(&env), "/fallback");
    }

    #[test]
    fn test_float_env_and_freshness_window() {
        unsafe {
            std::env::set_var("HERMES_TEST_FLOAT", "2.5");
        }
        assert_eq!(float_env("HERMES_TEST_FLOAT", 1.0), 2.5);
        unsafe {
            std::env::set_var("HERMES_TEST_FLOAT", "bad");
        }
        assert_eq!(float_env("HERMES_TEST_FLOAT", 1.0), 1.0);
        unsafe {
            std::env::remove_var("HERMES_TEST_FLOAT");
        }
        assert_eq!(float_env("HERMES_TEST_FLOAT", 7.0), 7.0);

        unsafe {
            std::env::remove_var("HERMES_AUTO_CONTINUE_FRESHNESS");
        }
        assert_eq!(auto_continue_freshness_window(), AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT);
        unsafe {
            std::env::set_var("HERMES_AUTO_CONTINUE_FRESHNESS", "120");
        }
        assert_eq!(auto_continue_freshness_window(), 120.0);
        unsafe {
            std::env::remove_var("HERMES_AUTO_CONTINUE_FRESHNESS");
        }
    }

    #[test]
    fn test_home_target_env_var() {
        let mut overrides = BTreeMap::new();
        overrides.insert("matrix".to_string(), "MATRIX_HOME_ROOM".to_string());
        assert_eq!(home_target_env_var("matrix", &overrides), "MATRIX_HOME_ROOM");
        assert_eq!(home_target_env_var("discord", &overrides), "DISCORD_HOME_CHANNEL");
        assert_eq!(
            home_thread_env_var("discord", &overrides),
            "DISCORD_HOME_CHANNEL_THREAD_ID"
        );
    }

    #[test]
    fn test_voice_key() {
        assert_eq!(voice_key("telegram", "123"), "telegram:123");
    }
}
