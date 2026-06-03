//! Native port of `hermes_cli/dump.py`.
//!
//! Outputs a compact, plain-text summary of the user's Hermes setup that can be
//! copy-pasted into Discord/GitHub/Telegram for support context. No ANSI
//! colors, no checkmarks — just data.
//!
//! The Python entry point is `run_dump(args)`; the only argument it consults is
//! `args.show_keys`. The equivalent here is [`run_dump`], which takes a
//! [`DumpArgs`] and returns the rendered multi-line string (the caller prints
//! it). [`render_dump`] is a lower-level entry point that takes pre-resolved
//! inputs so it can be unit-tested without touching the environment.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

/// Arguments accepted by `hermes dump`.
#[derive(Debug, Clone, Default)]
pub struct DumpArgs {
    /// When true, partially-redacted API key values are shown instead of
    /// "set" / "not set".
    pub show_keys: bool,
}

/// API keys reported by the dump, as `(env_var, label)`.
///
/// Order is significant: it matches the Python `api_keys` list verbatim so the
/// output is byte-for-byte identical.
pub const API_KEYS: &[(&str, &str)] = &[
    ("OPENROUTER_API_KEY", "openrouter"),
    ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("ANTHROPIC_TOKEN", "anthropic_token"),
    ("NOUS_API_KEY", "nous"),
    ("GOOGLE_API_KEY", "google/gemini"),
    ("GEMINI_API_KEY", "gemini"),
    ("GLM_API_KEY", "glm/zai"),
    ("ZAI_API_KEY", "zai"),
    ("KIMI_API_KEY", "kimi"),
    ("MINIMAX_API_KEY", "minimax"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("DASHSCOPE_API_KEY", "dashscope"),
    ("HF_TOKEN", "huggingface"),
    ("NVIDIA_API_KEY", "nvidia"),
    ("AI_GATEWAY_API_KEY", "ai_gateway"),
    ("OPENCODE_ZEN_API_KEY", "opencode_zen"),
    ("OPENCODE_GO_API_KEY", "opencode_go"),
    ("KILOCODE_API_KEY", "kilocode"),
    ("FIRECRAWL_API_KEY", "firecrawl"),
    ("TAVILY_API_KEY", "tavily"),
    ("BROWSERBASE_API_KEY", "browserbase"),
    ("FAL_KEY", "fal"),
    ("ELEVENLABS_API_KEY", "elevenlabs"),
    ("GITHUB_TOKEN", "github"),
];

/// Messaging platforms and the env var whose presence indicates configuration.
///
/// Mirrors Python's `_configured_platforms` `checks` dict (insertion order
/// preserved).
pub const PLATFORM_CHECKS: &[(&str, &str)] = &[
    ("telegram", "TELEGRAM_BOT_TOKEN"),
    ("discord", "DISCORD_BOT_TOKEN"),
    ("slack", "SLACK_BOT_TOKEN"),
    ("whatsapp", "WHATSAPP_ENABLED"),
    ("signal", "SIGNAL_HTTP_URL"),
    ("email", "EMAIL_ADDRESS"),
    ("sms", "TWILIO_ACCOUNT_SID"),
    ("matrix", "MATRIX_HOMESERVER_URL"),
    ("mattermost", "MATTERMOST_URL"),
    ("homeassistant", "HASS_TOKEN"),
    ("dingtalk", "DINGTALK_CLIENT_ID"),
    ("feishu", "FEISHU_APP_ID"),
    ("wecom", "WECOM_BOT_ID"),
    ("wecom_callback", "WECOM_CALLBACK_CORP_ID"),
    ("weixin", "WEIXIN_ACCOUNT_ID"),
    ("qqbot", "QQ_APP_ID"),
];

/// `(section, key)` config dotpaths reported as overrides when they differ from
/// the packaged default. Mirrors Python's `interesting_paths`.
pub const INTERESTING_OVERRIDES: &[(&str, &str)] = &[
    ("agent", "max_turns"),
    ("agent", "gateway_timeout"),
    ("agent", "tool_use_enforcement"),
    ("terminal", "backend"),
    ("terminal", "docker_image"),
    ("terminal", "persistent_shell"),
    ("browser", "allow_private_urls"),
    ("compression", "enabled"),
    ("compression", "threshold"),
    ("display", "streaming"),
    ("display", "skin"),
    ("display", "show_reasoning"),
    ("privacy", "redact_pii"),
    ("tts", "provider"),
];

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Return short git commit hash, or `"(unknown)"`.
///
/// Faithful port of `_get_git_commit`: runs `git rev-parse --short=8 HEAD` in
/// `project_root` and returns the trimmed stdout on success.
pub fn get_git_commit(project_root: &Path) -> String {
    let result = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .current_dir(project_root)
        .output();
    if let Ok(out) = result {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    "(unknown)".to_string()
}

/// Redact all but first 4 and last 4 chars.
///
/// Thin wrapper over [`crate::agent_redact::mask_secret_default`] in the wider
/// build; here we inline the identical algorithm so the module stands alone if
/// `agent.redact` is unported. Returns `""` for an empty value (matching the
/// historical behavior — `hermes dump` formats empty values as blank, not as
/// `"(not set)"`).
pub fn redact(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    let len = chars.len();
    // mask_secret default: floor=12, head=4, tail=4.
    if len < 12 {
        return "***".to_string();
    }
    let head: String = chars[..4].iter().collect();
    let tail: String = chars[len - 4..].iter().collect();
    format!("{head}...{tail}")
}

/// Count installed skills under `<hermes_home>/skills` by recursively counting
/// `SKILL.md` files. Returns 0 if the directory does not exist.
pub fn count_skills(hermes_home: &Path) -> usize {
    let skills_dir = hermes_home.join("skills");
    if !skills_dir.is_dir() {
        return 0;
    }
    count_skill_md(&skills_dir)
}

fn count_skill_md(dir: &Path) -> usize {
    let mut count = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            count += count_skill_md(&path);
        } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
            count += 1;
        }
    }
    count
}

/// Count configured MCP servers from `config["mcp"]["servers"]`.
pub fn count_mcp_servers(config: &YamlValue) -> usize {
    config
        .get("mcp")
        .and_then(|m| m.get("servers"))
        .and_then(|s| s.as_mapping())
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Return cron jobs summary string from `<hermes_home>/cron/jobs.json`.
///
/// `"0"` when the file is absent, `"<active> active / <total> total"` when it
/// parses, `"(error reading)"` on read/parse failure.
pub fn cron_summary(hermes_home: &Path) -> String {
    let jobs_file = hermes_home.join("cron").join("jobs.json");
    if !jobs_file.exists() {
        return "0".to_string();
    }
    let text = match std::fs::read_to_string(&jobs_file) {
        Ok(t) => t,
        Err(_) => return "(error reading)".to_string(),
    };
    let data: JsonValue = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(_) => return "(error reading)".to_string(),
    };
    let empty: Vec<JsonValue> = Vec::new();
    let jobs = data
        .get("jobs")
        .and_then(|j| j.as_array())
        .unwrap_or(&empty);
    let total = jobs.len();
    // enabled defaults to true when key is absent or non-boolean.
    let active = jobs
        .iter()
        .filter(|j| {
            j.get("enabled")
                .map(|e| e.as_bool().unwrap_or(true))
                .unwrap_or(true)
        })
        .count();
    format!("{active} active / {total} total")
}

/// Return list of configured messaging platform names (those whose env var is
/// set to a non-empty value).
pub fn configured_platforms() -> Vec<String> {
    PLATFORM_CHECKS
        .iter()
        .filter(|(_, env)| env_nonempty(env))
        .map(|(name, _)| name.to_string())
        .collect()
}

fn env_nonempty(key: &str) -> bool {
    std::env::var(key).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Return the active memory provider name (`"built-in"` when unset/empty).
pub fn memory_provider(config: &YamlValue) -> String {
    let provider = config
        .get("memory")
        .and_then(|m| m.get("provider"))
        .and_then(|p| p.as_str())
        .unwrap_or("");
    if provider.is_empty() {
        "built-in".to_string()
    } else {
        provider.to_string()
    }
}

/// Extract `(model, provider)` from config.
///
/// `config["model"]` may be a mapping (`default`/`model`/`name` for model,
/// `provider` for provider), a plain string, or absent.
pub fn get_model_and_provider(config: &YamlValue) -> (String, String) {
    let model_cfg = config.get("model");
    match model_cfg {
        Some(YamlValue::Mapping(map)) => {
            let model = first_nonempty_str(map, &["default", "model", "name"])
                .unwrap_or_else(|| "(not set)".to_string());
            let provider = map
                .get(YamlValue::from("provider"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "(auto)".to_string());
            (model, provider)
        }
        Some(YamlValue::String(s)) => {
            let model = if s.is_empty() {
                "(not set)".to_string()
            } else {
                s.clone()
            };
            (model, "(auto)".to_string())
        }
        _ => ("(not set)".to_string(), "(auto)".to_string()),
    }
}

fn first_nonempty_str(map: &serde_yaml::Mapping, keys: &[&str]) -> Option<String> {
    // Mirrors `a or b or c`: the first key whose value is "truthy" (non-empty
    // string). Python's `or` would also accept a non-string truthy value but
    // these keys hold strings in practice.
    for key in keys {
        if let Some(v) = map.get(YamlValue::from(*key)) {
            if let Some(s) = v.as_str() {
                if !s.is_empty() {
                    return Some(s.to_string());
                }
            }
        }
    }
    None
}

/// Render a YAML scalar/collection the way Python's `str()` would for the
/// override report. Strings render bare; bools become `True`/`False`; lists
/// render Python-style (`['a', 'b']`).
fn py_str(value: &YamlValue) -> String {
    match value {
        YamlValue::Null => "None".to_string(),
        YamlValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        YamlValue::Number(n) => n.to_string(),
        YamlValue::String(s) => s.clone(),
        YamlValue::Sequence(seq) => {
            let items: Vec<String> = seq.iter().map(py_repr).collect();
            format!("[{}]", items.join(", "))
        }
        YamlValue::Mapping(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        YamlValue::Tagged(t) => py_str(&t.value),
    }
}

/// Python `repr()` of a value as it appears nested inside a list/dict.
fn py_repr(value: &YamlValue) -> String {
    match value {
        YamlValue::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        other => py_str(other),
    }
}

/// Find non-default config values worth reporting. Returns an ordered map of
/// dotpath -> Python-style stringified value.
pub fn config_overrides(config: &YamlValue, defaults: &YamlValue) -> Vec<(String, String)> {
    let mut overrides: Vec<(String, String)> = Vec::new();

    for (section, key) in INTERESTING_OVERRIDES {
        let default_section = defaults.get(section);
        let user_section = config.get(section);
        // Both must be mappings (matching the isinstance(..., dict) guard).
        let (Some(YamlValue::Mapping(_)), Some(YamlValue::Mapping(_))) =
            (default_section, user_section)
        else {
            continue;
        };
        let default_val = default_section.and_then(|s| s.get(key));
        let user_val = user_section.and_then(|s| s.get(key));

        // `user_val is not None and user_val != default_val`
        if let Some(uv) = user_val {
            if !matches!(uv, YamlValue::Null) && Some(uv) != default_val {
                overrides.push((format!("{section}.{key}"), py_str(uv)));
            }
        }
    }

    // Toolsets (if different from default).
    let default_toolsets = defaults.get("toolsets").cloned().unwrap_or(empty_seq());
    let user_toolsets = config.get("toolsets").cloned().unwrap_or(empty_seq());
    if user_toolsets != default_toolsets {
        overrides.push(("toolsets".to_string(), py_str(&user_toolsets)));
    }

    // Fallback providers (if any).
    if let Some(fallbacks) = config.get("fallback_providers") {
        let nonempty = match fallbacks {
            YamlValue::Sequence(s) => !s.is_empty(),
            YamlValue::Null => false,
            _ => true,
        };
        if nonempty {
            overrides.push(("fallback_providers".to_string(), py_str(fallbacks)));
        }
    }

    overrides
}

fn empty_seq() -> YamlValue {
    YamlValue::Sequence(Vec::new())
}

// ─── Inputs ─────────────────────────────────────────────────────────────────

/// Pre-resolved inputs for [`render_dump`], so rendering can be tested without
/// touching git/env/config-loading.
#[derive(Debug, Clone)]
pub struct DumpInputs {
    pub version: String,
    pub release_date: String,
    pub commit: String,
    pub os_info: String,
    pub runtime_version: String,
    /// Equivalent of Python's `openai.__version__` line; the native runtime has
    /// no OpenAI SDK, so callers typically pass `"native (rust)"` or
    /// `"not installed"`.
    pub openai_sdk: String,
    pub profile: String,
    pub display_hermes_home: String,
    pub config: YamlValue,
    pub defaults: YamlValue,
    pub hermes_home: std::path::PathBuf,
    pub gateway_status: String,
    pub show_keys: bool,
}

/// Render the dump text from fully-resolved inputs (no I/O beyond filesystem
/// scans for skills/cron, which are derived from `hermes_home`).
pub fn render_dump(input: &DumpInputs) -> String {
    let (model, provider) = get_model_and_provider(&input.config);

    let backend = input
        .config
        .get("terminal")
        .and_then(|t| t.get("backend"))
        .and_then(|b| b.as_str())
        .unwrap_or("local")
        .to_string();

    let mut lines: Vec<String> = Vec::new();
    lines.push("--- hermes dump ---".to_string());

    let mut ver_str = input.version.clone();
    if !input.release_date.is_empty() {
        ver_str.push_str(&format!(" ({})", input.release_date));
    }
    ver_str.push_str(&format!(" [{}]", input.commit));

    lines.push(format!("version:          {ver_str}"));
    lines.push(format!("os:               {}", input.os_info));
    lines.push(format!("python:           {}", input.runtime_version));
    lines.push(format!("openai_sdk:       {}", input.openai_sdk));
    lines.push(format!("profile:          {}", input.profile));
    lines.push(format!("hermes_home:      {}", input.display_hermes_home));
    lines.push(format!("model:            {model}"));
    lines.push(format!("provider:         {provider}"));
    lines.push(format!("terminal:         {backend}"));

    // API keys.
    lines.push(String::new());
    lines.push("api_keys:".to_string());
    for (env_var, label) in API_KEYS {
        let val = std::env::var(env_var).unwrap_or_default();
        let display = if input.show_keys && !val.is_empty() {
            redact(&val)
        } else if val.is_empty() {
            "not set".to_string()
        } else {
            "set".to_string()
        };
        // Python: f"  {label:<20} {display}" — label left-justified to 20.
        lines.push(format!("  {label:<20} {display}"));
    }

    // Features summary.
    lines.push(String::new());
    lines.push("features:".to_string());

    let toolsets = toolsets_list(&input.config);
    let toolsets_str = if toolsets.is_empty() {
        "(default)".to_string()
    } else {
        toolsets.join(", ")
    };
    lines.push(format!("  toolsets:           {toolsets_str}"));
    lines.push(format!(
        "  mcp_servers:        {}",
        count_mcp_servers(&input.config)
    ));
    lines.push(format!(
        "  memory_provider:    {}",
        memory_provider(&input.config)
    ));
    lines.push(format!("  gateway:            {}", input.gateway_status));

    let platforms = configured_platforms();
    let platforms_str = if platforms.is_empty() {
        "none".to_string()
    } else {
        platforms.join(", ")
    };
    lines.push(format!("  platforms:          {platforms_str}"));
    lines.push(format!(
        "  cron_jobs:          {}",
        cron_summary(&input.hermes_home)
    ));
    lines.push(format!(
        "  skills:             {}",
        count_skills(&input.hermes_home)
    ));

    // Config overrides (non-default values).
    let overrides = config_overrides(&input.config, &input.defaults);
    if !overrides.is_empty() {
        lines.push(String::new());
        lines.push("config_overrides:".to_string());
        for (key, val) in &overrides {
            lines.push(format!("  {key}: {val}"));
        }
    }

    lines.push("--- end dump ---".to_string());

    lines.join("\n")
}

/// Mirror Python's `config.get("toolsets", ["hermes-cli"])` for the features
/// line: default to `["hermes-cli"]` when the key is absent, otherwise use the
/// configured list (which may be empty -> rendered as "(default)" by caller).
fn toolsets_list(config: &YamlValue) -> Vec<String> {
    match config.get("toolsets") {
        Some(YamlValue::Sequence(seq)) => seq
            .iter()
            .map(|v| match v {
                YamlValue::String(s) => s.clone(),
                other => py_str(other),
            })
            .collect(),
        Some(_) => Vec::new(),
        None => vec!["hermes-cli".to_string()],
    }
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Resolve runtime inputs and render the dump.
///
/// This is the native equivalent of Python's `run_dump`. The wider build wires
/// it through the helper modules below; if any are unported the closures here
/// can be replaced with parameters. Returns the rendered string (the CLI
/// command prints it).
pub fn run_dump(args: &DumpArgs) -> String {
    use hermes_core::{HermesConfig, HermesContext};

    let ctx = HermesContext::detect();
    let hermes_home = ctx.hermes_home();

    // Python uses `get_project_root()`; the wider build can pass the repo root.
    // We fall back to the current working directory for the git probe.
    let project_root = std::env::current_dir().unwrap_or_else(|_| hermes_home.clone());
    let commit = get_git_commit(&project_root);

    // Loaded config (LoadedConfig.raw is a serde_yaml::Value — exactly what the
    // helpers consume). Falls back to an empty mapping on failure, matching the
    // Python `except Exception: config = {}`.
    let config = ctx
        .load_config_document()
        .map(|doc| doc.raw)
        .unwrap_or_else(|_| YamlValue::Mapping(Default::default()));

    // DEFAULT_CONFIG equivalent: serialize the typed default config.
    let defaults =
        serde_yaml::to_value(HermesConfig::default()).unwrap_or(YamlValue::Mapping(Default::default()));

    let profile = crate_active_profile();

    let input = DumpInputs {
        version: crate_version(),
        release_date: crate_release_date(),
        commit,
        os_info: os_info(),
        runtime_version: runtime_version(),
        openai_sdk: "native (rust)".to_string(),
        profile,
        display_hermes_home: ctx.display_hermes_home(),
        config,
        defaults,
        hermes_home: hermes_home.clone(),
        gateway_status: gateway_status_str(&hermes_home),
        show_keys: args.show_keys,
    };

    render_dump(&input)
}

// ─── Glue to ported helper modules ────────────────────────────────────────
//
// These wrappers isolate cross-module dependencies in one place.

fn crate_active_profile() -> String {
    let name = crate::cli_profiles::get_active_profile_name();
    if name.is_empty() {
        "(default)".to_string()
    } else {
        name
    }
}

fn crate_version() -> String {
    // Mirrors `from hermes_cli import __version__`. Falls back when unset.
    option_env!("CARGO_PKG_VERSION")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "(unknown)".to_string())
}

fn crate_release_date() -> String {
    // Python's `__release_date__`; empty when unknown.
    "".to_string()
}

/// OS info string: `"<system> <release> <machine>"`.
pub fn os_info() -> String {
    let system = match std::env::consts::OS {
        "macos" => "Darwin".to_string(),
        "linux" => "Linux".to_string(),
        "windows" => "Windows".to_string(),
        other => {
            let mut c = other.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => other.to_string(),
            }
        }
    };
    let release = os_release();
    let machine = std::env::consts::ARCH.to_string();
    format!("{system} {release} {machine}")
}

fn os_release() -> String {
    let out = Command::new("uname").arg("-r").output();
    if let Ok(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "".to_string()
}

/// Runtime version line (Python reports `sys.version`). For the native build we
/// report the Rust crate version of the runtime.
pub fn runtime_version() -> String {
    option_env!("CARGO_PKG_RUST_VERSION")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "native".to_string())
}

/// Short gateway status string derived from `hermes_home`. Mirrors the shape of
/// Python's `_gateway_status` output strings.
pub fn gateway_status_str(hermes_home: &Path) -> String {
    // Best-effort: detect a running gateway via the pid file. The full Python
    // helper consults the service manager; here we fall back to a simple
    // running/stopped report based on the pid file the gateway writes.
    let pid_path = hermes_home.join("gateway.pid");
    if let Ok(text) = std::fs::read_to_string(&pid_path) {
        if let Ok(pid) = text.trim().parse::<i32>() {
            if pid_alive(pid) {
                return format!("running (manual, pid {pid})");
            }
        }
    }
    "stopped (manual)".to_string()
}

#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    // kill(pid, 0) — succeeds if the process exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
fn pid_alive(_pid: i32) -> bool {
    false
}

// Suppress unused warning for the BTreeMap import in builds that don't exercise
// every helper. (Kept available for callers that want a map view of overrides.)
#[allow(dead_code)]
fn overrides_map(config: &YamlValue, defaults: &YamlValue) -> BTreeMap<String, String> {
    config_overrides(config, defaults).into_iter().collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::from_str as yaml;

    #[test]
    fn redact_empty_is_blank() {
        assert_eq!(redact(""), "");
    }

    #[test]
    fn redact_short_fully_masked() {
        // < 12 chars -> "***"
        assert_eq!(redact("short"), "***");
        assert_eq!(redact("12345678901"), "***");
    }

    #[test]
    fn redact_long_keeps_edges() {
        assert_eq!(redact("sk-abcdefghijklmnop"), "sk-a...mnop");
    }

    #[test]
    fn model_from_mapping_prefers_default() {
        let cfg: YamlValue = yaml(
            "model:\n  default: gpt-5\n  model: ignored\n  provider: openai\n",
        )
        .unwrap();
        assert_eq!(
            get_model_and_provider(&cfg),
            ("gpt-5".to_string(), "openai".to_string())
        );
    }

    #[test]
    fn model_from_mapping_falls_through_keys() {
        let cfg: YamlValue = yaml("model:\n  name: claude\n").unwrap();
        assert_eq!(
            get_model_and_provider(&cfg),
            ("claude".to_string(), "(auto)".to_string())
        );
    }

    #[test]
    fn model_from_string() {
        let cfg: YamlValue = yaml("model: gpt-4o\n").unwrap();
        assert_eq!(
            get_model_and_provider(&cfg),
            ("gpt-4o".to_string(), "(auto)".to_string())
        );
    }

    #[test]
    fn model_absent() {
        let cfg: YamlValue = yaml("other: 1\n").unwrap();
        assert_eq!(
            get_model_and_provider(&cfg),
            ("(not set)".to_string(), "(auto)".to_string())
        );
    }

    #[test]
    fn memory_provider_defaults_builtin() {
        let cfg: YamlValue = yaml("memory:\n  provider: ''\n").unwrap();
        assert_eq!(memory_provider(&cfg), "built-in");
        let cfg2: YamlValue = yaml("memory:\n  provider: mem0\n").unwrap();
        assert_eq!(memory_provider(&cfg2), "mem0");
        let cfg3: YamlValue = yaml("other: 1\n").unwrap();
        assert_eq!(memory_provider(&cfg3), "built-in");
    }

    #[test]
    fn count_mcp_servers_counts_mapping() {
        let cfg: YamlValue =
            yaml("mcp:\n  servers:\n    a: {}\n    b: {}\n").unwrap();
        assert_eq!(count_mcp_servers(&cfg), 2);
        let cfg2: YamlValue = yaml("mcp: {}\n").unwrap();
        assert_eq!(count_mcp_servers(&cfg2), 0);
    }

    #[test]
    fn cron_summary_missing_is_zero() {
        let dir = std::env::temp_dir().join(format!("hermes_dump_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(cron_summary(&dir), "0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cron_summary_counts_enabled() {
        let dir = std::env::temp_dir()
            .join(format!("hermes_dump_cron_{}", std::process::id()));
        let cron = dir.join("cron");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&cron).unwrap();
        std::fs::write(
            cron.join("jobs.json"),
            r#"{"jobs":[{"enabled":true},{"enabled":false},{}]}"#,
        )
        .unwrap();
        // enabled defaults true -> 2 active of 3
        assert_eq!(cron_summary(&dir), "2 active / 3 total");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_overrides_detects_difference() {
        let defaults: YamlValue =
            yaml("agent:\n  max_turns: 50\nterminal:\n  backend: local\n").unwrap();
        let config: YamlValue =
            yaml("agent:\n  max_turns: 100\nterminal:\n  backend: local\n").unwrap();
        let ov = config_overrides(&config, &defaults);
        assert!(ov
            .iter()
            .any(|(k, v)| k == "agent.max_turns" && v == "100"));
        // backend unchanged -> not present
        assert!(!ov.iter().any(|(k, _)| k == "terminal.backend"));
    }

    #[test]
    fn config_overrides_toolsets_and_fallbacks() {
        let defaults: YamlValue = yaml("toolsets:\n  - hermes-cli\n").unwrap();
        let config: YamlValue =
            yaml("toolsets:\n  - hermes-cli\n  - web\nfallback_providers:\n  - openai\n")
                .unwrap();
        let ov = config_overrides(&config, &defaults);
        let map: BTreeMap<_, _> = ov.into_iter().collect();
        assert_eq!(map.get("toolsets").map(String::as_str), Some("['hermes-cli', 'web']"));
        assert_eq!(
            map.get("fallback_providers").map(String::as_str),
            Some("['openai']")
        );
    }

    #[test]
    fn py_str_renders_bool_python_style() {
        assert_eq!(py_str(&YamlValue::Bool(true)), "True");
        assert_eq!(py_str(&YamlValue::Bool(false)), "False");
    }

    #[test]
    fn render_dump_basic_shape() {
        let input = DumpInputs {
            version: "1.2.3".to_string(),
            release_date: "2026.4.30".to_string(),
            commit: "abcd1234".to_string(),
            os_info: "Darwin 25.1.0 arm64".to_string(),
            runtime_version: "native".to_string(),
            openai_sdk: "native (rust)".to_string(),
            profile: "(default)".to_string(),
            display_hermes_home: "~/.hermes".to_string(),
            config: yaml("model: gpt-5\nterminal:\n  backend: docker\n").unwrap(),
            defaults: yaml("terminal:\n  backend: local\n").unwrap(),
            hermes_home: std::env::temp_dir().join("hermes_dump_render_nonexistent"),
            gateway_status: "stopped (manual)".to_string(),
            show_keys: false,
        };
        let out = render_dump(&input);
        assert!(out.starts_with("--- hermes dump ---\n"));
        assert!(out.ends_with("--- end dump ---"));
        assert!(out.contains("version:          1.2.3 (2026.4.30) [abcd1234]"));
        assert!(out.contains("model:            gpt-5"));
        assert!(out.contains("terminal:         docker"));
        assert!(out.contains("config_overrides:"));
        assert!(out.contains("  terminal.backend: docker"));
    }

    #[test]
    fn render_dump_api_keys_set_and_notset() {
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-testtesttesttest");
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
        let input = DumpInputs {
            version: "0".to_string(),
            release_date: String::new(),
            commit: "x".to_string(),
            os_info: "Linux 6 x86_64".to_string(),
            runtime_version: "native".to_string(),
            openai_sdk: "native (rust)".to_string(),
            profile: "default".to_string(),
            display_hermes_home: "~/.hermes".to_string(),
            config: YamlValue::Mapping(Default::default()),
            defaults: YamlValue::Mapping(Default::default()),
            hermes_home: std::env::temp_dir().join("hermes_dump_keys_nonexistent"),
            gateway_status: "unknown".to_string(),
            show_keys: false,
        };
        let out = render_dump(&input);
        assert!(out.contains("  openai               set"));
        assert!(out.contains("  anthropic            not set"));
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }
    }
}
