use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Args;
use hermes_core::{HermesConfig, HermesContext, LoadedConfig};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping, Value};

const RELEASE_DATE: &str = "2026.4.30";
const API_KEYS: &[(&str, &str)] = &[
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
const PLATFORM_CHECKS: &[(&str, &str)] = &[
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
const INTERESTING_OVERRIDES: &[&[&str]] = &[
    &["agent", "max_turns"],
    &["agent", "gateway_timeout"],
    &["terminal", "backend"],
    &["terminal", "cwd"],
    &["terminal", "timeout"],
    &["display", "compact"],
    &["display", "streaming"],
    &["display", "skin"],
    &["display", "language"],
    &["logging", "level"],
    &["logging", "max_size_mb"],
    &["logging", "backup_count"],
    &["memory", "memory_enabled"],
    &["memory", "user_profile_enabled"],
    &["memory", "provider"],
    &["network", "force_ipv4"],
    &["security", "redact_secrets"],
    &["toolsets"],
];

#[derive(Args, Debug)]
pub struct DumpArgs {
    #[arg(long)]
    pub show_keys: bool,
}

pub fn print_version() {
    println!("{}", version_label());
}

pub fn print_dump(
    context: &HermesContext,
    config: &LoadedConfig,
    args: DumpArgs,
) -> Result<(), Box<dyn Error>> {
    println!("{}", render_dump(context, config, args.show_keys)?);
    Ok(())
}

fn render_dump(
    context: &HermesContext,
    config: &LoadedConfig,
    show_keys: bool,
) -> Result<String, Box<dyn Error>> {
    let (model, provider) = configured_model_and_provider(config);
    let default_config = serde_yaml::to_value(HermesConfig::default())?;
    let gateway_status = gateway_status(context.hermes_home());
    let configured_platforms = configured_platforms();
    let overrides = config_overrides(config, &default_config);
    let mut lines = Vec::new();

    lines.push(String::from("--- hermes dump ---"));
    lines.push(format!("version:          {}", version_label()));
    lines.push(format!(
        "os:               {} {} {}",
        std::env::consts::OS,
        os_release().unwrap_or_else(|| String::from("(unknown)")),
        std::env::consts::ARCH
    ));
    lines.push(format!("rust:             {}", rust_version()));
    lines.push(format!(
        "profile:          {}",
        context.current_profile_name()
    ));
    lines.push(format!(
        "hermes_home:      {}",
        context.display_hermes_home()
    ));
    lines.push(format!("model:            {model}"));
    lines.push(format!("provider:         {provider}"));
    lines.push(format!(
        "terminal:         {}",
        config.config.terminal.backend
    ));
    lines.push(String::new());
    lines.push(String::from("api_keys:"));
    for (env_var, label) in API_KEYS {
        let display = match std::env::var(env_var) {
            Ok(value) if !value.trim().is_empty() => {
                if show_keys {
                    redact_secret(&value)
                } else {
                    String::from("set")
                }
            }
            _ => String::from("not set"),
        };
        lines.push(format!("  {label:<20} {display}"));
    }

    lines.push(String::new());
    lines.push(String::from("features:"));
    lines.push(format!(
        "  toolsets:           {}",
        join_or_none(&config.config.toolsets)
    ));
    lines.push(format!(
        "  mcp_servers:        {}",
        count_mcp_servers(config.raw.as_mapping())
    ));
    lines.push(format!(
        "  memory_provider:    {}",
        if config.config.memory.provider.trim().is_empty() {
            "built-in"
        } else {
            config.config.memory.provider.as_str()
        }
    ));
    lines.push(format!("  gateway:            {gateway_status}"));
    lines.push(format!(
        "  platforms:          {}",
        join_or_label(&configured_platforms, "none")
    ));
    lines.push(format!(
        "  cron_jobs:          {}",
        cron_summary(&context.hermes_home().join("cron").join("jobs.json"))
    ));
    lines.push(format!(
        "  skills:             {}",
        count_skills(&context.skills_dir())
    ));

    if !overrides.is_empty() {
        lines.push(String::new());
        lines.push(String::from("config_overrides:"));
        for (path, value) in overrides {
            lines.push(format!("  {path}: {value}"));
        }
    }

    lines.push(String::from("--- end dump ---"));
    Ok(lines.join("\n"))
}

fn version_label() -> String {
    format!(
        "{} ({RELEASE_DATE}) [{}]",
        env!("CARGO_PKG_VERSION"),
        git_commit()
    )
}

fn git_commit() -> String {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    match Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .current_dir(repo_root)
        .output()
    {
        Ok(output) if output.status.success() => {
            let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if commit.is_empty() {
                String::from("(unknown)")
            } else {
                commit
            }
        }
        _ => String::from("(unknown)"),
    }
}

fn rust_version() -> String {
    match Command::new("rustc").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if version.is_empty() {
                String::from("(unknown)")
            } else {
                version
            }
        }
        _ => String::from("(unknown)"),
    }
}

fn os_release() -> Option<String> {
    let mut version_id = None;
    if let Ok(contents) = fs::read_to_string("/etc/os-release") {
        for line in contents.lines() {
            if let Some(value) = line.strip_prefix("PRETTY_NAME=") {
                return Some(value.trim_matches('"').to_string());
            }
            if let Some(value) = line.strip_prefix("VERSION_ID=") {
                version_id = Some(value.trim_matches('"').to_string());
            }
        }
    }
    version_id
}

fn configured_model_and_provider(config: &LoadedConfig) -> (String, String) {
    let model = config
        .configured_model_name()
        .unwrap_or_else(|| String::from("(not set)"));
    let provider = config
        .configured_model_provider()
        .unwrap_or_else(|| String::from("(auto)"));
    (model, provider)
}

fn redact_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    let prefix = chars[..4].iter().collect::<String>();
    let suffix = chars[chars.len() - 4..].iter().collect::<String>();
    format!("{prefix}...{suffix}")
}

fn count_mcp_servers(root: Option<&Mapping>) -> usize {
    let Some(root) = root else {
        return 0;
    };

    mapping_len(mapping_value(root, "mcp").and_then(|value| match value {
        Value::Mapping(mapping) => mapping_value(mapping, "servers").and_then(Value::as_mapping),
        _ => None,
    }))
    .or_else(|| mapping_len(mapping_value(root, "mcp_servers").and_then(Value::as_mapping)))
    .unwrap_or(0)
}

fn mapping_len(mapping: Option<&Mapping>) -> Option<usize> {
    mapping.map(Mapping::len)
}

fn cron_summary(path: &Path) -> String {
    let Ok(contents) = fs::read_to_string(path) else {
        return String::from("0");
    };
    let Ok(json) = serde_json::from_str::<JsonValue>(&contents) else {
        return String::from("(error reading)");
    };
    let Some(jobs) = json.get("jobs").and_then(JsonValue::as_array) else {
        return String::from("0");
    };
    let active = jobs
        .iter()
        .filter(|job| {
            job.get("enabled")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true)
        })
        .count();
    format!("{active} active / {} total", jobs.len())
}

fn count_skills(skills_dir: &Path) -> usize {
    let mut stack = vec![skills_dir.to_path_buf()];
    let mut count = 0_usize;
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
                continue;
            }
            if child.file_name().and_then(|value| value.to_str()) == Some("SKILL.md") {
                count += 1;
            }
        }
    }
    count
}

fn gateway_status(hermes_home: PathBuf) -> String {
    let path = hermes_home.join("gateway.pid");
    let Ok(raw) = fs::read_to_string(&path) else {
        return String::from("stopped");
    };
    let Ok(pid) = raw.trim().parse::<i32>() else {
        return String::from("unknown");
    };
    if process_exists(pid) {
        format!("running (manual, pid {pid})")
    } else {
        String::from("stopped (manual)")
    }
}

#[cfg(unix)]
fn process_exists(pid: i32) -> bool {
    let status = unsafe { libc::kill(pid, 0) };
    if status == 0 {
        return true;
    }
    matches!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::EPERM)
    )
}

#[cfg(not(unix))]
fn process_exists(_pid: i32) -> bool {
    false
}

fn configured_platforms() -> Vec<String> {
    PLATFORM_CHECKS
        .iter()
        .filter_map(|(platform, env_var)| {
            std::env::var(env_var)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .map(|_| (*platform).to_string())
        })
        .collect()
}

fn join_or_none(values: &[String]) -> String {
    join_or_label(values, "(default)")
}

fn join_or_label(values: &[String], fallback: &str) -> String {
    if values.is_empty() {
        fallback.to_string()
    } else {
        values.join(", ")
    }
}

fn config_overrides(config: &LoadedConfig, defaults: &Value) -> Vec<(String, String)> {
    let mut overrides = Vec::new();
    for path in INTERESTING_OVERRIDES {
        let Some(current) = value_at_path(&config.raw, path) else {
            continue;
        };
        let Some(default) = value_at_path(defaults, path) else {
            continue;
        };
        if current == default {
            continue;
        }
        overrides.push((path.join("."), format_value(current)));
    }
    overrides
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        match current {
            Value::Mapping(mapping) => {
                current = mapping.get(Value::String((*key).to_string()))?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        Value::Sequence(items) => items
            .iter()
            .map(format_value)
            .collect::<Vec<_>>()
            .join(", "),
        Value::Mapping(_) | Value::Tagged(_) => serde_yaml::to_string(value)
            .unwrap_or_else(|_| String::from("(unprintable)"))
            .trim()
            .replace('\n', " "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-dump-{label}-{unique}"))
    }

    #[test]
    fn redacts_secret_edges() {
        assert_eq!(redact_secret(""), "");
        assert_eq!(redact_secret("12345678"), "********");
        assert_eq!(redact_secret("abcdefghijk"), "abcd...hijk");
    }

    #[test]
    fn counts_mcp_servers_from_both_config_shapes() {
        let nested =
            serde_yaml::from_str::<Value>("mcp:\n  servers:\n    alpha: {}\n    beta: {}\n")
                .unwrap();
        assert_eq!(count_mcp_servers(nested.as_mapping()), 2);

        let legacy =
            serde_yaml::from_str::<Value>("mcp_servers:\n  alpha: {}\n  beta: {}\n  gamma: {}\n")
                .unwrap();
        assert_eq!(count_mcp_servers(legacy.as_mapping()), 3);
    }

    #[test]
    fn cron_summary_counts_enabled_jobs() {
        let path = temp_path("cron-jobs.json");
        fs::write(
            &path,
            r#"{"jobs":[{"id":"a","enabled":true},{"id":"b","enabled":false},{"id":"c"}]}"#,
        )
        .unwrap();
        assert_eq!(cron_summary(&path), "2 active / 3 total");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reports_non_default_config_overrides() {
        let loaded = LoadedConfig {
            path: PathBuf::from("/tmp/config.yaml"),
            raw: serde_yaml::from_str::<Value>(
                "toolsets:\n  - hermes-cli\n  - web\nterminal:\n  backend: docker\n",
            )
            .unwrap(),
            config: HermesConfig::default(),
            warnings: Vec::new(),
        };
        let defaults = serde_yaml::to_value(HermesConfig::default()).unwrap();
        let overrides = config_overrides(&loaded, &defaults);
        assert!(
            overrides
                .iter()
                .any(|(key, value)| key == "toolsets" && value == "hermes-cli, web")
        );
        assert!(
            overrides
                .iter()
                .any(|(key, value)| key == "terminal.backend" && value == "docker")
        );
    }
}
