use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Subcommand;
use hermes_core::{EnvLoadReport, HermesConfig, HermesContext, LoadedConfig, ModelOverrides};
use serde_yaml::{Mapping, Number, Sequence, Value};

const API_KEYS: &[(&str, &str)] = &[
    ("OPENROUTER_API_KEY", "OpenRouter"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("ANTHROPIC_TOKEN", "Anthropic token"),
    ("NOUS_API_KEY", "Nous"),
    ("GOOGLE_API_KEY", "Google/Gemini"),
    ("GEMINI_API_KEY", "Gemini"),
    ("GLM_API_KEY", "GLM/ZAI"),
    ("ZAI_API_KEY", "ZAI"),
    ("KIMI_API_KEY", "Kimi"),
    ("MINIMAX_API_KEY", "MiniMax"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
    ("DASHSCOPE_API_KEY", "DashScope"),
    ("HF_TOKEN", "Hugging Face"),
    ("AI_GATEWAY_API_KEY", "AI Gateway"),
    ("FIRECRAWL_API_KEY", "Firecrawl"),
    ("TAVILY_API_KEY", "Tavily"),
    ("BROWSERBASE_API_KEY", "Browserbase"),
    ("FAL_KEY", "FAL"),
    ("ELEVENLABS_API_KEY", "ElevenLabs"),
    ("TELEGRAM_BOT_TOKEN", "Telegram"),
    ("DISCORD_BOT_TOKEN", "Discord"),
    ("SLACK_BOT_TOKEN", "Slack"),
];
const CONFIG_TO_ENV_SYNC: &[(&str, &str)] = &[
    ("terminal.backend", "TERMINAL_ENV"),
    ("terminal.modal_mode", "TERMINAL_MODAL_MODE"),
    ("terminal.docker_image", "TERMINAL_DOCKER_IMAGE"),
    ("terminal.singularity_image", "TERMINAL_SINGULARITY_IMAGE"),
    ("terminal.modal_image", "TERMINAL_MODAL_IMAGE"),
    ("terminal.daytona_image", "TERMINAL_DAYTONA_IMAGE"),
    ("terminal.vercel_runtime", "TERMINAL_VERCEL_RUNTIME"),
    (
        "terminal.docker_mount_cwd_to_workspace",
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
    ),
    (
        "terminal.docker_run_as_host_user",
        "TERMINAL_DOCKER_RUN_AS_HOST_USER",
    ),
    ("terminal.timeout", "TERMINAL_TIMEOUT"),
    ("terminal.sandbox_dir", "TERMINAL_SANDBOX_DIR"),
    ("terminal.persistent_shell", "TERMINAL_PERSISTENT_SHELL"),
    ("terminal.container_cpu", "TERMINAL_CONTAINER_CPU"),
    ("terminal.container_memory", "TERMINAL_CONTAINER_MEMORY"),
    ("terminal.container_disk", "TERMINAL_CONTAINER_DISK"),
    (
        "terminal.container_persistent",
        "TERMINAL_CONTAINER_PERSISTENT",
    ),
];

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    Show,
    Edit,
    Check,
    Set {
        key: String,
        value: String,
    },
    Path,
    #[command(name = "env-path")]
    EnvPath,
}

pub fn print_config(
    context: &HermesContext,
    env_report: &EnvLoadReport,
    loaded: &LoadedConfig,
    command: Option<ConfigCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(ConfigCommand::Show) {
        ConfigCommand::Show => {
            println!("{}", render_config(context, loaded));
        }
        ConfigCommand::Edit => {
            edit_config(context)?;
        }
        ConfigCommand::Check => {
            println!("{}", render_config_check(context, env_report, loaded));
        }
        ConfigCommand::Set { key, value } => {
            set_config_value(context, &key, &value)?;
        }
        ConfigCommand::Path => {
            println!("{}", context.config_path().display());
        }
        ConfigCommand::EnvPath => {
            println!("{}", context.env_path().display());
        }
    }
    Ok(())
}

fn render_config(context: &HermesContext, loaded: &LoadedConfig) -> String {
    let install_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let (model, provider) = configured_model_and_provider(loaded);
    let mut lines = Vec::new();

    lines.push(String::from("=== Hermes Config ==="));
    lines.push(String::new());
    lines.push(String::from("Paths"));
    lines.push(format!(
        "  config:       {}",
        context.config_path().display()
    ));
    lines.push(format!("  secrets:      {}", context.env_path().display()));
    lines.push(format!("  install:      {}", install_root.display()));
    lines.push(String::new());
    lines.push(String::from("Model"));
    lines.push(format!("  model:        {model}"));
    lines.push(format!("  provider:     {provider}"));
    lines.push(format!(
        "  toolsets:     {}",
        if loaded.config.toolsets.is_empty() {
            "(default)".to_string()
        } else {
            loaded.config.toolsets.join(", ")
        }
    ));
    lines.push(format!("  max_turns:    {}", loaded.config.agent.max_turns));
    lines.push(String::new());
    lines.push(String::from("Display"));
    lines.push(format!("  skin:         {}", loaded.config.display.skin));
    lines.push(format!("  compact:      {}", loaded.config.display.compact));
    lines.push(format!(
        "  streaming:    {}",
        loaded.config.display.streaming
    ));
    lines.push(format!(
        "  language:     {}",
        loaded.config.display.language
    ));
    lines.push(String::new());
    lines.push(String::from("Terminal"));
    lines.push(format!(
        "  backend:      {}",
        loaded.config.terminal.backend
    ));
    lines.push(format!("  cwd:          {}", loaded.config.terminal.cwd));
    lines.push(format!(
        "  timeout:      {}s",
        loaded.config.terminal.timeout
    ));
    lines.push(String::new());
    lines.push(String::from("Runtime"));
    lines.push(format!("  logging:      {}", loaded.config.logging.level));
    lines.push(format!(
        "  memory:       {}",
        if loaded.config.memory.provider.trim().is_empty() {
            "built-in".to_string()
        } else {
            loaded.config.memory.provider.clone()
        }
    ));
    lines.push(format!(
        "  memory_on:    {}",
        loaded.config.memory.memory_enabled
    ));
    lines.push(format!(
        "  profile_on:   {}",
        loaded.config.memory.user_profile_enabled
    ));
    lines.push(format!(
        "  force_ipv4:   {}",
        loaded.config.network.force_ipv4
    ));
    lines.push(format!(
        "  redact:       {}",
        loaded.config.security.redact_secrets
    ));
    lines.push(String::new());
    lines.push(String::from("API Keys"));
    for (env_var, label) in API_KEYS {
        let display = std::env::var(env_var)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| redact_secret(&value))
            .unwrap_or_else(|| String::from("(not set)"));
        lines.push(format!("  {label:<13} {display}"));
    }
    lines.push(String::new());
    lines.push(String::from("Commands"));
    lines.push(String::from("  hermes config edit"));
    lines.push(String::from("  hermes config set <key> <value>"));
    lines.push(String::from("  hermes config check"));
    lines.push(String::from("  hermes config path"));
    lines.push(String::from("  hermes config env-path"));
    lines.join("\n")
}

fn render_config_check(
    context: &HermesContext,
    env_report: &EnvLoadReport,
    loaded: &LoadedConfig,
) -> String {
    let mut issues = Vec::new();
    let mut lines = Vec::new();
    let config_exists = context.config_path().exists();
    let env_exists = context.env_path().exists();
    let runtime = context.resolve_model_runtime(loaded, &ModelOverrides::default());
    let configured_keys = API_KEYS
        .iter()
        .filter(|(env_var, _)| {
            std::env::var(env_var)
                .ok()
                .is_some_and(|value| !value.trim().is_empty())
        })
        .count();

    lines.push(String::from("=== Hermes Config Check ==="));
    lines.push(String::new());
    lines.push(String::from("Files"));
    lines.push(format!(
        "  config:       {} ({})",
        status_label(config_exists, "present", "missing"),
        context.config_path().display()
    ));
    lines.push(format!(
        "  env:          {} ({})",
        status_label(env_exists, "present", "missing"),
        context.env_path().display()
    ));

    if !config_exists {
        issues.push(format!("missing {}", context.config_path().display()));
    }
    if !env_exists {
        issues.push(format!("missing {}", context.env_path().display()));
    }

    lines.push(String::new());
    lines.push(String::from("Warnings"));
    if loaded.warnings.is_empty() && env_report.warnings.is_empty() {
        lines.push(String::from("  status:       none"));
    } else {
        for warning in &loaded.warnings {
            lines.push(format!("  config:       {warning}"));
            issues.push(warning.clone());
        }
        for warning in &env_report.warnings {
            lines.push(format!("  env:          {warning}"));
            issues.push(warning.clone());
        }
    }

    lines.push(String::new());
    lines.push(String::from("Model Runtime"));
    match runtime {
        Ok(runtime) => {
            lines.push(String::from("  status:       ok"));
            lines.push(format!("  provider:     {}", runtime.provider));
            lines.push(format!("  model:        {}", runtime.model));
            lines.push(format!("  api_mode:     {}", runtime.api_mode));
        }
        Err(error) => {
            lines.push(String::from("  status:       fail"));
            lines.push(format!("  detail:       {error}"));
            issues.push(format!("model runtime resolution failed: {error}"));
        }
    }

    lines.push(String::new());
    lines.push(String::from("Credentials"));
    lines.push(format!(
        "  api_keys:     {configured_keys}/{}",
        API_KEYS.len()
    ));
    if configured_keys == 0 {
        lines.push(String::from("  note:         no known API keys are loaded"));
    }

    lines.push(String::new());
    lines.push(String::from("Summary"));
    if issues.is_empty() {
        lines.push(String::from("  status:       ok"));
        lines.push(String::from("  detail:       configuration looks healthy"));
    } else {
        lines.push(String::from("  status:       issues"));
        lines.push(format!("  count:        {}", issues.len()));
        for issue in issues {
            lines.push(format!("  - {issue}"));
        }
    }

    lines.join("\n")
}

fn configured_model_and_provider(loaded: &LoadedConfig) -> (String, String) {
    let model = loaded
        .configured_model_name()
        .unwrap_or_else(|| String::from("(not set)"));
    let provider = loaded
        .configured_model_provider()
        .unwrap_or_else(|| String::from("(auto)"));
    (model, provider)
}

fn redact_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::from("(not set)");
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    let prefix = chars[..4].iter().collect::<String>();
    let suffix = chars[chars.len() - 4..].iter().collect::<String>();
    format!("{prefix}...{suffix}")
}

fn edit_config(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    ensure_config_file(&context.config_path())?;
    let Some(command) = resolve_editor_command()? else {
        println!("No editor found. Config file is at:");
        println!("  {}", context.config_path().display());
        return Ok(());
    };

    let display = command.join(" ");
    println!(
        "Opening {} in {}...",
        context.config_path().display(),
        display
    );

    let mut process = Command::new(&command[0]);
    process.args(&command[1..]).arg(context.config_path());
    let status = process.status()?;
    if status.success() {
        return Ok(());
    }
    Err(format!("editor exited with status {:?}", status.code()).into())
}

fn ensure_config_file(path: &Path) -> Result<bool, Box<dyn Error>> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_yaml::to_string(&HermesConfig::default())?;
    atomic_write(path, rendered.as_bytes())?;
    Ok(true)
}

fn resolve_editor_command() -> Result<Option<Vec<String>>, Box<dyn Error>> {
    for key in ["EDITOR", "VISUAL"] {
        let Some(raw) = std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let parsed = shell_words::split(&raw)?;
        if parsed.is_empty() {
            continue;
        }
        if command_exists(&parsed[0]) || Path::new(&parsed[0]).is_file() {
            return Ok(Some(parsed));
        }
    }

    for candidate in ["nano", "vim", "vi", "code", "notepad"] {
        if command_exists(candidate) {
            return Ok(Some(vec![candidate.to_string()]));
        }
    }
    Ok(None)
}

fn command_exists(command: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| {
        let full = dir.join(command);
        full.is_file() || cfg!(windows) && full.with_extension("exe").is_file()
    })
}

fn status_label(ok: bool, ok_text: &str, missing_text: &str) -> String {
    if ok {
        ok_text.to_string()
    } else {
        missing_text.to_string()
    }
}

fn set_config_value(
    context: &HermesContext,
    key: &str,
    raw_value: &str,
) -> Result<(), Box<dyn Error>> {
    let trimmed_key = key.trim();
    if trimmed_key.is_empty() {
        return Err("config key cannot be empty".into());
    }
    if raw_value.contains('\n') || raw_value.contains('\r') {
        return Err("config value cannot contain newlines".into());
    }

    if is_env_key(trimmed_key) {
        save_env_value(context.env_path(), trimmed_key, raw_value)?;
        println!("set {} in {}", trimmed_key, context.env_path().display());
        return Ok(());
    }

    let mut user_config = read_raw_yaml_mapping(&context.config_path())?;
    let parsed = parse_scalar_value(raw_value);
    set_nested_value(&mut user_config, trimmed_key, parsed.clone())?;
    write_yaml_mapping(&context.config_path(), &user_config)?;

    if let Some((_, env_key)) = CONFIG_TO_ENV_SYNC
        .iter()
        .find(|(path, _)| *path == trimmed_key)
    {
        save_env_value(context.env_path(), env_key, &scalar_to_env_string(&parsed))?;
    }

    println!(
        "set {} = {} in {}",
        trimmed_key,
        display_yaml_value(&parsed),
        context.config_path().display()
    );
    Ok(())
}

fn is_env_key(key: &str) -> bool {
    if key.contains('.') || key.is_empty() {
        return false;
    }
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_uppercase()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_uppercase() || ch.is_ascii_digit())
}

fn parse_scalar_value(value: &str) -> Value {
    let lower = value.to_ascii_lowercase();
    if matches!(lower.as_str(), "true" | "yes" | "on") {
        return Value::Bool(true);
    }
    if matches!(lower.as_str(), "false" | "no" | "off") {
        return Value::Bool(false);
    }
    if let Ok(number) = value.parse::<i64>() {
        return Value::Number(Number::from(number));
    }
    if let Ok(number) = value.parse::<u64>() {
        return Value::Number(Number::from(number));
    }
    if let Ok(number) = value.parse::<f64>() {
        return Value::Number(Number::from(number));
    }
    Value::String(value.to_string())
}

fn scalar_to_env_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        _ => serde_yaml::to_string(value)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

fn display_yaml_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => scalar_to_env_string(value),
    }
}

pub(crate) fn read_raw_yaml_mapping(path: &Path) -> Result<Mapping, Box<dyn Error>> {
    if !path.exists() {
        return Ok(Mapping::new());
    }
    let text = fs::read_to_string(path)?;
    if text.trim().is_empty() {
        return Ok(Mapping::new());
    }
    let parsed = serde_yaml::from_str::<Value>(&text)?;
    match parsed {
        Value::Mapping(mapping) => Ok(mapping),
        Value::Null => Ok(Mapping::new()),
        _ => Err(format!("{} must contain a YAML mapping", path.display()).into()),
    }
}

pub(crate) fn write_yaml_mapping(path: &Path, mapping: &Mapping) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_yaml::to_string(mapping)?;
    atomic_write(path, rendered.as_bytes())
}

fn save_env_value(path: PathBuf, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
    let sanitized = sanitize_env_value(key, value)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut lines = if path.exists() {
        fs::read_to_string(&path)?
            .lines()
            .map(|line| format!("{line}\n"))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut found = false;
    for line in &mut lines {
        if line
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
        {
            *line = format!("{key}={sanitized}\n");
            found = true;
            break;
        }
    }
    if !found {
        lines.push(format!("{key}={sanitized}\n"));
    }
    let payload = lines.concat();
    atomic_write(&path, payload.as_bytes())
}

fn sanitize_env_value(key: &str, value: &str) -> Result<String, Box<dyn Error>> {
    if !is_env_key(key) {
        return Err(format!("invalid environment variable name: {key}").into());
    }
    let stripped = value.replace('\n', "").replace('\r', "");
    if stripped.is_ascii() {
        return Ok(stripped);
    }
    Ok(stripped
        .chars()
        .filter(|ch| ch.is_ascii())
        .collect::<String>())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp-{unique}"));
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn set_nested_value(
    root: &mut Mapping,
    dotted_key: &str,
    value: Value,
) -> Result<(), Box<dyn Error>> {
    let parts = dotted_key
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err("config key cannot be empty".into());
    }
    set_nested_in_mapping(root, &parts, value)
}

fn set_nested_in_mapping(
    mapping: &mut Mapping,
    parts: &[&str],
    value: Value,
) -> Result<(), Box<dyn Error>> {
    if parts.len() == 1 {
        mapping.insert(Value::String(parts[0].to_string()), value);
        return Ok(());
    }

    let key = Value::String(parts[0].to_string());
    let next_is_index = parts[1].parse::<usize>().is_ok();
    let existing = mapping.get_mut(&key);
    let child = match existing {
        Some(Value::Mapping(child)) => child,
        Some(Value::Sequence(sequence)) if next_is_index => {
            return set_nested_in_sequence(sequence, &parts[1..], value);
        }
        Some(other) => {
            *other = if next_is_index {
                Value::Sequence(Sequence::new())
            } else {
                Value::Mapping(Mapping::new())
            };
            match other {
                Value::Mapping(child) => child,
                Value::Sequence(sequence) => {
                    return set_nested_in_sequence(sequence, &parts[1..], value);
                }
                _ => unreachable!(),
            }
        }
        None => {
            mapping.insert(
                key.clone(),
                if next_is_index {
                    Value::Sequence(Sequence::new())
                } else {
                    Value::Mapping(Mapping::new())
                },
            );
            match mapping.get_mut(&key) {
                Some(Value::Mapping(child)) => child,
                Some(Value::Sequence(sequence)) => {
                    return set_nested_in_sequence(sequence, &parts[1..], value);
                }
                _ => unreachable!(),
            }
        }
    };
    set_nested_in_mapping(child, &parts[1..], value)
}

fn set_nested_in_sequence(
    sequence: &mut Sequence,
    parts: &[&str],
    value: Value,
) -> Result<(), Box<dyn Error>> {
    let index = parts[0]
        .parse::<usize>()
        .map_err(|_| format!("segment '{}' is not a numeric list index", parts[0]))?;
    ensure_sequence_len(sequence, index + 1);
    if parts.len() == 1 {
        sequence[index] = value;
        return Ok(());
    }

    let next_is_index = parts[1].parse::<usize>().is_ok();
    let entry = &mut sequence[index];
    match entry {
        Value::Mapping(child) if !next_is_index => set_nested_in_mapping(child, &parts[1..], value),
        Value::Sequence(child) if next_is_index => {
            set_nested_in_sequence(child, &parts[1..], value)
        }
        _ => {
            *entry = if next_is_index {
                Value::Sequence(Sequence::new())
            } else {
                Value::Mapping(Mapping::new())
            };
            match entry {
                Value::Mapping(child) => set_nested_in_mapping(child, &parts[1..], value),
                Value::Sequence(child) => set_nested_in_sequence(child, &parts[1..], value),
                _ => unreachable!(),
            }
        }
    }
}

fn ensure_sequence_len(sequence: &mut Sequence, len: usize) {
    while sequence.len() < len {
        sequence.push(Value::Null);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};

    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-config-{label}-{unique}"))
    }

    #[test]
    fn nested_setter_handles_mapping_paths() {
        let mut mapping = Mapping::new();
        set_nested_value(
            &mut mapping,
            "terminal.backend",
            Value::String("docker".to_string()),
        )
        .unwrap();
        let terminal = mapping
            .get(Value::String("terminal".to_string()))
            .and_then(Value::as_mapping)
            .unwrap();
        assert_eq!(
            terminal.get(Value::String("backend".to_string())),
            Some(&Value::String("docker".to_string()))
        );
    }

    #[test]
    fn nested_setter_handles_list_indices() {
        let mut mapping = Mapping::new();
        set_nested_value(
            &mut mapping,
            "providers.1.name",
            Value::String("openrouter".to_string()),
        )
        .unwrap();
        let providers = mapping
            .get(Value::String("providers".to_string()))
            .and_then(Value::as_sequence)
            .unwrap();
        let item = providers[1].as_mapping().unwrap();
        assert_eq!(
            item.get(Value::String("name".to_string())),
            Some(&Value::String("openrouter".to_string()))
        );
    }

    #[test]
    fn save_env_value_updates_existing_keys() {
        let path = temp_path("env");
        fs::write(&path, "OPENAI_API_KEY=old\nOTHER=value\n").unwrap();
        save_env_value(path.clone(), "OPENAI_API_KEY", "new").unwrap();
        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("OPENAI_API_KEY=new\n"));
        assert!(written.contains("OTHER=value\n"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn config_sync_writes_terminal_env_mirror() {
        let home = temp_path("home");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        set_config_value(&context, "terminal.backend", "docker").unwrap();
        let config = fs::read_to_string(home.join("config.yaml")).unwrap();
        let env_file = fs::read_to_string(home.join(".env")).unwrap();
        assert!(config.contains("backend: docker"));
        assert!(env_file.contains("TERMINAL_ENV=docker"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn edit_config_creates_defaults_and_launches_editor() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("edit-home");
        let fake_bin = home.join("bin");
        let fake_editor = fake_bin.join("editor");
        let log = home.join("editor.log");
        fs::create_dir_all(&fake_bin).unwrap();
        fs::write(
            &fake_editor,
            format!(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$1\" >> \"{}\"\nexit 0\n",
                log.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&fake_editor).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&fake_editor, perms).unwrap();
        }

        let original_path = env::var("PATH").unwrap_or_default();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        unsafe {
            env::set_var("PATH", format!("{}:{}", fake_bin.display(), original_path));
            env::set_var("EDITOR", "editor");
        }

        edit_config(&context).unwrap();

        let written = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(written.contains("toolsets:"));
        let opened = fs::read_to_string(log).unwrap();
        assert!(opened.contains(&home.join("config.yaml").display().to_string()));

        unsafe {
            env::remove_var("EDITOR");
            env::set_var("PATH", original_path);
        }
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn render_config_check_reports_missing_files() {
        let home = temp_path("check-home");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = LoadedConfig {
            path: context.config_path(),
            raw: Value::Mapping(Mapping::new()),
            config: HermesConfig::default(),
            warnings: Vec::new(),
        };
        let report = EnvLoadReport::default();

        let rendered = render_config_check(&context, &report, &loaded);
        assert!(rendered.contains("config:       missing"));
        assert!(rendered.contains("env:          missing"));
        assert!(rendered.contains("status:       issues"));

        let _ = fs::remove_dir_all(home);
    }
}
