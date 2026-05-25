use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use hermes_core::{HermesContext, LoadedConfig};
use regex::Regex;
use serde_json::{Value as JsonValue, json};
use serde_yaml::Value as YamlValue;

const SLACK_REQUEST_URL: &str = "https://hermes-agent.local/slack/commands";
const SLACK_DEFAULT_NAME: &str = "Hermes";
const SLACK_DEFAULT_DESCRIPTION: &str = "Your Hermes agent on Slack";
const SLACK_WRITE_DEFAULT: &str = "__DEFAULT__";
const SLACK_MAX_SLASH_COMMANDS: usize = 50;
const SLACK_NAME_LIMIT: usize = 32;
const SLACK_RESERVED_COMMANDS: &[&str] = &[
    "me",
    "status",
    "away",
    "dnd",
    "shrug",
    "remind",
    "msg",
    "feed",
    "who",
    "collapse",
    "expand",
    "leave",
    "join",
    "open",
    "search",
    "topic",
    "mute",
    "pro",
    "shortcuts",
];

#[derive(Subcommand, Debug)]
pub enum SlackCommand {
    Manifest(SlackManifestArgs),
}

#[derive(Args, Debug)]
pub struct SlackManifestArgs {
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = SLACK_WRITE_DEFAULT,
        value_name = "PATH"
    )]
    pub write: Option<String>,
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub description: Option<String>,
    #[arg(long = "slashes-only")]
    pub slashes_only: bool,
}

#[derive(Debug, Clone)]
struct RegistryCommand {
    name: String,
    description: String,
    aliases: Vec<String>,
    args_hint: String,
    cli_only: bool,
    gateway_config_gate: Option<String>,
}

#[derive(Debug, Clone)]
struct SlackSlash {
    name: String,
    description: String,
    usage_hint: String,
}

pub fn print_slack(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<SlackCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        Some(SlackCommand::Manifest(args)) => print_slack_manifest(context, loaded, args)?,
        None => {
            eprintln!(
                "usage: hermes slack <subcommand>\n\nsubcommands:\n  manifest   Generate a Slack app manifest with every gateway\n             command registered as a native slash\n\nRun `hermes slack manifest -h` for details."
            );
        }
    }
    Ok(())
}

fn print_slack_manifest(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: SlackManifestArgs,
) -> Result<(), Box<dyn Error>> {
    let name = args.name.as_deref().unwrap_or(SLACK_DEFAULT_NAME);
    let description = args
        .description
        .as_deref()
        .unwrap_or(SLACK_DEFAULT_DESCRIPTION);
    let slashes = slack_native_slashes(loaded)?;
    let payload = if args.slashes_only {
        slashes_only_manifest(&slashes)
    } else {
        full_manifest(name, description, &slashes)
    };
    let rendered = serde_json::to_string_pretty(&payload)? + "\n";

    match args.write.as_deref() {
        None => {
            print!("{rendered}");
        }
        Some(SLACK_WRITE_DEFAULT) => {
            let target = context.hermes_home().join("slack-manifest.json");
            write_manifest(&target, &rendered)?;
            print_write_hint(&target);
        }
        Some(path) => {
            let target = PathBuf::from(path).expand_home();
            write_manifest(&target, &rendered)?;
            print_write_hint(&target);
        }
    }
    Ok(())
}

pub(crate) fn write_default_slack_manifest(
    context: &HermesContext,
) -> Result<PathBuf, Box<dyn Error>> {
    let loaded = context.load_config_document()?;
    let slashes = slack_native_slashes(&loaded)?;
    let payload = full_manifest(SLACK_DEFAULT_NAME, SLACK_DEFAULT_DESCRIPTION, &slashes);
    let rendered = serde_json::to_string_pretty(&payload)? + "\n";
    let target = context.hermes_home().join("slack-manifest.json");
    write_manifest(&target, &rendered)?;
    Ok(target)
}

fn slashes_only_manifest(slashes: &[SlackSlash]) -> JsonValue {
    JsonValue::Array(
        slashes
            .iter()
            .map(|slash| {
                let mut entry = json!({
                    "command": format!("/{}", slash.name),
                    "description": slash.description,
                    "should_escape": false,
                    "url": SLACK_REQUEST_URL,
                });
                if !slash.usage_hint.is_empty() {
                    entry["usage_hint"] = JsonValue::String(slash.usage_hint.clone());
                }
                entry
            })
            .collect(),
    )
}

fn full_manifest(name: &str, description: &str, slashes: &[SlackSlash]) -> JsonValue {
    json!({
        "_metadata": {
            "major_version": 1,
            "minor_version": 1,
        },
        "display_information": {
            "name": truncate_chars(name, 35),
            "description": truncate_chars(description, 140),
            "background_color": "#1a1a2e",
        },
        "features": {
            "bot_user": {
                "display_name": truncate_chars(name, 80),
                "always_online": true,
            },
            "slash_commands": slashes_only_manifest(slashes),
            "assistant_view": {
                "assistant_description": "Chat with Hermes in threads and DMs.",
            },
        },
        "oauth_config": {
            "scopes": {
                "bot": [
                    "app_mentions:read",
                    "assistant:write",
                    "channels:history",
                    "channels:read",
                    "chat:write",
                    "commands",
                    "files:read",
                    "files:write",
                    "groups:history",
                    "im:history",
                    "im:read",
                    "im:write",
                    "users:read",
                ],
            },
        },
        "settings": {
            "event_subscriptions": {
                "bot_events": [
                    "app_mention",
                    "assistant_thread_context_changed",
                    "assistant_thread_started",
                    "message.channels",
                    "message.groups",
                    "message.im",
                ],
            },
            "interactivity": {
                "is_enabled": true,
            },
            "org_deploy_enabled": false,
            "socket_mode_enabled": true,
            "token_rotation_enabled": false,
        },
    })
}

fn slack_native_slashes(loaded: &LoadedConfig) -> Result<Vec<SlackSlash>, Box<dyn Error>> {
    let commands = parse_command_registry(&command_registry_path())?;
    let mut entries = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    entries.push(SlackSlash {
        name: "hermes".to_string(),
        description: "Talk to Hermes or run a subcommand".to_string(),
        usage_hint: "[subcommand] [args]".to_string(),
    });
    seen.insert("hermes".to_string());

    for command in &commands {
        if !gateway_available(command, &loaded.raw) {
            continue;
        }
        add_slack_entry(
            &mut entries,
            &mut seen,
            &command.name,
            &command.description,
            &command.args_hint,
        );
    }
    for command in &commands {
        if !gateway_available(command, &loaded.raw) {
            continue;
        }
        for alias in &command.aliases {
            add_slack_entry(
                &mut entries,
                &mut seen,
                alias,
                &format!("Alias for /{} — {}", command.name, command.description),
                &command.args_hint,
            );
        }
    }
    Ok(entries)
}

fn add_slack_entry(
    entries: &mut Vec<SlackSlash>,
    seen: &mut std::collections::BTreeSet<String>,
    name: &str,
    description: &str,
    usage_hint: &str,
) {
    let slack_name = sanitize_slack_name(name);
    if slack_name.is_empty()
        || seen.contains(&slack_name)
        || SLACK_RESERVED_COMMANDS.contains(&slack_name.as_str())
        || entries.len() >= SLACK_MAX_SLASH_COMMANDS
    {
        return;
    }
    entries.push(SlackSlash {
        name: slack_name.clone(),
        description: truncate_chars(description, 140),
        usage_hint: truncate_chars(usage_hint, 100),
    });
    seen.insert(slack_name);
}

fn gateway_available(command: &RegistryCommand, raw_config: &YamlValue) -> bool {
    if !command.cli_only {
        return true;
    }
    command
        .gateway_config_gate
        .as_deref()
        .is_some_and(|gate| config_gate_truthy(raw_config, gate))
}

fn config_gate_truthy(raw_config: &YamlValue, gate: &str) -> bool {
    let mut node = raw_config;
    for part in gate.split('.') {
        match node {
            YamlValue::Mapping(mapping) => {
                let key = YamlValue::String(part.to_string());
                let Some(value) = mapping.get(&key) else {
                    return false;
                };
                node = value;
            }
            _ => return false,
        }
    }
    yaml_truthy(node)
}

fn yaml_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Bool(boolean) => *boolean,
        YamlValue::Number(number) => number
            .as_i64()
            .map(|value| value != 0)
            .or_else(|| number.as_u64().map(|value| value != 0))
            .or_else(|| number.as_f64().map(|value| value != 0.0))
            .unwrap_or(false),
        YamlValue::String(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        _ => false,
    }
}

fn parse_command_registry(path: &Path) -> Result<Vec<RegistryCommand>, Box<dyn Error>> {
    let source = fs::read_to_string(path)?;
    let Some(start) = source.find("COMMAND_REGISTRY") else {
        return Err(format!("could not locate COMMAND_REGISTRY in {}", path.display()).into());
    };
    let blocks = extract_command_blocks(&source[start..]);
    let mut commands = Vec::new();
    for block in blocks {
        if let Some(command) = parse_command_block(&block)? {
            commands.push(command);
        }
    }
    Ok(commands)
}

fn extract_command_blocks(source: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    while let Some(relative) = source[offset..].find("CommandDef(") {
        let start = offset + relative;
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        let mut end = None;
        for (index, ch) in source[start..].char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        continue;
                    }
                    depth -= 1;
                    if depth == 0 {
                        end = Some(start + index + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            break;
        };
        blocks.push(source[start..end].to_string());
        offset = end;
    }
    blocks
}

fn parse_command_block(block: &str) -> Result<Option<RegistryCommand>, Box<dyn Error>> {
    let strings = string_literals(block);
    if strings.len() < 3 {
        return Ok(None);
    }
    let aliases = regex_captures(block, r#"aliases=\((?P<body>[^)]*)\)"#)
        .map(|body| string_literals(&body))
        .unwrap_or_default();
    Ok(Some(RegistryCommand {
        name: strings[0].clone(),
        description: strings[1].clone(),
        aliases,
        args_hint: regex_captures(block, r#"args_hint="(?P<value>[^"]*)""#).unwrap_or_default(),
        cli_only: block.contains("cli_only=True"),
        gateway_config_gate: regex_captures(block, r#"gateway_config_gate="(?P<value>[^"]+)""#),
    }))
}

fn regex_captures(text: &str, pattern: &str) -> Option<String> {
    let regex = Regex::new(pattern).ok()?;
    let captures = regex.captures(text)?;
    captures
        .name("body")
        .or_else(|| captures.name("value"))
        .map(|capture| capture.as_str().to_string())
}

fn string_literals(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            if escaped {
                current.push(ch);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                values.push(current.clone());
                current.clear();
                in_string = false;
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_string = true;
        }
    }
    values
}

fn command_registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("hermes_cli")
        .join("commands.py")
}

fn sanitize_slack_name(raw: &str) -> String {
    let mut name = raw
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_'))
        .collect::<String>();
    name = name.trim_matches(|ch| ch == '-' || ch == '_').to_string();
    truncate_chars(&name, SLACK_NAME_LIMIT)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn write_manifest(path: &Path, payload: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, payload)?;
    Ok(())
}

fn print_write_hint(path: &Path) {
    eprintln!("Slack manifest written to: {}", path.display());
    eprintln!(
        "\nNext steps:\n  1. Open https://api.slack.com/apps and pick your Hermes app\n     (or create a new one: Create New App -> From an app manifest).\n  2. Features -> App Manifest -> paste the contents of\n     {}\n  3. Save; Slack will prompt to reinstall the app if scopes or\n     slash commands changed.\n  4. Make sure Socket Mode is enabled and you have a bot token\n     (xoxb-...) and app token (xapp-...) configured via\n     `hermes setup`.\n",
        path.display()
    );
}

trait ExpandHome {
    fn expand_home(self) -> PathBuf;
}

impl ExpandHome for PathBuf {
    fn expand_home(self) -> PathBuf {
        let raw = self.to_string_lossy();
        if raw == "~" || raw.starts_with("~/") {
            if let Some(home) = dirs::home_dir() {
                return home.join(raw.trim_start_matches("~/"));
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::HermesContext;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_home(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-slack-{label}-{unique}"))
    }

    #[test]
    fn slack_registry_parsing_includes_gateway_aliases() {
        let home = temp_home("aliases");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = context.load_config_document().unwrap();
        let slashes = slack_native_slashes(&loaded).unwrap();
        assert!(slashes.iter().any(|slash| slash.name == "hermes"));
        assert!(slashes.iter().any(|slash| slash.name == "background"));
        assert!(slashes.iter().any(|slash| slash.name == "bg"));
        assert!(!slashes.iter().any(|slash| slash.name == "clear"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn slack_registry_honors_config_gates() {
        let home = temp_home("gates");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "display:\n  tool_progress_command: true\n",
        )
        .unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let loaded = context.load_config_document().unwrap();
        let slashes = slack_native_slashes(&loaded).unwrap();
        assert!(slashes.iter().any(|slash| slash.name == "verbose"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn slashes_only_manifest_contains_request_url() {
        let payload = slashes_only_manifest(&[SlackSlash {
            name: "hermes".to_string(),
            description: "Talk to Hermes".to_string(),
            usage_hint: "[subcommand]".to_string(),
        }]);
        let array = payload.as_array().unwrap();
        assert_eq!(array[0]["command"], "/hermes");
        assert_eq!(array[0]["url"], SLACK_REQUEST_URL);
    }
}
