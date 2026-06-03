//! `hermes slack ...` CLI subcommands.
//!
//! Native Rust port of `hermes_cli/slack_cli.py`.
//!
//! Today only `hermes slack manifest` is implemented — it generates the
//! Slack app manifest JSON for registering every gateway command as a
//! native Slack slash (`/btw`, `/stop`, `/model`, …) so users get the
//! same first-class slash UX Discord and Telegram already have.
//!
//! Typical workflow:
//!
//! ```text
//! $ hermes slack manifest > slack-manifest.json
//! # or:
//! $ hermes slack manifest --write
//! ```
//!
//! Then paste the printed JSON into the Slack app config (Features → App
//! Manifest → Edit) and click Save. Slack diffs the manifest and prompts
//! for reinstall when scopes/commands change.

use std::path::PathBuf;

use serde_json::{json, Value};

/// Default Slack `request_url`. In Socket Mode Slack ignores this URL and
/// routes the command event through the WebSocket, so a placeholder is fine.
pub const SLACK_REQUEST_URL: &str = "https://hermes-agent.local/slack/commands";

/// Default bot display name.
pub const SLACK_DEFAULT_NAME: &str = "Hermes";

/// Default bot description.
pub const SLACK_DEFAULT_DESCRIPTION: &str = "Your Hermes agent on Slack";

/// Slack app manifest accepts up to 50 slash commands per app.
pub const SLACK_MAX_SLASH_COMMANDS: usize = 50;

/// Slack slash command names are clamped to 32 characters.
pub const SLACK_NAME_LIMIT: usize = 32;

/// Built-in Slack slash commands that apps cannot register.
/// <https://slack.com/help/articles/201259356-Use-built-in-slash-commands>
pub const SLACK_RESERVED_COMMANDS: &[&str] = &[
    "me", "status", "away", "dnd", "shrug", "remind", "msg", "feed", "who", "collapse", "expand",
    "leave", "join", "open", "search", "topic", "mute", "pro", "shortcuts",
];

/// A single command definition as needed to build the Slack slash list.
///
/// This mirrors the fields of `hermes_cli.commands.CommandDef` consumed by
/// `slack_native_slashes`. Other ported modules expose a richer static
/// registry (`crate::commands::command_registry`); this struct lets callers
/// feed either that registry or a custom list without coupling to it.
#[derive(Debug, Clone)]
pub struct SlackCommandDef {
    pub name: String,
    pub description: String,
    pub aliases: Vec<String>,
    pub args_hint: String,
    /// Whether this command is CLI-only (not gateway-available unless gated).
    pub cli_only: bool,
    /// Dotted config path that, when truthy, makes a CLI-only command
    /// gateway-available.
    pub gateway_config_gate: Option<String>,
}

/// A resolved Slack slash command entry: `(name, description, usage_hint)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlackSlash {
    pub name: String,
    pub description: String,
    pub usage_hint: String,
}

/// Arguments for `hermes slack manifest`, mirroring the flags parsed in
/// `hermes_cli/main.py`.
#[derive(Debug, Clone, Default)]
pub struct SlackManifestArgs {
    /// Override the bot display name (default: "Hermes").
    pub name: Option<String>,
    /// Override the bot description.
    pub description: Option<String>,
    /// Emit only the `features.slash_commands` array.
    pub slashes_only: bool,
    /// `--write` target. `None` ⇒ stdout. `Some(WriteTarget::Default)` ⇒
    /// `$HERMES_HOME/slack-manifest.json`. `Some(WriteTarget::Path(p))` ⇒ `p`.
    pub write: Option<WriteTarget>,
}

/// Where `--write` should send the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteTarget {
    /// `--write` with no value → default `$HERMES_HOME/slack-manifest.json`.
    Default,
    /// `--write PATH` → the given (tilde-expanded) path.
    Path(String),
}

// ---------------------------------------------------------------------------
// Slash-command generation (port of `slack_native_slashes` / `slack_app_manifest`)
// ---------------------------------------------------------------------------

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

/// Convert a command name to a valid Slack slash command name.
///
/// Slack allows lowercase a-z, digits, hyphens, and underscores. Max 32
/// chars. Uppercase is lowercased; invalid chars are stripped; leading and
/// trailing `-`/`_` are trimmed.
///
/// Port of `hermes_cli.commands._sanitize_slack_name`.
pub fn sanitize_slack_name(raw: &str) -> String {
    let lowered: String = raw
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_'))
        .collect();
    let trimmed = lowered.trim_matches(|ch| ch == '-' || ch == '_');
    truncate_chars(trimmed, SLACK_NAME_LIMIT)
}

/// Look up a dotted config gate in a JSON config object and report whether
/// the value is truthy (mirrors Python truthiness for bool/number/string).
fn config_gate_truthy(config: &Value, gate: &str) -> bool {
    let mut node = config;
    for part in gate.split('.') {
        match node.get(part) {
            Some(value) => node = value,
            None => return false,
        }
    }
    json_truthy(node)
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(boolean) => *boolean,
        Value::Number(number) => number
            .as_i64()
            .map(|value| value != 0)
            .or_else(|| number.as_u64().map(|value| value != 0))
            .or_else(|| number.as_f64().map(|value| value != 0.0))
            .unwrap_or(false),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::Null => false,
    }
}

/// Whether a command is surfaced to gateways (and therefore to Slack).
///
/// Non-CLI-only commands are always available. CLI-only commands are
/// available only when their `gateway_config_gate` is truthy in `config`.
fn is_gateway_available(cmd: &SlackCommandDef, config: &Value) -> bool {
    if !cmd.cli_only {
        return true;
    }
    cmd.gateway_config_gate
        .as_deref()
        .is_some_and(|gate| config_gate_truthy(config, gate))
}

fn add_slack_entry(
    entries: &mut Vec<SlackSlash>,
    seen: &mut std::collections::HashSet<String>,
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

/// Return `(slash_name, description, usage_hint)` triples for Slack.
///
/// Port of `hermes_cli.commands.slack_native_slashes`. `/hermes` is always
/// reserved as the first entry. Canonical names take precedence over
/// aliases when the 50-command cap is hit. Reserved Slack built-ins and
/// duplicate names are skipped. Plugin commands are not included here
/// (callers append them separately if needed).
pub fn slack_native_slashes(commands: &[SlackCommandDef], config: &Value) -> Vec<SlackSlash> {
    let mut entries: Vec<SlackSlash> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Reserve /hermes as the catch-all top-level command.
    entries.push(SlackSlash {
        name: "hermes".to_string(),
        description: "Talk to Hermes or run a subcommand".to_string(),
        usage_hint: "[subcommand] [args]".to_string(),
    });
    seen.insert("hermes".to_string());

    // First pass: canonical names (so they win slots if we hit the cap).
    for cmd in commands {
        if !is_gateway_available(cmd, config) {
            continue;
        }
        add_slack_entry(&mut entries, &mut seen, &cmd.name, &cmd.description, &cmd.args_hint);
    }

    // Second pass: aliases.
    for cmd in commands {
        if !is_gateway_available(cmd, config) {
            continue;
        }
        for alias in &cmd.aliases {
            add_slack_entry(
                &mut entries,
                &mut seen,
                alias,
                &format!("Alias for /{} — {}", cmd.name, cmd.description),
                &cmd.args_hint,
            );
        }
    }

    entries
}

/// Render the `features.slash_commands` array from resolved slash entries.
///
/// Port of the JSON shape produced by `hermes_cli.commands.slack_app_manifest`.
pub fn slash_commands_json(slashes: &[SlackSlash], request_url: &str) -> Value {
    let array: Vec<Value> = slashes
        .iter()
        .map(|slash| {
            let mut entry = json!({
                "command": format!("/{}", slash.name),
                "description": if slash.description.is_empty() {
                    format!("Run /{}", slash.name)
                } else {
                    slash.description.clone()
                },
                "should_escape": false,
                "url": request_url,
            });
            if !slash.usage_hint.is_empty() {
                entry["usage_hint"] = Value::String(slash.usage_hint.clone());
            }
            entry
        })
        .collect();
    Value::Array(array)
}

// ---------------------------------------------------------------------------
// Manifest building (port of `_build_full_manifest`)
// ---------------------------------------------------------------------------

/// Build a full Slack manifest merging display info + the slash list.
///
/// Port of `slack_cli._build_full_manifest`. The slash list is always
/// generated from the command registry so it stays in sync with the rest of
/// Hermes; other sections use sensible Hermes defaults.
pub fn build_full_manifest(bot_name: &str, bot_description: &str, slashes: &[SlackSlash]) -> Value {
    let description = if bot_description.is_empty() {
        SLACK_DEFAULT_DESCRIPTION
    } else {
        bot_description
    };
    json!({
        "_metadata": {
            "major_version": 1,
            "minor_version": 1,
        },
        "display_information": {
            "name": truncate_chars(bot_name, 35),
            "description": truncate_chars(description, 140),
            "background_color": "#1a1a2e",
        },
        "features": {
            "bot_user": {
                "display_name": truncate_chars(bot_name, 80),
                "always_online": true,
            },
            "slash_commands": slash_commands_json(slashes, SLACK_REQUEST_URL),
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

// ---------------------------------------------------------------------------
// Command entry point (port of `slack_manifest_command`)
// ---------------------------------------------------------------------------

/// Outcome of `slack_manifest_command`: what to emit and the process exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlackManifestOutcome {
    /// JSON payload (already serialised, trailing newline included).
    pub payload: String,
    /// `Some(path)` when written to a file; `None` when emitted to stdout.
    pub written_to: Option<PathBuf>,
    /// Process exit code (always 0 in the Python original).
    pub exit_code: i32,
}

/// Tilde-expand a `~`/`~/...` path prefix using `$HOME`.
fn expanduser(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    } else if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

/// Resolve the default `$HERMES_HOME/slack-manifest.json` target.
///
/// Mirrors Python's fallback: prefer `get_hermes_home`, then `$HERMES_HOME`,
/// then `~/.hermes`.
fn default_manifest_target() -> PathBuf {
    let home = std::env::var("HERMES_HOME")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hermes")
        });
    home.join("slack-manifest.json")
}

/// Render the manifest payload (JSON + trailing newline) for the given args.
///
/// Port of the payload-construction half of `slack_manifest_command`. This is
/// I/O-free so it is easy to test and reuse.
pub fn render_manifest(args: &SlackManifestArgs, commands: &[SlackCommandDef], config: &Value) -> String {
    let name = args
        .name
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or(SLACK_DEFAULT_NAME);
    let description = args
        .description
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or(SLACK_DEFAULT_DESCRIPTION);

    let slashes = slack_native_slashes(commands, config);
    let manifest = if args.slashes_only {
        slash_commands_json(&slashes, SLACK_REQUEST_URL)
    } else {
        build_full_manifest(name, description, &slashes)
    };

    // serde_json's pretty printer uses 2-space indent and does not escape
    // non-ASCII (matching `ensure_ascii=False`). Append a trailing newline.
    let mut payload = serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".to_string());
    payload.push('\n');
    payload
}

/// Print or write a Slack app manifest JSON.
///
/// Faithful port of `slack_cli.slack_manifest_command`. Performs the same
/// filesystem and stderr/stdout side effects as the Python original and
/// returns the resulting [`SlackManifestOutcome`].
pub fn slack_manifest_command(
    args: &SlackManifestArgs,
    commands: &[SlackCommandDef],
    config: &Value,
) -> std::io::Result<SlackManifestOutcome> {
    let payload = render_manifest(args, commands, config);

    match &args.write {
        Some(target) => {
            let path = match target {
                WriteTarget::Default => default_manifest_target(),
                WriteTarget::Path(value) => expanduser(value),
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, payload.as_bytes())?;
            eprintln!("Slack manifest written to: {}", path.display());
            eprintln!(
                "\nNext steps:\n  1. Open https://api.slack.com/apps and pick your Hermes app\n     (or create a new one: Create New App → From an app manifest).\n  2. Features → App Manifest → paste the contents of\n     {}\n  3. Save; Slack will prompt to reinstall the app if scopes or\n     slash commands changed.\n  4. Make sure Socket Mode is enabled and you have a bot token\n     (xoxb-...) and app token (xapp-...) configured via\n     `hermes setup`.\n",
                path.display()
            );
            Ok(SlackManifestOutcome {
                payload,
                written_to: Some(path),
                exit_code: 0,
            })
        }
        None => {
            print!("{payload}");
            Ok(SlackManifestOutcome {
                payload,
                written_to: None,
                exit_code: 0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_commands() -> Vec<SlackCommandDef> {
        vec![
            SlackCommandDef {
                name: "btw".to_string(),
                description: "Send a background message".to_string(),
                aliases: vec!["background".to_string(), "bg".to_string()],
                args_hint: "<text>".to_string(),
                cli_only: false,
                gateway_config_gate: None,
            },
            SlackCommandDef {
                name: "status".to_string(),
                description: "Reserved name, should be skipped".to_string(),
                aliases: vec![],
                args_hint: String::new(),
                cli_only: false,
                gateway_config_gate: None,
            },
            SlackCommandDef {
                name: "secret".to_string(),
                description: "CLI-only gated command".to_string(),
                aliases: vec![],
                args_hint: String::new(),
                cli_only: true,
                gateway_config_gate: Some("features.secret_enabled".to_string()),
            },
        ]
    }

    #[test]
    fn sanitize_lowercases_and_strips() {
        assert_eq!(sanitize_slack_name("Model"), "model");
        assert_eq!(sanitize_slack_name("--BG--"), "bg");
        assert_eq!(sanitize_slack_name("a b!c"), "abc");
        assert_eq!(sanitize_slack_name("____"), "");
        assert_eq!(sanitize_slack_name(&"x".repeat(40)).len(), SLACK_NAME_LIMIT);
    }

    #[test]
    fn hermes_is_first_and_reserved_skipped() {
        let slashes = slack_native_slashes(&sample_commands(), &json!({}));
        assert_eq!(slashes[0].name, "hermes");
        assert!(slashes.iter().any(|s| s.name == "btw"));
        // canonical /status collides with a Slack built-in → skipped.
        assert!(!slashes.iter().any(|s| s.name == "status"));
        // aliases are surfaced.
        assert!(slashes.iter().any(|s| s.name == "background"));
        assert!(slashes.iter().any(|s| s.name == "bg"));
    }

    #[test]
    fn gated_command_hidden_until_enabled() {
        let off = slack_native_slashes(&sample_commands(), &json!({}));
        assert!(!off.iter().any(|s| s.name == "secret"));
        let on = slack_native_slashes(
            &sample_commands(),
            &json!({"features": {"secret_enabled": true}}),
        );
        assert!(on.iter().any(|s| s.name == "secret"));
    }

    #[test]
    fn slash_commands_json_shape() {
        let slashes = vec![SlackSlash {
            name: "btw".to_string(),
            description: "Send".to_string(),
            usage_hint: "<text>".to_string(),
        }];
        let value = slash_commands_json(&slashes, SLACK_REQUEST_URL);
        let first = &value[0];
        assert_eq!(first["command"], "/btw");
        assert_eq!(first["description"], "Send");
        assert_eq!(first["should_escape"], false);
        assert_eq!(first["url"], SLACK_REQUEST_URL);
        assert_eq!(first["usage_hint"], "<text>");
    }

    #[test]
    fn slash_json_default_description_when_empty() {
        let slashes = vec![SlackSlash {
            name: "x".to_string(),
            description: String::new(),
            usage_hint: String::new(),
        }];
        let value = slash_commands_json(&slashes, SLACK_REQUEST_URL);
        assert_eq!(value[0]["description"], "Run /x");
        assert!(value[0].get("usage_hint").is_none());
    }

    #[test]
    fn full_manifest_shape() {
        let slashes = slack_native_slashes(&sample_commands(), &json!({}));
        let manifest = build_full_manifest("Hermes", "Your Hermes agent on Slack", &slashes);
        assert_eq!(manifest["_metadata"]["major_version"], 1);
        assert_eq!(manifest["display_information"]["name"], "Hermes");
        assert_eq!(manifest["display_information"]["background_color"], "#1a1a2e");
        assert_eq!(manifest["settings"]["socket_mode_enabled"], true);
        assert_eq!(manifest["settings"]["org_deploy_enabled"], false);
        assert!(manifest["features"]["slash_commands"].is_array());
        let bot_scopes = &manifest["oauth_config"]["scopes"]["bot"];
        assert!(bot_scopes.as_array().unwrap().iter().any(|v| v == "commands"));
    }

    #[test]
    fn full_manifest_truncates_long_name() {
        let long = "n".repeat(100);
        let manifest = build_full_manifest(&long, "", &[]);
        assert_eq!(
            manifest["display_information"]["name"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            35
        );
        assert_eq!(
            manifest["features"]["bot_user"]["display_name"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            80
        );
        // empty description falls back to the default.
        assert_eq!(
            manifest["display_information"]["description"],
            SLACK_DEFAULT_DESCRIPTION
        );
    }

    #[test]
    fn render_slashes_only_emits_array() {
        let args = SlackManifestArgs {
            slashes_only: true,
            ..Default::default()
        };
        let payload = render_manifest(&args, &sample_commands(), &json!({}));
        let parsed: Value = serde_json::from_str(payload.trim()).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed[0]["command"], "/hermes");
        assert!(payload.ends_with('\n'));
    }

    #[test]
    fn render_full_uses_overrides() {
        let args = SlackManifestArgs {
            name: Some("MyBot".to_string()),
            description: Some("Custom".to_string()),
            ..Default::default()
        };
        let payload = render_manifest(&args, &sample_commands(), &json!({}));
        let parsed: Value = serde_json::from_str(payload.trim()).unwrap();
        assert_eq!(parsed["display_information"]["name"], "MyBot");
        assert_eq!(parsed["display_information"]["description"], "Custom");
    }

    #[test]
    fn render_full_default_name_when_empty() {
        let args = SlackManifestArgs::default();
        let payload = render_manifest(&args, &sample_commands(), &json!({}));
        let parsed: Value = serde_json::from_str(payload.trim()).unwrap();
        assert_eq!(parsed["display_information"]["name"], SLACK_DEFAULT_NAME);
    }

    #[test]
    fn write_to_explicit_path() {
        let dir = std::env::temp_dir().join(format!("hermes_slack_test_{}", std::process::id()));
        let target = dir.join("nested").join("manifest.json");
        let args = SlackManifestArgs {
            write: Some(WriteTarget::Path(target.to_string_lossy().to_string())),
            ..Default::default()
        };
        let outcome = slack_manifest_command(&args, &sample_commands(), &json!({})).unwrap();
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.written_to.as_deref(), Some(target.as_path()));
        let on_disk = std::fs::read_to_string(&target).unwrap();
        assert_eq!(on_disk, outcome.payload);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_target_honours_hermes_home() {
        let dir = std::env::temp_dir().join(format!("hermes_home_test_{}", std::process::id()));
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        let target = default_manifest_target();
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
        assert_eq!(target, dir.join("slack-manifest.json"));
    }

    #[test]
    fn expanduser_handles_tilde() {
        if let Some(home) = home_dir() {
            assert_eq!(expanduser("~"), home);
            assert_eq!(expanduser("~/foo"), home.join("foo"));
        }
        assert_eq!(expanduser("/abs/path"), PathBuf::from("/abs/path"));
    }
}
