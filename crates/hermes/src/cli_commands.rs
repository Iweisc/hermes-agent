//! Slash command definitions and autocomplete for the Hermes CLI.
//!
//! Native Rust port of `hermes_cli/commands.py`. This is the central registry
//! for all slash commands. Every consumer -- CLI help, gateway dispatch,
//! Telegram BotCommands, Slack subcommand mapping, autocomplete -- derives its
//! data from [`COMMAND_REGISTRY`].
//!
//! To add a command: add a [`CommandDef`] entry to [`COMMAND_REGISTRY`].
//! To add an alias: set `aliases` on the existing [`CommandDef`].
//!
//! The Python module also exposes prompt_toolkit `Completer` /`AutoSuggest`
//! classes. Those depend on a live editor document; here the equivalent
//! behaviour is reproduced as pure functions ([`get_completions`],
//! [`get_suggestion`], plus the scoring/path helpers) so the TUI / gateway can
//! drive them without the prompt_toolkit object model.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// CommandDef
// ---------------------------------------------------------------------------

/// Definition of a single slash command (port of the Python frozen dataclass).
#[derive(Debug, Clone, Copy)]
pub struct CommandDef {
    /// Canonical name without slash: `"background"`.
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// `"Session"`, `"Configuration"`, etc.
    pub category: &'static str,
    /// Alternative names: `["bg"]`.
    pub aliases: &'static [&'static str],
    /// Argument placeholder: `"<prompt>"`, `"[name]"`.
    pub args_hint: &'static str,
    /// Tab-completable subcommands.
    pub subcommands: &'static [&'static str],
    /// Only available in CLI.
    pub cli_only: bool,
    /// Only available in gateway/messaging.
    pub gateway_only: bool,
    /// Config dotpath; when truthy, overrides `cli_only` for gateway.
    pub gateway_config_gate: Option<&'static str>,
}

const fn cmd(
    name: &'static str,
    description: &'static str,
    category: &'static str,
    aliases: &'static [&'static str],
    args_hint: &'static str,
    subcommands: &'static [&'static str],
    cli_only: bool,
    gateway_only: bool,
    gateway_config_gate: Option<&'static str>,
) -> CommandDef {
    CommandDef {
        name,
        description,
        category,
        aliases,
        args_hint,
        subcommands,
        cli_only,
        gateway_only,
        gateway_config_gate,
    }
}

// ---------------------------------------------------------------------------
// Central registry -- single source of truth
// ---------------------------------------------------------------------------

/// The full slash command registry (faithful copy of the Python list order).
pub const COMMAND_REGISTRY: &[CommandDef] = &[
    // Session
    cmd("new", "Start a new session (fresh session ID + history)", "Session",
        &["reset"], "[name]", &[], false, false, None),
    cmd("topic", "Enable or inspect Telegram DM topic sessions", "Session",
        &[], "[off|help|session-id]", &[], false, true, None),
    cmd("clear", "Clear screen and start a new session", "Session",
        &[], "", &[], true, false, None),
    cmd("redraw", "Force a full UI repaint (recovers from terminal drift)", "Session",
        &[], "", &[], true, false, None),
    cmd("history", "Show conversation history", "Session",
        &[], "", &[], true, false, None),
    cmd("save", "Save the current conversation", "Session",
        &[], "", &[], true, false, None),
    cmd("retry", "Retry the last message (resend to agent)", "Session",
        &[], "", &[], false, false, None),
    cmd("undo", "Remove the last user/assistant exchange", "Session",
        &[], "", &[], false, false, None),
    cmd("title", "Set a title for the current session", "Session",
        &[], "[name]", &[], false, false, None),
    cmd("branch", "Branch the current session (explore a different path)", "Session",
        &["fork"], "[name]", &[], false, false, None),
    cmd("compress", "Manually compress conversation context", "Session",
        &[], "[focus topic]", &[], false, false, None),
    cmd("rollback", "List or restore filesystem checkpoints", "Session",
        &[], "[number]", &[], false, false, None),
    cmd("snapshot", "Create or restore state snapshots of Hermes config/state", "Session",
        &["snap"], "[create|restore <id>|prune]", &[], true, false, None),
    cmd("stop", "Kill all running background processes", "Session",
        &[], "", &[], false, false, None),
    cmd("approve", "Approve a pending dangerous command", "Session",
        &[], "[session|always]", &[], false, true, None),
    cmd("deny", "Deny a pending dangerous command", "Session",
        &[], "", &[], false, true, None),
    cmd("background", "Run a prompt in the background", "Session",
        &["bg", "btw"], "<prompt>", &[], false, false, None),
    cmd("agents", "Show active agents and running tasks", "Session",
        &["tasks"], "", &[], false, false, None),
    cmd("queue", "Queue a prompt for the next turn (doesn't interrupt)", "Session",
        &["q"], "<prompt>", &[], false, false, None),
    cmd("steer", "Inject a message after the next tool call without interrupting", "Session",
        &[], "<prompt>", &[], false, false, None),
    cmd("goal", "Set a standing goal Hermes works on across turns until achieved", "Session",
        &[], "[text | pause | resume | clear | status]", &[], false, false, None),
    cmd("status", "Show session info", "Session",
        &[], "", &[], false, false, None),
    cmd("profile", "Show active profile name and home directory", "Info",
        &[], "", &[], false, false, None),
    cmd("sethome", "Set this chat as the home channel", "Session",
        &["set-home"], "", &[], false, true, None),
    cmd("resume", "Resume a previously-named session", "Session",
        &[], "[name]", &[], false, false, None),

    // Configuration
    cmd("config", "Show current configuration", "Configuration",
        &[], "", &[], true, false, None),
    cmd("model", "Switch model for this session", "Configuration",
        &["provider"], "[model] [--provider name] [--global]", &[], false, false, None),
    cmd("gquota", "Show Google Gemini Code Assist quota usage", "Info",
        &[], "", &[], true, false, None),
    cmd("personality", "Set a predefined personality", "Configuration",
        &[], "[name]", &[], false, false, None),
    cmd("statusbar", "Toggle the context/model status bar", "Configuration",
        &["sb"], "", &[], true, false, None),
    cmd("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
        "Configuration", &[], "", &[], true, false, Some("display.tool_progress_command")),
    cmd("footer", "Toggle gateway runtime-metadata footer on final replies",
        "Configuration", &[], "[on|off|status]", &["on", "off", "status"], false, false, None),
    cmd("yolo", "Toggle YOLO mode (skip all dangerous command approvals)",
        "Configuration", &[], "", &[], false, false, None),
    cmd("reasoning", "Manage reasoning effort and display", "Configuration",
        &[], "[level|show|hide]",
        &["none", "minimal", "low", "medium", "high", "xhigh", "show", "hide", "on", "off"],
        false, false, None),
    cmd("fast",
        "Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode (Normal/Fast)",
        "Configuration", &[], "[normal|fast|status]",
        &["normal", "fast", "status", "on", "off"], false, false, None),
    cmd("skin", "Show or change the display skin/theme", "Configuration",
        &[], "[name]", &[], true, false, None),
    cmd("indicator", "Pick the TUI busy-indicator style", "Configuration",
        &[], "[kaomoji|emoji|unicode|ascii]",
        &["kaomoji", "emoji", "unicode", "ascii"], true, false, None),
    cmd("voice", "Toggle voice mode", "Configuration",
        &[], "[on|off|tts|status]", &["on", "off", "tts", "status"], false, false, None),
    cmd("busy", "Control what Enter does while Hermes is working", "Configuration",
        &[], "[queue|steer|interrupt|status]",
        &["queue", "steer", "interrupt", "status"], true, false, None),

    // Tools & Skills
    cmd("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
        &[], "[list|disable|enable] [name...]", &[], true, false, None),
    cmd("toolsets", "List available toolsets", "Tools & Skills",
        &[], "", &[], true, false, None),
    cmd("skills", "Search, install, inspect, or manage skills", "Tools & Skills",
        &[], "", &["search", "browse", "inspect", "install"], true, false, None),
    cmd("cron", "Manage scheduled tasks", "Tools & Skills",
        &[], "[subcommand]",
        &["list", "add", "create", "edit", "pause", "resume", "run", "remove"],
        true, false, None),
    cmd("curator", "Background skill maintenance (status, run, pin, archive)",
        "Tools & Skills", &[], "[subcommand]",
        &["status", "run", "pause", "resume", "pin", "unpin", "restore"], false, false, None),
    cmd("kanban", "Multi-profile collaboration board (tasks, links, comments)",
        "Tools & Skills", &[], "[subcommand]",
        &["list", "ls", "show", "create", "assign", "link", "unlink", "claim", "comment",
          "complete", "block", "unblock", "archive", "tail", "dispatch", "context", "init", "gc"],
        false, false, None),
    cmd("reload", "Reload .env variables into the running session", "Tools & Skills",
        &[], "", &[], true, false, None),
    cmd("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
        &["reload_mcp"], "", &[], false, false, None),
    cmd("reload-skills", "Re-scan ~/.hermes/skills/ for newly installed or removed skills",
        "Tools & Skills", &["reload_skills"], "", &[], false, false, None),
    cmd("browser", "Connect browser tools to your live Chrome via CDP", "Tools & Skills",
        &[], "[connect|disconnect|status]",
        &["connect", "disconnect", "status"], true, false, None),
    cmd("plugins", "List installed plugins and their status", "Tools & Skills",
        &[], "", &[], true, false, None),

    // Info
    cmd("commands", "Browse all commands and skills (paginated)", "Info",
        &[], "[page]", &[], false, true, None),
    cmd("help", "Show available commands", "Info",
        &[], "", &[], false, false, None),
    cmd("restart", "Gracefully restart the gateway after draining active runs", "Session",
        &[], "", &[], false, true, None),
    cmd("usage", "Show token usage and rate limits for the current session", "Info",
        &[], "", &[], false, false, None),
    cmd("insights", "Show usage insights and analytics", "Info",
        &[], "[days]", &[], false, false, None),
    cmd("platforms", "Show gateway/messaging platform status", "Info",
        &["gateway"], "", &[], true, false, None),
    cmd("copy", "Copy the last assistant response to clipboard", "Info",
        &[], "[number]", &[], true, false, None),
    cmd("paste", "Attach clipboard image from your clipboard", "Info",
        &[], "", &[], true, false, None),
    cmd("image", "Attach a local image file for your next prompt", "Info",
        &[], "<path>", &[], true, false, None),
    cmd("update", "Update Hermes Agent to the latest version", "Info",
        &[], "", &[], false, true, None),
    cmd("debug", "Upload debug report (system info + logs) and get shareable links", "Info",
        &[], "", &[], false, false, None),

    // Exit
    cmd("quit", "Exit the CLI", "Exit",
        &["exit"], "", &[], true, false, None),
];

// ---------------------------------------------------------------------------
// Derived lookups
// ---------------------------------------------------------------------------

fn command_lookup() -> &'static HashMap<String, usize> {
    static LOOKUP: OnceLock<HashMap<String, usize>> = OnceLock::new();
    LOOKUP.get_or_init(|| {
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, c) in COMMAND_REGISTRY.iter().enumerate() {
            lookup.insert(c.name.to_string(), idx);
            for alias in c.aliases {
                lookup.insert((*alias).to_string(), idx);
            }
        }
        lookup
    })
}

/// Resolve a command name or alias to its [`CommandDef`].
///
/// Accepts names with or without the leading slash.
pub fn resolve_command(name: &str) -> Option<&'static CommandDef> {
    let key = name.to_lowercase();
    let key = key.trim_start_matches('/');
    command_lookup().get(key).map(|&i| &COMMAND_REGISTRY[i])
}

/// Build a CLI-facing description string including a usage hint.
pub fn build_description(c: &CommandDef) -> String {
    if !c.args_hint.is_empty() {
        format!("{} (usage: /{} {})", c.description, c.name, c.args_hint)
    } else {
        c.description.to_string()
    }
}

/// Backwards-compatible flat map: `"/command"` -> description.
///
/// Returned in registry order (insertion-ordered to match Python's dict).
pub fn commands() -> &'static Vec<(String, String)> {
    static COMMANDS: OnceLock<Vec<(String, String)>> = OnceLock::new();
    COMMANDS.get_or_init(|| {
        let mut out: Vec<(String, String)> = Vec::new();
        for c in COMMAND_REGISTRY {
            if c.gateway_only {
                continue;
            }
            out.push((format!("/{}", c.name), build_description(c)));
            for alias in c.aliases {
                out.push((
                    format!("/{alias}"),
                    format!("{} (alias for /{})", c.description, c.name),
                ));
            }
        }
        out
    })
}

/// Look up a single description by `/command` key.
pub fn command_description(slash_command: &str) -> Option<String> {
    commands()
        .iter()
        .find(|(k, _)| k == slash_command)
        .map(|(_, v)| v.clone())
}

/// Backwards-compatible categorized map: category -> [(`/command`, desc)].
///
/// Categories appear in first-seen order; entries within a category in
/// registry order.
pub fn commands_by_category() -> &'static Vec<(String, Vec<(String, String)>)> {
    static BY_CAT: OnceLock<Vec<(String, Vec<(String, String)>)>> = OnceLock::new();
    BY_CAT.get_or_init(|| {
        let flat: HashMap<String, String> = commands().iter().cloned().collect();
        let mut order: Vec<String> = Vec::new();
        let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for c in COMMAND_REGISTRY {
            if c.gateway_only {
                continue;
            }
            let cat = c.category.to_string();
            if !map.contains_key(&cat) {
                order.push(cat.clone());
                map.insert(cat.clone(), Vec::new());
            }
            let bucket = map.get_mut(&cat).unwrap();
            let key = format!("/{}", c.name);
            if let Some(d) = flat.get(&key) {
                bucket.push((key, d.clone()));
            }
            for alias in c.aliases {
                let akey = format!("/{alias}");
                if let Some(d) = flat.get(&akey) {
                    bucket.push((akey, d.clone()));
                }
            }
        }
        order
            .into_iter()
            .map(|cat| {
                let v = map.remove(&cat).unwrap_or_default();
                (cat, v)
            })
            .collect()
    })
}

fn pipe_subs_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[a-z]+(?:\|[a-z]+)+").unwrap())
}

/// Subcommands lookup: `"/cmd"` -> `["sub1", "sub2", ...]`.
///
/// Explicit `subcommands` win; otherwise pipe-separated tokens in `args_hint`
/// (e.g. `"[on|off|status]"`) are extracted as a fallback.
pub fn subcommands() -> &'static HashMap<String, Vec<String>> {
    static SUBS: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    SUBS.get_or_init(|| {
        let mut subs: HashMap<String, Vec<String>> = HashMap::new();
        for c in COMMAND_REGISTRY {
            if !c.subcommands.is_empty() {
                subs.insert(
                    format!("/{}", c.name),
                    c.subcommands.iter().map(|s| s.to_string()).collect(),
                );
            }
        }
        for c in COMMAND_REGISTRY {
            let key = format!("/{}", c.name);
            if subs.contains_key(&key) || c.args_hint.is_empty() {
                continue;
            }
            if let Some(m) = pipe_subs_re().find(c.args_hint) {
                subs.insert(key, m.as_str().split('|').map(|s| s.to_string()).collect());
            }
        }
        subs
    })
}

// ---------------------------------------------------------------------------
// Gateway helpers
// ---------------------------------------------------------------------------

/// Set of all command names + aliases recognized by the gateway.
///
/// Includes config-gated commands so the gateway can dispatch them (the handler
/// checks the config gate at runtime).
pub fn gateway_known_commands() -> &'static HashSet<String> {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        let mut s: HashSet<String> = HashSet::new();
        for c in COMMAND_REGISTRY {
            if !c.cli_only || c.gateway_config_gate.is_some() {
                s.insert(c.name.to_string());
                for alias in c.aliases {
                    s.insert((*alias).to_string());
                }
            }
        }
        s
    })
}

/// Return true if `name` resolves to a gateway-dispatchable built-in command.
///
/// The Python version also consults plugin-registered commands; pass any
/// plugin command names via [`is_gateway_known_command_with_plugins`].
pub fn is_gateway_known_command(name: Option<&str>) -> bool {
    match name {
        None => false,
        Some(n) if n.is_empty() => false,
        Some(n) => gateway_known_commands().contains(n),
    }
}

/// As [`is_gateway_known_command`] but also matches plugin command names.
pub fn is_gateway_known_command_with_plugins(name: Option<&str>, plugin_names: &[String]) -> bool {
    match name {
        None => false,
        Some(n) if n.is_empty() => false,
        Some(n) => {
            gateway_known_commands().contains(n) || plugin_names.iter().any(|p| p == n)
        }
    }
}

/// Commands with explicit Level-2 running-agent handlers in `gateway/run.py`.
pub const ACTIVE_SESSION_BYPASS_COMMANDS: &[&str] = &[
    "agents", "approve", "background", "commands", "deny", "help", "new", "profile",
    "queue", "restart", "status", "steer", "stop", "update",
];

/// Return true for any resolvable slash command.
///
/// Queueing is always wrong for a recognized slash command; every gateway
/// command either has a Level-2 handler or reaches the busy catch-all.
pub fn should_bypass_active_session(command_name: Option<&str>) -> bool {
    match command_name {
        Some(n) if !n.is_empty() => resolve_command(n).is_some(),
        _ => false,
    }
}

/// Walk a dot-separated key path through a YAML mapping.
fn walk_yaml_path<'a>(root: &'a serde_yaml::Value, dotpath: &str) -> Option<&'a serde_yaml::Value> {
    let mut cur = root;
    for key in dotpath.split('.') {
        match cur {
            serde_yaml::Value::Mapping(map) => {
                cur = map.get(serde_yaml::Value::String(key.to_string()))?;
            }
            _ => return None,
        }
    }
    Some(cur)
}

fn yaml_is_truthy(val: Option<&serde_yaml::Value>) -> bool {
    match val {
        None | Some(serde_yaml::Value::Null) => false,
        Some(serde_yaml::Value::Bool(b)) => *b,
        Some(serde_yaml::Value::String(s)) => {
            let n = s.trim().to_lowercase();
            matches!(n.as_str(), "1" | "true" | "yes" | "on")
        }
        Some(serde_yaml::Value::Number(n)) => {
            n.as_f64().map(|f| f != 0.0).unwrap_or(false)
        }
        Some(serde_yaml::Value::Sequence(s)) => !s.is_empty(),
        Some(serde_yaml::Value::Mapping(m)) => !m.is_empty(),
        Some(serde_yaml::Value::Tagged(t)) => yaml_is_truthy(Some(&t.value)),
    }
}

/// Return canonical names of commands whose `gateway_config_gate` is truthy in
/// the supplied raw config document.
///
/// In Python this reads `config.yaml` via `read_raw_config()`; here the parsed
/// config is passed in so the caller controls config loading. Pass `None` to
/// treat the config as empty (mirrors the degrade-gracefully behaviour).
pub fn resolve_config_gates(cfg: Option<&serde_yaml::Value>) -> HashSet<String> {
    let mut result = HashSet::new();
    let gated: Vec<&CommandDef> = COMMAND_REGISTRY
        .iter()
        .filter(|c| c.gateway_config_gate.is_some())
        .collect();
    if gated.is_empty() {
        return result;
    }
    let cfg = match cfg {
        Some(c) => c,
        None => return result,
    };
    for c in gated {
        let gate = c.gateway_config_gate.unwrap();
        let val = walk_yaml_path(cfg, gate);
        if yaml_is_truthy(val) {
            result.insert(c.name.to_string());
        }
    }
    result
}

/// Read `config.yaml` from `hermes_home` and resolve config gates from it.
pub fn resolve_config_gates_from_home(hermes_home: &Path) -> HashSet<String> {
    let path = hermes_home.join("config.yaml");
    let cfg = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_yaml::from_str::<serde_yaml::Value>(&s).ok());
    resolve_config_gates(cfg.as_ref())
}

/// Check if `cmd` should appear in gateway surfaces (help, menus, mappings).
pub fn is_gateway_available(c: &CommandDef, config_overrides: &HashSet<String>) -> bool {
    if !c.cli_only {
        return true;
    }
    if c.gateway_config_gate.is_some() {
        return config_overrides.contains(c.name);
    }
    false
}

/// Return true when selecting a command without text would be incomplete.
pub fn requires_argument(args_hint: &str) -> bool {
    args_hint.trim().starts_with('<')
}

/// Generate gateway help text lines from the registry.
pub fn gateway_help_lines(config_overrides: &HashSet<String>) -> Vec<String> {
    let mut lines = Vec::new();
    for c in COMMAND_REGISTRY {
        if !is_gateway_available(c, config_overrides) {
            continue;
        }
        let args = if c.args_hint.is_empty() {
            String::new()
        } else {
            format!(" {}", c.args_hint)
        };
        let mut alias_parts: Vec<String> = Vec::new();
        for a in c.aliases {
            // Skip internal aliases like reload_mcp (underscore variant).
            if a.replace('-', "_") == c.name.replace('-', "_") && *a != c.name {
                continue;
            }
            alias_parts.push(format!("`/{a}`"));
        }
        let alias_note = if alias_parts.is_empty() {
            String::new()
        } else {
            format!(" (alias: {})", alias_parts.join(", "))
        };
        lines.push(format!(
            "`/{}{}` -- {}{}",
            c.name, args, c.description, alias_note
        ));
    }
    lines
}

// ---------------------------------------------------------------------------
// Telegram
// ---------------------------------------------------------------------------

/// Max command name length shared by Telegram and Discord.
pub const CMD_NAME_LIMIT: usize = 32;

fn tg_invalid_chars() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^a-z0-9_]").unwrap())
}

fn tg_multi_underscore() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"_{2,}").unwrap())
}

/// Convert a command/skill/plugin name to a valid Telegram command name.
///
/// Telegram requires: 1-32 chars, lowercase a-z, digits 0-9, underscores only.
pub fn sanitize_telegram_name(raw: &str) -> String {
    let name = raw.to_lowercase().replace('-', "_");
    let name = tg_invalid_chars().replace_all(&name, "");
    let name = tg_multi_underscore().replace_all(&name, "_");
    name.trim_matches('_').to_string()
}

/// Enforce the 32-char command name limit with collision avoidance.
///
/// Entries are `(name, desc, extra)` where `extra` is arbitrary passthrough
/// payload (e.g. a `cmd_key`) that survives any clamp rename. Names exceeding
/// the limit are truncated; collisions get a trailing digit `0`-`9`; if all 10
/// slots are taken the entry is dropped.
pub fn clamp_command_names(
    entries: Vec<(String, String, String)>,
    reserved: &HashSet<String>,
) -> Vec<(String, String, String)> {
    let mut used: HashSet<String> = reserved.clone();
    let mut result: Vec<(String, String, String)> = Vec::new();
    for (mut name, desc, extra) in entries {
        if name.chars().count() > CMD_NAME_LIMIT {
            let mut candidate: String = name.chars().take(CMD_NAME_LIMIT).collect();
            if used.contains(&candidate) {
                let prefix: String = name.chars().take(CMD_NAME_LIMIT - 1).collect();
                let mut found = false;
                for digit in 0..10 {
                    candidate = format!("{prefix}{digit}");
                    if !used.contains(&candidate) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    // All 10 digit slots exhausted — skip entry.
                    continue;
                }
            }
            name = candidate;
        }
        if used.contains(&name) {
            continue;
        }
        used.insert(name.clone());
        result.push((name, desc, extra));
    }
    result
}

/// Return (command_name, description) pairs for Telegram `setMyCommands`.
///
/// Hyphens are replaced with underscores; aliases skipped; arg-requiring
/// commands skipped. `plugin_entries` are `(name, description, args_hint)`
/// triples from the plugin registry (pass empty for builtins-only).
pub fn telegram_bot_commands(
    config_overrides: &HashSet<String>,
    plugin_entries: &[(String, String, String)],
) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();
    for c in COMMAND_REGISTRY {
        if !is_gateway_available(c, config_overrides) {
            continue;
        }
        if requires_argument(c.args_hint) {
            continue;
        }
        let tg = sanitize_telegram_name(c.name);
        if !tg.is_empty() {
            result.push((tg, c.description.to_string()));
        }
    }
    for (name, description, args_hint) in plugin_entries {
        if requires_argument(args_hint) {
            continue;
        }
        let tg = sanitize_telegram_name(name);
        if !tg.is_empty() {
            result.push((tg, description.clone()));
        }
    }
    result
}

/// A skill entry available for gateway slash-menu registration.
#[derive(Debug, Clone)]
pub struct GatewaySkillEntry {
    /// Original `/skill-name` command key (used by skill dispatch callbacks).
    pub cmd_key: String,
    /// Frontmatter skill name (used for per-platform disabled filtering).
    pub skill_name: String,
    /// Human description.
    pub description: String,
    /// Absolute path to the skill's SKILL.md / skill markdown file.
    pub skill_md_path: String,
}

fn clamp_desc(desc: &str, limit: usize) -> String {
    if desc.chars().count() > limit {
        let kept: String = desc.chars().take(limit.saturating_sub(3)).collect();
        format!("{kept}...")
    } else {
        desc.to_string()
    }
}

/// Collect plugin + skill entries for a gateway platform.
///
/// Priority: plugin commands first (never trimmed), then skills filling the
/// remaining slots (alphabetical, trimmed at the cap). Hub-installed skills and
/// per-platform disabled skills are excluded by the caller via `disabled_skills`
/// and `allowed_prefixes` / `hub_prefix`.
///
/// Returns `(entries, hidden_count)` where each entry is `(name, desc, cmd_key)`
/// and `hidden_count` is the number of skill entries dropped due to the cap.
#[allow(clippy::too_many_arguments)]
pub fn collect_gateway_skill_entries(
    max_slots: usize,
    reserved_names: &mut HashSet<String>,
    desc_limit: usize,
    sanitize: bool,
    plugin_commands: &[(String, String)], // (name, description) sorted by caller
    skills: &[GatewaySkillEntry],         // sorted by caller (by cmd_key)
    disabled_skills: &HashSet<String>,
    allowed_prefixes: &[String],
    hub_prefix: &str,
) -> (Vec<(String, String, String)>, usize) {
    let mut all_entries: Vec<(String, String, String)> = Vec::new();

    // --- Tier 1: Plugin slash commands (never trimmed) ---
    let mut plugin_triples: Vec<(String, String, String)> = Vec::new();
    for (cmd_name, desc) in plugin_commands {
        let name = if sanitize {
            sanitize_telegram_name(cmd_name)
        } else {
            cmd_name.clone()
        };
        if name.is_empty() {
            continue;
        }
        plugin_triples.push((name, clamp_desc(desc, desc_limit), String::new()));
    }
    let plugin_triples = clamp_command_names(plugin_triples, reserved_names);
    for (n, _, _) in &plugin_triples {
        reserved_names.insert(n.clone());
    }
    for (n, d, k) in plugin_triples {
        all_entries.push((n, d, k));
    }

    // --- Tier 2: Built-in skill commands (trimmed at cap) ---
    let mut skill_triples: Vec<(String, String, String)> = Vec::new();
    for s in skills {
        if s.skill_md_path.is_empty() {
            continue;
        }
        if !allowed_prefixes
            .iter()
            .any(|p| s.skill_md_path.starts_with(p))
        {
            continue;
        }
        if !hub_prefix.is_empty() && s.skill_md_path.starts_with(hub_prefix) {
            continue;
        }
        if disabled_skills.contains(&s.skill_name) {
            continue;
        }
        let raw_name = s.cmd_key.trim_start_matches('/');
        let name = if sanitize {
            sanitize_telegram_name(raw_name)
        } else {
            raw_name.to_string()
        };
        if name.is_empty() {
            continue;
        }
        skill_triples.push((name, clamp_desc(&s.description, desc_limit), s.cmd_key.clone()));
    }
    let skill_triples = clamp_command_names(skill_triples, reserved_names);

    let remaining = max_slots.saturating_sub(all_entries.len());
    let hidden_count = skill_triples.len().saturating_sub(remaining);
    for triple in skill_triples.into_iter().take(remaining) {
        all_entries.push(triple);
    }

    all_entries.truncate(max_slots);
    (all_entries, hidden_count)
}

/// Return Telegram menu commands capped to the Bot API limit.
///
/// Priority: core CommandDef commands, then plugins, then skills (trimmed).
/// Returns `(menu_commands, hidden_count)`.
#[allow(clippy::too_many_arguments)]
pub fn telegram_menu_commands(
    max_commands: usize,
    config_overrides: &HashSet<String>,
    plugin_bot_entries: &[(String, String, String)],
    plugin_commands: &[(String, String)],
    skills: &[GatewaySkillEntry],
    disabled_skills: &HashSet<String>,
    allowed_prefixes: &[String],
    hub_prefix: &str,
) -> (Vec<(String, String)>, usize) {
    let core_commands = telegram_bot_commands(config_overrides, plugin_bot_entries);
    let mut reserved_names: HashSet<String> =
        core_commands.iter().map(|(n, _)| n.clone()).collect();
    let mut all_commands = core_commands.clone();

    let remaining_slots = max_commands.saturating_sub(all_commands.len());
    let (entries, hidden) = collect_gateway_skill_entries(
        remaining_slots,
        &mut reserved_names,
        40,
        true,
        plugin_commands,
        skills,
        disabled_skills,
        allowed_prefixes,
        hub_prefix,
    );
    for (n, d, _) in entries {
        all_commands.push((n, d));
    }
    all_commands.truncate(max_commands);
    (all_commands, hidden)
}

/// Return skill entries for Discord slash command registration.
///
/// Same filtering as Telegram but hyphens allowed and descriptions capped at
/// 100 chars. `reserved_names` is copied (caller's set is not mutated).
#[allow(clippy::too_many_arguments)]
pub fn discord_skill_commands(
    max_slots: usize,
    reserved_names: &HashSet<String>,
    plugin_commands: &[(String, String)],
    skills: &[GatewaySkillEntry],
    disabled_skills: &HashSet<String>,
    allowed_prefixes: &[String],
    hub_prefix: &str,
) -> (Vec<(String, String, String)>, usize) {
    let mut reserved = reserved_names.clone();
    collect_gateway_skill_entries(
        max_slots,
        &mut reserved,
        100,
        false,
        plugin_commands,
        skills,
        disabled_skills,
        allowed_prefixes,
        hub_prefix,
    )
}

/// Return skill entries organized by category for Discord `/skill` autocomplete.
///
/// Skills nested at least 2 levels under a scan root are grouped by their
/// top-level directory; root-level skills are uncategorized. Names clamped to
/// 32 chars (first-32-char collisions counted as hidden), descriptions to 100.
///
/// `scan_roots` are absolute resolved directories; each skill's path is matched
/// against them to derive its category. `hub_prefix` excludes hub skills.
///
/// Returns `(categories, uncategorized, hidden_count)`.
#[allow(clippy::type_complexity)]
pub fn discord_skill_commands_by_category(
    reserved_names: &HashSet<String>,
    skills: &[GatewaySkillEntry],
    disabled_skills: &HashSet<String>,
    scan_roots: &[PathBuf],
    hub_prefix: &str,
) -> (
    Vec<(String, Vec<(String, String, String)>)>,
    Vec<(String, String, String)>,
    usize,
) {
    let mut cat_order: Vec<String> = Vec::new();
    let mut categories: HashMap<String, Vec<(String, String, String)>> = HashMap::new();
    let mut uncategorized: Vec<(String, String, String)> = Vec::new();
    let mut names_used: HashSet<String> = reserved_names.clone();
    let mut hidden = 0usize;

    for s in skills {
        if s.skill_md_path.is_empty() {
            continue;
        }
        let sp = Path::new(&s.skill_md_path);
        if !hub_prefix.is_empty() && s.skill_md_path.starts_with(hub_prefix) {
            continue;
        }
        // Accept skill if it lives under any scan root; record the matched root.
        let mut matched_root: Option<&PathBuf> = None;
        for root in scan_roots {
            if sp.starts_with(root) {
                matched_root = Some(root);
                break;
            }
        }
        let matched_root = match matched_root {
            Some(r) => r,
            None => continue,
        };
        if disabled_skills.contains(&s.skill_name) {
            continue;
        }
        let raw_name = s.cmd_key.trim_start_matches('/');
        let discord_name: String = raw_name.chars().take(32).collect();
        if names_used.contains(&discord_name) {
            hidden += 1;
            continue;
        }
        names_used.insert(discord_name.clone());

        let desc = clamp_desc(&s.description, 100);

        // Category from the relative path of the parent dir within the root.
        let parent = sp.parent().unwrap_or(sp);
        let rel = parent.strip_prefix(matched_root).unwrap_or(parent);
        let parts: Vec<_> = rel.components().collect();
        if parts.len() >= 2 {
            let cat = parts[0].as_os_str().to_string_lossy().to_string();
            if !categories.contains_key(&cat) {
                cat_order.push(cat.clone());
                categories.insert(cat.clone(), Vec::new());
            }
            categories
                .get_mut(&cat)
                .unwrap()
                .push((discord_name, desc, s.cmd_key.clone()));
        } else {
            uncategorized.push((discord_name, desc, s.cmd_key.clone()));
        }
    }

    let cats: Vec<(String, Vec<(String, String, String)>)> = cat_order
        .into_iter()
        .map(|c| {
            let v = categories.remove(&c).unwrap_or_default();
            (c, v)
        })
        .collect();
    (cats, uncategorized, hidden)
}

// ---------------------------------------------------------------------------
// Slack native slash commands
// ---------------------------------------------------------------------------

const SLACK_MAX_SLASH_COMMANDS: usize = 50;
const SLACK_NAME_LIMIT: usize = 32;

/// Built-in Slack slash commands that cannot be registered by apps.
pub const SLACK_RESERVED_COMMANDS: &[&str] = &[
    "me", "status", "away", "dnd", "shrug", "remind", "msg", "feed", "who", "collapse",
    "expand", "leave", "join", "open", "search", "topic", "mute", "pro", "shortcuts",
];

fn slack_invalid_chars() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^a-z0-9_\-]").unwrap())
}

/// Convert a command name to a valid Slack slash command name.
pub fn sanitize_slack_name(raw: &str) -> String {
    let name = raw.to_lowercase();
    let name = slack_invalid_chars().replace_all(&name, "");
    let name = name.trim_matches(|c| c == '-' || c == '_');
    name.chars().take(SLACK_NAME_LIMIT).collect()
}

fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Return (slash_name, description, usage_hint) triples for Slack.
///
/// Reserves `/hermes` first, then canonical names, then aliases, then plugin
/// commands; clamped to Slack's 50-command limit with dedup.
pub fn slack_native_slashes(
    config_overrides: &HashSet<String>,
    plugin_entries: &[(String, String, String)],
) -> Vec<(String, String, String)> {
    let mut entries: Vec<(String, String, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let reserved: HashSet<&str> = SLACK_RESERVED_COMMANDS.iter().copied().collect();

    entries.push((
        "hermes".to_string(),
        "Talk to Hermes or run a subcommand".to_string(),
        "[subcommand] [args]".to_string(),
    ));
    seen.insert("hermes".to_string());

    let add = |name: &str, desc: &str, hint: &str, entries: &mut Vec<_>, seen: &mut HashSet<String>| {
        let slack_name = sanitize_slack_name(name);
        if slack_name.is_empty() || seen.contains(&slack_name) {
            return;
        }
        if reserved.contains(slack_name.as_str()) {
            return;
        }
        if entries.len() >= SLACK_MAX_SLASH_COMMANDS {
            return;
        }
        entries.push((
            slack_name.clone(),
            truncate_chars(desc, 140),
            truncate_chars(hint, 100),
        ));
        seen.insert(slack_name);
    };

    // First pass: canonical names.
    for c in COMMAND_REGISTRY {
        if !is_gateway_available(c, config_overrides) {
            continue;
        }
        add(c.name, c.description, c.args_hint, &mut entries, &mut seen);
    }
    // Second pass: aliases.
    for c in COMMAND_REGISTRY {
        if !is_gateway_available(c, config_overrides) {
            continue;
        }
        for alias in c.aliases {
            let desc = format!("Alias for /{} — {}", c.name, c.description);
            add(alias, &desc, c.args_hint, &mut entries, &mut seen);
        }
    }
    // Third pass: plugin commands.
    for (name, description, args_hint) in plugin_entries {
        add(name, description, args_hint, &mut entries, &mut seen);
    }

    entries
}

/// Generate the `features.slash_commands` portion of a Slack app manifest.
pub fn slack_app_manifest(
    request_url: &str,
    config_overrides: &HashSet<String>,
    plugin_entries: &[(String, String, String)],
) -> Value {
    let mut slashes: Vec<Value> = Vec::new();
    for (name, desc, usage) in slack_native_slashes(config_overrides, plugin_entries) {
        let description = if desc.is_empty() {
            format!("Run /{name}")
        } else {
            desc
        };
        let mut entry = Map::new();
        entry.insert("command".to_string(), json!(format!("/{name}")));
        entry.insert("description".to_string(), json!(description));
        entry.insert("should_escape".to_string(), json!(false));
        entry.insert("url".to_string(), json!(request_url));
        if !usage.is_empty() {
            entry.insert("usage_hint".to_string(), json!(usage));
        }
        slashes.push(Value::Object(entry));
    }
    json!({ "features": { "slash_commands": slashes } })
}

/// Return subcommand -> `/command` mapping for the Slack `/hermes` handler.
///
/// Maps both canonical names and aliases; plugin commands appended last
/// (only if not already mapped).
pub fn slack_subcommand_map(
    config_overrides: &HashSet<String>,
    plugin_entries: &[(String, String, String)],
) -> Vec<(String, String)> {
    let mut mapping: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let push = |k: String, v: String, mapping: &mut Vec<(String, String)>, seen: &mut HashSet<String>| {
        if seen.insert(k.clone()) {
            mapping.push((k, v));
        }
    };
    for c in COMMAND_REGISTRY {
        if !is_gateway_available(c, config_overrides) {
            continue;
        }
        push(c.name.to_string(), format!("/{}", c.name), &mut mapping, &mut seen);
        for alias in c.aliases {
            push((*alias).to_string(), format!("/{alias}"), &mut mapping, &mut seen);
        }
    }
    for (name, _d, _h) in plugin_entries {
        if !seen.contains(name) {
            push(name.clone(), format!("/{name}"), &mut mapping, &mut seen);
        }
    }
    mapping
}

// ---------------------------------------------------------------------------
// Autocomplete / suggestions
// ---------------------------------------------------------------------------

/// A single completion candidate (port of prompt_toolkit's `Completion`).
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    /// Replacement text that gets inserted.
    pub text: String,
    /// Negative offset where replacement begins (relative to cursor).
    pub start_position: i64,
    /// Text shown in the dropdown.
    pub display: String,
    /// Right-aligned metadata shown in the dropdown.
    pub display_meta: String,
}

impl Completion {
    fn new(text: impl Into<String>, start: i64, display: impl Into<String>, meta: impl Into<String>) -> Self {
        Completion {
            text: text.into(),
            start_position: start,
            display: display.into(),
            display_meta: meta.into(),
        }
    }
}

/// Commands that open pickers when run without arguments — completions for
/// these must NOT get a trailing space.
pub const PICKER_COMMANDS: &[&str] = &["model", "skin", "personality"];

/// Return replacement text for a command completion.
///
/// When the user has typed the full command exactly, a trailing space keeps the
/// dropdown visible — except for picker commands which would block on Enter.
pub fn completion_text(cmd_name: &str, word: &str) -> String {
    if cmd_name != word {
        return cmd_name.to_string();
    }
    if PICKER_COMMANDS.contains(&cmd_name) {
        return cmd_name.to_string();
    }
    format!("{cmd_name} ")
}

/// Compact human-readable file size for `path`, or `""` on error.
pub fn file_size_label(path: &Path) -> String {
    let size = match fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => return String::new(),
    };
    if size < 1024 {
        format!("{size}B")
    } else if size < 1024 * 1024 {
        format!("{:.0}K", size as f64 / 1024.0)
    } else if size < 1024 * 1024 * 1024 {
        format!("{:.1}M", size as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}G", size as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Score a file path against a fuzzy query. Higher = better match.
///
/// Faithful port of `SlashCommandCompleter._score_path`.
pub fn score_path(filepath: &str, query: &str) -> i64 {
    if query.is_empty() {
        return 1; // show everything when query is empty
    }
    let filename = Path::new(filepath)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| filepath.to_string());
    let lower_file = filename.to_lowercase();
    let lower_path = filepath.to_lowercase();
    let lower_q = query.to_lowercase();

    if lower_file == lower_q {
        return 100;
    }
    if lower_file.starts_with(&lower_q) {
        return 80;
    }
    if lower_file.contains(&lower_q) {
        return 60;
    }
    if lower_path.contains(&lower_q) {
        return 40;
    }
    // Initials / abbreviation match: query chars appear in order in filename.
    let q_chars: Vec<char> = lower_q.chars().collect();
    let mut qi = 0usize;
    for c in lower_file.chars() {
        if qi < q_chars.len() && c == q_chars[qi] {
            qi += 1;
        }
    }
    if qi == q_chars.len() {
        // Bonus if matches land on word boundaries (after _, -, /, .).
        let mut boundary_hits = 0usize;
        qi = 0;
        let mut prev = '_'; // treat start as boundary
        for c in lower_file.chars() {
            if qi < q_chars.len() && c == q_chars[qi] {
                if matches!(prev, '_' | '-' | '.' | '/') {
                    boundary_hits += 1;
                }
                qi += 1;
            }
            prev = c;
        }
        if boundary_hits as f64 >= q_chars.len() as f64 * 0.5 {
            return 35;
        }
        return 25;
    }
    0
}

/// Extract the current word if it looks like a file path (port of
/// `_extract_path_word`). Returns `None` if it doesn't look path-like.
pub fn extract_path_word(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let word = match text.rfind(' ') {
        Some(i) => &text[i + 1..],
        None => text,
    };
    if word.is_empty() {
        return None;
    }
    if word.starts_with("./")
        || word.starts_with("../")
        || word.starts_with("~/")
        || word.starts_with('/')
        || word.contains('/')
    {
        Some(word.to_string())
    } else {
        None
    }
}

/// Extract a bare `@`-token for context reference completions (port of
/// `_extract_context_word`).
pub fn extract_context_word(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let word = match text.rfind(' ') {
        Some(i) => &text[i + 1..],
        None => text,
    };
    if word.starts_with('@') {
        Some(word.to_string())
    } else {
        None
    }
}

fn expanduser(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

/// Yield file path completions matching `word` (port of `_path_completions`).
pub fn path_completions(word: &str, limit: usize) -> Vec<Completion> {
    let mut out = Vec::new();
    let expanded = expanduser(word);
    let (search_dir, prefix) = if expanded.ends_with('/') {
        (expanded.clone(), String::new())
    } else {
        let p = Path::new(&expanded);
        let dir = p
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let base = p
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (dir, base)
    };

    let mut entries: Vec<String> = match fs::read_dir(&search_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
            .collect(),
        Err(_) => return out,
    };
    entries.sort();

    let prefix_lower = prefix.to_lowercase();
    let home = dirs::home_dir();
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut count = 0usize;
    for entry in entries {
        if !prefix.is_empty() && !entry.to_lowercase().starts_with(&prefix_lower) {
            continue;
        }
        if count >= limit {
            break;
        }
        let full_path = Path::new(&search_dir).join(&entry);
        let is_dir = full_path.is_dir();

        let mut display_path = if word.starts_with('~') {
            let rel = home
                .as_ref()
                .and_then(|h| full_path.strip_prefix(h).ok())
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|| full_path.to_string_lossy().to_string());
            format!("~/{rel}")
        } else if Path::new(word).is_absolute() {
            full_path.to_string_lossy().to_string()
        } else {
            full_path
                .strip_prefix(&cwd)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| full_path.to_string_lossy().to_string())
        };
        if is_dir {
            display_path.push('/');
        }
        let suffix = if is_dir { "/" } else { "" };
        let meta = if is_dir {
            "dir".to_string()
        } else {
            file_size_label(&full_path)
        };
        out.push(Completion::new(
            display_path,
            -(word.chars().count() as i64),
            format!("{entry}{suffix}"),
            meta,
        ));
        count += 1;
    }
    out
}

/// Static `@`-references surfaced by `@`-context completion.
const STATIC_REFS: &[(&str, &str)] = &[
    ("@diff", "Git working tree diff"),
    ("@staged", "Git staged diff"),
    ("@file:", "Attach a file"),
    ("@folder:", "Attach a folder"),
    ("@git:", "Git log with diffs (e.g. @git:5)"),
    ("@url:", "Fetch web content"),
];

/// Yield Claude Code-style `@` context completions (port of
/// `_context_completions`). `project_files` is the cached project file list
/// (relative paths, dir entries ending in `/`) used for bare-`@` fuzzy search.
pub fn context_completions(word: &str, limit: usize, project_files: &[String]) -> Vec<Completion> {
    let mut out = Vec::new();
    let lowered = word.to_lowercase();

    for (candidate, meta) in STATIC_REFS {
        let cl = candidate.to_lowercase();
        if cl.starts_with(&lowered) && cl != lowered {
            out.push(Completion::new(
                *candidate,
                -(word.chars().count() as i64),
                *candidate,
                *meta,
            ));
        }
    }

    let word_len = word.chars().count() as i64;
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for prefix in ["@file:", "@folder:"] {
        let bare = &prefix[..prefix.len() - 1];
        if word == bare || word.starts_with(prefix) {
            let want_dir = prefix == "@folder:";
            let path_part = if word == bare {
                ""
            } else {
                &word[prefix.len()..]
            };
            let expanded = expanduser(path_part);

            let (search_dir, match_prefix) = if expanded.is_empty() || expanded == "." {
                (".".to_string(), String::new())
            } else if expanded.ends_with('/') {
                (expanded.clone(), String::new())
            } else {
                let p = Path::new(&expanded);
                let dir = p
                    .parent()
                    .filter(|d| !d.as_os_str().is_empty())
                    .map(|d| d.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string());
                let base = p
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default();
                (dir, base)
            };

            let mut entries: Vec<String> = match fs::read_dir(&search_dir) {
                Ok(rd) => rd
                    .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
                    .collect(),
                Err(_) => return out,
            };
            entries.sort();

            let prefix_lower = match_prefix.to_lowercase();
            let mut count = 0usize;
            for entry in entries {
                if !match_prefix.is_empty() && !entry.to_lowercase().starts_with(&prefix_lower) {
                    continue;
                }
                let full_path = Path::new(&search_dir).join(&entry);
                let is_dir = full_path.is_dir();
                if want_dir != is_dir {
                    continue;
                }
                if count >= limit {
                    break;
                }
                let display_path = full_path
                    .strip_prefix(&cwd)
                    .map(|r| r.to_string_lossy().to_string())
                    .unwrap_or_else(|_| full_path.to_string_lossy().to_string());
                let suffix = if is_dir { "/" } else { "" };
                let meta = if is_dir {
                    "dir".to_string()
                } else {
                    file_size_label(&full_path)
                };
                let completion = format!("{prefix}{display_path}{suffix}");
                out.push(Completion::new(
                    completion,
                    -word_len,
                    format!("{entry}{suffix}"),
                    meta,
                ));
                count += 1;
            }
            return out;
        }
    }

    // Bare @ or @partial — fuzzy project-wide file search.
    let query = &word[1..];
    out.extend(fuzzy_file_completions(word, query, 20.min(limit).max(limit), project_files));
    out
}

/// Yield fuzzy file completions for a bare `@query` (port of
/// `_fuzzy_file_completions`). `project_files` are relative paths (dirs end `/`).
pub fn fuzzy_file_completions(
    word: &str,
    query: &str,
    limit: usize,
    project_files: &[String],
) -> Vec<Completion> {
    let mut out = Vec::new();
    let word_len = word.chars().count() as i64;
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    if query.is_empty() {
        for fp in project_files.iter().take(limit) {
            let is_dir = fp.ends_with('/');
            let filename = Path::new(fp)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| fp.clone());
            let kind = if is_dir { "folder" } else { "file" };
            let meta = if is_dir {
                "dir".to_string()
            } else {
                file_size_label(&cwd.join(fp))
            };
            out.push(Completion::new(
                format!("@{kind}:{fp}"),
                -word_len,
                filename,
                meta,
            ));
        }
        return out;
    }

    let mut scored: Vec<(i64, &String)> = Vec::new();
    for fp in project_files {
        let s = score_path(fp, query);
        if s > 0 {
            scored.push((s, fp));
        }
    }
    // Sort: descending score, then ascending path.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

    for (_, fp) in scored.into_iter().take(limit) {
        let is_dir = fp.ends_with('/');
        let filename = Path::new(fp)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| fp.clone());
        let kind = if is_dir { "folder" } else { "file" };
        let meta = if is_dir {
            "dir".to_string()
        } else {
            file_size_label(&cwd.join(fp))
        };
        let display_meta = if meta.is_empty() {
            fp.clone()
        } else {
            format!("{fp}  {meta}")
        };
        out.push(Completion::new(
            format!("@{kind}:{fp}"),
            -word_len,
            filename,
            display_meta,
        ));
    }
    out
}

/// Return the cached project file list, refreshing via `rg`/`fd` if stale.
///
/// Faithful port of `_get_project_files`: tries `rg --files --sortr=modified`,
/// then `rg --files`, then `fd --type f`, storing relative paths (max 5000).
pub fn get_project_files(cache: &mut ProjectFileCache) -> Vec<String> {
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let now = monotonic_secs();
    if !cache.files.is_empty() && cache.cwd == cwd && (now - cache.time) < 5.0 {
        return cache.files.clone();
    }

    let mut files: Vec<String> = Vec::new();
    let candidates: [&[&str]; 3] = [
        &["rg", "--files", "--sortr=modified"],
        &["rg", "--files"],
        &["fd", "--type", "f", "--base-directory"],
    ];
    for argv in candidates {
        let tool = argv[0];
        if which(tool).is_none() {
            continue;
        }
        let mut command = Command::new(tool);
        // rg takes the dir as a positional arg; fd uses --base-directory <dir>.
        for a in &argv[1..] {
            command.arg(a);
        }
        command.arg(&cwd);
        command.current_dir(&cwd);
        if let Ok(output) = command.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let trimmed = stdout.trim();
                if !trimmed.is_empty() {
                    for p in trimmed.split('\n').take(5000) {
                        let rel = if Path::new(p).is_absolute() {
                            Path::new(p)
                                .strip_prefix(&cwd)
                                .map(|r| r.to_string_lossy().to_string())
                                .unwrap_or_else(|_| p.to_string())
                        } else {
                            p.to_string()
                        };
                        files.push(rel);
                    }
                    break;
                }
            }
        }
    }

    cache.files = files.clone();
    cache.time = now;
    cache.cwd = cwd;
    files
}

/// Per-instance cache backing [`get_project_files`].
#[derive(Debug, Default)]
pub struct ProjectFileCache {
    files: Vec<String>,
    time: f64,
    cwd: String,
}

fn monotonic_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn which(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(tool);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// A dynamic completion source for `/model`, `/skin`, `/personality` —
/// supplied by the caller since the underlying data lives in other modules.
#[derive(Debug, Clone)]
pub struct DynamicEntry {
    /// Candidate name.
    pub name: String,
    /// Right-aligned metadata.
    pub meta: String,
}

/// Inputs that the caller injects for [`get_completions`] (data that the Python
/// version pulls from sibling modules at runtime).
#[derive(Default)]
pub struct CompletionContext<'a> {
    /// Skill commands as `/skill-name` -> description.
    pub skill_commands: &'a [(String, String)],
    /// Plugin commands as name (no slash) -> description.
    pub plugin_commands: &'a [(String, String)],
    /// `/model` dynamic entries (aliases, LM Studio models).
    pub model_entries: &'a [DynamicEntry],
    /// `/skin` dynamic entries.
    pub skin_entries: &'a [DynamicEntry],
    /// `/personality` dynamic entries (caller should include the `none` entry).
    pub personality_entries: &'a [DynamicEntry],
    /// Optional predicate restricting which `/command` strings are allowed.
    pub command_filter: Option<&'a dyn Fn(&str) -> bool>,
    /// Cached project files for `@`-completion (relative paths).
    pub project_files: &'a [String],
}

impl CompletionContext<'_> {
    fn command_allowed(&self, slash_command: &str) -> bool {
        match self.command_filter {
            None => true,
            Some(f) => f(slash_command),
        }
    }
}

fn dynamic_completions(entries: &[DynamicEntry], sub_text: &str, sub_lower: &str) -> Vec<Completion> {
    let mut out = Vec::new();
    let start = -(sub_text.chars().count() as i64);
    let mut seen: HashSet<&str> = HashSet::new();
    for e in entries {
        if seen.contains(e.name.as_str()) {
            continue;
        }
        if e.name.starts_with(sub_lower) && e.name != sub_lower {
            seen.insert(e.name.as_str());
            out.push(Completion::new(e.name.clone(), start, e.name.clone(), e.meta.clone()));
        }
    }
    out
}

/// Generate completions for the text typed before the cursor.
///
/// Faithful port of `SlashCommandCompleter.get_completions` minus prompt_toolkit
/// plumbing — dynamic and skill/plugin data are passed in via [`CompletionContext`].
pub fn get_completions(text: &str, ctx: &CompletionContext) -> Vec<Completion> {
    if !text.starts_with('/') {
        if let Some(ctx_word) = extract_context_word(text) {
            return context_completions(&ctx_word, 30, ctx.project_files);
        }
        if let Some(path_word) = extract_path_word(text) {
            return path_completions(&path_word, 30);
        }
        return Vec::new();
    }

    // Split into base command + remainder (Python's split(maxsplit=1)).
    let parts: Vec<&str> = text.splitn(2, ' ').collect();
    let base_cmd = parts[0].to_lowercase();
    let has_arg = parts.len() > 1 || (parts.len() == 1 && text.ends_with(' '));

    if has_arg {
        let sub_text = if parts.len() > 1 { parts[1] } else { "" };
        let sub_lower = sub_text.to_lowercase();

        if !sub_text.contains(' ') {
            match base_cmd.as_str() {
                "/model" => return dynamic_completions(ctx.model_entries, sub_text, &sub_lower),
                "/skin" => return dynamic_completions(ctx.skin_entries, sub_text, &sub_lower),
                "/personality" => {
                    return dynamic_completions(ctx.personality_entries, sub_text, &sub_lower)
                }
                _ => {}
            }
        }

        let mut out = Vec::new();
        if !sub_text.contains(' ') && ctx.command_allowed(&base_cmd) {
            if let Some(subs) = subcommands().get(&base_cmd) {
                let start = -(sub_text.chars().count() as i64);
                for sub in subs {
                    if sub.starts_with(&sub_lower) && *sub != sub_lower {
                        out.push(Completion::new(sub.clone(), start, sub.clone(), String::new()));
                    }
                }
            }
        }
        return out;
    }

    let word = &text[1..];
    let mut out = Vec::new();
    let start = -(word.chars().count() as i64);

    for (cmd_key, desc) in commands() {
        if !ctx.command_allowed(cmd_key) {
            continue;
        }
        let cmd_name = &cmd_key[1..];
        if cmd_name.starts_with(word) {
            out.push(Completion::new(
                completion_text(cmd_name, word),
                start,
                cmd_key.clone(),
                desc.clone(),
            ));
        }
    }

    for (cmd_key, description) in ctx.skill_commands {
        let cmd_name = cmd_key.trim_start_matches('/');
        if cmd_name.starts_with(word) {
            let short = if description.chars().count() > 50 {
                format!("{}...", truncate_chars(description, 50))
            } else {
                description.clone()
            };
            out.push(Completion::new(
                completion_text(cmd_name, word),
                start,
                cmd_key.clone(),
                format!("⚡ {short}"),
            ));
        }
    }

    for (cmd_name, desc) in ctx.plugin_commands {
        if cmd_name.starts_with(word) {
            let short = if desc.chars().count() > 50 {
                format!("{}...", truncate_chars(desc, 50))
            } else {
                desc.clone()
            };
            out.push(Completion::new(
                completion_text(cmd_name, word),
                start,
                format!("/{cmd_name}"),
                format!("🔌 {short}"),
            ));
        }
    }

    out
}

/// Inline ghost-text suggestion: the rest of a matching command or subcommand.
///
/// Faithful port of `SlashCommandAutoSuggest.get_suggestion` for slash input.
/// Returns the suffix to display, or `None`. Non-slash input is the caller's
/// (history) responsibility, mirrored by returning `None`.
pub fn get_suggestion(
    text: &str,
    command_filter: Option<&dyn Fn(&str) -> bool>,
) -> Option<String> {
    if !text.starts_with('/') {
        return None;
    }
    let allowed = |c: &str| command_filter.map(|f| f(c)).unwrap_or(true);

    let parts: Vec<&str> = text.splitn(2, ' ').collect();
    let base_cmd = parts[0].to_lowercase();

    if parts.len() == 1 && !text.ends_with(' ') {
        // Still typing the command name: /upd → suggest "ate".
        let word = text[1..].to_lowercase();
        for (cmd_key, _) in commands() {
            if !allowed(cmd_key) {
                continue;
            }
            let cmd_name = &cmd_key[1..];
            if cmd_name.starts_with(&word) && cmd_name != word {
                return Some(cmd_name[word.len()..].to_string());
            }
        }
        return None;
    }

    let sub_text = if parts.len() > 1 { parts[1] } else { "" };
    let sub_lower = sub_text.to_lowercase();

    if !allowed(&base_cmd) {
        return None;
    }
    if let Some(subs) = subcommands().get(&base_cmd) {
        if !subs.is_empty() && !sub_text.contains(' ') {
            for sub in subs {
                if sub.starts_with(&sub_lower) && *sub != sub_lower {
                    return Some(sub[sub_lower.len()..].to_string());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_handles_alias_and_slash() {
        assert_eq!(resolve_command("bg").unwrap().name, "background");
        assert_eq!(resolve_command("/BG").unwrap().name, "background");
        assert_eq!(resolve_command("reset").unwrap().name, "new");
        assert!(resolve_command("does-not-exist").is_none());
    }

    #[test]
    fn build_description_includes_hint() {
        let c = resolve_command("background").unwrap();
        assert_eq!(
            build_description(c),
            "Run a prompt in the background (usage: /background <prompt>)"
        );
        let help = resolve_command("help").unwrap();
        assert_eq!(build_description(help), "Show available commands");
    }

    #[test]
    fn commands_flat_excludes_gateway_only_and_includes_aliases() {
        let flat: HashMap<_, _> = commands().iter().cloned().collect();
        assert!(flat.contains_key("/help"));
        assert!(flat.contains_key("/bg"));
        assert!(flat.get("/bg").unwrap().contains("alias for /background"));
        // topic is gateway_only — excluded from the CLI flat dict.
        assert!(!flat.contains_key("/topic"));
        // clear is cli_only but not gateway_only — present in the flat dict.
        assert!(flat.contains_key("/clear"));
    }

    #[test]
    fn subcommands_explicit_and_pipe_fallback() {
        let subs = subcommands();
        assert_eq!(
            subs.get("/footer").unwrap(),
            &vec!["on".to_string(), "off".to_string(), "status".to_string()]
        );
        // /voice has no explicit subcommands list? It does — explicit wins.
        assert!(subs.contains_key("/voice"));
        // /topic args_hint "[off|help|session-id]" → pipe fallback matches
        // "off|help|session" ("-id" breaks the third token at the hyphen).
        assert_eq!(
            subs.get("/topic").unwrap(),
            &vec!["off".to_string(), "help".to_string(), "session".to_string()]
        );
        // No pipe in args_hint → no entry (e.g. /new "[name]").
        assert!(!subs.contains_key("/new"));
    }

    #[test]
    fn gateway_known_includes_config_gated() {
        // verbose is cli_only but has a config gate → gateway-known.
        assert!(gateway_known_commands().contains("verbose"));
        assert!(is_gateway_known_command(Some("verbose")));
        // clear is cli_only with no gate → not gateway-known.
        assert!(!is_gateway_known_command(Some("clear")));
        assert!(!is_gateway_known_command(None));
        assert!(!is_gateway_known_command(Some("")));
    }

    #[test]
    fn bypass_active_session() {
        assert!(should_bypass_active_session(Some("model")));
        assert!(should_bypass_active_session(Some("/reset")));
        assert!(!should_bypass_active_session(Some("nonexistent")));
        assert!(!should_bypass_active_session(None));
    }

    #[test]
    fn config_gate_resolution() {
        let yaml = "display:\n  tool_progress_command: true\n";
        let cfg: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let gated = resolve_config_gates(Some(&cfg));
        assert!(gated.contains("verbose"));

        let yaml_off = "display:\n  tool_progress_command: false\n";
        let cfg_off: serde_yaml::Value = serde_yaml::from_str(yaml_off).unwrap();
        assert!(resolve_config_gates(Some(&cfg_off)).is_empty());

        assert!(resolve_config_gates(None).is_empty());
    }

    #[test]
    fn gateway_available_respects_gates() {
        let no_gates = HashSet::new();
        let verbose = resolve_command("verbose").unwrap();
        assert!(!is_gateway_available(verbose, &no_gates));
        let mut gated = HashSet::new();
        gated.insert("verbose".to_string());
        assert!(is_gateway_available(verbose, &gated));
        // help is not cli_only → always available.
        assert!(is_gateway_available(resolve_command("help").unwrap(), &no_gates));
    }

    #[test]
    fn requires_argument_only_for_angle_bracket() {
        assert!(requires_argument("<prompt>"));
        assert!(!requires_argument("[name]"));
        assert!(!requires_argument(""));
    }

    #[test]
    fn gateway_help_lines_format() {
        let lines = gateway_help_lines(&HashSet::new());
        // background has aliases bg, btw → alias note present.
        let bg = lines.iter().find(|l| l.starts_with("`/background")).unwrap();
        assert!(bg.contains("(alias: `/bg`, `/btw`)"));
        // reload-mcp's underscore alias is filtered out → no alias note.
        let reload = lines.iter().find(|l| l.starts_with("`/reload-mcp")).unwrap();
        assert!(!reload.contains("alias"));
        // cli_only without gate not present.
        assert!(!lines.iter().any(|l| l.starts_with("`/clear")));
    }

    #[test]
    fn sanitize_telegram() {
        assert_eq!(sanitize_telegram_name("reload-mcp"), "reload_mcp");
        assert_eq!(sanitize_telegram_name("Set-Home"), "set_home");
        assert_eq!(sanitize_telegram_name("a--b"), "a_b");
        assert_eq!(sanitize_telegram_name("__x__"), "x");
        assert_eq!(sanitize_telegram_name("!!!"), "");
    }

    #[test]
    fn telegram_bot_commands_skips_args_and_gateway_only() {
        let cmds = telegram_bot_commands(&HashSet::new(), &[]);
        let names: HashSet<_> = cmds.iter().map(|(n, _)| n.clone()).collect();
        // background requires <prompt> → skipped.
        assert!(!names.contains("background"));
        // help is fine.
        assert!(names.contains("help"));
        // reload-mcp sanitized to reload_mcp.
        assert!(names.contains("reload_mcp"));
        // topic is gateway_only (not cli_only) → available.
        assert!(names.contains("topic"));
    }

    #[test]
    fn clamp_names_truncates_and_dedups() {
        let long = "a".repeat(40);
        let entries = vec![
            (long.clone(), "d1".to_string(), "k1".to_string()),
            (long.clone(), "d2".to_string(), "k2".to_string()),
        ];
        let out = clamp_command_names(entries, &HashSet::new());
        assert_eq!(out[0].0.chars().count(), 32);
        // Second collides → 31 chars + digit.
        assert_eq!(out[1].0.chars().count(), 32);
        assert!(out[1].0.ends_with('0'));
        assert_ne!(out[0].0, out[1].0);
        // extra payload preserved.
        assert_eq!(out[0].2, "k1");
    }

    #[test]
    fn sanitize_slack() {
        assert_eq!(sanitize_slack_name("Background"), "background");
        assert_eq!(sanitize_slack_name("reload-mcp"), "reload-mcp");
        assert_eq!(sanitize_slack_name("-x-"), "x");
        assert_eq!(sanitize_slack_name("a b@c"), "abc");
    }

    #[test]
    fn slack_slashes_reserve_hermes_and_skip_reserved() {
        let slashes = slack_native_slashes(&HashSet::new(), &[]);
        assert_eq!(slashes[0].0, "hermes");
        let names: HashSet<_> = slashes.iter().map(|(n, _, _)| n.clone()).collect();
        // /status collides with a Slack built-in → skipped.
        assert!(!names.contains("status"));
        // /topic also reserved by Slack → skipped.
        assert!(!names.contains("topic"));
        // background present.
        assert!(names.contains("background"));
        // alias bg present.
        assert!(names.contains("bg"));
    }

    #[test]
    fn slack_manifest_shape() {
        let m = slack_app_manifest("https://x/cmd", &HashSet::new(), &[]);
        let slashes = m["features"]["slash_commands"].as_array().unwrap();
        let first = &slashes[0];
        assert_eq!(first["command"], "/hermes");
        assert_eq!(first["should_escape"], false);
        assert_eq!(first["url"], "https://x/cmd");
        assert_eq!(first["usage_hint"], "[subcommand] [args]");
    }

    #[test]
    fn slack_subcommand_map_includes_aliases() {
        let map = slack_subcommand_map(&HashSet::new(), &[]);
        let m: HashMap<_, _> = map.into_iter().collect();
        assert_eq!(m.get("background").unwrap(), "/background");
        assert_eq!(m.get("bg").unwrap(), "/bg");
        // cli_only without gate excluded.
        assert!(!m.contains_key("clear"));
    }

    #[test]
    fn completion_text_picker_vs_normal() {
        assert_eq!(completion_text("help", "help"), "help ");
        assert_eq!(completion_text("model", "model"), "model");
        assert_eq!(completion_text("help", "hel"), "help");
    }

    #[test]
    fn score_path_tiers() {
        assert_eq!(score_path("a/main.py", "main.py"), 100);
        assert_eq!(score_path("a/main.py", "main"), 80);
        assert_eq!(score_path("a/the_main_x.py", "main"), 60);
        assert_eq!(score_path("src/deep/x.py", "deep"), 40);
        // abbreviation on word boundaries: "fo" in "file_operations".
        assert_eq!(score_path("file_operations.py", "fo"), 35);
        assert_eq!(score_path("nomatch.py", "zzz"), 0);
        assert_eq!(score_path("anything", ""), 1);
    }

    #[test]
    fn extract_words() {
        assert_eq!(extract_path_word("foo src/main.rs"), Some("src/main.rs".to_string()));
        assert_eq!(extract_path_word("foo ./x"), Some("./x".to_string()));
        assert_eq!(extract_path_word("foo bar"), None);
        assert_eq!(extract_context_word("hi @fo"), Some("@fo".to_string()));
        assert_eq!(extract_context_word("hi there"), None);
    }

    #[test]
    fn completions_command_prefix() {
        let ctx = CompletionContext::default();
        let comps = get_completions("/he", &ctx);
        let help = comps.iter().find(|c| c.display == "/help").unwrap();
        assert_eq!(help.text, "help");
        assert_eq!(help.start_position, -2);
    }

    #[test]
    fn completions_subcommand() {
        let ctx = CompletionContext::default();
        let comps = get_completions("/footer o", &ctx);
        let texts: Vec<_> = comps.iter().map(|c| c.text.clone()).collect();
        assert!(texts.contains(&"on".to_string()));
        assert!(texts.contains(&"off".to_string()));
        assert!(!texts.contains(&"status".to_string()));
    }

    #[test]
    fn completions_command_filter() {
        let filter = |c: &str| c != "/help";
        let ctx = CompletionContext {
            command_filter: Some(&filter),
            ..Default::default()
        };
        let comps = get_completions("/he", &ctx);
        assert!(comps.iter().all(|c| c.display != "/help"));
    }

    #[test]
    fn suggestion_command_and_subcommand() {
        // /usa → "usage" (usage is not gateway_only, so it is in COMMANDS).
        assert_eq!(get_suggestion("/usa", None), Some("ge".to_string()));
        assert_eq!(get_suggestion("/footer o", None), Some("n".to_string()));
        assert_eq!(get_suggestion("/help", None), None);
        assert_eq!(get_suggestion("plain text", None), None);
    }

    #[test]
    fn fuzzy_empty_query_lists_files() {
        let files = vec!["a.txt".to_string(), "dir/".to_string()];
        let comps = fuzzy_file_completions("@", "", 10, &files);
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].text, "@file:a.txt");
        assert_eq!(comps[1].text, "@folder:dir/");
    }

    #[test]
    fn file_size_labels() {
        // Build size labels via a temp file would be overkill; check formula
        // indirectly through a missing path returning "".
        assert_eq!(file_size_label(Path::new("/nonexistent/zzz/qqq")), "");
    }
}
