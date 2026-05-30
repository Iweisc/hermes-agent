//! Native slash-command registry.
//!
//! Mirrors `hermes_cli/commands.py`: the `CommandDef` table is generated into
//! `commands_registry_data.rs` by `scripts/gen_command_registry.py`, and the
//! derivation logic below (resolve, description, subcommands, completion) is a
//! faithful port of the Python helpers. This replaces the `python3 -c` helpers
//! that the interactive TUI gateway previously spawned for `command.resolve`,
//! `commands.catalog`, and `complete.slash`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value, json};
use serde_yaml::Value as YamlValue;

use crate::commands_registry_data::COMMAND_REGISTRY;

/// Definition of a single slash command (port of the Python dataclass).
#[derive(Debug, Clone, Copy)]
pub struct CommandDef {
    pub name: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    pub aliases: &'static [&'static str],
    pub args_hint: &'static str,
    pub subcommands: &'static [&'static str],
    pub cli_only: bool,
    pub gateway_only: bool,
    pub gateway_config_gate: Option<&'static str>,
}

/// Return the full command registry.
pub fn command_registry() -> &'static [CommandDef] {
    COMMAND_REGISTRY
}

fn command_lookup() -> &'static HashMap<String, usize> {
    static LOOKUP: OnceLock<HashMap<String, usize>> = OnceLock::new();
    LOOKUP.get_or_init(|| {
        let mut lookup = HashMap::new();
        for (index, cmd) in COMMAND_REGISTRY.iter().enumerate() {
            lookup.insert(cmd.name.to_string(), index);
            for alias in cmd.aliases {
                lookup.insert((*alias).to_string(), index);
            }
        }
        lookup
    })
}

/// Resolve a command name or alias to its `CommandDef`.
///
/// Accepts names with or without the leading slash (mirrors
/// `hermes_cli.commands.resolve_command`).
pub fn resolve_command(name: &str) -> Option<&'static CommandDef> {
    let key = name.to_lowercase();
    let key = key.trim_start_matches('/');
    command_lookup().get(key).map(|&index| &COMMAND_REGISTRY[index])
}

/// Build a CLI-facing description string including the usage hint.
///
/// Port of `hermes_cli.commands._build_description`.
pub fn build_description(cmd: &CommandDef) -> String {
    if !cmd.args_hint.is_empty() {
        format!(
            "{} (usage: /{} {})",
            cmd.description, cmd.name, cmd.args_hint
        )
    } else {
        cmd.description.to_string()
    }
}

/// Backwards-compatible flat map: `/command` -> description (non-gateway-only),
/// preserving registry order. Aliases get an "(alias for /name)" description.
///
/// Port of the module-level `COMMANDS` dict in `hermes_cli/commands.py`.
pub fn commands_flat() -> &'static Vec<(String, String)> {
    static COMMANDS: OnceLock<Vec<(String, String)>> = OnceLock::new();
    COMMANDS.get_or_init(|| {
        let mut commands = Vec::new();
        for cmd in COMMAND_REGISTRY {
            if cmd.gateway_only {
                continue;
            }
            commands.push((format!("/{}", cmd.name), build_description(cmd)));
            for alias in cmd.aliases {
                commands.push((
                    format!("/{alias}"),
                    format!("{} (alias for /{})", cmd.description, cmd.name),
                ));
            }
        }
        commands
    })
}

fn pipe_subs_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[a-z]+(?:\|[a-z]+)+").expect("valid pipe-subs regex"))
}

/// Subcommands lookup: `/cmd` -> [sub, ...].
///
/// Mirrors the `SUBCOMMANDS` construction in `hermes_cli/commands.py`: explicit
/// `subcommands` first, then a fallback that extracts pipe-separated tokens from
/// `args_hint` for commands without explicit subcommands.
pub fn subcommands() -> &'static HashMap<String, Vec<String>> {
    static SUBCOMMANDS: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    SUBCOMMANDS.get_or_init(|| {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for cmd in COMMAND_REGISTRY {
            if !cmd.subcommands.is_empty() {
                map.insert(
                    format!("/{}", cmd.name),
                    cmd.subcommands.iter().map(|s| s.to_string()).collect(),
                );
            }
        }
        for cmd in COMMAND_REGISTRY {
            let key = format!("/{}", cmd.name);
            if map.contains_key(&key) || cmd.args_hint.is_empty() {
                continue;
            }
            if let Some(m) = pipe_subs_regex().find(cmd.args_hint) {
                map.insert(key, m.as_str().split('|').map(|s| s.to_string()).collect());
            }
        }
        map
    })
}

/// TUI-only commands that augment the catalog and completer. These have no
/// `CommandDef` (they're TUI-local) and are appended after the registry.
/// Mirrors the `TUI_EXTRA` / completer extras in the Python helpers.
pub const TUI_EXTRA: &[(&str, &str, &str)] = &[
    ("/compact", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    ("/mouse", "Toggle mouse/wheel tracking [on|off|toggle]", "TUI"),
];

/// Commands hidden from the TUI catalog (handled specially by the TUI).
/// Mirrors `TUI_HIDDEN` in the COMMANDS_CATALOG_HELPER python.
const TUI_HIDDEN: &[&str] = &[
    "new", "clear", "quit", "exit", "copy", "paste", "image", "commands", "approve", "deny",
    "sethome", "set-home", "update",
];

/// Resolve a command name (no slash) to canonical name + description + category.
///
/// Returns the JSON shape produced by the former `COMMAND_RESOLVE_HELPER`:
/// `{"canonical", "description", "category"}`, or `None` if unknown.
pub fn resolve_command_json(name: &str) -> Option<Value> {
    resolve_command(name).map(|cmd| {
        json!({
            "canonical": cmd.name,
            "description": cmd.description,
            "category": cmd.category,
        })
    })
}

/// Truncate a string to `max` chars, appending an ellipsis when truncated.
fn truncate_ellipsis(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > max {
        let mut out: String = chars[..max].iter().collect();
        out.push('…');
        out
    } else {
        text.to_string()
    }
}

/// Build the command catalog payload (port of `COMMANDS_CATALOG_HELPER`).
///
/// `quick_commands` is the parsed `quick_commands` config map (name -> spec),
/// and `skill_commands` is the scanned `/cmd -> {name, description}` map. Both
/// may be empty. The returned JSON matches the Python helper's shape:
/// `{pairs, sub, canon, categories, skill_count, warning}`.
pub fn commands_catalog(
    quick_commands: &Map<String, Value>,
    skill_commands: &[(String, String)],
) -> Value {
    let mut all_pairs: Vec<Value> = Vec::new();
    let mut canon: Map<String, Value> = Map::new();
    let mut cat_order: Vec<String> = Vec::new();
    let mut cat_map: HashMap<String, Vec<Value>> = HashMap::new();

    let push_cat = |cat_order: &mut Vec<String>,
                    cat_map: &mut HashMap<String, Vec<Value>>,
                    category: &str,
                    pair: Value| {
        if !cat_map.contains_key(category) {
            cat_map.insert(category.to_string(), Vec::new());
            cat_order.push(category.to_string());
        }
        cat_map.get_mut(category).unwrap().push(pair);
    };

    for cmd in COMMAND_REGISTRY {
        if TUI_HIDDEN.contains(&cmd.name) || cmd.gateway_only {
            continue;
        }
        let c = format!("/{}", cmd.name);
        canon.insert(c.to_lowercase(), Value::String(c.clone()));
        for alias in cmd.aliases {
            canon.insert(format!("/{alias}").to_lowercase(), Value::String(c.clone()));
        }
        let desc = build_description(cmd);
        let pair = json!([c, desc]);
        all_pairs.push(pair.clone());
        push_cat(&mut cat_order, &mut cat_map, cmd.category, pair);
    }

    for (name, desc, cat) in TUI_EXTRA {
        let pair = json!([name, desc]);
        all_pairs.push(pair.clone());
        push_cat(&mut cat_order, &mut cat_map, cat, pair);
    }

    // The Python helper only sets a warning when quick-command/skill discovery
    // raises; native discovery returns errors to the caller instead, so this is
    // always empty here.
    let warning = String::new();

    if !quick_commands.is_empty() {
        let bucket = "User commands";
        let mut names: Vec<&String> = quick_commands.keys().collect();
        names.sort();
        for qname in names {
            let Some(qc) = quick_commands.get(qname).and_then(Value::as_object) else {
                continue;
            };
            let key = format!("/{qname}");
            canon.insert(key.to_lowercase(), Value::String(key.clone()));
            let qtype = qc.get("type").and_then(Value::as_str).unwrap_or("");
            let default_desc = match qtype {
                "exec" => format!(
                    "exec: {}",
                    qc.get("command").and_then(Value::as_str).unwrap_or("")
                ),
                "alias" => format!(
                    "alias -> {}",
                    qc.get("target").and_then(Value::as_str).unwrap_or("")
                ),
                other if !other.is_empty() => other.to_string(),
                _ => "quick command".to_string(),
            };
            let qdesc_raw = qc
                .get("description")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or(default_desc);
            let qdesc = truncate_ellipsis(&qdesc_raw, 120);
            let pair = json!([key, qdesc]);
            all_pairs.push(pair.clone());
            push_cat(&mut cat_order, &mut cat_map, bucket, pair);
        }
    }

    let mut skill_count = 0u64;
    let mut skills_sorted: Vec<&(String, String)> = skill_commands.iter().collect();
    skills_sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, desc) in skills_sorted {
        all_pairs.push(json!([key, truncate_ellipsis(desc, 120)]));
        skill_count += 1;
    }

    let categories: Vec<Value> = cat_order
        .iter()
        .map(|cat| json!({"name": cat, "pairs": cat_map.get(cat).cloned().unwrap_or_default()}))
        .collect();

    let mut sub: Map<String, Value> = Map::new();
    for (key, subs) in subcommands() {
        sub.insert(key.clone(), json!(subs));
    }

    json!({
        "pairs": all_pairs,
        "sub": sub,
        "canon": canon,
        "categories": categories,
        "skill_count": skill_count,
        "warning": warning,
    })
}

/// Read `quick_commands` from `<hermes_home>/config.yaml` as a JSON object
/// (name -> spec), mirroring `read_raw_config().get("quick_commands", {})`.
/// Returns an empty map when the file is missing/unparsable or the key absent.
pub fn read_quick_commands(hermes_home: &Path) -> Map<String, Value> {
    let path = hermes_home.join("config.yaml");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Map::new();
    };
    let Ok(YamlValue::Mapping(root)) = serde_yaml::from_str::<YamlValue>(&text) else {
        return Map::new();
    };
    match root.get(YamlValue::String("quick_commands".to_string())) {
        Some(value) => match yaml_to_json(value) {
            Value::Object(map) => map,
            _ => Map::new(),
        },
        None => Map::new(),
    }
}

fn yaml_to_json(value: &YamlValue) -> Value {
    match value {
        YamlValue::Null => Value::Null,
        YamlValue::Bool(b) => Value::Bool(*b),
        YamlValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                json!(i)
            } else if let Some(u) = n.as_u64() {
                json!(u)
            } else if let Some(f) = n.as_f64() {
                json!(f)
            } else {
                Value::Null
            }
        }
        YamlValue::String(s) => Value::String(s.clone()),
        YamlValue::Sequence(seq) => Value::Array(seq.iter().map(yaml_to_json).collect()),
        YamlValue::Mapping(mapping) => {
            let mut object = Map::new();
            for (key, value) in mapping {
                if let Some(key) = key.as_str() {
                    object.insert(key.to_string(), yaml_to_json(value));
                }
            }
            Value::Object(object)
        }
        YamlValue::Tagged(tagged) => yaml_to_json(&tagged.value),
    }
}

/// Outcome of a native `command.dispatch`: a ready JSON-RPC `result` object, an
/// error to surface, or `NotHandled` to fall back to the Python helper for the
/// plugin/skill/queue/steer/unknown branches.
pub enum DispatchOutcome {
    Ok(Value),
    Err(String),
    NotHandled,
}

/// Handle the leading, self-contained branches of `COMMAND_DISPATCH_HELPER`:
/// command-alias resolution and `quick_commands` (exec/alias). These are
/// evaluated *before* the plugin-handler check in the Python helper, so native
/// handling here preserves exact precedence — anything not a quick command
/// returns [`DispatchOutcome::NotHandled`] and the caller defers to Python
/// (which then checks plugin handlers, skills, queue/steer in order).
///
/// The exec branch runs the shell command (30s timeout) and formats output
/// exactly like the Python: stdout and stderr joined with a newline when both
/// are present, trimmed, capped at 4000 chars; a non-zero exit becomes an error.
pub fn command_dispatch(hermes_home: &Path, name: &str, arg: &str) -> DispatchOutcome {
    // name comes in without a leading slash; resolve aliases to canonical.
    let mut name = name.trim_start_matches('/').to_string();
    if let Some(resolved) = resolve_command(&name) {
        name = resolved.name.to_string();
    }

    let quick = read_quick_commands(hermes_home);
    let Some(qc) = quick.get(&name).and_then(Value::as_object) else {
        // Not a quick command — let Python handle plugin/skill/queue/steer.
        let _ = arg;
        return DispatchOutcome::NotHandled;
    };

    match qc.get("type").and_then(Value::as_str) {
        Some("exec") => {
            let command = qc.get("command").and_then(Value::as_str).unwrap_or("");
            run_exec_quick_command(command)
        }
        Some("alias") => {
            let target = qc.get("target").and_then(Value::as_str).unwrap_or("");
            DispatchOutcome::Ok(json!({"type": "alias", "target": target}))
        }
        // A quick command of an unexpected type falls through to Python, which
        // would skip the quick-command block and continue its precedence chain.
        _ => DispatchOutcome::NotHandled,
    }
}

/// Run a `type: exec` quick command via the shell, faithfully reproducing the
/// Python helper's exec branch: `subprocess.run(cmd, shell=True, timeout=30)`,
/// output = stdout (+ "\n" if both) + stderr, trimmed and capped at 4000; a
/// non-zero exit raises (here: returns `Err`) with the output or an exit-code
/// message.
fn run_exec_quick_command(command: &str) -> DispatchOutcome {
    use std::process::{Command, Stdio};

    // shell=True parity: invoke via the platform shell.
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };

    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let child = match child {
        Ok(child) => child,
        Err(error) => return DispatchOutcome::Err(format!("quick command failed to start: {error}")),
    };

    // 30s timeout parity with the Python subprocess.run(..., timeout=30).
    let output = match wait_with_timeout(child, std::time::Duration::from_secs(30)) {
        Ok(output) => output,
        Err(error) => return DispatchOutcome::Err(error),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let joined = join_exec_output(&stdout, &stderr);

    if !output.status.success() {
        let code = output.status.code();
        let message = if joined.is_empty() {
            match code {
                Some(code) => format!("quick command failed with exit code {code}"),
                None => "quick command failed".to_string(),
            }
        } else {
            joined
        };
        return DispatchOutcome::Err(message);
    }
    DispatchOutcome::Ok(json!({"type": "exec", "output": joined}))
}

struct ExecOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Wait for a child with a timeout, reading stdout/stderr to completion.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> Result<ExecOutput, String> {
    use std::io::Read;
    use std::thread;
    use std::time::{Duration, Instant};

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_handle = thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = stdout_pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        bytes
    });
    let stderr_handle = thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = stderr_pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        bytes
    });

    let start = Instant::now();
    let timed_out = loop {
        match child.try_wait().map_err(|e| format!("waiting on quick command failed: {e}"))? {
            Some(_) => break false,
            None if start.elapsed() >= timeout => break true,
            None => thread::sleep(Duration::from_millis(10)),
        }
    };

    if timed_out {
        let _ = child.kill();
    }
    let status = child.wait().map_err(|e| format!("waiting on quick command failed: {e}"))?;
    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();
    if timed_out {
        return Err("quick command timed out after 30 seconds".to_string());
    }
    Ok(ExecOutput { status, stdout, stderr })
}

/// Join exec stdout/stderr the way the Python helper does:
/// `(stdout + "\n" if both else "") + stderr`, then `.strip()` and cap at 4000.
fn join_exec_output(stdout: &str, stderr: &str) -> String {
    let mut combined = String::new();
    combined.push_str(stdout);
    if !stdout.is_empty() && !stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(stderr);
    let trimmed = combined.trim();
    trimmed.chars().take(4000).collect()
}

/// Release date reported to the TUI, mirroring `hermes_cli.__release_date__`.
/// Kept in sync with `hermes_cli/__init__.py`.
pub const HERMES_RELEASE_DATE: &str = "2026.4.30";

/// Default model when none is configured (mirrors `_resolve_model`'s fallback).
const DEFAULT_TUI_MODEL: &str = "anthropic/claude-sonnet-4";

/// Resolve the TUI startup model. Port of `tui_gateway.server._resolve_model`:
/// `HERMES_MODEL`/`HERMES_INFERENCE_MODEL` env, then `config.yaml` `model`
/// (string, or `model.default` when it's a mapping), then the default.
pub fn resolve_tui_model(hermes_home: &Path) -> String {
    let env_model = std::env::var("HERMES_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("HERMES_INFERENCE_MODEL").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(model) = env_model {
        return model;
    }
    if let Some(config) = read_config_mapping_yaml(hermes_home) {
        match config.get(YamlValue::String("model".to_string())) {
            Some(YamlValue::Mapping(model)) => {
                if let Some(default) = model
                    .get(YamlValue::String("default".to_string()))
                    .and_then(YamlValue::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    return default.to_string();
                }
            }
            Some(YamlValue::String(model)) if !model.trim().is_empty() => {
                return model.trim().to_string();
            }
            _ => {}
        }
    }
    DEFAULT_TUI_MODEL.to_string()
}

fn read_config_mapping_yaml(hermes_home: &Path) -> Option<serde_yaml::Mapping> {
    let text = std::fs::read_to_string(hermes_home.join("config.yaml")).ok()?;
    match serde_yaml::from_str::<YamlValue>(&text).ok()? {
        YamlValue::Mapping(mapping) => Some(mapping),
        _ => None,
    }
}

/// Build the initial session-info payload (port of `INITIAL_SESSION_INFO_HELPER`).
///
/// `cwd` mirrors the Python helper: `TERMINAL_CWD` env, else the provided
/// working directory. `version` comes from the crate version and `release_date`
/// from [`HERMES_RELEASE_DATE`].
pub fn initial_session_info(hermes_home: &Path, work_dir: &str) -> Value {
    let cwd = std::env::var("TERMINAL_CWD")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| work_dir.to_string());
    json!({
        "model": resolve_tui_model(hermes_home),
        "tools": {},
        "skills": {},
        "cwd": cwd,
        "lazy": true,
        "version": env!("CARGO_PKG_VERSION"),
        "release_date": HERMES_RELEASE_DATE,
    })
}

/// Picker commands that should NOT get a trailing space in completions.
const PICKER_COMMANDS: &[&str] = &["model", "skin", "personality"];

/// Compute the replacement text for a command completion.
/// Port of `SlashCommandCompleter._completion_text`.
fn completion_text(cmd_name: &str, word: &str) -> String {
    if cmd_name != word {
        return cmd_name.to_string();
    }
    if PICKER_COMMANDS.contains(&cmd_name) {
        return cmd_name.to_string();
    }
    format!("{cmd_name} ")
}

/// Slash-command completion (port of `SLASH_COMPLETION_HELPER`, which wraps
/// `SlashCommandCompleter.get_completions` plus the TUI extras).
///
/// `skill_commands` is the scanned `/cmd -> {name, description}` map (key,
/// description). Returns `{items, replace_from}` matching the Python helper.
///
/// Fidelity note: the Python completer had a final branch that listed
/// plugin-registered slash commands via `hermes_cli.plugins.get_plugin_commands`.
/// That is intentionally omitted here — surfacing it would require spawning the
/// Python plugin bridge on every keystroke, which is exactly the per-keystroke
/// `python3` spawn this port removes. Plugin slash-commands therefore don't
/// appear in completions (the same large-surface boundary as `command.dispatch`,
/// which remains bridged); built-in and skill commands are fully covered.
pub fn complete_slash(text: &str, skill_commands: &[(String, String)]) -> Value {
    let mut items: Vec<Value> = Vec::new();

    if text.starts_with('/') {
        // Subcommand completion path: base command already typed.
        // Mirror Python's `text.split(maxsplit=1)`: the head is the first
        // whitespace-delimited token and the remainder has its leading
        // whitespace stripped.
        let (head, remainder) = match text.find(char::is_whitespace) {
            Some(idx) => (&text[..idx], text[idx..].trim_start()),
            None => (text, ""),
        };
        let base_cmd = head.to_lowercase();
        let has_remainder = text.len() > head.len();
        if has_remainder {
            let sub_text = remainder;
            let sub_lower = sub_text.to_lowercase();
            // Dynamic completions (model/skin/personality) require live data not
            // available here; the Python helper yields nothing static for them.
            if !sub_text.contains(' ')
                && !matches!(base_cmd.as_str(), "/model" | "/skin" | "/personality")
            {
                if let Some(subs) = subcommands().get(&base_cmd) {
                    for sub in subs {
                        if sub.starts_with(&sub_lower) && *sub != sub_lower {
                            items.push(json!({
                                "text": sub,
                                "display": sub,
                                "meta": "",
                            }));
                        }
                    }
                }
            }
        } else {
            let word = &text[1..];
            for (cmd, desc) in commands_flat() {
                let cmd_name = &cmd[1..];
                if cmd_name.starts_with(word) {
                    items.push(json!({
                        "text": completion_text(cmd_name, word),
                        "display": cmd,
                        "meta": desc,
                    }));
                }
            }
            for (cmd, description) in skill_commands {
                let cmd_name = &cmd[1..];
                if cmd_name.starts_with(word) {
                    let short = truncate_dots(description, 50);
                    items.push(json!({
                        "text": completion_text(cmd_name, word),
                        "display": cmd,
                        "meta": format!("⚡ {short}"),
                    }));
                }
            }
        }
    }

    items.truncate(30);

    // TUI-specific extras appended by the helper.
    let text_lower = text.to_lowercase();
    for extra in [
        ("/compact", "Toggle compact display mode"),
        ("/details", "Control agent detail visibility"),
        ("/logs", "Show recent gateway log lines"),
        ("/mouse", "Toggle mouse/wheel tracking [on|off|toggle]"),
    ] {
        if extra.0.starts_with(&text_lower)
            && !items
                .iter()
                .any(|item| item.get("text").and_then(Value::as_str) == Some(extra.0))
        {
            items.push(json!({"text": extra.0, "display": extra.0, "meta": extra.1}));
        }
    }

    let replace_from = if text.contains(' ') {
        text.rfind(' ').map(|i| i + 1).unwrap_or(1)
    } else {
        1
    };

    json!({"items": items, "replace_from": replace_from})
}

/// Truncate to `max` chars, appending "..." when truncated. Mirrors the
/// `desc[:50] + ("..." if len(desc) > 50 else "")` form in the Python completer.
fn truncate_dots(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > max {
        let head: String = chars[..max].iter().collect();
        format!("{head}...")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_and_alias() {
        assert_eq!(resolve_command("new").unwrap().name, "new");
        // alias "reset" -> "new"
        assert_eq!(resolve_command("reset").unwrap().name, "new");
        // leading slash + case-insensitive
        assert_eq!(resolve_command("/RESET").unwrap().name, "new");
        assert!(resolve_command("definitely-not-a-command").is_none());
    }

    #[test]
    fn build_description_with_and_without_hint() {
        let new_cmd = resolve_command("new").unwrap();
        assert_eq!(
            build_description(new_cmd),
            "Start a new session (fresh session ID + history) (usage: /new [name])"
        );
        let retry = resolve_command("retry").unwrap();
        assert_eq!(build_description(retry), "Retry the last message (resend to agent)");
    }

    #[test]
    fn subcommands_explicit_and_pipe_fallback() {
        let subs = subcommands();
        // explicit subcommands
        assert_eq!(
            subs.get("/reasoning").unwrap(),
            &vec![
                "none", "minimal", "low", "medium", "high", "xhigh", "show", "hide", "on", "off"
            ]
        );
        // Pipe fallback: /topic has no explicit subcommands but its args_hint
        // "[off|help|session-id]" yields ["off","help","session"] — the regex
        // stops at the hyphen, exactly as the Python `_PIPE_SUBS_RE` does.
        assert_eq!(
            subs.get("/topic").unwrap(),
            &vec!["off", "help", "session"]
        );
        // A command with no pipes and no explicit subcommands is absent.
        assert!(!subs.contains_key("/retry"));
    }

    #[test]
    fn resolve_command_json_shape() {
        let v = resolve_command_json("fork").unwrap();
        assert_eq!(v["canonical"], "branch");
        assert_eq!(v["category"], "Session");
    }

    #[test]
    fn complete_slash_prefix_matches() {
        let v = complete_slash("/re", &[]);
        let items = v["items"].as_array().unwrap();
        let texts: Vec<&str> = items
            .iter()
            .filter_map(|i| i["display"].as_str())
            .collect();
        // /retry, /redraw, /resume, /reset, /reload, /reload-mcp, /reasoning, /restart... at least these
        assert!(texts.contains(&"/retry"));
        assert!(texts.contains(&"/reset"));
        assert_eq!(v["replace_from"], 1);
    }

    #[test]
    fn complete_slash_appends_tui_extras() {
        let v = complete_slash("/mo", &[]);
        let items = v["items"].as_array().unwrap();
        // /mouse is a TUI extra and should appear for prefix "/mo"
        assert!(items
            .iter()
            .any(|i| i["text"].as_str() == Some("/mouse")));
    }

    #[test]
    fn resolve_tui_model_reads_config_default() {
        let temp = tempfile::TempDir::new().unwrap();
        let home = temp.path();
        // No config and no env -> default. (Guard against a HERMES_MODEL set in
        // the test environment by only asserting the config-backed paths.)
        std::fs::write(
            home.join("config.yaml"),
            "model:\n  default: openai/gpt-5\n",
        )
        .unwrap();
        // Env precedence is documented; here we assert the config mapping path
        // when the env vars are absent.
        if std::env::var_os("HERMES_MODEL").is_none()
            && std::env::var_os("HERMES_INFERENCE_MODEL").is_none()
        {
            assert_eq!(resolve_tui_model(home), "openai/gpt-5");
        }

        // String form of `model`.
        std::fs::write(home.join("config.yaml"), "model: anthropic/claude-x\n").unwrap();
        if std::env::var_os("HERMES_MODEL").is_none()
            && std::env::var_os("HERMES_INFERENCE_MODEL").is_none()
        {
            assert_eq!(resolve_tui_model(home), "anthropic/claude-x");
        }
    }

    #[test]
    fn initial_session_info_shape() {
        let temp = tempfile::TempDir::new().unwrap();
        let info = initial_session_info(temp.path(), "/work/dir");
        assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(info["release_date"], HERMES_RELEASE_DATE);
        assert_eq!(info["lazy"], true);
        assert!(info["model"].as_str().is_some());
        // cwd falls back to the work dir when TERMINAL_CWD is unset.
        if std::env::var_os("TERMINAL_CWD").is_none() {
            assert_eq!(info["cwd"], "/work/dir");
        }
    }

    #[test]
    fn dispatch_quick_command_exec_alias_and_fallthrough() {
        let temp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("config.yaml"),
            "quick_commands:\n  hi:\n    type: exec\n    command: \"echo hello\"\n  gp:\n    type: alias\n    target: \"git push\"\n",
        )
        .unwrap();

        // exec quick command runs and captures stdout
        match command_dispatch(temp.path(), "hi", "") {
            DispatchOutcome::Ok(v) => {
                assert_eq!(v["type"], "exec");
                assert_eq!(v["output"], "hello");
            }
            DispatchOutcome::Err(e) => panic!("exec errored: {e}"),
            DispatchOutcome::NotHandled => panic!("exec should be handled"),
        }

        // alias returns its target
        match command_dispatch(temp.path(), "/gp", "") {
            DispatchOutcome::Ok(v) => {
                assert_eq!(v["type"], "alias");
                assert_eq!(v["target"], "git push");
            }
            other => panic!("alias not handled: {}", matches!(other, DispatchOutcome::Ok(_))),
        }

        // a non-quick command falls through to the python helper
        assert!(matches!(
            command_dispatch(temp.path(), "queue", "do a thing"),
            DispatchOutcome::NotHandled
        ));
        // an unknown skill/plugin name also falls through
        assert!(matches!(
            command_dispatch(temp.path(), "some-skill", ""),
            DispatchOutcome::NotHandled
        ));
    }

    #[test]
    fn dispatch_exec_nonzero_exit_errors() {
        let temp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("config.yaml"),
            "quick_commands:\n  fail:\n    type: exec\n    command: \"echo oops 1>&2; exit 3\"\n",
        )
        .unwrap();
        match command_dispatch(temp.path(), "fail", "") {
            DispatchOutcome::Err(message) => assert!(message.contains("oops")),
            other => panic!("expected Err, got Ok={}", matches!(other, DispatchOutcome::Ok(_))),
        }
    }

    #[test]
    fn catalog_includes_quick_commands_and_skills() {
        let mut quick = Map::new();
        quick.insert(
            "deploy".to_string(),
            json!({"type": "exec", "command": "make deploy"}),
        );
        quick.insert(
            "gp".to_string(),
            json!({"type": "alias", "target": "git push"}),
        );
        let skills = vec![("/triage".to_string(), "Triage incoming issues".to_string())];
        let cat = commands_catalog(&quick, &skills);

        let pairs = cat["pairs"].as_array().unwrap();
        let find = |name: &str| {
            pairs
                .iter()
                .find(|p| p[0].as_str() == Some(name))
                .map(|p| p[1].as_str().unwrap_or_default().to_string())
        };
        // exec/alias quick commands render their derived descriptions
        assert_eq!(find("/deploy").as_deref(), Some("exec: make deploy"));
        assert_eq!(find("/gp").as_deref(), Some("alias -> git push"));
        // quick commands are canonicalized
        assert_eq!(cat["canon"]["/deploy"], "/deploy");
        // skills appear and are counted
        assert_eq!(cat["skill_count"], 1);
        assert_eq!(find("/triage").as_deref(), Some("Triage incoming issues"));
        // quick commands form their own category bucket
        let categories = cat["categories"].as_array().unwrap();
        assert!(categories
            .iter()
            .any(|c| c["name"] == "User commands"));
    }

    #[test]
    fn catalog_excludes_hidden_and_gateway_only() {
        let cat = commands_catalog(&Map::new(), &[]);
        let pairs = cat["pairs"].as_array().unwrap();
        let names: Vec<&str> = pairs
            .iter()
            .filter_map(|p| p[0].as_str())
            .collect();
        // hidden
        assert!(!names.contains(&"/new"));
        assert!(!names.contains(&"/quit"));
        // gateway_only
        assert!(!names.contains(&"/topic"));
        assert!(!names.contains(&"/restart"));
        // present
        assert!(names.contains(&"/retry"));
        // TUI extra
        assert!(names.contains(&"/compact"));
        // canon maps aliases
        assert_eq!(cat["canon"]["/fork"], "/branch");
    }
}
