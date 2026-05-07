use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Subcommand;
use hermes_core::{HermesContext, LoadedConfig};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};

const DEFAULT_TIMEOUT_SECONDS: u64 = 60;
const MAX_TIMEOUT_SECONDS: u64 = 300;
const ALLOWLIST_FILENAME: &str = "shell-hooks-allowlist.json";
const VALID_HOOKS: &[&str] = &[
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "on_session_start",
    "post_api_request",
    "post_llm_call",
    "post_tool_call",
    "pre_api_request",
    "pre_gateway_dispatch",
    "pre_llm_call",
    "pre_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_stop",
    "transform_terminal_output",
    "transform_tool_result",
];

#[derive(Subcommand, Debug)]
pub enum HooksCommand {
    #[command(alias = "ls")]
    List,
    Test {
        event: String,
        #[arg(long = "for-tool")]
        for_tool: Option<String>,
        #[arg(long = "payload-file")]
        payload_file: Option<PathBuf>,
    },
    #[command(alias = "remove", alias = "rm")]
    Revoke {
        command: String,
    },
    Doctor,
}

#[derive(Debug, Clone)]
struct ShellHookSpec {
    event: String,
    command: String,
    matcher: Option<String>,
    compiled_matcher: Option<Regex>,
    timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AllowlistFile {
    #[serde(default)]
    approvals: Vec<AllowlistEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllowlistEntry {
    event: String,
    command: String,
    approved_at: Option<String>,
    script_mtime_at_approval: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpawnResult {
    returncode: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
    elapsed_seconds: f64,
    error: Option<String>,
    parsed: Option<JsonValue>,
}

pub fn print_hooks(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<HooksCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        Some(HooksCommand::List) => print_list(context, loaded),
        Some(HooksCommand::Test {
            event,
            for_tool,
            payload_file,
        }) => print_test(
            context,
            loaded,
            &event,
            for_tool.as_deref(),
            payload_file.as_deref(),
        ),
        Some(HooksCommand::Revoke { command }) => print_revoke(context, &command),
        Some(HooksCommand::Doctor) => print_doctor(context, loaded),
        None => {
            println!("Usage: hermes hooks {{list|test|revoke|doctor}}");
            println!("Run 'hermes hooks --help' for details.");
            Ok(())
        }
    }
}

fn print_list(context: &HermesContext, loaded: &LoadedConfig) -> Result<(), Box<dyn Error>> {
    let specs = iter_configured_hooks(loaded);
    if specs.is_empty() {
        println!("No shell hooks configured in ~/.hermes/config.yaml.");
        return Ok(());
    }

    let allowlist = load_allowlist(context)?;
    let approved = allowlist
        .approvals
        .iter()
        .map(|entry| ((entry.event.clone(), entry.command.clone()), entry))
        .collect::<BTreeMap<_, _>>();

    let mut by_event = BTreeMap::<String, Vec<&ShellHookSpec>>::new();
    for spec in &specs {
        by_event.entry(spec.event.clone()).or_default().push(spec);
    }

    println!("Configured shell hooks ({} total):\n", specs.len());
    for (event, entries) in by_event {
        println!("  [{event}]");
        for spec in entries {
            let key = (spec.event.clone(), spec.command.clone());
            let approved_entry = approved.get(&key);
            let status = if approved_entry.is_some() {
                "allowed"
            } else {
                "not allowlisted"
            };
            let matcher = spec
                .matcher
                .as_deref()
                .map(|value| format!(" matcher={value:?}"))
                .unwrap_or_default();
            println!(
                "    - {}{} (timeout={}s, {})",
                spec.command, matcher, spec.timeout, status
            );
            if let Some(entry) = approved_entry {
                if let Some(value) = entry.approved_at.as_deref() {
                    println!("      approved_at: {value}");
                }
                let now = script_mtime_iso(&spec.command);
                if let (Some(current), Some(old)) =
                    (now.as_deref(), entry.script_mtime_at_approval.as_deref())
                    && current > old
                {
                    println!("      script modified since approval (was {old}, now {current})");
                }
            }
        }
        println!();
    }
    Ok(())
}

fn print_test(
    _context: &HermesContext,
    loaded: &LoadedConfig,
    event: &str,
    for_tool: Option<&str>,
    payload_file: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let normalized = event.trim();
    if !VALID_HOOKS.contains(&normalized) {
        return Err(format!(
            "Unknown event: {normalized:?}. Valid events: {}",
            VALID_HOOKS.join(", ")
        )
        .into());
    }

    let mut payload = default_payload(normalized);
    if let Some(tool_name) = for_tool {
        payload.insert(
            "tool_name".to_string(),
            JsonValue::String(tool_name.to_string()),
        );
    }
    if let Some(path) = payload_file {
        let raw = fs::read_to_string(path)?;
        let custom = serde_json::from_str::<JsonValue>(&raw)?;
        let Some(extra) = custom.as_object() else {
            return Err(format!("{} must contain a JSON object", path.display()).into());
        };
        for (key, value) in extra {
            payload.insert(key.clone(), value.clone());
        }
    }

    let mut specs = iter_configured_hooks(loaded)
        .into_iter()
        .filter(|spec| spec.event == normalized)
        .collect::<Vec<_>>();
    if let Some(tool_name) = for_tool {
        specs.retain(|spec| matches_tool(spec, Some(tool_name)));
    }
    if specs.is_empty() {
        println!("No shell hooks configured for event: {normalized}");
        return Ok(());
    }

    println!(
        "Firing {} hook(s) for event '{}':\n",
        specs.len(),
        normalized
    );
    for spec in specs {
        println!("  -> {}", spec.command);
        let result = run_once(&spec, &payload)?;
        print_run_result(&result);
        println!();
    }
    Ok(())
}

fn print_revoke(context: &HermesContext, command: &str) -> Result<(), Box<dyn Error>> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("command cannot be empty".into());
    }
    let removed = revoke_allowlist(context, trimmed)?;
    if removed == 0 {
        println!("No allowlist entry found for command: {trimmed}");
    } else {
        println!("Removed {removed} allowlist entry/entries for: {trimmed}");
    }
    Ok(())
}

fn print_doctor(context: &HermesContext, loaded: &LoadedConfig) -> Result<(), Box<dyn Error>> {
    let specs = iter_configured_hooks(loaded);
    if specs.is_empty() {
        println!("No shell hooks configured - nothing to check.");
        return Ok(());
    }

    println!("Checking {} configured shell hook(s)...\n", specs.len());
    let mut problems = 0_usize;
    for spec in specs {
        println!("  [{}] {}", spec.event, spec.command);
        problems += doctor_one(context, &spec)?;
        println!();
    }

    if problems == 0 {
        println!("All shell hooks look healthy.");
    } else {
        println!("{problems} issue(s) found. Fix before relying on these hooks.");
    }
    Ok(())
}

fn doctor_one(context: &HermesContext, spec: &ShellHookSpec) -> Result<usize, Box<dyn Error>> {
    let mut problems = 0_usize;
    if script_is_executable(&spec.command) {
        println!("      script exists and is executable");
    } else {
        problems += 1;
        println!("      script missing or not executable");
    }

    let allowlist = load_allowlist(context)?;
    let entry = allowlist_entry_for(&allowlist, &spec.event, &spec.command);
    if let Some(entry) = entry.as_ref() {
        println!(
            "      allowlisted (approved {})",
            entry.approved_at.as_deref().unwrap_or("?")
        );
        if let (Some(old), Some(current)) = (
            entry.script_mtime_at_approval.as_deref(),
            script_mtime_iso(&spec.command).as_deref(),
        ) {
            if current > old {
                problems += 1;
                println!("      script modified since approval (was {old}, now {current})");
            } else if current == old {
                println!("      script unchanged since approval");
            }
        }
    } else {
        problems += 1;
        println!("      not allowlisted");
    }

    if entry.is_none() {
        println!("      skipped JSON smoke test - not allowlisted yet");
        return Ok(problems);
    }
    if !script_is_executable(&spec.command) {
        return Ok(problems);
    }

    let payload = default_payload(&spec.event);
    let result = run_once(spec, &payload)?;
    if result.timed_out {
        problems += 1;
        println!(
            "      timed out after {:.3}s on synthetic payload (timeout={}s)",
            result.elapsed_seconds, spec.timeout
        );
        return Ok(problems);
    }
    if let Some(error) = result.error.as_deref() {
        problems += 1;
        println!("      execution error: {error}");
        return Ok(problems);
    }

    let stdout = result.stdout.trim();
    if stdout.is_empty() {
        println!(
            "      ran clean with empty stdout (exit={:?}, {:.3}s)",
            result.returncode, result.elapsed_seconds
        );
        return Ok(problems);
    }

    if result.parsed.is_some() {
        println!(
            "      produced valid JSON on synthetic payload (exit={:?}, {:.3}s)",
            result.returncode, result.elapsed_seconds
        );
    } else {
        problems += 1;
        println!(
            "      stdout was not valid hook JSON (exit={:?}, {:.3}s): {}",
            result.returncode,
            result.elapsed_seconds,
            truncate(stdout, 120)
        );
    }
    Ok(problems)
}

fn iter_configured_hooks(loaded: &LoadedConfig) -> Vec<ShellHookSpec> {
    let Some(root) = loaded.raw.as_mapping() else {
        return Vec::new();
    };
    let Some(hooks) = mapping_value(root, "hooks").and_then(YamlValue::as_mapping) else {
        return Vec::new();
    };

    let mut specs = Vec::new();
    for (event_key, entries) in hooks {
        let Some(event) = event_key.as_str().map(str::trim) else {
            continue;
        };
        if !VALID_HOOKS.contains(&event) {
            continue;
        }
        let Some(entries) = entries.as_sequence() else {
            continue;
        };
        for entry in entries {
            let Some(entry) = entry.as_mapping() else {
                continue;
            };
            let Some(command) = mapping_value(entry, "command")
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                continue;
            };
            let matcher = mapping_value(entry, "matcher")
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let timeout = mapping_value(entry, "timeout")
                .and_then(yaml_u64)
                .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
                .clamp(1, MAX_TIMEOUT_SECONDS);
            let compiled_matcher = matcher
                .as_deref()
                .filter(|_| matches!(event, "pre_tool_call" | "post_tool_call"))
                .and_then(|pattern| Regex::new(pattern).ok());
            specs.push(ShellHookSpec {
                event: event.to_string(),
                command: command.to_string(),
                matcher,
                compiled_matcher,
                timeout,
            });
        }
    }
    specs
}

fn run_once(
    spec: &ShellHookSpec,
    payload: &JsonMap<String, JsonValue>,
) -> Result<SpawnResult, Box<dyn Error>> {
    let stdin = serialize_payload(&spec.event, payload)?;
    let mut result = spawn(spec, &stdin)?;
    result.parsed = parse_response(&spec.event, &result.stdout);
    Ok(result)
}

fn spawn(spec: &ShellHookSpec, stdin_json: &str) -> Result<SpawnResult, Box<dyn Error>> {
    let argv = shell_words::split(&spec.command)
        .map_err(|error| format!("command {:?} cannot be parsed: {}", spec.command, error))?;
    if argv.is_empty() {
        return Ok(SpawnResult {
            returncode: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            elapsed_seconds: 0.0,
            error: Some("empty command".to_string()),
            parsed: None,
        });
    }

    let started = std::time::Instant::now();
    let mut child = match Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Ok(SpawnResult {
                returncode: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                elapsed_seconds: 0.0,
                error: Some(error.to_string()),
                parsed: None,
            });
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(stdin_json.as_bytes())?;
    }

    loop {
        if let Some(status) = child.try_wait()? {
            let mut stdout = String::new();
            let mut stderr = String::new();
            if let Some(mut handle) = child.stdout.take() {
                handle.read_to_string(&mut stdout)?;
            }
            if let Some(mut handle) = child.stderr.take() {
                handle.read_to_string(&mut stderr)?;
            }
            return Ok(SpawnResult {
                returncode: status.code(),
                stdout,
                stderr,
                timed_out: false,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                error: None,
                parsed: None,
            });
        }

        if started.elapsed() >= Duration::from_secs(spec.timeout) {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            return Ok(SpawnResult {
                returncode: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                timed_out: true,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                error: None,
                parsed: None,
            });
        }
        sleep(Duration::from_millis(20));
    }
}

fn serialize_payload(
    event: &str,
    payload: &JsonMap<String, JsonValue>,
) -> Result<String, Box<dyn Error>> {
    let tool_name = payload.get("tool_name").cloned().unwrap_or(JsonValue::Null);
    let tool_input = payload
        .get("args")
        .cloned()
        .filter(|value| value.is_object())
        .unwrap_or(JsonValue::Null);
    let session_id = payload
        .get("session_id")
        .or_else(|| payload.get("parent_session_id"))
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_string();
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let mut extra = JsonMap::new();
    for (key, value) in payload {
        if !matches!(
            key.as_str(),
            "tool_name" | "args" | "session_id" | "parent_session_id"
        ) {
            extra.insert(key.clone(), value.clone());
        }
    }
    Ok(serde_json::to_string(&json!({
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": session_id,
        "cwd": cwd,
        "extra": extra,
    }))?)
}

fn parse_response(event: &str, stdout: &str) -> Option<JsonValue> {
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return None;
    }
    let data = serde_json::from_str::<JsonValue>(stdout).ok()?;
    let object = data.as_object()?;
    if event == "pre_tool_call" {
        if object.get("action").and_then(JsonValue::as_str) == Some("block") {
            let message = object
                .get("message")
                .or_else(|| object.get("reason"))
                .and_then(JsonValue::as_str)?
                .trim();
            if !message.is_empty() {
                return Some(json!({"action":"block","message":message}));
            }
        }
        if object.get("decision").and_then(JsonValue::as_str) == Some("block") {
            let message = object
                .get("reason")
                .or_else(|| object.get("message"))
                .and_then(JsonValue::as_str)?
                .trim();
            if !message.is_empty() {
                return Some(json!({"action":"block","message":message}));
            }
        }
        return None;
    }

    let context = object.get("context").and_then(JsonValue::as_str)?.trim();
    (!context.is_empty()).then(|| json!({"context": context}))
}

fn print_run_result(result: &SpawnResult) {
    if let Some(error) = result.error.as_deref() {
        println!("      error: {error}");
        return;
    }
    if result.timed_out {
        println!("      timed out after {:.3}s", result.elapsed_seconds);
        return;
    }
    println!(
        "      exit={:?}  elapsed={:.3}s",
        result.returncode, result.elapsed_seconds
    );
    let stdout = result.stdout.trim();
    let stderr = result.stderr.trim();
    if !stdout.is_empty() {
        println!("      stdout: {}", truncate(stdout, 400));
    }
    if !stderr.is_empty() {
        println!("      stderr: {}", truncate(stderr, 400));
    }
    if let Some(parsed) = result.parsed.as_ref() {
        println!(
            "      parsed: {}",
            serde_json::to_string(parsed).unwrap_or_default()
        );
    } else {
        println!("      parsed: <none>");
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
    }
}

fn load_allowlist(context: &HermesContext) -> Result<AllowlistFile, Box<dyn Error>> {
    let path = allowlist_path(context);
    if !path.exists() {
        return Ok(AllowlistFile::default());
    }
    let raw = fs::read_to_string(path)?;
    let mut parsed = serde_json::from_str::<AllowlistFile>(&raw).unwrap_or_default();
    parsed
        .approvals
        .retain(|entry| !entry.event.trim().is_empty() && !entry.command.trim().is_empty());
    Ok(parsed)
}

fn save_allowlist(context: &HermesContext, data: &AllowlistFile) -> Result<(), Box<dyn Error>> {
    let path = allowlist_path(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(data)?;
    atomic_write(&path, rendered.as_bytes())
}

fn revoke_allowlist(context: &HermesContext, command: &str) -> Result<usize, Box<dyn Error>> {
    let mut data = load_allowlist(context)?;
    let before = data.approvals.len();
    data.approvals.retain(|entry| entry.command != command);
    let removed = before.saturating_sub(data.approvals.len());
    if removed > 0 {
        save_allowlist(context, &data)?;
    }
    Ok(removed)
}

fn allowlist_entry_for(
    allowlist: &AllowlistFile,
    event: &str,
    command: &str,
) -> Option<AllowlistEntry> {
    allowlist
        .approvals
        .iter()
        .find(|entry| entry.event == event && entry.command == command)
        .cloned()
}

fn allowlist_path(context: &HermesContext) -> PathBuf {
    context.hermes_home().join(ALLOWLIST_FILENAME)
}

fn script_mtime_iso(command: &str) -> Option<String> {
    let path = command_script_path(command)?;
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let timestamp = chrono::DateTime::<chrono::Utc>::from(modified);
    Some(timestamp.to_rfc3339().replace("+00:00", "Z"))
}

fn script_is_executable(command: &str) -> bool {
    let Some(path) = command_script_path(command) else {
        return false;
    };
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(_) => return false,
    };
    if !metadata.is_file() {
        return false;
    }
    let argv = match shell_words::split(command) {
        Ok(argv) => argv,
        Err(_) => return false,
    };
    let is_bare = argv
        .first()
        .is_some_and(|value| value == path.to_string_lossy().as_ref());
    if is_bare {
        #[cfg(unix)]
        {
            return metadata.permissions().mode() & 0o111 != 0;
        }
        #[cfg(not(unix))]
        {
            return true;
        }
    }
    true
}

fn command_script_path(command: &str) -> Option<PathBuf> {
    let argv = shell_words::split(command).ok()?;
    if argv.is_empty() {
        return None;
    }
    const SCRIPT_EXTENSIONS: &[&str] = &[
        ".sh", ".bash", ".zsh", ".fish", ".py", ".pyw", ".rb", ".pl", ".lua", ".js", ".mjs",
        ".cjs", ".ts",
    ];
    for part in &argv {
        if SCRIPT_EXTENSIONS
            .iter()
            .any(|suffix| part.to_ascii_lowercase().ends_with(suffix))
        {
            return Some(expand_home(part));
        }
    }
    for part in &argv {
        if part.contains('/') || part.starts_with('~') {
            return Some(expand_home(part));
        }
    }
    Some(expand_home(&argv[0]))
}

fn expand_home(path: &str) -> PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(rest);
    }
    PathBuf::from(path)
}

fn matches_tool(spec: &ShellHookSpec, tool_name: Option<&str>) -> bool {
    let Some(matcher) = spec.matcher.as_deref() else {
        return true;
    };
    let Some(tool_name) = tool_name else {
        return false;
    };
    if spec.event != "pre_tool_call" && spec.event != "post_tool_call" {
        return true;
    }
    if let Some(regex) = spec.compiled_matcher.as_ref() {
        return regex.is_match(tool_name)
            && regex
                .find(tool_name)
                .is_some_and(|m| m.as_str() == tool_name);
    }
    tool_name == matcher
}

fn default_payload(event: &str) -> JsonMap<String, JsonValue> {
    let value = match event {
        "pre_tool_call" => json!({
            "tool_name": "terminal",
            "args": {"command": "echo hello"},
            "session_id": "test-session",
            "task_id": "test-task",
            "tool_call_id": "test-call"
        }),
        "post_tool_call" => json!({
            "tool_name": "terminal",
            "args": {"command": "echo hello"},
            "session_id": "test-session",
            "task_id": "test-task",
            "tool_call_id": "test-call",
            "result": "{\"output\":\"hello\"}",
            "duration_ms": 42
        }),
        "pre_llm_call" => json!({
            "session_id": "test-session",
            "user_message": "What is the weather?",
            "conversation_history": [],
            "is_first_turn": true,
            "model": "gpt-4",
            "platform": "cli"
        }),
        "post_llm_call" => json!({
            "session_id": "test-session",
            "model": "gpt-4",
            "platform": "cli"
        }),
        "on_session_start" | "on_session_end" | "on_session_finalize" | "on_session_reset" => {
            json!({"session_id": "test-session"})
        }
        "pre_api_request" => json!({
            "session_id": "test-session",
            "task_id": "test-task",
            "platform": "cli",
            "model": "claude-sonnet-4-6",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "api_call_count": 1,
            "message_count": 4,
            "tool_count": 12,
            "approx_input_tokens": 2048,
            "request_char_count": 8192,
            "max_tokens": 4096
        }),
        "post_api_request" => json!({
            "session_id": "test-session",
            "task_id": "test-task",
            "platform": "cli",
            "model": "claude-sonnet-4-6",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "api_call_count": 1,
            "api_duration": 1.234,
            "finish_reason": "stop",
            "message_count": 4,
            "response_model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 2048, "output_tokens": 512},
            "assistant_content_chars": 1200,
            "assistant_tool_call_count": 0
        }),
        "subagent_stop" => json!({
            "parent_session_id": "parent-sess",
            "child_role": null,
            "child_summary": "Synthetic summary for hooks test",
            "child_status": "completed",
            "duration_ms": 1234
        }),
        _ => json!({"session_id": "test-session"}),
    };
    value.as_object().cloned().unwrap_or_default()
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_u64(value: &YamlValue) -> Option<u64> {
    match value {
        YamlValue::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|v| (v >= 0).then_some(v as u64))),
        YamlValue::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn temp_context() -> (TempDir, HermesContext, LoadedConfig) {
        let dir = TempDir::new().unwrap();
        let context =
            HermesContext::new(dir.path()).with_hermes_home_env(Some(dir.path().join(".hermes")));
        fs::create_dir_all(context.hermes_home()).unwrap();
        let raw = serde_yaml::from_str::<YamlValue>(
            r#"
hooks:
  pre_tool_call:
    - command: "~/hooks/block.sh"
      matcher: "^terminal$"
      timeout: 12
  pre_llm_call:
    - command: "~/hooks/context.py"
"#,
        )
        .unwrap();
        let loaded = LoadedConfig {
            path: context.config_path(),
            raw,
            config: hermes_core::HermesConfig::default(),
            warnings: Vec::new(),
        };
        (dir, context, loaded)
    }

    #[test]
    fn parses_hooks_from_raw_config() {
        let (_dir, _context, loaded) = temp_context();
        let specs = iter_configured_hooks(&loaded);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].event, "pre_tool_call");
        assert_eq!(specs[0].matcher.as_deref(), Some("^terminal$"));
        assert_eq!(specs[0].timeout, 12);
    }

    #[test]
    fn tool_matcher_supports_regex() {
        let spec = ShellHookSpec {
            event: "pre_tool_call".to_string(),
            command: "echo hi".to_string(),
            matcher: Some("^term.*$".to_string()),
            compiled_matcher: Some(Regex::new("^term.*$").unwrap()),
            timeout: 5,
        };
        assert!(matches_tool(&spec, Some("terminal")));
        assert!(!matches_tool(&spec, Some("read_file")));
    }

    #[test]
    fn parse_response_translates_block_shape() {
        let parsed = parse_response("pre_tool_call", r#"{"decision":"block","reason":"Nope"}"#);
        assert_eq!(parsed, Some(json!({"action":"block","message":"Nope"})));
    }

    #[test]
    fn revoke_removes_matching_entries() {
        let (_dir, context, _loaded) = temp_context();
        let data = AllowlistFile {
            approvals: vec![
                AllowlistEntry {
                    event: "pre_tool_call".to_string(),
                    command: "a".to_string(),
                    approved_at: None,
                    script_mtime_at_approval: None,
                },
                AllowlistEntry {
                    event: "pre_llm_call".to_string(),
                    command: "b".to_string(),
                    approved_at: None,
                    script_mtime_at_approval: None,
                },
            ],
        };
        save_allowlist(&context, &data).unwrap();
        assert_eq!(revoke_allowlist(&context, "a").unwrap(), 1);
        let loaded = load_allowlist(&context).unwrap();
        assert_eq!(loaded.approvals.len(), 1);
        assert_eq!(loaded.approvals[0].command, "b");
    }

    #[test]
    fn run_once_executes_script_and_parses_context() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("hook.sh");
        fs::write(
            &script,
            "#!/usr/bin/env bash\ncat >/dev/null\necho '{\"context\":\"extra\"}'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let spec = ShellHookSpec {
            event: "pre_llm_call".to_string(),
            command: script.display().to_string(),
            matcher: None,
            compiled_matcher: None,
            timeout: 2,
        };
        let payload = default_payload("pre_llm_call");
        let result = run_once(&spec, &payload).unwrap();
        assert_eq!(result.returncode, Some(0));
        assert_eq!(result.parsed, Some(json!({"context":"extra"})));
    }
}
