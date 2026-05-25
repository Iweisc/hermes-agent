use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};

use crate::{HermesContext, LoadedConfig};

const DEFAULT_TIMEOUT_SECONDS: u64 = 60;
const MAX_TIMEOUT_SECONDS: u64 = 300;
const ALLOWLIST_FILENAME: &str = "shell-hooks-allowlist.json";

#[derive(Debug, Clone)]
pub struct ShellHookRunner {
    specs: Vec<ShellHookSpec>,
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

impl ShellHookRunner {
    pub fn load_for_runtime(context: &HermesContext, loaded: &LoadedConfig) -> Option<Self> {
        let mut specs = iter_configured_hooks(loaded);
        if specs.is_empty() {
            return None;
        }

        let effective_accept = env_truthy("HERMES_ACCEPT_HOOKS")
            || raw_config_truthy(&loaded.raw, "hooks_auto_accept");
        let mut allowlist = match load_allowlist(context) {
            Ok(data) => data,
            Err(error) => {
                log::warn!("shell hook allowlist load failed: {error}");
                AllowlistFile::default()
            }
        };
        let mut allowlist_changed = false;

        specs.retain(|spec| {
            if allowlist_entry_for(&allowlist, &spec.event, &spec.command).is_some() {
                return true;
            }
            if !effective_accept {
                return false;
            }
            allowlist.approvals.push(AllowlistEntry {
                event: spec.event.clone(),
                command: spec.command.clone(),
                approved_at: Some(Utc::now().to_rfc3339().replace("+00:00", "Z")),
                script_mtime_at_approval: script_mtime_iso(&spec.command),
            });
            allowlist_changed = true;
            true
        });

        if allowlist_changed && let Err(error) = save_allowlist(context, &allowlist) {
            log::warn!("shell hook allowlist save failed: {error}");
        }

        (!specs.is_empty()).then_some(Self { specs })
    }

    pub fn pre_tool_call(
        &self,
        tool_name: &str,
        args: &JsonValue,
        session_id: Option<&str>,
        cwd: &Path,
    ) -> Option<String> {
        let payload = tool_payload(tool_name, args, session_id, None, None);
        for spec in self
            .specs
            .iter()
            .filter(|spec| spec.event == "pre_tool_call" && matches_tool(spec, Some(tool_name)))
        {
            let result = run_once(spec, &payload, cwd);
            if let Some(message) = block_message(&result) {
                return Some(message);
            }
        }
        None
    }

    pub fn post_tool_call(
        &self,
        tool_name: &str,
        args: &JsonValue,
        session_id: Option<&str>,
        cwd: &Path,
        result: &str,
        duration_ms: u64,
    ) {
        let payload = tool_payload(
            tool_name,
            args,
            session_id,
            Some(result.to_string()),
            Some(duration_ms),
        );
        for spec in self
            .specs
            .iter()
            .filter(|spec| spec.event == "post_tool_call" && matches_tool(spec, Some(tool_name)))
        {
            let _ = run_once(spec, &payload, cwd);
        }
    }
}

fn tool_payload(
    tool_name: &str,
    args: &JsonValue,
    session_id: Option<&str>,
    result: Option<String>,
    duration_ms: Option<u64>,
) -> JsonMap<String, JsonValue> {
    let mut payload = JsonMap::new();
    payload.insert(
        "tool_name".to_string(),
        JsonValue::String(tool_name.to_string()),
    );
    payload.insert(
        "args".to_string(),
        if args.is_object() {
            args.clone()
        } else {
            JsonValue::Null
        },
    );
    if let Some(session_id) = session_id {
        payload.insert(
            "session_id".to_string(),
            JsonValue::String(session_id.to_string()),
        );
    }
    if let Some(result) = result {
        payload.insert("result".to_string(), JsonValue::String(result));
    }
    if let Some(duration_ms) = duration_ms {
        payload.insert("duration_ms".to_string(), json!(duration_ms));
    }
    payload
}

fn block_message(result: &SpawnResult) -> Option<String> {
    if let Some(error) = result.error.as_deref() {
        log::warn!("shell hook execution failed: {error}");
    }
    if result.timed_out {
        log::warn!("shell hook timed out after {:.3}s", result.elapsed_seconds);
    }
    result
        .parsed
        .as_ref()
        .and_then(|value| value.get("message"))
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
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
        if !matches!(event, "pre_tool_call" | "post_tool_call") {
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
            specs.push(ShellHookSpec {
                event: event.to_string(),
                command: command.to_string(),
                compiled_matcher: matcher
                    .as_deref()
                    .and_then(|pattern| Regex::new(pattern).ok()),
                matcher,
                timeout,
            });
        }
    }
    specs
}

fn run_once(spec: &ShellHookSpec, payload: &JsonMap<String, JsonValue>, cwd: &Path) -> SpawnResult {
    let stdin = match serialize_payload(&spec.event, payload, cwd) {
        Ok(stdin) => stdin,
        Err(error) => {
            return SpawnResult {
                returncode: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                elapsed_seconds: 0.0,
                error: Some(error.to_string()),
                parsed: None,
            };
        }
    };
    let mut result = match spawn(spec, &stdin) {
        Ok(result) => result,
        Err(error) => SpawnResult {
            returncode: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            elapsed_seconds: 0.0,
            error: Some(error.to_string()),
            parsed: None,
        },
    };
    result.parsed = parse_response(&spec.event, &result.stdout);
    result
}

fn spawn(spec: &ShellHookSpec, stdin_json: &str) -> Result<SpawnResult, String> {
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

    if let Some(mut stdin) = child.stdin.take()
        && let Err(error) = stdin.write_all(stdin_json.as_bytes())
    {
        return Ok(SpawnResult {
            returncode: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            error: Some(error.to_string()),
            parsed: None,
        });
    }

    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            let mut stdout = String::new();
            let mut stderr = String::new();
            if let Some(mut handle) = child.stdout.take() {
                let _ = handle.read_to_string(&mut stdout);
            }
            if let Some(mut handle) = child.stderr.take() {
                let _ = handle.read_to_string(&mut stderr);
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
            let output = child
                .wait_with_output()
                .map_err(|error| error.to_string())?;
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
    cwd: &Path,
) -> Result<String, serde_json::Error> {
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
    let mut extra = JsonMap::new();
    for (key, value) in payload {
        if !matches!(
            key.as_str(),
            "tool_name" | "args" | "session_id" | "parent_session_id"
        ) {
            extra.insert(key.clone(), value.clone());
        }
    }
    serde_json::to_string(&json!({
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": session_id,
        "cwd": cwd.display().to_string(),
        "extra": extra,
    }))
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
    }
    None
}

fn load_allowlist(context: &HermesContext) -> Result<AllowlistFile, String> {
    let path = allowlist_path(context);
    if !path.exists() {
        return Ok(AllowlistFile::default());
    }
    let raw = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut parsed = serde_json::from_str::<AllowlistFile>(&raw).unwrap_or_default();
    parsed
        .approvals
        .retain(|entry| !entry.event.trim().is_empty() && !entry.command.trim().is_empty());
    Ok(parsed)
}

fn save_allowlist(context: &HermesContext, data: &AllowlistFile) -> Result<(), String> {
    let path = allowlist_path(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let rendered = serde_json::to_string_pretty(data).map_err(|error| error.to_string())?;
    atomic_write(&path, rendered.as_bytes())
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
    Some(
        chrono::DateTime::<Utc>::from(modified)
            .to_rfc3339()
            .replace("+00:00", "Z"),
    )
}

fn command_script_path(command: &str) -> Option<PathBuf> {
    let argv = shell_words::split(command).ok()?;
    if argv.is_empty() {
        return None;
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
    if let Some(regex) = spec.compiled_matcher.as_ref() {
        return regex.is_match(tool_name)
            && regex
                .find(tool_name)
                .is_some_and(|value| value.as_str() == tool_name);
    }
    tool_name == matcher
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_u64(value: &YamlValue) -> Option<u64> {
    match value {
        YamlValue::Number(number) => number.as_u64().or_else(|| {
            number
                .as_i64()
                .and_then(|value| (value >= 0).then_some(value as u64))
        }),
        YamlValue::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn raw_config_truthy(root: &YamlValue, key: &str) -> bool {
    let Some(mapping) = root.as_mapping() else {
        return false;
    };
    mapping_value(mapping, key).is_some_and(yaml_truthy)
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

fn env_truthy(key: &str) -> bool {
    std::env::var(key).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), String> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents).map_err(|error| error.to_string())?;
    fs::rename(&tmp, path).map_err(|error| error.to_string())?;
    Ok(())
}
