//! hermes hooks — inspect and manage shell-script hooks.
//!
//! Native Rust port of `hermes_cli/hooks.py`.
//!
//! Usage:
//!
//! ```text
//! hermes hooks list
//! hermes hooks test <event> [--for-tool X] [--payload-file F]
//! hermes hooks revoke <command>
//! hermes hooks doctor
//! ```
//!
//! Consent records live under `~/.hermes/shell-hooks-allowlist.json` and hook
//! definitions come from the `hooks:` block in `~/.hermes/config.yaml` (the
//! same config read by the CLI / gateway at startup).
//!
//! This module is a thin CLI shell over the shared shell-hooks logic; every
//! shared concern (payload serialisation, response parsing, allowlist format)
//! is reproduced here faithfully from `agent/shell_hooks.py` so the module is
//! self-contained.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_TIMEOUT_SECONDS: u64 = 60;
const MAX_TIMEOUT_SECONDS: u64 = 300;
const ALLOWLIST_FILENAME: &str = "shell-hooks-allowlist.json";

/// Valid hook event names — mirrors `hermes_cli.plugins.VALID_HOOKS`.
pub const VALID_HOOKS: &[&str] = &[
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_stop",
    "pre_gateway_dispatch",
    "pre_approval_request",
    "post_approval_response",
];

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// Parsed `hermes hooks` invocation — analogous to the `args` object the
/// Python `hooks_command` consumes.
#[derive(Debug, Clone, Default)]
pub struct HooksArgs {
    /// `list` | `ls` | `test` | `revoke` | `remove` | `rm` | `doctor`, or
    /// `None` when no subcommand was supplied.
    pub hooks_action: Option<String>,
    /// `test` event name.
    pub event: Option<String>,
    /// `--for-tool` matcher override.
    pub for_tool: Option<String>,
    /// `--payload-file` path.
    pub payload_file: Option<PathBuf>,
    /// `revoke` command string.
    pub command: Option<String>,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ShellHookSpec {
    pub event: String,
    pub command: String,
    pub matcher: Option<String>,
    pub timeout: u64,
    compiled_matcher: Option<Regex>,
}

impl ShellHookSpec {
    /// Mirrors `ShellHookSpec.matches_tool` (regex full-match semantics).
    pub fn matches_tool(&self, tool_name: Option<&str>) -> bool {
        let Some(matcher) = self.matcher.as_deref() else {
            return true;
        };
        let Some(tool_name) = tool_name else {
            return false;
        };
        if let Some(regex) = self.compiled_matcher.as_ref() {
            return regex
                .find(tool_name)
                .is_some_and(|m| m.start() == 0 && m.end() == tool_name.len());
        }
        tool_name == matcher
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    approved_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    script_mtime_at_approval: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub returncode: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
    pub parsed: Option<JsonValue>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point for `hermes hooks` — dispatches to the requested action.
///
/// `config` is the parsed `~/.hermes/config.yaml` (the same shape
/// `load_config()` returns). `hermes_home` is the directory the allowlist
/// lives under (typically `~/.hermes`).
pub fn hooks_command(args: &HooksArgs, config: &YamlValue, hermes_home: &Path) {
    let sub = args.hooks_action.as_deref();

    let Some(sub) = sub.filter(|value| !value.is_empty()) else {
        println!("Usage: hermes hooks {{list|test|revoke|doctor}}");
        println!("Run 'hermes hooks --help' for details.");
        return;
    };

    match sub {
        "list" | "ls" => cmd_list(config, hermes_home),
        "test" => cmd_test(args, config),
        "revoke" | "remove" | "rm" => cmd_revoke(args, hermes_home),
        "doctor" => cmd_doctor(config, hermes_home),
        other => println!("Unknown hooks subcommand: {other}"),
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

fn cmd_list(config: &YamlValue, hermes_home: &Path) {
    let specs = iter_configured_hooks(config);

    if specs.is_empty() {
        println!("No shell hooks configured in ~/.hermes/config.yaml.");
        println!("See `hermes hooks --help` or");
        println!("    website/docs/user-guide/features/hooks.md");
        println!("for the config schema and worked examples.");
        return;
    }

    let mut by_event: BTreeMap<String, Vec<&ShellHookSpec>> = BTreeMap::new();
    for spec in &specs {
        by_event.entry(spec.event.clone()).or_default().push(spec);
    }

    let allowlist = load_allowlist(hermes_home);
    let approved: std::collections::HashSet<(String, String)> = allowlist
        .approvals
        .iter()
        .map(|entry| (entry.event.clone(), entry.command.clone()))
        .collect();

    println!("Configured shell hooks ({} total):\n", specs.len());

    for (event, entries) in &by_event {
        println!("  [{event}]");
        for spec in entries {
            let key = (spec.event.clone(), spec.command.clone());
            let is_approved = approved.contains(&key);
            let status = if is_approved {
                "\u{2713} allowed"
            } else {
                "\u{2717} not allowlisted"
            };
            let matcher_part = spec
                .matcher
                .as_deref()
                .map(|value| format!(" matcher={}", py_repr(value)))
                .unwrap_or_default();
            println!(
                "    - {}{} (timeout={}s, {})",
                spec.command, matcher_part, spec.timeout, status
            );

            if is_approved {
                if let Some(entry) = allowlist_entry_for(&allowlist, &spec.event, &spec.command) {
                    if let Some(approved_at) = entry.approved_at.as_deref() {
                        println!("      approved_at: {approved_at}");
                        let mtime_now = script_mtime_iso(&spec.command);
                        let mtime_at = entry.script_mtime_at_approval.as_deref();
                        if let (Some(now), Some(at)) = (mtime_now.as_deref(), mtime_at) {
                            if now > at {
                                println!(
                                    "      \u{26a0} script modified since approval (was {at}, now {now}) — run `hermes hooks doctor` to re-validate"
                                );
                            }
                        }
                    }
                }
            }
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

fn default_payload(event: &str) -> JsonMap<String, JsonValue> {
    let value = match event {
        "pre_tool_call" => json!({
            "tool_name": "terminal",
            "args": {"command": "echo hello"},
            "session_id": "test-session",
            "task_id": "test-task",
            "tool_call_id": "test-call",
        }),
        "post_tool_call" => json!({
            "tool_name": "terminal",
            "args": {"command": "echo hello"},
            "session_id": "test-session",
            "task_id": "test-task",
            "tool_call_id": "test-call",
            "result": "{\"output\": \"hello\"}",
            "duration_ms": 42,
        }),
        "pre_llm_call" => json!({
            "session_id": "test-session",
            "user_message": "What is the weather?",
            "conversation_history": [],
            "is_first_turn": true,
            "model": "gpt-4",
            "platform": "cli",
        }),
        "post_llm_call" => json!({
            "session_id": "test-session",
            "model": "gpt-4",
            "platform": "cli",
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
            "max_tokens": 4096,
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
            "assistant_tool_call_count": 0,
        }),
        "subagent_stop" => json!({
            "parent_session_id": "parent-sess",
            "child_role": null,
            "child_summary": "Synthetic summary for hooks test",
            "child_status": "completed",
            "duration_ms": 1234,
        }),
        _ => json!({"session_id": "test-session"}),
    };
    value.as_object().cloned().unwrap_or_default()
}

/// True when a default-payload entry exists for `event` (`_DEFAULT_PAYLOADS`
/// membership test used by doctor's smoke test fallback).
fn has_default_payload(event: &str) -> bool {
    matches!(
        event,
        "pre_tool_call"
            | "post_tool_call"
            | "pre_llm_call"
            | "post_llm_call"
            | "on_session_start"
            | "on_session_end"
            | "on_session_finalize"
            | "on_session_reset"
            | "pre_api_request"
            | "post_api_request"
            | "subagent_stop"
    )
}

fn cmd_test(args: &HooksArgs, config: &YamlValue) {
    let event = args.event.as_deref().unwrap_or("");
    if !VALID_HOOKS.contains(&event) {
        println!("Unknown event: {}", py_repr(event));
        let mut sorted: Vec<&str> = VALID_HOOKS.to_vec();
        sorted.sort_unstable();
        println!("Valid events: {}", sorted.join(", "));
        return;
    }

    // Synthetic kwargs in the same shape invoke_hook() would pass. Merged with
    // --for-tool (overrides tool_name) and --payload-file (extra kwargs).
    let mut payload = default_payload(event);

    if let Some(for_tool) = args.for_tool.as_deref() {
        payload.insert(
            "tool_name".to_string(),
            JsonValue::String(for_tool.to_string()),
        );
    }

    if let Some(path) = args.payload_file.as_deref() {
        match fs::read_to_string(path).map_err(|e| e.to_string()).and_then(
            |raw| serde_json::from_str::<JsonValue>(&raw).map_err(|e| e.to_string()),
        ) {
            Ok(custom) => {
                if let Some(map) = custom.as_object() {
                    for (key, value) in map {
                        payload.insert(key.clone(), value.clone());
                    }
                } else {
                    println!(
                        "Warning: {} is not a JSON object; ignoring",
                        path.display()
                    );
                }
            }
            Err(exc) => {
                println!("Error reading payload file: {exc}");
                return;
            }
        }
    }

    let mut specs: Vec<ShellHookSpec> = iter_configured_hooks(config)
        .into_iter()
        .filter(|spec| spec.event == event)
        .collect();

    if let Some(for_tool) = args.for_tool.as_deref() {
        specs.retain(|spec| {
            !matches!(spec.event.as_str(), "pre_tool_call" | "post_tool_call")
                || spec.matches_tool(Some(for_tool))
        });
    }

    if specs.is_empty() {
        println!("No shell hooks configured for event: {event}");
        if let Some(for_tool) = args.for_tool.as_deref() {
            println!("(with matcher filter --for-tool={for_tool})");
        }
        return;
    }

    println!("Firing {} hook(s) for event '{}':\n", specs.len(), event);
    for spec in &specs {
        println!("  \u{2192} {}", spec.command);
        let result = run_once(spec, &payload);
        print_run_result(&result);
        println!();
    }
}

fn print_run_result(result: &RunResult) {
    if let Some(error) = result.error.as_deref() {
        println!("      \u{2717} error: {error}");
        return;
    }
    if result.timed_out {
        println!(
            "      \u{2717} timed out after {}s",
            fmt_seconds(result.elapsed_seconds)
        );
        return;
    }

    println!(
        "      exit={}  elapsed={}s",
        fmt_returncode(result.returncode),
        fmt_seconds(result.elapsed_seconds)
    );

    let stdout = result.stdout.trim();
    let stderr = result.stderr.trim();
    if !stdout.is_empty() {
        println!("      stdout: {}", truncate(stdout, 400));
    }
    if !stderr.is_empty() {
        println!("      stderr: {}", truncate(stderr, 400));
    }

    match result.parsed.as_ref() {
        Some(parsed) => println!(
            "      parsed (Hermes wire shape): {}",
            serde_json::to_string(parsed).unwrap_or_default()
        ),
        None => println!("      parsed: <none — hook contributed nothing to the dispatcher>"),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let kept: String = s.chars().take(n.saturating_sub(3)).collect();
        format!("{kept}...")
    }
}

// ---------------------------------------------------------------------------
// revoke
// ---------------------------------------------------------------------------

fn cmd_revoke(args: &HooksArgs, hermes_home: &Path) {
    let command = args.command.as_deref().unwrap_or("");
    let removed = revoke(hermes_home, command);
    if removed == 0 {
        println!("No allowlist entry found for command: {command}");
        return;
    }
    println!("Removed {removed} allowlist entry/entries for: {command}");
    println!(
        "Note: currently running CLI / gateway processes keep their already-registered callbacks until they restart."
    );
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

fn cmd_doctor(config: &YamlValue, hermes_home: &Path) {
    let specs = iter_configured_hooks(config);

    if specs.is_empty() {
        println!("No shell hooks configured — nothing to check.");
        return;
    }

    println!("Checking {} configured shell hook(s)...\n", specs.len());

    let mut problems = 0_usize;
    for spec in &specs {
        println!("  [{}] {}", spec.event, spec.command);
        problems += doctor_one(spec, hermes_home);
        println!();
    }

    if problems > 0 {
        println!("{problems} issue(s) found.  Fix before relying on these hooks.");
    } else {
        println!("All shell hooks look healthy.");
    }
}

fn doctor_one(spec: &ShellHookSpec, hermes_home: &Path) -> usize {
    let mut problems = 0_usize;

    // 1. Script exists and is executable
    if script_is_executable(&spec.command) {
        println!("      \u{2713} script exists and is executable");
    } else {
        problems += 1;
        println!(
            "      \u{2717} script missing or not executable (chmod +x the file, or fix the path)"
        );
    }

    // 2. Allowlist status
    let allowlist = load_allowlist(hermes_home);
    let entry = allowlist_entry_for(&allowlist, &spec.event, &spec.command);
    if let Some(entry) = entry.as_ref() {
        println!(
            "      \u{2713} allowlisted (approved {})",
            entry.approved_at.as_deref().unwrap_or("?")
        );
    } else {
        problems += 1;
        println!(
            "      \u{2717} not allowlisted — hook will NOT fire at runtime (run with --accept-hooks once, or confirm at the TTY prompt)"
        );
    }

    // 3. Mtime drift
    if let Some(entry) = entry.as_ref() {
        if let Some(mtime_at) = entry.script_mtime_at_approval.as_deref() {
            let mtime_now = script_mtime_iso(&spec.command);
            if let Some(now) = mtime_now.as_deref() {
                if now > mtime_at {
                    problems += 1;
                    println!(
                        "      \u{26a0} script modified since approval (was {mtime_at}, now {now}) — review changes, then `hermes hooks revoke` + re-approve to refresh"
                    );
                } else if now == mtime_at {
                    println!("      \u{2713} script unchanged since approval");
                }
            }
        }
    }

    // 4. Produces valid JSON for a synthetic payload — only when the entry is
    // already allowlisted.
    if entry.is_none() {
        println!(
            "      \u{2139} skipped JSON smoke test — not allowlisted yet. Approve the hook first (via TTY prompt or --accept-hooks), then re-run `hermes hooks doctor`."
        );
    } else if script_is_executable(&spec.command) {
        let payload = if has_default_payload(&spec.event) {
            default_payload(&spec.event)
        } else {
            json!({"extra": {}}).as_object().cloned().unwrap_or_default()
        };
        let result = run_once(spec, &payload);
        if result.timed_out {
            problems += 1;
            println!(
                "      \u{2717} timed out after {}s on synthetic payload (timeout={}s)",
                fmt_seconds(result.elapsed_seconds),
                spec.timeout
            );
        } else if let Some(error) = result.error.as_deref() {
            problems += 1;
            println!("      \u{2717} execution error: {error}");
        } else {
            let rc = fmt_returncode(result.returncode);
            let elapsed = fmt_seconds(result.elapsed_seconds);
            let stdout = result.stdout.trim();
            if !stdout.is_empty() {
                if serde_json::from_str::<JsonValue>(stdout).is_ok() {
                    println!(
                        "      \u{2713} produced valid JSON on synthetic payload (exit={rc}, {elapsed}s)"
                    );
                } else {
                    problems += 1;
                    println!(
                        "      \u{2717} stdout was not valid JSON (exit={rc}, {elapsed}s): {}",
                        truncate(stdout, 120)
                    );
                }
            } else {
                println!(
                    "      \u{2713} ran clean with empty stdout (exit={rc}, {elapsed}s) — hook is observer-only"
                );
            }
        }
    }

    problems
}

// ---------------------------------------------------------------------------
// Shared shell-hooks logic (ported from agent/shell_hooks.py)
// ---------------------------------------------------------------------------

/// Parse the `hooks:` block out of the config into validated specs.
pub fn iter_configured_hooks(config: &YamlValue) -> Vec<ShellHookSpec> {
    let Some(root) = config.as_mapping() else {
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
                .min(MAX_TIMEOUT_SECONDS);
            let compiled_matcher = matcher
                .as_deref()
                .and_then(|pattern| Regex::new(pattern).ok());
            specs.push(ShellHookSpec {
                event: event.to_string(),
                command: command.to_string(),
                matcher,
                timeout,
                compiled_matcher,
            });
        }
    }
    specs
}

/// Execute one hook once with a synthetic payload, returning its result with
/// the parsed Hermes wire shape attached.
pub fn run_once(spec: &ShellHookSpec, payload: &JsonMap<String, JsonValue>) -> RunResult {
    let stdin = match serialize_payload(&spec.event, payload) {
        Ok(value) => value,
        Err(error) => {
            return RunResult {
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
    let mut result = spawn(spec, &stdin);
    result.parsed = parse_response(&spec.event, &result.stdout);
    result
}

fn spawn(spec: &ShellHookSpec, stdin_json: &str) -> RunResult {
    let argv = match shell_words::split(&spec.command) {
        Ok(argv) => argv,
        Err(error) => {
            return RunResult {
                returncode: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                elapsed_seconds: 0.0,
                error: Some(format!(
                    "command {} cannot be parsed: {error}",
                    py_repr(&spec.command)
                )),
                parsed: None,
            };
        }
    };
    if argv.is_empty() {
        return RunResult {
            returncode: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            elapsed_seconds: 0.0,
            error: Some("empty command".to_string()),
            parsed: None,
        };
    }

    let started = Instant::now();
    let mut child = match Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return RunResult {
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

    if let Some(mut stdin) = child.stdin.take() {
        if let Err(error) = stdin.write_all(stdin_json.as_bytes()) {
            return RunResult {
                returncode: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                error: Some(error.to_string()),
                parsed: None,
            };
        }
    }

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut handle) = child.stdout.take() {
                    let _ = handle.read_to_string(&mut stdout);
                }
                if let Some(mut handle) = child.stderr.take() {
                    let _ = handle.read_to_string(&mut stderr);
                }
                return RunResult {
                    returncode: status.code(),
                    stdout,
                    stderr,
                    timed_out: false,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                    error: None,
                    parsed: None,
                };
            }
            Ok(None) => {}
            Err(error) => {
                return RunResult {
                    returncode: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    timed_out: false,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                    error: Some(error.to_string()),
                    parsed: None,
                };
            }
        }

        if started.elapsed() >= Duration::from_secs(spec.timeout) {
            let _ = child.kill();
            let (stdout, stderr, code) = match child.wait_with_output() {
                Ok(output) => (
                    String::from_utf8_lossy(&output.stdout).to_string(),
                    String::from_utf8_lossy(&output.stderr).to_string(),
                    output.status.code(),
                ),
                Err(_) => (String::new(), String::new(), None),
            };
            return RunResult {
                returncode: code,
                stdout,
                stderr,
                timed_out: true,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                error: None,
                parsed: None,
            };
        }
        sleep(Duration::from_millis(20));
    }
}

fn serialize_payload(
    event: &str,
    payload: &JsonMap<String, JsonValue>,
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
    serde_json::to_string(&json!({
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": session_id,
        "cwd": cwd,
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
                return Some(json!({"action": "block", "message": message}));
            }
        }
        if object.get("decision").and_then(JsonValue::as_str) == Some("block") {
            let message = object
                .get("reason")
                .or_else(|| object.get("message"))
                .and_then(JsonValue::as_str)?
                .trim();
            if !message.is_empty() {
                return Some(json!({"action": "block", "message": message}));
            }
        }
        return None;
    }

    let context = object.get("context").and_then(JsonValue::as_str)?.trim();
    (!context.is_empty()).then(|| json!({"context": context}))
}

// ---------------------------------------------------------------------------
// Allowlist
// ---------------------------------------------------------------------------

fn allowlist_path(hermes_home: &Path) -> PathBuf {
    hermes_home.join(ALLOWLIST_FILENAME)
}

fn load_allowlist(hermes_home: &Path) -> AllowlistFile {
    let path = allowlist_path(hermes_home);
    if !path.exists() {
        return AllowlistFile::default();
    }
    let Ok(raw) = fs::read_to_string(path) else {
        return AllowlistFile::default();
    };
    let mut parsed = serde_json::from_str::<AllowlistFile>(&raw).unwrap_or_default();
    parsed
        .approvals
        .retain(|entry| !entry.event.trim().is_empty() && !entry.command.trim().is_empty());
    parsed
}

fn save_allowlist(hermes_home: &Path, data: &AllowlistFile) -> Result<(), String> {
    let path = allowlist_path(hermes_home);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let rendered = serde_json::to_string_pretty(data).map_err(|error| error.to_string())?;
    atomic_write(&path, rendered.as_bytes())
}

/// Remove every allowlist entry whose command equals `command`. Returns the
/// number removed.
pub fn revoke(hermes_home: &Path, command: &str) -> usize {
    let mut data = load_allowlist(hermes_home);
    let before = data.approvals.len();
    data.approvals.retain(|entry| entry.command != command);
    let removed = before.saturating_sub(data.approvals.len());
    if removed > 0 {
        let _ = save_allowlist(hermes_home, &data);
    }
    removed
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

// ---------------------------------------------------------------------------
// Script inspection
// ---------------------------------------------------------------------------

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

fn script_is_executable(command: &str) -> bool {
    let Some(path) = command_script_path(command) else {
        return false;
    };
    let Ok(metadata) = fs::metadata(&path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    let Ok(argv) = shell_words::split(command) else {
        return false;
    };
    // When the command is a bare path to the script, the file itself must be
    // executable; otherwise an interpreter is invoking it and existence is
    // sufficient.
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

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_u64(value: &YamlValue) -> Option<u64> {
    match value {
        YamlValue::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|v| (v >= 0).then_some(v as u64)))
            .or_else(|| number.as_f64().and_then(|v| (v >= 0.0).then_some(v as u64))),
        YamlValue::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

/// Render an integer return code, or `None` as Python does (`exit=None`).
fn fmt_returncode(code: Option<i32>) -> String {
    match code {
        Some(value) => value.to_string(),
        None => "None".to_string(),
    }
}

/// Render an elapsed-seconds value matching Python's `str(float)` rounding
/// used by `agent.shell_hooks` (which stores `round(elapsed, 3)`).
fn fmt_seconds(seconds: f64) -> String {
    let rounded = (seconds * 1000.0).round() / 1000.0;
    if rounded == 0.0 {
        return "0".to_string();
    }
    let mut text = format!("{rounded:.3}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

/// Approximate Python's `repr()` for a string: single-quoted, escaping
/// backslashes and single quotes.
fn py_repr(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('\'', "\\'");
    format!("'{escaped}'")
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn config_from(yaml: &str) -> YamlValue {
        serde_yaml::from_str::<YamlValue>(yaml).unwrap()
    }

    #[test]
    fn iter_configured_hooks_parses_all_valid_events() {
        let config = config_from(
            r#"
hooks:
  pre_tool_call:
    - command: "~/hooks/block.sh"
      matcher: "^terminal$"
      timeout: 12
  pre_llm_call:
    - command: "~/hooks/context.py"
  not_a_real_event:
    - command: "/bin/true"
"#,
        );
        let specs = iter_configured_hooks(&config);
        assert_eq!(specs.len(), 2);
        let pre_tool = specs.iter().find(|s| s.event == "pre_tool_call").unwrap();
        assert_eq!(pre_tool.matcher.as_deref(), Some("^terminal$"));
        assert_eq!(pre_tool.timeout, 12);
        assert!(specs.iter().any(|s| s.event == "pre_llm_call"));
    }

    #[test]
    fn timeout_clamps_to_max() {
        let config = config_from(
            r#"
hooks:
  pre_tool_call:
    - command: "/bin/true"
      timeout: 9999
"#,
        );
        let specs = iter_configured_hooks(&config);
        assert_eq!(specs[0].timeout, MAX_TIMEOUT_SECONDS);
    }

    #[test]
    fn matches_tool_uses_fullmatch() {
        let config = config_from(
            r#"
hooks:
  pre_tool_call:
    - command: "/bin/true"
      matcher: "term.*"
"#,
        );
        let spec = &iter_configured_hooks(&config)[0];
        assert!(spec.matches_tool(Some("terminal")));
        // "term" is a prefix of "terminal_x" only as a substring; fullmatch of
        // "term.*" against "x_terminal" must fail (anchored at start).
        assert!(!spec.matches_tool(Some("x_terminal")));
        assert!(!spec.matches_tool(None));
    }

    #[test]
    fn matches_tool_without_matcher_is_always_true() {
        let spec = ShellHookSpec {
            event: "pre_tool_call".to_string(),
            command: "/bin/true".to_string(),
            matcher: None,
            timeout: 5,
            compiled_matcher: None,
        };
        assert!(spec.matches_tool(Some("anything")));
        assert!(spec.matches_tool(None));
    }

    #[test]
    fn parse_response_block_shape_action() {
        let parsed = parse_response("pre_tool_call", r#"{"action":"block","message":"Nope"}"#);
        assert_eq!(parsed, Some(json!({"action": "block", "message": "Nope"})));
    }

    #[test]
    fn parse_response_block_shape_decision() {
        let parsed = parse_response("pre_tool_call", r#"{"decision":"block","reason":"Stop"}"#);
        assert_eq!(parsed, Some(json!({"action": "block", "message": "Stop"})));
    }

    #[test]
    fn parse_response_context_for_other_events() {
        let parsed = parse_response("pre_llm_call", r#"{"context":"hi"}"#);
        assert_eq!(parsed, Some(json!({"context": "hi"})));
        assert_eq!(parse_response("pre_llm_call", ""), None);
    }

    #[test]
    fn revoke_removes_matching_entries() {
        let dir = TempDir::new().unwrap();
        let home = dir.path();
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
        save_allowlist(home, &data).unwrap();
        assert_eq!(revoke(home, "a"), 1);
        assert_eq!(revoke(home, "missing"), 0);
        let reloaded = load_allowlist(home);
        assert_eq!(reloaded.approvals.len(), 1);
        assert_eq!(reloaded.approvals[0].command, "b");
    }

    #[test]
    fn truncate_matches_python_semantics() {
        assert_eq!(truncate("hello", 400), "hello");
        assert_eq!(truncate("abcdef", 5), "ab...");
    }

    #[test]
    fn default_payload_shapes() {
        let pre = default_payload("pre_tool_call");
        assert_eq!(pre.get("tool_name").and_then(|v| v.as_str()), Some("terminal"));
        let fallback = default_payload("transform_tool_result");
        assert_eq!(
            fallback.get("session_id").and_then(|v| v.as_str()),
            Some("test-session")
        );
        assert!(has_default_payload("subagent_stop"));
        assert!(!has_default_payload("transform_tool_result"));
    }

    #[test]
    fn run_once_executes_and_parses_block() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("hook.sh");
        fs::write(
            &script,
            "#!/bin/sh\nread _ || true\necho '{\"action\":\"block\",\"message\":\"no\"}'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let config = config_from(&format!(
            "hooks:\n  pre_tool_call:\n    - command: \"{}\"\n",
            script.display()
        ));
        let spec = &iter_configured_hooks(&config)[0];
        let payload = default_payload("pre_tool_call");
        let result = run_once(spec, &payload);
        assert_eq!(result.returncode, Some(0));
        assert_eq!(result.parsed, Some(json!({"action": "block", "message": "no"})));
    }

    #[test]
    fn fmt_seconds_rounds_and_trims() {
        assert_eq!(fmt_seconds(0.0), "0");
        assert_eq!(fmt_seconds(1.2340), "1.234");
        assert_eq!(fmt_seconds(0.5), "0.5");
        assert_eq!(fmt_seconds(0.0009), "0.001");
    }

    #[test]
    fn py_repr_quotes_and_escapes() {
        assert_eq!(py_repr("terminal"), "'terminal'");
        assert_eq!(py_repr("a'b"), "'a\\'b'");
    }
}
