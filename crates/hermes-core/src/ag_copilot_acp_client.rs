//! OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.
//!
//! This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
//! backend. Each request starts a short-lived ACP session, sends the formatted
//! conversation as a single prompt, collects text chunks, and converts the
//! result back into the minimal shape Hermes expects from an OpenAI client.
//!
//! Native Rust port of `agent/copilot_acp_client.py`.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::agent_file_safety::{get_read_block_error, is_write_denied};
use crate::agent_redact::redact_sensitive_text;
use crate::mod_hermes_constants::get_subprocess_home;

pub const ACP_MARKER_BASE_URL: &str = "acp://copilot";
const DEFAULT_TIMEOUT_SECONDS: f64 = 900.0;

// ---------------------------------------------------------------------------
// Public result types (mirror the SimpleNamespace shapes returned in Python)
// ---------------------------------------------------------------------------

/// A single extracted tool call (mirrors the SimpleNamespace produced by
/// `_extract_tool_calls_from_text`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub call_id: String,
    pub response_item_id: Option<String>,
    /// Always "function".
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string of arguments.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub model: String,
}

/// Error type for ACP operations. Mirrors the Python RuntimeError/TimeoutError
/// distinction via the `Timeout` variant.
#[derive(Debug)]
pub enum AcpError {
    Runtime(String),
    Timeout(String),
}

impl std::fmt::Display for AcpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcpError::Runtime(m) => write!(f, "{m}"),
            AcpError::Timeout(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for AcpError {}

// ---------------------------------------------------------------------------
// Environment / command resolution
// ---------------------------------------------------------------------------

fn env_trimmed(key: &str) -> String {
    std::env::var(key).unwrap_or_default().trim().to_string()
}

fn resolve_command() -> String {
    let a = env_trimmed("HERMES_COPILOT_ACP_COMMAND");
    if !a.is_empty() {
        return a;
    }
    let b = env_trimmed("COPILOT_CLI_PATH");
    if !b.is_empty() {
        return b;
    }
    "copilot".to_string()
}

fn resolve_args() -> Vec<String> {
    let raw = env_trimmed("HERMES_COPILOT_ACP_ARGS");
    if raw.is_empty() {
        return vec!["--acp".to_string(), "--stdio".to_string()];
    }
    shlex_split(&raw)
}

/// Minimal POSIX-ish shell word splitter matching Python `shlex.split` for the
/// common cases (whitespace separation, single & double quotes, backslash
/// escapes).
pub fn shlex_split(input: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut has_token = false;
    let mut chars = input.chars().peekable();

    #[derive(PartialEq)]
    enum Mode {
        Normal,
        Single,
        Double,
    }
    let mut mode = Mode::Normal;

    while let Some(c) = chars.next() {
        match mode {
            Mode::Normal => {
                if c.is_whitespace() {
                    if has_token {
                        out.push(std::mem::take(&mut cur));
                        has_token = false;
                    }
                } else if c == '\'' {
                    mode = Mode::Single;
                    has_token = true;
                } else if c == '"' {
                    mode = Mode::Double;
                    has_token = true;
                } else if c == '\\' {
                    if let Some(&next) = chars.peek() {
                        cur.push(next);
                        chars.next();
                    }
                    has_token = true;
                } else {
                    cur.push(c);
                    has_token = true;
                }
            }
            Mode::Single => {
                if c == '\'' {
                    mode = Mode::Normal;
                } else {
                    cur.push(c);
                }
            }
            Mode::Double => {
                if c == '"' {
                    mode = Mode::Normal;
                } else if c == '\\' {
                    if let Some(&next) = chars.peek() {
                        // In double quotes, backslash only escapes a few chars.
                        if next == '"' || next == '\\' || next == '$' || next == '`' {
                            cur.push(next);
                            chars.next();
                        } else {
                            cur.push('\\');
                        }
                    } else {
                        cur.push('\\');
                    }
                } else {
                    cur.push(c);
                }
            }
        }
    }
    if has_token {
        out.push(cur);
    }
    out
}

/// Return a stable HOME for child ACP processes.
fn resolve_home_dir() -> String {
    if let Some(profile_home) = get_subprocess_home() {
        if !profile_home.is_empty() {
            return profile_home;
        }
    }

    let home = env_trimmed("HOME");
    if !home.is_empty() {
        return home;
    }

    if let Some(dir) = dirs::home_dir() {
        let expanded = dir.to_string_lossy().to_string();
        if !expanded.is_empty() && expanded != "~" {
            return expanded;
        }
    }

    // Last resort: /tmp (writable on any POSIX system).
    "/tmp".to_string()
}

fn build_subprocess_env() -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    let home = resolve_home_dir();
    let mut found = false;
    for kv in env.iter_mut() {
        if kv.0 == "HOME" {
            kv.1 = home.clone();
            found = true;
        }
    }
    if !found {
        env.push(("HOME".to_string(), home));
    }
    env
}

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

fn jsonrpc_error(message_id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    })
}

fn permission_denied(message_id: &Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    })
}

// ---------------------------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------------------------

/// Format a list of chat messages into a single ACP prompt string.
pub fn format_messages_as_prompt(
    messages: &[Value],
    model: Option<&str>,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
) -> String {
    let mut sections: Vec<String> = vec![
        "You are being used as the active ACP agent backend for Hermes.".to_string(),
        "Use ACP capabilities to complete tasks.".to_string(),
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.".to_string(),
        "If no tool is needed, answer normally.".to_string(),
    ];

    if let Some(m) = model {
        if !m.is_empty() {
            sections.push(format!("Hermes requested model hint: {m}"));
        }
    }

    if let Some(tools) = tools {
        if !tools.is_empty() {
            let mut tool_specs: Vec<Value> = Vec::new();
            for t in tools {
                let t = match t.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let fn_val = t.get("function").cloned().unwrap_or(Value::Null);
                let fn_obj = match fn_val.as_object() {
                    Some(o) => o.clone(),
                    None => {
                        // `t.get("function") or {}` -> if function missing/falsey,
                        // an empty dict is used (which has no name) -> skipped.
                        if fn_val.is_null() {
                            serde_json::Map::new()
                        } else {
                            // Non-dict truthy function: Python checks isinstance dict -> continue.
                            continue;
                        }
                    }
                };
                let name = match fn_obj.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => continue,
                };
                if name.trim().is_empty() {
                    continue;
                }
                let description = fn_obj
                    .get("description")
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new()));
                let parameters = fn_obj
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                tool_specs.push(json!({
                    "name": name.trim(),
                    "description": description,
                    "parameters": parameters,
                }));
            }
            if !tool_specs.is_empty() {
                let serialized = serde_json::to_string(&tool_specs).unwrap_or_else(|_| "[]".into());
                sections.push(format!(
                    "Available tools (OpenAI function schema). When using a tool, emit ONLY <tool_call>{{...}}</tool_call> with one JSON object containing id/type/function{{name,arguments}}. arguments must be a JSON string.\n{serialized}"
                ));
            }
        }
    }

    if let Some(tc) = tool_choice {
        if !tc.is_null() {
            let serialized = serde_json::to_string(tc).unwrap_or_else(|_| "null".into());
            sections.push(format!("Tool choice hint: {serialized}"));
        }
    }

    let mut transcript: Vec<String> = Vec::new();
    for message in messages {
        let message = match message.as_object() {
            Some(o) => o,
            None => continue,
        };
        let raw_role = message
            .get("role")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("unknown");
        let role_norm = raw_role.trim().to_lowercase();
        let role = if role_norm == "tool" {
            "tool".to_string()
        } else if matches!(role_norm.as_str(), "system" | "user" | "assistant") {
            role_norm.clone()
        } else {
            "context".to_string()
        };

        let content = message.get("content").cloned().unwrap_or(Value::Null);
        let rendered = render_message_content(&content);
        if rendered.is_empty() {
            continue;
        }

        let label = match role.as_str() {
            "system" => "System".to_string(),
            "user" => "User".to_string(),
            "assistant" => "Assistant".to_string(),
            "tool" => "Tool".to_string(),
            "context" => "Context".to_string(),
            other => title_case(other),
        };
        transcript.push(format!("{label}:\n{rendered}"));
    }

    if !transcript.is_empty() {
        sections.push(format!(
            "Conversation transcript:\n\n{}",
            transcript.join("\n\n")
        ));
    }

    sections.push("Continue the conversation from the latest user request.".to_string());

    sections
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Python `str.title()` semantics for the fallback label.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

/// Render a message's `content` field into a plain string.
pub fn render_message_content(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.trim().to_string(),
        Value::Object(map) => {
            if map.contains_key("text") {
                return map
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        // str(content.get("text") or "") for non-string/falsey
                        match map.get("text") {
                            Some(Value::Null) | None => String::new(),
                            Some(Value::Bool(false)) => String::new(),
                            Some(v) => stringify_value(v),
                        }
                    })
                    .trim()
                    .to_string();
            }
            if let Some(inner) = map.get("content") {
                if let Some(s) = inner.as_str() {
                    return s.trim().to_string();
                }
            }
            // json.dumps(content, ensure_ascii=True)
            json_dumps_ascii(content)
        }
        Value::Array(items) => {
            let mut parts: Vec<String> = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => parts.push(s.clone()),
                    Value::Object(o) => {
                        if let Some(text) = o.get("text").and_then(|v| v.as_str()) {
                            if !text.trim().is_empty() {
                                parts.push(text.trim().to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            parts.join("\n").trim().to_string()
        }
        other => stringify_value(other).trim().to_string(),
    }
}

/// Mimic Python `str(value)` for scalar JSON values used as content fallbacks.
fn stringify_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        Value::Number(n) => n.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Serialize a value the way Python `json.dumps(obj, ensure_ascii=True)` does:
/// non-ASCII characters become `\uXXXX` escapes.
fn json_dumps_ascii(value: &Value) -> String {
    let s = serde_json::to_string(value).unwrap_or_default();
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) < 0x80 {
            out.push(c);
        } else {
            let cp = c as u32;
            if cp > 0xFFFF {
                // Encode as UTF-16 surrogate pair.
                let v = cp - 0x10000;
                let hi = 0xD800 + (v >> 10);
                let lo = 0xDC00 + (v & 0x3FF);
                out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
            } else {
                out.push_str(&format!("\\u{cp:04x}"));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tool-call extraction
// ---------------------------------------------------------------------------

/// Extract `<tool_call>{...}</tool_call>` blocks (or bare JSON fallback) from
/// model output, returning the parsed calls and the cleaned remaining text.
pub fn extract_tool_calls_from_text(text: &str) -> (Vec<ToolCall>, String) {
    if text.trim().is_empty() {
        return (Vec::new(), String::new());
    }

    let mut extracted: Vec<ToolCall> = Vec::new();
    let mut consumed_spans: Vec<(usize, usize)> = Vec::new();

    let mut try_add = |raw_json: &str, extracted: &mut Vec<ToolCall>| {
        let obj: Value = match serde_json::from_str(raw_json) {
            Ok(v) => v,
            Err(_) => return,
        };
        let obj = match obj.as_object() {
            Some(o) => o,
            None => return,
        };
        let fn_obj = match obj.get("function").and_then(|v| v.as_object()) {
            Some(o) => o,
            None => return,
        };
        let fn_name = match fn_obj.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return,
        };
        if fn_name.trim().is_empty() {
            return;
        }
        let fn_args = match fn_obj.get("arguments") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
            None => "{}".to_string(),
        };
        let call_id = match obj.get("id").and_then(|v| v.as_str()) {
            Some(s) if !s.trim().is_empty() => s.to_string(),
            _ => format!("acp_call_{}", extracted.len() + 1),
        };
        extracted.push(ToolCall {
            id: call_id.clone(),
            call_id,
            response_item_id: None,
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: fn_name.trim().to_string(),
                arguments: fn_args,
            },
        });
    };

    // <tool_call>\s*(\{.*?\})\s*</tool_call> with DOTALL.
    for (start, end, inner) in find_tool_call_blocks(text) {
        try_add(&inner, &mut extracted);
        consumed_spans.push((start, end));
    }

    // Only try bare-JSON fallback when no XML blocks were found.
    if extracted.is_empty() {
        for (start, end, raw) in find_bare_json_calls(text) {
            try_add(&raw, &mut extracted);
            consumed_spans.push((start, end));
        }
    }

    if consumed_spans.is_empty() {
        return (extracted, text.trim().to_string());
    }

    consumed_spans.sort();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in consumed_spans {
        if merged.is_empty() || start > merged.last().unwrap().1 {
            merged.push((start, end));
        } else {
            let last = merged.last_mut().unwrap();
            last.1 = last.1.max(end);
        }
    }

    let bytes = text.as_bytes();
    let mut parts: Vec<String> = Vec::new();
    let mut cursor = 0usize;
    for (start, end) in merged {
        if cursor < start {
            parts.push(String::from_utf8_lossy(&bytes[cursor..start]).to_string());
        }
        cursor = cursor.max(end);
    }
    if cursor < bytes.len() {
        parts.push(String::from_utf8_lossy(&bytes[cursor..]).to_string());
    }

    let cleaned = parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    (extracted, cleaned)
}

/// Find `<tool_call> ... </tool_call>` blocks. Returns (start, end, inner_json)
/// in byte offsets, replicating the regex `<tool_call>\s*(\{.*?\})\s*</tool_call>`
/// with DOTALL (`.` matches newlines) and non-greedy inner capture.
fn find_tool_call_blocks(text: &str) -> Vec<(usize, usize, String)> {
    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";
    let bytes = text.as_bytes();
    let mut results = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel_open) = text[search_from..].find(open_tag) {
        let block_start = search_from + rel_open;
        let after_open = block_start + open_tag.len();

        // Skip leading whitespace (\s*).
        let mut i = after_open;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        // Must start with '{'.
        if i >= bytes.len() || bytes[i] != b'{' {
            search_from = after_open;
            continue;
        }
        let brace_start = i;

        // Non-greedy `\{.*?\}` followed by `\s*</tool_call>`: find the earliest
        // '}' such that what follows (after optional whitespace) is the close tag.
        let mut matched = false;
        let mut j = brace_start;
        while let Some(rel_brace) = text[j..].find('}') {
            let brace_end = j + rel_brace + 1; // include '}'
            let mut k = brace_end;
            while k < bytes.len() && (bytes[k] as char).is_whitespace() {
                k += 1;
            }
            if text[k..].starts_with(close_tag) {
                let block_end = k + close_tag.len();
                let inner = text[brace_start..brace_end].to_string();
                results.push((block_start, block_end, inner));
                search_from = block_end;
                matched = true;
                break;
            }
            // Try next '}'.
            j = brace_end;
        }
        if !matched {
            search_from = after_open;
        }
    }

    results
}

/// Find bare JSON tool-call objects matching the regex:
/// `\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{.*?\}\s*\}`
/// (DOTALL). Returns (start, end, raw) in byte offsets.
fn find_bare_json_calls(text: &str) -> Vec<(usize, usize, String)> {
    let re = regex::Regex::new(
        r#"(?s)\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{.*?\}\s*\}"#,
    )
    .expect("valid regex");
    re.find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

/// Ensure `path_text` is an absolute path resolving within `cwd`.
fn ensure_path_within_cwd(path_text: &str, cwd: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(path_text);
    if !candidate.is_absolute() {
        return Err("ACP file-system paths must be absolute.".to_string());
    }
    let resolved = normalize_path(candidate);
    let root = normalize_path(Path::new(cwd));
    if !resolved.starts_with(&root) {
        return Err(format!(
            "Path '{}' is outside the session cwd '{}'.",
            resolved.display(),
            root.display()
        ));
    }
    Ok(resolved)
}

/// Lexical path normalization (resolves `.` and `..`) similar to
/// `Path.resolve()` without requiring the path to exist.
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Timeout normalization
// ---------------------------------------------------------------------------

/// Timeout argument analogous to Python accepting either a float or an
/// httpx.Timeout-like object.
#[derive(Debug, Clone, Default)]
pub enum TimeoutSpec {
    #[default]
    Default,
    Seconds(f64),
    /// httpx.Timeout-like: read/write/connect/pool/timeout components.
    Components {
        read: Option<f64>,
        write: Option<f64>,
        connect: Option<f64>,
        pool: Option<f64>,
        timeout: Option<f64>,
    },
}

impl TimeoutSpec {
    fn effective(&self) -> f64 {
        match self {
            TimeoutSpec::Default => DEFAULT_TIMEOUT_SECONDS,
            TimeoutSpec::Seconds(s) => *s,
            TimeoutSpec::Components {
                read,
                write,
                connect,
                pool,
                timeout,
            } => {
                let candidates = [*read, *write, *connect, *pool, *timeout];
                let numeric: Vec<f64> = candidates.into_iter().flatten().collect();
                if numeric.is_empty() {
                    DEFAULT_TIMEOUT_SECONDS
                } else {
                    numeric.into_iter().fold(f64::NEG_INFINITY, f64::max)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

/// Minimal OpenAI-client-compatible facade for Copilot ACP.
pub struct CopilotACPClient {
    pub api_key: String,
    pub base_url: String,
    default_headers: std::collections::HashMap<String, String>,
    acp_command: String,
    acp_args: Vec<String>,
    acp_cwd: String,
    pub is_closed: Mutex<bool>,
    active_process: Arc<Mutex<Option<Child>>>,
}

/// Construction options mirroring the Python keyword arguments.
#[derive(Default)]
pub struct ClientOptions {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub default_headers: Option<std::collections::HashMap<String, String>>,
    pub acp_command: Option<String>,
    pub acp_args: Option<Vec<String>>,
    pub acp_cwd: Option<String>,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
}

impl CopilotACPClient {
    pub fn new(opts: ClientOptions) -> Self {
        let acp_command = opts
            .acp_command
            .or(opts.command)
            .unwrap_or_else(resolve_command);
        let acp_args = opts.acp_args.or(opts.args).unwrap_or_else(resolve_args);
        let cwd_base = opts
            .acp_cwd
            .unwrap_or_else(|| std::env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default());
        let acp_cwd = normalize_path(Path::new(&cwd_base))
            .to_string_lossy()
            .to_string();

        CopilotACPClient {
            api_key: opts.api_key.unwrap_or_else(|| "copilot-acp".to_string()),
            base_url: opts.base_url.unwrap_or_else(|| ACP_MARKER_BASE_URL.to_string()),
            default_headers: opts.default_headers.unwrap_or_default(),
            acp_command,
            acp_args,
            acp_cwd,
            is_closed: Mutex::new(false),
            active_process: Arc::new(Mutex::new(None)),
        }
    }

    pub fn default_headers(&self) -> &std::collections::HashMap<String, String> {
        &self.default_headers
    }

    pub fn close(&self) {
        let mut proc_opt = {
            let mut guard = self.active_process.lock().unwrap();
            guard.take()
        };
        *self.is_closed.lock().unwrap() = true;
        let proc = match proc_opt.as_mut() {
            Some(p) => p,
            None => return,
        };
        // proc.terminate(); proc.wait(timeout=2)
        let _ = terminate_child(proc);
        match wait_timeout(proc, Duration::from_secs(2)) {
            Some(_) => {}
            None => {
                let _ = proc.kill();
            }
        }
    }

    /// Create a chat completion. Mirrors `_create_chat_completion`.
    pub fn create_chat_completion(
        &self,
        model: Option<&str>,
        messages: &[Value],
        timeout: TimeoutSpec,
        tools: Option<&[Value]>,
        tool_choice: Option<&Value>,
    ) -> Result<ChatCompletion, AcpError> {
        let prompt_text = format_messages_as_prompt(messages, model, tools, tool_choice);
        let effective_timeout = timeout.effective();

        let (response_text, reasoning_text) =
            self.run_prompt(&prompt_text, effective_timeout)?;

        let (tool_calls, cleaned_text) = extract_tool_calls_from_text(&response_text);

        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
        };

        let reasoning = if reasoning_text.is_empty() {
            None
        } else {
            Some(reasoning_text.clone())
        };

        let assistant_message = AssistantMessage {
            content: cleaned_text,
            reasoning: reasoning.clone(),
            reasoning_content: reasoning,
            reasoning_details: None,
            tool_calls: tool_calls.clone(),
        };

        let finish_reason = if tool_calls.is_empty() {
            "stop".to_string()
        } else {
            "tool_calls".to_string()
        };

        let choice = Choice {
            message: assistant_message,
            finish_reason,
        };

        Ok(ChatCompletion {
            choices: vec![choice],
            usage,
            model: model
                .filter(|m| !m.is_empty())
                .map(|m| m.to_string())
                .unwrap_or_else(|| "copilot-acp".to_string()),
        })
    }

    fn run_prompt(
        &self,
        prompt_text: &str,
        timeout_seconds: f64,
    ) -> Result<(String, String), AcpError> {
        let mut cmd = Command::new(&self.acp_command);
        cmd.args(&self.acp_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&self.acp_cwd)
            .env_clear();
        for (k, v) in build_subprocess_env() {
            cmd.env(k, v);
        }

        let mut proc = match cmd.spawn() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(AcpError::Runtime(format!(
                    "Could not start Copilot ACP command '{}'. Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH.",
                    self.acp_command
                )));
            }
            Err(e) => {
                return Err(AcpError::Runtime(format!(
                    "Could not start Copilot ACP command '{}': {e}",
                    self.acp_command
                )));
            }
        };

        let stdin = proc.stdin.take();
        let stdout = proc.stdout.take();
        let stderr = proc.stderr.take();

        if stdin.is_none() || stdout.is_none() {
            let _ = proc.kill();
            return Err(AcpError::Runtime(
                "Copilot ACP process did not expose stdin/stdout pipes.".to_string(),
            ));
        }

        let mut stdin = stdin.unwrap();
        let stdout = stdout.unwrap();

        *self.is_closed.lock().unwrap() = false;

        // Reader threads: stdout -> inbox of parsed messages, stderr -> tail.
        let (inbox_tx, inbox_rx) = mpsc::channel::<Value>();
        let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));

        let out_handle = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                match serde_json::from_str::<Value>(&line) {
                    Ok(v) => {
                        if inbox_tx.send(v).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        let raw = json!({ "raw": line.trim_end_matches('\n') });
                        if inbox_tx.send(raw).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let err_tail_clone = stderr_tail.clone();
        let err_handle = if let Some(stderr) = stderr {
            Some(thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => break,
                    };
                    let mut tail = err_tail_clone.lock().unwrap();
                    if tail.len() >= 40 {
                        tail.pop_front();
                    }
                    tail.push_back(line.trim_end_matches('\n').to_string());
                }
            }))
        } else {
            None
        };

        // Register the active process so close() can terminate it.
        {
            let mut guard = self.active_process.lock().unwrap();
            *guard = Some(proc);
        }

        let result = self.run_session(
            &mut stdin,
            &inbox_rx,
            &stderr_tail,
            prompt_text,
            timeout_seconds,
        );

        // finally: self.close()
        self.close();

        // Join reader threads (process is dead/terminated, pipes will EOF).
        drop(stdin);
        let _ = out_handle.join();
        if let Some(h) = err_handle {
            let _ = h.join();
        }

        result
    }

    /// Drives the initialize / session.new / session.prompt sequence.
    fn run_session(
        &self,
        stdin: &mut std::process::ChildStdin,
        inbox_rx: &Receiver<Value>,
        stderr_tail: &Arc<Mutex<VecDeque<String>>>,
        prompt_text: &str,
        timeout_seconds: f64,
    ) -> Result<(String, String), AcpError> {
        let mut next_id: i64 = 0;
        let mut text_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();

        // initialize
        self.request(
            stdin,
            inbox_rx,
            stderr_tail,
            &mut next_id,
            "initialize",
            json!({
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": true,
                        "writeTextFile": true,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            }),
            timeout_seconds,
            None,
            None,
        )?;

        // session/new
        let session = self
            .request(
                stdin,
                inbox_rx,
                stderr_tail,
                &mut next_id,
                "session/new",
                json!({
                    "cwd": self.acp_cwd,
                    "mcpServers": [],
                }),
                timeout_seconds,
                None,
                None,
            )?
            .unwrap_or(Value::Null);

        let session_id = session
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if session_id.is_empty() {
            return Err(AcpError::Runtime(
                "Copilot ACP did not return a sessionId.".to_string(),
            ));
        }

        // session/prompt
        self.request(
            stdin,
            inbox_rx,
            stderr_tail,
            &mut next_id,
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
            }),
            timeout_seconds,
            Some(&mut text_parts),
            Some(&mut reasoning_parts),
        )?;

        Ok((text_parts.concat(), reasoning_parts.concat()))
    }

    #[allow(clippy::too_many_arguments)]
    fn request(
        &self,
        stdin: &mut std::process::ChildStdin,
        inbox_rx: &Receiver<Value>,
        stderr_tail: &Arc<Mutex<VecDeque<String>>>,
        next_id: &mut i64,
        method: &str,
        params: Value,
        timeout_seconds: f64,
        mut text_parts: Option<&mut Vec<String>>,
        mut reasoning_parts: Option<&mut Vec<String>>,
    ) -> Result<Option<Value>, AcpError> {
        *next_id += 1;
        let request_id = *next_id;
        let payload = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        });
        let line = serde_json::to_string(&payload).unwrap_or_default();
        if stdin.write_all(line.as_bytes()).is_err() || stdin.write_all(b"\n").is_err() {
            return Err(AcpError::Runtime(format!(
                "Copilot ACP {method} failed: could not write to stdin"
            )));
        }
        let _ = stdin.flush();

        let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds.max(0.0));

        while Instant::now() < deadline {
            // proc.poll() check: if process exited, break.
            if self.process_exited() {
                break;
            }

            let msg = match inbox_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(m) => m,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Reader thread closed (process ended). Loop will detect exit.
                    if self.process_exited() {
                        break;
                    }
                    continue;
                }
            };

            if self.handle_server_message(
                &msg,
                stdin,
                &self.acp_cwd,
                text_parts.as_deref_mut(),
                reasoning_parts.as_deref_mut(),
            ) {
                continue;
            }

            // Compare msg id to request_id (Python compares to int).
            let matches_id = msg
                .get("id")
                .map(|v| value_eq_int(v, request_id))
                .unwrap_or(false);
            if !matches_id {
                continue;
            }
            if let Some(err) = msg.get("error") {
                let err_msg = err
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| serde_json::to_string(err).unwrap_or_default());
                return Err(AcpError::Runtime(format!(
                    "Copilot ACP {method} failed: {err_msg}"
                )));
            }
            return Ok(msg.get("result").cloned());
        }

        let stderr_text = {
            let tail = stderr_tail.lock().unwrap();
            tail.iter().cloned().collect::<Vec<_>>().join("\n").trim().to_string()
        };
        if self.process_exited() && !stderr_text.is_empty() {
            return Err(AcpError::Runtime(format!(
                "Copilot ACP process exited early: {stderr_text}"
            )));
        }
        Err(AcpError::Timeout(format!(
            "Timed out waiting for Copilot ACP response to {method}."
        )))
    }

    fn process_exited(&self) -> bool {
        let mut guard = self.active_process.lock().unwrap();
        match guard.as_mut() {
            None => true,
            Some(child) => match child.try_wait() {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(_) => true,
            },
        }
    }

    /// Handle an inbound server message. Returns true if the message was a
    /// server-initiated notification/request that we handled (and the request
    /// loop should `continue`).
    fn handle_server_message(
        &self,
        msg: &Value,
        stdin: &mut std::process::ChildStdin,
        cwd: &str,
        text_parts: Option<&mut Vec<String>>,
        reasoning_parts: Option<&mut Vec<String>>,
    ) -> bool {
        let method = match msg.get("method").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => return false,
        };

        if method == "session/update" {
            let params = msg.get("params").cloned().unwrap_or(Value::Null);
            let update = params.get("update").cloned().unwrap_or(Value::Null);
            let kind = update
                .get("sessionUpdate")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let content = update.get("content").cloned().unwrap_or(Value::Null);
            let chunk_text = if content.is_object() {
                content
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };
            if kind == "agent_message_chunk" && !chunk_text.is_empty() {
                if let Some(parts) = text_parts {
                    parts.push(chunk_text);
                }
            } else if kind == "agent_thought_chunk" && !chunk_text.is_empty() {
                if let Some(parts) = reasoning_parts {
                    parts.push(chunk_text);
                }
            }
            return true;
        }

        // From here on the Python sends a response on stdin. stdin always
        // exists in our context.

        let message_id = msg.get("id").cloned().unwrap_or(Value::Null);
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        let response: Value = if method == "session/request_permission" {
            permission_denied(&message_id)
        } else if method == "fs/read_text_file" {
            match self.handle_read_text_file(&params, cwd, &message_id) {
                Ok(v) => v,
                Err(e) => jsonrpc_error(&message_id, -32602, &e),
            }
        } else if method == "fs/write_text_file" {
            match self.handle_write_text_file(&params, cwd, &message_id) {
                Ok(v) => v,
                Err(e) => jsonrpc_error(&message_id, -32602, &e),
            }
        } else {
            jsonrpc_error(
                &message_id,
                -32601,
                &format!("ACP client method '{method}' is not supported by Hermes yet."),
            )
        };

        let line = serde_json::to_string(&response).unwrap_or_default();
        let _ = stdin.write_all(line.as_bytes());
        let _ = stdin.write_all(b"\n");
        let _ = stdin.flush();
        true
    }

    fn handle_read_text_file(
        &self,
        params: &Value,
        cwd: &str,
        message_id: &Value,
    ) -> Result<Value, String> {
        let path_text = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let path = ensure_path_within_cwd(path_text, cwd)?;
        let path_str = path.to_string_lossy().to_string();

        if let Some(block_error) = get_read_block_error(&path_str) {
            return Err(block_error);
        }

        let mut content = if path.exists() {
            std::fs::read_to_string(&path).map_err(|e| e.to_string())?
        } else {
            String::new()
        };

        let line = params.get("line").and_then(value_as_int);
        let limit = params.get("limit").and_then(value_as_int);

        if let Some(line) = line {
            if line > 1 {
                let lines = splitlines_keepends(&content);
                let start = (line - 1) as usize;
                let slice: Vec<&String> = if let Some(limit) = limit {
                    if limit > 0 {
                        let end = start + limit as usize;
                        lines.iter().skip(start).take(end.saturating_sub(start)).collect()
                    } else {
                        lines.iter().skip(start).collect()
                    }
                } else {
                    lines.iter().skip(start).collect()
                };
                content = slice.into_iter().cloned().collect::<Vec<_>>().concat();
            }
        }

        if !content.is_empty() {
            content = redact_sensitive_text(&content, true, false);
        }

        Ok(json!({
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "content": content,
            },
        }))
    }

    fn handle_write_text_file(
        &self,
        params: &Value,
        cwd: &str,
        message_id: &Value,
    ) -> Result<Value, String> {
        let path_text = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let path = ensure_path_within_cwd(path_text, cwd)?;
        let path_str = path.to_string_lossy().to_string();

        if is_write_denied(&path_str) {
            return Err(format!(
                "Write denied: '{path_str}' is a protected system/credential file."
            ));
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
        std::fs::write(&path, content).map_err(|e| e.to_string())?;

        Ok(json!({
            "jsonrpc": "2.0",
            "id": message_id,
            "result": Value::Null,
        }))
    }
}

impl Drop for CopilotACPClient {
    fn drop(&mut self) {
        // Ensure any lingering subprocess is reaped.
        let mut proc_opt = {
            let mut guard = self.active_process.lock().unwrap();
            guard.take()
        };
        if let Some(proc) = proc_opt.as_mut() {
            let _ = terminate_child(proc);
            if wait_timeout(proc, Duration::from_secs(2)).is_none() {
                let _ = proc.kill();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn value_eq_int(v: &Value, target: i64) -> bool {
    v.as_i64().map(|n| n == target).unwrap_or(false)
}

/// Mimic Python: `isinstance(x, int)` (but not bool). serde_json numbers that
/// are integral.
fn value_as_int(v: &Value) -> Option<i64> {
    if v.is_boolean() {
        return None;
    }
    v.as_i64()
}

/// Python `str.splitlines(keepends=True)` for the line-slicing logic.
fn splitlines_keepends(s: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if c == '\n' {
            lines.push(std::mem::take(&mut cur));
        } else if c == '\r' {
            // CRLF stays together.
            if i + 1 < chars.len() && chars[i + 1] == '\n' {
                cur.push('\n');
                i += 1;
            }
            lines.push(std::mem::take(&mut cur));
        }
        i += 1;
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

#[cfg(unix)]
fn terminate_child(child: &mut Child) -> std::io::Result<()> {
    // SIGTERM, matching subprocess.terminate().
    let pid = child.id() as i32;
    let rc = unsafe { libc::kill(pid, libc::SIGTERM) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) -> std::io::Result<()> {
    child.kill()
}

/// Wait for a child up to `dur`. Returns Some(()) if it exited, None on timeout.
fn wait_timeout(child: &mut Child, dur: Duration) -> Option<()> {
    let deadline = Instant::now() + dur;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Some(()),
            Ok(None) => {
                if Instant::now() >= deadline {
                    return None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => return Some(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_shlex_split_basic() {
        assert_eq!(shlex_split("--acp --stdio"), vec!["--acp", "--stdio"]);
        assert_eq!(
            shlex_split(r#"--flag "hello world" 'single'"#),
            vec!["--flag", "hello world", "single"]
        );
        assert_eq!(shlex_split(""), Vec::<String>::new());
    }

    #[test]
    fn test_render_message_content_string() {
        assert_eq!(render_message_content(&json!("  hi  ")), "hi");
        assert_eq!(render_message_content(&Value::Null), "");
    }

    #[test]
    fn test_render_message_content_dict_text() {
        assert_eq!(render_message_content(&json!({"text": " hello "})), "hello");
        assert_eq!(
            render_message_content(&json!({"content": " inner "})),
            "inner"
        );
    }

    #[test]
    fn test_render_message_content_list() {
        let content = json!(["a", {"text": " b "}, {"image": "x"}, "c"]);
        assert_eq!(render_message_content(&content), "a\nb\nc");
    }

    #[test]
    fn test_render_message_content_dict_fallback_ascii() {
        // Unicode should be escaped (ensure_ascii=True).
        let out = render_message_content(&json!({"k": "café"}));
        assert!(out.contains("\\u00e9"), "got: {out}");
    }

    #[test]
    fn test_extract_tool_call_block() {
        let text = r#"Sure! <tool_call>{"id": "x1", "type": "function", "function": {"name": "do_it", "arguments": "{\"a\":1}"}}</tool_call> done"#;
        let (calls, cleaned) = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "x1");
        assert_eq!(calls[0].function.name, "do_it");
        assert_eq!(calls[0].function.arguments, "{\"a\":1}");
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(cleaned, "Sure!\ndone");
    }

    #[test]
    fn test_extract_tool_call_missing_id_generates() {
        let text = r#"<tool_call>{"type": "function", "function": {"name": "f"}}</tool_call>"#;
        let (calls, cleaned) = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "acp_call_1");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_extract_tool_call_non_string_args_serialized() {
        let text = r#"<tool_call>{"id":"i","type":"function","function":{"name":"f","arguments":{"x":2}}}</tool_call>"#;
        let (calls, _) = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"x":2}"#);
    }

    #[test]
    fn test_extract_no_tool_call() {
        let (calls, cleaned) = extract_tool_calls_from_text("just text");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "just text");
    }

    #[test]
    fn test_extract_bare_json_fallback() {
        let text = r#"prefix {"id": "abc", "type": "function", "function": {"name": "g", "arguments": "{}"}} suffix"#;
        let (calls, cleaned) = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "abc");
        assert_eq!(calls[0].function.name, "g");
        assert_eq!(cleaned, "prefix\nsuffix");
    }

    #[test]
    fn test_format_messages_basic() {
        let messages = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "user", "content": "hello"}),
        ];
        let out = format_messages_as_prompt(&messages, Some("gpt"), None, None);
        assert!(out.contains("active ACP agent backend"));
        assert!(out.contains("Hermes requested model hint: gpt"));
        assert!(out.contains("System:\nbe nice"));
        assert!(out.contains("User:\nhello"));
        assert!(out.contains("Continue the conversation"));
    }

    #[test]
    fn test_format_messages_unknown_role_becomes_context() {
        let messages = vec![json!({"role": "weird", "content": "data"})];
        let out = format_messages_as_prompt(&messages, None, None, None);
        assert!(out.contains("Context:\ndata"));
    }

    #[test]
    fn test_format_messages_with_tools() {
        let tools = vec![json!({
            "function": {"name": "search", "description": "find", "parameters": {"type": "object"}}
        })];
        let out = format_messages_as_prompt(&[], None, Some(&tools), None);
        assert!(out.contains("Available tools"));
        assert!(out.contains("\"search\""));
    }

    #[test]
    fn test_format_messages_tool_choice() {
        let tc = json!("auto");
        let out = format_messages_as_prompt(&[], None, None, Some(&tc));
        assert!(out.contains("Tool choice hint: \"auto\""));
    }

    #[test]
    fn test_jsonrpc_error_shape() {
        let id = json!(7);
        let e = jsonrpc_error(&id, -32601, "nope");
        assert_eq!(e["jsonrpc"], "2.0");
        assert_eq!(e["id"], 7);
        assert_eq!(e["error"]["code"], -32601);
        assert_eq!(e["error"]["message"], "nope");
    }

    #[test]
    fn test_permission_denied_shape() {
        let id = json!(3);
        let p = permission_denied(&id);
        assert_eq!(p["result"]["outcome"]["outcome"], "cancelled");
    }

    #[test]
    fn test_ensure_path_within_cwd_relative_rejected() {
        let r = ensure_path_within_cwd("relative/path", "/tmp/work");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("must be absolute"));
    }

    #[test]
    fn test_ensure_path_within_cwd_outside_rejected() {
        let r = ensure_path_within_cwd("/etc/passwd", "/tmp/work");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("outside the session cwd"));
    }

    #[test]
    fn test_ensure_path_within_cwd_ok() {
        let r = ensure_path_within_cwd("/tmp/work/sub/file.txt", "/tmp/work");
        assert!(r.is_ok());
    }

    #[test]
    fn test_timeout_spec_effective() {
        assert_eq!(TimeoutSpec::Default.effective(), DEFAULT_TIMEOUT_SECONDS);
        assert_eq!(TimeoutSpec::Seconds(12.0).effective(), 12.0);
        let spec = TimeoutSpec::Components {
            read: Some(5.0),
            write: Some(30.0),
            connect: None,
            pool: Some(10.0),
            timeout: None,
        };
        assert_eq!(spec.effective(), 30.0);
        let empty = TimeoutSpec::Components {
            read: None,
            write: None,
            connect: None,
            pool: None,
            timeout: None,
        };
        assert_eq!(empty.effective(), DEFAULT_TIMEOUT_SECONDS);
    }

    #[test]
    fn test_splitlines_keepends() {
        let lines = splitlines_keepends("a\nb\nc");
        assert_eq!(lines, vec!["a\n", "b\n", "c"]);
        let crlf = splitlines_keepends("x\r\ny");
        assert_eq!(crlf, vec!["x\r\n", "y"]);
    }

    #[test]
    fn test_resolve_args_default() {
        unsafe {
            std::env::remove_var("HERMES_COPILOT_ACP_ARGS");
        }
        assert_eq!(resolve_args(), vec!["--acp", "--stdio"]);
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("foo_bar"), "Foo_Bar");
        assert_eq!(title_case("hello world"), "Hello World");
    }

    #[test]
    fn test_client_defaults() {
        let client = CopilotACPClient::new(ClientOptions::default());
        assert_eq!(client.api_key, "copilot-acp");
        assert_eq!(client.base_url, ACP_MARKER_BASE_URL);
        assert!(!client.acp_cwd.is_empty());
    }

    #[test]
    fn test_handle_server_message_session_update_collects_text() {
        // We need a ChildStdin which is hard to fabricate; instead test the
        // parsing logic via a smaller surface by replicating chunk extraction.
        let msg = json!({
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "hello"}
                }
            }
        });
        // Manually emulate the chunk-routing branch.
        let update = msg["params"]["update"].clone();
        let kind = update["sessionUpdate"].as_str().unwrap();
        let chunk = update["content"]["text"].as_str().unwrap();
        assert_eq!(kind, "agent_message_chunk");
        assert_eq!(chunk, "hello");
    }
}
