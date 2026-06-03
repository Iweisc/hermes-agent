//! AWS Bedrock Converse API adapter for Hermes Agent (native Rust port).
//!
//! Native port of `agent/bedrock_adapter.py`. Provides the message/tool format
//! conversion between the OpenAI chat shape and the Bedrock Converse API shape,
//! response normalisation back to an OpenAI-compatible structure, streaming
//! event assembly, model discovery request/response handling, credential/region
//! detection, and Bedrock-specific error classification.
//!
//! ## Network note
//!
//! The Python original drove the Bedrock control/runtime planes through `boto3`,
//! which transparently handled AWS SigV4 request signing and the AWS event-stream
//! binary framing for `converse_stream()`. There is no AWS SDK crate available in
//! this workspace, so the request *construction* (`build_converse_kwargs`) and the
//! response *parsing* (`normalize_converse_response`, the streaming assembler, and
//! the discovery response decoders) are ported here faithfully against the exact
//! Bedrock JSON shapes. A caller that already holds an AWS-signed HTTP client (or a
//! future SigV4 layer) can feed the request body produced here to
//! `bedrock-runtime.converse` and pass the decoded JSON back into the normalisers.
//!
//! The credential/region helpers (`resolve_aws_auth_env_var`, `has_aws_credentials`,
//! `resolve_bedrock_region`) implement the environment-variable fast path that the
//! Python code used before falling back to the boto3 credential chain. The boto3
//! credential-resolver fallback (IMDS / ECS / EKS) is not reproducible without an
//! AWS SDK, so those branches are documented as best-effort env-only here.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::{json, Map, Value};

// ===========================================================================
// AWS credential detection
// ===========================================================================

/// Priority-ordered list of AWS credential env vars, mirroring OpenClaw's
/// `resolveAwsSdkEnvVarName()`.
pub const AWS_CREDENTIAL_ENV_VARS: &[&str] = &[
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
];

fn env_get<'a>(env: &'a HashMap<String, String>, key: &str) -> &'a str {
    env.get(key).map(|s| s.trim()).unwrap_or("")
}

/// Snapshot the process environment into a map (used as the default for the
/// credential/region helpers, matching the Python `env or os.environ`).
pub fn process_env() -> HashMap<String, String> {
    std::env::vars().collect()
}

/// Return the name of the AWS auth source that is active, or `None`.
///
/// Mirrors Python `resolve_aws_auth_env_var`. The boto3 implicit-credential
/// fallback (`"iam-role"`) is not reproducible without an AWS SDK, so only the
/// environment-variable sources are detected here.
pub fn resolve_aws_auth_env_var(env: &HashMap<String, String>) -> Option<String> {
    if !env_get(env, "AWS_BEARER_TOKEN_BEDROCK").is_empty() {
        return Some("AWS_BEARER_TOKEN_BEDROCK".to_string());
    }
    if !env_get(env, "AWS_ACCESS_KEY_ID").is_empty()
        && !env_get(env, "AWS_SECRET_ACCESS_KEY").is_empty()
    {
        return Some("AWS_ACCESS_KEY_ID".to_string());
    }
    if !env_get(env, "AWS_PROFILE").is_empty() {
        return Some("AWS_PROFILE".to_string());
    }
    if !env_get(env, "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_empty() {
        return Some("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI".to_string());
    }
    if !env_get(env, "AWS_WEB_IDENTITY_TOKEN_FILE").is_empty() {
        return Some("AWS_WEB_IDENTITY_TOKEN_FILE".to_string());
    }
    None
}

/// Return `true` if any AWS credential source is detected via environment vars.
///
/// Mirrors Python `has_aws_credentials` for the env-var fast path. The boto3
/// credential-resolver fallback (IMDS / ECS task role / Lambda) cannot be
/// reproduced without an AWS SDK and is therefore omitted.
pub fn has_aws_credentials(env: &HashMap<String, String>) -> bool {
    resolve_aws_auth_env_var(env).is_some()
}

/// Resolve the AWS region for Bedrock API calls.
///
/// Priority: `AWS_REGION` → `AWS_DEFAULT_REGION` → `us-east-1`. The boto3/config
/// (`~/.aws/config`) fallback is not reproducible without an AWS SDK.
pub fn resolve_bedrock_region(env: &HashMap<String, String>) -> String {
    let explicit = {
        let r = env_get(env, "AWS_REGION");
        if !r.is_empty() {
            r
        } else {
            env_get(env, "AWS_DEFAULT_REGION")
        }
    };
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    "us-east-1".to_string()
}

// ===========================================================================
// Tool-calling capability detection
// ===========================================================================

/// Patterns identifying Bedrock models that reject `toolConfig` in Converse.
pub const NON_TOOL_CALLING_PATTERNS: &[&str] = &[
    "deepseek.r1",
    "deepseek-r1",
    "stability.",
    "cohere.embed",
    "amazon.titan-embed",
];

/// Return `true` if the model is expected to support tool/function calling.
/// Unknown models default to `true`.
pub fn model_supports_tool_use(model_id: &str) -> bool {
    let model_lower = model_id.to_lowercase();
    !NON_TOOL_CALLING_PATTERNS
        .iter()
        .any(|p| model_lower.contains(p))
}

/// Return `true` if the model is an Anthropic Claude model on Bedrock (after
/// stripping a regional inference-profile prefix).
pub fn is_anthropic_bedrock_model(model_id: &str) -> bool {
    let mut model_lower = model_id.to_lowercase();
    for prefix in ["us.", "global.", "eu.", "ap.", "jp."] {
        if model_lower.starts_with(prefix) {
            model_lower = model_lower[prefix.len()..].to_string();
            break;
        }
    }
    model_lower.starts_with("anthropic.claude")
}

// ===========================================================================
// Message format conversion: OpenAI -> Bedrock Converse
// ===========================================================================

/// Convert OpenAI-format tool definitions to Bedrock Converse `toolConfig`
/// tool specs. Returns an empty vec when `tools` is empty.
pub fn convert_tools_to_converse(tools: &[Value]) -> Vec<Value> {
    if tools.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(tools.len());
    for t in tools {
        let fn_obj = t.get("function").cloned().unwrap_or_else(|| json!({}));
        let name = fn_obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let description = fn_obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let parameters = fn_obj
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        result.push(json!({
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {"json": parameters},
            }
        }));
    }
    result
}

/// Convert OpenAI message content (string, array, or null) into Converse
/// content blocks. Empty text blocks are replaced with a single-space block
/// because Bedrock rejects empty text blocks.
pub fn convert_content_to_converse(content: &Value) -> Vec<Value> {
    match content {
        Value::Null => vec![json!({"text": " "})],
        Value::String(s) => {
            if s.trim().is_empty() {
                vec![json!({"text": " "})]
            } else {
                vec![json!({"text": s})]
            }
        }
        Value::Array(parts) => {
            let mut blocks: Vec<Value> = Vec::new();
            for part in parts {
                match part {
                    Value::String(s) => {
                        blocks.push(json!({"text": s}));
                    }
                    Value::Object(_) => {
                        let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
                        if part_type == "text" {
                            let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                            if text.is_empty() {
                                blocks.push(json!({"text": " "}));
                            } else {
                                blocks.push(json!({"text": text}));
                            }
                        } else if part_type == "image_url" {
                            let url = part
                                .get("image_url")
                                .and_then(|iu| iu.get("url"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            if url.starts_with("data:") {
                                // data:image/jpeg;base64,/9j/4AAQ...
                                let (header, data) = match url.split_once(',') {
                                    Some((h, d)) => (h.to_string(), d.to_string()),
                                    None => (url.to_string(), String::new()),
                                };
                                let mut media_type = "image/jpeg".to_string();
                                if let Some(rest) = header.strip_prefix("data:") {
                                    let mime_part = rest.split(';').next().unwrap_or("");
                                    if !mime_part.is_empty() {
                                        media_type = mime_part.to_string();
                                    }
                                }
                                let format = if media_type.contains('/') {
                                    media_type.rsplit('/').next().unwrap_or("jpeg").to_string()
                                } else {
                                    "jpeg".to_string()
                                };
                                blocks.push(json!({
                                    "image": {
                                        "format": format,
                                        "source": {"bytes": data},
                                    }
                                }));
                            } else {
                                blocks.push(json!({"text": format!("[Image: {}]", url)}));
                            }
                        }
                        // other dict types are ignored
                    }
                    _ => {
                        // non-str, non-dict array entries are skipped
                    }
                }
            }
            if blocks.is_empty() {
                vec![json!({"text": " "})]
            } else {
                blocks
            }
        }
        other => vec![json!({"text": value_to_str(other)})],
    }
}

fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn content_str_nonempty(content: &Value) -> Option<&str> {
    match content {
        Value::String(s) if !s.trim().is_empty() => Some(s.as_str()),
        _ => None,
    }
}

/// Convert OpenAI-format messages to Bedrock Converse format.
///
/// Returns `(system_prompt, converse_messages)` where `system_prompt` is
/// `Some(blocks)` when any system content was found. Consecutive same-role
/// messages are merged and the conversation is padded so it both starts and
/// ends with a `user` message.
pub fn convert_messages_to_converse(messages: &[Value]) -> (Option<Vec<Value>>, Vec<Value>) {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut converse_msgs: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
        let content = msg.get("content").cloned().unwrap_or(Value::Null);

        match role {
            "system" => {
                if let Some(s) = content_str_nonempty(&content) {
                    system_blocks.push(json!({"text": s}));
                } else if let Value::Array(parts) = &content {
                    for part in parts {
                        if let Value::Object(_) = part {
                            if part.get("type").and_then(Value::as_str) == Some("text") {
                                let text =
                                    part.get("text").and_then(Value::as_str).unwrap_or("");
                                system_blocks.push(json!({"text": text}));
                            }
                        } else if let Value::String(s) = part {
                            system_blocks.push(json!({"text": s}));
                        }
                    }
                }
            }
            "tool" => {
                let tool_call_id = msg
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let result_content = match &content {
                    Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
                };
                let tool_result_block = json!({
                    "toolResult": {
                        "toolUseId": tool_call_id,
                        "content": [{"text": result_content}],
                    }
                });
                if last_role_is(&converse_msgs, "user") {
                    push_into_last(&mut converse_msgs, tool_result_block);
                } else {
                    converse_msgs.push(json!({
                        "role": "user",
                        "content": [tool_result_block],
                    }));
                }
            }
            "assistant" => {
                let mut content_blocks: Vec<Value> = Vec::new();
                if let Some(s) = content_str_nonempty(&content) {
                    content_blocks.push(json!({"text": s}));
                } else if let Value::Array(_) = &content {
                    content_blocks.extend(convert_content_to_converse(&content));
                }

                if let Some(Value::Array(tool_calls)) = msg.get("tool_calls").map(|v| v.clone()).as_ref()
                {
                    for tc in tool_calls {
                        let fn_obj = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                        let args_dict = parse_arguments(fn_obj.get("arguments"));
                        content_blocks.push(json!({
                            "toolUse": {
                                "toolUseId": tc.get("id").and_then(Value::as_str).unwrap_or(""),
                                "name": fn_obj.get("name").and_then(Value::as_str).unwrap_or(""),
                                "input": args_dict,
                            }
                        }));
                    }
                }

                if content_blocks.is_empty() {
                    content_blocks = vec![json!({"text": " "})];
                }

                if last_role_is(&converse_msgs, "assistant") {
                    extend_last(&mut converse_msgs, content_blocks);
                } else {
                    converse_msgs.push(json!({
                        "role": "assistant",
                        "content": content_blocks,
                    }));
                }
            }
            "user" => {
                let content_blocks = convert_content_to_converse(&content);
                if last_role_is(&converse_msgs, "user") {
                    extend_last(&mut converse_msgs, content_blocks);
                } else {
                    converse_msgs.push(json!({
                        "role": "user",
                        "content": content_blocks,
                    }));
                }
            }
            _ => {
                // Unknown roles are ignored, matching the Python control flow.
            }
        }
    }

    // First message must be from the user.
    if let Some(first) = converse_msgs.first() {
        if first.get("role").and_then(Value::as_str) != Some("user") {
            converse_msgs.insert(0, json!({"role": "user", "content": [{"text": " "}]}));
        }
    }
    // Last message must be from the user.
    if let Some(last) = converse_msgs.last() {
        if last.get("role").and_then(Value::as_str) != Some("user") {
            converse_msgs.push(json!({"role": "user", "content": [{"text": " "}]}));
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(system_blocks)
    };
    (system, converse_msgs)
}

fn parse_arguments(args: Option<&Value>) -> Value {
    match args {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
        Some(other) => other.clone(),
        None => json!({}),
    }
}

fn last_role_is(msgs: &[Value], role: &str) -> bool {
    msgs.last()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some(role)
}

fn push_into_last(msgs: &mut [Value], block: Value) {
    if let Some(last) = msgs.last_mut() {
        if let Some(Value::Array(arr)) = last.get_mut("content") {
            arr.push(block);
        }
    }
}

fn extend_last(msgs: &mut [Value], blocks: Vec<Value>) {
    if let Some(last) = msgs.last_mut() {
        if let Some(Value::Array(arr)) = last.get_mut("content") {
            arr.extend(blocks);
        }
    }
}

// ===========================================================================
// OpenAI-compatible response shapes
// ===========================================================================

/// OpenAI-compatible function payload inside a tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string (matches OpenAI's `function.arguments`).
    pub arguments: String,
}

/// OpenAI-compatible tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    pub function: FunctionCall,
}

/// OpenAI-compatible assistant message.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    /// `None` when there is no text content (e.g. tool-only turns).
    pub content: Option<String>,
    /// `None` when there are no tool calls.
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI-compatible token usage block.
#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// OpenAI-compatible choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub model: String,
}

// ===========================================================================
// Response format conversion: Bedrock Converse -> OpenAI
// ===========================================================================

/// Map Bedrock Converse stop reasons to OpenAI `finish_reason` values.
pub fn converse_stop_reason_to_openai(stop_reason: &str) -> String {
    match stop_reason {
        "end_turn" => "stop",
        "stop_sequence" => "stop",
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "content_filtered" => "content_filter",
        "guardrail_intervened" => "content_filter",
        _ => "stop",
    }
    .to_string()
}

fn usage_field(usage_data: &Value, key: &str) -> u64 {
    usage_data.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Convert a Bedrock Converse API response (decoded JSON) into an
/// OpenAI-compatible [`ChatCompletion`].
pub fn normalize_converse_response(response: &Value) -> ChatCompletion {
    let content_blocks = response
        .get("output")
        .and_then(|o| o.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let stop_reason = response
        .get("stopReason")
        .and_then(Value::as_str)
        .unwrap_or("end_turn");

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in &content_blocks {
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            text_parts.push(text.to_string());
        } else if let Some(tu) = block.get("toolUse") {
            let input = tu.get("input").cloned().unwrap_or_else(|| json!({}));
            tool_calls.push(ToolCall {
                id: tu.get("toolUseId").and_then(Value::as_str).unwrap_or("").to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tu.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                    arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()),
                },
            });
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };
    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls.clone())
    };

    let msg = ChatMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tool_calls_opt,
    };

    let usage_data = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    let prompt_tokens = usage_field(&usage_data, "inputTokens");
    let completion_tokens = usage_field(&usage_data, "outputTokens");
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    let mut finish_reason = converse_stop_reason_to_openai(stop_reason);
    if !tool_calls.is_empty() && finish_reason == "stop" {
        finish_reason = "tool_calls".to_string();
    }

    let choice = Choice {
        index: 0,
        message: msg,
        finish_reason,
    };

    ChatCompletion {
        choices: vec![choice],
        usage,
        model: response
            .get("modelId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    }
}

// ===========================================================================
// Streaming response conversion
// ===========================================================================

/// Optional real-time callbacks for [`stream_converse_with_callbacks`].
///
/// Mirrors the Python keyword callbacks. Each closure is invoked with the same
/// arguments and at the same points as the original.
#[derive(Default)]
pub struct StreamCallbacks<'a> {
    /// Called with each text chunk while no tool_use block has been seen yet.
    pub on_text_delta: Option<&'a mut dyn FnMut(&str)>,
    /// Called with the tool name when a toolUse block begins.
    pub on_tool_start: Option<&'a mut dyn FnMut(&str)>,
    /// Called with reasoning/thinking text chunks.
    pub on_reasoning_delta: Option<&'a mut dyn FnMut(&str)>,
    /// Called once per event; returning `true` stops streaming.
    pub on_interrupt_check: Option<&'a mut dyn FnMut() -> bool>,
}

/// Consume a decoded Bedrock ConverseStream event list and assemble an
/// OpenAI-compatible [`ChatCompletion`], firing the supplied callbacks.
///
/// `events` is the decoded `stream` array of the `converse_stream()` response
/// (each entry being one event object such as `{"contentBlockDelta": {...}}`).
pub fn stream_converse_with_callbacks(
    events: &[Value],
    callbacks: &mut StreamCallbacks<'_>,
) -> ChatCompletion {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool: Option<(String, String, String)> = None; // (id, name, input_json)
    let mut current_text_buffer: Vec<String> = Vec::new();
    let mut has_tool_use = false;
    let mut stop_reason = "end_turn".to_string();
    let mut usage_input: u64 = 0;
    let mut usage_output: u64 = 0;

    for event in events {
        if let Some(cb) = callbacks.on_interrupt_check.as_mut() {
            if cb() {
                break;
            }
        }

        if let Some(start) = event.get("contentBlockStart").and_then(|c| c.get("start")) {
            if let Some(tu) = start.get("toolUse") {
                has_tool_use = true;
                if !current_text_buffer.is_empty() {
                    text_parts.push(current_text_buffer.concat());
                    current_text_buffer.clear();
                }
                let id = tu.get("toolUseId").and_then(Value::as_str).unwrap_or("").to_string();
                let name = tu.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                if let Some(cb) = callbacks.on_tool_start.as_mut() {
                    cb(&name);
                }
                current_tool = Some((id, name, String::new()));
            }
        } else if let Some(delta) = event.get("contentBlockDelta").and_then(|c| c.get("delta")) {
            if let Some(text) = delta.get("text").and_then(Value::as_str) {
                current_text_buffer.push(text.to_string());
                if !has_tool_use {
                    if let Some(cb) = callbacks.on_text_delta.as_mut() {
                        cb(text);
                    }
                }
            } else if let Some(tu) = delta.get("toolUse") {
                if let Some((_, _, ref mut input_json)) = current_tool {
                    if let Some(input) = tu.get("input").and_then(Value::as_str) {
                        input_json.push_str(input);
                    }
                }
            } else if let Some(reasoning) = delta.get("reasoningContent") {
                if reasoning.is_object() {
                    if let Some(thinking) = reasoning.get("text").and_then(Value::as_str) {
                        if !thinking.is_empty() {
                            if let Some(cb) = callbacks.on_reasoning_delta.as_mut() {
                                cb(thinking);
                            }
                        }
                    }
                }
            }
        } else if event.get("contentBlockStop").is_some() {
            if let Some((id, name, input_json)) = current_tool.take() {
                let input_dict: Value = if input_json.is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&input_json).unwrap_or_else(|_| json!({}))
                };
                tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input_dict)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            } else if !current_text_buffer.is_empty() {
                text_parts.push(current_text_buffer.concat());
                current_text_buffer.clear();
            }
        } else if let Some(ms) = event.get("messageStop") {
            stop_reason = ms
                .get("stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_string();
        } else if let Some(meta) = event.get("metadata") {
            if let Some(meta_usage) = meta.get("usage") {
                usage_input = usage_field(meta_usage, "inputTokens");
                usage_output = usage_field(meta_usage, "outputTokens");
            }
        }
    }

    if !current_text_buffer.is_empty() {
        text_parts.push(current_text_buffer.concat());
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };
    let tool_calls_present = !tool_calls.is_empty();
    let tool_calls_opt = if tool_calls_present {
        Some(tool_calls)
    } else {
        None
    };

    let msg = ChatMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tool_calls_opt,
    };

    let usage = Usage {
        prompt_tokens: usage_input,
        completion_tokens: usage_output,
        total_tokens: usage_input + usage_output,
    };

    let mut finish_reason = converse_stop_reason_to_openai(&stop_reason);
    if tool_calls_present && finish_reason == "stop" {
        finish_reason = "tool_calls".to_string();
    }

    let choice = Choice {
        index: 0,
        message: msg,
        finish_reason,
    };

    ChatCompletion {
        choices: vec![choice],
        usage,
        model: String::new(),
    }
}

/// Consume a decoded Bedrock ConverseStream event list with no callbacks,
/// returning the assembled OpenAI-compatible response.
pub fn normalize_converse_stream_events(events: &[Value]) -> ChatCompletion {
    let mut cb = StreamCallbacks::default();
    stream_converse_with_callbacks(events, &mut cb)
}

// ===========================================================================
// High-level API: build Bedrock Converse request body
// ===========================================================================

/// Parameters for a Bedrock Converse request. `None` fields are omitted from
/// the produced request body, matching the Python kwargs behaviour.
#[derive(Debug, Clone, Default)]
pub struct ConverseParams<'a> {
    pub model: &'a str,
    pub messages: &'a [Value],
    pub tools: Option<&'a [Value]>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop_sequences: Option<&'a [String]>,
    pub guardrail_config: Option<&'a Value>,
}

/// Build the JSON request body for `bedrock-runtime.converse` /
/// `converse_stream` from OpenAI-format inputs.
///
/// Tools for known non-tool-calling models are stripped (and a warning is
/// logged), matching the Python `build_converse_kwargs`.
pub fn build_converse_kwargs(params: &ConverseParams<'_>) -> Value {
    let (system_prompt, converse_messages) = convert_messages_to_converse(params.messages);

    let mut inference_config = Map::new();
    inference_config.insert("maxTokens".to_string(), json!(params.max_tokens));
    if let Some(t) = params.temperature {
        inference_config.insert("temperature".to_string(), json!(t));
    }
    if let Some(p) = params.top_p {
        inference_config.insert("topP".to_string(), json!(p));
    }
    if let Some(stops) = params.stop_sequences {
        if !stops.is_empty() {
            inference_config.insert("stopSequences".to_string(), json!(stops));
        }
    }

    let mut kwargs = Map::new();
    kwargs.insert("modelId".to_string(), json!(params.model));
    kwargs.insert("messages".to_string(), json!(converse_messages));
    kwargs.insert(
        "inferenceConfig".to_string(),
        Value::Object(inference_config),
    );

    if let Some(system) = system_prompt {
        kwargs.insert("system".to_string(), json!(system));
    }

    if let Some(tools) = params.tools {
        if !tools.is_empty() {
            let converse_tools = convert_tools_to_converse(tools);
            if !converse_tools.is_empty() {
                if model_supports_tool_use(params.model) {
                    kwargs.insert("toolConfig".to_string(), json!({"tools": converse_tools}));
                } else {
                    log::warn!(
                        "Model {} does not support tool calling — tools stripped. \
                         The agent will operate in text-only mode.",
                        params.model
                    );
                }
            }
        }
    }

    if let Some(guardrail) = params.guardrail_config {
        kwargs.insert("guardrailConfig".to_string(), guardrail.clone());
    }

    Value::Object(kwargs)
}

// ===========================================================================
// Model discovery (response parsing)
// ===========================================================================

/// One discovered Bedrock model.
#[derive(Debug, Clone, PartialEq)]
pub struct BedrockModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub streaming: bool,
}

struct DiscoveryCacheEntry {
    timestamp: u64,
    models: Vec<BedrockModelInfo>,
}

fn discovery_cache() -> &'static Mutex<HashMap<String, DiscoveryCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<String, DiscoveryCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Discovery cache TTL in seconds (1 hour), matching the Python constant.
pub const DISCOVERY_CACHE_TTL_SECONDS: u64 = 3600;

/// Clear the model discovery cache. Used in tests.
pub fn reset_discovery_cache() {
    discovery_cache().lock().unwrap().clear();
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn discovery_cache_key(region: &str, provider_filter: Option<&[String]>) -> String {
    let mut filters: Vec<String> = provider_filter.unwrap_or(&[]).to_vec();
    filters.sort();
    format!("{}:{}", region, filters.join(","))
}

/// Extract the model provider from a Bedrock model ARN.
///
/// `arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2` → `anthropic`
pub fn extract_provider_from_arn(arn: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"foundation-model/([^.]+)").unwrap());
    re.captures(arn)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_default()
}

/// Parse the decoded `list_foundation_models` + `list_inference_profiles`
/// responses into the filtered, sorted model list.
///
/// This is the pure decoding/filtering/sorting half of the Python
/// `discover_bedrock_models` — it takes the already-fetched control-plane JSON
/// (which `boto3` produced) and applies the exact same provider filtering,
/// active/streaming/text-output gating, dedup, and sort. Pass
/// `inference_profiles` as the concatenation of all paginated
/// `inferenceProfileSummaries` arrays.
pub fn parse_discovered_models(
    foundation_models_response: &Value,
    inference_profiles: &[Value],
    provider_filter: Option<&[String]>,
) -> Vec<BedrockModelInfo> {
    let mut models: Vec<BedrockModelInfo> = Vec::new();
    let mut seen_ids: BTreeSet<String> = BTreeSet::new();
    let filter_set: BTreeSet<String> = provider_filter
        .unwrap_or(&[])
        .iter()
        .map(|f| f.to_lowercase())
        .collect();

    // 1. Foundation models
    if let Some(summaries) = foundation_models_response
        .get("modelSummaries")
        .and_then(Value::as_array)
    {
        for summary in summaries {
            let model_id = summary
                .get("modelId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if model_id.is_empty() {
                continue;
            }

            if !filter_set.is_empty() {
                let provider_name = summary
                    .get("providerName")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_lowercase();
                let model_prefix = if model_id.contains('.') {
                    model_id.split('.').next().unwrap_or("").to_lowercase()
                } else {
                    String::new()
                };
                if !filter_set.contains(&provider_name) && !filter_set.contains(&model_prefix) {
                    continue;
                }
            }

            let status = summary
                .get("modelLifecycle")
                .and_then(|l| l.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_uppercase();
            if status != "ACTIVE" {
                continue;
            }
            if !summary
                .get("responseStreamingSupported")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                continue;
            }
            let output_mods = str_vec(summary.get("outputModalities"));
            if !output_mods.iter().any(|m| m == "TEXT") {
                continue;
            }

            let name = {
                let n = summary
                    .get("modelName")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if n.is_empty() {
                    model_id.clone()
                } else {
                    n.to_string()
                }
            };

            models.push(BedrockModelInfo {
                id: model_id.clone(),
                name,
                provider: summary
                    .get("providerName")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                input_modalities: str_vec(summary.get("inputModalities")),
                output_modalities: output_mods,
                streaming: true,
            });
            seen_ids.insert(model_id.to_lowercase());
        }
    }

    // 2. Inference profiles
    for profile in inference_profiles {
        let profile_id = profile
            .get("inferenceProfileId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if profile_id.is_empty() {
            continue;
        }
        if profile.get("status").and_then(Value::as_str) != Some("ACTIVE") {
            continue;
        }
        if seen_ids.contains(&profile_id.to_lowercase()) {
            continue;
        }

        if !filter_set.is_empty() {
            let profile_models = profile.get("models").and_then(Value::as_array);
            let matches = profile_models.map_or(false, |pms| {
                pms.iter().any(|m| {
                    let arn = m.get("modelArn").and_then(Value::as_str).unwrap_or("");
                    filter_set.contains(&extract_provider_from_arn(arn).to_lowercase())
                })
            });
            if !matches {
                continue;
            }
        }

        let name = {
            let n = profile
                .get("inferenceProfileName")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if n.is_empty() {
                profile_id.clone()
            } else {
                n.to_string()
            }
        };

        models.push(BedrockModelInfo {
            id: profile_id.clone(),
            name,
            provider: "inference-profile".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            streaming: true,
        });
        seen_ids.insert(profile_id.to_lowercase());
    }

    // Sort: global cross-region profiles first, then alphabetical by name.
    models.sort_by(|a, b| {
        let a_key = (if a.id.starts_with("global.") { 0 } else { 1 }, a.name.to_lowercase());
        let b_key = (if b.id.starts_with("global.") { 0 } else { 1 }, b.name.to_lowercase());
        a_key.cmp(&b_key)
    });

    models
}

fn str_vec(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Store an already-computed model list in the discovery cache (the cache
/// half of `discover_bedrock_models`). Returns the cached list. Callers that
/// have just fetched and parsed the control-plane data use this to populate
/// the per-region cache.
pub fn cache_discovered_models(
    region: &str,
    provider_filter: Option<&[String]>,
    models: Vec<BedrockModelInfo>,
) -> Vec<BedrockModelInfo> {
    let key = discovery_cache_key(region, provider_filter);
    let mut cache = discovery_cache().lock().unwrap();
    cache.insert(
        key,
        DiscoveryCacheEntry {
            timestamp: now_secs(),
            models: models.clone(),
        },
    );
    models
}

/// Look up a non-expired cached discovery result, if any.
pub fn cached_discovered_models(
    region: &str,
    provider_filter: Option<&[String]>,
) -> Option<Vec<BedrockModelInfo>> {
    let key = discovery_cache_key(region, provider_filter);
    let cache = discovery_cache().lock().unwrap();
    if let Some(entry) = cache.get(&key) {
        if now_secs().saturating_sub(entry.timestamp) < DISCOVERY_CACHE_TTL_SECONDS {
            return Some(entry.models.clone());
        }
    }
    None
}

/// Flatten a discovered model list into model-ID strings.
pub fn model_ids(models: &[BedrockModelInfo]) -> Vec<String> {
    models.iter().map(|m| m.id.clone()).collect()
}

// ===========================================================================
// Error classification — Bedrock-specific
// ===========================================================================

fn context_overflow_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)ValidationException.*(?:input is too long|max input token|input token.*exceed)").unwrap(),
            Regex::new(r"(?i)ValidationException.*(?:exceeds? the (?:maximum|max) (?:number of )?(?:input )?tokens)").unwrap(),
            Regex::new(r"(?i)ModelStreamErrorException.*(?:Input is too long|too many input tokens)").unwrap(),
        ]
    })
}

fn throttle_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)ThrottlingException").unwrap(),
            Regex::new(r"(?i)Too many concurrent requests").unwrap(),
            Regex::new(r"(?i)ServiceQuotaExceededException").unwrap(),
        ]
    })
}

fn overload_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)ModelNotReadyException").unwrap(),
            Regex::new(r"(?i)ModelTimeoutException").unwrap(),
            Regex::new(r"(?i)InternalServerException").unwrap(),
        ]
    })
}

/// Return `true` if the error indicates the input context was too large.
pub fn is_context_overflow_error(error_message: &str) -> bool {
    context_overflow_patterns()
        .iter()
        .any(|p| p.is_match(error_message))
}

/// Classify a Bedrock error for retry/failover decisions.
///
/// Returns one of `"context_overflow"`, `"rate_limit"`, `"overloaded"`, or
/// `"unknown"`.
pub fn classify_bedrock_error(error_message: &str) -> &'static str {
    if is_context_overflow_error(error_message) {
        return "context_overflow";
    }
    if throttle_patterns().iter().any(|p| p.is_match(error_message)) {
        return "rate_limit";
    }
    if overload_patterns().iter().any(|p| p.is_match(error_message)) {
        return "overloaded";
    }
    "unknown"
}

// ===========================================================================
// Bedrock model context lengths
// ===========================================================================

/// Static fallback table for Bedrock model context windows. Ordered exactly
/// as the Python dict.
pub const BEDROCK_CONTEXT_LENGTHS: &[(&str, u64)] = &[
    ("anthropic.claude-opus-4-6", 200_000),
    ("anthropic.claude-sonnet-4-6", 200_000),
    ("anthropic.claude-sonnet-4-5", 200_000),
    ("anthropic.claude-haiku-4-5", 200_000),
    ("anthropic.claude-opus-4", 200_000),
    ("anthropic.claude-sonnet-4", 200_000),
    ("anthropic.claude-3-5-sonnet", 200_000),
    ("anthropic.claude-3-5-haiku", 200_000),
    ("anthropic.claude-3-opus", 200_000),
    ("anthropic.claude-3-sonnet", 200_000),
    ("anthropic.claude-3-haiku", 200_000),
    ("amazon.nova-pro", 300_000),
    ("amazon.nova-lite", 300_000),
    ("amazon.nova-micro", 128_000),
    ("meta.llama4-maverick", 128_000),
    ("meta.llama4-scout", 128_000),
    ("meta.llama3-3-70b-instruct", 128_000),
    ("mistral.mistral-large", 128_000),
    ("deepseek.v3", 128_000),
];

/// Default context length for unknown Bedrock models.
pub const BEDROCK_DEFAULT_CONTEXT_LENGTH: u64 = 128_000;

/// Look up the context window size for a Bedrock model using longest-substring
/// matching, so versioned IDs resolve correctly.
pub fn get_bedrock_context_length(model_id: &str) -> u64 {
    let model_lower = model_id.to_lowercase();
    let mut best_key = "";
    let mut best_val = BEDROCK_DEFAULT_CONTEXT_LENGTH;
    for (key, val) in BEDROCK_CONTEXT_LENGTHS {
        if model_lower.contains(key) && key.len() > best_key.len() {
            best_key = key;
            best_val = *val;
        }
    }
    best_val
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn auth_env_var_priority() {
        assert_eq!(
            resolve_aws_auth_env_var(&env(&[("AWS_BEARER_TOKEN_BEDROCK", "tok")])),
            Some("AWS_BEARER_TOKEN_BEDROCK".to_string())
        );
        assert_eq!(
            resolve_aws_auth_env_var(&env(&[
                ("AWS_ACCESS_KEY_ID", "ak"),
                ("AWS_SECRET_ACCESS_KEY", "sk")
            ])),
            Some("AWS_ACCESS_KEY_ID".to_string())
        );
        // Access key without secret does not count.
        assert_eq!(
            resolve_aws_auth_env_var(&env(&[("AWS_ACCESS_KEY_ID", "ak")])),
            None
        );
        assert_eq!(
            resolve_aws_auth_env_var(&env(&[("AWS_PROFILE", "prod")])),
            Some("AWS_PROFILE".to_string())
        );
        assert_eq!(resolve_aws_auth_env_var(&env(&[])), None);
        assert!(has_aws_credentials(&env(&[("AWS_PROFILE", "x")])));
        assert!(!has_aws_credentials(&env(&[])));
    }

    #[test]
    fn region_resolution() {
        assert_eq!(resolve_bedrock_region(&env(&[("AWS_REGION", "eu-west-1")])), "eu-west-1");
        assert_eq!(
            resolve_bedrock_region(&env(&[("AWS_DEFAULT_REGION", "ap-south-1")])),
            "ap-south-1"
        );
        assert_eq!(resolve_bedrock_region(&env(&[])), "us-east-1");
    }

    #[test]
    fn tool_support_detection() {
        assert!(model_supports_tool_use("anthropic.claude-sonnet-4-6"));
        assert!(!model_supports_tool_use("us.deepseek.r1-v1:0"));
        assert!(!model_supports_tool_use("amazon.titan-embed-text-v2"));
        assert!(!model_supports_tool_use("stability.stable-diffusion"));
    }

    #[test]
    fn anthropic_detection() {
        assert!(is_anthropic_bedrock_model("anthropic.claude-3-haiku"));
        assert!(is_anthropic_bedrock_model("us.anthropic.claude-sonnet-4-6"));
        assert!(is_anthropic_bedrock_model("global.anthropic.claude-opus-4"));
        assert!(!is_anthropic_bedrock_model("amazon.nova-pro"));
        assert!(!is_anthropic_bedrock_model("meta.llama4-scout"));
    }

    #[test]
    fn convert_tools() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        })];
        let out = convert_tools_to_converse(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["toolSpec"]["name"], "get_weather");
        assert_eq!(out[0]["toolSpec"]["inputSchema"]["json"]["type"], "object");
        assert!(convert_tools_to_converse(&[]).is_empty());
    }

    #[test]
    fn content_conversion_text_and_empty() {
        assert_eq!(convert_content_to_converse(&Value::Null), vec![json!({"text": " "})]);
        assert_eq!(
            convert_content_to_converse(&json!("hello")),
            vec![json!({"text": "hello"})]
        );
        assert_eq!(
            convert_content_to_converse(&json!("   ")),
            vec![json!({"text": " "})]
        );
    }

    #[test]
    fn content_conversion_image_data_url() {
        let content = json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}}
        ]);
        let out = convert_content_to_converse(&content);
        assert_eq!(out[0], json!({"text": "look"}));
        assert_eq!(out[1]["image"]["format"], "png");
        assert_eq!(out[1]["image"]["source"]["bytes"], "ABC123");
    }

    #[test]
    fn content_conversion_remote_image() {
        let content = json!([
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        ]);
        let out = convert_content_to_converse(&content);
        assert_eq!(out[0], json!({"text": "[Image: https://x/y.png]"}));
    }

    #[test]
    fn messages_system_extraction_and_alternation() {
        let messages = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let (system, msgs) = convert_messages_to_converse(&messages);
        assert_eq!(system, Some(vec![json!({"text": "be nice"})]));
        // last message must be user -> a padding user message is appended
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["text"], " ");
    }

    #[test]
    fn messages_tool_call_and_result() {
        let messages = vec![
            json!({"role": "user", "content": "weather?"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "sunny"}),
        ];
        let (_system, msgs) = convert_messages_to_converse(&messages);
        // user, assistant(toolUse), user(toolResult)
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["toolUse"]["toolUseId"], "call_1");
        assert_eq!(msgs[1]["content"][0]["toolUse"]["input"]["city"], "NYC");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["toolResult"]["toolUseId"], "call_1");
    }

    #[test]
    fn messages_consecutive_user_merge() {
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "user", "content": "b"}),
        ];
        let (_s, msgs) = convert_messages_to_converse(&messages);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn messages_first_must_be_user() {
        let messages = vec![json!({"role": "assistant", "content": "hi"})];
        let (_s, msgs) = convert_messages_to_converse(&messages);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs.last().unwrap()["role"], "user");
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(converse_stop_reason_to_openai("end_turn"), "stop");
        assert_eq!(converse_stop_reason_to_openai("tool_use"), "tool_calls");
        assert_eq!(converse_stop_reason_to_openai("max_tokens"), "length");
        assert_eq!(converse_stop_reason_to_openai("content_filtered"), "content_filter");
        assert_eq!(converse_stop_reason_to_openai("unknown_thing"), "stop");
    }

    #[test]
    fn normalize_response_text() {
        let resp = json!({
            "output": {"message": {"content": [{"text": "line1"}, {"text": "line2"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "modelId": "anthropic.claude-3-haiku"
        });
        let out = normalize_converse_response(&resp);
        assert_eq!(out.choices[0].message.content, Some("line1\nline2".to_string()));
        assert_eq!(out.choices[0].finish_reason, "stop");
        assert_eq!(out.usage.total_tokens, 15);
        assert_eq!(out.model, "anthropic.claude-3-haiku");
        assert!(out.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn normalize_response_tool_use_overrides_stop() {
        let resp = json!({
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "foo", "input": {"x": 1}}}
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 2}
        });
        let out = normalize_converse_response(&resp);
        assert_eq!(out.choices[0].finish_reason, "tool_calls");
        let tcs = out.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs[0].id, "t1");
        assert_eq!(tcs[0].function.name, "foo");
        assert_eq!(tcs[0].function.arguments, "{\"x\":1}");
        assert!(out.choices[0].message.content.is_none());
    }

    #[test]
    fn stream_assembly_text_and_tool() {
        let events = vec![
            json!({"contentBlockDelta": {"delta": {"text": "Hello "}}}),
            json!({"contentBlockDelta": {"delta": {"text": "world"}}}),
            json!({"contentBlockStop": {}}),
            json!({"contentBlockStart": {"start": {"toolUse": {"toolUseId": "tu1", "name": "calc"}}}}),
            json!({"contentBlockDelta": {"delta": {"toolUse": {"input": "{\"a\":"}}}}),
            json!({"contentBlockDelta": {"delta": {"toolUse": {"input": "1}"}}}}),
            json!({"contentBlockStop": {}}),
            json!({"messageStop": {"stopReason": "tool_use"}}),
            json!({"metadata": {"usage": {"inputTokens": 7, "outputTokens": 3}}}),
        ];
        let mut deltas: Vec<String> = Vec::new();
        let mut tools_started: Vec<String> = Vec::new();
        {
            let mut on_text = |t: &str| deltas.push(t.to_string());
            let mut on_tool = |n: &str| tools_started.push(n.to_string());
            let mut cb = StreamCallbacks {
                on_text_delta: Some(&mut on_text),
                on_tool_start: Some(&mut on_tool),
                ..Default::default()
            };
            let out = stream_converse_with_callbacks(&events, &mut cb);
            assert_eq!(out.choices[0].message.content, Some("Hello world".to_string()));
            let tcs = out.choices[0].message.tool_calls.as_ref().unwrap();
            assert_eq!(tcs[0].id, "tu1");
            assert_eq!(tcs[0].function.name, "calc");
            assert_eq!(tcs[0].function.arguments, "{\"a\":1}");
            assert_eq!(out.choices[0].finish_reason, "tool_calls");
            assert_eq!(out.usage.total_tokens, 10);
        }
        assert_eq!(deltas, vec!["Hello ".to_string(), "world".to_string()]);
        assert_eq!(tools_started, vec!["calc".to_string()]);
    }

    #[test]
    fn stream_interrupt_stops_early() {
        let events = vec![
            json!({"contentBlockDelta": {"delta": {"text": "a"}}}),
            json!({"contentBlockDelta": {"delta": {"text": "b"}}}),
        ];
        let mut count = 0;
        {
            let mut interrupt = || {
                count += 1;
                count > 1
            };
            let mut cb = StreamCallbacks {
                on_interrupt_check: Some(&mut interrupt),
                ..Default::default()
            };
            let out = stream_converse_with_callbacks(&events, &mut cb);
            // Only the first event processed before interrupt fires on the 2nd.
            assert_eq!(out.choices[0].message.content, Some("a".to_string()));
        }
    }

    #[test]
    fn build_kwargs_basic_and_strip_tools() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "f", "description": "", "parameters": {}}
        })];
        let params = ConverseParams {
            model: "anthropic.claude-3-haiku",
            messages: &messages,
            tools: Some(&tools),
            max_tokens: 1000,
            temperature: Some(0.5),
            top_p: None,
            stop_sequences: None,
            guardrail_config: None,
        };
        let body = build_converse_kwargs(&params);
        assert_eq!(body["modelId"], "anthropic.claude-3-haiku");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 1000);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.5);
        assert!(body.get("toolConfig").is_some());

        // Non-tool model strips tools.
        let params2 = ConverseParams {
            model: "us.deepseek.r1-v1:0",
            ..params.clone()
        };
        let body2 = build_converse_kwargs(&params2);
        assert!(body2.get("toolConfig").is_none());
    }

    #[test]
    fn discovery_filtering_and_sort() {
        let fm = json!({
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-haiku-v1:0",
                    "modelName": "Claude 3 Haiku",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": true,
                    "modelLifecycle": {"status": "ACTIVE"}
                },
                {
                    "modelId": "amazon.titan-embed-v1",
                    "modelName": "Titan Embed",
                    "providerName": "Amazon",
                    "outputModalities": ["EMBEDDING"],
                    "responseStreamingSupported": true,
                    "modelLifecycle": {"status": "ACTIVE"}
                },
                {
                    "modelId": "legacy.model",
                    "providerName": "X",
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": true,
                    "modelLifecycle": {"status": "LEGACY"}
                }
            ]
        });
        let profiles = vec![json!({
            "inferenceProfileId": "global.anthropic.claude-3-haiku",
            "inferenceProfileName": "Global Haiku",
            "status": "ACTIVE",
            "models": [{"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku"}]
        })];
        let out = parse_discovered_models(&fm, &profiles, None);
        // Titan (no TEXT output) and legacy (not ACTIVE) excluded.
        assert_eq!(out.len(), 2);
        // global.* sorts first.
        assert_eq!(out[0].id, "global.anthropic.claude-3-haiku");
        assert_eq!(out[0].provider, "inference-profile");
        assert_eq!(out[1].id, "anthropic.claude-3-haiku-v1:0");
    }

    #[test]
    fn discovery_provider_filter() {
        let fm = json!({
            "modelSummaries": [
                {
                    "modelId": "meta.llama3-3-70b",
                    "providerName": "Meta",
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": true,
                    "modelLifecycle": {"status": "ACTIVE"}
                }
            ]
        });
        let filter = vec!["anthropic".to_string()];
        let out = parse_discovered_models(&fm, &[], Some(&filter));
        assert!(out.is_empty());
    }

    #[test]
    fn arn_provider_extraction() {
        assert_eq!(
            extract_provider_from_arn(
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2"
            ),
            "anthropic"
        );
        assert_eq!(extract_provider_from_arn("not-an-arn"), "");
    }

    #[test]
    fn discovery_cache_roundtrip() {
        reset_discovery_cache();
        assert!(cached_discovered_models("us-east-1", None).is_none());
        let models = vec![BedrockModelInfo {
            id: "m".to_string(),
            name: "M".to_string(),
            provider: "P".to_string(),
            input_modalities: vec![],
            output_modalities: vec![],
            streaming: true,
        }];
        cache_discovered_models("us-east-1", None, models.clone());
        assert_eq!(cached_discovered_models("us-east-1", None), Some(models));
        assert_eq!(model_ids(&cached_discovered_models("us-east-1", None).unwrap()), vec!["m".to_string()]);
    }

    #[test]
    fn error_classification() {
        assert!(is_context_overflow_error(
            "ValidationException: input is too long for this model"
        ));
        assert_eq!(
            classify_bedrock_error("ValidationException: input is too long"),
            "context_overflow"
        );
        assert_eq!(classify_bedrock_error("ThrottlingException: slow down"), "rate_limit");
        assert_eq!(
            classify_bedrock_error("ServiceQuotaExceededException"),
            "rate_limit"
        );
        assert_eq!(
            classify_bedrock_error("ModelNotReadyException: warming up"),
            "overloaded"
        );
        assert_eq!(classify_bedrock_error("AccessDeniedException"), "unknown");
    }

    #[test]
    fn context_length_lookup() {
        assert_eq!(
            get_bedrock_context_length("anthropic.claude-sonnet-4-6-20250514-v1:0"),
            200_000
        );
        assert_eq!(get_bedrock_context_length("amazon.nova-pro-v1:0"), 300_000);
        assert_eq!(get_bedrock_context_length("amazon.nova-micro-v1:0"), 128_000);
        // longest-match wins: "amazon.nova-pro" over no shorter overlap
        assert_eq!(get_bedrock_context_length("totally-unknown-model"), BEDROCK_DEFAULT_CONTEXT_LENGTH);
    }
}
