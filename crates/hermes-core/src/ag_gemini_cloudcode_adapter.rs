//! OpenAI-compatible facade that talks to Google's Cloud Code Assist backend.
//!
//! This adapter lets Hermes use the `google-gemini-cli` provider as if it were
//! a standard OpenAI-shaped chat completion endpoint, while the underlying HTTP
//! traffic goes to `cloudcode-pa.googleapis.com/v1internal:{generateContent,
//! streamGenerateContent}` with a Bearer access token obtained via OAuth PKCE.
//!
//! Architecture
//! ------------
//! - [`GeminiCloudCodeClient`] exposes [`GeminiCloudCodeClient::create_chat_completion`]
//!   mirroring the subset of the OpenAI SDK that the agent loop uses.
//! - Incoming OpenAI `messages[]` / `tools[]` / `tool_choice` are translated
//!   to Gemini's native `contents[]` / `tools[].functionDeclarations` /
//!   `toolConfig` / `systemInstruction` shape.
//! - The request body is wrapped `{project, model, user_prompt_id, request}`
//!   per Code Assist API expectations.
//! - Responses (`candidates[].content.parts[]`) are converted back to
//!   OpenAI `choices[0].message` shape with `content` + `tool_calls`.
//! - Streaming uses SSE (`?alt=sse`) and yields OpenAI-shaped delta chunks.
//!
//! Attribution
//! -----------
//! Translation semantics follow jenslys/opencode-gemini-auth (MIT) and the public
//! Gemini API docs. Request envelope shape
//! (`{project, model, user_prompt_id, request}`) is documented nowhere; it is
//! reverse-engineered from the opencode-gemini-auth and clawdbot implementations.

use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gemini_schema::sanitize_gemini_tool_parameters;
use crate::google_code_assist::{CodeAssistError, ProjectContext, CODE_ASSIST_ENDPOINT};

// =============================================================================
// id / time helpers (no uuid crate available)
// =============================================================================

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Unix seconds, mirroring `int(time.time())`.
pub fn unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 12-hex-char unique suffix, mirroring `uuid.uuid4().hex[:12]`.
fn hex12() -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = unix_nanos();
    // Mix counter + nanos to produce a unique-enough hex string.
    let mixed = (nanos as u64) ^ (n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    format!("{:012x}", mixed & 0xFFFF_FFFF_FFFF)
}

/// A `uuid.uuid4()`-style id (used for `user_prompt_id`/`x-activity-request-id`).
fn uuid_like() -> String {
    let nanos = unix_nanos();
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let a = (nanos as u64) ^ n.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let b = (nanos >> 64) as u64 ^ n.rotate_left(17);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (a >> 32) as u32,
        ((a >> 16) & 0xFFFF) as u16,
        (a & 0xFFFF) as u16,
        ((b >> 48) & 0xFFFF) as u16,
        b & 0xFFFF_FFFF_FFFF,
    )
}

// =============================================================================
// Request translation: OpenAI -> Gemini
// =============================================================================

fn role_map_openai_to_gemini(role: &str) -> &'static str {
    match role {
        "user" => "user",
        "assistant" => "model",
        "system" => "user", // handled separately via systemInstruction
        "tool" => "user",   // functionResponse is wrapped in a user-role turn
        "function" => "user",
        _ => "user",
    }
}

/// OpenAI content may be str or a list of parts; reduce to plain text.
pub fn coerce_content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut pieces: Vec<String> = Vec::new();
            for p in arr {
                match p {
                    Value::String(s) => pieces.push(s.clone()),
                    Value::Object(obj) => {
                        let typ = obj.get("type").and_then(Value::as_str);
                        if typ == Some("text") {
                            if let Some(t) = obj.get("text").and_then(Value::as_str) {
                                pieces.push(t.to_string());
                            }
                        } else if typ == Some("image_url") || typ == Some("input_audio") {
                            log::debug!(
                                "Dropping multimodal part (not yet supported): {}",
                                typ.unwrap_or("")
                            );
                        }
                    }
                    _ => {}
                }
            }
            pieces.join("\n")
        }
        // Mirror Python's `str(content)` fallback for non-str/list/None.
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Object(_) => content.to_string(),
    }
}

/// OpenAI tool_call -> Gemini functionCall part.
pub fn translate_tool_call_to_gemini(tool_call: &Value) -> Value {
    let fn_obj = tool_call.get("function").cloned().unwrap_or(Value::Null);
    let args_raw = fn_obj.get("arguments");

    let args: Value = match args_raw {
        Some(Value::String(s)) if !s.is_empty() => match serde_json::from_str::<Value>(s) {
            Ok(v) => v,
            Err(_) => json!({ "_raw": s }),
        },
        // Empty string or absent -> {}
        Some(Value::String(_)) | None => json!({}),
        // Non-string arguments value present (unusual, but Python would JSON-load
        // only strings; anything else flows through the `not dict` guard).
        Some(other) => other.clone(),
    };

    // If parsed/given args are not a dict, wrap as {"_value": args}.
    let args = if args.is_object() {
        args
    } else {
        json!({ "_value": args })
    };

    let name = fn_obj
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    json!({
        "functionCall": {
            "name": name,
            "args": args,
        },
        // Sentinel signature — matches opencode-gemini-auth's approach.
        // Without this, Code Assist rejects function calls that originated
        // outside its own chain.
        "thoughtSignature": "skip_thought_signature_validator",
    })
}

/// OpenAI tool-role message -> Gemini functionResponse part.
pub fn translate_tool_result_to_gemini(message: &Value) -> Value {
    let name = message
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| message.get("tool_call_id").and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .unwrap_or("tool")
        .to_string();

    let content = coerce_content_to_text(message.get("content").unwrap_or(&Value::Null));

    // Gemini expects the response as a dict under `response`. We wrap plain
    // text in {"output": "..."}.
    let trimmed = content.trim_start();
    let parsed: Option<Value> = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str::<Value>(&content).ok()
    } else {
        None
    };
    let response = match parsed {
        Some(v) if v.is_object() => v,
        _ => json!({ "output": content }),
    };

    json!({
        "functionResponse": {
            "name": name,
            "response": response,
        },
    })
}

/// Convert OpenAI messages[] to Gemini contents[] + systemInstruction.
pub fn build_gemini_contents(messages: &[Value]) -> (Vec<Value>, Option<Value>) {
    let mut system_text_parts: Vec<String> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();

    for msg in messages {
        if !msg.is_object() {
            continue;
        }
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");

        if role == "system" {
            system_text_parts
                .push(coerce_content_to_text(msg.get("content").unwrap_or(&Value::Null)));
            continue;
        }

        // Tool result message — emit a user-role turn with functionResponse.
        if role == "tool" || role == "function" {
            contents.push(json!({
                "role": "user",
                "parts": [translate_tool_result_to_gemini(msg)],
            }));
            continue;
        }

        let gemini_role = role_map_openai_to_gemini(role);
        let mut parts: Vec<Value> = Vec::new();

        let text = coerce_content_to_text(msg.get("content").unwrap_or(&Value::Null));
        if !text.is_empty() {
            parts.push(json!({ "text": text }));
        }

        // Assistant messages can carry tool_calls.
        if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
            for tc in tool_calls {
                if tc.is_object() {
                    parts.push(translate_tool_call_to_gemini(tc));
                }
            }
        }

        if parts.is_empty() {
            // Gemini rejects empty parts; skip the turn entirely.
            continue;
        }

        contents.push(json!({ "role": gemini_role, "parts": parts }));
    }

    let joined_system = system_text_parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    let joined_system = joined_system.trim().to_string();

    let system_instruction = if !joined_system.is_empty() {
        Some(json!({
            "role": "system",
            "parts": [{ "text": joined_system }],
        }))
    } else {
        None
    };

    (contents, system_instruction)
}

/// OpenAI tools[] -> Gemini tools[].functionDeclarations[].
pub fn translate_tools_to_gemini(tools: Option<&Value>) -> Vec<Value> {
    let arr = match tools {
        Some(Value::Array(a)) if !a.is_empty() => a,
        _ => return Vec::new(),
    };

    let mut declarations: Vec<Value> = Vec::new();
    for t in arr {
        if !t.is_object() {
            continue;
        }
        let fn_obj = match t.get("function") {
            Some(v) if v.is_object() => v,
            _ => continue,
        };
        let name = match fn_obj.get("name").and_then(Value::as_str) {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let mut decl = Map::new();
        decl.insert("name".to_string(), Value::String(name.to_string()));
        if let Some(desc) = fn_obj.get("description") {
            // Python: `if fn.get("description"):` — truthy check, then str().
            if is_truthy(desc) {
                decl.insert("description".to_string(), Value::String(value_to_str(desc)));
            }
        }
        if let Some(params) = fn_obj.get("parameters") {
            if params.is_object() {
                decl.insert(
                    "parameters".to_string(),
                    sanitize_gemini_tool_parameters(params),
                );
            }
        }
        declarations.push(Value::Object(decl));
    }

    if declarations.is_empty() {
        return Vec::new();
    }
    vec![json!({ "functionDeclarations": declarations })]
}

/// OpenAI tool_choice -> Gemini toolConfig.functionCallingConfig.
pub fn translate_tool_choice_to_gemini(tool_choice: Option<&Value>) -> Option<Value> {
    let tc = match tool_choice {
        None | Some(Value::Null) => return None,
        Some(v) => v,
    };
    if let Some(s) = tc.as_str() {
        return match s {
            "auto" => Some(json!({ "functionCallingConfig": { "mode": "AUTO" } })),
            "required" => Some(json!({ "functionCallingConfig": { "mode": "ANY" } })),
            "none" => Some(json!({ "functionCallingConfig": { "mode": "NONE" } })),
            _ => None,
        };
    }
    if tc.is_object() {
        let fn_obj = tc.get("function").cloned().unwrap_or(Value::Null);
        if let Some(name) = fn_obj.get("name").and_then(Value::as_str) {
            if !name.is_empty() {
                return Some(json!({
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    },
                }));
            }
        }
    }
    None
}

/// Accept thinkingBudget / thinkingLevel / includeThoughts (+ snake_case).
pub fn normalize_thinking_config(config: Option<&Value>) -> Option<Value> {
    let obj = match config {
        Some(Value::Object(o)) if !o.is_empty() => o,
        _ => return None,
    };

    let budget = obj.get("thinkingBudget").or_else(|| obj.get("thinking_budget"));
    let level = obj.get("thinkingLevel").or_else(|| obj.get("thinking_level"));
    let include = obj
        .get("includeThoughts")
        .or_else(|| obj.get("include_thoughts"));

    let mut normalized = Map::new();

    // Python: isinstance(budget, (int, float)) — note bool IS int in Python,
    // but JSON bools are distinct here; we accept only numbers.
    if let Some(b) = budget.and_then(Value::as_f64) {
        // Skip JSON bools that as_f64 would not catch anyway.
        normalized.insert("thinkingBudget".to_string(), json!(b as i64));
    }
    if let Some(l) = level.and_then(Value::as_str) {
        if !l.trim().is_empty() {
            normalized.insert(
                "thinkingLevel".to_string(),
                Value::String(l.trim().to_lowercase()),
            );
        }
    }
    if let Some(i) = include.and_then(Value::as_bool) {
        normalized.insert("includeThoughts".to_string(), Value::Bool(i));
    }

    if normalized.is_empty() {
        None
    } else {
        Some(Value::Object(normalized))
    }
}

/// Optional generation parameters carried into [`build_gemini_request`].
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    /// String or list of strings (raw JSON value).
    pub stop: Option<Value>,
    pub thinking_config: Option<Value>,
}

/// Build the inner Gemini request body (goes inside `request` wrapper).
pub fn build_gemini_request(
    messages: &[Value],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    params: &GenerationParams,
) -> Value {
    let (contents, system_instruction) = build_gemini_contents(messages);

    let mut body = Map::new();
    body.insert("contents".to_string(), Value::Array(contents));
    if let Some(si) = system_instruction {
        body.insert("systemInstruction".to_string(), si);
    }

    let gemini_tools = translate_tools_to_gemini(tools);
    if !gemini_tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(gemini_tools));
    }
    if let Some(tool_cfg) = translate_tool_choice_to_gemini(tool_choice) {
        body.insert("toolConfig".to_string(), tool_cfg);
    }

    let mut generation_config = Map::new();
    if let Some(t) = params.temperature {
        generation_config.insert("temperature".to_string(), json!(t));
    }
    if let Some(mt) = params.max_tokens {
        if mt > 0 {
            generation_config.insert("maxOutputTokens".to_string(), json!(mt));
        }
    }
    if let Some(tp) = params.top_p {
        generation_config.insert("topP".to_string(), json!(tp));
    }
    match &params.stop {
        Some(Value::String(s)) if !s.is_empty() => {
            generation_config.insert("stopSequences".to_string(), json!([s]));
        }
        Some(Value::Array(arr)) if !arr.is_empty() => {
            // [str(s) for s in stop if s]
            let seqs: Vec<Value> = arr
                .iter()
                .filter(|s| is_truthy(s))
                .map(|s| Value::String(value_to_str(s)))
                .collect();
            generation_config.insert("stopSequences".to_string(), Value::Array(seqs));
        }
        _ => {}
    }
    if let Some(normalized) = normalize_thinking_config(params.thinking_config.as_ref()) {
        generation_config.insert("thinkingConfig".to_string(), normalized);
    }
    if !generation_config.is_empty() {
        body.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }

    Value::Object(body)
}

/// Wrap the inner Gemini request in the Code Assist envelope.
pub fn wrap_code_assist_request(
    project_id: &str,
    model: &str,
    inner_request: Value,
    user_prompt_id: Option<&str>,
) -> Value {
    json!({
        "project": project_id,
        "model": model,
        "user_prompt_id": user_prompt_id.map(|s| s.to_string()).unwrap_or_else(uuid_like),
        "request": inner_request,
    })
}

// =============================================================================
// OpenAI-shaped response types
// =============================================================================

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub index: usize,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptTokensDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

// =============================================================================
// Response translation: Gemini -> OpenAI
// =============================================================================

/// Map a Gemini finishReason to an OpenAI finish_reason.
pub fn map_gemini_finish_reason(reason: &str) -> &'static str {
    match reason.to_uppercase().as_str() {
        "STOP" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        "RECITATION" => "content_filter",
        "OTHER" => "stop",
        _ => "stop",
    }
}

fn empty_response(model: &str) -> ChatCompletion {
    ChatCompletion {
        id: format!("chatcmpl-{}", hex12()),
        object: "chat.completion".to_string(),
        created: unix_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some(String::new()),
                tool_calls: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage::default(),
    }
}

/// Non-streaming Gemini response -> OpenAI-shaped [`ChatCompletion`].
///
/// Code Assist wraps the actual Gemini response inside `response`, so we
/// unwrap it first if present.
pub fn translate_gemini_response(resp: &Value, model: &str) -> ChatCompletion {
    let inner = match resp.get("response") {
        Some(v) if v.is_object() => v,
        _ => resp,
    };

    let candidates = match inner.get("candidates") {
        Some(Value::Array(c)) if !c.is_empty() => c,
        _ => return empty_response(model),
    };

    let cand = &candidates[0];
    let content_obj = cand.get("content");
    let parts = content_obj
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    let mut text_pieces: Vec<String> = Vec::new();
    let mut reasoning_pieces: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    if let Some(parts) = parts {
        for (i, part) in parts.iter().enumerate() {
            if !part.is_object() {
                continue;
            }
            // Thought parts are model's internal reasoning — surface as reasoning,
            // don't mix into content.
            if part.get("thought") == Some(&Value::Bool(true)) {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    reasoning_pieces.push(t.to_string());
                }
                continue;
            }
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                text_pieces.push(t.to_string());
                continue;
            }
            if let Some(fc) = part.get("functionCall") {
                if fc.is_object() {
                    if let Some(name) = fc.get("name").and_then(Value::as_str) {
                        if !name.is_empty() {
                            let args = fc.get("args").cloned().unwrap_or(json!({}));
                            let args_str =
                                serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(ToolCall {
                                id: format!("call_{}", hex12()),
                                r#type: "function".to_string(),
                                index: i,
                                function: ToolCallFunction {
                                    name: name.to_string(),
                                    arguments: args_str,
                                },
                            });
                        }
                    }
                }
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls".to_string()
    } else {
        map_gemini_finish_reason(cand.get("finishReason").and_then(Value::as_str).unwrap_or(""))
            .to_string()
    };

    let usage_meta = inner.get("usageMetadata");
    let usage = Usage {
        prompt_tokens: meta_int(usage_meta, "promptTokenCount"),
        completion_tokens: meta_int(usage_meta, "candidatesTokenCount"),
        total_tokens: meta_int(usage_meta, "totalTokenCount"),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: meta_int(usage_meta, "cachedContentTokenCount"),
        },
    };

    let reasoning_joined = reasoning_pieces.concat();
    let message = ChatMessage {
        role: "assistant".to_string(),
        content: if text_pieces.is_empty() {
            None
        } else {
            Some(text_pieces.concat())
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        reasoning: if reasoning_joined.is_empty() {
            None
        } else {
            Some(reasoning_joined.clone())
        },
        reasoning_content: if reasoning_joined.is_empty() {
            None
        } else {
            Some(reasoning_joined)
        },
        reasoning_details: None,
    };

    ChatCompletion {
        id: format!("chatcmpl-{}", hex12()),
        object: "chat.completion".to_string(),
        created: unix_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
    }
}

fn meta_int(meta: Option<&Value>, key: &str) -> i64 {
    meta.and_then(|m| m.get(key))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

// =============================================================================
// Streaming SSE
// =============================================================================

/// An OpenAI ChatCompletionChunk-shaped delta.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

/// Internal description of a tool-call delta for [`make_stream_chunk`].
#[derive(Debug, Clone, Default)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

pub fn make_stream_chunk(
    model: &str,
    content: &str,
    tool_call_delta: Option<&ToolCallDelta>,
    finish_reason: Option<&str>,
    reasoning: &str,
) -> ChatCompletionChunk {
    let mut delta = StreamDelta {
        role: Some("assistant".to_string()),
        ..Default::default()
    };
    if !content.is_empty() {
        delta.content = Some(content.to_string());
    }
    if let Some(tcd) = tool_call_delta {
        let id = tcd
            .id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("call_{}", hex12()));
        delta.tool_calls = Some(vec![ToolCall {
            index: tcd.index,
            id,
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: tcd.name.clone(),
                arguments: tcd.arguments.clone(),
            },
        }]);
    }
    if !reasoning.is_empty() {
        delta.reasoning = Some(reasoning.to_string());
        delta.reasoning_content = Some(reasoning.to_string());
    }

    ChatCompletionChunk {
        id: format!("chatcmpl-{}", hex12()),
        object: "chat.completion.chunk".to_string(),
        created: unix_secs(),
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta,
            finish_reason: finish_reason.map(|s| s.to_string()),
        }],
        usage: None,
    }
}

/// Parse Server-Sent Events from a raw text body, returning decoded JSON events.
///
/// Mirrors Python's `_iter_sse_events`: splits on `\n`, strips trailing `\r`,
/// handles `data: ` lines, stops at `[DONE]`, skips non-JSON lines.
pub fn iter_sse_events(body: &str) -> Vec<Value> {
    let mut events: Vec<Value> = Vec::new();
    for raw_line in body.split('\n') {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                return events;
            }
            match serde_json::from_str::<Value>(data) {
                Ok(v) => events.push(v),
                Err(_) => {
                    let snippet: String = data.chars().take(200).collect();
                    log::debug!("Non-JSON SSE line: {snippet}");
                }
            }
        }
    }
    events
}

/// Unwrap Code Assist envelope and emit OpenAI-shaped chunk(s).
///
/// `tool_call_counter` is a mutable counter across events in the same stream.
/// Each `functionCall` part gets a fresh, unique OpenAI `index`.
pub fn translate_stream_event(
    event: &Value,
    model: &str,
    tool_call_counter: &mut usize,
) -> Vec<ChatCompletionChunk> {
    let inner = match event.get("response") {
        Some(v) if v.is_object() => v,
        _ => event,
    };
    let candidates = match inner.get("candidates") {
        Some(Value::Array(c)) if !c.is_empty() => c,
        _ => return Vec::new(),
    };
    let cand = &candidates[0];
    if !cand.is_object() {
        return Vec::new();
    }

    let mut chunks: Vec<ChatCompletionChunk> = Vec::new();

    let parts = cand
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    if let Some(parts) = parts {
        for part in parts {
            if !part.is_object() {
                continue;
            }
            if part.get("thought") == Some(&Value::Bool(true)) {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    chunks.push(make_stream_chunk(model, "", None, None, t));
                    continue;
                }
            }
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                if !t.is_empty() {
                    chunks.push(make_stream_chunk(model, t, None, None, ""));
                }
            }
            if let Some(fc) = part.get("functionCall") {
                if fc.is_object() {
                    if let Some(name) = fc.get("name").and_then(Value::as_str) {
                        if !name.is_empty() {
                            let idx = *tool_call_counter;
                            *tool_call_counter += 1;
                            let args = fc.get("args").cloned().unwrap_or(json!({}));
                            let args_str =
                                serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                            let tcd = ToolCallDelta {
                                index: idx,
                                id: None,
                                name: name.to_string(),
                                arguments: args_str,
                            };
                            chunks.push(make_stream_chunk(model, "", Some(&tcd), None, ""));
                        }
                    }
                }
            }
        }
    }

    let finish_reason_raw = cand.get("finishReason").and_then(Value::as_str).unwrap_or("");
    if !finish_reason_raw.is_empty() {
        let mapped = if *tool_call_counter > 0 {
            "tool_calls".to_string()
        } else {
            map_gemini_finish_reason(finish_reason_raw).to_string()
        };
        chunks.push(make_stream_chunk(model, "", None, Some(&mapped), ""));
    }
    chunks
}

// =============================================================================
// HTTP error translation
// =============================================================================

/// Translate a non-200 HTTP response (status + headers + body) into a
/// [`CodeAssistError`] with rich metadata.
///
/// Parses Google's error envelope (`{"error": {"code", "message", "status",
/// "details": [...]}}`) so the agent's error classifier can reason about the
/// failure. `retry_after_header` is the value of the `Retry-After` response
/// header (case-insensitive lookup done by the caller).
pub fn gemini_http_error(
    status: u16,
    body_text: &str,
    retry_after_header: Option<&str>,
) -> CodeAssistError {
    let body_json: Value = if body_text.is_empty() {
        Value::Null
    } else {
        serde_json::from_str::<Value>(body_text).unwrap_or(Value::Null)
    };
    let body_json = if body_json.is_object() {
        body_json
    } else {
        Value::Object(Map::new())
    };

    let err_obj = match body_json.get("error") {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Map::new()),
    };
    let err_status = err_obj
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let err_message = err_obj
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let empty_details: Vec<Value> = Vec::new();
    let err_details_list = err_obj
        .get("details")
        .and_then(Value::as_array)
        .unwrap_or(&empty_details);

    // Extract google.rpc.ErrorInfo reason + metadata; pick the first with a reason.
    let mut error_reason = String::new();
    let mut error_metadata: Value = Value::Object(Map::new());
    let mut retry_delay_seconds: Option<f64> = None;

    for detail in err_details_list {
        if !detail.is_object() {
            continue;
        }
        let type_url = detail.get("@type").and_then(Value::as_str).unwrap_or("");
        if error_reason.is_empty() && type_url.ends_with("/google.rpc.ErrorInfo") {
            if let Some(reason) = detail.get("reason").and_then(Value::as_str) {
                if !reason.is_empty() {
                    error_reason = reason.to_string();
                }
            }
            if let Some(md) = detail.get("metadata") {
                if md.is_object() {
                    error_metadata = md.clone();
                }
            }
        } else if retry_delay_seconds.is_none() && type_url.ends_with("/google.rpc.RetryInfo") {
            // retryDelay is a google.protobuf.Duration string like "30s" or "1.5s".
            match detail.get("retryDelay") {
                Some(Value::String(s)) if s.ends_with('s') => {
                    if let Ok(v) = s[..s.len() - 1].parse::<f64>() {
                        retry_delay_seconds = Some(v);
                    }
                }
                Some(Value::Number(n)) => {
                    if let Some(v) = n.as_f64() {
                        retry_delay_seconds = Some(v);
                    }
                }
                _ => {}
            }
        }
    }

    // Fall back to the Retry-After header if the body lacked RetryInfo.
    if retry_delay_seconds.is_none() {
        if let Some(h) = retry_after_header {
            if !h.is_empty() {
                if let Ok(v) = h.parse::<f64>() {
                    retry_delay_seconds = Some(v);
                }
            }
        }
    }

    // Classify the error code.
    let mut code = format!("code_assist_http_{status}");
    if status == 401 {
        code = "code_assist_unauthorized".to_string();
    } else if status == 429 {
        code = "code_assist_rate_limited".to_string();
        if error_reason == "MODEL_CAPACITY_EXHAUSTED" {
            code = "code_assist_capacity_exhausted".to_string();
        }
    }

    // Build a human-readable message.
    let model_hint = if error_metadata.is_object() {
        error_metadata
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| error_metadata.get("modelId").and_then(Value::as_str))
            .unwrap_or("")
            .trim()
            .to_string()
    } else {
        String::new()
    };

    let message: String;
    if status == 429 && error_reason == "MODEL_CAPACITY_EXHAUSTED" {
        let target = if model_hint.is_empty() {
            "this Gemini model".to_string()
        } else {
            model_hint.clone()
        };
        let mut m = format!(
            "Gemini capacity exhausted for {target} (Google-side throttle, not a Hermes issue). \
             Try a different Gemini model or set a fallback_providers entry to a non-Gemini provider."
        );
        if let Some(d) = retry_delay_seconds {
            m.push_str(&format!(" Google suggests retrying in {}s.", fmt_g(d)));
        }
        message = m;
    } else if status == 429 && err_status == "RESOURCE_EXHAUSTED" {
        let reason_txt = if err_message.is_empty() {
            "RESOURCE_EXHAUSTED".to_string()
        } else {
            err_message.clone()
        };
        let mut m = format!(
            "Gemini quota exhausted ({reason_txt}). Check /gquota for remaining daily requests."
        );
        if let Some(d) = retry_delay_seconds {
            m.push_str(&format!(" Retry suggested in {}s.", fmt_g(d)));
        }
        message = m;
    } else if status == 404 {
        let target = if !model_hint.is_empty() {
            model_hint.clone()
        } else if !err_message.is_empty() {
            err_message.clone()
        } else {
            "model".to_string()
        };
        message = format!(
            "Code Assist 404: {target} is not available at cloudcode-pa.googleapis.com. \
             It may have been renamed or retired. Check hermes_cli/models.py for the current list."
        );
    } else if !err_message.is_empty() {
        let st = if err_status.is_empty() {
            "error".to_string()
        } else {
            err_status.clone()
        };
        message = format!("Code Assist HTTP {status} ({st}): {err_message}");
    } else {
        let snippet: String = body_text.chars().take(500).collect();
        message = format!("Code Assist returned HTTP {status}: {snippet}");
    }

    let mut details = std::collections::BTreeMap::new();
    details.insert("status".to_string(), Value::String(err_status));
    details.insert("reason".to_string(), Value::String(error_reason));
    details.insert("metadata".to_string(), error_metadata);
    details.insert("message".to_string(), Value::String(err_message));

    CodeAssistError {
        message,
        code,
        status_code: Some(status),
        response: Some(body_text.to_string()),
        retry_after: retry_delay_seconds,
        details,
    }
}

/// Format a float like Python's `%g` (used in retry-delay messages).
fn fmt_g(v: f64) -> String {
    // Python "%g": strips trailing zeros, uses up to 6 sig figs.
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.6}", v);
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    }
}

// =============================================================================
// Small value helpers
// =============================================================================

/// Python truthiness for a JSON value (used for `if fn.get("description"):` etc).
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Equivalent of Python `str(v)` for scalar JSON values.
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

// =============================================================================
// GeminiCloudCodeClient — OpenAI-compatible facade
// =============================================================================

pub const MARKER_BASE_URL: &str = "cloudcode-pa://google";

/// Result of a chat completion call: either a full response or a stream.
pub enum CompletionResult {
    Single(ChatCompletion),
    Stream(Vec<ChatCompletionChunk>),
}

/// Parameters for [`GeminiCloudCodeClient::create_chat_completion`].
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub stream: bool,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub stop: Option<Value>,
    /// OpenAI `extra_body`; we look up `thinking_config`/`thinkingConfig` here.
    pub extra_body: Option<Value>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        ChatCompletionRequest {
            model: "gemini-2.5-flash".to_string(),
            messages: Vec::new(),
            stream: false,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            extra_body: None,
        }
    }
}

/// Minimal OpenAI-SDK-compatible facade over Code Assist v1internal.
pub struct GeminiCloudCodeClient {
    pub api_key: String,
    pub base_url: String,
    default_headers: Vec<(String, String)>,
    configured_project_id: String,
    project_context: Option<ProjectContext>,
    pub is_closed: bool,
    http: reqwest::blocking::Client,
}

impl GeminiCloudCodeClient {
    /// Construct a client. `api_key` is a dummy for OpenAI interface parity —
    /// real auth is the OAuth access token fetched on every call.
    pub fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        default_headers: Option<Vec<(String, String)>>,
        project_id: String,
    ) -> Self {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());

        GeminiCloudCodeClient {
            api_key: api_key.unwrap_or_else(|| "google-oauth".to_string()),
            base_url: base_url.unwrap_or_else(|| MARKER_BASE_URL.to_string()),
            default_headers: default_headers.unwrap_or_default(),
            configured_project_id: project_id,
            project_context: None,
            is_closed: false,
            http,
        }
    }

    pub fn close(&mut self) {
        self.is_closed = true;
    }

    /// Lazily resolve and cache the project context for this client.
    ///
    /// Mirrors the Python `_ensure_project_context`: prefer the project already
    /// baked into stored credentials; otherwise resolve via the networked
    /// discovery path and persist any discovered ids.
    pub fn ensure_project_context(
        &mut self,
        access_token: &str,
        model: &str,
    ) -> Result<ProjectContext, CodeAssistError> {
        if let Some(ctx) = &self.project_context {
            return Ok(ctx.clone());
        }

        let env_project = crate::ag_google_oauth::resolve_project_id_from_env();
        let creds = crate::ag_google_oauth::load_credentials();
        let stored_project = creds
            .as_ref()
            .map(|c| c.project_id.clone())
            .unwrap_or_default();

        // Prefer what's already baked into the creds.
        if !stored_project.is_empty() {
            let ctx = ProjectContext {
                project_id: stored_project,
                managed_project_id: creds
                    .as_ref()
                    .map(|c| c.managed_project_id.clone())
                    .unwrap_or_default(),
                tier_id: String::new(),
                source: "stored".to_string(),
            };
            self.project_context = Some(ctx.clone());
            return Ok(ctx);
        }

        // The networked discovery `resolve_project_context` is not ported to
        // native Rust yet (it lives in the Python `google_code_assist` module).
        // Fall back to env/configured ids so calls still succeed; record the
        // discovered ids back to the creds file when present. Without a stored
        // or configured/env project, surface the project-id-required error
        // rather than silently sending an empty project.
        let ctx = self.resolve_project_context_fallback(access_token, &env_project, model)?;
        if !ctx.project_id.is_empty() || !ctx.managed_project_id.is_empty() {
            let _ = crate::ag_google_oauth::update_project_ids(
                &ctx.project_id,
                &ctx.managed_project_id,
            );
        }
        self.project_context = Some(ctx.clone());
        Ok(ctx)
    }

    /// Best-effort project-context resolution from configured/env ids.
    fn resolve_project_context_fallback(
        &self,
        _access_token: &str,
        env_project: &str,
        _model: &str,
    ) -> Result<ProjectContext, CodeAssistError> {
        if !self.configured_project_id.is_empty() {
            return Ok(ProjectContext {
                project_id: self.configured_project_id.clone(),
                managed_project_id: String::new(),
                tier_id: crate::google_code_assist::STANDARD_TIER_ID.to_string(),
                source: "config".to_string(),
            });
        }
        if !env_project.is_empty() {
            return Ok(ProjectContext {
                project_id: env_project.to_string(),
                managed_project_id: String::new(),
                tier_id: crate::google_code_assist::STANDARD_TIER_ID.to_string(),
                source: "env".to_string(),
            });
        }
        Err(CodeAssistError::project_id_required(Some(
            "Could not resolve a Google Cloud project for Code Assist. Set a project id \
             via configuration or the GOOGLE_CLOUD_PROJECT environment variable."
                .to_string(),
        )))
    }

    /// Build the wrapped request body + headers for a chat completion.
    ///
    /// Split out from the network call so it can be unit-tested. Returns
    /// `(wrapped_body, headers, access_token)`.
    pub fn prepare_request(
        &mut self,
        req: &ChatCompletionRequest,
    ) -> Result<(Value, Vec<(String, String)>), CodeAssistError> {
        let access_token = crate::ag_google_oauth::get_valid_access_token(false)
            .map_err(|e| CodeAssistError::with_code(e.to_string(), "code_assist_auth_error"))?;
        let ctx = self.ensure_project_context(&access_token, &req.model)?;

        let thinking_config = req.extra_body.as_ref().and_then(|eb| {
            eb.get("thinking_config")
                .or_else(|| eb.get("thinkingConfig"))
                .cloned()
        });

        let params = GenerationParams {
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            top_p: req.top_p,
            stop: req.stop.clone(),
            thinking_config,
        };

        let inner = build_gemini_request(
            &req.messages,
            req.tools.as_ref(),
            req.tool_choice.as_ref(),
            &params,
        );
        let wrapped = wrap_code_assist_request(&ctx.project_id, &req.model, inner, None);

        let mut headers: Vec<(String, String)> = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Accept".to_string(), "application/json".to_string()),
            ("Authorization".to_string(), format!("Bearer {access_token}")),
            (
                "User-Agent".to_string(),
                "hermes-agent (gemini-cli-compat)".to_string(),
            ),
            ("X-Goog-Api-Client".to_string(), "gl-python/hermes".to_string()),
            ("x-activity-request-id".to_string(), uuid_like()),
        ];
        // self._default_headers override / append (dict.update semantics).
        for (k, v) in &self.default_headers {
            if let Some(slot) = headers.iter_mut().find(|(hk, _)| hk == k) {
                slot.1 = v.clone();
            } else {
                headers.push((k.clone(), v.clone()));
            }
        }

        Ok((wrapped, headers))
    }

    /// OpenAI-shaped `chat.completions.create`.
    pub fn create_chat_completion(
        &mut self,
        req: &ChatCompletionRequest,
    ) -> Result<CompletionResult, CodeAssistError> {
        let (wrapped, headers) = self.prepare_request(req)?;

        if req.stream {
            let chunks = self.stream_completion(&req.model, &wrapped, &headers)?;
            return Ok(CompletionResult::Stream(chunks));
        }

        let url = format!("{CODE_ASSIST_ENDPOINT}/v1internal:generateContent");
        let mut builder = self.http.post(&url).json(&wrapped);
        for (k, v) in &headers {
            builder = builder.header(k, v);
        }
        let response = builder
            .send()
            .map_err(|e| CodeAssistError::network(e.to_string()))?;

        let status = response.status().as_u16();
        if status != 200 {
            let retry_after = header_value(response.headers(), "retry-after");
            let body = response.text().unwrap_or_default();
            return Err(gemini_http_error(status, &body, retry_after.as_deref()));
        }

        let payload: Value = response.json().map_err(|exc| {
            CodeAssistError::with_code(
                format!("Invalid JSON from Code Assist: {exc}"),
                "code_assist_invalid_json",
            )
        })?;
        Ok(CompletionResult::Single(translate_gemini_response(
            &payload, &req.model,
        )))
    }

    /// Materialize the SSE stream into OpenAI-shaped chunks.
    ///
    /// The Python original returns a lazy generator; here we read the full
    /// stream body and parse it (reqwest::blocking's body is consumed eagerly
    /// anyway for error diagnostics).
    pub fn stream_completion(
        &self,
        model: &str,
        wrapped: &Value,
        headers: &[(String, String)],
    ) -> Result<Vec<ChatCompletionChunk>, CodeAssistError> {
        let url = format!("{CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse");
        let mut builder = self.http.post(&url).json(wrapped);
        for (k, v) in headers {
            // Accept overridden to text/event-stream for streaming.
            if k == "Accept" {
                builder = builder.header("Accept", "text/event-stream");
            } else {
                builder = builder.header(k, v);
            }
        }

        let response = builder.send().map_err(|exc| {
            CodeAssistError::with_code(
                format!("Streaming request failed: {exc}"),
                "code_assist_stream_error",
            )
        })?;

        let status = response.status().as_u16();
        if status != 200 {
            let retry_after = header_value(response.headers(), "retry-after");
            let body = response.text().unwrap_or_default();
            return Err(gemini_http_error(status, &body, retry_after.as_deref()));
        }

        let body = response.text().map_err(|exc| {
            CodeAssistError::with_code(
                format!("Streaming request failed: {exc}"),
                "code_assist_stream_error",
            )
        })?;

        let mut chunks: Vec<ChatCompletionChunk> = Vec::new();
        let mut tool_call_counter: usize = 0;
        for event in iter_sse_events(&body) {
            chunks.extend(translate_stream_event(&event, model, &mut tool_call_counter));
        }
        Ok(chunks)
    }
}

fn header_value(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coerce_text_str_and_list() {
        assert_eq!(coerce_content_to_text(&json!("hi")), "hi");
        assert_eq!(coerce_content_to_text(&Value::Null), "");
        let parts = json!([
            {"type": "text", "text": "a"},
            "b",
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "c"}
        ]);
        assert_eq!(coerce_content_to_text(&parts), "a\nb\nc");
    }

    #[test]
    fn tool_call_translation_parses_args() {
        let tc = json!({
            "function": {"name": "read_file", "arguments": "{\"path\": \"x\"}"}
        });
        let g = translate_tool_call_to_gemini(&tc);
        assert_eq!(g["functionCall"]["name"], "read_file");
        assert_eq!(g["functionCall"]["args"]["path"], "x");
        assert_eq!(g["thoughtSignature"], "skip_thought_signature_validator");
    }

    #[test]
    fn tool_call_bad_json_args_become_raw() {
        let tc = json!({"function": {"name": "f", "arguments": "not json"}});
        let g = translate_tool_call_to_gemini(&tc);
        assert_eq!(g["functionCall"]["args"]["_raw"], "not json");
    }

    #[test]
    fn tool_call_empty_args() {
        let tc = json!({"function": {"name": "f", "arguments": ""}});
        let g = translate_tool_call_to_gemini(&tc);
        assert_eq!(g["functionCall"]["args"], json!({}));
    }

    #[test]
    fn tool_result_wraps_plain_text() {
        let msg = json!({"role": "tool", "name": "f", "content": "hello"});
        let g = translate_tool_result_to_gemini(&msg);
        assert_eq!(g["functionResponse"]["name"], "f");
        assert_eq!(g["functionResponse"]["response"]["output"], "hello");
    }

    #[test]
    fn tool_result_parses_json_object() {
        let msg = json!({"role": "tool", "tool_call_id": "abc", "content": "{\"k\": 1}"});
        let g = translate_tool_result_to_gemini(&msg);
        assert_eq!(g["functionResponse"]["name"], "abc");
        assert_eq!(g["functionResponse"]["response"]["k"], 1);
    }

    #[test]
    fn contents_split_system_and_turns() {
        let messages = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "yo"}),
            json!({"role": "system", "content": "and helpful"}),
        ];
        let (contents, sys) = build_gemini_contents(&messages);
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        let sys = sys.unwrap();
        assert_eq!(sys["parts"][0]["text"], "be nice\nand helpful");
    }

    #[test]
    fn empty_parts_turn_skipped() {
        let messages = vec![json!({"role": "user", "content": ""})];
        let (contents, sys) = build_gemini_contents(&messages);
        assert!(contents.is_empty());
        assert!(sys.is_none());
    }

    #[test]
    fn tools_translation() {
        let tools = json!([
            {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}},
            {"type": "function", "function": {}},
            "garbage"
        ]);
        let g = translate_tools_to_gemini(Some(&tools));
        assert_eq!(g.len(), 1);
        let decls = g[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "f");
        assert_eq!(decls[0]["description"], "d");
    }

    #[test]
    fn tools_empty_returns_empty() {
        assert!(translate_tools_to_gemini(None).is_empty());
        assert!(translate_tools_to_gemini(Some(&json!([]))).is_empty());
    }

    #[test]
    fn tool_choice_variants() {
        assert_eq!(
            translate_tool_choice_to_gemini(Some(&json!("auto"))).unwrap()["functionCallingConfig"]
                ["mode"],
            "AUTO"
        );
        assert_eq!(
            translate_tool_choice_to_gemini(Some(&json!("required"))).unwrap()
                ["functionCallingConfig"]["mode"],
            "ANY"
        );
        assert_eq!(
            translate_tool_choice_to_gemini(Some(&json!("none"))).unwrap()["functionCallingConfig"]
                ["mode"],
            "NONE"
        );
        assert!(translate_tool_choice_to_gemini(None).is_none());
        let named = json!({"type": "function", "function": {"name": "x"}});
        let g = translate_tool_choice_to_gemini(Some(&named)).unwrap();
        assert_eq!(g["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(g["functionCallingConfig"]["allowedFunctionNames"][0], "x");
    }

    #[test]
    fn thinking_config_normalization() {
        let cfg = json!({"thinking_budget": 1024, "thinkingLevel": " HIGH ", "include_thoughts": true});
        let n = normalize_thinking_config(Some(&cfg)).unwrap();
        assert_eq!(n["thinkingBudget"], 1024);
        assert_eq!(n["thinkingLevel"], "high");
        assert_eq!(n["includeThoughts"], true);
        assert!(normalize_thinking_config(Some(&json!({}))).is_none());
        assert!(normalize_thinking_config(None).is_none());
    }

    #[test]
    fn build_request_generation_config() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = GenerationParams {
            temperature: Some(0.5),
            max_tokens: Some(100),
            top_p: Some(0.9),
            stop: Some(json!(["END", "", "STOP"])),
            thinking_config: None,
        };
        let body = build_gemini_request(&messages, None, None, &params);
        let gc = &body["generationConfig"];
        assert_eq!(gc["temperature"], 0.5);
        assert_eq!(gc["maxOutputTokens"], 100);
        assert_eq!(gc["topP"], 0.9);
        assert_eq!(gc["stopSequences"], json!(["END", "STOP"]));
    }

    #[test]
    fn build_request_max_tokens_zero_omitted() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = GenerationParams {
            max_tokens: Some(0),
            ..Default::default()
        };
        let body = build_gemini_request(&messages, None, None, &params);
        assert!(body.get("generationConfig").is_none());
    }

    #[test]
    fn wrap_envelope() {
        let inner = json!({"contents": []});
        let w = wrap_code_assist_request("proj", "gemini-2.5-flash", inner, Some("pid-1"));
        assert_eq!(w["project"], "proj");
        assert_eq!(w["model"], "gemini-2.5-flash");
        assert_eq!(w["user_prompt_id"], "pid-1");
        assert!(w["request"].is_object());
    }

    #[test]
    fn response_text_and_usage() {
        let resp = json!({
            "response": {
                "candidates": [{
                    "content": {"parts": [{"text": "hello"}, {"text": " world"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                    "cachedContentTokenCount": 3
                }
            }
        });
        let c = translate_gemini_response(&resp, "gemini-2.5-flash");
        assert_eq!(c.choices[0].message.content.as_deref(), Some("hello world"));
        assert_eq!(c.choices[0].finish_reason, "stop");
        assert_eq!(c.usage.prompt_tokens, 10);
        assert_eq!(c.usage.total_tokens, 15);
        assert_eq!(c.usage.prompt_tokens_details.cached_tokens, 3);
    }

    #[test]
    fn response_tool_calls_and_thoughts() {
        let resp = json!({
            "candidates": [{
                "content": {"parts": [
                    {"thought": true, "text": "thinking..."},
                    {"functionCall": {"name": "read_file", "args": {"path": "x"}}}
                ]},
                "finishReason": "STOP"
            }]
        });
        let c = translate_gemini_response(&resp, "m");
        assert_eq!(c.choices[0].finish_reason, "tool_calls");
        assert!(c.choices[0].message.content.is_none());
        let tcs = c.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "read_file");
        assert_eq!(tcs[0].r#type, "function");
        assert_eq!(c.choices[0].message.reasoning.as_deref(), Some("thinking..."));
    }

    #[test]
    fn response_empty_candidates() {
        let c = translate_gemini_response(&json!({"candidates": []}), "m");
        assert_eq!(c.choices[0].finish_reason, "stop");
        assert_eq!(c.choices[0].message.content.as_deref(), Some(""));
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(map_gemini_finish_reason("STOP"), "stop");
        assert_eq!(map_gemini_finish_reason("max_tokens"), "length");
        assert_eq!(map_gemini_finish_reason("SAFETY"), "content_filter");
        assert_eq!(map_gemini_finish_reason("RECITATION"), "content_filter");
        assert_eq!(map_gemini_finish_reason("WEIRD"), "stop");
    }

    #[test]
    fn sse_parsing() {
        let body = "data: {\"a\": 1}\r\n\ndata: {\"b\": 2}\nbogus\ndata: [DONE]\ndata: {\"c\": 3}\n";
        let events = iter_sse_events(body);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["a"], 1);
        assert_eq!(events[1]["b"], 2);
    }

    #[test]
    fn stream_event_parallel_tool_calls_get_unique_index() {
        let event = json!({
            "response": {
                "candidates": [{
                    "content": {"parts": [
                        {"functionCall": {"name": "read", "args": {"f": "a"}}},
                        {"functionCall": {"name": "read", "args": {"f": "b"}}}
                    ]},
                    "finishReason": "STOP"
                }]
            }
        });
        let mut counter = 0usize;
        let chunks = translate_stream_event(&event, "m", &mut counter);
        // two tool-call chunks + one finish chunk
        assert_eq!(chunks.len(), 3);
        let i0 = chunks[0].choices[0].delta.tool_calls.as_ref().unwrap()[0].index;
        let i1 = chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].index;
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(counter, 2);
    }

    #[test]
    fn stream_event_text_and_thought() {
        let event = json!({
            "candidates": [{
                "content": {"parts": [
                    {"thought": true, "text": "reasoning"},
                    {"text": "answer"}
                ]}
            }]
        });
        let mut counter = 0usize;
        let chunks = translate_stream_event(&event, "m", &mut counter);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.reasoning.as_deref(), Some("reasoning"));
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn http_error_capacity_exhausted() {
        let body = r#"{"error": {"code": 429, "message": "boom", "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "MODEL_CAPACITY_EXHAUSTED", "metadata": {"model": "gemini-2.5-pro"}},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}
            ]}}"#;
        let err = gemini_http_error(429, body, None);
        assert_eq!(err.code, "code_assist_capacity_exhausted");
        assert_eq!(err.status_code, Some(429));
        assert_eq!(err.retry_after, Some(30.0));
        assert!(err.message.contains("gemini-2.5-pro"));
        assert!(err.message.contains("retrying in 30s"));
        assert_eq!(err.details["reason"], "MODEL_CAPACITY_EXHAUSTED");
    }

    #[test]
    fn http_error_resource_exhausted_quota() {
        let body = r#"{"error": {"code": 429, "message": "quota gone", "status": "RESOURCE_EXHAUSTED"}}"#;
        let err = gemini_http_error(429, body, Some("12"));
        assert_eq!(err.code, "code_assist_rate_limited");
        assert_eq!(err.retry_after, Some(12.0));
        assert!(err.message.contains("quota exhausted"));
        assert!(err.message.contains("Retry suggested in 12s"));
    }

    #[test]
    fn http_error_404() {
        let body = r#"{"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}}"#;
        let err = gemini_http_error(404, body, None);
        assert_eq!(err.code, "code_assist_http_404");
        assert!(err.message.contains("not available at cloudcode-pa"));
    }

    #[test]
    fn http_error_401() {
        let err = gemini_http_error(401, r#"{"error": {"message": "bad token", "status": "UNAUTHENTICATED"}}"#, None);
        assert_eq!(err.code, "code_assist_unauthorized");
        assert!(err.message.contains("UNAUTHENTICATED"));
        assert!(err.message.contains("bad token"));
    }

    #[test]
    fn http_error_raw_body_fallback() {
        let err = gemini_http_error(500, "internal explosion", None);
        assert_eq!(err.code, "code_assist_http_500");
        assert!(err.message.contains("internal explosion"));
    }

    #[test]
    fn fmt_g_integers_and_floats() {
        assert_eq!(fmt_g(30.0), "30");
        assert_eq!(fmt_g(1.5), "1.5");
    }
}
