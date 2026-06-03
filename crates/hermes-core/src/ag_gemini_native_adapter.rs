//! OpenAI-compatible facade over Google AI Studio's native Gemini API.
//!
//! Hermes keeps `api_mode='chat_completions'` for the `gemini` provider so the
//! main agent loop can keep using its existing OpenAI-shaped message flow.
//! This adapter is the transport shim that converts those OpenAI-style
//! `messages[]` / `tools[]` requests into Gemini's native
//! `models/{model}:generateContent` schema and converts the responses back.
//!
//! Ported faithfully from `agent/gemini_native_adapter.py` (965 LOC).
//!
//! Because Rust is statically typed, the Python `SimpleNamespace` response and
//! stream-chunk objects are modelled here as concrete structs
//! (`ChatCompletion`, `StreamChunk`, ...) that serialize to the same
//! OpenAI-shaped JSON the rest of Hermes consumes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gemini_schema::sanitize_gemini_tool_parameters;

pub const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

const FREE_TIER_GUIDANCE: &str = concat!(
    "\n\nYour Google API key is on the free tier (<= 250 requests/day for ",
    "gemini-2.5-flash). Hermes typically makes 3-10 API calls per user turn, ",
    "so the free tier is exhausted in a handful of messages and cannot sustain ",
    "an agent session. Enable billing on your Google Cloud project and ",
    "regenerate the key in a billing-enabled project: ",
    "https://aistudio.google.com/apikey"
);

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn rand_hex12() -> String {
    // 12 hex chars, like uuid4().hex[:12]. We don't need crypto strength here.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // mix in a thread-local-ish counter via address of a stack value
    let mix = (&nanos as *const u128 as usize) as u128;
    let v = nanos ^ (mix.rotate_left(17)) ^ ((nanos.wrapping_mul(0x9E3779B97F4A7C15)) >> 3);
    format!("{:012x}", (v & 0xff_ffff_ffff_ffff) as u64)[..12].to_string()
}

fn chatcmpl_id() -> String {
    format!("chatcmpl-{}", rand_hex12())
}

fn call_id() -> String {
    format!("call_{}", rand_hex12())
}

// ---------------------------------------------------------------------------
// endpoint / tier helpers
// ---------------------------------------------------------------------------

/// Return true when the endpoint speaks Gemini's native REST API.
pub fn is_native_gemini_base_url(base_url: &str) -> bool {
    let normalized = base_url.trim().trim_end_matches('/').to_lowercase();
    if normalized.is_empty() {
        return false;
    }
    if !normalized.contains("generativelanguage.googleapis.com") {
        return false;
    }
    !normalized.ends_with("/openai")
}

/// Return true when a Gemini 429 message indicates free-tier exhaustion.
pub fn is_free_tier_quota_error(error_message: &str) -> bool {
    if error_message.is_empty() {
        return false;
    }
    error_message.to_lowercase().contains("free_tier")
}

fn normalize_probe_base(base_url: &str) -> String {
    let trimmed = base_url.trim();
    let mut normalized = if trimmed.is_empty() {
        DEFAULT_GEMINI_BASE_URL.to_string()
    } else {
        trimmed.trim_end_matches('/').to_string()
    };
    if normalized.is_empty() {
        normalized = DEFAULT_GEMINI_BASE_URL.to_string();
    }
    if normalized.to_lowercase().ends_with("/openai") {
        let len = normalized.len() - "/openai".len();
        normalized.truncate(len);
    }
    normalized
}

/// Probe a Google AI Studio API key and return its tier:
/// `"free"`, `"paid"`, or `"unknown"`.
pub fn probe_gemini_tier(
    api_key: &str,
    base_url: &str,
    model: &str,
    timeout_secs: f64,
) -> String {
    let key = api_key.trim();
    if key.is_empty() {
        return "unknown".to_string();
    }
    let normalized_base = normalize_probe_base(base_url);
    let url = format!("{}/models/{}:generateContent", normalized_base, model);
    let payload = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 1},
    });

    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs_f64(timeout_secs))
        .build()
    {
        Ok(c) => c,
        Err(_) => return "unknown".to_string(),
    };

    let resp = match client
        .post(&url)
        .query(&[("key", key)])
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
    {
        Ok(r) => r,
        Err(_) => return "unknown".to_string(),
    };

    let status = resp.status().as_u16();
    // lower-case header lookup
    let mut headers_lower: HashMap<String, String> = HashMap::new();
    for (k, v) in resp.headers().iter() {
        if let Ok(val) = v.to_str() {
            headers_lower.insert(k.as_str().to_lowercase(), val.to_string());
        }
    }
    let rpd_header = headers_lower.get("x-ratelimit-limit-requests-per-day");
    if let Some(h) = rpd_header {
        if !h.is_empty() {
            if let Ok(rpd_val) = h.parse::<i64>() {
                if rpd_val <= 1000 {
                    return "free".to_string();
                }
                if rpd_val > 1000 {
                    return "paid".to_string();
                }
            }
        }
    }

    if status == 429 {
        let body_text = resp.text().unwrap_or_default();
        if body_text.to_lowercase().contains("free_tier") {
            return "free".to_string();
        }
        return "paid".to_string();
    }

    if (200..300).contains(&status) {
        return "paid".to_string();
    }

    "unknown".to_string()
}

// ---------------------------------------------------------------------------
// error type
// ---------------------------------------------------------------------------

/// Error shape compatible with Hermes retry/error classification.
#[derive(Debug, Clone)]
pub struct GeminiAPIError {
    pub message: String,
    pub code: String,
    pub status_code: Option<u16>,
    pub retry_after: Option<f64>,
    /// The raw response body text (the Python version held the httpx.Response).
    pub response_body: Option<String>,
    pub details: Map<String, Value>,
}

impl GeminiAPIError {
    pub fn new(message: impl Into<String>) -> Self {
        GeminiAPIError {
            message: message.into(),
            code: "gemini_api_error".to_string(),
            status_code: None,
            retry_after: None,
            response_body: None,
            details: Map::new(),
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = code.into();
        self
    }
}

impl std::fmt::Display for GeminiAPIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeminiAPIError {}

// ---------------------------------------------------------------------------
// content coercion helpers
// ---------------------------------------------------------------------------

fn coerce_content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut pieces: Vec<String> = Vec::new();
            for part in items {
                match part {
                    Value::String(s) => pieces.push(s.clone()),
                    Value::Object(obj) => {
                        if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                            if let Some(Value::String(t)) = obj.get("text") {
                                pieces.push(t.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
            pieces.join("\n")
        }
        other => other.to_string(),
    }
}

fn extract_multimodal_parts(content: &Value) -> Vec<Value> {
    if !content.is_array() {
        let text = coerce_content_to_text(content);
        if text.is_empty() {
            return vec![];
        }
        return vec![json!({"text": text})];
    }

    let mut parts: Vec<Value> = Vec::new();
    for item in content.as_array().unwrap() {
        match item {
            Value::String(s) => {
                parts.push(json!({"text": s}));
            }
            Value::Object(obj) => {
                let ptype = obj.get("type").and_then(|v| v.as_str());
                match ptype {
                    Some("text") => {
                        if let Some(Value::String(t)) = obj.get("text") {
                            if !t.is_empty() {
                                parts.push(json!({"text": t}));
                            }
                        }
                    }
                    Some("image_url") => {
                        let url = obj
                            .get("image_url")
                            .and_then(|v| v.as_object())
                            .and_then(|m| m.get("url"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !url.starts_with("data:") {
                            continue;
                        }
                        // header,encoded  ; header == "data:<mime>;..."
                        let split: Vec<&str> = url.splitn(2, ',').collect();
                        if split.len() != 2 {
                            continue;
                        }
                        let header = split[0];
                        let encoded = split[1];
                        // mime = header.split(":",1)[1].split(";",1)[0]
                        let after_colon = match header.splitn(2, ':').nth(1) {
                            Some(s) => s,
                            None => continue,
                        };
                        let mime = after_colon.splitn(2, ';').next().unwrap_or("");
                        // decode then re-encode (validates the base64 round-trip)
                        use base64::Engine;
                        let raw = match base64::engine::general_purpose::STANDARD.decode(encoded) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        let re = base64::engine::general_purpose::STANDARD.encode(&raw);
                        parts.push(json!({
                            "inlineData": {
                                "mimeType": mime,
                                "data": re,
                            }
                        }));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// tool-call / tool-result translation
// ---------------------------------------------------------------------------

fn tool_call_extra_signature(tool_call: &Map<String, Value>) -> Option<String> {
    let extra = match tool_call.get("extra_content") {
        Some(Value::Object(m)) => m,
        _ => return None,
    };
    let google = extra.get("google").or_else(|| extra.get("thought_signature"));
    match google {
        Some(Value::Object(g)) => {
            let sig = g
                .get("thought_signature")
                .or_else(|| g.get("thoughtSignature"));
            match sig {
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            }
        }
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

fn translate_tool_call_to_gemini(tool_call: &Map<String, Value>) -> Value {
    let fn_obj = tool_call
        .get("function")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let args_raw = fn_obj.get("arguments");

    let args: Value = match args_raw {
        Some(Value::String(s)) if !s.is_empty() => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => parsed,
            Err(_) => json!({"_raw": s}),
        },
        Some(Value::String(_)) => json!({}),
        None => json!({}),
        Some(other) => other.clone(),
    };
    // ensure dict shape
    let args = if args.is_object() {
        args
    } else {
        json!({"_value": args})
    };

    let name = fn_obj
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut part = Map::new();
    part.insert(
        "functionCall".to_string(),
        json!({"name": name, "args": args}),
    );
    if let Some(sig) = tool_call_extra_signature(tool_call) {
        part.insert("thoughtSignature".to_string(), Value::String(sig));
    }
    Value::Object(part)
}

fn translate_tool_result_to_gemini(
    message: &Map<String, Value>,
    tool_name_by_call_id: &HashMap<String, String>,
) -> Value {
    let tool_call_id = message
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = message
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            tool_name_by_call_id
                .get(&tool_call_id)
                .filter(|s| !s.is_empty())
                .cloned()
        })
        .or_else(|| {
            if tool_call_id.is_empty() {
                None
            } else {
                Some(tool_call_id.clone())
            }
        })
        .unwrap_or_else(|| "tool".to_string());

    let content = coerce_content_to_text(message.get("content").unwrap_or(&Value::Null));
    let trimmed = content.trim_start();
    let parsed: Option<Value> = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str::<Value>(&content).ok()
    } else {
        None
    };
    let response = match parsed {
        Some(Value::Object(m)) => Value::Object(m),
        _ => json!({"output": content}),
    };
    json!({
        "functionResponse": {
            "name": name,
            "response": response,
        }
    })
}

/// Build Gemini `contents[]` and an optional `systemInstruction` from
/// OpenAI-style messages. Returns `(contents, system_instruction)`.
pub fn build_gemini_contents(messages: &[Value]) -> (Vec<Value>, Option<Value>) {
    let mut system_text_parts: Vec<String> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    let mut tool_name_by_call_id: HashMap<String, String> = HashMap::new();

    for msg in messages {
        let msg = match msg.as_object() {
            Some(m) => m,
            None => continue,
        };
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

        if role == "system" {
            system_text_parts
                .push(coerce_content_to_text(msg.get("content").unwrap_or(&Value::Null)));
            continue;
        }

        if role == "tool" || role == "function" {
            contents.push(json!({
                "role": "user",
                "parts": [translate_tool_result_to_gemini(msg, &tool_name_by_call_id)],
            }));
            continue;
        }

        let gemini_role = if role == "assistant" { "model" } else { "user" };
        let mut parts: Vec<Value> =
            extract_multimodal_parts(msg.get("content").unwrap_or(&Value::Null));

        if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
            for tool_call in tool_calls {
                if let Some(tc) = tool_call.as_object() {
                    let tcid = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .or_else(|| tc.get("call_id").and_then(|v| v.as_str()))
                        .unwrap_or("")
                        .to_string();
                    let tname = tc
                        .get("function")
                        .and_then(|v| v.as_object())
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !tcid.is_empty() && !tname.is_empty() {
                        tool_name_by_call_id.insert(tcid, tname);
                    }
                    parts.push(translate_tool_call_to_gemini(tc));
                }
            }
        }

        if !parts.is_empty() {
            contents.push(json!({"role": gemini_role, "parts": parts}));
        }
    }

    let joined_system = system_text_parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    let system_instruction = if joined_system.is_empty() {
        None
    } else {
        Some(json!({"parts": [{"text": joined_system}]}))
    };

    (contents, system_instruction)
}

fn translate_tools_to_gemini(tools: &Value) -> Vec<Value> {
    let arr = match tools.as_array() {
        Some(a) => a,
        None => return vec![],
    };
    let mut declarations: Vec<Value> = Vec::new();
    for tool in arr {
        let tool = match tool.as_object() {
            Some(t) => t,
            None => continue,
        };
        let fn_obj = match tool.get("function").and_then(|v| v.as_object()) {
            Some(f) => f,
            None => continue,
        };
        let name = match fn_obj.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };
        let mut decl = Map::new();
        decl.insert("name".to_string(), Value::String(name.to_string()));
        if let Some(desc) = fn_obj.get("description").and_then(|v| v.as_str()) {
            if !desc.is_empty() {
                decl.insert("description".to_string(), Value::String(desc.to_string()));
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
        vec![]
    } else {
        vec![json!({"functionDeclarations": declarations})]
    }
}

fn translate_tool_choice_to_gemini(tool_choice: &Value) -> Option<Value> {
    match tool_choice {
        Value::Null => None,
        Value::String(s) => match s.as_str() {
            "auto" => Some(json!({"functionCallingConfig": {"mode": "AUTO"}})),
            "required" => Some(json!({"functionCallingConfig": {"mode": "ANY"}})),
            "none" => Some(json!({"functionCallingConfig": {"mode": "NONE"}})),
            _ => None,
        },
        Value::Object(obj) => {
            let name = obj
                .get("function")
                .and_then(|v| v.as_object())
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str());
            match name {
                Some(n) if !n.is_empty() => Some(json!({
                    "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [n]}
                })),
                _ => None,
            }
        }
        _ => None,
    }
}

fn normalize_thinking_config(config: &Value) -> Option<Value> {
    let obj = match config.as_object() {
        Some(m) if !m.is_empty() => m,
        _ => return None,
    };
    let budget = obj.get("thinkingBudget").or_else(|| obj.get("thinking_budget"));
    let include = obj
        .get("includeThoughts")
        .or_else(|| obj.get("include_thoughts"));
    let level = obj.get("thinkingLevel").or_else(|| obj.get("thinking_level"));

    let mut normalized = Map::new();
    // isinstance(budget, (int, float)) and bool IS an int in Python, but
    // serde_json bools are not numbers, so we mirror "number" only.
    if let Some(b) = budget {
        if b.is_i64() || b.is_u64() || b.is_f64() {
            if let Some(n) = b.as_f64() {
                normalized.insert("thinkingBudget".to_string(), json!(n as i64));
            }
        }
    }
    if let Some(Value::Bool(inc)) = include {
        normalized.insert("includeThoughts".to_string(), Value::Bool(*inc));
    }
    if let Some(Value::String(l)) = level {
        if !l.trim().is_empty() {
            normalized.insert(
                "thinkingLevel".to_string(),
                Value::String(l.trim().to_lowercase()),
            );
        }
    }
    if normalized.is_empty() {
        None
    } else {
        Some(Value::Object(normalized))
    }
}

/// Parameters for building a native Gemini request.
#[derive(Default, Clone)]
pub struct GeminiRequestParams<'a> {
    pub messages: &'a [Value],
    pub tools: Value,
    pub tool_choice: Value,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub stop: Value,
    pub thinking_config: Value,
}

/// Build the native Gemini `generateContent` request body.
pub fn build_gemini_request(params: &GeminiRequestParams) -> Value {
    let (contents, system_instruction) = build_gemini_contents(params.messages);
    let mut request = Map::new();
    request.insert("contents".to_string(), json!(contents));
    if let Some(si) = system_instruction {
        request.insert("systemInstruction".to_string(), si);
    }

    let gemini_tools = translate_tools_to_gemini(&params.tools);
    if !gemini_tools.is_empty() {
        request.insert("tools".to_string(), json!(gemini_tools));
    }

    if let Some(tc) = translate_tool_choice_to_gemini(&params.tool_choice) {
        request.insert("toolConfig".to_string(), tc);
    }

    let mut generation_config = Map::new();
    if let Some(t) = params.temperature {
        generation_config.insert("temperature".to_string(), json!(t));
    }
    if let Some(m) = params.max_tokens {
        generation_config.insert("maxOutputTokens".to_string(), json!(m));
    }
    if let Some(p) = params.top_p {
        generation_config.insert("topP".to_string(), json!(p));
    }
    // `if stop:` -> truthy. Lists/strings/numbers that are non-empty/non-zero.
    if value_is_truthy(&params.stop) {
        let stop_seq = match &params.stop {
            Value::Array(a) => Value::Array(a.clone()),
            other => json!([value_to_str(other)]),
        };
        generation_config.insert("stopSequences".to_string(), stop_seq);
    }
    if let Some(nt) = normalize_thinking_config(&params.thinking_config) {
        generation_config.insert("thinkingConfig".to_string(), nt);
    }
    if !generation_config.is_empty() {
        request.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }

    Value::Object(request)
}

fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn map_gemini_finish_reason(reason: &str) -> &'static str {
    match reason.to_uppercase().as_str() {
        "STOP" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        "RECITATION" => "content_filter",
        "OTHER" => "stop",
        _ => "stop",
    }
}

fn tool_call_extra_from_part(part: &Map<String, Value>) -> Option<Value> {
    if let Some(Value::String(sig)) = part.get("thoughtSignature") {
        if !sig.is_empty() {
            return Some(json!({"google": {"thought_signature": sig}}));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// OpenAI-shaped response structs (replacing Python SimpleNamespace)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub index: usize,
    pub function: FunctionCall,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

fn empty_response(model: &str) -> ChatCompletion {
    ChatCompletion {
        id: chatcmpl_id(),
        object: "chat.completion".to_string(),
        created: now_unix(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: Some(String::new()),
                tool_calls: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
        },
    }
}

fn usage_int(meta: &Map<String, Value>, key: &str) -> i64 {
    meta.get(key)
        .and_then(|v| v.as_i64())
        .or_else(|| meta.get(key).and_then(|v| v.as_f64()).map(|f| f as i64))
        .unwrap_or(0)
}

/// Translate a native Gemini `generateContent` response into an OpenAI-shaped
/// `ChatCompletion`.
pub fn translate_gemini_response(resp: &Value, model: &str) -> ChatCompletion {
    let candidates = match resp.get("candidates").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => c,
        _ => return empty_response(model),
    };

    let cand = candidates[0].as_object().cloned().unwrap_or_default();
    let parts: Vec<Value> = cand
        .get("content")
        .and_then(|v| v.as_object())
        .and_then(|c| c.get("parts"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut text_pieces: Vec<String> = Vec::new();
    let mut reasoning_pieces: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for (index, part) in parts.iter().enumerate() {
        let part = match part.as_object() {
            Some(p) => p,
            None => continue,
        };
        let is_thought = part.get("thought") == Some(&Value::Bool(true));
        if is_thought {
            if let Some(Value::String(t)) = part.get("text") {
                reasoning_pieces.push(t.clone());
                continue;
            }
        }
        if let Some(Value::String(t)) = part.get("text") {
            text_pieces.push(t.clone());
            continue;
        }
        if let Some(fc) = part.get("functionCall").and_then(|v| v.as_object()) {
            let name = fc.get("name").and_then(|v| v.as_str());
            if let Some(name) = name {
                if !name.is_empty() {
                    let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                    let extra = tool_call_extra_from_part(part);
                    tool_calls.push(ToolCall {
                        id: call_id(),
                        call_type: "function".to_string(),
                        index,
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: args_str,
                        },
                        extra_content: extra,
                    });
                }
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls".to_string()
    } else {
        map_gemini_finish_reason(
            cand.get("finishReason").and_then(|v| v.as_str()).unwrap_or(""),
        )
        .to_string()
    };

    let usage_meta = resp
        .get("usageMetadata")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let usage = Usage {
        prompt_tokens: usage_int(&usage_meta, "promptTokenCount"),
        completion_tokens: usage_int(&usage_meta, "candidatesTokenCount"),
        total_tokens: usage_int(&usage_meta, "totalTokenCount"),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: usage_int(&usage_meta, "cachedContentTokenCount"),
        },
    };

    let reasoning = if reasoning_pieces.is_empty() {
        None
    } else {
        Some(reasoning_pieces.concat())
    };
    let content = if text_pieces.is_empty() {
        None
    } else {
        Some(text_pieces.concat())
    };

    ChatCompletion {
        id: chatcmpl_id(),
        object: "chat.completion".to_string(),
        created: now_unix(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                reasoning: reasoning.clone(),
                reasoning_content: reasoning,
                reasoning_details: None,
            },
            finish_reason,
        }],
        usage,
    }
}

// ---------------------------------------------------------------------------
// streaming structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

/// Internal description of a tool-call delta passed to `make_stream_chunk`.
struct ToolDeltaInput {
    index: usize,
    id: Option<String>,
    name: String,
    arguments: String,
    extra_content: Option<Value>,
}

fn make_stream_chunk(
    model: &str,
    content: &str,
    tool_call_delta: Option<ToolDeltaInput>,
    finish_reason: Option<String>,
    reasoning: &str,
) -> StreamChunk {
    let mut delta = Delta {
        role: "assistant".to_string(),
        content: None,
        tool_calls: None,
        reasoning: None,
        reasoning_content: None,
    };
    if !content.is_empty() {
        delta.content = Some(content.to_string());
    }
    if let Some(tcd) = tool_call_delta {
        let id = tcd.id.filter(|s| !s.is_empty()).unwrap_or_else(call_id);
        let mut td = ToolCallDelta {
            index: tcd.index,
            id,
            call_type: "function".to_string(),
            function: FunctionCall {
                name: tcd.name,
                arguments: tcd.arguments,
            },
            extra_content: None,
        };
        if let Some(extra) = tcd.extra_content {
            if extra.is_object() {
                td.extra_content = Some(extra);
            }
        }
        delta.tool_calls = Some(vec![td]);
    }
    if !reasoning.is_empty() {
        delta.reasoning = Some(reasoning.to_string());
        delta.reasoning_content = Some(reasoning.to_string());
    }
    StreamChunk {
        id: chatcmpl_id(),
        object: "chat.completion.chunk".to_string(),
        created: now_unix(),
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
    }
}

/// Per-call slot state used to dedupe streamed function-call argument deltas.
#[derive(Default, Clone)]
pub struct ToolCallSlot {
    pub index: usize,
    pub id: String,
    pub last_arguments: String,
}

/// State carried across `translate_stream_event` calls within one stream.
#[derive(Default)]
pub struct ToolCallIndices {
    map: indexmap_like::OrderedMap,
}

// We need insertion-order-preserving semantics like Python dict for
// `len(tool_call_indices)` ordering. A tiny ordered-map shim avoids an extra
// crate dependency.
mod indexmap_like {
    use super::ToolCallSlot;

    #[derive(Default)]
    pub struct OrderedMap {
        keys: Vec<String>,
        vals: Vec<ToolCallSlot>,
    }

    impl OrderedMap {
        pub fn len(&self) -> usize {
            self.keys.len()
        }
        pub fn is_empty(&self) -> bool {
            self.keys.is_empty()
        }
        pub fn get(&self, key: &str) -> Option<&ToolCallSlot> {
            self.keys
                .iter()
                .position(|k| k == key)
                .map(|i| &self.vals[i])
        }
        pub fn get_mut(&mut self, key: &str) -> Option<&mut ToolCallSlot> {
            match self.keys.iter().position(|k| k == key) {
                Some(i) => Some(&mut self.vals[i]),
                None => None,
            }
        }
        pub fn insert(&mut self, key: String, val: ToolCallSlot) {
            match self.keys.iter().position(|k| *k == key) {
                Some(i) => self.vals[i] = val,
                None => {
                    self.keys.push(key);
                    self.vals.push(val);
                }
            }
        }
    }
}

impl ToolCallIndices {
    pub fn new() -> Self {
        ToolCallIndices::default()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// Translate a single Gemini SSE event into a list of OpenAI-style stream
/// chunks, mutating `tool_call_indices` to track argument-delta state.
pub fn translate_stream_event(
    event: &Value,
    model: &str,
    tool_call_indices: &mut ToolCallIndices,
) -> Vec<StreamChunk> {
    let candidates = match event.get("candidates").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => c,
        _ => return vec![],
    };
    let cand = candidates[0].as_object().cloned().unwrap_or_default();
    let parts: Vec<Value> = cand
        .get("content")
        .and_then(|v| v.as_object())
        .and_then(|c| c.get("parts"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut chunks: Vec<StreamChunk> = Vec::new();

    for (part_index, part) in parts.iter().enumerate() {
        let part = match part.as_object() {
            Some(p) => p,
            None => continue,
        };
        let is_thought = part.get("thought") == Some(&Value::Bool(true));
        if is_thought {
            if let Some(Value::String(t)) = part.get("text") {
                chunks.push(make_stream_chunk(model, "", None, None, t));
                continue;
            }
        }
        if let Some(Value::String(t)) = part.get("text") {
            if !t.is_empty() {
                chunks.push(make_stream_chunk(model, t, None, None, ""));
            }
        }
        if let Some(fc) = part.get("functionCall").and_then(|v| v.as_object()) {
            let name = match fc.get("name").and_then(|v| v.as_str()) {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
            // json.dumps(..., sort_keys=True) -> sort object keys
            let args_str = dumps_sorted(&args);
            let thought_signature = match part.get("thoughtSignature") {
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            let call_key = dumps_sorted(&json!({
                "part_index": part_index,
                "name": name,
                "thought_signature": thought_signature,
            }));

            let (slot_index, slot_id);
            if let Some(existing) = tool_call_indices.map.get(&call_key) {
                slot_index = existing.index;
                slot_id = existing.id.clone();
            } else {
                let new_slot = ToolCallSlot {
                    index: tool_call_indices.map.len(),
                    id: call_id(),
                    last_arguments: String::new(),
                };
                slot_index = new_slot.index;
                slot_id = new_slot.id.clone();
                tool_call_indices.map.insert(call_key.clone(), new_slot);
            }

            let last_arguments = tool_call_indices
                .map
                .get(&call_key)
                .map(|s| s.last_arguments.clone())
                .unwrap_or_default();

            let mut emitted_arguments = args_str.clone();
            if !last_arguments.is_empty() {
                if args_str == last_arguments {
                    emitted_arguments = String::new();
                } else if args_str.starts_with(&last_arguments) {
                    emitted_arguments = args_str[last_arguments.len()..].to_string();
                }
            }
            if let Some(slot) = tool_call_indices.map.get_mut(&call_key) {
                slot.last_arguments = args_str.clone();
            }

            chunks.push(make_stream_chunk(
                model,
                "",
                Some(ToolDeltaInput {
                    index: slot_index,
                    id: Some(slot_id),
                    name,
                    arguments: emitted_arguments,
                    extra_content: tool_call_extra_from_part(part),
                }),
                None,
                "",
            ));
        }
    }

    let finish_reason_raw = cand.get("finishReason").and_then(|v| v.as_str()).unwrap_or("");
    if !finish_reason_raw.is_empty() {
        let mapped = if !tool_call_indices.is_empty() {
            "tool_calls".to_string()
        } else {
            map_gemini_finish_reason(finish_reason_raw).to_string()
        };
        let mut finish_chunk = make_stream_chunk(model, "", None, Some(mapped), "");
        if let Some(usage_meta) = event.get("usageMetadata").and_then(|v| v.as_object()) {
            if !usage_meta.is_empty() {
                finish_chunk.usage = Some(Usage {
                    prompt_tokens: usage_int(usage_meta, "promptTokenCount"),
                    completion_tokens: usage_int(usage_meta, "candidatesTokenCount"),
                    total_tokens: usage_int(usage_meta, "totalTokenCount"),
                    prompt_tokens_details: PromptTokensDetails {
                        cached_tokens: usage_int(usage_meta, "cachedContentTokenCount"),
                    },
                });
            }
        }
        chunks.push(finish_chunk);
    }

    chunks
}

/// Serialize a JSON value with object keys sorted, matching Python's
/// `json.dumps(..., sort_keys=True)`. serde_json with the `preserve_order`
/// feature would keep insertion order, so we sort explicitly.
fn dumps_sorted(value: &Value) -> String {
    let sorted = sort_value(value);
    serde_json::to_string(&sorted).unwrap_or_else(|_| "{}".to_string())
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), sort_value(&m[k]));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// Parse Server-Sent-Events lines from a Gemini stream body, returning the
/// JSON `data:` payloads in order. Stops at a `[DONE]` sentinel.
pub fn parse_sse_events(body: &str) -> Vec<Value> {
    let mut events: Vec<Value> = Vec::new();
    for raw_line in body.split('\n') {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if !line.starts_with("data: ") {
            continue;
        }
        let data = &line[6..];
        if data == "[DONE]" {
            break;
        }
        if let Ok(payload) = serde_json::from_str::<Value>(data) {
            if payload.is_object() {
                events.push(payload);
            }
        }
    }
    events
}

// ---------------------------------------------------------------------------
// HTTP error construction
// ---------------------------------------------------------------------------

/// Build a `GeminiAPIError` from an HTTP status, body text, and a Retry-After
/// header value. (The Python version took the httpx.Response object; here we
/// pass the already-extracted pieces so this is testable without a live call.)
pub fn gemini_http_error(status: u16, body_text: &str, retry_after_header: Option<&str>) -> GeminiAPIError {
    let mut body_json: Map<String, Value> = Map::new();
    if !body_text.is_empty() {
        if let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(body_text) {
            body_json = parsed;
        }
    }

    let err_obj = body_json
        .get("error")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let err_status = err_obj
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let err_message = err_obj
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let details_list = err_obj
        .get("details")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut reason = String::new();
    let mut metadata: Value = json!({});
    for detail in &details_list {
        let detail = match detail.as_object() {
            Some(d) => d,
            None => continue,
        };
        let type_url = detail.get("@type").and_then(|v| v.as_str()).unwrap_or("");
        if reason.is_empty() && type_url.ends_with("/google.rpc.ErrorInfo") {
            if let Some(Value::String(r)) = detail.get("reason") {
                reason = r.clone();
            }
            if let Some(md) = detail.get("metadata").filter(|v| v.is_object()) {
                metadata = md.clone();
            }
        }
    }

    let retry_after: Option<f64> = retry_after_header.and_then(|h| h.parse::<f64>().ok());

    let code = match status {
        401 => "gemini_unauthorized".to_string(),
        429 => "gemini_rate_limited".to_string(),
        404 => "gemini_model_not_found".to_string(),
        other => format!("gemini_http_{}", other),
    };

    let mut message = if !err_message.is_empty() {
        let status_label = if err_status.is_empty() {
            "error"
        } else {
            &err_status
        };
        format!("Gemini HTTP {} ({}): {}", status, status_label, err_message)
    } else {
        let truncated: String = body_text.chars().take(500).collect();
        format!("Gemini returned HTTP {}: {}", status, truncated)
    };

    let free_check = if !err_message.is_empty() {
        err_message.clone()
    } else {
        body_text.to_string()
    };
    if status == 429 && is_free_tier_quota_error(&free_check) {
        message.push_str(FREE_TIER_GUIDANCE);
    }

    let mut details = Map::new();
    details.insert("status".to_string(), Value::String(err_status));
    details.insert("reason".to_string(), Value::String(reason));
    details.insert("metadata".to_string(), metadata);
    details.insert("message".to_string(), Value::String(err_message));

    GeminiAPIError {
        message,
        code,
        status_code: Some(status),
        retry_after,
        response_body: Some(body_text.to_string()),
        details,
    }
}

// ---------------------------------------------------------------------------
// client
// ---------------------------------------------------------------------------

/// Minimal OpenAI-SDK-compatible facade over Gemini's native REST API,
/// implemented with `reqwest::blocking`.
pub struct GeminiNativeClient {
    pub api_key: String,
    pub base_url: String,
    pub is_closed: bool,
    default_headers: HashMap<String, String>,
    http: reqwest::blocking::Client,
}

/// Options for a chat-completion call, mirroring the Python `create(**kwargs)`.
#[derive(Clone)]
pub struct ChatCompletionOptions {
    pub model: String,
    pub messages: Vec<Value>,
    pub stream: bool,
    pub tools: Value,
    pub tool_choice: Value,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub stop: Value,
    pub extra_body: Value,
    pub timeout_secs: Option<f64>,
}

impl Default for ChatCompletionOptions {
    fn default() -> Self {
        ChatCompletionOptions {
            model: "gemini-2.5-flash".to_string(),
            messages: vec![],
            stream: false,
            tools: Value::Null,
            tool_choice: Value::Null,
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: Value::Null,
            extra_body: Value::Null,
            timeout_secs: None,
        }
    }
}

impl GeminiNativeClient {
    /// Construct a client. Returns an error if no API key is supplied.
    pub fn new(
        api_key: &str,
        base_url: Option<&str>,
        default_headers: Option<HashMap<String, String>>,
        timeout_secs: Option<f64>,
    ) -> Result<Self, GeminiAPIError> {
        if api_key.trim().is_empty() {
            return Err(GeminiAPIError::new(
                "Gemini native client requires an API key, but none was provided. \
                 Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment / ~/.hermes/.env \
                 (get one at https://aistudio.google.com/app/apikey), or run `hermes setup` \
                 to configure the Google provider.",
            ));
        }

        let mut normalized_base = base_url
            .unwrap_or(DEFAULT_GEMINI_BASE_URL)
            .trim_end_matches('/')
            .to_string();
        if normalized_base.ends_with("/openai") {
            let len = normalized_base.len() - "/openai".len();
            normalized_base.truncate(len);
        }

        // Python default httpx.Timeout(connect=15, read=600, write=30, pool=30).
        // reqwest::blocking uses a single overall timeout; mirror the read cap.
        let read_timeout = timeout_secs.unwrap_or(600.0);
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs_f64(read_timeout))
            .build()
            .map_err(|e| GeminiAPIError::new(format!("Failed to build HTTP client: {e}")))?;

        Ok(GeminiNativeClient {
            api_key: api_key.to_string(),
            base_url: normalized_base,
            is_closed: false,
            default_headers: default_headers.unwrap_or_default(),
            http,
        })
    }

    pub fn close(&mut self) {
        self.is_closed = true;
    }

    fn headers(&self) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Accept".to_string(), "application/json".to_string()),
            ("x-goog-api-key".to_string(), self.api_key.clone()),
            (
                "User-Agent".to_string(),
                "hermes-agent (gemini-native)".to_string(),
            ),
        ];
        for (k, v) in &self.default_headers {
            // dict.update semantics: override or append
            if let Some(entry) = headers.iter_mut().find(|(hk, _)| hk == k) {
                entry.1 = v.clone();
            } else {
                headers.push((k.clone(), v.clone()));
            }
        }
        headers
    }

    /// Non-streaming chat completion. For streaming, use
    /// [`GeminiNativeClient::stream_completion`].
    pub fn create_chat_completion(
        &self,
        opts: &ChatCompletionOptions,
    ) -> Result<ChatCompletion, GeminiAPIError> {
        if opts.stream {
            return Err(GeminiAPIError::new(
                "create_chat_completion called with stream=true; use stream_completion instead",
            )
            .with_code("gemini_api_error"));
        }
        let request = self.build_request(opts);
        let url = format!("{}/models/{}:generateContent", self.base_url, opts.model);

        let mut req = self.http.post(&url).json(&request);
        for (k, v) in self.headers() {
            req = req.header(k, v);
        }
        if let Some(t) = opts.timeout_secs {
            req = req.timeout(std::time::Duration::from_secs_f64(t));
        }

        let response = req
            .send()
            .map_err(|e| GeminiAPIError::new(format!("Gemini request failed: {e}")).with_code("gemini_stream_error"))?;

        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("Retry-After")
            .or_else(|| response.headers().get("retry-after"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        if status != 200 {
            let body = response.text().unwrap_or_default();
            return Err(gemini_http_error(status, &body, retry_after.as_deref()));
        }

        let body = response.text().unwrap_or_default();
        let payload: Value = serde_json::from_str(&body).map_err(|e| GeminiAPIError {
            message: format!("Invalid JSON from Gemini native API: {e}"),
            code: "gemini_invalid_json".to_string(),
            status_code: Some(status),
            retry_after: None,
            response_body: Some(body.clone()),
            details: Map::new(),
        })?;

        Ok(translate_gemini_response(&payload, &opts.model))
    }

    fn build_request(&self, opts: &ChatCompletionOptions) -> Value {
        let thinking_config = match opts.extra_body.as_object() {
            Some(eb) => eb
                .get("thinking_config")
                .or_else(|| eb.get("thinkingConfig"))
                .cloned()
                .unwrap_or(Value::Null),
            None => Value::Null,
        };
        let params = GeminiRequestParams {
            messages: &opts.messages,
            tools: opts.tools.clone(),
            tool_choice: opts.tool_choice.clone(),
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            top_p: opts.top_p,
            stop: opts.stop.clone(),
            thinking_config,
        };
        build_gemini_request(&params)
    }

    /// Streaming chat completion. Performs the request, reads the SSE body,
    /// and returns the translated chunks in order. (The Python version yielded
    /// chunks lazily; here we collect them since reqwest::blocking buffers.)
    pub fn stream_completion(
        &self,
        opts: &ChatCompletionOptions,
    ) -> Result<Vec<StreamChunk>, GeminiAPIError> {
        let request = self.build_request(opts);
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, opts.model
        );

        let mut req = self.http.post(&url).json(&request);
        for (k, v) in self.headers() {
            let k = if k == "Accept" { k } else { k };
            req = req.header(k, v);
        }
        // Override Accept for SSE.
        req = req.header("Accept", "text/event-stream");
        if let Some(t) = opts.timeout_secs {
            req = req.timeout(std::time::Duration::from_secs_f64(t));
        }

        let response = req.send().map_err(|e| {
            GeminiAPIError::new(format!("Gemini streaming request failed: {e}"))
                .with_code("gemini_stream_error")
        })?;

        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("Retry-After")
            .or_else(|| response.headers().get("retry-after"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        if status != 200 {
            let body = response.text().unwrap_or_default();
            return Err(gemini_http_error(status, &body, retry_after.as_deref()));
        }

        let body = response.text().map_err(|e| {
            GeminiAPIError::new(format!("Gemini streaming request failed: {e}"))
                .with_code("gemini_stream_error")
        })?;

        let events = parse_sse_events(&body);
        let mut tool_call_indices = ToolCallIndices::new();
        let mut chunks: Vec<StreamChunk> = Vec::new();
        for event in &events {
            chunks.extend(translate_stream_event(event, &opts.model, &mut tool_call_indices));
        }
        Ok(chunks)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_native_gemini_base_url() {
        assert!(is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta"
        ));
        assert!(is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta/"
        ));
        assert!(!is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ));
        assert!(!is_native_gemini_base_url("https://api.openai.com/v1"));
        assert!(!is_native_gemini_base_url(""));
    }

    #[test]
    fn test_is_free_tier_quota_error() {
        assert!(is_free_tier_quota_error("quota exceeded for FREE_TIER"));
        assert!(is_free_tier_quota_error("free_tier limit"));
        assert!(!is_free_tier_quota_error("paid tier"));
        assert!(!is_free_tier_quota_error(""));
    }

    #[test]
    fn test_coerce_content_to_text() {
        assert_eq!(coerce_content_to_text(&Value::Null), "");
        assert_eq!(coerce_content_to_text(&json!("hello")), "hello");
        assert_eq!(
            coerce_content_to_text(&json!([
                {"type": "text", "text": "a"},
                "b",
                {"type": "image_url"},
                {"type": "text", "text": "c"}
            ])),
            "a\nc"
        );
    }

    #[test]
    fn test_extract_multimodal_image() {
        // "data:image/png;base64,<b64 of 'hi'>"
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"hi");
        let url = format!("data:image/png;base64,{}", b64);
        let parts = extract_multimodal_parts(&json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": url}}
        ]));
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], json!({"text": "look"}));
        assert_eq!(parts[1]["inlineData"]["mimeType"], json!("image/png"));
        assert_eq!(parts[1]["inlineData"]["data"], json!(b64));
    }

    #[test]
    fn test_translate_tool_call_to_gemini() {
        let tc: Map<String, Value> = json!({
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}
        })
        .as_object()
        .unwrap()
        .clone();
        let part = translate_tool_call_to_gemini(&tc);
        assert_eq!(part["functionCall"]["name"], json!("get_weather"));
        assert_eq!(part["functionCall"]["args"]["city"], json!("NYC"));
    }

    #[test]
    fn test_translate_tool_call_bad_args() {
        let tc: Map<String, Value> = json!({
            "function": {"name": "f", "arguments": "not json"}
        })
        .as_object()
        .unwrap()
        .clone();
        let part = translate_tool_call_to_gemini(&tc);
        assert_eq!(part["functionCall"]["args"]["_raw"], json!("not json"));
    }

    #[test]
    fn test_tool_call_extra_signature() {
        let tc: Map<String, Value> = json!({
            "extra_content": {"google": {"thought_signature": "sig123"}}
        })
        .as_object()
        .unwrap()
        .clone();
        assert_eq!(tool_call_extra_signature(&tc), Some("sig123".to_string()));
    }

    #[test]
    fn test_build_contents_system_and_tool() {
        let messages = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "system", "content": "be brief"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "tool_a", "arguments": "{}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "{\"result\": 42}"}),
        ];
        let (contents, sys) = build_gemini_contents(&messages);
        let sys = sys.unwrap();
        assert_eq!(sys["parts"][0]["text"], json!("be nice\nbe brief"));
        // user, model(tool_call), tool(functionResponse)
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], json!("user"));
        assert_eq!(contents[1]["role"], json!("model"));
        assert_eq!(
            contents[1]["parts"][0]["functionCall"]["name"],
            json!("tool_a")
        );
        assert_eq!(contents[2]["role"], json!("user"));
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["name"],
            json!("tool_a")
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["result"],
            json!(42)
        );
    }

    #[test]
    fn test_translate_tool_choice() {
        assert_eq!(
            translate_tool_choice_to_gemini(&json!("auto")).unwrap(),
            json!({"functionCallingConfig": {"mode": "AUTO"}})
        );
        assert_eq!(
            translate_tool_choice_to_gemini(&json!("required")).unwrap(),
            json!({"functionCallingConfig": {"mode": "ANY"}})
        );
        assert_eq!(
            translate_tool_choice_to_gemini(&json!("none")).unwrap(),
            json!({"functionCallingConfig": {"mode": "NONE"}})
        );
        assert_eq!(
            translate_tool_choice_to_gemini(&json!({"function": {"name": "f"}})).unwrap(),
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}})
        );
        assert!(translate_tool_choice_to_gemini(&Value::Null).is_none());
    }

    #[test]
    fn test_normalize_thinking_config() {
        let cfg = json!({"thinking_budget": 1024, "include_thoughts": true, "thinking_level": "HIGH"});
        let n = normalize_thinking_config(&cfg).unwrap();
        assert_eq!(n["thinkingBudget"], json!(1024));
        assert_eq!(n["includeThoughts"], json!(true));
        assert_eq!(n["thinkingLevel"], json!("high"));
        assert!(normalize_thinking_config(&json!({})).is_none());
    }

    #[test]
    fn test_build_request_generation_config() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = GeminiRequestParams {
            messages: &messages,
            temperature: Some(0.5),
            max_tokens: Some(100),
            top_p: Some(0.9),
            stop: json!(["END"]),
            ..Default::default()
        };
        let req = build_gemini_request(&params);
        let gc = &req["generationConfig"];
        assert_eq!(gc["temperature"], json!(0.5));
        assert_eq!(gc["maxOutputTokens"], json!(100));
        assert_eq!(gc["topP"], json!(0.9));
        assert_eq!(gc["stopSequences"], json!(["END"]));
    }

    #[test]
    fn test_stop_string_coerced_to_list() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = GeminiRequestParams {
            messages: &messages,
            stop: json!("STOP"),
            ..Default::default()
        };
        let req = build_gemini_request(&params);
        assert_eq!(req["generationConfig"]["stopSequences"], json!(["STOP"]));
    }

    #[test]
    fn test_map_finish_reason() {
        assert_eq!(map_gemini_finish_reason("STOP"), "stop");
        assert_eq!(map_gemini_finish_reason("MAX_TOKENS"), "length");
        assert_eq!(map_gemini_finish_reason("SAFETY"), "content_filter");
        assert_eq!(map_gemini_finish_reason("weird"), "stop");
    }

    #[test]
    fn test_translate_gemini_response_text() {
        let resp = json!({
            "candidates": [{
                "content": {"parts": [
                    {"thought": true, "text": "thinking..."},
                    {"text": "Hello"},
                    {"text": " world"}
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "cachedContentTokenCount": 2
            }
        });
        let out = translate_gemini_response(&resp, "gemini-2.5-flash");
        let choice = &out.choices[0];
        assert_eq!(choice.message.content.as_deref(), Some("Hello world"));
        assert_eq!(choice.message.reasoning.as_deref(), Some("thinking..."));
        assert_eq!(choice.finish_reason, "stop");
        assert_eq!(out.usage.prompt_tokens, 10);
        assert_eq!(out.usage.total_tokens, 15);
        assert_eq!(out.usage.prompt_tokens_details.cached_tokens, 2);
    }

    #[test]
    fn test_translate_gemini_response_tool_call() {
        let resp = json!({
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "fn", "args": {"x": 1}}, "thoughtSignature": "sig"}
                ]},
                "finishReason": "STOP"
            }]
        });
        let out = translate_gemini_response(&resp, "m");
        let choice = &out.choices[0];
        assert_eq!(choice.finish_reason, "tool_calls");
        let tcs = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs[0].function.name, "fn");
        assert_eq!(
            tcs[0].extra_content.as_ref().unwrap()["google"]["thought_signature"],
            json!("sig")
        );
    }

    #[test]
    fn test_empty_response() {
        let out = translate_gemini_response(&json!({"candidates": []}), "m");
        assert_eq!(out.choices[0].message.content.as_deref(), Some(""));
        assert_eq!(out.choices[0].finish_reason, "stop");
    }

    #[test]
    fn test_parse_sse_events() {
        let body = "data: {\"a\": 1}\n\ndata: {\"b\": 2}\ndata: [DONE]\ndata: {\"c\": 3}\n";
        let events = parse_sse_events(body);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["a"], json!(1));
        assert_eq!(events[1]["b"], json!(2));
    }

    #[test]
    fn test_translate_stream_event_text_and_finish() {
        let mut idx = ToolCallIndices::new();
        let event = json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 4}
        });
        let chunks = translate_stream_event(&event, "m", &mut idx);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("hi"));
        assert_eq!(chunks[1].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[1].usage.as_ref().unwrap().prompt_tokens, 3);
    }

    #[test]
    fn test_translate_stream_event_tool_call_dedup() {
        let mut idx = ToolCallIndices::new();
        // First event: partial args.
        let e1 = json!({
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "fn", "args": {"a": 1}}}
            ]}}]
        });
        let c1 = translate_stream_event(&e1, "m", &mut idx);
        assert_eq!(c1.len(), 1);
        let td1 = &c1[0].choices[0].delta.tool_calls.as_ref().unwrap()[0];
        let first_args = td1.function.arguments.clone();
        assert_eq!(first_args, "{\"a\":1}");

        // Second event: superset of args (prefix match) -> emit the delta only.
        let e2 = json!({
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "fn", "args": {"a": 1}}}
            ]}, "finishReason": "STOP"}]
        });
        let c2 = translate_stream_event(&e2, "m", &mut idx);
        // tool-call chunk + finish chunk
        let td2 = &c2[0].choices[0].delta.tool_calls.as_ref().unwrap()[0];
        // identical args -> emitted empty
        assert_eq!(td2.function.arguments, "");
        // finish maps to tool_calls because tool_call_indices non-empty
        assert_eq!(
            c2.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn test_gemini_http_error_429_free_tier() {
        let body = json!({
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for FREE_TIER requests",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "RATE_LIMIT_EXCEEDED",
                    "metadata": {"quota": "x"}
                }]
            }
        })
        .to_string();
        let err = gemini_http_error(429, &body, Some("30"));
        assert_eq!(err.code, "gemini_rate_limited");
        assert_eq!(err.status_code, Some(429));
        assert_eq!(err.retry_after, Some(30.0));
        assert!(err.message.contains("free tier"));
        assert_eq!(err.details["reason"], json!("RATE_LIMIT_EXCEEDED"));
    }

    #[test]
    fn test_gemini_http_error_401() {
        let err = gemini_http_error(401, "{\"error\": {\"message\": \"bad key\"}}", None);
        assert_eq!(err.code, "gemini_unauthorized");
        assert!(err.message.contains("bad key"));
    }

    #[test]
    fn test_gemini_http_error_no_body() {
        let err = gemini_http_error(500, "internal error text", None);
        assert_eq!(err.code, "gemini_http_500");
        assert!(err.message.contains("internal error text"));
    }

    #[test]
    fn test_client_requires_api_key() {
        assert!(GeminiNativeClient::new("", None, None, None).is_err());
        assert!(GeminiNativeClient::new("   ", None, None, None).is_err());
    }

    #[test]
    fn test_client_normalizes_openai_suffix() {
        let c = GeminiNativeClient::new(
            "k",
            Some("https://generativelanguage.googleapis.com/v1beta/openai/"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            c.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn test_dumps_sorted() {
        let v = json!({"b": 1, "a": 2, "c": {"z": 1, "y": 2}});
        assert_eq!(dumps_sorted(&v), "{\"a\":2,\"b\":1,\"c\":{\"y\":2,\"z\":1}}");
    }

    #[test]
    fn test_rand_hex12_format() {
        let id = call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 5 + 12);
        let h = chatcmpl_id();
        assert!(h.starts_with("chatcmpl-"));
    }
}
