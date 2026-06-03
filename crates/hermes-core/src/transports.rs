//! Provider transport layer: normalized response types + per-provider adapters.
//!
//! This is a faithful Rust port of the Python `agent/transports/` package,
//! merged into one flat module:
//!
//! * `types.py`     -> [`ToolCall`], [`Usage`], [`NormalizedResponse`],
//!                     [`build_tool_call`], [`map_finish_reason`]
//! * `base.py`      -> the [`Transport`] trait (shared interface + default hooks)
//! * the 3 adapters -> [`AnthropicTransport`], [`BedrockTransport`],
//!                     [`ResponsesApiTransport`]
//! * `__init__.py`  -> the [`get_transport`] dispatch/registry
//!
//! A transport owns the data path for one `api_mode`:
//! `convert_messages → convert_tools → build_kwargs → normalize_response`.
//! It does NOT own client construction, streaming, credential refresh, prompt
//! caching, interrupt handling, or retry logic — those live on the agent.
//!
//! NOTE: `convert_messages` / `convert_tools` / `normalize_response` in the
//! Python originals delegate to the per-provider `*_adapter` modules
//! (`anthropic_adapter`, `bedrock_adapter`, `codex_responses_adapter`). Those
//! adapters are ported separately; here those trait methods are documented
//! hooks. All *self-contained* logic (stop-reason maps, structural validation,
//! cache-stat extraction, and the Codex `build_kwargs` payload assembly) is
//! ported natively and faithfully.

use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// types.py  — normalized response types
// ---------------------------------------------------------------------------

/// A normalized tool call from any provider.
///
/// `id` is the protocol's canonical identifier — what gets used in
/// `tool_call_id` / `tool_use_id` when constructing tool-result messages. May
/// be `None` when the provider omits it; the agent fills it via a deterministic
/// call id before storing in history.
///
/// `provider_data` carries per-tool-call protocol metadata that only
/// protocol-aware code reads:
/// * Codex:  `{"call_id": "call_XXX", "response_item_id": "fc_XXX"}`
/// * Gemini: `{"extra_content": {"google": {"thought_signature": "..."}}}`
/// * Others: `None`
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    /// JSON string of arguments.
    pub arguments: String,
    pub provider_data: Option<Map<String, Value>>,
}

impl ToolCall {
    pub fn new(id: Option<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        ToolCall {
            id,
            name: name.into(),
            arguments: arguments.into(),
            provider_data: None,
        }
    }

    /// Backward-compat: `tc.type` was always `"function"`.
    pub fn call_type(&self) -> &'static str {
        "function"
    }

    fn pd_str(&self, key: &str) -> Option<String> {
        self.provider_data
            .as_ref()
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Codex `call_id` from `provider_data`.
    pub fn call_id(&self) -> Option<String> {
        self.pd_str("call_id")
    }

    /// Codex `response_item_id` from `provider_data`.
    pub fn response_item_id(&self) -> Option<String> {
        self.pd_str("response_item_id")
    }

    /// Gemini `extra_content` (thought_signature) from `provider_data`.
    pub fn extra_content(&self) -> Option<Value> {
        self.provider_data
            .as_ref()
            .and_then(|m| m.get("extra_content"))
            .cloned()
    }
}

/// Token usage from an API response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub cached_tokens: i64,
}

/// Normalized API response from any provider.
///
/// Shared fields are truly cross-provider — every caller can rely on them
/// without branching on `api_mode`. Protocol-specific state goes in
/// `provider_data` so only protocol-aware code paths read it.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    /// One of "stop", "tool_calls", "length", "content_filter".
    pub finish_reason: String,
    pub reasoning: Option<String>,
    pub usage: Option<Usage>,
    pub provider_data: Option<Map<String, Value>>,
}

impl NormalizedResponse {
    fn pd_get(&self, key: &str) -> Option<&Value> {
        self.provider_data.as_ref().and_then(|m| m.get(key))
    }

    /// Backward-compat accessor mapped from `provider_data`.
    pub fn reasoning_content(&self) -> Option<String> {
        self.pd_get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Backward-compat accessor mapped from `provider_data`.
    pub fn reasoning_details(&self) -> Option<Value> {
        self.pd_get("reasoning_details").cloned()
    }

    /// Backward-compat accessor mapped from `provider_data`.
    pub fn codex_reasoning_items(&self) -> Option<Value> {
        self.pd_get("codex_reasoning_items").cloned()
    }

    /// Backward-compat accessor mapped from `provider_data`.
    pub fn codex_message_items(&self) -> Option<Value> {
        self.pd_get("codex_message_items").cloned()
    }
}

// ---------------------------------------------------------------------------
// Factory helpers (types.py)
// ---------------------------------------------------------------------------

/// Build a [`ToolCall`], auto-serialising `arguments` if it's a JSON object.
///
/// Mirrors Python's `build_tool_call`: a dict argument is `json.dumps`-ed; any
/// other value is stringified. Extra `provider_fields` are collected into
/// `provider_data` (empty -> `None`).
pub fn build_tool_call(
    id: Option<String>,
    name: impl Into<String>,
    arguments: &Value,
    provider_fields: Map<String, Value>,
) -> ToolCall {
    let args_str = match arguments {
        Value::Object(_) => serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            // Python str(True) == "True"
            if *b { "True".to_string() } else { "False".to_string() }
        }
        other => other.to_string(),
    };
    let pd = if provider_fields.is_empty() {
        None
    } else {
        Some(provider_fields)
    };
    ToolCall {
        id,
        name: name.into(),
        arguments: args_str,
        provider_data: pd,
    }
}

/// Translate a provider-specific stop reason to the normalised set.
///
/// Falls back to `"stop"` for unknown or `None` reasons.
pub fn map_finish_reason(reason: Option<&str>, mapping: &[(&str, &str)]) -> String {
    match reason {
        None => "stop".to_string(),
        Some(r) => mapping
            .iter()
            .find(|(k, _)| *k == r)
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| "stop".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Canonical stop-reason maps (one per provider, promoted to module level)
// ---------------------------------------------------------------------------

/// Anthropic `stop_reason` -> OpenAI `finish_reason`.
pub const ANTHROPIC_STOP_REASON_MAP: &[(&str, &str)] = &[
    ("end_turn", "stop"),
    ("tool_use", "tool_calls"),
    ("max_tokens", "length"),
    ("stop_sequence", "stop"),
    ("refusal", "content_filter"),
    ("model_context_window_exceeded", "length"),
];

/// Bedrock stop reason -> OpenAI `finish_reason`.
pub const BEDROCK_STOP_REASON_MAP: &[(&str, &str)] = &[
    ("end_turn", "stop"),
    ("tool_use", "tool_calls"),
    ("max_tokens", "length"),
    ("stop_sequence", "stop"),
    ("guardrail_intervened", "content_filter"),
    ("content_filtered", "content_filter"),
];

/// Codex `response.status` -> OpenAI `finish_reason`. The caller must check
/// `incomplete_details.reason` separately for `max_output_tokens`.
pub const CODEX_STATUS_MAP: &[(&str, &str)] = &[
    ("completed", "stop"),
    ("incomplete", "length"),
    ("failed", "stop"),
    ("cancelled", "stop"),
];

const MCP_PREFIX: &str = "mcp_";

// ---------------------------------------------------------------------------
// base.py  — the Transport trait
// ---------------------------------------------------------------------------

/// Base interface for provider-specific format conversion and normalization.
///
/// Required methods mirror the Python `@abstractmethod` set; the rest provide
/// the same defaults as the ABC.
pub trait Transport {
    /// The `api_mode` string this transport handles (e.g. `"anthropic_messages"`).
    fn api_mode(&self) -> &'static str;

    /// Convert OpenAI-format messages to provider-native format.
    ///
    /// In the Python original this delegates to the per-provider `*_adapter`
    /// module (out of scope here); the native build wires the adapter port.
    fn convert_messages(&self, messages: &Value, opts: &Value) -> Value;

    /// Convert OpenAI-format tool definitions to provider-native format.
    fn convert_tools(&self, tools: &Value) -> Value;

    /// Build the complete API-call kwargs dict.
    fn build_kwargs(
        &self,
        model: &str,
        messages: &Value,
        tools: Option<&Value>,
        params: &Map<String, Value>,
    ) -> Map<String, Value>;

    /// Normalize a raw provider response to [`NormalizedResponse`].
    fn normalize_response(&self, response: &Value, opts: &Value) -> NormalizedResponse;

    /// Optional: structural validity check. Default: always valid.
    fn validate_response(&self, _response: &Value) -> bool {
        true
    }

    /// Optional: provider-specific cache hit/creation stats.
    /// Returns `(cached_tokens, creation_tokens)` or `None`. Default: `None`.
    fn extract_cache_stats(&self, _response: &Value) -> Option<(i64, i64)> {
        None
    }

    /// Optional: map provider stop reason to OpenAI equivalent.
    /// Default returns the raw reason unchanged.
    fn map_finish_reason(&self, raw_reason: &str) -> String {
        raw_reason.to_string()
    }
}

// ---------------------------------------------------------------------------
// anthropic.py  — AnthropicTransport
// ---------------------------------------------------------------------------

/// Transport for `api_mode == "anthropic_messages"`.
#[derive(Debug, Default, Clone)]
pub struct AnthropicTransport;

impl Transport for AnthropicTransport {
    fn api_mode(&self) -> &'static str {
        "anthropic_messages"
    }

    fn convert_messages(&self, messages: &Value, _opts: &Value) -> Value {
        // Delegates to anthropic_adapter::convert_messages_to_anthropic (out of scope).
        messages.clone()
    }

    fn convert_tools(&self, tools: &Value) -> Value {
        // Delegates to anthropic_adapter::convert_tools_to_anthropic (out of scope).
        tools.clone()
    }

    fn build_kwargs(
        &self,
        _model: &str,
        _messages: &Value,
        _tools: Option<&Value>,
        _params: &Map<String, Value>,
    ) -> Map<String, Value> {
        // Delegates to anthropic_adapter::build_anthropic_kwargs (out of scope).
        Map::new()
    }

    /// Normalize Anthropic content blocks (text / thinking / tool_use), map
    /// `stop_reason`, and collect `reasoning_details` in `provider_data`.
    ///
    /// `opts` may carry `{"strip_tool_prefix": true}`.
    fn normalize_response(&self, response: &Value, opts: &Value) -> NormalizedResponse {
        let strip_tool_prefix = opts
            .get("strip_tool_prefix")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut text_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut reasoning_details: Vec<Value> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        if let Some(blocks) = response.get("content").and_then(|c| c.as_array()) {
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(t.to_string());
                        }
                    }
                    Some("thinking") => {
                        if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                            reasoning_parts.push(t.to_string());
                        }
                        if block.is_object() {
                            reasoning_details.push(block.clone());
                        }
                    }
                    Some("tool_use") => {
                        let mut name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if strip_tool_prefix && name.starts_with(MCP_PREFIX) {
                            name = name[MCP_PREFIX.len()..].to_string();
                        }
                        let input = block.get("input").cloned().unwrap_or(json!({}));
                        let args = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        tool_calls.push(ToolCall::new(id, name, args));
                    }
                    _ => {}
                }
            }
        }

        let stop_reason = response.get("stop_reason").and_then(|v| v.as_str());
        let finish_reason = map_finish_reason(stop_reason, ANTHROPIC_STOP_REASON_MAP);

        let provider_data = if reasoning_details.is_empty() {
            None
        } else {
            let mut m = Map::new();
            m.insert("reasoning_details".to_string(), Value::Array(reasoning_details));
            Some(m)
        };

        NormalizedResponse {
            content: if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join("\n"))
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            finish_reason,
            reasoning: if reasoning_parts.is_empty() {
                None
            } else {
                Some(reasoning_parts.join("\n\n"))
            },
            usage: None,
            provider_data,
        }
    }

    /// An empty content list is legitimate only when `stop_reason == "end_turn"`.
    fn validate_response(&self, response: &Value) -> bool {
        if response.is_null() {
            return false;
        }
        let content = match response.get("content") {
            Some(Value::Array(a)) => a,
            _ => return false,
        };
        if content.is_empty() {
            return response.get("stop_reason").and_then(|v| v.as_str()) == Some("end_turn");
        }
        true
    }

    fn extract_cache_stats(&self, response: &Value) -> Option<(i64, i64)> {
        let usage = response.get("usage")?;
        let cached = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let written = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if cached != 0 || written != 0 {
            Some((cached, written))
        } else {
            None
        }
    }

    fn map_finish_reason(&self, raw_reason: &str) -> String {
        map_finish_reason(Some(raw_reason), ANTHROPIC_STOP_REASON_MAP)
    }
}

// ---------------------------------------------------------------------------
// bedrock.py  — BedrockTransport
// ---------------------------------------------------------------------------

/// Transport for `api_mode == "bedrock_converse"`.
#[derive(Debug, Default, Clone)]
pub struct BedrockTransport;

impl Transport for BedrockTransport {
    fn api_mode(&self) -> &'static str {
        "bedrock_converse"
    }

    fn convert_messages(&self, messages: &Value, _opts: &Value) -> Value {
        // Delegates to bedrock_adapter::convert_messages_to_converse (out of scope).
        messages.clone()
    }

    fn convert_tools(&self, tools: &Value) -> Value {
        // Delegates to bedrock_adapter::convert_tools_to_converse (out of scope).
        tools.clone()
    }

    /// Build Bedrock `converse()` kwargs. The body delegates to
    /// `bedrock_adapter::build_converse_kwargs` (out of scope); the sentinel
    /// dispatch keys are added here faithfully.
    fn build_kwargs(
        &self,
        _model: &str,
        _messages: &Value,
        _tools: Option<&Value>,
        params: &Map<String, Value>,
    ) -> Map<String, Value> {
        let region = params
            .get("region")
            .and_then(|v| v.as_str())
            .unwrap_or("us-east-1")
            .to_string();
        let mut kwargs = Map::new();
        // Sentinel keys for dispatch — agent pops these before the boto3 call.
        kwargs.insert("__bedrock_converse__".to_string(), Value::Bool(true));
        kwargs.insert("__bedrock_region__".to_string(), Value::String(region));
        kwargs
    }

    /// Normalize a Bedrock response. Expects an already-normalized
    /// OpenAI-compatible shape with `choices[0].message`.
    fn normalize_response(&self, response: &Value, _opts: &Value) -> NormalizedResponse {
        let choice = response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first());
        let msg = choice.and_then(|c| c.get("message"));

        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .unwrap_or("stop")
            .to_string();

        let tool_calls = msg
            .and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array())
            .filter(|a| !a.is_empty())
            .map(|arr| {
                arr.iter()
                    .map(|tc| {
                        let func = tc.get("function");
                        ToolCall::new(
                            tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
                            func.and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or(""),
                            func.and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}"),
                        )
                    })
                    .collect::<Vec<_>>()
            });

        let usage = msg
            .and_then(|_| response.get("usage"))
            .filter(|u| !u.is_null())
            .map(|u| Usage {
                prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                completion_tokens: u
                    .get("completion_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                cached_tokens: 0,
            });

        let reasoning = msg.and_then(|m| {
            m.get("reasoning")
                .and_then(|v| v.as_str())
                .or_else(|| m.get("reasoning_content").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
        });

        NormalizedResponse {
            content: msg
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            tool_calls,
            finish_reason,
            reasoning,
            usage,
            provider_data: None,
        }
    }

    /// Raw Bedrock dict needs an `"output"` key; an already-normalized shape
    /// needs non-empty `choices`.
    fn validate_response(&self, response: &Value) -> bool {
        if response.is_null() {
            return false;
        }
        // Already-normalized shape: distinguish by presence of "choices".
        if let Some(choices) = response.get("choices") {
            return choices.as_array().map(|a| !a.is_empty()).unwrap_or(false);
        }
        // Raw Bedrock dict response — check for "output" key.
        if response.is_object() {
            return response.get("output").is_some();
        }
        false
    }

    fn map_finish_reason(&self, raw_reason: &str) -> String {
        map_finish_reason(Some(raw_reason), BEDROCK_STOP_REASON_MAP)
    }
}

// ---------------------------------------------------------------------------
// codex.py  — ResponsesApiTransport
// ---------------------------------------------------------------------------

/// Transport for `api_mode == "codex_responses"` (OpenAI Responses API).
#[derive(Debug, Default, Clone)]
pub struct ResponsesApiTransport;

/// Default system identity used when no instructions are resolvable. Mirrors
/// `run_agent.DEFAULT_AGENT_IDENTITY` reference in the Python original.
pub const DEFAULT_AGENT_IDENTITY: &str = "You are a helpful assistant.";

impl ResponsesApiTransport {
    /// Helper: merge string headers into an existing `extra_headers` object,
    /// preserving existing entries and dropping null values — matching the
    /// Python comprehension `{str(k): str(v) ... if k and v is not None}`.
    fn merge_extra_headers(existing: Option<&Value>, additions: &[(&str, &str)]) -> Value {
        let mut merged = Map::new();
        if let Some(Value::Object(m)) = existing {
            for (k, v) in m {
                if !k.is_empty() && !v.is_null() {
                    let vs = match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    merged.insert(k.clone(), Value::String(vs));
                }
            }
        }
        for (k, v) in additions {
            merged.insert(k.to_string(), Value::String(v.to_string()));
        }
        Value::Object(merged)
    }
}

impl Transport for ResponsesApiTransport {
    fn api_mode(&self) -> &'static str {
        "codex_responses"
    }

    fn convert_messages(&self, messages: &Value, _opts: &Value) -> Value {
        // Delegates to codex_responses_adapter::_chat_messages_to_responses_input.
        messages.clone()
    }

    fn convert_tools(&self, tools: &Value) -> Value {
        // Delegates to codex_responses_adapter::_responses_tools.
        tools.clone()
    }

    /// Build Responses API kwargs. This is the one adapter method with real
    /// branching logic, ported faithfully. `messages` must be a JSON array;
    /// the system message (if any, and no explicit `instructions`) is hoisted
    /// out as `instructions` and dropped from the input payload. The actual
    /// `input`/`tools` conversion is left to the codex adapter port.
    fn build_kwargs(
        &self,
        model: &str,
        messages: &Value,
        tools: Option<&Value>,
        params: &Map<String, Value>,
    ) -> Map<String, Value> {
        let msg_arr = messages.as_array().cloned().unwrap_or_default();

        let mut instructions = params
            .get("instructions")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut payload_messages = msg_arr.clone();
        if instructions.is_empty() {
            if let Some(first) = msg_arr.first() {
                if first.get("role").and_then(|v| v.as_str()) == Some("system") {
                    instructions = first
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    payload_messages = msg_arr[1..].to_vec();
                }
            }
        }
        if instructions.is_empty() {
            instructions = DEFAULT_AGENT_IDENTITY.to_string();
        }

        let is_github_responses = params
            .get("is_github_responses")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let is_codex_backend = params
            .get("is_codex_backend")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let is_xai_responses = params
            .get("is_xai_responses")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Resolve reasoning effort.
        let mut reasoning_effort = "medium".to_string();
        let mut reasoning_enabled = true;
        if let Some(rc) = params.get("reasoning_config").and_then(|v| v.as_object()) {
            if rc.get("enabled") == Some(&Value::Bool(false)) {
                reasoning_enabled = false;
            } else if let Some(eff) = rc.get("effort").and_then(|v| v.as_str()) {
                if !eff.is_empty() {
                    reasoning_effort = eff.to_string();
                }
            }
        }
        // _effort_clamp = {"minimal": "low"}
        if reasoning_effort == "minimal" {
            reasoning_effort = "low".to_string();
        }

        let mut kwargs = Map::new();
        kwargs.insert("model".to_string(), Value::String(model.to_string()));
        kwargs.insert("instructions".to_string(), Value::String(instructions));
        // input/tools conversion delegated to the codex adapter port.
        kwargs.insert("input".to_string(), Value::Array(payload_messages));
        kwargs.insert(
            "tools".to_string(),
            tools.cloned().unwrap_or(Value::Array(vec![])),
        );
        kwargs.insert("tool_choice".to_string(), Value::String("auto".into()));
        kwargs.insert("parallel_tool_calls".to_string(), Value::Bool(true));
        kwargs.insert("store".to_string(), Value::Bool(false));

        let session_id = params
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if !is_github_responses {
            if let Some(sid) = &session_id {
                kwargs.insert("prompt_cache_key".to_string(), Value::String(sid.clone()));
            }
        }

        if reasoning_enabled && is_xai_responses {
            kwargs.insert(
                "include".to_string(),
                json!(["reasoning.encrypted_content"]),
            );
        } else if reasoning_enabled {
            if is_github_responses {
                if let Some(gr) = params.get("github_reasoning_extra") {
                    if !gr.is_null() {
                        kwargs.insert("reasoning".to_string(), gr.clone());
                    }
                }
            } else {
                kwargs.insert(
                    "reasoning".to_string(),
                    json!({"effort": reasoning_effort, "summary": "auto"}),
                );
                kwargs.insert(
                    "include".to_string(),
                    json!(["reasoning.encrypted_content"]),
                );
            }
        } else if !is_github_responses && !is_xai_responses {
            kwargs.insert("include".to_string(), Value::Array(vec![]));
        }

        if let Some(overrides) = params.get("request_overrides").and_then(|v| v.as_object()) {
            for (k, v) in overrides {
                kwargs.insert(k.clone(), v.clone());
            }
        }

        if is_codex_backend {
            let prompt_cache_key = kwargs
                .get("prompt_cache_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let cache_scope_id = prompt_cache_key
                .or_else(|| session_id.clone())
                .unwrap_or_default();
            let cache_scope_id = cache_scope_id.trim().to_string();
            if !cache_scope_id.is_empty() {
                let merged = Self::merge_extra_headers(
                    kwargs.get("extra_headers"),
                    &[
                        ("session_id", &cache_scope_id),
                        ("x-client-request-id", &cache_scope_id),
                    ],
                );
                kwargs.insert("extra_headers".to_string(), merged);
            }
        }

        if let Some(max_tokens) = params.get("max_tokens") {
            if !max_tokens.is_null() && !is_codex_backend {
                kwargs.insert("max_output_tokens".to_string(), max_tokens.clone());
            }
        }

        if is_xai_responses {
            if let Some(sid) = &session_id {
                let merged =
                    Self::merge_extra_headers(kwargs.get("extra_headers"), &[("x-grok-conv-id", sid)]);
                kwargs.insert("extra_headers".to_string(), merged);
            }
        }

        kwargs
    }

    /// Normalize a Codex Responses API response. The heavy lifting
    /// (`_normalize_codex_response`) lives in the codex adapter port; here we
    /// faithfully assemble from an already-normalized `{message, finish_reason}`
    /// intermediate shape when present.
    fn normalize_response(&self, response: &Value, _opts: &Value) -> NormalizedResponse {
        let msg = response.get("message");
        let finish_reason = response
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("stop")
            .to_string();

        let tool_calls = msg
            .and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array())
            .filter(|a| !a.is_empty())
            .map(|arr| {
                arr.iter()
                    .map(|tc| {
                        let func = tc.get("function");
                        let mut pd = Map::new();
                        if let Some(cid) = tc.get("call_id").and_then(|v| v.as_str()) {
                            if !cid.is_empty() {
                                pd.insert("call_id".into(), Value::String(cid.into()));
                            }
                        }
                        if let Some(rid) = tc.get("response_item_id").and_then(|v| v.as_str()) {
                            if !rid.is_empty() {
                                pd.insert("response_item_id".into(), Value::String(rid.into()));
                            }
                        }
                        let id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                func.and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            });
                        let name = func
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .or_else(|| tc.get("name").and_then(|v| v.as_str()))
                            .unwrap_or("")
                            .to_string();
                        let arguments = func
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .or_else(|| tc.get("arguments").and_then(|v| v.as_str()))
                            .unwrap_or("{}")
                            .to_string();
                        ToolCall {
                            id,
                            name,
                            arguments,
                            provider_data: if pd.is_empty() { None } else { Some(pd) },
                        }
                    })
                    .collect::<Vec<_>>()
            });

        let mut provider_data = Map::new();
        if let Some(m) = msg {
            for key in [
                "codex_reasoning_items",
                "codex_message_items",
                "reasoning_details",
            ] {
                if let Some(v) = m.get(key) {
                    if !v.is_null() && !(v.is_array() && v.as_array().unwrap().is_empty()) {
                        provider_data.insert(key.to_string(), v.clone());
                    }
                }
            }
        }

        NormalizedResponse {
            content: msg
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            tool_calls,
            finish_reason,
            reasoning: msg
                .and_then(|m| m.get("reasoning"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            usage: None,
            provider_data: if provider_data.is_empty() {
                None
            } else {
                Some(provider_data)
            },
        }
    }

    /// True only if `response.output` is a non-empty list.
    fn validate_response(&self, response: &Value) -> bool {
        if response.is_null() {
            return false;
        }
        response
            .get("output")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    }

    fn map_finish_reason(&self, raw_reason: &str) -> String {
        map_finish_reason(Some(raw_reason), CODEX_STATUS_MAP)
    }
}

// ---------------------------------------------------------------------------
// __init__.py  — registry / dispatch
// ---------------------------------------------------------------------------

/// Register a transport class for an `api_mode`.
///
/// Retained for API parity with the Python registry. The native build resolves
/// transports by static dispatch in [`get_transport`], so this is a no-op stub
/// that exists only so call sites that *registered* a transport still compile.
pub fn register_transport(_api_mode: &str, _ctor: fn() -> Box<dyn Transport>) {
    // Native dispatch is static; nothing to record.
}

/// Get a transport instance for the given `api_mode`.
///
/// Returns `None` if no transport is known for this `api_mode`, allowing
/// gradual migration — call sites can check for `None` and fall back to a
/// legacy code path. `chat_completions` has no transport in scope and resolves
/// to `None`, matching the Python discovery which tolerates a missing module.
pub fn get_transport(api_mode: &str) -> Option<Box<dyn Transport>> {
    match api_mode {
        "anthropic_messages" => Some(Box::new(AnthropicTransport)),
        "bedrock_converse" => Some(Box::new(BedrockTransport)),
        "codex_responses" => Some(Box::new(ResponsesApiTransport)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_finish_reason_fallbacks_to_stop() {
        assert_eq!(map_finish_reason(None, ANTHROPIC_STOP_REASON_MAP), "stop");
        assert_eq!(
            map_finish_reason(Some("totally_unknown"), ANTHROPIC_STOP_REASON_MAP),
            "stop"
        );
    }

    #[test]
    fn anthropic_finish_reason_map() {
        let t = AnthropicTransport;
        assert_eq!(t.map_finish_reason("end_turn"), "stop");
        assert_eq!(t.map_finish_reason("tool_use"), "tool_calls");
        assert_eq!(t.map_finish_reason("max_tokens"), "length");
        assert_eq!(t.map_finish_reason("refusal"), "content_filter");
        assert_eq!(t.map_finish_reason("model_context_window_exceeded"), "length");
        assert_eq!(t.map_finish_reason("???"), "stop");
    }

    #[test]
    fn bedrock_finish_reason_map() {
        let t = BedrockTransport;
        assert_eq!(t.map_finish_reason("guardrail_intervened"), "content_filter");
        assert_eq!(t.map_finish_reason("content_filtered"), "content_filter");
        assert_eq!(t.map_finish_reason("tool_use"), "tool_calls");
        assert_eq!(t.map_finish_reason("end_turn"), "stop");
    }

    #[test]
    fn codex_status_map() {
        let t = ResponsesApiTransport;
        assert_eq!(t.map_finish_reason("completed"), "stop");
        assert_eq!(t.map_finish_reason("incomplete"), "length");
        assert_eq!(t.map_finish_reason("failed"), "stop");
        assert_eq!(t.map_finish_reason("cancelled"), "stop");
        assert_eq!(t.map_finish_reason("weird"), "stop");
    }

    #[test]
    fn build_tool_call_serializes_dict_and_collects_provider_data() {
        let mut pd = Map::new();
        pd.insert("call_id".into(), Value::String("call_1".into()));
        let tc = build_tool_call(
            Some("id1".into()),
            "do_thing",
            &json!({"a": 1, "b": "x"}),
            pd,
        );
        assert_eq!(tc.id.as_deref(), Some("id1"));
        assert_eq!(tc.name, "do_thing");
        // arguments must be a JSON string
        let parsed: Value = serde_json::from_str(&tc.arguments).unwrap();
        assert_eq!(parsed, json!({"a": 1, "b": "x"}));
        assert_eq!(tc.call_id().as_deref(), Some("call_1"));
        assert_eq!(tc.call_type(), "function");
    }

    #[test]
    fn build_tool_call_string_arg_passthrough_and_empty_pd() {
        let tc = build_tool_call(None, "f", &Value::String("{\"x\":1}".into()), Map::new());
        assert_eq!(tc.arguments, "{\"x\":1}");
        assert!(tc.provider_data.is_none());
        assert!(tc.call_id().is_none());
    }

    #[test]
    fn toolcall_accessors_read_provider_data() {
        let mut pd = Map::new();
        pd.insert("response_item_id".into(), Value::String("fc_1".into()));
        pd.insert("extra_content".into(), json!({"google": {"sig": "z"}}));
        let tc = ToolCall {
            id: None,
            name: "n".into(),
            arguments: "{}".into(),
            provider_data: Some(pd),
        };
        assert_eq!(tc.response_item_id().as_deref(), Some("fc_1"));
        assert_eq!(tc.extra_content().unwrap(), json!({"google": {"sig": "z"}}));
    }

    #[test]
    fn normalized_response_backcompat_accessors() {
        let mut pd = Map::new();
        pd.insert("reasoning_content".into(), Value::String("thinking".into()));
        pd.insert("codex_reasoning_items".into(), json!([{"x": 1}]));
        let nr = NormalizedResponse {
            content: None,
            tool_calls: None,
            finish_reason: "stop".into(),
            reasoning: None,
            usage: None,
            provider_data: Some(pd),
        };
        assert_eq!(nr.reasoning_content().as_deref(), Some("thinking"));
        assert_eq!(nr.codex_reasoning_items().unwrap(), json!([{"x": 1}]));
        assert!(nr.reasoning_details().is_none());
    }

    #[test]
    fn anthropic_normalize_full() {
        let t = AnthropicTransport;
        let resp = json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "hmm"},
                {"type": "tool_use", "id": "tu_1", "name": "mcp_search", "input": {"q": "rust"}},
            ],
            "stop_reason": "tool_use",
        });
        let opts = json!({"strip_tool_prefix": true});
        let nr = t.normalize_response(&resp, &opts);
        assert_eq!(nr.content.as_deref(), Some("hello"));
        assert_eq!(nr.reasoning.as_deref(), Some("hmm"));
        assert_eq!(nr.finish_reason, "tool_calls");
        let tcs = nr.tool_calls.clone().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "search"); // mcp_ prefix stripped
        assert_eq!(tcs[0].id.as_deref(), Some("tu_1"));
        // reasoning_details captured
        assert!(nr.reasoning_details().is_some());
    }

    #[test]
    fn anthropic_validate_empty_content() {
        let t = AnthropicTransport;
        // empty content + end_turn => valid
        assert!(t.validate_response(&json!({"content": [], "stop_reason": "end_turn"})));
        // empty content + other reason => invalid
        assert!(!t.validate_response(&json!({"content": [], "stop_reason": "tool_use"})));
        // non-list content => invalid
        assert!(!t.validate_response(&json!({"content": "x"})));
        // null => invalid
        assert!(!t.validate_response(&Value::Null));
        // non-empty => valid
        assert!(t.validate_response(&json!({"content": [{"type": "text", "text": "y"}]})));
    }

    #[test]
    fn anthropic_cache_stats() {
        let t = AnthropicTransport;
        let resp = json!({"usage": {"cache_read_input_tokens": 5, "cache_creation_input_tokens": 3}});
        assert_eq!(t.extract_cache_stats(&resp), Some((5, 3)));
        let none = json!({"usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}});
        assert_eq!(t.extract_cache_stats(&none), None);
        assert_eq!(t.extract_cache_stats(&json!({})), None);
    }

    #[test]
    fn bedrock_validate_shapes() {
        let t = BedrockTransport;
        assert!(t.validate_response(&json!({"output": {"message": {}}})));
        assert!(!t.validate_response(&json!({"no_output": 1})));
        assert!(t.validate_response(&json!({"choices": [{"message": {}}]})));
        assert!(!t.validate_response(&json!({"choices": []})));
        assert!(!t.validate_response(&Value::Null));
    }

    #[test]
    fn bedrock_build_kwargs_sentinels() {
        let t = BedrockTransport;
        let mut params = Map::new();
        params.insert("region".into(), Value::String("eu-west-1".into()));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        assert_eq!(kw.get("__bedrock_converse__"), Some(&Value::Bool(true)));
        assert_eq!(
            kw.get("__bedrock_region__"),
            Some(&Value::String("eu-west-1".into()))
        );
        // default region
        let kw2 = t.build_kwargs("m", &json!([]), None, &Map::new());
        assert_eq!(
            kw2.get("__bedrock_region__"),
            Some(&Value::String("us-east-1".into()))
        );
    }

    #[test]
    fn codex_validate_output() {
        let t = ResponsesApiTransport;
        assert!(t.validate_response(&json!({"output": [{"type": "message"}]})));
        assert!(!t.validate_response(&json!({"output": []})));
        assert!(!t.validate_response(&json!({"output": "x"})));
        assert!(!t.validate_response(&Value::Null));
    }

    #[test]
    fn codex_build_kwargs_default_reasoning_and_cache_key() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("session_id".into(), Value::String("sess-1".into()));
        let messages = json!([
            {"role": "system", "content": "  be nice  "},
            {"role": "user", "content": "hi"},
        ]);
        let kw = t.build_kwargs("gpt-x", &messages, None, &params);
        // system hoisted + trimmed
        assert_eq!(kw.get("instructions"), Some(&Value::String("be nice".into())));
        // system dropped from input
        assert_eq!(kw.get("input").unwrap().as_array().unwrap().len(), 1);
        // prompt_cache_key set (not github)
        assert_eq!(
            kw.get("prompt_cache_key"),
            Some(&Value::String("sess-1".into()))
        );
        // default reasoning: medium + encrypted include
        assert_eq!(kw.get("reasoning"), Some(&json!({"effort": "medium", "summary": "auto"})));
        assert_eq!(kw.get("include"), Some(&json!(["reasoning.encrypted_content"])));
        assert_eq!(kw.get("store"), Some(&Value::Bool(false)));
    }

    #[test]
    fn codex_build_kwargs_effort_clamp_and_default_identity() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("reasoning_config".into(), json!({"effort": "minimal"}));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        // minimal clamped to low
        assert_eq!(kw.get("reasoning"), Some(&json!({"effort": "low", "summary": "auto"})));
        // no system, no instructions => default identity
        assert_eq!(
            kw.get("instructions"),
            Some(&Value::String(DEFAULT_AGENT_IDENTITY.into()))
        );
    }

    #[test]
    fn codex_build_kwargs_github_skips_cache_key_and_uses_extra() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("is_github_responses".into(), Value::Bool(true));
        params.insert("session_id".into(), Value::String("s".into()));
        params.insert("github_reasoning_extra".into(), json!({"effort": "high"}));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        assert!(kw.get("prompt_cache_key").is_none());
        assert_eq!(kw.get("reasoning"), Some(&json!({"effort": "high"})));
    }

    #[test]
    fn codex_build_kwargs_xai_include_and_header() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("is_xai_responses".into(), Value::Bool(true));
        params.insert("session_id".into(), Value::String("conv-9".into()));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        assert_eq!(kw.get("include"), Some(&json!(["reasoning.encrypted_content"])));
        let headers = kw.get("extra_headers").unwrap().as_object().unwrap();
        assert_eq!(
            headers.get("x-grok-conv-id"),
            Some(&Value::String("conv-9".into()))
        );
    }

    #[test]
    fn codex_build_kwargs_codex_backend_headers_and_max_tokens_gate() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("is_codex_backend".into(), Value::Bool(true));
        params.insert("session_id".into(), Value::String("cache-7".into()));
        params.insert("max_tokens".into(), json!(2048));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        // max_output_tokens suppressed for codex backend
        assert!(kw.get("max_output_tokens").is_none());
        let headers = kw.get("extra_headers").unwrap().as_object().unwrap();
        assert_eq!(
            headers.get("session_id"),
            Some(&Value::String("cache-7".into()))
        );
        assert_eq!(
            headers.get("x-client-request-id"),
            Some(&Value::String("cache-7".into()))
        );
    }

    #[test]
    fn codex_build_kwargs_max_tokens_set_when_not_codex() {
        let t = ResponsesApiTransport;
        let mut params = Map::new();
        params.insert("max_tokens".into(), json!(1234));
        let kw = t.build_kwargs("m", &json!([]), None, &params);
        assert_eq!(kw.get("max_output_tokens"), Some(&json!(1234)));
    }

    #[test]
    fn codex_normalize_with_provider_data() {
        let t = ResponsesApiTransport;
        let resp = json!({
            "message": {
                "content": "ok",
                "reasoning": "because",
                "tool_calls": [
                    {"id": "t1", "call_id": "call_1", "response_item_id": "fc_1",
                     "function": {"name": "search", "arguments": "{\"q\":1}"}},
                ],
                "codex_reasoning_items": [{"r": 1}],
            },
            "finish_reason": "stop",
        });
        let nr = t.normalize_response(&resp, &Value::Null);
        assert_eq!(nr.content.as_deref(), Some("ok"));
        assert_eq!(nr.reasoning.as_deref(), Some("because"));
        let tcs = nr.tool_calls.clone().unwrap();
        assert_eq!(tcs[0].call_id().as_deref(), Some("call_1"));
        assert_eq!(tcs[0].response_item_id().as_deref(), Some("fc_1"));
        assert_eq!(tcs[0].name, "search");
        assert_eq!(nr.codex_reasoning_items().unwrap(), json!([{"r": 1}]));
    }

    #[test]
    fn bedrock_normalize_basic() {
        let t = BedrockTransport;
        let resp = json!({
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "rc",
                    "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        });
        let nr = t.normalize_response(&resp, &Value::Null);
        assert_eq!(nr.content.as_deref(), Some("answer"));
        assert_eq!(nr.reasoning.as_deref(), Some("rc"));
        assert_eq!(nr.finish_reason, "tool_calls");
        assert_eq!(nr.tool_calls.as_ref().unwrap()[0].name, "f");
        let u = nr.usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn dispatch_get_transport() {
        assert_eq!(
            get_transport("anthropic_messages").unwrap().api_mode(),
            "anthropic_messages"
        );
        assert_eq!(
            get_transport("bedrock_converse").unwrap().api_mode(),
            "bedrock_converse"
        );
        assert_eq!(
            get_transport("codex_responses").unwrap().api_mode(),
            "codex_responses"
        );
        // unregistered (e.g. chat_completions) => None
        assert!(get_transport("chat_completions").is_none());
        assert!(get_transport("nonsense").is_none());
    }
}
