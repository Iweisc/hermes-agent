//! OpenAI Chat Completions transport.
//!
//! Faithful Rust port of `agent/transports/chat_completions.py`.
//!
//! Handles the default `api_mode` (`"chat_completions"`) used by ~16
//! OpenAI-compatible providers (OpenRouter, Nous, NVIDIA, Qwen, Ollama,
//! DeepSeek, xAI, Kimi, etc.).
//!
//! Messages and tools are already in OpenAI format — `convert_messages` and
//! `convert_tools` are near-identity. The complexity lives in `build_kwargs`,
//! which has provider-specific conditionals for max_tokens defaults, reasoning
//! configuration, temperature handling, and `extra_body` assembly.
//!
//! Cross-module references:
//! * [`crate::moonshot_schema::is_moonshot_model`] /
//!   [`crate::moonshot_schema::sanitize_moonshot_tools`]
//! * [`crate::lmstudio_reasoning::resolve_lmstudio_effort`]
//! * [`crate::transports::Transport`] (the trait this implements) and the
//!   normalized response types [`crate::transports::NormalizedResponse`],
//!   [`crate::transports::ToolCall`], [`crate::transports::Usage`].
//!
//! ## Provider-profile path
//!
//! Python's `_build_kwargs_from_profile` delegates per-provider quirks to a
//! `ProviderProfile` object (`providers/base.py`). Those profiles are ported
//! separately; here the profile contract is captured by the
//! [`ProviderProfile`] trait so callers that already hold a profile can route
//! through [`build_kwargs_from_profile`]. The flag-based legacy fallback is
//! ported in full natively in [`build_kwargs_legacy`].

use serde_json::{Map, Value};

use crate::lmstudio_reasoning::resolve_lmstudio_effort;
use crate::moonshot_schema::{is_moonshot_model, sanitize_moonshot_tools};
use crate::transports::{NormalizedResponse, ToolCall, Transport, Usage};

/// `agent/prompt_builder.py`: `DEVELOPER_ROLE_MODELS = ("gpt-5", "codex")`.
///
/// System messages are swapped to the `"developer"` role for these models.
pub const DEVELOPER_ROLE_MODELS: &[&str] = &["gpt-5", "codex"];

// ---------------------------------------------------------------------------
// Gemini thinking-config helpers
// ---------------------------------------------------------------------------

/// Translate Hermes/OpenRouter-style reasoning config to a Gemini
/// `thinkingConfig` object. Returns `None` when no config applies.
///
/// `thinking_config` is a Gemini-only request parameter. The same `gemini`
/// provider also serves Gemma; those reject the field with HTTP 400, so the
/// field is omitted entirely on non-Gemini models. (#17426)
pub fn build_gemini_thinking_config(
    model: &str,
    reasoning_config: Option<&Value>,
) -> Option<Map<String, Value>> {
    // `if reasoning_config is None or not isinstance(reasoning_config, dict)`
    let cfg = match reasoning_config {
        Some(Value::Object(m)) => m,
        _ => return None,
    };

    let mut normalized_model = model.trim().to_lowercase();
    if let Some(stripped) = normalized_model.strip_prefix("google/") {
        normalized_model = stripped.to_string();
    }

    if !normalized_model.starts_with("gemini") {
        return None;
    }

    // `reasoning_config.get("enabled") is False`
    if cfg.get("enabled") == Some(&Value::Bool(false)) {
        let mut m = Map::new();
        m.insert("includeThoughts".to_string(), Value::Bool(false));
        return Some(m);
    }

    // `str(reasoning_config.get("effort", "medium") or "medium").strip().lower()`
    let mut effort = effort_str(cfg.get("effort"), "medium");
    if effort == "none" {
        let mut m = Map::new();
        m.insert("includeThoughts".to_string(), Value::Bool(false));
        return Some(m);
    }

    let mut thinking_config = Map::new();
    thinking_config.insert("includeThoughts".to_string(), Value::Bool(true));

    // Gemini 2.5 accepts thinkingBudget; don't guess. `includeThoughts` alone
    // is enough to surface thought parts.
    if normalized_model.starts_with("gemini-2.5-") {
        return Some(thinking_config);
    }

    // `if effort not in {minimal, low, medium, high, xhigh}: effort = "medium"`
    if !matches!(
        effort.as_str(),
        "minimal" | "low" | "medium" | "high" | "xhigh"
    ) {
        effort = "medium".to_string();
    }

    // Gemini 3 Flash documents low/medium/high; Gemini 3 Pro is stricter
    // (low/high). Clamp Hermes' wider effort set to what each family accepts.
    if normalized_model.starts_with("gemini-3") || normalized_model.starts_with("gemini-3.1") {
        if normalized_model.contains("flash") {
            let level = match effort.as_str() {
                "minimal" | "low" => "low",
                "high" | "xhigh" => "high",
                _ => "medium",
            };
            thinking_config.insert("thinkingLevel".to_string(), Value::String(level.to_string()));
        } else if normalized_model.contains("pro") {
            let level = if matches!(effort.as_str(), "high" | "xhigh") {
                "high"
            } else {
                "low"
            };
            thinking_config.insert("thinkingLevel".to_string(), Value::String(level.to_string()));
        }
    }

    Some(thinking_config)
}

/// Convert Gemini thinking-config keys to the OpenAI-compat field names.
///
/// Returns `None` when nothing translated (matching Python's `or None`).
pub fn snake_case_gemini_thinking_config(
    config: Option<&Map<String, Value>>,
) -> Option<Map<String, Value>> {
    // `if not isinstance(config, dict) or not config`
    let config = match config {
        Some(c) if !c.is_empty() => c,
        _ => return None,
    };

    let mut translated = Map::new();
    if let Some(Value::Bool(b)) = config.get("includeThoughts") {
        translated.insert("include_thoughts".to_string(), Value::Bool(*b));
    }
    if let Some(Value::String(s)) = config.get("thinkingLevel") {
        if !s.trim().is_empty() {
            translated.insert(
                "thinking_level".to_string(),
                Value::String(s.trim().to_lowercase()),
            );
        }
    }
    // `isinstance(config.get("thinkingBudget"), (int, float))` -> int(...)
    if let Some(v) = config.get("thinkingBudget") {
        if let Some(n) = number_as_i64(v) {
            translated.insert("thinking_budget".to_string(), Value::from(n));
        }
    }

    if translated.is_empty() {
        None
    } else {
        Some(translated)
    }
}

/// Whether `base_url` is a Gemini OpenAI-compat base
/// (`...generativelanguage.googleapis.com/.../openai`).
pub fn is_gemini_openai_compat_base_url(base_url: Option<&Value>) -> bool {
    let raw = value_to_str(base_url);
    let normalized = raw
        .trim()
        .trim_end_matches('/')
        .to_lowercase();
    if normalized.is_empty() {
        return false;
    }
    if !normalized.contains("generativelanguage.googleapis.com") {
        return false;
    }
    normalized.ends_with("/openai")
}

// ---------------------------------------------------------------------------
// Provider profile abstraction (providers/base.py)
// ---------------------------------------------------------------------------

/// Sentinel mirroring `providers.base.OMIT_TEMPERATURE`.
///
/// A profile whose `fixed_temperature()` returns
/// [`FixedTemperature::Omit`] requests that the `temperature` field be left
/// out of the request entirely (distinct from "no fixed value" which falls
/// back to the caller's `temperature`).
#[derive(Debug, Clone, PartialEq)]
pub enum FixedTemperature {
    /// `profile.fixed_temperature is OMIT_TEMPERATURE` — omit the field.
    Omit,
    /// `profile.fixed_temperature is not None` — force this value.
    Value(Value),
    /// `profile.fixed_temperature is None` — use the caller's temperature.
    None,
}

/// Contract for the per-provider profile object (`providers.base.ProviderProfile`).
///
/// The concrete profiles live under `providers/` in Python and are ported
/// separately; this trait lets the chat-completions transport delegate to a
/// profile without depending on those concrete types.
pub trait ProviderProfile {
    /// Profile message preprocessing (`profile.prepare_messages`).
    fn prepare_messages(&self, messages: Vec<Value>) -> Vec<Value> {
        messages
    }

    /// `profile.fixed_temperature`.
    fn fixed_temperature(&self) -> FixedTemperature {
        FixedTemperature::None
    }

    /// `profile.default_max_tokens` (falsy `0`/`None` -> `None`).
    fn default_max_tokens(&self) -> Option<i64> {
        None
    }

    /// `profile.build_api_kwargs_extras(...)` -> `(extra_body, top_level)`.
    ///
    /// Returns provider-specific extras: the first map is merged into
    /// `extra_body`, the second is merged at the top level of the kwargs dict.
    fn build_api_kwargs_extras(
        &self,
        _reasoning_config: Option<&Value>,
        _supports_reasoning: bool,
        _qwen_session_metadata: Option<&Value>,
        _model: &str,
        _ollama_num_ctx: Option<&Value>,
    ) -> (Map<String, Value>, Map<String, Value>) {
        (Map::new(), Map::new())
    }

    /// `profile.build_extra_body(...)`.
    fn build_extra_body(
        &self,
        _session_id: Option<&Value>,
        _provider_preferences: Option<&Value>,
        _model: &str,
        _base_url: Option<&Value>,
        _reasoning_config: Option<&Value>,
    ) -> Map<String, Value> {
        Map::new()
    }
}

// ---------------------------------------------------------------------------
// ChatCompletionsTransport
// ---------------------------------------------------------------------------

/// Transport for `api_mode == "chat_completions"`.
///
/// The default path for OpenAI-compatible providers.
#[derive(Debug, Default, Clone)]
pub struct ChatCompletionsTransport;

impl ChatCompletionsTransport {
    pub fn new() -> Self {
        ChatCompletionsTransport
    }

    /// `convert_messages`: messages are already OpenAI format — sanitize Codex
    /// leaks only.
    ///
    /// Strips Codex Responses API fields (`codex_reasoning_items` /
    /// `codex_message_items` on the message, `call_id` / `response_item_id` on
    /// tool_calls) that strict chat-completions providers reject with 400/422.
    ///
    /// Returns the input unchanged when no sanitization is required (so callers
    /// can cheaply detect the no-op case); otherwise a deep-cloned, scrubbed
    /// list.
    pub fn convert_messages_native(&self, messages: &[Value]) -> Vec<Value> {
        let mut needs_sanitize = false;
        'outer: for msg in messages {
            let obj = match msg {
                Value::Object(m) => m,
                _ => continue,
            };
            if obj.contains_key("codex_reasoning_items")
                || obj.contains_key("codex_message_items")
            {
                needs_sanitize = true;
                break;
            }
            if let Some(Value::Array(tcs)) = obj.get("tool_calls") {
                for tc in tcs {
                    if let Value::Object(tcm) = tc {
                        if tcm.contains_key("call_id") || tcm.contains_key("response_item_id") {
                            needs_sanitize = true;
                            break 'outer;
                        }
                    }
                }
            }
        }

        if !needs_sanitize {
            return messages.to_vec();
        }

        // `copy.deepcopy` — serde clone is a deep copy.
        let mut sanitized: Vec<Value> = messages.to_vec();
        for msg in sanitized.iter_mut() {
            let obj = match msg {
                Value::Object(m) => m,
                _ => continue,
            };
            obj.remove("codex_reasoning_items");
            obj.remove("codex_message_items");
            if let Some(Value::Array(tcs)) = obj.get_mut("tool_calls") {
                for tc in tcs.iter_mut() {
                    if let Value::Object(tcm) = tc {
                        tcm.remove("call_id");
                        tcm.remove("response_item_id");
                    }
                }
            }
        }
        sanitized
    }
}

impl Transport for ChatCompletionsTransport {
    fn api_mode(&self) -> &'static str {
        "chat_completions"
    }

    fn convert_messages(&self, messages: &Value, _opts: &Value) -> Value {
        match messages {
            Value::Array(arr) => Value::Array(self.convert_messages_native(arr)),
            other => other.clone(),
        }
    }

    /// Tools are already in OpenAI format — identity.
    fn convert_tools(&self, tools: &Value) -> Value {
        tools.clone()
    }

    fn build_kwargs(
        &self,
        model: &str,
        messages: &Value,
        tools: Option<&Value>,
        params: &Map<String, Value>,
    ) -> Map<String, Value> {
        // Codex sanitization: drop reasoning_items / call_id / response_item_id.
        let msgs: Vec<Value> = match messages {
            Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        let sanitized = self.convert_messages_native(&msgs);

        let tools_vec: Option<Vec<Value>> = match tools {
            Some(Value::Array(a)) => Some(a.clone()),
            _ => None,
        };

        // Provider profile is delegated through `build_kwargs_from_profile` by
        // callers that hold a concrete profile; the trait-object cannot be
        // passed through a `Value` param, so the in-trait path is always the
        // legacy fallback. Callers with a profile should call
        // `build_kwargs_from_profile` directly.
        build_kwargs_legacy(model, sanitized, tools_vec, params)
    }

    /// Normalize an OpenAI `ChatCompletion` (as JSON) to `NormalizedResponse`.
    fn normalize_response(&self, response: &Value, _opts: &Value) -> NormalizedResponse {
        normalize_chat_completion(response)
    }

    /// Check that the response has valid `choices`.
    fn validate_response(&self, response: &Value) -> bool {
        // `if response is None`
        if response.is_null() {
            return false;
        }
        match response.get("choices") {
            // `not hasattr` / `choices is None`
            None | Some(Value::Null) => false,
            // `if not response.choices` (empty list is falsy)
            Some(Value::Array(a)) => !a.is_empty(),
            // Any non-null, non-empty value is truthy.
            Some(_) => true,
        }
    }

    /// Extract OpenRouter/OpenAI cache stats from `prompt_tokens_details`.
    ///
    /// Returns `(cached_tokens, creation_tokens)` or `None`.
    fn extract_cache_stats(&self, response: &Value) -> Option<(i64, i64)> {
        let usage = response.get("usage")?;
        if usage.is_null() {
            return None;
        }
        let details = usage.get("prompt_tokens_details")?;
        if details.is_null() {
            return None;
        }
        let cached = details
            .get("cached_tokens")
            .and_then(number_as_i64)
            .unwrap_or(0);
        let written = details
            .get("cache_write_tokens")
            .and_then(number_as_i64)
            .unwrap_or(0);
        if cached != 0 || written != 0 {
            Some((cached, written))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// build_kwargs — legacy (flag-based) fallback
// ---------------------------------------------------------------------------

/// Legacy flag-based `build_kwargs` assembly.
///
/// Reached only when `get_provider_profile()` returned `None` (custom /
/// unregistered providers). Known providers route through
/// [`build_kwargs_from_profile`].
///
/// `sanitized` must already have had Codex fields stripped (the trait method
/// does this). `params` mirrors the Python `**params` kwargs as a JSON object.
pub fn build_kwargs_legacy(
    model: &str,
    mut sanitized: Vec<Value>,
    tools: Option<Vec<Value>>,
    params: &Map<String, Value>,
) -> Map<String, Value> {
    // Developer role swap for GPT-5/Codex models.
    let model_lower = params
        .get("model_lower")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| model.to_lowercase());
    maybe_swap_developer_role(&mut sanitized, &model_lower);

    let mut api_kwargs = Map::new();
    api_kwargs.insert("model".to_string(), Value::String(model.to_string()));
    api_kwargs.insert("messages".to_string(), Value::Array(sanitized));

    if let Some(timeout) = non_null(params.get("timeout")) {
        api_kwargs.insert("timeout".to_string(), timeout.clone());
    }

    // Tools.
    if let Some(t) = tools {
        if !t.is_empty() {
            // Moonshot/Kimi uses a stricter flavored JSON Schema.
            let t = if is_moonshot_model(Some(model)) {
                sanitize_moonshot_tools(&t)
            } else {
                t
            };
            api_kwargs.insert("tools".to_string(), Value::Array(t));
        }
    }

    // max_tokens resolution — priority: ephemeral > user > provider default.
    let has_max_fn = params
        .get("max_tokens_param_fn")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let ephemeral = non_null(params.get("ephemeral_max_output_tokens"));
    let max_tokens = non_null(params.get("max_tokens"));
    let anthropic_max_out = non_null(params.get("anthropic_max_output"));

    if let (Some(eph), true) = (ephemeral, has_max_fn) {
        merge_into(&mut api_kwargs, apply_max_tokens_fn(params, eph));
    } else if let (Some(mt), true) = (max_tokens, has_max_fn) {
        merge_into(&mut api_kwargs, apply_max_tokens_fn(params, mt));
    } else if let Some(am) = anthropic_max_out {
        api_kwargs.insert("max_tokens".to_string(), am.clone());
    }

    let is_kimi = bool_param(params, "is_kimi");
    let is_tokenhub = bool_param(params, "is_tokenhub");
    let is_lmstudio = bool_param(params, "is_lmstudio");
    let supports_reasoning = bool_param(params, "supports_reasoning");
    let reasoning_config = non_null(params.get("reasoning_config"));

    // Kimi: top-level reasoning_effort (unless thinking disabled).
    if is_kimi {
        let thinking_off = reasoning_enabled_is_false(reasoning_config);
        if !thinking_off {
            let effort = effort_from_config(reasoning_config, "medium", &["low", "medium", "high"]);
            api_kwargs.insert("reasoning_effort".to_string(), Value::String(effort));
        }
    }

    // Tencent TokenHub: top-level reasoning_effort (unless thinking disabled).
    if is_tokenhub {
        let thinking_off = reasoning_enabled_is_false(reasoning_config);
        if !thinking_off {
            let effort = effort_from_config(reasoning_config, "high", &["low", "medium", "high"]);
            api_kwargs.insert("reasoning_effort".to_string(), Value::String(effort));
        }
    }

    // LM Studio: top-level reasoning_effort, gated by supports_reasoning.
    if is_lmstudio && supports_reasoning {
        let allowed: Option<Vec<String>> = params
            .get("lmstudio_reasoning_options")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            });
        let effort = resolve_lmstudio_effort(reasoning_config, allowed.as_deref());
        if let Some(e) = effort {
            api_kwargs.insert("reasoning_effort".to_string(), Value::String(e));
        }
    }

    // extra_body assembly.
    let mut extra_body: Map<String, Value> = Map::new();

    let is_openrouter = bool_param(params, "is_openrouter");
    let is_github_models = bool_param(params, "is_github_models");
    let provider_name = params
        .get("provider_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let base_url = params.get("base_url");

    let provider_prefs = non_null(params.get("provider_preferences"));
    if let Some(pp) = provider_prefs {
        if is_truthy(Some(pp)) && is_openrouter {
            extra_body.insert("provider".to_string(), pp.clone());
        }
    }

    // Kimi extra_body.thinking.
    if is_kimi {
        let mut thinking_enabled = true;
        if let Some(Value::Object(cfg)) = reasoning_config {
            if cfg.get("enabled") == Some(&Value::Bool(false)) {
                thinking_enabled = false;
            }
        }
        let mut thinking = Map::new();
        thinking.insert(
            "type".to_string(),
            Value::String(
                if thinking_enabled { "enabled" } else { "disabled" }.to_string(),
            ),
        );
        extra_body.insert("thinking".to_string(), Value::Object(thinking));
    }

    // Reasoning. LM Studio handled above via top-level reasoning_effort.
    if supports_reasoning && !is_lmstudio {
        if is_github_models {
            if let Some(gh) = non_null(params.get("github_reasoning_extra")) {
                extra_body.insert("reasoning".to_string(), gh.clone());
            }
        } else {
            let mut reasoning = Map::new();
            reasoning.insert("enabled".to_string(), Value::Bool(true));
            reasoning.insert("effort".to_string(), Value::String("medium".to_string()));
            extra_body.insert("reasoning".to_string(), Value::Object(reasoning));
        }
    }

    if provider_name == "gemini" {
        let raw_thinking_config = build_gemini_thinking_config(model, reasoning_config);
        if is_gemini_openai_compat_base_url(base_url) {
            let thinking_config = snake_case_gemini_thinking_config(raw_thinking_config.as_ref());
            if let Some(tc) = thinking_config {
                // openai_compat_extra = extra_body.get("extra_body", {})
                let mut openai_compat_extra = match extra_body.get("extra_body") {
                    Some(Value::Object(m)) => m.clone(),
                    _ => Map::new(),
                };
                let mut google_extra = match openai_compat_extra.get("google") {
                    Some(Value::Object(m)) => m.clone(),
                    _ => Map::new(),
                };
                google_extra.insert("thinking_config".to_string(), Value::Object(tc));
                openai_compat_extra.insert("google".to_string(), Value::Object(google_extra));
                extra_body.insert("extra_body".to_string(), Value::Object(openai_compat_extra));
            }
        } else if let Some(rtc) = raw_thinking_config {
            extra_body.insert("thinking_config".to_string(), Value::Object(rtc));
        }
    } else if provider_name == "google-gemini-cli" {
        if let Some(tc) = build_gemini_thinking_config(model, reasoning_config) {
            extra_body.insert("thinking_config".to_string(), Value::Object(tc));
        }
    }

    // Merge any pre-built extra_body additions.
    if let Some(Value::Object(additions)) = non_null(params.get("extra_body_additions")) {
        for (k, v) in additions {
            extra_body.insert(k.clone(), v.clone());
        }
    }

    if !extra_body.is_empty() {
        api_kwargs.insert("extra_body".to_string(), Value::Object(extra_body));
    }

    // Request overrides last (service_tier etc.).
    if let Some(Value::Object(overrides)) = non_null(params.get("request_overrides")) {
        for (k, v) in overrides {
            api_kwargs.insert(k.clone(), v.clone());
        }
    }

    api_kwargs
}

// ---------------------------------------------------------------------------
// build_kwargs — provider-profile path
// ---------------------------------------------------------------------------

/// Build API kwargs using a [`ProviderProfile`] — single path, no legacy flags.
///
/// Mirrors `_build_kwargs_from_profile`: every quirk comes from the profile
/// object. `sanitized` must already be Codex-stripped (call
/// [`ChatCompletionsTransport::convert_messages_native`] first).
pub fn build_kwargs_from_profile(
    profile: &dyn ProviderProfile,
    model: &str,
    sanitized: Vec<Value>,
    tools: Option<Vec<Value>>,
    params: &Map<String, Value>,
) -> Map<String, Value> {
    // Message preprocessing.
    let mut sanitized = profile.prepare_messages(sanitized);

    // Developer role swap — model-name-based, applies to all providers.
    let model_lower = model.to_lowercase();
    maybe_swap_developer_role(&mut sanitized, &model_lower);

    let mut api_kwargs = Map::new();
    api_kwargs.insert("model".to_string(), Value::String(model.to_string()));
    api_kwargs.insert("messages".to_string(), Value::Array(sanitized));

    // Temperature.
    match profile.fixed_temperature() {
        FixedTemperature::Omit => {} // Don't include temperature at all.
        FixedTemperature::Value(v) => {
            api_kwargs.insert("temperature".to_string(), v);
        }
        FixedTemperature::None => {
            if let Some(temp) = non_null(params.get("temperature")) {
                api_kwargs.insert("temperature".to_string(), temp.clone());
            }
        }
    }

    // Timeout.
    if let Some(timeout) = non_null(params.get("timeout")) {
        api_kwargs.insert("timeout".to_string(), timeout.clone());
    }

    // Tools — apply Moonshot/Kimi schema sanitization regardless of path.
    if let Some(t) = tools {
        if !t.is_empty() {
            let t = if is_moonshot_model(Some(model)) {
                sanitize_moonshot_tools(&t)
            } else {
                t
            };
            api_kwargs.insert("tools".to_string(), Value::Array(t));
        }
    }

    // max_tokens resolution — priority: ephemeral > user > profile default.
    let has_max_fn = params
        .get("max_tokens_param_fn")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let ephemeral = non_null(params.get("ephemeral_max_output_tokens"));
    let user_max = non_null(params.get("max_tokens"));
    let anthropic_max = non_null(params.get("anthropic_max_output"));
    let default_max = profile.default_max_tokens().filter(|n| *n != 0);

    if let (Some(eph), true) = (ephemeral, has_max_fn) {
        merge_into(&mut api_kwargs, apply_max_tokens_fn(params, eph));
    } else if let (Some(um), true) = (user_max, has_max_fn) {
        merge_into(&mut api_kwargs, apply_max_tokens_fn(params, um));
    } else if let (Some(dm), true) = (default_max, has_max_fn) {
        merge_into(&mut api_kwargs, apply_max_tokens_fn(params, &Value::from(dm)));
    } else if let Some(am) = anthropic_max {
        api_kwargs.insert("max_tokens".to_string(), am.clone());
    }

    // Provider-specific api_kwargs extras (reasoning_effort, metadata, etc.).
    let reasoning_config = non_null(params.get("reasoning_config"));
    let (extra_body_from_profile, top_level_from_profile) = profile.build_api_kwargs_extras(
        reasoning_config,
        bool_param(params, "supports_reasoning"),
        non_null(params.get("qwen_session_metadata")),
        model,
        non_null(params.get("ollama_num_ctx")),
    );
    for (k, v) in top_level_from_profile {
        api_kwargs.insert(k, v);
    }

    // extra_body assembly.
    let mut extra_body: Map<String, Value> = Map::new();

    // Profile's extra_body (tags, provider prefs, vl_high_resolution, etc.).
    let profile_body = profile.build_extra_body(
        non_null(params.get("session_id")),
        non_null(params.get("provider_preferences")),
        model,
        non_null(params.get("base_url")),
        reasoning_config,
    );
    if !profile_body.is_empty() {
        for (k, v) in profile_body {
            extra_body.insert(k, v);
        }
    }

    // Profile's reasoning/thinking extra_body entries.
    if !extra_body_from_profile.is_empty() {
        for (k, v) in extra_body_from_profile {
            extra_body.insert(k, v);
        }
    }

    // Merge any pre-built extra_body additions from the caller.
    if let Some(Value::Object(additions)) = non_null(params.get("extra_body_additions")) {
        for (k, v) in additions {
            extra_body.insert(k.clone(), v.clone());
        }
    }

    // Request overrides (user config).
    if let Some(Value::Object(overrides)) = non_null(params.get("request_overrides")) {
        for (k, v) in overrides {
            if k == "extra_body" {
                if let Value::Object(vm) = v {
                    for (ek, ev) in vm {
                        extra_body.insert(ek.clone(), ev.clone());
                    }
                    continue;
                }
            }
            api_kwargs.insert(k.clone(), v.clone());
        }
    }

    if !extra_body.is_empty() {
        api_kwargs.insert("extra_body".to_string(), Value::Object(extra_body));
    }

    api_kwargs
}

// ---------------------------------------------------------------------------
// normalize_response
// ---------------------------------------------------------------------------

/// Normalize an OpenAI `ChatCompletion` (as JSON) to [`NormalizedResponse`].
///
/// For chat_completions this is near-identity — the response is already in
/// OpenAI format. `extra_content` on tool_calls (Gemini `thought_signature`)
/// is preserved via `ToolCall.provider_data`. `reasoning_details` (OpenRouter
/// unified format) and `reasoning_content` (DeepSeek/Moonshot) are preserved
/// for downstream replay.
pub fn normalize_chat_completion(response: &Value) -> NormalizedResponse {
    // `choice = response.choices[0]`
    let choice = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(Value::Null);
    let msg = choice.get("message").cloned().unwrap_or(Value::Null);

    // `finish_reason = choice.finish_reason or "stop"`
    let finish_reason = match choice.get("finish_reason") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => "stop".to_string(),
    };

    // Tool calls.
    let mut tool_calls: Option<Vec<ToolCall>> = None;
    if let Some(Value::Array(tcs)) = msg.get("tool_calls") {
        if !tcs.is_empty() {
            let mut list = Vec::with_capacity(tcs.len());
            for tc in tcs {
                let mut provider_data: Map<String, Value> = Map::new();

                // extra = getattr(tc, "extra_content", None)
                // (model_extra fallback collapses to the same JSON key.)
                let extra = tc.get("extra_content");
                if let Some(extra) = extra {
                    if !extra.is_null() {
                        provider_data.insert("extra_content".to_string(), extra.clone());
                    }
                }

                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let func = tc.get("function").cloned().unwrap_or(Value::Null);
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                list.push(ToolCall {
                    id,
                    name,
                    arguments,
                    provider_data: if provider_data.is_empty() {
                        None
                    } else {
                        Some(provider_data)
                    },
                });
            }
            tool_calls = Some(list);
        }
    }

    // Usage.
    let usage = match response.get("usage") {
        Some(u) if !u.is_null() => Some(Usage {
            prompt_tokens: u.get("prompt_tokens").and_then(number_as_i64).unwrap_or(0),
            completion_tokens: u
                .get("completion_tokens")
                .and_then(number_as_i64)
                .unwrap_or(0),
            total_tokens: u.get("total_tokens").and_then(number_as_i64).unwrap_or(0),
            cached_tokens: 0,
        }),
        _ => None,
    };

    // Reasoning fields.
    let reasoning = msg
        .get("reasoning")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // reasoning_content: direct field, else model_extra fallback (same key).
    let reasoning_content = match msg.get("reasoning_content") {
        Some(v) if !v.is_null() => Some(v.clone()),
        _ => None,
    };

    let mut provider_data: Map<String, Value> = Map::new();
    if let Some(rc) = reasoning_content {
        provider_data.insert("reasoning_content".to_string(), rc);
    }
    if let Some(rd) = msg.get("reasoning_details") {
        if is_truthy(Some(rd)) {
            provider_data.insert("reasoning_details".to_string(), rd.clone());
        }
    }

    NormalizedResponse {
        content: msg
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        tool_calls,
        finish_reason,
        reasoning,
        usage,
        provider_data: if provider_data.is_empty() {
            None
        } else {
            Some(provider_data)
        },
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply the developer-role swap to the first system message when the model
/// matches [`DEVELOPER_ROLE_MODELS`]. Mirrors:
///
/// ```python
/// if sanitized and sanitized[0].get("role") == "system" and any(...):
///     sanitized[0] = {**sanitized[0], "role": "developer"}
/// ```
fn maybe_swap_developer_role(sanitized: &mut [Value], model_lower: &str) {
    let matches_model = DEVELOPER_ROLE_MODELS.iter().any(|p| model_lower.contains(p));
    if !matches_model {
        return;
    }
    if let Some(Value::Object(first)) = sanitized.first_mut() {
        if first.get("role").and_then(|v| v.as_str()) == Some("system") {
            first.insert("role".to_string(), Value::String("developer".to_string()));
        }
    }
}

/// Stand-in for the caller-provided `max_tokens_param_fn` callable.
///
/// The Python callable returns `{max_tokens: N}` or `{max_completion_tokens: N}`.
/// Since a Rust closure cannot live in a `Value`, the param map may carry a
/// hint `"max_tokens_param_key"` (defaults to `"max_tokens"`). The function
/// emits `{<key>: value}`.
fn apply_max_tokens_fn(params: &Map<String, Value>, value: &Value) -> Map<String, Value> {
    let key = params
        .get("max_tokens_param_key")
        .and_then(|v| v.as_str())
        .unwrap_or("max_tokens");
    let mut m = Map::new();
    m.insert(key.to_string(), value.clone());
    m
}

fn merge_into(dst: &mut Map<String, Value>, src: Map<String, Value>) {
    for (k, v) in src {
        dst.insert(k, v);
    }
}

/// `params.get(key) is True`-ish: only an explicit JSON `true` counts.
fn bool_param(params: &Map<String, Value>, key: &str) -> bool {
    params.get(key) == Some(&Value::Bool(true))
}

/// Python `is not None` filter: `None` and JSON `null` both drop out.
fn non_null(v: Option<&Value>) -> Option<&Value> {
    match v {
        Some(Value::Null) | None => None,
        other => other,
    }
}

/// Python truthiness for the cases used here: `None`/`null`/`false`/`0`/`""`/
/// empty-collection are falsy.
fn is_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `bool(reasoning_config and isinstance(dict) and .get("enabled") is False)`.
fn reasoning_enabled_is_false(reasoning_config: Option<&Value>) -> bool {
    matches!(
        reasoning_config,
        Some(Value::Object(cfg)) if cfg.get("enabled") == Some(&Value::Bool(false))
    )
}

/// Resolve an effort string from a reasoning config, clamped to `valid`.
///
/// Mirrors the Kimi/TokenHub pattern: default `default`, then if the config's
/// `effort` (stripped/lowercased) is in `valid`, use it.
fn effort_from_config(reasoning_config: Option<&Value>, default: &str, valid: &[&str]) -> String {
    let mut effort = default.to_string();
    if let Some(Value::Object(cfg)) = reasoning_config {
        let e = cfg
            .get("effort")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if valid.contains(&e.as_str()) {
            effort = e;
        }
    }
    effort
}

/// `str(reasoning_config.get("effort", default) or default).strip().lower()`.
fn effort_str(effort: Option<&Value>, default: &str) -> String {
    match effort {
        // Present and truthy string -> use it; falsy ("" / null / missing) -> default.
        Some(Value::String(s)) if !s.is_empty() => s.trim().to_lowercase(),
        Some(Value::String(_)) => default.to_string(),
        None | Some(Value::Null) => default.to_string(),
        // Non-string values: Python `str(x)`. Best-effort for booleans/numbers.
        Some(Value::Bool(b)) => {
            // `False or "medium"` -> default; `True` -> "true"
            if *b {
                "true".to_string()
            } else {
                default.to_string()
            }
        }
        Some(other) => other.to_string().trim().to_lowercase(),
    }
}

/// Convert a JSON number (int or float) to `i64`, truncating like `int(...)`.
fn number_as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                Some(u as i64)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        _ => None,
    }
}

/// `str(value or "")` for a base-url-like field.
fn value_to_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => Map::new(),
        }
    }

    #[test]
    fn convert_messages_noop_when_clean() {
        let t = ChatCompletionsTransport::new();
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let out = t.convert_messages_native(&msgs);
        assert_eq!(out, msgs);
    }

    #[test]
    fn convert_messages_strips_codex_fields() {
        let t = ChatCompletionsTransport::new();
        let msgs = vec![json!({
            "role": "assistant",
            "codex_reasoning_items": [1, 2],
            "codex_message_items": [3],
            "tool_calls": [
                {"id": "x", "call_id": "call_1", "response_item_id": "fc_1",
                 "function": {"name": "f", "arguments": "{}"}}
            ]
        })];
        let out = t.convert_messages_native(&msgs);
        let m = &out[0];
        assert!(m.get("codex_reasoning_items").is_none());
        assert!(m.get("codex_message_items").is_none());
        let tc = &m["tool_calls"][0];
        assert!(tc.get("call_id").is_none());
        assert!(tc.get("response_item_id").is_none());
        assert_eq!(tc["id"], json!("x"));
    }

    #[test]
    fn legacy_basic_kwargs() {
        let params = Map::new();
        let kwargs = build_kwargs_legacy(
            "some-model",
            vec![json!({"role": "user", "content": "hi"})],
            None,
            &params,
        );
        assert_eq!(kwargs["model"], json!("some-model"));
        assert_eq!(kwargs["messages"], json!([{"role": "user", "content": "hi"}]));
        assert!(!kwargs.contains_key("extra_body"));
    }

    #[test]
    fn developer_role_swap_for_gpt5() {
        let mut params = Map::new();
        params.insert("model_lower".to_string(), json!("gpt-5-mini"));
        let kwargs = build_kwargs_legacy(
            "GPT-5-mini",
            vec![
                json!({"role": "system", "content": "sys"}),
                json!({"role": "user", "content": "hi"}),
            ],
            None,
            &params,
        );
        assert_eq!(kwargs["messages"][0]["role"], json!("developer"));
        assert_eq!(kwargs["messages"][1]["role"], json!("user"));
    }

    #[test]
    fn no_developer_swap_for_other_models() {
        let params = Map::new();
        let kwargs = build_kwargs_legacy(
            "llama-3",
            vec![json!({"role": "system", "content": "sys"})],
            None,
            &params,
        );
        assert_eq!(kwargs["messages"][0]["role"], json!("system"));
    }

    #[test]
    fn kimi_reasoning_effort_and_thinking() {
        let mut params = Map::new();
        params.insert("is_kimi".to_string(), json!(true));
        params.insert("reasoning_config".to_string(), json!({"effort": "high"}));
        let kwargs = build_kwargs_legacy("kimi-k2", Vec::new(), None, &params);
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
        assert_eq!(kwargs["extra_body"]["thinking"]["type"], json!("enabled"));
    }

    #[test]
    fn kimi_thinking_disabled() {
        let mut params = Map::new();
        params.insert("is_kimi".to_string(), json!(true));
        params.insert("reasoning_config".to_string(), json!({"enabled": false}));
        let kwargs = build_kwargs_legacy("kimi-k2", Vec::new(), None, &params);
        // thinking disabled => no top-level reasoning_effort
        assert!(!kwargs.contains_key("reasoning_effort"));
        assert_eq!(kwargs["extra_body"]["thinking"]["type"], json!("disabled"));
    }

    #[test]
    fn tokenhub_default_high() {
        let mut params = Map::new();
        params.insert("is_tokenhub".to_string(), json!(true));
        let kwargs = build_kwargs_legacy("hunyuan", Vec::new(), None, &params);
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn openrouter_provider_prefs_in_extra_body() {
        let mut params = Map::new();
        params.insert("is_openrouter".to_string(), json!(true));
        params.insert(
            "provider_preferences".to_string(),
            json!({"order": ["x"]}),
        );
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(kwargs["extra_body"]["provider"], json!({"order": ["x"]}));
    }

    #[test]
    fn supports_reasoning_default_extra_body() {
        let mut params = Map::new();
        params.insert("supports_reasoning".to_string(), json!(true));
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(
            kwargs["extra_body"]["reasoning"],
            json!({"enabled": true, "effort": "medium"})
        );
    }

    #[test]
    fn github_reasoning_extra_used() {
        let mut params = Map::new();
        params.insert("supports_reasoning".to_string(), json!(true));
        params.insert("is_github_models".to_string(), json!(true));
        params.insert("github_reasoning_extra".to_string(), json!({"x": 1}));
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(kwargs["extra_body"]["reasoning"], json!({"x": 1}));
    }

    #[test]
    fn lmstudio_top_level_reasoning_effort_skips_extra_body() {
        let mut params = Map::new();
        params.insert("is_lmstudio".to_string(), json!(true));
        params.insert("supports_reasoning".to_string(), json!(true));
        params.insert("reasoning_config".to_string(), json!({"effort": "high"}));
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
        // No extra_body.reasoning for LM Studio.
        assert!(kwargs.get("extra_body").is_none());
    }

    #[test]
    fn max_tokens_priority_and_param_key() {
        let mut params = Map::new();
        params.insert("max_tokens_param_fn".to_string(), json!("present"));
        params.insert(
            "max_tokens_param_key".to_string(),
            json!("max_completion_tokens"),
        );
        params.insert("ephemeral_max_output_tokens".to_string(), json!(123));
        params.insert("max_tokens".to_string(), json!(999));
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        // ephemeral wins
        assert_eq!(kwargs["max_completion_tokens"], json!(123));
    }

    #[test]
    fn anthropic_max_output_fallback() {
        let mut params = Map::new();
        params.insert("anthropic_max_output".to_string(), json!(8000));
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(kwargs["max_tokens"], json!(8000));
    }

    #[test]
    fn request_overrides_applied_last() {
        let mut params = Map::new();
        params.insert(
            "request_overrides".to_string(),
            json!({"service_tier": "flex"}),
        );
        let kwargs = build_kwargs_legacy("foo", Vec::new(), None, &params);
        assert_eq!(kwargs["service_tier"], json!("flex"));
    }

    #[test]
    fn gemini_thinking_config_basic() {
        let cfg = json!({"effort": "medium"});
        let tc = build_gemini_thinking_config("gemini-3-pro", Some(&cfg)).unwrap();
        assert_eq!(tc.get("includeThoughts"), Some(&json!(true)));
        // pro: medium -> low
        assert_eq!(tc.get("thinkingLevel"), Some(&json!("low")));
    }

    #[test]
    fn gemini_thinking_config_flash_high() {
        let cfg = json!({"effort": "xhigh"});
        let tc = build_gemini_thinking_config("gemini-3-flash", Some(&cfg)).unwrap();
        assert_eq!(tc.get("thinkingLevel"), Some(&json!("high")));
    }

    #[test]
    fn gemini_thinking_config_25_no_level() {
        let cfg = json!({"effort": "high"});
        let tc = build_gemini_thinking_config("gemini-2.5-pro", Some(&cfg)).unwrap();
        assert_eq!(tc.get("includeThoughts"), Some(&json!(true)));
        assert!(tc.get("thinkingLevel").is_none());
    }

    #[test]
    fn gemini_thinking_config_disabled() {
        let cfg = json!({"enabled": false});
        let tc = build_gemini_thinking_config("gemini-3-pro", Some(&cfg)).unwrap();
        assert_eq!(tc.get("includeThoughts"), Some(&json!(false)));
    }

    #[test]
    fn gemini_thinking_config_non_gemini_none() {
        let cfg = json!({"effort": "high"});
        assert!(build_gemini_thinking_config("gemma-2", Some(&cfg)).is_none());
        assert!(build_gemini_thinking_config("google/gemma-2", Some(&cfg)).is_none());
    }

    #[test]
    fn gemini_base_url_detection() {
        assert!(is_gemini_openai_compat_base_url(Some(&json!(
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        ))));
        assert!(!is_gemini_openai_compat_base_url(Some(&json!(
            "https://generativelanguage.googleapis.com/v1beta"
        ))));
        assert!(!is_gemini_openai_compat_base_url(Some(&json!(
            "https://api.openai.com/v1"
        ))));
        assert!(!is_gemini_openai_compat_base_url(None));
    }

    #[test]
    fn snake_case_thinking_config() {
        let cfg = obj(json!({"includeThoughts": true, "thinkingLevel": "HIGH", "thinkingBudget": 1024.0}));
        let out = snake_case_gemini_thinking_config(Some(&cfg)).unwrap();
        assert_eq!(out.get("include_thoughts"), Some(&json!(true)));
        assert_eq!(out.get("thinking_level"), Some(&json!("high")));
        assert_eq!(out.get("thinking_budget"), Some(&json!(1024)));
    }

    #[test]
    fn gemini_provider_openai_compat_path() {
        let mut params = Map::new();
        params.insert("provider_name".to_string(), json!("gemini"));
        params.insert(
            "base_url".to_string(),
            json!("https://generativelanguage.googleapis.com/v1beta/openai"),
        );
        params.insert("reasoning_config".to_string(), json!({"effort": "high"}));
        let kwargs = build_kwargs_legacy("gemini-3-pro", Vec::new(), None, &params);
        let google = &kwargs["extra_body"]["extra_body"]["google"]["thinking_config"];
        assert_eq!(google["include_thoughts"], json!(true));
        assert_eq!(google["thinking_level"], json!("high"));
    }

    #[test]
    fn gemini_provider_native_path() {
        let mut params = Map::new();
        params.insert("provider_name".to_string(), json!("gemini"));
        params.insert("base_url".to_string(), json!("https://example.com/v1"));
        params.insert("reasoning_config".to_string(), json!({"effort": "high"}));
        let kwargs = build_kwargs_legacy("gemini-3-pro", Vec::new(), None, &params);
        assert_eq!(
            kwargs["extra_body"]["thinking_config"]["thinkingLevel"],
            json!("high")
        );
    }

    #[test]
    fn google_gemini_cli_thinking_config() {
        let mut params = Map::new();
        params.insert("provider_name".to_string(), json!("google-gemini-cli"));
        params.insert("reasoning_config".to_string(), json!({"effort": "low"}));
        let kwargs = build_kwargs_legacy("gemini-3-pro", Vec::new(), None, &params);
        assert_eq!(
            kwargs["extra_body"]["thinking_config"]["thinkingLevel"],
            json!("low")
        );
    }

    #[test]
    fn moonshot_tools_sanitized() {
        let params = Map::new();
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}
        })];
        let kwargs = build_kwargs_legacy("kimi-k2", Vec::new(), Some(tools), &params);
        assert!(kwargs.contains_key("tools"));
    }

    #[test]
    fn normalize_basic_response() {
        let resp = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let nr = normalize_chat_completion(&resp);
        assert_eq!(nr.content.as_deref(), Some("hello"));
        assert_eq!(nr.finish_reason, "stop");
        assert!(nr.tool_calls.is_none());
        let u = nr.usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn normalize_tool_calls_with_extra_content() {
        let resp = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "do_thing", "arguments": "{\"a\":1}"},
                        "extra_content": {"google": {"thought_signature": "sig"}}
                    }]
                }
            }]
        });
        let nr = normalize_chat_completion(&resp);
        assert_eq!(nr.finish_reason, "tool_calls");
        let tcs = nr.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id.as_deref(), Some("call_1"));
        assert_eq!(tcs[0].name, "do_thing");
        assert_eq!(tcs[0].arguments, "{\"a\":1}");
        let pd = tcs[0].provider_data.as_ref().unwrap();
        assert_eq!(
            pd["extra_content"]["google"]["thought_signature"],
            json!("sig")
        );
    }

    #[test]
    fn normalize_reasoning_fields() {
        let resp = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "x",
                    "reasoning": "thinking...",
                    "reasoning_content": "deepseek-cot",
                    "reasoning_details": [{"type": "reasoning.text"}]
                }
            }]
        });
        let nr = normalize_chat_completion(&resp);
        assert_eq!(nr.reasoning.as_deref(), Some("thinking..."));
        let pd = nr.provider_data.unwrap();
        assert_eq!(pd["reasoning_content"], json!("deepseek-cot"));
        assert_eq!(pd["reasoning_details"], json!([{"type": "reasoning.text"}]));
    }

    #[test]
    fn normalize_finish_reason_fallback() {
        let resp = json!({"choices": [{"message": {"content": "x"}}]});
        let nr = normalize_chat_completion(&resp);
        assert_eq!(nr.finish_reason, "stop");
    }

    #[test]
    fn validate_response_checks() {
        let t = ChatCompletionsTransport::new();
        assert!(!t.validate_response(&Value::Null));
        assert!(!t.validate_response(&json!({})));
        assert!(!t.validate_response(&json!({"choices": null})));
        assert!(!t.validate_response(&json!({"choices": []})));
        assert!(t.validate_response(&json!({"choices": [{"message": {}}]})));
    }

    #[test]
    fn extract_cache_stats_present() {
        let t = ChatCompletionsTransport::new();
        let resp = json!({
            "usage": {"prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20}}
        });
        assert_eq!(t.extract_cache_stats(&resp), Some((100, 20)));
    }

    #[test]
    fn extract_cache_stats_absent() {
        let t = ChatCompletionsTransport::new();
        assert_eq!(t.extract_cache_stats(&json!({})), None);
        assert_eq!(
            t.extract_cache_stats(&json!({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}})),
            None
        );
    }

    // ── Profile path ──────────────────────────────────────────────

    struct OmitTempProfile;
    impl ProviderProfile for OmitTempProfile {
        fn fixed_temperature(&self) -> FixedTemperature {
            FixedTemperature::Omit
        }
        fn default_max_tokens(&self) -> Option<i64> {
            Some(4096)
        }
        fn build_api_kwargs_extras(
            &self,
            _rc: Option<&Value>,
            _sr: bool,
            _qm: Option<&Value>,
            _model: &str,
            _ctx: Option<&Value>,
        ) -> (Map<String, Value>, Map<String, Value>) {
            let mut top = Map::new();
            top.insert("reasoning_effort".to_string(), json!("high"));
            let mut eb = Map::new();
            eb.insert("foo".to_string(), json!("bar"));
            (eb, top)
        }
    }

    #[test]
    fn profile_omit_temp_and_extras() {
        let profile = OmitTempProfile;
        let mut params = Map::new();
        params.insert("max_tokens_param_fn".to_string(), json!("present"));
        let kwargs = build_kwargs_from_profile(
            &profile,
            "some-model",
            vec![json!({"role": "user", "content": "hi"})],
            None,
            &params,
        );
        // omit temperature
        assert!(!kwargs.contains_key("temperature"));
        // profile default max tokens applied via fn
        assert_eq!(kwargs["max_tokens"], json!(4096));
        // top-level extras
        assert_eq!(kwargs["reasoning_effort"], json!("high"));
        // extra_body from profile extras
        assert_eq!(kwargs["extra_body"]["foo"], json!("bar"));
    }

    struct FixedTempProfile;
    impl ProviderProfile for FixedTempProfile {
        fn fixed_temperature(&self) -> FixedTemperature {
            FixedTemperature::Value(json!(0.0))
        }
    }

    #[test]
    fn profile_fixed_temperature() {
        let profile = FixedTempProfile;
        let params = Map::new();
        let kwargs = build_kwargs_from_profile(&profile, "m", Vec::new(), None, &params);
        assert_eq!(kwargs["temperature"], json!(0.0));
    }

    #[test]
    fn profile_overrides_extra_body_merges() {
        struct P;
        impl ProviderProfile for P {}
        let params_v = json!({
            "request_overrides": {"extra_body": {"k": "v"}, "service_tier": "flex"}
        });
        let params = obj(params_v);
        let kwargs = build_kwargs_from_profile(&P, "m", Vec::new(), None, &params);
        assert_eq!(kwargs["service_tier"], json!("flex"));
        assert_eq!(kwargs["extra_body"]["k"], json!("v"));
    }
}
