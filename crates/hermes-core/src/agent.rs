use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aws_config::BehaviorVersion as AwsBehaviorVersion;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_bedrockruntime::config::Region as BedrockRegion;
use aws_sdk_bedrockruntime::primitives::Blob as BedrockBlob;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole as BedrockConversationRole,
    ConverseOutput as BedrockConverseMessageOutput, ImageBlock as BedrockImageBlock,
    ImageFormat as BedrockImageFormat, ImageSource as BedrockImageSource,
    InferenceConfiguration as BedrockInferenceConfiguration, Message as BedrockMessage,
    StopReason as BedrockStopReason, SystemContentBlock as BedrockSystemContentBlock,
    Tool as BedrockTool, ToolConfiguration as BedrockToolConfiguration,
    ToolInputSchema as BedrockToolInputSchema, ToolResultBlock as BedrockToolResultBlock,
    ToolResultContentBlock as BedrockToolResultContentBlock,
    ToolSpecification as BedrockToolSpecification,
};
use aws_smithy_types::{Document as SmithyDocument, Number as SmithyNumber};
use base64::Engine as _;
use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    HermesContext, HermesError, LoadedConfig, MessageAppend, ModelOverrides, SessionCreate,
    SessionStore, ToolRuntime, dispatch_tool, get_tool_definitions,
};

const DEFAULT_AGENT_TIMEOUT_SECS: u64 = 300;
const MAX_HTTP_ERROR_BODY_CHARS: usize = 4000;
const CREDENTIAL_POOL_EXHAUSTED_STATUS: &str = "exhausted";
const CREDENTIAL_POOL_EXHAUSTED_TTL_SECONDS: f64 = 60.0 * 60.0;
const NOUS_RATE_LIMIT_STATE_SUBDIR: &str = "rate_limits";
const NOUS_RATE_LIMIT_STATE_FILE: &str = "nous.json";
const NOUS_RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS: f64 = 300.0;
const NOUS_RATE_LIMIT_MIN_BREAKER_RESET_SECONDS: f64 = 60.0;
const KANBAN_GUIDANCE: &str = "# Kanban task execution protocol\nUse kanban_show first to orient on the assigned task. Work inside HERMES_KANBAN_WORKSPACE unless the task explicitly requires otherwise. Heartbeat during long-running work, block when you need human input you cannot infer, and finish with kanban_complete(summary=..., metadata=...) or kanban_block(reason=...). Use kanban_create for real follow-up work instead of silently scope-creeping into it.";
const COPILOT_ACP_MARKER_BASE_URL: &str = "acp://copilot";
const GOOGLE_CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const GOOGLE_CODE_ASSIST_FALLBACK_ENDPOINTS: &[&str] = &[
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
];
const GOOGLE_CODE_ASSIST_BASE_URL_ENV: &str = "HERMES_GOOGLE_CODE_ASSIST_BASE_URL";
const GOOGLE_CODE_ASSIST_CONTROL_USER_AGENT: &str = "google-api-nodejs-client/9.15.1 (gzip)";
const GOOGLE_CODE_ASSIST_CONTROL_API_CLIENT: &str = "gl-node/24.0.0";
const GOOGLE_CODE_ASSIST_INFERENCE_USER_AGENT: &str = "hermes-agent (gemini-cli-compat)";
const GOOGLE_CODE_ASSIST_INFERENCE_API_CLIENT: &str = "gl-python/hermes";
const GOOGLE_CODE_ASSIST_FREE_TIER_ID: &str = "free-tier";
const QWEN_CODE_VERSION: &str = "0.14.1";
const CONTEXT_PROBE_TIERS: &[u64] = &[256_000, 128_000, 64_000, 32_000, 16_000, 8_000];
const DEFAULT_FALLBACK_CONTEXT: u64 = 256_000;
const ANTHROPIC_STANDARD_CONTEXT_TIER: u64 = 200_000;
const POST_TOOL_EMPTY_NUDGE: &str = "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.";
const IMAGE_SHRINK_TARGET_BYTES: usize = 4 * 1024 * 1024;
const BILLING_PATTERNS: &[&str] = &[
    "insufficient credits",
    "insufficient_quota",
    "insufficient balance",
    "credit balance",
    "credits have been exhausted",
    "top up your credits",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
    "spending limit",
];
const RATE_LIMIT_PATTERNS: &[&str] = &[
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "rate increased too quickly",
    "throttlingexception",
    "too many concurrent requests",
    "servicequotaexceededexception",
];
const USAGE_LIMIT_PATTERNS: &[&str] = &[
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
];
const USAGE_LIMIT_TRANSIENT_SIGNALS: &[&str] = &[
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
];
const MODEL_NOT_FOUND_PATTERNS: &[&str] = &[
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
];
const PROVIDER_POLICY_BLOCKED_PATTERNS: &[&str] = &[
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
];
const IMAGE_TOO_LARGE_PATTERNS: &[&str] = &[
    "image exceeds",
    "image too large",
    "image_too_large",
    "image size exceeds",
];

static THINK_BLOCK_RE: OnceLock<Regex> = OnceLock::new();
static THINK_OPEN_TAIL_RE: OnceLock<Regex> = OnceLock::new();
static THINK_TAG_RE: OnceLock<Regex> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTurnResult {
    pub final_response: String,
    pub api_calls: u64,
    pub tool_calls: u64,
    pub model: String,
    pub provider: String,
    pub base_url: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments_raw: String,
    json: Value,
}

#[derive(Debug, Clone)]
struct NormalizedAssistantResponse {
    content: Option<String>,
    tool_calls: Vec<PendingToolCall>,
    finish_reason: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    reasoning_details: Option<Value>,
    codex_reasoning_items: Option<Value>,
    codex_message_items: Option<Value>,
}

#[derive(Debug, Clone)]
struct CredentialPoolEntry {
    priority: i64,
    access_token: String,
    agent_key: String,
    base_url: String,
    inference_base_url: String,
    last_status: Option<String>,
    last_error_reset_at: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NousRateLimitState {
    reset_at: f64,
    recorded_at: f64,
    reset_seconds: f64,
}

#[derive(Debug, Clone, Default)]
struct NousRateLimitBucketSnapshot {
    limit: i64,
    remaining: i64,
    reset_seconds: f64,
}

#[derive(Debug, Clone, Default)]
struct NousObservedRateLimitState {
    requests_min: NousRateLimitBucketSnapshot,
    requests_hour: NousRateLimitBucketSnapshot,
    tokens_min: NousRateLimitBucketSnapshot,
    tokens_hour: NousRateLimitBucketSnapshot,
}

thread_local! {
    static LAST_NOUS_RATE_LIMIT_STATE: RefCell<Option<NousObservedRateLimitState>> = const { RefCell::new(None) };
}

impl HermesContext {
    pub fn render_system_prompt(
        &self,
        runtime: &ToolRuntime,
        memory_config: &crate::MemoryConfig,
    ) -> Result<String, HermesError> {
        let mut runtime = runtime.clone();
        let _ = runtime.load_memory_store(memory_config);
        build_system_prompt(&runtime, memory_config)
    }

    pub fn run_chat_completions_turn(
        &self,
        loaded: &LoadedConfig,
        prompt: &str,
        runtime: &ToolRuntime,
        enabled_toolsets: Option<&[String]>,
        overrides: &ModelOverrides,
        session_hint: Option<&str>,
        session_store: Option<&SessionStore>,
    ) -> Result<AgentTurnResult, HermesError> {
        self.run_chat_turn_with_user_content(
            loaded,
            Value::String(prompt.trim().to_string()),
            runtime,
            enabled_toolsets,
            overrides,
            session_hint,
            session_store,
        )
    }

    pub fn run_chat_turn_with_user_content(
        &self,
        loaded: &LoadedConfig,
        user_content: Value,
        runtime: &ToolRuntime,
        enabled_toolsets: Option<&[String]>,
        overrides: &ModelOverrides,
        session_hint: Option<&str>,
        session_store: Option<&SessionStore>,
    ) -> Result<AgentTurnResult, HermesError> {
        if !content_has_meaningful_user_input(&user_content) {
            return Err(HermesError::State {
                action: "running agent turn",
                detail: "Prompt must not be empty.".to_string(),
            });
        }

        let disabled_toolsets =
            (!loaded.config.memory.any_enabled()).then(|| vec![String::from("memory")]);
        let tools = get_tool_definitions(enabled_toolsets, disabled_toolsets.as_deref());
        self.run_chat_turn_with_tools(
            loaded,
            user_content,
            runtime,
            tools,
            overrides,
            session_hint,
            session_store,
        )
    }

    fn run_chat_turn_with_tools(
        &self,
        loaded: &LoadedConfig,
        user_content: Value,
        runtime: &ToolRuntime,
        tools: Vec<crate::ToolDefinition>,
        overrides: &ModelOverrides,
        session_hint: Option<&str>,
        session_store: Option<&SessionStore>,
    ) -> Result<AgentTurnResult, HermesError> {
        let runtime_model = self.resolve_model_runtime(loaded, overrides)?;
        self.run_chat_turn_with_resolved_runtime(
            loaded,
            user_content,
            runtime,
            tools,
            runtime_model,
            Some(overrides),
            session_hint,
            session_store,
        )
    }

    fn run_chat_turn_with_resolved_runtime(
        &self,
        loaded: &LoadedConfig,
        user_content: Value,
        runtime: &ToolRuntime,
        tools: Vec<crate::ToolDefinition>,
        runtime_model: crate::ModelRuntimeConfig,
        runtime_overrides: Option<&ModelOverrides>,
        session_hint: Option<&str>,
        session_store: Option<&SessionStore>,
    ) -> Result<AgentTurnResult, HermesError> {
        let mut runtime_model = runtime_model;
        let fallback_chain = resolve_fallback_chain(loaded);
        let mut fallback_index = 0_usize;
        let mut context_length = loaded
            .configured_model_context_length()
            .unwrap_or(DEFAULT_FALLBACK_CONTEXT);
        let mut max_output_tokens: Option<u64> = None;
        let mut tool_runtime = runtime.clone();
        if let Err(error) = tool_runtime.load_memory_store(&loaded.config.memory) {
            log::warn!(target: "run_agent", "memory bootstrap skipped: {error}");
        }
        let mut tools = tools;
        tool_runtime = tool_runtime.with_available_tool_names(
            tools
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>(),
        );
        let mut system_prompt = build_system_prompt(&tool_runtime, &loaded.config.memory)?;
        let mut messages = Vec::new();
        let mut session_id = None;

        if let Some(store) = session_store {
            if let Some(session_hint) = session_hint.and_then(non_empty_trimmed) {
                let resolved =
                    store
                        .resolve_session_id(&session_hint)?
                        .ok_or_else(|| HermesError::State {
                            action: "resolving session",
                            detail: format!("No unique session matched '{session_hint}'."),
                        })?;
                let session = store
                    .get_session(&resolved)?
                    .ok_or_else(|| HermesError::State {
                        action: "loading session",
                        detail: format!("Session '{resolved}' no longer exists."),
                    })?;
                if let Some(saved_prompt) =
                    session.system_prompt.as_deref().and_then(non_empty_trimmed)
                {
                    system_prompt = saved_prompt;
                }
                let prior_messages = store.get_messages(&resolved)?;
                tool_runtime = tool_runtime.with_current_session_id(Some(resolved.clone()));
                tool_runtime.hydrate_todo_from_messages(&prior_messages);
                messages.push(json!({
                    "role": "system",
                    "content": system_prompt,
                }));
                for prior in prior_messages {
                    messages.push(message_record_to_chat_message(prior)?);
                }
                session_id = Some(resolved);
            }
        }

        if messages.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_prompt,
            }));
        }
        messages.push(json!({
            "role": "user",
            "content": user_content.clone(),
        }));

        if let Some(store) = session_store {
            if session_id.is_none() {
                let new_session_id = format!("rs_{:x}", unix_ts_nanos());
                store.create_session(&SessionCreate {
                    id: new_session_id.clone(),
                    source: "rust-agent".to_string(),
                    user_id: None,
                    model: Some(runtime_model.model.clone()),
                    model_config: Some(model_runtime_metadata(&runtime_model)),
                    system_prompt: Some(system_prompt.clone()),
                    parent_session_id: None,
                })?;
                tool_runtime = tool_runtime.with_current_session_id(Some(new_session_id.clone()));
                session_id = Some(new_session_id);
            }
            if let Some(session_id) = session_id.as_deref() {
                let _ = store.append_message(
                    session_id,
                    &MessageAppend {
                        role: "user".to_string(),
                        content: Some(user_content.clone()),
                        tool_call_id: None,
                        tool_calls: None,
                        tool_name: None,
                        token_count: None,
                        finish_reason: None,
                        reasoning: None,
                        reasoning_content: None,
                        reasoning_details: None,
                        codex_reasoning_items: None,
                        codex_message_items: None,
                    },
                );
            }
        }

        let client = build_http_client()?;
        let mut api_calls = 0_u64;
        let mut tool_calls = 0_u64;
        let mut empty_content_retries = 0_u64;
        let mut thinking_prefill_retries = 0_u64;
        let mut cleared_nous_rate_limit_snapshot = false;
        let max_retries = loaded.config.agent.api_max_retries.max(1);
        let max_compactions = max_retries + 1;

        for _ in 0..loaded.config.agent.max_turns {
            let mut request_retries = 0_u64;
            let mut compactions = 0_u64;
            let mut image_shrink_retry_attempted = false;
            let mut auth_refresh_retry_attempted = false;
            let mut llama_cpp_schema_retry_attempted = false;
            let mut thinking_signature_retry_attempted = false;
            let mut oauth_long_context_beta_retry_attempted = false;
            let mut pool_rate_limit_retry_attempted = false;
            let mut pool_auth_refresh_retry_attempted = false;
            if runtime_model.provider == "nous" && !cleared_nous_rate_limit_snapshot {
                clear_captured_nous_rate_limit_state();
                cleared_nous_rate_limit_snapshot = true;
            }
            let response = loop {
                if loaded.config.compression.enabled
                    && should_compact_messages(
                        &messages,
                        Some(context_length),
                        loaded.config.compression.threshold,
                    )
                    && compactions < max_compactions
                {
                    if let Some(compacted) = compact_messages_for_context_budget(
                        &messages,
                        loaded.config.compression.protect_last_n,
                        loaded.config.compression.target_ratio,
                    ) {
                        messages = compacted;
                        compactions += 1;
                        if let Some(store) = session_store
                            && let Some(active_session_id) = session_id.clone()
                        {
                            let rotated = rotate_compressed_session(
                                store,
                                &active_session_id,
                                &runtime_model,
                                &system_prompt,
                                &messages,
                            )?;
                            tool_runtime =
                                tool_runtime.with_current_session_id(Some(rotated.clone()));
                            session_id = Some(rotated);
                        }
                    }
                }

                if runtime_model.provider == "nous"
                    && let Some(remaining) = nous_rate_limit_remaining(self)
                {
                    if let Some(fallback_runtime) = activate_next_fallback_runtime(
                        self,
                        loaded,
                        &fallback_chain,
                        &mut fallback_index,
                    )? {
                        runtime_model = fallback_runtime;
                        if let Some(store) = session_store
                            && let Some(active_session_id) = session_id.as_deref()
                        {
                            let _ = store.update_session_runtime(
                                active_session_id,
                                Some(&runtime_model.model),
                                Some(&model_runtime_metadata(&runtime_model)),
                            );
                        }
                        request_retries = 0;
                        compactions = 0;
                        max_output_tokens = None;
                        continue;
                    }
                    return Err(HermesError::State {
                        action: "calling Nous runtime",
                        detail: format!(
                            "Nous Portal rate limit active — resets in {}.",
                            format_duration_remaining(remaining)
                        ),
                    });
                }

                api_calls += 1;
                match send_model_request(
                    &client,
                    &runtime_model,
                    &messages,
                    &tools,
                    session_id.as_deref(),
                    max_output_tokens,
                ) {
                    Ok(response) => {
                        if runtime_model.provider == "nous" {
                            clear_nous_rate_limit_state(self);
                        }
                        let response_text = visible_assistant_text(response.content.as_deref());
                        if response.tool_calls.is_empty()
                            && response_text.is_none()
                            && !response_has_reasoning_signal(&response)
                        {
                            if messages
                                .last()
                                .and_then(|message| message.get("role"))
                                .and_then(Value::as_str)
                                == Some("tool")
                            {
                                let mut empty_message = assistant_response_message(&response, None);
                                empty_message["content"] = Value::String("(empty)".to_string());
                                messages.push(empty_message.clone());
                                messages.push(json!({
                                    "role": "user",
                                    "content": POST_TOOL_EMPTY_NUDGE,
                                }));
                                if let Some(store) = session_store
                                    && let Some(active_session_id) = session_id.as_deref()
                                {
                                    let mut empty_append =
                                        assistant_response_append(&response, None);
                                    empty_append.content =
                                        Some(Value::String("(empty)".to_string()));
                                    let _ = store.append_message(active_session_id, &empty_append);
                                    let _ = store.append_message(
                                        active_session_id,
                                        &MessageAppend {
                                            role: "user".to_string(),
                                            content: Some(Value::String(
                                                POST_TOOL_EMPTY_NUDGE.to_string(),
                                            )),
                                            tool_call_id: None,
                                            tool_calls: None,
                                            tool_name: None,
                                            token_count: None,
                                            finish_reason: None,
                                            reasoning: None,
                                            reasoning_content: None,
                                            reasoning_details: None,
                                            codex_reasoning_items: None,
                                            codex_message_items: None,
                                        },
                                    );
                                }
                                max_output_tokens = None;
                                continue;
                            }
                            if request_retries < max_retries {
                                request_retries += 1;
                                thread::sleep(retry_backoff(request_retries));
                                continue;
                            }
                            if let Some(fallback_runtime) = activate_next_fallback_runtime(
                                self,
                                loaded,
                                &fallback_chain,
                                &mut fallback_index,
                            )? {
                                runtime_model = fallback_runtime;
                                if let Some(store) = session_store
                                    && let Some(active_session_id) = session_id.as_deref()
                                {
                                    let _ = store.update_session_runtime(
                                        active_session_id,
                                        Some(&runtime_model.model),
                                        Some(&model_runtime_metadata(&runtime_model)),
                                    );
                                }
                                request_retries = 0;
                                compactions = 0;
                                max_output_tokens = None;
                                continue;
                            }
                            return Err(HermesError::State {
                                action: "parsing assistant response",
                                detail: "Assistant response had no text content.".to_string(),
                            });
                        }
                        max_output_tokens = None;
                        break response;
                    }
                    Err(error) => {
                        if error_blocks_provider_fallback(&error) {
                            return Err(error);
                        }

                        let pool_can_recover = error_supports_credential_pool_rotation(&error)
                            && credential_pool_alternative_available(self, &runtime_model);
                        if error_is_rate_limit_like(&error)
                            && !pool_rate_limit_retry_attempted
                            && pool_can_recover
                        {
                            pool_rate_limit_retry_attempted = true;
                        } else if (error_is_billing_exhaustion(&error)
                            || (error_is_rate_limit_like(&error)
                                && pool_rate_limit_retry_attempted))
                            && pool_can_recover
                            && let Some(rotated_runtime) = try_rotate_credential_pool_runtime(
                                self,
                                loaded,
                                &runtime_model,
                                error_http_status(&error),
                            )?
                        {
                            runtime_model = rotated_runtime;
                            request_retries = 0;
                            compactions = 0;
                            max_output_tokens = None;
                            pool_rate_limit_retry_attempted = false;
                            continue;
                        }

                        if error_is_auth_like(&error)
                            && !pool_auth_refresh_retry_attempted
                            && credential_pool_supports_current_entry_auth_refresh(&runtime_model)
                            && credential_pool_has_current_runtime(self, &runtime_model)
                        {
                            pool_auth_refresh_retry_attempted = true;
                            if let Some(refreshed_runtime) =
                                try_refresh_current_credential_pool_runtime(self, &runtime_model)?
                            {
                                runtime_model = refreshed_runtime;
                                request_retries = 0;
                                compactions = 0;
                                max_output_tokens = None;
                                pool_rate_limit_retry_attempted = false;
                                continue;
                            }
                            if pool_can_recover
                                && let Some(rotated_runtime) = try_rotate_credential_pool_runtime(
                                    self,
                                    loaded,
                                    &runtime_model,
                                    error_http_status(&error),
                                )?
                            {
                                runtime_model = rotated_runtime;
                                request_retries = 0;
                                compactions = 0;
                                max_output_tokens = None;
                                pool_rate_limit_retry_attempted = false;
                                continue;
                            }
                        }

                        if error_is_auth_like(&error)
                            && pool_can_recover
                            && credential_pool_prefers_auth_rotation(&runtime_model)
                            && let Some(rotated_runtime) = try_rotate_credential_pool_runtime(
                                self,
                                loaded,
                                &runtime_model,
                                error_http_status(&error),
                            )?
                        {
                            runtime_model = rotated_runtime;
                            request_retries = 0;
                            compactions = 0;
                            max_output_tokens = None;
                            pool_rate_limit_retry_attempted = false;
                            continue;
                        }

                        if !auth_refresh_retry_attempted
                            && runtime_overrides.is_some()
                            && !(pool_auth_refresh_retry_attempted
                                && credential_pool_supports_current_entry_auth_refresh(
                                    &runtime_model,
                                )
                                && credential_pool_has_current_runtime(self, &runtime_model))
                            && should_retry_with_reresolved_auth(&runtime_model, &error)
                        {
                            auth_refresh_retry_attempted = true;
                            if refresh_runtime_auth_for_retry(self, &runtime_model)
                                && let Ok(refreshed_runtime) =
                                    self.resolve_model_runtime(loaded, runtime_overrides.unwrap())
                            {
                                runtime_model = refreshed_runtime;
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if error_is_auth_like(&error)
                            && pool_can_recover
                            && let Some(rotated_runtime) = try_rotate_credential_pool_runtime(
                                self,
                                loaded,
                                &runtime_model,
                                error_http_status(&error),
                            )?
                        {
                            runtime_model = rotated_runtime;
                            request_retries = 0;
                            compactions = 0;
                            max_output_tokens = None;
                            pool_rate_limit_retry_attempted = false;
                            continue;
                        }

                        if !image_shrink_retry_attempted && error_is_image_too_large(&error) {
                            image_shrink_retry_attempted = true;
                            if try_shrink_image_parts_in_messages(
                                &mut messages,
                                IMAGE_SHRINK_TARGET_BYTES,
                            ) {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if runtime_model.api_mode == "anthropic_messages"
                            && !oauth_long_context_beta_retry_attempted
                            && error_is_oauth_long_context_beta_forbidden(&error)
                        {
                            oauth_long_context_beta_retry_attempted = true;
                            if disable_anthropic_context_beta_header(
                                &mut runtime_model.default_headers,
                            ) > 0
                            {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if !thinking_signature_retry_attempted
                            && error_is_thinking_signature(&error)
                        {
                            thinking_signature_retry_attempted = true;
                            if strip_reasoning_details_from_messages(&mut messages) > 0 {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if !llama_cpp_schema_retry_attempted
                            && error_is_llama_cpp_grammar_pattern(&error)
                        {
                            llama_cpp_schema_retry_attempted = true;
                            if strip_pattern_and_format_from_tools(&mut tools) > 0 {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if error_is_long_context_tier(&error) && compactions < max_compactions {
                            let old_ctx = context_length;
                            context_length = context_length.min(ANTHROPIC_STANDARD_CONTEXT_TIER);
                            compactions += 1;
                            if old_ctx != context_length {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if loaded.config.compression.enabled
                            && (error_is_context_budget_related(&error)
                                || error_is_generic_large_session_context_overflow(
                                    &error,
                                    estimate_messages_tokens_rough(&messages),
                                    context_length,
                                    messages.len().saturating_sub(1),
                                ))
                            && compactions < max_compactions
                        {
                            if let Some(available_out) =
                                parse_available_output_tokens_from_error(&error)
                            {
                                let safe_out = available_out.saturating_sub(64).max(1);
                                max_output_tokens = Some(safe_out);
                                compactions += 1;
                                continue;
                            }

                            let old_ctx = context_length;
                            if let Some(parsed_limit) = parse_context_length_hint(&error) {
                                context_length = context_length.min(parsed_limit);
                            } else if !is_minimax_delta_only_overflow(
                                &runtime_model.provider,
                                &runtime_model.base_url,
                                &error,
                            ) && let Some(next_tier) = get_next_probe_tier(context_length)
                            {
                                context_length = next_tier;
                            }
                            compactions += 1;
                            if let Some(compacted) = compact_messages_for_context_budget(
                                &messages,
                                loaded.config.compression.protect_last_n,
                                loaded.config.compression.target_ratio,
                            ) {
                                messages = compacted;
                                max_output_tokens = None;
                                if let Some(store) = session_store
                                    && let Some(active_session_id) = session_id.clone()
                                {
                                    let rotated = rotate_compressed_session(
                                        store,
                                        &active_session_id,
                                        &runtime_model,
                                        &system_prompt,
                                        &messages,
                                    )?;
                                    tool_runtime =
                                        tool_runtime.with_current_session_id(Some(rotated.clone()));
                                    session_id = Some(rotated);
                                }
                                continue;
                            }
                            if context_length < old_ctx && compactions < max_compactions {
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        let eager_fallback = error_should_immediately_try_fallback(&error);
                        let can_retry = error_is_retryable(&error);
                        if runtime_model.provider == "nous"
                            && error_is_rate_limit_like(&error)
                            && nous_rate_limit_looks_genuine(&error)
                        {
                            record_nous_rate_limit_state(self, &error);
                        }
                        let suppress_rate_limit_fallback = error_is_rate_limit_like(&error)
                            && credential_pool_may_recover_rate_limit(self, &runtime_model);
                        if (eager_fallback || request_retries >= max_retries)
                            && !suppress_rate_limit_fallback
                        {
                            if let Some(fallback_runtime) = activate_next_fallback_runtime(
                                self,
                                loaded,
                                &fallback_chain,
                                &mut fallback_index,
                            )? {
                                runtime_model = fallback_runtime;
                                if let Some(store) = session_store
                                    && let Some(active_session_id) = session_id.as_deref()
                                {
                                    let _ = store.update_session_runtime(
                                        active_session_id,
                                        Some(&runtime_model.model),
                                        Some(&model_runtime_metadata(&runtime_model)),
                                    );
                                }
                                request_retries = 0;
                                compactions = 0;
                                max_output_tokens = None;
                                continue;
                            }
                        }

                        if can_retry && request_retries < max_retries {
                            request_retries += 1;
                            thread::sleep(retry_backoff(request_retries));
                            continue;
                        }
                        return Err(error);
                    }
                }
            };
            let NormalizedAssistantResponse {
                content: assistant_content,
                tool_calls: pending_tool_calls,
                finish_reason,
                reasoning,
                reasoning_content,
                reasoning_details,
                codex_reasoning_items,
                codex_message_items,
            } = response;

            if pending_tool_calls.is_empty() {
                if let Some(final_response) = visible_assistant_text(assistant_content.as_deref()) {
                    if let Some(store) = session_store
                        && let Some(session_id) = session_id.as_deref()
                    {
                        let _ = store.append_message(
                            session_id,
                            &MessageAppend {
                                role: "assistant".to_string(),
                                content: Some(Value::String(final_response.clone())),
                                tool_call_id: None,
                                tool_calls: None,
                                tool_name: None,
                                token_count: None,
                                finish_reason: finish_reason.clone(),
                                reasoning: reasoning.clone(),
                                reasoning_content: reasoning_content.clone(),
                                reasoning_details: reasoning_details.clone(),
                                codex_reasoning_items: codex_reasoning_items.clone(),
                                codex_message_items: codex_message_items.clone(),
                            },
                        );
                    }
                    return Ok(AgentTurnResult {
                        final_response,
                        api_calls,
                        tool_calls,
                        model: runtime_model.model.clone(),
                        provider: runtime_model.provider.clone(),
                        base_url: runtime_model.base_url.clone(),
                        session_id,
                    });
                }

                let response = NormalizedAssistantResponse {
                    content: assistant_content,
                    tool_calls: pending_tool_calls,
                    finish_reason,
                    reasoning,
                    reasoning_content,
                    reasoning_details,
                    codex_reasoning_items,
                    codex_message_items,
                };
                if response_has_reasoning_signal(&response) && thinking_prefill_retries < 2 {
                    thinking_prefill_retries += 1;
                    messages.push(assistant_response_message(&response, None));
                    if let Some(store) = session_store
                        && let Some(session_id) = session_id.as_deref()
                    {
                        let _ = store.append_message(
                            session_id,
                            &assistant_response_append(&response, None),
                        );
                    }
                    continue;
                }

                if empty_content_retries < 3 {
                    empty_content_retries += 1;
                    continue;
                }

                if let Some(fallback_runtime) = activate_next_fallback_runtime(
                    self,
                    loaded,
                    &fallback_chain,
                    &mut fallback_index,
                )? {
                    runtime_model = fallback_runtime;
                    if let Some(store) = session_store
                        && let Some(active_session_id) = session_id.as_deref()
                    {
                        let _ = store.update_session_runtime(
                            active_session_id,
                            Some(&runtime_model.model),
                            Some(&model_runtime_metadata(&runtime_model)),
                        );
                    }
                    empty_content_retries = 0;
                    thinking_prefill_retries = 0;
                    continue;
                }

                return Err(HermesError::State {
                    action: "parsing assistant response",
                    detail: "Assistant response had no text content.".to_string(),
                });
            }
            empty_content_retries = 0;
            thinking_prefill_retries = 0;

            let normalized_tool_calls = pending_tool_calls
                .iter()
                .map(|tool_call| {
                    json!({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments_raw,
                        }
                    })
                })
                .collect::<Vec<_>>();

            let response = NormalizedAssistantResponse {
                content: assistant_content,
                tool_calls: pending_tool_calls,
                finish_reason,
                reasoning,
                reasoning_content,
                reasoning_details,
                codex_reasoning_items,
                codex_message_items,
            };
            messages.push(assistant_response_message(
                &response,
                Some(normalized_tool_calls.clone()),
            ));
            if let Some(store) = session_store
                && let Some(session_id) = session_id.as_deref()
            {
                let _ = store.append_message(
                    session_id,
                    &assistant_response_append(&response, Some(normalized_tool_calls.clone())),
                );
            }

            for tool_call in response.tool_calls {
                tool_calls += 1;
                let result = dispatch_tool(&tool_call.name, tool_call.json, &tool_runtime);
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }));
                if let Some(store) = session_store
                    && let Some(session_id) = session_id.as_deref()
                {
                    let _ = store.append_message(
                        session_id,
                        &MessageAppend {
                            role: "tool".to_string(),
                            content: Some(Value::String(result)),
                            tool_call_id: Some(tool_call.id),
                            tool_calls: None,
                            tool_name: Some(tool_call.name),
                            token_count: None,
                            finish_reason: None,
                            reasoning: None,
                            reasoning_content: None,
                            reasoning_details: None,
                            codex_reasoning_items: None,
                            codex_message_items: None,
                        },
                    );
                }
            }
        }

        Err(HermesError::State {
            action: "running agent turn",
            detail: format!(
                "Reached max_turns ({}) without a final assistant response.",
                loaded.config.agent.max_turns
            ),
        })
    }
}

fn content_has_meaningful_user_input(content: &Value) -> bool {
    match content {
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(parts) => parts.iter().any(content_part_is_meaningful),
        Value::Object(object) => content_part_is_meaningful(&Value::Object(object.clone())),
        _ => false,
    }
}

fn content_part_is_meaningful(part: &Value) -> bool {
    let Some(object) = part.as_object() else {
        return false;
    };
    match object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "text" | "input_text" | "output_text" => object
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.trim().is_empty()),
        "image_url" => object
            .get("image_url")
            .and_then(|value| match value {
                Value::String(url) => Some(url.as_str()),
                Value::Object(image) => image.get("url").and_then(Value::as_str),
                _ => None,
            })
            .is_some_and(|url| !url.trim().is_empty()),
        "input_image" => object
            .get("image_url")
            .and_then(Value::as_str)
            .is_some_and(|url| !url.trim().is_empty()),
        _ => false,
    }
}

fn model_runtime_metadata(runtime_model: &crate::ModelRuntimeConfig) -> Value {
    json!({
        "provider": runtime_model.provider,
        "base_url": runtime_model.base_url,
        "api_mode": runtime_model.api_mode,
    })
}

fn resolve_fallback_chain(loaded: &LoadedConfig) -> Vec<crate::ModelOverrides> {
    let mut chain = loaded
        .config
        .fallback_providers
        .iter()
        .filter_map(|entry| {
            fallback_entry_to_overrides(
                &entry.provider,
                &entry.model,
                Some(&entry.base_url),
                Some(&entry.api_key),
                Some(&entry.api_mode),
                Some(&entry.key_env),
            )
        })
        .collect::<Vec<_>>();
    if !chain.is_empty() {
        return chain;
    }

    let Some(value) = loaded.cfg_get(&["fallback_model"]) else {
        return Vec::new();
    };
    match value {
        serde_yaml::Value::Sequence(entries) => {
            for entry in entries {
                if let Some(overrides) = fallback_overrides_from_yaml_value(entry) {
                    chain.push(overrides);
                }
            }
        }
        other => {
            if let Some(overrides) = fallback_overrides_from_yaml_value(other) {
                chain.push(overrides);
            }
        }
    }
    chain
}

fn fallback_overrides_from_yaml_value(value: &serde_yaml::Value) -> Option<crate::ModelOverrides> {
    let object = value.as_mapping()?;
    fallback_entry_to_overrides(
        object
            .get(serde_yaml::Value::String("provider".to_string()))
            .and_then(serde_yaml::Value::as_str)?,
        object
            .get(serde_yaml::Value::String("model".to_string()))
            .and_then(serde_yaml::Value::as_str)?,
        object
            .get(serde_yaml::Value::String("base_url".to_string()))
            .and_then(serde_yaml::Value::as_str),
        object
            .get(serde_yaml::Value::String("api_key".to_string()))
            .and_then(serde_yaml::Value::as_str),
        object
            .get(serde_yaml::Value::String("api_mode".to_string()))
            .and_then(serde_yaml::Value::as_str),
        object
            .get(serde_yaml::Value::String("key_env".to_string()))
            .and_then(serde_yaml::Value::as_str),
    )
}

fn fallback_entry_to_overrides(
    provider: &str,
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
    api_mode: Option<&str>,
    key_env: Option<&str>,
) -> Option<crate::ModelOverrides> {
    let provider = non_empty_trimmed(provider)?;
    let model = non_empty_trimmed(model)?;
    let api_key = api_key.and_then(non_empty_trimmed).or_else(|| {
        key_env
            .and_then(non_empty_trimmed)
            .and_then(|name| env::var(name).ok())
    });
    Some(crate::ModelOverrides {
        model: Some(model),
        provider: Some(provider),
        base_url: base_url.and_then(non_empty_trimmed),
        api_key,
        api_mode: api_mode.and_then(non_empty_trimmed),
    })
}

fn activate_next_fallback_runtime(
    context: &HermesContext,
    loaded: &LoadedConfig,
    chain: &[crate::ModelOverrides],
    index: &mut usize,
) -> Result<Option<crate::ModelRuntimeConfig>, HermesError> {
    while *index < chain.len() {
        let overrides = chain[*index].clone();
        *index += 1;
        if let Ok(runtime) = context.resolve_model_runtime(loaded, &overrides) {
            return Ok(Some(runtime));
        }
    }
    Ok(None)
}

fn error_http_status(error: &HermesError) -> Option<u16> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    let suffix = detail.strip_prefix("HTTP ")?;
    suffix
        .split_whitespace()
        .next()
        .and_then(|value| value.trim().trim_end_matches(':').parse::<u16>().ok())
}

fn error_code_lower(error: &HermesError) -> Option<String> {
    let body = error_http_body_json(error)?;
    let code = body
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error_obj| {
            error_obj
                .get("code")
                .or_else(|| error_obj.get("type"))
                .and_then(|value| match value {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Number(number) => Some(number.to_string()),
                    _ => None,
                })
        })
        .or_else(|| {
            body.get("code")
                .or_else(|| body.get("error_code"))
                .and_then(|value| match value {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Number(number) => Some(number.to_string()),
                    _ => None,
                })
        })?;
    (!code.is_empty()).then(|| code.to_ascii_lowercase())
}

fn error_http_body_json(error: &HermesError) -> Option<Value> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    let body = detail.split_once(": ")?.1.trim();
    serde_json::from_str::<Value>(body).ok()
}

fn error_is_retryable(error: &HermesError) -> bool {
    if error_is_billing_exhaustion(error)
        || error_is_model_not_found(error)
        || error_is_provider_policy_blocked(error)
        || error_is_auth_like(error)
    {
        return false;
    }
    if let Some(status) = error_http_status(error) {
        if matches!(status, 400 | 402) {
            return error_is_rate_limit_like(error);
        }
        return matches!(
            status,
            404 | 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504 | 529
        );
    }
    let HermesError::State { detail, .. } = error else {
        return false;
    };
    let lowered = detail.to_ascii_lowercase();
    lowered.contains("timed out")
        || lowered.contains("timeout")
        || lowered.contains("connection reset")
        || lowered.contains("connection refused")
        || lowered.contains("temporary")
        || error_is_rate_limit_like(error)
}

fn error_should_immediately_try_fallback(error: &HermesError) -> bool {
    if error_is_provider_policy_blocked(error) {
        return false;
    }
    error_is_rate_limit_like(error)
        || error_is_billing_exhaustion(error)
        || error_is_model_not_found(error)
        || error_is_auth_like(error)
        || error_is_generic_client_format_error(error)
}

fn error_blocks_provider_fallback(error: &HermesError) -> bool {
    error_is_provider_policy_blocked(error)
}

fn error_is_auth_like(error: &HermesError) -> bool {
    matches!(error_http_status(error), Some(401))
        || (matches!(error_http_status(error), Some(403)) && !error_is_billing_exhaustion(error))
}

fn should_retry_with_reresolved_auth(
    runtime_model: &crate::ModelRuntimeConfig,
    error: &HermesError,
) -> bool {
    matches!(error_http_status(error), Some(401))
        && matches!(
            runtime_model.provider.as_str(),
            "copilot"
                | "nous"
                | "openai-codex"
                | "google-gemini-cli"
                | "qwen-oauth"
                | "minimax-oauth"
        )
        || (matches!(error_http_status(error), Some(401))
            && runtime_model.provider == "anthropic"
            && runtime_model.auth_type == "oauth_external")
}

fn refresh_runtime_auth_for_retry(
    ctx: &HermesContext,
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    match runtime_model.provider.as_str() {
        "copilot" => {
            let _ = crate::clear_provider_runtime_cache(&runtime_model.provider);
            true
        }
        "nous" => crate::force_refresh_nous_runtime_credentials(
            &ctx.hermes_home(),
            nous_retry_min_key_ttl_seconds(),
            nous_retry_timeout_seconds(),
        )
        .is_ok(),
        "anthropic" => crate::force_refresh_anthropic_token()
            .ok()
            .flatten()
            .is_some(),
        "google-gemini-cli" => {
            crate::force_refresh_google_gemini_runtime_credentials(&ctx.hermes_home()).is_ok()
        }
        "qwen-oauth" => crate::force_refresh_qwen_runtime_credentials().is_ok(),
        "minimax-oauth" => {
            crate::force_refresh_minimax_oauth_runtime_credentials(&ctx.hermes_home()).is_ok()
        }
        "openai-codex" => crate::force_refresh_codex_access_token(&ctx.hermes_home()).is_ok(),
        _ => false,
    }
}

fn error_supports_credential_pool_rotation(error: &HermesError) -> bool {
    error_is_rate_limit_like(error)
        || error_is_billing_exhaustion(error)
        || error_is_auth_like(error)
}

fn credential_pool_prefers_auth_rotation(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    matches!(
        runtime_model.provider.as_str(),
        "google-gemini-cli" | "qwen-oauth" | "minimax-oauth"
    )
}

fn credential_pool_supports_current_entry_auth_refresh(
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    matches!(runtime_model.provider.as_str(), "nous" | "openai-codex")
}

fn credential_pool_entry_from_value(value: &Value) -> Option<CredentialPoolEntry> {
    let object = value.as_object()?.clone();
    Some(CredentialPoolEntry {
        priority: object.get("priority").and_then(Value::as_i64).unwrap_or(0),
        access_token: object
            .get("access_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        agent_key: object
            .get("agent_key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        base_url: object
            .get("base_url")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        inference_base_url: object
            .get("inference_base_url")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string(),
        last_status: object
            .get("last_status")
            .and_then(Value::as_str)
            .map(|value| value.trim().to_string()),
        last_error_reset_at: object.get("last_error_reset_at").and_then(value_to_f64),
    })
}

fn credential_pool_runtime_api_key(entry: &CredentialPoolEntry, provider: &str) -> String {
    if provider == "nous" && !entry.agent_key.trim().is_empty() {
        return entry.agent_key.clone();
    }
    entry.access_token.clone()
}

fn credential_pool_runtime_base_url(entry: &CredentialPoolEntry, provider: &str) -> String {
    if provider == "nous" && !entry.inference_base_url.trim().is_empty() {
        return entry.inference_base_url.clone();
    }
    if !entry.base_url.trim().is_empty() {
        return entry.base_url.clone();
    }
    entry.inference_base_url.clone()
}

fn credential_pool_entry_matches_runtime(
    entry: &CredentialPoolEntry,
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    let entry_key = credential_pool_runtime_api_key(entry, &runtime_model.provider);
    !entry_key.trim().is_empty() && entry_key == runtime_model.api_key
}

fn credential_pool_alternative_available(
    ctx: &HermesContext,
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    let auth_path = ctx.hermes_home().join("auth.json");
    let Ok(raw) = fs::read_to_string(&auth_path) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    let Some(entries) = parsed
        .get("credential_pool")
        .and_then(Value::as_object)
        .and_then(|pool| pool.get(&runtime_model.provider))
        .and_then(Value::as_array)
    else {
        return false;
    };
    let now = unix_ts_seconds_f64();
    let mut matched_current = false;
    let mut available_alternative = false;
    for entry in entries.iter().filter_map(credential_pool_entry_from_value) {
        if entry.last_status.as_deref() == Some(CREDENTIAL_POOL_EXHAUSTED_STATUS)
            && entry
                .last_error_reset_at
                .is_some_and(|reset_at| reset_at > now)
        {
            continue;
        }
        if credential_pool_entry_matches_runtime(&entry, runtime_model) {
            matched_current = true;
            continue;
        }
        if !credential_pool_runtime_api_key(&entry, &runtime_model.provider)
            .trim()
            .is_empty()
        {
            available_alternative = true;
        }
    }
    matched_current && available_alternative
}

fn credential_pool_has_current_runtime(
    ctx: &HermesContext,
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    let auth_path = ctx.hermes_home().join("auth.json");
    let Ok(raw) = fs::read_to_string(&auth_path) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    parsed
        .get("credential_pool")
        .and_then(Value::as_object)
        .and_then(|pool| pool.get(&runtime_model.provider))
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries.iter().any(|value| {
                credential_pool_entry_from_value(value).is_some_and(|entry| {
                    credential_pool_entry_matches_runtime(&entry, runtime_model)
                })
            })
        })
}

fn credential_pool_may_recover_rate_limit(
    ctx: &HermesContext,
    runtime_model: &crate::ModelRuntimeConfig,
) -> bool {
    if matches!(runtime_model.provider.as_str(), "google-gemini-cli")
        || runtime_model.base_url.starts_with("cloudcode-pa://")
    {
        return false;
    }
    credential_pool_alternative_available(ctx, runtime_model)
}

fn try_rotate_credential_pool_runtime(
    ctx: &HermesContext,
    loaded: &LoadedConfig,
    runtime_model: &crate::ModelRuntimeConfig,
    status_code: Option<u16>,
) -> Result<Option<crate::ModelRuntimeConfig>, HermesError> {
    let auth_path = ctx.hermes_home().join("auth.json");
    if !auth_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&auth_path).map_err(|source| HermesError::Io {
        action: "reading",
        path: auth_path.clone(),
        source,
    })?;
    let mut parsed = serde_json::from_str::<Value>(&raw).map_err(|error| HermesError::State {
        action: "parsing auth store",
        detail: format!("{}: {error}", auth_path.display()),
    })?;
    let Some(entries) = parsed
        .get_mut("credential_pool")
        .and_then(Value::as_object_mut)
        .and_then(|pool| pool.get_mut(&runtime_model.provider))
        .and_then(Value::as_array_mut)
    else {
        return Ok(None);
    };

    let now = unix_ts_seconds_f64();
    let mut pool_entries = entries
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            credential_pool_entry_from_value(value).map(|entry| (index, entry))
        })
        .collect::<Vec<_>>();
    pool_entries.sort_by_key(|(_, entry)| entry.priority);

    let Some(current_index) = pool_entries
        .iter()
        .find(|(_, entry)| credential_pool_entry_matches_runtime(entry, runtime_model))
        .map(|(index, _)| *index)
    else {
        return Ok(None);
    };

    for value in entries.iter_mut() {
        if let Some(object) = value.as_object_mut()
            && object.get("last_status").and_then(Value::as_str)
                == Some(CREDENTIAL_POOL_EXHAUSTED_STATUS)
            && object
                .get("last_error_reset_at")
                .and_then(value_to_f64)
                .is_some_and(|reset_at| reset_at <= now)
        {
            object.insert("last_status".to_string(), Value::Null);
            object.insert("last_status_at".to_string(), Value::Null);
            object.insert("last_error_code".to_string(), Value::Null);
            object.insert("last_error_reason".to_string(), Value::Null);
            object.insert("last_error_message".to_string(), Value::Null);
            object.insert("last_error_reset_at".to_string(), Value::Null);
        }
    }

    if let Some(object) = entries
        .get_mut(current_index)
        .and_then(Value::as_object_mut)
    {
        object.insert(
            "last_status".to_string(),
            Value::String(CREDENTIAL_POOL_EXHAUSTED_STATUS.to_string()),
        );
        object.insert("last_status_at".to_string(), Value::from(now));
        object.insert(
            "last_error_code".to_string(),
            status_code.map(Value::from).unwrap_or(Value::Null),
        );
        object.insert(
            "last_error_reset_at".to_string(),
            Value::from(now + CREDENTIAL_POOL_EXHAUSTED_TTL_SECONDS),
        );
    }

    let next_entry = entries
        .iter()
        .filter_map(credential_pool_entry_from_value)
        .filter(|entry| !credential_pool_entry_matches_runtime(entry, runtime_model))
        .filter(|entry| {
            if entry.last_status.as_deref() == Some(CREDENTIAL_POOL_EXHAUSTED_STATUS)
                && entry
                    .last_error_reset_at
                    .is_some_and(|reset_at| reset_at > now)
            {
                return false;
            }
            !credential_pool_runtime_api_key(entry, &runtime_model.provider)
                .trim()
                .is_empty()
        })
        .min_by_key(|entry| entry.priority);

    let payload = serde_json::to_string_pretty(&parsed).map_err(|error| HermesError::State {
        action: "serializing auth store",
        detail: error.to_string(),
    })?;
    fs::write(&auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.clone(),
        source,
    })?;

    let Some(next_entry) = next_entry else {
        return Ok(None);
    };
    let next_api_key = credential_pool_runtime_api_key(&next_entry, &runtime_model.provider);
    if next_api_key.trim().is_empty() {
        return Ok(None);
    }
    let next_base_url = credential_pool_runtime_base_url(&next_entry, &runtime_model.provider);
    let runtime_overrides = ModelOverrides {
        model: Some(runtime_model.model.clone()),
        provider: Some(runtime_model.provider.clone()),
        base_url: Some(if next_base_url.trim().is_empty() {
            runtime_model.base_url.clone()
        } else {
            next_base_url
        }),
        api_key: Some(next_api_key),
        api_mode: Some(runtime_model.api_mode.clone()),
    };
    ctx.resolve_model_runtime(loaded, &runtime_overrides)
        .map(Some)
}

fn try_refresh_current_credential_pool_runtime(
    ctx: &HermesContext,
    runtime_model: &crate::ModelRuntimeConfig,
) -> Result<Option<crate::ModelRuntimeConfig>, HermesError> {
    if !credential_pool_supports_current_entry_auth_refresh(runtime_model) {
        return Ok(None);
    }

    let auth_path = ctx.hermes_home().join("auth.json");
    if !auth_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&auth_path).map_err(|source| HermesError::Io {
        action: "reading",
        path: auth_path.clone(),
        source,
    })?;
    let mut parsed = serde_json::from_str::<Value>(&raw).map_err(|error| HermesError::State {
        action: "parsing auth store",
        detail: format!("{}: {error}", auth_path.display()),
    })?;
    let refreshed_value = {
        let Some(entries) = parsed
            .get_mut("credential_pool")
            .and_then(Value::as_object_mut)
            .and_then(|pool| pool.get_mut(&runtime_model.provider))
            .and_then(Value::as_array_mut)
        else {
            return Ok(None);
        };
        let Some(current_index) = entries.iter().position(|value| {
            credential_pool_entry_from_value(value)
                .is_some_and(|entry| credential_pool_entry_matches_runtime(&entry, runtime_model))
        }) else {
            return Ok(None);
        };
        let Some(current_entry) = entries
            .get(current_index)
            .and_then(Value::as_object)
            .cloned()
        else {
            return Ok(None);
        };

        let refreshed_entry = match runtime_model.provider.as_str() {
            "nous" => crate::force_refresh_nous_credential_pool_entry(
                &current_entry,
                nous_retry_min_key_ttl_seconds(),
                nous_retry_timeout_seconds(),
            ),
            "openai-codex" => crate::force_refresh_codex_credential_pool_entry(&current_entry),
            _ => return Ok(None),
        };
        let Ok(refreshed_entry) = refreshed_entry else {
            return Ok(None);
        };
        entries[current_index] = Value::Object(refreshed_entry);
        entries[current_index].clone()
    };

    let payload = serde_json::to_string_pretty(&parsed).map_err(|error| HermesError::State {
        action: "serializing auth store",
        detail: error.to_string(),
    })?;
    fs::write(&auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.clone(),
        source,
    })?;

    let refreshed =
        credential_pool_entry_from_value(&refreshed_value).ok_or_else(|| HermesError::State {
            action: "refreshing credential pool entry",
            detail: "Updated credential pool entry could not be parsed.".to_string(),
        })?;
    let next_api_key = credential_pool_runtime_api_key(&refreshed, &runtime_model.provider);
    if next_api_key.trim().is_empty() {
        return Ok(None);
    }
    let next_base_url = credential_pool_runtime_base_url(&refreshed, &runtime_model.provider);
    let mut updated_runtime = runtime_model.clone();
    updated_runtime.api_key = next_api_key;
    if !next_base_url.trim().is_empty() {
        updated_runtime.base_url = next_base_url;
    }
    Ok(Some(updated_runtime))
}

fn nous_retry_min_key_ttl_seconds() -> i64 {
    env::var("HERMES_NOUS_MIN_KEY_TTL_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<i64>().ok())
        .unwrap_or(1800)
}

fn nous_retry_timeout_seconds() -> f64 {
    env::var("HERMES_NOUS_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .unwrap_or(15.0)
}

fn error_is_billing_exhaustion(error: &HermesError) -> bool {
    if error_code_lower(error).is_some_and(|code| {
        matches!(
            code.as_str(),
            "insufficient_quota" | "billing_not_active" | "payment_required"
        )
    }) {
        return true;
    }
    let message = error_message_lower(error).unwrap_or_default();
    let status = error_http_status(error);
    if matches!(status, Some(403))
        && (message.contains("key limit exceeded") || message.contains("spending limit"))
    {
        return true;
    }
    if matches!(status, Some(402)) {
        let has_usage_limit = contains_any(&message, USAGE_LIMIT_PATTERNS);
        let has_transient_signal = contains_any(&message, USAGE_LIMIT_TRANSIENT_SIGNALS);
        return !(has_usage_limit && has_transient_signal);
    }
    contains_any(&message, BILLING_PATTERNS)
}

fn error_is_rate_limit_like(error: &HermesError) -> bool {
    if matches!(error_http_status(error), Some(429)) {
        return true;
    }
    if error_code_lower(error).is_some_and(|code| {
        matches!(
            code.as_str(),
            "resource_exhausted" | "throttled" | "rate_limit_exceeded"
        )
    }) {
        return true;
    }
    let message = error_message_lower(error).unwrap_or_default();
    if matches!(error_http_status(error), Some(402)) {
        let has_usage_limit = contains_any(&message, USAGE_LIMIT_PATTERNS);
        let has_transient_signal = contains_any(&message, USAGE_LIMIT_TRANSIENT_SIGNALS);
        return has_usage_limit && has_transient_signal;
    }
    contains_any(&message, RATE_LIMIT_PATTERNS)
}

fn error_is_model_not_found(error: &HermesError) -> bool {
    if error_code_lower(error).is_some_and(|code| {
        matches!(
            code.as_str(),
            "model_not_found" | "model_not_available" | "invalid_model"
        )
    }) {
        return true;
    }
    let message = error_message_lower(error).unwrap_or_default();
    contains_any(&message, MODEL_NOT_FOUND_PATTERNS)
}

fn error_is_provider_policy_blocked(error: &HermesError) -> bool {
    let message = error_message_lower(error).unwrap_or_default();
    contains_any(&message, PROVIDER_POLICY_BLOCKED_PATTERNS)
}

fn contains_any(haystack: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| haystack.contains(pattern))
}

fn error_is_context_budget_related(error: &HermesError) -> bool {
    if matches!(error_http_status(error), Some(413)) {
        return true;
    }
    if error_code_lower(error).is_some_and(|code| {
        matches!(
            code.as_str(),
            "context_length_exceeded" | "max_tokens_exceeded"
        )
    }) {
        return true;
    }
    let HermesError::State { detail, .. } = error else {
        return false;
    };
    let lowered = detail.to_ascii_lowercase();
    if lowered.contains("max_tokens")
        && (lowered.contains("available_tokens")
            || lowered.contains("available tokens")
            || lowered.contains("context_window"))
    {
        return true;
    }
    (lowered.contains("context") || lowered.contains("prompt"))
        && (lowered.contains("too long")
            || lowered.contains("too large")
            || lowered.contains("exceed")
            || lowered.contains("maximum")
            || lowered.contains("limit"))
}

fn error_is_image_too_large(error: &HermesError) -> bool {
    if error_code_lower(error).is_some_and(|code| code == "image_too_large") {
        return true;
    }
    let message = error_message_lower(error).unwrap_or_default();
    contains_any(&message, IMAGE_TOO_LARGE_PATTERNS)
}

fn error_is_generic_large_session_context_overflow(
    error: &HermesError,
    approx_tokens: u64,
    context_length: u64,
    num_messages: usize,
) -> bool {
    if !matches!(error_http_status(error), Some(400)) {
        return false;
    }
    if error_is_context_budget_related(error)
        || error_is_image_too_large(error)
        || error_is_provider_policy_blocked(error)
        || error_is_model_not_found(error)
        || error_is_rate_limit_like(error)
        || error_is_billing_exhaustion(error)
    {
        return false;
    }

    let body = error_http_body_json(error);
    let err_body_msg = body
        .as_ref()
        .and_then(|body| {
            body.get("error")
                .and_then(Value::as_object)
                .and_then(|error_obj| error_obj.get("message"))
                .and_then(Value::as_str)
                .or_else(|| body.get("message").and_then(Value::as_str))
        })
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let is_generic = err_body_msg.len() < 30 || err_body_msg == "error";
    let is_large = approx_tokens > context_length.saturating_mul(4) / 10
        || (context_length <= 256_000 && (approx_tokens > 80_000 || num_messages > 80));
    is_generic && is_large
}

fn error_is_generic_client_format_error(error: &HermesError) -> bool {
    let Some(status) = error_http_status(error) else {
        return false;
    };
    if !(400..500).contains(&status) {
        return false;
    }
    if matches!(status, 402 | 404 | 408 | 409 | 413 | 425 | 429) {
        return false;
    }
    !error_is_context_budget_related(error) && !error_is_image_too_large(error)
}

fn error_is_thinking_signature(error: &HermesError) -> bool {
    matches!(error_http_status(error), Some(400))
        && error_message_lower(error)
            .is_some_and(|message| message.contains("signature") && message.contains("thinking"))
}

fn error_is_oauth_long_context_beta_forbidden(error: &HermesError) -> bool {
    matches!(error_http_status(error), Some(400))
        && error_message_lower(error).is_some_and(|message| {
            message.contains("long context beta") && message.contains("not yet available")
        })
}

fn error_is_llama_cpp_grammar_pattern(error: &HermesError) -> bool {
    matches!(error_http_status(error), Some(400))
        && error_message_lower(error).is_some_and(|message| {
            message.contains("error parsing grammar")
                || message.contains("json-schema-to-grammar")
                || (message.contains("unable to generate parser") && message.contains("template"))
        })
}

fn error_is_long_context_tier(error: &HermesError) -> bool {
    matches!(error_http_status(error), Some(429))
        && error_message_lower(error)
            .is_some_and(|msg| msg.contains("extra usage") && msg.contains("long context"))
}

fn error_message_lower(error: &HermesError) -> Option<String> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    Some(detail.to_ascii_lowercase())
}

fn get_next_probe_tier(current_length: u64) -> Option<u64> {
    for tier in CONTEXT_PROBE_TIERS {
        if *tier < current_length {
            return Some(*tier);
        }
    }
    None
}

fn parse_context_length_hint(error: &HermesError) -> Option<u64> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    let lowered = detail.to_ascii_lowercase();
    if !lowered.contains("context") && !lowered.contains("prompt") {
        return None;
    }

    let normalized = detail.replace(',', "");
    for needle in [
        "context window limit",
        "context length limit",
        "maximum context length",
        "max context length",
        "context window",
        "token limit",
        " limit ",
    ] {
        if let Some(value) = parse_first_number_after(&normalized, needle) {
            return Some(value);
        }
    }

    let mut best: Option<u64> = None;
    let mut digits = String::new();
    for ch in normalized.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        if !digits.is_empty() {
            if let Ok(value) = digits.parse::<u64>()
                && value >= 1024
            {
                best = Some(best.map(|current| current.max(value)).unwrap_or(value));
            }
            digits.clear();
        }
    }
    if !digits.is_empty()
        && let Ok(value) = digits.parse::<u64>()
        && value >= 1024
    {
        best = Some(best.map(|current| current.max(value)).unwrap_or(value));
    }
    best
}

fn parse_first_number_after(text: &str, needle: &str) -> Option<u64> {
    let lower = text.to_ascii_lowercase();
    let start = lower.find(needle)?;
    let suffix = &text[start + needle.len()..];
    let mut digits = String::new();
    let mut seen_digit = false;
    for ch in suffix.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            seen_digit = true;
            continue;
        }
        if seen_digit {
            break;
        }
    }
    digits.parse::<u64>().ok().filter(|value| *value > 0)
}

fn parse_available_output_tokens_from_error(error: &HermesError) -> Option<u64> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    let lowered = detail.to_ascii_lowercase();
    if !(lowered.contains("max_tokens")
        && (lowered.contains("available_tokens") || lowered.contains("available tokens")))
    {
        return None;
    }
    for needle in ["available_tokens", "available tokens"] {
        if let Some(value) = parse_first_number_after(&lowered, needle) {
            return Some(value);
        }
    }
    let equals_suffix = lowered.rsplit('=').next()?.trim();
    let digits = equals_suffix
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse::<u64>().ok().filter(|value| *value > 0)
}

fn is_minimax_delta_only_overflow(provider: &str, base_url: &str, error: &HermesError) -> bool {
    let provider = provider.trim().to_ascii_lowercase();
    let base_url = base_url.trim().trim_end_matches('/').to_ascii_lowercase();
    let message = error_message_lower(error).unwrap_or_default();
    (provider == "minimax"
        || provider == "minimax-cn"
        || base_url.starts_with("https://api.minimax.io/anthropic")
        || base_url.starts_with("https://api.minimaxi.com/anthropic"))
        && message.contains("context window exceeds limit (")
}

fn retry_backoff(attempt: u64) -> Duration {
    Duration::from_millis((attempt.max(1) * 250).min(2_000))
}

fn should_compact_messages(
    messages: &[Value],
    context_length: Option<u64>,
    threshold: f64,
) -> bool {
    let Some(limit) = context_length else {
        return false;
    };
    if !(0.0..=1.0).contains(&threshold) {
        return false;
    }
    let budget = ((limit as f64) * threshold).floor() as u64;
    budget > 0 && estimate_messages_tokens_rough(messages) >= budget
}

fn estimate_messages_tokens_rough(messages: &[Value]) -> u64 {
    messages.iter().map(estimate_message_tokens_rough).sum()
}

fn estimate_message_tokens_rough(message: &Value) -> u64 {
    let Some(object) = message.as_object() else {
        return estimate_content_tokens_rough(message);
    };
    let role_cost = object
        .get("role")
        .and_then(Value::as_str)
        .map(|value| ((value.len() + 3) / 4) as u64)
        .unwrap_or(0);
    let content_cost = estimate_content_tokens_rough(object.get("content").unwrap_or(&Value::Null));
    let tool_cost = object
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    serde_json::to_string(call)
                        .ok()
                        .map(|text| ((text.len() + 3) / 4) as u64)
                        .unwrap_or(0)
                })
                .sum::<u64>()
        })
        .unwrap_or(0);
    role_cost + content_cost + tool_cost + 12
}

fn estimate_content_tokens_rough(content: &Value) -> u64 {
    match content {
        Value::Null => 0,
        Value::String(text) => ((text.chars().count() + 3) / 4) as u64,
        Value::Array(parts) => parts.iter().map(estimate_content_tokens_rough).sum(),
        Value::Object(object) => object
            .get("text")
            .map(estimate_content_tokens_rough)
            .unwrap_or_else(|| {
                serde_json::to_string(object)
                    .ok()
                    .map(|text| ((text.len() + 3) / 4) as u64)
                    .unwrap_or(0)
            }),
        other => serde_json::to_string(other)
            .ok()
            .map(|text| ((text.len() + 3) / 4) as u64)
            .unwrap_or(0),
    }
}

fn compact_messages_for_context_budget(
    messages: &[Value],
    protect_last_n: usize,
    target_ratio: f64,
) -> Option<Vec<Value>> {
    if messages.len() < 8 {
        return None;
    }

    let system_message = messages
        .first()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .cloned();
    let start_index = usize::from(system_message.is_some());
    let conversation = &messages[start_index..];
    if conversation.len() <= protect_last_n.saturating_add(3) {
        return None;
    }

    let keep_head = 2_usize.min(conversation.len());
    let keep_tail = protect_last_n.min(conversation.len().saturating_sub(keep_head + 1));
    let middle_start = keep_head;
    let middle_end = conversation.len().saturating_sub(keep_tail);
    if middle_start >= middle_end {
        return None;
    }
    let middle = &conversation[middle_start..middle_end];
    if middle.is_empty() {
        return None;
    }

    let char_budget = ((conversation
        .iter()
        .map(render_message_preview)
        .map(|line| line.chars().count())
        .sum::<usize>() as f64)
        * target_ratio.clamp(0.05, 0.9))
    .round() as usize;
    let char_budget = char_budget.clamp(240, 4_000);
    let mut consumed = 0_usize;
    let mut summary_lines = Vec::new();
    for message in middle {
        let preview = render_message_preview(message);
        if preview.is_empty() {
            continue;
        }
        let line = truncate_chars(&preview, 240);
        consumed += line.chars().count();
        summary_lines.push(line);
        if consumed >= char_budget {
            break;
        }
    }
    if summary_lines.is_empty() {
        summary_lines.push(format!(
            "{} messages were compacted to fit the context budget.",
            middle.len()
        ));
    }

    let mut compacted = Vec::new();
    if let Some(system_message) = system_message {
        compacted.push(system_message);
    }
    compacted.extend_from_slice(&conversation[..keep_head]);
    compacted.push(json!({
        "role": "assistant",
        "content": format!(
            "Context compacted to fit the context budget. Earlier messages summary:\n{}",
            summary_lines.join("\n")
        ),
    }));
    compacted.extend_from_slice(&conversation[middle_end..]);
    (compacted.len() < messages.len()).then_some(compacted)
}

fn render_message_preview(message: &Value) -> String {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("context")
        .to_ascii_lowercase();
    let content = render_message_content(message.get("content"));
    if content.is_empty() {
        if let Some(tool_name) = message.get("tool_name").and_then(Value::as_str) {
            return format!("{role}: tool={tool_name}");
        }
        if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
            let names = tool_calls
                .iter()
                .filter_map(|call| {
                    call.get("function")
                        .and_then(Value::as_object)
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                })
                .collect::<Vec<_>>();
            if !names.is_empty() {
                return format!("{role}: tool_calls={}", names.join(", "));
            }
        }
        return role;
    }
    format!("{role}: {}", truncate_chars(&content, 240))
}

fn render_message_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.split_whitespace().collect::<Vec<_>>().join(" "),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) => Some(text.as_str()),
                Value::Object(object) => object.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        Some(Value::Object(object)) => object
            .get("text")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        truncated.push_str("...");
    }
    truncated
}

fn strip_pattern_and_format_from_tools(tools: &mut [crate::ToolDefinition]) -> usize {
    let mut stripped = 0_usize;
    for tool in tools {
        if let Some(parameters) = tool.schema.get_mut("parameters") {
            stripped += strip_pattern_and_format_from_schema(parameters);
        }
    }
    stripped
}

fn strip_pattern_and_format_from_schema(schema: &mut Value) -> usize {
    match schema {
        Value::Array(items) => items
            .iter_mut()
            .map(strip_pattern_and_format_from_schema)
            .sum(),
        Value::Object(object) => {
            let mut stripped = 0_usize;
            if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
                for value in properties.values_mut() {
                    stripped += strip_pattern_and_format_from_schema(value);
                }
            }
            for key in ["$defs", "definitions"] {
                if let Some(definitions) = object.get_mut(key).and_then(Value::as_object_mut) {
                    for value in definitions.values_mut() {
                        stripped += strip_pattern_and_format_from_schema(value);
                    }
                }
            }
            for key in ["items", "additionalProperties"] {
                if let Some(value) = object.get_mut(key) {
                    stripped += strip_pattern_and_format_from_schema(value);
                }
            }
            for key in ["anyOf", "oneOf", "allOf"] {
                if let Some(values) = object.get_mut(key).and_then(Value::as_array_mut) {
                    for value in values {
                        stripped += strip_pattern_and_format_from_schema(value);
                    }
                }
            }
            if object.remove("pattern").is_some() {
                stripped += 1;
            }
            if object.remove("format").is_some() {
                stripped += 1;
            }
            stripped
        }
        _ => 0,
    }
}

fn strip_reasoning_details_from_messages(messages: &mut [Value]) -> usize {
    let mut stripped = 0_usize;
    for message in messages {
        if let Some(object) = message.as_object_mut()
            && object.remove("reasoning_details").is_some()
        {
            stripped += 1;
        }
    }
    stripped
}

fn disable_anthropic_context_beta_header(headers: &mut Vec<(String, String)>) -> usize {
    let mut removed = 0_usize;
    let mut retained = Vec::with_capacity(headers.len());
    for (name, value) in headers.drain(..) {
        if !name.eq_ignore_ascii_case("anthropic-beta") {
            retained.push((name, value));
            continue;
        }

        let kept_parts = value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .filter(|part| {
                let lower = part.to_ascii_lowercase();
                let is_1m = lower.contains("context-1m");
                if is_1m {
                    removed += 1;
                }
                !is_1m
            })
            .collect::<Vec<_>>();
        if !kept_parts.is_empty() {
            retained.push((name, kept_parts.join(", ")));
        }
    }
    *headers = retained;
    removed
}

fn try_shrink_image_parts_in_messages(messages: &mut [Value], target_bytes: usize) -> bool {
    let mut changed = false;
    for message in messages {
        changed |= shrink_image_parts_in_value(message, target_bytes);
    }
    changed
}

fn shrink_image_parts_in_value(value: &mut Value, target_bytes: usize) -> bool {
    match value {
        Value::Array(items) => items
            .iter_mut()
            .any(|item| shrink_image_parts_in_value(item, target_bytes)),
        Value::Object(object) => {
            let mut changed = false;
            let part_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match part_type {
                "image_url" => {
                    if let Some(image_url) = object.get_mut("image_url") {
                        match image_url {
                            Value::String(url) => {
                                if let Some(shrunk) = shrink_data_url_image(url, target_bytes) {
                                    *url = shrunk;
                                    changed = true;
                                }
                            }
                            Value::Object(image) => {
                                if let Some(url) =
                                    image.get_mut("url").and_then(|value| value.as_str())
                                    && let Some(shrunk) = shrink_data_url_image(url, target_bytes)
                                {
                                    image.insert("url".to_string(), Value::String(shrunk));
                                    changed = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "input_image" => {
                    if let Some(url) = object.get_mut("image_url").and_then(|value| value.as_str())
                        && let Some(shrunk) = shrink_data_url_image(url, target_bytes)
                    {
                        object.insert("image_url".to_string(), Value::String(shrunk));
                        changed = true;
                    }
                }
                _ => {}
            }
            for child in object.values_mut() {
                changed |= shrink_image_parts_in_value(child, target_bytes);
            }
            changed
        }
        _ => false,
    }
}

fn shrink_data_url_image(url: &str, target_bytes: usize) -> Option<String> {
    if !url.starts_with("data:") || url.len() <= target_bytes {
        return None;
    }
    let (header, encoded) = url.strip_prefix("data:")?.split_once(',')?;
    if !header.to_ascii_lowercase().contains(";base64") {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let image = image::load_from_memory(&bytes).ok()?;
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return None;
    }

    let qualities = [85_u8, 75, 65, 55, 45];
    let scales = [1.0_f32, 0.85, 0.72, 0.60, 0.50, 0.40];
    let mut best: Option<String> = None;
    for scale in scales {
        let next_width = ((width as f32) * scale).round().max(1.0) as u32;
        let next_height = ((height as f32) * scale).round().max(1.0) as u32;
        let resized = if next_width == width && next_height == height {
            image.clone()
        } else {
            image.resize(next_width, next_height, FilterType::Lanczos3)
        };
        let rgb = resized.to_rgb8();
        for quality in qualities {
            let mut output = Vec::new();
            let mut encoder = JpegEncoder::new_with_quality(&mut output, quality);
            if encoder
                .encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    image::ColorType::Rgb8.into(),
                )
                .is_err()
            {
                continue;
            }
            let candidate = format!(
                "data:image/jpeg;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(output)
            );
            if candidate.len() >= url.len() {
                continue;
            }
            if candidate.len() <= target_bytes {
                return Some(candidate);
            }
            let replace = best
                .as_ref()
                .map(|current| candidate.len() < current.len())
                .unwrap_or(true);
            if replace {
                best = Some(candidate);
            }
        }
    }
    best
}

fn rotate_compressed_session(
    store: &SessionStore,
    current_session_id: &str,
    runtime_model: &crate::ModelRuntimeConfig,
    system_prompt: &str,
    messages: &[Value],
) -> Result<String, HermesError> {
    let new_session_id = format!("rs_{:x}", unix_ts_nanos());
    let prior_title = store.get_session_title(current_session_id).ok().flatten();
    store.end_session(current_session_id, "compression")?;
    store.create_session(&SessionCreate {
        id: new_session_id.clone(),
        source: "rust-agent".to_string(),
        user_id: None,
        model: Some(runtime_model.model.clone()),
        model_config: Some(model_runtime_metadata(runtime_model)),
        system_prompt: Some(system_prompt.to_string()),
        parent_session_id: Some(current_session_id.to_string()),
    })?;
    if let Some(prior_title) = prior_title
        && let Ok(next_title) = store.get_next_title_in_lineage(&prior_title)
    {
        let _ = store.set_session_title(&new_session_id, &next_title);
    }
    for message in messages {
        let Some(role) = message.get("role").and_then(Value::as_str) else {
            continue;
        };
        if role == "system" {
            continue;
        }
        let _ = store.append_message(&new_session_id, &message_to_append(message)?);
    }
    Ok(new_session_id)
}

fn message_to_append(message: &Value) -> Result<MessageAppend, HermesError> {
    let object = message.as_object().ok_or_else(|| HermesError::State {
        action: "rotating compressed session",
        detail: "compressed message was not an object".to_string(),
    })?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| HermesError::State {
            action: "rotating compressed session",
            detail: "compressed message was missing a role".to_string(),
        })?;
    Ok(MessageAppend {
        role: role.to_string(),
        content: object.get("content").cloned(),
        tool_call_id: object
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        tool_calls: object.get("tool_calls").cloned(),
        tool_name: object
            .get("tool_name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        token_count: None,
        finish_reason: None,
        reasoning: object
            .get("reasoning")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        reasoning_content: object
            .get("reasoning_content")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        reasoning_details: object.get("reasoning_details").cloned(),
        codex_reasoning_items: object.get("codex_reasoning_items").cloned(),
        codex_message_items: object.get("codex_message_items").cloned(),
    })
}

fn assistant_response_message(
    response: &NormalizedAssistantResponse,
    tool_calls: Option<Vec<Value>>,
) -> Value {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String("assistant".to_string()));
    map.insert(
        "content".to_string(),
        response
            .content
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    if let Some(tool_calls) = tool_calls {
        map.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    if let Some(reasoning) = response
        .reasoning
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        map.insert("reasoning".to_string(), Value::String(reasoning));
    }
    if let Some(reasoning_content) = response
        .reasoning_content
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        map.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content),
        );
    }
    if let Some(reasoning_details) = response.reasoning_details.clone() {
        map.insert("reasoning_details".to_string(), reasoning_details);
    }
    if let Some(codex_reasoning_items) = response.codex_reasoning_items.clone() {
        map.insert("codex_reasoning_items".to_string(), codex_reasoning_items);
    }
    if let Some(codex_message_items) = response.codex_message_items.clone() {
        map.insert("codex_message_items".to_string(), codex_message_items);
    }
    Value::Object(map)
}

fn assistant_response_append(
    response: &NormalizedAssistantResponse,
    tool_calls: Option<Vec<Value>>,
) -> MessageAppend {
    MessageAppend {
        role: "assistant".to_string(),
        content: response.content.clone().map(Value::String),
        tool_call_id: None,
        tool_calls: tool_calls.map(Value::Array),
        tool_name: None,
        token_count: None,
        finish_reason: response.finish_reason.clone(),
        reasoning: response.reasoning.clone(),
        reasoning_content: response.reasoning_content.clone(),
        reasoning_details: response.reasoning_details.clone(),
        codex_reasoning_items: response.codex_reasoning_items.clone(),
        codex_message_items: response.codex_message_items.clone(),
    }
}

fn build_system_prompt(
    runtime: &ToolRuntime,
    memory_config: &crate::MemoryConfig,
) -> Result<String, HermesError> {
    let soul_path = runtime.hermes_home().join("SOUL.md");
    let soul = fs::read_to_string(&soul_path).map_err(|source| HermesError::Io {
        action: "reading",
        path: soul_path,
        source,
    })?;

    let mut sections = vec![soul.trim().to_string()];
    let agents_path = runtime.cwd().join("AGENTS.md");
    if agents_path.is_file()
        && let Ok(agents) = fs::read_to_string(&agents_path)
    {
        let trimmed = agents.trim();
        if !trimmed.is_empty() {
            sections.push(format!("Project instructions:\n{trimmed}"));
        }
    }
    if memory_config.memory_enabled
        && let Some(block) = runtime.memory_system_prompt_block("memory")
    {
        sections.push(block);
    }
    if memory_config.user_profile_enabled
        && let Some(block) = runtime.memory_system_prompt_block("user")
    {
        sections.push(block);
    }
    if runtime
        .available_tool_names()
        .is_some_and(|tools| tools.contains("kanban_show"))
    {
        sections.push(KANBAN_GUIDANCE.to_string());
    }
    Ok(sections.join("\n\n"))
}

fn build_http_client() -> Result<Client, HermesError> {
    build_http_client_with_timeout(DEFAULT_AGENT_TIMEOUT_SECS)
}

pub(crate) fn build_http_client_with_timeout(timeout_secs: u64) -> Result<Client, HermesError> {
    Client::builder()
        .timeout(Duration::from_secs(timeout_secs.max(1)))
        .build()
        .map_err(|error| HermesError::State {
            action: "building API client",
            detail: error.to_string(),
        })
}

fn send_model_request(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
    session_id: Option<&str>,
    max_output_tokens: Option<u64>,
) -> Result<NormalizedAssistantResponse, HermesError> {
    match runtime_model.api_mode.as_str() {
        "chat_completions" => send_chat_completion(
            client,
            runtime_model,
            messages,
            tools,
            session_id,
            max_output_tokens,
        ),
        "anthropic_messages" => {
            send_anthropic_message(client, runtime_model, messages, tools, max_output_tokens)
        }
        "codex_responses" => send_codex_response(client, runtime_model, messages, tools),
        "bedrock_converse" => send_bedrock_converse(client, runtime_model, messages, tools),
        other => Err(HermesError::State {
            action: "running agent turn",
            detail: format!("Unsupported runtime api_mode '{other}'."),
        }),
    }
}

pub(crate) fn request_model_text(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
) -> Result<Option<String>, HermesError> {
    let response = send_model_request(client, runtime_model, messages, &[], None, None)?;
    Ok(response.content.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }))
}

fn send_chat_completion(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
    session_id: Option<&str>,
    max_output_tokens: Option<u64>,
) -> Result<NormalizedAssistantResponse, HermesError> {
    if is_copilot_acp_runtime(runtime_model) {
        return send_copilot_acp_chat_completion(runtime_model, messages, tools);
    }
    if is_google_gemini_cli_runtime(runtime_model) {
        return send_google_gemini_chat_completion(client, runtime_model, messages, tools);
    }
    let request_messages = prepare_chat_completion_messages(runtime_model, messages);
    let mut payload = json!({
        "model": runtime_model.model,
        "messages": request_messages,
    });
    if is_qwen_portal_runtime(runtime_model) {
        payload["vl_high_resolution_images"] = Value::Bool(true);
        payload["metadata"] = json!({
            "sessionId": session_id.unwrap_or("hermes"),
            "promptId": unix_ts_nanos().to_string(),
        });
    }
    if !tools.is_empty() {
        payload["tools"] = Value::Array(tools.iter().map(|tool| tool.openai_schema()).collect());
        payload["tool_choice"] = Value::String("auto".to_string());
    }
    if let Some(max_output_tokens) = max_output_tokens {
        payload["max_tokens"] = json!(max_output_tokens);
    }

    let url = format!(
        "{}/chat/completions",
        runtime_model.base_url.trim_end_matches('/')
    );
    let response = client.post(&url).json(&payload);
    let response = apply_chat_auth_headers(response, runtime_model)?
        .send()
        .map_err(|error| HermesError::State {
            action: "calling chat completions",
            detail: error.to_string(),
        })?;
    let success_headers = response.headers().clone();
    let body = read_json_response(response, "calling chat completions")?;
    if runtime_model.provider == "nous" {
        capture_nous_rate_limit_state_from_headers(&success_headers);
    }
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding chat completions response",
        detail: format!("{error}: {body}"),
    })?;
    let choice = extract_choice(&parsed)?;
    let assistant_message = choice.get("message").ok_or_else(|| HermesError::State {
        action: "parsing assistant response",
        detail: format!("Response missing choices[0].message: {parsed}"),
    })?;

    Ok(NormalizedAssistantResponse {
        content: extract_message_text(assistant_message.get("content")),
        tool_calls: parse_tool_calls(assistant_message.get("tool_calls")),
        finish_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        reasoning: extract_chat_reasoning_text(assistant_message, "reasoning"),
        reasoning_content: extract_chat_reasoning_text(assistant_message, "reasoning_content"),
        reasoning_details: assistant_message.get("reasoning_details").cloned(),
        codex_reasoning_items: None,
        codex_message_items: None,
    })
}

fn send_anthropic_message(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
    max_output_tokens: Option<u64>,
) -> Result<NormalizedAssistantResponse, HermesError> {
    let (system_prompt, anthropic_messages) =
        convert_messages_to_anthropic(messages, runtime_model)?;
    let mut payload = json!({
        "model": runtime_model.model,
        "messages": anthropic_messages,
        "max_tokens": max_output_tokens.unwrap_or(16_384),
    });
    if let Some(system_prompt) = system_prompt {
        payload["system"] = Value::String(system_prompt);
    }
    if !tools.is_empty() {
        payload["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool
                            .schema
                            .get("parameters")
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                    })
                })
                .collect(),
        );
        payload["tool_choice"] = json!({"type": "auto"});
    }

    let url = anthropic_messages_url(&runtime_model.base_url);
    let response = client.post(&url).json(&payload);
    let response = apply_anthropic_auth_headers(response, runtime_model)?
        .send()
        .map_err(|error| HermesError::State {
            action: "calling anthropic messages",
            detail: error.to_string(),
        })?;
    let body = read_json_response(response, "calling anthropic messages")?;
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding anthropic messages response",
        detail: format!("{error}: {body}"),
    })?;

    Ok(NormalizedAssistantResponse {
        content: extract_message_text(parsed.get("content")),
        tool_calls: parse_anthropic_tool_calls(parsed.get("content")),
        finish_reason: parsed
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(map_anthropic_finish_reason),
        reasoning: None,
        reasoning_content: None,
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
    })
}

fn anthropic_messages_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/v1") {
        format!("{normalized}/messages")
    } else {
        format!("{normalized}/v1/messages")
    }
}

fn send_codex_response(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
) -> Result<NormalizedAssistantResponse, HermesError> {
    let instructions = messages
        .iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|message| extract_message_text(message.get("content")))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "You are Hermes Agent.".to_string());

    let input = chat_messages_to_responses_input(messages)?;
    let mut payload = json!({
        "model": runtime_model.model,
        "instructions": instructions,
        "input": input,
        "tools": responses_tools(tools),
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "store": false,
    });
    if payload["tools"]
        .as_array()
        .is_some_and(|items| items.is_empty())
    {
        payload
            .as_object_mut()
            .expect("payload object")
            .remove("tools");
    }

    let url = format!("{}/responses", runtime_model.base_url.trim_end_matches('/'));
    let response = client.post(&url).json(&payload);
    let response = apply_chat_auth_headers(response, runtime_model)?
        .send()
        .map_err(|error| HermesError::State {
            action: "calling codex responses",
            detail: error.to_string(),
        })?;
    let body = read_json_response(response, "calling codex responses")?;
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding codex responses response",
        detail: format!("{error}: {body}"),
    })?;

    normalize_codex_response(&parsed)
}

fn send_bedrock_converse(
    _client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
) -> Result<NormalizedAssistantResponse, HermesError> {
    let region = parse_bedrock_region(&runtime_model.base_url)
        .or_else(|| {
            env::var("AWS_REGION")
                .ok()
                .as_deref()
                .and_then(non_empty_trimmed)
        })
        .or_else(|| {
            env::var("AWS_DEFAULT_REGION")
                .ok()
                .as_deref()
                .and_then(non_empty_trimmed)
        })
        .unwrap_or_else(|| "us-east-1".to_string());
    let client = bedrock_client(&region)?;
    let (system, converse_messages) = bedrock_convert_messages(messages)?;
    let inference_config = BedrockInferenceConfiguration::builder()
        .max_tokens(4096)
        .build();
    let mut request = client
        .converse()
        .model_id(runtime_model.model.clone())
        .set_messages(Some(converse_messages))
        .inference_config(inference_config);
    if let Some(system) = system {
        request = request.set_system(Some(system));
    }
    if !tools.is_empty() {
        if bedrock_model_supports_tool_use(&runtime_model.model) {
            let tool_config = BedrockToolConfiguration::builder()
                .set_tools(Some(bedrock_convert_tools(tools)?))
                .build()
                .map_err(|error| HermesError::State {
                    action: "building bedrock tool config",
                    detail: error.to_string(),
                })?;
            request = request.tool_config(tool_config);
        } else {
            log::warn!(
                target: "run_agent",
                "bedrock tools stripped for non-tool model: {}",
                runtime_model.model
            );
        }
    }

    let response = bedrock_tokio_runtime()
        .block_on(async { request.send().await })
        .map_err(|error| HermesError::State {
            action: "calling bedrock converse",
            detail: error.to_string(),
        })?;
    normalize_bedrock_response(response)
}

fn bedrock_tokio_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("bedrock tokio runtime must initialize")
    })
}

fn cached_bedrock_clients() -> &'static Mutex<HashMap<String, BedrockClient>> {
    static CLIENTS: OnceLock<Mutex<HashMap<String, BedrockClient>>> = OnceLock::new();
    CLIENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bedrock_client(region: &str) -> Result<BedrockClient, HermesError> {
    let cache = cached_bedrock_clients();
    if let Some(client) = cache
        .lock()
        .map_err(|_| HermesError::State {
            action: "locking bedrock client cache",
            detail: "cache lock poisoned".to_string(),
        })?
        .get(region)
        .cloned()
    {
        return Ok(client);
    }

    let shared_config = bedrock_tokio_runtime().block_on(async {
        aws_config::defaults(AwsBehaviorVersion::latest())
            .region(BedrockRegion::new(region.to_string()))
            .load()
            .await
    });
    let client = BedrockClient::new(&shared_config);
    cache
        .lock()
        .map_err(|_| HermesError::State {
            action: "locking bedrock client cache",
            detail: "cache lock poisoned".to_string(),
        })?
        .insert(region.to_string(), client.clone());
    Ok(client)
}

fn bedrock_model_supports_tool_use(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    ![
        "deepseek.r1",
        "deepseek-r1",
        "stability.",
        "cohere.embed",
        "amazon.titan-embed",
    ]
    .iter()
    .any(|pattern| model_id.contains(pattern))
}

fn bedrock_convert_tools(tools: &[crate::ToolDefinition]) -> Result<Vec<BedrockTool>, HermesError> {
    tools
        .iter()
        .map(|tool| {
            let parameters = tool
                .schema
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            let input_schema =
                BedrockToolInputSchema::Json(serde_json_to_smithy_document(&parameters));
            let mut builder = BedrockToolSpecification::builder()
                .name(tool.name.clone())
                .input_schema(input_schema);
            if !tool.description.trim().is_empty() {
                builder = builder.description(tool.description.clone());
            }
            let tool_spec = builder.build().map_err(|error| HermesError::State {
                action: "building bedrock tool spec",
                detail: error.to_string(),
            })?;
            Ok(BedrockTool::ToolSpec(tool_spec))
        })
        .collect()
}

fn bedrock_convert_messages(
    messages: &[Value],
) -> Result<(Option<Vec<BedrockSystemContentBlock>>, Vec<BedrockMessage>), HermesError> {
    let mut system_blocks = Vec::new();
    let mut converse_messages = Vec::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let content = message.get("content");

        match role {
            "system" => {
                if let Some(content) = content {
                    match content {
                        Value::String(text) => {
                            if !text.trim().is_empty() {
                                system_blocks.push(BedrockSystemContentBlock::Text(text.clone()));
                            }
                        }
                        Value::Array(parts) => {
                            for part in parts {
                                match part {
                                    Value::String(text) if !text.trim().is_empty() => {
                                        system_blocks
                                            .push(BedrockSystemContentBlock::Text(text.clone()));
                                    }
                                    Value::Object(object)
                                        if object.get("type").and_then(Value::as_str)
                                            == Some("text") =>
                                    {
                                        if let Some(text) =
                                            object.get("text").and_then(Value::as_str)
                                            && !text.trim().is_empty()
                                        {
                                            system_blocks.push(BedrockSystemContentBlock::Text(
                                                text.to_string(),
                                            ));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "tool" => {
                let tool_call_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if tool_call_id.is_empty() {
                    continue;
                }
                let result_text = match content {
                    Some(Value::String(text)) => text.clone(),
                    Some(other) => {
                        serde_json::to_string(other).unwrap_or_else(|_| "null".to_string())
                    }
                    None => String::new(),
                };
                let tool_result = BedrockToolResultBlock::builder()
                    .tool_use_id(tool_call_id)
                    .content(BedrockToolResultContentBlock::Text(result_text))
                    .build()
                    .map_err(|error| HermesError::State {
                        action: "building bedrock tool result",
                        detail: error.to_string(),
                    })?;
                push_or_merge_bedrock_message(
                    &mut converse_messages,
                    BedrockConversationRole::User,
                    vec![BedrockContentBlock::ToolResult(tool_result)],
                )?;
            }
            "assistant" => {
                let mut content_blocks = Vec::new();
                if let Some(value) = content {
                    content_blocks.extend(bedrock_content_blocks(value)?);
                }
                for tool_call in parse_tool_calls(message.get("tool_calls")) {
                    let input = serde_json_to_smithy_document(&tool_call.json);
                    let tool_use = aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                        .tool_use_id(tool_call.id)
                        .name(tool_call.name)
                        .input(input)
                        .build()
                        .map_err(|error| HermesError::State {
                            action: "building bedrock tool use",
                            detail: error.to_string(),
                        })?;
                    content_blocks.push(BedrockContentBlock::ToolUse(tool_use));
                }
                if content_blocks.is_empty() {
                    content_blocks.push(blank_bedrock_text_block());
                }
                push_or_merge_bedrock_message(
                    &mut converse_messages,
                    BedrockConversationRole::Assistant,
                    content_blocks,
                )?;
            }
            "user" => {
                let content_blocks = match content {
                    Some(value) => bedrock_content_blocks(value)?,
                    None => vec![blank_bedrock_text_block()],
                };
                push_or_merge_bedrock_message(
                    &mut converse_messages,
                    BedrockConversationRole::User,
                    content_blocks,
                )?;
            }
            _ => {}
        }
    }

    if converse_messages
        .first()
        .map(|message| message.role() != &BedrockConversationRole::User)
        .unwrap_or(false)
    {
        converse_messages.insert(
            0,
            BedrockMessage::builder()
                .role(BedrockConversationRole::User)
                .content(blank_bedrock_text_block())
                .build()
                .map_err(|error| HermesError::State {
                    action: "building bedrock message",
                    detail: error.to_string(),
                })?,
        );
    }
    if converse_messages
        .last()
        .map(|message| message.role() != &BedrockConversationRole::User)
        .unwrap_or(false)
    {
        converse_messages.push(
            BedrockMessage::builder()
                .role(BedrockConversationRole::User)
                .content(blank_bedrock_text_block())
                .build()
                .map_err(|error| HermesError::State {
                    action: "building bedrock message",
                    detail: error.to_string(),
                })?,
        );
    }

    Ok((
        (!system_blocks.is_empty()).then_some(system_blocks),
        converse_messages,
    ))
}

fn push_or_merge_bedrock_message(
    messages: &mut Vec<BedrockMessage>,
    role: BedrockConversationRole,
    mut content: Vec<BedrockContentBlock>,
) -> Result<(), HermesError> {
    if let Some(last) = messages.last_mut()
        && last.role() == &role
    {
        last.content.append(&mut content);
        return Ok(());
    }
    messages.push(
        BedrockMessage::builder()
            .role(role)
            .set_content(Some(content))
            .build()
            .map_err(|error| HermesError::State {
                action: "building bedrock message",
                detail: error.to_string(),
            })?,
    );
    Ok(())
}

fn blank_bedrock_text_block() -> BedrockContentBlock {
    BedrockContentBlock::Text(" ".to_string())
}

fn bedrock_content_blocks(content: &Value) -> Result<Vec<BedrockContentBlock>, HermesError> {
    match content {
        Value::Null => Ok(vec![blank_bedrock_text_block()]),
        Value::String(text) => Ok(vec![if text.trim().is_empty() {
            blank_bedrock_text_block()
        } else {
            BedrockContentBlock::Text(text.clone())
        }]),
        Value::Array(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                match part {
                    Value::String(text) => blocks.push(BedrockContentBlock::Text(text.clone())),
                    Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = object
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or(" ")
                                .to_string();
                            blocks.push(BedrockContentBlock::Text(if text.is_empty() {
                                " ".to_string()
                            } else {
                                text
                            }));
                        }
                        Some("image_url") => {
                            let url = object
                                .get("image_url")
                                .and_then(Value::as_object)
                                .and_then(|image| image.get("url"))
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            if let Some(data_url) = url.strip_prefix("data:") {
                                let (header, encoded) =
                                    data_url.split_once(',').unwrap_or((data_url, ""));
                                let media_type = header.split(';').next().unwrap_or("image/jpeg");
                                if let Ok(bytes) =
                                    base64::engine::general_purpose::STANDARD.decode(encoded)
                                {
                                    let image = BedrockImageBlock::builder()
                                        .format(bedrock_image_format(media_type))
                                        .source(BedrockImageSource::Bytes(BedrockBlob::new(bytes)))
                                        .build()
                                        .map_err(|error| HermesError::State {
                                            action: "building bedrock image block",
                                            detail: error.to_string(),
                                        })?;
                                    blocks.push(BedrockContentBlock::Image(image));
                                } else {
                                    blocks.push(BedrockContentBlock::Text(
                                        "[Image omitted: invalid data URL]".to_string(),
                                    ));
                                }
                            } else if !url.is_empty() {
                                blocks.push(BedrockContentBlock::Text(format!("[Image: {url}]")));
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            if blocks.is_empty() {
                blocks.push(blank_bedrock_text_block());
            }
            Ok(blocks)
        }
        other => Ok(vec![BedrockContentBlock::Text(other.to_string())]),
    }
}

fn bedrock_image_format(media_type: &str) -> BedrockImageFormat {
    let subtype = media_type
        .rsplit('/')
        .next()
        .unwrap_or("jpeg")
        .trim()
        .to_ascii_lowercase();
    match subtype.as_str() {
        "gif" => BedrockImageFormat::Gif,
        "png" => BedrockImageFormat::Png,
        "webp" => BedrockImageFormat::Webp,
        _ => BedrockImageFormat::Jpeg,
    }
}

fn serde_json_to_smithy_document(value: &Value) -> SmithyDocument {
    match value {
        Value::Null => SmithyDocument::Null,
        Value::Bool(boolean) => SmithyDocument::Bool(*boolean),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                SmithyDocument::Number(SmithyNumber::PosInt(value))
            } else if let Some(value) = number.as_i64() {
                if value < 0 {
                    SmithyDocument::Number(SmithyNumber::NegInt(value))
                } else {
                    SmithyDocument::Number(SmithyNumber::PosInt(value as u64))
                }
            } else {
                SmithyDocument::Number(SmithyNumber::Float(number.as_f64().unwrap_or_default()))
            }
        }
        Value::String(text) => SmithyDocument::String(text.clone()),
        Value::Array(items) => SmithyDocument::Array(
            items
                .iter()
                .map(serde_json_to_smithy_document)
                .collect::<Vec<_>>(),
        ),
        Value::Object(object) => SmithyDocument::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), serde_json_to_smithy_document(value)))
                .collect::<HashMap<_, _>>(),
        ),
    }
}

fn smithy_document_to_json_value(value: &SmithyDocument) -> Value {
    match value {
        SmithyDocument::Null => Value::Null,
        SmithyDocument::Bool(boolean) => Value::Bool(*boolean),
        SmithyDocument::Number(number) => match number {
            SmithyNumber::PosInt(value) => Value::Number(serde_json::Number::from(*value)),
            SmithyNumber::NegInt(value) => Value::Number(serde_json::Number::from(*value)),
            SmithyNumber::Float(value) => serde_json::Number::from_f64(*value)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        },
        SmithyDocument::String(text) => Value::String(text.clone()),
        SmithyDocument::Array(items) => Value::Array(
            items
                .iter()
                .map(smithy_document_to_json_value)
                .collect::<Vec<_>>(),
        ),
        SmithyDocument::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), smithy_document_to_json_value(value)))
                .collect::<Map<_, _>>(),
        ),
    }
}

fn normalize_bedrock_response(
    response: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
) -> Result<NormalizedAssistantResponse, HermesError> {
    let output = response.output().ok_or_else(|| HermesError::State {
        action: "parsing bedrock response",
        detail: "response missing output".to_string(),
    })?;
    let BedrockConverseMessageOutput::Message(message) = output else {
        return Err(HermesError::State {
            action: "parsing bedrock response",
            detail: format!("unsupported output variant: {output:?}"),
        });
    };

    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in message.content() {
        match block {
            BedrockContentBlock::Text(text) if !text.trim().is_empty() => {
                text_parts.push(text.clone());
            }
            BedrockContentBlock::ToolUse(tool_use) => {
                let json = smithy_document_to_json_value(tool_use.input());
                let arguments_raw =
                    serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string());
                if tool_use.tool_use_id().trim().is_empty() || tool_use.name().trim().is_empty() {
                    return Err(HermesError::State {
                        action: "parsing bedrock response",
                        detail: "tool use block missing id or name".to_string(),
                    });
                }
                tool_calls.push(PendingToolCall {
                    id: tool_use.tool_use_id().to_string(),
                    name: tool_use.name().to_string(),
                    arguments_raw,
                    json,
                });
            }
            _ => {}
        }
    }

    let has_tool_calls = !tool_calls.is_empty();
    Ok(NormalizedAssistantResponse {
        content: (!text_parts.is_empty()).then(|| text_parts.join("\n")),
        tool_calls,
        finish_reason: Some(map_bedrock_finish_reason(
            Some(response.stop_reason()),
            has_tool_calls,
        )),
        reasoning: None,
        reasoning_content: None,
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
    })
}

fn map_bedrock_finish_reason(
    stop_reason: Option<&BedrockStopReason>,
    has_tool_calls: bool,
) -> String {
    if has_tool_calls {
        return "tool_calls".to_string();
    }
    match stop_reason {
        Some(BedrockStopReason::EndTurn | BedrockStopReason::StopSequence) => "stop".to_string(),
        Some(BedrockStopReason::ToolUse | BedrockStopReason::MalformedToolUse) => {
            "tool_calls".to_string()
        }
        Some(BedrockStopReason::MaxTokens | BedrockStopReason::ModelContextWindowExceeded) => {
            "length".to_string()
        }
        Some(BedrockStopReason::ContentFiltered | BedrockStopReason::GuardrailIntervened) => {
            "content_filter".to_string()
        }
        Some(other) => other.as_str().to_string(),
        None => "stop".to_string(),
    }
}

fn apply_chat_auth_headers(
    request: reqwest::blocking::RequestBuilder,
    runtime_model: &crate::ModelRuntimeConfig,
) -> Result<reqwest::blocking::RequestBuilder, HermesError> {
    let mut request = request.header("Content-Type", "application/json");
    if !runtime_model.api_key.is_empty() {
        request = request.bearer_auth(&runtime_model.api_key);
    }
    if is_qwen_portal_runtime(runtime_model) {
        let user_agent = format!(
            "QwenCode/{} ({}; {})",
            QWEN_CODE_VERSION,
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        request = request
            .header("User-Agent", &user_agent)
            .header("X-DashScope-CacheControl", "enable")
            .header("X-DashScope-UserAgent", &user_agent)
            .header("X-DashScope-AuthType", "qwen-oauth");
    }
    for (name, value) in &runtime_model.default_headers {
        request = request.header(name, value);
    }
    Ok(request)
}

fn apply_anthropic_auth_headers(
    request: reqwest::blocking::RequestBuilder,
    runtime_model: &crate::ModelRuntimeConfig,
) -> Result<reqwest::blocking::RequestBuilder, HermesError> {
    let mut request = request
        .header("Content-Type", "application/json")
        .header("anthropic-version", "2023-06-01");
    if requires_bearer_anthropic_auth(&runtime_model.base_url)
        || runtime_model.provider.starts_with("minimax")
        || (runtime_model.provider == "anthropic" && runtime_model.auth_type == "oauth_external")
    {
        request = request.bearer_auth(&runtime_model.api_key);
    } else if !runtime_model.api_key.is_empty() {
        request = request.header("x-api-key", &runtime_model.api_key);
    }
    for (name, value) in &runtime_model.default_headers {
        request = request.header(name, value);
    }
    Ok(request)
}

fn read_json_response(
    response: reqwest::blocking::Response,
    action: &'static str,
) -> Result<String, HermesError> {
    let status = response.status();
    let header_summary = collect_http_error_header_summary(response.headers());
    let body = response.text().map_err(|error| HermesError::State {
        action: "reading provider response",
        detail: error.to_string(),
    })?;

    if !status.is_success() {
        let trimmed = body
            .chars()
            .take(MAX_HTTP_ERROR_BODY_CHARS)
            .collect::<String>();
        return Err(HermesError::State {
            action,
            detail: if header_summary.is_empty() {
                format!("HTTP {}: {}", status.as_u16(), trimmed)
            } else {
                format!("HTTP {} [{}]: {}", status.as_u16(), header_summary, trimmed)
            },
        });
    }
    Ok(body)
}

fn collect_http_error_header_summary(headers: &reqwest::header::HeaderMap) -> String {
    let mut items = headers
        .iter()
        .filter_map(|(name, value)| {
            let lowered = name.as_str().to_ascii_lowercase();
            (lowered == "retry-after" || lowered.starts_with("x-ratelimit-")).then(|| {
                value
                    .to_str()
                    .ok()
                    .map(|raw| format!("{}={}", lowered, raw.trim()))
            })?
        })
        .collect::<Vec<_>>();
    items.sort();
    items.join("; ")
}

fn extract_choice(response: &Value) -> Result<&Value, HermesError> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| HermesError::State {
            action: "parsing assistant response",
            detail: format!("Response missing choices[0]: {response}"),
        })
}

fn requires_bearer_anthropic_auth(base_url: &str) -> bool {
    let normalized = base_url.trim().trim_end_matches('/').to_ascii_lowercase();
    normalized.starts_with("https://api.minimax.io/anthropic")
        || normalized.starts_with("https://api.minimaxi.com/anthropic")
}

fn parse_bedrock_region(base_url: &str) -> Option<String> {
    let host = reqwest::Url::parse(base_url)
        .ok()?
        .host_str()?
        .to_ascii_lowercase();
    let prefix = "bedrock-runtime.";
    let suffix = ".amazonaws.com";
    if host.starts_with(prefix) && host.ends_with(suffix) {
        let region = &host[prefix.len()..host.len() - suffix.len()];
        return non_empty_trimmed(region);
    }
    None
}

fn is_qwen_portal_runtime(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    runtime_model.provider == "qwen-oauth"
        || reqwest::Url::parse(&runtime_model.base_url)
            .ok()
            .and_then(|url| {
                url.host_str()
                    .map(|host| host.eq_ignore_ascii_case("portal.qwen.ai"))
            })
            .unwrap_or(false)
}

fn prepare_chat_completion_messages(
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
) -> Vec<Value> {
    let mut prepared = messages.to_vec();
    for message in &mut prepared {
        patch_reasoning_content_for_chat_replay(runtime_model, message);
    }
    if is_qwen_portal_runtime(runtime_model) {
        normalize_qwen_messages(&mut prepared);
    }
    prepared
}

fn normalize_qwen_messages(prepared: &mut [Value]) {
    for message in &mut *prepared {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let Some(content) = object.get_mut("content") else {
            continue;
        };
        normalize_qwen_content(content);
    }
    for message in &mut *prepared {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some("system") {
            continue;
        }
        let Some(content) = object.get_mut("content").and_then(Value::as_array_mut) else {
            break;
        };
        let Some(last) = content.last_mut().and_then(Value::as_object_mut) else {
            break;
        };
        last.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
        break;
    }
}

fn patch_reasoning_content_for_chat_replay(
    runtime_model: &crate::ModelRuntimeConfig,
    message: &mut Value,
) {
    let Some(object) = message.as_object_mut() else {
        return;
    };
    if object.get("role").and_then(Value::as_str) != Some("assistant") {
        return;
    }

    let needs_pad = needs_thinking_reasoning_pad(runtime_model);
    if let Some(existing) = object.get("reasoning_content") {
        if let Some(existing) = existing.as_str() {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(if existing.is_empty() && needs_pad {
                    " ".to_string()
                } else {
                    existing.to_string()
                }),
            );
        } else {
            object.remove("reasoning_content");
        }
        return;
    }

    let reasoning = object
        .get("reasoning")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    if needs_pad
        && object
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
        && reasoning.is_some()
    {
        object.insert(
            "reasoning_content".to_string(),
            Value::String(" ".to_string()),
        );
        return;
    }
    if let Some(reasoning) = reasoning {
        object.insert("reasoning_content".to_string(), Value::String(reasoning));
        return;
    }
    if needs_pad {
        object.insert(
            "reasoning_content".to_string(),
            Value::String(" ".to_string()),
        );
    }
}

fn needs_thinking_reasoning_pad(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    needs_deepseek_tool_reasoning(runtime_model) || needs_kimi_tool_reasoning(runtime_model)
}

fn needs_kimi_tool_reasoning(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    matches!(
        runtime_model.provider.as_str(),
        "kimi-coding" | "kimi-coding-cn"
    ) || base_url_host_matches_any(
        &runtime_model.base_url,
        &["api.kimi.com", "moonshot.ai", "moonshot.cn"],
    )
}

fn needs_deepseek_tool_reasoning(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    runtime_model.provider == "deepseek"
        || runtime_model
            .model
            .to_ascii_lowercase()
            .contains("deepseek")
        || base_url_host_matches_any(&runtime_model.base_url, &["api.deepseek.com"])
}

fn base_url_host_matches_any(base_url: &str, hosts: &[&str]) -> bool {
    let Some(host) = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_ascii_lowercase()))
    else {
        return false;
    };
    hosts.iter().any(|candidate| {
        let candidate = candidate.trim().to_ascii_lowercase();
        host == candidate
            || host
                .strip_suffix(&candidate)
                .is_some_and(|prefix| prefix.ends_with('.'))
    })
}

fn normalize_qwen_content(content: &mut Value) {
    match content {
        Value::String(text) => {
            *content = Value::Array(vec![json!({
                "type": "text",
                "text": text.clone(),
            })]);
        }
        Value::Array(parts) => {
            let mut normalized = Vec::new();
            for part in parts.iter() {
                match part {
                    Value::String(text) => normalized.push(json!({
                        "type": "text",
                        "text": text,
                    })),
                    Value::Object(_) => normalized.push(part.clone()),
                    _ => {}
                }
            }
            if !normalized.is_empty() {
                *content = Value::Array(normalized);
            }
        }
        _ => {}
    }
}

#[derive(Debug, Clone, Default)]
struct GoogleProjectContext {
    project_id: String,
    managed_project_id: String,
}

fn is_google_gemini_cli_runtime(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    runtime_model.provider == "google-gemini-cli"
        || runtime_model
            .base_url
            .trim()
            .to_ascii_lowercase()
            .starts_with("cloudcode-pa://")
}

fn is_copilot_acp_runtime(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    let normalized = runtime_model.base_url.trim().to_ascii_lowercase();
    runtime_model.provider == "copilot-acp"
        || normalized.starts_with(COPILOT_ACP_MARKER_BASE_URL)
        || normalized.starts_with("acp+tcp://")
}

fn send_copilot_acp_chat_completion(
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
) -> Result<NormalizedAssistantResponse, HermesError> {
    let creds = crate::resolve_copilot_acp_runtime_credentials()?;
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let prompt_text = format_copilot_acp_prompt(
        messages,
        &runtime_model.model,
        tools,
        (!tools.is_empty()).then(|| Value::String(String::from("auto"))),
    );
    let timeout = Duration::from_secs(900);

    let mut command = Command::new(&creds.command);
    command
        .args(resolve_copilot_acp_args())
        .current_dir(&cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("HOME", copilot_acp_subprocess_home());
    let mut child = command.spawn().map_err(|error| HermesError::State {
        action: "starting Copilot ACP runtime",
        detail: format!("{} failed: {error}", creds.command),
    })?;

    let mut stdin = child.stdin.take().ok_or_else(|| HermesError::State {
        action: "starting Copilot ACP runtime",
        detail: "stdin pipe was not available".to_string(),
    })?;
    let stdout = child.stdout.take().ok_or_else(|| HermesError::State {
        action: "starting Copilot ACP runtime",
        detail: "stdout pipe was not available".to_string(),
    })?;
    let stderr = child.stderr.take().ok_or_else(|| HermesError::State {
        action: "starting Copilot ACP runtime",
        detail: "stderr pipe was not available".to_string(),
    })?;

    let (tx, rx) = mpsc::channel::<Value>();
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                if tx.send(value).is_err() {
                    break;
                }
            }
        }
    });
    let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::with_capacity(40)));
    let stderr_tail_reader = Arc::clone(&stderr_tail);
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            let mut tail = stderr_tail_reader.lock().expect("stderr tail lock");
            if tail.len() == 40 {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    });

    let result = (|| {
        let mut next_id = 0_u64;
        let _ = copilot_acp_request(
            &mut stdin,
            &mut child,
            &rx,
            &stderr_tail,
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
            &cwd,
            None,
            None,
            timeout,
        )?;
        let session = copilot_acp_request(
            &mut stdin,
            &mut child,
            &rx,
            &stderr_tail,
            &mut next_id,
            "session/new",
            json!({
                "cwd": cwd.to_string_lossy().to_string(),
                "mcpServers": [],
            }),
            &cwd,
            None,
            None,
            timeout,
        )?;
        let session_id = session
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| HermesError::State {
                action: "calling Copilot ACP",
                detail: "Copilot ACP did not return a sessionId.".to_string(),
            })?;

        let mut text_chunks = Vec::new();
        let mut reasoning_chunks = Vec::new();
        let _ = copilot_acp_request(
            &mut stdin,
            &mut child,
            &rx,
            &stderr_tail,
            &mut next_id,
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{
                    "type": "text",
                    "text": prompt_text,
                }],
            }),
            &cwd,
            Some(&mut text_chunks),
            Some(&mut reasoning_chunks),
            timeout,
        )?;

        let response_text = text_chunks.concat();
        let reasoning = reasoning_chunks.concat();
        let (tool_calls, content) = extract_copilot_acp_tool_calls(&response_text)?;
        let finish_reason = if tool_calls.is_empty() {
            String::from("stop")
        } else {
            String::from("tool_calls")
        };
        Ok(NormalizedAssistantResponse {
            content: (!content.trim().is_empty()).then_some(content),
            tool_calls,
            finish_reason: Some(finish_reason),
            reasoning: (!reasoning.trim().is_empty()).then_some(reasoning),
            reasoning_content: None,
            reasoning_details: None,
            codex_reasoning_items: None,
            codex_message_items: None,
        })
    })();

    let _ = child.kill();
    let _ = child.wait();
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();
    result
}

fn resolve_copilot_acp_args() -> Vec<String> {
    match env::var("HERMES_COPILOT_ACP_ARGS") {
        Ok(raw) if !raw.trim().is_empty() => shell_words::split(&raw)
            .unwrap_or_else(|_| raw.split_whitespace().map(str::to_string).collect()),
        _ => vec![String::from("--acp"), String::from("--stdio")],
    }
}

fn copilot_acp_subprocess_home() -> PathBuf {
    if let Some(hermes_home) = env::var_os("HERMES_HOME").map(PathBuf::from) {
        let profile_home = hermes_home.join("home");
        if profile_home.is_dir() {
            return profile_home;
        }
    }
    env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn format_copilot_acp_prompt(
    messages: &[Value],
    model: &str,
    tools: &[crate::ToolDefinition],
    tool_choice: Option<Value>,
) -> String {
    let mut sections = vec![
        String::from("You are being used as the active ACP agent backend for Hermes."),
        String::from("Use ACP capabilities to complete tasks."),
        String::from(
            "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        ),
        String::from("If no tool is needed, answer normally."),
    ];
    if !model.trim().is_empty() {
        sections.push(format!("Hermes requested model hint: {model}"));
    }

    if !tools.is_empty() {
        let tool_specs = tools
            .iter()
            .filter_map(|tool| {
                let schema = tool.openai_schema();
                let function = schema.get("function")?.as_object()?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())?;
                Some(json!({
                    "name": name,
                    "description": function.get("description").cloned().unwrap_or(Value::String(String::new())),
                    "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({})),
                }))
            })
            .collect::<Vec<_>>();
        if !tool_specs.is_empty() {
            sections.push(format!(
                "Available tools (OpenAI function schema). When using a tool, emit ONLY <tool_call>{{...}}</tool_call> with one JSON object containing id/type/function{{name,arguments}}. arguments must be a JSON string.\n{}",
                serde_json::to_string(&tool_specs).unwrap_or_else(|_| String::from("[]"))
            ));
        }
    }

    if let Some(choice) = tool_choice {
        sections.push(format!(
            "Tool choice hint: {}",
            serde_json::to_string(&choice).unwrap_or_else(|_| String::from("null"))
        ));
    }

    let transcript = messages
        .iter()
        .filter_map(|message| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .trim()
                .to_ascii_lowercase();
            let label = match role.as_str() {
                "system" => "System",
                "user" => "User",
                "assistant" => "Assistant",
                "tool" => "Tool",
                _ => "Context",
            };
            let rendered = render_copilot_acp_message_content(message.get("content"))?;
            Some(format!("{label}:\n{rendered}"))
        })
        .collect::<Vec<_>>();
    if !transcript.is_empty() {
        sections.push(format!(
            "Conversation transcript:\n\n{}",
            transcript.join("\n\n")
        ));
    }
    sections.push(String::from(
        "Continue the conversation from the latest user request.",
    ));
    sections.join("\n\n")
}

fn render_copilot_acp_message_content(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Some(Value::Object(map)) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            if let Some(text) = map.get("content").and_then(Value::as_str) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            serde_json::to_string(map).ok()
        }
        Some(Value::Array(items)) => {
            let parts = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(text) => {
                        let trimmed = text.trim();
                        (!trimmed.is_empty()).then(|| trimmed.to_string())
                    }
                    Value::Object(map) => {
                        map.get("text").and_then(Value::as_str).and_then(|text| {
                            let trimmed = text.trim();
                            (!trimmed.is_empty()).then(|| trimmed.to_string())
                        })
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

fn extract_copilot_acp_tool_calls(
    text: &str,
) -> Result<(Vec<PendingToolCall>, String), HermesError> {
    let mut tool_calls = Vec::new();
    let mut spans = Vec::new();
    let mut cursor = 0_usize;

    while let Some(start_offset) = text[cursor..].find("<tool_call>") {
        let start = cursor + start_offset;
        let body_start = start + "<tool_call>".len();
        let Some(end_offset) = text[body_start..].find("</tool_call>") else {
            break;
        };
        let end = body_start + end_offset + "</tool_call>".len();
        let raw_json = text[body_start..body_start + end_offset].trim();
        if let Some(call) = parse_copilot_acp_tool_call(raw_json, tool_calls.len() + 1)? {
            tool_calls.push(call);
            spans.push((start, end));
        }
        cursor = end;
    }

    if spans.is_empty() {
        let cleaned = text.trim().to_string();
        return Ok((tool_calls, cleaned));
    }

    let mut cleaned_parts = Vec::new();
    let mut last = 0_usize;
    for (start, end) in spans {
        if last < start {
            let chunk = text[last..start].trim();
            if !chunk.is_empty() {
                cleaned_parts.push(chunk.to_string());
            }
        }
        last = end;
    }
    if last < text.len() {
        let chunk = text[last..].trim();
        if !chunk.is_empty() {
            cleaned_parts.push(chunk.to_string());
        }
    }
    Ok((tool_calls, cleaned_parts.join("\n")))
}

fn parse_copilot_acp_tool_call(
    raw_json: &str,
    index: usize,
) -> Result<Option<PendingToolCall>, HermesError> {
    let Ok(value) = serde_json::from_str::<Value>(raw_json) else {
        return Ok(None);
    };
    let Some(function) = value.get("function").and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(name) = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let arguments_raw = match function.get("arguments") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| String::from("{}")),
        None => String::from("{}"),
    };
    let json = serde_json::from_str(&arguments_raw).unwrap_or_else(|_| json!({}));
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("acp_call_{index}"));
    Ok(Some(PendingToolCall {
        id,
        name: name.to_string(),
        arguments_raw,
        json,
    }))
}

fn copilot_acp_request(
    stdin: &mut ChildStdin,
    child: &mut Child,
    rx: &mpsc::Receiver<Value>,
    stderr_tail: &Arc<Mutex<VecDeque<String>>>,
    next_id: &mut u64,
    method: &str,
    params: Value,
    cwd: &Path,
    text_chunks: Option<&mut Vec<String>>,
    reasoning_chunks: Option<&mut Vec<String>>,
    timeout: Duration,
) -> Result<Value, HermesError> {
    *next_id += 1;
    let request_id = *next_id;
    write_copilot_acp_line(
        stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }),
        "writing Copilot ACP request",
    )?;

    let deadline = Instant::now() + timeout;
    let mut text_chunks = text_chunks;
    let mut reasoning_chunks = reasoning_chunks;
    loop {
        if Instant::now() >= deadline {
            break;
        }
        if let Some(status) = child.try_wait().map_err(|error| HermesError::State {
            action: "waiting for Copilot ACP response",
            detail: error.to_string(),
        })? {
            let stderr = stderr_tail
                .lock()
                .expect("stderr tail lock")
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            let detail = if stderr.is_empty() {
                format!(
                    "Copilot ACP process exited with code {} while handling {method}",
                    exit_code_or_default(status)
                )
            } else {
                format!("Copilot ACP process exited early: {stderr}")
            };
            return Err(HermesError::State {
                action: "calling Copilot ACP",
                detail,
            });
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        let wait = remaining.min(Duration::from_millis(100));
        let Ok(message) = rx.recv_timeout(wait) else {
            continue;
        };
        let text_target = text_chunks.as_mut().map(|chunks| &mut **chunks);
        let reasoning_target = reasoning_chunks.as_mut().map(|chunks| &mut **chunks);
        if handle_copilot_acp_server_message(&message, stdin, cwd, text_target, reasoning_target)? {
            continue;
        }
        if message.get("id").and_then(Value::as_u64) != Some(request_id) {
            continue;
        }
        if let Some(error) = message.get("error") {
            let detail = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_else(|| error.as_str().unwrap_or("unknown Copilot ACP error"));
            return Err(HermesError::State {
                action: "calling Copilot ACP",
                detail: format!("Copilot ACP {method} failed: {detail}"),
            });
        }
        return Ok(message.get("result").cloned().unwrap_or(Value::Null));
    }

    Err(HermesError::State {
        action: "calling Copilot ACP",
        detail: format!("Timed out waiting for Copilot ACP response to {method}."),
    })
}

fn write_copilot_acp_line(
    stdin: &mut ChildStdin,
    payload: &Value,
    action: &'static str,
) -> Result<(), HermesError> {
    let encoded = serde_json::to_string(payload).map_err(|error| HermesError::State {
        action,
        detail: error.to_string(),
    })?;
    stdin
        .write_all(encoded.as_bytes())
        .and_then(|_| stdin.write_all(b"\n"))
        .and_then(|_| stdin.flush())
        .map_err(|error| HermesError::State {
            action,
            detail: error.to_string(),
        })
}

fn handle_copilot_acp_server_message(
    message: &Value,
    stdin: &mut ChildStdin,
    cwd: &Path,
    text_chunks: Option<&mut Vec<String>>,
    reasoning_chunks: Option<&mut Vec<String>>,
) -> Result<bool, HermesError> {
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return Ok(false);
    };

    if method == "session/update" {
        let kind = message
            .get("params")
            .and_then(|params| params.get("update"))
            .and_then(|update| update.get("sessionUpdate"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let chunk_text = message
            .get("params")
            .and_then(|params| params.get("update"))
            .and_then(|update| update.get("content"))
            .and_then(|content| content.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !chunk_text.is_empty() {
            if kind == "agent_message_chunk" {
                if let Some(chunks) = text_chunks {
                    chunks.push(chunk_text);
                }
            } else if kind == "agent_thought_chunk"
                && let Some(chunks) = reasoning_chunks
            {
                chunks.push(chunk_text);
            }
        }
        return Ok(true);
    }

    let message_id = message.get("id").cloned().unwrap_or(Value::Null);
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let response = match method {
        "session/request_permission" => json!({
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "outcome": {
                    "outcome": "cancelled",
                }
            }
        }),
        "fs/read_text_file" => match handle_copilot_acp_read_text_file(&params, cwd) {
            Ok(content) => json!({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "content": content,
                }
            }),
            Err(detail) => json!({
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32602,
                    "message": detail,
                }
            }),
        },
        "fs/write_text_file" => match handle_copilot_acp_write_text_file(&params, cwd) {
            Ok(()) => json!({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": Value::Null,
            }),
            Err(detail) => json!({
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32602,
                    "message": detail,
                }
            }),
        },
        _ => json!({
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {
                "code": -32601,
                "message": format!("ACP client method '{method}' is not supported by Hermes yet."),
            }
        }),
    };
    write_copilot_acp_line(stdin, &response, "writing Copilot ACP response")?;
    Ok(true)
}

fn handle_copilot_acp_read_text_file(params: &Value, cwd: &Path) -> Result<String, String> {
    let path_text = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| String::from("ACP file-system paths must be absolute."))?;
    let path = ensure_copilot_acp_path_within_cwd(path_text, cwd)?;
    if let Some(detail) = copilot_acp_read_block_error(&path) {
        return Err(detail);
    }
    let mut content = fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let line = params.get("line").and_then(Value::as_u64);
    let limit = params.get("limit").and_then(Value::as_u64);
    if let Some(line) = line.filter(|line| *line > 1) {
        let lines = content.lines().collect::<Vec<_>>();
        let start = (line.saturating_sub(1)) as usize;
        let end = limit
            .filter(|limit| *limit > 0)
            .map(|limit| start.saturating_add(limit as usize))
            .unwrap_or(lines.len())
            .min(lines.len());
        content = lines[start.min(lines.len())..end].join("\n");
    }
    Ok(content)
}

fn handle_copilot_acp_write_text_file(params: &Value, cwd: &Path) -> Result<(), String> {
    let path_text = params
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| String::from("ACP file-system paths must be absolute."))?;
    let path = ensure_copilot_acp_path_within_cwd(path_text, cwd)?;
    if copilot_acp_is_write_denied(&path) {
        return Err(format!(
            "Write denied: '{}' is a protected system/credential file.",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let content = params
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    fs::write(path, content).map_err(|error| error.to_string())
}

fn ensure_copilot_acp_path_within_cwd(path_text: &str, cwd: &Path) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(path_text);
    if !candidate.is_absolute() {
        return Err(String::from("ACP file-system paths must be absolute."));
    }
    let root = cwd
        .canonicalize()
        .map_err(|error| format!("Could not resolve session cwd '{}': {error}", cwd.display()))?;
    let resolved = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|error| format!("Could not resolve '{}': {error}", candidate.display()))?
    } else {
        let parent = candidate.parent().ok_or_else(|| {
            format!(
                "Path '{}' must have an existing parent directory.",
                candidate.display()
            )
        })?;
        let resolved_parent = parent
            .canonicalize()
            .map_err(|error| format!("Could not resolve '{}': {error}", parent.display()))?;
        let name = candidate.file_name().ok_or_else(|| {
            format!(
                "Path '{}' must point to a file inside the session cwd.",
                candidate.display()
            )
        })?;
        resolved_parent.join(name)
    };
    if !resolved.starts_with(&root) {
        return Err(format!(
            "Path '{}' is outside the session cwd '{}'.",
            resolved.display(),
            root.display()
        ));
    }
    Ok(resolved)
}

fn copilot_acp_read_block_error(path: &Path) -> Option<String> {
    let hermes_home = env::var_os("HERMES_HOME").map(PathBuf::from)?;
    let resolved = path.canonicalize().ok()?;
    let blocked_dirs = [
        hermes_home.join("skills").join(".hub").join("index-cache"),
        hermes_home.join("skills").join(".hub"),
    ];
    for blocked in blocked_dirs {
        if resolved.starts_with(&blocked) {
            return Some(format!(
                "Access denied: {} is an internal Hermes cache file and cannot be read directly to prevent prompt injection. Use the skills_list or skill_view tools instead.",
                path.display()
            ));
        }
    }
    None
}

fn copilot_acp_is_write_denied(path: &Path) -> bool {
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|value| !value.as_os_str().is_empty())
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("/"));
    let hermes_home = env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".hermes"));

    let denied_exact = [
        home.join(".ssh").join("authorized_keys"),
        home.join(".ssh").join("id_rsa"),
        home.join(".ssh").join("id_ed25519"),
        home.join(".ssh").join("config"),
        hermes_home.join(".env"),
        home.join(".bashrc"),
        home.join(".zshrc"),
        home.join(".profile"),
        home.join(".bash_profile"),
        home.join(".zprofile"),
        home.join(".netrc"),
        home.join(".pgpass"),
        home.join(".npmrc"),
        home.join(".pypirc"),
        PathBuf::from("/etc/sudoers"),
        PathBuf::from("/etc/passwd"),
        PathBuf::from("/etc/shadow"),
    ];
    if denied_exact.iter().any(|candidate| path == candidate) {
        return true;
    }

    let denied_prefixes = [
        home.join(".ssh"),
        home.join(".aws"),
        home.join(".gnupg"),
        home.join(".kube"),
        PathBuf::from("/etc/sudoers.d"),
        PathBuf::from("/etc/systemd"),
        home.join(".docker"),
        home.join(".azure"),
        home.join(".config").join("gh"),
    ];
    if denied_prefixes
        .iter()
        .any(|prefix| path.starts_with(prefix))
    {
        return true;
    }

    if let Some(root) = env::var_os("HERMES_WRITE_SAFE_ROOT").map(PathBuf::from)
        && !root.as_os_str().is_empty()
        && path != root
        && !path.starts_with(&root)
    {
        return true;
    }
    false
}

fn exit_code_or_default(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

fn send_google_gemini_chat_completion(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
) -> Result<NormalizedAssistantResponse, HermesError> {
    let creds = google_gemini_request_credentials(runtime_model)?;
    let access_token = if !creds.access_token.trim().is_empty() {
        creds.access_token.clone()
    } else {
        runtime_model.api_key.trim().to_string()
    };
    if access_token.trim().is_empty() {
        return Err(HermesError::State {
            action: "calling Google Code Assist",
            detail:
                "No Google OAuth access token resolved. Run `hermes auth add google-gemini-cli` first."
                    .to_string(),
        });
    }

    let project_context =
        resolve_google_project_context(client, runtime_model, &access_token, &creds)?;
    let request = build_google_gemini_request(messages, tools);
    let wrapped = json!({
        "project": project_context.project_id,
        "model": runtime_model.model,
        "user_prompt_id": format!("{:x}", unix_ts_nanos()),
        "request": request,
    });
    let url = format!(
        "{}/v1internal:generateContent",
        google_code_assist_primary_base_url().trim_end_matches('/')
    );
    let response = client.post(&url).json(&wrapped);
    let response = apply_google_code_assist_headers(
        response,
        runtime_model,
        &access_token,
        &runtime_model.model,
        false,
    )?
    .send()
    .map_err(|error| HermesError::State {
        action: "calling Google Code Assist",
        detail: error.to_string(),
    })?;
    let body = read_json_response(response, "calling Google Code Assist")?;
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Google Code Assist response",
        detail: format!("{error}: {body}"),
    })?;
    normalize_google_gemini_response(&parsed)
}

fn google_gemini_request_credentials(
    runtime_model: &crate::ModelRuntimeConfig,
) -> Result<crate::GoogleGeminiRuntimeCredentials, HermesError> {
    let hermes_home = crate::HermesContext::detect().hermes_home();
    match crate::resolve_google_gemini_runtime_credentials(&hermes_home) {
        Ok(creds) => Ok(creds),
        Err(error) if !runtime_model.api_key.trim().is_empty() => {
            log::warn!(target: "run_agent", "google oauth state unavailable, falling back to runtime token: {error}");
            Ok(crate::GoogleGeminiRuntimeCredentials {
                access_token: runtime_model.api_key.clone(),
                refresh_token: String::new(),
                project_id: String::new(),
                managed_project_id: String::new(),
                email: String::new(),
            })
        }
        Err(error) => Err(error),
    }
}

fn resolve_google_project_context(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    access_token: &str,
    creds: &crate::GoogleGeminiRuntimeCredentials,
) -> Result<GoogleProjectContext, HermesError> {
    if let Some(project_id) = google_project_id_from_env() {
        return Ok(GoogleProjectContext {
            project_id,
            managed_project_id: String::new(),
        });
    }
    if !creds.project_id.trim().is_empty() {
        return Ok(GoogleProjectContext {
            project_id: creds.project_id.clone(),
            managed_project_id: creds.managed_project_id.clone(),
        });
    }

    let mut last_error = None;
    for endpoint in google_code_assist_probe_base_urls() {
        match load_google_code_assist(client, &endpoint, access_token, &runtime_model.model) {
            Ok((tier_id, project_id)) => {
                let context = if project_id.trim().is_empty() && tier_id.trim().is_empty() {
                    onboard_google_code_assist(
                        client,
                        &endpoint,
                        access_token,
                        &runtime_model.model,
                    )?
                } else {
                    GoogleProjectContext {
                        project_id: project_id.clone(),
                        managed_project_id: if tier_id == GOOGLE_CODE_ASSIST_FREE_TIER_ID {
                            project_id
                        } else {
                            String::new()
                        },
                    }
                };
                if !context.project_id.trim().is_empty()
                    || !context.managed_project_id.trim().is_empty()
                {
                    let hermes_home = crate::HermesContext::detect().hermes_home();
                    if let Err(error) = crate::auth::persist_google_gemini_project_ids(
                        &hermes_home,
                        &context.project_id,
                        &context.managed_project_id,
                    ) {
                        log::warn!(target: "run_agent", "google project persistence skipped: {error}");
                    }
                }
                return Ok(context);
            }
            Err(error) => {
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| HermesError::State {
        action: "resolving Google Code Assist project",
        detail: "Code Assist project discovery failed.".to_string(),
    }))
}

fn google_project_id_from_env() -> Option<String> {
    for key in [
        "HERMES_GEMINI_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_PROJECT_ID",
    ] {
        if let Some(value) = env::var(key).ok().as_deref().and_then(non_empty_trimmed) {
            return Some(value);
        }
    }
    None
}

fn google_code_assist_primary_base_url() -> String {
    env::var(GOOGLE_CODE_ASSIST_BASE_URL_ENV)
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
        .unwrap_or_else(|| GOOGLE_CODE_ASSIST_ENDPOINT.to_string())
}

fn google_code_assist_probe_base_urls() -> Vec<String> {
    if let Some(override_url) = env::var(GOOGLE_CODE_ASSIST_BASE_URL_ENV)
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
    {
        return vec![override_url];
    }
    let mut urls = Vec::with_capacity(1 + GOOGLE_CODE_ASSIST_FALLBACK_ENDPOINTS.len());
    urls.push(GOOGLE_CODE_ASSIST_ENDPOINT.to_string());
    urls.extend(
        GOOGLE_CODE_ASSIST_FALLBACK_ENDPOINTS
            .iter()
            .map(|endpoint| (*endpoint).to_string()),
    );
    urls
}

fn google_code_assist_metadata(project_id: &str) -> Value {
    json!({
        "duetProject": project_id,
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    })
}

fn load_google_code_assist(
    client: &Client,
    endpoint: &str,
    access_token: &str,
    model: &str,
) -> Result<(String, String), HermesError> {
    let payload = json!({
        "metadata": google_code_assist_metadata(""),
    });
    let response = client
        .post(format!(
            "{}/v1internal:loadCodeAssist",
            endpoint.trim_end_matches('/')
        ))
        .json(&payload);
    let response = apply_google_code_assist_headers(
        response,
        &empty_google_runtime_model(),
        access_token,
        model,
        true,
    )?
    .send()
    .map_err(|error| HermesError::State {
        action: "loading Google Code Assist account state",
        detail: error.to_string(),
    })?;
    let body = read_json_response(response, "loading Google Code Assist account state")?;
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Google Code Assist account state",
        detail: format!("{error}: {body}"),
    })?;
    let tier_id = parsed
        .get("currentTier")
        .and_then(Value::as_object)
        .and_then(|tier| tier.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let project_id = parsed
        .get("cloudaicompanionProject")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Ok((tier_id, project_id))
}

fn onboard_google_code_assist(
    client: &Client,
    endpoint: &str,
    access_token: &str,
    model: &str,
) -> Result<GoogleProjectContext, HermesError> {
    let payload = json!({
        "tierId": GOOGLE_CODE_ASSIST_FREE_TIER_ID,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    });
    let response = client
        .post(format!(
            "{}/v1internal:onboardUser",
            endpoint.trim_end_matches('/')
        ))
        .json(&payload);
    let response = apply_google_code_assist_headers(
        response,
        &empty_google_runtime_model(),
        access_token,
        model,
        true,
    )?
    .send()
    .map_err(|error| HermesError::State {
        action: "onboarding Google Code Assist account",
        detail: error.to_string(),
    })?;
    let body = read_json_response(response, "onboarding Google Code Assist account")?;
    let parsed = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Google Code Assist onboarding response",
        detail: format!("{error}: {body}"),
    })?;
    let project_id = parsed
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("cloudaicompanionProject"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Ok(GoogleProjectContext {
        managed_project_id: project_id.clone(),
        project_id,
    })
}

fn build_google_gemini_request(messages: &[Value], tools: &[crate::ToolDefinition]) -> Value {
    let (contents, system_instruction) = build_google_gemini_contents(messages);
    let mut request = json!({
        "contents": contents,
    });
    if let Some(system_instruction) = system_instruction {
        request["systemInstruction"] = system_instruction;
    }
    let translated_tools = translate_google_gemini_tools(tools);
    if !translated_tools.is_empty() {
        request["tools"] = Value::Array(translated_tools);
        request["toolConfig"] = json!({
            "functionCallingConfig": {
                "mode": "AUTO",
            }
        });
    }
    request
}

fn build_google_gemini_contents(messages: &[Value]) -> (Vec<Value>, Option<Value>) {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut tool_call_names = HashMap::<String, String>::new();

    for message in messages {
        let Some(object) = message.as_object() else {
            continue;
        };
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .trim();

        if role == "system" {
            let text = coerce_google_gemini_text(object.get("content"));
            if !text.is_empty() {
                system_parts.push(text);
            }
            continue;
        }

        if role == "tool" || role == "function" {
            let tool_call_id = object
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .and_then(|value| {
                    let trimmed = value.trim();
                    (!trimmed.is_empty()).then_some(trimmed)
                })
                .map(ToOwned::to_owned)
                .or_else(|| tool_call_names.get(tool_call_id).cloned())
                .unwrap_or_else(|| {
                    if tool_call_id.is_empty() {
                        "tool".to_string()
                    } else {
                        tool_call_id.to_string()
                    }
                });
            let content = coerce_google_gemini_text(object.get("content"));
            let response = parse_google_tool_result_payload(&content);
            contents.push(json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "response": response,
                    }
                }]
            }));
            continue;
        }

        let mut parts = Vec::new();
        let text = coerce_google_gemini_text(object.get("content"));
        if !text.is_empty() {
            parts.push(json!({ "text": text }));
        }
        if role == "assistant"
            && let Some(tool_calls) = object.get("tool_calls").and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                let Some(tool_object) = tool_call.as_object() else {
                    continue;
                };
                let function = tool_object
                    .get("function")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if name.is_empty() {
                    continue;
                }
                if let Some(tool_id) = tool_object.get("id").and_then(Value::as_str)
                    && !tool_id.trim().is_empty()
                {
                    tool_call_names.insert(tool_id.trim().to_string(), name.clone());
                }
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(parse_google_tool_arguments)
                    .unwrap_or_else(|| json!({}));
                parts.push(json!({
                    "functionCall": {
                        "name": name,
                        "args": arguments,
                    },
                    "thoughtSignature": "skip_thought_signature_validator",
                }));
            }
        }
        if parts.is_empty() {
            continue;
        }
        contents.push(json!({
            "role": if role == "assistant" { "model" } else { "user" },
            "parts": parts,
        }));
    }

    let system_instruction = (!system_parts.is_empty()).then(|| {
        json!({
            "role": "system",
            "parts": [{
                "text": system_parts.join("\n"),
            }]
        })
    });
    (contents, system_instruction)
}

fn coerce_google_gemini_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(text) => text.trim().to_string(),
        Value::Array(parts) => {
            let mut pieces = Vec::new();
            for part in parts {
                match part {
                    Value::String(text) => {
                        if !text.trim().is_empty() {
                            pieces.push(text.trim().to_string());
                        }
                    }
                    Value::Object(object) => {
                        let part_type = object
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if matches!(part_type, "text" | "input_text" | "output_text")
                            && let Some(text) = object.get("text").and_then(Value::as_str)
                            && !text.trim().is_empty()
                        {
                            pieces.push(text.trim().to_string());
                        }
                    }
                    _ => {}
                }
            }
            pieces.join("\n")
        }
        _ => String::new(),
    }
}

fn parse_google_tool_arguments(arguments_raw: &str) -> Value {
    let parsed = serde_json::from_str::<Value>(arguments_raw).unwrap_or_else(|_| json!({}));
    if parsed.is_object() {
        parsed
    } else {
        json!({ "_value": parsed })
    }
}

fn parse_google_tool_result_payload(content: &str) -> Value {
    let trimmed = content.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(parsed) = serde_json::from_str::<Value>(trimmed)
        && parsed.is_object()
    {
        return parsed;
    }
    json!({ "output": content })
}

fn translate_google_gemini_tools(tools: &[crate::ToolDefinition]) -> Vec<Value> {
    let mut declarations = Vec::new();
    for tool in tools {
        let mut declaration = Map::new();
        declaration.insert("name".to_string(), Value::String(tool.name.clone()));
        if !tool.description.trim().is_empty() {
            declaration.insert(
                "description".to_string(),
                Value::String(tool.description.clone()),
            );
        }
        let parameters = tool
            .schema
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
        declaration.insert(
            "parameters".to_string(),
            sanitize_google_gemini_schema(&parameters),
        );
        declarations.push(Value::Object(declaration));
    }
    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![json!({ "functionDeclarations": declarations })]
    }
}

fn sanitize_google_gemini_schema(schema: &Value) -> Value {
    const ALLOWED_KEYS: &[&str] = &[
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
        "propertyOrdering",
        "default",
        "items",
        "minimum",
        "maximum",
    ];
    let Some(object) = schema.as_object() else {
        return json!({ "type": "object", "properties": {} });
    };
    let mut cleaned = Map::new();
    for (key, value) in object {
        if !ALLOWED_KEYS.contains(&key.as_str()) {
            continue;
        }
        match key.as_str() {
            "properties" => {
                let Some(properties) = value.as_object() else {
                    continue;
                };
                let mut nested = Map::new();
                for (prop_name, prop_schema) in properties {
                    nested.insert(
                        prop_name.clone(),
                        sanitize_google_gemini_schema(prop_schema),
                    );
                }
                cleaned.insert(key.clone(), Value::Object(nested));
            }
            "items" => {
                cleaned.insert(key.clone(), sanitize_google_gemini_schema(value));
            }
            "anyOf" => {
                let Some(items) = value.as_array() else {
                    continue;
                };
                cleaned.insert(
                    key.clone(),
                    Value::Array(
                        items
                            .iter()
                            .filter(|item| item.is_object())
                            .map(sanitize_google_gemini_schema)
                            .collect(),
                    ),
                );
            }
            _ => {
                cleaned.insert(key.clone(), value.clone());
            }
        }
    }
    let drop_enum = cleaned
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|value| matches!(value, "integer" | "number" | "boolean"))
        && cleaned
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|item| !item.is_string()));
    if drop_enum {
        cleaned.remove("enum");
    }
    if cleaned.is_empty() {
        json!({ "type": "object", "properties": {} })
    } else {
        Value::Object(cleaned)
    }
}

fn normalize_google_gemini_response(
    response: &Value,
) -> Result<NormalizedAssistantResponse, HermesError> {
    let inner = response.get("response").unwrap_or(response);
    let candidate = inner
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .ok_or_else(|| HermesError::State {
            action: "parsing Google Code Assist response",
            detail: format!("Response missing candidates[0]: {response}"),
        })?;

    let parts = candidate
        .get("content")
        .and_then(Value::as_object)
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut text = Vec::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();

    for (index, part) in parts.iter().enumerate() {
        let Some(object) = part.as_object() else {
            continue;
        };
        if object.get("thought").and_then(Value::as_bool) == Some(true) {
            if let Some(value) = object.get("text").and_then(Value::as_str)
                && !value.trim().is_empty()
            {
                reasoning.push(value.to_string());
            }
            continue;
        }
        if let Some(value) = object.get("text").and_then(Value::as_str)
            && !value.trim().is_empty()
        {
            text.push(value.to_string());
        }
        let Some(function_call) = object.get("functionCall").and_then(Value::as_object) else {
            continue;
        };
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        let args = function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let arguments_raw = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
        tool_calls.push(PendingToolCall {
            id: format!("call_{:x}_{index}", unix_ts_nanos()),
            name,
            arguments_raw,
            json: if args.is_object() {
                args
            } else {
                json!({ "_value": args })
            },
        });
    }

    let has_tool_calls = !tool_calls.is_empty();
    Ok(NormalizedAssistantResponse {
        content: (!text.is_empty()).then(|| text.join("")),
        tool_calls,
        finish_reason: Some(map_google_gemini_finish_reason(
            candidate
                .get("finishReason")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            has_tool_calls,
        )),
        reasoning: (!reasoning.is_empty()).then(|| reasoning.join("")),
        reasoning_content: None,
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
    })
}

fn map_google_gemini_finish_reason(reason: &str, has_tool_calls: bool) -> String {
    if has_tool_calls {
        return "tool_calls".to_string();
    }
    match reason.trim().to_ascii_uppercase().as_str() {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY" | "RECITATION" => "content_filter".to_string(),
        other if !other.is_empty() => other.to_ascii_lowercase(),
        _ => "stop".to_string(),
    }
}

fn apply_google_code_assist_headers(
    mut request: reqwest::blocking::RequestBuilder,
    runtime_model: &crate::ModelRuntimeConfig,
    access_token: &str,
    model: &str,
    control_plane: bool,
) -> Result<reqwest::blocking::RequestBuilder, HermesError> {
    let user_agent = if control_plane {
        format!("{GOOGLE_CODE_ASSIST_CONTROL_USER_AGENT} model/{model}")
    } else {
        GOOGLE_CODE_ASSIST_INFERENCE_USER_AGENT.to_string()
    };
    request = request
        .header("Content-Type", "application/json")
        .header(
            "Accept",
            if control_plane {
                "application/json"
            } else {
                "application/json"
            },
        )
        .bearer_auth(access_token)
        .header("User-Agent", user_agent)
        .header(
            "X-Goog-Api-Client",
            if control_plane {
                GOOGLE_CODE_ASSIST_CONTROL_API_CLIENT
            } else {
                GOOGLE_CODE_ASSIST_INFERENCE_API_CLIENT
            },
        )
        .header(
            "x-activity-request-id",
            format!("hermes-{:x}", unix_ts_nanos()),
        );
    for (name, value) in &runtime_model.default_headers {
        request = request.header(name, value);
    }
    Ok(request)
}

fn empty_google_runtime_model() -> crate::ModelRuntimeConfig {
    crate::ModelRuntimeConfig {
        model: String::new(),
        provider: "google-gemini-cli".to_string(),
        base_url: "cloudcode-pa://google".to_string(),
        api_key: String::new(),
        api_mode: "chat_completions".to_string(),
        auth_type: "oauth_external".to_string(),
        default_headers: Vec::new(),
    }
}

fn responses_tools(tools: &[crate::ToolDefinition]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "strict": false,
                    "parameters": tool
                        .schema
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                })
            })
            .collect(),
    )
}

fn chat_messages_to_responses_input(messages: &[Value]) -> Result<Value, HermesError> {
    let mut items = Vec::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if role == "system" {
            continue;
        }

        if role == "assistant" {
            if let Some(codex_reasoning_items) = message
                .get("codex_reasoning_items")
                .and_then(Value::as_array)
                .cloned()
            {
                items.extend(codex_reasoning_items);
            }
            if let Some(codex_message_items) = message
                .get("codex_message_items")
                .and_then(Value::as_array)
                .cloned()
            {
                items.extend(codex_message_items);
            } else if let Some(text) = extract_message_text(message.get("content"))
                && !text.trim().is_empty()
            {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }));
            }

            for tool_call in parse_tool_calls(message.get("tool_calls")) {
                let (call_id, response_item_id) = split_responses_tool_id(&tool_call.id);
                let call_id = call_id.unwrap_or_else(|| tool_call.id.clone());
                items.push(json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_raw,
                    "id": response_item_id.unwrap_or_else(|| derive_responses_function_call_id(&call_id, None)),
                }));
            }
            continue;
        }

        if role == "tool" {
            let raw_tool_call_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let (call_id, _response_item_id) = split_responses_tool_id(raw_tool_call_id);
            let Some(call_id) = call_id.or_else(|| non_empty_trimmed(raw_tool_call_id)) else {
                continue;
            };
            let output = extract_message_text(message.get("content"))
                .unwrap_or_else(|| String::from("(no output)"));
            items.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }));
            continue;
        }

        let content = responses_content_parts(message.get("content"), false);
        items.push(json!({
            "role": "user",
            "content": content,
        }));
    }

    Ok(Value::Array(items))
}

fn convert_messages_to_anthropic(
    messages: &[Value],
    runtime_model: &crate::ModelRuntimeConfig,
) -> Result<(Option<String>, Vec<Value>), HermesError> {
    let mut system_prompt = None;
    let mut converted = Vec::new();
    let preserve_unsigned_thinking = preserve_unsigned_thinking_for_anthropic(runtime_model);

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        match role {
            "system" => {
                system_prompt = extract_message_text(message.get("content"));
            }
            "assistant" => {
                if is_thinking_only_assistant_message(message) {
                    continue;
                }
                let mut blocks = Vec::new();
                if let Some(content_blocks) = anthropic_preserved_assistant_blocks(message) {
                    blocks.extend(content_blocks);
                }
                for tool_call in parse_tool_calls(message.get("tool_calls")) {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.json,
                    }));
                }
                let has_thinking = blocks
                    .iter()
                    .any(|block| block.get("type").and_then(Value::as_str) == Some("thinking"));
                if preserve_unsigned_thinking
                    && !has_thinking
                    && message
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| !calls.is_empty())
                    && let Some(reasoning_content) =
                        message.get("reasoning_content").and_then(Value::as_str)
                {
                    blocks.insert(
                        0,
                        json!({
                            "type": "thinking",
                            "thinking": reasoning_content,
                        }),
                    );
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": "(empty)"}));
                }
                converted.push(json!({
                    "role": "assistant",
                    "content": blocks,
                }));
            }
            "tool" => {
                let tool_call_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let text = extract_message_text(message.get("content"))
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "(no output)".to_string());
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": text,
                });
                if let Some(last) = converted.last_mut()
                    && last.get("role").and_then(Value::as_str) == Some("user")
                    && last
                        .get("content")
                        .and_then(Value::as_array)
                        .and_then(|items| items.first())
                        .and_then(|item| item.get("type"))
                        .and_then(Value::as_str)
                        == Some("tool_result")
                    && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
                {
                    content.push(block);
                } else {
                    push_anthropic_message(&mut converted, "user", Value::Array(vec![block]));
                }
            }
            _ => {
                let content = extract_message_text(message.get("content"))
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "(empty message)".to_string());
                push_anthropic_message(&mut converted, "user", Value::String(content));
            }
        }
    }

    if system_prompt
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        system_prompt = None;
    }
    sanitize_anthropic_thinking_blocks(&mut converted, runtime_model);
    Ok((system_prompt, converted))
}

fn preserve_unsigned_thinking_for_anthropic(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    is_kimi_family_anthropic_endpoint(&runtime_model.base_url, &runtime_model.model)
        || is_deepseek_anthropic_endpoint(&runtime_model.base_url)
}

fn is_third_party_anthropic_endpoint(runtime_model: &crate::ModelRuntimeConfig) -> bool {
    !base_url_host_matches_any(&runtime_model.base_url, &["api.anthropic.com"])
}

fn is_kimi_family_anthropic_endpoint(base_url: &str, model: &str) -> bool {
    base_url_host_matches_any(base_url, &["api.kimi.com", "moonshot.ai", "moonshot.cn"])
        || model_name_is_kimi_family(model)
}

fn model_name_is_kimi_family(model: &str) -> bool {
    let mut normalized = model.trim().to_ascii_lowercase();
    if let Some((_, suffix)) = normalized.rsplit_once('/') {
        normalized = suffix.to_string();
    }
    [
        "kimi-",
        "kimi_",
        "moonshot-",
        "moonshot_",
        "k1.",
        "k1-",
        "k2.",
        "k2-",
        "k25",
        "k2.5",
    ]
    .iter()
    .any(|prefix| normalized.starts_with(prefix))
}

fn is_deepseek_anthropic_endpoint(base_url: &str) -> bool {
    if !base_url_host_matches_any(base_url, &["api.deepseek.com"]) {
        return false;
    }
    base_url
        .trim()
        .trim_end_matches('/')
        .to_ascii_lowercase()
        .contains("/anthropic")
}

fn anthropic_preserved_assistant_blocks(message: &Value) -> Option<Vec<Value>> {
    let mut blocks = extract_reasoning_detail_thinking_blocks(message);
    match message.get("content") {
        Some(Value::Array(parts)) => {
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or_default();
                match part_type {
                    "thinking" | "redacted_thinking" => blocks.push(part.clone()),
                    "text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str)
                            && !text.trim().is_empty()
                        {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(Value::String(text)) if !text.trim().is_empty() => {
            blocks.push(json!({"type": "text", "text": text}));
        }
        _ => {}
    }
    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

fn extract_reasoning_detail_thinking_blocks(message: &Value) -> Vec<Value> {
    message
        .get("reasoning_details")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|detail| {
            let block_type = detail
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            matches!(block_type, "thinking" | "redacted_thinking").then(|| detail.clone())
        })
        .collect()
}

fn sanitize_anthropic_thinking_blocks(
    converted: &mut [Value],
    runtime_model: &crate::ModelRuntimeConfig,
) {
    let preserve_unsigned = preserve_unsigned_thinking_for_anthropic(runtime_model);
    let is_third_party = is_third_party_anthropic_endpoint(runtime_model);
    let last_assistant_idx = converted
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, message)| {
            (message.get("role").and_then(Value::as_str) == Some("assistant")).then_some(idx)
        });

    for (idx, message) in converted.iter_mut().enumerate() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };

        let mut new_content = Vec::new();
        for block in content.iter() {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !matches!(block_type, "thinking" | "redacted_thinking") {
                new_content.push(block.clone());
                continue;
            }
            if preserve_unsigned {
                let has_signature = block.get("signature").is_some() || block.get("data").is_some();
                if !has_signature {
                    new_content.push(block.clone());
                }
                continue;
            }
            if is_third_party || Some(idx) != last_assistant_idx {
                continue;
            }
            if block_type == "redacted_thinking" {
                if block.get("data").is_some() {
                    new_content.push(block.clone());
                }
                continue;
            }
            if block.get("signature").is_some() {
                new_content.push(block.clone());
            } else if let Some(text) = block.get("thinking").and_then(Value::as_str)
                && !text.trim().is_empty()
            {
                new_content.push(json!({"type": "text", "text": text}));
            }
        }
        for block in &mut new_content {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if matches!(block_type, "thinking" | "redacted_thinking")
                && let Some(object) = block.as_object_mut()
            {
                object.remove("cache_control");
            }
        }
        if new_content.is_empty() {
            new_content.push(json!({
                "type": "text",
                "text": if preserve_unsigned {
                    "(empty)"
                } else if is_third_party || Some(idx) != last_assistant_idx {
                    "(thinking elided)"
                } else {
                    "(empty)"
                }
            }));
        }
        *content = new_content;
    }
}

fn push_anthropic_message(converted: &mut Vec<Value>, role: &str, content: Value) {
    if role == "user"
        && let Some(last) = converted.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
    {
        let merged = merge_anthropic_content(last.get("content"), &content);
        if let Some(object) = last.as_object_mut() {
            object.insert("content".to_string(), merged);
            return;
        }
    }
    converted.push(json!({
        "role": role,
        "content": content,
    }));
}

fn merge_anthropic_content(existing: Option<&Value>, next: &Value) -> Value {
    let mut merged = anthropic_content_to_blocks(existing);
    merged.extend(anthropic_content_to_blocks(Some(next)));
    Value::Array(merged)
}

fn anthropic_content_to_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(text)) => vec![json!({"type": "text", "text": text})],
        Some(other) => vec![other.clone()],
        None => Vec::new(),
    }
}

fn extract_message_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let mut chunks = Vec::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.trim().is_empty() {
                        chunks.push(text.to_string());
                    }
                    continue;
                }
                if part.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    chunks.push(text.to_string());
                }
            }
            if chunks.is_empty() {
                None
            } else {
                Some(chunks.join("\n"))
            }
        }
        _ => None,
    }
}

fn extract_chat_reasoning_text(message: &Value, field: &str) -> Option<String> {
    let value = message.get(field)?;
    match value {
        Value::String(text) => non_empty_trimmed(text),
        Value::Array(parts) => {
            let chunks = parts
                .iter()
                .filter_map(|part| {
                    part.as_str()
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .map(ToOwned::to_owned)
                        .or_else(|| {
                            part.get("text")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|text| !text.is_empty())
                                .map(ToOwned::to_owned)
                        })
                        .or_else(|| {
                            part.get("summary")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|text| !text.is_empty())
                                .map(ToOwned::to_owned)
                        })
                })
                .collect::<Vec<_>>();
            if chunks.is_empty() {
                None
            } else {
                Some(chunks.join("\n\n"))
            }
        }
        _ => None,
    }
}

fn responses_content_parts(content: Option<&Value>, assistant: bool) -> Value {
    let text_type = if assistant {
        "output_text"
    } else {
        "input_text"
    };
    match content {
        Some(Value::String(text)) => Value::Array(vec![json!({"type": text_type, "text": text})]),
        Some(Value::Array(parts)) => {
            let mut normalized = Vec::new();
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or_default();
                match part_type {
                    "text" | "input_text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            normalized.push(json!({"type": text_type, "text": text}));
                        }
                    }
                    "image_url" => {
                        if let Some(url) = part
                            .get("image_url")
                            .and_then(Value::as_object)
                            .and_then(|image| image.get("url"))
                            .and_then(Value::as_str)
                        {
                            normalized.push(json!({
                                "type": "input_image",
                                "image_url": url,
                            }));
                        }
                    }
                    _ => {}
                }
            }
            if normalized.is_empty() {
                Value::Array(vec![json!({"type": text_type, "text": ""})])
            } else {
                Value::Array(normalized)
            }
        }
        _ => Value::Array(vec![json!({"type": text_type, "text": ""})]),
    }
}

fn parse_tool_calls(value: Option<&Value>) -> Vec<PendingToolCall> {
    let Some(tool_calls) = value.and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for tool_call in tool_calls {
        let Some(id) = tool_call
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let Some(function) = tool_call.get("function") else {
            continue;
        };
        let Some(name) = function
            .get("name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let arguments_raw = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}")
            .to_string();
        let json = serde_json::from_str::<Value>(&arguments_raw)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        result.push(PendingToolCall {
            id,
            name,
            arguments_raw,
            json,
        });
    }
    result
}

fn parse_anthropic_tool_calls(value: Option<&Value>) -> Vec<PendingToolCall> {
    let Some(blocks) = value.and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("tool_use") {
            continue;
        }
        let Some(id) = block
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let Some(name) = block
            .get("name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let json = block
            .get("input")
            .cloned()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let arguments_raw = serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string());
        result.push(PendingToolCall {
            id,
            name,
            arguments_raw,
            json,
        });
    }
    result
}

fn map_anthropic_finish_reason(reason: &str) -> String {
    match reason {
        "tool_use" => "tool_calls".to_string(),
        "max_tokens" | "model_context_window_exceeded" => "length".to_string(),
        "refusal" => "content_filter".to_string(),
        _ => "stop".to_string(),
    }
}

fn normalize_codex_response(response: &Value) -> Result<NormalizedAssistantResponse, HermesError> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| {
            response
                .get("output_text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(|text| {
                    vec![json!({
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text}],
                    })]
                })
        })
        .ok_or_else(|| HermesError::State {
            action: "parsing codex response",
            detail: "Responses API returned no output items.".to_string(),
        })?;

    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut reasoning_items = Vec::new();
    let mut message_items = Vec::new();
    let mut tool_calls = Vec::new();
    let mut incomplete = response
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "queued" | "in_progress" | "incomplete"));

    for item in output {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        let item_status = item
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(item_status, "queued" | "in_progress" | "incomplete") {
            incomplete = true;
        }

        match item_type {
            "message" => {
                let text = extract_responses_message_text(&item);
                if !text.trim().is_empty() {
                    content_parts.push(text.clone());
                    let mut raw_item = json!({
                        "type": "message",
                        "role": "assistant",
                        "status": normalize_responses_message_status(item_status),
                        "content": [{"type": "output_text", "text": text}],
                    });
                    if let Some(id) = item.get("id").and_then(Value::as_str)
                        && let Some(object) = raw_item.as_object_mut()
                    {
                        object.insert("id".to_string(), Value::String(id.to_string()));
                    }
                    if let Some(phase) = item.get("phase").and_then(Value::as_str)
                        && !phase.trim().is_empty()
                        && let Some(object) = raw_item.as_object_mut()
                    {
                        object.insert("phase".to_string(), Value::String(phase.trim().to_string()));
                    }
                    message_items.push(raw_item);
                }
            }
            "reasoning" => {
                if let Some(text) = extract_responses_reasoning_text(&item)
                    && !text.trim().is_empty()
                {
                    reasoning_parts.push(text);
                }
                if let Some(encrypted) = item.get("encrypted_content").and_then(Value::as_str)
                    && !encrypted.trim().is_empty()
                {
                    let mut raw_item = json!({
                        "type": "reasoning",
                        "encrypted_content": encrypted,
                    });
                    if let Some(id) = item.get("id").and_then(Value::as_str)
                        && let Some(object) = raw_item.as_object_mut()
                    {
                        object.insert("id".to_string(), Value::String(id.to_string()));
                    }
                    if let Some(summary) = item.get("summary").and_then(Value::as_array)
                        && let Some(object) = raw_item.as_object_mut()
                    {
                        object.insert("summary".to_string(), Value::Array(summary.clone()));
                    }
                    reasoning_items.push(raw_item);
                }
            }
            "function_call" => {
                if matches!(item_status, "queued" | "in_progress" | "incomplete") {
                    continue;
                }
                if let Some(tool_call) = parse_codex_tool_call(&item, false) {
                    tool_calls.push(tool_call);
                }
            }
            "custom_tool_call" => {
                if let Some(tool_call) = parse_codex_tool_call(&item, true) {
                    tool_calls.push(tool_call);
                }
            }
            _ => {}
        }
    }

    let content = if content_parts.is_empty() {
        response
            .get("output_text")
            .and_then(Value::as_str)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
    } else {
        Some(content_parts.join("\n").trim().to_string())
    };

    let finish_reason = if !tool_calls.is_empty() {
        Some(String::from("tool_calls"))
    } else if incomplete {
        Some(String::from("length"))
    } else {
        Some(String::from("stop"))
    };

    Ok(NormalizedAssistantResponse {
        content,
        tool_calls,
        finish_reason,
        reasoning: (!reasoning_parts.is_empty()).then(|| reasoning_parts.join("\n\n")),
        reasoning_content: None,
        reasoning_details: None,
        codex_reasoning_items: (!reasoning_items.is_empty()).then(|| Value::Array(reasoning_items)),
        codex_message_items: (!message_items.is_empty()).then(|| Value::Array(message_items)),
    })
}

fn parse_codex_tool_call(item: &Value, custom_input: bool) -> Option<PendingToolCall> {
    let name = item.get("name").and_then(Value::as_str)?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments_raw = if custom_input {
        match item.get("input") {
            Some(Value::String(text)) => text.clone(),
            Some(value) => serde_json::to_string(value).ok()?,
            None => String::from("{}"),
        }
    } else {
        match item.get("arguments") {
            Some(Value::String(text)) => text.clone(),
            Some(value) => serde_json::to_string(value).ok()?,
            None => String::from("{}"),
        }
    };
    let json = serde_json::from_str::<Value>(&arguments_raw)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            item.get("id")
                .and_then(Value::as_str)
                .and_then(|value| split_responses_tool_id(value).0)
        })
        .unwrap_or_else(|| format!("call_{}", unix_ts_nanos()));
    let response_item_id =
        derive_responses_function_call_id(&call_id, item.get("id").and_then(Value::as_str));
    Some(PendingToolCall {
        id: format!("{call_id}|{response_item_id}"),
        name,
        arguments_raw,
        json,
    })
}

fn extract_responses_message_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| {
                    let part_type = part.get("type").and_then(Value::as_str).unwrap_or_default();
                    matches!(part_type, "output_text" | "text")
                        .then(|| part.get("text").and_then(Value::as_str))
                        .flatten()
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn extract_responses_reasoning_text(item: &Value) -> Option<String> {
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        let text = summary
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    item.get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_responses_message_status(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "completed" | "incomplete" | "in_progress" => {
            value.trim().to_ascii_lowercase().replace('-', "_")
        }
        _ => String::from("completed"),
    }
}

fn split_responses_tool_id(value: &str) -> (Option<String>, Option<String>) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return (None, None);
    }
    if let Some((call_id, response_item_id)) = trimmed.split_once('|') {
        let call_id = non_empty_trimmed(call_id);
        let response_item_id = non_empty_trimmed(response_item_id);
        return (call_id, response_item_id);
    }
    if trimmed.starts_with("fc_") {
        return (None, Some(trimmed.to_string()));
    }
    (Some(trimmed.to_string()), None)
}

fn derive_responses_function_call_id(call_id: &str, response_item_id: Option<&str>) -> String {
    if let Some(existing) = response_item_id
        .map(str::trim)
        .filter(|value| value.starts_with("fc_") && !value.is_empty())
    {
        return existing.to_string();
    }
    let source = call_id.trim();
    if source.starts_with("fc_") {
        return source.to_string();
    }
    if let Some(suffix) = source.strip_prefix("call_")
        && !suffix.is_empty()
    {
        return format!("fc_{suffix}");
    }
    let sanitized = source
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        .collect::<String>();
    if sanitized.starts_with("fc_") {
        return sanitized;
    }
    if let Some(suffix) = sanitized.strip_prefix("call_")
        && !suffix.is_empty()
    {
        return format!("fc_{suffix}");
    }
    if !sanitized.is_empty() {
        return format!("fc_{}", sanitized.chars().take(48).collect::<String>());
    }
    format!("fc_{:x}", unix_ts_nanos())
}

fn message_record_to_chat_message(message: crate::MessageRecord) -> Result<Value, HermesError> {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String(message.role.clone()));
    if let Some(content) = message.content {
        map.insert("content".to_string(), content);
    } else {
        map.insert("content".to_string(), Value::Null);
    }
    if let Some(tool_call_id) = message.tool_call_id {
        map.insert("tool_call_id".to_string(), Value::String(tool_call_id));
    }
    if let Some(tool_calls) = message.tool_calls {
        if !tool_calls.is_array() {
            return Err(HermesError::State {
                action: "loading session messages",
                detail: "Stored tool_calls payload was not an array.".to_string(),
            });
        }
        map.insert("tool_calls".to_string(), tool_calls);
    }
    if let Some(reasoning) = message.reasoning {
        map.insert("reasoning".to_string(), Value::String(reasoning));
    }
    if let Some(reasoning_content) = message.reasoning_content {
        map.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content),
        );
    }
    if let Some(reasoning_details) = message.reasoning_details {
        map.insert("reasoning_details".to_string(), reasoning_details);
    }
    if let Some(codex_reasoning_items) = message.codex_reasoning_items {
        map.insert("codex_reasoning_items".to_string(), codex_reasoning_items);
    }
    if let Some(codex_message_items) = message.codex_message_items {
        map.insert("codex_message_items".to_string(), codex_message_items);
    }
    Ok(Value::Object(map))
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn response_has_reasoning_signal(response: &NormalizedAssistantResponse) -> bool {
    response
        .reasoning
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        || response
            .reasoning_content
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        || response
            .reasoning_details
            .as_ref()
            .is_some_and(value_has_content)
        || response
            .codex_reasoning_items
            .as_ref()
            .is_some_and(value_has_content)
        || response
            .content
            .as_deref()
            .is_some_and(content_has_inline_thinking)
}

fn value_has_content(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(_) | Value::Number(_) => true,
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(items) => items.iter().any(value_has_content),
        Value::Object(map) => map.values().any(value_has_content),
    }
}

fn visible_assistant_text(content: Option<&str>) -> Option<String> {
    let content = content?;
    let stripped = strip_think_blocks(content);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn content_has_inline_thinking(content: &str) -> bool {
    let lowered = content.to_ascii_lowercase();
    [
        "<think",
        "<thinking",
        "<reasoning",
        "<thought",
        "<reasoning_scratchpad",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn strip_think_blocks(content: &str) -> String {
    let without_blocks = think_block_re().replace_all(content, "");
    let without_open_tail = think_open_tail_re().replace_all(&without_blocks, "");
    think_tag_re()
        .replace_all(&without_open_tail, "")
        .into_owned()
}

fn think_block_re() -> &'static Regex {
    THINK_BLOCK_RE.get_or_init(|| {
        Regex::new(
            r"(?is)<(?:think|thinking|reasoning|thought|reasoning_scratchpad)\b[^>]*>.*?</(?:think|thinking|reasoning|thought|reasoning_scratchpad)>",
        )
        .expect("valid think-block regex")
    })
}

fn think_open_tail_re() -> &'static Regex {
    THINK_OPEN_TAIL_RE.get_or_init(|| {
        Regex::new(
            r"(?is)(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|reasoning_scratchpad)\b[^>]*>.*$",
        )
        .expect("valid open-tail think regex")
    })
}

fn think_tag_re() -> &'static Regex {
    THINK_TAG_RE.get_or_init(|| {
        Regex::new(r"(?i)</?(?:think|thinking|reasoning|thought|reasoning_scratchpad)\b[^>]*>")
            .expect("valid think-tag regex")
    })
}

fn is_thinking_only_assistant_message(message: &Value) -> bool {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }
    if message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
    {
        return false;
    }
    if visible_assistant_text(message.get("content").and_then(Value::as_str)).is_some() {
        return false;
    }
    message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
        || message
            .get("reasoning")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
        || message
            .get("reasoning_details")
            .is_some_and(value_has_content)
}

fn value_to_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|raw| raw as f64))
        .or_else(|| value.as_u64().map(|raw| raw as f64))
        .or_else(|| {
            value
                .as_str()
                .and_then(|raw| raw.trim().parse::<f64>().ok())
        })
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn unix_ts_seconds_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn nous_rate_limit_state_path(ctx: &HermesContext) -> PathBuf {
    ctx.hermes_home()
        .join(NOUS_RATE_LIMIT_STATE_SUBDIR)
        .join(NOUS_RATE_LIMIT_STATE_FILE)
}

fn nous_rate_limit_remaining(ctx: &HermesContext) -> Option<f64> {
    let path = nous_rate_limit_state_path(ctx);
    let raw = fs::read_to_string(&path).ok()?;
    let state = serde_json::from_str::<NousRateLimitState>(&raw).ok()?;
    let remaining = state.reset_at - unix_ts_seconds_f64();
    if remaining > 0.0 {
        Some(remaining)
    } else {
        let _ = fs::remove_file(path);
        None
    }
}

fn clear_nous_rate_limit_state(ctx: &HermesContext) {
    let _ = fs::remove_file(nous_rate_limit_state_path(ctx));
}

fn format_duration_remaining(seconds: f64) -> String {
    let total = seconds.max(0.0).floor() as u64;
    if total < 60 {
        return format!("{total}s");
    }
    if total < 3600 {
        let minutes = total / 60;
        let secs = total % 60;
        return if secs > 0 {
            format!("{minutes}m {secs}s")
        } else {
            format!("{minutes}m")
        };
    }
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    if minutes > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{hours}h")
    }
}

fn parse_i64_header(headers: &reqwest::header::HeaderMap, name: &str) -> i64 {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .map(|parsed| parsed as i64)
        .unwrap_or_default()
}

fn parse_f64_header(headers: &reqwest::header::HeaderMap, name: &str) -> f64 {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .unwrap_or_default()
}

fn capture_nous_rate_limit_state_from_headers(headers: &reqwest::header::HeaderMap) {
    let has_any = headers.keys().any(|name| {
        name.as_str()
            .to_ascii_lowercase()
            .starts_with("x-ratelimit-")
    });
    if !has_any {
        return;
    }
    let state = NousObservedRateLimitState {
        requests_min: NousRateLimitBucketSnapshot {
            limit: parse_i64_header(headers, "x-ratelimit-limit-requests"),
            remaining: parse_i64_header(headers, "x-ratelimit-remaining-requests"),
            reset_seconds: parse_f64_header(headers, "x-ratelimit-reset-requests"),
        },
        requests_hour: NousRateLimitBucketSnapshot {
            limit: parse_i64_header(headers, "x-ratelimit-limit-requests-1h"),
            remaining: parse_i64_header(headers, "x-ratelimit-remaining-requests-1h"),
            reset_seconds: parse_f64_header(headers, "x-ratelimit-reset-requests-1h"),
        },
        tokens_min: NousRateLimitBucketSnapshot {
            limit: parse_i64_header(headers, "x-ratelimit-limit-tokens"),
            remaining: parse_i64_header(headers, "x-ratelimit-remaining-tokens"),
            reset_seconds: parse_f64_header(headers, "x-ratelimit-reset-tokens"),
        },
        tokens_hour: NousRateLimitBucketSnapshot {
            limit: parse_i64_header(headers, "x-ratelimit-limit-tokens-1h"),
            remaining: parse_i64_header(headers, "x-ratelimit-remaining-tokens-1h"),
            reset_seconds: parse_f64_header(headers, "x-ratelimit-reset-tokens-1h"),
        },
    };
    LAST_NOUS_RATE_LIMIT_STATE.with(|slot| {
        slot.replace(Some(state));
    });
}

fn clear_captured_nous_rate_limit_state() {
    LAST_NOUS_RATE_LIMIT_STATE.with(|slot| {
        slot.replace(None);
    });
}

fn nous_last_known_state_has_exhausted_bucket() -> bool {
    LAST_NOUS_RATE_LIMIT_STATE.with(|slot| {
        slot.borrow().as_ref().is_some_and(|state| {
            [
                &state.requests_min,
                &state.requests_hour,
                &state.tokens_min,
                &state.tokens_hour,
            ]
            .iter()
            .any(|bucket| {
                bucket.limit > 0
                    && bucket.remaining <= 0
                    && bucket.reset_seconds >= NOUS_RATE_LIMIT_MIN_BREAKER_RESET_SECONDS
            })
        })
    })
}

fn error_http_header_value(error: &HermesError, header_name: &str) -> Option<String> {
    let HermesError::State { detail, .. } = error else {
        return None;
    };
    let block = detail
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(headers, _)| headers)?;
    for item in block.split(';') {
        let (name, value) = item.trim().split_once('=')?;
        if name.trim().eq_ignore_ascii_case(header_name) {
            return non_empty_trimmed(value);
        }
    }
    None
}

fn parse_nous_rate_limit_reset_seconds(error: &HermesError) -> Option<f64> {
    for header_name in [
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-requests",
        "retry-after",
    ] {
        if let Some(value) = error_http_header_value(error, header_name)
            && let Ok(parsed) = value.trim().parse::<f64>()
            && parsed > 0.0
        {
            return Some(parsed);
        }
    }
    None
}

fn nous_rate_limit_looks_genuine(error: &HermesError) -> bool {
    if !matches!(error_http_status(error), Some(429)) {
        return false;
    }
    let reset_seconds = parse_nous_rate_limit_reset_seconds(error)
        .unwrap_or(NOUS_RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS);
    if reset_seconds < NOUS_RATE_LIMIT_MIN_BREAKER_RESET_SECONDS {
        return nous_last_known_state_has_exhausted_bucket();
    }
    for header_name in [
        "x-ratelimit-remaining-requests-1h",
        "x-ratelimit-remaining-requests",
    ] {
        if let Some(value) = error_http_header_value(error, header_name)
            && let Ok(parsed) = value.trim().parse::<f64>()
        {
            return parsed <= 0.0;
        }
    }
    error_http_header_value(error, "retry-after").is_some()
        || nous_last_known_state_has_exhausted_bucket()
}

fn record_nous_rate_limit_state(ctx: &HermesContext, error: &HermesError) {
    let now = unix_ts_seconds_f64();
    let reset_seconds = parse_nous_rate_limit_reset_seconds(error)
        .unwrap_or(NOUS_RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS);
    let state = NousRateLimitState {
        reset_at: now + reset_seconds.max(1.0),
        recorded_at: now,
        reset_seconds: reset_seconds.max(1.0),
    };
    let path = nous_rate_limit_state_path(ctx);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(payload) = serde_json::to_string_pretty(&state) {
        let _ = fs::write(path, format!("{payload}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use tempfile::TempDir;

    fn serve_chat_sequence(responses: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let responses = Arc::new(responses);
        let counter = Arc::new(AtomicUsize::new(0));

        thread::spawn({
            let responses = Arc::clone(&responses);
            let counter = Arc::clone(&counter);
            move || {
                for stream in listener.incoming().take(responses.len()) {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let mut content_length = 0usize;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or_default() == 0 {
                            break;
                        }
                        let trimmed = line.trim_end();
                        if trimmed.is_empty() {
                            break;
                        }
                        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                            content_length = value.trim().parse::<usize>().unwrap_or_default();
                        }
                    }
                    let mut body = vec![0_u8; content_length];
                    let _ = reader.read_exact(&mut body);
                    let idx = counter.fetch_add(1, Ordering::SeqCst);
                    let response = &responses[idx];
                    let http = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response.len(),
                        response
                    );
                    let _ = stream.write_all(http.as_bytes());
                }
            }
        });

        format!("http://{}", addr)
    }

    fn serve_http_sequence(responses: Vec<(u16, String)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let responses = Arc::new(responses);
        let counter = Arc::new(AtomicUsize::new(0));

        thread::spawn({
            let responses = Arc::clone(&responses);
            let counter = Arc::clone(&counter);
            move || {
                for stream in listener.incoming().take(responses.len()) {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let mut content_length = 0usize;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or_default() == 0 {
                            break;
                        }
                        let trimmed = line.trim_end();
                        if trimmed.is_empty() {
                            break;
                        }
                        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                            content_length = value.trim().parse::<usize>().unwrap_or_default();
                        }
                    }
                    let mut body = vec![0_u8; content_length];
                    let _ = reader.read_exact(&mut body);
                    let idx = counter.fetch_add(1, Ordering::SeqCst);
                    let (status, response) = &responses[idx];
                    let status_text = match *status {
                        200 => "OK",
                        413 => "Payload Too Large",
                        429 => "Too Many Requests",
                        500 => "Internal Server Error",
                        _ => "Error",
                    };
                    let http = format!(
                        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        status,
                        status_text,
                        response.len(),
                        response
                    );
                    let _ = stream.write_all(http.as_bytes());
                }
            }
        });

        format!("http://{}", addr)
    }

    fn large_png_data_url() -> String {
        let width = 1600_u32;
        let height = 1600_u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 13 + y * 7) % 256) as u8);
                pixels.push(((x * 5 + y * 11) % 256) as u8);
                pixels.push(((x * 17 + y * 3) % 256) as u8);
            }
        }
        let mut encoded = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut encoded);
        encoder
            .write_image(&pixels, width, height, image::ColorType::Rgb8.into())
            .unwrap();
        let data_url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(encoded)
        );
        assert!(data_url.len() > IMAGE_SHRINK_TARGET_BYTES);
        data_url
    }

    #[test]
    fn config_runtime_resolution_prefers_openrouter_for_auto() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "sk-or-test");
            std::env::remove_var("OPENAI_API_KEY");
        }
        let config_path = context.config_path();
        fs::write(
            &config_path,
            "model:\n  default: google/gemini-3-flash-preview\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = context
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .unwrap();
        assert_eq!(runtime.provider, "openrouter");
        assert_eq!(runtime.model, "google/gemini-3-flash-preview");
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
    }

    #[test]
    fn config_runtime_resolution_normalizes_anthropic_and_infers_minimax() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();

        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test");
            std::env::set_var("MINIMAX_API_KEY", "mmx-test");
        }

        fs::write(
            context.config_path(),
            "model:\n  default: anthropic/claude-sonnet-4.6\n  provider: claude\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let anthropic_runtime = context
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .unwrap();
        assert_eq!(anthropic_runtime.provider, "anthropic");
        assert_eq!(anthropic_runtime.model, "claude-sonnet-4-6");
        assert_eq!(anthropic_runtime.api_mode, "anthropic_messages");

        let minimax_runtime = context
            .resolve_model_runtime(
                &loaded,
                &ModelOverrides {
                    model: Some("minimax/MiniMax-M2.7".to_string()),
                    provider: Some("auto".to_string()),
                    base_url: Some("https://api.minimax.io/anthropic".to_string()),
                    api_key: None,
                    api_mode: None,
                },
            )
            .unwrap();
        assert_eq!(minimax_runtime.provider, "minimax");
        assert_eq!(minimax_runtime.model, "MiniMax-M2.7");
        assert_eq!(minimax_runtime.api_mode, "anthropic_messages");

        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("MINIMAX_API_KEY");
        }
    }

    #[test]
    fn qwen_oauth_chat_requests_include_portal_headers_and_payload_shape() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            assert!(request_line.starts_with("POST /v1/chat/completions "));

            let mut content_length = 0usize;
            let mut auth = String::new();
            let mut user_agent = String::new();
            let mut dashscope_cache = String::new();
            let mut dashscope_user_agent = String::new();
            let mut dashscope_auth_type = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end();
                if trimmed.is_empty() {
                    break;
                }
                let lower = trimmed.to_ascii_lowercase();
                if let Some(value) = lower.strip_prefix("content-length:") {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                } else if lower.starts_with("authorization:") {
                    auth = trimmed
                        .split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                        .unwrap_or_default();
                } else if lower.starts_with("user-agent:") {
                    user_agent = trimmed
                        .split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                        .unwrap_or_default();
                } else if lower.starts_with("x-dashscope-cachecontrol:") {
                    dashscope_cache = trimmed
                        .split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                        .unwrap_or_default();
                } else if lower.starts_with("x-dashscope-useragent:") {
                    dashscope_user_agent = trimmed
                        .split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                        .unwrap_or_default();
                } else if lower.starts_with("x-dashscope-authtype:") {
                    dashscope_auth_type = trimmed
                        .split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                        .unwrap_or_default();
                }
            }
            assert_eq!(auth, "Bearer qwen-token");
            assert_eq!(dashscope_cache, "enable");
            assert_eq!(dashscope_auth_type, "qwen-oauth");
            assert_eq!(dashscope_user_agent, user_agent);
            assert!(user_agent.starts_with("QwenCode/0.14.1 ("));

            let mut body = vec![0_u8; content_length];
            reader.read_exact(&mut body).unwrap();
            let payload = serde_json::from_slice::<Value>(&body).unwrap();
            assert_eq!(payload["vl_high_resolution_images"], Value::Bool(true));
            assert_eq!(payload["metadata"]["sessionId"], json!("hermes"));
            assert!(
                payload["metadata"]["promptId"]
                    .as_str()
                    .is_some_and(|value| !value.trim().is_empty())
            );
            let messages = payload["messages"].as_array().unwrap();
            assert!(messages[0]["content"].is_array());
            assert_eq!(messages[0]["content"][0]["text"], json!("Be helpful"));
            assert_eq!(
                messages[0]["content"][0]["cache_control"],
                json!({"type": "ephemeral"})
            );
            assert!(messages[1]["content"].is_array());
            assert_eq!(messages[1]["content"][0]["text"], json!("hello"));

            let body = json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Qwen portal smoke passed."},
                    "finish_reason": "stop"
                }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let client = build_http_client().unwrap();
        let runtime = crate::ModelRuntimeConfig {
            model: "qwen3.5-plus".to_string(),
            provider: "qwen-oauth".to_string(),
            base_url: format!("http://{addr}/v1"),
            api_key: "qwen-token".to_string(),
            api_mode: "chat_completions".to_string(),
            auth_type: "oauth_external".to_string(),
            default_headers: Vec::new(),
        };
        let messages = vec![
            json!({"role": "system", "content": "Be helpful"}),
            json!({"role": "user", "content": "hello"}),
        ];

        let result = request_model_text(&client, &runtime, &messages).unwrap();
        server.join().unwrap();
        assert_eq!(result.as_deref(), Some("Qwen portal smoke passed."));
    }

    #[test]
    fn qwen_oauth_chat_completion_retries_after_401_with_refreshed_runtime_token() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");
        let previous_qwen_token_url = env::var_os("HERMES_QWEN_OAUTH_TOKEN_URL");

        let temp = TempDir::new().unwrap();
        let qwen_dir = temp.path().join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-token-stale",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = 0usize;
                let mut auth = String::new();
                let mut auth_type = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    } else if lower.starts_with("x-dashscope-authtype:") {
                        auth_type = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-stale");
                        assert_eq!(auth_type, "qwen-oauth");
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "token expired"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /oauth2/token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=qwen-refresh-old"));
                        (
                            200,
                            "OK",
                            json!({
                                "access_token": "qwen-token-fresh",
                                "refresh_token": "qwen-refresh-new",
                                "expires_in": 7200,
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-fresh");
                        assert_eq!(auth_type, "qwen-oauth");
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Qwen 401 recovery passed."},
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("HERMES_QWEN_BASE_URL", format!("http://{addr}/v1"));
            env::set_var(
                "HERMES_QWEN_OAUTH_TOKEN_URL",
                format!("http://{addr}/oauth2/token"),
            );
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("qwen3.5-plus".to_string()),
                    provider: Some("qwen-oauth".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }
        match previous_qwen_token_url {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_OAUTH_TOKEN_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_OAUTH_TOKEN_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(qwen_dir.join("oauth_creds.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Qwen 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(persisted["access_token"], "qwen-token-fresh");
        assert_eq!(persisted["refresh_token"], "qwen-refresh-new");
    }

    #[test]
    fn qwen_oauth_pool_rotates_after_second_rate_limit() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");

        let temp = TempDir::new().unwrap();
        let qwen_dir = temp.path().join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-token-stale",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "credential_pool": {
                    "qwen-oauth": [
                        {
                            "id": "stale",
                            "priority": 0,
                            "access_token": "qwen-token-stale",
                            "base_url": format!("http://{addr}/v1")
                        },
                        {
                            "id": "fresh",
                            "priority": 1,
                            "access_token": "qwen-token-fresh",
                            "base_url": format!("http://{addr}/v1")
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 | 1 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-stale");
                        (
                            429,
                            "Too Many Requests",
                            json!({"error": {"message": "rate limit exceeded"}}).to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-fresh");
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Qwen pool rate-limit rotation passed."},
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("HERMES_QWEN_BASE_URL", format!("http://{addr}/v1"));
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("qwen3.5-plus".to_string()),
                    provider: Some("qwen-oauth".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            result.final_response,
            "Qwen pool rate-limit rotation passed."
        );
        assert_eq!(result.api_calls, 3);
        assert_eq!(
            persisted["credential_pool"]["qwen-oauth"][0]["last_status"],
            "exhausted"
        );
    }

    #[test]
    fn qwen_oauth_pool_rotates_immediately_on_billing_error() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");

        let temp = TempDir::new().unwrap();
        let qwen_dir = temp.path().join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-token-stale",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "credential_pool": {
                    "qwen-oauth": [
                        {
                            "id": "stale",
                            "priority": 0,
                            "access_token": "qwen-token-stale",
                            "base_url": format!("http://{addr}/v1")
                        },
                        {
                            "id": "fresh",
                            "priority": 1,
                            "access_token": "qwen-token-fresh",
                            "base_url": format!("http://{addr}/v1")
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-stale");
                        (
                            402,
                            "Payment Required",
                            json!({"error": {"message": "insufficient credits"}}).to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-fresh");
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Qwen pool billing rotation passed."},
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("HERMES_QWEN_BASE_URL", format!("http://{addr}/v1"));
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("qwen3.5-plus".to_string()),
                    provider: Some("qwen-oauth".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Qwen pool billing rotation passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["credential_pool"]["qwen-oauth"][0]["last_status"],
            "exhausted"
        );
    }

    #[test]
    fn qwen_oauth_pool_rotates_on_401_before_singleton_refresh() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");
        let previous_qwen_token_url = env::var_os("HERMES_QWEN_OAUTH_TOKEN_URL");

        let temp = TempDir::new().unwrap();
        let qwen_dir = temp.path().join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-token-stale",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "credential_pool": {
                    "qwen-oauth": [
                        {
                            "id": "stale",
                            "priority": 0,
                            "access_token": "qwen-token-stale",
                            "base_url": format!("http://{addr}/v1")
                        },
                        {
                            "id": "fresh",
                            "priority": 1,
                            "access_token": "qwen-token-fresh",
                            "base_url": format!("http://{addr}/v1")
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = 0usize;
                let mut auth = String::new();
                let mut auth_type = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    } else if lower.starts_with("x-dashscope-authtype:") {
                        auth_type = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-stale");
                        assert_eq!(auth_type, "qwen-oauth");
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "token expired"}}).to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-fresh");
                        assert_eq!(auth_type, "qwen-oauth");
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Qwen pool auth rotation passed."},
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("HERMES_QWEN_BASE_URL", format!("http://{addr}/v1"));
            env::set_var(
                "HERMES_QWEN_OAUTH_TOKEN_URL",
                format!("http://{addr}/oauth2/token"),
            );
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("qwen3.5-plus".to_string()),
                    provider: Some("qwen-oauth".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }
        match previous_qwen_token_url {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_OAUTH_TOKEN_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_OAUTH_TOKEN_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Qwen pool auth rotation passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["credential_pool"]["qwen-oauth"][0]["last_status"],
            "exhausted"
        );
        let singleton = serde_json::from_str::<Value>(
            &fs::read_to_string(qwen_dir.join("oauth_creds.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(singleton["access_token"], "qwen-token-stale");
        assert_eq!(singleton["refresh_token"], "qwen-refresh-old");
    }

    #[test]
    fn qwen_oauth_pool_recovers_before_eager_fallback_on_rate_limit() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");

        let temp = TempDir::new().unwrap();
        let qwen_dir = temp.path().join(".qwen");
        fs::create_dir_all(&qwen_dir).unwrap();
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-token-stale",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by fallback too early."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "credential_pool": {
                    "qwen-oauth": [
                        {
                            "id": "stale",
                            "priority": 0,
                            "access_token": "qwen-token-stale",
                            "base_url": format!("http://{addr}/v1")
                        },
                        {
                            "id": "fresh",
                            "priority": 1,
                            "access_token": "qwen-token-fresh",
                            "base_url": format!("http://{addr}/v1")
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 | 1 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-stale");
                        (
                            429,
                            "Too Many Requests",
                            json!({"error": {"message": "rate limit exceeded"}}).to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(auth, "Bearer qwen-token-fresh");
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Qwen pool recovered before fallback."},
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HOME", temp.path());
            env::set_var("HERMES_QWEN_BASE_URL", format!("http://{addr}/v1"));
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("qwen3.5-plus".to_string()),
                    provider: Some("qwen-oauth".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }

        assert_eq!(
            result.final_response,
            "Qwen pool recovered before fallback."
        );
        assert_eq!(result.api_calls, 3);
    }

    #[test]
    fn nous_genuine_rate_limit_trips_shared_guard_for_next_turn() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access",
                        "refresh_token": "nous-refresh",
                        "portal_base_url": "https://portal.nousresearch.com",
                        "inference_base_url": "http://127.0.0.1:1/v1",
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback after genuine Nous 429."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback from shared guard."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            assert!(request_line.starts_with("POST /v1/chat/completions "));
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end();
                if trimmed.is_empty() {
                    break;
                }
                if let Some(value) = trimmed.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut body = vec![0_u8; content_length];
            reader.read_exact(&mut body).unwrap();
            let response_body = json!({"error": {"message": "rate limit exceeded"}}).to_string();
            let response = format!(
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nx-ratelimit-remaining-requests-1h: 0\r\nx-ratelimit-reset-requests-1h: 180\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-nous-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let overrides = ModelOverrides {
            model: Some("Nous-Hermes-2-Mixtral-8x7B-DPO".to_string()),
            provider: Some("nous".to_string()),
            base_url: Some(format!("http://{addr}/v1")),
            api_mode: Some("chat_completions".to_string()),
            ..ModelOverrides::default()
        };

        let first = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();
        server.join().unwrap();

        let state_path = nous_rate_limit_state_path(&context);
        assert!(state_path.exists());
        let first_state =
            serde_json::from_str::<NousRateLimitState>(&fs::read_to_string(&state_path).unwrap())
                .unwrap();
        assert!(first_state.reset_seconds >= 179.0);
        assert_eq!(first.final_response, "Fallback after genuine Nous 429.");
        assert_eq!(first.api_calls, 2);

        let second = context
            .run_chat_completions_turn(
                &loaded,
                "hello again",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();

        assert_eq!(second.final_response, "Fallback from shared guard.");
        assert_eq!(second.api_calls, 1);
    }

    #[test]
    fn nous_short_reset_rate_limit_does_not_trip_shared_guard() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access",
                        "refresh_token": "nous-refresh",
                        "portal_base_url": "https://portal.nousresearch.com",
                        "inference_base_url": "http://127.0.0.1:1/v1",
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback after transient Nous 429."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback after second transient Nous 429."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/chat/completions "));
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some(value) =
                        trimmed.to_ascii_lowercase().strip_prefix("content-length:")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let response_body =
                    json!({"error": {"message": "rate limit exceeded"}}).to_string();
                let response = format!(
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nx-ratelimit-remaining-requests-1h: 0\r\nx-ratelimit-reset-requests-1h: 30\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-nous-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let overrides = ModelOverrides {
            model: Some("Nous-Hermes-2-Mixtral-8x7B-DPO".to_string()),
            provider: Some("nous".to_string()),
            base_url: Some(format!("http://{addr}/v1")),
            api_mode: Some("chat_completions".to_string()),
            ..ModelOverrides::default()
        };

        let first = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();
        assert_eq!(first.final_response, "Fallback after transient Nous 429.");
        assert_eq!(first.api_calls, 2);
        assert!(!nous_rate_limit_state_path(&context).exists());

        let second = context
            .run_chat_completions_turn(
                &loaded,
                "hello again",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();
        server.join().unwrap();

        assert_eq!(
            second.final_response,
            "Fallback after second transient Nous 429."
        );
        assert_eq!(second.api_calls, 2);
        assert!(!nous_rate_limit_state_path(&context).exists());
    }

    #[test]
    fn nous_headerless_429_uses_last_successful_bucket_state() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access",
                        "refresh_token": "nous-refresh",
                        "portal_base_url": "https://portal.nousresearch.com",
                        "inference_base_url": "http://127.0.0.1:1/v1",
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback after headerless Nous 429."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Fallback from headerless guard."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/chat/completions "));
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some(value) =
                        trimmed.to_ascii_lowercase().strip_prefix("content-length:")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, response_body, extra_headers) = match request_index {
                    0 => (
                        200,
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "todo",
                                            "arguments": "{\"action\":\"read\"}"
                                        }
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        })
                        .to_string(),
                        "x-ratelimit-limit-requests-1h: 100\r\nx-ratelimit-remaining-requests-1h: 0\r\nx-ratelimit-reset-requests-1h: 180\r\n".to_string(),
                    ),
                    _ => (
                        429,
                        json!({"error": {"message": "rate limit exceeded"}}).to_string(),
                        String::new(),
                    ),
                };
                let status_text = if status == 200 {
                    "OK"
                } else {
                    "Too Many Requests"
                };
                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    extra_headers,
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-nous-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let overrides = ModelOverrides {
            model: Some("Nous-Hermes-2-Mixtral-8x7B-DPO".to_string()),
            provider: Some("nous".to_string()),
            base_url: Some(format!("http://{addr}/v1")),
            api_mode: Some("chat_completions".to_string()),
            ..ModelOverrides::default()
        };

        let first = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();
        server.join().unwrap();

        assert_eq!(first.final_response, "Fallback after headerless Nous 429.");
        assert_eq!(first.api_calls, 3);
        assert!(nous_rate_limit_state_path(&context).exists());

        let second = context
            .run_chat_completions_turn(
                &loaded,
                "hello again",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                None,
            )
            .unwrap();

        assert_eq!(second.final_response, "Fallback from headerless guard.");
        assert_eq!(second.api_calls, 1);
    }

    #[test]
    fn minimax_oauth_anthropic_turn_retries_after_401_with_refreshed_runtime_token() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "minimax-oauth": {
                        "access_token": "minimax-token-stale",
                        "refresh_token": "minimax-refresh-old",
                        "portal_base_url": format!("http://{addr}"),
                        "inference_base_url": format!("http://{addr}/anthropic"),
                        "client_id": "minimax-client",
                        "expires_at": "2999-01-01T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /anthropic/v1/messages "));
                        assert_eq!(auth, "Bearer minimax-token-stale");
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "token expired"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /oauth/token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=minimax-refresh-old"));
                        (
                            200,
                            "OK",
                            json!({
                                "status": "success",
                                "access_token": "minimax-token-fresh",
                                "refresh_token": "minimax-refresh-new",
                                "expired_in": 3600,
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /anthropic/v1/messages "));
                        assert_eq!(auth, "Bearer minimax-token-fresh");
                        (
                            200,
                            "OK",
                            json!({
                                "id": "msg_minimax_retry",
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "text",
                                    "text": "MiniMax OAuth 401 recovery passed."
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                stream.write_all(http.as_bytes()).unwrap();
            }
        });

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("MiniMax-M2.7-highspeed".to_string()),
                    provider: Some("minimax-oauth".to_string()),
                    api_mode: Some("anthropic_messages".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "MiniMax OAuth 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["providers"]["minimax-oauth"]["access_token"],
            "minimax-token-fresh"
        );
        assert_eq!(
            persisted["providers"]["minimax-oauth"]["refresh_token"],
            "minimax-refresh-new"
        );
    }

    #[test]
    fn chat_completion_turn_retries_transient_http_failure() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let base_url = serve_http_sequence(vec![
            (
                500,
                json!({"error": {"message": "temporary upstream failure"}}).to_string(),
            ),
            (
                200,
                json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": "Recovered after retry."},
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            ),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Recovered after retry.");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_provider_on_rate_limit() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            429,
            json!({"error": {"message": "rate limit exceeded"}}).to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by fallback.");
        assert_eq!(result.model, "fallback-model");
        assert_eq!(result.provider, "custom");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_legacy_fallback_model_on_rate_limit() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            429,
            json!({"error": {"message": "rate limit exceeded"}}).to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by legacy fallback_model."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_model:\n  provider: custom\n  model: legacy-fallback-model\n  base_url: {fallback_url}\n  api_key: fallback-key\n  api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by legacy fallback_model.");
        assert_eq!(result.model, "legacy-fallback-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_billing_error() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            402,
            json!({"error": {"message": "insufficient credits"}}).to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by billing fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-billing-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by billing fallback.");
        assert_eq!(result.model, "fallback-billing-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_transient_usage_limit() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            402,
            json!({
                "error": {
                    "message": "usage limit exceeded for this window, please retry after 30s"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by transient-usage fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-transient-usage\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by transient-usage fallback.");
        assert_eq!(result.model, "fallback-transient-usage");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_bad_request_rate_limit_pattern() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            400,
            json!({
                "error": {
                    "message": "rate limit exceeded for this endpoint, please retry after 10s"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by 400 rate-limit fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-400-rate-limit\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by 400 rate-limit fallback.");
        assert_eq!(result.model, "fallback-400-rate-limit");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_structured_rate_limit_code() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            400,
            json!({
                "error": {
                    "code": "resource_exhausted",
                    "message": "Error"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by structured rate-limit fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-structured-rate-limit\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Served by structured rate-limit fallback."
        );
        assert_eq!(result.model, "fallback-structured-rate-limit");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_structured_invalid_model_code() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            400,
            json!({
                "error": {
                    "type": "invalid_model",
                    "message": "Request failed"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by structured invalid-model fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-structured-invalid-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("missing-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Served by structured invalid-model fallback."
        );
        assert_eq!(result.model, "fallback-structured-invalid-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_generic_bad_request_format_error() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            400,
            json!({
                "error": {
                    "message": "Malformed request body"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by generic 400 fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-generic-400\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by generic 400 fallback.");
        assert_eq!(result.model, "fallback-generic-400");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_unprocessable_entity_format_error() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            422,
            json!({
                "error": {
                    "message": "Validation failed"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by 422 fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-422\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by 422 fallback.");
        assert_eq!(result.model, "fallback-422");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_retries_after_llama_cpp_grammar_schema_strip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert_eq!(request_line.trim_end(), "POST /chat/completions HTTP/1.1");

                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let parameters = &payload["tools"][0]["function"]["parameters"];
                let pattern_property = &parameters["properties"]["pattern"];
                let timestamp_property = &parameters["properties"]["timestamp"];

                if request_index == 0 {
                    assert_eq!(pattern_property["pattern"], json!("\\d+"));
                    assert_eq!(timestamp_property["format"], json!("date-time"));
                    let response_body = json!({
                        "error": {
                            "message": "Unable to generate parser for this template. error parsing grammar"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert!(parameters["properties"].get("pattern").is_some());
                    assert!(pattern_property.get("pattern").is_none());
                    assert!(timestamp_property.get("format").is_none());
                    let response_body = json!({
                        "choices": [{
                            "message": {"role": "assistant", "content": "Recovered after stripping schema keywords."},
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let tools = vec![crate::ToolDefinition {
            name: "schema_sensitive_tool".to_string(),
            toolset: "test".to_string(),
            description: "Tool with strict schema keywords.".to_string(),
            emoji: "x".to_string(),
            schema: json!({
                "name": "schema_sensitive_tool",
                "description": "Tool with strict schema keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "pattern": "\\d+"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    },
                    "required": ["pattern"]
                }
            }),
        }];

        let result = context
            .run_chat_turn_with_tools(
                &loaded,
                Value::String("hello".to_string()),
                &runtime,
                tools,
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(
            result.final_response,
            "Recovered after stripping schema keywords."
        );
        assert_eq!(result.model, "primary-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_retries_after_thinking_signature_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert_eq!(request_line.trim_end(), "POST /chat/completions HTTP/1.1");

                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                let restored_assistant = messages
                    .iter()
                    .find(|message| {
                        message.get("role").and_then(Value::as_str) == Some("assistant")
                            && message
                                .get("content")
                                .and_then(Value::as_str)
                                .is_some_and(|text| text.contains("Prior assistant message"))
                    })
                    .unwrap();

                if request_index == 0 {
                    assert!(restored_assistant.get("reasoning_details").is_some());
                    let response_body = json!({
                        "error": {
                            "message": "Thinking signature invalid for this request"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert!(restored_assistant.get("reasoning_details").is_none());
                    let response_body = json!({
                        "choices": [{
                            "message": {"role": "assistant", "content": "Recovered after stripping thinking details."},
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let session_id = "thinking_signature_seed".to_string();
        session_store
            .create_session(&SessionCreate {
                id: session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("primary-model".to_string()),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": format!("http://{addr}"),
                    "api_mode": "chat_completions",
                })),
                system_prompt: Some("Be helpful.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        session_store
            .append_message(
                &session_id,
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String("Prior assistant message.".to_string())),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: Some(json!({
                        "type": "thinking",
                        "signature": "sig_123",
                        "content": "prior hidden reasoning"
                    })),
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&session_id),
                Some(&session_store),
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(
            result.final_response,
            "Recovered after stripping thinking details."
        );
        assert_eq!(result.model, "primary-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn anthropic_turn_retries_after_disabling_context_beta_header() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/messages "));

                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let all_headers = headers.join("\n").to_ascii_lowercase();

                if request_index == 0 {
                    assert!(all_headers.contains("anthropic-beta: context-1m-2025-08-07"));
                    let response_body = json!({
                        "error": {
                            "message": "The long context beta is not yet available for this subscription."
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert!(!all_headers.contains("anthropic-beta: context-1m-2025-08-07"));
                    assert_eq!(payload["max_tokens"], json!(16384));
                    let response_body = json!({
                        "id": "msg_beta_retry",
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": "Recovered after disabling the beta header."
                        }],
                        "stop_reason": "end_turn"
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let runtime_model = crate::ModelRuntimeConfig {
            model: "anthropic/claude-sonnet-4.6".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{addr}"),
            api_key: "sk-ant-test".to_string(),
            api_mode: "anthropic_messages".to_string(),
            auth_type: "api_key".to_string(),
            default_headers: vec![(
                "anthropic-beta".to_string(),
                "context-1m-2025-08-07".to_string(),
            )],
        };

        let result = context
            .run_chat_turn_with_resolved_runtime(
                &loaded,
                Value::String("Need an anthropic answer.".to_string()),
                &runtime,
                Vec::new(),
                runtime_model,
                None,
                None,
                None,
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(
            result.final_response,
            "Recovered after disabling the beta header."
        );
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.provider, "anthropic");
    }

    #[test]
    fn chat_completion_turn_uses_fallback_on_model_not_found() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            404,
            json!({"error": {"message": "model not found"}}).to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Served by model-not-found fallback."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-model-not-found\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("missing-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Served by model-not-found fallback.");
        assert_eq!(result.model, "fallback-model-not-found");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_retries_generic_not_found_before_failing_over() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![
            (
                404,
                json!({"error": {"message": "route not found"}}).to_string(),
            ),
            (
                200,
                json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": "Recovered after generic 404 retry."},
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            ),
        ]);
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Recovered after generic 404 retry.");
        assert_eq!(result.model, "primary-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_retries_on_structured_context_length_code() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![
            (
                400,
                json!({
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "Error"
                    }
                })
                .to_string(),
            ),
            (
                200,
                json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": "Recovered after structured context retry."},
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            ),
        ]);
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Recovered after structured context retry."
        );
        assert_eq!(result.model, "primary-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_does_not_fallback_on_provider_policy_block() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_http_sequence(vec![(
            404,
            json!({
                "error": {
                    "message": "No endpoints available matching your guardrail restrictions and data policy. Configure: https://openrouter.ai/settings/privacy"
                }
            })
            .to_string(),
        )]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "This should not be used."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: should-not-run\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let error = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("policy-blocked-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap_err();

        let message = format!("{error}");
        assert!(message.contains("guardrail"));
    }

    #[test]
    fn chat_completion_turn_uses_fallback_after_empty_responses() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let primary_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "   "},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Recovered from empty primary responses."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.config_path(),
            format!(
                "agent:\n  api_max_retries: 1\nfallback_providers:\n  - provider: custom\n    model: fallback-empty-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(primary_url),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Recovered from empty primary responses."
        );
        assert_eq!(result.model, "fallback-empty-model");
        assert_eq!(result.api_calls, 3);
    }

    #[test]
    fn chat_completion_turn_prefills_after_inline_thinking_only_response() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let messages = payload["messages"].as_array().unwrap();

                let response = if expected == 0 {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 2);
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "<think>working through the plan</think>"
                            },
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string()
                } else {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 3);
                    assert_eq!(
                        messages[2]["content"].as_str(),
                        Some("<think>working through the plan</think>")
                    );
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "Visible answer after thinking."
                            },
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string()
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("thinking-inline-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        assert_eq!(result.final_response, "Visible answer after thinking.");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_prefills_after_structured_reasoning_only_response() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let messages = payload["messages"].as_array().unwrap();

                let response = if expected == 0 {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 2);
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "hidden structured reasoning"
                            },
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string()
                } else {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 3);
                    assert_eq!(messages[2]["content"].as_str(), Some(""));
                    assert_eq!(
                        messages[2]["reasoning_content"].as_str(),
                        Some("hidden structured reasoning")
                    );
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "Structured reasoning recovery passed."
                            },
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string()
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("thinking-structured-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        assert_eq!(
            result.final_response,
            "Structured reasoning recovery passed."
        );
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn context_budget_compaction_rotates_session_when_threshold_exceeded() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            "model:\n  default: test-model\n  provider: custom\n  api_key: test-key\n  api_mode: chat_completions\n  context_length: 200\ncompression:\n  enabled: true\n  threshold: 0.10\n  protect_last_n: 2\n  target_ratio: 0.15\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let original_session_id = String::from("seed_session");
        session_store
            .create_session(&SessionCreate {
                id: original_session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("test-model".to_string()),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": "http://example.invalid",
                    "api_mode": "chat_completions",
                })),
                system_prompt: Some("Original system prompt.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        for idx in 0..5 {
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String(format!(
                        "History user message {idx} with enough text to inflate the prompt budget."
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String(format!("History assistant message {idx} with enough text to inflate the prompt budget."))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
        }

        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Compaction recovered the turn."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Continue after a very large history block.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&original_session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(result.final_response, "Compaction recovered the turn.");
        let rotated_session_id = result.session_id.unwrap();
        assert_ne!(rotated_session_id, original_session_id);
        let original = session_store
            .get_session(&original_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(original.end_reason.as_deref(), Some("compression"));
        let rotated = session_store
            .get_session(&rotated_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            rotated.parent_session_id.as_deref(),
            Some(original_session_id.as_str())
        );
    }

    #[test]
    fn context_budget_compaction_recovers_from_payload_error() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            "model:\n  default: test-model\n  provider: custom\n  api_key: test-key\n  api_mode: chat_completions\n  context_length: 100000\ncompression:\n  enabled: true\n  threshold: 0.95\n  protect_last_n: 2\n  target_ratio: 0.15\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let original_session_id = String::from("payload_seed_session");
        session_store
            .create_session(&SessionCreate {
                id: original_session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("test-model".to_string()),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": "http://example.invalid",
                    "api_mode": "chat_completions",
                })),
                system_prompt: Some("Original system prompt.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        for idx in 0..5 {
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String(format!(
                        "Payload history user message {idx} with enough text to require reactive compaction."
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String(format!(
                        "Payload history assistant message {idx} with enough text to require reactive compaction."
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
        }

        let base_url = serve_http_sequence(vec![
            (
                413,
                json!({"error": {"message": "prompt exceeds context window limit 200 tokens"}})
                    .to_string(),
            ),
            (
                200,
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Recovered after reactive payload compaction."
                        },
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            ),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Continue after provider payload rejection.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&original_session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Recovered after reactive payload compaction."
        );
        let rotated_session_id = result.session_id.unwrap();
        assert_ne!(rotated_session_id, original_session_id);
        let original = session_store
            .get_session(&original_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(original.end_reason.as_deref(), Some("compression"));
        let rotated = session_store
            .get_session(&rotated_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            rotated.parent_session_id.as_deref(),
            Some(original_session_id.as_str())
        );
    }

    #[test]
    fn context_budget_compaction_recovers_from_generic_short_bad_request() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            "model:\n  default: test-model\n  provider: custom\n  api_key: test-key\n  api_mode: chat_completions\n  context_length: 200000\ncompression:\n  enabled: true\n  threshold: 0.95\n  protect_last_n: 2\n  target_ratio: 0.15\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let original_session_id = String::from("generic_400_seed_session");
        session_store
            .create_session(&SessionCreate {
                id: original_session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("test-model".to_string()),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": "http://example.invalid",
                    "api_mode": "chat_completions",
                })),
                system_prompt: Some("Original system prompt.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        for idx in 0..41 {
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String(format!(
                        "Generic 400 history user message {idx} with enough text to require heuristic compaction."
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String(format!(
                        "Generic 400 history assistant message {idx} with enough text to require heuristic compaction."
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
        }

        let base_url = serve_http_sequence(vec![
            (400, json!({"error": {"message": "Error"}}).to_string()),
            (
                200,
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Recovered after heuristic generic-400 compaction."
                        },
                        "finish_reason": "stop"
                    }]
                })
                .to_string(),
            ),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Continue after a generic provider error.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&original_session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(
            result.final_response,
            "Recovered after heuristic generic-400 compaction."
        );
        let rotated_session_id = result.session_id.unwrap();
        assert_ne!(rotated_session_id, original_session_id);
        let original = session_store
            .get_session(&original_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(original.end_reason.as_deref(), Some("compression"));
        let rotated = session_store
            .get_session(&rotated_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            rotated.parent_session_id.as_deref(),
            Some(original_session_id.as_str())
        );
    }

    #[test]
    fn google_gemini_cli_requests_code_assist_payload_and_persists_project() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HERMES_HOME");
        let previous_base = env::var_os(GOOGLE_CODE_ASSIST_BASE_URL_ENV);

        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join("hermes-home");
        fs::create_dir_all(hermes_home.join("auth")).unwrap();
        fs::write(temp.path().join("SOUL.md"), "You are Hermes Agent.").unwrap();
        fs::write(
            hermes_home.join("auth").join("google_oauth.json"),
            json!({
                "refresh": "google-refresh",
                "access": "google-token",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = 0usize;
                let mut auth = String::new();
                let mut user_agent = String::new();
                let mut goog_client = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    } else if lower.starts_with("user-agent:") {
                        user_agent = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    } else if lower.starts_with("x-goog-api-client:") {
                        goog_client = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                assert_eq!(auth, "Bearer google-token");

                let response_body = match request_index {
                    0 => {
                        assert!(request_line.starts_with("POST /v1internal:loadCodeAssist "));
                        assert!(user_agent.starts_with(
                            "google-api-nodejs-client/9.15.1 (gzip) model/gemini-2.5-pro"
                        ));
                        assert_eq!(goog_client, "gl-node/24.0.0");
                        assert_eq!(payload["metadata"]["pluginType"], json!("GEMINI"));
                        json!({
                            "currentTier": {},
                            "cloudaicompanionProject": ""
                        })
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /v1internal:onboardUser "));
                        assert!(user_agent.starts_with(
                            "google-api-nodejs-client/9.15.1 (gzip) model/gemini-2.5-pro"
                        ));
                        assert_eq!(goog_client, "gl-node/24.0.0");
                        assert_eq!(payload["tierId"], json!("free-tier"));
                        json!({
                            "response": {
                                "cloudaicompanionProject": "managed-proj"
                            }
                        })
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1internal:generateContent "));
                        assert_eq!(user_agent, "hermes-agent (gemini-cli-compat)");
                        assert_eq!(goog_client, "gl-python/hermes");
                        assert_eq!(payload["project"], json!("managed-proj"));
                        assert_eq!(payload["model"], json!("gemini-2.5-pro"));
                        assert_eq!(
                            payload["request"]["systemInstruction"]["parts"][0]["text"],
                            json!("Be helpful")
                        );
                        assert_eq!(payload["request"]["contents"][0]["role"], json!("user"));
                        assert_eq!(
                            payload["request"]["contents"][0]["parts"][0]["text"],
                            json!("hello")
                        );
                        assert_eq!(
                            payload["request"]["tools"][0]["functionDeclarations"][0]["name"],
                            json!("todo")
                        );
                        json!({
                            "response": {
                                "candidates": [{
                                    "content": {
                                        "parts": [{"text": "Gemini CLI smoke passed."}]
                                    },
                                    "finishReason": "STOP"
                                }]
                            }
                        })
                    }
                }
                .to_string();

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HERMES_HOME", &hermes_home);
            env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, format!("http://{addr}"));
        }

        let client = build_http_client().unwrap();
        let runtime = crate::ModelRuntimeConfig {
            model: "gemini-2.5-pro".to_string(),
            provider: "google-gemini-cli".to_string(),
            base_url: "cloudcode-pa://google".to_string(),
            api_key: "google-token".to_string(),
            api_mode: "chat_completions".to_string(),
            auth_type: "oauth_external".to_string(),
            default_headers: Vec::new(),
        };
        let messages = vec![
            json!({"role": "system", "content": "Be helpful"}),
            json!({"role": "user", "content": "hello"}),
        ];
        let tools = vec![crate::ToolDefinition {
            name: "todo".to_string(),
            toolset: "todo".to_string(),
            description: "Manage a todo list.".to_string(),
            emoji: "x".to_string(),
            schema: json!({
                "name": "todo",
                "description": "Manage a todo list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read", "write"]},
                        "count": {"type": "integer", "enum": [1, 2]},
                    }
                }
            }),
        }];

        let result = send_model_request(&client, &runtime, &messages, &tools, None, None).unwrap();
        server.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HERMES_HOME", value) },
            None => unsafe { env::remove_var("HERMES_HOME") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, value) },
            None => unsafe { env::remove_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV) },
        }

        assert_eq!(result.content.as_deref(), Some("Gemini CLI smoke passed."));
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(hermes_home.join("auth").join("google_oauth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            persisted["refresh"],
            json!("google-refresh|managed-proj|managed-proj")
        );
    }

    #[test]
    fn google_gemini_cli_retries_after_401_with_refreshed_runtime_token() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HERMES_HOME");
        let previous_base = env::var_os(GOOGLE_CODE_ASSIST_BASE_URL_ENV);
        let previous_token_url = env::var_os("HERMES_GEMINI_TOKEN_URL");

        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join("hermes-home");
        fs::write(temp.path().join("SOUL.md"), "You are Hermes Agent.").unwrap();
        fs::create_dir_all(hermes_home.join("auth")).unwrap();
        fs::write(
            hermes_home.join("auth").join("google_oauth.json"),
            json!({
                "refresh": "google-refresh-old",
                "access": "google-token-stale",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(hermes_home.clone()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let (status, status_text, response_body) = match request_index {
                    0 => {
                        let payload = serde_json::from_slice::<Value>(&body).unwrap();
                        assert!(request_line.starts_with("POST /v1internal:loadCodeAssist "));
                        assert_eq!(auth, "Bearer google-token-stale");
                        assert_eq!(payload["metadata"]["pluginType"], json!("GEMINI"));
                        (
                            200,
                            "OK",
                            json!({
                                "currentTier": {"id": "free-tier"},
                                "cloudaicompanionProject": "managed-proj"
                            })
                            .to_string(),
                        )
                    }
                    1 => {
                        let payload = serde_json::from_slice::<Value>(&body).unwrap();
                        assert!(request_line.starts_with("POST /v1internal:generateContent "));
                        assert_eq!(auth, "Bearer google-token-stale");
                        assert_eq!(payload["model"], json!("gemini-2.5-pro"));
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "token expired"}}).to_string(),
                        )
                    }
                    2 => {
                        assert!(request_line.starts_with("POST /token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=google-refresh-old"));
                        (
                            200,
                            "OK",
                            json!({
                                "access_token": "google-token-fresh",
                                "refresh_token": "google-refresh-new",
                                "expires_in": 7200,
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        let payload = serde_json::from_slice::<Value>(&body).unwrap();
                        assert!(request_line.starts_with("POST /v1internal:generateContent "));
                        assert_eq!(auth, "Bearer google-token-fresh");
                        assert_eq!(payload["project"], json!("managed-proj"));
                        (
                            200,
                            "OK",
                            json!({
                                "response": {
                                    "candidates": [{
                                        "content": {
                                            "parts": [{"text": "Google Gemini 401 recovery passed."}]
                                        },
                                        "finishReason": "STOP"
                                    }]
                                }
                            })
                            .to_string(),
                        )
                    }
                };

                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HERMES_HOME", &hermes_home);
            env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, format!("http://{addr}"));
            env::set_var("HERMES_GEMINI_TOKEN_URL", format!("http://{addr}/token"));
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gemini-2.5-pro".to_string()),
                    provider: Some("google-gemini-cli".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        server.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HERMES_HOME", value) },
            None => unsafe { env::remove_var("HERMES_HOME") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, value) },
            None => unsafe { env::remove_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV) },
        }
        match previous_token_url {
            Some(value) => unsafe { env::set_var("HERMES_GEMINI_TOKEN_URL", value) },
            None => unsafe { env::remove_var("HERMES_GEMINI_TOKEN_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(hermes_home.join("auth").join("google_oauth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Google Gemini 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(persisted["access"], "google-token-fresh");
        assert_eq!(
            persisted["refresh"],
            json!("google-refresh-new|managed-proj|managed-proj")
        );
    }

    #[test]
    fn google_gemini_cli_pool_falls_back_immediately_on_rate_limit() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HERMES_HOME");
        let previous_base = env::var_os(GOOGLE_CODE_ASSIST_BASE_URL_ENV);

        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join("hermes-home");
        fs::write(temp.path().join("SOUL.md"), "You are Hermes Agent.").unwrap();
        fs::create_dir_all(hermes_home.join("auth")).unwrap();
        fs::write(
            hermes_home.join("auth").join("google_oauth.json"),
            json!({
                "refresh": "google-refresh-stale",
                "access": "google-token-stale",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(hermes_home.clone()));
        context.ensure_hermes_home().unwrap();
        let fallback_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "Gemini fallback served."},
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "credential_pool": {
                    "google-gemini-cli": [
                        {
                            "id": "stale",
                            "priority": 0,
                            "access_token": "google-token-stale"
                        },
                        {
                            "id": "fresh",
                            "priority": 1,
                            "access_token": "google-token-fresh"
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            context.config_path(),
            format!(
                "fallback_providers:\n  - provider: custom\n    model: fallback-google-model\n    base_url: {fallback_url}\n    api_key: fallback-key\n    api_mode: chat_completions\n"
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = 0usize;
                let mut auth = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    } else if lower.starts_with("authorization:") {
                        auth = trimmed
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                            .unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let (status, status_text, response_body) = match request_index {
                    0 => {
                        assert!(request_line.starts_with("POST /v1internal:loadCodeAssist "));
                        assert_eq!(auth, "Bearer google-token-stale");
                        assert_eq!(payload["metadata"]["pluginType"], json!("GEMINI"));
                        (
                            200,
                            "OK",
                            json!({
                                "currentTier": {"id": "free-tier"},
                                "cloudaicompanionProject": "managed-proj"
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1internal:generateContent "));
                        assert_eq!(auth, "Bearer google-token-stale");
                        assert_eq!(payload["project"], json!("managed-proj"));
                        (
                            429,
                            "Too Many Requests",
                            json!({"error": {"message": "rate limit exceeded"}}).to_string(),
                        )
                    }
                };

                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });

        unsafe {
            env::set_var("HERMES_HOME", &hermes_home);
            env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, format!("http://{addr}"));
        }

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gemini-2.5-pro".to_string()),
                    provider: Some("google-gemini-cli".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        server.join().unwrap();

        match previous_home {
            Some(value) => unsafe { env::set_var("HERMES_HOME", value) },
            None => unsafe { env::remove_var("HERMES_HOME") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV, value) },
            None => unsafe { env::remove_var(GOOGLE_CODE_ASSIST_BASE_URL_ENV) },
        }

        assert_eq!(result.final_response, "Gemini fallback served.");
        assert_eq!(result.model, "fallback-google-model");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn copilot_acp_bridge_returns_text_response() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_command = env::var_os("HERMES_COPILOT_ACP_COMMAND");
        let previous_base = env::var_os("COPILOT_ACP_BASE_URL");

        let temp = TempDir::new().unwrap();
        let script = temp.path().join("fake-copilot-acp.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json
import sys

for raw in sys.stdin:
    message = json.loads(raw)
    method = message.get("method")
    msg_id = message.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}}), flush=True)
    elif method == "session/new":
        print(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {"sessionId": "copilot-test"}}), flush=True)
    elif method == "session/prompt":
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "Copilot ACP smoke passed."}
                }
            }
        }), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}}), flush=True)
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&script).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script, permissions).unwrap();
        }

        unsafe {
            env::set_var("HERMES_COPILOT_ACP_COMMAND", &script);
            env::set_var("COPILOT_ACP_BASE_URL", "acp://copilot");
        }

        let runtime = crate::ModelRuntimeConfig {
            model: "claude-sonnet-4.6".to_string(),
            provider: "copilot-acp".to_string(),
            base_url: "acp://copilot".to_string(),
            api_key: "copilot-acp".to_string(),
            api_mode: "chat_completions".to_string(),
            auth_type: "external_process".to_string(),
            default_headers: Vec::new(),
        };
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let client = build_http_client().unwrap();
        let result = request_model_text(&client, &runtime, &messages).unwrap();

        match previous_command {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_ACP_COMMAND", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_ACP_COMMAND") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var("COPILOT_ACP_BASE_URL", value) },
            None => unsafe { env::remove_var("COPILOT_ACP_BASE_URL") },
        }

        assert_eq!(result.as_deref(), Some("Copilot ACP smoke passed."));
    }

    #[test]
    fn extract_copilot_acp_tool_calls_removes_xml_blocks() {
        let input = concat!(
            "Planning...\n",
            "<tool_call>{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"todo\",\"arguments\":\"{\\\"action\\\":\\\"read\\\"}\"}}</tool_call>\n",
            "Done."
        );

        let (tool_calls, content) = extract_copilot_acp_tool_calls(input).unwrap();

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].name, "todo");
        assert_eq!(tool_calls[0].arguments_raw, "{\"action\":\"read\"}");
        assert_eq!(tool_calls[0].json, json!({"action": "read"}));
        assert_eq!(content, "Planning...\nDone.");
    }

    #[test]
    fn chat_completion_turn_executes_tools_and_returns_final_text() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{\"path\":\"notes.txt\",\"content\":\"hello from tool\"}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Finished writing the file."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file named notes.txt",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Finished writing the file.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("notes.txt")).unwrap(),
            "hello from tool"
        );
    }

    #[test]
    fn chat_completion_turn_nudges_after_empty_response_following_tool_results() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let messages = payload["messages"].as_array().unwrap();

                let response = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(messages.len(), 2);
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": "{\"path\":\"notes.txt\",\"content\":\"hello from tool\"}"
                                        }
                                    }]
                                }
                            }]
                        })
                        .to_string()
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(messages.len(), 4);
                        assert_eq!(messages[3]["role"].as_str(), Some("tool"));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "   "
                                },
                                "finish_reason": "stop"
                            }]
                        })
                        .to_string()
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        assert_eq!(messages.len(), 6);
                        assert_eq!(messages[4]["role"].as_str(), Some("assistant"));
                        assert_eq!(messages[4]["content"].as_str(), Some("(empty)"));
                        assert_eq!(messages[5]["role"].as_str(), Some("user"));
                        assert_eq!(messages[5]["content"].as_str(), Some(POST_TOOL_EMPTY_NUDGE));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "Finished after the tool-result nudge."
                                }
                            }]
                        })
                        .to_string()
                    }
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file named notes.txt",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        assert_eq!(
            result.final_response,
            "Finished after the tool-result nudge."
        );
        assert_eq!(result.api_calls, 3);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("notes.txt")).unwrap(),
            "hello from tool"
        );
    }

    #[test]
    fn deepseek_tool_replay_injects_reasoning_content_placeholder() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let messages = payload["messages"].as_array().unwrap();

                let response = if expected == 0 {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 2);
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "id": "call_deepseek_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": "{\"path\":\"deepseek.txt\",\"content\":\"hello from deepseek tool\"}"
                                    }
                                }],
                                "reasoning": "private reasoning"
                            },
                            "finish_reason": "tool_calls"
                        }]
                    })
                    .to_string()
                } else {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    assert_eq!(messages.len(), 4);
                    assert_eq!(messages[2]["role"].as_str(), Some("assistant"));
                    assert!(messages[2]["tool_calls"].is_array());
                    assert_eq!(messages[2]["reasoning"].as_str(), Some("private reasoning"));
                    assert_eq!(messages[2]["reasoning_content"].as_str(), Some(" "));
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "DeepSeek replay accepted."
                            }
                        }]
                    })
                    .to_string()
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file named deepseek.txt",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("deepseek-v4-pro".to_string()),
                    provider: Some("deepseek".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        assert_eq!(result.final_response, "DeepSeek replay accepted.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("deepseek.txt")).unwrap(),
            "hello from deepseek tool"
        );
    }

    #[test]
    fn copilot_chat_completion_uses_exchanged_token_and_copilot_headers() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_copilot = env::var_os("COPILOT_GITHUB_TOKEN");
        let previous_gh = env::var_os("GH_TOKEN");
        let previous_github = env::var_os("GITHUB_TOKEN");
        let previous_exchange = env::var_os("HERMES_COPILOT_TOKEN_EXCHANGE_URL");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let response = if expected == 0 {
                    assert!(request_line.starts_with("GET /copilot-token "));
                    let all_headers = headers.join("\n").to_ascii_lowercase();
                    assert!(all_headers.contains("authorization: token gho_runtime_agent"));
                    json!({
                        "token": "copilot-api-token",
                        "expires_at": 4102444800_u64
                    })
                    .to_string()
                } else {
                    assert!(request_line.starts_with("POST /v1/chat/completions "));
                    let all_headers = headers.join("\n").to_ascii_lowercase();
                    assert!(all_headers.contains("authorization: bearer copilot-api-token"));
                    assert!(all_headers.contains("editor-version: vscode/1.104.1"));
                    assert!(all_headers.contains("copilot-integration-id: vscode-chat"));
                    assert!(all_headers.contains("openai-intent: conversation-edits"));
                    assert!(all_headers.contains("x-initiator: agent"));
                    json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "Copilot Rust smoke passed."
                            }
                        }]
                    })
                    .to_string()
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::set_var("COPILOT_GITHUB_TOKEN", "gho_runtime_agent");
            env::remove_var("GH_TOKEN");
            env::remove_var("GITHUB_TOKEN");
            env::set_var(
                "HERMES_COPILOT_TOKEN_EXCHANGE_URL",
                format!("http://{addr}/copilot-token"),
            );
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Hello from copilot",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-4.1".to_string()),
                    provider: Some("copilot".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_copilot {
            Some(value) => unsafe { env::set_var("COPILOT_GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("COPILOT_GITHUB_TOKEN") },
        }
        match previous_gh {
            Some(value) => unsafe { env::set_var("GH_TOKEN", value) },
            None => unsafe { env::remove_var("GH_TOKEN") },
        }
        match previous_github {
            Some(value) => unsafe { env::set_var("GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("GITHUB_TOKEN") },
        }
        match previous_exchange {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL") },
        }

        assert_eq!(result.final_response, "Copilot Rust smoke passed.");
    }

    #[test]
    fn copilot_chat_completion_retries_after_401_with_refreshed_runtime_token() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_copilot = env::var_os("COPILOT_GITHUB_TOKEN");
        let previous_gh = env::var_os("GH_TOKEN");
        let previous_github = env::var_os("GITHUB_TOKEN");
        let previous_exchange = env::var_os("HERMES_COPILOT_TOKEN_EXCHANGE_URL");

        let _ = crate::clear_provider_runtime_cache("copilot");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let (status, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("GET /copilot-token "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: token gho_runtime_refresh"));
                        (
                            200,
                            json!({
                                "token": "copilot-api-token-1",
                                "expires_at": 4102444800_u64
                            })
                            .to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer copilot-api-token-1"));
                        (
                            401,
                            json!({"error": {"message": "unauthorized"}}).to_string(),
                        )
                    }
                    2 => {
                        assert!(request_line.starts_with("GET /copilot-token "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: token gho_runtime_refresh"));
                        (
                            200,
                            json!({
                                "token": "copilot-api-token-2",
                                "expires_at": 4102444800_u64
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer copilot-api-token-2"));
                        (
                            200,
                            json!({
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "Copilot 401 recovery passed."
                                    }
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let status_text = if status == 200 { "OK" } else { "Unauthorized" };
                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::set_var("COPILOT_GITHUB_TOKEN", "gho_runtime_refresh");
            env::remove_var("GH_TOKEN");
            env::remove_var("GITHUB_TOKEN");
            env::set_var(
                "HERMES_COPILOT_TOKEN_EXCHANGE_URL",
                format!("http://{addr}/copilot-token"),
            );
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Hello from copilot",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-4.1".to_string()),
                    provider: Some("copilot".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_copilot {
            Some(value) => unsafe { env::set_var("COPILOT_GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("COPILOT_GITHUB_TOKEN") },
        }
        match previous_gh {
            Some(value) => unsafe { env::set_var("GH_TOKEN", value) },
            None => unsafe { env::remove_var("GH_TOKEN") },
        }
        match previous_github {
            Some(value) => unsafe { env::set_var("GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("GITHUB_TOKEN") },
        }
        match previous_exchange {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL") },
        }
        let _ = crate::clear_provider_runtime_cache("copilot");

        assert_eq!(result.final_response, "Copilot 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn nous_chat_completion_retries_after_401_with_refreshed_agent_key() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_nous_api_key = env::var_os("NOUS_API_KEY");
        let previous_inference_base = env::var_os("NOUS_INFERENCE_BASE_URL");
        let previous_portal_base = env::var_os("NOUS_PORTAL_BASE_URL");
        let previous_hermes_portal_base = env::var_os("HERMES_PORTAL_BASE_URL");
        let previous_min_ttl = env::var_os("HERMES_NOUS_MIN_KEY_TTL_SECONDS");
        let previous_timeout = env::var_os("HERMES_NOUS_TIMEOUT_SECONDS");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access-stable",
                        "refresh_token": "nous-refresh-stable",
                        "portal_base_url": format!("http://{addr}"),
                        "inference_base_url": format!("http://{addr}/v1"),
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key-stale",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer nous-agent-key-stale"));
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "agent key expired"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /api/oauth/agent-key "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer nous-access-stable"));
                        assert!(
                            String::from_utf8_lossy(&body).contains("\"min_ttl_seconds\":1800")
                        );
                        (
                            200,
                            "OK",
                            json!({
                                "api_key": "nous-agent-key-fresh",
                                "key_id": "nous-key-2",
                                "expires_at": "2999-01-03T00:00:00Z",
                                "expires_in": 7200,
                                "inference_base_url": format!("http://{addr}/minted/v1"),
                                "reused": false
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /minted/v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer nous-agent-key-fresh"));
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "Nous 401 recovery passed."
                                    }
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::remove_var("NOUS_API_KEY");
            env::remove_var("NOUS_INFERENCE_BASE_URL");
            env::remove_var("NOUS_PORTAL_BASE_URL");
            env::remove_var("HERMES_PORTAL_BASE_URL");
            env::set_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800");
            env::set_var("HERMES_NOUS_TIMEOUT_SECONDS", "15");
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Hello from Nous",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("Nous-Hermes-2-Mixtral-8x7B-DPO".to_string()),
                    provider: Some("nous".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_nous_api_key {
            Some(value) => unsafe { env::set_var("NOUS_API_KEY", value) },
            None => unsafe { env::remove_var("NOUS_API_KEY") },
        }
        match previous_inference_base {
            Some(value) => unsafe { env::set_var("NOUS_INFERENCE_BASE_URL", value) },
            None => unsafe { env::remove_var("NOUS_INFERENCE_BASE_URL") },
        }
        match previous_portal_base {
            Some(value) => unsafe { env::set_var("NOUS_PORTAL_BASE_URL", value) },
            None => unsafe { env::remove_var("NOUS_PORTAL_BASE_URL") },
        }
        match previous_hermes_portal_base {
            Some(value) => unsafe { env::set_var("HERMES_PORTAL_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_PORTAL_BASE_URL") },
        }
        match previous_min_ttl {
            Some(value) => unsafe { env::set_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS", value) },
            None => unsafe { env::remove_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS") },
        }
        match previous_timeout {
            Some(value) => unsafe { env::set_var("HERMES_NOUS_TIMEOUT_SECONDS", value) },
            None => unsafe { env::remove_var("HERMES_NOUS_TIMEOUT_SECONDS") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Nous 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["providers"]["nous"]["agent_key"],
            "nous-agent-key-fresh"
        );
        assert_eq!(
            persisted["providers"]["nous"]["inference_base_url"],
            format!("http://{addr}/minted/v1")
        );
    }

    #[test]
    fn nous_credential_pool_refreshes_rotated_entry_after_401() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_nous_api_key = env::var_os("NOUS_API_KEY");
        let previous_inference_base = env::var_os("NOUS_INFERENCE_BASE_URL");
        let previous_portal_base = env::var_os("NOUS_PORTAL_BASE_URL");
        let previous_hermes_portal_base = env::var_os("HERMES_PORTAL_BASE_URL");
        let previous_min_ttl = env::var_os("HERMES_NOUS_MIN_KEY_TTL_SECONDS");
        let previous_timeout = env::var_os("HERMES_NOUS_TIMEOUT_SECONDS");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access-a",
                        "refresh_token": "nous-refresh-a",
                        "portal_base_url": format!("http://{addr}"),
                        "inference_base_url": format!("http://{addr}/a/v1"),
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key-a",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                },
                "credential_pool": {
                    "nous": [
                        {
                            "id": "entry-a",
                            "priority": 0,
                            "access_token": "nous-access-a",
                            "refresh_token": "nous-refresh-a",
                            "portal_base_url": format!("http://{addr}"),
                            "inference_base_url": format!("http://{addr}/a/v1"),
                            "client_id": "hermes-cli",
                            "agent_key": "nous-agent-key-a",
                            "agent_key_expires_at": "2999-01-02T00:00:00Z"
                        },
                        {
                            "id": "entry-b",
                            "priority": 1,
                            "access_token": "nous-access-b",
                            "refresh_token": "nous-refresh-b",
                            "portal_base_url": format!("http://{addr}"),
                            "inference_base_url": format!("http://{addr}/b/v1"),
                            "client_id": "hermes-cli",
                            "agent_key": "nous-agent-key-b",
                            "agent_key_expires_at": "2999-01-02T00:00:00Z"
                        }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();

        let join = thread::spawn(move || {
            for expected in 0..5 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /a/v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer nous-agent-key-a"));
                        (
                            402,
                            "Payment Required",
                            json!({"error": {"message": "insufficient credits"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /b/v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer nous-agent-key-b"));
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "agent key expired"}}).to_string(),
                        )
                    }
                    2 => {
                        assert!(request_line.starts_with("POST /oauth/token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=nous-refresh-b"));
                        (
                            200,
                            "OK",
                            json!({
                                "access_token": "nous-access-b-refreshed",
                                "refresh_token": "nous-refresh-b-new",
                                "token_type": "Bearer",
                                "scope": "inference:mint_agent_key",
                                "inference_base_url": format!("http://{addr}/b-refreshed/v1"),
                                "expires_in": 3600
                            })
                            .to_string(),
                        )
                    }
                    3 => {
                        assert!(request_line.starts_with("POST /api/oauth/agent-key "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(
                            all_headers.contains("authorization: bearer nous-access-b-refreshed")
                        );
                        assert!(
                            String::from_utf8_lossy(&body).contains("\"min_ttl_seconds\":1800")
                        );
                        (
                            200,
                            "OK",
                            json!({
                                "api_key": "nous-agent-key-b-fresh",
                                "key_id": "nous-key-b",
                                "expires_at": "2999-01-03T00:00:00Z",
                                "expires_in": 7200,
                                "inference_base_url": format!("http://{addr}/b-minted/v1"),
                                "reused": false
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /b-minted/v1/chat/completions "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(
                            all_headers.contains("authorization: bearer nous-agent-key-b-fresh")
                        );
                        (
                            200,
                            "OK",
                            json!({
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "Nous pool auth recovery passed."
                                    }
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::remove_var("NOUS_API_KEY");
            env::remove_var("NOUS_INFERENCE_BASE_URL");
            env::remove_var("NOUS_PORTAL_BASE_URL");
            env::remove_var("HERMES_PORTAL_BASE_URL");
            env::set_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800");
            env::set_var("HERMES_NOUS_TIMEOUT_SECONDS", "15");
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Hello from Nous pool",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("Nous-Hermes-2-Mixtral-8x7B-DPO".to_string()),
                    provider: Some("nous".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_nous_api_key {
            Some(value) => unsafe { env::set_var("NOUS_API_KEY", value) },
            None => unsafe { env::remove_var("NOUS_API_KEY") },
        }
        match previous_inference_base {
            Some(value) => unsafe { env::set_var("NOUS_INFERENCE_BASE_URL", value) },
            None => unsafe { env::remove_var("NOUS_INFERENCE_BASE_URL") },
        }
        match previous_portal_base {
            Some(value) => unsafe { env::set_var("NOUS_PORTAL_BASE_URL", value) },
            None => unsafe { env::remove_var("NOUS_PORTAL_BASE_URL") },
        }
        match previous_hermes_portal_base {
            Some(value) => unsafe { env::set_var("HERMES_PORTAL_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_PORTAL_BASE_URL") },
        }
        match previous_min_ttl {
            Some(value) => unsafe { env::set_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS", value) },
            None => unsafe { env::remove_var("HERMES_NOUS_MIN_KEY_TTL_SECONDS") },
        }
        match previous_timeout {
            Some(value) => unsafe { env::set_var("HERMES_NOUS_TIMEOUT_SECONDS", value) },
            None => unsafe { env::remove_var("HERMES_NOUS_TIMEOUT_SECONDS") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Nous pool auth recovery passed.");
        assert_eq!(result.api_calls, 3);
        assert_eq!(
            persisted["credential_pool"]["nous"][0]["last_status"],
            "exhausted"
        );
        assert_eq!(
            persisted["credential_pool"]["nous"][1]["access_token"],
            "nous-access-b-refreshed"
        );
        assert_eq!(
            persisted["credential_pool"]["nous"][1]["refresh_token"],
            "nous-refresh-b-new"
        );
        assert_eq!(
            persisted["credential_pool"]["nous"][1]["agent_key"],
            "nous-agent-key-b-fresh"
        );
        assert_eq!(
            persisted["credential_pool"]["nous"][1]["inference_base_url"],
            format!("http://{addr}/b-minted/v1")
        );
    }

    #[test]
    fn codex_turn_retries_after_401_with_refreshed_runtime_token() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_refresh_url = env::var_os("HERMES_CODEX_OAUTH_TOKEN_URL");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "codex-token-stale",
                            "refresh_token": "codex-refresh-old"
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/responses "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer codex-token-stale"));
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "token rejected"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /oauth/token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=codex-refresh-old"));
                        (
                            200,
                            "OK",
                            json!({
                                "access_token": "codex-token-fresh",
                                "refresh_token": "codex-refresh-new"
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/responses "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(all_headers.contains("authorization: bearer codex-token-fresh"));
                        (
                            200,
                            "OK",
                            json!({
                                "id": "resp_codex_retry",
                                "status": "completed",
                                "output": [{
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{
                                        "type": "output_text",
                                        "text": "Codex 401 recovery passed."
                                    }]
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::set_var(
                "HERMES_CODEX_OAUTH_TOKEN_URL",
                format!("http://{addr}/oauth/token"),
            );
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Hello from codex",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-5.4".to_string()),
                    provider: Some("openai-codex".to_string()),
                    base_url: Some(format!("http://{addr}/v1")),
                    api_mode: Some("codex_responses".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_refresh_url {
            Some(value) => unsafe { env::set_var("HERMES_CODEX_OAUTH_TOKEN_URL", value) },
            None => unsafe { env::remove_var("HERMES_CODEX_OAUTH_TOKEN_URL") },
        }

        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(context.hermes_home().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(result.final_response, "Codex 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["providers"]["openai-codex"]["tokens"]["access_token"],
            "codex-token-fresh"
        );
        assert_eq!(
            persisted["providers"]["openai-codex"]["tokens"]["refresh_token"],
            "codex-refresh-new"
        );
    }

    #[test]
    fn anthropic_turn_retries_after_401_with_refreshed_runtime_token() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_path = env::var_os("HERMES_ANTHROPIC_CREDENTIALS_PATH");
        let previous_refresh_url = env::var_os("HERMES_ANTHROPIC_OAUTH_TOKEN_URL");
        let previous_token = env::var_os("ANTHROPIC_TOKEN");
        let previous_cc = env::var_os("CLAUDE_CODE_OAUTH_TOKEN");
        let previous_api_key = env::var_os("ANTHROPIC_API_KEY");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let credentials_path = context.hermes_home().join("anthropic_credentials.json");
        fs::write(
            &credentials_path,
            json!({
                "claudeAiOauth": {
                    "accessToken": "cc-anthropic-oauth-stale",
                    "refreshToken": "anthropic-refresh-old",
                    "expiresAt": i64::MAX / 2,
                    "scopes": ["user:inference", "user:profile"]
                }
            })
            .to_string(),
        )
        .unwrap();

        let join = thread::spawn(move || {
            for expected in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("Content-Length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let (status, status_text, response) = match expected {
                    0 => {
                        assert!(request_line.starts_with("POST /v1/messages "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(
                            all_headers.contains("authorization: bearer cc-anthropic-oauth-stale")
                        );
                        assert!(all_headers.contains("anthropic-version: 2023-06-01"));
                        assert!(all_headers.contains("x-app: cli"));
                        assert!(all_headers.contains("oauth-2025-04-20"));
                        (
                            401,
                            "Unauthorized",
                            json!({"error": {"message": "unauthorized"}}).to_string(),
                        )
                    }
                    1 => {
                        assert!(request_line.starts_with("POST /oauth/token "));
                        let body_text = String::from_utf8_lossy(&body);
                        assert!(body_text.contains("grant_type=refresh_token"));
                        assert!(body_text.contains("refresh_token=anthropic-refresh-old"));
                        assert!(
                            body_text.contains("client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e")
                        );
                        (
                            200,
                            "OK",
                            json!({
                                "access_token": "cc-anthropic-oauth-fresh",
                                "refresh_token": "anthropic-refresh-new",
                                "expires_in": 7200,
                            })
                            .to_string(),
                        )
                    }
                    _ => {
                        assert!(request_line.starts_with("POST /v1/messages "));
                        let all_headers = headers.join("\n").to_ascii_lowercase();
                        assert!(
                            all_headers.contains("authorization: bearer cc-anthropic-oauth-fresh")
                        );
                        assert!(all_headers.contains("x-app: cli"));
                        (
                            200,
                            "OK",
                            json!({
                                "id": "msg_auth_refresh",
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "text",
                                    "text": "Anthropic 401 recovery passed."
                                }]
                            })
                            .to_string(),
                        )
                    }
                };

                let http = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text,
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::set_var("HERMES_ANTHROPIC_CREDENTIALS_PATH", &credentials_path);
            env::set_var(
                "HERMES_ANTHROPIC_OAUTH_TOKEN_URL",
                format!("http://{addr}/oauth/token"),
            );
            env::remove_var("ANTHROPIC_TOKEN");
            env::remove_var("CLAUDE_CODE_OAUTH_TOKEN");
            env::remove_var("ANTHROPIC_API_KEY");
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Need an anthropic auth refresh.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic/claude-sonnet-4.6".to_string()),
                    provider: Some("anthropic".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_mode: Some("anthropic_messages".to_string()),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_path {
            Some(value) => unsafe { env::set_var("HERMES_ANTHROPIC_CREDENTIALS_PATH", value) },
            None => unsafe { env::remove_var("HERMES_ANTHROPIC_CREDENTIALS_PATH") },
        }
        match previous_refresh_url {
            Some(value) => unsafe { env::set_var("HERMES_ANTHROPIC_OAUTH_TOKEN_URL", value) },
            None => unsafe { env::remove_var("HERMES_ANTHROPIC_OAUTH_TOKEN_URL") },
        }
        match previous_token {
            Some(value) => unsafe { env::set_var("ANTHROPIC_TOKEN", value) },
            None => unsafe { env::remove_var("ANTHROPIC_TOKEN") },
        }
        match previous_cc {
            Some(value) => unsafe { env::set_var("CLAUDE_CODE_OAUTH_TOKEN", value) },
            None => unsafe { env::remove_var("CLAUDE_CODE_OAUTH_TOKEN") },
        }
        match previous_api_key {
            Some(value) => unsafe { env::set_var("ANTHROPIC_API_KEY", value) },
            None => unsafe { env::remove_var("ANTHROPIC_API_KEY") },
        }

        let persisted =
            serde_json::from_str::<Value>(&fs::read_to_string(&credentials_path).unwrap()).unwrap();
        assert_eq!(result.final_response, "Anthropic 401 recovery passed.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(
            persisted["claudeAiOauth"]["accessToken"],
            "cc-anthropic-oauth-fresh"
        );
        assert_eq!(
            persisted["claudeAiOauth"]["refreshToken"],
            "anthropic-refresh-new"
        );
    }

    #[test]
    fn copilot_claude_turn_uses_messages_api_and_copilot_headers() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_copilot = env::var_os("COPILOT_GITHUB_TOKEN");
        let previous_gh = env::var_os("GH_TOKEN");
        let previous_github = env::var_os("GITHUB_TOKEN");
        let previous_exchange = env::var_os("HERMES_COPILOT_TOKEN_EXCHANGE_URL");

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            for expected in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut content_length = 0usize;
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                    headers.push(trimmed);
                }
                let mut body = vec![0_u8; content_length];
                let _ = reader.read_exact(&mut body);

                let response = if expected == 0 {
                    assert!(request_line.starts_with("GET /copilot-token "));
                    let all_headers = headers.join("\n").to_ascii_lowercase();
                    assert!(all_headers.contains("authorization: token gho_runtime_claude"));
                    json!({
                        "token": "copilot-api-claude-token",
                        "expires_at": 4102444800_u64
                    })
                    .to_string()
                } else {
                    assert!(request_line.starts_with("POST /v1/messages "));
                    let all_headers = headers.join("\n").to_ascii_lowercase();
                    assert!(all_headers.contains("x-api-key: copilot-api-claude-token"));
                    assert!(all_headers.contains("anthropic-version: 2023-06-01"));
                    assert!(all_headers.contains("editor-version: vscode/1.104.1"));
                    assert!(all_headers.contains("copilot-integration-id: vscode-chat"));
                    assert!(all_headers.contains("openai-intent: conversation-edits"));
                    assert!(all_headers.contains("x-initiator: agent"));
                    json!({
                        "id": "msg_copilot_claude",
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": "Copilot Claude flow complete."
                        }],
                        "stop_reason": "end_turn"
                    })
                    .to_string()
                };

                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        unsafe {
            env::set_var("COPILOT_GITHUB_TOKEN", "gho_runtime_claude");
            env::remove_var("GH_TOKEN");
            env::remove_var("GITHUB_TOKEN");
            env::set_var(
                "HERMES_COPILOT_TOKEN_EXCHANGE_URL",
                format!("http://{addr}/copilot-token"),
            );
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Route through Copilot Claude",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic/claude-sonnet-4.6".to_string()),
                    provider: Some("copilot".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    ..ModelOverrides::default()
                },
                None,
                None,
            )
            .unwrap();
        join.join().unwrap();

        match previous_copilot {
            Some(value) => unsafe { env::set_var("COPILOT_GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("COPILOT_GITHUB_TOKEN") },
        }
        match previous_gh {
            Some(value) => unsafe { env::set_var("GH_TOKEN", value) },
            None => unsafe { env::remove_var("GH_TOKEN") },
        }
        match previous_github {
            Some(value) => unsafe { env::set_var("GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("GITHUB_TOKEN") },
        }
        match previous_exchange {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL") },
        }

        assert_eq!(result.provider, "copilot");
        assert_eq!(result.model, "claude-sonnet-4.6");
        assert_eq!(result.final_response, "Copilot Claude flow complete.");
    }

    #[test]
    fn chat_turn_with_user_content_sends_multimodal_payload_and_persists_it() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            assert_eq!(request_line.trim_end(), "POST /chat/completions HTTP/1.1");
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut body = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut body);
            let payload: Value = serde_json::from_slice(&body).unwrap();
            let messages = payload["messages"].as_array().unwrap();
            let user = messages
                .iter()
                .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
                .unwrap();
            let content = user["content"].as_array().unwrap();
            assert_eq!(content[0]["type"], json!("text"));
            assert_eq!(content[0]["text"], json!("What is in this image?"));
            assert_eq!(content[1]["type"], json!("image_url"));
            assert_eq!(
                content[1]["image_url"]["url"],
                json!("data:image/png;base64,aGVsbG8=")
            );

            let response = json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The multimodal payload arrived."
                    }
                }]
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            let _ = stream.write_all(http.as_bytes());
        });

        fs::write(
            context.config_path(),
            format!(
                "model:\n  default: test-model\n  provider: custom\n  base_url: http://{}\n  api_key: test-key\n  api_mode: chat_completions\n",
                addr
            ),
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_turn_with_user_content(
                &loaded,
                json!([
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]),
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides::default(),
                None,
                Some(&store),
            )
            .unwrap();

        assert_eq!(result.final_response, "The multimodal payload arrived.");
        let session_id = result.session_id.as_deref().unwrap();
        let messages = store.get_messages(session_id).unwrap();
        assert_eq!(messages[0].role, "user");
        assert_eq!(
            messages[0].content.as_ref().unwrap(),
            &json!([
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ])
        );
        join.join().unwrap();
    }

    #[test]
    fn chat_turn_retries_after_image_too_large_error_with_shrunk_data_url() {
        let data_url = large_png_data_url();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut prior_len = None;
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert_eq!(request_line.trim_end(), "POST /chat/completions HTTP/1.1");

                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = trimmed.split_once(':')
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                let image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
                    .as_str()
                    .unwrap();

                if request_index == 0 {
                    assert!(image_url.len() > IMAGE_SHRINK_TARGET_BYTES);
                    prior_len = Some(image_url.len());
                    let response_body = json!({
                        "error": {
                            "message": "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert!(image_url.len() < prior_len.unwrap());
                    let response_body = json!({
                        "choices": [{
                            "message": {"role": "assistant", "content": "Recovered after shrinking the image."},
                            "finish_reason": "stop"
                        }]
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_turn_with_user_content(
                &loaded,
                json!([
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]),
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("primary-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("primary-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(
            result.final_response,
            "Recovered after shrinking the image."
        );
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn chat_completion_turn_can_execute_code_tool() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let code = "from hermes_tools import write_file\nwrite_file('via_exec.txt', 'hello from execute_code')\nprint('sandbox done')";
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_exec_1",
                            "type": "function",
                            "function": {
                                "name": "execute_code",
                                "arguments": serde_json::to_string(&json!({"code": code})).unwrap()
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "execute_code complete."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Use execute_code to write a file",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "execute_code complete.");
        assert_eq!(
            fs::read_to_string(temp.path().join("via_exec.txt")).unwrap(),
            "hello from execute_code"
        );
    }

    #[test]
    fn anthropic_turn_executes_tools_and_returns_final_text() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_file",
                    "input": {
                        "path": "anthropic.txt",
                        "content": "hello from anthropic tool"
                    }
                }],
                "stop_reason": "tool_use"
            })
            .to_string(),
            json!({
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Anthropic flow complete."
                }],
                "stop_reason": "end_turn"
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file through anthropic mode",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic/claude-sonnet-4.6".to_string()),
                    provider: Some("anthropic".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("sk-ant-test".to_string()),
                    api_mode: Some("anthropic_messages".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.provider, "anthropic");
        assert_eq!(result.model, "claude-sonnet-4-6");
        assert_eq!(result.final_response, "Anthropic flow complete.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("anthropic.txt")).unwrap(),
            "hello from anthropic tool"
        );
    }

    #[test]
    fn anthropic_turn_retries_with_reduced_output_cap_from_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/messages "));

                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                if request_index == 0 {
                    assert_eq!(payload["max_tokens"], json!(16384));
                    let response_body = json!({
                        "error": {
                            "message": "max_tokens: 16384 > context_window: 200000 - input_tokens: 190000 = available_tokens: 10000"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert_eq!(payload["max_tokens"], json!(9936));
                    let response_body = json!({
                        "id": "msg_out_cap",
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": "Reduced output cap worked."
                        }],
                        "stop_reason": "end_turn"
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Need an anthropic answer.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic/claude-sonnet-4.6".to_string()),
                    provider: Some("anthropic".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("sk-ant-test".to_string()),
                    api_mode: Some("anthropic_messages".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(result.final_response, "Reduced output cap worked.");
        assert_eq!(result.api_calls, 2);
    }

    #[test]
    fn anthropic_long_context_tier_reduces_context_and_compacts() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            "model:\n  default: anthropic/claude-sonnet-4.6\n  provider: anthropic\n  api_key: sk-ant-test\n  api_mode: anthropic_messages\n  context_length: 400000\ncompression:\n  enabled: true\n  threshold: 0.002\n  protect_last_n: 2\n  target_ratio: 0.15\n",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let original_session_id = String::from("anthropic_long_ctx_seed");
        session_store
            .create_session(&SessionCreate {
                id: original_session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("claude-sonnet-4-6".to_string()),
                model_config: Some(json!({
                    "provider": "anthropic",
                    "base_url": "http://example.invalid",
                    "api_mode": "anthropic_messages",
                })),
                system_prompt: Some("Original system prompt.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        for idx in 0..5 {
            let content = "X".repeat(260);
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String(format!(
                        "Long-context user message {idx}: {content}"
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
            let _ = session_store.append_message(
                &original_session_id,
                &MessageAppend {
                    role: "assistant".to_string(),
                    content: Some(Value::String(format!(
                        "Long-context assistant message {idx}: {content}"
                    ))),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            );
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for request_index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/messages "));

                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or_default() == 0 {
                        break;
                    }
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length:") {
                        content_length = value.trim().parse::<usize>().unwrap_or_default();
                    }
                }

                let mut body = vec![0_u8; content_length];
                reader.read_exact(&mut body).unwrap();
                let payload = serde_json::from_slice::<Value>(&body).unwrap();
                if request_index == 0 {
                    let response_body = json!({
                        "error": {
                            "message": "Extra usage is required for long context requests"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                } else {
                    assert_eq!(payload["max_tokens"], json!(16384));
                    let response_body = json!({
                        "id": "msg_long_ctx",
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": "Recovered from long-context tier gate."
                        }],
                        "stop_reason": "end_turn"
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Continue the long-context session.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic/claude-sonnet-4.6".to_string()),
                    provider: Some("anthropic".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("sk-ant-test".to_string()),
                    api_mode: Some("anthropic_messages".to_string()),
                },
                Some(&original_session_id),
                Some(&session_store),
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(
            result.final_response,
            "Recovered from long-context tier gate."
        );
        let rotated_session_id = result.session_id.unwrap();
        assert_ne!(rotated_session_id, original_session_id);
        let original = session_store
            .get_session(&original_session_id)
            .unwrap()
            .unwrap();
        assert_eq!(original.end_reason.as_deref(), Some("compression"));
    }

    #[test]
    fn anthropic_messages_url_adds_v1_suffix_when_missing() {
        assert_eq!(
            anthropic_messages_url("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            anthropic_messages_url("https://api.minimax.io/anthropic"),
            "https://api.minimax.io/anthropic/v1/messages"
        );
        assert_eq!(
            anthropic_messages_url("https://opencode.ai/zen/v1"),
            "https://opencode.ai/zen/v1/messages"
        );
    }

    #[test]
    fn codex_turn_executes_tools_and_returns_final_text() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "id": "fc_abc123",
                    "call_id": "call_abc123",
                    "name": "write_file",
                    "arguments": "{\"path\":\"codex.txt\",\"content\":\"hello from codex tool\"}",
                    "status": "completed"
                }]
            })
            .to_string(),
            json!({
                "id": "resp_2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "Codex flow complete."
                    }]
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file through codex mode",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-5.4".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("codex_responses".to_string()),
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(result.final_response, "Codex flow complete.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("codex.txt")).unwrap(),
            "hello from codex tool"
        );
    }

    #[test]
    fn codex_turn_can_resume_existing_session() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let base_url_1 = serve_chat_sequence(vec![
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "First codex turn."
                    }]
                }]
            })
            .to_string(),
        ]);
        let first = context
            .run_chat_completions_turn(
                &loaded,
                "hello codex",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-5.4".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_1),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("codex_responses".to_string()),
                },
                None,
                Some(&session_store),
            )
            .unwrap();

        let session_id = first.session_id.clone().unwrap();
        let base_url_2 = serve_chat_sequence(vec![
            json!({
                "id": "resp_2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "Second codex turn."
                    }]
                }]
            })
            .to_string(),
        ]);
        let second = context
            .run_chat_completions_turn(
                &loaded,
                "continue codex",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("gpt-5.4".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_2),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("codex_responses".to_string()),
                },
                Some(&session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(second.session_id.as_deref(), Some(session_id.as_str()));
        assert_eq!(second.final_response, "Second codex turn.");
        assert_eq!(session_store.get_messages(&session_id).unwrap().len(), 4);
    }

    #[test]
    fn bedrock_convert_messages_merges_roles_and_tool_results() {
        let messages = vec![
            json!({"role": "system", "content": "Follow the system prompt."}),
            json!({"role": "user", "content": "First user turn."}),
            json!({"role": "user", "content": [{"type": "text", "text": "Second user turn."}]}),
            json!({
                "role": "assistant",
                "content": "Calling a tool.",
                "tool_calls": [{
                    "id": "bedrock-call-1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"bedrock.txt\",\"content\":\"hello from bedrock tool\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "bedrock-call-1",
                "content": "{\"success\":true}"
            }),
            json!({"role": "assistant", "content": "Done."}),
        ];

        let (system, converted) = bedrock_convert_messages(&messages).unwrap();
        assert_eq!(
            system,
            Some(vec![BedrockSystemContentBlock::Text(
                "Follow the system prompt.".to_string()
            )])
        );
        assert_eq!(converted.len(), 5);
        assert_eq!(converted[0].role(), &BedrockConversationRole::User);
        assert_eq!(converted[1].role(), &BedrockConversationRole::Assistant);
        assert_eq!(converted[2].role(), &BedrockConversationRole::User);
        assert_eq!(converted[3].role(), &BedrockConversationRole::Assistant);
        assert_eq!(converted[4].role(), &BedrockConversationRole::User);
        assert_eq!(converted[0].content().len(), 2);
        assert_eq!(converted[1].content().len(), 2);
        assert!(matches!(
            &converted[1].content()[1],
            BedrockContentBlock::ToolUse(tool_use)
                if tool_use.tool_use_id() == "bedrock-call-1" && tool_use.name() == "write_file"
        ));
        assert!(matches!(
            &converted[2].content()[0],
            BedrockContentBlock::ToolResult(tool_result)
                if tool_result.tool_use_id() == "bedrock-call-1"
        ));
    }

    #[test]
    fn normalize_bedrock_response_extracts_tool_calls_and_finish_reason() {
        let tool_input = serde_json_to_smithy_document(&json!({
            "path": "bedrock.txt",
            "content": "hello from bedrock tool"
        }));
        let response = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
            .output(BedrockConverseMessageOutput::Message(
                BedrockMessage::builder()
                    .role(BedrockConversationRole::Assistant)
                    .content(BedrockContentBlock::Text("Calling a tool.".to_string()))
                    .content(BedrockContentBlock::ToolUse(
                        aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                            .tool_use_id("bedrock-call-1")
                            .name("write_file")
                            .input(tool_input)
                            .build()
                            .unwrap(),
                    ))
                    .build()
                    .unwrap(),
            ))
            .stop_reason(BedrockStopReason::ToolUse)
            .build()
            .unwrap();

        let normalized = normalize_bedrock_response(response).unwrap();
        assert_eq!(normalized.content.as_deref(), Some("Calling a tool."));
        assert_eq!(normalized.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(normalized.tool_calls.len(), 1);
        assert_eq!(normalized.tool_calls[0].id, "bedrock-call-1");
        assert_eq!(normalized.tool_calls[0].name, "write_file");
        assert_eq!(
            normalized.tool_calls[0].json,
            json!({
                "path": "bedrock.txt",
                "content": "hello from bedrock tool"
            })
        );
    }

    #[test]
    fn convert_messages_to_anthropic_drops_thinking_only_assistant_turns() {
        let messages = vec![
            json!({"role": "system", "content": "You are Hermes."}),
            json!({"role": "user", "content": "Question one."}),
            json!({
                "role": "assistant",
                "content": "",
                "reasoning_content": "hidden chain of thought"
            }),
            json!({"role": "user", "content": "Question two."}),
        ];

        let runtime_model = crate::ModelRuntimeConfig {
            model: "claude-sonnet-4-6".to_string(),
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            api_key: "sk-ant-test".to_string(),
            api_mode: "anthropic_messages".to_string(),
            auth_type: "api_key".to_string(),
            default_headers: Vec::new(),
        };
        let (system_prompt, converted) =
            convert_messages_to_anthropic(&messages, &runtime_model).unwrap();

        assert_eq!(system_prompt.as_deref(), Some("You are Hermes."));
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"].as_str(), Some("user"));
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert_eq!(content[0]["text"].as_str(), Some("Question one."));
        assert_eq!(content[1]["type"].as_str(), Some("text"));
        assert_eq!(content[1]["text"].as_str(), Some("Question two."));
    }

    #[test]
    fn convert_messages_to_anthropic_preserves_signed_latest_thinking_for_native_anthropic() {
        let messages = vec![
            json!({"role": "system", "content": "You are Hermes."}),
            json!({"role": "user", "content": "Use the tool."}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_native_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"native.txt\",\"content\":\"hello\"}"
                    }
                }],
                "reasoning_details": [{
                    "type": "thinking",
                    "thinking": "signed thought",
                    "signature": "sig-native"
                }]
            }),
        ];
        let runtime_model = crate::ModelRuntimeConfig {
            model: "claude-sonnet-4-6".to_string(),
            provider: "anthropic".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            api_key: "sk-ant-test".to_string(),
            api_mode: "anthropic_messages".to_string(),
            auth_type: "api_key".to_string(),
            default_headers: Vec::new(),
        };

        let (_, converted) = convert_messages_to_anthropic(&messages, &runtime_model).unwrap();

        assert_eq!(converted.len(), 2);
        let assistant = &converted[1];
        let content = assistant["content"].as_array().unwrap();
        assert_eq!(content[0]["type"].as_str(), Some("thinking"));
        assert_eq!(content[0]["signature"].as_str(), Some("sig-native"));
        assert_eq!(content[1]["type"].as_str(), Some("tool_use"));
    }

    #[test]
    fn convert_messages_to_anthropic_strips_signed_thinking_for_third_party_endpoint() {
        let messages = vec![
            json!({"role": "user", "content": "Use the tool."}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_proxy_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"proxy.txt\",\"content\":\"hello\"}"
                    }
                }],
                "reasoning_details": [{
                    "type": "thinking",
                    "thinking": "signed thought",
                    "signature": "sig-proxy"
                }]
            }),
        ];
        let runtime_model = crate::ModelRuntimeConfig {
            model: "claude-sonnet-4-6".to_string(),
            provider: "custom".to_string(),
            base_url: "https://example-proxy.invalid/anthropic".to_string(),
            api_key: "test-key".to_string(),
            api_mode: "anthropic_messages".to_string(),
            auth_type: "api_key".to_string(),
            default_headers: Vec::new(),
        };

        let (_, converted) = convert_messages_to_anthropic(&messages, &runtime_model).unwrap();

        assert_eq!(converted.len(), 2);
        let assistant = &converted[1];
        let content = assistant["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"].as_str(), Some("tool_use"));
    }

    #[test]
    fn kimi_anthropic_replay_preserves_reasoning_content_as_thinking_block() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("Content-Length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut body = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut body);
            let payload = serde_json::from_slice::<Value>(&body).unwrap();
            let messages = payload["messages"].as_array().unwrap();
            assert!(request_line.starts_with("POST /v1/messages "));
            assert_eq!(messages.len(), 3);
            let assistant = &messages[1];
            assert_eq!(assistant["role"].as_str(), Some("assistant"));
            let content = assistant["content"].as_array().unwrap();
            assert_eq!(content[0]["type"].as_str(), Some("thinking"));
            assert_eq!(content[0]["thinking"].as_str(), Some(""));
            assert_eq!(content[1]["type"].as_str(), Some("tool_use"));
            assert_eq!(content[1]["id"].as_str(), Some("call_kimi_1"));

            let response_body = json!({
                "id": "msg_kimi",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Kimi replay accepted."
                }],
                "stop_reason": "end_turn"
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let runtime_model = crate::ModelRuntimeConfig {
            model: "moonshotai/Kimi-K2.5".to_string(),
            provider: "custom".to_string(),
            base_url: format!("http://{addr}"),
            api_key: "test-key".to_string(),
            api_mode: "anthropic_messages".to_string(),
            auth_type: "api_key".to_string(),
            default_headers: Vec::new(),
        };
        let messages = vec![
            json!({"role": "system", "content": "You are Hermes."}),
            json!({"role": "user", "content": "Use the tool."}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_kimi_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"kimi.txt\",\"content\":\"hello from kimi\"}"
                    }
                }],
                "reasoning_content": ""
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_kimi_1",
                "content": "tool done"
            }),
            json!({"role": "user", "content": "Continue."}),
        ];
        let client = build_http_client().unwrap();
        let response =
            send_anthropic_message(&client, &runtime_model, &messages, &[], None).unwrap();

        server.join().unwrap();
        assert_eq!(response.content.as_deref(), Some("Kimi replay accepted."));
    }

    #[test]
    fn anthropic_turn_ignores_replayed_thinking_only_assistant_message() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or_default() == 0 {
                    break;
                }
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() {
                    break;
                }
                if let Some((name, value)) = trimmed.split_once(':')
                    && name.eq_ignore_ascii_case("Content-Length")
                {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                }
            }
            let mut body = vec![0_u8; content_length];
            let _ = reader.read_exact(&mut body);
            let payload = serde_json::from_slice::<Value>(&body).unwrap();
            let messages = payload["messages"].as_array().unwrap();
            assert!(request_line.starts_with("POST /v1/messages "));
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0]["role"].as_str(), Some("user"));
            let content = messages[0]["content"].as_array().unwrap();
            assert_eq!(content.len(), 2);
            assert_eq!(content[0]["text"].as_str(), Some("First user turn."));
            assert_eq!(content[1]["text"].as_str(), Some("Second user turn."));

            let response_body = json!({
                "id": "msg_cleaned",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Thinking-only replay cleaned."
                }],
                "stop_reason": "end_turn"
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let session_id = "thinking_only_history".to_string();
        session_store
            .create_session(&SessionCreate {
                id: session_id.clone(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("claude-sonnet-4-6".to_string()),
                model_config: Some(json!({
                    "provider": "anthropic",
                    "base_url": format!("http://{addr}"),
                    "api_mode": "anthropic_messages",
                })),
                system_prompt: Some("You are Hermes.".to_string()),
                parent_session_id: None,
            })
            .unwrap();
        let _ = session_store.append_message(
            &session_id,
            &MessageAppend {
                role: "user".to_string(),
                content: Some(Value::String("First user turn.".to_string())),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        );
        let _ = session_store.append_message(
            &session_id,
            &MessageAppend {
                role: "assistant".to_string(),
                content: Some(Value::String(String::new())),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: Some("hidden chain of thought".to_string()),
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        );

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Second user turn.",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("claude-sonnet-4-6".to_string()),
                    provider: Some("anthropic".to_string()),
                    base_url: Some(format!("http://{addr}")),
                    api_key: Some("sk-ant-test".to_string()),
                    api_mode: Some("anthropic_messages".to_string()),
                },
                Some(&session_id),
                Some(&session_store),
            )
            .unwrap();

        server.join().unwrap();
        assert_eq!(result.final_response, "Thinking-only replay cleaned.");
    }

    #[test]
    fn chat_completion_turn_can_resume_existing_session() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let base_url_1 = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "First turn."
                    }
                }]
            })
            .to_string(),
        ]);
        let first = context
            .run_chat_completions_turn(
                &loaded,
                "hello",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_1),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                Some(&session_store),
            )
            .unwrap();

        let session_id = first.session_id.clone().unwrap();
        let base_url_2 = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Second turn."
                    }
                }]
            })
            .to_string(),
        ]);
        let second = context
            .run_chat_completions_turn(
                &loaded,
                "continue",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_2),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(second.session_id.as_deref(), Some(session_id.as_str()));
        assert_eq!(second.final_response, "Second turn.");
        assert_eq!(session_store.get_messages(&session_id).unwrap().len(), 4);
    }

    #[test]
    fn chat_resume_restores_todo_state_from_prior_tool_results() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let base_url_1 = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_todo_write",
                            "type": "function",
                            "function": {
                                "name": "todo",
                                "arguments": "{\"todos\":[{\"id\":\"1\",\"content\":\"draft plan\",\"status\":\"in_progress\"}]}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Planned."
                    }
                }]
            })
            .to_string(),
        ]);
        let first = context
            .run_chat_completions_turn(
                &loaded,
                "Plan the work",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_1),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                Some(&session_store),
            )
            .unwrap();

        let session_id = first.session_id.clone().unwrap();
        let base_url_2 = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_todo_read",
                            "type": "function",
                            "function": {
                                "name": "todo",
                                "arguments": "{}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Resumed."
                    }
                }]
            })
            .to_string(),
        ]);
        let second = context
            .run_chat_completions_turn(
                &loaded,
                "Continue the work",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url_2),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                Some(&session_id),
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(second.final_response, "Resumed.");

        let last_todo_message = session_store
            .get_messages(&session_id)
            .unwrap()
            .into_iter()
            .rev()
            .find(|message| message.tool_name.as_deref() == Some("todo"))
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(last_todo_message.content.unwrap().as_str().unwrap())
                .unwrap();
        let todos = payload["todos"].as_array().unwrap();
        assert_eq!(todos.len(), 1);
        assert_eq!(todos[0]["id"], json!("1"));
        assert_eq!(todos[0]["content"], json!("draft plan"));
        assert_eq!(todos[0]["status"], json!("in_progress"));
    }

    #[test]
    fn chat_completion_turn_can_use_clarify_tool() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_clarify_1",
                            "type": "function",
                            "function": {
                                "name": "clarify",
                                "arguments": "{\"question\":\"Pick one\",\"choices\":[\"Alpha\",\"Beta\"]}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Thanks."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_clarify_callback(|question, choices| {
                assert_eq!(question, "Pick one");
                assert_eq!(choices.unwrap(), &["Alpha".to_string(), "Beta".to_string()]);
                Ok("Beta".to_string())
            });
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Need a decision",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(result.final_response, "Thanks.");
        assert_eq!(result.tool_calls, 1);

        let session_id = result.session_id.unwrap();
        let tool_message = session_store
            .get_messages(&session_id)
            .unwrap()
            .into_iter()
            .find(|message| message.tool_name.as_deref() == Some("clarify"))
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(tool_message.content.unwrap().as_str().unwrap()).unwrap();
        assert_eq!(payload["user_response"], json!("Beta"));
    }

    #[test]
    fn chat_completion_turn_can_use_delegate_task() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_delegate_1",
                            "type": "function",
                            "function": {
                                "name": "delegate_task",
                                "arguments": "{\"goal\":\"Inspect src and summarize findings\"}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Child summary."
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Delegated."
                    }
                }]
            })
            .to_string(),
        ]);

        let overrides = ModelOverrides {
            model: Some("test-model".to_string()),
            provider: Some("custom".to_string()),
            base_url: Some(base_url),
            api_key: Some("test-key".to_string()),
            api_mode: Some("chat_completions".to_string()),
        };
        let delegate = crate::DelegateExecutor::new(
            context.clone(),
            loaded.clone(),
            "rust-delegate",
            vec!["hermes-cli".to_string()],
            overrides.clone(),
            temp.path(),
        );
        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_delegate_callback(move |request| delegate.execute(request));
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Need a child summary",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &overrides,
                None,
                Some(&session_store),
            )
            .unwrap();

        assert_eq!(result.final_response, "Delegated.");
        assert_eq!(result.tool_calls, 1);

        let session_id = result.session_id.unwrap();
        let tool_message = session_store
            .get_messages(&session_id)
            .unwrap()
            .into_iter()
            .find(|message| message.tool_name.as_deref() == Some("delegate_task"))
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(tool_message.content.unwrap().as_str().unwrap()).unwrap();
        assert_eq!(payload["results"][0]["summary"], json!("Child summary."));
        assert_eq!(payload["results"][0]["status"], json!("completed"));
    }

    #[test]
    fn memory_tool_updates_disk_but_not_the_saved_system_prompt_snapshot() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            temp.path().join("memories/MEMORY.md"),
            "Existing environment fact",
        )
        .unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_memory_add",
                            "type": "function",
                            "function": {
                                "name": "memory",
                                "arguments": "{\"action\":\"add\",\"target\":\"memory\",\"content\":\"New durable fact\"}"
                            }
                        }]
                    }
                }]
            })
            .to_string(),
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Saved memory."
                    }
                }]
            })
            .to_string(),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Remember a stable fact",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("test-model".to_string()),
                    provider: Some("custom".to_string()),
                    base_url: Some(base_url),
                    api_key: Some("test-key".to_string()),
                    api_mode: Some("chat_completions".to_string()),
                },
                None,
                Some(&session_store),
            )
            .unwrap();

        let session = session_store
            .get_session(result.session_id.as_deref().unwrap())
            .unwrap()
            .unwrap();
        let saved_prompt = session.system_prompt.unwrap();
        assert!(saved_prompt.contains("Existing environment fact"));
        assert!(!saved_prompt.contains("New durable fact"));
        let memory_file = fs::read_to_string(temp.path().join("memories/MEMORY.md")).unwrap();
        assert!(memory_file.contains("Existing environment fact"));
        assert!(memory_file.contains("New durable fact"));
    }
}
