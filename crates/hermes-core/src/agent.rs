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
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    HermesContext, HermesError, LoadedConfig, MessageAppend, ModelOverrides, SessionCreate,
    SessionStore, ToolRuntime, get_tool_definitions_for_runtime,
};

const DEFAULT_AGENT_TIMEOUT_SECS: u64 = 300;
const MAX_HTTP_ERROR_BODY_CHARS: usize = 4000;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTurnResult {
    pub final_response: String,
    pub reasoning: Option<String>,
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ResponseUsage {
    input_tokens: i64,
    output_tokens: i64,
    cache_read_tokens: i64,
    cache_write_tokens: i64,
    reasoning_tokens: i64,
}

#[derive(Debug, Clone)]
struct NormalizedAssistantResponse {
    content: Option<String>,
    tool_calls: Vec<PendingToolCall>,
    finish_reason: Option<String>,
    reasoning: Option<String>,
    reasoning_details: Option<Value>,
    codex_reasoning_items: Option<Value>,
    codex_message_items: Option<Value>,
    usage: Option<ResponseUsage>,
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

        let runtime_model = self.resolve_model_runtime(loaded, overrides)?;
        let mut tool_runtime = runtime.clone();
        tool_runtime.load_approvals(loaded);
        tool_runtime.load_context_engine(loaded);
        tool_runtime.load_shell_hooks(self, loaded);
        tool_runtime.load_checkpoints(loaded);
        if let Err(error) = tool_runtime.load_memory_store(&loaded.config.memory) {
            log::warn!(target: "run_agent", "memory bootstrap skipped: {error}");
        }
        let disabled_toolsets =
            (!loaded.config.memory.any_enabled()).then(|| vec![String::from("memory")]);
        let tools = get_tool_definitions_for_runtime(
            &tool_runtime,
            enabled_toolsets,
            disabled_toolsets.as_deref(),
        );
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
                    model_config: Some(json!({
                        "provider": runtime_model.provider,
                        "base_url": runtime_model.base_url,
                        "api_mode": runtime_model.api_mode,
                    })),
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
        if let Some(active_session_id) = session_id.as_deref().or(tool_runtime.current_session_id())
        {
            tool_runtime.start_context_engine_session(
                active_session_id,
                &runtime_model.model,
                &runtime_model.provider,
            );
        }

        let client = build_http_client()?;
        let mut api_calls = 0_u64;
        let mut tool_calls = 0_u64;

        for _ in 0..loaded.config.agent.max_turns {
            api_calls += 1;
            tool_runtime.begin_tool_turn();
            tool_runtime.emit_step(crate::StepUpdate {
                iteration: api_calls,
                prev_tools: extract_prev_tools(&messages),
            });
            apply_pending_steer_before_api_call(
                &tool_runtime,
                &mut messages,
                session_store,
                session_id.as_deref(),
            );
            let response = send_model_request(
                &client,
                &runtime_model,
                &messages,
                &tools,
                session_id.as_deref(),
            )?;
            let NormalizedAssistantResponse {
                content: assistant_content,
                tool_calls: pending_tool_calls,
                finish_reason,
                reasoning,
                reasoning_details,
                codex_reasoning_items,
                codex_message_items,
                usage,
            } = response;
            if let Some(store) = session_store
                && let Some(session_id) = session_id.as_deref()
            {
                let mut delta = crate::SessionUsageDelta {
                    api_call_count: 1,
                    ..crate::SessionUsageDelta::default()
                };
                if let Some(usage) = usage.as_ref() {
                    delta.input_tokens = usage.input_tokens;
                    delta.output_tokens = usage.output_tokens;
                    delta.cache_read_tokens = usage.cache_read_tokens;
                    delta.cache_write_tokens = usage.cache_write_tokens;
                    delta.reasoning_tokens = usage.reasoning_tokens;
                }
                let _ = store.record_session_usage(session_id, &delta);
            }

            if pending_tool_calls.is_empty() {
                let final_response = assistant_content
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| HermesError::State {
                        action: "parsing assistant response",
                        detail: "Assistant response had no text content.".to_string(),
                    })?
                    .to_string();
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
                            reasoning_content: None,
                            reasoning_details: reasoning_details.clone(),
                            codex_reasoning_items: codex_reasoning_items.clone(),
                            codex_message_items: codex_message_items.clone(),
                        },
                    );
                }
                return Ok(AgentTurnResult {
                    final_response,
                    reasoning,
                    api_calls,
                    tool_calls,
                    model: runtime_model.model,
                    provider: runtime_model.provider,
                    base_url: runtime_model.base_url,
                    session_id,
                });
            }

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

            messages.push(json!({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": normalized_tool_calls,
            }));
            if let Some(store) = session_store
                && let Some(session_id) = session_id.as_deref()
            {
                let _ = store.append_message(
                    session_id,
                    &MessageAppend {
                        role: "assistant".to_string(),
                        content: assistant_content.clone().map(Value::String),
                        tool_call_id: None,
                        tool_calls: Some(Value::Array(normalized_tool_calls.clone())),
                        tool_name: None,
                        token_count: None,
                        finish_reason: None,
                        reasoning: reasoning.clone(),
                        reasoning_content: None,
                        reasoning_details: reasoning_details.clone(),
                        codex_reasoning_items: codex_reasoning_items.clone(),
                        codex_message_items: codex_message_items.clone(),
                    },
                );
            }

            for tool_call in pending_tool_calls {
                tool_calls += 1;
                tool_runtime.maybe_checkpoint_before_tool(&tool_call.name, &tool_call.json);
                let mut result = crate::tools::dispatch_tool_with_messages(
                    &tool_call.name,
                    tool_call.json,
                    &tool_runtime,
                    &messages,
                );
                if let Some(steer_text) = tool_runtime.drain_pending_steer() {
                    result.push_str(&steer_marker(&steer_text));
                }
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

fn extract_prev_tools(messages: &[Value]) -> Vec<crate::StepToolRecord> {
    for (offset, message) in messages.iter().enumerate().rev() {
        let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        let start = offset + 1;
        let mut results_by_id = HashMap::new();
        for next in &messages[start..] {
            if next.get("role").and_then(Value::as_str) != Some("tool") {
                break;
            }
            if let Some(tool_call_id) = next.get("tool_call_id").and_then(Value::as_str) {
                results_by_id.insert(
                    tool_call_id.to_string(),
                    next.get("content")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                );
            }
        }

        return tool_calls
            .iter()
            .filter_map(|tool_call| {
                let function = tool_call.get("function")?.as_object()?;
                Some(crate::StepToolRecord {
                    name: function.get("name")?.as_str()?.to_string(),
                    result: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .and_then(|tool_call_id| results_by_id.get(tool_call_id).cloned())
                        .flatten(),
                    arguments: function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                })
            })
            .collect();
    }
    Vec::new()
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
) -> Result<NormalizedAssistantResponse, HermesError> {
    match runtime_model.api_mode.as_str() {
        "chat_completions" => {
            send_chat_completion(client, runtime_model, messages, tools, session_id)
        }
        "anthropic_messages" => send_anthropic_message(client, runtime_model, messages, tools),
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
    let response = send_model_request(client, runtime_model, messages, &[], None)?;
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
) -> Result<NormalizedAssistantResponse, HermesError> {
    if is_copilot_acp_runtime(runtime_model) {
        return send_copilot_acp_chat_completion(runtime_model, messages, tools);
    }
    if is_google_gemini_cli_runtime(runtime_model) {
        return send_google_gemini_chat_completion(client, runtime_model, messages, tools);
    }
    let qwen_messages = prepare_qwen_messages(messages);
    let request_messages = if is_qwen_portal_runtime(runtime_model) {
        &qwen_messages
    } else {
        messages
    };
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
    let body = read_json_response(response, "calling chat completions")?;
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
        reasoning: extract_message_text(assistant_message.get("reasoning"))
            .or_else(|| extract_message_text(assistant_message.get("reasoning_content"))),
        reasoning_details: assistant_message.get("reasoning_details").cloned(),
        codex_reasoning_items: assistant_message.get("codex_reasoning_items").cloned(),
        codex_message_items: assistant_message.get("codex_message_items").cloned(),
        usage: extract_openai_usage(parsed.get("usage")),
    })
}

fn send_anthropic_message(
    client: &Client,
    runtime_model: &crate::ModelRuntimeConfig,
    messages: &[Value],
    tools: &[crate::ToolDefinition],
) -> Result<NormalizedAssistantResponse, HermesError> {
    let (system_prompt, anthropic_messages) = convert_messages_to_anthropic(messages)?;
    let mut payload = json!({
        "model": runtime_model.model,
        "messages": anthropic_messages,
        "max_tokens": 16_384,
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
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
        usage: extract_anthropic_usage(parsed.get("usage")),
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
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
        usage: response.usage().map(extract_bedrock_usage),
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
    if requires_bearer_anthropic_auth(&runtime_model.base_url) {
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
            detail: format!("HTTP {}: {}", status.as_u16(), trimmed),
        });
    }
    Ok(body)
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

fn prepare_qwen_messages(messages: &[Value]) -> Vec<Value> {
    let mut prepared = messages.to_vec();
    for message in &mut prepared {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let Some(content) = object.get_mut("content") else {
            continue;
        };
        normalize_qwen_content(content);
    }

    for message in &mut prepared {
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

    prepared
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
            reasoning_details: None,
            codex_reasoning_items: None,
            codex_message_items: None,
            usage: None,
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
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
        usage: extract_google_usage(inner.get("usageMetadata")),
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
) -> Result<(Option<String>, Vec<Value>), HermesError> {
    let mut system_prompt = None;
    let mut converted = Vec::new();

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
                let mut blocks = Vec::new();
                if let Some(text) = extract_message_text(message.get("content"))
                    && !text.trim().is_empty()
                {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for tool_call in parse_tool_calls(message.get("tool_calls")) {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.json,
                    }));
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
                    converted.push(json!({
                        "role": "user",
                        "content": [block],
                    }));
                }
            }
            _ => {
                let content = extract_message_text(message.get("content"))
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "(empty message)".to_string());
                converted.push(json!({
                    "role": "user",
                    "content": content,
                }));
            }
        }
    }

    if system_prompt
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        system_prompt = None;
    }
    Ok((system_prompt, converted))
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
        reasoning_details: None,
        codex_reasoning_items: (!reasoning_items.is_empty()).then(|| Value::Array(reasoning_items)),
        codex_message_items: (!message_items.is_empty()).then(|| Value::Array(message_items)),
        usage: extract_responses_usage(response.get("usage")),
    })
}

fn extract_openai_usage(value: Option<&Value>) -> Option<ResponseUsage> {
    let usage = value?.as_object()?;
    Some(ResponseUsage {
        input_tokens: value_i64(usage.get("prompt_tokens")),
        output_tokens: value_i64(usage.get("completion_tokens")),
        cache_read_tokens: usage
            .get("prompt_tokens_details")
            .and_then(Value::as_object)
            .map_or(0, |details| value_i64(details.get("cached_tokens"))),
        cache_write_tokens: 0,
        reasoning_tokens: usage
            .get("completion_tokens_details")
            .and_then(Value::as_object)
            .map_or(0, |details| value_i64(details.get("reasoning_tokens"))),
    })
}

fn extract_anthropic_usage(value: Option<&Value>) -> Option<ResponseUsage> {
    let usage = value?.as_object()?;
    Some(ResponseUsage {
        input_tokens: value_i64(usage.get("input_tokens")),
        output_tokens: value_i64(usage.get("output_tokens")),
        cache_read_tokens: value_i64(usage.get("cache_read_input_tokens")),
        cache_write_tokens: value_i64(usage.get("cache_creation_input_tokens")),
        reasoning_tokens: 0,
    })
}

fn extract_google_usage(value: Option<&Value>) -> Option<ResponseUsage> {
    let usage = value?.as_object()?;
    Some(ResponseUsage {
        input_tokens: value_i64(usage.get("promptTokenCount")),
        output_tokens: value_i64(usage.get("candidatesTokenCount")),
        cache_read_tokens: value_i64(usage.get("cachedContentTokenCount")),
        cache_write_tokens: 0,
        reasoning_tokens: value_i64(usage.get("thoughtsTokenCount")),
    })
}

fn extract_responses_usage(value: Option<&Value>) -> Option<ResponseUsage> {
    let usage = value?.as_object()?;
    Some(ResponseUsage {
        input_tokens: value_i64(usage.get("input_tokens")),
        output_tokens: value_i64(usage.get("output_tokens")),
        cache_read_tokens: value_i64(usage.get("input_tokens_details").and_then(|details| {
            details
                .as_object()
                .and_then(|object| object.get("cached_tokens"))
        })),
        cache_write_tokens: 0,
        reasoning_tokens: value_i64(usage.get("output_tokens_details").and_then(|details| {
            details
                .as_object()
                .and_then(|object| object.get("reasoning_tokens"))
        })),
    })
}

fn extract_bedrock_usage(usage: &aws_sdk_bedrockruntime::types::TokenUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: i64::from(usage.input_tokens()),
        output_tokens: i64::from(usage.output_tokens()),
        cache_read_tokens: i64::from(usage.cache_read_input_tokens().unwrap_or_default()),
        cache_write_tokens: i64::from(usage.cache_write_input_tokens().unwrap_or_default()),
        reasoning_tokens: 0,
    }
}

fn value_i64(value: Option<&Value>) -> i64 {
    value.and_then(Value::as_i64).unwrap_or_default()
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

fn steer_marker(text: &str) -> String {
    format!("\n\nUser guidance: {}", text.trim())
}

fn apply_pending_steer_before_api_call(
    runtime: &ToolRuntime,
    messages: &mut [Value],
    session_store: Option<&SessionStore>,
    session_id: Option<&str>,
) {
    let Some(steer_text) = runtime.drain_pending_steer() else {
        return;
    };
    let marker = steer_marker(&steer_text);
    if append_steer_to_last_tool_message(messages, &marker) {
        if let (Some(store), Some(session_id)) = (session_store, session_id) {
            let _ = store.append_to_last_tool_message(session_id, &marker);
        }
        return;
    }
    runtime.restore_pending_steer(&steer_text);
}

fn append_steer_to_last_tool_message(messages: &mut [Value], marker: &str) -> bool {
    for message in messages.iter_mut().rev() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        match object.get_mut("content") {
            Some(Value::String(content)) => content.push_str(marker),
            Some(content) => {
                let replacement = format!("{}{}", content, marker);
                *content = Value::String(replacement);
            }
            None => {
                object.insert("content".to_string(), Value::String(marker.to_string()));
            }
        }
        return true;
    }
    false
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::path::Path;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use sha2::{Digest, Sha256};
    use tempfile::TempDir;

    #[derive(Clone)]
    struct FakeContextEngine {
        starts: Arc<Mutex<Vec<(String, crate::ContextEngineSessionStart)>>>,
        calls: Arc<Mutex<Vec<(String, Value, usize)>>>,
    }

    impl crate::ContextEngine for FakeContextEngine {
        fn name(&self) -> &str {
            "lcm"
        }

        fn tool_definitions(&self) -> Vec<crate::ToolDefinition> {
            vec![crate::ToolDefinition {
                name: "lcm_expand".to_string(),
                toolset: "context_engine".to_string(),
                description: "Expand prior context.".to_string(),
                emoji: "🧠".to_string(),
                schema: json!({
                    "name": "lcm_expand",
                    "description": "Expand context for a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                }),
            }]
        }

        fn on_session_start(
            &self,
            session_id: &str,
            event: &crate::ContextEngineSessionStart,
        ) -> Result<(), String> {
            self.starts
                .lock()
                .unwrap()
                .push((session_id.to_string(), event.clone()));
            Ok(())
        }

        fn handle_tool_call(&self, name: &str, args: &Value, messages: &[Value]) -> String {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone(), messages.len()));
            json!({
                "success": true,
                "engine": "lcm",
                "query": args.get("query").cloned().unwrap_or(Value::Null),
            })
            .to_string()
        }
    }

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

    fn checkpoint_commit_count(hermes_home: &Path, workdir: &Path) -> usize {
        git_checkpoint_output(
            hermes_home,
            workdir,
            &[
                "rev-list",
                "--count",
                &format!("refs/hermes/{}", checkpoint_project_hash(workdir)),
            ],
        )
        .trim()
        .parse::<usize>()
        .unwrap()
    }

    fn checkpoint_latest_reason(hermes_home: &Path, workdir: &Path) -> String {
        git_checkpoint_output(
            hermes_home,
            workdir,
            &[
                "log",
                "--format=%s",
                "-1",
                &format!("refs/hermes/{}", checkpoint_project_hash(workdir)),
            ],
        )
    }

    fn git_checkpoint_output(hermes_home: &Path, workdir: &Path, args: &[&str]) -> String {
        let store = hermes_home.join("checkpoints").join("store");
        let output = Command::new("git")
            .args(args)
            .current_dir(workdir)
            .env("GIT_DIR", &store)
            .env("GIT_WORK_TREE", workdir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn checkpoint_project_hash(workdir: &Path) -> String {
        let mut hasher = Sha256::new();
        hasher.update(
            fs::canonicalize(workdir)
                .unwrap()
                .display()
                .to_string()
                .as_bytes(),
        );
        let digest = hasher.finalize();
        digest[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
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
    fn google_gemini_cli_requests_code_assist_payload_and_persists_project() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HERMES_HOME");
        let previous_base = env::var_os(GOOGLE_CODE_ASSIST_BASE_URL_ENV);

        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join("hermes-home");
        fs::create_dir_all(hermes_home.join("auth")).unwrap();
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

        let result = send_model_request(&client, &runtime, &messages, &tools, None).unwrap();
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
    fn chat_completion_turn_returns_reasoning_in_result() {
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
                        "content": "Finished directly.",
                        "reasoning": "Plan first."
                    },
                    "finish_reason": "stop"
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Reply directly",
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

        assert_eq!(result.final_response, "Finished directly.");
        assert_eq!(result.reasoning.as_deref(), Some("Plan first."));
        assert_eq!(result.api_calls, 1);
        assert_eq!(result.tool_calls, 0);
    }

    #[test]
    fn chat_completion_turn_auto_loads_shell_hooks_from_config() {
        let _guard = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let script = temp.path().join("block.sh");
        fs::write(
            &script,
            "#!/usr/bin/env bash\ncat >/dev/null\nprintf '{\"decision\":\"block\",\"reason\":\"hook blocked write\"}\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let mut loaded = context.load_config_document().unwrap();
        loaded.raw = serde_yaml::from_str(&format!(
            "hooks:\n  pre_tool_call:\n    - command: \"{}\"\n      matcher: \"^write_file$\"\n",
            script.display()
        ))
        .unwrap();

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
                        "content": "Hook blocked the write."
                    }
                }]
            })
            .to_string(),
        ]);

        unsafe { std::env::set_var("HERMES_ACCEPT_HOOKS", "1") };
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
        unsafe { std::env::remove_var("HERMES_ACCEPT_HOOKS") };

        assert_eq!(result.final_response, "Hook blocked the write.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert!(!temp.path().join("notes.txt").exists());
        assert!(temp.path().join("shell-hooks-allowlist.json").exists());
    }

    #[test]
    fn chat_completion_turn_auto_loads_checkpoints_before_write_file() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let project = temp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        fs::write(project.join("tracked.txt"), "before\n").unwrap();

        let mut loaded = context.load_config_document().unwrap();
        loaded.raw = serde_yaml::from_str(
            "checkpoints:\n  enabled: true\n  max_snapshots: 20\n  max_total_size_mb: 500\n  max_file_size_mb: 10\n",
        )
        .unwrap();

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
                                "arguments": "{\"path\":\"tracked.txt\",\"content\":\"after\\n\"}"
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
                        "content": "Updated the tracked file."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(&project).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Update tracked.txt",
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

        assert_eq!(result.final_response, "Updated the tracked file.");
        assert_eq!(
            fs::read_to_string(project.join("tracked.txt")).unwrap(),
            "after\n"
        );
        assert_eq!(checkpoint_commit_count(temp.path(), &project), 1);
        assert_eq!(
            checkpoint_latest_reason(temp.path(), &project),
            "before write_file"
        );
    }

    #[test]
    fn chat_completion_turn_checkpoints_destructive_terminal_commands() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let project = temp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        fs::write(project.join("tracked.txt"), "before\n").unwrap();

        let mut loaded = context.load_config_document().unwrap();
        loaded.raw = serde_yaml::from_str(
            "checkpoints:\n  enabled: true\n  max_snapshots: 20\n  max_total_size_mb: 500\n  max_file_size_mb: 10\n",
        )
        .unwrap();

        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": "{\"command\":\"printf 'after\\\\n' > tracked.txt\",\"workdir\":\".\",\"timeout\":30}"
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
                        "content": "Terminal command finished."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(&project).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Overwrite tracked.txt from the terminal.",
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

        assert_eq!(result.final_response, "Terminal command finished.");
        assert_eq!(
            fs::read_to_string(project.join("tracked.txt")).unwrap(),
            "after\n"
        );
        assert_eq!(checkpoint_commit_count(temp.path(), &project), 1);
        assert_eq!(
            checkpoint_latest_reason(temp.path(), &project),
            "before terminal: printf 'after\\n' > tracked.txt"
        );
    }

    #[test]
    fn chat_completion_turn_auto_loads_approvals_from_config() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let mut loaded = context.load_config_document().unwrap();
        loaded.raw = serde_yaml::from_str("approvals:\n  mode: manual\n").unwrap();

        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": "{\"command\":\"bash -c \\\"printf approved\\\"\",\"timeout\":30}"
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
                        "content": "Approval-gated command finished."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_approval_callback(|_| Ok("always".to_string()));
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Run the guarded command.",
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

        assert_eq!(result.final_response, "Approval-gated command finished.");
        let saved = fs::read_to_string(temp.path().join("config.yaml")).unwrap();
        assert!(saved.contains("command_allowlist"));
        assert!(saved.contains("shell command via -c/-lc flag"));
    }

    #[test]
    fn chat_completion_turn_auto_loads_context_engine_from_config() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(temp.path().join("config.yaml"), "context:\n  engine: lcm\n").unwrap();
        let loaded = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();

        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lcm_expand",
                                "arguments": "{\"query\":\"alpha\"}"
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
                        "content": "Context expansion finished."
                    }
                }]
            })
            .to_string(),
        ]);

        let starts = Arc::new(Mutex::new(
            Vec::<(String, crate::ContextEngineSessionStart)>::new(),
        ));
        let calls = Arc::new(Mutex::new(Vec::<(String, Value, usize)>::new()));
        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_context_engine(FakeContextEngine {
                starts: Arc::clone(&starts),
                calls: Arc::clone(&calls),
            });
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Expand prior context for alpha.",
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
                Some(&store),
            )
            .unwrap();

        assert_eq!(result.final_response, "Context expansion finished.");
        let starts = starts.lock().unwrap().clone();
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].1.model, "test-model");
        assert_eq!(starts[0].1.provider, "custom");
        assert_eq!(result.session_id.as_deref(), Some(starts[0].0.as_str()));

        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "lcm_expand");
        assert_eq!(calls[0].1["query"], json!("alpha"));
        assert_eq!(calls[0].2, 3);
    }

    #[test]
    fn chat_completion_turn_emits_step_and_tool_progress_callbacks() {
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

        let progress = Arc::new(Mutex::new(Vec::<crate::ToolProgressUpdate>::new()));
        let progress_capture = Arc::clone(&progress);
        let steps = Arc::new(Mutex::new(Vec::<crate::StepUpdate>::new()));
        let steps_capture = Arc::clone(&steps);
        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_tool_progress_callback(move |update| {
                progress_capture.lock().unwrap().push(update.clone())
            })
            .with_step_callback(move |update| steps_capture.lock().unwrap().push(update.clone()));
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
        let progress = progress.lock().unwrap().clone();
        assert_eq!(progress.len(), 2);
        assert_eq!(progress[0].event_type, "tool.started");
        assert_eq!(progress[0].function_name.as_deref(), Some("write_file"));
        assert_eq!(progress[1].event_type, "tool.completed");
        assert_eq!(progress[1].is_error, Some(false));

        let steps = steps.lock().unwrap().clone();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].iteration, 1);
        assert!(steps[0].prev_tools.is_empty());
        assert_eq!(steps[1].iteration, 2);
        assert_eq!(steps[1].prev_tools.len(), 1);
        assert_eq!(steps[1].prev_tools[0].name, "write_file");
        let step_result: Value =
            serde_json::from_str(steps[1].prev_tools[0].result.as_deref().unwrap()).unwrap();
        assert_eq!(step_result["success"], json!(true));
        assert_eq!(
            step_result["path"],
            json!(temp.path().join("notes.txt").display().to_string())
        );
        assert_eq!(step_result["bytes_written"], json!(15));
        assert_eq!(step_result["line_count"], json!(1));
        assert_eq!(
            steps[1].prev_tools[0].arguments.as_deref(),
            Some("{\"path\":\"notes.txt\",\"content\":\"hello from tool\"}")
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
    fn normalize_codex_response_extracts_usage_counters() {
        let normalized = normalize_codex_response(&json!({
            "status": "completed",
            "usage": {
                "input_tokens": 90,
                "output_tokens": 21,
                "input_tokens_details": {
                    "cached_tokens": 7
                },
                "output_tokens_details": {
                    "reasoning_tokens": 5
                }
            },
            "output": [{
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "Codex usage."
                }]
            }]
        }))
        .unwrap();

        assert_eq!(normalized.content.as_deref(), Some("Codex usage."));
        assert_eq!(
            normalized.usage,
            Some(ResponseUsage {
                input_tokens: 90,
                output_tokens: 21,
                cache_read_tokens: 7,
                cache_write_tokens: 0,
                reasoning_tokens: 5,
            })
        );
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
    fn chat_completion_turn_persists_session_usage_counters() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let session_store = context.open_session_store().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let base_url = serve_chat_sequence(vec![
            json!({
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "prompt_tokens_details": {
                        "cached_tokens": 11
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 9
                    }
                },
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Tracked."
                    }
                }]
            })
            .to_string(),
        ]);

        let result = context
            .run_chat_completions_turn(
                &loaded,
                "hello usage",
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

        let usage = session_store
            .get_session_usage(result.session_id.as_deref().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(usage.api_call_count, 1);
        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.output_tokens, 45);
        assert_eq!(usage.cache_read_tokens, 11);
        assert_eq!(usage.cache_write_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 9);
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
