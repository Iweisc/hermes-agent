use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    HermesContext, HermesError, LoadedConfig, MessageAppend, ModelOverrides, SessionCreate,
    SessionStore, ToolRuntime, dispatch_tool, get_tool_definitions,
};

const DEFAULT_AGENT_TIMEOUT_SECS: u64 = 300;
const MAX_HTTP_ERROR_BODY_CHARS: usize = 4000;
const KANBAN_GUIDANCE: &str = "# Kanban task execution protocol\nUse kanban_show first to orient on the assigned task. Work inside HERMES_KANBAN_WORKSPACE unless the task explicitly requires otherwise. Heartbeat during long-running work, block when you need human input you cannot infer, and finish with kanban_complete(summary=..., metadata=...) or kanban_block(reason=...). Use kanban_create for real follow-up work instead of silently scope-creeping into it.";
const QWEN_CODE_VERSION: &str = "0.14.1";

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
    reasoning_details: Option<Value>,
    codex_reasoning_items: Option<Value>,
    codex_message_items: Option<Value>,
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
        if let Err(error) = tool_runtime.load_memory_store(&loaded.config.memory) {
            log::warn!(target: "run_agent", "memory bootstrap skipped: {error}");
        }
        let disabled_toolsets =
            (!loaded.config.memory.any_enabled()).then(|| vec![String::from("memory")]);
        let tools = get_tool_definitions(enabled_toolsets, disabled_toolsets.as_deref());
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

        let client = build_http_client()?;
        let mut api_calls = 0_u64;
        let mut tool_calls = 0_u64;

        for _ in 0..loaded.config.agent.max_turns {
            api_calls += 1;
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
            } = response;

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
        reasoning: None,
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
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

    let url = format!("{}/messages", runtime_model.base_url.trim_end_matches('/'));
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
    })
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
    let python = resolve_python_interpreter();
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
    let payload = json!({
        "region": region,
        "model": runtime_model.model,
        "messages": messages,
        "tools": tools.iter().map(|tool| tool.openai_schema()).collect::<Vec<_>>(),
        "max_tokens": 4096,
    });
    let script = r#"
import json
import sys

from agent.bedrock_adapter import call_converse

payload = json.load(sys.stdin)
response = call_converse(
    region=payload["region"],
    model=payload["model"],
    messages=payload.get("messages") or [],
    tools=payload.get("tools") or [],
    max_tokens=int(payload.get("max_tokens") or 4096),
)
choice = response.choices[0] if getattr(response, "choices", None) else None
message = getattr(choice, "message", None)
tool_calls = []
for tool_call in (getattr(message, "tool_calls", None) or []):
    function = getattr(tool_call, "function", None)
    tool_calls.append({
        "id": getattr(tool_call, "id", ""),
        "name": getattr(function, "name", ""),
        "arguments": getattr(function, "arguments", "{}"),
    })
usage = getattr(response, "usage", None)
print(json.dumps({
    "content": getattr(message, "content", None),
    "tool_calls": tool_calls,
    "finish_reason": getattr(choice, "finish_reason", None),
    "usage": {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    },
}))
"#;

    let mut command = Command::new(&python);
    command
        .arg("-c")
        .arg(script)
        .current_dir(repo_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut pythonpath_entries = vec![repo_root()];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        pythonpath_entries.extend(env::split_paths(&existing));
    }
    if let Ok(joined) = env::join_paths(pythonpath_entries) {
        command.env("PYTHONPATH", joined);
    }

    let mut child = command.spawn().map_err(|error| HermesError::State {
        action: "starting bedrock converse bridge",
        detail: format!("{} failed: {error}", python.display()),
    })?;
    if let Some(mut stdin) = child.stdin.take() {
        let bytes = serde_json::to_vec(&payload).map_err(|error| HermesError::State {
            action: "encoding bedrock converse payload",
            detail: error.to_string(),
        })?;
        stdin
            .write_all(&bytes)
            .map_err(|error| HermesError::State {
                action: "writing bedrock converse payload",
                detail: error.to_string(),
            })?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| HermesError::State {
            action: "waiting for bedrock converse bridge",
            detail: error.to_string(),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!(
                "bridge exited with code {}",
                output.status.code().unwrap_or(-1)
            )
        } else {
            stderr
        };
        return Err(HermesError::State {
            action: "calling bedrock converse",
            detail,
        });
    }

    let parsed =
        serde_json::from_slice::<Value>(&output.stdout).map_err(|error| HermesError::State {
            action: "decoding bedrock converse response",
            detail: format!("{}: {}", error, String::from_utf8_lossy(&output.stdout)),
        })?;
    Ok(NormalizedAssistantResponse {
        content: extract_message_text(parsed.get("content")),
        tool_calls: parse_bedrock_tool_calls(parsed.get("tool_calls"))?,
        finish_reason: parsed
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        reasoning: None,
        reasoning_details: None,
        codex_reasoning_items: None,
        codex_message_items: None,
    })
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

fn parse_bedrock_tool_calls(value: Option<&Value>) -> Result<Vec<PendingToolCall>, HermesError> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::new();
    for item in items {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let arguments_raw = item
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}")
            .to_string();
        let json = serde_json::from_str(&arguments_raw).unwrap_or_else(|_| json!({}));
        if id.trim().is_empty() || name.trim().is_empty() {
            return Err(HermesError::State {
                action: "parsing bedrock response",
                detail: format!("Invalid tool call payload: {item}"),
            });
        }
        parsed.push(PendingToolCall {
            id,
            name,
            arguments_raw,
            json,
        });
    }
    Ok(parsed)
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn resolve_python_interpreter() -> PathBuf {
    if let Some(venv) = env::var_os("VIRTUAL_ENV") {
        let candidate = PathBuf::from(venv).join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        });
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(if python_command_available("python3") {
        "python3"
    } else {
        "python"
    })
}

fn python_command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
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
    fn bedrock_turn_executes_tools_and_returns_final_text() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let loaded = context.load_config_document().unwrap();
        let pyroot = temp.path().join("pyroot");
        fs::create_dir_all(&pyroot).unwrap();
        fs::write(
            pyroot.join("boto3.py"),
            r#"
class _Client:
    def converse(self, **kwargs):
        messages = kwargs.get("messages") or []
        saw_tool_result = False
        for message in messages:
            for block in message.get("content", []):
                if "toolResult" in block:
                    saw_tool_result = True
                    break
            if saw_tool_result:
                break
        if not saw_tool_result:
            tool_config = kwargs.get("toolConfig", {})
            tools = tool_config.get("tools", [])
            assert any(
                tool.get("toolSpec", {}).get("name") == "write_file"
                for tool in tools
            ), kwargs
            return {
                "modelId": kwargs.get("modelId", ""),
                "output": {
                    "message": {
                        "content": [{
                            "toolUse": {
                                "toolUseId": "bedrock-call-1",
                                "name": "write_file",
                                "input": {
                                    "path": "bedrock.txt",
                                    "content": "hello from bedrock tool"
                                }
                            }
                        }]
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 11, "outputTokens": 7},
            }
        return {
            "modelId": kwargs.get("modelId", ""),
            "output": {
                "message": {
                    "content": [{
                        "text": "Bedrock flow complete."
                    }]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 13, "outputTokens": 5},
        }

def client(service_name, region_name=None):
    assert service_name == "bedrock-runtime", service_name
    assert region_name == "us-west-2", region_name
    return _Client()
"#,
        )
        .unwrap();

        let previous_pythonpath = std::env::var_os("PYTHONPATH");
        unsafe {
            std::env::set_var("PYTHONPATH", &pyroot);
        }
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let result = context
            .run_chat_completions_turn(
                &loaded,
                "Create a file through bedrock mode",
                &runtime,
                Some(&["hermes-cli".to_string()]),
                &ModelOverrides {
                    model: Some("anthropic.claude-sonnet-4-6-20250514-v1:0".to_string()),
                    provider: Some("bedrock".to_string()),
                    base_url: Some("https://bedrock-runtime.us-west-2.amazonaws.com".to_string()),
                    api_key: None,
                    api_mode: Some("bedrock_converse".to_string()),
                },
                None,
                None,
            )
            .unwrap();
        match previous_pythonpath {
            Some(value) => unsafe { std::env::set_var("PYTHONPATH", value) },
            None => unsafe { std::env::remove_var("PYTHONPATH") },
        }

        assert_eq!(result.provider, "bedrock");
        assert_eq!(result.model, "anthropic.claude-sonnet-4-6-20250514-v1:0");
        assert_eq!(result.final_response, "Bedrock flow complete.");
        assert_eq!(result.api_calls, 2);
        assert_eq!(result.tool_calls, 1);
        assert_eq!(
            fs::read_to_string(temp.path().join("bedrock.txt")).unwrap(),
            "hello from bedrock tool"
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
