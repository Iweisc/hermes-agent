//! HermesAgentLoop -- Reusable Multi-Turn Agent Engine.
//!
//! Native Rust port of `environments/agent_loop.py`.
//!
//! Runs the hermes-agent tool-calling loop using standard OpenAI-spec tool
//! calling. Works with any server that returns ChatCompletion objects with
//! `tool_calls`:
//!   - Phase 1: OpenAI server type (VLLM, SGLang, OpenRouter, OpenAI API)
//!   - Phase 2: ManagedServer with a client-side tool call parser
//!
//! The loop passes `tools=` and inspects `response.choices[0].message.tool_calls`,
//! identical to hermes-agent's `run_agent.py`. Tool execution is dispatched via a
//! caller-supplied dispatcher (the equivalent of `handle_function_call()` from
//! `model_tools.py`).
//!
//! ## Design notes / seams
//!
//! The Python module reaches into a lot of runtime singletons (a `server` object
//! with an async `chat_completion`, `handle_function_call`, the
//! tool-result-storage helpers, `get_active_env`, the per-loop `TodoStore`, ...).
//! Many of those are not uniformly available as concrete Rust types yet, so they
//! are modelled as trait/closure seams that the caller wires up:
//!
//! - [`Server`] — the chat-completion backend. `chat_completion` is synchronous
//!   here; the Python `async` is collapsed because the surrounding native runtime
//!   drives concurrency at a higher level (the Python thread-pool dance for
//!   asyncio-using backends is unnecessary in Rust).
//! - [`ToolDispatcher`] — executes a single non-builtin tool call. Wraps the
//!   equivalent of `handle_function_call()`.
//! - [`TodoTool`] — the per-loop todo handler (Python's `todo_tool` + `TodoStore`).
//! - [`PersistHooks`] — the tool-result persistence / budget layer
//!   (`maybe_persist_tool_result` + `enforce_turn_budget`). When `None`, results
//!   pass through unchanged.
//!
//! All seams have no-op / pass-through defaults so the loop is usable standalone.

use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::env_hermes_parser::HermesToolCallParser;

// ---------------------------------------------------------------------------
// Data structures (faithful to the Python @dataclasses)
// ---------------------------------------------------------------------------

/// Record of a tool execution error during the agent loop.
///
/// Mirrors the Python `ToolError` dataclass field-for-field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    /// Which turn the error occurred on (1-based).
    pub turn: usize,
    /// Which tool was called.
    pub tool_name: String,
    /// The arguments passed (truncated to 200 chars, matching Python).
    pub arguments: String,
    /// The error message.
    pub error: String,
    /// The raw result returned to the model.
    pub tool_result: String,
}

/// Result of running the agent loop.
///
/// Mirrors the Python `AgentResult` dataclass.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentResult {
    /// Full conversation history in OpenAI message format.
    pub messages: Vec<Value>,
    /// `ManagedServer.get_state()` if available (Phase 2), `None` otherwise.
    pub managed_state: Option<Value>,
    /// How many LLM calls were made.
    pub turns_used: usize,
    /// True if the model stopped calling tools naturally (vs hitting max_turns).
    pub finished_naturally: bool,
    /// Extracted reasoning content per turn.
    pub reasoning_per_turn: Vec<Option<String>>,
    /// Tool errors encountered during the loop.
    pub tool_errors: Vec<ToolError>,
}

/// A single tool call as seen on an assistant message.
///
/// Normalises both the OpenAI object form and the vLLM dict form into one shape.
/// Mirrors the data the Python `_tc_to_dict` produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw arguments JSON string (as the model emitted it).
    pub arguments: String,
}

impl ToolCall {
    /// Build the conversation-history dict for this tool call.
    ///
    /// Equivalent to Python's `_tc_to_dict`.
    pub fn to_history_value(&self) -> Value {
        json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        })
    }
}

/// An assistant message returned by [`Server::chat_completion`].
///
/// Captures the subset of `response.choices[0].message` the loop consumes.
/// `tool_calls` being empty corresponds to Python's falsy `tool_calls`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssistantMessage {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Reasoning extracted by the server already (Phase 1) or parsed from the
    /// raw provider fields below.
    pub reasoning_content: Option<String>,
    pub reasoning: Option<String>,
    /// OpenRouter-style reasoning details (`[{text: ...}]`).
    pub reasoning_details: Vec<Value>,
}

/// A parsed chat-completion response.
///
/// `choices` empty mirrors Python's `not response.choices`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatCompletion {
    pub choices: Vec<AssistantMessage>,
}

impl ChatCompletion {
    /// Convenience constructor from a single assistant message.
    pub fn single(message: AssistantMessage) -> Self {
        ChatCompletion {
            choices: vec![message],
        }
    }
}

/// Parameters for one `chat_completion` call, matching the Python `chat_kwargs`.
#[derive(Debug, Clone)]
pub struct ChatRequest<'a> {
    pub messages: &'a [Value],
    pub n: u32,
    pub temperature: f64,
    /// Only set when there are tool schemas (Python: `if self.tool_schemas`).
    pub tools: Option<&'a [Value]>,
    /// Only set when explicitly configured.
    pub max_tokens: Option<u32>,
    /// Provider-specific passthrough (OpenRouter provider prefs, transforms).
    pub extra_body: Option<&'a Value>,
}

// ---------------------------------------------------------------------------
// Seams
// ---------------------------------------------------------------------------

/// The chat-completion backend.
///
/// `chat_completion` returns `Err(..)` to model the Python `except Exception`
/// branch around the API call. `get_state` models the optional
/// `ManagedServer.get_state()`.
pub trait Server {
    fn chat_completion(&self, req: &ChatRequest<'_>) -> Result<ChatCompletion, String>;

    /// Returns `ManagedServer.get_state()` if the server supports it.
    ///
    /// Default `None` matches a regular OpenAI server (no `get_state`).
    fn get_state(&self) -> Option<Value> {
        None
    }
}

/// Executes a single non-builtin tool call.
///
/// Equivalent to `handle_function_call(tool_name, args, task_id, user_task)`.
/// Returns the raw tool-result string. Errors map to the Python
/// `except Exception` branch (recorded as a `ToolError` + JSON error result).
pub type ToolDispatcher<'a> =
    dyn Fn(&str, &Value, &str, Option<&str>) -> Result<String, String> + 'a;

/// The per-loop todo tool handler (Python's `todo_tool` over a `TodoStore`).
///
/// `todos` is the optional `todos` argument, `merge` the merge flag. Returns the
/// JSON result string.
pub type TodoTool<'a> = dyn FnMut(Option<&Value>, bool) -> String + 'a;

/// Tool-result persistence / per-turn budget hooks.
///
/// These wrap `maybe_persist_tool_result` and `enforce_turn_budget`. When the
/// whole struct is `None` the loop performs no persistence (pass-through).
pub struct PersistHooks<'a> {
    /// `maybe_persist_tool_result(content, tool_name, tool_use_id) -> content`.
    pub persist: Box<dyn FnMut(&str, &str, &str) -> String + 'a>,
    /// `enforce_turn_budget(last_n_tool_messages)`.
    ///
    /// Receives the slice of tool-role message Values appended this turn.
    pub enforce: Box<dyn FnMut(&mut [Value]) + 'a>,
}

// ---------------------------------------------------------------------------
// Reasoning extraction
// ---------------------------------------------------------------------------

/// Extract reasoning content from an assistant message.
///
/// Handles multiple provider formats, in priority order:
/// 1. `reasoning_content` field
/// 2. `reasoning` field
/// 3. `reasoning_details[].text` (OpenRouter style)
///
/// Port of `_extract_reasoning_from_message`. Note: `<think>` block extraction
/// from content is intentionally NOT done here (handled upstream).
pub fn extract_reasoning_from_message(message: &AssistantMessage) -> Option<String> {
    if let Some(rc) = &message.reasoning_content {
        if !rc.is_empty() {
            return Some(rc.clone());
        }
    }
    if let Some(r) = &message.reasoning {
        if !r.is_empty() {
            return Some(r.clone());
        }
    }
    for detail in &message.reasoning_details {
        if let Some(Value::String(text)) = detail.get("text") {
            if !text.is_empty() {
                return Some(text.clone());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// Runs hermes-agent's tool-calling loop using standard OpenAI-spec tool calling.
///
/// Same pattern as `run_agent.py`: pass `tools=`, inspect
/// `response.choices[0].message.tool_calls`, dispatch via the tool dispatcher.
pub struct HermesAgentLoop {
    pub tool_schemas: Vec<Value>,
    pub valid_tool_names: BTreeSet<String>,
    pub max_turns: usize,
    pub task_id: String,
    pub temperature: f64,
    pub max_tokens: Option<u32>,
    pub extra_body: Option<Value>,
}

impl HermesAgentLoop {
    /// Initialise the agent loop.
    ///
    /// `task_id` of `None` is replaced by a generated unique id (mirroring
    /// Python's `task_id or str(uuid.uuid4())`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tool_schemas: Vec<Value>,
        valid_tool_names: BTreeSet<String>,
        max_turns: usize,
        task_id: Option<String>,
        temperature: f64,
        max_tokens: Option<u32>,
        extra_body: Option<Value>,
    ) -> Self {
        HermesAgentLoop {
            tool_schemas,
            valid_tool_names,
            max_turns,
            task_id: task_id.unwrap_or_else(generate_task_id),
            temperature,
            max_tokens,
            extra_body,
        }
    }

    /// Get ManagedServer state if the server supports it.
    fn get_managed_state(&self, server: &dyn Server) -> Option<Value> {
        server.get_state()
    }

    /// Execute the full agent loop using standard OpenAI tool calling.
    ///
    /// `messages` is the initial conversation (system + user). It is modified in
    /// place as the conversation progresses, exactly as in Python.
    ///
    /// `dispatcher` runs non-builtin tools. `todo_tool` handles the `todo` tool
    /// against the per-loop store. `persist` is the optional tool-result
    /// persistence/budget layer.
    pub fn run(
        &self,
        server: &dyn Server,
        messages: &mut Vec<Value>,
        dispatcher: &ToolDispatcher<'_>,
        todo_tool: &mut TodoTool<'_>,
        mut persist: Option<&mut PersistHooks<'_>>,
    ) -> AgentResult {
        let mut reasoning_per_turn: Vec<Option<String>> = Vec::new();
        let mut tool_errors: Vec<ToolError> = Vec::new();

        // Extract user task from first user message for browser context. Capped
        // to 500 chars to avoid huge strings.
        let user_task: Option<String> = extract_user_task(messages);

        let task_prefix: String = self.task_id.chars().take(8).collect();

        for turn in 0..self.max_turns {
            // Build the chat_completion request -- standard OpenAI spec.
            let tools_ref: Option<&[Value]> = if self.tool_schemas.is_empty() {
                None
            } else {
                Some(&self.tool_schemas)
            };
            let req = ChatRequest {
                messages,
                n: 1,
                temperature: self.temperature,
                tools: tools_ref,
                max_tokens: self.max_tokens,
                extra_body: self.extra_body.as_ref(),
            };

            // Make the API call.
            let response = match server.chat_completion(&req) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("API call failed on turn {}: {}", turn + 1, e);
                    return AgentResult {
                        managed_state: self.get_managed_state(server),
                        turns_used: turn + 1,
                        finished_naturally: false,
                        reasoning_per_turn,
                        tool_errors,
                        messages: std::mem::take(messages),
                    };
                }
            };

            if response.choices.is_empty() {
                log::warn!("Empty response on turn {}", turn + 1);
                return AgentResult {
                    managed_state: self.get_managed_state(server),
                    turns_used: turn + 1,
                    finished_naturally: false,
                    reasoning_per_turn,
                    tool_errors,
                    messages: std::mem::take(messages),
                };
            }

            let mut assistant_msg = response.choices.into_iter().next().unwrap();

            // Extract reasoning content (all provider formats).
            let reasoning = extract_reasoning_from_message(&assistant_msg);
            reasoning_per_turn.push(reasoning.clone());

            // Fallback parser: if no structured tool_calls but content has raw
            // <tool_call> tags, parse them with the hermes parser.
            if assistant_msg.tool_calls.is_empty()
                && !self.tool_schemas.is_empty()
                && assistant_msg
                    .content
                    .as_deref()
                    .map(|c| c.contains("<tool_call>"))
                    .unwrap_or(false)
            {
                if let Some(content) = assistant_msg.content.clone() {
                    let parser = HermesToolCallParser::new();
                    let (parsed_content, parsed_calls) = parser.parse(&content);
                    if let Some(calls) = parsed_calls {
                        if !calls.is_empty() {
                            assistant_msg.tool_calls = calls
                                .into_iter()
                                .map(|c| ToolCall {
                                    id: c.id,
                                    name: c.name,
                                    arguments: c.arguments,
                                })
                                .collect();
                            // Python: only overwrite content when parsed_content
                            // is not None.
                            if let Some(pc) = parsed_content {
                                assistant_msg.content = Some(pc);
                            }
                            log::debug!(
                                "Fallback parser extracted {} tool calls from raw content",
                                assistant_msg.tool_calls.len()
                            );
                        }
                    }
                }
            }

            if !assistant_msg.tool_calls.is_empty() {
                let content_str = assistant_msg.content.clone().unwrap_or_default();

                // Build the assistant message dict for conversation history.
                let mut msg_dict = json!({
                    "role": "assistant",
                    "content": content_str,
                    "tool_calls": assistant_msg
                        .tool_calls
                        .iter()
                        .map(ToolCall::to_history_value)
                        .collect::<Vec<_>>(),
                });
                // Preserve reasoning_content for multi-turn chat templates.
                if let Some(r) = &reasoning {
                    msg_dict["reasoning_content"] = json!(r);
                }
                messages.push(msg_dict);

                let num_tcs = assistant_msg.tool_calls.len();

                // Execute each tool call.
                for tc in &assistant_msg.tool_calls {
                    let tool_name = tc.name.clone();
                    let tool_args_raw = tc.arguments.clone();
                    let mut tool_result: String;

                    if !self.valid_tool_names.contains(&tool_name) {
                        let mut names: Vec<&String> = self.valid_tool_names.iter().collect();
                        names.sort();
                        tool_result = json!({
                            "error": format!(
                                "Unknown tool '{}'. Available tools: {}",
                                tool_name,
                                py_sorted_list_repr(&names),
                            )
                        })
                        .to_string();
                        tool_errors.push(ToolError {
                            turn: turn + 1,
                            tool_name: tool_name.clone(),
                            arguments: truncate(&tool_args_raw, 200),
                            error: format!("Unknown tool '{}'", tool_name),
                            tool_result: tool_result.clone(),
                        });
                        log::warn!(
                            "Model called unknown tool '{}' on turn {}",
                            tool_name,
                            turn + 1
                        );
                    } else {
                        // Parse arguments.
                        let args: Option<Value> = match serde_json::from_str::<Value>(&tool_args_raw)
                        {
                            Ok(v) => Some(v),
                            Err(e) => {
                                let res = json!({
                                    "error": format!(
                                        "Invalid JSON in tool arguments: {}. Please retry with valid JSON.",
                                        e
                                    )
                                })
                                .to_string();
                                tool_errors.push(ToolError {
                                    turn: turn + 1,
                                    tool_name: tool_name.clone(),
                                    arguments: truncate(&tool_args_raw, 200),
                                    error: format!("Invalid JSON: {}", e),
                                    tool_result: res.clone(),
                                });
                                log::warn!(
                                    "Invalid JSON in tool call arguments for '{}': {}",
                                    tool_name,
                                    truncate(&tool_args_raw, 200)
                                );
                                // tool_result holds the JSON-error message.
                                // args is None so dispatch is skipped.
                                // We stash it below.
                                None
                            }
                        };

                        // When JSON parse failed, tool_result must be the error
                        // message we built. Recompute it deterministically: the
                        // last pushed tool_error for this branch carries it.
                        // Simpler: keep an explicit holder.
                        tool_result = String::new();
                        if args.is_none() {
                            // Re-derive the invalid-JSON result string. It equals
                            // the tool_result of the just-pushed ToolError.
                            tool_result = tool_errors
                                .last()
                                .map(|te| te.tool_result.clone())
                                .unwrap_or_default();
                        }

                        // Dispatch tool only if arguments parsed successfully.
                        if let Some(ref args) = args {
                            if tool_name == "terminal" {
                                let cmd_preview = args
                                    .get("command")
                                    .and_then(|v| v.as_str())
                                    .map(|s| truncate(s, 80))
                                    .unwrap_or_default();
                                log::info!("[{}] $ {}", task_prefix, cmd_preview);
                            }

                            if tool_name == "todo" {
                                let merge = args
                                    .get("merge")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                let todos = args.get("todos");
                                tool_result = todo_tool(todos, merge);
                            } else if tool_name == "memory" {
                                tool_result = json!({
                                    "error": "Memory is not available in RL environments."
                                })
                                .to_string();
                            } else if tool_name == "session_search" {
                                tool_result = json!({
                                    "error": "Session search is not available in RL environments."
                                })
                                .to_string();
                            } else {
                                match dispatcher(
                                    &tool_name,
                                    args,
                                    &self.task_id,
                                    user_task.as_deref(),
                                ) {
                                    Ok(res) => {
                                        tool_result = res;
                                    }
                                    Err(e) => {
                                        tool_result = json!({
                                            "error": format!("Tool execution failed: {}", e)
                                        })
                                        .to_string();
                                        tool_errors.push(ToolError {
                                            turn: turn + 1,
                                            tool_name: tool_name.clone(),
                                            arguments: truncate(&tool_args_raw, 200),
                                            error: e.clone(),
                                            tool_result: tool_result.clone(),
                                        });
                                        log::error!(
                                            "Tool '{}' execution failed on turn {}: {}",
                                            tool_name,
                                            turn + 1,
                                            e
                                        );
                                    }
                                }
                            }
                        }

                        // Also check if the tool returned an error in its JSON
                        // result (err present AND exit_code present AND < 0).
                        if let Ok(Value::Object(map)) =
                            serde_json::from_str::<Value>(&tool_result)
                        {
                            let err = map.get("error");
                            let exit_code = map.get("exit_code").and_then(|v| v.as_i64());
                            let err_truthy = err
                                .map(|v| !v.is_null() && truthy(v))
                                .unwrap_or(false);
                            if err_truthy {
                                if let Some(code) = exit_code {
                                    if code < 0 {
                                        tool_errors.push(ToolError {
                                            turn: turn + 1,
                                            tool_name: tool_name.clone(),
                                            arguments: truncate(&tool_args_raw, 200),
                                            error: value_to_py_str(err.unwrap()),
                                            tool_result: truncate(&tool_result, 500),
                                        });
                                    }
                                }
                            }
                        }
                    }

                    let tc_id = tc.id.clone();

                    if let Some(hooks) = persist.as_deref_mut() {
                        tool_result =
                            (hooks.persist)(&tool_result, &tool_name, &tc_id);
                    }

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    }));
                }

                // Per-turn aggregate budget enforcement over the tool messages
                // appended this turn (the last `num_tcs` entries).
                if num_tcs > 0 {
                    if let Some(hooks) = persist.as_deref_mut() {
                        let start = messages.len() - num_tcs;
                        (hooks.enforce)(&mut messages[start..]);
                    }
                }

                log::info!(
                    "[{}] turn {}: {} tools",
                    task_prefix,
                    turn + 1,
                    num_tcs
                );
            } else {
                // No tool calls -- model is done.
                let content_str = assistant_msg.content.clone().unwrap_or_default();
                let mut msg_dict = json!({
                    "role": "assistant",
                    "content": content_str,
                });
                if let Some(r) = &reasoning {
                    msg_dict["reasoning_content"] = json!(r);
                }
                messages.push(msg_dict);

                log::info!(
                    "[{}] turn {}: no tools (finished)",
                    task_prefix,
                    turn + 1
                );

                return AgentResult {
                    managed_state: self.get_managed_state(server),
                    turns_used: turn + 1,
                    finished_naturally: true,
                    reasoning_per_turn,
                    tool_errors,
                    messages: std::mem::take(messages),
                };
            }
        }

        // Hit max turns without the model stopping.
        log::info!(
            "Agent hit max_turns ({}) without finishing",
            self.max_turns
        );
        AgentResult {
            managed_state: self.get_managed_state(server),
            turns_used: self.max_turns,
            finished_naturally: false,
            reasoning_per_turn,
            tool_errors,
            messages: std::mem::take(messages),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the first non-empty user-message content, trimmed and capped at 500
/// chars. Equivalent to the Python `_user_task` extraction loop, which breaks on
/// the *first* user message regardless of whether it had usable content.
fn extract_user_task(messages: &[Value]) -> Option<String> {
    for msg in messages {
        if msg.get("role").and_then(|v| v.as_str()) == Some("user") {
            // Python: `content = msg.get("content", "")`; only strings count.
            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.chars().take(500).collect());
                }
            }
            // Python breaks after the first user message either way.
            break;
        }
    }
    None
}

/// Truncate a string to at most `max` characters (Python slice semantics on a
/// `str`, which is char-based; close enough for the truncation use here which is
/// applied to ASCII-dominant tool argument strings). Uses char boundaries.
fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Python-style truthiness for a JSON value, used to mirror `if err`.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Render a JSON value the way Python's `str(err)` would for the common cases
/// (a string error renders without quotes).
fn value_to_py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render a sorted list of tool names the way Python's `sorted(set)` repr does
/// inside an f-string: `['a', 'b', 'c']`.
fn py_sorted_list_repr(names: &[&String]) -> String {
    let inner = names
        .iter()
        .map(|n| format!("'{}'", n))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", inner)
}

/// Generate a task id (uuid4-like). Mirrors `str(uuid.uuid4())`.
fn generate_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut state = (nanos as u64) ^ 0x9E37_79B9_7F4A_7C15;
    let mut hex = String::with_capacity(32);
    const HEXC: &[u8] = b"0123456789abcdef";
    for _ in 0..32 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        hex.push(HEXC[(r % 16) as usize] as char);
    }
    // Lay out as 8-4-4-4-12 with version/variant nibbles set.
    let b = hex.as_bytes();
    fn s(b: &[u8], a: usize, z: usize) -> String {
        String::from_utf8_lossy(&b[a..z]).to_string()
    }
    format!(
        "{}-{}-4{}-8{}-{}",
        s(b, 0, 8),
        s(b, 8, 12),
        s(b, 13, 16),
        s(b, 17, 20),
        s(b, 20, 32)
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn names(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    // A scripted server that returns queued responses in order.
    struct ScriptedServer {
        responses: RefCell<Vec<Result<ChatCompletion, String>>>,
        state: Option<Value>,
    }
    impl Server for ScriptedServer {
        fn chat_completion(&self, _req: &ChatRequest<'_>) -> Result<ChatCompletion, String> {
            self.responses
                .borrow_mut()
                .remove(0)
        }
        fn get_state(&self) -> Option<Value> {
            self.state.clone()
        }
    }

    fn assistant(content: Option<&str>, calls: Vec<ToolCall>) -> AssistantMessage {
        AssistantMessage {
            content: content.map(|s| s.to_string()),
            tool_calls: calls,
            ..Default::default()
        }
    }

    #[test]
    fn finishes_naturally_with_no_tool_calls() {
        let server = ScriptedServer {
            responses: RefCell::new(vec![Ok(ChatCompletion::single(assistant(
                Some("all done"),
                vec![],
            )))]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![],
            names(&["terminal"]),
            30,
            Some("task1234abcd".to_string()),
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "hi there"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok("{}".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(res.finished_naturally);
        assert_eq!(res.turns_used, 1);
        assert_eq!(res.reasoning_per_turn, vec![None]);
        // user msg + assistant msg
        assert_eq!(res.messages.len(), 2);
        assert_eq!(res.messages[1]["role"], "assistant");
        assert_eq!(res.messages[1]["content"], "all done");
        assert!(res.messages[1].get("tool_calls").is_none());
    }

    #[test]
    fn dispatches_tool_then_finishes() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            name: "terminal".into(),
            arguments: "{\"command\": \"ls\"}".into(),
        }];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(Some("running"), calls))),
                Ok(ChatCompletion::single(assistant(Some("done"), vec![]))),
            ]),
            state: Some(json!({"k": "v"})),
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({"type": "function"})],
            names(&["terminal"]),
            30,
            Some("abcd1234efgh".to_string()),
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "list files"})];
        let dispatcher =
            |name: &str, args: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                assert_eq!(name, "terminal");
                assert_eq!(args["command"], "ls");
                Ok(json!({"stdout": "a\nb"}).to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(res.finished_naturally);
        assert_eq!(res.turns_used, 2);
        // user, assistant(tool_calls), tool, assistant(final)
        assert_eq!(res.messages.len(), 4);
        assert_eq!(res.messages[1]["role"], "assistant");
        assert_eq!(res.messages[2]["role"], "tool");
        assert_eq!(res.messages[2]["tool_call_id"], "call_1");
        assert_eq!(res.managed_state, Some(json!({"k": "v"})));
        assert!(res.tool_errors.is_empty());
    }

    #[test]
    fn unknown_tool_records_error() {
        let calls = vec![ToolCall {
            id: "call_x".into(),
            name: "nope".into(),
            arguments: "{}".into(),
        }];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(None, calls))),
                Ok(ChatCompletion::single(assistant(Some("ok"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal", "read_file"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                panic!("should not dispatch unknown tool")
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert_eq!(res.tool_errors.len(), 1);
        assert_eq!(res.tool_errors[0].tool_name, "nope");
        assert!(res.tool_errors[0].error.contains("Unknown tool"));
        // sorted available names appear in the result
        let tr = &res.messages[2]["content"].as_str().unwrap();
        assert!(tr.contains("'read_file'"));
        assert!(tr.contains("'terminal'"));
    }

    #[test]
    fn invalid_json_arguments_record_error_and_skip_dispatch() {
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "terminal".into(),
            arguments: "{not json".into(),
        }];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(None, calls))),
                Ok(ChatCompletion::single(assistant(Some("ok"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                panic!("should not dispatch with invalid json")
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert_eq!(res.tool_errors.len(), 1);
        assert!(res.tool_errors[0].error.starts_with("Invalid JSON"));
        let tr = res.messages[2]["content"].as_str().unwrap();
        assert!(tr.contains("Invalid JSON in tool arguments"));
    }

    #[test]
    fn max_turns_reached() {
        // Always returns a tool call so the loop never finishes naturally.
        struct Looper;
        impl Server for Looper {
            fn chat_completion(
                &self,
                _req: &ChatRequest<'_>,
            ) -> Result<ChatCompletion, String> {
                Ok(ChatCompletion::single(AssistantMessage {
                    content: Some("again".into()),
                    tool_calls: vec![ToolCall {
                        id: "c".into(),
                        name: "terminal".into(),
                        arguments: "{}".into(),
                    }],
                    ..Default::default()
                }))
            }
        }
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal"]),
            3,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok("{}".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &Looper,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(!res.finished_naturally);
        assert_eq!(res.turns_used, 3);
    }

    #[test]
    fn empty_response_returns_unfinished() {
        let server = ScriptedServer {
            responses: RefCell::new(vec![Ok(ChatCompletion::default())]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(vec![], names(&[]), 30, None, 1.0, None, None);
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok("{}".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(!res.finished_naturally);
        assert_eq!(res.turns_used, 1);
    }

    #[test]
    fn api_error_returns_unfinished() {
        let server = ScriptedServer {
            responses: RefCell::new(vec![Err("boom".to_string())]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(vec![], names(&[]), 30, None, 1.0, None, None);
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok("{}".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(!res.finished_naturally);
        assert_eq!(res.turns_used, 1);
        assert_eq!(res.reasoning_per_turn.len(), 0);
    }

    #[test]
    fn fallback_parser_extracts_tool_calls_from_content() {
        let raw = "thinking <tool_call>{\"name\": \"terminal\", \"arguments\": {\"command\": \"ls\"}}</tool_call>";
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(Some(raw), vec![]))),
                Ok(ChatCompletion::single(assistant(Some("done"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatched = std::cell::Cell::new(false);
        let dispatcher =
            |name: &str, args: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                dispatched.set(true);
                assert_eq!(name, "terminal");
                assert_eq!(args["command"], "ls");
                Ok("{}".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert!(dispatched.get(), "fallback-parsed tool call should dispatch");
        // assistant content trimmed to text before the tag
        assert_eq!(res.messages[1]["content"], "thinking");
    }

    #[test]
    fn reasoning_extraction_priority() {
        let m = AssistantMessage {
            reasoning_content: Some("rc".into()),
            reasoning: Some("r".into()),
            ..Default::default()
        };
        assert_eq!(extract_reasoning_from_message(&m), Some("rc".into()));

        let m = AssistantMessage {
            reasoning: Some("r".into()),
            ..Default::default()
        };
        assert_eq!(extract_reasoning_from_message(&m), Some("r".into()));

        let m = AssistantMessage {
            reasoning_details: vec![json!({"text": "deet"})],
            ..Default::default()
        };
        assert_eq!(extract_reasoning_from_message(&m), Some("deet".into()));

        let m = AssistantMessage::default();
        assert_eq!(extract_reasoning_from_message(&m), None);
    }

    #[test]
    fn todo_and_special_tools_handled_locally() {
        let calls = vec![
            ToolCall {
                id: "t1".into(),
                name: "todo".into(),
                arguments: "{\"merge\": true, \"todos\": [1,2]}".into(),
            },
            ToolCall {
                id: "m1".into(),
                name: "memory".into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "s1".into(),
                name: "session_search".into(),
                arguments: "{}".into(),
            },
        ];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(None, calls))),
                Ok(ChatCompletion::single(assistant(Some("ok"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["todo", "memory", "session_search"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                panic!("builtin tools must not dispatch")
            };
        let mut todo = |todos: Option<&Value>, merge: bool| {
            assert!(merge);
            assert_eq!(todos, Some(&json!([1, 2])));
            String::from("{\"ok\": true}")
        };
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        // user, assistant(3 calls), 3 tool msgs, assistant(final)
        assert_eq!(res.messages.len(), 6);
        assert_eq!(res.messages[2]["content"], "{\"ok\": true}");
        assert!(res.messages[3]["content"]
            .as_str()
            .unwrap()
            .contains("Memory is not available"));
        assert!(res.messages[4]["content"]
            .as_str()
            .unwrap()
            .contains("Session search is not available"));
    }

    #[test]
    fn json_error_with_negative_exit_code_records_error() {
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "terminal".into(),
            arguments: "{}".into(),
        }];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(None, calls))),
                Ok(ChatCompletion::single(assistant(Some("ok"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok(json!({"error": "timed out", "exit_code": -1}).to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            None,
        );
        assert_eq!(res.tool_errors.len(), 1);
        assert_eq!(res.tool_errors[0].error, "timed out");
    }

    #[test]
    fn persist_hooks_invoked() {
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "terminal".into(),
            arguments: "{}".into(),
        }];
        let server = ScriptedServer {
            responses: RefCell::new(vec![
                Ok(ChatCompletion::single(assistant(None, calls))),
                Ok(ChatCompletion::single(assistant(Some("ok"), vec![]))),
            ]),
            state: None,
        };
        let loop_ = HermesAgentLoop::new(
            vec![json!({})],
            names(&["terminal"]),
            30,
            None,
            1.0,
            None,
            None,
        );
        let mut messages = vec![json!({"role": "user", "content": "go"})];
        let dispatcher =
            |_n: &str, _a: &Value, _t: &str, _u: Option<&str>| -> Result<String, String> {
                Ok("BIGRESULT".to_string())
            };
        let mut todo = |_t: Option<&Value>, _m: bool| String::from("{}");
        let enforced = std::cell::Cell::new(false);
        let mut hooks = PersistHooks {
            persist: Box::new(|content: &str, _name: &str, _id: &str| {
                format!("[persisted]{}", content)
            }),
            enforce: Box::new(|_msgs: &mut [Value]| {
                enforced.set(true);
            }),
        };
        let res = loop_.run(
            &server,
            &mut messages,
            &dispatcher,
            &mut todo,
            Some(&mut hooks),
        );
        assert_eq!(res.messages[2]["content"], "[persisted]BIGRESULT");
        assert!(enforced.get());
    }

    #[test]
    fn user_task_extraction_caps_and_trims() {
        let long: String = "x".repeat(600);
        let messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": format!("  {}  ", long)}),
        ];
        let task = extract_user_task(&messages).unwrap();
        assert_eq!(task.chars().count(), 500);
    }

    #[test]
    fn task_id_default_is_uuid_shaped() {
        let id = generate_task_id();
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }
}
