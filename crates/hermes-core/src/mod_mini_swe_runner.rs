//! Native Rust port of `mini_swe_runner.py`.
//!
//! A SWE runner that drives an OpenAI-compatible tool-calling loop, executes
//! the model's `terminal` commands in an execution environment, and emits a
//! trajectory in the **Hermes** `from`/`value` format (compatible with
//! `batch_runner.py` / `trajectory_compressor.py`).
//!
//! This port reproduces the portable core of the original:
//!   * the `terminal` tool definition (byte-for-byte description),
//!   * the Hermes-format system message and the message→trajectory conversion,
//!   * the OpenAI-compatible `chat/completions` request construction and
//!     response parsing (via `reqwest::blocking`),
//!   * the tool-calling agent loop including the `MINI_SWE_AGENT_FINAL_OUTPUT`
//!     completion signal and `max_iterations` guard,
//!   * a `LocalEnvironment` command executor (the `local` env type).
//!
//! The `docker`/`modal` backends in the Python original delegate to
//! `tools.environments.*`; here they are represented by the [`Environment`]
//! trait so callers can plug in their own backends, with [`LocalEnvironment`]
//! provided natively.
//!
//! The temperature contract reuses [`crate::ag_auxiliary_client`].

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;

use crate::ag_auxiliary_client::{fixed_temperature_for_model, TemperatureDirective};

/// Sentinel emitted by a finished agent run.
pub const FINAL_OUTPUT_SIGNAL: &str = "MINI_SWE_AGENT_FINAL_OUTPUT";

// ============================================================================
// Terminal tool definition
// ============================================================================

/// The exact textual description used for the `terminal` tool.
pub const TERMINAL_TOOL_DESCRIPTION: &str = r#"Execute bash commands in a sandboxed environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Provide the command to execute via the 'command' parameter
- Optional 'timeout' parameter in seconds (default: 60)

**Examples:**
- Run command: `{"command": "ls -la"}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Use non-interactive commands (avoid vim, nano, interactive python)
- Pipe to cat if output might be large
- Install tools with apt-get or pip as needed

**Completion:**
- When task is complete, output: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by your result
"#;

/// Build the OpenAI-style `terminal` tool definition (matches Hermes format).
pub fn terminal_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "terminal",
            "description": TERMINAL_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 60)"
                    }
                },
                "required": ["command"]
            }
        }
    })
}

/// The Hermes-format system message embedding the tool signatures.
///
/// `tools` is the list of OpenAI-style tool definitions.
pub fn hermes_system_message(tools: &[Value]) -> String {
    format!(
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. \
You may call one or more functions to assist with the user query. If available tools are not relevant in assisting \
with user query, just respond in natural conversational language. Don't make assumptions about what values to plug \
into functions. After calling & executing the functions, you will be provided with function results within \
<tool_response> </tool_response> XML tags. Here are the available tools:\n\
<tools>\n{}\n</tools>\n\
For each function call return a JSON object, with the following pydantic model json schema for each:\n\
{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'type': 'string'}}, \
'arguments': {{'title': 'Arguments', 'type': 'object'}}}}, 'required': ['name', 'arguments']}}\n\
Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n\
Example:\n<tool_call>\n{{'name': <function-name>,'arguments': <args-dict>}}\n</tool_call>",
        format_tools_for_system_message(tools)
    )
}

/// Reproduce `_format_tools_for_system_message`: re-shape each tool into a
/// flat `{name, description, parameters, required: null}` object and serialise.
pub fn format_tools_for_system_message(tools: &[Value]) -> String {
    let formatted: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let func = &tool["function"];
            json!({
                "name": func.get("name").cloned().unwrap_or(Value::Null),
                "description": func.get("description").cloned().unwrap_or_else(|| Value::String(String::new())),
                "parameters": func.get("parameters").cloned().unwrap_or_else(|| json!({})),
                "required": Value::Null,
            })
        })
        .collect();
    serde_json::to_string(&formatted).unwrap_or_else(|_| "[]".to_string())
}

// ============================================================================
// Execution environment abstraction
// ============================================================================

/// Result of executing a single command in an [`Environment`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecOutcome {
    pub output: String,
    pub exit_code: i64,
    pub error: Option<String>,
}

/// A command-execution backend (`local`, `docker`, `modal`, ...).
pub trait Environment {
    /// Execute `command` with an optional timeout (seconds), returning the
    /// combined output and an exit code. Filesystem state persists across calls.
    fn execute(&mut self, command: &str, timeout_secs: u64) -> ExecOutcome;
    /// Tear down any resources held by the environment.
    fn cleanup(&mut self) {}
}

/// Local-shell environment (`env_type == "local"`).
///
/// Runs commands with `bash -c` from a fixed working directory, capturing
/// merged stdout+stderr. Filesystem state persists because it operates on the
/// host filesystem.
pub struct LocalEnvironment {
    pub cwd: String,
    pub default_timeout: u64,
}

impl LocalEnvironment {
    pub fn new(cwd: impl Into<String>, default_timeout: u64) -> Self {
        Self {
            cwd: cwd.into(),
            default_timeout,
        }
    }
}

impl Environment for LocalEnvironment {
    fn execute(&mut self, command: &str, timeout_secs: u64) -> ExecOutcome {
        let _ = timeout_secs; // host execution; timeout is advisory here
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .output();
        match output {
            Ok(out) => {
                let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stderr.is_empty() {
                    combined.push_str(&stderr);
                }
                ExecOutcome {
                    output: combined,
                    exit_code: out.status.code().map(|c| c as i64).unwrap_or(-1),
                    error: None,
                }
            }
            Err(e) => ExecOutcome {
                output: String::new(),
                exit_code: -1,
                error: Some(e.to_string()),
            },
        }
    }
}

/// Recognised environment types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvType {
    Local,
    Docker,
    Modal,
}

impl EnvType {
    pub fn parse(s: &str) -> Result<EnvType, String> {
        match s {
            "local" => Ok(EnvType::Local),
            "docker" => Ok(EnvType::Docker),
            "modal" => Ok(EnvType::Modal),
            other => Err(format!(
                "Unknown environment type: {}. Use 'local', 'docker', or 'modal'",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EnvType::Local => "local",
            EnvType::Docker => "docker",
            EnvType::Modal => "modal",
        }
    }
}

// ============================================================================
// Internal message representation
// ============================================================================

/// A tool call as carried on an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolFunction {
    pub name: String,
    /// Raw arguments string as returned by the API (usually JSON text).
    pub arguments: String,
}

/// An internal conversation message (the runner's mutable history).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: "user".into(),
            content: Some(content.into()),
            ..Default::default()
        }
    }
    pub fn assistant_text(content: impl Into<String>) -> Self {
        Message {
            role: "assistant".into(),
            content: Some(content.into()),
            ..Default::default()
        }
    }
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Message {
            role: "tool".into(),
            content: Some(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            ..Default::default()
        }
    }
}

// ============================================================================
// Hermes trajectory format
// ============================================================================

/// A single `{"from": ..., "value": ...}` trajectory turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryTurn {
    pub from: String,
    pub value: String,
}

impl TrajectoryTurn {
    fn new(from: &str, value: impl Into<String>) -> Self {
        TrajectoryTurn {
            from: from.into(),
            value: value.into(),
        }
    }
}

/// Build a single XML `<tool_call>...</tool_call>` block for an assistant turn.
fn render_tool_call(tc: &ToolCall) -> String {
    // Parse arguments string as JSON; on failure use an empty object.
    let arguments: Value =
        serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}));
    let payload = json!({ "name": tc.function.name, "arguments": arguments });
    let serialized = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    format!("<tool_call>\n{}\n</tool_call>\n", serialized)
}

/// Convert internal messages into the Hermes trajectory format.
///
/// Mirrors `_convert_to_hermes_format`. `messages[0]` is assumed to be the
/// first user message and is replaced by `user_query`; subsequent messages are
/// walked, collapsing assistant tool-call turns together with the run of
/// `tool` responses that follow.
pub fn convert_to_hermes_format(
    messages: &[Message],
    user_query: &str,
    tools: &[Value],
) -> Vec<TrajectoryTurn> {
    let mut trajectory: Vec<TrajectoryTurn> = Vec::new();

    trajectory.push(TrajectoryTurn::new("system", hermes_system_message(tools)));
    trajectory.push(TrajectoryTurn::new("human", user_query));

    let mut i = 1usize;
    while i < messages.len() {
        let msg = &messages[i];

        if msg.role == "assistant" {
            if !msg.tool_calls.is_empty() {
                // Assistant message carrying tool calls.
                let mut content = String::new();
                if let Some(reasoning) = msg.reasoning.as_ref().filter(|r| !r.is_empty()) {
                    content = format!("<think>{}</think>", reasoning);
                }
                if let Some(c) = msg.content.as_ref().filter(|c| !c.is_empty()) {
                    content.push_str(c);
                    content.push('\n');
                }
                for tc in &msg.tool_calls {
                    content.push_str(&render_tool_call(tc));
                }
                trajectory.push(TrajectoryTurn::new("gpt", content.trim_end().to_string()));

                // Collect the consecutive run of tool responses.
                let mut tool_responses: Vec<String> = Vec::new();
                let mut j = i + 1;
                while j < messages.len() && messages[j].role == "tool" {
                    let tool_msg = &messages[j];
                    let raw = tool_msg.content.clone().unwrap_or_default();

                    // Try to embed parsed JSON; otherwise embed raw string.
                    let content_value: Value = {
                        let trimmed = raw.trim_start();
                        if trimmed.starts_with('{') || trimmed.starts_with('[') {
                            serde_json::from_str(&raw).unwrap_or(Value::String(raw.clone()))
                        } else {
                            Value::String(raw.clone())
                        }
                    };

                    let idx = tool_responses.len();
                    let name = if idx < msg.tool_calls.len() {
                        msg.tool_calls[idx].function.name.clone()
                    } else {
                        "unknown".to_string()
                    };

                    let body = json!({
                        "tool_call_id": tool_msg.tool_call_id.clone().unwrap_or_default(),
                        "name": name,
                        "content": content_value,
                    });
                    let serialized =
                        serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string());
                    tool_responses.push(format!("<tool_response>\n{}\n</tool_response>", serialized));
                    j += 1;
                }

                if !tool_responses.is_empty() {
                    trajectory.push(TrajectoryTurn::new("tool", tool_responses.join("\n")));
                    i = j - 1;
                }
            } else {
                // Plain assistant turn (no tool calls).
                let mut content = String::new();
                if let Some(reasoning) = msg.reasoning.as_ref().filter(|r| !r.is_empty()) {
                    content = format!("<think>{}</think>", reasoning);
                }
                content.push_str(msg.content.as_deref().unwrap_or(""));
                trajectory.push(TrajectoryTurn::new("gpt", content));
            }
        } else if msg.role == "user" {
            trajectory.push(TrajectoryTurn::new(
                "human",
                msg.content.clone().unwrap_or_default(),
            ));
        }

        i += 1;
    }

    trajectory
}

// ============================================================================
// OpenAI-compatible chat request / response
// ============================================================================

/// Convert an internal [`Message`] into the OpenAI request JSON shape.
pub fn message_to_api_json(msg: &Message) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), Value::String(msg.role.clone()));
    obj.insert(
        "content".into(),
        msg.content
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    if !msg.tool_calls.is_empty() {
        let calls: Vec<Value> = msg
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": tc.call_type,
                    "function": { "name": tc.function.name, "arguments": tc.function.arguments }
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &msg.tool_call_id {
        obj.insert("tool_call_id".into(), Value::String(id.clone()));
    }
    Value::Object(obj)
}

/// Build the JSON body for `POST /chat/completions`.
///
/// Adds a `temperature` key only when the model contract requires a fixed
/// value; omits it for [`TemperatureDirective::Omit`] / `None`.
pub fn build_chat_request_body(
    model: &str,
    base_url: Option<&str>,
    api_messages: &[Message],
    tools: &[Value],
) -> Value {
    let messages: Vec<Value> = api_messages.iter().map(message_to_api_json).collect();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "tools": tools,
    });

    if let TemperatureDirective::Fixed(v) = fixed_temperature_for_model(Some(model), base_url) {
        body["temperature"] = json!(v);
    }
    body
}

/// Parsed assistant message extracted from a chat-completion response.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantResponse {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Parse the first choice's `message` from an OpenAI-compatible response body.
pub fn parse_chat_response(body: &Value) -> Result<AssistantResponse, String> {
    let message = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| "response missing choices[0].message".to_string())?;

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let reasoning = message
        .get("reasoning")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut tool_calls = Vec::new();
    if let Some(arr) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in arr {
            let func = tc.get("function");
            let name = func
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = func
                .and_then(|f| f.get("arguments"))
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            tool_calls.push(ToolCall {
                id: tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                call_type: tc
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("function")
                    .to_string(),
                function: ToolFunction { name, arguments },
            });
        }
    }

    Ok(AssistantResponse {
        content,
        reasoning,
        tool_calls,
    })
}

// ============================================================================
// Tool-call argument helpers
// ============================================================================

/// Extract `command` (defaulting to the no-command echo) and `timeout`
/// (defaulting to `default_timeout`) from a tool call's arguments string.
pub fn parse_terminal_args(arguments: &str, default_timeout: u64) -> (String, u64) {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
    let command = parsed
        .get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "echo 'No command provided'".to_string());
    let timeout = parsed
        .get("timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(default_timeout);
    (command, timeout)
}

/// Build the JSON string stored as a `tool` message's content.
pub fn format_tool_result_json(outcome: &ExecOutcome) -> String {
    let body = json!({
        "content": {
            "output": outcome.output,
            "exit_code": outcome.exit_code,
            "error": outcome.error,
        }
    });
    serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string())
}

// ============================================================================
// Runner
// ============================================================================

/// The default ephemeral system prompt sent to the LLM (not saved to the
/// trajectory).
pub const AGENT_SYSTEM_PROMPT: &str = r#"You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."#;

/// Configuration mirroring `MiniSWERunner.__init__`.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub env_type: EnvType,
    pub image: String,
    pub cwd: String,
    pub max_iterations: usize,
    pub command_timeout: u64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            model: "anthropic/claude-sonnet-4.6".to_string(),
            base_url: None,
            api_key: None,
            env_type: EnvType::Local,
            image: "python:3.11-slim".to_string(),
            cwd: "/tmp".to_string(),
            max_iterations: 15,
            command_timeout: 60,
        }
    }
}

impl RunnerConfig {
    /// Resolve the effective base URL (defaults to OpenRouter, like the Python).
    pub fn effective_base_url(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string())
    }

    /// Resolve the effective API key, falling back through the same env var
    /// chain as the Python original.
    pub fn effective_api_key(&self) -> String {
        if let Some(k) = &self.api_key {
            return k.clone();
        }
        std::env::var("OPENROUTER_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_default()
    }
}

/// Result of a single task run (mirrors `run_task`'s return dict).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskResult {
    pub conversations: Vec<TrajectoryTurn>,
    pub completed: bool,
    pub api_calls: usize,
    pub metadata: TaskMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskMetadata {
    pub model: String,
    pub env_type: String,
    pub timestamp: String,
}

/// Drives the tool-calling loop for a single task.
pub struct MiniSweRunner {
    pub config: RunnerConfig,
    tools: Vec<Value>,
}

impl MiniSweRunner {
    pub fn new(config: RunnerConfig) -> Self {
        MiniSweRunner {
            config,
            tools: vec![terminal_tool_definition()],
        }
    }

    /// The tool definitions used by this runner.
    pub fn tools(&self) -> &[Value] {
        &self.tools
    }

    /// Send a single chat-completion request via `reqwest::blocking`.
    fn call_api(&self, api_messages: &[Message]) -> Result<AssistantResponse, String> {
        let body = build_chat_request_body(
            &self.config.model,
            self.config.base_url.as_deref(),
            api_messages,
            &self.tools,
        );
        let url = format!(
            "{}/chat/completions",
            self.config.effective_base_url().trim_end_matches('/')
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .post(&url)
            .bearer_auth(self.config.effective_api_key())
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        let text = resp.text().map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("API call failed: HTTP {} — {}", status, text));
        }
        let parsed: Value =
            serde_json::from_str(&text).map_err(|e| format!("invalid JSON response: {}", e))?;
        parse_chat_response(&parsed)
    }

    /// Run a single task against the supplied environment, returning the
    /// Hermes-format trajectory result. Mirrors `run_task`.
    pub fn run_task(&self, task: &str, env: &mut dyn Environment) -> TaskResult {
        let mut messages: Vec<Message> = vec![Message::user(task)];
        let mut api_call_count: usize = 0;
        let mut completed = false;

        while api_call_count < self.config.max_iterations {
            api_call_count += 1;

            // [system prompt] + messages
            let mut api_messages = Vec::with_capacity(messages.len() + 1);
            api_messages.push(Message {
                role: "system".into(),
                content: Some(AGENT_SYSTEM_PROMPT.to_string()),
                ..Default::default()
            });
            api_messages.extend(messages.iter().cloned());

            let assistant = match self.call_api(&api_messages) {
                Ok(a) => a,
                Err(e) => {
                    log::error!("{}", e);
                    break;
                }
            };

            if !assistant.tool_calls.is_empty() {
                messages.push(Message {
                    role: "assistant".into(),
                    content: assistant.content.clone(),
                    reasoning: assistant.reasoning.clone(),
                    tool_calls: assistant.tool_calls.clone(),
                    tool_call_id: None,
                });

                for tc in &assistant.tool_calls {
                    let (command, timeout) =
                        parse_terminal_args(&tc.function.arguments, self.config.command_timeout);
                    let outcome = env.execute(&command, timeout);
                    let result_json = format_tool_result_json(&outcome);

                    if outcome.output.contains(FINAL_OUTPUT_SIGNAL) {
                        completed = true;
                    }

                    messages.push(Message::tool(result_json, tc.id.clone()));
                }

                if completed {
                    break;
                }
            } else {
                let final_response = assistant.content.clone().unwrap_or_default();
                messages.push(Message::assistant_text(final_response));
                completed = true;
                break;
            }
        }

        env.cleanup();

        let trajectory = convert_to_hermes_format(&messages, task, &self.tools);

        TaskResult {
            conversations: trajectory,
            completed,
            api_calls: api_call_count,
            metadata: TaskMetadata {
                model: self.config.model.clone(),
                env_type: self.config.env_type.as_str().to_string(),
                timestamp: chrono::Local::now().to_rfc3339(),
            },
        }
    }
}

// ============================================================================
// Batch / prompt-file helpers
// ============================================================================

/// Parse a prompts JSONL file's lines into prompt strings.
///
/// Mirrors the `main()` batch loader: each non-empty line is parsed as JSON,
/// preferring the `prompt` then `task` key; on parse failure the raw line is
/// used verbatim. JSON objects lacking both keys yield an empty string.
pub fn parse_prompts_jsonl(contents: &str) -> Vec<String> {
    let mut prompts = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(entry) => {
                let p = entry
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .or_else(|| entry.get("task").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                prompts.push(p);
            }
            Err(_) => prompts.push(line.to_string()),
        }
    }
    prompts
}

/// Serialise a [`TaskResult`] as a single JSONL line (no trailing newline).
pub fn task_result_to_jsonl(result: &TaskResult) -> String {
    serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string())
}

/// Convenience: model the env-var fallback map the Python uses, for callers.
pub fn api_key_env_fallback() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("primary", "OPENROUTER_API_KEY");
    m.insert("secondary", "ANTHROPIC_API_KEY");
    m.insert("tertiary", "OPENAI_API_KEY");
    m
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_type_parse_roundtrip() {
        assert_eq!(EnvType::parse("local").unwrap(), EnvType::Local);
        assert_eq!(EnvType::parse("docker").unwrap(), EnvType::Docker);
        assert_eq!(EnvType::parse("modal").unwrap(), EnvType::Modal);
        assert!(EnvType::parse("k8s").is_err());
        assert_eq!(EnvType::Local.as_str(), "local");
    }

    #[test]
    fn tool_definition_shape() {
        let t = terminal_tool_definition();
        assert_eq!(t["function"]["name"], "terminal");
        assert_eq!(t["function"]["parameters"]["required"][0], "command");
        assert!(t["function"]["description"]
            .as_str()
            .unwrap()
            .contains("MINI_SWE_AGENT_FINAL_OUTPUT"));
    }

    #[test]
    fn format_tools_flattens_with_null_required() {
        let tools = vec![terminal_tool_definition()];
        let s = format_tools_for_system_message(&tools);
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v[0]["name"], "terminal");
        assert!(v[0]["required"].is_null());
        assert!(v[0]["parameters"]["properties"]["command"].is_object());
    }

    #[test]
    fn system_message_embeds_tools() {
        let tools = vec![terminal_tool_definition()];
        let msg = hermes_system_message(&tools);
        assert!(msg.contains("<tools>"));
        assert!(msg.contains("</tools>"));
        assert!(msg.contains("<tool_call>"));
        assert!(msg.contains("function calling AI model"));
    }

    #[test]
    fn parse_terminal_args_defaults() {
        let (cmd, to) = parse_terminal_args("{}", 60);
        assert_eq!(cmd, "echo 'No command provided'");
        assert_eq!(to, 60);

        let (cmd, to) = parse_terminal_args(r#"{"command":"ls -la","timeout":120}"#, 60);
        assert_eq!(cmd, "ls -la");
        assert_eq!(to, 120);

        // Invalid JSON → defaults.
        let (cmd, to) = parse_terminal_args("not json", 30);
        assert_eq!(cmd, "echo 'No command provided'");
        assert_eq!(to, 30);
    }

    #[test]
    fn tool_result_json_shape() {
        let outcome = ExecOutcome {
            output: "hello".into(),
            exit_code: 0,
            error: None,
        };
        let s = format_tool_result_json(&outcome);
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["content"]["output"], "hello");
        assert_eq!(v["content"]["exit_code"], 0);
        assert!(v["content"]["error"].is_null());
    }

    #[test]
    fn parse_chat_response_with_tool_calls() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": "running",
                    "reasoning": "I should list files",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "terminal", "arguments": "{\"command\":\"ls\"}" }
                    }]
                }
            }]
        });
        let parsed = parse_chat_response(&body).unwrap();
        assert_eq!(parsed.content.as_deref(), Some("running"));
        assert_eq!(parsed.reasoning.as_deref(), Some("I should list files"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_1");
        assert_eq!(parsed.tool_calls[0].function.name, "terminal");
        assert_eq!(parsed.tool_calls[0].function.arguments, "{\"command\":\"ls\"}");
    }

    #[test]
    fn parse_chat_response_arguments_object_stringified() {
        // Some providers return arguments as an object rather than a string.
        let body = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "c", "type": "function",
                        "function": { "name": "terminal", "arguments": {"command": "pwd"} }
                    }]
                }
            }]
        });
        let parsed = parse_chat_response(&body).unwrap();
        let args = &parsed.tool_calls[0].function.arguments;
        let v: Value = serde_json::from_str(args).unwrap();
        assert_eq!(v["command"], "pwd");
    }

    #[test]
    fn parse_chat_response_missing_message_errs() {
        assert!(parse_chat_response(&json!({})).is_err());
    }

    #[test]
    fn build_request_body_includes_tools_and_messages() {
        let msgs = vec![Message::user("do a thing")];
        let body = build_chat_request_body("some-model", None, &msgs, &[terminal_tool_definition()]);
        assert_eq!(body["model"], "some-model");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "do a thing");
        assert_eq!(body["tools"][0]["function"]["name"], "terminal");
    }

    #[test]
    fn message_to_api_json_tool_message() {
        let m = Message::tool("{\"x\":1}", "call_9");
        let v = message_to_api_json(&m);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_9");
        assert_eq!(v["content"], "{\"x\":1}");
    }

    #[test]
    fn convert_simple_assistant_turn() {
        let tools = vec![terminal_tool_definition()];
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hello back"),
        ];
        let traj = convert_to_hermes_format(&messages, "hi", &tools);
        assert_eq!(traj[0].from, "system");
        assert_eq!(traj[1].from, "human");
        assert_eq!(traj[1].value, "hi");
        assert_eq!(traj[2].from, "gpt");
        assert_eq!(traj[2].value, "hello back");
    }

    #[test]
    fn convert_tool_call_turn_collapses_responses() {
        let tools = vec![terminal_tool_definition()];
        let tc = ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: ToolFunction {
                name: "terminal".into(),
                arguments: "{\"command\":\"ls\"}".into(),
            },
        };
        let assistant = Message {
            role: "assistant".into(),
            content: Some("listing".into()),
            reasoning: None,
            tool_calls: vec![tc],
            tool_call_id: None,
        };
        let tool_resp = Message::tool(
            "{\"content\":{\"output\":\"file.txt\",\"exit_code\":0,\"error\":null}}",
            "call_1",
        );
        let messages = vec![Message::user("list files"), assistant, tool_resp];

        let traj = convert_to_hermes_format(&messages, "list files", &tools);
        // system, human, gpt(tool_call), tool(response)
        assert_eq!(traj.len(), 4);
        assert_eq!(traj[2].from, "gpt");
        assert!(traj[2].value.contains("<tool_call>"));
        assert!(traj[2].value.contains("listing"));
        assert_eq!(traj[3].from, "tool");
        assert!(traj[3].value.contains("<tool_response>"));
        assert!(traj[3].value.contains("\"name\":\"terminal\""));
        // The embedded content should be parsed JSON (object), not a string.
        assert!(traj[3].value.contains("\"output\":\"file.txt\""));
    }

    #[test]
    fn convert_includes_reasoning_as_think() {
        let tools = vec![terminal_tool_definition()];
        let assistant = Message {
            role: "assistant".into(),
            content: None,
            reasoning: Some("ponder".into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        let messages = vec![Message::user("q"), assistant];
        let traj = convert_to_hermes_format(&messages, "q", &tools);
        assert_eq!(traj[2].value, "<think>ponder</think>");
    }

    #[test]
    fn render_tool_call_bad_arguments_uses_empty_object() {
        let tc = ToolCall {
            id: "x".into(),
            call_type: "function".into(),
            function: ToolFunction {
                name: "terminal".into(),
                arguments: "not json".into(),
            },
        };
        let s = render_tool_call(&tc);
        assert!(s.contains("\"arguments\":{}"));
        assert!(s.contains("\"name\":\"terminal\""));
    }

    #[test]
    fn parse_prompts_jsonl_variants() {
        let input = "{\"prompt\": \"a\"}\n{\"task\": \"b\"}\n\nraw line\n{\"other\": 1}\n";
        let prompts = parse_prompts_jsonl(input);
        assert_eq!(prompts, vec!["a", "b", "raw line", ""]);
    }

    #[test]
    fn local_environment_executes() {
        let mut env = LocalEnvironment::new(".", 60);
        let outcome = env.execute("echo hello123", 60);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.output.contains("hello123"));
        assert!(outcome.error.is_none());
    }

    #[test]
    fn local_environment_nonzero_exit() {
        let mut env = LocalEnvironment::new(".", 60);
        let outcome = env.execute("exit 3", 60);
        assert_eq!(outcome.exit_code, 3);
    }

    #[test]
    fn config_defaults_and_resolution() {
        let cfg = RunnerConfig::default();
        assert_eq!(cfg.max_iterations, 15);
        assert_eq!(cfg.effective_base_url(), "https://openrouter.ai/api/v1");
        let cfg2 = RunnerConfig {
            base_url: Some("https://example.com/v1".into()),
            api_key: Some("key123".into()),
            ..RunnerConfig::default()
        };
        assert_eq!(cfg2.effective_base_url(), "https://example.com/v1");
        assert_eq!(cfg2.effective_api_key(), "key123");
    }

    #[test]
    fn task_result_serialises() {
        let result = TaskResult {
            conversations: vec![TrajectoryTurn::new("system", "s")],
            completed: true,
            api_calls: 2,
            metadata: TaskMetadata {
                model: "m".into(),
                env_type: "local".into(),
                timestamp: "2026-01-01T00:00:00+00:00".into(),
            },
        };
        let line = task_result_to_jsonl(&result);
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["completed"], true);
        assert_eq!(v["api_calls"], 2);
        assert_eq!(v["metadata"]["model"], "m");
        assert_eq!(v["conversations"][0]["from"], "system");
    }
}
