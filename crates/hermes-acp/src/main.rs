use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{LocalResult, TimeZone, Utc};
use clap::{Parser, Subcommand};
use hermes_core::{
    EnvLoadReport, HermesContext, LoadedConfig, LoggingMode, MessageAppend, MessageRecord,
    ModelOverrides, SessionCreate, SessionRecord, SessionStore, attach_python_plugin_runtime,
    get_provider_profile, get_tool_definitions, normalize_provider_alias,
};
use serde_json::{Value, json};

const ACP_PROTOCOL_VERSION: i64 = 1;
const ACP_SESSION_SOURCE: &str = "rust-acp";
const ACP_PAGE_SIZE: usize = 50;
const DEFAULT_MODE_ID: &str = "default";

#[derive(Parser, Debug)]
#[command(name = "hermes-acp", version, about = "Hermes ACP Rust adapter")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Serve,
    Env,
    Status,
}

#[derive(Debug, Clone)]
struct AcpSessionState {
    cwd: PathBuf,
    overrides: ModelOverrides,
    mode_id: String,
    config_options: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
struct PromptInput {
    user_content: Value,
    display_text: String,
    plain_text: Option<String>,
}

impl AcpSessionState {
    fn new(cwd: PathBuf) -> Self {
        Self {
            cwd,
            overrides: ModelOverrides::default(),
            mode_id: DEFAULT_MODE_ID.to_string(),
            config_options: BTreeMap::new(),
        }
    }

    fn from_record(record: &SessionRecord, cwd_override: Option<PathBuf>) -> Result<Self, String> {
        let mut state = AcpSessionState::new(cwd_override.unwrap_or_else(|| PathBuf::from("/")));
        if let Some(config) = record.model_config.as_ref().and_then(Value::as_object) {
            if let Some(cwd) = config
                .get("cwd")
                .and_then(Value::as_str)
                .and_then(non_empty_trimmed)
            {
                state.cwd = PathBuf::from(cwd);
            }
            state.overrides.provider = config
                .get("provider")
                .and_then(Value::as_str)
                .and_then(non_empty_trimmed);
            state.overrides.base_url = config
                .get("base_url")
                .and_then(Value::as_str)
                .and_then(non_empty_trimmed);
            state.overrides.api_mode = config
                .get("api_mode")
                .and_then(Value::as_str)
                .and_then(non_empty_trimmed);
        }
        state.overrides.model = record.model.as_deref().and_then(non_empty_trimmed);
        if !state.cwd.is_absolute() {
            return Err(format!(
                "Session '{}' has a non-absolute cwd '{}'",
                record.id,
                state.cwd.display()
            ));
        }
        Ok(state)
    }
}

struct AcpServer<'a> {
    context: &'a HermesContext,
    config: &'a LoadedConfig,
    session_store: &'a SessionStore,
    sessions: HashMap<String, AcpSessionState>,
}

impl<'a> AcpServer<'a> {
    fn new(
        context: &'a HermesContext,
        config: &'a LoadedConfig,
        session_store: &'a SessionStore,
    ) -> Self {
        Self {
            context,
            config,
            session_store,
            sessions: HashMap::new(),
        }
    }

    fn run<R: BufRead, W: Write>(&mut self, reader: &mut R, writer: &mut W) -> io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let message = match serde_json::from_str::<Value>(trimmed) {
                Ok(value) => value,
                Err(error) => {
                    write_jsonrpc_error(
                        writer,
                        Value::Null,
                        -32700,
                        "Parse error",
                        Some(json!({"detail": error.to_string()})),
                    )?;
                    continue;
                }
            };
            if !message.is_object() {
                write_jsonrpc_error(
                    writer,
                    Value::Null,
                    -32600,
                    "Invalid Request",
                    Some(json!({"detail": "JSON-RPC frame must be an object"})),
                )?;
                continue;
            }

            let id = message.get("id").cloned();
            let method = message.get("method").and_then(Value::as_str);
            if let Some(method) = method {
                let params = message.get("params").cloned().unwrap_or(Value::Null);
                if let Some(id) = id {
                    if let Err(error) = self.handle_request(method, id, params, writer) {
                        write_jsonrpc_error(
                            writer,
                            message.get("id").cloned().unwrap_or(Value::Null),
                            -32603,
                            "Internal error",
                            Some(json!({"detail": error})),
                        )?;
                    }
                } else if let Err(error) = self.handle_notification(method, params) {
                    log::warn!(target: "acp_adapter", "notification {} failed: {}", method, error);
                }
                continue;
            }
        }
        Ok(())
    }

    fn handle_request<W: Write>(
        &mut self,
        method: &str,
        id: Value,
        params: Value,
        writer: &mut W,
    ) -> Result<(), String> {
        match method {
            "initialize" => {
                let result = self.handle_initialize();
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "authenticate" => {
                write_jsonrpc_result(writer, id, Value::Null).map_err(|error| error.to_string())
            }
            "session/new" => {
                let result = self.handle_new_session(&params)?;
                let session_id = result
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())?;
                if let Some(session_id) = session_id.as_deref() {
                    self.send_available_commands_update(writer, session_id)?;
                }
                Ok(())
            }
            "session/load" => {
                let response = self.handle_load_session(&params)?;
                write_jsonrpc_result(writer, id, response.clone())
                    .map_err(|error| error.to_string())?;
                if let Some(session_id) = params.get("sessionId").and_then(Value::as_str)
                    && !response.is_null()
                {
                    self.replay_history(writer, session_id)?;
                    self.send_available_commands_update(writer, session_id)?;
                }
                Ok(())
            }
            "session/resume" => {
                let response = self.handle_resume_session(&params)?;
                write_jsonrpc_result(writer, id, response.clone())
                    .map_err(|error| error.to_string())?;
                if let Some(session_id) = params.get("sessionId").and_then(Value::as_str) {
                    if !response.is_null() {
                        self.replay_history(writer, session_id)?;
                    }
                    self.send_available_commands_update(writer, session_id)?;
                }
                Ok(())
            }
            "session/fork" => {
                let result = self.handle_fork_session(&params)?;
                write_jsonrpc_result(writer, id, result.clone())
                    .map_err(|error| error.to_string())?;
                if let Some(session_id) = result.get("sessionId").and_then(Value::as_str) {
                    self.send_available_commands_update(writer, session_id)?;
                }
                Ok(())
            }
            "session/list" => {
                let result = self.handle_list_sessions(&params)?;
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "session/close" => {
                let result = self.handle_close_session(&params)?;
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "session/set_model" => {
                let result = self.handle_set_session_model(&params)?;
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "session/set_mode" => {
                let result = self.handle_set_session_mode(&params)?;
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "session/set_config_option" => {
                let result = self.handle_set_config_option(&params)?;
                write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string())
            }
            "session/prompt" => self.handle_prompt(id, &params, writer),
            other => write_jsonrpc_error(
                writer,
                id,
                -32601,
                "Method not found",
                Some(json!({"method": other})),
            )
            .map_err(|error| error.to_string()),
        }
    }

    fn handle_notification(&mut self, method: &str, params: Value) -> Result<(), String> {
        match method {
            "session/cancel" => {
                let session_id = required_string(&params, "sessionId")?;
                log::info!(target: "acp_adapter", "cancel requested for session {}", session_id);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn handle_initialize(&self) -> Value {
        json!({
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "agentInfo": {
                "name": "hermes-agent",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "agentCapabilities": {
                "loadSession": true,
                "mcpCapabilities": {
                    "http": false,
                    "sse": false,
                },
                "promptCapabilities": {
                    "audio": false,
                    "embeddedContext": false,
                    "image": true,
                },
                "sessionCapabilities": {
                    "close": {},
                    "fork": {},
                    "list": {},
                    "resume": {},
                }
            },
            "authMethods": [],
        })
    }

    fn handle_new_session(&mut self, params: &Value) -> Result<Value, String> {
        let cwd_raw = required_string(params, "cwd")?;
        let cwd = normalize_cwd(&cwd_raw)?;
        let session_id = format!("acp_{:x}", unix_ts_nanos());
        let state = AcpSessionState::new(cwd.clone());
        self.create_or_update_persisted_session(&session_id, &state, None)?;
        self.sessions.insert(session_id.clone(), state.clone());
        Ok(json!({
            "sessionId": session_id,
            "models": self.model_state_json(&state),
            "modes": mode_state_json(&state.mode_id),
        }))
    }

    fn handle_load_session(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let cwd_raw = required_string(params, "cwd")?;
        let cwd = normalize_cwd(&cwd_raw)?;
        let Some(state) = self.load_session_state(&session_id, Some(cwd))? else {
            return Ok(Value::Null);
        };
        Ok(json!({
            "models": self.model_state_json(&state),
            "modes": mode_state_json(&state.mode_id),
        }))
    }

    fn handle_resume_session(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let cwd_raw = required_string(params, "cwd")?;
        let cwd = normalize_cwd(&cwd_raw)?;
        let state = match self.load_session_state(&session_id, Some(cwd.clone()))? {
            Some(state) => state,
            None => {
                let state = AcpSessionState::new(cwd);
                self.create_or_update_persisted_session(&session_id, &state, None)?;
                self.sessions.insert(session_id.clone(), state.clone());
                state
            }
        };
        Ok(json!({
            "models": self.model_state_json(&state),
            "modes": mode_state_json(&state.mode_id),
        }))
    }

    fn handle_fork_session(&mut self, params: &Value) -> Result<Value, String> {
        let source_session_id = required_string(params, "sessionId")?;
        let cwd_raw = required_string(params, "cwd")?;
        let cwd = normalize_cwd(&cwd_raw)?;
        let source_state = self
            .load_session_state(&source_session_id, Some(cwd.clone()))?
            .ok_or_else(|| format!("Session '{}' not found", source_session_id))?;
        let source_record = self
            .session_store
            .get_session(&source_session_id)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("Session '{}' not found", source_session_id))?;
        let source_messages = self
            .session_store
            .get_messages(&source_session_id)
            .map_err(|error| error.to_string())?;

        let fork_id = format!("acp_{:x}", unix_ts_nanos());
        self.create_or_update_persisted_session(&fork_id, &source_state, Some(&source_record.id))?;
        for message in source_messages {
            self.session_store
                .append_message(&fork_id, &message_append_from_record(&message))
                .map_err(|error| error.to_string())?;
        }
        self.sessions.insert(fork_id.clone(), source_state.clone());
        Ok(json!({
            "sessionId": fork_id,
            "models": self.model_state_json(&source_state),
            "modes": mode_state_json(&source_state.mode_id),
        }))
    }

    fn handle_list_sessions(&self, params: &Value) -> Result<Value, String> {
        let cursor = optional_string(params, "cursor")?;
        let cwd_filter = optional_string(params, "cwd")?
            .as_deref()
            .map(normalize_cwd)
            .transpose()?;

        let rows = self
            .session_store
            .search_sessions(Some(ACP_SESSION_SOURCE), 1_000, 0)
            .map_err(|error| error.to_string())?;
        let mut sessions = Vec::new();
        for row in rows {
            let Some(record) = self
                .session_store
                .get_session(&row.id)
                .map_err(|error| error.to_string())?
            else {
                continue;
            };
            let state = match AcpSessionState::from_record(&record, None) {
                Ok(state) => state,
                Err(_) => continue,
            };
            if let Some(filter) = cwd_filter.as_ref()
                && &state.cwd != filter
            {
                continue;
            }
            sessions.push(json!({
                "sessionId": row.id,
                "cwd": state.cwd.display().to_string(),
                "title": row.title,
                "updatedAt": iso_timestamp(row.last_active),
            }));
        }

        if let Some(cursor) = cursor {
            if let Some(index) = sessions.iter().position(|item| {
                item.get("sessionId").and_then(Value::as_str) == Some(cursor.as_str())
            }) {
                sessions = sessions.into_iter().skip(index + 1).collect();
            } else {
                sessions.clear();
            }
        }

        let has_more = sessions.len() > ACP_PAGE_SIZE;
        let page = sessions.into_iter().take(ACP_PAGE_SIZE).collect::<Vec<_>>();
        let next_cursor = if has_more {
            page.last()
                .and_then(|item| item.get("sessionId"))
                .cloned()
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        let mut response = json!({
            "sessions": page,
        });
        if !next_cursor.is_null()
            && let Some(object) = response.as_object_mut()
        {
            object.insert("nextCursor".to_string(), next_cursor);
        }
        Ok(response)
    }

    fn handle_close_session(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let existed = self
            .session_store
            .get_session(&session_id)
            .map_err(|error| error.to_string())?
            .is_some();
        if !existed {
            return Ok(Value::Null);
        }
        self.session_store
            .end_session(&session_id, "closed")
            .map_err(|error| error.to_string())?;
        self.sessions.remove(&session_id);
        Ok(json!({}))
    }

    fn handle_set_session_model(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let model_id = required_string(params, "modelId")?;
        let mut state = match self.load_session_state(&session_id, None)? {
            Some(state) => state,
            None => return Ok(Value::Null),
        };
        apply_model_selection(&mut state, &model_id);
        self.persist_runtime_metadata(&session_id, &state)?;
        self.sessions.insert(session_id, state);
        Ok(json!({}))
    }

    fn handle_set_session_mode(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let mode_id = required_string(params, "modeId")?;
        let mut state = match self.load_session_state(&session_id, None)? {
            Some(state) => state,
            None => return Ok(Value::Null),
        };
        state.mode_id = non_empty_trimmed(&mode_id).unwrap_or_else(|| DEFAULT_MODE_ID.to_string());
        self.sessions.insert(session_id, state);
        Ok(json!({}))
    }

    fn handle_set_config_option(&mut self, params: &Value) -> Result<Value, String> {
        let session_id = required_string(params, "sessionId")?;
        let config_id = required_string(params, "configId")?;
        let Some(value) = params.get("value").cloned() else {
            return Err("value is required".to_string());
        };
        if !value.is_boolean() && !value.is_string() {
            return Err("value must be a boolean or string".to_string());
        }
        let mut state = match self.load_session_state(&session_id, None)? {
            Some(state) => state,
            None => return Ok(Value::Null),
        };
        state.config_options.insert(config_id, value);
        self.sessions.insert(session_id, state);
        Ok(json!({
            "configOptions": [],
        }))
    }

    fn handle_prompt<W: Write>(
        &mut self,
        id: Value,
        params: &Value,
        writer: &mut W,
    ) -> Result<(), String> {
        let session_id = required_string(params, "sessionId")?;
        let message_id = optional_string(params, "messageId")?;
        let prompt = params
            .get("prompt")
            .ok_or_else(|| "prompt is required".to_string())?;
        let state = match self.load_session_state(&session_id, None)? {
            Some(state) => state,
            None => {
                let result = json!({
                    "stopReason": "refusal",
                    "userMessageId": message_id,
                });
                return write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string());
            }
        };
        let prompt_input = match extract_prompt_input(prompt) {
            Ok(value) => value,
            Err(error) => {
                self.send_text_update(writer, &session_id, "agent_message_chunk", &error)?;
                let result = json!({
                    "stopReason": "refusal",
                    "userMessageId": message_id,
                });
                return write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string());
            }
        };
        if !prompt_input.display_text.trim().is_empty() {
            self.send_text_update(
                writer,
                &session_id,
                "user_message_chunk",
                &prompt_input.display_text,
            )?;
        }
        if !has_meaningful_prompt_content(&prompt_input.user_content) {
            let result = json!({
                "stopReason": "end_turn",
                "userMessageId": message_id,
            });
            return write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string());
        }

        if let Some(prompt_text) = prompt_input.plain_text.as_deref()
            && let Some(local_response) = self.handle_slash_command(&session_id, prompt_text)?
        {
            self.append_text_message(&session_id, "user", prompt_text)?;
            self.append_text_message(&session_id, "assistant", &local_response)?;
            self.send_text_update(writer, &session_id, "agent_message_chunk", &local_response)?;
            let result = json!({
                "stopReason": "end_turn",
                "userMessageId": message_id,
            });
            return write_jsonrpc_result(writer, id, result).map_err(|error| error.to_string());
        }

        let runtime = attach_python_plugin_runtime(
            &self.context.hermes_home(),
            hermes_core::ToolRuntime::new(&state.cwd)
                .with_hermes_home(self.context.hermes_home())
                .with_current_session_id(Some(session_id.clone())),
        )
        .map_err(|error| error.to_string())?;
        let enabled_toolsets = vec![String::from("hermes-acp")];
        let result = self.context.run_chat_turn_with_user_content(
            self.config,
            prompt_input.user_content,
            &runtime,
            Some(&enabled_toolsets),
            &state.overrides,
            Some(&session_id),
            Some(self.session_store),
        );
        let final_response = match result {
            Ok(turn) => turn.final_response,
            Err(error) => format!("Error: {error}"),
        };
        self.send_text_update(writer, &session_id, "agent_message_chunk", &final_response)?;

        let response = json!({
            "stopReason": "end_turn",
            "userMessageId": message_id,
        });
        write_jsonrpc_result(writer, id, response).map_err(|error| error.to_string())
    }

    fn handle_slash_command(
        &mut self,
        session_id: &str,
        text: &str,
    ) -> Result<Option<String>, String> {
        if !text.starts_with('/') {
            return Ok(None);
        }
        let parts = text.splitn(2, char::is_whitespace).collect::<Vec<_>>();
        let command = parts[0].trim_start_matches('/').trim().to_ascii_lowercase();
        let args = parts.get(1).map(|value| value.trim()).unwrap_or("");
        let response = match command.as_str() {
            "help" => Some(help_text()),
            "version" => Some(format!("Hermes Agent v{}", env!("CARGO_PKG_VERSION"))),
            "tools" => Some(self.tools_text()),
            "model" => Some(self.handle_model_command(session_id, args)?),
            _ => None,
        };
        Ok(response)
    }

    fn handle_model_command(&mut self, session_id: &str, args: &str) -> Result<String, String> {
        let mut state = self
            .load_session_state(session_id, None)?
            .ok_or_else(|| format!("Session '{}' not found", session_id))?;
        if args.is_empty() {
            let current = self.model_state_json(&state);
            let current_model = current
                .get("currentModelId")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return Ok(format!("Current model: {current_model}"));
        }
        apply_model_selection(&mut state, args);
        self.persist_runtime_metadata(session_id, &state)?;
        let current = self.model_state_json(&state);
        let current_model = current
            .get("currentModelId")
            .and_then(Value::as_str)
            .unwrap_or(args);
        self.sessions.insert(session_id.to_string(), state);
        Ok(format!("Model switched to: {current_model}"))
    }

    fn tools_text(&self) -> String {
        let disabled =
            (!self.config.config.memory.any_enabled()).then(|| vec![String::from("memory")]);
        let enabled = vec![String::from("hermes-acp")];
        let tools = get_tool_definitions(Some(&enabled), disabled.as_deref());
        if tools.is_empty() {
            return "No tools available.".to_string();
        }
        let mut lines = vec![format!("Available tools ({}):", tools.len())];
        for tool in tools {
            lines.push(format!("  {}: {}", tool.name, tool.description));
        }
        lines.join("\n")
    }

    fn send_available_commands_update<W: Write>(
        &self,
        writer: &mut W,
        session_id: &str,
    ) -> Result<(), String> {
        let update = json!({
            "sessionUpdate": "available_commands_update",
            "availableCommands": [
                {"name": "help", "description": "List available commands"},
                {"name": "model", "description": "Show or change the current model"},
                {"name": "tools", "description": "List available tools"},
                {"name": "version", "description": "Show Hermes version"}
            ]
        });
        write_session_update(writer, session_id, update).map_err(|error| error.to_string())
    }

    fn replay_history<W: Write>(&self, writer: &mut W, session_id: &str) -> Result<(), String> {
        let messages = self
            .session_store
            .get_messages(session_id)
            .map_err(|error| error.to_string())?;
        for message in messages {
            let role = message.role.as_str();
            if !matches!(role, "user" | "assistant") {
                continue;
            }
            let text = message_display_text(&message);
            if text.is_empty() {
                continue;
            }
            let update_kind = if role == "user" {
                "user_message_chunk"
            } else {
                "agent_message_chunk"
            };
            self.send_text_update(writer, session_id, update_kind, &text)?;
        }
        Ok(())
    }

    fn send_text_update<W: Write>(
        &self,
        writer: &mut W,
        session_id: &str,
        update_kind: &str,
        text: &str,
    ) -> Result<(), String> {
        let update = json!({
            "sessionUpdate": update_kind,
            "content": {
                "type": "text",
                "text": text,
            }
        });
        write_session_update(writer, session_id, update).map_err(|error| error.to_string())
    }

    fn append_text_message(&self, session_id: &str, role: &str, text: &str) -> Result<(), String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        self.session_store
            .append_message(
                session_id,
                &MessageAppend {
                    role: role.to_string(),
                    content: Some(Value::String(trimmed.to_string())),
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
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn load_session_state(
        &mut self,
        session_id: &str,
        cwd_override: Option<PathBuf>,
    ) -> Result<Option<AcpSessionState>, String> {
        if let Some(state) = self.sessions.get(session_id) {
            let mut state = state.clone();
            if let Some(cwd) = cwd_override {
                state.cwd = cwd;
                self.persist_runtime_metadata(session_id, &state)?;
                self.sessions.insert(session_id.to_string(), state.clone());
            }
            return Ok(Some(state));
        }
        let Some(record) = self
            .session_store
            .get_session(session_id)
            .map_err(|error| error.to_string())?
        else {
            return Ok(None);
        };
        if record.source != ACP_SESSION_SOURCE {
            return Ok(None);
        }
        let mut state = AcpSessionState::from_record(&record, cwd_override)?;
        if !state.cwd.is_absolute() {
            state.cwd = normalize_cwd(state.cwd.to_string_lossy().as_ref())?;
        }
        self.persist_runtime_metadata(session_id, &state)?;
        self.sessions.insert(session_id.to_string(), state.clone());
        Ok(Some(state))
    }

    fn create_or_update_persisted_session(
        &self,
        session_id: &str,
        state: &AcpSessionState,
        parent_session_id: Option<&str>,
    ) -> Result<(), String> {
        let runtime = self
            .context
            .resolve_model_runtime(self.config, &state.overrides)
            .ok();
        let model = runtime
            .as_ref()
            .map(|value| value.model.clone())
            .or_else(|| {
                state
                    .overrides
                    .model
                    .as_deref()
                    .and_then(non_empty_trimmed)
                    .or_else(|| self.config.configured_model_name())
            });
        let tool_runtime =
            hermes_core::ToolRuntime::new(&state.cwd).with_hermes_home(self.context.hermes_home());
        let system_prompt = self
            .context
            .render_system_prompt(&tool_runtime, &self.config.config.memory)
            .map_err(|error| error.to_string())?;
        self.session_store
            .create_session(&SessionCreate {
                id: session_id.to_string(),
                source: ACP_SESSION_SOURCE.to_string(),
                user_id: None,
                model: model.clone(),
                model_config: Some(self.session_model_config_json(state, runtime.as_ref())),
                system_prompt: Some(system_prompt),
                parent_session_id: parent_session_id.map(ToOwned::to_owned),
            })
            .map_err(|error| error.to_string())?;
        self.persist_runtime_metadata(session_id, state)
    }

    fn persist_runtime_metadata(
        &self,
        session_id: &str,
        state: &AcpSessionState,
    ) -> Result<(), String> {
        let runtime = self
            .context
            .resolve_model_runtime(self.config, &state.overrides)
            .ok();
        let configured_model_name = self.config.configured_model_name();
        let model = runtime
            .as_ref()
            .map(|value| value.model.as_str())
            .or_else(|| state.overrides.model.as_deref())
            .or(configured_model_name.as_deref());
        self.session_store
            .update_session_runtime(
                session_id,
                model,
                Some(&self.session_model_config_json(state, runtime.as_ref())),
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn session_model_config_json(
        &self,
        state: &AcpSessionState,
        runtime: Option<&hermes_core::ModelRuntimeConfig>,
    ) -> Value {
        json!({
            "provider": runtime
                .map(|value| value.provider.clone())
                .or_else(|| state.overrides.provider.clone()),
            "base_url": runtime
                .map(|value| value.base_url.clone())
                .or_else(|| state.overrides.base_url.clone()),
            "api_mode": runtime
                .map(|value| value.api_mode.clone())
                .or_else(|| state.overrides.api_mode.clone()),
            "cwd": state.cwd.display().to_string(),
        })
    }

    fn model_state_json(&self, state: &AcpSessionState) -> Value {
        let (model_id, description) = match self
            .context
            .resolve_model_runtime(self.config, &state.overrides)
        {
            Ok(runtime) => (
                runtime.model.clone(),
                Some(format!("Provider: {}", runtime.provider)),
            ),
            Err(_) => (
                state
                    .overrides
                    .model
                    .clone()
                    .or_else(|| self.config.configured_model_name())
                    .unwrap_or_default(),
                state
                    .overrides
                    .provider
                    .clone()
                    .or_else(|| self.config.configured_model_provider())
                    .map(|provider| format!("Provider: {}", provider)),
            ),
        };
        json!({
            "availableModels": [{
                "modelId": model_id,
                "name": model_id,
                "description": description,
            }],
            "currentModelId": model_id,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let detected = HermesContext::detect();
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let profile_override = detected.apply_profile_override(&raw_args)?;
    let context = match profile_override.hermes_home.clone() {
        Some(home) => detected.with_hermes_home_env(Some(home)),
        None => detected,
    };
    unsafe { std::env::set_var("HERMES_HOME", context.hermes_home()) };
    context.ensure_hermes_home()?;
    let env_report = context.load_hermes_dotenv(None)?;
    let config = context.load_config_document()?;
    let _logging = context.setup_logging(&config, LoggingMode::Cli)?;
    let session_store = context.open_session_store()?;
    emit_warnings(&env_report, &config);
    log::info!(
        target: "acp_adapter",
        "startup profile={} home={}",
        context.current_profile_name(),
        context.hermes_home().display()
    );
    let argv = std::iter::once(String::from("hermes-acp")).chain(profile_override.args);
    let cli = Cli::parse_from(argv);

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => run_server(&context, &config, &session_store)?,
        Command::Env => {
            println!("hermes_home={}", context.hermes_home().display());
            println!("env={}", context.env_path().display());
            println!("state_db={}", session_store.path().display());
            println!("active_profile={}", context.active_profile());
            println!("loaded_env_files={}", env_report.loaded_paths.len());
        }
        Command::Status => {
            println!("app=hermes-acp");
            println!("mode=rust-stdio-server");
            println!("current_profile={}", context.current_profile_name());
            println!("hermes_home={}", context.hermes_home().display());
            println!("config={}", config.path.display());
            println!("state_db={}", session_store.path().display());
            println!(
                "session_count={}",
                session_store.session_count().unwrap_or_default()
            );
            println!("loaded_env_files={}", env_report.loaded_paths.len());
            println!("logging_level={}", config.config.logging.level);
        }
    }

    Ok(())
}

fn run_server(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &SessionStore,
) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut server = AcpServer::new(context, config, session_store);
    server.run(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn help_text() -> String {
    [
        "Available commands:",
        "",
        "  /help     List available commands",
        "  /model    Show or change the current model",
        "  /tools    List available tools",
        "  /version  Show Hermes version",
    ]
    .join("\n")
}

fn mode_state_json(mode_id: &str) -> Value {
    let normalized = non_empty_trimmed(mode_id).unwrap_or_else(|| DEFAULT_MODE_ID.to_string());
    let mut modes = vec![json!({
        "id": DEFAULT_MODE_ID,
        "name": "Default",
        "description": "Default Hermes mode",
    })];
    if normalized != DEFAULT_MODE_ID {
        modes.push(json!({
            "id": normalized,
            "name": normalized,
            "description": format!("Session mode '{}'", normalized),
        }));
    }
    json!({
        "availableModes": modes,
        "currentModeId": normalized,
    })
}

fn apply_model_selection(state: &mut AcpSessionState, raw: &str) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    if let Some((provider, model)) = trimmed.split_once(':')
        && get_provider_profile(&normalize_provider_alias(provider)).is_some()
        && !model.trim().is_empty()
    {
        state.overrides.provider = Some(normalize_provider_alias(provider));
        state.overrides.model = Some(model.trim().to_string());
        state.overrides.base_url = None;
        state.overrides.api_mode = None;
        return;
    }
    state.overrides.model = Some(trimmed.to_string());
}

fn extract_prompt_input(prompt: &Value) -> Result<PromptInput, String> {
    let items = prompt
        .as_array()
        .ok_or_else(|| "prompt must be an array".to_string())?;
    let mut display_parts = Vec::new();
    let mut text_parts = Vec::new();
    let mut content_parts = Vec::new();
    for item in items {
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "text" => {
                if let Some(text) = item.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    let trimmed = text.trim().to_string();
                    display_parts.push(trimmed.clone());
                    text_parts.push(trimmed.clone());
                    content_parts.push(json!({
                        "type": "text",
                        "text": trimmed,
                    }));
                }
            }
            "resource_link" => {
                let title = item.get("title").and_then(Value::as_str);
                let uri = item.get("uri").and_then(Value::as_str);
                let name = item.get("name").and_then(Value::as_str);
                let label = title.or(name).or(uri).unwrap_or("resource");
                display_parts.push(format!("[Resource: {label}]"));
            }
            "resource" => {
                if let Some(resource) = item.get("resource").and_then(Value::as_object) {
                    if let Some(text) = resource.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        let trimmed = text.trim().to_string();
                        display_parts.push(trimmed.clone());
                        text_parts.push(trimmed);
                        continue;
                    }
                    let label = resource
                        .get("uri")
                        .and_then(Value::as_str)
                        .or_else(|| resource.get("mimeType").and_then(Value::as_str))
                        .unwrap_or("embedded resource");
                    display_parts.push(format!("[Resource: {label}]"));
                }
            }
            "image" => {
                let image_part = extract_image_prompt_part(item)?;
                if let Some(image_part) = image_part {
                    display_parts.push("[Image attachment]".to_string());
                    content_parts.push(image_part);
                }
            }
            "audio" => {
                return Err(
                    "Audio prompts are not supported by the Rust ACP runtime yet.".to_string(),
                );
            }
            other => display_parts.push(format!("[Unsupported content block: {other}]")),
        }
    }
    let user_content = if content_parts.is_empty() {
        Value::String(String::new())
    } else if content_parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
    {
        Value::String(text_parts.join("\n"))
    } else {
        Value::Array(content_parts)
    };
    let plain_text = (!text_parts.is_empty() && matches!(user_content, Value::String(_)))
        .then(|| text_parts.join("\n"));
    Ok(PromptInput {
        user_content,
        display_text: display_parts.join("\n"),
        plain_text,
    })
}

fn extract_image_prompt_part(item: &Value) -> Result<Option<Value>, String> {
    let data = item
        .get("data")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let uri = item
        .get("uri")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let mime_type = item
        .get("mimeType")
        .and_then(Value::as_str)
        .or_else(|| item.get("mime_type").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("image/png");

    let Some(url) = data
        .map(|value| {
            if value.starts_with("data:") {
                value.to_string()
            } else {
                format!("data:{mime_type};base64,{value}")
            }
        })
        .or_else(|| uri.map(ToOwned::to_owned))
    else {
        return Ok(None);
    };
    Ok(Some(json!({
        "type": "image_url",
        "image_url": {
            "url": url,
        }
    })))
}

fn has_meaningful_prompt_content(content: &Value) -> bool {
    match content {
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(parts) => parts.iter().any(|part| {
            let Some(kind) = part.get("type").and_then(Value::as_str) else {
                return false;
            };
            match kind {
                "text" => part
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty()),
                "image_url" => part
                    .get("image_url")
                    .and_then(Value::as_object)
                    .and_then(|image| image.get("url"))
                    .and_then(Value::as_str)
                    .is_some_and(|url| !url.trim().is_empty()),
                _ => false,
            }
        }),
        _ => false,
    }
}

fn message_display_text(message: &MessageRecord) -> String {
    let Some(content) = message.content.as_ref() else {
        return String::new();
    };
    match content {
        Value::String(text) => text.trim().to_string(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| match item.as_object() {
                Some(map) if map.get("type").and_then(Value::as_str) == Some("image_url") => {
                    Some("[Image attachment]".to_string())
                }
                Some(map) => map
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|text| text.trim().to_string()),
                None => item.as_str().map(|text| text.trim().to_string()),
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => map
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.trim().to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn message_append_from_record(message: &MessageRecord) -> MessageAppend {
    MessageAppend {
        role: message.role.clone(),
        content: message.content.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message.tool_calls.clone(),
        tool_name: message.tool_name.clone(),
        token_count: message.token_count,
        finish_reason: message.finish_reason.clone(),
        reasoning: message.reasoning.clone(),
        reasoning_content: message.reasoning_content.clone(),
        reasoning_details: message.reasoning_details.clone(),
        codex_reasoning_items: message.codex_reasoning_items.clone(),
        codex_message_items: message.codex_message_items.clone(),
    }
}

fn normalize_cwd(cwd: &str) -> Result<PathBuf, String> {
    let trimmed = non_empty_trimmed(cwd).ok_or_else(|| "cwd is required".to_string())?;
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err("cwd must be an absolute path".to_string());
    }
    Ok(path)
}

fn required_string(params: &Value, key: &str) -> Result<String, String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .and_then(non_empty_trimmed)
        .ok_or_else(|| format!("{key} is required"))
}

fn optional_string(params: &Value, key: &str) -> Result<Option<String>, String> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(non_empty_trimmed(value)),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn iso_timestamp(seconds: f64) -> Option<String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let secs = seconds.floor() as i64;
    let nanos = ((seconds - secs as f64) * 1_000_000_000_f64)
        .round()
        .clamp(0.0, 999_999_999_f64) as u32;
    match Utc.timestamp_opt(secs, nanos) {
        LocalResult::Single(value) => Some(value.to_rfc3339()),
        _ => None,
    }
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0)
}

fn write_jsonrpc_result<W: Write>(writer: &mut W, id: Value, result: Value) -> io::Result<()> {
    let message = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    write_json_line(writer, &message)
}

fn write_jsonrpc_error<W: Write>(
    writer: &mut W,
    id: Value,
    code: i64,
    message: &str,
    data: Option<Value>,
) -> io::Result<()> {
    let mut error = json!({
        "code": code,
        "message": message,
    });
    if let Some(data) = data
        && let Some(object) = error.as_object_mut()
    {
        object.insert("data".to_string(), data);
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": error,
    });
    write_json_line(writer, &payload)
}

fn write_session_update<W: Write>(
    writer: &mut W,
    session_id: &str,
    update: Value,
) -> io::Result<()> {
    let payload = json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        }
    });
    write_json_line(writer, &payload)
}

fn write_json_line<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn emit_warnings(env_report: &EnvLoadReport, config: &LoadedConfig) {
    for warning in &env_report.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "acp_adapter", "{warning}");
    }
    for warning in &config.warnings {
        eprintln!("warning: {warning}");
        log::warn!(target: "acp_adapter", "{warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn extract_prompt_input_supports_text_resources_and_images() {
        let prompt = json!([
            {"type": "text", "text": "hello"},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
            {"type": "resource_link", "uri": "file:///tmp/a.txt", "name": "a.txt"},
            {"type": "resource", "resource": {"text": "embedded", "uri": "mem://1"}}
        ]);
        let parsed = extract_prompt_input(&prompt).unwrap();
        assert!(parsed.display_text.contains("hello"));
        assert!(parsed.display_text.contains("[Image attachment]"));
        assert!(parsed.display_text.contains("[Resource: a.txt]"));
        assert!(parsed.display_text.contains("embedded"));
        assert!(parsed.plain_text.is_none());
        assert_eq!(
            parsed.user_content,
            json!([
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ])
        );
    }

    #[test]
    fn initialize_advertises_image_prompt_capability() {
        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(&home).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let server = AcpServer::new(&context, &config, &store);

        let initialize = server.handle_initialize();
        assert_eq!(
            initialize["agentCapabilities"]["promptCapabilities"]["image"],
            json!(true)
        );

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn message_display_text_uses_image_placeholder_for_multimodal_messages() {
        let message = MessageRecord {
            id: 1,
            session_id: "acp_test".to_string(),
            role: "user".to_string(),
            content: Some(json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ])),
            tool_call_id: None,
            tool_calls: None,
            tool_name: None,
            timestamp: 0.0,
            token_count: None,
            finish_reason: None,
            reasoning: None,
            reasoning_content: None,
            reasoning_details: None,
            codex_reasoning_items: None,
            codex_message_items: None,
        };
        assert_eq!(message_display_text(&message), "look\n[Image attachment]");
    }

    #[test]
    fn session_state_from_record_reads_cwd_and_runtime_fields() {
        let record = SessionRecord {
            id: "acp_test".to_string(),
            source: ACP_SESSION_SOURCE.to_string(),
            user_id: None,
            model: Some("gpt-test".to_string()),
            model_config: Some(json!({
                "provider": "openai",
                "base_url": "https://api.example.com/v1",
                "api_mode": "chat_completions",
                "cwd": "/tmp/project"
            })),
            system_prompt: None,
            parent_session_id: None,
            started_at: 0.0,
            ended_at: None,
            end_reason: None,
            message_count: 0,
            tool_call_count: 0,
            title: None,
            api_call_count: 0,
        };
        let state = AcpSessionState::from_record(&record, None).unwrap();
        assert_eq!(state.cwd, PathBuf::from("/tmp/project"));
        assert_eq!(state.overrides.model.as_deref(), Some("gpt-test"));
        assert_eq!(state.overrides.provider.as_deref(), Some("openai"));
        assert_eq!(
            state.overrides.base_url.as_deref(),
            Some("https://api.example.com/v1")
        );
    }

    #[test]
    fn slash_prompt_persists_messages_for_replay() {
        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(&home).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = AcpServer::new(&context, &config, &store);

        let created = server
            .handle_new_session(&json!({"cwd": home.display().to_string()}))
            .unwrap();
        let session_id = created
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();

        let mut writer = Vec::new();
        server
            .handle_prompt(
                json!(1),
                &json!({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "/version"}],
                }),
                &mut writer,
            )
            .unwrap();

        let messages = store.get_messages(&session_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(message_display_text(&messages[0]), "/version");
        assert_eq!(messages[1].role, "assistant");
        assert!(message_display_text(&messages[1]).contains("Hermes Agent v"));

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn image_prompt_runs_end_to_end_through_acp() {
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
            assert_eq!(content[0]["text"], json!("What is this?"));
            assert_eq!(
                content[1]["image_url"]["url"],
                json!("data:image/png;base64,aGVsbG8=")
            );

            let response = json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "ACP image prompt ok."
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

        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(&home).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            format!(
                "model:\n  default: test-model\n  provider: custom\n  base_url: http://{}\n  api_key: test-key\n  api_mode: chat_completions\n",
                addr
            ),
        )
        .unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = AcpServer::new(&context, &config, &store);

        let created = server
            .handle_new_session(&json!({"cwd": home.display().to_string()}))
            .unwrap();
        let session_id = created
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();

        let mut writer = Vec::new();
        server
            .handle_prompt(
                json!(1),
                &json!({
                    "sessionId": session_id,
                    "prompt": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}
                    ],
                }),
                &mut writer,
            )
            .unwrap();

        let messages = store.get_messages(&session_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].content.as_ref().unwrap(),
            &json!([
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ])
        );
        assert_eq!(
            message_display_text(&messages[0]),
            "What is this?\n[Image attachment]"
        );
        assert!(message_display_text(&messages[1]).contains("ACP image prompt ok."));
        join.join().unwrap();
        let _ = fs::remove_dir_all(&home);
    }
}
