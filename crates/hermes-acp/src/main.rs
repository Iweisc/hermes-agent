use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{LocalResult, TimeZone, Utc};
use clap::{Parser, Subcommand};
use hermes_core::{
    ClarifyRequest, EnvLoadReport, HermesContext, InteractiveTurnEvent, InteractiveTurnOptions,
    InteractiveTurnRequest, LoadedConfig, LoggingMode, MessageAppend, MessageRecord,
    ModelOverrides, SessionCreate, SessionRecord, SessionStore, StepUpdate, ToolProgressUpdate,
    ToolRuntime, attach_python_plugin_runtime, get_provider_profile, get_tool_definitions,
    normalize_provider_alias, spawn_chat_turn_with_events,
};
use serde_json::{Value, json};

const ACP_PROTOCOL_VERSION: i64 = 1;
const ACP_SESSION_SOURCE: &str = "rust-acp";
const ACP_PAGE_SIZE: usize = 50;
const DEFAULT_MODE_ID: &str = "default";
static ACP_TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(1);
static ACP_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct ActiveToolCall {
    id: String,
    args: Option<Value>,
}

#[derive(Debug)]
enum PendingClientRequest {
    Clarify {
        request: ClarifyRequest,
        response_tx: mpsc::Sender<Result<String, String>>,
    },
    Approval {
        response_tx: mpsc::Sender<Result<String, String>>,
    },
}

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
    input_rx: Option<mpsc::Receiver<Option<String>>>,
    pending_client_requests: HashMap<String, PendingClientRequest>,
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
            input_rx: None,
            pending_client_requests: HashMap::new(),
        }
    }

    fn run<R: BufRead + Send, W: Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
    ) -> io::Result<()> {
        let (tx, rx) = mpsc::channel::<Option<String>>();
        self.input_rx = Some(rx);
        let result = thread::scope(|scope| -> io::Result<()> {
            scope.spawn(|| {
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => {
                            let _ = tx.send(None);
                            break;
                        }
                        Ok(_) => {
                            let _ = tx.send(Some(line.clone()));
                        }
                        Err(_) => {
                            let _ = tx.send(None);
                            break;
                        }
                    }
                }
            });

            while let Some(line) = self.recv_input_line_blocking() {
                self.handle_input_line(&line, writer, false)?;
            }
            Ok(())
        });
        self.input_rx = None;
        self.pending_client_requests.clear();
        result
    }

    fn recv_input_line_blocking(&self) -> Option<String> {
        self.input_rx
            .as_ref()
            .and_then(|rx| rx.recv().ok())
            .flatten()
    }

    fn drain_input_lines<W: Write>(
        &mut self,
        writer: &mut W,
        prompt_active: bool,
    ) -> Result<(), String> {
        let Some(rx) = self.input_rx.as_ref() else {
            return Ok(());
        };
        let mut lines = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(Some(line)) => lines.push(line),
                Ok(None) | Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        for line in lines {
            self.handle_input_line(&line, writer, prompt_active)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn handle_input_line<W: Write>(
        &mut self,
        line: &str,
        writer: &mut W,
        prompt_active: bool,
    ) -> io::Result<()> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let message = match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => value,
            Err(error) => {
                return write_jsonrpc_error(
                    writer,
                    Value::Null,
                    -32700,
                    "Parse error",
                    Some(json!({"detail": error.to_string()})),
                );
            }
        };
        if !message.is_object() {
            return write_jsonrpc_error(
                writer,
                Value::Null,
                -32600,
                "Invalid Request",
                Some(json!({"detail": "JSON-RPC frame must be an object"})),
            );
        }
        self.handle_message(message, writer, prompt_active)
    }

    fn handle_message<W: Write>(
        &mut self,
        message: Value,
        writer: &mut W,
        prompt_active: bool,
    ) -> io::Result<()> {
        let id = message.get("id").cloned();
        if let Some(method) = message.get("method").and_then(Value::as_str) {
            let params = message.get("params").cloned().unwrap_or(Value::Null);
            if let Some(id) = id {
                if prompt_active {
                    return write_jsonrpc_error(
                        writer,
                        id,
                        -32000,
                        "Request cannot be processed while a prompt is active",
                        Some(json!({"method": method})),
                    );
                }
                if let Err(error) = self.handle_request(method, id.clone(), params, writer) {
                    return write_jsonrpc_error(
                        writer,
                        id,
                        -32603,
                        "Internal error",
                        Some(json!({"detail": error})),
                    );
                }
            } else if let Err(error) = self.handle_notification(method, params) {
                log::warn!(target: "acp_adapter", "notification {} failed: {}", method, error);
            }
            return Ok(());
        }

        if let Some(id) = id {
            self.resolve_client_response(id, &message);
            return Ok(());
        }

        write_jsonrpc_error(
            writer,
            Value::Null,
            -32600,
            "Invalid Request",
            Some(json!({"detail": "JSON-RPC frame must include method or id"})),
        )
    }

    fn resolve_client_response(&mut self, id: Value, message: &Value) {
        let Some(key) = jsonrpc_id_key(&id) else {
            return;
        };
        let Some(request) = self.pending_client_requests.remove(&key) else {
            return;
        };
        if let Some(error) = message.get("error") {
            let detail = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Client request failed");
            match request {
                PendingClientRequest::Clarify { response_tx, .. }
                | PendingClientRequest::Approval { response_tx } => {
                    let _ = response_tx.send(Err(detail.to_string()));
                }
            }
            return;
        }
        let result = message.get("result").cloned().unwrap_or(Value::Null);
        match request {
            PendingClientRequest::Clarify {
                request,
                response_tx,
            } => {
                let _ = response_tx.send(map_input_response_to_clarify(&result, &request));
            }
            PendingClientRequest::Approval { response_tx } => {
                let _ = response_tx.send(Ok(map_permission_response_to_approval(&result)));
            }
        }
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
                    "audio": true,
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
        self.emit_session_start_hook(&session_id, &state);
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
                self.emit_session_start_hook(&session_id, &state);
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
        self.emit_session_start_hook(&fork_id, &source_state);
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
        let session_state = match self.load_session_state(&session_id, None) {
            Ok(state) => state,
            Err(error) => {
                log::warn!(
                    target: "acp_adapter",
                    "failed to load ACP session state for finalize hook {}: {}",
                    session_id,
                    error
                );
                None
            }
        };
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
        if let Some(state) = session_state {
            self.emit_session_finalize_hook(&session_id, &state);
        }
        self.sessions.remove(&session_id);
        Ok(json!({}))
    }

    fn emit_session_start_hook(&self, session_id: &str, state: &AcpSessionState) {
        self.invoke_session_hook(
            session_id,
            state,
            "on_session_start",
            json!({
                "session_id": session_id,
                "model": self.resolve_session_model_name(state),
                "platform": "acp",
            }),
        );
    }

    fn emit_session_finalize_hook(&self, session_id: &str, state: &AcpSessionState) {
        self.invoke_session_hook(
            session_id,
            state,
            "on_session_finalize",
            json!({
                "session_id": session_id,
                "platform": "acp",
            }),
        );
    }

    fn invoke_session_hook(
        &self,
        session_id: &str,
        state: &AcpSessionState,
        hook_name: &str,
        payload: Value,
    ) {
        let runtime = attach_python_plugin_runtime(
            &self.context.hermes_home(),
            ToolRuntime::new(&state.cwd)
                .with_hermes_home(self.context.hermes_home())
                .with_platform("acp")
                .with_current_session_id(Some(session_id.to_string())),
        );
        match runtime {
            Ok(runtime) => {
                let _ = runtime.invoke_hook(hook_name, &payload);
            }
            Err(error) => {
                log::warn!(
                    target: "acp_adapter",
                    "failed to attach ACP plugin runtime for {} hook {}: {}",
                    hook_name,
                    session_id,
                    error
                );
            }
        }
    }

    fn resolve_session_model_name(&self, state: &AcpSessionState) -> String {
        self.context
            .resolve_model_runtime(self.config, &state.overrides)
            .ok()
            .map(|runtime| runtime.model)
            .or_else(|| state.overrides.model.clone())
            .or_else(|| self.config.configured_model_name())
            .unwrap_or_default()
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

        let enabled_toolsets = vec![String::from("hermes-acp")];
        let approval_timeout = approval_timeout_seconds(self.config);
        let clarify_timeout = clarify_timeout_seconds(self.config);
        let interactive_client = self.input_rx.is_some();
        let runtime = attach_python_plugin_runtime(
            &self.context.hermes_home(),
            ToolRuntime::new(&state.cwd)
                .with_hermes_home(self.context.hermes_home())
                .with_platform("acp")
                .with_current_session_id(Some(session_id.clone())),
        )
        .map_err(|error| error.to_string())?;
        let rx = spawn_chat_turn_with_events(
            self.context.clone(),
            self.config.clone(),
            prompt_input.user_content,
            runtime,
            enabled_toolsets.clone(),
            state.overrides.clone(),
            Some(session_id.clone()),
            InteractiveTurnOptions {
                enable_client_requests: interactive_client,
                clarify_timeout: Duration::from_secs(clarify_timeout),
                approval_timeout: Duration::from_secs(approval_timeout),
            },
        );

        let mut active_tool_calls = HashMap::<String, Vec<ActiveToolCall>>::new();
        let final_response = loop {
            self.drain_input_lines(writer, true)?;
            match rx.recv_timeout(Duration::from_millis(25)) {
                Ok(InteractiveTurnEvent::ToolProgress(update)) => {
                    if update.event_type == "tool.started" {
                        self.handle_tool_progress_update(
                            writer,
                            &session_id,
                            &mut active_tool_calls,
                            &update,
                        )?;
                    }
                }
                Ok(InteractiveTurnEvent::Step(update)) => {
                    self.handle_step_update(writer, &session_id, &mut active_tool_calls, &update)?;
                }
                Ok(InteractiveTurnEvent::ClarifyRequest(request)) => {
                    self.send_clarify_request(writer, &session_id, request)?;
                }
                Ok(InteractiveTurnEvent::ApprovalRequest(request)) => {
                    self.send_approval_request(writer, &session_id, request)?;
                }
                Ok(InteractiveTurnEvent::Final(result)) => match result {
                    Ok(turn) => break turn.final_response,
                    Err(error) => break format!("Error: {error}"),
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break String::from("Error: ACP worker channel closed unexpectedly.");
                }
            }
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

    fn handle_tool_progress_update<W: Write>(
        &self,
        writer: &mut W,
        session_id: &str,
        active_tool_calls: &mut HashMap<String, Vec<ActiveToolCall>>,
        update: &ToolProgressUpdate,
    ) -> Result<(), String> {
        let Some(tool_name) = update.function_name.as_deref().and_then(non_empty_trimmed) else {
            return Ok(());
        };
        let tool_call_id = make_tool_call_id();
        let function_args = update.function_args.clone();
        active_tool_calls
            .entry(tool_name.clone())
            .or_default()
            .push(ActiveToolCall {
                id: tool_call_id.clone(),
                args: function_args.clone(),
            });
        let payload = json!({
            "sessionUpdate": "tool_call",
            "tool_call_id": tool_call_id,
            "title": build_tool_title(&tool_name, function_args.as_ref(), update.preview.as_deref()),
            "kind": tool_kind(&tool_name),
            "content": build_tool_start_content(&tool_name, function_args.as_ref(), update.preview.as_deref()),
        });
        write_session_update(writer, session_id, payload).map_err(|error| error.to_string())
    }

    fn send_clarify_request<W: Write>(
        &mut self,
        writer: &mut W,
        session_id: &str,
        request: InteractiveTurnRequest<ClarifyRequest>,
    ) -> Result<(), String> {
        let (request, response_tx) = request.into_parts();
        let request_id = next_acp_request_id();
        let params = build_input_request_params(session_id, &request_id, &request);
        let Some(key) = jsonrpc_id_key(&Value::String(request_id.clone())) else {
            let _ = response_tx.send(Err(String::from("ACP client request id was invalid.")));
            return Ok(());
        };
        self.pending_client_requests.insert(
            key,
            PendingClientRequest::Clarify {
                request,
                response_tx,
            },
        );
        write_jsonrpc_request(
            writer,
            Value::String(request_id),
            "session/request_input",
            params,
        )
        .map_err(|error| error.to_string())
    }

    fn send_approval_request<W: Write>(
        &mut self,
        writer: &mut W,
        session_id: &str,
        request: InteractiveTurnRequest<hermes_core::ApprovalRequest>,
    ) -> Result<(), String> {
        let (request, response_tx) = request.into_parts();
        let request_id = next_acp_request_id();
        let params = build_permission_request_params(session_id, &request_id, &request);
        let Some(key) = jsonrpc_id_key(&Value::String(request_id.clone())) else {
            let _ = response_tx.send(Err(String::from("ACP client request id was invalid.")));
            return Ok(());
        };
        self.pending_client_requests
            .insert(key, PendingClientRequest::Approval { response_tx });
        write_jsonrpc_request(
            writer,
            Value::String(request_id),
            "session/request_permission",
            params,
        )
        .map_err(|error| error.to_string())
    }

    fn handle_step_update<W: Write>(
        &self,
        writer: &mut W,
        session_id: &str,
        active_tool_calls: &mut HashMap<String, Vec<ActiveToolCall>>,
        update: &StepUpdate,
    ) -> Result<(), String> {
        for tool in &update.prev_tools {
            let Some(active) = active_tool_calls
                .get_mut(&tool.name)
                .and_then(|calls| (!calls.is_empty()).then(|| calls.remove(0)))
            else {
                continue;
            };
            if active_tool_calls
                .get(&tool.name)
                .is_some_and(|calls| calls.is_empty())
            {
                active_tool_calls.remove(&tool.name);
            }
            let payload = json!({
                "sessionUpdate": "tool_call_update",
                "tool_call_id": active.id,
                "status": "completed",
                "kind": tool_kind(&tool.name),
                "content": build_tool_complete_content(
                    tool.result.as_deref(),
                    tool.arguments.as_deref(),
                    active.args.as_ref(),
                ),
                "raw_output": tool.result.clone(),
            });
            write_session_update(writer, session_id, payload).map_err(|error| error.to_string())?;
        }
        Ok(())
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
        let tool_runtime = hermes_core::ToolRuntime::new(&state.cwd)
            .with_hermes_home(self.context.hermes_home())
            .with_platform("acp");
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
    let mut reader = BufReader::new(stdin);
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
                let audio_part = extract_audio_prompt_part(item)?;
                if let Some(audio_part) = audio_part {
                    display_parts.push("[Audio attachment]".to_string());
                    content_parts.push(audio_part);
                }
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

fn extract_audio_prompt_part(item: &Value) -> Result<Option<Value>, String> {
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
        .filter(|value| !value.is_empty());

    let (audio_data, inferred_mime_type) = match data {
        Some(value) if !value.starts_with("data:") => (value.to_string(), mime_type),
        Some(value) => parse_audio_data_url(value)?,
        None => match uri {
            Some(value) if value.starts_with("data:") => parse_audio_data_url(value)?,
            Some(_) => {
                return Err(
                    "Audio prompts must include inline base64 data (either `data` or a data: URI)."
                        .to_string(),
                );
            }
            None => return Ok(None),
        },
    };
    let format = audio_prompt_format(mime_type.or(inferred_mime_type))?;
    Ok(Some(json!({
        "type": "input_audio",
        "input_audio": {
            "data": audio_data,
            "format": format,
        }
    })))
}

fn parse_audio_data_url(value: &str) -> Result<(String, Option<&str>), String> {
    let Some((header, encoded)) = value
        .strip_prefix("data:")
        .and_then(|raw| raw.split_once(','))
    else {
        return Err("Audio data URI is malformed.".to_string());
    };
    if !header
        .split(';')
        .any(|segment| segment.eq_ignore_ascii_case("base64"))
    {
        return Err("Audio data URI must be base64-encoded.".to_string());
    }
    let mime_type = header
        .split(';')
        .next()
        .map(str::trim)
        .filter(|segment| !segment.is_empty());
    if encoded.trim().is_empty() {
        return Err("Audio data URI contained no payload.".to_string());
    }
    Ok((encoded.trim().to_string(), mime_type))
}

fn audio_prompt_format(mime_type: Option<&str>) -> Result<&'static str, String> {
    let normalized = mime_type.unwrap_or("audio/wav").trim().to_ascii_lowercase();
    match normalized.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" | "audio/vnd.wave" => Ok("wav"),
        "audio/mpeg" | "audio/mp3" | "audio/mpeg3" | "audio/x-mpeg-3" => Ok("mp3"),
        other => Err(format!("Audio prompts must be WAV or MP3; got '{other}'.")),
    }
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
                "input_audio" => part
                    .get("input_audio")
                    .and_then(Value::as_object)
                    .and_then(|audio| audio.get("data"))
                    .and_then(Value::as_str)
                    .is_some_and(|data| !data.trim().is_empty()),
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
                Some(map) if map.get("type").and_then(Value::as_str) == Some("input_audio") => {
                    Some("[Audio attachment]".to_string())
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

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn make_tool_call_id() -> String {
    format!(
        "tc-{:x}",
        ACP_TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn next_acp_request_id() -> String {
    format!(
        "acp-req-{:x}",
        ACP_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn approval_timeout_seconds(_config: &LoadedConfig) -> u64 {
    60
}

fn clarify_timeout_seconds(_config: &LoadedConfig) -> u64 {
    120
}

fn build_input_request_params(
    session_id: &str,
    request_id: &str,
    request: &ClarifyRequest,
) -> Value {
    let choices = request
        .choices
        .as_ref()
        .map(|items| {
            items
                .iter()
                .enumerate()
                .map(|(index, choice)| {
                    json!({
                        "choiceId": format!("choice_{}", index + 1),
                        "label": choice,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "sessionId": session_id,
        "requestId": request_id,
        "title": "Clarify",
        "message": request.question,
        "choices": choices,
        "allowFreeText": true,
    })
}

fn build_permission_request_params(
    session_id: &str,
    request_id: &str,
    request: &hermes_core::ApprovalRequest,
) -> Value {
    let mut options = vec![
        json!({
            "optionId": "allow_once",
            "kind": "allow_once",
            "name": "Allow once",
        }),
        json!({
            "optionId": "deny",
            "kind": "reject_once",
            "name": "Deny",
        }),
    ];
    if request.allow_permanent {
        options.insert(
            1,
            json!({
                "optionId": "allow_always",
                "kind": "allow_always",
                "name": "Allow always",
            }),
        );
    }
    let description = truncate_chars(&request.description, 240);
    json!({
        "sessionId": session_id,
        "options": options,
        "toolCall": {
            "sessionUpdate": "tool_call",
            "toolCallId": request_id,
            "title": format!("approval: {}", truncate_chars(&request.command, 80)),
            "kind": "execute",
            "rawInput": {
                "command": request.command,
                "description": request.description,
                "patternKeys": request.pattern_keys,
            },
            "content": [{
                "type": "content",
                "content": {
                    "type": "text",
                    "text": format!("{description}\n\n{}", request.command),
                }
            }],
        },
    })
}

fn map_permission_response_to_approval(response: &Value) -> String {
    let outcome = response
        .get("outcome")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if outcome.get("outcome").and_then(Value::as_str) != Some("selected") {
        return String::from("deny");
    }
    match outcome.get("optionId").and_then(Value::as_str) {
        Some("allow_always") => String::from("always"),
        Some("allow_once") => String::from("once"),
        _ => String::from("deny"),
    }
}

fn map_input_response_to_clarify(
    response: &Value,
    request: &ClarifyRequest,
) -> Result<String, String> {
    let answer = response
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| String::from("ACP clarify response did not include non-empty text."))?;
    if let Some(choices) = request.choices.as_ref()
        && let Ok(index) = answer.parse::<usize>()
        && (1..=choices.len()).contains(&index)
    {
        return Ok(choices[index - 1].clone());
    }
    Ok(answer.to_string())
}

fn jsonrpc_id_key(id: &Value) -> Option<String> {
    match id {
        Value::String(text) => Some(format!("s:{text}")),
        Value::Number(number) => Some(format!("n:{number}")),
        _ => None,
    }
}

fn tool_kind(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" | "skill_view" | "skills_list" | "vision_analyze" | "browser_snapshot"
        | "browser_vision" | "browser_get_images" => "read",
        "write_file" | "patch" | "skill_manage" => "edit",
        "search_files" => "search",
        "web_search" | "web_extract" | "browser_navigate" => "fetch",
        "terminal" | "process" | "execute_code" | "delegate_task" | "image_generate"
        | "text_to_speech" | "browser_click" | "browser_type" | "browser_scroll"
        | "browser_press" | "browser_back" => "execute",
        _ => "other",
    }
}

fn build_tool_title(tool_name: &str, args: Option<&Value>, preview: Option<&str>) -> String {
    let preview = preview
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let args_object = args.and_then(Value::as_object);
    match tool_name {
        "terminal" => {
            let command = args_object
                .and_then(|args| args.get("command"))
                .and_then(Value::as_str)
                .or(preview.as_deref())
                .unwrap_or("terminal");
            format!("terminal: {}", truncate_chars(command, 80))
        }
        "read_file" => format!(
            "read: {}",
            args_object
                .and_then(|args| args.get("path"))
                .and_then(Value::as_str)
                .unwrap_or("?")
        ),
        "write_file" => format!(
            "write: {}",
            args_object
                .and_then(|args| args.get("path"))
                .and_then(Value::as_str)
                .unwrap_or("?")
        ),
        "patch" => format!(
            "patch: {}",
            args_object
                .and_then(|args| args.get("path"))
                .and_then(Value::as_str)
                .unwrap_or("apply patch")
        ),
        "search_files" => format!(
            "search: {}",
            args_object
                .and_then(|args| args.get("pattern"))
                .and_then(Value::as_str)
                .unwrap_or("?")
        ),
        "web_search" => format!(
            "web search: {}",
            args_object
                .and_then(|args| args.get("query"))
                .and_then(Value::as_str)
                .unwrap_or("?")
        ),
        _ => preview.unwrap_or_else(|| tool_name.to_string()),
    }
}

fn build_tool_start_content(tool_name: &str, args: Option<&Value>, preview: Option<&str>) -> Value {
    let text = preview
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            args.and_then(|value| serde_json::to_string_pretty(value).ok())
                .map(|text| truncate_chars(&text, 1_200))
        })
        .unwrap_or_else(|| tool_name.to_string());
    json!([{ "type": "text", "text": text }])
}

fn build_tool_complete_content(
    result: Option<&str>,
    function_args: Option<&str>,
    fallback_args: Option<&Value>,
) -> Value {
    let parsed_args = function_args
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .or_else(|| fallback_args.cloned());
    let summary = format_tool_result_summary(result, parsed_args.as_ref())
        .or_else(|| result.map(|value| truncate_chars(value, 4_000)))
        .unwrap_or_else(|| String::from("Tool completed."));
    json!([{ "type": "text", "text": summary }])
}

fn format_tool_result_summary(result: Option<&str>, args: Option<&Value>) -> Option<String> {
    let result = result?;
    let parsed = serde_json::from_str::<Value>(result).ok()?;
    let object = parsed.as_object()?;
    if let Some(error) = object.get("error").and_then(Value::as_str) {
        return Some(format!("Error: {error}"));
    }
    if let Some(output) = object.get("output").and_then(Value::as_str) {
        return Some(truncate_chars(output, 4_000));
    }
    if let Some(content) = object.get("content").and_then(Value::as_str) {
        let path = args
            .and_then(Value::as_object)
            .and_then(|value| value.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("file");
        return Some(format!("Read {path}\n\n{content}"));
    }
    if let Some(path) = object.get("path").and_then(Value::as_str)
        && let Some(bytes_written) = object.get("bytes_written").and_then(Value::as_u64)
    {
        return Some(format!("Wrote {bytes_written} bytes to {path}"));
    }
    if let Some(matches) = object.get("matches").and_then(Value::as_array) {
        return Some(format!("Search returned {} match(es).", matches.len()));
    }
    if let Some(success) = object.get("success").and_then(Value::as_bool)
        && success
    {
        return Some(String::from("Tool completed successfully."));
    }
    Some(truncate_chars(result, 4_000))
}

fn write_jsonrpc_result<W: Write>(writer: &mut W, id: Value, result: Value) -> io::Result<()> {
    let message = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    write_json_line(writer, &message)
}

fn write_jsonrpc_request<W: Write>(
    writer: &mut W,
    id: Value,
    method: &str,
    params: Value,
) -> io::Result<()> {
    let message = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
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
    #[cfg(unix)]
    use std::net::Shutdown;
    use std::net::TcpListener;
    #[cfg(unix)]
    use std::os::unix::net::UnixStream;
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
    fn extract_prompt_input_supports_inline_audio() {
        let prompt = json!([
            {"type": "text", "text": "Transcribe this"},
            {"type": "audio", "data": "aGVsbG8=", "mimeType": "audio/wav"}
        ]);
        let parsed = extract_prompt_input(&prompt).unwrap();
        assert_eq!(
            parsed.user_content,
            json!([
                {"type": "text", "text": "Transcribe this"},
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}
            ])
        );
        assert_eq!(parsed.display_text, "Transcribe this\n[Audio attachment]");
        assert!(parsed.plain_text.is_none());
    }

    #[test]
    fn initialize_advertises_image_and_audio_prompt_capabilities() {
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
        assert_eq!(
            initialize["agentCapabilities"]["promptCapabilities"]["audio"],
            json!(true)
        );

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn message_display_text_uses_multimodal_placeholders() {
        let message = MessageRecord {
            id: 1,
            session_id: "acp_test".to_string(),
            role: "user".to_string(),
            content: Some(json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "mp3"}}
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
        assert_eq!(
            message_display_text(&message),
            "look\n[Image attachment]\n[Audio attachment]"
        );
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
    fn close_session_emits_finalize_hook_for_plugins() {
        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        let plugin_dir = home.join("plugins").join("finalizer");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: finalizer\ndescription: ACP finalize observer\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
import os
from pathlib import Path


def register(ctx):
    def on_session_finalize(**kwargs):
        Path(os.environ["HERMES_HOME"]).joinpath("finalize.log").write_text(
            kwargs.get("session_id", ""),
            encoding="utf-8",
        )

    ctx.register_hook("on_session_finalize", on_session_finalize)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - finalizer\n",
        )
        .unwrap();

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

        server
            .handle_close_session(&json!({"sessionId": session_id.clone()}))
            .unwrap();

        let finalized = fs::read_to_string(home.join("finalize.log")).unwrap();
        assert_eq!(finalized, session_id);

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn acp_session_creation_paths_emit_start_hook_for_plugins() {
        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        let plugin_dir = home.join("plugins").join("starter");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: starter\ndescription: ACP start observer\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
import os
from pathlib import Path


def register(ctx):
    def on_session_start(**kwargs):
        path = Path(os.environ["HERMES_HOME"]).joinpath("start.log")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{kwargs.get('session_id', '')}|{kwargs.get('platform', '')}\n"
            )

    ctx.register_hook("on_session_start", on_session_start)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - starter\n",
        )
        .unwrap();

        let context = HermesContext::new(&home).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = AcpServer::new(&context, &config, &store);

        let created = server
            .handle_new_session(&json!({"cwd": home.display().to_string()}))
            .unwrap();
        let created_session_id = created
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();

        let resumed_session_id = format!("acp_resume_{:x}", unix_ts_nanos());
        server
            .handle_resume_session(&json!({
                "sessionId": resumed_session_id,
                "cwd": home.display().to_string()
            }))
            .unwrap();

        let forked = server
            .handle_fork_session(&json!({
                "sessionId": created_session_id,
                "cwd": home.display().to_string()
            }))
            .unwrap();
        let forked_session_id = forked
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap()
            .to_string();

        let lines = fs::read_to_string(home.join("start.log"))
            .unwrap()
            .lines()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        assert!(lines.contains(&format!("{created_session_id}|acp")));
        assert!(lines.contains(&format!("{resumed_session_id}|acp")));
        assert!(lines.contains(&format!("{forked_session_id}|acp")));

        let _ = fs::remove_dir_all(&home);
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

    #[test]
    fn prompt_emits_tool_call_start_and_completion_updates() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            for response in [
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{\"path\":\"notes.txt\",\"content\":\"hello from acp\"}"
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
                            "content": "Done writing."
                        }
                    }]
                })
                .to_string(),
            ] {
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
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
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
                    "prompt": [{"type": "text", "text": "Create notes.txt"}],
                }),
                &mut writer,
            )
            .unwrap();

        let lines = String::from_utf8(writer)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let updates = lines
            .iter()
            .filter(|line| line.get("method").and_then(Value::as_str) == Some("session/update"))
            .map(|line| line["params"]["update"].clone())
            .collect::<Vec<_>>();
        let start = updates
            .iter()
            .find(|update| update["sessionUpdate"] == "tool_call")
            .cloned()
            .expect("tool_call update");
        let complete = updates
            .iter()
            .find(|update| update["sessionUpdate"] == "tool_call_update")
            .cloned()
            .expect("tool_call_update update");
        assert_eq!(start["title"], json!("write: notes.txt"));
        assert_eq!(complete["status"], json!("completed"));
        assert_eq!(start["tool_call_id"], complete["tool_call_id"]);
        let completion_text = complete["content"][0]["text"].as_str().unwrap();
        assert!(completion_text.contains("Wrote 14 bytes to"));
        assert!(completion_text.contains("notes.txt"));

        let final_result = lines
            .iter()
            .find(|line| line.get("id") == Some(&json!(1)))
            .cloned()
            .expect("final result");
        assert_eq!(final_result["result"]["stopReason"], json!("end_turn"));
        join.join().unwrap();
        let _ = fs::remove_dir_all(&home);
    }

    #[cfg(unix)]
    #[test]
    fn prompt_requests_clarify_input_and_continues_after_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let final_text = "Clarify result received.";
        let join = thread::spawn(move || {
            for response in [
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_clarify_1",
                                "type": "function",
                                "function": {
                                    "name": "clarify",
                                    "arguments": serde_json::to_string(&json!({
                                        "question": "Pick a mode",
                                        "choices": ["safe", "fast"],
                                    })).unwrap()
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
                            "content": final_text
                        }
                    }]
                })
                .to_string(),
            ] {
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
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let home = std::env::temp_dir().join(format!("hermes-acp-clarify-{}", unix_ts_nanos()));
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

        let (server_stream, client_stream) = UnixStream::pair().unwrap();
        let server_thread = thread::spawn(move || {
            let mut reader = BufReader::new(server_stream.try_clone().unwrap());
            let mut writer = server_stream;
            let mut server = AcpServer::new(&context, &config, &store);
            server.run(&mut reader, &mut writer).unwrap();
        });

        let mut client_reader = BufReader::new(client_stream.try_clone().unwrap());
        let mut client_writer = client_stream;
        client_reader
            .get_ref()
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        for request in [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/resume",
                "params": {
                    "sessionId": "acp_clarify",
                    "cwd": home.display().to_string(),
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": "acp_clarify",
                    "prompt": [{"type": "text", "text": "Ask me to choose a mode."}],
                }
            }),
        ] {
            serde_json::to_writer(&mut client_writer, &request).unwrap();
            client_writer.write_all(b"\n").unwrap();
            client_writer.flush().unwrap();
        }

        let mut saw_clarify = false;
        let mut saw_final = false;
        let mut saw_final_update = false;
        loop {
            let mut line = String::new();
            if client_reader.read_line(&mut line).unwrap_or_default() == 0 {
                break;
            }
            let message: Value = serde_json::from_str(line.trim()).unwrap();
            if message.get("method").and_then(Value::as_str) == Some("session/request_input") {
                saw_clarify = true;
                assert_eq!(message["params"]["sessionId"], json!("acp_clarify"));
                assert_eq!(message["params"]["title"], json!("Clarify"));
                assert_eq!(message["params"]["message"], json!("Pick a mode"));
                assert_eq!(message["params"]["choices"][0]["label"], json!("safe"));
                assert_eq!(message["params"]["choices"][1]["label"], json!("fast"));
                let request_id = message.get("id").cloned().unwrap();
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "text": "2"
                    }
                });
                serde_json::to_writer(&mut client_writer, &response).unwrap();
                client_writer.write_all(b"\n").unwrap();
                client_writer.flush().unwrap();
                continue;
            }
            if message.get("method").and_then(Value::as_str) == Some("session/update") {
                let update = &message["params"]["update"];
                if update["sessionUpdate"] == "agent_message_chunk"
                    && update["content"]["text"] == json!(final_text)
                {
                    saw_final_update = true;
                }
                continue;
            }
            if message.get("id") == Some(&json!(3)) {
                assert_eq!(message["result"]["stopReason"], json!("end_turn"));
                saw_final = true;
                break;
            }
        }

        assert!(saw_clarify);
        assert!(saw_final_update);
        assert!(saw_final);

        let _ = client_writer.shutdown(Shutdown::Both);
        drop(client_writer);
        drop(client_reader);
        server_thread.join().unwrap();
        join.join().unwrap();
        let _ = fs::remove_dir_all(&home);
    }

    #[cfg(unix)]
    #[test]
    fn prompt_requests_permission_and_continues_after_allow_once() {
        let target_root = std::env::temp_dir().join(format!("hermes-acp-rm-{}", unix_ts_nanos()));
        fs::create_dir_all(target_root.join("nested")).unwrap();
        fs::write(target_root.join("nested").join("keep.txt"), "hello").unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let command = format!("rm -rf {}", target_root.display());
        let final_text = "Approval-gated command finished.";
        let join = thread::spawn(move || {
            for response in [
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": serde_json::to_string(&json!({
                                        "command": command,
                                    })).unwrap()
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
                            "content": final_text
                        }
                    }]
                })
                .to_string(),
            ] {
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
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                let _ = stream.write_all(http.as_bytes());
            }
        });

        let home = std::env::temp_dir().join(format!("hermes-acp-test-{}", unix_ts_nanos()));
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(&home).with_hermes_home_env(Some(home.clone()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            format!(
                "model:\n  default: test-model\n  provider: custom\n  base_url: http://{}\n  api_key: test-key\n  api_mode: chat_completions\napprovals:\n  mode: manual\n",
                addr
            ),
        )
        .unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();

        let (server_stream, client_stream) = UnixStream::pair().unwrap();
        let server_thread = thread::spawn(move || {
            let mut reader = BufReader::new(server_stream.try_clone().unwrap());
            let mut writer = server_stream;
            let mut server = AcpServer::new(&context, &config, &store);
            server.run(&mut reader, &mut writer).unwrap();
        });

        let mut client_reader = BufReader::new(client_stream.try_clone().unwrap());
        let mut client_writer = client_stream;
        client_reader
            .get_ref()
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        for request in [
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/resume",
                "params": {
                    "sessionId": "acp_perm",
                    "cwd": home.display().to_string(),
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": "acp_perm",
                    "prompt": [{"type": "text", "text": "Delete the temporary directory."}],
                }
            }),
        ] {
            serde_json::to_writer(&mut client_writer, &request).unwrap();
            client_writer.write_all(b"\n").unwrap();
            client_writer.flush().unwrap();
        }

        let mut saw_permission = false;
        let mut saw_final = false;
        let mut saw_final_update = false;
        loop {
            let mut line = String::new();
            if client_reader.read_line(&mut line).unwrap_or_default() == 0 {
                break;
            }
            let message: Value = serde_json::from_str(line.trim()).unwrap();
            if message.get("method").and_then(Value::as_str) == Some("session/request_permission") {
                saw_permission = true;
                assert_eq!(message["params"]["sessionId"], json!("acp_perm"));
                assert_eq!(message["params"]["toolCall"]["kind"], json!("execute"));
                let title = message["params"]["toolCall"]["title"].as_str().unwrap();
                assert!(title.contains("approval: rm -rf"));
                let request_id = message.get("id").cloned().unwrap();
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": "allow_once"
                        }
                    }
                });
                serde_json::to_writer(&mut client_writer, &response).unwrap();
                client_writer.write_all(b"\n").unwrap();
                client_writer.flush().unwrap();
                continue;
            }
            if message.get("method").and_then(Value::as_str) == Some("session/update") {
                let update = &message["params"]["update"];
                if update["sessionUpdate"] == "agent_message_chunk"
                    && update["content"]["text"] == json!(final_text)
                {
                    saw_final_update = true;
                }
                continue;
            }
            if message.get("id") == Some(&json!(3)) {
                assert_eq!(message["result"]["stopReason"], json!("end_turn"));
                saw_final = true;
                break;
            }
        }

        assert!(saw_permission);
        assert!(saw_final_update);
        assert!(saw_final);
        assert!(!target_root.exists());

        let _ = client_writer.shutdown(Shutdown::Both);
        drop(client_writer);
        drop(client_reader);
        server_thread.join().unwrap();
        join.join().unwrap();
        let _ = fs::remove_dir_all(&home);
    }
}
