use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
#[cfg(test)]
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use hermes_core::{
    GatewaySessionPoll, GatewayTurnSession, HermesContext, LoadedConfig, MessageRecord,
    ModelOverrides, SessionCreate, SessionRecord, SessionStore, ToolRuntime,
    build_skill_invocation_message, parse_reasoning_effort, shell_command_block_reason,
    spawn_chat_turn_with_events,
};
use regex::Regex;
use serde_json::{Value, json};
use serde_yaml::{Mapping, Value as YamlValue};

use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::python_bridge::{project_root, resolve_repo_python};

#[derive(Debug, Clone)]
struct NativeGatewaySessionState {
    cwd: PathBuf,
    cols: u16,
    pending_steer: Option<String>,
    overrides: ModelOverrides,
}

struct NativeGatewayServer<'a> {
    context: &'a HermesContext,
    config: &'a LoadedConfig,
    session_store: &'a SessionStore,
    sessions: HashMap<String, NativeGatewaySessionState>,
    slash_workers: HashMap<String, SlashWorker>,
    input_rx: Option<mpsc::Receiver<Option<String>>>,
    input_closed: bool,
    active_turn: Option<ActiveTurn>,
}

struct ActiveTurn {
    session_id: String,
    runtime: ToolRuntime,
    turn: GatewayTurnSession,
}

struct SlashWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
    next_request_id: u64,
}

#[cfg(test)]
static SLASH_WORKER_PYTHON_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

const SHELL_EXEC_TIMEOUT_SECS: u64 = 30;
const SHELL_EXEC_STDOUT_LIMIT: usize = 4_000;
const SHELL_EXEC_STDERR_LIMIT: usize = 2_000;

impl<'a> NativeGatewayServer<'a> {
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
            slash_workers: HashMap::new(),
            input_rx: None,
            input_closed: false,
            active_turn: None,
        }
    }

    fn run<R: BufRead + Send, W: Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
    ) -> io::Result<()> {
        let (tx, rx) = mpsc::channel::<Option<String>>();
        self.input_rx = Some(rx);
        self.input_closed = false;
        write_jsonrpc_event(writer, json!({"type": "gateway.ready", "payload": {}}))?;
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
        self.input_closed = false;
        self.active_turn = None;
        result
    }

    fn recv_input_line_blocking(&mut self) -> Option<String> {
        if self.input_closed {
            return None;
        }
        match self.input_rx.as_ref().and_then(|rx| rx.recv().ok()) {
            Some(Some(line)) => Some(line),
            Some(None) | None => {
                self.input_closed = true;
                None
            }
        }
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
                Ok(None) => {
                    self.input_closed = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
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
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return write_jsonrpc_error(
                writer,
                id,
                -32600,
                "Invalid Request",
                Some(json!({"detail": "JSON-RPC frame must include method"})),
            );
        };
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        self.handle_request(method, id, params, writer, prompt_active)
    }

    fn handle_request<W: Write>(
        &mut self,
        method: &str,
        id: Value,
        params: Value,
        writer: &mut W,
        prompt_active: bool,
    ) -> io::Result<()> {
        if prompt_active
            && !matches!(
                method,
                "clarify.respond" | "approval.respond" | "session.interrupt" | "session.steer"
            )
        {
            return write_jsonrpc_error(
                writer,
                id,
                -32000,
                "Request cannot be processed while a prompt is active",
                Some(json!({"method": method})),
            );
        }

        let result = match method {
            "setup.status" => Ok(json!({"provider_configured": true})),
            "session.create" => self.handle_session_create(params),
            "session.list" => self.handle_session_list(),
            "session.most_recent" => self.handle_session_most_recent(),
            "session.resume" => self.handle_session_resume(params),
            "session.branch" => self.handle_session_branch(params),
            "session.steer" => self.handle_session_steer(params),
            "session.undo" => self.handle_session_undo(params),
            "session.usage" => self.handle_session_usage(params),
            "session.delete" => self.handle_session_delete(params),
            "session.title" => self.handle_session_title(params),
            "session.status" => self.handle_session_status(params),
            "session.save" => self.handle_session_save(params),
            "session.close" => self.handle_session_close(params),
            "terminal.resize" => self.handle_terminal_resize(params),
            "input.detect_drop" => Ok(json!({
                "matched": false,
                "text": params.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            "commands.catalog" => self.handle_commands_catalog(),
            "config.get" => self.handle_config_get(params),
            "config.set" => self.handle_config_set(params),
            "complete.path" => self.handle_complete_path(params),
            "complete.slash" => self.handle_complete_slash(params),
            "shell.exec" => self.handle_shell_exec(params),
            "command.dispatch" => self.handle_command_dispatch(params),
            "slash.exec" => self.handle_slash_exec(params),
            "prompt.submit" => {
                return match self.handle_prompt_submit(writer, params) {
                    Ok(value) => write_jsonrpc_result(writer, id, value),
                    Err((code, message, data)) => {
                        write_jsonrpc_error(writer, id, code, &message, data)
                    }
                };
            }
            "clarify.respond" => self.handle_clarify_respond(params),
            "approval.respond" => self.handle_approval_respond(params),
            "session.interrupt" => self.handle_session_interrupt(params),
            _ => Err((
                -32601,
                String::from("Method not found"),
                Some(json!({"method": method})),
            )),
        };

        match result {
            Ok(value) => write_jsonrpc_result(writer, id, value),
            Err((code, message, data)) => write_jsonrpc_error(writer, id, code, &message, data),
        }
    }

    fn handle_session_create(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = next_gateway_session_id();
        self.session_store
            .create_session(&SessionCreate {
                id: session_id.clone(),
                source: String::from("rust-gateway"),
                user_id: None,
                model: self.config.configured_model_name(),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .map_err(|error| (-32603, error.to_string(), None))?;
        self.sessions.insert(
            session_id.clone(),
            NativeGatewaySessionState {
                cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                cols: parse_terminal_cols(params.get("cols"))?,
                pending_steer: None,
                overrides: ModelOverrides::default(),
            },
        );
        Ok(json!({
            "session_id": session_id,
            "info": session_info_payload(self.config),
        }))
    }

    fn handle_session_list(&self) -> Result<Value, (i64, String, Option<Value>)> {
        let sessions = self
            .session_store
            .list_sessions(50, 0)
            .map_err(|error| (-32603, error.to_string(), None))?
            .into_iter()
            .map(|session| {
                json!({
                    "id": session.id,
                    "title": session.title.unwrap_or_else(|| String::from("(untitled)")),
                    "preview": session.preview,
                    "message_count": session.message_count,
                    "started_at": session.started_at,
                    "source": session.source,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({ "sessions": sessions }))
    }

    fn handle_session_most_recent(&self) -> Result<Value, (i64, String, Option<Value>)> {
        let recent = self
            .session_store
            .list_sessions(200, 0)
            .map_err(|error| (-32603, error.to_string(), None))?
            .into_iter()
            .find(|session| !session.source.eq_ignore_ascii_case("tool"));
        Ok(match recent {
            Some(session) => json!({
                "session_id": session.id,
                "source": session.source,
                "started_at": session.started_at,
                "title": session.title,
            }),
            None => json!({
                "session_id": Value::Null,
            }),
        })
    }

    fn handle_session_resume(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let resolved = self
            .session_store
            .resolve_session_id(&session_id)
            .map_err(|error| (-32603, error.to_string(), None))?
            .ok_or_else(|| (-32004, format!("Session '{}' not found", session_id), None))?;
        let record = self
            .session_store
            .get_session(&resolved)
            .map_err(|error| (-32603, error.to_string(), None))?
            .ok_or_else(|| (-32004, format!("Session '{}' not found", resolved), None))?;
        let messages = self
            .session_store
            .get_messages(&resolved)
            .map_err(|error| (-32603, error.to_string(), None))?;
        self.sessions.insert(
            resolved.clone(),
            NativeGatewaySessionState {
                cwd: cwd_from_record(&record),
                cols: 80,
                pending_steer: None,
                overrides: overrides_from_record(&record),
            },
        );
        Ok(json!({
            "session_id": resolved,
            "messages": transcript_messages(&messages),
            "message_count": messages.len(),
            "info": session_info_payload(self.config),
        }))
    }

    fn handle_session_delete(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)
            .map_err(|_| (4006, String::from("session_id required"), None))?;
        if self.sessions.contains_key(&session_id)
            || self
                .active_turn
                .as_ref()
                .is_some_and(|turn| turn.session_id == session_id)
        {
            return Err((4023, String::from("cannot delete an active session"), None));
        }
        let deleted = self
            .session_store
            .delete_session(&session_id)
            .map_err(|error| (5036, format!("delete failed: {error}"), None))?;
        if !deleted {
            return Err((4007, String::from("session not found"), None));
        }
        Ok(json!({ "deleted": session_id }))
    }

    fn handle_session_undo(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        if !self.sessions.contains_key(&session_id) {
            return Err((4007, String::from("session not found"), None));
        }
        if self
            .active_turn
            .as_ref()
            .is_some_and(|turn| turn.session_id == session_id)
        {
            return Err((
                4009,
                String::from("session busy — /interrupt the current turn before /undo"),
                None,
            ));
        }
        let removed = self
            .session_store
            .trim_last_exchange(&session_id)
            .map_err(|error| (5007, error.to_string(), None))?;
        Ok(json!({ "removed": removed }))
    }

    fn handle_session_branch(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let state = self
            .sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let record = self
            .session_store
            .get_session(&session_id)
            .map_err(|error| (5008, format!("branch failed: {error}"), None))?
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let messages = self
            .session_store
            .get_messages(&session_id)
            .map_err(|error| (5008, format!("branch failed: {error}"), None))?;
        if messages.is_empty() {
            return Err((
                4008,
                String::from("nothing to branch — send a message first"),
                None,
            ));
        }

        let new_session_id = next_gateway_session_id();
        let title = resolved_branch_title(self.session_store, &record, &params)
            .map_err(|error| (5008, format!("branch failed: {error}"), None))?;
        self.session_store
            .create_session(&SessionCreate {
                id: new_session_id.clone(),
                source: record.source.clone(),
                user_id: record.user_id.clone(),
                model: state
                    .overrides
                    .model
                    .clone()
                    .or_else(|| record.model.clone()),
                model_config: record.model_config.clone(),
                system_prompt: record.system_prompt.clone(),
                parent_session_id: Some(session_id.clone()),
            })
            .map_err(|error| (5008, format!("branch failed: {error}"), None))?;
        for message in &messages {
            self.session_store
                .append_message(&new_session_id, &branch_message_append(message))
                .map_err(|error| (5008, format!("branch failed: {error}"), None))?;
        }
        self.session_store
            .set_session_title(&new_session_id, &title)
            .map_err(|error| (5008, format!("branch failed: {error}"), None))?;
        let mut branched_state = state;
        branched_state.pending_steer = None;
        self.sessions.insert(new_session_id.clone(), branched_state);
        Ok(json!({
            "session_id": new_session_id,
            "title": title,
            "parent": session_id,
        }))
    }

    fn handle_session_steer(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let text = params
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (4002, String::from("text is required"), None))?;
        if let Some(active) = self.active_turn.as_ref()
            && active.session_id == session_id
        {
            let accepted = active.runtime.steer(text);
            return Ok(json!({
                "status": if accepted { "queued" } else { "rejected" },
                "text": text,
            }));
        }
        let state = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        if let Some(existing) = state.pending_steer.as_mut() {
            existing.push('\n');
            existing.push_str(text);
        } else {
            state.pending_steer = Some(text.to_string());
        }
        Ok(json!({
            "status": "queued",
            "text": text,
        }))
    }

    fn handle_session_usage(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let state = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let usage = self
            .session_store
            .get_session_usage(&session_id)
            .map_err(|error| (5007, error.to_string(), None))?
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let model = state
            .overrides
            .model
            .clone()
            .or(usage.model.clone())
            .unwrap_or_default();
        let total = usage.input_tokens + usage.output_tokens;
        let (cost_status, cost_usd) = if let Some(actual) = usage.actual_cost_usd {
            (Some(String::from("exact")), Some(actual))
        } else {
            (usage.cost_status.clone(), usage.estimated_cost_usd)
        };
        Ok(json!({
            "model": model,
            "calls": usage.api_call_count,
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": total,
            "cache_read": usage.cache_read_tokens,
            "cache_write": usage.cache_write_tokens,
            "cost_status": cost_status,
            "cost_usd": cost_usd,
        }))
    }

    fn handle_session_title(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        if !self.sessions.contains_key(&session_id) {
            return Err((4007, String::from("session not found"), None));
        }
        if !params
            .as_object()
            .is_some_and(|obj| obj.contains_key("title"))
        {
            let title = self
                .session_store
                .get_session_title(&session_id)
                .map_err(|error| (5007, error.to_string(), None))?
                .unwrap_or_default();
            return Ok(json!({
                "title": title,
                "session_key": session_id,
            }));
        }
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (4021, String::from("title required"), None))?;
        match self.session_store.set_session_title(&session_id, title) {
            Ok(true) => Ok(json!({"pending": false, "title": title})),
            Ok(false) => {
                let existing = self
                    .session_store
                    .get_session_title(&session_id)
                    .map_err(|error| (5007, error.to_string(), None))?
                    .unwrap_or_else(|| title.to_string());
                Ok(json!({"pending": false, "title": existing}))
            }
            Err(error) => Err((4022, error.to_string(), None)),
        }
    }

    fn handle_session_status(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let state = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let session = self
            .session_store
            .get_session(&session_id)
            .map_err(|error| (5007, error.to_string(), None))?
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let last_activity = self
            .session_store
            .get_messages(&session_id)
            .map_err(|error| (5007, error.to_string(), None))?
            .last()
            .map(|message| message.timestamp)
            .unwrap_or(session.started_at);
        let running = self
            .active_turn
            .as_ref()
            .is_some_and(|turn| turn.session_id == session_id);
        let provider = state
            .overrides
            .provider
            .clone()
            .unwrap_or_else(|| String::from("unknown"));
        let model = state
            .overrides
            .model
            .clone()
            .or_else(|| session.model.clone())
            .unwrap_or_else(|| String::from("(unknown)"));
        let mut lines = vec![
            String::from("Hermes TUI Status"),
            String::new(),
            format!("Session ID: {session_id}"),
            format!("Path: {}", self.context.display_hermes_home()),
        ];
        if let Some(title) = session
            .title
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            lines.push(format!("Title: {title}"));
        }
        lines.extend([
            format!("Model: {model} ({provider})"),
            format!("Created: {}", format_local_timestamp(session.started_at)),
            format!("Last Activity: {}", format_local_timestamp(last_activity)),
            format!("Messages: {}", session.message_count),
            format!("Agent Running: {}", if running { "Yes" } else { "No" }),
        ]);
        Ok(json!({ "output": lines.join("\n") }))
    }

    fn handle_session_save(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        if !self.sessions.contains_key(&session_id) {
            return Err((4007, String::from("session not found"), None));
        }
        let session = self
            .session_store
            .get_session(&session_id)
            .map_err(|error| (5011, error.to_string(), None))?
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        let messages = self
            .session_store
            .get_messages(&session_id)
            .map_err(|error| (5011, error.to_string(), None))?;
        let filename = std::env::current_dir()
            .map_err(|error| (5011, error.to_string(), None))?
            .join(format!(
                "hermes_conversation_{}.json",
                Local::now().format("%Y%m%d_%H%M%S")
            ));
        let payload = json!({
            "model": session.model.unwrap_or_default(),
            "messages": transcript_messages(&messages),
        });
        fs::write(
            &filename,
            serde_json::to_vec_pretty(&payload).map_err(|error| (5011, error.to_string(), None))?,
        )
        .map_err(|error| (5011, error.to_string(), None))?;
        Ok(json!({ "file": filename }))
    }

    fn handle_session_close(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let existed = self.sessions.remove(&session_id).is_some();
        self.slash_workers.remove(&session_id);
        if self
            .active_turn
            .as_ref()
            .is_some_and(|turn| turn.session_id == session_id)
        {
            if let Some(active) = self.active_turn.as_mut() {
                active.turn.cancel_pending_requests("closed");
            }
            self.active_turn = None;
        }
        Ok(json!({ "ok": existed, "closed": existed }))
    }

    fn handle_terminal_resize(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let cols = parse_terminal_cols(params.get("cols"))?;
        let state = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| (4007, String::from("session not found"), None))?;
        state.cols = cols;
        Ok(json!({ "cols": cols }))
    }

    fn handle_prompt_submit<W: Write>(
        &mut self,
        writer: &mut W,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        if self.active_turn.is_some() {
            return Err((-32000, String::from("session busy"), None));
        }
        let session_id = required_session_id(&params)?;
        let text = params
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (-32602, String::from("text is required"), None))?
            .to_string();
        let mut state = self
            .sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| (-32004, String::from("session not found"), None))?;
        let loaded = self
            .context
            .load_config_document()
            .map_err(|error| (-32603, error.to_string(), None))?;
        let runtime = ToolRuntime::new(&state.cwd)
            .with_hermes_home(self.context.hermes_home())
            .with_current_session_id(Some(session_id.clone()));
        if let Some(pending_steer) = state.pending_steer.take() {
            let _ = runtime.steer(&pending_steer);
            if let Some(saved_state) = self.sessions.get_mut(&session_id) {
                saved_state.pending_steer = None;
            }
        }
        let active_runtime = runtime.clone();
        let rx = spawn_chat_turn_with_events(
            self.context.clone(),
            loaded.clone(),
            Value::String(text),
            runtime,
            loaded.config.toolsets.clone(),
            state.overrides.clone(),
            Some(session_id.clone()),
            hermes_core::InteractiveTurnOptions {
                enable_client_requests: true,
                ..hermes_core::InteractiveTurnOptions::default()
            },
        );
        self.active_turn = Some(ActiveTurn {
            session_id: session_id.clone(),
            runtime: active_runtime,
            turn: GatewayTurnSession::new(rx),
        });
        write_jsonrpc_event(
            writer,
            json!({"type": "message.start", "session_id": session_id}),
        )
        .map_err(|error| (-32603, error.to_string(), None))?;
        let final_result = loop {
            self.drain_input_lines(writer, true)
                .map_err(|error| (-32603, error, None))?;
            let Some(active) = self.active_turn.as_mut() else {
                break Err(String::from("turn interrupted"));
            };
            match active
                .turn
                .poll_next()
                .map_err(|error| (-32603, error, None))?
            {
                GatewaySessionPoll::Events { events, .. } => {
                    for event in events {
                        let value = match event.payload {
                            Some(payload) => {
                                json!({"type": event.event_type, "session_id": session_id, "payload": payload})
                            }
                            None => json!({"type": event.event_type, "session_id": session_id}),
                        };
                        write_jsonrpc_event(writer, value)
                            .map_err(|error| (-32603, error.to_string(), None))?;
                    }
                }
                GatewaySessionPoll::Final { result, events } => {
                    for event in events {
                        let value = match event.payload {
                            Some(payload) => {
                                json!({"type": event.event_type, "session_id": session_id, "payload": payload})
                            }
                            None => json!({"type": event.event_type, "session_id": session_id}),
                        };
                        write_jsonrpc_event(writer, value)
                            .map_err(|error| (-32603, error.to_string(), None))?;
                    }
                    break result;
                }
            }
        };
        self.active_turn = None;
        match final_result {
            Ok(_) => Ok(json!({"ok": true})),
            Err(error) => {
                write_jsonrpc_event(
                    writer,
                    json!({"type": "error", "session_id": session_id, "payload": {"message": error}}),
                )
                .map_err(|write_error| (-32603, write_error.to_string(), None))?;
                Ok(json!({"ok": false}))
            }
        }
    }

    fn handle_clarify_respond(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let request_id = params
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (-32602, String::from("request_id is required"), None))?;
        let answer = params
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let active = self
            .active_turn
            .as_mut()
            .ok_or_else(|| (-32000, String::from("no active turn"), None))?;
        active
            .turn
            .resolve_clarify(request_id, Ok(answer))
            .map_err(|error| (-32000, error, None))?;
        Ok(json!({"ok": true}))
    }

    fn handle_approval_respond(
        &mut self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let choice = params
            .get("choice")
            .and_then(Value::as_str)
            .unwrap_or("deny");
        let mapped = match choice {
            "session" => "session",
            "always" => "always",
            "once" => "once",
            _ => "deny",
        };
        let active = self
            .active_turn
            .as_mut()
            .ok_or_else(|| (-32000, String::from("no active turn"), None))?;
        active
            .turn
            .resolve_approval(Ok(mapped.to_string()))
            .map_err(|error| (-32000, error, None))?;
        Ok(json!({"ok": true}))
    }

    fn handle_session_interrupt(
        &mut self,
        _params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        if let Some(active) = self.active_turn.as_mut() {
            active.turn.cancel_pending_requests("interrupted");
        }
        Ok(json!({"ok": true}))
    }

    fn handle_complete_path(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let word = params
            .get("word")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Ok(json!({
            "items": complete_path_items(word, &cwd),
        }))
    }

    fn handle_complete_slash(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let text = params
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (items, replace_from) =
            complete_slash_items(text, &self.config.raw).map_err(|error| (-32603, error, None))?;
        Ok(json!({
            "items": items,
            "replace_from": replace_from,
        }))
    }

    fn handle_shell_exec(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let command = params
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (4004, String::from("empty command"), None))?;
        if let Some(message) = shell_command_block_reason(command) {
            return Err((4005, message, None));
        }
        let cwd = std::env::current_dir().map_err(|error| (5003, error.to_string(), None))?;
        run_shell_exec(command, &cwd)
    }

    fn handle_command_dispatch(
        &self,
        params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let raw_name = params
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (4004, String::from("name required"), None))?;
        let name = resolve_gateway_command_name(raw_name, &self.config.raw)
            .map_err(|error| (5003, error, None))?;
        if name != raw_name.trim_start_matches('/') {
            return Ok(json!({"type": "alias", "target": name}));
        }
        let arg = params
            .get("arg")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();

        if let Some(result) = dispatch_quick_command(&self.config.raw, &name, &arg)? {
            return Ok(result);
        }

        let session_id = params
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some((skill_name, message)) =
            build_skill_invocation_message(&self.context.hermes_home(), &name, &arg, session_id)
                .map_err(|error| (5003, error, None))?
        {
            return Ok(json!({
                "type": "skill",
                "name": skill_name,
                "message": message,
            }));
        }

        if matches!(name.as_str(), "snapshot" | "snap") {
            let subcommand = arg
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            if matches!(subcommand.as_str(), "restore" | "rewind") {
                return Ok(json!({
                    "type": "exec",
                    "output": "/snapshot restore is blocked in the TUI because it changes config/state on disk while the live agent has cached settings. Run it in the classic CLI, then restart the TUI."
                }));
            }
        }

        Err((
            4018,
            format!("not a quick/plugin/skill command: {name}"),
            None,
        ))
    }

    fn handle_slash_exec(&mut self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = required_session_id(&params)?;
        let command = params
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (-32602, String::from("command is required"), None))?;
        let (command_base, command_arg) = split_slash_command(command);
        if pending_input_commands().contains(&command_base.as_str()) {
            return Err((
                4018,
                format!("pending-input command: use command.dispatch for /{command_base}"),
                None,
            ));
        }
        if worker_blocked_commands().contains(&command_base.as_str()) {
            let subcommand = command_arg
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            if matches!(subcommand.as_str(), "restore" | "rewind") {
                return Err((
                    4018,
                    String::from(
                        "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore",
                    ),
                    None,
                ));
            }
        }
        let state = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| (-32004, String::from("session not found"), None))?;
        let model = state
            .overrides
            .model
            .clone()
            .or_else(|| self.config.configured_model_name())
            .unwrap_or_default();
        if !self.slash_workers.contains_key(&session_id) {
            let worker = SlashWorker::start(&session_id, &model)
                .map_err(|error| (5030, format!("slash worker start failed: {error}"), None))?;
            self.slash_workers.insert(session_id.clone(), worker);
        }
        let worker = self.slash_workers.get_mut(&session_id).ok_or_else(|| {
            (
                5030,
                String::from("slash worker missing after startup"),
                None,
            )
        })?;
        match worker.run(command) {
            Ok(output) => Ok(json!({
                "output": if output.is_empty() { "(no output)" } else { output.as_str() },
            })),
            Err(error) => {
                self.slash_workers.remove(&session_id);
                Err((5030, error, None))
            }
        }
    }

    fn handle_commands_catalog(&self) -> Result<Value, (i64, String, Option<Value>)> {
        let commands =
            parse_gateway_commands(&self.config.raw).map_err(|error| (-32603, error, None))?;
        let mut pairs = Vec::new();
        let mut canon = serde_json::Map::new();
        let mut grouped = HashMap::<String, Vec<Value>>::new();
        let mut category_order = Vec::<String>::new();

        for command in commands {
            let text = format!("/{}", command.name);
            pairs.push(json!([text, command.description]));
            canon.insert(text.to_ascii_lowercase(), Value::String(text.clone()));
            for alias in &command.aliases {
                canon.insert(
                    format!("/{}", alias).to_ascii_lowercase(),
                    Value::String(text.clone()),
                );
            }
            if !grouped.contains_key(&command.category) {
                category_order.push(command.category.clone());
            }
            grouped
                .entry(command.category)
                .or_default()
                .push(json!([text, command.description]));
        }

        for (name, description, category) in tui_extra_commands() {
            let text = format!("/{}", name);
            pairs.push(json!([text, description]));
            canon.insert(text.to_ascii_lowercase(), Value::String(text.clone()));
            if !grouped.contains_key(category) {
                category_order.push(category.to_string());
            }
            grouped
                .entry(category.to_string())
                .or_default()
                .push(json!([text, description]));
        }

        let categories = category_order
            .into_iter()
            .map(|name| {
                json!({
                    "name": name,
                    "pairs": grouped.remove(&name).unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({
            "pairs": pairs,
            "canon": canon,
            "categories": categories,
            "sub": {},
            "skill_count": 0,
            "warning": "",
        }))
    }

    fn handle_config_get(&self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let key = params
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let loaded = self
            .context
            .load_config_document()
            .map_err(|error| (-32603, error.to_string(), None))?;
        match key {
            "full" => Ok(json!({
                "config": serde_json::to_value(&loaded.raw).unwrap_or_else(|_| json!({})),
            })),
            "mtime" => Ok(json!({
                "mtime": self
                    .context
                    .config_path()
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_secs_f64())
                    .unwrap_or(0.0),
            })),
            "profile" => Ok(json!({
                "home": self.context.hermes_home(),
                "display": self.context.display_hermes_home(),
            })),
            "skin" => Ok(json!({
                "value": config_string_value(loaded.cfg_get(&["display", "skin"]))
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| String::from("default")),
            })),
            "indicator" => Ok(json!({
                "value": normalize_indicator_value(loaded.cfg_get(&["display", "tui_status_indicator"])),
            })),
            "busy" => Ok(json!({
                "value": normalize_busy_input_mode(loaded.cfg_get(&["display", "busy_input_mode"])),
            })),
            "details_mode" => Ok(json!({
                "value": normalize_details_mode(loaded.cfg_get(&["display", "details_mode"])),
            })),
            "compact" => Ok(json!({
                "value": if yaml_bool_with_default(loaded.cfg_get(&["display", "tui_compact"]), false) {
                    "on"
                } else {
                    "off"
                },
            })),
            "statusbar" => Ok(json!({
                "value": coerce_statusbar_value(loaded.cfg_get(&["display", "tui_statusbar"])),
            })),
            "mouse" => Ok(json!({
                "value": if display_mouse_tracking(
                    loaded.cfg_get(&["display"]).and_then(YamlValue::as_mapping),
                ) {
                    "on"
                } else {
                    "off"
                },
            })),
            "reasoning" => Ok(json!({
                "value": config_string_value(loaded.cfg_get(&["agent", "reasoning_effort"]))
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| String::from("medium")),
                "display": if yaml_bool_with_default(
                    loaded.cfg_get(&["display", "show_reasoning"]),
                    false,
                ) {
                    "show"
                } else {
                    "hide"
                },
            })),
            "fast" => Ok(json!({
                "value": if is_fast_service_tier(loaded.cfg_get(&["agent", "service_tier"])) {
                    "fast"
                } else {
                    "normal"
                },
            })),
            _ => Err((-32602, format!("unsupported config key: {key}"), None)),
        }
    }

    fn handle_config_set(&mut self, params: Value) -> Result<Value, (i64, String, Option<Value>)> {
        let key = params
            .get("key")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (-32602, String::from("config key is required"), None))?;
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let mut root = read_raw_yaml_mapping(&self.context.config_path())
            .map_err(|error| (-32603, error.to_string(), None))?;

        let result = match key {
            "busy" => {
                let current = normalize_busy_input_mode(raw_config_value(
                    &root,
                    &["display", "busy_input_mode"],
                ));
                if matches!(value, "" | "status") {
                    json!({"key": key, "value": current})
                } else {
                    let next = match value {
                        "queue" | "steer" | "interrupt" => value,
                        _ => {
                            return Err((-32602, format!("unknown busy mode: {value}"), None));
                        }
                    };
                    ensure_mapping_child_mut(&mut root, "display").insert(
                        yaml_key("busy_input_mode"),
                        YamlValue::String(next.to_string()),
                    );
                    write_yaml_mapping(&self.context.config_path(), &root)
                        .map_err(|error| (-32603, error.to_string(), None))?;
                    json!({"key": key, "value": next})
                }
            }
            "details_mode" => {
                let next = validate_details_mode(value)?;
                let display = ensure_mapping_child_mut(&mut root, "display");
                display.insert(
                    yaml_key("details_mode"),
                    YamlValue::String(next.to_string()),
                );
                let sections = ensure_mapping_child_mut(display, "sections");
                for section in detail_section_names() {
                    sections.insert(yaml_key(section), YamlValue::String(next.to_string()));
                }
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": next})
            }
            _ if key.starts_with("details_mode.") => {
                let section = key.trim_start_matches("details_mode.");
                if !detail_section_names().contains(&section) {
                    return Err((-32602, format!("unknown section: {section}"), None));
                }
                let sections = ensure_mapping_child_mut(
                    ensure_mapping_child_mut(&mut root, "display"),
                    "sections",
                );
                if value.is_empty() {
                    sections.remove(yaml_key(section));
                    write_yaml_mapping(&self.context.config_path(), &root)
                        .map_err(|error| (-32603, error.to_string(), None))?;
                    json!({"key": key, "value": ""})
                } else {
                    let next = validate_details_mode(value)?;
                    sections.insert(yaml_key(section), YamlValue::String(next.to_string()));
                    write_yaml_mapping(&self.context.config_path(), &root)
                        .map_err(|error| (-32603, error.to_string(), None))?;
                    json!({"key": key, "value": next})
                }
            }
            "compact" => {
                let current = yaml_bool_with_default(
                    raw_config_value(&root, &["display", "tui_compact"]),
                    false,
                );
                let next = match value {
                    "" | "toggle" => !current,
                    "on" => true,
                    "off" => false,
                    _ => {
                        return Err((-32602, format!("unknown compact value: {value}"), None));
                    }
                };
                ensure_mapping_child_mut(&mut root, "display")
                    .insert(yaml_key("tui_compact"), YamlValue::Bool(next));
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": if next { "on" } else { "off" }})
            }
            "statusbar" => {
                let current =
                    coerce_statusbar_value(raw_config_value(&root, &["display", "tui_statusbar"]));
                let next = match value {
                    "" | "toggle" => {
                        if current == "off" {
                            "top"
                        } else {
                            "off"
                        }
                    }
                    "on" => "top",
                    "off" | "top" | "bottom" => value,
                    _ => {
                        return Err((-32602, format!("unknown statusbar value: {value}"), None));
                    }
                };
                ensure_mapping_child_mut(&mut root, "display").insert(
                    yaml_key("tui_statusbar"),
                    YamlValue::String(next.to_string()),
                );
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": next})
            }
            "mouse" => {
                let current = display_mouse_tracking(
                    raw_config_value(&root, &["display"]).and_then(YamlValue::as_mapping),
                );
                let next = match value {
                    "" | "toggle" => !current,
                    "on" => true,
                    "off" => false,
                    _ => {
                        return Err((-32602, format!("unknown mouse value: {value}"), None));
                    }
                };
                ensure_mapping_child_mut(&mut root, "display")
                    .insert(yaml_key("mouse_tracking"), YamlValue::Bool(next));
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": if next { "on" } else { "off" }})
            }
            "indicator" => {
                let next = normalize_indicator_input(value)?;
                ensure_mapping_child_mut(&mut root, "display").insert(
                    yaml_key("tui_status_indicator"),
                    YamlValue::String(next.to_string()),
                );
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": next})
            }
            "skin" => {
                if value.is_empty() {
                    return Err((-32602, String::from("skin value required"), None));
                }
                ensure_mapping_child_mut(&mut root, "display")
                    .insert(yaml_key("skin"), YamlValue::String(value.to_string()));
                write_yaml_mapping(&self.context.config_path(), &root)
                    .map_err(|error| (-32603, error.to_string(), None))?;
                json!({"key": key, "value": value})
            }
            "reasoning" => {
                if matches!(value, "show" | "on" | "hide" | "off") {
                    let display = ensure_mapping_child_mut(&mut root, "display");
                    let show = matches!(value, "show" | "on");
                    display.insert(yaml_key("show_reasoning"), YamlValue::Bool(show));
                    ensure_mapping_child_mut(display, "sections").insert(
                        yaml_key("thinking"),
                        YamlValue::String(if show { "expanded" } else { "hidden" }.to_string()),
                    );
                    write_yaml_mapping(&self.context.config_path(), &root)
                        .map_err(|error| (-32603, error.to_string(), None))?;
                    json!({"key": key, "value": if show { "show" } else { "hide" }})
                } else {
                    if parse_reasoning_effort(value).is_none() {
                        return Err((-32602, format!("unknown reasoning value: {value}"), None));
                    }
                    ensure_mapping_child_mut(&mut root, "agent").insert(
                        yaml_key("reasoning_effort"),
                        YamlValue::String(value.to_string()),
                    );
                    write_yaml_mapping(&self.context.config_path(), &root)
                        .map_err(|error| (-32603, error.to_string(), None))?;
                    json!({"key": key, "value": value})
                }
            }
            _ => return Err((-32602, format!("unknown config key: {key}"), None)),
        };

        Ok(result)
    }
}

fn yaml_key(key: &str) -> YamlValue {
    YamlValue::String(key.to_string())
}

fn raw_config_value<'a>(root: &'a Mapping, path: &[&str]) -> Option<&'a YamlValue> {
    let mut current = root;
    for (index, key) in path.iter().enumerate() {
        let value = current.get(yaml_key(key))?;
        if index == path.len() - 1 {
            return Some(value);
        }
        current = value.as_mapping()?;
    }
    None
}

fn ensure_mapping_child_mut<'a>(mapping: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let key_value = yaml_key(key);
    let entry = mapping
        .entry(key_value)
        .or_insert_with(|| YamlValue::Mapping(Mapping::new()));
    if !matches!(entry, YamlValue::Mapping(_)) {
        *entry = YamlValue::Mapping(Mapping::new());
    }
    match entry {
        YamlValue::Mapping(child) => child,
        _ => unreachable!(),
    }
}

fn config_string_value(value: Option<&YamlValue>) -> Option<String> {
    match value {
        Some(YamlValue::String(text)) => Some(text.trim().to_string()),
        Some(YamlValue::Number(number)) => Some(number.to_string()),
        Some(YamlValue::Bool(boolean)) => Some(boolean.to_string()),
        _ => None,
    }
}

fn yaml_bool_with_default(value: Option<&YamlValue>, default: bool) -> bool {
    match value {
        Some(YamlValue::Bool(boolean)) => *boolean,
        Some(YamlValue::Number(number)) => {
            number.as_i64().map(|value| value != 0).unwrap_or(default)
        }
        Some(YamlValue::String(text)) => {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                default
            } else {
                !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
            }
        }
        _ => default,
    }
}

fn normalize_indicator_value(value: Option<&YamlValue>) -> &'static str {
    match config_string_value(value)
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("ascii") => "ascii",
        Some("emoji") => "emoji",
        Some("unicode") => "unicode",
        Some("kaomoji") => "kaomoji",
        _ => "kaomoji",
    }
}

fn normalize_indicator_input(value: &str) -> Result<&str, (i64, String, Option<Value>)> {
    match value {
        "ascii" | "emoji" | "kaomoji" | "unicode" => Ok(value),
        _ => Err((
            -32602,
            format!("unknown indicator: {value:?}; pick one of ascii|emoji|kaomoji|unicode"),
            None,
        )),
    }
}

fn normalize_busy_input_mode(value: Option<&YamlValue>) -> &'static str {
    match config_string_value(value)
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("queue") => "queue",
        Some("steer") => "steer",
        Some("interrupt") => "interrupt",
        _ => "queue",
    }
}

fn normalize_details_mode(value: Option<&YamlValue>) -> &'static str {
    match config_string_value(value)
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("hidden") => "hidden",
        Some("expanded") => "expanded",
        Some("collapsed") => "collapsed",
        _ => "collapsed",
    }
}

fn validate_details_mode(value: &str) -> Result<&str, (i64, String, Option<Value>)> {
    match value {
        "hidden" | "collapsed" | "expanded" => Ok(value),
        _ => Err((-32602, format!("unknown details_mode: {value}"), None)),
    }
}

fn detail_section_names() -> [&'static str; 4] {
    ["thinking", "tools", "subagents", "activity"]
}

fn coerce_statusbar_value(value: Option<&YamlValue>) -> &'static str {
    match value {
        Some(YamlValue::Bool(false)) => "off",
        Some(YamlValue::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "off" => "off",
            "bottom" => "bottom",
            "top" => "top",
            _ => "top",
        },
        _ => "top",
    }
}

fn display_mouse_tracking(display: Option<&Mapping>) -> bool {
    let Some(display) = display else {
        return true;
    };
    let raw = display
        .get(yaml_key("mouse_tracking"))
        .or_else(|| display.get(yaml_key("tui_mouse")));
    yaml_bool_with_default(raw, true)
}

fn is_fast_service_tier(value: Option<&YamlValue>) -> bool {
    matches!(
        config_string_value(value)
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("fast" | "priority" | "on")
    )
}

impl SlashWorker {
    fn start(session_key: &str, model: &str) -> Result<Self, String> {
        let root = project_root();
        #[cfg(test)]
        let python = SLASH_WORKER_PYTHON_OVERRIDE
            .lock()
            .map_err(|_| String::from("slash worker override lock poisoned"))?
            .clone()
            .or_else(|| resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON")));
        #[cfg(not(test))]
        let python = resolve_repo_python(&root, Some("HERMES_GATEWAY_PYTHON"));
        let python = python
            .ok_or_else(|| String::from("could not find a Python interpreter for slash worker"))?;
        let mut child = Command::new(python)
            .current_dir(&root)
            .arg("-m")
            .arg("tui_gateway.slash_worker")
            .arg("--session-key")
            .arg(session_key)
            .arg("--model")
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| error.to_string())?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| String::from("slash worker stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| String::from("slash worker stdout unavailable"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| String::from("slash worker stderr unavailable"))?;
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            stderr: BufReader::new(stderr),
            next_request_id: 1,
        })
    }

    fn run(&mut self, command: &str) -> Result<String, String> {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let payload = json!({
            "id": request_id,
            "command": command,
        });
        serde_json::to_writer(&mut self.stdin, &payload).map_err(|error| error.to_string())?;
        self.stdin
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        self.stdin.flush().map_err(|error| error.to_string())?;

        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Err(self.read_worker_error("slash worker exited without a response"));
        }
        let response = serde_json::from_str::<Value>(line.trim())
            .map_err(|error| format!("invalid slash worker response: {error}"))?;
        if response.get("id").and_then(Value::as_u64) != Some(request_id) {
            return Err(String::from("slash worker response id mismatch"));
        }
        if response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            Ok(response
                .get("output")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string())
        } else {
            Err(response
                .get("error")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| String::from("slash worker request failed")))
        }
    }

    fn read_worker_error(&mut self, fallback: &str) -> String {
        let mut stderr = String::new();
        let _ = self.stderr.read_to_string(&mut stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            fallback.to_string()
        } else {
            format!("{fallback}: {detail}")
        }
    }
}

impl Drop for SlashWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn run_shell_exec(command: &str, cwd: &Path) -> Result<Value, (i64, String, Option<Value>)> {
    let child = shell_exec_command(command, cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| (5003, error.to_string(), None))?;
    let pid = child.id() as i32;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let output = match rx.recv_timeout(Duration::from_secs(SHELL_EXEC_TIMEOUT_SECS)) {
        Ok(result) => result.map_err(|error| (5003, error.to_string(), None))?,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            kill_shell_process(pid);
            return Err((
                5002,
                format!("command timed out ({SHELL_EXEC_TIMEOUT_SECS}s)"),
                None,
            ));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err((5003, String::from("shell worker disconnected"), None));
        }
    };

    Ok(json!({
        "stdout": truncate_tail(&String::from_utf8_lossy(&output.stdout), SHELL_EXEC_STDOUT_LIMIT),
        "stderr": truncate_tail(&String::from_utf8_lossy(&output.stderr), SHELL_EXEC_STDERR_LIMIT),
        "code": output.status.code().unwrap_or_default(),
    }))
}

fn shell_exec_command(command: &str, cwd: &Path) -> Command {
    #[cfg(windows)]
    {
        let mut builder = Command::new("cmd");
        builder.arg("/C").arg(command).current_dir(cwd);
        builder
    }
    #[cfg(not(windows))]
    {
        let mut builder = Command::new("bash");
        builder.arg("-lc").arg(command).current_dir(cwd);
        #[cfg(unix)]
        unsafe {
            builder.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        builder
    }
}

fn truncate_tail(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    chars[chars.len() - max_chars..].iter().collect()
}

fn kill_shell_process(pid: i32) {
    #[cfg(unix)]
    {
        if pid <= 0 {
            return;
        }
        // SAFETY: negative pid targets the process group created in pre_exec.
        let _ = unsafe { libc::kill(-pid, libc::SIGTERM) };
        thread::sleep(Duration::from_millis(250));
        // SAFETY: same as above, escalated after a short grace period.
        let _ = unsafe { libc::kill(-pid, libc::SIGKILL) };
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

fn format_local_timestamp(timestamp: f64) -> String {
    let secs = timestamp.floor() as i64;
    let nanos = ((timestamp.fract().max(0.0)) * 1_000_000_000.0) as u32;
    Local
        .timestamp_opt(secs, nanos)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| String::from("(unknown)"))
}

fn next_gateway_session_id() -> String {
    format!("rust-gw-{:x}", unix_ts_nanos())
}

fn parse_terminal_cols(value: Option<&Value>) -> Result<u16, (i64, String, Option<Value>)> {
    let raw = value.and_then(Value::as_u64).unwrap_or(80);
    let cols = u16::try_from(raw)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| (4004, String::from("cols must be a positive integer"), None))?;
    Ok(cols)
}

fn resolved_branch_title(
    session_store: &SessionStore,
    record: &SessionRecord,
    params: &Value,
) -> Result<String, String> {
    let requested = params
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(name) = requested {
        return SessionStore::sanitize_title(Some(name))
            .map_err(|error| error.to_string())?
            .ok_or_else(|| String::from("branch title required"));
    }
    let base = record.title.as_deref().unwrap_or("branch");
    session_store
        .get_next_title_in_lineage(base)
        .map_err(|error| error.to_string())
}

fn branch_message_append(message: &MessageRecord) -> hermes_core::MessageAppend {
    hermes_core::MessageAppend {
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

fn resolve_gateway_command_name(name: &str, raw_config: &YamlValue) -> Result<String, String> {
    let needle = name.trim_start_matches('/').to_ascii_lowercase();
    let commands = parse_gateway_commands(raw_config)?;
    for command in commands {
        if command.name.eq_ignore_ascii_case(&needle) {
            return Ok(command.name);
        }
        if command
            .aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(&needle))
        {
            return Ok(command.name);
        }
    }
    Ok(needle)
}

fn dispatch_quick_command(
    raw_config: &YamlValue,
    name: &str,
    arg: &str,
) -> Result<Option<Value>, (i64, String, Option<Value>)> {
    let Some(mapping) = raw_config.as_mapping() else {
        return Ok(None);
    };
    let Some(commands) = mapping_value(mapping, "quick_commands").and_then(YamlValue::as_mapping)
    else {
        return Ok(None);
    };
    let Some(entry) = commands.get(YamlValue::String(name.to_string())) else {
        return Ok(None);
    };
    let Some(command_map) = entry.as_mapping() else {
        return Ok(None);
    };
    let command_type = mapping_value(command_map, "type")
        .and_then(YamlValue::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match command_type.as_str() {
        "alias" => Ok(Some(json!({
            "type": "alias",
            "target": mapping_value(command_map, "target").and_then(YamlValue::as_str).unwrap_or_default(),
        }))),
        "exec" => {
            let command = mapping_value(command_map, "command")
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| (4018, String::from("quick command missing command"), None))?;
            let cwd = std::env::current_dir().map_err(|error| (5003, error.to_string(), None))?;
            let result = run_shell_exec(command, &cwd)?;
            let stdout = result
                .get("stdout")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let stderr = result
                .get("stderr")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let code = result
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let output = [stdout, stderr]
                .into_iter()
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            if code != 0 {
                return Err((
                    4018,
                    if output.is_empty() {
                        format!("quick command failed with exit code {code}")
                    } else {
                        output
                    },
                    None,
                ));
            }
            Ok(Some(json!({
                "type": "exec",
                "output": output,
                "arg": arg,
            })))
        }
        _ => Ok(None),
    }
}

fn mapping_value<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn required_session_id(params: &Value) -> Result<String, (i64, String, Option<Value>)> {
    params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| (-32602, String::from("session_id is required"), None))
}

fn cwd_from_record(record: &SessionRecord) -> PathBuf {
    record
        .model_config
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|cfg| cfg.get("cwd"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .filter(|cwd| cwd.is_absolute())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn overrides_from_record(record: &SessionRecord) -> ModelOverrides {
    let mut overrides = ModelOverrides {
        model: record.model.clone(),
        ..ModelOverrides::default()
    };
    if let Some(config) = record.model_config.as_ref().and_then(Value::as_object) {
        overrides.provider = config
            .get("provider")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        overrides.base_url = config
            .get("base_url")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        overrides.api_mode = config
            .get("api_mode")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
    overrides
}

fn transcript_messages(messages: &[MessageRecord]) -> Vec<Value> {
    messages
        .iter()
        .filter_map(|message| {
            let role = message.role.as_str();
            if !matches!(role, "user" | "assistant" | "tool" | "system") {
                return None;
            }
            let text = message_display_text(message);
            Some(json!({
                "role": role,
                "name": message.tool_name,
                "text": text,
            }))
        })
        .collect()
}

fn message_display_text(message: &MessageRecord) -> String {
    match message.content.as_ref() {
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_default(),
        None => String::new(),
    }
}

fn session_info_payload(config: &LoadedConfig) -> Value {
    json!({
        "model": config.configured_model_name().unwrap_or_default(),
        "skills": {},
        "tools": {},
        "version": env!("CARGO_PKG_VERSION"),
    })
}

fn complete_path_items(word: &str, cwd: &Path) -> Vec<Value> {
    if word.is_empty() {
        return Vec::new();
    }

    if word == "@" {
        return static_context_items();
    }

    if let Some(query) = word.strip_prefix('@') {
        if query.is_empty() {
            return static_context_items();
        }
        if !query.contains('/') && !query.contains(':') {
            let prefixed = static_context_items()
                .into_iter()
                .filter(|item| {
                    item["text"]
                        .as_str()
                        .is_some_and(|text| text.starts_with(word))
                })
                .collect::<Vec<_>>();
            if !prefixed.is_empty() {
                return prefixed;
            }
        }
        if query == "file" || query == "folder" {
            return list_path_items("", cwd, Some(query));
        }
        if let Some(path_part) = query.strip_prefix("file:") {
            return list_path_items(path_part, cwd, Some("file"));
        }
        if let Some(path_part) = query.strip_prefix("folder:") {
            return list_path_items(path_part, cwd, Some("folder"));
        }
    }

    list_path_items(word, cwd, None)
}

fn static_context_items() -> Vec<Value> {
    vec![
        json!({"text":"@diff","display":"@diff","meta":"git diff"}),
        json!({"text":"@staged","display":"@staged","meta":"staged diff"}),
        json!({"text":"@file:","display":"@file:","meta":"attach file"}),
        json!({"text":"@folder:","display":"@folder:","meta":"attach folder"}),
        json!({"text":"@url:","display":"@url:","meta":"fetch url"}),
        json!({"text":"@git:","display":"@git:","meta":"git log"}),
    ]
}

fn list_path_items(word: &str, cwd: &Path, context_kind: Option<&str>) -> Vec<Value> {
    let expanded = normalize_completion_path(word, cwd);
    let (search_dir, match_prefix) =
        if expanded.as_os_str().is_empty() || expanded == PathBuf::from(".") {
            (cwd.to_path_buf(), String::new())
        } else if word.ends_with('/') {
            (expanded, String::new())
        } else {
            let search_dir = if expanded.is_dir() && word.ends_with(std::path::MAIN_SEPARATOR) {
                expanded.clone()
            } else {
                expanded
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| cwd.to_path_buf())
            };
            let match_prefix = expanded
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            (search_dir, match_prefix)
        };

    let Ok(entries) = fs::read_dir(&search_dir) else {
        return Vec::new();
    };

    let mut items = Vec::new();
    let match_lower = match_prefix.to_ascii_lowercase();
    let mut rows = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let name = file_name.to_str()?.to_string();
            if !match_prefix.is_empty() && !name.to_ascii_lowercase().starts_with(&match_lower) {
                return None;
            }
            if context_kind.is_none() && word.starts_with('@') && name.starts_with('.') {
                return None;
            }
            let full = entry.path();
            let is_dir = full.is_dir();
            match context_kind {
                Some("file") if is_dir => return None,
                Some("folder") if !is_dir => return None,
                _ => {}
            }
            Some((name, full, is_dir))
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));

    for (name, full, is_dir) in rows.into_iter().take(30) {
        let rel = make_relative_display_path(&full, cwd);
        let suffix = if is_dir { "/" } else { "" };
        let text = if let Some(kind) = context_kind {
            format!("@{kind}:{rel}{suffix}")
        } else if word.starts_with("~/") {
            let home = dirs::home_dir().unwrap_or_else(|| cwd.to_path_buf());
            let home_rel = full.strip_prefix(&home).unwrap_or(full.as_path());
            format!("~/{}{}", home_rel.display(), suffix)
        } else if word.starts_with('@') {
            let kind = if is_dir { "folder" } else { "file" };
            format!("@{kind}:{rel}{suffix}")
        } else if word.starts_with("./") {
            format!("./{rel}{suffix}")
        } else if full.is_absolute() && word.starts_with('/') {
            format!("{}{}", full.display(), suffix)
        } else {
            format!("{rel}{suffix}")
        };
        items.push(json!({
            "text": text,
            "display": format!("{name}{suffix}"),
            "meta": if is_dir { "dir" } else { "" },
        }));
    }

    items
}

fn normalize_completion_path(word: &str, cwd: &Path) -> PathBuf {
    let raw = word.strip_prefix('@').unwrap_or(word);
    let raw = raw
        .strip_prefix("file:")
        .or_else(|| raw.strip_prefix("folder:"))
        .unwrap_or(raw);
    if raw.is_empty() {
        return cwd.to_path_buf();
    }
    if raw == "." {
        return cwd.to_path_buf();
    }
    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| cwd.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| cwd.to_path_buf())
            .join(rest);
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn make_relative_display_path(path: &Path, cwd: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn complete_slash_items(text: &str, raw_config: &YamlValue) -> Result<(Vec<Value>, usize), String> {
    if !text.starts_with('/') {
        return Ok((Vec::new(), 1));
    }

    if let Some((items, replace_from)) = details_completions(text) {
        return Ok((items, replace_from));
    }

    let mut items = parse_gateway_commands(raw_config)?
        .into_iter()
        .filter_map(|command| {
            let completion = format!("/{}", command.name);
            let aliases = command
                .aliases
                .iter()
                .map(|alias| format!("/{}", alias))
                .collect::<Vec<_>>();
            let mut values = vec![completion];
            values.extend(aliases);
            let matched = values
                .into_iter()
                .filter(|value| value.starts_with(text))
                .map(|value| {
                    json!({
                        "text": value,
                        "display": value,
                        "meta": command.description,
                    })
                })
                .collect::<Vec<_>>();
            (!matched.is_empty()).then_some(matched)
        })
        .flatten()
        .collect::<Vec<_>>();

    for extra in [
        ("compact", "Toggle compact display mode"),
        ("details", "Control agent detail visibility"),
        ("logs", "Show recent gateway log lines"),
        ("mouse", "Toggle mouse/wheel tracking [on|off|toggle]"),
    ] {
        let value = format!("/{}", extra.0);
        if value.starts_with(text) && !items.iter().any(|item| item["text"] == json!(value)) {
            items.push(json!({
                "text": value,
                "display": value,
                "meta": extra.1,
            }));
        }
    }

    items.truncate(30);
    Ok((items, 1))
}

fn split_slash_command(command: &str) -> (String, &str) {
    let trimmed = command.trim();
    let raw = trimmed.strip_prefix('/').unwrap_or(trimmed);
    let mut parts = raw.splitn(2, char::is_whitespace);
    let base = parts.next().unwrap_or_default().to_ascii_lowercase();
    let arg = parts.next().unwrap_or_default().trim_start();
    (base, arg)
}

fn pending_input_commands() -> &'static [&'static str] {
    &["retry", "queue", "q", "steer", "plan", "goal"]
}

fn worker_blocked_commands() -> &'static [&'static str] {
    &["snapshot", "snap"]
}

fn details_completions(text: &str) -> Option<(Vec<Value>, usize)> {
    let lower = text.to_ascii_lowercase();
    if !lower.starts_with("/details") {
        return None;
    }

    let body = text.strip_prefix("/details").unwrap_or_default();
    let body = body.strip_prefix(' ').unwrap_or(body);
    let has_trailing_space = text.ends_with(' ');
    let parts = if body.is_empty() {
        Vec::new()
    } else {
        body.split_whitespace().collect::<Vec<_>>()
    };
    let sections = ["thinking", "tools", "subagents", "activity"];
    let modes = ["hidden", "collapsed", "expanded"];

    if body.is_empty() || (parts.is_empty() && has_trailing_space) {
        let mut items = modes
            .iter()
            .map(|mode| {
                let value = if has_trailing_space {
                    (*mode).to_string()
                } else {
                    format!(" {mode}")
                };
                json!({"text": value, "display": *mode, "meta": "global mode"})
            })
            .collect::<Vec<_>>();
        items.push(json!({
            "text": if has_trailing_space { "cycle".to_string() } else { " cycle".to_string() },
            "display":"cycle",
            "meta":"cycle global mode"
        }));
        items.extend(sections.iter().map(|section| {
            let value = if has_trailing_space {
                (*section).to_string()
            } else {
                format!(" {section}")
            };
            json!({"text": value, "display": *section, "meta": "section override"})
        }));
        return Some((
            items,
            text.rfind(' ').map(|index| index + 1).unwrap_or(text.len()),
        ));
    }

    if parts.len() == 1 && !has_trailing_space {
        let prefix = parts[0].to_ascii_lowercase();
        let mut items = Vec::new();
        for mode in modes {
            if mode.starts_with(&prefix) && mode != prefix {
                items.push(json!({"text":mode,"display":mode,"meta":"global mode"}));
            }
        }
        if "cycle".starts_with(&prefix) && prefix != "cycle" {
            items.push(json!({"text":"cycle","display":"cycle","meta":"cycle global mode"}));
        }
        for section in sections {
            if section.starts_with(&prefix) && section != prefix {
                items.push(json!({"text":section,"display":section,"meta":"section override"}));
            }
        }
        return Some((
            items,
            text.rfind(' ').map(|index| index + 1).unwrap_or(text.len()),
        ));
    }

    if parts.len() == 1
        && has_trailing_space
        && sections.contains(&parts[0].to_ascii_lowercase().as_str())
    {
        let section = parts[0].to_ascii_lowercase();
        let mut items = modes
            .iter()
            .map(|mode| json!({"text": *mode, "display": *mode, "meta": format!("set {section}")}))
            .collect::<Vec<_>>();
        items.push(
            json!({"text":"reset","display":"reset","meta": format!("clear {section} override")}),
        );
        return Some((
            items,
            text.rfind(' ').map(|index| index + 1).unwrap_or(text.len()),
        ));
    }

    if parts.len() == 2
        && !has_trailing_space
        && sections.contains(&parts[0].to_ascii_lowercase().as_str())
    {
        let section = parts[0].to_ascii_lowercase();
        let prefix = parts[1].to_ascii_lowercase();
        let mut items = Vec::new();
        for mode in modes {
            if mode.starts_with(&prefix) && mode != prefix {
                items.push(json!({"text":mode,"display":mode,"meta": format!("set {section}")}));
            }
        }
        if "reset".starts_with(&prefix) && prefix != "reset" {
            items.push(json!({"text":"reset","display":"reset","meta": format!("clear {section} override")}));
        }
        return Some((
            items,
            text.rfind(' ').map(|index| index + 1).unwrap_or(text.len()),
        ));
    }

    Some((
        Vec::new(),
        text.rfind(' ').map(|index| index + 1).unwrap_or(text.len()),
    ))
}

#[derive(Debug)]
struct RegistryCommand {
    name: String,
    description: String,
    aliases: Vec<String>,
    category: String,
}

fn parse_gateway_commands(raw_config: &YamlValue) -> Result<Vec<RegistryCommand>, String> {
    let source = fs::read_to_string(command_registry_path())
        .map_err(|error| format!("reading command registry failed: {error}"))?;
    let start = source
        .find("COMMAND_REGISTRY")
        .ok_or_else(|| String::from("could not locate COMMAND_REGISTRY"))?;
    let regex = Regex::new(r#"CommandDef\((?s:.*?)\)"#)
        .map_err(|error| format!("building registry regex failed: {error}"))?;
    let alias_regex = Regex::new(r#"aliases=\((?P<body>[^)]*)\)"#)
        .map_err(|error| format!("building alias regex failed: {error}"))?;
    let gate_regex = Regex::new(r#"gateway_config_gate="(?P<value>[^"]+)""#)
        .map_err(|error| format!("building config gate regex failed: {error}"))?;

    let mut commands = Vec::new();
    for capture in regex.find_iter(&source[start..]) {
        let block = capture.as_str();
        let strings = string_literals(block);
        if strings.len() < 3 {
            continue;
        }
        if block.contains("gateway_only=True") || !block.contains("cli_only=True") {
            commands.push(RegistryCommand {
                name: strings[0].clone(),
                description: strings[1].clone(),
                aliases: alias_regex
                    .captures(block)
                    .and_then(|captures| captures.name("body"))
                    .map(|body| string_literals(body.as_str()))
                    .unwrap_or_default(),
                category: strings[2].clone(),
            });
            continue;
        }
        let gate = gate_regex
            .captures(block)
            .and_then(|captures| captures.name("value"))
            .map(|value| value.as_str().to_string());
        if gate
            .as_deref()
            .is_some_and(|value| config_gate_truthy(raw_config, value))
        {
            commands.push(RegistryCommand {
                name: strings[0].clone(),
                description: strings[1].clone(),
                aliases: alias_regex
                    .captures(block)
                    .and_then(|captures| captures.name("body"))
                    .map(|body| string_literals(body.as_str()))
                    .unwrap_or_default(),
                category: strings[2].clone(),
            });
        }
    }

    Ok(commands)
}

fn tui_extra_commands() -> [(&'static str, &'static str, &'static str); 4] {
    [
        ("compact", "Toggle compact display mode", "TUI"),
        ("details", "Control agent detail visibility", "TUI"),
        ("logs", "Show recent gateway log lines", "TUI"),
        (
            "mouse",
            "Toggle mouse/wheel tracking [on|off|toggle]",
            "TUI",
        ),
    ]
}

fn command_registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("hermes_cli")
        .join("commands.py")
}

fn string_literals(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            if escaped {
                current.push(ch);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                values.push(current.clone());
                current.clear();
                in_string = false;
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_string = true;
        }
    }
    values
}

fn config_gate_truthy(raw_config: &YamlValue, gate: &str) -> bool {
    let mut node = raw_config;
    for part in gate.split('.') {
        match node {
            YamlValue::Mapping(mapping) => {
                let key = YamlValue::String(part.to_string());
                let Some(value) = mapping.get(&key) else {
                    return false;
                };
                node = value;
            }
            _ => return false,
        }
    }
    yaml_truthy(node)
}

fn yaml_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Bool(boolean) => *boolean,
        YamlValue::Number(number) => number
            .as_i64()
            .map(|value| value != 0)
            .or_else(|| number.as_u64().map(|value| value != 0))
            .or_else(|| number.as_f64().map(|value| value != 0.0))
            .unwrap_or(false),
        YamlValue::String(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        _ => false,
    }
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default()
}

fn write_jsonrpc_result<W: Write>(writer: &mut W, id: Value, result: Value) -> io::Result<()> {
    write_json_line(
        writer,
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
    )
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
    if let Some(data) = data {
        error["data"] = data;
    }
    write_json_line(
        writer,
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": error,
        }),
    )
}

fn write_jsonrpc_event<W: Write>(writer: &mut W, params: Value) -> io::Result<()> {
    write_json_line(
        writer,
        json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": params,
        }),
    )
}

fn write_json_line<W: Write>(writer: &mut W, value: Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, &value)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

pub fn run_native_gateway_stdio(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &SessionStore,
) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut writer = BufWriter::new(stdout);
    run_native_gateway_jsonrpc(context, config, session_store, &mut reader, &mut writer)
}

pub(crate) fn run_native_gateway_jsonrpc<R: BufRead + Send, W: Write>(
    context: &HermesContext,
    config: &LoadedConfig,
    session_store: &SessionStore,
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()> {
    NativeGatewayServer::new(context, config, session_store).run(reader, writer)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Cursor, Read, Write};
    use std::net::TcpListener;
    #[cfg(not(windows))]
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use hermes_core::{MessageAppend, SessionUsageDelta};
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    static PYTHON_ENV_LOCK: Mutex<()> = Mutex::new(());
    static SESSION_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct CurrentDirGuard(PathBuf);

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn pushd(path: &Path) -> CurrentDirGuard {
        let old = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        CurrentDirGuard(old)
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

    #[cfg(not(windows))]
    fn write_fake_python_worker(root: &Path, args_log: &Path, request_log: &Path) -> PathBuf {
        let script = root.join("fake-python.sh");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s ' \"$@\" >> '{}'\nprintf '\\n' >> '{}'\ncount=0\nwhile IFS= read -r line; do\n  printf '%s\\n' \"$line\" >> '{}'\n  count=$((count + 1))\n  printf '{{\"id\":%s,\"ok\":true,\"output\":\"worker:%s\"}}\\n' \"$count\" \"$count\"\ndone\n",
                args_log.display(),
                args_log.display(),
                request_log.display()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        script
    }

    #[test]
    fn native_gateway_emits_ready_and_creates_session() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let input = [
            json!({"jsonrpc":"2.0","id":1,"method":"setup.status","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","id":2,"method":"session.create","params":{"cols":80}})
                .to_string(),
        ]
        .join("\n")
            + "\n";
        let mut output = Vec::new();
        run_native_gateway_jsonrpc(
            &context,
            &config,
            &store,
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();

        let lines = String::from_utf8(output).unwrap();
        let frames = lines
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(frames[0]["method"], json!("event"));
        assert_eq!(frames[0]["params"]["type"], json!("gateway.ready"));
        assert_eq!(frames[1]["result"]["provider_configured"], json!(true));
        assert!(
            frames[2]["result"]["session_id"]
                .as_str()
                .unwrap()
                .starts_with("rust-gw-")
        );
    }

    #[test]
    fn native_gateway_prompt_submit_streams_message_complete() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "hello from native gateway"
                    }
                }]
            })
            .to_string(),
        ]);
        let store = context.open_session_store().unwrap();
        let session_id = store
            .create_session(&SessionCreate {
                id: String::from("rust-gw-test"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": base_url,
                    "api_mode": "chat_completions",
                })),
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        let input = [
            json!({"jsonrpc":"2.0","id":1,"method":"session.resume","params":{"session_id":session_id}}).to_string(),
            json!({"jsonrpc":"2.0","id":2,"method":"prompt.submit","params":{"session_id":"rust-gw-test","text":"hello"}}).to_string(),
        ]
        .join("\n")
            + "\n";
        let mut output = Vec::new();
        run_native_gateway_jsonrpc(
            &context,
            &config,
            &store,
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();
        let lines = String::from_utf8(output).unwrap();
        let frames = lines
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(
            frames
                .iter()
                .any(|frame| frame["params"]["type"] == json!("message.start"))
        );
        assert!(frames.iter().any(|frame| {
            frame["params"]["type"] == json!("message.complete")
                && frame["params"]["payload"]["text"] == json!("hello from native gateway")
        }));
        assert_eq!(frames.last().unwrap()["result"]["ok"], json!(true));
    }

    #[test]
    fn native_gateway_direct_prompt_submit_completes() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let base_url = serve_chat_sequence(vec![
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "hello from direct prompt"
                    }
                }]
            })
            .to_string(),
        ]);
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("rust-gw-direct"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": base_url,
                    "api_mode": "chat_completions",
                })),
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("rust-gw-direct"),
            NativeGatewaySessionState {
                cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("test-model")),
                    provider: Some(String::from("custom")),
                    base_url: Some(base_url),
                    api_key: Some(String::from("test-key")),
                    api_mode: Some(String::from("chat_completions")),
                },
            },
        );

        let mut output = Vec::new();
        let result = server
            .handle_prompt_submit(
                &mut output,
                json!({"session_id":"rust-gw-direct","text":"hello"}),
            )
            .unwrap();

        assert_eq!(result["ok"], json!(true));
        let lines = String::from_utf8(output).unwrap();
        let frames = lines
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(frames.iter().any(|frame| {
            frame["params"]["type"] == json!("message.complete")
                && frame["params"]["payload"]["text"] == json!("hello from direct prompt")
        }));
    }

    #[test]
    fn native_gateway_complete_path_lists_context_refs_and_files() {
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("alpha.txt"), "alpha").unwrap();
        fs::create_dir_all(temp.path().join("docs")).unwrap();

        let refs = complete_path_items("@", temp.path());
        assert!(refs.iter().any(|item| item["text"] == json!("@diff")));
        assert!(refs.iter().any(|item| item["text"] == json!("@file:")));

        let ref_prefix = complete_path_items("@fi", temp.path());
        assert!(
            ref_prefix
                .iter()
                .any(|item| item["text"] == json!("@file:"))
        );

        let files = complete_path_items("@file:a", temp.path());
        assert!(
            files
                .iter()
                .any(|item| item["text"] == json!("@file:alpha.txt"))
        );

        let folders = complete_path_items("@folder:d", temp.path());
        assert!(
            folders
                .iter()
                .any(|item| item["text"] == json!("@folder:docs/"))
        );
    }

    #[test]
    fn native_gateway_complete_slash_lists_gateway_commands_and_details_modes() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();

        let (slash_items, slash_replace_from) = complete_slash_items("/he", &config.raw).unwrap();
        assert_eq!(slash_replace_from, 1);
        assert!(
            slash_items
                .iter()
                .any(|item| item["text"] == json!("/help"))
        );

        let (detail_items, detail_replace_from) =
            complete_slash_items("/details thinking ", &config.raw).unwrap();
        assert_eq!(detail_replace_from, "/details thinking ".len());
        assert!(
            detail_items
                .iter()
                .any(|item| item["text"] == json!("hidden"))
        );
        assert!(
            detail_items
                .iter()
                .any(|item| item["text"] == json!("reset"))
        );

        let (detail_root_items, detail_root_replace_from) =
            complete_slash_items("/details", &config.raw).unwrap();
        assert_eq!(detail_root_replace_from, "/details".len());
        assert!(
            detail_root_items
                .iter()
                .any(|item| item["text"] == json!(" hidden"))
        );
    }

    #[test]
    fn native_gateway_startup_rpcs_return_catalog_config_and_recent_session() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            "display:\n  tui_auto_resume_recent: true\n  tui_compact: true\n",
        )
        .unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("recent-session"),
                source: String::from("rust-agent"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let input = [
            json!({"jsonrpc":"2.0","id":1,"method":"commands.catalog","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","id":2,"method":"config.get","params":{"key":"full"}})
                .to_string(),
            json!({"jsonrpc":"2.0","id":3,"method":"config.get","params":{"key":"mtime"}})
                .to_string(),
            json!({"jsonrpc":"2.0","id":4,"method":"session.most_recent","params":{}}).to_string(),
        ]
        .join("\n")
            + "\n";
        let mut output = Vec::new();
        run_native_gateway_jsonrpc(
            &context,
            &config,
            &store,
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();

        let frames = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(frames[1]["result"]["canon"]["/help"], json!("/help"));
        assert!(
            frames[1]["result"]["pairs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|pair| pair[0] == json!("/compact"))
        );
        assert_eq!(
            frames[2]["result"]["config"]["display"]["tui_compact"],
            json!(true)
        );
        assert!(frames[3]["result"]["mtime"].as_f64().unwrap() > 0.0);
        assert_eq!(frames[4]["result"]["session_id"], json!("recent-session"));
    }

    #[test]
    fn native_gateway_config_get_normalizes_supported_values() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        fs::write(
            context.config_path(),
            concat!(
                "display:\n",
                "  skin: slate\n",
                "  tui_status_indicator: \" Emoji \"\n",
                "  busy_input_mode: drop\n",
                "  details_mode: strange\n",
                "  tui_compact: true\n",
                "  tui_statusbar: false\n",
                "  mouse_tracking: off\n",
                "  show_reasoning: true\n",
                "agent:\n",
                "  reasoning_effort: high\n",
                "  service_tier: priority\n",
            ),
        )
        .unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let server = NativeGatewayServer::new(&context, &config, &store);

        assert_eq!(
            server.handle_config_get(json!({"key":"skin"})).unwrap()["value"],
            json!("slate")
        );
        assert_eq!(
            server
                .handle_config_get(json!({"key":"indicator"}))
                .unwrap()["value"],
            json!("emoji")
        );
        assert_eq!(
            server.handle_config_get(json!({"key":"busy"})).unwrap()["value"],
            json!("queue")
        );
        assert_eq!(
            server
                .handle_config_get(json!({"key":"details_mode"}))
                .unwrap()["value"],
            json!("collapsed")
        );
        assert_eq!(
            server.handle_config_get(json!({"key":"compact"})).unwrap()["value"],
            json!("on")
        );
        assert_eq!(
            server
                .handle_config_get(json!({"key":"statusbar"}))
                .unwrap()["value"],
            json!("off")
        );
        assert_eq!(
            server.handle_config_get(json!({"key":"mouse"})).unwrap()["value"],
            json!("off")
        );
        let reasoning = server
            .handle_config_get(json!({"key":"reasoning"}))
            .unwrap();
        assert_eq!(reasoning["value"], json!("high"));
        assert_eq!(reasoning["display"], json!("show"));
        assert_eq!(
            server.handle_config_get(json!({"key":"fast"})).unwrap()["value"],
            json!("fast")
        );
    }

    #[test]
    fn native_gateway_config_set_persists_supported_values() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = NativeGatewayServer::new(&context, &config, &store);

        assert_eq!(
            server
                .handle_config_set(json!({"key":"busy","value":"steer"}))
                .unwrap()["value"],
            json!("steer")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"details_mode","value":"expanded"}))
                .unwrap()["value"],
            json!("expanded")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"details_mode.tools","value":"hidden"}))
                .unwrap()["value"],
            json!("hidden")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"compact","value":"on"}))
                .unwrap()["value"],
            json!("on")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"statusbar","value":"bottom"}))
                .unwrap()["value"],
            json!("bottom")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"mouse","value":"off"}))
                .unwrap()["value"],
            json!("off")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"indicator","value":"unicode"}))
                .unwrap()["value"],
            json!("unicode")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"skin","value":"slate"}))
                .unwrap()["value"],
            json!("slate")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"reasoning","value":"hide"}))
                .unwrap()["value"],
            json!("hide")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"reasoning","value":"high"}))
                .unwrap()["value"],
            json!("high")
        );
        assert_eq!(
            server
                .handle_config_set(json!({"key":"details_mode.tools","value":""}))
                .unwrap()["value"],
            json!("")
        );

        let root = read_raw_yaml_mapping(&context.config_path()).unwrap();
        assert_eq!(
            raw_config_value(&root, &["display", "busy_input_mode"]).and_then(YamlValue::as_str),
            Some("steer")
        );
        assert_eq!(
            raw_config_value(&root, &["display", "details_mode"]).and_then(YamlValue::as_str),
            Some("expanded")
        );
        assert_eq!(
            raw_config_value(&root, &["display", "tui_compact"]).and_then(YamlValue::as_bool),
            Some(true)
        );
        assert_eq!(
            raw_config_value(&root, &["display", "tui_statusbar"]).and_then(YamlValue::as_str),
            Some("bottom")
        );
        assert_eq!(
            raw_config_value(&root, &["display", "mouse_tracking"]).and_then(YamlValue::as_bool),
            Some(false)
        );
        assert_eq!(
            raw_config_value(&root, &["display", "tui_status_indicator"])
                .and_then(YamlValue::as_str),
            Some("unicode")
        );
        assert_eq!(
            raw_config_value(&root, &["display", "skin"]).and_then(YamlValue::as_str),
            Some("slate")
        );
        assert_eq!(
            raw_config_value(&root, &["display", "show_reasoning"]).and_then(YamlValue::as_bool),
            Some(false)
        );
        assert_eq!(
            raw_config_value(&root, &["display", "sections", "thinking"])
                .and_then(YamlValue::as_str),
            Some("hidden")
        );
        assert_eq!(
            raw_config_value(&root, &["agent", "reasoning_effort"]).and_then(YamlValue::as_str),
            Some("high")
        );
        assert!(raw_config_value(&root, &["display", "sections", "tools"]).is_none());
    }

    #[test]
    fn native_gateway_shell_exec_runs_command_and_blocks_dangerous_commands() {
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = NativeGatewayServer::new(&context, &config, &store);

        let mut output = Vec::new();
        server
            .handle_request(
                "shell.exec",
                json!(1),
                json!({"command":"printf alpha; printf beta >&2; exit 7"}),
                &mut output,
                false,
            )
            .unwrap();
        let frame = serde_json::from_slice::<Value>(&output).unwrap();
        assert_eq!(frame["result"]["stdout"], json!("alpha"));
        assert_eq!(frame["result"]["stderr"], json!("beta"));
        assert_eq!(frame["result"]["code"], json!(7));

        let mut blocked = Vec::new();
        server
            .handle_request(
                "shell.exec",
                json!(2),
                json!({"command":"rm -rf /tmp/native-gateway-test"}),
                &mut blocked,
                false,
            )
            .unwrap();
        let blocked_frame = serde_json::from_slice::<Value>(&blocked).unwrap();
        assert_eq!(blocked_frame["error"]["code"], json!(4005));
        assert!(
            blocked_frame["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Use the agent for dangerous commands.")
        );
    }

    #[test]
    fn native_gateway_session_management_rpcs_round_trip() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let _cwd = pushd(temp.path());

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("session-rpc-test"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: Some(json!({
                    "provider": "custom",
                    "base_url": "http://example.test",
                    "api_mode": "chat_completions",
                })),
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "session-rpc-test",
                &hermes_core::MessageAppend {
                    role: String::from("user"),
                    content: Some(json!("hello")),
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
            .unwrap();

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("session-rpc-test"),
            NativeGatewaySessionState {
                cwd: temp.path().to_path_buf(),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("test-model")),
                    provider: Some(String::from("custom")),
                    base_url: Some(String::from("http://example.test")),
                    api_mode: Some(String::from("chat_completions")),
                    ..ModelOverrides::default()
                },
            },
        );

        let mut title_set = Vec::new();
        server
            .handle_request(
                "session.title",
                json!(1),
                json!({"session_id":"session-rpc-test","title":"Sprint"}),
                &mut title_set,
                false,
            )
            .unwrap();
        let title_set_frame = serde_json::from_slice::<Value>(&title_set).unwrap();
        assert_eq!(title_set_frame["result"]["title"], json!("Sprint"));

        let mut title_get = Vec::new();
        server
            .handle_request(
                "session.title",
                json!(2),
                json!({"session_id":"session-rpc-test"}),
                &mut title_get,
                false,
            )
            .unwrap();
        let title_get_frame = serde_json::from_slice::<Value>(&title_get).unwrap();
        assert_eq!(title_get_frame["result"]["title"], json!("Sprint"));

        let mut status = Vec::new();
        server
            .handle_request(
                "session.status",
                json!(3),
                json!({"session_id":"session-rpc-test"}),
                &mut status,
                false,
            )
            .unwrap();
        let status_frame = serde_json::from_slice::<Value>(&status).unwrap();
        assert!(
            status_frame["result"]["output"]
                .as_str()
                .unwrap()
                .contains("Title: Sprint")
        );

        let mut save = Vec::new();
        server
            .handle_request(
                "session.save",
                json!(4),
                json!({"session_id":"session-rpc-test"}),
                &mut save,
                false,
            )
            .unwrap();
        let save_frame = serde_json::from_slice::<Value>(&save).unwrap();
        let saved_file = PathBuf::from(save_frame["result"]["file"].as_str().unwrap());
        assert!(saved_file.exists());
        let saved_json = serde_json::from_slice::<Value>(&fs::read(&saved_file).unwrap()).unwrap();
        assert_eq!(saved_json["model"], json!("test-model"));
        assert_eq!(saved_json["messages"][0]["text"], json!("hello"));

        let mut delete_active = Vec::new();
        server
            .handle_request(
                "session.delete",
                json!(5),
                json!({"session_id":"session-rpc-test"}),
                &mut delete_active,
                false,
            )
            .unwrap();
        let delete_active_frame = serde_json::from_slice::<Value>(&delete_active).unwrap();
        assert_eq!(delete_active_frame["error"]["code"], json!(4023));

        let mut close = Vec::new();
        server
            .handle_request(
                "session.close",
                json!(6),
                json!({"session_id":"session-rpc-test"}),
                &mut close,
                false,
            )
            .unwrap();
        let close_frame = serde_json::from_slice::<Value>(&close).unwrap();
        assert_eq!(close_frame["result"]["closed"], json!(true));

        let mut delete_closed = Vec::new();
        server
            .handle_request(
                "session.delete",
                json!(7),
                json!({"session_id":"session-rpc-test"}),
                &mut delete_closed,
                false,
            )
            .unwrap();
        let delete_closed_frame = serde_json::from_slice::<Value>(&delete_closed).unwrap();
        assert_eq!(
            delete_closed_frame["result"]["deleted"],
            json!("session-rpc-test")
        );
    }

    #[test]
    fn native_gateway_session_undo_removes_last_exchange() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("undo-session"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        for message in [
            MessageAppend {
                role: String::from("user"),
                content: Some(json!("keep")),
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
            MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("keep answer")),
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
            MessageAppend {
                role: String::from("user"),
                content: Some(json!("remove")),
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
            MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("remove answer")),
                tool_call_id: None,
                tool_calls: Some(json!([{"name":"search"}])),
                tool_name: None,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ] {
            store.append_message("undo-session", &message).unwrap();
        }

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("undo-session"),
            NativeGatewaySessionState {
                cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides::default(),
            },
        );

        let mut undo = Vec::new();
        server
            .handle_request(
                "session.undo",
                json!(1),
                json!({"session_id":"undo-session"}),
                &mut undo,
                false,
            )
            .unwrap();
        let undo_frame = serde_json::from_slice::<Value>(&undo).unwrap();
        assert_eq!(undo_frame["result"]["removed"], json!(2));

        let messages = store.get_messages("undo-session").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, Some(json!("keep")));
        assert_eq!(messages[1].content, Some(json!("keep answer")));
    }

    #[test]
    fn native_gateway_session_branch_copies_history_and_lineage_title() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("branch-source"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store.set_session_title("branch-source", "Sprint").unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("branch-older"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: Some(String::from("branch-source")),
            })
            .unwrap();
        store
            .set_session_title("branch-older", "Sprint #2")
            .unwrap();
        for message in [
            MessageAppend {
                role: String::from("user"),
                content: Some(json!("question")),
                tool_call_id: None,
                tool_calls: None,
                tool_name: None,
                token_count: Some(3),
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            MessageAppend {
                role: String::from("assistant"),
                content: Some(json!("answer")),
                tool_call_id: None,
                tool_calls: Some(json!([{"name":"search"}])),
                tool_name: None,
                token_count: Some(5),
                finish_reason: Some(String::from("stop")),
                reasoning: Some(String::from("brief")),
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ] {
            store.append_message("branch-source", &message).unwrap();
        }

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("branch-source"),
            NativeGatewaySessionState {
                cwd: PathBuf::from("/tmp/demo"),
                cols: 120,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("override-model")),
                    provider: Some(String::from("demo-provider")),
                    ..ModelOverrides::default()
                },
            },
        );

        let mut branch = Vec::new();
        server
            .handle_request(
                "session.branch",
                json!(1),
                json!({"session_id":"branch-source"}),
                &mut branch,
                false,
            )
            .unwrap();
        let branch_frame = serde_json::from_slice::<Value>(&branch).unwrap();
        let new_session_id = branch_frame["result"]["session_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(new_session_id.starts_with("rust-gw-"));
        assert_eq!(branch_frame["result"]["title"], json!("Sprint #3"));
        assert_eq!(branch_frame["result"]["parent"], json!("branch-source"));

        let branched_session = store.get_session(&new_session_id).unwrap().unwrap();
        assert_eq!(
            branched_session.parent_session_id,
            Some(String::from("branch-source"))
        );
        assert_eq!(branched_session.title, Some(String::from("Sprint #3")));
        assert_eq!(branched_session.model, Some(String::from("override-model")));

        let branched_messages = store.get_messages(&new_session_id).unwrap();
        assert_eq!(branched_messages.len(), 2);
        assert_eq!(branched_messages[0].content, Some(json!("question")));
        assert_eq!(branched_messages[1].content, Some(json!("answer")));
        assert_eq!(
            branched_messages[1].tool_calls,
            Some(json!([{"name":"search"}]))
        );
        assert_eq!(branched_messages[1].reasoning, Some(String::from("brief")));

        let state = server.sessions.get(&new_session_id).unwrap();
        assert_eq!(state.cwd, PathBuf::from("/tmp/demo"));
        assert_eq!(state.cols, 120);
        assert_eq!(state.overrides.provider.as_deref(), Some("demo-provider"));
    }

    #[test]
    fn native_gateway_terminal_resize_updates_session_columns() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("resize-session"),
            NativeGatewaySessionState {
                cwd: PathBuf::from("."),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides::default(),
            },
        );

        let mut resize = Vec::new();
        server
            .handle_request(
                "terminal.resize",
                json!(1),
                json!({"session_id":"resize-session","cols":132}),
                &mut resize,
                false,
            )
            .unwrap();
        let resize_frame = serde_json::from_slice::<Value>(&resize).unwrap();
        assert_eq!(resize_frame["result"]["cols"], json!(132));
        assert_eq!(server.sessions.get("resize-session").unwrap().cols, 132);
    }

    #[test]
    fn native_gateway_session_usage_reads_persisted_counters() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("usage-session"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("stored-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .record_session_usage(
                "usage-session",
                &SessionUsageDelta {
                    api_call_count: 3,
                    input_tokens: 150,
                    output_tokens: 55,
                    cache_read_tokens: 13,
                    cache_write_tokens: 7,
                    reasoning_tokens: 10,
                    estimated_cost_usd: Some(0.1334),
                    actual_cost_usd: None,
                    cost_status: Some(String::from("estimated")),
                },
            )
            .unwrap();

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("usage-session"),
            NativeGatewaySessionState {
                cwd: PathBuf::from("."),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("override-model")),
                    ..ModelOverrides::default()
                },
            },
        );

        let mut usage = Vec::new();
        server
            .handle_request(
                "session.usage",
                json!(1),
                json!({"session_id":"usage-session"}),
                &mut usage,
                false,
            )
            .unwrap();
        let usage_frame = serde_json::from_slice::<Value>(&usage).unwrap();
        assert_eq!(usage_frame["result"]["model"], json!("override-model"));
        assert_eq!(usage_frame["result"]["calls"], json!(3));
        assert_eq!(usage_frame["result"]["input"], json!(150));
        assert_eq!(usage_frame["result"]["output"], json!(55));
        assert_eq!(usage_frame["result"]["total"], json!(205));
        assert_eq!(usage_frame["result"]["cache_read"], json!(13));
        assert_eq!(usage_frame["result"]["cache_write"], json!(7));
        assert_eq!(usage_frame["result"]["cost_status"], json!("estimated"));
        assert_eq!(usage_frame["result"]["cost_usd"], json!(0.1334));
    }

    #[test]
    fn native_gateway_session_steer_queues_idle_guidance_for_next_prompt() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
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
                                "name": "write_file",
                                "arguments": "{\"path\":\"steer.txt\",\"content\":\"hello\"}"
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
                        "content": "Queued steer applied."
                    }
                }]
            })
            .to_string(),
        ]);
        store
            .create_session(&SessionCreate {
                id: String::from("steer-session"),
                source: String::from("rust-gateway"),
                user_id: None,
                model: Some(String::from("test-model")),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();

        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("steer-session"),
            NativeGatewaySessionState {
                cwd: temp.path().to_path_buf(),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("test-model")),
                    provider: Some(String::from("custom")),
                    base_url: Some(base_url),
                    api_key: Some(String::from("test-key")),
                    api_mode: Some(String::from("chat_completions")),
                    ..ModelOverrides::default()
                },
            },
        );

        let mut steer = Vec::new();
        server
            .handle_request(
                "session.steer",
                json!(1),
                json!({"session_id":"steer-session","text":"Prefer concise tool output"}),
                &mut steer,
                false,
            )
            .unwrap();
        let steer_frame = serde_json::from_slice::<Value>(&steer).unwrap();
        assert_eq!(steer_frame["result"]["status"], json!("queued"));
        assert_eq!(
            server.sessions["steer-session"].pending_steer.as_deref(),
            Some("Prefer concise tool output")
        );

        let mut output = Vec::new();
        let result = server
            .handle_prompt_submit(
                &mut output,
                json!({"session_id":"steer-session","text":"Write the file"}),
            )
            .unwrap();
        assert_eq!(result["ok"], json!(true));
        assert_eq!(server.sessions["steer-session"].pending_steer, None);

        let tool_message = store
            .get_messages("steer-session")
            .unwrap()
            .into_iter()
            .find(|message| message.tool_name.as_deref() == Some("write_file"))
            .unwrap();
        assert!(
            tool_message
                .content
                .unwrap()
                .as_str()
                .unwrap()
                .contains("User guidance: Prefer concise tool output")
        );
    }

    #[test]
    fn native_gateway_command_dispatch_handles_alias_quick_and_skill_paths() {
        let _guard = lock_mutex(&SESSION_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let _cwd = pushd(temp.path());
        fs::write(
            temp.path().join("config.yaml"),
            "quick_commands:\n  demoalias:\n    type: alias\n    target: help\n  demoexec:\n    type: exec\n    command: \"printf quick\"\n",
        )
        .unwrap();
        let skill_dir = temp.path().join("skills").join("ops").join("deploy-agent");
        fs::create_dir_all(skill_dir.join("references")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy-agent\ndescription: deploy helper\n---\n\n# Deploy\n\nUse this flow.\n",
        )
        .unwrap();
        fs::write(skill_dir.join("references").join("guide.md"), "guide").unwrap();

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let server = NativeGatewayServer::new(&context, &config, &store);

        let quick_alias = server
            .handle_command_dispatch(json!({"name":"demoalias","arg":"","session_id":null}))
            .unwrap();
        assert_eq!(quick_alias["type"], json!("alias"));
        assert_eq!(quick_alias["target"], json!("help"));

        let quick_exec = server
            .handle_command_dispatch(json!({"name":"demoexec","arg":"","session_id":null}))
            .unwrap();
        assert_eq!(quick_exec["type"], json!("exec"));
        assert_eq!(quick_exec["output"], json!("quick"));

        let skill = server
            .handle_command_dispatch(
                json!({"name":"deploy_agent","arg":"use staging","session_id":"sess-1"}),
            )
            .unwrap();
        assert_eq!(skill["type"], json!("skill"));
        assert_eq!(skill["name"], json!("deploy-agent"));
        assert!(skill["message"].as_str().unwrap().contains("use staging"));

        let snapshot = server
            .handle_command_dispatch(
                json!({"name":"snapshot","arg":"restore abc","session_id":null}),
            )
            .unwrap();
        assert_eq!(snapshot["type"], json!("exec"));
        assert!(
            snapshot["output"]
                .as_str()
                .unwrap()
                .contains("blocked in the TUI")
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn native_gateway_slash_exec_reuses_worker_and_rejects_pending_input_commands() {
        let _guard = lock_mutex(&PYTHON_ENV_LOCK);
        let temp = TempDir::new().unwrap();
        let args_log = temp.path().join("worker-args.log");
        let request_log = temp.path().join("worker-requests.log");
        let fake_python = write_fake_python_worker(temp.path(), &args_log, &request_log);
        let old_python = {
            let mut override_path = SLASH_WORKER_PYTHON_OVERRIDE.lock().unwrap();
            let old = override_path.clone();
            *override_path = Some(fake_python);
            old
        };

        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let config = context.load_config_document().unwrap();
        let store = context.open_session_store().unwrap();
        let mut server = NativeGatewayServer::new(&context, &config, &store);
        server.sessions.insert(
            String::from("slash-session"),
            NativeGatewaySessionState {
                cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                cols: 80,
                pending_steer: None,
                overrides: ModelOverrides {
                    model: Some(String::from("test-model")),
                    ..ModelOverrides::default()
                },
            },
        );

        let mut first = Vec::new();
        server
            .handle_request(
                "slash.exec",
                json!(1),
                json!({"session_id":"slash-session","command":"/help"}),
                &mut first,
                false,
            )
            .unwrap();
        let mut second = Vec::new();
        server
            .handle_request(
                "slash.exec",
                json!(2),
                json!({"session_id":"slash-session","command":"/logs"}),
                &mut second,
                false,
            )
            .unwrap();
        let mut blocked = Vec::new();
        server
            .handle_request(
                "slash.exec",
                json!(3),
                json!({"session_id":"slash-session","command":"/plan next"}),
                &mut blocked,
                false,
            )
            .unwrap();

        *SLASH_WORKER_PYTHON_OVERRIDE.lock().unwrap() = old_python;

        let first_frame = serde_json::from_slice::<Value>(&first).unwrap();
        let second_frame = serde_json::from_slice::<Value>(&second).unwrap();
        let blocked_frame = serde_json::from_slice::<Value>(&blocked).unwrap();
        assert_eq!(first_frame["result"]["output"], json!("worker:1"));
        assert_eq!(second_frame["result"]["output"], json!("worker:2"));
        assert_eq!(blocked_frame["error"]["code"], json!(4018));
        assert!(
            blocked_frame["error"]["message"]
                .as_str()
                .unwrap()
                .contains("command.dispatch")
        );

        let args_lines = fs::read_to_string(args_log)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(args_lines.len(), 1);
        assert!(args_lines[0].contains("-m"));
        assert!(args_lines[0].contains("tui_gateway.slash_worker"));
        assert!(args_lines[0].contains("--session-key"));
        assert!(args_lines[0].contains("slash-session"));

        let request_lines = fs::read_to_string(request_log)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(request_lines.len(), 2);
        assert!(
            request_lines
                .iter()
                .any(|line| line.contains("\"command\":\"/help\""))
        );
        assert!(
            request_lines
                .iter()
                .any(|line| line.contains("\"command\":\"/logs\""))
        );
        assert!(
            !request_lines
                .iter()
                .any(|line| line.contains("\"command\":\"/plan next\""))
        );
    }
}
