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

use hermes_core::{
    GatewaySessionPoll, GatewayTurnSession, HermesContext, LoadedConfig, MessageRecord,
    ModelOverrides, SessionCreate, SessionRecord, SessionStore, ToolRuntime,
    shell_command_block_reason, spawn_chat_turn_with_events,
};
use regex::Regex;
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;

use crate::python_bridge::{project_root, resolve_repo_python};

#[derive(Debug, Clone)]
struct NativeGatewaySessionState {
    cwd: PathBuf,
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
                "clarify.respond" | "approval.respond" | "session.interrupt"
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
            "input.detect_drop" => Ok(json!({
                "matched": false,
                "text": params.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            "commands.catalog" => self.handle_commands_catalog(),
            "config.get" => self.handle_config_get(params),
            "complete.path" => self.handle_complete_path(params),
            "complete.slash" => self.handle_complete_slash(params),
            "shell.exec" => self.handle_shell_exec(params),
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
        _params: Value,
    ) -> Result<Value, (i64, String, Option<Value>)> {
        let session_id = format!("rust-gw-{:x}", unix_ts_nanos());
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
        let state = self
            .sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| (-32004, String::from("session not found"), None))?;
        let rx = spawn_chat_turn_with_events(
            self.context.clone(),
            self.config.clone(),
            Value::String(text),
            ToolRuntime::new(&state.cwd)
                .with_hermes_home(self.context.hermes_home())
                .with_current_session_id(Some(session_id.clone())),
            self.config.config.toolsets.clone(),
            state.overrides.clone(),
            Some(session_id.clone()),
            hermes_core::InteractiveTurnOptions {
                enable_client_requests: true,
                ..hermes_core::InteractiveTurnOptions::default()
            },
        );
        self.active_turn = Some(ActiveTurn {
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
        match key {
            "full" => Ok(json!({
                "config": serde_json::to_value(&self.config.raw).unwrap_or_else(|_| json!({})),
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
            _ => Err((-32602, format!("unsupported config key: {key}"), None)),
        }
    }
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

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    static PYTHON_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    #[cfg(not(windows))]
    #[test]
    fn native_gateway_slash_exec_reuses_worker_and_rejects_pending_input_commands() {
        let _guard = PYTHON_ENV_LOCK.lock().unwrap();
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
