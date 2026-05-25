use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use hermes_core::{
    GatewaySessionPoll, GatewayTurnSession, HermesContext, LoadedConfig, MessageRecord,
    ModelOverrides, SessionCreate, SessionRecord, SessionStore, ToolRuntime,
    spawn_chat_turn_with_events,
};
use serde_json::{Value, json};

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
    input_rx: Option<mpsc::Receiver<Option<String>>>,
    input_closed: bool,
    active_turn: Option<ActiveTurn>,
}

struct ActiveTurn {
    turn: GatewayTurnSession,
}

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
            "session.resume" => self.handle_session_resume(params),
            "input.detect_drop" => Ok(json!({
                "matched": false,
                "text": params.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

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
}
