use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde_json::Value;

use crate::{
    AgentTurnResult, ApprovalRequest, ClarifyRequest, HermesContext, LoadedConfig, ModelOverrides,
    StepUpdate, ToolProgressUpdate, ToolRuntime,
};

#[derive(Debug)]
pub struct InteractiveTurnRequest<T> {
    pub request: T,
    response_tx: mpsc::Sender<Result<String, String>>,
}

impl<T> InteractiveTurnRequest<T> {
    pub(crate) fn new(request: T, response_tx: mpsc::Sender<Result<String, String>>) -> Self {
        Self {
            request,
            response_tx,
        }
    }

    pub fn respond(self, response: Result<String, String>) -> Result<(), String> {
        self.response_tx
            .send(response)
            .map_err(|_| String::from("Interactive turn request receiver was dropped."))
    }

    pub fn into_parts(self) -> (T, mpsc::Sender<Result<String, String>>) {
        (self.request, self.response_tx)
    }
}

#[derive(Debug)]
pub enum InteractiveTurnEvent {
    ToolProgress(ToolProgressUpdate),
    Step(StepUpdate),
    ClarifyRequest(InteractiveTurnRequest<ClarifyRequest>),
    ApprovalRequest(InteractiveTurnRequest<ApprovalRequest>),
    Final(Result<AgentTurnResult, String>),
}

#[derive(Debug, Clone)]
pub struct InteractiveTurnOptions {
    pub enable_client_requests: bool,
    pub clarify_timeout: Duration,
    pub approval_timeout: Duration,
}

impl Default for InteractiveTurnOptions {
    fn default() -> Self {
        Self {
            enable_client_requests: false,
            clarify_timeout: Duration::from_secs(300),
            approval_timeout: Duration::from_secs(300),
        }
    }
}

pub fn spawn_chat_turn_with_events(
    context: HermesContext,
    loaded: LoadedConfig,
    user_content: Value,
    runtime: ToolRuntime,
    enabled_toolsets: Vec<String>,
    overrides: ModelOverrides,
    session_id: Option<String>,
    options: InteractiveTurnOptions,
) -> mpsc::Receiver<InteractiveTurnEvent> {
    let (tx, rx) = mpsc::channel::<InteractiveTurnEvent>();
    thread::spawn(move || {
        let mut runtime = runtime
            .with_tool_progress_callback({
                let tx = tx.clone();
                move |update| {
                    let _ = tx.send(InteractiveTurnEvent::ToolProgress(update.clone()));
                }
            })
            .with_step_callback({
                let tx = tx.clone();
                move |update| {
                    let _ = tx.send(InteractiveTurnEvent::Step(update.clone()));
                }
            });

        if options.enable_client_requests {
            runtime = runtime.with_clarify_callback({
                let tx = tx.clone();
                let timeout = options.clarify_timeout;
                move |question, choices| {
                    let request = ClarifyRequest {
                        question: question.to_string(),
                        choices: choices.map(|items| items.to_vec()),
                    };
                    let (response_tx, response_rx) = mpsc::channel::<Result<String, String>>();
                    if tx
                        .send(InteractiveTurnEvent::ClarifyRequest(
                            InteractiveTurnRequest::new(request, response_tx),
                        ))
                        .is_err()
                    {
                        return Err(String::from(
                            "Interactive client disconnected before clarify input could be requested.",
                        ));
                    }
                    match response_rx.recv_timeout(timeout) {
                        Ok(response) => response,
                        Err(_) => Err(String::from(
                            "Clarification timed out before the interactive client responded.",
                        )),
                    }
                }
            });
            runtime = runtime.with_approval_callback({
                let tx = tx.clone();
                let timeout = options.approval_timeout;
                move |request| {
                    let (response_tx, response_rx) = mpsc::channel::<Result<String, String>>();
                    if tx
                        .send(InteractiveTurnEvent::ApprovalRequest(
                            InteractiveTurnRequest::new(request.clone(), response_tx),
                        ))
                        .is_err()
                    {
                        return Ok(String::from("deny"));
                    }
                    match response_rx.recv_timeout(timeout) {
                        Ok(Ok(response)) => Ok(response),
                        Ok(Err(_)) | Err(_) => Ok(String::from("deny")),
                    }
                }
            });
        }

        let store = context.open_session_store();
        let result = match store {
            Ok(store) => context
                .run_chat_turn_with_user_content(
                    &loaded,
                    user_content,
                    &runtime,
                    Some(&enabled_toolsets),
                    &overrides,
                    session_id.as_deref(),
                    Some(&store),
                )
                .map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let _ = tx.send(InteractiveTurnEvent::Final(result));
    });
    rx
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
                        if let Some((name, value)) = trimmed.split_once(':')
                            && name.trim().eq_ignore_ascii_case("content-length")
                        {
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
    fn spawned_turn_emits_progress_step_and_final_events() {
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

        let rx = spawn_chat_turn_with_events(
            context,
            loaded,
            Value::String(String::from("Create a file named notes.txt")),
            ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
            vec![String::from("hermes-cli")],
            ModelOverrides {
                model: Some(String::from("test-model")),
                provider: Some(String::from("custom")),
                base_url: Some(base_url),
                api_key: Some(String::from("test-key")),
                api_mode: Some(String::from("chat_completions")),
            },
            None,
            InteractiveTurnOptions::default(),
        );

        let mut saw_started = false;
        let mut saw_completion_step = false;
        let mut final_response = None;
        while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
            match event {
                InteractiveTurnEvent::ToolProgress(update) => {
                    if update.event_type == "tool.started"
                        && update.function_name.as_deref() == Some("write_file")
                    {
                        saw_started = true;
                    }
                }
                InteractiveTurnEvent::Step(update) => {
                    if update.iteration == 2
                        && update.prev_tools.len() == 1
                        && update.prev_tools[0].name == "write_file"
                    {
                        saw_completion_step = true;
                    }
                }
                InteractiveTurnEvent::Final(result) => {
                    final_response = Some(result.unwrap().final_response);
                    break;
                }
                InteractiveTurnEvent::ClarifyRequest(_)
                | InteractiveTurnEvent::ApprovalRequest(_) => {}
            }
        }

        assert!(saw_started);
        assert!(saw_completion_step);
        assert_eq!(
            final_response.as_deref(),
            Some("Finished writing the file.")
        );
    }

    #[test]
    fn spawned_turn_bridges_clarify_requests() {
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
                                "name": "clarify",
                                "arguments": "{\"question\":\"Choose one\",\"choices\":[\"A\",\"B\"]}"
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
                        "content": "Thanks for clarifying."
                    }
                }]
            })
            .to_string(),
        ]);

        let rx = spawn_chat_turn_with_events(
            context,
            loaded,
            Value::String(String::from("Pick between A and B")),
            ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
            vec![String::from("hermes-cli")],
            ModelOverrides {
                model: Some(String::from("test-model")),
                provider: Some(String::from("custom")),
                base_url: Some(base_url),
                api_key: Some(String::from("test-key")),
                api_mode: Some(String::from("chat_completions")),
            },
            None,
            InteractiveTurnOptions {
                enable_client_requests: true,
                clarify_timeout: Duration::from_secs(5),
                approval_timeout: Duration::from_secs(5),
            },
        );

        let mut saw_clarify = false;
        let mut final_response = None;
        while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
            match event {
                InteractiveTurnEvent::ClarifyRequest(request) => {
                    assert_eq!(request.request.question, "Choose one");
                    assert_eq!(
                        request.request.choices.as_deref(),
                        Some(&[String::from("A"), String::from("B")][..])
                    );
                    request.respond(Ok(String::from("B"))).unwrap();
                    saw_clarify = true;
                }
                InteractiveTurnEvent::Final(result) => {
                    final_response = Some(result.unwrap().final_response);
                    break;
                }
                InteractiveTurnEvent::ToolProgress(_)
                | InteractiveTurnEvent::Step(_)
                | InteractiveTurnEvent::ApprovalRequest(_) => {}
            }
        }

        assert!(saw_clarify);
        assert_eq!(final_response.as_deref(), Some("Thanks for clarifying."));
    }

    #[test]
    fn spawned_turn_defaults_approval_to_deny_on_timeout() {
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
                                "name": "terminal",
                                "arguments": "{\"command\":\"rm -rf /tmp/demo\"}"
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
                        "content": "Terminal request was denied."
                    }
                }]
            })
            .to_string(),
        ]);

        let rx = spawn_chat_turn_with_events(
            context,
            loaded,
            Value::String(String::from("Delete the demo directory")),
            ToolRuntime::new(temp.path()).with_hermes_home(temp.path()),
            vec![String::from("hermes-cli")],
            ModelOverrides {
                model: Some(String::from("test-model")),
                provider: Some(String::from("custom")),
                base_url: Some(base_url),
                api_key: Some(String::from("test-key")),
                api_mode: Some(String::from("chat_completions")),
            },
            None,
            InteractiveTurnOptions {
                enable_client_requests: true,
                clarify_timeout: Duration::from_secs(5),
                approval_timeout: Duration::from_millis(25),
            },
        );

        let mut saw_approval = false;
        let mut final_response = None;
        while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
            match event {
                InteractiveTurnEvent::ApprovalRequest(_request) => {
                    saw_approval = true;
                }
                InteractiveTurnEvent::Final(result) => {
                    final_response = Some(result.unwrap().final_response);
                    break;
                }
                InteractiveTurnEvent::ToolProgress(_)
                | InteractiveTurnEvent::Step(_)
                | InteractiveTurnEvent::ClarifyRequest(_) => {}
            }
        }

        assert!(saw_approval);
        assert_eq!(
            final_response.as_deref(),
            Some("Terminal request was denied.")
        );
    }

    #[test]
    fn spawned_turn_applies_live_steer_to_next_model_iteration() {
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
                                "name": "terminal",
                                "arguments": "{\"command\":\"sleep 0.2; printf tool-done\"}"
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
                        "content": "Steer applied."
                    }
                }]
            })
            .to_string(),
        ]);

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let runtime_control = runtime.clone();
        let rx = spawn_chat_turn_with_events(
            context.clone(),
            loaded,
            Value::String(String::from("Run a terminal command")),
            runtime,
            vec![String::from("hermes-cli")],
            ModelOverrides {
                model: Some(String::from("test-model")),
                provider: Some(String::from("custom")),
                base_url: Some(base_url),
                api_key: Some(String::from("test-key")),
                api_mode: Some(String::from("chat_completions")),
            },
            None,
            InteractiveTurnOptions::default(),
        );

        let mut final_response = None;
        let mut final_session_id = None;
        while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
            match event {
                InteractiveTurnEvent::ToolProgress(update)
                    if update.event_type == "tool.started"
                        && update.function_name.as_deref() == Some("terminal") =>
                {
                    assert!(runtime_control.steer("Prefer terse output"));
                }
                InteractiveTurnEvent::Final(result) => {
                    let result = result.unwrap();
                    final_session_id = result.session_id.clone();
                    final_response = Some(result.final_response);
                    break;
                }
                InteractiveTurnEvent::ToolProgress(_)
                | InteractiveTurnEvent::Step(_)
                | InteractiveTurnEvent::ClarifyRequest(_)
                | InteractiveTurnEvent::ApprovalRequest(_) => {}
            }
        }

        assert_eq!(final_response.as_deref(), Some("Steer applied."));
        let session_id = final_session_id.expect("session id");
        let session_store = context.open_session_store().unwrap();
        let tool_message = session_store
            .get_messages(&session_id)
            .unwrap()
            .into_iter()
            .find(|message| message.tool_name.as_deref() == Some("terminal"))
            .unwrap();
        assert!(
            tool_message
                .content
                .unwrap()
                .as_str()
                .unwrap()
                .contains("User guidance: Prefer terse output")
        );
    }
}
