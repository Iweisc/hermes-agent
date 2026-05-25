use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    AgentTurnResult, ApprovalRequest, ClarifyRequest, InteractiveTurnEvent, StepToolRecord,
    StepUpdate, ToolProgressUpdate, ToolRuntime,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatewayEventEnvelope {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

#[derive(Debug, Clone)]
struct ActiveToolCall {
    id: String,
    arguments: Option<Value>,
    context: Option<String>,
    started_at: Instant,
}

#[derive(Debug, Default, Clone)]
pub struct GatewayEventBridge {
    next_tool_id: u64,
    next_request_id: u64,
    active_tool_calls: HashMap<String, Vec<ActiveToolCall>>,
}

impl GatewayEventBridge {
    pub fn gateway_ready(&self) -> GatewayEventEnvelope {
        GatewayEventEnvelope {
            event_type: "gateway.ready".to_string(),
            payload: None,
        }
    }

    pub fn message_start(&self) -> GatewayEventEnvelope {
        GatewayEventEnvelope {
            event_type: "message.start".to_string(),
            payload: None,
        }
    }

    pub fn on_tool_progress(&mut self, update: &ToolProgressUpdate) -> Vec<GatewayEventEnvelope> {
        let Some(name) = update
            .function_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Vec::new();
        };
        if update.event_type != "tool.started" {
            return Vec::new();
        }

        let tool_id = format!("tool-{:x}", self.next_tool_id);
        self.next_tool_id += 1;
        self.active_tool_calls
            .entry(name.to_string())
            .or_default()
            .push(ActiveToolCall {
                id: tool_id.clone(),
                arguments: update.function_args.clone(),
                context: update.preview.clone(),
                started_at: Instant::now(),
            });

        let mut start_payload = Map::new();
        start_payload.insert("tool_id".to_string(), Value::String(tool_id));
        start_payload.insert("name".to_string(), Value::String(name.to_string()));
        if let Some(context) = update.preview.as_ref() {
            start_payload.insert("context".to_string(), Value::String(context.clone()));
        }

        vec![
            gateway_event("tool.start", Value::Object(start_payload)),
            gateway_event(
                "tool.progress",
                json!({
                    "name": name,
                    "preview": update.preview.clone().unwrap_or_default(),
                }),
            ),
        ]
    }

    pub fn on_step(&mut self, update: &StepUpdate) -> Vec<GatewayEventEnvelope> {
        let mut events = Vec::new();
        for tool in &update.prev_tools {
            let active = self.pop_active_tool_call(tool);
            let mut payload = Map::new();
            payload.insert("tool_id".to_string(), Value::String(active.id));
            payload.insert("name".to_string(), Value::String(tool.name.clone()));
            if let Some(context) = active.context {
                payload.insert("context".to_string(), Value::String(context));
            }
            if let Some(arguments) = active.arguments {
                payload.insert("arguments".to_string(), arguments);
            }
            if let Some(raw_output) = tool.result.as_ref() {
                payload.insert("raw_output".to_string(), Value::String(raw_output.clone()));
            }
            payload.insert(
                "duration_s".to_string(),
                Value::from(active.started_at.elapsed().as_secs_f64()),
            );
            if let Some(error) = tool_completion_error(tool) {
                payload.insert("error".to_string(), Value::String(error));
            }
            if let Some(todos) = tool_completion_todos(tool) {
                payload.insert("todos".to_string(), todos);
            }
            if let Some(summary) = tool_completion_summary(tool) {
                payload.insert("summary".to_string(), Value::String(summary));
            }
            events.push(gateway_event("tool.complete", Value::Object(payload)));
        }
        events
    }

    pub fn prepare_clarify_request(
        &mut self,
        request: &ClarifyRequest,
    ) -> (String, GatewayEventEnvelope) {
        let request_id = format!("clarify-{:x}", self.next_request_id);
        self.next_request_id += 1;
        let event = gateway_event(
            "clarify.request",
            json!({
                "question": request.question,
                "choices": request.choices,
                "request_id": request_id,
            }),
        );
        (request_id, event)
    }

    pub fn on_clarify_request(&mut self, request: &ClarifyRequest) -> GatewayEventEnvelope {
        self.prepare_clarify_request(request).1
    }

    pub fn on_approval_request(&self, request: &ApprovalRequest) -> GatewayEventEnvelope {
        gateway_event(
            "approval.request",
            json!({
                "command": request.command,
                "description": request.description,
                "pattern_keys": request.pattern_keys,
                "choices": request.choices,
                "allow_permanent": request.allow_permanent,
            }),
        )
    }

    pub fn on_final_response(&self, result: &AgentTurnResult) -> GatewayEventEnvelope {
        gateway_event(
            "message.complete",
            json!({
                "text": result.final_response,
                "reasoning": result.reasoning,
                "api_calls": result.api_calls,
                "tool_calls": result.tool_calls,
                "model": result.model,
                "provider": result.provider,
                "base_url": result.base_url,
                "session_id": result.session_id,
            }),
        )
    }

    fn pop_active_tool_call(&mut self, tool: &StepToolRecord) -> ActiveToolCall {
        if let Some(calls) = self.active_tool_calls.get_mut(&tool.name)
            && !calls.is_empty()
        {
            let active = calls.remove(0);
            if calls.is_empty() {
                self.active_tool_calls.remove(&tool.name);
            }
            return active;
        }

        let tool_id = format!("tool-{:x}", self.next_tool_id);
        self.next_tool_id += 1;
        ActiveToolCall {
            id: tool_id,
            arguments: tool
                .arguments
                .as_deref()
                .and_then(|value| serde_json::from_str::<Value>(value).ok()),
            context: None,
            started_at: Instant::now(),
        }
    }
}

#[derive(Debug)]
pub enum GatewayTurnOutcome {
    Events(Vec<GatewayEventEnvelope>),
    Final {
        result: Result<AgentTurnResult, String>,
        events: Vec<GatewayEventEnvelope>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayClarifyPrompt {
    pub request_id: String,
    pub question: String,
    pub choices: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayApprovalPrompt {
    pub request: ApprovalRequest,
}

#[derive(Debug)]
pub enum GatewaySessionPoll {
    Events {
        events: Vec<GatewayEventEnvelope>,
        clarify_requests: Vec<GatewayClarifyPrompt>,
        approval_requests: Vec<GatewayApprovalPrompt>,
    },
    Final {
        result: Result<AgentTurnResult, String>,
        events: Vec<GatewayEventEnvelope>,
    },
}

#[derive(Debug, Default)]
pub struct GatewayTurnBridge {
    bridge: GatewayEventBridge,
    pending_clarify: HashMap<String, mpsc::Sender<Result<String, String>>>,
    pending_approval: VecDeque<mpsc::Sender<Result<String, String>>>,
}

impl GatewayTurnBridge {
    pub fn gateway_ready(&self) -> GatewayEventEnvelope {
        self.bridge.gateway_ready()
    }

    pub fn message_start(&self) -> GatewayEventEnvelope {
        self.bridge.message_start()
    }

    pub fn handle_event(&mut self, event: InteractiveTurnEvent) -> GatewayTurnOutcome {
        match event {
            InteractiveTurnEvent::ToolProgress(update) => {
                GatewayTurnOutcome::Events(self.bridge.on_tool_progress(&update))
            }
            InteractiveTurnEvent::Step(update) => {
                GatewayTurnOutcome::Events(self.bridge.on_step(&update))
            }
            InteractiveTurnEvent::ClarifyRequest(request) => {
                let (request, response_tx) = request.into_parts();
                let (request_id, event) = self.bridge.prepare_clarify_request(&request);
                self.pending_clarify.insert(request_id, response_tx);
                GatewayTurnOutcome::Events(vec![event])
            }
            InteractiveTurnEvent::ApprovalRequest(request) => {
                let (request, response_tx) = request.into_parts();
                self.pending_approval.push_back(response_tx);
                GatewayTurnOutcome::Events(vec![self.bridge.on_approval_request(&request)])
            }
            InteractiveTurnEvent::Final(result) => {
                let events = match &result {
                    Ok(result) => vec![self.bridge.on_final_response(result)],
                    Err(_) => Vec::new(),
                };
                GatewayTurnOutcome::Final { result, events }
            }
        }
    }

    pub fn respond_clarify(
        &mut self,
        request_id: &str,
        answer: impl Into<String>,
    ) -> Result<(), String> {
        self.resolve_clarify(request_id, Ok(answer.into()))
    }

    pub fn resolve_clarify(
        &mut self,
        request_id: &str,
        response: Result<String, String>,
    ) -> Result<(), String> {
        let Some(response_tx) = self.pending_clarify.remove(request_id) else {
            return Err(format!(
                "No pending clarify request found for '{request_id}'."
            ));
        };
        response_tx
            .send(response)
            .map_err(|_| String::from("Clarify request receiver was dropped."))
    }

    pub fn respond_approval(&mut self, choice: impl Into<String>) -> Result<(), String> {
        self.resolve_approval(Ok(choice.into()))
    }

    pub fn resolve_approval(&mut self, response: Result<String, String>) -> Result<(), String> {
        let Some(response_tx) = self.pending_approval.pop_front() else {
            return Err(String::from("No pending approval request found."));
        };
        response_tx
            .send(response)
            .map_err(|_| String::from("Approval request receiver was dropped."))
    }

    pub fn pending_clarify_count(&self) -> usize {
        self.pending_clarify.len()
    }

    pub fn pending_approval_count(&self) -> usize {
        self.pending_approval.len()
    }

    pub fn cancel_pending_requests(&mut self, reason: &str) {
        let reason = reason.to_string();
        for (_, response_tx) in self.pending_clarify.drain() {
            let _ = response_tx.send(Err(reason.clone()));
        }
        while let Some(response_tx) = self.pending_approval.pop_front() {
            let _ = response_tx.send(Ok(String::from("deny")));
        }
    }
}

pub struct GatewayTurnSession {
    bridge: GatewayTurnBridge,
    rx: mpsc::Receiver<InteractiveTurnEvent>,
}

impl GatewayTurnSession {
    pub fn new(rx: mpsc::Receiver<InteractiveTurnEvent>) -> Self {
        Self {
            bridge: GatewayTurnBridge::default(),
            rx,
        }
    }

    pub fn gateway_ready(&self) -> GatewayEventEnvelope {
        self.bridge.gateway_ready()
    }

    pub fn message_start(&self) -> GatewayEventEnvelope {
        self.bridge.message_start()
    }

    pub fn poll_next(&mut self) -> Result<GatewaySessionPoll, String> {
        let event = self
            .rx
            .recv()
            .map_err(|_| String::from("gateway event runner disconnected"))?;
        self.handle_event(event)
    }

    pub fn handle_event(
        &mut self,
        event: InteractiveTurnEvent,
    ) -> Result<GatewaySessionPoll, String> {
        match self.bridge.handle_event(event) {
            GatewayTurnOutcome::Events(events) => Ok(GatewaySessionPoll::Events {
                clarify_requests: collect_clarify_requests(&events)?,
                approval_requests: collect_approval_requests(&events)?,
                events,
            }),
            GatewayTurnOutcome::Final { result, events } => {
                Ok(GatewaySessionPoll::Final { result, events })
            }
        }
    }

    pub fn resolve_clarify(
        &mut self,
        request_id: &str,
        response: Result<String, String>,
    ) -> Result<(), String> {
        self.bridge.resolve_clarify(request_id, response)
    }

    pub fn resolve_approval(&mut self, response: Result<String, String>) -> Result<(), String> {
        self.bridge.resolve_approval(response)
    }

    pub fn cancel_pending_requests(&mut self, reason: &str) {
        self.bridge.cancel_pending_requests(reason);
    }
}

pub fn attach_gateway_event_callbacks<F>(
    runtime: ToolRuntime,
    bridge: Arc<Mutex<GatewayEventBridge>>,
    emit: F,
) -> ToolRuntime
where
    F: Fn(&GatewayEventEnvelope) + Send + Sync + 'static,
{
    let emit = Arc::new(emit);
    runtime
        .with_tool_progress_callback({
            let bridge = Arc::clone(&bridge);
            let emit = Arc::clone(&emit);
            move |update| {
                let events = bridge.lock().unwrap().on_tool_progress(update);
                for event in events {
                    emit(&event);
                }
            }
        })
        .with_step_callback({
            let bridge = Arc::clone(&bridge);
            let emit = Arc::clone(&emit);
            move |update| {
                let events = bridge.lock().unwrap().on_step(update);
                for event in events {
                    emit(&event);
                }
            }
        })
        .with_clarify_request_callback({
            let bridge = Arc::clone(&bridge);
            let emit = Arc::clone(&emit);
            move |request| emit(&bridge.lock().unwrap().on_clarify_request(request))
        })
        .with_approval_request_callback({
            let bridge = Arc::clone(&bridge);
            move |request| emit(&bridge.lock().unwrap().on_approval_request(request))
        })
}

fn collect_clarify_requests(
    events: &[GatewayEventEnvelope],
) -> Result<Vec<GatewayClarifyPrompt>, String> {
    let mut prompts = Vec::new();
    for event in events {
        if event.event_type != "clarify.request" {
            continue;
        }
        let payload = event
            .payload
            .as_ref()
            .and_then(Value::as_object)
            .ok_or_else(|| String::from("clarify.request payload was missing or invalid"))?;
        let request_id = payload
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| String::from("clarify.request payload was missing request_id"))?
            .to_string();
        let question = payload
            .get("question")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| String::from("clarify.request payload was missing question"))?
            .to_string();
        let choices = payload
            .get("choices")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            });
        prompts.push(GatewayClarifyPrompt {
            request_id,
            question,
            choices,
        });
    }
    Ok(prompts)
}

fn collect_approval_requests(
    events: &[GatewayEventEnvelope],
) -> Result<Vec<GatewayApprovalPrompt>, String> {
    let mut prompts = Vec::new();
    for event in events {
        if event.event_type != "approval.request" {
            continue;
        }
        let payload = event
            .payload
            .as_ref()
            .and_then(Value::as_object)
            .ok_or_else(|| String::from("approval.request payload was missing or invalid"))?;
        let command = payload
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| String::from("approval.request payload was missing command"))?
            .to_string();
        let description = payload
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let pattern_keys = payload
            .get("pattern_keys")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let choices = payload
            .get("choices")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let allow_permanent = payload
            .get("allow_permanent")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        prompts.push(GatewayApprovalPrompt {
            request: ApprovalRequest {
                command,
                description,
                pattern_keys,
                choices,
                allow_permanent,
            },
        });
    }
    Ok(prompts)
}

fn gateway_event(event_type: &str, payload: Value) -> GatewayEventEnvelope {
    GatewayEventEnvelope {
        event_type: event_type.to_string(),
        payload: Some(payload),
    }
}

fn tool_completion_summary(tool: &StepToolRecord) -> Option<String> {
    if let Some(result) = tool.result.as_deref() {
        if let Ok(parsed) = serde_json::from_str::<Value>(result) {
            for key in ["error", "message", "summary", "output", "content", "path"] {
                if let Some(text) = parsed
                    .get(key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    return Some(truncate_chars(text, 160));
                }
            }
        }
        let trimmed = result.trim();
        if !trimmed.is_empty() {
            return Some(truncate_chars(trimmed, 160));
        }
    }
    Some(format!("{} completed", tool.name))
}

fn tool_completion_error(tool: &StepToolRecord) -> Option<String> {
    let result = tool.result.as_deref()?;
    let parsed = serde_json::from_str::<Value>(result).ok()?;
    parsed
        .get("error")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn tool_completion_todos(tool: &StepToolRecord) -> Option<Value> {
    let result = tool.result.as_deref()?;
    let parsed = serde_json::from_str::<Value>(result).ok()?;
    parsed
        .get("todos")
        .cloned()
        .filter(|value| value.is_array())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn gateway_event_bridge_pairs_started_and_completed_tool_calls() {
        let mut bridge = GatewayEventBridge::default();
        let started = bridge.on_tool_progress(&ToolProgressUpdate {
            event_type: String::from("tool.started"),
            function_name: Some(String::from("write_file")),
            preview: Some(String::from("notes.txt")),
            function_args: Some(json!({"path": "notes.txt"})),
            duration_ms: None,
            is_error: None,
        });
        assert_eq!(started.len(), 2);
        assert_eq!(started[0].event_type, "tool.start");
        assert_eq!(started[1].event_type, "tool.progress");
        let tool_id = started[0].payload.as_ref().unwrap()["tool_id"]
            .as_str()
            .unwrap()
            .to_string();

        let completed = bridge.on_step(&StepUpdate {
            iteration: 2,
            prev_tools: vec![StepToolRecord {
                name: String::from("write_file"),
                result: Some(String::from(
                    "{\"success\":true,\"path\":\"notes.txt\",\"bytes_written\":4}",
                )),
                arguments: Some(String::from("{\"path\":\"notes.txt\"}")),
            }],
        });
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].event_type, "tool.complete");
        assert_eq!(
            completed[0].payload.as_ref().unwrap()["tool_id"],
            json!(tool_id)
        );
        assert_eq!(
            completed[0].payload.as_ref().unwrap()["context"],
            json!("notes.txt")
        );
        assert_eq!(
            completed[0].payload.as_ref().unwrap()["raw_output"],
            json!("{\"success\":true,\"path\":\"notes.txt\",\"bytes_written\":4}")
        );
        assert!(
            completed[0].payload.as_ref().unwrap()["duration_s"]
                .as_f64()
                .unwrap()
                >= 0.0
        );
    }

    #[test]
    fn gateway_event_bridge_emits_interactive_and_final_events() {
        let mut bridge = GatewayEventBridge::default();
        let ready = bridge.gateway_ready();
        assert_eq!(ready.event_type, "gateway.ready");
        assert!(ready.payload.is_none());

        let started = bridge.message_start();
        assert_eq!(started.event_type, "message.start");
        assert!(started.payload.is_none());

        let clarify = bridge.on_clarify_request(&ClarifyRequest {
            question: String::from("Choose"),
            choices: Some(vec![String::from("A"), String::from("B")]),
        });
        assert_eq!(clarify.event_type, "clarify.request");
        assert_eq!(
            clarify.payload.as_ref().unwrap()["question"],
            json!("Choose")
        );
        assert_eq!(
            clarify.payload.as_ref().unwrap()["request_id"],
            json!("clarify-0")
        );

        let approval = bridge.on_approval_request(&ApprovalRequest {
            command: String::from("rm -rf /tmp/demo"),
            description: String::from("dangerous command"),
            pattern_keys: vec![String::from("delete")],
            choices: vec![String::from("once"), String::from("deny")],
            allow_permanent: false,
        });
        assert_eq!(approval.event_type, "approval.request");
        assert_eq!(
            approval.payload.as_ref().unwrap()["command"],
            json!("rm -rf /tmp/demo")
        );

        let complete = bridge.on_final_response(&AgentTurnResult {
            final_response: String::from("done"),
            reasoning: Some(String::from("thought process")),
            api_calls: 2,
            tool_calls: 1,
            model: String::from("test-model"),
            provider: String::from("custom"),
            base_url: String::from("http://localhost"),
            session_id: Some(String::from("session_123")),
            completed: true,
            interrupted: false,
            turn_exit_reason: String::from("completed"),
        });
        assert_eq!(complete.event_type, "message.complete");
        assert_eq!(complete.payload.as_ref().unwrap()["text"], json!("done"));
        assert_eq!(
            complete.payload.as_ref().unwrap()["reasoning"],
            json!("thought process")
        );
        assert_eq!(complete.payload.as_ref().unwrap()["api_calls"], json!(2));
    }

    #[test]
    fn gateway_event_bridge_carries_todo_and_error_details() {
        let mut bridge = GatewayEventBridge::default();
        let _ = bridge.on_tool_progress(&ToolProgressUpdate {
            event_type: String::from("tool.started"),
            function_name: Some(String::from("todo")),
            preview: Some(String::from("update todos")),
            function_args: Some(json!({
                "todos": [{"id": "a", "content": "task", "status": "in_progress"}]
            })),
            duration_ms: None,
            is_error: None,
        });
        let events = bridge.on_step(&StepUpdate {
            iteration: 2,
            prev_tools: vec![StepToolRecord {
                name: String::from("todo"),
                result: Some(String::from(
                    "{\"error\":\"validation failed\",\"todos\":[{\"id\":\"a\",\"content\":\"task\",\"status\":\"pending\"}]}",
                )),
                arguments: None,
            }],
        });
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].payload.as_ref().unwrap()["error"],
            json!("validation failed")
        );
        assert_eq!(
            events[0].payload.as_ref().unwrap()["todos"],
            json!([{"id":"a","content":"task","status":"pending"}])
        );
    }

    #[test]
    fn gateway_turn_bridge_routes_interactive_responses() {
        let mut bridge = GatewayTurnBridge::default();

        let (clarify_tx, clarify_rx) = mpsc::channel();
        let clarify = bridge.handle_event(InteractiveTurnEvent::ClarifyRequest(
            crate::InteractiveTurnRequest::new(
                ClarifyRequest {
                    question: String::from("Choose"),
                    choices: Some(vec![String::from("A"), String::from("B")]),
                },
                clarify_tx,
            ),
        ));
        let GatewayTurnOutcome::Events(clarify_events) = clarify else {
            panic!("expected clarify events");
        };
        assert_eq!(clarify_events.len(), 1);
        assert_eq!(clarify_events[0].event_type, "clarify.request");
        let request_id = clarify_events[0].payload.as_ref().unwrap()["request_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(bridge.pending_clarify_count(), 1);
        bridge.respond_clarify(&request_id, "B").unwrap();
        assert_eq!(clarify_rx.recv().unwrap(), Ok(String::from("B")));
        assert_eq!(bridge.pending_clarify_count(), 0);

        let (approval_tx, approval_rx) = mpsc::channel();
        let approval = bridge.handle_event(InteractiveTurnEvent::ApprovalRequest(
            crate::InteractiveTurnRequest::new(
                ApprovalRequest {
                    command: String::from("rm -rf /tmp/demo"),
                    description: String::from("dangerous command"),
                    pattern_keys: vec![String::from("delete")],
                    choices: vec![String::from("once"), String::from("deny")],
                    allow_permanent: false,
                },
                approval_tx,
            ),
        ));
        let GatewayTurnOutcome::Events(approval_events) = approval else {
            panic!("expected approval events");
        };
        assert_eq!(approval_events.len(), 1);
        assert_eq!(approval_events[0].event_type, "approval.request");
        assert_eq!(bridge.pending_approval_count(), 1);
        bridge.respond_approval("once").unwrap();
        assert_eq!(approval_rx.recv().unwrap(), Ok(String::from("once")));
        assert_eq!(bridge.pending_approval_count(), 0);
    }

    #[test]
    fn gateway_turn_bridge_emits_tool_and_final_events() {
        let mut bridge = GatewayTurnBridge::default();
        let start = bridge.handle_event(InteractiveTurnEvent::ToolProgress(ToolProgressUpdate {
            event_type: String::from("tool.started"),
            function_name: Some(String::from("write_file")),
            preview: Some(String::from("notes.txt")),
            function_args: Some(json!({"path": "notes.txt"})),
            duration_ms: None,
            is_error: None,
        }));
        let GatewayTurnOutcome::Events(start_events) = start else {
            panic!("expected tool events");
        };
        assert_eq!(start_events.len(), 2);
        assert_eq!(start_events[0].event_type, "tool.start");

        let complete = bridge.handle_event(InteractiveTurnEvent::Step(StepUpdate {
            iteration: 2,
            prev_tools: vec![StepToolRecord {
                name: String::from("write_file"),
                result: Some(String::from("{\"success\":true,\"path\":\"notes.txt\"}")),
                arguments: Some(String::from("{\"path\":\"notes.txt\"}")),
            }],
        }));
        let GatewayTurnOutcome::Events(complete_events) = complete else {
            panic!("expected completion events");
        };
        assert_eq!(complete_events.len(), 1);
        assert_eq!(complete_events[0].event_type, "tool.complete");

        let final_outcome = bridge.handle_event(InteractiveTurnEvent::Final(Ok(AgentTurnResult {
            final_response: String::from("done"),
            reasoning: Some(String::from("thought process")),
            api_calls: 1,
            tool_calls: 1,
            model: String::from("test-model"),
            provider: String::from("custom"),
            base_url: String::from("http://localhost"),
            session_id: Some(String::from("session_123")),
            completed: true,
            interrupted: false,
            turn_exit_reason: String::from("completed"),
        })));
        let GatewayTurnOutcome::Final { result, events } = final_outcome else {
            panic!("expected final outcome");
        };
        assert_eq!(result.unwrap().final_response, "done");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message.complete");
        assert_eq!(
            events[0].payload.as_ref().unwrap()["reasoning"],
            json!("thought process")
        );
    }

    #[test]
    fn gateway_turn_session_yields_typed_prompts_and_final_events() {
        let (event_tx, event_rx) = mpsc::channel();
        let mut session = GatewayTurnSession::new(event_rx);

        let (clarify_response_tx, clarify_response_rx) = mpsc::channel();
        event_tx
            .send(InteractiveTurnEvent::ClarifyRequest(
                crate::InteractiveTurnRequest::new(
                    ClarifyRequest {
                        question: String::from("Choose"),
                        choices: Some(vec![String::from("A"), String::from("B")]),
                    },
                    clarify_response_tx,
                ),
            ))
            .unwrap();

        let poll = session.poll_next().unwrap();
        let GatewaySessionPoll::Events {
            events,
            clarify_requests,
            approval_requests,
        } = poll
        else {
            panic!("expected typed event poll");
        };
        assert_eq!(events.len(), 1);
        assert!(approval_requests.is_empty());
        assert_eq!(clarify_requests.len(), 1);
        assert_eq!(clarify_requests[0].question, "Choose");
        session
            .resolve_clarify(&clarify_requests[0].request_id, Ok(String::from("B")))
            .unwrap();
        assert_eq!(clarify_response_rx.recv().unwrap(), Ok(String::from("B")));

        event_tx
            .send(InteractiveTurnEvent::Final(Ok(AgentTurnResult {
                final_response: String::from("done"),
                reasoning: None,
                api_calls: 1,
                tool_calls: 0,
                model: String::from("test-model"),
                provider: String::from("custom"),
                base_url: String::from("http://localhost"),
                session_id: None,
                completed: true,
                interrupted: false,
                turn_exit_reason: String::from("completed"),
            })))
            .unwrap();

        let poll = session.poll_next().unwrap();
        let GatewaySessionPoll::Final { result, events } = poll else {
            panic!("expected typed final poll");
        };
        assert_eq!(result.unwrap().final_response, "done");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message.complete");
    }

    #[test]
    fn gateway_turn_bridge_cancels_pending_requests() {
        let mut bridge = GatewayTurnBridge::default();

        let (clarify_tx, clarify_rx) = mpsc::channel();
        let clarify = bridge.handle_event(InteractiveTurnEvent::ClarifyRequest(
            crate::InteractiveTurnRequest::new(
                ClarifyRequest {
                    question: String::from("Choose"),
                    choices: Some(vec![String::from("A"), String::from("B")]),
                },
                clarify_tx,
            ),
        ));
        let GatewayTurnOutcome::Events(_) = clarify else {
            panic!("expected clarify events");
        };

        let (approval_tx, approval_rx) = mpsc::channel();
        let approval = bridge.handle_event(InteractiveTurnEvent::ApprovalRequest(
            crate::InteractiveTurnRequest::new(
                ApprovalRequest {
                    command: String::from("rm -rf /tmp/demo"),
                    description: String::from("dangerous command"),
                    pattern_keys: vec![String::from("delete")],
                    choices: vec![String::from("once"), String::from("deny")],
                    allow_permanent: false,
                },
                approval_tx,
            ),
        ));
        let GatewayTurnOutcome::Events(_) = approval else {
            panic!("expected approval events");
        };

        bridge.cancel_pending_requests("interrupted");
        assert_eq!(bridge.pending_clarify_count(), 0);
        assert_eq!(bridge.pending_approval_count(), 0);
        assert_eq!(clarify_rx.recv().unwrap(), Err(String::from("interrupted")));
        assert_eq!(approval_rx.recv().unwrap(), Ok(String::from("deny")));
    }
}
