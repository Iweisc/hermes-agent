use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    AgentTurnResult, ApprovalRequest, ClarifyRequest, StepToolRecord, StepUpdate,
    ToolProgressUpdate, ToolRuntime,
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

    pub fn on_clarify_request(&mut self, request: &ClarifyRequest) -> GatewayEventEnvelope {
        let request_id = format!("clarify-{:x}", self.next_request_id);
        self.next_request_id += 1;
        gateway_event(
            "clarify.request",
            json!({
                "question": request.question,
                "choices": request.choices,
                "request_id": request_id,
            }),
        )
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
}
