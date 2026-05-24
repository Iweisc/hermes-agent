use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::thread::sleep;
use std::time::{Duration, Instant};

use hermes_core::{HermesContext, MessageRecord, ToolRuntime, dispatch_tool};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

const MCP_PROTOCOL_VERSION: &str = "2025-03-26";
const QUEUE_LIMIT: usize = 1000;
const DEFAULT_EVENT_LIMIT: usize = 20;
const DEFAULT_CONVERSATION_LIMIT: usize = 50;
const DEFAULT_MESSAGE_LIMIT: usize = 50;
const POLL_INTERVAL: Duration = Duration::from_millis(200);
const MAX_WAIT_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
struct QueueEvent {
    cursor: u64,
    kind: String,
    session_key: String,
    data: JsonMap<String, JsonValue>,
}

#[derive(Debug, Default)]
struct EventBridge {
    queue: VecDeque<QueueEvent>,
    cursor: u64,
    last_seen_message_ids: BTreeMap<String, i64>,
    pending_approvals: BTreeMap<String, JsonValue>,
}

pub fn run_mcp_stdio(context: &HermesContext, verbose: bool) -> Result<(), Box<dyn Error>> {
    if verbose {
        eprintln!("hermes mcp serve: native Rust stdio server");
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_mcp_jsonrpc(context, stdin.lock(), stdout.lock())
}

pub(crate) fn run_mcp_jsonrpc<R: BufRead, W: Write>(
    context: &HermesContext,
    mut input: R,
    mut output: W,
) -> Result<(), Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| context.home_dir().to_path_buf());
    let runtime = ToolRuntime::new(cwd).with_hermes_home(context.hermes_home());
    let mut server = NativeMcpServer {
        context: context.clone(),
        runtime,
        bridge: EventBridge::default(),
    };

    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let frame = match serde_json::from_str::<JsonValue>(trimmed) {
            Ok(value) => value,
            Err(error) => {
                write_jsonrpc_error(
                    &mut output,
                    JsonValue::Null,
                    -32700,
                    "Parse error",
                    Some(json!({ "detail": error.to_string() })),
                )?;
                continue;
            }
        };
        server.handle_frame(frame, &mut output)?;
    }
    Ok(())
}

struct NativeMcpServer {
    context: HermesContext,
    runtime: ToolRuntime,
    bridge: EventBridge,
}

impl NativeMcpServer {
    fn handle_frame<W: Write>(
        &mut self,
        frame: JsonValue,
        output: &mut W,
    ) -> Result<(), Box<dyn Error>> {
        let Some(object) = frame.as_object() else {
            write_jsonrpc_error(
                output,
                JsonValue::Null,
                -32600,
                "Invalid Request",
                Some(json!({ "detail": "JSON-RPC frame must be an object" })),
            )?;
            return Ok(());
        };
        let id = object.get("id").cloned();
        let method = object.get("method").and_then(JsonValue::as_str);
        let params = object.get("params").cloned().unwrap_or(JsonValue::Null);

        let Some(method) = method else {
            if let Some(id) = id {
                write_jsonrpc_error(output, id, -32600, "Invalid Request", None)?;
            }
            return Ok(());
        };

        let Some(id) = id else {
            self.handle_notification(method, params);
            return Ok(());
        };

        match self.handle_request(method, params) {
            Ok(result) => write_jsonrpc_result(output, id, result)?,
            Err((code, message, data)) => write_jsonrpc_error(output, id, code, &message, data)?,
        }
        Ok(())
    }

    fn handle_notification(&mut self, _method: &str, _params: JsonValue) {}

    fn handle_request(
        &mut self,
        method: &str,
        params: JsonValue,
    ) -> Result<JsonValue, (i64, String, Option<JsonValue>)> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "hermes",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_specs() })),
            "tools/call" => self.handle_tool_call(params),
            other => Err((
                -32601,
                String::from("Method not found"),
                Some(json!({ "method": other })),
            )),
        }
    }

    fn handle_tool_call(
        &mut self,
        params: JsonValue,
    ) -> Result<JsonValue, (i64, String, Option<JsonValue>)> {
        let name = params
            .get("name")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| (-32602, String::from("Missing tool name"), None))?;
        let arguments = params
            .get("arguments")
            .cloned()
            .filter(JsonValue::is_object)
            .unwrap_or_else(|| json!({}));

        let text = self.call_tool(name, &arguments);
        let is_error = serde_json::from_str::<JsonValue>(&text)
            .ok()
            .and_then(|value| value.get("error").cloned())
            .is_some();
        Ok(json!({
            "content": [{ "type": "text", "text": text }],
            "isError": is_error,
        }))
    }

    fn call_tool(&mut self, name: &str, args: &JsonValue) -> String {
        match name {
            "conversations_list" => self.conversations_list(args),
            "conversation_get" => self.conversation_get(args),
            "messages_read" => self.messages_read(args),
            "attachments_fetch" => self.attachments_fetch(args),
            "events_poll" => self.events_poll(args),
            "events_wait" => self.events_wait(args),
            "messages_send" => self.messages_send(args),
            "channels_list" => self.channels_list(args),
            "permissions_list_open" => self.permissions_list_open(),
            "permissions_respond" => self.permissions_respond(args),
            _ => json!({ "error": format!("Unknown tool: {name}") }).to_string(),
        }
    }

    fn conversations_list(&self, args: &JsonValue) -> String {
        let platform = arg_string(args, "platform").map(|value| value.to_ascii_lowercase());
        let search = arg_string(args, "search").map(|value| value.to_ascii_lowercase());
        let limit = arg_usize(args, "limit", DEFAULT_CONVERSATION_LIMIT);
        let entries = load_sessions_index(&self.context);

        let mut conversations = entries
            .into_iter()
            .filter_map(|(key, entry)| {
                let origin = entry.get("origin").and_then(JsonValue::as_object);
                let entry_platform = entry_string(&entry, "platform")
                    .or_else(|| nested_entry_string(origin, "platform"))
                    .unwrap_or_default();
                if let Some(platform) = platform.as_deref()
                    && entry_platform.to_ascii_lowercase() != platform
                {
                    return None;
                }
                let display_name = entry_string(&entry, "display_name").unwrap_or_default();
                let chat_name = nested_entry_string(origin, "chat_name").unwrap_or_default();
                if let Some(search) = search.as_deref() {
                    let haystack = format!("{display_name}\n{chat_name}\n{key}").to_ascii_lowercase();
                    if !haystack.contains(search) {
                        return None;
                    }
                }
                Some(json!({
                    "session_key": key,
                    "session_id": entry_string(&entry, "session_id").unwrap_or_default(),
                    "platform": entry_platform,
                    "chat_type": entry_string(&entry, "chat_type")
                        .or_else(|| nested_entry_string(origin, "chat_type"))
                        .unwrap_or_default(),
                    "display_name": display_name,
                    "chat_name": chat_name,
                    "user_name": nested_entry_string(origin, "user_name").unwrap_or_default(),
                    "updated_at": entry.get("updated_at").cloned().unwrap_or(JsonValue::String(String::new())),
                }))
            })
            .collect::<Vec<_>>();

        conversations.sort_by(|left, right| {
            value_sort_key(right.get("updated_at")).cmp(&value_sort_key(left.get("updated_at")))
        });
        conversations.truncate(limit);
        json!({
            "count": conversations.len(),
            "conversations": conversations,
        })
        .to_string()
    }

    fn conversation_get(&self, args: &JsonValue) -> String {
        let Some(session_key) = arg_string(args, "session_key") else {
            return json!({ "error": "session_key is required" }).to_string();
        };
        let entries = load_sessions_index(&self.context);
        let Some(entry) = entries.get(&session_key) else {
            return json!({ "error": format!("Conversation not found: {session_key}") })
                .to_string();
        };
        let origin = entry.get("origin").and_then(JsonValue::as_object);
        json!({
            "session_key": session_key,
            "session_id": entry_string(entry, "session_id").unwrap_or_default(),
            "platform": entry_string(entry, "platform")
                .or_else(|| nested_entry_string(origin, "platform"))
                .unwrap_or_default(),
            "chat_type": entry_string(entry, "chat_type")
                .or_else(|| nested_entry_string(origin, "chat_type"))
                .unwrap_or_default(),
            "display_name": entry_string(entry, "display_name").unwrap_or_default(),
            "user_name": nested_entry_string(origin, "user_name").unwrap_or_default(),
            "chat_name": nested_entry_string(origin, "chat_name").unwrap_or_default(),
            "chat_id": nested_entry_string(origin, "chat_id").unwrap_or_default(),
            "thread_id": origin
                .and_then(|origin| origin.get("thread_id"))
                .cloned()
                .unwrap_or(JsonValue::Null),
            "updated_at": entry.get("updated_at").cloned().unwrap_or(JsonValue::String(String::new())),
            "created_at": entry.get("created_at").cloned().unwrap_or(JsonValue::String(String::new())),
            "input_tokens": entry.get("input_tokens").cloned().unwrap_or(json!(0)),
            "output_tokens": entry.get("output_tokens").cloned().unwrap_or(json!(0)),
            "total_tokens": entry.get("total_tokens").cloned().unwrap_or(json!(0)),
        })
        .to_string()
    }

    fn messages_read(&self, args: &JsonValue) -> String {
        let Some((session_key, session_id)) = self.resolve_session_key(args) else {
            return json!({ "error": "Conversation not found or missing session_id" }).to_string();
        };
        let limit = arg_usize(args, "limit", DEFAULT_MESSAGE_LIMIT);
        let messages = match self.context.open_session_store() {
            Ok(store) => match store.get_messages(&session_id) {
                Ok(messages) => messages,
                Err(error) => {
                    return json!({ "error": format!("Failed to read messages: {error}") })
                        .to_string();
                }
            },
            Err(error) => {
                return json!({ "error": format!("Session database unavailable: {error}") })
                    .to_string();
            }
        };
        let mut filtered = messages
            .iter()
            .filter(|message| message.role == "user" || message.role == "assistant")
            .filter_map(|message| {
                let content = extract_message_content(message)?;
                Some(json!({
                    "id": message.id.to_string(),
                    "role": message.role,
                    "content": truncate_chars(&content, 2000),
                    "timestamp": message.timestamp,
                }))
            })
            .collect::<Vec<_>>();
        let total = filtered.len();
        if filtered.len() > limit {
            filtered = filtered.split_off(filtered.len() - limit);
        }
        json!({
            "session_key": session_key,
            "count": filtered.len(),
            "total_in_session": total,
            "messages": filtered,
        })
        .to_string()
    }

    fn attachments_fetch(&self, args: &JsonValue) -> String {
        let Some(message_id) = arg_string(args, "message_id") else {
            return json!({ "error": "message_id is required" }).to_string();
        };
        let Some((_session_key, session_id)) = self.resolve_session_key(args) else {
            return json!({ "error": "Conversation not found or missing session_id" }).to_string();
        };
        let messages = match self.context.open_session_store() {
            Ok(store) => match store.get_messages(&session_id) {
                Ok(messages) => messages,
                Err(error) => {
                    return json!({ "error": format!("Failed to read messages: {error}") })
                        .to_string();
                }
            },
            Err(error) => {
                return json!({ "error": format!("Session database unavailable: {error}") })
                    .to_string();
            }
        };
        let Some(message) = messages
            .iter()
            .find(|message| message.id.to_string() == message_id)
        else {
            return json!({ "error": format!("Message not found: {message_id}") }).to_string();
        };
        let attachments = extract_attachments(message);
        json!({
            "message_id": message_id,
            "count": attachments.len(),
            "attachments": attachments,
        })
        .to_string()
    }

    fn events_poll(&mut self, args: &JsonValue) -> String {
        self.bridge.poll_once(&self.context);
        let after_cursor = arg_u64(args, "after_cursor", 0);
        let session_key = arg_string(args, "session_key");
        let limit = arg_usize(args, "limit", DEFAULT_EVENT_LIMIT);
        json!(
            self.bridge
                .poll_events(after_cursor, session_key.as_deref(), limit)
        )
        .to_string()
    }

    fn events_wait(&mut self, args: &JsonValue) -> String {
        let after_cursor = arg_u64(args, "after_cursor", 0);
        let session_key = arg_string(args, "session_key");
        let timeout =
            Duration::from_millis(arg_u64(args, "timeout_ms", 30_000)).min(MAX_WAIT_TIMEOUT);
        let deadline = Instant::now() + timeout;
        loop {
            self.bridge.poll_once(&self.context);
            if let Some(event) = self.bridge.next_event(after_cursor, session_key.as_deref()) {
                return json!({ "event": event }).to_string();
            }
            if Instant::now() >= deadline {
                return json!({ "event": null, "reason": "timeout" }).to_string();
            }
            sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    fn messages_send(&self, args: &JsonValue) -> String {
        let target = arg_string(args, "target").unwrap_or_default();
        let message = arg_string(args, "message").unwrap_or_default();
        if target.is_empty() || message.is_empty() {
            return json!({ "error": "Both target and message are required" }).to_string();
        }
        dispatch_tool(
            "send_message",
            json!({ "action": "send", "target": target, "message": message }),
            &self.runtime,
        )
    }

    fn channels_list(&self, args: &JsonValue) -> String {
        let platform = arg_string(args, "platform").map(|value| value.to_ascii_lowercase());
        let directory = load_channel_directory(&self.context);
        if let Some(directory) = directory.filter(JsonValue::is_object) {
            let channels = directory
                .as_object()
                .into_iter()
                .flat_map(|object| object.iter())
                .filter(|(name, _)| {
                    platform
                        .as_deref()
                        .is_none_or(|platform| name.to_ascii_lowercase() == platform)
                })
                .flat_map(|(platform_name, entries)| {
                    entries.as_array().into_iter().flat_map(move |items| {
                        items.iter().filter_map(move |item| {
                            let item = item.as_object()?;
                            let chat_id = nested_entry_string(Some(item), "id")
                                .or_else(|| nested_entry_string(Some(item), "chat_id"))
                                .unwrap_or_default();
                            Some(json!({
                                "target": if chat_id.is_empty() {
                                    platform_name.to_string()
                                } else {
                                    format!("{platform_name}:{chat_id}")
                                },
                                "platform": platform_name,
                                "name": nested_entry_string(Some(item), "name")
                                    .or_else(|| nested_entry_string(Some(item), "display_name"))
                                    .unwrap_or_default(),
                                "chat_type": nested_entry_string(Some(item), "type").unwrap_or_default(),
                            }))
                        })
                    })
                })
                .collect::<Vec<_>>();
            return json!({ "count": channels.len(), "channels": channels }).to_string();
        }

        let mut seen = BTreeMap::new();
        for (key, entry) in load_sessions_index(&self.context) {
            let origin = entry.get("origin").and_then(JsonValue::as_object);
            let platform_name = entry_string(&entry, "platform")
                .or_else(|| nested_entry_string(origin, "platform"))
                .unwrap_or_default();
            if platform_name.is_empty() {
                continue;
            }
            if let Some(platform) = platform.as_deref()
                && platform_name.to_ascii_lowercase() != platform
            {
                continue;
            }
            let chat_id = nested_entry_string(origin, "chat_id")
                .or_else(|| entry_string(&entry, "chat_id"))
                .unwrap_or_default();
            if chat_id.is_empty() {
                continue;
            }
            let target = format!("{platform_name}:{chat_id}");
            seen.entry(target.clone()).or_insert_with(|| {
                json!({
                    "target": target,
                    "platform": platform_name,
                    "name": entry_string(&entry, "display_name")
                        .or_else(|| nested_entry_string(origin, "chat_name"))
                        .unwrap_or(key),
                    "chat_type": entry_string(&entry, "chat_type")
                        .or_else(|| nested_entry_string(origin, "chat_type"))
                        .unwrap_or_default(),
                })
            });
        }
        let channels = seen.into_values().collect::<Vec<_>>();
        json!({ "count": channels.len(), "channels": channels }).to_string()
    }

    fn permissions_list_open(&self) -> String {
        let approvals = self
            .bridge
            .pending_approvals
            .values()
            .cloned()
            .collect::<Vec<_>>();
        json!({ "count": approvals.len(), "approvals": approvals }).to_string()
    }

    fn permissions_respond(&mut self, args: &JsonValue) -> String {
        let approval_id = arg_string(args, "id").unwrap_or_default();
        let decision = arg_string(args, "decision").unwrap_or_default();
        if !matches!(decision.as_str(), "allow-once" | "allow-always" | "deny") {
            return json!({
                "error": format!("Invalid decision: {decision}. Must be allow-once, allow-always, or deny")
            })
            .to_string();
        }
        let Some(approval) = self.bridge.pending_approvals.remove(&approval_id) else {
            return json!({ "error": format!("Approval not found: {approval_id}") }).to_string();
        };
        let session_key = approval
            .get("session_key")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        self.bridge.enqueue(
            String::from("approval_resolved"),
            session_key,
            json_map(json!({ "approval_id": approval_id, "decision": decision })),
        );
        json!({ "resolved": true, "approval_id": approval_id, "decision": decision }).to_string()
    }

    fn resolve_session_key(&self, args: &JsonValue) -> Option<(String, String)> {
        let session_key = arg_string(args, "session_key")?;
        let entries = load_sessions_index(&self.context);
        let entry = entries.get(&session_key)?;
        let session_id = entry_string(entry, "session_id")?;
        (!session_id.is_empty()).then_some((session_key, session_id))
    }
}

impl EventBridge {
    fn poll_once(&mut self, context: &HermesContext) {
        let entries = load_sessions_index(context);
        let Ok(store) = context.open_session_store() else {
            return;
        };
        for (session_key, entry) in entries {
            let Some(session_id) = entry_string(&entry, "session_id") else {
                continue;
            };
            let Ok(messages) = store.get_messages(&session_id) else {
                continue;
            };
            let last_seen = self
                .last_seen_message_ids
                .get(&session_key)
                .copied()
                .unwrap_or(0);
            let mut max_seen = last_seen;
            for message in messages {
                max_seen = max_seen.max(message.id);
                if message.id <= last_seen {
                    continue;
                }
                if message.role != "user" && message.role != "assistant" {
                    continue;
                }
                let Some(content) = extract_message_content(&message) else {
                    continue;
                };
                self.enqueue(
                    String::from("message"),
                    session_key.clone(),
                    json_map(json!({
                        "role": message.role,
                        "content": truncate_chars(&content, 500),
                        "timestamp": message.timestamp,
                        "message_id": message.id.to_string(),
                    })),
                );
            }
            if max_seen > last_seen {
                self.last_seen_message_ids.insert(session_key, max_seen);
            }
        }
    }

    fn enqueue(&mut self, kind: String, session_key: String, data: JsonMap<String, JsonValue>) {
        self.cursor += 1;
        self.queue.push_back(QueueEvent {
            cursor: self.cursor,
            kind,
            session_key,
            data,
        });
        while self.queue.len() > QUEUE_LIMIT {
            self.queue.pop_front();
        }
    }

    fn poll_events(&self, after_cursor: u64, session_key: Option<&str>, limit: usize) -> JsonValue {
        let events = self
            .queue
            .iter()
            .filter(|event| event.cursor > after_cursor)
            .filter(|event| session_key.is_none_or(|key| event.session_key == key))
            .take(limit)
            .map(event_to_json)
            .collect::<Vec<_>>();
        let next_cursor = events
            .last()
            .and_then(|event| event.get("cursor"))
            .and_then(JsonValue::as_u64)
            .unwrap_or(after_cursor);
        json!({ "events": events, "next_cursor": next_cursor })
    }

    fn next_event(&self, after_cursor: u64, session_key: Option<&str>) -> Option<JsonValue> {
        self.queue
            .iter()
            .find(|event| {
                event.cursor > after_cursor
                    && session_key.is_none_or(|key| event.session_key == key)
            })
            .map(event_to_json)
    }
}

fn event_to_json(event: &QueueEvent) -> JsonValue {
    let mut object = event.data.clone();
    object.insert(String::from("cursor"), json!(event.cursor));
    object.insert(String::from("type"), JsonValue::String(event.kind.clone()));
    object.insert(
        String::from("session_key"),
        JsonValue::String(event.session_key.clone()),
    );
    JsonValue::Object(object)
}

fn tool_specs() -> Vec<JsonValue> {
    vec![
        tool_spec(
            "conversations_list",
            "List active messaging conversations across connected platforms.",
            json!({
                "platform": string_schema("Optional platform filter."),
                "limit": integer_schema("Maximum conversations to return."),
                "search": string_schema("Optional case-insensitive text search."),
            }),
        ),
        tool_spec(
            "conversation_get",
            "Get detailed info about one conversation by session key.",
            json!({ "session_key": string_schema("Session key from conversations_list.") }),
        ),
        tool_spec(
            "messages_read",
            "Read recent user and assistant messages from a conversation.",
            json!({
                "session_key": string_schema("Session key from conversations_list."),
                "limit": integer_schema("Maximum messages to return."),
            }),
        ),
        tool_spec(
            "attachments_fetch",
            "List non-text attachments for a message.",
            json!({
                "session_key": string_schema("Session key from conversations_list."),
                "message_id": string_schema("Message id from messages_read."),
            }),
        ),
        tool_spec(
            "events_poll",
            "Poll for new conversation events since a cursor.",
            json!({
                "after_cursor": integer_schema("Return events after this cursor."),
                "session_key": string_schema("Optional session-key filter."),
                "limit": integer_schema("Maximum events to return."),
            }),
        ),
        tool_spec(
            "events_wait",
            "Wait for the next conversation event or timeout.",
            json!({
                "after_cursor": integer_schema("Return events after this cursor."),
                "session_key": string_schema("Optional session-key filter."),
                "timeout_ms": integer_schema("Maximum wait time in milliseconds."),
            }),
        ),
        tool_spec(
            "messages_send",
            "Send a message to a platform conversation.",
            json!({
                "target": string_schema("Platform target, such as telegram:123456."),
                "message": string_schema("Message text to send."),
            }),
        ),
        tool_spec(
            "channels_list",
            "List available messaging channels and targets.",
            json!({ "platform": string_schema("Optional platform filter.") }),
        ),
        tool_spec(
            "permissions_list_open",
            "List pending approval requests observed during this bridge session.",
            json!({}),
        ),
        tool_spec(
            "permissions_respond",
            "Respond to a pending approval request.",
            json!({
                "id": string_schema("Approval id."),
                "decision": {
                    "type": "string",
                    "enum": ["allow-once", "allow-always", "deny"],
                },
            }),
        ),
    ]
}

fn tool_spec(name: &str, description: &str, properties: JsonValue) -> JsonValue {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
        },
    })
}

fn string_schema(description: &str) -> JsonValue {
    json!({ "type": "string", "description": description })
}

fn integer_schema(description: &str) -> JsonValue {
    json!({ "type": "integer", "description": description })
}

fn load_sessions_index(context: &HermesContext) -> BTreeMap<String, JsonValue> {
    let path = context.hermes_home().join("sessions").join("sessions.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    serde_json::from_str::<BTreeMap<String, JsonValue>>(&text).unwrap_or_default()
}

fn load_channel_directory(context: &HermesContext) -> Option<JsonValue> {
    let path = context.hermes_home().join("channel_directory.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<JsonValue>(&text).ok()
}

fn extract_message_content(message: &MessageRecord) -> Option<String> {
    content_to_text(message.content.as_ref()).filter(|value| !value.is_empty())
}

fn content_to_text(value: Option<&JsonValue>) -> Option<String> {
    match value? {
        JsonValue::String(text) => Some(text.clone()),
        JsonValue::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    let part = part.as_object()?;
                    (part.get("type").and_then(JsonValue::as_str) == Some("text"))
                        .then(|| part.get("text").and_then(JsonValue::as_str))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        JsonValue::Null => None,
        other => Some(other.to_string()),
    }
}

fn extract_attachments(message: &MessageRecord) -> Vec<JsonValue> {
    let mut attachments = Vec::new();
    if let Some(JsonValue::Array(parts)) = message.content.as_ref() {
        for part in parts {
            let Some(object) = part.as_object() else {
                continue;
            };
            match object
                .get("type")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
            {
                "image_url" => {
                    let url = object
                        .get("image_url")
                        .and_then(JsonValue::as_object)
                        .and_then(|value| value.get("url"))
                        .and_then(JsonValue::as_str);
                    if let Some(url) = url {
                        attachments.push(json!({ "type": "image", "url": url }));
                    }
                }
                "image" => {
                    let url = object
                        .get("url")
                        .or_else(|| object.get("source").and_then(|value| value.get("url")))
                        .and_then(JsonValue::as_str);
                    if let Some(url) = url {
                        attachments.push(json!({ "type": "image", "url": url }));
                    }
                }
                "text" => {}
                other => attachments.push(json!({ "type": other, "data": part })),
            }
        }
    }
    if let Some(text) = extract_message_content(message) {
        let mut rest = text.as_str();
        while let Some(index) = rest.find("MEDIA:") {
            let after = rest[index + "MEDIA:".len()..].trim_start();
            if let Some(path) = after.split_whitespace().next() {
                attachments.push(json!({ "type": "media", "path": path }));
                rest = &after[path.len()..];
            } else {
                break;
            }
        }
    }
    attachments
}

fn arg_string(args: &JsonValue, key: &str) -> Option<String> {
    args.get(key)
        .and_then(value_to_string)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn arg_usize(args: &JsonValue, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(JsonValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn arg_u64(args: &JsonValue, key: &str, default: u64) -> u64 {
    args.get(key).and_then(JsonValue::as_u64).unwrap_or(default)
}

fn entry_string(entry: &JsonValue, key: &str) -> Option<String> {
    entry.as_object()?.get(key).and_then(value_to_string)
}

fn nested_entry_string(object: Option<&JsonMap<String, JsonValue>>, key: &str) -> Option<String> {
    object?.get(key).and_then(value_to_string)
}

fn value_to_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(text) => Some(text.clone()),
        JsonValue::Number(number) => Some(number.to_string()),
        JsonValue::Bool(boolean) => Some(boolean.to_string()),
        _ => None,
    }
}

fn value_sort_key(value: Option<&JsonValue>) -> String {
    value.and_then(value_to_string).unwrap_or_default()
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn json_map(value: JsonValue) -> JsonMap<String, JsonValue> {
    value.as_object().cloned().unwrap_or_default()
}

fn write_jsonrpc_result<W: Write>(
    output: &mut W,
    id: JsonValue,
    result: JsonValue,
) -> io::Result<()> {
    writeln!(
        output,
        "{}",
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    )?;
    output.flush()
}

fn write_jsonrpc_error<W: Write>(
    output: &mut W,
    id: JsonValue,
    code: i64,
    message: &str,
    data: Option<JsonValue>,
) -> io::Result<()> {
    let mut error = json!({ "code": code, "message": message });
    if let Some(data) = data
        && let Some(object) = error.as_object_mut()
    {
        object.insert(String::from("data"), data);
    }
    writeln!(
        output,
        "{}",
        json!({ "jsonrpc": "2.0", "id": id, "error": error })
    )?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::{MessageAppend, SessionCreate};
    use std::fs;
    use std::io::Cursor;
    use tempfile::TempDir;

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        context.ensure_hermes_home().unwrap();
        (temp, context)
    }

    fn seed_gateway_session(context: &HermesContext) {
        let sessions_dir = context.hermes_home().join("sessions");
        fs::create_dir_all(&sessions_dir).unwrap();
        fs::write(
            sessions_dir.join("sessions.json"),
            json!({
                "agent:main:telegram:dm:123456": {
                    "session_key": "agent:main:telegram:dm:123456",
                    "session_id": "session-1",
                    "platform": "telegram",
                    "chat_type": "dm",
                    "display_name": "Alice",
                    "updated_at": "2026-03-29T14:30:00",
                    "input_tokens": 50,
                    "origin": {
                        "platform": "telegram",
                        "chat_id": "123456",
                        "chat_name": "Alice",
                        "chat_type": "dm",
                        "user_name": "Alice"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let store = context.open_session_store().unwrap();
        store
            .create_session(&SessionCreate {
                id: String::from("session-1"),
                source: String::from("gateway"),
                user_id: None,
                model: None,
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        for (role, content) in [
            ("user", "Hello Alice!"),
            ("assistant", "Hi! MEDIA: /tmp/screenshot.png"),
            ("tool", "{\"ok\": true}"),
        ] {
            store
                .append_message(
                    "session-1",
                    &MessageAppend {
                        role: role.to_string(),
                        content: Some(JsonValue::String(content.to_string())),
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
        }
    }

    fn tool_text(response: &JsonValue) -> JsonValue {
        let text = response
            .get("result")
            .and_then(|result| result.get("content"))
            .and_then(JsonValue::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("text"))
            .and_then(JsonValue::as_str)
            .unwrap();
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn native_mcp_server_lists_tools_and_reads_messages() {
        let (_temp, context) = test_context();
        seed_gateway_session(&context);
        let input = [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","method":"notifications/initialized","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"conversations_list","arguments":{}}}).to_string(),
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"messages_read","arguments":{"session_key":"agent:main:telegram:dm:123456"}}}).to_string(),
            json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"attachments_fetch","arguments":{"session_key":"agent:main:telegram:dm:123456","message_id":"2"}}}).to_string(),
        ]
        .join("\n");
        let mut output = Vec::new();

        run_mcp_jsonrpc(&context, Cursor::new(input), &mut output).unwrap();

        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<JsonValue>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 5);
        assert_eq!(
            responses[0]["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );
        let tools = responses[1]["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["name"] == "messages_read"));

        let conversations = tool_text(&responses[2]);
        assert_eq!(conversations["count"], 1);
        assert_eq!(conversations["conversations"][0]["display_name"], "Alice");

        let messages = tool_text(&responses[3]);
        assert_eq!(messages["count"], 2);
        assert_eq!(messages["messages"][0]["content"], "Hello Alice!");

        let attachments = tool_text(&responses[4]);
        assert_eq!(attachments["count"], 1);
        assert_eq!(attachments["attachments"][0]["path"], "/tmp/screenshot.png");
    }

    #[test]
    fn native_mcp_events_poll_tracks_new_messages_once() {
        let (_temp, context) = test_context();
        seed_gateway_session(&context);
        let mut server = NativeMcpServer {
            context: context.clone(),
            runtime: ToolRuntime::new("/tmp").with_hermes_home(context.hermes_home()),
            bridge: EventBridge::default(),
        };

        let first = serde_json::from_str::<JsonValue>(&server.events_poll(&json!({}))).unwrap();
        assert_eq!(first["events"].as_array().unwrap().len(), 2);
        assert_eq!(first["next_cursor"], 2);

        let second = serde_json::from_str::<JsonValue>(&server.events_poll(&json!({
            "after_cursor": 2
        })))
        .unwrap();
        assert_eq!(second["events"].as_array().unwrap().len(), 0);
        assert_eq!(second["next_cursor"], 2);
    }
}
