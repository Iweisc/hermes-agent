//! Codex Responses API adapter.
//!
//! Pure format-conversion and normalization logic for the OpenAI Responses API
//! (used by OpenAI Codex, xAI, GitHub Models, and other Responses-compatible
//! endpoints).
//!
//! Ported faithfully from `agent/codex_responses_adapter.py`. All functions are
//! stateless — they operate on the data passed in and return transformed
//! results. Dynamic Python `dict`/`list` payloads are represented as
//! [`serde_json::Value`]; the Responses response object (a structured SDK
//! object in Python, accessed via `getattr`) is likewise represented as a
//! `Value` here, since the agent decodes the HTTP JSON body into a `Value`
//! before normalization.

use std::collections::HashSet;

use regex::Regex;
use serde_json::{json, Map, Value};
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::Sha256;

/// Default system identity used when no instructions are resolvable.
///
/// Mirrors `agent.prompt_builder.DEFAULT_AGENT_IDENTITY`. The transports module
/// already exposes a `DEFAULT_AGENT_IDENTITY` constant with a placeholder
/// value; we keep the full canonical Python string local to this module so the
/// preflight produces byte-identical instructions to the Python adapter.
pub const DEFAULT_AGENT_IDENTITY: &str = concat!(
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. ",
    "You are helpful, knowledgeable, and direct. You assist users with a wide ",
    "range of tasks including answering questions, writing and editing code, ",
    "analyzing information, creative work, and executing actions via your tools. ",
    "You communicate clearly, admit uncertainty when appropriate, and prioritize ",
    "being genuinely useful over being verbose unless otherwise directed below. ",
    "Be targeted and efficient in your exploration and investigations."
);

/// Error type for preflight validation failures. Mirrors Python `ValueError`
/// (and `RuntimeError` from `normalize_codex_response`, which we surface as the
/// same error category since callers treat both as request/response failures).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexAdapterError(pub String);

impl std::fmt::Display for CodexAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodexAdapterError {}

fn err<S: Into<String>>(msg: S) -> CodexAdapterError {
    CodexAdapterError(msg.into())
}

/// Matches Codex/Harmony tool-call serialization that occasionally leaks into
/// assistant-message content when the model fails to emit a structured
/// `function_call` item. Accepts the common forms:
///
/// ```text
///   to=functions.exec_command
///   assistant to=functions.exec_command
///   <|channel|>commentary to=functions.exec_command
/// ```
///
/// `to=functions.<name>` is the stable marker — the optional `assistant` or
/// Harmony channel prefix varies by degeneration mode. Case-insensitive to
/// cover lowercase/uppercase `assistant` variants.
fn tool_call_leak_pattern() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*").unwrap()
    })
}

// ---------------------------------------------------------------------------
// Small helpers mirroring Python idioms
// ---------------------------------------------------------------------------

/// `str(x)` for a JSON value, mirroring Python's `str()` on the scalar types
/// that flow through this adapter.
fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Equivalent of `isinstance(v, str)` returning the &str when true.
fn as_str(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Python truthiness for a non-empty stripped string check: returns the
/// stripped string only when it is a non-empty `str`.
fn nonempty_stripped(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Multimodal content helpers
// ---------------------------------------------------------------------------

/// Convert chat-style multimodal content to Responses API input parts.
///
/// Input:  `[{"type":"text"|"image_url", ...}]` (native OpenAI Chat format)
/// Output: `[{"type":"input_text"|"output_text"|"input_image", ...}]`
///
/// The `role` parameter controls the text content type:
/// - `"user"` (default) → `"input_text"`
/// - `"assistant"` → `"output_text"`
///
/// Returns an empty vec when `content` is not a list or contains no recognized
/// parts — callers fall back to the string path.
pub fn chat_content_to_responses_parts(content: &Value, role: &str) -> Vec<Value> {
    let text_type = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    let list = match content.as_array() {
        Some(l) => l,
        None => return Vec::new(),
    };
    let mut converted: Vec<Value> = Vec::new();
    for part in list {
        if let Value::String(s) = part {
            if !s.is_empty() {
                converted.push(json!({"type": text_type, "text": s}));
            }
            continue;
        }
        let obj = match part.as_object() {
            Some(o) => o,
            None => continue,
        };
        let ptype = obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if ptype == "text" || ptype == "input_text" || ptype == "output_text" {
            if let Some(Value::String(text)) = obj.get("text") {
                if !text.is_empty() {
                    converted.push(json!({"type": text_type, "text": text}));
                }
            }
            continue;
        }
        if ptype == "image_url" || ptype == "input_image" {
            let image_ref = obj.get("image_url");
            let mut detail = obj.get("detail").cloned();
            let url: Option<String>;
            match image_ref {
                Some(Value::Object(m)) => {
                    url = m.get("url").and_then(|v| v.as_str()).map(|s| s.to_string());
                    if let Some(d) = m.get("detail") {
                        detail = Some(d.clone());
                    }
                }
                Some(Value::String(s)) => url = Some(s.clone()),
                _ => url = None,
            }
            let url = match url {
                Some(u) if !u.is_empty() => u,
                _ => continue,
            };
            let mut image_part = Map::new();
            image_part.insert("type".into(), Value::String("input_image".into()));
            image_part.insert("image_url".into(), Value::String(url));
            if let Some(Value::String(d)) = detail.as_ref() {
                if !d.trim().is_empty() {
                    image_part.insert("detail".into(), Value::String(d.trim().to_string()));
                }
            }
            converted.push(Value::Object(image_part));
        }
    }
    converted
}

/// Return a short text summary of a user message for logging/trajectory.
pub fn summarize_user_message_for_log(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(list) => {
            let mut text_bits: Vec<String> = Vec::new();
            let mut image_count = 0usize;
            for part in list {
                if let Value::String(s) = part {
                    if !s.is_empty() {
                        text_bits.push(s.clone());
                    }
                    continue;
                }
                let obj = match part.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let ptype = obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                if ptype == "text" || ptype == "input_text" || ptype == "output_text" {
                    if let Some(Value::String(text)) = obj.get("text") {
                        if !text.is_empty() {
                            text_bits.push(text.clone());
                        }
                    }
                } else if ptype == "image_url" || ptype == "input_image" {
                    image_count += 1;
                }
            }
            let summary = text_bits.join(" ").trim().to_string();
            if image_count > 0 {
                let note = format!(
                    "[{} image{}]",
                    image_count,
                    if image_count != 1 { "s" } else { "" }
                );
                if summary.is_empty() {
                    note
                } else {
                    format!("{} {}", note, summary)
                }
            } else {
                summary
            }
        }
        other => py_str(other),
    }
}

// ---------------------------------------------------------------------------
// ID helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic call_id from tool call content.
///
/// Used as a fallback when the API doesn't provide a call_id. Deterministic IDs
/// prevent cache invalidation — random UUIDs would make every API call's prefix
/// unique, breaking OpenAI's prompt cache.
pub fn deterministic_call_id(fn_name: &str, arguments: &str, index: usize) -> String {
    let seed = format!("{}:{}:{}", fn_name, arguments, index);
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    format!("call_{}", &hex[..12])
}

/// Split a stored tool id into `(call_id, response_item_id)`.
pub fn split_responses_tool_id(raw_id: Option<&Value>) -> (Option<String>, Option<String>) {
    let value = match raw_id {
        Some(Value::String(s)) => s.trim(),
        _ => return (None, None),
    };
    if value.is_empty() {
        return (None, None);
    }
    if let Some(idx) = value.find('|') {
        let call_id = value[..idx].trim();
        let response_item_id = value[idx + 1..].trim();
        let call_id = if call_id.is_empty() {
            None
        } else {
            Some(call_id.to_string())
        };
        let response_item_id = if response_item_id.is_empty() {
            None
        } else {
            Some(response_item_id.to_string())
        };
        return (call_id, response_item_id);
    }
    if value.starts_with("fc_") {
        return (None, Some(value.to_string()));
    }
    (Some(value.to_string()), None)
}

/// Build a valid Responses `function_call.id` (must start with `fc_`).
pub fn derive_responses_function_call_id(
    call_id: &str,
    response_item_id: Option<&str>,
) -> String {
    if let Some(rid) = response_item_id {
        let candidate = rid.trim();
        if candidate.starts_with("fc_") {
            return candidate.to_string();
        }
    }

    let source = call_id.trim();
    if source.starts_with("fc_") {
        return source.to_string();
    }
    if source.starts_with("call_") && source.len() > "call_".len() {
        return format!("fc_{}", &source["call_".len()..]);
    }

    let sanitized: String = source
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized.starts_with("fc_") {
        return sanitized;
    }
    if sanitized.starts_with("call_") && sanitized.len() > "call_".len() {
        return format!("fc_{}", &sanitized["call_".len()..]);
    }
    if !sanitized.is_empty() {
        let truncated: String = sanitized.chars().take(48).collect();
        return format!("fc_{}", truncated);
    }

    // seed = source or str(response_item_id or "") or uuid4().hex
    let seed = if !source.is_empty() {
        source.to_string()
    } else {
        let rid = response_item_id.unwrap_or("");
        if !rid.is_empty() {
            rid.to_string()
        } else {
            // uuid4().hex equivalent: 32 hex chars
            let mut s = String::with_capacity(32);
            for _ in 0..16 {
                let b: u8 = fastrand_byte();
                s.push_str(&format!("{:02x}", b));
            }
            s
        }
    };
    let mut hasher = Sha1::new();
    hasher.update(seed.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    format!("fc_{}", &hex[..24])
}

/// Minimal random byte source for the uuid4 fallback (only hit when there is
/// genuinely no source material to hash, matching the Python `uuid.uuid4()`
/// fallback). Uses the system time + a counter; sufficient for the
/// deterministic-id last resort which never feeds prefix caching.
fn fastrand_byte() -> u8 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static STATE: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let prev = STATE.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    let mut x = prev ^ nanos.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    (x >> 56) as u8
}

// ---------------------------------------------------------------------------
// Schema conversion
// ---------------------------------------------------------------------------

/// Convert chat-completions tool schemas to Responses function-tool schemas.
///
/// Returns `None` when there are no usable tools (mirrors Python returning
/// `None` for empty input or `converted or None`).
pub fn responses_tools(tools: Option<&Value>) -> Option<Vec<Value>> {
    let list = match tools {
        Some(Value::Array(l)) if !l.is_empty() => l,
        _ => return None,
    };

    let mut converted: Vec<Value> = Vec::new();
    for item in list {
        let fn_obj = item
            .as_object()
            .and_then(|o| o.get("function"))
            .and_then(|f| f.as_object());
        let fn_obj = match fn_obj {
            Some(f) => f,
            None => {
                // item.get("function", {}) yields {} for dicts without it, and
                // a non-dict item makes fn = {}. Either way name lookup fails.
                continue;
            }
        };
        let name = match fn_obj.get("name") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => continue,
        };
        let description = fn_obj
            .get("description")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        let parameters = fn_obj.get("parameters").cloned().unwrap_or_else(|| {
            json!({"type": "object", "properties": {}})
        });
        converted.push(json!({
            "type": "function",
            "name": name,
            "description": description,
            "strict": false,
            "parameters": parameters,
        }));
    }
    if converted.is_empty() {
        None
    } else {
        Some(converted)
    }
}

// ---------------------------------------------------------------------------
// Message format conversion
// ---------------------------------------------------------------------------

fn is_response_message_status(s: &str) -> bool {
    matches!(s, "completed" | "incomplete" | "in_progress")
}

/// Normalize a Responses assistant message status for replay.
pub fn normalize_responses_message_status(value: Option<&Value>, default: &str) -> String {
    if let Some(Value::String(v)) = value {
        let status = v
            .trim()
            .to_lowercase()
            .replace('-', "_")
            .replace(' ', "_");
        if is_response_message_status(&status) {
            return status;
        }
    }
    default.to_string()
}

/// Convert internal chat-style messages to Responses input items.
pub fn chat_messages_to_responses_input(messages: &[Value]) -> Vec<Value> {
    let mut items: Vec<Value> = Vec::new();
    let mut seen_item_ids: HashSet<String> = HashSet::new();

    for msg in messages {
        let msg = match msg.as_object() {
            Some(m) => m,
            None => continue,
        };
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "system" {
            continue;
        }

        if role == "user" || role == "assistant" {
            let content_val = msg.get("content").cloned().unwrap_or(Value::String(String::new()));
            let (content_parts, content_text): (Vec<Value>, String) =
                if content_val.is_array() {
                    let parts = chat_content_to_responses_parts(&content_val, role);
                    let text_type = if role == "assistant" {
                        "output_text"
                    } else {
                        "input_text"
                    };
                    let text: String = parts
                        .iter()
                        .filter_map(|p| {
                            let o = p.as_object()?;
                            if o.get("type").and_then(|v| v.as_str()) == Some(text_type) {
                                o.get("text").and_then(|v| v.as_str())
                            } else {
                                None
                            }
                        })
                        .collect();
                    (parts, text)
                } else {
                    let text = match &content_val {
                        Value::Null => String::new(),
                        other => py_str(other),
                    };
                    (Vec::new(), text)
                };

            if role == "assistant" {
                // Replay encrypted reasoning items from previous turns.
                let mut has_codex_reasoning = false;
                if let Some(Value::Array(codex_reasoning)) = msg.get("codex_reasoning_items") {
                    for ri in codex_reasoning {
                        if let Some(ri_obj) = ri.as_object() {
                            let enc = ri_obj.get("encrypted_content");
                            let enc_truthy = match enc {
                                Some(Value::String(s)) => !s.is_empty(),
                                Some(Value::Null) | None => false,
                                Some(other) => !other.is_null(),
                            };
                            if enc_truthy {
                                let item_id =
                                    ri_obj.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                if let Some(ref iid) = item_id {
                                    if seen_item_ids.contains(iid) {
                                        continue;
                                    }
                                }
                                // Strip the "id" field.
                                let mut replay_item = ri_obj.clone();
                                replay_item.remove("id");
                                items.push(Value::Object(replay_item));
                                if let Some(iid) = item_id {
                                    seen_item_ids.insert(iid);
                                }
                                has_codex_reasoning = true;
                            }
                        }
                    }
                }

                // Replay exact assistant message items (with id/phase).
                let mut replayed_message_items = 0usize;
                if let Some(Value::Array(codex_message_items)) = msg.get("codex_message_items") {
                    for raw_item in codex_message_items {
                        let raw = match raw_item.as_object() {
                            Some(o) => o,
                            None => continue,
                        };
                        if raw.get("type").and_then(|v| v.as_str()) != Some("message")
                            || raw.get("role").and_then(|v| v.as_str()) != Some("assistant")
                        {
                            continue;
                        }
                        let raw_content_parts = match raw.get("content") {
                            Some(Value::Array(c)) => c,
                            _ => continue,
                        };

                        let mut normalized_content_parts: Vec<Value> = Vec::new();
                        for part in raw_content_parts {
                            let p = match part.as_object() {
                                Some(o) => o,
                                None => continue,
                            };
                            let part_type =
                                p.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
                            if part_type != "output_text" && part_type != "text" {
                                continue;
                            }
                            let text = match p.get("text") {
                                Some(Value::Null) | None => String::new(),
                                Some(Value::String(s)) => s.clone(),
                                Some(other) => py_str(other),
                            };
                            normalized_content_parts
                                .push(json!({"type": "output_text", "text": text}));
                        }

                        if normalized_content_parts.is_empty() {
                            continue;
                        }

                        let mut replay_item = Map::new();
                        replay_item.insert("type".into(), Value::String("message".into()));
                        replay_item.insert("role".into(), Value::String("assistant".into()));
                        replay_item.insert(
                            "status".into(),
                            Value::String(normalize_responses_message_status(
                                raw.get("status"),
                                "completed",
                            )),
                        );
                        replay_item.insert(
                            "content".into(),
                            Value::Array(normalized_content_parts),
                        );
                        if let Some(iid) = nonempty_stripped(raw.get("id")) {
                            replay_item.insert("id".into(), Value::String(iid));
                        }
                        if let Some(phase) = nonempty_stripped(raw.get("phase")) {
                            replay_item.insert("phase".into(), Value::String(phase));
                        }
                        items.push(Value::Object(replay_item));
                        replayed_message_items += 1;
                    }
                }

                if replayed_message_items > 0 {
                    // pass
                } else if !content_parts.is_empty() {
                    items.push(json!({"role": "assistant", "content": content_parts}));
                } else if !content_text.trim().is_empty() {
                    items.push(json!({"role": "assistant", "content": content_text}));
                } else if has_codex_reasoning {
                    // The Responses API requires a following item after each
                    // reasoning item.
                    items.push(json!({"role": "assistant", "content": ""}));
                }

                if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
                    for tc in tool_calls {
                        let tc = match tc.as_object() {
                            Some(o) => o,
                            None => continue,
                        };
                        let fn_obj = tc.get("function").and_then(|f| f.as_object());
                        let fn_name = match fn_obj.and_then(|f| f.get("name")) {
                            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
                            _ => continue,
                        };

                        let (embedded_call_id, embedded_response_item_id) =
                            split_responses_tool_id(tc.get("id"));

                        let mut call_id = nonempty_stripped(tc.get("call_id"));
                        if call_id.is_none() {
                            call_id = embedded_call_id.clone();
                        }
                        let call_id = match call_id {
                            Some(c) if !c.trim().is_empty() => c.trim().to_string(),
                            _ => {
                                if let Some(ref rid) = embedded_response_item_id {
                                    if rid.starts_with("fc_") && rid.len() > "fc_".len() {
                                        format!("call_{}", &rid["fc_".len()..])
                                    } else {
                                        let raw_args = fn_obj
                                            .and_then(|f| f.get("arguments"))
                                            .map(py_str)
                                            .unwrap_or_else(|| "{}".to_string());
                                        deterministic_call_id(&fn_name, &raw_args, items.len())
                                    }
                                } else {
                                    let raw_args = fn_obj
                                        .and_then(|f| f.get("arguments"))
                                        .map(py_str)
                                        .unwrap_or_else(|| "{}".to_string());
                                    deterministic_call_id(&fn_name, &raw_args, items.len())
                                }
                            }
                        };

                        let arguments = normalize_arguments(
                            fn_obj.and_then(|f| f.get("arguments")),
                        );

                        items.push(json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": arguments,
                        }));
                    }
                }
                continue;
            }

            // Non-assistant (user) role.
            if !content_parts.is_empty() {
                items.push(json!({"role": role, "content": content_parts}));
            } else {
                items.push(json!({"role": role, "content": content_text}));
            }
            continue;
        }

        if role == "tool" {
            let raw_tool_call_id = msg.get("tool_call_id");
            let (call_id_opt, _) = split_responses_tool_id(raw_tool_call_id);
            let mut call_id = match call_id_opt {
                Some(c) if !c.trim().is_empty() => Some(c),
                _ => None,
            };
            if call_id.is_none() {
                if let Some(s) = nonempty_stripped(raw_tool_call_id) {
                    call_id = Some(s);
                }
            }
            let call_id = match call_id {
                Some(c) if !c.trim().is_empty() => c.trim().to_string(),
                _ => continue,
            };
            // str(msg.get("content", "") or "")
            let content = match msg.get("content") {
                None | Some(Value::Null) => String::new(),
                Some(Value::String(s)) if s.is_empty() => String::new(),
                Some(Value::Bool(false)) => String::new(),
                Some(other) => {
                    // Python `x or ""` only replaces falsy values.
                    if is_falsy(other) {
                        String::new()
                    } else {
                        py_str(other)
                    }
                }
            };
            items.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            }));
        }
    }

    items
}

/// Python falsy check for the JSON value types relevant here.
fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !*b,
        Value::String(s) => s.is_empty(),
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// Normalize a tool-call `arguments` field: dict→JSON string, non-str→str,
/// then `.strip() or "{}"`.
fn normalize_arguments(arguments: Option<&Value>) -> String {
    let s = match arguments {
        Some(Value::Object(_)) => {
            // json.dumps(arguments, ensure_ascii=False)
            serde_json::to_string(arguments.unwrap()).unwrap_or_else(|_| "{}".to_string())
        }
        Some(Value::String(s)) => s.clone(),
        Some(other) => py_str(other),
        None => "{}".to_string(),
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        "{}".to_string()
    } else {
        trimmed.to_string()
    }
}

// ---------------------------------------------------------------------------
// Input preflight / validation
// ---------------------------------------------------------------------------

/// Validate and normalize a list of Responses input items.
pub fn preflight_codex_input_items(raw_items: &Value) -> Result<Vec<Value>, CodexAdapterError> {
    let list = match raw_items.as_array() {
        Some(l) => l,
        None => return Err(err("Codex Responses input must be a list of input items.")),
    };

    let mut normalized: Vec<Value> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for (idx, item) in list.iter().enumerate() {
        let item_obj = match item.as_object() {
            Some(o) => o,
            None => return Err(err(format!("Codex Responses input[{}] must be an object.", idx))),
        };

        let item_type = item_obj.get("type").and_then(|v| v.as_str());

        if item_type == Some("function_call") {
            let call_id = nonempty_stripped(item_obj.get("call_id")).ok_or_else(|| {
                err(format!(
                    "Codex Responses input[{}] function_call is missing call_id.",
                    idx
                ))
            })?;
            let name = nonempty_stripped(item_obj.get("name")).ok_or_else(|| {
                err(format!(
                    "Codex Responses input[{}] function_call is missing name.",
                    idx
                ))
            })?;
            let arguments = normalize_arguments(item_obj.get("arguments"));
            normalized.push(json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }));
            continue;
        }

        if item_type == Some("function_call_output") {
            let call_id = nonempty_stripped(item_obj.get("call_id")).ok_or_else(|| {
                err(format!(
                    "Codex Responses input[{}] function_call_output is missing call_id.",
                    idx
                ))
            })?;
            let output = match item_obj.get("output") {
                None => String::new(),
                Some(Value::Null) => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => py_str(other),
            };
            normalized.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }));
            continue;
        }

        if item_type == Some("reasoning") {
            if let Some(Value::String(encrypted)) = item_obj.get("encrypted_content") {
                if !encrypted.is_empty() {
                    if let Some(Value::String(item_id)) = item_obj.get("id") {
                        if !item_id.is_empty() {
                            if seen_ids.contains(item_id) {
                                continue;
                            }
                            seen_ids.insert(item_id.clone());
                        }
                    }
                    let mut reasoning_item = Map::new();
                    reasoning_item
                        .insert("type".into(), Value::String("reasoning".into()));
                    reasoning_item.insert(
                        "encrypted_content".into(),
                        Value::String(encrypted.clone()),
                    );
                    match item_obj.get("summary") {
                        Some(Value::Array(s)) => {
                            reasoning_item
                                .insert("summary".into(), Value::Array(s.clone()));
                        }
                        _ => {
                            reasoning_item
                                .insert("summary".into(), Value::Array(Vec::new()));
                        }
                    }
                    normalized.push(Value::Object(reasoning_item));
                }
            }
            continue;
        }

        if item_type == Some("message") {
            let role = item_obj.get("role").and_then(|v| v.as_str());
            if role != Some("assistant") {
                return Err(err(format!(
                    "Codex Responses input[{}] message items must have role='assistant'.",
                    idx
                )));
            }
            let content = match item_obj.get("content") {
                Some(Value::Array(c)) => c,
                _ => {
                    return Err(err(format!(
                        "Codex Responses input[{}] message item must have content list.",
                        idx
                    )))
                }
            };
            let mut normalized_content: Vec<Value> = Vec::new();
            for (part_idx, part) in content.iter().enumerate() {
                let p = part.as_object().ok_or_else(|| {
                    err(format!(
                        "Codex Responses input[{}] message content[{}] must be an object.",
                        idx, part_idx
                    ))
                })?;
                let part_type = p.get("type").and_then(|v| v.as_str());
                if part_type != Some("output_text") && part_type != Some("text") {
                    return Err(err(format!(
                        "Codex Responses input[{}] message content[{}] has unsupported type {}.",
                        idx,
                        part_idx,
                        py_repr_type(p.get("type"))
                    )));
                }
                let text = match p.get("text") {
                    None | Some(Value::Null) => String::new(),
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => py_str(other),
                };
                normalized_content.push(json!({"type": "output_text", "text": text}));
            }
            if normalized_content.is_empty() {
                return Err(err(format!(
                    "Codex Responses input[{}] message item must contain at least one text part.",
                    idx
                )));
            }
            let mut normalized_item = Map::new();
            normalized_item.insert("type".into(), Value::String("message".into()));
            normalized_item.insert("role".into(), Value::String("assistant".into()));
            normalized_item.insert(
                "status".into(),
                Value::String(normalize_responses_message_status(
                    item_obj.get("status"),
                    "completed",
                )),
            );
            normalized_item.insert("content".into(), Value::Array(normalized_content));
            if let Some(iid) = nonempty_stripped(item_obj.get("id")) {
                normalized_item.insert("id".into(), Value::String(iid));
            }
            if let Some(phase) = nonempty_stripped(item_obj.get("phase")) {
                normalized_item.insert("phase".into(), Value::String(phase));
            }
            normalized.push(Value::Object(normalized_item));
            continue;
        }

        let role = item_obj.get("role").and_then(|v| v.as_str());
        if role == Some("user") || role == Some("assistant") {
            let role = role.unwrap();
            let content = match item_obj.get("content") {
                None => Value::String(String::new()),
                Some(Value::Null) => Value::String(String::new()),
                Some(v) => v.clone(),
            };
            if let Value::Array(parts) = &content {
                let text_type = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                let mut validated: Vec<Value> = Vec::new();
                for (part_idx, part) in parts.iter().enumerate() {
                    if let Value::String(s) = part {
                        if !s.is_empty() {
                            validated.push(json!({"type": text_type, "text": s}));
                        }
                        continue;
                    }
                    let p = part.as_object().ok_or_else(|| {
                        err(format!(
                            "Codex Responses input[{}].content[{}] must be an object or string.",
                            idx, part_idx
                        ))
                    })?;
                    let ptype = p
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_lowercase();
                    if ptype == "input_text" || ptype == "text" || ptype == "output_text" {
                        let text = match p.get("text") {
                            Some(Value::String(s)) => s.clone(),
                            None | Some(Value::Null) => String::new(),
                            Some(other) => {
                                // str(text or "")
                                if is_falsy(other) {
                                    String::new()
                                } else {
                                    py_str(other)
                                }
                            }
                        };
                        validated.push(json!({"type": text_type, "text": text}));
                    } else if ptype == "input_image" || ptype == "image_url" {
                        let image_ref = p.get("image_url");
                        let mut detail = p.get("detail").cloned();
                        let url: String;
                        match image_ref {
                            Some(Value::Object(m)) => {
                                url = match m.get("url") {
                                    Some(Value::String(s)) => s.clone(),
                                    None | Some(Value::Null) => String::new(),
                                    Some(other) => {
                                        if is_falsy(other) {
                                            String::new()
                                        } else {
                                            py_str(other)
                                        }
                                    }
                                };
                                if let Some(d) = m.get("detail") {
                                    detail = Some(d.clone());
                                }
                            }
                            Some(Value::String(s)) => url = s.clone(),
                            None | Some(Value::Null) => url = String::new(),
                            Some(other) => {
                                if is_falsy(other) {
                                    url = String::new();
                                } else {
                                    url = py_str(other);
                                }
                            }
                        }
                        let mut image_part = Map::new();
                        image_part
                            .insert("type".into(), Value::String("input_image".into()));
                        image_part.insert("image_url".into(), Value::String(url));
                        if let Some(Value::String(d)) = detail.as_ref() {
                            if !d.trim().is_empty() {
                                image_part.insert(
                                    "detail".into(),
                                    Value::String(d.trim().to_string()),
                                );
                            }
                        }
                        validated.push(Value::Object(image_part));
                    } else {
                        return Err(err(format!(
                            "Codex Responses input[{}].content[{}] has unsupported type {}.",
                            idx,
                            part_idx,
                            py_repr_type(p.get("type"))
                        )));
                    }
                }
                normalized.push(json!({"role": role, "content": validated}));
                continue;
            }
            // not a list — coerce to string
            let content_str = match &content {
                Value::String(s) => s.clone(),
                other => py_str(other),
            };
            normalized.push(json!({"role": role, "content": content_str}));
            continue;
        }

        return Err(err(format!(
            "Codex Responses input[{}] has unsupported item shape (type={}, role={}).",
            idx,
            py_repr_type(item_obj.get("type")),
            py_repr_type(item_obj.get("role"))
        )));
    }

    Ok(normalized)
}

/// Render a value the way Python `repr()` would for the small set of types used
/// in these error messages (`None`, `'str'`, numbers).
fn py_repr_type(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => format!("'{}'", s),
        Some(other) => other.to_string(),
    }
}

/// Validate and normalize a full Responses request kwargs object.
pub fn preflight_codex_api_kwargs(
    api_kwargs: &Value,
    allow_stream: bool,
) -> Result<Value, CodexAdapterError> {
    let kwargs = match api_kwargs.as_object() {
        Some(m) => m,
        None => return Err(err("Codex Responses request must be a dict.")),
    };

    let required = ["model", "instructions", "input"];
    let mut missing: Vec<&str> = required
        .iter()
        .filter(|k| !kwargs.contains_key(**k))
        .copied()
        .collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(err(format!(
            "Codex Responses request missing required field(s): {}.",
            missing.join(", ")
        )));
    }

    let model = match kwargs.get("model") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            return Err(err(
                "Codex Responses request 'model' must be a non-empty string.",
            ))
        }
    };

    // instructions: None→"", non-str→str, then strip or DEFAULT_AGENT_IDENTITY
    let instructions = match kwargs.get("instructions") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => py_str(other),
    };
    let instructions = {
        let t = instructions.trim();
        if t.is_empty() {
            DEFAULT_AGENT_IDENTITY.to_string()
        } else {
            t.to_string()
        }
    };

    let normalized_input = preflight_codex_input_items(
        kwargs.get("input").unwrap_or(&Value::Null),
    )?;

    let mut normalized_tools: Option<Vec<Value>> = None;
    if let Some(tools) = kwargs.get("tools") {
        if !tools.is_null() {
            let tool_list = match tools.as_array() {
                Some(l) => l,
                None => {
                    return Err(err(
                        "Codex Responses request 'tools' must be a list when provided.",
                    ))
                }
            };
            let mut out: Vec<Value> = Vec::new();
            for (idx, tool) in tool_list.iter().enumerate() {
                let t = tool.as_object().ok_or_else(|| {
                    err(format!("Codex Responses tools[{}] must be an object.", idx))
                })?;
                if t.get("type").and_then(|v| v.as_str()) != Some("function") {
                    return Err(err(format!(
                        "Codex Responses tools[{}] has unsupported type {}.",
                        idx,
                        py_repr_type(t.get("type"))
                    )));
                }
                let name = nonempty_stripped(t.get("name")).ok_or_else(|| {
                    err(format!(
                        "Codex Responses tools[{}] is missing a valid name.",
                        idx
                    ))
                })?;
                let parameters = match t.get("parameters") {
                    Some(p @ Value::Object(_)) => p.clone(),
                    _ => {
                        return Err(err(format!(
                            "Codex Responses tools[{}] is missing valid parameters.",
                            idx
                        )))
                    }
                };
                let description = match t.get("description") {
                    None | Some(Value::Null) => String::new(),
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => py_str(other),
                };
                let strict = match t.get("strict") {
                    Some(Value::Bool(b)) => *b,
                    None => false,
                    Some(other) => !is_falsy(other),
                };
                out.push(json!({
                    "type": "function",
                    "name": name,
                    "description": description,
                    "strict": strict,
                    "parameters": parameters,
                }));
            }
            normalized_tools = Some(out);
        }
    }

    // store must be exactly false (Python: store is not False → error).
    match kwargs.get("store") {
        None => {} // default False
        Some(Value::Bool(false)) => {}
        Some(_) => {
            return Err(err(
                "Codex Responses contract requires 'store' to be false.",
            ))
        }
    }

    let mut allowed_keys: HashSet<&str> = [
        "model",
        "instructions",
        "input",
        "tools",
        "store",
        "reasoning",
        "include",
        "max_output_tokens",
        "temperature",
        "tool_choice",
        "parallel_tool_calls",
        "prompt_cache_key",
        "service_tier",
        "extra_headers",
    ]
    .into_iter()
    .collect();

    let mut normalized = Map::new();
    normalized.insert("model".into(), Value::String(model));
    normalized.insert("instructions".into(), Value::String(instructions));
    normalized.insert("input".into(), Value::Array(normalized_input));
    normalized.insert("store".into(), Value::Bool(false));
    if let Some(tools) = normalized_tools {
        normalized.insert("tools".into(), Value::Array(tools));
    }

    if let Some(reasoning) = kwargs.get("reasoning") {
        if reasoning.is_object() {
            normalized.insert("reasoning".into(), reasoning.clone());
        }
    }
    if let Some(include) = kwargs.get("include") {
        if include.is_array() {
            normalized.insert("include".into(), include.clone());
        }
    }
    if let Some(st) = nonempty_stripped(kwargs.get("service_tier")) {
        normalized.insert("service_tier".into(), Value::String(st));
    }

    // max_output_tokens: int/float and > 0 → int(value)
    if let Some(v) = kwargs.get("max_output_tokens") {
        if let Some(f) = number_if_not_bool(v) {
            if f > 0.0 {
                normalized.insert(
                    "max_output_tokens".into(),
                    Value::Number((f as i64).into()),
                );
            }
        }
    }
    // temperature: int/float → float(value)
    if let Some(v) = kwargs.get("temperature") {
        if let Some(f) = number_if_not_bool(v) {
            let num = serde_json::Number::from_f64(f)
                .unwrap_or_else(|| serde_json::Number::from(0));
            normalized.insert("temperature".into(), Value::Number(num));
        }
    }

    for key in ["tool_choice", "parallel_tool_calls", "prompt_cache_key"] {
        if let Some(val) = kwargs.get(key) {
            if !val.is_null() {
                normalized.insert(key.to_string(), val.clone());
            }
        }
    }

    if let Some(extra_headers) = kwargs.get("extra_headers") {
        if !extra_headers.is_null() {
            let eh = match extra_headers.as_object() {
                Some(o) => o,
                None => {
                    return Err(err(
                        "Codex Responses request 'extra_headers' must be an object.",
                    ))
                }
            };
            let mut normalized_headers = Map::new();
            for (key, value) in eh {
                if key.trim().is_empty() {
                    return Err(err(
                        "Codex Responses request 'extra_headers' keys must be non-empty strings.",
                    ));
                }
                if value.is_null() {
                    continue;
                }
                normalized_headers.insert(key.trim().to_string(), Value::String(py_str(value)));
            }
            if !normalized_headers.is_empty() {
                normalized.insert("extra_headers".into(), Value::Object(normalized_headers));
            }
        }
    }

    if allow_stream {
        match kwargs.get("stream") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(true)) => {
                normalized.insert("stream".into(), Value::Bool(true));
            }
            Some(_) => {
                return Err(err("Codex Responses 'stream' must be true when set."));
            }
        }
        allowed_keys.insert("stream");
    } else if kwargs.contains_key("stream") {
        return Err(err(
            "Codex Responses stream flag is only allowed in fallback streaming requests.",
        ));
    }

    let mut unexpected: Vec<&str> = kwargs
        .keys()
        .filter(|k| !allowed_keys.contains(k.as_str()))
        .map(|k| k.as_str())
        .collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(err(format!(
            "Codex Responses request has unsupported field(s): {}.",
            unexpected.join(", ")
        )));
    }

    Ok(Value::Object(normalized))
}

/// `isinstance(v, (int, float)) and not isinstance(v, bool)` semantics: bools
/// are excluded (Python `isinstance(True, int)` is True, but the original code
/// relies on the numeric guards; we mirror by excluding bools which never pass
/// the `> 0` / float intent meaningfully here). Returns the numeric value.
fn number_if_not_bool(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        // Python treats bool as int; True→1.0, False→0.0.
        Value::Bool(true) => Some(1.0),
        Value::Bool(false) => Some(0.0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Response extraction helpers
// ---------------------------------------------------------------------------

/// Extract assistant text from a Responses message output item.
pub fn extract_responses_message_text(item: &Value) -> String {
    let content = match item.get("content") {
        Some(Value::Array(c)) => c,
        _ => return String::new(),
    };
    let mut chunks: Vec<String> = Vec::new();
    for part in content {
        let ptype = part.get("type").and_then(|v| v.as_str());
        if ptype != Some("output_text") && ptype != Some("text") {
            continue;
        }
        if let Some(Value::String(text)) = part.get("text") {
            if !text.is_empty() {
                chunks.push(text.clone());
            }
        }
    }
    chunks.concat().trim().to_string()
}

/// Extract a compact reasoning text from a Responses reasoning item.
pub fn extract_responses_reasoning_text(item: &Value) -> String {
    if let Some(Value::Array(summary)) = item.get("summary") {
        let mut chunks: Vec<String> = Vec::new();
        for part in summary {
            if let Some(Value::String(text)) = part.get("text") {
                if !text.is_empty() {
                    chunks.push(text.clone());
                }
            }
        }
        if !chunks.is_empty() {
            return chunks.join("\n").trim().to_string();
        }
    }
    if let Some(Value::String(text)) = item.get("text") {
        if !text.is_empty() {
            return text.trim().to_string();
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Full response normalization
// ---------------------------------------------------------------------------

/// A normalized tool call, mirroring the Python `SimpleNamespace` shape.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedToolCall {
    pub id: String,
    pub call_id: String,
    pub response_item_id: String,
    /// Always "function".
    pub r#type: String,
    pub function_name: String,
    pub function_arguments: String,
}

/// The normalized assistant message, mirroring the Python `SimpleNamespace`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NormalizedAssistantMessage {
    pub content: String,
    pub tool_calls: Vec<NormalizedToolCall>,
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning_details: Option<Value>,
    pub codex_reasoning_items: Option<Vec<Value>>,
    pub codex_message_items: Option<Vec<Value>>,
}

/// Normalize a Responses API object (decoded JSON) to an assistant_message-like
/// object plus a finish_reason string.
///
/// The Python original accepts a structured SDK object and may mutate
/// `response.output`. Here `response` is a decoded JSON `Value`; the
/// empty-output/output_text synthesis path is applied internally and reflected
/// in the returned normalization (the caller does not need the mutated object).
pub fn normalize_codex_response(
    response: &Value,
) -> Result<(NormalizedAssistantMessage, String), CodexAdapterError> {
    // output = getattr(response, "output", None)
    let synthesized: Option<Vec<Value>>;
    let output: &[Value] = match response.get("output") {
        Some(Value::Array(arr)) if !arr.is_empty() => {
            synthesized = None;
            arr.as_slice()
        }
        _ => {
            // empty or non-list output: fall back to output_text
            let out_text = response.get("output_text").and_then(|v| v.as_str());
            match out_text {
                Some(t) if !t.trim().is_empty() => {
                    let synth = vec![json!({
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": t.trim()}],
                    })];
                    synthesized = Some(synth);
                    synthesized.as_ref().unwrap().as_slice()
                }
                _ => return Err(err("Responses API returned no output items")),
            }
        }
    };

    // response_status
    let response_status: Option<String> = match response.get("status") {
        Some(Value::String(s)) => Some(s.trim().to_lowercase()),
        _ => None,
    };

    if matches!(response_status.as_deref(), Some("failed") | Some("cancelled")) {
        let error_obj = response.get("error");
        let error_msg = match error_obj {
            Some(Value::Object(m)) => match m.get("message") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                _ => Value::Object(m.clone()).to_string(),
            },
            Some(e) if !is_falsy(e) => py_str(e),
            _ => format!(
                "Responses API returned status '{}'",
                response_status.as_deref().unwrap_or("")
            ),
        };
        return Err(err(error_msg));
    }

    let mut content_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut reasoning_items_raw: Vec<Value> = Vec::new();
    let mut message_items_raw: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<NormalizedToolCall> = Vec::new();
    let mut has_incomplete_items = matches!(
        response_status.as_deref(),
        Some("queued") | Some("in_progress") | Some("incomplete")
    );
    let mut saw_commentary_phase = false;
    let mut saw_final_answer_phase = false;

    for item in output {
        let item_type = item.get("type").and_then(|v| v.as_str());
        let item_status: Option<String> = match item.get("status") {
            Some(Value::String(s)) => Some(s.trim().to_lowercase()),
            _ => None,
        };

        if matches!(
            item_status.as_deref(),
            Some("queued") | Some("in_progress") | Some("incomplete")
        ) {
            has_incomplete_items = true;
        }

        match item_type {
            Some("message") => {
                let mut normalized_phase: Option<String> = None;
                if let Some(Value::String(item_phase)) = item.get("phase") {
                    let np = item_phase.trim().to_lowercase();
                    if np == "commentary" || np == "analysis" {
                        saw_commentary_phase = true;
                    } else if np == "final_answer" || np == "final" {
                        saw_final_answer_phase = true;
                    }
                    normalized_phase = Some(np);
                }
                let message_text = extract_responses_message_text(item);
                if !message_text.is_empty() {
                    content_parts.push(message_text.clone());
                    let mut raw_message_item = Map::new();
                    raw_message_item.insert("type".into(), Value::String("message".into()));
                    raw_message_item.insert("role".into(), Value::String("assistant".into()));
                    let status_val = item_status.clone().map(Value::String);
                    raw_message_item.insert(
                        "status".into(),
                        Value::String(normalize_responses_message_status(
                            status_val.as_ref(),
                            "completed",
                        )),
                    );
                    raw_message_item.insert(
                        "content".into(),
                        json!([{"type": "output_text", "text": message_text}]),
                    );
                    if let Some(Value::String(item_id)) = item.get("id") {
                        if !item_id.is_empty() {
                            raw_message_item
                                .insert("id".into(), Value::String(item_id.clone()));
                        }
                    }
                    if let Some(np) = normalized_phase {
                        raw_message_item.insert("phase".into(), Value::String(np));
                    }
                    message_items_raw.push(Value::Object(raw_message_item));
                }
            }
            Some("reasoning") => {
                let reasoning_text = extract_responses_reasoning_text(item);
                if !reasoning_text.is_empty() {
                    reasoning_parts.push(reasoning_text);
                }
                if let Some(Value::String(encrypted)) = item.get("encrypted_content") {
                    if !encrypted.is_empty() {
                        let mut raw_item = Map::new();
                        raw_item.insert("type".into(), Value::String("reasoning".into()));
                        raw_item.insert(
                            "encrypted_content".into(),
                            Value::String(encrypted.clone()),
                        );
                        if let Some(Value::String(item_id)) = item.get("id") {
                            if !item_id.is_empty() {
                                raw_item.insert("id".into(), Value::String(item_id.clone()));
                            }
                        }
                        if let Some(Value::Array(summary)) = item.get("summary") {
                            let mut raw_summary: Vec<Value> = Vec::new();
                            for part in summary {
                                if let Some(Value::String(text)) = part.get("text") {
                                    raw_summary.push(
                                        json!({"type": "summary_text", "text": text}),
                                    );
                                }
                            }
                            raw_item.insert("summary".into(), Value::Array(raw_summary));
                        }
                        reasoning_items_raw.push(Value::Object(raw_item));
                    }
                }
            }
            Some("function_call") => {
                if matches!(
                    item_status.as_deref(),
                    Some("queued") | Some("in_progress") | Some("incomplete")
                ) {
                    continue;
                }
                let fn_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let arguments = arguments_to_string(item.get("arguments"));
                let raw_call_id = item.get("call_id");
                let raw_item_id = item.get("id");
                let (embedded_call_id, _) = split_responses_tool_id(raw_item_id);
                let mut call_id = match raw_call_id {
                    Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
                    _ => embedded_call_id,
                };
                let call_id = match call_id.take() {
                    Some(c) if !c.trim().is_empty() => c.trim().to_string(),
                    _ => deterministic_call_id(&fn_name, &arguments, tool_calls.len()),
                };
                let response_item_id_in = as_str(raw_item_id);
                let response_item_id =
                    derive_responses_function_call_id(&call_id, response_item_id_in);
                tool_calls.push(NormalizedToolCall {
                    id: call_id.clone(),
                    call_id,
                    response_item_id,
                    r#type: "function".into(),
                    function_name: fn_name,
                    function_arguments: arguments,
                });
            }
            Some("custom_tool_call") => {
                let fn_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let arguments = arguments_to_string(item.get("input"));
                let raw_call_id = item.get("call_id");
                let raw_item_id = item.get("id");
                let (embedded_call_id, _) = split_responses_tool_id(raw_item_id);
                let mut call_id = match raw_call_id {
                    Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
                    _ => embedded_call_id,
                };
                let call_id = match call_id.take() {
                    Some(c) if !c.trim().is_empty() => c.trim().to_string(),
                    _ => deterministic_call_id(&fn_name, &arguments, tool_calls.len()),
                };
                let response_item_id_in = as_str(raw_item_id);
                let response_item_id =
                    derive_responses_function_call_id(&call_id, response_item_id_in);
                tool_calls.push(NormalizedToolCall {
                    id: call_id.clone(),
                    call_id,
                    response_item_id,
                    r#type: "function".into(),
                    function_name: fn_name,
                    function_arguments: arguments,
                });
            }
            _ => {}
        }
    }

    let mut final_text = content_parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if final_text.is_empty() {
        if let Some(Value::String(out_text)) = response.get("output_text") {
            final_text = out_text.trim().to_string();
        }
    }

    // Tool-call leak recovery.
    let mut leaked_tool_call_text = false;
    if !final_text.is_empty()
        && tool_calls.is_empty()
        && tool_call_leak_pattern().is_match(&final_text)
    {
        leaked_tool_call_text = true;
        log::warn!(
            "Codex response contains leaked tool-call text in assistant content \
             (no structured function_call items). Treating as incomplete so the \
             continuation path can re-elicit a proper tool call. Leaked snippet: {:?}",
            &final_text.chars().take(300).collect::<String>()
        );
        final_text = String::new();
    }

    let reasoning = if !reasoning_parts.is_empty() {
        Some(reasoning_parts.join("\n\n").trim().to_string())
    } else {
        None
    };

    let assistant_message = NormalizedAssistantMessage {
        content: final_text.clone(),
        tool_calls: tool_calls.clone(),
        reasoning,
        reasoning_content: None,
        reasoning_details: None,
        codex_reasoning_items: if reasoning_items_raw.is_empty() {
            None
        } else {
            Some(reasoning_items_raw)
        },
        codex_message_items: if message_items_raw.is_empty() {
            None
        } else {
            Some(message_items_raw)
        },
    };

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else if leaked_tool_call_text {
        "incomplete"
    } else if has_incomplete_items || (saw_commentary_phase && !saw_final_answer_phase) {
        "incomplete"
    } else if assistant_message.codex_reasoning_items.is_some() && final_text.is_empty() {
        "incomplete"
    } else {
        "stop"
    };

    Ok((assistant_message, finish_reason.to_string()))
}

/// `arguments` field of a function_call output item: non-str→json.dumps.
fn arguments_to_string(arguments: Option<&Value>) -> String {
    match arguments {
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_call_id_is_stable() {
        let a = deterministic_call_id("exec", "{\"a\":1}", 0);
        let b = deterministic_call_id("exec", "{\"a\":1}", 0);
        assert_eq!(a, b);
        assert!(a.starts_with("call_"));
        assert_eq!(a.len(), "call_".len() + 12);
        // Different index → different id.
        assert_ne!(a, deterministic_call_id("exec", "{\"a\":1}", 1));
    }

    #[test]
    fn split_tool_id_variants() {
        assert_eq!(
            split_responses_tool_id(Some(&json!("call_abc|fc_def"))),
            (Some("call_abc".into()), Some("fc_def".into()))
        );
        assert_eq!(
            split_responses_tool_id(Some(&json!("fc_xyz"))),
            (None, Some("fc_xyz".into()))
        );
        assert_eq!(
            split_responses_tool_id(Some(&json!("call_plain"))),
            (Some("call_plain".into()), None)
        );
        assert_eq!(split_responses_tool_id(Some(&json!("  "))), (None, None));
        assert_eq!(split_responses_tool_id(Some(&json!(5))), (None, None));
    }

    #[test]
    fn derive_function_call_id_rules() {
        assert_eq!(
            derive_responses_function_call_id("call_abc", Some("fc_keep")),
            "fc_keep"
        );
        assert_eq!(
            derive_responses_function_call_id("call_abc", None),
            "fc_abc"
        );
        assert_eq!(derive_responses_function_call_id("fc_zzz", None), "fc_zzz");
        // `call_` prefix is handled before sanitization, so the suffix is kept verbatim.
        assert_eq!(
            derive_responses_function_call_id("call_a!b@c", None),
            "fc_a!b@c"
        );
        // sanitization path: no recognized prefix, strip disallowed chars.
        assert_eq!(
            derive_responses_function_call_id("a!b@c", None),
            "fc_abc"
        );
        // empty → sha1-based fc_
        let r = derive_responses_function_call_id("", None);
        assert!(r.starts_with("fc_"));
        assert_eq!(r.len(), "fc_".len() + 24);
    }

    #[test]
    fn responses_tools_conversion() {
        let tools = json!([
            {"type": "function", "function": {"name": "do_it", "description": "d", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "  "}},
            {"not_function": true},
        ]);
        let out = responses_tools(Some(&tools)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["name"], json!("do_it"));
        assert_eq!(out[0]["strict"], json!(false));
        assert_eq!(out[0]["type"], json!("function"));
        // missing parameters → default
        let tools2 = json!([{"function": {"name": "x"}}]);
        let out2 = responses_tools(Some(&tools2)).unwrap();
        assert_eq!(out2[0]["parameters"], json!({"type": "object", "properties": {}}));
        assert!(responses_tools(None).is_none());
        assert!(responses_tools(Some(&json!([]))).is_none());
    }

    #[test]
    fn chat_content_parts_roles() {
        let content = json!([
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "http://x", "detail": "low"}},
            "tail",
        ]);
        let user = chat_content_to_responses_parts(&content, "user");
        assert_eq!(user[0], json!({"type": "input_text", "text": "hi"}));
        assert_eq!(
            user[1],
            json!({"type": "input_image", "image_url": "http://x", "detail": "low"})
        );
        assert_eq!(user[2], json!({"type": "input_text", "text": "tail"}));
        let asst = chat_content_to_responses_parts(&content, "assistant");
        assert_eq!(asst[0], json!({"type": "output_text", "text": "hi"}));
        // non-list → empty
        assert!(chat_content_to_responses_parts(&json!("x"), "user").is_empty());
    }

    #[test]
    fn summarize_message_with_images() {
        let content = json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "u"}},
        ]);
        assert_eq!(summarize_user_message_for_log(&content), "[1 image] look");
        assert_eq!(summarize_user_message_for_log(&json!("plain")), "plain");
        assert_eq!(summarize_user_message_for_log(&Value::Null), "");
        let two = json!([
            {"type": "image_url", "image_url": {"url": "u"}},
            {"type": "image_url", "image_url": {"url": "v"}},
        ]);
        assert_eq!(summarize_user_message_for_log(&two), "[2 images]");
    }

    #[test]
    fn chat_messages_to_input_tool_calls() {
        let messages = vec![
            json!({"role": "system", "content": "ignored"}),
            json!({"role": "user", "content": "hello"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_xyz", "function": {"name": "run", "arguments": {"k": "v"}}}
                ],
            }),
            json!({"role": "tool", "tool_call_id": "call_xyz", "content": "result"}),
        ];
        let items = chat_messages_to_responses_input(&messages);
        // system dropped; user, function_call, function_call_output
        assert_eq!(items[0], json!({"role": "user", "content": "hello"}));
        let fc = &items[1];
        assert_eq!(fc["type"], json!("function_call"));
        assert_eq!(fc["call_id"], json!("call_xyz"));
        assert_eq!(fc["name"], json!("run"));
        assert_eq!(fc["arguments"], json!("{\"k\":\"v\"}"));
        let out = items.last().unwrap();
        assert_eq!(out["type"], json!("function_call_output"));
        assert_eq!(out["call_id"], json!("call_xyz"));
        assert_eq!(out["output"], json!("result"));
    }

    #[test]
    fn reasoning_replay_strips_id_and_dedups() {
        let messages = vec![json!({
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "blob", "id": "r1"},
                {"type": "reasoning", "encrypted_content": "blob2", "id": "r1"},
            ],
        })];
        let items = chat_messages_to_responses_input(&messages);
        // second dropped via dedup; then empty assistant follower added
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], json!("reasoning"));
        assert!(items[0].get("id").is_none());
        assert_eq!(items[0]["encrypted_content"], json!("blob"));
        assert_eq!(items[1], json!({"role": "assistant", "content": ""}));
    }

    #[test]
    fn preflight_kwargs_happy_path() {
        let kwargs = json!({
            "model": " gpt-5 ",
            "instructions": "  ",
            "input": [{"role": "user", "content": "hi"}],
            "store": false,
            "temperature": 1,
            "max_output_tokens": 100,
            "tools": [
                {"type": "function", "name": "f", "parameters": {"type": "object"}}
            ],
        });
        let out = preflight_codex_api_kwargs(&kwargs, false).unwrap();
        assert_eq!(out["model"], json!("gpt-5"));
        // empty instructions → DEFAULT_AGENT_IDENTITY
        assert_eq!(out["instructions"], json!(DEFAULT_AGENT_IDENTITY));
        assert_eq!(out["store"], json!(false));
        assert_eq!(out["temperature"], json!(1.0));
        assert_eq!(out["max_output_tokens"], json!(100));
        assert_eq!(out["tools"][0]["strict"], json!(false));
    }

    #[test]
    fn preflight_rejects_missing_fields_and_store() {
        let r = preflight_codex_api_kwargs(&json!({"model": "m"}), false);
        assert!(r.unwrap_err().0.contains("missing required field(s): input, instructions"));

        let r2 = preflight_codex_api_kwargs(
            &json!({"model": "m", "instructions": "i", "input": [], "store": true}),
            false,
        );
        assert!(r2.unwrap_err().0.contains("'store' to be false"));

        let r3 = preflight_codex_api_kwargs(
            &json!({"model": "m", "instructions": "i", "input": [], "stream": true}),
            false,
        );
        assert!(r3.unwrap_err().0.contains("stream flag is only allowed"));
    }

    #[test]
    fn preflight_unexpected_field() {
        let r = preflight_codex_api_kwargs(
            &json!({"model": "m", "instructions": "i", "input": [], "bogus": 1, "also_bad": 2}),
            false,
        );
        let msg = r.unwrap_err().0;
        assert!(msg.contains("unsupported field(s): also_bad, bogus"));
    }

    #[test]
    fn preflight_input_reasoning_and_message() {
        let items = json!([
            {"type": "reasoning", "encrypted_content": "x", "id": "a"},
            {"type": "reasoning", "encrypted_content": "x", "id": "a"},
            {"type": "message", "role": "assistant", "status": "in-progress",
             "content": [{"type": "text", "text": "hello"}], "id": "m1", "phase": "final"},
        ]);
        let out = preflight_codex_input_items(&items).unwrap();
        // dedup drops the second reasoning
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["type"], json!("reasoning"));
        assert_eq!(out[0]["summary"], json!([]));
        assert!(out[0].get("id").is_none());
        let m = &out[1];
        assert_eq!(m["status"], json!("in_progress"));
        assert_eq!(m["content"][0], json!({"type": "output_text", "text": "hello"}));
        assert_eq!(m["id"], json!("m1"));
        assert_eq!(m["phase"], json!("final"));
    }

    #[test]
    fn normalize_message_response() {
        let resp = json!({
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Hi there"}]}
            ],
        });
        let (msg, fr) = normalize_codex_response(&resp).unwrap();
        assert_eq!(msg.content, "Hi there");
        assert_eq!(fr, "stop");
        assert!(msg.tool_calls.is_empty());
        assert!(msg.codex_message_items.is_some());
    }

    #[test]
    fn normalize_tool_call_response() {
        let resp = json!({
            "status": "completed",
            "output": [
                {"type": "function_call", "name": "run", "arguments": "{}",
                 "call_id": "call_abc", "id": "fc_abc"}
            ],
        });
        let (msg, fr) = normalize_codex_response(&resp).unwrap();
        assert_eq!(fr, "tool_calls");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].call_id, "call_abc");
        assert_eq!(msg.tool_calls[0].response_item_id, "fc_abc");
        assert_eq!(msg.tool_calls[0].function_name, "run");
    }

    #[test]
    fn normalize_leaked_tool_call_text() {
        let resp = json!({
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "assistant to=functions.exec_command {\"x\":1}"}]}
            ],
        });
        let (msg, fr) = normalize_codex_response(&resp).unwrap();
        assert_eq!(fr, "incomplete");
        assert_eq!(msg.content, "");
    }

    #[test]
    fn normalize_empty_output_uses_output_text() {
        let resp = json!({
            "status": "completed",
            "output": [],
            "output_text": "  fallback  ",
        });
        let (msg, fr) = normalize_codex_response(&resp).unwrap();
        assert_eq!(msg.content, "fallback");
        assert_eq!(fr, "stop");
    }

    #[test]
    fn normalize_failed_status_raises() {
        let resp = json!({
            "status": "failed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "x"}]}],
            "error": {"message": "boom"},
        });
        let r = normalize_codex_response(&resp);
        assert_eq!(r.unwrap_err().0, "boom");
    }

    #[test]
    fn normalize_reasoning_only_incomplete() {
        let resp = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "encrypted_content": "blob",
                 "summary": [{"text": "thinking"}]}
            ],
        });
        let (msg, fr) = normalize_codex_response(&resp).unwrap();
        assert_eq!(fr, "incomplete");
        assert!(msg.codex_reasoning_items.is_some());
        assert_eq!(msg.reasoning.as_deref(), Some("thinking"));
    }

    #[test]
    fn normalize_no_output_errors() {
        let resp = json!({"status": "completed"});
        assert_eq!(
            normalize_codex_response(&resp).unwrap_err().0,
            "Responses API returned no output items"
        );
    }

    #[test]
    fn message_status_normalization() {
        assert_eq!(
            normalize_responses_message_status(Some(&json!("IN-PROGRESS")), "completed"),
            "in_progress"
        );
        assert_eq!(
            normalize_responses_message_status(Some(&json!("weird")), "completed"),
            "completed"
        );
        assert_eq!(
            normalize_responses_message_status(None, "incomplete"),
            "incomplete"
        );
    }
}
