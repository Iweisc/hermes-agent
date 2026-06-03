//! Mistral tool call parser.
//!
//! Supports two formats depending on tokenizer version:
//! - Pre-v11: `content[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]`
//! - v11+:    `content[TOOL_CALLS]tool_name1{"arg": "val"}[TOOL_CALLS]tool_name2{"arg": "val"}`
//!
//! Based on VLLM's `MistralToolParser.extract_tool_calls()`.
//! The `[TOOL_CALLS]` token is the `bot_token` used by Mistral models.

use serde_json::Value;

/// A parsed tool call, mirroring OpenAI's `ChatCompletionMessageToolCall`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Mistral tool call IDs are 9-char alphanumeric strings.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name.
    pub name: String,
    /// The function arguments, as a JSON string.
    pub arguments: String,
}

impl ToolCall {
    fn function(id: String, name: String, arguments: String) -> Self {
        ToolCall {
            id,
            call_type: "function".to_string(),
            name,
            arguments,
        }
    }
}

/// Result of parsing: leftover content (if any) plus any extracted tool calls.
///
/// Mirrors the Python `ParseResult = tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: the text before `[TOOL_CALLS]`, or `None` if empty.
/// - `tool_calls`: the parsed tool calls, or `None` if none were found.
pub type ParseResult = (Option<String>, Option<Vec<ToolCall>>);

/// The `[TOOL_CALLS]` token -- may appear as different strings depending on tokenizer.
pub const BOT_TOKEN: &str = "[TOOL_CALLS]";

const ID_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Mistral tool call IDs are 9-char alphanumeric strings.
pub fn generate_mistral_id() -> String {
    // Simple, dependency-free PRNG seeded from system time + a per-call counter.
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static COUNTER: Cell<u64> = const { Cell::new(0) };
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let count = COUNTER.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    });

    // xorshift64* style mixing.
    let mut state = nanos
        ^ (count.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ (&count as *const u64 as u64);
    if state == 0 {
        state = 0x1234_5678_9ABC_DEF0;
    }

    let mut out = String::with_capacity(9);
    for _ in 0..9 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let idx = (r % ID_ALPHABET.len() as u64) as usize;
        out.push(ID_ALPHABET[idx] as char);
    }
    out
}

/// Parser for Mistral-format tool calls.
///
/// Detects format by checking if the content after `[TOOL_CALLS]` starts with `[`
/// (pre-v11 JSON array) or `{` or with a tool name (v11+ format).
#[derive(Debug, Default, Clone)]
pub struct MistralToolCallParser;

impl MistralToolCallParser {
    pub fn new() -> Self {
        MistralToolCallParser
    }

    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(BOT_TOKEN) {
            return (Some(text.to_string()), None);
        }

        // Mirror Python's `text.split(self.BOT_TOKEN)`: parts[0] is content,
        // parts[1:] are the raw tool-call segments.
        let mut parts = text.split(BOT_TOKEN);
        let content = parts.next().unwrap_or("").trim().to_string();
        let raw_tool_calls: Vec<&str> = parts.collect();

        let first_raw = raw_tool_calls
            .first()
            .map(|s| s.trim())
            .unwrap_or("");
        let is_pre_v11 = first_raw.starts_with('[') || first_raw.starts_with('{');

        let mut tool_calls: Vec<ToolCall> = Vec::new();

        if !is_pre_v11 {
            // v11+ format: tool_name{args}[TOOL_CALLS]tool_name2{args2}
            for raw in &raw_tool_calls {
                let raw = raw.trim();
                if raw.is_empty() || !raw.contains('{') {
                    continue;
                }

                let brace_idx = raw.find('{').unwrap();
                let tool_name = raw[..brace_idx].trim().to_string();
                let mut args_str = raw[brace_idx..].to_string();

                // Validate and clean the JSON arguments. Keep raw if parsing fails.
                if let Ok(parsed) = serde_json::from_str::<Value>(&args_str) {
                    if let Ok(reser) = serde_json::to_string(&parsed) {
                        args_str = reser;
                    }
                }

                tool_calls.push(ToolCall::function(
                    generate_mistral_id(),
                    tool_name,
                    args_str,
                ));
            }
        } else {
            // Pre-v11 format: [{"name": ..., "arguments": {...}}]
            match serde_json::from_str::<Value>(first_raw) {
                Ok(parsed) => {
                    let items: Vec<Value> = match parsed {
                        Value::Array(arr) => arr,
                        other => vec![other],
                    };
                    for tc in items {
                        if let Some(call) = tool_call_from_value(&tc) {
                            tool_calls.push(call);
                        }
                    }
                }
                Err(_) => {
                    // Fallback: extract JSON objects one at a time, advancing
                    // a byte at a time on failure (mirrors json.raw_decode).
                    extract_objects_fallback(first_raw, &mut tool_calls);
                }
            }
        }

        if tool_calls.is_empty() {
            return (Some(text.to_string()), None);
        }

        let content_out = if content.is_empty() {
            None
        } else {
            Some(content)
        };
        (content_out, Some(tool_calls))
    }
}

/// Build a `ToolCall` from a JSON value of the shape
/// `{"name": ..., "arguments": {...}}`. Returns `None` if `name` is missing.
fn tool_call_from_value(tc: &Value) -> Option<ToolCall> {
    let obj = tc.as_object()?;
    let name = obj.get("name")?;
    let name = match name {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    let args = match obj.get("arguments") {
        // If arguments is an object (dict), serialize to a JSON string.
        Some(Value::Object(_)) | Some(Value::Array(_)) => {
            serde_json::to_string(obj.get("arguments").unwrap()).unwrap_or_else(|_| "{}".to_string())
        }
        // A string is kept as-is (Python: only dicts are json.dumps'd).
        Some(Value::String(s)) => s.clone(),
        // Any other scalar: keep the JSON representation.
        Some(other) => other.to_string(),
        // Missing: default {} (Python default is {} dict -> json.dumps -> "{}").
        None => "{}".to_string(),
    };

    Some(ToolCall::function(generate_mistral_id(), name, args))
}

/// Fallback object extraction: scan the string, decoding the longest valid JSON
/// value starting at each index, collecting dict objects that contain `"name"`.
fn extract_objects_fallback(s: &str, out: &mut Vec<ToolCall>) {
    let bytes = s.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        // Use a streaming deserializer to peel off one JSON value and learn
        // where it ended (analogous to json.JSONDecoder.raw_decode).
        let mut de = serde_json::Deserializer::from_str(&s[idx..]).into_iter::<Value>();
        match de.next() {
            Some(Ok(obj)) => {
                let consumed = de.byte_offset();
                if obj.is_object() && obj.get("name").is_some() {
                    if let Some(call) = tool_call_from_value(&obj) {
                        out.push(call);
                    }
                }
                if consumed == 0 {
                    idx += 1;
                } else {
                    idx += consumed;
                }
            }
            _ => {
                idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn no_bot_token_returns_text() {
        let p = MistralToolCallParser::new();
        let (content, calls) = p.parse("hello world");
        assert_eq!(content, Some("hello world".to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn generate_id_is_9_alnum() {
        let id = generate_mistral_id();
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
        // Two consecutive ids should generally differ.
        let id2 = generate_mistral_id();
        assert_eq!(id2.len(), 9);
    }

    #[test]
    fn v11_single_tool_call() {
        let p = MistralToolCallParser::new();
        let text = "some content[TOOL_CALLS]get_weather{\"city\": \"Paris\"}";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some("some content".to_string()));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["city"], "Paris");
    }

    #[test]
    fn v11_multiple_tool_calls() {
        let p = MistralToolCallParser::new();
        let text = "[TOOL_CALLS]a{\"x\": 1}[TOOL_CALLS]b{\"y\": 2}";
        let (content, calls) = p.parse(text);
        assert_eq!(content, None);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        let v0: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v0["x"], 1);
        let v1: Value = serde_json::from_str(&calls[1].arguments).unwrap();
        assert_eq!(v1["y"], 2);
    }

    #[test]
    fn v11_invalid_json_kept_raw() {
        let p = MistralToolCallParser::new();
        let text = "[TOOL_CALLS]foo{not valid json";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "foo");
        assert_eq!(calls[0].arguments, "{not valid json");
    }

    #[test]
    fn pre_v11_array() {
        let p = MistralToolCallParser::new();
        let text = "[TOOL_CALLS] [{\"name\": \"f1\", \"arguments\": {\"a\": 1}}, {\"name\": \"f2\", \"arguments\": {\"b\": 2}}]";
        let (content, calls) = p.parse(text);
        assert_eq!(content, None);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f1");
        let v0: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v0["a"], 1);
        assert_eq!(calls[1].name, "f2");
    }

    #[test]
    fn pre_v11_single_object() {
        let p = MistralToolCallParser::new();
        let text = "content[TOOL_CALLS]{\"name\": \"solo\", \"arguments\": {\"k\": \"v\"}}";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some("content".to_string()));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "solo");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["k"], "v");
    }

    #[test]
    fn pre_v11_missing_name_skipped() {
        let p = MistralToolCallParser::new();
        let text = "[TOOL_CALLS][{\"arguments\": {\"a\": 1}}, {\"name\": \"ok\"}]";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ok");
        // Missing arguments default to "{}".
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn pre_v11_string_arguments_kept() {
        let p = MistralToolCallParser::new();
        let text = "[TOOL_CALLS]{\"name\": \"f\", \"arguments\": \"already_string\"}";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "already_string");
    }

    #[test]
    fn fallback_concatenated_objects() {
        let p = MistralToolCallParser::new();
        // Not valid as a single JSON value (two objects back-to-back), and does
        // not start with '[' after a leading object char, forcing the fallback.
        let text = "[TOOL_CALLS]{\"name\": \"f1\", \"arguments\": {\"a\": 1}} {\"name\": \"f2\", \"arguments\": {\"b\": 2}}";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f1");
        assert_eq!(calls[1].name, "f2");
    }

    #[test]
    fn no_tool_calls_returns_original_text() {
        let p = MistralToolCallParser::new();
        // v11 path but segment has no '{', so nothing parses.
        let text = "abc[TOOL_CALLS]justaname";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some(text.to_string()));
        assert!(calls.is_none());
    }
}
