//! Llama 3.x / 4 tool call parser (native Rust port of
//! `environments/tool_call_parsers/llama_parser.py`).
//!
//! Format: the model outputs JSON objects with `"name"` and `"arguments"`
//! (or `"parameters"`) keys. The output may be preceded by a `<|python_tag|>`
//! token. Multiple JSON objects may be separated by free-form content or
//! semicolons.
//!
//! This mirrors VLLM's `Llama3JsonToolParser.extract_tool_calls()`. It scans
//! the text for `{` braces and, at each candidate position, attempts to decode
//! a single JSON value (the equivalent of Python's
//! `json.JSONDecoder().raw_decode`). Successfully decoded objects that contain
//! a non-empty `"name"` and a non-null `"arguments"`/`"parameters"` become tool
//! calls; the scanner then skips past the consumed region so it does not
//! re-parse braces nested inside an already-extracted object.

use serde_json::Value;

/// A single extracted tool call, mirroring the relevant fields of OpenAI's
/// `ChatCompletionMessageToolCall`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Synthetic id of the form `call_<8 hex chars>`.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name.
    pub name: String,
    /// The function arguments, serialised to a JSON string.
    pub arguments: String,
}

/// Result of [`parse`]: `(content, tool_calls)`.
///
/// * `content` is the text before the first tool call (stripped), or `None`
///   when there is no leading content / no tool calls were found and the
///   original text is returned as content.
/// * `tool_calls` is `Some(..)` when at least one tool call was extracted,
///   otherwise `None`.
pub type ParseResult = (Option<String>, Option<Vec<ToolCall>>);

/// The Llama JSON tool-call parser.
///
/// Registered under the names `llama3_json` and `llama4_json` in the Python
/// implementation; [`PARSER_NAMES`] preserves those identifiers.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlamaToolCallParser;

/// Parser names this implementation answers to.
pub const PARSER_NAMES: &[&str] = &["llama3_json", "llama4_json"];

/// Begin-of-tool token that may precede the JSON payload.
pub const BOT_TOKEN: &str = "<|python_tag|>";

impl LlamaToolCallParser {
    /// Construct a new parser.
    pub fn new() -> Self {
        LlamaToolCallParser
    }

    /// Parse raw model output for Llama-format tool calls.
    ///
    /// See [`parse`] for the exact semantics; this is a thin method wrapper.
    pub fn parse(&self, text: &str) -> ParseResult {
        parse(text)
    }
}

/// Generate a synthetic tool-call id of the form `call_<8 hex chars>`.
///
/// The Python version uses `uuid.uuid4().hex[:8]`. The exact value is not
/// semantically meaningful (callers treat ids as opaque), so we derive 8 hex
/// characters from a process-local counter mixed with the system clock to keep
/// ids unique within a run without pulling in a uuid dependency.
fn generate_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // Mix counter and time, then take the low 32 bits -> 8 hex chars.
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(counter.wrapping_mul(0x1000_0000_1));
    format!("call_{:08x}", (mixed as u32))
}

/// Attempt to decode a single JSON value starting at the beginning of `s`.
///
/// This is the Rust analogue of Python's `json.JSONDecoder().raw_decode`: it
/// returns the parsed [`Value`] together with the byte offset (relative to `s`)
/// just past the consumed JSON, ignoring any trailing text. Returns `None` if
/// the prefix of `s` is not a valid JSON value.
fn raw_decode(s: &str) -> Option<(Value, usize)> {
    let mut stream = serde_json::Deserializer::from_str(s).into_iter::<Value>();
    match stream.next() {
        Some(Ok(value)) => Some((value, stream.byte_offset())),
        _ => None,
    }
}

/// Parse raw model output text for Llama-format tool calls.
///
/// Returns `(content, tool_calls)`:
/// * On success, `tool_calls` is `Some(non-empty Vec)` and `content` is the
///   stripped text preceding the first tool call (or `None` if there is none).
/// * When no tool calls are found, returns `(Some(text), None)` — the original
///   text echoed back as content, matching the Python behaviour of returning
///   `(text, None)`.
pub fn parse(text: &str) -> ParseResult {
    // Quick check: need either the bot token or a JSON brace.
    if !text.contains(BOT_TOKEN) && !text.contains('{') {
        return (Some(text.to_string()), None);
    }

    let bytes = text.as_bytes();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    // Byte index where the last parsed JSON ended; -1 sentinel via Option/i64.
    // Python uses `end_index = -1` and skips braces with `start <= end_index`.
    let mut end_index: i64 = -1;

    // Iterate over every `{` brace position (byte offset).
    for (start, _) in text.match_indices('{') {
        // Skip braces inside a previously parsed JSON object.
        if (start as i64) <= end_index {
            continue;
        }

        let slice = &text[start..];
        let (obj, json_end) = match raw_decode(slice) {
            Some(v) => v,
            None => continue,
        };
        end_index = (start + json_end) as i64;

        // Must be an object with a "name" and "arguments"/"parameters".
        let map = match obj.as_object() {
            Some(m) => m,
            None => continue,
        };

        // name: must be a non-empty/"truthy" value. Python's `if not name`
        // treats empty string, null, false, 0, and missing as falsy.
        let name = match map.get("name") {
            Some(v) => v,
            None => continue,
        };
        let name_str = match value_is_truthy_name(name) {
            Some(s) => s,
            None => continue,
        };

        // args: obj.get("arguments", obj.get("parameters")); None if absent or null.
        let args_val = map
            .get("arguments")
            .or_else(|| map.get("parameters"));
        let args_val = match args_val {
            Some(v) if !v.is_null() => v,
            _ => continue,
        };

        // Normalise arguments to a JSON string. If it is already a JSON string,
        // use its inner value verbatim; otherwise serialise the value.
        let args_str = match args_val {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };

        tool_calls.push(ToolCall {
            id: generate_call_id(),
            call_type: "function".to_string(),
            name: name_str,
            arguments: args_str,
        });

        // Silence unused warning for `bytes` on platforms where it is optimised
        // out; it documents that offsets are byte-based.
        let _ = bytes;
    }

    if tool_calls.is_empty() {
        return (Some(text.to_string()), None);
    }

    // Content is everything before the first tool call.
    // Python: first_tc_start = text.find("{"); if BOT_TOKEN present, use its index.
    let mut first_tc_start = text.find('{').map(|i| i as i64).unwrap_or(-1);
    if let Some(bot_idx) = text.find(BOT_TOKEN) {
        first_tc_start = bot_idx as i64;
    }

    let content = if first_tc_start > 0 {
        let prefix = &text[..first_tc_start as usize];
        Some(prefix.trim().to_string())
    } else {
        None
    };

    (content, Some(tool_calls))
}

/// Return the string form of `name` if it is "truthy" per Python semantics,
/// otherwise `None`.
///
/// In the Python original, `name = obj.get("name")` is checked with
/// `if not name`. A name is usable when it is a non-empty string. Numbers and
/// booleans are technically truthy in Python but the downstream `Function`
/// type expects a string, so we mirror the practical behaviour: accept
/// non-empty strings; reject everything else (empty string, null, false, 0).
fn value_is_truthy_name(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_brace_no_token_returns_text() {
        let (content, calls) = parse("just some plain text");
        assert_eq!(content.as_deref(), Some("just some plain text"));
        assert!(calls.is_none());
    }

    #[test]
    fn single_tool_call_arguments() {
        let text = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let (content, calls) = parse(text);
        assert!(content.is_none());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].id.starts_with("call_"));
        // arguments normalised to JSON string
        let parsed: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(parsed["city"], "Paris");
    }

    #[test]
    fn parameters_alias_supported() {
        let text = r#"{"name": "f", "parameters": {"x": 1}}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls.len(), 1);
        let parsed: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(parsed["x"], 1);
    }

    #[test]
    fn python_tag_prefix_stripped_from_content() {
        let text = r#"<|python_tag|>{"name": "f", "arguments": {}}"#;
        let (content, calls) = parse(text);
        // BOT token is at index 0 so first_tc_start == 0 -> content None.
        assert!(content.is_none());
        assert_eq!(calls.unwrap().len(), 1);
    }

    #[test]
    fn leading_content_preserved_and_stripped() {
        let text = r#"Here you go:  {"name": "f", "arguments": {"a": true}}"#;
        let (content, calls) = parse(text);
        assert_eq!(content.as_deref(), Some("Here you go:"));
        assert_eq!(calls.unwrap().len(), 1);
    }

    #[test]
    fn multiple_tool_calls_separated_by_content() {
        let text = r#"{"name": "a", "arguments": {"x": 1}} and then {"name": "b", "arguments": {"y": 2}}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn nested_braces_not_double_parsed() {
        // The inner {"k": "v"} brace must be skipped because it lives inside
        // the already-consumed outer object.
        let text = r#"{"name": "a", "arguments": {"k": "v", "nested": {"deep": 1}}}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn object_missing_name_skipped() {
        let text = r#"{"arguments": {"x": 1}}"#;
        let (content, calls) = parse(text);
        assert_eq!(content.as_deref(), Some(text));
        assert!(calls.is_none());
    }

    #[test]
    fn object_with_null_args_skipped() {
        let text = r#"{"name": "f", "arguments": null}"#;
        let (content, calls) = parse(text);
        assert_eq!(content.as_deref(), Some(text));
        assert!(calls.is_none());
    }

    #[test]
    fn empty_name_skipped() {
        let text = r#"{"name": "", "arguments": {"x": 1}}"#;
        let (_content, calls) = parse(text);
        assert!(calls.is_none());
    }

    #[test]
    fn args_string_passed_through() {
        // When arguments is already a JSON string, the inner string is used.
        let text = r#"{"name": "f", "arguments": "{\"x\": 1}"}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls[0].arguments, r#"{"x": 1}"#);
    }

    #[test]
    fn args_non_dict_non_string_serialised() {
        let text = r#"{"name": "f", "arguments": [1, 2, 3]}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls[0].arguments, "[1,2,3]");
    }

    #[test]
    fn invalid_json_brace_skipped() {
        let text = r#"not json { broken and {"name": "f", "arguments": {}}"#;
        let (_content, calls) = parse(text);
        let calls = calls.expect("tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn method_wrapper_matches_free_fn() {
        let p = LlamaToolCallParser::new();
        let text = r#"{"name": "f", "arguments": {}}"#;
        let (_c1, calls1) = p.parse(text);
        let (_c2, calls2) = parse(text);
        assert_eq!(calls1.unwrap().len(), calls2.unwrap().len());
    }

    #[test]
    fn parser_names_registered() {
        assert!(PARSER_NAMES.contains(&"llama3_json"));
        assert!(PARSER_NAMES.contains(&"llama4_json"));
    }
}
