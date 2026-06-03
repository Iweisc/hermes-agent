//! Longcat Flash Chat tool call parser.
//!
//! Format: `<longcat_tool_call>{"name": "func", "arguments": {...}}</longcat_tool_call>`
//! Same as Hermes but uses `<longcat_tool_call>` tags instead of `<tool_call>`.
//! Based on VLLM's `LongcatFlashToolParser` (extends `Hermes2ProToolParser`).
//!
//! Matches `<longcat_tool_call>...</longcat_tool_call>` tags containing JSON with
//! `"name"` and `"arguments"`. Also handles an unclosed `<longcat_tool_call>` at
//! end-of-string (truncated generation).
//!
//! Port of `environments/tool_call_parsers/longcat_parser.py`.

use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

/// A parsed tool call, mirroring OpenAI's `ChatCompletionMessageToolCall`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Tool call IDs look like `call_<8 hex chars>`.
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
/// - `content`: the text before the first `<longcat_tool_call>` tag, or `None` if empty.
/// - `tool_calls`: the parsed tool calls, or `None` if none were found.
pub type ParseResult = (Option<String>, Option<Vec<ToolCall>>);

/// The opening tag marking the start of a Longcat tool call.
pub const LONGCAT_TOOL_CALL_TAG: &str = "<longcat_tool_call>";

/// Matches both closed and unclosed `longcat_tool_call` tags.
///
/// Mirrors the Python regex
/// `r"<longcat_tool_call>\s*(.*?)\s*</longcat_tool_call>|<longcat_tool_call>\s*(.*)"`
/// compiled with `re.DOTALL`. `(?s)` enables dotall so `.` matches newlines.
fn pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?s)<longcat_tool_call>\s*(.*?)\s*</longcat_tool_call>|<longcat_tool_call>\s*(.*)",
        )
        .expect("valid longcat tool_call regex")
    })
}

const ID_HEX: &[u8] = b"0123456789abcdef";

/// Tool call IDs are `call_` followed by 8 hex chars (first 8 of a uuid4 hex).
pub fn generate_longcat_id() -> String {
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

    let mut out = String::with_capacity(13);
    out.push_str("call_");
    for _ in 0..8 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let idx = (r % ID_HEX.len() as u64) as usize;
        out.push(ID_HEX[idx] as char);
    }
    out
}

/// Parser for Longcat-format tool calls.
#[derive(Debug, Default, Clone)]
pub struct LongcatToolCallParser;

impl LongcatToolCallParser {
    pub fn new() -> Self {
        LongcatToolCallParser
    }

    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(LONGCAT_TOOL_CALL_TAG) {
            return (Some(text.to_string()), None);
        }

        // The Python implementation wraps the whole body in `try/except` and
        // returns the original text on any error. We achieve the same by
        // delegating to a fallible helper.
        match parse_inner(text) {
            Some(result) => result,
            None => (Some(text.to_string()), None),
        }
    }
}

/// Inner parse that returns `None` to signal a failure equivalent to Python's
/// `except Exception: return text, None`.
fn parse_inner(text: &str) -> Option<ParseResult> {
    let re = pattern();

    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for caps in re.captures_iter(text) {
        // `findall` with two groups yields a tuple (closed, unclosed). The first
        // non-empty group wins, matching `match[0] if match[0] else match[1]`.
        // When a group did not participate, Python yields "" for it.
        let closed = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let unclosed = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let raw_json = if !closed.is_empty() { closed } else { unclosed };

        if raw_json.trim().is_empty() {
            continue;
        }

        // json.loads -- any parse error aborts the whole parse in Python.
        let tc_data: Value = serde_json::from_str(raw_json).ok()?;

        // Python uses `tc_data["name"]` directly: a missing "name" key raises
        // KeyError, which is caught by the outer try/except, returning the
        // original text. So a missing name aborts the entire parse (return None
        // here -> caller returns the original text).
        let name = match tc_data.get("name") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => return None,
        };

        // arguments default = {}; serialized with ensure_ascii=False (serde_json
        // does not escape non-ASCII, matching that flag).
        let arguments = match tc_data.get("arguments") {
            Some(v) => serde_json::to_string(v).ok()?,
            None => "{}".to_string(),
        };

        tool_calls.push(ToolCall::function(generate_longcat_id(), name, arguments));
    }

    if tool_calls.is_empty() {
        return Some((Some(text.to_string()), None));
    }

    // Content is everything before the first <longcat_tool_call> tag.
    let content = match text.find(LONGCAT_TOOL_CALL_TAG) {
        Some(idx) => text[..idx].trim().to_string(),
        None => String::new(),
    };
    let content_out = if content.is_empty() {
        None
    } else {
        Some(content)
    };

    Some((content_out, Some(tool_calls)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tag_returns_text() {
        let p = LongcatToolCallParser::new();
        let (content, calls) = p.parse("hello world");
        assert_eq!(content, Some("hello world".to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn generate_id_format() {
        let id = generate_longcat_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 13);
        let hex = &id[5..];
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn single_closed_tool_call() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, None);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].id.starts_with("call_"));
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["city"], "Paris");
    }

    #[test]
    fn content_before_tool_call_kept() {
        let p = LongcatToolCallParser::new();
        let text = "Let me check.  <longcat_tool_call>{\"name\": \"f\", \"arguments\": {}}</longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some("Let me check.".to_string()));
        assert_eq!(calls.unwrap().len(), 1);
    }

    #[test]
    fn multiple_tool_calls() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}</longcat_tool_call>\n<longcat_tool_call>{\"name\": \"b\", \"arguments\": {\"y\": 2}}</longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, None);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        let v1: Value = serde_json::from_str(&calls[1].arguments).unwrap();
        assert_eq!(v1["y"], 2);
    }

    #[test]
    fn unclosed_tool_call_at_end() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"name\": \"trunc\", \"arguments\": {\"k\": \"v\"}}";
        let (content, calls) = p.parse(text);
        assert_eq!(content, None);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "trunc");
    }

    #[test]
    fn missing_arguments_defaults_empty_object() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"name\": \"noargs\"}</longcat_tool_call>";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn missing_name_aborts_and_returns_original_text() {
        // Python indexes `tc_data["name"]` -> KeyError -> outer except returns
        // the original text. So a missing name aborts the entire parse.
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"arguments\": {\"a\": 1}}</longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some(text.to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn invalid_json_returns_original_text() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{not valid json}</longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some(text.to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn empty_tool_call_body_skipped() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>  </longcat_tool_call>";
        let (content, calls) = p.parse(text);
        assert_eq!(content, Some(text.to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn dotall_multiline_arguments() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\n  \"name\": \"ml\",\n  \"arguments\": {\n    \"a\": 1\n  }\n}</longcat_tool_call>";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ml");
    }

    #[test]
    fn non_ascii_arguments_preserved() {
        let p = LongcatToolCallParser::new();
        let text = "<longcat_tool_call>{\"name\": \"f\", \"arguments\": {\"city\": \"São Paulo\"}}</longcat_tool_call>";
        let (_content, calls) = p.parse(text);
        let calls = calls.unwrap();
        // ensure_ascii=False -> non-ASCII kept literally, not escaped.
        assert!(calls[0].arguments.contains("São Paulo"));
    }
}
