//! Qwen 2.5 tool call parser (native Rust port of
//! `environments/tool_call_parsers/qwen_parser.py`).
//!
//! Qwen 2.5 uses the exact same `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
//! format as Hermes. The Python original is a trivial subclass of
//! `HermesToolCallParser` that registers under the name `"qwen"` and inherits all
//! behaviour. This module reproduces that behaviour directly so the parser is
//! self-contained.
//!
//! Behaviour mirrored from `hermes_parser.py::HermesToolCallParser.parse`:
//! - If the text contains no `<tool_call>` token, return the text unchanged.
//! - Match every `<tool_call>...</tool_call>` block (DOTALL), plus a final
//!   *unclosed* `<tool_call>...` block at end-of-string (truncated generation).
//! - For each block, JSON-parse the inner payload. Skip blocks whose payload is
//!   empty/whitespace or lacks a `"name"` key.
//! - Build a tool call with id `call_<8 hex chars>`, type `"function"`, and
//!   `arguments` re-serialised as JSON (with `ensure_ascii=False` semantics:
//!   non-ASCII characters are NOT escaped). Missing `arguments` defaults to `{}`.
//! - Content is everything before the first `<tool_call>` token, stripped;
//!   `None` if empty.
//! - Any error (e.g. invalid JSON) falls back to returning the original text.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

/// A parsed tool call, matching the shape of OpenAI's
/// `ChatCompletionMessageToolCall` that the Python original constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Unique id, formatted as `call_<8 hex chars>`.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name.
    pub name: String,
    /// JSON-encoded argument object (with `ensure_ascii=False` semantics:
    /// non-ASCII characters are NOT escaped).
    pub arguments: String,
}

/// Result of [`QwenToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = Tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: text with tool-call markup stripped, or `None` if there is no
///   leading content.
/// - `tool_calls`: parsed tool calls, or `None` when none were found (in which
///   case `content` carries the original text unchanged).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ParseResult {
    /// Helper for the "no tool calls, return text as-is" case
    /// (`return text, None` in Python).
    fn passthrough(text: &str) -> Self {
        ParseResult {
            content: Some(text.to_string()),
            tool_calls: None,
        }
    }
}

const START_TOKEN: &str = "<tool_call>";

/// Matches both closed and unclosed tool_call tags, DOTALL.
///
/// Equivalent to the Python:
/// `r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)"` with `re.DOTALL`.
fn tool_call_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)").unwrap()
    })
}

/// Generate an id of the form `call_<8 lowercase hex chars>`, matching
/// `f"call_{uuid.uuid4().hex[:8]}"`.
fn generate_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Pseudo-random hex without pulling in a uuid crate.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let addr = &nanos as *const u128 as u128;
    let mut state = nanos ^ addr.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = String::with_capacity(13);
    out.push_str("call_");
    for _ in 0..8 {
        // xorshift-style mixing.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let nibble = (state & 0xF) as u8;
        let c = match nibble {
            0..=9 => (b'0' + nibble) as char,
            _ => (b'a' + (nibble - 10)) as char,
        };
        out.push(c);
    }
    out
}

/// Serialise a `serde_json::Value` with `ensure_ascii=False` semantics.
///
/// `serde_json::to_string` already emits non-ASCII characters literally (it does
/// not escape them), so this matches Python's `json.dumps(..., ensure_ascii=False)`.
fn dumps(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

/// Parser for Qwen 2.5 tool calls (Hermes `<tool_call>` JSON format).
#[derive(Debug, Default, Clone, Copy)]
pub struct QwenToolCallParser;

impl QwenToolCallParser {
    pub fn new() -> Self {
        QwenToolCallParser
    }

    /// Parse raw model output text for tool calls.
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(START_TOKEN) {
            return ParseResult::passthrough(text);
        }

        // The Python code wraps everything in try/except and returns `(text, None)`
        // on any failure. Here, parse errors are handled per-block (skipped),
        // which matches the observable outcome: a fully-unparseable single block
        // yields no tool calls -> passthrough.
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for caps in tool_call_regex().captures_iter(text) {
            // group 1 = closed content, group 2 = unclosed content.
            let raw_json = caps
                .get(1)
                .filter(|m| !m.as_str().is_empty())
                .or_else(|| caps.get(2))
                .map(|m| m.as_str())
                .unwrap_or("");

            if raw_json.trim().is_empty() {
                continue;
            }

            let tc_data: Value = match serde_json::from_str(raw_json) {
                Ok(v) => v,
                // Python: json.loads raising propagates to the outer except,
                // aborting the whole parse and returning the original text.
                Err(_) => return ParseResult::passthrough(text),
            };

            let name = match tc_data.get("name").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                // No "name" key (or not a string) -> skip this block.
                None => {
                    if tc_data.get("name").is_none() {
                        continue;
                    }
                    // "name" present but non-string: Python would pass the raw
                    // value into Function(name=...); reproduce by stringifying.
                    tc_data
                        .get("name")
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                }
            };

            let arguments_value = tc_data
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

            tool_calls.push(ToolCall {
                id: generate_call_id(),
                call_type: "function".to_string(),
                name,
                arguments: dumps(&arguments_value),
            });
        }

        if tool_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        // Content is everything before the first <tool_call> tag, stripped.
        let content = match text.find(START_TOKEN) {
            Some(idx) => {
                let trimmed = text[..idx].trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            None => None,
        };

        ParseResult {
            content,
            tool_calls: Some(tool_calls),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_value(tc: &ToolCall) -> Value {
        serde_json::from_str(&tc.arguments).unwrap()
    }

    #[test]
    fn no_tool_call_token_passthrough() {
        let text = "just plain text with no tool call";
        let res = QwenToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some(text));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn single_tool_call_basic() {
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("expected tool calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(tcs[0].call_type, "function");
        assert!(tcs[0].id.starts_with("call_"));
        assert_eq!(tcs[0].id.len(), "call_".len() + 8);
        let args = args_value(&tcs[0]);
        assert_eq!(args["city"], Value::String("Paris".to_string()));
        // Block at index 0 -> no leading content.
        assert!(res.content.is_none());
    }

    #[test]
    fn content_before_tool_call() {
        let text = r#"Sure, let me check.
<tool_call>{"name": "ping", "arguments": {"host": "localhost"}}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("Sure, let me check."));
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "ping");
    }

    #[test]
    fn multiple_tool_calls() {
        let text = r#"<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>
<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "a");
        assert_eq!(tcs[1].name, "b");
        assert_eq!(args_value(&tcs[0])["x"], Value::from(1));
        assert_eq!(args_value(&tcs[1])["y"], Value::from(2));
    }

    #[test]
    fn missing_arguments_defaults_to_empty_object() {
        let text = r#"<tool_call>{"name": "noop"}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "noop");
        assert_eq!(tcs[0].arguments, "{}");
    }

    #[test]
    fn block_without_name_is_skipped() {
        // Single block lacking "name" -> no tool calls -> passthrough.
        let text = r#"<tool_call>{"arguments": {"x": 1}}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        assert!(res.tool_calls.is_none());
        assert_eq!(res.content.as_deref(), Some(text));
    }

    #[test]
    fn mixed_valid_and_nameless_blocks() {
        let text = r#"<tool_call>{"arguments": {}}</tool_call><tool_call>{"name": "ok"}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "ok");
    }

    #[test]
    fn invalid_json_falls_back_to_text() {
        let text = r#"<tool_call>{not valid json}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        assert!(res.tool_calls.is_none());
        assert_eq!(res.content.as_deref(), Some(text));
    }

    #[test]
    fn unclosed_tool_call_at_end() {
        let text = r#"<tool_call>{"name": "f", "arguments": {"k": "v"}}"#;
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("should parse unclosed block");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "f");
        assert_eq!(args_value(&tcs[0])["k"], Value::String("v".to_string()));
    }

    #[test]
    fn empty_tool_call_block_skipped() {
        let text = "<tool_call></tool_call>";
        let res = QwenToolCallParser::new().parse(text);
        assert!(res.tool_calls.is_none());
        assert_eq!(res.content.as_deref(), Some(text));
    }

    #[test]
    fn non_ascii_arguments_not_escaped() {
        let text = r#"<tool_call>{"name": "say", "arguments": {"msg": "héllo"}}</tool_call>"#;
        let res = QwenToolCallParser::new().parse(text);
        let tc = &res.tool_calls.unwrap()[0];
        // ensure_ascii=False semantics: literal non-ascii char in the JSON.
        assert!(tc.arguments.contains('é'));
    }

    #[test]
    fn whitespace_inside_tags_trimmed_by_regex() {
        let text = "<tool_call>\n  {\"name\": \"f\", \"arguments\": {}}  \n</tool_call>";
        let res = QwenToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "f");
    }
}
