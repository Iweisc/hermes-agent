//! DeepSeek V3 tool call parser.
//!
//! Format uses special unicode tokens:
//! ```text
//! <｜tool▁calls▁begin｜>
//! <｜tool▁call▁begin｜>type<｜tool▁sep｜>function_name
//! ```json
//! {"arg": "value"}
//! ```
//! <｜tool▁call▁end｜>
//! <｜tool▁calls▁end｜>
//! ```
//!
//! Port of `environments/tool_call_parsers/deepseek_v3_parser.py`.
//! Supports multiple simultaneous tool calls (Issue #989).

use regex::Regex;
use std::sync::OnceLock;

/// A parsed tool call, mirroring OpenAI's `ChatCompletionMessageToolCall`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// DeepSeek tool call IDs look like `call_<8 hex chars>`.
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
/// - `content`: the text before the first tool-call block, or `None` if empty.
/// - `tool_calls`: the parsed tool calls, or `None` if none were found.
pub type ParseResult = (Option<String>, Option<Vec<ToolCall>>);

/// The token that marks the beginning of a tool-call sequence.
pub const START_TOKEN: &str = "<｜tool▁calls▁begin｜>";

/// Lazily-compiled regex matching a single tool-call block.
///
/// Mirrors the Python `PATTERN` (using `\s*` rather than literal newlines for
/// robustness against formatting variations), compiled with `DOTALL` so `.`
/// matches across newlines.
fn pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?s)<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)\s*```json\s*(?P<function_arguments>.*?)\s*```\s*<｜tool▁call▁end｜>",
        )
        .expect("DeepSeek V3 tool-call pattern is valid")
    })
}

/// Generate a DeepSeek-style call id: `call_` followed by 8 lowercase hex chars.
///
/// Mirrors Python's `f"call_{uuid.uuid4().hex[:8]}"`.
pub fn generate_call_id() -> String {
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

    // xorshift64* style mixing for a small, dependency-free random hex string.
    let mut state = nanos
        ^ (count.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ (&count as *const u64 as u64);
    if state == 0 {
        state = 0x1234_5678_9ABC_DEF0;
    }
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);

    format!("call_{:08x}", (r & 0xFFFF_FFFF) as u32)
}

/// Parser for DeepSeek V3 format tool calls.
#[derive(Debug, Default, Clone)]
pub struct DeepSeekV3ToolCallParser;

impl DeepSeekV3ToolCallParser {
    pub fn new() -> Self {
        DeepSeekV3ToolCallParser
    }

    /// Parse `text`, extracting all available tool calls.
    ///
    /// If the start token is absent, or no tool-call blocks match, returns the
    /// original text and no tool calls. Otherwise returns the content preceding
    /// the first tool-call block (stripped; `None` if empty) and the calls.
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(START_TOKEN) {
            return (Some(text.to_string()), None);
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for caps in pattern().captures_iter(text) {
            let func_name = caps
                .name("function_name")
                .map(|m| m.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let func_args = caps
                .name("function_arguments")
                .map(|m| m.as_str())
                .unwrap_or("")
                .trim()
                .to_string();

            tool_calls.push(ToolCall::function(generate_call_id(), func_name, func_args));
        }

        if tool_calls.is_empty() {
            return (Some(text.to_string()), None);
        }

        // Content is the text before the first start token.
        let content_index = text.find(START_TOKEN).unwrap_or(0);
        let content = text[..content_index].trim();
        let content_out = if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        };

        (content_out, Some(tool_calls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const BEGIN_CALL: &str = "<｜tool▁call▁begin｜>";
    const SEP: &str = "<｜tool▁sep｜>";
    const END_CALL: &str = "<｜tool▁call▁end｜>";
    const END_CALLS: &str = "<｜tool▁calls▁end｜>";

    fn block(name: &str, args: &str) -> String {
        format!(
            "{BEGIN_CALL}function{SEP}{name}\n```json\n{args}\n```\n{END_CALL}"
        )
    }

    #[test]
    fn no_start_token_returns_text() {
        let p = DeepSeekV3ToolCallParser::new();
        let (content, calls) = p.parse("just some text");
        assert_eq!(content, Some("just some text".to_string()));
        assert!(calls.is_none());
    }

    #[test]
    fn single_tool_call() {
        let p = DeepSeekV3ToolCallParser::new();
        let text = format!(
            "{START_TOKEN}\n{}\n{END_CALLS}",
            block("get_weather", "{\"city\": \"Paris\"}")
        );
        let (content, calls) = p.parse(&text);
        assert!(content.is_none());
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["city"], "Paris");
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].id.len(), "call_".len() + 8);
    }

    #[test]
    fn multiple_tool_calls() {
        let p = DeepSeekV3ToolCallParser::new();
        let text = format!(
            "{START_TOKEN}\n{}\n{}\n{END_CALLS}",
            block("a", "{\"x\": 1}"),
            block("b", "{\"y\": 2}")
        );
        let (content, calls) = p.parse(&text);
        assert!(content.is_none());
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
    fn leading_content_is_captured() {
        let p = DeepSeekV3ToolCallParser::new();
        let text = format!(
            "Sure, let me check.  {START_TOKEN}\n{}\n{END_CALLS}",
            block("f", "{}")
        );
        let (content, calls) = p.parse(&text);
        assert_eq!(content, Some("Sure, let me check.".to_string()));
        assert_eq!(calls.unwrap().len(), 1);
    }

    #[test]
    fn start_token_present_but_no_match_returns_text() {
        let p = DeepSeekV3ToolCallParser::new();
        // Has the start token but no valid call block.
        let text = format!("{START_TOKEN} oops malformed {END_CALLS}");
        let (content, calls) = p.parse(&text);
        assert_eq!(content, Some(text.clone()));
        assert!(calls.is_none());
    }

    #[test]
    fn names_and_args_are_trimmed() {
        let p = DeepSeekV3ToolCallParser::new();
        // Whitespace around the name and args should be stripped.
        let text = format!(
            "{START_TOKEN}\n{BEGIN_CALL}function{SEP}  my_func  \n```json\n  {{\"k\": \"v\"}}  \n```\n{END_CALL}\n{END_CALLS}"
        );
        let (_content, calls) = p.parse(&text);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_func");
        let v: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["k"], "v");
    }

    #[test]
    fn generate_call_id_format() {
        let id = generate_call_id();
        assert!(id.starts_with("call_"));
        let hex = &id["call_".len()..];
        assert_eq!(hex.len(), 8);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
