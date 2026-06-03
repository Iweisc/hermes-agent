//! DeepSeek V3.1 tool call parser (native Rust port of
//! `environments/tool_call_parsers/deepseek_v3_1_parser.py`).
//!
//! Format (note: the delimiters use full-width vertical bars `｜` and the
//! lower-one-eighth-block `▁` character, exactly as DeepSeek emits them):
//!
//! ```text
//! <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>
//! ```
//!
//! Unlike V3, the function name comes before the separator and the arguments
//! come after (no `type` field, no JSON code-block wrapper). Based on VLLM's
//! `DeepSeekV31ToolParser.extract_tool_calls()`.

use std::sync::OnceLock;

use regex::Regex;

/// A parsed tool call, matching the shape of OpenAI's
/// `ChatCompletionMessageToolCall` that the Python original constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Unique id, formatted as `call_<8 hex chars>`.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name (stripped of surrounding whitespace).
    pub name: String,
    /// The raw argument string (stripped of surrounding whitespace). The
    /// Python original does not parse/validate this as JSON.
    pub arguments: String,
}

/// Result of [`DeepSeekV31ToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = Tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: text before the first tool call (stripped), or `None` when it
///   is empty / the whole output was tool calls.
/// - `tool_calls`: parsed tool calls, or `None` when none were found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ParseResult {
    /// Helper for the common "no tool calls, return text as-is" case
    /// (`return text, None` in Python).
    fn passthrough(text: &str) -> Self {
        ParseResult {
            content: Some(text.to_string()),
            tool_calls: None,
        }
    }
}

/// Token whose presence triggers parsing. Matches the Python `START_TOKEN`.
pub const START_TOKEN: &str = "<｜tool▁calls▁begin｜>";

fn pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `(?s)` = DOTALL so that `.` matches newlines, matching Python's re.DOTALL.
    RE.get_or_init(|| {
        Regex::new(
            r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
        )
        .unwrap()
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
        // xorshift-style mixing
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

/// Parser for DeepSeek V3.1 tool calls.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeekV31ToolCallParser;

impl DeepSeekV31ToolCallParser {
    pub fn new() -> Self {
        DeepSeekV31ToolCallParser
    }

    /// Parse raw model output text for tool calls.
    ///
    /// Faithful port of the Python `parse`:
    /// - If the start token is absent, return the text unchanged.
    /// - Find every `begin..sep..end` triple; build one tool call per match.
    /// - Content is the (stripped) text preceding the first start token, or
    ///   `None` when empty.
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(START_TOKEN) {
            return ParseResult::passthrough(text);
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for caps in pattern().captures_iter(text) {
            let func_name = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let func_args = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            tool_calls.push(ToolCall {
                id: generate_call_id(),
                call_type: "function".to_string(),
                name: func_name.trim().to_string(),
                arguments: func_args.trim().to_string(),
            });
        }

        if tool_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        // Content before the first start token.
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

    const BEGIN: &str = "<｜tool▁call▁begin｜>";
    const SEP: &str = "<｜tool▁sep｜>";
    const END: &str = "<｜tool▁call▁end｜>";

    fn wrap(body: &str) -> String {
        format!("{}{}", START_TOKEN, body)
    }

    #[test]
    fn no_start_token_passthrough() {
        let text = "just some plain text";
        let res = DeepSeekV31ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("just some plain text"));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn single_tool_call() {
        let text = wrap(&format!(
            "{BEGIN}get_weather{SEP}{{\"city\": \"Paris\"}}{END}"
        ));
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        let tcs = res.tool_calls.expect("expected tool calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(tcs[0].call_type, "function");
        assert_eq!(tcs[0].arguments, "{\"city\": \"Paris\"}");
        assert!(tcs[0].id.starts_with("call_"));
        assert_eq!(tcs[0].id.len(), "call_".len() + 8);
        // Start token at index 0 -> no leading content.
        assert!(res.content.is_none());
    }

    #[test]
    fn content_before_tool_call() {
        let text = format!(
            "Let me check.\n{}{BEGIN}ping{SEP}{{}}{END}",
            START_TOKEN
        );
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        assert_eq!(res.content.as_deref(), Some("Let me check."));
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "ping");
        assert_eq!(tcs[0].arguments, "{}");
    }

    #[test]
    fn multiple_tool_calls() {
        let text = wrap(&format!(
            "{BEGIN}a{SEP}{{\"x\": 1}}{END}{BEGIN}b{SEP}{{\"y\": 2}}{END}"
        ));
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "a");
        assert_eq!(tcs[0].arguments, "{\"x\": 1}");
        assert_eq!(tcs[1].name, "b");
        assert_eq!(tcs[1].arguments, "{\"y\": 2}");
    }

    #[test]
    fn name_and_args_are_stripped() {
        let text = wrap(&format!("{BEGIN}  fn_name  {SEP}  {{\"a\": 1}}  {END}"));
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "fn_name");
        assert_eq!(tcs[0].arguments, "{\"a\": 1}");
    }

    #[test]
    fn dotall_args_with_newlines() {
        let text = wrap(&format!(
            "{BEGIN}f{SEP}{{\n  \"k\": \"v\"\n}}{END}"
        ));
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "f");
        assert_eq!(tcs[0].arguments, "{\n  \"k\": \"v\"\n}");
    }

    #[test]
    fn start_token_present_but_no_match() {
        // Start token present, but no begin/sep/end triple -> passthrough.
        let text = format!("{}garbage with no triple", START_TOKEN);
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        assert_eq!(res.content.as_deref(), Some(text.as_str()));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn arguments_not_validated_as_json() {
        // Arguments are kept raw; not required to be valid JSON.
        let text = wrap(&format!("{BEGIN}f{SEP}not json at all{END}"));
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].arguments, "not json at all");
    }

    #[test]
    fn content_only_whitespace_becomes_none() {
        let text = format!("   \n  {}{BEGIN}f{SEP}{{}}{END}", START_TOKEN);
        let res = DeepSeekV31ToolCallParser::new().parse(&text);
        assert!(res.content.is_none());
        assert_eq!(res.tool_calls.unwrap().len(), 1);
    }
}
