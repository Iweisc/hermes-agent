//! Qwen3-Coder tool call parser (native Rust port of
//! `environments/tool_call_parsers/qwen3_coder_parser.py`).
//!
//! Format uses XML-style nested tags:
//! ```text
//! <tool_call>
//! <function=function_name>
//! <parameter=param_name>value</parameter>
//! <parameter=param_name2>value2</parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameters are extracted from `<parameter=name>value</parameter>` tags and
//! type-converted (null / JSON / Python-literal fallback / raw string),
//! mirroring the upstream VLLM `Qwen3CoderToolParser.extract_tool_calls()`.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

/// A parsed tool call, matching the shape of OpenAI's
/// `ChatCompletionMessageToolCall` that the Python original constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Unique id, formatted as `call_<24 hex chars>`.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name.
    pub name: String,
    /// JSON-encoded argument object (with `ensure_ascii=False` semantics:
    /// non-ASCII characters are NOT escaped).
    pub arguments: String,
}

/// Result of [`Qwen3CoderToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = Tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: text with tool-call markup stripped, or `None` if the whole
///   output was tool calls.
/// - `tool_calls`: parsed tool calls, or `None` when none were found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ParseResult {
    /// Helper for the common "no tool calls, return text as-is" case.
    fn passthrough(text: &str) -> Self {
        ParseResult {
            content: Some(text.to_string()),
            tool_calls: None,
        }
    }
}

const START_TOKEN: &str = "<tool_call>";
const FUNCTION_PREFIX: &str = "<function=";

fn tool_call_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `(?s)` = DOTALL. Two alternatives: a closed block, or an unclosed block
    // running to end of input.
    RE.get_or_init(|| Regex::new(r"(?s)<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$").unwrap())
}

fn function_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<function=(.*?)</function>|<function=(.*)$").unwrap())
}

/// Try to convert a parameter value string to a native value, returning a
/// `serde_json::Value`.
///
/// Order mirrors the Python `_try_convert_value`:
/// 1. `null` (case-insensitive) -> JSON null
/// 2. `serde_json` parse (objects, arrays, strings, numbers, booleans)
/// 3. Python `ast.literal_eval` subset (Python bools/None, single-quoted
///    strings, tuples-as-arrays)
/// 4. fallback: the stripped string
pub fn try_convert_value(value: &str) -> Value {
    let stripped = value.trim();

    if stripped.eq_ignore_ascii_case("null") {
        return Value::Null;
    }

    if let Ok(v) = serde_json::from_str::<Value>(stripped) {
        return v;
    }

    if let Some(v) = py_literal_eval(stripped) {
        return v;
    }

    Value::String(stripped.to_string())
}

/// A best-effort port of `ast.literal_eval` for the cases that JSON cannot
/// already handle. Supports Python literals `True`/`False`/`None`,
/// single-quoted strings, and parenthesised tuples (mapped to JSON arrays).
///
/// Returns `None` when the input is not a recognised literal, so the caller
/// falls back to the raw string (matching Python, which catches
/// `ValueError`/`SyntaxError`/`TypeError`).
fn py_literal_eval(s: &str) -> Option<Value> {
    let t = s.trim();
    match t {
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "None" => return Some(Value::Null),
        _ => {}
    }

    // Single-quoted string -> normalise to a JSON double-quoted string.
    if t.len() >= 2 && t.starts_with('\'') && t.ends_with('\'') {
        let inner = &t[1..t.len() - 1];
        // Reject embedded unescaped single quotes (not a simple literal).
        if !inner.contains('\'') {
            return Some(Value::String(inner.replace("\\'", "'")));
        }
    }

    // Tuple literal `(a, b, ...)` -> JSON array. Only handle the simple case
    // where the contents are themselves JSON-parseable after rebracketing.
    if t.starts_with('(') && t.ends_with(')') {
        let inner = &t[1..t.len() - 1];
        let candidate = format!("[{}]", inner);
        if let Ok(v) = serde_json::from_str::<Value>(&candidate) {
            return Some(v);
        }
    }

    None
}

/// Generate an id of the form `call_<24 lowercase hex chars>`, matching
/// `f"call_{uuid.uuid4().hex[:24]}"`.
fn generate_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Deterministic-enough pseudo-random hex without pulling in a uuid crate.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let addr = &nanos as *const u128 as u128;
    let mut state = nanos ^ addr.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = String::with_capacity(29);
    out.push_str("call_");
    for _ in 0..24 {
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

/// Parser for Qwen3-Coder XML-format tool calls.
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen3CoderToolCallParser;

impl Qwen3CoderToolCallParser {
    pub fn new() -> Self {
        Qwen3CoderToolCallParser
    }

    /// Extract `<parameter=name>value</parameter>` segments from a parameter
    /// string. Mirrors the Python PARAMETER_REGEX which terminates each match
    /// at `</parameter>`, the next `<parameter=`, `</function>`, or end-of-input
    /// (whichever comes first), and skips segments that contain no `>`.
    fn extract_param_segments(params_str: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let bytes = params_str;
        let mut search_from = 0usize;

        while let Some(rel) = bytes[search_from..].find("<parameter=") {
            let start = search_from + rel + "<parameter=".len();
            // Determine the end: earliest of </parameter>, next <parameter=,
            // </function>, or EOF.
            let rest = &bytes[start..];
            let mut end_rel = rest.len();

            for delim in ["</parameter>", "<parameter=", "</function>"] {
                if let Some(idx) = rest.find(delim) {
                    if idx < end_rel {
                        end_rel = idx;
                    }
                }
            }

            let segment = &rest[..end_rel];
            segments.push(segment.to_string());

            // Advance: if we stopped at </parameter>, skip past it; otherwise
            // continue from the delimiter position (so the next <parameter= is
            // re-found on the next iteration).
            let abs_end = start + end_rel;
            if rest[end_rel..].starts_with("</parameter>") {
                search_from = abs_end + "</parameter>".len();
            } else {
                search_from = abs_end;
            }
            if search_from >= bytes.len() {
                break;
            }
        }

        segments
    }

    /// Parse a single `<function=name>...</function>` block (with the leading
    /// `<function=` already stripped) into a [`ToolCall`].
    fn parse_function_call(function_str: &str) -> Option<ToolCall> {
        let gt_idx = function_str.find('>')?;
        let func_name = function_str[..gt_idx].trim().to_string();
        let params_str = &function_str[gt_idx + 1..];

        let mut param_map = serde_json::Map::new();
        for segment in Self::extract_param_segments(params_str) {
            let eq_idx = match segment.find('>') {
                Some(i) => i,
                None => continue,
            };
            let param_name = segment[..eq_idx].trim().to_string();
            let mut param_value = &segment[eq_idx + 1..];

            // Clean up a single leading / trailing newline.
            if let Some(stripped) = param_value.strip_prefix('\n') {
                param_value = stripped;
            }
            if let Some(stripped) = param_value.strip_suffix('\n') {
                param_value = stripped;
            }

            param_map.insert(param_name, try_convert_value(param_value));
        }

        let arguments = serde_json::to_string(&Value::Object(param_map))
            .unwrap_or_else(|_| "{}".to_string());

        Some(ToolCall {
            id: generate_call_id(),
            call_type: "function".to_string(),
            name: func_name,
            arguments,
        })
    }

    /// Parse raw model output text for tool calls.
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(FUNCTION_PREFIX) {
            return ParseResult::passthrough(text);
        }

        // Find all tool_call blocks.
        let mut raw_blocks: Vec<String> = Vec::new();
        for caps in tool_call_regex().captures_iter(text) {
            let block = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            raw_blocks.push(block);
        }

        // Fallback: if no tool_call tags, try the whole text.
        if raw_blocks.is_empty() {
            raw_blocks.push(text.to_string());
        }

        // Find function blocks within each tool_call.
        let mut function_strs: Vec<String> = Vec::new();
        for block in &raw_blocks {
            for caps in function_regex().captures_iter(block) {
                let f = caps
                    .get(1)
                    .or_else(|| caps.get(2))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                function_strs.push(f);
            }
        }

        if function_strs.is_empty() {
            return ParseResult::passthrough(text);
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for func_str in &function_strs {
            if let Some(tc) = Self::parse_function_call(func_str) {
                tool_calls.push(tc);
            }
        }

        if tool_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        // Content before tool calls.
        let mut first_tc = text.find(START_TOKEN);
        if first_tc.is_none() {
            first_tc = text.find(FUNCTION_PREFIX);
        }
        let content = match first_tc {
            // Note: Python uses `if first_tc > 0`, so a tool-call block at
            // index 0 yields no content.
            Some(idx) if idx > 0 => {
                let trimmed = text[..idx].trim();
                Some(trimmed.to_string())
            }
            _ => None,
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
    fn no_function_prefix_passthrough() {
        let text = "just some plain text";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("just some plain text"));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn single_tool_call_basic() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("expected tool calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(tcs[0].call_type, "function");
        assert!(tcs[0].id.starts_with("call_"));
        assert_eq!(tcs[0].id.len(), "call_".len() + 24);
        let args = args_value(&tcs[0]);
        assert_eq!(args["city"], Value::String("Paris".to_string()));
        // Block starts at index 0 -> no content.
        assert!(res.content.is_none());
    }

    #[test]
    fn content_before_tool_call() {
        let text = "Sure, let me check.\n<tool_call>\n<function=ping>\n<parameter=host>localhost</parameter>\n</function>\n</tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("Sure, let me check."));
        assert_eq!(res.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn type_conversion_number_bool_json() {
        let text = "<tool_call><function=f>\
<parameter=n>42</parameter>\
<parameter=flag>true</parameter>\
<parameter=obj>{\"a\": 1}</parameter>\
<parameter=arr>[1, 2, 3]</parameter>\
<parameter=nothing>null</parameter>\
</function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_value(&tcs[0]);
        assert_eq!(args["n"], Value::from(42));
        assert_eq!(args["flag"], Value::Bool(true));
        assert_eq!(args["obj"], serde_json::json!({"a": 1}));
        assert_eq!(args["arr"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["nothing"], Value::Null);
    }

    #[test]
    fn string_fallback() {
        let text =
            "<tool_call><function=f><parameter=path>/etc/hosts and stuff</parameter></function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_value(&tcs[0]);
        assert_eq!(
            args["path"],
            Value::String("/etc/hosts and stuff".to_string())
        );
    }

    #[test]
    fn null_case_insensitive() {
        assert_eq!(try_convert_value("NULL"), Value::Null);
        assert_eq!(try_convert_value("  null  "), Value::Null);
        assert_eq!(try_convert_value("None"), Value::Null);
        assert_eq!(try_convert_value("True"), Value::Bool(true));
        assert_eq!(try_convert_value("False"), Value::Bool(false));
    }

    #[test]
    fn py_single_quote_string() {
        assert_eq!(
            try_convert_value("'hello'"),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn multiple_tool_calls() {
        let text = "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>\
<tool_call><function=b><parameter=y>2</parameter></function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "a");
        assert_eq!(tcs[1].name, "b");
    }

    #[test]
    fn multiple_params_one_function() {
        let text = "<tool_call><function=move>\
<parameter=src>old.txt</parameter>\
<parameter=dst>new.txt</parameter>\
</function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let args = args_value(&res.tool_calls.unwrap()[0]);
        assert_eq!(args["src"], Value::String("old.txt".to_string()));
        assert_eq!(args["dst"], Value::String("new.txt".to_string()));
    }

    #[test]
    fn unclosed_tool_call_at_end() {
        let text = "<tool_call><function=f><parameter=k>v</parameter></function>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("should still parse unclosed block");
        assert_eq!(tcs[0].name, "f");
        let args = args_value(&tcs[0]);
        assert_eq!(args["k"], Value::String("v".to_string()));
    }

    #[test]
    fn function_prefix_but_no_valid_function() {
        // Contains FUNCTION_PREFIX trigger but no '>' to delimit a name.
        let text = "<function=";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        // function_str is empty -> parse_function_call returns None -> passthrough.
        assert_eq!(res.content.as_deref(), Some("<function="));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn newline_trimming_in_value() {
        let text = "<tool_call><function=f><parameter=body>\nline\n</parameter></function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let args = args_value(&res.tool_calls.unwrap()[0]);
        // One leading and one trailing newline stripped.
        assert_eq!(args["body"], Value::String("line".to_string()));
    }

    #[test]
    fn non_ascii_not_escaped() {
        let text = "<tool_call><function=f><parameter=msg>héllo</parameter></function></tool_call>";
        let res = Qwen3CoderToolCallParser::new().parse(text);
        let tc = &res.tool_calls.unwrap()[0];
        // ensure_ascii=False semantics: literal non-ascii char in the JSON.
        assert!(tc.arguments.contains('é'));
    }
}
