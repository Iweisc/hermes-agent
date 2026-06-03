//! GLM 4.7 tool call parser (native Rust port of
//! `environments/tool_call_parsers/glm47_parser.py`, which subclasses
//! `environments/tool_call_parsers/glm45_parser.py`).
//!
//! Format uses custom `arg_key`/`arg_value` tags rather than standard JSON:
//! ```text
//! <tool_call>function_name
//! <arg_key>param1</arg_key><arg_value>value1</arg_value>
//! <arg_key>param2</arg_key><arg_value>value2</arg_value>
//! </tool_call>
//! ```
//!
//! GLM 4.7 differs from GLM 4.5 only in two regexes (see below); the overall
//! `parse()` flow is identical to `Glm45ToolCallParser.parse()`.
//!
//! Argument values are deserialized via JSON, then a Python-`literal_eval`-like
//! fallback, then finally the raw (stripped) string. Each arg value's resolved
//! native value is stored in a JSON object, which is then re-serialized
//! (matching the Python `json.dumps(arg_dict, ensure_ascii=False)`).
//!
//! Based on VLLM's `Glm47MoeModelToolParser` (extends `Glm4MoeModelToolParser`).

use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

/// A parsed tool call, matching the shape of OpenAI's
/// `ChatCompletionMessageToolCall` that the Python original constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Synthetic id, `call_<8 hex chars>` (matching `f"call_{uuid4().hex[:8]}"`).
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name (the text before the first arg tag / closing tag).
    pub name: String,
    /// The JSON-serialized argument object (`ensure_ascii=False` equivalent).
    pub arguments: String,
}

/// Result of [`Glm47ToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = Tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: text preceding the first `<tool_call>` token, or `None`.
/// - `tool_calls`: parsed tool calls, or `None` when none were found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ParseResult {
    /// Helper for the common "no tool calls, return text as-is" case
    /// (the Python `return text, None`).
    fn passthrough(text: &str) -> Self {
        ParseResult {
            content: Some(text.to_string()),
            tool_calls: None,
        }
    }
}

/// Start token; presence is required before any parsing is attempted.
pub const START_TOKEN: &str = "<tool_call>";

/// `FUNC_CALL_REGEX = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)`
fn func_call_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap())
}

/// GLM 4.7 detail regex (overrides GLM 4.5's):
/// `FUNC_DETAIL_REGEX = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)`
///
/// Group 1 is the (non-greedy) function-name region, group 2 is an optional
/// arg block that is captured but unused by `parse()`.
fn func_detail_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<tool_call>(.*?)(<arg_key>.*?)?</tool_call>").unwrap())
}

/// GLM 4.7 arg regex (overrides GLM 4.5's), handling newlines between tags:
/// `FUNC_ARG_REGEX = re.compile(
///      r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>", re.DOTALL)`
///
/// Note: the Python pattern's `\\n` is a literal backslash-n (two chars), not a
/// newline — `(?:\\n|\s)*` matches either the two-character sequence `\n` or any
/// single whitespace char, repeated. Real newlines are already covered by `\s`.
fn func_arg_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>").unwrap()
    })
}

/// Parser for GLM 4.7 tool calls.
#[derive(Debug, Default, Clone, Copy)]
pub struct Glm47ToolCallParser;

impl Glm47ToolCallParser {
    pub fn new() -> Self {
        Glm47ToolCallParser
    }

    /// Parse raw model output text for tool calls.
    ///
    /// Faithful port of `Glm45ToolCallParser.parse()` using GLM 4.7's regexes.
    /// Any internal error path falls back to `(text, None)` (the Python
    /// `except Exception` guard).
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(START_TOKEN) {
            return ParseResult::passthrough(text);
        }

        // `self.FUNC_CALL_REGEX.findall(text)` — full-match substrings.
        let matched_calls: Vec<&str> =
            func_call_regex().find_iter(text).map(|m| m.as_str()).collect();
        if matched_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for call in &matched_calls {
            // `detail = self.FUNC_DETAIL_REGEX.search(match)`
            let detail = match func_detail_regex().captures(call) {
                Some(c) => c,
                None => continue, // `if not detail: continue`
            };

            // `func_name = detail.group(1).strip()`
            let func_name = detail
                .get(1)
                .map(|m| m.as_str())
                .unwrap_or("")
                .trim()
                .to_string();

            // `func_args_raw = detail.group(2)` — the optional arg block.
            // In GLM 4.7 the detail regex's group 2 only captures up to the
            // first `</arg_key>` boundary region (it is non-greedy and
            // optional), so it does NOT reliably contain the full arg list.
            // We mirror the Python exactly: the arg pairs are extracted from
            // whatever `group(2)` captured. When unset, no pairs are produced.
            let func_args_raw = detail.get(2).map(|m| m.as_str());

            // `pairs = self.FUNC_ARG_REGEX.findall(func_args_raw) if func_args_raw else []`
            let mut arg_map: Map<String, Value> = Map::new();
            if let Some(raw) = func_args_raw {
                for caps in func_arg_regex().captures_iter(raw) {
                    let key = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
                    let raw_val = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim();
                    arg_map.insert(key.to_string(), deserialize_value(raw_val));
                }
            }

            let arguments = serde_json::to_string(&Value::Object(arg_map))
                .unwrap_or_else(|_| "{}".to_string());

            tool_calls.push(ToolCall {
                id: format!("call_{}", short_hex_id()),
                call_type: "function".to_string(),
                name: func_name,
                arguments,
            });
        }

        if tool_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        // `content = text[: text.find(self.START_TOKEN)].strip()`
        let start_idx = text.find(START_TOKEN).unwrap_or(0);
        let content = text[..start_idx].trim();
        let content = if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        };

        ParseResult {
            content,
            tool_calls: Some(tool_calls),
        }
    }
}

/// Try to deserialize a string value to a native JSON value, mirroring the
/// Python `_deserialize_value`: `json.loads` -> `ast.literal_eval` -> raw string.
///
/// `serde_json::from_str` covers the `json.loads` step. The
/// `ast.literal_eval` fallback handles Python literals that are not valid
/// JSON — most notably single-quoted strings, `True`/`False`/`None`, and tuples.
/// We support the common cases; anything unrecognized falls back to the raw
/// string (matching Python's final fallback).
pub fn deserialize_value(value: &str) -> Value {
    // Step 1: json.loads
    if let Ok(v) = serde_json::from_str::<Value>(value) {
        return v;
    }
    // Step 2: ast.literal_eval (best-effort).
    if let Some(v) = literal_eval(value) {
        return v;
    }
    // Step 3: raw string.
    Value::String(value.to_string())
}

/// Best-effort emulation of Python's `ast.literal_eval` for the value forms a
/// model is likely to emit that are not already valid JSON.
fn literal_eval(s: &str) -> Option<Value> {
    let t = s.trim();
    match t {
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "None" => return Some(Value::Null),
        _ => {}
    }

    // Single-quoted string literal: 'foo'  (no embedded escaping handling
    // beyond the simple case — Python literal_eval would unescape, but the
    // common case for tool args is a plain single-quoted string).
    if t.len() >= 2 && t.starts_with('\'') && t.ends_with('\'') {
        let inner = &t[1..t.len() - 1];
        // Reject if the inner contains an unescaped single quote (ambiguous);
        // keep it simple and only accept clean single-quoted strings.
        if !inner.contains('\'') {
            return Some(Value::String(inner.replace("\\'", "'")));
        }
    }

    // Convert a Python-ish literal into JSON by swapping single quotes and
    // Python keywords, then re-attempting a JSON parse. This handles dicts /
    // lists / tuples like `{'a': 1}` or `[1, 'two', True]`.
    if (t.starts_with('{') && t.ends_with('}'))
        || (t.starts_with('[') && t.ends_with(']'))
        || (t.starts_with('(') && t.ends_with(')'))
    {
        if let Some(json_like) = pythonish_to_json(t) {
            if let Ok(v) = serde_json::from_str::<Value>(&json_like) {
                return Some(v);
            }
        }
    }

    None
}

/// Translate a small subset of Python container/scalar literal syntax into JSON
/// text. Tuples `(...)` become arrays. Single-quoted strings become
/// double-quoted. `True`/`False`/`None` become `true`/`false`/`null`.
///
/// This is intentionally conservative: it only performs the transformation
/// outside of double-quoted string regions to avoid mangling content, and
/// returns `None` if it encounters something it cannot confidently convert.
fn pythonish_to_json(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_double {
            out.push(c);
            if c == '\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == '"' {
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_single {
            if c == '\\' && i + 1 < bytes.len() {
                let nxt = bytes[i + 1] as char;
                // Preserve escapes; turn \' into a plain ' inside the now
                // double-quoted JSON string.
                if nxt == '\'' {
                    out.push('\'');
                } else {
                    out.push('\\');
                    out.push(nxt);
                }
                i += 2;
                continue;
            }
            if c == '\'' {
                out.push('"');
                in_single = false;
                i += 1;
                continue;
            }
            // Escape a double quote that appears inside a single-quoted string.
            if c == '"' {
                out.push('\\');
                out.push('"');
                i += 1;
                continue;
            }
            out.push(c);
            i += 1;
            continue;
        }

        match c {
            '"' => {
                in_double = true;
                out.push(c);
                i += 1;
            }
            '\'' => {
                in_single = true;
                out.push('"');
                i += 1;
            }
            '(' => {
                out.push('[');
                i += 1;
            }
            ')' => {
                out.push(']');
                i += 1;
            }
            _ => {
                // Replace Python keywords on word boundaries.
                if let Some((repl, len)) = match_keyword(&s[i..]) {
                    out.push_str(repl);
                    i += len;
                } else {
                    out.push(c);
                    i += 1;
                }
            }
        }
    }

    if in_single || in_double {
        return None;
    }
    Some(out)
}

/// If `s` begins with a Python keyword (`True`/`False`/`None`) on a word
/// boundary, return its JSON replacement and the consumed length.
fn match_keyword(s: &str) -> Option<(&'static str, usize)> {
    for (kw, repl) in [("True", "true"), ("False", "false"), ("None", "null")] {
        if let Some(rest) = s.strip_prefix(kw) {
            let boundary_before = true; // caller only invokes at a non-alnum split
            let boundary_after = rest
                .chars()
                .next()
                .map(|c| !(c.is_alphanumeric() || c == '_'))
                .unwrap_or(true);
            if boundary_before && boundary_after {
                return Some((repl, kw.len()));
            }
        }
    }
    None
}

/// Produce 8 lowercase hex characters, emulating `uuid.uuid4().hex[:8]`.
///
/// We do not depend on an external RNG crate; a small splitmix64-style PRNG
/// seeded from system entropy sources is sufficient since the id only needs to
/// be unique-ish per call (the Python value is also random).
fn short_hex_id() -> String {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new(seed());
    }

    fn seed() -> u64 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let addr = &nanos as *const u64 as u64;
        nanos ^ addr.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15
    }

    STATE.with(|st| {
        let mut x = st.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        x = z;
        st.set(x);
        format!("{:08x}", (x & 0xFFFF_FFFF) as u32)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_json(tc: &ToolCall) -> Value {
        serde_json::from_str(&tc.arguments).unwrap()
    }

    #[test]
    fn no_start_token_passthrough() {
        let text = "just some text without tool calls";
        let res = Glm47ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some(text));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn id_format_is_call_plus_8_hex() {
        let text = "<tool_call>do_thing\n</tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        let id = &tcs[0].id;
        assert!(id.starts_with("call_"));
        let hex = &id["call_".len()..];
        assert_eq!(hex.len(), 8);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn function_name_only_no_args() {
        let text = "<tool_call>get_time\n</tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "get_time");
        assert_eq!(tcs[0].call_type, "function");
        // No arg block captured -> empty object.
        assert_eq!(args_json(&tcs[0]), serde_json::json!({}));
    }

    #[test]
    fn arg_pairs_with_json_values() {
        // The arg block is captured by detail group 2 (starts at <arg_key>).
        let text = "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>\"Paris\"</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "get_weather");
        let args = args_json(&tcs[0]);
        assert_eq!(args["city"], serde_json::json!("Paris"));
        assert_eq!(args["days"], serde_json::json!(3));
    }

    #[test]
    fn arg_values_deserialized_to_native_types() {
        let text = "<tool_call>f\n<arg_key>num</arg_key><arg_value>42</arg_value><arg_key>flag</arg_key><arg_value>true</arg_value><arg_key>obj</arg_key><arg_value>{\"a\": 1}</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_json(&tcs[0]);
        assert_eq!(args["num"], serde_json::json!(42));
        assert_eq!(args["flag"], serde_json::json!(true));
        assert_eq!(args["obj"], serde_json::json!({"a": 1}));
    }

    #[test]
    fn raw_string_fallback() {
        let text = "<tool_call>f\n<arg_key>q</arg_key><arg_value>hello world</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_json(&tcs[0]);
        assert_eq!(args["q"], serde_json::json!("hello world"));
    }

    #[test]
    fn newline_between_arg_tags_handled() {
        // GLM 4.7 allows whitespace/newlines between </arg_key> and <arg_value>.
        let text = "<tool_call>f\n<arg_key>k</arg_key>\n  <arg_value>1</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_json(&tcs[0]);
        assert_eq!(args["k"], serde_json::json!(1));
    }

    #[test]
    fn literal_backslash_n_between_arg_tags() {
        // The literal two-char sequence \n is also accepted between tags.
        let text = "<tool_call>f\n<arg_key>k</arg_key>\\n<arg_value>1</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_json(&tcs[0]);
        assert_eq!(args["k"], serde_json::json!(1));
    }

    #[test]
    fn content_before_tool_call_is_extracted() {
        let text = "Sure, let me help.\n<tool_call>f\n</tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("Sure, let me help."));
        assert_eq!(res.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn empty_content_becomes_none() {
        let text = "<tool_call>f\n</tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        assert!(res.content.is_none());
    }

    #[test]
    fn multiple_tool_calls() {
        let text = "<tool_call>a\n<arg_key>x</arg_key><arg_value>1</arg_value></tool_call><tool_call>b\n<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "a");
        assert_eq!(tcs[1].name, "b");
        assert_eq!(args_json(&tcs[0])["x"], serde_json::json!(1));
        assert_eq!(args_json(&tcs[1])["y"], serde_json::json!(2));
    }

    #[test]
    fn dotall_multiline_arg_value() {
        let text = "<tool_call>f\n<arg_key>data</arg_key><arg_value>{\n  \"a\": 1,\n  \"b\": 2\n}</arg_value></tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        let args = args_json(&tcs[0]);
        assert_eq!(args["data"], serde_json::json!({"a": 1, "b": 2}));
    }

    #[test]
    fn start_token_but_no_complete_call_passthrough() {
        let text = "<tool_call>unterminated function with no closing tag";
        let res = Glm47ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some(text));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn deserialize_value_python_literals() {
        assert_eq!(deserialize_value("True"), Value::Bool(true));
        assert_eq!(deserialize_value("False"), Value::Bool(false));
        assert_eq!(deserialize_value("None"), Value::Null);
        assert_eq!(deserialize_value("'hi'"), serde_json::json!("hi"));
        assert_eq!(deserialize_value("[1, 'two', True]"), serde_json::json!([1, "two", true]));
        assert_eq!(deserialize_value("{'a': 1, 'b': None}"), serde_json::json!({"a": 1, "b": null}));
        // Tuple -> array.
        assert_eq!(deserialize_value("(1, 2)"), serde_json::json!([1, 2]));
    }

    #[test]
    fn deserialize_value_json_precedence() {
        // Valid JSON is handled by step 1.
        assert_eq!(deserialize_value("123"), serde_json::json!(123));
        assert_eq!(deserialize_value("\"str\""), serde_json::json!("str"));
        assert_eq!(deserialize_value("[1,2,3]"), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn function_name_is_stripped() {
        let text = "<tool_call>  spaced_name  \n</tool_call>";
        let res = Glm47ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "spaced_name");
    }
}
