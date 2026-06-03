//! GLM 4.5 (GLM-4-MoE) tool call parser.
//!
//! Format uses custom `arg_key`/`arg_value` tags rather than standard JSON:
//!
//! ```text
//! <tool_call>function_name
//! <arg_key>param1</arg_key><arg_value>value1</arg_value>
//! <arg_key>param2</arg_key><arg_value>value2</arg_value>
//! </tool_call>
//! ```
//!
//! Values are deserialized using a JSON parse, then a Python-literal-style
//! fallback, then a raw-string fallback.
//!
//! Ported from the Python `Glm45ToolCallParser` which is in turn based on
//! VLLM's `Glm4MoeModelToolParser.extract_tool_calls()`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::Value;

/// A single parsed tool call, mirroring the relevant fields of OpenAI's
/// `ChatCompletionMessageToolCall`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Generated id of the form `call_<8 hex chars>`.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The called function.
    pub function: Function,
}

/// The function portion of a [`ToolCall`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments object (with `ensure_ascii=False` semantics,
    /// i.e. non-ASCII characters are emitted verbatim).
    pub arguments: String,
}

/// Result of [`Glm45ToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = (content, tool_calls)`:
/// - `content` is the text preceding the first tool call (or the whole text
///   when nothing parsed), and may be `None`.
/// - `tool_calls` is the list of parsed calls, or `None` when none were found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    /// Leading content text, if any.
    pub content: Option<String>,
    /// Parsed tool calls, if any.
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Try to deserialize a string value to a native JSON value.
///
/// Attempts a JSON parse, then a Python-literal-style parse (`True`/`False`/
/// `None`, single-quoted strings, numbers, lists/tuples/dicts), then falls
/// back to the raw string.
pub fn deserialize_value(value: &str) -> Value {
    if let Ok(v) = serde_json::from_str::<Value>(value) {
        return v;
    }
    if let Some(v) = literal_eval(value) {
        return v;
    }
    Value::String(value.to_string())
}

/// A small approximation of Python's `ast.literal_eval` for the common scalar
/// and container cases. Returns `None` when the input cannot be interpreted as
/// a Python literal, matching the Python code's fallback to the raw string.
fn literal_eval(value: &str) -> Option<Value> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed {
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "None" => return Some(Value::Null),
        _ => {}
    }

    // Integers / floats.
    if let Ok(i) = trimmed.parse::<i64>() {
        return Some(Value::Number(i.into()));
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Some(Value::Number(n));
        }
    }

    // Single-quoted string literal: 'foo' -> "foo".
    if trimmed.len() >= 2 && trimmed.starts_with('\'') && trimmed.ends_with('\'') {
        let inner = &trimmed[1..trimmed.len() - 1];
        // Reject embedded unescaped single quotes (not a simple literal).
        if !inner.contains('\'') {
            return Some(Value::String(unescape_simple(inner)));
        }
    }

    // Python containers: convert literal-ish syntax to JSON and reparse.
    let first = trimmed.chars().next().unwrap();
    let last = trimmed.chars().last().unwrap();
    let looks_container = matches!(
        (first, last),
        ('[', ']') | ('(', ')') | ('{', '}')
    );
    if looks_container {
        if let Some(json_like) = pythonish_to_json(trimmed) {
            if let Ok(v) = serde_json::from_str::<Value>(&json_like) {
                return Some(v);
            }
        }
    }

    None
}

/// Unescape a small set of Python backslash escapes inside a single-quoted
/// string literal.
fn unescape_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('\'') => out.push('\''),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Best-effort conversion of a Python literal container into JSON text:
/// - tuples `(...)` become lists `[...]`
/// - single-quoted strings become double-quoted
/// - `True`/`False`/`None` become `true`/`false`/`null`
///
/// Operates character by character while tracking string state so that
/// punctuation inside strings is preserved.
fn pythonish_to_json(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            '(' => out.push('['),
            ')' => out.push(']'),
            '\'' => {
                // Consume a single-quoted string literal.
                let mut j = i + 1;
                let mut inner = String::new();
                let mut closed = false;
                while j < bytes.len() {
                    let cc = bytes[j];
                    if cc == '\\' && j + 1 < bytes.len() {
                        inner.push(cc);
                        inner.push(bytes[j + 1]);
                        j += 2;
                        continue;
                    }
                    if cc == '\'' {
                        closed = true;
                        j += 1;
                        break;
                    }
                    inner.push(cc);
                    j += 1;
                }
                if !closed {
                    return None;
                }
                // Re-emit as a JSON string: escape double quotes and
                // backslashes that are not already part of an escape.
                out.push('"');
                let mut k = 0;
                let ichars: Vec<char> = inner.chars().collect();
                while k < ichars.len() {
                    let ic = ichars[k];
                    if ic == '\\' && k + 1 < ichars.len() {
                        let nxt = ichars[k + 1];
                        match nxt {
                            '\'' => out.push('\''),
                            '"' => {
                                out.push('\\');
                                out.push('"');
                            }
                            _ => {
                                out.push('\\');
                                out.push(nxt);
                            }
                        }
                        k += 2;
                        continue;
                    }
                    if ic == '"' {
                        out.push('\\');
                        out.push('"');
                    } else if ic == '\\' {
                        out.push('\\');
                        out.push('\\');
                    } else {
                        out.push(ic);
                    }
                    k += 1;
                }
                out.push('"');
                i = j;
                continue;
            }
            _ => {
                // Bare identifiers True/False/None outside of strings.
                if c == 'T' && starts_with_word(&bytes, i, "True") {
                    out.push_str("true");
                    i += 4;
                    continue;
                }
                if c == 'F' && starts_with_word(&bytes, i, "False") {
                    out.push_str("false");
                    i += 5;
                    continue;
                }
                if c == 'N' && starts_with_word(&bytes, i, "None") {
                    out.push_str("null");
                    i += 4;
                    continue;
                }
                out.push(c);
            }
        }
        i += 1;
    }
    Some(out)
}

/// Whether `word` begins at position `i` in `bytes` and is not part of a longer
/// identifier (boundary check on the following character).
fn starts_with_word(bytes: &[char], i: usize, word: &str) -> bool {
    let wchars: Vec<char> = word.chars().collect();
    if i + wchars.len() > bytes.len() {
        return false;
    }
    for (off, wc) in wchars.iter().enumerate() {
        if bytes[i + off] != *wc {
            return false;
        }
    }
    if let Some(after) = bytes.get(i + wchars.len()) {
        if after.is_alphanumeric() || *after == '_' {
            return false;
        }
    }
    true
}

/// Serialize a JSON object map to a string with `ensure_ascii=False`
/// semantics. `serde_json` already emits UTF-8 without escaping non-ASCII,
/// so this is a straightforward serialization, but ordering must follow
/// insertion order to mirror the Python `dict`.
fn dump_args(pairs: &[(String, Value)]) -> String {
    // Build the object body manually to preserve insertion order regardless
    // of the serde_json `preserve_order` feature flag.
    let mut out = String::from("{");
    for (idx, (k, v)) in pairs.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        // Keys are JSON strings.
        out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| format!("{:?}", k)));
        out.push_str(": ");
        out.push_str(&serde_json::to_string(v).unwrap_or_else(|_| "null".to_string()));
    }
    out.push('}');
    out
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate an id of the form `call_<8 hex chars>`, matching the Python
/// `f"call_{uuid.uuid4().hex[:8]}"` shape (8 lowercase hex digits).
fn generate_call_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Mix the time and counter, then take the low 32 bits for 8 hex chars.
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(counter.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let hex = (mixed & 0xFFFF_FFFF) as u32;
    format!("call_{:08x}", hex)
}

/// Parser for GLM 4.5 (GLM-4-MoE) tool calls.
///
/// Uses `<tool_call>...</tool_call>` tags with `<arg_key>`/`<arg_value>`
/// pairs instead of standard JSON arguments.
pub struct Glm45ToolCallParser {
    func_call_regex: Regex,
    func_detail_regex: Regex,
    func_arg_regex: Regex,
}

impl Glm45ToolCallParser {
    /// The token whose presence triggers parsing.
    pub const START_TOKEN: &'static str = "<tool_call>";

    /// Construct a new parser with its compiled regexes.
    pub fn new() -> Self {
        // `(?s)` enables DOTALL so `.` matches newlines, matching Python's
        // `re.DOTALL`.
        Self {
            func_call_regex: Regex::new(r"(?s)<tool_call>.*?</tool_call>")
                .expect("valid func_call regex"),
            func_detail_regex: Regex::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
                .expect("valid func_detail regex"),
            func_arg_regex: Regex::new(
                r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            )
            .expect("valid func_arg regex"),
        }
    }

    /// Parse `text` into leading content plus any tool calls.
    pub fn parse(&self, text: &str) -> ParseResult {
        if !text.contains(Self::START_TOKEN) {
            return ParseResult {
                content: Some(text.to_string()),
                tool_calls: None,
            };
        }

        let matched_calls: Vec<&str> = self
            .func_call_regex
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();

        if matched_calls.is_empty() {
            return ParseResult {
                content: Some(text.to_string()),
                tool_calls: None,
            };
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for matched in matched_calls {
            let detail = match self.func_detail_regex.captures(matched) {
                Some(c) => c,
                None => continue,
            };

            let func_name = detail
                .get(1)
                .map(|m| m.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let func_args_raw = detail.get(2).map(|m| m.as_str()).unwrap_or("");

            let mut pairs: Vec<(String, Value)> = Vec::new();
            if !func_args_raw.is_empty() {
                for cap in self.func_arg_regex.captures_iter(func_args_raw) {
                    let key = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
                    let val_raw = cap.get(2).map(|m| m.as_str()).unwrap_or("").trim();
                    let val = deserialize_value(val_raw);
                    // Last write wins on duplicate keys, mirroring a Python dict.
                    if let Some(existing) = pairs.iter_mut().find(|(k, _)| k == key) {
                        existing.1 = val;
                    } else {
                        pairs.push((key.to_string(), val));
                    }
                }
            }

            tool_calls.push(ToolCall {
                id: generate_call_id(),
                call_type: "function".to_string(),
                function: Function {
                    name: func_name,
                    arguments: dump_args(&pairs),
                },
            });
        }

        if tool_calls.is_empty() {
            return ParseResult {
                content: Some(text.to_string()),
                tool_calls: None,
            };
        }

        let start = text.find(Self::START_TOKEN).unwrap_or(0);
        let content_slice = text[..start].trim();
        let content = if content_slice.is_empty() {
            None
        } else {
            Some(content_slice.to_string())
        };

        ParseResult {
            content,
            tool_calls: Some(tool_calls),
        }
    }
}

impl Default for Glm45ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args_value(tc: &ToolCall) -> Value {
        serde_json::from_str(&tc.function.arguments).expect("arguments must be valid JSON")
    }

    #[test]
    fn no_start_token_returns_text_and_none() {
        let p = Glm45ToolCallParser::new();
        let r = p.parse("just some plain text");
        assert_eq!(r.content.as_deref(), Some("just some plain text"));
        assert!(r.tool_calls.is_none());
    }

    #[test]
    fn single_call_with_string_and_number() {
        let p = Glm45ToolCallParser::new();
        let text = "<tool_call>get_weather\n\
            <arg_key>city</arg_key><arg_value>\"London\"</arg_value>\n\
            <arg_key>days</arg_key><arg_value>3</arg_value>\n\
            </tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.expect("expected calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].id.len(), "call_".len() + 8);
        let v = args_value(&calls[0]);
        assert_eq!(v, json!({"city": "London", "days": 3}));
        // No leading content.
        assert!(r.content.is_none());
    }

    #[test]
    fn leading_content_is_captured_and_trimmed() {
        let p = Glm45ToolCallParser::new();
        let text = "Here you go:  \n<tool_call>do_it\n<arg_key>x</arg_key><arg_value>1</arg_value>\n</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.content.as_deref(), Some("Here you go:"));
        assert!(r.tool_calls.is_some());
    }

    #[test]
    fn multiple_calls() {
        let p = Glm45ToolCallParser::new();
        let text = "<tool_call>a\n<arg_key>k</arg_key><arg_value>1</arg_value>\n</tool_call>\
            <tool_call>b\n<arg_key>k</arg_key><arg_value>2</arg_value>\n</tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.expect("calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(args_value(&calls[0]), json!({"k": 1}));
        assert_eq!(args_value(&calls[1]), json!({"k": 2}));
    }

    #[test]
    fn raw_string_fallback() {
        let p = Glm45ToolCallParser::new();
        let text =
            "<tool_call>f\n<arg_key>name</arg_key><arg_value>plain text here</arg_value>\n</tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.unwrap();
        assert_eq!(args_value(&calls[0]), json!({"name": "plain text here"}));
    }

    #[test]
    fn json_object_value() {
        let p = Glm45ToolCallParser::new();
        let text = "<tool_call>f\n<arg_key>cfg</arg_key><arg_value>{\"a\": 1, \"b\": [2, 3]}</arg_value>\n</tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.unwrap();
        assert_eq!(args_value(&calls[0]), json!({"cfg": {"a": 1, "b": [2, 3]}}));
    }

    #[test]
    fn deserialize_value_variants() {
        assert_eq!(deserialize_value("42"), json!(42));
        assert_eq!(deserialize_value("3.5"), json!(3.5));
        assert_eq!(deserialize_value("true"), json!(true));
        assert_eq!(deserialize_value("\"hi\""), json!("hi"));
        // Python-literal-only forms.
        assert_eq!(deserialize_value("True"), json!(true));
        assert_eq!(deserialize_value("False"), json!(false));
        assert_eq!(deserialize_value("None"), Value::Null);
        assert_eq!(deserialize_value("'single'"), json!("single"));
        // Raw fallback.
        assert_eq!(deserialize_value("not parseable!"), json!("not parseable!"));
    }

    #[test]
    fn python_list_with_single_quotes() {
        // Not valid JSON, but valid python literal.
        let v = deserialize_value("['a', 'b', 'c']");
        assert_eq!(v, json!(["a", "b", "c"]));
    }

    #[test]
    fn python_tuple_becomes_list() {
        let v = deserialize_value("(1, 2, 3)");
        assert_eq!(v, json!([1, 2, 3]));
    }

    #[test]
    fn unclosed_tool_call_yields_no_calls() {
        let p = Glm45ToolCallParser::new();
        // START_TOKEN present but no closing tag -> func_call regex finds nothing.
        let r = p.parse("<tool_call>f\n<arg_key>x</arg_key><arg_value>1</arg_value>");
        assert!(r.tool_calls.is_none());
        assert!(r.content.is_some());
    }

    #[test]
    fn non_ascii_preserved_in_arguments() {
        let p = Glm45ToolCallParser::new();
        let text =
            "<tool_call>f\n<arg_key>msg</arg_key><arg_value>\"héllo 世界\"</arg_value>\n</tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.unwrap();
        // Verbatim non-ASCII (ensure_ascii=False).
        assert!(calls[0].function.arguments.contains("héllo 世界"));
    }

    #[test]
    fn empty_args_produces_empty_object() {
        let p = Glm45ToolCallParser::new();
        let text = "<tool_call>noargs\n</tool_call>";
        let r = p.parse(text);
        let calls = r.tool_calls.unwrap();
        assert_eq!(calls[0].function.name, "noargs");
        assert_eq!(args_value(&calls[0]), json!({}));
    }
}
