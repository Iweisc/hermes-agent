//! Kimi K2 tool call parser (native Rust port of
//! `environments/tool_call_parsers/kimi_k2_parser.py`).
//!
//! Format:
//! ```text
//! <|tool_calls_section_begin|>
//! <|tool_call_begin|>function_id:0<|tool_call_argument_begin|>{"arg": "val"}<|tool_call_end|>
//! <|tool_calls_section_end|>
//! ```
//!
//! The `function_id` is typically `functions.func_name:index` or
//! `func_name:index`. The function name is everything after the last `.` and
//! before the trailing `:index`.
//!
//! Based on VLLM's `KimiK2ToolParser.extract_tool_calls()`.

/// A parsed tool call, matching the shape of OpenAI's
/// `ChatCompletionMessageToolCall` that the Python original constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// The original `function_id` (e.g. `functions.get_weather:0`), preserved verbatim.
    pub id: String,
    /// Always `"function"`.
    pub call_type: String,
    /// The function name (after the last `.`, before the trailing `:index`).
    pub name: String,
    /// The raw (stripped) argument string, typically a JSON object.
    pub arguments: String,
}

/// Result of [`KimiK2ToolCallParser::parse`].
///
/// Mirrors the Python `ParseResult = Tuple[Optional[str], Optional[List[...]]]`:
/// - `content`: text preceding the tool-calls section, or `None`.
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

/// Section start tokens (both singular and plural variants are accepted),
/// matching the Python `START_TOKENS`.
pub const START_TOKENS: [&str; 2] = [
    "<|tool_calls_section_begin|>",
    "<|tool_call_section_begin|>",
];

const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";

/// Parser for Kimi K2 tool calls.
#[derive(Debug, Default, Clone, Copy)]
pub struct KimiK2ToolCallParser;

impl KimiK2ToolCallParser {
    pub fn new() -> Self {
        KimiK2ToolCallParser
    }

    /// Parse raw model output text for tool calls.
    pub fn parse(&self, text: &str) -> ParseResult {
        // Check for any variant of the start token.
        let has_start = START_TOKENS.iter().any(|t| text.contains(t));
        if !has_start {
            return ParseResult::passthrough(text);
        }

        let matches = extract_matches(text);
        if matches.is_empty() {
            return ParseResult::passthrough(text);
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for (function_id, function_args) in &matches {
            // Extract function name from ID format:
            // "functions.get_weather:0" -> "get_weather".
            // Mirrors `function_id.split(":")[0].split(".")[-1]`.
            let before_colon = function_id.split(':').next().unwrap_or(function_id);
            let function_name = before_colon.rsplit('.').next().unwrap_or(before_colon);

            tool_calls.push(ToolCall {
                id: function_id.clone(),
                call_type: "function".to_string(),
                name: function_name.to_string(),
                arguments: function_args.trim().to_string(),
            });
        }

        if tool_calls.is_empty() {
            return ParseResult::passthrough(text);
        }

        // Content is everything before the earliest tool-calls section token.
        let mut earliest_start = text.len();
        for token in START_TOKENS.iter() {
            if let Some(idx) = text.find(token) {
                if idx < earliest_start {
                    earliest_start = idx;
                }
            }
        }

        let content = text[..earliest_start].trim();
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

/// Replicate the Python regex:
/// ```text
/// <\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*
/// <\|tool_call_argument_begin\|>\s*
/// (?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*
/// <\|tool_call_end\|>
/// ```
/// with `re.DOTALL`.
///
/// The `regex` crate has no negative lookahead, so the argument-body capture
/// (which must not span another `<|tool_call_begin|>`) is scanned manually.
/// Returns `(tool_call_id, function_arguments)` pairs in order of appearance.
fn extract_matches(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find(TOOL_CALL_BEGIN) {
        let begin_at = search_from + rel;
        let after_begin = begin_at + TOOL_CALL_BEGIN.len();

        // `\s*` after the begin token.
        let id_start = after_begin + leading_ws_len(&text[after_begin..]);

        // `(?P<tool_call_id>[^<]+:\d+)`: one-or-more non-`<` chars ending in
        // `:<digits>`. We need at least one char, the id must contain no `<`,
        // and must end with `:` followed by digits. The regex is greedy, so it
        // grabs the longest `[^<]+` run, then backtracks to satisfy `:\d+`
        // followed by the rest of the pattern. We emulate by taking the run of
        // non-`<` chars and finding the last `:\d+` boundary within it that
        // leaves a valid continuation.
        let id_region = &text[id_start..];
        let id_run_len = id_region.find('<').unwrap_or(id_region.len());
        let id_run = &id_region[..id_run_len];

        // Find the rightmost position where `id_run[..k]` ends with `:\d+`.
        let id_match = greedy_id_match(id_run);
        let id = match id_match {
            Some(id) => id,
            None => {
                // No valid id here; advance past this begin token and retry,
                // mirroring regex restarting the search.
                search_from = after_begin;
                continue;
            }
        };

        let after_id = id_start + id.len();
        // `\s*` then the argument-begin token.
        let ws1 = leading_ws_len(&text[after_id..]);
        let arg_begin_pos = after_id + ws1;
        if !text[arg_begin_pos..].starts_with(TOOL_CALL_ARGUMENT_BEGIN) {
            search_from = after_begin;
            continue;
        }
        let after_arg_begin = arg_begin_pos + TOOL_CALL_ARGUMENT_BEGIN.len();
        // `\s*` before the argument body.
        let body_start = after_arg_begin + leading_ws_len(&text[after_arg_begin..]);

        // `(?:(?!<\|tool_call_begin\|>).)*?` (non-greedy) then optional `\s*`
        // then `<|tool_call_end|>`. Find the earliest `<|tool_call_end|>` that
        // is not preceded by another `<|tool_call_begin|>` within the body.
        let rest = &text[body_start..];
        let end_rel = match find_body_end(rest) {
            Some(e) => e,
            None => {
                // No closing end token reachable; stop (no further matches).
                break;
            }
        };

        let raw_body = &rest[..end_rel];
        // The trailing `\s*` outside the capture trims trailing whitespace.
        let body = raw_body.trim_end_matches(is_ascii_or_unicode_ws);

        out.push((id, body.to_string()));

        // Continue searching after this `<|tool_call_end|>`.
        let end_token_abs = body_start + end_rel;
        search_from = end_token_abs + TOOL_CALL_END.len();
    }

    out
}

/// Length (in bytes) of the leading whitespace run, matching Python `\s`
/// (which includes Unicode whitespace under default `re`). We use Rust's
/// `char::is_whitespace`.
fn leading_ws_len(s: &str) -> usize {
    let mut len = 0usize;
    for c in s.chars() {
        if c.is_whitespace() {
            len += c.len_utf8();
        } else {
            break;
        }
    }
    len
}

fn is_ascii_or_unicode_ws(c: char) -> bool {
    c.is_whitespace()
}

/// Given the run of non-`<` characters, find the longest prefix that matches
/// `[^<]+:\d+` and could be the captured `tool_call_id`. The original regex is
/// greedy with backtracking; the practical result is the longest prefix ending
/// in `:<digits>` such that the digit run is maximal. We return that prefix.
fn greedy_id_match(run: &str) -> Option<String> {
    // Find every `:` followed by at least one digit, taking the rightmost such
    // `:` whose trailing digit run we can extend maximally. Greedy regex would
    // prefer the longest overall match, i.e. the last `:\d+` in the run, with
    // the digit run extended as far as digits continue.
    let bytes = run.as_bytes();
    let mut best: Option<usize> = None; // end index (exclusive) of `:\d+`

    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b':' {
            // Count following digits.
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 {
                // Need at least one char before the colon (`[^<]+`).
                if i >= 1 {
                    best = Some(j);
                }
            }
        }
        i += 1;
    }

    best.map(|end| run[..end].to_string())
}

/// Scan the argument body for the terminating `<|tool_call_end|>`, ensuring the
/// non-greedy capture does not cross a `<|tool_call_begin|>` (the negative
/// lookahead in the source pattern). Returns the byte offset (within `rest`) of
/// the matched `<|tool_call_end|>`, or `None` if it cannot be reached.
fn find_body_end(rest: &str) -> Option<usize> {
    let next_end = rest.find(TOOL_CALL_END);
    let next_begin = rest.find(TOOL_CALL_BEGIN);

    match (next_end, next_begin) {
        (Some(e), Some(b)) => {
            // If another begin token appears before the end token, the negative
            // lookahead forbids consuming it as body, so this match fails.
            if b < e {
                None
            } else {
                Some(e)
            }
        }
        (Some(e), None) => Some(e),
        (None, _) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_section_token_passthrough() {
        let text = "plain text with no tool calls";
        let res = KimiK2ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("plain text with no tool calls"));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn single_tool_call_with_namespace() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("expected tool calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "functions.get_weather:0");
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(tcs[0].call_type, "function");
        assert_eq!(tcs[0].arguments, "{\"city\": \"Paris\"}");
        assert!(res.content.is_none());
    }

    #[test]
    fn function_id_without_namespace() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>get_weather:3<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].id, "get_weather:3");
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(tcs[0].arguments, "{}");
    }

    #[test]
    fn content_before_section() {
        let text = "Let me check.\n<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some("Let me check."));
        assert_eq!(res.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn multiple_tool_calls() {
        let text = "<|tool_calls_section_begin|>\
<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{\"x\": 1}<|tool_call_end|>\
<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{\"y\": 2}<|tool_call_end|>\
<|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "a");
        assert_eq!(tcs[0].arguments, "{\"x\": 1}");
        assert_eq!(tcs[1].name, "b");
        assert_eq!(tcs[1].id, "functions.b:1");
        assert_eq!(tcs[1].arguments, "{\"y\": 2}");
    }

    #[test]
    fn singular_section_token_variant() {
        let text = "<|tool_call_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\"a\":true}<|tool_call_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.expect("singular variant should be recognised");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "f");
    }

    #[test]
    fn whitespace_around_tokens_and_args() {
        let text = "<|tool_calls_section_begin|>\n  <|tool_call_begin|>  functions.do_it:7  <|tool_call_argument_begin|>  {\"k\": \"v\"}  <|tool_call_end|>\n<|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].id, "functions.do_it:7");
        assert_eq!(tcs[0].name, "do_it");
        // Arguments are stripped of surrounding whitespace.
        assert_eq!(tcs[0].arguments, "{\"k\": \"v\"}");
    }

    #[test]
    fn section_token_but_no_valid_calls() {
        // Has a section begin token but no well-formed tool_call blocks.
        let text = "<|tool_calls_section_begin|>garbage without proper tokens";
        let res = KimiK2ToolCallParser::new().parse(text);
        assert_eq!(res.content.as_deref(), Some(text));
        assert!(res.tool_calls.is_none());
    }

    #[test]
    fn dotall_multiline_arguments() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\n  \"a\": 1,\n  \"b\": 2\n}<|tool_call_end|><|tool_calls_section_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].arguments, "{\n  \"a\": 1,\n  \"b\": 2\n}");
    }

    #[test]
    fn empty_content_becomes_none() {
        // Section begins at index 0 -> stripped prefix is empty -> content None.
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>f:0<|tool_call_argument_begin|>{}<|tool_call_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        assert!(res.content.is_none());
        assert_eq!(res.tool_calls.unwrap()[0].name, "f");
    }

    #[test]
    fn name_takes_last_dot_segment() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>a.b.c.tool:12<|tool_call_argument_begin|>{}<|tool_call_end|>";
        let res = KimiK2ToolCallParser::new().parse(text);
        let tcs = res.tool_calls.unwrap();
        assert_eq!(tcs[0].name, "tool");
        assert_eq!(tcs[0].id, "a.b.c.tool:12");
    }

    #[test]
    fn greedy_id_match_basic() {
        assert_eq!(greedy_id_match("functions.f:0").as_deref(), Some("functions.f:0"));
        assert_eq!(greedy_id_match("f:42").as_deref(), Some("f:42"));
        // No `:\d+` -> None.
        assert!(greedy_id_match("nope").is_none());
        // Colon with no digits -> None.
        assert!(greedy_id_match("a:").is_none());
    }
}
