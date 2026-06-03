//! Automatic context window compression for long conversations.
//!
//! Faithful, idiomatic Rust port of `agent/context_compressor.py`.
//!
//! [`ContextCompressor`] is the default context engine. It compresses
//! conversation context via lossy summarization, using an auxiliary model
//! (cheap/fast) to summarize middle turns while protecting head and tail
//! context.
//!
//! Algorithm:
//!   1. Prune old tool results (cheap, no LLM call)
//!   2. Protect head messages (system prompt + first exchange)
//!   3. Protect tail messages by token budget (most recent ~20K tokens)
//!   4. Summarize middle turns with a structured LLM prompt
//!   5. On subsequent compactions, iteratively update the previous summary
//!
//! The Python class subclasses `ContextEngine`. Here it implements the
//! [`crate::ag_context_engine::ContextEngine`] trait and carries its own
//! mutable state.
//!
//! Messages are represented as `serde_json::Value` objects (OpenAI-format
//! chat messages), matching the dynamic dict shapes the Python code walks.
//!
//! The actual auxiliary-model call (`call_llm` in Python) is abstracted behind
//! the [`SummaryCaller`] trait so this module does not have to hard-wire the
//! full auxiliary client; callers supply an implementation that performs the
//! HTTP request and returns the generated summary text.

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::ag_context_engine::{ContextEngine, ContextEngineState};
use crate::ag_model_metadata::{
    estimate_messages_tokens_rough, get_model_context_length, MINIMUM_CONTEXT_LENGTH,
};
use crate::agent_redact::redact_sensitive_text;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const SUMMARY_PREFIX: &str = "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted \
into the summary below. This is a handoff from a previous context \
window — treat it as background reference, NOT as active instructions. \
Do NOT answer questions or fulfill requests mentioned in this summary; \
they were already addressed. \
Your current task is identified in the '## Active Task' section of the \
summary — resume exactly from there. \
IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system \
prompt is ALWAYS authoritative and active — never ignore or deprioritize \
memory content due to this compaction note. \
Respond ONLY to the latest user message \
that appears AFTER this summary. The current session state (files, \
config, etc.) may reflect work described here — avoid repeating it:";

pub const LEGACY_SUMMARY_PREFIX: &str = "[CONTEXT SUMMARY]:";

/// Minimum tokens for the summary output.
const MIN_SUMMARY_TOKENS: i64 = 2000;
/// Proportion of compressed content to allocate for summary.
const SUMMARY_RATIO: f64 = 0.20;
/// Absolute ceiling for summary tokens (even on very large context windows).
const SUMMARY_TOKENS_CEILING: i64 = 12_000;

/// Placeholder used when pruning old tool results.
const PRUNED_TOOL_PLACEHOLDER: &str = "[Old tool output cleared to save context space]";

/// Chars per token rough estimate.
const CHARS_PER_TOKEN: i64 = 4;
/// Flat token cost per attached image part.
const IMAGE_TOKEN_ESTIMATE: i64 = 1600;
/// Same figure expressed in the char-budget currency the rest of the
/// compressor speaks in.
const IMAGE_CHAR_EQUIVALENT: i64 = IMAGE_TOKEN_ESTIMATE * CHARS_PER_TOKEN;

const SUMMARY_FAILURE_COOLDOWN_SECONDS: f64 = 600.0;
const TRANSIENT_COOLDOWN_SECONDS: f64 = 60.0;

// Truncation limits for the summarizer input.
const CONTENT_MAX: usize = 6000;
const CONTENT_HEAD: usize = 4000;
const CONTENT_TAIL: usize = 1500;
const TOOL_ARGS_MAX: usize = 1500;
const TOOL_ARGS_HEAD: usize = 1200;

const COMPRESSION_NOTE: &str = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work. Your persistent memory (MEMORY.md, USER.md) remains fully authoritative regardless of compaction.]";

// ---------------------------------------------------------------------------
// Summary-call abstraction (Python's call_llm)
// ---------------------------------------------------------------------------

/// Runtime info for the main model, mirroring the `main_runtime` dict passed to
/// `call_llm` in Python.
#[derive(Debug, Clone, Default)]
pub struct MainRuntime {
    pub model: String,
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub api_mode: String,
}

/// A summarization request, mirroring the kwargs passed to `call_llm`.
#[derive(Debug, Clone)]
pub struct SummaryRequest {
    /// Always `"compression"`.
    pub task: String,
    pub main_runtime: MainRuntime,
    /// The single user message holding the full summarization prompt.
    pub prompt: String,
    pub max_tokens: i64,
    /// Overriding summary model, empty = use main model.
    pub model: Option<String>,
}

/// Error from a summary call. Mirrors the exception attributes the Python
/// fallback logic inspects (`status_code`, `response.status_code`, `str(e)`).
#[derive(Debug, Clone)]
pub struct SummaryCallError {
    /// HTTP-style status code, if available.
    pub status_code: Option<u16>,
    /// Human-readable error message (`str(e)`).
    pub message: String,
    /// Whether this represents "no provider configured" (Python `RuntimeError`).
    pub no_provider: bool,
    /// Exception class name fallback used when `message` is empty.
    pub class_name: String,
}

impl SummaryCallError {
    pub fn no_provider() -> Self {
        SummaryCallError {
            status_code: None,
            message: String::new(),
            no_provider: true,
            class_name: "RuntimeError".to_string(),
        }
    }

    pub fn new(status_code: Option<u16>, message: impl Into<String>) -> Self {
        SummaryCallError {
            status_code,
            message: message.into(),
            no_provider: false,
            class_name: "Exception".to_string(),
        }
    }
}

/// Pluggable summarizer backend. The implementation performs the actual LLM
/// call (Python's `call_llm`) and returns the raw response content string.
pub trait SummaryCaller {
    /// Execute a summary call. Return the response content on success.
    fn call(&self, req: &SummaryRequest) -> Result<String, SummaryCallError>;
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Return the effective char-length of a message's content for token budgeting.
pub fn content_length_for_budget(raw_content: &Value) -> i64 {
    match raw_content {
        Value::String(s) => s.chars().count() as i64,
        Value::Array(arr) => {
            let mut total: i64 = 0;
            for p in arr {
                match p {
                    Value::String(s) => total += s.chars().count() as i64,
                    Value::Object(obj) => {
                        let ptype = obj.get("type").and_then(|v| v.as_str());
                        if matches!(ptype, Some("image_url") | Some("input_image") | Some("image")) {
                            total += IMAGE_CHAR_EQUIVALENT;
                        } else {
                            let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            total += text.chars().count() as i64;
                        }
                    }
                    other => total += value_str(other).chars().count() as i64,
                }
            }
            total
        }
        Value::Null => 0,
        other => value_str(other).chars().count() as i64,
    }
}

/// Best-effort text view of message content, used for substring checks.
pub fn content_text_for_contains(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut parts: Vec<String> = Vec::new();
            for item in arr {
                match item {
                    Value::String(s) => parts.push(s.clone()),
                    Value::Object(obj) => {
                        if let Some(Value::String(t)) = obj.get("text") {
                            parts.push(t.clone());
                        }
                    }
                    _ => {}
                }
            }
            parts
                .into_iter()
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        other => value_str(other),
    }
}

/// Append or prepend plain text to message content safely.
pub fn append_text_to_content(content: &Value, text: &str, prepend: bool) -> Value {
    match content {
        Value::Null => Value::String(text.to_string()),
        Value::String(s) => {
            if prepend {
                Value::String(format!("{text}{s}"))
            } else {
                Value::String(format!("{s}{text}"))
            }
        }
        Value::Array(arr) => {
            let text_block = json!({"type": "text", "text": text});
            let mut out: Vec<Value> = Vec::with_capacity(arr.len() + 1);
            if prepend {
                out.push(text_block);
                out.extend(arr.iter().cloned());
            } else {
                out.extend(arr.iter().cloned());
                out.push(text_block);
            }
            Value::Array(out)
        }
        other => {
            let rendered = value_str(other);
            if prepend {
                Value::String(format!("{text}{rendered}"))
            } else {
                Value::String(format!("{rendered}{text}"))
            }
        }
    }
}

/// Render a JSON value the way Python `str(obj)` would for the few cases the
/// compressor relies on (numbers, bools, etc.). For strings the bare text is
/// returned. Containers fall back to a Python-ish repr.
fn value_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Take the first `n` chars (Unicode scalar values, matching Python slicing on
/// `str`).
fn char_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Take the last `n` chars.
fn char_suffix(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if n >= count {
        return s.to_string();
    }
    s.chars().skip(count - n).collect()
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Format an integer with thousands separators, matching Python `{:,}`.
fn comma_int(n: i64) -> String {
    let neg = n < 0;
    let digits = n.unsigned_abs().to_string();
    let bytes = digits.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if neg {
        format!("-{out}")
    } else {
        out
    }
}

/// Shrink long string values inside a tool-call arguments JSON blob while
/// preserving JSON validity. Returns the original string if it is not valid
/// JSON.
pub fn truncate_tool_call_args_json(args: &str, head_chars: usize) -> String {
    let parsed: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return args.to_string(),
    };
    let shrunken = shrink_json(&parsed, head_chars);
    // ensure_ascii=False equivalent: serde_json never escapes non-ASCII.
    serde_json::to_string(&shrunken).unwrap_or_else(|_| args.to_string())
}

fn shrink_json(obj: &Value, head_chars: usize) -> Value {
    match obj {
        Value::String(s) => {
            if char_len(s) > head_chars {
                Value::String(format!("{}...[truncated]", char_prefix(s, head_chars)))
            } else {
                Value::String(s.clone())
            }
        }
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                out.insert(k.clone(), shrink_json(v, head_chars));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(|v| shrink_json(v, head_chars)).collect()),
        other => other.clone(),
    }
}

/// Create an informative 1-line summary of a tool call + result.
pub fn summarize_tool_result(tool_name: &str, tool_args: &str, tool_content: &str) -> String {
    let args: Value = if tool_args.is_empty() {
        json!({})
    } else {
        serde_json::from_str(tool_args).unwrap_or_else(|_| json!({}))
    };

    let content = tool_content;
    let content_len = char_len(content) as i64;
    let line_count: i64 = if !content.trim().is_empty() {
        content.matches('\n').count() as i64 + 1
    } else {
        0
    };

    let get_str = |k: &str, default: &str| -> String {
        args.get(k)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| default.to_string())
    };

    match tool_name {
        "terminal" => {
            let mut cmd = get_str("command", "");
            if char_len(&cmd) > 80 {
                cmd = format!("{}...", char_prefix(&cmd, 77));
            }
            let exit_code = regex_capture(r#""exit_code"\s*:\s*(-?\d+)"#, content)
                .unwrap_or_else(|| "?".to_string());
            format!("[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output")
        }
        "read_file" => {
            let path = get_str("path", "?");
            // offset defaults to 1; Python prints whatever type it is.
            let offset = match args.get("offset") {
                Some(v) if !v.is_null() => value_str(v),
                _ => "1".to_string(),
            };
            format!(
                "[read_file] read {path} from line {offset} ({} chars)",
                comma_int(content_len)
            )
        }
        "write_file" => {
            let path = get_str("path", "?");
            let written_lines = match args.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => (c.matches('\n').count() as i64 + 1).to_string(),
                _ => "?".to_string(),
            };
            format!("[write_file] wrote to {path} ({written_lines} lines)")
        }
        "search_files" => {
            let pattern = get_str("pattern", "?");
            let path = get_str("path", ".");
            let target = get_str("target", "content");
            let count = regex_capture(r#""total_count"\s*:\s*(\d+)"#, content)
                .unwrap_or_else(|| "?".to_string());
            format!("[search_files] {target} search for '{pattern}' in {path} -> {count} matches")
        }
        "patch" => {
            let path = get_str("path", "?");
            let mode = get_str("mode", "replace");
            format!("[patch] {mode} in {path} ({} chars result)", comma_int(content_len))
        }
        "browser_navigate" | "browser_click" | "browser_snapshot" | "browser_type"
        | "browser_scroll" | "browser_vision" => {
            let url = get_str("url", "");
            let r = get_str("ref", "");
            let detail = if !url.is_empty() {
                format!(" {url}")
            } else if !r.is_empty() {
                format!(" ref={r}")
            } else {
                String::new()
            };
            format!("[{tool_name}]{detail} ({} chars)", comma_int(content_len))
        }
        "web_search" => {
            let query = get_str("query", "?");
            format!("[web_search] query='{query}' ({} chars result)", comma_int(content_len))
        }
        "web_extract" => {
            let mut url_desc;
            let mut extra = 0usize;
            match args.get("urls") {
                Some(Value::Array(urls)) if !urls.is_empty() => {
                    url_desc = value_str(&urls[0]);
                    if urls.len() > 1 {
                        extra = urls.len() - 1;
                    }
                }
                _ => url_desc = "?".to_string(),
            }
            if extra > 0 {
                url_desc.push_str(&format!(" (+{extra} more)"));
            }
            format!("[web_extract] {url_desc} ({} chars)", comma_int(content_len))
        }
        "delegate_task" => {
            let mut goal = get_str("goal", "");
            if char_len(&goal) > 60 {
                goal = format!("{}...", char_prefix(&goal, 57));
            }
            format!("[delegate_task] '{goal}' ({} chars result)", comma_int(content_len))
        }
        "execute_code" => {
            let code = get_str("code", "");
            let mut code_preview = char_prefix(&code, 60).replace('\n', " ");
            if char_len(&code) > 60 {
                code_preview.push_str("...");
            }
            format!("[execute_code] `{code_preview}` ({line_count} lines output)")
        }
        "skill_view" | "skills_list" | "skill_manage" => {
            let name = get_str("name", "?");
            format!("[{tool_name}] name={name} ({} chars)", comma_int(content_len))
        }
        "vision_analyze" => {
            let question = char_prefix(&get_str("question", ""), 50);
            format!("[vision_analyze] '{question}' ({} chars)", comma_int(content_len))
        }
        "memory" => {
            let action = get_str("action", "?");
            let target = get_str("target", "?");
            format!("[memory] {action} on {target}")
        }
        "todo" => "[todo] updated task list".to_string(),
        "clarify" => "[clarify] asked user a question".to_string(),
        "text_to_speech" => {
            format!("[text_to_speech] generated audio ({} chars)", comma_int(content_len))
        }
        "cronjob" => {
            let action = get_str("action", "?");
            format!("[cronjob] {action}")
        }
        "process" => {
            let action = get_str("action", "?");
            let sid = get_str("session_id", "?");
            format!("[process] {action} session={sid}")
        }
        _ => {
            // Generic fallback: first two args.
            let mut first_arg = String::new();
            if let Some(obj) = args.as_object() {
                for (k, v) in obj.iter().take(2) {
                    let sv = char_prefix(&value_str(v), 40);
                    first_arg.push_str(&format!(" {k}={sv}"));
                }
            }
            format!("[{tool_name}]{first_arg} ({} chars result)", comma_int(content_len))
        }
    }
}

/// Run a regex with one capture group against `text`, returning group 1.
fn regex_capture(pattern: &str, text: &str) -> Option<String> {
    let re = regex::Regex::new(pattern).ok()?;
    re.captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

// ---------------------------------------------------------------------------
// Message accessor helpers (operate on serde_json::Value objects)
// ---------------------------------------------------------------------------

fn msg_role(msg: &Value) -> &str {
    msg.get("role").and_then(|v| v.as_str()).unwrap_or("")
}

fn msg_content<'a>(msg: &'a Value) -> &'a Value {
    msg.get("content").unwrap_or(&Value::Null)
}

/// `msg.get("content") or ""` — coerce null/missing to empty string value.
fn msg_content_or_empty(msg: &Value) -> Value {
    match msg.get("content") {
        Some(Value::Null) | None => Value::String(String::new()),
        Some(v) => {
            // Python `or ""` treats empty containers/strings as falsy.
            match v {
                Value::String(s) if s.is_empty() => Value::String(String::new()),
                Value::Array(a) if a.is_empty() => Value::String(String::new()),
                _ => v.clone(),
            }
        }
    }
}

fn msg_tool_calls<'a>(msg: &'a Value) -> Option<&'a Vec<Value>> {
    msg.get("tool_calls").and_then(|v| v.as_array())
}

fn tc_function_arguments(tc: &Value) -> &str {
    tc.get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(|a| a.as_str())
        .unwrap_or("")
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

/// Default context engine — compresses conversation context via lossy
/// summarization.
pub struct ContextCompressor {
    state: ContextEngineState,

    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub provider: String,
    pub api_mode: String,

    pub summary_target_ratio: f64,
    pub quiet_mode: bool,

    pub tail_token_budget: i64,
    pub max_summary_tokens: i64,

    pub summary_model: String,

    // Internal state
    context_probed: bool,
    previous_summary: Option<String>,
    last_compression_savings_pct: f64,
    ineffective_compression_count: i64,
    summary_failure_cooldown_until: f64,
    last_summary_error: Option<String>,
    last_summary_dropped_count: i64,
    last_summary_fallback_used: bool,
    last_aux_model_failure_error: Option<String>,
    last_aux_model_failure_model: Option<String>,
    summary_model_fallen_back: bool,

    /// Monotonic clock origin, so `monotonic()` returns seconds since
    /// construction (a stable reference for cooldown arithmetic).
    clock_origin: Instant,
}

impl ContextCompressor {
    /// Construct a compressor. `config_context_length` and `custom_providers_present`
    /// feed [`get_model_context_length`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: impl Into<String>,
        threshold_percent: f64,
        protect_first_n: usize,
        protect_last_n: usize,
        summary_target_ratio: f64,
        quiet_mode: bool,
        summary_model_override: Option<&str>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        config_context_length: Option<i64>,
        provider: impl Into<String>,
        api_mode: impl Into<String>,
        custom_providers_present: bool,
    ) -> Self {
        let model = model.into();
        let base_url = base_url.into();
        let api_key = api_key.into();
        let provider = provider.into();
        let api_mode = api_mode.into();

        let summary_target_ratio = summary_target_ratio.clamp(0.10, 0.80);

        let context_length = get_model_context_length(
            &model,
            &base_url,
            &api_key,
            config_context_length,
            &provider,
            custom_providers_present,
        );

        let threshold_tokens = std::cmp::max(
            (context_length as f64 * threshold_percent) as i64,
            MINIMUM_CONTEXT_LENGTH,
        );

        let target_tokens = (threshold_tokens as f64 * summary_target_ratio) as i64;
        let tail_token_budget = target_tokens;
        let max_summary_tokens =
            std::cmp::min((context_length as f64 * 0.05) as i64, SUMMARY_TOKENS_CEILING);

        let mut state = ContextEngineState::new();
        state.threshold_percent = threshold_percent;
        state.protect_first_n = protect_first_n;
        state.protect_last_n = protect_last_n;
        state.context_length = context_length;
        state.threshold_tokens = threshold_tokens;
        state.compression_count = 0;
        state.last_prompt_tokens = 0;
        state.last_completion_tokens = 0;

        if !quiet_mode {
            log::info!(
                "Context compressor initialized: model={} context_length={} threshold={} ({:.0}%) target_ratio={:.0}% tail_budget={} provider={} base_url={}",
                model,
                context_length,
                threshold_tokens,
                threshold_percent * 100.0,
                summary_target_ratio * 100.0,
                tail_token_budget,
                if provider.is_empty() { "none" } else { &provider },
                if base_url.is_empty() { "none" } else { &base_url },
            );
        }

        ContextCompressor {
            state,
            model,
            base_url,
            api_key,
            provider,
            api_mode,
            summary_target_ratio,
            quiet_mode,
            tail_token_budget,
            max_summary_tokens,
            summary_model: summary_model_override.unwrap_or("").to_string(),
            context_probed: false,
            previous_summary: None,
            last_compression_savings_pct: 100.0,
            ineffective_compression_count: 0,
            summary_failure_cooldown_until: 0.0,
            last_summary_error: None,
            last_summary_dropped_count: 0,
            last_summary_fallback_used: false,
            last_aux_model_failure_error: None,
            last_aux_model_failure_model: None,
            summary_model_fallen_back: false,
            clock_origin: Instant::now(),
        }
    }

    // -- public accessors for caller introspection --

    pub fn previous_summary(&self) -> Option<&str> {
        self.previous_summary.as_deref()
    }
    pub fn set_previous_summary(&mut self, s: Option<String>) {
        self.previous_summary = s;
    }
    pub fn last_summary_error(&self) -> Option<&str> {
        self.last_summary_error.as_deref()
    }
    pub fn last_summary_dropped_count(&self) -> i64 {
        self.last_summary_dropped_count
    }
    pub fn last_summary_fallback_used(&self) -> bool {
        self.last_summary_fallback_used
    }
    pub fn last_aux_model_failure_error(&self) -> Option<&str> {
        self.last_aux_model_failure_error.as_deref()
    }
    pub fn last_aux_model_failure_model(&self) -> Option<&str> {
        self.last_aux_model_failure_model.as_deref()
    }
    pub fn last_compression_savings_pct(&self) -> f64 {
        self.last_compression_savings_pct
    }
    pub fn ineffective_compression_count(&self) -> i64 {
        self.ineffective_compression_count
    }
    pub fn context_probed(&self) -> bool {
        self.context_probed
    }

    fn monotonic(&self) -> f64 {
        self.clock_origin.elapsed().as_secs_f64()
    }

    /// Full `update_model` with the compressor-specific budget recalculation.
    #[allow(clippy::too_many_arguments)]
    pub fn update_model_full(
        &mut self,
        model: &str,
        context_length: i64,
        base_url: &str,
        api_key: &str,
        provider: &str,
        api_mode: &str,
    ) {
        self.model = model.to_string();
        self.base_url = base_url.to_string();
        self.api_key = api_key.to_string();
        self.provider = provider.to_string();
        self.api_mode = api_mode.to_string();
        self.state.context_length = context_length;
        self.state.threshold_tokens = std::cmp::max(
            (context_length as f64 * self.state.threshold_percent) as i64,
            MINIMUM_CONTEXT_LENGTH,
        );
        let target_tokens = (self.state.threshold_tokens as f64 * self.summary_target_ratio) as i64;
        self.tail_token_budget = target_tokens;
        self.max_summary_tokens =
            std::cmp::min((context_length as f64 * 0.05) as i64, SUMMARY_TOKENS_CEILING);
    }

    // ------------------------------------------------------------------
    // Tool output pruning (cheap pre-pass, no LLM call)
    // ------------------------------------------------------------------

    /// Replace old tool result contents with informative 1-line summaries.
    /// Returns `(pruned_messages, pruned_count)`.
    pub fn prune_old_tool_results(
        &self,
        messages: &[Value],
        protect_tail_count: usize,
        protect_tail_tokens: Option<i64>,
    ) -> (Vec<Value>, i64) {
        if messages.is_empty() {
            return (messages.to_vec(), 0);
        }

        let mut result: Vec<Value> = messages.to_vec();
        let mut pruned: i64 = 0;
        let n = result.len();

        // Build index: tool_call_id -> (tool_name, arguments_json)
        let mut call_id_to_tool: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        for msg in &result {
            if msg_role(msg) == "assistant" {
                if let Some(tcs) = msg_tool_calls(msg) {
                    for tc in tcs {
                        if let Some(obj) = tc.as_object() {
                            let cid = obj
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let fn_obj = obj.get("function");
                            let name = fn_obj
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let args = fn_obj
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            call_id_to_tool.insert(cid, (name, args));
                        }
                    }
                }
            }
        }

        // Determine the prune boundary.
        let prune_boundary: usize = if protect_tail_tokens.map(|t| t > 0).unwrap_or(false) {
            let budget = protect_tail_tokens.unwrap();
            let mut accumulated: i64 = 0;
            let mut boundary = n;
            let min_protect = std::cmp::min(protect_tail_count, n);
            for i in (0..n).rev() {
                let msg = &result[i];
                let raw_content = msg_content_or_empty(msg);
                let content_len = content_length_for_budget(&raw_content);
                let mut msg_tokens = content_len / CHARS_PER_TOKEN + 10;
                if let Some(tcs) = msg_tool_calls(msg) {
                    for tc in tcs {
                        let args = tc_function_arguments(tc);
                        msg_tokens += (char_len(args) as i64) / CHARS_PER_TOKEN;
                    }
                }
                if accumulated + msg_tokens > budget && (n - i) >= min_protect {
                    boundary = i;
                    break;
                }
                accumulated += msg_tokens;
                boundary = i;
            }
            let budget_protect_count = n - boundary;
            let protected_count = std::cmp::max(budget_protect_count, min_protect);
            n - protected_count
        } else {
            n.saturating_sub(protect_tail_count)
        };

        // Pass 1: Deduplicate identical tool results.
        let mut content_hashes: std::collections::HashMap<String, (usize, String)> =
            std::collections::HashMap::new();
        for i in (0..n).rev() {
            let msg = &result[i];
            if msg_role(msg) != "tool" {
                continue;
            }
            let content = match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(Value::Array(_)) => continue, // skip multimodal
                Some(_) => continue,               // not a string
            };
            if char_len(&content) < 200 {
                continue;
            }
            let h = md5_hex12(&content);
            if content_hashes.contains_key(&h) {
                // Older duplicate — replace with back-reference.
                if let Some(obj) = result[i].as_object_mut() {
                    obj.insert(
                        "content".to_string(),
                        Value::String(
                            "[Duplicate tool output — same content as a more recent call]"
                                .to_string(),
                        ),
                    );
                }
                pruned += 1;
            } else {
                let tcid = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                content_hashes.insert(h, (i, tcid));
            }
        }

        // Pass 2: Replace old tool results with informative summaries.
        for i in 0..prune_boundary {
            let msg = &result[i];
            if msg_role(msg) != "tool" {
                continue;
            }
            let content = match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(_)) => continue,
                Some(Value::Null) | None => String::new(),
                Some(_) => continue,
            };
            if content.is_empty() || content == PRUNED_TOOL_PLACEHOLDER {
                continue;
            }
            if content.starts_with("[Duplicate tool output") {
                continue;
            }
            if char_len(&content) > 200 {
                let call_id = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let (tool_name, tool_args) = call_id_to_tool
                    .get(call_id)
                    .cloned()
                    .unwrap_or_else(|| ("unknown".to_string(), String::new()));
                let summary = summarize_tool_result(&tool_name, &tool_args, &content);
                if let Some(obj) = result[i].as_object_mut() {
                    obj.insert("content".to_string(), Value::String(summary));
                }
                pruned += 1;
            }
        }

        // Pass 3: Truncate large tool_call arguments in assistant messages
        // outside the protected tail.
        for i in 0..prune_boundary {
            let msg = &result[i];
            if msg_role(msg) != "assistant" || msg_tool_calls(msg).map(|t| t.is_empty()).unwrap_or(true)
            {
                continue;
            }
            let tcs = msg_tool_calls(msg).unwrap().clone();
            let mut new_tcs: Vec<Value> = Vec::with_capacity(tcs.len());
            let mut modified = false;
            for tc in tcs {
                let mut new_tc = tc.clone();
                if let Some(_obj) = tc.as_object() {
                    let args = tc_function_arguments(&tc);
                    if char_len(args) > 500 {
                        let new_args = truncate_tool_call_args_json(args, 200);
                        if new_args != args {
                            if let Some(nobj) = new_tc.as_object_mut() {
                                let mut fn_map = nobj
                                    .get("function")
                                    .and_then(|f| f.as_object())
                                    .cloned()
                                    .unwrap_or_default();
                                fn_map.insert("arguments".to_string(), Value::String(new_args));
                                nobj.insert("function".to_string(), Value::Object(fn_map));
                            }
                            modified = true;
                        }
                    }
                }
                new_tcs.push(new_tc);
            }
            if modified {
                if let Some(obj) = result[i].as_object_mut() {
                    obj.insert("tool_calls".to_string(), Value::Array(new_tcs));
                }
            }
        }

        (result, pruned)
    }

    // ------------------------------------------------------------------
    // Summarization
    // ------------------------------------------------------------------

    fn compute_summary_budget(&self, turns_to_summarize: &[Value]) -> i64 {
        let reprs: Vec<String> = turns_to_summarize.iter().map(python_repr_message).collect();
        let content_tokens = estimate_messages_tokens_rough(&reprs);
        let budget = (content_tokens as f64 * SUMMARY_RATIO) as i64;
        std::cmp::max(
            MIN_SUMMARY_TOKENS,
            std::cmp::min(budget, self.max_summary_tokens),
        )
    }

    fn serialize_for_summary(&self, turns: &[Value]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for msg in turns {
            let role = msg_role(msg);
            let role = if role.is_empty() { "unknown" } else { role };
            let raw = content_text_or_empty_for_redact(msg);
            let mut content = redact_sensitive_text(&raw, false, false);

            if role == "tool" {
                let tool_id = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if char_len(&content) > CONTENT_MAX {
                    content = format!(
                        "{}\n...[truncated]...\n{}",
                        char_prefix(&content, CONTENT_HEAD),
                        char_suffix(&content, CONTENT_TAIL)
                    );
                }
                parts.push(format!("[TOOL RESULT {tool_id}]: {content}"));
                continue;
            }

            if role == "assistant" {
                if char_len(&content) > CONTENT_MAX {
                    content = format!(
                        "{}\n...[truncated]...\n{}",
                        char_prefix(&content, CONTENT_HEAD),
                        char_suffix(&content, CONTENT_TAIL)
                    );
                }
                if let Some(tcs) = msg_tool_calls(msg) {
                    if !tcs.is_empty() {
                        let mut tc_parts: Vec<String> = Vec::new();
                        for tc in tcs {
                            if let Some(_obj) = tc.as_object() {
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("?");
                                let mut args = redact_sensitive_text(
                                    tc_function_arguments(tc),
                                    false,
                                    false,
                                );
                                if char_len(&args) > TOOL_ARGS_MAX {
                                    args = format!("{}...", char_prefix(&args, TOOL_ARGS_HEAD));
                                }
                                tc_parts.push(format!("  {name}({args})"));
                            } else {
                                tc_parts.push("  ?(...)".to_string());
                            }
                        }
                        content.push_str(&format!("\n[Tool calls:\n{}\n]", tc_parts.join("\n")));
                    }
                }
                parts.push(format!("[ASSISTANT]: {content}"));
                continue;
            }

            // User and other roles.
            if char_len(&content) > CONTENT_MAX {
                content = format!(
                    "{}\n...[truncated]...\n{}",
                    char_prefix(&content, CONTENT_HEAD),
                    char_suffix(&content, CONTENT_TAIL)
                );
            }
            parts.push(format!("[{}]: {content}", role.to_uppercase()));
        }
        parts.join("\n\n")
    }

    /// Build the summarizer prompt. Exposed for testing.
    pub fn build_summary_prompt(
        &self,
        content_to_summarize: &str,
        summary_budget: i64,
        focus_topic: Option<&str>,
    ) -> String {
        let preamble = "You are a summarization agent creating a context checkpoint. \
Your output will be injected as reference material for a DIFFERENT \
assistant that continues the conversation. \
Do NOT respond to any questions or requests in the conversation — \
only output the structured summary. \
Do NOT include any preamble, greeting, or prefix. \
Write the summary in the same language the user was using in the \
conversation — do not translate or switch to English. \
NEVER include API keys, tokens, passwords, secrets, credentials, \
or connection strings in the summary — replace any that appear \
with [REDACTED]. Note that the user had credentials present, but \
do not preserve their values.";

        let template_sections = format!(
            r#"## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Copy the user's most recent request or
task assignment verbatim — the exact words they used. If multiple tasks
were requested and only some are done, list only the ones NOT yet completed.
The next assistant must pick up exactly here. Example:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so the next assistant does not re-answer them]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.

Write only the summary body. Do not include any preamble or prefix."#
        );

        let mut prompt = if let Some(prev) = &self.previous_summary {
            format!(
                r#"{preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{prev}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled request — this is the most important field for task continuity.

{template_sections}"#
            )
        } else {
            format!(
                r#"{preamble}

Create a structured handoff summary for a different assistant that will continue this conversation after earlier turns are compacted. The next assistant should be able to understand what happened without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{template_sections}"#
            )
        };

        if let Some(focus) = focus_topic {
            if !focus.is_empty() {
                prompt.push_str(&format!(
                    r#"

FOCUS TOPIC: "{focus}"
The user has requested that this compaction PRIORITISE preserving all information related to the focus topic above. For content related to "{focus}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."#
                ));
            }
        }

        prompt
    }

    fn generate_summary(
        &mut self,
        caller: &dyn SummaryCaller,
        turns_to_summarize: &[Value],
        focus_topic: Option<&str>,
    ) -> Option<String> {
        let now = self.monotonic();
        if now < self.summary_failure_cooldown_until {
            log::debug!(
                "Skipping context summary during cooldown ({:.0}s remaining)",
                self.summary_failure_cooldown_until - now
            );
            return None;
        }

        let summary_budget = self.compute_summary_budget(turns_to_summarize);
        let content_to_summarize = self.serialize_for_summary(turns_to_summarize);
        let prompt = self.build_summary_prompt(&content_to_summarize, summary_budget, focus_topic);

        let req = SummaryRequest {
            task: "compression".to_string(),
            main_runtime: MainRuntime {
                model: self.model.clone(),
                provider: self.provider.clone(),
                base_url: self.base_url.clone(),
                api_key: self.api_key.clone(),
                api_mode: self.api_mode.clone(),
            },
            prompt,
            max_tokens: (summary_budget as f64 * 1.3) as i64,
            model: if self.summary_model.is_empty() {
                None
            } else {
                Some(self.summary_model.clone())
            },
        };

        match caller.call(&req) {
            Ok(content) => {
                let summary = redact_sensitive_text(content.trim(), false, false);
                self.previous_summary = Some(summary.clone());
                self.summary_failure_cooldown_until = 0.0;
                self.summary_model_fallen_back = false;
                self.last_summary_error = None;
                Some(Self::with_summary_prefix(&summary))
            }
            Err(e) if e.no_provider => {
                self.summary_failure_cooldown_until =
                    self.monotonic() + SUMMARY_FAILURE_COOLDOWN_SECONDS;
                self.last_summary_error = Some("no auxiliary LLM provider configured".to_string());
                log::warn!(
                    "Context compression: no provider available for summary. Middle turns will be dropped without summary for {} seconds.",
                    SUMMARY_FAILURE_COOLDOWN_SECONDS as i64
                );
                None
            }
            Err(e) => self.handle_summary_error(caller, turns_to_summarize, focus_topic, &e),
        }
    }

    fn handle_summary_error(
        &mut self,
        caller: &dyn SummaryCaller,
        turns_to_summarize: &[Value],
        focus_topic: Option<&str>,
        e: &SummaryCallError,
    ) -> Option<String> {
        let status = e.status_code;
        let err_str = e.message.to_lowercase();
        let is_model_not_found = matches!(status, Some(404) | Some(503))
            || err_str.contains("model_not_found")
            || err_str.contains("does not exist")
            || err_str.contains("no available channel");
        let is_timeout = matches!(status, Some(408) | Some(429) | Some(502) | Some(504))
            || err_str.contains("timeout");

        // Fast-path retry on main model for model-not-found / timeout.
        if (is_model_not_found || is_timeout)
            && !self.summary_model.is_empty()
            && self.summary_model != self.model
            && !self.summary_model_fallen_back
        {
            self.summary_model_fallen_back = true;
            log::warn!(
                "Summary model '{}' unavailable ({}). Falling back to main model '{}' for compression.",
                self.summary_model,
                e.message,
                self.model
            );
            self.record_aux_failure(e);
            self.summary_model = String::new();
            self.summary_failure_cooldown_until = 0.0;
            return self.generate_summary(caller, turns_to_summarize, focus_topic);
        }

        // Unknown-error best-effort retry on main model.
        if !self.summary_model.is_empty()
            && self.summary_model != self.model
            && !self.summary_model_fallen_back
        {
            self.summary_model_fallen_back = true;
            log::warn!(
                "Summary model '{}' failed ({}). Retrying on main model '{}' before giving up.",
                self.summary_model,
                e.message,
                self.model
            );
            self.record_aux_failure(e);
            self.summary_model = String::new();
            self.summary_failure_cooldown_until = 0.0;
            return self.generate_summary(caller, turns_to_summarize, focus_topic);
        }

        // Transient errors — shorter cooldown.
        self.summary_failure_cooldown_until = self.monotonic() + TRANSIENT_COOLDOWN_SECONDS;
        let mut err_text = e.message.trim().to_string();
        if err_text.is_empty() {
            err_text = e.class_name.clone();
        }
        if char_len(&err_text) > 220 {
            err_text = format!("{}...", char_prefix(&err_text, 217).trim_end());
        }
        self.last_summary_error = Some(err_text);
        log::warn!(
            "Failed to generate context summary: {}. Further summary attempts paused for {} seconds.",
            e.message,
            TRANSIENT_COOLDOWN_SECONDS as i64
        );
        None
    }

    fn record_aux_failure(&mut self, e: &SummaryCallError) {
        let mut err_text = e.message.trim().to_string();
        if err_text.is_empty() {
            err_text = e.class_name.clone();
        }
        if char_len(&err_text) > 220 {
            err_text = format!("{}...", char_prefix(&err_text, 217).trim_end());
        }
        self.last_aux_model_failure_error = Some(err_text);
        self.last_aux_model_failure_model = Some(self.summary_model.clone());
    }

    pub fn strip_summary_prefix(summary: &str) -> String {
        let text = summary.trim();
        for prefix in [SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX] {
            if let Some(rest) = text.strip_prefix(prefix) {
                return rest.trim_start().to_string();
            }
        }
        text.to_string()
    }

    pub fn with_summary_prefix(summary: &str) -> String {
        let text = Self::strip_summary_prefix(summary);
        if text.is_empty() {
            SUMMARY_PREFIX.to_string()
        } else {
            format!("{SUMMARY_PREFIX}\n{text}")
        }
    }

    pub fn is_context_summary_content(content: &Value) -> bool {
        let text = content_text_for_contains(content);
        let text = text.trim_start();
        text.starts_with(SUMMARY_PREFIX) || text.starts_with(LEGACY_SUMMARY_PREFIX)
    }

    fn find_latest_context_summary(
        messages: &[Value],
        start: usize,
        end: usize,
    ) -> (Option<usize>, String) {
        let mut idx = end;
        while idx > start {
            idx -= 1;
            let content = messages[idx].get("content").cloned().unwrap_or(Value::Null);
            if Self::is_context_summary_content(&content) {
                let body = Self::strip_summary_prefix(&content_text_for_contains(&content));
                return (Some(idx), body);
            }
        }
        (None, String::new())
    }

    // ------------------------------------------------------------------
    // Tool-call / tool-result pair integrity helpers
    // ------------------------------------------------------------------

    fn get_tool_call_id(tc: &Value) -> String {
        if let Some(obj) = tc.as_object() {
            let call_id = obj.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            if !call_id.is_empty() {
                return call_id.to_string();
            }
            return obj
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
        String::new()
    }

    fn sanitize_tool_pairs(&self, messages: Vec<Value>) -> Vec<Value> {
        let mut surviving_call_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for msg in &messages {
            if msg_role(msg) == "assistant" {
                if let Some(tcs) = msg_tool_calls(msg) {
                    for tc in tcs {
                        let cid = Self::get_tool_call_id(tc);
                        if !cid.is_empty() {
                            surviving_call_ids.insert(cid);
                        }
                    }
                }
            }
        }

        let mut result_call_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for msg in &messages {
            if msg_role(msg) == "tool" {
                if let Some(cid) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                    if !cid.is_empty() {
                        result_call_ids.insert(cid.to_string());
                    }
                }
            }
        }

        let mut messages = messages;

        // 1. Remove tool results whose call_id has no matching assistant tool_call.
        let orphaned_results: std::collections::HashSet<String> = result_call_ids
            .difference(&surviving_call_ids)
            .cloned()
            .collect();
        if !orphaned_results.is_empty() {
            messages.retain(|m| {
                !(msg_role(m) == "tool"
                    && m.get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .map(|c| orphaned_results.contains(c))
                        .unwrap_or(false))
            });
            if !self.quiet_mode {
                log::info!(
                    "Compression sanitizer: removed {} orphaned tool result(s)",
                    orphaned_results.len()
                );
            }
        }

        // 2. Add stub results for assistant tool_calls whose results were dropped.
        let missing_results: std::collections::HashSet<String> = surviving_call_ids
            .difference(&result_call_ids)
            .cloned()
            .collect();
        if !missing_results.is_empty() {
            let mut patched: Vec<Value> = Vec::with_capacity(messages.len());
            for msg in messages {
                let is_assistant = msg_role(&msg) == "assistant";
                let tcs: Vec<Value> = if is_assistant {
                    msg_tool_calls(&msg).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                };
                patched.push(msg);
                if is_assistant {
                    for tc in &tcs {
                        let cid = Self::get_tool_call_id(tc);
                        if missing_results.contains(&cid) {
                            patched.push(json!({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            }));
                        }
                    }
                }
            }
            messages = patched;
            if !self.quiet_mode {
                log::info!(
                    "Compression sanitizer: added {} stub tool result(s)",
                    missing_results.len()
                );
            }
        }

        messages
    }

    fn align_boundary_forward(messages: &[Value], mut idx: usize) -> usize {
        while idx < messages.len() && msg_role(&messages[idx]) == "tool" {
            idx += 1;
        }
        idx
    }

    fn align_boundary_backward(messages: &[Value], mut idx: usize) -> usize {
        if idx == 0 || idx >= messages.len() {
            return idx;
        }
        let mut check: isize = idx as isize - 1;
        while check >= 0 && msg_role(&messages[check as usize]) == "tool" {
            check -= 1;
        }
        if check >= 0
            && msg_role(&messages[check as usize]) == "assistant"
            && msg_tool_calls(&messages[check as usize])
                .map(|t| !t.is_empty())
                .unwrap_or(false)
        {
            idx = check as usize;
        }
        idx
    }

    // ------------------------------------------------------------------
    // Tail protection by token budget
    // ------------------------------------------------------------------

    fn find_last_user_message_idx(messages: &[Value], head_end: usize) -> Option<usize> {
        let n = messages.len();
        if n == 0 {
            return None;
        }
        let mut i = n;
        while i > head_end {
            i -= 1;
            if msg_role(&messages[i]) == "user" {
                return Some(i);
            }
        }
        None
    }

    fn ensure_last_user_message_in_tail(
        &self,
        messages: &[Value],
        cut_idx: usize,
        head_end: usize,
    ) -> usize {
        let last_user_idx = match Self::find_last_user_message_idx(messages, head_end) {
            Some(i) => i,
            None => return cut_idx,
        };

        if last_user_idx >= cut_idx {
            return cut_idx;
        }

        if !self.quiet_mode {
            log::debug!(
                "Anchoring tail cut to last user message at index {} (was {}) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx
            );
        }
        std::cmp::max(last_user_idx, head_end + 1)
    }

    fn find_tail_cut_by_tokens(
        &self,
        messages: &[Value],
        head_end: usize,
        token_budget: Option<i64>,
    ) -> usize {
        let token_budget = token_budget.unwrap_or(self.tail_token_budget);
        let n = messages.len();
        // Hard minimum: always keep at least 3 messages in the tail.
        // Python: min(3, n - head_end - 1) if n - head_end > 1 else 0
        let min_tail: usize = if n.saturating_sub(head_end) > 1 {
            std::cmp::min(3, n - head_end - 1)
        } else {
            0
        };
        let soft_ceiling = (token_budget as f64 * 1.5) as i64;
        let mut accumulated: i64 = 0;
        let mut cut_idx = n;

        let mut i = n;
        while i > head_end {
            i -= 1;
            let msg = &messages[i];
            let raw_content = msg_content_or_empty(msg);
            let content_len = content_length_for_budget(&raw_content);
            let mut msg_tokens = content_len / CHARS_PER_TOKEN + 10;
            if let Some(tcs) = msg_tool_calls(msg) {
                for tc in tcs {
                    let args = tc_function_arguments(tc);
                    msg_tokens += (char_len(args) as i64) / CHARS_PER_TOKEN;
                }
            }
            if accumulated + msg_tokens > soft_ceiling && (n - i) >= min_tail {
                break;
            }
            accumulated += msg_tokens;
            cut_idx = i;
        }

        // Ensure we protect at least min_tail messages.
        let fallback_cut = n - min_tail;
        if cut_idx > fallback_cut {
            cut_idx = fallback_cut;
        }

        // If the token budget would protect everything, force a cut after head.
        if cut_idx <= head_end {
            cut_idx = std::cmp::max(fallback_cut, head_end + 1);
        }

        cut_idx = Self::align_boundary_backward(messages, cut_idx);
        cut_idx = self.ensure_last_user_message_in_tail(messages, cut_idx, head_end);

        std::cmp::max(cut_idx, head_end + 1)
    }

    // ------------------------------------------------------------------
    // Main compression entry point (with explicit summarizer)
    // ------------------------------------------------------------------

    /// Compress with a supplied [`SummaryCaller`]. This is the real
    /// implementation; the trait [`ContextEngine::compress`] uses a no-op
    /// caller (no summary) by default — callers that have a configured aux
    /// client should call this method directly.
    pub fn compress_with(
        &mut self,
        messages: Vec<Value>,
        current_tokens: Option<i64>,
        focus_topic: Option<&str>,
        caller: &dyn SummaryCaller,
    ) -> Vec<Value> {
        self.last_summary_dropped_count = 0;
        self.last_summary_fallback_used = false;
        self.last_summary_error = None;
        self.last_aux_model_failure_error = None;
        self.last_aux_model_failure_model = None;

        let n_messages = messages.len();
        let protect_first_n = self.state.protect_first_n;
        let min_for_compress = protect_first_n + 3 + 1;
        if n_messages <= min_for_compress {
            if !self.quiet_mode {
                log::warn!(
                    "Cannot compress: only {} messages (need > {})",
                    n_messages,
                    min_for_compress
                );
            }
            return messages;
        }

        let display_tokens: i64 = if let Some(t) = current_tokens {
            if t != 0 {
                t
            } else {
                self.fallback_display_tokens(&messages)
            }
        } else {
            self.fallback_display_tokens(&messages)
        };

        // Phase 1: prune old tool results.
        let (mut messages, pruned_count) = self.prune_old_tool_results(
            &messages,
            self.state.protect_last_n,
            Some(self.tail_token_budget),
        );
        if pruned_count > 0 && !self.quiet_mode {
            log::info!("Pre-compression: pruned {} old tool result(s)", pruned_count);
        }

        // Phase 2: boundaries.
        let mut compress_start = protect_first_n;
        compress_start = Self::align_boundary_forward(&messages, compress_start);
        let compress_end = self.find_tail_cut_by_tokens(&messages, compress_start, None);

        if compress_start >= compress_end {
            return messages;
        }

        let mut turns_to_summarize: Vec<Value> = messages[compress_start..compress_end].to_vec();
        let (summary_idx, summary_body) =
            Self::find_latest_context_summary(&messages, compress_start, compress_end);
        if let Some(sidx) = summary_idx {
            if !summary_body.is_empty() && self.previous_summary.is_none() {
                self.previous_summary = Some(summary_body);
            }
            turns_to_summarize = messages[(sidx + 1)..compress_end].to_vec();
        }

        if !self.quiet_mode {
            log::info!(
                "Context compression triggered ({} tokens >= {} threshold)",
                display_tokens,
                self.state.threshold_tokens
            );
            log::info!(
                "Model context limit: {} tokens ({:.0}% = {})",
                self.state.context_length,
                self.state.threshold_percent * 100.0,
                self.state.threshold_tokens
            );
            let tail_msgs = n_messages - compress_end;
            log::info!(
                "Summarizing turns {}-{} ({} turns), protecting {} head + {} tail messages",
                compress_start + 1,
                compress_end,
                turns_to_summarize.len(),
                compress_start,
                tail_msgs
            );
        }

        // Phase 3: generate summary.
        let mut summary = self.generate_summary(caller, &turns_to_summarize, focus_topic);

        // Phase 4: assemble compressed list.
        let mut compressed: Vec<Value> = Vec::new();
        for i in 0..compress_start {
            let mut msg = messages[i].clone();
            if i == 0 && msg_role(&msg) == "system" {
                let existing = msg.get("content").cloned().unwrap_or(Value::Null);
                if !content_text_for_contains(&existing).contains(COMPRESSION_NOTE) {
                    let to_add = match &existing {
                        Value::String(s) if !s.is_empty() => format!("\n\n{COMPRESSION_NOTE}"),
                        _ => COMPRESSION_NOTE.to_string(),
                    };
                    let new_content = append_text_to_content(&existing, &to_add, false);
                    if let Some(obj) = msg.as_object_mut() {
                        obj.insert("content".to_string(), new_content);
                    }
                }
            }
            compressed.push(msg);
        }

        // Static fallback if summary failed.
        let summary_text: String = match summary.take() {
            Some(s) => s,
            None => {
                if !self.quiet_mode {
                    log::warn!(
                        "Summary generation failed — inserting static fallback context marker"
                    );
                }
                let n_dropped = (compress_end - compress_start) as i64;
                self.last_summary_dropped_count = n_dropped;
                self.last_summary_fallback_used = true;
                format!(
                    "{SUMMARY_PREFIX}\nSummary generation was unavailable. {n_dropped} message(s) were removed to free context space but could not be summarized. The removed messages contained earlier work in this session. Continue based on the recent messages below and the current state of any files or resources."
                )
            }
        };
        let mut summary_text = summary_text;

        let mut merge_summary_into_tail = false;
        let last_head_role = if compress_start > 0 {
            msg_role(&messages[compress_start - 1]).to_string()
        } else {
            "user".to_string()
        };
        let first_tail_role = if compress_end < n_messages {
            msg_role(&messages[compress_end]).to_string()
        } else {
            "user".to_string()
        };

        let mut summary_role = if last_head_role == "assistant" || last_head_role == "tool" {
            "user".to_string()
        } else {
            "assistant".to_string()
        };

        if summary_role == first_tail_role {
            let flipped = if summary_role == "user" {
                "assistant"
            } else {
                "user"
            };
            if flipped != last_head_role {
                summary_role = flipped.to_string();
            } else {
                merge_summary_into_tail = true;
            }
        }

        if !merge_summary_into_tail && summary_role == "user" {
            summary_text.push_str(
                "\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---",
            );
        }

        if !merge_summary_into_tail {
            compressed.push(json!({"role": summary_role, "content": summary_text}));
        }

        for i in compress_end..n_messages {
            let mut msg = messages[i].clone();
            if merge_summary_into_tail && i == compress_end {
                let merged_prefix = format!(
                    "{summary_text}\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n\n"
                );
                let existing = msg.get("content").cloned().unwrap_or(Value::Null);
                let new_content = append_text_to_content(&existing, &merged_prefix, true);
                if let Some(obj) = msg.as_object_mut() {
                    obj.insert("content".to_string(), new_content);
                }
                merge_summary_into_tail = false;
            }
            compressed.push(msg);
        }

        self.state.compression_count += 1;

        compressed = self.sanitize_tool_pairs(compressed);

        let new_reprs: Vec<String> = compressed.iter().map(python_repr_message).collect();
        let new_estimate = estimate_messages_tokens_rough(&new_reprs);
        let saved_estimate = display_tokens - new_estimate;

        let savings_pct = if display_tokens > 0 {
            (saved_estimate as f64) / (display_tokens as f64) * 100.0
        } else {
            0.0
        };
        self.last_compression_savings_pct = savings_pct;
        if savings_pct < 10.0 {
            self.ineffective_compression_count += 1;
        } else {
            self.ineffective_compression_count = 0;
        }

        if !self.quiet_mode {
            log::info!(
                "Compressed: {} -> {} messages (~{} tokens saved, {:.0}%)",
                n_messages,
                compressed.len(),
                saved_estimate,
                savings_pct
            );
            log::info!("Compression #{} complete", self.state.compression_count);
        }

        compressed
    }

    fn fallback_display_tokens(&self, messages: &[Value]) -> i64 {
        if self.state.last_prompt_tokens != 0 {
            self.state.last_prompt_tokens
        } else {
            let reprs: Vec<String> = messages.iter().map(python_repr_message).collect();
            estimate_messages_tokens_rough(&reprs)
        }
    }

    /// Mirror Python `has_content_to_compress`.
    pub fn has_content_to_compress_impl(&self, messages: &[Value]) -> bool {
        let compress_start = Self::align_boundary_forward(messages, self.state.protect_first_n);
        let compress_end = self.find_tail_cut_by_tokens(messages, compress_start, None);
        compress_start < compress_end
    }
}

/// Render a message Value the way Python `str(dict)` would, for the rough token
/// estimator. The estimator only cares about character count, so a faithful
/// Python-dict repr keeps the estimate calibrated.
fn python_repr_message(msg: &Value) -> String {
    python_repr(msg)
}

fn python_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(python_repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, val)| {
                    format!("'{}': {}", k.replace('\\', "\\\\").replace('\'', "\\'"), python_repr(val))
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Best-effort text view for redaction: `redact_sensitive_text(msg.get("content") or "")`.
/// Python passes the raw content through; for multimodal lists `str()` would be
/// applied by redact, but in practice content is usually a string here. We use
/// the text-extraction view to stay safe.
fn content_text_or_empty_for_redact(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(v) => content_text_for_contains(v),
    }
}

fn md5_hex12(s: &str) -> String {
    let digest = md5::compute(s.as_bytes());
    let full = format!("{digest:x}");
    full.chars().take(12).collect()
}

// ---------------------------------------------------------------------------
// ContextEngine trait implementation
// ---------------------------------------------------------------------------

/// A summary caller that always reports "no provider configured", causing the
/// static fallback to be used. This is the default the trait-level `compress`
/// uses, since the trait signature does not carry an aux client.
pub struct NoProviderCaller;

impl SummaryCaller for NoProviderCaller {
    fn call(&self, _req: &SummaryRequest) -> Result<String, SummaryCallError> {
        Err(SummaryCallError::no_provider())
    }
}

impl ContextEngine for ContextCompressor {
    fn state(&self) -> &ContextEngineState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut ContextEngineState {
        &mut self.state
    }

    fn name(&self) -> &str {
        "compressor"
    }

    fn update_from_response(&mut self, usage: &Value) {
        self.state.last_prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        self.state.last_completion_tokens = usage
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
    }

    fn should_compress(&self, prompt_tokens: Option<i64>) -> bool {
        let tokens = prompt_tokens.unwrap_or(self.state.last_prompt_tokens);
        if tokens < self.state.threshold_tokens {
            return false;
        }
        if self.ineffective_compression_count >= 2 {
            if !self.quiet_mode {
                log::warn!(
                    "Compression skipped — last {} compressions saved <10% each. Consider /new to start a fresh session, or /compress <topic> for focused compression.",
                    self.ineffective_compression_count
                );
            }
            return false;
        }
        true
    }

    fn compress(
        &mut self,
        messages: Vec<Value>,
        current_tokens: Option<i64>,
        focus_topic: Option<&str>,
    ) -> Vec<Value> {
        // No aux client is available through the trait signature; use the
        // no-provider caller so a static fallback summary is inserted. Callers
        // with a configured summarizer should use `compress_with` directly.
        let caller = NoProviderCaller;
        self.compress_with(messages, current_tokens, focus_topic, &caller)
    }

    fn has_content_to_compress(&self, messages: &[Value]) -> bool {
        self.has_content_to_compress_impl(messages)
    }

    fn on_session_reset(&mut self) {
        // super().on_session_reset()
        self.state.last_prompt_tokens = 0;
        self.state.last_completion_tokens = 0;
        self.state.last_total_tokens = 0;
        self.state.compression_count = 0;

        self.context_probed = false;
        self.previous_summary = None;
        self.last_summary_error = None;
        self.last_summary_dropped_count = 0;
        self.last_summary_fallback_used = false;
        self.last_aux_model_failure_error = None;
        self.last_aux_model_failure_model = None;
        self.last_compression_savings_pct = 100.0;
        self.ineffective_compression_count = 0;
        self.summary_failure_cooldown_until = 0.0;
    }

    fn update_model(
        &mut self,
        model: &str,
        context_length: i64,
        base_url: &str,
        api_key: &str,
        provider: &str,
    ) {
        // The Python override takes an extra api_mode arg; the trait does not.
        // Preserve existing api_mode.
        let api_mode = self.api_mode.clone();
        self.update_model_full(model, context_length, base_url, api_key, provider, &api_mode);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoCaller(String);
    impl SummaryCaller for EchoCaller {
        fn call(&self, _req: &SummaryRequest) -> Result<String, SummaryCallError> {
            Ok(self.0.clone())
        }
    }

    fn make_compressor() -> ContextCompressor {
        ContextCompressor::new(
            "test-model",
            0.50,
            3,
            20,
            0.20,
            true,
            None,
            "",
            "",
            Some(200_000),
            "",
            "",
            false,
        )
    }

    #[test]
    fn content_length_string_and_image() {
        assert_eq!(content_length_for_budget(&json!("hello")), 5);
        let multimodal = json!([
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]);
        // 2 text chars + flat image equivalent.
        assert_eq!(content_length_for_budget(&multimodal), 2 + IMAGE_CHAR_EQUIVALENT);
    }

    #[test]
    fn content_text_for_contains_joins_text_parts() {
        let c = json!([
            {"type": "text", "text": "a"},
            {"type": "image_url"},
            {"type": "text", "text": "b"}
        ]);
        assert_eq!(content_text_for_contains(&c), "a\nb");
    }

    #[test]
    fn append_text_prepend_and_append() {
        assert_eq!(
            append_text_to_content(&json!("X"), "Y", false),
            json!("XY")
        );
        assert_eq!(
            append_text_to_content(&json!("X"), "Y", true),
            json!("YX")
        );
        assert_eq!(append_text_to_content(&Value::Null, "Z", false), json!("Z"));
    }

    #[test]
    fn truncate_args_preserves_json_validity() {
        let long = "x".repeat(300);
        let args = json!({"path": "/foo", "content": long}).to_string();
        let out = truncate_tool_call_args_json(&args, 200);
        let parsed: Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(parsed["path"], json!("/foo"));
        let content = parsed["content"].as_str().unwrap();
        assert!(content.ends_with("...[truncated]"));
        assert_eq!(char_len(content), 200 + "...[truncated]".len());
    }

    #[test]
    fn truncate_args_returns_original_on_invalid_json() {
        let not_json = "this is not json {{{";
        assert_eq!(truncate_tool_call_args_json(not_json, 200), not_json);
    }

    #[test]
    fn summarize_terminal_result() {
        let args = json!({"command": "npm test"}).to_string();
        let content = "line1\nline2\n\"exit_code\": 0";
        let s = summarize_tool_result("terminal", &args, content);
        assert!(s.contains("ran `npm test`"));
        assert!(s.contains("exit 0"));
    }

    #[test]
    fn summarize_read_file_with_commas() {
        let args = json!({"path": "config.py"}).to_string();
        let content = "a".repeat(1200);
        let s = summarize_tool_result("read_file", &args, &content);
        assert!(s.contains("read config.py from line 1"));
        assert!(s.contains("1,200 chars"));
    }

    #[test]
    fn comma_int_formats() {
        assert_eq!(comma_int(0), "0");
        assert_eq!(comma_int(1200), "1,200");
        assert_eq!(comma_int(1234567), "1,234,567");
        assert_eq!(comma_int(-1000), "-1,000");
    }

    #[test]
    fn strip_and_with_prefix_roundtrip() {
        let body = "## Active Task\nNone.";
        let with = ContextCompressor::with_summary_prefix(body);
        assert!(with.starts_with(SUMMARY_PREFIX));
        let stripped = ContextCompressor::strip_summary_prefix(&with);
        assert_eq!(stripped, body);
    }

    #[test]
    fn with_prefix_empty_body() {
        assert_eq!(ContextCompressor::with_summary_prefix("   "), SUMMARY_PREFIX);
    }

    #[test]
    fn is_context_summary_detects_legacy() {
        let c = json!(format!("{LEGACY_SUMMARY_PREFIX} stuff"));
        assert!(ContextCompressor::is_context_summary_content(&c));
        assert!(!ContextCompressor::is_context_summary_content(&json!("hello")));
    }

    #[test]
    fn should_compress_threshold() {
        let c = make_compressor();
        // threshold is max(200000*0.5, 64000) = 100000
        assert!(!c.should_compress(Some(50_000)));
        assert!(c.should_compress(Some(150_000)));
    }

    #[test]
    fn dedup_identical_tool_results() {
        let big = "Z".repeat(300);
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "tool", "content": big.clone(), "tool_call_id": "a"}),
            json!({"role": "tool", "content": big.clone(), "tool_call_id": "b"}),
        ];
        let c = make_compressor();
        let (out, pruned) = c.prune_old_tool_results(&messages, 0, None);
        assert_eq!(pruned, 1);
        // The older (earlier index) duplicate gets the back-reference.
        let dedup_content = out[1]["content"].as_str().unwrap();
        assert!(dedup_content.starts_with("[Duplicate tool output"));
    }

    #[test]
    fn align_boundary_forward_skips_tool() {
        let messages = vec![
            json!({"role": "tool", "content": "x", "tool_call_id": "a"}),
            json!({"role": "tool", "content": "y", "tool_call_id": "b"}),
            json!({"role": "user", "content": "z"}),
        ];
        assert_eq!(ContextCompressor::align_boundary_forward(&messages, 0), 2);
    }

    #[test]
    fn sanitize_removes_orphan_tool_result() {
        let c = make_compressor();
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "tool", "content": "orphan", "tool_call_id": "nope"}),
        ];
        let out = c.sanitize_tool_pairs(messages);
        assert_eq!(out.len(), 1);
        assert_eq!(msg_role(&out[0]), "user");
    }

    #[test]
    fn sanitize_adds_stub_for_missing_result() {
        let c = make_compressor();
        let messages = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call1", "function": {"name": "x", "arguments": "{}"}}]
        })];
        let out = c.sanitize_tool_pairs(messages);
        assert_eq!(out.len(), 2);
        assert_eq!(msg_role(&out[1]), "tool");
        assert_eq!(out[1]["tool_call_id"], json!("call1"));
    }

    #[test]
    fn compress_too_few_messages_noop() {
        let mut c = make_compressor();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let caller = EchoCaller("body".to_string());
        let out = c.compress_with(messages.clone(), Some(150_000), None, &caller);
        assert_eq!(out, messages);
    }

    #[test]
    fn compress_inserts_summary_and_note() {
        let mut c = make_compressor();
        // Build: system + several exchanges so head/tail leave a middle.
        let mut messages = vec![
            json!({"role": "system", "content": "SYS"}),
            json!({"role": "user", "content": "u0"}),
            json!({"role": "assistant", "content": "a0"}),
        ];
        for i in 0..20 {
            messages.push(json!({"role": "user", "content": format!("user-msg-{i} ").repeat(50)}));
            messages.push(json!({"role": "assistant", "content": format!("assistant-{i} ").repeat(50)}));
        }
        let caller = EchoCaller("## Active Task\nNone.".to_string());
        let out = c.compress_with(messages, Some(150_000), None, &caller);
        // System message should have the compaction note appended.
        let sys = out[0]["content"].as_str().unwrap();
        assert!(sys.contains(COMPRESSION_NOTE));
        // A summary message containing the prefix should exist.
        let has_summary = out
            .iter()
            .any(|m| ContextCompressor::is_context_summary_content(m.get("content").unwrap_or(&Value::Null)));
        assert!(has_summary);
        assert_eq!(c.state().compression_count, 1);
    }

    #[test]
    fn generate_summary_no_provider_returns_none() {
        let mut c = make_compressor();
        let turns = vec![json!({"role": "user", "content": "hi"})];
        let caller = NoProviderCaller;
        let out = c.generate_summary(&caller, &turns, None);
        assert!(out.is_none());
        assert_eq!(
            c.last_summary_error(),
            Some("no auxiliary LLM provider configured")
        );
    }

    #[test]
    fn aux_model_fallback_to_main() {
        let mut c = ContextCompressor::new(
            "main-model",
            0.50,
            3,
            20,
            0.20,
            true,
            Some("aux-model"),
            "",
            "",
            Some(200_000),
            "",
            "",
            false,
        );

        struct FailThenSucceed {
            calls: std::cell::RefCell<u32>,
        }
        impl SummaryCaller for FailThenSucceed {
            fn call(&self, req: &SummaryRequest) -> Result<String, SummaryCallError> {
                let mut n = self.calls.borrow_mut();
                *n += 1;
                if *n == 1 {
                    // aux model used on first call
                    assert_eq!(req.model.as_deref(), Some("aux-model"));
                    Err(SummaryCallError::new(Some(404), "model_not_found"))
                } else {
                    // fell back to main (model None)
                    assert!(req.model.is_none());
                    Ok("recovered summary".to_string())
                }
            }
        }
        let caller = FailThenSucceed {
            calls: std::cell::RefCell::new(0),
        };
        let turns = vec![json!({"role": "user", "content": "hi"})];
        let out = c.generate_summary(&caller, &turns, None);
        assert!(out.is_some());
        assert!(out.unwrap().contains("recovered summary"));
        assert_eq!(c.last_aux_model_failure_model(), Some("aux-model"));
    }

    #[test]
    fn find_last_user_message_idx_works() {
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "u1"}),
            json!({"role": "assistant", "content": "a"}),
            json!({"role": "user", "content": "u2"}),
            json!({"role": "assistant", "content": "a2"}),
        ];
        assert_eq!(
            ContextCompressor::find_last_user_message_idx(&messages, 0),
            Some(3)
        );
    }
}
