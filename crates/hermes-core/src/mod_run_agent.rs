//! Native Rust port of `run_agent.py` — the AI Agent runner with tool calling.
//!
//! The Python module is dominated by a single ~13K-line stateful `AIAgent`
//! class that drives the OpenAI/Anthropic streaming loop, credential-pool
//! failover, threaded tool execution, and dozens of provider-specific
//! quirks. That stateful orchestration is deeply coupled to the OpenAI SDK,
//! Python threading, and many sibling Python sub-packages, so the value that
//! ports cleanly and idiomatically into a self-contained Rust module is the
//! large surface of **pure / deterministic helper logic** that the class and
//! the module rely on:
//!
//! * module-level helpers (`_get_proxy_from_env`, `_is_destructive_command`,
//!   parallelization safety checks, surrogate / non-ASCII sanitisation, tool
//!   call argument JSON repair, provider headers, pool-recovery decisions);
//! * `IterationBudget` (thread-safe iteration counter);
//! * pure / static `AIAgent` methods (`_strip_think_blocks`,
//!   `_extract_reasoning`, `_summarize_api_error`, `_sanitize_api_messages`,
//!   `_drop_thinking_only_and_merge_users`, `_deduplicate_tool_calls`,
//!   `_repair_tool_call`, `_sanitize_tool_calls_for_strict_api`,
//!   `_sanitize_tool_call_arguments`, error-context extraction, etc.).
//!
//! These are reproduced faithfully, byte-for-byte where the Python relies on
//! exact string/regex behaviour. Behaviour that is intrinsically tied to live
//! OpenAI SDK objects, sockets, threads or unported Python packages is exposed
//! through small Rust data structures (`ToolCall`, `Message`) so the logic is
//! independently testable and reusable by other ported modules.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use regex::Regex;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Module-level constants (mirrors run_agent.py module scope)
// ---------------------------------------------------------------------------

/// QwenCode CLI version string used for portal.qwen.ai compatibility headers.
pub const QWEN_CODE_VERSION: &str = "0.14.1";

/// Marker prepended to tool messages when tool-call arguments were corrupted
/// and dropped to keep the conversation alive (see issue #15236).
pub const TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER: &str =
    "[hermes-agent: tool call arguments were corrupted in this session and \
have been dropped to keep the conversation alive. See issue #15236.]";

/// Maximum number of concurrent worker threads for parallel tool execution.
pub const MAX_TOOL_WORKERS: usize = 8;

/// Tools that must never run concurrently (interactive / user-facing).
pub const NEVER_PARALLEL_TOOLS: &[&str] = &["clarify"];

/// Read-only tools with no shared mutable session state.
pub const PARALLEL_SAFE_TOOLS: &[&str] = &[
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
];

/// File tools that can run concurrently when they target independent paths.
pub const PATH_SCOPED_TOOLS: &[&str] = &["read_file", "write_file", "patch"];

/// Valid API roles accepted by the chat completions endpoint.
pub const VALID_API_ROLES: &[&str] = &[
    "system",
    "user",
    "assistant",
    "tool",
    "function",
    "developer",
];

// ---------------------------------------------------------------------------
// Small data structures bridging Python's duck-typed objects.
// ---------------------------------------------------------------------------

/// A function payload inside a tool call (`tc.function`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

/// A tool call entry, equivalent to the OpenAI SDK `tool_call` object/dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolFunction,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        ToolCall {
            id: id.into(),
            function: ToolFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy-compiled regexes (module-level patterns from run_agent.py)
// ---------------------------------------------------------------------------

fn surrogate_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    // Rust strings cannot contain lone surrogates (UTF-8 invariant), so this
    // pattern can only ever match if a producer encoded surrogate code points
    // as their literal char escapes. We keep the pattern for parity and to
    // detect escaped representations.
    RE.get_or_init(|| Regex::new(r"[\x{d800}-\x{dfff}]").unwrap_or_else(|_| Regex::new("$^").unwrap()))
}

fn destructive_patterns_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // VERBOSE pattern from run_agent.py, flattened.
        Regex::new(
            r"(?:^|\s|&&|\|\||;|`)(?:rm\s|rmdir\s|cp\s|install\s|mv\s|sed\s+-i|truncate\s|dd\s|shred\s|git\s+(?:reset|clean|checkout)\s)",
        )
        .unwrap()
    })
}

fn redirect_overwrite_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^>]>[^>]|^>[^>]").unwrap())
}

// ---------------------------------------------------------------------------
// Proxy environment helpers
// ---------------------------------------------------------------------------

/// Normalize a proxy URL — mirrors `utils.normalize_proxy_url`.
///
/// The Python helper trims whitespace and prepends `http://` when no scheme is
/// present. We reproduce that minimal behaviour here so the module is
/// self-contained; callers that have the real `crate::mod_utils` helper should
/// prefer it.
pub fn normalize_proxy_url(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

/// Read proxy URL from environment variables.
///
/// Checks HTTPS_PROXY, HTTP_PROXY, ALL_PROXY (and lowercase variants) in order.
/// Returns the first valid proxy URL found, or None if no proxy is configured.
pub fn get_proxy_from_env() -> Option<String> {
    for key in [
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(normalize_proxy_url(value));
            }
        }
    }
    None
}

/// Extract the hostname from a base URL — minimal port of
/// `utils.base_url_hostname`.
pub fn base_url_hostname(base_url: &str) -> Option<String> {
    if base_url.is_empty() {
        return None;
    }
    url::Url::parse(base_url).ok().and_then(|u| u.host_str().map(|s| s.to_string()))
}

// ---------------------------------------------------------------------------
// IterationBudget — thread-safe iteration counter for an agent.
// ---------------------------------------------------------------------------

/// Thread-safe iteration counter for an agent.
///
/// Each agent (parent or subagent) gets its own `IterationBudget`. The
/// parent's budget is capped at `max_iterations`. `execute_code` iterations
/// are refunded via [`IterationBudget::refund`] so they don't eat the budget.
#[derive(Debug)]
pub struct IterationBudget {
    pub max_total: i64,
    used: Mutex<i64>,
}

impl IterationBudget {
    pub fn new(max_total: i64) -> Self {
        IterationBudget {
            max_total,
            used: Mutex::new(0),
        }
    }

    /// Try to consume one iteration. Returns `true` if allowed.
    pub fn consume(&self) -> bool {
        let mut used = self.used.lock().unwrap();
        if *used >= self.max_total {
            return false;
        }
        *used += 1;
        true
    }

    /// Give back one iteration (e.g. for execute_code turns).
    pub fn refund(&self) {
        let mut used = self.used.lock().unwrap();
        if *used > 0 {
            *used -= 1;
        }
    }

    pub fn used(&self) -> i64 {
        *self.used.lock().unwrap()
    }

    pub fn remaining(&self) -> i64 {
        let used = *self.used.lock().unwrap();
        std::cmp::max(0, self.max_total - used)
    }
}

// ---------------------------------------------------------------------------
// Destructive-command heuristics
// ---------------------------------------------------------------------------

/// Heuristic: does this terminal command look like it modifies/deletes files?
pub fn is_destructive_command(cmd: &str) -> bool {
    if cmd.is_empty() {
        return false;
    }
    if destructive_patterns_re().is_match(cmd) {
        return true;
    }
    if redirect_overwrite_re().is_match(cmd) {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Parallelization safety
// ---------------------------------------------------------------------------

/// Return the normalized file target for path-scoped tools.
///
/// `cwd` is the current working directory used when resolving relative paths
/// (Python uses `Path.cwd()`).
pub fn extract_parallel_scope_path(
    tool_name: &str,
    function_args: &Value,
    cwd: &Path,
) -> Option<PathBuf> {
    if !PATH_SCOPED_TOOLS.contains(&tool_name) {
        return None;
    }
    let raw_path = function_args.get("path").and_then(|v| v.as_str())?;
    if raw_path.trim().is_empty() {
        return None;
    }

    let expanded = expanduser(raw_path);
    if expanded.is_absolute() {
        return Some(abspath(&expanded));
    }
    Some(abspath(&cwd.join(&expanded)))
}

/// Minimal `~` expansion (Python `Path.expanduser`).
fn expanduser(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    } else if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(p)
}

/// Normalise a path lexically the way Python's `os.path.abspath` does: it does
/// NOT touch the filesystem, collapses `.` and `..`, but keeps symlinks.
fn abspath(p: &Path) -> PathBuf {
    let mut components: Vec<std::ffi::OsString> = Vec::new();
    let mut is_abs = false;
    for comp in p.components() {
        use std::path::Component::*;
        match comp {
            RootDir => {
                is_abs = true;
                components.clear();
            }
            Prefix(prefix) => {
                components.push(prefix.as_os_str().to_os_string());
            }
            CurDir => {}
            ParentDir => {
                if components.last().map(|c| c.to_str() != Some("/")).unwrap_or(false) {
                    components.pop();
                }
            }
            Normal(part) => components.push(part.to_os_string()),
        }
    }
    let mut out = PathBuf::new();
    if is_abs {
        out.push("/");
    }
    for c in components {
        out.push(c);
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Return `true` when two paths may refer to the same subtree.
pub fn paths_overlap(left: &Path, right: &Path) -> bool {
    let left_parts: Vec<_> = left.components().collect();
    let right_parts: Vec<_> = right.components().collect();
    if left_parts.is_empty() || right_parts.is_empty() {
        return (!left_parts.is_empty()) == (!right_parts.is_empty()) && !left_parts.is_empty();
    }
    let common_len = left_parts.len().min(right_parts.len());
    left_parts[..common_len] == right_parts[..common_len]
}

/// Return `true` when a tool-call batch is safe to run concurrently.
///
/// `mcp_parallel_safe` is a callback for the lazy `tools.mcp_tool`
/// `is_mcp_tool_parallel_safe` import — pass a closure returning `false` if
/// unavailable. `cwd` resolves relative path-scoped targets.
pub fn should_parallelize_tool_batch<F>(
    tool_calls: &[ToolCall],
    cwd: &Path,
    mcp_parallel_safe: F,
) -> bool
where
    F: Fn(&str) -> bool,
{
    if tool_calls.len() <= 1 {
        return false;
    }

    let tool_names: Vec<&str> = tool_calls.iter().map(|tc| tc.function.name.as_str()).collect();
    if tool_names.iter().any(|name| NEVER_PARALLEL_TOOLS.contains(name)) {
        return false;
    }

    let mut reserved_paths: Vec<PathBuf> = Vec::new();
    for tool_call in tool_calls {
        let tool_name = tool_call.function.name.as_str();
        let function_args: Value = match serde_json::from_str(&tool_call.function.arguments) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !function_args.is_object() {
            return false;
        }

        if PATH_SCOPED_TOOLS.contains(&tool_name) {
            let scoped_path = match extract_parallel_scope_path(tool_name, &function_args, cwd) {
                Some(p) => p,
                None => return false,
            };
            if reserved_paths.iter().any(|existing| paths_overlap(&scoped_path, existing)) {
                return false;
            }
            reserved_paths.push(scoped_path);
            continue;
        }

        if !PARALLEL_SAFE_TOOLS.contains(&tool_name) && !mcp_parallel_safe(tool_name) {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Surrogate sanitisation
// ---------------------------------------------------------------------------

/// Replace lone surrogate code points with U+FFFD (replacement character).
///
/// Rust `&str` is guaranteed valid UTF-8 so it cannot literally hold a lone
/// surrogate; this is a fast no-op in practice but matches the Python contract
/// for the escaped-representation case.
pub fn sanitize_surrogates(text: &str) -> String {
    if surrogate_re().is_match(text) {
        surrogate_re().replace_all(text, "\u{fffd}").into_owned()
    } else {
        text.to_string()
    }
}

/// Replace surrogate code points in nested JSON payloads in-place.
/// Returns `true` if any surrogates were replaced.
pub fn sanitize_structure_surrogates(payload: &mut Value) -> bool {
    let mut found = false;
    walk_sanitize(payload, &mut found, &|s| {
        if surrogate_re().is_match(s) {
            Some(surrogate_re().replace_all(s, "\u{fffd}").into_owned())
        } else {
            None
        }
    });
    found
}

/// Generic in-place walker: applies `transform` to every string in a JSON tree.
/// `transform` returns `Some(new)` when the string changed.
fn walk_sanitize<F>(node: &mut Value, found: &mut bool, transform: &F)
where
    F: Fn(&str) -> Option<String>,
{
    match node {
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() {
                if let Value::String(s) = v {
                    if let Some(new) = transform(s) {
                        *s = new;
                        *found = true;
                    }
                } else if v.is_object() || v.is_array() {
                    walk_sanitize(v, found, transform);
                }
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                if let Value::String(s) = v {
                    if let Some(new) = transform(s) {
                        *s = new;
                        *found = true;
                    }
                } else if v.is_object() || v.is_array() {
                    walk_sanitize(v, found, transform);
                }
            }
        }
        _ => {}
    }
}

/// Sanitize surrogate characters from all string content in a messages list.
/// Operates in-place on the JSON message objects; returns `true` if any were
/// found and replaced.
pub fn sanitize_messages_surrogates(messages: &mut [Value]) -> bool {
    sanitize_messages_with(messages, &|s| {
        if surrogate_re().is_match(s) {
            Some(surrogate_re().replace_all(s, "\u{fffd}").into_owned())
        } else {
            None
        }
    })
}

/// Shared message-walker for both surrogate and non-ASCII sanitisation.
///
/// Mirrors the field coverage of `_sanitize_messages_surrogates` /
/// `_sanitize_messages_non_ascii`: content (str or list of text parts), name,
/// tool_calls (id, function.name, function.arguments), and any other
/// top-level string / nested structured field.
fn sanitize_messages_with<F>(messages: &mut [Value], transform: &F) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    let mut found = false;
    for msg in messages.iter_mut() {
        let obj = match msg.as_object_mut() {
            Some(o) => o,
            None => continue,
        };

        // content
        if let Some(content) = obj.get_mut("content") {
            match content {
                Value::String(s) => {
                    if let Some(new) = transform(s) {
                        *s = new;
                        found = true;
                    }
                }
                Value::Array(parts) => {
                    for part in parts.iter_mut() {
                        if let Some(po) = part.as_object_mut() {
                            if let Some(Value::String(text)) = po.get_mut("text") {
                                if let Some(new) = transform(text) {
                                    *text = new;
                                    found = true;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // name
        if let Some(Value::String(name)) = obj.get_mut("name") {
            if let Some(new) = transform(name) {
                *name = new;
                found = true;
            }
        }

        // tool_calls
        if let Some(Value::Array(tool_calls)) = obj.get_mut("tool_calls") {
            for tc in tool_calls.iter_mut() {
                if let Some(tco) = tc.as_object_mut() {
                    if let Some(Value::String(id)) = tco.get_mut("id") {
                        if let Some(new) = transform(id) {
                            *id = new;
                            found = true;
                        }
                    }
                    if let Some(Value::Object(fn_obj)) = tco.get_mut("function") {
                        if let Some(Value::String(fname)) = fn_obj.get_mut("name") {
                            if let Some(new) = transform(fname) {
                                *fname = new;
                                found = true;
                            }
                        }
                        if let Some(Value::String(fargs)) = fn_obj.get_mut("arguments") {
                            if let Some(new) = transform(fargs) {
                                *fargs = new;
                                found = true;
                            }
                        }
                    }
                }
            }
        }

        // any additional top-level string / nested structured field
        let keys: Vec<String> = obj
            .keys()
            .filter(|k| !matches!(k.as_str(), "content" | "name" | "tool_calls" | "role"))
            .cloned()
            .collect();
        for key in keys {
            if let Some(value) = obj.get_mut(&key) {
                match value {
                    Value::String(s) => {
                        if let Some(new) = transform(s) {
                            *s = new;
                            found = true;
                        }
                    }
                    Value::Object(_) | Value::Array(_) => {
                        let mut sub_found = false;
                        walk_sanitize(value, &mut sub_found, transform);
                        if sub_found {
                            found = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Non-ASCII sanitisation
// ---------------------------------------------------------------------------

/// Remove non-ASCII characters (Python `text.encode('ascii', 'ignore')`).
pub fn strip_non_ascii(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii()).collect()
}

/// Strip non-ASCII characters from all string content in a messages list.
/// Returns `true` if any non-ASCII content was found and sanitized.
pub fn sanitize_messages_non_ascii(messages: &mut [Value]) -> bool {
    sanitize_messages_with(messages, &|s| {
        let stripped = strip_non_ascii(s);
        if stripped != *s {
            Some(stripped)
        } else {
            None
        }
    })
}

/// Strip non-ASCII characters from nested dict/list payloads in-place.
pub fn sanitize_structure_non_ascii(payload: &mut Value) -> bool {
    let mut found = false;
    walk_sanitize(payload, &mut found, &|s| {
        let stripped = strip_non_ascii(s);
        if stripped != *s {
            Some(stripped)
        } else {
            None
        }
    });
    found
}

/// Strip non-ASCII characters from tool payloads in-place.
pub fn sanitize_tools_non_ascii(tools: &mut Value) -> bool {
    sanitize_structure_non_ascii(tools)
}

// ---------------------------------------------------------------------------
// Tool-call argument JSON repair
// ---------------------------------------------------------------------------

/// Escape unescaped control chars inside JSON string values.
///
/// Walks the raw JSON character-by-character, tracking whether we are inside a
/// double-quoted string. Inside strings, replaces literal control characters
/// (0x00-0x1F) not already part of an escape sequence with `\uXXXX`.
pub fn escape_invalid_chars_in_json_strings(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        if in_string {
            if ch == '\\' && i + 1 < n {
                out.push(ch);
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if ch == '"' {
                in_string = false;
                out.push(ch);
            } else if (ch as u32) < 0x20 {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            } else {
                out.push(ch);
            }
        } else {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
        }
        i += 1;
    }
    out
}

/// Attempt to repair malformed tool_call argument JSON.
///
/// Faithful port of `_repair_tool_call_arguments`. Returns the repaired JSON
/// string, or `"{}"` as a last resort so the request never crashes the session.
pub fn repair_tool_call_arguments(raw_args: &str, _tool_name: &str) -> String {
    let raw_stripped = raw_args.trim();

    // Fast-path: empty / whitespace-only -> empty object
    if raw_stripped.is_empty() {
        return "{}".to_string();
    }

    // Python-literal None -> normalise to {}
    if raw_stripped == "None" {
        return "{}".to_string();
    }

    // Repair pass 0: parse with lenient settings then re-serialise compactly.
    // serde_json is already lenient about whitespace; it rejects literal
    // control chars, matching neither Python's strict=False exactly nor
    // strict=True — we emulate strict=False by first escaping control chars
    // only for this probe so a successful parse re-serialises like Python.
    if let Ok(parsed) = serde_json::from_str::<Value>(raw_stripped) {
        return compact_json(&parsed);
    }
    // strict=False equivalent: tolerate literal control chars in strings.
    let pass0_escaped = escape_invalid_chars_in_json_strings(raw_stripped);
    if pass0_escaped != raw_stripped {
        if let Ok(parsed) = serde_json::from_str::<Value>(&pass0_escaped) {
            return compact_json(&parsed);
        }
    }

    // Attempt common JSON repairs.
    let mut fixed = raw_stripped.to_string();
    // 1. Strip trailing commas before } or ]
    let trailing_comma = Regex::new(r",\s*([}\]])").unwrap();
    fixed = trailing_comma.replace_all(&fixed, "$1").into_owned();
    // 2. Close unclosed structures
    let open_curly = count_char(&fixed, '{') as i64 - count_char(&fixed, '}') as i64;
    let open_bracket = count_char(&fixed, '[') as i64 - count_char(&fixed, ']') as i64;
    if open_curly > 0 {
        fixed.push_str(&"}".repeat(open_curly as usize));
    }
    if open_bracket > 0 {
        fixed.push_str(&"]".repeat(open_bracket as usize));
    }
    // 3. Remove excess closing braces/brackets (bounded to 50 iterations)
    for _ in 0..50 {
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            break;
        }
        if fixed.ends_with('}') && count_char(&fixed, '}') > count_char(&fixed, '{') {
            fixed.pop();
        } else if fixed.ends_with(']') && count_char(&fixed, ']') > count_char(&fixed, '[') {
            fixed.pop();
        } else {
            break;
        }
    }

    if serde_json::from_str::<Value>(&fixed).is_ok() {
        return fixed;
    }

    // Repair pass 4: escape unescaped control chars inside JSON strings.
    let escaped = escape_invalid_chars_in_json_strings(&fixed);
    if escaped != fixed && serde_json::from_str::<Value>(&escaped).is_ok() {
        return escaped;
    }

    // Last resort.
    "{}".to_string()
}

/// Serialise compactly with `,`/`:` separators, matching Python
/// `json.dumps(parsed, separators=(",", ":"))`.
fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|x| *x == c).count()
}

// ---------------------------------------------------------------------------
// Provider headers & pool recovery
// ---------------------------------------------------------------------------

/// Return the User-Agent RouterMint needs to avoid Cloudflare 1010 blocks.
///
/// `hermes_version` is the value of `hermes_cli.__version__`.
pub fn routermint_headers(hermes_version: &str) -> HashMap<String, String> {
    let mut h = HashMap::new();
    h.insert("User-Agent".to_string(), format!("HermesAgent/{hermes_version}"));
    h
}

/// Return default HTTP headers required by Qwen Portal API.
///
/// `system_lower` is `platform.system().lower()` and `machine` is
/// `platform.machine()`.
pub fn qwen_portal_headers(system_lower: &str, machine: &str) -> HashMap<String, String> {
    let ua = format!("QwenCode/{QWEN_CODE_VERSION} ({system_lower}; {machine})");
    let mut h = HashMap::new();
    h.insert("User-Agent".to_string(), ua.clone());
    h.insert("X-DashScope-CacheControl".to_string(), "enable".to_string());
    h.insert("X-DashScope-UserAgent".to_string(), ua);
    h.insert("X-DashScope-AuthType".to_string(), "qwen-oauth".to_string());
    h
}

/// Minimal view of a credential pool for the rate-limit recovery decision.
pub trait PoolView {
    fn has_available(&self) -> bool;
    fn entries_len(&self) -> usize;
}

/// Decide whether to wait for credential-pool rotation instead of falling back.
///
/// Faithful port of `_pool_may_recover_from_rate_limit`. Returns `true` only
/// when rotation has somewhere to go.
pub fn pool_may_recover_from_rate_limit<P: PoolView>(
    pool: Option<&P>,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> bool {
    let pool = match pool {
        Some(p) => p,
        None => return false,
    };
    if !pool.has_available() {
        return false;
    }
    // CloudCode / Gemini CLI quotas are account-wide — rotation can't recover.
    if provider == Some("google-gemini-cli")
        || base_url.unwrap_or("").starts_with("cloudcode-pa://")
    {
        return false;
    }
    pool.entries_len() > 1
}

// ---------------------------------------------------------------------------
// AIAgent pure / static method ports
// ---------------------------------------------------------------------------

/// Return `true` for models that require the Responses API path
/// (`_model_requires_responses_api`). GPT-5.x models are rejected on chat
/// completions by both OpenAI and OpenRouter.
pub fn model_requires_responses_api(model: &str) -> bool {
    let mut m = model.to_lowercase();
    if let Some(idx) = m.rfind('/') {
        m = m[idx + 1..].to_string();
    }
    m.starts_with("gpt-5")
}

/// Return the correct max-tokens kwarg key for the current provider.
///
/// `is_direct_openai` / `is_azure_openai` correspond to the two predicate
/// methods on `AIAgent`. Returns `("max_completion_tokens", value)` for those,
/// otherwise `("max_tokens", value)`.
pub fn max_tokens_param(value: i64, is_direct_openai: bool, is_azure_openai: bool) -> (String, i64) {
    if is_direct_openai || is_azure_openai {
        ("max_completion_tokens".to_string(), value)
    } else {
        ("max_tokens".to_string(), value)
    }
}

/// Heuristic: does visible assistant text look intentionally finished?
/// (`_has_natural_response_ending`)
pub fn has_natural_response_ending(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    let stripped = content.trim_end();
    if stripped.is_empty() {
        return false;
    }
    if stripped.ends_with("```") {
        return true;
    }
    // Last char in the terminal-punctuation set.
    let last = stripped.chars().last().unwrap();
    const TERMINALS: &str = ".!?:)\"']}。！？：）】」』》";
    TERMINALS.chars().any(|c| c == last)
}

/// Remove reasoning/thinking blocks from content, returning only visible text.
///
/// Faithful port of `_strip_think_blocks`, preserving the exact regex passes
/// and ordering.
pub fn strip_think_blocks(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut content = content.to_string();

    // 1. Closed tag pairs (case-insensitive, dotall).
    for tag in ["think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought"] {
        let re = Regex::new(&format!(r"(?is)<{tag}>.*?</{tag}>")).unwrap();
        content = re.replace_all(&content, "").into_owned();
    }

    // 1b. Tool-call XML blocks.
    for tc_name in [
        "tool_call",
        "tool_calls",
        "tool_result",
        "function_call",
        "function_calls",
    ] {
        let re = Regex::new(&format!(r"(?is)<{tc_name}\b[^>]*>.*?</{tc_name}>")).unwrap();
        content = re.replace_all(&content, "").into_owned();
    }

    // 1c. <function name="...">...</function> at a block boundary.
    // Rust's regex crate lacks lookbehind, so emulate the (?:(?<=^)|(?<=[\n\r.!?:]))
    // prefix with a captured leading boundary char that we re-insert.
    {
        let re = Regex::new(
            r#"(?is)(^|[\n\r.!?:])[ \t]*<function\b[^>]*\bname\s*=[^>]*>(?:(?:(?:[^<]|<(?:[^/]|/[^f]|/f[^u]))*?))</function>"#,
        );
        // The negative-lookahead `(?!</function>)` from Python is approximated;
        // fall back to a simpler non-greedy match if the elaborate pattern
        // fails to compile.
        let re = re.unwrap_or_else(|_| {
            Regex::new(r"(?is)(^|[\n\r.!?:])[ \t]*<function\b[^>]*\bname\s*=[^>]*>.*?</function>")
                .unwrap()
        });
        content = re.replace_all(&content, "$1").into_owned();
    }

    // 2. Unterminated reasoning block — open tag at a block boundary.
    {
        let re = Regex::new(
            r"(?is)(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$",
        )
        .unwrap();
        content = re.replace_all(&content, "").into_owned();
    }

    // 3. Stray orphan open/close tags.
    {
        let re = Regex::new(
            r"(?i)</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*",
        )
        .unwrap();
        content = re.replace_all(&content, "").into_owned();
    }

    // 3b. Stray tool-call closers.
    {
        let re = Regex::new(
            r"(?i)</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*",
        )
        .unwrap();
        content = re.replace_all(&content, "").into_owned();
    }

    content
}

/// Check if content has actual text after any reasoning/thinking blocks.
/// (`_has_content_after_think_block`)
pub fn has_content_after_think_block(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    !strip_think_blocks(content).trim().is_empty()
}

/// Assistant message view for reasoning extraction. Mirrors the duck-typed
/// attributes the Python code reads off the SDK message object.
#[derive(Debug, Default, Clone)]
pub struct AssistantMessageView {
    pub reasoning: Option<String>,
    pub reasoning_content: Option<String>,
    /// `reasoning_details` array (OpenRouter unified format).
    pub reasoning_details: Vec<Value>,
    pub content: Option<String>,
}

/// Extract reasoning/thinking content from an assistant message.
/// Faithful port of `_extract_reasoning`.
pub fn extract_reasoning(msg: &AssistantMessageView) -> Option<String> {
    let mut reasoning_parts: Vec<String> = Vec::new();

    if let Some(r) = &msg.reasoning {
        if !r.is_empty() {
            reasoning_parts.push(r.clone());
        }
    }

    if let Some(rc) = &msg.reasoning_content {
        if !rc.is_empty() && !reasoning_parts.contains(rc) {
            reasoning_parts.push(rc.clone());
        }
    }

    for detail in &msg.reasoning_details {
        if let Some(obj) = detail.as_object() {
            let summary = obj
                .get("summary")
                .or_else(|| obj.get("thinking"))
                .or_else(|| obj.get("content"))
                .or_else(|| obj.get("text"))
                .and_then(|v| v.as_str());
            if let Some(summary) = summary {
                if !summary.is_empty() && !reasoning_parts.iter().any(|p| p == summary) {
                    reasoning_parts.push(summary.to_string());
                }
            }
        }
    }

    // Inline fallback only when no structured reasoning was found.
    if reasoning_parts.is_empty() {
        if let Some(content) = &msg.content {
            if !content.is_empty() {
                let inline_patterns = [
                    r"(?is)<think>(.*?)</think>",
                    r"(?is)<thinking>(.*?)</thinking>",
                    r"(?is)<thought>(.*?)</thought>",
                    r"(?is)<reasoning>(.*?)</reasoning>",
                    r"(?is)<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
                ];
                for pattern in inline_patterns {
                    let re = Regex::new(pattern).unwrap();
                    for cap in re.captures_iter(content) {
                        if let Some(block) = cap.get(1) {
                            let cleaned = block.as_str().trim();
                            if !cleaned.is_empty()
                                && !reasoning_parts.iter().any(|p| p == cleaned)
                            {
                                reasoning_parts.push(cleaned.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n\n"))
    }
}

/// Detect a planning/ack message that should continue instead of ending the
/// turn. Faithful port of `_looks_like_codex_intermediate_ack`.
pub fn looks_like_codex_intermediate_ack(
    user_message: &str,
    assistant_content: &str,
    messages: &[Value],
) -> bool {
    // Any tool message present? Then not an intermediate ack.
    if messages.iter().any(|m| {
        m.as_object()
            .and_then(|o| o.get("role"))
            .and_then(|r| r.as_str())
            == Some("tool")
    }) {
        return false;
    }

    let assistant_text = strip_think_blocks(assistant_content).trim().to_lowercase();
    if assistant_text.is_empty() {
        return false;
    }
    if assistant_text.chars().count() > 1200 {
        return false;
    }

    let future_ack = Regex::new(r"(?i)\b(i'll|i’ll|i will|let me|i can do that|i can help with that)\b")
        .unwrap();
    if !future_ack.is_match(&assistant_text) {
        return false;
    }

    const ACTION_MARKERS: &[&str] = &[
        "look into", "look at", "inspect", "scan", "check", "analyz", "review", "explore",
        "read", "open", "run", "test", "fix", "debug", "search", "find", "walkthrough",
        "report back", "summarize",
    ];
    const WORKSPACE_MARKERS: &[&str] = &[
        "directory", "current directory", "current dir", "cwd", "repo", "repository",
        "codebase", "project", "folder", "filesystem", "file tree", "files", "path",
    ];

    let user_text = user_message.trim().to_lowercase();
    let user_targets_workspace = WORKSPACE_MARKERS.iter().any(|m| user_text.contains(m))
        || user_text.contains("~/")
        || user_text.contains('/');
    let assistant_mentions_action = ACTION_MARKERS.iter().any(|m| assistant_text.contains(m));
    let assistant_targets_workspace = WORKSPACE_MARKERS.iter().any(|m| assistant_text.contains(m));

    (user_targets_workspace || assistant_targets_workspace) && assistant_mentions_action
}

/// Mask an API key for logs. (`_mask_api_key_for_logs`)
pub fn mask_api_key_for_logs(key: Option<&str>) -> Option<String> {
    let key = key?;
    if key.is_empty() {
        return None;
    }
    if key.chars().count() <= 12 {
        return Some("***".to_string());
    }
    let chars: Vec<char> = key.chars().collect();
    let prefix: String = chars[..8].iter().collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    Some(format!("{prefix}...{suffix}"))
}

/// Clean up error messages for user display. (`_clean_error_message`)
pub fn clean_error_message(error_msg: &str) -> String {
    if error_msg.is_empty() {
        return "Unknown error".to_string();
    }
    if error_msg.trim_start().starts_with("<!DOCTYPE html") || error_msg.contains("<html") {
        return "Service temporarily unavailable (HTML error page returned)".to_string();
    }
    // Collapse whitespace (Python ' '.join(error_msg.split())).
    let cleaned = error_msg.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() > 150 {
        let truncated: String = cleaned.chars().take(150).collect();
        format!("{truncated}...")
    } else {
        cleaned
    }
}

/// A structured view of an API error for [`summarize_api_error`] and
/// [`extract_api_error_context`]. Mirrors the duck-typed attributes the Python
/// code reads from SDK exception objects.
#[derive(Debug, Default, Clone)]
pub struct ApiErrorView {
    /// `str(error)`.
    pub raw: String,
    /// `error.status_code`.
    pub status_code: Option<i64>,
    /// `error.body` (the parsed JSON body, if a dict).
    pub body: Option<Value>,
    /// `error.response.headers` flattened to a map (lowercased keys preserved
    /// as the caller stored them; both casings can be present).
    pub response_headers: HashMap<String, String>,
}

/// Extract a human-readable one-liner from an API error.
/// Faithful port of `_summarize_api_error`.
pub fn summarize_api_error(error: &ApiErrorView) -> String {
    let raw = &error.raw;

    if raw.contains("<!DOCTYPE") || raw.contains("<html") {
        let title_re = Regex::new(r"(?i)<title[^>]*>([^<]+)</title>").unwrap();
        let title = title_re
            .captures(raw)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| "HTML error page (title not found)".to_string());
        let ray_re = Regex::new(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>").unwrap();
        let ray_id = ray_re
            .captures(raw)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string());
        let mut parts: Vec<String> = Vec::new();
        if let Some(code) = error.status_code {
            parts.push(format!("HTTP {code}"));
        }
        parts.push(title);
        if let Some(ray_id) = ray_id {
            parts.push(format!("Ray {ray_id}"));
        }
        return parts.join(" — ");
    }

    // JSON body errors from OpenAI/Anthropic SDKs.
    if let Some(Value::Object(body)) = &error.body {
        let msg = match body.get("error") {
            Some(Value::Object(err_obj)) => err_obj.get("message").and_then(|v| v.as_str()),
            _ => body.get("message").and_then(|v| v.as_str()),
        };
        if let Some(msg) = msg {
            let prefix = error
                .status_code
                .map(|c| format!("HTTP {c}: "))
                .unwrap_or_default();
            let truncated: String = msg.chars().take(300).collect();
            return format!("{prefix}{truncated}");
        }
    }

    let prefix = error
        .status_code
        .map(|c| format!("HTTP {c}: "))
        .unwrap_or_default();
    let truncated: String = raw.chars().take(500).collect();
    format!("{prefix}{truncated}")
}

/// Extract structured rate-limit details from provider errors.
/// Faithful port of `_extract_api_error_context`. `now` is `time.time()`.
pub fn extract_api_error_context(error: &ApiErrorView, now: f64) -> HashMap<String, Value> {
    let mut context: HashMap<String, Value> = HashMap::new();

    let payload: Option<&Value> = match &error.body {
        Some(Value::Object(body)) => match body.get("error") {
            Some(v @ Value::Object(_)) => Some(v),
            _ => error.body.as_ref(),
        },
        _ => None,
    };

    if let Some(Value::Object(payload)) = payload {
        let reason = payload
            .get("code")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("error").and_then(|v| v.as_str()));
        if let Some(reason) = reason {
            if !reason.trim().is_empty() {
                context.insert("reason".to_string(), Value::String(reason.trim().to_string()));
            }
        }
        let message = payload
            .get("message")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("error_description").and_then(|v| v.as_str()));
        if let Some(message) = message {
            if !message.trim().is_empty() {
                context.insert("message".to_string(), Value::String(message.trim().to_string()));
            }
        }
        for key in ["resets_at", "reset_at"] {
            if let Some(value) = payload.get(key) {
                if !value.is_null() && value.as_str() != Some("") {
                    context.insert("reset_at".to_string(), value.clone());
                    break;
                }
            }
        }
        if !context.contains_key("reset_at") {
            if let Some(retry_after) = payload.get("retry_after") {
                if !retry_after.is_null() && retry_after.as_str() != Some("") {
                    if let Some(secs) = value_as_f64(retry_after) {
                        context.insert(
                            "reset_at".to_string(),
                            Value::from(now + secs),
                        );
                    }
                }
            }
        }
    }

    if !error.response_headers.is_empty() && !context.contains_key("reset_at") {
        let retry_after = error
            .response_headers
            .get("retry-after")
            .or_else(|| error.response_headers.get("Retry-After"));
        if let Some(retry_after) = retry_after {
            if let Ok(secs) = retry_after.parse::<f64>() {
                context.insert("reset_at".to_string(), Value::from(now + secs));
            }
        }
        if !context.contains_key("reset_at") {
            if let Some(reset) = error.response_headers.get("x-ratelimit-reset") {
                context.insert("reset_at".to_string(), Value::String(reset.clone()));
            }
        }
    }

    if !context.contains_key("message") {
        let raw_message = error.raw.trim();
        if !raw_message.is_empty() {
            let truncated: String = raw_message.chars().take(500).collect();
            context.insert("message".to_string(), Value::String(truncated));
        }
    }

    if !context.contains_key("reset_at") {
        let message = context
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // Note: the Python source has an escaped \\d in the quotaResetDelay
        // pattern which never matches a real digit; we faithfully reproduce
        // the *effective* behaviour — only the retry-seconds pattern below can
        // match — by leaving the quota pattern non-matching.
        let sec_re = Regex::new(
            r"(?i)retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
        )
        .unwrap();
        if let Some(cap) = sec_re.captures(&message) {
            if let Ok(secs) = cap[1].parse::<f64>() {
                context.insert("reset_at".to_string(), Value::from(now + secs));
            }
        }
    }

    context
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tool-call id/name extraction & message sanitisation
// ---------------------------------------------------------------------------

/// Extract a call ID from a tool_call JSON entry.
/// (`_get_tool_call_id_static`)
pub fn get_tool_call_id_static(tc: &Value) -> String {
    if let Some(obj) = tc.as_object() {
        if let Some(s) = obj.get("call_id").and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return s.to_string();
            }
        }
        if let Some(s) = obj.get("id").and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    String::new()
}

/// Extract function name from a tool_call JSON entry.
/// (`_get_tool_call_name_static`)
pub fn get_tool_call_name_static(tc: &Value) -> String {
    if let Some(obj) = tc.as_object() {
        if let Some(Value::Object(fn_obj)) = obj.get("function") {
            return fn_obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
    }
    String::new()
}

/// Fix orphaned tool_call / tool_result pairs before every LLM call.
/// Faithful port of `_sanitize_api_messages`. Takes/returns owned message JSON.
pub fn sanitize_api_messages(messages: Vec<Value>) -> Vec<Value> {
    // Role allowlist.
    let mut filtered: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg.as_object().and_then(|o| o.get("role")).and_then(|r| r.as_str());
        if let Some(role) = role {
            if VALID_API_ROLES.contains(&role) {
                filtered.push(msg);
            }
        }
    }
    let mut messages = filtered;

    let mut surviving_call_ids: HashSet<String> = HashSet::new();
    for msg in &messages {
        if msg_role(msg) == Some("assistant") {
            if let Some(Value::Array(tcs)) = msg.get("tool_calls") {
                for tc in tcs {
                    let cid = get_tool_call_id_static(tc);
                    if !cid.is_empty() {
                        surviving_call_ids.insert(cid);
                    }
                }
            }
        }
    }

    let mut result_call_ids: HashSet<String> = HashSet::new();
    for msg in &messages {
        if msg_role(msg) == Some("tool") {
            if let Some(cid) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                if !cid.is_empty() {
                    result_call_ids.insert(cid.to_string());
                }
            }
        }
    }

    // 1. Drop tool results with no matching assistant call.
    let orphaned: HashSet<String> =
        result_call_ids.difference(&surviving_call_ids).cloned().collect();
    if !orphaned.is_empty() {
        messages.retain(|m| {
            !(msg_role(m) == Some("tool")
                && m.get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .map(|c| orphaned.contains(c))
                    .unwrap_or(false))
        });
    }

    // 2. Inject stub results for calls whose result was dropped.
    let missing: HashSet<String> =
        surviving_call_ids.difference(&result_call_ids).cloned().collect();
    if !missing.is_empty() {
        let mut patched: Vec<Value> = Vec::with_capacity(messages.len());
        for msg in messages {
            let is_assistant = msg_role(&msg) == Some("assistant");
            let tool_calls = if is_assistant {
                msg.get("tool_calls").and_then(|v| v.as_array()).cloned()
            } else {
                None
            };
            patched.push(msg);
            if let Some(tcs) = tool_calls {
                for tc in &tcs {
                    let cid = get_tool_call_id_static(tc);
                    if missing.contains(&cid) {
                        let mut stub = serde_json::Map::new();
                        stub.insert("role".to_string(), Value::String("tool".to_string()));
                        stub.insert(
                            "name".to_string(),
                            Value::String(get_tool_call_name_static(tc)),
                        );
                        stub.insert(
                            "content".to_string(),
                            Value::String(
                                "[Result unavailable — see context summary above]".to_string(),
                            ),
                        );
                        stub.insert("tool_call_id".to_string(), Value::String(cid));
                        patched.push(Value::Object(stub));
                    }
                }
            }
        }
        messages = patched;
    }

    messages
}

fn msg_role(msg: &Value) -> Option<&str> {
    msg.as_object().and_then(|o| o.get("role")).and_then(|r| r.as_str())
}

/// Return `true` if `msg` is an assistant turn whose only payload is reasoning.
/// Faithful port of `_is_thinking_only_assistant`.
pub fn is_thinking_only_assistant(msg: &Value) -> bool {
    let obj = match msg.as_object() {
        Some(o) => o,
        None => return false,
    };
    if obj.get("role").and_then(|r| r.as_str()) != Some("assistant") {
        return false;
    }
    // Has tool_calls (truthy)?
    if let Some(tc) = obj.get("tool_calls") {
        let truthy = match tc {
            Value::Array(a) => !a.is_empty(),
            Value::Null => false,
            Value::Bool(b) => *b,
            _ => true,
        };
        if truthy {
            return false;
        }
    }

    // Any actual output?
    match obj.get("content") {
        Some(Value::String(s)) => {
            if !s.trim().is_empty() {
                return false;
            }
        }
        Some(Value::Array(blocks)) => {
            for block in blocks {
                match block.as_object() {
                    None => {
                        // non-dict; truthy non-dict counts as payload
                        if json_truthy(block) {
                            return false;
                        }
                    }
                    Some(bo) => {
                        let btype = bo.get("type").and_then(|v| v.as_str());
                        match btype {
                            Some("thinking") | Some("redacted_thinking") => continue,
                            Some("text") => {
                                let text =
                                    bo.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                if !text.trim().is_empty() {
                                    return false;
                                }
                                continue;
                            }
                            _ => return false,
                        }
                    }
                }
            }
        }
        Some(Value::Null) | None => {}
        Some(other) => {
            // content is not None and not "" → real payload
            if other.as_str() != Some("") {
                return false;
            }
        }
    }

    // Empty-ish content. Reasoning present?
    let reasoning = obj
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| obj.get("reasoning").and_then(|v| v.as_str()).filter(|s| !s.is_empty()));
    if let Some(r) = reasoning {
        if !r.trim().is_empty() {
            return true;
        }
    }
    if let Some(Value::Array(rd)) = obj.get("reasoning_details") {
        if !rd.is_empty() {
            return true;
        }
    }
    false
}

fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Drop thinking-only assistant turns; merge adjacent user messages left
/// behind. Faithful port of `_drop_thinking_only_and_merge_users`.
pub fn drop_thinking_only_and_merge_users(messages: Vec<Value>) -> Vec<Value> {
    if messages.is_empty() {
        return messages;
    }

    let original_len = messages.len();
    let kept: Vec<Value> = messages
        .iter()
        .filter(|m| !is_thinking_only_assistant(m))
        .cloned()
        .collect();
    let dropped = original_len - kept.len();
    if dropped == 0 {
        return messages;
    }

    let mut merged: Vec<Value> = Vec::with_capacity(kept.len());
    for m in kept {
        let prev_is_user = merged
            .last()
            .map(|p| msg_role(p) == Some("user"))
            .unwrap_or(false);
        let cur_is_user = msg_role(&m) == Some("user");

        if prev_is_user && cur_is_user {
            let prev = merged.last().unwrap().clone();
            let mut prev_copy = prev.as_object().unwrap().clone();
            let prev_content = prev.get("content").cloned().unwrap_or(Value::String(String::new()));
            let cur_content = m.get("content").cloned().unwrap_or(Value::String(String::new()));

            let new_content: Option<Value> = match (&prev_content, &cur_content) {
                (Value::String(p), Value::String(c)) => {
                    let sep = if !p.is_empty() && !c.is_empty() { "\n\n" } else { "" };
                    Some(Value::String(format!("{p}{sep}{c}")))
                }
                (Value::Array(p), Value::Array(c)) => {
                    let mut v = p.clone();
                    v.extend(c.clone());
                    Some(Value::Array(v))
                }
                (Value::Array(p), Value::String(c)) => {
                    let mut v = p.clone();
                    if !c.is_empty() {
                        let mut text_block = serde_json::Map::new();
                        text_block.insert("type".to_string(), Value::String("text".to_string()));
                        text_block.insert("text".to_string(), Value::String(c.clone()));
                        v.push(Value::Object(text_block));
                    }
                    Some(Value::Array(v))
                }
                (Value::String(p), Value::Array(c)) => {
                    let mut v: Vec<Value> = Vec::new();
                    if !p.is_empty() {
                        let mut text_block = serde_json::Map::new();
                        text_block.insert("type".to_string(), Value::String("text".to_string()));
                        text_block.insert("text".to_string(), Value::String(p.clone()));
                        v.push(Value::Object(text_block));
                    }
                    v.extend(c.clone());
                    Some(Value::Array(v))
                }
                _ => None,
            };

            match new_content {
                Some(content) => {
                    prev_copy.insert("content".to_string(), content);
                    *merged.last_mut().unwrap() = Value::Object(prev_copy);
                }
                None => {
                    // Unknown content shape — append separately.
                    merged.push(m);
                }
            }
        } else {
            merged.push(m);
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// Tool-call list operations
// ---------------------------------------------------------------------------

/// Truncate excess `delegate_task` calls to `max_children`.
/// Faithful port of `_cap_delegate_task_calls`.
pub fn cap_delegate_task_calls(tool_calls: Vec<ToolCall>, max_children: usize) -> Vec<ToolCall> {
    let delegate_count = tool_calls
        .iter()
        .filter(|tc| tc.function.name == "delegate_task")
        .count();
    if delegate_count <= max_children {
        return tool_calls;
    }
    let mut kept_delegates = 0usize;
    let mut truncated: Vec<ToolCall> = Vec::new();
    for tc in tool_calls {
        if tc.function.name == "delegate_task" {
            if kept_delegates < max_children {
                truncated.push(tc);
                kept_delegates += 1;
            }
        } else {
            truncated.push(tc);
        }
    }
    truncated
}

/// Remove duplicate (tool_name, arguments) pairs within a single turn.
/// Returns the original list if no duplicates were found.
/// Faithful port of `_deduplicate_tool_calls`.
pub fn deduplicate_tool_calls(tool_calls: Vec<ToolCall>) -> Vec<ToolCall> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut unique: Vec<ToolCall> = Vec::new();
    for tc in &tool_calls {
        let key = (tc.function.name.clone(), tc.function.arguments.clone());
        if seen.insert(key) {
            unique.push(tc.clone());
        }
    }
    if unique.len() < tool_calls.len() {
        unique
    } else {
        tool_calls
    }
}

/// Attempt to repair a mismatched tool name before aborting.
/// Faithful port of `_repair_tool_call`. `valid_tool_names` is the set of
/// known tool names.
pub fn repair_tool_call(tool_name: &str, valid_tool_names: &HashSet<String>) -> Option<String> {
    if tool_name.is_empty() {
        return None;
    }

    fn norm(s: &str) -> String {
        s.to_lowercase().replace('-', "_").replace(' ', "_")
    }
    fn camel_snake(s: &str) -> String {
        // re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if i > 0 && c.is_ascii_uppercase() {
                out.push('_');
            }
            out.push(c);
        }
        out.to_lowercase()
    }
    fn strip_tool_suffix(s: &str) -> Option<String> {
        let lc = s.to_lowercase();
        for suffix in ["_tool", "-tool", "tool"] {
            if lc.ends_with(suffix) {
                let cut = s.len() - suffix.len();
                let trimmed = s[..cut].trim_end_matches(['_', '-']);
                return Some(trimmed.to_string());
            }
        }
        None
    }

    let lowered = tool_name.to_lowercase();
    if valid_tool_names.contains(&lowered) {
        return Some(lowered);
    }
    let normalized = norm(tool_name);
    if valid_tool_names.contains(&normalized) {
        return Some(normalized);
    }

    let mut cands: HashSet<String> = HashSet::new();
    cands.insert(tool_name.to_string());
    cands.insert(lowered.clone());
    cands.insert(normalized.clone());
    cands.insert(camel_snake(tool_name));

    for _ in 0..2 {
        let mut extra: HashSet<String> = HashSet::new();
        for c in &cands {
            if let Some(stripped) = strip_tool_suffix(c) {
                extra.insert(stripped.clone());
                extra.insert(norm(&stripped));
                extra.insert(camel_snake(&stripped));
            }
        }
        cands.extend(extra);
    }

    for c in &cands {
        if !c.is_empty() && valid_tool_names.contains(c) {
            return Some(c.clone());
        }
    }

    // Fuzzy match (difflib get_close_matches, cutoff=0.7).
    fuzzy_close_match(&lowered, valid_tool_names, 0.7)
}

/// Approximate Python `difflib.get_close_matches(n=1, cutoff)` using the
/// difflib ratio (`2*M / T`). Returns the single best match at or above
/// `cutoff`.
fn fuzzy_close_match(word: &str, candidates: &HashSet<String>, cutoff: f64) -> Option<String> {
    let mut best: Option<(f64, String)> = None;
    for cand in candidates {
        let ratio = difflib_ratio(word, cand);
        if ratio >= cutoff {
            match &best {
                Some((b, _)) if *b >= ratio => {}
                _ => best = Some((ratio, cand.clone())),
            }
        }
    }
    best.map(|(_, c)| c)
}

/// difflib.SequenceMatcher.ratio(): 2*matches / (len(a)+len(b)).
/// Uses the longest-common-subsequence-style matching blocks via a simple
/// recursive longest-matching-block algorithm (good enough for short tool
/// names; matches difflib for the common cases).
fn difflib_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let matches = matching_blocks_count(&a, &b);
    2.0 * matches as f64 / total as f64
}

/// Count of matching characters via difflib's recursive longest-match approach.
fn matching_blocks_count(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    // Find longest matching block.
    let (mut best_i, mut best_j, mut best_size) = (0usize, 0usize, 0usize);
    // j2len maps j -> length of match ending at b[j] for the previous i.
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, &ca) in a.iter().enumerate() {
        let mut new_j2len: HashMap<usize, usize> = HashMap::new();
        for (j, &cb) in b.iter().enumerate() {
            if ca == cb {
                let k = if j > 0 { *j2len.get(&(j - 1)).unwrap_or(&0) + 1 } else { 1 };
                new_j2len.insert(j, k);
                if k > best_size {
                    best_size = k;
                    best_i = i + 1 - k;
                    best_j = j + 1 - k;
                }
            }
        }
        j2len = new_j2len;
    }
    if best_size == 0 {
        return 0;
    }
    let left = matching_blocks_count(&a[..best_i], &b[..best_j]);
    let right = matching_blocks_count(&a[best_i + best_size..], &b[best_j + best_size..]);
    left + best_size + right
}

// ---------------------------------------------------------------------------
// Strict-API tool-call sanitisation
// ---------------------------------------------------------------------------

/// Strip Codex Responses API fields (`call_id`, `response_item_id`) from
/// tool_calls for strict providers. Faithful port of
/// `_sanitize_tool_calls_for_strict_api`. Operates in-place on `api_msg`.
pub fn sanitize_tool_calls_for_strict_api(api_msg: &mut Value) {
    const STRIP_KEYS: &[&str] = &["call_id", "response_item_id"];
    if let Some(obj) = api_msg.as_object_mut() {
        if let Some(Value::Array(tool_calls)) = obj.get_mut("tool_calls") {
            for tc in tool_calls.iter_mut() {
                if let Some(tco) = tc.as_object_mut() {
                    for key in STRIP_KEYS {
                        tco.remove(*key);
                    }
                }
            }
        }
    }
}

/// Repair corrupted assistant tool-call argument JSON in-place.
/// Faithful port of `_sanitize_tool_call_arguments`. Returns the number of
/// repaired tool calls. Mutates `messages` in place (may insert stub tool
/// messages).
pub fn sanitize_tool_call_arguments(messages: &mut Vec<Value>) -> usize {
    let mut repaired = 0usize;
    let marker = TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER;

    let mut message_index = 0usize;
    while message_index < messages.len() {
        // Determine if this is an assistant message with tool calls.
        let is_assistant_with_tools = messages[message_index]
            .as_object()
            .map(|o| {
                o.get("role").and_then(|r| r.as_str()) == Some("assistant")
                    && o.get("tool_calls")
                        .and_then(|v| v.as_array())
                        .map(|a| !a.is_empty())
                        .unwrap_or(false)
            })
            .unwrap_or(false);

        if !is_assistant_with_tools {
            message_index += 1;
            continue;
        }

        // Collect repair work: tool_call index -> (tool_call_id, function_name).
        // We mutate function arguments in place first, then handle stub
        // insertion afterward to avoid borrow conflicts.
        let mut to_patch: Vec<(Option<String>, String)> = Vec::new();

        // Snapshot the number of tool calls.
        let tc_len = messages[message_index]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        for tc_idx in 0..tc_len {
            // Read function arguments.
            let (args_value, tool_call_id, function_name) = {
                let tc = &messages[message_index]["tool_calls"][tc_idx];
                let tco = match tc.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let function = match tco.get("function").and_then(|f| f.as_object()) {
                    Some(f) => f,
                    None => continue,
                };
                let args = function.get("arguments").cloned();
                let tcid = tco
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let fname = function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                (args, tcid, fname)
            };

            // Determine the action.
            enum Action {
                SetEmpty,
                Skip,
                Repair,
            }
            let action = match &args_value {
                None => Action::SetEmpty,
                Some(Value::String(s)) if s.is_empty() => Action::SetEmpty,
                Some(Value::String(s)) if s.trim().is_empty() => Action::SetEmpty,
                Some(Value::String(s)) => {
                    if serde_json::from_str::<Value>(s).is_ok() {
                        Action::Skip
                    } else {
                        Action::Repair
                    }
                }
                Some(_) => Action::Skip, // not a string
            };

            match action {
                Action::SetEmpty => {
                    if let Some(f) = messages[message_index]["tool_calls"][tc_idx]
                        .get_mut("function")
                        .and_then(|f| f.as_object_mut())
                    {
                        f.insert("arguments".to_string(), Value::String("{}".to_string()));
                    }
                }
                Action::Skip => {}
                Action::Repair => {
                    if let Some(f) = messages[message_index]["tool_calls"][tc_idx]
                        .get_mut("function")
                        .and_then(|f| f.as_object_mut())
                    {
                        f.insert("arguments".to_string(), Value::String("{}".to_string()));
                    }
                    to_patch.push((tool_call_id, function_name));
                    repaired += 1;
                }
            }
        }

        // Now handle stub insertion / marker prepend for each repaired call.
        let mut insert_at = message_index + 1;
        for (tool_call_id, function_name) in to_patch {
            // Find an existing tool message with matching id in the immediately
            // following run of tool messages.
            let mut existing_idx: Option<usize> = None;
            let mut scan_index = message_index + 1;
            while scan_index < messages.len() {
                let cand = &messages[scan_index];
                if msg_role(cand) != Some("tool") {
                    break;
                }
                let cand_id = cand.get("tool_call_id").and_then(|v| v.as_str());
                if cand_id.map(|s| s.to_string()) == tool_call_id {
                    existing_idx = Some(scan_index);
                    break;
                }
                scan_index += 1;
            }

            match existing_idx {
                None => {
                    let mut stub = serde_json::Map::new();
                    stub.insert("role".to_string(), Value::String("tool".to_string()));
                    stub.insert(
                        "name".to_string(),
                        Value::String(if function_name != "?" {
                            function_name.clone()
                        } else {
                            String::new()
                        }),
                    );
                    stub.insert(
                        "tool_call_id".to_string(),
                        match &tool_call_id {
                            Some(id) => Value::String(id.clone()),
                            None => Value::Null,
                        },
                    );
                    stub.insert("content".to_string(), Value::String(marker.to_string()));
                    messages.insert(insert_at, Value::Object(stub));
                    insert_at += 1;
                }
                Some(idx) => {
                    prepend_marker(&mut messages[idx], marker);
                }
            }
        }

        message_index += 1;
    }

    repaired
}

/// Prepend the corruption marker to a tool message's content.
/// (`_prepend_marker` inner helper)
fn prepend_marker(tool_msg: &mut Value, marker: &str) {
    let obj = match tool_msg.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    match obj.get("content") {
        Some(Value::String(existing)) => {
            if existing.is_empty() {
                obj.insert("content".to_string(), Value::String(marker.to_string()));
            } else if !existing.starts_with(marker) {
                let combined = format!("{marker}\n{existing}");
                obj.insert("content".to_string(), Value::String(combined));
            }
        }
        None | Some(Value::Null) => {
            obj.insert("content".to_string(), Value::String(marker.to_string()));
        }
        Some(other) => {
            let existing_text = serde_json::to_string(other).unwrap_or_else(|_| other.to_string());
            let combined = format!("{marker}\n{existing_text}");
            obj.insert("content".to_string(), Value::String(combined));
        }
    }
}

// ---------------------------------------------------------------------------
// Background-review action summary
// ---------------------------------------------------------------------------

/// Build the human-facing action summary for a background review pass.
/// Faithful port of `_summarize_background_review_actions`.
pub fn summarize_background_review_actions(
    review_messages: &[Value],
    prior_snapshot: &[Value],
) -> Vec<String> {
    let mut existing_tool_call_ids: HashSet<String> = HashSet::new();
    let mut existing_tool_contents: HashSet<String> = HashSet::new();
    for prior in prior_snapshot {
        if msg_role(prior) != Some("tool") {
            continue;
        }
        match prior.get("tool_call_id").and_then(|v| v.as_str()) {
            Some(tcid) if !tcid.is_empty() => {
                existing_tool_call_ids.insert(tcid.to_string());
            }
            _ => {
                if let Some(content) = prior.get("content").and_then(|v| v.as_str()) {
                    existing_tool_contents.insert(content.to_string());
                }
            }
        }
    }

    let mut actions: Vec<String> = Vec::new();
    for msg in review_messages {
        if msg_role(msg) != Some("tool") {
            continue;
        }
        let tcid = msg.get("tool_call_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
        if let Some(tcid) = tcid {
            if existing_tool_call_ids.contains(tcid) {
                continue;
            }
        } else {
            // No tool_call_id — fall back to content equality.
            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                if existing_tool_contents.contains(content) {
                    continue;
                }
            }
        }

        let content_str = msg.get("content").and_then(|v| v.as_str()).unwrap_or("{}");
        let data: Value = match serde_json::from_str(content_str) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match data.as_object() {
            Some(o) => o,
            None => continue,
        };
        if !obj.get("success").map(json_truthy).unwrap_or(false) {
            continue;
        }
        let message = obj.get("message").and_then(|v| v.as_str()).unwrap_or("");
        let target = obj.get("target").and_then(|v| v.as_str()).unwrap_or("");
        let msg_lower = message.to_lowercase();

        let label = || -> String {
            match target {
                "memory" => "Memory".to_string(),
                "user" => "User profile".to_string(),
                other => other.to_string(),
            }
        };

        if msg_lower.contains("created") {
            actions.push(message.to_string());
        } else if msg_lower.contains("updated") {
            actions.push(message.to_string());
        } else if msg_lower.contains("added") || (!target.is_empty() && msg_lower.contains("add")) {
            actions.push(format!("{} updated", label()));
        } else if message.contains("Entry added") {
            actions.push(format!("{} updated", label()));
        } else if msg_lower.contains("removed") || msg_lower.contains("replaced") {
            actions.push(format!("{} updated", label()));
        }
    }
    actions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_iteration_budget() {
        let b = IterationBudget::new(3);
        assert!(b.consume());
        assert!(b.consume());
        assert!(b.consume());
        assert!(!b.consume());
        assert_eq!(b.used(), 3);
        assert_eq!(b.remaining(), 0);
        b.refund();
        assert_eq!(b.used(), 2);
        assert_eq!(b.remaining(), 1);
        assert!(b.consume());
    }

    #[test]
    fn test_is_destructive_command() {
        assert!(is_destructive_command("rm -rf /tmp/x"));
        assert!(is_destructive_command("foo && mv a b"));
        assert!(is_destructive_command("git reset --hard"));
        assert!(is_destructive_command("echo hi > file.txt"));
        assert!(!is_destructive_command("ls -la"));
        assert!(!is_destructive_command("echo hi >> file.txt"));
        assert!(!is_destructive_command(""));
        assert!(is_destructive_command("sed -i 's/a/b/' f"));
    }

    #[test]
    fn test_strip_non_ascii() {
        assert_eq!(strip_non_ascii("héllo"), "hllo");
        assert_eq!(strip_non_ascii("abc"), "abc");
        assert_eq!(strip_non_ascii("日本語"), "");
    }

    #[test]
    fn test_repair_tool_call_arguments() {
        assert_eq!(repair_tool_call_arguments("", "x"), "{}");
        assert_eq!(repair_tool_call_arguments("   ", "x"), "{}");
        assert_eq!(repair_tool_call_arguments("None", "x"), "{}");
        // valid JSON re-serialised compactly
        assert_eq!(repair_tool_call_arguments("{\"a\": 1}", "x"), "{\"a\":1}");
        // trailing comma
        assert_eq!(repair_tool_call_arguments("{\"a\":1,}", "x"), "{\"a\":1}");
        // unclosed
        assert_eq!(repair_tool_call_arguments("{\"a\":1", "x"), "{\"a\":1}");
        // unrepairable
        assert_eq!(repair_tool_call_arguments("not json at all", "x"), "{}");
    }

    #[test]
    fn test_escape_control_chars() {
        let out = escape_invalid_chars_in_json_strings("{\"a\":\"x\ty\"}");
        assert_eq!(out, "{\"a\":\"x\\u0009y\"}");
        // already escaped passes through
        let out2 = escape_invalid_chars_in_json_strings("{\"a\":\"x\\ny\"}");
        assert_eq!(out2, "{\"a\":\"x\\ny\"}");
    }

    #[test]
    fn test_strip_think_blocks() {
        assert_eq!(strip_think_blocks("<think>secret</think>visible"), "visible");
        assert_eq!(
            strip_think_blocks("before<thinking>x</thinking>after"),
            "beforeafter"
        );
        // unterminated at boundary
        assert_eq!(strip_think_blocks("ok\n<think>leak to end"), "ok");
        // case-insensitive
        assert_eq!(strip_think_blocks("<THINK>x</THINK>hi"), "hi");
        assert_eq!(strip_think_blocks(""), "");
    }

    #[test]
    fn test_has_content_after_think_block() {
        assert!(has_content_after_think_block("<think>x</think>real"));
        assert!(!has_content_after_think_block("<think>only</think>"));
        assert!(!has_content_after_think_block(""));
    }

    #[test]
    fn test_has_natural_response_ending() {
        assert!(has_natural_response_ending("Done."));
        assert!(has_natural_response_ending("code:\n```"));
        assert!(has_natural_response_ending("こんにちは。"));
        assert!(!has_natural_response_ending("this is unfinished and"));
        assert!(!has_natural_response_ending(""));
    }

    #[test]
    fn test_extract_reasoning() {
        let mut m = AssistantMessageView::default();
        m.reasoning = Some("step1".to_string());
        m.reasoning_content = Some("step1".to_string()); // dup, skipped
        assert_eq!(extract_reasoning(&m), Some("step1".to_string()));

        let mut m2 = AssistantMessageView::default();
        m2.reasoning_details = vec![json!({"type": "reasoning.summary", "summary": "thoughts"})];
        assert_eq!(extract_reasoning(&m2), Some("thoughts".to_string()));

        let mut m3 = AssistantMessageView::default();
        m3.content = Some("<think>inline reasoning</think>answer".to_string());
        assert_eq!(extract_reasoning(&m3), Some("inline reasoning".to_string()));

        let m4 = AssistantMessageView::default();
        assert_eq!(extract_reasoning(&m4), None);
    }

    #[test]
    fn test_mask_api_key() {
        assert_eq!(mask_api_key_for_logs(None), None);
        assert_eq!(mask_api_key_for_logs(Some("")), None);
        assert_eq!(mask_api_key_for_logs(Some("short")), Some("***".to_string()));
        assert_eq!(
            mask_api_key_for_logs(Some("sk-1234567890abcdef")),
            Some("sk-12345...cdef".to_string())
        );
    }

    #[test]
    fn test_clean_error_message() {
        assert_eq!(clean_error_message(""), "Unknown error");
        assert_eq!(
            clean_error_message("<!DOCTYPE html><html>..."),
            "Service temporarily unavailable (HTML error page returned)"
        );
        assert_eq!(clean_error_message("foo   bar\n baz"), "foo bar baz");
        let long = "a".repeat(200);
        let cleaned = clean_error_message(&long);
        assert!(cleaned.ends_with("..."));
        assert_eq!(cleaned.chars().count(), 153);
    }

    #[test]
    fn test_summarize_api_error_html() {
        let err = ApiErrorView {
            raw: "<!DOCTYPE html><title>502 Bad Gateway</title>".to_string(),
            status_code: Some(502),
            ..Default::default()
        };
        assert_eq!(summarize_api_error(&err), "HTTP 502 — 502 Bad Gateway");
    }

    #[test]
    fn test_summarize_api_error_json_body() {
        let err = ApiErrorView {
            raw: "RateLimitError".to_string(),
            status_code: Some(429),
            body: Some(json!({"error": {"message": "rate limited"}})),
            ..Default::default()
        };
        assert_eq!(summarize_api_error(&err), "HTTP 429: rate limited");
    }

    #[test]
    fn test_extract_api_error_context() {
        let err = ApiErrorView {
            raw: "boom".to_string(),
            body: Some(json!({"error": {"code": "rate_limit", "message": "slow down"}})),
            ..Default::default()
        };
        let ctx = extract_api_error_context(&err, 1000.0);
        assert_eq!(ctx.get("reason").and_then(|v| v.as_str()), Some("rate_limit"));
        assert_eq!(ctx.get("message").and_then(|v| v.as_str()), Some("slow down"));

        let err2 = ApiErrorView {
            raw: "please retry after 30 seconds".to_string(),
            ..Default::default()
        };
        let ctx2 = extract_api_error_context(&err2, 1000.0);
        assert_eq!(ctx2.get("reset_at").and_then(|v| v.as_f64()), Some(1030.0));
    }

    #[test]
    fn test_sanitize_api_messages_orphan_result() {
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "tool", "tool_call_id": "abc", "content": "orphan"}),
        ];
        let out = sanitize_api_messages(messages);
        // orphan tool result dropped
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
    }

    #[test]
    fn test_sanitize_api_messages_missing_result() {
        let messages = vec![
            json!({"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}]}),
        ];
        let out = sanitize_api_messages(messages);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["role"], "tool");
        assert_eq!(out[1]["tool_call_id"], "c1");
        assert_eq!(out[1]["name"], "read_file");
    }

    #[test]
    fn test_sanitize_api_messages_invalid_role() {
        let messages = vec![
            json!({"role": "bogus", "content": "x"}),
            json!({"role": "user", "content": "y"}),
        ];
        let out = sanitize_api_messages(messages);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
    }

    #[test]
    fn test_is_thinking_only_assistant() {
        assert!(is_thinking_only_assistant(
            &json!({"role": "assistant", "content": "", "reasoning": "thinking"})
        ));
        assert!(!is_thinking_only_assistant(
            &json!({"role": "assistant", "content": "real text"})
        ));
        assert!(!is_thinking_only_assistant(
            &json!({"role": "assistant", "tool_calls": [{"id": "x"}], "reasoning": "t"})
        ));
        assert!(!is_thinking_only_assistant(&json!({"role": "user", "content": ""})));
        // content list of thinking blocks only
        assert!(is_thinking_only_assistant(&json!({
            "role": "assistant",
            "content": [{"type": "thinking", "text": "x"}],
            "reasoning": "y"
        })));
    }

    #[test]
    fn test_drop_thinking_only_and_merge_users() {
        let messages = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "", "reasoning": "thinking only"}),
            json!({"role": "user", "content": "second"}),
        ];
        let out = drop_thinking_only_and_merge_users(messages);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"], "first\n\nsecond");
    }

    #[test]
    fn test_drop_thinking_only_noop() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let out = drop_thinking_only_and_merge_users(messages.clone());
        assert_eq!(out, messages);
    }

    #[test]
    fn test_cap_delegate_task_calls() {
        let calls = vec![
            ToolCall::new("1", "delegate_task", "{}"),
            ToolCall::new("2", "delegate_task", "{}"),
            ToolCall::new("3", "read_file", "{}"),
            ToolCall::new("4", "delegate_task", "{}"),
        ];
        let out = cap_delegate_task_calls(calls, 2);
        assert_eq!(out.len(), 3);
        let names: Vec<&str> = out.iter().map(|c| c.function.name.as_str()).collect();
        assert_eq!(names, vec!["delegate_task", "delegate_task", "read_file"]);
    }

    #[test]
    fn test_deduplicate_tool_calls() {
        let calls = vec![
            ToolCall::new("1", "read_file", "{\"path\":\"a\"}"),
            ToolCall::new("2", "read_file", "{\"path\":\"a\"}"),
            ToolCall::new("3", "read_file", "{\"path\":\"b\"}"),
        ];
        let out = deduplicate_tool_calls(calls);
        assert_eq!(out.len(), 2);

        let no_dups = vec![ToolCall::new("1", "x", "{}"), ToolCall::new("2", "y", "{}")];
        let out2 = deduplicate_tool_calls(no_dups.clone());
        assert_eq!(out2, no_dups);
    }

    #[test]
    fn test_repair_tool_call() {
        let mut valid: HashSet<String> = HashSet::new();
        valid.insert("todo".to_string());
        valid.insert("read_file".to_string());
        valid.insert("browser_click".to_string());

        assert_eq!(repair_tool_call("TODO", &valid), Some("todo".to_string()));
        assert_eq!(
            repair_tool_call("read-file", &valid),
            Some("read_file".to_string())
        );
        assert_eq!(repair_tool_call("TodoTool_tool", &valid), Some("todo".to_string()));
        assert_eq!(
            repair_tool_call("BrowserClick", &valid),
            Some("browser_click".to_string())
        );
        assert_eq!(repair_tool_call("", &valid), None);
        assert_eq!(repair_tool_call("xyzzy_no_match", &valid), None);
    }

    #[test]
    fn test_sanitize_tool_calls_for_strict_api() {
        let mut msg = json!({
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "call_id": "c1", "response_item_id": "r1", "function": {"name": "x"}}
            ]
        });
        sanitize_tool_calls_for_strict_api(&mut msg);
        let tc = &msg["tool_calls"][0];
        assert!(tc.get("call_id").is_none());
        assert!(tc.get("response_item_id").is_none());
        assert_eq!(tc["id"], "1");
    }

    #[test]
    fn test_sanitize_tool_call_arguments_repair() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "not json"}}
            ]
        })];
        let repaired = sanitize_tool_call_arguments(&mut messages);
        assert_eq!(repaired, 1);
        // arguments reset
        assert_eq!(messages[0]["tool_calls"][0]["function"]["arguments"], "{}");
        // stub tool message inserted
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "c1");
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .starts_with(TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER));
    }

    #[test]
    fn test_sanitize_tool_call_arguments_empty() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": ""}}
            ]
        })];
        let repaired = sanitize_tool_call_arguments(&mut messages);
        assert_eq!(repaired, 0);
        assert_eq!(messages[0]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_sanitize_tool_call_arguments_marker_existing() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "x", "arguments": "broken"}}
                ]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "old result"}),
        ];
        sanitize_tool_call_arguments(&mut messages);
        let content = messages[1]["content"].as_str().unwrap();
        assert!(content.starts_with(TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER));
        assert!(content.contains("old result"));
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_should_parallelize_tool_batch() {
        let cwd = Path::new("/tmp");
        let no_mcp = |_: &str| false;

        // single call → false
        let one = vec![ToolCall::new("1", "read_file", "{\"path\":\"/a\"}")];
        assert!(!should_parallelize_tool_batch(&one, cwd, no_mcp));

        // two safe reads on distinct paths → true
        let safe = vec![
            ToolCall::new("1", "web_search", "{\"q\":\"a\"}"),
            ToolCall::new("2", "web_search", "{\"q\":\"b\"}"),
        ];
        assert!(should_parallelize_tool_batch(&safe, cwd, no_mcp));

        // overlapping paths → false
        let overlap = vec![
            ToolCall::new("1", "write_file", "{\"path\":\"/a/b\"}"),
            ToolCall::new("2", "read_file", "{\"path\":\"/a/b/c\"}"),
        ];
        assert!(!should_parallelize_tool_batch(&overlap, cwd, no_mcp));

        // distinct paths → true
        let distinct = vec![
            ToolCall::new("1", "read_file", "{\"path\":\"/a\"}"),
            ToolCall::new("2", "read_file", "{\"path\":\"/b\"}"),
        ];
        assert!(should_parallelize_tool_batch(&distinct, cwd, no_mcp));

        // clarify never parallel
        let clarify = vec![
            ToolCall::new("1", "clarify", "{}"),
            ToolCall::new("2", "web_search", "{}"),
        ];
        assert!(!should_parallelize_tool_batch(&clarify, cwd, no_mcp));

        // unknown tool, no mcp → false
        let unknown = vec![
            ToolCall::new("1", "mystery", "{}"),
            ToolCall::new("2", "web_search", "{}"),
        ];
        assert!(!should_parallelize_tool_batch(&unknown, cwd, no_mcp));
    }

    #[test]
    fn test_paths_overlap() {
        assert!(paths_overlap(Path::new("/a/b"), Path::new("/a/b/c")));
        assert!(paths_overlap(Path::new("/a/b/c"), Path::new("/a/b")));
        assert!(!paths_overlap(Path::new("/a/b"), Path::new("/a/c")));
    }

    #[test]
    fn test_model_requires_responses_api() {
        assert!(model_requires_responses_api("gpt-5.4"));
        assert!(model_requires_responses_api("openai/gpt-5.1"));
        assert!(!model_requires_responses_api("gpt-4o"));
        assert!(!model_requires_responses_api("claude-opus-4"));
    }

    #[test]
    fn test_max_tokens_param() {
        assert_eq!(
            max_tokens_param(100, true, false),
            ("max_completion_tokens".to_string(), 100)
        );
        assert_eq!(
            max_tokens_param(100, false, true),
            ("max_completion_tokens".to_string(), 100)
        );
        assert_eq!(max_tokens_param(100, false, false), ("max_tokens".to_string(), 100));
    }

    #[test]
    fn test_qwen_portal_headers() {
        let h = qwen_portal_headers("darwin", "arm64");
        assert_eq!(h.get("User-Agent").unwrap(), "QwenCode/0.14.1 (darwin; arm64)");
        assert_eq!(h.get("X-DashScope-CacheControl").unwrap(), "enable");
        assert_eq!(h.get("X-DashScope-AuthType").unwrap(), "qwen-oauth");
    }

    #[test]
    fn test_routermint_headers() {
        let h = routermint_headers("1.2.3");
        assert_eq!(h.get("User-Agent").unwrap(), "HermesAgent/1.2.3");
    }

    struct FakePool {
        available: bool,
        entries: usize,
    }
    impl PoolView for FakePool {
        fn has_available(&self) -> bool {
            self.available
        }
        fn entries_len(&self) -> usize {
            self.entries
        }
    }

    #[test]
    fn test_pool_may_recover_from_rate_limit() {
        let none: Option<&FakePool> = None;
        assert!(!pool_may_recover_from_rate_limit(none, None, None));

        let unavailable = FakePool { available: false, entries: 5 };
        assert!(!pool_may_recover_from_rate_limit(Some(&unavailable), None, None));

        let single = FakePool { available: true, entries: 1 };
        assert!(!pool_may_recover_from_rate_limit(Some(&single), None, None));

        let multi = FakePool { available: true, entries: 3 };
        assert!(pool_may_recover_from_rate_limit(Some(&multi), None, None));

        // gemini-cli account-wide quota → no recover even with multi pool
        assert!(!pool_may_recover_from_rate_limit(
            Some(&multi),
            Some("google-gemini-cli"),
            None
        ));
        assert!(!pool_may_recover_from_rate_limit(
            Some(&multi),
            None,
            Some("cloudcode-pa://foo")
        ));
    }

    #[test]
    fn test_looks_like_codex_intermediate_ack() {
        let no_tools: Vec<Value> = vec![];
        assert!(looks_like_codex_intermediate_ack(
            "look at the repository",
            "I'll inspect the codebase for you",
            &no_tools
        ));
        // with a tool message → false
        let with_tool = vec![json!({"role": "tool", "content": "x"})];
        assert!(!looks_like_codex_intermediate_ack(
            "look at the repository",
            "I'll inspect the codebase",
            &with_tool
        ));
        // no future-ack phrasing → false
        assert!(!looks_like_codex_intermediate_ack(
            "look at the repo",
            "Here is the answer.",
            &no_tools
        ));
    }

    #[test]
    fn test_get_tool_call_id_name_static() {
        let tc = json!({"call_id": "cc", "id": "ii", "function": {"name": "read_file"}});
        assert_eq!(get_tool_call_id_static(&tc), "cc");
        assert_eq!(get_tool_call_name_static(&tc), "read_file");

        let tc2 = json!({"id": "ii"});
        assert_eq!(get_tool_call_id_static(&tc2), "ii");
        assert_eq!(get_tool_call_name_static(&tc2), "");
    }

    #[test]
    fn test_summarize_background_review_actions() {
        let prior = vec![json!({"role": "tool", "tool_call_id": "old", "content": "stale"})];
        let review = vec![
            json!({"role": "tool", "tool_call_id": "new1",
                   "content": "{\"success\": true, \"message\": \"Memory created\"}"}),
            json!({"role": "tool", "tool_call_id": "new2",
                   "content": "{\"success\": true, \"message\": \"Entry added\", \"target\": \"memory\"}"}),
            json!({"role": "tool", "tool_call_id": "old",
                   "content": "{\"success\": true, \"message\": \"should be skipped\"}"}),
        ];
        let actions = summarize_background_review_actions(&review, &prior);
        assert_eq!(actions, vec!["Memory created".to_string(), "Memory updated".to_string()]);
    }

    #[test]
    fn test_sanitize_messages_non_ascii() {
        let mut messages = vec![json!({"role": "user", "content": "héllo"})];
        let found = sanitize_messages_non_ascii(&mut messages);
        assert!(found);
        assert_eq!(messages[0]["content"], "hllo");

        let mut clean = vec![json!({"role": "user", "content": "hello"})];
        assert!(!sanitize_messages_non_ascii(&mut clean));
    }

    #[test]
    fn test_sanitize_messages_non_ascii_tool_calls() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{\"q\":\"café\"}"}}],
            "reasoning_content": "résumé"
        })];
        let found = sanitize_messages_non_ascii(&mut messages);
        assert!(found);
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"caf\"}"
        );
        assert_eq!(messages[0]["reasoning_content"], "rsum");
    }

    #[test]
    fn test_get_proxy_from_env() {
        // Save and clear all proxy vars first.
        let keys = [
            "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
        ];
        let saved: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
        unsafe {
            for k in keys {
                std::env::remove_var(k);
            }
        }
        assert_eq!(get_proxy_from_env(), None);
        unsafe {
            std::env::set_var("HTTP_PROXY", "proxy.example.com:8080");
        }
        assert_eq!(
            get_proxy_from_env(),
            Some("http://proxy.example.com:8080".to_string())
        );
        // restore
        unsafe {
            for (k, v) in saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn test_difflib_ratio() {
        // identical strings → 1.0
        assert!((difflib_ratio("abc", "abc") - 1.0).abs() < 1e-9);
        // no overlap → 0.0
        assert!(difflib_ratio("abc", "xyz").abs() < 1e-9);
        // partial
        let r = difflib_ratio("browser_click", "browserclick");
        assert!(r > 0.7);
    }
}
