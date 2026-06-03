//! Tool result persistence -- preserves large outputs instead of truncating.
//!
//! Defense against context-window overflow operates at three levels:
//!
//! 1. **Per-tool output cap** (inside each tool): Tools like `search_files`
//!    pre-truncate their own output before returning. This is the first line
//!    of defense and the only one the tool author controls.
//!
//! 2. **Per-result persistence** (`maybe_persist_tool_result`): After a tool
//!    returns, if its output exceeds the tool's registered threshold, the full
//!    output is written INTO THE SANDBOX temp dir (for example
//!    `/tmp/hermes-results/{tool_use_id}.txt` on standard Linux, or
//!    `$TMPDIR/hermes-results/{tool_use_id}.txt` on Termux) via `env.execute()`.
//!    The in-context content is replaced with a preview + file path reference.
//!    The model can `read_file` to access the full output on any backend.
//!
//! 3. **Per-turn aggregate budget** (`enforce_turn_budget`): After all tool
//!    results in a single assistant turn are collected, if the total exceeds
//!    `turn_budget` (200K), the largest non-persisted results are spilled to
//!    disk until the aggregate is under budget. This catches cases where many
//!    medium-sized results combine to overflow context.
//!
//! Ported faithfully from `tools/tool_result_storage.py` and
//! `tools/budget_config.py`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// budget_config constants (single source of truth)
// ---------------------------------------------------------------------------

/// Defaults matching the historical hardcoded values.
pub const DEFAULT_RESULT_SIZE_CHARS: usize = 100_000;
pub const DEFAULT_TURN_BUDGET_CHARS: usize = 200_000;
pub const DEFAULT_PREVIEW_SIZE_CHARS: usize = 1_500;

pub const PERSISTED_OUTPUT_TAG: &str = "<persisted-output>";
pub const PERSISTED_OUTPUT_CLOSING_TAG: &str = "</persisted-output>";
pub const STORAGE_DIR: &str = "/tmp/hermes-results";
pub const HEREDOC_MARKER: &str = "HERMES_PERSIST_EOF";

const BUDGET_TOOL_NAME: &str = "__budget_enforcement__";

/// Tools whose thresholds must never be overridden.
///
/// `read_file` resolves to an infinite threshold, which prevents infinite
/// persist -> read -> persist loops.
pub fn pinned_threshold(tool_name: &str) -> Option<Threshold> {
    match tool_name {
        "read_file" => Some(Threshold::Infinite),
        _ => None,
    }
}

/// A persistence threshold in characters, or infinite (never persist).
///
/// Mirrors Python's `int | float` where `float("inf")` is used as a sentinel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    Chars(usize),
    Infinite,
}

impl Threshold {
    /// Returns `true` if `content_len` should trigger persistence
    /// (i.e. strictly exceeds this threshold).
    pub fn exceeded_by(&self, content_len: usize) -> bool {
        match self {
            Threshold::Infinite => false,
            Threshold::Chars(n) => content_len > *n,
        }
    }
}

/// Trait abstracting the per-tool registry lookup used during threshold
/// resolution. In Python this is `tools.registry.registry.get_max_result_size`.
///
/// Implementors return the registered max result size (in chars) for a tool,
/// or `None` to fall back to the config default.
pub trait ResultSizeRegistry {
    fn get_max_result_size(&self, tool_name: &str) -> Option<usize>;
}

/// A registry that knows nothing -- always falls back to the config default.
/// Matches behaviour when no tool overrides are registered.
#[derive(Debug, Default, Clone)]
pub struct EmptyRegistry;

impl ResultSizeRegistry for EmptyRegistry {
    fn get_max_result_size(&self, _tool_name: &str) -> Option<usize> {
        None
    }
}

/// Immutable budget constants for the 3-layer tool result persistence system.
///
/// - Layer 2 (per-result): `resolve_threshold(tool_name)` -> threshold in chars.
/// - Layer 3 (per-turn):   `turn_budget` -> aggregate char budget across all
///   tool results in a single assistant turn.
/// - Preview:              `preview_size` -> inline snippet size after
///   persistence.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    pub default_result_size: usize,
    pub turn_budget: usize,
    pub preview_size: usize,
    pub tool_overrides: HashMap<String, usize>,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        BudgetConfig {
            default_result_size: DEFAULT_RESULT_SIZE_CHARS,
            turn_budget: DEFAULT_TURN_BUDGET_CHARS,
            preview_size: DEFAULT_PREVIEW_SIZE_CHARS,
            tool_overrides: HashMap::new(),
        }
    }
}

impl BudgetConfig {
    /// Default config -- matches the historical hardcoded behaviour exactly.
    pub fn default_budget() -> Self {
        BudgetConfig::default()
    }

    /// Resolve the persistence threshold for a tool.
    ///
    /// Priority: pinned -> `tool_overrides` -> registry per-tool -> default.
    pub fn resolve_threshold(
        &self,
        tool_name: &str,
        registry: &dyn ResultSizeRegistry,
    ) -> Threshold {
        if let Some(t) = pinned_threshold(tool_name) {
            return t;
        }
        if let Some(n) = self.tool_overrides.get(tool_name) {
            return Threshold::Chars(*n);
        }
        match registry.get_max_result_size(tool_name) {
            Some(n) => Threshold::Chars(n),
            None => Threshold::Chars(self.default_result_size),
        }
    }
}

// ---------------------------------------------------------------------------
// Environment abstraction
// ---------------------------------------------------------------------------

/// Result of an `env.execute()` call. Mirrors the Python dict returned by
/// `BaseEnvironment.execute`, of which only `returncode` is consulted here.
#[derive(Debug, Clone, Default)]
pub struct ExecResult {
    pub returncode: i32,
}

/// Abstraction over the active `BaseEnvironment`. Only the two pieces of
/// behaviour used by this module are modelled:
///
/// - `get_temp_dir`: optionally returns a temp-backed directory for the env.
/// - `execute`: run a shell command in the sandbox with a timeout (seconds).
pub trait Environment {
    /// Return the best temp-backed directory for this environment, if any.
    /// Default: `None` (use the standard `/tmp` storage dir).
    fn get_temp_dir(&self) -> Option<String> {
        None
    }

    /// Execute `cmd` in the sandbox with the given timeout in seconds.
    /// Errors map to the Python `except` path (treated as a failed write).
    fn execute(&self, cmd: &str, timeout: u64) -> Result<ExecResult, String>;
}

/// Return the best temp-backed storage dir for this environment.
fn resolve_storage_dir(env: Option<&dyn Environment>) -> String {
    if let Some(env) = env {
        match env.get_temp_dir() {
            Some(temp_dir) if !temp_dir.is_empty() => {
                // temp_dir.rstrip("/") or "/"
                let trimmed = temp_dir.trim_end_matches('/');
                let base = if trimmed.is_empty() { "/" } else { trimmed };
                return format!("{base}/hermes-results");
            }
            _ => {}
        }
    }
    STORAGE_DIR.to_string()
}

// ---------------------------------------------------------------------------
// Preview generation
// ---------------------------------------------------------------------------

/// Truncate at last newline within `max_chars`. Returns `(preview, has_more)`.
///
/// Note: like the Python original, lengths are measured in characters
/// (Unicode scalar values), not bytes.
pub fn generate_preview(content: &str, max_chars: usize) -> (String, bool) {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= max_chars {
        return (content.to_string(), false);
    }
    let mut truncated: String = chars[..max_chars].iter().collect();
    // last_nl = truncated.rfind("\n")  (char index)
    if let Some(byte_idx) = truncated.rfind('\n') {
        // Convert the byte index to a char index for the `> max_chars // 2` check.
        let char_idx = truncated[..byte_idx].chars().count();
        if char_idx > max_chars / 2 {
            // truncated[:last_nl + 1] -- keep up to and including the newline.
            let kept: String = truncated.chars().take(char_idx + 1).collect();
            truncated = kept;
        }
    }
    (truncated, true)
}

// ---------------------------------------------------------------------------
// Sandbox write
// ---------------------------------------------------------------------------

/// Return a heredoc delimiter that doesn't collide with `content`.
fn heredoc_marker(content: &str) -> String {
    if !content.contains(HEREDOC_MARKER) {
        return HEREDOC_MARKER.to_string();
    }
    // uuid4().hex[:8]
    let hex = uuid_hex8();
    format!("HERMES_PERSIST_{hex}")
}

/// Produce 8 hex chars of randomness (mirrors `uuid.uuid4().hex[:8]`).
fn uuid_hex8() -> String {
    // Cheap, dependency-free randomness sufficient for a heredoc nonce.
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let addr = &nanos as *const _ as usize as u128;
    let mixed = nanos ^ (addr << 17) ^ (addr.rotate_left(5));
    format!("{:08x}", (mixed as u32))
}

/// Shell-quote a string the way Python's `shlex.quote` does: wrap in single
/// quotes and escape embedded single quotes as `'"'"'`. Empty strings become
/// `''`.
fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // shlex.quote leaves "safe" strings unquoted. Safe chars are
    // [A-Za-z0-9_@%+=:,./-]. Match that to preserve byte-for-byte cmd parity.
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    });
    if safe {
        return s.to_string();
    }
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

/// `os.path.dirname` equivalent for the limited POSIX path use here.
fn dirname(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => "/".to_string(),
        Some(idx) => path[..idx].to_string(),
        None => String::new(),
    }
}

/// Write content into the sandbox via `env.execute()`. Returns `true` on
/// success.
fn write_to_sandbox(content: &str, remote_path: &str, env: &dyn Environment) -> Result<bool, String> {
    let marker = heredoc_marker(content);
    let storage_dir = dirname(remote_path);
    let cmd = format!(
        "mkdir -p {} && cat > {} << '{}'\n{}\n{}",
        shlex_quote(&storage_dir),
        shlex_quote(remote_path),
        marker,
        content,
        marker,
    );
    let result = env.execute(&cmd, 30)?;
    Ok(result.returncode == 0)
}

// ---------------------------------------------------------------------------
// Message construction
// ---------------------------------------------------------------------------

/// Format an integer with comma thousands separators (Python `{:,}`).
fn comma_int(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Build the `<persisted-output>` replacement block.
fn build_persisted_message(
    preview: &str,
    has_more: bool,
    original_size: usize,
    file_path: &str,
) -> String {
    let size_kb = original_size as f64 / 1024.0;
    let size_str = if size_kb >= 1024.0 {
        format!("{:.1} MB", size_kb / 1024.0)
    } else {
        format!("{:.1} KB", size_kb)
    };

    let preview_len = preview.chars().count();
    let mut msg = format!("{PERSISTED_OUTPUT_TAG}\n");
    msg.push_str(&format!(
        "This tool result was too large ({} characters, {}).\n",
        comma_int(original_size),
        size_str
    ));
    msg.push_str(&format!("Full output saved to: {file_path}\n"));
    msg.push_str(
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n",
    );
    msg.push_str(&format!("Preview (first {preview_len} chars):\n"));
    msg.push_str(preview);
    if has_more {
        msg.push_str("\n...");
    }
    msg.push_str(&format!("\n{PERSISTED_OUTPUT_CLOSING_TAG}"));
    msg
}

// ---------------------------------------------------------------------------
// Layer 2: per-result persistence
// ---------------------------------------------------------------------------

/// Layer 2: persist oversized result into the sandbox, return preview + path.
///
/// Writes via `env.execute()` so the file is accessible from any backend
/// (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation if
/// the write fails or no env is available.
///
/// `threshold` is an explicit override; when `Some(..)` it takes precedence
/// over config resolution (matching the Python `threshold` parameter).
pub fn maybe_persist_tool_result(
    content: &str,
    tool_name: &str,
    tool_use_id: &str,
    env: Option<&dyn Environment>,
    config: &BudgetConfig,
    registry: &dyn ResultSizeRegistry,
    threshold: Option<Threshold>,
) -> String {
    let effective_threshold =
        threshold.unwrap_or_else(|| config.resolve_threshold(tool_name, registry));

    if effective_threshold == Threshold::Infinite {
        return content.to_string();
    }

    let content_len = content.chars().count();
    if !effective_threshold.exceeded_by(content_len) {
        return content.to_string();
    }

    let storage_dir = resolve_storage_dir(env);
    let remote_path = format!("{storage_dir}/{tool_use_id}.txt");
    let (preview, has_more) = generate_preview(content, config.preview_size);

    if let Some(env) = env {
        match write_to_sandbox(content, &remote_path, env) {
            Ok(true) => {
                log::info!(
                    "Persisted large tool result: {} ({}, {} chars -> {})",
                    tool_name,
                    tool_use_id,
                    content_len,
                    remote_path
                );
                return build_persisted_message(&preview, has_more, content_len, &remote_path);
            }
            Ok(false) => {}
            Err(exc) => {
                log::warn!("Sandbox write failed for {tool_use_id}: {exc}");
            }
        }
    }

    log::info!(
        "Inline-truncating large tool result: {} ({} chars, no sandbox write)",
        tool_name,
        content_len
    );
    format!(
        "{}\n\n[Truncated: tool response was {} chars. Full output could not be saved to sandbox.]",
        preview,
        comma_int(content_len)
    )
}

// ---------------------------------------------------------------------------
// Layer 3: per-turn aggregate budget
// ---------------------------------------------------------------------------

/// A tool result message. Mirrors the relevant fields of the Python dict.
#[derive(Debug, Clone)]
pub struct ToolMessage {
    pub content: String,
    /// `tool_call_id`; when absent the index-based default `budget_{idx}` is used.
    pub tool_call_id: Option<String>,
}

/// Layer 3: enforce aggregate budget across all tool results in a turn.
///
/// If total chars exceed the budget, persist the largest non-persisted results
/// first (via sandbox write) until under budget. Already-persisted results are
/// skipped.
///
/// Mutates the slice in-place; returns nothing extra (the Python version
/// returns the same list it mutates).
pub fn enforce_turn_budget(
    tool_messages: &mut [ToolMessage],
    env: Option<&dyn Environment>,
    config: &BudgetConfig,
    registry: &dyn ResultSizeRegistry,
) {
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let mut total_size: usize = 0;

    for (i, msg) in tool_messages.iter().enumerate() {
        let size = msg.content.chars().count();
        total_size += size;
        if !msg.content.contains(PERSISTED_OUTPUT_TAG) {
            candidates.push((i, size));
        }
    }

    if total_size <= config.turn_budget {
        return;
    }

    // Sort by size descending. Python's list.sort is stable; preserve the
    // original index order for ties to match exactly.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    for (idx, size) in candidates {
        if total_size <= config.turn_budget {
            break;
        }
        let content = tool_messages[idx].content.clone();
        let tool_use_id = tool_messages[idx]
            .tool_call_id
            .clone()
            .unwrap_or_else(|| format!("budget_{idx}"));

        let replacement = maybe_persist_tool_result(
            &content,
            BUDGET_TOOL_NAME,
            &tool_use_id,
            env,
            config,
            registry,
            Some(Threshold::Chars(0)),
        );

        if replacement != content {
            total_size -= size;
            total_size += replacement.chars().count();
            tool_messages[idx].content = replacement;
            log::info!(
                "Budget enforcement: persisted tool result {tool_use_id} ({size} chars)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct MockEnv {
        temp_dir: Option<String>,
        succeed: bool,
        calls: RefCell<Vec<String>>,
    }

    impl MockEnv {
        fn new(succeed: bool) -> Self {
            MockEnv {
                temp_dir: None,
                succeed,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Environment for MockEnv {
        fn get_temp_dir(&self) -> Option<String> {
            self.temp_dir.clone()
        }
        fn execute(&self, cmd: &str, _timeout: u64) -> Result<ExecResult, String> {
            self.calls.borrow_mut().push(cmd.to_string());
            Ok(ExecResult {
                returncode: if self.succeed { 0 } else { 1 },
            })
        }
    }

    #[test]
    fn preview_small_content_unchanged() {
        let (p, more) = generate_preview("hello", 100);
        assert_eq!(p, "hello");
        assert!(!more);
    }

    #[test]
    fn preview_truncates_at_newline_past_half() {
        // 20 chars, max 10. newline at index 8 (> 5 == 10//2) -> keep through nl.
        let content = "abcdefgh\nij klmnopqrstuvwxyz";
        let (p, more) = generate_preview(content, 10);
        assert!(more);
        assert_eq!(p, "abcdefgh\n");
    }

    #[test]
    fn preview_newline_before_half_keeps_hard_cut() {
        // newline at index 2 (<= 5) -> no rewind, hard cut at 10 chars.
        let content = "ab\ncdefghijklmnop";
        let (p, more) = generate_preview(content, 10);
        assert!(more);
        assert_eq!(p.chars().count(), 10);
        assert_eq!(p, "ab\ncdefghi");
    }

    #[test]
    fn resolve_threshold_pinned_read_file_is_infinite() {
        let cfg = BudgetConfig::default();
        let t = cfg.resolve_threshold("read_file", &EmptyRegistry);
        assert_eq!(t, Threshold::Infinite);
    }

    #[test]
    fn resolve_threshold_override_beats_registry() {
        let mut cfg = BudgetConfig::default();
        cfg.tool_overrides.insert("foo".to_string(), 5);
        let t = cfg.resolve_threshold("foo", &EmptyRegistry);
        assert_eq!(t, Threshold::Chars(5));
    }

    #[test]
    fn resolve_threshold_default_fallback() {
        let cfg = BudgetConfig::default();
        let t = cfg.resolve_threshold("bar", &EmptyRegistry);
        assert_eq!(t, Threshold::Chars(DEFAULT_RESULT_SIZE_CHARS));
    }

    #[test]
    fn maybe_persist_below_threshold_returns_original() {
        let cfg = BudgetConfig::default();
        let out = maybe_persist_tool_result(
            "small",
            "some_tool",
            "id1",
            None,
            &cfg,
            &EmptyRegistry,
            Some(Threshold::Chars(100)),
        );
        assert_eq!(out, "small");
    }

    #[test]
    fn maybe_persist_infinite_returns_original() {
        let cfg = BudgetConfig::default();
        let big = "x".repeat(10_000);
        let out = maybe_persist_tool_result(
            &big,
            "read_file",
            "id1",
            None,
            &cfg,
            &EmptyRegistry,
            None,
        );
        assert_eq!(out, big);
    }

    #[test]
    fn maybe_persist_writes_to_sandbox_on_success() {
        let cfg = BudgetConfig::default();
        let env = MockEnv::new(true);
        let big = "y".repeat(2000);
        let out = maybe_persist_tool_result(
            &big,
            "some_tool",
            "abc123",
            Some(&env),
            &cfg,
            &EmptyRegistry,
            Some(Threshold::Chars(100)),
        );
        assert!(out.starts_with(PERSISTED_OUTPUT_TAG));
        assert!(out.ends_with(PERSISTED_OUTPUT_CLOSING_TAG));
        assert!(out.contains("/tmp/hermes-results/abc123.txt"));
        assert!(out.contains("2,000 characters"));
        let calls = env.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains("mkdir -p /tmp/hermes-results"));
        assert!(calls[0].contains(HEREDOC_MARKER));
    }

    #[test]
    fn maybe_persist_falls_back_inline_on_write_failure() {
        let cfg = BudgetConfig::default();
        let env = MockEnv::new(false);
        let big = "z".repeat(2000);
        let out = maybe_persist_tool_result(
            &big,
            "some_tool",
            "abc123",
            Some(&env),
            &cfg,
            &EmptyRegistry,
            Some(Threshold::Chars(100)),
        );
        assert!(out.contains("[Truncated: tool response was 2,000 chars."));
        assert!(!out.starts_with(PERSISTED_OUTPUT_TAG));
    }

    #[test]
    fn maybe_persist_no_env_inline_truncates() {
        let cfg = BudgetConfig::default();
        let big = "q".repeat(500);
        let out = maybe_persist_tool_result(
            &big,
            "some_tool",
            "x",
            None,
            &cfg,
            &EmptyRegistry,
            Some(Threshold::Chars(100)),
        );
        assert!(out.contains("[Truncated: tool response was 500 chars."));
    }

    #[test]
    fn resolve_storage_dir_uses_env_temp() {
        struct E;
        impl Environment for E {
            fn get_temp_dir(&self) -> Option<String> {
                Some("/data/data/com.termux/files/usr/tmp/".to_string())
            }
            fn execute(&self, _: &str, _: u64) -> Result<ExecResult, String> {
                Ok(ExecResult::default())
            }
        }
        let dir = resolve_storage_dir(Some(&E));
        assert_eq!(dir, "/data/data/com.termux/files/usr/tmp/hermes-results");
    }

    #[test]
    fn resolve_storage_dir_root_slash() {
        struct E;
        impl Environment for E {
            fn get_temp_dir(&self) -> Option<String> {
                Some("/".to_string())
            }
            fn execute(&self, _: &str, _: u64) -> Result<ExecResult, String> {
                Ok(ExecResult::default())
            }
        }
        let dir = resolve_storage_dir(Some(&E));
        assert_eq!(dir, "//hermes-results");
    }

    #[test]
    fn shlex_quote_matches_python() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("/tmp/hermes-results"), "/tmp/hermes-results");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn dirname_works() {
        assert_eq!(dirname("/tmp/hermes-results/x.txt"), "/tmp/hermes-results");
        assert_eq!(dirname("/x.txt"), "/");
        assert_eq!(dirname("x.txt"), "");
    }

    #[test]
    fn comma_int_formats() {
        assert_eq!(comma_int(0), "0");
        assert_eq!(comma_int(999), "999");
        assert_eq!(comma_int(1000), "1,000");
        assert_eq!(comma_int(1234567), "1,234,567");
    }

    #[test]
    fn heredoc_marker_avoids_collision() {
        let content = format!("text containing {HEREDOC_MARKER} inside");
        let m = heredoc_marker(&content);
        assert_ne!(m, HEREDOC_MARKER);
        assert!(m.starts_with("HERMES_PERSIST_"));
    }

    #[test]
    fn build_persisted_message_mb_units() {
        let msg = build_persisted_message("preview", true, 2 * 1024 * 1024, "/p.txt");
        assert!(msg.contains("2.0 MB"));
        assert!(msg.ends_with("\n..."));
    }

    #[test]
    fn enforce_turn_budget_under_budget_noop() {
        let cfg = BudgetConfig::default();
        let mut msgs = vec![ToolMessage {
            content: "short".to_string(),
            tool_call_id: Some("a".to_string()),
        }];
        let original = msgs[0].content.clone();
        enforce_turn_budget(&mut msgs, None, &cfg, &EmptyRegistry);
        assert_eq!(msgs[0].content, original);
    }

    #[test]
    fn enforce_turn_budget_spills_largest_first() {
        let mut cfg = BudgetConfig::default();
        cfg.turn_budget = 1000;
        let env = MockEnv::new(true);
        let mut msgs = vec![
            ToolMessage {
                content: "a".repeat(300),
                tool_call_id: Some("small".to_string()),
            },
            ToolMessage {
                content: "b".repeat(900),
                tool_call_id: Some("big".to_string()),
            },
        ];
        enforce_turn_budget(&mut msgs, Some(&env), &cfg, &EmptyRegistry);
        // The big one should be persisted; the small one likely untouched.
        assert!(msgs[1].content.starts_with(PERSISTED_OUTPUT_TAG));
    }

    #[test]
    fn enforce_turn_budget_skips_already_persisted() {
        let mut cfg = BudgetConfig::default();
        cfg.turn_budget = 10;
        let env = MockEnv::new(true);
        let persisted = format!("{PERSISTED_OUTPUT_TAG} already done");
        let mut msgs = vec![ToolMessage {
            content: persisted.clone(),
            tool_call_id: Some("x".to_string()),
        }];
        enforce_turn_budget(&mut msgs, Some(&env), &cfg, &EmptyRegistry);
        // No candidate -> untouched, no execute call.
        assert_eq!(msgs[0].content, persisted);
        assert_eq!(env.calls.borrow().len(), 0);
    }
}
