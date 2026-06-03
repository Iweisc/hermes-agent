//! File Operations Module (native Rust port of `tools/file_operations.py`).
//!
//! Provides file manipulation capabilities (read, write, patch, search) that
//! work across all terminal backends (local, docker, ssh, singularity, modal,
//! daytona, vercel_sandbox).
//!
//! The key insight is that all file operations can be expressed as shell
//! commands, so we wrap the terminal backend's `execute()` interface to provide
//! a unified file API.
//!
//! The Python original duck-typed a `terminal_env` object exposing
//! `execute(command, cwd=..., timeout=..., stdin_data=...) -> {"output", "returncode"}`
//! plus a live `cwd` attribute. Here that contract is expressed as the
//! [`TerminalEnv`] trait; concrete backends implement it. [`ShellFileOperations`]
//! is generic over it.
//!
//! Cross-module dependencies (already ported):
//!   * `crate::agent_file_safety` — write deny-list / safe-root.
//!   * `crate::tool_binary_extensions` — binary extension set.
//!   * `crate::tool_tool_output_limits` — configurable read/line caps.
//!   * `crate::tool_fuzzy_match` — fuzzy find-and-replace for `patch_replace`.
//!   * `crate::tool_patch_parser` — V4A parse + apply for `patch_v4a`.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::agent_file_safety::is_write_denied;
use crate::tool_binary_extensions::has_binary_extension;
use crate::tool_fuzzy_match::{
    fuzzy_find_and_replace, format_no_match_hint,
};
use crate::tool_patch_parser::{
    apply_v4a_operations, parse_v4a_patch, splitlines_keepends, unified_diff, FileOpResult,
    FileOps, FuzzyMatcher, FuzzyResult as PpFuzzyResult,
};
use crate::tool_tool_output_limits::{get_max_line_length, get_max_lines};

// ---------------------------------------------------------------------------
// Terminal fence-leak stripping
// ---------------------------------------------------------------------------

fn osc_sequence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // \x1b][^\x07\x1b]*(?:\x07|\x1b\\)
    RE.get_or_init(|| Regex::new("\u{1b}\\][^\u{7}\u{1b}]*(?:\u{7}|\u{1b}\\\\)").unwrap())
}

fn fence_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // '?\x07?__HERMES_FENCE_[A-Za-z0-9]+__\x07?'?
    RE.get_or_init(|| {
        Regex::new("'?\u{7}?__HERMES_FENCE_[A-Za-z0-9]+__\u{7}?'?").unwrap()
    })
}

/// Split text into lines preserving line endings (mirrors Python
/// `str.splitlines(keepends=True)` for the `\n`/`\r\n`/`\r` cases that matter
/// here — we reuse the patch-parser helper which keeps `\n` terminators).
fn splitlines_keepends_local(text: &str) -> Vec<String> {
    // Python's splitlines splits on more boundaries (\r, \r\n, \n, \v, \f, etc).
    // For fence stripping we only need \n / \r\n behavior; the patch_parser
    // helper handles \n. Handle a lone trailing \r-terminated line as well by
    // delegating to the shared keepends splitter, which is sufficient for the
    // outputs produced by shell backends here.
    splitlines_keepends(text)
}

/// Strip leaked terminal fence wrappers from file read output.
pub fn strip_terminal_fence_leaks(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let mut cleaned_lines: Vec<String> = Vec::new();
    for line in splitlines_keepends_local(text) {
        let had_terminal_wrapper = line.contains("__HERMES_FENCE_") || line.contains("\u{1b}]");
        let cleaned = osc_sequence_re().replace_all(&line, "");
        let cleaned = fence_marker_re().replace_all(&cleaned, "");
        let cleaned = cleaned.replace('\u{7}', "");
        if had_terminal_wrapper {
            // Python: cleaned.strip("'\r\n\t ") == ""
            let stripped = cleaned.trim_matches(|c| matches!(c, '\'' | '\r' | '\n' | '\t' | ' '));
            if stripped.is_empty() {
                continue;
            }
        }
        cleaned_lines.push(cleaned);
    }
    cleaned_lines.join("")
}

// =============================================================================
// Result Data Classes
// =============================================================================

/// Result from reading a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReadResult {
    pub content: String,
    pub total_lines: i64,
    pub file_size: i64,
    pub truncated: bool,
    pub hint: Option<String>,
    pub is_binary: bool,
    pub is_image: bool,
    pub base64_content: Option<String>,
    pub mime_type: Option<String>,
    /// For images: "WIDTHxHEIGHT".
    pub dimensions: Option<String>,
    pub error: Option<String>,
    pub similar_files: Vec<String>,
}

impl ReadResult {
    /// Mirror Python `to_dict()`: drop keys that are `None`/empty-list.
    /// `0`/`false`/`""` scalar defaults are kept (Python's filter only drops
    /// `None` and `[]`).
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("content".into(), self.content.clone().into());
        m.insert("total_lines".into(), self.total_lines.into());
        m.insert("file_size".into(), self.file_size.into());
        m.insert("truncated".into(), self.truncated.into());
        if let Some(h) = &self.hint {
            m.insert("hint".into(), h.clone().into());
        }
        m.insert("is_binary".into(), self.is_binary.into());
        m.insert("is_image".into(), self.is_image.into());
        if let Some(b) = &self.base64_content {
            m.insert("base64_content".into(), b.clone().into());
        }
        if let Some(mt) = &self.mime_type {
            m.insert("mime_type".into(), mt.clone().into());
        }
        if let Some(d) = &self.dimensions {
            m.insert("dimensions".into(), d.clone().into());
        }
        if let Some(e) = &self.error {
            m.insert("error".into(), e.clone().into());
        }
        if !self.similar_files.is_empty() {
            m.insert(
                "similar_files".into(),
                serde_json::Value::Array(
                    self.similar_files.iter().cloned().map(Into::into).collect(),
                ),
            );
        }
        serde_json::Value::Object(m)
    }

    fn error_msg(msg: impl Into<String>) -> Self {
        ReadResult {
            error: Some(msg.into()),
            ..Default::default()
        }
    }
}

/// Result from writing a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WriteResult {
    pub bytes_written: i64,
    pub dirs_created: bool,
    pub lint: Option<serde_json::Value>,
    pub error: Option<String>,
    pub warning: Option<String>,
}

impl WriteResult {
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("bytes_written".into(), self.bytes_written.into());
        m.insert("dirs_created".into(), self.dirs_created.into());
        if let Some(l) = &self.lint {
            m.insert("lint".into(), l.clone());
        }
        if let Some(e) = &self.error {
            m.insert("error".into(), e.clone().into());
        }
        if let Some(w) = &self.warning {
            m.insert("warning".into(), w.clone().into());
        }
        serde_json::Value::Object(m)
    }

    fn error_msg(msg: impl Into<String>) -> Self {
        WriteResult {
            error: Some(msg.into()),
            ..Default::default()
        }
    }
}

/// Result from patching a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PatchResult {
    pub success: bool,
    pub diff: String,
    pub files_modified: Vec<String>,
    pub files_created: Vec<String>,
    pub files_deleted: Vec<String>,
    pub lint: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl PatchResult {
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("success".into(), self.success.into());
        if !self.diff.is_empty() {
            m.insert("diff".into(), self.diff.clone().into());
        }
        if !self.files_modified.is_empty() {
            m.insert(
                "files_modified".into(),
                serde_json::Value::Array(
                    self.files_modified.iter().cloned().map(Into::into).collect(),
                ),
            );
        }
        if !self.files_created.is_empty() {
            m.insert(
                "files_created".into(),
                serde_json::Value::Array(
                    self.files_created.iter().cloned().map(Into::into).collect(),
                ),
            );
        }
        if !self.files_deleted.is_empty() {
            m.insert(
                "files_deleted".into(),
                serde_json::Value::Array(
                    self.files_deleted.iter().cloned().map(Into::into).collect(),
                ),
            );
        }
        if let Some(l) = &self.lint {
            m.insert("lint".into(), l.clone());
        }
        if let Some(e) = &self.error {
            m.insert("error".into(), e.clone().into());
        }
        serde_json::Value::Object(m)
    }

    fn error_msg(msg: impl Into<String>) -> Self {
        PatchResult {
            error: Some(msg.into()),
            ..Default::default()
        }
    }
}

/// A single search match.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchMatch {
    pub path: String,
    pub line_number: i64,
    pub content: String,
    /// Modification time for sorting.
    pub mtime: f64,
}

/// Result from searching.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchResult {
    pub matches: Vec<SearchMatch>,
    pub files: Vec<String>,
    pub counts: Vec<(String, i64)>,
    pub total_count: i64,
    pub truncated: bool,
    pub error: Option<String>,
}

impl SearchResult {
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("total_count".into(), self.total_count.into());
        if !self.matches.is_empty() {
            let arr: Vec<serde_json::Value> = self
                .matches
                .iter()
                .map(|sm| {
                    serde_json::json!({
                        "path": sm.path,
                        "line": sm.line_number,
                        "content": sm.content,
                    })
                })
                .collect();
            m.insert("matches".into(), serde_json::Value::Array(arr));
        }
        if !self.files.is_empty() {
            m.insert(
                "files".into(),
                serde_json::Value::Array(self.files.iter().cloned().map(Into::into).collect()),
            );
        }
        if !self.counts.is_empty() {
            let mut cm = serde_json::Map::new();
            for (k, v) in &self.counts {
                cm.insert(k.clone(), (*v).into());
            }
            m.insert("counts".into(), serde_json::Value::Object(cm));
        }
        if self.truncated {
            m.insert("truncated".into(), true.into());
        }
        if let Some(e) = &self.error {
            m.insert("error".into(), e.clone().into());
        }
        serde_json::Value::Object(m)
    }

    fn error_with_count(msg: impl Into<String>, total: i64) -> Self {
        SearchResult {
            error: Some(msg.into()),
            total_count: total,
            ..Default::default()
        }
    }
}

/// Result from linting a file.
#[derive(Debug, Clone, PartialEq)]
pub struct LintResult {
    pub success: bool,
    pub skipped: bool,
    pub output: String,
    pub message: String,
}

impl Default for LintResult {
    fn default() -> Self {
        LintResult {
            success: true,
            skipped: false,
            output: String::new(),
            message: String::new(),
        }
    }
}

impl LintResult {
    pub fn to_json(&self) -> serde_json::Value {
        if self.skipped {
            return serde_json::json!({"status": "skipped", "message": self.message});
        }
        let mut m = serde_json::Map::new();
        m.insert(
            "status".into(),
            if self.success { "ok" } else { "error" }.into(),
        );
        m.insert("output".into(), self.output.clone().into());
        if !self.message.is_empty() {
            m.insert("message".into(), self.message.clone().into());
        }
        serde_json::Value::Object(m)
    }
}

/// Result from executing a shell command.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecuteResult {
    pub stdout: String,
    pub exit_code: i64,
    /// Optional stderr; Python duck-typed `getattr(result, "stderr", None)`.
    pub stderr: Option<String>,
}

/// Parse grep/rg context output in `path-line-content` format.
///
/// Context lines are ambiguous because filenames may legitimately contain
/// `-<digits>-` segments. Prefer the rightmost numeric separator.
pub fn parse_search_context_line(line: &str) -> Option<(String, i64, String)> {
    if line.is_empty() || line == "--" {
        return None;
    }
    let re = {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"-(\d+)-").unwrap())
    };
    // Mirror Python's `re.finditer` (non-overlapping); keep the rightmost.
    let mut last: Option<(usize, usize, i64)> = None; // (start, end, number)
    for caps in re.captures_iter(line) {
        let whole = caps.get(0).unwrap();
        let g1 = caps.get(1).unwrap();
        let num: i64 = g1.as_str().parse().unwrap_or(0);
        last = Some((whole.start(), whole.end(), num));
    }
    let (start, end, num) = last?;
    let path = &line[..start];
    if path.is_empty() {
        return None;
    }
    Some((path.to_string(), num, line[end..].to_string()))
}

// =============================================================================
// Terminal backend interface
// =============================================================================

/// Abstract interface for the terminal environment a `ShellFileOperations`
/// runs commands through. Mirrors the duck-typed Python `terminal_env`
/// (`execute(...)` + live `cwd`).
pub trait TerminalEnv {
    /// Execute `command` and return its combined output + return code.
    ///
    /// * `cwd` — working directory (resolved by the caller).
    /// * `timeout` — optional timeout in seconds.
    /// * `stdin_data` — optional bytes piped to the process's stdin.
    fn execute(
        &self,
        command: &str,
        cwd: &str,
        timeout: Option<u64>,
        stdin_data: Option<&str>,
    ) -> ExecuteResult;

    /// Live working directory the backend currently tracks (picks up `cd`).
    /// `None` if the backend doesn't track a cwd.
    fn cwd(&self) -> Option<String> {
        None
    }
}

// =============================================================================
// Constants
// =============================================================================

/// Image extensions (subset of binary that we can return as base64).
pub const IMAGE_EXTENSIONS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
];

/// Shell-based linters by file extension. The format string contains `{file}`
/// which is replaced with the (escaped) path.
pub fn shell_linter_for(ext: &str) -> Option<&'static str> {
    match ext {
        ".py" => Some("python -m py_compile {file} 2>&1"),
        ".js" => Some("node --check {file} 2>&1"),
        ".ts" => Some("npx tsc --noEmit {file} 2>&1"),
        ".go" => Some("go vet {file} 2>&1"),
        ".rs" => Some("rustfmt --check {file} 2>&1"),
        _ => None,
    }
}

/// Which extensions have an in-process linter (preferred over shell linters).
pub fn has_inproc_linter(ext: &str) -> bool {
    matches!(ext, ".py" | ".json" | ".yaml" | ".yml" | ".toml")
}

// Max limits for read operations.
pub const MAX_LINES: i64 = 2000;
pub const MAX_LINE_LENGTH: i64 = 2000;
pub const MAX_FILE_SIZE: i64 = 50 * 1024; // 50KB
pub const DEFAULT_READ_OFFSET: i64 = 1;
pub const DEFAULT_READ_LIMIT: i64 = 500;
pub const DEFAULT_SEARCH_OFFSET: i64 = 0;
pub const DEFAULT_SEARCH_LIMIT: i64 = 50;

// =============================================================================
// In-process linters
// =============================================================================

/// In-process JSON syntax check. Returns `(ok, error_message)`.
pub fn lint_json_inproc(content: &str) -> (bool, String) {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(_) => (true, String::new()),
        Err(e) => {
            // serde_json reports line/column; mirror the Python flavour loosely.
            (
                false,
                format!(
                    "JSONDecodeError: {} (line {}, column {})",
                    e, e.line(), e.column()
                ),
            )
        }
    }
}

/// In-process YAML syntax check. Returns `(ok, error_message)`.
pub fn lint_yaml_inproc(content: &str) -> (bool, String) {
    match serde_yaml::from_str::<serde_yaml::Value>(content) {
        Ok(_) => (true, String::new()),
        Err(e) => (false, format!("YAMLError: {}", e)),
    }
}

/// In-process TOML syntax check. The `toml` crate isn't in the allowed set, so
/// this signals `__SKIP__` (treated as "no linter") to preserve the Python
/// graceful-skip semantics when no TOML parser is available.
pub fn lint_toml_inproc(_content: &str) -> (bool, String) {
    (true, "__SKIP__".to_string())
}

/// In-process Python syntax check. Rust cannot parse Python AST; signal
/// `__SKIP__` so callers fall back to the shell linter (`py_compile`),
/// matching the Python module's degrade-gracefully behaviour when the
/// in-process linter is unavailable.
pub fn lint_python_inproc(_content: &str) -> (bool, String) {
    (true, "__SKIP__".to_string())
}

/// Dispatch the in-process linter for an extension. Returns `None` if there's
/// no in-process linter for that extension.
fn run_inproc_linter(ext: &str, content: &str) -> Option<(bool, String)> {
    match ext {
        ".py" => Some(lint_python_inproc(content)),
        ".json" => Some(lint_json_inproc(content)),
        ".yaml" | ".yml" => Some(lint_yaml_inproc(content)),
        ".toml" => Some(lint_toml_inproc(content)),
        _ => None,
    }
}

// =============================================================================
// Pagination helpers
// =============================================================================

/// Best-effort integer coercion for tool pagination inputs.
pub fn coerce_int(value: Option<i64>, default: i64) -> i64 {
    value.unwrap_or(default)
}

/// Return safe `read_file` pagination bounds. The upper bound on `limit`
/// comes from `tool_output.max_lines`.
pub fn normalize_read_pagination(offset: Option<i64>, limit: Option<i64>) -> (i64, i64) {
    let max_lines = get_max_lines(None);
    let normalized_offset = std::cmp::max(1, coerce_int(offset, DEFAULT_READ_OFFSET));
    let mut normalized_limit = coerce_int(limit, DEFAULT_READ_LIMIT);
    normalized_limit = std::cmp::max(1, std::cmp::min(normalized_limit, max_lines));
    (normalized_offset, normalized_limit)
}

/// Return safe search pagination bounds for shell head/tail pipelines.
pub fn normalize_search_pagination(offset: Option<i64>, limit: Option<i64>) -> (i64, i64) {
    let normalized_offset = std::cmp::max(0, coerce_int(offset, DEFAULT_SEARCH_OFFSET));
    let normalized_limit = std::cmp::max(1, coerce_int(limit, DEFAULT_SEARCH_LIMIT));
    (normalized_offset, normalized_limit)
}

// =============================================================================
// Path helpers (mirroring os.path semantics used by the Python module)
// =============================================================================

/// `os.path.splitext` — returns `(root, ext)` where `ext` includes the dot.
fn splitext(path: &str) -> (String, String) {
    let base = basename(path);
    // Find last dot not at the start of the basename.
    if let Some(idx) = base.rfind('.') {
        if idx > 0 {
            let dot_pos = path.len() - base.len() + idx;
            return (path[..dot_pos].to_string(), path[dot_pos..].to_string());
        }
    }
    (path.to_string(), String::new())
}

/// Lowercased extension (including dot) of a path.
fn ext_lower(path: &str) -> String {
    splitext(path).1.to_lowercase()
}

/// `os.path.basename`.
fn basename(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[i + 1..].to_string(),
        None => path.to_string(),
    }
}

/// `os.path.dirname`.
fn dirname(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => path[..i].to_string(),
        None => String::new(),
    }
}

/// `os.path.join(a, b)` for the simple two-arg case used here.
fn path_join(a: &str, b: &str) -> String {
    if a.is_empty() {
        return b.to_string();
    }
    if b.starts_with('/') {
        return b.to_string();
    }
    if a.ends_with('/') {
        format!("{}{}", a, b)
    } else {
        format!("{}/{}", a, b)
    }
}

// =============================================================================
// Shell-based implementation
// =============================================================================

/// File operations implemented via shell commands. Works with any terminal
/// backend that implements [`TerminalEnv`].
pub struct ShellFileOperations<E: TerminalEnv> {
    pub env: E,
    /// Fallback cwd used only when the env doesn't track a live cwd.
    pub cwd: String,
    command_cache: HashMap<String, bool>,
}

impl<E: TerminalEnv> ShellFileOperations<E> {
    /// Construct with a terminal env and optional explicit fallback cwd.
    ///
    /// `config_cwd` mirrors the Python fallback chain
    /// (`cwd or env.cwd or env.config.cwd or "/"`). The live `env.cwd()` still
    /// takes precedence at exec time.
    pub fn new(env: E, cwd: Option<String>, config_cwd: Option<String>) -> Self {
        let resolved = cwd
            .or_else(|| env.cwd())
            .or(config_cwd)
            .unwrap_or_else(|| "/".to_string());
        ShellFileOperations {
            env,
            cwd: resolved,
            command_cache: HashMap::new(),
        }
    }

    /// Execute a command via the terminal backend, resolving cwd from the live
    /// env first, then the init-time fallback.
    pub fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        timeout: Option<u64>,
        stdin_data: Option<&str>,
    ) -> ExecuteResult {
        let effective_cwd = cwd
            .map(|s| s.to_string())
            .or_else(|| self.env.cwd())
            .unwrap_or_else(|| self.cwd.clone());
        self.env.execute(command, &effective_cwd, timeout, stdin_data)
    }

    /// Check if a command exists in the environment (cached).
    pub fn has_command(&mut self, cmd: &str) -> bool {
        if let Some(&cached) = self.command_cache.get(cmd) {
            return cached;
        }
        let result = self.exec(
            &format!("command -v {} >/dev/null 2>&1 && echo 'yes'", cmd),
            None,
            None,
            None,
        );
        let available = result.stdout.trim() == "yes";
        self.command_cache.insert(cmd.to_string(), available);
        available
    }

    /// Check if a file is likely binary (extension check + content analysis).
    fn is_likely_binary(&self, path: &str, content_sample: Option<&str>) -> bool {
        if has_binary_extension(path) {
            return true;
        }
        if let Some(sample) = content_sample {
            if sample.is_empty() {
                return false;
            }
            let chars: Vec<char> = sample.chars().take(1000).collect();
            let non_printable = chars
                .iter()
                .filter(|&&c| (c as u32) < 32 && !matches!(c, '\n' | '\r' | '\t'))
                .count();
            // Python: min(len(content_sample), 1000) — denominator uses the
            // raw length (chars) of the (possibly longer) sample, capped at 1000.
            let denom = std::cmp::min(sample.chars().count(), 1000);
            if denom == 0 {
                return false;
            }
            return (non_printable as f64) / (denom as f64) > 0.30;
        }
        false
    }

    /// Check if file is an image we can return as base64.
    fn is_image(&self, path: &str) -> bool {
        let ext = ext_lower(path);
        IMAGE_EXTENSIONS.contains(&ext.as_str())
    }

    /// Add line numbers in `LINE_NUM|CONTENT` format.
    fn add_line_numbers(&self, content: &str, start_line: i64) -> String {
        let max_line_length = get_max_line_length(None) as usize;
        let lines: Vec<&str> = content.split('\n').collect();
        let mut numbered: Vec<String> = Vec::with_capacity(lines.len());
        let mut i = start_line;
        for line in lines {
            // Truncate long lines (by Unicode code point, like Python slicing).
            let line_owned: String = if line.chars().count() > max_line_length {
                let truncated: String = line.chars().take(max_line_length).collect();
                format!("{}... [truncated]", truncated)
            } else {
                line.to_string()
            };
            numbered.push(format!("{:6}|{}", i, line_owned));
            i += 1;
        }
        numbered.join("\n")
    }

    /// Expand shell-style paths like `~` and `~user` to absolute paths.
    fn expand_path(&self, path: &str) -> String {
        if path.is_empty() {
            return path.to_string();
        }
        if path.starts_with('~') {
            let result = self.exec("echo $HOME", None, None, None);
            if result.exit_code == 0 && !result.stdout.trim().is_empty() {
                let home = result.stdout.trim().to_string();
                if path == "~" {
                    return home;
                } else if let Some(rest) = path.strip_prefix("~/") {
                    return format!("{}/{}", home, rest);
                }
                // ~username format.
                let rest = &path[1..];
                let username = match rest.find('/') {
                    Some(idx) => &rest[..idx],
                    None => rest,
                };
                let user_re = {
                    static RE: OnceLock<Regex> = OnceLock::new();
                    RE.get_or_init(|| Regex::new(r"^[a-zA-Z0-9._-]+$").unwrap())
                };
                if !username.is_empty() && user_re.is_match(username) {
                    let expand_result =
                        self.exec(&format!("echo ~{}", username), None, None, None);
                    if expand_result.exit_code == 0 && !expand_result.stdout.trim().is_empty() {
                        let user_home = expand_result.stdout.trim().to_string();
                        let suffix = &path[1 + username.len()..];
                        return format!("{}{}", user_home, suffix);
                    }
                }
            }
        }
        path.to_string()
    }

    /// Escape a string for safe use in single-quoted shell context.
    pub fn escape_shell_arg(&self, arg: &str) -> String {
        format!("'{}'", arg.replace('\'', "'\"'\"'"))
    }

    /// Generate unified diff between old and new content.
    fn unified_diff_str(&self, old_content: &str, new_content: &str, filename: &str) -> String {
        let old_lines = splitlines_keepends(old_content);
        let new_lines = splitlines_keepends(new_content);
        unified_diff(
            &old_lines,
            &new_lines,
            &format!("a/{}", filename),
            &format!("b/{}", filename),
        )
    }

    // =========================================================================
    // READ
    // =========================================================================

    /// Read a file with pagination, binary detection, and line numbers.
    pub fn read_file(&mut self, path: &str, offset: i64, limit: i64) -> ReadResult {
        let path = self.expand_path(path);
        let (offset, limit) = normalize_read_pagination(Some(offset), Some(limit));

        let stat_cmd = format!("wc -c < {} 2>/dev/null", self.escape_shell_arg(&path));
        let stat_result = self.exec(&stat_cmd, None, None, None);
        if stat_result.exit_code != 0 {
            return self.suggest_similar_files(&path);
        }
        let stat_output = strip_terminal_fence_leaks(&stat_result.stdout);
        let file_size: i64 = stat_output.trim().parse().unwrap_or(0);

        // file_size > MAX_FILE_SIZE: Python only `pass`es (no-op warning).

        if self.is_image(&path) {
            return ReadResult {
                is_image: true,
                is_binary: true,
                file_size,
                hint: Some(
                    "Image file detected. Automatically redirected to vision_analyze tool. \
                     Use vision_analyze with this file path to inspect the image contents."
                        .to_string(),
                ),
                ..Default::default()
            };
        }

        let sample_cmd = format!("head -c 1000 {} 2>/dev/null", self.escape_shell_arg(&path));
        let sample_result = self.exec(&sample_cmd, None, None, None);
        let sample_output = strip_terminal_fence_leaks(&sample_result.stdout);

        if self.is_likely_binary(&path, Some(&sample_output)) {
            return ReadResult {
                is_binary: true,
                file_size,
                error: Some(
                    "Binary file - cannot display as text. Use appropriate tools to handle this file type."
                        .to_string(),
                ),
                ..Default::default()
            };
        }

        let end_line = offset + limit - 1;
        let read_cmd = format!(
            "sed -n '{},{}p' {}",
            offset,
            end_line,
            self.escape_shell_arg(&path)
        );
        let read_result = self.exec(&read_cmd, None, None, None);
        if read_result.exit_code != 0 {
            return ReadResult::error_msg(format!("Failed to read file: {}", read_result.stdout));
        }
        let read_output = strip_terminal_fence_leaks(&read_result.stdout);

        let wc_cmd = format!("wc -l < {}", self.escape_shell_arg(&path));
        let wc_result = self.exec(&wc_cmd, None, None, None);
        let wc_output = strip_terminal_fence_leaks(&wc_result.stdout);
        let total_lines: i64 = wc_output.trim().parse().unwrap_or(0);

        let truncated = total_lines > end_line;
        let hint = if truncated {
            Some(format!(
                "Use offset={} to continue reading (showing {}-{} of {} lines)",
                end_line + 1,
                offset,
                end_line,
                total_lines
            ))
        } else {
            None
        };

        ReadResult {
            content: self.add_line_numbers(&read_output, offset),
            total_lines,
            file_size,
            truncated,
            hint,
            ..Default::default()
        }
    }

    /// Suggest similar files when the requested file is not found.
    fn suggest_similar_files(&self, path: &str) -> ReadResult {
        let dir_path = {
            let d = dirname(path);
            if d.is_empty() { ".".to_string() } else { d }
        };
        let filename = basename(path);
        let basename_no_ext = splitext(&filename).0;
        let ext = ext_lower(&filename);
        let lower_name = filename.to_lowercase();

        let ls_cmd = format!(
            "ls -1 {} 2>/dev/null | head -50",
            self.escape_shell_arg(&dir_path)
        );
        let ls_result = self.exec(&ls_cmd, None, None, None);

        let mut scored: Vec<(i64, String)> = Vec::new();
        if ls_result.exit_code == 0 && !ls_result.stdout.trim().is_empty() {
            for f in ls_result.stdout.trim().split('\n') {
                if f.is_empty() {
                    continue;
                }
                let lf = f.to_lowercase();
                let mut score: i64 = 0;

                if lf == lower_name {
                    score = 100;
                } else if splitext(f).0.to_lowercase() == basename_no_ext.to_lowercase() {
                    score = 90;
                } else if lf.starts_with(&lower_name) || lower_name.starts_with(&lf) {
                    score = 70;
                } else if lf.contains(&lower_name) {
                    score = 60;
                } else if lower_name.contains(&lf) && lf.len() > 2 {
                    score = 40;
                } else if !ext.is_empty() && ext_lower(f) == ext {
                    let set_name: std::collections::HashSet<char> = lower_name.chars().collect();
                    let set_lf: std::collections::HashSet<char> = lf.chars().collect();
                    let common = set_name.intersection(&set_lf).count();
                    let threshold =
                        (std::cmp::max(lower_name.chars().count(), lf.chars().count()) as f64) * 0.4;
                    if (common as f64) >= threshold {
                        score = 30;
                    }
                }

                if score > 0 {
                    scored.push((score, path_join(&dir_path, f)));
                }
            }
        }

        // Stable sort by descending score (Python sort is stable).
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let similar: Vec<String> = scored.into_iter().take(5).map(|(_, fp)| fp).collect();

        ReadResult {
            error: Some(format!("File not found: {}", path)),
            similar_files: similar,
            ..Default::default()
        }
    }

    /// Read the complete file content as a plain string (no pagination).
    pub fn read_file_raw(&mut self, path: &str) -> ReadResult {
        let path = self.expand_path(path);
        let stat_cmd = format!("wc -c < {} 2>/dev/null", self.escape_shell_arg(&path));
        let stat_result = self.exec(&stat_cmd, None, None, None);
        if stat_result.exit_code != 0 {
            return self.suggest_similar_files(&path);
        }
        let stat_output = strip_terminal_fence_leaks(&stat_result.stdout);
        let file_size: i64 = stat_output.trim().parse().unwrap_or(0);
        if self.is_image(&path) {
            return ReadResult {
                is_image: true,
                is_binary: true,
                file_size,
                ..Default::default()
            };
        }
        let sample_result = self.exec(
            &format!("head -c 1000 {} 2>/dev/null", self.escape_shell_arg(&path)),
            None,
            None,
            None,
        );
        let sample_output = strip_terminal_fence_leaks(&sample_result.stdout);
        if self.is_likely_binary(&path, Some(&sample_output)) {
            return ReadResult {
                is_binary: true,
                file_size,
                error: Some("Binary file — cannot display as text.".to_string()),
                ..Default::default()
            };
        }
        let cat_result = self.exec(&format!("cat {}", self.escape_shell_arg(&path)), None, None, None);
        if cat_result.exit_code != 0 {
            return ReadResult::error_msg(format!("Failed to read file: {}", cat_result.stdout));
        }
        ReadResult {
            content: strip_terminal_fence_leaks(&cat_result.stdout),
            file_size,
            ..Default::default()
        }
    }

    /// Delete a file via `rm`.
    pub fn delete_file(&mut self, path: &str) -> WriteResult {
        let path = self.expand_path(path);
        if is_write_denied(&path) {
            return WriteResult::error_msg(format!("Delete denied: {} is a protected path", path));
        }
        let result = self.exec(&format!("rm -f {}", self.escape_shell_arg(&path)), None, None, None);
        if result.exit_code != 0 {
            return WriteResult::error_msg(format!("Failed to delete {}: {}", path, result.stdout));
        }
        WriteResult::default()
    }

    /// Move a file via `mv`.
    pub fn move_file(&mut self, src: &str, dst: &str) -> WriteResult {
        let src = self.expand_path(src);
        let dst = self.expand_path(dst);
        for p in [&src, &dst] {
            if is_write_denied(p) {
                return WriteResult::error_msg(format!("Move denied: {} is a protected path", p));
            }
        }
        let result = self.exec(
            &format!(
                "mv {} {}",
                self.escape_shell_arg(&src),
                self.escape_shell_arg(&dst)
            ),
            None,
            None,
            None,
        );
        if result.exit_code != 0 {
            return WriteResult::error_msg(format!(
                "Failed to move {} -> {}: {}",
                src, dst, result.stdout
            ));
        }
        WriteResult::default()
    }

    // =========================================================================
    // WRITE
    // =========================================================================

    /// Write content to a file, creating parent directories as needed.
    pub fn write_file(&mut self, path: &str, content: &str) -> WriteResult {
        let path = self.expand_path(path);

        if is_write_denied(&path) {
            return WriteResult::error_msg(format!(
                "Write denied: '{}' is a protected system/credential file.",
                path
            ));
        }

        let ext = ext_lower(&path);
        let mut pre_content: Option<String> = None;
        if has_inproc_linter(&ext) {
            let read_cmd = format!("cat {} 2>/dev/null", self.escape_shell_arg(&path));
            let read_result = self.exec(&read_cmd, None, None, None);
            if read_result.exit_code == 0 && !read_result.stdout.is_empty() {
                pre_content = Some(read_result.stdout);
            }
        }

        let parent = dirname(&path);
        let mut dirs_created = false;
        if !parent.is_empty() {
            let mkdir_cmd = format!("mkdir -p {}", self.escape_shell_arg(&parent));
            let mkdir_result = self.exec(&mkdir_cmd, None, None, None);
            if mkdir_result.exit_code == 0 {
                dirs_created = true;
            }
        }

        let write_cmd = format!("cat > {}", self.escape_shell_arg(&path));
        let write_result = self.exec(&write_cmd, None, None, Some(content));
        if write_result.exit_code != 0 {
            return WriteResult::error_msg(format!("Failed to write file: {}", write_result.stdout));
        }

        let stat_cmd = format!("wc -c < {} 2>/dev/null", self.escape_shell_arg(&path));
        let stat_result = self.exec(&stat_cmd, None, None, None);
        let bytes_written: i64 = stat_result
            .stdout
            .trim()
            .parse()
            .unwrap_or_else(|_| content.as_bytes().len() as i64);

        let lint_result =
            self.check_lint_delta(&path, pre_content.as_deref(), Some(content));

        WriteResult {
            bytes_written,
            dirs_created,
            lint: Some(lint_result.to_json()),
            ..Default::default()
        }
    }

    // =========================================================================
    // PATCH (replace mode)
    // =========================================================================

    /// Replace text in a file using fuzzy matching.
    pub fn patch_replace(
        &mut self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> PatchResult {
        let path = self.expand_path(path);

        if is_write_denied(&path) {
            return PatchResult::error_msg(format!(
                "Write denied: '{}' is a protected system/credential file.",
                path
            ));
        }

        let read_cmd = format!("cat {} 2>/dev/null", self.escape_shell_arg(&path));
        let read_result = self.exec(&read_cmd, None, None, None);
        if read_result.exit_code != 0 {
            return PatchResult::error_msg(format!("Failed to read file: {}", path));
        }
        let content = read_result.stdout;

        let fr = fuzzy_find_and_replace(&content, old_string, new_string, replace_all);
        let new_content = fr.content;
        let match_count = fr.match_count;
        let error = fr.error;

        if error.is_some() || match_count == 0 {
            let mut err_msg = error
                .clone()
                .unwrap_or_else(|| format!("Could not find match for old_string in {}", path));
            // format_no_match_hint appends a hint when applicable.
            let hint =
                format_no_match_hint(Some(&err_msg), match_count, old_string, &content);
            err_msg.push_str(&hint);
            return PatchResult::error_msg(err_msg);
        }

        let write_result = self.write_file(&path, &new_content);
        if let Some(e) = write_result.error {
            return PatchResult::error_msg(format!("Failed to write changes: {}", e));
        }

        // Post-write verification.
        let verify_cmd = format!("cat {} 2>/dev/null", self.escape_shell_arg(&path));
        let verify_result = self.exec(&verify_cmd, None, None, None);
        if verify_result.exit_code != 0 {
            return PatchResult::error_msg(format!(
                "Post-write verification failed: could not re-read {}",
                path
            ));
        }
        if verify_result.stdout != new_content {
            return PatchResult::error_msg(format!(
                "Post-write verification failed for {}: on-disk content differs from intended write \
                 (wrote {} chars, read back {}). \
                 The patch did not persist. Re-read the file and try again.",
                path,
                new_content.chars().count(),
                verify_result.stdout.chars().count()
            ));
        }

        let diff = self.unified_diff_str(&content, &new_content, &path);
        let lint_result = self.check_lint_delta(&path, Some(&content), Some(&new_content));

        PatchResult {
            success: true,
            diff,
            files_modified: vec![path],
            lint: Some(lint_result.to_json()),
            ..Default::default()
        }
    }

    /// Apply a V4A format patch.
    pub fn patch_v4a(&mut self, patch_content: &str) -> PatchResult {
        let (operations, parse_error) = parse_v4a_patch(patch_content);
        if let Some(e) = parse_error {
            return PatchResult::error_msg(format!("Failed to parse patch: {}", e));
        }

        // Adapt this ShellFileOperations to the patch_parser FileOps trait via a
        // thin wrapper, and use the real fuzzy matcher.
        let mut adapter = V4aFileOpsAdapter { ops: self };
        let matcher = RealFuzzyMatcher;
        let pp_result = apply_v4a_operations(&operations, &mut adapter, &matcher);
        pp_result_to_patch_result(pp_result)
    }

    /// Run syntax check on a file after editing.
    pub fn check_lint(&mut self, path: &str, content: Option<&str>) -> LintResult {
        let ext = ext_lower(path);

        // In-process linter when available.
        if has_inproc_linter(&ext) {
            let owned_content;
            let content_str: &str = match content {
                Some(c) => c,
                None => {
                    let read_cmd = format!("cat {} 2>/dev/null", self.escape_shell_arg(path));
                    let read_result = self.exec(&read_cmd, None, None, None);
                    if read_result.exit_code != 0 {
                        return LintResult {
                            skipped: true,
                            message: format!("Failed to read {} for lint", path),
                            ..Default::default()
                        };
                    }
                    owned_content = read_result.stdout;
                    &owned_content
                }
            };
            if let Some((ok, err)) = run_inproc_linter(&ext, content_str) {
                if err == "__SKIP__" {
                    // Mirror Python: in-process linter unavailable. Python returns
                    // skipped here. (For .py / .toml our Rust in-proc linters
                    // always skip, so we fall through to the shell linter to
                    // preserve py_compile/etc. behaviour rather than skipping.)
                    if ext == ".py" {
                        // fall through to shell linter below
                    } else if ext == ".toml" {
                        return LintResult {
                            skipped: true,
                            message: format!(
                                "No linter available for {} (missing dependency)",
                                ext
                            ),
                            ..Default::default()
                        };
                    } else {
                        return LintResult {
                            skipped: true,
                            message: format!(
                                "No linter available for {} (missing dependency)",
                                ext
                            ),
                            ..Default::default()
                        };
                    }
                } else {
                    return LintResult {
                        success: ok,
                        output: if ok { String::new() } else { err },
                        ..Default::default()
                    };
                }
            }
        }

        // Shell linter fallback.
        let linter_cmd = match shell_linter_for(&ext) {
            Some(c) => c,
            None => {
                return LintResult {
                    skipped: true,
                    message: format!("No linter for {} files", ext),
                    ..Default::default()
                };
            }
        };
        let base_cmd = linter_cmd.split_whitespace().next().unwrap_or("");
        if !self.has_command(base_cmd) {
            return LintResult {
                skipped: true,
                message: format!("{} not available", base_cmd),
                ..Default::default()
            };
        }
        let cmd = linter_cmd.replace("{file}", &self.escape_shell_arg(path));
        let result = self.exec(&cmd, None, Some(30), None);
        let out = result.stdout.trim();
        LintResult {
            success: result.exit_code == 0,
            output: if out.is_empty() { String::new() } else { out.to_string() },
            ..Default::default()
        }
    }

    /// Run post-write lint with pre-write baseline comparison (delta refinement).
    pub fn check_lint_delta(
        &mut self,
        path: &str,
        pre_content: Option<&str>,
        post_content: Option<&str>,
    ) -> LintResult {
        let post = self.check_lint(path, post_content);

        if post.success || post.skipped {
            return post;
        }

        let pre_content = match pre_content {
            Some(c) => c,
            None => return post,
        };

        let pre = self.check_lint(path, Some(pre_content));
        if pre.success || pre.skipped || pre.output.is_empty() {
            return post;
        }

        let pre_lines: std::collections::HashSet<String> = pre
            .output
            .split('\n')
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let post_lines: Vec<String> = post
            .output
            .split('\n')
            .filter(|l| !l.trim().is_empty() && !pre_lines.contains(l.trim()))
            .map(|l| l.to_string())
            .collect();

        if post_lines.is_empty() {
            return LintResult {
                success: false,
                output: post.output.clone(),
                message:
                    "Pre-existing lint errors — this edit didn't introduce new ones but the file is still broken."
                        .to_string(),
                ..Default::default()
            };
        }

        LintResult {
            success: false,
            output: format!(
                "New lint errors introduced by this edit (pre-existing errors filtered out):\n{}",
                post_lines.join("\n")
            ),
            ..Default::default()
        }
    }

    // =========================================================================
    // SEARCH
    // =========================================================================

    /// Search for content or files.
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &mut self,
        pattern: &str,
        path: &str,
        target: &str,
        file_glob: Option<&str>,
        limit: i64,
        offset: i64,
        output_mode: &str,
        context: i64,
    ) -> SearchResult {
        let (offset, limit) = normalize_search_pagination(Some(offset), Some(limit));
        let path = self.expand_path(path);

        let check = self.exec(
            &format!(
                "test -e {} && echo exists || echo not_found",
                self.escape_shell_arg(&path)
            ),
            None,
            None,
            None,
        );
        if check.stdout.contains("not_found") {
            let parent = {
                let d = dirname(&path);
                if d.is_empty() { ".".to_string() } else { d }
            };
            let basename_query = basename(&path);
            let mut hint_parts = vec![format!("Path not found: {}", path)];
            let parent_check = self.exec(
                &format!(
                    "test -d {} && echo yes || echo no",
                    self.escape_shell_arg(&parent)
                ),
                None,
                None,
                None,
            );
            if parent_check.stdout.contains("yes") && !basename_query.is_empty() {
                let ls_result = self.exec(
                    &format!(
                        "ls -1 {} 2>/dev/null | head -20",
                        self.escape_shell_arg(&parent)
                    ),
                    None,
                    None,
                    None,
                );
                if ls_result.exit_code == 0 && !ls_result.stdout.trim().is_empty() {
                    let lower_q = basename_query.to_lowercase();
                    let lower_q3: String = lower_q.chars().take(3).collect();
                    let mut candidates: Vec<String> = Vec::new();
                    for entry in ls_result.stdout.trim().split('\n') {
                        if entry.is_empty() {
                            continue;
                        }
                        let le = entry.to_lowercase();
                        if le.contains(&lower_q)
                            || lower_q.contains(&le)
                            || le.starts_with(&lower_q3)
                        {
                            candidates.push(path_join(&parent, entry));
                        }
                    }
                    if !candidates.is_empty() {
                        let take: Vec<String> = candidates.into_iter().take(5).collect();
                        hint_parts.push(format!("Similar paths: {}", take.join(", ")));
                    }
                }
            }
            return SearchResult::error_with_count(hint_parts.join(". "), 0);
        }

        if target == "files" {
            self.search_files(pattern, &path, limit, offset)
        } else {
            self.search_content(pattern, &path, file_glob, limit, offset, output_mode, context)
        }
    }

    fn search_files(&mut self, pattern: &str, path: &str, limit: i64, offset: i64) -> SearchResult {
        let search_pattern = if !pattern.starts_with("**/") && !pattern.contains('/') {
            pattern.to_string()
        } else {
            pattern.rsplit('/').next().unwrap_or(pattern).to_string()
        };

        // has_hidden_path_ancestor: any path component (other than . / ..)
        // that starts with a dot.
        let has_hidden_path_ancestor = path
            .split('/')
            .any(|part| part != "." && part != ".." && part.starts_with('.'));

        if self.has_command("rg") {
            return self.search_files_rg(&search_pattern, path, limit, offset);
        }

        if !self.has_command("find") {
            return SearchResult {
                error: Some(
                    "File search requires 'rg' (ripgrep) or 'find'. \
                     Install ripgrep for best results: \
                     https://github.com/BurntSushi/ripgrep#installation"
                        .to_string(),
                ),
                ..Default::default()
            };
        }

        let hidden_filter_expr = if !has_hidden_path_ancestor {
            " -not -path '*/.*'".to_string()
        } else {
            String::new()
        };

        let pagination_expr = if !has_hidden_path_ancestor {
            format!(" | tail -n +{} | head -n {}", offset + 1, limit)
        } else {
            String::new()
        };

        let cmd = format!(
            "find {}{} -type f -name {} -printf '%T@ %p\\n' 2>/dev/null | sort -rn{}",
            self.escape_shell_arg(path),
            hidden_filter_expr,
            self.escape_shell_arg(&search_pattern),
            pagination_expr
        );
        let mut result = self.exec(&cmd, None, Some(60), None);

        if result.stdout.trim().is_empty() {
            let cmd_simple = format!(
                "find {}{} -type f -name {} 2>/dev/null | sort -rn{}",
                self.escape_shell_arg(path),
                hidden_filter_expr,
                self.escape_shell_arg(&search_pattern),
                pagination_expr
            );
            result = self.exec(&cmd_simple, None, Some(60), None);
        }

        let mut files: Vec<String> = Vec::new();
        for line in result.stdout.trim().split('\n') {
            if line.is_empty() {
                continue;
            }
            // line.split(' ', 1)
            if let Some((first, rest)) = line.split_once(' ') {
                let is_numeric = !first.is_empty()
                    && first.replace('.', "").chars().all(|c| c.is_ascii_digit())
                    && first.replace('.', "").chars().next().is_some();
                if is_numeric && !first.replace('.', "").is_empty() {
                    files.push(rest.to_string());
                } else {
                    files.push(line.to_string());
                }
            } else {
                files.push(line.to_string());
            }
        }

        if has_hidden_path_ancestor {
            // Apply descendant filtering after command execution.
            let mut filtered_files: Vec<String> = Vec::new();
            for file_path in &files {
                // We can't reliably resolve() inside the backend FS from here;
                // approximate the Python relative-to behaviour by stripping the
                // search root prefix and inspecting the remaining components.
                let rel = file_path
                    .strip_prefix(path)
                    .map(|s| s.trim_start_matches('/'))
                    .unwrap_or(file_path);
                let has_hidden = rel
                    .split('/')
                    .any(|part| part != "." && part != ".." && part.starts_with('.'));
                if has_hidden {
                    continue;
                }
                filtered_files.push(file_path.clone());
            }
            let start = offset as usize;
            let end = (offset + limit) as usize;
            files = filtered_files
                .into_iter()
                .skip(start)
                .take(end.saturating_sub(start))
                .collect();
        }

        let total = files.len() as i64;
        SearchResult {
            files,
            total_count: total,
            ..Default::default()
        }
    }

    fn search_files_rg(&mut self, pattern: &str, path: &str, limit: i64, offset: i64) -> SearchResult {
        let glob_pattern = if !pattern.contains('/') && !pattern.starts_with('*') {
            format!("*{}", pattern)
        } else {
            pattern.to_string()
        };

        let fetch_limit = limit + offset;
        let cmd_sorted = format!(
            "rg --files --sortr=modified -g {} {} 2>/dev/null | head -n {}",
            self.escape_shell_arg(&glob_pattern),
            self.escape_shell_arg(path),
            fetch_limit
        );
        let mut result = self.exec(&cmd_sorted, None, Some(60), None);
        let mut all_files: Vec<String> = result
            .stdout
            .trim()
            .split('\n')
            .filter(|f| !f.is_empty())
            .map(|s| s.to_string())
            .collect();

        if all_files.is_empty() {
            let cmd_plain = format!(
                "rg --files -g {} {} 2>/dev/null | head -n {}",
                self.escape_shell_arg(&glob_pattern),
                self.escape_shell_arg(path),
                fetch_limit
            );
            result = self.exec(&cmd_plain, None, Some(60), None);
            all_files = result
                .stdout
                .trim()
                .split('\n')
                .filter(|f| !f.is_empty())
                .map(|s| s.to_string())
                .collect();
        }

        let total = all_files.len() as i64;
        let start = offset as usize;
        let end = (offset + limit) as usize;
        let page: Vec<String> = all_files
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .cloned()
            .collect();

        SearchResult {
            files: page,
            total_count: total,
            truncated: total >= fetch_limit,
            ..Default::default()
        }
    }

    fn search_content(
        &mut self,
        pattern: &str,
        path: &str,
        file_glob: Option<&str>,
        limit: i64,
        offset: i64,
        output_mode: &str,
        context: i64,
    ) -> SearchResult {
        if self.has_command("rg") {
            self.search_with_rg(pattern, path, file_glob, limit, offset, output_mode, context)
        } else if self.has_command("grep") {
            self.search_with_grep(pattern, path, file_glob, limit, offset, output_mode, context)
        } else {
            SearchResult {
                error: Some(
                    "Content search requires ripgrep (rg) or grep. \
                     Install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
                        .to_string(),
                ),
                ..Default::default()
            }
        }
    }

    fn search_with_rg(
        &mut self,
        pattern: &str,
        path: &str,
        file_glob: Option<&str>,
        limit: i64,
        offset: i64,
        output_mode: &str,
        context: i64,
    ) -> SearchResult {
        let mut cmd_parts: Vec<String> = vec![
            "rg".into(),
            "--line-number".into(),
            "--no-heading".into(),
            "--with-filename".into(),
        ];
        if context > 0 {
            cmd_parts.push("-C".into());
            cmd_parts.push(context.to_string());
        }
        if let Some(g) = file_glob {
            cmd_parts.push("--glob".into());
            cmd_parts.push(self.escape_shell_arg(g));
        }
        if output_mode == "files_only" {
            cmd_parts.push("-l".into());
        } else if output_mode == "count" {
            cmd_parts.push("-c".into());
        }
        cmd_parts.push(self.escape_shell_arg(pattern));
        cmd_parts.push(self.escape_shell_arg(path));

        let fetch_limit = if context > 0 {
            limit + offset + 200
        } else {
            limit + offset
        };
        cmd_parts.push("|".into());
        cmd_parts.push("head".into());
        cmd_parts.push("-n".into());
        cmd_parts.push(fetch_limit.to_string());

        let cmd = cmd_parts.join(" ");
        let result = self.exec(&cmd, None, Some(60), None);

        if result.exit_code == 2 && result.stdout.trim().is_empty() {
            let error_msg = result
                .stderr
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("Search error");
            return SearchResult::error_with_count(format!("Search failed: {}", error_msg), 0);
        }

        self.parse_search_output(&result.stdout, output_mode, limit, offset, context)
    }

    fn search_with_grep(
        &mut self,
        pattern: &str,
        path: &str,
        file_glob: Option<&str>,
        limit: i64,
        offset: i64,
        output_mode: &str,
        context: i64,
    ) -> SearchResult {
        let mut cmd_parts: Vec<String> = vec!["grep".into(), "-rnH".into()];
        cmd_parts.push("--exclude-dir='.*'".into());
        if context > 0 {
            cmd_parts.push("-C".into());
            cmd_parts.push(context.to_string());
        }
        if let Some(g) = file_glob {
            cmd_parts.push("--include".into());
            cmd_parts.push(self.escape_shell_arg(g));
        }
        if output_mode == "files_only" {
            cmd_parts.push("-l".into());
        } else if output_mode == "count" {
            cmd_parts.push("-c".into());
        }
        cmd_parts.push(self.escape_shell_arg(pattern));
        cmd_parts.push(self.escape_shell_arg(path));

        let fetch_limit = limit + offset + if context > 0 { 200 } else { 0 };
        cmd_parts.push("|".into());
        cmd_parts.push("head".into());
        cmd_parts.push("-n".into());
        cmd_parts.push(fetch_limit.to_string());

        let cmd = cmd_parts.join(" ");
        let result = self.exec(&cmd, None, Some(60), None);

        if result.exit_code == 2 && result.stdout.trim().is_empty() {
            let error_msg = result
                .stderr
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("Search error");
            return SearchResult::error_with_count(format!("Search failed: {}", error_msg), 0);
        }

        self.parse_search_output(&result.stdout, output_mode, limit, offset, context)
    }

    /// Shared parser for rg/grep stdout — both backends emit identical formats.
    fn parse_search_output(
        &self,
        stdout: &str,
        output_mode: &str,
        limit: i64,
        offset: i64,
        context: i64,
    ) -> SearchResult {
        if output_mode == "files_only" {
            let all_files: Vec<String> = stdout
                .trim()
                .split('\n')
                .filter(|f| !f.is_empty())
                .map(|s| s.to_string())
                .collect();
            let total = all_files.len() as i64;
            let start = offset as usize;
            let end = (offset + limit) as usize;
            let page: Vec<String> = all_files
                .into_iter()
                .skip(start)
                .take(end.saturating_sub(start))
                .collect();
            return SearchResult {
                files: page,
                total_count: total,
                ..Default::default()
            };
        }

        if output_mode == "count" {
            let mut counts: Vec<(String, i64)> = Vec::new();
            let mut sum: i64 = 0;
            for line in stdout.trim().split('\n') {
                if line.contains(':') {
                    if let Some(idx) = line.rfind(':') {
                        let key = &line[..idx];
                        let val_str = &line[idx + 1..];
                        if let Ok(v) = val_str.parse::<i64>() {
                            counts.push((key.to_string(), v));
                            sum += v;
                        }
                    }
                }
            }
            return SearchResult {
                counts,
                total_count: sum,
                ..Default::default()
            };
        }

        // content mode
        let match_re = {
            static RE: OnceLock<Regex> = OnceLock::new();
            RE.get_or_init(|| Regex::new(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$").unwrap())
        };
        let mut matches: Vec<SearchMatch> = Vec::new();
        for line in stdout.trim().split('\n') {
            if line.is_empty() || line == "--" {
                continue;
            }
            if let Some(caps) = match_re.captures(line) {
                let drive = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let path_part = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                let lineno: i64 = caps
                    .get(3)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(0);
                let content_part = caps.get(4).map(|m| m.as_str()).unwrap_or("");
                matches.push(SearchMatch {
                    path: format!("{}{}", drive, path_part),
                    line_number: lineno,
                    content: take_chars(content_part, 500),
                    mtime: 0.0,
                });
                continue;
            }
            if context > 0 {
                if let Some((p, ln, c)) = parse_search_context_line(line) {
                    matches.push(SearchMatch {
                        path: p,
                        line_number: ln,
                        content: take_chars(&c, 500),
                        mtime: 0.0,
                    });
                }
            }
        }

        let total = matches.len() as i64;
        let start = offset as usize;
        let end = (offset + limit) as usize;
        let page: Vec<SearchMatch> = matches
            .into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect();
        SearchResult {
            matches: page,
            total_count: total,
            truncated: total > offset + limit,
            ..Default::default()
        }
    }
}

/// Truncate a string to at most `n` Unicode code points (mirrors Python `s[:n]`).
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// =============================================================================
// V4A integration adapters
// =============================================================================

/// Adapts a [`ShellFileOperations`] to the patch_parser [`FileOps`] trait.
struct V4aFileOpsAdapter<'a, E: TerminalEnv> {
    ops: &'a mut ShellFileOperations<E>,
}

impl<'a, E: TerminalEnv> FileOps for V4aFileOpsAdapter<'a, E> {
    fn read_file_raw(&self, path: &str) -> FileOpResult {
        // read_file_raw needs &mut self in our impl (uses exec which is &self,
        // but the public method takes &mut self). We work around by calling the
        // immutable exec directly here, replicating read_file_raw's success
        // path without the command cache mutation.
        let stat_cmd = format!("wc -c < {} 2>/dev/null", self.ops.escape_shell_arg(path));
        let stat_result = self.ops.exec(&stat_cmd, None, None, None);
        if stat_result.exit_code != 0 {
            // mirror suggest_similar_files error shape (just an error string)
            return FileOpResult::err(format!("File not found: {}", path));
        }
        if self.ops.is_image(path) {
            return FileOpResult::ok(String::new());
        }
        let sample_result = self.ops.exec(
            &format!("head -c 1000 {} 2>/dev/null", self.ops.escape_shell_arg(path)),
            None,
            None,
            None,
        );
        let sample_output = strip_terminal_fence_leaks(&sample_result.stdout);
        if self.ops.is_likely_binary(path, Some(&sample_output)) {
            return FileOpResult::err("Binary file — cannot display as text.".to_string());
        }
        let cat_result =
            self.ops
                .exec(&format!("cat {}", self.ops.escape_shell_arg(path)), None, None, None);
        if cat_result.exit_code != 0 {
            return FileOpResult::err(format!("Failed to read file: {}", cat_result.stdout));
        }
        FileOpResult::ok(strip_terminal_fence_leaks(&cat_result.stdout))
    }

    fn write_file(&mut self, path: &str, content: &str) -> FileOpResult {
        let wr = self.ops.write_file(path, content);
        match wr.error {
            Some(e) => FileOpResult::err(e),
            None => FileOpResult::ok(String::new()),
        }
    }

    fn delete_file(&mut self, path: &str) -> FileOpResult {
        let wr = self.ops.delete_file(path);
        match wr.error {
            Some(e) => FileOpResult::err(e),
            None => FileOpResult::ok(String::new()),
        }
    }

    fn move_file(&mut self, src: &str, dst: &str) -> FileOpResult {
        let wr = self.ops.move_file(src, dst);
        match wr.error {
            Some(e) => FileOpResult::err(e),
            None => FileOpResult::ok(String::new()),
        }
    }
}

/// Real fuzzy matcher backed by `crate::tool_fuzzy_match`.
struct RealFuzzyMatcher;

impl FuzzyMatcher for RealFuzzyMatcher {
    fn fuzzy_find_and_replace(
        &self,
        content: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> PpFuzzyResult {
        let r = fuzzy_find_and_replace(content, old_string, new_string, replace_all);
        PpFuzzyResult {
            new_content: r.content,
            count: r.match_count,
            strategy: r.strategy,
            error: r.error,
        }
    }

    fn format_no_match_hint(
        &self,
        error: Option<&str>,
        count: usize,
        search_pattern: &str,
        content: &str,
    ) -> String {
        format_no_match_hint(error, count, search_pattern, content)
    }
}

/// Convert a patch_parser `PatchResult` into the file_operations `PatchResult`.
fn pp_result_to_patch_result(pp: crate::tool_patch_parser::PatchResult) -> PatchResult {
    let lint = pp.lint.map(|pairs| {
        let mut m = serde_json::Map::new();
        for (k, v) in pairs {
            // The patch_parser stores serialized lint strings; try to parse
            // them back to JSON, else store the raw string.
            let val = serde_json::from_str::<serde_json::Value>(&v)
                .unwrap_or(serde_json::Value::String(v));
            m.insert(k, val);
        }
        serde_json::Value::Object(m)
    });
    PatchResult {
        success: pp.success,
        diff: pp.diff,
        files_modified: pp.files_modified,
        files_created: pp.files_created,
        files_deleted: pp.files_deleted,
        lint,
        error: pp.error,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A scriptable fake terminal: maps command -> (stdout, exit_code).
    /// Also records a fake in-memory FS for write/read round-trips.
    struct FakeEnv {
        responses: RefCell<Vec<(String, ExecuteResult)>>,
        cwd: Option<String>,
    }

    impl FakeEnv {
        fn new() -> Self {
            FakeEnv {
                responses: RefCell::new(Vec::new()),
                cwd: Some("/work".to_string()),
            }
        }
        fn push(&self, substr: &str, stdout: &str, code: i64) {
            self.responses.borrow_mut().push((
                substr.to_string(),
                ExecuteResult {
                    stdout: stdout.to_string(),
                    exit_code: code,
                    stderr: None,
                },
            ));
        }
    }

    impl TerminalEnv for FakeEnv {
        fn execute(
            &self,
            command: &str,
            _cwd: &str,
            _timeout: Option<u64>,
            _stdin_data: Option<&str>,
        ) -> ExecuteResult {
            for (substr, resp) in self.responses.borrow().iter() {
                if command.contains(substr.as_str()) {
                    return resp.clone();
                }
            }
            ExecuteResult::default()
        }
        fn cwd(&self) -> Option<String> {
            self.cwd.clone()
        }
    }

    #[test]
    fn test_escape_shell_arg() {
        let env = FakeEnv::new();
        let ops = ShellFileOperations::new(env, None, None);
        assert_eq!(ops.escape_shell_arg("simple"), "'simple'");
        assert_eq!(ops.escape_shell_arg("it's"), "'it'\"'\"'s'");
    }

    #[test]
    fn test_strip_terminal_fence_leaks() {
        // A pure fence-marker line is dropped.
        let input = "__HERMES_FENCE_abc123__\nreal content\n";
        let out = strip_terminal_fence_leaks(input);
        assert!(out.contains("real content"));
        assert!(!out.contains("HERMES_FENCE"));
    }

    #[test]
    fn test_strip_empty() {
        assert_eq!(strip_terminal_fence_leaks(""), "");
    }

    #[test]
    fn test_parse_search_context_line_rightmost() {
        // dir/file-12-name.py-8-context => path dir/file-12-name.py, line 8
        let (p, n, c) = parse_search_context_line("dir/file-12-name.py-8-context").unwrap();
        assert_eq!(p, "dir/file-12-name.py");
        assert_eq!(n, 8);
        assert_eq!(c, "context");
    }

    #[test]
    fn test_parse_search_context_line_none() {
        assert!(parse_search_context_line("--").is_none());
        assert!(parse_search_context_line("").is_none());
        assert!(parse_search_context_line("no-separators-here").is_none());
    }

    #[test]
    fn test_normalize_read_pagination() {
        let (o, l) = normalize_read_pagination(Some(0), Some(10));
        assert_eq!(o, 1); // clamped to >= 1
        assert_eq!(l, 10);
        let (o2, l2) = normalize_read_pagination(Some(-5), Some(0));
        assert_eq!(o2, 1);
        assert_eq!(l2, 1); // limit clamped to >= 1
        let (o3, l3) = normalize_read_pagination(None, None);
        assert_eq!(o3, DEFAULT_READ_OFFSET);
        assert_eq!(l3, DEFAULT_READ_LIMIT);
    }

    #[test]
    fn test_normalize_search_pagination() {
        let (o, l) = normalize_search_pagination(Some(-3), Some(-1));
        assert_eq!(o, 0);
        assert_eq!(l, 1);
    }

    #[test]
    fn test_splitext_and_helpers() {
        assert_eq!(splitext("foo.py"), ("foo".into(), ".py".into()));
        assert_eq!(splitext("dir/foo.tar.gz"), ("dir/foo.tar".into(), ".gz".into()));
        assert_eq!(splitext(".bashrc"), (".bashrc".into(), "".into()));
        assert_eq!(splitext("noext"), ("noext".into(), "".into()));
        assert_eq!(basename("a/b/c.txt"), "c.txt");
        assert_eq!(dirname("a/b/c.txt"), "a/b");
        assert_eq!(dirname("file"), "");
        assert_eq!(path_join("a/b", "c"), "a/b/c");
        assert_eq!(path_join("", "c"), "c");
    }

    #[test]
    fn test_lint_json_inproc() {
        assert!(lint_json_inproc("{\"a\": 1}").0);
        assert!(!lint_json_inproc("{not json").0);
    }

    #[test]
    fn test_lint_yaml_inproc() {
        assert!(lint_yaml_inproc("a: 1\nb: 2\n").0);
        assert!(!lint_yaml_inproc("a: [1, 2\n  bad").0);
    }

    #[test]
    fn test_read_file_not_found_suggests() {
        let env = FakeEnv::new();
        // wc -c fails (exit 1) -> suggest_similar_files
        env.push("wc -c <", "", 1);
        env.push("ls -1", "config.yaml\nother.txt\n", 0);
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.read_file("/x/config.yml", 1, 500);
        assert!(r.error.as_deref().unwrap().contains("File not found"));
        // config.yaml shares basename "config" -> high score suggestion
        assert!(r.similar_files.iter().any(|f| f.contains("config.yaml")));
    }

    #[test]
    fn test_read_file_image_redirect() {
        let env = FakeEnv::new();
        env.push("wc -c <", "12345\n", 0);
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.read_file("/x/pic.png", 1, 500);
        assert!(r.is_image);
        assert!(r.is_binary);
        assert_eq!(r.file_size, 12345);
        assert!(r.hint.unwrap().contains("vision_analyze"));
    }

    #[test]
    fn test_read_file_content_with_line_numbers() {
        let env = FakeEnv::new();
        env.push("wc -c <", "20\n", 0);
        env.push("head -c 1000", "hello\nworld\n", 0);
        env.push("sed -n", "hello\nworld", 0);
        env.push("wc -l <", "2\n", 0);
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.read_file("/x/a.txt", 1, 500);
        assert!(r.error.is_none());
        assert!(r.content.contains("|hello"));
        assert!(r.content.contains("|world"));
        assert_eq!(r.total_lines, 2);
        assert!(!r.truncated);
    }

    #[test]
    fn test_write_denied() {
        let env = FakeEnv::new();
        let mut ops = ShellFileOperations::new(env, None, None);
        // /etc/passwd is on the static deny list.
        let r = ops.write_file("/etc/passwd", "x");
        assert!(r.error.as_deref().unwrap().contains("Write denied"));
    }

    #[test]
    fn test_delete_denied() {
        let env = FakeEnv::new();
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.delete_file("/etc/shadow");
        assert!(r.error.as_deref().unwrap().contains("Delete denied"));
    }

    #[test]
    fn test_search_path_not_found() {
        let env = FakeEnv::new();
        env.push("test -e", "not_found\n", 0);
        env.push("test -d", "yes\n", 0);
        env.push("ls -1", "alpha.py\nbeta.py\n", 0);
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.search("pat", "/dir/alph", "content", None, 50, 0, "content", 0);
        assert!(r.error.is_some());
        assert!(r.error.unwrap().contains("Path not found"));
    }

    #[test]
    fn test_search_content_rg_parse() {
        let env = FakeEnv::new();
        env.push("test -e", "exists\n", 0);
        // has_command rg -> yes
        env.push("command -v rg", "yes\n", 0);
        env.push(
            "rg --line-number",
            "src/a.py:10:found it\nsrc/b.py:20:found again\n",
            0,
        );
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.search("found", "/repo", "content", None, 50, 0, "content", 0);
        assert_eq!(r.total_count, 2);
        assert_eq!(r.matches.len(), 2);
        assert_eq!(r.matches[0].path, "src/a.py");
        assert_eq!(r.matches[0].line_number, 10);
        assert_eq!(r.matches[0].content, "found it");
    }

    #[test]
    fn test_search_count_mode() {
        let env = FakeEnv::new();
        env.push("test -e", "exists\n", 0);
        env.push("command -v rg", "yes\n", 0);
        env.push("rg --line-number", "src/a.py:3\nsrc/b.py:5\n", 0);
        let mut ops = ShellFileOperations::new(env, None, None);
        let r = ops.search("x", "/repo", "content", None, 50, 0, "count", 0);
        assert_eq!(r.total_count, 8);
        assert_eq!(r.counts.len(), 2);
    }

    #[test]
    fn test_to_json_shapes() {
        let rr = ReadResult {
            content: "abc".into(),
            total_lines: 3,
            file_size: 10,
            ..Default::default()
        };
        let j = rr.to_json();
        assert_eq!(j["content"], "abc");
        assert_eq!(j["total_lines"], 3);
        // hint is None -> omitted
        assert!(j.get("hint").is_none());

        let lr = LintResult {
            skipped: true,
            message: "no linter".into(),
            ..Default::default()
        };
        assert_eq!(lr.to_json()["status"], "skipped");

        let lr2 = LintResult {
            success: false,
            output: "err".into(),
            ..Default::default()
        };
        assert_eq!(lr2.to_json()["status"], "error");
    }

    #[test]
    fn test_is_likely_binary_extension() {
        let env = FakeEnv::new();
        let ops = ShellFileOperations::new(env, None, None);
        // .png is a binary extension
        assert!(ops.is_likely_binary("a.png", None));
        // plain text sample
        assert!(!ops.is_likely_binary("a.txt", Some("hello world")));
        // mostly control chars
        let bin: String = (0..100).map(|_| '\u{1}').collect();
        assert!(ops.is_likely_binary("a.dat", Some(&bin)));
    }
}
