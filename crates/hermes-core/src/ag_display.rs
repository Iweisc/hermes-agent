//! CLI presentation -- spinner, kawaii faces, tool preview formatting.
//!
//! Pure display functions and types with no agent dependency. This is a native
//! Rust port of `agent/display.py`. It reproduces the tool-preview one-liners,
//! the cute completion lines, unified-diff rendering, and the `KawaiiSpinner`
//! animation used by the CLI during tool execution.
//!
//! Skin-engine lookups in the Python original are resolved lazily from a global
//! skin singleton; that singleton is not available in this crate, so the
//! defaults (the fallback branches) are used. Hooks are provided
//! ([`set_skin_provider`]) so an integrator can wire a real skin source later.

use std::io::{IsTerminal, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::Value;

// ANSI escape codes for coloring tool failure indicators.
pub const RED: &str = "\x1b[31m";
pub const RESET: &str = "\x1b[0m";

pub const ANSI_RESET: &str = "\x1b[0m";

const MAX_INLINE_DIFF_FILES: usize = 6;
const MAX_INLINE_DIFF_LINES: usize = 80;

// =========================================================================
// Skin provider hook
// =========================================================================

/// Minimal skin description consumed by the display layer.
///
/// The Python code resolves these values from `hermes_cli.skin_engine`. Since
/// that engine is not part of this crate, an integrator may install a provider
/// returning a [`SkinInfo`]; otherwise the documented defaults are used.
#[derive(Clone, Debug, Default)]
pub struct SkinInfo {
    /// Tool output prefix character (default `┊`).
    pub tool_prefix: Option<String>,
    /// Per-tool emoji overrides keyed by tool name.
    pub tool_emojis: std::collections::HashMap<String, String>,
    /// `banner_dim` color as a `#rrggbb` hex string.
    pub banner_dim: Option<String>,
    pub session_label: Option<String>,
    pub session_border: Option<String>,
    pub ui_error: Option<String>,
    pub ui_ok: Option<String>,
    /// Spinner overrides.
    pub waiting_faces: Vec<String>,
    pub thinking_faces: Vec<String>,
    pub thinking_verbs: Vec<String>,
    /// Spinner "wings" pairs `(left, right)`.
    pub spinner_wings: Vec<(String, String)>,
}

type SkinProvider = dyn Fn() -> Option<SkinInfo> + Send + Sync;

static SKIN_PROVIDER: OnceLock<Box<SkinProvider>> = OnceLock::new();

/// Install a callback that resolves the active skin. Only the first call wins,
/// matching the lazy/once nature of the Python import.
pub fn set_skin_provider<F>(provider: F)
where
    F: Fn() -> Option<SkinInfo> + Send + Sync + 'static,
{
    let _ = SKIN_PROVIDER.set(Box::new(provider));
}

fn get_skin() -> Option<SkinInfo> {
    SKIN_PROVIDER.get().and_then(|f| f())
}

// =========================================================================
// Diff colors — resolved lazily from the skin engine so they adapt to themes.
// =========================================================================

#[derive(Clone, Debug)]
pub struct DiffColors {
    pub dim: String,
    pub file: String,
    pub hunk: String,
    pub minus: String,
    pub plus: String,
}

static DIFF_COLORS_CACHED: OnceLock<DiffColors> = OnceLock::new();

fn hex_fg(hex: &Option<String>, fallback: (u8, u8, u8)) -> String {
    if let Some(h) = hex {
        if h.len() == 7 && h.starts_with('#') {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&h[1..3], 16),
                u8::from_str_radix(&h[3..5], 16),
                u8::from_str_radix(&h[5..7], 16),
            ) {
                return format!("\x1b[38;2;{r};{g};{b}m");
            }
        }
    }
    let (r, g, b) = fallback;
    format!("\x1b[38;2;{r};{g};{b}m")
}

fn parse_hex(hex: &Option<String>) -> Option<(u8, u8, u8)> {
    let h = hex.as_ref()?;
    if h.len() == 7 && h.starts_with('#') {
        let r = u8::from_str_radix(&h[1..3], 16).ok()?;
        let g = u8::from_str_radix(&h[3..5], 16).ok()?;
        let b = u8::from_str_radix(&h[5..7], 16).ok()?;
        Some((r, g, b))
    } else {
        None
    }
}

/// Return ANSI escapes for diff display, resolved from the active skin.
pub fn diff_ansi() -> DiffColors {
    if let Some(cached) = DIFF_COLORS_CACHED.get() {
        return cached.clone();
    }

    // Defaults that work on dark terminals.
    let mut dim = "\x1b[38;2;150;150;150m".to_string();
    let mut file_c = "\x1b[38;2;180;160;255m".to_string();
    let mut hunk = "\x1b[38;2;120;120;140m".to_string();
    let mut minus = "\x1b[38;2;255;255;255;48;2;120;20;20m".to_string();
    let mut plus = "\x1b[38;2;255;255;255;48;2;20;90;20m".to_string();

    if let Some(skin) = get_skin() {
        dim = hex_fg(&skin.banner_dim, (150, 150, 150));
        file_c = hex_fg(&skin.session_label, (180, 160, 255));
        hunk = hex_fg(&skin.session_border, (120, 120, 140));
        // minus/plus use background colors — derive from ui_error/ui_ok.
        let err = skin.ui_error.clone().or_else(|| Some("#ef5350".to_string()));
        let ok = skin.ui_ok.clone().or_else(|| Some("#4caf50".to_string()));
        if let Some((er, eg, eb)) = parse_hex(&err) {
            minus = format!(
                "\x1b[38;2;255;255;255;48;2;{};{};{}m",
                (er / 2).max(20),
                (eg / 4).max(10),
                (eb / 4).max(10)
            );
        }
        if let Some((or_, og, ob)) = parse_hex(&ok) {
            plus = format!(
                "\x1b[38;2;255;255;255;48;2;{};{};{}m",
                (or_ / 4).max(10),
                (og / 2).max(20),
                (ob / 4).max(10)
            );
        }
    }

    let colors = DiffColors {
        dim,
        file: file_c,
        hunk,
        minus,
        plus,
    };
    let _ = DIFF_COLORS_CACHED.set(colors.clone());
    colors
}

fn diff_dim() -> String {
    diff_ansi().dim
}
fn diff_file() -> String {
    diff_ansi().file
}
fn diff_hunk() -> String {
    diff_ansi().hunk
}
fn diff_minus() -> String {
    diff_ansi().minus
}
fn diff_plus() -> String {
    diff_ansi().plus
}

// =========================================================================
// LocalEditSnapshot
// =========================================================================

/// Pre-tool filesystem snapshot used to render diffs locally after writes.
#[derive(Clone, Debug, Default)]
pub struct LocalEditSnapshot {
    pub paths: Vec<PathBuf>,
    /// `None` value means the file did not exist / was unreadable.
    pub before: std::collections::HashMap<String, Option<String>>,
}

// =========================================================================
// Configurable tool preview length (0 = no limit)
// =========================================================================

static TOOL_PREVIEW_MAX_LEN: AtomicI64 = AtomicI64::new(0);

/// Set the global max length for tool call previews. 0 = no limit.
pub fn set_tool_preview_max_len(n: i64) {
    let v = if n != 0 { n.max(0) } else { 0 };
    TOOL_PREVIEW_MAX_LEN.store(v, Ordering::Relaxed);
}

/// Return the configured max preview length (0 = unlimited).
pub fn get_tool_preview_max_len() -> i64 {
    TOOL_PREVIEW_MAX_LEN.load(Ordering::Relaxed)
}

// =========================================================================
// Skin-aware helpers
// =========================================================================

/// Get tool output prefix character from active skin.
pub fn get_skin_tool_prefix() -> String {
    if let Some(skin) = get_skin() {
        if let Some(p) = skin.tool_prefix {
            return p;
        }
    }
    "┊".to_string()
}

/// Get the display emoji for a tool.
///
/// Resolution order:
/// 1. Active skin's `tool_emojis` overrides
/// 2. (registry default — not available in this crate)
/// 3. *default* fallback
pub fn get_tool_emoji(tool_name: &str, default: &str) -> String {
    if let Some(skin) = get_skin() {
        if let Some(override_emoji) = skin.tool_emojis.get(tool_name) {
            if !override_emoji.is_empty() {
                return override_emoji.clone();
            }
        }
    }
    default.to_string()
}

// =========================================================================
// Tool preview (one-line summary of a tool call's primary argument)
// =========================================================================

/// Collapse whitespace (including newlines) to single spaces.
fn oneline(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Truncate `s` to at most `n` chars (by Unicode scalar), counting like Python
/// slicing on `str`.
fn char_slice(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Python-style `s[start..]` from the end: last `n` chars.
fn char_slice_tail(s: &str, n: usize) -> String {
    let total = char_len(s);
    let skip = total.saturating_sub(n);
    s.chars().skip(skip).collect()
}

fn arg_str<'a>(args: &'a Value, key: &str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Build a short preview of a tool call's primary argument for display.
///
/// `max_len` of `None` defers to the global preview length; `0` means unlimited.
pub fn build_tool_preview(tool_name: &str, args: &Value, max_len: Option<i64>) -> Option<String> {
    let max_len = max_len.unwrap_or_else(get_tool_preview_max_len);
    let obj = args.as_object()?;
    if obj.is_empty() {
        return None;
    }

    if tool_name == "process" {
        let action = arg_str(args, "action");
        let sid = arg_str(args, "session_id");
        let data = arg_str(args, "data");
        let timeout_val = args.get("timeout");
        let mut parts: Vec<String> = vec![action.to_string()];
        if !sid.is_empty() {
            parts.push(char_slice(sid, 16));
        }
        if !data.is_empty() {
            parts.push(format!("\"{}\"", oneline(&char_slice(data, 20))));
        }
        if let Some(t) = timeout_val {
            let truthy = !t.is_null()
                && t.as_bool() != Some(false)
                && t.as_i64() != Some(0)
                && t.as_str() != Some("");
            if truthy && action == "wait" {
                let tstr = match t {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                parts.push(format!("{tstr}s"));
            }
        }
        return if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        };
    }

    if tool_name == "todo" {
        let todos_arg = args.get("todos");
        let merge = args.get("merge").and_then(Value::as_bool).unwrap_or(false);
        return match todos_arg {
            None | Some(Value::Null) => Some("reading task list".to_string()),
            Some(v) => {
                let len = v.as_array().map(|a| a.len()).unwrap_or(0);
                if merge {
                    Some(format!("updating {len} task(s)"))
                } else {
                    Some(format!("planning {len} task(s)"))
                }
            }
        };
    }

    if tool_name == "session_search" {
        let query = oneline(arg_str(args, "query"));
        let ell = if char_len(&query) > 25 { "..." } else { "" };
        return Some(format!("recall: \"{}{}\"", char_slice(&query, 25), ell));
    }

    if tool_name == "memory" {
        let action = arg_str(args, "action");
        let target = arg_str(args, "target");
        match action {
            "add" => {
                let content = oneline(arg_str(args, "content"));
                let ell = if char_len(&content) > 25 { "..." } else { "" };
                return Some(format!(
                    "+{target}: \"{}{}\"",
                    char_slice(&content, 25),
                    ell
                ));
            }
            "replace" => {
                let raw = oneline(arg_str(args, "old_text"));
                let old = if raw.is_empty() {
                    "<missing old_text>".to_string()
                } else {
                    raw
                };
                return Some(format!("~{target}: \"{}\"", char_slice(&old, 20)));
            }
            "remove" => {
                let raw = oneline(arg_str(args, "old_text"));
                let old = if raw.is_empty() {
                    "<missing old_text>".to_string()
                } else {
                    raw
                };
                return Some(format!("-{target}: \"{}\"", char_slice(&old, 20)));
            }
            _ => return Some(action.to_string()),
        }
    }

    if tool_name == "send_message" {
        let target = args.get("target").and_then(Value::as_str).unwrap_or("?");
        let mut msg = oneline(arg_str(args, "message"));
        if char_len(&msg) > 20 {
            msg = format!("{}...", char_slice(&msg, 17));
        }
        return Some(format!("to {target}: \"{msg}\""));
    }

    if let Some(stripped) = tool_name.strip_prefix("rl_") {
        let _ = stripped;
        let result = match tool_name {
            "rl_list_environments" => Some("listing envs".to_string()),
            "rl_select_environment" => Some(arg_str(args, "name").to_string()),
            "rl_get_current_config" => Some("reading config".to_string()),
            "rl_edit_config" => Some(format!(
                "{}={}",
                arg_str(args, "field"),
                arg_str(args, "value")
            )),
            "rl_start_training" => Some("starting".to_string()),
            "rl_check_status" => Some(char_slice(arg_str(args, "run_id"), 16)),
            "rl_stop_training" => Some(format!("stopping {}", char_slice(arg_str(args, "run_id"), 16))),
            "rl_get_results" => Some(char_slice(arg_str(args, "run_id"), 16)),
            "rl_list_runs" => Some("listing runs".to_string()),
            "rl_test_inference" => {
                let steps = args.get("num_steps").and_then(Value::as_i64).unwrap_or(3);
                Some(format!("{steps} steps"))
            }
            _ => None,
        };
        return result;
    }

    let primary = primary_arg_key(tool_name);
    let mut key: Option<&str> = primary;
    if key.is_none() {
        for fallback in [
            "query", "text", "command", "path", "name", "prompt", "code", "goal",
        ] {
            if obj.contains_key(fallback) {
                key = Some(fallback);
                break;
            }
        }
    }

    let key = key?;
    let value = obj.get(key)?;

    // If list, take first element (or "").
    let value_str: String = match value {
        Value::Array(items) => match items.first() {
            Some(Value::String(s)) => s.clone(),
            Some(other) => json_to_plain(other),
            None => String::new(),
        },
        Value::String(s) => s.clone(),
        other => json_to_plain(other),
    };

    let mut preview = oneline(&value_str);
    if preview.is_empty() {
        return None;
    }
    if max_len > 0 && (char_len(&preview) as i64) > max_len {
        let keep = (max_len - 3).max(0) as usize;
        preview = format!("{}...", char_slice(&preview, keep));
    }
    Some(preview)
}

fn primary_arg_key(tool_name: &str) -> Option<&'static str> {
    Some(match tool_name {
        "terminal" => "command",
        "web_search" => "query",
        "web_extract" => "urls",
        "read_file" => "path",
        "write_file" => "path",
        "patch" => "path",
        "search_files" => "pattern",
        "browser_navigate" => "url",
        "browser_click" => "ref",
        "browser_type" => "text",
        "image_generate" => "prompt",
        "text_to_speech" => "text",
        "vision_analyze" => "question",
        "mixture_of_agents" => "user_prompt",
        "skill_view" => "name",
        "skills_list" => "category",
        "cronjob" => "action",
        "execute_code" => "code",
        "delegate_task" => "goal",
        "clarify" => "question",
        "skill_manage" => "name",
        _ => return None,
    })
}

/// Render a JSON scalar the way Python's `str()` would for display purposes.
fn json_to_plain(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

// =========================================================================
// Inline diff previews for write actions
// =========================================================================

/// Resolve a possibly-relative filesystem path against the current cwd.
pub fn resolved_path(path: &str) -> PathBuf {
    let expanded = expanduser(path);
    let candidate = PathBuf::from(&expanded);
    if candidate.is_absolute() {
        candidate
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(candidate)
    }
}

fn expanduser(path: &str) -> String {
    if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().into_owned();
        }
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

/// Return UTF-8 file content, or None for missing/unreadable/non-UTF-8 files.
fn snapshot_text(path: &Path) -> Option<String> {
    std::fs::read(path).ok().and_then(|b| String::from_utf8(b).ok())
}

/// Prefer cwd-relative paths in diffs when available.
fn display_diff_path(path: &Path) -> String {
    if let (Ok(resolved), Ok(cwd)) = (path.canonicalize(), std::env::current_dir()) {
        if let Ok(cwd_resolved) = cwd.canonicalize() {
            if let Ok(rel) = resolved.strip_prefix(&cwd_resolved) {
                return rel.to_string_lossy().into_owned();
            }
        }
    }
    path.to_string_lossy().into_owned()
}

/// Resolve local filesystem targets for write-capable tools.
///
/// `skill_manage` resolution requires the skill-manager registry, which is not
/// part of this crate; for that tool an empty list is returned (the integrator
/// can override capture).
pub fn resolve_local_edit_paths(tool_name: &str, function_args: Option<&Value>) -> Vec<PathBuf> {
    let args = match function_args {
        Some(v) if v.is_object() => v,
        _ => return Vec::new(),
    };

    match tool_name {
        "write_file" => match args.get("path").and_then(Value::as_str) {
            Some(p) if !p.is_empty() => vec![resolved_path(p)],
            _ => Vec::new(),
        },
        "patch" => match args.get("path").and_then(Value::as_str) {
            Some(p) if !p.is_empty() => vec![resolved_path(p)],
            _ => Vec::new(),
        },
        // skill_manage requires the registry — not resolvable here.
        _ => Vec::new(),
    }
}

/// Capture before-state for local write previews.
pub fn capture_local_edit_snapshot(
    tool_name: &str,
    function_args: Option<&Value>,
) -> Option<LocalEditSnapshot> {
    let paths = resolve_local_edit_paths(tool_name, function_args);
    if paths.is_empty() {
        return None;
    }
    let mut snapshot = LocalEditSnapshot {
        paths: paths.clone(),
        ..Default::default()
    };
    for path in &paths {
        snapshot
            .before
            .insert(path.to_string_lossy().into_owned(), snapshot_text(path));
    }
    Some(snapshot)
}

/// Best-effort JSON parse mirroring the Python `safe_json_loads`.
fn safe_json_loads(result: &str) -> Option<Value> {
    serde_json::from_str(result).ok()
}

/// Conservatively detect whether a tool result represents success.
fn result_succeeded(result: Option<&str>) -> bool {
    let result = match result {
        Some(r) if !r.is_empty() => r,
        _ => return false,
    };
    let data = match safe_json_loads(result) {
        Some(d) => d,
        None => return false,
    };
    let obj = match data.as_object() {
        Some(o) => o,
        None => return false,
    };
    if let Some(err) = obj.get("error") {
        // Python: `if data.get("error")` — truthy check.
        if json_truthy(err) {
            return false;
        }
    }
    if let Some(success) = obj.get("success") {
        return json_truthy(success);
    }
    true
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

/// Build a unified diff with `difflib`-compatible output for two text blobs.
///
/// Mirrors Python's `difflib.unified_diff(..., keepends=True)` joined into a
/// single string. Lines are compared with `splitlines(keepends=True)` semantics.
fn unified_diff(before: Option<&str>, after: Option<&str>, fromfile: &str, tofile: &str) -> String {
    let a = before.map(splitlines_keepends).unwrap_or_default();
    let b = after.map(splitlines_keepends).unwrap_or_default();
    let opcodes = sequence_matcher_opcodes(&a, &b);

    let mut out = String::new();
    let mut started = false;

    // Group opcodes into hunks (difflib default n=3 context lines).
    let groups = grouped_opcodes(&opcodes, 3);
    for group in &groups {
        if !started {
            started = true;
            out.push_str(&format!("--- {fromfile}\n"));
            out.push_str(&format!("+++ {tofile}\n"));
        }
        let first = group.first().unwrap();
        let last = group.last().unwrap();
        let (i1, i2) = (first.1, last.2);
        let (j1, j2) = (first.3, last.4);
        out.push_str(&format!(
            "@@ -{} +{} @@\n",
            format_range_unified(i1, i2),
            format_range_unified(j1, j2)
        ));
        for op in group {
            let tag = &op.0;
            match tag.as_str() {
                "equal" => {
                    for line in &a[op.1..op.2] {
                        out.push_str(&prefix_line(' ', line));
                    }
                }
                "replace" | "delete" => {
                    for line in &a[op.1..op.2] {
                        out.push_str(&prefix_line('-', line));
                    }
                    if tag == "replace" {
                        for line in &b[op.3..op.4] {
                            out.push_str(&prefix_line('+', line));
                        }
                    }
                }
                "insert" => {
                    for line in &b[op.3..op.4] {
                        out.push_str(&prefix_line('+', line));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn prefix_line(sign: char, line: &str) -> String {
    // difflib does not re-add newlines; lines already carry their keepends.
    format!("{sign}{line}")
}

fn format_range_unified(start: usize, stop: usize) -> String {
    let length = stop.saturating_sub(start);
    let beginning = start + 1;
    if length == 1 {
        format!("{beginning}")
    } else if length == 0 {
        format!("{},0", beginning.saturating_sub(1))
    } else {
        format!("{beginning},{length}")
    }
}

/// Python `str.splitlines(keepends=True)`.
fn splitlines_keepends(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        if c == '\n' {
            out.push(std::mem::take(&mut current));
        } else if c == '\r' {
            // \r\n stays together.
            if i + 1 < chars.len() && chars[i + 1] == '\n' {
                current.push('\n');
                i += 1;
            }
            out.push(std::mem::take(&mut current));
        }
        i += 1;
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

type Opcode = (String, usize, usize, usize, usize);

/// Compute difflib-style opcodes for two sequences of lines.
fn sequence_matcher_opcodes(a: &[String], b: &[String]) -> Vec<Opcode> {
    let blocks = matching_blocks(a, b);
    let mut opcodes: Vec<Opcode> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    for (ai, bj, size) in blocks {
        let mut tag = "";
        if i < ai && j < bj {
            tag = "replace";
        } else if i < ai {
            tag = "delete";
        } else if j < bj {
            tag = "insert";
        }
        if !tag.is_empty() {
            opcodes.push((tag.to_string(), i, ai, j, bj));
        }
        i = ai + size;
        j = bj + size;
        if size > 0 {
            opcodes.push(("equal".to_string(), ai, i, bj, j));
        }
    }
    opcodes
}

/// difflib `get_matching_blocks`.
fn matching_blocks(a: &[String], b: &[String]) -> Vec<(usize, usize, usize)> {
    let mut b2j: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
    for (idx, line) in b.iter().enumerate() {
        b2j.entry(line.as_str()).or_default().push(idx);
    }

    let mut queue = vec![(0usize, a.len(), 0usize, b.len())];
    let mut matching: Vec<(usize, usize, usize)> = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest_match(a, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            matching.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    matching.sort_unstable();

    // Merge adjacent equal blocks.
    let (mut i1, mut j1, mut k1) = (0usize, 0usize, 0usize);
    let mut non_adjacent: Vec<(usize, usize, usize)> = Vec::new();
    for (i2, j2, k2) in matching {
        if i1 + k1 == i2 && j1 + k1 == j2 {
            k1 += k2;
        } else {
            if k1 > 0 {
                non_adjacent.push((i1, j1, k1));
            }
            i1 = i2;
            j1 = j2;
            k1 = k2;
        }
    }
    if k1 > 0 {
        non_adjacent.push((i1, j1, k1));
    }
    non_adjacent.push((a.len(), b.len(), 0));
    non_adjacent
}

fn find_longest_match(
    a: &[String],
    b2j: &std::collections::HashMap<&str, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in alo..ahi {
        let mut newj2len: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        if let Some(js) = b2j.get(a[i].as_str()) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j > 0 {
                    j2len.get(&(j - 1)).copied().unwrap_or(0) + 1
                } else {
                    1
                };
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    (besti, bestj, bestsize)
}

/// difflib `get_grouped_opcodes(n)`.
fn grouped_opcodes(opcodes: &[Opcode], n: usize) -> Vec<Vec<Opcode>> {
    let mut codes: Vec<Opcode> = if opcodes.is_empty() {
        vec![("equal".to_string(), 0, 1, 0, 1)]
    } else {
        opcodes.to_vec()
    };

    // Fix up leading/trailing equal context.
    if let Some(first) = codes.first().cloned() {
        if first.0 == "equal" {
            let (tag, i1, i2, j1, j2) = first;
            codes[0] = (tag, i2.saturating_sub(n).max(i1), i2, j2.saturating_sub(n).max(j1), j2);
        }
    }
    if let Some(last) = codes.last().cloned() {
        if last.0 == "equal" {
            let idx = codes.len() - 1;
            let (tag, i1, i2, j1, j2) = last;
            codes[idx] = (
                tag,
                i1,
                i2.min(i1 + n),
                j1,
                j2.min(j1 + n),
            );
        }
    }

    let nn = n * 2;
    let mut groups: Vec<Vec<Opcode>> = Vec::new();
    let mut group: Vec<Opcode> = Vec::new();
    for code in &codes {
        let (tag, mut i1, i2, mut j1, j2) = code.clone();
        if tag == "equal" && i2.saturating_sub(i1) > nn {
            group.push((
                tag.clone(),
                i1,
                i2.min(i1 + n),
                j1,
                j2.min(j1 + n),
            ));
            groups.push(std::mem::take(&mut group));
            i1 = i2.saturating_sub(n).max(i1);
            j1 = j2.saturating_sub(n).max(j1);
        }
        group.push((tag, i1, i2, j1, j2));
    }
    if !group.is_empty() && !(group.len() == 1 && group[0].0 == "equal") {
        groups.push(group);
    }
    groups
}

/// Generate unified diff text from a stored before-state and current files.
pub fn diff_from_snapshot(snapshot: Option<&LocalEditSnapshot>) -> Option<String> {
    let snapshot = snapshot?;
    let mut chunks: Vec<String> = Vec::new();
    for path in &snapshot.paths {
        let key = path.to_string_lossy().into_owned();
        let before = snapshot.before.get(&key).cloned().flatten();
        let after = snapshot_text(path);
        if before == after {
            continue;
        }
        let display_path = display_diff_path(path);
        let diff = unified_diff(
            before.as_deref(),
            after.as_deref(),
            &format!("a/{display_path}"),
            &format!("b/{display_path}"),
        );
        if !diff.is_empty() {
            chunks.push(diff);
        }
    }
    if chunks.is_empty() {
        return None;
    }
    let joined: String = chunks
        .into_iter()
        .map(|c| {
            if c.ends_with('\n') {
                c
            } else {
                format!("{c}\n")
            }
        })
        .collect();
    Some(joined)
}

/// Extract a unified diff from a file-edit tool result.
pub fn extract_edit_diff(
    tool_name: &str,
    result: Option<&str>,
    snapshot: Option<&LocalEditSnapshot>,
) -> Option<String> {
    if tool_name == "patch" {
        if let Some(r) = result {
            if let Some(data) = safe_json_loads(r) {
                if let Some(diff) = data.get("diff").and_then(Value::as_str) {
                    if !diff.trim().is_empty() {
                        return Some(diff.to_string());
                    }
                }
            }
        }
    }

    if !matches!(tool_name, "write_file" | "patch" | "skill_manage") {
        return None;
    }
    if !result_succeeded(result) {
        return None;
    }
    diff_from_snapshot(snapshot)
}

/// Render unified diff lines in Hermes' inline transcript style.
pub fn render_inline_unified_diff(diff: &str) -> Vec<String> {
    let mut rendered: Vec<String> = Vec::new();
    let mut from_file: Option<String> = None;

    for raw_line in diff.split('\n') {
        if let Some(rest) = raw_line.strip_prefix("--- ") {
            from_file = Some(rest.trim().to_string());
            continue;
        }
        if let Some(rest) = raw_line.strip_prefix("+++ ") {
            let to_file = rest.trim().to_string();
            if from_file.is_some() || !to_file.is_empty() {
                let f = from_file.clone().filter(|s| !s.is_empty());
                let t = if to_file.is_empty() { None } else { Some(to_file) };
                rendered.push(format!(
                    "{}{} → {}{}",
                    diff_file(),
                    f.unwrap_or_else(|| "a/?".to_string()),
                    t.unwrap_or_else(|| "b/?".to_string()),
                    ANSI_RESET
                ));
            }
            continue;
        }
        if raw_line.starts_with("@@") {
            rendered.push(format!("{}{}{}", diff_hunk(), raw_line, ANSI_RESET));
            continue;
        }
        if raw_line.starts_with('-') {
            rendered.push(format!("{}{}{}", diff_minus(), raw_line, ANSI_RESET));
            continue;
        }
        if raw_line.starts_with('+') {
            rendered.push(format!("{}{}{}", diff_plus(), raw_line, ANSI_RESET));
            continue;
        }
        if raw_line.starts_with(' ') {
            rendered.push(format!("{}{}{}", diff_dim(), raw_line, ANSI_RESET));
            continue;
        }
        if !raw_line.is_empty() {
            rendered.push(raw_line.to_string());
        }
    }

    rendered
}

/// Split a unified diff into per-file sections.
fn split_unified_diff_sections(diff: &str) -> Vec<String> {
    let mut sections: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in diff.split('\n') {
        if line.starts_with("--- ") && !current.is_empty() {
            sections.push(std::mem::take(&mut current));
            current.push(line.to_string());
            continue;
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        sections.push(current);
    }
    sections
        .into_iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.join("\n"))
        .collect()
}

/// Render diff sections while capping file count and total line count.
pub fn summarize_rendered_diff_sections(
    diff: &str,
    max_files: usize,
    max_lines: usize,
) -> Vec<String> {
    let sections = split_unified_diff_sections(diff);
    let mut rendered: Vec<String> = Vec::new();
    let mut omitted_files: usize = 0;
    let mut omitted_lines: usize = 0;

    for (idx, section) in sections.iter().enumerate() {
        if idx >= max_files {
            omitted_files += 1;
            omitted_lines += render_inline_unified_diff(section).len();
            continue;
        }

        let section_lines = render_inline_unified_diff(section);
        let remaining_budget = max_lines as isize - rendered.len() as isize;
        if remaining_budget <= 0 {
            omitted_lines += section_lines.len();
            omitted_files += 1;
            continue;
        }
        let remaining_budget = remaining_budget as usize;

        if section_lines.len() <= remaining_budget {
            rendered.extend(section_lines);
            continue;
        }

        rendered.extend(section_lines[..remaining_budget].iter().cloned());
        omitted_lines += section_lines.len() - remaining_budget;
        omitted_files += 1 + sections.len().saturating_sub(idx + 1);
        for leftover in &sections[idx + 1..] {
            omitted_lines += render_inline_unified_diff(leftover).len();
        }
        break;
    }

    if omitted_files > 0 || omitted_lines > 0 {
        let mut summary = format!("… omitted {omitted_lines} diff line(s)");
        if omitted_files > 0 {
            summary.push_str(&format!(
                " across {omitted_files} additional file(s)/section(s)"
            ));
        }
        rendered.push(format!("{}{}{}", diff_hunk(), summary, ANSI_RESET));
    }

    rendered
}

/// Emit rendered diff text through a CLI-supplied printer. Returns whether
/// anything was emitted.
pub fn emit_inline_diff<F: FnMut(&str)>(diff_text: &str, print_fn: Option<&mut F>) -> bool {
    let print_fn = match print_fn {
        Some(f) if !diff_text.is_empty() => f,
        _ => return false,
    };
    print_fn("  ┊ review diff");
    for line in diff_text.trim_end_matches('\n').split('\n') {
        print_fn(line);
    }
    true
}

/// Render an edit diff inline without taking over the terminal UI.
///
/// Returns whether anything was rendered.
pub fn render_edit_diff_with_delta<F: FnMut(&str)>(
    tool_name: &str,
    result: Option<&str>,
    snapshot: Option<&LocalEditSnapshot>,
    print_fn: Option<&mut F>,
) -> bool {
    let diff = match extract_edit_diff(tool_name, result, snapshot) {
        Some(d) => d,
        None => return false,
    };
    let rendered_lines =
        summarize_rendered_diff_sections(&diff, MAX_INLINE_DIFF_FILES, MAX_INLINE_DIFF_LINES);
    emit_inline_diff(&rendered_lines.join("\n"), print_fn)
}

// =========================================================================
// KawaiiSpinner
// =========================================================================

/// Static spinner frame definitions, keyed by name.
pub fn spinner_frames_for(name: &str) -> Vec<&'static str> {
    let table: &[(&str, &[&str])] = &[
        (
            "dots",
            &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        ),
        ("bounce", &["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]),
        (
            "grow",
            &[
                "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂",
            ],
        ),
        ("arrows", &["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]),
        ("star", &["✶", "✷", "✸", "✹", "✺", "✹", "✸", "✷"]),
        ("moon", &["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]),
        ("pulse", &["◜", "◠", "◝", "◞", "◡", "◟"]),
        ("brain", &["🧠", "💭", "💡", "✨", "💫", "🌟", "💡", "💭"]),
        ("sparkle", &["⁺", "˚", "*", "✧", "✦", "✧", "*", "˚"]),
    ];
    for (key, frames) in table {
        if *key == name {
            return frames.to_vec();
        }
    }
    // Fallback to dots.
    table[0].1.to_vec()
}

pub const KAWAII_WAITING: &[&str] = &[
    "(｡◕‿◕｡)",
    "(◕‿◕✿)",
    "٩(◕‿◕｡)۶",
    "(✿◠‿◠)",
    "( ˘▽˘)っ",
    "♪(´ε` )",
    "(◕ᴗ◕✿)",
    "ヾ(＾∇＾)",
    "(≧◡≦)",
    "(★ω★)",
];

pub const KAWAII_THINKING: &[&str] = &[
    "(｡•́︿•̀｡)",
    "(◔_◔)",
    "(¬‿¬)",
    "( •_•)>⌐■-■",
    "(⌐■_■)",
    "(´･_･`)",
    "◉_◉",
    "(°ロ°)",
    "( ˘⌣˘)♡",
    "ヽ(>∀<☆)☆",
    "٩(๑❛ᴗ❛๑)۶",
    "(⊙_⊙)",
    "(¬_¬)",
    "( ͡° ͜ʖ ͡°)",
    "ಠ_ಠ",
];

pub const THINKING_VERBS: &[&str] = &[
    "pondering",
    "contemplating",
    "musing",
    "cogitating",
    "ruminating",
    "deliberating",
    "mulling",
    "reflecting",
    "processing",
    "reasoning",
    "analyzing",
    "computing",
    "synthesizing",
    "formulating",
    "brainstorming",
];

/// Return waiting faces from the active skin, falling back to [`KAWAII_WAITING`].
pub fn get_waiting_faces() -> Vec<String> {
    if let Some(skin) = get_skin() {
        if !skin.waiting_faces.is_empty() {
            return skin.waiting_faces;
        }
    }
    KAWAII_WAITING.iter().map(|s| s.to_string()).collect()
}

/// Return thinking faces from the active skin, falling back to [`KAWAII_THINKING`].
pub fn get_thinking_faces() -> Vec<String> {
    if let Some(skin) = get_skin() {
        if !skin.thinking_faces.is_empty() {
            return skin.thinking_faces;
        }
    }
    KAWAII_THINKING.iter().map(|s| s.to_string()).collect()
}

/// Return thinking verbs from the active skin, falling back to [`THINKING_VERBS`].
pub fn get_thinking_verbs() -> Vec<String> {
    if let Some(skin) = get_skin() {
        if !skin.thinking_verbs.is_empty() {
            return skin.thinking_verbs;
        }
    }
    THINKING_VERBS.iter().map(|s| s.to_string()).collect()
}

/// Output sink for a spinner. Either an injected print callback, or stdout.
type PrintFn = Arc<dyn Fn(&str) + Send + Sync>;

struct SpinnerShared {
    message: Mutex<String>,
    running: std::sync::atomic::AtomicBool,
    frame_idx: AtomicI64,
    last_line_len: AtomicI64,
    start_time: Mutex<Option<Instant>>,
    print_fn: Option<PrintFn>,
}

impl SpinnerShared {
    fn write(&self, text: &str, end: &str, flush: bool) {
        if let Some(pf) = &self.print_fn {
            pf(text);
            return;
        }
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(text.as_bytes());
        let _ = lock.write_all(end.as_bytes());
        if flush {
            let _ = lock.flush();
        }
    }
}

/// Animated spinner with kawaii faces for CLI feedback during tool execution.
pub struct KawaiiSpinner {
    spinner_frames: Vec<String>,
    shared: Arc<SpinnerShared>,
    thread: Option<JoinHandle<()>>,
    has_print_fn: bool,
}

impl KawaiiSpinner {
    /// Construct a spinner. `spinner_type` selects the frame set; unknown names
    /// fall back to `dots`. `print_fn` routes all output through a callback when
    /// supplied (allowing silencing via a no-op).
    pub fn new(message: impl Into<String>, spinner_type: &str, print_fn: Option<PrintFn>) -> Self {
        let frames = spinner_frames_for(spinner_type)
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let has_print_fn = print_fn.is_some();
        let shared = Arc::new(SpinnerShared {
            message: Mutex::new(message.into()),
            running: std::sync::atomic::AtomicBool::new(false),
            frame_idx: AtomicI64::new(0),
            last_line_len: AtomicI64::new(0),
            start_time: Mutex::new(None),
            print_fn,
        });
        KawaiiSpinner {
            spinner_frames: frames,
            shared,
            thread: None,
            has_print_fn,
        }
    }

    fn is_tty(&self) -> bool {
        if self.has_print_fn {
            // Output routed through callback: not a real terminal.
            return false;
        }
        std::io::stdout().is_terminal()
    }

    /// Start the animation thread (idempotent while running).
    pub fn start(&mut self) {
        if self.shared.running.load(Ordering::SeqCst) {
            return;
        }
        self.shared.running.store(true, Ordering::SeqCst);
        *self.shared.start_time.lock().unwrap() = Some(Instant::now());

        let shared = Arc::clone(&self.shared);
        let frames = self.spinner_frames.clone();
        let is_tty = self.is_tty();
        let wings: Vec<(String, String)> = get_skin().map(|s| s.spinner_wings).unwrap_or_default();

        self.thread = Some(thread::spawn(move || {
            animate(&shared, &frames, is_tty, &wings);
        }));
    }

    pub fn update_text(&self, new_message: impl Into<String>) {
        *self.shared.message.lock().unwrap() = new_message.into();
    }

    /// Print a line above the spinner without disrupting animation.
    pub fn print_above(&self, text: &str) {
        if !self.shared.running.load(Ordering::SeqCst) {
            self.shared.write(&format!("  {text}"), "\n", true);
            return;
        }
        let pad = (self.shared.last_line_len.load(Ordering::SeqCst) + 5).max(40) as usize;
        let blanks = " ".repeat(pad);
        self.shared
            .write(&format!("\r{blanks}\r  {text}"), "\n", true);
    }

    /// Stop the animation. Optionally print a final completion message.
    pub fn stop(&mut self, final_message: Option<&str>) {
        self.shared.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            // Best effort join with a short bound — detach if it lingers.
            let joined = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let j2 = Arc::clone(&joined);
            let watcher = thread::spawn(move || {
                let _ = handle.join();
                j2.store(true, Ordering::SeqCst);
            });
            let deadline = Instant::now() + Duration::from_millis(500);
            while Instant::now() < deadline && !joined.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(10));
            }
            let _ = watcher;
        }

        let is_tty = self.is_tty();
        if is_tty {
            let pad = (self.shared.last_line_len.load(Ordering::SeqCst) + 5).max(40) as usize;
            let blanks = " ".repeat(pad);
            self.shared.write(&format!("\r{blanks}\r"), "", true);
        }
        if let Some(msg) = final_message {
            let elapsed = self
                .shared
                .start_time
                .lock()
                .unwrap()
                .map(|s| format!(" ({:.1}s)", s.elapsed().as_secs_f64()))
                .unwrap_or_default();
            if is_tty {
                self.shared.write(&format!("  {msg}"), "\n", true);
            } else {
                self.shared
                    .write(&format!("  [done] {msg}{elapsed}"), "\n", true);
            }
        }
    }
}

impl Drop for KawaiiSpinner {
    fn drop(&mut self) {
        if self.shared.running.load(Ordering::SeqCst) {
            self.stop(None);
        }
    }
}

fn animate(shared: &SpinnerShared, frames: &[String], is_tty: bool, wings: &[(String, String)]) {
    if !is_tty {
        let msg = shared.message.lock().unwrap().clone();
        shared.write(&format!("  [tool] {msg}"), "\n", true);
        while shared.running.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(500));
        }
        return;
    }

    while shared.running.load(Ordering::SeqCst) {
        if std::env::var("HERMES_SPINNER_PAUSE").is_ok() {
            thread::sleep(Duration::from_millis(100));
            continue;
        }
        let idx = shared.frame_idx.load(Ordering::SeqCst) as usize;
        let frame = &frames[idx % frames.len()];
        let elapsed = shared
            .start_time
            .lock()
            .unwrap()
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let msg = shared.message.lock().unwrap().clone();
        let line = if !wings.is_empty() {
            let (left, right) = &wings[idx % wings.len()];
            format!("  {left} {frame} {msg} {right} ({elapsed:.1}s)")
        } else {
            format!("  {frame} {msg} ({elapsed:.1}s)")
        };
        let pad =
            (shared.last_line_len.load(Ordering::SeqCst) - char_len(&line) as i64).max(0) as usize;
        shared.write(&format!("\r{line}{}", " ".repeat(pad)), "", true);
        shared
            .last_line_len
            .store(char_len(&line) as i64, Ordering::SeqCst);
        shared.frame_idx.fetch_add(1, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(120));
    }
}

// =========================================================================
// Cute tool message (completion line that replaces the spinner)
// =========================================================================

/// Inspect a tool result string for signs of failure. Returns
/// `(is_failure, suffix)`.
pub fn detect_tool_failure(tool_name: &str, result: Option<&str>) -> (bool, String) {
    let result = match result {
        Some(r) => r,
        None => return (false, String::new()),
    };

    if tool_name == "terminal" {
        if let Some(data) = safe_json_loads(result) {
            if let Some(obj) = data.as_object() {
                if let Some(exit_code) = obj.get("exit_code") {
                    if !exit_code.is_null() && exit_code.as_i64() != Some(0) {
                        let code = exit_code.as_i64().map(|c| c.to_string()).unwrap_or_else(|| {
                            exit_code.as_str().map(|s| s.to_string()).unwrap_or_else(|| exit_code.to_string())
                        });
                        return (true, format!(" [exit {code}]"));
                    }
                }
            }
        }
        return (false, String::new());
    }

    if tool_name == "memory" {
        if let Some(data) = safe_json_loads(result) {
            if let Some(obj) = data.as_object() {
                let success_false = obj.get("success").and_then(Value::as_bool) == Some(false);
                let err = obj.get("error").and_then(Value::as_str).unwrap_or("");
                if success_false && err.contains("exceed the limit") {
                    return (true, " [full]".to_string());
                }
            }
        }
    }

    // Generic heuristic for non-terminal tools.
    let head: String = result.chars().take(500).collect();
    let lower = head.to_lowercase();
    if lower.contains("\"error\"") || lower.contains("\"failed\"") || result.starts_with("Error") {
        return (true, " [error]".to_string());
    }

    (false, String::new())
}

/// Truncate for general display fields (Python `_trunc`).
fn trunc(s: &str, n: usize) -> String {
    if get_tool_preview_max_len() == 0 {
        return s.to_string();
    }
    if char_len(s) > n {
        format!("{}...", char_slice(s, n.saturating_sub(3)))
    } else {
        s.to_string()
    }
}

/// Truncate a path keeping the tail (Python `_path`).
fn path_trunc(p: &str, n: usize) -> String {
    if get_tool_preview_max_len() == 0 {
        return p.to_string();
    }
    if char_len(p) > n {
        format!("...{}", char_slice_tail(p, n.saturating_sub(3)))
    } else {
        p.to_string()
    }
}

/// Generate a formatted tool completion line for CLI quiet mode.
///
/// Format: `┊ {emoji} {verb:9} {detail}  {duration}`. When `result` is provided
/// the line is checked for failure indicators: failures get a suffix.
pub fn get_cute_tool_message(
    tool_name: &str,
    args: &Value,
    duration: f64,
    result: Option<&str>,
) -> String {
    let dur = format!("{duration:.1}s");
    let (is_failure, failure_suffix) = detect_tool_failure(tool_name, result);
    let skin_prefix = get_skin_tool_prefix();

    let wrap = |line: String| -> String {
        let mut line = line;
        if skin_prefix != "┊" {
            line = line.replacen('┊', &skin_prefix, 1);
        }
        if !is_failure {
            line
        } else {
            format!("{line}{failure_suffix}")
        }
    };

    let domain_of = |url: &str| -> String {
        url.replace("https://", "")
            .replace("http://", "")
            .split('/')
            .next()
            .unwrap_or("")
            .to_string()
    };

    match tool_name {
        "web_search" => {
            return wrap(format!(
                "┊ 🔍 search    {}  {dur}",
                trunc(arg_str(args, "query"), 42)
            ));
        }
        "web_extract" => {
            let urls = args.get("urls");
            if let Some(urls_val) = urls {
                let (first, count) = match urls_val {
                    Value::Array(items) if !items.is_empty() => {
                        let first = match &items[0] {
                            Value::String(s) => s.clone(),
                            other => json_to_plain(other),
                        };
                        (Some(first), items.len())
                    }
                    Value::Array(_) => (None, 0),
                    Value::String(s) if !s.is_empty() => (Some(s.clone()), 1),
                    Value::Null => (None, 0),
                    other => {
                        let s = json_to_plain(other);
                        if s.is_empty() {
                            (None, 0)
                        } else {
                            (Some(s), 1)
                        }
                    }
                };
                if let Some(url) = first {
                    let domain = domain_of(&url);
                    let extra = if count > 1 {
                        format!(" +{}", count - 1)
                    } else {
                        String::new()
                    };
                    return wrap(format!("┊ 📄 fetch     {}{}  {dur}", trunc(&domain, 35), extra));
                }
            }
            return wrap(format!("┊ 📄 fetch     pages  {dur}"));
        }
        "web_crawl" => {
            let domain = domain_of(arg_str(args, "url"));
            return wrap(format!("┊ 🕸️  crawl     {}  {dur}", trunc(&domain, 35)));
        }
        "terminal" => {
            return wrap(format!(
                "┊ 💻 $         {}  {dur}",
                trunc(arg_str(args, "command"), 42)
            ));
        }
        "process" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("?");
            let sid = char_slice(arg_str(args, "session_id"), 12);
            let label = match action {
                "list" => "ls processes".to_string(),
                "poll" => format!("poll {sid}"),
                "log" => format!("log {sid}"),
                "wait" => format!("wait {sid}"),
                "kill" => format!("kill {sid}"),
                "write" => format!("write {sid}"),
                "submit" => format!("submit {sid}"),
                _ => format!("{action} {sid}"),
            };
            return wrap(format!("┊ ⚙️  proc      {label}  {dur}"));
        }
        "read_file" => {
            return wrap(format!(
                "┊ 📖 read      {}  {dur}",
                path_trunc(arg_str(args, "path"), 35)
            ));
        }
        "write_file" => {
            return wrap(format!(
                "┊ ✍️  write     {}  {dur}",
                path_trunc(arg_str(args, "path"), 35)
            ));
        }
        "patch" => {
            return wrap(format!(
                "┊ 🔧 patch     {}  {dur}",
                path_trunc(arg_str(args, "path"), 35)
            ));
        }
        "search_files" => {
            let pattern = trunc(arg_str(args, "pattern"), 35);
            let target = args.get("target").and_then(Value::as_str).unwrap_or("content");
            let verb = if target == "files" { "find" } else { "grep" };
            return wrap(format!("┊ 🔎 {:9} {pattern}  {dur}", verb));
        }
        "browser_navigate" => {
            let domain = domain_of(arg_str(args, "url"));
            return wrap(format!("┊ 🌐 navigate  {}  {dur}", trunc(&domain, 35)));
        }
        "browser_snapshot" => {
            let full = args.get("full").map(json_truthy).unwrap_or(false);
            let mode = if full { "full" } else { "compact" };
            return wrap(format!("┊ 📸 snapshot  {mode}  {dur}"));
        }
        "browser_click" => {
            let r = args.get("ref").and_then(Value::as_str).unwrap_or("?");
            return wrap(format!("┊ 👆 click     {r}  {dur}"));
        }
        "browser_type" => {
            return wrap(format!(
                "┊ ⌨️  type      \"{}\"  {dur}",
                trunc(arg_str(args, "text"), 30)
            ));
        }
        "browser_scroll" => {
            let d = args.get("direction").and_then(Value::as_str).unwrap_or("down");
            let arrow = match d {
                "down" => "↓",
                "up" => "↑",
                "right" => "→",
                "left" => "←",
                _ => "↓",
            };
            return wrap(format!("┊ {arrow}  scroll    {d}  {dur}"));
        }
        "browser_back" => {
            return wrap(format!("┊ ◀️  back      {dur}"));
        }
        "browser_press" => {
            let key = args.get("key").and_then(Value::as_str).unwrap_or("?");
            return wrap(format!("┊ ⌨️  press     {key}  {dur}"));
        }
        "browser_get_images" => {
            return wrap(format!("┊ 🖼️  images    extracting  {dur}"));
        }
        "browser_vision" => {
            return wrap(format!("┊ 👁️  vision    analyzing page  {dur}"));
        }
        "todo" => {
            let todos_arg = args.get("todos");
            let merge = args.get("merge").and_then(Value::as_bool).unwrap_or(false);
            match todos_arg {
                None | Some(Value::Null) => {
                    return wrap(format!("┊ 📋 plan      reading tasks  {dur}"));
                }
                Some(v) => {
                    let len = v.as_array().map(|a| a.len()).unwrap_or(0);
                    if merge {
                        return wrap(format!("┊ 📋 plan      update {len} task(s)  {dur}"));
                    } else {
                        return wrap(format!("┊ 📋 plan      {len} task(s)  {dur}"));
                    }
                }
            }
        }
        "session_search" => {
            return wrap(format!(
                "┊ 🔍 recall    \"{}\"  {dur}",
                trunc(arg_str(args, "query"), 35)
            ));
        }
        "memory" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("?");
            let target = arg_str(args, "target");
            match action {
                "add" => {
                    return wrap(format!(
                        "┊ 🧠 memory    +{target}: \"{}\"  {dur}",
                        trunc(arg_str(args, "content"), 30)
                    ));
                }
                "replace" => {
                    let raw = arg_str(args, "old_text");
                    let old = if raw.is_empty() { "<missing old_text>" } else { raw };
                    return wrap(format!(
                        "┊ 🧠 memory    ~{target}: \"{}\"  {dur}",
                        trunc(old, 20)
                    ));
                }
                "remove" => {
                    let raw = arg_str(args, "old_text");
                    let old = if raw.is_empty() { "<missing old_text>" } else { raw };
                    return wrap(format!(
                        "┊ 🧠 memory    -{target}: \"{}\"  {dur}",
                        trunc(old, 20)
                    ));
                }
                _ => {
                    return wrap(format!("┊ 🧠 memory    {action}  {dur}"));
                }
            }
        }
        "skills_list" => {
            let cat = args.get("category").and_then(Value::as_str).unwrap_or("all");
            return wrap(format!("┊ 📚 skills    list {cat}  {dur}"));
        }
        "skill_view" => {
            return wrap(format!(
                "┊ 📚 skill     {}  {dur}",
                trunc(arg_str(args, "name"), 30)
            ));
        }
        "image_generate" => {
            return wrap(format!(
                "┊ 🎨 create    {}  {dur}",
                trunc(arg_str(args, "prompt"), 35)
            ));
        }
        "text_to_speech" => {
            return wrap(format!(
                "┊ 🔊 speak     {}  {dur}",
                trunc(arg_str(args, "text"), 30)
            ));
        }
        "vision_analyze" => {
            return wrap(format!(
                "┊ 👁️  vision    {}  {dur}",
                trunc(arg_str(args, "question"), 30)
            ));
        }
        "mixture_of_agents" => {
            return wrap(format!(
                "┊ 🧠 reason    {}  {dur}",
                trunc(arg_str(args, "user_prompt"), 30)
            ));
        }
        "send_message" => {
            let target = args.get("target").and_then(Value::as_str).unwrap_or("?");
            return wrap(format!(
                "┊ 📨 send      {target}: \"{}\"  {dur}",
                trunc(arg_str(args, "message"), 25)
            ));
        }
        "cronjob" => {
            let action = args.get("action").and_then(Value::as_str).unwrap_or("?");
            if action == "create" {
                // skills = args.skills or ([] if not skill else [skill])
                let skills: Vec<String> = match args.get("skills") {
                    Some(Value::Array(a)) if !a.is_empty() => a
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect(),
                    _ => match args.get("skill").and_then(Value::as_str) {
                        Some(s) if !s.is_empty() => vec![s.to_string()],
                        _ => Vec::new(),
                    },
                };
                let name = args.get("name").and_then(Value::as_str).unwrap_or("");
                let label = if !name.is_empty() {
                    name.to_string()
                } else if let Some(s) = skills.first() {
                    s.clone()
                } else {
                    let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("task");
                    if prompt.is_empty() { "task".to_string() } else { prompt.to_string() }
                };
                return wrap(format!("┊ ⏰ cron      create {}  {dur}", trunc(&label, 24)));
            }
            if action == "list" {
                return wrap(format!("┊ ⏰ cron      listing  {dur}"));
            }
            let job_id = arg_str(args, "job_id");
            return wrap(format!("┊ ⏰ cron      {action} {job_id}  {dur}"));
        }
        _ => {}
    }

    if let Some(stripped) = tool_name.strip_prefix("rl_") {
        let _ = stripped;
        let run_id12 = |k: &str| char_slice(arg_str(args, k), 12);
        let label = match tool_name {
            "rl_list_environments" => "list envs".to_string(),
            "rl_select_environment" => format!("select {}", arg_str(args, "name")),
            "rl_get_current_config" => "get config".to_string(),
            "rl_edit_config" => format!(
                "set {}",
                args.get("field").and_then(Value::as_str).unwrap_or("?")
            ),
            "rl_start_training" => "start training".to_string(),
            "rl_check_status" => format!(
                "status {}",
                if arg_str(args, "run_id").is_empty() {
                    char_slice("?", 12)
                } else {
                    run_id12("run_id")
                }
            ),
            "rl_stop_training" => format!(
                "stop {}",
                if arg_str(args, "run_id").is_empty() {
                    char_slice("?", 12)
                } else {
                    run_id12("run_id")
                }
            ),
            "rl_get_results" => format!(
                "results {}",
                if arg_str(args, "run_id").is_empty() {
                    char_slice("?", 12)
                } else {
                    run_id12("run_id")
                }
            ),
            "rl_list_runs" => "list runs".to_string(),
            "rl_test_inference" => "test inference".to_string(),
            _ => tool_name.strip_prefix("rl_").unwrap_or(tool_name).to_string(),
        };
        return wrap(format!("┊ 🧪 rl        {label}  {dur}"));
    }

    if tool_name == "execute_code" {
        let code = arg_str(args, "code");
        let first_line = if code.trim().is_empty() {
            ""
        } else {
            code.trim().split('\n').next().unwrap_or("")
        };
        return wrap(format!("┊ 🐍 exec      {}  {dur}", trunc(first_line, 35)));
    }

    if tool_name == "delegate_task" {
        if let Some(Value::Array(tasks)) = args.get("tasks") {
            return wrap(format!("┊ 🔀 delegate  {} parallel tasks  {dur}", tasks.len()));
        }
        return wrap(format!(
            "┊ 🔀 delegate  {}  {dur}",
            trunc(arg_str(args, "goal"), 35)
        ));
    }

    let preview = build_tool_preview(tool_name, args, None).unwrap_or_default();
    // Python: `{tool_name[:9]:9}` — truncate to 9 then left-pad to width 9.
    let name9 = char_slice(tool_name, 9);
    wrap(format!(
        "┊ ⚡ {:9} {}  {dur}",
        name9,
        trunc(&preview, 35)
    ))
}

/// Unused-by-default global guard kept to mirror Python's module-level cache
/// of resolved skin lookups; exposed for tests that want to reset state.
static TEST_RESET_GUARD: RwLock<()> = RwLock::new(());

#[doc(hidden)]
pub fn _test_reset_guard() -> &'static RwLock<()> {
    &TEST_RESET_GUARD
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn reset_len() {
        set_tool_preview_max_len(0);
    }

    #[test]
    fn oneline_collapses_whitespace() {
        assert_eq!(oneline("a  b\n c\t d"), "a b c d");
    }

    #[test]
    fn tool_preview_process() {
        reset_len();
        let args = json!({"action": "wait", "session_id": "0123456789abcdefXYZ", "data": "hello world here", "timeout": 30});
        let p = build_tool_preview("process", &args, None).unwrap();
        assert!(p.starts_with("wait 0123456789abcdef "));
        assert!(p.contains("\"hello world here\""));
        assert!(p.ends_with("30s"));
    }

    #[test]
    fn tool_preview_todo() {
        reset_len();
        assert_eq!(
            build_tool_preview("todo", &json!({}), None),
            None // empty args -> None
        );
        assert_eq!(
            build_tool_preview("todo", &json!({"merge": true, "x": 1}), None),
            Some("reading task list".to_string())
        );
        assert_eq!(
            build_tool_preview("todo", &json!({"todos": [1, 2, 3]}), None),
            Some("planning 3 task(s)".to_string())
        );
        assert_eq!(
            build_tool_preview("todo", &json!({"todos": [1, 2], "merge": true}), None),
            Some("updating 2 task(s)".to_string())
        );
    }

    #[test]
    fn tool_preview_memory_add_replace_remove() {
        reset_len();
        assert_eq!(
            build_tool_preview("memory", &json!({"action": "add", "target": "T", "content": "abc"}), None),
            Some("+T: \"abc\"".to_string())
        );
        assert_eq!(
            build_tool_preview("memory", &json!({"action": "replace", "target": "T"}), None),
            Some("~T: \"<missing old_text>\"".to_string())
        );
        assert_eq!(
            build_tool_preview("memory", &json!({"action": "remove", "target": "T", "old_text": "x"}), None),
            Some("-T: \"x\"".to_string())
        );
        assert_eq!(
            build_tool_preview("memory", &json!({"action": "list"}), None),
            Some("list".to_string())
        );
    }

    #[test]
    fn tool_preview_send_message_truncates() {
        reset_len();
        let args = json!({"target": "alice", "message": "this is a fairly long message indeed"});
        let p = build_tool_preview("send_message", &args, None).unwrap();
        assert!(p.starts_with("to alice: \""));
        assert!(p.ends_with("...\""));
    }

    #[test]
    fn tool_preview_rl() {
        reset_len();
        assert_eq!(
            build_tool_preview("rl_list_environments", &json!({"x": 1}), None),
            Some("listing envs".to_string())
        );
        assert_eq!(
            build_tool_preview("rl_test_inference", &json!({"x": 1}), None),
            Some("3 steps".to_string())
        );
        assert_eq!(
            build_tool_preview("rl_edit_config", &json!({"field": "lr", "value": "0.1"}), None),
            Some("lr=0.1".to_string())
        );
    }

    #[test]
    fn tool_preview_primary_and_fallback() {
        reset_len();
        assert_eq!(
            build_tool_preview("web_search", &json!({"query": "rust diff"}), None),
            Some("rust diff".to_string())
        );
        // urls is a list -> first element
        assert_eq!(
            build_tool_preview("web_extract", &json!({"urls": ["https://a.com", "https://b.com"]}), None),
            Some("https://a.com".to_string())
        );
        // fallback: unknown tool with a 'text' key
        assert_eq!(
            build_tool_preview("unknown", &json!({"text": "hi there"}), None),
            Some("hi there".to_string())
        );
        // no usable key
        assert_eq!(build_tool_preview("unknown", &json!({"zzz": 1}), None), None);
    }

    #[test]
    fn tool_preview_max_len_truncation() {
        let args = json!({"command": "a very long command line that exceeds the limit"});
        let p = build_tool_preview("terminal", &args, Some(10)).unwrap();
        assert_eq!(char_len(&p), 10);
        assert!(p.ends_with("..."));
    }

    #[test]
    fn detect_failure_terminal() {
        assert_eq!(
            detect_tool_failure("terminal", Some("{\"exit_code\": 1}")),
            (true, " [exit 1]".to_string())
        );
        assert_eq!(
            detect_tool_failure("terminal", Some("{\"exit_code\": 0}")),
            (false, String::new())
        );
    }

    #[test]
    fn detect_failure_memory_full() {
        let r = "{\"success\": false, \"error\": \"would exceed the limit\"}";
        assert_eq!(detect_tool_failure("memory", Some(r)), (true, " [full]".to_string()));
    }

    #[test]
    fn detect_failure_generic() {
        assert_eq!(
            detect_tool_failure("foo", Some("{\"error\": \"boom\"}")),
            (true, " [error]".to_string())
        );
        assert_eq!(
            detect_tool_failure("foo", Some("Error: nope")),
            (true, " [error]".to_string())
        );
        assert_eq!(detect_tool_failure("foo", Some("all good")), (false, String::new()));
        assert_eq!(detect_tool_failure("foo", None), (false, String::new()));
    }

    #[test]
    fn cute_message_web_search() {
        reset_len();
        let msg = get_cute_tool_message("web_search", &json!({"query": "hi"}), 1.25, None);
        assert_eq!(msg, "┊ 🔍 search    hi  1.2s");
    }

    #[test]
    fn cute_message_failure_suffix() {
        reset_len();
        let msg = get_cute_tool_message(
            "terminal",
            &json!({"command": "ls"}),
            0.5,
            Some("{\"exit_code\": 2}"),
        );
        assert!(msg.ends_with("[exit 2]"));
        assert!(msg.contains("💻 $"));
    }

    #[test]
    fn cute_message_web_extract_multi() {
        reset_len();
        let msg = get_cute_tool_message(
            "web_extract",
            &json!({"urls": ["https://example.com/x", "https://b.com"]}),
            0.1,
            None,
        );
        assert!(msg.contains("example.com +1"));
    }

    #[test]
    fn cute_message_search_files_verbs() {
        reset_len();
        let grep = get_cute_tool_message("search_files", &json!({"pattern": "x"}), 0.1, None);
        assert!(grep.contains("🔎 grep"));
        let find = get_cute_tool_message(
            "search_files",
            &json!({"pattern": "x", "target": "files"}),
            0.1,
            None,
        );
        assert!(find.contains("🔎 find"));
    }

    #[test]
    fn cute_message_fallback() {
        reset_len();
        let msg = get_cute_tool_message("custom_tool", &json!({"text": "hello"}), 2.0, None);
        assert!(msg.starts_with("┊ ⚡ custom_to"));
        assert!(msg.contains("hello"));
        assert!(msg.ends_with("2.0s"));
    }

    #[test]
    fn unified_diff_basic_insert() {
        let d = unified_diff(Some("a\nb\n"), Some("a\nb\nc\n"), "a/f", "b/f");
        assert!(d.contains("--- a/f\n"));
        assert!(d.contains("+++ b/f\n"));
        assert!(d.contains("+c\n"));
    }

    #[test]
    fn unified_diff_replace() {
        let d = unified_diff(Some("hello\n"), Some("world\n"), "a/x", "b/x");
        assert!(d.contains("-hello\n"));
        assert!(d.contains("+world\n"));
    }

    #[test]
    fn diff_from_snapshot_detects_change() {
        let dir = std::env::temp_dir().join(format!("ag_display_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        std::fs::write(&file, "one\ntwo\n").unwrap();
        let mut snap = LocalEditSnapshot {
            paths: vec![file.clone()],
            ..Default::default()
        };
        snap.before.insert(
            file.to_string_lossy().into_owned(),
            Some("one\ntwo\n".to_string()),
        );
        std::fs::write(&file, "one\ntwo\nthree\n").unwrap();
        let diff = diff_from_snapshot(Some(&snap)).unwrap();
        assert!(diff.contains("+three\n"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_edit_diff_patch_inline() {
        let r = "{\"diff\": \"--- a\\n+++ b\\n+x\\n\", \"success\": true}";
        let d = extract_edit_diff("patch", Some(r), None).unwrap();
        assert!(d.contains("+x"));
    }

    #[test]
    fn render_inline_diff_colors_lines() {
        let diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n unchanged";
        let lines = render_inline_unified_diff(diff);
        assert!(lines[0].contains("a/f → b/f"));
        assert!(lines[1].contains("@@"));
        assert!(lines[2].contains("-old"));
        assert!(lines[3].contains("+new"));
        assert!(lines[4].contains(" unchanged"));
    }

    #[test]
    fn summarize_caps_lines() {
        // Build a diff with many lines.
        let mut diff = String::from("--- a/f\n+++ b/f\n@@ -1,100 +1,100 @@\n");
        for i in 0..200 {
            diff.push_str(&format!("+line{i}\n"));
        }
        let rendered = summarize_rendered_diff_sections(&diff, 6, 80);
        assert!(rendered.len() <= 81); // 80 + summary line
        assert!(rendered.last().unwrap().contains("omitted"));
    }

    #[test]
    fn result_succeeded_variants() {
        assert!(!result_succeeded(None));
        assert!(!result_succeeded(Some("not json")));
        assert!(!result_succeeded(Some("{\"error\": \"x\"}")));
        assert!(result_succeeded(Some("{\"success\": true}")));
        assert!(!result_succeeded(Some("{\"success\": false}")));
        assert!(result_succeeded(Some("{\"data\": 1}")));
    }

    #[test]
    fn spinner_frames_lookup() {
        assert_eq!(spinner_frames_for("arrows")[0], "←");
        assert_eq!(spinner_frames_for("unknown")[0], "⠋"); // dots fallback
    }

    #[test]
    fn skin_defaults_present() {
        assert!(!get_waiting_faces().is_empty());
        assert!(!get_thinking_faces().is_empty());
        assert!(!get_thinking_verbs().is_empty());
        assert_eq!(get_skin_tool_prefix(), "┊");
        assert_eq!(get_tool_emoji("anything", "⚡"), "⚡");
    }

    #[test]
    fn spinner_runs_with_print_fn() {
        use std::sync::Mutex as StdMutex;
        let buf = Arc::new(StdMutex::new(String::new()));
        let b2 = Arc::clone(&buf);
        let pf: PrintFn = Arc::new(move |s: &str| {
            b2.lock().unwrap().push_str(s);
            b2.lock().unwrap().push('\n');
        });
        let mut sp = KawaiiSpinner::new("working", "dots", Some(pf));
        sp.start();
        thread::sleep(Duration::from_millis(50));
        sp.stop(Some("done"));
        let out = buf.lock().unwrap().clone();
        assert!(out.contains("working"));
        assert!(out.contains("done"));
    }
}
