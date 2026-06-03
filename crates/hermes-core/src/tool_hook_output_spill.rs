//! Spill oversized hook-injected context to disk with a preview placeholder.
//!
//! Ported from the Python module `tools/hook_output_spill.py`, itself adapted
//! from openai/codex PR #21069 ("Spill large hook outputs from context").
//!
//! # Background
//! Both shell hooks (`agent/shell_hooks.py`) and Python plugins (`pre_llm_call`
//! hook in `run_agent.py`) can return `{"context": "..."}` which gets
//! concatenated into the current turn's user message on EVERY subsequent API
//! call. If a hook emits a large blob (e.g. a debug dump, a full file, or a
//! runaway prompt-engineering script), that blob inflates every turn of the
//! session and blows out the prompt cache prefix the moment it's appended.
//!
//! This mirrors what Codex does for its `PreToolUse`/`Stop`/feedback hooks:
//! once the injected text exceeds a configured budget, write the full content
//! to a per-session directory on disk and replace the in-prompt payload with a
//! head/tail preview plus the saved path. The model can still inspect the full
//! content via `read_file` or `terminal` if it needs to.
//!
//! Config (`config.yaml`):
//! ```yaml
//! hooks:
//!   output_spill:
//!     enabled: true          # default: true; set false to disable spilling
//!     max_chars: 10000       # default; context above this is spilled
//!     preview_head: 500      # chars shown at the start of the preview
//!     preview_tail: 500      # chars shown at the end of the preview
//!     directory: null        # default: <HERMES_HOME>/hook_outputs
//! ```
//!
//! # Design invariants
//! * Behaviour-preserving when `enabled: false` or when content is under the
//!   cap — return the input string unchanged.
//! * Never panics. Any I/O error (disk full, permission denied, missing
//!   HERMES_HOME, etc.) falls back to a preview with an in-prompt notice — the
//!   hook context still reaches the model, just bounded in size.
//! * Spill files are grouped by session so a `/new` session doesn't grow them
//!   forever in one directory.

use std::path::{Path, PathBuf};

use serde_yaml::Value;

pub const DEFAULT_MAX_CHARS: usize = 10_000;
pub const DEFAULT_PREVIEW_HEAD: usize = 500;
pub const DEFAULT_PREVIEW_TAIL: usize = 500;
pub const DEFAULT_ENABLED: bool = true;

/// Resolved hook output-spill configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillConfig {
    pub enabled: bool,
    pub max_chars: usize,
    pub preview_head: usize,
    pub preview_tail: usize,
    pub directory: Option<String>,
}

impl Default for SpillConfig {
    fn default() -> Self {
        SpillConfig {
            enabled: DEFAULT_ENABLED,
            max_chars: DEFAULT_MAX_CHARS,
            preview_head: DEFAULT_PREVIEW_HEAD,
            preview_tail: DEFAULT_PREVIEW_TAIL,
            directory: None,
        }
    }
}

/// Mirror of Python's `int(value)` coercion for YAML scalars: accepts ints,
/// floats (truncated toward zero), and numeric strings. Returns `None` when the
/// value cannot be interpreted as an integer (matching `TypeError`/`ValueError`).
fn coerce_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                Some(u as i64)
            } else {
                // Python's int(float) truncates toward zero.
                n.as_f64().map(|f| f.trunc() as i64)
            }
        }
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        Value::String(s) => {
            let t = s.trim();
            // Python int("...") accepts optional sign and surrounding whitespace
            // but rejects floats like "1.5". Try a plain integer parse first.
            if let Ok(i) = t.parse::<i64>() {
                Some(i)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn coerce_positive_int(value: Option<&Value>, default: usize) -> usize {
    match value.and_then(coerce_int) {
        Some(iv) if iv > 0 => iv as usize,
        _ => default,
    }
}

/// Like [`coerce_positive_int`] but allows zero (e.g. empty tail).
fn coerce_non_negative_int(value: Option<&Value>, default: usize) -> usize {
    match value.and_then(coerce_int) {
        Some(iv) if iv >= 0 => iv as usize,
        _ => default,
    }
}

/// Load the raw `hooks.output_spill` mapping from `config.yaml`, if present.
///
/// Returns `None` on any failure (missing file, parse error, wrong shape) so
/// callers fall back to defaults — this never raises.
fn load_section() -> Option<Value> {
    let path: PathBuf = crate::mod_hermes_constants::get_hermes_home().join("config.yaml");
    let contents = std::fs::read_to_string(&path).ok()?;
    let cfg: Value = serde_yaml::from_str(&contents).ok()?;
    let hooks = cfg.get("hooks")?;
    let sub = hooks.get("output_spill")?;
    if sub.is_mapping() {
        Some(sub.clone())
    } else {
        None
    }
}

/// Return resolved hook output-spill config. Never raises.
pub fn get_spill_config() -> SpillConfig {
    let section = load_section();
    resolve_spill_config(section.as_ref())
}

/// Resolve a [`SpillConfig`] from an optional raw YAML `output_spill` mapping.
///
/// Exposed so tests (and callers that already hold a parsed config) can resolve
/// without touching the filesystem.
pub fn resolve_spill_config(section: Option<&Value>) -> SpillConfig {
    let get = |key: &str| -> Option<&Value> { section.and_then(|s| s.get(key)) };

    // enabled: Python does `bool(enabled_raw) if enabled_raw is not None else DEFAULT`.
    // YAML `null` -> default; otherwise truthiness of the value.
    let enabled = match get("enabled") {
        None => DEFAULT_ENABLED,
        Some(Value::Null) => DEFAULT_ENABLED,
        Some(v) => value_truthy(v),
    };

    // directory: only a string is accepted; anything else (incl. null) -> None.
    let directory = match get("directory") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };

    SpillConfig {
        enabled,
        max_chars: coerce_positive_int(get("max_chars"), DEFAULT_MAX_CHARS),
        preview_head: coerce_non_negative_int(get("preview_head"), DEFAULT_PREVIEW_HEAD),
        preview_tail: coerce_non_negative_int(get("preview_tail"), DEFAULT_PREVIEW_TAIL),
        directory,
    }
}

/// Mirror Python truthiness for the values that can appear in YAML config.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else {
                n.as_f64().map(|f| f != 0.0).unwrap_or(true)
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Sequence(s) => !s.is_empty(),
        Value::Mapping(m) => !m.is_empty(),
        // serde_yaml tagged values are non-empty objects -> truthy.
        _ => true,
    }
}

/// Return the directory where spill files for this session live.
fn resolve_spill_dir(directory_override: Option<&str>, session_id: Option<&str>) -> PathBuf {
    let base: PathBuf = match directory_override.filter(|s| !s.is_empty()) {
        Some(dir) => PathBuf::from(expanduser(dir)),
        None => crate::mod_hermes_constants::get_hermes_home().join("hook_outputs"),
    };

    // Group by session so spills are contained per conversation.
    let raw = session_id.filter(|s| !s.is_empty()).unwrap_or("no-session");
    // Defensive: strip path separators so a weird session id can't escape the
    // directory. Matches Python's str.replace chain (order matters).
    let session_segment = raw
        .replace('/', "_")
        .replace('\\', "_")
        .replace("..", "_");
    base.join(session_segment)
}

/// Expand a leading `~` to the user's home directory, mirroring
/// `os.path.expanduser`. Only handles the common `~` / `~/...` prefix.
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

/// Format an integer with thousands separators, mirroring Python's `{:,}`.
fn comma_format(n: usize) -> String {
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

/// Take the first `n` characters (Unicode scalar values), matching Python's
/// `text[:n]` semantics where indices are by character, not byte.
fn head_chars(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// Take the last `n` characters (Unicode scalar values), matching Python's
/// `text[-n:]`.
fn tail_chars(text: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let start = chars.len().saturating_sub(n);
    chars[start..].iter().collect()
}

/// Number of Unicode scalar values in `text` (matches Python `len(str)`).
fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Assemble the in-prompt preview with head/tail and saved-path footer.
fn build_preview(
    text: &str,
    head: usize,
    tail: usize,
    saved_path: Option<&str>,
    source: &str,
) -> String {
    let total = char_len(text);
    let head_chunk = if head > 0 { head_chars(text, head) } else { String::new() };
    // Python: `text[-tail:] if tail > 0 and total > head else ""`
    let tail_chunk = if tail > 0 && total > head {
        tail_chars(text, tail)
    } else {
        String::new()
    };

    let footer = match saved_path {
        Some(p) => format!("saved to {p}]"),
        None => "unavailable — spill write failed]".to_string(),
    };
    let header = format!(
        "[{source} output truncated — {total} chars; full content {footer}",
        total = comma_format(total),
    );

    let mut parts: Vec<String> = vec![header];
    if !head_chunk.is_empty() {
        parts.push("--- head ---".to_string());
        parts.push(head_chunk);
    }
    if !tail_chunk.is_empty() {
        parts.push("--- tail ---".to_string());
        parts.push(tail_chunk);
    }
    parts.join("\n")
}

/// Spill `text` to disk if it exceeds the configured cap.
///
/// Returns either `text` unchanged (when under the cap, disabled, or empty) or
/// a preview string with a filesystem path pointing at the full content.
///
/// # Parameters
/// * `text` — the raw injected-context string from a hook.
/// * `session_id` — used to group spill files by conversation. Falls back to
///   `"no-session"` if missing.
/// * `source` — human-readable label used in the preview header (`"hook"`,
///   `"plugin hook"`, `"shell hook"`, etc.). Free-form.
/// * `config` — optional override for tests; normally resolved from
///   `config.yaml`.
pub fn spill_if_oversized(
    text: &str,
    session_id: Option<&str>,
    source: &str,
    config: Option<&SpillConfig>,
) -> String {
    let owned_cfg;
    let cfg: &SpillConfig = match config {
        Some(c) => c,
        None => {
            owned_cfg = get_spill_config();
            &owned_cfg
        }
    };

    if !cfg.enabled {
        return text.to_string();
    }

    // Python uses `cfg.get("max_chars") or DEFAULT` — a 0/falsy value falls back
    // to the default. The resolver never produces 0 for max_chars, but guard
    // anyway to preserve the semantics.
    let max_chars = if cfg.max_chars == 0 { DEFAULT_MAX_CHARS } else { cfg.max_chars };

    if char_len(text) <= max_chars {
        return text.to_string();
    }

    let head = cfg.preview_head;
    let tail = cfg.preview_tail;
    let directory_override = cfg.directory.as_deref();

    // Try to write the spill file. If that fails we still need to return
    // something bounded — never let a disk failure blow up the turn.
    let saved_path: Option<String> = match write_spill(text, directory_override, session_id) {
        Ok(p) => Some(p),
        Err(err) => {
            log::warn!("hook output spill failed: {err}");
            None
        }
    };

    build_preview(text, head, tail, saved_path.as_deref(), source)
}

/// Write the full text to a uniquely-named file under the session spill dir and
/// return the absolute path as a string.
fn write_spill(
    text: &str,
    directory_override: Option<&str>,
    session_id: Option<&str>,
) -> std::io::Result<String> {
    let spill_dir = resolve_spill_dir(directory_override, session_id);
    std::fs::create_dir_all(&spill_dir)?;
    let filename = format!("{}.txt", uuid_hex());
    let spill_path: PathBuf = spill_dir.join(filename);
    // Write the raw text plus a trailing newline so tail readers (`tail -f`,
    // editors) don't report "missing newline".
    let payload = if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    };
    std::fs::write(&spill_path, payload.as_bytes())?;
    Ok(path_to_string(&spill_path))
}

fn path_to_string(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Generate a 32-char lowercase-hex random id, equivalent to
/// `uuid.uuid4().hex`. Uses the system RNG via timestamp + process entropy to
/// avoid a new crate dependency.
fn uuid_hex() -> String {
    // Build 16 bytes from multiple entropy sources, then set the version/variant
    // bits to look like a UUIDv4 (cosmetic — only the hex string is consumed).
    let mut bytes = [0u8; 16];

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    // A per-call counter mixed in via the address of a stack local for extra
    // intra-process uniqueness.
    let stack_marker = &bytes as *const _ as u128;

    // Simple xorshift-style mixing across two 64-bit lanes.
    let mut lo = (now ^ stack_marker ^ pid) as u64;
    let mut hi = (now.rotate_left(64) ^ (pid << 17) ^ stack_marker.rotate_left(32)) as u64;

    let mut splitmix = |state: &mut u64| -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    let a = splitmix(&mut lo);
    let b = splitmix(&mut hi);
    bytes[..8].copy_from_slice(&a.to_le_bytes());
    bytes[8..].copy_from_slice(&b.to_le_bytes());

    // UUIDv4 version/variant cosmetics.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    let mut s = String::with_capacity(32);
    for byte in &bytes {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(enabled: bool, max: usize, head: usize, tail: usize, dir: Option<&str>) -> SpillConfig {
        SpillConfig {
            enabled,
            max_chars: max,
            preview_head: head,
            preview_tail: tail,
            directory: dir.map(|s| s.to_string()),
        }
    }

    #[test]
    fn under_cap_returns_unchanged() {
        let c = cfg(true, 100, 10, 10, None);
        let out = spill_if_oversized("hello", None, "hook", Some(&c));
        assert_eq!(out, "hello");
    }

    #[test]
    fn at_cap_returns_unchanged() {
        let c = cfg(true, 5, 2, 2, None);
        let out = spill_if_oversized("hello", None, "hook", Some(&c));
        assert_eq!(out, "hello");
    }

    #[test]
    fn disabled_returns_unchanged_even_when_oversized() {
        let c = cfg(false, 1, 1, 1, None);
        let big = "x".repeat(100);
        let out = spill_if_oversized(&big, None, "hook", Some(&c));
        assert_eq!(out, big);
    }

    #[test]
    fn oversized_spills_to_disk_and_previews() {
        let tmp = std::env::temp_dir().join(format!("hermes_spill_test_{}", uuid_hex()));
        let c = cfg(true, 10, 4, 4, Some(tmp.to_str().unwrap()));
        let text = "abcdefghijklmnopqrstuvwxyz"; // 26 chars > 10
        let out = spill_if_oversized(text, Some("sess1"), "plugin hook", Some(&c));

        assert!(out.contains("[plugin hook output truncated — 26 chars; full content saved to "));
        assert!(out.contains("--- head ---"));
        assert!(out.contains("abcd"));
        assert!(out.contains("--- tail ---"));
        assert!(out.contains("wxyz"));

        // The session subdir must exist and hold exactly one .txt file with the
        // full content plus a trailing newline.
        let session_dir = tmp.join("sess1");
        assert!(session_dir.is_dir());
        let entries: Vec<_> = std::fs::read_dir(&session_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let contents = std::fs::read_to_string(entries[0].path()).unwrap();
        assert_eq!(contents, format!("{text}\n"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn trailing_newline_not_duplicated() {
        let tmp = std::env::temp_dir().join(format!("hermes_spill_nl_{}", uuid_hex()));
        let c = cfg(true, 3, 2, 2, Some(tmp.to_str().unwrap()));
        let text = "abcdef\n";
        let _ = spill_if_oversized(text, Some("s"), "hook", Some(&c));
        let session_dir = tmp.join("s");
        let entry = std::fs::read_dir(&session_dir).unwrap().next().unwrap().unwrap();
        let contents = std::fs::read_to_string(entry.path()).unwrap();
        assert_eq!(contents, text);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn write_failure_falls_back_to_preview_without_path() {
        // Point at a path whose parent is a file -> create_dir_all fails.
        let tmp = std::env::temp_dir().join(format!("hermes_spill_fail_{}", uuid_hex()));
        std::fs::write(&tmp, b"i am a file").unwrap();
        let blocked = tmp.join("subdir");
        let c = cfg(true, 5, 3, 3, Some(blocked.to_str().unwrap()));
        let text = "abcdefghij";
        let out = spill_if_oversized(text, Some("x"), "shell hook", Some(&c));
        assert!(out.contains("unavailable — spill write failed]"));
        assert!(!out.contains("saved to"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn tail_skipped_when_total_not_greater_than_head() {
        // head=10, tail=5, content len 10 — Python: tail only if total > head.
        // Make it oversized via tiny max so we hit the preview path.
        let tmp = std::env::temp_dir().join(format!("hermes_spill_tail_{}", uuid_hex()));
        let c = cfg(true, 2, 10, 5, Some(tmp.to_str().unwrap()));
        let text = "abcdefghij"; // 10 chars, total == head
        let out = spill_if_oversized(text, Some("z"), "hook", Some(&c));
        assert!(out.contains("--- head ---"));
        assert!(!out.contains("--- tail ---"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn comma_format_matches_python() {
        assert_eq!(comma_format(0), "0");
        assert_eq!(comma_format(999), "999");
        assert_eq!(comma_format(1_000), "1,000");
        assert_eq!(comma_format(1_234_567), "1,234,567");
        assert_eq!(comma_format(26), "26");
    }

    #[test]
    fn session_id_path_traversal_sanitized() {
        let dir = resolve_spill_dir(Some("/tmp/base"), Some("../../etc/passwd"));
        let s = dir.to_string_lossy();
        assert!(!s.contains(".."));
        assert!(!s.contains("etc/passwd"));
        // slashes and dotdot replaced with underscores
        assert!(s.ends_with("_____etc_passwd") || s.contains("__"));
    }

    #[test]
    fn missing_session_uses_no_session() {
        let dir = resolve_spill_dir(Some("/tmp/base"), None);
        assert!(dir.ends_with("no-session"));
    }

    #[test]
    fn coerce_positive_int_rejects_zero_and_negative() {
        assert_eq!(coerce_positive_int(Some(&Value::from(0)), 99), 99);
        assert_eq!(coerce_positive_int(Some(&Value::from(-3)), 99), 99);
        assert_eq!(coerce_positive_int(Some(&Value::from(7)), 99), 7);
        assert_eq!(coerce_positive_int(None, 99), 99);
        assert_eq!(coerce_positive_int(Some(&Value::from("abc")), 99), 99);
        assert_eq!(coerce_positive_int(Some(&Value::from("42")), 99), 42);
    }

    #[test]
    fn coerce_non_negative_allows_zero() {
        assert_eq!(coerce_non_negative_int(Some(&Value::from(0)), 99), 0);
        assert_eq!(coerce_non_negative_int(Some(&Value::from(-1)), 99), 99);
    }

    #[test]
    fn resolve_config_defaults_when_section_missing() {
        let c = resolve_spill_config(None);
        assert_eq!(c, SpillConfig::default());
    }

    #[test]
    fn resolve_config_from_yaml() {
        let yaml = "enabled: false\nmax_chars: 50\npreview_head: 0\npreview_tail: 12\ndirectory: /var/spill\n";
        let v: Value = serde_yaml::from_str(yaml).unwrap();
        let c = resolve_spill_config(Some(&v));
        assert!(!c.enabled);
        assert_eq!(c.max_chars, 50);
        assert_eq!(c.preview_head, 0);
        assert_eq!(c.preview_tail, 12);
        assert_eq!(c.directory.as_deref(), Some("/var/spill"));
    }

    #[test]
    fn resolve_config_null_directory_and_enabled() {
        let yaml = "enabled: null\ndirectory: null\n";
        let v: Value = serde_yaml::from_str(yaml).unwrap();
        let c = resolve_spill_config(Some(&v));
        // null enabled -> default true
        assert!(c.enabled);
        assert_eq!(c.directory, None);
    }

    #[test]
    fn resolve_config_non_string_directory_ignored() {
        let yaml = "directory: 1234\n";
        let v: Value = serde_yaml::from_str(yaml).unwrap();
        let c = resolve_spill_config(Some(&v));
        assert_eq!(c.directory, None);
    }

    #[test]
    fn uuid_hex_is_32_hex_chars_and_unique() {
        let a = uuid_hex();
        let b = uuid_hex();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn unicode_char_counting() {
        // 5 multibyte chars; cap of 3 chars should spill.
        let tmp = std::env::temp_dir().join(format!("hermes_spill_uni_{}", uuid_hex()));
        let c = cfg(true, 3, 2, 2, Some(tmp.to_str().unwrap()));
        let text = "héllo wörld ☃☃"; // > 3 chars
        let out = spill_if_oversized(text, Some("u"), "hook", Some(&c));
        let total = char_len(text);
        assert!(out.contains(&format!("{} chars", comma_format(total))));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
