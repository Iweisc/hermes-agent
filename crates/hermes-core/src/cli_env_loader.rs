//! Helpers for loading Hermes `.env` files consistently across entrypoints.
//!
//! Faithful native port of `hermes_cli/env_loader.py`.
//!
//! Behaviour summary:
//! - `~/.hermes/.env` overrides stale shell-exported values when present.
//! - the project `.env` acts as a dev fallback and only fills missing values
//!   when the user env exists; if no user env exists, the project `.env` also
//!   overrides stale shell vars.
//! - credential-suffixed env vars are stripped of non-ASCII characters after
//!   load (API keys must be pure ASCII since they become HTTP header values),
//!   emitting a one-line warning to stderr the first time a given key is
//!   stripped within a process.
//! - corrupted `.env` files (concatenated KEY=VALUE pairs missing newlines) are
//!   pre-sanitized via [`crate::cli_config::sanitize_env_lines`] before parsing.

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Env var name suffixes that indicate credential values. These are the only
/// env vars whose values we sanitize on load — we must not silently alter
/// arbitrary user env vars, but credentials are known to require pure ASCII
/// (they become HTTP header values).
pub const CREDENTIAL_SUFFIXES: &[&str] = &["_API_KEY", "_TOKEN", "_SECRET", "_KEY"];

/// Names we've already warned about during this process, so repeated
/// [`load_hermes_dotenv`] calls (user env + project env, gateway hot-reload,
/// tests) don't spam the same warning multiple times.
static WARNED_KEYS: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);

fn warned_contains_or_insert(key: &str) -> bool {
    let mut guard = WARNED_KEYS.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(BTreeSet::new);
    if set.contains(key) {
        true
    } else {
        set.insert(key.to_string());
        false
    }
}

/// Reset the per-process warned-key cache. Primarily useful for tests.
pub fn reset_warned_keys() {
    let mut guard = WARNED_KEYS.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(BTreeSet::new());
}

/// Return a compact `U+XXXX ('c'), ...` summary of non-ASCII codepoints.
///
/// Faithful port of `_format_offending_chars`, including dedup, the `limit`
/// cap, and the printable-glyph annotation.
pub fn format_offending_chars(value: &str, limit: usize) -> String {
    let mut seen: Vec<String> = Vec::new();
    for ch in value.chars() {
        if (ch as u32) > 127 {
            let mut label = format!("U+{:04X}", ch as u32);
            if is_printable(ch) {
                label.push_str(&format!(" ({})", py_char_repr(ch)));
            }
            if !seen.iter().any(|s| s == &label) {
                seen.push(label);
            }
            if seen.len() >= limit {
                break;
            }
        }
    }
    seen.join(", ")
}

/// Approximate Python's `str.isprintable()` for a single character.
///
/// Python treats a char as printable when it is not in the "Other" or
/// "Separator" Unicode categories, except for the ASCII space (U+0020).
/// We approximate with Rust's control-character and whitespace checks, which
/// covers the practical cases here (lookalike letters, ZWSP, etc.).
fn is_printable(ch: char) -> bool {
    if ch == ' ' {
        return true;
    }
    if ch.is_control() {
        return false;
    }
    // Zero-width / separator-ish characters Python reports as non-printable.
    if matches!(
        ch as u32,
        0x200B | 0x200C | 0x200D | 0xFEFF | 0x00A0 | 0x2028 | 0x2029
    ) {
        return false;
    }
    // Any other whitespace (besides the ASCII space handled above) is also
    // non-printable per Python's definition.
    !ch.is_whitespace()
}

/// Render a char the way Python's `repr()` would for the `{ch!r}` format spec
/// (single-quoted, with the printable glyph in the common case).
fn py_char_repr(ch: char) -> String {
    format!("'{}'", ch)
}

/// Strip non-ASCII characters from credential env vars in the process
/// environment.
///
/// Called after dotenv loads so the rest of the codebase never sees non-ASCII
/// API keys. Only touches env vars whose names end with a known credential
/// suffix (`_API_KEY`, `_TOKEN`, `_SECRET`, `_KEY`).
///
/// Emits a one-line warning to stderr the first time characters are stripped
/// for a given key. Silent stripping would mask copy-paste corruption (Unicode
/// lookalike glyphs from PDFs / rich-text editors, ZWSP from web pages) as
/// opaque provider-side "invalid API key" errors (see #6843).
pub fn sanitize_loaded_credentials() {
    // Snapshot first so we don't mutate while iterating (mirrors `list(...)`).
    let items: Vec<(String, String)> = std::env::vars().collect();
    for (key, value) in items {
        if !CREDENTIAL_SUFFIXES.iter().any(|suf| key.ends_with(suf)) {
            continue;
        }
        if value.is_ascii() {
            continue;
        }
        let cleaned: String = value.chars().filter(|c| c.is_ascii()).collect();
        // SAFETY: edition 2024 requires set_var to be wrapped in unsafe.
        unsafe {
            std::env::set_var(&key, &cleaned);
        }
        if warned_contains_or_insert(&key) {
            continue;
        }
        // Python counts stripped characters by character length difference.
        let stripped = value.chars().count() - cleaned.chars().count();
        let detail = {
            let d = format_offending_chars(&value, 3);
            if d.is_empty() {
                "non-printable".to_string()
            } else {
                d
            }
        };
        let plural = if stripped != 1 { "s" } else { "" };
        let stderr = std::io::stderr();
        let mut h = stderr.lock();
        let _ = writeln!(
            h,
            "  Warning: {key} contained {stripped} non-ASCII character{plural} ({detail}) — stripped so the key can be sent as an HTTP header."
        );
        let _ = writeln!(
            h,
            "  This usually means the key was copy-pasted from a PDF, rich-text editor, or web page that substituted lookalike\n  Unicode glyphs for ASCII letters. If authentication fails (e.g. \"API key not valid\"), re-copy the key from the\n  provider's dashboard and run `hermes setup` (or edit the .env file in a plain-text editor)."
        );
    }
}

/// Read a `.env` file as text, falling back from UTF-8 to latin-1 on decode
/// failure — mirrors `_load_dotenv_with_fallback`'s `UnicodeDecodeError`
/// handling. latin-1 maps every byte 1:1 to U+0000..=U+00FF.
fn read_env_text(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    match std::str::from_utf8(&bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Ok(bytes.iter().map(|&b| b as char).collect()),
    }
}

/// Parse a single dotenv file into ordered KEY/VALUE pairs.
///
/// Implements the subset of python-dotenv parsing that Hermes `.env` files
/// rely on: `KEY=VALUE` and `export KEY=VALUE`, blank lines and `#` comments,
/// single/double quoted values (with surrounding quotes stripped), and inline
/// comments on unquoted values. Surrounding whitespace around the key and the
/// unquoted value is trimmed.
pub fn parse_dotenv(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let line = line.trim_start();
        let Some(eq) = line.find('=') else {
            continue;
        };
        let key = line[..eq].trim();
        if key.is_empty() {
            continue;
        }
        let rest = &line[eq + 1..];
        let value = parse_value(rest);
        out.push((key.to_string(), value));
    }
    out
}

/// Parse the value side of a `KEY=VALUE` assignment, honouring quoting and
/// inline comments the way python-dotenv does.
fn parse_value(rest: &str) -> String {
    let trimmed = rest.trim_start();
    let bytes = trimmed.as_bytes();
    if let Some(&first) = bytes.first() {
        if first == b'"' || first == b'\'' {
            let quote = first as char;
            // Find the matching closing quote.
            let inner = &trimmed[1..];
            if let Some(end) = inner.find(quote) {
                let value = &inner[..end];
                return if quote == '"' {
                    unescape_double_quoted(value)
                } else {
                    value.to_string()
                };
            }
            // No closing quote — fall through to plain handling.
        }
    }
    // Unquoted: strip an inline comment (` #...`) then trim trailing space.
    let mut value = trimmed;
    if let Some(pos) = find_inline_comment(value) {
        value = &value[..pos];
    }
    value.trim_end().to_string()
}

/// Find the start of an inline comment in an unquoted value. python-dotenv
/// treats a `#` as a comment only when preceded by whitespace (or at start).
fn find_inline_comment(value: &str) -> Option<usize> {
    let chars: Vec<(usize, char)> = value.char_indices().collect();
    for (i, (idx, c)) in chars.iter().enumerate() {
        if *c == '#' {
            if i == 0 {
                return Some(*idx);
            }
            if let Some((_, prev)) = chars.get(i - 1) {
                if prev.is_whitespace() {
                    return Some(*idx);
                }
            }
        }
    }
    None
}

/// Unescape common backslash sequences in a double-quoted dotenv value.
fn unescape_double_quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
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

/// Load a single dotenv file into the process environment, mirroring
/// python-dotenv's `override` semantics, then sanitize credential vars.
///
/// When `override` is false, an existing process env var is preserved (the
/// `.env` value only fills a gap). When true, the `.env` value wins.
fn load_dotenv_with_fallback(path: &Path, override_existing: bool) {
    let text = match read_env_text(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    for (key, value) in parse_dotenv(&text) {
        let exists = std::env::var_os(&key).is_some();
        if exists && !override_existing {
            continue;
        }
        // SAFETY: edition 2024 requires set_var to be wrapped in unsafe.
        unsafe {
            std::env::set_var(&key, &value);
        }
    }
    sanitize_loaded_credentials();
}

/// Pre-sanitize a `.env` file before it is parsed.
///
/// python-dotenv does not handle corrupted lines where multiple KEY=VALUE
/// pairs are concatenated on a single line (missing newline). This produces
/// mangled values — e.g. a bot token duplicated 8× (see #8908).
///
/// Delegates to [`crate::cli_config::sanitize_env_lines`], which knows all
/// valid Hermes env-var names and can split concatenated lines correctly. The
/// rewrite is atomic (temp file + fsync + [`crate::mod_utils::atomic_replace`])
/// and best-effort — any failure is swallowed so gateway startup is never
/// blocked.
pub fn sanitize_env_file_if_needed(path: &Path) {
    if !path.exists() {
        return;
    }
    // utf-8 with errors="replace" — lossy decode keeps the rewrite robust.
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return,
    };
    let text = String::from_utf8_lossy(&bytes);

    // Python uses readlines(): each element keeps its trailing newline. We
    // reproduce that so equality comparison against the sanitizer output (which
    // appends "\n" to each emitted line) matches Python's `sanitized != original`.
    let original = readlines(&text);
    let sanitized = crate::cli_config::sanitize_env_lines(&original);
    if sanitized == original {
        return;
    }

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    if write_atomic(&parent, path, &sanitized).is_err() {
        // best-effort — don't block gateway startup
    }
}

/// Reproduce Python's `file.readlines()`: split keeping the trailing `\n`,
/// and only emit a final entry without a newline when the file does not end
/// in one.
fn readlines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '\n' {
            lines.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Atomically rewrite `path` with `lines`, via a temp file in `parent`,
/// fsync, and [`crate::mod_utils::atomic_replace`]. Cleans up the temp file on
/// any failure (mirroring the `BaseException` cleanup in Python).
fn write_atomic(parent: &Path, path: &Path, lines: &[String]) -> std::io::Result<()> {
    let mut tmp = parent.to_path_buf();
    let unique = format!(
        ".env_{}_{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    tmp.push(unique);

    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        for line in lines {
            f.write_all(line.as_bytes())?;
        }
        f.flush()?;
        f.sync_all()?;
        drop(f);
        crate::mod_utils::atomic_replace(&tmp, path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Resolve the Hermes home directory the way Python does:
/// explicit `hermes_home` arg → `HERMES_HOME` env var → `~/.hermes`.
fn resolve_home(hermes_home: Option<&Path>) -> PathBuf {
    if let Some(h) = hermes_home {
        return h.to_path_buf();
    }
    if let Some(v) = std::env::var_os("HERMES_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    match dirs::home_dir() {
        Some(home) => home.join(".hermes"),
        None => PathBuf::from(".hermes"),
    }
}

/// Load Hermes environment files with user config taking precedence.
///
/// Faithful port of `load_hermes_dotenv`. Returns the list of files that were
/// actually loaded, in load order.
///
/// Behaviour:
/// - `~/.hermes/.env` (or `<hermes_home>/.env`) overrides stale shell-exported
///   values when present.
/// - the project `.env` acts as a dev fallback and only fills missing values
///   when the user env exists.
/// - if no user env exists, the project `.env` also overrides stale shell vars.
pub fn load_hermes_dotenv(
    hermes_home: Option<&Path>,
    project_env: Option<&Path>,
) -> Vec<PathBuf> {
    let mut loaded: Vec<PathBuf> = Vec::new();

    let home_path = resolve_home(hermes_home);
    let user_env = home_path.join(".env");
    let project_env_path: Option<PathBuf> = project_env.map(|p| p.to_path_buf());

    // Fix corrupted .env files before they are parsed (#8908).
    if user_env.exists() {
        sanitize_env_file_if_needed(&user_env);
    }
    if let Some(ref pe) = project_env_path {
        if pe.exists() {
            sanitize_env_file_if_needed(pe);
        }
    }

    if user_env.exists() {
        load_dotenv_with_fallback(&user_env, true);
        loaded.push(user_env);
    }

    if let Some(pe) = project_env_path {
        if pe.exists() {
            // override = not loaded → true only when nothing loaded yet.
            let override_existing = loaded.is_empty();
            load_dotenv_with_fallback(&pe, override_existing);
            loaded.push(pe);
        }
    }

    loaded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests that mutate the global process environment.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn format_offending_chars_basic() {
        // 'ʋ' is U+028B, a printable lookalike for 'v'.
        let s = "abʋc";
        let out = format_offending_chars(s, 3);
        assert!(out.starts_with("U+028B"), "got {out}");
        assert!(out.contains("(\u{0027}ʋ\u{0027})") || out.contains("'ʋ'"), "got {out}");
    }

    #[test]
    fn format_offending_chars_dedups_and_limits() {
        // Repeated codepoint should appear once; limit caps distinct entries.
        let s = "ʋʋʋ\u{200B}\u{2014}\u{2026}";
        let out = format_offending_chars(s, 3);
        let count = out.matches(',').count() + 1;
        assert_eq!(count, 3, "expected 3 entries, got {out}");
        // U+028B should only show once despite three occurrences.
        assert_eq!(out.matches("U+028B").count(), 1, "got {out}");
    }

    #[test]
    fn format_offending_chars_empty_for_ascii() {
        assert_eq!(format_offending_chars("plain-ascii_123", 3), "");
    }

    #[test]
    fn parse_dotenv_basic_and_quotes() {
        let text = "# comment\nFOO=bar\nexport BAZ = qux \nQUOTED=\"hello world\"\nSINGLE='no #comment here'\nINLINE=value # trailing\n";
        let map = parse_dotenv(text);
        let get = |k: &str| map.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("FOO"), Some("bar".to_string()));
        assert_eq!(get("BAZ"), Some("qux".to_string()));
        assert_eq!(get("QUOTED"), Some("hello world".to_string()));
        assert_eq!(get("SINGLE"), Some("no #comment here".to_string()));
        assert_eq!(get("INLINE"), Some("value".to_string()));
    }

    #[test]
    fn parse_dotenv_double_quote_escapes() {
        let text = "MULTI=\"line1\\nline2\"\n";
        let map = parse_dotenv(text);
        assert_eq!(map[0].1, "line1\nline2");
    }

    #[test]
    fn sanitize_loaded_credentials_strips_non_ascii() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_warned_keys();
        let key = "TESTPROV_API_KEY";
        // SAFETY: tests on the global env; serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(key, "sk-ʋalid\u{200B}123");
        }
        sanitize_loaded_credentials();
        let cleaned = std::env::var(key).unwrap();
        assert_eq!(cleaned, "sk-alid123");
        assert!(cleaned.is_ascii());
        // SAFETY: cleanup.
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn sanitize_loaded_credentials_ignores_non_credential_vars() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_warned_keys();
        let key = "TEST_REGULAR_VAR";
        // SAFETY: tests on the global env; serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(key, "valʋe");
        }
        sanitize_loaded_credentials();
        assert_eq!(std::env::var(key).unwrap(), "valʋe");
        // SAFETY: cleanup.
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn sanitize_loaded_credentials_leaves_ascii_untouched() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_warned_keys();
        let key = "TESTASCII_TOKEN";
        // SAFETY: tests on the global env; serialized by ENV_LOCK.
        unsafe {
            std::env::set_var(key, "sk-plain-ascii-123");
        }
        sanitize_loaded_credentials();
        assert_eq!(std::env::var(key).unwrap(), "sk-plain-ascii-123");
        // SAFETY: cleanup.
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn readlines_matches_python_semantics() {
        assert_eq!(readlines("a\nb\n"), vec!["a\n".to_string(), "b\n".to_string()]);
        assert_eq!(readlines("a\nb"), vec!["a\n".to_string(), "b".to_string()]);
        assert_eq!(readlines(""), Vec::<String>::new());
    }

    #[test]
    fn load_hermes_dotenv_user_overrides_shell() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_warned_keys();
        let dir = std::env::temp_dir().join(format!("hermes_envtest_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let env_file = dir.join(".env");
        std::fs::write(&env_file, "HERMES_TEST_OVR=from_file\n").unwrap();

        // Pre-set a stale shell value that the user env must override.
        // SAFETY: serialized by ENV_LOCK.
        unsafe {
            std::env::set_var("HERMES_TEST_OVR", "stale");
        }

        let loaded = load_hermes_dotenv(Some(&dir), None);
        assert_eq!(loaded, vec![env_file.clone()]);
        assert_eq!(std::env::var("HERMES_TEST_OVR").unwrap(), "from_file");

        // SAFETY: cleanup.
        unsafe {
            std::env::remove_var("HERMES_TEST_OVR");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_hermes_dotenv_project_only_overrides_when_no_user_env() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_warned_keys();
        let dir = std::env::temp_dir().join(format!("hermes_envtest2_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        // No user env in this empty home.
        let home = dir.join("home");
        let _ = std::fs::create_dir_all(&home);
        let proj = dir.join("proj.env");
        std::fs::write(&proj, "HERMES_TEST_PROJ=proj_value\n").unwrap();

        // SAFETY: serialized by ENV_LOCK.
        unsafe {
            std::env::set_var("HERMES_TEST_PROJ", "stale");
        }

        let loaded = load_hermes_dotenv(Some(&home), Some(&proj));
        assert_eq!(loaded, vec![proj.clone()]);
        // No user env → project overrides stale shell var.
        assert_eq!(std::env::var("HERMES_TEST_PROJ").unwrap(), "proj_value");

        // SAFETY: cleanup.
        unsafe {
            std::env::remove_var("HERMES_TEST_PROJ");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
