//! Native Rust port of `agent/context_references.py`.
//!
//! Parses and expands `@`-style context references embedded in user messages
//! (`@file:...`, `@folder:...`, `@diff`, `@staged`, `@git:N`, `@url:...`) into
//! attached context blocks, with token-budget enforcement and sensitive-path
//! guards.
//!
//! Behaviour mirrors the Python implementation closely. The async URL-fetcher
//! plumbing from Python collapses to a synchronous `Fn(&str) -> String`
//! callback here (Rust has no need for the asyncio thread-pool shim).

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;

use crate::ag_model_metadata::estimate_tokens_rough;

const TRAILING_PUNCTUATION: &[char] = &[',', '.', ';', '!', '?'];

const SENSITIVE_HOME_DIRS: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".azure",
    ".config/gh",
];

/// Directories relative to the Hermes home that must never be attached.
const SENSITIVE_HERMES_DIRS: &[&str] = &["skills/.hub"];

const SENSITIVE_HOME_FILES: &[&str] = &[
    ".ssh/authorized_keys",
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".ssh/config",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".zprofile",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
];

// Quoted reference value: backtick, double, or single quoted, no embedded
// matching quote / newline.
const QUOTED_REFERENCE_VALUE: &str = r#"(?:`[^`\n]+`|"[^"\n]+"|'[^'\n]+')"#;

/// The master reference-matching pattern. Equivalent to the Python
/// `REFERENCE_PATTERN`, except the leading negative-lookbehind `(?<![\w/])` is
/// emulated manually in [`finditer_references`] (the `regex` crate has no
/// lookbehind support).
static REFERENCE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    let pat = format!(
        r"@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>{}(?::\d+(?:-\d+)?)?|\S+))",
        QUOTED_REFERENCE_VALUE
    );
    Regex::new(&pat).expect("reference pattern compiles")
});

static QUOTED_FILE_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    // Equivalent to: ^(quote)(path)(quote)(:start(-end)?)?$ with a backreference
    // to the quote. The `regex` crate lacks backreferences, so we try each
    // quote character explicitly via three patterns in `parse_file_reference_value`.
    Regex::new(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$").expect("range pattern compiles")
});

static WHITESPACE_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s{2,}").expect("ws run compiles"));
static SPACE_BEFORE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+([,.;:!?])").expect("space-before-punct compiles"));

/// A single parsed `@` reference. Equivalent to the Python `ContextReference`
/// frozen dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextReference {
    pub raw: String,
    pub kind: String,
    pub target: String,
    /// Byte offset (into the original message) where the match starts.
    pub start: usize,
    /// Byte offset where the match ends.
    pub end: usize,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
}

/// Result of expanding references in a message. Equivalent to the Python
/// `ContextReferenceResult` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextReferenceResult {
    pub message: String,
    pub original_message: String,
    pub references: Vec<ContextReference>,
    pub warnings: Vec<String>,
    pub injected_tokens: i64,
    pub expanded: bool,
    pub blocked: bool,
}

impl ContextReferenceResult {
    fn passthrough(message: &str) -> Self {
        ContextReferenceResult {
            message: message.to_string(),
            original_message: message.to_string(),
            references: Vec::new(),
            warnings: Vec::new(),
            injected_tokens: 0,
            expanded: false,
            blocked: false,
        }
    }
}

/// Synchronous URL fetcher callback. Returns the extracted content (may be
/// empty). Equivalent to the Python `url_fetcher` parameter (the async variant
/// collapses to sync here).
pub type UrlFetcher<'a> = dyn Fn(&str) -> String + 'a;

/// Parse all `@` references from `message`. Equivalent to Python
/// `parse_context_references`.
pub fn parse_context_references(message: &str) -> Vec<ContextReference> {
    let mut refs: Vec<ContextReference> = Vec::new();
    if message.is_empty() {
        return refs;
    }

    for m in finditer_references(message) {
        let start = m.start;
        let end = m.end;
        let raw = message[start..end].to_string();

        if let Some(simple) = m.simple.clone() {
            refs.push(ContextReference {
                raw,
                kind: simple,
                target: String::new(),
                start,
                end,
                line_start: None,
                line_end: None,
            });
            continue;
        }

        let kind = m.kind.clone().unwrap_or_default();
        let value = strip_trailing_punctuation(m.value.as_deref().unwrap_or(""));
        let mut line_start: Option<usize> = None;
        let mut line_end: Option<usize> = None;
        let mut target = strip_reference_wrappers(&value);

        if kind == "file" {
            let (t, ls, le) = parse_file_reference_value(&value);
            target = t;
            line_start = ls;
            line_end = le;
        }

        refs.push(ContextReference {
            raw,
            kind,
            target,
            start,
            end,
            line_start,
            line_end,
        });
    }

    refs
}

struct RawMatch {
    start: usize,
    end: usize,
    simple: Option<String>,
    kind: Option<String>,
    value: Option<String>,
}

/// Iterate references while emulating the Python negative-lookbehind
/// `(?<![\w/])` before each `@`.
fn finditer_references(message: &str) -> Vec<RawMatch> {
    let bytes = message.as_bytes();
    let mut out: Vec<RawMatch> = Vec::new();
    for caps in REFERENCE_PATTERN.captures_iter(message) {
        let whole = caps.get(0).unwrap();
        let start = whole.start();
        // Emulate (?<![\w/]): the char immediately before the '@' must not be a
        // word char or '/'.
        if start > 0 {
            let prev = preceding_char(message, start);
            if let Some(c) = prev {
                if c == '/' || c == '_' || c.is_alphanumeric() {
                    continue;
                }
            }
        }
        let _ = bytes;
        out.push(RawMatch {
            start,
            end: whole.end(),
            simple: caps.name("simple").map(|m| m.as_str().to_string()),
            kind: caps.name("kind").map(|m| m.as_str().to_string()),
            value: caps.name("value").map(|m| m.as_str().to_string()),
        });
    }
    out
}

/// Return the char immediately preceding byte offset `idx`.
fn preceding_char(message: &str, idx: usize) -> Option<char> {
    message[..idx].chars().next_back()
}

/// Synchronous entry point. Equivalent to Python `preprocess_context_references`
/// / `preprocess_context_references_async` (collapsed; no event loop needed).
pub fn preprocess_context_references(
    message: &str,
    cwd: &Path,
    context_length: i64,
    url_fetcher: Option<&UrlFetcher>,
    allowed_root: Option<&Path>,
) -> ContextReferenceResult {
    let refs = parse_context_references(message);
    if refs.is_empty() {
        return ContextReferenceResult::passthrough(message);
    }

    let cwd_path = expanduser_resolve(cwd);
    // Default to the current working directory so @ references cannot escape
    // the active workspace unless a caller explicitly widens the root.
    let allowed_root_path = match allowed_root {
        Some(p) => expanduser_resolve(p),
        None => cwd_path.clone(),
    };

    let mut warnings: Vec<String> = Vec::new();
    let mut blocks: Vec<String> = Vec::new();
    let mut injected_tokens: i64 = 0;

    for r in &refs {
        let (warning, block) =
            expand_reference(r, &cwd_path, url_fetcher, Some(&allowed_root_path));
        if let Some(w) = warning {
            warnings.push(w);
        }
        if let Some(b) = block {
            injected_tokens += estimate_tokens_rough(&b);
            blocks.push(b);
        }
    }

    let hard_limit = std::cmp::max(1, (context_length as f64 * 0.50) as i64);
    let soft_limit = std::cmp::max(1, (context_length as f64 * 0.25) as i64);

    if injected_tokens > hard_limit {
        warnings.push(format!(
            "@ context injection refused: {} tokens exceeds the 50% hard limit ({}).",
            injected_tokens, hard_limit
        ));
        return ContextReferenceResult {
            message: message.to_string(),
            original_message: message.to_string(),
            references: refs,
            warnings,
            injected_tokens,
            expanded: false,
            blocked: true,
        };
    }

    if injected_tokens > soft_limit {
        warnings.push(format!(
            "@ context injection warning: {} tokens exceeds the 25% soft limit ({}).",
            injected_tokens, soft_limit
        ));
    }

    let stripped = remove_reference_tokens(message, &refs);
    let mut final_text = stripped;
    if !warnings.is_empty() {
        let joined = warnings
            .iter()
            .map(|w| format!("- {}", w))
            .collect::<Vec<_>>()
            .join("\n");
        final_text = format!("{}\n\n--- Context Warnings ---\n{}", final_text, joined);
    }
    if !blocks.is_empty() {
        final_text = format!(
            "{}\n\n--- Attached Context ---\n\n{}",
            final_text,
            blocks.join("\n\n")
        );
    }

    let expanded = !blocks.is_empty() || !warnings.is_empty();
    ContextReferenceResult {
        message: final_text.trim().to_string(),
        original_message: message.to_string(),
        references: refs,
        warnings,
        injected_tokens,
        expanded,
        blocked: false,
    }
}

/// Expand a single reference. Returns `(warning, block)`. Equivalent to Python
/// `_expand_reference`.
fn expand_reference(
    r: &ContextReference,
    cwd: &Path,
    url_fetcher: Option<&UrlFetcher>,
    allowed_root: Option<&Path>,
) -> (Option<String>, Option<String>) {
    let result: Result<(Option<String>, Option<String>), String> = (|| match r.kind.as_str() {
        "file" => expand_file_reference(r, cwd, allowed_root),
        "folder" => expand_folder_reference(r, cwd, allowed_root),
        "diff" => expand_git_reference(r, cwd, &["diff"], "git diff"),
        "staged" => expand_git_reference(r, cwd, &["diff", "--staged"], "git diff --staged"),
        "git" => {
            let parsed: i64 = if r.target.is_empty() {
                1
            } else {
                r.target.parse::<i64>().map_err(|e| e.to_string())?
            };
            let count = parsed.clamp(1, 10);
            let count_arg = format!("-{}", count);
            let label = format!("git log -{} -p", count);
            expand_git_reference(r, cwd, &["log", &count_arg, "-p"], &label)
        }
        "url" => {
            let content = fetch_url_content(&r.target, url_fetcher);
            if content.is_empty() {
                Ok((Some(format!("{}: no content extracted", r.raw)), None))
            } else {
                Ok((
                    None,
                    Some(format!(
                        "🌐 {} ({} tokens)\n{}",
                        r.raw,
                        estimate_tokens_rough(&content),
                        content
                    )),
                ))
            }
        }
        _ => Ok((Some(format!("{}: unsupported reference type", r.raw)), None)),
    })();

    match result {
        Ok(v) => v,
        Err(e) => (Some(format!("{}: {}", r.raw, e)), None),
    }
}

fn expand_file_reference(
    r: &ContextReference,
    cwd: &Path,
    allowed_root: Option<&Path>,
) -> Result<(Option<String>, Option<String>), String> {
    let path = resolve_path(cwd, &r.target, allowed_root)?;
    ensure_reference_path_allowed(&path)?;
    if !path.exists() {
        return Ok((Some(format!("{}: file not found", r.raw)), None));
    }
    if !path.is_file() {
        return Ok((Some(format!("{}: path is not a file", r.raw)), None));
    }
    if is_binary_file(&path) {
        return Ok((
            Some(format!("{}: binary files are not supported", r.raw)),
            None,
        ));
    }

    let mut text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    if let Some(line_start) = r.line_start {
        let lines: Vec<&str> = splitlines(&text);
        let start_idx = line_start.saturating_sub(1);
        let end_idx = std::cmp::min(r.line_end.unwrap_or(line_start), lines.len());
        let slice: Vec<&str> = if start_idx < end_idx {
            lines[start_idx..end_idx].to_vec()
        } else {
            Vec::new()
        };
        text = slice.join("\n");
    }

    let lang = code_fence_language(&path);
    let label = &r.raw;
    Ok((
        None,
        Some(format!(
            "📄 {} ({} tokens)\n```{}\n{}\n```",
            label,
            estimate_tokens_rough(&text),
            lang,
            text
        )),
    ))
}

fn expand_folder_reference(
    r: &ContextReference,
    cwd: &Path,
    allowed_root: Option<&Path>,
) -> Result<(Option<String>, Option<String>), String> {
    let path = resolve_path(cwd, &r.target, allowed_root)?;
    ensure_reference_path_allowed(&path)?;
    if !path.exists() {
        return Ok((Some(format!("{}: folder not found", r.raw)), None));
    }
    if !path.is_dir() {
        return Ok((Some(format!("{}: path is not a folder", r.raw)), None));
    }

    let listing = build_folder_listing(&path, cwd, 200);
    Ok((
        None,
        Some(format!(
            "📁 {} ({} tokens)\n{}",
            r.raw,
            estimate_tokens_rough(&listing),
            listing
        )),
    ))
}

fn expand_git_reference(
    r: &ContextReference,
    cwd: &Path,
    args: &[&str],
    label: &str,
) -> Result<(Option<String>, Option<String>), String> {
    let mut cmd = Command::new("git");
    cmd.args(args).current_dir(cwd);

    let output = run_with_timeout(&mut cmd, Duration::from_secs(30));
    let output = match output {
        Ok(Some(o)) => o,
        Ok(None) => {
            return Ok((
                Some(format!("{}: git command timed out (30s)", r.raw)),
                None,
            ))
        }
        Err(e) => return Ok((Some(format!("{}: {}", r.raw, e)), None)),
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let msg = if stderr.is_empty() {
            "git command failed"
        } else {
            stderr
        };
        return Ok((Some(format!("{}: {}", r.raw, msg)), None));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut content = stdout.trim().to_string();
    if content.is_empty() {
        content = "(no output)".to_string();
    }
    Ok((
        None,
        Some(format!(
            "🧾 {} ({} tokens)\n```diff\n{}\n```",
            label,
            estimate_tokens_rough(&content),
            content
        )),
    ))
}

/// Run a command with a wall-clock timeout. Returns `Ok(Some(output))` on
/// completion, `Ok(None)` on timeout, `Err` on spawn failure.
fn run_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> Result<Option<std::process::Output>, String> {
    use std::io::Read;
    use std::process::Stdio;
    use std::thread;

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    // Poll for completion so we can kill the child on timeout.
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                let mut out = Vec::new();
                let mut err = Vec::new();
                if let Some(p) = stdout_pipe.as_mut() {
                    let _ = p.read_to_end(&mut out);
                }
                if let Some(p) = stderr_pipe.as_mut() {
                    let _ = p.read_to_end(&mut err);
                }
                return Ok(Some(std::process::Output {
                    status,
                    stdout: out,
                    stderr: err,
                }));
            }
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
}

/// Fetch and trim URL content via the supplied fetcher (no default fetcher is
/// available in the native build — the Python default delegated to
/// `tools.web_tools.web_extract_tool`).
fn fetch_url_content(url: &str, url_fetcher: Option<&UrlFetcher>) -> String {
    match url_fetcher {
        Some(f) => f(url).trim().to_string(),
        None => String::new(),
    }
}

/// Resolve a reference target against `cwd`, enforcing `allowed_root`
/// containment. Equivalent to Python `_resolve_path`.
fn resolve_path(cwd: &Path, target: &str, allowed_root: Option<&Path>) -> Result<PathBuf, String> {
    let expanded = expanduser(target);
    let path = if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    };
    let resolved = normalize_path(&path);

    if let Some(root) = allowed_root {
        if !resolved.starts_with(root) {
            return Err("path is outside the allowed workspace".to_string());
        }
    }
    Ok(resolved)
}

/// Reject sensitive credential / internal Hermes paths. Equivalent to Python
/// `_ensure_reference_path_allowed`.
fn ensure_reference_path_allowed(path: &Path) -> Result<(), String> {
    let home = expanduser_resolve(Path::new("~"));
    let hermes_home = normalize_path(&crate::mod_hermes_constants::get_hermes_home());

    let mut blocked_exact: Vec<PathBuf> = SENSITIVE_HOME_FILES
        .iter()
        .map(|rel| home.join(rel))
        .collect();
    blocked_exact.push(hermes_home.join(".env"));

    let mut blocked_dirs: Vec<PathBuf> =
        SENSITIVE_HOME_DIRS.iter().map(|rel| home.join(rel)).collect();
    blocked_dirs.extend(SENSITIVE_HERMES_DIRS.iter().map(|rel| hermes_home.join(rel)));

    if blocked_exact.iter().any(|b| b == path) {
        return Err("path is a sensitive credential file and cannot be attached".to_string());
    }

    for blocked_dir in &blocked_dirs {
        if path.starts_with(blocked_dir) {
            return Err(
                "path is a sensitive credential or internal Hermes path and cannot be attached"
                    .to_string(),
            );
        }
    }

    Ok(())
}

/// Equivalent to Python `_strip_trailing_punctuation`.
fn strip_trailing_punctuation(value: &str) -> String {
    let mut stripped: String = value
        .trim_end_matches(|c| TRAILING_PUNCTUATION.contains(&c))
        .to_string();

    loop {
        let last = stripped.chars().next_back();
        let closer = match last {
            Some(c @ (')' | ']' | '}')) => c,
            _ => break,
        };
        let opener = match closer {
            ')' => '(',
            ']' => '[',
            '}' => '{',
            _ => unreachable!(),
        };
        let closer_count = stripped.matches(closer).count();
        let opener_count = stripped.matches(opener).count();
        if closer_count > opener_count {
            // Drop the trailing closer.
            let mut chars = stripped.chars();
            chars.next_back();
            stripped = chars.as_str().to_string();
            continue;
        }
        break;
    }
    stripped
}

/// Equivalent to Python `_strip_reference_wrappers`.
fn strip_reference_wrappers(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 2 {
        let first = chars[0];
        let last = chars[chars.len() - 1];
        if first == last && (first == '`' || first == '"' || first == '\'') {
            return chars[1..chars.len() - 1].iter().collect();
        }
    }
    value.to_string()
}

/// Equivalent to Python `_parse_file_reference_value`.
fn parse_file_reference_value(value: &str) -> (String, Option<usize>, Option<usize>) {
    // Try quoted forms (backtick, double, single) with optional :start(-end)?.
    for quote in ['`', '"', '\''] {
        if let Some(parsed) = try_quoted_file_value(value, quote) {
            return parsed;
        }
    }

    // Unquoted "path:start(-end)?" form.
    if let Some(caps) = QUOTED_FILE_VALUE.captures(value) {
        let path = caps.name("path").unwrap().as_str().to_string();
        let start: usize = caps.name("start").unwrap().as_str().parse().unwrap_or(0);
        let end: usize = caps
            .name("end")
            .map(|m| m.as_str().parse().unwrap_or(start))
            .unwrap_or(start);
        return (path, Some(start), Some(end));
    }

    (strip_reference_wrappers(value), None, None)
}

/// Match `^(quote)(path)(quote)(:start(-end)?)?$` for a single `quote` char.
fn try_quoted_file_value(
    value: &str,
    quote: char,
) -> Option<(String, Option<usize>, Option<usize>)> {
    let q = regex::escape(&quote.to_string());
    let pat = format!(r"^{q}(?P<path>.+?){q}(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$");
    let re = Regex::new(&pat).ok()?;
    let caps = re.captures(value)?;
    let path = caps.name("path")?.as_str().to_string();
    let start = caps.name("start").map(|m| m.as_str().parse().unwrap_or(0));
    let line_end = match start {
        Some(s) => Some(
            caps.name("end")
                .map(|m| m.as_str().parse().unwrap_or(s))
                .unwrap_or(s),
        ),
        None => None,
    };
    Some((path, start, line_end))
}

/// Equivalent to Python `_remove_reference_tokens`.
fn remove_reference_tokens(message: &str, refs: &[ContextReference]) -> String {
    let mut pieces: Vec<&str> = Vec::new();
    let mut cursor = 0usize;
    for r in refs {
        pieces.push(&message[cursor..r.start]);
        cursor = r.end;
    }
    pieces.push(&message[cursor..]);
    let joined: String = pieces.concat();

    let collapsed = WHITESPACE_RUN.replace_all(&joined, " ");
    let no_space_punct = SPACE_BEFORE_PUNCT.replace_all(&collapsed, "$1");
    no_space_punct.trim().to_string()
}

/// Equivalent to Python `_is_binary_file`.
fn is_binary_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let text_exts = [
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts",
    ];
    if let Some(mime) = guess_mime(name) {
        if !mime.starts_with("text/") && !text_exts.iter().any(|ext| name.ends_with(ext)) {
            return true;
        }
    }
    match std::fs::read(path) {
        Ok(bytes) => {
            let chunk = &bytes[..std::cmp::min(4096, bytes.len())];
            chunk.contains(&0u8)
        }
        Err(_) => false,
    }
}

/// Minimal MIME guesser covering the cases the Python `mimetypes` module hits
/// for the extensions exercised by this module's logic. Returns `None` when the
/// type is unknown (matching `mimetypes.guess_type` returning `(None, None)`).
fn guess_mime(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let ext = lower.rsplit_once('.').map(|(_, e)| e)?;
    let mime = match ext {
        "txt" | "text" | "conf" | "def" | "list" | "log" | "in" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "csv" => "text/csv",
        "xml" => "text/xml",
        "py" => "text/x-python",
        "js" | "mjs" => "text/javascript",
        "json" => "application/json",
        "md" | "markdown" => "text/markdown",
        "yaml" | "yml" => "application/x-yaml",
        "toml" => "application/toml",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "zip" => "application/zip",
        "gz" | "tgz" => "application/gzip",
        "tar" => "application/x-tar",
        "wasm" => "application/wasm",
        "mp3" => "audio/mpeg",
        "wav" => "audio/x-wav",
        "mp4" => "video/mp4",
        "bin" | "exe" | "so" | "dll" | "o" | "a" => "application/octet-stream",
        _ => return None,
    };
    Some(mime)
}

/// Equivalent to Python `_build_folder_listing`.
fn build_folder_listing(path: &Path, cwd: &Path, limit: usize) -> String {
    let path_rel = rel_to(path, cwd);
    let path_rel_parts = path_components_count(&path_rel);
    let mut lines: Vec<String> = vec![format!("{}/", path_rel.display())];

    let entries = iter_visible_entries(path, cwd, limit);
    for entry in &entries {
        let rel = rel_to(entry, cwd);
        let rel_parts = path_components_count(&rel);
        let indent_units = rel_parts.saturating_sub(path_rel_parts + 1);
        let indent = "  ".repeat(indent_units);
        let name = entry
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if entry.is_dir() {
            lines.push(format!("{}- {}/", indent, name));
        } else {
            let meta = file_metadata(entry);
            lines.push(format!("{}- {} ({})", indent, name, meta));
        }
    }
    if entries.len() >= limit {
        lines.push("- ...".to_string());
    }
    lines.join("\n")
}

/// Equivalent to Python `_iter_visible_entries`.
fn iter_visible_entries(path: &Path, cwd: &Path, limit: usize) -> Vec<PathBuf> {
    if let Some(rg_entries) = rg_files(path, cwd, limit) {
        let mut output: Vec<PathBuf> = Vec::new();
        let mut seen_dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for rel in &rg_entries {
            let full = cwd.join(rel);
            // Add ancestor directories within `path` that we have not seen.
            for parent in full.ancestors().skip(1) {
                if parent == cwd || seen_dirs.contains(parent) {
                    continue;
                }
                // `path not in {parent, *parent.parents}` -> skip when `path`
                // is an ancestor-or-equal of `parent` is FALSE; i.e. only keep
                // parents that are descendants of `path`.
                if !is_within(parent, path) || parent == path {
                    continue;
                }
                seen_dirs.insert(parent.to_path_buf());
                output.push(parent.to_path_buf());
            }
            output.push(full);
        }
        // sorted({existing}, key=(not is_dir, str(p)))
        let mut deduped: BTreeSet<PathBuf> = BTreeSet::new();
        for p in output {
            if p.exists() {
                deduped.insert(p);
            }
        }
        let mut sorted: Vec<PathBuf> = deduped.into_iter().collect();
        sorted.sort_by(|a, b| {
            let ka = (!a.is_dir(), a.to_string_lossy().to_string());
            let kb = (!b.is_dir(), b.to_string_lossy().to_string());
            ka.cmp(&kb)
        });
        return sorted;
    }

    // Fallback: manual walk (os.walk equivalent).
    let mut output: Vec<PathBuf> = Vec::new();
    walk_dir(path, limit, &mut output);
    output
}

/// `os.walk`-equivalent traversal that skips dotfiles and `__pycache__`,
/// emitting directories then files per level, capped at `limit`.
fn walk_dir(root: &Path, limit: usize, output: &mut Vec<PathBuf>) {
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let read = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut dirs: Vec<PathBuf> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in read.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let p = entry.path();
            if p.is_dir() {
                if name.starts_with('.') || name == "__pycache__" {
                    continue;
                }
                dirs.push(p);
            } else {
                if name.starts_with('.') {
                    continue;
                }
                files.push(p);
            }
        }
        dirs.sort_by_key(|p| p.file_name().map(|n| n.to_os_string()));
        files.sort_by_key(|p| p.file_name().map(|n| n.to_os_string()));

        for d in &dirs {
            output.push(d.clone());
            if output.len() >= limit {
                return;
            }
        }
        for f in &files {
            output.push(f.clone());
            if output.len() >= limit {
                return;
            }
        }
        // Recurse into subdirectories (push in reverse to preserve order).
        for d in dirs.into_iter().rev() {
            stack.push(d);
        }
    }
}

/// Equivalent to Python `_rg_files`. Returns `None` when `rg` is unavailable or
/// fails (signalling the caller to fall back to a manual walk).
fn rg_files(path: &Path, cwd: &Path, limit: usize) -> Option<Vec<PathBuf>> {
    let rel = rel_to(path, cwd);
    let mut cmd = Command::new("rg");
    cmd.arg("--files")
        .arg(rel.to_string_lossy().to_string())
        .current_dir(cwd);
    let output = match run_with_timeout(&mut cmd, Duration::from_secs(10)) {
        Ok(Some(o)) => o,
        Ok(None) => return None,
        Err(_) => return None,
    };
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<PathBuf> = stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();
    Some(files.into_iter().take(limit).collect())
}

/// Equivalent to Python `_file_metadata`.
fn file_metadata(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if is_binary_file(path) {
        return format!("{} bytes", size);
    }
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let line_count = text.matches('\n').count() + 1;
            format!("{} lines", line_count)
        }
        Err(_) => format!("{} bytes", size),
    }
}

/// Equivalent to Python `_code_fence_language`.
fn code_fence_language(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("py") => "python",
        Some("js") => "javascript",
        Some("ts") => "typescript",
        Some("tsx") => "tsx",
        Some("jsx") => "jsx",
        Some("json") => "json",
        Some("md") => "markdown",
        Some("sh") => "bash",
        Some("yml") => "yaml",
        Some("yaml") => "yaml",
        Some("toml") => "toml",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Expand a leading `~` to the user's home directory (Python `os.path.expanduser`).
fn expanduser(target: &str) -> PathBuf {
    if target == "~" {
        if let Some(h) = dirs::home_dir() {
            return h;
        }
    } else if let Some(rest) = target.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest);
        }
    }
    PathBuf::from(target)
}

/// Expand `~` then logically normalize (Python `Path(...).expanduser().resolve()`).
/// We use lexical normalization plus `canonicalize` when the path exists, so
/// the result is stable for both existing and non-existing paths.
fn expanduser_resolve(p: &Path) -> PathBuf {
    let expanded = if let Some(s) = p.to_str() {
        expanduser(s)
    } else {
        p.to_path_buf()
    };
    normalize_path(&expanded)
}

/// Resolve to an absolute, lexically-normalized path. Uses `canonicalize` when
/// possible (resolves symlinks like Python's `resolve()`), otherwise falls back
/// to manual normalization against the current dir.
fn normalize_path(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let base = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(p)
    };
    lexical_normalize(&base)
}

/// Collapse `.` and `..` components without touching the filesystem.
fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out: Vec<Component> = Vec::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.last(), Some(Component::Normal(_))) {
                    out.pop();
                } else {
                    out.push(comp);
                }
            }
            other => out.push(other),
        }
    }
    let mut result = PathBuf::new();
    for c in out {
        result.push(c.as_os_str());
    }
    result
}

/// Return `path` relative to `base` (Python `path.relative_to(base)`), falling
/// back to `path` itself when not a descendant.
fn rel_to(path: &Path, base: &Path) -> PathBuf {
    path.strip_prefix(base)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| path.to_path_buf())
}

/// True when `path` is `base` or a descendant of `base`.
fn is_within(path: &Path, base: &Path) -> bool {
    path == base || path.starts_with(base)
}

fn path_components_count(p: &Path) -> usize {
    p.components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .count()
}

/// Python `str.splitlines()` for the line-range slicing case. Splits on `\n`
/// and drops a single trailing empty element produced by a final newline,
/// matching CPython's behaviour closely enough for line indexing.
fn splitlines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut parts: Vec<&str> = text.split('\n').collect();
    if let Some(last) = parts.last() {
        if last.is_empty() {
            parts.pop();
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_diff_and_staged() {
        let refs = parse_context_references("show me @diff and @staged please");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].kind, "diff");
        assert_eq!(refs[0].raw, "@diff");
        assert_eq!(refs[1].kind, "staged");
        assert!(refs[0].target.is_empty());
    }

    #[test]
    fn lookbehind_blocks_email_and_path() {
        // '@' preceded by a word char (email) or '/' must not match.
        let refs = parse_context_references("contact me at user@diff.com");
        assert!(refs.is_empty());
        let refs2 = parse_context_references("path/@diff");
        assert!(refs2.is_empty());
    }

    #[test]
    fn parses_file_reference_with_lines() {
        let refs = parse_context_references("look at @file:src/main.rs:10-20 now");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].kind, "file");
        assert_eq!(refs[0].target, "src/main.rs");
        assert_eq!(refs[0].line_start, Some(10));
        assert_eq!(refs[0].line_end, Some(20));
    }

    #[test]
    fn parses_file_reference_single_line() {
        let refs = parse_context_references("@file:a.txt:5");
        assert_eq!(refs[0].line_start, Some(5));
        assert_eq!(refs[0].line_end, Some(5));
    }

    #[test]
    fn parses_quoted_file_reference() {
        let refs = parse_context_references("see @file:\"my file.txt\":3-4 here");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].target, "my file.txt");
        assert_eq!(refs[0].line_start, Some(3));
        assert_eq!(refs[0].line_end, Some(4));
    }

    #[test]
    fn parses_git_count() {
        let refs = parse_context_references("@git:5");
        assert_eq!(refs[0].kind, "git");
        assert_eq!(refs[0].target, "5");
    }

    #[test]
    fn parses_url_reference() {
        let refs = parse_context_references("read @url:https://example.com please");
        assert_eq!(refs[0].kind, "url");
        assert_eq!(refs[0].target, "https://example.com");
    }

    #[test]
    fn strip_trailing_punctuation_keeps_balanced_parens() {
        assert_eq!(strip_trailing_punctuation("foo(bar)"), "foo(bar)");
        assert_eq!(strip_trailing_punctuation("foo,"), "foo");
        // Unbalanced trailing closer is dropped.
        assert_eq!(strip_trailing_punctuation("foo)"), "foo");
        assert_eq!(strip_trailing_punctuation("foo."), "foo");
    }

    #[test]
    fn strip_reference_wrappers_removes_matching_quotes() {
        assert_eq!(strip_reference_wrappers("`x`"), "x");
        assert_eq!(strip_reference_wrappers("\"x\""), "x");
        assert_eq!(strip_reference_wrappers("'x'"), "x");
        assert_eq!(strip_reference_wrappers("x"), "x");
    }

    #[test]
    fn remove_reference_tokens_collapses_whitespace() {
        let refs = parse_context_references("hello @diff world");
        let out = remove_reference_tokens("hello @diff world", &refs);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn remove_reference_tokens_fixes_punctuation() {
        let refs = parse_context_references("see @diff , done");
        let out = remove_reference_tokens("see @diff , done", &refs);
        assert_eq!(out, "see, done");
    }

    #[test]
    fn no_references_passthrough() {
        let cwd = std::env::temp_dir();
        let res = preprocess_context_references("just text", &cwd, 1000, None, None);
        assert_eq!(res.message, "just text");
        assert!(!res.expanded);
        assert!(!res.blocked);
        assert!(res.references.is_empty());
    }

    #[test]
    fn expands_file_reference_block() {
        let dir = std::env::temp_dir().join(format!("ctxref_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("hello.py");
        std::fs::write(&file, "print('hi')\n").unwrap();

        let res = preprocess_context_references(
            "look @file:hello.py here",
            &dir,
            100_000,
            None,
            Some(&dir),
        );
        assert!(res.expanded);
        assert!(!res.blocked);
        assert!(res.message.contains("📄 @file:hello.py"));
        assert!(res.message.contains("```python"));
        assert!(res.message.contains("print('hi')"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_outside_allowed_root_warns() {
        let dir = std::env::temp_dir().join(format!("ctxref_root_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Reference an absolute path outside the allowed root.
        let res = preprocess_context_references(
            "@file:/etc/hosts done",
            &dir,
            100_000,
            None,
            Some(&dir),
        );
        // Should produce a warning about being outside the workspace, not a block.
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("outside the allowed workspace")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn url_reference_uses_fetcher() {
        let cwd = std::env::temp_dir();
        let fetcher = |_url: &str| "fetched body".to_string();
        let res = preprocess_context_references(
            "@url:https://x.test now",
            &cwd,
            100_000,
            Some(&fetcher),
            None,
        );
        assert!(res.message.contains("🌐 @url:https://x.test"));
        assert!(res.message.contains("fetched body"));
    }

    #[test]
    fn url_reference_empty_content_warns() {
        let cwd = std::env::temp_dir();
        let fetcher = |_url: &str| "   ".to_string();
        let res = preprocess_context_references(
            "@url:https://x.test now",
            &cwd,
            100_000,
            Some(&fetcher),
            None,
        );
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("no content extracted")));
    }

    #[test]
    fn hard_limit_blocks_injection() {
        let dir = std::env::temp_dir().join(format!("ctxref_lim_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("big.txt");
        std::fs::write(&file, "x".repeat(10_000)).unwrap();

        // Tiny context length so even small injection exceeds the 50% hard limit.
        let res =
            preprocess_context_references("@file:big.txt", &dir, 4, None, Some(&dir));
        assert!(res.blocked);
        assert_eq!(res.message, "@file:big.txt");
        assert!(res.warnings.iter().any(|w| w.contains("hard limit")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn code_fence_language_mapping() {
        assert_eq!(code_fence_language(Path::new("a.py")), "python");
        assert_eq!(code_fence_language(Path::new("a.TS")), "typescript");
        assert_eq!(code_fence_language(Path::new("a.unknown")), "");
    }

    #[test]
    fn parse_file_value_unquoted_no_lines() {
        let (t, s, e) = parse_file_reference_value("path/to/file.rs");
        assert_eq!(t, "path/to/file.rs");
        assert_eq!(s, None);
        assert_eq!(e, None);
    }
}
