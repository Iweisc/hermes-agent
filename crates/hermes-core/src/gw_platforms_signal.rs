//! Signal messenger platform adapter, ported from `gateway/platforms/signal.py`.
//!
//! The Python module connects to a `signal-cli` daemon running in HTTP mode:
//! inbound messages arrive via SSE (Server-Sent Events) streaming, and outbound
//! messages / actions use JSON-RPC 2.0 over HTTP.
//!
//! This port reproduces the *pure*, behaviour-defining logic faithfully and
//! idiomatically:
//!   - magic-byte extension guessing + extension/MIME classification,
//!   - Signal-service-id / E.164 recognition and recipient-cache mapping,
//!   - mention rendering (`￼` placeholder substitution),
//!   - the full markdown → Signal `bodyRanges` (textStyles) converter, including
//!     UTF-16 code-unit offset computation,
//!   - envelope parsing (`_handle_envelope`) into a normalized [`crate::gw_platforms_base::MessageEvent`],
//!   - JSON-RPC payload construction (`_rpc`, `send`, `sendTyping`, `sendReaction`, …)
//!     and response parsing, with the typing-indicator backoff state machine,
//!   - the per-chat typing failure/cooldown state machine.
//!
//! Network calls use `reqwest::blocking`; the SSE listener / health-monitor
//! asyncio task orchestration is modelled as synchronous state + helpers rather
//! than dragging in an event loop.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageEvent`, `MessageType`, `SendResult`,
//!     `ProcessingOutcome`, `SessionSource`, cache helpers.
//!   - [`crate::gw_signal_rate_limit`] — scheduler + rate-limit detection.
//!   - [`crate::gw_helpers::redact_phone`] — phone redaction for logs.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde_json::{json, Map, Value};

use crate::gw_platforms_base::{
    MessageEvent, MessageType, ProcessingOutcome, SendResult, SessionSource,
};
use crate::gw_signal_rate_limit::{
    extract_retry_after_seconds, is_signal_rate_limit_error, SignalRateLimitError,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 100 MB attachment ceiling.
pub const SIGNAL_MAX_ATTACHMENT_SIZE: u64 = 100 * 1024 * 1024;
/// Signal message size limit.
pub const MAX_MESSAGE_LENGTH: usize = 8000;
/// Seconds between typing-indicator refreshes.
pub const TYPING_INTERVAL: f64 = 8.0;
pub const SSE_RETRY_DELAY_INITIAL: f64 = 2.0;
pub const SSE_RETRY_DELAY_MAX: f64 = 60.0;
/// Seconds between health checks.
pub const HEALTH_CHECK_INTERVAL: f64 = 30.0;
/// Seconds without SSE activity before concern.
pub const HEALTH_CHECK_STALE_THRESHOLD: f64 = 120.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split a comma-separated string into a list, stripping whitespace and
/// dropping empties. Mirrors `_parse_comma_list`.
pub fn parse_comma_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .collect()
}

/// Guess a file extension from leading magic bytes. Mirrors `_guess_extension`.
pub fn guess_extension(data: &[u8]) -> &'static str {
    if data.len() >= 4 && &data[..4] == b"\x89PNG" {
        return ".png";
    }
    if data.len() >= 2 && &data[..2] == b"\xff\xd8" {
        return ".jpg";
    }
    if data.len() >= 4 && &data[..4] == b"GIF8" {
        return ".gif";
    }
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return ".webp";
    }
    if data.len() >= 4 && &data[..4] == b"%PDF" {
        return ".pdf";
    }
    if data.len() >= 8 && &data[4..8] == b"ftyp" {
        return ".mp4";
    }
    if data.len() >= 4 && &data[..4] == b"OggS" {
        return ".ogg";
    }
    if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
        return ".mp3";
    }
    if data.len() >= 2 && &data[..2] == b"PK" {
        return ".zip";
    }
    ".bin"
}

/// True for image extensions. Mirrors `_is_image_ext`.
pub fn is_image_ext(ext: &str) -> bool {
    matches!(
        ext.to_lowercase().as_str(),
        ".jpg" | ".jpeg" | ".png" | ".gif" | ".webp"
    )
}

/// True for audio extensions. Mirrors `_is_audio_ext`.
pub fn is_audio_ext(ext: &str) -> bool {
    matches!(
        ext.to_lowercase().as_str(),
        ".mp3" | ".wav" | ".ogg" | ".m4a" | ".aac"
    )
}

/// Map a file extension to a MIME type, defaulting to
/// `application/octet-stream`. Mirrors `_ext_to_mime` / `_EXT_TO_MIME`.
pub fn ext_to_mime(ext: &str) -> &'static str {
    match ext.to_lowercase().as_str() {
        ".jpg" | ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".ogg" => "audio/ogg",
        ".mp3" => "audio/mpeg",
        ".wav" => "audio/wav",
        ".m4a" => "audio/mp4",
        ".aac" => "audio/aac",
        ".mp4" => "video/mp4",
        ".pdf" => "application/pdf",
        ".zip" => "application/zip",
        _ => "application/octet-stream",
    }
}

/// A single Signal mention descriptor (from `dataMessage.mentions[]`).
#[derive(Debug, Clone, Default)]
pub struct Mention {
    pub start: usize,
    pub length: usize,
    pub number: Option<String>,
    pub uuid: Option<String>,
}

impl Mention {
    /// Parse a mention object from JSON, applying Python `.get` defaults
    /// (`start` -> 0, `length` -> 1).
    pub fn from_json(v: &Value) -> Mention {
        let obj = v.as_object();
        let start = obj
            .and_then(|o| o.get("start"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let length = obj
            .and_then(|o| o.get("length"))
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let number = obj
            .and_then(|o| o.get("number"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let uuid = obj
            .and_then(|o| o.get("uuid"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        Mention {
            start,
            length,
            number,
            uuid,
        }
    }

    /// The `@identifier` replacement: number or uuid or "user".
    fn identifier(&self) -> String {
        if let Some(n) = &self.number {
            return n.clone();
        }
        if let Some(u) = &self.uuid {
            return u.clone();
        }
        "user".to_string()
    }
}

/// Replace Signal mention placeholders (`￼`) with readable `@identifiers`.
///
/// Signal encodes @mentions as the Unicode object-replacement character with
/// out-of-band metadata containing the mentioned user's UUID/number. Positions
/// in `start`/`length` are measured in Python code points (Unicode scalar
/// values), so we slice on `char` boundaries to match exactly.
///
/// Mirrors `_render_mentions`.
pub fn render_mentions(text: &str, mentions: &[Mention]) -> String {
    if mentions.is_empty() || !text.contains('\u{FFFC}') {
        return text.to_string();
    }
    // Sort by start position descending so earlier replacements don't shift
    // later indices. (Python uses a stable sort; ties keep input order.)
    let mut sorted: Vec<&Mention> = mentions.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    // Operate over a Vec<char> for code-point indexing identical to Python.
    let mut chars: Vec<char> = text.chars().collect();
    for mention in sorted {
        let start = mention.start;
        let length = mention.length;
        if start > chars.len() {
            // Python slicing clamps; replicate by treating start as end.
            let repl: Vec<char> = format!("@{}", mention.identifier()).chars().collect();
            chars.extend(repl);
            continue;
        }
        let end = (start + length).min(chars.len());
        let repl: Vec<char> = format!("@{}", mention.identifier()).chars().collect();
        let mut rebuilt: Vec<char> = Vec::with_capacity(chars.len());
        rebuilt.extend_from_slice(&chars[..start]);
        rebuilt.extend_from_slice(&repl);
        rebuilt.extend_from_slice(&chars[end..]);
        chars = rebuilt;
    }
    chars.into_iter().collect()
}

/// Return True if *value* already looks like a Signal service identifier.
///
/// Mirrors `_is_signal_service_id`: accepts `PNI:`/`u:` prefixes or a parseable
/// UUID.
pub fn is_signal_service_id(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    if value.starts_with("PNI:") || value.starts_with("u:") {
        return true;
    }
    parse_uuid(value)
}

/// Return True for a plausible E.164 phone number. Mirrors
/// `_looks_like_e164_number`.
pub fn looks_like_e164_number(value: &str) -> bool {
    if value.is_empty() || !value.starts_with('+') {
        return false;
    }
    let digits = &value[1..];
    !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && (7..=15).contains(&digits.len())
}

/// Loose UUID validator matching Python `uuid.UUID(value)` acceptance for the
/// canonical hyphenated and 32-hex-digit forms (the shapes signal-cli emits).
fn parse_uuid(value: &str) -> bool {
    // Python's uuid.UUID accepts braces, urn:uuid:, hyphenated, and bare hex.
    let mut s = value.trim();
    if let Some(rest) = s.strip_prefix("urn:uuid:") {
        s = rest;
    }
    let s = s.trim_start_matches('{').trim_end_matches('}');
    let hex: String = s.chars().filter(|&c| c != '-').collect();
    hex.len() == 32 && hex.chars().all(|c| c.is_ascii_hexdigit())
}

/// Check if Signal is configured (has URL and account). Mirrors
/// `check_signal_requirements`.
pub fn check_signal_requirements() -> bool {
    let url = std::env::var("SIGNAL_HTTP_URL").unwrap_or_default();
    let account = std::env::var("SIGNAL_ACCOUNT").unwrap_or_default();
    !url.is_empty() && !account.is_empty()
}

// ---------------------------------------------------------------------------
// Markdown → Signal textStyles
// ---------------------------------------------------------------------------

/// Length of `s` in UTF-16 code units. Mirrors the inner `_utf16_len`.
fn utf16_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// A formatting span recorded in Python code points: (start, length, style).
type CpStyle = (usize, usize, &'static str);

/// Convert markdown to plain text + Signal `textStyles` list.
///
/// Signal doesn't render markdown; it uses `bodyRanges` (exposed by signal-cli
/// as `textStyle` / `textStyles` params) of the form `start:length:STYLE`,
/// where positions are **UTF-16 code units**.
///
/// Supported styles: BOLD, ITALIC, STRIKETHROUGH, MONOSPACE.
///
/// Returns `(plain_text, styles_list)`. Faithful port of `_markdown_to_signal`,
/// operating on `Vec<char>` so all offsets are Python-code-point offsets until
/// the final UTF-16 conversion.
pub fn markdown_to_signal(text: &str) -> (String, Vec<String>) {
    // Pre-process: collapse 3+ newlines to 2, then strip.
    let collapsed = collapse_blank_lines(text);
    let mut chars: Vec<char> = collapsed.trim().chars().collect();

    let mut styles: Vec<CpStyle> = Vec::new();

    // --- Phase 1: fenced code blocks ```...``` → MONOSPACE ---
    loop {
        match find_code_block(&chars) {
            Some((mstart, mend, inner_start, inner_end)) => {
                // inner = group(1).rstrip('\n')
                let mut ie = inner_end;
                while ie > inner_start && chars[ie - 1] == '\n' {
                    ie -= 1;
                }
                let inner: Vec<char> = chars[inner_start..ie].to_vec();
                let inner_len = inner.len();
                let mut rebuilt: Vec<char> = Vec::new();
                rebuilt.extend_from_slice(&chars[..mstart]);
                rebuilt.extend_from_slice(&inner);
                rebuilt.extend_from_slice(&chars[mend..]);
                chars = rebuilt;
                styles.push((mstart, inner_len, "MONOSPACE"));
            }
            None => break,
        }
    }

    // --- Phase 2: heading markers  # Foo → Foo (BOLD) ---
    {
        let headings = find_headings(&chars);
        let mut new_text: Vec<char> = Vec::new();
        let mut last_end = 0usize;
        for (mstart, mend) in headings {
            new_text.extend_from_slice(&chars[last_end..mstart]);
            // eol = find '\n' from mend, else len
            let mut eol = mend;
            while eol < chars.len() && chars[eol] != '\n' {
                eol += 1;
            }
            let heading_text: Vec<char> = chars[mend..eol].to_vec();
            let start = new_text.len();
            let hlen = heading_text.len();
            new_text.extend_from_slice(&heading_text);
            styles.push((start, hlen, "BOLD"));
            last_end = eol;
        }
        new_text.extend_from_slice(&chars[last_end..]);
        chars = new_text;
    }

    // --- Phase 3: inline patterns (single pass) ---
    let all_matches = collect_inline_matches(&chars);

    // Build removal list to adjust Phase 1/2 styles.
    let mut removals: Vec<(usize, usize)> = Vec::new();
    for m in &all_matches {
        if m.g1_start > m.start {
            removals.push((m.start, m.g1_start - m.start));
        }
        if m.end > m.g1_end {
            removals.push((m.g1_end, m.end - m.g1_end));
        }
    }
    removals.sort();

    let adj = |pos: usize| -> usize {
        let mut shift = 0usize;
        for &(rp, rl) in &removals {
            if rp < pos {
                shift += rl.min(pos - rp);
            } else {
                break;
            }
        }
        pos - shift
    };

    let mut adjusted_prior: Vec<CpStyle> = Vec::new();
    for &(s, l, st) in &styles {
        let ns = adj(s);
        let ne = adj(s + l);
        if ne > ns {
            adjusted_prior.push((ns, ne - ns, st));
        }
    }

    // Strip inline markers in one pass.
    let mut result: Vec<char> = Vec::new();
    let mut last_end = 0usize;
    let mut inline_styles: Vec<CpStyle> = Vec::new();
    for m in &all_matches {
        result.extend_from_slice(&chars[last_end..m.start]);
        let pos = result.len();
        let inner: Vec<char> = chars[m.g1_start..m.g1_end].to_vec();
        let inner_len = inner.len();
        result.extend_from_slice(&inner);
        inline_styles.push((pos, inner_len, m.style));
        last_end = m.end;
    }
    result.extend_from_slice(&chars[last_end..]);
    chars = result;

    let mut all_styles: Vec<CpStyle> = adjusted_prior;
    all_styles.extend(inline_styles);

    // Convert code-point offsets → UTF-16 code-unit offsets.
    // Python sorts tuples (start, length, style) ascending.
    all_styles.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));

    let total_len = chars.len();
    let mut style_strings = Vec::new();
    for (cp_start, cp_len, stype) in all_styles {
        if cp_start + cp_len > total_len {
            // cp_start is usize so the < 0 branch is unreachable.
            continue;
        }
        let prefix: String = chars[..cp_start].iter().collect();
        let span: String = chars[cp_start..cp_start + cp_len].iter().collect();
        let u16_start = utf16_len(&prefix);
        let u16_len = utf16_len(&span);
        style_strings.push(format!("{}:{}:{}", u16_start, u16_len, stype));
    }

    let plain: String = chars.into_iter().collect();
    (plain, style_strings)
}

/// Collapse runs of 3+ newlines down to exactly 2. Mirrors `re.sub(r"\n{3,}", "\n\n", text)`.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0usize;
    for c in text.chars() {
        if c == '\n' {
            run += 1;
        } else {
            if run >= 3 {
                out.push('\n');
                out.push('\n');
            } else {
                for _ in 0..run {
                    out.push('\n');
                }
            }
            run = 0;
            out.push(c);
        }
    }
    if run >= 3 {
        out.push('\n');
        out.push('\n');
    } else {
        for _ in 0..run {
            out.push('\n');
        }
    }
    out
}

/// Find the first fenced code block in `chars`.
/// Returns `(match_start, match_end, inner_start, inner_end)` where inner is
/// group 1: ``` ```[lang]\n?(...)``` ``` with DOTALL semantics.
fn find_code_block(chars: &[char]) -> Option<(usize, usize, usize, usize)> {
    let n = chars.len();
    let mut i = 0;
    while i + 2 < n {
        if chars[i] == '`' && chars[i + 1] == '`' && chars[i + 2] == '`' {
            let mstart = i;
            let mut j = i + 3;
            // [a-zA-Z0-9_+-]* language tag
            while j < n && is_lang_char(chars[j]) {
                j += 1;
            }
            // optional single \n
            if j < n && chars[j] == '\n' {
                j += 1;
            }
            let inner_start = j;
            // find closing ```
            let mut k = inner_start;
            while k + 2 < n + 1 {
                if k + 2 < n + 1 && k + 2 <= n && k + 3 <= n {
                    // bounds guard below
                }
                if k + 3 <= n && chars[k] == '`' && chars[k + 1] == '`' && chars[k + 2] == '`' {
                    let inner_end = k;
                    let mend = k + 3;
                    return Some((mstart, mend, inner_start, inner_end));
                }
                k += 1;
            }
            // No closing fence: not a match, advance past this position.
        }
        i += 1;
    }
    None
}

fn is_lang_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '+' || c == '-'
}

/// Find heading markers: `^#{1,6}\s+` (MULTILINE). Returns (match_start, match_end).
fn find_headings(chars: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let n = chars.len();
    let mut i = 0;
    // line start tracking
    let mut at_line_start = true;
    while i < n {
        if at_line_start && chars[i] == '#' {
            let mstart = i;
            let mut h = i;
            let mut hashes = 0;
            while h < n && chars[h] == '#' && hashes < 6 {
                h += 1;
                hashes += 1;
            }
            // require 1..6 hashes then at least one whitespace
            if hashes >= 1 && h < n && is_re_space(chars[h]) {
                let mut w = h;
                while w < n && is_re_space(chars[w]) {
                    w += 1;
                }
                out.push((mstart, w));
                // continue scanning after the consumed whitespace
                at_line_start = chars.get(w - 1) == Some(&'\n');
                i = w;
                continue;
            }
        }
        at_line_start = chars[i] == '\n';
        i += 1;
    }
    out
}

/// `\s` for the regex engine: space, tab, newline, CR, FF, VT.
fn is_re_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c' | '\x0b')
}

#[derive(Debug, Clone)]
struct InlineMatch {
    start: usize,
    end: usize,
    g1_start: usize,
    g1_end: usize,
    style: &'static str,
}

/// Collect non-overlapping inline-markdown matches across the six patterns,
/// earlier patterns winning ties. Returns them sorted by start.
fn collect_inline_matches(chars: &[char]) -> Vec<InlineMatch> {
    let mut all: Vec<InlineMatch> = Vec::new();
    let mut occupied: Vec<(usize, usize)> = Vec::new();

    // Pattern order matches Python `_PATTERNS`.
    let patterns: &[fn(&[char]) -> Vec<InlineMatch>] = &[
        find_double_star, // **bold**
        find_double_under, // __bold__
        find_double_tilde, // ~~strike~~
        find_backtick,     // `mono`
        find_single_star,  // *italic*
        find_single_under, // _italic_
    ];

    for pat in patterns {
        for m in pat(chars) {
            let (ms, me) = (m.start, m.end);
            let overlaps = occupied.iter().any(|&(os, oe)| ms < oe && me > os);
            if !overlaps {
                occupied.push((ms, me));
                all.push(m);
            }
        }
    }
    all.sort_by(|a, b| a.start.cmp(&b.start));
    all
}

// Each finder implements a non-greedy `(.+?)` between delimiters with the
// lookaround constraints of the original regexes.

fn find_double_star(chars: &[char]) -> Vec<InlineMatch> {
    find_paired(chars, &['*', '*'], &['*', '*'], "BOLD", true)
}
fn find_double_under(chars: &[char]) -> Vec<InlineMatch> {
    find_paired(chars, &['_', '_'], &['_', '_'], "BOLD", true)
}
fn find_double_tilde(chars: &[char]) -> Vec<InlineMatch> {
    find_paired(chars, &['~', '~'], &['~', '~'], "STRIKETHROUGH", true)
}
fn find_backtick(chars: &[char]) -> Vec<InlineMatch> {
    // `(.+?)` — single backtick, no DOTALL (newline not allowed).
    find_paired(chars, &['`'], &['`'], "MONOSPACE", false)
}

/// Generic paired-delimiter, non-greedy single match scanner.
/// `dotall` controls whether the inner group may span newlines.
fn find_paired(
    chars: &[char],
    open: &[char],
    close: &[char],
    style: &'static str,
    dotall: bool,
) -> Vec<InlineMatch> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i + open.len() <= n {
        if slice_eq(chars, i, open) {
            let g1s = i + open.len();
            // (.+?) needs at least one char
            let mut j = g1s;
            let mut found = None;
            while j < n {
                let c = chars[j];
                if !dotall && c == '\n' {
                    break;
                }
                if j + close.len() <= n && slice_eq(chars, j, close) && j > g1s {
                    found = Some((j, j + close.len()));
                    break;
                }
                j += 1;
            }
            // Special: when open==close==single, the (.+?) greedy-min still
            // requires the closing not at g1s; handled by `j > g1s`.
            if let Some((g1e, end)) = found {
                out.push(InlineMatch {
                    start: i,
                    end,
                    g1_start: g1s,
                    g1_end: g1e,
                    style,
                });
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// `(?<!\*)\*(?!\*| )(.+?)(?<!\*)\*(?!\*)` — single-star italic.
fn find_single_star(chars: &[char]) -> Vec<InlineMatch> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i] == '*' {
            // (?<!\*): preceding char not '*'
            let prev_star = i > 0 && chars[i - 1] == '*';
            // (?!\*| ): next char not '*' and not ' '
            let next = chars.get(i + 1).copied();
            let next_bad = matches!(next, Some('*') | Some(' '));
            if !prev_star && !next_bad {
                let g1s = i + 1;
                let mut j = g1s;
                let mut found = None;
                while j < n {
                    if chars[j] == '*' && j > g1s {
                        // (?<!\*) before closing star
                        let before = chars[j - 1] != '*';
                        // (?!\*) after closing star
                        let after_ok = chars.get(j + 1) != Some(&'*');
                        if before && after_ok {
                            found = Some((j, j + 1));
                            break;
                        }
                    }
                    j += 1;
                }
                if let Some((g1e, end)) = found {
                    out.push(InlineMatch {
                        start: i,
                        end,
                        g1_start: g1s,
                        g1_end: g1e,
                        style: "ITALIC",
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// `(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)` — single-underscore italic.
fn find_single_under(chars: &[char]) -> Vec<InlineMatch> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i] == '_' {
            let prev_word = i > 0 && is_word_char(chars[i - 1]);
            let next_under = chars.get(i + 1) == Some(&'_');
            if !prev_word && !next_under {
                let g1s = i + 1;
                let mut j = g1s;
                let mut found = None;
                while j < n {
                    if chars[j] == '_' && j > g1s {
                        let before = chars[j - 1] != '_';
                        let after_ok = chars.get(j + 1).map_or(true, |&c| !is_word_char(c));
                        if before && after_ok {
                            found = Some((j, j + 1));
                            break;
                        }
                    }
                    j += 1;
                }
                if let Some((g1e, end)) = found {
                    out.push(InlineMatch {
                        start: i,
                        end,
                        g1_start: g1s,
                        g1_end: g1e,
                        style: "ITALIC",
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// Python regex `\w`: alphanumerics + underscore (plus Unicode word chars).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn slice_eq(chars: &[char], at: usize, pat: &[char]) -> bool {
    if at + pat.len() > chars.len() {
        return false;
    }
    chars[at..at + pat.len()] == *pat
}

// ---------------------------------------------------------------------------
// Recipient cache + identifier extraction
// ---------------------------------------------------------------------------

/// Best-effort number↔UUID mapping observed from Signal envelopes / contacts.
///
/// Mirrors the adapter fields `_recipient_uuid_by_number` /
/// `_recipient_number_by_uuid`.
#[derive(Debug, Clone, Default)]
pub struct RecipientCache {
    pub uuid_by_number: HashMap<String, String>,
    pub number_by_uuid: HashMap<String, String>,
}

impl RecipientCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache any number↔UUID mapping. Mirrors `_remember_recipient_identifiers`.
    pub fn remember(&mut self, number: Option<&str>, service_id: Option<&str>) {
        let (number, service_id) = match (number, service_id) {
            (Some(n), Some(s)) if !n.is_empty() && !s.is_empty() => (n, s),
            _ => return,
        };
        if !is_signal_service_id(service_id) {
            return;
        }
        self.uuid_by_number
            .insert(number.to_string(), service_id.to_string());
        self.number_by_uuid
            .insert(service_id.to_string(), number.to_string());
    }

    /// Resolve the preferred Signal recipient identifier for a direct chat,
    /// using only cached mappings. Mirrors the cache hit-path of
    /// `_resolve_recipient`; the RPC fallback lives in [`SignalRpc::resolve_recipient`].
    pub fn resolve(&self, chat_id: &str) -> String {
        if chat_id.is_empty()
            || chat_id.starts_with("group:")
            || is_signal_service_id(chat_id)
            || !looks_like_e164_number(chat_id)
        {
            return chat_id.to_string();
        }
        self.uuid_by_number
            .get(chat_id)
            .cloned()
            .unwrap_or_else(|| chat_id.to_string())
    }
}

/// Best-effort extraction of a Signal service ID from a `listContacts` entry.
///
/// Mirrors `_extract_contact_uuid`.
pub fn extract_contact_uuid(contact: &Value, phone_number: &str) -> Option<String> {
    let obj = contact.as_object()?;
    let number = obj.get("number").and_then(Value::as_str);
    let recipient = obj.get("recipient").and_then(Value::as_str);

    let mut service_id = obj
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| obj.get("serviceId").and_then(Value::as_str))
        .map(|s| s.to_string());

    if service_id.is_none() {
        if let Some(profile) = obj.get("profile").and_then(Value::as_object) {
            service_id = profile
                .get("serviceId")
                .and_then(Value::as_str)
                .or_else(|| profile.get("uuid").and_then(Value::as_str))
                .map(|s| s.to_string());
        }
    }

    if let Some(sid) = service_id {
        if is_signal_service_id(&sid) {
            let matches_number = number == Some(phone_number) || recipient == Some(phone_number);
            if matches_number {
                return Some(sid);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Adapter configuration (mirrors SignalAdapter.__init__ env/extra parsing)
// ---------------------------------------------------------------------------

/// Settings derived from config `extra` + env vars at construction.
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// signal-cli HTTP base, trailing slashes stripped.
    pub http_url: String,
    pub account: String,
    pub ignore_stories: bool,
    /// Group allowlist (`SIGNAL_GROUP_ALLOWED_USERS`); empty means groups disabled.
    pub group_allow_from: HashSet<String>,
    /// DM allowlist (`SIGNAL_ALLOWED_USERS`, default `*`).
    pub dm_allow_from: HashSet<String>,
    /// Normalized account for self-message filtering.
    pub account_normalized: String,
}

impl SignalConfig {
    /// Build from an `extra` map and the environment, mirroring `__init__`.
    pub fn from_extra(extra: &HashMap<String, String>) -> Self {
        let http_url = extra
            .get("http_url")
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string())
            .trim_end_matches('/')
            .to_string();
        let account = extra.get("account").cloned().unwrap_or_default();
        let ignore_stories = match extra.get("ignore_stories") {
            Some(v) => parse_extra_bool(v, true),
            None => true,
        };

        let group_allowed = std::env::var("SIGNAL_GROUP_ALLOWED_USERS").unwrap_or_default();
        let group_allow_from: HashSet<String> =
            parse_comma_list(&group_allowed).into_iter().collect();

        let dm_allowed =
            std::env::var("SIGNAL_ALLOWED_USERS").unwrap_or_else(|_| "*".to_string());
        let dm_allow_from: HashSet<String> = parse_comma_list(&dm_allowed).into_iter().collect();

        let account_normalized = account.trim().to_string();

        SignalConfig {
            http_url,
            account,
            ignore_stories,
            group_allow_from,
            dm_allow_from,
            account_normalized,
        }
    }

    /// Are URL + account both set? (Python `connect` precondition.)
    pub fn is_configured(&self) -> bool {
        !self.http_url.is_empty() && !self.account.is_empty()
    }
}

fn parse_extra_bool(v: &str, default: bool) -> bool {
    match v.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        "" => default,
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// Typing-indicator backoff state machine (mirrors send_typing's bookkeeping)
// ---------------------------------------------------------------------------

/// Per-chat typing-indicator backoff state. Tracks consecutive failures and a
/// cooldown deadline so we stop hammering signal-cli for unreachable recipients.
///
/// Mirrors `_typing_failures` / `_typing_skip_until`. Time is expressed as a
/// monotonic seconds value the caller supplies (`time.monotonic()` equivalent).
#[derive(Debug, Clone, Default)]
pub struct TypingBackoff {
    failures: HashMap<String, u32>,
    skip_until: HashMap<String, f64>,
}

impl TypingBackoff {
    pub fn new() -> Self {
        Self::default()
    }

    /// Should we skip the sendTyping RPC right now? Mirrors the early-return at
    /// the top of `send_typing`.
    pub fn should_skip(&self, chat_id: &str, now: f64) -> bool {
        now < self.skip_until.get(chat_id).copied().unwrap_or(0.0)
    }

    /// Current consecutive-failure count (used to gate `log_failures`).
    pub fn failures(&self, chat_id: &str) -> u32 {
        self.failures.get(chat_id).copied().unwrap_or(0)
    }

    /// Whether the next RPC should log at WARNING (`fails == 0`).
    pub fn log_failures(&self, chat_id: &str) -> bool {
        self.failures(chat_id) == 0
    }

    /// Record a sendTyping failure. After 3 consecutive failures, set an
    /// exponential cooldown (16s, 32s, 60s cap). Mirrors the `result is None`
    /// branch of `send_typing`.
    pub fn on_failure(&mut self, chat_id: &str, now: f64) {
        let fails = self.failures(chat_id) + 1;
        self.failures.insert(chat_id.to_string(), fails);
        if fails >= 3 {
            let backoff = f64::min(60.0, 16.0 * 2f64.powi((fails - 3) as i32));
            self.skip_until.insert(chat_id.to_string(), now + backoff);
        }
    }

    /// Record a sendTyping success: clears the counters. Mirrors the `else`
    /// branch of `send_typing`.
    pub fn on_success(&mut self, chat_id: &str) {
        self.failures.remove(chat_id);
        self.skip_until.remove(chat_id);
    }

    /// Reset per-chat backoff (called from `_stop_typing_indicator`).
    pub fn reset(&mut self, chat_id: &str) {
        self.failures.remove(chat_id);
        self.skip_until.remove(chat_id);
    }
}

// ---------------------------------------------------------------------------
// Recent-sent-timestamp tracker (echo-back filtering)
// ---------------------------------------------------------------------------

/// Tracks recently sent message timestamps to suppress Note-to-Self echoes.
///
/// Mirrors `_recent_sent_timestamps` (a `set`, capped at 50). Python's
/// `set.pop()` removes an arbitrary element; we use insertion order as a
/// deterministic stand-in (the bound is the only behaviour callers rely on).
#[derive(Debug, Clone, Default)]
pub struct RecentSentTimestamps {
    set: HashSet<i64>,
    order: Vec<i64>,
    max: usize,
}

impl RecentSentTimestamps {
    pub fn new() -> Self {
        RecentSentTimestamps {
            set: HashSet::new(),
            order: Vec::new(),
            max: 50,
        }
    }

    /// Record an outbound timestamp from an RPC `send` result.
    /// Mirrors `_track_sent_timestamp`.
    pub fn track(&mut self, rpc_result: &Value) {
        if let Some(ts) = rpc_result
            .as_object()
            .and_then(|o| o.get("timestamp"))
            .and_then(Value::as_i64)
        {
            self.add(ts);
        }
    }

    /// Add a timestamp directly, enforcing the cap.
    pub fn add(&mut self, ts: i64) {
        if self.set.insert(ts) {
            self.order.push(ts);
            if self.order.len() > self.max {
                let removed = self.order.remove(0);
                self.set.remove(&removed);
            }
        }
    }

    /// True if `ts` is currently tracked.
    pub fn contains(&self, ts: i64) -> bool {
        self.set.contains(&ts)
    }

    /// Remove a timestamp (echo consumed). Mirrors `set.discard`.
    pub fn discard(&mut self, ts: i64) {
        if self.set.remove(&ts) {
            self.order.retain(|&t| t != ts);
        }
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC payload construction + dispatch
// ---------------------------------------------------------------------------

/// Build the JSON-RPC 2.0 request body. Mirrors the `payload` dict in `_rpc`.
pub fn build_rpc_payload(method: &str, params: &Value, rpc_id: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": rpc_id,
    })
}

/// Default RPC id when none supplied: `f"{method}_{int(time.time()*1000)}"`.
/// `now_ms` is the caller's current epoch-milliseconds.
pub fn default_rpc_id(method: &str, now_ms: i64) -> String {
    format!("{}_{}", method, now_ms)
}

/// Classification of a parsed JSON-RPC response.
#[derive(Debug, Clone)]
pub enum RpcOutcome {
    /// `data["result"]` (may be `Value::Null` if the key was absent).
    Result(Value),
    /// `data["error"]` was present (and not raised as a rate-limit error).
    Error(Value),
    /// A rate-limit error that the caller opted to surface.
    RateLimit(SignalRateLimitError),
}

/// Parse a decoded JSON-RPC response body, mirroring the post-`resp.json()`
/// logic of `_rpc`.
///
/// When `raise_on_rate_limit` is true and `error` is a Signal rate-limit error,
/// returns [`RpcOutcome::RateLimit`]. Otherwise an `error` yields
/// [`RpcOutcome::Error`]; absence of `error` yields
/// [`RpcOutcome::Result`] with `data.get("result")` (Null if missing).
pub fn parse_rpc_response(data: &Value, raise_on_rate_limit: bool) -> RpcOutcome {
    if let Some(err) = data.get("error") {
        if raise_on_rate_limit && is_signal_rate_limit_error(err) {
            let err_msg = match err.as_object().and_then(|o| o.get("message")) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) => "None".to_string(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            let retry_after = extract_retry_after_seconds(err);
            return RpcOutcome::RateLimit(SignalRateLimitError::new(err_msg, retry_after));
        }
        return RpcOutcome::Error(err.clone());
    }
    RpcOutcome::Result(data.get("result").cloned().unwrap_or(Value::Null))
}

/// Recipient routing for an outbound action: either a `groupId` or a resolved
/// `recipient` list. `recipient_resolver` mirrors `_resolve_recipient`.
fn apply_routing(params: &mut Map<String, Value>, chat_id: &str, recipient: &str) {
    if let Some(group) = chat_id.strip_prefix("group:") {
        params.insert("groupId".to_string(), Value::String(group.to_string()));
    } else {
        params.insert(
            "recipient".to_string(),
            Value::Array(vec![Value::String(recipient.to_string())]),
        );
    }
}

/// Build the `send` params for a text message. `recipient` is the
/// already-resolved identifier (the caller resolves via [`RecipientCache`] /
/// RPC). Mirrors the params built in `send`.
pub fn build_send_text_params(
    account: &str,
    chat_id: &str,
    recipient: &str,
    content: &str,
) -> Value {
    let (plain_text, text_styles) = markdown_to_signal(content);
    let mut params = Map::new();
    params.insert("account".to_string(), Value::String(account.to_string()));
    params.insert("message".to_string(), Value::String(plain_text));

    if !text_styles.is_empty() {
        if text_styles.len() == 1 {
            params.insert("textStyle".to_string(), Value::String(text_styles[0].clone()));
        } else {
            params.insert(
                "textStyles".to_string(),
                Value::Array(text_styles.into_iter().map(Value::String).collect()),
            );
        }
    }

    apply_routing(&mut params, chat_id, recipient);
    Value::Object(params)
}

/// Build the `sendTyping` params. Mirrors `send_typing`.
pub fn build_typing_params(account: &str, chat_id: &str, recipient: &str) -> Value {
    let mut params = Map::new();
    params.insert("account".to_string(), Value::String(account.to_string()));
    apply_routing(&mut params, chat_id, recipient);
    Value::Object(params)
}

/// Build the `send` params for an attachment send (single file + optional
/// caption). Mirrors `_send_attachment` / `send_image`.
pub fn build_send_attachment_params(
    account: &str,
    chat_id: &str,
    recipient: &str,
    file_path: &str,
    caption: Option<&str>,
) -> Value {
    let mut params = Map::new();
    params.insert("account".to_string(), Value::String(account.to_string()));
    params.insert(
        "message".to_string(),
        Value::String(caption.unwrap_or("").to_string()),
    );
    params.insert(
        "attachments".to_string(),
        Value::Array(vec![Value::String(file_path.to_string())]),
    );
    apply_routing(&mut params, chat_id, recipient);
    Value::Object(params)
}

/// Build the base params for a multi-image batch send (empty body), to which
/// each chunk's `attachments` array is added. Mirrors `base_params` in
/// `send_multiple_images`.
pub fn build_multi_image_base_params(account: &str, chat_id: &str, recipient: &str) -> Value {
    let mut params = Map::new();
    params.insert("account".to_string(), Value::String(account.to_string()));
    params.insert("message".to_string(), Value::String(String::new()));
    apply_routing(&mut params, chat_id, recipient);
    Value::Object(params)
}

/// Build `sendReaction` params. `remove` toggles the removal variant (empty
/// emoji + `remove: true`). Mirrors `send_reaction` / `remove_reaction`.
pub fn build_reaction_params(
    account: &str,
    chat_id: &str,
    emoji: &str,
    target_author: &str,
    target_timestamp: i64,
    remove: bool,
) -> Value {
    let mut params = Map::new();
    params.insert("account".to_string(), Value::String(account.to_string()));
    params.insert("emoji".to_string(), Value::String(emoji.to_string()));
    params.insert(
        "targetAuthor".to_string(),
        Value::String(target_author.to_string()),
    );
    params.insert(
        "targetTimestamp".to_string(),
        Value::Number(target_timestamp.into()),
    );
    if remove {
        params.insert("remove".to_string(), Value::Bool(true));
    }
    // Reactions route directly (no recipient resolution) — recipient == chat_id.
    apply_routing(&mut params, chat_id, chat_id);
    Value::Object(params)
}

/// Chunk a slice of attachment paths into per-message batches.
/// Mirrors the `att_batches` comprehension in `send_multiple_images`.
pub fn chunk_attachments(attachments: &[String], per_msg: usize) -> Vec<Vec<String>> {
    if per_msg == 0 {
        return vec![attachments.to_vec()];
    }
    attachments.chunks(per_msg).map(|c| c.to_vec()).collect()
}

// ---------------------------------------------------------------------------
// Envelope parsing → MessageEvent
// ---------------------------------------------------------------------------

/// Outcome of attempting to parse an inbound envelope.
#[derive(Debug, Clone)]
pub enum EnvelopeOutcome {
    /// Envelope produced a dispatchable message.
    Message(Box<MessageEvent>),
    /// Envelope was intentionally ignored (filtering, contentless, etc.).
    Ignored,
    /// A self-echo of our own outbound reply was detected; the matching
    /// timestamp should be discarded from the recent-sent set.
    EchoConsumed(i64),
}

/// Side effect: a number↔service-id pair to remember from this envelope.
#[derive(Debug, Clone, Default)]
pub struct EnvelopeSideEffects {
    pub remember: Option<(String, String)>,
    /// Attachment descriptors worth fetching (id, size, content_type).
    pub attachments: Vec<AttachmentRef>,
}

/// A pending attachment reference extracted from a dataMessage.
#[derive(Debug, Clone)]
pub struct AttachmentRef {
    pub id: String,
    pub size: u64,
    pub content_type: Option<String>,
}

/// Parsed-but-not-yet-fetched envelope: everything decided synchronously, with
/// attachment fetching left to the caller (it needs network + caching).
#[derive(Debug, Clone)]
pub struct ParsedEnvelope {
    pub sender: String,
    pub sender_name: String,
    pub sender_uuid: String,
    pub is_note_to_self: bool,
    pub is_group: bool,
    pub group_id: Option<String>,
    pub group_name: Option<String>,
    pub chat_id: String,
    pub chat_type: String,
    pub text: String,
    pub reply_to_id: Option<String>,
    pub reply_to_text: Option<String>,
    pub attachments: Vec<AttachmentRef>,
    pub timestamp_ms: i64,
}

/// Result of the synchronous envelope pre-parse.
#[derive(Debug, Clone)]
pub enum PreParse {
    Ignored,
    EchoConsumed(i64),
    Parsed(Box<ParsedEnvelope>),
}

/// Synchronously parse an inbound signal-cli envelope as far as possible
/// without network calls (attachment fetching is deferred to the caller).
///
/// Faithful port of the decision logic in `_handle_envelope`. `account_normalized`,
/// `ignore_stories`, `ignore_attachments`, and `group_allow_from` mirror the
/// adapter's config; `recent_contains` reports whether a timestamp is in the
/// recent-sent set (for echo suppression).
///
/// The caller is responsible for:
///   - calling [`RecipientCache::remember`] with `(sender, sender_uuid)` (the
///     Python `_remember_recipient_identifiers` runs unconditionally before the
///     no-sender guard),
///   - fetching the listed attachments and building the final `MessageEvent`
///     (or using [`finish_envelope`] once attachments resolve).
pub fn pre_parse_envelope(
    envelope: &Value,
    account_normalized: &str,
    ignore_stories: bool,
    ignore_attachments: bool,
    group_allow_from: &HashSet<String>,
    recent_contains: impl Fn(i64) -> bool,
) -> (PreParse, EnvelopeSideEffects) {
    let mut side = EnvelopeSideEffects::default();

    // Unwrap nested envelope if present.
    let mut envelope_data: Value = envelope
        .get("envelope")
        .cloned()
        .unwrap_or_else(|| envelope.clone());

    // syncMessage handling -> Note to Self promotion.
    let mut is_note_to_self = false;
    if envelope_data.get("syncMessage").is_some() {
        let sync_msg = envelope_data.get("syncMessage").cloned();
        if let Some(sync_msg) = sync_msg.as_ref().and_then(Value::as_object) {
            if let Some(sent_msg) = sync_msg.get("sentMessage").and_then(Value::as_object) {
                let dest = sent_msg
                    .get("destinationNumber")
                    .and_then(Value::as_str)
                    .or_else(|| sent_msg.get("destination").and_then(Value::as_str));
                let sent_ts = sent_msg.get("timestamp").and_then(Value::as_i64);
                if dest == Some(account_normalized) {
                    if let Some(ts) = sent_ts {
                        if recent_contains(ts) {
                            return (PreParse::EchoConsumed(ts), side);
                        }
                    }
                    is_note_to_self = true;
                    // Promote sentMessage to dataMessage.
                    if let Some(obj) = envelope_data.as_object_mut() {
                        obj.insert(
                            "dataMessage".to_string(),
                            Value::Object(sent_msg.clone()),
                        );
                    }
                }
            }
        }
        if !is_note_to_self {
            return (PreParse::Ignored, side);
        }
    }

    // Sender extraction.
    let sender = envelope_data
        .get("sourceNumber")
        .and_then(Value::as_str)
        .or_else(|| envelope_data.get("sourceUuid").and_then(Value::as_str))
        .or_else(|| envelope_data.get("source").and_then(Value::as_str))
        .map(|s| s.to_string());
    let sender_name = envelope_data
        .get("sourceName")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let sender_uuid = envelope_data
        .get("sourceUuid")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // _remember_recipient_identifiers runs before the no-sender guard.
    if let Some(s) = sender.as_deref() {
        if !sender_uuid.is_empty() && is_signal_service_id(&sender_uuid) {
            side.remember = Some((s.to_string(), sender_uuid.clone()));
        }
    }

    let sender = match sender {
        Some(s) if !s.is_empty() => s,
        _ => return (PreParse::Ignored, side),
    };

    // Self-message filtering (allow Note to Self).
    if !account_normalized.is_empty() && sender == account_normalized && !is_note_to_self {
        return (PreParse::Ignored, side);
    }

    // Filter stories.
    if ignore_stories && envelope_data.get("storyMessage").is_some() {
        // Python: `if self.ignore_stories and envelope_data.get("storyMessage")`
        // — a falsy storyMessage (null/empty) would not trigger. Replicate
        // truthiness.
        if json_truthy(envelope_data.get("storyMessage")) {
            return (PreParse::Ignored, side);
        }
    }

    // dataMessage or editMessage.dataMessage.
    let data_message = envelope_data
        .get("dataMessage")
        .filter(|v| json_truthy(Some(v)))
        .cloned()
        .or_else(|| {
            envelope_data
                .get("editMessage")
                .and_then(Value::as_object)
                .and_then(|e| e.get("dataMessage"))
                .filter(|v| json_truthy(Some(v)))
                .cloned()
        });
    let data_message = match data_message {
        Some(dm) => dm,
        None => return (PreParse::Ignored, side),
    };

    // Group info.
    let group_info = data_message.get("groupInfo").and_then(Value::as_object);
    let group_id = group_info
        .and_then(|g| g.get("groupId"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let is_group = group_id.is_some();

    // Group allowlist filtering.
    if is_group {
        if group_allow_from.is_empty() {
            return (PreParse::Ignored, side);
        }
        let gid = group_id.as_deref().unwrap_or("");
        if !group_allow_from.contains("*") && !group_allow_from.contains(gid) {
            return (PreParse::Ignored, side);
        }
    }

    // Chat info.
    let chat_id = if is_group {
        format!("group:{}", group_id.as_deref().unwrap_or(""))
    } else {
        sender.clone()
    };
    let chat_type = if is_group { "group" } else { "dm" }.to_string();

    // Text + mention rendering.
    let mut text = data_message
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mentions: Vec<Mention> = data_message
        .get("mentions")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().map(Mention::from_json).collect())
        .unwrap_or_default();
    if !text.is_empty() && !mentions.is_empty() {
        text = render_mentions(&text, &mentions);
    }

    // Quote (reply-to) context.
    let quote = data_message.get("quote").and_then(Value::as_object);
    let reply_to_id = quote
        .and_then(|q| q.get("id"))
        .filter(|v| json_truthy(Some(v)))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });
    let reply_to_text = quote
        .and_then(|q| q.get("text"))
        .and_then(Value::as_str)
        .map(|s| s.to_string());

    // Attachments (deferred fetch).
    let mut attachments: Vec<AttachmentRef> = Vec::new();
    if !ignore_attachments {
        if let Some(arr) = data_message.get("attachments").and_then(Value::as_array) {
            for att in arr {
                let att_obj = match att.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let att_id = att_obj.get("id").and_then(Value::as_str);
                let att_size = att_obj.get("size").and_then(Value::as_u64).unwrap_or(0);
                let att_id = match att_id {
                    Some(id) if !id.is_empty() => id.to_string(),
                    _ => continue,
                };
                if att_size > SIGNAL_MAX_ATTACHMENT_SIZE {
                    continue;
                }
                let content_type = att_obj
                    .get("contentType")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                attachments.push(AttachmentRef {
                    id: att_id,
                    size: att_size,
                    content_type,
                });
            }
        }
    }
    side.attachments = attachments.clone();

    // Timestamp.
    let timestamp_ms = envelope_data
        .get("timestamp")
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let group_name = group_info
        .and_then(|g| g.get("groupName"))
        .and_then(Value::as_str)
        .map(|s| s.to_string());

    let parsed = ParsedEnvelope {
        sender,
        sender_name,
        sender_uuid,
        is_note_to_self,
        is_group,
        group_id,
        group_name,
        chat_id,
        chat_type,
        text,
        reply_to_id,
        reply_to_text,
        attachments,
        timestamp_ms,
    };
    (PreParse::Parsed(Box::new(parsed)), side)
}

/// Finish building a [`MessageEvent`] from a [`ParsedEnvelope`] once the caller
/// has resolved attachments to `(cached_path, content_type)` pairs.
///
/// Applies the contentless-envelope skip (no text + no media) and message-type
/// derivation. Mirrors the tail of `_handle_envelope`. Returns `None` when the
/// envelope should be skipped.
pub fn finish_envelope(
    parsed: &ParsedEnvelope,
    media: Vec<(String, String)>,
) -> Option<MessageEvent> {
    let media_urls: Vec<String> = media.iter().map(|(p, _)| p.clone()).collect();
    let media_types: Vec<String> = media.iter().map(|(_, t)| t.clone()).collect();

    // Skip contentless envelopes.
    if parsed.text.trim().is_empty() && media_urls.is_empty() {
        return None;
    }

    // Session source.
    let chat_name = if parsed.is_group {
        parsed.group_name.clone()
    } else if parsed.sender_name.is_empty() {
        None
    } else {
        Some(parsed.sender_name.clone())
    };
    let user_name = if parsed.sender_name.is_empty() {
        parsed.sender.clone()
    } else {
        parsed.sender_name.clone()
    };

    let source = SessionSource {
        platform: "signal".to_string(),
        chat_id: parsed.chat_id.clone(),
        chat_name,
        chat_type: parsed.chat_type.clone(),
        user_id: Some(parsed.sender.clone()),
        user_name: Some(user_name),
        thread_id: None,
        chat_topic: None,
        user_id_alt: if parsed.sender_uuid.is_empty() {
            None
        } else {
            Some(parsed.sender_uuid.clone())
        },
        chat_id_alt: if parsed.is_group {
            parsed.group_id.clone()
        } else {
            None
        },
        is_bot: false,
        guild_id: None,
        parent_chat_id: None,
        message_id: None,
    };

    // Message type from media.
    let msg_type = if !media_types.is_empty() {
        if media_types.iter().any(|mt| mt.starts_with("audio/")) {
            MessageType::Voice
        } else if media_types.iter().any(|mt| mt.starts_with("image/")) {
            MessageType::Photo
        } else {
            MessageType::Text
        }
    } else {
        MessageType::Text
    };

    Some(MessageEvent {
        text: parsed.text.clone(),
        message_type: msg_type,
        source,
        message_id: None,
        platform_update_id: None,
        media_urls,
        media_types,
        reply_to_message_id: parsed.reply_to_id.clone(),
        reply_to_text: parsed.reply_to_text.clone(),
        auto_skill: Vec::new(),
        channel_prompt: None,
        internal: false,
    })
}

/// Python truthiness for an optional JSON value: None/null/false/0/""/[]/{} are
/// falsy.
fn json_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Reaction hooks
// ---------------------------------------------------------------------------

/// Extract `(target_author, target_timestamp)` from a raw-message-style value.
///
/// Mirrors `_extract_reaction_target`. The raw message is the dict stored in
/// `MessageEvent.raw_message`: `{"sender": ..., "timestamp_ms": ...}`. Since the
/// ported `MessageEvent` doesn't carry `raw_message`, callers pass the values
/// directly; this validates them (both must be truthy).
pub fn extract_reaction_target(sender: Option<&str>, ts_ms: i64) -> Option<(String, i64)> {
    let author = sender.filter(|s| !s.is_empty())?;
    if ts_ms == 0 {
        return None;
    }
    Some((author.to_string(), ts_ms))
}

/// Whether reactions are enabled for an event.
///
/// Two gates, mirroring `_reactions_enabled`:
/// 1. `SIGNAL_REACTIONS` env var — `false`/`0`/`no` (case-insensitive) disables.
/// 2. DM allowlist — if `dm_allow_from` lacks `*`, only senders in it pass.
pub fn reactions_enabled(sender: Option<&str>, dm_allow_from: &HashSet<String>) -> bool {
    let env = std::env::var("SIGNAL_REACTIONS").unwrap_or_else(|_| "true".to_string());
    if matches!(env.to_lowercase().as_str(), "false" | "0" | "no") {
        return false;
    }
    if let Some(s) = sender {
        if !s.is_empty()
            && !dm_allow_from.contains("*")
            && !dm_allow_from.contains(s)
        {
            return false;
        }
    }
    true
}

/// The reaction emoji to apply for a terminal processing outcome.
/// `CANCELLED` returns `None` (leave 👀 in place). Mirrors
/// `on_processing_complete`.
pub fn outcome_reaction(outcome: ProcessingOutcome) -> Option<&'static str> {
    match outcome {
        ProcessingOutcome::Success => Some("✅"),
        ProcessingOutcome::Failure => Some("❌"),
        ProcessingOutcome::Cancelled => None,
    }
}

/// The progress reaction emoji applied when processing starts. Mirrors
/// `on_processing_start`.
pub const PROGRESS_REACTION: &str = "👀";

// ---------------------------------------------------------------------------
// Chat info
// ---------------------------------------------------------------------------

/// Build the `getContact` RPC params. Mirrors `get_chat_info`'s RPC call.
pub fn build_get_contact_params(account: &str, chat_id: &str) -> Value {
    json!({
        "account": account,
        "contactAddress": chat_id,
    })
}

/// Build the chat-info dict for a chat id, given an optional resolved contact
/// name from `getContact`. Mirrors `get_chat_info`.
pub fn build_chat_info(chat_id: &str, contact_result: Option<&Value>) -> Value {
    if chat_id.starts_with("group:") {
        return json!({
            "name": chat_id,
            "type": "group",
            "chat_id": chat_id,
        });
    }
    let name = contact_result
        .and_then(Value::as_object)
        .and_then(|o| {
            o.get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    o.get("profileName")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                })
        })
        .unwrap_or(chat_id);
    json!({
        "name": name,
        "type": "dm",
        "chat_id": chat_id,
    })
}

// ---------------------------------------------------------------------------
// Blocking RPC client (request construction + response parsing)
// ---------------------------------------------------------------------------

/// Thin blocking JSON-RPC client over the signal-cli HTTP daemon.
///
/// Wraps request construction + the parsing performed in `_rpc`. The async
/// adapter's higher-level orchestration (typing tasks, SSE, scheduler waits)
/// stays at the call sites; this provides the synchronous network primitive.
pub struct SignalRpc {
    pub http_url: String,
    pub account: String,
    client: reqwest::blocking::Client,
}

impl SignalRpc {
    /// Construct an RPC client. `http_url` should already have trailing slashes
    /// stripped (see [`SignalConfig`]).
    pub fn new(http_url: impl Into<String>, account: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        SignalRpc {
            http_url: http_url.into(),
            account: account.into(),
            client,
        }
    }

    /// Health check against `/api/v1/check`. Returns the HTTP status code, or
    /// `Err` on transport failure. Mirrors the `connect` / `_health_monitor`
    /// probe.
    pub fn check(&self) -> Result<u16, String> {
        let url = format!("{}/api/v1/check", self.http_url);
        self.client
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .map(|r| r.status().as_u16())
            .map_err(|e| e.to_string())
    }

    /// Send a JSON-RPC 2.0 request and return the parsed outcome.
    ///
    /// Faithful to `_rpc`: posts to `/api/v1/rpc`, raises on HTTP status (we map
    /// that to `RpcOutcome::Error` via the error string), then classifies the
    /// JSON body. `rpc_id` defaults to `f"{method}_{now_ms}"` when `None`.
    pub fn rpc(
        &self,
        method: &str,
        params: &Value,
        rpc_id: Option<&str>,
        raise_on_rate_limit: bool,
        timeout_secs: f64,
        now_ms: i64,
    ) -> RpcOutcome {
        let id_owned;
        let id = match rpc_id {
            Some(s) => s,
            None => {
                id_owned = default_rpc_id(method, now_ms);
                &id_owned
            }
        };
        let payload = build_rpc_payload(method, params, id);
        let url = format!("{}/api/v1/rpc", self.http_url);

        let resp = self
            .client
            .post(&url)
            .timeout(Duration::from_secs_f64(timeout_secs.max(0.0)))
            .json(&payload)
            .send();

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                return RpcOutcome::Error(Value::String(e.to_string()));
            }
        };

        if let Err(e) = resp.error_for_status_ref() {
            // raise_for_status() equivalent → swallowed as failure in Python.
            return RpcOutcome::Error(Value::String(e.to_string()));
        }

        let text = match resp.text() {
            Ok(t) => t,
            Err(e) => return RpcOutcome::Error(Value::String(e.to_string())),
        };
        let data: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => return RpcOutcome::Error(Value::String(e.to_string())),
        };

        parse_rpc_response(&data, raise_on_rate_limit)
    }

    /// Resolve the preferred recipient identifier for a direct chat, consulting
    /// the cache first, then `listContacts` over RPC. Mirrors `_resolve_recipient`.
    ///
    /// `cache` is read and updated in place.
    pub fn resolve_recipient(&self, cache: &mut RecipientCache, chat_id: &str, now_ms: i64) -> String {
        if chat_id.is_empty()
            || chat_id.starts_with("group:")
            || is_signal_service_id(chat_id)
            || !looks_like_e164_number(chat_id)
        {
            return chat_id.to_string();
        }
        if let Some(cached) = cache.uuid_by_number.get(chat_id) {
            return cached.clone();
        }

        let params = json!({"account": self.account, "allRecipients": true});
        if let RpcOutcome::Result(Value::Array(contacts)) =
            self.rpc("listContacts", &params, None, false, 30.0, now_ms)
        {
            for contact in &contacts {
                let number = contact.as_object().and_then(|o| o.get("number")).and_then(Value::as_str);
                let service_id = extract_contact_uuid(contact, chat_id);
                if let (Some(num), Some(sid)) = (number, service_id) {
                    cache.remember(Some(num), Some(&sid));
                }
            }
        }
        cache
            .uuid_by_number
            .get(chat_id)
            .cloned()
            .unwrap_or_else(|| chat_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// SSE line parsing (inbound stream framing)
// ---------------------------------------------------------------------------

/// One parsed SSE line outcome. Mirrors the per-line handling in `_sse_listener`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseLine {
    /// Blank line: ignored.
    Empty,
    /// Keepalive comment (starts with ':'): proves liveness, no payload.
    Keepalive,
    /// A `data:` line carrying a (trimmed) JSON payload string.
    Data(String),
    /// A `data:` line with an empty payload: ignored.
    DataEmpty,
    /// Any other non-blank line: ignored.
    Other,
}

/// Classify a single SSE line (already stripped of trailing newline, then
/// `.strip()`-ed). Mirrors the inner loop of `_sse_listener`.
pub fn parse_sse_line(line: &str) -> SseLine {
    let line = line.trim();
    if line.is_empty() {
        return SseLine::Empty;
    }
    if line.starts_with(':') {
        return SseLine::Keepalive;
    }
    if let Some(rest) = line.strip_prefix("data:") {
        let data_str = rest.trim();
        if data_str.is_empty() {
            return SseLine::DataEmpty;
        }
        return SseLine::Data(data_str.to_string());
    }
    SseLine::Other
}

/// Build the SSE events URL for an account. Mirrors `_sse_listener`'s URL.
/// The account is percent-encoded with `safe=''` (encode everything).
pub fn sse_events_url(http_url: &str, account: &str) -> String {
    format!("{}/api/v1/events?account={}", http_url, quote_all(account))
}

/// Percent-encode with no safe characters, matching Python `quote(s, safe='')`.
pub fn quote_all(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{:02X}", b));
        }
    }
    out
}

/// Decode a percent-encoded string, matching Python `unquote`. Invalid escapes
/// are left verbatim. Used for `file://` path extraction in image sends.
pub fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h = hex_val(bytes[i + 1]);
            let l = hex_val(bytes[i + 2]);
            if let (Some(h), Some(l)) = (h, l) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Extract a local file path from a `file://` URL (used by send_image /
/// send_multiple_images). Mirrors `unquote(image_url[7:])`.
pub fn file_url_path(url: &str) -> Option<String> {
    url.strip_prefix("file://").map(unquote)
}

// ---------------------------------------------------------------------------
// SSE reconnect backoff
// ---------------------------------------------------------------------------

/// Compute the next SSE reconnect backoff value: `min(backoff * 2, MAX)`.
/// Mirrors the tail of `_sse_listener`.
pub fn next_sse_backoff(backoff: f64) -> f64 {
    f64::min(backoff * 2.0, SSE_RETRY_DELAY_MAX)
}

/// Compute the jittered sleep before reconnect: `backoff + backoff*0.2*rand`,
/// where `rand` in [0,1). Caller supplies the random fraction. Mirrors the
/// `jitter` computation.
pub fn sse_reconnect_sleep(backoff: f64, rand_fraction: f64) -> f64 {
    backoff + backoff * 0.2 * rand_fraction
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_comma_list_strips_and_drops_empty() {
        assert_eq!(parse_comma_list(" a , b ,, c "), vec!["a", "b", "c"]);
        assert_eq!(parse_comma_list("").len(), 0);
        assert_eq!(parse_comma_list(" , , ").len(), 0);
    }

    #[test]
    fn guess_extension_magic_bytes() {
        assert_eq!(guess_extension(b"\x89PNG...."), ".png");
        assert_eq!(guess_extension(b"\xff\xd8\xff"), ".jpg");
        assert_eq!(guess_extension(b"GIF89a"), ".gif");
        assert_eq!(guess_extension(b"RIFF\x00\x00\x00\x00WEBPxxxx"), ".webp");
        assert_eq!(guess_extension(b"%PDF-1.4"), ".pdf");
        assert_eq!(guess_extension(b"\x00\x00\x00\x18ftypmp42"), ".mp4");
        assert_eq!(guess_extension(b"OggS...."), ".ogg");
        assert_eq!(guess_extension(b"\xff\xfb\x90"), ".mp3");
        assert_eq!(guess_extension(b"PK\x03\x04"), ".zip");
        assert_eq!(guess_extension(b"random"), ".bin");
        assert_eq!(guess_extension(b""), ".bin");
    }

    #[test]
    fn ext_classification() {
        assert!(is_image_ext(".JPG"));
        assert!(is_image_ext(".webp"));
        assert!(!is_image_ext(".mp3"));
        assert!(is_audio_ext(".OGG"));
        assert!(!is_audio_ext(".png"));
        assert_eq!(ext_to_mime(".jpg"), "image/jpeg");
        assert_eq!(ext_to_mime(".unknown"), "application/octet-stream");
        assert_eq!(ext_to_mime(".PDF"), "application/pdf");
    }

    #[test]
    fn service_id_and_e164() {
        assert!(is_signal_service_id("PNI:abc"));
        assert!(is_signal_service_id("u:abc"));
        assert!(is_signal_service_id("12345678-1234-1234-1234-123456789012"));
        assert!(!is_signal_service_id(""));
        assert!(!is_signal_service_id("+15551234567"));

        assert!(looks_like_e164_number("+15551234567"));
        assert!(!looks_like_e164_number("15551234567")); // no +
        assert!(!looks_like_e164_number("+12")); // too short
        assert!(!looks_like_e164_number("+abc"));
        assert!(!looks_like_e164_number(""));
    }

    #[test]
    fn render_mentions_replaces_placeholder() {
        let mentions = vec![Mention {
            start: 6,
            length: 1,
            number: Some("+1555".to_string()),
            uuid: None,
        }];
        let text = "Hello \u{FFFC}!";
        assert_eq!(render_mentions(text, &mentions), "Hello @+1555!");
    }

    #[test]
    fn render_mentions_multiple_reverse_order() {
        // Two placeholders; replace from end to start so indices stay valid.
        let text = "\u{FFFC} and \u{FFFC}";
        let mentions = vec![
            Mention { start: 0, length: 1, number: Some("a".into()), uuid: None },
            Mention { start: 6, length: 1, number: Some("b".into()), uuid: None },
        ];
        assert_eq!(render_mentions(text, &mentions), "@a and @b");
    }

    #[test]
    fn render_mentions_noop_without_placeholder() {
        let mentions = vec![Mention { start: 0, length: 1, number: Some("a".into()), uuid: None }];
        assert_eq!(render_mentions("no marker", &mentions), "no marker");
    }

    #[test]
    fn markdown_bold_italic_strike_mono() {
        let (text, styles) = markdown_to_signal("**bold** and *italic* and ~~strike~~ and `code`");
        assert_eq!(text, "bold and italic and strike and code");
        // bold at 0:4, italic at 9:6, strike at 20:6, mono at 31:4
        assert!(styles.contains(&"0:4:BOLD".to_string()), "{:?}", styles);
        assert!(styles.contains(&"9:6:ITALIC".to_string()), "{:?}", styles);
        assert!(styles.contains(&"20:6:STRIKETHROUGH".to_string()), "{:?}", styles);
        assert!(styles.contains(&"31:4:MONOSPACE".to_string()), "{:?}", styles);
    }

    #[test]
    fn markdown_heading_becomes_bold() {
        let (text, styles) = markdown_to_signal("# Title\nbody");
        assert_eq!(text, "Title\nbody");
        assert_eq!(styles, vec!["0:5:BOLD".to_string()]);
    }

    #[test]
    fn markdown_code_block_monospace() {
        let (text, styles) = markdown_to_signal("```rust\nlet x = 1;\n```");
        assert_eq!(text, "let x = 1;");
        assert_eq!(styles, vec!["0:10:MONOSPACE".to_string()]);
    }

    #[test]
    fn markdown_double_underscore_bold() {
        let (text, styles) = markdown_to_signal("__strong__");
        assert_eq!(text, "strong");
        assert_eq!(styles, vec!["0:6:BOLD".to_string()]);
    }

    #[test]
    fn markdown_no_formatting_returns_empty_styles() {
        let (text, styles) = markdown_to_signal("plain text");
        assert_eq!(text, "plain text");
        assert!(styles.is_empty());
    }

    #[test]
    fn markdown_utf16_offsets_for_emoji() {
        // An astral emoji counts as 2 UTF-16 code units before the bold span.
        let (text, styles) = markdown_to_signal("\u{1F600}**hi**");
        assert_eq!(text, "\u{1F600}hi");
        // prefix emoji = 2 u16 units, span "hi" = 2 units
        assert_eq!(styles, vec!["2:2:BOLD".to_string()]);
    }

    #[test]
    fn markdown_collapses_blank_lines_and_strips() {
        let (text, _styles) = markdown_to_signal("\n\n\nhello\n\n\n\nworld\n\n\n");
        assert_eq!(text, "hello\n\nworld");
    }

    #[test]
    fn markdown_italic_not_in_word_underscore() {
        // _x_ surrounded by word chars should NOT match (lookbehind \w).
        let (text, _styles) = markdown_to_signal("a_b_c");
        assert_eq!(text, "a_b_c");
    }

    #[test]
    fn recipient_cache_remember_and_resolve() {
        let mut cache = RecipientCache::new();
        cache.remember(Some("+15551234567"), Some("12345678-1234-1234-1234-123456789012"));
        assert_eq!(
            cache.resolve("+15551234567"),
            "12345678-1234-1234-1234-123456789012"
        );
        // group / already-uuid / non-e164 pass through
        assert_eq!(cache.resolve("group:abc"), "group:abc");
        assert_eq!(cache.resolve("PNI:x"), "PNI:x");
        assert_eq!(cache.resolve("notaphone"), "notaphone");
        // unknown e164 returns itself
        assert_eq!(cache.resolve("+19998887777"), "+19998887777");
    }

    #[test]
    fn recipient_cache_ignores_non_service_id() {
        let mut cache = RecipientCache::new();
        cache.remember(Some("+1555"), Some("not-a-uuid"));
        assert!(cache.uuid_by_number.is_empty());
    }

    #[test]
    fn extract_contact_uuid_matches_number() {
        let contact = json!({
            "number": "+15551234567",
            "uuid": "12345678-1234-1234-1234-123456789012"
        });
        assert_eq!(
            extract_contact_uuid(&contact, "+15551234567"),
            Some("12345678-1234-1234-1234-123456789012".to_string())
        );
        // Different number => no match
        assert_eq!(extract_contact_uuid(&contact, "+19998887777"), None);
    }

    #[test]
    fn extract_contact_uuid_from_profile() {
        let contact = json!({
            "recipient": "+15551234567",
            "profile": {"serviceId": "PNI:abc123"}
        });
        assert_eq!(
            extract_contact_uuid(&contact, "+15551234567"),
            Some("PNI:abc123".to_string())
        );
    }

    #[test]
    fn typing_backoff_state_machine() {
        let mut tb = TypingBackoff::new();
        let chat = "c1";
        assert!(!tb.should_skip(chat, 0.0));
        assert!(tb.log_failures(chat));

        tb.on_failure(chat, 100.0); // fails=1
        assert_eq!(tb.failures(chat), 1);
        assert!(!tb.log_failures(chat));
        assert!(!tb.should_skip(chat, 100.0)); // no cooldown yet

        tb.on_failure(chat, 100.0); // fails=2
        tb.on_failure(chat, 100.0); // fails=3 -> backoff 16s
        assert!(tb.should_skip(chat, 110.0));
        assert!(!tb.should_skip(chat, 117.0));

        tb.on_success(chat);
        assert_eq!(tb.failures(chat), 0);
        assert!(!tb.should_skip(chat, 0.0));
    }

    #[test]
    fn typing_backoff_cap_at_60s() {
        let mut tb = TypingBackoff::new();
        let chat = "c";
        // Drive to a high failure count; backoff caps at 60.
        for _ in 0..10 {
            tb.on_failure(chat, 0.0);
        }
        // skip_until should be exactly 60 (cap)
        assert!(tb.should_skip(chat, 59.9));
        assert!(!tb.should_skip(chat, 60.0));
    }

    #[test]
    fn recent_sent_timestamps_cap_and_discard() {
        let mut r = RecentSentTimestamps::new();
        for i in 0..60 {
            r.add(i);
        }
        // Only the last 50 retained (FIFO eviction).
        assert!(!r.contains(0));
        assert!(r.contains(59));
        r.discard(59);
        assert!(!r.contains(59));
    }

    #[test]
    fn recent_sent_track_from_result() {
        let mut r = RecentSentTimestamps::new();
        r.track(&json!({"timestamp": 12345}));
        assert!(r.contains(12345));
        r.track(&json!({"no_ts": true}));
        // unchanged
        assert!(r.contains(12345));
    }

    #[test]
    fn build_send_text_params_text_only() {
        let p = build_send_text_params("+acct", "+15551234567", "+15551234567", "hello");
        assert_eq!(p["account"], "+acct");
        assert_eq!(p["message"], "hello");
        assert_eq!(p["recipient"], json!(["+15551234567"]));
        assert!(p.get("textStyle").is_none());
        assert!(p.get("textStyles").is_none());
    }

    #[test]
    fn build_send_text_params_single_style() {
        let p = build_send_text_params("+acct", "c", "c", "**bold**");
        assert_eq!(p["message"], "bold");
        assert_eq!(p["textStyle"], "0:4:BOLD");
        assert!(p.get("textStyles").is_none());
    }

    #[test]
    fn build_send_text_params_multi_style() {
        let p = build_send_text_params("+acct", "c", "c", "**a** *b*");
        assert!(p.get("textStyles").is_some());
        assert!(p.get("textStyle").is_none());
    }

    #[test]
    fn build_send_text_params_group_routing() {
        let p = build_send_text_params("+acct", "group:GID", "ignored", "hi");
        assert_eq!(p["groupId"], "GID");
        assert!(p.get("recipient").is_none());
    }

    #[test]
    fn build_attachment_and_reaction_params() {
        let a = build_send_attachment_params("+acct", "c", "c", "/tmp/x.png", Some("cap"));
        assert_eq!(a["message"], "cap");
        assert_eq!(a["attachments"], json!(["/tmp/x.png"]));

        let react = build_reaction_params("+acct", "group:G", "👀", "+author", 999, false);
        assert_eq!(react["emoji"], "👀");
        assert_eq!(react["targetAuthor"], "+author");
        assert_eq!(react["targetTimestamp"], 999);
        assert_eq!(react["groupId"], "G");
        assert!(react.get("remove").is_none());

        let remove = build_reaction_params("+acct", "+c", "", "+author", 1, true);
        assert_eq!(remove["emoji"], "");
        assert_eq!(remove["remove"], true);
        assert_eq!(remove["recipient"], json!(["+c"]));
    }

    #[test]
    fn rpc_payload_and_id() {
        let p = build_rpc_payload("send", &json!({"a": 1}), "send_42");
        assert_eq!(p["jsonrpc"], "2.0");
        assert_eq!(p["method"], "send");
        assert_eq!(p["id"], "send_42");
        assert_eq!(default_rpc_id("send", 1000), "send_1000");
    }

    #[test]
    fn parse_rpc_result_and_error() {
        match parse_rpc_response(&json!({"result": {"timestamp": 5}}), false) {
            RpcOutcome::Result(v) => assert_eq!(v["timestamp"], 5),
            _ => panic!("expected result"),
        }
        match parse_rpc_response(&json!({"error": {"code": -1, "message": "bad"}}), false) {
            RpcOutcome::Error(e) => assert_eq!(e["code"], -1),
            _ => panic!("expected error"),
        }
        // Missing result -> Null result
        match parse_rpc_response(&json!({}), false) {
            RpcOutcome::Result(Value::Null) => {}
            _ => panic!("expected null result"),
        }
    }

    #[test]
    fn parse_rpc_rate_limit_raises_when_opted_in() {
        let err = json!({"code": -5, "message": "Retry after 4 seconds"});
        match parse_rpc_response(&json!({"error": err}), true) {
            RpcOutcome::RateLimit(e) => {
                assert_eq!(e.retry_after, Some(4.0));
                assert_eq!(e.message, "Retry after 4 seconds");
            }
            _ => panic!("expected rate-limit"),
        }
        // Without opt-in it's a plain error.
        let err2 = json!({"code": -5, "message": "x"});
        match parse_rpc_response(&json!({"error": err2}), false) {
            RpcOutcome::Error(_) => {}
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn chunk_attachments_batches() {
        let atts: Vec<String> = (0..70).map(|i| i.to_string()).collect();
        let batches = chunk_attachments(&atts, 32);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 32);
        assert_eq!(batches[2].len(), 6);
    }

    #[test]
    fn sse_line_parsing() {
        assert_eq!(parse_sse_line(""), SseLine::Empty);
        assert_eq!(parse_sse_line("   "), SseLine::Empty);
        assert_eq!(parse_sse_line(": keepalive"), SseLine::Keepalive);
        assert_eq!(parse_sse_line("data:"), SseLine::DataEmpty);
        assert_eq!(parse_sse_line("data:   "), SseLine::DataEmpty);
        assert_eq!(
            parse_sse_line("data: {\"a\":1}"),
            SseLine::Data("{\"a\":1}".to_string())
        );
        assert_eq!(parse_sse_line("event: message"), SseLine::Other);
    }

    #[test]
    fn sse_url_and_quote() {
        assert_eq!(quote_all("+1 555/abc"), "%2B1%20555%2Fabc");
        let url = sse_events_url("http://127.0.0.1:8080", "+15551234567");
        assert_eq!(
            url,
            "http://127.0.0.1:8080/api/v1/events?account=%2B15551234567"
        );
    }

    #[test]
    fn unquote_and_file_url() {
        assert_eq!(unquote("%2Ftmp%2Fa%20b.png"), "/tmp/a b.png");
        assert_eq!(unquote("plain"), "plain");
        assert_eq!(
            file_url_path("file:///tmp/a%20b.png"),
            Some("/tmp/a b.png".to_string())
        );
        assert_eq!(file_url_path("http://x"), None);
    }

    #[test]
    fn sse_backoff_progression() {
        assert_eq!(next_sse_backoff(2.0), 4.0);
        assert_eq!(next_sse_backoff(40.0), 60.0); // cap
        assert_eq!(next_sse_backoff(60.0), 60.0);
        assert_eq!(sse_reconnect_sleep(10.0, 0.0), 10.0);
        assert_eq!(sse_reconnect_sleep(10.0, 1.0), 12.0);
    }

    #[test]
    fn config_from_extra_defaults() {
        let extra = HashMap::new();
        // Ensure env doesn't leak across runs.
        unsafe {
            std::env::remove_var("SIGNAL_GROUP_ALLOWED_USERS");
            std::env::remove_var("SIGNAL_ALLOWED_USERS");
        }
        let cfg = SignalConfig::from_extra(&extra);
        assert_eq!(cfg.http_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.account, "");
        assert!(cfg.ignore_stories);
        assert!(cfg.group_allow_from.is_empty());
        // default dm allow == {"*"}
        assert!(cfg.dm_allow_from.contains("*"));
        assert!(!cfg.is_configured());
    }

    #[test]
    fn config_strips_trailing_slash_and_parses_allowlists() {
        let mut extra = HashMap::new();
        extra.insert("http_url".to_string(), "http://host:1/".to_string());
        extra.insert("account".to_string(), "+1555".to_string());
        extra.insert("ignore_stories".to_string(), "false".to_string());
        unsafe {
            std::env::set_var("SIGNAL_GROUP_ALLOWED_USERS", "g1, g2");
            std::env::set_var("SIGNAL_ALLOWED_USERS", "+1, +2");
        }
        let cfg = SignalConfig::from_extra(&extra);
        assert_eq!(cfg.http_url, "http://host:1");
        assert_eq!(cfg.account, "+1555");
        assert!(!cfg.ignore_stories);
        assert!(cfg.group_allow_from.contains("g1"));
        assert!(cfg.group_allow_from.contains("g2"));
        assert!(cfg.dm_allow_from.contains("+1"));
        assert!(!cfg.dm_allow_from.contains("*"));
        assert!(cfg.is_configured());
        unsafe {
            std::env::remove_var("SIGNAL_GROUP_ALLOWED_USERS");
            std::env::remove_var("SIGNAL_ALLOWED_USERS");
        }
    }

    #[test]
    fn reactions_enabled_gates() {
        let mut allow: HashSet<String> = HashSet::new();
        allow.insert("*".to_string());
        unsafe {
            std::env::remove_var("SIGNAL_REACTIONS");
        }
        assert!(reactions_enabled(Some("+1"), &allow));

        unsafe {
            std::env::set_var("SIGNAL_REACTIONS", "no");
        }
        assert!(!reactions_enabled(Some("+1"), &allow));
        unsafe {
            std::env::set_var("SIGNAL_REACTIONS", "true");
        }
        // Restricted allowlist blocks unknown sender.
        let mut restricted: HashSet<String> = HashSet::new();
        restricted.insert("+allowed".to_string());
        assert!(!reactions_enabled(Some("+other"), &restricted));
        assert!(reactions_enabled(Some("+allowed"), &restricted));
        unsafe {
            std::env::remove_var("SIGNAL_REACTIONS");
        }
    }

    #[test]
    fn outcome_reaction_mapping() {
        assert_eq!(outcome_reaction(ProcessingOutcome::Success), Some("✅"));
        assert_eq!(outcome_reaction(ProcessingOutcome::Failure), Some("❌"));
        assert_eq!(outcome_reaction(ProcessingOutcome::Cancelled), None);
    }

    #[test]
    fn extract_reaction_target_validation() {
        assert_eq!(
            extract_reaction_target(Some("+author"), 123),
            Some(("+author".to_string(), 123))
        );
        assert_eq!(extract_reaction_target(Some("+author"), 0), None);
        assert_eq!(extract_reaction_target(None, 123), None);
        assert_eq!(extract_reaction_target(Some(""), 123), None);
    }

    #[test]
    fn build_chat_info_group_and_dm() {
        let g = build_chat_info("group:abc", None);
        assert_eq!(g["type"], "group");
        assert_eq!(g["name"], "group:abc");

        let dm = build_chat_info("+1555", Some(&json!({"name": "Alice"})));
        assert_eq!(dm["type"], "dm");
        assert_eq!(dm["name"], "Alice");

        let dm2 = build_chat_info("+1555", Some(&json!({"profileName": "Bob"})));
        assert_eq!(dm2["name"], "Bob");

        let dm3 = build_chat_info("+1555", None);
        assert_eq!(dm3["name"], "+1555");
    }

    #[test]
    fn pre_parse_basic_dm_message() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "sourceName": "Alice",
                "timestamp": 1700000000000i64,
                "dataMessage": {"message": "hi there"}
            }
        });
        let allow = HashSet::new();
        let (res, side) = pre_parse_envelope(
            &env, "+myaccount", true, false, &allow, |_| false,
        );
        assert!(side.remember.is_none());
        match res {
            PreParse::Parsed(p) => {
                assert_eq!(p.sender, "+15551234567");
                assert_eq!(p.chat_id, "+15551234567");
                assert_eq!(p.chat_type, "dm");
                assert_eq!(p.text, "hi there");
                assert_eq!(p.timestamp_ms, 1700000000000);
                assert!(!p.is_group);
                let ev = finish_envelope(&p, vec![]).unwrap();
                assert_eq!(ev.text, "hi there");
                assert_eq!(ev.message_type, MessageType::Text);
                assert_eq!(ev.source.user_name.as_deref(), Some("Alice"));
            }
            _ => panic!("expected parsed"),
        }
    }

    #[test]
    fn pre_parse_self_message_ignored() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+myaccount",
                "dataMessage": {"message": "echo"}
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        assert!(matches!(res, PreParse::Ignored));
    }

    #[test]
    fn pre_parse_note_to_self_promoted() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+myaccount",
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+myaccount",
                        "timestamp": 555,
                        "message": "note"
                    }
                }
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        match res {
            PreParse::Parsed(p) => {
                assert!(p.is_note_to_self);
                assert_eq!(p.text, "note");
            }
            _ => panic!("expected parsed note to self"),
        }
    }

    #[test]
    fn pre_parse_note_to_self_echo_consumed() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+myaccount",
                "syncMessage": {
                    "sentMessage": {
                        "destination": "+myaccount",
                        "timestamp": 777,
                        "message": "our own reply"
                    }
                }
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(
            &env, "+myaccount", true, false, &allow, |ts| ts == 777,
        );
        assert!(matches!(res, PreParse::EchoConsumed(777)));
    }

    #[test]
    fn pre_parse_group_not_allowed() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "dataMessage": {
                    "message": "hi",
                    "groupInfo": {"groupId": "GID", "groupName": "G"}
                }
            }
        });
        let empty = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+myaccount", true, false, &empty, |_| false);
        assert!(matches!(res, PreParse::Ignored)); // no group allowlist
    }

    #[test]
    fn pre_parse_group_allowed_wildcard() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "dataMessage": {
                    "message": "hi",
                    "groupInfo": {"groupId": "GID", "groupName": "GroupName"}
                }
            }
        });
        let mut allow = HashSet::new();
        allow.insert("*".to_string());
        let (res, _) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        match res {
            PreParse::Parsed(p) => {
                assert!(p.is_group);
                assert_eq!(p.chat_id, "group:GID");
                assert_eq!(p.group_name.as_deref(), Some("GroupName"));
                let ev = finish_envelope(&p, vec![]).unwrap();
                assert_eq!(ev.source.chat_name.as_deref(), Some("GroupName"));
                assert_eq!(ev.source.chat_id_alt.as_deref(), Some("GID"));
            }
            _ => panic!("expected parsed group"),
        }
    }

    #[test]
    fn pre_parse_contentless_skipped_in_finish() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "dataMessage": {"message": "   "}
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        match res {
            PreParse::Parsed(p) => {
                assert!(finish_envelope(&p, vec![]).is_none());
            }
            _ => panic!("expected parsed"),
        }
    }

    #[test]
    fn pre_parse_remembers_identifiers() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "sourceUuid": "12345678-1234-1234-1234-123456789012",
                "dataMessage": {"message": "hi"}
            }
        });
        let allow = HashSet::new();
        let (_, side) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        assert_eq!(
            side.remember,
            Some((
                "+15551234567".to_string(),
                "12345678-1234-1234-1234-123456789012".to_string()
            ))
        );
    }

    #[test]
    fn pre_parse_attachments_extracted_and_typed() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "dataMessage": {
                    "message": "",
                    "attachments": [
                        {"id": "att1", "size": 100, "contentType": "image/png"},
                        {"id": "", "size": 1},
                        {"id": "toobig", "size": SIGNAL_MAX_ATTACHMENT_SIZE + 1},
                        {"id": "att2", "size": 5}
                    ]
                }
            }
        });
        let allow = HashSet::new();
        let (res, side) = pre_parse_envelope(&env, "+myaccount", true, false, &allow, |_| false);
        assert_eq!(side.attachments.len(), 2);
        match res {
            PreParse::Parsed(p) => {
                assert_eq!(p.attachments[0].id, "att1");
                assert_eq!(p.attachments[0].content_type.as_deref(), Some("image/png"));
                assert_eq!(p.attachments[1].id, "att2");
                // finish with resolved media -> PHOTO (image content type)
                let ev = finish_envelope(
                    &p,
                    vec![("/tmp/x.png".to_string(), "image/png".to_string())],
                )
                .unwrap();
                assert_eq!(ev.message_type, MessageType::Photo);
                assert_eq!(ev.media_urls, vec!["/tmp/x.png".to_string()]);
            }
            _ => panic!("expected parsed"),
        }
    }

    #[test]
    fn pre_parse_audio_attachment_is_voice() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+1",
                "dataMessage": {"message": "", "attachments": [{"id": "a", "size": 1}]}
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+acct", true, false, &allow, |_| false);
        if let PreParse::Parsed(p) = res {
            let ev = finish_envelope(
                &p,
                vec![("/tmp/v.ogg".to_string(), "audio/ogg".to_string())],
            )
            .unwrap();
            assert_eq!(ev.message_type, MessageType::Voice);
        } else {
            panic!("expected parsed");
        }
    }

    #[test]
    fn pre_parse_reply_quote() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+1",
                "dataMessage": {
                    "message": "reply",
                    "quote": {"id": 999, "text": "original"}
                }
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+acct", true, false, &allow, |_| false);
        if let PreParse::Parsed(p) = res {
            assert_eq!(p.reply_to_id.as_deref(), Some("999"));
            assert_eq!(p.reply_to_text.as_deref(), Some("original"));
        } else {
            panic!("expected parsed");
        }
    }

    #[test]
    fn pre_parse_edit_message_data() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+1",
                "editMessage": {"dataMessage": {"message": "edited"}}
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+acct", true, false, &allow, |_| false);
        if let PreParse::Parsed(p) = res {
            assert_eq!(p.text, "edited");
        } else {
            panic!("expected parsed");
        }
    }

    #[test]
    fn pre_parse_story_ignored() {
        let env = json!({
            "envelope": {
                "sourceNumber": "+1",
                "storyMessage": {"x": 1},
                "dataMessage": {"message": "hi"}
            }
        });
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+acct", true, false, &allow, |_| false);
        assert!(matches!(res, PreParse::Ignored));
        // With ignore_stories=false, the story is not filtered.
        let (res2, _) = pre_parse_envelope(&env, "+acct", false, false, &allow, |_| false);
        assert!(matches!(res2, PreParse::Parsed(_)));
    }

    #[test]
    fn pre_parse_no_sender_ignored() {
        let env = json!({"envelope": {"dataMessage": {"message": "hi"}}});
        let allow = HashSet::new();
        let (res, _) = pre_parse_envelope(&env, "+acct", true, false, &allow, |_| false);
        assert!(matches!(res, PreParse::Ignored));
    }

    #[test]
    fn check_signal_requirements_env() {
        unsafe {
            std::env::remove_var("SIGNAL_HTTP_URL");
            std::env::remove_var("SIGNAL_ACCOUNT");
        }
        assert!(!check_signal_requirements());
        unsafe {
            std::env::set_var("SIGNAL_HTTP_URL", "http://x");
            std::env::set_var("SIGNAL_ACCOUNT", "+1");
        }
        assert!(check_signal_requirements());
        unsafe {
            std::env::remove_var("SIGNAL_HTTP_URL");
            std::env::remove_var("SIGNAL_ACCOUNT");
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: build a text SendResult mirroring `send`'s return.
// ---------------------------------------------------------------------------

/// Translate an RPC outcome from a text `send` into a [`SendResult`].
/// Signal has no editable message id, so success returns `message_id == None`.
/// Mirrors the tail of `send`.
pub fn send_text_result(outcome: &RpcOutcome) -> SendResult {
    match outcome {
        RpcOutcome::Result(v) if !v.is_null() => SendResult::ok(None),
        _ => SendResult::fail("RPC send failed"),
    }
}

/// Translate an RPC outcome from an attachment `send` into a [`SendResult`].
/// `media_label` shapes the failure message (e.g. "Image", "File").
/// Mirrors `_send_attachment` / `send_image`.
pub fn send_attachment_result(outcome: &RpcOutcome, media_label: &str) -> SendResult {
    match outcome {
        RpcOutcome::Result(v) if !v.is_null() => SendResult::ok(None),
        _ => SendResult::fail(format!("RPC send {} failed", media_label.to_lowercase())),
    }
}
