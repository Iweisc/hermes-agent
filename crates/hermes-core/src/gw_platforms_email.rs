//! Email platform adapter, ported from `gateway/platforms/email.py`.
//!
//! The Python module lets users talk to Hermes over email: it polls IMAP for new
//! messages and replies via SMTP. The bulk of behaviour-defining logic is pure:
//!   - environment-driven configuration ([`EmailConfig::from_env`],
//!     [`check_email_requirements`]),
//!   - automated/noreply sender detection ([`is_automated_sender`]),
//!   - RFC 2047 header decoding ([`decode_header_value`]),
//!   - MIME text-body extraction with HTML fallback ([`extract_text_body`]),
//!   - naive HTML tag stripping ([`strip_html`]),
//!   - `Name <addr>` address extraction ([`extract_email_address`]),
//!   - attachment extraction + caching ([`extract_attachments`]),
//!   - seen-UID trimming ([`trim_seen_uids`]),
//!   - inbound dispatch decisioning ([`evaluate_dispatch`], [`build_dispatch_event`]),
//!   - reply subject/threading-header construction + Message-ID generation
//!     ([`reply_subject`], [`build_outgoing_headers`], [`gen_message_id`]),
//!   - MIME message construction for replies / attachments
//!     ([`build_reply_mime`], [`build_attachment_mime`], [`serialize_mime`]).
//!
//! The IMAP/SMTP network sessions and the asyncio poll loop are intimately tied
//! to the CPython event loop and the `imaplib`/`smtplib` libraries; those are
//! modelled here as synchronous state + decision helpers that mirror the Python
//! state transitions. A minimal MIME message model is provided so callers (or a
//! `lettre`-based transport layer) can serialize and send.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageEvent`, `MessageType`, `SendResult`,
//!     `SessionSource`, `cache_image_from_bytes`, `cache_document_from_bytes`.

use std::collections::{HashMap, HashSet};

use base64::Engine;
use regex::Regex;

use crate::gw_platforms_base::{
    cache_document_from_bytes, cache_image_from_bytes, MessageEvent, MessageType, SendResult,
    SessionSource,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Automated sender substrings — emails whose address contains any of these are
/// silently ignored. Mirrors `_NOREPLY_PATTERNS`.
pub const NOREPLY_PATTERNS: &[&str] = &[
    "noreply",
    "no-reply",
    "no_reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
    "bounce",
    "notifications@",
    "automated@",
    "auto-confirm",
    "auto-reply",
    "automailer",
];

/// Gmail-safe max length per email body. Mirrors `MAX_MESSAGE_LENGTH`.
pub const MAX_MESSAGE_LENGTH: usize = 50_000;

/// Supported image extensions for inline detection. Mirrors `_IMAGE_EXTS`.
pub const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".gif", ".webp"];

/// Default IMAP port.
pub const DEFAULT_IMAP_PORT: u16 = 993;
/// Default SMTP port.
pub const DEFAULT_SMTP_PORT: u16 = 587;
/// Default poll interval, seconds.
pub const DEFAULT_POLL_INTERVAL: u64 = 15;
/// Cap on tracked seen-UIDs to prevent unbounded memory growth.
pub const SEEN_UIDS_MAX: usize = 2000;

// ===========================================================================
// Automated-sender detection
// ===========================================================================

/// Apply the RFC automated-header checks. Mirrors `_AUTOMATED_HEADERS`.
///
/// Returns True if any header present in `headers` indicates bulk/automated
/// mail. Header lookup is case-sensitive to match Python's `dict.get`, which is
/// how `email.message.Message.items()` keys are surfaced (canonical casing).
fn automated_header_hit(headers: &HashMap<String, String>) -> bool {
    // Auto-Submitted: != "no" (when present)
    if let Some(v) = headers.get("Auto-Submitted") {
        if !v.is_empty() && v.to_lowercase() != "no" {
            return true;
        }
    }
    // Precedence: bulk / list / junk
    if let Some(v) = headers.get("Precedence") {
        if !v.is_empty() {
            let lower = v.to_lowercase();
            if lower == "bulk" || lower == "list" || lower == "junk" {
                return true;
            }
        }
    }
    // X-Auto-Response-Suppress: any truthy value
    if let Some(v) = headers.get("X-Auto-Response-Suppress") {
        if !v.is_empty() {
            return true;
        }
    }
    // List-Unsubscribe: any truthy value
    if let Some(v) = headers.get("List-Unsubscribe") {
        if !v.is_empty() {
            return true;
        }
    }
    false
}

/// Return True if this email is from an automated/noreply source.
/// Mirrors `_is_automated_sender`.
pub fn is_automated_sender(address: &str, headers: &HashMap<String, String>) -> bool {
    let addr = address.to_lowercase();
    if NOREPLY_PATTERNS.iter().any(|pat| addr.contains(pat)) {
        return true;
    }
    automated_header_hit(headers)
}

/// Check if email platform dependencies (env credentials) are available.
/// Mirrors `check_email_requirements`.
pub fn check_email_requirements() -> bool {
    let getenv = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    getenv("EMAIL_ADDRESS").is_some()
        && getenv("EMAIL_PASSWORD").is_some()
        && getenv("EMAIL_IMAP_HOST").is_some()
        && getenv("EMAIL_SMTP_HOST").is_some()
}

// ===========================================================================
// RFC 2047 header decoding
// ===========================================================================

/// Decode an RFC 2047 encoded email header into a plain string.
/// Mirrors `_decode_header_value` (Python's `email.header.decode_header` +
/// charset decode, then `" ".join(...)`).
///
/// Handles `=?charset?B?...?=` (base64) and `=?charset?Q?...?=`
/// (quoted-printable) encoded-words, splitting the raw header into encoded and
/// non-encoded runs the same way CPython's parser does.
pub fn decode_header_value(raw: &str) -> String {
    let parts = decode_header(raw);
    let mut decoded: Vec<String> = Vec::new();
    for (bytes, charset) in parts {
        match charset {
            Some(cs) => decoded.push(decode_bytes(&bytes, &cs)),
            None => {
                // Non-encoded run: Python yields a str (already decoded by the
                // ASCII header parser). We keep the raw UTF-8 bytes.
                decoded.push(String::from_utf8_lossy(&bytes).into_owned());
            }
        }
    }
    decoded.join(" ")
}

/// One decoded segment: raw bytes + optional charset (None means it was an
/// unencoded text run).
type HeaderSegment = (Vec<u8>, Option<String>);

/// Split an RFC 2047 header into `(bytes, charset)` segments, mirroring
/// `email.header.decode_header`. Adjacent encoded-words separated only by
/// whitespace collapse (the whitespace is dropped) the way CPython does.
fn decode_header(raw: &str) -> Vec<HeaderSegment> {
    // Encoded-word pattern: =?charset?(B|Q)?text?=
    let ew = Regex::new(r"=\?([^?]+)\?([bBqQ])\?([^?]*)\?=").unwrap();

    let mut segments: Vec<HeaderSegment> = Vec::new();
    let mut last_end = 0usize;
    let mut prev_was_encoded = false;

    for caps in ew.captures_iter(raw) {
        let m = caps.get(0).unwrap();
        let between = &raw[last_end..m.start()];
        // Whitespace separating two encoded-words is discarded per RFC 2047.
        if !between.is_empty() && !(prev_was_encoded && between.trim().is_empty()) {
            segments.push((between.as_bytes().to_vec(), None));
        }

        let charset = caps.get(1).unwrap().as_str().to_string();
        let enc = caps.get(2).unwrap().as_str().to_ascii_uppercase();
        let text = caps.get(3).unwrap().as_str();
        let bytes = match enc.as_str() {
            "B" => base64::engine::general_purpose::STANDARD
                .decode(text)
                .or_else(|_| {
                    // tolerate missing padding
                    base64::engine::general_purpose::STANDARD_NO_PAD
                        .decode(text.trim_end_matches('='))
                })
                .unwrap_or_default(),
            _ => decode_q(text),
        };
        segments.push((bytes, Some(charset)));

        last_end = m.end();
        prev_was_encoded = true;
    }

    if last_end < raw.len() {
        let tail = &raw[last_end..];
        if !tail.is_empty() {
            segments.push((tail.as_bytes().to_vec(), None));
        }
    }

    if segments.is_empty() {
        segments.push((raw.as_bytes().to_vec(), None));
    }
    segments
}

/// Decode a Q-encoded (quoted-printable, RFC 2047 variant) string into bytes.
/// `_` maps to a space; `=XX` is a hex byte.
fn decode_q(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'_' => {
                out.push(b' ');
                i += 1;
            }
            b'=' if i + 2 < bytes.len() => {
                let hex = &text[i + 1..i + 3];
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(b'=');
                    i += 1;
                }
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    out
}

/// Decode bytes using a (best-effort) charset name, replacing invalid sequences
/// — mirrors Python `.decode(charset, errors="replace")` for the charsets that
/// actually occur in practice (utf-8, ascii, latin-1 family). Unknown charsets
/// fall back to UTF-8 lossy.
fn decode_bytes(bytes: &[u8], charset: &str) -> String {
    let cs = charset.trim().to_lowercase();
    match cs.as_str() {
        "utf-8" | "utf8" | "us-ascii" | "ascii" | "" => {
            String::from_utf8_lossy(bytes).into_owned()
        }
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" | "cp1252" | "windows-1252" => {
            // Latin-1: every byte maps 1:1 to U+0000..U+00FF.
            bytes.iter().map(|&b| b as char).collect()
        }
        _ => String::from_utf8_lossy(bytes).into_owned(),
    }
}

// ===========================================================================
// HTML stripping
// ===========================================================================

/// Naive HTML tag stripper for fallback text extraction. Mirrors `_strip_html`.
pub fn strip_html(html: &str) -> String {
    let br = Regex::new(r"(?i)<br\s*/?>").unwrap();
    let mut text = br.replace_all(html, "\n").into_owned();
    let p_open = Regex::new(r"(?i)<p[^>]*>").unwrap();
    text = p_open.replace_all(&text, "\n").into_owned();
    let p_close = Regex::new(r"(?i)</p>").unwrap();
    text = p_close.replace_all(&text, "\n").into_owned();
    let any_tag = Regex::new(r"<[^>]+>").unwrap();
    text = any_tag.replace_all(&text, "").into_owned();
    text = text
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">");
    let blanks = Regex::new(r"\n{3,}").unwrap();
    text = blanks.replace_all(&text, "\n\n").into_owned();
    text.trim().to_string()
}

// ===========================================================================
// Address extraction
// ===========================================================================

/// Extract bare email address from `Name <addr>` format. Mirrors
/// `_extract_email_address`.
pub fn extract_email_address(raw: &str) -> String {
    let re = Regex::new(r"<([^>]+)>").unwrap();
    if let Some(caps) = re.captures(raw) {
        return caps.get(1).unwrap().as_str().trim().to_lowercase();
    }
    raw.trim().to_lowercase()
}

/// Extract the display name from a raw `From` header, mirroring the Python
/// logic: decode the header, then if it still contains `<` strip everything from
/// the first `<` and remove surrounding double-quotes.
pub fn extract_sender_name(sender_raw: &str) -> String {
    let mut name = decode_header_value(sender_raw);
    if name.contains('<') {
        let prefix = name.split('<').next().unwrap_or("");
        name = prefix.trim().trim_matches('"').to_string();
    }
    name
}

// ===========================================================================
// MIME body / attachment model + extraction
// ===========================================================================

/// A single MIME part of a parsed inbound message. This is a deliberately small
/// model carrying just the fields the extractors need; a real IMAP fetch layer
/// populates these from the parsed `email.message.Message` walk.
#[derive(Debug, Clone, Default)]
pub struct MimePart {
    pub content_type: String,
    /// Content-Disposition header value (lowercase comparison applied in logic).
    pub disposition: String,
    /// Decoded payload bytes (Python `get_payload(decode=True)`), empty if none.
    pub payload: Vec<u8>,
    /// Charset from the part (defaults applied by callers).
    pub charset: Option<String>,
    /// Filename from `get_filename()` (already RFC2047-encoded raw, if present).
    pub filename: Option<String>,
    /// `get_content_subtype()` used to build a fallback filename.
    pub content_subtype: Option<String>,
}

/// Parsed inbound email: either a single part or a multipart walk.
#[derive(Debug, Clone, Default)]
pub struct ParsedEmail {
    pub multipart: bool,
    /// Parts in walk order (for multipart). For non-multipart, a single entry.
    pub parts: Vec<MimePart>,
}

impl ParsedEmail {
    pub fn is_multipart(&self) -> bool {
        self.multipart
    }
}

fn decode_payload(part: &MimePart) -> String {
    let charset = part.charset.clone().unwrap_or_else(|| "utf-8".to_string());
    decode_bytes(&part.payload, &charset)
}

/// Extract the plain-text body from a (potentially multipart) email.
/// Mirrors `_extract_text_body`.
pub fn extract_text_body(msg: &ParsedEmail) -> String {
    if msg.is_multipart() {
        // First pass: text/plain non-attachment parts.
        for part in &msg.parts {
            let disposition = part.disposition.to_lowercase();
            if disposition.contains("attachment") {
                continue;
            }
            if part.content_type == "text/plain" && !part.payload.is_empty() {
                return decode_payload(part);
            }
        }
        // Fallback: text/html stripped.
        for part in &msg.parts {
            let disposition = part.disposition.to_lowercase();
            if disposition.contains("attachment") {
                continue;
            }
            if part.content_type == "text/html" && !part.payload.is_empty() {
                return strip_html(&decode_payload(part));
            }
        }
        String::new()
    } else {
        match msg.parts.first() {
            Some(part) if !part.payload.is_empty() => {
                let text = decode_payload(part);
                if part.content_type == "text/html" {
                    strip_html(&text)
                } else {
                    text
                }
            }
            _ => String::new(),
        }
    }
}

/// An extracted attachment, cached locally. Mirrors the dicts produced by
/// `_extract_attachments`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    pub path: String,
    pub filename: String,
    /// "image" or "document".
    pub kind: String,
    pub media_type: String,
}

/// Lowercase file-extension (including the leading dot) of `filename`, mirroring
/// `Path(filename).suffix.lower()`.
fn suffix_lower(filename: &str) -> String {
    // Python's PurePath.suffix: the last component's extension, "" if none, and
    // dot-files like ".hidden" have no suffix.
    let base = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
    match base.rfind('.') {
        Some(idx) if idx > 0 => base[idx..].to_lowercase(),
        _ => String::new(),
    }
}

/// Extract attachment metadata and cache files locally. Mirrors
/// `_extract_attachments`. When `skip_attachments` is True, all
/// attachment/inline parts are ignored.
pub fn extract_attachments(msg: &ParsedEmail, skip_attachments: bool) -> Vec<Attachment> {
    let mut attachments: Vec<Attachment> = Vec::new();
    if !msg.is_multipart() {
        return attachments;
    }

    for part in &msg.parts {
        let disposition = part.disposition.to_lowercase();
        let is_attachment = disposition.contains("attachment");
        let is_inline = disposition.contains("inline");
        if skip_attachments && (is_attachment || is_inline) {
            continue;
        }
        if !is_attachment && !is_inline {
            continue;
        }
        // Skip text/plain and text/html body parts unless explicitly attachment.
        if (part.content_type == "text/plain" || part.content_type == "text/html")
            && !is_attachment
        {
            continue;
        }

        let filename = match &part.filename {
            Some(f) => decode_header_value(f),
            None => {
                let ext = part
                    .content_subtype
                    .clone()
                    .unwrap_or_else(|| "bin".to_string());
                format!("attachment.{ext}")
            }
        };

        if part.payload.is_empty() {
            continue;
        }

        let ext = suffix_lower(&filename);
        if IMAGE_EXTS.contains(&ext.as_str()) {
            match cache_image_from_bytes(&part.payload, &ext) {
                Ok(cached_path) => attachments.push(Attachment {
                    path: cached_path,
                    filename,
                    kind: "image".to_string(),
                    media_type: part.content_type.clone(),
                }),
                Err(_) => {
                    log::debug!("Skipping non-image attachment {filename} (invalid magic bytes)");
                    continue;
                }
            }
        } else {
            match cache_document_from_bytes(&part.payload, &filename) {
                Ok(cached_path) => attachments.push(Attachment {
                    path: cached_path,
                    filename,
                    kind: "document".to_string(),
                    media_type: part.content_type.clone(),
                }),
                Err(_) => continue,
            }
        }
    }

    attachments
}

// ===========================================================================
// Configuration
// ===========================================================================

/// Email adapter configuration sourced from environment + platform config.
/// Mirrors the fields read in `EmailAdapter.__init__`.
#[derive(Debug, Clone)]
pub struct EmailConfig {
    pub address: String,
    pub password: String,
    pub imap_host: String,
    pub imap_port: u16,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub poll_interval: u64,
    pub skip_attachments: bool,
}

impl EmailConfig {
    /// Build configuration from the process environment + the adapter's
    /// `config.extra` JSON (for `skip_attachments`). Mirrors `__init__`.
    ///
    /// Numeric env vars are parsed with `int(...)`; an unparsable value would
    /// raise in Python — here we fall back to the default to avoid panicking,
    /// which is the behaviour callers want for a config read.
    pub fn from_env(config_extra: &serde_json::Value) -> Self {
        let getenv = |k: &str| std::env::var(k).unwrap_or_default();
        let parse_port = |k: &str, default: u16| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<u16>().ok())
                .unwrap_or(default)
        };
        let skip_attachments = config_extra
            .get("skip_attachments")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        EmailConfig {
            address: getenv("EMAIL_ADDRESS"),
            password: getenv("EMAIL_PASSWORD"),
            imap_host: getenv("EMAIL_IMAP_HOST"),
            imap_port: parse_port("EMAIL_IMAP_PORT", DEFAULT_IMAP_PORT),
            smtp_host: getenv("EMAIL_SMTP_HOST"),
            smtp_port: parse_port("EMAIL_SMTP_PORT", DEFAULT_SMTP_PORT),
            poll_interval: std::env::var("EMAIL_POLL_INTERVAL")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(DEFAULT_POLL_INTERVAL),
            skip_attachments,
        }
    }
}

/// Parse `EMAIL_ALLOWED_USERS` into a lowercase set. Returns `None` when the
/// env var is empty/unset (meaning "no allowlist"). Mirrors the dispatch-time
/// parsing in `_dispatch_message`.
pub fn allowed_users_from_env() -> Option<HashSet<String>> {
    let raw = std::env::var("EMAIL_ALLOWED_USERS").unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let set: HashSet<String> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    Some(set)
}

// ===========================================================================
// Seen-UID tracking
// ===========================================================================

/// Keep only the most recent UIDs to prevent unbounded memory growth.
/// Mirrors `_trim_seen_uids`. UIDs are numeric strings; the highest `cap/2` are
/// kept. If any UID fails to parse as an integer, falls back to keeping the last
/// `cap/2` in iteration order.
pub fn trim_seen_uids(seen: &mut HashSet<String>, cap: usize) {
    if seen.len() <= cap {
        return;
    }
    let keep = cap / 2;
    // Try numeric sort.
    let mut as_nums: Vec<(u64, String)> = Vec::with_capacity(seen.len());
    let mut parse_ok = true;
    for u in seen.iter() {
        match u.parse::<u64>() {
            Ok(n) => as_nums.push((n, u.clone())),
            Err(_) => {
                parse_ok = false;
                break;
            }
        }
    }
    if parse_ok {
        as_nums.sort_by_key(|(n, _)| *n);
        let kept: HashSet<String> = as_nums
            .into_iter()
            .rev()
            .take(keep)
            .map(|(_, s)| s)
            .collect();
        *seen = kept;
        log::debug!("[Email] Trimmed seen UIDs to {} entries", seen.len());
    } else {
        // Fallback: keep last `keep` in arbitrary iteration order.
        let kept: HashSet<String> = seen
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .take(keep)
            .collect();
        *seen = kept;
    }
}

// ===========================================================================
// Inbound dispatch decisioning
// ===========================================================================

/// A fetched email's relevant fields. Mirrors the dict produced by
/// `_fetch_new_messages`.
#[derive(Debug, Clone, Default)]
pub struct FetchedMessage {
    pub uid: String,
    pub sender_addr: String,
    pub sender_name: String,
    pub subject: String,
    pub message_id: String,
    pub in_reply_to: String,
    pub body: String,
    pub attachments: Vec<Attachment>,
    pub date: String,
}

/// Outcome of the dispatch-time gating checks. Mirrors the early returns in
/// `_dispatch_message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Drop: sender is the agent itself.
    SelfMessage,
    /// Drop: automated/noreply sender.
    Automated,
    /// Drop: sender not in the allowlist.
    NotAllowlisted,
    /// Proceed to build a MessageEvent.
    Proceed,
}

/// Decide whether to dispatch a fetched message, mirroring the guard sequence in
/// `_dispatch_message`. `own_address` is the agent's `EMAIL_ADDRESS`;
/// `allowed_users` is the parsed allowlist (None = no allowlist).
pub fn evaluate_dispatch(
    sender_addr: &str,
    own_address: &str,
    allowed_users: Option<&HashSet<String>>,
) -> DispatchDecision {
    if sender_addr == own_address.to_lowercase() {
        return DispatchDecision::SelfMessage;
    }
    // Python calls `_is_automated_sender(sender_addr, {})` here — header checks
    // pass an empty dict, so only the noreply substring check applies.
    if is_automated_sender(sender_addr, &HashMap::new()) {
        return DispatchDecision::Automated;
    }
    if let Some(allowed) = allowed_users {
        if !allowed.contains(&sender_addr.to_lowercase()) {
            return DispatchDecision::NotAllowlisted;
        }
    }
    DispatchDecision::Proceed
}

/// Build the `MessageEvent` for a fetched message, mirroring the body of
/// `_dispatch_message` after the gating checks. Also returns the thread-context
/// entry (`subject`, `message_id`) the caller stores keyed by sender.
///
/// `platform` is the adapter's platform name (e.g. "email") used to fill
/// `SessionSource.platform`.
pub fn build_dispatch_event(msg: &FetchedMessage, platform: &str) -> (MessageEvent, ThreadContext) {
    let subject = msg.subject.clone();
    let body = msg.body.trim().to_string();

    // Build message text: include subject as context unless a reply.
    let text = if !subject.is_empty() && !subject.starts_with("Re:") {
        format!("[Subject: {subject}]\n\n{body}")
    } else {
        body
    };

    let mut media_urls: Vec<String> = Vec::new();
    let mut media_types: Vec<String> = Vec::new();
    let mut msg_type = MessageType::Text;
    for att in &msg.attachments {
        media_urls.push(att.path.clone());
        media_types.push(att.media_type.clone());
        if att.kind == "image" {
            msg_type = MessageType::Photo;
        }
    }

    let thread = ThreadContext {
        subject: subject.clone(),
        message_id: msg.message_id.clone(),
    };

    let name = if msg.sender_name.is_empty() {
        msg.sender_addr.clone()
    } else {
        msg.sender_name.clone()
    };

    let source = SessionSource {
        platform: platform.to_string(),
        chat_id: msg.sender_addr.clone(),
        chat_name: Some(name.clone()),
        chat_type: "dm".to_string(),
        user_id: Some(msg.sender_addr.clone()),
        user_name: Some(name),
        ..Default::default()
    };

    let event = MessageEvent {
        text: if text.is_empty() {
            "(empty email)".to_string()
        } else {
            text
        },
        message_type: msg_type,
        source,
        message_id: if msg.message_id.is_empty() {
            None
        } else {
            Some(msg.message_id.clone())
        },
        media_urls,
        media_types,
        reply_to_message_id: if msg.in_reply_to.is_empty() {
            None
        } else {
            Some(msg.in_reply_to.clone())
        },
        ..Default::default()
    };

    (event, thread)
}

// ===========================================================================
// Threading context + outgoing headers
// ===========================================================================

/// Reply-threading context stored per sender. Mirrors entries in
/// `_thread_context`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadContext {
    pub subject: String,
    pub message_id: String,
}

/// Compute the reply subject from a stored context subject. Mirrors the
/// `subject = ctx.get("subject", "Hermes Agent"); if not Re: -> "Re: ..."`
/// logic shared by all `_send_email*` helpers.
///
/// `ctx_subject` is `None` when there's no stored context (Python's
/// `ctx.get("subject", "Hermes Agent")` default).
pub fn reply_subject(ctx_subject: Option<&str>) -> String {
    let subject = ctx_subject.unwrap_or("Hermes Agent");
    if subject.starts_with("Re:") {
        subject.to_string()
    } else {
        format!("Re: {subject}")
    }
}

/// Generate a Message-ID like `<hermes-{12hex}@{domain}>`, where `domain` is the
/// part after `@` in the agent's own address. Mirrors
/// `f"<hermes-{uuid.uuid4().hex[:12]}@{self._address.split('@')[1]}>"`.
pub fn gen_message_id(own_address: &str) -> String {
    let domain = own_address.split('@').nth(1).unwrap_or("");
    format!("<hermes-{}@{}>", uuid12(), domain)
}

/// 12 hex chars, mirroring `uuid.uuid4().hex[:12]` (only needs to be unique
/// enough for a Message-ID).
fn uuid12() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    format!("{:012x}", mixed & 0xFFFF_FFFF_FFFF)
}

/// RFC 2822 date string (localtime), mirroring `email.utils.formatdate(localtime=True)`.
pub fn format_date_now() -> String {
    let now = chrono::Local::now();
    now.format("%a, %d %b %Y %H:%M:%S %z").to_string()
}

// ===========================================================================
// MIME message construction
// ===========================================================================

/// A minimal serialized email message. Headers preserve insertion order; the
/// body is `text/plain; charset=utf-8` plus optional base64 attachments, which
/// is exactly what the Python `_send_email*` helpers build.
#[derive(Debug, Clone)]
pub struct OutgoingEmail {
    /// Ordered headers (key, value).
    pub headers: Vec<(String, String)>,
    pub message_id: String,
    /// The plain-text body (may be empty for attachment-only messages).
    pub body: String,
    pub attachments: Vec<OutgoingAttachment>,
}

/// A file attached to an outgoing email.
#[derive(Debug, Clone)]
pub struct OutgoingAttachment {
    pub filename: String,
    pub data: Vec<u8>,
}

/// Build the common outgoing headers (From/To/Subject/threading/Date/Message-ID),
/// mirroring the shared header block in the three `_send_email*` helpers.
///
/// `reply_to_msg_id` overrides the stored context's message-id for threading
/// (used by `_send_email`); pass `None` for the attachment helpers which only
/// use the stored context.
pub fn build_outgoing_headers(
    own_address: &str,
    to_addr: &str,
    ctx: Option<&ThreadContext>,
    reply_to_msg_id: Option<&str>,
) -> (Vec<(String, String)>, String) {
    let mut headers: Vec<(String, String)> = Vec::new();
    headers.push(("From".to_string(), own_address.to_string()));
    headers.push(("To".to_string(), to_addr.to_string()));

    let ctx_subject = ctx.map(|c| c.subject.as_str());
    let subject = reply_subject(ctx_subject);
    headers.push(("Subject".to_string(), subject));

    // Threading headers.
    let ctx_msg_id = ctx.map(|c| c.message_id.as_str()).filter(|s| !s.is_empty());
    let original_msg_id = reply_to_msg_id.filter(|s| !s.is_empty()).or(ctx_msg_id);
    if let Some(mid) = original_msg_id {
        headers.push(("In-Reply-To".to_string(), mid.to_string()));
        headers.push(("References".to_string(), mid.to_string()));
    }

    headers.push(("Date".to_string(), format_date_now()));
    let msg_id = gen_message_id(own_address);
    headers.push(("Message-ID".to_string(), msg_id.clone()));

    (headers, msg_id)
}

/// Build a plain-text reply email. Mirrors `_send_email`.
pub fn build_reply_mime(
    own_address: &str,
    to_addr: &str,
    body: &str,
    ctx: Option<&ThreadContext>,
    reply_to_msg_id: Option<&str>,
) -> OutgoingEmail {
    let (headers, message_id) = build_outgoing_headers(own_address, to_addr, ctx, reply_to_msg_id);
    OutgoingEmail {
        headers,
        message_id,
        body: body.to_string(),
        attachments: Vec::new(),
    }
}

/// Build an email with one or more attachments. Mirrors
/// `_send_email_with_attachments` / `_send_email_with_attachment`. The body part
/// is only attached when non-empty (matching `if body:`).
pub fn build_attachment_mime(
    own_address: &str,
    to_addr: &str,
    body: &str,
    attachments: Vec<OutgoingAttachment>,
    ctx: Option<&ThreadContext>,
) -> OutgoingEmail {
    let (headers, message_id) = build_outgoing_headers(own_address, to_addr, ctx, None);
    OutgoingEmail {
        headers,
        message_id,
        body: body.to_string(),
        attachments,
    }
}

/// Serialize an [`OutgoingEmail`] to a MIME wire string (multipart/mixed with a
/// boundary), mirroring the structure CPython's `MIMEMultipart` produces:
/// a `text/plain; charset="utf-8"` body part (when present) followed by
/// base64-encoded `application/octet-stream` attachments.
pub fn serialize_mime(email: &OutgoingEmail) -> String {
    let boundary = format!("===============hermes-{}==", uuid12());
    let mut out = String::new();
    for (k, v) in &email.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("MIME-Version: 1.0\r\n");
    out.push_str(&format!(
        "Content-Type: multipart/mixed; boundary=\"{boundary}\"\r\n"
    ));
    out.push_str("\r\n");

    if !email.body.is_empty() {
        out.push_str(&format!("--{boundary}\r\n"));
        out.push_str("Content-Type: text/plain; charset=\"utf-8\"\r\n");
        out.push_str("Content-Transfer-Encoding: base64\r\n\r\n");
        let encoded = base64::engine::general_purpose::STANDARD.encode(email.body.as_bytes());
        out.push_str(&wrap_base64(&encoded));
        out.push_str("\r\n");
    }

    for att in &email.attachments {
        out.push_str(&format!("--{boundary}\r\n"));
        out.push_str("Content-Type: application/octet-stream\r\n");
        out.push_str("Content-Transfer-Encoding: base64\r\n");
        out.push_str(&format!(
            "Content-Disposition: attachment; filename={}\r\n\r\n",
            att.filename
        ));
        let encoded = base64::engine::general_purpose::STANDARD.encode(&att.data);
        out.push_str(&wrap_base64(&encoded));
        out.push_str("\r\n");
    }

    out.push_str(&format!("--{boundary}--\r\n"));
    out
}

/// Wrap base64 text at 76 columns (RFC line-length), matching the email lib.
fn wrap_base64(s: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        let end = (i + 76).min(s.len());
        out.push_str(&s[i..end]);
        out.push_str("\r\n");
        i = end;
    }
    out
}

// ===========================================================================
// send_image / send_multiple_images body construction
// ===========================================================================

/// Compose the body for `send_image`: caption + linked URL. Mirrors
/// `send_image`'s `text = caption or ""; text += "\n\nImage: {url}"; .strip()`.
pub fn compose_image_body(caption: Option<&str>, image_url: &str) -> String {
    let mut text = caption.unwrap_or("").to_string();
    text.push_str(&format!("\n\nImage: {image_url}"));
    text.trim().to_string()
}

/// Result of partitioning `send_multiple_images` inputs. Mirrors the loop in
/// `send_multiple_images`: `file://` URLs that exist become local attachments,
/// everything else (and alt-text) becomes body lines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiImagePlan {
    /// Local file paths to attach.
    pub local_paths: Vec<String>,
    /// Body assembled from alt-texts and remote-image links, joined by `\n\n`.
    pub body: String,
    /// True when there's nothing to send (early `return`).
    pub empty: bool,
}

/// Partition `(image_url, alt_text)` pairs into local attachments + a body.
/// `path_exists` validates `file://` paths (inject `|p| Path::new(p).exists()`).
/// Mirrors `send_multiple_images` up to the SMTP send.
pub fn plan_multiple_images<F>(images: &[(String, String)], path_exists: F) -> MultiImagePlan
where
    F: Fn(&str) -> bool,
{
    if images.is_empty() {
        return MultiImagePlan {
            empty: true,
            ..Default::default()
        };
    }

    let mut body_parts: Vec<String> = Vec::new();
    let mut local_paths: Vec<String> = Vec::new();
    for (image_url, alt_text) in images {
        if !alt_text.is_empty() {
            body_parts.push(alt_text.clone());
        }
        if let Some(rest) = image_url.strip_prefix("file://") {
            let local_path = url_unquote(rest);
            if path_exists(&local_path) {
                local_paths.push(local_path);
            } else {
                log::warn!("[Email] Skipping missing image: {local_path}");
            }
        } else {
            body_parts.push(format!("Image: {image_url}"));
        }
    }

    if local_paths.is_empty() && body_parts.is_empty() {
        return MultiImagePlan {
            empty: true,
            ..Default::default()
        };
    }

    MultiImagePlan {
        local_paths,
        body: body_parts.join("\n\n"),
        empty: false,
    }
}

/// Percent-decode a URL component, mirroring `urllib.parse.unquote`.
fn url_unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ===========================================================================
// get_chat_info
// ===========================================================================

/// Build the basic chat-info map for `get_chat_info`. `subject` is the stored
/// context subject (empty when no context). Returns a JSON object matching the
/// Python dict shape.
pub fn chat_info(chat_id: &str, subject: &str) -> serde_json::Value {
    serde_json::json!({
        "name": chat_id,
        "type": "dm",
        "chat_id": chat_id,
        "subject": subject,
    })
}

// ===========================================================================
// SendResult convenience for send()
// ===========================================================================

/// Build a successful SendResult carrying the generated Message-ID. Mirrors the
/// `send`/`send_document` success path.
pub fn send_ok(message_id: String) -> SendResult {
    SendResult::ok(Some(message_id))
}

/// Build a failed SendResult from an error string. Mirrors the `except` path.
pub fn send_err(error: impl Into<String>) -> SendResult {
    SendResult::fail(error)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automated_sender_noreply_patterns() {
        let empty = HashMap::new();
        assert!(is_automated_sender("noreply@x.com", &empty));
        assert!(is_automated_sender("MAILER-DAEMON@x.com", &empty));
        assert!(is_automated_sender("notifications@github.com", &empty));
        assert!(!is_automated_sender("alice@x.com", &empty));
    }

    #[test]
    fn automated_sender_headers() {
        let mut h = HashMap::new();
        h.insert("Precedence".to_string(), "bulk".to_string());
        assert!(is_automated_sender("alice@x.com", &h));

        let mut h2 = HashMap::new();
        h2.insert("Auto-Submitted".to_string(), "auto-generated".to_string());
        assert!(is_automated_sender("alice@x.com", &h2));
        // Auto-Submitted: no -> not automated.
        let mut h3 = HashMap::new();
        h3.insert("Auto-Submitted".to_string(), "no".to_string());
        assert!(!is_automated_sender("alice@x.com", &h3));

        let mut h4 = HashMap::new();
        h4.insert("List-Unsubscribe".to_string(), "<mailto:x>".to_string());
        assert!(is_automated_sender("alice@x.com", &h4));

        // Precedence: normal -> not automated.
        let mut h5 = HashMap::new();
        h5.insert("Precedence".to_string(), "normal".to_string());
        assert!(!is_automated_sender("alice@x.com", &h5));
    }

    #[test]
    fn decode_header_plain_and_encoded() {
        assert_eq!(decode_header_value("Hello World"), "Hello World");
        // base64 encoded-word "Hello" in utf-8.
        assert_eq!(decode_header_value("=?utf-8?B?SGVsbG8=?="), "Hello");
        // Q-encoded with underscore-as-space.
        assert_eq!(decode_header_value("=?utf-8?Q?Hello_World?="), "Hello World");
        // Q-encoded hex byte (é = C3 A9 in utf-8).
        assert_eq!(decode_header_value("=?utf-8?Q?caf=C3=A9?="), "café");
    }

    #[test]
    fn extract_address_variants() {
        assert_eq!(
            extract_email_address("Alice <Alice@Example.COM>"),
            "alice@example.com"
        );
        assert_eq!(extract_email_address("  Bob@X.com  "), "bob@x.com");
    }

    #[test]
    fn sender_name_strips_angle_and_quotes() {
        assert_eq!(extract_sender_name("\"Alice Smith\" <a@x.com>"), "Alice Smith");
        assert_eq!(extract_sender_name("Bob <b@x.com>"), "Bob");
        assert_eq!(extract_sender_name("plainname"), "plainname");
    }

    #[test]
    fn strip_html_basics() {
        let html = "<p>Hello</p><br/>World &amp; <b>more</b>&nbsp;text";
        let out = strip_html(html);
        assert!(out.contains("Hello"));
        assert!(out.contains("World & "));
        assert!(out.contains("more"));
        assert!(!out.contains('<'));
    }

    #[test]
    fn suffix_lower_cases() {
        assert_eq!(suffix_lower("Photo.JPG"), ".jpg");
        assert_eq!(suffix_lower("archive.tar.gz"), ".gz");
        assert_eq!(suffix_lower("noext"), "");
        assert_eq!(suffix_lower(".hidden"), "");
        assert_eq!(suffix_lower("dir/file.PNG"), ".png");
    }

    #[test]
    fn extract_text_body_multipart_prefers_plain() {
        let msg = ParsedEmail {
            multipart: true,
            parts: vec![
                MimePart {
                    content_type: "text/html".into(),
                    payload: b"<p>html body</p>".to_vec(),
                    ..Default::default()
                },
                MimePart {
                    content_type: "text/plain".into(),
                    payload: b"plain body".to_vec(),
                    ..Default::default()
                },
            ],
        };
        assert_eq!(extract_text_body(&msg), "plain body");
    }

    #[test]
    fn extract_text_body_html_fallback() {
        let msg = ParsedEmail {
            multipart: true,
            parts: vec![MimePart {
                content_type: "text/html".into(),
                payload: b"<p>hi</p>".to_vec(),
                ..Default::default()
            }],
        };
        assert_eq!(extract_text_body(&msg), "hi");
    }

    #[test]
    fn extract_text_body_singlepart() {
        let msg = ParsedEmail {
            multipart: false,
            parts: vec![MimePart {
                content_type: "text/plain".into(),
                payload: b"single".to_vec(),
                ..Default::default()
            }],
        };
        assert_eq!(extract_text_body(&msg), "single");
    }

    #[test]
    fn trim_seen_uids_numeric() {
        let mut seen: HashSet<String> = (1..=10).map(|n| n.to_string()).collect();
        trim_seen_uids(&mut seen, 4); // keep top 2
        assert_eq!(seen.len(), 2);
        assert!(seen.contains("10"));
        assert!(seen.contains("9"));
    }

    #[test]
    fn trim_seen_uids_below_cap_noop() {
        let mut seen: HashSet<String> = (1..=3).map(|n| n.to_string()).collect();
        trim_seen_uids(&mut seen, 10);
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn dispatch_decisions() {
        let mut allowed = HashSet::new();
        allowed.insert("alice@x.com".to_string());

        assert_eq!(
            evaluate_dispatch("agent@x.com", "Agent@X.com", Some(&allowed)),
            DispatchDecision::SelfMessage
        );
        assert_eq!(
            evaluate_dispatch("noreply@x.com", "agent@x.com", None),
            DispatchDecision::Automated
        );
        assert_eq!(
            evaluate_dispatch("bob@x.com", "agent@x.com", Some(&allowed)),
            DispatchDecision::NotAllowlisted
        );
        assert_eq!(
            evaluate_dispatch("alice@x.com", "agent@x.com", Some(&allowed)),
            DispatchDecision::Proceed
        );
        // No allowlist -> proceed for any non-automated, non-self sender.
        assert_eq!(
            evaluate_dispatch("bob@x.com", "agent@x.com", None),
            DispatchDecision::Proceed
        );
    }

    #[test]
    fn build_event_subject_prefix_and_type() {
        let msg = FetchedMessage {
            sender_addr: "alice@x.com".into(),
            sender_name: "Alice".into(),
            subject: "Hello".into(),
            message_id: "<m1>".into(),
            in_reply_to: "<m0>".into(),
            body: "  hi there  ".into(),
            attachments: vec![Attachment {
                path: "/cache/img.png".into(),
                filename: "img.png".into(),
                kind: "image".into(),
                media_type: "image/png".into(),
            }],
            ..Default::default()
        };
        let (ev, thread) = build_dispatch_event(&msg, "email");
        assert_eq!(ev.text, "[Subject: Hello]\n\nhi there");
        assert_eq!(ev.message_type, MessageType::Photo);
        assert_eq!(ev.media_urls, vec!["/cache/img.png".to_string()]);
        assert_eq!(ev.reply_to_message_id.as_deref(), Some("<m0>"));
        assert_eq!(ev.source.chat_id, "alice@x.com");
        assert_eq!(ev.source.platform, "email");
        assert_eq!(thread.subject, "Hello");
        assert_eq!(thread.message_id, "<m1>");
    }

    #[test]
    fn build_event_reply_subject_not_prefixed() {
        let msg = FetchedMessage {
            sender_addr: "alice@x.com".into(),
            subject: "Re: Hello".into(),
            body: "reply body".into(),
            ..Default::default()
        };
        let (ev, _) = build_dispatch_event(&msg, "email");
        assert_eq!(ev.text, "reply body");
    }

    #[test]
    fn build_event_empty_email_placeholder() {
        let msg = FetchedMessage {
            sender_addr: "alice@x.com".into(),
            subject: "Re:".into(),
            body: "   ".into(),
            ..Default::default()
        };
        let (ev, _) = build_dispatch_event(&msg, "email");
        assert_eq!(ev.text, "(empty email)");
    }

    #[test]
    fn reply_subject_logic() {
        assert_eq!(reply_subject(None), "Re: Hermes Agent");
        assert_eq!(reply_subject(Some("Hello")), "Re: Hello");
        assert_eq!(reply_subject(Some("Re: Hello")), "Re: Hello");
    }

    #[test]
    fn message_id_format() {
        let mid = gen_message_id("agent@example.com");
        assert!(mid.starts_with("<hermes-"));
        assert!(mid.ends_with("@example.com>"));
    }

    #[test]
    fn outgoing_headers_threading() {
        let ctx = ThreadContext {
            subject: "Hello".into(),
            message_id: "<orig>".into(),
        };
        let (headers, _mid) = build_outgoing_headers("agent@x.com", "alice@x.com", Some(&ctx), None);
        let get = |k: &str| {
            headers
                .iter()
                .find(|(hk, _)| hk == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("From"), Some("agent@x.com"));
        assert_eq!(get("To"), Some("alice@x.com"));
        assert_eq!(get("Subject"), Some("Re: Hello"));
        assert_eq!(get("In-Reply-To"), Some("<orig>"));
        assert_eq!(get("References"), Some("<orig>"));

        // reply_to_msg_id overrides the context message-id.
        let (headers2, _) =
            build_outgoing_headers("agent@x.com", "alice@x.com", Some(&ctx), Some("<override>"));
        let get2 = |k: &str| {
            headers2
                .iter()
                .find(|(hk, _)| hk == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get2("In-Reply-To"), Some("<override>"));
    }

    #[test]
    fn outgoing_headers_no_context() {
        let (headers, _) = build_outgoing_headers("agent@x.com", "alice@x.com", None, None);
        let subject = headers
            .iter()
            .find(|(k, _)| k == "Subject")
            .map(|(_, v)| v.as_str());
        assert_eq!(subject, Some("Re: Hermes Agent"));
        // No threading headers without a message-id.
        assert!(!headers.iter().any(|(k, _)| k == "In-Reply-To"));
    }

    #[test]
    fn serialize_reply_contains_body() {
        let email = build_reply_mime("agent@x.com", "alice@x.com", "the body", None, None);
        let wire = serialize_mime(&email);
        assert!(wire.contains("From: agent@x.com"));
        assert!(wire.contains("multipart/mixed"));
        // body base64.
        let b64 = base64::engine::general_purpose::STANDARD.encode("the body");
        assert!(wire.contains(&b64));
    }

    #[test]
    fn serialize_attachment_disposition() {
        let email = build_attachment_mime(
            "agent@x.com",
            "alice@x.com",
            "",
            vec![OutgoingAttachment {
                filename: "report.pdf".into(),
                data: b"PDFDATA".to_vec(),
            }],
            None,
        );
        let wire = serialize_mime(&email);
        assert!(wire.contains("Content-Disposition: attachment; filename=report.pdf"));
        // empty body -> no text/plain part.
        assert!(!wire.contains("text/plain"));
    }

    #[test]
    fn compose_image_body_format() {
        assert_eq!(
            compose_image_body(Some("cap"), "https://x/a.png"),
            "cap\n\nImage: https://x/a.png"
        );
        assert_eq!(
            compose_image_body(None, "https://x/a.png"),
            "Image: https://x/a.png"
        );
    }

    #[test]
    fn plan_images_local_and_remote() {
        let images = vec![
            ("file:///tmp/exists.png".to_string(), "alt1".to_string()),
            ("file:///tmp/missing.png".to_string(), "".to_string()),
            ("https://x/remote.jpg".to_string(), "alt2".to_string()),
        ];
        let plan = plan_multiple_images(&images, |p| p == "/tmp/exists.png");
        assert!(!plan.empty);
        assert_eq!(plan.local_paths, vec!["/tmp/exists.png".to_string()]);
        assert!(plan.body.contains("alt1"));
        assert!(plan.body.contains("alt2"));
        assert!(plan.body.contains("Image: https://x/remote.jpg"));
    }

    #[test]
    fn plan_images_empty() {
        let plan = plan_multiple_images(&[], |_| true);
        assert!(plan.empty);
    }

    #[test]
    fn url_unquote_decodes() {
        assert_eq!(url_unquote("/tmp/a%20b.png"), "/tmp/a b.png");
        assert_eq!(url_unquote("/plain/path.png"), "/plain/path.png");
    }

    #[test]
    fn allowed_users_parsing() {
        unsafe {
            std::env::set_var("EMAIL_ALLOWED_USERS", " Alice@X.com , bob@y.com ,");
        }
        let set = allowed_users_from_env().unwrap();
        assert!(set.contains("alice@x.com"));
        assert!(set.contains("bob@y.com"));
        assert_eq!(set.len(), 2);
        unsafe {
            std::env::set_var("EMAIL_ALLOWED_USERS", "");
        }
        assert!(allowed_users_from_env().is_none());
        unsafe {
            std::env::remove_var("EMAIL_ALLOWED_USERS");
        }
    }

    #[test]
    fn chat_info_shape() {
        let info = chat_info("alice@x.com", "Hello");
        assert_eq!(info["name"], "alice@x.com");
        assert_eq!(info["type"], "dm");
        assert_eq!(info["chat_id"], "alice@x.com");
        assert_eq!(info["subject"], "Hello");
    }

    #[test]
    fn extract_attachments_skip_flag() {
        let msg = ParsedEmail {
            multipart: true,
            parts: vec![MimePart {
                content_type: "application/pdf".into(),
                disposition: "attachment; filename=\"a.pdf\"".into(),
                payload: b"data".to_vec(),
                filename: Some("a.pdf".into()),
                ..Default::default()
            }],
        };
        // skip_attachments -> nothing.
        assert!(extract_attachments(&msg, true).is_empty());
    }

    #[test]
    fn config_from_env_defaults() {
        unsafe {
            std::env::remove_var("EMAIL_IMAP_PORT");
            std::env::remove_var("EMAIL_SMTP_PORT");
            std::env::remove_var("EMAIL_POLL_INTERVAL");
            std::env::set_var("EMAIL_ADDRESS", "agent@x.com");
        }
        let cfg = EmailConfig::from_env(&serde_json::json!({"skip_attachments": true}));
        assert_eq!(cfg.address, "agent@x.com");
        assert_eq!(cfg.imap_port, DEFAULT_IMAP_PORT);
        assert_eq!(cfg.smtp_port, DEFAULT_SMTP_PORT);
        assert_eq!(cfg.poll_interval, DEFAULT_POLL_INTERVAL);
        assert!(cfg.skip_attachments);
        unsafe {
            std::env::remove_var("EMAIL_ADDRESS");
        }
    }
}
