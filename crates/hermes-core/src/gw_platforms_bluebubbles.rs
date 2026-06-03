//! BlueBubbles iMessage platform adapter — native Rust port of
//! `gateway/platforms/bluebubbles.py`.
//!
//! The Python module wraps a local BlueBubbles macOS server: it uses an
//! `httpx.AsyncClient` for outbound REST sends (text / media / typing / read /
//! tapback) and an `aiohttp` web server to receive inbound webhooks. The async
//! lifecycle (the aiohttp webhook listener, `asyncio.create_task` fire-and-forget
//! handlers, `httpx.AsyncClient` connection pooling) is intimately tied to the
//! CPython event loop and lives in the async runtime layer that drives this
//! state.
//!
//! What is ported here is the **deterministic, side-effect-free logic** plus
//! request-construction / response-parsing that other Hermes code and tests
//! depend on, reproducing Python behaviour exactly:
//!
//! - [`redact`] — phone / email redaction for logs.
//! - [`check_bluebubbles_requirements`] — dependency gate.
//! - [`normalize_server_url`] — server URL canonicalisation.
//! - [`BlueBubblesConfig`] — env/extra parsing done once in `__init__`.
//! - [`api_url`] — credential-bearing URL construction (`urllib.parse.quote`,
//!   `safe=''`).
//! - [`webhook_url`] / [`webhook_register_url`] — webhook registration URLs.
//! - [`tapback_added`] / [`tapback_removed`] / [`is_tapback_type`] — tapback codes.
//! - [`truncate_message`] — pagination-suffix-stripped truncation.
//! - [`split_send_chunks`] — paragraph + truncation chunking for `send`.
//! - [`attachment_ext_for_mime`] / [`AttachmentKind`] — inbound download routing.
//! - [`is_message_event`] — webhook event-type gate.
//! - [`WebhookOutcome`] / [`parse_webhook`] — full inbound webhook parsing.
//! - [`is_group_chat`] — group-chat detection.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageEvent`, `MessageType`, `SendResult`,
//!     `SessionSource`, `truncate_message`, cache helpers.
//!   - [`crate::gw_helpers::strip_markdown`] — `format_message`.

use serde_json::Value;

use crate::gw_platforms_base::{MessageType, SessionSource};

// ===========================================================================
// Constants
// ===========================================================================

pub const DEFAULT_WEBHOOK_HOST: &str = "127.0.0.1";
pub const DEFAULT_WEBHOOK_PORT: u16 = 8645;
pub const DEFAULT_WEBHOOK_PATH: &str = "/bluebubbles-webhook";
pub const MAX_TEXT_LENGTH: usize = 4000;

/// Tapback "added" reaction codes (`associatedMessageType` 2000–2005).
pub fn tapback_added() -> &'static [(i64, &'static str)] {
    &[
        (2000, "love"),
        (2001, "like"),
        (2002, "dislike"),
        (2003, "laugh"),
        (2004, "emphasize"),
        (2005, "question"),
    ]
}

/// Tapback "removed" reaction codes (`associatedMessageType` 3000–3005).
pub fn tapback_removed() -> &'static [(i64, &'static str)] {
    &[
        (3000, "love"),
        (3001, "like"),
        (3002, "dislike"),
        (3003, "laugh"),
        (3004, "emphasize"),
        (3005, "question"),
    ]
}

/// Return True when `code` is any added or removed tapback type.
pub fn is_tapback_type(code: i64) -> bool {
    tapback_added().iter().any(|(c, _)| *c == code)
        || tapback_removed().iter().any(|(c, _)| *c == code)
}

/// Webhook event types that carry user messages (`_MESSAGE_EVENTS`).
pub const MESSAGE_EVENTS: &[&str] = &["new-message", "message", "updated-message"];

// ===========================================================================
// Log redaction
// ===========================================================================

thread_local! {
    static PHONE_RE: regex::Regex = regex::Regex::new(r"\+?\d{7,15}").unwrap();
    static EMAIL_RE: regex::Regex = regex::Regex::new(r"[\w.+-]+@[\w-]+\.[\w.]+").unwrap();
    static SERVER_URL_RE: regex::Regex = regex::Regex::new(r"(?i)^https?://").unwrap();
    static PAGINATION_RE: regex::Regex = regex::Regex::new(r"\s*\(\d+/\d+\)$").unwrap();
    static PHONE_PREFIX_RE: regex::Regex = regex::Regex::new(r"^\+\d+").unwrap();
    static PARAGRAPH_RE: regex::Regex = regex::Regex::new(r"\n\s*\n").unwrap();
}

/// Redact phone numbers and emails from log output. Mirrors `_redact`.
pub fn redact(text: &str) -> String {
    PHONE_RE.with(|phone| {
        EMAIL_RE.with(|email| {
            let stage1 = phone.replace_all(text, "[REDACTED]").into_owned();
            email.replace_all(&stage1, "[REDACTED]").into_owned()
        })
    })
}

// ===========================================================================
// Requirements gate
// ===========================================================================

/// Mirrors `check_bluebubbles_requirements`. In Python this probes for the
/// `aiohttp` and `httpx` imports; in the native runtime those transports are
/// provided by the Rust HTTP stack, so the precondition is always satisfied.
pub fn check_bluebubbles_requirements() -> bool {
    true
}

// ===========================================================================
// Server URL normalization
// ===========================================================================

/// Normalize a raw server URL. Mirrors `_normalize_server_url`:
/// trims, prepends `http://` if no scheme, strips trailing slashes.
pub fn normalize_server_url(raw: &str) -> String {
    let value = raw.trim();
    if value.is_empty() {
        return String::new();
    }
    let value = if SERVER_URL_RE.with(|re| re.is_match(value)) {
        value.to_string()
    } else {
        format!("http://{value}")
    };
    value.trim_end_matches('/').to_string()
}

// ===========================================================================
// Config parsing (mirrors __init__)
// ===========================================================================

/// Parsed BlueBubbles configuration, mirroring the field assignments done once
/// in `BlueBubblesAdapter.__init__`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBubblesConfig {
    pub server_url: String,
    pub password: String,
    pub webhook_host: String,
    pub webhook_port: u16,
    pub webhook_path: String,
    pub send_read_receipts: bool,
}

/// A lookup function for env + extra values. `extra` is the adapter's
/// `config.extra` mapping as JSON; env reads use `std::env::var`.
fn extra_str(extra: &Value, key: &str) -> Option<String> {
    match extra.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        // Python `extra.get("server_url") or os.getenv(...)` — only non-empty
        // strings are truthy; other JSON types fall through to env.
        _ => None,
    }
}

impl BlueBubblesConfig {
    /// Build config from the adapter's `config.extra` JSON object plus the
    /// process environment. Mirrors `BlueBubblesAdapter.__init__`.
    pub fn from_extra(extra: &Value) -> Self {
        let env = |k: &str| std::env::var(k).unwrap_or_default();

        let server_url_raw = extra_str(extra, "server_url")
            .unwrap_or_else(|| env("BLUEBUBBLES_SERVER_URL"));
        let server_url = normalize_server_url(&server_url_raw);

        let password = extra_str(extra, "password")
            .unwrap_or_else(|| env("BLUEBUBBLES_PASSWORD"));

        let webhook_host = extra_str(extra, "webhook_host").unwrap_or_else(|| {
            let v = env("BLUEBUBBLES_WEBHOOK_HOST");
            if v.is_empty() {
                DEFAULT_WEBHOOK_HOST.to_string()
            } else {
                v
            }
        });

        // Python: int(extra.get("webhook_port") or os.getenv(..., str(DEFAULT))).
        let webhook_port = {
            let raw = extra
                .get("webhook_port")
                .and_then(port_from_value)
                .unwrap_or_else(|| {
                    let v = env("BLUEBUBBLES_WEBHOOK_PORT");
                    if v.is_empty() {
                        DEFAULT_WEBHOOK_PORT.to_string()
                    } else {
                        v
                    }
                });
            raw.trim().parse::<u16>().unwrap_or(DEFAULT_WEBHOOK_PORT)
        };

        let mut webhook_path = extra_str(extra, "webhook_path").unwrap_or_else(|| {
            let v = env("BLUEBUBBLES_WEBHOOK_PATH");
            if v.is_empty() {
                DEFAULT_WEBHOOK_PATH.to_string()
            } else {
                v
            }
        });
        if !webhook_path.starts_with('/') {
            webhook_path = format!("/{webhook_path}");
        }

        // bool(extra.get("send_read_receipts", True)).
        let send_read_receipts = match extra.get("send_read_receipts") {
            None => true,
            Some(v) => value_truthy(v),
        };

        BlueBubblesConfig {
            server_url,
            password,
            webhook_host,
            webhook_port,
            webhook_path,
            send_read_receipts,
        }
    }

    /// True when both server URL and password are present — the connect()
    /// precondition.
    pub fn is_configured(&self) -> bool {
        !self.server_url.is_empty() && !self.password.is_empty()
    }

    /// Compute the external webhook URL for BlueBubbles registration.
    /// Mirrors the `_webhook_url` property.
    pub fn webhook_url(&self) -> String {
        let host = match self.webhook_host.as_str() {
            "0.0.0.0" | "127.0.0.1" | "localhost" | "::" => "localhost",
            other => other,
        };
        format!(
            "http://{}:{}{}",
            host, self.webhook_port, self.webhook_path
        )
    }

    /// Webhook URL registered with BlueBubbles, including the password as a
    /// query param. Mirrors the `_webhook_register_url` property.
    pub fn webhook_register_url(&self) -> String {
        let base = self.webhook_url();
        if !self.password.is_empty() {
            format!("{base}?password={}", quote(&self.password))
        } else {
            base
        }
    }
}

/// Coerce a JSON value to a `webhook_port` string candidate. Returns None for
/// falsy values (Python `extra.get("webhook_port") or ...`).
fn port_from_value(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Bool(false) => None,
        Value::Number(n) => {
            if n.as_i64() == Some(0) || n.as_f64() == Some(0.0) {
                None
            } else {
                Some(n.to_string())
            }
        }
        Value::String(s) if s.is_empty() => None,
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Python truthiness for a JSON value used by `bool(...)`.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

// ===========================================================================
// URL construction
// ===========================================================================

/// Percent-encode the way `urllib.parse.quote(value, safe='')` does — i.e. no
/// safe characters at all beyond the always-unreserved set
/// (`A-Z a-z 0-9 _ . - ~`). Notably `/` is encoded.
pub fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Build a credential-bearing API URL. Mirrors `_api_url`: appends
/// `password=<quoted>` as a query param, joining with `&` if `path` already has
/// a `?`, else `?`.
pub fn api_url(server_url: &str, path: &str, password: &str) -> String {
    let sep = if path.contains('?') { '&' } else { '?' };
    format!(
        "{server_url}{path}{sep}password={}",
        quote(password)
    )
}

/// Format outbound content for iMessage. Mirrors `format_message`: strips
/// markdown so plaintext bubbles render cleanly.
pub fn format_message(content: &str) -> String {
    crate::gw_helpers::strip_markdown(content)
}

// ===========================================================================
// Chat GUID / group detection
// ===========================================================================

/// True when `chat_id` looks like a group chat. Mirrors `";+;" in chat_id`.
pub fn is_group_chat(chat_id: &str) -> bool {
    chat_id.contains(";+;")
}

/// True when a raw target is already a GUID (`";" in target`).
pub fn is_raw_guid(target: &str) -> bool {
    target.contains(';')
}

/// True when a target looks like a creatable address (email or `+digits`).
/// Mirrors the `"@" in chat_id or re.match(r"^\+\d+", chat_id)` test used to
/// decide whether to create a new chat.
pub fn looks_like_address(chat_id: &str) -> bool {
    chat_id.contains('@') || PHONE_PREFIX_RE.with(|re| re.is_match(chat_id))
}

// ===========================================================================
// Text truncation + chunking
// ===========================================================================

/// Truncate `content`, stripping pagination indicators. Mirrors the static
/// `BlueBubblesAdapter.truncate_message`: it calls the base splitter then
/// removes the trailing `(n/m)` suffix from each chunk so iMessage bubbles flow
/// naturally.
pub fn truncate_message(content: &str, max_length: usize) -> Vec<String> {
    let chunks = crate::gw_platforms_base::truncate_message(content, max_length, None);
    PAGINATION_RE.with(|re| {
        chunks
            .into_iter()
            .map(|c| re.replace(&c, "").into_owned())
            .collect()
    })
}

/// Split a formatted message into iMessage bubbles. Mirrors the chunking logic
/// in `send`: split on paragraph breaks (double newlines), then truncate any
/// paragraph still longer than `max_length`.
pub fn split_send_chunks(text: &str, max_length: usize) -> Vec<String> {
    let paragraphs: Vec<String> = PARAGRAPH_RE.with(|re| {
        re.split(text)
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect()
    });

    let source: Vec<String> = if paragraphs.is_empty() {
        vec![text.to_string()]
    } else {
        paragraphs
    };

    let mut chunks: Vec<String> = Vec::new();
    for para in source {
        if para.chars().count() <= max_length {
            chunks.push(para);
        } else {
            chunks.extend(truncate_message(&para, max_length));
        }
    }
    chunks
}

// ===========================================================================
// Inbound attachment routing
// ===========================================================================

/// The cache bucket an attachment maps to, based on its MIME type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentKind {
    Image,
    Audio,
    Document,
}

/// Return `(kind, extension_or_filename_hint)` for an attachment MIME type.
///
/// For images and audio the second element is the file extension (incl. dot);
/// for documents it is the empty string (the caller supplies a transfer name).
/// Mirrors the ext-map logic inside `_download_attachment`.
pub fn attachment_ext_for_mime(mime: &str) -> (AttachmentKind, &'static str) {
    let mime = mime.to_lowercase();
    if mime.starts_with("image/") {
        let ext = match mime.as_str() {
            "image/jpeg" => ".jpg",
            "image/png" => ".png",
            "image/gif" => ".gif",
            "image/webp" => ".webp",
            "image/heic" => ".jpg",
            "image/heif" => ".jpg",
            "image/tiff" => ".jpg",
            _ => ".jpg",
        };
        return (AttachmentKind::Image, ext);
    }
    if mime.starts_with("audio/") {
        let ext = match mime.as_str() {
            "audio/mp3" => ".mp3",
            "audio/mpeg" => ".mp3",
            "audio/ogg" => ".ogg",
            "audio/wav" => ".wav",
            "audio/x-caf" => ".mp3",
            "audio/mp4" => ".m4a",
            "audio/aac" => ".m4a",
            _ => ".mp3",
        };
        return (AttachmentKind::Audio, ext);
    }
    (AttachmentKind::Document, "")
}

/// Classify the `MessageType` for a single inbound attachment, mirroring the
/// per-attachment branch inside `_handle_webhook`.
pub fn message_type_for_attachment(mime: &str, uti: &str) -> MessageType {
    let mime = mime.to_lowercase();
    if mime.starts_with("image/") {
        MessageType::Photo
    } else if mime.starts_with("audio/") || uti.ends_with("caf") {
        MessageType::Voice
    } else if mime.starts_with("video/") {
        MessageType::Video
    } else {
        MessageType::Document
    }
}

// ===========================================================================
// Webhook event gating
// ===========================================================================

/// True when an event type should be processed. Mirrors:
/// `if event_type and event_type not in _MESSAGE_EVENTS: ack`.
/// An empty event type is treated as a message event (Python only skips when
/// `event_type` is truthy AND not in the set).
pub fn is_message_event(event_type: &str) -> bool {
    event_type.is_empty() || MESSAGE_EVENTS.contains(&event_type)
}

// ===========================================================================
// Webhook payload helpers
// ===========================================================================

/// Extract the message record from a webhook payload. Mirrors
/// `_extract_payload_record`: prefer `data` (dict, or first dict in a list),
/// then `message` (dict), then the payload itself.
pub fn extract_payload_record(payload: &Value) -> Option<Value> {
    match payload.get("data") {
        Some(Value::Object(_)) => return payload.get("data").cloned(),
        Some(Value::Array(items)) => {
            for item in items {
                if item.is_object() {
                    return Some(item.clone());
                }
            }
        }
        _ => {}
    }
    if let Some(Value::Object(_)) = payload.get("message") {
        return payload.get("message").cloned();
    }
    if payload.is_object() {
        return Some(payload.clone());
    }
    None
}

/// Mirrors the static `_value(*candidates)`: return the first candidate that is
/// a non-blank string, trimmed.
fn first_str_value(candidates: &[Option<&Value>]) -> Option<String> {
    for c in candidates {
        if let Some(Value::String(s)) = c {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn get_str<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key)
}

// ===========================================================================
// Full webhook parsing
// ===========================================================================

/// The decision produced by parsing an inbound webhook body.
#[derive(Debug, Clone, PartialEq)]
pub enum WebhookOutcome {
    /// Authentication failed (401). Mirrors the unauthorized branch.
    Unauthorized,
    /// Body could not be parsed as JSON or form payload (400).
    InvalidPayload,
    /// Required message fields missing (400).
    MissingFields,
    /// Event acknowledged with no message dispatch (200 "ok"): non-message
    /// event, from-me message, or tapback reaction.
    Acknowledged,
    /// A user message was extracted and should be dispatched.
    Message(Box<ParsedMessage>),
}

/// A fully parsed inbound message, ready to be turned into a `MessageEvent`.
/// `media_urls` are left empty here because downloading attachments is an async
/// network side effect performed by the runtime; the caller fills them and the
/// derived `message_type` after downloading. `attachment_guids` lists the
/// `guid`/`mimeType`/`uti` triples to download, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedMessage {
    pub text: String,
    pub message_type: MessageType,
    pub chat_guid: Option<String>,
    pub chat_identifier: Option<String>,
    pub sender: String,
    pub session_chat_id: String,
    pub is_group: bool,
    pub message_id: Option<String>,
    pub reply_to_message_id: Option<String>,
    pub attachment_guids: Vec<AttachmentRef>,
    pub send_read_receipt: bool,
}

/// A reference to an attachment that must be downloaded from the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentRef {
    pub guid: String,
    pub mime: String,
    pub uti: String,
}

/// Authenticate the inbound webhook. Mirrors the token-extraction logic in
/// `_handle_webhook`: the first present of `?password`, `?guid`,
/// `x-password`, `x-guid`, `x-bluebubbles-guid` must equal the configured
/// password.
pub fn webhook_authorized(
    password: &str,
    query_password: Option<&str>,
    query_guid: Option<&str>,
    header_x_password: Option<&str>,
    header_x_guid: Option<&str>,
    header_x_bluebubbles_guid: Option<&str>,
) -> bool {
    let token = query_password
        .or(query_guid)
        .or(header_x_password)
        .or(header_x_guid)
        .or(header_x_bluebubbles_guid)
        .unwrap_or("");
    token == password
}

/// Parse a decoded JSON webhook payload into a [`WebhookOutcome`].
///
/// This reproduces the non-network portion of `_handle_webhook` after auth and
/// JSON decoding. Attachment downloading (which mutates `media_urls`,
/// `media_types`, and `msg_type`) is deferred to the caller via
/// [`ParsedMessage::attachment_guids`]; here `message_type` reflects only the
/// text-only baseline (`MessageType::Text`), and the caller updates it after a
/// successful download.
///
/// `send_read_receipts_cfg` is the adapter's configured flag (set on the
/// returned message for the fire-and-forget read receipt).
pub fn parse_webhook(payload: &Value, send_read_receipts_cfg: bool) -> WebhookOutcome {
    let event_type =
        first_str_value(&[payload.get("type"), payload.get("event")]).unwrap_or_default();
    if !is_message_event(&event_type) {
        return WebhookOutcome::Acknowledged;
    }

    let record = extract_payload_record(payload).unwrap_or_else(|| Value::Object(Default::default()));

    let is_from_me = value_truthy(record.get("isFromMe").unwrap_or(&Value::Null))
        || value_truthy(record.get("fromMe").unwrap_or(&Value::Null))
        || value_truthy(record.get("is_from_me").unwrap_or(&Value::Null));
    if is_from_me {
        return WebhookOutcome::Acknowledged;
    }

    // Skip tapback reactions delivered as messages.
    if let Some(Value::Number(n)) = record.get("associatedMessageType") {
        if let Some(code) = n.as_i64() {
            if is_tapback_type(code) {
                return WebhookOutcome::Acknowledged;
            }
        }
    }

    let mut text = first_str_value(&[
        record.get("text"),
        record.get("message"),
        record.get("body"),
    ])
    .unwrap_or_default();

    // --- Inbound attachment references (downloaded by the caller). ---
    let mut attachment_guids: Vec<AttachmentRef> = Vec::new();
    let mut msg_type = MessageType::Text;
    let mut has_image_mime = false;
    if let Some(Value::Array(atts)) = record.get("attachments") {
        for att in atts {
            let guid = att
                .get("guid")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if guid.is_empty() {
                continue;
            }
            let mime = att
                .get("mimeType")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();
            let uti = att
                .get("uti")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if mime.starts_with("image/") {
                has_image_mime = true;
            }
            // Last attachment wins for the single-type derivation (mirrors the
            // Python loop reassigning msg_type each iteration).
            msg_type = message_type_for_attachment(&mime, &uti);
            attachment_guids.push(AttachmentRef { guid, mime, uti });
        }
    }
    // With multiple attachments, prefer PHOTO if any images present.
    if attachment_guids.len() > 1 && has_image_mime {
        msg_type = MessageType::Photo;
    }
    if text.is_empty() && !attachment_guids.is_empty() {
        text = "(attachment)".to_string();
    }
    // --- End attachment handling ---

    let mut chat_guid = first_str_value(&[
        record.get("chatGuid"),
        payload.get("chatGuid"),
        record.get("chat_guid"),
        payload.get("chat_guid"),
        payload.get("guid"),
    ]);
    // Fallback: BB v1.9+ nests the chat GUID under data.chats[0].guid.
    if chat_guid.is_none() {
        if let Some(Value::Array(chats)) = record.get("chats") {
            if let Some(first) = chats.first() {
                if first.is_object() {
                    chat_guid = first_str_value(&[first.get("guid"), first.get("chatGuid")]);
                }
            }
        }
    }

    let mut chat_identifier = first_str_value(&[
        record.get("chatIdentifier"),
        record.get("identifier"),
        payload.get("chatIdentifier"),
        payload.get("identifier"),
    ]);

    let handle_addr = match record.get("handle") {
        Some(Value::Object(_)) => record.get("handle").and_then(|h| h.get("address")),
        _ => None,
    };
    let sender = first_str_value(&[
        handle_addr,
        record.get("sender"),
        record.get("from"),
        record.get("address"),
    ])
    .or_else(|| chat_identifier.clone())
    .or_else(|| chat_guid.clone());

    if chat_guid.is_none() && chat_identifier.is_none() {
        if let Some(s) = &sender {
            chat_identifier = Some(s.clone());
        }
    }

    let sender = match sender {
        Some(s) if !s.is_empty() => s,
        _ => return WebhookOutcome::MissingFields,
    };
    if chat_guid.is_none() && chat_identifier.is_none() {
        return WebhookOutcome::MissingFields;
    }
    if text.is_empty() {
        return WebhookOutcome::MissingFields;
    }

    let session_chat_id = chat_guid
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| chat_identifier.clone())
        .unwrap_or_default();

    let is_group = value_truthy(record.get("isGroup").unwrap_or(&Value::Null))
        || chat_guid.as_deref().map(is_group_chat).unwrap_or(false);

    let message_id = first_str_value(&[
        get_str(&record, "guid"),
        get_str(&record, "messageGuid"),
        get_str(&record, "id"),
    ]);
    let reply_to_message_id = first_str_value(&[
        get_str(&record, "threadOriginatorGuid"),
        get_str(&record, "associatedMessageGuid"),
    ]);

    let send_read_receipt = send_read_receipts_cfg && !session_chat_id.is_empty();

    WebhookOutcome::Message(Box::new(ParsedMessage {
        text,
        message_type: msg_type,
        chat_guid,
        chat_identifier,
        sender,
        session_chat_id,
        is_group,
        message_id,
        reply_to_message_id,
        attachment_guids,
        send_read_receipt,
    }))
}

impl ParsedMessage {
    /// Build the `SessionSource` for this message. Mirrors the `build_source`
    /// call in `_handle_webhook`.
    pub fn build_source(&self) -> SessionSource {
        SessionSource {
            platform: "bluebubbles".to_string(),
            chat_id: self.session_chat_id.clone(),
            chat_name: Some(
                self.chat_identifier
                    .clone()
                    .unwrap_or_else(|| self.sender.clone()),
            ),
            chat_type: if self.is_group { "group" } else { "dm" }.to_string(),
            user_id: Some(self.sender.clone()),
            user_name: Some(self.sender.clone()),
            chat_id_alt: self.chat_identifier.clone(),
            ..Default::default()
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redact_phone_and_email() {
        assert_eq!(redact("call +15551234567 now"), "call [REDACTED] now");
        assert_eq!(redact("mail a.b+c@example.com ok"), "mail [REDACTED] ok");
    }

    #[test]
    fn normalize_server_url_variants() {
        assert_eq!(normalize_server_url("  "), "");
        assert_eq!(normalize_server_url("localhost:1234"), "http://localhost:1234");
        assert_eq!(
            normalize_server_url("https://bb.example.com/"),
            "https://bb.example.com"
        );
        assert_eq!(
            normalize_server_url("HTTP://x.com///"),
            "HTTP://x.com"
        );
    }

    #[test]
    fn quote_encodes_slash_and_special() {
        // safe='' means / is encoded.
        assert_eq!(quote("a/b c"), "a%2Fb%20c");
        assert_eq!(quote("p@ss:w0rd"), "p%40ss%3Aw0rd");
        assert_eq!(quote("safe-_.~AZ09"), "safe-_.~AZ09");
    }

    #[test]
    fn api_url_appends_password() {
        assert_eq!(
            api_url("http://h:1", "/api/v1/ping", "pw/1"),
            "http://h:1/api/v1/ping?password=pw%2F1"
        );
        // existing query -> &
        assert_eq!(
            api_url("http://h:1", "/api/v1/chat/x?with=participants", "p"),
            "http://h:1/api/v1/chat/x?with=participants&password=p"
        );
    }

    #[test]
    fn config_defaults_from_empty_extra() {
        let extra = json!({"server_url": "bb.local:8080", "password": "secret"});
        let cfg = BlueBubblesConfig::from_extra(&extra);
        assert_eq!(cfg.server_url, "http://bb.local:8080");
        assert_eq!(cfg.password, "secret");
        assert_eq!(cfg.webhook_host, DEFAULT_WEBHOOK_HOST);
        assert_eq!(cfg.webhook_port, DEFAULT_WEBHOOK_PORT);
        assert_eq!(cfg.webhook_path, DEFAULT_WEBHOOK_PATH);
        assert!(cfg.send_read_receipts);
        assert!(cfg.is_configured());
    }

    #[test]
    fn config_webhook_path_gets_leading_slash() {
        let extra = json!({
            "server_url": "x", "password": "p",
            "webhook_path": "hook", "webhook_port": 9000,
            "send_read_receipts": false
        });
        let cfg = BlueBubblesConfig::from_extra(&extra);
        assert_eq!(cfg.webhook_path, "/hook");
        assert_eq!(cfg.webhook_port, 9000);
        assert!(!cfg.send_read_receipts);
    }

    #[test]
    fn config_port_as_string() {
        let extra = json!({"server_url": "x", "password": "p", "webhook_port": "7000"});
        let cfg = BlueBubblesConfig::from_extra(&extra);
        assert_eq!(cfg.webhook_port, 7000);
    }

    #[test]
    fn webhook_urls() {
        let cfg = BlueBubblesConfig {
            server_url: "http://x".into(),
            password: "p/w".into(),
            webhook_host: "127.0.0.1".into(),
            webhook_port: 8645,
            webhook_path: "/hook".into(),
            send_read_receipts: true,
        };
        assert_eq!(cfg.webhook_url(), "http://localhost:8645/hook");
        assert_eq!(
            cfg.webhook_register_url(),
            "http://localhost:8645/hook?password=p%2Fw"
        );
    }

    #[test]
    fn webhook_url_keeps_custom_host() {
        let cfg = BlueBubblesConfig {
            server_url: "http://x".into(),
            password: "".into(),
            webhook_host: "10.0.0.5".into(),
            webhook_port: 80,
            webhook_path: "/h".into(),
            send_read_receipts: true,
        };
        assert_eq!(cfg.webhook_url(), "http://10.0.0.5:80/h");
        // No password -> no query suffix.
        assert_eq!(cfg.webhook_register_url(), "http://10.0.0.5:80/h");
    }

    #[test]
    fn tapback_detection() {
        assert!(is_tapback_type(2000));
        assert!(is_tapback_type(2005));
        assert!(is_tapback_type(3003));
        assert!(!is_tapback_type(2006));
        assert!(!is_tapback_type(0));
    }

    #[test]
    fn group_and_address_detection() {
        assert!(is_group_chat("iMessage;+;chat123"));
        assert!(!is_group_chat("iMessage;-;user@x.com"));
        assert!(is_raw_guid("iMessage;-;user@x.com"));
        assert!(!is_raw_guid("user@x.com"));
        assert!(looks_like_address("user@x.com"));
        assert!(looks_like_address("+15551234567"));
        assert!(!looks_like_address("plainhandle"));
    }

    #[test]
    fn truncate_strips_pagination() {
        // A long message that the base splitter paginates; suffixes removed.
        let body = "word ".repeat(2000); // 10000 chars
        let chunks = truncate_message(&body, 100);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(!c.contains("/"), "chunk should not retain (n/m): {c}");
        }
    }

    #[test]
    fn split_send_paragraphs() {
        let text = "first para\n\nsecond para";
        let chunks = split_send_chunks(text, 4000);
        assert_eq!(chunks, vec!["first para".to_string(), "second para".to_string()]);
    }

    #[test]
    fn split_send_single_when_no_paragraphs() {
        let chunks = split_send_chunks("just one", 4000);
        assert_eq!(chunks, vec!["just one".to_string()]);
    }

    #[test]
    fn attachment_routing() {
        assert_eq!(
            attachment_ext_for_mime("image/heic"),
            (AttachmentKind::Image, ".jpg")
        );
        assert_eq!(
            attachment_ext_for_mime("image/png"),
            (AttachmentKind::Image, ".png")
        );
        assert_eq!(
            attachment_ext_for_mime("audio/x-caf"),
            (AttachmentKind::Audio, ".mp3")
        );
        assert_eq!(
            attachment_ext_for_mime("audio/aac"),
            (AttachmentKind::Audio, ".m4a")
        );
        assert_eq!(
            attachment_ext_for_mime("application/pdf"),
            (AttachmentKind::Document, "")
        );
        // unknown image/audio subtypes fall to defaults
        assert_eq!(
            attachment_ext_for_mime("image/x-weird"),
            (AttachmentKind::Image, ".jpg")
        );
        assert_eq!(
            attachment_ext_for_mime("audio/x-weird"),
            (AttachmentKind::Audio, ".mp3")
        );
    }

    #[test]
    fn message_type_routing() {
        assert_eq!(message_type_for_attachment("image/png", ""), MessageType::Photo);
        assert_eq!(message_type_for_attachment("audio/mp3", ""), MessageType::Voice);
        assert_eq!(message_type_for_attachment("", "public.caf"), MessageType::Voice);
        assert_eq!(message_type_for_attachment("video/mp4", ""), MessageType::Video);
        assert_eq!(
            message_type_for_attachment("application/zip", ""),
            MessageType::Document
        );
    }

    #[test]
    fn event_gating() {
        assert!(is_message_event("new-message"));
        assert!(is_message_event("updated-message"));
        assert!(is_message_event("")); // empty -> processed
        assert!(!is_message_event("typing-indicator"));
    }

    #[test]
    fn extract_record_prefers_data_dict() {
        let p = json!({"data": {"text": "hi"}});
        assert_eq!(extract_payload_record(&p), Some(json!({"text": "hi"})));
        // data list -> first dict
        let p2 = json!({"data": [1, {"text": "x"}]});
        assert_eq!(extract_payload_record(&p2), Some(json!({"text": "x"})));
        // fallback to message
        let p3 = json!({"message": {"text": "m"}});
        assert_eq!(extract_payload_record(&p3), Some(json!({"text": "m"})));
        // fallback to payload itself
        let p4 = json!({"text": "self"});
        assert_eq!(extract_payload_record(&p4), Some(p4.clone()));
    }

    #[test]
    fn webhook_auth() {
        assert!(webhook_authorized("pw", Some("pw"), None, None, None, None));
        assert!(webhook_authorized("pw", None, Some("pw"), None, None, None));
        assert!(webhook_authorized("pw", None, None, Some("pw"), None, None));
        assert!(!webhook_authorized("pw", Some("nope"), None, None, None, None));
        // empty token vs empty password
        assert!(webhook_authorized("", None, None, None, None, None));
    }

    #[test]
    fn parse_basic_text_message() {
        let payload = json!({
            "type": "new-message",
            "data": {
                "guid": "msg-1",
                "text": "hello there",
                "chatGuid": "iMessage;-;user@x.com",
                "handle": {"address": "user@x.com"}
            }
        });
        match parse_webhook(&payload, true) {
            WebhookOutcome::Message(m) => {
                assert_eq!(m.text, "hello there");
                assert_eq!(m.message_type, MessageType::Text);
                assert_eq!(m.chat_guid.as_deref(), Some("iMessage;-;user@x.com"));
                assert_eq!(m.sender, "user@x.com");
                assert_eq!(m.session_chat_id, "iMessage;-;user@x.com");
                assert!(!m.is_group);
                assert_eq!(m.message_id.as_deref(), Some("msg-1"));
                assert!(m.send_read_receipt);
                let src = m.build_source();
                assert_eq!(src.chat_type, "dm");
                assert_eq!(src.user_id.as_deref(), Some("user@x.com"));
            }
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn parse_skips_from_me() {
        let payload = json!({
            "type": "new-message",
            "data": {"text": "hi", "isFromMe": true, "chatGuid": "g", "handle": {"address": "a"}}
        });
        assert_eq!(parse_webhook(&payload, true), WebhookOutcome::Acknowledged);
    }

    #[test]
    fn parse_skips_tapback() {
        let payload = json!({
            "type": "new-message",
            "data": {"text": "Loved", "associatedMessageType": 2000, "chatGuid": "g"}
        });
        assert_eq!(parse_webhook(&payload, true), WebhookOutcome::Acknowledged);
    }

    #[test]
    fn parse_skips_non_message_event() {
        let payload = json!({"type": "typing-indicator", "data": {"text": "x"}});
        assert_eq!(parse_webhook(&payload, true), WebhookOutcome::Acknowledged);
    }

    #[test]
    fn parse_missing_fields() {
        // No sender / chat info.
        let payload = json!({"type": "new-message", "data": {"text": "hi"}});
        assert_eq!(parse_webhook(&payload, true), WebhookOutcome::MissingFields);
    }

    #[test]
    fn parse_group_with_attachment() {
        let payload = json!({
            "type": "new-message",
            "data": {
                "guid": "m2",
                "text": "",
                "chatGuid": "iMessage;+;chat42",
                "isGroup": true,
                "sender": "+15551112222",
                "attachments": [
                    {"guid": "att-1", "mimeType": "image/jpeg"},
                    {"guid": "att-2", "mimeType": "video/mp4"}
                ]
            }
        });
        match parse_webhook(&payload, false) {
            WebhookOutcome::Message(m) => {
                assert!(m.is_group);
                assert_eq!(m.text, "(attachment)");
                // multiple attachments with an image present -> PHOTO
                assert_eq!(m.message_type, MessageType::Photo);
                assert_eq!(m.attachment_guids.len(), 2);
                assert_eq!(m.attachment_guids[0].guid, "att-1");
                assert!(!m.send_read_receipt); // cfg false
                let src = m.build_source();
                assert_eq!(src.chat_type, "group");
            }
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn parse_chat_guid_fallback_to_chats() {
        let payload = json!({
            "type": "new-message",
            "data": {
                "text": "x",
                "sender": "user@x.com",
                "chats": [{"guid": "iMessage;-;user@x.com"}]
            }
        });
        match parse_webhook(&payload, true) {
            WebhookOutcome::Message(m) => {
                assert_eq!(m.chat_guid.as_deref(), Some("iMessage;-;user@x.com"));
            }
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn parse_sender_fallback_sets_identifier() {
        // Only a sender, no chat guid/identifier -> identifier := sender.
        let payload = json!({
            "type": "new-message",
            "data": {"text": "hey", "sender": "user@x.com"}
        });
        match parse_webhook(&payload, true) {
            WebhookOutcome::Message(m) => {
                assert_eq!(m.chat_identifier.as_deref(), Some("user@x.com"));
                assert_eq!(m.session_chat_id, "user@x.com");
                assert_eq!(m.sender, "user@x.com");
            }
            other => panic!("expected message, got {other:?}"),
        }
    }
}
