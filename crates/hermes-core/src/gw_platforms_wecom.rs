//! WeCom (Enterprise WeChat) platform adapter — native Rust port of
//! `gateway/platforms/wecom.py`.
//!
//! The Python original is an `asyncio`/`aiohttp` WebSocket adapter for the
//! WeCom AI Bot gateway. This port reproduces the *deterministic,
//! behavior-defining* logic faithfully and idiomatically:
//!
//!   - configuration parsing (`bot_id`/`secret`/`ws_url`, DM/group policies,
//!     allowlists, per-group config),
//!   - list coercion + allowlist-entry normalization / matching,
//!   - DM & group access policy decisions,
//!   - inbound text / quote extraction and media-reference extraction,
//!   - inbound message-type derivation,
//!   - base64 decode + image-extension sniffing + mime helpers,
//!   - outbound content-type normalization + WeCom media-type detection,
//!   - file-size limit application (reject / downgrade), including the exact
//!     Chinese user-facing notices,
//!   - WeCom AES-256-CBC + PKCS#7 inbound media decryption,
//!   - filename / extension guessing from URL / content-disposition,
//!   - request-body construction for `aibot_send_msg`,
//!     `aibot_respond_msg`, and the `aibot_upload_media_*` chunked upload,
//!   - text-batch merging (handling WeCom client-side 4000-char splits),
//!   - the WeCom QR scan credential flow (request/response shapes).
//!
//! The async task-orchestration machinery (websocket listen loop, heartbeat,
//! pending-response futures) is intimately tied to the CPython event loop and
//! is modelled here via plain data structures + synchronous helpers; the
//! actual coroutine scheduling lives in the async runtime layer that drives
//! this state. Network surfaces use `reqwest::blocking` and keep the API
//! request/response shapes identical to the Python original.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageType`, `SendResult`,
//!     image/document caching, `cache_image_from_bytes`,
//!     `cache_document_from_bytes`.
//!   - [`crate::tool_url_safety::is_safe_url`] — SSRF protection on downloads.

use std::collections::HashMap;
use std::path::Path;

use aes::cipher::{Array, BlockCipherDecrypt, KeyInit};
use aes::Aes256;
use base64::Engine;
use regex::Regex;
use serde_json::{json, Value};

use crate::gw_platforms_base::MessageType;

// ===========================================================================
// Constants (mirror module-level constants in wecom.py)
// ===========================================================================

pub const DEFAULT_WS_URL: &str = "wss://openws.work.weixin.qq.com";

pub const APP_CMD_SUBSCRIBE: &str = "aibot_subscribe";
pub const APP_CMD_CALLBACK: &str = "aibot_msg_callback";
pub const APP_CMD_LEGACY_CALLBACK: &str = "aibot_callback";
pub const APP_CMD_EVENT_CALLBACK: &str = "aibot_event_callback";
pub const APP_CMD_SEND: &str = "aibot_send_msg";
pub const APP_CMD_RESPONSE: &str = "aibot_respond_msg";
pub const APP_CMD_PING: &str = "ping";
pub const APP_CMD_UPLOAD_MEDIA_INIT: &str = "aibot_upload_media_init";
pub const APP_CMD_UPLOAD_MEDIA_CHUNK: &str = "aibot_upload_media_chunk";
pub const APP_CMD_UPLOAD_MEDIA_FINISH: &str = "aibot_upload_media_finish";

/// Commands that represent inbound message callbacks (`CALLBACK_COMMANDS`).
pub fn callback_commands() -> &'static [&'static str] {
    &[APP_CMD_CALLBACK, APP_CMD_LEGACY_CALLBACK]
}

/// Commands that must NOT be treated as correlated responses
/// (`NON_RESPONSE_COMMANDS`).
pub fn non_response_commands() -> &'static [&'static str] {
    &[APP_CMD_CALLBACK, APP_CMD_LEGACY_CALLBACK, APP_CMD_EVENT_CALLBACK]
}

pub const MAX_MESSAGE_LENGTH: usize = 4000;
pub const CONNECT_TIMEOUT_SECONDS: f64 = 20.0;
pub const REQUEST_TIMEOUT_SECONDS: f64 = 15.0;
pub const HEARTBEAT_INTERVAL_SECONDS: f64 = 30.0;
pub const RECONNECT_BACKOFF: &[u64] = &[2, 5, 10, 30, 60];

pub const DEDUP_MAX_SIZE: usize = 1000;

pub const IMAGE_MAX_BYTES: u64 = 10 * 1024 * 1024;
pub const VIDEO_MAX_BYTES: u64 = 10 * 1024 * 1024;
pub const VOICE_MAX_BYTES: u64 = 2 * 1024 * 1024;
pub const FILE_MAX_BYTES: u64 = 20 * 1024 * 1024;
pub const ABSOLUTE_MAX_BYTES: u64 = FILE_MAX_BYTES;
pub const UPLOAD_CHUNK_SIZE: usize = 512 * 1024;
pub const MAX_UPLOAD_CHUNKS: usize = 100;

/// MIME types WeCom natively supports for voice messages.
pub fn voice_supported_mimes() -> &'static [&'static str] {
    &["audio/amr"]
}

/// Threshold for detecting WeCom client-side message splits (`_SPLIT_THRESHOLD`).
pub const SPLIT_THRESHOLD: usize = 3900;

// ===========================================================================
// Config parsing helpers
// ===========================================================================

/// Coerce a JSON config value into a trimmed string list. Mirrors `_coerce_list`.
pub fn coerce_list(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => s
            .split(',')
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(|item| item.to_string())
            .collect(),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|item| value_to_str(item).trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        Some(other) => {
            let s = value_to_str(other).trim().to_string();
            if s.is_empty() {
                Vec::new()
            } else {
                vec![s]
            }
        }
    }
}

/// Stringify a JSON scalar the way Python's `str()` would for config values.
fn value_to_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => {
            // Python str(True) == "True".
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Normalize allowlist entries such as `wecom:user:foo`. Mirrors `_normalize_entry`.
pub fn normalize_entry(raw: &str) -> String {
    let value = raw.trim();
    let re_wecom = Regex::new(r"(?i)^wecom:").unwrap();
    let value = re_wecom.replace(value, "");
    let re_prefix = Regex::new(r"(?i)^(user|group):").unwrap();
    let value = re_prefix.replace(&value, "");
    value.trim().to_string()
}

/// Case-insensitive allowlist match with `*` support. Mirrors `_entry_matches`.
pub fn entry_matches(entries: &[String], target: &str) -> bool {
    let normalized_target = target.trim().to_lowercase();
    for entry in entries {
        let normalized = normalize_entry(entry).to_lowercase();
        if normalized == "*" || normalized == normalized_target {
            return true;
        }
    }
    false
}

// ===========================================================================
// Adapter configuration (mirrors the fields parsed in WeComAdapter.__init__)
// ===========================================================================

/// Parsed WeCom adapter configuration. Mirrors the constructor field parsing
/// in `WeComAdapter.__init__`. `extra` is the platform `config.extra` object;
/// environment variables are consulted as documented fallbacks.
#[derive(Debug, Clone)]
pub struct WeComConfig {
    pub bot_id: String,
    pub secret: String,
    pub ws_url: String,
    pub dm_policy: String,
    pub allow_from: Vec<String>,
    pub group_policy: String,
    pub group_allow_from: Vec<String>,
    /// Per-group config map; `groups` from `extra` when it is an object.
    pub groups: Value,
    pub text_batch_delay_seconds: f64,
    pub text_batch_split_delay_seconds: f64,
}

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.trim().to_string())
}

fn extra_str<'a>(extra: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let obj = extra.as_object()?;
    for k in keys {
        if let Some(Value::String(s)) = obj.get(*k) {
            return Some(s.as_str());
        }
    }
    None
}

fn extra_value<'a>(extra: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let obj = extra.as_object()?;
    for k in keys {
        if let Some(v) = obj.get(*k) {
            return Some(v);
        }
    }
    None
}

impl WeComConfig {
    /// Build a config from the platform `extra` object, consulting env vars as
    /// fallbacks exactly like `WeComAdapter.__init__`.
    pub fn from_extra(extra: &Value) -> Self {
        let bot_id = extra_str(extra, &["bot_id"])
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| env_str("WECOM_BOT_ID"))
            .unwrap_or_default()
            .trim()
            .to_string();

        let secret = extra_str(extra, &["secret"])
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| env_str("WECOM_SECRET"))
            .unwrap_or_default()
            .trim()
            .to_string();

        let ws_url = {
            let candidate = extra_str(extra, &["websocket_url", "websocketUrl"])
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    std::env::var("WECOM_WEBSOCKET_URL").ok()
                })
                .unwrap_or_else(|| DEFAULT_WS_URL.to_string());
            let trimmed = candidate.trim().to_string();
            if trimmed.is_empty() {
                DEFAULT_WS_URL.to_string()
            } else {
                trimmed
            }
        };

        let dm_policy = extra_str(extra, &["dm_policy"])
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| env_str("WECOM_DM_POLICY"))
            .unwrap_or_else(|| "open".to_string())
            .trim()
            .to_lowercase();

        let allow_from = coerce_list(extra_value(extra, &["allow_from", "allowFrom"]));

        let group_policy = extra_str(extra, &["group_policy"])
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| env_str("WECOM_GROUP_POLICY"))
            .unwrap_or_else(|| "open".to_string())
            .trim()
            .to_lowercase();

        let group_allow_from =
            coerce_list(extra_value(extra, &["group_allow_from", "groupAllowFrom"]));

        let groups = match extra.get("groups") {
            Some(v) if v.is_object() => v.clone(),
            _ => json!({}),
        };

        let text_batch_delay_seconds = env_str("HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.6);
        let text_batch_split_delay_seconds =
            env_str("HERMES_WECOM_TEXT_BATCH_SPLIT_DELAY_SECONDS")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(2.0);

        WeComConfig {
            bot_id,
            secret,
            ws_url,
            dm_policy,
            allow_from,
            group_policy,
            group_allow_from,
            groups,
            text_batch_delay_seconds,
            text_batch_split_delay_seconds,
        }
    }
}

// ===========================================================================
// Access policy
// ===========================================================================

/// Whether a DM from `sender_id` is allowed. Mirrors `_is_dm_allowed`.
pub fn is_dm_allowed(dm_policy: &str, allow_from: &[String], sender_id: &str) -> bool {
    if dm_policy == "disabled" {
        return false;
    }
    if dm_policy == "allowlist" {
        return entry_matches(allow_from, sender_id);
    }
    true
}

/// Resolve per-group config for a chat. Mirrors `_resolve_group_cfg`.
pub fn resolve_group_cfg(groups: &Value, chat_id: &str) -> Value {
    let obj = match groups.as_object() {
        Some(o) => o,
        None => return json!({}),
    };
    if let Some(v) = obj.get(chat_id) {
        if v.is_object() {
            return v.clone();
        }
    }
    let lowered = chat_id.to_lowercase();
    for (key, value) in obj {
        if key.to_lowercase() == lowered && value.is_object() {
            return value.clone();
        }
    }
    match obj.get("*") {
        Some(v) if v.is_object() => v.clone(),
        _ => json!({}),
    }
}

/// Whether a group message is allowed. Mirrors `_is_group_allowed`.
pub fn is_group_allowed(
    group_policy: &str,
    group_allow_from: &[String],
    groups: &Value,
    chat_id: &str,
    sender_id: &str,
) -> bool {
    if group_policy == "disabled" {
        return false;
    }
    if group_policy == "allowlist" && !entry_matches(group_allow_from, chat_id) {
        return false;
    }

    let group_cfg = resolve_group_cfg(groups, chat_id);
    let sender_allow = coerce_list(extra_value(&group_cfg, &["allow_from", "allowFrom"]));
    if !sender_allow.is_empty() {
        return entry_matches(&sender_allow, sender_id);
    }
    true
}

// ===========================================================================
// Inbound text / quote extraction
// ===========================================================================

fn lower_str(value: Option<&Value>) -> String {
    value.map(value_to_str).unwrap_or_default().to_lowercase()
}

fn obj_str(obj: &Value, key: &str) -> String {
    obj.get(key).map(value_to_str).unwrap_or_default()
}

/// Extract plain text and quoted text from a callback `body`.
/// Mirrors `_extract_text`. Returns `(text, reply_text)`.
pub fn extract_text(body: &Value) -> (String, Option<String>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut reply_text: Option<String> = None;
    let msgtype = lower_str(body.get("msgtype"));

    if msgtype == "mixed" {
        let mixed = body.get("mixed").filter(|v| v.is_object());
        let items = mixed
            .and_then(|m| m.get("msg_item"))
            .and_then(|v| v.as_array());
        if let Some(items) = items {
            for item in items {
                if !item.is_object() {
                    continue;
                }
                if lower_str(item.get("msgtype")) == "text" {
                    let content = item
                        .get("text")
                        .filter(|v| v.is_object())
                        .map(|tb| obj_str(tb, "content"))
                        .unwrap_or_default();
                    let content = content.trim();
                    if !content.is_empty() {
                        text_parts.push(content.to_string());
                    }
                }
            }
        }
    } else {
        let content = body
            .get("text")
            .filter(|v| v.is_object())
            .map(|tb| obj_str(tb, "content"))
            .unwrap_or_default();
        let content = content.trim();
        if !content.is_empty() {
            text_parts.push(content.to_string());
        }

        if msgtype == "voice" {
            let voice_text = body
                .get("voice")
                .filter(|v| v.is_object())
                .map(|vb| obj_str(vb, "content"))
                .unwrap_or_default();
            let voice_text = voice_text.trim();
            if !voice_text.is_empty() {
                text_parts.push(voice_text.to_string());
            }
        }

        if msgtype == "appmsg" {
            let title = body
                .get("appmsg")
                .filter(|v| v.is_object())
                .map(|a| obj_str(a, "title"))
                .unwrap_or_default();
            let title = title.trim();
            if !title.is_empty() {
                text_parts.push(title.to_string());
            }
        }
    }

    if let Some(quote) = body.get("quote").filter(|v| v.is_object()) {
        let quote_type = lower_str(quote.get("msgtype"));
        if quote_type == "text" {
            let c = quote
                .get("text")
                .filter(|v| v.is_object())
                .map(|t| obj_str(t, "content"))
                .unwrap_or_default();
            let c = c.trim();
            reply_text = if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            };
        } else if quote_type == "voice" {
            let c = quote
                .get("voice")
                .filter(|v| v.is_object())
                .map(|t| obj_str(t, "content"))
                .unwrap_or_default();
            let c = c.trim();
            reply_text = if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            };
        }
    }

    let joined: String = text_parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    (joined.trim().to_string(), reply_text)
}

/// A reference to an inbound media object: `(kind, ref_object)`.
/// `kind` is "image" or "file".
pub type MediaRef = (String, Value);

/// Collect inbound media references from a callback `body`.
/// Mirrors the reference-collection portion of `_extract_media`.
pub fn collect_media_refs(body: &Value) -> Vec<MediaRef> {
    let mut refs: Vec<MediaRef> = Vec::new();
    let msgtype = lower_str(body.get("msgtype"));

    if msgtype == "mixed" {
        let items = body
            .get("mixed")
            .filter(|v| v.is_object())
            .and_then(|m| m.get("msg_item"))
            .and_then(|v| v.as_array());
        if let Some(items) = items {
            for item in items {
                if !item.is_object() {
                    continue;
                }
                if lower_str(item.get("msgtype")) == "image" {
                    if let Some(img) = item.get("image").filter(|v| v.is_object()) {
                        refs.push(("image".to_string(), img.clone()));
                    }
                }
            }
        }
    } else {
        if let Some(img) = body.get("image").filter(|v| v.is_object()) {
            refs.push(("image".to_string(), img.clone()));
        }
        if msgtype == "file" {
            if let Some(file) = body.get("file").filter(|v| v.is_object()) {
                refs.push(("file".to_string(), file.clone()));
            }
        }
        if msgtype == "appmsg" {
            if let Some(appmsg) = body.get("appmsg").filter(|v| v.is_object()) {
                if let Some(file) = appmsg.get("file").filter(|v| v.is_object()) {
                    refs.push(("file".to_string(), file.clone()));
                } else if let Some(img) = appmsg.get("image").filter(|v| v.is_object()) {
                    refs.push(("image".to_string(), img.clone()));
                }
            }
        }
    }

    if let Some(quote) = body.get("quote").filter(|v| v.is_object()) {
        let quote_type = lower_str(quote.get("msgtype"));
        if quote_type == "image" {
            if let Some(img) = quote.get("image").filter(|v| v.is_object()) {
                refs.push(("image".to_string(), img.clone()));
            }
        } else if quote_type == "file" {
            if let Some(file) = quote.get("file").filter(|v| v.is_object()) {
                refs.push(("file".to_string(), file.clone()));
            }
        }
    }

    refs
}

/// Choose the normalized inbound message type. Mirrors `_derive_message_type`.
pub fn derive_message_type(body: &Value, text: &str, media_types: &[String]) -> MessageType {
    if media_types
        .iter()
        .any(|m| m.starts_with("application/") || m.starts_with("text/"))
    {
        return MessageType::Document;
    }
    if media_types.iter().any(|m| m.starts_with("image/")) {
        return if text.is_empty() {
            MessageType::Photo
        } else {
            MessageType::Text
        };
    }
    if lower_str(body.get("msgtype")) == "voice" {
        return MessageType::Voice;
    }
    MessageType::Text
}

/// Strip a leading `@mention` (group chat normalization). Mirrors the inline
/// `re.sub(r"^@\S+\s*", "", text)` performed on group messages.
pub fn strip_leading_mention(text: &str) -> String {
    let re = Regex::new(r"^@\S+\s*").unwrap();
    re.replace(text, "").trim().to_string()
}

// ===========================================================================
// base64 / image sniffing / mime helpers
// ===========================================================================

/// Decode a (possibly data-URI prefixed) base64 string. Mirrors `_decode_base64`.
pub fn decode_base64(data: &str) -> Result<Vec<u8>, String> {
    let payload = data.rsplit(',').next().unwrap_or(data).trim();
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| e.to_string())
}

/// Sniff an image file extension from magic bytes. Mirrors `_detect_image_ext`.
pub fn detect_image_ext(data: &[u8]) -> &'static str {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return ".png";
    }
    if data.starts_with(b"\xff\xd8\xff") {
        return ".jpg";
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return ".gif";
    }
    if data.starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"WEBP" {
        return ".webp";
    }
    ".jpg"
}

/// Mime type for a known file extension (subset of Python's `mimetypes.types_map`).
/// Mirrors `_mime_for_ext`.
pub fn mime_for_ext(ext: &str, fallback: &str) -> String {
    match ext.to_lowercase().as_str() {
        ".png" => "image/png".to_string(),
        ".jpg" | ".jpe" => "image/jpeg".to_string(),
        ".jpeg" => "image/jpeg".to_string(),
        ".gif" => "image/gif".to_string(),
        ".webp" => "image/webp".to_string(),
        ".bmp" => "image/bmp".to_string(),
        ".svg" => "image/svg+xml".to_string(),
        ".txt" => "text/plain".to_string(),
        ".html" | ".htm" => "text/html".to_string(),
        ".csv" => "text/csv".to_string(),
        ".pdf" => "application/pdf".to_string(),
        ".json" => "application/json".to_string(),
        ".xml" => "text/xml".to_string(),
        ".zip" => "application/zip".to_string(),
        ".mp4" => "video/mp4".to_string(),
        ".mov" => "video/quicktime".to_string(),
        ".mp3" => "audio/mpeg".to_string(),
        ".wav" => "audio/x-wav".to_string(),
        ".amr" => "audio/amr".to_string(),
        _ => fallback.to_string(),
    }
}

/// Guess a file extension from a content-type (best-effort `mimetypes.guess_extension`).
pub fn guess_extension_for_mime(content_type: &str) -> Option<&'static str> {
    match content_type.to_lowercase().as_str() {
        "image/png" => Some(".png"),
        "image/jpeg" => Some(".jpg"),
        "image/gif" => Some(".gif"),
        "image/webp" => Some(".webp"),
        "image/bmp" => Some(".bmp"),
        "text/plain" => Some(".txt"),
        "text/html" => Some(".html"),
        "text/csv" => Some(".csv"),
        "application/pdf" => Some(".pdf"),
        "application/json" => Some(".json"),
        "application/zip" => Some(".zip"),
        "video/mp4" => Some(".mp4"),
        "audio/mpeg" => Some(".mp3"),
        "audio/amr" => Some(".amr"),
        _ => None,
    }
}

/// Best-effort `mimetypes.guess_type` for outbound filenames.
pub fn guess_mime_type(filename: &str) -> String {
    let ext = Path::new(filename)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let guessed = mime_for_ext(&ext, "");
    if !guessed.is_empty() {
        return guessed;
    }
    if ext == ".amr" {
        return "audio/amr".to_string();
    }
    "application/octet-stream".to_string()
}

fn url_path(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        parsed.path().to_string()
    } else {
        // fall back to stripping scheme/query manually
        let no_q = url.split('?').next().unwrap_or(url);
        no_q.to_string()
    }
}

/// Guess an extension from URL/content-type with a fallback. Mirrors `_guess_extension`.
pub fn guess_extension(url: &str, content_type: &str, fallback: &str) -> String {
    if !content_type.is_empty() {
        if let Some(ext) = guess_extension_for_mime(content_type) {
            return ext.to_string();
        }
    }
    let path = url_path(url);
    if let Some(ext) = Path::new(&path).extension() {
        return format!(".{}", ext.to_string_lossy());
    }
    fallback.to_string()
}

/// Guess a filename from URL / content-disposition / content-type.
/// Mirrors `_guess_filename`.
pub fn guess_filename(url: &str, content_disposition: Option<&str>, content_type: &str) -> String {
    if let Some(cd) = content_disposition {
        let re = Regex::new(r#"filename="?([^";]+)"?"#).unwrap();
        if let Some(caps) = re.captures(cd) {
            if let Some(m) = caps.get(1) {
                return m.as_str().to_string();
            }
        }
    }
    let path = url_path(url);
    let mut name = Path::new(&path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if name.is_empty() {
        name = "document".to_string();
    }
    if !name.contains('.') {
        let ext = guess_extension_for_mime(content_type).unwrap_or(".bin");
        name = format!("{name}{ext}");
    }
    name
}

// ===========================================================================
// Outbound content-type / media-type detection + file-size limits
// ===========================================================================

/// Normalize a server-provided content-type, falling back to a filename guess.
/// Mirrors `_normalize_content_type`.
pub fn normalize_content_type(content_type: &str, filename: &str) -> String {
    let normalized = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let guessed = guess_mime_type(filename);
    if normalized.is_empty() {
        return guessed;
    }
    if normalized == "application/octet-stream" || normalized == "text/plain" {
        return guessed;
    }
    normalized
}

/// Map a MIME type to a WeCom media-type bucket. Mirrors `_detect_wecom_media_type`.
pub fn detect_wecom_media_type(content_type: &str) -> &'static str {
    let mime = content_type.trim().to_lowercase();
    if mime.starts_with("image/") {
        return "image";
    }
    if mime.starts_with("video/") {
        return "video";
    }
    if mime.starts_with("audio/") || mime == "application/ogg" {
        return "voice";
    }
    "file"
}

/// Result of applying WeCom file-size limits. Mirrors the dict returned by
/// `_apply_file_size_limits`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeCheck {
    pub final_type: String,
    pub rejected: bool,
    pub reject_reason: Option<String>,
    pub downgraded: bool,
    pub downgrade_note: Option<String>,
}

/// Format a byte count as `X.YZMB` the way Python's `f"{x:.2f}MB"` does.
fn fmt_mb(file_size: u64) -> String {
    let mb = file_size as f64 / (1024.0 * 1024.0);
    format!("{mb:.2}")
}

/// Apply WeCom size/type limits, reproducing the exact Chinese notices.
/// Mirrors `_apply_file_size_limits`.
pub fn apply_file_size_limits(
    file_size: u64,
    detected_type: &str,
    content_type: Option<&str>,
) -> SizeCheck {
    let normalized_type = if detected_type.is_empty() {
        "file".to_string()
    } else {
        detected_type.to_lowercase()
    };
    let normalized_content_type = content_type.unwrap_or("").trim().to_lowercase();
    let mb = fmt_mb(file_size);

    if file_size > ABSOLUTE_MAX_BYTES {
        return SizeCheck {
            final_type: normalized_type,
            rejected: true,
            reject_reason: Some(format!(
                "文件大小 {mb}MB 超过了企业微信允许的最大限制 20MB，无法发送。请尝试压缩文件或减小文件大小。"
            )),
            downgraded: false,
            downgrade_note: None,
        };
    }

    if normalized_type == "image" && file_size > IMAGE_MAX_BYTES {
        return SizeCheck {
            final_type: "file".to_string(),
            rejected: false,
            reject_reason: None,
            downgraded: true,
            downgrade_note: Some(format!("图片大小 {mb}MB 超过 10MB 限制，已转为文件格式发送")),
        };
    }

    if normalized_type == "video" && file_size > VIDEO_MAX_BYTES {
        return SizeCheck {
            final_type: "file".to_string(),
            rejected: false,
            reject_reason: None,
            downgraded: true,
            downgrade_note: Some(format!("视频大小 {mb}MB 超过 10MB 限制，已转为文件格式发送")),
        };
    }

    if normalized_type == "voice" {
        if !normalized_content_type.is_empty()
            && !voice_supported_mimes().contains(&normalized_content_type.as_str())
        {
            return SizeCheck {
                final_type: "file".to_string(),
                rejected: false,
                reject_reason: None,
                downgraded: true,
                downgrade_note: Some(format!(
                    "语音格式 {normalized_content_type} 不支持，企微仅支持 AMR 格式，已转为文件格式发送"
                )),
            };
        }
        if file_size > VOICE_MAX_BYTES {
            return SizeCheck {
                final_type: "file".to_string(),
                rejected: false,
                reject_reason: None,
                downgraded: true,
                downgrade_note: Some(format!("语音大小 {mb}MB 超过 2MB 限制，已转为文件格式发送")),
            };
        }
    }

    SizeCheck {
        final_type: normalized_type,
        rejected: false,
        reject_reason: None,
        downgraded: false,
        downgrade_note: None,
    }
}

// ===========================================================================
// Response error handling
// ===========================================================================

/// Extract a WeCom error string from a response, if `errcode` is non-zero.
/// Mirrors `_response_error`.
pub fn response_error(response: &Value) -> Option<String> {
    let errcode = response.get("errcode");
    let is_zero_or_null = match errcode {
        None | Some(Value::Null) => true,
        Some(Value::Number(n)) => n.as_i64() == Some(0) || n.as_f64() == Some(0.0),
        _ => false,
    };
    if is_zero_or_null {
        return None;
    }
    let errcode_str = errcode.map(value_to_str).unwrap_or_default();
    let errmsg = response
        .get("errmsg")
        .map(value_to_str)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown error".to_string());
    Some(format!("WeCom errcode {errcode_str}: {errmsg}"))
}

/// Raise (return Err) when a WeCom response carries an error. Mirrors `_raise_for_wecom_error`.
pub fn raise_for_wecom_error(response: &Value, operation: &str) -> Result<(), String> {
    match response_error(response) {
        Some(error) => Err(format!("{operation} failed: {error}")),
        None => Ok(()),
    }
}

// ===========================================================================
// WeCom AES-256-CBC + PKCS#7 inbound media decryption
// ===========================================================================

const BLOCK_SIZE: usize = 16;

/// Decrypt WeCom-encrypted media bytes. Mirrors `_decrypt_file_bytes`.
///
/// The `aes_key` is base64 (WeCom does not pad it); it is padded then decoded
/// into a 32-byte AES-256 key. AES-CBC uses the first 16 bytes of the key as
/// the IV, and the plaintext is PKCS#7 padded (pad length 1..=32).
pub fn decrypt_file_bytes(encrypted_data: &[u8], aes_key: &str) -> Result<Vec<u8>, String> {
    if encrypted_data.is_empty() {
        return Err("encrypted_data is empty".to_string());
    }
    if aes_key.is_empty() {
        return Err("aes_key is required".to_string());
    }

    // WeCom doesn't pad base64 keys; add padding if needed.
    let pad = (4 - aes_key.len() % 4) % 4;
    let padded_key = format!("{aes_key}{}", "=".repeat(pad));
    let key = base64::engine::general_purpose::STANDARD
        .decode(&padded_key)
        .map_err(|e| format!("invalid aes_key: {e}"))?;
    if key.len() != 32 {
        return Err(format!(
            "Invalid WeCom AES key length: expected 32 bytes, got {}",
            key.len()
        ));
    }

    if encrypted_data.len() % BLOCK_SIZE != 0 {
        return Err("encrypted_data is not block-aligned".to_string());
    }

    let mut key_arr = [0u8; 32];
    key_arr.copy_from_slice(&key);
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&key[..16]);

    let cipher = Aes256::new(&Array(key_arr));
    let mut decrypted = Vec::with_capacity(encrypted_data.len());
    let mut prev = iv;
    for chunk in encrypted_data.chunks(BLOCK_SIZE) {
        let mut ciph = [0u8; BLOCK_SIZE];
        ciph.copy_from_slice(chunk);
        let mut arr = Array(ciph);
        cipher.decrypt_block(&mut arr);
        for i in 0..BLOCK_SIZE {
            decrypted.push(arr.0[i] ^ prev[i]);
        }
        prev = ciph;
    }

    let pad_len = *decrypted.last().unwrap() as usize;
    if pad_len < 1 || pad_len > 32 || pad_len > decrypted.len() {
        return Err(format!("Invalid PKCS#7 padding value: {pad_len}"));
    }
    let tail = &decrypted[decrypted.len() - pad_len..];
    if tail.iter().any(|&b| b as usize != pad_len) {
        return Err("Invalid PKCS#7 padding: padding bytes mismatch".to_string());
    }
    Ok(decrypted[..decrypted.len() - pad_len].to_vec())
}

// ===========================================================================
// Request-body construction
// ===========================================================================

/// Build the body for a proactive markdown `aibot_send_msg`. Mirrors the body
/// constructed in `send` (proactive branch).
pub fn build_send_markdown_body(chat_id: &str, content: &str) -> Value {
    json!({
        "chatid": chat_id,
        "msgtype": "markdown",
        "markdown": {"content": truncate_to_max(content)},
    })
}

/// Build the body for a reply markdown `aibot_respond_msg`. Mirrors `_send_reply_markdown`.
pub fn build_reply_markdown_body(content: &str) -> Value {
    json!({
        "msgtype": "markdown",
        "markdown": {"content": truncate_to_max(content)},
    })
}

/// Build the body for a proactive media `aibot_send_msg`. Mirrors `_send_media_message`.
pub fn build_send_media_body(chat_id: &str, media_type: &str, media_id: &str) -> Value {
    json!({
        "chatid": chat_id,
        "msgtype": media_type,
        media_type: {"media_id": media_id},
    })
}

/// Build the body for a reply media message. Mirrors `_send_reply_media_message`.
pub fn build_reply_media_body(media_type: &str, media_id: &str) -> Value {
    json!({
        "msgtype": media_type,
        media_type: {"media_id": media_id},
    })
}

/// Build a full request frame `{cmd, headers: {req_id}, body}`. Mirrors `_send_request`.
pub fn build_request_frame(cmd: &str, req_id: &str, body: Value) -> Value {
    json!({
        "cmd": cmd,
        "headers": {"req_id": req_id},
        "body": body,
    })
}

/// Build the `aibot_subscribe` body. Mirrors `_open_connection`.
pub fn build_subscribe_body(bot_id: &str, secret: &str, device_id: &str) -> Value {
    json!({
        "bot_id": bot_id,
        "secret": secret,
        "device_id": device_id,
    })
}

/// Truncate a markdown string to `MAX_MESSAGE_LENGTH` codepoints (Python slicing).
pub fn truncate_to_max(content: &str) -> String {
    content.chars().take(MAX_MESSAGE_LENGTH).collect()
}

/// Build the `aibot_upload_media_init` body. Mirrors `_upload_media_bytes` init.
pub fn build_upload_init_body(
    media_type: &str,
    filename: &str,
    total_size: usize,
    total_chunks: usize,
    md5_hex: &str,
) -> Value {
    json!({
        "type": media_type,
        "filename": filename,
        "total_size": total_size,
        "total_chunks": total_chunks,
        "md5": md5_hex,
    })
}

/// Build the `aibot_upload_media_chunk` body. Mirrors `_upload_media_bytes` chunk.
pub fn build_upload_chunk_body(upload_id: &str, chunk_index: usize, chunk: &[u8]) -> Value {
    json!({
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "base64_data": base64::engine::general_purpose::STANDARD.encode(chunk),
    })
}

/// Build the `aibot_upload_media_finish` body. Mirrors `_upload_media_bytes` finish.
pub fn build_upload_finish_body(upload_id: &str) -> Value {
    json!({"upload_id": upload_id})
}

/// Compute total chunk count for a payload, mirroring the ceil-div in `_upload_media_bytes`.
/// Returns `Err` if the chunk count would exceed `MAX_UPLOAD_CHUNKS`.
pub fn upload_chunk_count(total_size: usize) -> Result<usize, String> {
    let total_chunks = total_size.div_ceil(UPLOAD_CHUNK_SIZE);
    if total_chunks > MAX_UPLOAD_CHUNKS {
        return Err(format!(
            "File too large: {total_chunks} chunks exceeds maximum of {MAX_UPLOAD_CHUNKS} chunks"
        ));
    }
    Ok(total_chunks)
}

/// MD5 hex digest used in the upload-init body.
pub fn md5_hex(data: &[u8]) -> String {
    let digest = md5::compute(data);
    format!("{digest:x}")
}

// ===========================================================================
// req_id / payload helpers
// ===========================================================================

/// Extract the `req_id` from a payload's `headers`. Mirrors `_payload_req_id`.
pub fn payload_req_id(payload: &Value) -> String {
    payload
        .get("headers")
        .filter(|v| v.is_object())
        .map(|h| obj_str(h, "req_id"))
        .unwrap_or_default()
}

/// Whether a `media_source` string looks like an http(s) URL. Mirrors `_looks_like_url`.
pub fn looks_like_url(media_source: &str) -> bool {
    if let Ok(parsed) = url::Url::parse(media_source.trim()) {
        let scheme = parsed.scheme();
        return scheme == "http" || scheme == "https";
    }
    false
}

/// Resolve a `reply_req_id` for a `reply_to` message id. Mirrors `_reply_req_id_for_message`.
pub fn reply_req_id_for_message(
    reply_req_ids: &HashMap<String, String>,
    reply_to: Option<&str>,
) -> Option<String> {
    let normalized = reply_to.unwrap_or("").trim();
    if normalized.is_empty() || normalized.starts_with("quote:") {
        return None;
    }
    reply_req_ids.get(normalized).cloned()
}

/// Insert into a bounded map (FIFO-ish eviction), mirroring `_remember_reply_req_id`
/// / `_remember_chat_req_id` bookkeeping. Both arguments must be non-empty after
/// trimming or the call is a no-op.
pub fn remember_bounded(
    map: &mut HashMap<String, String>,
    order: &mut Vec<String>,
    key: &str,
    value: &str,
    max_size: usize,
) {
    let key = key.trim();
    let value = value.trim();
    if key.is_empty() || value.is_empty() {
        return;
    }
    if !map.contains_key(key) {
        order.push(key.to_string());
    }
    map.insert(key.to_string(), value.to_string());
    while map.len() > max_size {
        if order.is_empty() {
            break;
        }
        let oldest = order.remove(0);
        map.remove(&oldest);
    }
}

// ===========================================================================
// Text-batch merging (handles WeCom client-side 4000-char splits)
// ===========================================================================

/// A buffered text event for batching. Mirrors the relevant `MessageEvent`
/// fields plus the dynamically-attached `_last_chunk_len`.
#[derive(Debug, Clone, Default)]
pub struct TextBatch {
    pub text: String,
    pub media_urls: Vec<String>,
    pub media_types: Vec<String>,
    pub last_chunk_len: usize,
}

/// Merge a new text chunk into an existing batch (or create one). Mirrors
/// `_enqueue_text_event` buffering logic. Returns the updated/created batch.
pub fn enqueue_text_batch(
    batches: &mut HashMap<String, TextBatch>,
    key: &str,
    text: &str,
    media_urls: &[String],
    media_types: &[String],
) {
    let chunk_len = text.chars().count();
    match batches.get_mut(key) {
        None => {
            batches.insert(
                key.to_string(),
                TextBatch {
                    text: text.to_string(),
                    media_urls: media_urls.to_vec(),
                    media_types: media_types.to_vec(),
                    last_chunk_len: chunk_len,
                },
            );
        }
        Some(existing) => {
            if !text.is_empty() {
                existing.text = if !existing.text.is_empty() {
                    format!("{}\n{}", existing.text, text)
                } else {
                    text.to_string()
                };
            }
            existing.last_chunk_len = chunk_len;
            if !media_urls.is_empty() {
                existing.media_urls.extend(media_urls.iter().cloned());
                existing.media_types.extend(media_types.iter().cloned());
            }
        }
    }
}

/// Compute the flush delay for a batch given its last chunk length.
/// Mirrors the delay selection in `_flush_text_batch`.
pub fn flush_delay_for(
    last_chunk_len: usize,
    text_batch_delay_seconds: f64,
    text_batch_split_delay_seconds: f64,
) -> f64 {
    if last_chunk_len >= SPLIT_THRESHOLD {
        text_batch_split_delay_seconds
    } else {
        text_batch_delay_seconds
    }
}

// ===========================================================================
// Remote media download (reqwest::blocking)
// ===========================================================================

/// Download remote bytes with a WeCom size cap and SSRF protection.
/// Mirrors `_download_remote_bytes`. Returns `(bytes, lowercased-headers)`.
pub fn download_remote_bytes(
    url: &str,
    max_bytes: u64,
) -> Result<(Vec<u8>, HashMap<String, String>), String> {
    if !crate::tool_url_safety::is_safe_url(url, None) {
        let snippet: String = url.chars().take(80).collect();
        return Err(format!("Blocked unsafe URL (SSRF protection): {snippet}"));
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get(url)
        .header("User-Agent", "HermesAgent/1.0")
        .header("Accept", "*/*")
        .send()
        .map_err(|e| e.to_string())?;
    let response = response.error_for_status().map_err(|e| e.to_string())?;

    let mut headers: HashMap<String, String> = HashMap::new();
    for (k, v) in response.headers().iter() {
        headers.insert(
            k.as_str().to_lowercase(),
            v.to_str().unwrap_or("").to_string(),
        );
    }

    if let Some(cl) = headers.get("content-length") {
        if cl.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = cl.parse::<u64>() {
                if n > max_bytes {
                    return Err(format!(
                        "Remote media exceeds WeCom limit: {n} bytes > {max_bytes} bytes"
                    ));
                }
            }
        }
    }

    let data = response.bytes().map_err(|e| e.to_string())?;
    if data.len() as u64 > max_bytes {
        return Err(format!(
            "Remote media exceeds WeCom limit while downloading: {} bytes > {} bytes",
            data.len(),
            max_bytes
        ));
    }

    Ok((data.to_vec(), headers))
}

// ===========================================================================
// QR-scan credential flow
// ===========================================================================

pub const QR_GENERATE_URL: &str = "https://work.weixin.qq.com/ai/qc/generate";
pub const QR_QUERY_URL: &str = "https://work.weixin.qq.com/ai/qc/query_result";
pub const QR_CODE_PAGE: &str = "https://work.weixin.qq.com/ai/qc/gen?source=hermes&scode=";
pub const QR_POLL_INTERVAL: u64 = 3;
pub const QR_POLL_TIMEOUT: u64 = 300;

/// Parsed result of a QR generate response. Mirrors the `data` extraction in
/// `qr_scan_for_bot_info` step 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrGenerate {
    pub scode: String,
    pub auth_url: String,
}

/// Parse the `/ai/qc/generate` JSON response. Returns `None` when scode/auth_url
/// are missing (mirrors the "unexpected response format" branch).
pub fn parse_qr_generate(raw: &Value) -> Option<QrGenerate> {
    let data = raw.get("data").filter(|v| v.is_object());
    let scode = data
        .map(|d| obj_str(d, "scode"))
        .unwrap_or_default()
        .trim()
        .to_string();
    let auth_url = data
        .map(|d| obj_str(d, "auth_url"))
        .unwrap_or_default()
        .trim()
        .to_string();
    if scode.is_empty() || auth_url.is_empty() {
        return None;
    }
    Some(QrGenerate { scode, auth_url })
}

/// Outcome of polling the QR query endpoint once. Mirrors the status handling
/// in `qr_scan_for_bot_info` step 3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QrPollOutcome {
    /// Success with valid credentials.
    Credentials { bot_id: String, secret: String },
    /// status == "success" but bot_info was missing/incomplete.
    SuccessButMissing,
    /// Not yet complete; keep polling.
    Pending,
}

/// Parse one `/ai/qc/query_result` JSON response. Mirrors the success/credential
/// extraction logic.
pub fn parse_qr_query(result: &Value) -> QrPollOutcome {
    let data = result.get("data").filter(|v| v.is_object());
    let status = data
        .map(|d| obj_str(d, "status"))
        .unwrap_or_default()
        .to_lowercase();
    if status != "success" {
        return QrPollOutcome::Pending;
    }
    let bot_info = data
        .and_then(|d| d.get("bot_info"))
        .filter(|v| v.is_object());
    let bot_id = bot_info
        .map(|b| {
            let v = obj_str(b, "botid");
            if v.trim().is_empty() {
                obj_str(b, "bot_id")
            } else {
                v
            }
        })
        .unwrap_or_default()
        .trim()
        .to_string();
    let secret = bot_info
        .map(|b| obj_str(b, "secret"))
        .unwrap_or_default()
        .trim()
        .to_string();
    if !bot_id.is_empty() && !secret.is_empty() {
        QrPollOutcome::Credentials { bot_id, secret }
    } else {
        QrPollOutcome::SuccessButMissing
    }
}

/// The QR-scan generate URL (with `source=hermes`).
pub fn qr_generate_url() -> String {
    format!("{QR_GENERATE_URL}?source=hermes")
}

/// The QR-scan query URL for a given `scode` (URL-encoded).
pub fn qr_query_url(scode: &str) -> String {
    let encoded = url_encode_component(scode);
    format!("{QR_QUERY_URL}?scode={encoded}")
}

/// The user-facing QR page URL for a given `scode` (URL-encoded).
pub fn qr_page_url(scode: &str) -> String {
    let encoded = url_encode_component(scode);
    format!("{QR_CODE_PAGE}{encoded}")
}

/// Percent-encode a URL query component the way `urllib.parse.quote` does (safe
/// chars: unreserved + `/`, but scodes are alnum so this is conservative).
fn url_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coerce_list_variants() {
        assert_eq!(coerce_list(None), Vec::<String>::new());
        assert_eq!(coerce_list(Some(&Value::Null)), Vec::<String>::new());
        assert_eq!(
            coerce_list(Some(&json!("a, b ,,c"))),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            coerce_list(Some(&json!(["x", " y ", ""]))),
            vec!["x".to_string(), "y".to_string()]
        );
        assert_eq!(coerce_list(Some(&json!(42))), vec!["42".to_string()]);
    }

    #[test]
    fn normalize_and_match_entries() {
        assert_eq!(normalize_entry("wecom:user:foo"), "foo");
        assert_eq!(normalize_entry("WeCom:Group:bar"), "bar");
        assert_eq!(normalize_entry("  baz  "), "baz");
        let entries = vec!["wecom:user:Alice".to_string(), "bob".to_string()];
        assert!(entry_matches(&entries, "alice"));
        assert!(entry_matches(&entries, "BOB"));
        assert!(!entry_matches(&entries, "carol"));
        assert!(entry_matches(&vec!["*".to_string()], "anyone"));
    }

    #[test]
    fn dm_policy_decisions() {
        let allow = vec!["alice".to_string()];
        assert!(is_dm_allowed("open", &allow, "anyone"));
        assert!(!is_dm_allowed("disabled", &allow, "alice"));
        assert!(is_dm_allowed("allowlist", &allow, "alice"));
        assert!(!is_dm_allowed("allowlist", &allow, "bob"));
    }

    #[test]
    fn group_policy_and_per_group_allow() {
        let groups = json!({
            "g1": {"allow_from": ["alice"]},
            "*": {"allow_from": ["wildcarduser"]},
        });
        // open policy, group has allow list -> only alice
        assert!(is_group_allowed("open", &[], &groups, "g1", "alice"));
        assert!(!is_group_allowed("open", &[], &groups, "g1", "bob"));
        // group not listed -> falls to wildcard config
        assert!(is_group_allowed("open", &[], &groups, "other", "wildcarduser"));
        assert!(!is_group_allowed("open", &[], &groups, "other", "stranger"));
        // disabled
        assert!(!is_group_allowed("disabled", &[], &groups, "g1", "alice"));
        // allowlist policy gating the chat id itself
        let allow = vec!["g1".to_string()];
        assert!(is_group_allowed("allowlist", &allow, &json!({}), "g1", "x"));
        assert!(!is_group_allowed("allowlist", &allow, &json!({}), "g2", "x"));
    }

    #[test]
    fn resolve_group_cfg_case_insensitive() {
        let groups = json!({"GroupABC": {"allow_from": ["a"]}});
        let cfg = resolve_group_cfg(&groups, "groupabc");
        assert!(cfg.get("allow_from").is_some());
        let empty = resolve_group_cfg(&groups, "nope");
        assert_eq!(empty, json!({}));
    }

    #[test]
    fn extract_text_plain_and_quote() {
        let body = json!({
            "msgtype": "text",
            "text": {"content": "  hello world  "},
            "quote": {"msgtype": "text", "text": {"content": "earlier"}},
        });
        let (text, reply) = extract_text(&body);
        assert_eq!(text, "hello world");
        assert_eq!(reply.as_deref(), Some("earlier"));
    }

    #[test]
    fn extract_text_mixed_and_appmsg() {
        let mixed = json!({
            "msgtype": "mixed",
            "mixed": {"msg_item": [
                {"msgtype": "text", "text": {"content": "part1"}},
                {"msgtype": "image", "image": {"url": "x"}},
                {"msgtype": "text", "text": {"content": "part2"}},
            ]},
        });
        let (text, _) = extract_text(&mixed);
        assert_eq!(text, "part1\npart2");

        let appmsg = json!({
            "msgtype": "appmsg",
            "appmsg": {"title": "report.pdf"},
        });
        let (text, _) = extract_text(&appmsg);
        assert_eq!(text, "report.pdf");
    }

    #[test]
    fn collect_refs_image_file_quote() {
        let body = json!({
            "msgtype": "file",
            "image": {"url": "i"},
            "file": {"url": "f"},
            "quote": {"msgtype": "image", "image": {"url": "q"}},
        });
        let refs = collect_media_refs(&body);
        let kinds: Vec<&str> = refs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(kinds, vec!["image", "file", "image"]);
    }

    #[test]
    fn message_type_derivation() {
        assert_eq!(
            derive_message_type(&json!({}), "", &["application/pdf".into()]),
            MessageType::Document
        );
        assert_eq!(
            derive_message_type(&json!({}), "", &["image/png".into()]),
            MessageType::Photo
        );
        assert_eq!(
            derive_message_type(&json!({}), "caption", &["image/png".into()]),
            MessageType::Text
        );
        assert_eq!(
            derive_message_type(&json!({"msgtype": "voice"}), "", &[]),
            MessageType::Voice
        );
        assert_eq!(derive_message_type(&json!({}), "", &[]), MessageType::Text);
    }

    #[test]
    fn strip_mention() {
        assert_eq!(strip_leading_mention("@BotName /approve"), "/approve");
        assert_eq!(strip_leading_mention("no mention"), "no mention");
    }

    #[test]
    fn base64_and_image_detection() {
        let data = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nrest");
        let with_prefix = format!("data:image/png;base64,{data}");
        let decoded = decode_base64(&with_prefix).unwrap();
        assert_eq!(detect_image_ext(&decoded), ".png");
        assert_eq!(detect_image_ext(b"\xff\xd8\xffrest"), ".jpg");
        assert_eq!(detect_image_ext(b"GIF89asomething"), ".gif");
        let mut webp = b"RIFF1234WEBPmore".to_vec();
        webp.extend_from_slice(b"x");
        assert_eq!(detect_image_ext(&webp), ".webp");
        assert_eq!(detect_image_ext(b"unknown"), ".jpg");
    }

    #[test]
    fn mime_helpers() {
        assert_eq!(mime_for_ext(".PNG", "fallback"), "image/png");
        assert_eq!(mime_for_ext(".unknown", "application/octet-stream"), "application/octet-stream");
        assert_eq!(guess_mime_type("a.amr"), "audio/amr");
        assert_eq!(guess_mime_type("a.weirdext"), "application/octet-stream");
    }

    #[test]
    fn guess_filename_from_disposition_and_url() {
        assert_eq!(
            guess_filename("https://x.com/p", Some("attachment; filename=\"report.pdf\""), ""),
            "report.pdf"
        );
        assert_eq!(
            guess_filename("https://x.com/path/doc.pdf", None, "application/pdf"),
            "doc.pdf"
        );
        // no extension in path -> append guessed ext
        assert_eq!(
            guess_filename("https://x.com/noext", None, "application/pdf"),
            "noext.pdf"
        );
    }

    #[test]
    fn content_type_normalization() {
        assert_eq!(normalize_content_type("application/pdf; charset=x", "a.pdf"), "application/pdf");
        // octet-stream falls back to guess
        assert_eq!(normalize_content_type("application/octet-stream", "a.png"), "image/png");
        // empty -> guess
        assert_eq!(normalize_content_type("", "a.mp4"), "video/mp4");
    }

    #[test]
    fn wecom_media_type_buckets() {
        assert_eq!(detect_wecom_media_type("image/png"), "image");
        assert_eq!(detect_wecom_media_type("video/mp4"), "video");
        assert_eq!(detect_wecom_media_type("audio/amr"), "voice");
        assert_eq!(detect_wecom_media_type("application/ogg"), "voice");
        assert_eq!(detect_wecom_media_type("application/pdf"), "file");
    }

    #[test]
    fn size_limit_reject_over_absolute() {
        let check = apply_file_size_limits(FILE_MAX_BYTES + 1, "file", None);
        assert!(check.rejected);
        assert!(check.reject_reason.unwrap().contains("20MB"));
    }

    #[test]
    fn size_limit_downgrade_image() {
        let check = apply_file_size_limits(IMAGE_MAX_BYTES + 1, "image", Some("image/png"));
        assert!(!check.rejected);
        assert!(check.downgraded);
        assert_eq!(check.final_type, "file");
        assert!(check.downgrade_note.unwrap().contains("图片大小"));
    }

    #[test]
    fn size_limit_voice_unsupported_mime() {
        let check = apply_file_size_limits(1000, "voice", Some("audio/mpeg"));
        assert!(check.downgraded);
        assert_eq!(check.final_type, "file");
        assert!(check.downgrade_note.unwrap().contains("AMR"));
    }

    #[test]
    fn size_limit_voice_amr_ok() {
        let check = apply_file_size_limits(1000, "voice", Some("audio/amr"));
        assert!(!check.downgraded);
        assert_eq!(check.final_type, "voice");
    }

    #[test]
    fn size_limit_clean_pass() {
        let check = apply_file_size_limits(100, "image", Some("image/png"));
        assert!(!check.rejected);
        assert!(!check.downgraded);
        assert_eq!(check.final_type, "image");
    }

    #[test]
    fn fmt_mb_two_decimals() {
        // 10MB + 1 byte
        let s = fmt_mb(IMAGE_MAX_BYTES + 1);
        assert!(s.starts_with("10.00"));
    }

    #[test]
    fn response_error_handling() {
        assert_eq!(response_error(&json!({"errcode": 0})), None);
        assert_eq!(response_error(&json!({})), None);
        assert_eq!(response_error(&json!({"errcode": null})), None);
        let err = response_error(&json!({"errcode": 40001, "errmsg": "bad token"})).unwrap();
        assert_eq!(err, "WeCom errcode 40001: bad token");
        let err2 = response_error(&json!({"errcode": 1})).unwrap();
        assert_eq!(err2, "WeCom errcode 1: unknown error");
    }

    #[test]
    fn raise_for_error_wraps_operation() {
        assert!(raise_for_wecom_error(&json!({"errcode": 0}), "op").is_ok());
        let e = raise_for_wecom_error(&json!({"errcode": 5, "errmsg": "nope"}), "send media").unwrap_err();
        assert_eq!(e, "send media failed: WeCom errcode 5: nope");
    }

    #[test]
    fn decrypt_round_trips_against_known_layout() {
        // Build a 32-byte key, derive iv, AES-CBC encrypt a PKCS#7 padded
        // plaintext, then verify decrypt_file_bytes recovers it.
        use aes::cipher::{Array, BlockCipherEncrypt, KeyInit};
        let key = [7u8; 32];
        let iv = {
            let mut iv = [0u8; 16];
            iv.copy_from_slice(&key[..16]);
            iv
        };
        let plaintext = b"hello wecom media payload".to_vec();
        // PKCS#7 pad to 16-byte blocks.
        let pad = 16 - (plaintext.len() % 16);
        let mut padded = plaintext.clone();
        padded.extend(std::iter::repeat(pad as u8).take(pad));

        let cipher = Aes256::new(&Array(key));
        let mut encrypted = Vec::new();
        let mut prev = iv;
        for chunk in padded.chunks(16) {
            let mut block = [0u8; 16];
            for i in 0..16 {
                block[i] = chunk[i] ^ prev[i];
            }
            let mut arr = Array(block);
            cipher.encrypt_block(&mut arr);
            prev = arr.0;
            encrypted.extend_from_slice(&arr.0);
        }

        // base64-encode the key without padding (WeCom style): 32 bytes -> 44
        // chars with one '=' of padding; strip it to exercise the re-pad path.
        let key_b64 = base64::engine::general_purpose::STANDARD.encode(key);
        let key_b64_nopad = key_b64.trim_end_matches('=');

        let recovered = decrypt_file_bytes(&encrypted, key_b64_nopad).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn decrypt_rejects_bad_inputs() {
        assert!(decrypt_file_bytes(&[], "key").is_err());
        assert!(decrypt_file_bytes(&[1, 2, 3], "").is_err());
        // wrong key length
        let short_key = base64::engine::general_purpose::STANDARD.encode([1u8; 16]);
        assert!(decrypt_file_bytes(&[0u8; 16], short_key.trim_end_matches('=')).is_err());
    }

    #[test]
    fn build_bodies_shape() {
        let send = build_send_markdown_body("c1", "hi");
        assert_eq!(send["chatid"], "c1");
        assert_eq!(send["msgtype"], "markdown");
        assert_eq!(send["markdown"]["content"], "hi");

        let media = build_send_media_body("c1", "image", "mid");
        assert_eq!(media["msgtype"], "image");
        assert_eq!(media["image"]["media_id"], "mid");

        let reply_media = build_reply_media_body("file", "mid2");
        assert_eq!(reply_media["msgtype"], "file");
        assert_eq!(reply_media["file"]["media_id"], "mid2");

        let frame = build_request_frame(APP_CMD_SEND, "r1", send);
        assert_eq!(frame["cmd"], APP_CMD_SEND);
        assert_eq!(frame["headers"]["req_id"], "r1");
    }

    #[test]
    fn truncate_respects_codepoints() {
        let long: String = "a".repeat(MAX_MESSAGE_LENGTH + 50);
        assert_eq!(truncate_to_max(&long).chars().count(), MAX_MESSAGE_LENGTH);
        assert_eq!(truncate_to_max("short"), "short");
    }

    #[test]
    fn upload_chunking_and_md5() {
        assert_eq!(upload_chunk_count(0), Ok(0));
        assert_eq!(upload_chunk_count(1), Ok(1));
        assert_eq!(upload_chunk_count(UPLOAD_CHUNK_SIZE), Ok(1));
        assert_eq!(upload_chunk_count(UPLOAD_CHUNK_SIZE + 1), Ok(2));
        // too many chunks
        assert!(upload_chunk_count(UPLOAD_CHUNK_SIZE * (MAX_UPLOAD_CHUNKS + 1)).is_err());

        let body = build_upload_init_body("image", "a.png", 100, 1, &md5_hex(b"abc"));
        assert_eq!(body["type"], "image");
        assert_eq!(body["total_chunks"], 1);
        // known md5 of "abc"
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");

        let chunk = build_upload_chunk_body("uid", 0, b"abc");
        assert_eq!(chunk["chunk_index"], 0);
        assert_eq!(
            chunk["base64_data"],
            base64::engine::general_purpose::STANDARD.encode(b"abc")
        );
    }

    #[test]
    fn payload_req_id_extraction() {
        assert_eq!(payload_req_id(&json!({"headers": {"req_id": "abc"}})), "abc");
        assert_eq!(payload_req_id(&json!({})), "");
        assert_eq!(payload_req_id(&json!({"headers": "notobj"})), "");
    }

    #[test]
    fn looks_like_url_detection() {
        assert!(looks_like_url("https://x.com/a"));
        assert!(looks_like_url("http://x.com"));
        assert!(!looks_like_url("/local/path"));
        assert!(!looks_like_url("ftp://x.com"));
    }

    #[test]
    fn reply_req_id_resolution() {
        let mut map = HashMap::new();
        map.insert("msg1".to_string(), "req-1".to_string());
        assert_eq!(
            reply_req_id_for_message(&map, Some("msg1")),
            Some("req-1".to_string())
        );
        assert_eq!(reply_req_id_for_message(&map, Some("quote:msg1")), None);
        assert_eq!(reply_req_id_for_message(&map, None), None);
        assert_eq!(reply_req_id_for_message(&map, Some("unknown")), None);
    }

    #[test]
    fn remember_bounded_evicts_oldest() {
        let mut map: HashMap<String, String> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for i in 0..5 {
            remember_bounded(&mut map, &mut order, &format!("k{i}"), &format!("v{i}"), 3);
        }
        assert_eq!(map.len(), 3);
        // oldest two evicted
        assert!(!map.contains_key("k0"));
        assert!(!map.contains_key("k1"));
        assert!(map.contains_key("k4"));
        // empty key/value no-ops
        remember_bounded(&mut map, &mut order, "", "x", 3);
        remember_bounded(&mut map, &mut order, "k", "", 3);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn text_batch_merge_and_delay() {
        let mut batches: HashMap<String, TextBatch> = HashMap::new();
        enqueue_text_batch(&mut batches, "k", "first", &[], &[]);
        enqueue_text_batch(&mut batches, "k", "second", &["u".into()], &["image/png".into()]);
        let b = &batches["k"];
        assert_eq!(b.text, "first\nsecond");
        assert_eq!(b.media_urls, vec!["u".to_string()]);
        assert_eq!(b.last_chunk_len, "second".chars().count());

        // near-split chunk uses the longer delay
        assert_eq!(flush_delay_for(SPLIT_THRESHOLD, 0.6, 2.0), 2.0);
        assert_eq!(flush_delay_for(SPLIT_THRESHOLD - 1, 0.6, 2.0), 0.6);
    }

    #[test]
    fn config_from_extra_defaults_and_env() {
        let extra = json!({
            "bot_id": "B1",
            "secret": "S1",
            "dm_policy": "Allowlist",
            "allow_from": ["u1", "u2"],
            "group_policy": "open",
            "groups": {"g1": {"allow_from": ["a"]}},
        });
        let cfg = WeComConfig::from_extra(&extra);
        assert_eq!(cfg.bot_id, "B1");
        assert_eq!(cfg.secret, "S1");
        assert_eq!(cfg.ws_url, DEFAULT_WS_URL);
        assert_eq!(cfg.dm_policy, "allowlist");
        assert_eq!(cfg.allow_from, vec!["u1".to_string(), "u2".to_string()]);
        assert_eq!(cfg.group_policy, "open");
        assert!(cfg.groups.get("g1").is_some());
        assert_eq!(cfg.text_batch_delay_seconds, 0.6);
    }

    #[test]
    fn config_env_fallback_for_credentials() {
        unsafe {
            std::env::set_var("WECOM_BOT_ID", "envbot");
            std::env::set_var("WECOM_SECRET", "envsecret");
        }
        let cfg = WeComConfig::from_extra(&json!({}));
        assert_eq!(cfg.bot_id, "envbot");
        assert_eq!(cfg.secret, "envsecret");
        unsafe {
            std::env::remove_var("WECOM_BOT_ID");
            std::env::remove_var("WECOM_SECRET");
        }
    }

    #[test]
    fn qr_generate_parsing() {
        let raw = json!({"data": {"scode": " abc ", "auth_url": " https://x "}});
        let parsed = parse_qr_generate(&raw).unwrap();
        assert_eq!(parsed.scode, "abc");
        assert_eq!(parsed.auth_url, "https://x");
        assert_eq!(parse_qr_generate(&json!({"data": {}})), None);
        assert_eq!(parse_qr_generate(&json!({})), None);
    }

    #[test]
    fn qr_query_parsing() {
        let pending = parse_qr_query(&json!({"data": {"status": "scanning"}}));
        assert_eq!(pending, QrPollOutcome::Pending);

        let creds = parse_qr_query(&json!({
            "data": {"status": "success", "bot_info": {"botid": "B", "secret": "S"}}
        }));
        assert_eq!(
            creds,
            QrPollOutcome::Credentials {
                bot_id: "B".to_string(),
                secret: "S".to_string()
            }
        );

        // fallback to bot_id key
        let creds2 = parse_qr_query(&json!({
            "data": {"status": "success", "bot_info": {"bot_id": "B2", "secret": "S2"}}
        }));
        assert_eq!(
            creds2,
            QrPollOutcome::Credentials {
                bot_id: "B2".to_string(),
                secret: "S2".to_string()
            }
        );

        let missing = parse_qr_query(&json!({"data": {"status": "success", "bot_info": {}}}));
        assert_eq!(missing, QrPollOutcome::SuccessButMissing);
    }

    #[test]
    fn qr_urls() {
        assert_eq!(qr_generate_url(), "https://work.weixin.qq.com/ai/qc/generate?source=hermes");
        assert!(qr_query_url("ab cd").contains("ab%20cd"));
        assert!(qr_page_url("xyz").ends_with("scode=xyz"));
    }
}
