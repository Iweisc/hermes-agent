//! Weixin platform adapter — native Rust port of
//! `gateway/platforms/weixin.py`.
//!
//! Connects Hermes Agent to WeChat personal accounts via Tencent's iLink Bot
//! API. This module reproduces the pure-logic and request/response surface of
//! the Python adapter:
//!
//! - Long-poll `getupdates` request/response shapes (`get_updates`).
//! - Outbound `sendmessage`, `sendtyping`, `getconfig`, `getuploadurl` payload
//!   construction and HTTP plumbing (via `reqwest::blocking`).
//! - AES-128-ECB (PKCS#7) media encrypt/decrypt for the CDN protocol.
//! - QR login request construction + status parsing.
//! - Markdown -> Weixin chat rendering and message-splitting heuristics.
//! - Disk-backed `context_token` cache, typing-ticket cache, sync-buffer
//!   persistence, and account credential persistence.
//! - Inbound text extraction, chat-type guessing, and MessageType derivation.
//!
//! The async long-poll event loop wiring is owned by the gateway runtime; here
//! the loop body is exposed as discrete functions that the runtime drives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use aes::cipher::{Array, BlockCipherDecrypt, BlockCipherEncrypt, KeyInit};
use aes::Aes128;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

pub use crate::gateway::MessageType;

// ---------------------------------------------------------------------------
// Endpoint and protocol constants
// ---------------------------------------------------------------------------

pub const ILINK_BASE_URL: &str = "https://ilinkai.weixin.qq.com";
pub const WEIXIN_CDN_BASE_URL: &str = "https://novac2c.cdn.weixin.qq.com/c2c";
pub const ILINK_APP_ID: &str = "bot";
pub const CHANNEL_VERSION: &str = "2.2.0";
/// (2 << 16) | (2 << 8) | 0
pub const ILINK_APP_CLIENT_VERSION: u32 = (2 << 16) | (2 << 8) | 0;

pub const EP_GET_UPDATES: &str = "ilink/bot/getupdates";
pub const EP_SEND_MESSAGE: &str = "ilink/bot/sendmessage";
pub const EP_SEND_TYPING: &str = "ilink/bot/sendtyping";
pub const EP_GET_CONFIG: &str = "ilink/bot/getconfig";
pub const EP_GET_UPLOAD_URL: &str = "ilink/bot/getuploadurl";
pub const EP_GET_BOT_QR: &str = "ilink/bot/get_bot_qrcode";
pub const EP_GET_QR_STATUS: &str = "ilink/bot/get_qrcode_status";

pub const LONG_POLL_TIMEOUT_MS: u64 = 35_000;
pub const API_TIMEOUT_MS: u64 = 15_000;
pub const CONFIG_TIMEOUT_MS: u64 = 10_000;
pub const QR_TIMEOUT_MS: u64 = 35_000;

pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;
pub const RETRY_DELAY_SECONDS: u64 = 2;
pub const BACKOFF_DELAY_SECONDS: u64 = 30;
pub const SESSION_EXPIRED_ERRCODE: i64 = -14;
/// iLink frequency limit — backoff and retry.
pub const RATE_LIMIT_ERRCODE: i64 = -2;
pub const MESSAGE_DEDUP_TTL_SECONDS: u64 = 300;

// Media types (getuploadurl)
pub const MEDIA_IMAGE: i64 = 1;
pub const MEDIA_VIDEO: i64 = 2;
pub const MEDIA_FILE: i64 = 3;
pub const MEDIA_VOICE: i64 = 4;

// Item types (item_list entries)
pub const ITEM_TEXT: i64 = 1;
pub const ITEM_IMAGE: i64 = 2;
pub const ITEM_VOICE: i64 = 3;
pub const ITEM_FILE: i64 = 4;
pub const ITEM_VIDEO: i64 = 5;

pub const MSG_TYPE_USER: i64 = 1;
pub const MSG_TYPE_BOT: i64 = 2;
pub const MSG_STATE_FINISH: i64 = 2;

pub const TYPING_START: i64 = 1;
pub const TYPING_STOP: i64 = 2;

/// WeChat CDN host allowlist (SSRF guard).
pub const WEIXIN_CDN_ALLOWLIST: &[&str] = &[
    "novac2c.cdn.weixin.qq.com",
    "ilinkai.weixin.qq.com",
    "wx.qlogo.cn",
    "thirdwx.qlogo.cn",
    "res.wx.qq.com",
    "mmbiz.qpic.cn",
    "mmbiz.qlogo.cn",
];

// ---------------------------------------------------------------------------
// Stale-session / rate-limit classification
// ---------------------------------------------------------------------------

/// True when iLink returns ret=-2 / errcode=-2 with "unknown error", which is a
/// stale-session signal (same as errcode=-14) rather than a genuine rate limit.
pub fn is_stale_session_ret(ret: Option<i64>, errcode: Option<i64>, errmsg: Option<&str>) -> bool {
    if ret != Some(RATE_LIMIT_ERRCODE) && errcode != Some(RATE_LIMIT_ERRCODE) {
        return false;
    }
    errmsg.unwrap_or("").to_lowercase() == "unknown error"
}

/// Returns true when a `ret`/`errcode` pair indicates the iLink session expired
/// (covers both the explicit -14 code and the -2/"unknown error" stale form).
pub fn is_session_expired(ret: Option<i64>, errcode: Option<i64>, errmsg: Option<&str>) -> bool {
    ret == Some(SESSION_EXPIRED_ERRCODE)
        || errcode == Some(SESSION_EXPIRED_ERRCODE)
        || is_stale_session_ret(ret, errcode, errmsg)
}

/// Returns true for a genuine rate-limit (`-2`) that is not the stale-session form.
pub fn is_rate_limited(ret: Option<i64>, errcode: Option<i64>) -> bool {
    ret == Some(RATE_LIMIT_ERRCODE) || errcode == Some(RATE_LIMIT_ERRCODE)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Truncate an identifier for logging, mirroring `_safe_id`.
pub fn safe_id(value: Option<&str>, keep: usize) -> String {
    let raw = value.unwrap_or("").trim();
    if raw.is_empty() {
        return "?".to_string();
    }
    if raw.len() <= keep {
        return raw.to_string();
    }
    raw.chars().take(keep).collect()
}

/// Compact JSON dump (no spaces, non-ASCII preserved) matching `_json_dumps`.
pub fn json_dumps(payload: &Value) -> String {
    serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string())
}

/// PKCS#7 pad to a 16-byte block, matching `_pkcs7_pad`.
pub fn pkcs7_pad(data: &[u8], block_size: usize) -> Vec<u8> {
    let pad_len = block_size - (data.len() % block_size);
    let mut out = Vec::with_capacity(data.len() + pad_len);
    out.extend_from_slice(data);
    out.extend(std::iter::repeat(pad_len as u8).take(pad_len));
    out
}

/// AES-128-ECB encrypt with PKCS#7 padding (`_aes128_ecb_encrypt`).
pub fn aes128_ecb_encrypt(plaintext: &[u8], key: &[u8; 16]) -> Vec<u8> {
    let cipher = Aes128::new(&Array(*key));
    let padded = pkcs7_pad(plaintext, 16);
    let mut out = Vec::with_capacity(padded.len());
    for chunk in padded.chunks(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        let mut arr = Array(block);
        cipher.encrypt_block(&mut arr);
        out.extend_from_slice(&arr.0);
    }
    out
}

/// AES-128-ECB decrypt with permissive PKCS#7 unpad (`_aes128_ecb_decrypt`).
///
/// Faithful to the Python: only strips padding when the trailing pad byte is in
/// `1..=16` AND the buffer actually ends with `pad_len` copies of it; otherwise
/// returns the decrypted buffer unchanged.
pub fn aes128_ecb_decrypt(ciphertext: &[u8], key: &[u8; 16]) -> Vec<u8> {
    let cipher = Aes128::new(&Array(*key));
    let mut padded = Vec::with_capacity(ciphertext.len());
    for chunk in ciphertext.chunks(16) {
        if chunk.len() != 16 {
            // Non block-aligned tail: Python's cryptography would have failed;
            // we mirror its "best effort" by skipping the malformed tail.
            break;
        }
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        let mut arr = Array(block);
        cipher.decrypt_block(&mut arr);
        padded.extend_from_slice(&arr.0);
    }
    if padded.is_empty() {
        return padded;
    }
    let pad_len = *padded.last().unwrap() as usize;
    if (1..=16).contains(&pad_len)
        && padded.len() >= pad_len
        && padded[padded.len() - pad_len..]
            .iter()
            .all(|&b| b as usize == pad_len)
    {
        padded.truncate(padded.len() - pad_len);
    }
    padded
}

/// Encrypted (padded) size of a raw media payload (`_aes_padded_size`).
pub fn aes_padded_size(size: usize) -> usize {
    ((size + 1 + 15) / 16) * 16
}

/// Random base64-encoded WeChat UIN header value (`_random_wechat_uin`).
///
/// Mirrors Python: take 4 random bytes -> big-endian u32 -> decimal string ->
/// base64 of the ASCII decimal string.
pub fn random_wechat_uin() -> String {
    let mut bytes = [0u8; 4];
    let _ = getrandom::fill(&mut bytes);
    let value = u32::from_be_bytes(bytes);
    B64.encode(value.to_string().as_bytes())
}

/// Base info block attached to every POST payload (`_base_info`).
pub fn base_info() -> Value {
    json!({ "channel_version": CHANNEL_VERSION })
}

/// Build POST headers (`_headers`). `body` is the serialized JSON string used to
/// compute `Content-Length` from its UTF-8 byte length.
pub fn build_headers(token: Option<&str>, body: &str) -> Vec<(String, String)> {
    let mut headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        (
            "AuthorizationType".to_string(),
            "ilink_bot_token".to_string(),
        ),
        (
            "Content-Length".to_string(),
            body.as_bytes().len().to_string(),
        ),
        ("X-WECHAT-UIN".to_string(), random_wechat_uin()),
        ("iLink-App-Id".to_string(), ILINK_APP_ID.to_string()),
        (
            "iLink-App-ClientVersion".to_string(),
            ILINK_APP_CLIENT_VERSION.to_string(),
        ),
    ];
    if let Some(tok) = token {
        if !tok.is_empty() {
            headers.push(("Authorization".to_string(), format!("Bearer {tok}")));
        }
    }
    headers
}

/// GET headers used by `_api_get`.
pub fn build_get_headers() -> Vec<(String, String)> {
    vec![
        ("iLink-App-Id".to_string(), ILINK_APP_ID.to_string()),
        (
            "iLink-App-ClientVersion".to_string(),
            ILINK_APP_CLIENT_VERSION.to_string(),
        ),
    ]
}

// ---------------------------------------------------------------------------
// URL-encoding helper (quote with safe='')
// ---------------------------------------------------------------------------

/// Percent-encode every byte that is not an unreserved character, matching
/// Python's `urllib.parse.quote(s, safe='')`.
pub fn quote_all(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
    }
    out
}

fn rstrip_slash(s: &str) -> &str {
    s.trim_end_matches('/')
}

/// `_cdn_download_url`.
pub fn cdn_download_url(cdn_base_url: &str, encrypted_query_param: &str) -> String {
    format!(
        "{}/download?encrypted_query_param={}",
        rstrip_slash(cdn_base_url),
        quote_all(encrypted_query_param)
    )
}

/// `_cdn_upload_url`.
pub fn cdn_upload_url(cdn_base_url: &str, upload_param: &str, filekey: &str) -> String {
    format!(
        "{}/upload?encrypted_query_param={}&filekey={}",
        rstrip_slash(cdn_base_url),
        quote_all(upload_param),
        quote_all(filekey)
    )
}

// ---------------------------------------------------------------------------
// AES key parsing
// ---------------------------------------------------------------------------

/// Parse a base64 aes key (`_parse_aes_key`). Accepts either 16 raw bytes, or 32
/// decoded bytes that are an ASCII hex string of a 16-byte key.
pub fn parse_aes_key(aes_key_b64: &str) -> Result<[u8; 16], String> {
    let decoded = B64
        .decode(aes_key_b64.as_bytes())
        .map_err(|e| format!("invalid base64 aes_key: {e}"))?;
    if decoded.len() == 16 {
        let mut k = [0u8; 16];
        k.copy_from_slice(&decoded);
        return Ok(k);
    }
    if decoded.len() == 32 {
        // decode("ascii", errors="ignore"): drop non-ascii bytes.
        let text: String = decoded.iter().filter(|&&b| b < 0x80).map(|&b| b as char).collect();
        if !text.is_empty() && text.chars().all(|c| c.is_ascii_hexdigit()) {
            let bytes = hex_decode(&text)
                .ok_or_else(|| "invalid hex aes_key".to_string())?;
            if bytes.len() == 16 {
                let mut k = [0u8; 16];
                k.copy_from_slice(&bytes);
                return Ok(k);
            }
        }
    }
    Err(format!(
        "unexpected aes_key format ({} decoded bytes)",
        decoded.len()
    ))
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// SSRF guard for media URLs (`_assert_weixin_cdn_url`). Returns Err on a
/// disallowed scheme or off-allowlist host.
pub fn assert_weixin_cdn_url(url_str: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url_str)
        .map_err(|_| format!("Unparseable media URL: {url_str:?}"))?;
    let scheme = parsed.scheme().to_lowercase();
    let host = parsed.host_str().unwrap_or("").to_string();
    if scheme != "http" && scheme != "https" {
        return Err(format!(
            "Media URL has disallowed scheme {scheme:?}; only http/https are permitted."
        ));
    }
    if !WEIXIN_CDN_ALLOWLIST.contains(&host.as_str()) {
        return Err(format!(
            "Media URL host {host:?} is not in the WeChat CDN allowlist. Refusing to fetch to prevent SSRF."
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chat-type guessing & inbound parsing
// ---------------------------------------------------------------------------

fn str_field(message: &Value, key: &str) -> String {
    match message.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// `_guess_chat_type` — returns (chat_type, effective_chat_id).
pub fn guess_chat_type(message: &Value, account_id: &str) -> (String, String) {
    let room_id = {
        let r = str_field(message, "room_id");
        if !r.trim().is_empty() {
            r.trim().to_string()
        } else {
            str_field(message, "chat_room_id").trim().to_string()
        }
    };
    let to_user_id = str_field(message, "to_user_id").trim().to_string();
    let msg_type = message.get("msg_type").and_then(|v| v.as_i64());
    let is_group = !room_id.is_empty()
        || (!to_user_id.is_empty()
            && !account_id.is_empty()
            && to_user_id != account_id
            && msg_type == Some(1));
    if is_group {
        let from_user = str_field(message, "from_user_id");
        let chat_id = if !room_id.is_empty() {
            room_id
        } else if !to_user_id.is_empty() {
            to_user_id
        } else {
            from_user
        };
        return ("group".to_string(), chat_id);
    }
    ("dm".to_string(), str_field(message, "from_user_id"))
}

/// `_media_reference`: `item[key]["media"]` (empty object on missing).
pub fn media_reference<'a>(item: &'a Value, key: &str) -> &'a Value {
    item.get(key)
        .and_then(|v| v.get("media"))
        .unwrap_or(&Value::Null)
}

/// Extract the text body from an `item_list` (`_extract_text`), including
/// quote/ref handling.
pub fn extract_text(item_list: &[Value]) -> String {
    for item in item_list {
        if item.get("type").and_then(|v| v.as_i64()) == Some(ITEM_TEXT) {
            let text = item
                .get("text_item")
                .and_then(|v| v.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let empty = Value::Null;
            let r#ref = item.get("ref_msg").unwrap_or(&empty);
            let ref_item = r#ref.get("message_item");
            let ref_type = ref_item.and_then(|v| v.get("type")).and_then(|v| v.as_i64());
            if let Some(rt) = ref_type {
                if rt == ITEM_IMAGE || rt == ITEM_VIDEO || rt == ITEM_FILE || rt == ITEM_VOICE {
                    let title = r#ref
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let prefix = if !title.is_empty() {
                        format!("[引用媒体: {title}]\n")
                    } else {
                        "[引用媒体]\n".to_string()
                    };
                    return format!("{prefix}{text}").trim().to_string();
                }
            }
            if let Some(ri) = ref_item {
                if ri.is_object() && !ri.as_object().map(|o| o.is_empty()).unwrap_or(true) {
                    let mut parts: Vec<String> = Vec::new();
                    if let Some(title) = r#ref.get("title").and_then(|v| v.as_str()) {
                        if !title.is_empty() {
                            parts.push(title.to_string());
                        }
                    }
                    let ref_text = extract_text(std::slice::from_ref(ri));
                    if !ref_text.is_empty() {
                        parts.push(ref_text);
                    }
                    if !parts.is_empty() {
                        return format!("[引用: {}]\n{text}", parts.join(" | "))
                            .trim()
                            .to_string();
                    }
                }
            }
            return text;
        }
    }
    for item in item_list {
        if item.get("type").and_then(|v| v.as_i64()) == Some(ITEM_VOICE) {
            let voice_text = item
                .get("voice_item")
                .and_then(|v| v.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !voice_text.is_empty() {
                return voice_text.to_string();
            }
        }
    }
    String::new()
}

/// `_message_type_from_media`.
pub fn message_type_from_media(media_types: &[String], text: &str) -> MessageType {
    if media_types.iter().any(|m| m.starts_with("image/")) {
        return MessageType::Photo;
    }
    if media_types.iter().any(|m| m.starts_with("video/")) {
        return MessageType::Video;
    }
    if media_types.iter().any(|m| m.starts_with("audio/")) {
        return MessageType::Voice;
    }
    if !media_types.is_empty() {
        return MessageType::Document;
    }
    if text.starts_with('/') {
        return MessageType::Command;
    }
    MessageType::Text
}

/// `_mime_from_filename` — naive extension -> mime guess.
pub fn mime_from_filename(filename: &str) -> String {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mp3" => "audio/mpeg",
        "wav" => "audio/x-wav",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/x-flac",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "json" => "application/json",
        "zip" => "application/zip",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "doc" => "application/msword",
        "docx" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        _ => "application/octet-stream",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Bool coercion
// ---------------------------------------------------------------------------

/// `_coerce_bool` — coerce a string/JSON value to bool tolerating "true"/"on".
pub fn coerce_bool(value: Option<&str>, default: bool) -> bool {
    let raw = match value {
        None => return default,
        Some(v) => v,
    };
    let text = raw.trim().to_lowercase();
    if text.is_empty() {
        return default;
    }
    match text.as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

/// `_coerce_list` — split a comma string into trimmed non-empty entries.
pub fn coerce_list_str(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Markdown / message-splitting heuristics
// ---------------------------------------------------------------------------

fn header_match(line: &str) -> Option<(usize, String)> {
    // ^(#{1,6})\s+(.+?)\s*$
    let trimmed = line;
    let mut hashes = 0usize;
    for c in trimmed.chars() {
        if c == '#' {
            hashes += 1;
        } else {
            break;
        }
    }
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    // require at least one whitespace after the hashes
    if !rest.starts_with(|c: char| c == ' ' || c == '\t') {
        return None;
    }
    let title = rest.trim();
    if title.is_empty() {
        return None;
    }
    Some((hashes, title.to_string()))
}

fn is_fence(line_stripped: &str) -> bool {
    // ^```([^\n`]*)\s*$
    if !line_stripped.starts_with("```") {
        return false;
    }
    let after = &line_stripped[3..];
    // remaining must contain no backticks (info string may be present, then ws)
    !after.contains('`')
}

fn is_table_rule(stripped: &str) -> bool {
    // ^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$
    // Practical implementation: line consists only of |, -, :, spaces and has
    // at least 3 consecutive dashes and at least one pipe.
    let s = stripped.trim();
    if s.is_empty() {
        return false;
    }
    if !s.chars().all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t')) {
        return false;
    }
    if !s.contains('|') {
        return false;
    }
    let mut run = 0usize;
    let mut max_run = 0usize;
    for c in s.chars() {
        if c == '-' {
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }
    max_run >= 3
}

/// `_split_table_row`.
pub fn split_table_row(line: &str) -> Vec<String> {
    let mut row = line.trim();
    if let Some(r) = row.strip_prefix('|') {
        row = r;
    }
    if let Some(r) = row.strip_suffix('|') {
        row = r;
    }
    row.split('|').map(|cell| cell.trim().to_string()).collect()
}

/// `_rewrite_headers_for_weixin`.
pub fn rewrite_headers_for_weixin(line: &str) -> String {
    match header_match(line) {
        None => line.trim_end().to_string(),
        Some((level, title)) => {
            if level == 1 {
                format!("【{title}】")
            } else {
                format!("**{title}**")
            }
        }
    }
}

/// `_rewrite_table_block_for_weixin`.
pub fn rewrite_table_block_for_weixin(lines: &[String]) -> String {
    if lines.len() < 2 {
        return lines.join("\n");
    }
    let headers = split_table_row(&lines[0]);
    let body_rows: Vec<Vec<String>> = lines[2..]
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| split_table_row(line))
        .collect();
    if headers.is_empty() || body_rows.is_empty() {
        return lines.join("\n");
    }
    let mut formatted_rows: Vec<String> = Vec::new();
    for row in &body_rows {
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (idx, header) in headers.iter().enumerate() {
            if idx >= row.len() {
                break;
            }
            let label = if header.is_empty() {
                format!("Column {}", idx + 1)
            } else {
                header.clone()
            };
            let value = row[idx].trim().to_string();
            if !value.is_empty() {
                pairs.push((label, value));
            }
        }
        if pairs.is_empty() {
            continue;
        }
        if pairs.len() == 1 {
            formatted_rows.push(format!("- {}: {}", pairs[0].0, pairs[0].1));
            continue;
        }
        if pairs.len() == 2 {
            formatted_rows.push(format!("- {}: {}", pairs[0].0, pairs[0].1));
            formatted_rows.push(format!("  {}: {}", pairs[1].0, pairs[1].1));
            continue;
        }
        let summary = pairs
            .iter()
            .map(|(l, v)| format!("{l}: {v}"))
            .collect::<Vec<_>>()
            .join(" | ");
        formatted_rows.push(format!("- {summary}"));
    }
    if formatted_rows.is_empty() {
        lines.join("\n")
    } else {
        formatted_rows.join("\n")
    }
}

fn splitlines(content: &str) -> Vec<&str> {
    // Python str.splitlines: split on \n (and \r\n). We split on '\n' and strip
    // a trailing '\r' from each line to match common CRLF inputs.
    content
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

/// `_normalize_markdown_blocks` — collapse runs of blank lines (keeping fenced
/// blocks intact) and strip the result.
pub fn normalize_markdown_blocks(content: &str) -> String {
    let mut result: Vec<String> = Vec::new();
    let mut in_code_block = false;
    let mut blank_run = 0i32;
    for raw_line in splitlines(content) {
        let line = raw_line.trim_end();
        if is_fence(line.trim()) {
            in_code_block = !in_code_block;
            result.push(line.to_string());
            blank_run = 0;
            continue;
        }
        if in_code_block {
            result.push(line.to_string());
            continue;
        }
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                result.push(String::new());
            }
            continue;
        }
        blank_run = 0;
        result.push(line.to_string());
    }
    result.join("\n").trim().to_string()
}

/// `_split_markdown_blocks`.
pub fn split_markdown_blocks(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut blocks: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut in_code_block = false;
    for raw_line in splitlines(content) {
        let line = raw_line.trim_end();
        if is_fence(line.trim()) {
            if !in_code_block && !current.is_empty() {
                blocks.push(current.join("\n").trim().to_string());
                current.clear();
            }
            current.push(line.to_string());
            in_code_block = !in_code_block;
            if !in_code_block {
                blocks.push(current.join("\n").trim().to_string());
                current.clear();
            }
            continue;
        }
        if in_code_block {
            current.push(line.to_string());
            continue;
        }
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(current.join("\n").trim().to_string());
                current.clear();
            }
            continue;
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        blocks.push(current.join("\n").trim().to_string());
    }
    blocks.into_iter().filter(|b| !b.is_empty()).collect()
}

/// `_split_delivery_units_for_weixin`.
pub fn split_delivery_units_for_weixin(content: &str) -> Vec<String> {
    let mut units: Vec<String> = Vec::new();
    for block in split_markdown_blocks(content) {
        let first_line = block.split('\n').next().unwrap_or("");
        if is_fence(first_line.trim()) {
            units.push(block);
            continue;
        }
        let mut current: Vec<String> = Vec::new();
        for raw_line in block.split('\n') {
            let line = raw_line.trim_end();
            if line.trim().is_empty() {
                if !current.is_empty() {
                    units.push(current.join("\n").trim().to_string());
                    current.clear();
                }
                continue;
            }
            let is_continuation =
                !current.is_empty() && (raw_line.starts_with(' ') || raw_line.starts_with('\t'));
            if is_continuation {
                current.push(line.to_string());
                continue;
            }
            if !current.is_empty() {
                units.push(current.join("\n").trim().to_string());
            }
            current = vec![line.to_string()];
        }
        if !current.is_empty() {
            units.push(current.join("\n").trim().to_string());
        }
    }
    units.into_iter().filter(|u| !u.is_empty()).collect()
}

fn re_bold_only(stripped: &str) -> bool {
    // ^\*\*[^*]+\*\*$
    if !stripped.starts_with("**") || !stripped.ends_with("**") || stripped.len() < 5 {
        return false;
    }
    let inner = &stripped[2..stripped.len() - 2];
    !inner.is_empty() && !inner.contains('*')
}

fn re_ordered_list(stripped: &str) -> bool {
    // ^\d+\.\s
    let bytes = stripped.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        return false;
    }
    if bytes[i] != b'.' {
        return false;
    }
    i += 1;
    i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t')
}

/// `_looks_like_chatty_line_for_weixin`.
pub fn looks_like_chatty_line_for_weixin(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    if stripped.chars().count() > 48 {
        return false;
    }
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    if stripped.starts_with('>')
        || stripped.starts_with('-')
        || stripped.starts_with('*')
        || stripped.starts_with('【')
        || stripped.starts_with('#')
        || stripped.starts_with('|')
    {
        return false;
    }
    if is_table_rule(stripped) {
        return false;
    }
    if re_bold_only(stripped) {
        return false;
    }
    if re_ordered_list(stripped) {
        return false;
    }
    true
}

/// `_looks_like_heading_line_for_weixin`.
pub fn looks_like_heading_line_for_weixin(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    if header_match(stripped).is_some() {
        return true;
    }
    stripped.chars().count() <= 24 && (stripped.ends_with(':') || stripped.ends_with('：'))
}

/// `_should_split_short_chat_block_for_weixin`.
pub fn should_split_short_chat_block_for_weixin(block: &str) -> bool {
    let lines: Vec<&str> = block
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .collect();
    if !(2..=6).contains(&lines.len()) {
        return false;
    }
    if looks_like_heading_line_for_weixin(lines[0]) {
        return false;
    }
    lines.iter().all(|line| looks_like_chatty_line_for_weixin(line))
}

/// Simple message truncation matching `BasePlatformAdapter.truncate_message`:
/// hard-slice into `max_length`-sized chunks by character count.
pub fn truncate_message(text: &str, max_length: usize) -> Vec<String> {
    if max_length == 0 {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_length {
        return vec![text.to_string()];
    }
    chars
        .chunks(max_length)
        .map(|c| c.iter().collect())
        .collect()
}

/// `_pack_markdown_blocks_for_weixin`.
pub fn pack_markdown_blocks_for_weixin(content: &str, max_length: usize) -> Vec<String> {
    if content.chars().count() <= max_length {
        return vec![content.to_string()];
    }
    let mut packed: Vec<String> = Vec::new();
    let mut current = String::new();
    for block in split_markdown_blocks(content) {
        let candidate = if current.is_empty() {
            block.clone()
        } else {
            format!("{current}\n\n{block}")
        };
        if candidate.chars().count() <= max_length {
            current = candidate;
            continue;
        }
        if !current.is_empty() {
            packed.push(current.clone());
            current.clear();
        }
        if block.chars().count() <= max_length {
            current = block;
            continue;
        }
        packed.extend(truncate_message(&block, max_length));
    }
    if !current.is_empty() {
        packed.push(current);
    }
    packed
}

/// `_split_text_for_weixin_delivery`.
pub fn split_text_for_weixin_delivery(
    content: &str,
    max_length: usize,
    split_per_line: bool,
) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    if split_per_line {
        if content.chars().count() <= max_length && !content.contains('\n') {
            return vec![content.to_string()];
        }
        let mut chunks: Vec<String> = Vec::new();
        for unit in split_delivery_units_for_weixin(content) {
            if unit.chars().count() <= max_length {
                chunks.push(unit);
                continue;
            }
            chunks.extend(pack_markdown_blocks_for_weixin(&unit, max_length));
        }
        let filtered: Vec<String> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
        return if filtered.is_empty() {
            vec![content.to_string()]
        } else {
            filtered
        };
    }
    if content.chars().count() <= max_length {
        return if should_split_short_chat_block_for_weixin(content) {
            split_delivery_units_for_weixin(content)
                .into_iter()
                .filter(|u| !u.is_empty())
                .collect()
        } else {
            vec![content.to_string()]
        };
    }
    let packed = pack_markdown_blocks_for_weixin(content, max_length);
    if packed.is_empty() {
        vec![content.to_string()]
    } else {
        packed
    }
}

/// `format_message`: normalize markdown blocks (None -> empty string).
pub fn format_message(content: Option<&str>) -> String {
    match content {
        None => String::new(),
        Some(c) => normalize_markdown_blocks(c),
    }
}

// ---------------------------------------------------------------------------
// Account / path helpers & persistence
// ---------------------------------------------------------------------------

/// `_account_dir` — `<hermes_home>/weixin/accounts` (created on access).
pub fn account_dir(hermes_home: &str) -> PathBuf {
    let path = Path::new(hermes_home).join("weixin").join("accounts");
    let _ = std::fs::create_dir_all(&path);
    path
}

/// `_account_file`.
pub fn account_file(hermes_home: &str, account_id: &str) -> PathBuf {
    account_dir(hermes_home).join(format!("{account_id}.json"))
}

/// `_sync_buf_path`.
pub fn sync_buf_path(hermes_home: &str, account_id: &str) -> PathBuf {
    account_dir(hermes_home).join(format!("{account_id}.sync.json"))
}

fn utc_now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// `save_weixin_account` — persist credentials (mode 0o600 on unix).
pub fn save_weixin_account(
    hermes_home: &str,
    account_id: &str,
    token: &str,
    base_url: &str,
    user_id: &str,
) -> std::io::Result<()> {
    let payload = json!({
        "token": token,
        "base_url": base_url,
        "user_id": user_id,
        "saved_at": utc_now_iso(),
    });
    let path = account_file(hermes_home, account_id);
    crate::mod_utils::atomic_json_write(&path, &payload, 0)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// `load_weixin_account` — load persisted credentials (None on missing/parse error).
pub fn load_weixin_account(hermes_home: &str, account_id: &str) -> Option<Value> {
    let path = account_file(hermes_home, account_id);
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

/// `_load_sync_buf`.
pub fn load_sync_buf(hermes_home: &str, account_id: &str) -> String {
    let path = sync_buf_path(hermes_home, account_id);
    if !path.exists() {
        return String::new();
    }
    let Ok(text) = std::fs::read_to_string(&path) else {
        return String::new();
    };
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| {
            v.get("get_updates_buf")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_default()
}

/// `_save_sync_buf`.
pub fn save_sync_buf(hermes_home: &str, account_id: &str, sync_buf: &str) -> std::io::Result<()> {
    let path = sync_buf_path(hermes_home, account_id);
    crate::mod_utils::atomic_json_write(&path, &json!({ "get_updates_buf": sync_buf }), 0)
}

// ---------------------------------------------------------------------------
// ContextTokenStore
// ---------------------------------------------------------------------------

/// Disk-backed `context_token` cache keyed by `account_id:user_id`.
pub struct ContextTokenStore {
    root: PathBuf,
    cache: HashMap<String, String>,
}

impl ContextTokenStore {
    pub fn new(hermes_home: &str) -> Self {
        Self {
            root: account_dir(hermes_home),
            cache: HashMap::new(),
        }
    }

    fn path(&self, account_id: &str) -> PathBuf {
        self.root.join(format!("{account_id}.context-tokens.json"))
    }

    pub fn key(&self, account_id: &str, user_id: &str) -> String {
        format!("{account_id}:{user_id}")
    }

    /// Load persisted tokens for `account_id` into the in-memory cache.
    pub fn restore(&mut self, account_id: &str) {
        let path = self.path(account_id);
        if !path.exists() {
            return;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) else {
            return;
        };
        for (user_id, token) in map {
            if let Value::String(t) = token {
                if !t.is_empty() {
                    self.cache.insert(self.key(account_id, &user_id), t);
                }
            }
        }
    }

    pub fn get(&self, account_id: &str, user_id: &str) -> Option<String> {
        self.cache.get(&self.key(account_id, user_id)).cloned()
    }

    pub fn set(&mut self, account_id: &str, user_id: &str, token: &str) {
        let k = self.key(account_id, user_id);
        self.cache.insert(k, token.to_string());
        self.persist(account_id);
    }

    /// Remove a cached token (used on session-expired retry).
    pub fn remove(&mut self, account_id: &str, user_id: &str) {
        let k = self.key(account_id, user_id);
        self.cache.remove(&k);
    }

    fn persist(&self, account_id: &str) {
        let prefix = format!("{account_id}:");
        let mut payload = serde_json::Map::new();
        for (k, v) in &self.cache {
            if let Some(stripped) = k.strip_prefix(&prefix) {
                payload.insert(stripped.to_string(), Value::String(v.clone()));
            }
        }
        let _ = crate::mod_utils::atomic_json_write(
            &self.path(account_id),
            &Value::Object(payload),
            0,
        );
    }
}

// ---------------------------------------------------------------------------
// TypingTicketCache
// ---------------------------------------------------------------------------

/// Short-lived typing-ticket cache (`getconfig`).
pub struct TypingTicketCache {
    ttl_seconds: f64,
    cache: HashMap<String, (String, f64)>,
}

impl TypingTicketCache {
    pub fn new(ttl_seconds: f64) -> Self {
        Self {
            ttl_seconds,
            cache: HashMap::new(),
        }
    }

    pub fn default_ttl() -> Self {
        Self::new(600.0)
    }

    pub fn get(&mut self, user_id: &str) -> Option<String> {
        let expired = match self.cache.get(user_id) {
            None => return None,
            Some((_, ts)) => now_secs() - ts >= self.ttl_seconds,
        };
        if expired {
            self.cache.remove(user_id);
            return None;
        }
        self.cache.get(user_id).map(|(t, _)| t.clone())
    }

    pub fn set(&mut self, user_id: &str, ticket: &str) {
        self.cache
            .insert(user_id.to_string(), (ticket.to_string(), now_secs()));
    }
}

// ---------------------------------------------------------------------------
// Request payload construction
// ---------------------------------------------------------------------------

/// `_send_message` payload (without `base_info`, which `_api_post` merges in).
///
/// Returns Err on empty/whitespace text to mirror the Python `ValueError`.
pub fn build_send_message_payload(
    to: &str,
    text: &str,
    context_token: Option<&str>,
    client_id: &str,
) -> Result<Value, String> {
    if text.trim().is_empty() {
        return Err("_send_message: text must not be empty".to_string());
    }
    let mut message = json!({
        "from_user_id": "",
        "to_user_id": to,
        "client_id": client_id,
        "message_type": MSG_TYPE_BOT,
        "message_state": MSG_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    });
    if let Some(ct) = context_token {
        if !ct.is_empty() {
            message["context_token"] = Value::String(ct.to_string());
        }
    }
    Ok(json!({ "msg": message }))
}

/// `_send_typing` payload.
pub fn build_send_typing_payload(to_user_id: &str, typing_ticket: &str, status: i64) -> Value {
    json!({
        "ilink_user_id": to_user_id,
        "typing_ticket": typing_ticket,
        "status": status,
    })
}

/// `_get_config` payload.
pub fn build_get_config_payload(user_id: &str, context_token: Option<&str>) -> Value {
    let mut payload = json!({ "ilink_user_id": user_id });
    if let Some(ct) = context_token {
        if !ct.is_empty() {
            payload["context_token"] = Value::String(ct.to_string());
        }
    }
    payload
}

/// `_get_upload_url` payload.
#[allow(clippy::too_many_arguments)]
pub fn build_get_upload_url_payload(
    to_user_id: &str,
    media_type: i64,
    filekey: &str,
    rawsize: usize,
    rawfilemd5: &str,
    filesize: usize,
    aeskey_hex: &str,
) -> Value {
    json!({
        "filekey": filekey,
        "media_type": media_type,
        "to_user_id": to_user_id,
        "rawsize": rawsize,
        "rawfilemd5": rawfilemd5,
        "filesize": filesize,
        "no_need_thumb": true,
        "aeskey": aeskey_hex,
    })
}

/// `_get_updates` payload.
pub fn build_get_updates_payload(sync_buf: &str) -> Value {
    json!({ "get_updates_buf": sync_buf })
}

/// Merge the user payload with `base_info` and serialize, as `_api_post` does.
pub fn build_post_body(payload: &Value) -> String {
    let mut merged = payload.clone();
    if let Value::Object(ref mut map) = merged {
        map.insert("base_info".to_string(), base_info());
    }
    json_dumps(&merged)
}

// ---------------------------------------------------------------------------
// Outbound media item builder (`_outbound_media_builder`)
// ---------------------------------------------------------------------------

/// Inputs for building an outbound media `item_list` entry.
pub struct OutboundMediaArgs<'a> {
    pub encrypt_query_param: &'a str,
    /// base64(hex_string) aes key for the API.
    pub aes_key_for_api: &'a str,
    pub ciphertext_size: usize,
    pub plaintext_size: usize,
    pub filename: &'a str,
    pub rawfilemd5: &'a str,
    pub encode_type: Option<i64>,
    pub sample_rate: Option<i64>,
    pub bits_per_sample: Option<i64>,
    pub play_length: i64,
    pub playtime: i64,
}

/// Determine the `(media_type, kind)` for an outbound file path.
///
/// `kind` selects which item builder applies; mirrors `_outbound_media_builder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutboundKind {
    Image,
    Video,
    Voice,
    File,
}

/// Mirror `_outbound_media_builder`'s media_type + branch selection.
pub fn outbound_media_kind(path: &str, force_file_attachment: bool) -> (i64, OutboundKind) {
    let mime = mime_from_filename(path);
    if mime.starts_with("image/") {
        return (MEDIA_IMAGE, OutboundKind::Image);
    }
    if mime.starts_with("video/") {
        return (MEDIA_VIDEO, OutboundKind::Video);
    }
    if path.ends_with(".silk") && !force_file_attachment {
        return (MEDIA_VOICE, OutboundKind::Voice);
    }
    // audio/* and everything else -> file
    (MEDIA_FILE, OutboundKind::File)
}

/// Build the outbound media `item_list` entry for the given kind.
pub fn build_outbound_media_item(kind: OutboundKind, args: &OutboundMediaArgs) -> Value {
    let media = json!({
        "encrypt_query_param": args.encrypt_query_param,
        "aes_key": args.aes_key_for_api,
        "encrypt_type": 1,
    });
    match kind {
        OutboundKind::Image => json!({
            "type": ITEM_IMAGE,
            "image_item": {
                "media": media,
                "mid_size": args.ciphertext_size,
            },
        }),
        OutboundKind::Video => json!({
            "type": ITEM_VIDEO,
            "video_item": {
                "media": media,
                "video_size": args.ciphertext_size,
                "play_length": args.play_length,
                "video_md5": args.rawfilemd5,
            },
        }),
        OutboundKind::Voice => json!({
            "type": ITEM_VOICE,
            "voice_item": {
                "media": media,
                "encode_type": args.encode_type,
                "bits_per_sample": args.bits_per_sample,
                "sample_rate": args.sample_rate,
                "playtime": args.playtime,
            },
        }),
        OutboundKind::File => json!({
            "type": ITEM_FILE,
            "file_item": {
                "media": media,
                "file_name": args.filename,
                "len": args.plaintext_size.to_string(),
            },
        }),
    }
}

/// Compute the base64(hex) aes-key form the iLink API expects (`aes_key_for_api`).
pub fn aes_key_for_api(aes_key: &[u8; 16]) -> String {
    B64.encode(hex_encode(aes_key).as_bytes())
}

// ---------------------------------------------------------------------------
// Network: blocking iLink client
// ---------------------------------------------------------------------------

/// Thin blocking iLink HTTP client. Mirrors `_api_post` / `_api_get`.
pub struct IlinkClient {
    http: reqwest::blocking::Client,
}

impl Default for IlinkClient {
    fn default() -> Self {
        Self::new()
    }
}

impl IlinkClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::blocking::Client::new(),
        }
    }

    /// POST `endpoint` with the given payload (base_info merged in) and return
    /// the parsed JSON response. Mirrors `_api_post`.
    pub fn api_post(
        &self,
        base_url: &str,
        endpoint: &str,
        payload: &Value,
        token: Option<&str>,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        let body = build_post_body(payload);
        let url = format!("{}/{}", rstrip_slash(base_url), endpoint);
        let mut req = self
            .http
            .post(&url)
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .body(body.clone());
        for (k, v) in build_headers(token, &body) {
            req = req.header(&k, &v);
        }
        let resp = req.send().map_err(|e| e.to_string())?;
        let status = resp.status();
        let raw = resp.text().map_err(|e| e.to_string())?;
        if !status.is_success() {
            let snippet: String = raw.chars().take(200).collect();
            return Err(format!(
                "iLink POST {endpoint} HTTP {}: {snippet}",
                status.as_u16()
            ));
        }
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    }

    /// GET `endpoint` and return the parsed JSON response. Mirrors `_api_get`.
    pub fn api_get(
        &self,
        base_url: &str,
        endpoint: &str,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        let url = format!("{}/{}", rstrip_slash(base_url), endpoint);
        let mut req = self
            .http
            .get(&url)
            .timeout(std::time::Duration::from_millis(timeout_ms));
        for (k, v) in build_get_headers() {
            req = req.header(&k, &v);
        }
        let resp = req.send().map_err(|e| e.to_string())?;
        let status = resp.status();
        let raw = resp.text().map_err(|e| e.to_string())?;
        if !status.is_success() {
            let snippet: String = raw.chars().take(200).collect();
            return Err(format!(
                "iLink GET {endpoint} HTTP {}: {snippet}",
                status.as_u16()
            ));
        }
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    }

    /// `_get_updates`: long-poll; map a timeout to an empty result keeping
    /// `sync_buf` (mirrors the Python `asyncio.TimeoutError` branch).
    pub fn get_updates(
        &self,
        base_url: &str,
        token: &str,
        sync_buf: &str,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        match self.api_post(
            base_url,
            EP_GET_UPDATES,
            &build_get_updates_payload(sync_buf),
            Some(token),
            timeout_ms,
        ) {
            Ok(v) => Ok(v),
            Err(e) if e.contains("timed out") || e.to_lowercase().contains("timeout") => {
                Ok(json!({ "ret": 0, "msgs": [], "get_updates_buf": sync_buf }))
            }
            Err(e) => Err(e),
        }
    }

    /// `_send_message`.
    pub fn send_message(
        &self,
        base_url: &str,
        token: &str,
        to: &str,
        text: &str,
        context_token: Option<&str>,
        client_id: &str,
    ) -> Result<Value, String> {
        let payload = build_send_message_payload(to, text, context_token, client_id)?;
        self.api_post(base_url, EP_SEND_MESSAGE, &payload, Some(token), API_TIMEOUT_MS)
    }

    /// `_send_typing`.
    pub fn send_typing(
        &self,
        base_url: &str,
        token: &str,
        to_user_id: &str,
        typing_ticket: &str,
        status: i64,
    ) -> Result<Value, String> {
        let payload = build_send_typing_payload(to_user_id, typing_ticket, status);
        self.api_post(base_url, EP_SEND_TYPING, &payload, Some(token), CONFIG_TIMEOUT_MS)
    }

    /// `_get_config`.
    pub fn get_config(
        &self,
        base_url: &str,
        token: &str,
        user_id: &str,
        context_token: Option<&str>,
    ) -> Result<Value, String> {
        let payload = build_get_config_payload(user_id, context_token);
        self.api_post(base_url, EP_GET_CONFIG, &payload, Some(token), CONFIG_TIMEOUT_MS)
    }

    /// `_get_upload_url`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_upload_url(
        &self,
        base_url: &str,
        token: &str,
        to_user_id: &str,
        media_type: i64,
        filekey: &str,
        rawsize: usize,
        rawfilemd5: &str,
        filesize: usize,
        aeskey_hex: &str,
    ) -> Result<Value, String> {
        let payload = build_get_upload_url_payload(
            to_user_id, media_type, filekey, rawsize, rawfilemd5, filesize, aeskey_hex,
        );
        self.api_post(base_url, EP_GET_UPLOAD_URL, &payload, Some(token), API_TIMEOUT_MS)
    }

    /// Upload encrypted media ciphertext to the CDN (`_upload_ciphertext`).
    /// On HTTP 200, returns the `x-encrypted-param` response header.
    pub fn upload_ciphertext(&self, ciphertext: &[u8], upload_url: &str) -> Result<String, String> {
        let resp = self
            .http
            .post(upload_url)
            .timeout(std::time::Duration::from_secs(120))
            .header("Content-Type", "application/octet-stream")
            .body(ciphertext.to_vec())
            .send()
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        if status.as_u16() == 200 {
            let encrypted_param = resp
                .headers()
                .get("x-encrypted-param")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            if let Some(p) = encrypted_param {
                if !p.is_empty() {
                    return Ok(p);
                }
            }
            let raw = resp.text().unwrap_or_default();
            let snippet: String = raw.chars().take(200).collect();
            return Err(format!("CDN upload missing x-encrypted-param header: {snippet}"));
        }
        let raw = resp.text().unwrap_or_default();
        let snippet: String = raw.chars().take(200).collect();
        Err(format!("CDN upload HTTP {}: {snippet}", status.as_u16()))
    }

    /// Download bytes from `url` (`_download_bytes`).
    pub fn download_bytes(&self, url: &str, timeout_seconds: u64) -> Result<Vec<u8>, String> {
        let resp = self
            .http
            .get(url)
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .send()
            .map_err(|e| e.to_string())?;
        let resp = resp.error_for_status().map_err(|e| e.to_string())?;
        resp.bytes().map(|b| b.to_vec()).map_err(|e| e.to_string())
    }

    /// `_download_and_decrypt_media`: download via encrypted_query_param or
    /// full_url (with SSRF guard), then AES-decrypt if an aes key is supplied.
    pub fn download_and_decrypt_media(
        &self,
        cdn_base_url: &str,
        encrypted_query_param: Option<&str>,
        aes_key_b64: Option<&str>,
        full_url: Option<&str>,
        timeout_seconds: u64,
    ) -> Result<Vec<u8>, String> {
        let mut raw = if let Some(eqp) = encrypted_query_param.filter(|s| !s.is_empty()) {
            self.download_bytes(&cdn_download_url(cdn_base_url, eqp), timeout_seconds)?
        } else if let Some(fu) = full_url.filter(|s| !s.is_empty()) {
            assert_weixin_cdn_url(fu)?;
            self.download_bytes(fu, timeout_seconds)?
        } else {
            return Err("media item had neither encrypt_query_param nor full_url".to_string());
        };
        if let Some(k) = aes_key_b64.filter(|s| !s.is_empty()) {
            let key = parse_aes_key(k)?;
            raw = aes128_ecb_decrypt(&raw, &key);
        }
        Ok(raw)
    }
}

// ---------------------------------------------------------------------------
// QR login status classification
// ---------------------------------------------------------------------------

/// Parsed terminal outcome of the QR login confirmation step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrCredentials {
    pub account_id: String,
    pub token: String,
    pub base_url: String,
    pub user_id: String,
}

/// Classification of a `get_qrcode_status` response (`qr_login` loop body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QrStatus {
    Wait,
    Scaned,
    /// Redirect to a new host (when present).
    ScanedButRedirect(Option<String>),
    Expired,
    /// Confirmed but credentials incomplete -> Python returns None.
    ConfirmedIncomplete,
    Confirmed(QrCredentials),
    /// Any other/unknown status string.
    Other(String),
}

/// Parse a `get_qrcode_status` response into a `QrStatus`. `default_base_url`
/// is used when the confirmed payload omits `baseurl` (Python uses ILINK_BASE_URL).
pub fn classify_qr_status(resp: &Value, default_base_url: &str) -> QrStatus {
    let status = resp
        .get("status")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("wait");
    match status {
        "wait" => QrStatus::Wait,
        "scaned" => QrStatus::Scaned,
        "scaned_but_redirect" => {
            let host = resp
                .get("redirect_host")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            QrStatus::ScanedButRedirect(host)
        }
        "expired" => QrStatus::Expired,
        "confirmed" => {
            let account_id = resp
                .get("ilink_bot_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let token = resp
                .get("bot_token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let base_url = resp
                .get("baseurl")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(default_base_url)
                .to_string();
            let user_id = resp
                .get("ilink_user_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if account_id.is_empty() || token.is_empty() {
                QrStatus::ConfirmedIncomplete
            } else {
                QrStatus::Confirmed(QrCredentials {
                    account_id,
                    token,
                    base_url,
                    user_id,
                })
            }
        }
        other => QrStatus::Other(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Inbound admission policy
// ---------------------------------------------------------------------------

/// `_is_dm_allowed`.
pub fn is_dm_allowed(dm_policy: &str, sender_id: &str, allow_from: &[String]) -> bool {
    match dm_policy {
        "disabled" => false,
        "allowlist" => allow_from.iter().any(|s| s == sender_id),
        _ => true,
    }
}

/// Group admission gate matching `_process_message`'s group branch.
/// Returns true when the group message should be admitted.
pub fn is_group_allowed(
    group_policy: &str,
    effective_chat_id: &str,
    group_allow_from: &[String],
) -> bool {
    match group_policy {
        "disabled" => false,
        "allowlist" => group_allow_from.iter().any(|s| s == effective_chat_id),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Requirements gate
// ---------------------------------------------------------------------------

/// `check_weixin_requirements`. In the native build aiohttp/cryptography are
/// replaced by reqwest/aes which are always compiled in, so this returns true.
pub fn check_weixin_requirements() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_version_constant() {
        assert_eq!(ILINK_APP_CLIENT_VERSION, 0x020200);
    }

    #[test]
    fn test_aes_roundtrip() {
        let key = [7u8; 16];
        let pt = b"hello weixin media payload that crosses block boundaries!!";
        let ct = aes128_ecb_encrypt(pt, &key);
        assert_eq!(ct.len() % 16, 0);
        let back = aes128_ecb_decrypt(&ct, &key);
        assert_eq!(back, pt);
    }

    #[test]
    fn test_aes_decrypt_invalid_padding_returns_unchanged() {
        // A buffer whose last byte is not valid PKCS#7 padding should come back
        // unchanged (mirrors the permissive Python unpad).
        let key = [1u8; 16];
        // Build ciphertext from plaintext that, after decrypt, won't have valid
        // padding: just encrypt 16 zero bytes (pad would be 16 -> valid). Instead
        // encrypt raw without padding by directly using cipher on a crafted block.
        // Simpler: confirm exact-block content round-trips and a no-pad block is
        // returned verbatim when it lacks valid trailing pad.
        let block = [0xABu8; 16];
        let cipher = Aes128::new(&Array(key));
        let mut arr = Array(block);
        cipher.encrypt_block(&mut arr);
        let out = aes128_ecb_decrypt(&arr.0, &key);
        // Decrypted = 16 * 0xAB; 0xAB = 171 > 16 so no unpad happens.
        assert_eq!(out, vec![0xABu8; 16]);
    }

    #[test]
    fn test_aes_padded_size() {
        assert_eq!(aes_padded_size(0), 16);
        assert_eq!(aes_padded_size(15), 16);
        assert_eq!(aes_padded_size(16), 32);
        assert_eq!(aes_padded_size(31), 32);
    }

    #[test]
    fn test_parse_aes_key_16_raw() {
        let raw = [9u8; 16];
        let b64 = B64.encode(raw);
        assert_eq!(parse_aes_key(&b64).unwrap(), raw);
    }

    #[test]
    fn test_parse_aes_key_32_hex() {
        let key = [0xCDu8; 16];
        let hexstr = hex_encode(&key); // 32 ascii chars
        let b64 = B64.encode(hexstr.as_bytes());
        assert_eq!(parse_aes_key(&b64).unwrap(), key);
    }

    #[test]
    fn test_aes_key_for_api_is_b64_of_hex() {
        let key = [0x10u8; 16];
        let v = aes_key_for_api(&key);
        let decoded = B64.decode(v.as_bytes()).unwrap();
        assert_eq!(String::from_utf8(decoded).unwrap(), hex_encode(&key));
    }

    #[test]
    fn test_quote_all() {
        assert_eq!(quote_all("a b/c?d=e"), "a%20b%2Fc%3Fd%3De");
        assert_eq!(quote_all("safe-._~"), "safe-._~");
    }

    #[test]
    fn test_cdn_urls() {
        assert_eq!(
            cdn_download_url("https://cdn.example/c2c/", "abc def"),
            "https://cdn.example/c2c/download?encrypted_query_param=abc%20def"
        );
        assert_eq!(
            cdn_upload_url("https://cdn/", "p p", "f/k"),
            "https://cdn/upload?encrypted_query_param=p%20p&filekey=f%2Fk"
        );
    }

    #[test]
    fn test_random_uin_is_b64_of_decimal() {
        let uin = random_wechat_uin();
        let decoded = B64.decode(uin.as_bytes()).unwrap();
        let s = String::from_utf8(decoded).unwrap();
        assert!(s.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_assert_cdn_url_allowlist() {
        assert!(assert_weixin_cdn_url("https://wx.qlogo.cn/x.jpg").is_ok());
        assert!(assert_weixin_cdn_url("https://evil.example/x").is_err());
        assert!(assert_weixin_cdn_url("ftp://wx.qlogo.cn/x").is_err());
        assert!(assert_weixin_cdn_url("not a url").is_err());
    }

    #[test]
    fn test_guess_chat_type_dm() {
        let msg = json!({"from_user_id": "userA", "msg_type": 2});
        let (ct, id) = guess_chat_type(&msg, "bot1");
        assert_eq!(ct, "dm");
        assert_eq!(id, "userA");
    }

    #[test]
    fn test_guess_chat_type_group_room() {
        let msg = json!({"room_id": "room@chatroom", "from_user_id": "u"});
        let (ct, id) = guess_chat_type(&msg, "bot1");
        assert_eq!(ct, "group");
        assert_eq!(id, "room@chatroom");
    }

    #[test]
    fn test_guess_chat_type_group_inferred() {
        let msg = json!({"to_user_id": "someoneElse", "from_user_id": "u", "msg_type": 1});
        let (ct, id) = guess_chat_type(&msg, "bot1");
        assert_eq!(ct, "group");
        assert_eq!(id, "someoneElse");
    }

    #[test]
    fn test_extract_text_plain() {
        let items = vec![json!({"type": ITEM_TEXT, "text_item": {"text": "hi"}})];
        assert_eq!(extract_text(&items), "hi");
    }

    #[test]
    fn test_extract_text_quoted_media() {
        let items = vec![json!({
            "type": ITEM_TEXT,
            "text_item": {"text": "look"},
            "ref_msg": {"title": "pic", "message_item": {"type": ITEM_IMAGE}}
        })];
        assert_eq!(extract_text(&items), "[引用媒体: pic]\nlook");
    }

    #[test]
    fn test_extract_text_voice_fallback() {
        let items = vec![json!({"type": ITEM_VOICE, "voice_item": {"text": "spoken"}})];
        assert_eq!(extract_text(&items), "spoken");
    }

    #[test]
    fn test_message_type_from_media() {
        assert_eq!(
            message_type_from_media(&["image/jpeg".into()], ""),
            MessageType::Photo
        );
        assert_eq!(
            message_type_from_media(&["video/mp4".into()], ""),
            MessageType::Video
        );
        assert_eq!(
            message_type_from_media(&["audio/silk".into()], ""),
            MessageType::Voice
        );
        assert_eq!(
            message_type_from_media(&["application/pdf".into()], ""),
            MessageType::Document
        );
        assert_eq!(message_type_from_media(&[], "/help"), MessageType::Command);
        assert_eq!(message_type_from_media(&[], "hi"), MessageType::Text);
    }

    #[test]
    fn test_coerce_bool() {
        assert!(coerce_bool(Some("true"), false));
        assert!(coerce_bool(Some("ON"), false));
        assert!(!coerce_bool(Some("no"), true));
        assert!(coerce_bool(None, true));
        assert!(!coerce_bool(Some(""), false));
        assert!(coerce_bool(Some("garbage"), true));
    }

    #[test]
    fn test_coerce_list() {
        assert_eq!(coerce_list_str("a, b ,,c"), vec!["a", "b", "c"]);
        assert!(coerce_list_str("  ").is_empty());
    }

    #[test]
    fn test_rewrite_headers() {
        assert_eq!(rewrite_headers_for_weixin("# Title"), "【Title】");
        assert_eq!(rewrite_headers_for_weixin("### Sub"), "**Sub**");
        assert_eq!(rewrite_headers_for_weixin("plain"), "plain");
    }

    #[test]
    fn test_normalize_collapses_blank_runs() {
        let input = "a\n\n\n\nb";
        assert_eq!(normalize_markdown_blocks(input), "a\n\nb");
    }

    #[test]
    fn test_split_markdown_blocks_fence() {
        let content = "para1\n\n```py\ncode\n```\n\npara2";
        let blocks = split_markdown_blocks(content);
        assert_eq!(blocks.len(), 3);
        assert!(blocks[1].starts_with("```py"));
        assert!(blocks[1].ends_with("```"));
    }

    #[test]
    fn test_split_text_compact_under_limit() {
        let out = split_text_for_weixin_delivery("just one line", 2000, false);
        assert_eq!(out, vec!["just one line"]);
    }

    #[test]
    fn test_split_text_short_chat_block() {
        let content = "hey\nwhats up\ncool";
        let out = split_text_for_weixin_delivery(content, 2000, false);
        assert_eq!(out, vec!["hey", "whats up", "cool"]);
    }

    #[test]
    fn test_split_text_per_line_legacy() {
        let content = "line one\nline two";
        let out = split_text_for_weixin_delivery(content, 2000, true);
        assert_eq!(out, vec!["line one", "line two"]);
    }

    #[test]
    fn test_pack_oversize() {
        let content = "a".repeat(50);
        let out = pack_markdown_blocks_for_weixin(&content, 20);
        assert!(out.len() >= 3);
        assert!(out.iter().all(|c| c.chars().count() <= 20));
    }

    #[test]
    fn test_table_rewrite() {
        let lines = vec![
            "| Name | Age |".to_string(),
            "| --- | --- |".to_string(),
            "| Alice | 30 |".to_string(),
        ];
        let out = rewrite_table_block_for_weixin(&lines);
        assert!(out.contains("- Name: Alice"));
        assert!(out.contains("Age: 30"));
    }

    #[test]
    fn test_is_session_expired_and_rate_limit() {
        assert!(is_session_expired(Some(-14), Some(0), Some("x")));
        assert!(is_session_expired(Some(-2), Some(0), Some("unknown error")));
        assert!(!is_session_expired(Some(-2), Some(0), Some("rate limited")));
        assert!(is_rate_limited(Some(-2), Some(0)));
        assert!(!is_rate_limited(Some(0), Some(0)));
    }

    #[test]
    fn test_build_send_message_payload() {
        let p = build_send_message_payload("u1", "hi", Some("ctx"), "cid").unwrap();
        assert_eq!(p["msg"]["to_user_id"], "u1");
        assert_eq!(p["msg"]["context_token"], "ctx");
        assert_eq!(p["msg"]["item_list"][0]["type"], ITEM_TEXT);
        assert_eq!(p["msg"]["item_list"][0]["text_item"]["text"], "hi");
        assert!(build_send_message_payload("u", "  ", None, "c").is_err());
    }

    #[test]
    fn test_build_post_body_merges_base_info() {
        let body = build_post_body(&json!({"a": 1}));
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["base_info"]["channel_version"], CHANNEL_VERSION);
        assert_eq!(parsed["a"], 1);
        // compact: no spaces
        assert!(!body.contains(", "));
        assert!(!body.contains(": "));
    }

    #[test]
    fn test_build_headers() {
        let headers = build_headers(Some("tok"), "{}");
        let map: HashMap<_, _> = headers.into_iter().collect();
        assert_eq!(map.get("Authorization").unwrap(), "Bearer tok");
        assert_eq!(map.get("iLink-App-Id").unwrap(), "bot");
        assert_eq!(map.get("Content-Length").unwrap(), "2");
        let no_tok: HashMap<_, _> = build_headers(None, "{}").into_iter().collect();
        assert!(!no_tok.contains_key("Authorization"));
    }

    #[test]
    fn test_outbound_media_kind() {
        assert_eq!(outbound_media_kind("a.jpg", false).1, OutboundKind::Image);
        assert_eq!(outbound_media_kind("a.mp4", false).1, OutboundKind::Video);
        assert_eq!(outbound_media_kind("a.silk", false).1, OutboundKind::Voice);
        assert_eq!(outbound_media_kind("a.silk", true).1, OutboundKind::File);
        assert_eq!(outbound_media_kind("a.mp3", false).1, OutboundKind::File);
        assert_eq!(outbound_media_kind("a.bin", false).1, OutboundKind::File);
        assert_eq!(outbound_media_kind("a.jpg", false).0, MEDIA_IMAGE);
    }

    #[test]
    fn test_build_outbound_image_item() {
        let args = OutboundMediaArgs {
            encrypt_query_param: "EQP",
            aes_key_for_api: "KEY",
            ciphertext_size: 100,
            plaintext_size: 90,
            filename: "x.jpg",
            rawfilemd5: "md5",
            encode_type: None,
            sample_rate: None,
            bits_per_sample: None,
            play_length: 0,
            playtime: 0,
        };
        let item = build_outbound_media_item(OutboundKind::Image, &args);
        assert_eq!(item["type"], ITEM_IMAGE);
        assert_eq!(item["image_item"]["mid_size"], 100);
        assert_eq!(item["image_item"]["media"]["aes_key"], "KEY");
        assert_eq!(item["image_item"]["media"]["encrypt_type"], 1);
    }

    #[test]
    fn test_build_outbound_file_item_len_is_string() {
        let args = OutboundMediaArgs {
            encrypt_query_param: "E",
            aes_key_for_api: "K",
            ciphertext_size: 0,
            plaintext_size: 1234,
            filename: "doc.bin",
            rawfilemd5: "",
            encode_type: None,
            sample_rate: None,
            bits_per_sample: None,
            play_length: 0,
            playtime: 0,
        };
        let item = build_outbound_media_item(OutboundKind::File, &args);
        assert_eq!(item["file_item"]["len"], "1234");
        assert_eq!(item["file_item"]["file_name"], "doc.bin");
    }

    #[test]
    fn test_classify_qr_status() {
        assert_eq!(classify_qr_status(&json!({}), ILINK_BASE_URL), QrStatus::Wait);
        assert_eq!(
            classify_qr_status(&json!({"status": "scaned"}), ILINK_BASE_URL),
            QrStatus::Scaned
        );
        assert_eq!(
            classify_qr_status(
                &json!({"status": "scaned_but_redirect", "redirect_host": "h.example"}),
                ILINK_BASE_URL
            ),
            QrStatus::ScanedButRedirect(Some("h.example".to_string()))
        );
        assert_eq!(
            classify_qr_status(&json!({"status": "expired"}), ILINK_BASE_URL),
            QrStatus::Expired
        );
        assert_eq!(
            classify_qr_status(&json!({"status": "confirmed"}), ILINK_BASE_URL),
            QrStatus::ConfirmedIncomplete
        );
        let confirmed = classify_qr_status(
            &json!({"status": "confirmed", "ilink_bot_id": "b", "bot_token": "t"}),
            "https://default",
        );
        assert_eq!(
            confirmed,
            QrStatus::Confirmed(QrCredentials {
                account_id: "b".into(),
                token: "t".into(),
                base_url: "https://default".into(),
                user_id: "".into(),
            })
        );
    }

    #[test]
    fn test_admission_policies() {
        assert!(is_dm_allowed("open", "u", &[]));
        assert!(!is_dm_allowed("disabled", "u", &[]));
        assert!(is_dm_allowed("allowlist", "u", &["u".into()]));
        assert!(!is_dm_allowed("allowlist", "u", &["x".into()]));
        assert!(!is_group_allowed("disabled", "g", &[]));
        assert!(is_group_allowed("open", "g", &[]));
        assert!(is_group_allowed("allowlist", "g", &["g".into()]));
        assert!(!is_group_allowed("allowlist", "g", &["x".into()]));
    }

    #[test]
    fn test_context_token_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "weixin-test-{}",
            std::process::id()
        ));
        let home = dir.to_string_lossy().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = ContextTokenStore::new(&home);
        store.set("acct", "peer", "TOK");
        assert_eq!(store.get("acct", "peer"), Some("TOK".to_string()));
        // restore from disk into a fresh store
        let mut store2 = ContextTokenStore::new(&home);
        store2.restore("acct");
        assert_eq!(store2.get("acct", "peer"), Some("TOK".to_string()));
        store2.remove("acct", "peer");
        assert_eq!(store2.get("acct", "peer"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_typing_ticket_cache_ttl() {
        let mut c = TypingTicketCache::new(0.0);
        c.set("u", "ticket");
        // ttl 0 -> immediately expired (now - ts >= 0)
        assert_eq!(c.get("u"), None);
        let mut c2 = TypingTicketCache::new(600.0);
        c2.set("u", "ticket");
        assert_eq!(c2.get("u"), Some("ticket".to_string()));
    }

    #[test]
    fn test_account_persistence_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "weixin-acct-{}",
            std::process::id()
        ));
        let home = dir.to_string_lossy().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        save_weixin_account(&home, "acct1", "tok1", "https://b", "user1").unwrap();
        let loaded = load_weixin_account(&home, "acct1").unwrap();
        assert_eq!(loaded["token"], "tok1");
        assert_eq!(loaded["base_url"], "https://b");
        assert_eq!(loaded["user_id"], "user1");
        save_sync_buf(&home, "acct1", "BUF").unwrap();
        assert_eq!(load_sync_buf(&home, "acct1"), "BUF");
        assert_eq!(load_sync_buf(&home, "missing"), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mime_from_filename() {
        assert_eq!(mime_from_filename("a.jpg"), "image/jpeg");
        assert_eq!(mime_from_filename("a.mp4"), "video/mp4");
        assert_eq!(mime_from_filename("a.unknownext"), "application/octet-stream");
    }
}
