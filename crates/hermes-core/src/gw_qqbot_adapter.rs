//! QQ Bot platform adapter — native Rust port of
//! `gateway/platforms/qqbot/adapter.py`.
//!
//! The Python adapter connects to the QQ Bot WebSocket Gateway for inbound
//! events and uses the REST API (`api.sgroup.qq.com`) for outbound messages and
//! media uploads. The original module is built on `aiohttp` (websocket +
//! session) and `httpx` (REST client) with an asyncio reconnect loop, none of
//! which translate one-to-one into a synchronous Rust port.
//!
//! This module therefore ports the **deterministic, side-effect-free logic**
//! plus the **request-construction and response-parsing** that other Hermes
//! code (and tests) depend on, reproducing the Python behavior faithfully:
//!
//! - Configuration extraction ([`QQConfig::from_extra`]) and ACL policies
//!   ([`QQAdapter::is_dm_allowed`], [`is_group_allowed`], [`entry_matches`]).
//! - Message-type detection ([`detect_message_type`]) and attachment
//!   classification ([`is_voice_content_type`], [`guess_ext_from_data`],
//!   [`looks_like_silk`]).
//! - Outbound body construction: [`QQAdapter::build_text_body`],
//!   [`build_media_body`], [`build_input_notify_body`], and the REST request
//!   descriptors ([`api_request`], gateway URL / token request shapes).
//! - Response parsing: [`parse_token_response`], [`parse_gateway_response`],
//!   [`parse_stt_response`], [`parse_send_response`].
//! - WebSocket payload dispatch logic: [`classify_payload`],
//!   [`identify_payload`], [`resume_payload`], [`heartbeat_payload`],
//!   [`hello_interval_seconds`], close-code handling ([`close_code_action`]).
//! - Misc helpers: [`strip_at_mention`], [`coerce_list`], [`is_url`],
//!   [`build_user_agent`], [`next_msg_seq`], [`parse_qq_timestamp`],
//!   [`SeenMessages`] dedup, [`resolve_stt_config`].
//!
//! The live aiohttp websocket loop, asyncio task scheduling, ffmpeg/pilk audio
//! conversion subprocesses, and the on-disk media caching are out of scope —
//! they require runtime wiring the caller supplies. This module provides the
//! pure logic those paths call into so the network shapes stay byte-exact.

use std::collections::HashMap;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

use crate::gw_qq_constants::{
    DEDUP_MAX_SIZE, DEDUP_WINDOW_SECONDS, MAX_MESSAGE_LENGTH, MEDIA_TYPE_FILE, MSG_TYPE_INPUT_NOTIFY,
    MSG_TYPE_MARKDOWN, MSG_TYPE_MEDIA, MSG_TYPE_TEXT, QQBOT_VERSION,
};

// ---------------------------------------------------------------------------
// Result of a platform send/edit operation. Local mirror of the Python
// `SendResult` dataclass (the gateway-wide `SendResult` lacks `retryable` /
// `raw_response`, which the QQ adapter relies on).
// ---------------------------------------------------------------------------

/// Outcome of a QQ send operation, mirroring the Python `SendResult`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SendResult {
    pub success: bool,
    pub message_id: Option<String>,
    pub error: Option<String>,
    pub retryable: bool,
    pub raw_response: Option<Value>,
}

impl SendResult {
    /// A successful result with no message id (e.g. empty content short-circuit).
    pub fn ok_empty() -> Self {
        SendResult {
            success: true,
            ..Default::default()
        }
    }

    /// A successful result carrying a message id and the raw API response.
    pub fn ok(message_id: impl Into<String>, raw_response: Option<Value>) -> Self {
        SendResult {
            success: true,
            message_id: Some(message_id.into()),
            error: None,
            retryable: false,
            raw_response,
        }
    }

    /// A failure result with an error string.
    pub fn fail(error: impl Into<String>) -> Self {
        SendResult {
            success: false,
            message_id: None,
            error: Some(error.into()),
            retryable: false,
            raw_response: None,
        }
    }

    /// A failure result that is safe to retry.
    pub fn fail_retryable(error: impl Into<String>) -> Self {
        SendResult {
            success: false,
            message_id: None,
            error: Some(error.into()),
            retryable: true,
            raw_response: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Error carried when the QQ WebSocket closes with a specific code.
// Mirrors Python `QQCloseError`.
// ---------------------------------------------------------------------------

/// Carries the close code and reason for a QQ WebSocket close frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QQCloseError {
    pub code: Option<i64>,
    pub reason: String,
}

impl QQCloseError {
    pub fn new(code: Option<i64>, reason: impl Into<String>) -> Self {
        QQCloseError {
            code,
            reason: reason.into(),
        }
    }

    /// The Python `str(QQCloseError)` message form.
    pub fn message(&self) -> String {
        format!(
            "WebSocket closed (code={}, reason={})",
            self.code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "None".to_string()),
            self.reason
        )
    }
}

impl std::fmt::Display for QQCloseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for QQCloseError {}

// ---------------------------------------------------------------------------
// User-Agent — mirrors `gateway/platforms/qqbot/utils.build_user_agent`.
// ---------------------------------------------------------------------------

/// Build a descriptive User-Agent string.
///
/// Format: `QQBotAdapter/<qqbot_version> (Python/<py>; <os>; Hermes/<hermes>)`.
///
/// In this Rust port there is no embedded Python interpreter, so the `Python/`
/// token is filled from `QQ_PY_VERSION` (defaulting to `3.11.0`) and the
/// Hermes version from the `CARGO_PKG_VERSION` (falling back to `dev`), so the
/// shape stays identical to the Python module.
pub fn build_user_agent() -> String {
    let py_version = env::var("QQ_PY_VERSION").unwrap_or_else(|_| "3.11.0".to_string());
    let os_name = std::env::consts::OS.to_lowercase();
    let hermes_version = option_env!("CARGO_PKG_VERSION").unwrap_or("dev");
    format!(
        "QQBotAdapter/{QQBOT_VERSION} (Python/{py_version}; {os_name}; Hermes/{hermes_version})"
    )
}

/// Standard HTTP headers for QQBot portal API requests.
///
/// `q.qq.com` requires `Accept: application/json`; without it the server
/// returns a JavaScript anti-bot challenge page.
pub fn get_api_headers() -> Vec<(String, String)> {
    vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Accept".to_string(), "application/json".to_string()),
        ("User-Agent".to_string(), build_user_agent()),
    ]
}

// ---------------------------------------------------------------------------
// Config coercion — mirrors `utils.coerce_list`.
// ---------------------------------------------------------------------------

/// Coerce a config JSON value into a trimmed string list.
///
/// Accepts comma-separated strings, arrays, or single scalar values — matching
/// the Python `coerce_list`.
pub fn coerce_list(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => s
            .split(',')
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(|item| item.to_string())
            .collect(),
        Some(Value::Array(items)) => items
            .iter()
            .map(value_to_plain_string)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(other) => {
            let s = value_to_plain_string(other);
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
    }
}

/// Render a JSON scalar to its Python `str(...)` form (used by `coerce_list`).
fn value_to_plain_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

/// Read a string from a config `extra` map, falling back to an env var, then a
/// default — matching `str(extra.get(key) or os.getenv(env, "")).strip()`.
fn extra_str_or_env(extra: &Value, key: &str, env_key: &str) -> String {
    let from_extra = extra.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(value_to_plain_string(other)),
    });
    let raw = match from_extra {
        Some(s) if !s.is_empty() => s,
        _ => env::var(env_key).unwrap_or_default(),
    };
    raw.trim().to_string()
}

// ---------------------------------------------------------------------------
// Configuration extracted from `PlatformConfig.extra`.
// ---------------------------------------------------------------------------

/// Resolved QQ adapter configuration (the `extra` block of `PlatformConfig`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QQConfig {
    pub app_id: String,
    pub client_secret: String,
    pub markdown_support: bool,
    pub dm_policy: String,
    pub allow_from: Vec<String>,
    pub group_policy: String,
    pub group_allow_from: Vec<String>,
}

impl QQConfig {
    /// Build a config from the platform's `extra` JSON object, applying env-var
    /// fallbacks exactly as the Python `__init__` does.
    pub fn from_extra(extra: &Value) -> Self {
        let extra = if extra.is_object() {
            extra.clone()
        } else {
            Value::Object(Default::default())
        };

        let app_id = extra_str_or_env(&extra, "app_id", "QQ_APP_ID");
        let client_secret = extra_str_or_env(&extra, "client_secret", "QQ_CLIENT_SECRET");

        // `bool(extra.get("markdown_support", True))` — default True; any falsey
        // JSON value (false, null, 0, "", []) maps to false.
        let markdown_support = match extra.get("markdown_support") {
            None => true,
            Some(v) => json_truthy(v),
        };

        let dm_policy = string_field(&extra, "dm_policy", "open")
            .trim()
            .to_lowercase();
        let allow_from = coerce_list(extra.get("allow_from").or_else(|| extra.get("allowFrom")));
        let group_policy = string_field(&extra, "group_policy", "open")
            .trim()
            .to_lowercase();
        let group_allow_from = coerce_list(
            extra
                .get("group_allow_from")
                .or_else(|| extra.get("groupAllowFrom")),
        );

        QQConfig {
            app_id,
            client_secret,
            markdown_support,
            dm_policy,
            allow_from,
            group_policy,
            group_allow_from,
        }
    }
}

/// `str(extra.get(key, default)).strip()`-style string read with a default.
fn string_field(extra: &Value, key: &str, default: &str) -> String {
    match extra.get(key) {
        None | Some(Value::Null) => default.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => value_to_plain_string(other),
    }
}

/// Python truthiness for a JSON value (used for `markdown_support`).
fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Message types — mirrors the gateway `MessageType` subset the QQ adapter uses.
// ---------------------------------------------------------------------------

/// Inbound message classification, matching the gateway `MessageType` variants
/// that the QQ adapter produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Text,
    Photo,
    Voice,
    Video,
}

/// Determine the [`MessageType`] from inbound attachment content types.
///
/// Faithful port of `_detect_message_type`.
pub fn detect_message_type(media_urls: &[String], media_types: &[String]) -> MessageType {
    if media_urls.is_empty() {
        return MessageType::Text;
    }
    if media_types.is_empty() {
        return MessageType::Photo;
    }
    let first_type = media_types[0].to_lowercase();
    if first_type.contains("audio") || first_type.contains("voice") || first_type.contains("silk")
    {
        return MessageType::Voice;
    }
    if first_type.contains("video") {
        return MessageType::Video;
    }
    if first_type.contains("image") || first_type.contains("photo") {
        return MessageType::Photo;
    }
    MessageType::Text
}

// ---------------------------------------------------------------------------
// Chat-type routing.
// ---------------------------------------------------------------------------

/// Determine whether a string is an http(s) URL — port of `_is_url`.
pub fn is_url(source: &str) -> bool {
    let lower = source.trim().to_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

// ---------------------------------------------------------------------------
// Dedup window — port of `_is_duplicate` / `_seen_messages`.
// ---------------------------------------------------------------------------

/// Sliding-window deduplication of inbound message ids.
///
/// Mirrors the Python `_seen_messages` dict and `_is_duplicate`: once the cache
/// exceeds [`DEDUP_MAX_SIZE`] entries, entries older than
/// [`DEDUP_WINDOW_SECONDS`] are evicted on the next check.
#[derive(Debug, Default)]
pub struct SeenMessages {
    seen: HashMap<String, f64>,
}

impl SeenMessages {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return `true` if `msg_id` was seen before; otherwise record it and
    /// return `false`. `now` is the current unix time in seconds.
    pub fn is_duplicate_at(&mut self, msg_id: &str, now: f64) -> bool {
        if self.seen.len() > DEDUP_MAX_SIZE {
            let cutoff = now - DEDUP_WINDOW_SECONDS as f64;
            self.seen.retain(|_, ts| *ts > cutoff);
        }
        if self.seen.contains_key(msg_id) {
            return true;
        }
        self.seen.insert(msg_id.to_string(), now);
        false
    }

    /// Convenience wrapper using the current wall-clock time.
    pub fn is_duplicate(&mut self, msg_id: &str) -> bool {
        self.is_duplicate_at(msg_id, unix_now())
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Current unix time as fractional seconds.
fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Message-seq generation — port of `_next_msg_seq`.
// ---------------------------------------------------------------------------

/// Generate a message sequence number in the 0..65535 range.
///
/// Faithful port of `_next_msg_seq`: `(int(time())%1e8 ^ rand16) % 65536`.
/// `rand_hex4` is the value of the first 4 hex chars of a UUID (0..=0xFFFF);
/// callers pass a fresh random value to match Python's `uuid4().hex[:4]`.
pub fn next_msg_seq_with(now_secs: u64, rand_hex4: u32) -> u32 {
    let time_part = (now_secs % 100_000_000) as u32;
    (time_part ^ rand_hex4) % 65536
}

/// `_next_msg_seq` using the current time and a freshly generated 16-bit random.
pub fn next_msg_seq() -> u32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    next_msg_seq_with(now, rand_u16())
}

/// Generate a pseudo-random 16-bit value (mirrors `uuid4().hex[:4]` width).
fn rand_u16() -> u32 {
    // Derive entropy from the high-resolution clock; matches the *range* of
    // the Python value (0..=0xFFFF) without pulling in an RNG crate.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Mix the nanos so successive calls within the same second still differ.
    let mixed = nanos
        .wrapping_mul(2_654_435_761)
        .rotate_left(13)
        .wrapping_add(0x9E37_79B9);
    mixed & 0xFFFF
}

// ---------------------------------------------------------------------------
// Timestamp parsing — port of `_parse_qq_timestamp`.
// ---------------------------------------------------------------------------

/// Parse a QQ API timestamp (ISO 8601 string or integer milliseconds).
///
/// The QQ API changed from integer milliseconds to ISO 8601 strings; this
/// handles both, falling back to "now" on failure. Faithful port of
/// `_parse_qq_timestamp`.
pub fn parse_qq_timestamp(raw: &str) -> DateTime<Utc> {
    if raw.is_empty() {
        return Utc::now();
    }
    // 1. ISO 8601 (datetime.fromisoformat). Try RFC3339 first, then a few
    //    common variants Python's fromisoformat accepts.
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return dt.with_timezone(&Utc);
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S") {
        return DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
    }
    // 2. Integer milliseconds since epoch.
    if let Ok(ms) = raw.trim().parse::<i64>() {
        if let Some(dt) = DateTime::<Utc>::from_timestamp(ms / 1000, ((ms % 1000) * 1_000_000) as u32)
        {
            return dt;
        }
    }
    Utc::now()
}

// ---------------------------------------------------------------------------
// @-mention stripping — port of `_strip_at_mention`.
// ---------------------------------------------------------------------------

/// Strip a leading `@mention` prefix from group message content.
///
/// Faithful port of `_strip_at_mention`, which applies `re.sub(r"^@\S+\s*", "")`
/// to the stripped content.
pub fn strip_at_mention(content: &str) -> String {
    let trimmed = content.trim();
    let bytes = trimmed.as_bytes();
    if bytes.first() != Some(&b'@') {
        return trimmed.to_string();
    }
    // Match `@` followed by one-or-more non-whitespace, then trailing whitespace.
    let chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 1; // skip '@'
    let mut saw_nonspace = false;
    while idx < chars.len() && !chars[idx].is_whitespace() {
        saw_nonspace = true;
        idx += 1;
    }
    if !saw_nonspace {
        // `@` alone with no following \S — regex `\S+` requires at least one.
        return trimmed.to_string();
    }
    while idx < chars.len() && chars[idx].is_whitespace() {
        idx += 1;
    }
    chars[idx..].iter().collect()
}

// ---------------------------------------------------------------------------
// ACL policies — port of `_is_dm_allowed`, `_is_group_allowed`, `_entry_matches`.
// ---------------------------------------------------------------------------

/// Case-insensitive allowlist match, with `*` acting as a wildcard.
///
/// Faithful port of `_entry_matches`.
pub fn entry_matches(entries: &[String], target: &str) -> bool {
    let normalized_target = target.trim().to_lowercase();
    for entry in entries {
        let normalized = entry.trim().to_lowercase();
        if normalized == "*" || normalized == normalized_target {
            return true;
        }
    }
    false
}

/// Whether a DM from `user_id` is permitted under `dm_policy`/`allow_from`.
pub fn is_dm_allowed(dm_policy: &str, allow_from: &[String], user_id: &str) -> bool {
    if dm_policy == "disabled" {
        return false;
    }
    if dm_policy == "allowlist" {
        return entry_matches(allow_from, user_id);
    }
    true
}

/// Whether a group message in `group_id` is permitted under
/// `group_policy`/`group_allow_from`. The `user_id` argument is accepted to
/// mirror the Python signature (the allowlist is keyed on the group id).
pub fn is_group_allowed(
    group_policy: &str,
    group_allow_from: &[String],
    group_id: &str,
    _user_id: &str,
) -> bool {
    if group_policy == "disabled" {
        return false;
    }
    if group_policy == "allowlist" {
        return entry_matches(group_allow_from, group_id);
    }
    true
}

// ---------------------------------------------------------------------------
// Voice / audio attachment classification.
// ---------------------------------------------------------------------------

const VOICE_EXTENSIONS: &[&str] = &[
    ".silk", ".amr", ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".speex", ".flac",
];

/// Check whether an attachment is a voice/audio message.
///
/// Faithful port of `_is_voice_content_type`.
pub fn is_voice_content_type(content_type: &str, filename: &str) -> bool {
    let ct = content_type.trim().to_lowercase();
    let fn_l = filename.trim().to_lowercase();
    if ct == "voice" || ct.starts_with("audio/") {
        return true;
    }
    VOICE_EXTENSIONS.iter().any(|ext| fn_l.ends_with(ext))
}

/// Guess a file extension from magic bytes — faithful port of
/// `_guess_ext_from_data`. Defaults to `.amr` (QQ's most common voice format).
pub fn guess_ext_from_data(data: &[u8]) -> &'static str {
    // Python: data[:9] == b"#!SILK_V3" or data[:5] == b"#!SILK".
    // Note b"#!SILK" is 6 bytes, so `data[:5] == b"#!SILK"` is always False in
    // Python (slice length 5 != literal length 6); reproduced faithfully by
    // only matching the 9-byte SILK_V3 header here.
    if data.len() >= 9 && &data[..9] == b"#!SILK_V3" {
        return ".silk";
    }
    if data.len() >= 2 && &data[..2] == b"\x02!" {
        return ".silk";
    }
    if data.len() >= 4 && &data[..4] == b"RIFF" {
        return ".wav";
    }
    if data.len() >= 4 && &data[..4] == b"fLaC" {
        return ".flac";
    }
    if data.len() >= 2
        && (&data[..2] == b"\xff\xfb" || &data[..2] == b"\xff\xf3" || &data[..2] == b"\xff\xf2")
    {
        return ".mp3";
    }
    if data.len() >= 4 && (&data[..4] == b"\x30\x26\xb2\x75" || &data[..4] == b"\x4f\x67\x67\x53") {
        return ".ogg";
    }
    if data.len() >= 4 && (&data[..4] == b"\x00\x00\x00\x20" || &data[..4] == b"\x00\x00\x00\x1c") {
        return ".amr";
    }
    ".amr"
}

/// Check whether bytes look like a SILK audio file — port of `_looks_like_silk`.
///
/// Mirrors the Python check `data[:4] == b"#!SILK" or data[:2] == b"\x02!"
/// or data[:9] == b"#!SILK_V3"`. Note the first comparison can never match a
/// 4-byte slice against a 6-byte literal, so it is reproduced as `false` for
/// fidelity but the SILK header is still caught by the 9-byte check.
pub fn looks_like_silk(data: &[u8]) -> bool {
    // data[:4] == b"#!SILK" is always False in Python (len mismatch).
    let by9 = data.len() >= 9 && &data[..9] == b"#!SILK_V3";
    let by2 = data.len() >= 2 && &data[..2] == b"\x02!";
    by9 || by2
}

// ---------------------------------------------------------------------------
// STT configuration resolution — port of `_resolve_stt_config`.
// ---------------------------------------------------------------------------

/// Resolved speech-to-text backend configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

/// Resolve STT backend configuration from config `extra` / environment.
///
/// Faithful port of `_resolve_stt_config` (priority: plugin `stt` config →
/// `QQ_STT_*` env vars → `None`).
pub fn resolve_stt_config(extra: &Value) -> Option<SttConfig> {
    // 1. Plugin-specific STT config.
    if let Some(stt_cfg) = extra.get("stt").and_then(|v| v.as_object()) {
        let enabled = stt_cfg.get("enabled");
        let disabled = matches!(enabled, Some(Value::Bool(false)));
        if !disabled {
            let base_url = stt_cfg
                .get("baseUrl")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| stt_cfg.get("base_url").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();
            let api_key = stt_cfg
                .get("apiKey")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| stt_cfg.get("api_key").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();
            let model = stt_cfg
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            if !base_url.is_empty() && !api_key.is_empty() {
                return Some(SttConfig {
                    base_url: base_url.trim_end_matches('/').to_string(),
                    api_key,
                    model: if model.is_empty() {
                        "whisper-1".to_string()
                    } else {
                        model
                    },
                });
            }
            // Provider-only config.
            if !api_key.is_empty() {
                let provider = stt_cfg
                    .get("provider")
                    .and_then(Value::as_str)
                    .unwrap_or("zai");
                let provider_base = match provider {
                    "zai" | "glm" => "https://open.bigmodel.cn/api/coding/paas/v4",
                    "openai" => "https://api.openai.com/v1",
                    _ => "",
                };
                if !provider_base.is_empty() {
                    let default_model = if provider == "zai" || provider == "glm" {
                        "glm-asr"
                    } else {
                        "whisper-1"
                    };
                    return Some(SttConfig {
                        base_url: provider_base.to_string(),
                        api_key,
                        model: if model.is_empty() {
                            default_model.to_string()
                        } else {
                            model
                        },
                    });
                }
            }
        }
    }

    // 2. QQ-specific env vars.
    let qq_stt_key = env::var("QQ_STT_API_KEY").unwrap_or_default();
    if !qq_stt_key.is_empty() {
        let base_url = env::var("QQ_STT_BASE_URL")
            .unwrap_or_else(|_| "https://open.bigmodel.cn/api/coding/paas/v4".to_string());
        let model = env::var("QQ_STT_MODEL").unwrap_or_else(|_| "glm-asr".to_string());
        return Some(SttConfig {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: qq_stt_key,
            model,
        });
    }

    None
}

/// Parse an STT transcription response.
///
/// Faithful port of the parsing block in `_call_stt`: prefers the
/// Zhipu/GLM `choices[0].message.content` shape, then the OpenAI/Whisper
/// `text` field; returns `None` if neither yields non-empty text.
pub fn parse_stt_response(result: &Value) -> Option<String> {
    if let Some(choices) = result.get("choices").and_then(Value::as_array) {
        if let Some(first) = choices.first() {
            if let Some(content) = first
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
            {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    if let Some(text) = result.get("text").and_then(Value::as_str) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Token + gateway response parsing.
// ---------------------------------------------------------------------------

/// A parsed access-token response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenResponse {
    pub access_token: String,
    pub expires_in: i64,
}

/// JSON body sent to the token endpoint.
pub fn token_request_body(app_id: &str, client_secret: &str) -> Value {
    json!({"appId": app_id, "clientSecret": client_secret})
}

/// Parse the access-token response — port of the body of `_ensure_token`.
///
/// Returns an error string if `access_token` is missing. `expires_in` defaults
/// to 7200 when absent or unparseable.
pub fn parse_token_response(data: &Value) -> Result<TokenResponse, String> {
    let token = data.get("access_token").and_then(Value::as_str);
    let token = match token {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => {
            return Err(format!(
                "QQ Bot token response missing access_token: {data}"
            ));
        }
    };
    let expires_in = data
        .get("expires_in")
        .and_then(|v| match v {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse::<i64>().ok(),
            _ => None,
        })
        .unwrap_or(7200);
    Ok(TokenResponse {
        access_token: token,
        expires_in,
    })
}

/// Parse the gateway-URL response — port of the body of `_get_gateway_url`.
pub fn parse_gateway_response(data: &Value) -> Result<String, String> {
    match data.get("url").and_then(Value::as_str) {
        Some(url) if !url.is_empty() => Ok(url.to_string()),
        _ => Err(format!("QQ Bot gateway response missing url: {data}")),
    }
}

/// Authorization header value for REST/gateway calls (`QQBot <token>`).
pub fn auth_header(token: &str) -> String {
    format!("QQBot {token}")
}

// ---------------------------------------------------------------------------
// WebSocket payloads — identify / resume / heartbeat / dispatch routing.
// ---------------------------------------------------------------------------

/// Compose the Identify intents bitmask used by `_send_identify`:
/// `(1 << 25) | (1 << 30) | (1 << 12)`.
pub const IDENTIFY_INTENTS: i64 = (1 << 25) | (1 << 30) | (1 << 12);

/// Build the op-2 Identify payload — faithful port of `_send_identify`.
pub fn identify_payload(token: &str) -> Value {
    json!({
        "op": 2,
        "d": {
            "token": auth_header(token),
            "intents": IDENTIFY_INTENTS,
            "shard": [0, 1],
            "properties": {
                "$os": "macOS",
                "$browser": "hermes-agent",
                "$device": "hermes-agent",
            },
        },
    })
}

/// Build the op-6 Resume payload — faithful port of `_send_resume`.
pub fn resume_payload(token: &str, session_id: Option<&str>, seq: Option<i64>) -> Value {
    json!({
        "op": 6,
        "d": {
            "token": auth_header(token),
            "session_id": session_id,
            "seq": seq,
        },
    })
}

/// Build the op-1 Heartbeat payload — faithful port of `_heartbeat_loop`.
pub fn heartbeat_payload(last_seq: Option<i64>) -> Value {
    json!({"op": 1, "d": last_seq})
}

/// Compute the heartbeat send interval from a Hello (`op 10`) payload.
///
/// Returns the server `heartbeat_interval` (ms, default 30000) scaled to
/// seconds at 80% — faithful port of the `op == 10` branch in
/// `_dispatch_payload`.
pub fn hello_interval_seconds(d: &Value) -> f64 {
    let interval_ms = d
        .get("heartbeat_interval")
        .and_then(Value::as_f64)
        .unwrap_or(30000.0);
    interval_ms / 1000.0 * 0.8
}

/// What a dispatched WebSocket payload represents, after `_dispatch_payload`
/// classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchAction {
    /// op 10 — Hello; reply with Resume if a session exists, else Identify.
    Hello { resume: bool },
    /// op 0, t == "READY".
    Ready,
    /// op 0, t == "RESUMED".
    Resumed,
    /// op 0 with a message-create dispatch type.
    Message(String),
    /// op 0 with an unhandled dispatch type.
    UnhandledDispatch(String),
    /// op 11 — Heartbeat ACK.
    HeartbeatAck,
    /// Any other / unknown op.
    Unknown,
}

const MESSAGE_DISPATCH_TYPES: &[&str] = &[
    "C2C_MESSAGE_CREATE",
    "GROUP_AT_MESSAGE_CREATE",
    "DIRECT_MESSAGE_CREATE",
    "GUILD_MESSAGE_CREATE",
    "GUILD_AT_MESSAGE_CREATE",
];

/// Result of feeding a payload through [`classify_payload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchResult {
    /// Updated `last_seq` (if the payload carried a newer `s`).
    pub updated_last_seq: Option<i64>,
    pub action: DispatchAction,
}

/// Classify an inbound WebSocket payload and compute the next `last_seq`.
///
/// Faithful port of the routing in `_dispatch_payload`. `session_id` and
/// `current_last_seq` reflect adapter state needed to decide Resume vs Identify.
pub fn classify_payload(
    payload: &Value,
    session_id: Option<&str>,
    current_last_seq: Option<i64>,
) -> DispatchResult {
    let op = payload.get("op").and_then(Value::as_i64);
    let t = payload.get("t").and_then(Value::as_str);
    let s = payload.get("s").and_then(Value::as_i64);
    let d = payload.get("d");

    // Update last_seq if `s` is an int and strictly greater (or none yet).
    let mut last_seq = current_last_seq;
    let mut updated_last_seq = None;
    if let Some(s_val) = s {
        if current_last_seq.is_none() || s_val > current_last_seq.unwrap() {
            last_seq = Some(s_val);
            updated_last_seq = Some(s_val);
        }
    }

    // op 10 = Hello.
    if op == Some(10) {
        // Resume if we have both a session id and a known seq; else Identify.
        let resume = session_id.is_some() && last_seq.is_some();
        return DispatchResult {
            updated_last_seq,
            action: DispatchAction::Hello { resume },
        };
    }

    // op 0 = Dispatch.
    if op == Some(0) {
        if let Some(t_val) = t.filter(|s| !s.is_empty()) {
            let action = if t_val == "READY" {
                DispatchAction::Ready
            } else if t_val == "RESUMED" {
                DispatchAction::Resumed
            } else if MESSAGE_DISPATCH_TYPES.contains(&t_val) {
                DispatchAction::Message(t_val.to_string())
            } else {
                DispatchAction::UnhandledDispatch(t_val.to_string())
            };
            return DispatchResult {
                updated_last_seq,
                action,
            };
        }
    }

    // op 11 = Heartbeat ACK.
    if op == Some(11) {
        return DispatchResult {
            updated_last_seq,
            action: DispatchAction::HeartbeatAck,
        };
    }

    let _ = d;
    DispatchResult {
        updated_last_seq,
        action: DispatchAction::Unknown,
    }
}

/// Extract the `session_id` from a READY dispatch payload — port of
/// `_handle_ready`.
pub fn ready_session_id(d: &Value) -> Option<String> {
    d.get("session_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// WebSocket close-code handling — port of the `_listen_loop` close branch.
// ---------------------------------------------------------------------------

/// What the reconnect loop should do for a given WebSocket close code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseAction {
    /// Fatal — stop reconnecting (4914 offline/sandbox, 4915 banned).
    Fatal { banned: bool },
    /// Rate limited (4008) — wait `RATE_LIMIT_DELAY` then reconnect.
    RateLimited,
    /// Token invalid (4004) — clear token, then reconnect.
    RefreshToken,
    /// Session invalid — clear session and re-identify, then reconnect.
    ClearSession,
    /// Plain reconnect.
    Reconnect,
}

const SESSION_ERROR_CODES: &[i64] = &[
    4006, 4007, 4009, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912,
    4913,
];

/// Map a WebSocket close code to the reconnect-loop action, faithfully
/// reproducing the priority order in `_listen_loop`.
pub fn close_code_action(code: Option<i64>) -> CloseAction {
    match code {
        Some(4914) => CloseAction::Fatal { banned: false },
        Some(4915) => CloseAction::Fatal { banned: true },
        Some(4008) => CloseAction::RateLimited,
        Some(4004) => CloseAction::RefreshToken,
        Some(c) if SESSION_ERROR_CODES.contains(&c) => CloseAction::ClearSession,
        _ => CloseAction::Reconnect,
    }
}

// ---------------------------------------------------------------------------
// Attachment URL normalisation — port of the loop body in `_process_attachments`.
// ---------------------------------------------------------------------------

/// Normalise an attachment URL: prefix protocol-relative `//host` with
/// `https:`; return `None` when empty (the attachment is skipped).
///
/// Mirrors the URL-handling block in `_process_attachments` / `_stt_voice_attachment`.
pub fn normalize_attachment_url(url_raw: &str) -> Option<String> {
    let trimmed = url_raw.trim();
    if let Some(rest) = trimmed.strip_prefix("//") {
        Some(format!("https://{rest}"))
    } else if !trimmed.is_empty() {
        Some(trimmed.to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Outbound body construction.
// ---------------------------------------------------------------------------

/// Truncate `content` to at most [`MAX_MESSAGE_LENGTH`] characters
/// (`content[:MAX_MESSAGE_LENGTH]`), counting by Unicode scalar values to
/// match Python string slicing.
pub fn truncate_to_max(content: &str) -> String {
    content.chars().take(MAX_MESSAGE_LENGTH).collect()
}

/// Build the C2C/group text message body — faithful port of `_build_text_body`.
///
/// `msg_seq` is supplied by the caller (computed via [`next_msg_seq`]). When
/// `markdown_support` is false and `reply_to` is set, a `message_reference` is
/// added (markdown mode relies on `msg_id`, attached by the caller).
pub fn build_text_body(
    content: &str,
    reply_to: Option<&str>,
    markdown_support: bool,
    msg_seq: u32,
) -> Value {
    let truncated = truncate_to_max(content);
    let mut body = if markdown_support {
        json!({
            "markdown": {"content": truncated},
            "msg_type": MSG_TYPE_MARKDOWN,
            "msg_seq": msg_seq,
        })
    } else {
        json!({
            "content": truncated,
            "msg_type": MSG_TYPE_TEXT,
            "msg_seq": msg_seq,
        })
    };

    if let Some(reply) = reply_to {
        if !markdown_support {
            body["message_reference"] = json!({"message_id": reply});
        }
    }

    body
}

/// Build a guild-channel text body — port of `_send_guild_text` body shape.
pub fn build_guild_text_body(content: &str, reply_to: Option<&str>) -> Value {
    let mut body = json!({"content": truncate_to_max(content)});
    if let Some(reply) = reply_to {
        body["msg_id"] = json!(reply);
    }
    body
}

/// Build a media (`msg_type 7`) message body — port of the send block in
/// `_send_media`.
pub fn build_media_body(
    file_info: &str,
    caption: Option<&str>,
    reply_to: Option<&str>,
    msg_seq: u32,
) -> Value {
    let mut body = json!({
        "msg_type": MSG_TYPE_MEDIA,
        "media": {"file_info": file_info},
        "msg_seq": msg_seq,
    });
    if let Some(cap) = caption.filter(|c| !c.is_empty()) {
        body["content"] = json!(truncate_to_max(cap));
    }
    if let Some(reply) = reply_to {
        body["msg_id"] = json!(reply);
    }
    body
}

/// Build the input-notify (typing) body — port of the body in `send_typing`.
pub fn build_input_notify_body(msg_id: &str, msg_seq: u32, input_seconds: i64) -> Value {
    json!({
        "msg_type": MSG_TYPE_INPUT_NOTIFY,
        "msg_id": msg_id,
        "input_notify": {
            "input_type": 1,
            "input_second": input_seconds,
        },
        "msg_seq": msg_seq,
    })
}

/// Build the media-upload request body — port of the body in `_upload_media`.
///
/// Either `url` or `file_data` should be set (the Python code prefers `url`).
/// `file_name` is included only for `file_type == MEDIA_TYPE_FILE`.
pub fn build_upload_body(
    file_type: i32,
    srv_send_msg: bool,
    url: Option<&str>,
    file_data: Option<&str>,
    file_name: Option<&str>,
) -> Value {
    let mut body = json!({
        "file_type": file_type,
        "srv_send_msg": srv_send_msg,
    });
    if let Some(u) = url {
        body["url"] = json!(u);
    } else if let Some(fd) = file_data {
        body["file_data"] = json!(fd);
    }
    if file_type == MEDIA_TYPE_FILE {
        if let Some(name) = file_name {
            body["file_name"] = json!(name);
        }
    }
    body
}

/// The REST path for a media upload — port of `_upload_media` path selection.
pub fn upload_media_path(target_type: &str, target_id: &str) -> String {
    if target_type == "c2c" {
        format!("/v2/users/{target_id}/files")
    } else {
        format!("/v2/groups/{target_id}/files")
    }
}

/// The REST path for sending a message to a chat of the given type.
///
/// Mirrors the path selection scattered across `_send_c2c_text`,
/// `_send_group_text`, `_send_guild_text`, `_send_media`.
pub fn message_path(chat_type: &str, chat_id: &str) -> String {
    match chat_type {
        "c2c" => format!("/v2/users/{chat_id}/messages"),
        "guild" => format!("/channels/{chat_id}/messages"),
        // group (and the c2c/group media branch) use the groups path.
        _ => format!("/v2/groups/{chat_id}/messages"),
    }
}

// ---------------------------------------------------------------------------
// API request / response.
// ---------------------------------------------------------------------------

/// Authorization headers for a REST API call — port of `_api_request` headers.
pub fn api_request_headers(token: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".to_string(), auth_header(token)),
        ("Content-Type".to_string(), "application/json".to_string()),
        ("User-Agent".to_string(), build_user_agent()),
    ]
}

/// Inspect an API response, raising the Python-style error for status >= 400.
///
/// Faithful port of the error-handling tail of `_api_request`: builds
/// `QQ Bot API error [<status>] <path>: <message-or-body>`.
pub fn check_api_response(status: u16, path: &str, data: &Value) -> Result<Value, String> {
    if status >= 400 {
        let detail = match data.get("message") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => data.to_string(),
        };
        return Err(format!("QQ Bot API error [{status}] {path}: {detail}"));
    }
    Ok(data.clone())
}

/// Whether an upload error message should be retried, per `_upload_media`.
///
/// Python: errors containing any of `400`, `401`, `Invalid`, `timeout`,
/// `Timeout` are NOT retried (re-raised); everything else is retried.
pub fn upload_error_is_permanent(err_msg: &str) -> bool {
    ["400", "401", "Invalid", "timeout", "Timeout"]
        .iter()
        .any(|kw| err_msg.contains(kw))
}

/// Whether a send error is permanent (don't retry) — port of `_send_chunk`.
///
/// Python lowercases the error and checks for `invalid`, `forbidden`,
/// `not found`, `bad request`.
pub fn send_error_is_permanent(err_msg: &str) -> bool {
    let lower = err_msg.to_lowercase();
    ["invalid", "forbidden", "not found", "bad request"]
        .iter()
        .any(|kw| lower.contains(kw))
}

/// Whether a final send error should be reported as retryable — port of the
/// tail of `_send_chunk` (`not any(... in (invalid, forbidden, not found))`).
pub fn send_error_is_retryable(err_msg: &str) -> bool {
    let lower = err_msg.to_lowercase();
    !["invalid", "forbidden", "not found"]
        .iter()
        .any(|kw| lower.contains(kw))
}

/// Extract the message id from a send response, mirroring
/// `str(data.get("id", uuid4().hex[:12]))`. When `id` is absent, the caller
/// supplies a fallback (a fresh 12-char hex token).
pub fn parse_send_response(data: &Value, fallback_id: &str) -> SendResult {
    let msg_id = match data.get("id") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => value_to_plain_string(other),
        None => fallback_id.to_string(),
    };
    SendResult::ok(msg_id, Some(data.clone()))
}

/// Exponential backoff delay (seconds) for the Nth send retry attempt
/// (`1.0 * 2**attempt`), per `_send_chunk`.
pub fn send_retry_delay(attempt: u32) -> f64 {
    1.0 * (2f64).powi(attempt as i32)
}

/// Backoff delay (seconds) for the Nth upload retry attempt
/// (`1.5 * (attempt + 1)`), per `_upload_media`.
pub fn upload_retry_delay(attempt: u32) -> f64 {
    1.5 * (attempt as f64 + 1.0)
}

// ---------------------------------------------------------------------------
// Adapter state container — the deterministic per-instance state of QQAdapter.
// ---------------------------------------------------------------------------

/// Per-instance QQ adapter state (the portable subset of the Python
/// `QQAdapter` fields and the methods that operate purely on them).
#[derive(Debug, Default)]
pub struct QQAdapter {
    pub config: QQConfig,
    /// chat_id → "c2c" | "group" | "guild" | "dm".
    pub chat_type_map: HashMap<String, String>,
    /// chat_id → last inbound message id (used by `send_typing`).
    pub last_msg_id: HashMap<String, String>,
    /// chat_id → last `send_typing` timestamp (debounce).
    pub typing_sent_at: HashMap<String, f64>,
    pub seen_messages: SeenMessages,
    pub session_id: Option<String>,
    pub last_seq: Option<i64>,
    pub heartbeat_interval: f64,
}

impl QQAdapter {
    /// The `input_notify` duration reported to QQ (seconds).
    pub const TYPING_INPUT_SECONDS: i64 = 60;
    /// Refresh typing before it expires (seconds).
    pub const TYPING_DEBOUNCE_SECONDS: f64 = 50.0;
    /// QQ Bot API does not support editing sent messages.
    pub const SUPPORTS_MESSAGE_EDITING: bool = false;
    pub const MAX_MESSAGE_LENGTH: usize = MAX_MESSAGE_LENGTH;

    /// Construct from a `PlatformConfig.extra` JSON object.
    pub fn from_extra(extra: &Value) -> Self {
        QQAdapter {
            config: QQConfig::from_extra(extra),
            heartbeat_interval: 30.0,
            ..Default::default()
        }
    }

    /// Platform display name.
    pub fn name(&self) -> &'static str {
        "QQBot"
    }

    /// Log prefix including app_id for multi-instance disambiguation —
    /// port of the `_log_tag` property.
    pub fn log_tag(&self) -> String {
        if self.config.app_id.is_empty() {
            "QQBot".to_string()
        } else {
            format!("QQBot:{}", self.config.app_id)
        }
    }

    /// Determine chat type from stored inbound metadata, defaulting to `c2c` —
    /// port of `_guess_chat_type`.
    pub fn guess_chat_type(&self, chat_id: &str) -> String {
        self.chat_type_map
            .get(chat_id)
            .cloned()
            .unwrap_or_else(|| "c2c".to_string())
    }

    /// ACL: whether a DM from `user_id` is permitted.
    pub fn is_dm_allowed(&self, user_id: &str) -> bool {
        is_dm_allowed(&self.config.dm_policy, &self.config.allow_from, user_id)
    }

    /// ACL: whether a group message in `group_id` is permitted.
    pub fn is_group_allowed(&self, group_id: &str, user_id: &str) -> bool {
        is_group_allowed(
            &self.config.group_policy,
            &self.config.group_allow_from,
            group_id,
            user_id,
        )
    }

    /// Format a message for QQ — port of `format_message`. Markdown mode passes
    /// content through; otherwise markdown is stripped via the shared helper.
    pub fn format_message(&self, content: &str) -> String {
        if self.config.markdown_support {
            content.to_string()
        } else {
            crate::gw_helpers::strip_markdown(content)
        }
    }

    /// Chat info heuristics — port of `get_chat_info`.
    pub fn get_chat_info(&self, chat_id: &str) -> Value {
        let chat_type = self.guess_chat_type(chat_id);
        let kind = if chat_type == "group" || chat_type == "guild" {
            "group"
        } else {
            "dm"
        };
        json!({"name": chat_id, "type": kind})
    }

    /// Whether `send_typing` should fire for `chat_id` at time `now`.
    ///
    /// Port of the gates in `send_typing`: only C2C chats, requires a known
    /// `last_msg_id`, and is debounced to once per `TYPING_DEBOUNCE_SECONDS`.
    /// Returns the originating `msg_id` to use, or `None` to skip.
    pub fn typing_should_send(&self, chat_id: &str, now: f64) -> Option<String> {
        if self.guess_chat_type(chat_id) != "c2c" {
            return None;
        }
        let msg_id = self.last_msg_id.get(chat_id)?;
        let last_sent = self.typing_sent_at.get(chat_id).copied().unwrap_or(0.0);
        if now - last_sent < Self::TYPING_DEBOUNCE_SECONDS {
            return None;
        }
        Some(msg_id.clone())
    }

    /// Record that a typing indicator was sent for `chat_id` at `now`.
    pub fn typing_mark_sent(&mut self, chat_id: &str, now: f64) {
        self.typing_sent_at.insert(chat_id.to_string(), now);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coerce_list_variants() {
        assert_eq!(coerce_list(None), Vec::<String>::new());
        assert_eq!(coerce_list(Some(&Value::Null)), Vec::<String>::new());
        assert_eq!(
            coerce_list(Some(&json!("a, b ,,c"))),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            coerce_list(Some(&json!(["x", " y ", "", "z"]))),
            vec!["x".to_string(), "y".to_string(), "z".to_string()]
        );
        assert_eq!(coerce_list(Some(&json!(42))), vec!["42".to_string()]);
    }

    #[test]
    fn config_from_extra_defaults() {
        let cfg = QQConfig::from_extra(&json!({}));
        assert!(cfg.markdown_support);
        assert_eq!(cfg.dm_policy, "open");
        assert_eq!(cfg.group_policy, "open");
        assert!(cfg.allow_from.is_empty());
    }

    #[test]
    fn config_markdown_disable_and_policies() {
        let cfg = QQConfig::from_extra(&json!({
            "markdown_support": false,
            "dm_policy": "ALLOWLIST",
            "allow_from": "openid_1, openid_2",
            "group_policy": "Disabled",
            "groupAllowFrom": ["g1"],
        }));
        assert!(!cfg.markdown_support);
        assert_eq!(cfg.dm_policy, "allowlist");
        assert_eq!(cfg.allow_from, vec!["openid_1", "openid_2"]);
        assert_eq!(cfg.group_policy, "disabled");
        assert_eq!(cfg.group_allow_from, vec!["g1"]);
    }

    #[test]
    fn config_env_fallback() {
        unsafe {
            env::set_var("QQ_APP_ID", "env-app");
            env::set_var("QQ_CLIENT_SECRET", "env-secret");
        }
        let cfg = QQConfig::from_extra(&json!({}));
        assert_eq!(cfg.app_id, "env-app");
        assert_eq!(cfg.client_secret, "env-secret");
        // extra takes priority over env
        let cfg2 = QQConfig::from_extra(&json!({"app_id": "x"}));
        assert_eq!(cfg2.app_id, "x");
        unsafe {
            env::remove_var("QQ_APP_ID");
            env::remove_var("QQ_CLIENT_SECRET");
        }
    }

    #[test]
    fn detect_message_type_rules() {
        assert_eq!(detect_message_type(&[], &[]), MessageType::Text);
        assert_eq!(
            detect_message_type(&["u".to_string()], &[]),
            MessageType::Photo
        );
        assert_eq!(
            detect_message_type(&["u".to_string()], &["audio/wav".to_string()]),
            MessageType::Voice
        );
        assert_eq!(
            detect_message_type(&["u".to_string()], &["video/mp4".to_string()]),
            MessageType::Video
        );
        assert_eq!(
            detect_message_type(&["u".to_string()], &["image/png".to_string()]),
            MessageType::Photo
        );
        assert_eq!(
            detect_message_type(&["u".to_string()], &["application/pdf".to_string()]),
            MessageType::Text
        );
    }

    #[test]
    fn acl_policies() {
        assert!(is_dm_allowed("open", &[], "anyone"));
        assert!(!is_dm_allowed("disabled", &["x".to_string()], "x"));
        assert!(is_dm_allowed(
            "allowlist",
            &["X".to_string()],
            "x"
        ));
        assert!(!is_dm_allowed("allowlist", &["y".to_string()], "x"));
        assert!(is_dm_allowed("allowlist", &["*".to_string()], "x"));

        assert!(is_group_allowed("open", &[], "g", "u"));
        assert!(!is_group_allowed("disabled", &[], "g", "u"));
        assert!(is_group_allowed(
            "allowlist",
            &["g".to_string()],
            "g",
            "u"
        ));
    }

    #[test]
    fn entry_matches_wildcard_and_case() {
        assert!(entry_matches(&["*".to_string()], "anything"));
        assert!(entry_matches(&[" ABC ".to_string()], "abc"));
        assert!(!entry_matches(&["abc".to_string()], "xyz"));
    }

    #[test]
    fn strip_at_mention_cases() {
        assert_eq!(strip_at_mention("@bot hello world"), "hello world");
        assert_eq!(strip_at_mention("  @user123   hi"), "hi");
        assert_eq!(strip_at_mention("no mention here"), "no mention here");
        assert_eq!(strip_at_mention("@ leading space"), "@ leading space");
        assert_eq!(strip_at_mention("@only"), "");
    }

    #[test]
    fn voice_content_type_detection() {
        assert!(is_voice_content_type("voice", ""));
        assert!(is_voice_content_type("audio/amr", ""));
        assert!(is_voice_content_type("", "clip.silk"));
        assert!(is_voice_content_type("", "song.MP3"));
        assert!(!is_voice_content_type("image/png", "a.png"));
    }

    #[test]
    fn guess_ext_magic_bytes() {
        assert_eq!(guess_ext_from_data(b"#!SILK_V3rest"), ".silk");
        assert_eq!(guess_ext_from_data(b"\x02!stuff"), ".silk");
        assert_eq!(guess_ext_from_data(b"RIFFxxxx"), ".wav");
        assert_eq!(guess_ext_from_data(b"fLaCxxxx"), ".flac");
        assert_eq!(guess_ext_from_data(b"\xff\xfbxx"), ".mp3");
        assert_eq!(guess_ext_from_data(b"OggSxxxx"), ".ogg");
        assert_eq!(guess_ext_from_data(b"unknown"), ".amr");
    }

    #[test]
    fn looks_like_silk_check() {
        assert!(looks_like_silk(b"#!SILK_V3 data"));
        assert!(looks_like_silk(b"\x02! data"));
        assert!(!looks_like_silk(b"RIFFxxxx"));
    }

    #[test]
    fn is_url_check() {
        assert!(is_url("http://x.com"));
        assert!(is_url("HTTPS://x.com"));
        assert!(!is_url("/local/path"));
        assert!(!is_url("ftp://x"));
    }

    #[test]
    fn normalize_attachment_url_cases() {
        assert_eq!(
            normalize_attachment_url("//cdn.qq.com/a.jpg"),
            Some("https://cdn.qq.com/a.jpg".to_string())
        );
        assert_eq!(
            normalize_attachment_url("https://x.com/a"),
            Some("https://x.com/a".to_string())
        );
        assert_eq!(normalize_attachment_url("   "), None);
    }

    #[test]
    fn dedup_window() {
        let mut seen = SeenMessages::new();
        assert!(!seen.is_duplicate_at("a", 100.0));
        assert!(seen.is_duplicate_at("a", 100.0));
        assert!(!seen.is_duplicate_at("b", 100.0));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn msg_seq_in_range() {
        for r in [0u32, 1, 0xFFFF, 12345] {
            let seq = next_msg_seq_with(1_700_000_000, r);
            assert!(seq < 65536);
        }
    }

    #[test]
    fn build_text_body_markdown() {
        let body = build_text_body("hello", None, true, 42);
        assert_eq!(body["msg_type"], json!(MSG_TYPE_MARKDOWN));
        assert_eq!(body["markdown"]["content"], json!("hello"));
        assert_eq!(body["msg_seq"], json!(42));
        assert!(body.get("content").is_none());
    }

    #[test]
    fn build_text_body_plain_with_reply() {
        let body = build_text_body("hi", Some("m1"), false, 7);
        assert_eq!(body["msg_type"], json!(MSG_TYPE_TEXT));
        assert_eq!(body["content"], json!("hi"));
        assert_eq!(body["message_reference"]["message_id"], json!("m1"));
    }

    #[test]
    fn build_text_body_markdown_reply_no_reference() {
        let body = build_text_body("hi", Some("m1"), true, 7);
        // markdown mode does NOT add message_reference
        assert!(body.get("message_reference").is_none());
    }

    #[test]
    fn build_media_body_shape() {
        let body = build_media_body("FILEINFO", Some("cap"), Some("m1"), 9);
        assert_eq!(body["msg_type"], json!(MSG_TYPE_MEDIA));
        assert_eq!(body["media"]["file_info"], json!("FILEINFO"));
        assert_eq!(body["content"], json!("cap"));
        assert_eq!(body["msg_id"], json!("m1"));
        assert_eq!(body["msg_seq"], json!(9));
    }

    #[test]
    fn build_input_notify_shape() {
        let body = build_input_notify_body("m1", 3, 60);
        assert_eq!(body["msg_type"], json!(MSG_TYPE_INPUT_NOTIFY));
        assert_eq!(body["input_notify"]["input_type"], json!(1));
        assert_eq!(body["input_notify"]["input_second"], json!(60));
    }

    #[test]
    fn upload_body_url_and_file() {
        let url_body = build_upload_body(MEDIA_TYPE_FILE, false, Some("http://x/a"), None, Some("a.bin"));
        assert_eq!(url_body["url"], json!("http://x/a"));
        assert_eq!(url_body["file_name"], json!("a.bin"));
        let data_body = build_upload_body(crate::gw_qq_constants::MEDIA_TYPE_IMAGE, true, None, Some("BASE64"), Some("ignored"));
        assert_eq!(data_body["file_data"], json!("BASE64"));
        // file_name only present for MEDIA_TYPE_FILE
        assert!(data_body.get("file_name").is_none());
    }

    #[test]
    fn paths() {
        assert_eq!(upload_media_path("c2c", "u1"), "/v2/users/u1/files");
        assert_eq!(upload_media_path("group", "g1"), "/v2/groups/g1/files");
        assert_eq!(message_path("c2c", "u1"), "/v2/users/u1/messages");
        assert_eq!(message_path("group", "g1"), "/v2/groups/g1/messages");
        assert_eq!(message_path("guild", "c1"), "/channels/c1/messages");
    }

    #[test]
    fn token_parse() {
        let r = parse_token_response(&json!({"access_token": "tok", "expires_in": 100})).unwrap();
        assert_eq!(r.access_token, "tok");
        assert_eq!(r.expires_in, 100);
        let r2 = parse_token_response(&json!({"access_token": "tok"})).unwrap();
        assert_eq!(r2.expires_in, 7200);
        assert!(parse_token_response(&json!({"foo": 1})).is_err());
    }

    #[test]
    fn gateway_parse() {
        assert_eq!(
            parse_gateway_response(&json!({"url": "wss://x"})).unwrap(),
            "wss://x"
        );
        assert!(parse_gateway_response(&json!({})).is_err());
    }

    #[test]
    fn stt_parse_formats() {
        let glm = json!({"choices": [{"message": {"content": " hi "}}]});
        assert_eq!(parse_stt_response(&glm), Some("hi".to_string()));
        let whisper = json!({"text": " hello "});
        assert_eq!(parse_stt_response(&whisper), Some("hello".to_string()));
        assert_eq!(parse_stt_response(&json!({"text": "  "})), None);
    }

    #[test]
    fn resolve_stt_plugin_config() {
        let extra = json!({"stt": {"baseUrl": "https://api/", "apiKey": "k", "model": "m"}});
        let cfg = resolve_stt_config(&extra).unwrap();
        assert_eq!(cfg.base_url, "https://api");
        assert_eq!(cfg.api_key, "k");
        assert_eq!(cfg.model, "m");
    }

    #[test]
    fn resolve_stt_provider_only() {
        let extra = json!({"stt": {"apiKey": "k", "provider": "openai"}});
        let cfg = resolve_stt_config(&extra).unwrap();
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
        assert_eq!(cfg.model, "whisper-1");
    }

    #[test]
    fn resolve_stt_disabled() {
        let extra = json!({"stt": {"enabled": false, "apiKey": "k", "baseUrl": "u"}});
        unsafe { env::remove_var("QQ_STT_API_KEY"); }
        assert!(resolve_stt_config(&extra).is_none());
    }

    #[test]
    fn classify_hello_identify_vs_resume() {
        let hello = json!({"op": 10, "d": {"heartbeat_interval": 40000}});
        let r = classify_payload(&hello, None, None);
        assert_eq!(r.action, DispatchAction::Hello { resume: false });
        let r2 = classify_payload(&hello, Some("sess"), Some(5));
        assert_eq!(r2.action, DispatchAction::Hello { resume: true });
        assert!((hello_interval_seconds(&hello["d"]) - 32.0).abs() < 1e-9);
    }

    #[test]
    fn classify_dispatch_and_seq() {
        let msg = json!({"op": 0, "t": "C2C_MESSAGE_CREATE", "s": 9, "d": {}});
        let r = classify_payload(&msg, None, Some(3));
        assert_eq!(r.updated_last_seq, Some(9));
        assert_eq!(
            r.action,
            DispatchAction::Message("C2C_MESSAGE_CREATE".to_string())
        );

        let ready = json!({"op": 0, "t": "READY", "d": {"session_id": "abc"}});
        assert_eq!(classify_payload(&ready, None, None).action, DispatchAction::Ready);
        assert_eq!(ready_session_id(&ready["d"]), Some("abc".to_string()));

        let ack = json!({"op": 11});
        assert_eq!(classify_payload(&ack, None, None).action, DispatchAction::HeartbeatAck);

        let unhandled = json!({"op": 0, "t": "SOMETHING_ELSE"});
        assert_eq!(
            classify_payload(&unhandled, None, None).action,
            DispatchAction::UnhandledDispatch("SOMETHING_ELSE".to_string())
        );
    }

    #[test]
    fn seq_not_decreasing() {
        let lower = json!({"op": 0, "t": "READY", "s": 2});
        let r = classify_payload(&lower, None, Some(5));
        assert_eq!(r.updated_last_seq, None);
    }

    #[test]
    fn close_codes() {
        assert_eq!(close_code_action(Some(4914)), CloseAction::Fatal { banned: false });
        assert_eq!(close_code_action(Some(4915)), CloseAction::Fatal { banned: true });
        assert_eq!(close_code_action(Some(4008)), CloseAction::RateLimited);
        assert_eq!(close_code_action(Some(4004)), CloseAction::RefreshToken);
        assert_eq!(close_code_action(Some(4006)), CloseAction::ClearSession);
        assert_eq!(close_code_action(Some(4913)), CloseAction::ClearSession);
        assert_eq!(close_code_action(Some(1006)), CloseAction::Reconnect);
        assert_eq!(close_code_action(None), CloseAction::Reconnect);
    }

    #[test]
    fn api_response_errors() {
        let ok = check_api_response(200, "/p", &json!({"id": "1"})).unwrap();
        assert_eq!(ok["id"], json!("1"));
        let err = check_api_response(403, "/p", &json!({"message": "forbidden"}))
            .unwrap_err();
        assert!(err.contains("[403]"));
        assert!(err.contains("forbidden"));
    }

    #[test]
    fn error_classification() {
        assert!(upload_error_is_permanent("HTTP 400 Bad"));
        assert!(upload_error_is_permanent("timeout occurred"));
        assert!(!upload_error_is_permanent("connection reset"));
        assert!(send_error_is_permanent("Invalid request"));
        assert!(send_error_is_permanent("Not Found"));
        assert!(!send_error_is_permanent("server error"));
        assert!(!send_error_is_retryable("forbidden by acl"));
        assert!(send_error_is_retryable("temporary glitch"));
    }

    #[test]
    fn send_response_parse() {
        let r = parse_send_response(&json!({"id": "msg42"}), "fb");
        assert!(r.success);
        assert_eq!(r.message_id, Some("msg42".to_string()));
        let r2 = parse_send_response(&json!({}), "fallback");
        assert_eq!(r2.message_id, Some("fallback".to_string()));
    }

    #[test]
    fn retry_delays() {
        assert!((send_retry_delay(0) - 1.0).abs() < 1e-9);
        assert!((send_retry_delay(1) - 2.0).abs() < 1e-9);
        assert!((send_retry_delay(2) - 4.0).abs() < 1e-9);
        assert!((upload_retry_delay(0) - 1.5).abs() < 1e-9);
        assert!((upload_retry_delay(1) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn timestamp_parsing() {
        let iso = parse_qq_timestamp("2024-01-02T03:04:05+00:00");
        assert_eq!(iso.timestamp(), 1704164645);
        let ms = parse_qq_timestamp("1704164645000");
        assert_eq!(ms.timestamp(), 1704164645);
        // empty falls back to ~now (just ensure it returns something valid)
        let _ = parse_qq_timestamp("");
    }

    #[test]
    fn adapter_state_helpers() {
        let mut a = QQAdapter::from_extra(&json!({"app_id": "id1"}));
        assert_eq!(a.log_tag(), "QQBot:id1");
        assert_eq!(a.name(), "QQBot");
        assert_eq!(a.guess_chat_type("unknown"), "c2c");
        a.chat_type_map.insert("g1".to_string(), "group".to_string());
        assert_eq!(a.guess_chat_type("g1"), "group");
        assert_eq!(a.get_chat_info("g1")["type"], json!("group"));
    }

    #[test]
    fn adapter_typing_debounce() {
        let mut a = QQAdapter::from_extra(&json!({}));
        a.chat_type_map.insert("u1".to_string(), "c2c".to_string());
        // no last_msg_id → skip
        assert_eq!(a.typing_should_send("u1", 100.0), None);
        a.last_msg_id.insert("u1".to_string(), "m1".to_string());
        assert_eq!(a.typing_should_send("u1", 100.0), Some("m1".to_string()));
        a.typing_mark_sent("u1", 100.0);
        // within debounce window → skip
        assert_eq!(a.typing_should_send("u1", 120.0), None);
        // after window → allowed
        assert_eq!(a.typing_should_send("u1", 200.0), Some("m1".to_string()));
        // non-c2c → skip
        a.chat_type_map.insert("g1".to_string(), "group".to_string());
        a.last_msg_id.insert("g1".to_string(), "m2".to_string());
        assert_eq!(a.typing_should_send("g1", 100.0), None);
    }

    #[test]
    fn format_message_markdown_vs_plain() {
        let md = QQAdapter::from_extra(&json!({"markdown_support": true}));
        assert_eq!(md.format_message("**x**"), "**x**");
        let plain = QQAdapter::from_extra(&json!({"markdown_support": false}));
        assert_eq!(plain.format_message("**x**"), "x");
    }

    #[test]
    fn payloads_shape() {
        let id = identify_payload("tok");
        assert_eq!(id["op"], json!(2));
        assert_eq!(id["d"]["token"], json!("QQBot tok"));
        assert_eq!(id["d"]["intents"], json!(IDENTIFY_INTENTS));
        assert_eq!(id["d"]["shard"], json!([0, 1]));

        let res = resume_payload("tok", Some("sess"), Some(7));
        assert_eq!(res["op"], json!(6));
        assert_eq!(res["d"]["session_id"], json!("sess"));
        assert_eq!(res["d"]["seq"], json!(7));

        let hb = heartbeat_payload(Some(3));
        assert_eq!(hb["op"], json!(1));
        assert_eq!(hb["d"], json!(3));
        let hb_null = heartbeat_payload(None);
        assert_eq!(hb_null["d"], Value::Null);
    }

    #[test]
    fn user_agent_shape() {
        let ua = build_user_agent();
        assert!(ua.starts_with("QQBotAdapter/1.1.0 (Python/"));
        assert!(ua.contains("Hermes/"));
    }

    #[test]
    fn truncate_respects_char_limit() {
        let long = "x".repeat(MAX_MESSAGE_LENGTH + 50);
        assert_eq!(truncate_to_max(&long).chars().count(), MAX_MESSAGE_LENGTH);
        assert_eq!(truncate_to_max("short"), "short");
    }
}
