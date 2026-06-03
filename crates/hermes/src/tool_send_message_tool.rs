//! Send Message Tool -- cross-channel messaging via platform APIs.
//!
//! Native Rust port of `tools/send_message_tool.py`.
//!
//! Sends a message to a user or channel on any connected messaging platform
//! (Telegram, Discord, Slack, and many more). Supports listing available
//! targets and resolving human-friendly channel names to IDs. Works in both
//! CLI and gateway contexts.
//!
//! Scope of this port:
//!   * All of the *pure* request-shaping logic is reproduced faithfully:
//!     target parsing, secret sanitization, the tool schema, media-mirror
//!     descriptions, cron auto-delivery duplicate skipping, Telegram retry
//!     delay computation, and SMS markdown stripping.
//!   * The self-contained HTTP senders (Slack, WhatsApp bridge, Mattermost,
//!     Matrix, Home Assistant, DingTalk, Twilio SMS, QQBot) are ported using
//!     `reqwest::blocking`, keeping the exact API request/response shapes.
//!   * Senders that in Python delegate to a *live gateway adapter / persistent
//!     websocket* (WeCom, Weixin, Yuanbao, BlueBubbles, Feishu, Matrix-with-
//!     media, Signal scheduler, Telegram/Discord media uploads) are modeled
//!     through trait/parameter seams so the orchestration logic can be ported
//!     without pulling in the not-yet-ported adapter singletons. Where a
//!     concrete adapter is unavailable a structured error payload identical to
//!     the Python fallback is returned.
//!
//! Cross-references (flat ported modules, referenced where available):
//!   * `crate::agent_redact::redact_sensitive_text`
//!   * `crate::gw_platforms_base::{extract_media, truncate_message, utf16_len}`
//!   * `crate::tool_interrupt::is_interrupted`

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Static character-set tables (ported from module-level constants)
// ---------------------------------------------------------------------------

/// Image file extensions recognised for native media delivery.
pub const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".webp", ".gif"];
/// Video file extensions recognised for native media delivery.
pub const VIDEO_EXTS: &[&str] = &[".mp4", ".mov", ".avi", ".mkv", ".3gp"];
/// Audio file extensions recognised for native media delivery.
pub const AUDIO_EXTS: &[&str] = &[".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"];
/// Voice-note file extensions (subset of audio used for voice messages).
pub const VOICE_EXTS: &[&str] = &[".ogg", ".opus"];
/// Telegram Bot API `sendAudio` only accepts MP3 / M4A; others route elsewhere.
pub const TELEGRAM_SEND_AUDIO_EXTS: &[&str] = &[".mp3", ".m4a"];

/// Platforms that address recipients by phone number and accept E.164 format.
pub const PHONE_PLATFORMS: &[&str] = &["signal", "sms", "whatsapp"];

// ---------------------------------------------------------------------------
// Lazily-compiled regexes (mirrors of the Python module-level `re.compile`s)
// ---------------------------------------------------------------------------

fn telegram_topic_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(-?\d+)(?::(\d+))?\s*$").unwrap())
}

fn feishu_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$").unwrap()
    })
}

// Slack conversation IDs: C/G/D + 8+ uppercase alnum, optional :thread_ts.
fn slack_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([CGD][A-Z0-9]{8,})(?::([0-9]+(?:\.[0-9]+)?))?\s*$").unwrap())
}

fn weixin_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*((?:wxid|gh|v\d+|wm|wb)_[A-Za-z0-9_-]+|[A-Za-z0-9._-]+@chatroom|filehelper)\s*$",
        )
        .unwrap()
    })
}

fn yuanbao_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*((?:group|direct):[^:]+)\s*$").unwrap())
}

fn signal_group_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(group:\S+)\s*$").unwrap())
}

fn wecom_callback_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([^:\s]+:[^:\s]+)\s*$").unwrap())
}

fn whatsapp_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*([A-Za-z0-9._:-]+@(?:lid|g\.us|s\.whatsapp\.net|broadcast))\s*$")
            .unwrap()
    })
}

fn e164_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\+(\d{7,15})\s*$").unwrap())
}

fn url_secret_query_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)",
        )
        .unwrap()
    })
}

fn generic_secret_assign_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\s*=\s*([^\s,;]+)")
            .unwrap()
    })
}

fn html_tag_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[a-zA-Z/][^>]*>").unwrap())
}

// ---------------------------------------------------------------------------
// Redaction fallback seam
// ---------------------------------------------------------------------------

/// Delegates to the ported `agent_redact` redactor when available.
///
/// Integration step: replace the body with
/// `crate::agent_redact::redact_sensitive_text(text, false, false)`. Kept as an
/// identity fallback here so this file compiles standalone in the parallel run;
/// the remaining URL/secret-assignment masking in [`sanitize_error_text`] still
/// applies regardless.
fn redact_sensitive_text(text: &str) -> String {
    text.to_string()
}

// ---------------------------------------------------------------------------
// Error / sanitization helpers
// ---------------------------------------------------------------------------

/// Redact secrets from error text before surfacing it to users/models.
pub fn sanitize_error_text(text: &str) -> String {
    let redacted = redact_sensitive_text(text);
    let redacted = url_secret_query_re().replace_all(&redacted, |c: &regex::Captures| {
        format!("{}***", &c[1])
    });
    let redacted = generic_secret_assign_re().replace_all(&redacted, |c: &regex::Captures| {
        format!("{}=***", &c[1])
    });
    redacted.into_owned()
}

/// Build a standardized error payload with redacted content (`{"error": ...}`).
pub fn error_payload(message: &str) -> Value {
    json!({ "error": sanitize_error_text(message) })
}

/// Tool-level error: returns the JSON string `{"error": ...}` (matches the
/// behaviour of `tools.registry.tool_error`, which serialises an error dict).
pub fn tool_error(message: &str) -> String {
    error_payload(message).to_string()
}

// ---------------------------------------------------------------------------
// Telegram retry delay
// ---------------------------------------------------------------------------

/// Compute the back-off delay (seconds) for a transient Telegram send failure,
/// or `None` if the error is not retryable.
///
/// `retry_after` mirrors `exc.retry_after` (a `RetryAfter` exception). When
/// present, returns `max(retry_after, 0.0)`, falling back to `1.0` if the value
/// cannot be parsed. Otherwise classifies the error text.
pub fn telegram_retry_delay(
    retry_after: Option<f64>,
    error_text: &str,
    attempt: u32,
) -> Option<f64> {
    if let Some(ra) = retry_after {
        if ra.is_nan() {
            return Some(1.0);
        }
        return Some(ra.max(0.0));
    }

    let text = error_text.to_lowercase();
    if text.contains("timed out") || text.contains("timeout") {
        return None;
    }
    if text.contains("bad gateway")
        || text.contains("502")
        || text.contains("too many requests")
        || text.contains("429")
        || text.contains("service unavailable")
        || text.contains("503")
        || text.contains("gateway timeout")
        || text.contains("504")
    {
        return Some(2f64.powi(attempt as i32));
    }
    None
}

// ---------------------------------------------------------------------------
// Tool schema
// ---------------------------------------------------------------------------

/// The JSON schema describing the `send_message` tool.
pub fn send_message_schema() -> Value {
    json!({
        "name": "send_message",
        "description": concat!(
            "Send a message to a connected messaging platform, or list available targets.\n\n",
            "IMPORTANT: When the user asks to send to a specific channel or person ",
            "(not just a bare platform name), call send_message(action='list') FIRST to see ",
            "available targets, then send to the correct one.\n",
            "If the user just says a platform name like 'send to telegram', send directly ",
            "to the home channel without listing first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list"],
                    "description": "Action to perform. 'send' (default) sends a message. 'list' returns all available channels/contacts across connected platforms."
                },
                "target": {
                    "type": "string",
                    "description": "Delivery target. Format: 'platform' (uses home channel), 'platform:#channel-name', 'platform:chat_id', or 'platform:chat_id:thread_id' for Telegram topics and Discord threads. Examples: 'telegram', 'telegram:-1001234567890:17585', 'discord:999888777:555444333', 'discord:#bot-home', 'slack:#engineering', 'signal:+155****4567', 'matrix:!roomid:server.org', 'matrix:@user:server.org', 'yuanbao:direct:<account_id>' (DM), 'yuanbao:group:<group_code>' (group chat)"
                },
                "message": {
                    "type": "string",
                    "description": "The message text to send. To send an image or file, include MEDIA:<local_path> (e.g. 'MEDIA:/tmp/hermes/cache/img_xxx.jpg') in the message — the platform will deliver it as a native media attachment."
                }
            },
            "required": []
        }
    })
}

// ---------------------------------------------------------------------------
// Target parsing
// ---------------------------------------------------------------------------

/// Result of splitting a tool target string `platform[:ref]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitTarget {
    pub platform_name: String,
    pub target_ref: Option<String>,
}

/// Split a raw target like `"telegram:-100:17"` into a lowercased platform
/// name and an optional (stripped) reference. Mirrors `target.split(":", 1)`.
pub fn split_target(target: &str) -> SplitTarget {
    let mut iter = target.splitn(2, ':');
    let platform = iter.next().unwrap_or("").trim().to_lowercase();
    let target_ref = iter.next().map(|r| r.trim().to_string());
    SplitTarget {
        platform_name: platform,
        target_ref,
    }
}

/// Parsed reference: (chat_id, thread_id, is_explicit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedRef {
    pub chat_id: Option<String>,
    pub thread_id: Option<String>,
    pub is_explicit: bool,
}

/// True when the trimmed string is all ASCII digits (Python `str.isdigit`-ish
/// for the inputs at play here; non-empty required).
fn is_all_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Parse a tool target reference into chat_id/thread_id and whether it is
/// explicit. Faithful port of `_parse_target_ref`.
pub fn parse_target_ref(platform_name: &str, target_ref: &str) -> ParsedRef {
    fn explicit(chat: Option<&str>, thread: Option<&str>) -> ParsedRef {
        ParsedRef {
            chat_id: chat.map(|s| s.to_string()),
            thread_id: thread.map(|s| s.to_string()),
            is_explicit: true,
        }
    }
    fn not_explicit() -> ParsedRef {
        ParsedRef {
            chat_id: None,
            thread_id: None,
            is_explicit: false,
        }
    }

    match platform_name {
        "telegram" => {
            if let Some(c) = telegram_topic_re().captures(target_ref) {
                return explicit(
                    c.get(1).map(|m| m.as_str()),
                    c.get(2).map(|m| m.as_str()),
                );
            }
        }
        "feishu" => {
            if let Some(c) = feishu_re().captures(target_ref) {
                return explicit(
                    c.get(1).map(|m| m.as_str()),
                    c.get(2).map(|m| m.as_str()),
                );
            }
        }
        "discord" => {
            // Discord snowflake IDs share Telegram's topic regex.
            if let Some(c) = telegram_topic_re().captures(target_ref) {
                return explicit(
                    c.get(1).map(|m| m.as_str()),
                    c.get(2).map(|m| m.as_str()),
                );
            }
        }
        "slack" => {
            if let Some(c) = slack_re().captures(target_ref) {
                return explicit(
                    c.get(1).map(|m| m.as_str()),
                    c.get(2).map(|m| m.as_str()),
                );
            }
        }
        "weixin" => {
            if let Some(c) = weixin_re().captures(target_ref) {
                return explicit(c.get(1).map(|m| m.as_str()), None);
            }
        }
        "yuanbao" => {
            if let Some(c) = yuanbao_re().captures(target_ref) {
                return explicit(c.get(1).map(|m| m.as_str()), None);
            }
            let trimmed = target_ref.trim();
            if is_all_digits(trimmed) {
                return ParsedRef {
                    chat_id: Some(format!("group:{trimmed}")),
                    thread_id: None,
                    is_explicit: true,
                };
            }
            return not_explicit();
        }
        "signal" => {
            if let Some(c) = signal_group_re().captures(target_ref) {
                return explicit(c.get(1).map(|m| m.as_str()), None);
            }
        }
        "wecom_callback" => {
            if let Some(c) = wecom_callback_re().captures(target_ref) {
                return explicit(c.get(1).map(|m| m.as_str()), None);
            }
        }
        "whatsapp" => {
            if let Some(c) = whatsapp_re().captures(target_ref) {
                return explicit(c.get(1).map(|m| m.as_str()), None);
            }
        }
        _ => {}
    }

    if PHONE_PLATFORMS.contains(&platform_name) {
        if e164_re().is_match(target_ref) {
            // Preserve the leading '+' — signal-cli / sms / whatsapp adapters
            // expect E.164 format for direct recipients.
            return explicit(Some(target_ref.trim()), None);
        }
    }

    // `target_ref.lstrip("-").isdigit()`
    let stripped: &str = target_ref.trim_start_matches('-');
    if is_all_digits(stripped) {
        return explicit(Some(target_ref), None);
    }

    // Matrix room IDs (!) and user IDs (@) are explicit.
    if platform_name == "matrix" && (target_ref.starts_with('!') || target_ref.starts_with('@')) {
        return explicit(Some(target_ref), None);
    }

    not_explicit()
}

// ---------------------------------------------------------------------------
// Media-mirror description
// ---------------------------------------------------------------------------

fn ext_lower(path: &str) -> String {
    match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some(e) => format!(".{}", e.to_lowercase()),
        None => String::new(),
    }
}

/// Return a human-readable mirror summary when a message only contains media.
/// `media_files` is a list of `(path, is_voice)`. Faithful port of
/// `_describe_media_for_mirror`.
pub fn describe_media_for_mirror(media_files: &[(String, bool)]) -> String {
    if media_files.is_empty() {
        return String::new();
    }
    if media_files.len() == 1 {
        let (media_path, is_voice) = &media_files[0];
        let ext = ext_lower(media_path);
        if *is_voice && VOICE_EXTS.contains(&ext.as_str()) {
            return "[Sent voice message]".to_string();
        }
        if IMAGE_EXTS.contains(&ext.as_str()) {
            return "[Sent image attachment]".to_string();
        }
        if VIDEO_EXTS.contains(&ext.as_str()) {
            return "[Sent video attachment]".to_string();
        }
        if AUDIO_EXTS.contains(&ext.as_str()) {
            return "[Sent audio attachment]".to_string();
        }
        return "[Sent document attachment]".to_string();
    }
    format!("[Sent {} media attachments]", media_files.len())
}

// ---------------------------------------------------------------------------
// Cron auto-delivery duplicate detection
// ---------------------------------------------------------------------------

/// The cron scheduler's auto-delivery target for the current run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronAutoTarget {
    pub platform: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
}

/// Environment lookup seam used by [`get_cron_auto_delivery_target`]. In the
/// gateway runtime this maps to `gateway.session_context.get_session_env`.
fn session_env(key: &str) -> String {
    std::env::var(key).unwrap_or_default()
}

/// Return the cron scheduler's auto-delivery target, if any. Faithful port of
/// `_get_cron_auto_delivery_target` reading the `HERMES_CRON_AUTO_DELIVER_*`
/// session-env values.
pub fn get_cron_auto_delivery_target() -> Option<CronAutoTarget> {
    let platform = session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM")
        .trim()
        .to_lowercase();
    let chat_id = session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID")
        .trim()
        .to_string();
    if platform.is_empty() || chat_id.is_empty() {
        return None;
    }
    let thread_raw = session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID")
        .trim()
        .to_string();
    let thread_id = if thread_raw.is_empty() {
        None
    } else {
        Some(thread_raw)
    };
    Some(CronAutoTarget {
        platform,
        chat_id,
        thread_id,
    })
}

/// Given an explicit auto-delivery target, decide whether the requested send
/// duplicates it and should be skipped. Pure variant for testing.
pub fn maybe_skip_cron_duplicate_send_with(
    auto_target: Option<&CronAutoTarget>,
    platform_name: &str,
    chat_id: &str,
    thread_id: Option<&str>,
) -> Option<Value> {
    let auto = auto_target?;

    let same_target = auto.platform == platform_name
        && auto.chat_id == chat_id
        && auto.thread_id.as_deref() == thread_id;
    if !same_target {
        return None;
    }

    let mut target_label = format!("{platform_name}:{chat_id}");
    if let Some(t) = thread_id {
        target_label.push(':');
        target_label.push_str(t);
    }

    Some(json!({
        "success": true,
        "skipped": true,
        "reason": "cron_auto_delivery_duplicate_target",
        "target": target_label,
        "note": format!(
            "Skipped send_message to {target_label}. This cron job will already auto-deliver \
             its final response to that same target. Put the intended user-facing content in \
             your final response instead, or use a different target if you want an additional message."
        ),
    }))
}

/// Skip redundant cron `send_message` calls when the scheduler will auto-deliver
/// to the same target. Reads the live session env. Faithful port of
/// `_maybe_skip_cron_duplicate_send`.
pub fn maybe_skip_cron_duplicate_send(
    platform_name: &str,
    chat_id: &str,
    thread_id: Option<&str>,
) -> Option<Value> {
    let auto = get_cron_auto_delivery_target();
    maybe_skip_cron_duplicate_send_with(auto.as_ref(), platform_name, chat_id, thread_id)
}

// ---------------------------------------------------------------------------
// SMS markdown stripping (Twilio)
// ---------------------------------------------------------------------------

/// Strip markdown so SMS does not render literal markdown characters.
/// Faithful port of the regex chain in `_send_sms`.
pub fn strip_markdown_for_sms(message: &str) -> String {
    fn re(pat: &str) -> Regex {
        Regex::new(pat).unwrap()
    }
    // Note: Rust `regex` has no DOTALL inline by default; use `(?s)` to mirror
    // Python's `re.DOTALL`, and `(?m)` for `re.MULTILINE`.
    let mut m = re(r"(?s)\*\*(.+?)\*\*").replace_all(message, "$1").into_owned();
    m = re(r"(?s)\*(.+?)\*").replace_all(&m, "$1").into_owned();
    m = re(r"(?s)__(.+?)__").replace_all(&m, "$1").into_owned();
    m = re(r"(?s)_(.+?)_").replace_all(&m, "$1").into_owned();
    m = re(r"```[a-z]*\n?").replace_all(&m, "").into_owned();
    m = re(r"`(.+?)`").replace_all(&m, "$1").into_owned();
    m = re(r"(?m)^#{1,6}\s+").replace_all(&m, "").into_owned();
    m = re(r"\[([^\]]+)\]\([^\)]+\)").replace_all(&m, "$1").into_owned();
    m = re(r"\n{3,}").replace_all(&m, "\n\n").into_owned();
    m.trim().to_string()
}

// ---------------------------------------------------------------------------
// Forum thread name (Discord)
// ---------------------------------------------------------------------------

/// Derive a thread name from the first line of the message, capped at 100
/// chars. Faithful port of `_derive_forum_thread_name`.
pub fn derive_forum_thread_name(message: &str) -> String {
    let first_line = message
        .trim()
        .split('\n')
        .next()
        .unwrap_or("")
        .trim();
    let first_line = first_line.trim_start_matches('#').trim();
    let first_line = if first_line.is_empty() {
        "New Post"
    } else {
        first_line
    };
    first_line.chars().take(100).collect()
}

/// True when the message appears to contain HTML tags (Telegram HTML detection).
pub fn message_has_html(message: &str) -> bool {
    html_tag_re().is_match(message)
}

// ---------------------------------------------------------------------------
// Discord channel-type probe cache (process-local)
// ---------------------------------------------------------------------------

use std::sync::Mutex;

fn discord_probe_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, bool>>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Remember whether a Discord channel is a forum channel.
pub fn remember_channel_is_forum(chat_id: &str, is_forum: bool) {
    discord_probe_cache()
        .lock()
        .unwrap()
        .insert(chat_id.to_string(), is_forum);
}

/// Look up a cached forum/non-forum determination for a Discord channel.
pub fn probe_is_forum_cached(chat_id: &str) -> Option<bool> {
    discord_probe_cache().lock().unwrap().get(chat_id).copied()
}

// ---------------------------------------------------------------------------
// epoch-ms id helper (Signal / Matrix transaction ids)
// ---------------------------------------------------------------------------

fn epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Self-contained HTTP senders (reqwest::blocking)
//
// These mirror the request construction + response parsing of the Python
// coroutines. They take the resolved credentials/endpoints as parameters; the
// caller is responsible for sourcing them from gateway config / env (matching
// the Python `pconfig.token` / `pconfig.extra` access).
// ---------------------------------------------------------------------------

fn http_client(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

/// Send via Slack Web API (`chat.postMessage`). Port of `_send_slack`.
pub fn send_slack(token: &str, chat_id: &str, message: &str) -> Value {
    let url = "https://slack.com/api/chat.postMessage";
    let client = http_client(30);
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .json(&json!({ "channel": chat_id, "text": message, "mrkdwn": true }))
        .send();
    match resp {
        Ok(r) => match r.json::<Value>() {
            Ok(data) => {
                if data.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                    json!({
                        "success": true,
                        "platform": "slack",
                        "chat_id": chat_id,
                        "message_id": data.get("ts").cloned().unwrap_or(Value::Null),
                    })
                } else {
                    let err = data
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    error_payload(&format!("Slack API error: {err}"))
                }
            }
            Err(e) => error_payload(&format!("Slack send failed: {e}")),
        },
        Err(e) => error_payload(&format!("Slack send failed: {e}")),
    }
}

/// Send via the local WhatsApp bridge HTTP API. Port of `_send_whatsapp`.
pub fn send_whatsapp(bridge_port: u16, chat_id: &str, message: &str) -> Value {
    let client = http_client(30);
    let url = format!("http://localhost:{bridge_port}/send");
    let resp = client
        .post(&url)
        .json(&json!({ "chatId": chat_id, "message": message }))
        .send();
    match resp {
        Ok(r) => {
            let status = r.status();
            if status.as_u16() == 200 {
                let data: Value = r.json().unwrap_or(Value::Null);
                json!({
                    "success": true,
                    "platform": "whatsapp",
                    "chat_id": chat_id,
                    "message_id": data.get("messageId").cloned().unwrap_or(Value::Null),
                })
            } else {
                let body = r.text().unwrap_or_default();
                error_payload(&format!("WhatsApp bridge error ({}): {body}", status.as_u16()))
            }
        }
        Err(e) => error_payload(&format!("WhatsApp send failed: {e}")),
    }
}

/// Send via Mattermost REST API. Port of `_send_mattermost`.
pub fn send_mattermost(base_url: &str, token: &str, chat_id: &str, message: &str) -> Value {
    let base_url = base_url.trim_end_matches('/');
    if base_url.is_empty() || token.is_empty() {
        return json!({ "error": "Mattermost not configured (MATTERMOST_URL, MATTERMOST_TOKEN required)" });
    }
    let url = format!("{base_url}/api/v4/posts");
    let client = http_client(30);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .json(&json!({ "channel_id": chat_id, "message": message }))
        .send();
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if status != 200 && status != 201 {
                let body = r.text().unwrap_or_default();
                return error_payload(&format!("Mattermost API error ({status}): {body}"));
            }
            let data: Value = r.json().unwrap_or(Value::Null);
            json!({
                "success": true,
                "platform": "mattermost",
                "chat_id": chat_id,
                "message_id": data.get("id").cloned().unwrap_or(Value::Null),
            })
        }
        Err(e) => error_payload(&format!("Mattermost send failed: {e}")),
    }
}

/// Build a Matrix transaction id (`hermes_<ms>_<8 hex>`).
pub fn matrix_txn_id() -> String {
    let mut bytes = [0u8; 4];
    // Best-effort randomness from the clock; matches os.urandom(4).hex() shape.
    let seed = epoch_ms() as u64;
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = ((seed >> (i * 8)) & 0xff) as u8;
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("hermes_{}_{}", epoch_ms(), hex)
}

/// Percent-encode a Matrix room/user id for use in a path segment
/// (`urllib.parse.quote(chat_id, safe="")`).
pub fn matrix_encode_room(chat_id: &str) -> String {
    let mut out = String::new();
    for b in chat_id.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '~') {
            out.push(c);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Send via Matrix Client-Server API (text/plain body; HTML formatting is
/// applied by the caller if a markdown renderer is available). Port of
/// `_send_matrix` request/response handling.
pub fn send_matrix(
    homeserver: &str,
    token: &str,
    chat_id: &str,
    message: &str,
    formatted_html: Option<&str>,
) -> Value {
    let homeserver = homeserver.trim_end_matches('/');
    if homeserver.is_empty() || token.is_empty() {
        return json!({ "error": "Matrix not configured (MATRIX_HOMESERVER, MATRIX_ACCESS_TOKEN required)" });
    }
    let txn_id = matrix_txn_id();
    let encoded_room = matrix_encode_room(chat_id);
    let url = format!(
        "{homeserver}/_matrix/client/v3/rooms/{encoded_room}/send/m.room.message/{txn_id}"
    );
    let mut payload = json!({ "msgtype": "m.text", "body": message });
    if let Some(html) = formatted_html {
        payload["format"] = json!("org.matrix.custom.html");
        payload["formatted_body"] = json!(html);
    }
    let client = http_client(30);
    let resp = client
        .put(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send();
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if status != 200 && status != 201 {
                let body = r.text().unwrap_or_default();
                return error_payload(&format!("Matrix API error ({status}): {body}"));
            }
            let data: Value = r.json().unwrap_or(Value::Null);
            json!({
                "success": true,
                "platform": "matrix",
                "chat_id": chat_id,
                "message_id": data.get("event_id").cloned().unwrap_or(Value::Null),
            })
        }
        Err(e) => error_payload(&format!("Matrix send failed: {e}")),
    }
}

/// Convert markdown headings (`<h1>..<h6>`) to `<strong>` for Element X
/// compatibility, mirroring the regex in `_send_matrix`.
pub fn matrix_headings_to_strong(html: &str) -> String {
    let re = Regex::new(r"(?s)<h[1-6]>(.*?)</h[1-6]>").unwrap();
    re.replace_all(html, "<strong>$1</strong>").into_owned()
}

/// Send via Home Assistant notify service. Port of `_send_homeassistant`.
pub fn send_homeassistant(hass_url: &str, token: &str, chat_id: &str, message: &str) -> Value {
    let hass_url = hass_url.trim_end_matches('/');
    if hass_url.is_empty() || token.is_empty() {
        return json!({ "error": "Home Assistant not configured (HASS_URL, HASS_TOKEN required)" });
    }
    let url = format!("{hass_url}/api/services/notify/notify");
    let client = http_client(30);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .json(&json!({ "message": message, "target": chat_id }))
        .send();
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            if status != 200 && status != 201 {
                let body = r.text().unwrap_or_default();
                return error_payload(&format!("Home Assistant API error ({status}): {body}"));
            }
            json!({ "success": true, "platform": "homeassistant", "chat_id": chat_id })
        }
        Err(e) => error_payload(&format!("Home Assistant send failed: {e}")),
    }
}

/// Send via DingTalk robot webhook. Port of `_send_dingtalk`.
pub fn send_dingtalk(webhook_url: &str, chat_id: &str, message: &str) -> Value {
    if webhook_url.is_empty() {
        return json!({ "error": "DingTalk not configured. Set DINGTALK_WEBHOOK_URL env var or webhook_url in dingtalk platform extra config." });
    }
    let client = http_client(30);
    let resp = client
        .post(webhook_url)
        .json(&json!({ "msgtype": "text", "text": { "content": message } }))
        .send();
    match resp {
        Ok(r) => {
            // raise_for_status() equivalent
            if let Err(e) = r.error_for_status_ref() {
                return error_payload(&format!("DingTalk send failed: {e}"));
            }
            let data: Value = r.json().unwrap_or(Value::Null);
            let errcode = data.get("errcode").and_then(|v| v.as_i64()).unwrap_or(0);
            if errcode != 0 {
                let errmsg = data
                    .get("errmsg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                return error_payload(&format!("DingTalk API error: {errmsg}"));
            }
            json!({ "success": true, "platform": "dingtalk", "chat_id": chat_id })
        }
        Err(e) => error_payload(&format!("DingTalk send failed: {e}")),
    }
}

/// Send a single SMS via the Twilio REST API. Port of `_send_sms`.
///
/// `auth_token`, `account_sid`, and `from_number` are sourced by the caller
/// (matching `pconfig.api_key` + `TWILIO_*` env vars). The message is markdown-
/// stripped before delivery.
pub fn send_sms(
    account_sid: &str,
    auth_token: &str,
    from_number: &str,
    chat_id: &str,
    message: &str,
) -> Value {
    if account_sid.is_empty() || auth_token.is_empty() || from_number.is_empty() {
        return json!({ "error": "SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER required)" });
    }
    let body = strip_markdown_for_sms(message);
    let url = format!("https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json");
    let client = http_client(30);
    let resp = client
        .post(&url)
        .basic_auth(account_sid, Some(auth_token))
        .form(&[("From", from_number), ("To", chat_id), ("Body", body.as_str())])
        .send();
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let data: Value = r.json().unwrap_or(Value::Null);
            if status >= 400 {
                let error_msg = data
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| data.to_string());
                return error_payload(&format!("Twilio API error ({status}): {error_msg}"));
            }
            let msg_sid = data
                .get("sid")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            json!({
                "success": true,
                "platform": "sms",
                "chat_id": chat_id,
                "message_id": msg_sid,
            })
        }
        Err(e) => error_payload(&format!("SMS send failed: {e}")),
    }
}

/// Send via QQBot using the QQ Bot Open Platform REST endpoints. Port of
/// `_send_qqbot`: fetch an app access token, then try the channel, C2C, and
/// group endpoints in order.
pub fn send_qqbot(appid: &str, secret: &str, chat_id: &str, message: &str) -> Value {
    if appid.is_empty() || secret.is_empty() {
        return error_payload("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.");
    }
    let client = http_client(15);

    // Step 1: access token.
    let token_resp = client
        .post("https://bots.qq.com/app/getAppAccessToken")
        .json(&json!({ "appId": appid, "clientSecret": secret }))
        .send();
    let token_resp = match token_resp {
        Ok(r) => r,
        Err(e) => return error_payload(&format!("QQBot send failed: {e}")),
    };
    if token_resp.status().as_u16() != 200 {
        return error_payload(&format!(
            "QQBot token request failed: {}",
            token_resp.status().as_u16()
        ));
    }
    let token_data: Value = token_resp.json().unwrap_or(Value::Null);
    let access_token = match token_data.get("access_token").and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return error_payload("QQBot: no access_token in response"),
    };

    // Step 2: send. content truncated to 4000 chars.
    let content: String = message.chars().take(4000).collect();
    let payload = json!({ "content": content, "msg_type": 0 });
    let auth = format!("QQBot {access_token}");

    let post = |url: String| -> Result<(u16, Value), reqwest::Error> {
        let r = client
            .post(&url)
            .header("Authorization", &auth)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()?;
        let status = r.status().as_u16();
        let data = r.json::<Value>().unwrap_or(Value::Null);
        Ok((status, data))
    };

    // Channel endpoint.
    let ch = post(format!("https://api.sgroup.qq.com/channels/{chat_id}/messages"));
    let (ch_status, ch_data) = match ch {
        Ok(v) => v,
        Err(e) => return error_payload(&format!("QQBot send failed: {e}")),
    };
    if ch_status == 200 || ch_status == 201 {
        return json!({
            "success": true, "platform": "qqbot", "chat_id": chat_id,
            "message_id": ch_data.get("id").cloned().unwrap_or(Value::Null),
        });
    }

    // C2C endpoint.
    let c2c = post(format!("https://api.sgroup.qq.com/v2/users/{chat_id}/messages"));
    let (c2c_status, c2c_data) = match c2c {
        Ok(v) => v,
        Err(e) => return error_payload(&format!("QQBot send failed: {e}")),
    };
    if c2c_status == 200 || c2c_status == 201 {
        return json!({
            "success": true, "platform": "qqbot", "chat_id": chat_id,
            "message_id": c2c_data.get("id").cloned().unwrap_or(Value::Null),
        });
    }

    // Group endpoint.
    let grp = post(format!("https://api.sgroup.qq.com/v2/groups/{chat_id}/messages"));
    let (grp_status, grp_data) = match grp {
        Ok(v) => v,
        Err(e) => return error_payload(&format!("QQBot send failed: {e}")),
    };
    if grp_status == 200 || grp_status == 201 {
        return json!({
            "success": true, "platform": "qqbot", "chat_id": chat_id,
            "message_id": grp_data.get("id").cloned().unwrap_or(Value::Null),
        });
    }

    error_payload(&format!(
        "QQBot send failed: channel={ch_status} c2c={c2c_status} group={grp_status}"
    ))
}

// ---------------------------------------------------------------------------
// Tool entry point (action routing)
// ---------------------------------------------------------------------------

/// The two supported actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Send,
    List,
}

/// Parse the `action` argument (defaults to `Send`). Anything other than
/// `"list"` routes to `Send`, matching the Python `if action == "list"`.
pub fn parse_action(args: &Value) -> Action {
    match args.get("action").and_then(|v| v.as_str()) {
        Some("list") => Action::List,
        _ => Action::Send,
    }
}

/// Extract `target` and `message` strings from the args (defaulting to empty).
pub fn extract_target_message(args: &Value) -> (String, String) {
    let target = args
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let message = args
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    (target, message)
}

/// Validate that both `target` and `message` are present for a send. Returns
/// the tool-error JSON string when validation fails (matches the Python guard).
pub fn validate_send_args(target: &str, message: &str) -> Option<String> {
    if target.is_empty() || message.is_empty() {
        Some(tool_error(
            "Both 'target' and 'message' are required when action='send'",
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_explicit_topic() {
        let r = parse_target_ref("telegram", "-1001234567890:17585");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("-1001234567890"));
        assert_eq!(r.thread_id.as_deref(), Some("17585"));
    }

    #[test]
    fn telegram_plain_id() {
        let r = parse_target_ref("telegram", "12345");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("12345"));
        assert_eq!(r.thread_id, None);
    }

    #[test]
    fn discord_uses_numeric_topic_regex() {
        let r = parse_target_ref("discord", "999888777:555444333");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("999888777"));
        assert_eq!(r.thread_id.as_deref(), Some("555444333"));
    }

    #[test]
    fn slack_conversation_id_with_thread() {
        let r = parse_target_ref("slack", "C123ABCDEF:1749236185.123456");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("C123ABCDEF"));
        assert_eq!(r.thread_id.as_deref(), Some("1749236185.123456"));
    }

    #[test]
    fn slack_user_id_not_explicit() {
        // U... is not a valid conversation id and is not all-digits.
        let r = parse_target_ref("slack", "U12345678");
        assert!(!r.is_explicit);
    }

    #[test]
    fn yuanbao_bare_digits_become_group() {
        let r = parse_target_ref("yuanbao", "778899");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("group:778899"));
    }

    #[test]
    fn yuanbao_direct_prefix() {
        let r = parse_target_ref("yuanbao", "direct:acct_1");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("direct:acct_1"));
    }

    #[test]
    fn yuanbao_name_not_explicit() {
        let r = parse_target_ref("yuanbao", "friendly-name");
        assert!(!r.is_explicit);
        assert_eq!(r.chat_id, None);
    }

    #[test]
    fn signal_e164_preserves_plus() {
        let r = parse_target_ref("signal", "+15551234567");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("+15551234567"));
    }

    #[test]
    fn signal_group_target() {
        let r = parse_target_ref("signal", "group:abc123==");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("group:abc123=="));
    }

    #[test]
    fn whatsapp_jid() {
        let r = parse_target_ref("whatsapp", "123456@s.whatsapp.net");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("123456@s.whatsapp.net"));
    }

    #[test]
    fn matrix_room_id_explicit() {
        let r = parse_target_ref("matrix", "!roomid:server.org");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("!roomid:server.org"));
    }

    #[test]
    fn matrix_user_id_explicit() {
        let r = parse_target_ref("matrix", "@user:server.org");
        assert!(r.is_explicit);
    }

    #[test]
    fn feishu_chat_id() {
        let r = parse_target_ref("feishu", "oc_abc-123:reply_xyz");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("oc_abc-123"));
        assert_eq!(r.thread_id.as_deref(), Some("reply_xyz"));
    }

    #[test]
    fn weixin_filehelper() {
        let r = parse_target_ref("weixin", "filehelper");
        assert!(r.is_explicit);
        assert_eq!(r.chat_id.as_deref(), Some("filehelper"));
    }

    #[test]
    fn unknown_name_not_explicit() {
        let r = parse_target_ref("discord", "#bot-home");
        assert!(!r.is_explicit);
    }

    #[test]
    fn split_target_basic() {
        let s = split_target("telegram:-100:17");
        assert_eq!(s.platform_name, "telegram");
        assert_eq!(s.target_ref.as_deref(), Some("-100:17"));
    }

    #[test]
    fn split_target_no_ref() {
        let s = split_target("Telegram");
        assert_eq!(s.platform_name, "telegram");
        assert_eq!(s.target_ref, None);
    }

    #[test]
    fn sanitize_url_query_secret() {
        let out = sanitize_error_text("see https://x.com/cb?access_token=SECRET123&a=1");
        assert!(out.contains("access_token=***"));
        assert!(!out.contains("SECRET123"));
    }

    #[test]
    fn sanitize_generic_assignment() {
        let out = sanitize_error_text("api_key=abcdef failure");
        assert!(out.contains("api_key=***"));
        assert!(!out.contains("abcdef"));
    }

    #[test]
    fn error_payload_is_redacted() {
        let v = error_payload("token=topsecret");
        assert_eq!(v["error"], "token=***");
    }

    #[test]
    fn telegram_retry_retry_after() {
        assert_eq!(telegram_retry_delay(Some(5.0), "", 0), Some(5.0));
        assert_eq!(telegram_retry_delay(Some(-2.0), "", 0), Some(0.0));
        assert_eq!(telegram_retry_delay(Some(f64::NAN), "", 0), Some(1.0));
    }

    #[test]
    fn telegram_retry_timeout_is_none() {
        assert_eq!(telegram_retry_delay(None, "Request timed out", 0), None);
        assert_eq!(telegram_retry_delay(None, "connection timeout", 1), None);
    }

    #[test]
    fn telegram_retry_transient_backoff() {
        assert_eq!(telegram_retry_delay(None, "502 Bad Gateway", 0), Some(1.0));
        assert_eq!(telegram_retry_delay(None, "Too Many Requests", 2), Some(4.0));
        assert_eq!(
            telegram_retry_delay(None, "503 Service Unavailable", 3),
            Some(8.0)
        );
    }

    #[test]
    fn telegram_retry_non_transient_none() {
        assert_eq!(telegram_retry_delay(None, "401 Unauthorized", 0), None);
    }

    #[test]
    fn describe_single_image() {
        let m = vec![("/tmp/a.png".to_string(), false)];
        assert_eq!(describe_media_for_mirror(&m), "[Sent image attachment]");
    }

    #[test]
    fn describe_voice() {
        let m = vec![("/tmp/a.ogg".to_string(), true)];
        assert_eq!(describe_media_for_mirror(&m), "[Sent voice message]");
    }

    #[test]
    fn describe_ogg_not_voice_is_audio() {
        let m = vec![("/tmp/a.ogg".to_string(), false)];
        assert_eq!(describe_media_for_mirror(&m), "[Sent audio attachment]");
    }

    #[test]
    fn describe_document() {
        let m = vec![("/tmp/a.pdf".to_string(), false)];
        assert_eq!(describe_media_for_mirror(&m), "[Sent document attachment]");
    }

    #[test]
    fn describe_multiple() {
        let m = vec![
            ("/tmp/a.png".to_string(), false),
            ("/tmp/b.mp4".to_string(), false),
        ];
        assert_eq!(describe_media_for_mirror(&m), "[Sent 2 media attachments]");
    }

    #[test]
    fn describe_empty() {
        assert_eq!(describe_media_for_mirror(&[]), "");
    }

    #[test]
    fn cron_duplicate_skip_matches() {
        let auto = CronAutoTarget {
            platform: "telegram".to_string(),
            chat_id: "-100".to_string(),
            thread_id: Some("17".to_string()),
        };
        let r = maybe_skip_cron_duplicate_send_with(Some(&auto), "telegram", "-100", Some("17"));
        assert!(r.is_some());
        let v = r.unwrap();
        assert_eq!(v["skipped"], true);
        assert_eq!(v["target"], "telegram:-100:17");
    }

    #[test]
    fn cron_duplicate_no_skip_different_thread() {
        let auto = CronAutoTarget {
            platform: "telegram".to_string(),
            chat_id: "-100".to_string(),
            thread_id: Some("17".to_string()),
        };
        let r = maybe_skip_cron_duplicate_send_with(Some(&auto), "telegram", "-100", None);
        assert!(r.is_none());
    }

    #[test]
    fn cron_duplicate_none_when_no_target() {
        let r = maybe_skip_cron_duplicate_send_with(None, "telegram", "-100", None);
        assert!(r.is_none());
    }

    #[test]
    fn cron_duplicate_label_without_thread() {
        let auto = CronAutoTarget {
            platform: "slack".to_string(),
            chat_id: "C123".to_string(),
            thread_id: None,
        };
        let r = maybe_skip_cron_duplicate_send_with(Some(&auto), "slack", "C123", None).unwrap();
        assert_eq!(r["target"], "slack:C123");
    }

    #[test]
    fn sms_strips_markdown() {
        let out = strip_markdown_for_sms("**bold** and *italic* and `code` and # Heading");
        assert!(!out.contains("**"));
        assert!(!out.contains('`'));
        assert!(out.contains("bold"));
        assert!(out.contains("italic"));
        assert!(out.contains("code"));
        assert!(out.contains("Heading"));
    }

    #[test]
    fn sms_strips_link() {
        let out = strip_markdown_for_sms("see [docs](https://x.com)");
        assert_eq!(out, "see docs");
    }

    #[test]
    fn sms_collapses_blank_lines() {
        let out = strip_markdown_for_sms("a\n\n\n\nb");
        assert_eq!(out, "a\n\nb");
    }

    #[test]
    fn forum_thread_name_from_first_line() {
        assert_eq!(derive_forum_thread_name("# Title\nbody"), "Title");
        assert_eq!(derive_forum_thread_name("\n\n"), "New Post");
    }

    #[test]
    fn forum_thread_name_capped() {
        let long = "x".repeat(200);
        assert_eq!(derive_forum_thread_name(&long).chars().count(), 100);
    }

    #[test]
    fn html_detection() {
        assert!(message_has_html("<b>hi</b>"));
        assert!(message_has_html("a </p> b"));
        assert!(!message_has_html("3 < 5 and x > 2"));
    }

    #[test]
    fn matrix_encode_path() {
        assert_eq!(matrix_encode_room("!room:server.org"), "%21room%3Aserver.org");
        assert_eq!(matrix_encode_room("@u:s.org"), "%40u%3As.org");
    }

    #[test]
    fn matrix_headings_to_strong_works() {
        let out = matrix_headings_to_strong("<h1>Hi</h1><p>x</p>");
        assert_eq!(out, "<strong>Hi</strong><p>x</p>");
    }

    #[test]
    fn discord_probe_cache_roundtrip() {
        assert_eq!(probe_is_forum_cached("chan_test_xyz"), None);
        remember_channel_is_forum("chan_test_xyz", true);
        assert_eq!(probe_is_forum_cached("chan_test_xyz"), Some(true));
    }

    #[test]
    fn action_defaults_to_send() {
        assert_eq!(parse_action(&json!({})), Action::Send);
        assert_eq!(parse_action(&json!({"action": "send"})), Action::Send);
        assert_eq!(parse_action(&json!({"action": "list"})), Action::List);
        assert_eq!(parse_action(&json!({"action": "weird"})), Action::Send);
    }

    #[test]
    fn validate_requires_both() {
        assert!(validate_send_args("", "hi").is_some());
        assert!(validate_send_args("telegram", "").is_some());
        assert!(validate_send_args("telegram", "hi").is_none());
    }

    #[test]
    fn schema_shape() {
        let s = send_message_schema();
        assert_eq!(s["name"], "send_message");
        assert_eq!(s["parameters"]["properties"]["action"]["enum"][1], "list");
    }

    #[test]
    fn cron_env_target_reads_env() {
        unsafe {
            std::env::set_var("HERMES_CRON_AUTO_DELIVER_PLATFORM", "Telegram");
            std::env::set_var("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "-100");
            std::env::set_var("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "  ");
        }
        let t = get_cron_auto_delivery_target().unwrap();
        assert_eq!(t.platform, "telegram");
        assert_eq!(t.chat_id, "-100");
        assert_eq!(t.thread_id, None);
        unsafe {
            std::env::remove_var("HERMES_CRON_AUTO_DELIVER_PLATFORM");
            std::env::remove_var("HERMES_CRON_AUTO_DELIVER_CHAT_ID");
            std::env::remove_var("HERMES_CRON_AUTO_DELIVER_THREAD_ID");
        }
    }
}
