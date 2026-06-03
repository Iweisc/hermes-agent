//! DingTalk platform adapter — native Rust port of
//! `gateway/platforms/dingtalk.py`.
//!
//! The Python adapter drives the `dingtalk-stream` SDK over a long-lived
//! WebSocket and replies through DingTalk's per-session webhook (markdown) or
//! via the AI-Card SDK (`alibabacloud_dingtalk.card_1_0` /
//! `alibabacloud_dingtalk.robot_1_0`). This port reproduces the *behavior-
//! defining* surface of that adapter:
//!
//!   - dependency / configuration probing ([`check_dingtalk_requirements`])
//!   - group-chat gating: `require_mention`, `free_response_chats`,
//!     regex wake-word patterns, `allowed_users`
//!   - inbound message parsing: text extraction (legacy dict / TextContent /
//!     rich-text shapes), media extraction (image + rich-text download codes),
//!     timestamp parsing
//!   - session-webhook caching with size cap + expiry safety-margin
//!   - markdown normalization for DingTalk's renderer
//!   - AI-Card streaming lifecycle state (streaming-sibling tracking,
//!     done-reaction idempotency) + request-payload construction
//!   - outbound webhook send (`reqwest::blocking`) request construction
//!
//! The CPython `asyncio` machinery (WebSocket reconnect loop, fire-and-forget
//! background reaction tasks, cross-thread dispatch) is modelled here as plain
//! synchronous state + helpers; the live event loop wiring belongs to the
//! gateway runtime layer that drives this state.
//!
//! Cross-refs:
//!   - [`crate::gw_helpers::MessageDeduplicator`]
//!   - [`crate::gw_platforms_base::MessageEvent`] / `MessageType` / `SendResult`
//!     / `SessionSource`

use std::collections::HashMap;

use regex::Regex;
use serde_json::{Value, json};

use crate::gw_helpers::MessageDeduplicator;
use crate::gw_platforms_base::{MessageEvent, MessageType, SendResult, SessionSource};

// ===========================================================================
// Constants
// ===========================================================================

/// DingTalk markdown message hard cap (chars). Mirrors `MAX_MESSAGE_LENGTH`.
pub const MAX_MESSAGE_LENGTH: usize = 20000;

/// Reconnect backoff ladder (seconds). Mirrors `RECONNECT_BACKOFF`.
pub const RECONNECT_BACKOFF: &[u64] = &[2, 5, 10, 30, 60];

/// Maximum number of cached session webhooks. Mirrors `_SESSION_WEBHOOKS_MAX`.
pub const SESSION_WEBHOOKS_MAX: usize = 500;

/// 5-minute safety margin (ms) applied when checking webhook expiry.
pub const WEBHOOK_SAFETY_MARGIN_MS: i64 = 5 * 60 * 1000;

/// Return the compiled DingTalk webhook host pattern. Mirrors
/// `_DINGTALK_WEBHOOK_RE`.
pub fn dingtalk_webhook_re() -> Regex {
    Regex::new(r"^https://(?:api|oapi)\.dingtalk\.com/").unwrap()
}

/// Map a DingTalk message type to a runtime content type. Mirrors
/// `DINGTALK_TYPE_MAPPING` (unknown types fall back to `"file"`).
pub fn dingtalk_type_mapping(item_type: &str) -> &'static str {
    match item_type {
        "picture" => "image",
        "voice" => "audio",
        _ => "file",
    }
}

// ===========================================================================
// Requirement / configuration probing
// ===========================================================================

/// Check if DingTalk dependencies are available and configured.
///
/// Mirrors `check_dingtalk_requirements`. In the native runtime the SDK
/// availability flags are always true (the WebSocket/HTTP work is native), so
/// this reduces to verifying the two credential env vars are present and
/// non-empty.
pub fn check_dingtalk_requirements() -> bool {
    let id = std::env::var("DINGTALK_CLIENT_ID").unwrap_or_default();
    let secret = std::env::var("DINGTALK_CLIENT_SECRET").unwrap_or_default();
    !id.is_empty() && !secret.is_empty()
}

// ===========================================================================
// Truthy env parsing
// ===========================================================================

/// Python `value.lower() in ("true", "1", "yes", "on")`.
fn is_truthy_str(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

// ===========================================================================
// Inbound message model
//
// A lightweight mirror of the dingtalk-stream `ChatbotMessage` fields the
// adapter actually reads. The native WebSocket layer populates this from the
// raw callback payload (see [`ChatbotMessage::from_value`]).
// ===========================================================================

/// Single rich-text element (a dict in the SDK payload).
#[derive(Debug, Clone, Default)]
pub struct RichTextItem {
    pub fields: HashMap<String, Value>,
}

impl RichTextItem {
    pub fn get_str(&self, key: &str) -> Option<String> {
        match self.fields.get(key) {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        }
    }
    pub fn set_str(&mut self, key: &str, value: &str) {
        self.fields
            .insert(key.to_string(), Value::String(value.to_string()));
    }
}

/// Minimal mirror of the dingtalk-stream `ChatbotMessage`.
#[derive(Debug, Clone, Default)]
pub struct ChatbotMessage {
    pub message_id: Option<String>,
    pub conversation_id: String,
    /// `"1"` = DM, `"2"` = group.
    pub conversation_type: String,
    pub conversation_title: Option<String>,
    pub sender_id: String,
    pub sender_nick: String,
    pub sender_staff_id: String,
    pub message_type: String,
    /// Plain text content (already unwrapped from dict/TextContent).
    pub text: Option<String>,
    pub rich_text_list: Vec<RichTextItem>,
    /// Single-image download code.
    pub image_download_code: Option<String>,
    pub session_webhook: String,
    pub session_webhook_expired_time: i64,
    pub is_in_at_list: bool,
    pub robot_code: Option<String>,
    /// Epoch milliseconds.
    pub create_at: Option<i64>,
}

impl ChatbotMessage {
    /// Parse a raw DingTalk callback payload (`CallbackMessage.data`) into a
    /// `ChatbotMessage`. Mirrors the SDK `ChatbotMessage.from_dict()` for the
    /// fields this adapter consumes, including the `_IncomingHandler` fallbacks
    /// for `sessionWebhook` and `isInAtList` field-name variance.
    pub fn from_value(data: &Value) -> ChatbotMessage {
        let obj = match data.as_object() {
            Some(o) => o,
            None => return ChatbotMessage::default(),
        };
        let get_str = |keys: &[&str]| -> String {
            for k in keys {
                if let Some(Value::String(s)) = obj.get(*k) {
                    return s.clone();
                }
            }
            String::new()
        };

        let mut msg = ChatbotMessage::default();
        let mid = get_str(&["msgId", "messageId", "message_id"]);
        msg.message_id = if mid.is_empty() { None } else { Some(mid) };
        msg.conversation_id = get_str(&["conversationId", "conversation_id"]);
        let ctype = get_str(&["conversationType", "conversation_type"]);
        msg.conversation_type = if ctype.is_empty() {
            "1".to_string()
        } else {
            ctype
        };
        let ctitle = get_str(&["conversationTitle", "conversation_title"]);
        msg.conversation_title = if ctitle.is_empty() { None } else { Some(ctitle) };
        msg.sender_id = get_str(&["senderId", "sender_id"]);
        msg.sender_nick = get_str(&["senderNick", "sender_nick"]);
        msg.sender_staff_id = get_str(&["senderStaffId", "sender_staff_id"]);
        msg.message_type = get_str(&["msgtype", "message_type"]);

        // Text content: payload shape is {"text": {"content": "..."}}.
        if let Some(text_obj) = obj.get("text").and_then(|v| v.as_object()) {
            if let Some(Value::String(c)) = text_obj.get("content") {
                msg.text = Some(c.clone());
            }
        } else if let Some(Value::String(s)) = obj.get("text") {
            msg.text = Some(s.clone());
        }

        // Rich text content: {"content": {"richText": [ {..}, ... ]}}.
        let rich = obj
            .get("content")
            .and_then(|v| v.get("richText"))
            .or_else(|| obj.get("richText"))
            .or_else(|| obj.get("rich_text"));
        if let Some(Value::Array(arr)) = rich {
            for item in arr {
                if let Some(o) = item.as_object() {
                    msg.rich_text_list.push(RichTextItem {
                        fields: o.clone().into_iter().collect(),
                    });
                }
            }
        }

        // Single image content: {"content": {"downloadCode": "..."}} for
        // msgtype == "picture".
        if let Some(dc) = obj
            .get("content")
            .and_then(|v| v.get("downloadCode"))
            .and_then(|v| v.as_str())
        {
            msg.image_download_code = Some(dc.to_string());
        }

        // Session webhook with field-name fallback (matches _IncomingHandler).
        msg.session_webhook = get_str(&["sessionWebhook", "session_webhook"]);
        msg.session_webhook_expired_time = obj
            .get("sessionWebhookExpiredTime")
            .or_else(|| obj.get("session_webhook_expired_time"))
            .and_then(value_as_i64)
            .unwrap_or(0);

        // is_in_at_list fallback.
        msg.is_in_at_list = obj
            .get("isInAtList")
            .or_else(|| obj.get("is_in_at_list"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        msg.robot_code = {
            let rc = get_str(&["robotCode", "robot_code"]);
            if rc.is_empty() { None } else { Some(rc) }
        };

        msg.create_at = obj
            .get("createAt")
            .or_else(|| obj.get("create_at"))
            .and_then(value_as_i64);

        msg
    }
}

fn value_as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

// ===========================================================================
// Group gating
// ===========================================================================

/// Group-chat gating configuration resolved from `config.extra` + env vars.
///
/// Mirrors the cluster of `_dingtalk_require_mention`,
/// `_dingtalk_free_response_chats`, `_compile_mention_patterns`, and
/// `_load_allowed_users` helpers. Construct once from the adapter config; the
/// per-message gate is [`GroupGating::should_process_message`].
#[derive(Debug)]
pub struct GroupGating {
    pub mention_patterns: Vec<Regex>,
    /// Lowercased allowed-user ids (staff_id / sender_id). `*` disables.
    pub allowed_users: std::collections::HashSet<String>,
    pub require_mention: bool,
    pub free_response_chats: std::collections::HashSet<String>,
}

impl GroupGating {
    /// Build gating state from the adapter's `config.extra` JSON object.
    pub fn from_config(extra: &Value) -> GroupGating {
        GroupGating {
            mention_patterns: compile_mention_patterns(extra),
            allowed_users: load_allowed_users(extra),
            require_mention: resolve_require_mention(extra),
            free_response_chats: resolve_free_response_chats(extra),
        }
    }

    /// Mirrors `_is_user_allowed`.
    pub fn is_user_allowed(&self, sender_id: &str, sender_staff_id: &str) -> bool {
        if self.allowed_users.is_empty() || self.allowed_users.contains("*") {
            return true;
        }
        let mut candidates: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        candidates.insert(sender_id.to_lowercase());
        candidates.insert(sender_staff_id.to_lowercase());
        candidates.remove("");
        candidates.intersection(&self.allowed_users).next().is_some()
    }

    /// True if `text` matches a configured regex wake-word pattern.
    /// Mirrors `_message_matches_mention_patterns`.
    pub fn message_matches_mention_patterns(&self, text: &str) -> bool {
        if text.is_empty() || self.mention_patterns.is_empty() {
            return false;
        }
        self.mention_patterns.iter().any(|p| p.is_match(text))
    }

    /// Apply DingTalk group trigger rules. Mirrors `_should_process_message`.
    ///
    /// `mentions_bot` is the structured `is_in_at_list` flag.
    pub fn should_process_message(
        &self,
        text: &str,
        is_group: bool,
        chat_id: &str,
        mentions_bot: bool,
    ) -> bool {
        if !is_group {
            return true;
        }
        if !chat_id.is_empty() && self.free_response_chats.contains(chat_id) {
            return true;
        }
        if !self.require_mention {
            return true;
        }
        if mentions_bot {
            return true;
        }
        self.message_matches_mention_patterns(text)
    }
}

/// Mirrors `_dingtalk_require_mention`.
pub fn resolve_require_mention(extra: &Value) -> bool {
    if let Some(configured) = extra.get("require_mention") {
        if !configured.is_null() {
            return match configured {
                Value::String(s) => is_truthy_str(s),
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                Value::Null => false,
                // Non-empty arrays/objects are truthy in Python's bool().
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
        }
    }
    is_truthy_str(&std::env::var("DINGTALK_REQUIRE_MENTION").unwrap_or_else(|_| "false".into()))
}

/// Mirrors `_dingtalk_free_response_chats`.
pub fn resolve_free_response_chats(extra: &Value) -> std::collections::HashSet<String> {
    let raw = extra.get("free_response_chats");
    match raw {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(value_to_string)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::Null) | None => {
            let env_raw = std::env::var("DINGTALK_FREE_RESPONSE_CHATS").unwrap_or_default();
            split_csv_set(&env_raw)
        }
        Some(other) => split_csv_set(&value_to_string(other)),
    }
}

fn split_csv_set(raw: &str) -> std::collections::HashSet<String> {
    raw.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Mirrors `_compile_mention_patterns`. Patterns compile case-insensitively;
/// invalid patterns are skipped.
pub fn compile_mention_patterns(extra: &Value) -> Vec<Regex> {
    let mut patterns: Option<Vec<String>> = None;

    match extra.get("mention_patterns") {
        Some(Value::Null) | None => {
            let raw = std::env::var("DINGTALK_MENTION_PATTERNS")
                .unwrap_or_default()
                .trim()
                .to_string();
            if !raw.is_empty() {
                // Try JSON first.
                let loaded: Vec<String> = match serde_json::from_str::<Value>(&raw) {
                    Ok(Value::Array(arr)) => {
                        arr.iter().map(value_to_string).collect()
                    }
                    Ok(Value::String(s)) => vec![s],
                    Ok(_) | Err(_) => {
                        // Fall back to line-split, then comma-split.
                        let by_line: Vec<String> = raw
                            .lines()
                            .map(|p| p.trim().to_string())
                            .filter(|p| !p.is_empty())
                            .collect();
                        if !by_line.is_empty() {
                            by_line
                        } else {
                            raw.split(',')
                                .map(|p| p.trim().to_string())
                                .filter(|p| !p.is_empty())
                                .collect()
                        }
                    }
                };
                patterns = Some(loaded);
            }
        }
        Some(Value::String(s)) => patterns = Some(vec![s.clone()]),
        Some(Value::Array(arr)) => {
            patterns = Some(arr.iter().map(value_to_string).collect());
        }
        Some(_) => {
            // Not a list or string -> warn + empty.
            return Vec::new();
        }
    }

    let patterns = match patterns {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut compiled = Vec::new();
    for pattern in patterns {
        if pattern.trim().is_empty() {
            continue;
        }
        // re.IGNORECASE
        if let Ok(re) = Regex::new(&format!("(?i){pattern}")) {
            compiled.push(re);
        }
    }
    compiled
}

/// Mirrors `_load_allowed_users`.
pub fn load_allowed_users(extra: &Value) -> std::collections::HashSet<String> {
    let items: Vec<String> = match extra.get("allowed_users") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(value_to_string)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::Null) | None => {
            let raw = std::env::var("DINGTALK_ALLOWED_USERS").unwrap_or_default();
            raw.split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect()
        }
        Some(other) => value_to_string(other)
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
    };
    items.into_iter().map(|s| s.to_lowercase()).collect()
}

// ===========================================================================
// Text / media extraction
// ===========================================================================

/// Extract plain text from a DingTalk chatbot message. Mirrors `_extract_text`.
///
/// We do NOT strip "@bot" — the mention is a routing signal delivered
/// structurally via `is_in_at_list`.
pub fn extract_text(message: &ChatbotMessage) -> String {
    let mut content = message
        .text
        .clone()
        .unwrap_or_default()
        .trim()
        .to_string();

    if content.is_empty() && !message.rich_text_list.is_empty() {
        let mut parts: Vec<String> = Vec::new();
        for item in &message.rich_text_list {
            if let Some(t) = item.get_str("text").filter(|s| !s.is_empty()) {
                parts.push(t);
            } else if let Some(c) = item.get_str("content").filter(|s| !s.is_empty()) {
                parts.push(c);
            }
        }
        content = parts.join(" ").trim().to_string();
    }

    content
}

/// Extract media info from a message. Mirrors `_extract_media`. Returns
/// `(MessageType, media_urls/codes, media_types)`.
pub fn extract_media(message: &ChatbotMessage) -> (MessageType, Vec<String>, Vec<String>) {
    let mut msg_type = MessageType::Text;
    let mut media_urls: Vec<String> = Vec::new();
    let mut media_types: Vec<String> = Vec::new();

    // Single image / picture.
    if let Some(code) = &message.image_download_code {
        if !code.is_empty() {
            media_urls.push(code.clone());
            media_types.push("image".to_string());
            msg_type = MessageType::Photo;
        }
    }

    // Rich text with mixed content.
    for item in &message.rich_text_list {
        let dl_code = item
            .get_str("downloadCode")
            .or_else(|| item.get_str("download_code"))
            .unwrap_or_default();
        let item_type = item.get_str("type").unwrap_or_default();
        if !dl_code.is_empty() {
            let mapped = dingtalk_type_mapping(&item_type);
            media_urls.push(dl_code);
            match mapped {
                "image" => {
                    media_types.push("image".to_string());
                    if msg_type == MessageType::Text {
                        msg_type = MessageType::Photo;
                    }
                }
                "audio" => {
                    media_types.push("audio".to_string());
                    if msg_type == MessageType::Text {
                        msg_type = MessageType::Audio;
                    }
                }
                "video" => {
                    media_types.push("video".to_string());
                    if msg_type == MessageType::Text {
                        msg_type = MessageType::Video;
                    }
                }
                _ => {
                    media_types.push("application/octet-stream".to_string());
                    if msg_type == MessageType::Text {
                        msg_type = MessageType::Document;
                    }
                }
            }
        }
    }

    let msg_type_str = message.message_type.as_str();
    if msg_type_str == "picture" && media_urls.is_empty() {
        msg_type = MessageType::Photo;
    } else if msg_type_str == "richText" {
        msg_type = if media_types.iter().any(|t| t.contains("image")) {
            MessageType::Photo
        } else {
            MessageType::Text
        };
    }

    (msg_type, media_urls, media_types)
}

// ===========================================================================
// Markdown normalization
// ===========================================================================

/// Normalize markdown for DingTalk's parser. Mirrors `_normalize_markdown`.
///
/// - Inserts a blank line before numbered list items that follow a
///   non-numbered, non-blank line.
/// - Dedents indented fenced code blocks (lines whose stripped form starts
///   with ```` ``` ````).
pub fn normalize_markdown(text: &str) -> String {
    let numbered = Regex::new(r"^\d+\.\s").unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<String> = Vec::new();

    for (i, &line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let is_numbered = numbered.is_match(trimmed);
        if is_numbered && i > 0 {
            let prev = lines[i - 1];
            let prev_trim = prev.trim();
            if !prev_trim.is_empty() && !numbered.is_match(prev_trim) {
                out.push(String::new());
            }
        }
        // Dedent fenced code blocks.
        let mut emitted = line.to_string();
        let lstripped = line.trim_start();
        if lstripped.starts_with("```") && line != lstripped {
            emitted = lstripped.to_string();
        }
        out.push(emitted);
    }

    out.join("\n")
}

// ===========================================================================
// Session-webhook cache
// ===========================================================================

/// Per-chat session-webhook cache with FIFO size cap + expiry safety-margin.
///
/// Mirrors the `_session_webhooks` dict plus `_get_valid_webhook`. Values are
/// `(webhook_url, expired_time_ms)`. A `BTreeMap`-backed insertion order is not
/// required since Python's eviction pops an arbitrary `next(iter(...))`; we use
/// an explicit insertion-order queue to keep eviction deterministic.
#[derive(Debug, Default)]
pub struct SessionWebhookCache {
    map: HashMap<String, (String, i64)>,
    order: Vec<String>,
}

impl SessionWebhookCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate a candidate webhook against the host pattern, then insert,
    /// evicting the oldest entry when at capacity. Mirrors the storage block in
    /// `_on_message`. Returns true when stored.
    pub fn store(&mut self, chat_id: &str, webhook: &str, expired_time_ms: i64) -> bool {
        if webhook.is_empty() || chat_id.is_empty() {
            return false;
        }
        if !dingtalk_webhook_re().is_match(webhook) {
            return false;
        }
        if !self.map.contains_key(chat_id) && self.map.len() >= SESSION_WEBHOOKS_MAX {
            if let Some(oldest) = self.order.first().cloned() {
                self.map.remove(&oldest);
                self.order.remove(0);
            }
        }
        if !self.map.contains_key(chat_id) {
            self.order.push(chat_id.to_string());
        }
        self.map
            .insert(chat_id.to_string(), (webhook.to_string(), expired_time_ms));
        true
    }

    /// Get a valid (non-expired) session webhook. Mirrors `_get_valid_webhook`.
    /// Expired entries are evicted and `None` returned. `now_ms` is the current
    /// epoch milliseconds (injected for testability).
    pub fn get_valid(&mut self, chat_id: &str, now_ms: i64) -> Option<(String, i64)> {
        let info = self.map.get(chat_id).cloned()?;
        let (_, expired_time_ms) = info.clone();
        if expired_time_ms > 0 && now_ms + WEBHOOK_SAFETY_MARGIN_MS >= expired_time_ms {
            self.map.remove(chat_id);
            self.order.retain(|k| k != chat_id);
            return None;
        }
        Some(info)
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ===========================================================================
// Outbound webhook payload + send
// ===========================================================================

/// Build the markdown webhook payload. Mirrors the `payload` dict in `send()`.
///
/// Truncates `content` to [`MAX_MESSAGE_LENGTH`] chars (codepoints), then
/// normalizes markdown for DingTalk.
pub fn build_webhook_payload(content: &str) -> Value {
    let truncated: String = content.chars().take(MAX_MESSAGE_LENGTH).collect();
    let normalized = normalize_markdown(&truncated);
    json!({
        "msgtype": "markdown",
        "markdown": {
            "title": "Hermes",
            "text": normalized,
        }
    })
}

/// Post a markdown reply to a DingTalk session webhook (blocking).
///
/// Mirrors the webhook branch of `send()`: 15s timeout, `<300` is success.
/// Returns a [`SendResult`]; on success `message_id` is a fresh 12-hex token
/// (matching `uuid.uuid4().hex[:12]`).
pub fn post_webhook(session_webhook: &str, content: &str) -> SendResult {
    let payload = build_webhook_payload(content);
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => return SendResult::fail(e.to_string()),
    };
    match client.post(session_webhook).json(&payload).send() {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if status < 300 {
                SendResult::ok(Some(uuid12()))
            } else {
                let body = resp.text().unwrap_or_default();
                let body: String = body.chars().take(200).collect();
                SendResult::fail(format!("HTTP {status}: {body}"))
            }
        }
        Err(e) => {
            if e.is_timeout() {
                SendResult::fail("Timeout sending message to DingTalk")
            } else {
                SendResult::fail(e.to_string())
            }
        }
    }
}

// ===========================================================================
// AI-Card lifecycle state + request construction
// ===========================================================================

/// Tracks AI-Card streaming state + done-reaction idempotency for an adapter.
///
/// Mirrors `_streaming_cards`, `_done_emoji_fired`, and `_message_contexts`
/// (the chat-keyed parts the lifecycle logic reads). The networked SDK calls
/// (`create_card`, `deliver_card`, `streaming_update`, emotion reply/recall)
/// live in the runtime layer; this struct owns the deterministic bookkeeping.
#[derive(Debug, Default)]
pub struct CardLifecycle {
    /// chat_id -> { out_track_id -> last_content }.
    pub streaming_cards: HashMap<String, HashMap<String, String>>,
    /// chats whose Done reaction has already fired this inbound cycle.
    pub done_emoji_fired: std::collections::HashSet<String>,
}

impl CardLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the per-chat Done marker. Mirrors the
    /// `self._done_emoji_fired.discard(chat_id)` reset on each inbound message.
    pub fn reset_done_marker(&mut self, chat_id: &str) {
        self.done_emoji_fired.remove(chat_id);
    }

    /// Idempotent "fire done reaction" guard. Mirrors the first lines of
    /// `_fire_done_reaction`: returns true the first time (caller should fire),
    /// false on subsequent calls for the same chat.
    pub fn mark_done_fired(&mut self, chat_id: &str) -> bool {
        self.done_emoji_fired.insert(chat_id.to_string())
    }

    /// Track a card kept open for streaming. Mirrors
    /// `self._streaming_cards.setdefault(chat_id, {})[out_track_id] = content`.
    pub fn track_streaming(&mut self, chat_id: &str, out_track_id: &str, content: &str) {
        self.streaming_cards
            .entry(chat_id.to_string())
            .or_default()
            .insert(out_track_id.to_string(), content.to_string());
    }

    /// Pop+return all streaming siblings for a chat. Mirrors
    /// `self._streaming_cards.pop(chat_id, None)` in `_close_streaming_siblings`.
    pub fn take_streaming_siblings(&mut self, chat_id: &str) -> Option<HashMap<String, String>> {
        self.streaming_cards.remove(chat_id)
    }

    /// Remove a finalized card from streaming tracking, pruning the chat entry
    /// when empty. Mirrors the finalize branch of `edit_message`.
    pub fn untrack_streaming(&mut self, chat_id: &str, out_track_id: &str) {
        if let Some(cards) = self.streaming_cards.get_mut(chat_id) {
            cards.remove(out_track_id);
            if cards.is_empty() {
                self.streaming_cards.remove(chat_id);
            }
        }
    }
}

/// Generate a fresh AI-Card `out_track_id` (`hermes_<12 hex>`). Mirrors the
/// `f"hermes_{uuid.uuid4().hex[:12]}"` in `_create_and_stream_card`.
pub fn new_out_track_id() -> String {
    format!("hermes_{}", uuid12())
}

/// Open-space id for delivering an AI Card. Mirrors the `open_space_id`
/// construction in `_create_and_stream_card`.
pub fn card_open_space_id(is_group: bool, conversation_id: &str, sender_staff_id: &str) -> String {
    if is_group {
        format!("dtv1.card//IM_GROUP.{conversation_id}")
    } else {
        format!("dtv1.card//IM_ROBOT.{sender_staff_id}")
    }
}

/// Build the `card_param_map` payload for card creation. Mirrors the
/// `card_data.card_param_map={"content": ""}` create request.
pub fn build_create_card_param_map() -> Value {
    json!({ "content": "" })
}

/// Build the streaming-update request body. Mirrors `StreamingUpdateRequest`
/// in `_stream_card_content` (full content, truncated to `MAX_MESSAGE_LENGTH`).
pub fn build_streaming_update_request(
    out_track_id: &str,
    guid: &str,
    content: &str,
    finalize: bool,
) -> Value {
    let truncated: String = content.chars().take(MAX_MESSAGE_LENGTH).collect();
    json!({
        "outTrackId": out_track_id,
        "guid": guid,
        "key": "content",
        "content": truncated,
        "isFull": true,
        "isFinalize": finalize,
        "isError": false,
    })
}

// ===========================================================================
// Inbound processing pipeline (pure portion)
// ===========================================================================

/// Outcome of the early inbound-admission checks. Mirrors the first half of
/// `_on_message` before media resolution / dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Drop: duplicate message id.
    Duplicate,
    /// Drop: sender not in `allowed_users`.
    NotAllowed,
    /// Drop: group message failed the mention gate.
    MentionGate,
    /// Accept and process.
    Process,
}

/// Run the inbound-admission gate. Mirrors the dedup → allowed-users →
/// mention-gate sequence at the top of `_on_message`. `dedup` is mutated.
pub fn admit_message(
    message: &ChatbotMessage,
    gating: &GroupGating,
    dedup: &mut MessageDeduplicator,
) -> (AdmissionDecision, String, bool) {
    let msg_id = message
        .message_id
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(uuid_hex);

    if dedup.is_duplicate(&msg_id) {
        return (AdmissionDecision::Duplicate, msg_id, false);
    }

    let conversation_id = message.conversation_id.clone();
    let is_group = message.conversation_type == "2";
    let sender_id = message.sender_id.clone();
    let chat_id = if !conversation_id.is_empty() {
        conversation_id
    } else {
        sender_id.clone()
    };

    if !gating.is_user_allowed(&sender_id, &message.sender_staff_id) {
        return (AdmissionDecision::NotAllowed, msg_id, is_group);
    }

    let early_text = extract_text(message);
    if !gating.should_process_message(&early_text, is_group, &chat_id, message.is_in_at_list) {
        return (AdmissionDecision::MentionGate, msg_id, is_group);
    }

    let _ = chat_id;
    (AdmissionDecision::Process, msg_id, is_group)
}

/// Build a normalized [`MessageEvent`] from an admitted message. Mirrors the
/// `MessageEvent(...)` construction in `_on_message` (text + media extraction,
/// source build, timestamp parse). Returns `None` for the empty-message skip
/// (`not text and not media_urls`).
///
/// `now_ms` is the current epoch milliseconds used as the fallback timestamp.
pub fn build_message_event(
    message: &ChatbotMessage,
    msg_id: &str,
    now_ms: i64,
) -> Option<MessageEvent> {
    let conversation_id = message.conversation_id.clone();
    let is_group = message.conversation_type == "2";
    let sender_id = message.sender_id.clone();
    let sender_nick = if message.sender_nick.is_empty() {
        sender_id.clone()
    } else {
        message.sender_nick.clone()
    };
    let chat_id = if !conversation_id.is_empty() {
        conversation_id
    } else {
        sender_id.clone()
    };
    let chat_type = if is_group { "group" } else { "dm" };

    let text = extract_text(message);
    let (msg_type, media_urls, media_types) = extract_media(message);

    if text.is_empty() && media_urls.is_empty() {
        return None;
    }

    let source = SessionSource {
        platform: "dingtalk".to_string(),
        chat_id: chat_id.clone(),
        chat_name: message.conversation_title.clone(),
        chat_type: chat_type.to_string(),
        user_id: Some(sender_id.clone()),
        user_name: Some(sender_nick),
        user_id_alt: if message.sender_staff_id.is_empty() {
            None
        } else {
            Some(message.sender_staff_id.clone())
        },
        ..Default::default()
    };

    // Timestamp: create_at is epoch ms; on parse failure use now_ms.
    let ts_ms = message.create_at.filter(|&v| v != 0).unwrap_or(now_ms);
    let timestamp = chrono::DateTime::from_timestamp_millis(ts_ms)
        .unwrap_or_else(|| {
            chrono::DateTime::from_timestamp_millis(now_ms)
                .unwrap_or_else(chrono::Utc::now)
        });
    let _ = timestamp; // MessageEvent carries no timestamp field in the base port.

    Some(MessageEvent {
        text,
        message_type: msg_type,
        source,
        message_id: Some(msg_id.to_string()),
        media_urls,
        media_types,
        ..Default::default()
    })
}

// ===========================================================================
// uuid helpers (match uuid.uuid4().hex / .hex[:12])
// ===========================================================================

fn uuid_bytes() -> [u8; 16] {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mut x = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0xD6E8_FEB8_6659_FD93);
    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(8) {
        // xorshift-ish mixing
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        let v = (x & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        chunk.copy_from_slice(&v.to_le_bytes()[..chunk.len()]);
    }
    out
}

/// 32-hex-char uuid (matches `uuid.uuid4().hex`).
pub fn uuid_hex() -> String {
    let b = uuid_bytes();
    let mut s = String::with_capacity(32);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

/// 12-hex-char uuid prefix (matches `uuid.uuid4().hex[:12]`).
pub fn uuid12() -> String {
    uuid_hex().chars().take(12).collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_need_both_creds() {
        unsafe {
            std::env::remove_var("DINGTALK_CLIENT_ID");
            std::env::remove_var("DINGTALK_CLIENT_SECRET");
        }
        assert!(!check_dingtalk_requirements());
        unsafe {
            std::env::set_var("DINGTALK_CLIENT_ID", "id");
        }
        assert!(!check_dingtalk_requirements());
        unsafe {
            std::env::set_var("DINGTALK_CLIENT_SECRET", "secret");
        }
        assert!(check_dingtalk_requirements());
        unsafe {
            std::env::remove_var("DINGTALK_CLIENT_ID");
            std::env::remove_var("DINGTALK_CLIENT_SECRET");
        }
    }

    #[test]
    fn webhook_regex_matches_dingtalk_hosts() {
        let re = dingtalk_webhook_re();
        assert!(re.is_match("https://api.dingtalk.com/robot/send?access_token=x"));
        assert!(re.is_match("https://oapi.dingtalk.com/robot/send"));
        assert!(!re.is_match("https://evil.com/robot/send"));
        assert!(!re.is_match("http://api.dingtalk.com/robot/send")); // not https
    }

    #[test]
    fn type_mapping_defaults_to_file() {
        assert_eq!(dingtalk_type_mapping("picture"), "image");
        assert_eq!(dingtalk_type_mapping("voice"), "audio");
        assert_eq!(dingtalk_type_mapping("unknown"), "file");
        assert_eq!(dingtalk_type_mapping("video"), "file");
    }

    #[test]
    fn require_mention_config_and_env() {
        // explicit bool true
        let cfg = json!({"require_mention": true});
        assert!(resolve_require_mention(&cfg));
        // explicit string
        let cfg = json!({"require_mention": "yes"});
        assert!(resolve_require_mention(&cfg));
        let cfg = json!({"require_mention": "no"});
        assert!(!resolve_require_mention(&cfg));
        // env fallback
        unsafe {
            std::env::set_var("DINGTALK_REQUIRE_MENTION", "on");
        }
        assert!(resolve_require_mention(&json!({})));
        unsafe {
            std::env::remove_var("DINGTALK_REQUIRE_MENTION");
        }
        assert!(!resolve_require_mention(&json!({})));
    }

    #[test]
    fn free_response_chats_list_and_csv() {
        let cfg = json!({"free_response_chats": ["cidA==", " cidB== "]});
        let set = resolve_free_response_chats(&cfg);
        assert!(set.contains("cidA=="));
        assert!(set.contains("cidB=="));
        // env csv
        unsafe {
            std::env::set_var("DINGTALK_FREE_RESPONSE_CHATS", "x, y ,");
        }
        let set = resolve_free_response_chats(&json!({}));
        assert!(set.contains("x"));
        assert!(set.contains("y"));
        assert_eq!(set.len(), 2);
        unsafe {
            std::env::remove_var("DINGTALK_FREE_RESPONSE_CHATS");
        }
    }

    #[test]
    fn allowed_users_lowercased_and_wildcard() {
        let cfg = json!({"allowed_users": ["Manager1234", "Boss"]});
        let g = GroupGating::from_config(&cfg);
        assert!(g.is_user_allowed("manager1234", ""));
        assert!(g.is_user_allowed("", "boss"));
        assert!(!g.is_user_allowed("intruder", "nope"));

        let cfg = json!({"allowed_users": ["*"]});
        let g = GroupGating::from_config(&cfg);
        assert!(g.is_user_allowed("anyone", "anything"));

        // empty -> allow all
        let g = GroupGating::from_config(&json!({}));
        unsafe {
            std::env::remove_var("DINGTALK_ALLOWED_USERS");
        }
        let g2 = GroupGating::from_config(&json!({}));
        let _ = g;
        assert!(g2.is_user_allowed("x", "y"));
    }

    #[test]
    fn mention_patterns_compile_and_match() {
        let cfg = json!({"mention_patterns": ["^小马", "hermes"]});
        let g = GroupGating::from_config(&cfg);
        assert!(g.message_matches_mention_patterns("小马 hello"));
        assert!(g.message_matches_mention_patterns("hey HERMES")); // ignorecase
        assert!(!g.message_matches_mention_patterns("nothing here"));

        // single string form
        let cfg = json!({"mention_patterns": "^bot"});
        let g = GroupGating::from_config(&cfg);
        assert!(g.message_matches_mention_patterns("bot wake"));
    }

    #[test]
    fn should_process_message_rules() {
        // DM always processed
        let g = GroupGating::from_config(&json!({"require_mention": true}));
        assert!(g.should_process_message("hi", false, "chat", false));
        // group, require_mention, no mention, no pattern -> drop
        assert!(!g.should_process_message("hi", true, "chat", false));
        // group, mentioned -> process
        assert!(g.should_process_message("hi", true, "chat", true));
        // free_response_chats bypass
        let g = GroupGating::from_config(
            &json!({"require_mention": true, "free_response_chats": ["chatX"]}),
        );
        assert!(g.should_process_message("hi", true, "chatX", false));
        // require_mention disabled -> process
        let g = GroupGating::from_config(&json!({"require_mention": false}));
        assert!(g.should_process_message("hi", true, "chat", false));
    }

    #[test]
    fn extract_text_dict_and_richtext() {
        let mut m = ChatbotMessage {
            text: Some("  hello world  ".into()),
            ..Default::default()
        };
        assert_eq!(extract_text(&m), "hello world");

        // rich text fallback
        m.text = None;
        let mut item1 = RichTextItem::default();
        item1.set_str("text", "alpha");
        let mut item2 = RichTextItem::default();
        item2.set_str("content", "beta");
        m.rich_text_list = vec![item1, item2];
        assert_eq!(extract_text(&m), "alpha beta");
    }

    #[test]
    fn extract_text_from_value_payload() {
        let data = json!({
            "msgId": "m1",
            "conversationType": "2",
            "text": {"content": "hi there"},
            "isInAtList": true,
            "sessionWebhook": "https://oapi.dingtalk.com/robot/send?access_token=t",
        });
        let m = ChatbotMessage::from_value(&data);
        assert_eq!(m.message_id.as_deref(), Some("m1"));
        assert_eq!(m.conversation_type, "2");
        assert_eq!(extract_text(&m), "hi there");
        assert!(m.is_in_at_list);
        assert!(m.session_webhook.contains("oapi.dingtalk.com"));
    }

    #[test]
    fn extract_media_image_and_richtext() {
        let m = ChatbotMessage {
            image_download_code: Some("dl_img".into()),
            ..Default::default()
        };
        let (t, urls, types) = extract_media(&m);
        assert_eq!(t, MessageType::Photo);
        assert_eq!(urls, vec!["dl_img".to_string()]);
        assert_eq!(types, vec!["image".to_string()]);

        // rich text audio + file
        let mut audio = RichTextItem::default();
        audio.set_str("downloadCode", "dl_a");
        audio.set_str("type", "voice");
        let mut file = RichTextItem::default();
        file.set_str("downloadCode", "dl_f");
        file.set_str("type", "other");
        let m = ChatbotMessage {
            rich_text_list: vec![audio, file],
            ..Default::default()
        };
        let (t, urls, types) = extract_media(&m);
        assert_eq!(t, MessageType::Audio);
        assert_eq!(urls, vec!["dl_a".to_string(), "dl_f".to_string()]);
        assert_eq!(
            types,
            vec!["audio".to_string(), "application/octet-stream".to_string()]
        );
    }

    #[test]
    fn normalize_markdown_blank_line_before_numbered() {
        let input = "Intro text\n1. first\n2. second";
        let out = normalize_markdown(input);
        let lines: Vec<&str> = out.split('\n').collect();
        // blank line inserted before "1. first"
        assert_eq!(lines[0], "Intro text");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "1. first");
        assert_eq!(lines[3], "2. second");
    }

    #[test]
    fn normalize_markdown_dedents_fence() {
        let input = "    ```python";
        let out = normalize_markdown(input);
        assert_eq!(out, "```python");
    }

    #[test]
    fn session_webhook_cache_store_and_expiry() {
        let mut cache = SessionWebhookCache::new();
        let wh = "https://api.dingtalk.com/robot/send?access_token=t";
        // reject non-dingtalk host
        assert!(!cache.store("c1", "https://evil.com/x", 0));
        // store valid, no expiry
        assert!(cache.store("c1", wh, 0));
        assert_eq!(cache.get_valid("c1", 1_000_000).map(|(w, _)| w), Some(wh.to_string()));

        // expiring entry: now + 5min margin >= expiry
        assert!(cache.store("c2", wh, 1_000_000 + WEBHOOK_SAFETY_MARGIN_MS));
        // at now=1_000_001 the margin pushes us past expiry -> evicted
        assert_eq!(cache.get_valid("c2", 1_000_001), None);
        assert!(cache.get_valid("c2", 1).is_none() || cache.is_empty());
    }

    #[test]
    fn session_webhook_cache_evicts_oldest_at_cap() {
        let mut cache = SessionWebhookCache::new();
        let wh = "https://api.dingtalk.com/robot/send";
        for i in 0..SESSION_WEBHOOKS_MAX {
            assert!(cache.store(&format!("c{i}"), wh, 0));
        }
        assert_eq!(cache.len(), SESSION_WEBHOOKS_MAX);
        // one more evicts oldest (c0)
        assert!(cache.store("cNEW", wh, 0));
        assert_eq!(cache.len(), SESSION_WEBHOOKS_MAX);
        assert!(cache.get_valid("c0", 0).is_none());
        assert!(cache.get_valid("cNEW", 0).is_some());
    }

    #[test]
    fn build_webhook_payload_shape() {
        let payload = build_webhook_payload("hello\n1. x");
        assert_eq!(payload["msgtype"], "markdown");
        assert_eq!(payload["markdown"]["title"], "Hermes");
        let text = payload["markdown"]["text"].as_str().unwrap();
        // blank line inserted before numbered item
        assert!(text.contains("hello\n\n1. x"));
    }

    #[test]
    fn streaming_update_request_truncates_and_flags() {
        let big = "x".repeat(MAX_MESSAGE_LENGTH + 50);
        let req = build_streaming_update_request("ot1", "guid1", &big, true);
        assert_eq!(req["outTrackId"], "ot1");
        assert_eq!(req["key"], "content");
        assert_eq!(req["isFull"], true);
        assert_eq!(req["isFinalize"], true);
        assert_eq!(req["isError"], false);
        assert_eq!(
            req["content"].as_str().unwrap().chars().count(),
            MAX_MESSAGE_LENGTH
        );
    }

    #[test]
    fn card_open_space_id_group_vs_dm() {
        assert_eq!(
            card_open_space_id(true, "cid123", "staff"),
            "dtv1.card//IM_GROUP.cid123"
        );
        assert_eq!(
            card_open_space_id(false, "cid123", "staff789"),
            "dtv1.card//IM_ROBOT.staff789"
        );
    }

    #[test]
    fn card_lifecycle_done_idempotency() {
        let mut lc = CardLifecycle::new();
        assert!(lc.mark_done_fired("chat"));
        assert!(!lc.mark_done_fired("chat"));
        lc.reset_done_marker("chat");
        assert!(lc.mark_done_fired("chat"));
    }

    #[test]
    fn card_lifecycle_streaming_tracking() {
        let mut lc = CardLifecycle::new();
        lc.track_streaming("chat", "ot1", "content1");
        lc.track_streaming("chat", "ot2", "content2");
        let siblings = lc.take_streaming_siblings("chat").unwrap();
        assert_eq!(siblings.len(), 2);
        // popped -> gone
        assert!(lc.take_streaming_siblings("chat").is_none());

        lc.track_streaming("chat", "ot1", "c");
        lc.untrack_streaming("chat", "ot1");
        assert!(lc.streaming_cards.get("chat").is_none());
    }

    #[test]
    fn out_track_id_prefix() {
        let id = new_out_track_id();
        assert!(id.starts_with("hermes_"));
        assert_eq!(id.len(), "hermes_".len() + 12);
    }

    #[test]
    fn admit_message_flow() {
        let gating = GroupGating::from_config(&json!({"require_mention": true}));
        let mut dedup = MessageDeduplicator::new();

        // group message without mention -> MentionGate
        let m = ChatbotMessage {
            message_id: Some("g1".into()),
            conversation_id: "cid".into(),
            conversation_type: "2".into(),
            text: Some("hello".into()),
            ..Default::default()
        };
        let (d, id, is_group) = admit_message(&m, &gating, &mut dedup);
        assert_eq!(d, AdmissionDecision::MentionGate);
        assert_eq!(id, "g1");
        assert!(is_group);

        // DM -> Process
        let m = ChatbotMessage {
            message_id: Some("d1".into()),
            conversation_id: "cid2".into(),
            conversation_type: "1".into(),
            text: Some("hi".into()),
            ..Default::default()
        };
        let (d, _, is_group) = admit_message(&m, &gating, &mut dedup);
        assert_eq!(d, AdmissionDecision::Process);
        assert!(!is_group);

        // duplicate -> Duplicate
        let (d, _, _) = admit_message(&m, &gating, &mut dedup);
        assert_eq!(d, AdmissionDecision::Duplicate);
    }

    #[test]
    fn admit_message_allowed_users_gate() {
        let gating = GroupGating::from_config(&json!({"allowed_users": ["boss"]}));
        let mut dedup = MessageDeduplicator::new();
        let m = ChatbotMessage {
            message_id: Some("x1".into()),
            conversation_type: "1".into(),
            sender_id: "intruder".into(),
            text: Some("hi".into()),
            ..Default::default()
        };
        let (d, _, _) = admit_message(&m, &gating, &mut dedup);
        assert_eq!(d, AdmissionDecision::NotAllowed);
    }

    #[test]
    fn build_event_empty_skips() {
        let m = ChatbotMessage {
            conversation_type: "1".into(),
            sender_id: "s".into(),
            text: None,
            ..Default::default()
        };
        assert!(build_message_event(&m, "id", 0).is_none());
    }

    #[test]
    fn build_event_populates_source() {
        let m = ChatbotMessage {
            message_id: Some("m1".into()),
            conversation_id: "cid".into(),
            conversation_type: "2".into(),
            conversation_title: Some("Group X".into()),
            sender_id: "u1".into(),
            sender_nick: "Alice".into(),
            sender_staff_id: "staff1".into(),
            text: Some("hello".into()),
            create_at: Some(1_700_000_000_000),
            ..Default::default()
        };
        let ev = build_message_event(&m, "m1", 0).unwrap();
        assert_eq!(ev.text, "hello");
        assert_eq!(ev.source.chat_id, "cid");
        assert_eq!(ev.source.chat_type, "group");
        assert_eq!(ev.source.chat_name.as_deref(), Some("Group X"));
        assert_eq!(ev.source.user_id.as_deref(), Some("u1"));
        assert_eq!(ev.source.user_name.as_deref(), Some("Alice"));
        assert_eq!(ev.source.user_id_alt.as_deref(), Some("staff1"));
        assert_eq!(ev.message_id.as_deref(), Some("m1"));
    }

    #[test]
    fn uuid12_is_12_hex() {
        let s = uuid12();
        assert_eq!(s.len(), 12);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
        // uuid_hex is 32
        assert_eq!(uuid_hex().len(), 32);
    }
}
