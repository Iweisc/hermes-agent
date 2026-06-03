//! Mattermost platform adapter — native Rust port of
//! `gateway/platforms/mattermost.py`.
//!
//! The Python original drives an `aiohttp` REST client + WebSocket listener with
//! `asyncio` task orchestration and exponential-backoff reconnection. That live
//! coroutine machinery has no faithful equivalent in the synchronous crate set
//! available here, so this module ports the **deterministic, behavior-defining
//! logic** that the rest of Hermes (and tests) depend on, reproducing the Python
//! behavior exactly:
//!
//! - Requirements gating ([`check_mattermost_requirements`]).
//! - Config / env resolution ([`MattermostConfig`]): base URL, token, reply
//!   mode, mention gating, free-response channels.
//! - HTTP request construction / response parsing with `reqwest::blocking`
//!   ([`api_get`], [`api_post`], [`api_put`], [`upload_file`], [`create_post`]).
//! - Pure helpers: [`headers`], [`api_url`], [`websocket_url`],
//!   [`ws_auth_message`], [`format_message`], [`channel_type`],
//!   [`reconnect_delay_with_jitter`], [`next_reconnect_delay`],
//!   [`is_permanent_ws_error`].
//! - Inbound `posted` event parsing + mention-gating + message-type
//!   determination ([`parse_posted_event`], [`PostedDecision`]).
//!
//! The live WebSocket loop and the cross-coroutine session lifecycle are out of
//! scope; the async runtime layer drives this logic. Network helpers preserve
//! the exact API shapes Mattermost v4 expects.
//!
//! Cross-refs:
//!   - [`crate::gw_helpers::MessageDeduplicator`]
//!   - [`crate::gw_platforms_base::MessageType`]
//!   - [`crate::gw_platforms_base::SendResult`]
//!   - [`crate::tool_url_safety::is_safe_url`]

use std::collections::HashMap;

use regex::Regex;
use serde_json::{json, Value};

use crate::gw_platforms_base::{MessageType, SendResult};

// ===========================================================================
// Constants
// ===========================================================================

/// Mattermost post size limit (server default is 16383, but 4000 is the
/// practical limit for readable messages). Mirrors `MAX_POST_LENGTH`.
pub const MAX_POST_LENGTH: usize = 4000;

/// Reconnect base delay in seconds. Mirrors `_RECONNECT_BASE_DELAY`.
pub const RECONNECT_BASE_DELAY: f64 = 2.0;
/// Reconnect max delay in seconds. Mirrors `_RECONNECT_MAX_DELAY`.
pub const RECONNECT_MAX_DELAY: f64 = 60.0;
/// Reconnect jitter fraction. Mirrors `_RECONNECT_JITTER`.
pub const RECONNECT_JITTER: f64 = 0.2;

/// Mattermost post `file_ids` cap (used when batching multi-image sends).
pub const MULTI_IMAGE_CHUNK: usize = 5;

/// Channel type codes returned by the Mattermost API, mapped to Hermes chat
/// types. Mirrors `_CHANNEL_TYPE_MAP`.
pub fn channel_type(code: &str) -> &'static str {
    match code {
        "D" => "dm",
        "G" => "group",
        "P" => "group", // private channel → treat as group
        "O" => "channel",
        _ => "channel",
    }
}

// ===========================================================================
// Requirements gating
// ===========================================================================

/// Return True if the Mattermost adapter can be used.
///
/// Mirrors `check_mattermost_requirements`. The Python version also checks for
/// the `aiohttp` import; in the native build the HTTP client is always
/// available, so that branch is implicitly satisfied (`http_available` defaults
/// to true via [`check_mattermost_requirements`]).
pub fn check_mattermost_requirements_from(token: &str, url: &str, http_available: bool) -> bool {
    if token.is_empty() {
        return false;
    }
    if url.is_empty() {
        return false;
    }
    http_available
}

/// Convenience wrapper reading `MATTERMOST_TOKEN` / `MATTERMOST_URL` from env.
pub fn check_mattermost_requirements() -> bool {
    let token = std::env::var("MATTERMOST_TOKEN").unwrap_or_default();
    let url = std::env::var("MATTERMOST_URL").unwrap_or_default();
    check_mattermost_requirements_from(&token, &url, true)
}

// ===========================================================================
// Config resolution
// ===========================================================================

/// Resolved Mattermost adapter configuration. Mirrors the fields computed in
/// `MattermostAdapter.__init__`.
#[derive(Debug, Clone, Default)]
pub struct MattermostConfig {
    pub base_url: String,
    pub token: String,
    pub reply_mode: String,
}

impl MattermostConfig {
    /// Resolve config from `config.extra` (JSON), the platform `config.token`,
    /// and environment variables. Mirrors the `__init__` precedence:
    ///   base_url = extra["url"] or MATTERMOST_URL, rstrip("/")
    ///   token    = config.token or MATTERMOST_TOKEN
    ///   reply_mode = (extra["reply_mode"] or MATTERMOST_REPLY_MODE or "off").lower()
    pub fn resolve(
        extra: &HashMap<String, Value>,
        config_token: &str,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        let extra_str = |key: &str| -> String {
            match extra.get(key) {
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            }
        };

        let url_extra = extra_str("url");
        let base_url_raw = if !url_extra.is_empty() {
            url_extra
        } else {
            env("MATTERMOST_URL").unwrap_or_default()
        };
        let base_url = base_url_raw.trim_end_matches('/').to_string();

        let token = if !config_token.is_empty() {
            config_token.to_string()
        } else {
            env("MATTERMOST_TOKEN").unwrap_or_default()
        };

        let reply_extra = extra_str("reply_mode");
        let reply_mode_raw = if !reply_extra.is_empty() {
            reply_extra
        } else {
            env("MATTERMOST_REPLY_MODE").unwrap_or_else(|| "off".to_string())
        };
        let reply_mode = reply_mode_raw.to_lowercase();

        MattermostConfig {
            base_url,
            token,
            reply_mode,
        }
    }

    /// Whether thread replies should be nested. Mirrors the `reply_mode == "thread"`
    /// gate applied before adding `root_id` to a post payload.
    pub fn thread_mode(&self) -> bool {
        self.reply_mode == "thread"
    }
}

// ===========================================================================
// HTTP helpers (request construction + response parsing)
// ===========================================================================

/// Build the JSON request headers. Mirrors `_headers`.
pub fn headers(token: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".to_string(), format!("Bearer {token}")),
        ("Content-Type".to_string(), "application/json".to_string()),
    ]
}

/// Build a `/api/v4/{path}` URL. Mirrors the `_api_*` URL construction
/// (`f"{base}/api/v4/{path.lstrip('/')}"`).
pub fn api_url(base_url: &str, path: &str) -> String {
    format!("{base_url}/api/v4/{}", path.trim_start_matches('/'))
}

/// Build the file-download URL for a file id (`/api/v4/files/{fid}`).
pub fn file_download_url(base_url: &str, file_id: &str) -> String {
    format!("{base_url}/api/v4/files/{file_id}")
}

/// Build the file-upload URL (`/api/v4/files`).
pub fn file_upload_url(base_url: &str) -> String {
    format!("{base_url}/api/v4/files")
}

/// Outcome of an HTTP helper call. Mirrors the Python convention of returning an
/// empty dict on any error (status >= 400, network failure, or non-JSON body).
fn parse_response(resp: reqwest::blocking::Response, method: &str, path: &str) -> Value {
    let status = resp.status();
    if status.as_u16() >= 400 {
        let body = resp.text().unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        log::error!("MM API {method} {path} → {} : {snippet}", status.as_u16());
        return Value::Object(serde_json::Map::new());
    }
    match resp.json::<Value>() {
        Ok(v) => v,
        Err(_) => Value::Object(serde_json::Map::new()),
    }
}

fn apply_headers(
    mut builder: reqwest::blocking::RequestBuilder,
    hdrs: &[(String, String)],
) -> reqwest::blocking::RequestBuilder {
    for (k, v) in hdrs {
        builder = builder.header(k.as_str(), v.as_str());
    }
    builder
}

/// `GET /api/v4/{path}`. Returns the parsed JSON object, or an empty object on
/// any error. Mirrors `_api_get`.
pub fn api_get(client: &reqwest::blocking::Client, base_url: &str, token: &str, path: &str) -> Value {
    let url = api_url(base_url, path);
    let req = apply_headers(
        client
            .get(&url)
            .timeout(std::time::Duration::from_secs(30)),
        &headers(token),
    );
    match req.send() {
        Ok(resp) => parse_response(resp, "GET", path),
        Err(exc) => {
            log::error!("MM API GET {path} network error: {exc}");
            Value::Object(serde_json::Map::new())
        }
    }
}

/// `POST /api/v4/{path}` with a JSON body. Mirrors `_api_post`.
pub fn api_post(
    client: &reqwest::blocking::Client,
    base_url: &str,
    token: &str,
    path: &str,
    payload: &Value,
) -> Value {
    let url = api_url(base_url, path);
    let req = apply_headers(
        client
            .post(&url)
            .timeout(std::time::Duration::from_secs(30)),
        &headers(token),
    )
    .json(payload);
    match req.send() {
        Ok(resp) => parse_response(resp, "POST", path),
        Err(exc) => {
            log::error!("MM API POST {path} network error: {exc}");
            Value::Object(serde_json::Map::new())
        }
    }
}

/// `PUT /api/v4/{path}` with a JSON body. Mirrors `_api_put` (no explicit
/// timeout in the Python original).
pub fn api_put(
    client: &reqwest::blocking::Client,
    base_url: &str,
    token: &str,
    path: &str,
    payload: &Value,
) -> Value {
    let url = api_url(base_url, path);
    let req = apply_headers(client.put(&url), &headers(token)).json(payload);
    match req.send() {
        Ok(resp) => parse_response(resp, "PUT", path),
        Err(exc) => {
            log::error!("MM API PUT {path} network error: {exc}");
            Value::Object(serde_json::Map::new())
        }
    }
}

/// Upload a file and return its file ID, or None on failure. Mirrors
/// `_upload_file`: multipart form with `channel_id` + the `files` part.
pub fn upload_file(
    client: &reqwest::blocking::Client,
    base_url: &str,
    token: &str,
    channel_id: &str,
    file_data: Vec<u8>,
    filename: &str,
    content_type: &str,
) -> Option<String> {
    let url = file_upload_url(base_url);
    let part = reqwest::blocking::multipart::Part::bytes(file_data)
        .file_name(filename.to_string())
        .mime_str(content_type)
        .ok()?;
    let form = reqwest::blocking::multipart::Form::new()
        .text("channel_id", channel_id.to_string())
        .part("files", part);

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .timeout(std::time::Duration::from_secs(60))
        .multipart(form)
        .send()
        .ok()?;

    let status = resp.status();
    if status.as_u16() >= 400 {
        let body = resp.text().unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        log::error!("MM file upload → {} : {snippet}", status.as_u16());
        return None;
    }
    let data: Value = resp.json().ok()?;
    extract_uploaded_file_id(&data)
}

/// Extract the first uploaded file id from a `/files` upload response.
/// Mirrors `data.get("file_infos", [])` then `infos[0]["id"]`.
pub fn extract_uploaded_file_id(data: &Value) -> Option<String> {
    let infos = data.get("file_infos")?.as_array()?;
    let first = infos.first()?;
    first.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
}

// ===========================================================================
// Post payload construction
// ===========================================================================

/// Build a `posts` payload for a text message. Mirrors the `payload` dict in
/// `send`. `root_id` is added only when a thread reply is requested AND the
/// adapter is in thread reply mode.
pub fn build_post_payload(channel_id: &str, message: &str, root_id: Option<&str>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("channel_id".to_string(), json!(channel_id));
    obj.insert("message".to_string(), json!(message));
    if let Some(rid) = root_id {
        obj.insert("root_id".to_string(), json!(rid));
    }
    Value::Object(obj)
}

/// Build a `posts` payload that attaches one or more uploaded files. Mirrors
/// the file-post payload in `_send_url_as_file` / `_send_local_file` /
/// `send_multiple_images`.
pub fn build_file_post_payload(
    channel_id: &str,
    message: &str,
    file_ids: &[String],
    root_id: Option<&str>,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("channel_id".to_string(), json!(channel_id));
    obj.insert("message".to_string(), json!(message));
    obj.insert(
        "file_ids".to_string(),
        Value::Array(file_ids.iter().map(|f| json!(f)).collect()),
    );
    if let Some(rid) = root_id {
        obj.insert("root_id".to_string(), json!(rid));
    }
    Value::Object(obj)
}

/// Build the typing-indicator payload. Mirrors `send_typing`'s
/// `{"channel_id": chat_id}` body posted to `users/{bot}/typing`.
pub fn build_typing_payload(channel_id: &str) -> Value {
    json!({ "channel_id": channel_id })
}

/// Create a post and return a [`SendResult`]. Mirrors the `_api_post("posts",...)`
/// then `data["id"]` check shared by `send` and the file-send helpers.
///
/// `error_on_fail` is the error message used when the post fails (matching the
/// per-call-site strings: "Failed to create post" / "Failed to post with file").
pub fn create_post(
    client: &reqwest::blocking::Client,
    base_url: &str,
    token: &str,
    payload: &Value,
    error_on_fail: &str,
) -> SendResult {
    let data = api_post(client, base_url, token, "posts", payload);
    match post_id(&data) {
        Some(id) => SendResult::ok(Some(id)),
        None => SendResult::fail(error_on_fail.to_string()),
    }
}

/// Extract the `id` field from a created-post (or patched-post) response.
pub fn post_id(data: &Value) -> Option<String> {
    data.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
}

// ===========================================================================
// Message formatting
// ===========================================================================

/// Mattermost uses standard Markdown — mostly pass through. Strip image
/// markdown into plain links (files are uploaded separately). Mirrors
/// `format_message`: `![alt](url)` → `url`.
pub fn format_message(content: &str) -> String {
    let re = Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").unwrap();
    re.replace_all(content, "$2").into_owned()
}

/// Build the chat-info response for a channel. Mirrors `get_chat_info`:
/// name = display_name or name or chat_id; type via [`channel_type`].
pub fn parse_chat_info(data: &Value, chat_id: &str) -> (String, String) {
    if !data.is_object() || data.as_object().map(|m| m.is_empty()).unwrap_or(true) {
        return (chat_id.to_string(), "channel".to_string());
    }
    let code = data.get("type").and_then(|v| v.as_str()).unwrap_or("O");
    let ch_type = channel_type(code).to_string();
    let display_name = data
        .get("display_name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| data.get("name").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .unwrap_or(chat_id)
        .to_string();
    (display_name, ch_type)
}

// ===========================================================================
// WebSocket helpers
// ===========================================================================

/// Build the WebSocket URL: `https://` → `wss://`, `http://` → `ws://`, then
/// append `/api/v4/websocket`. Mirrors `re.sub(r"^http", "ws", base) + "/api/v4/websocket"`.
pub fn websocket_url(base_url: &str) -> String {
    // Python's `re.sub(r"^http", "ws", ...)` replaces only a leading "http"
    // prefix (so "https" → "wss", "http" → "ws"). Anything else is untouched.
    let rewritten = if let Some(rest) = base_url.strip_prefix("http") {
        format!("ws{rest}")
    } else {
        base_url.to_string()
    };
    format!("{rewritten}/api/v4/websocket")
}

/// Build the WebSocket authentication-challenge message. Mirrors `auth_msg`.
pub fn ws_auth_message(token: &str) -> Value {
    json!({
        "seq": 1,
        "action": "authentication_challenge",
        "data": { "token": token },
    })
}

/// Compute the next reconnect delay (exponential backoff, capped). Mirrors
/// `delay = min(delay * 2, _RECONNECT_MAX_DELAY)`.
pub fn next_reconnect_delay(delay: f64) -> f64 {
    (delay * 2.0).min(RECONNECT_MAX_DELAY)
}

/// Compute the sleep duration before reconnecting given a jitter draw in
/// `[0, 1)`. Mirrors `delay + delay * _RECONNECT_JITTER * random()`.
pub fn reconnect_delay_with_jitter(delay: f64, random_draw: f64) -> f64 {
    delay + delay * RECONNECT_JITTER * random_draw
}

/// Whether a WebSocket error string indicates a permanent auth/permission
/// failure (no point retrying). Mirrors the `"401"/"403"/"unauthorized"` check
/// in `_ws_loop`.
pub fn is_permanent_ws_error(error: &str) -> bool {
    let lowered = error.to_lowercase();
    lowered.contains("401") || lowered.contains("403") || lowered.contains("unauthorized")
}

// ===========================================================================
// Inbound `posted` event parsing
// ===========================================================================

/// Mention-gating configuration for non-DM channels. Mirrors the env reads in
/// `_handle_ws_event`.
#[derive(Debug, Clone)]
pub struct MentionConfig {
    /// `MATTERMOST_REQUIRE_MENTION` (default true; false/0/no disables).
    pub require_mention: bool,
    /// `MATTERMOST_FREE_RESPONSE_CHANNELS` parsed into a set of channel ids.
    pub free_channels: Vec<String>,
}

impl MentionConfig {
    /// Resolve from environment-style getters. Mirrors:
    ///   require_mention = MATTERMOST_REQUIRE_MENTION.lower() not in (false,0,no)
    ///   free_channels   = {c.strip() for c in CSV.split(",") if c.strip()}
    pub fn resolve(env: &dyn Fn(&str) -> Option<String>) -> Self {
        let raw = env("MATTERMOST_REQUIRE_MENTION").unwrap_or_else(|| "true".to_string());
        let lowered = raw.to_lowercase();
        let require_mention = !matches!(lowered.as_str(), "false" | "0" | "no");

        let free_raw = env("MATTERMOST_FREE_RESPONSE_CHANNELS").unwrap_or_default();
        let mut free_channels: Vec<String> = Vec::new();
        for part in free_raw.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() && !free_channels.contains(&trimmed.to_string()) {
                free_channels.push(trimmed.to_string());
            }
        }

        MentionConfig {
            require_mention,
            free_channels,
        }
    }
}

/// Result of attempting to gate / strip a non-DM channel message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MentionGate {
    /// The message must be skipped (no mention, mention required, not free).
    Skip,
    /// The message is allowed; the (possibly mention-stripped) text is returned.
    Allow(String),
}

/// Apply mention gating to a non-DM channel message. Mirrors the
/// `if channel_type_raw != "D":` block of `_handle_ws_event`.
///
/// When the bot is mentioned, the mention tokens (`@username`, `@user_id`) are
/// stripped from the text (case-insensitively) and the result is trimmed.
pub fn apply_mention_gate(
    cfg: &MentionConfig,
    channel_id: &str,
    bot_username: &str,
    bot_user_id: &str,
    message_text: &str,
) -> MentionGate {
    let is_free_channel = cfg.free_channels.iter().any(|c| c == channel_id);

    let mention_patterns = [format!("@{bot_username}"), format!("@{bot_user_id}")];
    let text_lower = message_text.to_lowercase();
    let has_mention = mention_patterns
        .iter()
        .any(|p| !p.is_empty() && p.len() > 1 && text_lower.contains(&p.to_lowercase()));

    if cfg.require_mention && !is_free_channel && !has_mention {
        return MentionGate::Skip;
    }

    if has_mention {
        let mut stripped = message_text.to_string();
        for pattern in &mention_patterns {
            if pattern.len() <= 1 {
                continue;
            }
            let re = Regex::new(&format!("(?i){}", regex::escape(pattern))).unwrap();
            stripped = re.replace_all(&stripped, "").into_owned();
        }
        return MentionGate::Allow(stripped.trim().to_string());
    }

    MentionGate::Allow(message_text.to_string())
}

/// A decision about how to handle a parsed `posted` event. Mirrors the control
/// flow of `_handle_ws_event` up to the point where media downloads begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostedDecision {
    /// The event should be ignored (wrong type, own/system message, malformed,
    /// duplicate, or mention-gated). The `&'static str` is the reason.
    Ignore(&'static str),
    /// The event yields a message to process.
    Process(Box<ParsedPost>),
}

/// The salient fields extracted from a `posted` event's inner post, after
/// gating/stripping. Mirrors the locals assembled in `_handle_ws_event`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPost {
    pub post_id: String,
    pub channel_id: String,
    pub chat_type: String,
    pub channel_type_raw: String,
    pub message_text: String,
    pub sender_id: String,
    pub sender_name: String,
    pub thread_id: Option<String>,
    pub file_ids: Vec<String>,
    /// Initial message type before media downloads adjust it.
    pub message_type: MessageType,
}

/// Parse and gate a raw `posted` WebSocket event. Mirrors `_handle_ws_event`
/// through the construction of the message event (excluding the actual media
/// download + `handle_message` dispatch, which are runtime side effects).
///
/// `is_duplicate` is invoked with the post id to honor the dedup cache.
pub fn parse_posted_event(
    event: &Value,
    bot_user_id: &str,
    bot_username: &str,
    mention_cfg: &MentionConfig,
    is_duplicate: &mut dyn FnMut(&str) -> bool,
) -> PostedDecision {
    if event.get("event").and_then(|v| v.as_str()) != Some("posted") {
        return PostedDecision::Ignore("not_posted");
    }

    let data = event.get("data").cloned().unwrap_or(Value::Null);
    let raw_post_str = match data.get("post").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s,
        _ => return PostedDecision::Ignore("no_post"),
    };

    let post: Value = match serde_json::from_str(raw_post_str) {
        Ok(v) => v,
        Err(_) => return PostedDecision::Ignore("bad_post_json"),
    };

    // Ignore own messages.
    if post.get("user_id").and_then(|v| v.as_str()) == Some(bot_user_id) && !bot_user_id.is_empty() {
        return PostedDecision::Ignore("own_message");
    }

    // Ignore system posts (`post.type` truthy).
    if let Some(t) = post.get("type").and_then(|v| v.as_str()) {
        if !t.is_empty() {
            return PostedDecision::Ignore("system_post");
        }
    }

    let post_id = post.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();

    // Dedup.
    if is_duplicate(&post_id) {
        return PostedDecision::Ignore("duplicate");
    }

    let channel_id = post
        .get("channel_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let channel_type_raw = data
        .get("channel_type")
        .and_then(|v| v.as_str())
        .unwrap_or("O")
        .to_string();
    let chat_type = channel_type(&channel_type_raw).to_string();

    let mut message_text = post
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Mention-gating for non-DM channels.
    if channel_type_raw != "D" {
        match apply_mention_gate(
            mention_cfg,
            &channel_id,
            bot_username,
            bot_user_id,
            &message_text,
        ) {
            MentionGate::Skip => return PostedDecision::Ignore("no_mention"),
            MentionGate::Allow(text) => message_text = text,
        }
    }

    let sender_id = post
        .get("user_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sender_name_raw = data
        .get("sender_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim_start_matches('@')
        .to_string();
    let sender_name = if sender_name_raw.is_empty() {
        sender_id.clone()
    } else {
        sender_name_raw
    };

    // Thread support: `root_id` if present and non-empty.
    let thread_id = post
        .get("root_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let file_ids: Vec<String> = post
        .get("file_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Determine initial message type (commands start with "/").
    let message_type = if message_text.starts_with('/') {
        MessageType::Command
    } else {
        MessageType::Text
    };

    PostedDecision::Process(Box::new(ParsedPost {
        post_id,
        channel_id,
        chat_type,
        channel_type_raw,
        message_text,
        sender_id,
        sender_name,
        thread_id,
        file_ids,
        message_type,
    }))
}

/// Adjust a TEXT message type based on downloaded media mime types. Mirrors the
/// `if media_types and msg_type == MessageType.TEXT:` block.
pub fn refine_message_type(current: MessageType, media_types: &[String]) -> MessageType {
    if media_types.is_empty() || current != MessageType::Text {
        return current;
    }
    if media_types.iter().any(|m| m.starts_with("image/")) {
        MessageType::Photo
    } else if media_types.iter().any(|m| m.starts_with("audio/")) {
        MessageType::Voice
    } else {
        MessageType::Document
    }
}

/// Classify the cache target for a downloaded file based on its mime type.
/// Mirrors the `if mime.startswith("image/")` / `audio/` / else branches in the
/// file-download loop of `_handle_ws_event`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
    Document,
}

/// Return `(MediaKind, default_extension)` for a file mime type + filename.
/// Image defaults to `.png`, audio to `.ogg`; documents preserve the filename.
pub fn classify_media(mime: &str) -> MediaKind {
    if mime.starts_with("image/") {
        MediaKind::Image
    } else if mime.starts_with("audio/") {
        MediaKind::Audio
    } else {
        MediaKind::Document
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn env_map(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn channel_type_map() {
        assert_eq!(channel_type("D"), "dm");
        assert_eq!(channel_type("G"), "group");
        assert_eq!(channel_type("P"), "group");
        assert_eq!(channel_type("O"), "channel");
        assert_eq!(channel_type("X"), "channel");
    }

    #[test]
    fn requirements_gating() {
        assert!(check_mattermost_requirements_from("tok", "https://mm", true));
        assert!(!check_mattermost_requirements_from("", "https://mm", true));
        assert!(!check_mattermost_requirements_from("tok", "", true));
        assert!(!check_mattermost_requirements_from("tok", "https://mm", false));
    }

    #[test]
    fn config_precedence_and_rstrip() {
        let mut extra = HashMap::new();
        extra.insert("url".to_string(), Value::String("https://mm.test/".to_string()));
        extra.insert("reply_mode".to_string(), Value::String("Thread".to_string()));
        let env = env_map(&[("MATTERMOST_TOKEN", "envtok")]);
        let cfg = MattermostConfig::resolve(&extra, "cfgtok", &env);
        assert_eq!(cfg.base_url, "https://mm.test"); // trailing slash stripped
        assert_eq!(cfg.token, "cfgtok"); // config.token wins
        assert_eq!(cfg.reply_mode, "thread"); // lowercased
        assert!(cfg.thread_mode());
    }

    #[test]
    fn config_env_fallbacks() {
        let extra = HashMap::new();
        let env = env_map(&[
            ("MATTERMOST_URL", "https://env.mm/"),
            ("MATTERMOST_TOKEN", "envtok"),
        ]);
        let cfg = MattermostConfig::resolve(&extra, "", &env);
        assert_eq!(cfg.base_url, "https://env.mm");
        assert_eq!(cfg.token, "envtok");
        assert_eq!(cfg.reply_mode, "off"); // default
        assert!(!cfg.thread_mode());
    }

    #[test]
    fn header_construction() {
        let h = headers("abc");
        assert_eq!(h[0], ("Authorization".into(), "Bearer abc".into()));
        assert_eq!(h[1], ("Content-Type".into(), "application/json".into()));
    }

    #[test]
    fn api_url_strips_leading_slash() {
        assert_eq!(api_url("https://mm", "users/me"), "https://mm/api/v4/users/me");
        assert_eq!(api_url("https://mm", "/users/me"), "https://mm/api/v4/users/me");
        assert_eq!(file_download_url("https://mm", "F1"), "https://mm/api/v4/files/F1");
        assert_eq!(file_upload_url("https://mm"), "https://mm/api/v4/files");
    }

    #[test]
    fn format_message_strips_image_markdown() {
        assert_eq!(
            format_message("see ![alt](https://x.com/a.png) here"),
            "see https://x.com/a.png here"
        );
        assert_eq!(format_message("plain text"), "plain text");
    }

    #[test]
    fn parse_chat_info_variants() {
        let data = json!({"type": "P", "display_name": "Secret", "name": "secret"});
        assert_eq!(parse_chat_info(&data, "C1"), ("Secret".to_string(), "group".to_string()));
        // Empty display_name → fall back to name.
        let data2 = json!({"type": "O", "display_name": "", "name": "general"});
        assert_eq!(parse_chat_info(&data2, "C1"), ("general".to_string(), "channel".to_string()));
        // Empty object → defaults.
        assert_eq!(parse_chat_info(&json!({}), "C9"), ("C9".to_string(), "channel".to_string()));
    }

    #[test]
    fn websocket_url_scheme_rewrite() {
        assert_eq!(websocket_url("https://mm.test"), "wss://mm.test/api/v4/websocket");
        assert_eq!(websocket_url("http://mm.test"), "ws://mm.test/api/v4/websocket");
        // No leading http: untouched prefix.
        assert_eq!(websocket_url("ws://mm.test"), "ws://mm.test/api/v4/websocket");
    }

    #[test]
    fn ws_auth_message_shape() {
        let m = ws_auth_message("tok");
        assert_eq!(m["seq"], 1);
        assert_eq!(m["action"], "authentication_challenge");
        assert_eq!(m["data"]["token"], "tok");
    }

    #[test]
    fn backoff_and_jitter() {
        assert_eq!(next_reconnect_delay(2.0), 4.0);
        assert_eq!(next_reconnect_delay(40.0), RECONNECT_MAX_DELAY);
        // jitter with draw 0 → no added jitter; draw 1 → max jitter.
        assert_eq!(reconnect_delay_with_jitter(2.0, 0.0), 2.0);
        assert!((reconnect_delay_with_jitter(2.0, 1.0) - (2.0 + 2.0 * 0.2)).abs() < 1e-9);
    }

    #[test]
    fn permanent_ws_error_detection() {
        assert!(is_permanent_ws_error("HTTP 401 Unauthorized"));
        assert!(is_permanent_ws_error("got 403 forbidden"));
        assert!(is_permanent_ws_error("Unauthorized"));
        assert!(!is_permanent_ws_error("connection reset"));
    }

    #[test]
    fn build_payloads() {
        let p = build_post_payload("C1", "hi", None);
        assert_eq!(p["channel_id"], "C1");
        assert_eq!(p["message"], "hi");
        assert!(p.get("root_id").is_none());

        let p2 = build_post_payload("C1", "hi", Some("R1"));
        assert_eq!(p2["root_id"], "R1");

        let fp = build_file_post_payload("C1", "cap", &["F1".into(), "F2".into()], Some("R2"));
        assert_eq!(fp["file_ids"], json!(["F1", "F2"]));
        assert_eq!(fp["root_id"], "R2");

        assert_eq!(build_typing_payload("C1"), json!({"channel_id": "C1"}));
    }

    #[test]
    fn uploaded_file_id_extraction() {
        let data = json!({"file_infos": [{"id": "FID1"}, {"id": "FID2"}]});
        assert_eq!(extract_uploaded_file_id(&data).as_deref(), Some("FID1"));
        assert_eq!(extract_uploaded_file_id(&json!({"file_infos": []})), None);
        assert_eq!(extract_uploaded_file_id(&json!({})), None);
    }

    #[test]
    fn post_id_extraction() {
        assert_eq!(post_id(&json!({"id": "P1"})).as_deref(), Some("P1"));
        assert_eq!(post_id(&json!({})), None);
    }

    #[test]
    fn mention_config_resolution() {
        let env = env_map(&[
            ("MATTERMOST_REQUIRE_MENTION", "false"),
            ("MATTERMOST_FREE_RESPONSE_CHANNELS", "C1, C2 ,,C1"),
        ]);
        let cfg = MentionConfig::resolve(&env);
        assert!(!cfg.require_mention);
        assert_eq!(cfg.free_channels, vec!["C1".to_string(), "C2".to_string()]);

        // Default require_mention is true.
        let empty = env_map(&[]);
        let cfg2 = MentionConfig::resolve(&empty);
        assert!(cfg2.require_mention);
        assert!(cfg2.free_channels.is_empty());
    }

    #[test]
    fn mention_gate_skip_when_required() {
        let cfg = MentionConfig {
            require_mention: true,
            free_channels: vec![],
        };
        let gate = apply_mention_gate(&cfg, "C1", "bot", "UBOT", "hello there");
        assert_eq!(gate, MentionGate::Skip);
    }

    #[test]
    fn mention_gate_strips_mention() {
        let cfg = MentionConfig {
            require_mention: true,
            free_channels: vec![],
        };
        let gate = apply_mention_gate(&cfg, "C1", "bot", "UBOT", "hey @bot do this");
        assert_eq!(gate, MentionGate::Allow("hey  do this".to_string()));
    }

    #[test]
    fn mention_gate_free_channel_passes_unstripped() {
        let cfg = MentionConfig {
            require_mention: true,
            free_channels: vec!["C1".to_string()],
        };
        let gate = apply_mention_gate(&cfg, "C1", "bot", "UBOT", "no mention here");
        assert_eq!(gate, MentionGate::Allow("no mention here".to_string()));
    }

    #[test]
    fn parse_posted_dm_message() {
        let post = json!({
            "id": "P1",
            "user_id": "U1",
            "channel_id": "DM1",
            "message": "hi bot",
            "file_ids": [],
        });
        let event = json!({
            "event": "posted",
            "data": {
                "post": post.to_string(),
                "channel_type": "D",
                "sender_name": "@alice",
            },
        });
        let cfg = MentionConfig { require_mention: true, free_channels: vec![] };
        let mut dup = |_: &str| false;
        let decision = parse_posted_event(&event, "UBOT", "bot", &cfg, &mut dup);
        match decision {
            PostedDecision::Process(p) => {
                assert_eq!(p.post_id, "P1");
                assert_eq!(p.channel_id, "DM1");
                assert_eq!(p.chat_type, "dm");
                assert_eq!(p.message_text, "hi bot");
                assert_eq!(p.sender_id, "U1");
                assert_eq!(p.sender_name, "alice");
                assert_eq!(p.message_type, MessageType::Text);
                assert!(p.thread_id.is_none());
            }
            other => panic!("expected Process, got {other:?}"),
        }
    }

    #[test]
    fn parse_posted_ignores_own_and_system_and_dup() {
        let cfg = MentionConfig { require_mention: false, free_channels: vec![] };

        // own message
        let own = json!({"event": "posted", "data": {"post": json!({"id":"P","user_id":"UBOT","channel_id":"D","message":"x"}).to_string(), "channel_type":"D"}});
        let mut dup = |_: &str| false;
        assert_eq!(parse_posted_event(&own, "UBOT", "bot", &cfg, &mut dup), PostedDecision::Ignore("own_message"));

        // system post (type set)
        let sys = json!({"event": "posted", "data": {"post": json!({"id":"P","user_id":"U1","type":"system_join_channel","channel_id":"D"}).to_string(), "channel_type":"D"}});
        assert_eq!(parse_posted_event(&sys, "UBOT", "bot", &cfg, &mut dup), PostedDecision::Ignore("system_post"));

        // wrong event
        let other = json!({"event": "typing"});
        assert_eq!(parse_posted_event(&other, "UBOT", "bot", &cfg, &mut dup), PostedDecision::Ignore("not_posted"));

        // duplicate
        let msg = json!({"event": "posted", "data": {"post": json!({"id":"P","user_id":"U1","channel_id":"D","message":"x"}).to_string(), "channel_type":"D"}});
        let mut always_dup = |_: &str| true;
        assert_eq!(parse_posted_event(&msg, "UBOT", "bot", &cfg, &mut always_dup), PostedDecision::Ignore("duplicate"));
    }

    #[test]
    fn parse_posted_channel_mention_gating() {
        let cfg = MentionConfig { require_mention: true, free_channels: vec![] };
        let mut dup = |_: &str| false;

        // No mention in channel → skip.
        let no_mention = json!({"event": "posted", "data": {"post": json!({"id":"P","user_id":"U1","channel_id":"C1","message":"hello"}).to_string(), "channel_type":"O"}});
        assert_eq!(parse_posted_event(&no_mention, "UBOT", "bot", &cfg, &mut dup), PostedDecision::Ignore("no_mention"));

        // Mention present → processed and stripped.
        let mention = json!({"event": "posted", "data": {"post": json!({"id":"P2","user_id":"U1","channel_id":"C1","message":"@bot hi"}).to_string(), "channel_type":"O"}});
        match parse_posted_event(&mention, "UBOT", "bot", &cfg, &mut dup) {
            PostedDecision::Process(p) => {
                assert_eq!(p.message_text, "hi");
                assert_eq!(p.chat_type, "channel");
            }
            other => panic!("expected Process, got {other:?}"),
        }
    }

    #[test]
    fn parse_posted_command_and_thread() {
        let cfg = MentionConfig { require_mention: false, free_channels: vec![] };
        let mut dup = |_: &str| false;
        let event = json!({"event": "posted", "data": {"post": json!({"id":"P","user_id":"U1","channel_id":"D","message":"/help","root_id":"ROOT1","file_ids":["F1"]}).to_string(), "channel_type":"D"}});
        match parse_posted_event(&event, "UBOT", "bot", &cfg, &mut dup) {
            PostedDecision::Process(p) => {
                assert_eq!(p.message_type, MessageType::Command);
                assert_eq!(p.thread_id.as_deref(), Some("ROOT1"));
                assert_eq!(p.file_ids, vec!["F1".to_string()]);
            }
            other => panic!("expected Process, got {other:?}"),
        }
    }

    #[test]
    fn refine_type_from_media() {
        assert_eq!(refine_message_type(MessageType::Text, &["image/png".into()]), MessageType::Photo);
        assert_eq!(refine_message_type(MessageType::Text, &["audio/ogg".into()]), MessageType::Voice);
        assert_eq!(refine_message_type(MessageType::Text, &["application/pdf".into()]), MessageType::Document);
        // Existing non-text type preserved.
        assert_eq!(refine_message_type(MessageType::Command, &["image/png".into()]), MessageType::Command);
        // No media → unchanged.
        assert_eq!(refine_message_type(MessageType::Text, &[]), MessageType::Text);
    }

    #[test]
    fn media_classification() {
        assert_eq!(classify_media("image/png"), MediaKind::Image);
        assert_eq!(classify_media("audio/ogg"), MediaKind::Audio);
        assert_eq!(classify_media("application/pdf"), MediaKind::Document);
    }
}
