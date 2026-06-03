//! WhatsApp platform adapter, ported from `gateway/platforms/whatsapp.py`.
//!
//! WhatsApp has no official personal-account bot API, so Hermes drives a
//! Node.js *bridge* subprocess that runs the actual WhatsApp Web client and
//! exposes a small HTTP API (`/health`, `/messages`, `/send`, `/edit`,
//! `/send-media`, `/typing`, `/chat/{id}`). This adapter manages that bridge
//! process and translates between bridge JSON and Hermes [`MessageEvent`]s.
//!
//! This Rust port reproduces the *behavior-defining* logic faithfully:
//!   - DM / group access policies (`dm_policy`, `group_policy`, allow-lists)
//!   - mention / reply / free-response gating for group messages
//!   - WhatsApp-ID canonicalisation and bot-mention detection / cleaning
//!   - markdown -> WhatsApp markup conversion (`format_message`)
//!   - bridge HTTP request construction + response parsing (via
//!     `reqwest::blocking`), keeping the exact JSON shapes the Node bridge
//!     expects
//!   - the `_build_message_event` media-caching / text-injection pipeline
//!   - process-tree termination + port-killing helpers
//!
//! The async `asyncio` orchestration (`connect`, `_poll_messages`,
//! `disconnect`) is intimately tied to the CPython event loop. It is modelled
//! here as plain synchronous helpers + a [`WhatsAppAdapter`] state struct that
//! mirrors the Python configuration and lifecycle flags without dragging in a
//! runtime.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageEvent`, `MessageType`,
//!     `SessionSource`, `SendResult`, `truncate_message`,
//!     `supported_document_types`
//!   - [`crate::mod_hermes_constants::get_hermes_dir`]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::{json, Value};

use crate::gw_platforms_base::{
    supported_document_types, truncate_message, MessageEvent, MessageType, SendResult,
    SessionSource,
};

/// Practical UX limit for a WhatsApp message (protocol allows ~65K).
pub const MAX_MESSAGE_LENGTH: usize = 4096;

/// Default bridge HTTP port.
pub const DEFAULT_BRIDGE_PORT: u16 = 3000;

#[cfg(target_os = "windows")]
const IS_WINDOWS: bool = true;
#[cfg(not(target_os = "windows"))]
const IS_WINDOWS: bool = false;

// ===========================================================================
// Process / port management helpers
// ===========================================================================

/// Kill any process listening on the given TCP port.
///
/// Mirrors `_kill_port_process`. All errors are swallowed (best-effort), just
/// like the Python `try/except Exception: pass`.
pub fn kill_port_process(port: u16) {
    let _ = std::panic::catch_unwind(|| {
        if IS_WINDOWS {
            kill_port_process_windows(port);
        } else {
            kill_port_process_unix(port);
        }
    });
}

#[cfg(target_os = "windows")]
fn kill_port_process_windows(port: u16) {
    use std::process::Command;
    let out = Command::new("netstat").args(["-ano", "-p", "TCP"]).output();
    let Ok(out) = out else { return };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let suffix = format!(":{port}");
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 5 && parts[3] == "LISTENING" {
            let local_addr = parts[1];
            if local_addr.ends_with(&suffix) {
                let _ = Command::new("taskkill")
                    .args(["/PID", parts[4], "/F"])
                    .output();
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn kill_port_process_windows(_port: u16) {}

#[cfg(not(target_os = "windows"))]
fn kill_port_process_unix(port: u16) {
    use std::process::Command;
    let spec = format!("{port}/tcp");
    let probe = Command::new("fuser").arg(&spec).output();
    if let Ok(probe) = probe {
        if probe.status.success() {
            let _ = Command::new("fuser").args(["-k", &spec]).output();
        }
    }
}

#[cfg(target_os = "windows")]
fn kill_port_process_unix(_port: u16) {}

// ===========================================================================
// Allow-list / policy parsing
// ===========================================================================

/// Allow-list value coming from config: either a YAML/JSON list, or a
/// comma-separated string.
#[derive(Debug, Clone)]
pub enum AllowListSpec {
    List(Vec<String>),
    Csv(String),
    None,
}

/// Parse `allow_from` / `group_allow_from` from config or env var.
///
/// Mirrors `_coerce_allow_list`: a list is split element-wise (stringified +
/// trimmed, empties dropped); a string is split on commas (trimmed, empties
/// dropped); `None` yields an empty set.
pub fn coerce_allow_list(raw: &AllowListSpec) -> HashSet<String> {
    match raw {
        AllowListSpec::None => HashSet::new(),
        AllowListSpec::List(items) => items
            .iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        AllowListSpec::Csv(s) => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
    }
}

/// DM-handling policy. Mirrors the `dm_policy` string values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmPolicy {
    Open,
    Allowlist,
    Disabled,
}

impl DmPolicy {
    /// Parse a raw policy string (already `.strip().lower()` in Python). Any
    /// unrecognised value falls through to `Open`, matching the Python logic
    /// where only the explicit `"disabled"` / `"allowlist"` branches gate.
    pub fn parse(raw: &str) -> DmPolicy {
        match raw.trim().to_lowercase().as_str() {
            "disabled" => DmPolicy::Disabled,
            "allowlist" => DmPolicy::Allowlist,
            _ => DmPolicy::Open,
        }
    }
}

/// Group-handling policy. Mirrors the `group_policy` string values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupPolicy {
    Open,
    Allowlist,
    Disabled,
}

impl GroupPolicy {
    pub fn parse(raw: &str) -> GroupPolicy {
        match raw.trim().to_lowercase().as_str() {
            "disabled" => GroupPolicy::Disabled,
            "allowlist" => GroupPolicy::Allowlist,
            _ => GroupPolicy::Open,
        }
    }
}

/// Check whether a DM from the given sender should be processed.
///
/// Mirrors `_is_dm_allowed`.
pub fn is_dm_allowed(policy: DmPolicy, allow_from: &HashSet<String>, sender_id: &str) -> bool {
    match policy {
        DmPolicy::Disabled => false,
        DmPolicy::Allowlist => allow_from.contains(sender_id),
        DmPolicy::Open => true,
    }
}

/// Check whether a group chat should be processed.
///
/// Mirrors `_is_group_allowed`.
pub fn is_group_allowed(
    policy: GroupPolicy,
    group_allow_from: &HashSet<String>,
    chat_id: &str,
) -> bool {
    match policy {
        GroupPolicy::Disabled => false,
        GroupPolicy::Allowlist => group_allow_from.contains(chat_id),
        GroupPolicy::Open => true,
    }
}

// ===========================================================================
// Env-driven group response settings
// ===========================================================================

/// Truthy env-style values. Mirrors `... in ("true", "1", "yes", "on")`.
fn is_truthy(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

/// Resolve whether a mention is required for group messages.
///
/// Mirrors `_whatsapp_require_mention`. `configured` is the raw
/// `config.extra["require_mention"]` value:
///   - `Some(BoolOrStr)` overrides the env var
///   - a string is truthy iff it matches the env-truthy set
///   - a bool is used directly
/// When unset, falls back to `WHATSAPP_REQUIRE_MENTION` (default `"false"`).
pub fn whatsapp_require_mention(configured: Option<&ConfigBool>) -> bool {
    match configured {
        Some(ConfigBool::Str(s)) => is_truthy(s),
        Some(ConfigBool::Bool(b)) => *b,
        None => {
            let raw = std::env::var("WHATSAPP_REQUIRE_MENTION").unwrap_or_else(|_| "false".into());
            is_truthy(&raw)
        }
    }
}

/// A config value that may be either a native bool or a string in YAML.
#[derive(Debug, Clone)]
pub enum ConfigBool {
    Bool(bool),
    Str(String),
}

/// Resolve the set of "free response" chat IDs for groups.
///
/// Mirrors `_whatsapp_free_response_chats`. `raw` is the
/// `config.extra["free_response_chats"]` value; when `None`, falls back to
/// `WHATSAPP_FREE_RESPONSE_CHATS` (comma-separated, default empty).
pub fn whatsapp_free_response_chats(raw: Option<&AllowListSpec>) -> HashSet<String> {
    match raw {
        Some(AllowListSpec::List(items)) => items
            .iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        Some(AllowListSpec::Csv(s)) => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        Some(AllowListSpec::None) | None => {
            let env = std::env::var("WHATSAPP_FREE_RESPONSE_CHATS").unwrap_or_default();
            env.split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect()
        }
    }
}

// ===========================================================================
// Mention-pattern compilation
// ===========================================================================

/// Compile WhatsApp mention patterns from config / env.
///
/// Mirrors `_compile_mention_patterns`. `configured` is the raw
/// `config.extra["mention_patterns"]` value. When `None`, the
/// `WHATSAPP_MENTION_PATTERNS` env var is consulted: it is first parsed as
/// JSON; on failure it is split on newlines (then commas) into a list.
/// A single string is wrapped in a one-element list. Invalid regexes are
/// dropped. Patterns are compiled case-insensitively.
pub fn compile_mention_patterns(configured: Option<&MentionPatternSpec>) -> Vec<Regex> {
    let raw_patterns: Option<Vec<String>> = match configured {
        Some(MentionPatternSpec::List(items)) => Some(items.clone()),
        Some(MentionPatternSpec::Str(s)) => Some(vec![s.clone()]),
        Some(MentionPatternSpec::None) | None => {
            let env = std::env::var("WHATSAPP_MENTION_PATTERNS").unwrap_or_default();
            let raw = env.trim();
            if raw.is_empty() {
                None
            } else {
                // Try JSON first.
                let parsed: Option<Vec<String>> = match serde_json::from_str::<Value>(raw) {
                    Ok(Value::Array(arr)) => Some(
                        arr.into_iter()
                            .map(|v| match v {
                                Value::String(s) => s,
                                other => other.to_string(),
                            })
                            .collect(),
                    ),
                    Ok(Value::String(s)) => Some(vec![s]),
                    _ => None,
                };
                match parsed {
                    Some(p) => Some(p),
                    None => {
                        let mut lines: Vec<String> = raw
                            .lines()
                            .map(|p| p.trim().to_string())
                            .filter(|p| !p.is_empty())
                            .collect();
                        if lines.is_empty() {
                            lines = raw
                                .split(',')
                                .map(|p| p.trim().to_string())
                                .filter(|p| !p.is_empty())
                                .collect();
                        }
                        Some(lines)
                    }
                }
            }
        }
    };

    let patterns = match raw_patterns {
        None => return Vec::new(),
        Some(p) => p,
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

/// Config shape of `mention_patterns`.
#[derive(Debug, Clone)]
pub enum MentionPatternSpec {
    List(Vec<String>),
    Str(String),
    None,
}

// ===========================================================================
// WhatsApp-ID normalization & bot-mention detection
// ===========================================================================

/// Canonicalise a WhatsApp ID.
///
/// Mirrors `_normalize_whatsapp_id`: strips whitespace, and if the value
/// contains both `:` and `@`, replaces the *first* `:` with `@` (device-suffix
/// normalisation, e.g. `12345:6@s.whatsapp.net` -> `12345@6@s.whatsapp.net`).
pub fn normalize_whatsapp_id(value: Option<&str>) -> String {
    let value = match value {
        Some(v) if !v.is_empty() => v,
        _ => return String::new(),
    };
    let normalized = value.trim();
    if normalized.contains(':') && normalized.contains('@') {
        // Replace only the first ':'.
        return normalized.replacen(':', "@", 1);
    }
    normalized.to_string()
}

/// Extract the set of normalised bot IDs from a bridge message.
///
/// Mirrors `_bot_ids_from_message` — reads `data["botIds"]`.
pub fn bot_ids_from_message(data: &Value) -> HashSet<String> {
    let mut bot_ids = HashSet::new();
    if let Some(arr) = data.get("botIds").and_then(|v| v.as_array()) {
        for candidate in arr {
            if let Some(s) = candidate.as_str() {
                let normalized = normalize_whatsapp_id(Some(s));
                if !normalized.is_empty() {
                    bot_ids.insert(normalized);
                }
            }
        }
    }
    bot_ids
}

/// Whether a message is a reply directed at the bot.
///
/// Mirrors `_message_is_reply_to_bot`.
pub fn message_is_reply_to_bot(data: &Value) -> bool {
    let quoted = normalize_whatsapp_id(data.get("quotedParticipant").and_then(|v| v.as_str()));
    if quoted.is_empty() {
        return false;
    }
    bot_ids_from_message(data).contains(&quoted)
}

/// Whether a message mentions the bot (via `mentionedIds` or `@bareid` in body).
///
/// Mirrors `_message_mentions_bot`.
pub fn message_mentions_bot(data: &Value) -> bool {
    let bot_ids = bot_ids_from_message(data);
    if bot_ids.is_empty() {
        return false;
    }
    let mentioned_ids: HashSet<String> = data
        .get("mentionedIds")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.as_str())
                .map(|c| normalize_whatsapp_id(Some(c)))
                .filter(|nid| !nid.is_empty())
                .collect()
        })
        .unwrap_or_default();

    if !mentioned_ids.is_disjoint(&bot_ids) {
        return true;
    }

    let body = data.get("body").and_then(|v| v.as_str()).unwrap_or("");
    let lower_body = body.to_lowercase();
    for bot_id in &bot_ids {
        let bare_id = bot_id.split('@').next().unwrap_or("").to_lowercase();
        if !bare_id.is_empty()
            && (lower_body.contains(&format!("@{bare_id}")) || lower_body.contains(&bare_id))
        {
            return true;
        }
    }
    false
}

/// Whether the body matches any configured mention regex.
///
/// Mirrors `_message_matches_mention_patterns`.
pub fn message_matches_mention_patterns(data: &Value, patterns: &[Regex]) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let body = data.get("body").and_then(|v| v.as_str()).unwrap_or("");
    patterns.iter().any(|p| p.is_match(body))
}

/// Strip leading bot @-mentions from message text.
///
/// Mirrors `_clean_bot_mention_text`: for each bot ID, removes
/// `@<bareid>` plus any trailing `,:-` and whitespace. Returns the original
/// text if cleaning produces an empty string.
pub fn clean_bot_mention_text(text: &str, data: &Value) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let bot_ids = bot_ids_from_message(data);
    let mut cleaned = text.to_string();
    for bot_id in &bot_ids {
        let bare_id = bot_id.split('@').next().unwrap_or("");
        if !bare_id.is_empty() {
            let pat = format!(r"@{}\b[,:\-]*\s*", regex::escape(bare_id));
            if let Ok(re) = Regex::new(&pat) {
                cleaned = re.replace_all(&cleaned, "").into_owned();
            }
        }
    }
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        text.to_string()
    } else {
        trimmed.to_string()
    }
}

// ===========================================================================
// Message gating
// ===========================================================================

/// Configuration relevant to `_should_process_message`, grouped for reuse.
#[derive(Debug, Clone)]
pub struct ProcessGate<'a> {
    pub dm_policy: DmPolicy,
    pub allow_from: &'a HashSet<String>,
    pub group_policy: GroupPolicy,
    pub group_allow_from: &'a HashSet<String>,
    pub free_response_chats: &'a HashSet<String>,
    pub require_mention: bool,
    pub mention_patterns: &'a [Regex],
}

/// Decide whether an incoming bridge message should be processed.
///
/// Mirrors `_should_process_message` exactly, including the early-return order:
/// group allow-list -> DM allow-list (DMs that pass are always processed) ->
/// free-response chats -> require-mention bypass -> `/` command bypass ->
/// reply-to-bot -> mentions-bot -> mention-pattern match.
pub fn should_process_message(data: &Value, gate: &ProcessGate) -> bool {
    let is_group = data
        .get("isGroup")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_group {
        let chat_id = data.get("chatId").and_then(|v| v.as_str()).unwrap_or("");
        if !is_group_allowed(gate.group_policy, gate.group_allow_from, chat_id) {
            return false;
        }
    } else {
        let sender_id = data
            .get("senderId")
            .and_then(|v| v.as_str())
            .or_else(|| data.get("from").and_then(|v| v.as_str()))
            .unwrap_or("");
        if !is_dm_allowed(gate.dm_policy, gate.allow_from, sender_id) {
            return false;
        }
        // DMs that pass the policy gate are always processed.
        return true;
    }

    // Group messages: check mention / free-response settings.
    let chat_id = data.get("chatId").and_then(|v| v.as_str()).unwrap_or("");
    if gate.free_response_chats.contains(chat_id) {
        return true;
    }
    if !gate.require_mention {
        return true;
    }
    let body = data
        .get("body")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if body.starts_with('/') {
        return true;
    }
    if message_is_reply_to_bot(data) {
        return true;
    }
    if message_mentions_bot(data) {
        return true;
    }
    message_matches_mention_patterns(data, gate.mention_patterns)
}

// ===========================================================================
// Markdown -> WhatsApp formatting
// ===========================================================================

/// Convert standard markdown to WhatsApp-compatible formatting.
///
/// Mirrors `format_message`. Fenced code blocks and inline code are protected
/// via NUL-delimited placeholders before conversion, then restored:
///   - `**x**` / `__x__` -> `*x*` (bold)
///   - `~~x~~` -> `~x~` (strikethrough)
///   - `# Header` -> `*Header*` (per-line)
///   - `[text](url)` -> `text (url)`
pub fn format_message(content: &str) -> String {
    if content.is_empty() {
        return content.to_string();
    }

    // --- 1. Protect fenced code blocks ---
    let fence_ph = "\u{0}FENCE";
    let mut fences: Vec<String> = Vec::new();
    let fence_re = Regex::new(r"(?s)```.*?```").unwrap();
    let mut result = fence_re
        .replace_all(content, |caps: &regex::Captures| {
            fences.push(caps[0].to_string());
            format!("{fence_ph}{}\u{0}", fences.len() - 1)
        })
        .into_owned();

    // --- 2. Protect inline code ---
    let code_ph = "\u{0}CODE";
    let mut codes: Vec<String> = Vec::new();
    let code_re = Regex::new(r"`[^`\n]+`").unwrap();
    result = code_re
        .replace_all(&result, |caps: &regex::Captures| {
            codes.push(caps[0].to_string());
            format!("{code_ph}{}\u{0}", codes.len() - 1)
        })
        .into_owned();

    // --- 3. Convert markdown formatting to WhatsApp syntax ---
    // Bold: **text** or __text__ -> *text*
    let bold_star = Regex::new(r"\*\*(.+?)\*\*").unwrap();
    result = bold_star.replace_all(&result, "*$1*").into_owned();
    let bold_under = Regex::new(r"__(.+?)__").unwrap();
    result = bold_under.replace_all(&result, "*$1*").into_owned();
    // Strikethrough: ~~text~~ -> ~text~
    let strike = Regex::new(r"~~(.+?)~~").unwrap();
    result = strike.replace_all(&result, "~$1~").into_owned();

    // --- 4. Headers: # Header -> *Header* (multiline) ---
    let header = Regex::new(r"(?m)^#{1,6}\s+(.+)$").unwrap();
    result = header.replace_all(&result, "*$1*").into_owned();

    // --- 5. Links: [text](url) -> text (url) ---
    let link = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
    result = link.replace_all(&result, "$1 ($2)").into_owned();

    // --- 6. Restore protected sections ---
    for (i, fence) in fences.iter().enumerate() {
        result = result.replace(&format!("{fence_ph}{i}\u{0}"), fence);
    }
    for (i, code) in codes.iter().enumerate() {
        result = result.replace(&format!("{code_ph}{i}\u{0}"), code);
    }

    result
}

// ===========================================================================
// Bridge HTTP request construction & response parsing
// ===========================================================================

/// Base URL for the local bridge HTTP server.
pub fn bridge_base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// Parsed `/health` response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeHealth {
    pub status: String,
}

/// Parse a `/health` JSON body. Mirrors `data.get("status", "unknown")`.
pub fn parse_health(body: &str) -> BridgeHealth {
    let status = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());
    BridgeHealth { status }
}

/// Build the JSON payload for a `/send` request.
///
/// Mirrors the per-chunk payload in `send`: always `chatId` + `message`, plus
/// `replyTo` when a reply target is set on the first chunk (`include_reply_to`).
pub fn build_send_payload(
    chat_id: &str,
    message: &str,
    reply_to: Option<&str>,
    include_reply_to: bool,
) -> Value {
    let mut payload = json!({ "chatId": chat_id, "message": message });
    if include_reply_to {
        if let Some(rt) = reply_to {
            payload["replyTo"] = json!(rt);
        }
    }
    payload
}

/// Build the JSON payload for an `/edit` request. Mirrors `edit_message`.
pub fn build_edit_payload(chat_id: &str, message_id: &str, message: &str) -> Value {
    json!({ "chatId": chat_id, "messageId": message_id, "message": message })
}

/// Build the JSON payload for a `/send-media` request.
///
/// Mirrors `_send_media_to_bridge`: always `chatId` + `filePath` + `mediaType`,
/// plus optional `caption` (when non-empty) and `fileName`.
pub fn build_send_media_payload(
    chat_id: &str,
    file_path: &str,
    media_type: &str,
    caption: Option<&str>,
    file_name: Option<&str>,
) -> Value {
    let mut payload = json!({
        "chatId": chat_id,
        "filePath": file_path,
        "mediaType": media_type,
    });
    // Python: `if caption:` — truthy means non-empty string.
    if let Some(c) = caption {
        if !c.is_empty() {
            payload["caption"] = json!(c);
        }
    }
    if let Some(f) = file_name {
        payload["fileName"] = json!(f);
    }
    payload
}

/// Build the JSON payload for a `/typing` request. Mirrors `send_typing`.
pub fn build_typing_payload(chat_id: &str) -> Value {
    json!({ "chatId": chat_id })
}

/// Parse the `messageId` field from a bridge send/edit/media response body.
pub fn parse_message_id(body: &str) -> Option<String> {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("messageId").and_then(|m| m.as_str()).map(str::to_string))
}

/// Chat-info result returned by [`parse_chat_info`]. Mirrors `get_chat_info`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatInfo {
    pub name: String,
    /// `"group"` or `"dm"`.
    pub chat_type: String,
    pub participants: Vec<String>,
}

/// Parse a `/chat/{id}` response body. On failure / non-200, callers should
/// fall back to `{name: chat_id, type: "dm"}` — see `get_chat_info`.
pub fn parse_chat_info(body: &str, chat_id: &str) -> ChatInfo {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| chat_id.to_string());
    let is_group = v.get("isGroup").and_then(|g| g.as_bool()).unwrap_or(false);
    let participants = v
        .get("participants")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    ChatInfo {
        name,
        chat_type: if is_group { "group" } else { "dm" }.to_string(),
        participants,
    }
}

/// Default chat-info fallback used when the bridge is unavailable.
pub fn default_chat_info(chat_id: &str) -> ChatInfo {
    ChatInfo {
        name: chat_id.to_string(),
        chat_type: "dm".to_string(),
        participants: Vec::new(),
    }
}

// ===========================================================================
// Media type classification & event building
// ===========================================================================

/// Classify the [`MessageType`] of a bridge message.
///
/// Mirrors the `_build_message_event` media-type detection: when `hasMedia`,
/// inspect `mediaType` for `image`/`video`/`audio`/`ptt`, defaulting to
/// document; otherwise text.
pub fn classify_message_type(data: &Value) -> MessageType {
    let has_media = data
        .get("hasMedia")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !has_media {
        return MessageType::Text;
    }
    let media_type = data
        .get("mediaType")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if media_type.contains("image") {
        MessageType::Photo
    } else if media_type.contains("video") {
        MessageType::Video
    } else if media_type.contains("audio") || media_type.contains("ptt") {
        MessageType::Voice
    } else {
        MessageType::Document
    }
}

/// One resolved media URL plus its inferred mime type, produced while building
/// a [`MessageEvent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMedia {
    pub url: String,
    pub media_type: String,
}

/// Whether a URL is an `http(s)` URL. Mirrors `url.startswith(("http://", "https://"))`.
fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Whether a path is absolute. Mirrors `os.path.isabs`.
fn is_abs_path(p: &str) -> bool {
    Path::new(p).is_absolute()
}

/// Lower-cased file extension including the dot (e.g. `.txt`), or empty string.
fn suffix_lower(path: &str) -> String {
    Path::new(path)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

/// Look up a document mime type by extension, defaulting to
/// `application/octet-stream`. Mirrors `SUPPORTED_DOCUMENT_TYPES.get(ext, ...)`.
pub fn document_mime_for_ext(ext: &str) -> String {
    supported_document_types()
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, m)| m.to_string())
        .unwrap_or_else(|| "application/octet-stream".to_string())
}

/// A caching hook: given a remote URL and an extension, return a local path.
/// Mirrors `cache_image_from_url` / `cache_audio_from_url`, which are
/// network-bound and supplied by the caller.
pub type CacheFn<'a> = dyn Fn(&str, &str) -> Result<String, String> + 'a;

/// Resolve a single raw media URL the way `_build_message_event` does.
///
/// For `Photo`/`Voice` http(s) URLs, the corresponding cache function is
/// invoked (image -> `.jpg`/`image/jpeg`; voice -> `.ogg`/`audio/ogg`); on
/// cache failure the original URL is kept with the same mime. Absolute local
/// paths are kept as-is with the type-appropriate mime (documents look up the
/// extension). Anything else is kept with mime `"unknown"`.
pub fn resolve_media_url(
    url: &str,
    msg_type: MessageType,
    cache_image: Option<&CacheFn>,
    cache_audio: Option<&CacheFn>,
) -> ResolvedMedia {
    match msg_type {
        MessageType::Photo if is_http_url(url) => {
            let resolved = cache_image
                .and_then(|f| f(url, ".jpg").ok())
                .unwrap_or_else(|| url.to_string());
            ResolvedMedia {
                url: resolved,
                media_type: "image/jpeg".to_string(),
            }
        }
        MessageType::Photo if is_abs_path(url) => ResolvedMedia {
            url: url.to_string(),
            media_type: "image/jpeg".to_string(),
        },
        MessageType::Voice if is_http_url(url) => {
            let resolved = cache_audio
                .and_then(|f| f(url, ".ogg").ok())
                .unwrap_or_else(|| url.to_string());
            ResolvedMedia {
                url: resolved,
                media_type: "audio/ogg".to_string(),
            }
        }
        MessageType::Voice if is_abs_path(url) => ResolvedMedia {
            url: url.to_string(),
            media_type: "audio/ogg".to_string(),
        },
        MessageType::Document if is_abs_path(url) => {
            let ext = suffix_lower(url);
            ResolvedMedia {
                url: url.to_string(),
                media_type: document_mime_for_ext(&ext),
            }
        }
        MessageType::Video if is_abs_path(url) => ResolvedMedia {
            url: url.to_string(),
            media_type: "video/mp4".to_string(),
        },
        _ => ResolvedMedia {
            url: url.to_string(),
            media_type: "unknown".to_string(),
        },
    }
}

/// Max text-document size eligible for inline injection (100 KB), matching
/// Telegram/Discord/Slack. Mirrors `MAX_TEXT_INJECT_BYTES`.
pub const MAX_TEXT_INJECT_BYTES: u64 = 100 * 1024;

/// Document extensions whose textual content is injected into the message body.
pub const TEXT_INJECT_EXTS: &[&str] = &[
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html",
    ".css",
];

/// Compute the display name for an injected document.
///
/// Mirrors the `doc_<hex>_` prefix-stripping: the basename is split on `_`
/// into at most 3 parts and, when there are >=3, the third part is used.
pub fn injection_display_name(file_name: &str) -> String {
    if file_name.contains('_') {
        let parts: Vec<&str> = file_name.splitn(3, '_').collect();
        if parts.len() >= 3 {
            return parts[2].to_string();
        }
    }
    file_name.to_string()
}

/// Build the injected-content prefix for a readable document.
///
/// Mirrors `f"[Content of {display_name}]:\n{content}"` plus the
/// `f"{injection}\n\n{body}"` / bare-injection merge.
pub fn merge_injection(body: &str, display_name: &str, content: &str) -> String {
    let injection = format!("[Content of {display_name}]:\n{content}");
    if body.is_empty() {
        injection
    } else {
        format!("{injection}\n\n{body}")
    }
}

/// Build a [`MessageEvent`] from bridge message JSON.
///
/// Mirrors `_build_message_event`. Returns `None` when the message is filtered
/// by [`should_process_message`]. `cache_image` / `cache_audio` are the
/// network-bound caching hooks; `read_doc` reads a local document's text (used
/// for inline injection) and reports its size — both are supplied by the
/// caller so this remains pure/testable. `platform` is the adapter platform
/// string used to stamp the [`SessionSource`].
#[allow(clippy::too_many_arguments)]
pub fn build_message_event(
    data: &Value,
    gate: &ProcessGate,
    platform: &str,
    cache_image: Option<&CacheFn>,
    cache_audio: Option<&CacheFn>,
    read_doc: Option<&dyn Fn(&str) -> Result<(String, u64), String>>,
) -> Option<MessageEvent> {
    if !should_process_message(data, gate) {
        return None;
    }

    let msg_type = classify_message_type(data);
    let is_group = data
        .get("isGroup")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let chat_type = if is_group { "group" } else { "dm" };

    let source = SessionSource {
        platform: platform.to_string(),
        chat_id: data
            .get("chatId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        chat_name: data
            .get("chatName")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        chat_type: chat_type.to_string(),
        user_id: data
            .get("senderId")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        user_name: data
            .get("senderName")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        ..SessionSource::default()
    };

    // Resolve media URLs.
    let raw_urls: Vec<String> = data
        .get("mediaUrls")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|u| u.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut cached_urls: Vec<String> = Vec::new();
    let mut media_types: Vec<String> = Vec::new();
    for url in &raw_urls {
        let resolved = resolve_media_url(url, msg_type, cache_image, cache_audio);
        cached_urls.push(resolved.url);
        media_types.push(resolved.media_type);
    }

    // Body, with group bot-mention cleaning.
    let mut body = data
        .get("body")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if is_group {
        body = clean_bot_mention_text(&body, data);
    }

    // Inline-inject readable document text.
    if msg_type == MessageType::Document && !cached_urls.is_empty() {
        for doc_path in cached_urls.clone() {
            let ext = suffix_lower(&doc_path);
            if TEXT_INJECT_EXTS.contains(&ext.as_str()) {
                if let Some(reader) = read_doc {
                    match reader(&doc_path) {
                        Ok((content, size)) => {
                            if size > MAX_TEXT_INJECT_BYTES {
                                continue;
                            }
                            let fname = Path::new(&doc_path)
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            let display_name = injection_display_name(&fname);
                            body = merge_injection(&body, &display_name, &content);
                        }
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    Some(MessageEvent {
        text: body,
        message_type: msg_type,
        source,
        message_id: data
            .get("messageId")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        media_urls: cached_urls,
        media_types,
        ..MessageEvent::default()
    })
}

// ===========================================================================
// Adapter state struct
// ===========================================================================

/// Holds the resolved configuration + lifecycle flags for a WhatsApp adapter,
/// mirroring the Python `WhatsAppAdapter.__init__` field initialisation.
///
/// The async orchestration (`connect`/`disconnect`/`_poll_messages`) is left to
/// the gateway runtime; this captures the behavior-defining derived state.
#[derive(Debug, Clone)]
pub struct WhatsAppAdapter {
    pub name: String,
    pub bridge_port: u16,
    pub bridge_script: PathBuf,
    pub session_path: PathBuf,
    pub reply_prefix: Option<String>,
    pub dm_policy: DmPolicy,
    pub allow_from: HashSet<String>,
    pub group_policy: GroupPolicy,
    pub group_allow_from: HashSet<String>,
    pub mention_patterns: Vec<Regex>,
    pub shutting_down: bool,
}

/// Default bridge directory relative to the install root.
///
/// Mirrors `_DEFAULT_BRIDGE_DIR = parents[2] / "scripts" / "whatsapp-bridge"`.
pub fn default_bridge_dir(install_root: &Path) -> PathBuf {
    install_root.join("scripts").join("whatsapp-bridge")
}

/// Resolved default session path: `get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")`.
pub fn default_session_path() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_dir(
        "platforms/whatsapp/session",
        "whatsapp/session",
    )
}

impl WhatsAppAdapter {
    /// Build the bridge subprocess argv. Mirrors the `subprocess.Popen` argv:
    /// `node <bridge> --port <port> --session <session> --mode <mode>`.
    pub fn bridge_argv(&self, whatsapp_mode: &str) -> Vec<String> {
        vec![
            "node".to_string(),
            self.bridge_script.to_string_lossy().into_owned(),
            "--port".to_string(),
            self.bridge_port.to_string(),
            "--session".to_string(),
            self.session_path.to_string_lossy().into_owned(),
            "--mode".to_string(),
            whatsapp_mode.to_string(),
        ]
    }

    /// Path of the bridge log file: `session_path.parent / "bridge.log"`.
    pub fn bridge_log_path(&self) -> PathBuf {
        let parent = self
            .session_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        parent.join("bridge.log")
    }

    /// Build the [`ProcessGate`] view over this adapter's policy fields.
    pub fn process_gate<'a>(
        &'a self,
        free_response_chats: &'a HashSet<String>,
        require_mention: bool,
    ) -> ProcessGate<'a> {
        ProcessGate {
            dm_policy: self.dm_policy,
            allow_from: &self.allow_from,
            group_policy: self.group_policy,
            group_allow_from: &self.group_allow_from,
            free_response_chats,
            require_mention,
            mention_patterns: &self.mention_patterns,
        }
    }
}

/// WhatsApp operating mode env default. Mirrors
/// `os.getenv("WHATSAPP_MODE", "self-chat")`.
pub fn whatsapp_mode() -> String {
    std::env::var("WHATSAPP_MODE").unwrap_or_else(|_| "self-chat".to_string())
}

/// Decide whether a managed-bridge exit code is a planned shutdown.
///
/// Mirrors the `_check_managed_bridge_exit` shutdown branch: during shutdown,
/// codes `0`, `-2` (SIGINT) and `-15` (SIGTERM) are informational, not fatal.
pub fn is_planned_shutdown_exit(shutting_down: bool, returncode: i32) -> bool {
    shutting_down && matches!(returncode, 0 | -2 | -15)
}

/// Build the fatal-error message for an unexpected bridge exit. Mirrors the
/// `_check_managed_bridge_exit` crash branch.
pub fn bridge_exit_message(returncode: i32) -> String {
    format!("WhatsApp bridge process exited unexpectedly (code {returncode}).")
}

/// Result of a `send` attempt that carries an optional raw response, since the
/// base [`SendResult`] does not model `raw_response`.
#[derive(Debug, Clone, Default)]
pub struct MediaSendResult {
    pub result: SendResult,
    pub raw_response: Option<Value>,
}

/// Whether a `send` of empty/whitespace content should short-circuit to a
/// success with no message id. Mirrors `if not content or not content.strip()`.
pub fn is_empty_send(content: &str) -> bool {
    content.trim().is_empty()
}

/// Chunk + format a message body the way `send` does: format markdown, then
/// split into `MAX_MESSAGE_LENGTH`-bounded chunks.
pub fn chunk_message(content: &str) -> Vec<String> {
    let formatted = format_message(content);
    truncate_message(&formatted, MAX_MESSAGE_LENGTH, None)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_list_csv_and_list() {
        let csv = coerce_allow_list(&AllowListSpec::Csv(" a , b ,, c ".into()));
        assert_eq!(csv, ["a", "b", "c"].iter().map(|s| s.to_string()).collect());
        let list = coerce_allow_list(&AllowListSpec::List(vec![
            " x ".into(),
            "".into(),
            "y".into(),
        ]));
        assert_eq!(list, ["x", "y"].iter().map(|s| s.to_string()).collect());
        assert!(coerce_allow_list(&AllowListSpec::None).is_empty());
    }

    #[test]
    fn dm_and_group_policies() {
        let allow: HashSet<String> = ["12345"].iter().map(|s| s.to_string()).collect();
        assert!(is_dm_allowed(DmPolicy::Open, &allow, "999"));
        assert!(!is_dm_allowed(DmPolicy::Disabled, &allow, "12345"));
        assert!(is_dm_allowed(DmPolicy::Allowlist, &allow, "12345"));
        assert!(!is_dm_allowed(DmPolicy::Allowlist, &allow, "999"));

        let g: HashSet<String> = ["g@x"].iter().map(|s| s.to_string()).collect();
        assert!(is_group_allowed(GroupPolicy::Open, &g, "other@x"));
        assert!(!is_group_allowed(GroupPolicy::Disabled, &g, "g@x"));
        assert!(is_group_allowed(GroupPolicy::Allowlist, &g, "g@x"));
    }

    #[test]
    fn policy_parse_defaults_to_open() {
        assert_eq!(DmPolicy::parse("DISABLED"), DmPolicy::Disabled);
        assert_eq!(DmPolicy::parse(" allowlist "), DmPolicy::Allowlist);
        assert_eq!(DmPolicy::parse("weird"), DmPolicy::Open);
        assert_eq!(GroupPolicy::parse("Allowlist"), GroupPolicy::Allowlist);
    }

    #[test]
    fn normalize_id_replaces_first_colon() {
        assert_eq!(normalize_whatsapp_id(None), "");
        assert_eq!(normalize_whatsapp_id(Some("  ")), "");
        // Has both ':' and '@' -> first ':' becomes '@'.
        assert_eq!(
            normalize_whatsapp_id(Some(" 12345:6@s.whatsapp.net ")),
            "12345@6@s.whatsapp.net"
        );
        // No '@' -> unchanged (just trimmed).
        assert_eq!(normalize_whatsapp_id(Some("12345:6")), "12345:6");
    }

    #[test]
    fn mentions_bot_via_ids_and_body() {
        let data = json!({
            "botIds": ["98765:1@s.whatsapp.net"],
            "mentionedIds": ["98765:1@s.whatsapp.net"],
            "body": "hi",
        });
        assert!(message_mentions_bot(&data));

        let data2 = json!({
            "botIds": ["98765@s.whatsapp.net"],
            "body": "hey @98765 please help",
        });
        assert!(message_mentions_bot(&data2));

        let data3 = json!({ "botIds": [], "body": "@98765" });
        assert!(!message_mentions_bot(&data3));
    }

    #[test]
    fn reply_to_bot_detection() {
        let data = json!({
            "botIds": ["555:2@s.whatsapp.net"],
            "quotedParticipant": "555:2@s.whatsapp.net",
        });
        assert!(message_is_reply_to_bot(&data));
        let no_quote = json!({ "botIds": ["555@s.whatsapp.net"] });
        assert!(!message_is_reply_to_bot(&no_quote));
    }

    #[test]
    fn clean_mention_text_strips_bot_prefix() {
        let data = json!({ "botIds": ["98765@s.whatsapp.net"] });
        assert_eq!(clean_bot_mention_text("@98765, hello there", &data), "hello there");
        // Cleaning to empty -> original returned.
        assert_eq!(clean_bot_mention_text("@98765", &data), "@98765");
    }

    #[test]
    fn should_process_dm_open_always() {
        let empty: HashSet<String> = HashSet::new();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &empty,
            require_mention: true,
            mention_patterns: &pats,
        };
        let dm = json!({ "isGroup": false, "senderId": "12345", "body": "no mention" });
        assert!(should_process_message(&dm, &gate));
    }

    #[test]
    fn should_process_group_requires_mention() {
        let empty: HashSet<String> = HashSet::new();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &empty,
            require_mention: true,
            mention_patterns: &pats,
        };
        // No mention, not a command, not a reply -> filtered out.
        let g = json!({ "isGroup": true, "chatId": "g@x", "body": "hello world" });
        assert!(!should_process_message(&g, &gate));
        // Slash command bypasses.
        let cmd = json!({ "isGroup": true, "chatId": "g@x", "body": "/new" });
        assert!(should_process_message(&cmd, &gate));
    }

    #[test]
    fn should_process_group_free_response_chat() {
        let empty: HashSet<String> = HashSet::new();
        let free: HashSet<String> = ["g@x"].iter().map(|s| s.to_string()).collect();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &free,
            require_mention: true,
            mention_patterns: &pats,
        };
        let g = json!({ "isGroup": true, "chatId": "g@x", "body": "no mention" });
        assert!(should_process_message(&g, &gate));
    }

    #[test]
    fn format_message_conversions() {
        assert_eq!(format_message("**bold**"), "*bold*");
        assert_eq!(format_message("__bold__"), "*bold*");
        assert_eq!(format_message("~~strike~~"), "~strike~");
        assert_eq!(format_message("## Header"), "*Header*");
        assert_eq!(
            format_message("[text](http://x.com)"),
            "text (http://x.com)"
        );
        // Existing single-* italic left alone.
        assert_eq!(format_message("*italic*"), "*italic*");
    }

    #[test]
    fn format_message_protects_code() {
        // Inline code with ** inside should be preserved verbatim.
        let out = format_message("text `**not bold**` more");
        assert_eq!(out, "text `**not bold**` more");
        // Fenced block preserved.
        let out2 = format_message("```\n**keep**\n```");
        assert_eq!(out2, "```\n**keep**\n```");
    }

    #[test]
    fn format_message_empty() {
        assert_eq!(format_message(""), "");
    }

    #[test]
    fn send_payload_reply_only_on_first_chunk() {
        let with = build_send_payload("c", "m", Some("r"), true);
        assert_eq!(with["replyTo"], json!("r"));
        let without = build_send_payload("c", "m", Some("r"), false);
        assert!(without.get("replyTo").is_none());
        // No reply target -> no key even when included.
        let none = build_send_payload("c", "m", None, true);
        assert!(none.get("replyTo").is_none());
    }

    #[test]
    fn media_payload_optional_fields() {
        let p = build_send_media_payload("c", "/f.png", "image", Some("cap"), None);
        assert_eq!(p["caption"], json!("cap"));
        assert!(p.get("fileName").is_none());
        // Empty caption omitted.
        let p2 = build_send_media_payload("c", "/f.png", "image", Some(""), Some("f.png"));
        assert!(p2.get("caption").is_none());
        assert_eq!(p2["fileName"], json!("f.png"));
    }

    #[test]
    fn parse_health_defaults_unknown() {
        assert_eq!(parse_health(r#"{"status":"connected"}"#).status, "connected");
        assert_eq!(parse_health("{}").status, "unknown");
        assert_eq!(parse_health("not json").status, "unknown");
    }

    #[test]
    fn parse_chat_info_and_default() {
        let info = parse_chat_info(
            r#"{"name":"Team","isGroup":true,"participants":["a","b"]}"#,
            "fallback",
        );
        assert_eq!(info.name, "Team");
        assert_eq!(info.chat_type, "group");
        assert_eq!(info.participants, vec!["a", "b"]);
        // Missing name -> chat_id fallback, dm type.
        let info2 = parse_chat_info("{}", "id123");
        assert_eq!(info2.name, "id123");
        assert_eq!(info2.chat_type, "dm");
        assert_eq!(default_chat_info("x"), ChatInfo {
            name: "x".into(),
            chat_type: "dm".into(),
            participants: vec![],
        });
    }

    #[test]
    fn classify_types() {
        assert_eq!(classify_message_type(&json!({})), MessageType::Text);
        assert_eq!(
            classify_message_type(&json!({"hasMedia": true, "mediaType": "image/jpeg"})),
            MessageType::Photo
        );
        assert_eq!(
            classify_message_type(&json!({"hasMedia": true, "mediaType": "video/mp4"})),
            MessageType::Video
        );
        assert_eq!(
            classify_message_type(&json!({"hasMedia": true, "mediaType": "ptt"})),
            MessageType::Voice
        );
        assert_eq!(
            classify_message_type(&json!({"hasMedia": true, "mediaType": "application/pdf"})),
            MessageType::Document
        );
    }

    #[test]
    fn resolve_media_photo_cache_and_fallback() {
        let ok: Box<CacheFn> = Box::new(|_url, ext| Ok(format!("/cache/file{ext}")));
        let r = resolve_media_url(
            "https://x.com/a.jpg",
            MessageType::Photo,
            Some(&*ok),
            None,
        );
        assert_eq!(r.url, "/cache/file.jpg");
        assert_eq!(r.media_type, "image/jpeg");

        // Cache failure -> keep original URL, same mime.
        let bad: Box<CacheFn> = Box::new(|_u, _e| Err("nope".into()));
        let r2 = resolve_media_url(
            "https://x.com/a.jpg",
            MessageType::Photo,
            Some(&*bad),
            None,
        );
        assert_eq!(r2.url, "https://x.com/a.jpg");
        assert_eq!(r2.media_type, "image/jpeg");
    }

    #[test]
    fn resolve_media_abs_and_unknown() {
        let doc = resolve_media_url("/tmp/file.json", MessageType::Document, None, None);
        assert_eq!(doc.media_type, "application/json");
        let vid = resolve_media_url("/tmp/v.mp4", MessageType::Video, None, None);
        assert_eq!(vid.media_type, "video/mp4");
        // Non-http, non-abs photo -> unknown.
        let unk = resolve_media_url("relative/x", MessageType::Photo, None, None);
        assert_eq!(unk.media_type, "unknown");
    }

    #[test]
    fn injection_display_name_strips_prefix() {
        assert_eq!(injection_display_name("doc_ab12_report.txt"), "report.txt");
        assert_eq!(injection_display_name("plain.txt"), "plain.txt");
        assert_eq!(injection_display_name("a_b"), "a_b");
    }

    #[test]
    fn merge_injection_with_and_without_body() {
        assert_eq!(
            merge_injection("question?", "f.txt", "hello"),
            "[Content of f.txt]:\nhello\n\nquestion?"
        );
        assert_eq!(
            merge_injection("", "f.txt", "hello"),
            "[Content of f.txt]:\nhello"
        );
    }

    #[test]
    fn build_event_filters_and_caches() {
        let empty: HashSet<String> = HashSet::new();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &empty,
            require_mention: false,
            mention_patterns: &pats,
        };
        let data = json!({
            "isGroup": false,
            "senderId": "12345",
            "chatId": "12345@c.us",
            "messageId": "MID1",
            "body": "look at this",
            "hasMedia": true,
            "mediaType": "image/jpeg",
            "mediaUrls": ["https://x.com/a.jpg"],
        });
        let cache: Box<CacheFn> = Box::new(|_u, ext| Ok(format!("/cache/img{ext}")));
        let ev = build_message_event(&data, &gate, "whatsapp", Some(&*cache), None, None)
            .expect("event");
        assert_eq!(ev.message_type, MessageType::Photo);
        assert_eq!(ev.media_urls, vec!["/cache/img.jpg"]);
        assert_eq!(ev.media_types, vec!["image/jpeg"]);
        assert_eq!(ev.message_id.as_deref(), Some("MID1"));
        assert_eq!(ev.source.platform, "whatsapp");
        assert_eq!(ev.source.chat_type, "dm");

        // Disabled DM -> filtered.
        let gate2 = ProcessGate { dm_policy: DmPolicy::Disabled, ..gate.clone() };
        assert!(build_message_event(&data, &gate2, "whatsapp", None, None, None).is_none());
    }

    #[test]
    fn build_event_injects_document_text() {
        let empty: HashSet<String> = HashSet::new();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &empty,
            require_mention: false,
            mention_patterns: &pats,
        };
        let data = json!({
            "isGroup": false,
            "senderId": "1",
            "chatId": "c",
            "body": "see attached",
            "hasMedia": true,
            "mediaType": "application/json",
            "mediaUrls": ["/tmp/doc_ff00_data.json"],
        });
        let reader: Box<dyn Fn(&str) -> Result<(String, u64), String>> =
            Box::new(|_p| Ok(("{\"k\":1}".to_string(), 8)));
        let ev =
            build_message_event(&data, &gate, "whatsapp", None, None, Some(&*reader)).unwrap();
        assert_eq!(ev.message_type, MessageType::Document);
        assert!(ev.text.starts_with("[Content of data.json]:\n{\"k\":1}"));
        assert!(ev.text.ends_with("see attached"));
    }

    #[test]
    fn build_event_skips_large_document() {
        let empty: HashSet<String> = HashSet::new();
        let pats: Vec<Regex> = Vec::new();
        let gate = ProcessGate {
            dm_policy: DmPolicy::Open,
            allow_from: &empty,
            group_policy: GroupPolicy::Open,
            group_allow_from: &empty,
            free_response_chats: &empty,
            require_mention: false,
            mention_patterns: &pats,
        };
        let data = json!({
            "isGroup": false,
            "senderId": "1",
            "chatId": "c",
            "body": "big",
            "hasMedia": true,
            "mediaType": "text/plain",
            "mediaUrls": ["/tmp/huge.txt"],
        });
        let reader: Box<dyn Fn(&str) -> Result<(String, u64), String>> =
            Box::new(|_p| Ok(("x".to_string(), MAX_TEXT_INJECT_BYTES + 1)));
        let ev =
            build_message_event(&data, &gate, "whatsapp", None, None, Some(&*reader)).unwrap();
        // Oversized -> not injected; body unchanged.
        assert_eq!(ev.text, "big");
    }

    #[test]
    fn planned_shutdown_exit_codes() {
        assert!(is_planned_shutdown_exit(true, 0));
        assert!(is_planned_shutdown_exit(true, -15));
        assert!(is_planned_shutdown_exit(true, -2));
        assert!(!is_planned_shutdown_exit(true, 1));
        assert!(!is_planned_shutdown_exit(false, 0));
        assert_eq!(
            bridge_exit_message(7),
            "WhatsApp bridge process exited unexpectedly (code 7)."
        );
    }

    #[test]
    fn empty_send_short_circuit() {
        assert!(is_empty_send("   "));
        assert!(is_empty_send(""));
        assert!(!is_empty_send("hi"));
    }

    #[test]
    fn mention_patterns_env_json_and_csv() {
        unsafe { std::env::set_var("WHATSAPP_MENTION_PATTERNS", r#"["\\bbot\\b", "assistant"]"#) };
        let p = compile_mention_patterns(None);
        assert_eq!(p.len(), 2);
        assert!(message_matches_mention_patterns(&json!({"body": "hey bot"}), &p));
        unsafe { std::env::remove_var("WHATSAPP_MENTION_PATTERNS") };

        // Config string -> single pattern.
        let p2 = compile_mention_patterns(Some(&MentionPatternSpec::Str("hello".into())));
        assert_eq!(p2.len(), 1);
    }

    #[test]
    fn require_mention_env_and_config() {
        unsafe { std::env::remove_var("WHATSAPP_REQUIRE_MENTION") };
        assert!(!whatsapp_require_mention(None));
        assert!(whatsapp_require_mention(Some(&ConfigBool::Bool(true))));
        assert!(whatsapp_require_mention(Some(&ConfigBool::Str("yes".into()))));
        assert!(!whatsapp_require_mention(Some(&ConfigBool::Str("nope".into()))));
        unsafe { std::env::set_var("WHATSAPP_REQUIRE_MENTION", "on") };
        assert!(whatsapp_require_mention(None));
        unsafe { std::env::remove_var("WHATSAPP_REQUIRE_MENTION") };
    }

    #[test]
    fn chunk_short_message_single() {
        let chunks = chunk_message("**bold**");
        assert_eq!(chunks, vec!["*bold*".to_string()]);
    }

    #[test]
    fn bridge_argv_shape() {
        let adapter = WhatsAppAdapter {
            name: "whatsapp".into(),
            bridge_port: 3000,
            bridge_script: PathBuf::from("/b/bridge.js"),
            session_path: PathBuf::from("/s/session"),
            reply_prefix: None,
            dm_policy: DmPolicy::Open,
            allow_from: HashSet::new(),
            group_policy: GroupPolicy::Open,
            group_allow_from: HashSet::new(),
            mention_patterns: Vec::new(),
            shutting_down: false,
        };
        let argv = adapter.bridge_argv("self-chat");
        assert_eq!(
            argv,
            vec![
                "node",
                "/b/bridge.js",
                "--port",
                "3000",
                "--session",
                "/s/session",
                "--mode",
                "self-chat"
            ]
        );
        assert_eq!(adapter.bridge_log_path(), PathBuf::from("/s/bridge.log"));
    }
}
