//! Feishu/Lark platform adapter — native Rust port of
//! `gateway/platforms/feishu.py`.
//!
//! This module reproduces the pure-logic surface of the Python adapter:
//!
//! - Post / card / merge-forward / share-chat message normalization
//! - Markdown rendering helpers (text element styling, code fences, link
//!   stripping) used to build Feishu `post` payloads and plain-text fallbacks
//! - Mention parsing, self-mention stripping and mention hints
//! - Inbound admission policy (self-echo, bot gating, group policy, mention
//!   gating) with per-group rules
//! - Persistent dedup cache with TTL + size cap
//! - Webhook security: rate-limiting, anomaly tracking, signature verification,
//!   verification-token check, card-action dedup
//! - Text/media batching key + compatibility logic
//! - Outbound payload construction (`text` vs `post`, media post payloads,
//!   approval cards, resolved approval cards, file routing)
//! - QR scan-to-create onboarding request construction + response parsing and
//!   the `bot/v3/info` probe (using `reqwest::blocking`)
//!
//! The async SDK orchestration (websockets long-connection, aiohttp webhook
//! server, lark_oapi request/response objects) is represented here as the
//! request-construction + response-parsing logic; the live event loop wiring is
//! handled by the gateway runtime layer.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Cross-module type aliases. We reference the canonical gateway types when
// helpful; the adapter logic operates on lightweight local structs that mirror
// the Python dataclasses.
// ---------------------------------------------------------------------------

pub use crate::gateway::MessageType;

// ---------------------------------------------------------------------------
// Media type sets and upload constants
// ---------------------------------------------------------------------------

pub const IMAGE_EXTENSIONS: &[&str] = &[".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"];
pub const AUDIO_EXTENSIONS: &[&str] =
    &[".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus", ".webm"];
pub const VIDEO_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"];

pub const FEISHU_IMAGE_UPLOAD_TYPE: &str = "message";
pub const FEISHU_FILE_UPLOAD_TYPE: &str = "stream";
pub const FEISHU_OPUS_UPLOAD_EXTENSIONS: &[&str] = &[".ogg", ".opus"];
pub const FEISHU_MEDIA_UPLOAD_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".avi", ".m4v"];

/// `.ext -> upload doc type` for `_resolve_outbound_file_routing`.
pub fn feishu_doc_upload_type(ext: &str) -> Option<&'static str> {
    match ext {
        ".pdf" => Some("pdf"),
        ".doc" | ".docx" => Some("doc"),
        ".xls" | ".xlsx" => Some("xls"),
        ".ppt" | ".pptx" => Some("ppt"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Connection, retry, batching tuning
// ---------------------------------------------------------------------------

pub const MAX_TEXT_INJECT_BYTES: u64 = 100 * 1024;
pub const FEISHU_CONNECT_ATTEMPTS: u32 = 3;
pub const FEISHU_SEND_ATTEMPTS: u32 = 3;
pub const FEISHU_APP_LOCK_SCOPE: &str = "feishu-app-id";
pub const DEFAULT_TEXT_BATCH_DELAY_SECONDS: f64 = 0.6;
pub const DEFAULT_TEXT_BATCH_MAX_MESSAGES: i64 = 8;
pub const DEFAULT_TEXT_BATCH_MAX_CHARS: i64 = 4000;
pub const DEFAULT_MEDIA_BATCH_DELAY_SECONDS: f64 = 0.8;
pub const DEFAULT_DEDUP_CACHE_SIZE: i64 = 2048;
pub const DEFAULT_WEBHOOK_HOST: &str = "127.0.0.1";
pub const DEFAULT_WEBHOOK_PORT: i64 = 8765;
pub const DEFAULT_WEBHOOK_PATH: &str = "/feishu/webhook";

// ---------------------------------------------------------------------------
// TTL, rate-limit and webhook security constants
// ---------------------------------------------------------------------------

pub const FEISHU_DEDUP_TTL_SECONDS: f64 = 24.0 * 60.0 * 60.0;
pub const FEISHU_SENDER_NAME_TTL_SECONDS: f64 = 10.0 * 60.0;
pub const FEISHU_WEBHOOK_MAX_BODY_BYTES: usize = 1024 * 1024;
pub const FEISHU_WEBHOOK_RATE_WINDOW_SECONDS: f64 = 60.0;
pub const FEISHU_WEBHOOK_RATE_LIMIT_MAX: u32 = 120;
pub const FEISHU_WEBHOOK_RATE_MAX_KEYS: usize = 4096;
pub const FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS: u64 = 30;
pub const FEISHU_WEBHOOK_ANOMALY_THRESHOLD: u32 = 25;
pub const FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS: f64 = 6.0 * 60.0 * 60.0;
pub const FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS: f64 = 15.0 * 60.0;

pub const FEISHU_BOT_MSG_TRACK_SIZE: usize = 512;
/// reply target withdrawn/missing → create fallback
pub const FEISHU_REPLY_FALLBACK_CODES: &[i64] = &[230011, 231003];

pub const FEISHU_REACTION_IN_PROGRESS: &str = "Typing";
pub const FEISHU_REACTION_FAILURE: &str = "CrossMark";
pub const FEISHU_PROCESSING_REACTION_CACHE_SIZE: usize = 1024;

// QR onboarding constants
pub fn onboard_accounts_url(domain: &str) -> &'static str {
    match domain {
        "lark" => "https://accounts.larksuite.com",
        _ => "https://accounts.feishu.cn",
    }
}
pub fn onboard_open_url(domain: &str) -> &'static str {
    match domain {
        "lark" => "https://open.larksuite.com",
        _ => "https://open.feishu.cn",
    }
}
pub const REGISTRATION_PATH: &str = "/oauth/v1/app/registration";
pub const ONBOARD_REQUEST_TIMEOUT_S: u64 = 10;

// ---------------------------------------------------------------------------
// Fallback display strings
// ---------------------------------------------------------------------------

pub const FALLBACK_POST_TEXT: &str = "[Rich text message]";
pub const FALLBACK_FORWARD_TEXT: &str = "[Merged forward message]";
pub const FALLBACK_SHARE_CHAT_TEXT: &str = "[Shared chat]";
pub const FALLBACK_INTERACTIVE_TEXT: &str = "[Interactive message]";
pub const FALLBACK_IMAGE_TEXT: &str = "[Image]";
pub const FALLBACK_ATTACHMENT_TEXT: &str = "[Attachment]";

const PREFERRED_LOCALES: &[&str] = &["zh_cn", "en_us"];

const SUPPORTED_CARD_TEXT_KEYS: &[&str] = &[
    "title",
    "text",
    "content",
    "label",
    "value",
    "name",
    "summary",
    "subtitle",
    "description",
    "placeholder",
    "hint",
];

fn is_skip_text_key(key: &str) -> bool {
    matches!(
        key,
        "tag" | "type"
            | "msg_type"
            | "message_type"
            | "chat_id"
            | "open_chat_id"
            | "share_chat_id"
            | "file_key"
            | "image_key"
            | "user_id"
            | "open_id"
            | "union_id"
            | "url"
            | "href"
            | "link"
            | "token"
            | "template"
            | "locale"
    )
}

const MENTION_BOUNDARY_CHARS: &str = " \t\n\r.,;:!?、，。；：！？()[]{}<>\"'`";
const TRAILING_TERMINAL_PUNCT: &str = " \t\n\r.!?。！？";

// ---------------------------------------------------------------------------
// Approval choice maps
// ---------------------------------------------------------------------------

pub fn approval_choice_map(action: &str) -> &'static str {
    match action {
        "approve_once" => "once",
        "approve_session" => "session",
        "approve_always" => "always",
        "deny" => "deny",
        _ => "deny",
    }
}

pub fn approval_label_map(choice: &str) -> &'static str {
    match choice {
        "once" => "Approved once",
        "session" => "Approved for session",
        "always" => "Approved permanently",
        "deny" => "Denied",
        _ => "Resolved",
    }
}

// ---------------------------------------------------------------------------
// Dataclasses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeishuPostMediaRef {
    pub file_key: String,
    pub file_name: String,
    pub resource_type: String,
}

impl FeishuPostMediaRef {
    pub fn new(file_key: impl Into<String>) -> Self {
        FeishuPostMediaRef {
            file_key: file_key.into(),
            file_name: String::new(),
            resource_type: "file".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeishuMentionRef {
    pub name: String,
    pub open_id: String,
    pub is_all: bool,
    pub is_self: bool,
}

impl FeishuMentionRef {
    pub fn all() -> Self {
        FeishuMentionRef {
            is_all: true,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeishuBotIdentity {
    pub open_id: String,
    pub user_id: String,
    pub name: String,
}

impl FeishuBotIdentity {
    /// Precedence: open_id > user_id > name. IDs are authoritative when both
    /// sides have them; the next tier is only considered when either side
    /// lacks the current one.
    pub fn matches(&self, open_id: &str, user_id: &str, name: &str) -> bool {
        if !open_id.is_empty() && !self.open_id.is_empty() {
            return open_id == self.open_id;
        }
        if !user_id.is_empty() && !self.user_id.is_empty() {
            return user_id == self.user_id;
        }
        !self.name.is_empty() && name == self.name
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeishuPostParseResult {
    pub text_content: String,
    pub image_keys: Vec<String>,
    pub media_refs: Vec<FeishuPostMediaRef>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeishuNormalizedMessage {
    pub raw_type: String,
    pub text_content: String,
    pub preferred_message_type: String,
    pub image_keys: Vec<String>,
    pub media_refs: Vec<FeishuPostMediaRef>,
    pub mentions: Vec<FeishuMentionRef>,
    pub relation_kind: String,
    pub metadata: BTreeMap<String, Value>,
}

impl FeishuNormalizedMessage {
    fn empty(raw_type: &str) -> Self {
        FeishuNormalizedMessage {
            raw_type: raw_type.to_string(),
            preferred_message_type: "text".to_string(),
            relation_kind: "plain".to_string(),
            ..Default::default()
        }
    }
}

/// Per-group policy rule (mirrors `FeishuGroupRule`).
#[derive(Debug, Clone, Default)]
pub struct FeishuGroupRule {
    pub policy: String,
    pub allowlist: HashSet<String>,
    pub blacklist: HashSet<String>,
    /// None = inherit global.
    pub require_mention: Option<bool>,
}

// ---------------------------------------------------------------------------
// Admission types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    SelfEcho,
    SelfIdsUnknown,
    BotsDisabled,
    BotNotMentioned,
    GroupPolicyRejected,
}

impl RejectReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            RejectReason::SelfEcho => "self_echo",
            RejectReason::SelfIdsUnknown => "self_ids_unknown",
            RejectReason::BotsDisabled => "bots_disabled",
            RejectReason::BotNotMentioned => "bot_not_mentioned",
            RejectReason::GroupPolicyRejected => "group_policy_rejected",
        }
    }
}

/// receive_v1 docs say {user, bot}; accept "app" defensively.
pub fn is_bot_sender(sender_type: &str) -> bool {
    sender_type == "bot" || sender_type == "app"
}

// ---------------------------------------------------------------------------
// Markdown rendering helpers
// ---------------------------------------------------------------------------

const MARKDOWN_SPECIAL_CHARS: &str = "\\`*_{}[]()#+-!|>~";

pub fn escape_markdown_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if MARKDOWN_SPECIAL_CHARS.contains(ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

pub fn to_boolean(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_i64() == Some(1),
        Value::String(s) => s == "true",
        _ => false,
    }
}

fn is_style_enabled(style: Option<&Value>, key: &str) -> bool {
    match style {
        Some(Value::Object(map)) => map.get(key).map(to_boolean).unwrap_or(false),
        _ => false,
    }
}

/// Wrap text in a backtick fence longer than any run of backticks inside it.
pub fn wrap_inline_code(text: &str) -> String {
    let mut max_run = 0usize;
    let mut cur = 0usize;
    for ch in text.chars() {
        if ch == '`' {
            cur += 1;
            if cur > max_run {
                max_run = cur;
            }
        } else {
            cur = 0;
        }
    }
    let fence = "`".repeat(max_run + 1);
    let body = if text.starts_with('`') || text.ends_with('`') {
        format!(" {text} ")
    } else {
        text.to_string()
    };
    format!("{fence}{body}{fence}")
}

fn sanitize_fence_language(language: &str) -> String {
    language.trim().replace('\n', " ").replace('\r', " ")
}

fn value_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Render a Feishu post `text` element to markdown (`_render_text_element`).
fn render_text_element(element: &Value) -> String {
    let text = match element.get("text") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    let style = element.get("style");
    if is_style_enabled(style, "code") {
        return wrap_inline_code(&text);
    }
    let rendered = escape_markdown_text(&text);
    if rendered.is_empty() {
        return String::new();
    }
    let mut rendered = rendered;
    if is_style_enabled(style, "bold") {
        rendered = format!("**{rendered}**");
    }
    if is_style_enabled(style, "italic") {
        rendered = format!("*{rendered}*");
    }
    if is_style_enabled(style, "underline") {
        rendered = format!("<u>{rendered}</u>");
    }
    if is_style_enabled(style, "strikethrough") {
        rendered = format!("~~{rendered}~~");
    }
    rendered
}

fn render_code_block_element(element: &Value) -> String {
    let mut language = value_str(element.get("language"));
    if language.is_empty() {
        language = value_str(element.get("lang"));
    }
    let language = sanitize_fence_language(&language);
    let mut code = value_str(element.get("text"));
    if code.is_empty() {
        code = value_str(element.get("content"));
    }
    let code = code.replace("\r\n", "\n");
    let trailing = if code.ends_with('\n') { "" } else { "\n" };
    format!("```{language}\n{code}{trailing}```")
}

/// Strip markdown to plain text for Feishu text fallbacks.
pub fn strip_markdown_to_plain_text(text: &str) -> String {
    let mut plain = text.replace("\r\n", "\n");
    // [label](url) -> "label (url)"
    plain = replace_markdown_links(&plain, |label, url| format!("{label} ({})", url.trim()));
    // ^>\s? per line
    plain = plain
        .lines()
        .map(|line| {
            if let Some(rest) = line.strip_prefix("> ") {
                rest
            } else if let Some(rest) = line.strip_prefix('>') {
                rest
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    // ^\s*---+\s*$ -> "---"
    plain = plain
        .lines()
        .map(|line| {
            let t = line.trim();
            if t.len() >= 3 && t.chars().all(|c| c == '-') {
                "---"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    // ~~strikethrough~~
    plain = strip_paired_marker(&plain, "~~", '~');
    // <u>...</u>
    plain = strip_html_underline(&plain);
    // shared strip_markdown (headers, bold/italic, inline code, fences, lists)
    plain = generic_strip_markdown(&plain);
    plain
}

/// Mirror of `_MARKDOWN_LINK_RE.sub` with a callback.
fn replace_markdown_links(text: &str, f: impl Fn(&str, &str) -> String) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(close) = find_byte(bytes, i + 1, b']') {
                if close + 1 < bytes.len() && bytes[close + 1] == b'(' {
                    if let Some(paren) = find_byte(bytes, close + 2, b')') {
                        let label = &text[i + 1..close];
                        let url = &text[close + 2..paren];
                        if !label.contains(']') && !url.contains(')') {
                            out.push_str(&f(label, url));
                            i = paren + 1;
                            continue;
                        }
                    }
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn find_byte(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
    (start..bytes.len()).find(|&j| bytes[j] == target)
}

/// Strip a paired inline marker like `~~text~~` where the inner text contains
/// no newline and no occurrence of `inner_char` (mirrors `~~([^~\n]+)~~`).
fn strip_paired_marker(text: &str, marker: &str, inner_char: char) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(marker) {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + marker.len()..];
        if let Some(close) = after_open.find(marker) {
            let inner = &after_open[..close];
            if !inner.is_empty() && !inner.contains('\n') && !inner.contains(inner_char) {
                out.push_str(inner);
                rest = &after_open[close + marker.len()..];
                continue;
            }
        }
        out.push_str(marker);
        rest = after_open;
    }
    out.push_str(rest);
    out
}

/// Strip `<u>...</u>` tags (mirrors `<u>([\s\S]*?)</u>`).
fn strip_html_underline(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<u>") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 3..];
        if let Some(close) = after.find("</u>") {
            out.push_str(&after[..close]);
            rest = &after[close + 4..];
        } else {
            out.push_str("<u>");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// A pragmatic port of the shared `strip_markdown` helper: removes heading
/// markers, list bullets, emphasis runs, inline code backticks and fences.
fn generic_strip_markdown(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut in_fence = false;
    for raw in text.split('\n') {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            lines.push(raw.to_string());
            continue;
        }
        let mut line = raw.to_string();
        // headings: leading #'s + space
        {
            let lt = line.trim_start();
            let hashes = lt.chars().take_while(|c| *c == '#').count();
            if (1..=6).contains(&hashes) {
                let after = &lt[hashes..];
                if after.starts_with(' ') {
                    line = after.trim_start().to_string();
                }
            }
        }
        // list bullets: -, *, +, or "N. "
        {
            let lt = line.trim_start();
            if let Some(rest) = lt
                .strip_prefix("- ")
                .or_else(|| lt.strip_prefix("* "))
                .or_else(|| lt.strip_prefix("+ "))
            {
                line = rest.to_string();
            } else {
                let digits = lt.chars().take_while(|c| c.is_ascii_digit()).count();
                if digits > 0 && lt[digits..].starts_with(". ") {
                    line = lt[digits + 2..].to_string();
                }
            }
        }
        lines.push(line);
    }
    let joined = lines.join("\n");
    // emphasis + inline code
    let joined = strip_emphasis(&joined, "**");
    let joined = strip_emphasis(&joined, "__");
    let joined = strip_emphasis(&joined, "*");
    let joined = strip_emphasis(&joined, "_");
    strip_inline_code_backticks(&joined)
}

fn strip_emphasis(text: &str, marker: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(marker) {
        out.push_str(&rest[..open]);
        let after = &rest[open + marker.len()..];
        if let Some(close) = after.find(marker) {
            let inner = &after[..close];
            if !inner.is_empty() && !inner.contains('\n') {
                out.push_str(inner);
                rest = &after[close + marker.len()..];
                continue;
            }
        }
        out.push_str(marker);
        rest = after;
    }
    out.push_str(rest);
    out
}

fn strip_inline_code_backticks(text: &str) -> String {
    text.replace('`', "")
}

/// Coerce value to int with optional default and minimum constraint.
pub fn coerce_int(value: &Value, default: Option<i64>, min_value: i64) -> Option<i64> {
    let parsed = match value {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    };
    match parsed {
        Some(p) if p >= min_value => Some(p),
        _ => default,
    }
}

pub fn coerce_required_int(value: &Value, default: i64, min_value: i64) -> i64 {
    coerce_int(value, Some(default), min_value).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Markdown hint / table detection (regex-equivalent scanners)
// ---------------------------------------------------------------------------

/// Detect markdown tables: a line starting with `|` followed by a separator
/// line `|[-|: ]+|`. Equivalent to `_MARKDOWN_TABLE_RE`.
pub fn has_markdown_table(content: &str) -> bool {
    let lines: Vec<&str> = content.split('\n').collect();
    for w in lines.windows(2) {
        let first = w[0];
        let second = w[1];
        if first.starts_with('|')
            && first[1..].contains('|')
            && second.starts_with('|')
            && second.ends_with('|')
            && second.len() >= 2
            && second[1..second.len() - 1]
                .chars()
                .all(|c| matches!(c, '-' | '|' | ':' | ' '))
            && !second[1..second.len() - 1].is_empty()
        {
            return true;
        }
    }
    false
}

/// Heuristic for whether content contains markdown formatting that warrants a
/// `post` payload rather than plain text. Equivalent to `_MARKDOWN_HINT_RE`.
pub fn has_markdown_hint(content: &str) -> bool {
    if content.contains("```") {
        return true;
    }
    for line in content.split('\n') {
        let lt = line.trim_start();
        // headings: #{1,6} followed by space
        let hashes = lt.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) && lt[hashes..].starts_with(' ') {
            return true;
        }
        // unordered list: -, *
        if lt.starts_with("- ") || lt.starts_with("* ") {
            return true;
        }
        // ordered list: N. followed by space
        let digits = lt.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && lt[digits..].starts_with(". ") {
            return true;
        }
        // horizontal rule
        let t = line.trim();
        if t.len() >= 3 && t.chars().all(|c| c == '-') {
            return true;
        }
        // blockquote
        if line.starts_with("> ") {
            return true;
        }
    }
    // inline code `x`
    if has_inline_code(content) {
        return true;
    }
    // bold **x**
    if has_paired(content, "**") {
        return true;
    }
    // strikethrough ~~x~~
    if has_paired(content, "~~") {
        return true;
    }
    // underline <u>x</u>
    if content.contains("<u>") && content.contains("</u>") {
        return true;
    }
    // italic *x*
    if has_paired(content, "*") {
        return true;
    }
    // links [x](y)
    has_markdown_link(content)
}

fn has_inline_code(text: &str) -> bool {
    let mut rest = text;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        if let Some(rel) = after.find('`') {
            let inner = &after[..rel];
            if !inner.is_empty() && !inner.contains('\n') {
                return true;
            }
            rest = &after[rel + 1..];
        } else {
            break;
        }
    }
    false
}

fn has_paired(text: &str, marker: &str) -> bool {
    let mut rest = text;
    while let Some(open) = rest.find(marker) {
        let after = &rest[open + marker.len()..];
        if let Some(rel) = after.find(marker) {
            let inner = &after[..rel];
            if !inner.is_empty()
                && !inner.starts_with(marker.chars().next().unwrap())
                && !inner.contains('\n')
            {
                return true;
            }
            rest = &after[rel + marker.len()..];
        } else {
            break;
        }
    }
    false
}

fn has_markdown_link(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(close) = find_byte(bytes, i + 1, b']') {
                if close + 1 < bytes.len()
                    && bytes[close + 1] == b'('
                    && find_byte(bytes, close + 2, b')').is_some()
                {
                    let label = &text[i + 1..close];
                    if !label.is_empty() && !label.contains(']') {
                        return true;
                    }
                }
            }
        }
        i += 1;
    }
    false
}

/// True if content matches `content format of the post type is incorrect`
/// (case-insensitive) — `_POST_CONTENT_INVALID_RE`.
pub fn is_post_content_invalid(text: &str) -> bool {
    text.to_lowercase()
        .contains("content format of the post type is incorrect")
}

// ---------------------------------------------------------------------------
// Post payload builders
// ---------------------------------------------------------------------------

const FENCE_OPEN_MIN: usize = 3;

fn is_fence_open(stripped: &str) -> bool {
    // ^```([^\n`]*)\s*$  — opening fence: ``` optionally followed by a language
    // token containing no backticks, then trailing whitespace.
    if !stripped.starts_with("```") {
        return false;
    }
    let after = &stripped[FENCE_OPEN_MIN..];
    let lang = after.trim_end();
    !lang.contains('`')
}

fn is_fence_close(stripped: &str) -> bool {
    // ^```\s*$
    stripped.starts_with("```") && stripped[FENCE_OPEN_MIN..].trim().is_empty()
}

/// Build Feishu post rows while isolating fenced code blocks.
/// Returns rows where each row is a list of `{tag, text}` elements.
pub fn build_markdown_post_rows(content: &str) -> Vec<Vec<Value>> {
    if content.is_empty() {
        return vec![vec![json!({"tag": "md", "text": ""})]];
    }
    if !content.contains("```") {
        return vec![vec![json!({"tag": "md", "text": content})]];
    }

    let mut rows: Vec<Vec<Value>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut in_code_block = false;

    let flush = |current: &mut Vec<String>, rows: &mut Vec<Vec<Value>>| {
        if current.is_empty() {
            return;
        }
        let segment = current.join("\n");
        if !segment.trim().is_empty() {
            rows.push(vec![json!({"tag": "md", "text": segment})]);
        }
        current.clear();
    };

    for raw_line in splitlines(content) {
        let stripped = raw_line.trim();
        let is_fence = if in_code_block {
            is_fence_close(stripped)
        } else {
            is_fence_open(stripped)
        };

        if is_fence {
            if !in_code_block {
                flush(&mut current, &mut rows);
            }
            current.push(raw_line.to_string());
            in_code_block = !in_code_block;
            if !in_code_block {
                flush(&mut current, &mut rows);
            }
            continue;
        }
        current.push(raw_line.to_string());
    }
    flush(&mut current, &mut rows);
    if rows.is_empty() {
        vec![vec![json!({"tag": "md", "text": content})]]
    } else {
        rows
    }
}

pub fn build_markdown_post_payload(content: &str) -> String {
    let rows = build_markdown_post_rows(content);
    let rows_json: Vec<Value> = rows.into_iter().map(Value::Array).collect();
    json!({ "zh_cn": { "content": rows_json } }).to_string()
}

/// Python `str.splitlines()` equivalent (splits on \n, dropping a trailing
/// empty element, preserving interior empties).
fn splitlines(s: &str) -> Vec<&str> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut parts: Vec<&str> = s.split('\n').collect();
    if let Some(last) = parts.last() {
        if last.is_empty() {
            parts.pop();
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// Post payload parsing
// ---------------------------------------------------------------------------

fn to_post_payload(candidate: &Value) -> Option<(String, Vec<Value>)> {
    let obj = candidate.as_object()?;
    let content = obj.get("content")?.as_array()?;
    let title = match obj.get("title") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    Some((title, content.clone()))
}

fn resolve_locale_payload(payload: &Value) -> Option<(String, Vec<Value>)> {
    if let Some(p) = to_post_payload(payload) {
        return Some(p);
    }
    let obj = payload.as_object()?;
    for key in PREFERRED_LOCALES {
        if let Some(v) = obj.get(*key) {
            if let Some(p) = to_post_payload(v) {
                return Some(p);
            }
        }
    }
    for v in obj.values() {
        if let Some(p) = to_post_payload(v) {
            return Some(p);
        }
    }
    None
}

fn resolve_post_payload(payload: &Value) -> Option<(String, Vec<Value>)> {
    if let Some(p) = to_post_payload(payload) {
        return Some(p);
    }
    let obj = payload.as_object()?;
    if let Some(wrapped) = obj.get("post") {
        if let Some(p) = resolve_locale_payload(wrapped) {
            return Some(p);
        }
    }
    resolve_locale_payload(payload)
}

/// Parse a Feishu `post` payload into text + image keys + media refs.
pub fn parse_feishu_post_payload(
    payload: &Value,
    mentions_map: &mut HashMap<String, FeishuMentionRef>,
) -> FeishuPostParseResult {
    let resolved = match resolve_post_payload(payload) {
        Some(r) => r,
        None => {
            return FeishuPostParseResult {
                text_content: FALLBACK_POST_TEXT.to_string(),
                ..Default::default()
            };
        }
    };
    let (title_raw, content_rows) = resolved;

    let mut image_keys: Vec<String> = Vec::new();
    let mut media_refs: Vec<FeishuPostMediaRef> = Vec::new();
    let mut parts: Vec<String> = Vec::new();

    let title = normalize_feishu_text(title_raw.trim(), None);
    if !title.is_empty() {
        parts.push(title);
    }

    for row in &content_rows {
        let row_items = match row.as_array() {
            Some(a) => a,
            None => continue,
        };
        let mut joined = String::new();
        for item in row_items {
            joined.push_str(&render_post_element(
                item,
                &mut image_keys,
                &mut media_refs,
                Some(mentions_map),
            ));
        }
        let row_text = normalize_feishu_text(&joined, None);
        if !row_text.is_empty() {
            parts.push(row_text);
        }
    }

    let text_content = {
        let joined = parts.join("\n");
        let trimmed = joined.trim();
        if trimmed.is_empty() {
            FALLBACK_POST_TEXT.to_string()
        } else {
            trimmed.to_string()
        }
    };

    FeishuPostParseResult {
        text_content,
        image_keys,
        media_refs,
    }
}

fn render_post_element(
    element: &Value,
    image_keys: &mut Vec<String>,
    media_refs: &mut Vec<FeishuPostMediaRef>,
    mut mentions_map: Option<&mut HashMap<String, FeishuMentionRef>>,
) -> String {
    if let Value::String(s) = element {
        return s.clone();
    }
    let obj = match element.as_object() {
        Some(o) => o,
        None => return String::new(),
    };

    let tag = value_str(obj.get("tag")).trim().to_lowercase();
    match tag.as_str() {
        "text" => return render_text_element(element),
        "a" => {
            let href = value_str(obj.get("href")).trim().to_string();
            let label_raw = match obj.get("text") {
                Some(Value::String(s)) => s.clone(),
                _ => href.clone(),
            };
            let label = label_raw.trim().to_string();
            if label.is_empty() {
                return String::new();
            }
            let escaped = escape_markdown_text(&label);
            return if !href.is_empty() {
                format!("[{escaped}]({href})")
            } else {
                escaped
            };
        }
        "at" => {
            let placeholder = value_str(obj.get("user_id")).trim().to_string();
            if placeholder == "@_all" {
                if let Some(map) = mentions_map.as_deref_mut() {
                    map.entry("@_all".to_string())
                        .or_insert_with(FeishuMentionRef::all);
                }
                return "@all".to_string();
            }
            let display_name = mentions_map
                .as_deref()
                .and_then(|m| m.get(&placeholder))
                .map(|r| {
                    if !r.name.is_empty() {
                        r.name.clone()
                    } else if !r.open_id.is_empty() {
                        r.open_id.clone()
                    } else {
                        "user".to_string()
                    }
                })
                .unwrap_or_else(|| {
                    let un = value_str(obj.get("user_name"));
                    let un = un.trim();
                    if un.is_empty() {
                        "user".to_string()
                    } else {
                        un.to_string()
                    }
                });
            return format!("@{}", escape_markdown_text(&display_name));
        }
        "img" | "image" => {
            let image_key = value_str(obj.get("image_key")).trim().to_string();
            if !image_key.is_empty() && !image_keys.contains(&image_key) {
                image_keys.push(image_key);
            }
            let mut alt = value_str(obj.get("text")).trim().to_string();
            if alt.is_empty() {
                alt = value_str(obj.get("alt")).trim().to_string();
            }
            return if !alt.is_empty() {
                format!("[Image: {alt}]")
            } else {
                "[Image]".to_string()
            };
        }
        "media" | "file" | "audio" | "video" => {
            let file_key = value_str(obj.get("file_key")).trim().to_string();
            let mut file_name = value_str(obj.get("file_name")).trim().to_string();
            if file_name.is_empty() {
                file_name = value_str(obj.get("title")).trim().to_string();
            }
            if file_name.is_empty() {
                file_name = value_str(obj.get("text")).trim().to_string();
            }
            if !file_key.is_empty() {
                let resource_type = if tag == "audio" || tag == "video" {
                    tag.clone()
                } else {
                    "file".to_string()
                };
                media_refs.push(FeishuPostMediaRef {
                    file_key,
                    file_name: file_name.clone(),
                    resource_type,
                });
            }
            return if !file_name.is_empty() {
                format!("[Attachment: {file_name}]")
            } else {
                "[Attachment]".to_string()
            };
        }
        "emotion" | "emoji" => {
            let mut label = value_str(obj.get("text")).trim().to_string();
            if label.is_empty() {
                label = value_str(obj.get("emoji_type")).trim().to_string();
            }
            return if !label.is_empty() {
                format!(":{}:", escape_markdown_text(&label))
            } else {
                "[Emoji]".to_string()
            };
        }
        "br" => return "\n".to_string(),
        "hr" | "divider" => return "\n\n---\n\n".to_string(),
        "code" => {
            let mut code = value_str(obj.get("text"));
            if code.is_empty() {
                code = value_str(obj.get("content"));
            }
            return if !code.is_empty() {
                wrap_inline_code(&code)
            } else {
                String::new()
            };
        }
        "code_block" | "pre" => return render_code_block_element(element),
        _ => {}
    }

    let mut nested_parts: Vec<String> = Vec::new();
    for key in ["text", "title", "content", "children", "elements"] {
        if let Some(val) = obj.get(key) {
            let extracted =
                render_nested_post(val, image_keys, media_refs, mentions_map.as_deref_mut());
            if !extracted.is_empty() {
                nested_parts.push(extracted);
            }
        }
    }
    nested_parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_nested_post(
    value: &Value,
    image_keys: &mut Vec<String>,
    media_refs: &mut Vec<FeishuPostMediaRef>,
    mut mentions_map: Option<&mut HashMap<String, FeishuMentionRef>>,
) -> String {
    match value {
        Value::String(s) => escape_markdown_text(s),
        Value::Array(arr) => arr
            .iter()
            .map(|item| {
                render_nested_post(item, image_keys, media_refs, mentions_map.as_deref_mut())
            })
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        Value::Object(obj) => {
            let direct =
                render_post_element(value, image_keys, media_refs, mentions_map.as_deref_mut());
            if !direct.is_empty() {
                return direct;
            }
            obj.values()
                .map(|item| {
                    render_nested_post(item, image_keys, media_refs, mentions_map.as_deref_mut())
                })
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Message normalization
// ---------------------------------------------------------------------------

/// A mention as supplied in event payloads. Mirrors the duck-typed Feishu
/// mention object (key + id variants + name).
#[derive(Debug, Clone, Default)]
pub struct RawMention {
    pub key: String,
    pub name: String,
    pub open_id: String,
    pub user_id: String,
}

fn load_feishu_payload(raw_content: &str) -> Value {
    if raw_content.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(raw_content) {
        Ok(Value::Object(map)) => Value::Object(map),
        Ok(other) => json!({ "content": other }),
        Err(_) => json!({ "text": raw_content }),
    }
}

fn build_mentions_map(
    mentions: &[RawMention],
    bot: &FeishuBotIdentity,
) -> HashMap<String, FeishuMentionRef> {
    let mut result: HashMap<String, FeishuMentionRef> = HashMap::new();
    for mention in mentions {
        let key = mention.key.clone();
        if key.is_empty() {
            continue;
        }
        if key == "@_all" {
            result.insert(key, FeishuMentionRef::all());
            continue;
        }
        let name = mention.name.trim().to_string();
        let is_self = bot.matches(&mention.open_id, &mention.user_id, &name);
        result.insert(
            key,
            FeishuMentionRef {
                name,
                open_id: mention.open_id.clone(),
                is_all: false,
                is_self,
            },
        );
    }
    result
}

/// Normalize a Feishu inbound message into text + media references + mentions.
pub fn normalize_feishu_message(
    message_type: &str,
    raw_content: &str,
    mentions: &[RawMention],
    bot: &FeishuBotIdentity,
) -> FeishuNormalizedMessage {
    let normalized_type = message_type.trim().to_lowercase();
    let payload = load_feishu_payload(raw_content);
    let mut mentions_map = build_mentions_map(mentions, bot);

    match normalized_type.as_str() {
        "text" => {
            let text = value_str(payload.get("text"));
            if text.contains("@_all") && !mentions_map.contains_key("@_all") {
                mentions_map.insert("@_all".to_string(), FeishuMentionRef::all());
            }
            FeishuNormalizedMessage {
                raw_type: normalized_type.clone(),
                text_content: normalize_feishu_text(&text, Some(&mentions_map)),
                preferred_message_type: "text".to_string(),
                relation_kind: "plain".to_string(),
                mentions: mention_values(&mentions_map),
                ..Default::default()
            }
        }
        "post" => {
            let parsed = parse_feishu_post_payload(&payload, &mut mentions_map);
            FeishuNormalizedMessage {
                raw_type: normalized_type.clone(),
                text_content: parsed.text_content,
                preferred_message_type: "text".to_string(),
                image_keys: parsed.image_keys,
                media_refs: parsed.media_refs,
                mentions: mention_values(&mentions_map),
                relation_kind: "post".to_string(),
                metadata: BTreeMap::new(),
            }
        }
        "image" => {
            let image_key = value_str(payload.get("image_key")).trim().to_string();
            let mut alt_source = value_str(payload.get("text"));
            if alt_source.is_empty() {
                alt_source = value_str(payload.get("alt"));
            }
            if alt_source.is_empty() {
                alt_source = FALLBACK_IMAGE_TEXT.to_string();
            }
            let alt_text = normalize_feishu_text(&alt_source, Some(&mentions_map));
            FeishuNormalizedMessage {
                raw_type: normalized_type.clone(),
                text_content: if alt_text != FALLBACK_IMAGE_TEXT {
                    alt_text
                } else {
                    String::new()
                },
                preferred_message_type: "photo".to_string(),
                image_keys: if image_key.is_empty() {
                    Vec::new()
                } else {
                    vec![image_key]
                },
                relation_kind: "image".to_string(),
                mentions: mention_values(&mentions_map),
                ..Default::default()
            }
        }
        "file" | "audio" | "media" => {
            let media_ref = build_media_ref_from_payload(&payload, &normalized_type);
            let placeholder = attachment_placeholder(&media_ref.file_name);
            let mut metadata = BTreeMap::new();
            metadata.insert("placeholder_text".to_string(), json!(placeholder));
            FeishuNormalizedMessage {
                raw_type: normalized_type.clone(),
                text_content: String::new(),
                preferred_message_type: if normalized_type == "audio" {
                    "audio".to_string()
                } else {
                    "document".to_string()
                },
                media_refs: if media_ref.file_key.is_empty() {
                    Vec::new()
                } else {
                    vec![media_ref]
                },
                relation_kind: normalized_type.clone(),
                metadata,
                mentions: mention_values(&mentions_map),
                ..Default::default()
            }
        }
        "merge_forward" => normalize_merge_forward_message(&payload),
        "share_chat" => normalize_share_chat_message(&payload),
        "interactive" | "card" => normalize_interactive_message(&normalized_type, &payload),
        _ => FeishuNormalizedMessage::empty(&normalized_type),
    }
}

/// Mentions in insertion order is non-deterministic for a HashMap in Python's
/// dict; Python preserves insertion order. We approximate by collecting values.
fn mention_values(map: &HashMap<String, FeishuMentionRef>) -> Vec<FeishuMentionRef> {
    map.values().cloned().collect()
}

fn normalize_merge_forward_message(payload: &Value) -> FeishuNormalizedMessage {
    let title = first_non_empty_text(&[
        payload.get("title"),
        payload.get("summary"),
        payload.get("preview"),
    ])
    .or_else(|| {
        find_first_text(payload, &["title", "summary", "preview", "description"])
            .filter(|s| !s.is_empty())
    })
    .unwrap_or_default();

    let entries = collect_forward_entries(payload);
    let mut lines: Vec<String> = Vec::new();
    if !title.is_empty() {
        lines.push(title.clone());
    }
    for e in entries.iter().take(8) {
        lines.push(e.clone());
    }
    let joined = lines.join("\n");
    let text_content = if joined.trim().is_empty() {
        FALLBACK_FORWARD_TEXT.to_string()
    } else {
        joined.trim().to_string()
    };
    let mut metadata = BTreeMap::new();
    metadata.insert("entry_count".to_string(), json!(entries.len()));
    metadata.insert("title".to_string(), json!(title));
    FeishuNormalizedMessage {
        raw_type: "merge_forward".to_string(),
        text_content,
        preferred_message_type: "text".to_string(),
        relation_kind: "merge_forward".to_string(),
        metadata,
        ..Default::default()
    }
}

fn normalize_share_chat_message(payload: &Value) -> FeishuNormalizedMessage {
    let chat_name = first_non_empty_text(&[
        payload.get("chat_name"),
        payload.get("name"),
        payload.get("title"),
    ])
    .or_else(|| find_first_text(payload, &["chat_name", "name", "title"]).filter(|s| !s.is_empty()))
    .unwrap_or_default();

    let share_id = first_non_empty_text(&[
        payload.get("chat_id"),
        payload.get("open_chat_id"),
        payload.get("share_chat_id"),
    ])
    .unwrap_or_default();

    let mut lines: Vec<String> = Vec::new();
    if !chat_name.is_empty() {
        lines.push(format!("Shared chat: {chat_name}"));
    } else {
        lines.push(FALLBACK_SHARE_CHAT_TEXT.to_string());
    }
    if !share_id.is_empty() {
        lines.push(format!("Chat ID: {share_id}"));
    }
    let mut metadata = BTreeMap::new();
    metadata.insert("chat_id".to_string(), json!(share_id));
    metadata.insert("chat_name".to_string(), json!(chat_name));
    FeishuNormalizedMessage {
        raw_type: "share_chat".to_string(),
        text_content: lines.join("\n"),
        preferred_message_type: "text".to_string(),
        relation_kind: "share_chat".to_string(),
        metadata,
        ..Default::default()
    }
}

fn normalize_interactive_message(message_type: &str, payload: &Value) -> FeishuNormalizedMessage {
    let card_payload = match payload.get("card") {
        Some(c @ Value::Object(_)) => c.clone(),
        _ => payload.clone(),
    };
    let title = first_non_empty_text(&[Some(&json!(find_header_title(&card_payload)))])
        .or_else(|| first_non_empty_text(&[payload.get("title")]))
        .or_else(|| {
            find_first_text(&card_payload, &["title", "summary", "subtitle"])
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_default();

    let body_lines = collect_card_lines(&card_payload);
    let actions = collect_action_labels(&card_payload);

    let mut lines: Vec<String> = Vec::new();
    if !title.is_empty() {
        lines.push(title.clone());
    }
    for line in &body_lines {
        if *line != title {
            lines.push(line.clone());
        }
    }
    if !actions.is_empty() {
        lines.push(format!("Actions: {}", actions.join(", ")));
    }

    let joined: String = lines.iter().take(12).cloned().collect::<Vec<_>>().join("\n");
    let text_content = if joined.trim().is_empty() {
        FALLBACK_INTERACTIVE_TEXT.to_string()
    } else {
        joined.trim().to_string()
    };

    let mut metadata = BTreeMap::new();
    metadata.insert("title".to_string(), json!(title));
    metadata.insert("actions".to_string(), json!(actions));
    FeishuNormalizedMessage {
        raw_type: message_type.to_string(),
        text_content,
        preferred_message_type: "text".to_string(),
        relation_kind: "interactive".to_string(),
        metadata,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Content extraction utilities
// ---------------------------------------------------------------------------

fn collect_forward_entries(payload: &Value) -> Vec<String> {
    let mut candidates: Vec<Value> = Vec::new();
    for key in ["messages", "items", "message_list", "records", "content"] {
        if let Some(Value::Array(arr)) = payload.get(key) {
            candidates.extend(arr.iter().cloned());
        }
    }
    let mut entries: Vec<String> = Vec::new();
    for item in &candidates {
        let obj = match item.as_object() {
            Some(o) => o,
            None => {
                let raw = match item {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                let text = normalize_feishu_text(&raw, None);
                if !text.is_empty() {
                    entries.push(format!("- {text}"));
                }
                continue;
            }
        };
        let sender = first_non_empty_text(&[
            obj.get("sender_name"),
            obj.get("user_name"),
            obj.get("sender"),
            obj.get("name"),
        ])
        .unwrap_or_default();

        let mut nested_type = value_str(obj.get("message_type"));
        if nested_type.is_empty() {
            nested_type = value_str(obj.get("msg_type"));
        }
        let nested_type = nested_type.trim().to_lowercase();

        let body = if nested_type == "post" {
            let inner = obj.get("content").cloned().unwrap_or_else(|| item.clone());
            let mut m: HashMap<String, FeishuMentionRef> = HashMap::new();
            parse_feishu_post_payload(&inner, &mut m).text_content
        } else {
            let raw = first_non_empty_text(&[
                obj.get("text"),
                obj.get("summary"),
                obj.get("preview"),
                obj.get("content"),
            ])
            .or_else(|| {
                find_first_text(item, &["text", "content", "summary", "preview", "title"])
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_default();
            raw
        };
        let body = normalize_feishu_text(&body, None);
        if !sender.is_empty() && !body.is_empty() {
            entries.push(format!("- {sender}: {body}"));
        } else if !body.is_empty() {
            entries.push(format!("- {body}"));
        }
    }
    unique_lines(&entries)
}

fn collect_card_lines(payload: &Value) -> Vec<String> {
    let lines = collect_text_segments(payload, false);
    let normalized: Vec<String> = lines.iter().map(|l| normalize_feishu_text(l, None)).collect();
    unique_lines(&normalized.into_iter().filter(|l| !l.is_empty()).collect::<Vec<_>>())
}

fn collect_action_labels(payload: &Value) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();
    for node in walk_nodes(payload) {
        let obj = match node.as_object() {
            Some(o) => o,
            None => continue,
        };
        let mut tag = value_str(obj.get("tag"));
        if tag.is_empty() {
            tag = value_str(obj.get("type"));
        }
        let tag = tag.trim().to_lowercase();
        if !matches!(
            tag.as_str(),
            "button" | "select_static" | "overflow" | "date_picker" | "picker"
        ) {
            continue;
        }
        let label = first_non_empty_text(&[obj.get("text"), obj.get("name"), obj.get("value")])
            .or_else(|| {
                find_first_text(node, &["text", "content", "name", "value"]).filter(|s| !s.is_empty())
            })
            .unwrap_or_default();
        if !label.is_empty() {
            labels.push(label);
        }
    }
    unique_lines(&labels)
}

fn collect_text_segments(value: &Value, in_rich_block: bool) -> Vec<String> {
    match value {
        Value::String(s) => {
            if in_rich_block {
                vec![normalize_feishu_text(s, None)]
            } else {
                Vec::new()
            }
        }
        Value::Array(arr) => {
            let mut segments = Vec::new();
            for item in arr {
                segments.extend(collect_text_segments(item, in_rich_block));
            }
            segments
        }
        Value::Object(obj) => {
            let mut tag = value_str(obj.get("tag"));
            if tag.is_empty() {
                tag = value_str(obj.get("type"));
            }
            let tag = tag.trim().to_lowercase();
            let next_in_rich = in_rich_block
                || matches!(
                    tag.as_str(),
                    "plain_text"
                        | "lark_md"
                        | "markdown"
                        | "note"
                        | "div"
                        | "column_set"
                        | "column"
                        | "action"
                        | "button"
                        | "select_static"
                        | "date_picker"
                );

            let mut segments: Vec<String> = Vec::new();
            for key in SUPPORTED_CARD_TEXT_KEYS {
                if let Some(Value::String(s)) = obj.get(*key) {
                    if next_in_rich {
                        let normalized = normalize_feishu_text(s, None);
                        if !normalized.is_empty() {
                            segments.push(normalized);
                        }
                    }
                }
            }
            for (key, item) in obj {
                if is_skip_text_key(key) {
                    continue;
                }
                segments.extend(collect_text_segments(item, next_in_rich));
            }
            segments
        }
        _ => Vec::new(),
    }
}

fn build_media_ref_from_payload(payload: &Value, resource_type: &str) -> FeishuPostMediaRef {
    let file_key = value_str(payload.get("file_key")).trim().to_string();
    let file_name = first_non_empty_text(&[
        payload.get("file_name"),
        payload.get("title"),
        payload.get("text"),
    ])
    .unwrap_or_default();
    let effective_type = if resource_type == "audio" || resource_type == "video" {
        resource_type.to_string()
    } else {
        "file".to_string()
    };
    FeishuPostMediaRef {
        file_key,
        file_name,
        resource_type: effective_type,
    }
}

fn attachment_placeholder(file_name: &str) -> String {
    let normalized = normalize_feishu_text(file_name, None);
    if !normalized.is_empty() {
        format!("[Attachment: {normalized}]")
    } else {
        FALLBACK_ATTACHMENT_TEXT.to_string()
    }
}

fn find_header_title(payload: &Value) -> String {
    let obj = match payload.as_object() {
        Some(o) => o,
        None => return String::new(),
    };
    let header = match obj.get("header").and_then(|h| h.as_object()) {
        Some(h) => h,
        None => return String::new(),
    };
    match header.get("title") {
        Some(Value::Object(t)) => first_non_empty_text(&[
            t.get("content"),
            t.get("text"),
            t.get("name"),
        ])
        .unwrap_or_default(),
        Some(other) => normalize_feishu_text(&value_str(Some(other)), None),
        None => String::new(),
    }
}

fn find_first_text(payload: &Value, keys: &[&str]) -> Option<String> {
    for node in walk_nodes(payload) {
        if let Some(obj) = node.as_object() {
            for key in keys {
                if let Some(Value::String(s)) = obj.get(*key) {
                    let normalized = normalize_feishu_text(s, None);
                    if !normalized.is_empty() {
                        return Some(normalized);
                    }
                }
            }
        }
    }
    Some(String::new())
}

/// Depth-first walk yielding every value node (dict-first then children),
/// mirroring `_walk_nodes`.
fn walk_nodes(value: &Value) -> Vec<&Value> {
    let mut out = Vec::new();
    walk_nodes_into(value, &mut out);
    out
}

fn walk_nodes_into<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
    match value {
        Value::Object(map) => {
            out.push(value);
            for v in map.values() {
                walk_nodes_into(v, out);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                walk_nodes_into(v, out);
            }
        }
        _ => {}
    }
}

fn first_non_empty_text(values: &[Option<&Value>]) -> Option<String> {
    for value in values.iter().flatten() {
        match value {
            Value::String(s) => {
                let normalized = normalize_feishu_text(s, None);
                if !normalized.is_empty() {
                    return Some(normalized);
                }
            }
            Value::Object(_) | Value::Array(_) | Value::Null => {}
            other => {
                let normalized = normalize_feishu_text(&other.to_string(), None);
                if !normalized.is_empty() {
                    return Some(normalized);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// General text utilities
// ---------------------------------------------------------------------------

fn is_mention_placeholder_start(bytes: &[u8], i: usize) -> Option<usize> {
    // Matches @_user_\d+ ; returns the end index (exclusive) on success.
    const PREFIX: &[u8] = b"@_user_";
    if i + PREFIX.len() > bytes.len() || &bytes[i..i + PREFIX.len()] != PREFIX {
        return None;
    }
    let mut j = i + PREFIX.len();
    let start_digits = j;
    while j < bytes.len() && bytes[j].is_ascii_digit() {
        j += 1;
    }
    if j > start_digits { Some(j) } else { None }
}

/// Normalize Feishu text: replace `@_user_N` placeholders with `@name`, map
/// `@_all` to `@all`, collapse whitespace, drop blank lines.
pub fn normalize_feishu_text(
    text: &str,
    mentions_map: Option<&HashMap<String, FeishuMentionRef>>,
) -> String {
    // Step 1: substitute @_user_N placeholders.
    let bytes = text.as_bytes();
    let mut substituted = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'@' {
            if let Some(end) = is_mention_placeholder_start(bytes, i) {
                let key = &text[i..end];
                match mentions_map.and_then(|m| m.get(key)) {
                    Some(reference) => {
                        let name = if !reference.name.is_empty() {
                            reference.name.clone()
                        } else if !reference.open_id.is_empty() {
                            reference.open_id.clone()
                        } else {
                            "user".to_string()
                        };
                        substituted.push('@');
                        substituted.push_str(&name);
                    }
                    None => substituted.push(' '),
                }
                i = end;
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap();
        substituted.push(ch);
        i += ch.len_utf8();
    }

    // Step 2: @_all -> @all
    let cleaned = substituted.replace("@_all", "@all");
    // Step 3: normalize line endings
    let cleaned = cleaned.replace("\r\n", "\n").replace('\r', "\n");
    // Step 4: per-line whitespace collapse + trim
    let per_line: Vec<String> = cleaned
        .split('\n')
        .map(|line| collapse_whitespace(line).trim().to_string())
        .collect();
    // Step 5: drop empty lines
    let non_empty: Vec<String> = per_line.into_iter().filter(|l| !l.is_empty()).collect();
    let joined = non_empty.join("\n");
    // Step 6: collapse runs of 2+ spaces/tabs
    let collapsed = collapse_multispace(&joined);
    collapsed.trim().to_string()
}

/// Replace any run of whitespace with a single space (`\s+` -> " ").
fn collapse_whitespace(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut prev_ws = false;
    for ch in line.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

/// Collapse runs of 2+ spaces/tabs to one space (`[ \t]{2,}` -> " ").
fn collapse_multispace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0usize;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            run += 1;
        } else {
            if run == 1 {
                out.push(' ');
            } else if run >= 2 {
                out.push(' ');
            }
            run = 0;
            out.push(ch);
        }
    }
    if run >= 1 {
        out.push(' ');
    }
    out
}

fn unique_lines(lines: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique: Vec<String> = Vec::new();
    for line in lines {
        if line.is_empty() || seen.contains(line) {
            continue;
        }
        seen.insert(line.clone());
        unique.push(line.clone());
    }
    unique
}

// ---------------------------------------------------------------------------
// Mention helpers
// ---------------------------------------------------------------------------

/// Build a `[Mentioned: ...]` hint excluding self-mentions.
pub fn build_mention_hint(mentions: &[FeishuMentionRef]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut seen: HashSet<(bool, String, String)> = HashSet::new();
    for reference in mentions {
        if reference.is_self {
            continue;
        }
        let signature = (
            reference.is_all,
            reference.open_id.clone(),
            reference.name.clone(),
        );
        if seen.contains(&signature) {
            continue;
        }
        seen.insert(signature);
        if reference.is_all {
            parts.push("@all".to_string());
        } else if !reference.open_id.is_empty() {
            let name = if reference.name.is_empty() {
                "unknown"
            } else {
                &reference.name
            };
            parts.push(format!("{name} (open_id={})", reference.open_id));
        } else {
            parts.push(if reference.name.is_empty() {
                "unknown".to_string()
            } else {
                reference.name.clone()
            });
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("[Mentioned: {}]", parts.join(", "))
    }
}

/// Strip leading/trailing self-mentions from text.
pub fn strip_edge_self_mentions(text: &str, mentions: &[FeishuMentionRef]) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let self_names: Vec<String> = mentions
        .iter()
        .filter(|r| r.is_self)
        .map(|r| {
            let name = if !r.name.is_empty() {
                r.name.clone()
            } else if !r.open_id.is_empty() {
                r.open_id.clone()
            } else {
                "user".to_string()
            };
            format!("@{name}")
        })
        .collect();
    if self_names.is_empty() {
        return text.to_string();
    }

    let mut remaining = text.trim_start().to_string();

    // Leading: strip consecutive self-mentions unconditionally (when followed
    // by a boundary char or end of string).
    'outer: loop {
        for nm in &self_names {
            if let Some(after) = remaining.strip_prefix(nm.as_str()) {
                let next_ok = after
                    .chars()
                    .next()
                    .map(|c| MENTION_BOUNDARY_CHARS.contains(c))
                    .unwrap_or(true);
                if next_ok {
                    remaining = after.trim_start().to_string();
                    continue 'outer;
                }
            }
        }
        break;
    }

    // Trailing: strip only when followed by whitespace/terminal punctuation.
    loop {
        // i = end of body before trailing terminal punctuation.
        let chars: Vec<char> = remaining.chars().collect();
        let mut i = chars.len();
        while i > 0 && TRAILING_TERMINAL_PUNCT.contains(chars[i - 1]) {
            i -= 1;
        }
        let body: String = chars[..i].iter().collect();
        let tail: String = chars[i..].iter().collect();
        let mut matched = false;
        for nm in &self_names {
            if body.ends_with(nm.as_str()) {
                let new_body = body[..body.len() - nm.len()].trim_end().to_string();
                remaining = format!("{new_body}{tail}");
                matched = true;
                break;
            }
        }
        if !matched {
            return remaining;
        }
    }
}

// ---------------------------------------------------------------------------
// Admission policy
// ---------------------------------------------------------------------------

/// Sender identity for admission. Carries the populated ID variants.
#[derive(Debug, Clone, Default)]
pub struct SenderIdentity {
    pub open_id: Option<String>,
    pub user_id: Option<String>,
    pub union_id: Option<String>,
}

impl SenderIdentity {
    /// Mirror of `_sender_identity`: the set of non-empty id variants.
    pub fn id_set(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        for v in [&self.open_id, &self.user_id, &self.union_id]
            .into_iter()
            .flatten()
        {
            if !v.is_empty() {
                set.insert(v.clone());
            }
        }
        set
    }
}

/// Snapshot of the adapter's admission configuration. Mirrors the settings
/// fields consulted by `_admit` / `_allow_group_message` / `_require_mention_for`.
#[derive(Debug, Clone, Default)]
pub struct AdmissionConfig {
    pub bot_open_id: String,
    pub bot_user_id: String,
    pub bot_name: String,
    /// "none" | "mentions" | "all"
    pub allow_bots: String,
    pub require_mention: bool,
    pub group_policy: String,
    pub default_group_policy: String,
    pub allowed_group_users: HashSet<String>,
    pub admins: HashSet<String>,
    pub group_rules: HashMap<String, FeishuGroupRule>,
}

impl AdmissionConfig {
    fn self_ids(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        if !self.bot_open_id.is_empty() {
            set.insert(self.bot_open_id.clone());
        }
        if !self.bot_user_id.is_empty() {
            set.insert(self.bot_user_id.clone());
        }
        set
    }

    pub fn bot_identity(&self) -> FeishuBotIdentity {
        FeishuBotIdentity {
            open_id: self.bot_open_id.clone(),
            user_id: self.bot_user_id.clone(),
            name: self.bot_name.clone(),
        }
    }

    pub fn require_mention_for(&self, chat_id: &str) -> bool {
        if !chat_id.is_empty() {
            if let Some(rule) = self.group_rules.get(chat_id) {
                if let Some(rm) = rule.require_mention {
                    return rm;
                }
            }
        }
        self.require_mention
    }
}

/// Inbound message metadata required for admission.
#[derive(Debug, Clone, Default)]
pub struct InboundMessageMeta {
    pub chat_type: String,
    pub chat_id: String,
    pub message_type: String,
    pub raw_content: String,
    pub mentions: Vec<RawMention>,
}

/// Admission decision: returns the reject reason, or `None` to admit.
pub fn admit(
    config: &AdmissionConfig,
    sender: &SenderIdentity,
    sender_type: &str,
    message: &InboundMessageMeta,
) -> Option<RejectReason> {
    let sender_ids = sender.id_set();
    let self_ids = config.self_ids();
    let is_bot = is_bot_sender(sender_type);
    let chat_type = if message.chat_type.is_empty() {
        "p2p".to_string()
    } else {
        message.chat_type.clone()
    };
    let is_group = chat_type != "p2p";
    let chat_id = message.chat_id.clone();
    let require_mention = is_group && config.require_mention_for(&chat_id);

    if !self_ids.is_empty() && !sender_ids.is_disjoint(&self_ids) {
        return Some(RejectReason::SelfEcho);
    }

    if is_bot {
        let mode = config.allow_bots.as_str();
        if mode != "mentions" && mode != "all" {
            return Some(RejectReason::BotsDisabled);
        }
        if self_ids.is_empty() || sender_ids.is_empty() {
            return Some(RejectReason::SelfIdsUnknown);
        }
        if mode == "mentions" && !require_mention && !mentions_self(config, message) {
            return Some(RejectReason::BotNotMentioned);
        }
    }

    if !is_group {
        return None;
    }

    if !allow_group_message(config, sender, &chat_id, is_bot) {
        return Some(RejectReason::GroupPolicyRejected);
    }
    if require_mention && !mentions_self(config, message) {
        return Some(RejectReason::GroupPolicyRejected);
    }
    None
}

/// Per-group policy gate for non-DM traffic.
pub fn allow_group_message(
    config: &AdmissionConfig,
    sender: &SenderIdentity,
    chat_id: &str,
    is_bot: bool,
) -> bool {
    let mut sender_ids: HashSet<String> = HashSet::new();
    if let Some(o) = sender.open_id.as_ref() {
        if !o.is_empty() {
            sender_ids.insert(o.clone());
        }
    }
    if let Some(u) = sender.user_id.as_ref() {
        if !u.is_empty() {
            sender_ids.insert(u.clone());
        }
    }

    if !sender_ids.is_empty()
        && !config.admins.is_empty()
        && !sender_ids.is_disjoint(&config.admins)
    {
        return true;
    }

    let (policy, allowlist, blacklist): (String, HashSet<String>, HashSet<String>) =
        match config.group_rules.get(chat_id) {
            Some(rule) if !chat_id.is_empty() => (
                rule.policy.clone(),
                rule.allowlist.clone(),
                rule.blacklist.clone(),
            ),
            _ => {
                let policy = if !config.default_group_policy.is_empty() {
                    config.default_group_policy.clone()
                } else {
                    config.group_policy.clone()
                };
                (policy, config.allowed_group_users.clone(), HashSet::new())
            }
        };

    if policy == "disabled" {
        return false;
    }
    if policy == "open" {
        return true;
    }
    if policy == "admin_only" {
        return false;
    }
    if is_bot {
        return true;
    }

    if policy == "allowlist" {
        return !sender_ids.is_empty() && !sender_ids.is_disjoint(&allowlist);
    }
    if policy == "blacklist" {
        return !sender_ids.is_empty() && sender_ids.is_disjoint(&blacklist);
    }

    !sender_ids.is_empty() && !sender_ids.is_disjoint(&config.allowed_group_users)
}

/// True if the message @-mentions the bot (`_mentions_self`).
pub fn mentions_self(config: &AdmissionConfig, message: &InboundMessageMeta) -> bool {
    if message.raw_content.contains("@_all") {
        return true;
    }
    if !message.mentions.is_empty() && message_mentions_bot(config, &message.mentions) {
        return true;
    }
    let bot = config.bot_identity();
    let normalized = normalize_feishu_message(
        &message.message_type,
        &message.raw_content,
        &message.mentions,
        &bot,
    );
    normalized.mentions.iter().any(|m| m.is_self)
}

fn message_mentions_bot(config: &AdmissionConfig, mentions: &[RawMention]) -> bool {
    for mention in mentions {
        let open_id = mention.open_id.trim();
        let user_id = mention.user_id.trim();
        let name = mention.name.trim();

        if !open_id.is_empty() && !config.bot_open_id.is_empty() {
            if open_id == config.bot_open_id {
                return true;
            }
            continue;
        }
        if !user_id.is_empty() && !config.bot_user_id.is_empty() {
            if user_id == config.bot_user_id {
                return true;
            }
            continue;
        }
        if !config.bot_name.is_empty() && name == config.bot_name {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Chat-type mapping
// ---------------------------------------------------------------------------

pub fn map_chat_type(raw_chat_type: &str) -> &'static str {
    let normalized = raw_chat_type.trim().to_lowercase();
    if normalized == "p2p" {
        return "dm";
    }
    if normalized.contains("topic") || normalized.contains("thread") || normalized.contains("forum")
    {
        return "forum";
    }
    if normalized == "group" {
        return "group";
    }
    "dm"
}

pub fn resolve_source_chat_type(chat_info_type: &str, event_chat_type: &str) -> &'static str {
    let resolved = chat_info_type.trim().to_lowercase();
    if resolved == "group" {
        return "group";
    }
    if resolved == "forum" {
        return "forum";
    }
    if event_chat_type == "p2p" {
        return "dm";
    }
    "group"
}

// ---------------------------------------------------------------------------
// Extension / media-type guessing
// ---------------------------------------------------------------------------

fn suffix_lower(name: &str) -> String {
    // Mirror of pathlib.Path(name).suffix.lower() — last dot-segment of the
    // final path component, only if it isn't the leading char.
    let base = name.rsplit('/').next().unwrap_or(name);
    if let Some(pos) = base.rfind('.') {
        if pos > 0 {
            return base[pos..].to_lowercase();
        }
    }
    String::new()
}

pub fn default_image_media_type(ext: &str) -> String {
    let normalized = ext.to_lowercase();
    if normalized == ".jpg" || normalized == ".jpeg" {
        return "image/jpeg".to_string();
    }
    let stripped = normalized.trim_start_matches('.');
    let stripped = if stripped.is_empty() { "jpeg" } else { stripped };
    format!("image/{stripped}")
}

pub fn guess_extension(filename: &str, content_type: &str, default: &str, allowed: &[&str]) -> String {
    let ext = suffix_lower(filename);
    if allowed.contains(&ext.as_str()) {
        return ext;
    }
    if let Some(guessed) = guess_extension_from_mime(content_type) {
        if allowed.contains(&guessed.as_str()) {
            return guessed;
        }
    }
    default.to_string()
}

/// Minimal mimetypes.guess_extension covering the types relevant here.
fn guess_extension_from_mime(content_type: &str) -> Option<String> {
    let normalized = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let ext = match normalized.as_str() {
        "image/jpeg" => ".jpg",
        "image/png" => ".png",
        "image/gif" => ".gif",
        "image/webp" => ".webp",
        "image/bmp" => ".bmp",
        "audio/ogg" => ".ogg",
        "audio/mpeg" => ".mp3",
        "audio/wav" | "audio/x-wav" => ".wav",
        "audio/mp4" | "audio/x-m4a" => ".m4a",
        "audio/aac" => ".aac",
        "audio/flac" => ".flac",
        "audio/opus" => ".opus",
        "audio/webm" => ".webm",
        "video/mp4" => ".mp4",
        "video/quicktime" => ".mov",
        "video/webm" => ".webm",
        "application/pdf" => ".pdf",
        "text/plain" => ".txt",
        "text/markdown" => ".md",
        _ => return None,
    };
    Some(ext.to_string())
}

pub fn normalize_media_type(content_type: &str, default: &str) -> String {
    let normalized = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if normalized.is_empty() {
        default.to_string()
    } else {
        normalized
    }
}

/// Best-effort mimetypes.guess_type for filename -> media type.
fn guess_type_from_filename(filename: &str) -> String {
    let ext = suffix_lower(filename);
    match ext.as_str() {
        ".jpg" | ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".ogg" => "audio/ogg",
        ".mp3" => "audio/mpeg",
        ".wav" => "audio/x-wav",
        ".m4a" => "audio/x-m4a",
        ".aac" => "audio/aac",
        ".flac" => "audio/flac",
        ".opus" => "audio/opus",
        ".webm" => "video/webm",
        ".mp4" => "video/mp4",
        ".mov" => "video/quicktime",
        ".avi" => "video/x-msvideo",
        ".mkv" => "video/x-matroska",
        ".pdf" => "application/pdf",
        ".txt" => "text/plain",
        ".md" => "text/markdown",
        _ => "",
    }
    .to_string()
}

pub fn guess_media_type_from_filename(filename: &str) -> String {
    let guessed = guess_type_from_filename(filename).to_lowercase();
    if !guessed.is_empty() {
        return guessed;
    }
    let ext = suffix_lower(filename);
    if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
        return format!("video/{}", ext.trim_start_matches('.'));
    }
    if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        return format!("audio/{}", ext.trim_start_matches('.'));
    }
    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        return default_image_media_type(&ext);
    }
    String::new()
}

/// Mirror of `_display_name_from_cached_path`: take the cache filename, drop
/// the two leading `_`-prefixed tokens, and sanitize.
pub fn display_name_from_cached_path(path: &str) -> String {
    let basename = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();
    // basename.split("_", 2) → keep [2] if 3+ parts
    let parts: Vec<&str> = basename.splitn(3, '_').collect();
    let display = if parts.len() >= 3 {
        parts[2].to_string()
    } else {
        basename.clone()
    };
    // re.sub(r"[^\w.\- ]", "_", ...)
    display
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Outbound payload construction
// ---------------------------------------------------------------------------

/// Decide msg_type + JSON payload for outbound text. Returns (msg_type, payload).
pub fn build_outbound_payload(content: &str) -> (String, String) {
    if has_markdown_table(content) {
        return ("text".to_string(), json!({ "text": content }).to_string());
    }
    if has_markdown_hint(content) {
        return ("post".to_string(), build_markdown_post_payload(content));
    }
    ("text".to_string(), json!({ "text": content }).to_string())
}

/// Build a post payload carrying a caption plus a single media tag element.
pub fn build_media_post_payload(caption: &str, media_tag: Value) -> String {
    let mut payload: Value =
        serde_json::from_str(&build_markdown_post_payload(caption)).unwrap_or_else(|_| json!({}));
    let obj = payload.as_object_mut().unwrap();
    let zh = obj
        .entry("zh_cn".to_string())
        .or_insert_with(|| json!({}));
    let zh_obj = zh.as_object_mut().unwrap();
    let content = zh_obj
        .entry("content".to_string())
        .or_insert_with(|| json!([]));
    if let Some(arr) = content.as_array_mut() {
        arr.push(json!([media_tag]));
    }
    payload.to_string()
}

/// Build the raw approval card JSON sent by `send_exec_approval`.
pub fn build_exec_approval_card(command: &str, description: &str, approval_id: i64) -> Value {
    let cmd_preview = if command.chars().count() > 3000 {
        let truncated: String = command.chars().take(3000).collect();
        format!("{truncated}...")
    } else {
        command.to_string()
    };
    let btn = |label: &str, action_name: &str, btn_type: &str| -> Value {
        json!({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": btn_type,
            "value": {"hermes_action": action_name, "approval_id": approval_id},
        })
    };
    json!({
        "config": {"wide_screen_mode": true},
        "header": {
            "title": {"content": "⚠️ Command Approval Required", "tag": "plain_text"},
            "template": "orange",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": format!("```\n{cmd_preview}\n```\n**Reason:** {description}"),
            },
            {
                "tag": "action",
                "actions": [
                    btn("✅ Allow Once", "approve_once", "primary"),
                    btn("✅ Session", "approve_session", "default"),
                    btn("✅ Always", "approve_always", "default"),
                    btn("❌ Deny", "deny", "danger"),
                ],
            },
        ],
    })
}

/// Build raw card JSON for a resolved approval action.
pub fn build_resolved_approval_card(choice: &str, user_name: &str) -> Value {
    let icon = if choice == "deny" { "❌" } else { "✅" };
    let label = approval_label_map(choice);
    json!({
        "config": {"wide_screen_mode": true},
        "header": {
            "title": {"content": format!("{icon} {label}"), "tag": "plain_text"},
            "template": if choice == "deny" { "red" } else { "green" },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": format!("{icon} **{label}** by {user_name}"),
            },
        ],
    })
}

/// Resolve `(upload_file_type, message_type)` for an outbound file.
pub fn resolve_outbound_file_routing(
    file_path: &str,
    _requested_message_type: &str,
) -> (String, String) {
    let ext = suffix_lower(file_path);
    if FEISHU_OPUS_UPLOAD_EXTENSIONS.contains(&ext.as_str()) {
        return ("opus".to_string(), "audio".to_string());
    }
    if FEISHU_MEDIA_UPLOAD_EXTENSIONS.contains(&ext.as_str()) {
        return ("mp4".to_string(), "media".to_string());
    }
    if let Some(doc_type) = feishu_doc_upload_type(&ext) {
        return (doc_type.to_string(), "file".to_string());
    }
    (FEISHU_FILE_UPLOAD_TYPE.to_string(), "file".to_string())
}

/// Detect whether `chat_id` is a user open_id (DM) or a chat_id (group).
pub fn receive_id_type_for(chat_id: &str) -> &'static str {
    if chat_id.starts_with("ou_") {
        "open_id"
    } else {
        "chat_id"
    }
}

// ---------------------------------------------------------------------------
// Webhook security: rate limit, anomaly tracking, signature verification
// ---------------------------------------------------------------------------

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Verify Feishu webhook signature using timing-safe comparison:
/// `SHA256(timestamp + nonce + encrypt_key + body_string)` hex == signature.
pub fn is_webhook_signature_valid(
    timestamp: &str,
    nonce: &str,
    signature: &str,
    encrypt_key: &str,
    body_bytes: &[u8],
) -> bool {
    if timestamp.is_empty() || nonce.is_empty() || signature.is_empty() {
        return false;
    }
    use sha2::{Digest, Sha256};
    let body_str = String::from_utf8_lossy(body_bytes);
    let content = format!("{timestamp}{nonce}{encrypt_key}{body_str}");
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let computed = hex_encode(&hasher.finalize());
    constant_time_eq(computed.as_bytes(), signature.as_bytes())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Constant-time byte comparison (mirrors `hmac.compare_digest`).
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Sliding-window rate limiter keyed on `app_id:path:remote_ip`.
#[derive(Debug, Default)]
pub struct WebhookRateLimiter {
    counts: HashMap<String, (u32, f64)>,
}

impl WebhookRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check(&mut self, rate_key: &str) -> bool {
        self.check_at(rate_key, now_secs())
    }

    /// Returns false when the key has exceeded the per-window maximum.
    pub fn check_at(&mut self, rate_key: &str, now: f64) -> bool {
        if let Some(&(count, window_start)) = self.counts.get(rate_key) {
            if now - window_start < FEISHU_WEBHOOK_RATE_WINDOW_SECONDS {
                if count >= FEISHU_WEBHOOK_RATE_LIMIT_MAX {
                    return false;
                }
                self.counts
                    .insert(rate_key.to_string(), (count + 1, window_start));
                return true;
            }
        }
        if self.counts.len() >= FEISHU_WEBHOOK_RATE_MAX_KEYS {
            let stale: Vec<String> = self
                .counts
                .iter()
                .filter(|(_, &(_, ws))| now - ws >= FEISHU_WEBHOOK_RATE_WINDOW_SECONDS)
                .map(|(k, _)| k.clone())
                .collect();
            for k in stale {
                self.counts.remove(&k);
            }
            if !self.counts.contains_key(rate_key)
                && self.counts.len() >= FEISHU_WEBHOOK_RATE_MAX_KEYS
            {
                return true;
            }
        }
        self.counts.insert(rate_key.to_string(), (1, now));
        true
    }
}

/// Result of recording a webhook anomaly: optionally a WARNING to log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyOutcome {
    None,
    Warn { count: u32 },
}

/// Tracks consecutive error responses per remote IP.
#[derive(Debug, Default)]
pub struct WebhookAnomalyTracker {
    counts: HashMap<String, (u32, String, f64)>,
}

impl WebhookAnomalyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, remote_ip: &str, status: &str) -> AnomalyOutcome {
        self.record_at(remote_ip, status, now_secs())
    }

    pub fn record_at(&mut self, remote_ip: &str, status: &str, now: f64) -> AnomalyOutcome {
        if let Some(&(count, _, first_seen)) = self.counts.get(remote_ip) {
            if now - first_seen < FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS {
                let count = count + 1;
                let outcome = if count % FEISHU_WEBHOOK_ANOMALY_THRESHOLD == 0 {
                    AnomalyOutcome::Warn { count }
                } else {
                    AnomalyOutcome::None
                };
                self.counts
                    .insert(remote_ip.to_string(), (count, status.to_string(), first_seen));
                return outcome;
            }
        }
        self.counts
            .insert(remote_ip.to_string(), (1, status.to_string(), now));
        AnomalyOutcome::None
    }

    pub fn clear(&mut self, remote_ip: &str) {
        self.counts.remove(remote_ip);
    }
}

/// Card-action token dedup window.
#[derive(Debug, Default)]
pub struct CardActionDedup {
    tokens: HashMap<String, f64>,
}

impl CardActionDedup {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_duplicate(&mut self, token: &str) -> bool {
        self.is_duplicate_at(token, now_secs())
    }

    pub fn is_duplicate_at(&mut self, token: &str, now: f64) -> bool {
        let expired: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, &ts)| now - ts > FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS)
            .map(|(t, _)| t.clone())
            .collect();
        for t in expired {
            self.tokens.remove(&t);
        }
        if self.tokens.contains_key(token) {
            return true;
        }
        self.tokens.insert(token.to_string(), now);
        false
    }
}

// ---------------------------------------------------------------------------
// Persistent dedup cache with TTL + size cap
// ---------------------------------------------------------------------------

/// Seen-message dedup cache (mirrors the `_seen_message_ids` / order pair).
#[derive(Debug)]
pub struct SeenMessageCache {
    ids: HashMap<String, f64>,
    order: VecDeque<String>,
    cache_size: usize,
}

impl SeenMessageCache {
    pub fn new(cache_size: usize) -> Self {
        SeenMessageCache {
            ids: HashMap::new(),
            order: VecDeque::new(),
            cache_size: cache_size.max(1),
        }
    }

    /// Load from a persisted payload `{"message_ids": {id: ts} | [id, ...]}`.
    pub fn load_from_payload(&mut self, payload: &Value, now: f64) {
        let seen_data = payload.get("message_ids");
        let ttl = FEISHU_DEDUP_TTL_SECONDS;

        let mut entries: HashMap<String, f64> = HashMap::new();
        match seen_data {
            Some(Value::Array(arr)) => {
                for item in arr {
                    let key = match item {
                        Value::String(s) => s.trim().to_string(),
                        other => other.to_string().trim().to_string(),
                    };
                    if !key.is_empty() {
                        entries.insert(key, 0.0);
                    }
                }
            }
            Some(Value::Object(map)) => {
                for (key, value) in map {
                    if key.trim().is_empty() {
                        continue;
                    }
                    let ts = match value {
                        Value::Number(n) => n.as_f64(),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    };
                    if let Some(ts) = ts {
                        entries.insert(key.clone(), ts);
                    }
                }
            }
            _ => return,
        }

        let valid: HashMap<String, f64> = entries
            .into_iter()
            .filter(|(_, ts)| *ts == 0.0 || ttl <= 0.0 || now - ts < ttl)
            .collect();

        let mut sorted_ids: Vec<(String, f64)> = valid.into_iter().collect();
        // sort by timestamp descending, then take cache_size.
        sorted_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_ids.truncate(self.cache_size);

        // order = reversed(sorted_ids)
        self.order = sorted_ids
            .iter()
            .rev()
            .map(|(k, _)| k.clone())
            .collect();
        self.ids = sorted_ids.into_iter().collect();
    }

    /// Build the persistence payload (the most-recent ids by order).
    pub fn to_payload(&self) -> Value {
        let recent: Vec<&String> = self
            .order
            .iter()
            .rev()
            .take(self.cache_size)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let mut map = serde_json::Map::new();
        for k in recent {
            if let Some(ts) = self.ids.get(k) {
                map.insert(k.clone(), json!(ts));
            }
        }
        json!({ "message_ids": Value::Object(map) })
    }

    /// Returns true if `message_id` was already seen within the TTL window.
    /// Otherwise records it (evicting oldest beyond the cap).
    pub fn is_duplicate(&mut self, message_id: &str) -> bool {
        self.is_duplicate_at(message_id, now_secs())
    }

    pub fn is_duplicate_at(&mut self, message_id: &str, now: f64) -> bool {
        let ttl = FEISHU_DEDUP_TTL_SECONDS;
        if let Some(&seen_at) = self.ids.get(message_id) {
            if ttl <= 0.0 || now - seen_at < ttl {
                return true;
            }
        }
        self.ids.insert(message_id.to_string(), now);
        self.order.push_back(message_id.to_string());
        while self.order.len() > self.cache_size {
            if let Some(stale) = self.order.pop_front() {
                self.ids.remove(&stale);
            }
        }
        false
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Sender-name TTL cache
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SenderNameCache {
    entries: HashMap<String, (String, f64)>,
}

impl SenderNameCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a cached name only while TTL is valid (empty string = "known
    /// nameless"). Evicts expired entries.
    pub fn get(&mut self, sender_id: &str) -> Option<String> {
        self.get_at(sender_id, now_secs())
    }

    pub fn get_at(&mut self, sender_id: &str, now: f64) -> Option<String> {
        if sender_id.is_empty() {
            return None;
        }
        if let Some((name, expire_at)) = self.entries.get(sender_id) {
            if now < *expire_at {
                return Some(name.clone());
            }
            self.entries.remove(sender_id);
        }
        None
    }

    pub fn put(&mut self, sender_id: &str, name: &str) {
        self.put_at(sender_id, name, now_secs());
    }

    pub fn put_at(&mut self, sender_id: &str, name: &str, now: f64) {
        self.entries.insert(
            sender_id.to_string(),
            (name.to_string(), now + FEISHU_SENDER_NAME_TTL_SECONDS),
        );
    }
}

// ---------------------------------------------------------------------------
// Sender profile resolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SenderProfile {
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub user_id_alt: Option<String>,
}

/// The id used to look up a sender's display name (`name_lookup_id` selection).
pub fn sender_name_lookup_id(sender: &SenderIdentity, is_bot: bool) -> Option<String> {
    let open_id = sender.open_id.clone().filter(|s| !s.is_empty());
    let user_id = sender.user_id.clone().filter(|s| !s.is_empty());
    let union_id = sender.union_id.clone().filter(|s| !s.is_empty());
    let primary = user_id.clone().or_else(|| open_id.clone());
    if is_bot {
        open_id
    } else {
        primary.or(union_id)
    }
}

/// Map Feishu's three-tier IDs onto Hermes' SessionSource fields.
pub fn build_sender_profile(sender: &SenderIdentity, display_name: Option<String>) -> SenderProfile {
    let open_id = sender.open_id.clone().filter(|s| !s.is_empty());
    let user_id = sender.user_id.clone().filter(|s| !s.is_empty());
    let union_id = sender.union_id.clone().filter(|s| !s.is_empty());
    let primary = user_id.or(open_id);
    SenderProfile {
        user_id: primary,
        user_name: display_name,
        user_id_alt: union_id,
    }
}

/// id_type for the contact GetUser request, based on the id prefix.
pub fn contact_user_id_type(id: &str) -> &'static str {
    if id.starts_with("ou_") {
        "open_id"
    } else if id.starts_with("on_") {
        "union_id"
    } else {
        "user_id"
    }
}

// ---------------------------------------------------------------------------
// Resolved-message-type helpers
// ---------------------------------------------------------------------------

pub fn resolve_media_message_type(media_type: &str, default: MessageType) -> MessageType {
    let normalized = media_type.to_lowercase();
    if normalized.starts_with("image/") {
        return MessageType::Photo;
    }
    if normalized.starts_with("audio/") {
        return MessageType::Audio;
    }
    if normalized.starts_with("video/") {
        return MessageType::Video;
    }
    default
}

pub fn resolve_normalized_message_type(
    normalized: &FeishuNormalizedMessage,
    media_types: &[String],
) -> MessageType {
    let first = media_types.first().map(|s| s.as_str()).unwrap_or("");
    match normalized.preferred_message_type.as_str() {
        "photo" => resolve_media_message_type(first, MessageType::Photo),
        "audio" => resolve_media_message_type(first, MessageType::Audio),
        "document" => resolve_media_message_type(first, MessageType::Document),
        _ => MessageType::Text,
    }
}

// ---------------------------------------------------------------------------
// Settings loader
// ---------------------------------------------------------------------------

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Parse the `allow_bots` env into a validated value (defaults to "none").
pub fn parse_allow_bots() -> String {
    let allow_bots = env_str("FEISHU_ALLOW_BOTS", "none").trim().to_lowercase();
    match allow_bots.as_str() {
        "none" | "mentions" | "all" => allow_bots,
        _ => "none".to_string(),
    }
}

/// Parse per-group rules from the platform `extra.group_rules` map.
pub fn parse_group_rules(raw: &Value) -> HashMap<String, FeishuGroupRule> {
    let mut rules = HashMap::new();
    let obj = match raw.as_object() {
        Some(o) => o,
        None => return rules,
    };
    for (chat_id, rule_cfg) in obj {
        let cfg = match rule_cfg.as_object() {
            Some(c) => c,
            None => continue,
        };
        let require_mention = if cfg.contains_key("require_mention") {
            Some(to_boolean(cfg.get("require_mention").unwrap()))
        } else {
            None
        };
        let policy = value_str(cfg.get("policy"));
        let policy = if policy.is_empty() {
            "open".to_string()
        } else {
            policy.trim().to_lowercase()
        };
        let allowlist = collect_id_set(cfg.get("allowlist"));
        let blacklist = collect_id_set(cfg.get("blacklist"));
        rules.insert(
            chat_id.clone(),
            FeishuGroupRule {
                policy,
                allowlist,
                blacklist,
                require_mention,
            },
        );
    }
    rules
}

fn collect_id_set(value: Option<&Value>) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(Value::Array(arr)) = value {
        for item in arr {
            let s = match item {
                Value::String(s) => s.trim().to_string(),
                other => other.to_string().trim().to_string(),
            };
            if !s.is_empty() {
                set.insert(s);
            }
        }
    }
    set
}

/// Build an [`AdmissionConfig`] from platform `extra` config + environment.
/// Mirrors the slice of `_load_settings` / `_apply_settings` relevant to
/// admission gating.
pub fn build_admission_config(extra: &Value) -> AdmissionConfig {
    let group_rules = parse_group_rules(extra.get("group_rules").unwrap_or(&Value::Null));
    let admins = collect_id_set(extra.get("admins"));
    let default_group_policy = value_str(extra.get("default_group_policy"))
        .trim()
        .to_lowercase();
    let allow_bots = parse_allow_bots();

    let bot_open_id = env_str("FEISHU_BOT_OPEN_ID", "").trim().to_string();
    let bot_user_id = env_str("FEISHU_BOT_USER_ID", "").trim().to_string();
    let bot_name = env_str("FEISHU_BOT_NAME", "").trim().to_string();
    let group_policy = env_str("FEISHU_GROUP_POLICY", "allowlist")
        .trim()
        .to_lowercase();
    let allowed_group_users: HashSet<String> = env_str("FEISHU_ALLOWED_USERS", "")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let require_mention = match extra.get("require_mention") {
        Some(v) => to_boolean(v),
        None => to_boolean(&Value::String(env_str("FEISHU_REQUIRE_MENTION", "true"))),
    };

    let default_group_policy = if default_group_policy.is_empty() {
        group_policy.clone()
    } else {
        default_group_policy
    };

    AdmissionConfig {
        bot_open_id,
        bot_user_id,
        bot_name,
        allow_bots,
        require_mention,
        group_policy,
        default_group_policy,
        allowed_group_users,
        admins,
        group_rules,
    }
}

// ---------------------------------------------------------------------------
// Text/media batching helpers
// ---------------------------------------------------------------------------

/// True when an event should be batched as media (`_should_batch_media_event`).
pub fn should_batch_media(message_type: MessageType, has_media: bool) -> bool {
    has_media
        && matches!(
            message_type,
            MessageType::Photo | MessageType::Video | MessageType::Document | MessageType::Audio
        )
}

/// Compatibility tuple for batching merges. Two events merge only when these
/// match exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchContext {
    pub reply_to_message_id: Option<String>,
    pub reply_to_text: Option<String>,
    pub thread_id: Option<String>,
}

pub fn text_batch_compatible(a: &BatchContext, b: &BatchContext) -> bool {
    a == b
}

pub fn media_batch_compatible(
    a: &BatchContext,
    a_type: MessageType,
    b: &BatchContext,
    b_type: MessageType,
) -> bool {
    a_type == b_type && a == b
}

/// Number of characters in a text chunk (used for batch char limits).
pub fn char_count(text: &str) -> usize {
    text.chars().count()
}

// ---------------------------------------------------------------------------
// Onboarding (QR scan-to-create) — request construction + response parsing
// ---------------------------------------------------------------------------

/// Result of the QR registration flow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegistrationResult {
    pub app_id: String,
    pub app_secret: String,
    pub domain: String,
    pub open_id: Option<String>,
    pub bot_name: Option<String>,
    pub bot_open_id: Option<String>,
}

/// `begin` response parsed into the device-code flow parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeginRegistration {
    pub device_code: String,
    pub qr_url: String,
    pub user_code: String,
    pub interval: i64,
    pub expire_in: i64,
}

/// Build the `init` registration form body.
pub fn registration_init_body() -> Vec<(&'static str, &'static str)> {
    vec![("action", "init")]
}

/// Build the `begin` registration form body.
pub fn registration_begin_body() -> Vec<(&'static str, &'static str)> {
    vec![
        ("action", "begin"),
        ("archetype", "PersonalAgent"),
        ("auth_method", "client_secret"),
        ("request_user_info", "open_id"),
    ]
}

/// Build the `poll` registration form body.
pub fn registration_poll_body(device_code: &str) -> Vec<(String, String)> {
    vec![
        ("action".to_string(), "poll".to_string()),
        ("device_code".to_string(), device_code.to_string()),
        ("tp".to_string(), "ob_app".to_string()),
    ]
}

/// POST form-encoded data to the registration endpoint, return parsed JSON.
/// The endpoint returns JSON even on 4xx, so the body is always parsed.
pub fn post_registration(
    base_url: &str,
    body: &[(&str, &str)],
) -> Result<Value, String> {
    post_registration_form(
        base_url,
        &body
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Vec<_>>(),
    )
}

pub fn post_registration_form(base_url: &str, body: &[(String, String)]) -> Result<Value, String> {
    let url = format!("{base_url}{REGISTRATION_PATH}");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(ONBOARD_REQUEST_TIMEOUT_S))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(body)
        .send()
        .map_err(|e| e.to_string())?;
    let text = resp.text().map_err(|e| e.to_string())?;
    serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())
}

/// Verify the `init` response supports client_secret auth.
pub fn check_init_response(res: &Value) -> Result<(), String> {
    let methods = res
        .get("supported_auth_methods")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let supported: Vec<String> = methods
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    if !supported.iter().any(|m| m == "client_secret") {
        return Err(format!(
            "Feishu / Lark registration environment does not support client_secret auth. Supported: {supported:?}"
        ));
    }
    Ok(())
}

/// Parse the `begin` response into device-code parameters.
pub fn parse_begin_response(res: &Value) -> Result<BeginRegistration, String> {
    let device_code = res
        .get("device_code")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "Feishu / Lark registration did not return a device_code".to_string())?
        .to_string();
    let mut qr_url = res
        .get("verification_uri_complete")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if qr_url.contains('?') {
        qr_url.push_str("&from=hermes&tp=hermes");
    } else {
        qr_url.push_str("?from=hermes&tp=hermes");
    }
    let user_code = res
        .get("user_code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let interval = res.get("interval").and_then(|v| v.as_i64()).unwrap_or(5);
    let interval = if interval == 0 { 5 } else { interval };
    let expire_in = res.get("expire_in").and_then(|v| v.as_i64()).unwrap_or(600);
    let expire_in = if expire_in == 0 { 600 } else { expire_in };
    Ok(BeginRegistration {
        device_code,
        qr_url,
        user_code,
        interval,
        expire_in,
    })
}

/// Outcome of a single poll iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollOutcome {
    /// Credentials returned.
    Success(RegistrationResult),
    /// Terminal error (access_denied / expired_token).
    Failure(String),
    /// Still waiting — caller should sleep and poll again.
    Pending { switch_to_lark: bool },
}

/// Interpret a single `poll` response, given the current domain and whether the
/// domain has already been switched to lark.
pub fn interpret_poll_response(
    res: &Value,
    current_domain: &str,
    domain_switched: bool,
) -> PollOutcome {
    let user_info = res.get("user_info").cloned().unwrap_or_else(|| json!({}));
    let tenant_brand = user_info
        .get("tenant_brand")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let switch_to_lark = tenant_brand == "lark" && !domain_switched;
    let effective_domain = if switch_to_lark { "lark" } else { current_domain };

    let client_id = res.get("client_id").and_then(|v| v.as_str());
    let client_secret = res.get("client_secret").and_then(|v| v.as_str());
    if let (Some(client_id), Some(client_secret)) = (client_id, client_secret) {
        if !client_id.is_empty() && !client_secret.is_empty() {
            return PollOutcome::Success(RegistrationResult {
                app_id: client_id.to_string(),
                app_secret: client_secret.to_string(),
                domain: effective_domain.to_string(),
                open_id: user_info
                    .get("open_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                bot_name: None,
                bot_open_id: None,
            });
        }
    }

    let error = res.get("error").and_then(|v| v.as_str()).unwrap_or("");
    if error == "access_denied" || error == "expired_token" {
        return PollOutcome::Failure(error.to_string());
    }

    PollOutcome::Pending { switch_to_lark }
}

/// Parse a `/bot/v3/info` response — accept both `bot.app_name` (new) and
/// `bot.bot_name` (legacy), and nested `data.bot`.
pub fn parse_bot_response(data: &Value) -> Option<(Option<String>, Option<String>)> {
    if data.get("code").and_then(|v| v.as_i64()) != Some(0) {
        return None;
    }
    let bot = data
        .get("bot")
        .cloned()
        .or_else(|| data.get("data").and_then(|d| d.get("bot")).cloned())
        .unwrap_or_else(|| json!({}));
    let bot_name = bot
        .get("app_name")
        .and_then(|v| v.as_str())
        .or_else(|| bot.get("bot_name").and_then(|v| v.as_str()))
        .map(|s| s.to_string());
    let bot_open_id = bot
        .get("open_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Some((bot_name, bot_open_id))
}

/// Probe bot connectivity via raw HTTP `/open-apis/bot/v3/info`.
/// Returns `(bot_name, bot_open_id)` on success.
pub fn probe_bot_http(
    app_id: &str,
    app_secret: &str,
    domain: &str,
) -> Option<(Option<String>, Option<String>)> {
    let base_url = onboard_open_url(domain);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(ONBOARD_REQUEST_TIMEOUT_S))
        .build()
        .ok()?;
    let token_res: Value = client
        .post(format!(
            "{base_url}/open-apis/auth/v3/tenant_access_token/internal"
        ))
        .header("Content-Type", "application/json")
        .json(&json!({"app_id": app_id, "app_secret": app_secret}))
        .send()
        .ok()?
        .json()
        .ok()?;
    let access_token = token_res.get("tenant_access_token").and_then(|v| v.as_str())?;
    if access_token.is_empty() {
        return None;
    }
    let bot_res: Value = client
        .get(format!("{base_url}/open-apis/bot/v3/info"))
        .header("Authorization", format!("Bearer {access_token}"))
        .header("Content-Type", "application/json")
        .send()
        .ok()?
        .json()
        .ok()?;
    parse_bot_response(&bot_res)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_markdown_text() {
        assert_eq!(escape_markdown_text("a*b_c"), "a\\*b\\_c");
        assert_eq!(escape_markdown_text("[link]"), "\\[link\\]");
    }

    #[test]
    fn test_wrap_inline_code() {
        assert_eq!(wrap_inline_code("x"), "`x`");
        assert_eq!(wrap_inline_code("a`b"), "``a`b``");
        assert_eq!(wrap_inline_code("`lead"), "`` `lead ``");
    }

    #[test]
    fn test_to_boolean() {
        assert!(to_boolean(&json!(true)));
        assert!(to_boolean(&json!(1)));
        assert!(to_boolean(&json!("true")));
        assert!(!to_boolean(&json!("false")));
        assert!(!to_boolean(&json!(0)));
        assert!(!to_boolean(&json!(2)));
    }

    #[test]
    fn test_normalize_feishu_text_basic() {
        assert_eq!(normalize_feishu_text("  hello   world  ", None), "hello world");
        assert_eq!(normalize_feishu_text("a\r\nb", None), "a\nb");
        assert_eq!(normalize_feishu_text("@_all hi", None), "@all hi");
    }

    #[test]
    fn test_normalize_feishu_text_mention() {
        let mut map = HashMap::new();
        map.insert(
            "@_user_1".to_string(),
            FeishuMentionRef {
                name: "Alice".to_string(),
                ..Default::default()
            },
        );
        assert_eq!(
            normalize_feishu_text("@_user_1 hello", Some(&map)),
            "@Alice hello"
        );
        // Unknown placeholder collapses to a space.
        assert_eq!(normalize_feishu_text("@_user_9 hi", Some(&map)), "hi");
    }

    #[test]
    fn test_has_markdown_table() {
        let table = "| a | b |\n|---|---|\n| 1 | 2 |";
        assert!(has_markdown_table(table));
        assert!(!has_markdown_table("just text"));
    }

    #[test]
    fn test_has_markdown_hint() {
        assert!(has_markdown_hint("# Heading"));
        assert!(has_markdown_hint("- item"));
        assert!(has_markdown_hint("1. item"));
        assert!(has_markdown_hint("```\ncode\n```"));
        assert!(has_markdown_hint("**bold**"));
        assert!(has_markdown_hint("`code`"));
        assert!(has_markdown_hint("[a](b)"));
        assert!(!has_markdown_hint("plain text here"));
    }

    #[test]
    fn test_build_outbound_payload() {
        let (t, _p) = build_outbound_payload("plain");
        assert_eq!(t, "text");
        let (t, _p) = build_outbound_payload("# heading");
        assert_eq!(t, "post");
        let (t, _p) = build_outbound_payload("| a | b |\n|---|---|");
        assert_eq!(t, "text");
    }

    #[test]
    fn test_build_markdown_post_rows_fences() {
        let rows = build_markdown_post_rows("before\n```py\ncode\n```\nafter");
        // Expect: prose row, code row, prose row.
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_parse_feishu_post_payload() {
        let payload = json!({
            "title": "Title",
            "content": [
                [{"tag": "text", "text": "Hello "}, {"tag": "a", "text": "link", "href": "http://x"}],
                [{"tag": "img", "image_key": "img_1"}],
            ]
        });
        let mut m = HashMap::new();
        let result = parse_feishu_post_payload(&payload, &mut m);
        assert!(result.text_content.contains("Title"));
        assert!(result.text_content.contains("Hello"));
        assert!(result.text_content.contains("[link](http://x)"));
        assert_eq!(result.image_keys, vec!["img_1".to_string()]);
    }

    #[test]
    fn test_normalize_feishu_message_text() {
        let bot = FeishuBotIdentity::default();
        let n = normalize_feishu_message("text", r#"{"text": "hi there"}"#, &[], &bot);
        assert_eq!(n.text_content, "hi there");
        assert_eq!(n.preferred_message_type, "text");
    }

    #[test]
    fn test_normalize_feishu_message_image() {
        let bot = FeishuBotIdentity::default();
        let n = normalize_feishu_message("image", r#"{"image_key": "k1"}"#, &[], &bot);
        assert_eq!(n.preferred_message_type, "photo");
        assert_eq!(n.image_keys, vec!["k1".to_string()]);
    }

    #[test]
    fn test_strip_edge_self_mentions() {
        let mentions = vec![FeishuMentionRef {
            name: "Bot".to_string(),
            is_self: true,
            ..Default::default()
        }];
        assert_eq!(strip_edge_self_mentions("@Bot hello", &mentions), "hello");
        assert_eq!(
            strip_edge_self_mentions("hello @Bot.", &mentions),
            "hello."
        );
        // Mid-sentence reference stays.
        assert_eq!(
            strip_edge_self_mentions("don't @Botagain", &mentions),
            "don't @Botagain"
        );
    }

    #[test]
    fn test_build_mention_hint() {
        let mentions = vec![
            FeishuMentionRef {
                name: "Alice".to_string(),
                open_id: "ou_1".to_string(),
                ..Default::default()
            },
            FeishuMentionRef {
                is_all: true,
                ..Default::default()
            },
            FeishuMentionRef {
                name: "Self".to_string(),
                is_self: true,
                ..Default::default()
            },
        ];
        let hint = build_mention_hint(&mentions);
        assert!(hint.contains("Alice (open_id=ou_1)"));
        assert!(hint.contains("@all"));
        assert!(!hint.contains("Self"));
    }

    #[test]
    fn test_admit_self_echo() {
        let mut config = AdmissionConfig {
            bot_open_id: "ou_bot".to_string(),
            allow_bots: "none".to_string(),
            ..Default::default()
        };
        config.require_mention = true;
        let sender = SenderIdentity {
            open_id: Some("ou_bot".to_string()),
            ..Default::default()
        };
        let msg = InboundMessageMeta {
            chat_type: "p2p".to_string(),
            ..Default::default()
        };
        assert_eq!(admit(&config, &sender, "user", &msg), Some(RejectReason::SelfEcho));
    }

    #[test]
    fn test_admit_dm_pass() {
        let config = AdmissionConfig::default();
        let sender = SenderIdentity {
            open_id: Some("ou_user".to_string()),
            ..Default::default()
        };
        let msg = InboundMessageMeta {
            chat_type: "p2p".to_string(),
            ..Default::default()
        };
        assert_eq!(admit(&config, &sender, "user", &msg), None);
    }

    #[test]
    fn test_admit_group_allowlist() {
        let mut allowed = HashSet::new();
        allowed.insert("ou_ok".to_string());
        let config = AdmissionConfig {
            group_policy: "allowlist".to_string(),
            default_group_policy: "allowlist".to_string(),
            allowed_group_users: allowed,
            require_mention: false,
            ..Default::default()
        };
        let ok_sender = SenderIdentity {
            open_id: Some("ou_ok".to_string()),
            ..Default::default()
        };
        let bad_sender = SenderIdentity {
            open_id: Some("ou_no".to_string()),
            ..Default::default()
        };
        let msg = InboundMessageMeta {
            chat_type: "group".to_string(),
            chat_id: "oc_1".to_string(),
            ..Default::default()
        };
        assert_eq!(admit(&config, &ok_sender, "user", &msg), None);
        assert_eq!(
            admit(&config, &bad_sender, "user", &msg),
            Some(RejectReason::GroupPolicyRejected)
        );
    }

    #[test]
    fn test_admit_bots_disabled() {
        let config = AdmissionConfig {
            allow_bots: "none".to_string(),
            ..Default::default()
        };
        let sender = SenderIdentity {
            open_id: Some("ou_peer".to_string()),
            ..Default::default()
        };
        let msg = InboundMessageMeta {
            chat_type: "p2p".to_string(),
            ..Default::default()
        };
        assert_eq!(
            admit(&config, &sender, "bot", &msg),
            Some(RejectReason::BotsDisabled)
        );
    }

    #[test]
    fn test_dedup_cache() {
        let mut cache = SeenMessageCache::new(3);
        assert!(!cache.is_duplicate_at("a", 100.0));
        assert!(cache.is_duplicate_at("a", 101.0));
        assert!(!cache.is_duplicate_at("b", 102.0));
        assert!(!cache.is_duplicate_at("c", 103.0));
        // Inserting d evicts a.
        assert!(!cache.is_duplicate_at("d", 104.0));
        assert!(!cache.is_duplicate_at("a", 105.0)); // a was evicted
    }

    #[test]
    fn test_dedup_ttl_expiry() {
        let mut cache = SeenMessageCache::new(10);
        assert!(!cache.is_duplicate_at("x", 0.0));
        // After TTL expiry, x is no longer a duplicate.
        assert!(!cache.is_duplicate_at("x", FEISHU_DEDUP_TTL_SECONDS + 1.0));
    }

    #[test]
    fn test_dedup_roundtrip_payload() {
        let mut cache = SeenMessageCache::new(10);
        cache.is_duplicate_at("m1", 100.0);
        cache.is_duplicate_at("m2", 200.0);
        let payload = cache.to_payload();
        let mut restored = SeenMessageCache::new(10);
        restored.load_from_payload(&payload, 250.0);
        assert!(restored.is_duplicate_at("m1", 251.0));
        assert!(restored.is_duplicate_at("m2", 251.0));
    }

    #[test]
    fn test_rate_limiter() {
        let mut rl = WebhookRateLimiter::new();
        let key = "app:path:ip";
        for _ in 0..FEISHU_WEBHOOK_RATE_LIMIT_MAX {
            assert!(rl.check_at(key, 1000.0));
        }
        // Next one in the same window is denied.
        assert!(!rl.check_at(key, 1000.0));
        // New window resets.
        assert!(rl.check_at(key, 1000.0 + FEISHU_WEBHOOK_RATE_WINDOW_SECONDS + 1.0));
    }

    #[test]
    fn test_anomaly_tracker() {
        let mut t = WebhookAnomalyTracker::new();
        let mut warned = false;
        for _ in 0..FEISHU_WEBHOOK_ANOMALY_THRESHOLD {
            if let AnomalyOutcome::Warn { count } = t.record_at("ip", "400", 1.0) {
                assert_eq!(count, FEISHU_WEBHOOK_ANOMALY_THRESHOLD);
                warned = true;
            }
        }
        assert!(warned);
        t.clear("ip");
        assert_eq!(t.record_at("ip", "400", 1.0), AnomalyOutcome::None);
    }

    #[test]
    fn test_card_action_dedup() {
        let mut d = CardActionDedup::new();
        assert!(!d.is_duplicate_at("tok", 1.0));
        assert!(d.is_duplicate_at("tok", 2.0));
        // After TTL, no longer duplicate.
        assert!(!d.is_duplicate_at("tok", 2.0 + FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS + 1.0));
    }

    #[test]
    fn test_signature_verification() {
        use sha2::{Digest, Sha256};
        let timestamp = "1700000000";
        let nonce = "abc123";
        let encrypt_key = "secret";
        let body = b"{\"a\":1}";
        let content = format!(
            "{timestamp}{nonce}{encrypt_key}{}",
            String::from_utf8_lossy(body)
        );
        let mut h = Sha256::new();
        h.update(content.as_bytes());
        let sig = hex_encode(&h.finalize());
        assert!(is_webhook_signature_valid(
            timestamp,
            nonce,
            &sig,
            encrypt_key,
            body
        ));
        assert!(!is_webhook_signature_valid(
            timestamp, nonce, "wrong", encrypt_key, body
        ));
        assert!(!is_webhook_signature_valid("", nonce, &sig, encrypt_key, body));
    }

    #[test]
    fn test_resolve_outbound_file_routing() {
        assert_eq!(
            resolve_outbound_file_routing("voice.opus", "audio"),
            ("opus".to_string(), "audio".to_string())
        );
        assert_eq!(
            resolve_outbound_file_routing("clip.mp4", "media"),
            ("mp4".to_string(), "media".to_string())
        );
        assert_eq!(
            resolve_outbound_file_routing("doc.pdf", "file"),
            ("pdf".to_string(), "file".to_string())
        );
        assert_eq!(
            resolve_outbound_file_routing("data.bin", "file"),
            (FEISHU_FILE_UPLOAD_TYPE.to_string(), "file".to_string())
        );
    }

    #[test]
    fn test_receive_id_type() {
        assert_eq!(receive_id_type_for("ou_user"), "open_id");
        assert_eq!(receive_id_type_for("oc_group"), "chat_id");
    }

    #[test]
    fn test_map_chat_type() {
        assert_eq!(map_chat_type("p2p"), "dm");
        assert_eq!(map_chat_type("group"), "group");
        assert_eq!(map_chat_type("topic_thread"), "forum");
        assert_eq!(map_chat_type("weird"), "dm");
    }

    #[test]
    fn test_resolve_source_chat_type() {
        assert_eq!(resolve_source_chat_type("group", "p2p"), "group");
        assert_eq!(resolve_source_chat_type("", "p2p"), "dm");
        assert_eq!(resolve_source_chat_type("", "group"), "group");
        assert_eq!(resolve_source_chat_type("forum", "group"), "forum");
    }

    #[test]
    fn test_default_image_media_type() {
        assert_eq!(default_image_media_type(".jpg"), "image/jpeg");
        assert_eq!(default_image_media_type(".png"), "image/png");
        assert_eq!(default_image_media_type(""), "image/jpeg");
    }

    #[test]
    fn test_guess_extension() {
        assert_eq!(
            guess_extension("photo.png", "", ".jpg", IMAGE_EXTENSIONS),
            ".png"
        );
        assert_eq!(
            guess_extension("noext", "image/png", ".jpg", IMAGE_EXTENSIONS),
            ".png"
        );
        assert_eq!(
            guess_extension("noext", "application/x", ".jpg", IMAGE_EXTENSIONS),
            ".jpg"
        );
    }

    #[test]
    fn test_display_name_from_cached_path() {
        assert_eq!(
            display_name_from_cached_path("/tmp/123_abc_report.pdf"),
            "report.pdf"
        );
        assert_eq!(display_name_from_cached_path("/tmp/plain.txt"), "plain.txt");
        assert_eq!(
            display_name_from_cached_path("/tmp/1_2_weird@name.pdf"),
            "weird_name.pdf"
        );
    }

    #[test]
    fn test_strip_markdown_to_plain_text() {
        let input = "# Title\n**bold** and `code`\n[link](http://x)\n> quote\n~~strike~~";
        let out = strip_markdown_to_plain_text(input);
        assert!(out.contains("Title"));
        assert!(out.contains("bold"));
        assert!(out.contains("code"));
        assert!(out.contains("link (http://x)"));
        assert!(out.contains("quote"));
        assert!(out.contains("strike"));
        assert!(!out.contains("**"));
        assert!(!out.contains('`'));
    }

    #[test]
    fn test_approval_maps() {
        assert_eq!(approval_choice_map("approve_once"), "once");
        assert_eq!(approval_choice_map("deny"), "deny");
        assert_eq!(approval_choice_map("unknown"), "deny");
        assert_eq!(approval_label_map("session"), "Approved for session");
        assert_eq!(approval_label_map("xx"), "Resolved");
    }

    #[test]
    fn test_build_exec_approval_card() {
        let card = build_exec_approval_card("rm -rf /", "dangerous", 7);
        assert_eq!(card["header"]["template"], "orange");
        let actions = &card["elements"][1]["actions"];
        assert_eq!(actions.as_array().unwrap().len(), 4);
        assert_eq!(actions[0]["value"]["approval_id"], 7);
    }

    #[test]
    fn test_build_resolved_approval_card() {
        let approve = build_resolved_approval_card("once", "Alice");
        assert_eq!(approve["header"]["template"], "green");
        let deny = build_resolved_approval_card("deny", "Bob");
        assert_eq!(deny["header"]["template"], "red");
    }

    #[test]
    fn test_build_media_post_payload() {
        let payload = build_media_post_payload(
            "caption",
            json!({"tag": "img", "image_key": "ik"}),
        );
        let parsed: Value = serde_json::from_str(&payload).unwrap();
        let content = &parsed["zh_cn"]["content"];
        let arr = content.as_array().unwrap();
        let last = &arr[arr.len() - 1];
        assert_eq!(last[0]["image_key"], "ik");
    }

    #[test]
    fn test_parse_begin_response() {
        let res = json!({
            "device_code": "dc1",
            "verification_uri_complete": "https://x/y?a=b",
            "user_code": "UC",
            "interval": 3,
            "expire_in": 120
        });
        let begin = parse_begin_response(&res).unwrap();
        assert_eq!(begin.device_code, "dc1");
        assert!(begin.qr_url.ends_with("&from=hermes&tp=hermes"));
        assert_eq!(begin.interval, 3);
    }

    #[test]
    fn test_parse_begin_response_no_query() {
        let res = json!({"device_code": "dc", "verification_uri_complete": "https://x/y"});
        let begin = parse_begin_response(&res).unwrap();
        assert!(begin.qr_url.ends_with("?from=hermes&tp=hermes"));
    }

    #[test]
    fn test_check_init_response() {
        assert!(check_init_response(&json!({"supported_auth_methods": ["client_secret"]})).is_ok());
        assert!(check_init_response(&json!({"supported_auth_methods": ["oauth"]})).is_err());
    }

    #[test]
    fn test_interpret_poll_response_success() {
        let res = json!({"client_id": "cid", "client_secret": "sec", "user_info": {"open_id": "ou_x"}});
        match interpret_poll_response(&res, "feishu", false) {
            PollOutcome::Success(r) => {
                assert_eq!(r.app_id, "cid");
                assert_eq!(r.app_secret, "sec");
                assert_eq!(r.open_id, Some("ou_x".to_string()));
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn test_interpret_poll_response_lark_switch() {
        let res = json!({"user_info": {"tenant_brand": "lark"}, "error": "authorization_pending"});
        match interpret_poll_response(&res, "feishu", false) {
            PollOutcome::Pending { switch_to_lark } => assert!(switch_to_lark),
            _ => panic!("expected pending"),
        }
    }

    #[test]
    fn test_interpret_poll_response_denied() {
        let res = json!({"error": "access_denied"});
        assert_eq!(
            interpret_poll_response(&res, "feishu", false),
            PollOutcome::Failure("access_denied".to_string())
        );
    }

    #[test]
    fn test_parse_bot_response() {
        let r = json!({"code": 0, "bot": {"app_name": "MyBot", "open_id": "ou_bot"}});
        let (name, oid) = parse_bot_response(&r).unwrap();
        assert_eq!(name, Some("MyBot".to_string()));
        assert_eq!(oid, Some("ou_bot".to_string()));
        // legacy bot_name
        let legacy = json!({"code": 0, "data": {"bot": {"bot_name": "Legacy", "open_id": "ou_l"}}});
        let (name, _) = parse_bot_response(&legacy).unwrap();
        assert_eq!(name, Some("Legacy".to_string()));
        // failure
        assert!(parse_bot_response(&json!({"code": 1})).is_none());
    }

    #[test]
    fn test_collect_forward_entries() {
        let payload = json!({
            "title": "Forwarded",
            "messages": [
                {"sender_name": "Alice", "text": "Hello"},
                {"text": "World"}
            ]
        });
        let n = normalize_feishu_message("merge_forward", &payload.to_string(), &[], &FeishuBotIdentity::default());
        assert!(n.text_content.contains("Forwarded"));
        assert!(n.text_content.contains("- Alice: Hello"));
        assert!(n.text_content.contains("- World"));
    }

    #[test]
    fn test_normalize_interactive() {
        let payload = json!({
            "card": {
                "header": {"title": {"content": "Card Title"}},
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": "Body line"}},
                    {"tag": "action", "actions": [{"tag": "button", "text": "Click"}]}
                ]
            }
        });
        let n = normalize_feishu_message("interactive", &payload.to_string(), &[], &FeishuBotIdentity::default());
        assert!(n.text_content.contains("Card Title"));
        assert!(n.text_content.contains("Actions: Click"));
        assert_eq!(n.relation_kind, "interactive");
    }

    #[test]
    fn test_share_chat() {
        let payload = json!({"chat_name": "Team", "chat_id": "oc_abc"});
        let n = normalize_feishu_message("share_chat", &payload.to_string(), &[], &FeishuBotIdentity::default());
        assert!(n.text_content.contains("Shared chat: Team"));
        assert!(n.text_content.contains("Chat ID: oc_abc"));
    }

    #[test]
    fn test_mentions_self_via_at_all() {
        let config = AdmissionConfig::default();
        let msg = InboundMessageMeta {
            raw_content: "hi @_all".to_string(),
            ..Default::default()
        };
        assert!(mentions_self(&config, &msg));
    }

    #[test]
    fn test_message_mentions_bot_id_match() {
        let config = AdmissionConfig {
            bot_open_id: "ou_bot".to_string(),
            ..Default::default()
        };
        let msg = InboundMessageMeta {
            message_type: "text".to_string(),
            raw_content: "{}".to_string(),
            mentions: vec![RawMention {
                key: "@_user_1".to_string(),
                open_id: "ou_bot".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(mentions_self(&config, &msg));
        // Different open_id does not match.
        let msg2 = InboundMessageMeta {
            message_type: "text".to_string(),
            raw_content: "{}".to_string(),
            mentions: vec![RawMention {
                key: "@_user_1".to_string(),
                open_id: "ou_other".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(!mentions_self(&config, &msg2));
    }

    #[test]
    fn test_should_batch_media() {
        assert!(should_batch_media(MessageType::Photo, true));
        assert!(!should_batch_media(MessageType::Photo, false));
        assert!(!should_batch_media(MessageType::Text, true));
    }

    #[test]
    fn test_resolve_normalized_message_type() {
        let mut n = FeishuNormalizedMessage::empty("image");
        n.preferred_message_type = "photo".to_string();
        assert_eq!(
            resolve_normalized_message_type(&n, &["image/png".to_string()]),
            MessageType::Photo
        );
        n.preferred_message_type = "audio".to_string();
        assert_eq!(
            resolve_normalized_message_type(&n, &["audio/ogg".to_string()]),
            MessageType::Audio
        );
        n.preferred_message_type = "document".to_string();
        // Document with audio content-type resolves to audio.
        assert_eq!(
            resolve_normalized_message_type(&n, &["audio/mp3".to_string()]),
            MessageType::Audio
        );
        n.preferred_message_type = "text".to_string();
        assert_eq!(resolve_normalized_message_type(&n, &[]), MessageType::Text);
    }

    #[test]
    fn test_sender_profile() {
        let sender = SenderIdentity {
            open_id: Some("ou_1".to_string()),
            user_id: Some("u_1".to_string()),
            union_id: Some("on_1".to_string()),
        };
        let profile = build_sender_profile(&sender, Some("Name".to_string()));
        assert_eq!(profile.user_id, Some("u_1".to_string())); // prefers user_id
        assert_eq!(profile.user_id_alt, Some("on_1".to_string()));
        assert_eq!(sender_name_lookup_id(&sender, false), Some("u_1".to_string()));
        assert_eq!(sender_name_lookup_id(&sender, true), Some("ou_1".to_string()));
    }

    #[test]
    fn test_contact_user_id_type() {
        assert_eq!(contact_user_id_type("ou_x"), "open_id");
        assert_eq!(contact_user_id_type("on_x"), "union_id");
        assert_eq!(contact_user_id_type("u_x"), "user_id");
    }

    #[test]
    fn test_parse_group_rules() {
        let extra = json!({
            "oc_1": {"policy": "Blacklist", "blacklist": ["ou_a"], "require_mention": false},
            "oc_2": {"policy": "open"},
            "bad": "notobj"
        });
        let rules = parse_group_rules(&extra);
        assert_eq!(rules.len(), 2);
        let r1 = &rules["oc_1"];
        assert_eq!(r1.policy, "blacklist");
        assert!(r1.blacklist.contains("ou_a"));
        assert_eq!(r1.require_mention, Some(false));
        assert_eq!(rules["oc_2"].require_mention, None);
    }

    #[test]
    fn test_build_admission_config_env() {
        unsafe {
            std::env::set_var("FEISHU_BOT_OPEN_ID", "ou_env_bot");
            std::env::set_var("FEISHU_ALLOWED_USERS", "ou_a, ou_b");
            std::env::set_var("FEISHU_ALLOW_BOTS", "mentions");
        }
        let cfg = build_admission_config(&json!({}));
        assert_eq!(cfg.bot_open_id, "ou_env_bot");
        assert!(cfg.allowed_group_users.contains("ou_a"));
        assert!(cfg.allowed_group_users.contains("ou_b"));
        assert_eq!(cfg.allow_bots, "mentions");
        unsafe {
            std::env::remove_var("FEISHU_BOT_OPEN_ID");
            std::env::remove_var("FEISHU_ALLOWED_USERS");
            std::env::remove_var("FEISHU_ALLOW_BOTS");
        }
    }

    #[test]
    fn test_coerce_int() {
        assert_eq!(coerce_int(&json!(5), None, 0), Some(5));
        assert_eq!(coerce_int(&json!("7"), None, 0), Some(7));
        assert_eq!(coerce_int(&json!(-1), Some(3), 0), Some(3));
        assert_eq!(coerce_int(&json!("notanum"), Some(9), 0), Some(9));
        assert_eq!(coerce_required_int(&json!(null), 30, 0), 30);
    }

    #[test]
    fn test_batch_compatible() {
        let a = BatchContext {
            reply_to_message_id: Some("m1".to_string()),
            reply_to_text: None,
            thread_id: None,
        };
        let b = a.clone();
        let c = BatchContext {
            reply_to_message_id: Some("m2".to_string()),
            ..a.clone()
        };
        assert!(text_batch_compatible(&a, &b));
        assert!(!text_batch_compatible(&a, &c));
        assert!(media_batch_compatible(
            &a,
            MessageType::Photo,
            &b,
            MessageType::Photo
        ));
        assert!(!media_batch_compatible(
            &a,
            MessageType::Photo,
            &b,
            MessageType::Video
        ));
    }
}
