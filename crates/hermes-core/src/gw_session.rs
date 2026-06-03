//! Native Rust port of `gateway/session.py`.
//!
//! Handles:
//! - Session context tracking (where messages come from)
//! - Session storage (conversations persisted to disk)
//! - Reset policy evaluation (when to start fresh)
//! - Dynamic system prompt injection (agent knows its context)
//!
//! Notes on the port:
//! - The upstream config types (`Platform`, `GatewayConfig`, `HomeChannel`,
//!   `SessionResetPolicy`) are not yet ported to native Rust, so minimal local
//!   equivalents are defined here. They mirror the Python field set/semantics
//!   used by this module.
//! - The SQLite `SessionDB` integration is represented by a trait
//!   (`SessionDb`) so callers can plug in a real implementation; when absent the
//!   JSON index + JSONL transcript behaviour is preserved exactly.
//! - `crate::gateway_whatsapp_identity::canonical_whatsapp_identifier` is used
//!   for WhatsApp identity canonicalisation (it requires a `hermes_home` path —
//!   see `SessionStore::whatsapp_home`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{Datelike, Duration, Local, NaiveDateTime, Timelike};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

/// Local-time "now", second-precision naive timestamp (matches Python's
/// `datetime.now()` used throughout this module for comparisons/ISO output).
pub fn now() -> NaiveDateTime {
    Local::now().naive_local()
}

/// ISO-8601 formatting matching Python's `datetime.isoformat()` for naive
/// timestamps (microsecond precision when sub-second != 0, else seconds).
pub fn iso_format(dt: &NaiveDateTime) -> String {
    if dt.and_utc().timestamp_subsec_micros() == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        // Python emits microseconds with 6 digits.
        dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// Parse an ISO timestamp produced by `iso_format` / Python `fromisoformat`.
pub fn parse_iso(s: &str) -> Option<NaiveDateTime> {
    // Try with fractional seconds first, then without.
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
}

// ---------------------------------------------------------------------------
// PII redaction helpers
// ---------------------------------------------------------------------------

/// Deterministic 12-char hex hash of an identifier (sha256, first 12 hex chars).
pub fn hash_id(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let hex = hex_lower(&digest);
    hex[..12].to_string()
}

/// Hash a sender ID to `user_<12hex>`.
pub fn hash_sender_id(value: &str) -> String {
    format!("user_{}", hash_id(value))
}

/// Hash the numeric portion of a chat ID, preserving the platform prefix.
///
/// `telegram:12345` -> `telegram:<hash>`, `12345` -> `<hash>`.
pub fn hash_chat_id(value: &str) -> String {
    if let Some(colon) = value.find(':') {
        if colon > 0 {
            let prefix = &value[..colon];
            return format!("{}:{}", prefix, hash_id(&value[colon + 1..]));
        }
    }
    hash_id(value)
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ---------------------------------------------------------------------------
// Minimal config types (local stand-ins for gateway/config.py)
// ---------------------------------------------------------------------------

/// Supported messaging platforms. Mirrors `gateway.config.Platform`.
///
/// Built-in members are explicit; any other string is treated as a dynamic
/// plugin platform (the Python enum allows this via `_missing_`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Platform {
    Local,
    Telegram,
    Discord,
    Whatsapp,
    Slack,
    Signal,
    Mattermost,
    Matrix,
    Homeassistant,
    Email,
    Sms,
    Dingtalk,
    ApiServer,
    Webhook,
    Feishu,
    Wecom,
    WecomCallback,
    Weixin,
    Bluebubbles,
    Qqbot,
    Yuanbao,
    /// Dynamic plugin platform (lowercased value).
    Plugin(String),
}

impl Platform {
    /// The string `.value` used in serialization and prompt text.
    pub fn value(&self) -> String {
        match self {
            Platform::Local => "local",
            Platform::Telegram => "telegram",
            Platform::Discord => "discord",
            Platform::Whatsapp => "whatsapp",
            Platform::Slack => "slack",
            Platform::Signal => "signal",
            Platform::Mattermost => "mattermost",
            Platform::Matrix => "matrix",
            Platform::Homeassistant => "homeassistant",
            Platform::Email => "email",
            Platform::Sms => "sms",
            Platform::Dingtalk => "dingtalk",
            Platform::ApiServer => "api_server",
            Platform::Webhook => "webhook",
            Platform::Feishu => "feishu",
            Platform::Wecom => "wecom",
            Platform::WecomCallback => "wecom_callback",
            Platform::Weixin => "weixin",
            Platform::Bluebubbles => "bluebubbles",
            Platform::Qqbot => "qqbot",
            Platform::Yuanbao => "yuanbao",
            Platform::Plugin(s) => return s.clone(),
        }
        .to_string()
    }

    /// Parse a platform from its string value. Unknown non-empty strings become
    /// `Plugin(lowercased)`; empty/blank returns `None`.
    pub fn from_value(value: &str) -> Option<Platform> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lowered = trimmed.to_lowercase();
        Some(match lowered.as_str() {
            "local" => Platform::Local,
            "telegram" => Platform::Telegram,
            "discord" => Platform::Discord,
            "whatsapp" => Platform::Whatsapp,
            "slack" => Platform::Slack,
            "signal" => Platform::Signal,
            "mattermost" => Platform::Mattermost,
            "matrix" => Platform::Matrix,
            "homeassistant" => Platform::Homeassistant,
            "email" => Platform::Email,
            "sms" => Platform::Sms,
            "dingtalk" => Platform::Dingtalk,
            "api_server" => Platform::ApiServer,
            "webhook" => Platform::Webhook,
            "feishu" => Platform::Feishu,
            "wecom" => Platform::Wecom,
            "wecom_callback" => Platform::WecomCallback,
            "weixin" => Platform::Weixin,
            "bluebubbles" => Platform::Bluebubbles,
            "qqbot" => Platform::Qqbot,
            "yuanbao" => Platform::Yuanbao,
            _ => Platform::Plugin(lowered),
        })
    }

    /// Title-cased platform name, matching Python's `str.title()` on the value.
    pub fn title(&self) -> String {
        title_case(&self.value())
    }
}

/// Python-like `str.title()`: capitalise the first letter of each
/// alphabetic run, lowercase the rest.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for ch in s.chars() {
        if ch.is_alphabetic() {
            if prev_alpha {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(ch);
            prev_alpha = false;
        }
    }
    out
}

/// Default destination for a platform. Mirrors `gateway.config.HomeChannel`.
#[derive(Debug, Clone, PartialEq)]
pub struct HomeChannel {
    pub platform: Platform,
    pub chat_id: String,
    pub name: String,
    pub thread_id: Option<String>,
}

impl HomeChannel {
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("platform".into(), Value::String(self.platform.value()));
        m.insert("chat_id".into(), Value::String(self.chat_id.clone()));
        m.insert("name".into(), Value::String(self.name.clone()));
        if let Some(tid) = &self.thread_id {
            m.insert("thread_id".into(), Value::String(tid.clone()));
        }
        Value::Object(m)
    }
}

/// Controls when sessions reset. Mirrors `gateway.config.SessionResetPolicy`.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionResetPolicy {
    /// "daily", "idle", "both", or "none".
    pub mode: String,
    pub at_hour: u32,
    pub idle_minutes: i64,
    pub notify: bool,
    pub notify_exclude_platforms: Vec<String>,
}

impl Default for SessionResetPolicy {
    fn default() -> Self {
        SessionResetPolicy {
            mode: "both".into(),
            at_hour: 4,
            idle_minutes: 1440,
            notify: true,
            notify_exclude_platforms: vec!["api_server".into(), "webhook".into()],
        }
    }
}

/// Minimal stand-in for `gateway.config.GatewayConfig`, exposing only the
/// surface this module touches. A real ported config can implement
/// [`GatewayConfigLike`] instead.
pub trait GatewayConfigLike {
    fn group_sessions_per_user(&self) -> bool {
        true
    }
    fn thread_sessions_per_user(&self) -> bool {
        false
    }
    fn get_reset_policy(
        &self,
        platform: Option<&Platform>,
        session_type: &str,
    ) -> SessionResetPolicy;
    fn get_connected_platforms(&self) -> Vec<Platform>;
    fn get_home_channel(&self, platform: &Platform) -> Option<HomeChannel>;
}

// ---------------------------------------------------------------------------
// SessionSource
// ---------------------------------------------------------------------------

/// Describes where a message originated from.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSource {
    pub platform: Platform,
    pub chat_id: String,
    pub chat_name: Option<String>,
    /// "dm", "group", "channel", "thread".
    pub chat_type: String,
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub thread_id: Option<String>,
    pub chat_topic: Option<String>,
    pub user_id_alt: Option<String>,
    pub chat_id_alt: Option<String>,
    pub is_bot: bool,
    pub guild_id: Option<String>,
    pub parent_chat_id: Option<String>,
    pub message_id: Option<String>,
}

impl SessionSource {
    /// Construct with defaults matching the Python dataclass.
    pub fn new(platform: Platform, chat_id: impl Into<String>) -> Self {
        SessionSource {
            platform,
            chat_id: chat_id.into(),
            chat_name: None,
            chat_type: "dm".into(),
            user_id: None,
            user_name: None,
            thread_id: None,
            chat_topic: None,
            user_id_alt: None,
            chat_id_alt: None,
            is_bot: false,
            guild_id: None,
            parent_chat_id: None,
            message_id: None,
        }
    }

    /// Human-readable description of the source.
    pub fn description(&self) -> String {
        if self.platform == Platform::Local {
            return "CLI terminal".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        match self.chat_type.as_str() {
            "dm" => {
                let who = self
                    .user_name
                    .clone()
                    .or_else(|| self.user_id.clone())
                    .unwrap_or_else(|| "user".to_string());
                parts.push(format!("DM with {}", who));
            }
            "group" => {
                let name = self.chat_name.clone().unwrap_or_else(|| self.chat_id.clone());
                parts.push(format!("group: {}", name));
            }
            "channel" => {
                let name = self.chat_name.clone().unwrap_or_else(|| self.chat_id.clone());
                parts.push(format!("channel: {}", name));
            }
            _ => {
                parts.push(self.chat_name.clone().unwrap_or_else(|| self.chat_id.clone()));
            }
        }
        if let Some(tid) = &self.thread_id {
            parts.push(format!("thread: {}", tid));
        }
        parts.join(", ")
    }

    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("platform".into(), Value::String(self.platform.value()));
        m.insert("chat_id".into(), Value::String(self.chat_id.clone()));
        m.insert("chat_name".into(), opt_str(&self.chat_name));
        m.insert("chat_type".into(), Value::String(self.chat_type.clone()));
        m.insert("user_id".into(), opt_str(&self.user_id));
        m.insert("user_name".into(), opt_str(&self.user_name));
        m.insert("thread_id".into(), opt_str(&self.thread_id));
        m.insert("chat_topic".into(), opt_str(&self.chat_topic));
        if let Some(v) = &self.user_id_alt {
            if !v.is_empty() {
                m.insert("user_id_alt".into(), Value::String(v.clone()));
            }
        }
        if let Some(v) = &self.chat_id_alt {
            if !v.is_empty() {
                m.insert("chat_id_alt".into(), Value::String(v.clone()));
            }
        }
        if let Some(v) = &self.guild_id {
            if !v.is_empty() {
                m.insert("guild_id".into(), Value::String(v.clone()));
            }
        }
        if let Some(v) = &self.parent_chat_id {
            if !v.is_empty() {
                m.insert("parent_chat_id".into(), Value::String(v.clone()));
            }
        }
        if let Some(v) = &self.message_id {
            if !v.is_empty() {
                m.insert("message_id".into(), Value::String(v.clone()));
            }
        }
        Value::Object(m)
    }

    /// Build from a JSON map. Returns `None` if the platform value is missing
    /// or invalid (mirrors Python raising on a bad `Platform(...)`).
    pub fn from_dict(data: &Value) -> Option<SessionSource> {
        let obj = data.as_object()?;
        let platform = Platform::from_value(obj.get("platform")?.as_str()?)?;
        let chat_id = json_to_str(obj.get("chat_id")?);
        Some(SessionSource {
            platform,
            chat_id,
            chat_name: opt_from(obj, "chat_name"),
            chat_type: obj
                .get("chat_type")
                .and_then(|v| v.as_str())
                .unwrap_or("dm")
                .to_string(),
            user_id: opt_from(obj, "user_id"),
            user_name: opt_from(obj, "user_name"),
            thread_id: opt_from(obj, "thread_id"),
            chat_topic: opt_from(obj, "chat_topic"),
            user_id_alt: opt_from(obj, "user_id_alt"),
            chat_id_alt: opt_from(obj, "chat_id_alt"),
            is_bot: false,
            guild_id: opt_from(obj, "guild_id"),
            parent_chat_id: opt_from(obj, "parent_chat_id"),
            message_id: opt_from(obj, "message_id"),
        })
    }
}

fn opt_str(v: &Option<String>) -> Value {
    match v {
        Some(s) => Value::String(s.clone()),
        None => Value::Null,
    }
}

fn opt_from(obj: &Map<String, Value>, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => None,
        Some(other) => Some(json_to_str(other)),
    }
}

fn json_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// SessionContext
// ---------------------------------------------------------------------------

/// Full context for a session, used for dynamic system prompt injection.
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub source: SessionSource,
    pub connected_platforms: Vec<Platform>,
    /// Insertion-ordered platform -> home channel mapping (Python dict order).
    pub home_channels: Vec<(Platform, HomeChannel)>,
    pub shared_multi_user_session: bool,
    pub session_key: String,
    pub session_id: String,
    pub created_at: Option<NaiveDateTime>,
    pub updated_at: Option<NaiveDateTime>,
}

impl SessionContext {
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("source".into(), self.source.to_dict());
        m.insert(
            "connected_platforms".into(),
            Value::Array(
                self.connected_platforms
                    .iter()
                    .map(|p| Value::String(p.value()))
                    .collect(),
            ),
        );
        let mut hc = Map::new();
        for (p, home) in &self.home_channels {
            hc.insert(p.value(), home.to_dict());
        }
        m.insert("home_channels".into(), Value::Object(hc));
        m.insert(
            "shared_multi_user_session".into(),
            Value::Bool(self.shared_multi_user_session),
        );
        m.insert("session_key".into(), Value::String(self.session_key.clone()));
        m.insert("session_id".into(), Value::String(self.session_id.clone()));
        m.insert(
            "created_at".into(),
            self.created_at
                .map(|d| Value::String(iso_format(&d)))
                .unwrap_or(Value::Null),
        );
        m.insert(
            "updated_at".into(),
            self.updated_at
                .map(|d| Value::String(iso_format(&d)))
                .unwrap_or(Value::Null),
        );
        Value::Object(m)
    }
}

// ---------------------------------------------------------------------------
// Prompt building
// ---------------------------------------------------------------------------

/// Platforms where user IDs can be safely redacted (no in-message mention
/// system requiring raw IDs). Discord is excluded.
pub fn is_pii_safe_platform(p: &Platform) -> bool {
    matches!(
        p,
        Platform::Whatsapp | Platform::Signal | Platform::Telegram | Platform::Bluebubbles
    )
}

/// Hook controlling whether the agent has Discord tools loaded this session.
///
/// The Python implementation gates on `DISCORD_BOT_TOKEN` plus the enabled
/// toolset (read via `hermes_cli`). Since those config layers are not ported,
/// callers can pass a custom predicate to `build_session_context_prompt`; the
/// default mirrors the safe path: only the token presence is checked, returning
/// `false` (keeping the stale-API disclaimer) otherwise.
pub fn default_discord_tools_loaded() -> bool {
    // Conservative default: without the ported tools_config layer we cannot
    // confirm the toolset is enabled, so keep the disclaimer (matches the
    // Python `except: return False` behaviour when config can't be read).
    false
}

/// Build the dynamic system-prompt section describing the session context.
///
/// `redact_pii` only takes effect when the source platform is PII-safe.
/// `discord_tools_loaded` provides the gate for injecting the Discord IDs block
/// (pass [`default_discord_tools_loaded`] for the conservative default).
/// `display_hermes_home` supplies the path string used in the local-delivery
/// hint (Python imports `hermes_constants.display_hermes_home`).
/// `plugin_pii_safe` is an optional override consulted when the built-in set
/// does not mark the platform PII-safe (mirrors the plugin registry check).
pub fn build_session_context_prompt(
    context: &SessionContext,
    redact_pii: bool,
    discord_tools_loaded: impl Fn() -> bool,
    display_hermes_home: &str,
    plugin_pii_safe: impl Fn(&Platform) -> bool,
) -> String {
    let mut is_pii_safe = is_pii_safe_platform(&context.source.platform);
    if !is_pii_safe && plugin_pii_safe(&context.source.platform) {
        is_pii_safe = true;
    }
    let redact_pii = redact_pii && is_pii_safe;

    let mut lines: Vec<String> = vec!["## Current Session Context".to_string(), String::new()];

    let platform_name = context.source.platform.title();
    if context.source.platform == Platform::Local {
        lines.push(format!(
            "**Source:** {} (the machine running this agent)",
            platform_name
        ));
    } else {
        let src = &context.source;
        let desc = if redact_pii {
            let uname = src.user_name.clone().unwrap_or_else(|| match &src.user_id {
                Some(uid) => hash_sender_id(uid),
                None => "user".to_string(),
            });
            let cname = src
                .chat_name
                .clone()
                .unwrap_or_else(|| hash_chat_id(&src.chat_id));
            match src.chat_type.as_str() {
                "dm" => format!("DM with {}", uname),
                "group" => format!("group: {}", cname),
                "channel" => format!("channel: {}", cname),
                _ => cname,
            }
        } else {
            src.description()
        };
        lines.push(format!("**Source:** {} ({})", platform_name, desc));
    }

    if let Some(topic) = &context.source.chat_topic {
        lines.push(format!("**Channel Topic:** {}", topic));
    }

    if context.shared_multi_user_session {
        let session_label = if context.source.thread_id.is_some() {
            "Multi-user thread"
        } else {
            "Multi-user session"
        };
        lines.push(format!(
            "**Session type:** {} — messages are prefixed with [sender name]. Multiple users may participate.",
            session_label
        ));
    } else if let Some(uname) = &context.source.user_name {
        lines.push(format!("**User:** {}", uname));
    } else if let Some(uid) = &context.source.user_id {
        let uid = if redact_pii {
            hash_sender_id(uid)
        } else {
            uid.clone()
        };
        lines.push(format!("**User ID:** {}", uid));
    }

    // Platform-specific behavioral notes
    match context.source.platform {
        Platform::Slack => {
            lines.push(String::new());
            lines.push(
                "**Platform notes:** You are running inside Slack. You do NOT have access to \
Slack-specific APIs — you cannot search channel history, pin/unpin messages, manage channels, \
or list users. Do not promise to perform these actions. The gateway may inline the current \
message's Slack block/attachment payload when available, but you still cannot call Slack APIs \
yourself."
                    .to_string(),
            );
        }
        Platform::Discord => {
            if discord_tools_loaded() {
                let src = &context.source;
                let mut id_lines: Vec<String> = vec![
                    String::new(),
                    "**Discord IDs (for the `discord` / `discord_admin` tools):**".to_string(),
                ];
                if let Some(gid) = &src.guild_id {
                    id_lines.push(format!("  - Guild: `{}`", gid));
                }
                match (&src.thread_id, &src.parent_chat_id) {
                    (Some(tid), Some(pcid)) => {
                        id_lines.push(format!("  - Parent channel: `{}`", pcid));
                        id_lines.push(format!(
                            "  - Thread: `{}` (use as `channel_id` for fetch_messages etc.)",
                            tid
                        ));
                    }
                    _ => {
                        id_lines.push(format!("  - Channel: `{}`", src.chat_id));
                    }
                }
                if let Some(mid) = &src.message_id {
                    id_lines.push(format!("  - Triggering message: `{}`", mid));
                }
                lines.extend(id_lines);
            } else {
                lines.push(String::new());
                lines.push(
                    "**Platform notes:** You are running inside Discord. You do NOT have access \
to Discord-specific APIs — you cannot search channel history, pin messages, manage roles, or \
list server members. Do not promise to perform these actions. If the user asks, explain that \
you can only read messages sent directly to you and respond."
                        .to_string(),
                );
            }
        }
        Platform::Bluebubbles => {
            lines.push(String::new());
            lines.push(
                "**Platform notes:** You are responding via iMessage. Keep responses short and \
conversational — think texts, not essays. Structure longer replies as separate short thoughts, \
each separated by a blank line (double newline). Each block between blank lines will be \
delivered as its own iMessage bubble, so write accordingly: one idea per bubble, 1–3 sentences \
each. If the user needs a detailed answer, give the short version first and offer to elaborate."
                    .to_string(),
            );
        }
        Platform::Yuanbao => {
            lines.push(String::new());
            lines.push(
                "**Platform notes:** You are running inside Yuanbao. You CAN send private (DM) \
messages via the send_message tool. Use target='yuanbao:direct:<account_id>' for DM and \
target='yuanbao:group:<group_code>' for group chat."
                    .to_string(),
            );
        }
        _ => {}
    }

    // Connected platforms
    let mut platforms_list: Vec<String> = vec!["local (files on this machine)".to_string()];
    for p in &context.connected_platforms {
        if *p != Platform::Local {
            platforms_list.push(format!("{}: Connected ✓", p.value()));
        }
    }
    lines.push(format!(
        "**Connected Platforms:** {}",
        platforms_list.join(", ")
    ));

    // Home channels
    if !context.home_channels.is_empty() {
        lines.push(String::new());
        lines.push("**Home Channels (default destinations):**".to_string());
        for (platform, home) in &context.home_channels {
            let hc_id = if redact_pii {
                hash_chat_id(&home.chat_id)
            } else {
                home.chat_id.clone()
            };
            lines.push(format!(
                "  - {}: {} (ID: {})",
                platform.value(),
                home.name,
                hc_id
            ));
        }
    }

    // Delivery options for scheduled tasks
    lines.push(String::new());
    lines.push("**Delivery options for scheduled tasks:**".to_string());

    if context.source.platform == Platform::Local {
        lines.push("- `\"origin\"` → Local output (saved to files)".to_string());
    } else {
        let origin_label = context.source.chat_name.clone().unwrap_or_else(|| {
            if redact_pii {
                hash_chat_id(&context.source.chat_id)
            } else {
                context.source.chat_id.clone()
            }
        });
        lines.push(format!("- `\"origin\"` → Back to this chat ({})", origin_label));
    }

    lines.push(format!(
        "- `\"local\"` → Save to local files only ({}/cron/output/)",
        display_hermes_home
    ));

    for (platform, home) in &context.home_channels {
        lines.push(format!(
            "- `\"{}\"` → Home channel ({})",
            platform.value(),
            home.name
        ));
    }

    lines.push(String::new());
    lines.push(
        "*For explicit targeting, use `\"platform:chat_id\"` format if the user provides a \
specific chat ID.*"
            .to_string(),
    );

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// SessionEntry
// ---------------------------------------------------------------------------

/// Entry in the session store.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionEntry {
    pub session_key: String,
    pub session_id: String,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub origin: Option<SessionSource>,
    pub display_name: Option<String>,
    pub platform: Option<Platform>,
    pub chat_type: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub total_tokens: i64,
    pub estimated_cost_usd: f64,
    pub cost_status: String,
    pub last_prompt_tokens: i64,
    pub was_auto_reset: bool,
    pub auto_reset_reason: Option<String>,
    pub reset_had_activity: bool,
    pub is_fresh_reset: bool,
    pub expiry_finalized: bool,
    pub suspended: bool,
    pub resume_pending: bool,
    pub resume_reason: Option<String>,
    pub last_resume_marked_at: Option<NaiveDateTime>,
}

impl SessionEntry {
    /// Create a new entry with the dataclass defaults for the unset fields.
    pub fn new(
        session_key: impl Into<String>,
        session_id: impl Into<String>,
        created_at: NaiveDateTime,
        updated_at: NaiveDateTime,
    ) -> Self {
        SessionEntry {
            session_key: session_key.into(),
            session_id: session_id.into(),
            created_at,
            updated_at,
            origin: None,
            display_name: None,
            platform: None,
            chat_type: "dm".into(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            total_tokens: 0,
            estimated_cost_usd: 0.0,
            cost_status: "unknown".into(),
            last_prompt_tokens: 0,
            was_auto_reset: false,
            auto_reset_reason: None,
            reset_had_activity: false,
            is_fresh_reset: false,
            expiry_finalized: false,
            suspended: false,
            resume_pending: false,
            resume_reason: None,
            last_resume_marked_at: None,
        }
    }

    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("session_key".into(), Value::String(self.session_key.clone()));
        m.insert("session_id".into(), Value::String(self.session_id.clone()));
        m.insert("created_at".into(), Value::String(iso_format(&self.created_at)));
        m.insert("updated_at".into(), Value::String(iso_format(&self.updated_at)));
        m.insert("display_name".into(), opt_str(&self.display_name));
        m.insert(
            "platform".into(),
            self.platform
                .as_ref()
                .map(|p| Value::String(p.value()))
                .unwrap_or(Value::Null),
        );
        m.insert("chat_type".into(), Value::String(self.chat_type.clone()));
        m.insert("input_tokens".into(), Value::from(self.input_tokens));
        m.insert("output_tokens".into(), Value::from(self.output_tokens));
        m.insert("cache_read_tokens".into(), Value::from(self.cache_read_tokens));
        m.insert("cache_write_tokens".into(), Value::from(self.cache_write_tokens));
        m.insert("total_tokens".into(), Value::from(self.total_tokens));
        m.insert("last_prompt_tokens".into(), Value::from(self.last_prompt_tokens));
        m.insert(
            "estimated_cost_usd".into(),
            Value::from(self.estimated_cost_usd),
        );
        m.insert("cost_status".into(), Value::String(self.cost_status.clone()));
        m.insert("expiry_finalized".into(), Value::Bool(self.expiry_finalized));
        m.insert("suspended".into(), Value::Bool(self.suspended));
        m.insert("resume_pending".into(), Value::Bool(self.resume_pending));
        m.insert("resume_reason".into(), opt_str(&self.resume_reason));
        m.insert(
            "last_resume_marked_at".into(),
            self.last_resume_marked_at
                .map(|d| Value::String(iso_format(&d)))
                .unwrap_or(Value::Null),
        );
        m.insert("is_fresh_reset".into(), Value::Bool(self.is_fresh_reset));
        if let Some(origin) = &self.origin {
            m.insert("origin".into(), origin.to_dict());
        }
        Value::Object(m)
    }

    /// Build from a JSON map. Returns `None` if a required key is missing or a
    /// timestamp is unparsable (mirrors the Python `(ValueError, KeyError)`
    /// skip in `_ensure_loaded`).
    pub fn from_dict(data: &Value) -> Option<SessionEntry> {
        let obj = data.as_object()?;

        let origin = match obj.get("origin") {
            Some(o) if !o.is_null() => SessionSource::from_dict(o),
            _ => None,
        };

        // Python: invalid platform value is logged & ignored (platform stays
        // None), it does NOT skip the whole entry.
        let platform = match obj.get("platform") {
            Some(Value::String(s)) if !s.is_empty() => Platform::from_value(s),
            _ => None,
        };

        let last_resume_marked_at = obj
            .get("last_resume_marked_at")
            .and_then(|v| v.as_str())
            .and_then(parse_iso);

        let session_key = obj.get("session_key")?.as_str()?.to_string();
        let session_id = obj.get("session_id")?.as_str()?.to_string();
        let created_at = parse_iso(obj.get("created_at")?.as_str()?)?;
        let updated_at = parse_iso(obj.get("updated_at")?.as_str()?)?;

        let get_i = |k: &str| obj.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
        let get_f = |k: &str, d: f64| obj.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let get_b = |k: &str| obj.get(k).and_then(|v| v.as_bool()).unwrap_or(false);

        // expiry_finalized falls back to legacy memory_flushed key.
        let expiry_finalized = match obj.get("expiry_finalized").and_then(|v| v.as_bool()) {
            Some(b) => b,
            None => obj
                .get("memory_flushed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };

        Some(SessionEntry {
            session_key,
            session_id,
            created_at,
            updated_at,
            origin,
            display_name: opt_from(obj, "display_name"),
            platform,
            chat_type: obj
                .get("chat_type")
                .and_then(|v| v.as_str())
                .unwrap_or("dm")
                .to_string(),
            input_tokens: get_i("input_tokens"),
            output_tokens: get_i("output_tokens"),
            cache_read_tokens: get_i("cache_read_tokens"),
            cache_write_tokens: get_i("cache_write_tokens"),
            total_tokens: get_i("total_tokens"),
            last_prompt_tokens: get_i("last_prompt_tokens"),
            estimated_cost_usd: get_f("estimated_cost_usd", 0.0),
            cost_status: obj
                .get("cost_status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            was_auto_reset: false,
            auto_reset_reason: None,
            reset_had_activity: false,
            is_fresh_reset: get_b("is_fresh_reset"),
            expiry_finalized,
            suspended: get_b("suspended"),
            resume_pending: get_b("resume_pending"),
            resume_reason: opt_from(obj, "resume_reason"),
            last_resume_marked_at,
        })
    }
}

// ---------------------------------------------------------------------------
// Session key construction
// ---------------------------------------------------------------------------

/// Return True when a non-DM session is shared across participants.
pub fn is_shared_multi_user_session(
    source: &SessionSource,
    group_sessions_per_user: bool,
    thread_sessions_per_user: bool,
) -> bool {
    if source.chat_type == "dm" {
        return false;
    }
    if source.thread_id.is_some() {
        return !thread_sessions_per_user;
    }
    !group_sessions_per_user
}

/// Build a deterministic session key from a message source.
///
/// `whatsapp_canonical` is consulted to canonicalise WhatsApp identifiers;
/// pass a closure wrapping
/// `crate::gateway_whatsapp_identity::canonical_whatsapp_identifier` bound to the
/// hermes home path. For non-WhatsApp sources it is never called.
pub fn build_session_key(
    source: &SessionSource,
    group_sessions_per_user: bool,
    thread_sessions_per_user: bool,
    whatsapp_canonical: impl Fn(&str) -> String,
) -> String {
    let platform = source.platform.value();

    if source.chat_type == "dm" {
        let mut dm_chat_id = source.chat_id.clone();
        if source.platform == Platform::Whatsapp {
            dm_chat_id = whatsapp_canonical(&source.chat_id);
        }
        if !dm_chat_id.is_empty() {
            if let Some(tid) = &source.thread_id {
                return format!("agent:main:{}:dm:{}:{}", platform, dm_chat_id, tid);
            }
            return format!("agent:main:{}:dm:{}", platform, dm_chat_id);
        }
        if let Some(tid) = &source.thread_id {
            return format!("agent:main:{}:dm:{}", platform, tid);
        }
        return format!("agent:main:{}:dm", platform);
    }

    let mut participant_id = source
        .user_id_alt
        .clone()
        .or_else(|| source.user_id.clone());
    if let Some(pid) = &participant_id {
        if source.platform == Platform::Whatsapp {
            let canon = whatsapp_canonical(pid);
            // Python: `canonical(...) or participant_id` — keep original if blank.
            if !canon.is_empty() {
                participant_id = Some(canon);
            }
        }
    }

    let mut key_parts: Vec<String> = vec!["agent:main".to_string(), platform, source.chat_type.clone()];

    if !source.chat_id.is_empty() {
        key_parts.push(source.chat_id.clone());
    }
    if let Some(tid) = &source.thread_id {
        key_parts.push(tid.clone());
    }

    let mut isolate_user = group_sessions_per_user;
    if source.thread_id.is_some() && !thread_sessions_per_user {
        isolate_user = false;
    }

    if isolate_user {
        if let Some(pid) = &participant_id {
            key_parts.push(pid.clone());
        }
    }

    key_parts.join(":")
}

// ---------------------------------------------------------------------------
// SessionDb trait (SQLite stand-in)
// ---------------------------------------------------------------------------

/// Pluggable persistence backend matching the `hermes_state.SessionDB`
/// methods this module invokes. All methods are best-effort; failures are
/// logged and swallowed (matching the Python `except` arms).
pub trait SessionDb: Send {
    fn session_count(&self) -> Result<i64, String>;
    fn create_session(
        &self,
        session_id: &str,
        source: &str,
        user_id: Option<&str>,
    ) -> Result<(), String>;
    fn end_session(&self, session_id: &str, reason: &str) -> Result<(), String>;
    fn reopen_session(&self, session_id: &str) -> Result<(), String>;
    fn append_message(&self, session_id: &str, message: &Value) -> Result<(), String>;
    fn replace_messages(&self, session_id: &str, messages: &[Value]) -> Result<(), String>;
    fn get_messages_as_conversation(&self, session_id: &str) -> Result<Vec<Value>, String>;
}

/// Callback type for "is this session backed by an active background process?".
pub type HasActiveProcessesFn = Box<dyn Fn(&str) -> bool + Send>;

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

/// Manages session storage and retrieval.
///
/// Mirrors `gateway.session.SessionStore`. SQLite (`SessionDB`) integration is
/// optional via the [`SessionDb`] trait; when absent the JSON index + JSONL
/// transcript behaviour is preserved.
pub struct SessionStore<C: GatewayConfigLike> {
    pub sessions_dir: PathBuf,
    pub config: C,
    inner: Mutex<Inner>,
    has_active_processes_fn: Option<HasActiveProcessesFn>,
    db: Option<Box<dyn SessionDb>>,
    /// Hermes home path used to canonicalise WhatsApp identities. Defaults to
    /// the user's `~/.hermes` when not set.
    whatsapp_home: PathBuf,
}

struct Inner {
    entries: BTreeMap<String, SessionEntry>,
    /// Insertion order of keys (Python dict preserves insertion order; the
    /// session index JSON must keep it for stable diffs).
    order: Vec<String>,
    loaded: bool,
}

impl<C: GatewayConfigLike> SessionStore<C> {
    pub fn new(sessions_dir: PathBuf, config: C) -> Self {
        let whatsapp_home = dirs::home_dir()
            .map(|h| h.join(".hermes"))
            .unwrap_or_else(|| PathBuf::from(".hermes"));
        SessionStore {
            sessions_dir,
            config,
            inner: Mutex::new(Inner {
                entries: BTreeMap::new(),
                order: Vec::new(),
                loaded: false,
            }),
            has_active_processes_fn: None,
            db: None,
            whatsapp_home,
        }
    }

    pub fn with_has_active_processes_fn(mut self, f: HasActiveProcessesFn) -> Self {
        self.has_active_processes_fn = Some(f);
        self
    }

    pub fn with_db(mut self, db: Box<dyn SessionDb>) -> Self {
        self.db = Some(db);
        self
    }

    pub fn with_whatsapp_home(mut self, home: PathBuf) -> Self {
        self.whatsapp_home = home;
        self
    }

    fn whatsapp_canonical(&self, value: &str) -> String {
        crate::gateway_whatsapp_identity::canonical_whatsapp_identifier(&self.whatsapp_home, value)
    }

    fn has_active(&self, session_key: &str) -> bool {
        match &self.has_active_processes_fn {
            Some(f) => f(session_key),
            None => false,
        }
    }

    fn ensure_loaded_locked(&self, inner: &mut Inner) {
        if inner.loaded {
            return;
        }
        let _ = fs::create_dir_all(&self.sessions_dir);
        let sessions_file = self.sessions_dir.join("sessions.json");
        if sessions_file.exists() {
            match fs::read_to_string(&sessions_file) {
                Ok(text) => match serde_json::from_str::<Value>(&text) {
                    Ok(Value::Object(map)) => {
                        for (key, entry_data) in map.iter() {
                            if let Some(entry) = SessionEntry::from_dict(entry_data) {
                                if !inner.entries.contains_key(key) {
                                    inner.order.push(key.clone());
                                }
                                inner.entries.insert(key.clone(), entry);
                            }
                            // else: skip entries with unknown/removed values
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[gateway] Warning: Failed to load sessions: {}", e);
                    }
                },
                Err(e) => {
                    eprintln!("[gateway] Warning: Failed to load sessions: {}", e);
                }
            }
        }
        inner.loaded = true;
    }

    fn save_locked(&self, inner: &Inner) {
        let _ = fs::create_dir_all(&self.sessions_dir);
        let sessions_file = self.sessions_dir.join("sessions.json");

        let mut map = Map::new();
        for key in &inner.order {
            if let Some(entry) = inner.entries.get(key) {
                map.insert(key.clone(), entry.to_dict());
            }
        }
        let data = Value::Object(map);
        let serialized = match serde_json::to_string_pretty(&data) {
            Ok(s) => s,
            Err(e) => {
                log::debug!("Failed to serialize sessions: {}", e);
                return;
            }
        };

        // Atomic write: temp file in the same dir + rename (mirrors
        // tempfile.mkstemp + atomic_replace).
        let tmp_path = self
            .sessions_dir
            .join(format!(".sessions_{}.tmp", std::process::id()));
        let write_result = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(serialized.as_bytes())?;
            f.flush()?;
            f.sync_all()?;
            // Resolve symlinks like atomic_replace does.
            let real_target = fs::read_link(&sessions_file)
                .map(|_| fs::canonicalize(&sessions_file).unwrap_or_else(|_| sessions_file.clone()))
                .unwrap_or_else(|_| sessions_file.clone());
            fs::rename(&tmp_path, &real_target)?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            log::debug!("Could not save sessions: {}", e);
        }
    }

    fn generate_session_key(&self, source: &SessionSource) -> String {
        build_session_key(
            source,
            self.config.group_sessions_per_user(),
            self.config.thread_sessions_per_user(),
            |v| self.whatsapp_canonical(v),
        )
    }

    /// Check if a session has expired based on its reset policy (entry-only).
    pub fn is_session_expired(&self, entry: &SessionEntry) -> bool {
        if self.has_active(&entry.session_key) {
            return false;
        }
        let policy = self
            .config
            .get_reset_policy(entry.platform.as_ref(), &entry.chat_type);
        Self::expired_for_policy(&policy, &entry.updated_at)
    }

    fn expired_for_policy(policy: &SessionResetPolicy, updated_at: &NaiveDateTime) -> bool {
        if policy.mode == "none" {
            return false;
        }
        let now = now();
        if policy.mode == "idle" || policy.mode == "both" {
            let idle_deadline = *updated_at + Duration::minutes(policy.idle_minutes);
            if now > idle_deadline {
                return true;
            }
        }
        if policy.mode == "daily" || policy.mode == "both" {
            let mut today_reset = now
                .with_hour(policy.at_hour)
                .and_then(|d| d.with_minute(0))
                .and_then(|d| d.with_second(0))
                .and_then(|d| d.with_nanosecond(0))
                .unwrap_or(now);
            if now.hour() < policy.at_hour {
                today_reset -= Duration::days(1);
            }
            if *updated_at < today_reset {
                return true;
            }
        }
        false
    }

    fn should_reset(&self, entry: &SessionEntry, source: &SessionSource) -> Option<String> {
        if self.has_active_processes_fn.is_some() {
            let session_key = self.generate_session_key(source);
            if self.has_active(&session_key) {
                return None;
            }
        }
        let policy = self
            .config
            .get_reset_policy(Some(&source.platform), &source.chat_type);

        if policy.mode == "none" {
            return None;
        }
        let now = now();
        if policy.mode == "idle" || policy.mode == "both" {
            let idle_deadline = entry.updated_at + Duration::minutes(policy.idle_minutes);
            if now > idle_deadline {
                return Some("idle".to_string());
            }
        }
        if policy.mode == "daily" || policy.mode == "both" {
            let mut today_reset = now
                .with_hour(policy.at_hour)
                .and_then(|d| d.with_minute(0))
                .and_then(|d| d.with_second(0))
                .and_then(|d| d.with_nanosecond(0))
                .unwrap_or(now);
            if now.hour() < policy.at_hour {
                today_reset -= Duration::days(1);
            }
            if entry.updated_at < today_reset {
                return Some("daily".to_string());
            }
        }
        None
    }

    /// Check if any sessions have ever been created (across all platforms).
    pub fn has_any_sessions(&self) -> bool {
        if let Some(db) = &self.db {
            if let Ok(count) = db.session_count() {
                return count > 1;
            }
        }
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        inner.entries.len() > 1
    }

    /// Get an existing session or create a new one, evaluating reset policy.
    pub fn get_or_create_session(
        &self,
        source: &SessionSource,
        force_new: bool,
    ) -> SessionEntry {
        let session_key = self.generate_session_key(source);
        let now_ts = now();

        let mut db_end_session_id: Option<String> = None;
        let mut db_create_kwargs: Option<(String, String, Option<String>)> = None;

        let result_entry: SessionEntry;

        {
            let mut inner = self.inner.lock().unwrap();
            self.ensure_loaded_locked(&mut inner);

            let mut was_auto_reset = false;
            let mut auto_reset_reason: Option<String> = None;
            let mut reset_had_activity = false;

            if inner.entries.contains_key(&session_key) && !force_new {
                // Decide reset reason without holding a mutable borrow conflict.
                let (suspended, resume_pending) = {
                    let e = inner.entries.get(&session_key).unwrap();
                    (e.suspended, e.resume_pending)
                };

                let reset_reason: Option<String> = if suspended {
                    Some("suspended".to_string())
                } else if resume_pending {
                    // Restart-interrupted session: preserve session_id, return.
                    let entry = inner.entries.get_mut(&session_key).unwrap();
                    entry.updated_at = now_ts;
                    let cloned = entry.clone();
                    self.save_locked(&inner);
                    return cloned;
                } else {
                    let entry = inner.entries.get(&session_key).unwrap().clone();
                    self.should_reset(&entry, source)
                };

                match reset_reason {
                    None => {
                        let entry = inner.entries.get_mut(&session_key).unwrap();
                        entry.updated_at = now_ts;
                        let cloned = entry.clone();
                        self.save_locked(&inner);
                        return cloned;
                    }
                    Some(reason) => {
                        was_auto_reset = true;
                        auto_reset_reason = Some(reason);
                        let entry = inner.entries.get(&session_key).unwrap();
                        reset_had_activity = entry.total_tokens > 0;
                        db_end_session_id = Some(entry.session_id.clone());
                    }
                }
            }

            // Create new session
            let session_id = format!(
                "{}_{}",
                now_ts.format("%Y%m%d_%H%M%S"),
                uuid_hex8()
            );

            let mut entry = SessionEntry::new(
                session_key.clone(),
                session_id.clone(),
                now_ts,
                now_ts,
            );
            entry.origin = Some(source.clone());
            entry.display_name = source.chat_name.clone();
            entry.platform = Some(source.platform.clone());
            entry.chat_type = source.chat_type.clone();
            entry.was_auto_reset = was_auto_reset;
            entry.auto_reset_reason = auto_reset_reason;
            entry.reset_had_activity = reset_had_activity;

            if !inner.entries.contains_key(&session_key) {
                inner.order.push(session_key.clone());
            }
            inner.entries.insert(session_key.clone(), entry.clone());
            self.save_locked(&inner);
            db_create_kwargs = Some((
                session_id,
                source.platform.value(),
                source.user_id.clone(),
            ));
            result_entry = entry;
        }

        // SQLite operations outside the lock
        if let (Some(db), Some(end_id)) = (&self.db, &db_end_session_id) {
            if let Err(e) = db.end_session(end_id, "session_reset") {
                log::debug!("Session DB operation failed: {}", e);
            }
        }
        if let (Some(db), Some((sid, src, uid))) = (&self.db, &db_create_kwargs) {
            if let Err(e) = db.create_session(sid, src, uid.as_deref()) {
                eprintln!("[gateway] Warning: Failed to create SQLite session: {}", e);
            }
        }

        result_entry
    }

    /// Update lightweight session metadata after an interaction.
    pub fn update_session(&self, session_key: &str, last_prompt_tokens: Option<i64>) {
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        if let Some(entry) = inner.entries.get_mut(session_key) {
            entry.updated_at = now();
            if let Some(lpt) = last_prompt_tokens {
                entry.last_prompt_tokens = lpt;
            }
            self.save_locked(&inner);
        }
    }

    /// Mark a session as suspended so it auto-resets on next access.
    pub fn suspend_session(&self, session_key: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        if let Some(entry) = inner.entries.get_mut(session_key) {
            entry.suspended = true;
            self.save_locked(&inner);
            return true;
        }
        false
    }

    /// Mark a session as resumable after a restart interruption.
    pub fn mark_resume_pending(&self, session_key: &str, reason: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        if let Some(entry) = inner.entries.get_mut(session_key) {
            if entry.suspended {
                return false;
            }
            entry.resume_pending = true;
            entry.resume_reason = Some(reason.to_string());
            entry.last_resume_marked_at = Some(now());
            self.save_locked(&inner);
            return true;
        }
        false
    }

    /// Clear the resume-pending flag after a successful resumed turn.
    pub fn clear_resume_pending(&self, session_key: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        let should_clear = match inner.entries.get(session_key) {
            Some(e) => e.resume_pending,
            None => false,
        };
        if !should_clear {
            return false;
        }
        let entry = inner.entries.get_mut(session_key).unwrap();
        entry.resume_pending = false;
        entry.resume_reason = None;
        entry.last_resume_marked_at = None;
        self.save_locked(&inner);
        true
    }

    /// Drop SessionEntry records older than `max_age_days`. Returns the count
    /// removed. `max_age_days <= 0` disables pruning (returns 0).
    pub fn prune_old_entries(&self, max_age_days: i64) -> usize {
        if max_age_days <= 0 {
            return 0;
        }
        let cutoff = now() - Duration::days(max_age_days);
        let mut removed_keys: Vec<String> = Vec::new();

        {
            let mut inner = self.inner.lock().unwrap();
            self.ensure_loaded_locked(&mut inner);

            let keys: Vec<String> = inner.order.clone();
            for key in keys {
                let entry = match inner.entries.get(&key) {
                    Some(e) => e,
                    None => continue,
                };
                if entry.suspended {
                    continue;
                }
                if self.has_active_processes_fn.is_some() && self.has_active(&entry.session_key) {
                    continue;
                }
                if entry.updated_at < cutoff {
                    removed_keys.push(key.clone());
                }
            }
            for key in &removed_keys {
                inner.entries.remove(key);
                inner.order.retain(|k| k != key);
            }
            if !removed_keys.is_empty() {
                self.save_locked(&inner);
            }
        }

        if !removed_keys.is_empty() {
            log::info!(
                "SessionStore pruned {} entries older than {} days",
                removed_keys.len(),
                max_age_days
            );
        }
        removed_keys.len()
    }

    /// Mark recently-active sessions resumable after an unexpected exit.
    pub fn suspend_recently_active(&self, max_age_seconds: i64) -> usize {
        let cutoff = now() - Duration::seconds(max_age_seconds);
        let mut count = 0usize;
        let mut inner = self.inner.lock().unwrap();
        self.ensure_loaded_locked(&mut inner);
        let keys: Vec<String> = inner.order.clone();
        for key in keys {
            if let Some(entry) = inner.entries.get_mut(&key) {
                if entry.resume_pending {
                    continue;
                }
                if !entry.suspended && entry.updated_at >= cutoff {
                    entry.resume_pending = true;
                    entry.resume_reason = Some("restart_interrupted".to_string());
                    entry.last_resume_marked_at = Some(now());
                    count += 1;
                }
            }
        }
        if count > 0 {
            self.save_locked(&inner);
        }
        count
    }

    /// Force reset a session, creating a new session ID.
    pub fn reset_session(
        &self,
        session_key: &str,
        display_name: Option<&str>,
    ) -> Option<SessionEntry> {
        let mut db_end_session_id: Option<String> = None;
        let mut db_create_kwargs: Option<(String, String, Option<String>)> = None;
        let new_entry: SessionEntry;

        {
            let mut inner = self.inner.lock().unwrap();
            self.ensure_loaded_locked(&mut inner);

            let old_entry = inner.entries.get(session_key)?.clone();
            db_end_session_id = Some(old_entry.session_id.clone());

            let now_ts = now();
            let session_id = format!("{}_{}", now_ts.format("%Y%m%d_%H%M%S"), uuid_hex8());

            let mut entry = SessionEntry::new(
                session_key.to_string(),
                session_id.clone(),
                now_ts,
                now_ts,
            );
            entry.origin = old_entry.origin.clone();
            entry.display_name = match display_name {
                Some(d) => Some(d.to_string()),
                None => old_entry.display_name.clone(),
            };
            entry.platform = old_entry.platform.clone();
            entry.chat_type = old_entry.chat_type.clone();
            entry.is_fresh_reset = true;

            inner.entries.insert(session_key.to_string(), entry.clone());
            self.save_locked(&inner);

            db_create_kwargs = Some((
                session_id,
                old_entry
                    .platform
                    .as_ref()
                    .map(|p| p.value())
                    .unwrap_or_else(|| "unknown".to_string()),
                old_entry.origin.as_ref().and_then(|o| o.user_id.clone()),
            ));
            new_entry = entry;
        }

        if let (Some(db), Some(end_id)) = (&self.db, &db_end_session_id) {
            if let Err(e) = db.end_session(end_id, "session_reset") {
                log::debug!("Session DB operation failed: {}", e);
            }
        }
        if let (Some(db), Some((sid, src, uid))) = (&self.db, &db_create_kwargs) {
            if let Err(e) = db.create_session(sid, src, uid.as_deref()) {
                log::debug!("Session DB operation failed: {}", e);
            }
        }

        Some(new_entry)
    }

    /// Switch a session key to point at an existing session ID (`/resume`).
    pub fn switch_session(
        &self,
        session_key: &str,
        target_session_id: &str,
    ) -> Option<SessionEntry> {
        let mut db_end_session_id: Option<String> = None;
        let new_entry: SessionEntry;

        {
            let mut inner = self.inner.lock().unwrap();
            self.ensure_loaded_locked(&mut inner);

            let old_entry = inner.entries.get(session_key)?.clone();

            if old_entry.session_id == target_session_id {
                return Some(old_entry);
            }

            db_end_session_id = Some(old_entry.session_id.clone());

            let now_ts = now();
            let mut entry = SessionEntry::new(
                session_key.to_string(),
                target_session_id.to_string(),
                now_ts,
                now_ts,
            );
            entry.origin = old_entry.origin.clone();
            entry.display_name = old_entry.display_name.clone();
            entry.platform = old_entry.platform.clone();
            entry.chat_type = old_entry.chat_type.clone();

            inner.entries.insert(session_key.to_string(), entry.clone());
            self.save_locked(&inner);
            new_entry = entry;
        }

        if let (Some(db), Some(end_id)) = (&self.db, &db_end_session_id) {
            if let Err(e) = db.end_session(end_id, "session_switch") {
                log::debug!("Session DB end_session failed: {}", e);
            }
        }
        if let Some(db) = &self.db {
            if let Err(e) = db.reopen_session(target_session_id) {
                log::debug!("Session DB reopen_session failed: {}", e);
            }
        }

        Some(new_entry)
    }

    /// List all sessions, optionally filtered by activity (minutes), newest
    /// first.
    pub fn list_sessions(&self, active_minutes: Option<i64>) -> Vec<SessionEntry> {
        let mut entries: Vec<SessionEntry> = {
            let mut inner = self.inner.lock().unwrap();
            self.ensure_loaded_locked(&mut inner);
            inner
                .order
                .iter()
                .filter_map(|k| inner.entries.get(k).cloned())
                .collect()
        };

        if let Some(minutes) = active_minutes {
            let cutoff = now() - Duration::minutes(minutes);
            entries.retain(|e| e.updated_at >= cutoff);
        }

        entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        entries
    }

    /// Path to a session's legacy transcript file.
    pub fn get_transcript_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{}.jsonl", session_id))
    }

    /// Append a message to a session's transcript (SQLite + legacy JSONL).
    ///
    /// When `skip_db` is true only the JSONL is written.
    pub fn append_to_transcript(&self, session_id: &str, message: &Value, skip_db: bool) {
        if let Some(db) = &self.db {
            if !skip_db {
                if let Err(e) = db.append_message(session_id, message) {
                    log::debug!("Session DB operation failed: {}", e);
                }
            }
        }

        let transcript_path = self.get_transcript_path(session_id);
        let line = match serde_json::to_string(message) {
            Ok(s) => s,
            Err(e) => {
                log::debug!("Could not serialize transcript message: {}", e);
                return;
            }
        };
        let _inner = self.inner.lock().unwrap();
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&transcript_path)
        {
            let _ = f.write_all(line.as_bytes());
            let _ = f.write_all(b"\n");
        }
    }

    /// Replace the entire transcript for a session.
    pub fn rewrite_transcript(&self, session_id: &str, messages: &[Value]) {
        if let Some(db) = &self.db {
            if let Err(e) = db.replace_messages(session_id, messages) {
                log::debug!("Failed to rewrite transcript in DB: {}", e);
            }
        }
        let transcript_path = self.get_transcript_path(session_id);
        if let Ok(mut f) = fs::File::create(&transcript_path) {
            for msg in messages {
                if let Ok(s) = serde_json::to_string(msg) {
                    let _ = f.write_all(s.as_bytes());
                    let _ = f.write_all(b"\n");
                }
            }
        }
    }

    /// Load all messages from a session's transcript, preferring whichever
    /// source (SQLite vs JSONL) has more messages.
    pub fn load_transcript(&self, session_id: &str) -> Vec<Value> {
        let db_messages: Vec<Value> = match &self.db {
            Some(db) => match db.get_messages_as_conversation(session_id) {
                Ok(m) => m,
                Err(e) => {
                    log::debug!("Could not load messages from DB: {}", e);
                    Vec::new()
                }
            },
            None => Vec::new(),
        };

        let transcript_path = self.get_transcript_path(session_id);
        let mut jsonl_messages: Vec<Value> = Vec::new();
        if transcript_path.exists() {
            if let Ok(text) = fs::read_to_string(&transcript_path) {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(line) {
                        Ok(v) => jsonl_messages.push(v),
                        Err(_) => {
                            let preview: String = line.chars().take(120).collect();
                            log::warn!(
                                "Skipping corrupt line in transcript {}: {}",
                                session_id,
                                preview
                            );
                        }
                    }
                }
            }
        }

        if jsonl_messages.len() > db_messages.len() {
            if !db_messages.is_empty() {
                log::debug!(
                    "Session {}: JSONL has {} messages vs SQLite {} — using JSONL (legacy session not yet fully migrated)",
                    session_id,
                    jsonl_messages.len(),
                    db_messages.len()
                );
            }
            return jsonl_messages;
        }
        db_messages
    }
}

// ---------------------------------------------------------------------------
// build_session_context
// ---------------------------------------------------------------------------

/// Build a full session context from a source and config.
pub fn build_session_context<C: GatewayConfigLike>(
    source: &SessionSource,
    config: &C,
    session_entry: Option<&SessionEntry>,
) -> SessionContext {
    let connected = config.get_connected_platforms();

    let mut home_channels: Vec<(Platform, HomeChannel)> = Vec::new();
    for platform in &connected {
        if let Some(home) = config.get_home_channel(platform) {
            home_channels.push((platform.clone(), home));
        }
    }

    let mut context = SessionContext {
        source: source.clone(),
        connected_platforms: connected,
        home_channels,
        shared_multi_user_session: is_shared_multi_user_session(
            source,
            config.group_sessions_per_user(),
            config.thread_sessions_per_user(),
        ),
        session_key: String::new(),
        session_id: String::new(),
        created_at: None,
        updated_at: None,
    };

    if let Some(entry) = session_entry {
        context.session_key = entry.session_key.clone();
        context.session_id = entry.session_id.clone();
        context.created_at = Some(entry.created_at);
        context.updated_at = Some(entry.updated_at);
    }

    context
}

// ---------------------------------------------------------------------------
// uuid hex helper (mirrors uuid.uuid4().hex[:8])
// ---------------------------------------------------------------------------

fn uuid_hex8() -> String {
    // 4 random bytes -> 8 hex chars. Use process/time entropy without pulling
    // in a uuid crate.
    let mut bytes = [0u8; 4];
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mut seed = nanos ^ (pid << 64) ^ (nanos.rotate_left(17));
    for b in bytes.iter_mut() {
        // xorshift-ish mixing for a bit of spread
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *b = (seed & 0xff) as u8;
    }
    hex_lower(&bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestConfig {
        group_per_user: bool,
        thread_per_user: bool,
        policy: SessionResetPolicy,
        connected: Vec<Platform>,
        homes: Vec<(Platform, HomeChannel)>,
    }

    impl Default for TestConfig {
        fn default() -> Self {
            TestConfig {
                group_per_user: true,
                thread_per_user: false,
                policy: SessionResetPolicy::default(),
                connected: vec![Platform::Local, Platform::Telegram],
                homes: vec![],
            }
        }
    }

    impl GatewayConfigLike for TestConfig {
        fn group_sessions_per_user(&self) -> bool {
            self.group_per_user
        }
        fn thread_sessions_per_user(&self) -> bool {
            self.thread_per_user
        }
        fn get_reset_policy(&self, _p: Option<&Platform>, _s: &str) -> SessionResetPolicy {
            self.policy.clone()
        }
        fn get_connected_platforms(&self) -> Vec<Platform> {
            self.connected.clone()
        }
        fn get_home_channel(&self, platform: &Platform) -> Option<HomeChannel> {
            self.homes
                .iter()
                .find(|(p, _)| p == platform)
                .map(|(_, h)| h.clone())
        }
    }

    fn no_canon(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn test_hash_helpers() {
        // sha256("abc")[:12]
        assert_eq!(hash_id("abc"), "ba7816bf8f01");
        assert_eq!(hash_sender_id("abc"), "user_ba7816bf8f01");
        assert_eq!(hash_chat_id("telegram:abc"), "telegram:ba7816bf8f01");
        assert_eq!(hash_chat_id("abc"), "ba7816bf8f01");
        // colon at position 0 -> not split
        assert_eq!(hash_chat_id(":abc"), hash_id(":abc"));
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("telegram"), "Telegram");
        assert_eq!(title_case("api_server"), "Api_Server");
        assert_eq!(title_case("wecom_callback"), "Wecom_Callback");
    }

    #[test]
    fn test_platform_roundtrip() {
        assert_eq!(Platform::from_value("telegram"), Some(Platform::Telegram));
        assert_eq!(Platform::from_value("api_server"), Some(Platform::ApiServer));
        assert_eq!(
            Platform::from_value("irc"),
            Some(Platform::Plugin("irc".into()))
        );
        assert_eq!(Platform::from_value("  "), None);
        assert_eq!(Platform::Plugin("irc".into()).value(), "irc");
        assert_eq!(Platform::Telegram.value(), "telegram");
    }

    #[test]
    fn test_build_session_key_dm() {
        let mut src = SessionSource::new(Platform::Telegram, "12345");
        src.chat_type = "dm".into();
        assert_eq!(
            build_session_key(&src, true, false, no_canon),
            "agent:main:telegram:dm:12345"
        );
        src.thread_id = Some("t1".into());
        assert_eq!(
            build_session_key(&src, true, false, no_canon),
            "agent:main:telegram:dm:12345:t1"
        );
        // No chat_id, only thread.
        let mut src2 = SessionSource::new(Platform::Telegram, "");
        src2.chat_type = "dm".into();
        src2.thread_id = Some("tonly".into());
        assert_eq!(
            build_session_key(&src2, true, false, no_canon),
            "agent:main:telegram:dm:tonly"
        );
        let mut src3 = SessionSource::new(Platform::Telegram, "");
        src3.chat_type = "dm".into();
        assert_eq!(
            build_session_key(&src3, true, false, no_canon),
            "agent:main:telegram:dm"
        );
    }

    #[test]
    fn test_build_session_key_group() {
        let mut src = SessionSource::new(Platform::Discord, "chan1");
        src.chat_type = "group".into();
        src.user_id = Some("u1".into());
        // group_sessions_per_user=true -> isolate
        assert_eq!(
            build_session_key(&src, true, false, no_canon),
            "agent:main:discord:group:chan1:u1"
        );
        // group_sessions_per_user=false -> shared
        assert_eq!(
            build_session_key(&src, false, false, no_canon),
            "agent:main:discord:group:chan1"
        );
        // thread shared by default (no user appended)
        src.thread_id = Some("th1".into());
        assert_eq!(
            build_session_key(&src, true, false, no_canon),
            "agent:main:discord:group:chan1:th1"
        );
        // thread per user enabled
        assert_eq!(
            build_session_key(&src, true, true, no_canon),
            "agent:main:discord:group:chan1:th1:u1"
        );
    }

    #[test]
    fn test_is_shared_multi_user() {
        let mut src = SessionSource::new(Platform::Slack, "c1");
        src.chat_type = "dm".into();
        assert!(!is_shared_multi_user_session(&src, true, false));
        src.chat_type = "group".into();
        assert!(!is_shared_multi_user_session(&src, true, false));
        assert!(is_shared_multi_user_session(&src, false, false));
        src.thread_id = Some("t".into());
        assert!(is_shared_multi_user_session(&src, true, false));
        assert!(!is_shared_multi_user_session(&src, true, true));
    }

    #[test]
    fn test_session_source_dict_roundtrip() {
        let mut src = SessionSource::new(Platform::Discord, "c1");
        src.chat_type = "group".into();
        src.user_id = Some("u1".into());
        src.guild_id = Some("g1".into());
        src.message_id = Some("m1".into());
        let d = src.to_dict();
        let back = SessionSource::from_dict(&d).unwrap();
        assert_eq!(back.platform, Platform::Discord);
        assert_eq!(back.chat_id, "c1");
        assert_eq!(back.guild_id, Some("g1".into()));
        assert_eq!(back.message_id, Some("m1".into()));
        // user_id_alt absent -> not serialized
        assert!(d.get("user_id_alt").is_none());
    }

    #[test]
    fn test_session_entry_dict_roundtrip() {
        let ts = now();
        let mut e = SessionEntry::new("k1", "sid1", ts, ts);
        e.platform = Some(Platform::Telegram);
        e.input_tokens = 42;
        e.suspended = true;
        e.resume_pending = true;
        e.resume_reason = Some("restart_timeout".into());
        e.last_resume_marked_at = Some(ts);
        let d = e.to_dict();
        let back = SessionEntry::from_dict(&d).unwrap();
        assert_eq!(back.session_key, "k1");
        assert_eq!(back.platform, Some(Platform::Telegram));
        assert_eq!(back.input_tokens, 42);
        assert!(back.suspended);
        assert!(back.resume_pending);
        assert_eq!(back.resume_reason, Some("restart_timeout".into()));
    }

    #[test]
    fn test_entry_legacy_memory_flushed() {
        let ts = now();
        let mut d = SessionEntry::new("k", "s", ts, ts).to_dict();
        let obj = d.as_object_mut().unwrap();
        obj.remove("expiry_finalized");
        obj.insert("memory_flushed".into(), Value::Bool(true));
        let back = SessionEntry::from_dict(&d).unwrap();
        assert!(back.expiry_finalized);
    }

    #[test]
    fn test_reset_policy_none_never_expires() {
        let policy = SessionResetPolicy {
            mode: "none".into(),
            ..Default::default()
        };
        let old = now() - Duration::days(365);
        assert!(!SessionStore::<TestConfig>::expired_for_policy(&policy, &old));
    }

    #[test]
    fn test_reset_policy_idle() {
        let policy = SessionResetPolicy {
            mode: "idle".into(),
            idle_minutes: 60,
            ..Default::default()
        };
        let old = now() - Duration::minutes(120);
        assert!(SessionStore::<TestConfig>::expired_for_policy(&policy, &old));
        let recent = now() - Duration::minutes(5);
        assert!(!SessionStore::<TestConfig>::expired_for_policy(&policy, &recent));
    }

    #[test]
    fn test_get_or_create_and_persistence() {
        let tmp = std::env::temp_dir().join(format!("gw_session_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let cfg = TestConfig::default();
        let store = SessionStore::new(tmp.clone(), cfg);

        let mut src = SessionSource::new(Platform::Telegram, "999");
        src.chat_type = "dm".into();
        src.chat_name = Some("Alice".into());

        let e1 = store.get_or_create_session(&src, false);
        assert_eq!(e1.session_key, "agent:main:telegram:dm:999");
        assert_eq!(e1.display_name, Some("Alice".into()));

        // Second call returns same session_id (not reset).
        let e2 = store.get_or_create_session(&src, false);
        assert_eq!(e1.session_id, e2.session_id);

        // sessions.json was written.
        assert!(tmp.join("sessions.json").exists());

        // Suspend -> next access resets.
        assert!(store.suspend_session(&e1.session_key));
        let e3 = store.get_or_create_session(&src, false);
        assert_ne!(e1.session_id, e3.session_id);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_resume_pending_preserves_id() {
        let tmp = std::env::temp_dir().join(format!("gw_session_resume_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let store = SessionStore::new(tmp.clone(), TestConfig::default());

        let mut src = SessionSource::new(Platform::Telegram, "1");
        src.chat_type = "dm".into();
        let e1 = store.get_or_create_session(&src, false);
        assert!(store.mark_resume_pending(&e1.session_key, "restart_timeout"));
        let e2 = store.get_or_create_session(&src, false);
        // Same session_id preserved.
        assert_eq!(e1.session_id, e2.session_id);
        // Clear works.
        assert!(store.clear_resume_pending(&e1.session_key));
        assert!(!store.clear_resume_pending(&e1.session_key));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_transcript_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("gw_session_tx_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::create_dir_all(&tmp);
        let store = SessionStore::new(tmp.clone(), TestConfig::default());

        let m1 = serde_json::json!({"role": "user", "content": "hi"});
        let m2 = serde_json::json!({"role": "assistant", "content": "hello"});
        store.append_to_transcript("sid", &m1, true);
        store.append_to_transcript("sid", &m2, true);
        let loaded = store.load_transcript("sid");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0]["content"], "hi");

        store.rewrite_transcript("sid", &[m2.clone()]);
        let loaded2 = store.load_transcript("sid");
        assert_eq!(loaded2.len(), 1);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_prompt_redaction() {
        let mut src = SessionSource::new(Platform::Telegram, "12345");
        src.chat_type = "dm".into();
        src.user_id = Some("secret".into());
        let ctx = SessionContext {
            source: src,
            connected_platforms: vec![Platform::Local, Platform::Telegram],
            home_channels: vec![],
            shared_multi_user_session: false,
            session_key: String::new(),
            session_id: String::new(),
            created_at: None,
            updated_at: None,
        };
        let prompt = build_session_context_prompt(
            &ctx,
            true,
            default_discord_tools_loaded,
            "/home/u/.hermes",
            |_| false,
        );
        // user_id should be hashed.
        assert!(prompt.contains(&hash_sender_id("secret")));
        assert!(!prompt.contains("secret"));
        assert!(prompt.contains("## Current Session Context"));
    }

    #[test]
    fn test_prompt_discord_disclaimer() {
        let mut src = SessionSource::new(Platform::Discord, "chan");
        src.chat_type = "channel".into();
        let ctx = SessionContext {
            source: src,
            connected_platforms: vec![Platform::Local],
            home_channels: vec![],
            shared_multi_user_session: false,
            session_key: String::new(),
            session_id: String::new(),
            created_at: None,
            updated_at: None,
        };
        // tools not loaded -> disclaimer
        let p = build_session_context_prompt(&ctx, false, || false, "/h", |_| false);
        assert!(p.contains("You do NOT have access to Discord-specific APIs"));
        // tools loaded -> IDs block
        let p2 = build_session_context_prompt(&ctx, false, || true, "/h", |_| false);
        assert!(p2.contains("Discord IDs (for the `discord`"));
        assert!(p2.contains("- Channel: `chan`"));
    }

    #[test]
    fn test_build_session_context() {
        let mut src = SessionSource::new(Platform::Slack, "c1");
        src.chat_type = "group".into();
        let cfg = TestConfig {
            group_per_user: false,
            ..Default::default()
        };
        let ctx = build_session_context(&src, &cfg, None);
        assert!(ctx.shared_multi_user_session);
        assert_eq!(ctx.connected_platforms.len(), 2);
    }
}
