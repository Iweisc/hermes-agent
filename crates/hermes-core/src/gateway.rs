use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Duration, Local, NaiveDateTime, Timelike};
use getrandom::fill as fill_random;
use regex::Regex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const GATEWAY_SERVICE_RESTART_EXIT_CODE: i32 = 75;
pub const MAX_PLATFORM_OUTPUT: usize = 4000;
pub const TRUNCATED_VISIBLE: usize = 3800;
pub const RESTART_NOTIFY_FILENAME: &str = ".restart_notify.json";
pub const RESTART_LAST_PROCESSED_FILENAME: &str = ".restart_last_processed.json";
pub const CLEAN_SHUTDOWN_FILENAME: &str = ".clean_shutdown";
pub const RESTART_FAILURE_COUNTS_FILENAME: &str = ".restart_failure_counts";
pub const STARTUP_RECENT_ACTIVITY_WINDOW_SECONDS: i64 = 120;
pub const STUCK_LOOP_THRESHOLD: u64 = 3;
pub const PAIRING_CODE_LENGTH: usize = 8;
pub const PAIRING_CODE_TTL_SECONDS: f64 = 3600.0;
pub const PAIRING_RATE_LIMIT_SECONDS: f64 = 600.0;
pub const PAIRING_LOCKOUT_SECONDS: f64 = 3600.0;
pub const PAIRING_MAX_PENDING_PER_PLATFORM: usize = 3;
pub const PAIRING_MAX_FAILED_ATTEMPTS: u64 = 5;

const PAIRING_ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_reset_hour() -> u8 {
    4
}

fn default_idle_minutes() -> u32 {
    1440
}

fn default_notify_exclude_platforms() -> Vec<Platform> {
    vec![
        Platform::parse("api_server").expect("builtin platform"),
        Platform::parse("webhook").expect("builtin platform"),
    ]
}

fn isoformat_local(value: &NaiveDateTime) -> String {
    value.format("%Y-%m-%dT%H:%M:%S%.f").to_string()
}

fn parse_datetime(raw: &str) -> Result<NaiveDateTime, String> {
    NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| {
            DateTime::parse_from_rfc3339(raw).map(|value| value.with_timezone(&Local).naive_local())
        })
        .map_err(|error| format!("invalid datetime '{raw}': {error}"))
}

mod datetime_serde {
    use super::{NaiveDateTime, isoformat_local, parse_datetime};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&isoformat_local(value))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        parse_datetime(&raw).map_err(serde::de::Error::custom)
    }
}

mod option_datetime_serde {
    use super::{NaiveDateTime, isoformat_local, parse_datetime};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<NaiveDateTime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(&isoformat_local(value)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<NaiveDateTime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Option::<String>::deserialize(deserializer)?;
        raw.map(|value| parse_datetime(&value))
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Platform(String);

impl Platform {
    pub fn parse(value: &str) -> Result<Self, String> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err("platform must not be empty".to_string());
        }
        if !normalized
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
        {
            return Err(format!(
                "platform '{value}' must contain only lowercase letters, digits, '_' or '-'"
            ));
        }
        Ok(Self(normalized))
    }

    pub fn local() -> Self {
        Self("local".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_local(&self) -> bool {
        self.0 == "local"
    }
}

impl Display for Platform {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomeChannel {
    pub platform: Platform,
    pub chat_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

impl HomeChannel {
    pub fn new(
        platform: Platform,
        chat_id: impl Into<String>,
        name: impl Into<String>,
        thread_id: Option<String>,
    ) -> Result<Self, String> {
        let chat_id = chat_id.into().trim().to_string();
        if chat_id.is_empty() {
            return Err("home channel chat_id must not be empty".to_string());
        }
        let name = name.into().trim().to_string();
        if name.is_empty() {
            return Err("home channel name must not be empty".to_string());
        }
        if thread_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("home channel thread_id must not be empty".to_string());
        }
        Ok(Self {
            platform,
            chat_id,
            name,
            thread_id,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResetMode {
    Daily,
    Idle,
    #[default]
    Both,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionResetPolicy {
    #[serde(default)]
    pub mode: ResetMode,
    #[serde(default = "default_reset_hour")]
    pub at_hour: u8,
    #[serde(default = "default_idle_minutes")]
    pub idle_minutes: u32,
    #[serde(default = "default_true")]
    pub notify: bool,
    #[serde(default = "default_notify_exclude_platforms")]
    pub notify_exclude_platforms: Vec<Platform>,
}

impl Default for SessionResetPolicy {
    fn default() -> Self {
        Self {
            mode: ResetMode::Both,
            at_hour: default_reset_hour(),
            idle_minutes: default_idle_minutes(),
            notify: true,
            notify_exclude_platforms: default_notify_exclude_platforms(),
        }
    }
}

impl SessionResetPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.at_hour > 23 {
            return Err("reset policy at_hour must be within 0..=23".to_string());
        }
        if self.idle_minutes == 0 && matches!(self.mode, ResetMode::Idle | ResetMode::Both) {
            return Err(
                "reset policy idle_minutes must be positive for idle-based resets".to_string(),
            );
        }
        Ok(())
    }

    pub fn reset_reason(
        &self,
        updated_at: NaiveDateTime,
        now: NaiveDateTime,
    ) -> Option<&'static str> {
        match self.mode {
            ResetMode::None => None,
            ResetMode::Idle => self.idle_reason(updated_at, now),
            ResetMode::Daily => self.daily_reason(updated_at, now),
            ResetMode::Both => self
                .idle_reason(updated_at, now)
                .or_else(|| self.daily_reason(updated_at, now)),
        }
    }

    fn idle_reason(&self, updated_at: NaiveDateTime, now: NaiveDateTime) -> Option<&'static str> {
        let deadline = updated_at + Duration::minutes(i64::from(self.idle_minutes));
        (now > deadline).then_some("idle")
    }

    fn daily_reason(&self, updated_at: NaiveDateTime, now: NaiveDateTime) -> Option<&'static str> {
        let mut boundary = now
            .date()
            .and_hms_opt(u32::from(self.at_hour), 0, 0)
            .expect("valid reset hour");
        if now.hour() < u32::from(self.at_hour) {
            boundary -= Duration::days(1);
        }
        (updated_at < boundary).then_some("daily")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub home_channels: HashMap<Platform, HomeChannel>,
    #[serde(default)]
    pub default_reset_policy: SessionResetPolicy,
    #[serde(default)]
    pub reset_by_type: HashMap<String, SessionResetPolicy>,
    #[serde(default)]
    pub reset_by_platform: HashMap<Platform, SessionResetPolicy>,
    #[serde(default = "default_true")]
    pub group_sessions_per_user: bool,
    #[serde(default = "default_false")]
    pub thread_sessions_per_user: bool,
    #[serde(default = "default_true")]
    pub always_log_local: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            home_channels: HashMap::new(),
            default_reset_policy: SessionResetPolicy::default(),
            reset_by_type: HashMap::new(),
            reset_by_platform: HashMap::new(),
            group_sessions_per_user: true,
            thread_sessions_per_user: false,
            always_log_local: true,
        }
    }
}

impl RuntimeConfig {
    pub fn get_home_channel(&self, platform: &Platform) -> Option<&HomeChannel> {
        self.home_channels.get(platform)
    }

    pub fn get_reset_policy(
        &self,
        platform: Option<&Platform>,
        session_type: Option<&str>,
    ) -> &SessionResetPolicy {
        if let Some(platform) = platform {
            if let Some(policy) = self.reset_by_platform.get(platform) {
                return policy;
            }
        }
        if let Some(session_type) = session_type {
            if let Some(policy) = self.reset_by_type.get(session_type) {
                return policy;
            }
        }
        &self.default_reset_policy
    }

    pub fn validate(&self) -> Result<(), String> {
        self.default_reset_policy.validate()?;
        for policy in self.reset_by_type.values() {
            policy.validate()?;
        }
        for policy in self.reset_by_platform.values() {
            policy.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSource {
    pub platform: Platform,
    pub chat_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_name: Option<String>,
    #[serde(default = "default_chat_type")]
    pub chat_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id_alt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_id_alt: Option<String>,
    #[serde(default)]
    pub is_bot: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guild_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_chat_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
}

fn default_chat_type() -> String {
    "dm".to_string()
}

impl SessionSource {
    pub fn validate(&self) -> Result<(), String> {
        if self.chat_id.trim().is_empty() {
            return Err("session source chat_id must not be empty".to_string());
        }
        if self.chat_type.trim().is_empty() {
            return Err("session source chat_type must not be empty".to_string());
        }
        if self
            .thread_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("session source thread_id must not be empty".to_string());
        }
        Ok(())
    }

    pub fn description(&self) -> String {
        if self.platform.is_local() {
            return "CLI terminal".to_string();
        }
        let mut parts = Vec::new();
        match self.chat_type.as_str() {
            "dm" => parts.push(format!(
                "DM with {}",
                self.user_name
                    .as_deref()
                    .or(self.user_id.as_deref())
                    .unwrap_or("user")
            )),
            "group" => parts.push(format!(
                "group: {}",
                self.chat_name.as_deref().unwrap_or(self.chat_id.as_str())
            )),
            "channel" => parts.push(format!(
                "channel: {}",
                self.chat_name.as_deref().unwrap_or(self.chat_id.as_str())
            )),
            _ => parts.push(
                self.chat_name
                    .clone()
                    .unwrap_or_else(|| self.chat_id.clone()),
            ),
        }
        if let Some(thread_id) = self.thread_id.as_deref() {
            parts.push(format!("thread: {thread_id}"));
        }
        parts.join(", ")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    #[default]
    Text,
    Command,
    Photo,
    Voice,
    Audio,
    Video,
    Document,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageEvent {
    pub text: String,
    #[serde(default)]
    pub message_type: MessageType,
    pub source: SessionSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform_update_id: Option<i64>,
    #[serde(default)]
    pub media_urls: Vec<String>,
    #[serde(default)]
    pub media_types: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to_message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_prompt: Option<String>,
    #[serde(default)]
    pub internal: bool,
}

impl MessageEvent {
    pub fn is_command(&self) -> bool {
        self.text.starts_with('/')
    }

    pub fn get_command(&self) -> Option<String> {
        if !self.is_command() {
            return None;
        }
        let first = self.text.split_whitespace().next()?;
        let mut raw = first.strip_prefix('/')?.to_ascii_lowercase();
        if let Some((command, _mention)) = raw.split_once('@') {
            raw = command.to_string();
        }
        if raw.contains('/') || raw.is_empty() {
            return None;
        }
        Some(raw)
    }

    pub fn get_command_args(&self) -> String {
        if !self.is_command() {
            return self.text.clone();
        }
        let args = self
            .text
            .split_once(char::is_whitespace)
            .map(|(_, args)| args.trim_start())
            .unwrap_or_default();
        args.replace("\u{2014}\u{2014}", "--")
            .replace('\u{2014}', "--")
            .replace('\u{2013}', "-")
    }

    pub fn coerce_plaintext_gateway_command(&mut self) {
        if self.message_type != MessageType::Text {
            return;
        }
        let trimmed = self.text.trim();
        if trimmed.is_empty() || trimmed.starts_with('/') || self.source.chat_type != "dm" {
            return;
        }
        if matches_plaintext_gateway_restart(trimmed) {
            self.text = "/restart".to_string();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    pub canonical: &'static str,
    pub typed: String,
    pub gateway_dispatchable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlashCommandDef {
    name: &'static str,
    aliases: &'static [&'static str],
    gateway_dispatchable: bool,
}

const COMMAND_DEFS: &[SlashCommandDef] = &[
    SlashCommandDef {
        name: "new",
        aliases: &["reset"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "topic",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "clear",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "redraw",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "history",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "save",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "retry",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "undo",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "title",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "branch",
        aliases: &["fork"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "compress",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "rollback",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "snapshot",
        aliases: &["snap"],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "stop",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "approve",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "deny",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "background",
        aliases: &["bg", "btw"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "agents",
        aliases: &["tasks"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "queue",
        aliases: &["q"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "steer",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "goal",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "status",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "profile",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "sethome",
        aliases: &["set-home"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "resume",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "config",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "model",
        aliases: &["provider"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "gquota",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "personality",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "statusbar",
        aliases: &["sb"],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "verbose",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "footer",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "yolo",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "reasoning",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "fast",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "skin",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "indicator",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "voice",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "busy",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "tools",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "toolsets",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "skills",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "cron",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "curator",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "kanban",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "reload",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "reload-mcp",
        aliases: &["reload_mcp"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "reload-skills",
        aliases: &["reload_skills"],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "browser",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "plugins",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "commands",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "help",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "restart",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "usage",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "insights",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "platforms",
        aliases: &["gateway"],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "copy",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "paste",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "image",
        aliases: &[],
        gateway_dispatchable: false,
    },
    SlashCommandDef {
        name: "update",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "debug",
        aliases: &[],
        gateway_dispatchable: true,
    },
    SlashCommandDef {
        name: "quit",
        aliases: &["exit"],
        gateway_dispatchable: false,
    },
];

pub fn resolve_command(name: &str) -> Option<ResolvedCommand> {
    let normalized = normalize_command_name(name)?;
    COMMAND_DEFS.iter().find_map(|command| {
        if command.name == normalized || command.aliases.iter().any(|alias| *alias == normalized) {
            Some(ResolvedCommand {
                canonical: command.name,
                typed: normalized.to_string(),
                gateway_dispatchable: command.gateway_dispatchable,
            })
        } else {
            None
        }
    })
}

pub fn is_gateway_known_command(name: &str) -> bool {
    resolve_command(name).is_some_and(|command| command.gateway_dispatchable)
}

pub fn should_bypass_active_session(name: Option<&str>) -> bool {
    name.and_then(resolve_command).is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BusyInputMode {
    #[default]
    Interrupt,
    Queue,
    Steer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningSessionState {
    pub draining: bool,
    pub queue_during_drain: bool,
    pub busy_input_mode: BusyInputMode,
    pub can_steer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveSessionIngressAction {
    DispatchCommand { canonical: &'static str },
    QueuePending,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunningSessionAction {
    DispatchCommand { canonical: &'static str },
    QueuePending,
    SteerActive,
    InterruptAndQueue,
    QueueDuringDrain,
    RejectDuringDrain,
    Reject { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveSessionPhase {
    PendingStart,
    Running { can_steer: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSession {
    pub session_key: String,
    pub phase: ActiveSessionPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingReason {
    Busy,
    Interrupt,
    Drain,
    ExplicitQueue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GatewayIngressDecision {
    DispatchCommand {
        session_key: String,
        canonical: &'static str,
        event: MessageEvent,
    },
    StartTurn {
        session_key: String,
        session: SessionEntry,
        event: MessageEvent,
    },
    QueuePending {
        session_key: String,
        depth: usize,
        reason: PendingReason,
    },
    SteerActive {
        session_key: String,
        event: MessageEvent,
    },
    Reject {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayNotification {
    pub platform: Platform,
    pub chat_id: String,
    pub thread_id: Option<String>,
    pub message: String,
}

impl GatewayNotification {
    pub fn dedupe_key(&self) -> (String, String, Option<String>) {
        (
            self.platform.to_string(),
            self.chat_id.clone(),
            self.thread_id.clone(),
        )
    }

    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        self.thread_id
            .as_ref()
            .map(|thread_id| HashMap::from([("thread_id".to_string(), thread_id.clone())]))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GatewayBusyReplyContext {
    pub ack_enabled: bool,
    pub cooldown_active: bool,
    pub elapsed_minutes: Option<u64>,
    pub api_call_count: Option<u32>,
    pub max_iterations: Option<u32>,
    pub current_tool: Option<String>,
    pub onboarding_hint: Option<String>,
}

impl GatewayBusyReplyContext {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .current_tool
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("busy reply current_tool must not be empty".to_string());
        }
        if self
            .onboarding_hint
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("busy reply onboarding_hint must not be empty".to_string());
        }
        if let Some(max_iterations) = self.max_iterations {
            if max_iterations == 0 {
                return Err("busy reply max_iterations must be positive".to_string());
            }
            if let Some(api_call_count) = self.api_call_count {
                if api_call_count > max_iterations {
                    return Err(
                        "busy reply api_call_count must not exceed max_iterations".to_string()
                    );
                }
            }
        }
        Ok(())
    }

    fn status_detail(&self) -> String {
        let mut parts = Vec::new();
        if let Some(elapsed_minutes) = self.elapsed_minutes {
            if elapsed_minutes > 0 {
                parts.push(format!("{elapsed_minutes} min elapsed"));
            }
        }
        if let Some(max_iterations) = self.max_iterations {
            let iteration = self.api_call_count.unwrap_or(0);
            parts.push(format!("iteration {iteration}/{max_iterations}"));
        }
        if let Some(current_tool) = self.current_tool.as_deref() {
            parts.push(format!("running: {current_tool}"));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!(" ({})", parts.join(", "))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GatewayBusyReplyStatus {
    pub elapsed_minutes: Option<u64>,
    pub api_call_count: Option<u32>,
    pub max_iterations: Option<u32>,
    pub current_tool: Option<String>,
    pub onboarding_hint: Option<String>,
}

impl GatewayBusyReplyStatus {
    pub fn validate(&self) -> Result<(), String> {
        GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: self.elapsed_minutes,
            api_call_count: self.api_call_count,
            max_iterations: self.max_iterations,
            current_tool: self.current_tool.clone(),
            onboarding_hint: self.onboarding_hint.clone(),
        }
        .validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayIngressHostResponse {
    pub message: String,
    pub reply_to_message_id: Option<String>,
    pub thread_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayIngressHostEffect {
    None,
    InterruptRunningAgent { session_key: String, reason: String },
    SteerRunningAgent { session_key: String, text: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayIngressHostPlan {
    pub decision: GatewayIngressDecision,
    pub response: Option<GatewayIngressHostResponse>,
    pub response_on_effect_failure: Option<GatewayIngressHostResponse>,
    pub effect: GatewayIngressHostEffect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayIngressExecutionResult {
    pub decision: GatewayIngressDecision,
    pub response: Option<GatewayIngressHostResponse>,
    pub effect: GatewayIngressHostEffect,
    pub effect_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayAuthorizationStatus {
    Authorized,
    Unauthorized,
    MissingUserId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnauthorizedDmBehavior {
    Ignore,
    Pair,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayPairingRequest {
    pub platform: Platform,
    pub chat_id: String,
    pub user_id: String,
    pub user_name: Option<String>,
}

impl GatewayPairingRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.chat_id.trim().is_empty() {
            return Err("pairing request chat_id must not be empty".to_string());
        }
        if self.user_id.trim().is_empty() {
            return Err("pairing request user_id must not be empty".to_string());
        }
        Ok(())
    }

    pub fn from_event(event: &MessageEvent) -> Result<Self, String> {
        if event.source.chat_type != "dm" {
            return Err("pairing request requires a dm event".to_string());
        }
        if event.source.chat_id.trim().is_empty() {
            return Err("pairing request chat_id must not be empty".to_string());
        }
        let user_id = event
            .source
            .user_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "pairing request user_id must not be empty".to_string())?;
        Ok(Self {
            platform: event.source.platform.clone(),
            chat_id: event.source.chat_id.clone(),
            user_id: user_id.to_string(),
            user_name: event.source.user_name.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatewayPairingPendingEntry {
    pub user_id: String,
    #[serde(default)]
    pub user_name: String,
    pub created_at: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatewayPairingApprovedEntry {
    #[serde(default)]
    pub user_name: String,
    pub approved_at: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayPairingApproval {
    pub user_id: String,
    pub user_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayPairingCodeDecision {
    Suppress,
    SendPairingCode { code: String },
    SendTryLater,
}

#[derive(Debug, Clone)]
pub struct GatewayPairingStore {
    pairing_dir: PathBuf,
}

impl GatewayPairingStore {
    pub fn new(pairing_dir: impl Into<PathBuf>) -> Result<Self, String> {
        let pairing_dir = pairing_dir.into();
        if pairing_dir.as_os_str().is_empty() {
            return Err("pairing directory must not be empty".to_string());
        }
        Ok(Self { pairing_dir })
    }

    pub fn is_approved(&self, platform: &str, user_id: &str) -> Result<bool, String> {
        let platform = normalize_pairing_platform(platform)?;
        let user_id = normalize_pairing_user_id(user_id)?;
        let approved = read_pairing_approved_entries(&self.pairing_dir, platform)?;
        Ok(approved.contains_key(user_id))
    }

    pub fn request_code(
        &self,
        request: &GatewayPairingRequest,
        now_seconds: f64,
    ) -> Result<GatewayPairingCodeDecision, String> {
        request.validate()?;
        validate_pairing_now_seconds(now_seconds)?;

        let platform = request.platform.as_str();
        let user_id = request.user_id.trim();
        let mut pending = read_pairing_pending_entries(&self.pairing_dir, platform)?;
        cleanup_expired_pairing_entries(&self.pairing_dir, platform, &mut pending, now_seconds)?;

        let mut limits = read_pairing_rate_limits(&self.pairing_dir)?;
        if is_pairing_rate_limited(&limits, platform, user_id, now_seconds) {
            return Ok(GatewayPairingCodeDecision::Suppress);
        }
        if is_pairing_locked_out(&limits, platform, now_seconds)
            || pending.len() >= PAIRING_MAX_PENDING_PER_PLATFORM
        {
            record_pairing_rate_limit(&mut limits, platform, user_id, now_seconds);
            write_pairing_rate_limits(&self.pairing_dir, &limits)?;
            return Ok(GatewayPairingCodeDecision::SendTryLater);
        }

        let code = generate_pairing_code(&pending)?;
        pending.insert(
            code.clone(),
            GatewayPairingPendingEntry {
                user_id: user_id.to_string(),
                user_name: request.user_name.clone().unwrap_or_default(),
                created_at: now_seconds,
            },
        );
        write_pairing_pending_entries(&self.pairing_dir, platform, &pending)?;
        record_pairing_rate_limit(&mut limits, platform, user_id, now_seconds);
        write_pairing_rate_limits(&self.pairing_dir, &limits)?;
        Ok(GatewayPairingCodeDecision::SendPairingCode { code })
    }

    pub fn approve_code(
        &self,
        platform: &str,
        code: &str,
        now_seconds: f64,
    ) -> Result<Option<GatewayPairingApproval>, String> {
        let platform = normalize_pairing_platform(platform)?;
        let code = normalize_pairing_code(code)?;
        validate_pairing_now_seconds(now_seconds)?;

        let mut pending = read_pairing_pending_entries(&self.pairing_dir, platform)?;
        cleanup_expired_pairing_entries(&self.pairing_dir, platform, &mut pending, now_seconds)?;

        let Some(entry) = pending.remove(&code) else {
            let mut limits = read_pairing_rate_limits(&self.pairing_dir)?;
            record_pairing_failed_attempt(&mut limits, platform, now_seconds);
            write_pairing_rate_limits(&self.pairing_dir, &limits)?;
            return Ok(None);
        };

        write_pairing_pending_entries(&self.pairing_dir, platform, &pending)?;
        let mut approved = read_pairing_approved_entries(&self.pairing_dir, platform)?;
        approved.insert(
            entry.user_id.clone(),
            GatewayPairingApprovedEntry {
                user_name: entry.user_name.clone(),
                approved_at: now_seconds,
            },
        );
        write_pairing_approved_entries(&self.pairing_dir, platform, &approved)?;
        Ok(Some(GatewayPairingApproval {
            user_id: entry.user_id,
            user_name: entry.user_name,
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GatewayHostHandleOutcome {
    DroppedMissingUserId {
        platform: Platform,
        chat_id: String,
    },
    DroppedUnauthorizedActiveSession {
        session_key: String,
    },
    DroppedUnauthorizedColdPath {
        session_key: String,
    },
    RequireUnauthorizedDmPairing {
        request: GatewayPairingRequest,
    },
    RepliedUnauthorizedDmPairing {
        session_key: String,
        response: GatewayIngressHostResponse,
    },
    Executed(GatewayIngressExecutionResult),
}

pub trait GatewayIngressEffectHandler {
    fn interrupt_running_agent(&mut self, session_key: &str, reason: &str) -> Result<(), String>;
    fn steer_running_agent(&mut self, session_key: &str, text: &str) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestartNotifyMarker {
    pub platform: Platform,
    pub chat_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

impl RestartNotifyMarker {
    pub fn from_event(event: &MessageEvent) -> Result<Self, String> {
        if event.source.chat_id.trim().is_empty() {
            return Err("restart notification chat_id must not be empty".to_string());
        }
        Ok(Self {
            platform: event.source.platform.clone(),
            chat_id: event.source.chat_id.clone(),
            thread_id: event.source.thread_id.clone(),
        })
    }

    pub fn to_notification(&self) -> GatewayNotification {
        GatewayNotification {
            platform: self.platform.clone(),
            chat_id: self.chat_id.clone(),
            thread_id: self.thread_id.clone(),
            message: "♻ Gateway restarted successfully. Your session continues.".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestartDedupMarker {
    pub platform: Platform,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update_id: Option<i64>,
    pub requested_at: f64,
}

impl RestartDedupMarker {
    pub fn from_event(event: &MessageEvent, requested_at: f64) -> Self {
        Self {
            platform: event.source.platform.clone(),
            update_id: event.platform_update_id.map(i64::from),
            requested_at,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GatewayHostAction {
    WriteJson {
        path: PathBuf,
        value: serde_json::Value,
    },
    WriteText {
        path: PathBuf,
        content: String,
    },
    DeleteFile {
        path: PathBuf,
    },
    SendNotification {
        notification: GatewayNotification,
    },
    ScheduleRestart {
        launch_mode: RestartLaunchMode,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayHostPlan {
    pub reply: Option<String>,
    pub actions: Vec<GatewayHostAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayStartupRestartFlow {
    pub restart_notification: GatewayHostPlan,
    pub home_notifications_if_delivered: GatewayHostPlan,
    pub home_notifications_if_not_delivered: GatewayHostPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayStartupHostFlow {
    pub recovery: GatewayStartupRecovery,
    pub pre_actions: Vec<GatewayHostAction>,
    pub restart: GatewayStartupRestartFlow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayShutdownHostFlow {
    pub drain: GatewayDrainPlan,
    pub pre_drain: GatewayHostPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayStartupExecutionReport {
    pub recovery: GatewayStartupRecovery,
    pub suspended_recent_sessions: usize,
    pub suspended_stuck_loop_sessions: usize,
    pub host_actions: GatewayHostExecutionReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayTurnCompletionReport {
    pub cleared_restart_failure_count: bool,
    pub cleared_resume_pending: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayTurnFinishReport {
    pub completion: GatewayTurnCompletionReport,
    pub next: Option<GatewayIngressDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayShutdownExecutionReport {
    pub outcome: GatewayShutdownOutcome,
    pub host_actions: GatewayHostExecutionReport,
    pub restart_failure_counts: HashMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayHostExecutionReport {
    pub attempted_notification_targets: Vec<(String, String, Option<String>)>,
    pub delivered_notification_targets: Vec<(String, String, Option<String>)>,
    pub failed_notification_targets: Vec<(String, String, Option<String>)>,
    pub scheduled_restarts: Vec<RestartLaunchMode>,
}

impl GatewayHostExecutionReport {
    fn empty() -> Self {
        Self {
            attempted_notification_targets: Vec::new(),
            delivered_notification_targets: Vec::new(),
            failed_notification_targets: Vec::new(),
            scheduled_restarts: Vec::new(),
        }
    }

    fn merge(&mut self, other: Self) {
        self.attempted_notification_targets
            .extend(other.attempted_notification_targets);
        self.delivered_notification_targets
            .extend(other.delivered_notification_targets);
        self.failed_notification_targets
            .extend(other.failed_notification_targets);
        self.scheduled_restarts.extend(other.scheduled_restarts);
    }
}

pub trait GatewayHostActionHandler {
    fn send_notification(&mut self, notification: &GatewayNotification) -> Result<bool, String>;
    fn schedule_restart(&mut self, launch_mode: RestartLaunchMode) -> Result<(), String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartLaunchMode {
    Detached,
    Service,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RestartCommandDecision {
    IgnoreRedelivery,
    AlreadyInProgress {
        message: String,
    },
    BeginRestart {
        launch_mode: RestartLaunchMode,
        notify_marker: RestartNotifyMarker,
        dedup_marker: RestartDedupMarker,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayDrainKind {
    Shutdown,
    Restart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayDrainRequest {
    pub kind: GatewayDrainKind,
    pub detached_restart: bool,
    pub service_restart: bool,
}

impl GatewayDrainRequest {
    pub fn shutdown() -> Self {
        Self {
            kind: GatewayDrainKind::Shutdown,
            detached_restart: false,
            service_restart: false,
        }
    }

    pub fn restart(detached_restart: bool, service_restart: bool) -> Result<Self, String> {
        let request = Self {
            kind: GatewayDrainKind::Restart,
            detached_restart,
            service_restart,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.kind == GatewayDrainKind::Shutdown
            && (self.detached_restart || self.service_restart)
        {
            return Err("shutdown drain request must not set restart flags".to_string());
        }
        if self.detached_restart && self.service_restart {
            return Err(
                "restart drain request cannot be both detached and service-managed".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayDrainPlan {
    pub request: GatewayDrainRequest,
    pub active_session_keys: Vec<String>,
    pub queue_during_drain: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayShutdownOutcome {
    pub request: GatewayDrainRequest,
    pub timed_out: bool,
    pub interrupted_session_keys: Vec<String>,
    pub marked_resume_pending: Vec<String>,
    pub resume_reason: Option<&'static str>,
    pub write_clean_shutdown_marker: bool,
    pub increment_restart_failure_counts: Vec<String>,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayStartupRecovery {
    pub consume_clean_shutdown_marker: bool,
    pub suspend_recently_active: bool,
}

pub fn queue_during_drain_enabled(restart_requested: bool, busy_input_mode: BusyInputMode) -> bool {
    restart_requested && matches!(busy_input_mode, BusyInputMode::Queue | BusyInputMode::Steer)
}

pub fn plan_startup_recovery(clean_shutdown_marker_exists: bool) -> GatewayStartupRecovery {
    GatewayStartupRecovery {
        consume_clean_shutdown_marker: clean_shutdown_marker_exists,
        suspend_recently_active: !clean_shutdown_marker_exists,
    }
}

fn source_has_user_identity(source: &SessionSource) -> bool {
    source
        .user_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
}

pub fn plan_unauthorized_dm_pairing_response(
    request: &GatewayPairingRequest,
    decision: &GatewayPairingCodeDecision,
) -> Result<Option<GatewayIngressHostResponse>, String> {
    request.validate()?;
    let message = match decision {
        GatewayPairingCodeDecision::Suppress => return Ok(None),
        GatewayPairingCodeDecision::SendPairingCode { code } => format!(
            "Hi~ I don't recognize you yet!\n\nHere's your pairing code: `{code}`\n\nAsk the bot owner to run:\n`hermes pairing approve {} {code}`",
            request.platform.as_str()
        ),
        GatewayPairingCodeDecision::SendTryLater => {
            "Too many pairing requests right now~ Please try again later!".to_string()
        }
    };
    Ok(Some(GatewayIngressHostResponse {
        message,
        reply_to_message_id: None,
        thread_id: None,
    }))
}

#[derive(Debug, Clone)]
pub struct GatewayRuntime {
    pub session_store: SessionStore,
    active_sessions: HashMap<String, ActiveSession>,
    pending_events: HashMap<String, MessageEvent>,
    queued_events: HashMap<String, VecDeque<MessageEvent>>,
    busy_ack_timestamps: HashMap<String, f64>,
    busy_input_mode: BusyInputMode,
    draining: bool,
    queue_during_drain: bool,
    restart_requested: bool,
    restart_detached: bool,
    restart_via_service: bool,
    drain_started_sessions: Vec<String>,
}

impl GatewayRuntime {
    pub fn new(
        sessions_dir: impl Into<PathBuf>,
        config: RuntimeConfig,
        busy_input_mode: BusyInputMode,
    ) -> Result<Self, String> {
        Ok(Self {
            session_store: SessionStore::new(sessions_dir, config)?,
            active_sessions: HashMap::new(),
            pending_events: HashMap::new(),
            queued_events: HashMap::new(),
            busy_ack_timestamps: HashMap::new(),
            busy_input_mode,
            draining: false,
            queue_during_drain: false,
            restart_requested: false,
            restart_detached: false,
            restart_via_service: false,
            drain_started_sessions: Vec::new(),
        })
    }

    pub fn busy_input_mode(&self) -> BusyInputMode {
        self.busy_input_mode
    }

    pub fn set_busy_input_mode(&mut self, busy_input_mode: BusyInputMode) {
        self.busy_input_mode = busy_input_mode;
    }

    pub fn prepare_busy_reply_context(
        &mut self,
        session_key: &str,
        ack_enabled: bool,
        now_seconds: f64,
        cooldown_seconds: f64,
        status: GatewayBusyReplyStatus,
    ) -> Result<GatewayBusyReplyContext, String> {
        let session_key = session_key.trim();
        if session_key.is_empty() {
            return Err("busy reply session key must not be empty".to_string());
        }
        if !now_seconds.is_finite() || now_seconds < 0.0 {
            return Err("busy reply now_seconds must be a non-negative finite number".to_string());
        }
        if !cooldown_seconds.is_finite() || cooldown_seconds < 0.0 {
            return Err(
                "busy reply cooldown_seconds must be a non-negative finite number".to_string(),
            );
        }
        status.validate()?;

        let cooldown_active = if ack_enabled {
            let last_ack = self
                .busy_ack_timestamps
                .get(session_key)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let active = now_seconds - last_ack < cooldown_seconds;
            if !active {
                self.busy_ack_timestamps
                    .insert(session_key.to_string(), now_seconds);
            }
            active
        } else {
            false
        };

        Ok(GatewayBusyReplyContext {
            ack_enabled,
            cooldown_active,
            elapsed_minutes: status.elapsed_minutes,
            api_call_count: status.api_call_count,
            max_iterations: status.max_iterations,
            current_tool: status.current_tool,
            onboarding_hint: status.onboarding_hint,
        })
    }

    pub fn set_draining(&mut self, draining: bool, queue_during_drain: bool) {
        self.draining = draining;
        self.queue_during_drain = queue_during_drain;
        if !draining {
            self.drain_started_sessions.clear();
        }
    }

    pub fn draining(&self) -> bool {
        self.draining
    }

    pub fn restart_requested(&self) -> bool {
        self.restart_requested
    }

    pub fn request_restart(
        &mut self,
        detached_restart: bool,
        service_restart: bool,
    ) -> Result<bool, String> {
        GatewayDrainRequest::restart(detached_restart, service_restart)?;
        if self.restart_requested || self.draining {
            return Ok(false);
        }
        self.restart_requested = true;
        self.restart_detached = detached_restart;
        self.restart_via_service = service_restart;
        Ok(true)
    }

    pub fn begin_shutdown(
        &mut self,
        restart: bool,
        detached_restart: bool,
        service_restart: bool,
    ) -> Result<GatewayDrainPlan, String> {
        let request = if restart {
            GatewayDrainRequest::restart(detached_restart, service_restart)?
        } else {
            let request = GatewayDrainRequest::shutdown();
            request.validate()?;
            request
        };

        if self.draining {
            return Ok(GatewayDrainPlan {
                request: self.current_drain_request(),
                active_session_keys: self.drain_started_sessions.clone(),
                queue_during_drain: self.queue_during_drain,
            });
        }

        self.restart_requested = request.kind == GatewayDrainKind::Restart;
        self.restart_detached = request.detached_restart;
        self.restart_via_service = request.service_restart;
        self.draining = true;
        self.queue_during_drain =
            queue_during_drain_enabled(self.restart_requested, self.busy_input_mode);

        self.drain_started_sessions = self.active_session_keys();
        Ok(GatewayDrainPlan {
            request,
            active_session_keys: self.drain_started_sessions.clone(),
            queue_during_drain: self.queue_during_drain,
        })
    }

    pub fn begin_shutdown_host(
        &mut self,
        restart: bool,
        detached_restart: bool,
        service_restart: bool,
    ) -> Result<GatewayShutdownHostFlow, String> {
        let drain = self.begin_shutdown(restart, detached_restart, service_restart)?;
        let pre_drain = self.plan_shutdown_notification_host()?;
        Ok(GatewayShutdownHostFlow { drain, pre_drain })
    }

    pub fn finish_shutdown(&mut self, timed_out: bool) -> Result<GatewayShutdownOutcome, String> {
        let request = self.current_drain_request();
        let interrupted_session_keys = self.active_session_keys();
        let increment_restart_failure_counts = self.drain_started_sessions.clone();
        let mut marked_resume_pending = Vec::new();
        let mut resume_reason = None;

        if timed_out {
            let reason = if self.restart_requested {
                "restart_timeout"
            } else {
                "shutdown_timeout"
            };
            resume_reason = Some(reason);
            for session_key in &interrupted_session_keys {
                if self
                    .session_store
                    .mark_resume_pending(session_key, reason)?
                {
                    marked_resume_pending.push(session_key.clone());
                }
            }
        }

        let exit_code = (self.restart_requested && self.restart_via_service)
            .then_some(GATEWAY_SERVICE_RESTART_EXIT_CODE);

        self.draining = false;
        self.queue_during_drain = false;
        self.restart_requested = false;
        self.restart_detached = false;
        self.restart_via_service = false;
        self.drain_started_sessions.clear();

        Ok(GatewayShutdownOutcome {
            request,
            timed_out,
            interrupted_session_keys,
            marked_resume_pending,
            resume_reason,
            write_clean_shutdown_marker: !timed_out,
            increment_restart_failure_counts,
            exit_code,
        })
    }

    pub fn execute_startup_host_flow<H: GatewayHostActionHandler>(
        &mut self,
        runtime_dir: &Path,
        handler: &mut H,
    ) -> Result<GatewayStartupExecutionReport, String> {
        let restart_marker = read_restart_notify_marker(runtime_dir)?;
        let flow = plan_startup_host_flow(
            runtime_dir,
            &self.session_store.config,
            clean_shutdown_marker_exists(runtime_dir),
            restart_marker.as_ref(),
        );

        let mut host_actions = execute_host_actions(&flow.pre_actions, handler)?;
        let suspended_recent_sessions = if flow.recovery.suspend_recently_active {
            self.session_store
                .suspend_recently_active(STARTUP_RECENT_ACTIVITY_WINDOW_SECONDS)?
        } else {
            0
        };
        let suspended_stuck_loop_sessions = suspend_stuck_loop_sessions(
            runtime_dir,
            &mut self.session_store,
            STUCK_LOOP_THRESHOLD,
        )?;
        host_actions.merge(execute_startup_restart_flow(&flow.restart, handler)?);

        Ok(GatewayStartupExecutionReport {
            recovery: flow.recovery,
            suspended_recent_sessions,
            suspended_stuck_loop_sessions,
            host_actions,
        })
    }

    pub fn record_successful_turn(
        &mut self,
        runtime_dir: &Path,
        session_key: &str,
    ) -> Result<GatewayTurnCompletionReport, String> {
        let cleared_restart_failure_count = clear_restart_failure_count(runtime_dir, session_key)?;
        let cleared_resume_pending = self.session_store.clear_resume_pending(session_key)?;
        Ok(GatewayTurnCompletionReport {
            cleared_restart_failure_count,
            cleared_resume_pending,
        })
    }

    pub fn complete_turn(
        &mut self,
        runtime_dir: &Path,
        session_key: &str,
    ) -> Result<GatewayTurnFinishReport, String> {
        let completion = self.record_successful_turn(runtime_dir, session_key)?;
        let next = self.finish_turn(session_key)?;
        Ok(GatewayTurnFinishReport { completion, next })
    }

    pub fn ingest_event_host(
        &mut self,
        event: MessageEvent,
        busy: Option<&GatewayBusyReplyContext>,
    ) -> Result<GatewayIngressHostPlan, String> {
        let reply_event = event.clone();
        let decision = self.ingest_event(event)?;
        let response = plan_ingress_host_response(&decision, &reply_event, busy)?;
        let response_on_effect_failure = match &decision {
            GatewayIngressDecision::SteerActive { session_key, .. } => {
                let fallback = GatewayIngressDecision::QueuePending {
                    session_key: session_key.clone(),
                    depth: self.queue_depth(session_key).max(1),
                    reason: PendingReason::Busy,
                };
                plan_ingress_host_response(&fallback, &reply_event, busy)?
            }
            _ => None,
        };
        let effect = plan_ingress_host_effect(&decision, &reply_event);
        Ok(GatewayIngressHostPlan {
            decision,
            response,
            response_on_effect_failure,
            effect,
        })
    }

    pub fn execute_ingress_host_plan<H: GatewayIngressEffectHandler>(
        &mut self,
        plan: GatewayIngressHostPlan,
        handler: &mut H,
    ) -> Result<GatewayIngressExecutionResult, String> {
        let mut decision = plan.decision;
        let mut response = plan.response;
        let effect = plan.effect;
        let mut effect_applied = false;

        match &effect {
            GatewayIngressHostEffect::None => {}
            GatewayIngressHostEffect::InterruptRunningAgent {
                session_key,
                reason,
            } => {
                handler.interrupt_running_agent(session_key, reason)?;
                effect_applied = true;
            }
            GatewayIngressHostEffect::SteerRunningAgent { session_key, text } => {
                let steered = handler.steer_running_agent(session_key, text)?;
                effect_applied = steered;
                if !steered {
                    let GatewayIngressDecision::SteerActive { event, .. } = &decision else {
                        return Err(
                            "steer host effect requires SteerActive ingress decision".to_string()
                        );
                    };
                    let depth = self.store_pending_event(session_key, event.clone(), true);
                    decision = GatewayIngressDecision::QueuePending {
                        session_key: session_key.clone(),
                        depth,
                        reason: PendingReason::Busy,
                    };
                    response = plan.response_on_effect_failure;
                }
            }
        }

        Ok(GatewayIngressExecutionResult {
            decision,
            response,
            effect,
            effect_applied,
        })
    }

    pub fn handle_event_host<H: GatewayIngressEffectHandler>(
        &mut self,
        event: MessageEvent,
        busy: Option<&GatewayBusyReplyContext>,
        handler: &mut H,
    ) -> Result<GatewayIngressExecutionResult, String> {
        let plan = self.ingest_event_host(event, busy)?;
        self.execute_ingress_host_plan(plan, handler)
    }

    pub fn handle_event_host_authorized<H: GatewayIngressEffectHandler>(
        &mut self,
        event: MessageEvent,
        authorized: bool,
        busy: Option<&GatewayBusyReplyContext>,
        handler: &mut H,
    ) -> Result<GatewayHostHandleOutcome, String> {
        let authorization = if event.internal {
            GatewayAuthorizationStatus::Authorized
        } else if !source_has_user_identity(&event.source) {
            GatewayAuthorizationStatus::MissingUserId
        } else if authorized {
            GatewayAuthorizationStatus::Authorized
        } else {
            GatewayAuthorizationStatus::Unauthorized
        };
        self.handle_event_host_with_authorization(
            event,
            authorization,
            UnauthorizedDmBehavior::Ignore,
            busy,
            handler,
        )
    }

    pub fn handle_event_host_with_authorization<H: GatewayIngressEffectHandler>(
        &mut self,
        event: MessageEvent,
        authorization: GatewayAuthorizationStatus,
        unauthorized_dm_behavior: UnauthorizedDmBehavior,
        busy: Option<&GatewayBusyReplyContext>,
        handler: &mut H,
    ) -> Result<GatewayHostHandleOutcome, String> {
        if event.internal {
            return self
                .handle_event_host(event, busy, handler)
                .map(GatewayHostHandleOutcome::Executed);
        }
        if !source_has_user_identity(&event.source)
            || authorization == GatewayAuthorizationStatus::MissingUserId
        {
            return Ok(GatewayHostHandleOutcome::DroppedMissingUserId {
                platform: event.source.platform.clone(),
                chat_id: event.source.chat_id.clone(),
            });
        }
        if authorization == GatewayAuthorizationStatus::Unauthorized {
            let session_key = build_session_key(
                &event.source,
                self.session_store.config.group_sessions_per_user,
                self.session_store.config.thread_sessions_per_user,
            )?;
            if self.active_sessions.contains_key(&session_key) {
                return Ok(GatewayHostHandleOutcome::DroppedUnauthorizedActiveSession {
                    session_key,
                });
            }
            if event.source.chat_type == "dm"
                && unauthorized_dm_behavior == UnauthorizedDmBehavior::Pair
            {
                return Ok(GatewayHostHandleOutcome::RequireUnauthorizedDmPairing {
                    request: GatewayPairingRequest::from_event(&event)?,
                });
            }
            return Ok(GatewayHostHandleOutcome::DroppedUnauthorizedColdPath { session_key });
        }

        self.handle_event_host(event, busy, handler)
            .map(GatewayHostHandleOutcome::Executed)
    }

    pub fn handle_event_host_with_pairing<H: GatewayIngressEffectHandler>(
        &mut self,
        event: MessageEvent,
        authorization: GatewayAuthorizationStatus,
        unauthorized_dm_behavior: UnauthorizedDmBehavior,
        pairing_store: &GatewayPairingStore,
        now_seconds: f64,
        busy: Option<&GatewayBusyReplyContext>,
        handler: &mut H,
    ) -> Result<GatewayHostHandleOutcome, String> {
        let session_key = build_session_key(
            &event.source,
            self.session_store.config.group_sessions_per_user,
            self.session_store.config.thread_sessions_per_user,
        )?;
        match self.handle_event_host_with_authorization(
            event,
            authorization,
            unauthorized_dm_behavior,
            busy,
            handler,
        )? {
            GatewayHostHandleOutcome::RequireUnauthorizedDmPairing { request } => {
                let decision = pairing_store.request_code(&request, now_seconds)?;
                let Some(response) = plan_unauthorized_dm_pairing_response(&request, &decision)?
                else {
                    return Ok(GatewayHostHandleOutcome::DroppedUnauthorizedColdPath {
                        session_key,
                    });
                };
                Ok(GatewayHostHandleOutcome::RepliedUnauthorizedDmPairing {
                    session_key,
                    response,
                })
            }
            other => Ok(other),
        }
    }

    pub fn plan_shutdown_notification_host(&mut self) -> Result<GatewayHostPlan, String> {
        let actions = self
            .plan_shutdown_notifications()?
            .into_iter()
            .map(|notification| GatewayHostAction::SendNotification { notification })
            .collect();
        Ok(GatewayHostPlan {
            reply: None,
            actions,
        })
    }

    pub fn execute_shutdown_post_drain<H: GatewayHostActionHandler>(
        &mut self,
        runtime_dir: &Path,
        timed_out: bool,
        handler: &mut H,
    ) -> Result<GatewayShutdownExecutionReport, String> {
        let outcome = self.finish_shutdown(timed_out)?;
        let restart_failure_counts = if outcome.increment_restart_failure_counts.is_empty() {
            HashMap::new()
        } else {
            update_restart_failure_counts(runtime_dir, &outcome.increment_restart_failure_counts)?
        };
        let post_drain = plan_shutdown_post_drain_host(runtime_dir, &outcome);
        let host_actions = execute_host_actions(&post_drain.actions, handler)?;
        Ok(GatewayShutdownExecutionReport {
            outcome,
            host_actions,
            restart_failure_counts,
        })
    }

    pub fn plan_shutdown_notifications(&mut self) -> Result<Vec<GatewayNotification>, String> {
        let action = if self.restart_requested {
            "restarting"
        } else {
            "shutting down"
        };
        let hint = if self.restart_requested {
            "Your current task will be interrupted. Send any message after restart and I'll try to resume where you left off."
        } else {
            "Your current task will be interrupted."
        };
        let message = format!("⚠️ Gateway {action} — {hint}");

        self.session_store.ensure_loaded()?;
        let mut deduped = std::collections::HashSet::new();
        let mut notifications = Vec::new();

        for session_key in self.active_running_session_keys() {
            let target = self
                .notification_target_for_session_key(&session_key)
                .or_else(|| parse_notification_target_from_session_key(&session_key));
            let Some((platform, chat_id, thread_id)) = target else {
                continue;
            };
            let notification = GatewayNotification {
                platform,
                chat_id,
                thread_id,
                message: message.clone(),
            };
            if deduped.insert(notification.dedupe_key()) {
                notifications.push(notification);
            }
        }

        let mut homes = self
            .session_store
            .config
            .home_channels
            .iter()
            .collect::<Vec<_>>();
        homes.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
        for (platform, home) in homes {
            if home.chat_id.trim().is_empty() {
                continue;
            }
            let notification = GatewayNotification {
                platform: platform.clone(),
                chat_id: home.chat_id.clone(),
                thread_id: home.thread_id.clone(),
                message: message.clone(),
            };
            if deduped.insert(notification.dedupe_key()) {
                notifications.push(notification);
            }
        }

        Ok(notifications)
    }

    pub fn plan_restart_command(
        &mut self,
        event: &MessageEvent,
        under_service: bool,
        now_seconds: f64,
        last_processed: Option<&RestartDedupMarker>,
    ) -> Result<RestartCommandDecision, String> {
        if is_stale_restart_redelivery(event, last_processed, now_seconds) {
            return Ok(RestartCommandDecision::IgnoreRedelivery);
        }

        let active_agents = self.active_running_session_keys().len();
        if self.restart_requested || self.draining {
            return Ok(RestartCommandDecision::AlreadyInProgress {
                message: if active_agents > 0 {
                    format!("⏳ Draining {active_agents} active agent(s) before restart...")
                } else {
                    "⏳ Gateway restart already in progress...".to_string()
                },
            });
        }

        let launch_mode = if under_service {
            self.request_restart(false, true)?;
            RestartLaunchMode::Service
        } else {
            self.request_restart(true, false)?;
            RestartLaunchMode::Detached
        };

        Ok(RestartCommandDecision::BeginRestart {
            launch_mode,
            notify_marker: RestartNotifyMarker::from_event(event)?,
            dedup_marker: RestartDedupMarker::from_event(event, now_seconds),
            message: if active_agents > 0 {
                format!("⏳ Draining {active_agents} active agent(s) before restart...")
            } else {
                "♻ Restarting gateway. If you aren't notified within 60 seconds, restart from the console with `hermes gateway restart`.".to_string()
            },
        })
    }

    pub fn plan_restart_command_host(
        &mut self,
        runtime_dir: &Path,
        event: &MessageEvent,
        under_service: bool,
        now_seconds: f64,
        last_processed: Option<&RestartDedupMarker>,
    ) -> Result<GatewayHostPlan, String> {
        let decision =
            self.plan_restart_command(event, under_service, now_seconds, last_processed)?;
        Ok(match decision {
            RestartCommandDecision::IgnoreRedelivery => GatewayHostPlan {
                reply: Some(String::new()),
                actions: Vec::new(),
            },
            RestartCommandDecision::AlreadyInProgress { message } => GatewayHostPlan {
                reply: Some(message),
                actions: Vec::new(),
            },
            RestartCommandDecision::BeginRestart {
                launch_mode,
                notify_marker,
                dedup_marker,
                message,
            } => GatewayHostPlan {
                reply: Some(message),
                actions: vec![
                    GatewayHostAction::WriteJson {
                        path: gateway_restart_notify_path(runtime_dir),
                        value: serde_json::to_value(notify_marker).map_err(|error| {
                            format!("serializing restart notify marker failed: {error}")
                        })?,
                    },
                    GatewayHostAction::WriteJson {
                        path: gateway_restart_last_processed_path(runtime_dir),
                        value: serde_json::to_value(dedup_marker).map_err(|error| {
                            format!("serializing restart dedup marker failed: {error}")
                        })?,
                    },
                    GatewayHostAction::ScheduleRestart { launch_mode },
                ],
            },
        })
    }

    pub fn active_session(&self, session_key: &str) -> Option<&ActiveSession> {
        self.active_sessions.get(session_key)
    }

    pub fn pending_event(&self, session_key: &str) -> Option<&MessageEvent> {
        self.pending_events.get(session_key)
    }

    pub fn queue_depth(&self, session_key: &str) -> usize {
        let head = usize::from(self.pending_events.contains_key(session_key));
        let overflow = self.queued_events.get(session_key).map_or(0, VecDeque::len);
        head + overflow
    }

    pub fn mark_turn_running(&mut self, session_key: &str, can_steer: bool) {
        self.active_sessions
            .entry(session_key.to_string())
            .and_modify(|state| state.phase = ActiveSessionPhase::Running { can_steer })
            .or_insert(ActiveSession {
                session_key: session_key.to_string(),
                phase: ActiveSessionPhase::Running { can_steer },
            });
    }

    pub fn enqueue_fifo_follow_up(
        &mut self,
        session_key: &str,
        queued_event: MessageEvent,
    ) -> usize {
        if self.pending_events.contains_key(session_key) {
            self.queued_events
                .entry(session_key.to_string())
                .or_default()
                .push_back(queued_event);
        } else {
            self.pending_events
                .insert(session_key.to_string(), queued_event);
        }
        self.queue_depth(session_key)
    }

    pub fn ingest_event(
        &mut self,
        mut event: MessageEvent,
    ) -> Result<GatewayIngressDecision, String> {
        event.coerce_plaintext_gateway_command();
        let session_key = build_session_key(
            &event.source,
            self.session_store.config.group_sessions_per_user,
            self.session_store.config.thread_sessions_per_user,
        )?;

        if self.active_sessions.contains_key(&session_key) {
            let can_steer = matches!(
                self.active_sessions
                    .get(&session_key)
                    .map(|state| state.phase),
                Some(ActiveSessionPhase::Running { can_steer: true })
            );
            return self.handle_active_session_event(session_key, event, can_steer);
        }

        if let Some(command) = event.get_command().as_deref().and_then(resolve_command) {
            if command.gateway_dispatchable {
                return Ok(GatewayIngressDecision::DispatchCommand {
                    session_key,
                    canonical: command.canonical,
                    event,
                });
            }
        }

        if self.draining {
            return Ok(GatewayIngressDecision::Reject {
                message: "⏳ Gateway is restarting and is not accepting new work right now."
                    .to_string(),
            });
        }

        let session = self
            .session_store
            .get_or_create_session(&event.source, false)?;
        self.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::PendingStart,
            },
        );
        Ok(GatewayIngressDecision::StartTurn {
            session_key,
            session,
            event,
        })
    }

    pub fn finish_turn(
        &mut self,
        session_key: &str,
    ) -> Result<Option<GatewayIngressDecision>, String> {
        self.active_sessions.remove(session_key);
        self.busy_ack_timestamps.remove(session_key);
        let pending_event = self.pending_events.remove(session_key);
        let next_pending = self.promote_queued_event(session_key, pending_event);
        let Some(event) = next_pending else {
            return Ok(None);
        };
        let session = self
            .session_store
            .get_or_create_session(&event.source, false)?;
        self.active_sessions.insert(
            session_key.to_string(),
            ActiveSession {
                session_key: session_key.to_string(),
                phase: ActiveSessionPhase::PendingStart,
            },
        );
        Ok(Some(GatewayIngressDecision::StartTurn {
            session_key: session_key.to_string(),
            session,
            event,
        }))
    }

    fn handle_active_session_event(
        &mut self,
        session_key: String,
        event: MessageEvent,
        can_steer: bool,
    ) -> Result<GatewayIngressDecision, String> {
        let state = RunningSessionState {
            draining: self.draining,
            queue_during_drain: self.queue_during_drain,
            busy_input_mode: self.busy_input_mode,
            can_steer,
        };
        match plan_running_session_action(&event, &state) {
            RunningSessionAction::DispatchCommand { canonical } => {
                Ok(GatewayIngressDecision::DispatchCommand {
                    session_key,
                    canonical,
                    event,
                })
            }
            RunningSessionAction::QueuePending => {
                let depth = self.store_pending_event(&session_key, event, true);
                Ok(GatewayIngressDecision::QueuePending {
                    session_key,
                    depth,
                    reason: PendingReason::Busy,
                })
            }
            RunningSessionAction::InterruptAndQueue => {
                let depth = self.store_pending_event(&session_key, event, true);
                Ok(GatewayIngressDecision::QueuePending {
                    session_key,
                    depth,
                    reason: PendingReason::Interrupt,
                })
            }
            RunningSessionAction::SteerActive => {
                Ok(GatewayIngressDecision::SteerActive { session_key, event })
            }
            RunningSessionAction::QueueDuringDrain => {
                let depth = self.store_pending_event(&session_key, event, true);
                Ok(GatewayIngressDecision::QueuePending {
                    session_key,
                    depth,
                    reason: PendingReason::Drain,
                })
            }
            RunningSessionAction::RejectDuringDrain => Ok(GatewayIngressDecision::Reject {
                message: "⏳ Gateway is restarting and is not accepting another turn right now."
                    .to_string(),
            }),
            RunningSessionAction::Reject { message } => {
                Ok(GatewayIngressDecision::Reject { message })
            }
        }
    }

    fn store_pending_event(
        &mut self,
        session_key: &str,
        event: MessageEvent,
        merge_text: bool,
    ) -> usize {
        match self.pending_events.get_mut(session_key) {
            Some(existing)
                if merge_text
                    && existing.message_type == MessageType::Text
                    && event.message_type == MessageType::Text =>
            {
                let existing_text = existing.text.trim();
                let incoming_text = event.text.trim();
                if existing_text.is_empty() {
                    existing.text = event.text;
                } else if !incoming_text.is_empty() {
                    existing.text = format!("{existing_text}\n{incoming_text}");
                }
            }
            Some(existing) => {
                *existing = event;
            }
            None => {
                self.pending_events.insert(session_key.to_string(), event);
            }
        }
        self.queue_depth(session_key)
    }

    fn promote_queued_event(
        &mut self,
        session_key: &str,
        pending_event: Option<MessageEvent>,
    ) -> Option<MessageEvent> {
        let mut overflow = self.queued_events.remove(session_key).unwrap_or_default();
        match pending_event {
            Some(head) => {
                if let Some(next) = overflow.pop_front() {
                    self.pending_events.insert(session_key.to_string(), next);
                }
                if !overflow.is_empty() {
                    self.queued_events.insert(session_key.to_string(), overflow);
                }
                Some(head)
            }
            None => {
                let head = overflow.pop_front();
                if !overflow.is_empty() {
                    self.queued_events.insert(session_key.to_string(), overflow);
                }
                head
            }
        }
    }

    fn active_session_keys(&self) -> Vec<String> {
        let mut session_keys = self.active_sessions.keys().cloned().collect::<Vec<_>>();
        session_keys.sort();
        session_keys
    }

    fn active_running_session_keys(&self) -> Vec<String> {
        let mut session_keys = self
            .active_sessions
            .iter()
            .filter(|(_session_key, state)| {
                matches!(state.phase, ActiveSessionPhase::Running { .. })
            })
            .map(|(session_key, _state)| session_key.clone())
            .collect::<Vec<_>>();
        session_keys.sort();
        session_keys
    }

    fn notification_target_for_session_key(
        &self,
        session_key: &str,
    ) -> Option<(Platform, String, Option<String>)> {
        let entry = self.session_store.entries.get(session_key)?;
        let origin = entry.origin.as_ref()?;
        Some((
            origin.platform.clone(),
            origin.chat_id.clone(),
            origin.thread_id.clone(),
        ))
    }

    fn current_drain_request(&self) -> GatewayDrainRequest {
        if self.restart_requested {
            GatewayDrainRequest {
                kind: GatewayDrainKind::Restart,
                detached_restart: self.restart_detached,
                service_restart: self.restart_via_service,
            }
        } else {
            GatewayDrainRequest::shutdown()
        }
    }
}

pub fn plan_startup_home_channel_notifications(
    config: &RuntimeConfig,
    skip_targets: &std::collections::HashSet<(String, String, Option<String>)>,
) -> Vec<GatewayNotification> {
    let mut notifications = Vec::new();
    let mut delivered = std::collections::HashSet::new();
    let message = "♻️ Gateway online — Hermes is back and ready.".to_string();

    let mut homes = config.home_channels.iter().collect::<Vec<_>>();
    homes.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
    for (platform, home) in homes {
        if home.chat_id.trim().is_empty() {
            continue;
        }
        let notification = GatewayNotification {
            platform: platform.clone(),
            chat_id: home.chat_id.clone(),
            thread_id: home.thread_id.clone(),
            message: message.clone(),
        };
        let key = notification.dedupe_key();
        if skip_targets.contains(&key) || !delivered.insert(key) {
            continue;
        }
        notifications.push(notification);
    }

    notifications
}

pub fn gateway_restart_notify_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(RESTART_NOTIFY_FILENAME)
}

pub fn gateway_restart_last_processed_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(RESTART_LAST_PROCESSED_FILENAME)
}

pub fn gateway_clean_shutdown_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(CLEAN_SHUTDOWN_FILENAME)
}

pub fn gateway_restart_failure_counts_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(RESTART_FAILURE_COUNTS_FILENAME)
}

pub fn restart_notification_pending(runtime_dir: &Path) -> bool {
    gateway_restart_notify_path(runtime_dir).exists()
}

pub fn clean_shutdown_marker_exists(runtime_dir: &Path) -> bool {
    gateway_clean_shutdown_path(runtime_dir).exists()
}

pub fn read_restart_notify_marker(
    runtime_dir: &Path,
) -> Result<Option<RestartNotifyMarker>, String> {
    read_json_marker(&gateway_restart_notify_path(runtime_dir))
}

pub fn read_restart_dedup_marker(runtime_dir: &Path) -> Result<Option<RestartDedupMarker>, String> {
    read_json_marker(&gateway_restart_last_processed_path(runtime_dir))
}

pub fn update_restart_failure_counts(
    runtime_dir: &Path,
    active_session_keys: &[String],
) -> Result<HashMap<String, u64>, String> {
    let mut next = HashMap::new();
    let previous = read_restart_failure_counts(runtime_dir).unwrap_or_default();
    for session_key in active_session_keys {
        let count = previous.get(session_key).copied().unwrap_or(0) + 1;
        next.insert(session_key.clone(), count);
    }
    write_restart_failure_counts(runtime_dir, &next)?;
    Ok(next)
}

pub fn clear_restart_failure_count(runtime_dir: &Path, session_key: &str) -> Result<bool, String> {
    let session_key = session_key.trim();
    if session_key.is_empty() {
        return Err("session key must not be empty".to_string());
    }
    let path = gateway_restart_failure_counts_path(runtime_dir);
    let mut counts = read_restart_failure_counts(runtime_dir).unwrap_or_default();
    let removed = counts.remove(session_key).is_some();
    if !removed {
        return Ok(false);
    }
    if counts.is_empty() {
        if let Err(error) = fs::remove_file(&path) {
            if error.kind() != ErrorKind::NotFound {
                return Err(format!("deleting {} failed: {error}", path.display()));
            }
        }
    } else {
        write_restart_failure_counts(runtime_dir, &counts)?;
    }
    Ok(true)
}

pub fn suspend_stuck_loop_sessions(
    runtime_dir: &Path,
    session_store: &mut SessionStore,
    threshold: u64,
) -> Result<usize, String> {
    if threshold == 0 {
        return Err("stuck-loop threshold must be positive".to_string());
    }
    let counts = read_restart_failure_counts(runtime_dir).unwrap_or_default();
    if counts.is_empty() {
        return Ok(0);
    }
    let mut suspended = 0_usize;
    for session_key in counts
        .iter()
        .filter(|(_session_key, count)| **count >= threshold)
        .map(|(session_key, _count)| session_key.clone())
        .collect::<Vec<_>>()
    {
        if session_store.suspend_session(&session_key)? {
            suspended += 1;
        }
    }
    let path = gateway_restart_failure_counts_path(runtime_dir);
    if let Err(error) = fs::remove_file(&path) {
        if error.kind() != ErrorKind::NotFound {
            return Err(format!("deleting {} failed: {error}", path.display()));
        }
    }
    Ok(suspended)
}

pub fn apply_filesystem_host_actions(
    actions: &[GatewayHostAction],
) -> Result<Vec<GatewayHostAction>, String> {
    let mut deferred = Vec::new();
    for action in actions {
        match action {
            GatewayHostAction::WriteJson { path, value } => {
                let bytes = serde_json::to_vec(value)
                    .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
                atomic_write(path, &bytes)?;
            }
            GatewayHostAction::WriteText { path, content } => {
                atomic_write(path, content.as_bytes())?;
            }
            GatewayHostAction::DeleteFile { path } => {
                if let Err(error) = fs::remove_file(path) {
                    if error.kind() != ErrorKind::NotFound {
                        return Err(format!("deleting {} failed: {error}", path.display()));
                    }
                }
            }
            other => deferred.push(other.clone()),
        }
    }
    Ok(deferred)
}

pub fn execute_host_actions<H: GatewayHostActionHandler>(
    actions: &[GatewayHostAction],
    handler: &mut H,
) -> Result<GatewayHostExecutionReport, String> {
    let mut report = GatewayHostExecutionReport::empty();

    for action in actions {
        match action {
            GatewayHostAction::WriteJson { path, value } => {
                let bytes = serde_json::to_vec(value)
                    .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
                atomic_write(path, &bytes)?;
            }
            GatewayHostAction::WriteText { path, content } => {
                atomic_write(path, content.as_bytes())?;
            }
            GatewayHostAction::DeleteFile { path } => {
                if let Err(error) = fs::remove_file(path) {
                    if error.kind() != ErrorKind::NotFound {
                        return Err(format!("deleting {} failed: {error}", path.display()));
                    }
                }
            }
            GatewayHostAction::SendNotification { notification } => {
                let target = notification.dedupe_key();
                report.attempted_notification_targets.push(target.clone());
                if handler.send_notification(notification)? {
                    report.delivered_notification_targets.push(target);
                } else {
                    report.failed_notification_targets.push(target);
                }
            }
            GatewayHostAction::ScheduleRestart { launch_mode } => {
                handler.schedule_restart(*launch_mode)?;
                report.scheduled_restarts.push(*launch_mode);
            }
        }
    }

    Ok(report)
}

pub fn execute_startup_restart_flow<H: GatewayHostActionHandler>(
    flow: &GatewayStartupRestartFlow,
    handler: &mut H,
) -> Result<GatewayHostExecutionReport, String> {
    let mut report = execute_host_actions(&flow.restart_notification.actions, handler)?;
    let follow_up = if report.delivered_notification_targets.is_empty() {
        &flow.home_notifications_if_not_delivered.actions
    } else {
        &flow.home_notifications_if_delivered.actions
    };
    let follow_up_report = execute_host_actions(follow_up, handler)?;
    report.merge(follow_up_report);
    Ok(report)
}

pub fn plan_restart_notification_host(
    runtime_dir: &Path,
    marker: Option<&RestartNotifyMarker>,
) -> GatewayHostPlan {
    let mut actions = vec![GatewayHostAction::DeleteFile {
        path: gateway_restart_notify_path(runtime_dir),
    }];
    if let Some(marker) = marker {
        actions.insert(
            0,
            GatewayHostAction::SendNotification {
                notification: marker.to_notification(),
            },
        );
    }
    GatewayHostPlan {
        reply: None,
        actions,
    }
}

pub fn plan_startup_restart_flow_host(
    runtime_dir: &Path,
    config: &RuntimeConfig,
    marker: Option<&RestartNotifyMarker>,
) -> GatewayStartupRestartFlow {
    let restart_notification = plan_restart_notification_host(runtime_dir, marker);
    let delivered_target = marker.map(|marker| marker.to_notification().dedupe_key());
    let home_notifications_if_delivered = match delivered_target {
        Some(delivered_target) => {
            plan_startup_restart_home_notifications_host(config, true, Some(delivered_target))
        }
        None => GatewayHostPlan {
            reply: None,
            actions: Vec::new(),
        },
    };
    let home_notifications_if_not_delivered = if marker.is_some() {
        plan_startup_restart_home_notifications_host(config, true, None)
    } else {
        GatewayHostPlan {
            reply: None,
            actions: Vec::new(),
        }
    };

    GatewayStartupRestartFlow {
        restart_notification,
        home_notifications_if_delivered,
        home_notifications_if_not_delivered,
    }
}

pub fn plan_startup_host_flow(
    runtime_dir: &Path,
    config: &RuntimeConfig,
    clean_shutdown_marker_exists: bool,
    restart_marker: Option<&RestartNotifyMarker>,
) -> GatewayStartupHostFlow {
    let recovery = plan_startup_recovery(clean_shutdown_marker_exists);
    let pre_actions = if clean_shutdown_marker_exists {
        vec![GatewayHostAction::DeleteFile {
            path: gateway_clean_shutdown_path(runtime_dir),
        }]
    } else {
        Vec::new()
    };
    let restart = plan_startup_restart_flow_host(runtime_dir, config, restart_marker);
    GatewayStartupHostFlow {
        recovery,
        pre_actions,
        restart,
    }
}

pub fn plan_shutdown_post_drain_host(
    runtime_dir: &Path,
    outcome: &GatewayShutdownOutcome,
) -> GatewayHostPlan {
    let mut actions = Vec::new();
    if outcome.write_clean_shutdown_marker {
        actions.push(GatewayHostAction::WriteText {
            path: gateway_clean_shutdown_path(runtime_dir),
            content: String::new(),
        });
    }
    if outcome.request.kind == GatewayDrainKind::Restart && outcome.request.detached_restart {
        actions.push(GatewayHostAction::ScheduleRestart {
            launch_mode: RestartLaunchMode::Detached,
        });
    }
    GatewayHostPlan {
        reply: None,
        actions,
    }
}

pub fn plan_startup_restart_home_notifications_host(
    config: &RuntimeConfig,
    restart_notification_pending: bool,
    delivered_restart_target: Option<(String, String, Option<String>)>,
) -> GatewayHostPlan {
    if !restart_notification_pending && delivered_restart_target.is_none() {
        return GatewayHostPlan {
            reply: None,
            actions: Vec::new(),
        };
    }
    let skip_targets = delivered_restart_target
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    let actions = plan_startup_home_channel_notifications(config, &skip_targets)
        .into_iter()
        .map(|notification| GatewayHostAction::SendNotification { notification })
        .collect::<Vec<_>>();
    GatewayHostPlan {
        reply: None,
        actions,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamConsumerConfig {
    pub edit_interval: f64,
    pub buffer_threshold: usize,
    pub cursor: String,
    pub buffer_only: bool,
    pub fresh_final_after_seconds: f64,
}

impl Default for StreamConsumerConfig {
    fn default() -> Self {
        Self {
            edit_interval: 1.0,
            buffer_threshold: 40,
            cursor: " ▉".to_string(),
            buffer_only: false,
            fresh_final_after_seconds: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamInput {
    Delta(String),
    SegmentBreak,
    Commentary(String),
    Finish,
    Tick,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOp {
    SendNew { text: String },
    EditCurrent { text: String, finalize: bool },
    SendCommentary { text: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamMessageHandle {
    Editable(String),
    NoEdit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamFollowUpEdit {
    StripCursor { message_id: String, text: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEditFailureOutcome {
    RetryLater,
    EnterFallback {
        strip_cursor: Option<StreamFollowUpEdit>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamFallbackFinalPlan {
    AlreadyDelivered {
        strip_cursor: Option<StreamFollowUpEdit>,
    },
    SendChunks {
        chunks: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamFinishPlan {
    Noop,
    MarkFinalSent,
    Deliver(StreamDeliveryAction),
    Fallback(StreamFallbackFinalPlan),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamSegmentBreakPlan {
    Noop,
    FlushTail {
        text: String,
        strip_cursor: Option<StreamFollowUpEdit>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDeliveryAction {
    Skip,
    Send {
        text: String,
    },
    Edit {
        message_id: String,
        text: String,
        finalize: bool,
    },
    FreshFinal {
        old_message_id: String,
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayStreamExecutionState {
    pub config: StreamConsumerConfig,
    pub max_message_length: usize,
    pub adapter_requires_finalize: bool,
    message_handle: Option<StreamMessageHandle>,
    message_created_at: Option<f64>,
    already_sent: bool,
    final_response_sent: bool,
    edit_supported: bool,
    last_sent_text: String,
    fallback_final_send: bool,
    fallback_prefix: String,
    flood_strikes: usize,
    current_edit_interval: f64,
}

impl GatewayStreamExecutionState {
    const MAX_FLOOD_STRIKES: usize = 3;

    pub fn new(
        config: StreamConsumerConfig,
        max_message_length: usize,
        adapter_requires_finalize: bool,
    ) -> Self {
        let edit_interval = config.edit_interval;
        Self {
            config,
            max_message_length,
            adapter_requires_finalize,
            message_handle: None,
            message_created_at: None,
            already_sent: false,
            final_response_sent: false,
            edit_supported: true,
            last_sent_text: String::new(),
            fallback_final_send: false,
            fallback_prefix: String::new(),
            flood_strikes: 0,
            current_edit_interval: edit_interval,
        }
    }

    pub fn message_handle(&self) -> Option<&StreamMessageHandle> {
        self.message_handle.as_ref()
    }

    pub fn already_sent(&self) -> bool {
        self.already_sent
    }

    pub fn final_response_sent(&self) -> bool {
        self.final_response_sent
    }

    pub fn fallback_final_send(&self) -> bool {
        self.fallback_final_send
    }

    pub fn current_edit_interval(&self) -> f64 {
        self.current_edit_interval
    }

    pub fn visible_prefix(&self) -> String {
        let mut prefix = self.last_sent_text.clone();
        if !self.config.cursor.is_empty() && prefix.ends_with(&self.config.cursor) {
            prefix.truncate(prefix.len().saturating_sub(self.config.cursor.len()));
        }
        GatewayStreamPlanner::clean_for_display(&prefix)
    }

    pub fn continuation_text(&self, final_text: &str) -> String {
        let prefix = if self.fallback_prefix.is_empty() {
            self.visible_prefix()
        } else {
            self.fallback_prefix.clone()
        };
        if !prefix.is_empty() && final_text.starts_with(&prefix) {
            final_text[prefix.len()..].trim_start().to_string()
        } else {
            final_text.to_string()
        }
    }

    pub fn split_text_chunks(text: &str, limit: usize) -> Vec<String> {
        if text.len() <= limit {
            return vec![text.to_string()];
        }
        let mut chunks = Vec::new();
        let mut remaining = text.to_string();
        while remaining.len() > limit {
            let mut split_at = remaining[..limit].rfind('\n').unwrap_or(limit);
            if split_at < limit / 2 {
                split_at = limit;
            }
            chunks.push(remaining[..split_at].to_string());
            remaining = remaining[split_at..].trim_start_matches('\n').to_string();
        }
        if !remaining.is_empty() {
            chunks.push(remaining);
        }
        chunks
    }

    pub fn plan_delivery_action(
        &self,
        text: &str,
        finalize: bool,
        now: f64,
    ) -> StreamDeliveryAction {
        let cleaned = GatewayStreamPlanner::clean_for_display(text);
        let visible_without_cursor = if self.config.cursor.is_empty() {
            cleaned.clone()
        } else {
            cleaned.replace(&self.config.cursor, "")
        };
        if visible_without_cursor.trim().is_empty() || cleaned.trim().is_empty() {
            return StreamDeliveryAction::Skip;
        }

        match self.message_handle.as_ref() {
            None => StreamDeliveryAction::Send { text: cleaned },
            Some(StreamMessageHandle::Editable(message_id)) if self.edit_supported => {
                if cleaned == self.last_sent_text && !(finalize && self.adapter_requires_finalize) {
                    return StreamDeliveryAction::Skip;
                }
                if finalize && self.should_send_fresh_final(now) {
                    return StreamDeliveryAction::FreshFinal {
                        old_message_id: message_id.clone(),
                        text: cleaned,
                    };
                }
                StreamDeliveryAction::Edit {
                    message_id: message_id.clone(),
                    text: cleaned,
                    finalize,
                }
            }
            Some(StreamMessageHandle::Editable(_)) | Some(StreamMessageHandle::NoEdit) => {
                StreamDeliveryAction::Skip
            }
        }
    }

    pub fn record_first_send_success(&mut self, text: &str, message_id: Option<&str>, now: f64) {
        let cleaned = GatewayStreamPlanner::clean_for_display(text);
        self.already_sent = true;
        self.last_sent_text = cleaned;
        self.final_response_sent = false;
        match message_id {
            Some(message_id) if !message_id.trim().is_empty() => {
                self.message_handle = Some(StreamMessageHandle::Editable(message_id.to_string()));
                self.message_created_at = Some(now);
            }
            _ => {
                self.message_handle = Some(StreamMessageHandle::NoEdit);
                self.message_created_at = None;
                self.edit_supported = false;
                self.fallback_prefix = self.visible_prefix();
                self.fallback_final_send = true;
            }
        }
    }

    pub fn record_edit_success(&mut self, text: &str) {
        self.already_sent = true;
        self.last_sent_text = GatewayStreamPlanner::clean_for_display(text);
        self.flood_strikes = 0;
        self.current_edit_interval = self.config.edit_interval;
    }

    pub fn record_edit_failure(&mut self, error: Option<&str>) -> StreamEditFailureOutcome {
        if Self::is_flood_error(error) {
            self.flood_strikes += 1;
            self.current_edit_interval = (self.current_edit_interval * 2.0).min(10.0);
            if self.flood_strikes < Self::MAX_FLOOD_STRIKES {
                return StreamEditFailureOutcome::RetryLater;
            }
        }

        self.fallback_prefix = self.visible_prefix();
        self.fallback_final_send = true;
        self.edit_supported = false;
        self.already_sent = true;
        StreamEditFailureOutcome::EnterFallback {
            strip_cursor: self.strip_cursor_follow_up(),
        }
    }

    pub fn should_send_fresh_final(&self, now: f64) -> bool {
        let threshold = self.config.fresh_final_after_seconds;
        if threshold <= 0.0 {
            return false;
        }
        if !matches!(self.message_handle, Some(StreamMessageHandle::Editable(_))) {
            return false;
        }
        let Some(created_at) = self.message_created_at else {
            return false;
        };
        now - created_at >= threshold
    }

    pub fn record_fresh_final_success(&mut self, text: &str, message_id: Option<&str>, now: f64) {
        let cleaned = GatewayStreamPlanner::clean_for_display(text);
        self.already_sent = true;
        self.final_response_sent = true;
        self.last_sent_text = cleaned;
        self.fallback_final_send = false;
        self.fallback_prefix.clear();
        match message_id {
            Some(message_id) if !message_id.trim().is_empty() => {
                self.message_handle = Some(StreamMessageHandle::Editable(message_id.to_string()));
                self.message_created_at = Some(now);
            }
            _ => {
                self.message_handle = Some(StreamMessageHandle::NoEdit);
                self.message_created_at = None;
                self.edit_supported = false;
            }
        }
    }

    pub fn plan_fallback_final(&mut self, final_text: &str) -> StreamFallbackFinalPlan {
        let final_text = GatewayStreamPlanner::clean_for_display(final_text);
        let mut continuation = self.continuation_text(&final_text);
        self.fallback_final_send = false;
        if continuation.trim().is_empty() {
            if !final_text.trim().is_empty() && final_text != self.visible_prefix() {
                continuation = final_text;
            } else {
                return StreamFallbackFinalPlan::AlreadyDelivered {
                    strip_cursor: self.strip_cursor_follow_up(),
                };
            }
        }
        let safe_limit = self.max_message_length.saturating_sub(100).max(500);
        StreamFallbackFinalPlan::SendChunks {
            chunks: Self::split_text_chunks(&continuation, safe_limit),
        }
    }

    pub fn plan_finish(
        &mut self,
        accumulated: &str,
        current_update_visible: bool,
        now: f64,
    ) -> StreamFinishPlan {
        if accumulated.is_empty() {
            return StreamFinishPlan::Noop;
        }
        if self.fallback_final_send {
            return StreamFinishPlan::Fallback(self.plan_fallback_final(accumulated));
        }
        if current_update_visible && !self.adapter_requires_finalize {
            return StreamFinishPlan::MarkFinalSent;
        }
        if self.message_handle.is_some() {
            return match self.plan_delivery_action(accumulated, true, now) {
                StreamDeliveryAction::Skip => StreamFinishPlan::Noop,
                action => StreamFinishPlan::Deliver(action),
            };
        }
        if !self.already_sent {
            return match self.plan_delivery_action(accumulated, false, now) {
                StreamDeliveryAction::Skip => StreamFinishPlan::Noop,
                action => StreamFinishPlan::Deliver(action),
            };
        }
        StreamFinishPlan::Noop
    }

    pub fn plan_segment_break(
        &self,
        accumulated: &str,
        current_update_visible: bool,
    ) -> StreamSegmentBreakPlan {
        if accumulated.is_empty() || current_update_visible {
            return StreamSegmentBreakPlan::Noop;
        }
        let Some(StreamMessageHandle::Editable(_)) = self.message_handle.as_ref() else {
            return StreamSegmentBreakPlan::Noop;
        };
        let tail = GatewayStreamPlanner::clean_for_display(&self.continuation_text(accumulated));
        if tail.trim().is_empty() {
            return StreamSegmentBreakPlan::Noop;
        }
        StreamSegmentBreakPlan::FlushTail {
            text: tail,
            strip_cursor: (!self.fallback_final_send)
                .then(|| self.strip_cursor_follow_up())
                .flatten(),
        }
    }

    pub fn mark_fallback_delivery_success(
        &mut self,
        last_message_id: Option<&str>,
        last_chunk: Option<&str>,
    ) {
        self.message_handle = last_message_id
            .filter(|message_id| !message_id.trim().is_empty())
            .map(|message_id| StreamMessageHandle::Editable(message_id.to_string()));
        self.already_sent = true;
        self.final_response_sent = true;
        self.last_sent_text = last_chunk.unwrap_or_default().to_string();
        self.fallback_prefix.clear();
        self.fallback_final_send = false;
    }

    pub fn mark_fallback_delivery_failure(
        &mut self,
        sent_any_chunk: bool,
        last_message_id: Option<&str>,
        last_successful_chunk: Option<&str>,
    ) {
        if sent_any_chunk {
            self.message_handle = last_message_id
                .filter(|message_id| !message_id.trim().is_empty())
                .map(|message_id| StreamMessageHandle::Editable(message_id.to_string()));
            self.already_sent = true;
            self.final_response_sent = true;
            self.last_sent_text = last_successful_chunk.unwrap_or_default().to_string();
        } else {
            self.message_handle = None;
            self.already_sent = false;
            self.final_response_sent = false;
            self.last_sent_text.clear();
        }
        self.fallback_prefix.clear();
        self.fallback_final_send = false;
    }

    pub fn reset_segment_state(&mut self, preserve_no_edit: bool) {
        if preserve_no_edit && matches!(self.message_handle, Some(StreamMessageHandle::NoEdit)) {
            return;
        }
        self.message_handle = None;
        self.message_created_at = None;
        self.last_sent_text.clear();
        self.fallback_final_send = false;
        self.fallback_prefix.clear();
    }

    pub fn mark_final_response_sent(&mut self) {
        self.final_response_sent = true;
    }

    pub fn handle_cancellation(&mut self, best_effort_ok: bool) {
        if best_effort_ok && !self.final_response_sent {
            self.final_response_sent = true;
        }
    }

    fn strip_cursor_follow_up(&self) -> Option<StreamFollowUpEdit> {
        let Some(StreamMessageHandle::Editable(message_id)) = self.message_handle.as_ref() else {
            return None;
        };
        let prefix = self.visible_prefix();
        if prefix.trim().is_empty() {
            return None;
        }
        if self.config.cursor.is_empty() || !self.last_sent_text.ends_with(&self.config.cursor) {
            return None;
        }
        Some(StreamFollowUpEdit::StripCursor {
            message_id: message_id.clone(),
            text: prefix,
        })
    }

    fn is_flood_error(error: Option<&str>) -> bool {
        let lowered = error.unwrap_or_default().to_ascii_lowercase();
        lowered.contains("flood") || lowered.contains("retry after") || lowered.contains("rate")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayStreamPlanner {
    pub config: StreamConsumerConfig,
    accumulated: String,
    message_open: bool,
    already_sent: bool,
    last_sent_text: String,
    last_edit_at: f64,
    in_think_block: bool,
    think_buffer: String,
}

impl GatewayStreamPlanner {
    const OPEN_THINK_TAGS: [&'static str; 6] = [
        "<REASONING_SCRATCHPAD>",
        "<think>",
        "<reasoning>",
        "<THINKING>",
        "<thinking>",
        "<thought>",
    ];
    const CLOSE_THINK_TAGS: [&'static str; 6] = [
        "</REASONING_SCRATCHPAD>",
        "</think>",
        "</reasoning>",
        "</THINKING>",
        "</thinking>",
        "</thought>",
    ];

    pub fn new(config: StreamConsumerConfig) -> Self {
        Self {
            config,
            accumulated: String::new(),
            message_open: false,
            already_sent: false,
            last_sent_text: String::new(),
            last_edit_at: 0.0,
            in_think_block: false,
            think_buffer: String::new(),
        }
    }

    pub fn already_sent(&self) -> bool {
        self.already_sent
    }

    pub fn visible_text(&self) -> &str {
        &self.last_sent_text
    }

    pub fn process(&mut self, input: StreamInput, now: f64) -> Vec<StreamOp> {
        let mut ops = Vec::new();
        match input {
            StreamInput::Delta(text) => {
                self.filter_and_accumulate(&text);
                if !self.config.buffer_only && self.should_flush(now) {
                    if let Some(op) = self.flush_accumulated(false) {
                        ops.push(op);
                        self.last_edit_at = now;
                    }
                }
            }
            StreamInput::SegmentBreak => {
                if let Some(op) = self.flush_accumulated(true) {
                    ops.push(op);
                }
                self.reset_segment_state();
            }
            StreamInput::Commentary(text) => {
                if let Some(op) = self.flush_accumulated(true) {
                    ops.push(op);
                }
                let clean = Self::clean_for_display(&text);
                if !clean.trim().is_empty() {
                    self.already_sent = true;
                    self.last_sent_text = clean.clone();
                    ops.push(StreamOp::SendCommentary { text: clean });
                }
                self.reset_segment_state();
            }
            StreamInput::Finish => {
                self.flush_think_buffer();
                if let Some(op) = self.flush_accumulated(true) {
                    ops.push(op);
                }
            }
            StreamInput::Tick => {
                if !self.config.buffer_only && self.should_flush(now) {
                    if let Some(op) = self.flush_accumulated(false) {
                        ops.push(op);
                        self.last_edit_at = now;
                    }
                }
            }
        }
        ops
    }

    pub fn clean_for_display(text: &str) -> String {
        if !text.contains("MEDIA:") && !text.contains("[[audio_as_voice]]") {
            return text.to_string();
        }
        let cleaned = text.replace("[[audio_as_voice]]", "");
        let media_re = Regex::new(r#"([`"']?MEDIA:\s*\S+[`"']?)"#).expect("valid media regex");
        let cleaned = media_re.replace_all(&cleaned, "").to_string();
        let newlines_re = Regex::new(r"\n{3,}").expect("valid newline regex");
        newlines_re
            .replace_all(cleaned.trim_end(), "\n\n")
            .to_string()
    }

    fn should_flush(&self, now: f64) -> bool {
        (!self.accumulated.is_empty())
            && ((now - self.last_edit_at) >= self.config.edit_interval
                || self.accumulated.len() >= self.config.buffer_threshold)
    }

    fn flush_accumulated(&mut self, finalize: bool) -> Option<StreamOp> {
        if self.accumulated.is_empty() {
            return None;
        }
        let mut display_text = self.accumulated.clone();
        if !finalize && !self.config.cursor.is_empty() {
            display_text.push_str(&self.config.cursor);
        }
        self.plan_send_or_edit(display_text, finalize)
    }

    fn plan_send_or_edit(&mut self, text: String, finalize: bool) -> Option<StreamOp> {
        let visible_without_cursor = if !self.config.cursor.is_empty() {
            text.replace(&self.config.cursor, "")
        } else {
            text.clone()
        };
        if visible_without_cursor.trim().is_empty() {
            return None;
        }

        let clean = Self::clean_for_display(&text);
        if clean.trim().is_empty() {
            return None;
        }

        if !self.message_open {
            if !finalize
                && !self.config.cursor.is_empty()
                && text.contains(&self.config.cursor)
                && visible_without_cursor.trim().chars().count() < 4
            {
                return None;
            }
            self.message_open = true;
            self.already_sent = true;
            self.last_sent_text = clean.clone();
            return Some(StreamOp::SendNew { text: clean });
        }

        self.already_sent = true;
        self.last_sent_text = clean.clone();
        Some(StreamOp::EditCurrent {
            text: clean,
            finalize,
        })
    }

    fn reset_segment_state(&mut self) {
        self.message_open = false;
        self.accumulated.clear();
        self.last_sent_text.clear();
    }

    fn filter_and_accumulate(&mut self, text: &str) {
        let mut buf = format!("{}{}", self.think_buffer, text);
        self.think_buffer.clear();

        while !buf.is_empty() {
            if self.in_think_block {
                let mut best_idx = None::<usize>;
                let mut best_len = 0_usize;
                for tag in Self::CLOSE_THINK_TAGS {
                    if let Some(idx) = buf.find(tag) {
                        if best_idx.is_none_or(|current| idx < current) {
                            best_idx = Some(idx);
                            best_len = tag.len();
                        }
                    }
                }
                if let Some(idx) = best_idx {
                    self.in_think_block = false;
                    buf = buf[idx + best_len..].to_string();
                } else {
                    let max_tag = Self::CLOSE_THINK_TAGS
                        .iter()
                        .map(|tag| tag.len())
                        .max()
                        .unwrap_or_default();
                    self.think_buffer = if buf.len() > max_tag {
                        buf[buf.len() - max_tag..].to_string()
                    } else {
                        buf
                    };
                    return;
                }
            } else {
                let mut best_idx = None::<usize>;
                let mut best_len = 0_usize;
                for tag in Self::OPEN_THINK_TAGS {
                    let mut search_start = 0_usize;
                    while let Some(rel_idx) = buf[search_start..].find(tag) {
                        let idx = search_start + rel_idx;
                        let is_boundary = if idx == 0 {
                            self.accumulated.is_empty() || self.accumulated.ends_with('\n')
                        } else {
                            let preceding = &buf[..idx];
                            match preceding.rfind('\n') {
                                Some(last_nl) => preceding[last_nl + 1..].trim().is_empty(),
                                None => {
                                    (self.accumulated.is_empty()
                                        || self.accumulated.ends_with('\n'))
                                        && preceding.trim().is_empty()
                                }
                            }
                        };
                        if is_boundary && best_idx.is_none_or(|current| idx < current) {
                            best_idx = Some(idx);
                            best_len = tag.len();
                            break;
                        }
                        search_start = idx + 1;
                    }
                }

                if let Some(idx) = best_idx {
                    self.accumulated.push_str(&buf[..idx]);
                    self.in_think_block = true;
                    buf = buf[idx + best_len..].to_string();
                } else {
                    let mut held_back = 0_usize;
                    for tag in Self::OPEN_THINK_TAGS {
                        for i in 1..tag.len() {
                            if buf.ends_with(&tag[..i]) && i > held_back {
                                held_back = i;
                            }
                        }
                    }
                    if held_back > 0 {
                        self.accumulated
                            .push_str(&buf[..buf.len().saturating_sub(held_back)]);
                        self.think_buffer = buf[buf.len() - held_back..].to_string();
                    } else {
                        self.accumulated.push_str(&buf);
                    }
                    return;
                }
            }
        }
    }

    fn flush_think_buffer(&mut self) {
        if !self.think_buffer.is_empty() && !self.in_think_block {
            self.accumulated.push_str(&self.think_buffer);
            self.think_buffer.clear();
        }
    }
}

pub fn plan_active_session_ingress(event: &MessageEvent) -> ActiveSessionIngressAction {
    match event.get_command().as_deref().and_then(resolve_command) {
        Some(command) => ActiveSessionIngressAction::DispatchCommand {
            canonical: command.canonical,
        },
        None => ActiveSessionIngressAction::QueuePending,
    }
}

pub fn plan_running_session_action(
    event: &MessageEvent,
    state: &RunningSessionState,
) -> RunningSessionAction {
    if let Some(command) = event.get_command().as_deref().and_then(resolve_command) {
        return match command.canonical {
            "status" | "restart" | "stop" | "new" | "queue" | "steer" | "approve" | "deny"
            | "agents" | "background" | "kanban" | "yolo" | "verbose" | "help" | "commands"
            | "profile" | "update" => RunningSessionAction::DispatchCommand {
                canonical: command.canonical,
            },
            "model" => RunningSessionAction::Reject {
                message: "Agent is running — wait or /stop first, then switch models.".to_string(),
            },
            "goal" => {
                let goal_arg = event.get_command_args().trim().to_ascii_lowercase();
                if goal_arg.is_empty()
                    || matches!(
                        goal_arg.as_str(),
                        "status" | "pause" | "resume" | "clear" | "stop" | "done"
                    )
                {
                    RunningSessionAction::DispatchCommand { canonical: "goal" }
                } else {
                    RunningSessionAction::Reject {
                        message: "Agent is running — use /goal status / pause / clear mid-run, or /stop before setting a new goal.".to_string(),
                    }
                }
            }
            _ => RunningSessionAction::Reject {
                message: format!(
                    "⏳ Agent is running — `/{}` can't run mid-turn. Wait for the current response or `/stop` first.",
                    command.canonical
                ),
            },
        };
    }

    if state.draining {
        return if state.queue_during_drain {
            RunningSessionAction::QueueDuringDrain
        } else {
            RunningSessionAction::RejectDuringDrain
        };
    }

    match state.busy_input_mode {
        BusyInputMode::Queue => RunningSessionAction::QueuePending,
        BusyInputMode::Steer => {
            if state.can_steer && !event.text.trim().is_empty() {
                RunningSessionAction::SteerActive
            } else {
                RunningSessionAction::QueuePending
            }
        }
        BusyInputMode::Interrupt => RunningSessionAction::InterruptAndQueue,
    }
}

pub fn plan_ingress_host_response(
    decision: &GatewayIngressDecision,
    event: &MessageEvent,
    busy: Option<&GatewayBusyReplyContext>,
) -> Result<Option<GatewayIngressHostResponse>, String> {
    if event.source.chat_id.trim().is_empty() {
        return Err("event source chat_id must not be empty".to_string());
    }
    if let Some(busy) = busy {
        busy.validate()?;
    }

    let message = match decision {
        GatewayIngressDecision::DispatchCommand { .. }
        | GatewayIngressDecision::StartTurn { .. } => None,
        GatewayIngressDecision::Reject { message } => Some(message.clone()),
        GatewayIngressDecision::SteerActive { .. } => {
            let Some(busy) = busy else {
                return Ok(None);
            };
            if !busy.ack_enabled || busy.cooldown_active {
                return Ok(None);
            }
            let mut message = format!(
                "⏩ Steered into current run{}. Your message arrives after the next tool call.",
                busy.status_detail()
            );
            if let Some(hint) = busy.onboarding_hint.as_deref() {
                message.push_str("\n\n");
                message.push_str(hint);
            }
            Some(message)
        }
        GatewayIngressDecision::QueuePending { reason, .. } => match reason {
            PendingReason::Drain => Some(
                "⏳ Gateway restarting — queued for the next turn after it comes back.".to_string(),
            ),
            PendingReason::ExplicitQueue => Some("Queued for the next turn.".to_string()),
            PendingReason::Busy | PendingReason::Interrupt => {
                let Some(busy) = busy else {
                    return Ok(None);
                };
                if !busy.ack_enabled || busy.cooldown_active {
                    return Ok(None);
                }
                let status_detail = busy.status_detail();
                let mut message = if *reason == PendingReason::Busy {
                    format!(
                        "⏳ Queued for the next turn{status_detail}. I'll respond once the current task finishes."
                    )
                } else {
                    format!(
                        "⚡ Interrupting current task{status_detail}. I'll respond to your message shortly."
                    )
                };
                if let Some(hint) = busy.onboarding_hint.as_deref() {
                    message.push_str("\n\n");
                    message.push_str(hint);
                }
                Some(message)
            }
        },
    };

    Ok(message.map(|message| GatewayIngressHostResponse {
        message,
        reply_to_message_id: event.message_id.clone(),
        thread_id: event.source.thread_id.clone(),
    }))
}

pub fn plan_ingress_host_effect(
    decision: &GatewayIngressDecision,
    event: &MessageEvent,
) -> GatewayIngressHostEffect {
    match decision {
        GatewayIngressDecision::QueuePending {
            session_key,
            reason: PendingReason::Interrupt,
            ..
        } => GatewayIngressHostEffect::InterruptRunningAgent {
            session_key: session_key.clone(),
            reason: event.text.clone(),
        },
        GatewayIngressDecision::SteerActive { session_key, event } => {
            GatewayIngressHostEffect::SteerRunningAgent {
                session_key: session_key.clone(),
                text: event.text.clone(),
            }
        }
        _ => GatewayIngressHostEffect::None,
    }
}

fn normalize_command_name(name: &str) -> Option<String> {
    let normalized = name.trim().trim_start_matches('/').to_ascii_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

fn parse_notification_target_from_session_key(
    session_key: &str,
) -> Option<(Platform, String, Option<String>)> {
    let parts = session_key.split(':').collect::<Vec<_>>();
    if parts.len() < 5
        || parts.first().copied() != Some("agent")
        || parts.get(1).copied() != Some("main")
    {
        return None;
    }
    let platform = Platform::parse(parts.get(2).copied()?).ok()?;
    let chat_type = parts.get(3).copied()?;
    let chat_id = parts.get(4).copied()?.to_string();
    if chat_id.trim().is_empty() {
        return None;
    }
    let thread_id = if chat_type == "dm" {
        parts.get(5).map(|value| (*value).to_string())
    } else {
        None
    };
    Some((platform, chat_id, thread_id))
}

pub fn is_stale_restart_redelivery(
    event: &MessageEvent,
    marker: Option<&RestartDedupMarker>,
    now_seconds: f64,
) -> bool {
    let Some(marker) = marker else {
        return false;
    };
    if event.platform_update_id.is_none() {
        return false;
    }
    if event.source.platform.as_str() != "telegram" || marker.platform.as_str() != "telegram" {
        return false;
    }
    let Some(recorded_uid) = marker.update_id else {
        return false;
    };
    if now_seconds - marker.requested_at > 300.0 {
        return false;
    }
    i64::from(event.platform_update_id.unwrap_or_default()) <= recorded_uid
}

fn matches_plaintext_gateway_restart(text: &str) -> bool {
    let cleaned = text
        .trim_end_matches(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '.' | '!' | '?'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    matches!(
        cleaned.as_str(),
        "restart gateway"
            | "please restart gateway"
            | "restart the gateway"
            | "please restart the gateway"
            | "restart the hermes gateway"
            | "please restart the hermes gateway"
            | "restart hermes"
            | "please restart hermes"
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionContext {
    pub source: SessionSource,
    #[serde(default)]
    pub connected_platforms: Vec<Platform>,
    #[serde(default)]
    pub home_channels: HashMap<Platform, HomeChannel>,
    #[serde(default)]
    pub shared_multi_user_session: bool,
    #[serde(default)]
    pub session_key: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default, with = "option_datetime_serde")]
    pub created_at: Option<NaiveDateTime>,
    #[serde(default, with = "option_datetime_serde")]
    pub updated_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    pub session_key: String,
    pub session_id: String,
    #[serde(with = "datetime_serde")]
    pub created_at: NaiveDateTime,
    #[serde(with = "datetime_serde")]
    pub updated_at: NaiveDateTime,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<SessionSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<Platform>,
    #[serde(default = "default_chat_type")]
    pub chat_type: String,
    #[serde(default)]
    pub input_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub cache_read_tokens: i64,
    #[serde(default)]
    pub cache_write_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub estimated_cost_usd: f64,
    #[serde(default = "default_cost_status")]
    pub cost_status: String,
    #[serde(default)]
    pub last_prompt_tokens: i64,
    #[serde(default)]
    pub was_auto_reset: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_reset_reason: Option<String>,
    #[serde(default)]
    pub reset_had_activity: bool,
    #[serde(default)]
    pub is_fresh_reset: bool,
    #[serde(default)]
    pub expiry_finalized: bool,
    #[serde(default)]
    pub suspended: bool,
    #[serde(default)]
    pub resume_pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_reason: Option<String>,
    #[serde(default, with = "option_datetime_serde")]
    pub last_resume_marked_at: Option<NaiveDateTime>,
}

fn default_cost_status() -> String {
    "unknown".to_string()
}

impl SessionEntry {
    pub fn validate(&self) -> Result<(), String> {
        if self.session_key.trim().is_empty() {
            return Err("session_key must not be empty".to_string());
        }
        if self.session_id.trim().is_empty() {
            return Err("session_id must not be empty".to_string());
        }
        if self.chat_type.trim().is_empty() {
            return Err("chat_type must not be empty".to_string());
        }
        Ok(())
    }
}

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

pub fn build_session_key(
    source: &SessionSource,
    group_sessions_per_user: bool,
    thread_sessions_per_user: bool,
) -> Result<String, String> {
    source.validate()?;
    let platform = source.platform.as_str();
    if source.chat_type == "dm" {
        if !source.chat_id.is_empty() {
            if let Some(thread_id) = source.thread_id.as_deref() {
                return Ok(format!(
                    "agent:main:{platform}:dm:{}:{thread_id}",
                    source.chat_id
                ));
            }
            return Ok(format!("agent:main:{platform}:dm:{}", source.chat_id));
        }
        if let Some(thread_id) = source.thread_id.as_deref() {
            return Ok(format!("agent:main:{platform}:dm:{thread_id}"));
        }
        return Ok(format!("agent:main:{platform}:dm"));
    }

    let participant_id = source
        .user_id_alt
        .as_deref()
        .or(source.user_id.as_deref())
        .map(str::to_string);

    let mut key_parts = vec![
        "agent:main".to_string(),
        platform.to_string(),
        source.chat_type.clone(),
    ];
    if !source.chat_id.is_empty() {
        key_parts.push(source.chat_id.clone());
    }
    if let Some(thread_id) = source.thread_id.as_deref() {
        key_parts.push(thread_id.to_string());
    }

    let isolate_user = if source.thread_id.is_some() && !thread_sessions_per_user {
        false
    } else {
        group_sessions_per_user
    };
    if isolate_user {
        if let Some(participant_id) = participant_id {
            key_parts.push(participant_id);
        }
    }
    Ok(key_parts.join(":"))
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    sessions_dir: PathBuf,
    pub config: RuntimeConfig,
    entries: HashMap<String, SessionEntry>,
    loaded: bool,
}

impl SessionStore {
    pub fn new(sessions_dir: impl Into<PathBuf>, config: RuntimeConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            sessions_dir: sessions_dir.into(),
            config,
            entries: HashMap::new(),
            loaded: false,
        })
    }

    pub fn sessions_path(&self) -> PathBuf {
        self.sessions_dir.join("sessions.json")
    }

    pub fn entry(&mut self, session_key: &str) -> Result<Option<&SessionEntry>, String> {
        self.ensure_loaded()?;
        Ok(self.entries.get(session_key))
    }

    pub fn entries_len(&mut self) -> Result<usize, String> {
        self.ensure_loaded()?;
        Ok(self.entries.len())
    }

    pub fn get_or_create_session(
        &mut self,
        source: &SessionSource,
        force_new: bool,
    ) -> Result<SessionEntry, String> {
        self.get_or_create_session_at(source, force_new, Local::now().naive_local())
    }

    pub fn get_or_create_session_at(
        &mut self,
        source: &SessionSource,
        force_new: bool,
        now: NaiveDateTime,
    ) -> Result<SessionEntry, String> {
        source.validate()?;
        self.ensure_loaded()?;
        let session_key = build_session_key(
            source,
            self.config.group_sessions_per_user,
            self.config.thread_sessions_per_user,
        )?;

        let mut reset_reason = None::<String>;
        if !force_new {
            if let Some(entry) = self.entries.get_mut(&session_key) {
                if entry.suspended {
                    reset_reason = Some("suspended".to_string());
                } else if entry.resume_pending {
                    entry.updated_at = now;
                    let resumed = entry.clone();
                    let _ = entry;
                    self.save()?;
                    return Ok(resumed);
                } else if let Some(reason) = self
                    .config
                    .get_reset_policy(Some(&source.platform), Some(source.chat_type.as_str()))
                    .reset_reason(entry.updated_at, now)
                {
                    reset_reason = Some(reason.to_string());
                } else {
                    entry.updated_at = now;
                    let current = entry.clone();
                    let _ = entry;
                    self.save()?;
                    return Ok(current);
                }
            }
        }

        let previous = self.entries.get(&session_key).cloned();
        let entry = SessionEntry {
            session_key: session_key.clone(),
            session_id: make_session_id(now),
            created_at: now,
            updated_at: now,
            origin: Some(source.clone()),
            display_name: source.chat_name.clone(),
            platform: Some(source.platform.clone()),
            chat_type: source.chat_type.clone(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            total_tokens: 0,
            estimated_cost_usd: 0.0,
            cost_status: default_cost_status(),
            last_prompt_tokens: 0,
            was_auto_reset: reset_reason.is_some(),
            auto_reset_reason: reset_reason.clone(),
            reset_had_activity: previous
                .as_ref()
                .is_some_and(|value| value.total_tokens > 0),
            is_fresh_reset: false,
            expiry_finalized: false,
            suspended: false,
            resume_pending: false,
            resume_reason: None,
            last_resume_marked_at: None,
        };
        self.entries.insert(session_key, entry.clone());
        self.save()?;
        Ok(entry)
    }

    pub fn update_session(
        &mut self,
        session_key: &str,
        last_prompt_tokens: Option<i64>,
    ) -> Result<(), String> {
        self.update_session_at(session_key, last_prompt_tokens, Local::now().naive_local())
    }

    pub fn update_session_at(
        &mut self,
        session_key: &str,
        last_prompt_tokens: Option<i64>,
        now: NaiveDateTime,
    ) -> Result<(), String> {
        self.ensure_loaded()?;
        if let Some(entry) = self.entries.get_mut(session_key) {
            entry.updated_at = now;
            if let Some(last_prompt_tokens) = last_prompt_tokens {
                entry.last_prompt_tokens = last_prompt_tokens;
            }
            self.save()?;
        }
        Ok(())
    }

    pub fn suspend_session(&mut self, session_key: &str) -> Result<bool, String> {
        self.ensure_loaded()?;
        let updated = self
            .entries
            .get_mut(session_key)
            .map(|entry| {
                entry.suspended = true;
            })
            .is_some();
        if updated {
            self.save()?;
        }
        Ok(updated)
    }

    pub fn mark_resume_pending(&mut self, session_key: &str, reason: &str) -> Result<bool, String> {
        let reason = reason.trim();
        if reason.is_empty() {
            return Err("resume reason must not be empty".to_string());
        }
        self.ensure_loaded()?;
        let now = Local::now().naive_local();
        let updated = self
            .entries
            .get_mut(session_key)
            .map(|entry| {
                if entry.suspended {
                    return false;
                }
                entry.resume_pending = true;
                entry.resume_reason = Some(reason.to_string());
                entry.last_resume_marked_at = Some(now);
                true
            })
            .unwrap_or(false);
        if updated {
            self.save()?;
        }
        Ok(updated)
    }

    pub fn clear_resume_pending(&mut self, session_key: &str) -> Result<bool, String> {
        self.ensure_loaded()?;
        let updated = self
            .entries
            .get_mut(session_key)
            .map(|entry| {
                if !entry.resume_pending {
                    return false;
                }
                entry.resume_pending = false;
                entry.resume_reason = None;
                entry.last_resume_marked_at = None;
                true
            })
            .unwrap_or(false);
        if updated {
            self.save()?;
        }
        Ok(updated)
    }

    pub fn prune_old_entries(&mut self, max_age_days: i64) -> Result<usize, String> {
        self.prune_old_entries_at(max_age_days, Local::now().naive_local())
    }

    pub fn prune_old_entries_at(
        &mut self,
        max_age_days: i64,
        now: NaiveDateTime,
    ) -> Result<usize, String> {
        if max_age_days <= 0 {
            return Ok(0);
        }
        self.ensure_loaded()?;
        let cutoff = now - Duration::days(max_age_days);
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.suspended || entry.updated_at >= cutoff);
        let removed = before.saturating_sub(self.entries.len());
        if removed > 0 {
            self.save()?;
        }
        Ok(removed)
    }

    pub fn suspend_recently_active(&mut self, max_age_seconds: i64) -> Result<usize, String> {
        self.suspend_recently_active_at(max_age_seconds, Local::now().naive_local())
    }

    pub fn suspend_recently_active_at(
        &mut self,
        max_age_seconds: i64,
        now: NaiveDateTime,
    ) -> Result<usize, String> {
        self.ensure_loaded()?;
        let cutoff = now - Duration::seconds(max_age_seconds);
        let mut count = 0_usize;
        for entry in self.entries.values_mut() {
            if entry.resume_pending || entry.suspended || entry.updated_at < cutoff {
                continue;
            }
            entry.resume_pending = true;
            entry.resume_reason = Some("restart_interrupted".to_string());
            entry.last_resume_marked_at = Some(now);
            count += 1;
        }
        if count > 0 {
            self.save()?;
        }
        Ok(count)
    }

    fn ensure_loaded(&mut self) -> Result<(), String> {
        if self.loaded {
            return Ok(());
        }
        fs::create_dir_all(&self.sessions_dir)
            .map_err(|error| format!("creating {} failed: {error}", self.sessions_dir.display()))?;
        let path = self.sessions_path();
        if path.exists() {
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
            if !text.trim().is_empty() {
                let raw = serde_json::from_str::<serde_json::Value>(&text)
                    .map_err(|error| format!("parsing {} failed: {error}", path.display()))?;
                self.entries = deserialize_entry_map(raw)?;
            }
        }
        self.loaded = true;
        Ok(())
    }

    fn save(&self) -> Result<(), String> {
        fs::create_dir_all(&self.sessions_dir)
            .map_err(|error| format!("creating {} failed: {error}", self.sessions_dir.display()))?;
        let path = self.sessions_path();
        let data = serde_json::to_vec_pretty(&self.entries)
            .map_err(|error| format!("serializing session store failed: {error}"))?;
        atomic_write(&path, &data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryTarget {
    pub platform: Platform,
    pub chat_id: Option<String>,
    pub thread_id: Option<String>,
    pub is_origin: bool,
    pub is_explicit: bool,
}

impl DeliveryTarget {
    pub fn parse(target: &str, origin: Option<&SessionSource>) -> Result<Self, String> {
        let target = target.trim();
        if target.is_empty() {
            return Err("delivery target must not be empty".to_string());
        }
        let lowered = target.to_ascii_lowercase();
        if lowered == "origin" {
            return Ok(match origin {
                Some(origin) => Self {
                    platform: origin.platform.clone(),
                    chat_id: Some(origin.chat_id.clone()),
                    thread_id: origin.thread_id.clone(),
                    is_origin: true,
                    is_explicit: false,
                },
                None => Self {
                    platform: Platform::local(),
                    chat_id: None,
                    thread_id: None,
                    is_origin: true,
                    is_explicit: false,
                },
            });
        }
        if lowered == "local" {
            return Ok(Self {
                platform: Platform::local(),
                chat_id: None,
                thread_id: None,
                is_origin: false,
                is_explicit: false,
            });
        }
        if let Some((platform, rest)) = target.split_once(':') {
            let Some(platform) = parse_delivery_platform(platform) else {
                return Ok(Self {
                    platform: Platform::local(),
                    chat_id: None,
                    thread_id: None,
                    is_origin: false,
                    is_explicit: false,
                });
            };
            if rest.trim().is_empty() {
                return Err("delivery target chat_id must not be empty".to_string());
            }
            let mut parts = rest.splitn(2, ':');
            let chat_id = parts.next().unwrap_or_default().trim().to_string();
            if chat_id.is_empty() {
                return Err("delivery target chat_id must not be empty".to_string());
            }
            let thread_id = parts
                .next()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            return Ok(Self {
                platform,
                chat_id: Some(chat_id),
                thread_id,
                is_origin: false,
                is_explicit: true,
            });
        }
        Ok(Self {
            platform: parse_delivery_platform(target).unwrap_or_else(Platform::local),
            chat_id: None,
            thread_id: None,
            is_origin: false,
            is_explicit: false,
        })
    }

    pub fn resolve(&self, config: &RuntimeConfig) -> Result<Self, String> {
        if self.platform.is_local() || self.chat_id.is_some() {
            return Ok(self.clone());
        }
        let home = config
            .get_home_channel(&self.platform)
            .ok_or_else(|| format!("no home channel configured for {}", self.platform))?;
        Ok(Self {
            platform: self.platform.clone(),
            chat_id: Some(home.chat_id.clone()),
            thread_id: home.thread_id.clone(),
            is_origin: self.is_origin,
            is_explicit: false,
        })
    }

    pub fn to_string(&self) -> String {
        if self.is_origin {
            return "origin".to_string();
        }
        if self.platform.is_local() {
            return "local".to_string();
        }
        match (&self.chat_id, &self.thread_id) {
            (Some(chat_id), Some(thread_id)) => {
                format!("{}:{chat_id}:{thread_id}", self.platform)
            }
            (Some(chat_id), None) => format!("{}:{chat_id}", self.platform),
            (None, _) => self.platform.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOp {
    WriteMarkdown {
        path: PathBuf,
        content: String,
    },
    WriteText {
        path: PathBuf,
        content: String,
    },
    PlatformSend {
        target: DeliveryTarget,
        content: String,
        metadata: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryPlan {
    pub target_label: String,
    pub ops: Vec<DeliveryOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayDeliveryPlanner {
    output_dir: PathBuf,
}

impl GatewayDeliveryPlanner {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    pub fn plan_delivery_at(
        &self,
        config: &RuntimeConfig,
        content: &str,
        targets: &[DeliveryTarget],
        job_id: Option<&str>,
        job_name: Option<&str>,
        metadata: &HashMap<String, String>,
        now: NaiveDateTime,
    ) -> Result<Vec<DeliveryPlan>, String> {
        if targets.is_empty() {
            return Err("delivery targets must not be empty".to_string());
        }

        let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
        let display_timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();
        let safe_job_id = sanitize_path_component(job_id.unwrap_or_default());
        let job_id = job_id
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let job_name = job_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);

        let mut plans = Vec::with_capacity(targets.len());
        for target in targets {
            if target.platform.is_local() {
                let path = if job_id.is_some() {
                    self.output_dir
                        .join(&safe_job_id)
                        .join(format!("{timestamp}.md"))
                } else {
                    self.output_dir.join("misc").join(format!("{timestamp}.md"))
                };
                let document = self.build_local_document(
                    content,
                    job_id.as_deref(),
                    job_name.as_deref(),
                    metadata,
                    &display_timestamp,
                );
                plans.push(DeliveryPlan {
                    target_label: target.to_string(),
                    ops: vec![DeliveryOp::WriteMarkdown {
                        path,
                        content: document,
                    }],
                });
                continue;
            }

            let resolved_target = target.resolve(config)?;
            let mut send_metadata = metadata.clone();
            if let Some(thread_id) = resolved_target.thread_id.as_deref() {
                send_metadata
                    .entry("thread_id".to_string())
                    .or_insert_with(|| thread_id.to_string());
            }

            let mut ops = Vec::new();
            let send_content = if content.len() > MAX_PLATFORM_OUTPUT {
                let truncated_job_id =
                    sanitize_path_component(job_id.as_deref().unwrap_or("unknown"));
                let full_output_path = self
                    .output_dir
                    .join(format!("{truncated_job_id}_{timestamp}.txt"));
                ops.push(DeliveryOp::WriteText {
                    path: full_output_path.clone(),
                    content: content.to_string(),
                });
                format!(
                    "{}\n\n... [truncated, full output saved to {}]",
                    slice_prefix(content, TRUNCATED_VISIBLE),
                    full_output_path.display()
                )
            } else {
                content.to_string()
            };
            ops.push(DeliveryOp::PlatformSend {
                target: resolved_target,
                content: send_content,
                metadata: send_metadata,
            });
            plans.push(DeliveryPlan {
                target_label: target.to_string(),
                ops,
            });
        }
        Ok(plans)
    }

    fn build_local_document(
        &self,
        content: &str,
        job_id: Option<&str>,
        job_name: Option<&str>,
        metadata: &HashMap<String, String>,
        display_timestamp: &str,
    ) -> String {
        let mut lines = Vec::new();
        lines.push(format!("# {}", job_name.unwrap_or("Delivery Output")));
        lines.push(String::new());
        lines.push(format!("**Timestamp:** {display_timestamp}"));
        if let Some(job_id) = job_id {
            lines.push(format!("**Job ID:** {job_id}"));
        }
        let mut metadata_keys = metadata.keys().collect::<Vec<_>>();
        metadata_keys.sort();
        for key in metadata_keys {
            if let Some(value) = metadata.get(key) {
                lines.push(format!("**{key}:** {value}"));
            }
        }
        lines.push(String::new());
        lines.push("---".to_string());
        lines.push(String::new());
        lines.push(content.to_string());
        lines.join("\n")
    }
}

fn validate_pairing_now_seconds(now_seconds: f64) -> Result<(), String> {
    if !now_seconds.is_finite() || now_seconds < 0.0 {
        return Err("pairing now_seconds must be a non-negative finite number".to_string());
    }
    Ok(())
}

fn normalize_pairing_platform(platform: &str) -> Result<&str, String> {
    let platform = platform.trim();
    if platform.is_empty() {
        return Err("pairing platform must not be empty".to_string());
    }
    Ok(platform)
}

fn normalize_pairing_user_id(user_id: &str) -> Result<&str, String> {
    let user_id = user_id.trim();
    if user_id.is_empty() {
        return Err("pairing user_id must not be empty".to_string());
    }
    Ok(user_id)
}

fn normalize_pairing_code(code: &str) -> Result<String, String> {
    let code = code.trim();
    if code.is_empty() {
        return Err("pairing code must not be empty".to_string());
    }
    Ok(code.to_ascii_uppercase())
}

fn pairing_pending_path(pairing_dir: &Path, platform: &str) -> PathBuf {
    pairing_dir.join(format!("{platform}-pending.json"))
}

fn pairing_approved_path(pairing_dir: &Path, platform: &str) -> PathBuf {
    pairing_dir.join(format!("{platform}-approved.json"))
}

fn pairing_rate_limits_path(pairing_dir: &Path) -> PathBuf {
    pairing_dir.join("_rate_limits.json")
}

fn read_pairing_pending_entries(
    pairing_dir: &Path,
    platform: &str,
) -> Result<HashMap<String, GatewayPairingPendingEntry>, String> {
    let path = pairing_pending_path(pairing_dir, platform);
    read_json_marker(&path).map(|entries| entries.unwrap_or_default())
}

fn write_pairing_pending_entries(
    pairing_dir: &Path,
    platform: &str,
    entries: &HashMap<String, GatewayPairingPendingEntry>,
) -> Result<(), String> {
    let path = pairing_pending_path(pairing_dir, platform);
    let bytes = serde_json::to_vec(entries)
        .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
    atomic_write(&path, &bytes)
}

fn read_pairing_approved_entries(
    pairing_dir: &Path,
    platform: &str,
) -> Result<HashMap<String, GatewayPairingApprovedEntry>, String> {
    let path = pairing_approved_path(pairing_dir, platform);
    read_json_marker(&path).map(|entries| entries.unwrap_or_default())
}

fn write_pairing_approved_entries(
    pairing_dir: &Path,
    platform: &str,
    entries: &HashMap<String, GatewayPairingApprovedEntry>,
) -> Result<(), String> {
    let path = pairing_approved_path(pairing_dir, platform);
    let bytes = serde_json::to_vec(entries)
        .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
    atomic_write(&path, &bytes)
}

fn read_pairing_rate_limits(
    pairing_dir: &Path,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let path = pairing_rate_limits_path(pairing_dir);
    read_json_marker(&path).map(|entries| entries.unwrap_or_default())
}

fn write_pairing_rate_limits(
    pairing_dir: &Path,
    entries: &HashMap<String, serde_json::Value>,
) -> Result<(), String> {
    let path = pairing_rate_limits_path(pairing_dir);
    let bytes = serde_json::to_vec(entries)
        .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
    atomic_write(&path, &bytes)
}

fn cleanup_expired_pairing_entries(
    pairing_dir: &Path,
    platform: &str,
    entries: &mut HashMap<String, GatewayPairingPendingEntry>,
    now_seconds: f64,
) -> Result<(), String> {
    let expired = entries
        .iter()
        .filter(|(_code, entry)| now_seconds - entry.created_at > PAIRING_CODE_TTL_SECONDS)
        .map(|(code, _entry)| code.clone())
        .collect::<Vec<_>>();
    if expired.is_empty() {
        return Ok(());
    }
    for code in expired {
        entries.remove(&code);
    }
    write_pairing_pending_entries(pairing_dir, platform, entries)
}

fn pairing_rate_limit_key(platform: &str, user_id: &str) -> String {
    format!("{platform}:{user_id}")
}

fn pairing_lockout_key(platform: &str) -> String {
    format!("_lockout:{platform}")
}

fn pairing_failures_key(platform: &str) -> String {
    format!("_failures:{platform}")
}

fn read_json_number(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(|value| value.as_f64())
}

fn is_pairing_rate_limited(
    limits: &HashMap<String, serde_json::Value>,
    platform: &str,
    user_id: &str,
    now_seconds: f64,
) -> bool {
    read_json_number(limits.get(&pairing_rate_limit_key(platform, user_id)))
        .is_some_and(|last_request| now_seconds - last_request < PAIRING_RATE_LIMIT_SECONDS)
}

fn record_pairing_rate_limit(
    limits: &mut HashMap<String, serde_json::Value>,
    platform: &str,
    user_id: &str,
    now_seconds: f64,
) {
    limits.insert(
        pairing_rate_limit_key(platform, user_id),
        serde_json::Value::from(now_seconds),
    );
}

fn is_pairing_locked_out(
    limits: &HashMap<String, serde_json::Value>,
    platform: &str,
    now_seconds: f64,
) -> bool {
    read_json_number(limits.get(&pairing_lockout_key(platform)))
        .is_some_and(|lockout_until| now_seconds < lockout_until)
}

fn record_pairing_failed_attempt(
    limits: &mut HashMap<String, serde_json::Value>,
    platform: &str,
    now_seconds: f64,
) {
    let fail_key = pairing_failures_key(platform);
    let failures = read_json_number(limits.get(&fail_key)).unwrap_or(0.0) as u64 + 1;
    if failures >= PAIRING_MAX_FAILED_ATTEMPTS {
        limits.insert(
            pairing_lockout_key(platform),
            serde_json::Value::from(now_seconds + PAIRING_LOCKOUT_SECONDS),
        );
        limits.insert(fail_key, serde_json::Value::from(0_u64));
    } else {
        limits.insert(fail_key, serde_json::Value::from(failures));
    }
}

fn generate_pairing_code(
    pending: &HashMap<String, GatewayPairingPendingEntry>,
) -> Result<String, String> {
    for _ in 0..16 {
        let mut bytes = [0_u8; PAIRING_CODE_LENGTH];
        fill_random(&mut bytes)
            .map_err(|error| format!("generating pairing code failed: {error}"))?;
        let code = bytes
            .iter()
            .map(|byte| PAIRING_ALPHABET[(byte & 31) as usize] as char)
            .collect::<String>();
        if !pending.contains_key(&code) {
            return Ok(code);
        }
    }
    Err("unable to generate unique pairing code".to_string())
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("path {} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("creating {} failed: {error}", parent.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system time before unix epoch: {error}"))?
        .as_nanos();
    let tmp_path = parent.join(format!(".{}.tmp", stamp));
    let mut file = File::create(&tmp_path)
        .map_err(|error| format!("creating {} failed: {error}", tmp_path.display()))?;
    file.write_all(data)
        .map_err(|error| format!("writing {} failed: {error}", tmp_path.display()))?;
    file.sync_all()
        .map_err(|error| format!("syncing {} failed: {error}", tmp_path.display()))?;
    fs::rename(&tmp_path, path).map_err(|error| {
        format!(
            "replacing {} with {} failed: {error}",
            path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

fn deserialize_entry_map(raw: serde_json::Value) -> Result<HashMap<String, SessionEntry>, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| "session store root must be a JSON object".to_string())?;
    let mut entries = HashMap::new();
    for (key, value) in object {
        let entry = deserialize_value::<SessionEntry>(value.clone())?;
        if entry.validate().is_ok() {
            entries.insert(key.clone(), entry);
        }
    }
    Ok(entries)
}

fn deserialize_value<T: DeserializeOwned>(value: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn read_json_marker<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(None);
    }
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| format!("parsing {} failed: {error}", path.display()))
}

fn read_restart_failure_counts(runtime_dir: &Path) -> Result<HashMap<String, u64>, String> {
    let path = gateway_restart_failure_counts_path(runtime_dir);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(HashMap::new());
    }
    serde_json::from_str::<HashMap<String, u64>>(&text)
        .map_err(|error| format!("parsing {} failed: {error}", path.display()))
}

fn write_restart_failure_counts(
    runtime_dir: &Path,
    counts: &HashMap<String, u64>,
) -> Result<(), String> {
    let path = gateway_restart_failure_counts_path(runtime_dir);
    let bytes = serde_json::to_vec(counts)
        .map_err(|error| format!("serializing {} failed: {error}", path.display()))?;
    atomic_write(&path, &bytes)
}

fn parse_delivery_platform(value: &str) -> Option<Platform> {
    const BUILTIN_PLATFORMS: &[&str] = &[
        "local",
        "telegram",
        "discord",
        "whatsapp",
        "slack",
        "signal",
        "mattermost",
        "matrix",
        "homeassistant",
        "email",
        "sms",
        "dingtalk",
        "api_server",
        "webhook",
        "feishu",
        "wecom",
        "wecom_callback",
        "weixin",
        "bluebubbles",
        "qqbot",
        "yuanbao",
    ];

    let platform = Platform::parse(value).ok()?;
    BUILTIN_PLATFORMS
        .contains(&platform.as_str())
        .then_some(platform)
}

fn make_session_id(now: NaiveDateTime) -> String {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| (value.as_nanos() & 0xffff_ffff) as u32)
        .unwrap_or_default();
    format!("{}_{suffix:08x}", now.format("%Y%m%d_%H%M%S"))
}

fn sanitize_path_component(value: &str) -> String {
    let sanitized = value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        "misc".to_string()
    } else {
        sanitized
    }
}

fn slice_prefix(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use tempfile::tempdir;

    fn telegram() -> Platform {
        Platform::parse("telegram").unwrap()
    }

    fn slack() -> Platform {
        Platform::parse("slack").unwrap()
    }

    fn matrix() -> Platform {
        Platform::parse("matrix").unwrap()
    }

    fn session_source(chat_type: &str) -> SessionSource {
        SessionSource {
            platform: telegram(),
            chat_id: "12345".to_string(),
            chat_name: None,
            chat_type: chat_type.to_string(),
            user_id: Some("user-1".to_string()),
            user_name: Some("Ada".to_string()),
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

    fn pairing_request(user_id: &str, user_name: &str) -> GatewayPairingRequest {
        GatewayPairingRequest {
            platform: telegram(),
            chat_id: "12345".to_string(),
            user_id: user_id.to_string(),
            user_name: Some(user_name.to_string()),
        }
    }

    #[derive(Default)]
    struct RecordingHostHandler {
        operations: Vec<String>,
        notification_results: VecDeque<bool>,
        notify_path: Option<PathBuf>,
        notify_path_exists_during_send: Vec<bool>,
    }

    impl RecordingHostHandler {
        fn with_notification_results(results: impl IntoIterator<Item = bool>) -> Self {
            Self {
                notification_results: results.into_iter().collect(),
                ..Self::default()
            }
        }
    }

    impl GatewayHostActionHandler for RecordingHostHandler {
        fn send_notification(
            &mut self,
            notification: &GatewayNotification,
        ) -> Result<bool, String> {
            self.operations.push(format!(
                "send:{}:{}:{}",
                notification.platform.as_str(),
                notification.chat_id,
                notification.thread_id.as_deref().unwrap_or("")
            ));
            if let Some(path) = &self.notify_path {
                self.notify_path_exists_during_send.push(path.exists());
            }
            Ok(self.notification_results.pop_front().unwrap_or(true))
        }

        fn schedule_restart(&mut self, launch_mode: RestartLaunchMode) -> Result<(), String> {
            self.operations.push(format!("restart:{launch_mode:?}"));
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingIngressHandler {
        interrupts: Vec<(String, String)>,
        steer_results: VecDeque<bool>,
        steers: Vec<(String, String)>,
    }

    impl RecordingIngressHandler {
        fn with_steer_results(results: impl IntoIterator<Item = bool>) -> Self {
            Self {
                steer_results: results.into_iter().collect(),
                ..Self::default()
            }
        }
    }

    impl GatewayIngressEffectHandler for RecordingIngressHandler {
        fn interrupt_running_agent(
            &mut self,
            session_key: &str,
            reason: &str,
        ) -> Result<(), String> {
            self.interrupts
                .push((session_key.to_string(), reason.to_string()));
            Ok(())
        }

        fn steer_running_agent(&mut self, session_key: &str, text: &str) -> Result<bool, String> {
            self.steers
                .push((session_key.to_string(), text.to_string()));
            Ok(self.steer_results.pop_front().unwrap_or(true))
        }
    }

    #[test]
    fn build_session_key_keeps_dm_isolated() {
        let mut source = session_source("dm");
        source.thread_id = Some("thread-7".to_string());
        let key = build_session_key(&source, true, false).unwrap();
        assert_eq!(key, "agent:main:telegram:dm:12345:thread-7");
    }

    #[test]
    fn build_session_key_uses_per_user_group_isolation_by_default() {
        let source = session_source("group");
        let key = build_session_key(&source, true, false).unwrap();
        assert_eq!(key, "agent:main:telegram:group:12345:user-1");
    }

    #[test]
    fn build_session_key_shares_threads_by_default() {
        let mut source = session_source("group");
        source.thread_id = Some("topic-9".to_string());
        let key = build_session_key(&source, true, false).unwrap();
        assert_eq!(key, "agent:main:telegram:group:12345:topic-9");
        assert!(is_shared_multi_user_session(&source, true, false));
    }

    #[test]
    fn delivery_target_parses_and_resolves_home_channel() {
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            slack(),
            HomeChannel::new(
                slack(),
                "C123",
                "Ops",
                Some("1710000000.000100".to_string()),
            )
            .unwrap(),
        );
        let target = DeliveryTarget::parse("slack", None).unwrap();
        let resolved = target.resolve(&config).unwrap();
        assert_eq!(resolved.chat_id.as_deref(), Some("C123"));
        assert_eq!(resolved.thread_id.as_deref(), Some("1710000000.000100"));
        assert_eq!(resolved.to_string(), "slack:C123:1710000000.000100");
    }

    #[test]
    fn delivery_target_origin_falls_back_to_local_without_source() {
        let target = DeliveryTarget::parse("origin", None).unwrap();
        assert!(target.platform.is_local());
        assert_eq!(target.to_string(), "origin");
    }

    #[test]
    fn delivery_target_unknown_platform_falls_back_to_local() {
        let target = DeliveryTarget::parse("unknown_platform", None).unwrap();
        assert!(target.platform.is_local());

        let explicit = DeliveryTarget::parse("unknown_platform:123", None).unwrap();
        assert!(explicit.platform.is_local());
        assert!(!explicit.is_explicit);
    }

    #[test]
    fn delivery_target_preserves_mixed_case_chat_id() {
        let target = DeliveryTarget::parse("slack:C123ABC:thread123", None).unwrap();
        assert_eq!(target.platform, slack());
        assert_eq!(target.chat_id.as_deref(), Some("C123ABC"));
        assert_eq!(target.thread_id.as_deref(), Some("thread123"));
        assert!(target.is_explicit);
    }

    #[test]
    fn delivery_planner_builds_local_markdown_document() {
        let planner = GatewayDeliveryPlanner::new("/tmp/out");
        let config = RuntimeConfig::default();
        let metadata = HashMap::from([
            ("job_id".to_string(), "job-7".to_string()),
            ("source".to_string(), "cron".to_string()),
        ]);
        let plans = planner
            .plan_delivery_at(
                &config,
                "hello from cron",
                &[DeliveryTarget::parse("local", None).unwrap()],
                Some("job-7"),
                Some("Nightly Build"),
                &metadata,
                parse_datetime("2026-05-25T12:34:56").unwrap(),
            )
            .unwrap();
        assert_eq!(plans.len(), 1);
        let DeliveryOp::WriteMarkdown { path, content } = &plans[0].ops[0] else {
            panic!("expected local markdown write");
        };
        assert_eq!(
            path,
            &PathBuf::from("/tmp/out")
                .join("job-7")
                .join("20260525_123456.md")
        );
        assert!(content.contains("# Nightly Build"));
        assert!(content.contains("**Timestamp:** 2026-05-25 12:34:56"));
        assert!(content.contains("**Job ID:** job-7"));
        assert!(content.contains("**source:** cron"));
        assert!(content.ends_with("hello from cron"));
    }

    #[test]
    fn delivery_planner_resolves_home_channel_and_injects_thread_metadata() {
        let planner = GatewayDeliveryPlanner::new("/tmp/out");
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            slack(),
            HomeChannel::new(
                slack(),
                "C123",
                "Ops",
                Some("1710000000.000100".to_string()),
            )
            .unwrap(),
        );
        let plans = planner
            .plan_delivery_at(
                &config,
                "deploy finished",
                &[DeliveryTarget::parse("slack", None).unwrap()],
                Some("deploy-1"),
                None,
                &HashMap::new(),
                parse_datetime("2026-05-25T12:34:56").unwrap(),
            )
            .unwrap();
        let DeliveryOp::PlatformSend {
            target,
            content,
            metadata,
        } = &plans[0].ops[0]
        else {
            panic!("expected platform send");
        };
        assert_eq!(target.chat_id.as_deref(), Some("C123"));
        assert_eq!(target.thread_id.as_deref(), Some("1710000000.000100"));
        assert_eq!(content, "deploy finished");
        assert_eq!(
            metadata.get("thread_id").map(String::as_str),
            Some("1710000000.000100")
        );
    }

    #[test]
    fn delivery_planner_truncates_platform_output_and_emits_full_text_artifact() {
        let planner = GatewayDeliveryPlanner::new("/tmp/out");
        let explicit = DeliveryTarget::parse("telegram:12345", None).unwrap();
        let long_content = "x".repeat(MAX_PLATFORM_OUTPUT + 50);
        let plans = planner
            .plan_delivery_at(
                &RuntimeConfig::default(),
                &long_content,
                &[explicit],
                Some("job/unsafe"),
                None,
                &HashMap::new(),
                parse_datetime("2026-05-25T12:34:56").unwrap(),
            )
            .unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].ops.len(), 2);
        let DeliveryOp::WriteText { path, content } = &plans[0].ops[0] else {
            panic!("expected full text artifact");
        };
        assert_eq!(
            path,
            &PathBuf::from("/tmp/out").join("job_unsafe_20260525_123456.txt")
        );
        assert_eq!(content, &long_content);
        let DeliveryOp::PlatformSend {
            target,
            content,
            metadata,
        } = &plans[0].ops[1]
        else {
            panic!("expected platform send");
        };
        assert_eq!(target.chat_id.as_deref(), Some("12345"));
        assert!(metadata.is_empty());
        assert!(
            content.contains(
                "[truncated, full output saved to /tmp/out/job_unsafe_20260525_123456.txt]"
            )
        );
        assert_eq!(
            content.chars().take(TRUNCATED_VISIBLE).count(),
            TRUNCATED_VISIBLE
        );
    }

    #[test]
    fn session_store_preserves_resume_pending_session_id() {
        let temp = tempdir().unwrap();
        let mut store = SessionStore::new(temp.path(), RuntimeConfig::default()).unwrap();
        let source = session_source("group");
        let created = store
            .get_or_create_session_at(
                &source,
                false,
                parse_datetime("2026-05-25T10:00:00").unwrap(),
            )
            .unwrap();
        assert!(
            store
                .mark_resume_pending(&created.session_key, "restart_timeout")
                .unwrap()
        );
        let resumed = store
            .get_or_create_session_at(
                &source,
                false,
                parse_datetime("2026-05-25T10:01:00").unwrap(),
            )
            .unwrap();
        assert_eq!(created.session_id, resumed.session_id);
        assert!(resumed.resume_pending);
    }

    #[test]
    fn session_store_resets_suspended_session() {
        let temp = tempdir().unwrap();
        let mut store = SessionStore::new(temp.path(), RuntimeConfig::default()).unwrap();
        let source = session_source("group");
        let first = store
            .get_or_create_session_at(
                &source,
                false,
                parse_datetime("2026-05-25T10:00:00").unwrap(),
            )
            .unwrap();
        assert!(store.suspend_session(&first.session_key).unwrap());
        let second = store
            .get_or_create_session_at(
                &source,
                false,
                parse_datetime("2026-05-25T10:01:00").unwrap(),
            )
            .unwrap();
        assert_ne!(first.session_id, second.session_id);
        assert_eq!(second.auto_reset_reason.as_deref(), Some("suspended"));
    }

    #[test]
    fn session_store_prunes_old_entries() {
        let temp = tempdir().unwrap();
        let mut store = SessionStore::new(temp.path(), RuntimeConfig::default()).unwrap();
        let source = session_source("group");
        let entry = store
            .get_or_create_session_at(
                &source,
                false,
                parse_datetime("2026-05-01T10:00:00").unwrap(),
            )
            .unwrap();
        let removed = store
            .prune_old_entries_at(7, parse_datetime("2026-05-25T10:00:00").unwrap())
            .unwrap();
        assert_eq!(removed, 1);
        assert!(store.entry(&entry.session_key).unwrap().is_none());
    }

    #[test]
    fn session_store_round_trips_sessions_file() {
        let temp = tempdir().unwrap();
        let source = session_source("group");
        {
            let mut store = SessionStore::new(temp.path(), RuntimeConfig::default()).unwrap();
            let _ = store
                .get_or_create_session_at(
                    &source,
                    false,
                    parse_datetime("2026-05-25T10:00:00").unwrap(),
                )
                .unwrap();
        }
        let mut store = SessionStore::new(temp.path(), RuntimeConfig::default()).unwrap();
        assert_eq!(store.entries_len().unwrap(), 1);
    }

    #[test]
    fn resolve_command_maps_aliases_and_gateway_scope() {
        let background = resolve_command("/btw").unwrap();
        assert_eq!(background.canonical, "background");
        assert!(background.gateway_dispatchable);

        let clear = resolve_command("clear").unwrap();
        assert_eq!(clear.canonical, "clear");
        assert!(!clear.gateway_dispatchable);

        assert!(resolve_command("/path/to/file.py").is_none());
        assert!(is_gateway_known_command("help"));
        assert!(!is_gateway_known_command("clear"));
    }

    #[test]
    fn should_bypass_active_session_matches_python_contract() {
        for command in [
            "model",
            "reasoning",
            "personality",
            "voice",
            "insights",
            "title",
            "resume",
            "retry",
            "undo",
            "compress",
            "usage",
            "reload-mcp",
            "sethome",
            "reset",
        ] {
            assert!(
                should_bypass_active_session(Some(command)),
                "/{command} should bypass"
            );
        }
        assert!(!should_bypass_active_session(Some("foobar")));
        assert!(!should_bypass_active_session(Some("path/to/file.py")));
        assert!(!should_bypass_active_session(None));
    }

    #[test]
    fn message_event_parses_commands_and_coerces_plaintext_restart() {
        let source = session_source("dm");
        let mut event = MessageEvent {
            text: "Please restart the gateway!".to_string(),
            message_type: MessageType::Text,
            source,
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        event.coerce_plaintext_gateway_command();
        assert_eq!(event.text, "/restart");
        assert_eq!(event.get_command().as_deref(), Some("restart"));

        let alias_event = MessageEvent {
            text: "/reset tomorrow — now".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(alias_event.get_command().as_deref(), Some("reset"));
        assert_eq!(alias_event.get_command_args(), "tomorrow -- now");
    }

    #[test]
    fn active_session_ingress_bypasses_only_known_commands() {
        let command_event = MessageEvent {
            text: "/status".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_active_session_ingress(&command_event),
            ActiveSessionIngressAction::DispatchCommand {
                canonical: "status"
            }
        );

        let text_event = MessageEvent {
            text: "hello world".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_active_session_ingress(&text_event),
            ActiveSessionIngressAction::QueuePending
        );
    }

    #[test]
    fn running_session_plan_dispatches_safe_mid_turn_commands() {
        let state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        let event = MessageEvent {
            text: "/btw summarize gateway state".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &state),
            RunningSessionAction::DispatchCommand {
                canonical: "background"
            }
        );
    }

    #[test]
    fn running_session_plan_rejects_model_mid_turn() {
        let state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        let event = MessageEvent {
            text: "/model gpt-5".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &state),
            RunningSessionAction::Reject {
                message: "Agent is running — wait or /stop first, then switch models.".to_string()
            }
        );
    }

    #[test]
    fn running_session_plan_allows_goal_control_but_not_new_goal_text() {
        let state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        let status_event = MessageEvent {
            text: "/goal status".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_running_session_action(&status_event, &state),
            RunningSessionAction::DispatchCommand { canonical: "goal" }
        );

        let set_event = MessageEvent {
            text: "/goal rewrite the gateway in rust".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            plan_running_session_action(&set_event, &state),
            RunningSessionAction::Reject {
                message: "Agent is running — use /goal status / pause / clear mid-run, or /stop before setting a new goal.".to_string()
            }
        );
    }

    #[test]
    fn running_session_plan_falls_back_to_busy_mode_for_plain_text() {
        let event = MessageEvent {
            text: "follow up after the tool call".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        let queue_state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Queue,
            can_steer: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &queue_state),
            RunningSessionAction::QueuePending
        );

        let steer_state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Steer,
            can_steer: true,
        };
        assert_eq!(
            plan_running_session_action(&event, &steer_state),
            RunningSessionAction::SteerActive
        );

        let interrupt_state = RunningSessionState {
            draining: false,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &interrupt_state),
            RunningSessionAction::InterruptAndQueue
        );
    }

    #[test]
    fn running_session_plan_handles_drain_mode() {
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let queue_state = RunningSessionState {
            draining: true,
            queue_during_drain: true,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &queue_state),
            RunningSessionAction::QueueDuringDrain
        );

        let reject_state = RunningSessionState {
            draining: true,
            queue_during_drain: false,
            busy_input_mode: BusyInputMode::Interrupt,
            can_steer: false,
        };
        assert_eq!(
            plan_running_session_action(&event, &reject_state),
            RunningSessionAction::RejectDuringDrain
        );
    }

    #[test]
    fn ingress_host_response_plans_queue_busy_ack_with_status_and_hint() {
        let event = MessageEvent {
            text: "follow up".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-7".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = GatewayIngressDecision::QueuePending {
            session_key: "agent:main:telegram:dm:12345".to_string(),
            depth: 1,
            reason: PendingReason::Busy,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(2),
            api_call_count: Some(3),
            max_iterations: Some(12),
            current_tool: Some("terminal".to_string()),
            onboarding_hint: Some("Hint text".to_string()),
        };

        let reply = plan_ingress_host_response(&decision, &event, Some(&busy))
            .unwrap()
            .unwrap();
        assert_eq!(reply.reply_to_message_id.as_deref(), Some("msg-7"));
        assert!(reply.message.contains("Queued for the next turn"));
        assert!(reply.message.contains("2 min elapsed"));
        assert!(reply.message.contains("iteration 3/12"));
        assert!(reply.message.contains("running: terminal"));
        assert!(reply.message.contains("Hint text"));
    }

    #[test]
    fn ingress_host_response_plans_interrupt_ack() {
        let event = MessageEvent {
            text: "stop that".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = GatewayIngressDecision::QueuePending {
            session_key: "agent:main:telegram:dm:12345".to_string(),
            depth: 1,
            reason: PendingReason::Interrupt,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: None,
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        let reply = plan_ingress_host_response(&decision, &event, Some(&busy))
            .unwrap()
            .unwrap();
        assert!(reply.message.contains("Interrupting current task"));
    }

    #[test]
    fn ingress_host_response_plans_steer_ack() {
        let event = MessageEvent {
            text: "focus on tests".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = GatewayIngressDecision::SteerActive {
            session_key: "agent:main:telegram:dm:12345".to_string(),
            event: event.clone(),
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(1),
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        let reply = plan_ingress_host_response(&decision, &event, Some(&busy))
            .unwrap()
            .unwrap();
        assert!(reply.message.contains("Steered into current run"));
        assert!(reply.message.contains("1 min elapsed"));
    }

    #[test]
    fn ingress_host_response_plans_drain_queue_and_reject_messages_without_busy_context() {
        let event = MessageEvent {
            text: "follow up".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let queued = GatewayIngressDecision::QueuePending {
            session_key: "agent:main:telegram:dm:12345".to_string(),
            depth: 1,
            reason: PendingReason::Drain,
        };
        let rejected = GatewayIngressDecision::Reject {
            message: "⏳ Gateway is restarting and is not accepting another turn right now."
                .to_string(),
        };

        let queued_reply = plan_ingress_host_response(&queued, &event, None)
            .unwrap()
            .unwrap();
        assert!(queued_reply.message.contains("queued for the next turn"));

        let rejected_reply = plan_ingress_host_response(&rejected, &event, None)
            .unwrap()
            .unwrap();
        assert!(
            rejected_reply
                .message
                .contains("not accepting another turn")
        );
    }

    #[test]
    fn ingress_host_response_suppresses_busy_ack_when_disabled_or_debounced() {
        let event = MessageEvent {
            text: "follow up".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = GatewayIngressDecision::QueuePending {
            session_key: "agent:main:telegram:dm:12345".to_string(),
            depth: 1,
            reason: PendingReason::Busy,
        };
        let disabled = GatewayBusyReplyContext {
            ack_enabled: false,
            cooldown_active: false,
            elapsed_minutes: None,
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };
        let debounced = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: true,
            elapsed_minutes: None,
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        assert!(
            plan_ingress_host_response(&decision, &event, Some(&disabled))
                .unwrap()
                .is_none()
        );
        assert!(
            plan_ingress_host_response(&decision, &event, Some(&debounced))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn runtime_prepare_busy_reply_context_tracks_cooldown_in_runtime_state() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let session_key = "agent:main:telegram:dm:12345";

        let first = runtime
            .prepare_busy_reply_context(
                session_key,
                true,
                100.0,
                30.0,
                GatewayBusyReplyStatus {
                    elapsed_minutes: Some(2),
                    api_call_count: Some(3),
                    max_iterations: Some(12),
                    current_tool: Some("terminal".to_string()),
                    onboarding_hint: None,
                },
            )
            .unwrap();
        assert!(!first.cooldown_active);
        assert_eq!(first.elapsed_minutes, Some(2));

        let second = runtime
            .prepare_busy_reply_context(
                session_key,
                true,
                110.0,
                30.0,
                GatewayBusyReplyStatus::default(),
            )
            .unwrap();
        assert!(second.cooldown_active);

        let third = runtime
            .prepare_busy_reply_context(
                session_key,
                true,
                131.0,
                30.0,
                GatewayBusyReplyStatus::default(),
            )
            .unwrap();
        assert!(!third.cooldown_active);
    }

    #[test]
    fn runtime_finish_turn_clears_busy_ack_cooldown_state() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime.active_sessions.insert(
            created.session_key.clone(),
            ActiveSession {
                session_key: created.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime
            .prepare_busy_reply_context(
                &created.session_key,
                true,
                100.0,
                30.0,
                GatewayBusyReplyStatus::default(),
            )
            .unwrap();

        runtime.finish_turn(&created.session_key).unwrap();

        let after = runtime
            .prepare_busy_reply_context(
                &created.session_key,
                true,
                110.0,
                30.0,
                GatewayBusyReplyStatus::default(),
            )
            .unwrap();
        assert!(!after.cooldown_active);
    }

    #[test]
    fn runtime_ingest_event_host_starts_turn_without_immediate_response() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-1".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        let plan = runtime.ingest_event_host(event, None).unwrap();
        assert!(plan.response.is_none());
        assert!(plan.response_on_effect_failure.is_none());
        assert_eq!(plan.effect, GatewayIngressHostEffect::None);
        assert!(matches!(
            plan.decision,
            GatewayIngressDecision::StartTurn { .. }
        ));
    }

    #[test]
    fn runtime_ingest_event_host_packages_busy_queue_reply() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Queue)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key,
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        let event = MessageEvent {
            text: "follow up".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-2".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(1),
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        let plan = runtime.ingest_event_host(event, Some(&busy)).unwrap();
        assert!(matches!(
            plan.decision,
            GatewayIngressDecision::QueuePending {
                reason: PendingReason::Busy,
                ..
            }
        ));
        assert_eq!(plan.effect, GatewayIngressHostEffect::None);
        assert!(plan.response_on_effect_failure.is_none());
        let response = plan.response.unwrap();
        assert!(response.message.contains("Queued for the next turn"));
        assert_eq!(response.reply_to_message_id.as_deref(), Some("msg-2"));
    }

    #[test]
    fn runtime_ingest_event_host_packages_interrupt_effect() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        let event = MessageEvent {
            text: "stop that".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: None,
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        let plan = runtime.ingest_event_host(event, Some(&busy)).unwrap();
        assert!(matches!(
            plan.decision,
            GatewayIngressDecision::QueuePending {
                reason: PendingReason::Interrupt,
                ..
            }
        ));
        assert_eq!(
            plan.effect,
            GatewayIngressHostEffect::InterruptRunningAgent {
                session_key,
                reason: "stop that".to_string(),
            }
        );
        assert!(plan.response_on_effect_failure.is_none());
        let response = plan.response.unwrap();
        assert!(response.message.contains("Interrupting current task"));
    }

    #[test]
    fn runtime_ingest_event_host_packages_drain_reject_reply() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        runtime.set_draining(true, false);
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-3".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        let plan = runtime.ingest_event_host(event, None).unwrap();
        assert!(matches!(
            plan.decision,
            GatewayIngressDecision::Reject { .. }
        ));
        assert_eq!(plan.effect, GatewayIngressHostEffect::None);
        assert!(plan.response_on_effect_failure.is_none());
        let response = plan.response.unwrap();
        assert!(response.message.contains("not accepting new work"));
        assert_eq!(response.reply_to_message_id.as_deref(), Some("msg-3"));
    }

    #[test]
    fn runtime_ingest_event_host_packages_steer_effect_and_queue_fallback_reply() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Steer)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: true },
            },
        );
        let event = MessageEvent {
            text: "focus on tests".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-4".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(1),
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };

        let plan = runtime.ingest_event_host(event, Some(&busy)).unwrap();
        assert!(matches!(
            plan.decision,
            GatewayIngressDecision::SteerActive { .. }
        ));
        assert_eq!(
            plan.effect,
            GatewayIngressHostEffect::SteerRunningAgent {
                session_key,
                text: "focus on tests".to_string(),
            }
        );
        let response = plan.response.unwrap();
        assert!(response.message.contains("Steered into current run"));
        let fallback = plan.response_on_effect_failure.unwrap();
        assert!(fallback.message.contains("Queued for the next turn"));
    }

    #[test]
    fn runtime_execute_ingress_host_plan_runs_interrupt_effect() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        let event = MessageEvent {
            text: "stop that".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: None,
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };
        let plan = runtime.ingest_event_host(event, Some(&busy)).unwrap();
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .execute_ingress_host_plan(plan, &mut handler)
            .unwrap();
        assert_eq!(
            handler.interrupts,
            vec![(session_key.clone(), "stop that".to_string())]
        );
        assert!(result.effect_applied);
        assert!(matches!(
            result.decision,
            GatewayIngressDecision::QueuePending {
                reason: PendingReason::Interrupt,
                ..
            }
        ));
        assert!(
            result
                .response
                .unwrap()
                .message
                .contains("Interrupting current task")
        );
    }

    #[test]
    fn runtime_execute_ingress_host_plan_falls_back_from_steer_to_queue() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Steer)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: true },
            },
        );
        let event = MessageEvent {
            text: "focus on tests".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(1),
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };
        let plan = runtime.ingest_event_host(event, Some(&busy)).unwrap();
        let mut handler = RecordingIngressHandler::with_steer_results([false]);

        let result = runtime
            .execute_ingress_host_plan(plan, &mut handler)
            .unwrap();
        assert_eq!(
            handler.steers,
            vec![(session_key.clone(), "focus on tests".to_string())]
        );
        assert!(!result.effect_applied);
        assert!(matches!(
            result.decision,
            GatewayIngressDecision::QueuePending {
                reason: PendingReason::Busy,
                ..
            }
        ));
        assert_eq!(runtime.queue_depth(&session_key), 1);
        assert!(
            result
                .response
                .unwrap()
                .message
                .contains("Queued for the next turn")
        );
    }

    #[test]
    fn runtime_handle_event_host_starts_turn_in_one_call() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: Some("msg-5".to_string()),
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host(event, None, &mut handler)
            .unwrap();
        assert!(matches!(
            result.decision,
            GatewayIngressDecision::StartTurn { .. }
        ));
        assert!(result.response.is_none());
        assert_eq!(result.effect, GatewayIngressHostEffect::None);
        assert!(!result.effect_applied);
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_handle_event_host_falls_back_from_steer_to_queue_in_one_call() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Steer)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: true },
            },
        );
        let event = MessageEvent {
            text: "focus on tests".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let busy = GatewayBusyReplyContext {
            ack_enabled: true,
            cooldown_active: false,
            elapsed_minutes: Some(1),
            api_call_count: None,
            max_iterations: None,
            current_tool: None,
            onboarding_hint: None,
        };
        let mut handler = RecordingIngressHandler::with_steer_results([false]);

        let result = runtime
            .handle_event_host(event, Some(&busy), &mut handler)
            .unwrap();
        assert_eq!(
            handler.steers,
            vec![(session_key.clone(), "focus on tests".to_string())]
        );
        assert!(matches!(
            result.decision,
            GatewayIngressDecision::QueuePending {
                reason: PendingReason::Busy,
                ..
            }
        ));
        assert_eq!(runtime.queue_depth(&session_key), 1);
        assert!(
            result
                .response
                .unwrap()
                .message
                .contains("Queued for the next turn")
        );
    }

    #[test]
    fn runtime_handle_event_host_authorized_drops_unauthorized_active_session_message() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        let event = MessageEvent {
            text: "intrude".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_authorized(event, false, None, &mut handler)
            .unwrap();
        assert_eq!(
            result,
            GatewayHostHandleOutcome::DroppedUnauthorizedActiveSession { session_key }
        );
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_handle_event_host_authorized_executes_when_authorized() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_authorized(event, true, None, &mut handler)
            .unwrap();
        assert!(matches!(
            result,
            GatewayHostHandleOutcome::Executed(GatewayIngressExecutionResult {
                decision: GatewayIngressDecision::StartTurn { .. },
                ..
            })
        ));
    }

    #[test]
    fn runtime_handle_event_host_with_authorization_drops_missing_user_id() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let mut source = session_source("group");
        source.user_id = None;
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: source.clone(),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_with_authorization(
                event,
                GatewayAuthorizationStatus::Authorized,
                UnauthorizedDmBehavior::Ignore,
                None,
                &mut handler,
            )
            .unwrap();
        assert_eq!(
            result,
            GatewayHostHandleOutcome::DroppedMissingUserId {
                platform: source.platform,
                chat_id: source.chat_id,
            }
        );
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_handle_event_host_with_authorization_drops_unauthorized_cold_path() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let source = session_source("group");
        let session_key = build_session_key(&source, true, false).unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source,
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_with_authorization(
                event,
                GatewayAuthorizationStatus::Unauthorized,
                UnauthorizedDmBehavior::Ignore,
                None,
                &mut handler,
            )
            .unwrap();
        assert_eq!(
            result,
            GatewayHostHandleOutcome::DroppedUnauthorizedColdPath { session_key }
        );
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_handle_event_host_with_authorization_requests_dm_pairing() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let source = session_source("dm");
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source,
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_with_authorization(
                event,
                GatewayAuthorizationStatus::Unauthorized,
                UnauthorizedDmBehavior::Pair,
                None,
                &mut handler,
            )
            .unwrap();
        assert_eq!(
            result,
            GatewayHostHandleOutcome::RequireUnauthorizedDmPairing {
                request: GatewayPairingRequest {
                    platform: telegram(),
                    chat_id: "12345".to_string(),
                    user_id: "user-1".to_string(),
                    user_name: Some("Ada".to_string()),
                },
            }
        );
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_handle_event_host_with_authorization_executes_internal_event_without_user_id() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let mut source = session_source("group");
        source.user_id = None;
        let event = MessageEvent {
            text: "system update".to_string(),
            message_type: MessageType::Text,
            source,
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: true,
        };
        let mut handler = RecordingIngressHandler::default();

        let result = runtime
            .handle_event_host_with_authorization(
                event,
                GatewayAuthorizationStatus::MissingUserId,
                UnauthorizedDmBehavior::Ignore,
                None,
                &mut handler,
            )
            .unwrap();
        assert!(matches!(
            result,
            GatewayHostHandleOutcome::Executed(GatewayIngressExecutionResult {
                decision: GatewayIngressDecision::StartTurn { .. },
                ..
            })
        ));
    }

    #[test]
    fn pairing_store_request_code_persists_and_approves_user() {
        let temp = tempdir().unwrap();
        let store = GatewayPairingStore::new(temp.path()).unwrap();
        let request = pairing_request("user-1", "Ada");

        let decision = store.request_code(&request, 100.0).unwrap();
        let GatewayPairingCodeDecision::SendPairingCode { code } = decision else {
            panic!("expected pairing code");
        };
        assert_eq!(code.len(), PAIRING_CODE_LENGTH);
        assert!(
            code.chars()
                .all(|ch| PAIRING_ALPHABET.contains(&(ch as u8)))
        );

        let response = plan_unauthorized_dm_pairing_response(
            &request,
            &GatewayPairingCodeDecision::SendPairingCode { code: code.clone() },
        )
        .unwrap()
        .unwrap();
        assert!(response.message.contains("Here's your pairing code"));
        assert!(response.message.contains(&code));

        let pending = read_pairing_pending_entries(temp.path(), "telegram").unwrap();
        assert_eq!(pending.len(), 1);

        let approval = store
            .approve_code("telegram", &code.to_ascii_lowercase(), 200.0)
            .unwrap();
        assert_eq!(
            approval,
            Some(GatewayPairingApproval {
                user_id: "user-1".to_string(),
                user_name: "Ada".to_string(),
            })
        );
        assert!(store.is_approved("telegram", "user-1").unwrap());
    }

    #[test]
    fn pairing_store_suppresses_repeat_request_within_rate_limit() {
        let temp = tempdir().unwrap();
        let store = GatewayPairingStore::new(temp.path()).unwrap();
        let request = pairing_request("user-1", "Ada");

        let first = store.request_code(&request, 100.0).unwrap();
        assert!(matches!(
            first,
            GatewayPairingCodeDecision::SendPairingCode { .. }
        ));
        let second = store.request_code(&request, 200.0).unwrap();
        assert_eq!(second, GatewayPairingCodeDecision::Suppress);
    }

    #[test]
    fn pairing_store_returns_try_later_when_pending_limit_is_reached() {
        let temp = tempdir().unwrap();
        let store = GatewayPairingStore::new(temp.path()).unwrap();
        for (offset, user_id) in ["user-1", "user-2", "user-3"].into_iter().enumerate() {
            let decision = store
                .request_code(&pairing_request(user_id, "Ada"), 100.0 + offset as f64)
                .unwrap();
            assert!(matches!(
                decision,
                GatewayPairingCodeDecision::SendPairingCode { .. }
            ));
        }

        let blocked = pairing_request("user-4", "Bea");
        let decision = store.request_code(&blocked, 104.0).unwrap();
        assert_eq!(decision, GatewayPairingCodeDecision::SendTryLater);
        let repeated = store.request_code(&blocked, 105.0).unwrap();
        assert_eq!(repeated, GatewayPairingCodeDecision::Suppress);
    }

    #[test]
    fn pairing_store_invalid_approvals_trigger_lockout() {
        let temp = tempdir().unwrap();
        let store = GatewayPairingStore::new(temp.path()).unwrap();
        for attempt in 0..PAIRING_MAX_FAILED_ATTEMPTS {
            assert!(
                store
                    .approve_code("telegram", "badcode", 100.0 + attempt as f64)
                    .unwrap()
                    .is_none()
            );
        }

        let decision = store
            .request_code(&pairing_request("user-9", "Kai"), 200.0)
            .unwrap();
        assert_eq!(decision, GatewayPairingCodeDecision::SendTryLater);
    }

    #[test]
    fn runtime_handle_event_host_with_pairing_replies_then_suppresses_repeat_dm() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let pairing_dir = temp.path().join("pairing");
        let pairing_store = GatewayPairingStore::new(&pairing_dir).unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let session_key = build_session_key(&event.source, true, false).unwrap();
        let mut handler = RecordingIngressHandler::default();

        let first = runtime
            .handle_event_host_with_pairing(
                event.clone(),
                GatewayAuthorizationStatus::Unauthorized,
                UnauthorizedDmBehavior::Pair,
                &pairing_store,
                100.0,
                None,
                &mut handler,
            )
            .unwrap();
        let GatewayHostHandleOutcome::RepliedUnauthorizedDmPairing {
            session_key: reply_session_key,
            response,
        } = first
        else {
            panic!("expected pairing reply");
        };
        assert_eq!(reply_session_key, session_key);
        assert!(response.message.contains("Here's your pairing code"));

        let second = runtime
            .handle_event_host_with_pairing(
                event,
                GatewayAuthorizationStatus::Unauthorized,
                UnauthorizedDmBehavior::Pair,
                &pairing_store,
                200.0,
                None,
                &mut handler,
            )
            .unwrap();
        assert_eq!(
            second,
            GatewayHostHandleOutcome::DroppedUnauthorizedColdPath { session_key }
        );
        assert!(handler.interrupts.is_empty());
        assert!(handler.steers.is_empty());
    }

    #[test]
    fn runtime_ingest_starts_turn_for_plain_text() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        let decision = runtime.ingest_event(event.clone()).unwrap();
        let GatewayIngressDecision::StartTurn {
            session_key,
            session,
            event: returned_event,
        } = decision
        else {
            panic!("expected StartTurn");
        };
        assert_eq!(returned_event.text, "hello");
        assert_eq!(session.session_key, session_key);
        assert_eq!(
            runtime
                .active_session(&session_key)
                .map(|state| state.phase),
            Some(ActiveSessionPhase::PendingStart)
        );
    }

    #[test]
    fn runtime_ingest_dispatches_gateway_command_without_claiming_turn() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "/help".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = runtime.ingest_event(event).unwrap();
        assert!(matches!(
            decision,
            GatewayIngressDecision::DispatchCommand {
                canonical: "help",
                ..
            }
        ));
        assert!(runtime.active_sessions.is_empty());
    }

    #[test]
    fn runtime_ingest_queues_busy_follow_up_and_finish_turn_restarts_it() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Queue)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        let event = MessageEvent {
            text: "follow up".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = runtime.ingest_event(event).unwrap();
        assert_eq!(
            decision,
            GatewayIngressDecision::QueuePending {
                session_key: session_key.clone(),
                depth: 1,
                reason: PendingReason::Busy,
            }
        );

        let next = runtime.finish_turn(&session_key).unwrap().unwrap();
        assert!(matches!(
            next,
            GatewayIngressDecision::StartTurn { session_key: ref key, .. } if key == &session_key
        ));
    }

    #[test]
    fn runtime_ingest_steers_when_running_session_supports_it() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Steer)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: true },
            },
        );
        let event = MessageEvent {
            text: "also check the logs".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = runtime.ingest_event(event.clone()).unwrap();
        assert_eq!(
            decision,
            GatewayIngressDecision::SteerActive { session_key, event }
        );
    }

    #[test]
    fn runtime_explicit_fifo_queue_promotes_overflow_in_order() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        let mk_event = |text: &str| MessageEvent {
            text: text.to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        assert_eq!(
            runtime.enqueue_fifo_follow_up(&session_key, mk_event("one")),
            1
        );
        assert_eq!(
            runtime.enqueue_fifo_follow_up(&session_key, mk_event("two")),
            2
        );
        assert_eq!(
            runtime.enqueue_fifo_follow_up(&session_key, mk_event("three")),
            3
        );

        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let first = runtime.finish_turn(&session_key).unwrap().unwrap();
        let GatewayIngressDecision::StartTurn { event, .. } = first else {
            panic!("expected StartTurn");
        };
        assert_eq!(event.text, "one");
        assert_eq!(runtime.queue_depth(&session_key), 2);

        runtime.mark_turn_running(&session_key, false);
        let second = runtime.finish_turn(&session_key).unwrap().unwrap();
        let GatewayIngressDecision::StartTurn { event, .. } = second else {
            panic!("expected StartTurn");
        };
        assert_eq!(event.text, "two");
        assert_eq!(runtime.queue_depth(&session_key), 1);
    }

    #[test]
    fn runtime_rejects_new_plain_text_while_draining() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        runtime.set_draining(true, false);
        let event = MessageEvent {
            text: "hello".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let decision = runtime.ingest_event(event).unwrap();
        assert_eq!(
            decision,
            GatewayIngressDecision::Reject {
                message: "⏳ Gateway is restarting and is not accepting new work right now."
                    .to_string()
            }
        );
    }

    #[test]
    fn drain_queue_policy_only_applies_to_restarts_in_queue_or_steer_modes() {
        assert!(queue_during_drain_enabled(true, BusyInputMode::Queue));
        assert!(queue_during_drain_enabled(true, BusyInputMode::Steer));
        assert!(!queue_during_drain_enabled(true, BusyInputMode::Interrupt));
        assert!(!queue_during_drain_enabled(false, BusyInputMode::Queue));
    }

    #[test]
    fn runtime_request_restart_is_idempotent_and_validates_mode() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        assert!(runtime.request_restart(true, false).unwrap());
        assert!(runtime.restart_requested());
        assert!(!runtime.request_restart(true, false).unwrap());
        assert!(runtime.request_restart(true, true).is_err());
    }

    #[test]
    fn runtime_begin_shutdown_snapshots_active_sessions_and_drain_policy() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Queue)
                .unwrap();
        let session_key = build_session_key(&session_source("dm"), true, false).unwrap();
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key: session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let plan = runtime.begin_shutdown(true, true, false).unwrap();
        assert_eq!(
            plan.request,
            GatewayDrainRequest::restart(true, false).unwrap()
        );
        assert_eq!(plan.active_session_keys, vec![session_key]);
        assert!(plan.queue_during_drain);
        assert!(runtime.draining());
    }

    #[test]
    fn runtime_begin_shutdown_disables_drain_queue_for_plain_shutdown() {
        let temp = tempdir().unwrap();
        let mut runtime =
            GatewayRuntime::new(temp.path(), RuntimeConfig::default(), BusyInputMode::Queue)
                .unwrap();
        let plan = runtime.begin_shutdown(false, false, false).unwrap();
        assert_eq!(plan.request, GatewayDrainRequest::shutdown());
        assert!(!plan.queue_during_drain);
        assert!(plan.active_session_keys.is_empty());
    }

    #[test]
    fn runtime_begin_shutdown_host_packages_drain_and_notifications() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "999", "Ops Home", None).unwrap(),
        );
        let mut runtime =
            GatewayRuntime::new(temp.path(), config, BusyInputMode::Interrupt).unwrap();
        runtime.request_restart(true, false).unwrap();
        runtime.active_sessions.insert(
            "agent:main:telegram:dm:12345".to_string(),
            ActiveSession {
                session_key: "agent:main:telegram:dm:12345".to_string(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let flow = runtime.begin_shutdown_host(true, true, false).unwrap();
        assert_eq!(
            flow.drain.request,
            GatewayDrainRequest::restart(true, false).unwrap()
        );
        assert_eq!(
            flow.drain.active_session_keys,
            vec!["agent:main:telegram:dm:12345".to_string()]
        );
        assert_eq!(flow.pre_drain.reply, None);
        assert_eq!(flow.pre_drain.actions.len(), 2);
    }

    #[test]
    fn runtime_finish_shutdown_marks_only_still_running_sessions_resume_pending() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let source_a = session_source("dm");
        let mut source_b = session_source("dm");
        source_b.chat_id = "67890".to_string();
        let session_a = runtime
            .session_store
            .get_or_create_session_at(
                &source_a,
                false,
                parse_datetime("2026-05-25T10:00:00").unwrap(),
            )
            .unwrap();
        let session_b = runtime
            .session_store
            .get_or_create_session_at(
                &source_b,
                false,
                parse_datetime("2026-05-25T10:00:00").unwrap(),
            )
            .unwrap();
        runtime.active_sessions.insert(
            session_a.session_key.clone(),
            ActiveSession {
                session_key: session_a.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime.active_sessions.insert(
            session_b.session_key.clone(),
            ActiveSession {
                session_key: session_b.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let plan = runtime.begin_shutdown(true, false, true).unwrap();
        assert_eq!(
            plan.active_session_keys,
            vec![session_a.session_key.clone(), session_b.session_key.clone()]
        );

        runtime.active_sessions.remove(&session_a.session_key);
        let outcome = runtime.finish_shutdown(true).unwrap();
        assert_eq!(
            outcome.interrupted_session_keys,
            vec![session_b.session_key.clone()]
        );
        assert_eq!(
            outcome.marked_resume_pending,
            vec![session_b.session_key.clone()]
        );
        assert_eq!(outcome.resume_reason, Some("restart_timeout"));
        assert!(!outcome.write_clean_shutdown_marker);
        assert_eq!(outcome.exit_code, Some(GATEWAY_SERVICE_RESTART_EXIT_CODE));
        assert_eq!(
            outcome.increment_restart_failure_counts,
            vec![session_a.session_key.clone(), session_b.session_key.clone()]
        );
        assert!(
            runtime
                .session_store
                .entry(&session_a.session_key)
                .unwrap()
                .unwrap()
                .resume_pending
                == false
        );
        assert!(
            runtime
                .session_store
                .entry(&session_b.session_key)
                .unwrap()
                .unwrap()
                .resume_pending
        );
    }

    #[test]
    fn startup_recovery_uses_clean_shutdown_marker_presence() {
        assert_eq!(
            plan_startup_recovery(true),
            GatewayStartupRecovery {
                consume_clean_shutdown_marker: true,
                suspend_recently_active: false,
            }
        );
        assert_eq!(
            plan_startup_recovery(false),
            GatewayStartupRecovery {
                consume_clean_shutdown_marker: false,
                suspend_recently_active: true,
            }
        );
    }

    #[test]
    fn shutdown_post_drain_host_plan_writes_clean_marker_and_schedules_detached_restart() {
        let temp = tempdir().unwrap();
        let outcome = GatewayShutdownOutcome {
            request: GatewayDrainRequest::restart(true, false).unwrap(),
            timed_out: false,
            interrupted_session_keys: Vec::new(),
            marked_resume_pending: Vec::new(),
            resume_reason: None,
            write_clean_shutdown_marker: true,
            increment_restart_failure_counts: Vec::new(),
            exit_code: None,
        };

        let plan = plan_shutdown_post_drain_host(temp.path(), &outcome);
        assert_eq!(plan.reply, None);
        assert_eq!(
            plan.actions,
            vec![
                GatewayHostAction::WriteText {
                    path: gateway_clean_shutdown_path(temp.path()),
                    content: String::new(),
                },
                GatewayHostAction::ScheduleRestart {
                    launch_mode: RestartLaunchMode::Detached,
                },
            ]
        );
    }

    #[test]
    fn shutdown_post_drain_host_plan_noops_for_timed_out_service_restart() {
        let outcome = GatewayShutdownOutcome {
            request: GatewayDrainRequest::restart(false, true).unwrap(),
            timed_out: true,
            interrupted_session_keys: Vec::new(),
            marked_resume_pending: Vec::new(),
            resume_reason: Some("restart_timeout"),
            write_clean_shutdown_marker: false,
            increment_restart_failure_counts: Vec::new(),
            exit_code: Some(GATEWAY_SERVICE_RESTART_EXIT_CODE),
        };

        let plan = plan_shutdown_post_drain_host(Path::new("/tmp/runtime"), &outcome);
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn shutdown_notifications_use_restart_message_and_home_channel_dedupe() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "999", "Ops Home", None).unwrap(),
        );
        let mut runtime =
            GatewayRuntime::new(temp.path(), config, BusyInputMode::Interrupt).unwrap();
        runtime.request_restart(true, false).unwrap();
        runtime.active_sessions.insert(
            "agent:main:telegram:dm:999".to_string(),
            ActiveSession {
                session_key: "agent:main:telegram:dm:999".to_string(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let notifications = runtime.plan_shutdown_notifications().unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].chat_id, "999");
        assert!(notifications[0].message.contains("restarting"));
        assert!(notifications[0].message.contains("try to resume"));
    }

    #[test]
    fn shutdown_notification_host_plan_wraps_notifications_as_host_actions() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        runtime.request_restart(true, false).unwrap();
        runtime.active_sessions.insert(
            "agent:main:telegram:dm:12345".to_string(),
            ActiveSession {
                session_key: "agent:main:telegram:dm:12345".to_string(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let plan = runtime.plan_shutdown_notification_host().unwrap();
        assert_eq!(plan.reply, None);
        assert_eq!(plan.actions.len(), 1);
        assert!(matches!(
            &plan.actions[0],
            GatewayHostAction::SendNotification { notification }
                if notification.chat_id == "12345"
        ));
    }

    #[test]
    fn shutdown_notifications_do_not_dedupe_across_threads() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "999", "Ops Home", None).unwrap(),
        );
        let mut runtime =
            GatewayRuntime::new(temp.path(), config, BusyInputMode::Interrupt).unwrap();
        let source = SessionSource {
            platform: telegram(),
            chat_id: "999".to_string(),
            chat_name: None,
            chat_type: "group".to_string(),
            user_id: Some("u1".to_string()),
            user_name: None,
            thread_id: Some("topic-7".to_string()),
            chat_topic: None,
            user_id_alt: None,
            chat_id_alt: None,
            is_bot: false,
            guild_id: None,
            parent_chat_id: None,
            message_id: None,
        };
        let session_key = build_session_key(&source, true, false).unwrap();
        runtime.session_store.entries.insert(
            session_key.clone(),
            SessionEntry {
                session_key: session_key.clone(),
                session_id: "sess-1".to_string(),
                created_at: parse_datetime("2026-05-25T10:00:00").unwrap(),
                updated_at: parse_datetime("2026-05-25T10:00:00").unwrap(),
                origin: Some(source),
                display_name: None,
                platform: Some(telegram()),
                chat_type: "group".to_string(),
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                total_tokens: 0,
                estimated_cost_usd: 0.0,
                cost_status: "unknown".to_string(),
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
            },
        );
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key,
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let notifications = runtime.plan_shutdown_notifications().unwrap();
        assert_eq!(notifications.len(), 2);
        assert_eq!(notifications[0].thread_id.as_deref(), Some("topic-7"));
        assert_eq!(notifications[1].thread_id, None);
    }

    #[test]
    fn runtime_execute_shutdown_post_drain_writes_clean_marker_and_schedules_detached_restart() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime.active_sessions.insert(
            created.session_key.clone(),
            ActiveSession {
                session_key: created.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime.begin_shutdown(true, true, false).unwrap();
        runtime.active_sessions.remove(&created.session_key);

        let mut handler = RecordingHostHandler::default();
        let report = runtime
            .execute_shutdown_post_drain(temp.path(), false, &mut handler)
            .unwrap();

        assert_eq!(handler.operations, vec!["restart:Detached".to_string()]);
        assert!(gateway_clean_shutdown_path(temp.path()).exists());
        assert_eq!(
            report.restart_failure_counts,
            HashMap::from([(created.session_key.clone(), 1)])
        );
        assert_eq!(
            report.host_actions.scheduled_restarts,
            vec![RestartLaunchMode::Detached]
        );
        assert!(report.outcome.write_clean_shutdown_marker);
        assert_eq!(report.outcome.exit_code, None);
    }

    #[test]
    fn runtime_execute_shutdown_post_drain_skips_clean_marker_for_timeout_and_service_restart() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime.active_sessions.insert(
            created.session_key.clone(),
            ActiveSession {
                session_key: created.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime.begin_shutdown(true, false, true).unwrap();

        let mut handler = RecordingHostHandler::default();
        let report = runtime
            .execute_shutdown_post_drain(temp.path(), true, &mut handler)
            .unwrap();

        assert!(handler.operations.is_empty());
        assert!(!gateway_clean_shutdown_path(temp.path()).exists());
        assert!(report.host_actions.scheduled_restarts.is_empty());
        assert_eq!(
            report.restart_failure_counts,
            HashMap::from([(created.session_key.clone(), 1)])
        );
        assert!(!report.outcome.write_clean_shutdown_marker);
        assert_eq!(
            report.outcome.exit_code,
            Some(GATEWAY_SERVICE_RESTART_EXIT_CODE)
        );
        assert!(
            runtime
                .session_store
                .entry(&created.session_key)
                .unwrap()
                .unwrap()
                .resume_pending
        );
    }

    #[test]
    fn shutdown_notifications_use_persisted_origin_for_colon_ids() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let source = SessionSource {
            platform: matrix(),
            chat_id: "!room123:example.org".to_string(),
            chat_name: None,
            chat_type: "group".to_string(),
            user_id: Some("u1".to_string()),
            user_name: None,
            thread_id: None,
            chat_topic: None,
            user_id_alt: None,
            chat_id_alt: None,
            is_bot: false,
            guild_id: None,
            parent_chat_id: None,
            message_id: None,
        };
        let session_key = build_session_key(&source, true, false).unwrap();
        runtime.session_store.entries.insert(
            session_key.clone(),
            SessionEntry {
                session_key: session_key.clone(),
                session_id: "sess-1".to_string(),
                created_at: parse_datetime("2026-05-25T10:00:00").unwrap(),
                updated_at: parse_datetime("2026-05-25T10:00:00").unwrap(),
                origin: Some(source),
                display_name: None,
                platform: Some(matrix()),
                chat_type: "group".to_string(),
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                total_tokens: 0,
                estimated_cost_usd: 0.0,
                cost_status: "unknown".to_string(),
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
            },
        );
        runtime.active_sessions.insert(
            session_key.clone(),
            ActiveSession {
                session_key,
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );

        let notifications = runtime.plan_shutdown_notifications().unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].platform, matrix());
        assert_eq!(notifications[0].chat_id, "!room123:example.org");
    }

    #[test]
    fn startup_home_notifications_preserve_thread_and_skip_exact_target() {
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", Some("topic-7".to_string())).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let skip =
            std::collections::HashSet::from([("slack".to_string(), "C123".to_string(), None)]);

        let notifications = plan_startup_home_channel_notifications(&config, &skip);
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].platform, telegram());
        assert_eq!(notifications[0].thread_id.as_deref(), Some("topic-7"));
        assert_eq!(
            notifications[0].metadata(),
            Some(HashMap::from([(
                "thread_id".to_string(),
                "topic-7".to_string()
            )]))
        );
        assert!(notifications[0].message.contains("Gateway online"));
    }

    #[test]
    fn stale_restart_redelivery_only_applies_to_recent_telegram_marker() {
        let event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(100),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let marker = RestartDedupMarker {
            platform: telegram(),
            update_id: Some(100),
            requested_at: 1000.0,
        };
        assert!(is_stale_restart_redelivery(&event, Some(&marker), 1200.0));
        assert!(!is_stale_restart_redelivery(&event, Some(&marker), 1401.0));

        let mut non_telegram = event.clone();
        non_telegram.source.platform = slack();
        assert!(!is_stale_restart_redelivery(
            &non_telegram,
            Some(&marker),
            1200.0
        ));
    }

    #[test]
    fn restart_command_plan_ignores_stale_redelivery() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(42),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let marker = RestartDedupMarker {
            platform: telegram(),
            update_id: Some(42),
            requested_at: 1000.0,
        };
        assert_eq!(
            runtime
                .plan_restart_command(&event, false, 1200.0, Some(&marker))
                .unwrap(),
            RestartCommandDecision::IgnoreRedelivery
        );
    }

    #[test]
    fn restart_command_plan_selects_service_or_detached_mode_and_markers() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let mut event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(99),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        event.source.thread_id = Some("topic-7".to_string());
        let decision = runtime
            .plan_restart_command(&event, true, 1234.0, None)
            .unwrap();
        let RestartCommandDecision::BeginRestart {
            launch_mode,
            notify_marker,
            dedup_marker,
            message,
        } = decision
        else {
            panic!("expected BeginRestart");
        };
        assert_eq!(launch_mode, RestartLaunchMode::Service);
        assert_eq!(notify_marker.chat_id, "12345");
        assert_eq!(notify_marker.thread_id.as_deref(), Some("topic-7"));
        assert_eq!(dedup_marker.update_id, Some(99));
        assert_eq!(dedup_marker.requested_at, 1234.0);
        assert!(message.contains("Restarting gateway"));
        assert!(runtime.restart_requested());
    }

    #[test]
    fn restart_command_plan_returns_draining_message_when_busy() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        runtime.active_sessions.insert(
            "agent:main:telegram:dm:12345".to_string(),
            ActiveSession {
                session_key: "agent:main:telegram:dm:12345".to_string(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime.request_restart(true, false).unwrap();

        let event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(1),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        assert_eq!(
            runtime
                .plan_restart_command(&event, false, 10.0, None)
                .unwrap(),
            RestartCommandDecision::AlreadyInProgress {
                message: "⏳ Draining 1 active agent(s) before restart...".to_string()
            }
        );
    }

    #[test]
    fn restart_notify_marker_maps_to_notification() {
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: Some("topic-7".to_string()),
        };
        let notification = marker.to_notification();
        assert_eq!(notification.chat_id, "42");
        assert_eq!(notification.thread_id.as_deref(), Some("topic-7"));
        assert!(notification.message.contains("restarted successfully"));
        assert_eq!(
            notification.metadata(),
            Some(HashMap::from([(
                "thread_id".to_string(),
                "topic-7".to_string()
            )]))
        );
    }

    #[test]
    fn restart_command_host_plan_writes_markers_and_schedules_restart() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(77),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };

        let plan = runtime
            .plan_restart_command_host(temp.path(), &event, false, 1234.0, None)
            .unwrap();
        assert!(
            plan.reply
                .as_deref()
                .unwrap()
                .contains("Restarting gateway")
        );
        assert_eq!(plan.actions.len(), 3);
        assert!(matches!(
            &plan.actions[0],
            GatewayHostAction::WriteJson { path, .. } if path == &gateway_restart_notify_path(temp.path())
        ));
        assert!(matches!(
            &plan.actions[1],
            GatewayHostAction::WriteJson { path, .. } if path == &gateway_restart_last_processed_path(temp.path())
        ));
        assert_eq!(
            plan.actions[2],
            GatewayHostAction::ScheduleRestart {
                launch_mode: RestartLaunchMode::Detached
            }
        );
    }

    #[test]
    fn restart_notification_host_plan_sends_then_cleans_up() {
        let temp = tempdir().unwrap();
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: Some("topic-7".to_string()),
        };
        let plan = plan_restart_notification_host(temp.path(), Some(&marker));
        assert_eq!(plan.reply, None);
        assert_eq!(plan.actions.len(), 2);
        assert_eq!(
            plan.actions[0],
            GatewayHostAction::SendNotification {
                notification: marker.to_notification()
            }
        );
        assert_eq!(
            plan.actions[1],
            GatewayHostAction::DeleteFile {
                path: gateway_restart_notify_path(temp.path())
            }
        );
    }

    #[test]
    fn restart_notification_host_plan_cleans_up_without_send_when_missing_marker() {
        let temp = tempdir().unwrap();
        let plan = plan_restart_notification_host(temp.path(), None);
        assert_eq!(
            plan.actions,
            vec![GatewayHostAction::DeleteFile {
                path: gateway_restart_notify_path(temp.path())
            }]
        );
    }

    #[test]
    fn startup_restart_home_notifications_host_skips_exact_delivered_target() {
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", None).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );

        let plan = plan_startup_restart_home_notifications_host(
            &config,
            true,
            Some(("telegram".to_string(), "42".to_string(), None)),
        );
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(
            plan.actions[0],
            GatewayHostAction::SendNotification {
                notification: GatewayNotification {
                    platform: slack(),
                    chat_id: "C123".to_string(),
                    thread_id: None,
                    message: "♻️ Gateway online — Hermes is back and ready.".to_string(),
                }
            }
        );
    }

    #[test]
    fn startup_restart_home_notifications_host_noops_without_restart_context() {
        let plan =
            plan_startup_restart_home_notifications_host(&RuntimeConfig::default(), false, None);
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn startup_restart_flow_host_branches_home_notifications_on_delivery_result() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", None).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: None,
        };

        let flow = plan_startup_restart_flow_host(temp.path(), &config, Some(&marker));
        assert_eq!(
            flow.restart_notification,
            plan_restart_notification_host(temp.path(), Some(&marker))
        );
        assert_eq!(flow.home_notifications_if_delivered.actions.len(), 1);
        assert_eq!(flow.home_notifications_if_not_delivered.actions.len(), 2);
        assert!(matches!(
            &flow.home_notifications_if_delivered.actions[0],
            GatewayHostAction::SendNotification { notification }
                if notification.platform == slack()
        ));
    }

    #[test]
    fn startup_restart_flow_host_noops_without_marker() {
        let flow = plan_startup_restart_flow_host(
            Path::new("/tmp/runtime"),
            &RuntimeConfig::default(),
            None,
        );
        assert!(flow.restart_notification.actions.len() == 1);
        assert!(flow.home_notifications_if_delivered.actions.is_empty());
        assert!(flow.home_notifications_if_not_delivered.actions.is_empty());
    }

    #[test]
    fn execute_host_actions_sends_before_deleting_restart_marker() {
        let temp = tempdir().unwrap();
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: Some("topic-7".to_string()),
        };
        let marker_path = gateway_restart_notify_path(temp.path());
        fs::write(&marker_path, "{\"pending\":true}").unwrap();

        let plan = plan_restart_notification_host(temp.path(), Some(&marker));
        let mut handler = RecordingHostHandler::with_notification_results([false]);
        handler.notify_path = Some(marker_path.clone());

        let report = execute_host_actions(&plan.actions, &mut handler).unwrap();

        assert_eq!(
            handler.operations,
            vec!["send:telegram:42:topic-7".to_string()]
        );
        assert_eq!(handler.notify_path_exists_during_send, vec![true]);
        assert!(!marker_path.exists());
        assert_eq!(
            report.failed_notification_targets,
            vec![(
                "telegram".to_string(),
                "42".to_string(),
                Some("topic-7".to_string())
            )]
        );
    }

    #[test]
    fn execute_startup_restart_flow_uses_delivered_branch_after_successful_direct_send() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", None).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: None,
        };
        let flow = plan_startup_restart_flow_host(temp.path(), &config, Some(&marker));
        let mut handler = RecordingHostHandler::with_notification_results([true, true]);

        let report = execute_startup_restart_flow(&flow, &mut handler).unwrap();

        assert_eq!(
            handler.operations,
            vec![
                "send:telegram:42:".to_string(),
                "send:slack:C123:".to_string(),
            ]
        );
        assert_eq!(
            report.delivered_notification_targets,
            vec![
                ("telegram".to_string(), "42".to_string(), None),
                ("slack".to_string(), "C123".to_string(), None),
            ]
        );
    }

    #[test]
    fn execute_startup_restart_flow_uses_fallback_branch_after_failed_direct_send() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", None).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: None,
        };
        let flow = plan_startup_restart_flow_host(temp.path(), &config, Some(&marker));
        let mut handler = RecordingHostHandler::with_notification_results([false, true, true]);

        let report = execute_startup_restart_flow(&flow, &mut handler).unwrap();

        assert_eq!(
            handler.operations,
            vec![
                "send:telegram:42:".to_string(),
                "send:slack:C123:".to_string(),
                "send:telegram:42:".to_string(),
            ]
        );
        assert_eq!(
            report.failed_notification_targets,
            vec![("telegram".to_string(), "42".to_string(), None)]
        );
        assert_eq!(
            report.delivered_notification_targets,
            vec![
                ("slack".to_string(), "C123".to_string(), None),
                ("telegram".to_string(), "42".to_string(), None),
            ]
        );
    }

    #[test]
    fn runtime_execute_startup_host_flow_skips_suspend_after_clean_shutdown() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        fs::write(gateway_clean_shutdown_path(temp.path()), "").unwrap();

        let mut handler = RecordingHostHandler::default();
        let report = runtime
            .execute_startup_host_flow(temp.path(), &mut handler)
            .unwrap();

        assert_eq!(
            report,
            GatewayStartupExecutionReport {
                recovery: GatewayStartupRecovery {
                    consume_clean_shutdown_marker: true,
                    suspend_recently_active: false,
                },
                suspended_recent_sessions: 0,
                suspended_stuck_loop_sessions: 0,
                host_actions: GatewayHostExecutionReport::empty(),
            }
        );
        assert!(!gateway_clean_shutdown_path(temp.path()).exists());
        let entry = runtime
            .session_store
            .entry(&created.session_key)
            .unwrap()
            .unwrap();
        assert!(!entry.resume_pending);
    }

    #[test]
    fn runtime_execute_startup_host_flow_suspends_recent_sessions_and_sends_restart_flow() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            telegram(),
            HomeChannel::new(telegram(), "42", "Ops Home", None).unwrap(),
        );
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let mut runtime =
            GatewayRuntime::new(temp.path(), config, BusyInputMode::Interrupt).unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: None,
        };
        fs::write(
            gateway_restart_notify_path(temp.path()),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();

        let mut handler = RecordingHostHandler::with_notification_results([true, true]);
        let report = runtime
            .execute_startup_host_flow(temp.path(), &mut handler)
            .unwrap();

        assert_eq!(
            handler.operations,
            vec![
                "send:telegram:42:".to_string(),
                "send:slack:C123:".to_string(),
            ]
        );
        assert_eq!(report.suspended_recent_sessions, 1);
        assert_eq!(report.suspended_stuck_loop_sessions, 0);
        assert_eq!(
            report.recovery,
            GatewayStartupRecovery {
                consume_clean_shutdown_marker: false,
                suspend_recently_active: true,
            }
        );
        assert!(!gateway_restart_notify_path(temp.path()).exists());
        let entry = runtime
            .session_store
            .entry(&created.session_key)
            .unwrap()
            .unwrap();
        assert!(entry.resume_pending);
        assert_eq!(entry.resume_reason.as_deref(), Some("restart_interrupted"));
        assert_eq!(
            report.host_actions.delivered_notification_targets,
            vec![
                ("telegram".to_string(), "42".to_string(), None),
                ("slack".to_string(), "C123".to_string(), None),
            ]
        );
    }

    #[test]
    fn restart_failure_counts_increment_active_sessions_and_drop_inactive_ones() {
        let temp = tempdir().unwrap();
        let first = update_restart_failure_counts(
            temp.path(),
            &["session-a".to_string(), "session-b".to_string()],
        )
        .unwrap();
        assert_eq!(first.get("session-a"), Some(&1));
        assert_eq!(first.get("session-b"), Some(&1));

        let second = update_restart_failure_counts(
            temp.path(),
            &["session-b".to_string(), "session-c".to_string()],
        )
        .unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(second.get("session-b"), Some(&2));
        assert_eq!(second.get("session-c"), Some(&1));
        assert!(!second.contains_key("session-a"));
    }

    #[test]
    fn clear_restart_failure_count_removes_entry_and_deletes_empty_file() {
        let temp = tempdir().unwrap();
        update_restart_failure_counts(
            temp.path(),
            &["session-a".to_string(), "session-b".to_string()],
        )
        .unwrap();
        assert!(clear_restart_failure_count(temp.path(), "session-a").unwrap());
        let counts = read_restart_failure_counts(temp.path()).unwrap();
        assert_eq!(counts, HashMap::from([("session-b".to_string(), 1)]));

        assert!(clear_restart_failure_count(temp.path(), "session-b").unwrap());
        assert!(!gateway_restart_failure_counts_path(temp.path()).exists());
    }

    #[test]
    fn suspend_stuck_loop_sessions_marks_threshold_entries_and_clears_file() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let a = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        let mut source_b = session_source("dm");
        source_b.chat_id = "67890".to_string();
        let b = runtime
            .session_store
            .get_or_create_session(&source_b, false)
            .unwrap();
        write_restart_failure_counts(
            temp.path(),
            &HashMap::from([(a.session_key.clone(), 3), (b.session_key.clone(), 2)]),
        )
        .unwrap();

        let suspended =
            suspend_stuck_loop_sessions(temp.path(), &mut runtime.session_store, 3).unwrap();
        assert_eq!(suspended, 1);
        assert!(!gateway_restart_failure_counts_path(temp.path()).exists());
        assert!(
            runtime
                .session_store
                .entry(&a.session_key)
                .unwrap()
                .unwrap()
                .suspended
        );
        assert!(
            !runtime
                .session_store
                .entry(&b.session_key)
                .unwrap()
                .unwrap()
                .suspended
        );
    }

    #[test]
    fn runtime_execute_startup_host_flow_suspends_stuck_loop_sessions() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        write_restart_failure_counts(
            temp.path(),
            &HashMap::from([(created.session_key.clone(), STUCK_LOOP_THRESHOLD)]),
        )
        .unwrap();

        let mut handler = RecordingHostHandler::default();
        let report = runtime
            .execute_startup_host_flow(temp.path(), &mut handler)
            .unwrap();

        assert_eq!(report.suspended_stuck_loop_sessions, 1);
        assert!(
            runtime
                .session_store
                .entry(&created.session_key)
                .unwrap()
                .unwrap()
                .suspended
        );
        assert!(!gateway_restart_failure_counts_path(temp.path()).exists());
    }

    #[test]
    fn runtime_record_successful_turn_clears_resume_pending_and_restart_failure_count() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime
            .session_store
            .mark_resume_pending(&created.session_key, "restart_timeout")
            .unwrap();
        write_restart_failure_counts(
            temp.path(),
            &HashMap::from([(created.session_key.clone(), 2)]),
        )
        .unwrap();

        let report = runtime
            .record_successful_turn(temp.path(), &created.session_key)
            .unwrap();
        assert_eq!(
            report,
            GatewayTurnCompletionReport {
                cleared_restart_failure_count: true,
                cleared_resume_pending: true,
            }
        );
        assert!(!gateway_restart_failure_counts_path(temp.path()).exists());
        let entry = runtime
            .session_store
            .entry(&created.session_key)
            .unwrap()
            .unwrap();
        assert!(!entry.resume_pending);
        assert_eq!(entry.resume_reason, None);
    }

    #[test]
    fn runtime_complete_turn_returns_cleanup_report_without_next_turn_when_queue_empty() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime.active_sessions.insert(
            created.session_key.clone(),
            ActiveSession {
                session_key: created.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime
            .session_store
            .mark_resume_pending(&created.session_key, "restart_timeout")
            .unwrap();
        write_restart_failure_counts(
            temp.path(),
            &HashMap::from([(created.session_key.clone(), 1)]),
        )
        .unwrap();

        let report = runtime
            .complete_turn(temp.path(), &created.session_key)
            .unwrap();
        assert_eq!(
            report,
            GatewayTurnFinishReport {
                completion: GatewayTurnCompletionReport {
                    cleared_restart_failure_count: true,
                    cleared_resume_pending: true,
                },
                next: None,
            }
        );
        assert!(runtime.active_session(&created.session_key).is_none());
    }

    #[test]
    fn runtime_complete_turn_cleans_up_and_promotes_queued_next_turn() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let created = runtime
            .session_store
            .get_or_create_session(&session_source("dm"), false)
            .unwrap();
        runtime.active_sessions.insert(
            created.session_key.clone(),
            ActiveSession {
                session_key: created.session_key.clone(),
                phase: ActiveSessionPhase::Running { can_steer: false },
            },
        );
        runtime
            .session_store
            .mark_resume_pending(&created.session_key, "restart_timeout")
            .unwrap();
        write_restart_failure_counts(
            temp.path(),
            &HashMap::from([(created.session_key.clone(), 2)]),
        )
        .unwrap();
        runtime.enqueue_fifo_follow_up(
            &created.session_key,
            MessageEvent {
                text: "follow up".to_string(),
                message_type: MessageType::Text,
                source: session_source("dm"),
                message_id: None,
                platform_update_id: None,
                media_urls: Vec::new(),
                media_types: Vec::new(),
                reply_to_message_id: None,
                reply_to_text: None,
                channel_prompt: None,
                internal: false,
            },
        );

        let report = runtime
            .complete_turn(temp.path(), &created.session_key)
            .unwrap();
        assert_eq!(
            report.completion,
            GatewayTurnCompletionReport {
                cleared_restart_failure_count: true,
                cleared_resume_pending: true,
            }
        );
        let Some(GatewayIngressDecision::StartTurn {
            session_key, event, ..
        }) = report.next
        else {
            panic!("expected queued StartTurn");
        };
        assert_eq!(session_key, created.session_key);
        assert_eq!(event.text, "follow up");
        assert_eq!(runtime.queue_depth(&created.session_key), 0);
    }

    #[test]
    fn startup_host_flow_deletes_clean_marker_and_skips_suspend_after_clean_exit() {
        let temp = tempdir().unwrap();
        let flow = plan_startup_host_flow(temp.path(), &RuntimeConfig::default(), true, None);
        assert_eq!(
            flow.recovery,
            GatewayStartupRecovery {
                consume_clean_shutdown_marker: true,
                suspend_recently_active: false,
            }
        );
        assert_eq!(
            flow.pre_actions,
            vec![GatewayHostAction::DeleteFile {
                path: gateway_clean_shutdown_path(temp.path())
            }]
        );
    }

    #[test]
    fn startup_host_flow_suspends_recent_sessions_after_unclean_exit() {
        let temp = tempdir().unwrap();
        let flow = plan_startup_host_flow(temp.path(), &RuntimeConfig::default(), false, None);
        assert_eq!(
            flow.recovery,
            GatewayStartupRecovery {
                consume_clean_shutdown_marker: false,
                suspend_recently_active: true,
            }
        );
        assert!(flow.pre_actions.is_empty());
    }

    #[test]
    fn startup_host_flow_embeds_restart_marker_flow() {
        let temp = tempdir().unwrap();
        let mut config = RuntimeConfig::default();
        config.home_channels.insert(
            slack(),
            HomeChannel::new(slack(), "C123", "Slack Home", None).unwrap(),
        );
        let marker = RestartNotifyMarker {
            platform: telegram(),
            chat_id: "42".to_string(),
            thread_id: None,
        };
        let flow = plan_startup_host_flow(temp.path(), &config, false, Some(&marker));
        assert_eq!(
            flow.restart,
            plan_startup_restart_flow_host(temp.path(), &config, Some(&marker))
        );
    }

    #[test]
    fn restart_notification_pending_reflects_marker_presence() {
        let temp = tempdir().unwrap();
        assert!(!restart_notification_pending(temp.path()));
        fs::write(gateway_restart_notify_path(temp.path()), "{}").unwrap();
        assert!(restart_notification_pending(temp.path()));
    }

    #[test]
    fn clean_shutdown_marker_exists_reflects_marker_presence() {
        let temp = tempdir().unwrap();
        assert!(!clean_shutdown_marker_exists(temp.path()));
        fs::write(gateway_clean_shutdown_path(temp.path()), "").unwrap();
        assert!(clean_shutdown_marker_exists(temp.path()));
    }

    #[test]
    fn filesystem_host_actions_write_and_read_restart_markers() {
        let temp = tempdir().unwrap();
        let mut runtime = GatewayRuntime::new(
            temp.path(),
            RuntimeConfig::default(),
            BusyInputMode::Interrupt,
        )
        .unwrap();
        let event = MessageEvent {
            text: "/restart".to_string(),
            message_type: MessageType::Text,
            source: session_source("dm"),
            message_id: None,
            platform_update_id: Some(88),
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            channel_prompt: None,
            internal: false,
        };
        let plan = runtime
            .plan_restart_command_host(temp.path(), &event, false, 2345.0, None)
            .unwrap();
        let deferred = apply_filesystem_host_actions(&plan.actions).unwrap();
        assert_eq!(
            deferred,
            vec![GatewayHostAction::ScheduleRestart {
                launch_mode: RestartLaunchMode::Detached
            }]
        );

        let notify = read_restart_notify_marker(temp.path()).unwrap().unwrap();
        assert_eq!(notify.chat_id, "12345");
        let dedup = read_restart_dedup_marker(temp.path()).unwrap().unwrap();
        assert_eq!(dedup.update_id, Some(88));
        assert_eq!(dedup.requested_at, 2345.0);
    }

    #[test]
    fn filesystem_host_actions_delete_marker_files() {
        let temp = tempdir().unwrap();
        fs::write(gateway_restart_notify_path(temp.path()), "{}").unwrap();
        fs::write(gateway_clean_shutdown_path(temp.path()), "").unwrap();

        let deferred = apply_filesystem_host_actions(&[
            GatewayHostAction::DeleteFile {
                path: gateway_restart_notify_path(temp.path()),
            },
            GatewayHostAction::DeleteFile {
                path: gateway_clean_shutdown_path(temp.path()),
            },
        ])
        .unwrap();
        assert!(deferred.is_empty());
        assert!(!gateway_restart_notify_path(temp.path()).exists());
        assert!(!gateway_clean_shutdown_path(temp.path()).exists());
    }

    #[test]
    fn stream_clean_for_display_strips_media_tags() {
        let text = "Here is the image\nMEDIA:/tmp/test.png\n[[audio_as_voice]]";
        let cleaned = GatewayStreamPlanner::clean_for_display(text);
        assert!(!cleaned.contains("MEDIA:"));
        assert!(!cleaned.contains("[[audio_as_voice]]"));
        assert!(cleaned.contains("Here is the image"));
    }

    #[test]
    fn stream_planner_strips_think_blocks_but_not_prose_mentions() {
        let mut planner = GatewayStreamPlanner::new(StreamConsumerConfig::default());
        planner.process(
            StreamInput::Delta("<think>hidden</think>answer".to_string()),
            0.0,
        );
        let ops = planner.process(StreamInput::Finish, 0.0);
        assert_eq!(
            ops,
            vec![StreamOp::SendNew {
                text: "answer".to_string()
            }]
        );

        let mut prose = GatewayStreamPlanner::new(StreamConsumerConfig::default());
        prose.process(
            StreamInput::Delta("The <think> tag is used for reasoning".to_string()),
            0.0,
        );
        let ops = prose.process(StreamInput::Finish, 0.0);
        assert_eq!(
            ops,
            vec![StreamOp::SendNew {
                text: "The <think> tag is used for reasoning".to_string()
            }]
        );
    }

    #[test]
    fn stream_planner_buffer_only_flushes_on_structural_boundaries() {
        let mut planner = GatewayStreamPlanner::new(StreamConsumerConfig {
            edit_interval: 0.01,
            buffer_threshold: 5,
            cursor: String::new(),
            buffer_only: true,
            fresh_final_after_seconds: 0.0,
        });

        assert!(
            planner
                .process(StreamInput::Delta("Hello".to_string()), 0.0)
                .is_empty()
        );
        assert!(
            planner
                .process(StreamInput::Delta(" world".to_string()), 0.05)
                .is_empty()
        );

        let ops = planner.process(
            StreamInput::Commentary("I'll search first.".to_string()),
            0.06,
        );
        assert_eq!(
            ops,
            vec![
                StreamOp::SendNew {
                    text: "Hello world".to_string()
                },
                StreamOp::SendCommentary {
                    text: "I'll search first.".to_string()
                }
            ]
        );

        planner.process(StreamInput::Delta("Result".to_string()), 0.07);
        let ops = planner.process(StreamInput::Finish, 0.08);
        assert_eq!(
            ops,
            vec![StreamOp::SendNew {
                text: "Result".to_string()
            }]
        );
    }

    #[test]
    fn stream_planner_short_cursor_first_send_is_suppressed() {
        let mut planner = GatewayStreamPlanner::new(StreamConsumerConfig {
            edit_interval: 0.01,
            buffer_threshold: 1,
            cursor: " ▉".to_string(),
            buffer_only: false,
            fresh_final_after_seconds: 0.0,
        });

        let ops = planner.process(StreamInput::Delta("I".to_string()), 1.0);
        assert!(ops.is_empty());
        assert!(!planner.already_sent());

        let ops = planner.process(StreamInput::Delta("ello".to_string()), 2.0);
        assert_eq!(
            ops,
            vec![StreamOp::SendNew {
                text: "Iello ▉".to_string()
            }]
        );
        assert!(planner.already_sent());
    }

    #[test]
    fn stream_planner_segment_break_finalizes_and_resets() {
        let mut planner = GatewayStreamPlanner::new(StreamConsumerConfig {
            edit_interval: 0.01,
            buffer_threshold: 1,
            cursor: " ▉".to_string(),
            buffer_only: false,
            fresh_final_after_seconds: 0.0,
        });

        let _ = planner.process(StreamInput::Delta("Thinking".to_string()), 1.0);
        let ops = planner.process(StreamInput::SegmentBreak, 1.1);
        assert_eq!(
            ops,
            vec![StreamOp::EditCurrent {
                text: "Thinking".to_string(),
                finalize: true
            }]
        );

        let ops = planner.process(StreamInput::Delta("Done".to_string()), 2.0);
        assert_eq!(
            ops,
            vec![StreamOp::SendNew {
                text: "Done ▉".to_string()
            }]
        );
    }

    #[test]
    fn stream_planner_commentary_stays_separate_from_final_stream() {
        let mut planner = GatewayStreamPlanner::new(StreamConsumerConfig {
            edit_interval: 0.01,
            buffer_threshold: 1,
            cursor: String::new(),
            buffer_only: true,
            fresh_final_after_seconds: 0.0,
        });
        planner.process(StreamInput::Delta("Working on it...".to_string()), 0.0);
        let ops = planner.process(
            StreamInput::Commentary("I'll inspect the repository first.".to_string()),
            0.1,
        );
        assert_eq!(
            ops,
            vec![
                StreamOp::SendNew {
                    text: "Working on it...".to_string()
                },
                StreamOp::SendCommentary {
                    text: "I'll inspect the repository first.".to_string()
                }
            ]
        );
    }

    #[test]
    fn stream_execution_no_message_id_arms_no_edit_fallback() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello ▉", None, 1.0);

        assert_eq!(state.message_handle(), Some(&StreamMessageHandle::NoEdit));
        assert!(state.already_sent());
        assert!(state.fallback_final_send());
        assert_eq!(state.visible_prefix(), "Hello");
        assert_eq!(state.continuation_text("Hello world"), "world");
    }

    #[test]
    fn stream_execution_preserves_no_edit_handle_across_segment_reset() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Phase 1", None, 1.0);
        state.reset_segment_state(true);
        assert_eq!(state.message_handle(), Some(&StreamMessageHandle::NoEdit));

        state.reset_segment_state(false);
        assert!(state.message_handle().is_none());
    }

    #[test]
    fn stream_execution_flood_failures_retry_then_enter_fallback() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello world ▉", Some("msg-1"), 1.0);

        assert_eq!(
            state.record_edit_failure(Some("flood_control:6")),
            StreamEditFailureOutcome::RetryLater
        );
        assert_eq!(state.current_edit_interval(), 2.0);

        assert_eq!(
            state.record_edit_failure(Some("flood_control:6")),
            StreamEditFailureOutcome::RetryLater
        );
        assert_eq!(state.current_edit_interval(), 4.0);

        assert_eq!(
            state.record_edit_failure(Some("flood_control:6")),
            StreamEditFailureOutcome::EnterFallback {
                strip_cursor: Some(StreamFollowUpEdit::StripCursor {
                    message_id: "msg-1".to_string(),
                    text: "Hello world".to_string(),
                })
            }
        );
        assert!(state.fallback_final_send());
    }

    #[test]
    fn stream_execution_fallback_final_sends_full_text_when_prefix_is_stale() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("I'll run that code now.", Some("msg-1"), 1.0);
        let _ = state.record_edit_failure(Some("network"));
        let plan = state.plan_fallback_final("Script timed out after 30s and was killed.");

        assert_eq!(
            plan,
            StreamFallbackFinalPlan::SendChunks {
                chunks: vec!["Script timed out after 30s and was killed.".to_string()]
            }
        );
    }

    #[test]
    fn stream_execution_fallback_final_splits_long_continuation() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 610, false);
        let prefix = "Hello world";
        let tail = "x".repeat(620);
        state.record_first_send_success(prefix, Some("msg-1"), 1.0);
        let _ = state.record_edit_failure(Some("flood_control:6"));
        let plan = state.plan_fallback_final(&format!("{prefix}{tail}"));

        let StreamFallbackFinalPlan::SendChunks { chunks } = plan else {
            panic!("expected chunked fallback plan");
        };
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks.concat(), tail);
    }

    #[test]
    fn stream_execution_fresh_final_gates_on_age_and_real_message_id() {
        let config = StreamConsumerConfig {
            edit_interval: 1.0,
            buffer_threshold: 40,
            cursor: " ▉".to_string(),
            buffer_only: false,
            fresh_final_after_seconds: 60.0,
        };
        let mut state = GatewayStreamExecutionState::new(config.clone(), 4096, false);
        state.record_first_send_success("hello ▉", Some("msg-1"), 10.0);
        assert!(!state.should_send_fresh_final(69.0));
        assert!(state.should_send_fresh_final(70.0));
        assert_eq!(
            state.plan_delivery_action("hello world", true, 70.0),
            StreamDeliveryAction::FreshFinal {
                old_message_id: "msg-1".to_string(),
                text: "hello world".to_string(),
            }
        );

        let mut no_edit = GatewayStreamExecutionState::new(config, 4096, false);
        no_edit.record_first_send_success("hello ▉", None, 10.0);
        assert!(!no_edit.should_send_fresh_final(100.0));
    }

    #[test]
    fn stream_execution_successful_fallback_completion_marks_final_sent() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello ▉", None, 1.0);
        state.mark_fallback_delivery_success(Some("msg-2"), Some("world"));

        assert_eq!(
            state.message_handle(),
            Some(&StreamMessageHandle::Editable("msg-2".to_string()))
        );
        assert!(state.already_sent());
        assert!(state.final_response_sent());
        assert_eq!(state.visible_prefix(), "world");
    }

    #[test]
    fn stream_execution_finish_marks_visible_final_when_finalize_not_required() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Done.", Some("msg-1"), 1.0);
        assert_eq!(
            state.plan_finish("Done.", true, 2.0),
            StreamFinishPlan::MarkFinalSent
        );
        state.mark_final_response_sent();
        assert!(state.final_response_sent());
    }

    #[test]
    fn stream_execution_finish_requests_finalize_edit_when_required() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, true);
        state.record_first_send_success("Done.", Some("msg-1"), 1.0);
        assert_eq!(
            state.plan_finish("Done.", true, 2.0),
            StreamFinishPlan::Deliver(StreamDeliveryAction::Edit {
                message_id: "msg-1".to_string(),
                text: "Done.".to_string(),
                finalize: true,
            })
        );
    }

    #[test]
    fn stream_execution_segment_break_flushes_only_unsent_tail() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello ▉", Some("msg-1"), 1.0);
        let plan = state.plan_segment_break("Hello world", false);
        assert_eq!(
            plan,
            StreamSegmentBreakPlan::FlushTail {
                text: "world".to_string(),
                strip_cursor: Some(StreamFollowUpEdit::StripCursor {
                    message_id: "msg-1".to_string(),
                    text: "Hello".to_string(),
                }),
            }
        );
    }

    #[test]
    fn stream_execution_segment_break_uses_fallback_prefix_without_cursor_strip() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello world ▉", Some("msg-1"), 1.0);
        let _ = state.record_edit_failure(Some("network"));
        let plan = state.plan_segment_break("Hello world more", false);
        assert_eq!(
            plan,
            StreamSegmentBreakPlan::FlushTail {
                text: "more".to_string(),
                strip_cursor: None,
            }
        );
    }

    #[test]
    fn stream_execution_cancellation_marks_final_only_after_best_effort_success() {
        let mut state =
            GatewayStreamExecutionState::new(StreamConsumerConfig::default(), 4096, false);
        state.record_first_send_success("Hello", Some("msg-1"), 1.0);
        state.handle_cancellation(false);
        assert!(!state.final_response_sent());
        state.handle_cancellation(true);
        assert!(state.final_response_sent());
    }
}
