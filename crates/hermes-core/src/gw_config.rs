//! Gateway configuration management.
//!
//! Native Rust port of `gateway/config.py`.
//!
//! Handles loading and validating configuration for:
//! - Connected platforms (Telegram, Discord, WhatsApp, ...)
//! - Home channels for each platform
//! - Session reset policies
//! - Delivery preferences
//!
//! The Python module leans heavily on dynamic dicts (`extra`, `quick_commands`,
//! arbitrary YAML), so this port keeps those as [`serde_json::Value`] maps for
//! faithful round-tripping. Cross-references:
//! - [`crate::mod_hermes_constants::get_hermes_home`]
//! - [`crate::mod_utils::is_truthy_value`]

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use serde_json::{Map, Value};

use crate::mod_hermes_constants::get_hermes_home;
use crate::mod_utils::{is_truthy_value, TruthyInput};

/// Environment variable name that overrides the path to `config.yaml`.
pub const OVERRIDE_CONFIG_PATH_ENV: &str = "HERMES_GATEWAY_CONFIG_PATH";

// ---------------------------------------------------------------------------
// Coercion helpers (mirror the module-level `_coerce_*` / `_normalize_*` fns).
// ---------------------------------------------------------------------------

/// Coerce a bool-ish [`Value`] preserving a caller-provided `default`.
///
/// Mirrors `_coerce_bool`:
/// - `None`/missing -> `default`
/// - string: `true/1/yes/on` -> true; `false/0/no/off` -> false; else `default`
/// - other -> `is_truthy_value(value, default)`
pub fn coerce_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::String(s)) => {
            let lowered = s.trim().to_lowercase();
            match lowered.as_str() {
                "true" | "1" | "yes" | "on" => true,
                "false" | "0" | "no" | "off" => false,
                _ => default,
            }
        }
        Some(Value::Bool(b)) => is_truthy_value(&TruthyInput::Bool(*b), default),
        Some(other) => {
            // Python `is_truthy_value` falls through to bool(value) for non-str
            // non-None objects. Reproduce truthiness for numbers/arrays/objects.
            let truthy = match other {
                Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
                _ => true,
            };
            is_truthy_value(&TruthyInput::Other(truthy), default)
        }
    }
}

/// Coerce a numeric [`Value`] to `f64`, falling back on malformed input.
///
/// Mirrors `_coerce_float`.
pub fn coerce_float(value: Option<&Value>, default: f64) -> f64 {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(default),
        Some(Value::Bool(b)) => {
            // Python float(True) == 1.0, float(False) == 0.0
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Some(_) => default,
    }
}

/// Coerce a numeric [`Value`] to `i64`, falling back on malformed input.
///
/// Mirrors `_coerce_int`. Python `int(float_str)` raises ValueError, so a
/// string like "1.5" falls back to default; "1.0" also falls back (int("1.0")
/// raises). We reproduce that: only integer-looking strings parse.
pub fn coerce_int(value: Option<&Value>, default: i64) -> i64 {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                // Python int(float) truncates toward zero.
                f.trunc() as i64
            } else {
                default
            }
        }
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(default),
        Some(Value::Bool(b)) => {
            if *b {
                1
            } else {
                0
            }
        }
        Some(_) => default,
    }
}

/// Normalize unauthorized-DM behavior to `"pair"` or `"ignore"`.
///
/// Mirrors `_normalize_unauthorized_dm_behavior`.
pub fn normalize_unauthorized_dm_behavior(value: Option<&Value>, default: &str) -> String {
    if let Some(Value::String(s)) = value {
        let normalized = s.trim().to_lowercase();
        if normalized == "pair" || normalized == "ignore" {
            return normalized;
        }
    }
    default.to_string()
}

/// Normalize notice-delivery mode to `"public"` or `"private"`.
///
/// Mirrors `_normalize_notice_delivery`.
pub fn normalize_notice_delivery(value: Option<&Value>, default: &str) -> String {
    if let Some(Value::String(s)) = value {
        let normalized = s.trim().to_lowercase();
        if normalized == "public" || normalized == "private" {
            return normalized;
        }
    }
    default.to_string()
}

// ---------------------------------------------------------------------------
// Platform
// ---------------------------------------------------------------------------

/// Supported messaging platforms.
///
/// Built-in platforms are enum-like variants. Unknown plugin platform names are
/// represented by [`Platform::Plugin`] so that `Platform::parse("irc")` works
/// without modifying this type. The Python `_missing_` machinery restricts
/// pseudo-members to bundled/registered plugins; here `Platform::parse` accepts
/// any non-empty string into `Plugin`, while [`Platform::parse_builtin`] only
/// returns built-ins (matching `Platform(value)` raising `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// All built-in platform variants, in declaration order.
pub const BUILTIN_PLATFORMS: &[Platform] = &[
    Platform::Local,
    Platform::Telegram,
    Platform::Discord,
    Platform::Whatsapp,
    Platform::Slack,
    Platform::Signal,
    Platform::Mattermost,
    Platform::Matrix,
    Platform::Homeassistant,
    Platform::Email,
    Platform::Sms,
    Platform::Dingtalk,
    Platform::ApiServer,
    Platform::Webhook,
    Platform::Feishu,
    Platform::Wecom,
    Platform::WecomCallback,
    Platform::Weixin,
    Platform::Bluebubbles,
    Platform::Qqbot,
    Platform::Yuanbao,
];

impl Platform {
    /// The wire/config string value for this platform (e.g. `"telegram"`).
    pub fn value(&self) -> String {
        match self {
            Platform::Local => "local".into(),
            Platform::Telegram => "telegram".into(),
            Platform::Discord => "discord".into(),
            Platform::Whatsapp => "whatsapp".into(),
            Platform::Slack => "slack".into(),
            Platform::Signal => "signal".into(),
            Platform::Mattermost => "mattermost".into(),
            Platform::Matrix => "matrix".into(),
            Platform::Homeassistant => "homeassistant".into(),
            Platform::Email => "email".into(),
            Platform::Sms => "sms".into(),
            Platform::Dingtalk => "dingtalk".into(),
            Platform::ApiServer => "api_server".into(),
            Platform::Webhook => "webhook".into(),
            Platform::Feishu => "feishu".into(),
            Platform::Wecom => "wecom".into(),
            Platform::WecomCallback => "wecom_callback".into(),
            Platform::Weixin => "weixin".into(),
            Platform::Bluebubbles => "bluebubbles".into(),
            Platform::Qqbot => "qqbot".into(),
            Platform::Yuanbao => "yuanbao".into(),
            Platform::Plugin(v) => v.clone(),
        }
    }

    /// Parse a string into a built-in [`Platform`] only.
    ///
    /// Returns `None` for unknown values, matching `Platform(value)` raising
    /// `ValueError` for non-plugin names. Used by `from_dict` (which skips
    /// unknown platforms).
    pub fn parse_builtin(value: &str) -> Option<Platform> {
        match value {
            "local" => Some(Platform::Local),
            "telegram" => Some(Platform::Telegram),
            "discord" => Some(Platform::Discord),
            "whatsapp" => Some(Platform::Whatsapp),
            "slack" => Some(Platform::Slack),
            "signal" => Some(Platform::Signal),
            "mattermost" => Some(Platform::Mattermost),
            "matrix" => Some(Platform::Matrix),
            "homeassistant" => Some(Platform::Homeassistant),
            "email" => Some(Platform::Email),
            "sms" => Some(Platform::Sms),
            "dingtalk" => Some(Platform::Dingtalk),
            "api_server" => Some(Platform::ApiServer),
            "webhook" => Some(Platform::Webhook),
            "feishu" => Some(Platform::Feishu),
            "wecom" => Some(Platform::Wecom),
            "wecom_callback" => Some(Platform::WecomCallback),
            "weixin" => Some(Platform::Weixin),
            "bluebubbles" => Some(Platform::Bluebubbles),
            "qqbot" => Some(Platform::Qqbot),
            "yuanbao" => Some(Platform::Yuanbao),
            _ => None,
        }
    }

    /// Parse a string into a [`Platform`], creating a [`Platform::Plugin`] for
    /// unknown non-empty names (lowercased+trimmed). Returns `None` for empty.
    pub fn parse(value: &str) -> Option<Platform> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lowered = trimmed.to_lowercase();
        if let Some(p) = Platform::parse_builtin(&lowered) {
            return Some(p);
        }
        Some(Platform::Plugin(lowered))
    }

    /// True when this is one of the explicitly-declared built-in platforms.
    pub fn is_builtin(&self) -> bool {
        !matches!(self, Platform::Plugin(_))
    }
}

// ---------------------------------------------------------------------------
// HomeChannel
// ---------------------------------------------------------------------------

/// Default destination for a platform.
#[derive(Debug, Clone, PartialEq)]
pub struct HomeChannel {
    pub platform: Platform,
    pub chat_id: String,
    pub name: String,
    pub thread_id: Option<String>,
}

impl HomeChannel {
    pub fn new(
        platform: Platform,
        chat_id: impl Into<String>,
        name: impl Into<String>,
        thread_id: Option<String>,
    ) -> Self {
        HomeChannel {
            platform,
            chat_id: chat_id.into(),
            name: name.into(),
            thread_id,
        }
    }

    /// Serialize to a JSON object (matching `HomeChannel.to_dict`).
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("platform".into(), Value::String(self.platform.value()));
        m.insert("chat_id".into(), Value::String(self.chat_id.clone()));
        m.insert("name".into(), Value::String(self.name.clone()));
        if let Some(tid) = &self.thread_id {
            if !tid.is_empty() {
                m.insert("thread_id".into(), Value::String(tid.clone()));
            }
        }
        Value::Object(m)
    }

    /// Deserialize from a JSON object (matching `HomeChannel.from_dict`).
    ///
    /// Returns `None` if the platform value is unknown (the Python version
    /// raises; callers in this module never call this with bad platforms).
    pub fn from_dict(data: &Value) -> Option<HomeChannel> {
        let platform = Platform::parse(value_as_str(data.get("platform")?)?.as_str())?;
        let chat_id = json_to_py_str(data.get("chat_id")?);
        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Home")
            .to_string();
        let thread_id = match data.get("thread_id") {
            Some(v) if is_truthy_json(v) => Some(json_to_py_str(v)),
            _ => None,
        };
        Some(HomeChannel {
            platform,
            chat_id,
            name,
            thread_id,
        })
    }
}

// ---------------------------------------------------------------------------
// SessionResetPolicy
// ---------------------------------------------------------------------------

/// Controls when sessions reset (lose context).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionResetPolicy {
    /// "daily", "idle", "both", or "none".
    pub mode: String,
    /// Hour for daily reset (0-23, local time).
    pub at_hour: i64,
    /// Minutes of inactivity before reset.
    pub idle_minutes: i64,
    /// Send a notification to the user when auto-reset occurs.
    pub notify: bool,
    /// Platforms that don't get reset notifications.
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

impl SessionResetPolicy {
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("mode".into(), Value::String(self.mode.clone()));
        m.insert("at_hour".into(), Value::Number(self.at_hour.into()));
        m.insert("idle_minutes".into(), Value::Number(self.idle_minutes.into()));
        m.insert("notify".into(), Value::Bool(self.notify));
        m.insert(
            "notify_exclude_platforms".into(),
            Value::Array(
                self.notify_exclude_platforms
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        );
        Value::Object(m)
    }

    /// Mirror `SessionResetPolicy.from_dict`, treating YAML null as missing.
    pub fn from_dict(data: &Value) -> SessionResetPolicy {
        let obj = data.as_object();
        let get = |k: &str| obj.and_then(|o| o.get(k)).filter(|v| !v.is_null());

        let mode = get("mode")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "both".into());
        let at_hour = match get("at_hour") {
            Some(v) => coerce_int_or_passthrough(v, 4),
            None => 4,
        };
        let idle_minutes = match get("idle_minutes") {
            Some(v) => coerce_int_or_passthrough(v, 1440),
            None => 1440,
        };
        let notify = coerce_bool(get("notify"), true);
        let notify_exclude_platforms = match get("notify_exclude_platforms") {
            Some(Value::Array(a)) => a.iter().map(json_to_py_str).collect(),
            Some(other) => {
                // Python tuple(non-iterable) would raise; tuple(str) iterates
                // chars. We keep it lenient: a single string becomes one element.
                vec![json_to_py_str(other)]
            }
            None => vec!["api_server".into(), "webhook".into()],
        };

        SessionResetPolicy {
            mode,
            at_hour,
            idle_minutes,
            notify,
            notify_exclude_platforms,
        }
    }
}

// ---------------------------------------------------------------------------
// PlatformConfig
// ---------------------------------------------------------------------------

/// Configuration for a single messaging platform.
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformConfig {
    pub enabled: bool,
    pub token: Option<String>,
    pub api_key: Option<String>,
    pub home_channel: Option<HomeChannel>,
    /// Reply threading mode: "off", "first" (default), or "all".
    pub reply_to_mode: String,
    /// Platform-specific settings.
    pub extra: Map<String, Value>,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        PlatformConfig {
            enabled: false,
            token: None,
            api_key: None,
            home_channel: None,
            reply_to_mode: "first".into(),
            extra: Map::new(),
        }
    }
}

impl PlatformConfig {
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("enabled".into(), Value::Bool(self.enabled));
        m.insert("extra".into(), Value::Object(self.extra.clone()));
        m.insert(
            "reply_to_mode".into(),
            Value::String(self.reply_to_mode.clone()),
        );
        if let Some(t) = &self.token {
            if !t.is_empty() {
                m.insert("token".into(), Value::String(t.clone()));
            }
        }
        if let Some(k) = &self.api_key {
            if !k.is_empty() {
                m.insert("api_key".into(), Value::String(k.clone()));
            }
        }
        if let Some(hc) = &self.home_channel {
            m.insert("home_channel".into(), hc.to_dict());
        }
        Value::Object(m)
    }

    /// Mirror `PlatformConfig.from_dict`.
    pub fn from_dict(data: &Value) -> PlatformConfig {
        let obj = data.as_object();
        let home_channel = obj
            .and_then(|o| o.get("home_channel"))
            .and_then(HomeChannel::from_dict);

        let token = obj
            .and_then(|o| o.get("token"))
            .and_then(opt_string_from_value);
        let api_key = obj
            .and_then(|o| o.get("api_key"))
            .and_then(opt_string_from_value);
        let reply_to_mode = obj
            .and_then(|o| o.get("reply_to_mode"))
            .and_then(|v| v.as_str())
            .unwrap_or("first")
            .to_string();
        let extra = obj
            .and_then(|o| o.get("extra"))
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        PlatformConfig {
            enabled: coerce_bool(obj.and_then(|o| o.get("enabled")), false),
            token,
            api_key,
            home_channel,
            reply_to_mode,
            extra,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration for real-time token streaming to messaging platforms.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingConfig {
    pub enabled: bool,
    /// "edit" (progressive editMessageText) or "off".
    pub transport: String,
    /// Seconds between message edits.
    pub edit_interval: f64,
    /// Chars before forcing an edit.
    pub buffer_threshold: i64,
    /// Cursor shown during streaming.
    pub cursor: String,
    /// When >0, deliver the final edit as a fresh message after this many
    /// seconds (Telegram only). 0 disables.
    pub fresh_final_after_seconds: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        StreamingConfig {
            enabled: false,
            transport: "edit".into(),
            edit_interval: 1.0,
            buffer_threshold: 40,
            cursor: " ▉".into(),
            fresh_final_after_seconds: 60.0,
        }
    }
}

impl StreamingConfig {
    pub fn to_dict(&self) -> Value {
        let mut m = Map::new();
        m.insert("enabled".into(), Value::Bool(self.enabled));
        m.insert("transport".into(), Value::String(self.transport.clone()));
        m.insert("edit_interval".into(), json_number(self.edit_interval));
        m.insert(
            "buffer_threshold".into(),
            Value::Number(self.buffer_threshold.into()),
        );
        m.insert("cursor".into(), Value::String(self.cursor.clone()));
        m.insert(
            "fresh_final_after_seconds".into(),
            json_number(self.fresh_final_after_seconds),
        );
        Value::Object(m)
    }

    /// Mirror `StreamingConfig.from_dict` (empty data -> all defaults).
    pub fn from_dict(data: &Value) -> StreamingConfig {
        let obj = match data.as_object() {
            Some(o) if !o.is_empty() => o,
            _ => return StreamingConfig::default(),
        };
        StreamingConfig {
            enabled: coerce_bool(obj.get("enabled"), false),
            transport: obj
                .get("transport")
                .and_then(|v| v.as_str())
                .unwrap_or("edit")
                .to_string(),
            edit_interval: coerce_float(obj.get("edit_interval"), 1.0),
            buffer_threshold: coerce_int(obj.get("buffer_threshold"), 40),
            cursor: obj
                .get("cursor")
                .and_then(|v| v.as_str())
                .unwrap_or(" ▉")
                .to_string(),
            fresh_final_after_seconds: coerce_float(obj.get("fresh_final_after_seconds"), 60.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in platform connection checkers
// ---------------------------------------------------------------------------

/// Mirror `_PLATFORM_CONNECTED_CHECKERS` plus the generic token branch.
fn platform_specific_connected(platform: &Platform, config: &PlatformConfig) -> Option<bool> {
    let extra = &config.extra;
    let truthy = |k: &str| extra.get(k).map(is_truthy_json).unwrap_or(false);
    let token_truthy = config
        .token
        .as_ref()
        .map(|t| !t.is_empty())
        .unwrap_or(false);

    match platform {
        Platform::Weixin => Some(truthy("account_id") && (token_truthy || truthy("token"))),
        Platform::Whatsapp => Some(true),
        Platform::Signal => Some(truthy("http_url")),
        Platform::Email => Some(truthy("address")),
        Platform::Sms => Some(env_truthy("TWILIO_ACCOUNT_SID")),
        Platform::ApiServer => Some(true),
        Platform::Webhook => Some(true),
        Platform::Feishu => Some(truthy("app_id")),
        Platform::Wecom => Some(truthy("bot_id")),
        Platform::WecomCallback => Some(truthy("corp_id") || truthy("apps")),
        Platform::Bluebubbles => Some(truthy("server_url") && truthy("password")),
        Platform::Qqbot => Some(truthy("app_id") && truthy("client_secret")),
        Platform::Yuanbao => Some(truthy("app_id") && truthy("app_secret")),
        Platform::Dingtalk => Some(
            (truthy("client_id") || env_truthy("DINGTALK_CLIENT_ID"))
                && (truthy("client_secret") || env_truthy("DINGTALK_CLIENT_SECRET")),
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// GatewayConfig
// ---------------------------------------------------------------------------

/// Main gateway configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayConfig {
    pub platforms: BTreeMap<Platform, PlatformConfig>,
    pub default_reset_policy: SessionResetPolicy,
    pub reset_by_type: BTreeMap<String, SessionResetPolicy>,
    pub reset_by_platform: BTreeMap<Platform, SessionResetPolicy>,
    pub reset_triggers: Vec<String>,
    pub quick_commands: Map<String, Value>,
    pub sessions_dir: PathBuf,
    pub always_log_local: bool,
    pub stt_enabled: bool,
    pub group_sessions_per_user: bool,
    pub thread_sessions_per_user: bool,
    pub unauthorized_dm_behavior: String,
    pub streaming: StreamingConfig,
    pub session_store_max_age_days: i64,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        GatewayConfig {
            platforms: BTreeMap::new(),
            default_reset_policy: SessionResetPolicy::default(),
            reset_by_type: BTreeMap::new(),
            reset_by_platform: BTreeMap::new(),
            reset_triggers: vec!["/new".into(), "/reset".into()],
            quick_commands: Map::new(),
            sessions_dir: get_hermes_home().join("sessions"),
            always_log_local: true,
            stt_enabled: true,
            group_sessions_per_user: true,
            thread_sessions_per_user: false,
            unauthorized_dm_behavior: "pair".into(),
            streaming: StreamingConfig::default(),
            session_store_max_age_days: 90,
        }
    }
}

impl GatewayConfig {
    /// Return list of platforms that are enabled and configured.
    pub fn get_connected_platforms(&self) -> Vec<Platform> {
        let mut connected = Vec::new();
        for (platform, config) in &self.platforms {
            if !config.enabled {
                continue;
            }
            if self.is_platform_connected(platform, config) {
                connected.push(platform.clone());
            }
        }
        connected
    }

    /// Check whether a single platform is sufficiently configured.
    ///
    /// Note: the Python version consults the plugin `platform_registry` for
    /// unknown platforms as a final fallback. That registry is not modelled
    /// here; for plugin platforms we conservatively return `false` (matching
    /// the Python branch when the registry has no usable entry/raises).
    pub fn is_platform_connected(&self, platform: &Platform, config: &PlatformConfig) -> bool {
        // Weixin requires account_id + a token (checked first).
        if *platform == Platform::Weixin {
            let account_id = config
                .extra
                .get("account_id")
                .map(is_truthy_json)
                .unwrap_or(false);
            let token_truthy = config.token.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
            let extra_token = config.extra.get("token").map(is_truthy_json).unwrap_or(false);
            return account_id && (token_truthy || extra_token);
        }

        // Generic token/api_key auth.
        if config.token.as_ref().map(|t| !t.is_empty()).unwrap_or(false)
            || config
                .api_key
                .as_ref()
                .map(|k| !k.is_empty())
                .unwrap_or(false)
        {
            return true;
        }

        // Platform-specific check.
        if let Some(result) = platform_specific_connected(platform, config) {
            return result;
        }

        // Plugin-registered platforms: registry not modelled -> not connected.
        false
    }

    /// Get the home channel for a platform.
    pub fn get_home_channel(&self, platform: &Platform) -> Option<&HomeChannel> {
        self.platforms
            .get(platform)
            .and_then(|c| c.home_channel.as_ref())
    }

    /// Get the appropriate reset policy for a session.
    ///
    /// Priority: platform override > type override > default.
    pub fn get_reset_policy(
        &self,
        platform: Option<&Platform>,
        session_type: Option<&str>,
    ) -> SessionResetPolicy {
        if let Some(p) = platform {
            if let Some(policy) = self.reset_by_platform.get(p) {
                return policy.clone();
            }
        }
        if let Some(st) = session_type {
            if let Some(policy) = self.reset_by_type.get(st) {
                return policy.clone();
            }
        }
        self.default_reset_policy.clone()
    }

    pub fn to_dict(&self) -> Value {
        let mut platforms = Map::new();
        for (p, c) in &self.platforms {
            platforms.insert(p.value(), c.to_dict());
        }
        let mut reset_by_type = Map::new();
        for (k, v) in &self.reset_by_type {
            reset_by_type.insert(k.clone(), v.to_dict());
        }
        let mut reset_by_platform = Map::new();
        for (p, v) in &self.reset_by_platform {
            reset_by_platform.insert(p.value(), v.to_dict());
        }

        let mut m = Map::new();
        m.insert("platforms".into(), Value::Object(platforms));
        m.insert(
            "default_reset_policy".into(),
            self.default_reset_policy.to_dict(),
        );
        m.insert("reset_by_type".into(), Value::Object(reset_by_type));
        m.insert("reset_by_platform".into(), Value::Object(reset_by_platform));
        m.insert(
            "reset_triggers".into(),
            Value::Array(
                self.reset_triggers
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        );
        m.insert(
            "quick_commands".into(),
            Value::Object(self.quick_commands.clone()),
        );
        m.insert(
            "sessions_dir".into(),
            Value::String(self.sessions_dir.to_string_lossy().to_string()),
        );
        m.insert("always_log_local".into(), Value::Bool(self.always_log_local));
        m.insert("stt_enabled".into(), Value::Bool(self.stt_enabled));
        m.insert(
            "group_sessions_per_user".into(),
            Value::Bool(self.group_sessions_per_user),
        );
        m.insert(
            "thread_sessions_per_user".into(),
            Value::Bool(self.thread_sessions_per_user),
        );
        m.insert(
            "unauthorized_dm_behavior".into(),
            Value::String(self.unauthorized_dm_behavior.clone()),
        );
        m.insert("streaming".into(), self.streaming.to_dict());
        m.insert(
            "session_store_max_age_days".into(),
            Value::Number(self.session_store_max_age_days.into()),
        );
        Value::Object(m)
    }

    /// Mirror `GatewayConfig.from_dict`.
    pub fn from_dict(data: &Value) -> GatewayConfig {
        let obj = data.as_object().cloned().unwrap_or_default();

        let mut platforms = BTreeMap::new();
        if let Some(Value::Object(plats)) = obj.get("platforms") {
            for (name, pdata) in plats {
                if let Some(platform) = Platform::parse_builtin(name) {
                    platforms.insert(platform, PlatformConfig::from_dict(pdata));
                }
                // Unknown platforms are skipped (matches `except ValueError`).
            }
        }

        let mut reset_by_type = BTreeMap::new();
        if let Some(Value::Object(rbt)) = obj.get("reset_by_type") {
            for (type_name, policy_data) in rbt {
                reset_by_type
                    .insert(type_name.clone(), SessionResetPolicy::from_dict(policy_data));
            }
        }

        let mut reset_by_platform = BTreeMap::new();
        if let Some(Value::Object(rbp)) = obj.get("reset_by_platform") {
            for (name, policy_data) in rbp {
                if let Some(platform) = Platform::parse_builtin(name) {
                    reset_by_platform.insert(platform, SessionResetPolicy::from_dict(policy_data));
                }
            }
        }

        let default_policy = match obj.get("default_reset_policy") {
            Some(v) => SessionResetPolicy::from_dict(v),
            None => SessionResetPolicy::default(),
        };

        let sessions_dir = match obj.get("sessions_dir") {
            Some(Value::String(s)) => PathBuf::from(s),
            Some(other) if !other.is_null() => PathBuf::from(json_to_py_str(other)),
            _ => get_hermes_home().join("sessions"),
        };

        let quick_commands = match obj.get("quick_commands") {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };

        // stt_enabled: prefer top-level, fall back to stt.enabled.
        let stt_enabled_val: Option<Value> = match obj.get("stt_enabled") {
            Some(v) if !v.is_null() => Some(v.clone()),
            _ => obj
                .get("stt")
                .and_then(|v| v.as_object())
                .and_then(|o| o.get("enabled"))
                .cloned(),
        };

        let group_sessions_per_user = obj.get("group_sessions_per_user");
        let thread_sessions_per_user = obj.get("thread_sessions_per_user");
        let unauthorized_dm_behavior =
            normalize_unauthorized_dm_behavior(obj.get("unauthorized_dm_behavior"), "pair");

        let session_store_max_age_days = match obj.get("session_store_max_age_days") {
            Some(v) => {
                let parsed = coerce_int_strict(v);
                match parsed {
                    Some(n) if n < 0 => 0,
                    Some(n) => n,
                    None => 90,
                }
            }
            None => 90,
        };

        let reset_triggers = match obj.get("reset_triggers") {
            Some(Value::Array(a)) => a.iter().map(json_to_py_str).collect(),
            Some(v) if !v.is_null() => vec![json_to_py_str(v)],
            _ => vec!["/new".into(), "/reset".into()],
        };

        GatewayConfig {
            platforms,
            default_reset_policy: default_policy,
            reset_by_type,
            reset_by_platform,
            reset_triggers,
            quick_commands,
            sessions_dir,
            always_log_local: coerce_bool(obj.get("always_log_local"), true),
            stt_enabled: coerce_bool(stt_enabled_val.as_ref(), true),
            group_sessions_per_user: coerce_bool(group_sessions_per_user, true),
            thread_sessions_per_user: coerce_bool(thread_sessions_per_user, false),
            unauthorized_dm_behavior,
            streaming: StreamingConfig::from_dict(obj.get("streaming").unwrap_or(&Value::Null)),
            session_store_max_age_days,
        }
    }

    /// Return the effective unauthorized-DM behavior for a platform.
    pub fn get_unauthorized_dm_behavior(&self, platform: Option<&Platform>) -> String {
        if let Some(p) = platform {
            if let Some(cfg) = self.platforms.get(p) {
                if cfg.extra.contains_key("unauthorized_dm_behavior") {
                    return normalize_unauthorized_dm_behavior(
                        cfg.extra.get("unauthorized_dm_behavior"),
                        &self.unauthorized_dm_behavior,
                    );
                }
            }
        }
        self.unauthorized_dm_behavior.clone()
    }

    /// Return the effective notice-delivery mode for a platform.
    pub fn get_notice_delivery(&self, platform: Option<&Platform>) -> String {
        if let Some(p) = platform {
            if let Some(cfg) = self.platforms.get(p) {
                if cfg.extra.contains_key("notice_delivery") {
                    return normalize_notice_delivery(cfg.extra.get("notice_delivery"), "public");
                }
            }
        }
        "public".to_string()
    }
}

// ---------------------------------------------------------------------------
// load_gateway_config
// ---------------------------------------------------------------------------

/// Load gateway configuration from multiple sources.
///
/// Priority (highest to lowest):
/// 1. Environment variables
/// 2. `~/.hermes/config.yaml`
/// 3. `~/.hermes/gateway.json` (legacy)
/// 4. Built-in defaults
///
/// Mirrors `load_gateway_config`. Side effects (setting `os.environ` keys from
/// YAML) are reproduced; the plugin-registry enable pass at the end is a no-op
/// (registry not modelled in this port).
pub fn load_gateway_config() -> GatewayConfig {
    let home = get_hermes_home();
    let mut gw_data = Map::new();

    // Legacy fallback: gateway.json provides the base layer.
    let gateway_json_path = home.join("gateway.json");
    if gateway_json_path.exists() {
        match std::fs::read_to_string(&gateway_json_path) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(o)) => {
                    gw_data = o;
                    log::info!(
                        "Loaded legacy {} — consider moving settings to config.yaml",
                        gateway_json_path.display()
                    );
                }
                Ok(_) | Err(_) => {
                    // `json.load(...) or {}` -> non-object becomes empty.
                }
            },
            Err(e) => {
                log::warn!("Failed to load {}: {}", gateway_json_path.display(), e);
            }
        }
    }

    // Primary source: config.yaml
    let config_yaml_override = env::var(OVERRIDE_CONFIG_PATH_ENV)
        .unwrap_or_default()
        .trim()
        .to_string();
    let config_yaml_path = if !config_yaml_override.is_empty() {
        PathBuf::from(config_yaml_override)
    } else {
        home.join("config.yaml")
    };

    if config_yaml_path.exists() {
        match load_yaml_into_gw_data(&config_yaml_path, &mut gw_data) {
            Ok(()) => {}
            Err(e) => {
                log::warn!(
                    "Failed to process config.yaml — falling back to .env / gateway.json values. \
                     Check {} for syntax errors. Error: {}",
                    home.join("config.yaml").display(),
                    e
                );
            }
        }
    }

    let mut config = GatewayConfig::from_dict(&Value::Object(gw_data));

    apply_env_overrides(&mut config);

    validate_gateway_config(&mut config);

    config
}

/// Parse `config.yaml` and merge its bridged keys into `gw_data` (mutating
/// process env vars as a side effect, matching the Python implementation).
fn load_yaml_into_gw_data(
    path: &std::path::Path,
    gw_data: &mut Map<String, Value>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let yaml_cfg: Value = match serde_yaml::from_str::<serde_yaml::Value>(&text) {
        Ok(v) => yaml_to_json(v),
        Err(e) => return Err(e.to_string()),
    };
    let yaml_cfg = match yaml_cfg {
        Value::Object(o) => o,
        _ => Map::new(), // `yaml.safe_load(...) or {}`
    };

    // session_reset -> default_reset_policy
    if let Some(Value::Object(sr)) = yaml_cfg.get("session_reset") {
        gw_data.insert("default_reset_policy".into(), Value::Object(sr.clone()));
    }

    // quick_commands
    if let Some(qc) = yaml_cfg.get("quick_commands") {
        if let Value::Object(_) = qc {
            gw_data.insert("quick_commands".into(), qc.clone());
        } else {
            log::warn!(
                "Ignoring invalid quick_commands in config.yaml (expected mapping, got {})",
                json_type_name(qc)
            );
        }
    }

    if let Some(stt) = yaml_cfg.get("stt") {
        if stt.is_object() {
            gw_data.insert("stt".into(), stt.clone());
        }
    }

    if let Some(v) = yaml_cfg.get("group_sessions_per_user") {
        gw_data.insert("group_sessions_per_user".into(), v.clone());
    }
    if let Some(v) = yaml_cfg.get("thread_sessions_per_user") {
        gw_data.insert("thread_sessions_per_user".into(), v.clone());
    }

    if let Some(streaming) = yaml_cfg.get("streaming") {
        if streaming.is_object() {
            gw_data.insert("streaming".into(), streaming.clone());
        }
    }

    if let Some(v) = yaml_cfg.get("reset_triggers") {
        gw_data.insert("reset_triggers".into(), v.clone());
    }
    if let Some(v) = yaml_cfg.get("always_log_local") {
        gw_data.insert("always_log_local".into(), v.clone());
    }

    if yaml_cfg.contains_key("unauthorized_dm_behavior") {
        let norm = normalize_unauthorized_dm_behavior(yaml_cfg.get("unauthorized_dm_behavior"), "pair");
        gw_data.insert("unauthorized_dm_behavior".into(), Value::String(norm));
    }

    // Merge platforms section.
    let yaml_platforms = yaml_cfg.get("platforms").cloned();
    // Ensure gw_data["platforms"] is an object.
    if !matches!(gw_data.get("platforms"), Some(Value::Object(_))) {
        gw_data.insert("platforms".into(), Value::Object(Map::new()));
    }

    if let Some(Value::Object(yp)) = yaml_platforms {
        let mut platforms_data = match gw_data.get("platforms") {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };
        for (plat_name, plat_block) in yp {
            let block = match plat_block {
                Value::Object(o) => o,
                _ => continue,
            };
            let existing = match platforms_data.get(&plat_name) {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            // Deep-merge extra dicts.
            let mut merged_extra = match existing.get("extra") {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            if let Some(Value::Object(bx)) = block.get("extra") {
                for (k, v) in bx {
                    merged_extra.insert(k.clone(), v.clone());
                }
            }
            if plat_name == Platform::Slack.value() && block.contains_key("enabled") {
                merged_extra.insert("_enabled_explicit".into(), Value::Bool(true));
            }
            // merged = {**existing, **plat_block}
            let mut merged = existing.clone();
            for (k, v) in &block {
                merged.insert(k.clone(), v.clone());
            }
            if !merged_extra.is_empty() {
                merged.insert("extra".into(), Value::Object(merged_extra));
            }
            platforms_data.insert(plat_name.clone(), Value::Object(merged));
        }
        gw_data.insert("platforms".into(), Value::Object(platforms_data));
    }

    // Per-platform top-level sections -> bridged extra keys.
    {
        let mut platforms_data = match gw_data.get("platforms") {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };
        for plat in BUILTIN_PLATFORMS {
            if *plat == Platform::Local {
                continue;
            }
            let platform_cfg = match yaml_cfg.get(&plat.value()) {
                Some(Value::Object(o)) => o,
                _ => continue,
            };
            let mut bridged = Map::new();
            if platform_cfg.contains_key("unauthorized_dm_behavior") {
                let default = gw_data
                    .get("unauthorized_dm_behavior")
                    .and_then(|v| v.as_str())
                    .unwrap_or("pair");
                bridged.insert(
                    "unauthorized_dm_behavior".into(),
                    Value::String(normalize_unauthorized_dm_behavior(
                        platform_cfg.get("unauthorized_dm_behavior"),
                        default,
                    )),
                );
            }
            if platform_cfg.contains_key("notice_delivery") {
                bridged.insert(
                    "notice_delivery".into(),
                    Value::String(normalize_notice_delivery(
                        platform_cfg.get("notice_delivery"),
                        "public",
                    )),
                );
            }
            for key in [
                "reply_prefix",
                "reply_in_thread",
                "require_mention",
                "free_response_channels",
                "mention_patterns",
                "dm_policy",
                "allow_from",
                "group_policy",
                "group_allow_from",
            ] {
                if let Some(v) = platform_cfg.get(key) {
                    bridged.insert(key.into(), v.clone());
                }
            }
            if (*plat == Platform::Discord || *plat == Platform::Slack)
                && platform_cfg.contains_key("channel_skill_bindings")
            {
                bridged.insert(
                    "channel_skill_bindings".into(),
                    platform_cfg.get("channel_skill_bindings").unwrap().clone(),
                );
            }
            if let Some(channel_prompts) = platform_cfg.get("channel_prompts") {
                if let Value::Object(cp) = channel_prompts {
                    // {str(k): v} — keys are already strings in JSON.
                    bridged.insert("channel_prompts".into(), Value::Object(cp.clone()));
                } else {
                    bridged.insert("channel_prompts".into(), channel_prompts.clone());
                }
            }
            let enabled_was_explicit = platform_cfg.contains_key("enabled");
            if bridged.is_empty() && !enabled_was_explicit {
                continue;
            }
            // plat_data = platforms_data.setdefault(plat.value, {})
            let mut plat_data = match platforms_data.get(&plat.value()) {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            if enabled_was_explicit {
                plat_data.insert("enabled".into(), platform_cfg.get("enabled").unwrap().clone());
            }
            let mut extra = match plat_data.get("extra") {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            if *plat == Platform::Slack && enabled_was_explicit {
                extra.insert("_enabled_explicit".into(), Value::Bool(true));
            }
            for (k, v) in &bridged {
                extra.insert(k.clone(), v.clone());
            }
            plat_data.insert("extra".into(), Value::Object(extra));
            platforms_data.insert(plat.value(), Value::Object(plat_data));
        }
        gw_data.insert("platforms".into(), Value::Object(platforms_data));
    }

    // Slack settings -> env vars (env vars take precedence).
    if let Some(Value::Object(slack)) = yaml_cfg.get("slack") {
        set_env_if_unset_lower(slack, "require_mention", "SLACK_REQUIRE_MENTION");
        set_env_if_unset_lower(slack, "strict_mention", "SLACK_STRICT_MENTION");
        set_env_if_unset_lower(slack, "allow_bots", "SLACK_ALLOW_BOTS");
        set_env_csv_if_unset(slack, "free_response_channels", "SLACK_FREE_RESPONSE_CHANNELS");
        set_env_if_unset_lower(slack, "reactions", "SLACK_REACTIONS");
    }

    // Discord settings -> env vars.
    if let Some(Value::Object(discord)) = yaml_cfg.get("discord") {
        set_env_if_unset_lower(discord, "require_mention", "DISCORD_REQUIRE_MENTION");
        set_env_csv_if_unset(discord, "free_response_channels", "DISCORD_FREE_RESPONSE_CHANNELS");
        set_env_if_unset_lower(discord, "auto_thread", "DISCORD_AUTO_THREAD");
        set_env_if_unset_lower(discord, "reactions", "DISCORD_REACTIONS");
        set_env_csv_if_unset(discord, "ignored_channels", "DISCORD_IGNORED_CHANNELS");
        set_env_csv_if_unset(discord, "allowed_channels", "DISCORD_ALLOWED_CHANNELS");
        set_env_csv_if_unset(discord, "no_thread_channels", "DISCORD_NO_THREAD_CHANNELS");
        if let Some(Value::Object(am)) = discord.get("allow_mentions") {
            for (yaml_key, env_key) in [
                ("everyone", "DISCORD_ALLOW_MENTION_EVERYONE"),
                ("roles", "DISCORD_ALLOW_MENTION_ROLES"),
                ("users", "DISCORD_ALLOW_MENTION_USERS"),
                ("replied_user", "DISCORD_ALLOW_MENTION_REPLIED_USER"),
            ] {
                set_env_if_unset_lower(am, yaml_key, env_key);
            }
        }
        set_reply_to_mode_env(discord, "DISCORD_REPLY_TO_MODE");
    }

    // Bridge top-level require_mention to Telegram.
    if let Some(tl_rm) = yaml_cfg.get("require_mention") {
        if !tl_rm.is_null() {
            let tg_section = yaml_cfg.get("telegram").and_then(|v| v.as_object());
            let tg_has_rm = tg_section
                .map(|o| o.contains_key("require_mention"))
                .unwrap_or(false);
            if !tg_has_rm {
                let mut platforms_data = match gw_data.get("platforms") {
                    Some(Value::Object(o)) => o.clone(),
                    _ => Map::new(),
                };
                let mut tg_plat = match platforms_data.get(&Platform::Telegram.value()) {
                    Some(Value::Object(o)) => o.clone(),
                    _ => Map::new(),
                };
                let mut tg_extra = match tg_plat.get("extra") {
                    Some(Value::Object(o)) => o.clone(),
                    _ => Map::new(),
                };
                tg_extra.entry("require_mention").or_insert(tl_rm.clone());
                tg_plat.insert("extra".into(), Value::Object(tg_extra));
                platforms_data.insert(Platform::Telegram.value(), Value::Object(tg_plat));
                gw_data.insert("platforms".into(), Value::Object(platforms_data));
            }
        }
    }

    // Telegram settings -> env vars.
    if let Some(Value::Object(telegram)) = yaml_cfg.get("telegram") {
        let effective_rm = telegram
            .get("require_mention")
            .or_else(|| yaml_cfg.get("require_mention"));
        if let Some(rm) = effective_rm {
            if !rm.is_null() && env::var("TELEGRAM_REQUIRE_MENTION").is_err() {
                set_env(("TELEGRAM_REQUIRE_MENTION", py_str_lower(rm)));
            }
        }
        if telegram.contains_key("mention_patterns") && env::var("TELEGRAM_MENTION_PATTERNS").is_err()
        {
            let json = serde_json::to_string(telegram.get("mention_patterns").unwrap())
                .unwrap_or_else(|_| "null".into());
            set_env(("TELEGRAM_MENTION_PATTERNS", json));
        }
        set_env_csv_if_unset(telegram, "free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS");
        set_env_csv_if_unset(telegram, "ignored_threads", "TELEGRAM_IGNORED_THREADS");
        set_env_if_unset_lower(telegram, "reactions", "TELEGRAM_REACTIONS");
        if telegram.contains_key("proxy_url") && env::var("TELEGRAM_PROXY").is_err() {
            let v = py_str(telegram.get("proxy_url").unwrap());
            set_env(("TELEGRAM_PROXY", v.trim().to_string()));
        }
        set_reply_to_mode_env(telegram, "TELEGRAM_REPLY_TO_MODE");
        set_env_csv_if_unset(telegram, "allow_from", "TELEGRAM_ALLOWED_USERS");
        set_env_csv_if_unset(telegram, "group_allow_from", "TELEGRAM_GROUP_ALLOWED_USERS");
        set_env_csv_if_unset(telegram, "group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS");
        if let Some(dlp) = telegram.get("disable_link_previews") {
            let mut platforms_data = match gw_data.get("platforms") {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            let mut plat_data = match platforms_data.get(&Platform::Telegram.value()) {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            let mut extra = match plat_data.get("extra") {
                Some(Value::Object(o)) => o.clone(),
                _ => Map::new(),
            };
            extra.insert("disable_link_previews".into(), dlp.clone());
            plat_data.insert("extra".into(), Value::Object(extra));
            platforms_data.insert(Platform::Telegram.value(), Value::Object(plat_data));
            gw_data.insert("platforms".into(), Value::Object(platforms_data));
        }
    }

    // WhatsApp settings -> env vars.
    if let Some(Value::Object(wa)) = yaml_cfg.get("whatsapp") {
        set_env_if_unset_lower(wa, "require_mention", "WHATSAPP_REQUIRE_MENTION");
        if wa.contains_key("mention_patterns") && env::var("WHATSAPP_MENTION_PATTERNS").is_err() {
            let json = serde_json::to_string(wa.get("mention_patterns").unwrap())
                .unwrap_or_else(|_| "null".into());
            set_env(("WHATSAPP_MENTION_PATTERNS", json));
        }
        set_env_csv_if_unset(wa, "free_response_chats", "WHATSAPP_FREE_RESPONSE_CHATS");
        set_env_if_unset_lower(wa, "dm_policy", "WHATSAPP_DM_POLICY");
        set_env_csv_if_unset(wa, "allow_from", "WHATSAPP_ALLOWED_USERS");
        set_env_if_unset_lower(wa, "group_policy", "WHATSAPP_GROUP_POLICY");
        set_env_csv_if_unset(wa, "group_allow_from", "WHATSAPP_GROUP_ALLOWED_USERS");
    }

    // DingTalk settings -> env vars.
    if let Some(Value::Object(dt)) = yaml_cfg.get("dingtalk") {
        set_env_if_unset_lower(dt, "require_mention", "DINGTALK_REQUIRE_MENTION");
        if dt.contains_key("mention_patterns") && env::var("DINGTALK_MENTION_PATTERNS").is_err() {
            let json = serde_json::to_string(dt.get("mention_patterns").unwrap())
                .unwrap_or_else(|_| "null".into());
            set_env(("DINGTALK_MENTION_PATTERNS", json));
        }
        set_env_csv_if_unset(dt, "free_response_chats", "DINGTALK_FREE_RESPONSE_CHATS");
        set_env_csv_if_unset(dt, "allowed_users", "DINGTALK_ALLOWED_USERS");
    }

    // Matrix settings -> env vars.
    if let Some(Value::Object(mx)) = yaml_cfg.get("matrix") {
        set_env_if_unset_lower(mx, "require_mention", "MATRIX_REQUIRE_MENTION");
        set_env_csv_if_unset(mx, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS");
        set_env_if_unset_lower(mx, "auto_thread", "MATRIX_AUTO_THREAD");
        set_env_if_unset_lower(mx, "dm_mention_threads", "MATRIX_DM_MENTION_THREADS");
    }

    // Feishu settings -> env vars.
    if let Some(Value::Object(fs)) = yaml_cfg.get("feishu") {
        set_env_if_unset_lower(fs, "allow_bots", "FEISHU_ALLOW_BOTS");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// validate_gateway_config
// ---------------------------------------------------------------------------

/// Validate and sanitize a loaded [`GatewayConfig`] in place.
///
/// Mirrors `_validate_gateway_config`.
pub fn validate_gateway_config(config: &mut GatewayConfig) {
    {
        let policy = &mut config.default_reset_policy;
        if !(0..=23).contains(&policy.at_hour) {
            log::warn!(
                "Invalid at_hour={} (must be 0-23). Using default 4.",
                policy.at_hour
            );
            policy.at_hour = 4;
        }
        if policy.idle_minutes <= 0 {
            log::warn!(
                "Invalid idle_minutes={} (must be positive). Using default 1440.",
                policy.idle_minutes
            );
            policy.idle_minutes = 1440;
        }
    }

    let token_env_names = token_env_names();

    // Warn about empty bot tokens.
    for (platform, pconfig) in &config.platforms {
        if !pconfig.enabled {
            continue;
        }
        if let Some(env_name) = token_env_names
            .iter()
            .find(|(p, _)| p == platform)
            .map(|(_, n)| *n)
        {
            if let Some(token) = &pconfig.token {
                if token.trim().is_empty() {
                    log::warn!(
                        "{} is enabled but {} is empty. The adapter will likely fail to connect.",
                        platform.value(),
                        env_name
                    );
                }
            }
        }
    }

    // Reject known-weak placeholder tokens.
    let mut to_disable: Vec<Platform> = Vec::new();
    for (platform, pconfig) in &config.platforms {
        if !pconfig.enabled {
            continue;
        }
        let env_name = match token_env_names
            .iter()
            .find(|(p, _)| p == platform)
            .map(|(_, n)| *n)
        {
            Some(n) => n,
            None => continue,
        };
        if let Some(token) = &pconfig.token {
            let t = token.trim();
            if !t.is_empty() && !has_usable_secret_min(token, 4) {
                let preview: String = t.chars().take(6).collect();
                log::error!(
                    "{} is enabled but {} is set to a placeholder value ('{}...'). \
                     Set a real bot token before starting the gateway. \
                     The adapter will NOT be started.",
                    platform.value(),
                    env_name,
                    preview
                );
                to_disable.push(platform.clone());
            }
        }
    }
    for p in to_disable {
        if let Some(cfg) = config.platforms.get_mut(&p) {
            cfg.enabled = false;
        }
    }
}

/// `has_usable_secret(value, min_length=4)` — mirror of
/// `hermes_cli.auth.has_usable_secret`: a secret is usable when it is a
/// non-empty, non-placeholder string of at least `min_length` chars.
///
/// Inlined here (rather than calling `crate::cli_runtime_provider`) so this
/// module compiles independently of the auth module's registration state. When
/// `cli_runtime_provider` is wired into the crate this can delegate instead.
fn has_usable_secret_min(value: &str, min_length: usize) -> bool {
    let trimmed = value.trim();
    if trimmed.chars().count() < min_length {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !matches!(
        lowered.as_str(),
        "no-key-required"
            | "none"
            | "null"
            | "n/a"
            | "na"
            | "placeholder"
            | "your-api-key"
            | "your_api_key"
            | "changeme"
            | "change-me"
            | "xxx"
    )
}

fn token_env_names() -> Vec<(Platform, &'static str)> {
    vec![
        (Platform::Telegram, "TELEGRAM_BOT_TOKEN"),
        (Platform::Discord, "DISCORD_BOT_TOKEN"),
        (Platform::Slack, "SLACK_BOT_TOKEN"),
        (Platform::Mattermost, "MATTERMOST_TOKEN"),
        (Platform::Matrix, "MATRIX_ACCESS_TOKEN"),
        (Platform::Weixin, "WEIXIN_TOKEN"),
    ]
}

// ---------------------------------------------------------------------------
// apply_env_overrides
// ---------------------------------------------------------------------------

/// Apply environment variable overrides to `config`.
///
/// Mirrors `_apply_env_overrides`. The final plugin-registry enable pass in the
/// Python version is omitted (registry not modelled here).
pub fn apply_env_overrides(config: &mut GatewayConfig) {
    // Telegram
    if let Some(token) = getenv_nonempty("TELEGRAM_BOT_TOKEN") {
        let p = config.platforms.entry(Platform::Telegram).or_default();
        p.enabled = true;
        p.token = Some(token);
    }
    let telegram_reply_mode = getenv_lower("TELEGRAM_REPLY_TO_MODE");
    if matches!(telegram_reply_mode.as_str(), "off" | "first" | "all") {
        let p = config.platforms.entry(Platform::Telegram).or_default();
        p.reply_to_mode = telegram_reply_mode;
    }
    if let Some(ips) = getenv_nonempty("TELEGRAM_FALLBACK_IPS") {
        let p = config.platforms.entry(Platform::Telegram).or_default();
        let list: Vec<Value> = ips
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .collect();
        p.extra.insert("fallback_ips".into(), Value::Array(list));
    }
    if let Some(home) = getenv_nonempty("TELEGRAM_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Telegram) {
            set_home(config, Platform::Telegram, home, "TELEGRAM_HOME_CHANNEL_NAME", "Home", "TELEGRAM_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Discord
    if let Some(token) = getenv_nonempty("DISCORD_BOT_TOKEN") {
        let p = config.platforms.entry(Platform::Discord).or_default();
        p.enabled = true;
        p.token = Some(token);
    }
    if let Some(home) = getenv_nonempty("DISCORD_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Discord) {
            set_home(config, Platform::Discord, home, "DISCORD_HOME_CHANNEL_NAME", "Home", "DISCORD_HOME_CHANNEL_THREAD_ID");
        }
    }
    let discord_reply_mode = getenv_lower("DISCORD_REPLY_TO_MODE");
    if matches!(discord_reply_mode.as_str(), "off" | "first" | "all") {
        let p = config.platforms.entry(Platform::Discord).or_default();
        p.reply_to_mode = discord_reply_mode;
    }

    // WhatsApp
    if getenv_in(
        "WHATSAPP_ENABLED",
        &["true", "1", "yes"],
    ) {
        let p = config.platforms.entry(Platform::Whatsapp).or_default();
        p.enabled = true;
    }
    if let Some(home) = getenv_nonempty("WHATSAPP_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Whatsapp) {
            set_home(config, Platform::Whatsapp, home, "WHATSAPP_HOME_CHANNEL_NAME", "Home", "WHATSAPP_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Slack
    if let Some(token) = getenv_nonempty("SLACK_BOT_TOKEN") {
        if !config.platforms.contains_key(&Platform::Slack) {
            let p = config.platforms.entry(Platform::Slack).or_default();
            p.enabled = true;
        } else {
            let p = config.platforms.get_mut(&Platform::Slack).unwrap();
            let enabled_was_explicit = matches!(
                p.extra.remove("_enabled_explicit"),
                Some(v) if is_truthy_json(&v)
            );
            if !p.enabled && !enabled_was_explicit {
                p.enabled = true;
            }
        }
        config.platforms.get_mut(&Platform::Slack).unwrap().token = Some(token);
    }
    if let Some(home) = getenv_nonempty("SLACK_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Slack) {
            set_home(config, Platform::Slack, home, "SLACK_HOME_CHANNEL_NAME", "", "SLACK_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Signal
    let signal_url = getenv_nonempty("SIGNAL_HTTP_URL");
    let signal_account = getenv_nonempty("SIGNAL_ACCOUNT");
    if let (Some(url), Some(account)) = (signal_url, signal_account) {
        let p = config.platforms.entry(Platform::Signal).or_default();
        p.enabled = true;
        p.extra.insert("http_url".into(), Value::String(url));
        p.extra.insert("account".into(), Value::String(account));
        p.extra.insert(
            "ignore_stories".into(),
            Value::Bool(getenv_in_default(
                "SIGNAL_IGNORE_STORIES",
                "true",
                &["true", "1", "yes"],
            )),
        );
    }
    if let Some(home) = getenv_nonempty("SIGNAL_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Signal) {
            set_home(config, Platform::Signal, home, "SIGNAL_HOME_CHANNEL_NAME", "Home", "SIGNAL_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Mattermost
    if let Some(token) = getenv_nonempty("MATTERMOST_TOKEN") {
        let url = env::var("MATTERMOST_URL").unwrap_or_default();
        if url.is_empty() {
            log::warn!("MATTERMOST_TOKEN set but MATTERMOST_URL is missing");
        }
        let p = config.platforms.entry(Platform::Mattermost).or_default();
        p.enabled = true;
        p.token = Some(token);
        p.extra.insert("url".into(), Value::String(url));
    }
    if let Some(home) = getenv_nonempty("MATTERMOST_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Mattermost) {
            set_home(config, Platform::Mattermost, home, "MATTERMOST_HOME_CHANNEL_NAME", "Home", "MATTERMOST_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Matrix
    let matrix_token = getenv_nonempty("MATRIX_ACCESS_TOKEN");
    let matrix_homeserver = env::var("MATRIX_HOMESERVER").unwrap_or_default();
    let matrix_password_present = getenv_nonempty("MATRIX_PASSWORD").is_some();
    if matrix_token.is_some() || matrix_password_present {
        if matrix_homeserver.is_empty() {
            log::warn!(
                "MATRIX_ACCESS_TOKEN/MATRIX_PASSWORD set but MATRIX_HOMESERVER is missing"
            );
        }
        let p = config.platforms.entry(Platform::Matrix).or_default();
        p.enabled = true;
        if let Some(t) = matrix_token {
            p.token = Some(t);
        }
        p.extra
            .insert("homeserver".into(), Value::String(matrix_homeserver));
        let matrix_user = env::var("MATRIX_USER_ID").unwrap_or_default();
        if !matrix_user.is_empty() {
            p.extra.insert("user_id".into(), Value::String(matrix_user));
        }
        let matrix_password = env::var("MATRIX_PASSWORD").unwrap_or_default();
        if !matrix_password.is_empty() {
            p.extra
                .insert("password".into(), Value::String(matrix_password));
        }
        let e2ee = env_in_default("MATRIX_ENCRYPTION", "", &["true", "1", "yes"]);
        p.extra.insert("encryption".into(), Value::Bool(e2ee));
        let device_id = env::var("MATRIX_DEVICE_ID").unwrap_or_default();
        if !device_id.is_empty() {
            p.extra.insert("device_id".into(), Value::String(device_id));
        }
    }
    if let Some(home) = getenv_nonempty("MATRIX_HOME_ROOM") {
        if config.platforms.contains_key(&Platform::Matrix) {
            set_home(config, Platform::Matrix, home, "MATRIX_HOME_ROOM_NAME", "Home", "MATRIX_HOME_ROOM_THREAD_ID");
        }
    }

    // Home Assistant
    if let Some(token) = getenv_nonempty("HASS_TOKEN") {
        let p = config.platforms.entry(Platform::Homeassistant).or_default();
        p.enabled = true;
        p.token = Some(token);
        if let Some(url) = getenv_nonempty("HASS_URL") {
            p.extra.insert("url".into(), Value::String(url));
        }
    }

    // Email
    let email_addr = getenv_nonempty("EMAIL_ADDRESS");
    let email_pwd = getenv_nonempty("EMAIL_PASSWORD");
    let email_imap = getenv_nonempty("EMAIL_IMAP_HOST");
    let email_smtp = getenv_nonempty("EMAIL_SMTP_HOST");
    if email_addr.is_some() && email_pwd.is_some() && email_imap.is_some() && email_smtp.is_some() {
        let addr = email_addr.unwrap();
        let imap = email_imap.unwrap();
        let smtp = email_smtp.unwrap();
        let p = config.platforms.entry(Platform::Email).or_default();
        p.enabled = true;
        p.extra.insert("address".into(), Value::String(addr));
        p.extra.insert("imap_host".into(), Value::String(imap));
        p.extra.insert("smtp_host".into(), Value::String(smtp));
    }
    if let Some(home) = getenv_nonempty("EMAIL_HOME_ADDRESS") {
        if config.platforms.contains_key(&Platform::Email) {
            set_home(config, Platform::Email, home, "EMAIL_HOME_ADDRESS_NAME", "Home", "EMAIL_HOME_ADDRESS_THREAD_ID");
        }
    }

    // SMS (Twilio)
    if getenv_nonempty("TWILIO_ACCOUNT_SID").is_some() {
        let p = config.platforms.entry(Platform::Sms).or_default();
        p.enabled = true;
        p.api_key = Some(env::var("TWILIO_AUTH_TOKEN").unwrap_or_default());
    }
    if let Some(home) = getenv_nonempty("SMS_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Sms) {
            set_home(config, Platform::Sms, home, "SMS_HOME_CHANNEL_NAME", "Home", "SMS_HOME_CHANNEL_THREAD_ID");
        }
    }

    // API Server
    let api_server_enabled = getenv_in("API_SERVER_ENABLED", &["true", "1", "yes"]);
    let api_server_key = env::var("API_SERVER_KEY").unwrap_or_default();
    let api_server_cors = env::var("API_SERVER_CORS_ORIGINS").unwrap_or_default();
    let api_server_port = getenv_nonempty("API_SERVER_PORT");
    let api_server_host = getenv_nonempty("API_SERVER_HOST");
    if api_server_enabled || !api_server_key.is_empty() {
        let p = config.platforms.entry(Platform::ApiServer).or_default();
        p.enabled = true;
        if !api_server_key.is_empty() {
            p.extra.insert("key".into(), Value::String(api_server_key));
        }
        if !api_server_cors.is_empty() {
            let origins: Vec<Value> = api_server_cors
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| Value::String(s.to_string()))
                .collect();
            if !origins.is_empty() {
                p.extra.insert("cors_origins".into(), Value::Array(origins));
            }
        }
        if let Some(port) = api_server_port {
            if let Ok(n) = port.parse::<i64>() {
                p.extra.insert("port".into(), Value::Number(n.into()));
            }
        }
        if let Some(host) = api_server_host {
            p.extra.insert("host".into(), Value::String(host));
        }
        let model_name = env::var("API_SERVER_MODEL_NAME").unwrap_or_default();
        if !model_name.is_empty() {
            p.extra.insert("model_name".into(), Value::String(model_name));
        }
    }

    // Webhook
    if getenv_in("WEBHOOK_ENABLED", &["true", "1", "yes"]) {
        let webhook_port = getenv_nonempty("WEBHOOK_PORT");
        let webhook_secret = env::var("WEBHOOK_SECRET").unwrap_or_default();
        let p = config.platforms.entry(Platform::Webhook).or_default();
        p.enabled = true;
        if let Some(port) = webhook_port {
            if let Ok(n) = port.parse::<i64>() {
                p.extra.insert("port".into(), Value::Number(n.into()));
            }
        }
        if !webhook_secret.is_empty() {
            p.extra.insert("secret".into(), Value::String(webhook_secret));
        }
    }

    // DingTalk
    let dingtalk_id = getenv_nonempty("DINGTALK_CLIENT_ID");
    let dingtalk_secret = getenv_nonempty("DINGTALK_CLIENT_SECRET");
    if let (Some(id), Some(secret)) = (dingtalk_id, dingtalk_secret) {
        let p = config.platforms.entry(Platform::Dingtalk).or_default();
        p.enabled = true;
        p.extra.insert("client_id".into(), Value::String(id));
        p.extra.insert("client_secret".into(), Value::String(secret));
        if let Some(home) = getenv_nonempty("DINGTALK_HOME_CHANNEL") {
            set_home(config, Platform::Dingtalk, home, "DINGTALK_HOME_CHANNEL_NAME", "Home", "DINGTALK_HOME_CHANNEL_THREAD_ID");
        }
    }

    // Feishu / Lark
    let feishu_id = getenv_nonempty("FEISHU_APP_ID");
    let feishu_secret = getenv_nonempty("FEISHU_APP_SECRET");
    if let (Some(id), Some(secret)) = (feishu_id, feishu_secret) {
        {
            let p = config.platforms.entry(Platform::Feishu).or_default();
            p.enabled = true;
            p.extra.insert("app_id".into(), Value::String(id));
            p.extra.insert("app_secret".into(), Value::String(secret));
            p.extra.insert(
                "domain".into(),
                Value::String(env::var("FEISHU_DOMAIN").unwrap_or_else(|_| "feishu".into())),
            );
            p.extra.insert(
                "connection_mode".into(),
                Value::String(
                    env::var("FEISHU_CONNECTION_MODE").unwrap_or_else(|_| "websocket".into()),
                ),
            );
            let encrypt_key = env::var("FEISHU_ENCRYPT_KEY").unwrap_or_default();
            if !encrypt_key.is_empty() {
                p.extra
                    .insert("encrypt_key".into(), Value::String(encrypt_key));
            }
            let verification_token = env::var("FEISHU_VERIFICATION_TOKEN").unwrap_or_default();
            if !verification_token.is_empty() {
                p.extra
                    .insert("verification_token".into(), Value::String(verification_token));
            }
        }
        if let Some(home) = getenv_nonempty("FEISHU_HOME_CHANNEL") {
            set_home(config, Platform::Feishu, home, "FEISHU_HOME_CHANNEL_NAME", "Home", "FEISHU_HOME_CHANNEL_THREAD_ID");
        }
    }

    // WeCom
    let wecom_bot = getenv_nonempty("WECOM_BOT_ID");
    let wecom_secret = getenv_nonempty("WECOM_SECRET");
    if let (Some(bot), Some(secret)) = (wecom_bot, wecom_secret) {
        {
            let p = config.platforms.entry(Platform::Wecom).or_default();
            p.enabled = true;
            p.extra.insert("bot_id".into(), Value::String(bot));
            p.extra.insert("secret".into(), Value::String(secret));
            let ws_url = env::var("WECOM_WEBSOCKET_URL").unwrap_or_default();
            if !ws_url.is_empty() {
                p.extra.insert("websocket_url".into(), Value::String(ws_url));
            }
        }
        if let Some(home) = getenv_nonempty("WECOM_HOME_CHANNEL") {
            set_home(config, Platform::Wecom, home, "WECOM_HOME_CHANNEL_NAME", "Home", "WECOM_HOME_CHANNEL_THREAD_ID");
        }
    }

    // WeCom callback mode
    let wcc_corp = getenv_nonempty("WECOM_CALLBACK_CORP_ID");
    let wcc_secret = getenv_nonempty("WECOM_CALLBACK_CORP_SECRET");
    if let (Some(corp), Some(secret)) = (wcc_corp, wcc_secret) {
        let p = config.platforms.entry(Platform::WecomCallback).or_default();
        p.enabled = true;
        p.extra.insert("corp_id".into(), Value::String(corp));
        p.extra.insert("corp_secret".into(), Value::String(secret));
        p.extra.insert(
            "agent_id".into(),
            Value::String(env::var("WECOM_CALLBACK_AGENT_ID").unwrap_or_default()),
        );
        p.extra.insert(
            "token".into(),
            Value::String(env::var("WECOM_CALLBACK_TOKEN").unwrap_or_default()),
        );
        p.extra.insert(
            "encoding_aes_key".into(),
            Value::String(env::var("WECOM_CALLBACK_ENCODING_AES_KEY").unwrap_or_default()),
        );
        p.extra.insert(
            "host".into(),
            Value::String(env::var("WECOM_CALLBACK_HOST").unwrap_or_else(|_| "0.0.0.0".into())),
        );
        let port = env::var("WECOM_CALLBACK_PORT")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(8645);
        p.extra.insert("port".into(), Value::Number(port.into()));
    }

    // Weixin
    let weixin_token = getenv_nonempty("WEIXIN_TOKEN");
    let weixin_account_id = getenv_nonempty("WEIXIN_ACCOUNT_ID");
    if weixin_token.is_some() || weixin_account_id.is_some() {
        {
            let p = config.platforms.entry(Platform::Weixin).or_default();
            p.enabled = true;
            if let Some(t) = &weixin_token {
                p.token = Some(t.clone());
            }
            if let Some(acc) = &weixin_account_id {
                p.extra.insert("account_id".into(), Value::String(acc.clone()));
            }
            let base_url = env::var("WEIXIN_BASE_URL").unwrap_or_default().trim().to_string();
            if !base_url.is_empty() {
                p.extra.insert(
                    "base_url".into(),
                    Value::String(base_url.trim_end_matches('/').to_string()),
                );
            }
            let cdn_base_url = env::var("WEIXIN_CDN_BASE_URL")
                .unwrap_or_default()
                .trim()
                .to_string();
            if !cdn_base_url.is_empty() {
                p.extra.insert(
                    "cdn_base_url".into(),
                    Value::String(cdn_base_url.trim_end_matches('/').to_string()),
                );
            }
            let dm_policy = env::var("WEIXIN_DM_POLICY")
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            if !dm_policy.is_empty() {
                p.extra.insert("dm_policy".into(), Value::String(dm_policy));
            }
            let group_policy = env::var("WEIXIN_GROUP_POLICY")
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            if !group_policy.is_empty() {
                p.extra
                    .insert("group_policy".into(), Value::String(group_policy));
            }
            let allowed = env::var("WEIXIN_ALLOWED_USERS").unwrap_or_default().trim().to_string();
            if !allowed.is_empty() {
                p.extra.insert("allow_from".into(), Value::String(allowed));
            }
            let group_allowed = env::var("WEIXIN_GROUP_ALLOWED_USERS")
                .unwrap_or_default()
                .trim()
                .to_string();
            if !group_allowed.is_empty() {
                p.extra
                    .insert("group_allow_from".into(), Value::String(group_allowed));
            }
            let split_multiline = env::var("WEIXIN_SPLIT_MULTILINE_MESSAGES")
                .unwrap_or_default()
                .trim()
                .to_string();
            if !split_multiline.is_empty() {
                p.extra.insert(
                    "split_multiline_messages".into(),
                    Value::String(split_multiline),
                );
            }
        }
        let weixin_home = env::var("WEIXIN_HOME_CHANNEL").unwrap_or_default().trim().to_string();
        if !weixin_home.is_empty() {
            set_home(config, Platform::Weixin, weixin_home, "WEIXIN_HOME_CHANNEL_NAME", "Home", "WEIXIN_HOME_CHANNEL_THREAD_ID");
        }
    }

    // BlueBubbles
    let bb_url = getenv_nonempty("BLUEBUBBLES_SERVER_URL");
    let bb_pwd = getenv_nonempty("BLUEBUBBLES_PASSWORD");
    if let (Some(url), Some(pwd)) = (bb_url, bb_pwd) {
        let p = config.platforms.entry(Platform::Bluebubbles).or_default();
        p.enabled = true;
        p.extra.insert(
            "server_url".into(),
            Value::String(url.trim_end_matches('/').to_string()),
        );
        p.extra.insert("password".into(), Value::String(pwd));
        p.extra.insert(
            "webhook_host".into(),
            Value::String(env::var("BLUEBUBBLES_WEBHOOK_HOST").unwrap_or_else(|_| "127.0.0.1".into())),
        );
        let wport = env::var("BLUEBUBBLES_WEBHOOK_PORT")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(8645);
        p.extra.insert("webhook_port".into(), Value::Number(wport.into()));
        p.extra.insert(
            "webhook_path".into(),
            Value::String(
                env::var("BLUEBUBBLES_WEBHOOK_PATH").unwrap_or_else(|_| "/bluebubbles-webhook".into()),
            ),
        );
        p.extra.insert(
            "send_read_receipts".into(),
            Value::Bool(env_in_default(
                "BLUEBUBBLES_SEND_READ_RECEIPTS",
                "true",
                &["true", "1", "yes"],
            )),
        );
    }
    if let Some(home) = getenv_nonempty("BLUEBUBBLES_HOME_CHANNEL") {
        if config.platforms.contains_key(&Platform::Bluebubbles) {
            set_home(config, Platform::Bluebubbles, home, "BLUEBUBBLES_HOME_CHANNEL_NAME", "Home", "BLUEBUBBLES_HOME_CHANNEL_THREAD_ID");
        }
    }

    // QQ
    let qq_app_id = getenv_nonempty("QQ_APP_ID");
    let qq_client_secret = getenv_nonempty("QQ_CLIENT_SECRET");
    if qq_app_id.is_some() || qq_client_secret.is_some() {
        {
            let p = config.platforms.entry(Platform::Qqbot).or_default();
            p.enabled = true;
            if let Some(id) = &qq_app_id {
                p.extra.insert("app_id".into(), Value::String(id.clone()));
            }
            if let Some(secret) = &qq_client_secret {
                p.extra
                    .insert("client_secret".into(), Value::String(secret.clone()));
            }
            let allowed = env::var("QQ_ALLOWED_USERS").unwrap_or_default().trim().to_string();
            if !allowed.is_empty() {
                p.extra.insert("allow_from".into(), Value::String(allowed));
            }
            let group_allowed = env::var("QQ_GROUP_ALLOWED_USERS")
                .unwrap_or_default()
                .trim()
                .to_string();
            if !group_allowed.is_empty() {
                p.extra
                    .insert("group_allow_from".into(), Value::String(group_allowed));
            }
        }
        let mut qq_home = env::var("QQBOT_HOME_CHANNEL").unwrap_or_default().trim().to_string();
        let mut qq_home_name_env = "QQBOT_HOME_CHANNEL_NAME";
        if qq_home.is_empty() {
            let legacy = env::var("QQ_HOME_CHANNEL").unwrap_or_default().trim().to_string();
            if !legacy.is_empty() {
                qq_home = legacy;
                qq_home_name_env = "QQ_HOME_CHANNEL_NAME";
                log::warn!(
                    "QQ_HOME_CHANNEL is deprecated; rename to QQBOT_HOME_CHANNEL \
                     in your .env for consistency with the platform key."
                );
            }
        }
        if !qq_home.is_empty() {
            let name = match env::var("QQBOT_HOME_CHANNEL_NAME") {
                Ok(v) if !v.is_empty() => v,
                _ => env::var(qq_home_name_env).unwrap_or_else(|_| "Home".into()),
            };
            let thread_id = env::var("QQBOT_HOME_CHANNEL_THREAD_ID")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| env::var("QQ_HOME_CHANNEL_THREAD_ID").ok().filter(|s| !s.is_empty()));
            let p = config.platforms.entry(Platform::Qqbot).or_default();
            p.home_channel = Some(HomeChannel::new(Platform::Qqbot, qq_home, name, thread_id));
        }
    }

    // Yuanbao
    let yuanbao_app_id = getenv_nonempty("YUANBAO_APP_ID").or_else(|| getenv_nonempty("YUANBAO_APP_KEY"));
    let yuanbao_app_secret = getenv_nonempty("YUANBAO_APP_SECRET");
    if let (Some(id), Some(secret)) = (yuanbao_app_id, yuanbao_app_secret) {
        {
            let p = config.platforms.entry(Platform::Yuanbao).or_default();
            p.enabled = true;
            p.extra.insert("app_id".into(), Value::String(id));
            p.extra.insert("app_secret".into(), Value::String(secret));
            if let Some(bot_id) = getenv_nonempty("YUANBAO_BOT_ID") {
                p.extra.insert("bot_id".into(), Value::String(bot_id));
            }
            if let Some(ws_url) = getenv_nonempty("YUANBAO_WS_URL") {
                p.extra.insert("ws_url".into(), Value::String(ws_url));
            }
            if let Some(api_domain) = getenv_nonempty("YUANBAO_API_DOMAIN") {
                p.extra.insert("api_domain".into(), Value::String(api_domain));
            }
            if let Some(route_env) = getenv_nonempty("YUANBAO_ROUTE_ENV") {
                p.extra.insert("route_env".into(), Value::String(route_env));
            }
        }
        if let Some(home) = getenv_nonempty("YUANBAO_HOME_CHANNEL") {
            set_home(config, Platform::Yuanbao, home, "YUANBAO_HOME_CHANNEL_NAME", "Home", "YUANBAO_HOME_CHANNEL_THREAD_ID");
        }
        let p = config.platforms.entry(Platform::Yuanbao).or_default();
        if let Some(dm_policy) = getenv_nonempty("YUANBAO_DM_POLICY") {
            p.extra
                .insert("dm_policy".into(), Value::String(dm_policy.trim().to_lowercase()));
        }
        if let Some(dm_allow) = getenv_nonempty("YUANBAO_DM_ALLOW_FROM") {
            p.extra.insert("dm_allow_from".into(), Value::String(dm_allow));
        }
        if let Some(group_policy) = getenv_nonempty("YUANBAO_GROUP_POLICY") {
            p.extra.insert(
                "group_policy".into(),
                Value::String(group_policy.trim().to_lowercase()),
            );
        }
        if let Some(group_allow) = getenv_nonempty("YUANBAO_GROUP_ALLOW_FROM") {
            p.extra
                .insert("group_allow_from".into(), Value::String(group_allow));
        }
    }

    // Session settings
    if let Some(idle) = getenv_nonempty("SESSION_IDLE_MINUTES") {
        if let Ok(n) = idle.parse::<i64>() {
            config.default_reset_policy.idle_minutes = n;
        }
    }
    if let Some(hour) = getenv_nonempty("SESSION_RESET_HOUR") {
        if let Ok(n) = hour.parse::<i64>() {
            config.default_reset_policy.at_hour = n;
        }
    }

    // Plugin-registry enable pass: registry not modelled in this port -> no-op.
}

/// Set a platform's home channel from env-var-derived values.
fn set_home(
    config: &mut GatewayConfig,
    platform: Platform,
    chat_id: String,
    name_env: &str,
    name_default: &str,
    thread_env: &str,
) {
    let name = match env::var(name_env) {
        Ok(v) if !v.is_empty() => v,
        _ => name_default.to_string(),
    };
    let thread_id = env::var(thread_env).ok().filter(|s| !s.is_empty());
    let p = config.platforms.entry(platform.clone()).or_default();
    p.home_channel = Some(HomeChannel::new(platform, chat_id, name, thread_id));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Set a process env var. Wrapped helper so the `unsafe` lives in one place.
fn set_env((key, value): (&str, String)) {
    unsafe {
        env::set_var(key, value);
    }
}

/// Read an env var, returning `Some` only when set and non-empty.
fn getenv_nonempty(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Read an env var, lowercased; empty when unset.
fn getenv_lower(name: &str) -> String {
    env::var(name).unwrap_or_default().to_lowercase()
}

/// True when the env var (lowercased) is in `choices`.
fn getenv_in(name: &str, choices: &[&str]) -> bool {
    let v = env::var(name).unwrap_or_default().to_lowercase();
    choices.contains(&v.as_str())
}

/// True when env var (default `default` when unset, lowercased) is in `choices`.
fn getenv_in_default(name: &str, default: &str, choices: &[&str]) -> bool {
    let v = env::var(name).unwrap_or_else(|_| default.to_string()).to_lowercase();
    choices.contains(&v.as_str())
}

/// Same as `getenv_in_default` but the default is applied before lowercasing.
fn env_in_default(name: &str, default: &str, choices: &[&str]) -> bool {
    getenv_in_default(name, default, choices)
}

/// True when env var is truthy via the shared truthy-string set (used by
/// connection checkers like `bool(os.getenv(...))`).
fn env_truthy(name: &str) -> bool {
    getenv_nonempty(name).is_some()
}

/// Set `env_key` to the lowercased string form of `map[yaml_key]` when present
/// and `env_key` is not already set. Mirrors `str(x).lower()` bridging.
fn set_env_if_unset_lower(map: &Map<String, Value>, yaml_key: &str, env_key: &str) {
    if let Some(v) = map.get(yaml_key) {
        if env::var(env_key).is_err() {
            set_env((env_key, py_str_lower(v)));
        }
    }
}

/// Bridge a list-or-scalar config value to a CSV env var when unset.
/// Mirrors `",".join(str(v) for v in frc)` for lists, else `str(frc)`.
fn set_env_csv_if_unset(map: &Map<String, Value>, yaml_key: &str, env_key: &str) {
    if let Some(v) = map.get(yaml_key) {
        if !v.is_null() && env::var(env_key).is_err() {
            let s = match v {
                Value::Array(a) => a.iter().map(py_str).collect::<Vec<_>>().join(","),
                other => py_str(other),
            };
            set_env((env_key, s));
        }
    }
}

/// Bridge reply_to_mode (top-level or extra.reply_to_mode) to an env var.
/// YAML 1.1 parses bare `off` as boolean false -> coerce to string "off".
fn set_reply_to_mode_env(section: &Map<String, Value>, env_key: &str) {
    let extra = section.get("extra").and_then(|v| v.as_object());
    let rtm = if section.contains_key("reply_to_mode") {
        section.get("reply_to_mode").cloned()
    } else {
        extra.and_then(|e| e.get("reply_to_mode")).cloned()
    };
    if let Some(v) = rtm {
        if !v.is_null() && env::var(env_key).is_err() {
            let s = if matches!(v, Value::Bool(false)) {
                "off".to_string()
            } else {
                py_str_lower(&v)
            };
            set_env((env_key, s));
        }
    }
}

/// Python `str(value)` for a JSON value (used when bridging YAML scalars).
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::Null => "None".into(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Python `str(value).lower()`.
fn py_str_lower(v: &Value) -> String {
    py_str(v).to_lowercase()
}

/// Python `str(value)` but treating JSON null/bool numerically as in dict keys.
/// Used when coercing config values to strings (e.g. chat_id, reset_triggers).
fn json_to_py_str(v: &Value) -> String {
    py_str(v)
}

fn value_as_str(v: &Value) -> Option<String> {
    Some(py_str(v))
}

/// Truthiness of a JSON value following Python `bool(x)` semantics.
fn is_truthy_json(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Build a JSON number from an f64, preserving integers where possible.
fn json_number(f: f64) -> Value {
    if f.fract() == 0.0 && f.is_finite() && f.abs() < 9.007e15 {
        Value::Number((f as i64).into())
    } else {
        serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

/// Extract an `Option<String>` from a JSON value: `null` -> None, string -> the
/// string, other -> Python `str(...)` form. Mirrors `data.get("token")` which
/// may yield None or a string.
fn opt_string_from_value(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(py_str(other)),
    }
}

/// Strict integer coercion used for `session_store_max_age_days`. Mirrors
/// `int(data.get(..., 90))` which raises (-> default) on non-numeric input.
/// Returns `None` when the value cannot be parsed as an int.
fn coerce_int_strict(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// Used by `SessionResetPolicy.from_dict` for at_hour/idle_minutes which Python
/// stores verbatim (no coercion) when present. We accept int/float/string and
/// fall back to `default` otherwise, since downstream validation expects ints.
fn coerce_int_or_passthrough(v: &Value, default: i64) -> i64 {
    coerce_int_strict(v).unwrap_or(default)
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Convert a `serde_yaml::Value` into a `serde_json::Value`, mapping YAML
/// mappings to JSON objects (non-string keys are stringified to match Python's
/// behavior where YAML keys become dict keys).
fn yaml_to_json(v: serde_yaml::Value) -> Value {
    match v {
        serde_yaml::Value::Null => Value::Null,
        serde_yaml::Value::Bool(b) => Value::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                Value::Number(u.into())
            } else if let Some(f) = n.as_f64() {
                json_number(f)
            } else {
                Value::Null
            }
        }
        serde_yaml::Value::String(s) => Value::String(s),
        serde_yaml::Value::Sequence(seq) => {
            Value::Array(seq.into_iter().map(yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let mut obj = Map::new();
            for (k, val) in map {
                let key = match k {
                    serde_yaml::Value::String(s) => s,
                    serde_yaml::Value::Bool(b) => b.to_string(),
                    serde_yaml::Value::Number(n) => n.to_string(),
                    serde_yaml::Value::Null => "null".to_string(),
                    other => serde_yaml::to_string(&other)
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                };
                obj.insert(key, yaml_to_json(val));
            }
            Value::Object(obj)
        }
        serde_yaml::Value::Tagged(t) => yaml_to_json(t.value),
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
    fn coerce_bool_strings_and_defaults() {
        assert!(coerce_bool(Some(&json!("yes")), false));
        assert!(!coerce_bool(Some(&json!("off")), true));
        assert!(coerce_bool(Some(&json!("maybe")), true));
        assert!(!coerce_bool(Some(&json!("maybe")), false));
        assert!(coerce_bool(None, true));
        assert!(!coerce_bool(Some(&Value::Null), false));
        assert!(coerce_bool(Some(&json!(true)), false));
    }

    #[test]
    fn coerce_float_and_int_fallbacks() {
        assert_eq!(coerce_float(Some(&json!("1.5")), 0.0), 1.5);
        assert_eq!(coerce_float(Some(&json!("bad")), 9.0), 9.0);
        assert_eq!(coerce_int(Some(&json!("40")), 1), 40);
        assert_eq!(coerce_int(Some(&json!("1.5")), 7), 7);
        assert_eq!(coerce_int(Some(&json!(40)), 1), 40);
    }

    #[test]
    fn normalize_helpers() {
        assert_eq!(
            normalize_unauthorized_dm_behavior(Some(&json!("IGNORE")), "pair"),
            "ignore"
        );
        assert_eq!(
            normalize_unauthorized_dm_behavior(Some(&json!("bogus")), "pair"),
            "pair"
        );
        assert_eq!(
            normalize_notice_delivery(Some(&json!("Private")), "public"),
            "private"
        );
        assert_eq!(normalize_notice_delivery(None, "public"), "public");
    }

    #[test]
    fn platform_parse_builtin_and_plugin() {
        assert_eq!(Platform::parse("telegram"), Some(Platform::Telegram));
        assert_eq!(Platform::parse("TELEGRAM"), Some(Platform::Telegram));
        assert_eq!(Platform::parse_builtin("irc"), None);
        assert_eq!(
            Platform::parse(" irc "),
            Some(Platform::Plugin("irc".into()))
        );
        assert_eq!(Platform::parse(""), None);
        assert_eq!(Platform::Telegram.value(), "telegram");
        assert_eq!(Platform::WecomCallback.value(), "wecom_callback");
    }

    #[test]
    fn home_channel_round_trip() {
        let hc = HomeChannel::new(Platform::Telegram, "123", "MyHome", Some("9".into()));
        let d = hc.to_dict();
        assert_eq!(d["platform"], json!("telegram"));
        assert_eq!(d["chat_id"], json!("123"));
        assert_eq!(d["thread_id"], json!("9"));
        let back = HomeChannel::from_dict(&d).unwrap();
        assert_eq!(back, hc);

        // chat_id stringified from int; missing thread_id -> None.
        let d2 = json!({"platform": "discord", "chat_id": 42});
        let hc2 = HomeChannel::from_dict(&d2).unwrap();
        assert_eq!(hc2.chat_id, "42");
        assert_eq!(hc2.name, "Home");
        assert_eq!(hc2.thread_id, None);
    }

    #[test]
    fn session_reset_policy_from_dict_nulls() {
        let d = json!({
            "mode": serde_json::Value::Null,
            "at_hour": serde_json::Value::Null,
            "idle_minutes": serde_json::Value::Null,
            "notify": serde_json::Value::Null,
            "notify_exclude_platforms": serde_json::Value::Null,
        });
        let p = SessionResetPolicy::from_dict(&d);
        assert_eq!(p, SessionResetPolicy::default());

        let d2 = json!({"mode": "idle", "at_hour": 6, "idle_minutes": 30, "notify": false, "notify_exclude_platforms": ["x"]});
        let p2 = SessionResetPolicy::from_dict(&d2);
        assert_eq!(p2.mode, "idle");
        assert_eq!(p2.at_hour, 6);
        assert_eq!(p2.idle_minutes, 30);
        assert!(!p2.notify);
        assert_eq!(p2.notify_exclude_platforms, vec!["x".to_string()]);
    }

    #[test]
    fn platform_config_from_dict() {
        let d = json!({
            "enabled": "yes",
            "token": "abc",
            "extra": {"foo": 1},
            "reply_to_mode": "all",
        });
        let pc = PlatformConfig::from_dict(&d);
        assert!(pc.enabled);
        assert_eq!(pc.token.as_deref(), Some("abc"));
        assert_eq!(pc.reply_to_mode, "all");
        assert_eq!(pc.extra.get("foo"), Some(&json!(1)));
    }

    #[test]
    fn streaming_config_empty_is_default() {
        assert_eq!(
            StreamingConfig::from_dict(&json!({})),
            StreamingConfig::default()
        );
        let s = StreamingConfig::from_dict(&json!({"enabled": true, "buffer_threshold": "80"}));
        assert!(s.enabled);
        assert_eq!(s.buffer_threshold, 80);
    }

    #[test]
    fn gateway_from_dict_skips_unknown_platforms() {
        let d = json!({
            "platforms": {
                "telegram": {"enabled": true, "token": "t"},
                "irc": {"enabled": true},
            },
            "session_store_max_age_days": -5,
        });
        let cfg = GatewayConfig::from_dict(&d);
        assert!(cfg.platforms.contains_key(&Platform::Telegram));
        assert!(!cfg.platforms.contains_key(&Platform::Plugin("irc".into())));
        // Negative clamps to 0.
        assert_eq!(cfg.session_store_max_age_days, 0);
    }

    #[test]
    fn gateway_stt_falls_back_to_nested() {
        let d = json!({"stt": {"enabled": false}});
        let cfg = GatewayConfig::from_dict(&d);
        assert!(!cfg.stt_enabled);

        let d2 = json!({"stt_enabled": false, "stt": {"enabled": true}});
        let cfg2 = GatewayConfig::from_dict(&d2);
        assert!(!cfg2.stt_enabled);
    }

    #[test]
    fn connected_platforms_generic_and_specific() {
        let mut cfg = GatewayConfig::default();
        let mut tg = PlatformConfig::default();
        tg.enabled = true;
        tg.token = Some("realtoken".into());
        cfg.platforms.insert(Platform::Telegram, tg);

        let mut weixin = PlatformConfig::default();
        weixin.enabled = true;
        weixin.token = Some("wtoken".into());
        // Missing account_id -> not connected despite token.
        cfg.platforms.insert(Platform::Weixin, weixin);

        let connected = cfg.get_connected_platforms();
        assert!(connected.contains(&Platform::Telegram));
        assert!(!connected.contains(&Platform::Weixin));

        // Add account_id -> now connected.
        cfg.platforms
            .get_mut(&Platform::Weixin)
            .unwrap()
            .extra
            .insert("account_id".into(), json!("acc"));
        assert!(cfg.get_connected_platforms().contains(&Platform::Weixin));
    }

    #[test]
    fn reset_policy_priority() {
        let mut cfg = GatewayConfig::default();
        let mut plat_policy = SessionResetPolicy::default();
        plat_policy.mode = "daily".into();
        cfg.reset_by_platform
            .insert(Platform::Telegram, plat_policy.clone());
        let mut type_policy = SessionResetPolicy::default();
        type_policy.mode = "idle".into();
        cfg.reset_by_type.insert("dm".into(), type_policy.clone());

        assert_eq!(
            cfg.get_reset_policy(Some(&Platform::Telegram), Some("dm")).mode,
            "daily"
        );
        assert_eq!(cfg.get_reset_policy(None, Some("dm")).mode, "idle");
        assert_eq!(cfg.get_reset_policy(None, None).mode, "both");
    }

    #[test]
    fn validate_clamps_at_hour_and_disables_placeholder_token() {
        let mut cfg = GatewayConfig::default();
        cfg.default_reset_policy.at_hour = 99;
        cfg.default_reset_policy.idle_minutes = 0;
        let mut tg = PlatformConfig::default();
        tg.enabled = true;
        tg.token = Some("none".into()); // placeholder
        cfg.platforms.insert(Platform::Telegram, tg);

        validate_gateway_config(&mut cfg);
        assert_eq!(cfg.default_reset_policy.at_hour, 4);
        assert_eq!(cfg.default_reset_policy.idle_minutes, 1440);
        assert!(!cfg.platforms.get(&Platform::Telegram).unwrap().enabled);
    }

    #[test]
    fn env_override_enables_telegram() {
        unsafe {
            std::env::set_var("TELEGRAM_BOT_TOKEN", "xyztoken");
        }
        let mut cfg = GatewayConfig::default();
        apply_env_overrides(&mut cfg);
        let tg = cfg.platforms.get(&Platform::Telegram).unwrap();
        assert!(tg.enabled);
        assert_eq!(tg.token.as_deref(), Some("xyztoken"));
        unsafe {
            std::env::remove_var("TELEGRAM_BOT_TOKEN");
        }
    }

    #[test]
    fn unauthorized_dm_and_notice_delivery_overrides() {
        let mut cfg = GatewayConfig::default();
        let mut pc = PlatformConfig::default();
        pc.extra
            .insert("unauthorized_dm_behavior".into(), json!("ignore"));
        pc.extra.insert("notice_delivery".into(), json!("private"));
        cfg.platforms.insert(Platform::Telegram, pc);
        assert_eq!(
            cfg.get_unauthorized_dm_behavior(Some(&Platform::Telegram)),
            "ignore"
        );
        assert_eq!(
            cfg.get_notice_delivery(Some(&Platform::Telegram)),
            "private"
        );
        assert_eq!(cfg.get_notice_delivery(None), "public");
    }

    #[test]
    fn yaml_to_json_mapping_keys() {
        let y: serde_yaml::Value = serde_yaml::from_str("123: foo\ntrue: bar\nname: baz").unwrap();
        let j = yaml_to_json(y);
        let obj = j.as_object().unwrap();
        assert_eq!(obj.get("123"), Some(&json!("foo")));
        assert_eq!(obj.get("true"), Some(&json!("bar")));
        assert_eq!(obj.get("name"), Some(&json!("baz")));
    }
}
