//! Delivery routing for cron job outputs and agent responses.
//!
//! Port of `gateway/delivery.py`.
//!
//! Routes messages to the appropriate destination based on:
//! - Explicit targets (e.g., `telegram:123456789`)
//! - Platform home channels (e.g., `telegram` -> home channel)
//! - Origin (back to where the job was created)
//! - Local (always saved to files)
//!
//! The Python module relies on three external seams:
//!   * `hermes_cli.config.get_hermes_home` -> [`crate::mod_hermes_constants::get_hermes_home`]
//!   * `gateway.config.Platform` -> [`crate::gw_config::Platform`]
//!   * `gateway.session.SessionSource` -> [`crate::gw_session::SessionSource`]
//!   * `tools.send_message_tool._parse_target_ref` -> [`parse_target_ref`] (ported below)
//!
//! Platform adapters (`Dict[Platform, Any]`) are abstracted behind the
//! [`DeliveryAdapter`] trait so this module does not depend on every concrete
//! adapter implementation. Callers register adapters keyed by [`Platform`].

use std::collections::BTreeMap;
use std::path::PathBuf;

use chrono::Local;
use regex::Regex;
use serde_json::{Map, Value};

use crate::gw_config::{GatewayConfig, Platform};
use crate::gw_session::SessionSource;
use crate::mod_hermes_constants::get_hermes_home;

/// Hard cap on platform output length before we truncate + spill to disk.
pub const MAX_PLATFORM_OUTPUT: usize = 4000;
/// How many leading chars remain visible when a message is truncated.
pub const TRUNCATED_VISIBLE: usize = 3800;

// ---------------------------------------------------------------------------
// _parse_target_ref  (ported from tools/send_message_tool.py)
// ---------------------------------------------------------------------------

/// Phone-addressed platforms that accept E.164 (with leading `+`).
const PHONE_PLATFORMS: &[&str] = &["signal", "sms", "whatsapp"];

fn telegram_topic_re() -> Regex {
    Regex::new(r"^\s*(-?\d+)(?::(\d+))?\s*$").unwrap()
}
fn feishu_re() -> Regex {
    Regex::new(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$").unwrap()
}
fn slack_re() -> Regex {
    Regex::new(r"^\s*([CGD][A-Z0-9]{8,})(?::([0-9]+(?:\.[0-9]+)?))?\s*$").unwrap()
}
fn weixin_re() -> Regex {
    Regex::new(
        r"^\s*((?:wxid|gh|v\d+|wm|wb)_[A-Za-z0-9_-]+|[A-Za-z0-9._-]+@chatroom|filehelper)\s*$",
    )
    .unwrap()
}
fn yuanbao_re() -> Regex {
    Regex::new(r"^\s*((?:group|direct):[^:]+)\s*$").unwrap()
}
fn signal_group_re() -> Regex {
    Regex::new(r"^\s*(group:\S+)\s*$").unwrap()
}
fn wecom_callback_re() -> Regex {
    Regex::new(r"^\s*([^:\s]+:[^:\s]+)\s*$").unwrap()
}
fn whatsapp_re() -> Regex {
    // re.IGNORECASE
    Regex::new(r"(?i)^\s*([A-Za-z0-9._:-]+@(?:lid|g\.us|s\.whatsapp\.net|broadcast))\s*$").unwrap()
}
fn e164_re() -> Regex {
    Regex::new(r"^\s*\+(\d{7,15})\s*$").unwrap()
}

/// True when `s`, with all leading `-` stripped, is non-empty and all ASCII
/// digits (mirrors Python `str.lstrip("-").isdigit()`).
fn is_lstrip_dash_digits(s: &str) -> bool {
    let stripped = s.trim_start_matches('-');
    !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit())
}

/// True when the trimmed string is non-empty and all ASCII digits (Python
/// `str.strip().isdigit()`).
fn is_strip_digits(s: &str) -> bool {
    let t = s.trim();
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

/// Parse a tool target into `(chat_id, thread_id, is_explicit)`.
///
/// Faithful port of `tools.send_message_tool._parse_target_ref`. `platform_name`
/// must already be lowercased to match the Python call sites.
pub fn parse_target_ref(
    platform_name: &str,
    target_ref: &str,
) -> (Option<String>, Option<String>, bool) {
    if platform_name == "telegram" {
        if let Some(m) = telegram_topic_re().captures(target_ref) {
            return (
                Some(m.get(1).unwrap().as_str().to_string()),
                m.get(2).map(|g| g.as_str().to_string()),
                true,
            );
        }
    }
    if platform_name == "feishu" {
        if let Some(m) = feishu_re().captures(target_ref) {
            return (
                Some(m.get(1).unwrap().as_str().to_string()),
                m.get(2).map(|g| g.as_str().to_string()),
                true,
            );
        }
    }
    if platform_name == "discord" {
        // _NUMERIC_TOPIC_RE == _TELEGRAM_TOPIC_TARGET_RE
        if let Some(m) = telegram_topic_re().captures(target_ref) {
            return (
                Some(m.get(1).unwrap().as_str().to_string()),
                m.get(2).map(|g| g.as_str().to_string()),
                true,
            );
        }
    }
    if platform_name == "slack" {
        if let Some(m) = slack_re().captures(target_ref) {
            return (
                Some(m.get(1).unwrap().as_str().to_string()),
                m.get(2).map(|g| g.as_str().to_string()),
                true,
            );
        }
    }
    if platform_name == "weixin" {
        if let Some(m) = weixin_re().captures(target_ref) {
            return (Some(m.get(1).unwrap().as_str().to_string()), None, true);
        }
    }
    if platform_name == "yuanbao" {
        if let Some(m) = yuanbao_re().captures(target_ref) {
            return (Some(m.get(1).unwrap().as_str().to_string()), None, true);
        }
        if is_strip_digits(target_ref) {
            return (Some(format!("group:{}", target_ref.trim())), None, true);
        }
        return (None, None, false);
    }
    if platform_name == "signal" {
        if let Some(m) = signal_group_re().captures(target_ref) {
            return (Some(m.get(1).unwrap().as_str().to_string()), None, true);
        }
    }
    if platform_name == "wecom_callback" {
        if let Some(m) = wecom_callback_re().captures(target_ref) {
            return (Some(m.get(1).unwrap().as_str().to_string()), None, true);
        }
    }
    if platform_name == "whatsapp" {
        if let Some(m) = whatsapp_re().captures(target_ref) {
            return (Some(m.get(1).unwrap().as_str().to_string()), None, true);
        }
    }
    if PHONE_PLATFORMS.contains(&platform_name) {
        if e164_re().is_match(target_ref) {
            // Preserve the leading '+' for E.164 recipients.
            return (Some(target_ref.trim().to_string()), None, true);
        }
    }
    if is_lstrip_dash_digits(target_ref) {
        return (Some(target_ref.to_string()), None, true);
    }
    // Matrix room IDs (start with !) and user IDs (start with @) are explicit.
    if platform_name == "matrix"
        && (target_ref.starts_with('!') || target_ref.starts_with('@'))
    {
        return (Some(target_ref.to_string()), None, true);
    }
    (None, None, false)
}

// ---------------------------------------------------------------------------
// DeliveryTarget
// ---------------------------------------------------------------------------

/// A single delivery target.
///
/// Represents where a message should be sent:
/// - `origin` -> back to source
/// - `local` -> save to local files
/// - `telegram` -> Telegram home channel
/// - `telegram:123456` -> specific Telegram chat
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryTarget {
    pub platform: Platform,
    /// `None` means use the home channel.
    pub chat_id: Option<String>,
    pub thread_id: Option<String>,
    pub is_origin: bool,
    /// True if `chat_id` was explicitly specified.
    pub is_explicit: bool,
}

impl DeliveryTarget {
    /// Construct with the Python dataclass defaults.
    pub fn new(platform: Platform) -> Self {
        DeliveryTarget {
            platform,
            chat_id: None,
            thread_id: None,
            is_origin: false,
            is_explicit: false,
        }
    }

    /// Parse a delivery target string.
    ///
    /// Formats:
    /// - `origin` -> back to source
    /// - `local` -> local files only
    /// - `telegram` -> Telegram home channel
    /// - `telegram:123456` -> specific Telegram chat
    pub fn parse(target: &str, origin: Option<&SessionSource>) -> DeliveryTarget {
        let target_stripped = target.trim();
        let target_lower = target_stripped.to_lowercase();

        if target_lower == "origin" {
            if let Some(origin) = origin {
                return DeliveryTarget {
                    platform: origin.platform.clone(),
                    chat_id: Some(origin.chat_id.clone()),
                    thread_id: origin.thread_id.clone(),
                    is_origin: true,
                    is_explicit: false,
                };
            }
            // Fallback to local if no origin.
            let mut t = DeliveryTarget::new(Platform::Local);
            t.is_origin = true;
            return t;
        }

        if target_lower == "local" {
            return DeliveryTarget::new(Platform::Local);
        }

        // Check for platform:chat_id or platform:chat_id:thread_id format.
        // Use the original case for chat_id/thread_id to preserve case-sensitive
        // IDs. Platform-specific parsing delegates to send_message's target
        // parser so cron/delivery routing stays aligned with explicit
        // send_message targets.
        if target_stripped.contains(':') {
            // platform_str, target_ref = split(":", 1)
            let (platform_str_raw, target_ref) = match target_stripped.split_once(':') {
                Some((a, b)) => (a, b),
                None => (target_stripped, ""),
            };
            let platform_str = platform_str_raw.to_lowercase();

            // Python: Platform(platform_str) raises ValueError for unknown -> local.
            let platform = match Platform::parse_builtin(&platform_str) {
                Some(p) => p,
                None => return DeliveryTarget::new(Platform::Local),
            };

            // Python wraps _parse_target_ref in try/except -> defaults on error.
            // Our port cannot panic, so this always succeeds.
            let (chat_id, thread_id, is_explicit) =
                parse_target_ref(&platform_str, target_ref);

            if is_explicit {
                return DeliveryTarget {
                    platform,
                    chat_id,
                    thread_id,
                    is_origin: false,
                    is_explicit: true,
                };
            }

            // Keep non-explicit targets intact. Some valid raw IDs and URLs
            // contain ':' but are not thread targets (e.g. Matrix aliases,
            // webhook URLs), and send_message/cron both pass them through as a
            // whole chat_id rather than inventing a thread split.
            return DeliveryTarget {
                platform,
                chat_id: Some(target_ref.to_string()),
                thread_id: None,
                is_origin: false,
                is_explicit: true,
            };
        }

        // Just a platform name (use home channel).
        match Platform::parse_builtin(&target_lower) {
            Some(platform) => DeliveryTarget::new(platform),
            None => DeliveryTarget::new(Platform::Local),
        }
    }

    /// Convert back to string format.
    pub fn to_string(&self) -> String {
        if self.is_origin {
            return "origin".to_string();
        }
        if self.platform == Platform::Local {
            return "local".to_string();
        }
        match (&self.chat_id, &self.thread_id) {
            (Some(chat_id), Some(thread_id)) if !chat_id.is_empty() && !thread_id.is_empty() => {
                format!("{}:{}:{}", self.platform.value(), chat_id, thread_id)
            }
            (Some(chat_id), _) if !chat_id.is_empty() => {
                format!("{}:{}", self.platform.value(), chat_id)
            }
            _ => self.platform.value(),
        }
    }
}

// ---------------------------------------------------------------------------
// Delivery results
// ---------------------------------------------------------------------------

/// Result of delivering to a single target.
#[derive(Debug, Clone)]
pub enum TargetResult {
    /// Delivery succeeded; carries the adapter/local result payload.
    Success(Value),
    /// Delivery failed with the given error string.
    Error(String),
}

impl TargetResult {
    /// Render as the dict Python stores in the results map.
    pub fn to_json(&self) -> Value {
        match self {
            TargetResult::Success(v) => {
                let mut m = Map::new();
                m.insert("success".into(), Value::Bool(true));
                m.insert("result".into(), v.clone());
                Value::Object(m)
            }
            TargetResult::Error(e) => {
                let mut m = Map::new();
                m.insert("success".into(), Value::Bool(false));
                m.insert("error".into(), Value::String(e.clone()));
                Value::Object(m)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DeliveryAdapter trait
// ---------------------------------------------------------------------------

/// Abstraction over a platform adapter's `send` coroutine.
///
/// In Python adapters are arbitrary objects exposing
/// `async def send(chat_id, content, metadata=None) -> dict`. We model the
/// metadata as an optional JSON object (`None` mirrors Python's
/// `metadata or None`).
pub trait DeliveryAdapter {
    /// Send `content` to `chat_id`. `metadata` is `None` when empty.
    ///
    /// Returns the adapter result payload or an error string.
    fn send(
        &self,
        chat_id: &str,
        content: &str,
        metadata: Option<&Map<String, Value>>,
    ) -> Result<Value, String>;
}

// ---------------------------------------------------------------------------
// DeliveryRouter
// ---------------------------------------------------------------------------

/// Routes messages to appropriate destinations.
///
/// Handles resolving delivery targets and dispatching messages to the right
/// platform adapters.
pub struct DeliveryRouter {
    pub config: GatewayConfig,
    pub adapters: BTreeMap<Platform, Box<dyn DeliveryAdapter>>,
    pub output_dir: PathBuf,
}

impl DeliveryRouter {
    /// Initialize the delivery router.
    pub fn new(config: GatewayConfig, adapters: BTreeMap<Platform, Box<dyn DeliveryAdapter>>) -> Self {
        let output_dir = get_hermes_home().join("cron").join("output");
        DeliveryRouter {
            config,
            adapters,
            output_dir,
        }
    }

    /// Deliver `content` to all specified targets.
    ///
    /// Returns a JSON object mapping `target.to_string()` -> result dict, exactly
    /// like the Python `deliver` coroutine.
    pub fn deliver(
        &self,
        content: &str,
        targets: &[DeliveryTarget],
        job_id: Option<&str>,
        job_name: Option<&str>,
        metadata: Option<&Map<String, Value>>,
    ) -> Map<String, Value> {
        let mut results: Map<String, Value> = Map::new();

        for target in targets {
            let result: Result<Value, String> = if target.platform == Platform::Local {
                self.deliver_local(content, job_id, job_name, metadata)
            } else {
                self.deliver_to_platform(target, content, metadata)
            };

            let entry = match result {
                Ok(v) => TargetResult::Success(v),
                Err(e) => TargetResult::Error(e),
            };
            // Python: results[target.to_string()] = {...}. Later targets with
            // identical keys overwrite earlier ones (dict semantics).
            results.insert(target.to_string(), entry.to_json());
        }

        results
    }

    /// Save content to local files.
    fn deliver_local(
        &self,
        content: &str,
        job_id: Option<&str>,
        job_name: Option<&str>,
        metadata: Option<&Map<String, Value>>,
    ) -> Result<Value, String> {
        let now = Local::now();
        let timestamp = now.format("%Y%m%d_%H%M%S").to_string();

        let output_path: PathBuf = match job_id {
            Some(jid) => self.output_dir.join(jid).join(format!("{timestamp}.md")),
            None => self.output_dir.join("misc").join(format!("{timestamp}.md")),
        };

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        // Build the output document.
        let mut lines: Vec<String> = Vec::new();
        match job_name {
            Some(name) => lines.push(format!("# {name}")),
            None => lines.push("# Delivery Output".to_string()),
        }

        lines.push(String::new());
        lines.push(format!(
            "**Timestamp:** {}",
            now.format("%Y-%m-%d %H:%M:%S")
        ));

        if let Some(jid) = job_id {
            lines.push(format!("**Job ID:** {jid}"));
        }

        if let Some(meta) = metadata {
            for (key, value) in meta.iter() {
                lines.push(format!("**{key}:** {}", json_value_to_py_str(value)));
            }
        }

        lines.push(String::new());
        lines.push("---".to_string());
        lines.push(String::new());
        lines.push(content.to_string());

        std::fs::write(&output_path, lines.join("\n")).map_err(|e| e.to_string())?;

        let mut m = Map::new();
        m.insert(
            "path".into(),
            Value::String(output_path.to_string_lossy().to_string()),
        );
        m.insert("timestamp".into(), Value::String(timestamp));
        Ok(Value::Object(m))
    }

    /// Save full cron output to disk and return the file path.
    fn save_full_output(&self, content: &str, job_id: &str) -> Result<PathBuf, String> {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let out_dir = get_hermes_home().join("cron").join("output");
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let path = out_dir.join(format!("{job_id}_{timestamp}.txt"));
        std::fs::write(&path, content).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Deliver content to a messaging platform.
    fn deliver_to_platform(
        &self,
        target: &DeliveryTarget,
        content: &str,
        metadata: Option<&Map<String, Value>>,
    ) -> Result<Value, String> {
        let adapter = self
            .adapters
            .get(&target.platform)
            .ok_or_else(|| format!("No adapter configured for {}", target.platform.value()))?;

        let chat_id = match &target.chat_id {
            Some(c) if !c.is_empty() => c.clone(),
            _ => {
                return Err(format!(
                    "No chat ID for {} delivery",
                    target.platform.value()
                ))
            }
        };

        // Guard: truncate oversized cron output to stay within platform limits.
        // Python measures len(content) in characters (Unicode code points).
        let mut content_owned = content.to_string();
        if content.chars().count() > MAX_PLATFORM_OUTPUT {
            let job_id = metadata
                .and_then(|m| m.get("job_id"))
                .map(json_value_to_py_str)
                .unwrap_or_else(|| "unknown".to_string());
            let saved_path = self.save_full_output(content, &job_id)?;
            log::info!(
                "Cron output truncated ({} chars) - full output: {}",
                content.chars().count(),
                saved_path.display()
            );
            let visible: String = content.chars().take(TRUNCATED_VISIBLE).collect();
            content_owned = format!(
                "{visible}\n\n... [truncated, full output saved to {}]",
                saved_path.display()
            );
        }

        // send_metadata = dict(metadata or {})
        let mut send_metadata: Map<String, Value> = metadata.cloned().unwrap_or_default();
        if let Some(thread_id) = &target.thread_id {
            if !send_metadata.contains_key("thread_id") {
                send_metadata.insert("thread_id".into(), Value::String(thread_id.clone()));
            }
        }

        // metadata=send_metadata or None  ->  None when empty.
        let meta_arg: Option<&Map<String, Value>> = if send_metadata.is_empty() {
            None
        } else {
            Some(&send_metadata)
        };

        adapter.send(&chat_id, &content_owned, meta_arg)
    }
}

/// Render a JSON value the way Python `str(value)` would inside an f-string,
/// used for the `**key:** value` metadata lines and the `job_id` lookup.
fn json_value_to_py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

/// Convenience helper: parse a list of target strings.
pub fn parse_targets(targets: &[&str], origin: Option<&SessionSource>) -> Vec<DeliveryTarget> {
    targets
        .iter()
        .map(|t| DeliveryTarget::parse(t, origin))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gw_session::SessionSource;

    #[test]
    fn parse_local_and_origin() {
        let t = DeliveryTarget::parse("local", None);
        assert_eq!(t.platform, Platform::Local);
        assert!(!t.is_origin);
        assert_eq!(t.to_string(), "local");

        // origin with no source -> local + is_origin
        let t = DeliveryTarget::parse("ORIGIN", None);
        assert_eq!(t.platform, Platform::Local);
        assert!(t.is_origin);
        assert_eq!(t.to_string(), "origin");
    }

    #[test]
    fn parse_origin_with_source() {
        let mut src = SessionSource::new(Platform::Telegram, "123");
        src.thread_id = Some("7".to_string());
        let t = DeliveryTarget::parse("origin", Some(&src));
        assert_eq!(t.platform, Platform::Telegram);
        assert_eq!(t.chat_id.as_deref(), Some("123"));
        assert_eq!(t.thread_id.as_deref(), Some("7"));
        assert!(t.is_origin);
        // is_origin wins in to_string
        assert_eq!(t.to_string(), "origin");
    }

    #[test]
    fn parse_bare_platform_home_channel() {
        let t = DeliveryTarget::parse("telegram", None);
        assert_eq!(t.platform, Platform::Telegram);
        assert!(t.chat_id.is_none());
        assert!(!t.is_explicit);
        assert_eq!(t.to_string(), "telegram");

        // Unknown platform -> local (Platform(value) ValueError path)
        let t = DeliveryTarget::parse("not_a_platform", None);
        assert_eq!(t.platform, Platform::Local);
    }

    #[test]
    fn parse_explicit_telegram_chat() {
        let t = DeliveryTarget::parse("telegram:123456", None);
        assert_eq!(t.platform, Platform::Telegram);
        assert_eq!(t.chat_id.as_deref(), Some("123456"));
        assert!(t.thread_id.is_none());
        assert!(t.is_explicit);
        assert_eq!(t.to_string(), "telegram:123456");
    }

    #[test]
    fn parse_explicit_telegram_topic() {
        let t = DeliveryTarget::parse("telegram:-1001234:55", None);
        assert_eq!(t.platform, Platform::Telegram);
        assert_eq!(t.chat_id.as_deref(), Some("-1001234"));
        assert_eq!(t.thread_id.as_deref(), Some("55"));
        assert!(t.is_explicit);
        assert_eq!(t.to_string(), "telegram:-1001234:55");
    }

    #[test]
    fn parse_non_explicit_colon_passthrough() {
        // Matrix alias with ':' but not explicit -> whole ref kept as chat_id.
        let t = DeliveryTarget::parse("matrix:#room:example.org", None);
        assert_eq!(t.platform, Platform::Matrix);
        assert_eq!(t.chat_id.as_deref(), Some("#room:example.org"));
        assert!(t.thread_id.is_none());
        assert!(t.is_explicit);
    }

    #[test]
    fn parse_unknown_platform_with_colon() {
        let t = DeliveryTarget::parse("bogus:abc", None);
        assert_eq!(t.platform, Platform::Local);
    }

    #[test]
    fn target_ref_phone_e164() {
        let (chat, thread, explicit) = parse_target_ref("signal", "+15551234567");
        assert_eq!(chat.as_deref(), Some("+15551234567"));
        assert!(thread.is_none());
        assert!(explicit);
    }

    #[test]
    fn target_ref_yuanbao_digits() {
        let (chat, _t, explicit) = parse_target_ref("yuanbao", "42");
        assert_eq!(chat.as_deref(), Some("group:42"));
        assert!(explicit);

        let (chat, _t, explicit) = parse_target_ref("yuanbao", "notdigits");
        assert!(chat.is_none());
        assert!(!explicit);
    }

    #[test]
    fn target_ref_matrix_userid() {
        let (chat, _t, explicit) = parse_target_ref("matrix", "@alice:example.org");
        assert_eq!(chat.as_deref(), Some("@alice:example.org"));
        assert!(explicit);
    }

    #[test]
    fn target_ref_bare_digits() {
        let (chat, _t, explicit) = parse_target_ref("discord", "9988");
        assert_eq!(chat.as_deref(), Some("9988"));
        assert!(explicit);
    }

    struct EchoAdapter;
    impl DeliveryAdapter for EchoAdapter {
        fn send(
            &self,
            chat_id: &str,
            content: &str,
            metadata: Option<&Map<String, Value>>,
        ) -> Result<Value, String> {
            let mut m = Map::new();
            m.insert("chat_id".into(), Value::String(chat_id.to_string()));
            m.insert("content_len".into(), Value::from(content.chars().count()));
            m.insert(
                "had_metadata".into(),
                Value::Bool(metadata.is_some()),
            );
            if let Some(meta) = metadata {
                if let Some(t) = meta.get("thread_id") {
                    m.insert("thread_id".into(), t.clone());
                }
            }
            Ok(Value::Object(m))
        }
    }

    #[test]
    fn deliver_to_platform_thread_injected() {
        let mut adapters: BTreeMap<Platform, Box<dyn DeliveryAdapter>> = BTreeMap::new();
        adapters.insert(Platform::Telegram, Box::new(EchoAdapter));
        let router = DeliveryRouter::new(GatewayConfig::default(), adapters);

        let target = DeliveryTarget {
            platform: Platform::Telegram,
            chat_id: Some("123".into()),
            thread_id: Some("9".into()),
            is_origin: false,
            is_explicit: true,
        };
        let results = router.deliver("hello", &[target], None, None, None);
        let entry = results.get("telegram:123:9").unwrap();
        assert_eq!(entry.get("success"), Some(&Value::Bool(true)));
        let result = entry.get("result").unwrap();
        assert_eq!(
            result.get("thread_id"),
            Some(&Value::String("9".to_string()))
        );
        assert_eq!(result.get("had_metadata"), Some(&Value::Bool(true)));
    }

    #[test]
    fn deliver_missing_adapter_errors() {
        let router = DeliveryRouter::new(GatewayConfig::default(), BTreeMap::new());
        let target = DeliveryTarget {
            platform: Platform::Telegram,
            chat_id: Some("123".into()),
            thread_id: None,
            is_origin: false,
            is_explicit: true,
        };
        let results = router.deliver("hi", &[target], None, None, None);
        let entry = results.get("telegram:123").unwrap();
        assert_eq!(entry.get("success"), Some(&Value::Bool(false)));
        assert_eq!(
            entry.get("error"),
            Some(&Value::String(
                "No adapter configured for telegram".to_string()
            ))
        );
    }

    #[test]
    fn deliver_no_chat_id_errors() {
        let mut adapters: BTreeMap<Platform, Box<dyn DeliveryAdapter>> = BTreeMap::new();
        adapters.insert(Platform::Telegram, Box::new(EchoAdapter));
        let router = DeliveryRouter::new(GatewayConfig::default(), adapters);
        let target = DeliveryTarget {
            platform: Platform::Telegram,
            chat_id: None,
            thread_id: None,
            is_origin: false,
            is_explicit: false,
        };
        let results = router.deliver("hi", &[target], None, None, None);
        // Bare platform -> to_string() == "telegram"
        let entry = results.get("telegram").unwrap();
        assert_eq!(entry.get("success"), Some(&Value::Bool(false)));
        assert_eq!(
            entry.get("error"),
            Some(&Value::String("No chat ID for telegram delivery".to_string()))
        );
    }

    #[test]
    fn deliver_local_writes_file() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes_gw_delivery_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let router = DeliveryRouter::new(GatewayConfig::default(), BTreeMap::new());

        let mut meta = Map::new();
        meta.insert("source".into(), Value::String("test".into()));
        let results = router.deliver(
            "the body content",
            &[DeliveryTarget::new(Platform::Local)],
            Some("job42"),
            Some("My Job"),
            Some(&meta),
        );
        let entry = results.get("local").unwrap();
        assert_eq!(entry.get("success"), Some(&Value::Bool(true)));
        let path = entry
            .get("result")
            .and_then(|r| r.get("path"))
            .and_then(|p| p.as_str())
            .unwrap();
        let written = std::fs::read_to_string(path).unwrap();
        assert!(written.starts_with("# My Job\n"));
        assert!(written.contains("**Job ID:** job42"));
        assert!(written.contains("**source:** test"));
        assert!(written.contains("the body content"));

        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }

    #[test]
    fn truncation_spills_to_disk() {
        let tmp = std::env::temp_dir().join(format!(
            "hermes_gw_delivery_trunc_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::set_var("HERMES_HOME", &tmp);
        }
        let mut adapters: BTreeMap<Platform, Box<dyn DeliveryAdapter>> = BTreeMap::new();
        adapters.insert(Platform::Telegram, Box::new(EchoAdapter));
        let router = DeliveryRouter::new(GatewayConfig::default(), adapters);

        let big = "x".repeat(MAX_PLATFORM_OUTPUT + 100);
        let target = DeliveryTarget {
            platform: Platform::Telegram,
            chat_id: Some("123".into()),
            thread_id: None,
            is_origin: false,
            is_explicit: true,
        };
        let mut meta = Map::new();
        meta.insert("job_id".into(), Value::String("bigjob".into()));
        let results = router.deliver(&big, &[target], None, None, Some(&meta));
        let entry = results.get("telegram:123").unwrap();
        let result = entry.get("result").unwrap();
        let sent_len = result.get("content_len").and_then(|v| v.as_u64()).unwrap() as usize;
        // Visible chars + truncation suffix, far below the original size.
        assert!(sent_len < MAX_PLATFORM_OUTPUT + 100);
        assert!(sent_len >= TRUNCATED_VISIBLE);

        let _ = std::fs::remove_dir_all(&tmp);
        unsafe {
            std::env::remove_var("HERMES_HOME");
        }
    }
}
