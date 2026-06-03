//! Home Assistant platform adapter — native Rust port of
//! `gateway/platforms/homeassistant.py`.
//!
//! The Python module connects to the Home Assistant WebSocket API for
//! real-time event monitoring. `state_changed` events are converted into
//! `MessageEvent` objects and forwarded to the agent; outbound messages are
//! delivered as HA persistent notifications via the REST API.
//!
//! The live WebSocket lifecycle (`aiohttp` sessions, `asyncio.Task` listen
//! loop, automatic-reconnect backoff) is intimately tied to the CPython event
//! loop. This port reproduces the **deterministic, behaviour-defining logic**
//! that the loop drives, modelled with plain Rust data structures and
//! synchronous helpers:
//!
//! - [`check_ha_requirements`] — env precondition gate.
//! - [`HomeAssistantConfig`] — `__init__` extra/env parsing (URL/token,
//!   watch filters, cooldown).
//! - WebSocket handshake message construction:
//!   [`build_auth_message`], [`build_subscribe_message`], plus
//!   [`ws_url_from_http`] and the [`HandshakeStep`] verification helpers.
//! - Event routing: [`HomeAssistantAdapter::should_forward`] (ignore / watch /
//!   watch_all filtering + per-entity cooldown).
//! - [`format_state_change`] — domain-specific human-readable descriptions.
//! - [`HomeAssistantAdapter::build_event_message`] — the full
//!   `_handle_ha_event` pipeline producing a [`MessageEvent`].
//! - Outbound send: [`build_notification_payload`], [`notification_url`],
//!   [`notification_headers`], and [`send_notification_blocking`]
//!   (`reqwest::blocking`, preserving request/response shapes).
//! - Reconnection backoff schedule: [`backoff_delay`].
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base::MessageEvent`]
//!   - [`crate::gw_platforms_base::MessageType`]
//!   - [`crate::gw_platforms_base::SessionSource`]
//!   - [`crate::gw_platforms_base::SendResult`]

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use crate::gw_platforms_base::{MessageEvent, MessageType, SendResult, SessionSource};

// ===========================================================================
// Constants
// ===========================================================================

/// HA persistent notifications truncate the message to this many characters.
pub const MAX_MESSAGE_LENGTH: usize = 4096;

/// Default HA URL when neither `extra.url` nor `HASS_URL` is set.
pub const DEFAULT_HASS_URL: &str = "http://homeassistant.local:8123";

/// Default per-entity cooldown (seconds) between forwarded events.
pub const DEFAULT_COOLDOWN_SECONDS: i64 = 30;

/// Reconnection backoff schedule (seconds). Mirrors `_BACKOFF_STEPS`.
pub const BACKOFF_STEPS: &[u64] = &[5, 10, 30, 60];

// ===========================================================================
// Requirements gate
// ===========================================================================

/// Check whether HA dependencies are available and configured.
///
/// Mirrors `check_ha_requirements`. In the Python module this also gates on
/// `aiohttp` being importable; in the native port the transport availability is
/// assumed (handled by the runtime layer), so this checks only that the
/// `HASS_TOKEN` env var is present and non-empty.
pub fn check_ha_requirements() -> bool {
    std::env::var("HASS_TOKEN")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

// ===========================================================================
// Configuration (mirrors __init__ extra/env parsing)
// ===========================================================================

/// Parsed Home Assistant adapter configuration.
///
/// Reproduces the field derivation done in `HomeAssistantAdapter.__init__`:
/// the resolved URL (trailing slash stripped), token, watch filters, the
/// `watch_all` flag, and the cooldown window.
#[derive(Debug, Clone)]
pub struct HomeAssistantConfig {
    pub hass_url: String,
    pub hass_token: String,
    pub watch_domains: HashSet<String>,
    pub watch_entities: HashSet<String>,
    pub ignore_entities: HashSet<String>,
    pub watch_all: bool,
    pub cooldown_seconds: i64,
}

impl HomeAssistantConfig {
    /// Build config from the adapter's `config.token`, `config.extra` (as JSON)
    /// and the process environment, mirroring `__init__`.
    ///
    /// Resolution order matches Python:
    ///   - token: `config.token` or `HASS_TOKEN` (default "")
    ///   - url: `extra["url"]` or `HASS_URL` (default `DEFAULT_HASS_URL`),
    ///     then `.rstrip("/")`
    pub fn from_parts(config_token: Option<&str>, extra: &Value) -> Self {
        let extra_obj = extra.as_object();

        let token = match config_token {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => std::env::var("HASS_TOKEN").unwrap_or_default(),
        };

        // url: extra.get("url") OR env HASS_URL OR default. Python's `or`
        // treats an empty string as falsy, so an empty extra["url"] falls
        // through to the env/default.
        let extra_url = extra_obj
            .and_then(|o| o.get("url"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let url = match extra_url {
            Some(u) => u.to_string(),
            None => std::env::var("HASS_URL").unwrap_or_else(|_| DEFAULT_HASS_URL.to_string()),
        };
        let hass_url = url.trim_end_matches('/').to_string();

        // token = config.token OR env HASS_TOKEN OR "" (Python treats empty as falsy).
        let extra_token = extra_obj
            .and_then(|o| o.get("token"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let hass_token = match extra_token {
            Some(t) => t.to_string(),
            None => std::env::var("HASS_TOKEN").unwrap_or_default(),
        };

        let watch_domains = string_set(extra_obj, "watch_domains");
        let watch_entities = string_set(extra_obj, "watch_entities");
        let ignore_entities = string_set(extra_obj, "ignore_entities");

        let watch_all = extra_obj
            .and_then(|o| o.get("watch_all"))
            .map(json_truthy)
            .unwrap_or(false);

        let cooldown_seconds = extra_obj
            .and_then(|o| o.get("cooldown_seconds"))
            .and_then(json_to_i64)
            .unwrap_or(DEFAULT_COOLDOWN_SECONDS);

        HomeAssistantConfig {
            hass_url,
            hass_token,
            watch_domains,
            watch_entities,
            ignore_entities,
            watch_all,
            cooldown_seconds,
        }
    }

    /// Return True when no filters are configured (mirrors the connect-time
    /// warning condition: no watch_domains, watch_entities, or watch_all).
    pub fn has_no_filters(&self) -> bool {
        self.watch_domains.is_empty() && self.watch_entities.is_empty() && !self.watch_all
    }
}

/// Convert a JSON array (or any iterable of strings) under `key` into a set of
/// strings. Non-string elements are stringified via `Value::to_string` only
/// when they are scalars; objects/arrays are skipped (HA configs only ever put
/// strings here).
fn string_set(extra: Option<&serde_json::Map<String, Value>>, key: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    if let Some(arr) = extra.and_then(|o| o.get(key)).and_then(|v| v.as_array()) {
        for item in arr {
            match item {
                Value::String(s) => {
                    out.insert(s.clone());
                }
                Value::Number(n) => {
                    out.insert(n.to_string());
                }
                Value::Bool(b) => {
                    out.insert(if *b { "True".into() } else { "False".into() });
                }
                _ => {}
            }
        }
    }
    out
}

/// Python `bool(value)` truthiness for the `watch_all` flag. Accepts JSON
/// bools, numbers (0 -> false), strings (empty -> false), and arrays/objects
/// (empty -> false).
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

/// Python `int(value)` for the cooldown: ints/floats truncate toward zero,
/// numeric strings parse. Mirrors `int(extra.get("cooldown_seconds", 30))`.
fn json_to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f.trunc() as i64)
            }
        }
        Value::String(s) => {
            let t = s.trim();
            if let Ok(i) = t.parse::<i64>() {
                Some(i)
            } else {
                t.parse::<f64>().ok().map(|f| f.trunc() as i64)
            }
        }
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

// ===========================================================================
// WebSocket handshake construction
// ===========================================================================

/// Derive the WebSocket URL from the HTTP(S) base URL. Mirrors `_ws_connect`.
///
/// `https://` -> `wss://`, `http://` -> `ws://`, then `/api/websocket` appended.
pub fn ws_url_from_http(hass_url: &str) -> String {
    let ws = hass_url
        .replace("https://", "wss://")
        .replace("http://", "ws://");
    format!("{ws}/api/websocket")
}

/// Build the auth message sent after receiving `auth_required`.
pub fn build_auth_message(token: &str) -> Value {
    json!({
        "type": "auth",
        "access_token": token,
    })
}

/// Build the `subscribe_events` message for `state_changed`, given the next id.
pub fn build_subscribe_message(msg_id: i64) -> Value {
    json!({
        "id": msg_id,
        "type": "subscribe_events",
        "event_type": "state_changed",
    })
}

/// The verifiable steps of the HA WebSocket handshake. Each variant validates a
/// received frame the way `_ws_connect` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeStep {
    /// Expect `{"type": "auth_required"}`.
    AuthRequired,
    /// Expect `{"type": "auth_ok"}`.
    AuthOk,
    /// Expect `{"success": true}`.
    SubscribeAck,
}

impl HandshakeStep {
    /// Validate a received JSON frame for this handshake step. Mirrors the
    /// per-step checks in `_ws_connect`.
    pub fn validate(&self, msg: &Value) -> bool {
        match self {
            HandshakeStep::AuthRequired => {
                msg.get("type").and_then(|v| v.as_str()) == Some("auth_required")
            }
            HandshakeStep::AuthOk => {
                msg.get("type").and_then(|v| v.as_str()) == Some("auth_ok")
            }
            HandshakeStep::SubscribeAck => {
                msg.get("success").map(json_truthy).unwrap_or(false)
            }
        }
    }
}

/// Compute the reconnect delay (seconds) for a given backoff index. Mirrors
/// `_BACKOFF_STEPS[min(idx, len-1)]`.
pub fn backoff_delay(backoff_idx: usize) -> u64 {
    let last = BACKOFF_STEPS.len() - 1;
    BACKOFF_STEPS[backoff_idx.min(last)]
}

// ===========================================================================
// State-change formatting
// ===========================================================================

/// Extract a JSON string field with a default, mirroring `dict.get(k, default)`
/// when the value is a string.
fn str_field<'a>(obj: &'a Value, key: &str, default: &'a str) -> &'a str {
    obj.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Render an attribute value the way Python's f-string `str()` would for the
/// common HA cases (strings unquoted, numbers/bools stringified). Used for
/// `current_temperature`, `temperature`, etc.
fn attr_display(value: Option<&Value>, default: &str) -> String {
    match value {
        None | Some(Value::Null) => default.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

/// Convert a `state_changed` event into a human-readable description.
///
/// Mirrors `_format_state_change`. Returns `None` when there is no new state or
/// the state did not actually change.
pub fn format_state_change(entity_id: &str, old_state: &Value, new_state: &Value) -> Option<String> {
    // `if not new_state` -> falsy (None / empty object) returns None.
    if !is_truthy_obj(new_state) {
        return None;
    }

    // old_state may be falsy (None / {}); Python uses "unknown" in that case.
    let old_val = if is_truthy_obj(old_state) {
        str_field(old_state, "state", "unknown").to_string()
    } else {
        "unknown".to_string()
    };
    let new_val = str_field(new_state, "state", "unknown").to_string();

    if old_val == new_val {
        return None;
    }

    let attrs = new_state.get("attributes").cloned().unwrap_or(Value::Null);
    let friendly_name = attrs
        .get("friendly_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| entity_id.to_string());

    let domain = if entity_id.contains('.') {
        entity_id.split('.').next().unwrap_or("")
    } else {
        ""
    };

    match domain {
        "climate" => {
            let temp = attr_display(attrs.get("current_temperature"), "?");
            let target = attr_display(attrs.get("temperature"), "?");
            Some(format!(
                "[Home Assistant] {friendly_name}: HVAC mode changed from \
'{old_val}' to '{new_val}' (current: {temp}, target: {target})"
            ))
        }
        "sensor" => {
            let unit = attrs
                .get("unit_of_measurement")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Some(format!(
                "[Home Assistant] {friendly_name}: changed from \
{old_val}{unit} to {new_val}{unit}"
            ))
        }
        "binary_sensor" => {
            let new_label = if new_val == "on" { "triggered" } else { "cleared" };
            let old_label = if old_val == "on" { "triggered" } else { "cleared" };
            Some(format!(
                "[Home Assistant] {friendly_name}: {new_label} (was {old_label})"
            ))
        }
        "light" | "switch" | "fan" => {
            let label = if new_val == "on" { "on" } else { "off" };
            Some(format!("[Home Assistant] {friendly_name}: turned {label}"))
        }
        "alarm_control_panel" => Some(format!(
            "[Home Assistant] {friendly_name}: alarm state changed from \
'{old_val}' to '{new_val}'"
        )),
        _ => Some(format!(
            "[Home Assistant] {friendly_name} ({entity_id}): \
changed from '{old_val}' to '{new_val}'"
        )),
    }
}

/// Python truthiness for an HA state dict: `None`/missing and empty objects are
/// falsy; a non-empty object is truthy.
fn is_truthy_obj(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Object(o) => !o.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
    }
}

// ===========================================================================
// Outbound notification (REST)
// ===========================================================================

/// REST endpoint for persistent-notification creation. Mirrors the `url` in
/// `send`.
pub fn notification_url(hass_url: &str) -> String {
    format!("{hass_url}/api/services/persistent_notification/create")
}

/// Headers for the notification POST. Mirrors `send`.
pub fn notification_headers(token: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".to_string(), format!("Bearer {token}")),
        ("Content-Type".to_string(), "application/json".to_string()),
    ]
}

/// Build the notification JSON payload, truncating the message to
/// [`MAX_MESSAGE_LENGTH`]. Mirrors `send`.
///
/// Python slices `content[:MAX_MESSAGE_LENGTH]` by codepoints, so we take the
/// first `MAX_MESSAGE_LENGTH` `char`s.
pub fn build_notification_payload(content: &str) -> Value {
    let message: String = content.chars().take(MAX_MESSAGE_LENGTH).collect();
    json!({
        "title": "Hermes Agent",
        "message": message,
    })
}

/// Send a persistent notification via the HA REST API, blocking.
///
/// Mirrors `HomeAssistantAdapter.send`: POSTs to
/// `{hass_url}/api/services/persistent_notification/create` with a bearer
/// token, a 10-second timeout, and classifies the response — `< 300` is a
/// success (with a random 12-hex message id), otherwise an `HTTP {status}:
/// {body}` error. Network/timeout failures become `SendResult::fail`.
pub fn send_notification_blocking(hass_url: &str, token: &str, content: &str) -> SendResult {
    let url = notification_url(hass_url);
    let payload = build_notification_payload(content);

    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => return SendResult::fail(e.to_string()),
    };

    let mut req = client.post(&url).json(&payload);
    for (k, v) in notification_headers(token) {
        req = req.header(k, v);
    }

    match req.send() {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if status < 300 {
                SendResult::ok(Some(random_msg_id()))
            } else {
                let body = resp.text().unwrap_or_default();
                SendResult::fail(format!("HTTP {status}: {body}"))
            }
        }
        Err(e) => {
            if e.is_timeout() {
                SendResult::fail("Timeout sending notification to HA")
            } else {
                SendResult::fail(e.to_string())
            }
        }
    }
}

/// Generate a 12-hex-char id matching `uuid.uuid4().hex[:12]`.
fn random_msg_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    format!("{:012x}", mixed & 0xFFFF_FFFF_FFFF)
}

// ===========================================================================
// Adapter (config + cooldown + event routing)
// ===========================================================================

/// Home Assistant adapter state: parsed config plus the per-entity cooldown
/// tracker. Models the behaviour-defining parts of `HomeAssistantAdapter`.
#[derive(Debug, Clone)]
pub struct HomeAssistantAdapter {
    pub config: HomeAssistantConfig,
    /// entity_id -> last forwarded-event timestamp (seconds). Mirrors
    /// `_last_event_time`.
    last_event_time: HashMap<String, f64>,
    /// Monotonic-ish WebSocket message id counter. Mirrors `_msg_id`.
    msg_id: i64,
}

impl HomeAssistantAdapter {
    pub fn new(config: HomeAssistantConfig) -> Self {
        HomeAssistantAdapter {
            config,
            last_event_time: HashMap::new(),
            msg_id: 0,
        }
    }

    /// Convenience constructor from raw parts (mirrors `__init__`).
    pub fn from_parts(config_token: Option<&str>, extra: &Value) -> Self {
        Self::new(HomeAssistantConfig::from_parts(config_token, extra))
    }

    /// Return the next WebSocket message id. Mirrors `_next_id`.
    pub fn next_id(&mut self) -> i64 {
        self.msg_id += 1;
        self.msg_id
    }

    /// Decide whether a `state_changed` event for `entity_id` should be
    /// forwarded, applying the ignore filter, the domain/entity watch filters
    /// (closed by default), and the per-entity cooldown.
    ///
    /// Mirrors the filtering portion of `_handle_ha_event`. `now` is the
    /// current time in seconds. On a forwarded event the entity's last-event
    /// time is updated, exactly as the Python code does.
    pub fn should_forward(&mut self, entity_id: &str, now: f64) -> bool {
        if entity_id.is_empty() {
            return false;
        }

        // Ignore filter.
        if self.config.ignore_entities.contains(entity_id) {
            return false;
        }

        // Domain/entity watch filters (closed by default).
        let domain = if entity_id.contains('.') {
            entity_id.split('.').next().unwrap_or("")
        } else {
            ""
        };
        if !self.config.watch_domains.is_empty() || !self.config.watch_entities.is_empty() {
            let domain_match = if !self.config.watch_domains.is_empty() {
                self.config.watch_domains.contains(domain)
            } else {
                false
            };
            let entity_match = if !self.config.watch_entities.is_empty() {
                self.config.watch_entities.contains(entity_id)
            } else {
                false
            };
            if !domain_match && !entity_match {
                return false;
            }
        } else if !self.config.watch_all {
            return false;
        }

        // Cooldown.
        let last = self.last_event_time.get(entity_id).copied().unwrap_or(0.0);
        if (now - last) < self.config.cooldown_seconds as f64 {
            return false;
        }
        self.last_event_time.insert(entity_id.to_string(), now);

        true
    }

    /// Run the full `_handle_ha_event` pipeline for an HA `event` payload (the
    /// dict under the WebSocket message's `"event"` key) and return the
    /// resulting [`MessageEvent`] when it should be forwarded.
    ///
    /// Returns `None` when the event is filtered out (missing entity_id,
    /// ignored, not watched, within cooldown) or produces no formatted message.
    ///
    /// `now_secs` is the integer second timestamp used both for cooldown and
    /// the `ha_{entity}_{int(now)}` message id. `timestamp_iso` is the value
    /// to record on the event (`datetime.now()` in Python); the caller supplies
    /// it so this stays side-effect-free.
    pub fn build_event_message(&mut self, event: &Value, now_secs: f64) -> Option<MessageEvent> {
        let event_data = event.get("data").cloned().unwrap_or(Value::Null);
        let entity_id = event_data
            .get("entity_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !self.should_forward(&entity_id, now_secs) {
            return None;
        }

        let old_state = event_data.get("old_state").cloned().unwrap_or(Value::Null);
        let new_state = event_data.get("new_state").cloned().unwrap_or(Value::Null);
        let message = format_state_change(&entity_id, &old_state, &new_state)?;

        let source = SessionSource {
            platform: "homeassistant".to_string(),
            chat_id: "ha_events".to_string(),
            chat_name: Some("Home Assistant Events".to_string()),
            chat_type: "channel".to_string(),
            user_id: Some("homeassistant".to_string()),
            user_name: Some("Home Assistant".to_string()),
            ..Default::default()
        };

        let message_id = format!("ha_{entity_id}_{}", now_secs as i64);

        Some(MessageEvent {
            text: message,
            message_type: MessageType::Text,
            source,
            message_id: Some(message_id),
            ..Default::default()
        })
    }

    /// Basic info about the HA event channel. Mirrors `get_chat_info`.
    pub fn get_chat_info(&self) -> Value {
        json!({
            "name": "Home Assistant Events",
            "type": "channel",
            "url": self.config.hass_url,
        })
    }

    /// Send a notification via the HA REST API. Mirrors `send`.
    pub fn send(&self, content: &str) -> SendResult {
        send_notification_blocking(&self.config.hass_url, &self.config.hass_token, content)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(extra: Value) -> HomeAssistantConfig {
        HomeAssistantConfig::from_parts(Some("tok"), &extra)
    }

    #[test]
    fn config_url_rstrip_and_token() {
        let c = cfg(json!({"url": "http://ha.local:8123/"}));
        assert_eq!(c.hass_url, "http://ha.local:8123");
        assert_eq!(c.hass_token, "tok");
    }

    #[test]
    fn config_default_url_when_empty() {
        let c = HomeAssistantConfig::from_parts(Some("t"), &json!({"url": ""}));
        // empty extra url is falsy; env unset -> default.
        assert_eq!(c.hass_url, DEFAULT_HASS_URL);
    }

    #[test]
    fn config_watch_filters_and_cooldown() {
        let c = cfg(json!({
            "watch_domains": ["light", "switch"],
            "watch_entities": ["sensor.temp"],
            "ignore_entities": ["light.x"],
            "watch_all": true,
            "cooldown_seconds": 5,
        }));
        assert!(c.watch_domains.contains("light"));
        assert!(c.watch_domains.contains("switch"));
        assert!(c.watch_entities.contains("sensor.temp"));
        assert!(c.ignore_entities.contains("light.x"));
        assert!(c.watch_all);
        assert_eq!(c.cooldown_seconds, 5);
        assert!(!c.has_no_filters());
    }

    #[test]
    fn config_no_filters_warning_condition() {
        let c = cfg(json!({}));
        assert!(c.has_no_filters());
        assert_eq!(c.cooldown_seconds, DEFAULT_COOLDOWN_SECONDS);
    }

    #[test]
    fn cooldown_string_parses() {
        let c = cfg(json!({"cooldown_seconds": "15"}));
        assert_eq!(c.cooldown_seconds, 15);
        let c2 = cfg(json!({"cooldown_seconds": 12.9}));
        assert_eq!(c2.cooldown_seconds, 12);
    }

    #[test]
    fn ws_url_scheme_swap() {
        assert_eq!(
            ws_url_from_http("https://ha.local:8123"),
            "wss://ha.local:8123/api/websocket"
        );
        assert_eq!(
            ws_url_from_http("http://ha.local:8123"),
            "ws://ha.local:8123/api/websocket"
        );
    }

    #[test]
    fn handshake_messages() {
        assert_eq!(
            build_auth_message("abc"),
            json!({"type": "auth", "access_token": "abc"})
        );
        assert_eq!(
            build_subscribe_message(3),
            json!({"id": 3, "type": "subscribe_events", "event_type": "state_changed"})
        );
    }

    #[test]
    fn handshake_validation() {
        assert!(HandshakeStep::AuthRequired.validate(&json!({"type": "auth_required"})));
        assert!(!HandshakeStep::AuthRequired.validate(&json!({"type": "result"})));
        assert!(HandshakeStep::AuthOk.validate(&json!({"type": "auth_ok"})));
        assert!(!HandshakeStep::AuthOk.validate(&json!({"type": "auth_invalid"})));
        assert!(HandshakeStep::SubscribeAck.validate(&json!({"success": true, "id": 1})));
        assert!(!HandshakeStep::SubscribeAck.validate(&json!({"success": false})));
        assert!(!HandshakeStep::SubscribeAck.validate(&json!({"id": 1})));
    }

    #[test]
    fn backoff_schedule() {
        assert_eq!(backoff_delay(0), 5);
        assert_eq!(backoff_delay(1), 10);
        assert_eq!(backoff_delay(2), 30);
        assert_eq!(backoff_delay(3), 60);
        // clamps to last step.
        assert_eq!(backoff_delay(99), 60);
    }

    #[test]
    fn next_id_increments() {
        let mut a = HomeAssistantAdapter::from_parts(Some("t"), &json!({}));
        assert_eq!(a.next_id(), 1);
        assert_eq!(a.next_id(), 2);
    }

    #[test]
    fn format_no_new_state_returns_none() {
        assert_eq!(format_state_change("light.a", &json!({}), &Value::Null), None);
        assert_eq!(format_state_change("light.a", &json!({}), &json!({})), None);
    }

    #[test]
    fn format_no_change_returns_none() {
        let old = json!({"state": "on"});
        let new = json!({"state": "on"});
        assert_eq!(format_state_change("light.a", &old, &new), None);
    }

    #[test]
    fn format_light_on_off() {
        let old = json!({"state": "off"});
        let new = json!({"state": "on", "attributes": {"friendly_name": "Lamp"}});
        let out = format_state_change("light.lamp", &old, &new).unwrap();
        assert_eq!(out, "[Home Assistant] Lamp: turned on");

        let new_off = json!({"state": "off"});
        let old_on = json!({"state": "on"});
        let out2 = format_state_change("switch.plug", &old_on, &new_off).unwrap();
        assert_eq!(out2, "[Home Assistant] switch.plug: turned off");
    }

    #[test]
    fn format_sensor_with_unit() {
        let old = json!({"state": "20"});
        let new = json!({
            "state": "21",
            "attributes": {"friendly_name": "Temp", "unit_of_measurement": "°C"}
        });
        let out = format_state_change("sensor.temp", &old, &new).unwrap();
        assert_eq!(out, "[Home Assistant] Temp: changed from 20°C to 21°C");
    }

    #[test]
    fn format_binary_sensor() {
        let old = json!({"state": "off"});
        let new = json!({"state": "on", "attributes": {"friendly_name": "Door"}});
        let out = format_state_change("binary_sensor.door", &old, &new).unwrap();
        assert_eq!(out, "[Home Assistant] Door: triggered (was cleared)");
    }

    #[test]
    fn format_climate() {
        let old = json!({"state": "off"});
        let new = json!({
            "state": "heat",
            "attributes": {
                "friendly_name": "Thermostat",
                "current_temperature": 19.5,
                "temperature": 22
            }
        });
        let out = format_state_change("climate.main", &old, &new).unwrap();
        assert_eq!(
            out,
            "[Home Assistant] Thermostat: HVAC mode changed from 'off' to 'heat' (current: 19.5, target: 22)"
        );
    }

    #[test]
    fn format_climate_missing_temps() {
        let old = json!({"state": "off"});
        let new = json!({"state": "cool", "attributes": {"friendly_name": "AC"}});
        let out = format_state_change("climate.ac", &old, &new).unwrap();
        assert!(out.contains("(current: ?, target: ?)"));
    }

    #[test]
    fn format_alarm() {
        let old = json!({"state": "disarmed"});
        let new = json!({"state": "armed_away", "attributes": {"friendly_name": "Alarm"}});
        let out = format_state_change("alarm_control_panel.home", &old, &new).unwrap();
        assert_eq!(
            out,
            "[Home Assistant] Alarm: alarm state changed from 'disarmed' to 'armed_away'"
        );
    }

    #[test]
    fn format_generic_fallback() {
        let old = json!({"state": "idle"});
        let new = json!({"state": "playing"});
        let out = format_state_change("media_player.tv", &old, &new).unwrap();
        assert_eq!(
            out,
            "[Home Assistant] media_player.tv (media_player.tv): changed from 'idle' to 'playing'"
        );
    }

    #[test]
    fn format_old_state_unknown_when_missing() {
        let new = json!({"state": "on", "attributes": {"friendly_name": "Lamp"}});
        // old_state empty/None -> "unknown" old_val, which differs from "on".
        let out = format_state_change("light.lamp", &Value::Null, &new).unwrap();
        assert_eq!(out, "[Home Assistant] Lamp: turned on");
    }

    #[test]
    fn should_forward_ignore_filter() {
        let mut a = HomeAssistantAdapter::from_parts(
            Some("t"),
            &json!({"watch_all": true, "ignore_entities": ["light.x"], "cooldown_seconds": 0}),
        );
        assert!(!a.should_forward("light.x", 100.0));
        assert!(a.should_forward("light.y", 100.0));
    }

    #[test]
    fn should_forward_closed_by_default() {
        let mut a = HomeAssistantAdapter::from_parts(Some("t"), &json!({}));
        // no filters, watch_all off -> drop.
        assert!(!a.should_forward("light.y", 100.0));
    }

    #[test]
    fn should_forward_domain_and_entity_filters() {
        let mut a = HomeAssistantAdapter::from_parts(
            Some("t"),
            &json!({
                "watch_domains": ["light"],
                "watch_entities": ["sensor.temp"],
                "cooldown_seconds": 0
            }),
        );
        assert!(a.should_forward("light.kitchen", 1.0));
        assert!(a.should_forward("sensor.temp", 1.0));
        assert!(!a.should_forward("switch.plug", 1.0));
        assert!(!a.should_forward("sensor.other", 1.0));
    }

    #[test]
    fn should_forward_cooldown() {
        let mut a = HomeAssistantAdapter::from_parts(
            Some("t"),
            &json!({"watch_all": true, "cooldown_seconds": 30}),
        );
        assert!(a.should_forward("light.a", 100.0));
        // within cooldown.
        assert!(!a.should_forward("light.a", 120.0));
        // past cooldown.
        assert!(a.should_forward("light.a", 131.0));
    }

    #[test]
    fn should_forward_empty_entity() {
        let mut a =
            HomeAssistantAdapter::from_parts(Some("t"), &json!({"watch_all": true}));
        assert!(!a.should_forward("", 1.0));
    }

    #[test]
    fn build_event_message_full_pipeline() {
        let mut a = HomeAssistantAdapter::from_parts(
            Some("t"),
            &json!({"watch_all": true, "cooldown_seconds": 0}),
        );
        let event = json!({
            "data": {
                "entity_id": "light.lamp",
                "old_state": {"state": "off"},
                "new_state": {"state": "on", "attributes": {"friendly_name": "Lamp"}}
            }
        });
        let msg = a.build_event_message(&event, 1700.0).unwrap();
        assert_eq!(msg.text, "[Home Assistant] Lamp: turned on");
        assert_eq!(msg.message_type, MessageType::Text);
        assert_eq!(msg.source.chat_id, "ha_events");
        assert_eq!(msg.source.platform, "homeassistant");
        assert_eq!(msg.message_id.as_deref(), Some("ha_light.lamp_1700"));
    }

    #[test]
    fn build_event_message_filtered_returns_none() {
        let mut a = HomeAssistantAdapter::from_parts(Some("t"), &json!({}));
        let event = json!({"data": {"entity_id": "light.lamp",
            "new_state": {"state": "on"}, "old_state": {"state": "off"}}});
        assert!(a.build_event_message(&event, 1.0).is_none());
    }

    #[test]
    fn build_event_message_no_state_change_none() {
        let mut a = HomeAssistantAdapter::from_parts(
            Some("t"),
            &json!({"watch_all": true, "cooldown_seconds": 0}),
        );
        // passes filters but state unchanged -> no message.
        let event = json!({"data": {"entity_id": "light.lamp",
            "new_state": {"state": "on"}, "old_state": {"state": "on"}}});
        assert!(a.build_event_message(&event, 1.0).is_none());
    }

    #[test]
    fn notification_payload_truncates() {
        let long = "x".repeat(MAX_MESSAGE_LENGTH + 100);
        let payload = build_notification_payload(&long);
        assert_eq!(payload["title"], "Hermes Agent");
        let msg = payload["message"].as_str().unwrap();
        assert_eq!(msg.chars().count(), MAX_MESSAGE_LENGTH);
    }

    #[test]
    fn notification_url_and_headers() {
        assert_eq!(
            notification_url("http://ha.local:8123"),
            "http://ha.local:8123/api/services/persistent_notification/create"
        );
        let h = notification_headers("tok");
        assert!(h.contains(&("Authorization".to_string(), "Bearer tok".to_string())));
        assert!(h.contains(&("Content-Type".to_string(), "application/json".to_string())));
    }

    #[test]
    fn chat_info_shape() {
        let a = HomeAssistantAdapter::from_parts(Some("t"), &json!({"url": "http://h:1"}));
        let info = a.get_chat_info();
        assert_eq!(info["name"], "Home Assistant Events");
        assert_eq!(info["type"], "channel");
        assert_eq!(info["url"], "http://h:1");
    }

    #[test]
    fn check_requirements_env_gate() {
        unsafe {
            std::env::remove_var("HASS_TOKEN");
        }
        assert!(!check_ha_requirements());
        unsafe {
            std::env::set_var("HASS_TOKEN", "abc");
        }
        assert!(check_ha_requirements());
        unsafe {
            std::env::set_var("HASS_TOKEN", "");
        }
        assert!(!check_ha_requirements());
        unsafe {
            std::env::remove_var("HASS_TOKEN");
        }
    }
}
