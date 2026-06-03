//! WeCom callback-mode adapter for self-built enterprise applications.
//!
//! Native Rust port of `gateway/platforms/wecom_callback.py`.
//!
//! Unlike the bot/websocket adapter in `wecom.py`, this handles the standard
//! WeCom callback flow: WeCom POSTs encrypted XML to an HTTP endpoint, the
//! adapter decrypts it, queues the message for the agent, and immediately
//! acknowledges. The agent's reply is delivered later via the proactive
//! `message/send` API using an access-token.
//!
//! Supports multiple self-built apps under one gateway instance, scoped by
//! `corp_id:user_id` to avoid cross-corp collisions.
//!
//! This port keeps the deterministic logic (app normalisation, dedup,
//! user→app mapping, event building, token caching, request/response shapes)
//! native. The aiohttp server wiring and asyncio queue/poll loop live in the
//! runtime layer; the handlers here return plain values the runtime can act on.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::gw_platforms_base::{MessageEvent, MessageType, SendResult, SessionSource};
use crate::gw_wecom_crypto::{WXBizMsgCrypt, WeComCryptoError};

/// Default bind host.
pub const DEFAULT_HOST: &str = "0.0.0.0";
/// Default bind port.
pub const DEFAULT_PORT: u16 = 8645;
/// Default callback path.
pub const DEFAULT_PATH: &str = "/wecom/callback";
/// Access-token TTL fallback in seconds.
pub const ACCESS_TOKEN_TTL_SECONDS: i64 = 7200;
/// How long a seen `MsgId` suppresses duplicates, in seconds.
pub const MESSAGE_DEDUP_TTL_SECONDS: f64 = 300.0;

/// Native ports always have the equivalent of aiohttp/httpx (reqwest)
/// available, so this mirrors the Python availability check by always
/// reporting `true`.
pub fn check_wecom_callback_requirements() -> bool {
    true
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A single self-built WeCom callback application.
///
/// Mirrors the per-app dict produced by `_normalize_apps`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WecomApp {
    pub name: String,
    pub corp_id: String,
    pub corp_secret: String,
    /// String form of the agent id (matches Python `str(agent_id)`).
    pub agent_id: String,
    pub token: String,
    pub encoding_aes_key: String,
}

impl WecomApp {
    /// Parse `agent_id` to an integer, mirroring `int(str(app.get("agent_id") or 0))`.
    pub fn agent_id_int(&self) -> i64 {
        let s = self.agent_id.trim();
        if s.is_empty() {
            return 0;
        }
        s.parse::<i64>().unwrap_or(0)
    }
}

/// A cached access-token with its absolute expiry (unix seconds).
#[derive(Debug, Clone, PartialEq)]
pub struct CachedToken {
    pub token: String,
    pub expires_at: f64,
}

/// Build the scoped `corp_id:user_id` key. Falls back to bare `user_id`
/// when `corp_id` is empty. Mirrors `_user_app_key`.
pub fn user_app_key(corp_id: &str, user_id: &str) -> String {
    if !corp_id.is_empty() {
        format!("{corp_id}:{user_id}")
    } else {
        user_id.to_string()
    }
}

/// Extract a `serde_json::Value` field as a string, treating missing/non-string
/// as empty. Used when normalising the `extra` config map.
fn str_field(map: &Value, key: &str) -> String {
    match map.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// Truthy check matching Python `extra.get(key)` used in a boolean context for
/// the corp_id branch (non-empty string / non-zero number / true).
fn is_truthy(v: Option<&Value>) -> bool {
    match v {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::Bool(b)) => *b,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        _ => false,
    }
}

/// Normalise the `extra` config object into a list of [`WecomApp`].
///
/// Mirrors `_normalize_apps`:
///   * if `extra["apps"]` is a non-empty list of objects, use those;
///   * else if `extra["corp_id"]` is truthy, synthesise a single default app;
///   * else return empty.
pub fn normalize_apps(extra: &Value) -> Vec<WecomApp> {
    if let Some(Value::Array(apps)) = extra.get("apps") {
        if !apps.is_empty() {
            let parsed: Vec<WecomApp> = apps
                .iter()
                .filter(|a| a.is_object())
                .map(|a| WecomApp {
                    // `name` keeps Python's behaviour: a bare app dict may omit
                    // it. The base config always carries one, but to stay faithful
                    // we leave it empty when absent rather than defaulting.
                    name: str_field(a, "name"),
                    corp_id: str_field(a, "corp_id"),
                    corp_secret: str_field(a, "corp_secret"),
                    agent_id: str_field(a, "agent_id"),
                    token: str_field(a, "token"),
                    encoding_aes_key: str_field(a, "encoding_aes_key"),
                })
                .collect();
            return parsed;
        }
    }
    if is_truthy(extra.get("corp_id")) {
        let name = {
            let n = str_field(extra, "name");
            if n.is_empty() {
                "default".to_string()
            } else {
                n
            }
        };
        return vec![WecomApp {
            name,
            corp_id: str_field(extra, "corp_id"),
            corp_secret: str_field(extra, "corp_secret"),
            agent_id: str_field(extra, "agent_id"),
            token: str_field(extra, "token"),
            encoding_aes_key: str_field(extra, "encoding_aes_key"),
        }];
    }
    Vec::new()
}

/// Minimal `ElementTree.findtext` equivalent for the flat, single-level XML
/// WeCom emits. Returns the text content of the first `<Tag>...</Tag>` element,
/// stripping a single layer of `<![CDATA[ ... ]]>` if present. Returns `None`
/// when the element is absent (mirrors `findtext(tag)` returning `None`).
pub fn findtext(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let after = start + open.len();
    let rest = &xml[after..];
    let end_rel = rest.find(&close)?;
    let inner = &rest[..end_rel];
    Some(strip_cdata(inner))
}

/// `findtext` with a default for the missing case (mirrors
/// `findtext(tag, default=...)`).
pub fn findtext_default(xml: &str, tag: &str, default: &str) -> String {
    findtext(xml, tag).unwrap_or_else(|| default.to_string())
}

fn strip_cdata(s: &str) -> String {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("<![CDATA[") {
        if let Some(inner) = rest.strip_suffix("]]>") {
            return inner.to_string();
        }
    }
    s.to_string()
}

/// Outcome of building an inbound event from decrypted XML.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildOutcome {
    /// A dispatchable event.
    Event(MessageEvent),
    /// Recognised but intentionally ignored (lifecycle events / unsupported
    /// message types) — caller should still ACK with "success".
    Ignore,
}

/// Construct a [`WXBizMsgCrypt`] for `app`. Mirrors `_crypt_for_app`.
pub fn crypt_for_app(app: &WecomApp) -> Result<WXBizMsgCrypt, WeComCryptoError> {
    WXBizMsgCrypt::new(&app.token, &app.encoding_aes_key, &app.corp_id)
}

/// Decrypt a POST body: parse `<Encrypt>` out of the XML, verify the signature,
/// and return the decrypted inner XML as UTF-8. Mirrors `_decrypt_request`.
pub fn decrypt_request(
    app: &WecomApp,
    body: &str,
    msg_signature: &str,
    timestamp: &str,
    nonce: &str,
) -> Result<String, WeComCryptoError> {
    let encrypt = findtext_default(body, "Encrypt", "");
    let crypt = crypt_for_app(app)?;
    let plain = crypt.decrypt(msg_signature, timestamp, nonce, &encrypt)?;
    String::from_utf8(plain).map_err(|e| WeComCryptoError::Decrypt(format!("invalid utf-8: {e}")))
}

/// Build a [`MessageEvent`] from decrypted inbound XML. Mirrors `_build_event`.
///
/// Returns [`BuildOutcome::Ignore`] for `enter_agent`/`subscribe` lifecycle
/// events and for any message type other than `text`/`event`.
pub fn build_event(app: &WecomApp, xml_text: &str, platform: &str) -> BuildOutcome {
    let msg_type = findtext_default(xml_text, "MsgType", "").to_lowercase();
    if msg_type == "event" {
        let event_name = findtext_default(xml_text, "Event", "").to_lowercase();
        if event_name == "enter_agent" || event_name == "subscribe" {
            return BuildOutcome::Ignore;
        }
    }
    if msg_type != "text" && msg_type != "event" {
        return BuildOutcome::Ignore;
    }

    let user_id = findtext_default(xml_text, "FromUserName", "");
    let corp_id = findtext_default(xml_text, "ToUserName", &app.corp_id);
    let scoped_chat_id = user_app_key(&corp_id, &user_id);
    let mut content = findtext_default(xml_text, "Content", "").trim().to_string();
    if content.is_empty() && msg_type == "event" {
        content = "/start".to_string();
    }
    let msg_id = match findtext(xml_text, "MsgId") {
        Some(id) if !id.is_empty() => id,
        _ => {
            let create_time = findtext_default(xml_text, "CreateTime", "0");
            format!("{user_id}:{create_time}")
        }
    };

    let source = SessionSource {
        platform: platform.to_string(),
        chat_id: scoped_chat_id,
        chat_name: Some(user_id.clone()),
        chat_type: "dm".to_string(),
        user_id: Some(user_id.clone()),
        user_name: Some(user_id.clone()),
        ..Default::default()
    };

    BuildOutcome::Event(MessageEvent {
        text: content,
        message_type: MessageType::Text,
        source,
        message_id: Some(msg_id),
        ..Default::default()
    })
}

/// In-memory adapter state: dedup cache, user→app map, and token cache.
///
/// Mirrors `_seen_messages`, `_user_app_map`, and `_access_tokens`.
#[derive(Debug, Default)]
pub struct WecomCallbackState {
    /// MsgId -> last-seen unix time.
    pub seen_messages: HashMap<String, f64>,
    /// scoped chat_id -> app name.
    pub user_app_map: HashMap<String, String>,
    /// app name -> cached token.
    pub access_tokens: HashMap<String, CachedToken>,
}

impl WecomCallbackState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Deduplicate by `message_id`. Returns `true` when the message is a fresh
    /// (non-duplicate) one that should be processed; `false` when it is a
    /// duplicate that should be skipped. Mirrors the dedup block in
    /// `_handle_callback`.
    ///
    /// `now` is the current unix time in seconds.
    pub fn record_message(&mut self, message_id: &str, now: f64) -> bool {
        if message_id.is_empty() {
            // No id: Python only dedups when message_id is truthy, so always
            // treat as fresh.
            return true;
        }
        if let Some(&seen_at) = self.seen_messages.get(message_id) {
            if now - seen_at < MESSAGE_DEDUP_TTL_SECONDS {
                return false;
            }
            self.seen_messages.remove(message_id);
        }
        self.seen_messages.insert(message_id.to_string(), now);
        // Prune expired entries when the cache grows large.
        if self.seen_messages.len() > 2000 {
            let cutoff = now - MESSAGE_DEDUP_TTL_SECONDS;
            self.seen_messages.retain(|_, &mut v| v > cutoff);
        }
        true
    }

    /// Record which app a user belongs to. Mirrors the `_user_app_map` write.
    pub fn record_user_app(&mut self, corp_id: &str, user_id: &str, app_name: &str) {
        let key = user_app_key(corp_id, user_id);
        self.user_app_map.insert(key, app_name.to_string());
    }
}

/// Pick the app name associated with `chat_id`, falling back to a unique
/// `:user_id` suffix match for legacy bare ids. Returns `None` when no mapping
/// is found (caller then defaults to `apps[0]`). Mirrors `_resolve_app_for_chat`
/// (the name-resolution part).
pub fn resolve_app_name_for_chat(
    user_app_map: &HashMap<String, String>,
    chat_id: &str,
) -> Option<String> {
    if let Some(name) = user_app_map.get(chat_id) {
        return Some(name.clone());
    }
    if !chat_id.contains(':') {
        let suffix = format!(":{chat_id}");
        let matching: Vec<&String> = user_app_map.keys().filter(|k| k.ends_with(&suffix)).collect();
        if matching.len() == 1 {
            return user_app_map.get(matching[0]).cloned();
        }
    }
    None
}

/// Resolve the [`WecomApp`] for a `chat_id`. Mirrors `_resolve_app_for_chat`:
/// looks up the user→app map, else falls back to the first configured app.
/// Returns `None` only when there are no apps at all.
pub fn resolve_app_for_chat<'a>(
    apps: &'a [WecomApp],
    user_app_map: &HashMap<String, String>,
    chat_id: &str,
) -> Option<&'a WecomApp> {
    if let Some(name) = resolve_app_name_for_chat(user_app_map, chat_id) {
        if let Some(app) = apps.iter().find(|a| a.name == name) {
            return Some(app);
        }
    }
    apps.first()
}

/// Split a `chat_id` into the `touser` value for the send API. Mirrors
/// `chat_id.split(":", 1)[1] if ":" in chat_id else chat_id`.
pub fn touser_from_chat_id(chat_id: &str) -> &str {
    match chat_id.split_once(':') {
        Some((_, rest)) => rest,
        None => chat_id,
    }
}

/// Build the `message/send` JSON payload. Mirrors the dict in `send()`,
/// including the `content[:2048]` truncation and `agentid` integer coercion.
pub fn build_send_payload(app: &WecomApp, touser: &str, content: &str) -> Value {
    let truncated: String = content.chars().take(2048).collect();
    json!({
        "touser": touser,
        "msgtype": "text",
        "agentid": app.agent_id_int(),
        "text": {"content": truncated},
        "safe": 0,
    })
}

/// Build the `message/send` URL for a given access token. Mirrors the f-string
/// in `send()`.
pub fn send_url(token: &str) -> String {
    format!("https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}")
}

/// Parse a `message/send` API JSON response into a [`SendResult`]. Mirrors the
/// response handling in `send()`: `errcode != 0` is failure (error = stringified
/// body), success carries `msgid`.
pub fn parse_send_response(data: &Value) -> SendResult {
    let errcode = data.get("errcode").and_then(Value::as_i64).unwrap_or(-1);
    if errcode != 0 {
        return SendResult::fail(data.to_string());
    }
    let msgid = data
        .get("msgid")
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    SendResult::ok(Some(msgid))
}

/// Send a text message proactively via the access-token `message/send` API
/// (blocking). Mirrors `send()`: construct payload, POST, parse response.
///
/// `token` is the resolved access token; the caller manages token caching via
/// [`get_access_token`] / [`refresh_access_token`].
pub fn send_message(app: &WecomApp, chat_id: &str, content: &str, token: &str) -> SendResult {
    let touser = touser_from_chat_id(chat_id);
    let payload = build_send_payload(app, touser, content);
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return SendResult::fail(e.to_string()),
    };
    match client.post(send_url(token)).json(&payload).send() {
        Ok(resp) => match resp.json::<Value>() {
            Ok(data) => parse_send_response(&data),
            Err(e) => SendResult::fail(e.to_string()),
        },
        Err(e) => SendResult::fail(e.to_string()),
    }
}

/// Build the `gettoken` query parameters. Mirrors the `params` dict in
/// `_refresh_access_token`.
pub fn gettoken_params(app: &WecomApp) -> Vec<(&'static str, String)> {
    vec![
        ("corpid", app.corp_id.clone()),
        ("corpsecret", app.corp_secret.clone()),
    ]
}

/// The `gettoken` endpoint URL.
pub const GETTOKEN_URL: &str = "https://qyapi.weixin.qq.com/cgi-bin/gettoken";

/// Parse a `gettoken` response into a [`CachedToken`]. Mirrors
/// `_refresh_access_token`'s response handling: `errcode != 0` is an error;
/// otherwise cache the `access_token` with `expires_at = now + expires_in`.
///
/// Returns `Err(stringified body)` on failure, matching the Python
/// `RuntimeError(f"WeCom token refresh failed: {data}")` message body.
pub fn parse_token_response(data: &Value, now: f64) -> Result<CachedToken, String> {
    let errcode = data.get("errcode").and_then(Value::as_i64).unwrap_or(-1);
    if errcode != 0 {
        return Err(format!("WeCom token refresh failed: {data}"));
    }
    let token = data
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("WeCom token refresh failed: {data}"))?
        .to_string();
    let expires_in = data
        .get("expires_in")
        .and_then(Value::as_i64)
        .unwrap_or(ACCESS_TOKEN_TTL_SECONDS);
    Ok(CachedToken {
        token,
        expires_at: now + expires_in as f64,
    })
}

/// Fetch and cache a fresh access token (blocking). Mirrors
/// `_refresh_access_token`. On success the token is stored in
/// `state.access_tokens[app.name]` and returned.
pub fn refresh_access_token(
    state: &mut WecomCallbackState,
    app: &WecomApp,
) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(GETTOKEN_URL)
        .query(&gettoken_params(app))
        .send()
        .map_err(|e| e.to_string())?;
    let data: Value = resp.json().map_err(|e| e.to_string())?;
    let cached = parse_token_response(&data, now_secs())?;
    let token = cached.token.clone();
    state.access_tokens.insert(app.name.clone(), cached);
    Ok(token)
}

/// Return a cached access token if still valid (>60s of headroom), else
/// refresh. Mirrors `_get_access_token`.
pub fn get_access_token(state: &mut WecomCallbackState, app: &WecomApp) -> Result<String, String> {
    let now = now_secs();
    if let Some(cached) = state.access_tokens.get(&app.name) {
        if cached.expires_at > now + 60.0 {
            return Ok(cached.token.clone());
        }
    }
    refresh_access_token(state, app)
}

/// Result of handling a POST callback against the configured apps. The runtime
/// translates these into HTTP responses ("success", 400, etc.) and dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackOutcome {
    /// A decrypted, dispatchable, non-duplicate event. The runtime should
    /// enqueue it and respond "success".
    Dispatch(MessageEvent),
    /// Decrypted successfully but no event to dispatch (duplicate / ignored
    /// lifecycle). The runtime should respond "success".
    Ack,
    /// No configured app could decrypt the payload. The runtime should respond
    /// 400 "invalid callback payload".
    Invalid,
}

/// Process a POST callback body against the configured apps. Mirrors the
/// per-app loop in `_handle_callback`, including dedup and user→app recording.
///
/// On the first app whose crypto succeeds:
///   * build the event; if ignored / duplicate → [`CallbackOutcome::Ack`];
///   * otherwise record the user→app mapping and return
///     [`CallbackOutcome::Dispatch`].
///
/// `WeComCryptoError` from an app means "try the next app". Any other error in
/// event building would break the loop in Python; here event building is
/// infallible, so only crypto failures advance the loop.
pub fn handle_callback(
    apps: &[WecomApp],
    state: &mut WecomCallbackState,
    body: &str,
    msg_signature: &str,
    timestamp: &str,
    nonce: &str,
    platform: &str,
) -> CallbackOutcome {
    for app in apps {
        let decrypted = match decrypt_request(app, body, msg_signature, timestamp, nonce) {
            Ok(d) => d,
            Err(_) => continue,
        };
        match build_event(app, &decrypted, platform) {
            BuildOutcome::Ignore => return CallbackOutcome::Ack,
            BuildOutcome::Event(event) => {
                let now = now_secs();
                if let Some(mid) = &event.message_id {
                    if !state.record_message(mid, now) {
                        return CallbackOutcome::Ack;
                    }
                }
                if let Some(uid) = &event.source.user_id {
                    if !uid.is_empty() {
                        state.record_user_app(&app.corp_id, uid, &app.name);
                    }
                }
                return CallbackOutcome::Dispatch(event);
            }
        }
    }
    CallbackOutcome::Invalid
}

/// Handle a GET verification handshake against the configured apps. Returns the
/// decrypted echo string on the first app that verifies, else `None` (caller
/// responds 403). Mirrors `_handle_verify`.
pub fn handle_verify(
    apps: &[WecomApp],
    msg_signature: &str,
    timestamp: &str,
    nonce: &str,
    echostr: &str,
) -> Option<String> {
    for app in apps {
        let crypt = match crypt_for_app(app) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Ok(plain) = crypt.verify_url(msg_signature, timestamp, nonce, echostr) {
            return Some(plain);
        }
    }
    None
}

/// The health endpoint JSON body. Mirrors `_handle_health`.
pub fn health_body() -> Value {
    json!({"status": "ok", "platform": "wecom_callback"})
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_app_key_scopes_with_corp() {
        assert_eq!(user_app_key("corp1", "alice"), "corp1:alice");
        assert_eq!(user_app_key("", "alice"), "alice");
    }

    #[test]
    fn normalize_apps_from_list() {
        let extra = json!({
            "apps": [
                {"name": "a1", "corp_id": "c1", "agent_id": 1000, "token": "t", "encoding_aes_key": "k"},
                {"name": "a2", "corp_id": "c2"},
                "not-a-dict"
            ]
        });
        let apps = normalize_apps(&extra);
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0].name, "a1");
        assert_eq!(apps[0].corp_id, "c1");
        assert_eq!(apps[0].agent_id, "1000");
        assert_eq!(apps[1].name, "a2");
    }

    #[test]
    fn normalize_apps_synthesises_default() {
        let extra = json!({
            "corp_id": "corpX",
            "corp_secret": "secret",
            "agent_id": 42,
            "token": "tok",
            "encoding_aes_key": "aes",
        });
        let apps = normalize_apps(&extra);
        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].name, "default");
        assert_eq!(apps[0].corp_id, "corpX");
        assert_eq!(apps[0].agent_id, "42");
    }

    #[test]
    fn normalize_apps_named_default() {
        let extra = json!({"corp_id": "corpX", "name": "primary"});
        let apps = normalize_apps(&extra);
        assert_eq!(apps[0].name, "primary");
    }

    #[test]
    fn normalize_apps_empty_when_no_corp() {
        let extra = json!({"host": "1.2.3.4"});
        assert!(normalize_apps(&extra).is_empty());
    }

    #[test]
    fn agent_id_int_parsing() {
        let mut app = WecomApp::default();
        app.agent_id = "1000".to_string();
        assert_eq!(app.agent_id_int(), 1000);
        app.agent_id = "".to_string();
        assert_eq!(app.agent_id_int(), 0);
        app.agent_id = "junk".to_string();
        assert_eq!(app.agent_id_int(), 0);
    }

    #[test]
    fn findtext_basic_and_cdata() {
        let xml = "<xml><MsgType>text</MsgType><Content><![CDATA[hello]]></Content></xml>";
        assert_eq!(findtext(xml, "MsgType").as_deref(), Some("text"));
        assert_eq!(findtext(xml, "Content").as_deref(), Some("hello"));
        assert_eq!(findtext(xml, "Missing"), None);
        assert_eq!(findtext_default(xml, "Missing", "fallback"), "fallback");
    }

    #[test]
    fn build_event_text_message() {
        let app = WecomApp {
            name: "default".into(),
            corp_id: "corpA".into(),
            ..Default::default()
        };
        let xml = "<xml><MsgType>text</MsgType><FromUserName>alice</FromUserName>\
                   <ToUserName>corpA</ToUserName><Content>hi there</Content>\
                   <MsgId>123456</MsgId></xml>";
        match build_event(&app, xml, "wecom_callback") {
            BuildOutcome::Event(ev) => {
                assert_eq!(ev.text, "hi there");
                assert_eq!(ev.message_id.as_deref(), Some("123456"));
                assert_eq!(ev.source.chat_id, "corpA:alice");
                assert_eq!(ev.source.user_id.as_deref(), Some("alice"));
                assert_eq!(ev.source.chat_type, "dm");
                assert_eq!(ev.message_type, MessageType::Text);
            }
            other => panic!("expected event, got {other:?}"),
        }
    }

    #[test]
    fn build_event_lifecycle_ignored() {
        let app = WecomApp::default();
        let xml = "<xml><MsgType>event</MsgType><Event>enter_agent</Event></xml>";
        assert_eq!(build_event(&app, xml, "p"), BuildOutcome::Ignore);
        let xml2 = "<xml><MsgType>event</MsgType><Event>subscribe</Event></xml>";
        assert_eq!(build_event(&app, xml2, "p"), BuildOutcome::Ignore);
    }

    #[test]
    fn build_event_other_event_gets_start() {
        let app = WecomApp {
            corp_id: "c".into(),
            ..Default::default()
        };
        let xml = "<xml><MsgType>event</MsgType><Event>click</Event>\
                   <FromUserName>bob</FromUserName><CreateTime>999</CreateTime></xml>";
        match build_event(&app, xml, "p") {
            BuildOutcome::Event(ev) => {
                assert_eq!(ev.text, "/start");
                // No MsgId → fallback "user:createtime".
                assert_eq!(ev.message_id.as_deref(), Some("bob:999"));
            }
            other => panic!("expected event, got {other:?}"),
        }
    }

    #[test]
    fn build_event_unsupported_type_ignored() {
        let app = WecomApp::default();
        let xml = "<xml><MsgType>image</MsgType><FromUserName>x</FromUserName></xml>";
        assert_eq!(build_event(&app, xml, "p"), BuildOutcome::Ignore);
    }

    #[test]
    fn dedup_records_and_skips() {
        let mut state = WecomCallbackState::new();
        assert!(state.record_message("m1", 1000.0));
        // Same id within TTL → duplicate.
        assert!(!state.record_message("m1", 1100.0));
        // Same id after TTL → fresh again.
        assert!(state.record_message("m1", 1000.0 + MESSAGE_DEDUP_TTL_SECONDS + 1.0));
        // Empty id always fresh.
        assert!(state.record_message("", 0.0));
    }

    #[test]
    fn resolve_app_name_direct_and_suffix() {
        let mut map = HashMap::new();
        map.insert("corpA:alice".to_string(), "app1".to_string());
        // Direct hit.
        assert_eq!(
            resolve_app_name_for_chat(&map, "corpA:alice").as_deref(),
            Some("app1")
        );
        // Bare user_id unique suffix match.
        assert_eq!(
            resolve_app_name_for_chat(&map, "alice").as_deref(),
            Some("app1")
        );
        // Bare user_id with no match.
        assert_eq!(resolve_app_name_for_chat(&map, "nobody"), None);
    }

    #[test]
    fn resolve_app_name_suffix_ambiguous() {
        let mut map = HashMap::new();
        map.insert("corpA:alice".to_string(), "app1".to_string());
        map.insert("corpB:alice".to_string(), "app2".to_string());
        // Two matches → no unique resolution.
        assert_eq!(resolve_app_name_for_chat(&map, "alice"), None);
    }

    #[test]
    fn resolve_app_falls_back_to_first() {
        let apps = vec![
            WecomApp {
                name: "first".into(),
                ..Default::default()
            },
            WecomApp {
                name: "second".into(),
                ..Default::default()
            },
        ];
        let map = HashMap::new();
        let app = resolve_app_for_chat(&apps, &map, "corpZ:zoe").unwrap();
        assert_eq!(app.name, "first");
    }

    #[test]
    fn touser_split() {
        assert_eq!(touser_from_chat_id("corpA:alice"), "alice");
        assert_eq!(touser_from_chat_id("alice"), "alice");
        // Only the first colon splits.
        assert_eq!(touser_from_chat_id("corpA:a:b"), "a:b");
    }

    #[test]
    fn send_payload_shape_and_truncation() {
        let app = WecomApp {
            agent_id: "7".into(),
            ..Default::default()
        };
        let long = "x".repeat(3000);
        let payload = build_send_payload(&app, "alice", &long);
        assert_eq!(payload["touser"], "alice");
        assert_eq!(payload["msgtype"], "text");
        assert_eq!(payload["agentid"], 7);
        assert_eq!(payload["safe"], 0);
        let content = payload["text"]["content"].as_str().unwrap();
        assert_eq!(content.chars().count(), 2048);
    }

    #[test]
    fn parse_send_response_success_and_failure() {
        let ok = json!({"errcode": 0, "msgid": "abc"});
        let r = parse_send_response(&ok);
        assert!(r.success);
        assert_eq!(r.message_id.as_deref(), Some("abc"));

        let bad = json!({"errcode": 81013, "errmsg": "user not found"});
        let r2 = parse_send_response(&bad);
        assert!(!r2.success);
        assert!(r2.error.unwrap().contains("81013"));
    }

    #[test]
    fn parse_token_response_caches_expiry() {
        let data = json!({"errcode": 0, "access_token": "TT", "expires_in": 100});
        let cached = parse_token_response(&data, 1000.0).unwrap();
        assert_eq!(cached.token, "TT");
        assert_eq!(cached.expires_at, 1100.0);

        let data_default = json!({"errcode": 0, "access_token": "UU"});
        let c2 = parse_token_response(&data_default, 0.0).unwrap();
        assert_eq!(c2.expires_at, ACCESS_TOKEN_TTL_SECONDS as f64);

        let err = json!({"errcode": 40013, "errmsg": "invalid corpid"});
        assert!(parse_token_response(&err, 0.0).is_err());
    }

    #[test]
    fn get_access_token_uses_cache() {
        let mut state = WecomCallbackState::new();
        let app = WecomApp {
            name: "default".into(),
            ..Default::default()
        };
        state.access_tokens.insert(
            "default".to_string(),
            CachedToken {
                token: "cached".to_string(),
                expires_at: now_secs() + 3600.0,
            },
        );
        assert_eq!(get_access_token(&mut state, &app).unwrap(), "cached");
    }

    #[test]
    fn gettoken_params_shape() {
        let app = WecomApp {
            corp_id: "c".into(),
            corp_secret: "s".into(),
            ..Default::default()
        };
        let params = gettoken_params(&app);
        assert_eq!(params[0], ("corpid", "c".to_string()));
        assert_eq!(params[1], ("corpsecret", "s".to_string()));
    }

    #[test]
    fn health_body_shape() {
        let b = health_body();
        assert_eq!(b["status"], "ok");
        assert_eq!(b["platform"], "wecom_callback");
    }

    #[test]
    fn requirements_always_available() {
        assert!(check_wecom_callback_requirements());
    }
}
