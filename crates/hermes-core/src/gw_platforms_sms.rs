//! SMS (Twilio) platform adapter, ported from `gateway/platforms/sms.py`.
//!
//! The Python module connects to the Twilio REST API for outbound SMS and runs
//! an aiohttp webhook server to receive inbound messages. Each inbound phone
//! number gets its own Hermes session (multi-tenant); replies are always sent
//! from the configured `TWILIO_PHONE_NUMBER`.
//!
//! This port reproduces the behaviour-defining logic faithfully and idiomatically:
//!   - environment-driven configuration ([`SmsConfig::from_env`]),
//!   - HTTP Basic auth header construction ([`SmsConfig::basic_auth_header`]),
//!   - markdown-stripping message formatting ([`format_message`]),
//!   - Twilio `X-Twilio-Signature` validation (HMAC-SHA1 / base64), including the
//!     default-port toggle variant ([`validate_twilio_signature`],
//!     [`check_signature`], [`port_variant_url`]),
//!   - inbound webhook form parsing + field extraction + echo / empty rejection
//!     ([`parse_webhook_form`], [`WebhookDecision`], [`evaluate_webhook`]),
//!   - the empty-TwiML response body ([`EMPTY_TWIML`]),
//!   - outbound send request construction ([`build_send_request`]) and Twilio
//!     JSON response parsing ([`parse_send_response`]).
//!
//! Network calls use `reqwest::blocking`; the aiohttp webhook server / asyncio
//! task orchestration is modelled as synchronous request-building + response /
//! webhook-decision helpers rather than dragging in an event loop.
//!
//! Cross-refs:
//!   - [`crate::gw_platforms_base`] — `MessageEvent`, `MessageType`, `SendResult`,
//!     `SessionSource`, `truncate_message`.
//!   - [`crate::gw_helpers`] — `redact_phone`, `strip_markdown`.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use hmac::{Hmac, Mac};
use sha1::Sha1;

use crate::gw_helpers::{redact_phone, strip_markdown};
use crate::gw_platforms_base::{
    truncate_message, MessageEvent, MessageType, SendResult, SessionSource,
};

type HmacSha1 = Hmac<Sha1>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const TWILIO_API_BASE: &str = "https://api.twilio.com/2010-04-01/Accounts";
/// ~10 SMS segments.
pub const MAX_SMS_LENGTH: usize = 1600;
pub const DEFAULT_WEBHOOK_PORT: u16 = 8080;
pub const DEFAULT_WEBHOOK_HOST: &str = "127.0.0.1";

/// The empty TwiML body returned for every webhook (replies go via REST API).
pub const EMPTY_TWIML: &str =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response></Response>";

/// Requirement check: SID + auth token present. Mirrors `check_sms_requirements`.
///
/// The Python version also checks that the optional `aiohttp` dependency is
/// importable; in the native port the HTTP stack is always available, so this
/// only reflects the credential check.
pub fn check_sms_requirements(account_sid: Option<&str>, auth_token: Option<&str>) -> bool {
    !account_sid.unwrap_or("").is_empty() && !auth_token.unwrap_or("").is_empty()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SMS adapter configuration sourced from environment variables. Mirrors the
/// fields read in `SmsAdapter.__init__`.
#[derive(Debug, Clone)]
pub struct SmsConfig {
    pub account_sid: String,
    pub auth_token: String,
    pub from_number: String,
    pub webhook_port: u16,
    pub webhook_host: String,
    pub webhook_url: String,
}

impl SmsConfig {
    /// Build configuration straight from the process environment. Mirrors the
    /// env reads in `SmsAdapter.__init__`. `TWILIO_ACCOUNT_SID` /
    /// `TWILIO_AUTH_TOKEN` are required (Python uses `os.environ[...]` which
    /// raises `KeyError` if missing); returns `Err` with the missing var name.
    pub fn from_env() -> Result<Self, String> {
        let account_sid = std::env::var("TWILIO_ACCOUNT_SID")
            .map_err(|_| "TWILIO_ACCOUNT_SID".to_string())?;
        let auth_token =
            std::env::var("TWILIO_AUTH_TOKEN").map_err(|_| "TWILIO_AUTH_TOKEN".to_string())?;
        let from_number = std::env::var("TWILIO_PHONE_NUMBER").unwrap_or_default();

        let webhook_port = std::env::var("SMS_WEBHOOK_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(DEFAULT_WEBHOOK_PORT);
        let webhook_host =
            std::env::var("SMS_WEBHOOK_HOST").unwrap_or_else(|_| DEFAULT_WEBHOOK_HOST.to_string());
        let webhook_url = std::env::var("SMS_WEBHOOK_URL")
            .unwrap_or_default()
            .trim()
            .to_string();

        Ok(SmsConfig {
            account_sid,
            auth_token,
            from_number,
            webhook_port,
            webhook_host,
            webhook_url,
        })
    }

    /// Whether `SMS_INSECURE_NO_SIGNATURE` is set to (case-insensitive) "true".
    pub fn insecure_no_signature() -> bool {
        std::env::var("SMS_INSECURE_NO_SIGNATURE")
            .unwrap_or_default()
            .to_lowercase()
            == "true"
    }

    /// Build the HTTP Basic auth header value for Twilio. Mirrors
    /// `_basic_auth_header`.
    pub fn basic_auth_header(&self) -> String {
        let creds = format!("{}:{}", self.account_sid, self.auth_token);
        let encoded = base64::engine::general_purpose::STANDARD.encode(creds.as_bytes());
        format!("Basic {encoded}")
    }
}

/// Outcome of validating preconditions in `connect`. Mirrors the early-return
/// branches: `Ok` means the server may start, `Err` carries the fatal error
/// reason code that Python passes to `_set_fatal_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectPrecheck {
    /// May start. `warn_insecure` is true when insecure-no-signature was set and
    /// no webhook URL is configured (Python logs a warning in this case).
    Ok { warn_insecure: bool },
    /// Fatal: reason code + message. Mirrors `_set_fatal_error(reason, msg)`.
    Fatal { reason: String, message: String },
}

/// Validate connect preconditions. Mirrors the head of `SmsAdapter.connect`.
pub fn connect_precheck(cfg: &SmsConfig, insecure_no_sig: bool) -> ConnectPrecheck {
    if cfg.from_number.is_empty() {
        return ConnectPrecheck::Fatal {
            reason: "sms_missing_phone_number".to_string(),
            message: "[sms] TWILIO_PHONE_NUMBER not set — cannot send replies".to_string(),
        };
    }

    if cfg.webhook_url.is_empty() && !insecure_no_sig {
        let msg = "[sms] Refusing to start: SMS_WEBHOOK_URL is required for Twilio \
signature validation. Set it to the public URL configured in your \
Twilio console (e.g. https://example.com/webhooks/twilio). \
For local development without validation, set \
SMS_INSECURE_NO_SIGNATURE=true (NOT recommended for production)."
            .to_string();
        return ConnectPrecheck::Fatal {
            reason: "sms_missing_webhook_url".to_string(),
            message: msg,
        };
    }

    let warn_insecure = insecure_no_sig && cfg.webhook_url.is_empty();
    ConnectPrecheck::Ok { warn_insecure }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Strip markdown — SMS renders it as literal characters. Mirrors
/// `format_message`.
pub fn format_message(content: &str) -> String {
    strip_markdown(content)
}

// ---------------------------------------------------------------------------
// Twilio signature validation
// ---------------------------------------------------------------------------

/// Validate the `X-Twilio-Signature` header. Tries both with and without the
/// default port for the URL scheme, since Twilio may sign with either variant.
/// Mirrors `_validate_twilio_signature`.
///
/// `post_params` are the flattened (single-valued) POST parameters.
pub fn validate_twilio_signature(
    auth_token: &str,
    url: &str,
    post_params: &BTreeMap<String, String>,
    signature: &str,
) -> bool {
    if check_signature(auth_token, url, post_params, signature) {
        return true;
    }
    if let Some(variant) = port_variant_url(url) {
        if check_signature(auth_token, &variant, post_params, signature) {
            return true;
        }
    }
    false
}

/// Compute and compare a single Twilio signature. Mirrors `_check_signature`.
///
/// Algorithm: concatenate the URL then, for each POST param sorted by key,
/// `key + value`; HMAC-SHA1 over that with the auth token; base64; constant-time
/// compare against the provided signature.
pub fn check_signature(
    auth_token: &str,
    url: &str,
    post_params: &BTreeMap<String, String>,
    signature: &str,
) -> bool {
    // BTreeMap iterates in sorted key order, matching `sorted(post_params.keys())`.
    let mut data_to_sign = String::from(url);
    for (key, value) in post_params.iter() {
        data_to_sign.push_str(key);
        data_to_sign.push_str(value);
    }

    let mut mac = match HmacSha1::new_from_slice(auth_token.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(data_to_sign.as_bytes());
    let digest = mac.finalize().into_bytes();
    let computed = base64::engine::general_purpose::STANDARD.encode(digest);

    constant_time_eq(computed.as_bytes(), signature.as_bytes())
}

/// Constant-time byte-string comparison. Mirrors `hmac.compare_digest`.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Return the URL with the default port toggled, or `None`. Only toggles default
/// ports (443 for https, 80 for http); non-standard ports are never modified.
/// Mirrors `_port_variant_url`.
pub fn port_variant_url(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    let scheme = parsed.scheme();
    let default_port = match scheme {
        "https" => 443u16,
        "http" => 80u16,
        _ => return None,
    };

    // `Url::port()` returns Some only when an *explicit* port is present;
    // `port_or_known_default()` falls back to the scheme default. We need to
    // distinguish "no explicit port" from "explicit default port", matching
    // urllib's `parsed.port` (None when absent).
    let explicit_port = parsed.port();
    let host = parsed.host_str().unwrap_or("");

    // Rebuild path + params/query/fragment exactly like urlunparse over the
    // urlsplit components (urllib treats `params` as empty for these URLs).
    let path = parsed.path();
    let query_suffix = match parsed.query() {
        Some(q) => format!("?{q}"),
        None => String::new(),
    };
    let fragment_suffix = match parsed.fragment() {
        Some(f) => format!("#{f}"),
        None => String::new(),
    };

    match explicit_port {
        Some(p) if p == default_port => {
            // Has explicit default port → strip it.
            Some(format!("{scheme}://{host}{path}{query_suffix}{fragment_suffix}"))
        }
        None => {
            // No port → add the default.
            Some(format!(
                "{scheme}://{host}:{default_port}{path}{query_suffix}{fragment_suffix}"
            ))
        }
        // Non-standard explicit port — no variant.
        Some(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Webhook form parsing + decision
// ---------------------------------------------------------------------------

/// Parse a Twilio form-encoded webhook body into a multi-valued map, preserving
/// blank values. Mirrors `urllib.parse.parse_qs(..., keep_blank_values=True)`.
pub fn parse_webhook_form(raw: &str) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for pair in raw.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        let key = form_urldecode(k);
        let val = form_urldecode(v);
        // parse_qs drops entries whose *key* is empty.
        if key.is_empty() {
            continue;
        }
        out.entry(key).or_default().push(val);
    }
    out
}

/// Decode an `application/x-www-form-urlencoded` component: `+` → space, then
/// percent-decode UTF-8.
fn form_urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h = hex_val(bytes[i + 1]);
                let l = hex_val(bytes[i + 2]);
                match (h, l) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Flatten a parsed form to single values, mirroring
/// `{k: v[0] for k, v in form.items() if v}`.
pub fn flatten_form(form: &BTreeMap<String, Vec<String>>) -> BTreeMap<String, String> {
    form.iter()
        .filter_map(|(k, v)| v.first().map(|first| (k.clone(), first.clone())))
        .collect()
}

/// First value of a form field, stripped, or empty string. Mirrors
/// `(form.get(key, [""]))[0].strip()`.
fn field(form: &BTreeMap<String, Vec<String>>, key: &str) -> String {
    form.get(key)
        .and_then(|v| v.first())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// HTTP response the webhook handler produces. The `body` is always
/// [`EMPTY_TWIML`] with content-type `application/xml`; `status` matches the
/// Python branches (200 / 400 / 403).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebhookResponse {
    pub status: u16,
    pub body: &'static str,
    pub content_type: &'static str,
}

impl WebhookResponse {
    fn twiml(status: u16) -> Self {
        WebhookResponse {
            status,
            body: EMPTY_TWIML,
            content_type: "application/xml",
        }
    }
}

/// The decision made for an inbound webhook request. Either a short-circuit
/// [`WebhookResponse`] (reject / echo / empty / parse error), or a constructed
/// [`MessageEvent`] to dispatch (always paired with the 200 empty-TwiML reply).
#[derive(Debug, Clone)]
pub enum WebhookDecision {
    /// Respond immediately without dispatching a message.
    Respond(WebhookResponse),
    /// Dispatch this event to the handler, then return the empty-TwiML 200.
    Dispatch {
        event: Box<MessageEvent>,
        response: WebhookResponse,
    },
}

/// Build the [`SessionSource`] for an inbound SMS. Mirrors `build_source(...)`
/// in `_handle_webhook`: chat/user id/name all default to the from-number, and
/// the chat type is `dm`.
pub fn build_source(from_number: &str) -> SessionSource {
    SessionSource {
        platform: "sms".to_string(),
        chat_id: from_number.to_string(),
        chat_name: Some(from_number.to_string()),
        chat_type: "dm".to_string(),
        user_id: Some(from_number.to_string()),
        user_name: Some(from_number.to_string()),
        ..SessionSource::default()
    }
}

/// Evaluate an inbound webhook request and decide what to do. Faithfully mirrors
/// `SmsAdapter._handle_webhook`.
///
/// - `raw_body`: the raw request body bytes (form-encoded).
/// - `twilio_signature`: the `X-Twilio-Signature` header value (empty if absent).
/// - `cfg`: adapter configuration (provides webhook URL, auth token, from-number).
///
/// Parse failures and signature rejections short-circuit; otherwise the
/// extracted fields are validated (non-empty from + text, echo suppression) and
/// a [`MessageEvent`] is built for dispatch.
pub fn evaluate_webhook(
    cfg: &SmsConfig,
    raw_body: &[u8],
    twilio_signature: &str,
) -> WebhookDecision {
    // Twilio sends form-encoded data, not JSON. UTF-8 decode then parse_qs.
    let raw = match std::str::from_utf8(raw_body) {
        Ok(s) => s,
        Err(_) => return WebhookDecision::Respond(WebhookResponse::twiml(400)),
    };
    let form = parse_webhook_form(raw);

    // Validate Twilio request signature when SMS_WEBHOOK_URL is configured.
    if !cfg.webhook_url.is_empty() {
        if twilio_signature.is_empty() {
            return WebhookDecision::Respond(WebhookResponse::twiml(403));
        }
        let flat = flatten_form(&form);
        if !validate_twilio_signature(
            &cfg.auth_token,
            &cfg.webhook_url,
            &flat,
            twilio_signature,
        ) {
            return WebhookDecision::Respond(WebhookResponse::twiml(403));
        }
    }

    let from_number = field(&form, "From");
    let _to_number = field(&form, "To");
    let text = field(&form, "Body");
    let message_sid = field(&form, "MessageSid");

    if from_number.is_empty() || text.is_empty() {
        return WebhookDecision::Respond(WebhookResponse::twiml(200));
    }

    // Ignore messages from our own number (echo prevention).
    if from_number == cfg.from_number {
        return WebhookDecision::Respond(WebhookResponse::twiml(200));
    }

    let source = build_source(&from_number);
    let event = MessageEvent {
        text,
        message_type: MessageType::Text,
        source,
        message_id: if message_sid.is_empty() {
            None
        } else {
            Some(message_sid)
        },
        ..MessageEvent::default()
    };

    WebhookDecision::Dispatch {
        event: Box::new(event),
        response: WebhookResponse::twiml(200),
    }
}

// ---------------------------------------------------------------------------
// Outbound send
// ---------------------------------------------------------------------------

/// A single outbound Twilio request (one per truncated chunk). Mirrors the
/// per-chunk POST built in `SmsAdapter.send`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SendRequest {
    pub url: String,
    pub authorization: String,
    /// Form fields: `From`, `To`, `Body`.
    pub form: Vec<(String, String)>,
}

/// Build the per-chunk send requests for `content` sent to `chat_id`. Mirrors
/// `format_message` → `truncate_message` → per-chunk FormData construction in
/// `SmsAdapter.send`.
pub fn build_send_request(cfg: &SmsConfig, chat_id: &str, content: &str) -> Vec<SendRequest> {
    let formatted = format_message(content);
    let chunks = truncate_message(&formatted, MAX_SMS_LENGTH, None);
    let url = format!("{}/{}/Messages.json", TWILIO_API_BASE, cfg.account_sid);
    let auth = cfg.basic_auth_header();

    chunks
        .into_iter()
        .map(|chunk| SendRequest {
            url: url.clone(),
            authorization: auth.clone(),
            form: vec![
                ("From".to_string(), cfg.from_number.clone()),
                ("To".to_string(), chat_id.to_string()),
                ("Body".to_string(), chunk),
            ],
        })
        .collect()
}

/// Parse a Twilio Messages.json response. Mirrors the per-response handling in
/// `SmsAdapter.send`: `status >= 400` → failure with
/// `"Twilio {status}: {message-or-body}"`; otherwise success with the `sid`.
pub fn parse_send_response(status: u16, body: &serde_json::Value) -> SendResult {
    if status >= 400 {
        let error_msg = body
            .get("message")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| body.to_string());
        return SendResult::fail(format!("Twilio {status}: {error_msg}"));
    }
    let sid = body
        .get("sid")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();
    SendResult::ok(Some(sid))
}

/// Execute the full outbound send over `reqwest::blocking`, mirroring
/// `SmsAdapter.send`. Sends each chunk sequentially; the first failing chunk
/// short-circuits (matching the Python early `return`). Returns the last
/// successful result, or the failure.
///
/// On transport error for a chunk, returns `SendResult::fail(<error>)`,
/// mirroring the `except Exception as e` branch.
pub fn send_blocking(cfg: &SmsConfig, chat_id: &str, content: &str) -> SendResult {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(e) => return SendResult::fail(e.to_string()),
    };

    let mut last_result = SendResult::ok(None);
    for req in build_send_request(cfg, chat_id, content) {
        let resp = client
            .post(&req.url)
            .header("Authorization", req.authorization)
            .form(&req.form)
            .send();

        let resp = match resp {
            Ok(r) => r,
            Err(e) => return SendResult::fail(e.to_string()),
        };
        let status = resp.status().as_u16();
        let body: serde_json::Value = match resp.json() {
            Ok(b) => b,
            Err(e) => return SendResult::fail(e.to_string()),
        };
        let result = parse_send_response(status, &body);
        if !result.success {
            return result;
        }
        last_result = result;
    }
    last_result
}

/// Static chat-info for an SMS conversation. Mirrors `get_chat_info`.
pub fn get_chat_info(chat_id: &str) -> serde_json::Value {
    serde_json::json!({ "name": chat_id, "type": "dm" })
}

// ---------------------------------------------------------------------------
// Logging helpers (parity with Python log lines)
// ---------------------------------------------------------------------------

/// Format the inbound log message. Mirrors the `logger.info("[sms] inbound ...")`
/// line: the body is truncated to the first 80 characters.
pub fn inbound_log_line(from_number: &str, to_number: &str, text: &str) -> String {
    let snippet: String = text.chars().take(80).collect();
    format!(
        "[sms] inbound from {} -> {}: {}",
        redact_phone(from_number),
        redact_phone(to_number),
        snippet
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(from: &str, url: &str) -> SmsConfig {
        SmsConfig {
            account_sid: "AC123".to_string(),
            auth_token: "tok".to_string(),
            from_number: from.to_string(),
            webhook_port: DEFAULT_WEBHOOK_PORT,
            webhook_host: DEFAULT_WEBHOOK_HOST.to_string(),
            webhook_url: url.to_string(),
        }
    }

    #[test]
    fn requirements_check() {
        assert!(check_sms_requirements(Some("AC"), Some("tok")));
        assert!(!check_sms_requirements(Some(""), Some("tok")));
        assert!(!check_sms_requirements(None, Some("tok")));
        assert!(!check_sms_requirements(Some("AC"), None));
    }

    #[test]
    fn basic_auth_header_matches_python() {
        let c = cfg("+15551234567", "");
        // base64("AC123:tok")
        let expected = base64::engine::general_purpose::STANDARD.encode("AC123:tok");
        assert_eq!(c.basic_auth_header(), format!("Basic {expected}"));
    }

    #[test]
    fn precheck_missing_phone_number() {
        let c = cfg("", "https://x.com/webhooks/twilio");
        match connect_precheck(&c, false) {
            ConnectPrecheck::Fatal { reason, .. } => {
                assert_eq!(reason, "sms_missing_phone_number");
            }
            _ => panic!("expected fatal"),
        }
    }

    #[test]
    fn precheck_missing_webhook_url_requires_url() {
        let c = cfg("+1555", "");
        match connect_precheck(&c, false) {
            ConnectPrecheck::Fatal { reason, .. } => {
                assert_eq!(reason, "sms_missing_webhook_url");
            }
            _ => panic!("expected fatal"),
        }
    }

    #[test]
    fn precheck_insecure_no_url_warns() {
        let c = cfg("+1555", "");
        assert_eq!(
            connect_precheck(&c, true),
            ConnectPrecheck::Ok { warn_insecure: true }
        );
    }

    #[test]
    fn precheck_ok_with_url() {
        let c = cfg("+1555", "https://x.com/webhooks/twilio");
        assert_eq!(
            connect_precheck(&c, false),
            ConnectPrecheck::Ok {
                warn_insecure: false
            }
        );
    }

    #[test]
    fn format_strips_markdown() {
        assert_eq!(format_message("**bold**"), "bold");
    }

    #[test]
    fn signature_roundtrip() {
        // Compute a signature the way Twilio does, then validate it.
        let auth = "my_auth_token";
        let url = "https://example.com/webhooks/twilio";
        let mut params = BTreeMap::new();
        params.insert("From".to_string(), "+15551234567".to_string());
        params.insert("To".to_string(), "+15557654321".to_string());
        params.insert("Body".to_string(), "Hello".to_string());

        let mut data = String::from(url);
        for (k, v) in params.iter() {
            data.push_str(k);
            data.push_str(v);
        }
        let mut mac = HmacSha1::new_from_slice(auth.as_bytes()).unwrap();
        mac.update(data.as_bytes());
        let sig =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

        assert!(validate_twilio_signature(auth, url, &params, &sig));
        assert!(!validate_twilio_signature(auth, url, &params, "wrong"));
    }

    #[test]
    fn signature_param_order_independent_of_insertion() {
        // BTreeMap sorts keys, matching sorted(post_params.keys()).
        let auth = "tok";
        let url = "https://h/";
        let mut a = BTreeMap::new();
        a.insert("b".to_string(), "2".to_string());
        a.insert("a".to_string(), "1".to_string());
        // Expected concatenation: url + "a" + "1" + "b" + "2"
        let mut data = String::from(url);
        data.push_str("a1b2");
        let mut mac = HmacSha1::new_from_slice(auth.as_bytes()).unwrap();
        mac.update(data.as_bytes());
        let sig =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
        assert!(check_signature(auth, url, &a, &sig));
    }

    #[test]
    fn port_variant_adds_default_when_absent() {
        assert_eq!(
            port_variant_url("https://example.com/webhooks/twilio"),
            Some("https://example.com:443/webhooks/twilio".to_string())
        );
        assert_eq!(
            port_variant_url("http://example.com/x"),
            Some("http://example.com:80/x".to_string())
        );
    }

    #[test]
    fn port_variant_strips_explicit_default() {
        assert_eq!(
            port_variant_url("https://example.com:443/webhooks/twilio"),
            Some("https://example.com/webhooks/twilio".to_string())
        );
        assert_eq!(
            port_variant_url("http://example.com:80/x"),
            Some("http://example.com/x".to_string())
        );
    }

    #[test]
    fn port_variant_none_for_nonstandard_or_unknown() {
        assert_eq!(port_variant_url("https://example.com:8443/x"), None);
        assert_eq!(port_variant_url("ftp://example.com/x"), None);
    }

    #[test]
    fn port_variant_preserves_query_and_fragment() {
        assert_eq!(
            port_variant_url("https://h/p?q=1#f"),
            Some("https://h:443/p?q=1#f".to_string())
        );
    }

    #[test]
    fn parse_form_keeps_blank_and_multivalue() {
        let form = parse_webhook_form("From=%2B1555&Body=&From=second");
        assert_eq!(form.get("From").unwrap().len(), 2);
        assert_eq!(form.get("From").unwrap()[0], "+1555");
        assert_eq!(form.get("Body").unwrap()[0], "");
    }

    #[test]
    fn form_decodes_plus_as_space() {
        let form = parse_webhook_form("Body=hello+world");
        assert_eq!(form.get("Body").unwrap()[0], "hello world");
    }

    #[test]
    fn flatten_takes_first() {
        let form = parse_webhook_form("From=a&From=b");
        let flat = flatten_form(&form);
        assert_eq!(flat.get("From").unwrap(), "a");
    }

    #[test]
    fn webhook_missing_signature_when_url_set() {
        let c = cfg("+1555", "https://h/twilio");
        let body = b"From=%2B1666&Body=hi";
        match evaluate_webhook(&c, body, "") {
            WebhookDecision::Respond(r) => assert_eq!(r.status, 403),
            _ => panic!("expected 403"),
        }
    }

    #[test]
    fn webhook_invalid_signature() {
        let c = cfg("+1555", "https://h/twilio");
        let body = b"From=%2B1666&Body=hi";
        match evaluate_webhook(&c, body, "deadbeef") {
            WebhookDecision::Respond(r) => assert_eq!(r.status, 403),
            _ => panic!("expected 403"),
        }
    }

    #[test]
    fn webhook_empty_body_returns_200_no_dispatch() {
        // No webhook_url → signature validation skipped.
        let c = cfg("+1555", "");
        let body = b"From=%2B1666&Body=";
        match evaluate_webhook(&c, body, "") {
            WebhookDecision::Respond(r) => assert_eq!(r.status, 200),
            _ => panic!("expected respond"),
        }
    }

    #[test]
    fn webhook_echo_suppressed() {
        let c = cfg("+1555", "");
        let body = b"From=%2B1555&Body=hi";
        match evaluate_webhook(&c, body, "") {
            WebhookDecision::Respond(r) => assert_eq!(r.status, 200),
            _ => panic!("expected respond (echo)"),
        }
    }

    #[test]
    fn webhook_dispatch_builds_event() {
        let c = cfg("+1555", "");
        let body = b"From=%2B1666&To=%2B1555&Body=hello&MessageSid=SM1";
        match evaluate_webhook(&c, body, "") {
            WebhookDecision::Dispatch { event, response } => {
                assert_eq!(response.status, 200);
                assert_eq!(event.text, "hello");
                assert_eq!(event.message_id.as_deref(), Some("SM1"));
                assert_eq!(event.source.chat_id, "+1666");
                assert_eq!(event.source.chat_type, "dm");
                assert_eq!(event.source.platform, "sms");
                assert_eq!(event.message_type, MessageType::Text);
            }
            _ => panic!("expected dispatch"),
        }
    }

    #[test]
    fn build_send_request_fields() {
        let c = cfg("+1555", "");
        let reqs = build_send_request(&c, "+1666", "hi");
        assert_eq!(reqs.len(), 1);
        let r = &reqs[0];
        assert_eq!(
            r.url,
            "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json"
        );
        assert!(r.authorization.starts_with("Basic "));
        assert_eq!(r.form[0], ("From".to_string(), "+1555".to_string()));
        assert_eq!(r.form[1], ("To".to_string(), "+1666".to_string()));
        assert_eq!(r.form[2], ("Body".to_string(), "hi".to_string()));
    }

    #[test]
    fn build_send_request_chunks_long_content() {
        let c = cfg("+1555", "");
        let long = "x".repeat(MAX_SMS_LENGTH * 2 + 50);
        let reqs = build_send_request(&c, "+1666", &long);
        assert!(reqs.len() > 1);
    }

    #[test]
    fn parse_response_success() {
        let body = serde_json::json!({ "sid": "SM999" });
        let r = parse_send_response(201, &body);
        assert!(r.success);
        assert_eq!(r.message_id.as_deref(), Some("SM999"));
    }

    #[test]
    fn parse_response_error_with_message() {
        let body = serde_json::json!({ "message": "bad number" });
        let r = parse_send_response(400, &body);
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("Twilio 400: bad number"));
    }

    #[test]
    fn parse_response_error_without_message_uses_body() {
        let body = serde_json::json!({ "code": 21211 });
        let r = parse_send_response(400, &body);
        assert!(!r.success);
        let err = r.error.unwrap();
        assert!(err.starts_with("Twilio 400: "));
        assert!(err.contains("21211"));
    }

    #[test]
    fn chat_info_shape() {
        let info = get_chat_info("+1666");
        assert_eq!(info["name"], "+1666");
        assert_eq!(info["type"], "dm");
    }

    #[test]
    fn inbound_log_truncates_body() {
        let line = inbound_log_line("+15551234567", "+15557654321", &"a".repeat(200));
        // 80 chars of body in the snippet
        let body_part = line.rsplit(": ").next().unwrap();
        assert_eq!(body_part.chars().count(), 80);
    }
}
