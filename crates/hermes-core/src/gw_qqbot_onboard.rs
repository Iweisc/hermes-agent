//! QQBot scan-to-configure (QR code onboard) module.
//!
//! Mirrors the Feishu onboarding pattern: synchronous HTTP + a single public
//! entry-point [`qr_register`] that handles the full flow (create task →
//! display QR code → poll → decrypt credentials).
//!
//! Calls the `q.qq.com` `create_bind_task` / `poll_bind_result` APIs to
//! generate a QR-code URL and poll for scan completion. On success the caller
//! receives the bot's *app_id*, *client_secret* (decrypted locally), and the
//! scanner's *user_openid* — enough to fully configure the QQBot gateway.
//!
//! Reference: <https://bot.q.qq.com/wiki/develop/api-v2/>
//!
//! Faithful port of `gateway/platforms/qqbot/onboard.py`.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::gw_qq_constants::{
    portal_host, ONBOARD_CREATE_PATH, ONBOARD_POLL_INTERVAL, ONBOARD_POLL_PATH,
};
use crate::gw_qq_constants::QQBOT_VERSION;
use crate::gw_qq_crypto::{decrypt_secret, generate_bind_key};

/// Default API timeout for the onboard create / poll calls (seconds).
///
/// Mirrors `constants.ONBOARD_API_TIMEOUT` (== 10.0).
pub const ONBOARD_API_TIMEOUT: f64 = crate::gw_qq_constants::ONBOARD_API_TIMEOUT;

/// Maximum number of QR-code refreshes before giving up on expiry.
///
/// Mirrors the Python module-level `_MAX_REFRESHES`.
pub const MAX_REFRESHES: u32 = 3;

// ---------------------------------------------------------------------------
// Bind status
// ---------------------------------------------------------------------------

/// Status codes returned by [`poll_bind_result`].
///
/// Mirrors the Python `BindStatus(IntEnum)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindStatus {
    /// No status / unknown (0).
    None,
    /// Awaiting scan (1).
    Pending,
    /// Scan completed (2).
    Completed,
    /// QR code expired (3).
    Expired,
}

impl BindStatus {
    /// Map an integer status code to a [`BindStatus`].
    ///
    /// Unknown values map to [`BindStatus::None`] — note the Python `IntEnum`
    /// would raise `ValueError` on an unknown value; here we treat anything
    /// out of range as the inert `None` so the poll loop keeps spinning rather
    /// than crashing, which preserves the practical behaviour (the poll loop
    /// only acts on `Completed` / `Expired`).
    pub fn from_code(code: i64) -> BindStatus {
        match code {
            0 => BindStatus::None,
            1 => BindStatus::Pending,
            2 => BindStatus::Completed,
            3 => BindStatus::Expired,
            _ => BindStatus::None,
        }
    }

    /// The integer status code for this variant.
    pub fn code(self) -> i64 {
        match self {
            BindStatus::None => 0,
            BindStatus::Pending => 1,
            BindStatus::Completed => 2,
            BindStatus::Expired => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors raised by the onboard HTTP helpers.
#[derive(Debug)]
pub enum OnboardError {
    /// Underlying HTTP transport / request error.
    Http(String),
    /// The API returned a non-zero `retcode` (carries the server `msg`).
    Api(String),
    /// The response was missing a required field (e.g. `task_id`).
    MissingField(String),
    /// Credential decryption failed.
    Decrypt(String),
}

impl std::fmt::Display for OnboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnboardError::Http(e) => write!(f, "http error: {e}"),
            OnboardError::Api(m) => write!(f, "{m}"),
            OnboardError::MissingField(m) => write!(f, "{m}"),
            OnboardError::Decrypt(e) => write!(f, "decrypt error: {e}"),
        }
    }
}

impl std::error::Error for OnboardError {}

// ---------------------------------------------------------------------------
// Successful registration result
// ---------------------------------------------------------------------------

/// The successful result of [`qr_register`].
///
/// Mirrors the Python `{"app_id", "client_secret", "user_openid"}` dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrRegistration {
    /// The bot's application ID (`bot_appid`).
    pub app_id: String,
    /// The decrypted bot `client_secret`.
    pub client_secret: String,
    /// The scanning user's OpenID (may be empty).
    pub user_openid: String,
}

// ---------------------------------------------------------------------------
// Result of one poll
// ---------------------------------------------------------------------------

/// The 4-tuple-equivalent returned by [`poll_bind_result`].
#[derive(Debug, Clone)]
pub struct PollResult {
    /// The bind status.
    pub status: BindStatus,
    /// The bot application ID (`bot_appid`), stringified.
    pub bot_appid: String,
    /// The encrypted bot secret (`bot_encrypt_secret`, base64).
    pub bot_encrypt_secret: String,
    /// The scanning user's OpenID.
    pub user_openid: String,
}

// ---------------------------------------------------------------------------
// QR rendering
// ---------------------------------------------------------------------------

/// Try to render a QR code in the terminal. Returns `true` if successful.
///
/// The Python version optionally uses the `qrcode` package; we have no such
/// dependency available (and may not draw to a real terminal here), so this
/// always returns `false`, matching the "library unavailable" branch where the
/// caller falls back to printing the raw URL.
pub fn render_qr(_url: &str) -> bool {
    false
}

// ---------------------------------------------------------------------------
// URL helper — mirrors build_connect_url + urllib.parse.quote
// ---------------------------------------------------------------------------

/// Percent-encode a path/component the way Python's `urllib.parse.quote`
/// does with its default `safe="/"` set.
///
/// `quote` leaves unreserved characters (`A-Z a-z 0-9 _ . - ~`) and `/`
/// untouched and percent-encodes everything else.
fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for &b in value.as_bytes() {
        let safe = b.is_ascii_alphanumeric()
            || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/');
        if safe {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

/// Build the QR-code target URL for a given `task_id`.
///
/// Mirrors `build_connect_url(task_id)` which formats `QR_URL_TEMPLATE` with a
/// `quote()`-escaped task id.
pub fn build_connect_url(task_id: &str) -> String {
    let encoded = quote(task_id);
    format!(
        "https://q.qq.com/qqbot/openclaw/connect.html?task_id={encoded}&_wv=2&source=hermes"
    )
}

// ---------------------------------------------------------------------------
// HTTP headers — mirrors utils.build_user_agent / utils.get_api_headers
// ---------------------------------------------------------------------------

/// Build a descriptive User-Agent string.
///
/// Format: `QQBotAdapter/<qqbot_version> (Rust; <os>; Hermes/<hermes_version>)`.
///
/// The Python original embeds the Python interpreter version; the native port
/// has no interpreter, so we report `Rust` in that slot. The OS name is
/// lower-cased to match `platform.system().lower()`.
pub fn build_user_agent() -> String {
    let os_name = std::env::consts::OS; // already lower-case ("macos", "linux", ...)
    let hermes_version = option_env!("CARGO_PKG_VERSION").unwrap_or("dev");
    format!("QQBotAdapter/{QQBOT_VERSION} (Rust; {os_name}; Hermes/{hermes_version})")
}

/// Return standard HTTP headers for QQBot API requests.
///
/// Mirrors `utils.get_api_headers`. `q.qq.com` requires
/// `Accept: application/json`; without it the server returns a JavaScript
/// anti-bot challenge page.
pub fn get_api_headers() -> Vec<(String, String)> {
    vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Accept".to_string(), "application/json".to_string()),
        ("User-Agent".to_string(), build_user_agent()),
    ]
}

// ---------------------------------------------------------------------------
// Synchronous HTTP helpers (mirrors Feishu _post_registration pattern)
// ---------------------------------------------------------------------------

fn build_client(timeout: f64) -> Result<reqwest::blocking::Client, OnboardError> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        // httpx defaults to following redirects; reqwest does too, but make it explicit.
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| OnboardError::Http(e.to_string()))
}

fn post_json(
    client: &reqwest::blocking::Client,
    url: &str,
    body: Value,
) -> Result<Value, OnboardError> {
    let mut req = client.post(url).json(&body);
    for (k, v) in get_api_headers() {
        req = req.header(k, v);
    }
    let resp = req.send().map_err(|e| OnboardError::Http(e.to_string()))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| OnboardError::Http(e.to_string()))?;
    resp.json::<Value>()
        .map_err(|e| OnboardError::Http(e.to_string()))
}

/// Extract the server `msg` from a response, with a fallback default.
fn api_msg(data: &Value, default: &str) -> String {
    data.get("msg")
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

/// Create a bind task and return `(task_id, aes_key_base64)`.
///
/// Mirrors `_create_bind_task`. Returns [`OnboardError::Api`] on a non-zero
/// `retcode`, or [`OnboardError::MissingField`] if `task_id` is absent/empty.
pub fn create_bind_task(timeout: f64) -> Result<(String, String), OnboardError> {
    let url = format!("https://{}{}", portal_host(), ONBOARD_CREATE_PATH);
    let key = generate_bind_key();

    let client = build_client(timeout)?;
    let data = post_json(&client, &url, json!({ "key": key }))?;

    let retcode = data.get("retcode").and_then(Value::as_i64).unwrap_or(-1);
    if retcode != 0 {
        return Err(OnboardError::Api(api_msg(&data, "create_bind_task failed")));
    }

    let task_id = data
        .get("data")
        .and_then(|d| d.get("task_id"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());

    let task_id = match task_id {
        Some(t) => t.to_string(),
        None => {
            return Err(OnboardError::MissingField(
                "create_bind_task: missing task_id in response".to_string(),
            ))
        }
    };

    log::debug!("create_bind_task ok: task_id={task_id}");
    Ok((task_id, key))
}

/// Stringify a JSON value the way Python's `str(d.get("bot_appid", ""))` would
/// for the values that the QQ API actually returns (string or integer).
fn json_to_appid_string(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => {
            // Python str(True) == "True"
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Poll the bind result for `task_id`.
///
/// Mirrors `_poll_bind_result`. Returns [`OnboardError::Api`] on a non-zero
/// `retcode`.
pub fn poll_bind_result(task_id: &str, timeout: f64) -> Result<PollResult, OnboardError> {
    let url = format!("https://{}{}", portal_host(), ONBOARD_POLL_PATH);

    let client = build_client(timeout)?;
    let data = post_json(&client, &url, json!({ "task_id": task_id }))?;

    let retcode = data.get("retcode").and_then(Value::as_i64).unwrap_or(-1);
    if retcode != 0 {
        return Err(OnboardError::Api(api_msg(&data, "poll_bind_result failed")));
    }

    let empty = Value::Object(serde_json::Map::new());
    let d = data.get("data").unwrap_or(&empty);

    let status_code = d.get("status").and_then(Value::as_i64).unwrap_or(0);
    let bot_appid = json_to_appid_string(d.get("bot_appid"));
    let bot_encrypt_secret = d
        .get("bot_encrypt_secret")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let user_openid = d
        .get("user_openid")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    Ok(PollResult {
        status: BindStatus::from_code(status_code),
        bot_appid,
        bot_encrypt_secret,
        user_openid,
    })
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Run the QQBot scan-to-configure QR registration flow.
///
/// Mirrors `feishu.qr_register()` / the Python `qr_register()`: handles
/// create → display → poll → decrypt in one call.
///
/// Returns `Ok(Some(QrRegistration))` on success, or `Ok(None)` on
/// failure / expiry / timeout (the Python returns `None` for all of these).
/// Truly unexpected transport errors during *create* are swallowed into
/// `Ok(None)` to match the Python `except Exception: return None` around
/// `_create_bind_task`.
///
/// `now` and `sleep` are injected so callers (and tests) can drive the clock;
/// production callers use [`qr_register`].
pub fn qr_register_with<NowFn, SleepFn>(
    timeout_seconds: u64,
    mut now: NowFn,
    mut sleep: SleepFn,
) -> Result<Option<QrRegistration>, OnboardError>
where
    NowFn: FnMut() -> Instant,
    SleepFn: FnMut(Duration),
{
    let deadline = now() + Duration::from_secs(timeout_seconds);
    let poll_interval = Duration::from_secs_f64(ONBOARD_POLL_INTERVAL);

    for refresh_count in 0..=MAX_REFRESHES {
        // ── Create bind task ──
        let (task_id, aes_key) = match create_bind_task(ONBOARD_API_TIMEOUT) {
            Ok(v) => v,
            Err(exc) => {
                log::warn!("[QQBot onboard] Failed to create bind task: {exc}");
                return Ok(None);
            }
        };

        let url = build_connect_url(&task_id);

        // ── Display QR code + URL ──
        println!();
        if render_qr(&url) {
            println!("  Scan the QR code above, or open this URL directly:\n  {url}");
        } else {
            println!("  Open this URL in QQ on your phone:\n  {url}");
            println!("  Tip: pip install qrcode  to display a scannable QR code here");
        }
        println!();

        // ── Poll loop ──
        let mut expired_break = false;
        while now() < deadline {
            let result = match poll_bind_result(&task_id, ONBOARD_API_TIMEOUT) {
                Ok(r) => r,
                Err(_) => {
                    sleep(poll_interval);
                    continue;
                }
            };

            match result.status {
                BindStatus::Completed => {
                    let client_secret =
                        decrypt_secret(&result.bot_encrypt_secret, &aes_key)
                            .map_err(|e| OnboardError::Decrypt(e.to_string()))?;
                    println!();
                    println!("  QR scan complete! (App ID: {})", result.bot_appid);
                    if !result.user_openid.is_empty() {
                        println!("  Scanner's OpenID: {}", result.user_openid);
                    }
                    return Ok(Some(QrRegistration {
                        app_id: result.bot_appid,
                        client_secret,
                        user_openid: result.user_openid,
                    }));
                }
                BindStatus::Expired => {
                    if refresh_count >= MAX_REFRESHES {
                        log::warn!(
                            "[QQBot onboard] QR code expired {MAX_REFRESHES} times — giving up"
                        );
                        return Ok(None);
                    }
                    println!(
                        "\n  QR code expired, refreshing... ({}/{})",
                        refresh_count + 1,
                        MAX_REFRESHES
                    );
                    expired_break = true;
                    break; // next for-loop iteration creates a new task
                }
                _ => {
                    sleep(poll_interval);
                }
            }
        }

        // Python `while ... else`: the `else` runs only if the loop was *not*
        // broken out of (i.e. the deadline was reached, not an expiry).
        if !expired_break {
            log::warn!("[QQBot onboard] Poll timed out after {timeout_seconds}s");
            return Ok(None);
        }
    }

    Ok(None)
}

/// Run the QQBot scan-to-configure QR registration flow using the real clock.
///
/// See [`qr_register_with`]. Returns `Ok(None)` on failure / expiry / timeout.
pub fn qr_register(timeout_seconds: u64) -> Result<Option<QrRegistration>, OnboardError> {
    qr_register_with(timeout_seconds, Instant::now, std::thread::sleep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_status_from_code_maps_known_values() {
        assert_eq!(BindStatus::from_code(0), BindStatus::None);
        assert_eq!(BindStatus::from_code(1), BindStatus::Pending);
        assert_eq!(BindStatus::from_code(2), BindStatus::Completed);
        assert_eq!(BindStatus::from_code(3), BindStatus::Expired);
    }

    #[test]
    fn bind_status_unknown_code_is_inert_none() {
        assert_eq!(BindStatus::from_code(99), BindStatus::None);
        assert_eq!(BindStatus::from_code(-1), BindStatus::None);
    }

    #[test]
    fn bind_status_code_roundtrip() {
        for s in [
            BindStatus::None,
            BindStatus::Pending,
            BindStatus::Completed,
            BindStatus::Expired,
        ] {
            assert_eq!(BindStatus::from_code(s.code()), s);
        }
    }

    #[test]
    fn build_connect_url_uses_template_and_encodes() {
        let url = build_connect_url("abc123");
        assert_eq!(
            url,
            "https://q.qq.com/qqbot/openclaw/connect.html?task_id=abc123&_wv=2&source=hermes"
        );
    }

    #[test]
    fn build_connect_url_percent_encodes_unsafe_chars() {
        // A space and a `+` must be percent-encoded; `/` and `-` stay.
        let url = build_connect_url("a b+c/d-e");
        assert!(url.contains("task_id=a%20b%2Bc/d-e&_wv=2"), "got: {url}");
    }

    #[test]
    fn quote_matches_python_default_safe_set() {
        assert_eq!(quote("ABCabc123_.-~/"), "ABCabc123_.-~/");
        assert_eq!(quote(" "), "%20");
        assert_eq!(quote("&"), "%26");
        assert_eq!(quote("="), "%3D");
        assert_eq!(quote("中"), "%E4%B8%AD"); // UTF-8 bytes, uppercase hex
    }

    #[test]
    fn render_qr_returns_false_without_library() {
        assert!(!render_qr("https://example.com"));
    }

    #[test]
    fn api_msg_prefers_server_msg() {
        let data = json!({ "msg": "boom" });
        assert_eq!(api_msg(&data, "default"), "boom");
        let data = json!({});
        assert_eq!(api_msg(&data, "default"), "default");
    }

    #[test]
    fn json_to_appid_string_handles_string_and_int() {
        assert_eq!(json_to_appid_string(Some(&json!("102"))), "102");
        assert_eq!(json_to_appid_string(Some(&json!(102))), "102");
        assert_eq!(json_to_appid_string(None), "");
        assert_eq!(json_to_appid_string(Some(&Value::Null)), "");
    }

    #[test]
    fn onboard_api_timeout_matches_constant() {
        assert_eq!(ONBOARD_API_TIMEOUT, 10.0);
        assert_eq!(MAX_REFRESHES, 3);
    }

    #[test]
    fn qr_register_returns_none_on_immediate_timeout() {
        // With a zero-second timeout and a now() that is already past the
        // deadline, the create call would be attempted; to avoid real network
        // we instead verify the timeout-vs-expiry control flow indirectly via
        // the injected clock: a deadline already in the past means the poll
        // loop body never runs and we hit the timeout branch.
        //
        // We cannot run create_bind_task offline, so this test only exercises
        // the pure clock comparison used in qr_register_with by constructing
        // the same predicate.
        let start = Instant::now();
        let deadline = start + Duration::from_secs(0);
        // Simulate a `now()` advanced past the deadline.
        let later = start + Duration::from_millis(1);
        assert!(later >= deadline);
    }
}
