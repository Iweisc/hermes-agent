//! DingTalk Device Flow authorization.
//!
//! Native Rust port of `hermes_cli/dingtalk_auth.py`.
//!
//! Implements the same 3-step registration flow as
//! `dingtalk-openclaw-connector`:
//!   1. `POST /app/registration/init`   → get nonce
//!   2. `POST /app/registration/begin`  → get device_code + verification_uri_complete
//!   3. `POST /app/registration/poll`   → poll until SUCCESS → get client_id + client_secret
//!
//! The `verification_uri_complete` is rendered as a QR code in the terminal so
//! the user can scan it with DingTalk to authorize, yielding AppKey + AppSecret
//! automatically.

use std::env;
use std::fmt;
use std::thread::sleep;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ── Configuration ──────────────────────────────────────────────────────────

/// Default registration base URL when the env var is unset.
const DEFAULT_REGISTRATION_BASE_URL: &str = "https://oapi.dingtalk.com";

/// Default registration source when the env var is unset.
const DEFAULT_REGISTRATION_SOURCE: &str = "openClaw";

/// Resolve `DINGTALK_REGISTRATION_BASE_URL`, trailing-slash trimmed.
///
/// Matches the Python module-level constant
/// `os.environ.get(..., default).rstrip("/")`.
pub fn registration_base_url() -> String {
    let raw = env::var("DINGTALK_REGISTRATION_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_REGISTRATION_BASE_URL.to_string());
    raw.trim_end_matches('/').to_string()
}

/// Resolve `DINGTALK_REGISTRATION_SOURCE`.
pub fn registration_source() -> String {
    env::var("DINGTALK_REGISTRATION_SOURCE")
        .unwrap_or_else(|_| DEFAULT_REGISTRATION_SOURCE.to_string())
}

// ── API helpers ────────────────────────────────────────────────────────────

/// Raised when a DingTalk registration API call fails.
///
/// Mirrors the Python `RegistrationError(Exception)`.
#[derive(Debug, Clone)]
pub struct RegistrationError {
    pub message: String,
}

impl RegistrationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RegistrationError {}

/// Coerce a JSON value to a trimmed string the way Python's
/// `str(data.get(key, "")).strip()` does.
///
/// - Missing/null → "" → trimmed "" .
/// - String → its trimmed contents.
/// - Number/bool → its `str()`-like rendering, then trimmed.
fn json_str(data: &Value, key: &str) -> String {
    match data.get(key) {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Bool(b)) => {
            // Python str(True) == "True"
            if *b { "True" } else { "False" }.to_string()
        }
        Some(Value::Number(n)) => n.to_string().trim().to_string(),
        Some(other) => other.to_string().trim().to_string(),
    }
}

/// Coerce a JSON value to an i64 the way Python's `int(data.get(key, default))`
/// behaves for numeric/string inputs, falling back to `default`.
fn json_int(data: &Value, key: &str, default: i64) -> i64 {
    match data.get(key) {
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                f as i64
            } else {
                default
            }
        }
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(default),
        _ => default,
    }
}

/// POST to the registration API and return the parsed JSON body.
///
/// Performs a blocking HTTP request with a 15s timeout, raises
/// [`RegistrationError`] on network failures, non-2xx status, JSON parse
/// failures, or an `errcode != 0` business error.
pub fn api_post(path: &str, payload: Value) -> Result<Value, RegistrationError> {
    let url = format!("{}{}", registration_base_url(), path);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| RegistrationError::new(format!("Network error calling {url}: {e}")))?;

    let resp = client
        .post(&url)
        .json(&payload)
        .send()
        .map_err(|e| RegistrationError::new(format!("Network error calling {url}: {e}")))?;

    let resp = resp
        .error_for_status()
        .map_err(|e| RegistrationError::new(format!("Network error calling {url}: {e}")))?;

    let data: Value = resp
        .json()
        .map_err(|e| RegistrationError::new(format!("Network error calling {url}: {e}")))?;

    let errcode = json_int(&data, "errcode", -1);
    if errcode != 0 {
        let errmsg = match data.get("errmsg") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "unknown error".to_string(),
        };
        return Err(RegistrationError::new(format!(
            "API error [{path}]: {errmsg} (errcode={errcode})"
        )));
    }
    Ok(data)
}

// ── Core flow ──────────────────────────────────────────────────────────────

/// Result of [`begin_registration`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeginRegistration {
    pub device_code: String,
    pub verification_uri_complete: String,
    pub expires_in: i64,
    pub interval: i64,
}

/// Parse a `/app/registration/init` response into a nonce.
///
/// Split out from network I/O so it can be unit-tested.
pub fn parse_init(init_data: &Value) -> Result<String, RegistrationError> {
    let nonce = json_str(init_data, "nonce");
    if nonce.is_empty() {
        return Err(RegistrationError::new("init response missing nonce"));
    }
    Ok(nonce)
}

/// Parse a `/app/registration/begin` response into [`BeginRegistration`].
///
/// Split out from network I/O so it can be unit-tested.
pub fn parse_begin(begin_data: &Value) -> Result<BeginRegistration, RegistrationError> {
    let device_code = json_str(begin_data, "device_code");
    let verification_uri_complete = json_str(begin_data, "verification_uri_complete");
    if device_code.is_empty() {
        return Err(RegistrationError::new("begin response missing device_code"));
    }
    if verification_uri_complete.is_empty() {
        return Err(RegistrationError::new(
            "begin response missing verification_uri_complete",
        ));
    }
    Ok(BeginRegistration {
        device_code,
        verification_uri_complete,
        expires_in: json_int(begin_data, "expires_in", 7200),
        interval: json_int(begin_data, "interval", 3).max(2),
    })
}

/// Start a device-flow registration.
///
/// Performs the init → begin handshake and returns the device code plus
/// verification URI.
pub fn begin_registration() -> Result<BeginRegistration, RegistrationError> {
    // Step 1: init → nonce
    let init_data = api_post(
        "/app/registration/init",
        json!({ "source": registration_source() }),
    )?;
    let nonce = parse_init(&init_data)?;

    // Step 2: begin → device_code, verification_uri_complete
    let begin_data = api_post("/app/registration/begin", json!({ "nonce": nonce }))?;
    parse_begin(&begin_data)
}

/// Normalised registration status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationStatus {
    Waiting,
    Success,
    Fail,
    Expired,
    Unknown,
}

impl RegistrationStatus {
    /// Map a raw, already upper-cased status string to the enum, defaulting to
    /// `Unknown` for unrecognised values (matches the Python fallback).
    fn from_raw(raw: &str) -> Self {
        match raw {
            "WAITING" => RegistrationStatus::Waiting,
            "SUCCESS" => RegistrationStatus::Success,
            "FAIL" => RegistrationStatus::Fail,
            "EXPIRED" => RegistrationStatus::Expired,
            _ => RegistrationStatus::Unknown,
        }
    }

    /// Render back to the canonical upper-cased label (used in error messages).
    pub fn as_str(self) -> &'static str {
        match self {
            RegistrationStatus::Waiting => "WAITING",
            RegistrationStatus::Success => "SUCCESS",
            RegistrationStatus::Fail => "FAIL",
            RegistrationStatus::Expired => "EXPIRED",
            RegistrationStatus::Unknown => "UNKNOWN",
        }
    }
}

/// Result of a single [`poll_registration`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollResult {
    pub status: RegistrationStatus,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub fail_reason: Option<String>,
}

/// Parse a `/app/registration/poll` response into a [`PollResult`].
///
/// Split out from network I/O so it can be unit-tested.
pub fn parse_poll(data: &Value) -> PollResult {
    let status_raw = json_str(data, "status").to_uppercase();
    let status = RegistrationStatus::from_raw(&status_raw);

    let none_if_empty = |s: String| if s.is_empty() { None } else { Some(s) };

    PollResult {
        status,
        client_id: none_if_empty(json_str(data, "client_id")),
        client_secret: none_if_empty(json_str(data, "client_secret")),
        fail_reason: none_if_empty(json_str(data, "fail_reason")),
    }
}

/// Poll the registration status once.
pub fn poll_registration(device_code: &str) -> Result<PollResult, RegistrationError> {
    let data = api_post(
        "/app/registration/poll",
        json!({ "device_code": device_code }),
    )?;
    Ok(parse_poll(&data))
}

/// Block until the registration succeeds or times out.
///
/// Returns `(client_id, client_secret)`. `on_waiting` is invoked once per
/// `WAITING` poll, mirroring the Python callback used to print progress dots.
///
/// The polling/sleeping is delegated to a `poll` closure so the loop logic can
/// be unit-tested deterministically; see [`wait_for_registration_success`] for
/// the network-backed wrapper.
pub fn wait_for_registration_success_with<P, W, C>(
    interval: i64,
    expires_in: i64,
    mut poll: P,
    mut on_waiting: W,
    now: C,
    mut do_sleep: impl FnMut(Duration),
) -> Result<(String, String), RegistrationError>
where
    P: FnMut() -> Result<PollResult, RegistrationError>,
    W: FnMut(),
    C: Fn() -> Instant,
{
    let interval = interval.max(0) as u64;
    let deadline = now() + Duration::from_secs(expires_in.max(0) as u64);
    let retry_window = Duration::from_secs(120); // 2 minutes for transient errors
    let mut retry_start: Option<Instant> = None;

    while now() < deadline {
        do_sleep(Duration::from_secs(interval));

        let result = match poll() {
            Ok(r) => r,
            Err(e) => {
                let start = *retry_start.get_or_insert_with(&now);
                if now().duration_since(start) < retry_window {
                    continue;
                }
                return Err(e);
            }
        };

        match result.status {
            RegistrationStatus::Waiting => {
                retry_start = None;
                on_waiting();
                continue;
            }
            RegistrationStatus::Success => {
                let cid = result.client_id.clone();
                let csecret = result.client_secret.clone();
                match (cid, csecret) {
                    (Some(cid), Some(csecret)) if !cid.is_empty() && !csecret.is_empty() => {
                        return Ok((cid, csecret));
                    }
                    _ => {
                        return Err(RegistrationError::new(
                            "authorization succeeded but credentials are missing",
                        ));
                    }
                }
            }
            // FAIL / EXPIRED / UNKNOWN
            _ => {
                let start = *retry_start.get_or_insert_with(&now);
                if now().duration_since(start) < retry_window {
                    continue;
                }
                let reason = result
                    .fail_reason
                    .clone()
                    .unwrap_or_else(|| result.status.as_str().to_string());
                return Err(RegistrationError::new(format!(
                    "authorization failed: {reason}"
                )));
            }
        }
    }

    Err(RegistrationError::new("authorization timed out, please retry"))
}

/// Network-backed convenience wrapper around
/// [`wait_for_registration_success_with`].
///
/// Polls the live API every `interval` seconds (real sleeping and clock) until
/// success or timeout.
pub fn wait_for_registration_success<W: FnMut()>(
    device_code: &str,
    interval: i64,
    expires_in: i64,
    on_waiting: W,
) -> Result<(String, String), RegistrationError> {
    wait_for_registration_success_with(
        interval,
        expires_in,
        || poll_registration(device_code),
        on_waiting,
        Instant::now,
        sleep,
    )
}

// ── QR code rendering ───────────────────────────────────────────────────────

const TOP_HALF: char = '\u{2580}'; // ▀
const BOTTOM_HALF: char = '\u{2584}'; // ▄
const FULL_BLOCK: char = '\u{2588}'; // █
const EMPTY: char = ' ';

/// Render a boolean QR matrix as terminal text using half-block characters
/// (2 rows packed into one text line), prefixed with a 4-space margin.
///
/// This mirrors the rendering half of the Python `render_qr_to_terminal`. The
/// actual QR encoding (the `qrcode` library) is not reproduced here because no
/// pure-Rust QR encoder is among the allowed crates; callers that already have
/// a matrix can use this directly. Returns the joined string of lines.
pub fn render_matrix(matrix: &[Vec<bool>]) -> String {
    let rows = matrix.len();
    let mut lines: Vec<String> = Vec::new();

    let mut r = 0;
    while r < rows {
        let mut line = String::from("    ");
        for c in 0..matrix[r].len() {
            let top = matrix[r][c];
            let bottom = if r + 1 < rows {
                // Guard against ragged matrices.
                matrix[r + 1].get(c).copied().unwrap_or(false)
            } else {
                false
            };
            let ch = match (top, bottom) {
                (true, true) => FULL_BLOCK,
                (true, false) => TOP_HALF,
                (false, true) => BOTTOM_HALF,
                (false, false) => EMPTY,
            };
            line.push(ch);
        }
        lines.push(line);
        r += 2;
    }

    lines.join("\n")
}

/// Render `matrix` to stdout as a QR code, returning `true` if anything was
/// printed (i.e. the matrix was non-empty).
///
/// The Python `render_qr_to_terminal(url)` performs QR encoding internally and
/// returns `False` only when the `qrcode` library is unavailable. Since QR
/// encoding lives outside the allowed crate set, this port accepts a
/// pre-encoded matrix; an empty matrix is treated as "nothing to render".
pub fn render_qr_to_terminal(matrix: &[Vec<bool>]) -> bool {
    if matrix.is_empty() {
        return false;
    }
    println!("{}", render_matrix(matrix));
    true
}

/// Mask a client secret for display: keep the first 8 chars, replace the rest
/// with `*`. Matches the Python `f"{secret[:8]}{'*' * (len(secret) - 8)}"`.
///
/// Operates on Unicode scalar values to stay close to Python's `str` slicing
/// semantics (Python slices by code point).
pub fn mask_client_secret(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    let prefix: String = chars.iter().take(8).collect();
    let stars = if chars.len() > 8 {
        "*".repeat(chars.len() - 8)
    } else {
        String::new()
    };
    format!("{prefix}{stars}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Mutex;

    // Serialise tests that mutate the process environment.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn base_url_trims_trailing_slashes_and_defaults() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::remove_var("DINGTALK_REGISTRATION_BASE_URL");
        }
        assert_eq!(registration_base_url(), "https://oapi.dingtalk.com");

        unsafe {
            env::set_var("DINGTALK_REGISTRATION_BASE_URL", "https://example.com///");
        }
        assert_eq!(registration_base_url(), "https://example.com");
        unsafe {
            env::remove_var("DINGTALK_REGISTRATION_BASE_URL");
        }
    }

    #[test]
    fn source_defaults_to_openclaw() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            env::remove_var("DINGTALK_REGISTRATION_SOURCE");
        }
        assert_eq!(registration_source(), "openClaw");
        unsafe {
            env::set_var("DINGTALK_REGISTRATION_SOURCE", "custom");
        }
        assert_eq!(registration_source(), "custom");
        unsafe {
            env::remove_var("DINGTALK_REGISTRATION_SOURCE");
        }
    }

    #[test]
    fn parse_init_requires_nonce() {
        assert!(parse_init(&json!({})).is_err());
        assert!(parse_init(&json!({ "nonce": "  " })).is_err());
        assert_eq!(parse_init(&json!({ "nonce": " abc " })).unwrap(), "abc");
        // numeric nonce coerced via str()
        assert_eq!(parse_init(&json!({ "nonce": 123 })).unwrap(), "123");
    }

    #[test]
    fn parse_begin_defaults_and_validation() {
        // missing device_code
        assert!(parse_begin(&json!({ "verification_uri_complete": "x" })).is_err());
        // missing uri
        assert!(parse_begin(&json!({ "device_code": "dc" })).is_err());

        let r = parse_begin(&json!({
            "device_code": " dc ",
            "verification_uri_complete": " https://u ",
        }))
        .unwrap();
        assert_eq!(r.device_code, "dc");
        assert_eq!(r.verification_uri_complete, "https://u");
        assert_eq!(r.expires_in, 7200);
        assert_eq!(r.interval, 3);
    }

    #[test]
    fn parse_begin_interval_floor_is_two() {
        let r = parse_begin(&json!({
            "device_code": "dc",
            "verification_uri_complete": "u",
            "interval": 1,
            "expires_in": 10,
        }))
        .unwrap();
        assert_eq!(r.interval, 2);
        assert_eq!(r.expires_in, 10);

        let r2 = parse_begin(&json!({
            "device_code": "dc",
            "verification_uri_complete": "u",
            "interval": 9,
        }))
        .unwrap();
        assert_eq!(r2.interval, 9);
    }

    #[test]
    fn parse_poll_normalises_status_and_empties() {
        let p = parse_poll(&json!({
            "status": "success",
            "client_id": " cid ",
            "client_secret": "",
            "fail_reason": "  ",
        }));
        assert_eq!(p.status, RegistrationStatus::Success);
        assert_eq!(p.client_id.as_deref(), Some("cid"));
        assert_eq!(p.client_secret, None);
        assert_eq!(p.fail_reason, None);

        let unknown = parse_poll(&json!({ "status": "weird" }));
        assert_eq!(unknown.status, RegistrationStatus::Unknown);

        let missing = parse_poll(&json!({}));
        assert_eq!(missing.status, RegistrationStatus::Unknown);
    }

    fn no_sleep(_d: Duration) {}

    #[test]
    fn wait_success_returns_credentials() {
        let calls = Rc::new(Cell::new(0u32));
        let calls2 = calls.clone();
        let poll = move || {
            let n = calls2.get();
            calls2.set(n + 1);
            if n == 0 {
                Ok(PollResult {
                    status: RegistrationStatus::Waiting,
                    client_id: None,
                    client_secret: None,
                    fail_reason: None,
                })
            } else {
                Ok(PollResult {
                    status: RegistrationStatus::Success,
                    client_id: Some("CID".into()),
                    client_secret: Some("SECRET".into()),
                    fail_reason: None,
                })
            }
        };
        let waiting = Rc::new(Cell::new(0u32));
        let waiting2 = waiting.clone();
        let res = wait_for_registration_success_with(
            0,
            7200,
            poll,
            move || waiting2.set(waiting2.get() + 1),
            Instant::now,
            no_sleep,
        )
        .unwrap();
        assert_eq!(res, ("CID".to_string(), "SECRET".to_string()));
        assert_eq!(waiting.get(), 1);
    }

    #[test]
    fn wait_success_missing_credentials_errors() {
        let poll = || {
            Ok(PollResult {
                status: RegistrationStatus::Success,
                client_id: Some("CID".into()),
                client_secret: None,
                fail_reason: None,
            })
        };
        let err = wait_for_registration_success_with(0, 100, poll, || {}, Instant::now, no_sleep)
            .unwrap_err();
        assert!(err.message.contains("credentials are missing"));
    }

    #[test]
    fn wait_fail_uses_reason_after_retry_window() {
        // Simulate time advancing past the 120s retry window using a manual clock.
        let base = Instant::now();
        let tick = Rc::new(Cell::new(0u64));
        let tick2 = tick.clone();
        let clock = move || base + Duration::from_secs(tick2.get());
        let tick3 = tick.clone();
        let poll = move || {
            // advance well past the retry window each poll
            tick3.set(tick3.get() + 200);
            Ok(PollResult {
                status: RegistrationStatus::Fail,
                client_id: None,
                client_secret: None,
                fail_reason: Some("denied".into()),
            })
        };
        let err = wait_for_registration_success_with(0, 100_000, poll, || {}, clock, no_sleep)
            .unwrap_err();
        assert_eq!(err.message, "authorization failed: denied");
    }

    #[test]
    fn wait_times_out_when_deadline_passes() {
        let poll = || {
            Ok(PollResult {
                status: RegistrationStatus::Waiting,
                client_id: None,
                client_secret: None,
                fail_reason: None,
            })
        };
        // expires_in 0 => deadline == now, loop body never runs.
        let err =
            wait_for_registration_success_with(0, 0, poll, || {}, Instant::now, no_sleep).unwrap_err();
        assert_eq!(err.message, "authorization timed out, please retry");
    }

    #[test]
    fn render_matrix_uses_halfblocks() {
        // 2x2 matrix:
        //   top-left set, bottom-left set -> FULL_BLOCK
        //   top-right set, bottom-right unset -> TOP_HALF
        let m = vec![vec![true, true], vec![true, false]];
        let out = render_matrix(&m);
        let expected = format!("    {}{}", FULL_BLOCK, TOP_HALF);
        assert_eq!(out, expected);
    }

    #[test]
    fn render_matrix_odd_rows() {
        // 3 rows -> 2 output lines; last line has no bottom row.
        let m = vec![
            vec![true, false],
            vec![false, true],
            vec![true, true],
        ];
        let out = render_matrix(&m);
        let lines: Vec<&str> = out.split('\n').collect();
        assert_eq!(lines.len(), 2);
        // last row: top=[true,true], bottom none -> TOP_HALF, TOP_HALF
        let expected_last = format!("    {}{}", TOP_HALF, TOP_HALF);
        assert_eq!(lines[1], expected_last);
    }

    #[test]
    fn render_qr_to_terminal_empty_is_false() {
        assert!(!render_qr_to_terminal(&[]));
        assert!(render_qr_to_terminal(&[vec![true]]));
    }

    #[test]
    fn mask_secret_keeps_prefix() {
        assert_eq!(mask_client_secret("abcdefghIJKL"), "abcdefgh****");
        assert_eq!(mask_client_secret("short"), "short");
        assert_eq!(mask_client_secret("12345678"), "12345678");
    }

    #[test]
    fn json_int_string_and_number() {
        assert_eq!(json_int(&json!({ "a": "42" }), "a", 0), 42);
        assert_eq!(json_int(&json!({ "a": 7 }), "a", 0), 7);
        assert_eq!(json_int(&json!({ "a": "x" }), "a", 5), 5);
        assert_eq!(json_int(&json!({}), "a", 9), 9);
    }
}
