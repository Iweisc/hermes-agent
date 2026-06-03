//! Cross-session rate limit guard for Nous Portal.
//!
//! Native Rust port of `agent/nous_rate_guard.py`.
//!
//! Writes rate limit state to a shared file so all sessions (CLI, gateway,
//! cron, auxiliary) can check whether Nous Portal is currently rate-limited
//! before making requests. Prevents retry amplification when RPH is tapped.
//!
//! Each 429 from Nous triggers up to 9 API calls per conversation turn
//! (3 SDK retries x 3 Hermes retries), and every one of those calls counts
//! against RPH. By recording the rate limit state on first 429 and checking
//! it before subsequent attempts, we eliminate the amplification effect.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// HTTP response headers, mirroring Python's `Mapping[str, str]`.
pub type HeaderMap = HashMap<String, String>;

const STATE_SUBDIR: &str = "rate_limits";
const STATE_FILENAME: &str = "nous.json";

/// Buckets with reset windows shorter than this are treated as transient
/// (upstream jitter, secondary throttling) rather than a genuine quota
/// exhaustion worth a cross-session breaker trip.
const MIN_RESET_FOR_BREAKER_SECONDS: f64 = 60.0;

/// Default cooldown (seconds) when no reset info is available. Mirrors the
/// Python `default_cooldown` keyword default of 300.0.
pub const DEFAULT_COOLDOWN: f64 = 300.0;

// ---------------------------------------------------------------------------
// Time + path helpers
// ---------------------------------------------------------------------------

/// Current unix time in seconds (float), mirroring Python `time.time()`.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Resolve the active HERMES_HOME (profile-aware) without circular imports.
///
/// Mirrors `get_hermes_home()` with the Python ImportError fallback: honour
/// `HERMES_HOME` when set & non-empty, otherwise fall back to `~/.hermes`.
fn hermes_home_path() -> PathBuf {
    if let Ok(val) = env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".hermes")
}

/// Path to the Nous rate limit state file.
fn state_path() -> PathBuf {
    hermes_home_path().join(STATE_SUBDIR).join(STATE_FILENAME)
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

/// Lowercase all keys of a header map, mirroring `{k.lower(): v ...}`.
fn lowered(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.clone()))
        .collect()
}

/// Extract the best available reset-time estimate from response headers.
///
/// Priority:
///   1. x-ratelimit-reset-requests-1h  (hourly RPH window — most useful)
///   2. x-ratelimit-reset-requests     (per-minute RPM window)
///   3. retry-after                     (generic HTTP header)
///
/// Returns seconds-from-now, or `None` if no usable header found.
fn parse_reset_seconds(headers: Option<&HeaderMap>) -> Option<f64> {
    let headers = headers?;
    if headers.is_empty() {
        return None;
    }
    let low = lowered(headers);
    for key in [
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-requests",
        "retry-after",
    ] {
        if let Some(raw) = low.get(key) {
            if let Ok(val) = raw.trim().parse::<f64>() {
                if val > 0.0 {
                    return Some(val);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Recording / querying
// ---------------------------------------------------------------------------

/// Record that Nous Portal is rate-limited.
///
/// Parses the reset time from response headers or error context. Falls back to
/// `default_cooldown` (5 minutes) if no reset info is available. Writes to a
/// shared file that all sessions can read.
pub fn record_nous_rate_limit(
    headers: Option<&HeaderMap>,
    error_context: Option<&Value>,
    default_cooldown: f64,
) {
    let now = now_secs();
    let mut reset_at: Option<f64> = None;

    // Try headers first (most accurate).
    if let Some(secs) = parse_reset_seconds(headers) {
        reset_at = Some(now + secs);
    }

    // Try error_context reset_at (from body parsing).
    if reset_at.is_none() {
        if let Some(ctx) = error_context {
            if let Some(ctx_reset) = ctx.get("reset_at").and_then(Value::as_f64) {
                if ctx_reset > now {
                    reset_at = Some(ctx_reset);
                }
            }
        }
    }

    // Default cooldown.
    let reset_at = reset_at.unwrap_or(now + default_cooldown);

    let path = state_path();
    let Some(state_dir) = path.parent().map(|p| p.to_path_buf()) else {
        log::debug!("Failed to write Nous rate limit state: no parent dir");
        return;
    };

    let state = serde_json::json!({
        "reset_at": reset_at,
        "recorded_at": now,
        "reset_seconds": reset_at - now,
    });

    match write_atomic(&state_dir, &path, &state) {
        Ok(()) => {
            log::info!(
                "Nous rate limit recorded: resets in {:.0}s (at {:.0})",
                reset_at - now,
                reset_at,
            );
        }
        Err(exc) => {
            log::debug!("Failed to write Nous rate limit state: {exc}");
        }
    }
}

/// Atomic write: write to a temp file in `state_dir`, then rename over `path`.
///
/// Mirrors Python's `tempfile.mkstemp(dir=...) + atomic_replace(...)`: the
/// rename is atomic on the same filesystem (like `os.replace`). On failure the
/// temp file is cleaned up before the error propagates.
fn write_atomic(state_dir: &PathBuf, path: &PathBuf, state: &Value) -> std::io::Result<()> {
    fs::create_dir_all(state_dir)?;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_path = state_dir.join(format!(".nous-{}-{}.tmp", process::id(), nanos));

    let write_then_rename = || -> std::io::Result<()> {
        let bytes = serde_json::to_vec(state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.flush()?;
        }
        fs::rename(&tmp_path, path)?;
        Ok(())
    };

    match write_then_rename() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Check if Nous Portal is currently rate-limited.
///
/// Returns seconds remaining until reset, or `None` if not rate-limited.
pub fn nous_rate_limit_remaining() -> Option<f64> {
    let path = state_path();
    let contents = fs::read_to_string(&path).ok()?;
    let state: Value = serde_json::from_str(&contents).ok()?;

    // Python: state.get("reset_at", 0)
    let reset_at = state.get("reset_at").and_then(Value::as_f64).unwrap_or(0.0);
    let remaining = reset_at - now_secs();
    if remaining > 0.0 {
        return Some(remaining);
    }
    // Expired — clean up.
    let _ = fs::remove_file(&path);
    None
}

/// Clear the rate limit state (e.g. after a successful Nous request).
pub fn clear_nous_rate_limit() {
    let path = state_path();
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => log::debug!("Failed to clear Nous rate limit state: {e}"),
    }
}

/// Format seconds remaining into human-readable duration.
pub fn format_remaining(seconds: f64) -> String {
    let s = if seconds < 0.0 { 0 } else { seconds as i64 };
    if s < 60 {
        return format!("{s}s");
    }
    if s < 3600 {
        let m = s / 60;
        let sec = s % 60;
        return if sec != 0 {
            format!("{m}m {sec}s")
        } else {
            format!("{m}m")
        };
    }
    let h = s / 3600;
    let remainder = s % 3600;
    let m = remainder / 60;
    if m != 0 {
        format!("{h}h {m}m")
    } else {
        format!("{h}h")
    }
}

// ---------------------------------------------------------------------------
// Genuine-rate-limit classification
// ---------------------------------------------------------------------------

/// A single rate-limit bucket snapshot, mirroring the duck-typed bucket objects
/// on `RateLimitState` from `agent.rate_limit_tracker`.
#[derive(Debug, Clone, Default)]
pub struct BucketLike {
    pub limit: i64,
    pub remaining: i64,
    /// Adjusted "remaining_seconds_now" property when present.
    pub remaining_seconds_now: Option<f64>,
    /// Raw reset window.
    pub reset_seconds: f64,
}

/// Last-known-good rate-limit state, mirroring the `RateLimitState` dataclass
/// (buckets exposed as attributes). Any bucket may be absent.
#[derive(Debug, Clone, Default)]
pub struct RateLimitStateLike {
    pub requests_min: Option<BucketLike>,
    pub requests_hour: Option<BucketLike>,
    pub tokens_min: Option<BucketLike>,
    pub tokens_hour: Option<BucketLike>,
}

/// Decide whether a 429 from Nous Portal is a real account rate limit.
///
/// Returns `true` when the evidence points at a genuine, caller-scoped quota
/// exhaustion (an exhausted bucket with a meaningful reset window), as opposed
/// to transient upstream-provider throttling.
pub fn is_genuine_nous_rate_limit(
    headers: Option<&HeaderMap>,
    last_known_state: Option<&RateLimitStateLike>,
) -> bool {
    // Signal 1: current 429 response headers.
    let buckets = parse_buckets_from_headers(headers);
    if has_exhausted_bucket(&buckets) {
        return true;
    }

    // Signal 2: last-known-good state from a recent successful response.
    if let Some(state) = last_known_state {
        if has_exhausted_bucket_in_object(state) {
            return true;
        }
    }

    false
}

/// Per-bucket `(remaining, reset_seconds)` extracted from `x-ratelimit-*`
/// headers. Returns an empty map when no rate-limit headers are present.
fn parse_buckets_from_headers(
    headers: Option<&HeaderMap>,
) -> HashMap<String, (Option<i64>, Option<f64>)> {
    let mut result: HashMap<String, (Option<i64>, Option<f64>)> = HashMap::new();
    let Some(headers) = headers else {
        return result;
    };
    if headers.is_empty() {
        return result;
    }
    let low = lowered(headers);
    if !low.keys().any(|k| k.starts_with("x-ratelimit-")) {
        return result;
    }

    // Python: int(float(raw)) — parse as float then truncate toward zero.
    let maybe_int = |raw: Option<&String>| -> Option<i64> {
        raw.and_then(|r| r.trim().parse::<f64>().ok())
            .map(|f| f.trunc() as i64)
    };
    let maybe_float =
        |raw: Option<&String>| -> Option<f64> { raw.and_then(|r| r.trim().parse::<f64>().ok()) };

    for tag in ["requests", "requests-1h", "tokens", "tokens-1h"] {
        let remaining = maybe_int(low.get(&format!("x-ratelimit-remaining-{tag}")));
        let reset = maybe_float(low.get(&format!("x-ratelimit-reset-{tag}")));
        if remaining.is_some() || reset.is_some() {
            result.insert(tag.to_string(), (remaining, reset));
        }
    }
    result
}

/// True when any bucket has `remaining == 0` AND a meaningful reset window.
fn has_exhausted_bucket(buckets: &HashMap<String, (Option<i64>, Option<f64>)>) -> bool {
    for (remaining, reset) in buckets.values() {
        match remaining {
            Some(r) if *r <= 0 => {}
            _ => continue, // None or > 0
        }
        let Some(reset) = reset else { continue };
        if *reset >= MIN_RESET_FOR_BREAKER_SECONDS {
            return true;
        }
    }
    false
}

/// Check a `RateLimitState`-like object for an exhausted bucket.
fn has_exhausted_bucket_in_object(state: &RateLimitStateLike) -> bool {
    let buckets = [
        &state.requests_min,
        &state.requests_hour,
        &state.tokens_min,
        &state.tokens_hour,
    ];
    for bucket in buckets {
        let Some(bucket) = bucket else { continue };
        let limit = bucket.limit;
        let remaining = bucket.remaining;
        // Prefer the adjusted "remaining_seconds_now" when present.
        let reset = bucket.remaining_seconds_now.unwrap_or(bucket.reset_seconds);
        if limit <= 0 {
            continue;
        }
        if remaining > 0 {
            continue;
        }
        if reset >= MIN_RESET_FOR_BREAKER_SECONDS {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // HERMES_HOME is process-global; serialize tests that mutate it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TempHome {
        _guard: std::sync::MutexGuard<'static, ()>,
        dir: PathBuf,
        prev: Option<String>,
    }

    impl TempHome {
        fn new() -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = env::temp_dir().join(format!("nous-rg-test-{}-{}", process::id(), nanos));
            fs::create_dir_all(&dir).unwrap();
            let prev = env::var("HERMES_HOME").ok();
            unsafe { env::set_var("HERMES_HOME", &dir); }
            TempHome {
                _guard: guard,
                dir,
                prev,
            }
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            // SAFETY: single-threaded test teardown.
            unsafe {
                match &self.prev {
                    Some(v) => env::set_var("HERMES_HOME", v),
                    None => env::remove_var("HERMES_HOME"),
                }
            }
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn hmap(pairs: &[(&str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parse_reset_seconds_priority() {
        let h = hmap(&[
            ("X-RateLimit-Reset-Requests-1h", "120"),
            ("x-ratelimit-reset-requests", "30"),
            ("retry-after", "10"),
        ]);
        assert_eq!(parse_reset_seconds(Some(&h)), Some(120.0));

        let h = hmap(&[("x-ratelimit-reset-requests", "45"), ("retry-after", "10")]);
        assert_eq!(parse_reset_seconds(Some(&h)), Some(45.0));

        let h = hmap(&[("Retry-After", "7")]);
        assert_eq!(parse_reset_seconds(Some(&h)), Some(7.0));

        // Non-positive / garbage values are skipped.
        let h = hmap(&[("retry-after", "0"), ("x-ratelimit-reset-requests", "nope")]);
        assert_eq!(parse_reset_seconds(Some(&h)), None);

        assert_eq!(parse_reset_seconds(None), None);
        assert_eq!(parse_reset_seconds(Some(&hmap(&[]))), None);
    }

    #[test]
    fn record_and_remaining_roundtrip() {
        let _home = TempHome::new();
        let h = hmap(&[("retry-after", "120")]);
        record_nous_rate_limit(Some(&h), None, DEFAULT_COOLDOWN);

        let rem = nous_rate_limit_remaining().expect("should be rate limited");
        // Roughly 120s minus a tiny bit.
        assert!(rem > 100.0 && rem <= 121.0, "rem={rem}");
    }

    #[test]
    fn record_uses_default_cooldown_when_no_signal() {
        let _home = TempHome::new();
        record_nous_rate_limit(None, None, 300.0);
        let rem = nous_rate_limit_remaining().expect("should be rate limited");
        assert!(rem > 290.0 && rem <= 301.0, "rem={rem}");
    }

    #[test]
    fn record_uses_error_context_reset_at() {
        let _home = TempHome::new();
        let future = now_secs() + 200.0;
        let ctx = serde_json::json!({ "reset_at": future });
        record_nous_rate_limit(None, Some(&ctx), 300.0);
        let rem = nous_rate_limit_remaining().expect("should be rate limited");
        assert!(rem > 190.0 && rem <= 201.0, "rem={rem}");
    }

    #[test]
    fn error_context_in_past_falls_back_to_cooldown() {
        let _home = TempHome::new();
        let past = now_secs() - 50.0;
        let ctx = serde_json::json!({ "reset_at": past });
        record_nous_rate_limit(None, Some(&ctx), 300.0);
        let rem = nous_rate_limit_remaining().expect("should be rate limited");
        assert!(rem > 290.0 && rem <= 301.0, "rem={rem}");
    }

    #[test]
    fn expired_state_is_cleaned_up() {
        let _home = TempHome::new();
        // Write a state already in the past via error_context being ignored;
        // instead write directly with a tiny negative cooldown.
        record_nous_rate_limit(None, None, -10.0);
        assert!(nous_rate_limit_remaining().is_none());
        // File should be gone after the expired read.
        assert!(!state_path().exists());
    }

    #[test]
    fn clear_removes_state() {
        let _home = TempHome::new();
        record_nous_rate_limit(None, None, 300.0);
        assert!(state_path().exists());
        clear_nous_rate_limit();
        assert!(!state_path().exists());
        // Idempotent: clearing again is fine.
        clear_nous_rate_limit();
    }

    #[test]
    fn remaining_none_when_no_file() {
        let _home = TempHome::new();
        assert!(nous_rate_limit_remaining().is_none());
    }

    #[test]
    fn format_remaining_buckets() {
        assert_eq!(format_remaining(-5.0), "0s");
        assert_eq!(format_remaining(0.0), "0s");
        assert_eq!(format_remaining(45.0), "45s");
        assert_eq!(format_remaining(59.9), "59s");
        assert_eq!(format_remaining(60.0), "1m");
        assert_eq!(format_remaining(90.0), "1m 30s");
        assert_eq!(format_remaining(3600.0), "1h");
        assert_eq!(format_remaining(3660.0), "1h 1m");
        assert_eq!(format_remaining(7320.0), "2h 2m");
    }

    #[test]
    fn genuine_when_header_bucket_exhausted() {
        let h = hmap(&[
            ("x-ratelimit-remaining-requests-1h", "0"),
            ("x-ratelimit-reset-requests-1h", "120"),
        ]);
        assert!(is_genuine_nous_rate_limit(Some(&h), None));
    }

    #[test]
    fn transient_short_window_is_not_genuine() {
        // remaining == 0 but reset window < 60s -> transient.
        let h = hmap(&[
            ("x-ratelimit-remaining-requests", "0"),
            ("x-ratelimit-reset-requests", "5"),
        ]);
        assert!(!is_genuine_nous_rate_limit(Some(&h), None));
    }

    #[test]
    fn remaining_above_zero_is_not_genuine() {
        let h = hmap(&[
            ("x-ratelimit-remaining-requests", "3"),
            ("x-ratelimit-reset-requests", "120"),
        ]);
        assert!(!is_genuine_nous_rate_limit(Some(&h), None));
    }

    #[test]
    fn no_ratelimit_headers_short_circuits() {
        let h = hmap(&[("content-type", "application/json"), ("retry-after", "120")]);
        // retry-after alone does not start with x-ratelimit-, so no buckets.
        let buckets = parse_buckets_from_headers(Some(&h));
        assert!(buckets.is_empty());
        assert!(!is_genuine_nous_rate_limit(Some(&h), None));
    }

    #[test]
    fn float_remaining_truncates_like_python_int() {
        // int(float("0.9")) == 0 -> exhausted.
        let h = hmap(&[
            ("x-ratelimit-remaining-tokens-1h", "0.9"),
            ("x-ratelimit-reset-tokens-1h", "300"),
        ]);
        let buckets = parse_buckets_from_headers(Some(&h));
        assert_eq!(buckets.get("tokens-1h"), Some(&(Some(0), Some(300.0))));
        assert!(is_genuine_nous_rate_limit(Some(&h), None));
    }

    #[test]
    fn genuine_from_last_known_state() {
        let state = RateLimitStateLike {
            requests_hour: Some(BucketLike {
                limit: 1000,
                remaining: 0,
                remaining_seconds_now: Some(180.0),
                reset_seconds: 0.0,
            }),
            ..Default::default()
        };
        assert!(is_genuine_nous_rate_limit(None, Some(&state)));
    }

    #[test]
    fn last_known_state_falls_back_to_reset_seconds() {
        let state = RateLimitStateLike {
            tokens_min: Some(BucketLike {
                limit: 500,
                remaining: 0,
                remaining_seconds_now: None,
                reset_seconds: 90.0,
            }),
            ..Default::default()
        };
        assert!(is_genuine_nous_rate_limit(None, Some(&state)));
    }

    #[test]
    fn last_known_state_zero_limit_skipped() {
        let state = RateLimitStateLike {
            requests_min: Some(BucketLike {
                limit: 0,
                remaining: 0,
                remaining_seconds_now: Some(300.0),
                reset_seconds: 300.0,
            }),
            ..Default::default()
        };
        assert!(!is_genuine_nous_rate_limit(None, Some(&state)));
    }

    #[test]
    fn last_known_state_short_window_skipped() {
        let state = RateLimitStateLike {
            requests_min: Some(BucketLike {
                limit: 100,
                remaining: 0,
                remaining_seconds_now: Some(10.0),
                reset_seconds: 10.0,
            }),
            ..Default::default()
        };
        assert!(!is_genuine_nous_rate_limit(None, Some(&state)));
    }
}
