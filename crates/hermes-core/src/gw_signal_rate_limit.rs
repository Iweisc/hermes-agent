//! Signal attachment rate-limit scheduler.
//!
//! Process-wide token-bucket simulator that mirrors the per-account
//! attachment rate limit signal-cli/Signal-Server enforce. Producers
//! (`SignalAdapter.send_multiple_images` and the `send_message` tool's
//! Signal path) call [`SignalAttachmentScheduler::acquire`] before an
//! attachment send; on a 429 they call
//! [`SignalAttachmentScheduler::feedback`] so the model recalibrates from
//! the server's authoritative hint.
//!
//! The scheduler serializes concurrent calls through a `tokio::sync::Mutex`,
//! giving FIFO fairness across agent sessions sharing one signal-cli daemon.
//!
//! Ported from `gateway/platforms/signal_rate_limit.py`.

use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Per-message attachment cap (source: Signal-{Android,Desktop} source code).
pub const SIGNAL_MAX_ATTACHMENTS_PER_MSG: u32 = 32;
/// Server-side token-bucket capacity for attachments rate limiting.
pub const SIGNAL_RATE_LIMIT_BUCKET_CAPACITY: u32 = 50;
/// Fallback token refill interval for signal-cli < v0.14.3.
pub const SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER: u32 = 4;
/// Initial attempt + 1 retry.
pub const SIGNAL_RATE_LIMIT_MAX_ATTEMPTS: u32 = 2;
/// If estimated waiting time > 10s, notify the user about the delay.
pub const SIGNAL_BATCH_PACING_NOTICE_THRESHOLD: f64 = 10.0;
/// signal-cli (v0.14.3+) JSON-RPC error code for RateLimitException.
pub const SIGNAL_RPC_ERROR_RATELIMIT: i64 = -5;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Raised by `SignalAdapter._rpc` for rate-limit responses when the caller
/// has opted in via `raise_on_rate_limit=True`.
///
/// Carries the server-supplied per-token Retry-After (in seconds) on
/// signal-cli >= v0.14.3; `retry_after` is `None` when the version doesn't
/// expose it.
#[derive(Debug, Clone)]
pub struct SignalRateLimitError {
    pub message: String,
    pub retry_after: Option<f64>,
}

impl SignalRateLimitError {
    pub fn new(message: impl Into<String>, retry_after: Option<f64>) -> Self {
        Self {
            message: message.into(),
            retry_after,
        }
    }
}

impl fmt::Display for SignalRateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SignalRateLimitError {}

/// Raised when the scheduler is misused (e.g. requesting more tokens than the
/// bucket's capacity).
#[derive(Debug, Clone)]
pub struct SignalSchedulerError {
    pub message: String,
}

impl SignalSchedulerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SignalSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SignalSchedulerError {}

// ---------------------------------------------------------------------------
// Detection helpers — used to fish a 429 out of signal-cli's various error
// shapes (typed code, [429] substring, libsignal-net RetryLaterException
// leaked through AttachmentInvalidException).
// ---------------------------------------------------------------------------

/// Pull the per-token Retry-After window from a signal-cli rate-limit error.
///
/// Tries two sources, in order:
/// 1. `error.data.response.results[*].retryAfterSeconds` — the structured
///    field signal-cli >= v0.14.3 surfaces for plain RateLimitException.
/// 2. `"Retry after N seconds"` parsed out of the message — covers
///    libsignal-net's RetryLaterException that gets wrapped as
///    AttachmentInvalidException during attachment upload, where the
///    structured field stays null.
///
/// Returns `None` when neither yields a value.
///
/// Mirrors `_extract_retry_after_seconds`. `err` is a JSON-RPC error value
/// (object or anything stringifiable); pass [`Value::String`] for a bare
/// string error.
pub fn extract_retry_after_seconds(err: &Value) -> Option<f64> {
    let msg: String;
    if let Some(obj) = err.as_object() {
        let results = obj
            .get("data")
            .and_then(Value::as_object)
            .and_then(|d| d.get("response"))
            .and_then(Value::as_object)
            .and_then(|r| r.get("results"))
            .and_then(Value::as_array);

        if let Some(results) = results {
            let mut candidates: Vec<f64> = Vec::new();
            for r in results {
                if let Some(robj) = r.as_object() {
                    if let Some(v) = robj.get("retryAfterSeconds") {
                        // Python truthiness: skip 0/null/false/missing.
                        if let Some(n) = json_truthy_number(v) {
                            candidates.push(n);
                        }
                    }
                }
            }
            if !candidates.is_empty() {
                let max = candidates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                return Some(max);
            }
        }

        // str(err.get("message", "")) — None coerces to the string "None" in
        // Python str(), but a missing key yields "". We replicate: missing key
        // -> "", present null -> "None".
        msg = match obj.get("message") {
            Some(Value::Null) => "None".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
    } else {
        msg = value_to_py_str(err);
    }

    parse_retry_after_from_message(&msg)
}

/// `_RETRY_AFTER_RE` equivalent: "Retry after N seconds" (case-insensitive,
/// optional decimal, "second" or "seconds").
fn parse_retry_after_from_message(msg: &str) -> Option<f64> {
    // Match: "retry after" <ws> <digits>[.<digits>] <optional ws> "second"
    let lower = msg.to_ascii_lowercase();
    let needle = "retry after";
    let idx = lower.find(needle)?;
    let rest = &msg[idx + needle.len()..];

    let bytes = rest.as_bytes();
    let mut i = 0;
    // skip whitespace
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let num_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == num_start {
        return None; // need at least one digit
    }
    // optional fractional part
    if i < bytes.len() && bytes[i] == b'.' {
        let frac_start = i + 1;
        let mut j = frac_start;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > frac_start {
            i = j;
        }
    }
    let num_str = &rest[num_start..i];

    // optional whitespace then "second"
    let mut k = i;
    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
        k += 1;
    }
    let tail = &rest[k..];
    if tail.to_ascii_lowercase().starts_with("second") {
        num_str.parse::<f64>().ok()
    } else {
        None
    }
}

/// True if a signal-cli RPC error reflects a rate-limit failure.
///
/// Matches three layers:
/// - typed `RATELIMIT_ERROR` code (signal-cli >= v0.14.3, plain
///   RateLimitException)
/// - legacy `[429]` / `RateLimitException` substrings
/// - libsignal-net's `RetryLaterException` / `Retry after N seconds` surfaced
///   inside `AttachmentInvalidException` when the rate limit is hit during
///   attachment upload — signal-cli never re-tags these as RateLimitException,
///   so substring is the only signal.
///
/// Mirrors `_is_signal_rate_limit_error`.
pub fn is_signal_rate_limit_error(err: &Value) -> bool {
    if let Some(obj) = err.as_object() {
        if let Some(code) = obj.get("code").and_then(Value::as_i64) {
            if code == SIGNAL_RPC_ERROR_RATELIMIT {
                return true;
            }
        }
    }

    let message = if let Some(obj) = err.as_object() {
        match obj.get("message") {
            Some(Value::Null) => "None".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        }
    } else {
        value_to_py_str(err)
    };

    let msg_lower = message.to_lowercase();
    message.contains("[429]")
        || msg_lower.contains("ratelimit")
        || msg_lower.contains("retrylaterexception")
        || msg_lower.contains("retry after")
}

/// Python `str(v)` for non-dict scalar/array JSON values, used so substring
/// matches behave like the original.
fn value_to_py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// Replicate Python truthiness for a JSON number field: returns the numeric
/// value only if it is "truthy" (non-zero number). Non-numbers / null / 0
/// return `None`.
fn json_truthy_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => {
            let f = n.as_f64()?;
            if f != 0.0 {
                Some(f)
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Human-friendly wait label for user-facing pacing notices.
///
/// Mirrors `_format_wait`.
pub fn format_wait(seconds: f64) -> String {
    let s = if seconds > 0.0 { seconds } else { 0.0 };
    if s < 90.0 {
        format!("{}s", py_round(s) as i64)
    } else {
        let mins = py_round(s / 60.0) as i64;
        format!("{} min", mins.max(1))
    }
}

/// HTTP timeout (in seconds) for a Signal `send` RPC.
///
/// signal-cli uploads attachments serially during the call, so the
/// server-side time scales with batch size. Default 30s is fine for text-only
/// sends but truncates large attachment batches mid-upload. Scale at
/// 5s/attachment with a 60s floor.
///
/// Mirrors `_signal_send_timeout`.
pub fn signal_send_timeout(num_attachments: i64) -> f64 {
    if num_attachments <= 0 {
        return 30.0;
    }
    f64::max(60.0, 5.0 * num_attachments as f64)
}

/// Python 3 `round()` semantics: banker's rounding (round-half-to-even).
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // exactly .5 -> round to even
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Mutable bucket state guarded by the scheduler's lock.
struct BucketState {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl BucketState {
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 && self.tokens < self.capacity {
            self.tokens = f64::min(self.capacity, self.tokens + elapsed * self.refill_rate);
        }
        self.last_refill = now;
    }
}

/// Snapshot of scheduler state for diagnostic logging (read-only).
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerState {
    pub tokens: f64,
    pub capacity: i64,
    pub refill_rate: f64,
    pub refill_seconds_per_token: f64,
}

impl SchedulerState {
    /// JSON view matching the Python `state()` dict shape.
    pub fn to_json(&self) -> Value {
        json!({
            "tokens": self.tokens,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "refill_seconds_per_token": self.refill_seconds_per_token,
        })
    }
}

/// Process-wide token-bucket simulator for Signal attachment sends.
///
/// The bucket holds up to `capacity` tokens (default 50, matching Signal's
/// server-side rate-limit bucket size). Each attachment consumes one token.
/// Tokens refill at `refill_rate` tokens/second, calibrated from the per-token
/// Retry-After hint we get from the server when a 429 fires. Until we've
/// observed one, we use the documented default (1 token / 4 seconds).
///
/// Concurrent [`SignalAttachmentScheduler::acquire`] calls serialize through a
/// `tokio::sync::Mutex` — natural FIFO across agent sessions hitting the same
/// daemon.
pub struct SignalAttachmentScheduler {
    inner: Mutex<BucketState>,
}

impl SignalAttachmentScheduler {
    /// Create a scheduler with default capacity / retry-after.
    pub fn new() -> Self {
        Self::with_params(
            SIGNAL_RATE_LIMIT_BUCKET_CAPACITY as f64,
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER as f64,
        )
    }

    /// Create a scheduler with explicit `capacity` (tokens) and
    /// `default_retry_after` (seconds per token).
    pub fn with_params(capacity: f64, default_retry_after: f64) -> Self {
        Self {
            inner: Mutex::new(BucketState {
                capacity,
                tokens: capacity,
                refill_rate: 1.0 / default_retry_after,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Current bucket capacity (tokens).
    pub async fn capacity(&self) -> f64 {
        self.inner.lock().await.capacity
    }

    /// Current refill rate (tokens/second).
    pub async fn refill_rate(&self) -> f64 {
        self.inner.lock().await.refill_rate
    }

    /// Best-effort estimate of the seconds until `n` tokens would be
    /// available. Used to decide whether to emit a user-facing pacing notice
    /// *before* committing to an `acquire` that may block silently.
    ///
    /// Mirrors `estimate_wait`. Acquires the lock briefly to read state; small
    /// races vs. concurrent acquires are benign for an informational notice.
    pub async fn estimate_wait(&self, n: i64) -> f64 {
        let state = self.inner.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let mut projected = state.tokens;
        if elapsed > 0.0 && projected < state.capacity {
            projected = f64::min(state.capacity, projected + elapsed * state.refill_rate);
        }
        let deficit = n as f64 - projected;
        if deficit <= 0.0 {
            return 0.0;
        }
        deficit / state.refill_rate
    }

    /// Block until at least `n` tokens are available, return the seconds slept.
    ///
    /// Does **not** deduct tokens — the bucket is a read-only model of
    /// server-side capacity. Call [`SignalAttachmentScheduler::report_rpc_duration`]
    /// after the RPC to synchronise the model with the server timeline.
    ///
    /// The lock is released during the sleep so other callers can interleave.
    /// A retry loop re-checks after each sleep in case the deadline was
    /// pessimistic.
    ///
    /// Mirrors `acquire`.
    pub async fn acquire(&self, n: i64) -> Result<f64, SignalSchedulerError> {
        if n <= 0 {
            return Ok(0.0);
        }

        // capacity check (read under lock to honour any feedback() change)
        {
            let state = self.inner.lock().await;
            if n as f64 > state.capacity {
                return Err(SignalSchedulerError::new(format!(
                    "Signal scheduler was called requesting {} tokens (max is {})",
                    n, state.capacity
                )));
            }
        }

        let mut total_slept = 0.0;
        let mut first_pass = true;
        loop {
            let deficit;
            {
                let mut state = self.inner.lock().await;
                state.refill();
                if state.tokens >= n as f64 {
                    if !first_pass || total_slept > 0.0 {
                        log::debug!(
                            "Signal scheduler: tokens sufficient for {} (remaining={:.1}, total_slept={:.1}s)",
                            n,
                            state.tokens,
                            total_slept,
                        );
                    }
                    return Ok(total_slept);
                }
                deficit = n as f64 - state.tokens;
                let wait = deficit / state.refill_rate;
                if first_pass {
                    log::info!(
                        "Signal scheduler: pausing {:.1}s for {} tokens (available={:.1}, deficit={:.1}, refill={:.4}/s ~ {:.1}s/token)",
                        wait,
                        n,
                        state.tokens,
                        deficit,
                        state.refill_rate,
                        1.0 / state.refill_rate,
                    );
                    first_pass = false;
                }
                // hold `wait` while lock is dropped below
            }
            // Recompute wait outside the lock using the deficit/rate captured
            // above. Re-read rate to be safe under feedback().
            let wait = {
                let state = self.inner.lock().await;
                deficit / state.refill_rate
            };
            sleep_secs(wait).await;
            total_slept += wait;
        }
    }

    /// Record an attachment-send RPC that just completed.
    ///
    /// Deducts `n_attachments` tokens without crediting refill during the
    /// upload window. Signal's server checks the bucket at RPC start and does
    /// *not* refill during request processing — refill resumes after the
    /// response. Crediting upload-time refill causes cumulative drift that
    /// eventually triggers 429s.
    ///
    /// Advances `last_refill` so the next `acquire` / refill starts counting
    /// from this point.
    ///
    /// Mirrors `report_rpc_duration`.
    pub async fn report_rpc_duration(&self, rpc_duration: f64, n_attachments: i64) {
        if n_attachments <= 0 {
            return;
        }

        let (token_before, token_after, refill_rate);
        {
            let mut state = self.inner.lock().await;
            let now = Instant::now();
            token_before = state.tokens;
            state.tokens = f64::max(0.0, token_before - n_attachments as f64);
            state.last_refill = now;
            token_after = state.tokens;
            refill_rate = state.refill_rate;
        }

        let msg = format!(
            "Signal scheduler: RPC for {} att took {:.1}s — tokens {:.1} -> {:.1} (deducted={}, no upload refill credited, refill={:.4}s^-1)",
            n_attachments, rpc_duration, token_before, token_after, n_attachments, refill_rate,
        );
        if rpc_duration > 10.0 && n_attachments > 5 {
            log::info!("{}", msg);
        } else {
            log::debug!("{}", msg);
        }
    }

    /// Apply server feedback after a 429.
    ///
    /// `retry_after` is the per-*token* refill window the server reports
    /// (`None` when signal-cli is older than v0.14.3 and didn't surface it).
    ///
    /// When present we calibrate `refill_rate` from it: the server is
    /// authoritative.
    ///
    /// Mirrors `feedback`.
    pub async fn feedback(&self, retry_after: Option<f64>, _n_attempted: i64) {
        let mut state = self.inner.lock().await;
        if let Some(ra) = retry_after {
            if ra > 0.0 {
                let new_rate = 1.0 / ra;
                if new_rate != state.refill_rate {
                    log::info!(
                        "Signal scheduler: calibrating refill_rate to {:.4} tokens/sec (server retry_after={:.1}s per token)",
                        new_rate,
                        ra,
                    );
                    state.refill_rate = new_rate;
                }
            }
        }
        state.tokens = 0.0;
        state.last_refill = Instant::now();
    }

    /// Return current scheduler state for diagnostic logging (read-only).
    ///
    /// Does not advance `last_refill` — safe to call from logging paths
    /// without perturbing the bucket.
    ///
    /// Mirrors `state`.
    pub async fn state(&self) -> SchedulerState {
        let state = self.inner.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let mut projected = state.tokens;
        if elapsed > 0.0 && projected < state.capacity {
            projected = f64::min(state.capacity, projected + elapsed * state.refill_rate);
        }
        let refill_seconds_per_token = if state.refill_rate > 0.0 {
            round1(1.0 / state.refill_rate)
        } else {
            f64::INFINITY
        };
        SchedulerState {
            tokens: round1(projected),
            capacity: state.capacity as i64,
            refill_rate: round4(state.refill_rate),
            refill_seconds_per_token,
        }
    }
}

impl Default for SignalAttachmentScheduler {
    fn default() -> Self {
        Self::new()
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// Sleep for `secs` seconds (clamped to non-negative). Extracted so it can be
/// shimmed; uses `tokio::time::sleep`.
async fn sleep_secs(secs: f64) {
    let secs = if secs > 0.0 { secs } else { 0.0 };
    tokio::time::sleep(Duration::from_secs_f64(secs)).await;
}

// ---------------------------------------------------------------------------
// Process-wide singleton
// ---------------------------------------------------------------------------

static SCHEDULER: OnceLock<Arc<SignalAttachmentScheduler>> = OnceLock::new();

/// Return the process-wide scheduler, creating it on first access.
///
/// Mirrors `get_scheduler`.
pub fn get_scheduler() -> Arc<SignalAttachmentScheduler> {
    SCHEDULER
        .get_or_init(|| {
            let sched = SignalAttachmentScheduler::new();
            log::info!(
                "Signal scheduler: created (capacity={} tokens, refill={:.4}/s ~ {:.1}s/token)",
                SIGNAL_RATE_LIMIT_BUCKET_CAPACITY,
                1.0 / SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER as f64,
                SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER as f64,
            );
            Arc::new(sched)
        })
        .clone()
}

// Note: the Python `_reset_scheduler` is test-only and relies on rebinding a
// module global. `OnceLock` cannot be reset once initialized; tests here
// construct fresh `SignalAttachmentScheduler` instances directly instead.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_retry_after_from_structured_field() {
        let err = json!({
            "data": {"response": {"results": [
                {"retryAfterSeconds": 4},
                {"retryAfterSeconds": 7},
            ]}}
        });
        assert_eq!(extract_retry_after_seconds(&err), Some(7.0));
    }

    #[test]
    fn extract_retry_after_skips_zero_and_missing() {
        let err = json!({
            "data": {"response": {"results": [
                {"retryAfterSeconds": 0},
                {"foo": 1},
                {"retryAfterSeconds": null},
            ]}},
            "message": "Retry after 5 seconds"
        });
        // No truthy structured candidate -> falls back to message parse.
        assert_eq!(extract_retry_after_seconds(&err), Some(5.0));
    }

    #[test]
    fn extract_retry_after_from_message_string() {
        let err = Value::String("RetryLaterException: Retry after 12 seconds".into());
        assert_eq!(extract_retry_after_seconds(&err), Some(12.0));
    }

    #[test]
    fn extract_retry_after_decimal_and_singular() {
        let err = Value::String("Retry after 4.5 second".into());
        assert_eq!(extract_retry_after_seconds(&err), Some(4.5));
    }

    #[test]
    fn extract_retry_after_case_insensitive() {
        let err = Value::String("RETRY AFTER 3 SECONDS".into());
        assert_eq!(extract_retry_after_seconds(&err), Some(3.0));
    }

    #[test]
    fn extract_retry_after_none_when_absent() {
        let err = Value::String("some unrelated error".into());
        assert_eq!(extract_retry_after_seconds(&err), None);
    }

    #[test]
    fn is_rate_limit_typed_code() {
        let err = json!({"code": SIGNAL_RPC_ERROR_RATELIMIT, "message": "x"});
        assert!(is_signal_rate_limit_error(&err));
    }

    #[test]
    fn is_rate_limit_substrings() {
        assert!(is_signal_rate_limit_error(&Value::String("[429] too many".into())));
        assert!(is_signal_rate_limit_error(&Value::String("RateLimitException".into())));
        assert!(is_signal_rate_limit_error(&Value::String(
            "RetryLaterException blah".into()
        )));
        assert!(is_signal_rate_limit_error(&Value::String(
            "please Retry after 4 seconds".into()
        )));
    }

    #[test]
    fn is_rate_limit_negative() {
        assert!(!is_signal_rate_limit_error(&Value::String("ordinary failure".into())));
        let err = json!({"code": -1, "message": "nope"});
        assert!(!is_signal_rate_limit_error(&err));
    }

    #[test]
    fn format_wait_seconds_and_minutes() {
        assert_eq!(format_wait(-3.0), "0s");
        assert_eq!(format_wait(0.0), "0s");
        assert_eq!(format_wait(42.4), "42s");
        assert_eq!(format_wait(89.0), "89s");
        assert_eq!(format_wait(90.0), "2 min"); // 90/60 = 1.5 -> banker's round to 2
        assert_eq!(format_wait(120.0), "2 min");
        assert_eq!(format_wait(95.0), "2 min");
    }

    #[test]
    fn send_timeout_scaling() {
        assert_eq!(signal_send_timeout(0), 30.0);
        assert_eq!(signal_send_timeout(-5), 30.0);
        assert_eq!(signal_send_timeout(1), 60.0); // floor
        assert_eq!(signal_send_timeout(20), 100.0); // 5 * 20
    }

    #[tokio::test]
    async fn acquire_returns_immediately_when_full() {
        let s = SignalAttachmentScheduler::new();
        let slept = s.acquire(10).await.unwrap();
        assert_eq!(slept, 0.0);
    }

    #[tokio::test]
    async fn acquire_zero_or_negative_is_noop() {
        let s = SignalAttachmentScheduler::new();
        assert_eq!(s.acquire(0).await.unwrap(), 0.0);
        assert_eq!(s.acquire(-3).await.unwrap(), 0.0);
    }

    #[tokio::test]
    async fn acquire_rejects_over_capacity() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        let err = s.acquire(51).await.unwrap_err();
        assert!(err.message.contains("51"));
        assert!(err.message.contains("50"));
    }

    #[tokio::test]
    async fn report_rpc_duration_deducts_tokens() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        s.report_rpc_duration(1.0, 10).await;
        let st = s.state().await;
        // tokens were 50, deduct 10 -> 40 (modulo tiny refill in the gap)
        assert!(st.tokens >= 40.0 && st.tokens <= 40.5, "tokens={}", st.tokens);
    }

    #[tokio::test]
    async fn report_rpc_duration_floors_at_zero() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        s.report_rpc_duration(1.0, 100).await;
        let st = s.state().await;
        assert!(st.tokens >= 0.0);
    }

    #[tokio::test]
    async fn feedback_calibrates_rate_and_zeroes_tokens() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        assert!((s.refill_rate().await - 0.25).abs() < 1e-9);
        s.feedback(Some(10.0), 5).await;
        assert!((s.refill_rate().await - 0.1).abs() < 1e-9);
        let st = s.state().await;
        assert_eq!(st.tokens, 0.0);
        assert_eq!(st.refill_rate, 0.1);
        assert_eq!(st.refill_seconds_per_token, 10.0);
    }

    #[tokio::test]
    async fn feedback_none_only_zeroes_tokens() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        s.feedback(None, 3).await;
        assert!((s.refill_rate().await - 0.25).abs() < 1e-9);
        let st = s.state().await;
        assert_eq!(st.tokens, 0.0);
    }

    #[tokio::test]
    async fn estimate_wait_zero_when_full() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        assert_eq!(s.estimate_wait(10).await, 0.0);
    }

    #[tokio::test]
    async fn estimate_wait_positive_after_drain() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        s.feedback(None, 0).await; // tokens -> 0, rate 0.25/s
        let w = s.estimate_wait(10).await;
        // need ~10 tokens at 0.25/s = ~40s
        assert!(w > 35.0 && w <= 40.5, "wait={}", w);
    }

    #[tokio::test]
    async fn state_shape_matches_python() {
        let s = SignalAttachmentScheduler::with_params(50.0, 4.0);
        let v = s.state().await.to_json();
        assert_eq!(v["capacity"], 50);
        assert_eq!(v["refill_rate"], 0.25);
        assert_eq!(v["refill_seconds_per_token"], 4.0);
    }

    #[test]
    fn singleton_is_stable() {
        let a = get_scheduler();
        let b = get_scheduler();
        assert!(Arc::ptr_eq(&a, &b));
    }
}
