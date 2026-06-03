//! Rate limit tracking for inference API responses.
//!
//! Captures `x-ratelimit-*` headers from provider responses and provides
//! formatted display for the `/usage` slash command. Currently supports the
//! Nous Portal header format (also used by OpenRouter and OpenAI-compatible
//! APIs that follow the same convention).
//!
//! Header schema (12 headers total):
//!   x-ratelimit-limit-requests          RPM cap
//!   x-ratelimit-limit-requests-1h       RPH cap
//!   x-ratelimit-limit-tokens            TPM cap
//!   x-ratelimit-limit-tokens-1h         TPH cap
//!   x-ratelimit-remaining-requests      requests left in minute window
//!   x-ratelimit-remaining-requests-1h   requests left in hour window
//!   x-ratelimit-remaining-tokens        tokens left in minute window
//!   x-ratelimit-remaining-tokens-1h     tokens left in hour window
//!   x-ratelimit-reset-requests          seconds until minute request window resets
//!   x-ratelimit-reset-requests-1h       seconds until hour request window resets
//!   x-ratelimit-reset-tokens            seconds until minute token window resets
//!   x-ratelimit-reset-tokens-1h         seconds until hour token window resets
//!
//! Faithful port of `agent/rate_limit_tracker.py`.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Current wall-clock time as epoch seconds (mirrors Python `time.time()`).
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// One rate-limit window (e.g. requests per minute).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RateLimitBucket {
    pub limit: i64,
    pub remaining: i64,
    pub reset_seconds: f64,
    /// `now_secs()` when this was captured.
    pub captured_at: f64,
}

impl RateLimitBucket {
    pub fn used(&self) -> i64 {
        (self.limit - self.remaining).max(0)
    }

    pub fn usage_pct(&self) -> f64 {
        if self.limit <= 0 {
            return 0.0;
        }
        (self.used() as f64 / self.limit as f64) * 100.0
    }

    /// Estimated seconds remaining until reset, adjusted for elapsed time.
    pub fn remaining_seconds_now(&self) -> f64 {
        let elapsed = now_secs() - self.captured_at;
        (self.reset_seconds - elapsed).max(0.0)
    }
}

/// Full rate-limit state parsed from response headers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RateLimitState {
    pub requests_min: RateLimitBucket,
    pub requests_hour: RateLimitBucket,
    pub tokens_min: RateLimitBucket,
    pub tokens_hour: RateLimitBucket,
    /// When the headers were captured.
    pub captured_at: f64,
    pub provider: String,
}

impl RateLimitState {
    pub fn has_data(&self) -> bool {
        self.captured_at > 0.0
    }

    pub fn age_seconds(&self) -> f64 {
        if !self.has_data() {
            return f64::INFINITY;
        }
        now_secs() - self.captured_at
    }
}

/// Mirror of Python `int(float(value))`: parse as float, truncate toward zero.
fn safe_int(value: Option<&str>) -> i64 {
    match value {
        Some(text) => match text.trim().parse::<f64>() {
            Ok(parsed) if parsed.is_finite() => parsed.trunc() as i64,
            _ => 0,
        },
        None => 0,
    }
}

/// Mirror of Python `float(value)`.
fn safe_float(value: Option<&str>) -> f64 {
    match value {
        Some(text) => text.trim().parse::<f64>().ok().filter(|v| v.is_finite()).unwrap_or(0.0),
        None => 0.0,
    }
}

/// Parse `x-ratelimit-*` headers into a [`RateLimitState`].
///
/// Returns `None` if no rate limit headers are present.
pub fn parse_rate_limit_headers(
    headers: &HashMap<String, String>,
    provider: &str,
) -> Option<RateLimitState> {
    parse_rate_limit_headers_iter(
        headers.iter().map(|(k, v)| (k.as_str(), v.as_str())),
        provider,
    )
}

/// Generic variant accepting any iterator of `(name, value)` pairs, handy for
/// `reqwest::header::HeaderMap`-style callers. Header names are matched
/// case-insensitively per RFC 7230.
pub fn parse_rate_limit_headers_iter<'a, I>(headers: I, provider: &str) -> Option<RateLimitState>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    // Normalize to lowercase so lookups work regardless of server casing.
    let lowered: HashMap<String, String> = headers
        .into_iter()
        .map(|(k, v)| (k.to_lowercase(), v.to_string()))
        .collect();

    // Quick check: at least one rate limit header must exist.
    let has_any = lowered.keys().any(|k| k.starts_with("x-ratelimit-"));
    if !has_any {
        return None;
    }

    let now = now_secs();

    let bucket = |resource: &str, suffix: &str| -> RateLimitBucket {
        let tag = format!("{resource}{suffix}");
        RateLimitBucket {
            limit: safe_int(lowered.get(&format!("x-ratelimit-limit-{tag}")).map(String::as_str)),
            remaining: safe_int(
                lowered
                    .get(&format!("x-ratelimit-remaining-{tag}"))
                    .map(String::as_str),
            ),
            reset_seconds: safe_float(
                lowered
                    .get(&format!("x-ratelimit-reset-{tag}"))
                    .map(String::as_str),
            ),
            captured_at: now,
        }
    };

    Some(RateLimitState {
        requests_min: bucket("requests", ""),
        requests_hour: bucket("requests", "-1h"),
        tokens_min: bucket("tokens", ""),
        tokens_hour: bucket("tokens", "-1h"),
        captured_at: now,
        provider: provider.to_string(),
    })
}

// ── Formatting ──────────────────────────────────────────────────────────

/// Human-friendly number: 7999856 -> "8.0M", 33599 -> "33.6K", 799 -> "799".
fn fmt_count(n: i64) -> String {
    if n >= 1_000_000 {
        return format!("{:.1}M", n as f64 / 1_000_000.0);
    }
    // The Python source has two identical >=10_000 and >=1_000 branches; both
    // produce "{n/1000:.1f}K", so a single >=1_000 check is equivalent.
    if n >= 1_000 {
        return format!("{:.1}K", n as f64 / 1_000.0);
    }
    n.to_string()
}

/// Seconds -> human-friendly duration: "58s", "2m 14s", "58m 57s", "1h 2m".
fn fmt_seconds(seconds: f64) -> String {
    // Python: max(0, int(seconds)) — int() truncates toward zero.
    let s = (seconds.trunc() as i64).max(0);
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

/// ASCII progress bar: `[████████░░░░░░░░░░░░]`.
fn bar(pct: f64, width: usize) -> String {
    let mut filled = (pct / 100.0 * width as f64) as i64; // int() truncates
    filled = filled.clamp(0, width as i64);
    let filled = filled as usize;
    let empty = width - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// Format one bucket as a single line.
fn bucket_line(label: &str, bucket: &RateLimitBucket, label_width: usize) -> String {
    if bucket.limit <= 0 {
        return format!("  {label:<label_width$}  (no data)");
    }

    let pct = bucket.usage_pct();
    let used = fmt_count(bucket.used());
    let limit = fmt_count(bucket.limit);
    let remaining = fmt_count(bucket.remaining);
    let reset = fmt_seconds(bucket.remaining_seconds_now());

    let bar = bar(pct, 20);
    format!(
        "  {label:<label_width$} {bar} {pct:5.1}%  {used}/{limit} used  ({remaining} left, resets in {reset})"
    )
}

/// Python `str.title()`: capitalize the first letter of each run of letters,
/// lowercase the rest.
fn title_case(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_is_alpha = false;
    for ch in input.chars() {
        if ch.is_alphabetic() {
            if prev_is_alpha {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
            prev_is_alpha = true;
        } else {
            out.push(ch);
            prev_is_alpha = false;
        }
    }
    out
}

/// Format rate limit state for terminal/chat display.
pub fn format_rate_limit_display(state: &RateLimitState) -> String {
    if !state.has_data() {
        return "No rate limit data yet — make an API request first.".to_string();
    }

    let age = state.age_seconds();
    let freshness = if age < 5.0 {
        "just now".to_string()
    } else if age < 60.0 {
        format!("{}s ago", age.trunc() as i64)
    } else {
        format!("{} ago", fmt_seconds(age))
    };

    let provider_label = if state.provider.is_empty() {
        "Provider".to_string()
    } else {
        title_case(&state.provider)
    };

    let mut lines = vec![
        format!("{provider_label} Rate Limits (captured {freshness}):"),
        String::new(),
        bucket_line("Requests/min", &state.requests_min, 14),
        bucket_line("Requests/hr", &state.requests_hour, 14),
        String::new(),
        bucket_line("Tokens/min", &state.tokens_min, 14),
        bucket_line("Tokens/hr", &state.tokens_hour, 14),
    ];

    // Add warnings if any bucket is getting hot.
    let mut warnings = Vec::new();
    for (label, bucket) in [
        ("requests/min", &state.requests_min),
        ("requests/hr", &state.requests_hour),
        ("tokens/min", &state.tokens_min),
        ("tokens/hr", &state.tokens_hour),
    ] {
        if bucket.limit > 0 && bucket.usage_pct() >= 80.0 {
            let reset = fmt_seconds(bucket.remaining_seconds_now());
            warnings.push(format!(
                "  ⚠ {label} at {:.0}% — resets in {reset}",
                bucket.usage_pct()
            ));
        }
    }

    if !warnings.is_empty() {
        lines.push(String::new());
        lines.extend(warnings);
    }

    lines.join("\n")
}

/// One-line compact summary for status bars / gateway messages.
pub fn format_rate_limit_compact(state: &RateLimitState) -> String {
    if !state.has_data() {
        return "No rate limit data.".to_string();
    }

    let rm = &state.requests_min;
    let tm = &state.tokens_min;
    let rh = &state.requests_hour;
    let th = &state.tokens_hour;

    let mut parts = Vec::new();
    if rm.limit > 0 {
        parts.push(format!("RPM: {}/{}", rm.remaining, rm.limit));
    }
    if rh.limit > 0 {
        parts.push(format!(
            "RPH: {}/{} (resets {})",
            fmt_count(rh.remaining),
            fmt_count(rh.limit),
            fmt_seconds(rh.remaining_seconds_now())
        ));
    }
    if tm.limit > 0 {
        parts.push(format!(
            "TPM: {}/{}",
            fmt_count(tm.remaining),
            fmt_count(tm.limit)
        ));
    }
    if th.limit > 0 {
        parts.push(format!(
            "TPH: {}/{} (resets {})",
            fmt_count(th.remaining),
            fmt_count(th.limit),
            fmt_seconds(th.remaining_seconds_now())
        ));
    }

    parts.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parse_returns_none_without_rate_limit_headers() {
        let h = headers(&[("content-type", "application/json"), ("x-request-id", "abc")]);
        assert!(parse_rate_limit_headers(&h, "nous").is_none());
    }

    #[test]
    fn parse_populates_all_buckets_case_insensitively() {
        let h = headers(&[
            ("X-RateLimit-Limit-Requests", "100"),
            ("x-ratelimit-remaining-requests", "37"),
            ("x-ratelimit-reset-requests", "12"),
            ("X-RATELIMIT-LIMIT-REQUESTS-1H", "5000"),
            ("x-ratelimit-remaining-requests-1h", "4900"),
            ("x-ratelimit-reset-requests-1h", "3000"),
            ("x-ratelimit-limit-tokens", "8000000"),
            ("x-ratelimit-remaining-tokens", "7999856"),
            ("x-ratelimit-reset-tokens", "30"),
            ("x-ratelimit-limit-tokens-1h", "40000000"),
            ("x-ratelimit-remaining-tokens-1h", "33599"),
            ("x-ratelimit-reset-tokens-1h", "3500"),
        ]);
        let state = parse_rate_limit_headers(&h, "nous").expect("state");
        assert!(state.has_data());
        assert_eq!(state.provider, "nous");

        assert_eq!(state.requests_min.limit, 100);
        assert_eq!(state.requests_min.remaining, 37);
        assert_eq!(state.requests_min.used(), 63);
        assert!((state.requests_min.usage_pct() - 63.0).abs() < 1e-9);

        assert_eq!(state.requests_hour.limit, 5000);
        assert_eq!(state.tokens_min.limit, 8_000_000);
        assert_eq!(state.tokens_hour.remaining, 33599);
    }

    #[test]
    fn safe_int_truncates_floats_toward_zero() {
        assert_eq!(safe_int(Some("12.9")), 12);
        assert_eq!(safe_int(Some("-3.7")), -3);
        assert_eq!(safe_int(Some("nope")), 0);
        assert_eq!(safe_int(None), 0);
    }

    #[test]
    fn usage_pct_zero_when_no_limit() {
        let bucket = RateLimitBucket {
            limit: 0,
            remaining: 0,
            reset_seconds: 0.0,
            captured_at: now_secs(),
        };
        assert_eq!(bucket.usage_pct(), 0.0);
        assert_eq!(bucket.used(), 0);
    }

    #[test]
    fn used_clamps_to_non_negative() {
        let bucket = RateLimitBucket {
            limit: 10,
            remaining: 25,
            reset_seconds: 0.0,
            captured_at: now_secs(),
        };
        assert_eq!(bucket.used(), 0);
    }

    #[test]
    fn fmt_count_matches_python() {
        assert_eq!(fmt_count(799), "799");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1_000), "1.0K");
        assert_eq!(fmt_count(33_599), "33.6K");
        assert_eq!(fmt_count(7_999_856), "8.0M");
        assert_eq!(fmt_count(1_500_000), "1.5M");
    }

    #[test]
    fn fmt_seconds_matches_python() {
        assert_eq!(fmt_seconds(58.0), "58s");
        assert_eq!(fmt_seconds(0.0), "0s");
        assert_eq!(fmt_seconds(-10.0), "0s");
        assert_eq!(fmt_seconds(134.0), "2m 14s");
        assert_eq!(fmt_seconds(120.0), "2m");
        assert_eq!(fmt_seconds(3537.0), "58m 57s");
        assert_eq!(fmt_seconds(3720.0), "1h 2m");
        assert_eq!(fmt_seconds(3600.0), "1h");
    }

    #[test]
    fn bar_fills_proportionally() {
        assert_eq!(bar(0.0, 20), format!("[{}]", "░".repeat(20)));
        assert_eq!(bar(100.0, 20), format!("[{}]", "█".repeat(20)));
        // 40% of 20 = 8 filled.
        assert_eq!(
            bar(40.0, 20),
            format!("[{}{}]", "█".repeat(8), "░".repeat(12))
        );
        // Over/under clamps.
        assert_eq!(bar(150.0, 20), format!("[{}]", "█".repeat(20)));
        assert_eq!(bar(-5.0, 20), format!("[{}]", "░".repeat(20)));
    }

    #[test]
    fn title_case_capitalizes_words() {
        assert_eq!(title_case("nous"), "Nous");
        assert_eq!(title_case("open router"), "Open Router");
        assert_eq!(title_case("OPENAI"), "Openai");
        assert_eq!(title_case("nous-portal"), "Nous-Portal");
    }

    #[test]
    fn display_reports_no_data() {
        let state = RateLimitState::default();
        assert_eq!(
            format_rate_limit_display(&state),
            "No rate limit data yet — make an API request first."
        );
        assert_eq!(format_rate_limit_compact(&state), "No rate limit data.");
    }

    #[test]
    fn display_includes_provider_and_buckets() {
        let now = now_secs();
        let state = RateLimitState {
            requests_min: RateLimitBucket {
                limit: 100,
                remaining: 90,
                reset_seconds: 30.0,
                captured_at: now,
            },
            requests_hour: RateLimitBucket {
                limit: 5000,
                remaining: 4000,
                reset_seconds: 3000.0,
                captured_at: now,
            },
            tokens_min: RateLimitBucket::default(),
            tokens_hour: RateLimitBucket::default(),
            captured_at: now,
            provider: "nous".to_string(),
        };
        let display = format_rate_limit_display(&state);
        assert!(display.contains("Nous Rate Limits"));
        assert!(display.contains("Requests/min"));
        assert!(display.contains("(no data)")); // tokens buckets empty
    }

    #[test]
    fn display_emits_warnings_when_hot() {
        let now = now_secs();
        let hot = RateLimitBucket {
            limit: 100,
            remaining: 5,
            reset_seconds: 42.0,
            captured_at: now,
        };
        let state = RateLimitState {
            requests_min: hot.clone(),
            requests_hour: RateLimitBucket::default(),
            tokens_min: RateLimitBucket::default(),
            tokens_hour: RateLimitBucket::default(),
            captured_at: now,
            provider: "nous".to_string(),
        };
        let display = format_rate_limit_display(&state);
        assert!(display.contains("⚠ requests/min at 95% — resets in"));
    }

    #[test]
    fn compact_only_includes_populated_buckets() {
        let now = now_secs();
        let state = RateLimitState {
            requests_min: RateLimitBucket {
                limit: 100,
                remaining: 90,
                reset_seconds: 30.0,
                captured_at: now,
            },
            requests_hour: RateLimitBucket::default(),
            tokens_min: RateLimitBucket {
                limit: 8_000_000,
                remaining: 7_999_856,
                reset_seconds: 30.0,
                captured_at: now,
            },
            tokens_hour: RateLimitBucket::default(),
            captured_at: now,
            provider: "nous".to_string(),
        };
        let compact = format_rate_limit_compact(&state);
        assert!(compact.contains("RPM: 90/100"));
        assert!(compact.contains("TPM: 8.0M/8.0M"));
        assert!(!compact.contains("RPH"));
        assert!(!compact.contains("TPH"));
    }

    #[test]
    fn age_seconds_infinite_without_data() {
        let state = RateLimitState::default();
        assert!(state.age_seconds().is_infinite());
    }
}
