//! Account-usage snapshots for Codex, Anthropic OAuth, and OpenRouter providers.
//!
//! Native Rust port of `agent/account_usage.py`. The Python module fetches
//! per-provider rate-limit / credit information from each backend's usage API
//! and renders it into human-readable lines.
//!
//! Network access uses `reqwest::blocking`. Request construction and response
//! parsing mirror the Python originals exactly (URLs, headers, JSON shapes).
//!
//! Credential resolution helpers (`resolve_codex_runtime_credentials`,
//! `_read_codex_tokens`, `resolve_runtime_provider`) are not yet ported with
//! matching signatures, so the fetch functions accept the resolved credential
//! values via lightweight structs (`CodexCredentials`, `RuntimeProvider`). This
//! keeps the module self-contained and testable while preserving the API shapes.

use chrono::{DateTime, Local, TimeZone, Utc};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One rate-limit / quota window within a usage snapshot.
///
/// Mirrors the frozen `AccountUsageWindow` dataclass.
#[derive(Debug, Clone, PartialEq)]
pub struct AccountUsageWindow {
    pub label: String,
    pub used_percent: Option<f64>,
    pub reset_at: Option<DateTime<Utc>>,
    pub detail: Option<String>,
}

impl AccountUsageWindow {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            used_percent: None,
            reset_at: None,
            detail: None,
        }
    }

    pub fn with_used_percent(mut self, used_percent: f64) -> Self {
        self.used_percent = Some(used_percent);
        self
    }

    pub fn with_reset_at(mut self, reset_at: Option<DateTime<Utc>>) -> Self {
        self.reset_at = reset_at;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// A point-in-time snapshot of a provider's account usage.
///
/// Mirrors the frozen `AccountUsageSnapshot` dataclass.
#[derive(Debug, Clone, PartialEq)]
pub struct AccountUsageSnapshot {
    pub provider: String,
    pub source: String,
    pub fetched_at: DateTime<Utc>,
    pub title: String,
    pub plan: Option<String>,
    pub windows: Vec<AccountUsageWindow>,
    pub details: Vec<String>,
    pub unavailable_reason: Option<String>,
}

impl AccountUsageSnapshot {
    /// Constructor mirroring the dataclass defaults (`title="Account limits"`).
    pub fn new(
        provider: impl Into<String>,
        source: impl Into<String>,
        fetched_at: DateTime<Utc>,
    ) -> Self {
        Self {
            provider: provider.into(),
            source: source.into(),
            fetched_at,
            title: "Account limits".to_string(),
            plan: None,
            windows: Vec::new(),
            details: Vec::new(),
            unavailable_reason: None,
        }
    }

    /// `available` property: true when there is something to show and no
    /// unavailable reason is set.
    pub fn available(&self) -> bool {
        (!self.windows.is_empty() || !self.details.is_empty())
            && self.unavailable_reason.is_none()
    }
}

// ---------------------------------------------------------------------------
// Credential / runtime-provider inputs
// ---------------------------------------------------------------------------

/// Resolved Codex runtime credentials, as produced by the Python
/// `resolve_codex_runtime_credentials` + `_read_codex_tokens` combination.
#[derive(Debug, Clone, Default)]
pub struct CodexCredentials {
    pub api_key: String,
    pub base_url: String,
    /// `tokens.account_id` from the on-disk codex token store, if present.
    pub account_id: Option<String>,
}

/// Resolved runtime provider info, as produced by `resolve_runtime_provider`.
#[derive(Debug, Clone, Default)]
pub struct RuntimeProvider {
    pub api_key: String,
    pub base_url: String,
}

// ---------------------------------------------------------------------------
// String / value helpers
// ---------------------------------------------------------------------------

/// `_title_case_slug`: trim, replace `_`/`-` with spaces, then title-case.
/// Returns `None` for empty/blank input.
fn title_case_slug(value: Option<&str>) -> Option<String> {
    let cleaned = value.unwrap_or("").trim();
    if cleaned.is_empty() {
        return None;
    }
    let spaced: String = cleaned
        .chars()
        .map(|c| if c == '_' || c == '-' { ' ' } else { c })
        .collect();
    Some(title_case(&spaced))
}

/// Python `str.title()`: capitalises the first cased char of every run of
/// letters and lower-cases the rest. Digits/symbols reset the "start of word".
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_is_cased = false;
    for ch in s.chars() {
        let is_cased = ch.is_alphabetic();
        if is_cased {
            if prev_is_cased {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
        } else {
            out.push(ch);
        }
        prev_is_cased = is_cased;
    }
    out
}

/// `_parse_dt`: accept ISO-8601 strings (with `Z` or offset) or numeric epoch
/// seconds. Returns `None` for null/empty/unparseable values.
fn parse_dt(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::Null => None,
        Value::Number(n) => {
            let secs = n.as_f64()?;
            // datetime.fromtimestamp(float, tz=utc)
            let whole = secs.trunc() as i64;
            let nanos = ((secs - secs.trunc()) * 1_000_000_000.0).round() as u32;
            Utc.timestamp_opt(whole, nanos).single()
        }
        Value::String(s) => parse_dt_str(s),
        _ => None,
    }
}

fn parse_dt_str(value: &str) -> Option<DateTime<Utc>> {
    let text = value.trim();
    if text.is_empty() {
        return None;
    }
    // Python replaces a trailing "Z" with "+00:00".
    let normalized = if let Some(stripped) = text.strip_suffix('Z') {
        format!("{stripped}+00:00")
    } else {
        text.to_string()
    };

    // Try full datetime with offset.
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Some(dt.with_timezone(&Utc));
    }
    // datetime.fromisoformat also accepts a space separator and naive values.
    let candidate = normalized.replacen(' ', "T", 1);
    if let Ok(dt) = DateTime::parse_from_rfc3339(&candidate) {
        return Some(dt.with_timezone(&Utc));
    }
    // Naive datetime (no tz) -> assume UTC (Python: replace(tzinfo=utc)).
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&candidate, fmt) {
            return Some(Utc.from_utc_datetime(&ndt));
        }
    }
    // Date-only.
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(&candidate, "%Y-%m-%d") {
        if let Some(ndt) = nd.and_hms_opt(0, 0, 0) {
            return Some(Utc.from_utc_datetime(&ndt));
        }
    }
    None
}

/// `_format_reset`: render a reset timestamp as a relative + absolute string.
fn format_reset(dt: Option<&DateTime<Utc>>) -> String {
    let dt = match dt {
        Some(d) => d,
        None => return "unknown".to_string(),
    };
    let local_dt: DateTime<Local> = dt.with_timezone(&Local);
    let abs = format_local(&local_dt);
    let delta = *dt - utc_now();
    let total_seconds = delta.num_seconds();
    if total_seconds <= 0 {
        return format!("now ({abs})");
    }
    let hours = total_seconds / 3600;
    let rem = total_seconds % 3600;
    let minutes = rem / 60;
    let rel = if hours >= 24 {
        let days = hours / 24;
        let hours = hours % 24;
        format!("in {days}d {hours}h")
    } else if hours > 0 {
        format!("in {hours}h {minutes}m")
    } else {
        format!("in {minutes}m")
    };
    format!("{rel} ({abs})")
}

/// Mirror of Python's `strftime('%Y-%m-%d %H:%M %Z')` for a local datetime.
fn format_local(dt: &DateTime<Local>) -> String {
    dt.format("%Y-%m-%d %H:%M %Z").to_string()
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// `render_account_usage_lines`: produce the human-readable line list.
pub fn render_account_usage_lines(
    snapshot: Option<&AccountUsageSnapshot>,
    markdown: bool,
) -> Vec<String> {
    let snapshot = match snapshot {
        Some(s) => s,
        None => return Vec::new(),
    };
    let bold = if markdown { "**" } else { "" };
    let header = format!("📈 {bold}{}{bold}", snapshot.title);
    let mut lines = vec![header];
    match &snapshot.plan {
        Some(plan) => lines.push(format!("Provider: {} ({plan})", snapshot.provider)),
        None => lines.push(format!("Provider: {}", snapshot.provider)),
    }
    for window in &snapshot.windows {
        let mut base = match window.used_percent {
            None => format!("{}: unavailable", window.label),
            Some(used_percent) => {
                let remaining = (100.0 - used_percent).round().max(0.0) as i64;
                let used = used_percent.round().max(0.0) as i64;
                format!(
                    "{}: {remaining}% remaining ({used}% used)",
                    window.label
                )
            }
        };
        if let Some(reset_at) = &window.reset_at {
            base += &format!(" • resets {}", format_reset(Some(reset_at)));
        } else if let Some(detail) = &window.detail {
            base += &format!(" • {detail}");
        }
        lines.push(base);
    }
    for detail in &snapshot.details {
        lines.push(detail.clone());
    }
    if let Some(reason) = &snapshot.unavailable_reason {
        lines.push(format!("Unavailable: {reason}"));
    }
    lines
}

// ---------------------------------------------------------------------------
// Codex usage
// ---------------------------------------------------------------------------

/// `_resolve_codex_usage_url`: derive the usage endpoint from a base URL.
pub fn resolve_codex_usage_url(base_url: &str) -> String {
    let mut normalized = base_url.trim().trim_end_matches('/').to_string();
    if normalized.is_empty() {
        normalized = "https://chatgpt.com/backend-api/codex".to_string();
    }
    if let Some(stripped) = normalized.strip_suffix("/codex") {
        normalized = stripped.to_string();
    }
    if normalized.contains("/backend-api") {
        format!("{normalized}/wham/usage")
    } else {
        format!("{normalized}/api/codex/usage")
    }
}

/// `_fetch_codex_account_usage`: build request, fetch, and parse the snapshot.
///
/// Credentials are supplied by the caller (resolved upstream).
pub fn fetch_codex_account_usage(
    creds: &CodexCredentials,
) -> Option<AccountUsageSnapshot> {
    let account_id = creds
        .account_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok()?;
    let mut request = client
        .get(resolve_codex_usage_url(&creds.base_url))
        .header("Authorization", format!("Bearer {}", creds.api_key))
        .header("Accept", "application/json")
        .header("User-Agent", "codex-cli");
    if let Some(account_id) = account_id {
        request = request.header("ChatGPT-Account-Id", account_id);
    }
    let response = request.send().ok()?;
    let response = response.error_for_status().ok()?;
    let payload: Value = response.json().ok()?;
    Some(parse_codex_usage_payload(&payload))
}

/// Pure parser for the Codex usage payload, factored out for testing.
pub fn parse_codex_usage_payload(payload: &Value) -> AccountUsageSnapshot {
    let rate_limit = payload.get("rate_limit").cloned().unwrap_or(Value::Null);
    let mut windows: Vec<AccountUsageWindow> = Vec::new();
    for (key, label) in [("primary_window", "Session"), ("secondary_window", "Weekly")] {
        let window = rate_limit.get(key).cloned().unwrap_or(Value::Null);
        let used = window.get("used_percent");
        let used = match used.and_then(json_to_f64) {
            Some(u) => u,
            None => continue,
        };
        windows.push(
            AccountUsageWindow::new(label)
                .with_used_percent(used)
                .with_reset_at(
                    window
                        .get("reset_at")
                        .map(|v| parse_dt(v))
                        .unwrap_or(None),
                ),
        );
    }

    let mut details: Vec<String> = Vec::new();
    let credits = payload.get("credits").cloned().unwrap_or(Value::Null);
    if json_truthy(credits.get("has_credits")) {
        if let Some(balance) = credits.get("balance").and_then(json_to_f64) {
            details.push(format!("Credits balance: ${balance:.2}"));
        } else if json_truthy(credits.get("unlimited")) {
            details.push("Credits balance: unlimited".to_string());
        }
    }

    let mut snapshot = AccountUsageSnapshot::new("openai-codex", "usage_api", utc_now());
    snapshot.plan = title_case_slug(payload.get("plan_type").and_then(Value::as_str));
    snapshot.windows = windows;
    snapshot.details = details;
    snapshot
}

// ---------------------------------------------------------------------------
// Anthropic usage
// ---------------------------------------------------------------------------

/// `_fetch_anthropic_account_usage`: OAuth-only Anthropic usage snapshot.
pub fn fetch_anthropic_account_usage() -> Option<AccountUsageSnapshot> {
    let token = crate::ag_anthropic_adapter::resolve_anthropic_token()
        .unwrap_or_default()
        .trim()
        .to_string();
    if token.is_empty() {
        return None;
    }
    if !crate::ag_anthropic_adapter::is_oauth_token(&token) {
        let mut snapshot =
            AccountUsageSnapshot::new("anthropic", "oauth_usage_api", utc_now());
        snapshot.unavailable_reason = Some(
            "Anthropic account limits are only available for OAuth-backed Claude accounts."
                .to_string(),
        );
        return Some(snapshot);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok()?;
    let response = client
        .get("https://api.anthropic.com/api/oauth/usage")
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .header("anthropic-beta", "oauth-2025-04-20")
        .header("User-Agent", "claude-code/2.1.0")
        .send()
        .ok()?;
    let response = response.error_for_status().ok()?;
    let payload: Value = response.json().ok()?;
    Some(parse_anthropic_usage_payload(&payload))
}

/// Pure parser for the Anthropic OAuth usage payload, factored out for testing.
pub fn parse_anthropic_usage_payload(payload: &Value) -> AccountUsageSnapshot {
    let mut windows: Vec<AccountUsageWindow> = Vec::new();
    let mapping = [
        ("five_hour", "Current session"),
        ("seven_day", "Current week"),
        ("seven_day_opus", "Opus week"),
        ("seven_day_sonnet", "Sonnet week"),
    ];
    for (key, label) in mapping {
        let window = payload.get(key).cloned().unwrap_or(Value::Null);
        let util = match window.get("utilization").and_then(json_to_f64) {
            Some(u) => u,
            None => continue,
        };
        let used = if util <= 1.0 { util * 100.0 } else { util };
        windows.push(
            AccountUsageWindow::new(label)
                .with_used_percent(used)
                .with_reset_at(
                    window
                        .get("resets_at")
                        .map(|v| parse_dt(v))
                        .unwrap_or(None),
                ),
        );
    }

    let mut details: Vec<String> = Vec::new();
    let extra = payload.get("extra_usage").cloned().unwrap_or(Value::Null);
    if json_truthy(extra.get("is_enabled")) {
        let used_credits = extra.get("used_credits").and_then(json_to_f64);
        let monthly_limit = extra.get("monthly_limit").and_then(json_to_f64);
        let currency = extra
            .get("currency")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "USD".to_string());
        if let (Some(used_credits), Some(monthly_limit)) = (used_credits, monthly_limit) {
            details.push(format!(
                "Extra usage: {used_credits:.2} / {monthly_limit:.2} {currency}"
            ));
        }
    }

    let mut snapshot =
        AccountUsageSnapshot::new("anthropic", "oauth_usage_api", utc_now());
    snapshot.windows = windows;
    snapshot.details = details;
    snapshot
}

// ---------------------------------------------------------------------------
// OpenRouter usage
// ---------------------------------------------------------------------------

/// `_fetch_openrouter_account_usage`: build requests, fetch credits + key info.
///
/// `runtime` is the resolved provider info (api_key + base_url).
pub fn fetch_openrouter_account_usage(
    runtime: &RuntimeProvider,
) -> Option<AccountUsageSnapshot> {
    let token = runtime.api_key.trim();
    if token.is_empty() {
        return None;
    }
    let normalized = runtime.base_url.trim_end_matches('/');
    let credits_url = format!("{normalized}/credits");
    let key_url = format!("{normalized}/key");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;

    let credits_resp = client
        .get(&credits_url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/json")
        .send()
        .ok()?;
    let credits_resp = credits_resp.error_for_status().ok()?;
    let credits_json: Value = credits_resp.json().ok()?;
    let credits = credits_json.get("data").cloned().unwrap_or(Value::Null);

    // The key request is best-effort; failures fall back to an empty object.
    let key_data = (|| -> Option<Value> {
        let resp = client
            .get(&key_url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Accept", "application/json")
            .send()
            .ok()?;
        let resp = resp.error_for_status().ok()?;
        let json: Value = resp.json().ok()?;
        Some(json.get("data").cloned().unwrap_or(Value::Null))
    })()
    .unwrap_or(Value::Null);

    Some(parse_openrouter_usage_payload(&credits, &key_data))
}

/// Pure parser for the OpenRouter credits + key payloads, factored out for testing.
pub fn parse_openrouter_usage_payload(credits: &Value, key_data: &Value) -> AccountUsageSnapshot {
    let total_credits = credits.get("total_credits").and_then(json_to_f64).unwrap_or(0.0);
    let total_usage = credits.get("total_usage").and_then(json_to_f64).unwrap_or(0.0);
    let balance = (total_credits - total_usage).max(0.0);
    let mut details: Vec<String> = vec![format!("Credits balance: ${balance:.2}")];

    let mut windows: Vec<AccountUsageWindow> = Vec::new();
    let limit = key_data.get("limit").and_then(json_to_f64);
    let limit_remaining = key_data.get("limit_remaining").and_then(json_to_f64);
    let limit_reset = key_data
        .get("limit_reset")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let usage = key_data.get("usage").and_then(json_to_f64);

    if let (Some(limit_value), Some(remaining_value)) = (limit, limit_remaining) {
        if limit_value > 0.0 && remaining_value >= 0.0 && remaining_value <= limit_value {
            let used_percent = ((limit_value - remaining_value) / limit_value) * 100.0;
            let mut detail_parts =
                vec![format!("${remaining_value:.2} of ${limit_value:.2} remaining")];
            if !limit_reset.is_empty() {
                detail_parts.push(format!("resets {limit_reset}"));
            }
            windows.push(
                AccountUsageWindow::new("API key quota")
                    .with_used_percent(used_percent)
                    .with_detail(detail_parts.join(" • ")),
            );
        }
    }

    if let Some(usage) = usage {
        let mut usage_parts = vec![format!("API key usage: ${usage:.2} total")];
        for (field, label) in [
            ("usage_daily", "today"),
            ("usage_weekly", "this week"),
            ("usage_monthly", "this month"),
        ] {
            if let Some(value) = key_data.get(field).and_then(json_to_f64) {
                if value > 0.0 {
                    usage_parts.push(format!("${value:.2} {label}"));
                }
            }
        }
        details.push(usage_parts.join(" • "));
    }

    let mut snapshot = AccountUsageSnapshot::new("openrouter", "credits_api", utc_now());
    snapshot.windows = windows;
    snapshot.details = details;
    snapshot
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Credential inputs for [`fetch_account_usage`]. The Codex and OpenRouter
/// branches need pre-resolved credentials (resolution lives outside this
/// module); the Anthropic branch resolves its own token internally.
#[derive(Debug, Clone, Default)]
pub struct UsageCredentials {
    pub codex: Option<CodexCredentials>,
    pub openrouter: Option<RuntimeProvider>,
}

/// `fetch_account_usage`: normalise the provider name and dispatch.
///
/// Any error in a branch yields `None` (matching the Python `try/except`).
pub fn fetch_account_usage(
    provider: Option<&str>,
    creds: &UsageCredentials,
) -> Option<AccountUsageSnapshot> {
    let normalized = provider.unwrap_or("").trim().to_lowercase();
    if normalized.is_empty() || normalized == "auto" || normalized == "custom" {
        return None;
    }
    match normalized.as_str() {
        "openai-codex" => {
            let codex = creds.codex.as_ref()?;
            fetch_codex_account_usage(codex)
        }
        "anthropic" => fetch_anthropic_account_usage(),
        "openrouter" => {
            let runtime = creds.openrouter.as_ref()?;
            fetch_openrouter_account_usage(runtime)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Coerce a JSON value to f64 for `int`/`float` inputs (Python
/// `isinstance(x, (int, float))`). Strings and bools are not coerced (bools in
/// Python would be ints, but these payloads never use bool for numeric fields).
fn json_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Python truthiness for the JSON values we care about (bool/number/string).
fn json_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
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
    fn test_title_case_slug() {
        assert_eq!(title_case_slug(None), None);
        assert_eq!(title_case_slug(Some("  ")), None);
        assert_eq!(title_case_slug(Some("pro_max")), Some("Pro Max".to_string()));
        assert_eq!(
            title_case_slug(Some("plus-plan")),
            Some("Plus Plan".to_string())
        );
        assert_eq!(title_case_slug(Some("FREE")), Some("Free".to_string()));
    }

    #[test]
    fn test_parse_dt_iso() {
        let dt = parse_dt(&json!("2026-06-03T12:30:00Z")).unwrap();
        assert_eq!(dt.timezone(), Utc);
        assert_eq!(dt.format("%Y-%m-%d %H:%M").to_string(), "2026-06-03 12:30");
    }

    #[test]
    fn test_parse_dt_offset() {
        let dt = parse_dt(&json!("2026-06-03T12:30:00+02:00")).unwrap();
        // 12:30 +02:00 == 10:30 UTC
        assert_eq!(dt.format("%H:%M").to_string(), "10:30");
    }

    #[test]
    fn test_parse_dt_naive_assumes_utc() {
        let dt = parse_dt(&json!("2026-06-03T12:30:00")).unwrap();
        assert_eq!(dt.format("%H:%M").to_string(), "12:30");
    }

    #[test]
    fn test_parse_dt_epoch() {
        let dt = parse_dt(&json!(0)).unwrap();
        assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "1970-01-01 00:00:00");
    }

    #[test]
    fn test_parse_dt_empty_and_null() {
        assert!(parse_dt(&json!("")).is_none());
        assert!(parse_dt(&json!(null)).is_none());
        assert!(parse_dt(&json!("not-a-date")).is_none());
    }

    #[test]
    fn test_resolve_codex_usage_url() {
        assert_eq!(
            resolve_codex_usage_url(""),
            "https://chatgpt.com/backend-api/codex/wham/usage"
        );
        assert_eq!(
            resolve_codex_usage_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/codex/wham/usage"
        );
        assert_eq!(
            resolve_codex_usage_url("https://chatgpt.com/backend-api/codex/"),
            "https://chatgpt.com/backend-api/codex/wham/usage"
        );
        assert_eq!(
            resolve_codex_usage_url("https://example.com/codex"),
            "https://example.com/api/codex/usage"
        );
        assert_eq!(
            resolve_codex_usage_url("https://example.com"),
            "https://example.com/api/codex/usage"
        );
    }

    #[test]
    fn test_format_reset_unknown() {
        assert_eq!(format_reset(None), "unknown");
    }

    #[test]
    fn test_format_reset_past_is_now() {
        let past = utc_now() - chrono::Duration::hours(1);
        assert!(format_reset(Some(&past)).starts_with("now ("));
    }

    #[test]
    fn test_format_reset_future_minutes() {
        let future = utc_now() + chrono::Duration::minutes(30);
        let s = format_reset(Some(&future));
        // ~30 minutes out -> "in 29m" or "in 30m" depending on rounding.
        assert!(s.starts_with("in 29m (") || s.starts_with("in 30m ("), "{s}");
    }

    #[test]
    fn test_format_reset_future_days() {
        let future = utc_now() + chrono::Duration::hours(50);
        let s = format_reset(Some(&future));
        assert!(s.starts_with("in 2d 1h (") || s.starts_with("in 2d 2h ("), "{s}");
    }

    #[test]
    fn test_render_none() {
        assert!(render_account_usage_lines(None, false).is_empty());
    }

    #[test]
    fn test_render_windows_and_details() {
        let mut snap =
            AccountUsageSnapshot::new("openrouter", "credits_api", utc_now());
        snap.plan = Some("Pro".to_string());
        snap.windows = vec![
            AccountUsageWindow::new("Session").with_used_percent(40.0),
            AccountUsageWindow::new("Weekly"), // unavailable
            AccountUsageWindow::new("API key quota")
                .with_used_percent(25.0)
                .with_detail("$75.00 of $100.00 remaining"),
        ];
        snap.details = vec!["Credits balance: $10.00".to_string()];
        let lines = render_account_usage_lines(Some(&snap), false);
        assert_eq!(lines[0], "📈 Account limits");
        assert_eq!(lines[1], "Provider: openrouter (Pro)");
        assert_eq!(lines[2], "Session: 60% remaining (40% used)");
        assert_eq!(lines[3], "Weekly: unavailable");
        assert_eq!(
            lines[4],
            "API key quota: 75% remaining (25% used) • $75.00 of $100.00 remaining"
        );
        assert_eq!(lines[5], "Credits balance: $10.00");
    }

    #[test]
    fn test_render_markdown_header_and_unavailable() {
        let mut snap = AccountUsageSnapshot::new("anthropic", "oauth_usage_api", utc_now());
        snap.unavailable_reason = Some("nope".to_string());
        let lines = render_account_usage_lines(Some(&snap), true);
        assert_eq!(lines[0], "📈 **Account limits**");
        assert_eq!(lines[1], "Provider: anthropic");
        assert_eq!(lines.last().unwrap(), "Unavailable: nope");
    }

    #[test]
    fn test_available_property() {
        let mut snap = AccountUsageSnapshot::new("x", "y", utc_now());
        assert!(!snap.available());
        snap.details = vec!["a".to_string()];
        assert!(snap.available());
        snap.unavailable_reason = Some("r".to_string());
        assert!(!snap.available());
    }

    #[test]
    fn test_parse_codex_usage_payload() {
        let payload = json!({
            "plan_type": "pro_plus",
            "rate_limit": {
                "primary_window": {"used_percent": 12.5, "reset_at": "2026-06-03T12:00:00Z"},
                "secondary_window": {"used_percent": 50}
            },
            "credits": {"has_credits": true, "balance": 9.5}
        });
        let snap = parse_codex_usage_payload(&payload);
        assert_eq!(snap.provider, "openai-codex");
        assert_eq!(snap.source, "usage_api");
        assert_eq!(snap.plan, Some("Pro Plus".to_string()));
        assert_eq!(snap.windows.len(), 2);
        assert_eq!(snap.windows[0].label, "Session");
        assert_eq!(snap.windows[0].used_percent, Some(12.5));
        assert!(snap.windows[0].reset_at.is_some());
        assert_eq!(snap.windows[1].label, "Weekly");
        assert_eq!(snap.windows[1].used_percent, Some(50.0));
        assert_eq!(snap.details, vec!["Credits balance: $9.50".to_string()]);
    }

    #[test]
    fn test_parse_codex_usage_unlimited_credits() {
        let payload = json!({
            "credits": {"has_credits": true, "unlimited": true}
        });
        let snap = parse_codex_usage_payload(&payload);
        assert_eq!(snap.details, vec!["Credits balance: unlimited".to_string()]);
        assert!(snap.windows.is_empty());
        assert_eq!(snap.plan, None);
    }

    #[test]
    fn test_parse_anthropic_usage_payload() {
        let payload = json!({
            "five_hour": {"utilization": 0.4, "resets_at": "2026-06-03T16:00:00Z"},
            "seven_day": {"utilization": 80},
            "seven_day_opus": {},
            "extra_usage": {
                "is_enabled": true,
                "used_credits": 3.5,
                "monthly_limit": 50,
                "currency": "EUR"
            }
        });
        let snap = parse_anthropic_usage_payload(&payload);
        assert_eq!(snap.provider, "anthropic");
        assert_eq!(snap.windows.len(), 2);
        assert_eq!(snap.windows[0].label, "Current session");
        assert_eq!(snap.windows[0].used_percent, Some(40.0));
        assert_eq!(snap.windows[1].label, "Current week");
        assert_eq!(snap.windows[1].used_percent, Some(80.0));
        assert_eq!(snap.details, vec!["Extra usage: 3.50 / 50.00 EUR".to_string()]);
    }

    #[test]
    fn test_parse_anthropic_default_currency() {
        let payload = json!({
            "extra_usage": {"is_enabled": true, "used_credits": 1, "monthly_limit": 10}
        });
        let snap = parse_anthropic_usage_payload(&payload);
        assert_eq!(snap.details, vec!["Extra usage: 1.00 / 10.00 USD".to_string()]);
    }

    #[test]
    fn test_parse_openrouter_usage_payload() {
        let credits = json!({"total_credits": 100.0, "total_usage": 40.0});
        let key_data = json!({
            "limit": 50.0,
            "limit_remaining": 30.0,
            "limit_reset": "2026-06-04",
            "usage": 20.0,
            "usage_daily": 2.0,
            "usage_weekly": 0.0,
            "usage_monthly": 15.0
        });
        let snap = parse_openrouter_usage_payload(&credits, &key_data);
        assert_eq!(snap.provider, "openrouter");
        assert_eq!(snap.windows.len(), 1);
        assert_eq!(snap.windows[0].label, "API key quota");
        assert_eq!(snap.windows[0].used_percent, Some(40.0));
        assert_eq!(
            snap.windows[0].detail,
            Some("$30.00 of $50.00 remaining • resets 2026-06-04".to_string())
        );
        assert_eq!(snap.details[0], "Credits balance: $60.00");
        // usage_weekly is 0 -> skipped
        assert_eq!(
            snap.details[1],
            "API key usage: $20.00 total • $2.00 today • $15.00 this month"
        );
    }

    #[test]
    fn test_parse_openrouter_no_key_data() {
        let credits = json!({"total_credits": 5.0, "total_usage": 8.0});
        let snap = parse_openrouter_usage_payload(&credits, &Value::Null);
        // balance clamped to 0
        assert_eq!(snap.details, vec!["Credits balance: $0.00".to_string()]);
        assert!(snap.windows.is_empty());
    }

    #[test]
    fn test_fetch_account_usage_skips_auto_custom_empty() {
        let creds = UsageCredentials::default();
        assert!(fetch_account_usage(None, &creds).is_none());
        assert!(fetch_account_usage(Some(""), &creds).is_none());
        assert!(fetch_account_usage(Some("auto"), &creds).is_none());
        assert!(fetch_account_usage(Some("CUSTOM"), &creds).is_none());
        assert!(fetch_account_usage(Some("unknown-provider"), &creds).is_none());
    }

    #[test]
    fn test_fetch_account_usage_no_codex_creds() {
        let creds = UsageCredentials::default();
        assert!(fetch_account_usage(Some("openai-codex"), &creds).is_none());
        assert!(fetch_account_usage(Some("openrouter"), &creds).is_none());
    }
}
