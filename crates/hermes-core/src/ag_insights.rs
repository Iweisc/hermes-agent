//! Session Insights Engine for Hermes Agent (native Rust port of `agent/insights.py`).
//!
//! Analyzes historical session data from the SQLite state database to produce
//! comprehensive usage insights — token consumption, cost estimates, tool usage
//! patterns, activity trends, model/platform breakdowns, and session metrics.
//!
//! This is a faithful port of the Python `InsightsEngine`. It depends on a small
//! subset of `agent/usage_pricing.py` (the official-docs pricing snapshot, billing
//! route resolution, cost estimation, `has_known_pricing`, and
//! `format_duration_compact`) which is reproduced inline here so the module is
//! self-contained until `usage_pricing` is ported as its own flat module.
//!
//! Usage:
//! ```ignore
//! let conn = rusqlite::Connection::open("state.db")?;
//! let engine = InsightsEngine::new(&conn);
//! let report = engine.generate(30, None)?;
//! println!("{}", engine.format_terminal(&report));
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::{Datelike, Local, NaiveDate, TimeZone, Timelike};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// =============================================================================
// Pricing subset (ported from agent/usage_pricing.py)
// =============================================================================

/// Canonical token usage buckets (subset of usage_pricing.CanonicalUsage).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CanonicalUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub request_count: i64,
}

impl CanonicalUsage {
    pub fn new(input: i64, output: i64, cache_read: i64, cache_write: i64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: cache_read,
            cache_write_tokens: cache_write,
            request_count: 1,
        }
    }
}

/// Cost status enum (mirrors usage_pricing.CostStatus string values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostStatus {
    Actual,
    Estimated,
    Included,
    Unknown,
}

impl CostStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CostStatus::Actual => "actual",
            CostStatus::Estimated => "estimated",
            CostStatus::Included => "included",
            CostStatus::Unknown => "unknown",
        }
    }
}

/// Resolved billing route (subset of usage_pricing.BillingRoute).
#[derive(Debug, Clone)]
pub struct BillingRoute {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub billing_mode: String,
}

/// A pricing entry expressed in USD per million tokens.
#[derive(Debug, Clone, Copy)]
struct PricingEntry {
    input_per_million: Option<f64>,
    output_per_million: Option<f64>,
    cache_read_per_million: Option<f64>,
    cache_write_per_million: Option<f64>,
}

/// Result of a cost estimation (subset of usage_pricing.CostResult).
#[derive(Debug, Clone)]
pub struct CostResult {
    pub amount_usd: Option<f64>,
    pub status: CostStatus,
}

/// Port of `utils.base_url_host_matches`: the host of `base_url` equals `domain`
/// or is a subdomain of it. Defined locally to avoid a hard dependency.
pub fn base_url_host_matches(base_url: &str, domain: &str) -> bool {
    if base_url.is_empty() || domain.is_empty() {
        return false;
    }
    // Extract host: strip scheme, then take up to the first '/', '?' or '#',
    // then strip any userinfo and port.
    let after_scheme = match base_url.find("://") {
        Some(i) => &base_url[i + 3..],
        None => base_url,
    };
    let host_part: &str = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let host_no_user = match host_part.rfind('@') {
        Some(i) => &host_part[i + 1..],
        None => host_part,
    };
    let host = host_no_user.split(':').next().unwrap_or(host_no_user);
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    let domain = domain.trim().to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Port of `resolve_billing_route` (subset relevant to insights cost lookups).
pub fn resolve_billing_route(
    model_name: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> BillingRoute {
    let mut provider_name = provider.unwrap_or("").trim().to_ascii_lowercase();
    let base = base_url.unwrap_or("").trim().to_ascii_lowercase();
    let mut model = model_name.trim().to_string();

    if provider_name.is_empty() && model.contains('/') {
        if let Some((inferred, bare)) = model.split_once('/') {
            if matches!(inferred, "anthropic" | "openai" | "google") {
                provider_name = inferred.to_string();
                model = bare.to_string();
            }
        }
    }

    let last_segment = |m: &str| -> String { m.rsplit('/').next().unwrap_or(m).to_string() };

    if provider_name == "openai-codex" {
        return BillingRoute {
            provider: "openai-codex".to_string(),
            model,
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "subscription_included".to_string(),
        };
    }
    if provider_name == "openrouter" || base_url_host_matches(base_url.unwrap_or(""), "openrouter.ai")
    {
        return BillingRoute {
            provider: "openrouter".to_string(),
            model,
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "official_models_api".to_string(),
        };
    }
    if provider_name == "anthropic" {
        return BillingRoute {
            provider: "anthropic".to_string(),
            model: last_segment(&model),
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "openai" {
        return BillingRoute {
            provider: "openai".to_string(),
            model: last_segment(&model),
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "minimax" || provider_name == "minimax-cn" {
        return BillingRoute {
            provider: provider_name,
            model: last_segment(&model),
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "custom" || provider_name == "local" || (!base.is_empty() && base.contains("localhost")) {
        let prov = if provider_name.is_empty() {
            "custom".to_string()
        } else {
            provider_name
        };
        return BillingRoute {
            provider: prov,
            model,
            base_url: base_url.unwrap_or("").to_string(),
            billing_mode: "unknown".to_string(),
        };
    }
    let prov = if provider_name.is_empty() {
        "unknown".to_string()
    } else {
        provider_name
    };
    let resolved_model = if model.is_empty() {
        String::new()
    } else {
        last_segment(&model)
    };
    BillingRoute {
        provider: prov,
        model: resolved_model,
        base_url: base_url.unwrap_or("").to_string(),
        billing_mode: "unknown".to_string(),
    }
}

/// Look up an official-docs pricing snapshot entry for a (provider, model) pair.
///
/// This is the `_OFFICIAL_DOCS_PRICING` table from usage_pricing.py. Network-backed
/// pricing (OpenRouter / OpenAI-compatible models APIs) is intentionally not
/// reproduced here; insights only needs the static snapshot for cost estimation
/// and `has_known_pricing` for the static-priced providers.
fn lookup_official_docs_pricing(route: &BillingRoute) -> Option<PricingEntry> {
    let model = route.model.to_ascii_lowercase();
    let entry = |i: f64, o: f64, cr: Option<f64>, cw: Option<f64>| PricingEntry {
        input_per_million: Some(i),
        output_per_million: Some(o),
        cache_read_per_million: cr,
        cache_write_per_million: cw,
    };
    match (route.provider.as_str(), model.as_str()) {
        ("anthropic", "claude-opus-4-20250514") => {
            Some(entry(15.00, 75.00, Some(1.50), Some(18.75)))
        }
        ("anthropic", "claude-sonnet-4-20250514") => {
            Some(entry(3.00, 15.00, Some(0.30), Some(3.75)))
        }
        ("anthropic", "claude-3-5-sonnet-20241022") => {
            Some(entry(3.00, 15.00, Some(0.30), Some(3.75)))
        }
        ("anthropic", "claude-3-5-haiku-20241022") => {
            Some(entry(0.80, 4.00, Some(0.08), Some(1.00)))
        }
        ("anthropic", "claude-3-opus-20240229") => {
            Some(entry(15.00, 75.00, Some(1.50), Some(18.75)))
        }
        ("anthropic", "claude-3-haiku-20240307") => {
            Some(entry(0.25, 1.25, Some(0.03), Some(0.30)))
        }
        ("openai", "gpt-4o") => Some(entry(2.50, 10.00, Some(1.25), None)),
        ("openai", "gpt-4o-mini") => Some(entry(0.15, 0.60, Some(0.075), None)),
        ("openai", "gpt-4.1") => Some(entry(2.00, 8.00, Some(0.50), None)),
        ("openai", "gpt-4.1-mini") => Some(entry(0.40, 1.60, Some(0.10), None)),
        ("openai", "gpt-4.1-nano") => Some(entry(0.10, 0.40, Some(0.025), None)),
        ("openai", "o3") => Some(entry(10.00, 40.00, Some(2.50), None)),
        ("openai", "o3-mini") => Some(entry(1.10, 4.40, Some(0.55), None)),
        ("deepseek", "deepseek-chat") => Some(entry(0.14, 0.28, None, None)),
        ("deepseek", "deepseek-reasoner") => Some(entry(0.55, 2.19, None, None)),
        ("google", "gemini-2.5-pro") => Some(entry(1.25, 10.00, None, None)),
        ("google", "gemini-2.5-flash") => Some(entry(0.15, 0.60, None, None)),
        ("google", "gemini-2.0-flash") => Some(entry(0.10, 0.40, None, None)),
        ("bedrock", "anthropic.claude-opus-4-6") => Some(entry(15.00, 75.00, None, None)),
        ("bedrock", "anthropic.claude-sonnet-4-6") => Some(entry(3.00, 15.00, None, None)),
        ("bedrock", "anthropic.claude-sonnet-4-5") => Some(entry(3.00, 15.00, None, None)),
        ("bedrock", "anthropic.claude-haiku-4-5") => Some(entry(0.80, 4.00, None, None)),
        ("bedrock", "amazon.nova-pro") => Some(entry(0.80, 3.20, None, None)),
        ("bedrock", "amazon.nova-lite") => Some(entry(0.06, 0.24, None, None)),
        ("bedrock", "amazon.nova-micro") => Some(entry(0.035, 0.14, None, None)),
        ("minimax", "minimax-m2.7") => Some(entry(0.30, 1.20, None, None)),
        ("minimax-cn", "minimax-m2.7") => Some(entry(0.30, 1.20, None, None)),
        _ => None,
    }
}

/// Port of `get_pricing_entry` restricted to statically-known routes.
///
/// For `subscription_included` routes returns an all-zero entry (matching the
/// Python "included-route" PricingEntry). For OpenRouter/OpenAI-compatible base
/// URLs the Python implementation reaches out to the network; here we fall back
/// to the official docs snapshot only.
fn get_pricing_entry(route: &BillingRoute) -> Option<PricingEntry> {
    if route.billing_mode == "subscription_included" {
        return Some(PricingEntry {
            input_per_million: Some(0.0),
            output_per_million: Some(0.0),
            cache_read_per_million: Some(0.0),
            cache_write_per_million: Some(0.0),
        });
    }
    lookup_official_docs_pricing(route)
}

/// Port of `estimate_usage_cost` (static-snapshot subset).
pub fn estimate_usage_cost(
    model_name: &str,
    usage: &CanonicalUsage,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> CostResult {
    let route = resolve_billing_route(model_name, provider, base_url);
    if route.billing_mode == "subscription_included" {
        return CostResult {
            amount_usd: Some(0.0),
            status: CostStatus::Included,
        };
    }
    let entry = match get_pricing_entry(&route) {
        Some(e) => e,
        None => {
            return CostResult {
                amount_usd: None,
                status: CostStatus::Unknown,
            }
        }
    };

    if usage.input_tokens != 0 && entry.input_per_million.is_none() {
        return CostResult { amount_usd: None, status: CostStatus::Unknown };
    }
    if usage.output_tokens != 0 && entry.output_per_million.is_none() {
        return CostResult { amount_usd: None, status: CostStatus::Unknown };
    }
    if usage.cache_read_tokens != 0 && entry.cache_read_per_million.is_none() {
        return CostResult { amount_usd: None, status: CostStatus::Unknown };
    }
    if usage.cache_write_tokens != 0 && entry.cache_write_per_million.is_none() {
        return CostResult { amount_usd: None, status: CostStatus::Unknown };
    }

    let mut amount = 0.0_f64;
    if let Some(rate) = entry.input_per_million {
        amount += usage.input_tokens as f64 * rate / 1_000_000.0;
    }
    if let Some(rate) = entry.output_per_million {
        amount += usage.output_tokens as f64 * rate / 1_000_000.0;
    }
    if let Some(rate) = entry.cache_read_per_million {
        amount += usage.cache_read_tokens as f64 * rate / 1_000_000.0;
    }
    if let Some(rate) = entry.cache_write_per_million {
        amount += usage.cache_write_tokens as f64 * rate / 1_000_000.0;
    }

    // Static snapshot entries always have source != "none", so the
    // "included when amount==0" branch only applies to subscription routes
    // (handled above). Everything reaching here is "estimated".
    CostResult {
        amount_usd: Some(amount),
        status: CostStatus::Estimated,
    }
}

/// Port of `has_known_pricing` (static-snapshot subset).
pub fn has_known_pricing(model_name: &str, provider: Option<&str>, base_url: Option<&str>) -> bool {
    let route = resolve_billing_route(model_name, provider, base_url);
    if route.billing_mode == "subscription_included" {
        return true;
    }
    get_pricing_entry(&route).is_some()
}

/// Port of `format_duration_compact`.
pub fn format_duration_compact(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{:.0}s", seconds);
    }
    let minutes = seconds / 60.0;
    if minutes < 60.0 {
        return format!("{:.0}m", minutes);
    }
    let hours = minutes / 60.0;
    if hours < 24.0 {
        let remaining_min = (minutes % 60.0) as i64;
        if remaining_min != 0 {
            return format!("{}h {}m", hours as i64, remaining_min);
        }
        return format!("{}h", hours as i64);
    }
    let days = hours / 24.0;
    format!("{:.1}d", days)
}

// =============================================================================
// Helpers (ported from module-level functions in insights.py)
// =============================================================================

/// Estimate the USD cost for a session row. Returns (amount_usd, status_str).
fn estimate_cost_for_session(s: &SessionRow) -> (f64, String) {
    let model = s.model.clone().unwrap_or_default();
    let usage = CanonicalUsage::new(
        s.input_tokens.unwrap_or(0),
        s.output_tokens.unwrap_or(0),
        s.cache_read_tokens.unwrap_or(0),
        s.cache_write_tokens.unwrap_or(0),
    );
    let result = estimate_usage_cost(
        &model,
        &usage,
        s.billing_provider.as_deref(),
        s.billing_base_url.as_deref(),
    );
    (
        result.amount_usd.unwrap_or(0.0),
        result.status.as_str().to_string(),
    )
}

fn format_duration(seconds: f64) -> String {
    format_duration_compact(seconds)
}

/// Create simple horizontal bar chart strings from values.
/// Port of `_bar_chart`.
pub fn bar_chart(values: &[i64], max_width: usize) -> Vec<String> {
    let peak = values.iter().copied().max().unwrap_or(1);
    if peak == 0 {
        return values.iter().map(|_| String::new()).collect();
    }
    values
        .iter()
        .map(|&v| {
            if v > 0 {
                let width = ((v as f64) / (peak as f64) * (max_width as f64)) as i64;
                let width = std::cmp::max(1, width) as usize;
                "█".repeat(width)
            } else {
                String::new()
            }
        })
        .collect()
}

/// Format an integer with thousands separators (matches Python `{:,}`).
fn comma(value: i64) -> String {
    let neg = value < 0;
    let digits = value.unsigned_abs().to_string();
    let bytes = digits.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if neg {
        format!("-{out}")
    } else {
        out
    }
}

// =============================================================================
// Data structures
// =============================================================================

/// A row from the `sessions` table (only the columns insights needs).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionRow {
    pub id: String,
    pub source: Option<String>,
    pub model: Option<String>,
    pub started_at: Option<f64>,
    pub ended_at: Option<f64>,
    pub message_count: Option<i64>,
    pub tool_call_count: Option<i64>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub billing_provider: Option<String>,
    pub billing_base_url: Option<String>,
    pub billing_mode: Option<String>,
    pub estimated_cost_usd: Option<f64>,
    pub actual_cost_usd: Option<f64>,
    pub cost_status: Option<String>,
    pub cost_source: Option<String>,
}

/// Aggregate tool usage record: `{tool_name, count}`.
#[derive(Debug, Clone)]
pub struct ToolUsage {
    pub tool_name: String,
    pub count: i64,
}

/// Per-skill usage record extracted from assistant tool calls.
#[derive(Debug, Clone)]
pub struct SkillUsage {
    pub skill: String,
    pub view_count: i64,
    pub manage_count: i64,
    pub last_used_at: Option<f64>,
}

/// Aggregate message statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageStats {
    pub total_messages: i64,
    pub user_messages: i64,
    pub assistant_messages: i64,
    pub tool_messages: i64,
}

/// Overview block of the report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Overview {
    pub total_sessions: i64,
    pub total_messages: i64,
    pub total_tool_calls: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cache_read_tokens: i64,
    pub total_cache_write_tokens: i64,
    pub total_tokens: i64,
    pub estimated_cost: f64,
    pub actual_cost: f64,
    pub total_hours: f64,
    pub avg_session_duration: f64,
    pub avg_messages_per_session: f64,
    pub avg_tokens_per_session: f64,
    pub user_messages: i64,
    pub assistant_messages: i64,
    pub tool_messages: i64,
    pub date_range_start: Option<f64>,
    pub date_range_end: Option<f64>,
    pub models_with_pricing: Vec<String>,
    pub models_without_pricing: Vec<String>,
    pub unknown_cost_sessions: i64,
    pub included_cost_sessions: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelBreakdown {
    pub model: String,
    pub sessions: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub total_tokens: i64,
    pub tool_calls: i64,
    pub cost: f64,
    pub has_pricing: bool,
    pub cost_status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformBreakdown {
    pub platform: String,
    pub sessions: i64,
    pub messages: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub total_tokens: i64,
    pub tool_calls: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolBreakdown {
    pub tool: String,
    pub count: i64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillEntry {
    pub skill: String,
    pub view_count: i64,
    pub manage_count: i64,
    pub total_count: i64,
    pub percentage: f64,
    pub last_used_at: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillSummary {
    pub total_skill_loads: i64,
    pub total_skill_edits: i64,
    pub total_skill_actions: i64,
    pub distinct_skills_used: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillBreakdown {
    pub summary: SkillSummary,
    pub top_skills: Vec<SkillEntry>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DayBucket {
    pub day: String,
    pub count: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HourBucket {
    pub hour: i64,
    pub count: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Activity {
    pub by_day: Vec<DayBucket>,
    pub by_hour: Vec<HourBucket>,
    pub busiest_day: Option<DayBucket>,
    pub busiest_hour: Option<HourBucket>,
    pub active_days: i64,
    pub max_streak: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopSession {
    pub label: String,
    pub session_id: String,
    pub value: String,
    pub date: String,
}

/// The complete insights report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InsightsReport {
    pub days: i64,
    pub source_filter: Option<String>,
    pub empty: bool,
    pub generated_at: Option<f64>,
    pub overview: Overview,
    pub models: Vec<ModelBreakdown>,
    pub platforms: Vec<PlatformBreakdown>,
    pub tools: Vec<ToolBreakdown>,
    pub skills: SkillBreakdown,
    pub activity: Activity,
    pub top_sessions: Vec<TopSession>,
}

// =============================================================================
// Engine
// =============================================================================

/// Analyzes session history and produces usage insights.
pub struct InsightsEngine<'a> {
    conn: &'a Connection,
}

const SESSION_COLS: &str = "id, source, model, started_at, ended_at, \
    message_count, tool_call_count, input_tokens, output_tokens, \
    cache_read_tokens, cache_write_tokens, billing_provider, \
    billing_base_url, billing_mode, estimated_cost_usd, \
    actual_cost_usd, cost_status, cost_source";

impl<'a> InsightsEngine<'a> {
    /// Initialize with a borrowed SQLite connection (the SessionDB `_conn`).
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Generate a complete insights report.
    pub fn generate(
        &self,
        days: i64,
        source: Option<&str>,
    ) -> rusqlite::Result<InsightsReport> {
        let cutoff = now_unix() - (days as f64) * 86400.0;

        let sessions = self.get_sessions(cutoff, source)?;
        let tool_usage = self.get_tool_usage(cutoff, source)?;
        let skill_usage = self.get_skill_usage(cutoff, source)?;
        let message_stats = self.get_message_stats(cutoff, source)?;

        if sessions.is_empty() {
            return Ok(InsightsReport {
                days,
                source_filter: source.map(|s| s.to_string()),
                empty: true,
                generated_at: None,
                overview: Overview::default(),
                models: Vec::new(),
                platforms: Vec::new(),
                tools: Vec::new(),
                skills: SkillBreakdown::default(),
                activity: Activity::default(),
                top_sessions: Vec::new(),
            });
        }

        let overview = compute_overview(&sessions, &message_stats);
        let models = compute_model_breakdown(&sessions);
        let platforms = compute_platform_breakdown(&sessions);
        let tools = compute_tool_breakdown(&tool_usage);
        let skills = compute_skill_breakdown(&skill_usage);
        let activity = compute_activity_patterns(&sessions);
        let top_sessions = compute_top_sessions(&sessions);

        Ok(InsightsReport {
            days,
            source_filter: source.map(|s| s.to_string()),
            empty: false,
            generated_at: Some(now_unix()),
            overview,
            models,
            platforms,
            tools,
            skills,
            activity,
            top_sessions,
        })
    }

    // -------------------------------------------------------------------------
    // Data gathering (SQL queries)
    // -------------------------------------------------------------------------

    fn get_sessions(&self, cutoff: f64, source: Option<&str>) -> rusqlite::Result<Vec<SessionRow>> {
        let sql = match source {
            Some(_) => format!(
                "SELECT {SESSION_COLS} FROM sessions \
                 WHERE started_at >= ? AND source = ? ORDER BY started_at DESC"
            ),
            None => format!(
                "SELECT {SESSION_COLS} FROM sessions \
                 WHERE started_at >= ? ORDER BY started_at DESC"
            ),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<SessionRow> {
            Ok(SessionRow {
                id: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                source: row.get(1)?,
                model: row.get(2)?,
                started_at: row.get(3)?,
                ended_at: row.get(4)?,
                message_count: row.get(5)?,
                tool_call_count: row.get(6)?,
                input_tokens: row.get(7)?,
                output_tokens: row.get(8)?,
                cache_read_tokens: row.get(9)?,
                cache_write_tokens: row.get(10)?,
                billing_provider: row.get(11)?,
                billing_base_url: row.get(12)?,
                billing_mode: row.get(13)?,
                estimated_cost_usd: row.get(14)?,
                actual_cost_usd: row.get(15)?,
                cost_status: row.get(16)?,
                cost_source: row.get(17)?,
            })
        };
        let rows: Vec<SessionRow> = match source {
            Some(src) => stmt
                .query_map(rusqlite::params![cutoff, src], map_row)?
                .collect::<rusqlite::Result<_>>()?,
            None => stmt
                .query_map(rusqlite::params![cutoff], map_row)?
                .collect::<rusqlite::Result<_>>()?,
        };
        Ok(rows)
    }

    fn get_tool_usage(&self, cutoff: f64, source: Option<&str>) -> rusqlite::Result<Vec<ToolUsage>> {
        // Source 1: explicit tool_name on tool response messages.
        let mut tool_counts: HashMap<String, i64> = HashMap::new();
        {
            let sql = match source {
                Some(_) => "SELECT m.tool_name, COUNT(*) as count \
                     FROM messages m JOIN sessions s ON s.id = m.session_id \
                     WHERE s.started_at >= ? AND s.source = ? \
                       AND m.role = 'tool' AND m.tool_name IS NOT NULL \
                     GROUP BY m.tool_name ORDER BY count DESC",
                None => "SELECT m.tool_name, COUNT(*) as count \
                     FROM messages m JOIN sessions s ON s.id = m.session_id \
                     WHERE s.started_at >= ? \
                       AND m.role = 'tool' AND m.tool_name IS NOT NULL \
                     GROUP BY m.tool_name ORDER BY count DESC",
            };
            let mut stmt = self.conn.prepare(sql)?;
            let map_row = |row: &rusqlite::Row| -> rusqlite::Result<(String, i64)> {
                Ok((row.get(0)?, row.get(1)?))
            };
            let rows: Vec<(String, i64)> = match source {
                Some(src) => stmt
                    .query_map(rusqlite::params![cutoff, src], map_row)?
                    .collect::<rusqlite::Result<_>>()?,
                None => stmt
                    .query_map(rusqlite::params![cutoff], map_row)?
                    .collect::<rusqlite::Result<_>>()?,
            };
            for (name, count) in rows {
                *tool_counts.entry(name).or_insert(0) += count;
            }
        }

        // Source 2: extract from tool_calls JSON on assistant messages.
        let mut tool_calls_counts: HashMap<String, i64> = HashMap::new();
        {
            let sql = match source {
                Some(_) => "SELECT m.tool_calls FROM messages m \
                     JOIN sessions s ON s.id = m.session_id \
                     WHERE s.started_at >= ? AND s.source = ? \
                       AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
                None => "SELECT m.tool_calls FROM messages m \
                     JOIN sessions s ON s.id = m.session_id \
                     WHERE s.started_at >= ? \
                       AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
            };
            let mut stmt = self.conn.prepare(sql)?;
            let map_row =
                |row: &rusqlite::Row| -> rusqlite::Result<Option<String>> { row.get(0) };
            let rows: Vec<Option<String>> = match source {
                Some(src) => stmt
                    .query_map(rusqlite::params![cutoff, src], map_row)?
                    .collect::<rusqlite::Result<_>>()?,
                None => stmt
                    .query_map(rusqlite::params![cutoff], map_row)?
                    .collect::<rusqlite::Result<_>>()?,
            };
            for raw in rows {
                let raw = match raw {
                    Some(s) => s,
                    None => continue,
                };
                let calls: Value = match serde_json::from_str(&raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Value::Array(arr) = calls {
                    for call in arr {
                        if let Some(name) = call
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                        {
                            *tool_calls_counts.entry(name.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Merge: prefer tool_name source, supplement with tool_calls.
        let merged: HashMap<String, i64> = if tool_counts.is_empty() && !tool_calls_counts.is_empty()
        {
            tool_calls_counts
        } else if !tool_counts.is_empty() && !tool_calls_counts.is_empty() {
            let mut all_tools: HashSet<String> = HashSet::new();
            all_tools.extend(tool_counts.keys().cloned());
            all_tools.extend(tool_calls_counts.keys().cloned());
            let mut m: HashMap<String, i64> = HashMap::new();
            for tool in all_tools {
                let a = tool_counts.get(&tool).copied().unwrap_or(0);
                let b = tool_calls_counts.get(&tool).copied().unwrap_or(0);
                m.insert(tool, std::cmp::max(a, b));
            }
            m
        } else {
            tool_counts
        };

        Ok(most_common(merged))
    }

    fn get_skill_usage(&self, cutoff: f64, source: Option<&str>) -> rusqlite::Result<Vec<SkillUsage>> {
        // Preserve first-seen insertion order, matching Python dict semantics.
        let mut order: Vec<String> = Vec::new();
        let mut skill_counts: HashMap<String, SkillUsage> = HashMap::new();

        let sql = match source {
            Some(_) => "SELECT m.tool_calls, m.timestamp FROM messages m \
                 JOIN sessions s ON s.id = m.session_id \
                 WHERE s.started_at >= ? AND s.source = ? \
                   AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
            None => "SELECT m.tool_calls, m.timestamp FROM messages m \
                 JOIN sessions s ON s.id = m.session_id \
                 WHERE s.started_at >= ? \
                   AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
        };
        let mut stmt = self.conn.prepare(sql)?;
        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<(Option<String>, Option<f64>)> {
            Ok((row.get(0)?, row.get(1)?))
        };
        let rows: Vec<(Option<String>, Option<f64>)> = match source {
            Some(src) => stmt
                .query_map(rusqlite::params![cutoff, src], map_row)?
                .collect::<rusqlite::Result<_>>()?,
            None => stmt
                .query_map(rusqlite::params![cutoff], map_row)?
                .collect::<rusqlite::Result<_>>()?,
        };

        for (raw, timestamp) in rows {
            let raw = match raw {
                Some(s) => s,
                None => continue,
            };
            let calls: Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let arr = match calls {
                Value::Array(a) => a,
                _ => continue,
            };
            for call in arr {
                let func = match call.get("function") {
                    Some(f) => f,
                    None => continue,
                };
                let tool_name = match func.get("name").and_then(|n| n.as_str()) {
                    Some(n) => n,
                    None => continue,
                };
                if tool_name != "skill_view" && tool_name != "skill_manage" {
                    continue;
                }
                // arguments may be a JSON string or an object.
                let args_val = func.get("arguments");
                let args_obj: Option<Value> = match args_val {
                    Some(Value::String(s)) => serde_json::from_str(s).ok(),
                    Some(v @ Value::Object(_)) => Some(v.clone()),
                    _ => None,
                };
                let args_obj = match args_obj {
                    Some(v @ Value::Object(_)) => v,
                    _ => continue,
                };
                let skill_name = match args_obj.get("name").and_then(|n| n.as_str()) {
                    Some(s) if !s.trim().is_empty() => s.to_string(),
                    _ => continue,
                };

                let entry = skill_counts.entry(skill_name.clone()).or_insert_with(|| {
                    order.push(skill_name.clone());
                    SkillUsage {
                        skill: skill_name.clone(),
                        view_count: 0,
                        manage_count: 0,
                        last_used_at: None,
                    }
                });
                if tool_name == "skill_view" {
                    entry.view_count += 1;
                } else {
                    entry.manage_count += 1;
                }
                if let Some(ts) = timestamp {
                    if entry.last_used_at.is_none() || ts > entry.last_used_at.unwrap() {
                        entry.last_used_at = Some(ts);
                    }
                }
            }
        }

        Ok(order
            .into_iter()
            .filter_map(|k| skill_counts.remove(&k))
            .collect())
    }

    fn get_message_stats(&self, cutoff: f64, source: Option<&str>) -> rusqlite::Result<MessageStats> {
        let sql = match source {
            Some(_) => "SELECT \
                 COUNT(*) as total_messages, \
                 SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages, \
                 SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages, \
                 SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages \
               FROM messages m JOIN sessions s ON s.id = m.session_id \
               WHERE s.started_at >= ? AND s.source = ?",
            None => "SELECT \
                 COUNT(*) as total_messages, \
                 SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages, \
                 SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages, \
                 SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages \
               FROM messages m JOIN sessions s ON s.id = m.session_id \
               WHERE s.started_at >= ?",
        };
        let mut stmt = self.conn.prepare(sql)?;
        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<MessageStats> {
            Ok(MessageStats {
                total_messages: row.get::<_, Option<i64>>(0)?.unwrap_or(0),
                user_messages: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                assistant_messages: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                tool_messages: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
            })
        };
        let result = match source {
            Some(src) => stmt
                .query_row(rusqlite::params![cutoff, src], map_row)
                .ok(),
            None => stmt.query_row(rusqlite::params![cutoff], map_row).ok(),
        };
        Ok(result.unwrap_or_default())
    }

    // -------------------------------------------------------------------------
    // Formatting (instance methods to mirror Python API)
    // -------------------------------------------------------------------------

    pub fn format_terminal(&self, report: &InsightsReport) -> String {
        format_terminal(report)
    }

    pub fn format_gateway(&self, report: &InsightsReport) -> String {
        format_gateway(report)
    }
}

// =============================================================================
// Free helpers used by the engine
// =============================================================================

fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Mimic `Counter.most_common()`: sort by count descending. Python's most_common
/// preserves insertion order for ties; HashMap has no stable order, so for ties
/// we use the tool name as a deterministic secondary key.
fn most_common(counts: HashMap<String, i64>) -> Vec<ToolUsage> {
    let mut v: Vec<ToolUsage> = counts
        .into_iter()
        .map(|(tool_name, count)| ToolUsage { tool_name, count })
        .collect();
    v.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.tool_name.cmp(&b.tool_name))
    });
    v
}

// =============================================================================
// Computation (ported from InsightsEngine._compute_*)
// =============================================================================

fn compute_overview(sessions: &[SessionRow], message_stats: &MessageStats) -> Overview {
    let total_input: i64 = sessions.iter().map(|s| s.input_tokens.unwrap_or(0)).sum();
    let total_output: i64 = sessions.iter().map(|s| s.output_tokens.unwrap_or(0)).sum();
    let total_cache_read: i64 = sessions.iter().map(|s| s.cache_read_tokens.unwrap_or(0)).sum();
    let total_cache_write: i64 = sessions.iter().map(|s| s.cache_write_tokens.unwrap_or(0)).sum();
    let total_tokens = total_input + total_output + total_cache_read + total_cache_write;
    let total_tool_calls: i64 = sessions.iter().map(|s| s.tool_call_count.unwrap_or(0)).sum();
    let total_messages: i64 = sessions.iter().map(|s| s.message_count.unwrap_or(0)).sum();

    let mut total_cost = 0.0_f64;
    let mut actual_cost = 0.0_f64;
    let mut models_with_pricing: BTreeSet<String> = BTreeSet::new();
    let mut models_without_pricing: BTreeSet<String> = BTreeSet::new();
    let mut unknown_cost_sessions = 0_i64;
    let mut included_cost_sessions = 0_i64;

    for s in sessions {
        let model = s.model.clone().unwrap_or_default();
        let (estimated, status) = estimate_cost_for_session(s);
        total_cost += estimated;
        actual_cost += s.actual_cost_usd.unwrap_or(0.0);
        let display = if model.contains('/') {
            model.rsplit('/').next().unwrap_or(&model).to_string()
        } else if model.is_empty() {
            "unknown".to_string()
        } else {
            model.clone()
        };
        if status == "included" {
            included_cost_sessions += 1;
        } else if status == "unknown" {
            unknown_cost_sessions += 1;
        }
        if has_known_pricing(&model, s.billing_provider.as_deref(), s.billing_base_url.as_deref()) {
            models_with_pricing.insert(display);
        } else {
            models_without_pricing.insert(display);
        }
    }

    let mut durations: Vec<f64> = Vec::new();
    for s in sessions {
        if let (Some(start), Some(end)) = (s.started_at, s.ended_at) {
            // Python truthiness: start/end must be non-zero too.
            if start != 0.0 && end != 0.0 && end > start {
                durations.push(end - start);
            }
        }
    }

    let total_hours = if !durations.is_empty() {
        durations.iter().sum::<f64>() / 3600.0
    } else {
        0.0
    };
    let avg_duration = if !durations.is_empty() {
        durations.iter().sum::<f64>() / durations.len() as f64
    } else {
        0.0
    };

    let started_timestamps: Vec<f64> = sessions
        .iter()
        .filter_map(|s| s.started_at.filter(|&t| t != 0.0))
        .collect();
    let date_range_start = started_timestamps
        .iter()
        .copied()
        .fold(None::<f64>, |acc, t| {
            Some(acc.map_or(t, |a| a.min(t)))
        });
    let date_range_end = started_timestamps
        .iter()
        .copied()
        .fold(None::<f64>, |acc, t| {
            Some(acc.map_or(t, |a| a.max(t)))
        });

    let n = sessions.len() as f64;

    Overview {
        total_sessions: sessions.len() as i64,
        total_messages,
        total_tool_calls,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        total_cache_read_tokens: total_cache_read,
        total_cache_write_tokens: total_cache_write,
        total_tokens,
        estimated_cost: total_cost,
        actual_cost,
        total_hours,
        avg_session_duration: avg_duration,
        avg_messages_per_session: if !sessions.is_empty() { total_messages as f64 / n } else { 0.0 },
        avg_tokens_per_session: if !sessions.is_empty() { total_tokens as f64 / n } else { 0.0 },
        user_messages: message_stats.user_messages,
        assistant_messages: message_stats.assistant_messages,
        tool_messages: message_stats.tool_messages,
        date_range_start,
        date_range_end,
        models_with_pricing: models_with_pricing.into_iter().collect(),
        models_without_pricing: models_without_pricing.into_iter().collect(),
        unknown_cost_sessions,
        included_cost_sessions,
    }
}

fn compute_model_breakdown(sessions: &[SessionRow]) -> Vec<ModelBreakdown> {
    let mut order: Vec<String> = Vec::new();
    let mut model_data: HashMap<String, ModelBreakdown> = HashMap::new();

    for s in sessions {
        let model = s.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| "unknown".to_string());
        let display_model = if model.contains('/') {
            model.rsplit('/').next().unwrap_or(&model).to_string()
        } else {
            model.clone()
        };
        let d = model_data.entry(display_model.clone()).or_insert_with(|| {
            order.push(display_model.clone());
            ModelBreakdown {
                model: display_model.clone(),
                ..Default::default()
            }
        });
        d.sessions += 1;
        let inp = s.input_tokens.unwrap_or(0);
        let out = s.output_tokens.unwrap_or(0);
        let cache_read = s.cache_read_tokens.unwrap_or(0);
        let cache_write = s.cache_write_tokens.unwrap_or(0);
        d.input_tokens += inp;
        d.output_tokens += out;
        d.cache_read_tokens += cache_read;
        d.cache_write_tokens += cache_write;
        d.total_tokens += inp + out + cache_read + cache_write;
        d.tool_calls += s.tool_call_count.unwrap_or(0);
        let (estimate, status) = estimate_cost_for_session(s);
        d.cost += estimate;
        d.has_pricing =
            has_known_pricing(&model, s.billing_provider.as_deref(), s.billing_base_url.as_deref());
        d.cost_status = status;
    }

    let mut result: Vec<ModelBreakdown> =
        order.into_iter().filter_map(|k| model_data.remove(&k)).collect();
    // Sort by tokens, then session count, descending.
    result.sort_by(|a, b| {
        (b.total_tokens, b.sessions).cmp(&(a.total_tokens, a.sessions))
    });
    result
}

fn compute_platform_breakdown(sessions: &[SessionRow]) -> Vec<PlatformBreakdown> {
    let mut order: Vec<String> = Vec::new();
    let mut platform_data: HashMap<String, PlatformBreakdown> = HashMap::new();

    for s in sessions {
        let source = s.source.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| "unknown".to_string());
        let d = platform_data.entry(source.clone()).or_insert_with(|| {
            order.push(source.clone());
            PlatformBreakdown {
                platform: source.clone(),
                ..Default::default()
            }
        });
        d.sessions += 1;
        d.messages += s.message_count.unwrap_or(0);
        let inp = s.input_tokens.unwrap_or(0);
        let out = s.output_tokens.unwrap_or(0);
        let cache_read = s.cache_read_tokens.unwrap_or(0);
        let cache_write = s.cache_write_tokens.unwrap_or(0);
        d.input_tokens += inp;
        d.output_tokens += out;
        d.cache_read_tokens += cache_read;
        d.cache_write_tokens += cache_write;
        d.total_tokens += inp + out + cache_read + cache_write;
        d.tool_calls += s.tool_call_count.unwrap_or(0);
    }

    let mut result: Vec<PlatformBreakdown> =
        order.into_iter().filter_map(|k| platform_data.remove(&k)).collect();
    result.sort_by(|a, b| b.sessions.cmp(&a.sessions));
    result
}

fn compute_tool_breakdown(tool_usage: &[ToolUsage]) -> Vec<ToolBreakdown> {
    let total_calls: i64 = tool_usage.iter().map(|t| t.count).sum();
    tool_usage
        .iter()
        .map(|t| ToolBreakdown {
            tool: t.tool_name.clone(),
            count: t.count,
            percentage: if total_calls != 0 {
                t.count as f64 / total_calls as f64 * 100.0
            } else {
                0.0
            },
        })
        .collect()
}

fn compute_skill_breakdown(skill_usage: &[SkillUsage]) -> SkillBreakdown {
    let total_skill_loads: i64 = skill_usage.iter().map(|s| s.view_count).sum();
    let total_skill_edits: i64 = skill_usage.iter().map(|s| s.manage_count).sum();
    let total_skill_actions = total_skill_loads + total_skill_edits;

    let mut top_skills: Vec<SkillEntry> = skill_usage
        .iter()
        .map(|skill| {
            let total_count = skill.view_count + skill.manage_count;
            let percentage = if total_skill_actions != 0 {
                total_count as f64 / total_skill_actions as f64 * 100.0
            } else {
                0.0
            };
            SkillEntry {
                skill: skill.skill.clone(),
                view_count: skill.view_count,
                manage_count: skill.manage_count,
                total_count,
                percentage,
                last_used_at: skill.last_used_at,
            }
        })
        .collect();

    // Sort by (total_count, view_count, manage_count, last_used_at or 0, skill) desc.
    top_skills.sort_by(|a, b| {
        let ka = (
            a.total_count,
            a.view_count,
            a.manage_count,
            a.last_used_at.unwrap_or(0.0),
        );
        let kb = (
            b.total_count,
            b.view_count,
            b.manage_count,
            b.last_used_at.unwrap_or(0.0),
        );
        // Descending on the numeric tuple; for the string key the Python tuple
        // sorts the skill name in the same (reverse) direction, so b.skill vs a.skill.
        kb.0.cmp(&ka.0)
            .then(kb.1.cmp(&ka.1))
            .then(kb.2.cmp(&ka.2))
            .then(kb.3.partial_cmp(&ka.3).unwrap_or(std::cmp::Ordering::Equal))
            .then(b.skill.cmp(&a.skill))
    });

    SkillBreakdown {
        summary: SkillSummary {
            total_skill_loads,
            total_skill_edits,
            total_skill_actions,
            distinct_skills_used: skill_usage.len() as i64,
        },
        top_skills,
    }
}

fn compute_activity_patterns(sessions: &[SessionRow]) -> Activity {
    let mut day_counts: HashMap<u32, i64> = HashMap::new();
    let mut hour_counts: HashMap<u32, i64> = HashMap::new();
    let mut daily_counts: HashMap<String, i64> = HashMap::new();

    for s in sessions {
        let ts = match s.started_at {
            Some(t) if t != 0.0 => t,
            _ => continue,
        };
        let dt = match local_datetime(ts) {
            Some(dt) => dt,
            None => continue,
        };
        // Python weekday(): Monday=0 ... Sunday=6. chrono num_days_from_monday() matches.
        let weekday = dt.weekday().num_days_from_monday();
        *day_counts.entry(weekday).or_insert(0) += 1;
        *hour_counts.entry(dt.hour()).or_insert(0) += 1;
        let datestr = dt.format("%Y-%m-%d").to_string();
        *daily_counts.entry(datestr).or_insert(0) += 1;
    }

    let day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    let day_breakdown: Vec<DayBucket> = (0..7u32)
        .map(|i| DayBucket {
            day: day_names[i as usize].to_string(),
            count: day_counts.get(&i).copied().unwrap_or(0),
        })
        .collect();
    let hour_breakdown: Vec<HourBucket> = (0..24i64)
        .map(|i| HourBucket {
            hour: i,
            count: hour_counts.get(&(i as u32)).copied().unwrap_or(0),
        })
        .collect();

    // max() with ties returns the first maximal element (Python max behavior).
    let busiest_day = day_breakdown
        .iter()
        .cloned()
        .reduce(|a, b| if b.count > a.count { b } else { a });
    let busiest_hour = hour_breakdown
        .iter()
        .cloned()
        .reduce(|a, b| if b.count > a.count { b } else { a });

    let active_days = daily_counts.len() as i64;

    let max_streak = if !daily_counts.is_empty() {
        let mut all_dates: Vec<String> = daily_counts.keys().cloned().collect();
        all_dates.sort();
        let mut current_streak = 1_i64;
        let mut max_streak = 1_i64;
        for i in 1..all_dates.len() {
            let d1 = NaiveDate::parse_from_str(&all_dates[i - 1], "%Y-%m-%d");
            let d2 = NaiveDate::parse_from_str(&all_dates[i], "%Y-%m-%d");
            if let (Ok(d1), Ok(d2)) = (d1, d2) {
                if (d2 - d1).num_days() == 1 {
                    current_streak += 1;
                    max_streak = max_streak.max(current_streak);
                } else {
                    current_streak = 1;
                }
            }
        }
        max_streak
    } else {
        0
    };

    Activity {
        by_day: day_breakdown,
        by_hour: hour_breakdown,
        busiest_day,
        busiest_hour,
        active_days,
        max_streak,
    }
}

fn compute_top_sessions(sessions: &[SessionRow]) -> Vec<TopSession> {
    let mut top: Vec<TopSession> = Vec::new();

    // Longest by duration.
    let sessions_with_duration: Vec<&SessionRow> = sessions
        .iter()
        .filter(|s| {
            s.started_at.map(|t| t != 0.0).unwrap_or(false)
                && s.ended_at.map(|t| t != 0.0).unwrap_or(false)
        })
        .collect();
    if !sessions_with_duration.is_empty() {
        // Python max() returns first on ties; reduce keeps first when not strictly greater.
        let longest = sessions_with_duration
            .iter()
            .copied()
            .reduce(|a, b| {
                let da = a.ended_at.unwrap() - a.started_at.unwrap();
                let db = b.ended_at.unwrap() - b.started_at.unwrap();
                if db > da {
                    b
                } else {
                    a
                }
            })
            .unwrap();
        let dur = longest.ended_at.unwrap() - longest.started_at.unwrap();
        top.push(TopSession {
            label: "Longest session".to_string(),
            session_id: truncate(&longest.id, 16),
            value: format_duration(dur),
            date: fmt_date_mon_day(longest.started_at).unwrap_or_else(|| "?".to_string()),
        });
    }

    // Most messages.
    if let Some(most_msgs) = max_by_key_first(sessions, |s| s.message_count.unwrap_or(0)) {
        let mc = most_msgs.message_count.unwrap_or(0);
        if mc > 0 {
            top.push(TopSession {
                label: "Most messages".to_string(),
                session_id: truncate(&most_msgs.id, 16),
                value: format!("{} msgs", mc),
                date: fmt_date_mon_day(most_msgs.started_at).unwrap_or_else(|| "?".to_string()),
            });
        }
    }

    // Most tokens (input + output).
    if let Some(most_tokens) = max_by_key_first(sessions, |s| {
        s.input_tokens.unwrap_or(0) + s.output_tokens.unwrap_or(0)
    }) {
        let token_total =
            most_tokens.input_tokens.unwrap_or(0) + most_tokens.output_tokens.unwrap_or(0);
        if token_total > 0 {
            top.push(TopSession {
                label: "Most tokens".to_string(),
                session_id: truncate(&most_tokens.id, 16),
                value: format!("{} tokens", comma(token_total)),
                date: fmt_date_mon_day(most_tokens.started_at).unwrap_or_else(|| "?".to_string()),
            });
        }
    }

    // Most tool calls.
    if let Some(most_tools) = max_by_key_first(sessions, |s| s.tool_call_count.unwrap_or(0)) {
        let tc = most_tools.tool_call_count.unwrap_or(0);
        if tc > 0 {
            top.push(TopSession {
                label: "Most tool calls".to_string(),
                session_id: truncate(&most_tools.id, 16),
                value: format!("{} calls", tc),
                date: fmt_date_mon_day(most_tools.started_at).unwrap_or_else(|| "?".to_string()),
            });
        }
    }

    top
}

/// Python `max(iterable, key=...)`: returns the first element with the maximal key.
fn max_by_key_first<F>(sessions: &[SessionRow], key: F) -> Option<&SessionRow>
where
    F: Fn(&SessionRow) -> i64,
{
    sessions.iter().reduce(|a, b| if key(b) > key(a) { b } else { a })
}

fn truncate(s: &str, n: usize) -> String {
    // Python slicing s[:16] works on characters.
    s.chars().take(n).collect()
}

/// Format a unix timestamp as "%b %d" in local time (e.g. "Jun 03").
fn fmt_date_mon_day(ts: Option<f64>) -> Option<String> {
    let ts = ts.filter(|&t| t != 0.0)?;
    let dt = local_datetime(ts)?;
    Some(dt.format("%b %d").to_string())
}

/// Convert a unix timestamp (seconds, possibly fractional) to local datetime.
/// Mirrors Python `datetime.fromtimestamp`.
fn local_datetime(ts: f64) -> Option<chrono::DateTime<Local>> {
    let secs = ts.floor() as i64;
    let nanos = ((ts - ts.floor()) * 1_000_000_000.0).round() as u32;
    match Local.timestamp_opt(secs, nanos) {
        chrono::LocalResult::Single(dt) => Some(dt),
        chrono::LocalResult::Ambiguous(dt, _) => Some(dt),
        chrono::LocalResult::None => None,
    }
}

// =============================================================================
// Formatting (ported from format_terminal / format_gateway)
// =============================================================================

/// Pad-right a string to a given display width (mimics Python `{:<width}`).
fn ljust(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - len))
    }
}

/// Pad-left to width (mimics `{:>width}`).
fn rjust(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(width - len), s)
    }
}

pub fn format_terminal(report: &InsightsReport) -> String {
    if report.empty {
        let days = report.days;
        let src = match &report.source_filter {
            Some(s) if !s.is_empty() => format!(" (source: {s})"),
            _ => String::new(),
        };
        return format!("  No sessions found in the last {days} days{src}.");
    }

    let mut lines: Vec<String> = Vec::new();
    let o = &report.overview;
    let days = report.days;
    let src_filter = report.source_filter.as_deref().filter(|s| !s.is_empty());

    lines.push(String::new());
    lines.push("  ╔══════════════════════════════════════════════════════════╗".to_string());
    lines.push("  ║                    📊 Hermes Insights                    ║".to_string());
    let mut period_label = format!("Last {days} days");
    if let Some(sf) = src_filter {
        period_label.push_str(&format!(" ({sf})"));
    }
    // padding = 58 - len - 2
    let padding = 58i64 - period_label.chars().count() as i64 - 2;
    let padding = padding.max(0) as usize;
    let left_pad = padding / 2;
    let right_pad = padding - left_pad;
    lines.push(format!(
        "  ║{} {} {}║",
        " ".repeat(left_pad),
        period_label,
        " ".repeat(right_pad)
    ));
    lines.push("  ╚══════════════════════════════════════════════════════════╝".to_string());
    lines.push(String::new());

    if let (Some(start), Some(end)) = (o.date_range_start, o.date_range_end) {
        if start != 0.0 && end != 0.0 {
            let start_str = local_datetime(start)
                .map(|d| d.format("%b %d, %Y").to_string())
                .unwrap_or_default();
            let end_str = local_datetime(end)
                .map(|d| d.format("%b %d, %Y").to_string())
                .unwrap_or_default();
            lines.push(format!("  Period: {start_str} — {end_str}"));
            lines.push(String::new());
        }
    }

    // Overview
    lines.push("  📋 Overview".to_string());
    lines.push(format!("  {}", "─".repeat(56)));
    lines.push(format!(
        "  Sessions:          {}  Messages:        {}",
        ljust(&o.total_sessions.to_string(), 12),
        comma(o.total_messages)
    ));
    lines.push(format!(
        "  Tool calls:        {}  User messages:   {}",
        ljust(&comma(o.total_tool_calls), 12),
        comma(o.user_messages)
    ));
    lines.push(format!(
        "  Input tokens:      {}  Output tokens:   {}",
        ljust(&comma(o.total_input_tokens), 12),
        comma(o.total_output_tokens)
    ));
    lines.push(format!("  Total tokens:      {}", comma(o.total_tokens)));
    if o.total_hours > 0.0 {
        lines.push(format!(
            "  Active time:       ~{}  Avg session:     ~{}",
            ljust(&format_duration(o.total_hours * 3600.0), 11),
            format_duration(o.avg_session_duration)
        ));
    }
    lines.push(format!(
        "  Avg msgs/session:  {:.1}",
        o.avg_messages_per_session
    ));
    lines.push(String::new());

    // Model breakdown
    if !report.models.is_empty() {
        lines.push("  🤖 Models Used".to_string());
        lines.push(format!("  {}", "─".repeat(56)));
        lines.push(format!(
            "  {} {} {}",
            ljust("Model", 30),
            rjust("Sessions", 8),
            rjust("Tokens", 12)
        ));
        for m in &report.models {
            let model_name = truncate(&m.model, 28);
            lines.push(format!(
                "  {} {} {}",
                ljust(&model_name, 30),
                rjust(&m.sessions.to_string(), 8),
                rjust(&comma(m.total_tokens), 12)
            ));
        }
        lines.push(String::new());
    }

    // Platform breakdown
    let show_platforms = report.platforms.len() > 1
        || (!report.platforms.is_empty() && report.platforms[0].platform != "cli");
    if show_platforms {
        lines.push("  📱 Platforms".to_string());
        lines.push(format!("  {}", "─".repeat(56)));
        lines.push(format!(
            "  {} {} {} {}",
            ljust("Platform", 14),
            rjust("Sessions", 8),
            rjust("Messages", 10),
            rjust("Tokens", 14)
        ));
        for p in &report.platforms {
            lines.push(format!(
                "  {} {} {} {}",
                ljust(&p.platform, 14),
                rjust(&p.sessions.to_string(), 8),
                rjust(&comma(p.messages), 10),
                rjust(&comma(p.total_tokens), 14)
            ));
        }
        lines.push(String::new());
    }

    // Tool usage
    if !report.tools.is_empty() {
        lines.push("  🔧 Top Tools".to_string());
        lines.push(format!("  {}", "─".repeat(56)));
        lines.push(format!(
            "  {} {} {}",
            ljust("Tool", 28),
            rjust("Calls", 8),
            rjust("%", 8)
        ));
        for t in report.tools.iter().take(15) {
            let pct = format!("{:.1}", t.percentage);
            lines.push(format!(
                "  {} {} {}%",
                ljust(&t.tool, 28),
                rjust(&comma(t.count), 8),
                rjust(&pct, 7)
            ));
        }
        if report.tools.len() > 15 {
            lines.push(format!("  ... and {} more tools", report.tools.len() - 15));
        }
        lines.push(String::new());
    }

    // Skill usage
    let top_skills = &report.skills.top_skills;
    if !top_skills.is_empty() {
        lines.push("  🧠 Top Skills".to_string());
        lines.push(format!("  {}", "─".repeat(56)));
        lines.push(format!(
            "  {} {} {} {}",
            ljust("Skill", 28),
            rjust("Loads", 7),
            rjust("Edits", 7),
            rjust("Last used", 11)
        ));
        for skill in top_skills.iter().take(10) {
            let last_used = match skill.last_used_at.filter(|&t| t != 0.0) {
                Some(ts) => local_datetime(ts)
                    .map(|d| d.format("%b %d").to_string())
                    .unwrap_or_else(|| "—".to_string()),
                None => "—".to_string(),
            };
            lines.push(format!(
                "  {} {} {} {}",
                ljust(&truncate(&skill.skill, 28), 28),
                rjust(&comma(skill.view_count), 7),
                rjust(&comma(skill.manage_count), 7),
                rjust(&last_used, 11)
            ));
        }
        let summary = &report.skills.summary;
        lines.push(format!(
            "  Distinct skills: {}  Loads: {}  Edits: {}",
            summary.distinct_skills_used,
            comma(summary.total_skill_loads),
            comma(summary.total_skill_edits)
        ));
        lines.push(String::new());
    }

    // Activity patterns
    let act = &report.activity;
    if !act.by_day.is_empty() {
        lines.push("  📅 Activity Patterns".to_string());
        lines.push(format!("  {}", "─".repeat(56)));

        let day_values: Vec<i64> = act.by_day.iter().map(|d| d.count).collect();
        let bars = bar_chart(&day_values, 15);
        for (i, d) in act.by_day.iter().enumerate() {
            let bar = bars.get(i).cloned().unwrap_or_default();
            lines.push(format!("  {}  {} {}", d.day, ljust(&bar, 15), d.count));
        }
        lines.push(String::new());

        // Peak hours (top 5 busiest with count > 0).
        let mut busy_hours: Vec<&HourBucket> = act.by_hour.iter().collect();
        // sorted desc by count, stable (Python list.sort is stable).
        busy_hours.sort_by(|a, b| b.count.cmp(&a.count));
        let busy_hours: Vec<&HourBucket> =
            busy_hours.into_iter().filter(|h| h.count > 0).take(5).collect();
        if !busy_hours.is_empty() {
            let hour_strs: Vec<String> = busy_hours
                .iter()
                .map(|h| {
                    let hr = h.hour;
                    let ampm = if hr < 12 { "AM" } else { "PM" };
                    let display_hr = {
                        let m = hr % 12;
                        if m == 0 { 12 } else { m }
                    };
                    format!("{}{} ({})", display_hr, ampm, h.count)
                })
                .collect();
            lines.push(format!("  Peak hours: {}", hour_strs.join(", ")));
        }

        if act.active_days != 0 {
            lines.push(format!("  Active days: {}", act.active_days));
        }
        if act.max_streak > 1 {
            lines.push(format!("  Best streak: {} consecutive days", act.max_streak));
        }
        lines.push(String::new());
    }

    // Notable sessions
    if !report.top_sessions.is_empty() {
        lines.push("  🏆 Notable Sessions".to_string());
        lines.push(format!("  {}", "─".repeat(56)));
        for ts in &report.top_sessions {
            lines.push(format!(
                "  {} {} ({}, {})",
                ljust(&ts.label, 20),
                ljust(&ts.value, 18),
                ts.date,
                ts.session_id
            ));
        }
        lines.push(String::new());
    }

    lines.join("\n")
}

pub fn format_gateway(report: &InsightsReport) -> String {
    if report.empty {
        let days = report.days;
        return format!("No sessions found in the last {days} days.");
    }

    let mut lines: Vec<String> = Vec::new();
    let o = &report.overview;
    let days = report.days;

    lines.push(format!("📊 **Hermes Insights** — Last {days} days\n"));

    lines.push(format!(
        "**Sessions:** {} | **Messages:** {} | **Tool calls:** {}",
        o.total_sessions,
        comma(o.total_messages),
        comma(o.total_tool_calls)
    ));
    lines.push(format!(
        "**Tokens:** {} (in: {} / out: {})",
        comma(o.total_tokens),
        comma(o.total_input_tokens),
        comma(o.total_output_tokens)
    ));
    if o.total_hours > 0.0 {
        lines.push(format!(
            "**Active time:** ~{} | **Avg session:** ~{}",
            format_duration(o.total_hours * 3600.0),
            format_duration(o.avg_session_duration)
        ));
    }
    lines.push(String::new());

    if !report.models.is_empty() {
        lines.push("**🤖 Models:**".to_string());
        for m in report.models.iter().take(5) {
            lines.push(format!(
                "  {} — {} sessions, {} tokens",
                truncate(&m.model, 25),
                m.sessions,
                comma(m.total_tokens)
            ));
        }
        lines.push(String::new());
    }

    if report.platforms.len() > 1 {
        lines.push("**📱 Platforms:**".to_string());
        for p in &report.platforms {
            lines.push(format!(
                "  {} — {} sessions, {} msgs",
                p.platform,
                p.sessions,
                comma(p.messages)
            ));
        }
        lines.push(String::new());
    }

    if !report.tools.is_empty() {
        lines.push("**🔧 Top Tools:**".to_string());
        for t in report.tools.iter().take(8) {
            lines.push(format!(
                "  {} — {} calls ({:.1}%)",
                t.tool,
                comma(t.count),
                t.percentage
            ));
        }
        lines.push(String::new());
    }

    if !report.skills.top_skills.is_empty() {
        lines.push("**🧠 Top Skills:**".to_string());
        for skill in report.skills.top_skills.iter().take(5) {
            let suffix = match skill.last_used_at.filter(|&t| t != 0.0) {
                Some(ts) => local_datetime(ts)
                    .map(|d| format!(", last used {}", d.format("%b %d")))
                    .unwrap_or_default(),
                None => String::new(),
            };
            lines.push(format!(
                "  {} — {} loads, {} edits{}",
                skill.skill,
                comma(skill.view_count),
                comma(skill.manage_count),
                suffix
            ));
        }
        lines.push(String::new());
    }

    let act = &report.activity;
    if let (Some(bd), Some(bh)) = (&act.busiest_day, &act.busiest_hour) {
        let hr = bh.hour;
        let ampm = if hr < 12 { "AM" } else { "PM" };
        let display_hr = {
            let m = hr % 12;
            if m == 0 { 12 } else { m }
        };
        lines.push(format!(
            "**📅 Busiest:** {}s ({} sessions), {}{} ({} sessions)",
            bd.day, bd.count, display_hr, ampm, bh.count
        ));
        if act.active_days != 0 {
            lines.push(format!("**Active days:** {}", act.active_days));
        }
        if act.max_streak > 1 {
            lines.push(format!("**Best streak:** {} consecutive days", act.max_streak));
        }
    }

    lines.join("\n")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                started_at REAL,
                ended_at REAL,
                message_count INTEGER,
                tool_call_count INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                timestamp REAL
            );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(format_duration_compact(30.0), "30s");
        assert_eq!(format_duration_compact(90.0), "2m"); // 1.5m -> {:.0} rounds to 2
        assert_eq!(format_duration_compact(3600.0), "1h");
        assert_eq!(format_duration_compact(3600.0 + 1800.0), "1h 30m");
        assert_eq!(format_duration_compact(86400.0 * 2.5), "2.5d");
    }

    #[test]
    fn test_comma_formatting() {
        assert_eq!(comma(0), "0");
        assert_eq!(comma(999), "999");
        assert_eq!(comma(1000), "1,000");
        assert_eq!(comma(1234567), "1,234,567");
        assert_eq!(comma(-12345), "-12,345");
    }

    #[test]
    fn test_bar_chart() {
        assert_eq!(bar_chart(&[], 20), Vec::<String>::new());
        assert_eq!(bar_chart(&[0, 0], 20), vec!["".to_string(), "".to_string()]);
        let bars = bar_chart(&[10, 5, 0], 10);
        assert_eq!(bars[0], "█".repeat(10));
        assert_eq!(bars[1], "█".repeat(5));
        assert_eq!(bars[2], "");
    }

    #[test]
    fn test_base_url_host_matches() {
        assert!(base_url_host_matches("https://api.moonshot.ai/v1", "moonshot.ai"));
        assert!(base_url_host_matches("https://moonshot.ai", "moonshot.ai"));
        assert!(!base_url_host_matches("https://evil.com/moonshot.ai/v1", "moonshot.ai"));
        assert!(!base_url_host_matches("https://moonshot.ai.evil/v1", "moonshot.ai"));
        assert!(base_url_host_matches("https://openrouter.ai/api/v1", "openrouter.ai"));
    }

    #[test]
    fn test_resolve_billing_route_inference() {
        let r = resolve_billing_route("anthropic/claude-opus-4-20250514", None, None);
        assert_eq!(r.provider, "anthropic");
        assert_eq!(r.model, "claude-opus-4-20250514");
        assert_eq!(r.billing_mode, "official_docs_snapshot");

        let r = resolve_billing_route("gpt-4o", Some("openai-codex"), None);
        assert_eq!(r.billing_mode, "subscription_included");

        let r = resolve_billing_route("foo", None, Some("https://openrouter.ai/api/v1"));
        assert_eq!(r.provider, "openrouter");
    }

    #[test]
    fn test_estimate_usage_cost_known() {
        let usage = CanonicalUsage::new(1_000_000, 1_000_000, 0, 0);
        let res = estimate_usage_cost("claude-opus-4-20250514", &usage, Some("anthropic"), None);
        assert_eq!(res.status, CostStatus::Estimated);
        // 15 + 75 = 90
        assert!((res.amount_usd.unwrap() - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_usage_cost_unknown() {
        let usage = CanonicalUsage::new(1000, 1000, 0, 0);
        let res = estimate_usage_cost("some-random-model", &usage, Some("unknownprovider"), None);
        assert_eq!(res.status, CostStatus::Unknown);
        assert!(res.amount_usd.is_none());
    }

    #[test]
    fn test_included_route() {
        let usage = CanonicalUsage::new(1000, 1000, 0, 0);
        let res = estimate_usage_cost("gpt-5", &usage, Some("openai-codex"), None);
        assert_eq!(res.status, CostStatus::Included);
        assert_eq!(res.amount_usd, Some(0.0));
        assert!(has_known_pricing("gpt-5", Some("openai-codex"), None));
    }

    #[test]
    fn test_empty_report() {
        let conn = setup_db();
        let engine = InsightsEngine::new(&conn);
        let report = engine.generate(30, None).unwrap();
        assert!(report.empty);
        assert_eq!(report.days, 30);
        let term = engine.format_terminal(&report);
        assert!(term.contains("No sessions found in the last 30 days"));
        let gw = engine.format_gateway(&report);
        assert!(gw.contains("No sessions found in the last 30 days."));
    }

    #[test]
    fn test_full_report() {
        let conn = setup_db();
        let now = now_unix();
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, ended_at, message_count, \
             tool_call_count, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, \
             billing_provider, actual_cost_usd) VALUES \
             (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![
                "session-abcdef0123456789",
                "telegram",
                "anthropic/claude-opus-4-20250514",
                now - 3600.0,
                now - 600.0,
                40i64,
                10i64,
                1_000_000i64,
                500_000i64,
                0i64,
                0i64,
                "anthropic",
                0.5f64,
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (id, session_id, role, tool_name, timestamp) \
             VALUES ('m1', 'session-abcdef0123456789', 'tool', 'Bash', ?1)",
            rusqlite::params![now - 1000.0],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (id, session_id, role, tool_calls, timestamp) \
             VALUES ('m2', 'session-abcdef0123456789', 'assistant', ?1, ?2)",
            rusqlite::params![
                r#"[{"function":{"name":"skill_view","arguments":"{\"name\":\"deep-research\"}"}}]"#,
                now - 1000.0
            ],
        )
        .unwrap();

        let engine = InsightsEngine::new(&conn);
        let report = engine.generate(30, None).unwrap();
        assert!(!report.empty);
        assert_eq!(report.overview.total_sessions, 1);
        assert_eq!(report.overview.total_messages, 40);
        assert_eq!(report.overview.total_input_tokens, 1_000_000);
        assert!(report.overview.estimated_cost > 0.0);

        // Model breakdown strips provider prefix.
        assert_eq!(report.models.len(), 1);
        assert_eq!(report.models[0].model, "claude-opus-4-20250514");
        assert_eq!(report.models[0].total_tokens, 1_500_000);
        assert!(report.models[0].has_pricing);

        // Platforms
        assert_eq!(report.platforms.len(), 1);
        assert_eq!(report.platforms[0].platform, "telegram");

        // Tools: Bash (from tool_name) + skill_view (from tool_calls)
        assert!(report.tools.iter().any(|t| t.tool == "Bash" && t.count == 1));
        assert!(report.tools.iter().any(|t| t.tool == "skill_view"));

        // Skills
        assert_eq!(report.skills.summary.distinct_skills_used, 1);
        assert_eq!(report.skills.top_skills[0].skill, "deep-research");
        assert_eq!(report.skills.top_skills[0].view_count, 1);

        // Top sessions: should include a longest session and most messages/tokens/tools
        assert!(report.top_sessions.iter().any(|t| t.label == "Longest session"));
        assert!(report.top_sessions.iter().any(|t| t.label == "Most messages"));

        // Terminal + gateway formatting should not panic and include the header.
        let term = engine.format_terminal(&report);
        assert!(term.contains("Hermes Insights"));
        assert!(term.contains("Models Used"));
        let gw = engine.format_gateway(&report);
        assert!(gw.contains("Hermes Insights"));
    }

    #[test]
    fn test_tool_merge_max() {
        // Both sources have data; merge takes the per-tool max.
        let mut a: HashMap<String, i64> = HashMap::new();
        a.insert("Bash".into(), 5);
        let mut b: HashMap<String, i64> = HashMap::new();
        b.insert("Bash".into(), 3);
        b.insert("Read".into(), 7);
        // Reproduce merge logic
        let mut all: HashSet<String> = HashSet::new();
        all.extend(a.keys().cloned());
        all.extend(b.keys().cloned());
        let mut merged: HashMap<String, i64> = HashMap::new();
        for t in all {
            merged.insert(
                t.clone(),
                std::cmp::max(a.get(&t).copied().unwrap_or(0), b.get(&t).copied().unwrap_or(0)),
            );
        }
        let common = most_common(merged);
        assert_eq!(common[0].tool_name, "Read");
        assert_eq!(common[0].count, 7);
        assert_eq!(common[1].tool_name, "Bash");
        assert_eq!(common[1].count, 5);
    }

    #[test]
    fn test_activity_streak() {
        let conn = setup_db();
        // Three consecutive days at noon local time, plus a gap day.
        let base = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let insert_day = |i: i64, day: NaiveDate| {
            let dt = day.and_hms_opt(12, 0, 0).unwrap();
            let ts = Local.from_local_datetime(&dt).unwrap().timestamp() as f64;
            conn.execute(
                "INSERT INTO sessions (id, source, model, started_at, ended_at, message_count) \
                 VALUES (?1, 'cli', 'gpt-4o', ?2, ?3, 1)",
                rusqlite::params![format!("s{i}"), ts, ts + 60.0],
            )
            .unwrap();
        };
        insert_day(0, base);
        insert_day(1, base.succ_opt().unwrap());
        insert_day(2, base.succ_opt().unwrap().succ_opt().unwrap());
        // gap, then another day
        insert_day(3, NaiveDate::from_ymd_opt(2026, 1, 20).unwrap());

        let engine = InsightsEngine::new(&conn);
        // large window so the 2026-01 dates are included regardless of "now"
        let report = engine.generate(100000, None).unwrap();
        assert_eq!(report.activity.active_days, 4);
        assert_eq!(report.activity.max_streak, 3);
    }
}
