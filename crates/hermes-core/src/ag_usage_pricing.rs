//! Usage normalization and cost estimation.
//!
//! Native Rust port of `agent/usage_pricing.py`. Resolves a billing route for a
//! given model/provider/base-url, looks up a pricing entry (official docs
//! snapshot, OpenRouter models API, or an OpenAI-compatible endpoint's `/models`
//! response), normalizes raw provider usage into canonical token buckets, and
//! estimates a USD cost.
//!
//! Cross references:
//! - [`crate::ag_model_metadata::fetch_model_metadata`]
//! - [`crate::ag_model_metadata::fetch_endpoint_model_metadata`]
//! - [`crate::mod_utils::base_url_host_matches`]
//!
//! The Python original uses `decimal.Decimal` for exact arithmetic. Pricing
//! magnitudes here (dollars-per-million-tokens with at most a few significant
//! digits, multiplied by integer token counts) are well within `f64`'s exact
//! range for the relevant operations, so cost amounts are represented as `f64`.
//! The `:.2f` label formatting and `== 0` "included" detection are preserved.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::ag_model_metadata::{fetch_endpoint_model_metadata, fetch_model_metadata};
use crate::mod_utils::base_url_host_matches;

const _ONE_MILLION: f64 = 1_000_000.0;

/// Cost confidence status. Mirrors the Python `CostStatus` literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostStatus {
    Actual,
    Estimated,
    Included,
    Unknown,
}

impl CostStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CostStatus::Actual => "actual",
            CostStatus::Estimated => "estimated",
            CostStatus::Included => "included",
            CostStatus::Unknown => "unknown",
        }
    }
}

/// Where a pricing figure came from. Mirrors the Python `CostSource` literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSource {
    ProviderCostApi,
    ProviderGenerationApi,
    ProviderModelsApi,
    OfficialDocsSnapshot,
    UserOverride,
    CustomContract,
    None,
}

impl CostSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CostSource::ProviderCostApi => "provider_cost_api",
            CostSource::ProviderGenerationApi => "provider_generation_api",
            CostSource::ProviderModelsApi => "provider_models_api",
            CostSource::OfficialDocsSnapshot => "official_docs_snapshot",
            CostSource::UserOverride => "user_override",
            CostSource::CustomContract => "custom_contract",
            CostSource::None => "none",
        }
    }
}

/// Canonical, provider-agnostic token buckets for a single request (or
/// aggregated set of requests via `request_count`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub reasoning_tokens: i64,
    pub request_count: i64,
    pub raw_usage: Option<Value>,
}

impl Default for CanonicalUsage {
    fn default() -> Self {
        CanonicalUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
            request_count: 1,
            raw_usage: None,
        }
    }
}

impl CanonicalUsage {
    pub fn prompt_tokens(&self) -> i64 {
        self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    pub fn total_tokens(&self) -> i64 {
        self.prompt_tokens() + self.output_tokens
    }
}

/// Resolved billing route: which provider/model/base-url combination is being
/// billed, and under which billing mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BillingRoute {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub billing_mode: String,
}

/// A pricing entry. All cost figures are dollars-per-million-tokens except
/// `request_cost`, which is dollars-per-request.
#[derive(Debug, Clone, PartialEq)]
pub struct PricingEntry {
    pub input_cost_per_million: Option<f64>,
    pub output_cost_per_million: Option<f64>,
    pub cache_read_cost_per_million: Option<f64>,
    pub cache_write_cost_per_million: Option<f64>,
    pub request_cost: Option<f64>,
    pub source: CostSource,
    pub source_url: Option<String>,
    pub pricing_version: Option<String>,
    pub fetched_at: Option<DateTime<Utc>>,
}

impl Default for PricingEntry {
    fn default() -> Self {
        PricingEntry {
            input_cost_per_million: None,
            output_cost_per_million: None,
            cache_read_cost_per_million: None,
            cache_write_cost_per_million: None,
            request_cost: None,
            source: CostSource::None,
            source_url: None,
            pricing_version: None,
            fetched_at: None,
        }
    }
}

/// Result of a cost estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct CostResult {
    pub amount_usd: Option<f64>,
    pub status: CostStatus,
    pub source: CostSource,
    pub label: String,
    pub fetched_at: Option<DateTime<Utc>>,
    pub pricing_version: Option<String>,
    pub notes: Vec<String>,
}

fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Raw usage payload as seen on a provider response. Mirrors the duck-typed
/// `response_usage` object the Python uses (it reads attributes off whatever
/// object the SDK returns). Construct from a `serde_json::Value` via
/// [`UsageFields::from_value`], or fill fields directly.
#[derive(Debug, Clone, Default)]
pub struct UsageFields {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    /// `input_tokens_details.cached_tokens`
    pub input_details_cached_tokens: Option<i64>,
    /// `input_tokens_details.cache_creation_tokens`
    pub input_details_cache_creation_tokens: Option<i64>,
    /// `prompt_tokens_details.cached_tokens`
    pub prompt_details_cached_tokens: Option<i64>,
    /// `prompt_tokens_details.cache_write_tokens`
    pub prompt_details_cache_write_tokens: Option<i64>,
    /// `output_tokens_details.reasoning_tokens`
    pub output_details_reasoning_tokens: Option<i64>,
    /// Whether any usage data was present at all (mirrors Python's
    /// `if not response_usage` truthiness check).
    pub present: bool,
}

impl UsageFields {
    /// Build from a JSON usage object, reading both flat and nested
    /// (`*_details`) shapes. An empty/`null` value yields a not-present record.
    pub fn from_value(usage: &Value) -> Self {
        let obj = match usage {
            Value::Object(m) if !m.is_empty() => m,
            _ => return UsageFields::default(),
        };
        let get_int = |key: &str| -> Option<i64> { obj.get(key).and_then(value_to_int) };
        let nested_int = |parent: &str, child: &str| -> Option<i64> {
            obj.get(parent)
                .and_then(|p| p.as_object())
                .and_then(|m| m.get(child))
                .and_then(value_to_int)
        };
        UsageFields {
            input_tokens: get_int("input_tokens"),
            output_tokens: get_int("output_tokens"),
            prompt_tokens: get_int("prompt_tokens"),
            completion_tokens: get_int("completion_tokens"),
            cache_read_input_tokens: get_int("cache_read_input_tokens"),
            cache_creation_input_tokens: get_int("cache_creation_input_tokens"),
            input_details_cached_tokens: nested_int("input_tokens_details", "cached_tokens"),
            input_details_cache_creation_tokens: nested_int(
                "input_tokens_details",
                "cache_creation_tokens",
            ),
            prompt_details_cached_tokens: nested_int("prompt_tokens_details", "cached_tokens"),
            prompt_details_cache_write_tokens: nested_int(
                "prompt_tokens_details",
                "cache_write_tokens",
            ),
            output_details_reasoning_tokens: nested_int("output_tokens_details", "reasoning_tokens"),
            present: true,
        }
    }
}

/// Convert a JSON value to a Decimal-like `f64`, mirroring Python `_to_decimal`.
/// Returns `None` for `null` or anything not parseable as a number.
fn to_decimal(value: &Value) -> Option<f64> {
    match value {
        Value::Null => None,
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Mirror of Python `_to_int`: coerce to int, treating None/missing/falsey as 0.
fn value_to_int(value: &Value) -> Option<i64> {
    match value {
        Value::Null => Some(0),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        Value::String(s) => s.trim().parse::<i64>().ok().or_else(|| {
            s.trim().parse::<f64>().ok().map(|f| f as i64)
        }),
        Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => Some(0),
    }
}

/// `_to_int` over an `Option<i64>` field (already coerced upstream).
fn opt_to_int(value: Option<i64>) -> i64 {
    value.unwrap_or(0)
}

/// Official-docs snapshot pricing table, keyed by `(provider, model)`.
/// Model keys are stored lowercase and looked up lowercase.
fn official_docs_pricing() -> &'static HashMap<(&'static str, &'static str), PricingEntry> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<HashMap<(&'static str, &'static str), PricingEntry>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m: HashMap<(&'static str, &'static str), PricingEntry> = HashMap::new();
        let anthropic_cache_url =
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching";
        let openai_url = "https://openai.com/api/pricing/";
        let deepseek_url = "https://api-docs.deepseek.com/quick_start/pricing";
        let google_url = "https://ai.google.dev/pricing";
        let bedrock_url = "https://aws.amazon.com/bedrock/pricing/";

        let mk = |input: Option<f64>,
                  output: Option<f64>,
                  cache_read: Option<f64>,
                  cache_write: Option<f64>,
                  source_url: Option<&'static str>,
                  pricing_version: &'static str|
         -> PricingEntry {
            PricingEntry {
                input_cost_per_million: input,
                output_cost_per_million: output,
                cache_read_cost_per_million: cache_read,
                cache_write_cost_per_million: cache_write,
                request_cost: None,
                source: CostSource::OfficialDocsSnapshot,
                source_url: source_url.map(|s| s.to_string()),
                pricing_version: Some(pricing_version.to_string()),
                fetched_at: None,
            }
        };

        // Anthropic (current generation)
        m.insert(
            ("anthropic", "claude-opus-4-20250514"),
            mk(
                Some(15.00),
                Some(75.00),
                Some(1.50),
                Some(18.75),
                Some(anthropic_cache_url),
                "anthropic-prompt-caching-2026-03-16",
            ),
        );
        m.insert(
            ("anthropic", "claude-sonnet-4-20250514"),
            mk(
                Some(3.00),
                Some(15.00),
                Some(0.30),
                Some(3.75),
                Some(anthropic_cache_url),
                "anthropic-prompt-caching-2026-03-16",
            ),
        );

        // OpenAI
        m.insert(
            ("openai", "gpt-4o"),
            mk(
                Some(2.50),
                Some(10.00),
                Some(1.25),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "gpt-4o-mini"),
            mk(
                Some(0.15),
                Some(0.60),
                Some(0.075),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "gpt-4.1"),
            mk(
                Some(2.00),
                Some(8.00),
                Some(0.50),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "gpt-4.1-mini"),
            mk(
                Some(0.40),
                Some(1.60),
                Some(0.10),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "gpt-4.1-nano"),
            mk(
                Some(0.10),
                Some(0.40),
                Some(0.025),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "o3"),
            mk(
                Some(10.00),
                Some(40.00),
                Some(2.50),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("openai", "o3-mini"),
            mk(
                Some(1.10),
                Some(4.40),
                Some(0.55),
                None,
                Some(openai_url),
                "openai-pricing-2026-03-16",
            ),
        );

        // Anthropic older models (pre-4.6 generation)
        m.insert(
            ("anthropic", "claude-3-5-sonnet-20241022"),
            mk(
                Some(3.00),
                Some(15.00),
                Some(0.30),
                Some(3.75),
                Some(anthropic_cache_url),
                "anthropic-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("anthropic", "claude-3-5-haiku-20241022"),
            mk(
                Some(0.80),
                Some(4.00),
                Some(0.08),
                Some(1.00),
                Some(anthropic_cache_url),
                "anthropic-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("anthropic", "claude-3-opus-20240229"),
            mk(
                Some(15.00),
                Some(75.00),
                Some(1.50),
                Some(18.75),
                Some(anthropic_cache_url),
                "anthropic-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("anthropic", "claude-3-haiku-20240307"),
            mk(
                Some(0.25),
                Some(1.25),
                Some(0.03),
                Some(0.30),
                Some(anthropic_cache_url),
                "anthropic-pricing-2026-03-16",
            ),
        );

        // DeepSeek
        m.insert(
            ("deepseek", "deepseek-chat"),
            mk(
                Some(0.14),
                Some(0.28),
                None,
                None,
                Some(deepseek_url),
                "deepseek-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("deepseek", "deepseek-reasoner"),
            mk(
                Some(0.55),
                Some(2.19),
                None,
                None,
                Some(deepseek_url),
                "deepseek-pricing-2026-03-16",
            ),
        );

        // Google Gemini
        m.insert(
            ("google", "gemini-2.5-pro"),
            mk(
                Some(1.25),
                Some(10.00),
                None,
                None,
                Some(google_url),
                "google-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("google", "gemini-2.5-flash"),
            mk(
                Some(0.15),
                Some(0.60),
                None,
                None,
                Some(google_url),
                "google-pricing-2026-03-16",
            ),
        );
        m.insert(
            ("google", "gemini-2.0-flash"),
            mk(
                Some(0.10),
                Some(0.40),
                None,
                None,
                Some(google_url),
                "google-pricing-2026-03-16",
            ),
        );

        // AWS Bedrock (on-demand)
        m.insert(
            ("bedrock", "anthropic.claude-opus-4-6"),
            mk(
                Some(15.00),
                Some(75.00),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "anthropic.claude-sonnet-4-6"),
            mk(
                Some(3.00),
                Some(15.00),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "anthropic.claude-sonnet-4-5"),
            mk(
                Some(3.00),
                Some(15.00),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "anthropic.claude-haiku-4-5"),
            mk(
                Some(0.80),
                Some(4.00),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "amazon.nova-pro"),
            mk(
                Some(0.80),
                Some(3.20),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "amazon.nova-lite"),
            mk(
                Some(0.06),
                Some(0.24),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );
        m.insert(
            ("bedrock", "amazon.nova-micro"),
            mk(
                Some(0.035),
                Some(0.14),
                None,
                None,
                Some(bedrock_url),
                "bedrock-pricing-2026-04",
            ),
        );

        // MiniMax (no source_url in the snapshot)
        m.insert(
            ("minimax", "minimax-m2.7"),
            mk(
                Some(0.30),
                Some(1.20),
                None,
                None,
                None,
                "minimax-pricing-2026-04",
            ),
        );
        m.insert(
            ("minimax-cn", "minimax-m2.7"),
            mk(
                Some(0.30),
                Some(1.20),
                None,
                None,
                None,
                "minimax-pricing-2026-04",
            ),
        );

        m
    })
}

/// Resolve which provider/model/base-url is being billed.
///
/// Mirrors Python `resolve_billing_route`.
pub fn resolve_billing_route(
    model_name: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> BillingRoute {
    let mut provider_name = provider.unwrap_or("").trim().to_lowercase();
    let base = base_url.unwrap_or("").trim().to_lowercase();
    let mut model = model_name.trim().to_string();
    let base_url_owned = base_url.unwrap_or("").to_string();

    if provider_name.is_empty() && model.contains('/') {
        if let Some((inferred_provider, bare_model)) = model.split_once('/') {
            if matches!(inferred_provider, "anthropic" | "openai" | "google") {
                provider_name = inferred_provider.to_string();
                model = bare_model.to_string();
            }
        }
    }

    if provider_name == "openai-codex" {
        return BillingRoute {
            provider: "openai-codex".to_string(),
            model,
            base_url: base_url_owned,
            billing_mode: "subscription_included".to_string(),
        };
    }
    if provider_name == "openrouter"
        || base_url_host_matches(base_url.unwrap_or(""), "openrouter.ai")
    {
        return BillingRoute {
            provider: "openrouter".to_string(),
            model,
            base_url: base_url_owned,
            billing_mode: "official_models_api".to_string(),
        };
    }
    if provider_name == "anthropic" {
        return BillingRoute {
            provider: "anthropic".to_string(),
            model: last_path_segment(&model),
            base_url: base_url_owned,
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "openai" {
        return BillingRoute {
            provider: "openai".to_string(),
            model: last_path_segment(&model),
            base_url: base_url_owned,
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "minimax" || provider_name == "minimax-cn" {
        return BillingRoute {
            provider: provider_name,
            model: last_path_segment(&model),
            base_url: base_url_owned,
            billing_mode: "official_docs_snapshot".to_string(),
        };
    }
    if provider_name == "custom"
        || provider_name == "local"
        || (!base.is_empty() && base.contains("localhost"))
    {
        let prov = if provider_name.is_empty() {
            "custom".to_string()
        } else {
            provider_name
        };
        return BillingRoute {
            provider: prov,
            model,
            base_url: base_url_owned,
            billing_mode: "unknown".to_string(),
        };
    }

    let prov = if provider_name.is_empty() {
        "unknown".to_string()
    } else {
        provider_name
    };
    let route_model = if model.is_empty() {
        String::new()
    } else {
        last_path_segment(&model)
    };
    BillingRoute {
        provider: prov,
        model: route_model,
        base_url: base_url_owned,
        billing_mode: "unknown".to_string(),
    }
}

/// Equivalent to Python `s.split("/")[-1]`.
fn last_path_segment(s: &str) -> String {
    s.rsplit('/').next().unwrap_or(s).to_string()
}

fn lookup_official_docs_pricing(route: &BillingRoute) -> Option<PricingEntry> {
    let model_lower = route.model.to_lowercase();
    official_docs_pricing()
        .get(&(route.provider.as_str(), model_lower.as_str()))
        .cloned()
}

fn openrouter_pricing_entry(route: &BillingRoute) -> Option<PricingEntry> {
    let metadata = fetch_model_metadata(false);
    pricing_entry_from_metadata(
        &metadata,
        &route.model,
        "https://openrouter.ai/docs/api/api-reference/models/get-models",
        "openrouter-models-api",
    )
}

/// Build a `PricingEntry` out of a metadata cache's per-token `pricing` figures.
///
/// Mirrors Python `_pricing_entry_from_metadata`: per-token figures are scaled
/// to per-million, `request` is left as-is. Returns `None` when the model isn't
/// in the cache, or when prompt/completion/request are all absent.
fn pricing_entry_from_metadata(
    metadata: &HashMap<String, HashMap<String, Value>>,
    model_id: &str,
    source_url: &str,
    pricing_version: &str,
) -> Option<PricingEntry> {
    let entry = metadata.get(model_id)?;
    let empty = serde_json::Map::new();
    let pricing: HashMap<String, Value> = match entry.get("pricing") {
        Some(Value::Object(m)) => m.clone().into_iter().collect(),
        _ => empty.clone().into_iter().collect(),
    };

    let get = |key: &str| pricing.get(key);
    let prompt = get("prompt").and_then(to_decimal);
    let completion = get("completion").and_then(to_decimal);
    let request = get("request").and_then(to_decimal);
    let cache_read = first_present(&pricing, &["cache_read", "cached_prompt", "input_cache_read"])
        .and_then(to_decimal);
    let cache_write = first_present(
        &pricing,
        &["cache_write", "cache_creation", "input_cache_write"],
    )
    .and_then(to_decimal);

    if prompt.is_none() && completion.is_none() && request.is_none() {
        return None;
    }

    let per_million = |v: Option<f64>| v.map(|x| x * _ONE_MILLION);

    Some(PricingEntry {
        input_cost_per_million: per_million(prompt),
        output_cost_per_million: per_million(completion),
        cache_read_cost_per_million: per_million(cache_read),
        cache_write_cost_per_million: per_million(cache_write),
        request_cost: request,
        source: CostSource::ProviderModelsApi,
        source_url: Some(source_url.to_string()),
        pricing_version: Some(pricing_version.to_string()),
        fetched_at: Some(utc_now()),
    })
}

/// Mirror of Python's `pricing.get(a) or pricing.get(b) or pricing.get(c)`:
/// return the first key whose value is "truthy". A `Value` is falsey when it is
/// `null`, an empty string, `false`, `0`, or an empty array/object.
fn first_present<'a>(pricing: &'a HashMap<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    for key in keys {
        if let Some(v) = pricing.get(*key) {
            if json_truthy(v) {
                return Some(v);
            }
        }
    }
    None
}

fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Look up a pricing entry for a model+route.
///
/// Mirrors Python `get_pricing_entry`. Network lookups (OpenRouter / endpoint
/// `/models`) are performed via the `ag_model_metadata` cached fetchers.
pub fn get_pricing_entry(
    model_name: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Option<PricingEntry> {
    let route = resolve_billing_route(model_name, provider, base_url);
    if route.billing_mode == "subscription_included" {
        return Some(PricingEntry {
            input_cost_per_million: Some(0.0),
            output_cost_per_million: Some(0.0),
            cache_read_cost_per_million: Some(0.0),
            cache_write_cost_per_million: Some(0.0),
            request_cost: None,
            source: CostSource::None,
            source_url: None,
            pricing_version: Some("included-route".to_string()),
            fetched_at: None,
        });
    }
    if route.provider == "openrouter" {
        return openrouter_pricing_entry(&route);
    }
    if !route.base_url.is_empty() {
        let metadata = fetch_endpoint_model_metadata(&route.base_url, api_key.unwrap_or(""), false);
        let source_url = format!("{}/models", route.base_url.trim_end_matches('/'));
        if let Some(entry) = pricing_entry_from_metadata(
            &metadata,
            &route.model,
            &source_url,
            "openai-compatible-models-api",
        ) {
            return Some(entry);
        }
    }
    lookup_official_docs_pricing(&route)
}

/// Normalize a raw provider usage payload into canonical token buckets.
///
/// Mirrors Python `normalize_usage`. Handles Anthropic Messages, Codex
/// Responses, and OpenAI Chat Completions usage shapes.
pub fn normalize_usage(
    response_usage: &UsageFields,
    provider: Option<&str>,
    api_mode: Option<&str>,
) -> CanonicalUsage {
    if !response_usage.present {
        return CanonicalUsage::default();
    }

    let provider_name = provider.unwrap_or("").trim().to_lowercase();
    let mode = api_mode.unwrap_or("").trim().to_lowercase();

    let input_tokens;
    let output_tokens;
    let cache_read_tokens;
    let cache_write_tokens;

    if mode == "anthropic_messages" || provider_name == "anthropic" {
        input_tokens = opt_to_int(response_usage.input_tokens);
        output_tokens = opt_to_int(response_usage.output_tokens);
        cache_read_tokens = opt_to_int(response_usage.cache_read_input_tokens);
        cache_write_tokens = opt_to_int(response_usage.cache_creation_input_tokens);
    } else if mode == "codex_responses" {
        let input_total = opt_to_int(response_usage.input_tokens);
        output_tokens = opt_to_int(response_usage.output_tokens);
        cache_read_tokens = opt_to_int(response_usage.input_details_cached_tokens);
        cache_write_tokens = opt_to_int(response_usage.input_details_cache_creation_tokens);
        input_tokens = (input_total - cache_read_tokens - cache_write_tokens).max(0);
    } else {
        let prompt_total = opt_to_int(response_usage.prompt_tokens);
        output_tokens = opt_to_int(response_usage.completion_tokens);
        // Primary: OpenAI-style prompt_tokens_details. Fallback: Anthropic-style
        // top-level fields surfaced by some OpenAI-compatible proxies.
        let mut cr = opt_to_int(response_usage.prompt_details_cached_tokens);
        if cr == 0 {
            cr = opt_to_int(response_usage.cache_read_input_tokens);
        }
        cache_read_tokens = cr;
        let mut cw = opt_to_int(response_usage.prompt_details_cache_write_tokens);
        if cw == 0 {
            cw = opt_to_int(response_usage.cache_creation_input_tokens);
        }
        cache_write_tokens = cw;
        input_tokens = (prompt_total - cache_read_tokens - cache_write_tokens).max(0);
    }

    let reasoning_tokens = opt_to_int(response_usage.output_details_reasoning_tokens);

    CanonicalUsage {
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        reasoning_tokens,
        request_count: 1,
        raw_usage: None,
    }
}

/// Estimate the USD cost of a usage record for a given model+route.
///
/// Mirrors Python `estimate_usage_cost`.
pub fn estimate_usage_cost(
    model_name: &str,
    usage: &CanonicalUsage,
    provider: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> CostResult {
    let route = resolve_billing_route(model_name, provider, base_url);
    if route.billing_mode == "subscription_included" {
        return CostResult {
            amount_usd: Some(0.0),
            status: CostStatus::Included,
            source: CostSource::None,
            label: "included".to_string(),
            fetched_at: None,
            pricing_version: Some("included-route".to_string()),
            notes: Vec::new(),
        };
    }

    let entry = match get_pricing_entry(model_name, provider, base_url, api_key) {
        Some(e) => e,
        None => {
            return CostResult {
                amount_usd: None,
                status: CostStatus::Unknown,
                source: CostSource::None,
                label: "n/a".to_string(),
                fetched_at: None,
                pricing_version: None,
                notes: Vec::new(),
            };
        }
    };

    let mut notes: Vec<String> = Vec::new();
    let mut amount: f64 = 0.0;

    let unknown = |source: CostSource, notes: Vec<String>| CostResult {
        amount_usd: None,
        status: CostStatus::Unknown,
        source,
        label: "n/a".to_string(),
        fetched_at: None,
        pricing_version: None,
        notes,
    };

    if usage.input_tokens != 0 && entry.input_cost_per_million.is_none() {
        return unknown(entry.source, Vec::new());
    }
    if usage.output_tokens != 0 && entry.output_cost_per_million.is_none() {
        return unknown(entry.source, Vec::new());
    }
    if usage.cache_read_tokens != 0 && entry.cache_read_cost_per_million.is_none() {
        return unknown(
            entry.source,
            vec!["cache-read pricing unavailable for route".to_string()],
        );
    }
    if usage.cache_write_tokens != 0 && entry.cache_write_cost_per_million.is_none() {
        return unknown(
            entry.source,
            vec!["cache-write pricing unavailable for route".to_string()],
        );
    }

    if let Some(rate) = entry.input_cost_per_million {
        amount += usage.input_tokens as f64 * rate / _ONE_MILLION;
    }
    if let Some(rate) = entry.output_cost_per_million {
        amount += usage.output_tokens as f64 * rate / _ONE_MILLION;
    }
    if let Some(rate) = entry.cache_read_cost_per_million {
        amount += usage.cache_read_tokens as f64 * rate / _ONE_MILLION;
    }
    if let Some(rate) = entry.cache_write_cost_per_million {
        amount += usage.cache_write_tokens as f64 * rate / _ONE_MILLION;
    }
    if let Some(rate) = entry.request_cost {
        if usage.request_count != 0 {
            amount += usage.request_count as f64 * rate;
        }
    }

    let mut status = CostStatus::Estimated;
    let mut label = format!("~${:.2}", amount);
    if entry.source == CostSource::None && amount == 0.0 {
        status = CostStatus::Included;
        label = "included".to_string();
    }

    if route.provider == "openrouter" {
        notes.push("OpenRouter cost is estimated from the models API until reconciled.".to_string());
    }

    CostResult {
        amount_usd: Some(amount),
        status,
        source: entry.source,
        label,
        fetched_at: entry.fetched_at,
        pricing_version: entry.pricing_version,
        notes,
    }
}

/// Whether pricing data exists for this model+route.
///
/// Mirrors Python `has_known_pricing`.
pub fn has_known_pricing(
    model_name: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> bool {
    let route = resolve_billing_route(model_name, provider, base_url);
    if route.billing_mode == "subscription_included" {
        return true;
    }
    get_pricing_entry(model_name, provider, base_url, api_key).is_some()
}

/// Format a duration as a compact human string (e.g. `45s`, `12m`, `3h 5m`,
/// `2.1d`). Mirrors Python `format_duration_compact`.
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
        let remaining_min = (minutes as i64) % 60;
        return if remaining_min != 0 {
            format!("{}h {}m", hours as i64, remaining_min)
        } else {
            format!("{}h", hours as i64)
        };
    }
    let days = hours / 24.0;
    format!("{:.1}d", days)
}

/// Format a token count compactly (e.g. `512`, `1.5K`, `2.34M`, `1B`).
/// Mirrors Python `format_token_count_compact`.
pub fn format_token_count_compact(value: i64) -> String {
    let abs_value = value.unsigned_abs();
    if abs_value < 1_000 {
        return value.to_string();
    }

    let sign = if value < 0 { "-" } else { "" };
    let units: [(u64, &str); 3] = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ];
    for (threshold, suffix) in units {
        if abs_value >= threshold {
            let scaled = abs_value as f64 / threshold as f64;
            let mut text = if scaled < 10.0 {
                format!("{:.2}", scaled)
            } else if scaled < 100.0 {
                format!("{:.1}", scaled)
            } else {
                format!("{:.0}", scaled)
            };
            if text.contains('.') {
                text = text.trim_end_matches('0').trim_end_matches('.').to_string();
            }
            return format!("{}{}{}", sign, text, suffix);
        }
    }

    // Unreachable for abs_value >= 1000, but mirror Python's grouped fallback.
    format_with_thousands(value)
}

fn format_with_thousands(value: i64) -> String {
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
        format!("-{}", out)
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn route_infers_provider_from_slash_prefix() {
        let route = resolve_billing_route("anthropic/claude-opus-4-20250514", None, None);
        assert_eq!(route.provider, "anthropic");
        assert_eq!(route.model, "claude-opus-4-20250514");
        assert_eq!(route.billing_mode, "official_docs_snapshot");
    }

    #[test]
    fn route_codex_is_subscription_included() {
        let route = resolve_billing_route("gpt-5", Some("openai-codex"), None);
        assert_eq!(route.billing_mode, "subscription_included");
    }

    #[test]
    fn route_openrouter_by_provider_and_host() {
        let r1 = resolve_billing_route("foo/bar", Some("openrouter"), None);
        assert_eq!(r1.provider, "openrouter");
        assert_eq!(r1.billing_mode, "official_models_api");
        // model is NOT split on '/' for openrouter
        assert_eq!(r1.model, "foo/bar");

        let r2 = resolve_billing_route("x", None, Some("https://openrouter.ai/api/v1"));
        assert_eq!(r2.provider, "openrouter");
    }

    #[test]
    fn route_localhost_is_custom_unknown() {
        let r = resolve_billing_route("mymodel", None, Some("http://localhost:1234/v1"));
        assert_eq!(r.provider, "custom");
        assert_eq!(r.billing_mode, "unknown");
        assert_eq!(r.model, "mymodel");
    }

    #[test]
    fn route_unknown_strips_path_segment() {
        let r = resolve_billing_route("vendor/the-model", Some("someprovider"), None);
        assert_eq!(r.provider, "someprovider");
        assert_eq!(r.model, "the-model");
        assert_eq!(r.billing_mode, "unknown");
    }

    #[test]
    fn official_docs_lookup() {
        let entry = get_pricing_entry("claude-opus-4-20250514", Some("anthropic"), None, None)
            .expect("entry");
        assert_eq!(entry.input_cost_per_million, Some(15.00));
        assert_eq!(entry.output_cost_per_million, Some(75.00));
        assert_eq!(entry.cache_read_cost_per_million, Some(1.50));
        assert_eq!(entry.cache_write_cost_per_million, Some(18.75));
        assert_eq!(entry.source, CostSource::OfficialDocsSnapshot);
    }

    #[test]
    fn subscription_included_pricing_entry() {
        let entry = get_pricing_entry("gpt-5", Some("openai-codex"), None, None).expect("entry");
        assert_eq!(entry.input_cost_per_million, Some(0.0));
        assert_eq!(entry.pricing_version.as_deref(), Some("included-route"));
    }

    #[test]
    fn estimate_cost_basic() {
        let usage = CanonicalUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..CanonicalUsage::default()
        };
        let result = estimate_usage_cost(
            "claude-opus-4-20250514",
            &usage,
            Some("anthropic"),
            None,
            None,
        );
        // 1M input @ $15 + 1M output @ $75 = $90.00
        assert_eq!(result.status, CostStatus::Estimated);
        assert!((result.amount_usd.unwrap() - 90.0).abs() < 1e-9);
        assert_eq!(result.label, "~$90.00");
    }

    #[test]
    fn estimate_cost_included_route() {
        let usage = CanonicalUsage::default();
        let result = estimate_usage_cost("gpt-5", &usage, Some("openai-codex"), None, None);
        assert_eq!(result.status, CostStatus::Included);
        assert_eq!(result.label, "included");
        assert_eq!(result.amount_usd, Some(0.0));
    }

    #[test]
    fn estimate_cost_cache_missing_pricing_is_unknown() {
        // DeepSeek has no cache pricing in the snapshot.
        let usage = CanonicalUsage {
            cache_read_tokens: 100,
            ..CanonicalUsage::default()
        };
        let result =
            estimate_usage_cost("deepseek-chat", &usage, Some("deepseek"), None, None);
        assert_eq!(result.status, CostStatus::Unknown);
        assert_eq!(result.label, "n/a");
        assert_eq!(result.amount_usd, None);
        assert_eq!(result.notes, vec!["cache-read pricing unavailable for route".to_string()]);
    }

    #[test]
    fn estimate_cost_unknown_model() {
        let usage = CanonicalUsage::default();
        let result =
            estimate_usage_cost("totally-unknown", &usage, Some("whoknows"), None, None);
        assert_eq!(result.status, CostStatus::Unknown);
        assert_eq!(result.label, "n/a");
        assert_eq!(result.amount_usd, None);
    }

    #[test]
    fn has_known_pricing_checks() {
        assert!(has_known_pricing(
            "claude-opus-4-20250514",
            Some("anthropic"),
            None,
            None
        ));
        assert!(has_known_pricing("gpt-5", Some("openai-codex"), None, None));
        assert!(!has_known_pricing(
            "definitely-not-a-model",
            Some("anthropic"),
            None,
            None
        ));
    }

    #[test]
    fn normalize_anthropic_usage() {
        let usage = UsageFields::from_value(&json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        }));
        let canon = normalize_usage(&usage, Some("anthropic"), None);
        assert_eq!(canon.input_tokens, 100);
        assert_eq!(canon.output_tokens, 50);
        assert_eq!(canon.cache_read_tokens, 20);
        assert_eq!(canon.cache_write_tokens, 10);
        assert_eq!(canon.prompt_tokens(), 130);
        assert_eq!(canon.total_tokens(), 180);
    }

    #[test]
    fn normalize_openai_usage_subtracts_cache() {
        let usage = UsageFields::from_value(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 30},
            "output_tokens_details": {"reasoning_tokens": 5},
        }));
        let canon = normalize_usage(&usage, Some("openai"), None);
        assert_eq!(canon.input_tokens, 70);
        assert_eq!(canon.output_tokens, 40);
        assert_eq!(canon.cache_read_tokens, 30);
        assert_eq!(canon.reasoning_tokens, 5);
    }

    #[test]
    fn normalize_openai_falls_back_to_anthropic_fields() {
        // Proxy surfaces Anthropic-style top-level cache fields under OpenAI mode.
        let usage = UsageFields::from_value(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 15,
        }));
        let canon = normalize_usage(&usage, None, None);
        assert_eq!(canon.cache_read_tokens, 25);
        assert_eq!(canon.cache_write_tokens, 15);
        assert_eq!(canon.input_tokens, 60);
    }

    #[test]
    fn normalize_codex_responses() {
        let usage = UsageFields::from_value(&json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 40, "cache_creation_tokens": 10},
        }));
        let canon = normalize_usage(&usage, None, Some("codex_responses"));
        assert_eq!(canon.cache_read_tokens, 40);
        assert_eq!(canon.cache_write_tokens, 10);
        assert_eq!(canon.input_tokens, 50);
    }

    #[test]
    fn normalize_empty_usage() {
        let canon = normalize_usage(&UsageFields::default(), None, None);
        assert_eq!(canon, CanonicalUsage::default());
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(format_duration_compact(45.0), "45s");
        assert_eq!(format_duration_compact(120.0), "2m");
        assert_eq!(format_duration_compact(3661.0), "1h 1m");
        assert_eq!(format_duration_compact(7200.0), "2h");
        assert_eq!(format_duration_compact(2.0 * 86400.0 + 3600.0 * 3.0), "2.1d");
    }

    #[test]
    fn token_count_formatting() {
        assert_eq!(format_token_count_compact(512), "512");
        assert_eq!(format_token_count_compact(1500), "1.5K");
        assert_eq!(format_token_count_compact(2_340_000), "2.34M");
        assert_eq!(format_token_count_compact(1_000_000_000), "1B");
        assert_eq!(format_token_count_compact(-1500), "-1.5K");
        assert_eq!(format_token_count_compact(0), "0");
    }

    #[test]
    fn pricing_entry_from_metadata_scales_to_per_million() {
        let mut entry: HashMap<String, Value> = HashMap::new();
        entry.insert(
            "pricing".to_string(),
            json!({"prompt": "0.000001", "completion": "0.000002", "request": "0.01"}),
        );
        let mut metadata: HashMap<String, HashMap<String, Value>> = HashMap::new();
        metadata.insert("my-model".to_string(), entry);

        let pe = pricing_entry_from_metadata(&metadata, "my-model", "u", "v").expect("entry");
        assert!((pe.input_cost_per_million.unwrap() - 1.0).abs() < 1e-9);
        assert!((pe.output_cost_per_million.unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(pe.request_cost, Some(0.01));
        assert_eq!(pe.source, CostSource::ProviderModelsApi);
    }
}
