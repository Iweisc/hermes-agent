//! Shared auxiliary client router for side tasks (native Rust port of
//! `agent/auxiliary_client.py`).
//!
//! The Python original is a large provider-resolution router that wires
//! together the `openai` SDK, the Anthropic adapter, the credential pool, the
//! runtime-provider resolver, and per-task config. Those subsystems live in
//! other (still-Python) Hermes modules, so this port focuses on the portable,
//! deterministic logic that does not depend on them:
//!
//!   * provider alias normalisation (`normalize_aux_provider`)
//!   * fixed-temperature / compression-threshold model contracts
//!   * per-provider default auxiliary model lookup
//!   * OpenRouter / AI-gateway attribution header construction
//!   * Codex Cloudflare header construction (JWT account-id extraction)
//!   * Anthropic-style → OpenAI base URL rewriting
//!   * Anthropic-Messages endpoint detection
//!   * error classification (payment / rate-limit / connection / auth /
//!     unsupported-parameter)
//!   * chat.completions ↔ Responses-API content conversion
//!   * OpenAI image_url → Anthropic image block conversion
//!   * per-task provider/model resolution shape
//!   * request-kwargs construction for `chat.completions.create`
//!
//! The actual HTTP request build/parse path is provided via
//! [`ChatCompletionsClient`], which uses `reqwest::blocking` and keeps the
//! external OpenAI-compatible wire shape exact.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use serde_json::{Map, Value};

// ── Constants ───────────────────────────────────────────────────────────────

/// OpenRouter base URL (mirrors `hermes_constants.OPENROUTER_BASE_URL`).
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub const OPENROUTER_MODEL: &str = "google/gemini-3-flash-preview";
pub const NOUS_MODEL: &str = "google/gemini-3-flash-preview";
pub const NOUS_DEFAULT_BASE_URL: &str = "https://inference-api.nousresearch.com/v1";
pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const CODEX_AUX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

pub const DEFAULT_AUX_TIMEOUT_SECS: f64 = 30.0;

/// Truthy values for boolean env-var parsing.
const TRUTHY_ENV_VALUES: &[&str] = &["1", "true", "yes", "on"];

// ── Temperature sentinel ─────────────────────────────────────────────────────

/// Result of [`fixed_temperature_for_model`].
///
/// Mirrors the Python sentinel system:
///   * `Omit`  → caller must strip the `temperature` key entirely so the
///     provider chooses its own default (Kimi / Moonshot).
///   * `Fixed(v)` → caller must use this exact value.
///   * `None` → no override; caller uses its own default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureDirective {
    Omit,
    Fixed(f64),
    None,
}

// ── Provider aliasing ────────────────────────────────────────────────────────

/// Provider alias table (mirrors `_PROVIDER_ALIASES`).
pub fn provider_aliases() -> BTreeMap<&'static str, &'static str> {
    let mut m = BTreeMap::new();
    let pairs: &[(&str, &str)] = &[
        ("google", "gemini"),
        ("google-gemini", "gemini"),
        ("google-ai-studio", "gemini"),
        ("x-ai", "xai"),
        ("x.ai", "xai"),
        ("grok", "xai"),
        ("glm", "zai"),
        ("z-ai", "zai"),
        ("z.ai", "zai"),
        ("zhipu", "zai"),
        ("kimi", "kimi-coding"),
        ("moonshot", "kimi-coding"),
        ("kimi-cn", "kimi-coding-cn"),
        ("moonshot-cn", "kimi-coding-cn"),
        ("gmi-cloud", "gmi"),
        ("gmicloud", "gmi"),
        ("minimax-china", "minimax-cn"),
        ("minimax_cn", "minimax-cn"),
        ("claude", "anthropic"),
        ("claude-code", "anthropic"),
        ("github", "copilot"),
        ("github-copilot", "copilot"),
        ("github-model", "copilot"),
        ("github-models", "copilot"),
        ("github-copilot-acp", "copilot-acp"),
        ("copilot-acp-agent", "copilot-acp"),
        ("tencent", "tencent-tokenhub"),
        ("tokenhub", "tencent-tokenhub"),
        ("tencent-cloud", "tencent-tokenhub"),
        ("tencentmaas", "tencent-tokenhub"),
    ];
    for (k, v) in pairs {
        m.insert(*k, *v);
    }
    m
}

/// Normalise an auxiliary provider name (mirrors `_normalize_aux_provider`).
///
/// `main_provider` is the user's configured main provider, used to resolve
/// the special `"main"` value (the Python original reads config; here it's
/// injected). Pass `None` to behave as if no main provider is configured.
pub fn normalize_aux_provider(provider: Option<&str>, main_provider: Option<&str>) -> String {
    let mut normalized = provider.unwrap_or("auto").trim().to_lowercase();
    if normalized.is_empty() {
        normalized = "auto".to_string();
    }
    if let Some(suffix) = normalized.strip_prefix("custom:") {
        let suffix = suffix.trim();
        if suffix.is_empty() {
            return "custom".to_string();
        }
        normalized = suffix.to_string();
    }
    if normalized == "codex" {
        return "openai-codex".to_string();
    }
    if normalized == "main" {
        let main = main_provider.unwrap_or("").trim().to_lowercase();
        if !main.is_empty() && main != "auto" && main != "main" {
            normalized = main;
        } else {
            return "custom".to_string();
        }
    }
    provider_aliases()
        .get(normalized.as_str())
        .map(|s| s.to_string())
        .unwrap_or(normalized)
}

// ── Model-specific contracts ─────────────────────────────────────────────────

fn bare_model(model: Option<&str>) -> String {
    model
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_string()
}

/// True for any Kimi / Moonshot model that manages temperature server-side.
pub fn is_kimi_model(model: Option<&str>) -> bool {
    let bare = bare_model(model);
    bare.starts_with("kimi-") || bare == "kimi"
}

/// True for Arcee Trinity Large Thinking (direct or via OpenRouter).
pub fn is_arcee_trinity_thinking(model: Option<&str>) -> bool {
    bare_model(model) == "trinity-large-thinking"
}

/// Return a temperature directive for models with strict contracts
/// (mirrors `_fixed_temperature_for_model`).
pub fn fixed_temperature_for_model(
    model: Option<&str>,
    _base_url: Option<&str>,
) -> TemperatureDirective {
    if is_kimi_model(model) {
        return TemperatureDirective::Omit;
    }
    if is_arcee_trinity_thinking(model) {
        return TemperatureDirective::Fixed(0.5);
    }
    TemperatureDirective::None
}

/// Return a context-compression threshold override for specific models
/// (mirrors `_compression_threshold_for_model`).
pub fn compression_threshold_for_model(model: Option<&str>) -> Option<f64> {
    if is_arcee_trinity_thinking(model) {
        return Some(0.75);
    }
    None
}

// ── Default auxiliary models per provider ────────────────────────────────────

/// Fallback table of cheap auxiliary models per provider
/// (mirrors `_API_KEY_PROVIDER_AUX_MODELS_FALLBACK`).
pub fn api_key_provider_aux_models_fallback() -> BTreeMap<&'static str, &'static str> {
    let mut m = BTreeMap::new();
    let pairs: &[(&str, &str)] = &[
        ("gemini", "gemini-3-flash-preview"),
        ("zai", "glm-4.5-flash"),
        ("kimi-coding", "kimi-k2-turbo-preview"),
        ("stepfun", "step-3.5-flash"),
        ("kimi-coding-cn", "kimi-k2-turbo-preview"),
        ("gmi", "google/gemini-3.1-flash-lite-preview"),
        ("minimax", "MiniMax-M2.7"),
        ("minimax-oauth", "MiniMax-M2.7-highspeed"),
        ("minimax-cn", "MiniMax-M2.7"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("ai-gateway", "google/gemini-3-flash"),
        ("opencode-zen", "gemini-3-flash"),
        ("opencode-go", "glm-5"),
        ("kilocode", "google/gemini-3-flash-preview"),
        ("ollama-cloud", "nemotron-3-nano:30b"),
        ("tencent-tokenhub", "hy3-preview"),
    ];
    for (k, v) in pairs {
        m.insert(*k, *v);
    }
    m
}

/// Return the cheap auxiliary model for a provider, or `""` when unknown
/// (mirrors `_get_aux_model_for_provider` minus the ProviderProfile lookup,
/// which lives in a separate Python module).
pub fn get_aux_model_for_provider(provider_id: &str) -> String {
    api_key_provider_aux_models_fallback()
        .get(provider_id)
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// Vision-specific model overrides for direct providers
/// (mirrors `_PROVIDER_VISION_MODELS`).
pub fn provider_vision_models() -> BTreeMap<&'static str, &'static str> {
    let mut m = BTreeMap::new();
    m.insert("xiaomi", "mimo-v2.5");
    m.insert("zai", "glm-5v-turbo");
    m
}

/// Providers whose endpoint does not accept image input
/// (mirrors `_PROVIDERS_WITHOUT_VISION`).
pub fn providers_without_vision() -> &'static [&'static str] {
    &["kimi-coding", "kimi-coding-cn"]
}

/// Anthropic-compatible providers (mirrors `_ANTHROPIC_COMPAT_PROVIDERS`).
pub fn anthropic_compat_providers() -> &'static [&'static str] {
    &["minimax", "minimax-oauth", "minimax-cn"]
}

/// Vision auto-detection provider order (mirrors `_VISION_AUTO_PROVIDER_ORDER`).
pub fn vision_auto_provider_order() -> &'static [&'static str] {
    &["openrouter", "nous"]
}

// ── URL helpers ──────────────────────────────────────────────────────────────

/// Return the lowercase hostname for a URL, or `""` when it can't be parsed.
/// Mirrors `utils.base_url_hostname`.
pub fn base_url_hostname(base_url: &str) -> String {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Ok(parsed) = url::Url::parse(trimmed) {
        return parsed.host_str().unwrap_or("").to_lowercase();
    }
    // No scheme — try prefixing https:// so a bare host parses.
    if let Ok(parsed) = url::Url::parse(&format!("https://{trimmed}")) {
        return parsed.host_str().unwrap_or("").to_lowercase();
    }
    String::new()
}

/// True when the URL's hostname equals `host` (case-insensitive).
/// Mirrors `utils.base_url_host_matches`.
pub fn base_url_host_matches(base_url: &str, host: &str) -> bool {
    base_url_hostname(base_url) == host.to_lowercase()
}

/// Extract query params from a URL.
/// Returns `(clean_url, Some(params))` when the URL had a query string,
/// otherwise `(url, None)`. Mirrors `_extract_url_query_params`.
pub fn extract_url_query_params(url_str: &str) -> (String, Option<BTreeMap<String, String>>) {
    if let Ok(parsed) = url::Url::parse(url_str) {
        if parsed.query().is_some() && !parsed.query().unwrap_or("").is_empty() {
            let mut params: BTreeMap<String, String> = BTreeMap::new();
            for (k, v) in parsed.query_pairs() {
                // parse_qs keeps the first value per key.
                params.entry(k.to_string()).or_insert_with(|| v.to_string());
            }
            let mut clean = parsed.clone();
            clean.set_query(None);
            // urlunparse drops trailing '?'; Url::to_string already omits it.
            let clean_str = clean.as_str().trim_end_matches('?').to_string();
            return (clean_str, Some(params));
        }
    }
    (url_str.to_string(), None)
}

/// Normalise an Anthropic-style base URL to OpenAI-compatible format.
/// Mirrors `_to_openai_base_url`.
pub fn to_openai_base_url(base_url: &str) -> String {
    let url = base_url.trim().trim_end_matches('/').to_string();
    if url.ends_with("/anthropic") {
        return format!("{}/v1", &url[..url.len() - "/anthropic".len()]);
    }
    if url.contains("api.kimi.com") && url.ends_with("/coding") {
        return format!("{url}/v1");
    }
    url
}

/// True if the endpoint at `base_url` speaks the Anthropic Messages protocol.
/// Mirrors `_endpoint_speaks_anthropic_messages`.
pub fn endpoint_speaks_anthropic_messages(base_url: &str) -> bool {
    let normalized = base_url.trim().to_lowercase();
    let normalized = normalized.trim_end_matches('/');
    if normalized.is_empty() {
        return false;
    }
    if normalized.ends_with("/anthropic") {
        return true;
    }
    let hostname = base_url_hostname(normalized);
    if hostname == "api.anthropic.com" {
        return true;
    }
    if hostname == "api.kimi.com" && normalized.contains("/coding") {
        return true;
    }
    false
}

/// Detect if an endpoint expects Anthropic-format content blocks.
/// Mirrors `_is_anthropic_compat_endpoint`.
pub fn is_anthropic_compat_endpoint(provider: &str, base_url: &str) -> bool {
    if anthropic_compat_providers().contains(&provider) {
        return true;
    }
    base_url.to_lowercase().contains("/anthropic")
}

// ── Codex Cloudflare headers ─────────────────────────────────────────────────

/// Headers required to avoid Cloudflare 403s on
/// `chatgpt.com/backend-api/codex`. Mirrors `_codex_cloudflare_headers`.
///
/// Extracts the `ChatGPT-Account-ID` from the OAuth JWT's
/// `chatgpt_account_id` claim; tolerates malformed tokens by dropping that
/// header rather than erroring.
pub fn codex_cloudflare_headers(access_token: &str) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();
    headers.insert(
        "User-Agent".to_string(),
        "codex_cli_rs/0.0.0 (Hermes Agent)".to_string(),
    );
    headers.insert("originator".to_string(), "codex_cli_rs".to_string());

    let token = access_token.trim();
    if token.is_empty() {
        return headers;
    }
    if let Some(acct_id) = jwt_chatgpt_account_id(token) {
        if !acct_id.is_empty() {
            headers.insert("ChatGPT-Account-ID".to_string(), acct_id);
        }
    }
    headers
}

/// Decode a JWT payload (base64url, unpadded) into JSON.
fn jwt_payload_claims(token: &str) -> Option<Value> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let mut payload = parts[1].to_string();
    let pad = (4 - payload.len() % 4) % 4;
    payload.push_str(&"=".repeat(pad));
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(payload.as_bytes())
        .ok()?;
    serde_json::from_slice(&decoded).ok()
}

fn jwt_chatgpt_account_id(token: &str) -> Option<String> {
    let claims = jwt_payload_claims(token)?;
    claims
        .get("https://api.openai.com/auth")
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// True when a Codex JWT access token is expired (`exp` claim in the past).
/// Non-JWT tokens (or tokens without `exp`) are treated as not-expired,
/// matching the Python `_read_codex_access_token` behaviour.
pub fn codex_token_expired(token: &str, now_unix: i64) -> bool {
    match jwt_payload_claims(token.trim()) {
        Some(claims) => match claims.get("exp").and_then(|v| v.as_i64()) {
            Some(exp) if exp != 0 => now_unix > exp,
            _ => false,
        },
        None => false,
    }
}

// ── OpenRouter / AI-gateway headers ──────────────────────────────────────────

/// Base OpenRouter attribution headers (mirrors `_OR_HEADERS_BASE`).
pub fn openrouter_headers_base() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        "HTTP-Referer".to_string(),
        "https://hermes-agent.nousresearch.com".to_string(),
    );
    m.insert("X-Title".to_string(), "Hermes Agent".to_string());
    m.insert(
        "X-OpenRouter-Categories".to_string(),
        "productivity,cli-agent".to_string(),
    );
    m
}

/// Configuration for OpenRouter response-cache headers, sourced from the
/// `openrouter` config section and/or env overrides. Mirrors the inputs to
/// `build_or_headers`.
#[derive(Debug, Clone, Default)]
pub struct OpenRouterConfig {
    pub response_cache: bool,
    /// Default TTL when caching is enabled (Python default is 300).
    pub response_cache_ttl: Option<i64>,
    /// Value of `HERMES_OPENROUTER_CACHE` env var (already read), if set.
    pub env_cache: Option<String>,
    /// Value of `HERMES_OPENROUTER_CACHE_TTL` env var (already read), if set.
    pub env_ttl: Option<String>,
}

/// Build OpenRouter headers, optionally including response-cache headers.
/// Mirrors `build_or_headers`. Precedence: env var > config > default.
pub fn build_or_headers(cfg: &OpenRouterConfig) -> BTreeMap<String, String> {
    let mut headers = openrouter_headers_base();

    let cache_enabled = match &cfg.env_cache {
        Some(v) if !v.trim().is_empty() => {
            TRUTHY_ENV_VALUES.contains(&v.trim().to_lowercase().as_str())
        }
        _ => cfg.response_cache,
    };

    if !cache_enabled {
        return headers;
    }

    headers.insert("X-OpenRouter-Cache".to_string(), "true".to_string());

    match &cfg.env_ttl {
        Some(v) if !v.trim().is_empty() => {
            let trimmed = v.trim();
            if trimmed.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(ttl) = trimmed.parse::<i64>() {
                    if (1..=86400).contains(&ttl) {
                        headers.insert("X-OpenRouter-Cache-TTL".to_string(), ttl.to_string());
                    }
                }
            }
        }
        _ => {
            let ttl = cfg.response_cache_ttl.unwrap_or(300);
            if (1..=86400).contains(&ttl) {
                headers.insert("X-OpenRouter-Cache-TTL".to_string(), ttl.to_string());
            }
        }
    }

    headers
}

/// Vercel AI Gateway attribution headers (mirrors `_AI_GATEWAY_HEADERS`).
pub fn ai_gateway_headers(hermes_version: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        "HTTP-Referer".to_string(),
        "https://hermes-agent.nousresearch.com".to_string(),
    );
    m.insert("X-Title".to_string(), "Hermes Agent".to_string());
    m.insert(
        "User-Agent".to_string(),
        format!("HermesAgent/{hermes_version}"),
    );
    m
}

/// Nous Portal extra_body for product attribution (mirrors `NOUS_EXTRA_BODY`).
pub fn nous_extra_body() -> Value {
    serde_json::json!({ "tags": ["product=hermes-agent"] })
}

// ── Error classification ─────────────────────────────────────────────────────

/// Detect payment / credit / quota exhaustion errors. Mirrors
/// `_is_payment_error`. `status` is the HTTP status (if any).
pub fn is_payment_error(status: Option<u16>, message: &str) -> bool {
    if status == Some(402) {
        return true;
    }
    let err_lower = message.to_lowercase();
    if matches!(status, Some(402) | Some(429) | None) {
        const KW: &[&str] = &[
            "credits",
            "insufficient funds",
            "can only afford",
            "billing",
            "payment required",
        ];
        if KW.iter().any(|k| err_lower.contains(k)) {
            return true;
        }
    }
    false
}

/// Detect rate-limit errors that warrant provider fallback. Mirrors
/// `_is_rate_limit_error`. `error_type_name` is the exception/class name
/// (Python checks for `RateLimitError`).
pub fn is_rate_limit_error(status: Option<u16>, message: &str, error_type_name: &str) -> bool {
    let err_lower = message.to_lowercase();
    if error_type_name == "RateLimitError" {
        return true;
    }
    if status == Some(429) {
        const RATE_KW: &[&str] = &[
            "rate limit",
            "rate_limit",
            "too many requests",
            "try again",
            "retry after",
            "resets in",
        ];
        if RATE_KW.iter().any(|k| err_lower.contains(k)) {
            return true;
        }
        const BILLING_KW: &[&str] = &[
            "credits",
            "insufficient funds",
            "billing",
            "payment required",
            "can only afford",
        ];
        if !BILLING_KW.iter().any(|k| err_lower.contains(k)) {
            return true;
        }
    }
    false
}

/// Detect connection/network errors that warrant provider fallback. Mirrors
/// `_is_connection_error`.
pub fn is_connection_error(error_type_name: &str, message: &str) -> bool {
    const TYPE_KW: &[&str] = &["Connection", "Timeout", "DNS", "SSL"];
    if TYPE_KW.iter().any(|k| error_type_name.contains(k)) {
        return true;
    }
    let err_lower = message.to_lowercase();
    const MSG_KW: &[&str] = &[
        "connection refused",
        "name or service not known",
        "no route to host",
        "network is unreachable",
        "timed out",
        "connection reset",
    ];
    MSG_KW.iter().any(|k| err_lower.contains(k))
}

/// Detect auth failures (mirrors `_is_auth_error`).
pub fn is_auth_error(status: Option<u16>, message: &str, error_type_name: &str) -> bool {
    if status == Some(401) {
        return true;
    }
    let err_lower = message.to_lowercase();
    err_lower.contains("error code: 401")
        || error_type_name.to_lowercase().contains("authenticationerror")
}

/// Detect provider 400s for an unsupported request parameter. Mirrors
/// `_is_unsupported_parameter_error`.
pub fn is_unsupported_parameter_error(message: &str, param: &str) -> bool {
    let param_lower = param.trim().to_lowercase();
    if param_lower.is_empty() {
        return false;
    }
    let err_lower = message.to_lowercase();
    if !err_lower.contains(&param_lower) {
        return false;
    }
    const MARKERS: &[&str] = &[
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized request argument",
        "unrecognized parameter",
        "invalid parameter",
    ];
    MARKERS.iter().any(|m| err_lower.contains(m))
}

/// Back-compat wrapper: detect API errors where the model rejects
/// `temperature`. Mirrors `_is_unsupported_temperature_error`.
pub fn is_unsupported_temperature_error(message: &str) -> bool {
    is_unsupported_parameter_error(message, "temperature")
}

// ── Codex Responses ↔ chat.completions content conversion ────────────────────

/// Convert chat.completions content to Responses API format.
/// Mirrors `_convert_content_for_responses`.
pub fn convert_content_for_responses(content: &Value) -> Value {
    if let Some(s) = content.as_str() {
        return Value::String(s.to_string());
    }
    let arr = match content.as_array() {
        Some(a) => a,
        None => {
            if content.is_null() {
                return Value::String(String::new());
            }
            return Value::String(content.to_string());
        }
    };

    let mut converted: Vec<Value> = Vec::new();
    for part in arr {
        let obj = match part.as_object() {
            Some(o) => o,
            None => continue,
        };
        let ptype = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ptype {
            "text" => {
                let text = obj.get("text").cloned().unwrap_or(Value::String(String::new()));
                converted.push(serde_json::json!({"type": "input_text", "text": text}));
            }
            "image_url" => {
                let image_data = obj.get("image_url");
                let (url_str, detail) = match image_data {
                    Some(Value::Object(m)) => (
                        m.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        m.get("detail").cloned(),
                    ),
                    Some(v) => (
                        v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                        None,
                    ),
                    None => (String::new(), None),
                };
                let mut entry = serde_json::Map::new();
                entry.insert("type".to_string(), Value::String("input_image".to_string()));
                entry.insert("image_url".to_string(), Value::String(url_str));
                if let Some(d) = detail {
                    if !d.is_null() {
                        entry.insert("detail".to_string(), d);
                    }
                }
                converted.push(Value::Object(entry));
            }
            "input_text" | "input_image" => {
                converted.push(part.clone());
            }
            _ => {
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        converted.push(serde_json::json!({"type": "input_text", "text": text}));
                    }
                }
            }
        }
    }

    if converted.is_empty() {
        Value::String(String::new())
    } else {
        Value::Array(converted)
    }
}

/// Convert OpenAI `image_url` content blocks to Anthropic `image` blocks.
/// Mirrors `_convert_openai_images_to_anthropic`. Operates on a list of
/// message objects.
pub fn convert_openai_images_to_anthropic(messages: &[Value]) -> Vec<Value> {
    let mut converted: Vec<Value> = Vec::new();
    for msg in messages {
        let content = msg.get("content");
        let content_arr = match content.and_then(|c| c.as_array()) {
            Some(a) => a,
            None => {
                converted.push(msg.clone());
                continue;
            }
        };
        let mut new_content: Vec<Value> = Vec::new();
        let mut changed = false;
        for block in content_arr {
            let btype = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if btype == "image_url" {
                let image_url_val = block
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(rest) = image_url_val.strip_prefix("data:") {
                    // data:<media_type>;base64,<data>
                    let (header, b64data) = match image_url_val.split_once(',') {
                        Some((h, d)) => (h.to_string(), d.to_string()),
                        None => (format!("data:{rest}"), String::new()),
                    };
                    let mut media_type = "image/png".to_string();
                    if header.contains(':') && header.contains(';') {
                        if let Some(after_colon) = header.splitn(2, ':').nth(1) {
                            media_type = after_colon
                                .splitn(2, ';')
                                .next()
                                .unwrap_or("image/png")
                                .to_string();
                        }
                    }
                    new_content.push(serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        }
                    }));
                } else {
                    new_content.push(serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        }
                    }));
                }
                changed = true;
            } else {
                new_content.push(block.clone());
            }
        }
        if changed {
            let mut new_msg = msg.as_object().cloned().unwrap_or_default();
            new_msg.insert("content".to_string(), Value::Array(new_content));
            converted.push(Value::Object(new_msg));
        } else {
            converted.push(msg.clone());
        }
    }
    converted
}

// ── Per-task provider/model resolution ───────────────────────────────────────

/// A resolved task routing decision (mirrors the tuple returned by
/// `_resolve_task_provider_model`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedTask {
    pub provider: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_mode: Option<String>,
}

/// The `auxiliary.<task>` config section (already loaded from config.yaml).
#[derive(Debug, Clone, Default)]
pub struct TaskConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_mode: Option<String>,
    pub timeout: Option<f64>,
    pub extra_body: Option<Value>,
}

fn clean_opt(s: Option<&str>) -> Option<String> {
    match s {
        Some(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Determine provider + model for a call. Mirrors
/// `_resolve_task_provider_model`. Explicit args win, then task config, then
/// `"auto"`.
pub fn resolve_task_provider_model(
    task_config: Option<&TaskConfig>,
    provider: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> ResolvedTask {
    let cfg_provider = task_config.and_then(|c| clean_opt(c.provider.as_deref()));
    let cfg_model = task_config.and_then(|c| clean_opt(c.model.as_deref()));
    let cfg_base_url = task_config.and_then(|c| clean_opt(c.base_url.as_deref()));
    let cfg_api_key = task_config.and_then(|c| clean_opt(c.api_key.as_deref()));
    let cfg_api_mode = task_config.and_then(|c| clean_opt(c.api_mode.as_deref()));

    let resolved_model = clean_opt(model).or(cfg_model);
    let resolved_api_mode = cfg_api_mode;

    if let Some(b) = clean_opt(base_url) {
        return ResolvedTask {
            provider: "custom".to_string(),
            model: resolved_model,
            base_url: Some(b),
            api_key: clean_opt(api_key),
            api_mode: resolved_api_mode,
        };
    }
    if let Some(p) = clean_opt(provider) {
        return ResolvedTask {
            provider: p,
            model: resolved_model,
            base_url: None,
            api_key: clean_opt(api_key),
            api_mode: resolved_api_mode,
        };
    }

    if task_config.is_some() {
        if let (Some(b), Some(k)) = (&cfg_base_url, &cfg_api_key) {
            return ResolvedTask {
                provider: "custom".to_string(),
                model: resolved_model,
                base_url: Some(b.clone()),
                api_key: Some(k.clone()),
                api_mode: resolved_api_mode,
            };
        }
        if let (Some(b), Some(p)) = (&cfg_base_url, &cfg_provider) {
            if p != "auto" {
                return ResolvedTask {
                    provider: p.clone(),
                    model: resolved_model,
                    base_url: Some(b.clone()),
                    api_key: None,
                    api_mode: resolved_api_mode,
                };
            }
        }
        if let Some(p) = &cfg_provider {
            if p != "auto" {
                return ResolvedTask {
                    provider: p.clone(),
                    model: resolved_model,
                    base_url: None,
                    api_key: None,
                    api_mode: resolved_api_mode,
                };
            }
        }
    }

    ResolvedTask {
        provider: "auto".to_string(),
        model: resolved_model,
        base_url: None,
        api_key: None,
        api_mode: resolved_api_mode,
    }
}

/// Read the task timeout (mirrors `_get_task_timeout`).
pub fn get_task_timeout(task_config: Option<&TaskConfig>, default: f64) -> f64 {
    task_config.and_then(|c| c.timeout).unwrap_or(default)
}

/// Read the task extra_body dict (mirrors `_get_task_extra_body`). Returns an
/// empty object when missing or not a dict.
pub fn get_task_extra_body(task_config: Option<&TaskConfig>) -> Value {
    match task_config.and_then(|c| c.extra_body.as_ref()) {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Map::new()),
    }
}

// ── Request kwargs construction ──────────────────────────────────────────────

/// Build the request body for `chat.completions.create`. Mirrors
/// `_build_call_kwargs` (sans the `timeout` kwarg, which the Rust HTTP layer
/// applies separately). `nous_active` mirrors the module-level
/// `auxiliary_is_nous` flag.
pub fn build_call_kwargs(
    provider: &str,
    model: &str,
    messages: Vec<Value>,
    temperature: Option<f64>,
    max_tokens: Option<i64>,
    tools: Option<Vec<Value>>,
    extra_body: Option<&Value>,
    base_url: Option<&str>,
    forbids_sampling_params: bool,
    nous_active: bool,
) -> Value {
    let mut kwargs = Map::new();
    kwargs.insert("model".to_string(), Value::String(model.to_string()));
    kwargs.insert("messages".to_string(), Value::Array(messages));

    let mut temp = temperature;
    match fixed_temperature_for_model(Some(model), base_url) {
        TemperatureDirective::Omit => temp = None,
        TemperatureDirective::Fixed(v) => temp = Some(v),
        TemperatureDirective::None => {}
    }
    // Opus 4.7+ rejects non-default sampling params.
    if forbids_sampling_params {
        temp = None;
    }
    if let Some(t) = temp {
        kwargs.insert(
            "temperature".to_string(),
            serde_json::Number::from_f64(t)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
    }

    if let Some(mt) = max_tokens {
        if provider == "custom" {
            let custom_base = base_url.unwrap_or("");
            if base_url_hostname(custom_base) == "api.openai.com" {
                kwargs.insert("max_completion_tokens".to_string(), Value::from(mt));
            } else {
                kwargs.insert("max_tokens".to_string(), Value::from(mt));
            }
        } else {
            kwargs.insert("max_tokens".to_string(), Value::from(mt));
        }
    }

    if let Some(tools) = tools {
        if !tools.is_empty() {
            // Defensive dedup by tool function name.
            let mut seen: Vec<String> = Vec::new();
            let mut deduped: Vec<Value> = Vec::new();
            for t in tools {
                let tname = t
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !tname.is_empty() && seen.contains(&tname) {
                    continue;
                }
                if !tname.is_empty() {
                    seen.push(tname);
                }
                deduped.push(t);
            }
            kwargs.insert("tools".to_string(), Value::Array(deduped));
        }
    }

    // Provider-specific extra_body.
    let mut merged: Map<String, Value> = match extra_body {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    if provider == "nous" || nous_active {
        let entry = merged
            .entry("tags".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(arr) = entry.as_array_mut() {
            arr.push(Value::String("product=hermes-agent".to_string()));
        }
    }
    if !merged.is_empty() {
        kwargs.insert("extra_body".to_string(), Value::Object(merged));
    }

    Value::Object(kwargs)
}

/// Return the `max_tokens` vs `max_completion_tokens` kwarg for the auxiliary
/// client's provider (mirrors `auxiliary_max_tokens_param`). `use_completion`
/// is the precomputed branch (direct OpenAI custom endpoint).
pub fn auxiliary_max_tokens_param(value: i64, use_completion: bool) -> Value {
    let key = if use_completion {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    serde_json::json!({ key: value })
}

// ── Response extraction ──────────────────────────────────────────────────────

/// Extract content from an LLM response JSON, falling back to reasoning
/// fields. Mirrors `extract_content_or_reasoning`.
pub fn extract_content_or_reasoning(response: &Value) -> String {
    let msg = match response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
    {
        Some(m) => m,
        None => return String::new(),
    };

    let content = msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !content.is_empty() {
        let cleaned = strip_think_blocks(&content);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }

    let mut reasoning_parts: Vec<String> = Vec::new();
    for field in ["reasoning", "reasoning_content"] {
        if let Some(val) = msg.get(field).and_then(|v| v.as_str()) {
            let t = val.trim();
            if !t.is_empty() && !reasoning_parts.iter().any(|p| p == val) {
                reasoning_parts.push(t.to_string());
            }
        }
    }

    if let Some(details) = msg.get("reasoning_details").and_then(|v| v.as_array()) {
        for detail in details {
            if let Some(obj) = detail.as_object() {
                let summary = obj
                    .get("summary")
                    .or_else(|| obj.get("content"))
                    .or_else(|| obj.get("text"));
                if let Some(s) = summary {
                    let text = match s.as_str() {
                        Some(st) => st.to_string(),
                        None => s.to_string(),
                    };
                    if !reasoning_parts.contains(&text) {
                        reasoning_parts.push(text);
                    }
                }
            }
        }
    }

    if !reasoning_parts.is_empty() {
        return reasoning_parts.join("\n\n");
    }
    String::new()
}

/// Strip inline `<think>`/`<reasoning>`/etc. blocks (mirrors the regex in
/// `extract_content_or_reasoning`).
fn strip_think_blocks(content: &str) -> String {
    // Case-insensitive, dot-matches-newline, balanced open/close tags.
    let re = regex::RegexBuilder::new(
        r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>.*?</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
    )
    .case_insensitive(true)
    .dot_matches_new_line(true)
    .build();
    match re {
        Ok(re) => re.replace_all(content, "").trim().to_string(),
        Err(_) => content.trim().to_string(),
    }
}

/// Validate that an LLM response has `.choices[0].message`. Mirrors
/// `_validate_llm_response`. Returns `Ok(())` when valid.
pub fn validate_llm_response(response: &Value, task: Option<&str>) -> Result<(), String> {
    let label = task.unwrap_or("call");
    if response.is_null() {
        return Err(format!("Auxiliary {label}: LLM returned None response"));
    }
    let ok = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .map(|first| first.get("message").is_some())
        .unwrap_or(false);
    if !ok {
        let preview: String = response.to_string().chars().take(120).collect();
        return Err(format!(
            "Auxiliary {label}: LLM returned invalid response: {preview:?}. \
             Expected object with .choices[0].message — check provider \
             adapter or custom endpoint compatibility."
        ));
    }
    Ok(())
}

// ── HTTP client (request build + response parse) ─────────────────────────────

/// A minimal OpenAI-compatible chat.completions client using
/// `reqwest::blocking`. Keeps the external wire shape exact: POSTs the request
/// body (as built by [`build_call_kwargs`], minus any `extra_body` nesting,
/// which is flattened into the top-level body to match the SDK) to
/// `{base_url}/chat/completions` with `Authorization: Bearer {api_key}`.
#[derive(Debug, Clone)]
pub struct ChatCompletionsClient {
    pub base_url: String,
    pub api_key: String,
    pub default_headers: BTreeMap<String, String>,
}

impl ChatCompletionsClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            default_headers: BTreeMap::new(),
        }
    }

    pub fn with_headers(mut self, headers: BTreeMap<String, String>) -> Self {
        self.default_headers = headers;
        self
    }

    /// Flatten the `extra_body` object into the top-level request body, as the
    /// OpenAI SDK does when forwarding `extra_body=`.
    pub fn flatten_body(kwargs: &Value) -> Value {
        let obj = match kwargs.as_object() {
            Some(o) => o,
            None => return kwargs.clone(),
        };
        let mut body = Map::new();
        for (k, v) in obj {
            if k == "extra_body" {
                continue;
            }
            body.insert(k.clone(), v.clone());
        }
        if let Some(Value::Object(extra)) = obj.get("extra_body") {
            for (k, v) in extra {
                body.insert(k.clone(), v.clone());
            }
        }
        Value::Object(body)
    }

    /// Execute a chat.completions request. `kwargs` is the value produced by
    /// [`build_call_kwargs`]. `timeout` is applied to the HTTP request.
    /// Returns the parsed JSON response, or an error string with the HTTP
    /// status preserved in the message (so [`is_payment_error`] et al. can
    /// classify it).
    pub fn create(&self, kwargs: &Value, timeout: Duration) -> Result<Value, ChatCallError> {
        let body = Self::flatten_body(kwargs);
        let url = format!("{}/chat/completions", self.base_url);

        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| ChatCallError {
                status: None,
                message: format!("client build error: {e}"),
                type_name: "ConnectionError".to_string(),
            })?;

        let mut req = client
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json");
        for (k, v) in &self.default_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req.json(&body).send().map_err(|e| {
            let type_name = if e.is_timeout() {
                "APITimeoutError"
            } else if e.is_connect() {
                "APIConnectionError"
            } else {
                "ConnectionError"
            };
            ChatCallError {
                status: None,
                message: e.to_string(),
                type_name: type_name.to_string(),
            }
        })?;

        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(ChatCallError {
                status: Some(status.as_u16()),
                message: format!("Error code: {} - {}", status.as_u16(), text),
                type_name: api_error_type_name(status.as_u16()),
            });
        }
        serde_json::from_str::<Value>(&text).map_err(|e| ChatCallError {
            status: Some(status.as_u16()),
            message: format!("invalid JSON response: {e}"),
            type_name: "APIError".to_string(),
        })
    }
}

/// Map an HTTP status code to a class-name string mirroring the OpenAI SDK's
/// exception hierarchy, so the error classifiers behave the same.
fn api_error_type_name(status: u16) -> String {
    match status {
        401 => "AuthenticationError",
        402 => "APIStatusError",
        429 => "RateLimitError",
        s if (500..600).contains(&s) => "InternalServerError",
        _ => "APIStatusError",
    }
    .to_string()
}

/// Error from a chat.completions call, carrying enough metadata to feed the
/// classifiers ([`is_payment_error`], [`is_rate_limit_error`], etc.).
#[derive(Debug, Clone)]
pub struct ChatCallError {
    pub status: Option<u16>,
    pub message: String,
    pub type_name: String,
}

impl std::fmt::Display for ChatCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ChatCallError {}

impl ChatCallError {
    pub fn is_payment(&self) -> bool {
        is_payment_error(self.status, &self.message)
    }
    pub fn is_rate_limit(&self) -> bool {
        is_rate_limit_error(self.status, &self.message, &self.type_name)
    }
    pub fn is_connection(&self) -> bool {
        is_connection_error(&self.type_name, &self.message)
    }
    pub fn is_auth(&self) -> bool {
        is_auth_error(self.status, &self.message, &self.type_name)
    }
    pub fn is_unsupported_param(&self, param: &str) -> bool {
        is_unsupported_parameter_error(&self.message, param)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alias_normalization() {
        assert_eq!(normalize_aux_provider(Some("glm"), None), "zai");
        assert_eq!(normalize_aux_provider(Some("kimi"), None), "kimi-coding");
        assert_eq!(normalize_aux_provider(Some("codex"), None), "openai-codex");
        assert_eq!(normalize_aux_provider(Some("CLAUDE"), None), "anthropic");
        assert_eq!(normalize_aux_provider(Some("custom:zai"), None), "zai");
        assert_eq!(normalize_aux_provider(Some("custom:"), None), "custom");
        assert_eq!(normalize_aux_provider(None, None), "auto");
        // "main" with a configured main provider resolves to it.
        assert_eq!(normalize_aux_provider(Some("main"), Some("alibaba")), "alibaba");
        // "main" with no main provider falls back to "custom".
        assert_eq!(normalize_aux_provider(Some("main"), Some("auto")), "custom");
        // unknown passes through unchanged.
        assert_eq!(normalize_aux_provider(Some("deepseek"), None), "deepseek");
    }

    #[test]
    fn temperature_contracts() {
        assert_eq!(
            fixed_temperature_for_model(Some("kimi-k2-turbo-preview"), None),
            TemperatureDirective::Omit
        );
        assert_eq!(
            fixed_temperature_for_model(Some("moonshot/kimi-latest"), None),
            TemperatureDirective::Omit
        );
        assert_eq!(
            fixed_temperature_for_model(Some("trinity-large-thinking"), None),
            TemperatureDirective::Fixed(0.5)
        );
        assert_eq!(
            fixed_temperature_for_model(Some("gpt-4o-mini"), None),
            TemperatureDirective::None
        );
        assert_eq!(
            compression_threshold_for_model(Some("trinity-large-thinking")),
            Some(0.75)
        );
        assert_eq!(compression_threshold_for_model(Some("gpt-4o")), None);
    }

    #[test]
    fn aux_model_lookup() {
        assert_eq!(get_aux_model_for_provider("gemini"), "gemini-3-flash-preview");
        assert_eq!(get_aux_model_for_provider("anthropic"), "claude-haiku-4-5-20251001");
        assert_eq!(get_aux_model_for_provider("nonexistent"), "");
    }

    #[test]
    fn base_url_normalization() {
        assert_eq!(
            to_openai_base_url("https://api.minimax.io/anthropic"),
            "https://api.minimax.io/v1"
        );
        assert_eq!(
            to_openai_base_url("https://api.kimi.com/coding"),
            "https://api.kimi.com/coding/v1"
        );
        assert_eq!(
            to_openai_base_url("https://example.com/v1/"),
            "https://example.com/v1"
        );
    }

    #[test]
    fn anthropic_endpoint_detection() {
        assert!(endpoint_speaks_anthropic_messages("https://x.com/anthropic"));
        assert!(endpoint_speaks_anthropic_messages("https://api.anthropic.com"));
        assert!(endpoint_speaks_anthropic_messages("https://api.kimi.com/coding"));
        assert!(!endpoint_speaks_anthropic_messages("https://openrouter.ai/api/v1"));
        assert!(!endpoint_speaks_anthropic_messages(""));

        assert!(is_anthropic_compat_endpoint("minimax", ""));
        assert!(is_anthropic_compat_endpoint("x", "https://h/anthropic/v1"));
        assert!(!is_anthropic_compat_endpoint("openrouter", "https://openrouter.ai"));
    }

    #[test]
    fn hostname_and_query() {
        assert_eq!(base_url_hostname("https://API.OpenAI.com/v1"), "api.openai.com");
        assert!(base_url_host_matches("https://openrouter.ai/api/v1", "openrouter.ai"));
        let (clean, params) = extract_url_query_params("https://h/v1?key=abc&x=1");
        assert_eq!(clean, "https://h/v1");
        let params = params.unwrap();
        assert_eq!(params.get("key").map(|s| s.as_str()), Some("abc"));
        assert_eq!(params.get("x").map(|s| s.as_str()), Some("1"));
        let (clean2, none) = extract_url_query_params("https://h/v1");
        assert_eq!(clean2, "https://h/v1");
        assert!(none.is_none());
    }

    #[test]
    fn codex_headers_and_jwt() {
        // Build a JWT with the chatgpt_account_id claim.
        let claims = json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
            "exp": 9999999999i64
        });
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&claims).unwrap());
        let token = format!("header.{payload}.sig");
        let headers = codex_cloudflare_headers(&token);
        assert_eq!(headers.get("originator").map(|s| s.as_str()), Some("codex_cli_rs"));
        assert_eq!(headers.get("ChatGPT-Account-ID").map(|s| s.as_str()), Some("acct_123"));
        assert!(!codex_token_expired(&token, 1_000_000_000));

        // Expired token.
        let expired_claims = json!({"exp": 100i64});
        let exp_payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&expired_claims).unwrap());
        let exp_token = format!("h.{exp_payload}.s");
        assert!(codex_token_expired(&exp_token, 200));

        // Malformed token: still returns base headers, no account id.
        let bad = codex_cloudflare_headers("not-a-jwt");
        assert!(bad.get("ChatGPT-Account-ID").is_none());
        assert_eq!(bad.get("originator").map(|s| s.as_str()), Some("codex_cli_rs"));
    }

    #[test]
    fn openrouter_headers() {
        let base = build_or_headers(&OpenRouterConfig::default());
        assert_eq!(base.get("X-Title").map(|s| s.as_str()), Some("Hermes Agent"));
        assert!(base.get("X-OpenRouter-Cache").is_none());

        let cfg = OpenRouterConfig {
            response_cache: true,
            response_cache_ttl: Some(600),
            ..Default::default()
        };
        let h = build_or_headers(&cfg);
        assert_eq!(h.get("X-OpenRouter-Cache").map(|s| s.as_str()), Some("true"));
        assert_eq!(h.get("X-OpenRouter-Cache-TTL").map(|s| s.as_str()), Some("600"));

        // Env override disables despite config-on.
        let cfg2 = OpenRouterConfig {
            response_cache: true,
            env_cache: Some("0".to_string()),
            ..Default::default()
        };
        let h2 = build_or_headers(&cfg2);
        assert!(h2.get("X-OpenRouter-Cache").is_none());

        // Env TTL override.
        let cfg3 = OpenRouterConfig {
            response_cache: true,
            env_ttl: Some("120".to_string()),
            ..Default::default()
        };
        let h3 = build_or_headers(&cfg3);
        assert_eq!(h3.get("X-OpenRouter-Cache-TTL").map(|s| s.as_str()), Some("120"));
    }

    #[test]
    fn error_classification() {
        assert!(is_payment_error(Some(402), ""));
        assert!(is_payment_error(Some(429), "insufficient funds to continue"));
        assert!(!is_payment_error(Some(429), "rate limit exceeded"));

        assert!(is_rate_limit_error(Some(429), "rate limit exceeded", ""));
        assert!(is_rate_limit_error(None, "", "RateLimitError"));
        assert!(is_rate_limit_error(Some(429), "generic error", ""));
        assert!(!is_rate_limit_error(Some(429), "out of credits, billing", ""));

        assert!(is_connection_error("APIConnectionError", ""));
        assert!(is_connection_error("", "connection refused by host"));
        assert!(!is_connection_error("ValueError", "bad input"));

        assert!(is_auth_error(Some(401), "", ""));
        assert!(is_auth_error(None, "Error code: 401 unauthorized", ""));
        assert!(is_auth_error(None, "", "AuthenticationError"));
        assert!(!is_auth_error(Some(403), "forbidden", ""));

        assert!(is_unsupported_parameter_error(
            "Unsupported parameter: temperature",
            "temperature"
        ));
        assert!(is_unsupported_temperature_error("temperature is not supported"));
        assert!(!is_unsupported_parameter_error("some other error", "temperature"));
        assert!(!is_unsupported_parameter_error("contains temperature word", "temperature"));
    }

    #[test]
    fn content_conversion_for_responses() {
        // Plain string passes through.
        assert_eq!(
            convert_content_for_responses(&json!("hello")),
            json!("hello")
        );
        // Text + image blocks convert.
        let content = json!([
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA", "detail": "high"}}
        ]);
        let out = convert_content_for_responses(&content);
        let arr = out.as_array().unwrap();
        assert_eq!(arr[0]["type"], "input_text");
        assert_eq!(arr[0]["text"], "describe");
        assert_eq!(arr[1]["type"], "input_image");
        assert_eq!(arr[1]["image_url"], "data:image/png;base64,AAA");
        assert_eq!(arr[1]["detail"], "high");
    }

    #[test]
    fn openai_image_to_anthropic() {
        let messages = vec![
            json!({"role": "user", "content": "plain"}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZZZ"}},
                {"type": "image_url", "image_url": {"url": "https://h/img.png"}}
            ]}),
        ];
        let out = convert_openai_images_to_anthropic(&messages);
        // First message unchanged.
        assert_eq!(out[0]["content"], "plain");
        let blocks = out[1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/jpeg");
        assert_eq!(blocks[1]["source"]["data"], "ZZZ");
        assert_eq!(blocks[2]["source"]["type"], "url");
        assert_eq!(blocks[2]["source"]["url"], "https://h/img.png");
    }

    #[test]
    fn task_resolution() {
        // Explicit provider wins.
        let r = resolve_task_provider_model(None, Some("openrouter"), Some("m"), None, None);
        assert_eq!(r.provider, "openrouter");
        assert_eq!(r.model.as_deref(), Some("m"));

        // base_url forces custom.
        let r = resolve_task_provider_model(None, None, None, Some("https://h"), None);
        assert_eq!(r.provider, "custom");
        assert_eq!(r.base_url.as_deref(), Some("https://h"));

        // No args, no config → auto.
        let r = resolve_task_provider_model(None, None, None, None, None);
        assert_eq!(r.provider, "auto");

        // Task config base_url + api_key → custom.
        let cfg = TaskConfig {
            base_url: Some("https://c".to_string()),
            api_key: Some("k".to_string()),
            ..Default::default()
        };
        let r = resolve_task_provider_model(Some(&cfg), None, None, None, None);
        assert_eq!(r.provider, "custom");
        assert_eq!(r.api_key.as_deref(), Some("k"));

        // Task config provider only.
        let cfg2 = TaskConfig {
            provider: Some("nous".to_string()),
            model: Some("cfgmodel".to_string()),
            ..Default::default()
        };
        let r = resolve_task_provider_model(Some(&cfg2), None, None, None, None);
        assert_eq!(r.provider, "nous");
        assert_eq!(r.model.as_deref(), Some("cfgmodel"));
    }

    #[test]
    fn call_kwargs_build() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        // Kimi model → temperature omitted entirely.
        let k = build_call_kwargs(
            "kimi-coding",
            "kimi-k2",
            messages.clone(),
            Some(0.7),
            Some(1000),
            None,
            None,
            None,
            false,
            false,
        );
        assert!(k.get("temperature").is_none());
        assert_eq!(k["max_tokens"], 1000);

        // Custom + api.openai.com → max_completion_tokens.
        let k2 = build_call_kwargs(
            "custom",
            "gpt-4o",
            messages.clone(),
            Some(0.5),
            Some(500),
            None,
            None,
            Some("https://api.openai.com/v1"),
            false,
            false,
        );
        assert!(k2.get("max_tokens").is_none());
        assert_eq!(k2["max_completion_tokens"], 500);
        assert_eq!(k2["temperature"], 0.5);

        // forbids_sampling_params drops temperature.
        let k3 = build_call_kwargs(
            "anthropic",
            "claude-opus-4-7",
            messages.clone(),
            Some(0.3),
            None,
            None,
            None,
            None,
            true,
            false,
        );
        assert!(k3.get("temperature").is_none());

        // nous active → extra_body tags appended.
        let k4 = build_call_kwargs(
            "nous", "m", messages.clone(), None, None, None, None, None, false, true,
        );
        let tags = k4["extra_body"]["tags"].as_array().unwrap();
        assert!(tags.iter().any(|t| t == "product=hermes-agent"));

        // tools dedup.
        let tools = vec![
            json!({"function": {"name": "a"}}),
            json!({"function": {"name": "a"}}),
            json!({"function": {"name": "b"}}),
        ];
        let k5 = build_call_kwargs(
            "openrouter",
            "m",
            messages,
            None,
            None,
            Some(tools),
            None,
            None,
            false,
            false,
        );
        assert_eq!(k5["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn content_extraction() {
        let resp = json!({"choices": [{"message": {"content": "  hello world  "}}]});
        assert_eq!(extract_content_or_reasoning(&resp), "hello world");

        // Think block stripped, reasoning fallback.
        let resp2 = json!({"choices": [{"message": {
            "content": "<think>internal</think>",
            "reasoning": "the reasoning"
        }}]});
        assert_eq!(extract_content_or_reasoning(&resp2), "the reasoning");

        // reasoning_details array.
        let resp3 = json!({"choices": [{"message": {
            "content": null,
            "reasoning_details": [{"summary": "summary text"}]
        }}]});
        assert_eq!(extract_content_or_reasoning(&resp3), "summary text");
    }

    #[test]
    fn response_validation() {
        let good = json!({"choices": [{"message": {"content": "x"}}]});
        assert!(validate_llm_response(&good, Some("compression")).is_ok());

        let bad = json!("just a string");
        let err = validate_llm_response(&bad, Some("vision")).unwrap_err();
        assert!(err.contains("vision"));

        let none = Value::Null;
        assert!(validate_llm_response(&none, None).is_err());
    }

    #[test]
    fn flatten_body_merges_extra() {
        let kwargs = json!({
            "model": "m",
            "messages": [],
            "extra_body": {"tags": ["x"], "reasoning": {"effort": "low"}}
        });
        let body = ChatCompletionsClient::flatten_body(&kwargs);
        assert_eq!(body["model"], "m");
        assert!(body.get("extra_body").is_none());
        assert_eq!(body["tags"], json!(["x"]));
        assert_eq!(body["reasoning"]["effort"], "low");
    }

    #[test]
    fn max_tokens_param() {
        assert_eq!(
            auxiliary_max_tokens_param(100, false),
            json!({"max_tokens": 100})
        );
        assert_eq!(
            auxiliary_max_tokens_param(100, true),
            json!({"max_completion_tokens": 100})
        );
    }
}
