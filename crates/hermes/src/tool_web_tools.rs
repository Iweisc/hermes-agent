//! Native Rust port of `tools/web_tools.py`.
//!
//! Generic web tools (`web_search`, `web_extract`, `web_crawl`) working against
//! multiple backend providers (Exa, Firecrawl, Parallel, Tavily, SearXNG) and,
//! for Nous Subscribers, through a managed Firecrawl tool-gateway.
//!
//! This port reproduces the Python module's behaviour faithfully:
//!
//! * Backend selection (`_get_backend`, `_get_search_backend`, `_get_extract_backend`,
//!   per-capability overrides, env-var auto-detection).
//! * Tavily request construction + response normalisation.
//! * Exa / Parallel / Firecrawl response shape extraction helpers.
//! * `clean_base64_images` regex stripping.
//! * The high-level `web_search_tool` (sync, fully wired to reqwest::blocking
//!   for Tavily/SearXNG and to backend callbacks for SDK backends).
//! * `web_extract_tool` / `web_crawl_tool` request-construction + normalisation
//!   (the LLM-summarisation pipeline is exposed via a pluggable callback so the
//!   async auxiliary-client dependency need not be ported here).
//!
//! Network calls that the Python code made through vendor SDKs (Exa, Parallel,
//! Firecrawl) are represented here as trait/callback seams (`WebBackends`) so the
//! request/response shapes stay exact while the actual SDK transport is injected
//! by the integration layer.
//!
//! Cross-references (when those modules are wired into the same crate graph):
//! * `crate::tool_managed_tool_gateway` — gateway resolution + Nous token reading.
//! * `crate::tool_tool_backend_helpers` — `prefers_gateway`.
//! * `crate::tool_url_safety` — `is_safe_url`.
//! * `crate::tool_website_policy` — `check_website_access`.
//! * `crate::tool_interrupt` — `is_interrupted`.
//! * `crate::tool_registry` — `tool_error`.
//! * `crate::tool_debug_helpers` — `DebugSession`.

use std::collections::HashMap;
use std::time::Duration;

use regex::Regex;
use serde_json::{json, Map, Value};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Minimum content length to trigger LLM summarisation (Python:
/// `DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION`).
pub const DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION: usize = 5000;

/// 2M chars — refuse to summarise content above this size.
pub const MAX_CONTENT_SIZE: usize = 2_000_000;
/// 500k chars — use chunked summarisation above this size.
pub const CHUNK_THRESHOLD: usize = 500_000;
/// 100k chars per chunk.
pub const CHUNK_SIZE: usize = 100_000;
/// Hard cap on final summariser output size.
pub const MAX_OUTPUT_SIZE: usize = 5000;

/// Valid backend identifiers recognised by config.
pub const VALID_BACKENDS: [&str; 5] = ["parallel", "firecrawl", "tavily", "exa", "searxng"];

// ─── Local error helper (mirrors `tools.registry.tool_error`) ────────────────

/// JSON string `{"error": <message>}` (or `{"error": ..., "success": false}`
/// when `success_false` is set), matching the Python `tool_error`.
///
/// When `crate::tool_registry::tool_error` is available you may route through
/// it instead; this local copy keeps the module self-contained.
pub fn tool_error(message: &str, success_false: bool) -> String {
    let mut map = Map::new();
    map.insert("error".to_string(), Value::String(message.to_string()));
    if success_false {
        map.insert("success".to_string(), Value::Bool(false));
    }
    Value::Object(map).to_string()
}

// ─── Environment helpers ─────────────────────────────────────────────────────

/// `True` when the named env var is set to a non-blank value. Mirrors `_has_env`.
pub fn has_env(name: &str) -> bool {
    std::env::var(name)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

fn env_trimmed(name: &str) -> String {
    std::env::var(name).unwrap_or_default().trim().to_string()
}

// ─── Backend configuration injection ─────────────────────────────────────────

/// External hooks the high-level tools need but whose dependency chains live in
/// other (possibly not-yet-ported) modules. Each field defaults to a behaviour
/// that never blocks startup, mirroring the Python fail-open semantics.
pub struct WebEnv<'a> {
    /// The `web:` section of `config.yaml` as a JSON object. Mirrors
    /// `_load_web_config()`.
    pub web_config: Value,
    /// `tools.tool_backend_helpers.managed_nous_tools_enabled`. Default: false.
    pub managed_nous_tools_enabled: Option<&'a dyn Fn() -> bool>,
    /// `tools.managed_tool_gateway.resolve_managed_tool_gateway("firecrawl", ...)`
    /// returning `Some((gateway_origin, nous_user_token))` when ready.
    pub resolve_firecrawl_gateway: Option<&'a dyn Fn() -> Option<(String, String)>>,
    /// `tools.tool_backend_helpers.prefers_gateway("web")`. Default: false.
    pub prefers_gateway_web: Option<&'a dyn Fn() -> bool>,
}

impl<'a> Default for WebEnv<'a> {
    fn default() -> Self {
        Self {
            web_config: Value::Object(Map::new()),
            managed_nous_tools_enabled: None,
            resolve_firecrawl_gateway: None,
            prefers_gateway_web: None,
        }
    }
}

impl<'a> WebEnv<'a> {
    fn managed_nous_enabled(&self) -> bool {
        self.managed_nous_tools_enabled.map(|f| f()).unwrap_or(false)
    }

    fn tool_gateway_ready(&self) -> bool {
        self.resolve_firecrawl_gateway
            .map(|f| f().is_some())
            .unwrap_or(false)
    }

    fn prefers_gateway(&self) -> bool {
        self.prefers_gateway_web.map(|f| f()).unwrap_or(false)
    }

    fn web_config_str(&self, key: &str) -> String {
        self.web_config
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_lowercase()
            .trim()
            .to_string()
    }
}

// ─── Backend selection ────────────────────────────────────────────────────────

/// Faithful port of `_get_backend()`: read `web.backend`, else auto-detect from
/// env in priority order, else default to `firecrawl`.
pub fn get_backend(env: &WebEnv) -> String {
    let configured = env.web_config_str("backend");
    if VALID_BACKENDS.contains(&configured.as_str()) {
        return configured;
    }

    // Fallback: pick highest-priority available backend.
    let firecrawl_avail =
        has_env("FIRECRAWL_API_KEY") || has_env("FIRECRAWL_API_URL") || env.tool_gateway_ready();
    let candidates: [(&str, bool); 5] = [
        ("firecrawl", firecrawl_avail),
        ("parallel", has_env("PARALLEL_API_KEY")),
        ("tavily", has_env("TAVILY_API_KEY")),
        ("exa", has_env("EXA_API_KEY")),
        ("searxng", has_env("SEARXNG_URL")),
    ];
    for (backend, available) in candidates {
        if available {
            return backend.to_string();
        }
    }
    "firecrawl".to_string()
}

/// `_get_search_backend()`.
pub fn get_search_backend(env: &WebEnv) -> String {
    get_capability_backend(env, "search")
}

/// `_get_extract_backend()`.
pub fn get_extract_backend(env: &WebEnv) -> String {
    get_capability_backend(env, "extract")
}

/// `_get_capability_backend(capability)`.
pub fn get_capability_backend(env: &WebEnv, capability: &str) -> String {
    let specific = env.web_config_str(&format!("{capability}_backend"));
    if !specific.is_empty() && is_backend_available(env, &specific) {
        return specific;
    }
    get_backend(env)
}

/// `_is_backend_available(backend)`.
pub fn is_backend_available(env: &WebEnv, backend: &str) -> bool {
    match backend {
        "exa" => has_env("EXA_API_KEY"),
        "parallel" => has_env("PARALLEL_API_KEY"),
        "firecrawl" => check_firecrawl_api_key(env),
        "tavily" => has_env("TAVILY_API_KEY"),
        "searxng" => has_env("SEARXNG_URL"),
        _ => false,
    }
}

/// `check_firecrawl_api_key()` — direct config OR tool-gateway readiness.
pub fn check_firecrawl_api_key(env: &WebEnv) -> bool {
    has_direct_firecrawl_config() || env.tool_gateway_ready()
}

/// `check_web_api_key()`.
pub fn check_web_api_key(env: &WebEnv) -> bool {
    let configured = env.web_config_str("backend");
    if VALID_BACKENDS.contains(&configured.as_str()) {
        return is_backend_available(env, &configured);
    }
    VALID_BACKENDS
        .iter()
        .any(|b| is_backend_available(env, b))
}

// ─── Firecrawl direct config ───────────────────────────────────────────────────

/// Cache key for the resolved Firecrawl client (mirrors the Python
/// `_firecrawl_client_config` tuple).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FirecrawlConfig {
    /// (`("direct", api_url, api_key)`)
    Direct {
        api_url: Option<String>,
        api_key: Option<String>,
    },
    /// (`("tool-gateway", api_url, token)`)
    ToolGateway { api_url: String, token: String },
}

/// `_get_direct_firecrawl_config()` — kwargs + cache key, or None when unset.
///
/// Returns `Some((kwargs, FirecrawlConfig::Direct))` where `kwargs` is the
/// `(api_key?, api_url?)` pair to pass to the SDK constructor.
pub fn get_direct_firecrawl_config() -> Option<(FirecrawlKwargs, FirecrawlConfig)> {
    let api_key = env_trimmed("FIRECRAWL_API_KEY");
    let api_url = env_trimmed("FIRECRAWL_API_URL")
        .trim_end_matches('/')
        .to_string();

    if api_key.is_empty() && api_url.is_empty() {
        return None;
    }

    let kwargs = FirecrawlKwargs {
        api_key: if api_key.is_empty() {
            None
        } else {
            Some(api_key.clone())
        },
        api_url: if api_url.is_empty() {
            None
        } else {
            Some(api_url.clone())
        },
    };
    let cfg = FirecrawlConfig::Direct {
        api_url: if api_url.is_empty() { None } else { Some(api_url) },
        api_key: if api_key.is_empty() { None } else { Some(api_key) },
    };
    Some((kwargs, cfg))
}

/// Constructor kwargs for the Firecrawl SDK client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirecrawlKwargs {
    pub api_key: Option<String>,
    pub api_url: Option<String>,
}

/// `_has_direct_firecrawl_config()`.
pub fn has_direct_firecrawl_config() -> bool {
    get_direct_firecrawl_config().is_some()
}

/// Resolve the Firecrawl client config (kwargs + cache key), mirroring
/// `_get_firecrawl_client()`'s config-resolution branch.
///
/// Returns `Err(message)` matching `_raise_web_backend_configuration_error()`
/// when neither direct config nor a managed gateway is available.
pub fn resolve_firecrawl_client_config(
    env: &WebEnv,
) -> Result<(FirecrawlKwargs, FirecrawlConfig), String> {
    let direct = get_direct_firecrawl_config();
    if let Some((kwargs, cfg)) = direct {
        if !env.prefers_gateway() {
            return Ok((kwargs, cfg));
        }
    }

    match env.resolve_firecrawl_gateway.and_then(|f| f()) {
        Some((gateway_origin, token)) => {
            let kwargs = FirecrawlKwargs {
                api_key: Some(token.clone()),
                api_url: Some(gateway_origin.clone()),
            };
            let cfg = FirecrawlConfig::ToolGateway {
                api_url: gateway_origin,
                token,
            };
            Ok((kwargs, cfg))
        }
        None => Err(web_backend_configuration_error(env)),
    }
}

/// `_raise_web_backend_configuration_error()` message text.
pub fn web_backend_configuration_error(env: &WebEnv) -> String {
    let mut message = String::from(
        "Web tools are not configured. \
Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL for a self-hosted Firecrawl instance.",
    );
    if env.managed_nous_enabled() {
        message.push_str(
            " With your Nous subscription you can also use the Tool Gateway — \
run `hermes tools` and select Nous Subscription as the web provider.",
        );
    }
    message
}

/// `_firecrawl_backend_help_suffix()`.
pub fn firecrawl_backend_help_suffix(env: &WebEnv) -> String {
    if !env.managed_nous_enabled() {
        return String::new();
    }
    ", or use the Nous Tool Gateway via your subscription \
(FIRECRAWL_GATEWAY_URL or TOOL_GATEWAY_DOMAIN)"
        .to_string()
}

/// `_web_requires_env()` — tool metadata env vars for enabled web backends.
pub fn web_requires_env(env: &WebEnv) -> Vec<String> {
    let mut requires = vec![
        "EXA_API_KEY".to_string(),
        "PARALLEL_API_KEY".to_string(),
        "TAVILY_API_KEY".to_string(),
        "FIRECRAWL_API_KEY".to_string(),
        "FIRECRAWL_API_URL".to_string(),
    ];
    if env.managed_nous_enabled() {
        requires.extend([
            "FIRECRAWL_GATEWAY_URL".to_string(),
            "TOOL_GATEWAY_DOMAIN".to_string(),
            "TOOL_GATEWAY_SCHEME".to_string(),
            "TOOL_GATEWAY_USER_TOKEN".to_string(),
        ]);
    }
    requires
}

// ─── Tavily ────────────────────────────────────────────────────────────────────

fn tavily_base_url() -> String {
    std::env::var("TAVILY_BASE_URL").unwrap_or_else(|_| "https://api.tavily.com".to_string())
}

/// Send a POST to the Tavily API. Auth is provided via `api_key` in the JSON
/// body. Mirrors `_tavily_request`. Returns the parsed JSON response.
pub fn tavily_request(endpoint: &str, mut payload: Map<String, Value>) -> Result<Value, String> {
    let api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Err(
            "TAVILY_API_KEY environment variable not set. \
Get your API key at https://app.tavily.com/home"
                .to_string(),
        );
    }
    payload.insert("api_key".to_string(), Value::String(api_key));

    let base = tavily_base_url();
    let url = format!("{}/{}", base, endpoint.trim_start_matches('/'));
    log::info!("Tavily {endpoint} request to {url}");

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .json(&Value::Object(payload))
        .send()
        .map_err(|e| e.to_string())?;
    let resp = resp.error_for_status().map_err(|e| e.to_string())?;
    resp.json::<Value>().map_err(|e| e.to_string())
}

/// `_normalize_tavily_search_results(response)`.
pub fn normalize_tavily_search_results(response: &Value) -> Value {
    let mut web_results = Vec::new();
    if let Some(results) = response.get("results").and_then(Value::as_array) {
        for (i, result) in results.iter().enumerate() {
            web_results.push(json!({
                "title": result.get("title").and_then(Value::as_str).unwrap_or(""),
                "url": result.get("url").and_then(Value::as_str).unwrap_or(""),
                "description": result.get("content").and_then(Value::as_str).unwrap_or(""),
                "position": i + 1,
            }));
        }
    }
    json!({ "success": true, "data": { "web": web_results } })
}

/// `_normalize_tavily_documents(response, fallback_url)`.
pub fn normalize_tavily_documents(response: &Value, fallback_url: &str) -> Vec<Value> {
    let mut documents = Vec::new();

    if let Some(results) = response.get("results").and_then(Value::as_array) {
        for result in results {
            let url = result
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or(fallback_url)
                .to_string();
            let raw = {
                let r = result.get("raw_content").and_then(Value::as_str).unwrap_or("");
                if r.is_empty() {
                    result.get("content").and_then(Value::as_str).unwrap_or("")
                } else {
                    r
                }
            }
            .to_string();
            let title = result.get("title").and_then(Value::as_str).unwrap_or("");
            documents.push(json!({
                "url": url,
                "title": title,
                "content": raw,
                "raw_content": raw,
                "metadata": { "sourceURL": url, "title": title },
            }));
        }
    }

    if let Some(failed) = response.get("failed_results").and_then(Value::as_array) {
        for fail in failed {
            let url = fail
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or(fallback_url)
                .to_string();
            documents.push(json!({
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": fail.get("error").and_then(Value::as_str).unwrap_or("extraction failed"),
                "metadata": { "sourceURL": url },
            }));
        }
    }

    if let Some(failed_urls) = response.get("failed_urls").and_then(Value::as_array) {
        for fail_url in failed_urls {
            let url_str = match fail_url {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            documents.push(json!({
                "url": url_str,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "extraction failed",
                "metadata": { "sourceURL": url_str },
            }));
        }
    }

    documents
}

// ─── Generic SDK payload normalisation ──────────────────────────────────────────

/// `_to_plain_object(value)` — already-plain JSON passes through; everything
/// else (SDK objects in Python) is assumed already converted to `Value` here.
pub fn to_plain_object(value: &Value) -> Value {
    value.clone()
}

/// `_normalize_result_list(values)` — keep only dict items from a list.
pub fn normalize_result_list(values: &Value) -> Vec<Value> {
    match values {
        Value::Array(items) => items
            .iter()
            .filter(|item| item.is_object())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

/// `_extract_web_search_results(response)` — Firecrawl search result extraction
/// across SDK / direct / gateway shapes.
pub fn extract_web_search_results(response: &Value) -> Vec<Value> {
    if let Value::Object(_) = response {
        if let Some(data) = response.get("data") {
            if data.is_array() {
                return normalize_result_list(data);
            }
            if data.is_object() {
                let data_web = normalize_result_list(data.get("web").unwrap_or(&Value::Null));
                if !data_web.is_empty() {
                    return data_web;
                }
                let data_results =
                    normalize_result_list(data.get("results").unwrap_or(&Value::Null));
                if !data_results.is_empty() {
                    return data_results;
                }
            }
        }

        let top_web = normalize_result_list(response.get("web").unwrap_or(&Value::Null));
        if !top_web.is_empty() {
            return top_web;
        }

        let top_results = normalize_result_list(response.get("results").unwrap_or(&Value::Null));
        if !top_results.is_empty() {
            return top_results;
        }
    }

    Vec::new()
}

/// `_extract_scrape_payload(scrape_result)` — Firecrawl scrape payload shape.
pub fn extract_scrape_payload(scrape_result: &Value) -> Value {
    if !scrape_result.is_object() {
        return Value::Object(Map::new());
    }
    if let Some(nested) = scrape_result.get("data") {
        if nested.is_object() {
            return nested.clone();
        }
    }
    scrape_result.clone()
}

// ─── base64 image cleaning ──────────────────────────────────────────────────────

/// `clean_base64_images(text)` — strip data-URI base64 images (with and without
/// surrounding parentheses) replacing them with `[BASE64_IMAGE_REMOVED]`.
pub fn clean_base64_images(text: &str) -> String {
    // Parentheses-wrapped first, then bare.
    static_replace_base64(text)
}

fn static_replace_base64(text: &str) -> String {
    let with_parens =
        Regex::new(r"\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)").expect("valid regex");
    let bare = Regex::new(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+").expect("valid regex");

    let step1 = with_parens.replace_all(text, "[BASE64_IMAGE_REMOVED]");
    let step2 = bare.replace_all(&step1, "[BASE64_IMAGE_REMOVED]");
    step2.into_owned()
}

// ─── Exa / Parallel response normalisation ──────────────────────────────────────

/// Normalise an Exa `search` SDK response into the standard search dict.
///
/// `results` is a list of objects with `url`, `title`, `highlights` (array of
/// strings). Mirrors `_exa_search`'s post-processing.
pub fn normalize_exa_search(results: &[Value]) -> Value {
    let mut web_results = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let highlights: Vec<&str> = result
            .get("highlights")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        web_results.push(json!({
            "url": result.get("url").and_then(Value::as_str).unwrap_or(""),
            "title": result.get("title").and_then(Value::as_str).unwrap_or(""),
            "description": highlights.join(" "),
            "position": i + 1,
        }));
    }
    json!({ "success": true, "data": { "web": web_results } })
}

/// Normalise an Exa `get_contents` SDK response into document dicts.
/// `results` is a list of objects with `text`, `url`, `title`.
pub fn normalize_exa_extract(results: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for result in results {
        let content = result.get("text").and_then(Value::as_str).unwrap_or("");
        let url = result.get("url").and_then(Value::as_str).unwrap_or("");
        let title = result.get("title").and_then(Value::as_str).unwrap_or("");
        out.push(json!({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": { "sourceURL": url, "title": title },
        }));
    }
    out
}

/// Normalise the Parallel search mode value. Mirrors `_parallel_search`'s
/// `PARALLEL_SEARCH_MODE` handling.
pub fn parallel_search_mode() -> String {
    let mode = std::env::var("PARALLEL_SEARCH_MODE")
        .unwrap_or_else(|_| "agentic".to_string())
        .to_lowercase()
        .trim()
        .to_string();
    if matches!(mode.as_str(), "fast" | "one-shot" | "agentic") {
        mode
    } else {
        "agentic".to_string()
    }
}

/// Normalise a Parallel `beta.search` response. `results` objects carry `url`,
/// `title`, `excerpts` (array of strings). Mirrors `_parallel_search`.
pub fn normalize_parallel_search(results: &[Value]) -> Value {
    let mut web_results = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let excerpts: Vec<&str> = result
            .get("excerpts")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        web_results.push(json!({
            "url": result.get("url").and_then(Value::as_str).unwrap_or(""),
            "title": result.get("title").and_then(Value::as_str).unwrap_or(""),
            "description": excerpts.join(" "),
            "position": i + 1,
        }));
    }
    json!({ "success": true, "data": { "web": web_results } })
}

/// Normalise a Parallel `beta.extract` response, including `errors`.
/// `results` objects carry `full_content`, `excerpts`, `url`, `title`.
/// `errors` objects carry `url`, `content`, `error_type`. Mirrors
/// `_parallel_extract`.
pub fn normalize_parallel_extract(results: &[Value], errors: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for result in results {
        let mut content = result
            .get("full_content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if content.is_empty() {
            let excerpts: Vec<&str> = result
                .get("excerpts")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            content = excerpts.join("\n\n");
        }
        let url = result.get("url").and_then(Value::as_str).unwrap_or("");
        let title = result.get("title").and_then(Value::as_str).unwrap_or("");
        out.push(json!({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": { "sourceURL": url, "title": title },
        }));
    }
    for error in errors {
        let url = error.get("url").and_then(Value::as_str).unwrap_or("");
        let err_msg = {
            let c = error.get("content").and_then(Value::as_str).unwrap_or("");
            if !c.is_empty() {
                c
            } else {
                let t = error
                    .get("error_type")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !t.is_empty() {
                    t
                } else {
                    "extraction failed"
                }
            }
        };
        out.push(json!({
            "url": url,
            "title": "",
            "content": "",
            "error": err_msg,
            "metadata": { "sourceURL": url },
        }));
    }
    out
}

// ─── Document trimming / shaping shared by extract+crawl ─────────────────────────

/// Trim a result dict to the minimal fields (`url`, `title`, `content`,
/// `error`, optional `blocked_by_policy`). Mirrors the `trimmed_results`
/// comprehensions in both `web_extract_tool` and `web_crawl_tool`.
pub fn trim_result(r: &Value) -> Value {
    let mut map = Map::new();
    map.insert(
        "url".to_string(),
        Value::String(r.get("url").and_then(Value::as_str).unwrap_or("").to_string()),
    );
    map.insert(
        "title".to_string(),
        Value::String(
            r.get("title")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
    );
    map.insert(
        "content".to_string(),
        Value::String(
            r.get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
    );
    // Python sets `"error": r.get("error")` which yields null when absent.
    map.insert(
        "error".to_string(),
        r.get("error").cloned().unwrap_or(Value::Null),
    );
    if let Some(b) = r.get("blocked_by_policy") {
        map.insert("blocked_by_policy".to_string(), b.clone());
    }
    Value::Object(map)
}

/// Build the SSRF-blocked result entry. Mirrors the `ssrf_blocked` dict.
pub fn ssrf_blocked_entry(url: &str) -> Value {
    json!({
        "url": url,
        "title": "",
        "content": "",
        "error": "Blocked: URL targets a private or internal network address",
    })
}

/// Build the website-policy-blocked result entry.
pub fn policy_blocked_entry(url: &str, host: &str, rule: &str, source: &str, message: &str) -> Value {
    json!({
        "url": url,
        "title": "",
        "content": "",
        "error": message,
        "blocked_by_policy": { "host": host, "rule": rule, "source": source },
    })
}

// ─── Firecrawl scrape result shaping (web_extract) ──────────────────────────────

/// Shape a single Firecrawl scrape payload into the extract result dict.
///
/// `scrape_result` is the raw SDK/gateway payload; `format` is the requested
/// output format (`"markdown"`, `"html"`, or `None`). `requested_url` is the
/// originally requested URL (used as `sourceURL` fallback). Returns the result
/// dict mirroring the success branch of `web_extract_tool`'s Firecrawl loop.
///
/// Returns `(result, final_url)` so the caller can run the redirect policy
/// re-check on `final_url` before keeping the entry.
pub fn shape_firecrawl_scrape(
    scrape_result: &Value,
    format: Option<&str>,
    requested_url: &str,
) -> (Value, String) {
    let payload = extract_scrape_payload(scrape_result);
    let metadata = payload.get("metadata").cloned().unwrap_or(Value::Null);
    let metadata = if metadata.is_object() {
        metadata
    } else {
        Value::Object(Map::new())
    };

    let title = metadata
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let final_url = metadata
        .get("sourceURL")
        .and_then(Value::as_str)
        .unwrap_or(requested_url)
        .to_string();

    let content_markdown = payload.get("markdown").and_then(Value::as_str);
    let content_html = payload.get("html").and_then(Value::as_str);

    // chosen_content = markdown if (format=="markdown" or (format is None and markdown))
    //                  else html or markdown or ""
    // Note: Python truthiness — an empty-string markdown is falsy, so the
    // `format is None and markdown` clause requires a non-empty markdown.
    let markdown_truthy = content_markdown.map(|m| !m.is_empty()).unwrap_or(false);
    let chosen = if format == Some("markdown") || (format.is_none() && markdown_truthy) {
        content_markdown.unwrap_or("").to_string()
    } else {
        // html or markdown or ""  (Python `or` skips empty strings)
        first_truthy(&[content_html, content_markdown])
    };

    let result = json!({
        "url": final_url,
        "title": title,
        "content": chosen,
        "raw_content": chosen,
        "metadata": metadata,
    });
    (result, final_url)
}

/// Determine the Firecrawl `formats` list for a requested output format.
/// Mirrors the `formats` computation in `web_extract_tool`.
pub fn firecrawl_formats(format: Option<&str>) -> Vec<String> {
    match format {
        Some("markdown") => vec!["markdown".to_string()],
        Some("html") => vec!["html".to_string()],
        _ => vec!["markdown".to_string(), "html".to_string()],
    }
}

// ─── Firecrawl crawl page shaping (web_crawl) ───────────────────────────────────

/// Shape a single crawled Firecrawl document into a page dict. Mirrors the
/// per-item loop in `web_crawl_tool`. `item` is a plain JSON object with
/// `markdown`, `html`, `metadata`.
///
/// Returns `(page, page_url)`. The caller is responsible for the per-page
/// website-policy re-check on `page_url`.
pub fn shape_firecrawl_crawl_page(item: &Value) -> (Value, String) {
    let content_markdown = item.get("markdown").and_then(Value::as_str);
    let content_html = item.get("html").and_then(Value::as_str);
    let metadata = item.get("metadata").cloned().unwrap_or(Value::Null);
    let metadata = if metadata.is_object() {
        metadata
    } else {
        Value::Object(Map::new())
    };

    let page_url = metadata
        .get("sourceURL")
        .and_then(Value::as_str)
        .or_else(|| metadata.get("url").and_then(Value::as_str))
        .unwrap_or("Unknown URL")
        .to_string();
    let title = metadata
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // content = markdown or html or ""  (Python `or` skips empty strings)
    let content = first_truthy(&[content_markdown, content_html]);

    let page = json!({
        "url": page_url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": metadata,
    });
    (page, page_url)
}

// ─── Embedded-secret URL guard (web_extract) ────────────────────────────────────

/// Detect URLs that carry what looks like an API key / token. Mirrors the
/// `_PREFIX_RE` check in `web_extract_tool` (run on both the raw URL and its
/// percent-decoded form).
///
/// When `crate::agent_redact` is wired in you should delegate to its
/// `_PREFIX_RE`; this conservative local regex matches the common prefixes
/// (`sk-`, `pk-`, `xoxb-`, `ghp_`, …) so the guard still fires standalone.
pub fn url_contains_secret(url: &str) -> bool {
    let decoded = percent_decode(url);
    let re = secret_prefix_regex();
    re.is_match(url) || re.is_match(&decoded)
}

fn secret_prefix_regex() -> Regex {
    // Word-boundary guarded common secret prefixes.
    Regex::new(
        r"(?i)\b(sk-[A-Za-z0-9]|pk-[A-Za-z0-9]|xox[baprs]-|ghp_|gho_|github_pat_|AKIA[0-9A-Z]{8}|AIza[0-9A-Za-z_\-]{8})",
    )
    .expect("valid regex")
}

/// Minimal percent-decoding (mirrors Python `urllib.parse.unquote`).
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ─── LLM summarisation output capping (sync, callback-driven) ───────────────────

/// Apply the post-processing output cap used by `process_content_with_llm`:
/// truncate to `MAX_OUTPUT_SIZE` with the standard suffix.
pub fn cap_summary_output(processed: &str) -> String {
    if processed.len() > MAX_OUTPUT_SIZE {
        let mut s = char_truncate(processed, MAX_OUTPUT_SIZE);
        s.push_str("\n\n[... summary truncated for context management ...]");
        s
    } else {
        processed.to_string()
    }
}

/// Decision returned by [`content_processing_decision`] describing what the
/// async summariser pipeline would do for a given content length, mirroring the
/// branch structure at the top of `process_content_with_llm`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentDecision {
    /// `content_len > MAX_CONTENT_SIZE` — refuse, returning this placeholder.
    Refuse(String),
    /// `content_len < min_length` — skip processing (Python returns None).
    TooShort,
    /// `content_len > CHUNK_THRESHOLD` — use chunked processing.
    Chunked,
    /// Standard single-pass processing.
    SinglePass,
}

/// Faithful port of the size-threshold branching in `process_content_with_llm`.
pub fn content_processing_decision(content_len: usize, min_length: usize) -> ContentDecision {
    if content_len > MAX_CONTENT_SIZE {
        let size_mb = content_len as f64 / 1_000_000.0;
        return ContentDecision::Refuse(format!(
            "[Content too large to process: {size_mb:.1}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        ));
    }
    if content_len < min_length {
        return ContentDecision::TooShort;
    }
    if content_len > CHUNK_THRESHOLD {
        return ContentDecision::Chunked;
    }
    ContentDecision::SinglePass
}

/// Build the context prefix string from optional title/url. Mirrors the
/// `context_str` construction in `process_content_with_llm`.
pub fn build_context_str(title: &str, url: &str) -> String {
    let mut parts = Vec::new();
    if !title.is_empty() {
        parts.push(format!("Title: {title}"));
    }
    if !url.is_empty() {
        parts.push(format!("Source: {url}"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", parts.join("\n"))
    }
}

/// The fallback returned when summarisation fails: first `MAX_OUTPUT_SIZE` chars
/// of the raw content plus an explanatory suffix when truncated. Mirrors the
/// `except` block of `process_content_with_llm`.
pub fn summarization_failure_fallback(content: &str) -> String {
    let mut truncated = char_truncate(content, MAX_OUTPUT_SIZE);
    if content.len() > MAX_OUTPUT_SIZE {
        truncated.push_str(&format!(
            "\n\n[Content truncated — showing first {} of {} chars. LLM summarization timed out. \
To fix: increase auxiliary.web_extract.timeout in config.yaml, \
or use a faster auxiliary model. Use browser_navigate for the full page.]",
            comma_group(MAX_OUTPUT_SIZE),
            comma_group(content.len()),
        ));
    }
    truncated
}

/// Return the first non-empty `&str` from the candidates, or `""` (mirrors a
/// Python `a or b or ""` chain, where empty strings are falsy).
fn first_truthy(candidates: &[Option<&str>]) -> String {
    for c in candidates {
        if let Some(s) = c {
            if !s.is_empty() {
                return s.to_string();
            }
        }
    }
    String::new()
}

/// Truncate a string to at most `max_chars` characters (char-boundary safe).
fn char_truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Format a usize with comma thousands separators (Python `{:,}`).
fn comma_group(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

// ─── web_search_tool (sync) ──────────────────────────────────────────────────────

/// Limit-clamping used by `web_search_tool`: parse, clamp to [1, 100].
pub fn clamp_search_limit(limit: Option<i64>) -> i64 {
    let limit = limit.unwrap_or(5);
    limit.clamp(1, 100)
}

/// Backend dispatch seam for SDK-based search backends (Exa, Parallel,
/// Firecrawl). Each returns the standard `{success, data:{web:[...]}}` value or
/// an `Err(message)`.
pub trait SearchBackends {
    fn exa_search(&self, query: &str, limit: i64) -> Result<Value, String>;
    fn parallel_search(&self, query: &str, limit: i64) -> Result<Value, String>;
    /// Firecrawl `.search(query, limit)` returning the raw SDK/gateway response,
    /// which is then run through [`extract_web_search_results`].
    fn firecrawl_search(&self, query: &str, limit: i64) -> Result<Value, String>;
    /// SearXNG provider `.search(query, limit)` returning the standard dict.
    fn searxng_search(&self, query: &str, limit: i64) -> Result<Value, String>;
}

/// Faithful port of `web_search_tool`. Network backends are supplied via
/// `backends`; Tavily is handled inline with reqwest::blocking.
///
/// `is_interrupted` mirrors `tools.interrupt.is_interrupted`.
pub fn web_search_tool(
    query: &str,
    limit: Option<i64>,
    env: &WebEnv,
    backends: &dyn SearchBackends,
    is_interrupted: &dyn Fn() -> bool,
) -> String {
    let limit = clamp_search_limit(limit);

    if is_interrupted() {
        return tool_error("Interrupted", true);
    }

    let backend = get_search_backend(env);

    let result: Result<Value, String> = match backend.as_str() {
        "parallel" => backends.parallel_search(query, limit),
        "exa" => backends.exa_search(query, limit),
        "searxng" => backends.searxng_search(query, limit),
        "tavily" => {
            log::info!("Tavily search: '{query}' (limit: {limit})");
            let mut payload = Map::new();
            payload.insert("query".to_string(), Value::String(query.to_string()));
            payload.insert(
                "max_results".to_string(),
                Value::from(limit.min(20)),
            );
            payload.insert("include_raw_content".to_string(), Value::Bool(false));
            payload.insert("include_images".to_string(), Value::Bool(false));
            tavily_request("search", payload).map(|raw| normalize_tavily_search_results(&raw))
        }
        _ => {
            log::info!("Searching the web for: '{query}' (limit: {limit})");
            backends.firecrawl_search(query, limit).map(|resp| {
                let web_results = extract_web_search_results(&resp);
                json!({ "success": true, "data": { "web": web_results } })
            })
        }
    };

    match result {
        Ok(data) => serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string()),
        Err(e) => {
            let error_msg = format!("Error searching web: {e}");
            log::debug!("{error_msg}");
            tool_error(&error_msg, false)
        }
    }
}

/// SearXNG-as-extract / crawl rejection messages, exposed for the integration
/// layer to short-circuit those backends. Mirrors the inline JSON in
/// `web_extract_tool` / `web_crawl_tool`.
pub fn searxng_extract_unsupported() -> String {
    json!({
        "success": false,
        "error": "SearXNG is a search-only backend and cannot extract URL content. \
Set web.extract_backend to firecrawl, tavily, exa, or parallel.",
    })
    .to_string()
}

/// SearXNG crawl rejection. Mirrors the inline JSON in `web_crawl_tool`.
pub fn searxng_crawl_unsupported() -> String {
    json!({
        "error": "SearXNG is a search-only backend and cannot crawl URLs. \
Set FIRECRAWL_API_KEY for crawling, or use web_search instead.",
        "success": false,
    })
    .to_string()
}

/// `web_crawl` Firecrawl-required rejection. Mirrors the inline JSON.
pub fn crawl_requires_firecrawl(env: &WebEnv) -> String {
    json!({
        "error": format!(
            "web_crawl requires Firecrawl. Set FIRECRAWL_API_KEY, FIRECRAWL_API_URL{}, or use web_search + web_extract instead.",
            firecrawl_backend_help_suffix(env)
        ),
        "success": false,
    })
    .to_string()
}

// ─── Tool schemas + metadata ─────────────────────────────────────────────────────

/// `WEB_SEARCH_SCHEMA`.
pub fn web_search_schema() -> Value {
    json!({
        "name": "web_search",
        "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 5.",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5
                }
            },
            "required": ["query"]
        }
    })
}

/// `WEB_EXTRACT_SCHEMA`.
pub fn web_extract_schema() -> Value {
    json!({
        "name": "web_extract",
        "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of URLs to extract content from (max 5 URLs per call)",
                    "maxItems": 5
                }
            },
            "required": ["urls"]
        }
    })
}

/// Compression metric entry shared by extract/crawl debug output.
pub fn compression_metric(
    url: &str,
    original_size: usize,
    processed_size: usize,
    model_used: Option<&str>,
    reason: Option<&str>,
) -> Value {
    let ratio = if original_size > 0 {
        processed_size as f64 / original_size as f64
    } else {
        1.0
    };
    let mut map = Map::new();
    map.insert("url".to_string(), Value::String(url.to_string()));
    map.insert("original_size".to_string(), Value::from(original_size));
    map.insert("processed_size".to_string(), Value::from(processed_size));
    map.insert(
        "compression_ratio".to_string(),
        Value::from(ratio),
    );
    map.insert(
        "model_used".to_string(),
        model_used
            .map(|m| Value::String(m.to_string()))
            .unwrap_or(Value::Null),
    );
    if let Some(r) = reason {
        map.insert("reason".to_string(), Value::String(r.to_string()));
    }
    Value::Object(map)
}

/// Convenience: map of backend -> human display string for the `__main__`-style
/// status output. Not used at runtime but mirrors the diagnostics text.
pub fn backend_display_names() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("exa", "Using Exa API (https://exa.ai)");
    m.insert("parallel", "Using Parallel API (https://parallel.ai)");
    m.insert("tavily", "Using Tavily API (https://tavily.com)");
    m
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_base64_images_parens_and_bare() {
        let text = "before (data:image/png;base64,AAAABBBB==) middle data:image/jpeg;base64,CCCCDDDD after";
        let cleaned = clean_base64_images(text);
        assert!(cleaned.contains("[BASE64_IMAGE_REMOVED]"));
        assert!(!cleaned.contains("base64,"));
        assert!(cleaned.contains("before"));
        assert!(cleaned.contains("after"));
    }

    #[test]
    fn test_clean_base64_images_no_match() {
        let text = "no images here, just text";
        assert_eq!(clean_base64_images(text), text);
    }

    #[test]
    fn test_clamp_search_limit() {
        assert_eq!(clamp_search_limit(None), 5);
        assert_eq!(clamp_search_limit(Some(0)), 1);
        assert_eq!(clamp_search_limit(Some(200)), 100);
        assert_eq!(clamp_search_limit(Some(7)), 7);
        assert_eq!(clamp_search_limit(Some(-3)), 1);
    }

    #[test]
    fn test_normalize_tavily_search_results() {
        let resp = json!({
            "results": [
                {"title": "T1", "url": "http://a", "content": "desc1"},
                {"title": "T2", "url": "http://b", "content": "desc2"},
            ]
        });
        let out = normalize_tavily_search_results(&resp);
        assert_eq!(out["success"], json!(true));
        let web = out["data"]["web"].as_array().unwrap();
        assert_eq!(web.len(), 2);
        assert_eq!(web[0]["position"], json!(1));
        assert_eq!(web[0]["description"], json!("desc1"));
        assert_eq!(web[1]["position"], json!(2));
    }

    #[test]
    fn test_normalize_tavily_documents_with_failures() {
        let resp = json!({
            "results": [{"url": "http://x", "title": "X", "raw_content": "raw"}],
            "failed_results": [{"url": "http://y", "error": "boom"}],
            "failed_urls": ["http://z"],
        });
        let docs = normalize_tavily_documents(&resp, "http://fallback");
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0]["content"], json!("raw"));
        assert_eq!(docs[1]["error"], json!("boom"));
        assert_eq!(docs[2]["url"], json!("http://z"));
        assert_eq!(docs[2]["error"], json!("extraction failed"));
    }

    #[test]
    fn test_normalize_tavily_documents_content_fallback() {
        // raw_content empty -> falls back to content
        let resp = json!({"results": [{"url": "http://x", "content": "fallback-content"}]});
        let docs = normalize_tavily_documents(&resp, "");
        assert_eq!(docs[0]["content"], json!("fallback-content"));
        assert_eq!(docs[0]["raw_content"], json!("fallback-content"));
    }

    #[test]
    fn test_extract_web_search_results_data_list() {
        let resp = json!({"data": [{"url": "a"}, {"url": "b"}, "not-a-dict"]});
        let out = extract_web_search_results(&resp);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_extract_web_search_results_data_web() {
        let resp = json!({"data": {"web": [{"url": "a"}], "results": [{"url": "ignored"}]}});
        let out = extract_web_search_results(&resp);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["url"], json!("a"));
    }

    #[test]
    fn test_extract_web_search_results_top_web() {
        let resp = json!({"web": [{"url": "a"}, {"url": "b"}]});
        let out = extract_web_search_results(&resp);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_extract_scrape_payload_nested() {
        let r = json!({"data": {"markdown": "md"}});
        assert_eq!(extract_scrape_payload(&r), json!({"markdown": "md"}));
        let r2 = json!({"markdown": "top"});
        assert_eq!(extract_scrape_payload(&r2), json!({"markdown": "top"}));
        assert_eq!(extract_scrape_payload(&json!("x")), json!({}));
    }

    #[test]
    fn test_firecrawl_formats() {
        assert_eq!(firecrawl_formats(Some("markdown")), vec!["markdown"]);
        assert_eq!(firecrawl_formats(Some("html")), vec!["html"]);
        assert_eq!(firecrawl_formats(None), vec!["markdown", "html"]);
        assert_eq!(firecrawl_formats(Some("other")), vec!["markdown", "html"]);
    }

    #[test]
    fn test_shape_firecrawl_scrape_markdown() {
        let scrape = json!({
            "data": {
                "markdown": "# Hi",
                "html": "<h1>Hi</h1>",
                "metadata": {"title": "Page", "sourceURL": "http://final"}
            }
        });
        let (result, final_url) = shape_firecrawl_scrape(&scrape, Some("markdown"), "http://orig");
        assert_eq!(final_url, "http://final");
        assert_eq!(result["content"], json!("# Hi"));
        assert_eq!(result["title"], json!("Page"));
    }

    #[test]
    fn test_shape_firecrawl_scrape_html_fallback() {
        // format html -> html chosen
        let scrape = json!({"markdown": "md", "html": "<p>", "metadata": {}});
        let (result, url) = shape_firecrawl_scrape(&scrape, Some("html"), "http://o");
        assert_eq!(result["content"], json!("<p>"));
        assert_eq!(url, "http://o"); // no sourceURL -> requested_url
    }

    #[test]
    fn test_shape_firecrawl_scrape_default_prefers_markdown() {
        let scrape = json!({"markdown": "m", "html": "h", "metadata": {}});
        let (result, _) = shape_firecrawl_scrape(&scrape, None, "http://o");
        assert_eq!(result["content"], json!("m"));
    }

    #[test]
    fn test_shape_firecrawl_crawl_page() {
        let item = json!({"markdown": "content", "metadata": {"sourceURL": "http://p", "title": "T"}});
        let (page, url) = shape_firecrawl_crawl_page(&item);
        assert_eq!(url, "http://p");
        assert_eq!(page["content"], json!("content"));
        assert_eq!(page["title"], json!("T"));
    }

    #[test]
    fn test_shape_firecrawl_crawl_page_url_fallback() {
        let item = json!({"html": "h", "metadata": {"url": "http://q"}});
        let (page, url) = shape_firecrawl_crawl_page(&item);
        assert_eq!(url, "http://q");
        assert_eq!(page["content"], json!("h"));
    }

    #[test]
    fn test_trim_result() {
        let r = json!({"url": "u", "title": "t", "content": "c", "raw_content": "raw", "metadata": {}});
        let t = trim_result(&r);
        assert_eq!(t["url"], json!("u"));
        assert_eq!(t["error"], Value::Null);
        assert!(t.get("metadata").is_none());
        assert!(t.get("blocked_by_policy").is_none());
    }

    #[test]
    fn test_trim_result_with_policy_block() {
        let r = json!({"url": "u", "error": "blocked", "blocked_by_policy": {"host": "h", "rule": "r", "source": "s"}});
        let t = trim_result(&r);
        assert_eq!(t["error"], json!("blocked"));
        assert_eq!(t["blocked_by_policy"]["host"], json!("h"));
    }

    #[test]
    fn test_normalize_exa_search() {
        let results = vec![
            json!({"url": "u1", "title": "t1", "highlights": ["a", "b"]}),
            json!({"url": "u2", "title": "t2"}),
        ];
        let out = normalize_exa_search(&results);
        let web = out["data"]["web"].as_array().unwrap();
        assert_eq!(web[0]["description"], json!("a b"));
        assert_eq!(web[1]["description"], json!(""));
        assert_eq!(web[1]["position"], json!(2));
    }

    #[test]
    fn test_normalize_parallel_extract_with_excerpt_fallback() {
        let results = vec![json!({"url": "u", "title": "t", "excerpts": ["e1", "e2"]})];
        let errors = vec![json!({"url": "e", "content": "err-content"})];
        let out = normalize_parallel_extract(&results, &errors);
        assert_eq!(out[0]["content"], json!("e1\n\ne2"));
        assert_eq!(out[1]["error"], json!("err-content"));
    }

    #[test]
    fn test_parallel_search_mode_default() {
        unsafe {
            std::env::remove_var("PARALLEL_SEARCH_MODE");
        }
        assert_eq!(parallel_search_mode(), "agentic");
        unsafe {
            std::env::set_var("PARALLEL_SEARCH_MODE", "FAST");
        }
        assert_eq!(parallel_search_mode(), "fast");
        unsafe {
            std::env::set_var("PARALLEL_SEARCH_MODE", "bogus");
        }
        assert_eq!(parallel_search_mode(), "agentic");
        unsafe {
            std::env::remove_var("PARALLEL_SEARCH_MODE");
        }
    }

    #[test]
    fn test_content_processing_decision() {
        assert_eq!(
            content_processing_decision(100, 5000),
            ContentDecision::TooShort
        );
        assert_eq!(
            content_processing_decision(10_000, 5000),
            ContentDecision::SinglePass
        );
        assert_eq!(
            content_processing_decision(600_000, 5000),
            ContentDecision::Chunked
        );
        match content_processing_decision(3_000_000, 5000) {
            ContentDecision::Refuse(msg) => assert!(msg.contains("3.0MB")),
            other => panic!("expected Refuse, got {other:?}"),
        }
    }

    #[test]
    fn test_build_context_str() {
        assert_eq!(build_context_str("", ""), "");
        assert_eq!(build_context_str("T", ""), "Title: T\n\n");
        assert_eq!(build_context_str("", "U"), "Source: U\n\n");
        assert_eq!(build_context_str("T", "U"), "Title: T\nSource: U\n\n");
    }

    #[test]
    fn test_cap_summary_output() {
        let short = "small";
        assert_eq!(cap_summary_output(short), "small");
        let long = "x".repeat(MAX_OUTPUT_SIZE + 100);
        let capped = cap_summary_output(&long);
        assert!(capped.contains("[... summary truncated for context management ...]"));
        assert_eq!(capped.chars().filter(|c| *c == 'x').count(), MAX_OUTPUT_SIZE);
    }

    #[test]
    fn test_summarization_failure_fallback() {
        let long = "y".repeat(MAX_OUTPUT_SIZE + 10);
        let out = summarization_failure_fallback(&long);
        assert!(out.contains("LLM summarization timed out"));
        assert!(out.contains(&comma_group(MAX_OUTPUT_SIZE)));
        let short = "short";
        assert_eq!(summarization_failure_fallback(short), "short");
    }

    #[test]
    fn test_comma_group() {
        assert_eq!(comma_group(5000), "5,000");
        assert_eq!(comma_group(1_234_567), "1,234,567");
        assert_eq!(comma_group(100), "100");
    }

    #[test]
    fn test_url_contains_secret() {
        assert!(url_contains_secret("https://x.com?k=sk-abcd1234"));
        assert!(url_contains_secret("https://x.com?k=%73k-abcd")); // percent-encoded 'sk-'
        assert!(url_contains_secret("https://x.com/ghp_abcdEFGH"));
        assert!(!url_contains_secret("https://example.com/normal/path"));
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("%73k-"), "sk-");
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("nothing"), "nothing");
        assert_eq!(percent_decode("trailing%"), "trailing%");
    }

    #[test]
    fn test_tool_error() {
        let e = tool_error("oops", false);
        let v: Value = serde_json::from_str(&e).unwrap();
        assert_eq!(v["error"], json!("oops"));
        assert!(v.get("success").is_none());
        let e2 = tool_error("interrupt", true);
        let v2: Value = serde_json::from_str(&e2).unwrap();
        assert_eq!(v2["success"], json!(false));
    }

    #[test]
    fn test_get_backend_configured() {
        let env = WebEnv {
            web_config: json!({"backend": "  TAVILY "}),
            ..Default::default()
        };
        assert_eq!(get_backend(&env), "tavily");
    }

    #[test]
    fn test_get_backend_invalid_falls_through_to_default() {
        let env = WebEnv {
            web_config: json!({"backend": "nonsense"}),
            ..Default::default()
        };
        // No env keys set in this test context -> default firecrawl
        unsafe {
            std::env::remove_var("FIRECRAWL_API_KEY");
            std::env::remove_var("FIRECRAWL_API_URL");
            std::env::remove_var("PARALLEL_API_KEY");
            std::env::remove_var("TAVILY_API_KEY");
            std::env::remove_var("EXA_API_KEY");
            std::env::remove_var("SEARXNG_URL");
        }
        assert_eq!(get_backend(&env), "firecrawl");
    }

    #[test]
    fn test_get_capability_backend_override() {
        unsafe {
            std::env::set_var("EXA_API_KEY", "k");
        }
        let env = WebEnv {
            web_config: json!({"backend": "firecrawl", "search_backend": "exa"}),
            ..Default::default()
        };
        assert_eq!(get_search_backend(&env), "exa");
        // extract has no override -> shared backend
        assert_eq!(get_extract_backend(&env), "firecrawl");
        unsafe {
            std::env::remove_var("EXA_API_KEY");
        }
    }

    #[test]
    fn test_web_requires_env_default() {
        let env = WebEnv::default();
        let reqs = web_requires_env(&env);
        assert!(reqs.contains(&"FIRECRAWL_API_KEY".to_string()));
        assert!(!reqs.contains(&"TOOL_GATEWAY_DOMAIN".to_string()));
    }

    #[test]
    fn test_web_requires_env_with_nous() {
        let enabled = || true;
        let env = WebEnv {
            managed_nous_tools_enabled: Some(&enabled),
            ..Default::default()
        };
        let reqs = web_requires_env(&env);
        assert!(reqs.contains(&"TOOL_GATEWAY_DOMAIN".to_string()));
    }

    #[test]
    fn test_schemas() {
        assert_eq!(web_search_schema()["name"], json!("web_search"));
        assert_eq!(web_extract_schema()["name"], json!("web_extract"));
        assert_eq!(
            web_extract_schema()["parameters"]["properties"]["urls"]["maxItems"],
            json!(5)
        );
    }

    #[test]
    fn test_searxng_unsupported_messages() {
        let e = searxng_extract_unsupported();
        assert!(e.contains("search-only"));
        let c = searxng_crawl_unsupported();
        assert!(c.contains("cannot crawl"));
    }

    #[test]
    fn test_resolve_firecrawl_client_config_gateway() {
        unsafe {
            std::env::remove_var("FIRECRAWL_API_KEY");
            std::env::remove_var("FIRECRAWL_API_URL");
        }
        let gw = || Some(("https://gw.example".to_string(), "tok123".to_string()));
        let env = WebEnv {
            resolve_firecrawl_gateway: Some(&gw),
            ..Default::default()
        };
        let (kwargs, cfg) = resolve_firecrawl_client_config(&env).unwrap();
        assert_eq!(kwargs.api_key.as_deref(), Some("tok123"));
        assert_eq!(kwargs.api_url.as_deref(), Some("https://gw.example"));
        assert!(matches!(cfg, FirecrawlConfig::ToolGateway { .. }));
    }

    #[test]
    fn test_resolve_firecrawl_client_config_error() {
        unsafe {
            std::env::remove_var("FIRECRAWL_API_KEY");
            std::env::remove_var("FIRECRAWL_API_URL");
        }
        let env = WebEnv::default();
        let err = resolve_firecrawl_client_config(&env).unwrap_err();
        assert!(err.contains("Web tools are not configured"));
    }

    #[test]
    fn test_compression_metric() {
        let m = compression_metric("u", 1000, 500, Some("model-x"), None);
        assert_eq!(m["compression_ratio"], json!(0.5));
        assert_eq!(m["model_used"], json!("model-x"));
        let m2 = compression_metric("u", 0, 0, None, Some("content_too_short"));
        assert_eq!(m2["compression_ratio"], json!(1.0));
        assert_eq!(m2["reason"], json!("content_too_short"));
        assert_eq!(m2["model_used"], Value::Null);
    }
}
