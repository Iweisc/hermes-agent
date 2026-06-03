//! Model metadata, context lengths, and token estimation utilities.
//!
//! Native Rust port of `agent/model_metadata.py`. Pure-ish utility functions
//! with no `AIAgent` dependency — used by context compression and pre-flight
//! context checks.
//!
//! Network calls use `reqwest::blocking`. The various in-memory caches from
//! the Python module (model metadata, endpoint metadata, codex OAuth context)
//! are mirrored with process-global `Mutex`-protected state.
//!
//! NOTE on parity gaps versus the Python original:
//! - `detect_local_server_type`, `query_ollama_num_ctx`, `_query_local_context_length`
//!   used `httpx` in Python; here they use `reqwest::blocking`.
//! - `get_model_context_length` calls into a handful of sibling Python modules
//!   (`bedrock_adapter`, `hermes_cli.models`, `agent.models_dev`,
//!   `hermes_cli.config`, `providers`) which are not yet ported to Rust. Those
//!   branches are represented by optional hook callbacks installable at runtime
//!   (`set_models_dev_hook`, `set_bedrock_hook`, etc.). When a hook is unset the
//!   branch is skipped, matching the Python `except (ImportError|Exception)`
//!   fall-through behaviour.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::Value;

use crate::OPENROUTER_MODELS_URL;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Descending tiers for context length probing when the model is unknown.
pub const CONTEXT_PROBE_TIERS: [i64; 6] = [256_000, 128_000, 64_000, 32_000, 16_000, 8_000];

/// Default context length when no detection method succeeds.
pub const DEFAULT_FALLBACK_CONTEXT: i64 = CONTEXT_PROBE_TIERS[0];

/// Minimum context length required to run Hermes Agent.
pub const MINIMUM_CONTEXT_LENGTH: i64 = 64_000;

const MODEL_CACHE_TTL: f64 = 3600.0;
const ENDPOINT_MODEL_CACHE_TTL: f64 = 300.0;
const CODEX_OAUTH_CONTEXT_CACHE_TTL: f64 = 3600.0;

/// Keys that may hold a context length in arbitrary provider payloads.
pub const CONTEXT_LENGTH_KEYS: [&str; 11] = [
    "context_length",
    "context_window",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
    "ctx_size",
];

/// Keys that may hold a max-completion / output token count.
pub const MAX_COMPLETION_KEYS: [&str; 3] =
    ["max_completion_tokens", "max_output_tokens", "max_tokens"];

const LOCAL_HOSTS: [&str; 4] = ["localhost", "127.0.0.1", "::1", "0.0.0.0"];

const CONTAINER_LOCAL_SUFFIXES: [&str; 3] =
    [".docker.internal", ".containers.internal", ".lima.internal"];

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

/// Provider names that can appear as a `provider:` prefix before a model ID.
fn provider_prefixes() -> &'static std::collections::HashSet<&'static str> {
    static SET: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "openrouter",
            "nous",
            "openai-codex",
            "copilot",
            "copilot-acp",
            "gemini",
            "ollama-cloud",
            "zai",
            "kimi-coding",
            "kimi-coding-cn",
            "stepfun",
            "minimax",
            "minimax-oauth",
            "minimax-cn",
            "anthropic",
            "deepseek",
            "opencode-zen",
            "opencode-go",
            "ai-gateway",
            "kilocode",
            "alibaba",
            "qwen-oauth",
            "xiaomi",
            "arcee",
            "gmi",
            "tencent-tokenhub",
            "custom",
            "local",
            // Common aliases
            "google",
            "google-gemini",
            "google-ai-studio",
            "glm",
            "z-ai",
            "z.ai",
            "zhipu",
            "github",
            "github-copilot",
            "github-models",
            "kimi",
            "moonshot",
            "kimi-cn",
            "moonshot-cn",
            "claude",
            "deep-seek",
            "ollama",
            "opencode",
            "zen",
            "go",
            "vercel",
            "kilo",
            "dashscope",
            "aliyun",
            "qwen",
            "mimo",
            "xiaomi-mimo",
            "tencent",
            "tokenhub",
            "tencent-cloud",
            "tencentmaas",
            "arcee-ai",
            "arceeai",
            "gmi-cloud",
            "gmicloud",
            "xai",
            "x-ai",
            "x.ai",
            "grok",
            "nvidia",
            "nim",
            "nvidia-nim",
            "nemotron",
            "qwen-portal",
        ]
        .into_iter()
        .collect()
    })
}

fn ollama_tag_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)")
            .unwrap()
    })
}

/// Thin fallback defaults — only broad model family patterns. Ordered list of
/// `(key, context_length)`. Lookup uses lowercase substring matching, longest
/// key first; an exact-key lookup is also supported.
pub fn default_context_lengths() -> &'static [(&'static str, i64)] {
    &[
        ("claude-opus-4-7", 1_000_000),
        ("claude-opus-4.7", 1_000_000),
        ("claude-opus-4-6", 1_000_000),
        ("claude-sonnet-4-6", 1_000_000),
        ("claude-opus-4.6", 1_000_000),
        ("claude-sonnet-4.6", 1_000_000),
        ("claude", 200_000),
        ("gpt-5.5", 1_050_000),
        ("gpt-5.4-nano", 400_000),
        ("gpt-5.4-mini", 400_000),
        ("gpt-5.4", 1_050_000),
        ("gpt-5.1-chat", 128_000),
        ("gpt-5", 400_000),
        ("gpt-4.1", 1_047_576),
        ("gpt-4", 128_000),
        ("gemini", 1_048_576),
        ("gemma-4", 256_000),
        ("gemma4", 256_000),
        ("gemma-4-31b", 256_000),
        ("gemma-3", 131_072),
        ("gemma", 8_192),
        ("deepseek-v4-pro", 1_000_000),
        ("deepseek-v4-flash", 1_000_000),
        ("deepseek-chat", 1_000_000),
        ("deepseek-reasoner", 1_000_000),
        ("deepseek", 128_000),
        ("llama", 131_072),
        ("qwen3-coder-plus", 1_000_000),
        ("qwen3-coder", 262_144),
        ("qwen", 131_072),
        ("minimax", 204_800),
        ("glm", 202_752),
        ("grok-code-fast", 256_000),
        ("grok-4-1-fast", 2_000_000),
        ("grok-2-vision", 8_192),
        ("grok-4-fast", 2_000_000),
        ("grok-4.20", 2_000_000),
        ("grok-4", 256_000),
        ("grok-3", 131_072),
        ("grok-2", 131_072),
        ("grok", 131_072),
        ("kimi", 262_144),
        ("hy3-preview", 256_000),
        ("nemotron", 131_072),
        ("trinity", 262_144),
        ("elephant", 262_144),
        ("Qwen/Qwen3.5-397B-A17B", 131_072),
        ("Qwen/Qwen3.5-35B-A3B", 131_072),
        ("deepseek-ai/DeepSeek-V3.2", 65_536),
        ("moonshotai/Kimi-K2.5", 262_144),
        ("moonshotai/Kimi-K2.6", 262_144),
        ("moonshotai/Kimi-K2-Thinking", 262_144),
        ("MiniMaxAI/MiniMax-M2.5", 204_800),
        ("XiaomiMiMo/MiMo-V2-Flash", 262_144),
        ("mimo-v2-pro", 1_048_576),
        ("mimo-v2.5-pro", 1_048_576),
        ("mimo-v2.5", 1_048_576),
        ("mimo-v2-omni", 262_144),
        ("mimo-v2-flash", 262_144),
        ("zai-org/GLM-5", 202_752),
    ]
}

/// Known ChatGPT Codex OAuth context windows. Ordered list — Python sorts by
/// key length descending at lookup time; we preserve insertion and sort at use.
fn codex_oauth_context_fallback() -> &'static [(&'static str, i64)] {
    &[
        ("gpt-5.1-codex-max", 272_000),
        ("gpt-5.1-codex-mini", 272_000),
        ("gpt-5.3-codex", 272_000),
        ("gpt-5.2-codex", 272_000),
        ("gpt-5.4-mini", 272_000),
        ("gpt-5.5", 272_000),
        ("gpt-5.4", 272_000),
        ("gpt-5.2", 272_000),
        ("gpt-5", 272_000),
    ]
}

/// Mapping of known hostnames -> models.dev provider name.
fn url_to_provider() -> &'static [(&'static str, &'static str)] {
    &[
        ("api.openai.com", "openai"),
        ("chatgpt.com", "openai"),
        ("api.anthropic.com", "anthropic"),
        ("api.z.ai", "zai"),
        ("open.bigmodel.cn", "zai"),
        ("api.moonshot.ai", "kimi-coding"),
        ("api.moonshot.cn", "kimi-coding-cn"),
        ("api.kimi.com", "kimi-coding"),
        ("api.stepfun.ai", "stepfun"),
        ("api.stepfun.com", "stepfun"),
        ("api.arcee.ai", "arcee"),
        ("api.minimax", "minimax"),
        ("dashscope.aliyuncs.com", "alibaba"),
        ("dashscope-intl.aliyuncs.com", "alibaba"),
        ("portal.qwen.ai", "qwen-oauth"),
        ("openrouter.ai", "openrouter"),
        ("generativelanguage.googleapis.com", "gemini"),
        ("inference-api.nousresearch.com", "nous"),
        ("api.deepseek.com", "deepseek"),
        ("api.githubcopilot.com", "copilot"),
        ("models.github.ai", "copilot"),
        ("api.fireworks.ai", "fireworks"),
        ("opencode.ai", "opencode-go"),
        ("api.x.ai", "xai"),
        ("integrate.api.nvidia.com", "nvidia"),
        ("api.xiaomimimo.com", "xiaomi"),
        ("xiaomimimo.com", "xiaomi"),
        ("api.gmi-serving.com", "gmi"),
        ("tokenhub.tencentmaas.com", "tencent-tokenhub"),
        ("ollama.com", "ollama-cloud"),
    ]
}

// ---------------------------------------------------------------------------
// Global mutable caches
// ---------------------------------------------------------------------------

type ModelEntry = HashMap<String, Value>;
type ModelCache = HashMap<String, ModelEntry>;

struct GlobalState {
    model_metadata_cache: ModelCache,
    model_metadata_cache_time: f64,
    endpoint_model_metadata_cache: HashMap<String, ModelCache>,
    endpoint_model_metadata_cache_time: HashMap<String, f64>,
    codex_oauth_context_cache: HashMap<String, i64>,
    codex_oauth_context_cache_time: f64,
}

fn global_state() -> &'static Mutex<GlobalState> {
    static STATE: OnceLock<Mutex<GlobalState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(GlobalState {
            model_metadata_cache: HashMap::new(),
            model_metadata_cache_time: 0.0,
            endpoint_model_metadata_cache: HashMap::new(),
            endpoint_model_metadata_cache_time: HashMap::new(),
            codex_oauth_context_cache: HashMap::new(),
            codex_oauth_context_cache_time: 0.0,
        })
    })
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Runtime hooks for not-yet-ported sibling modules
// ---------------------------------------------------------------------------

type ProviderModelHook = fn(provider: &str, model: &str) -> Option<i64>;
type ModelHook = fn(model: &str) -> Option<i64>;
type ModelKeyHook = fn(model: &str, api_key: &str) -> Option<i64>;
type CustomProviderHook = fn(model: &str, base_url: &str) -> Option<i64>;

struct Hooks {
    models_dev: Option<ProviderModelHook>,
    bedrock: Option<ModelHook>,
    copilot: Option<ModelKeyHook>,
    custom_provider: Option<CustomProviderHook>,
}

fn hooks() -> &'static Mutex<Hooks> {
    static H: OnceLock<Mutex<Hooks>> = OnceLock::new();
    H.get_or_init(|| {
        Mutex::new(Hooks {
            models_dev: None,
            bedrock: None,
            copilot: None,
            custom_provider: None,
        })
    })
}

/// Install the models.dev context lookup hook (`agent.models_dev.lookup_models_dev_context`).
pub fn set_models_dev_hook(f: ProviderModelHook) {
    hooks().lock().unwrap().models_dev = Some(f);
}
/// Install the AWS Bedrock context lookup hook (`agent.bedrock_adapter.get_bedrock_context_length`).
pub fn set_bedrock_hook(f: ModelHook) {
    hooks().lock().unwrap().bedrock = Some(f);
}
/// Install the Copilot context lookup hook (`hermes_cli.models.get_copilot_model_context`).
pub fn set_copilot_hook(f: ModelKeyHook) {
    hooks().lock().unwrap().copilot = Some(f);
}
/// Install the custom-provider per-model context hook
/// (`hermes_cli.config.get_custom_provider_context_length`).
pub fn set_custom_provider_hook(f: CustomProviderHook) {
    hooks().lock().unwrap().custom_provider = Some(f);
}

// ---------------------------------------------------------------------------
// SSL verify resolution
// ---------------------------------------------------------------------------

/// Resolve SSL verify setting from env vars. Returns a CA bundle path if one of
/// `HERMES_CA_BUNDLE` / `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` points at a real
/// file, otherwise `None` (defer to default trust store).
pub fn resolve_requests_verify() -> Option<PathBuf> {
    for env_var in ["HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"] {
        if let Ok(val) = std::env::var(env_var) {
            let p = PathBuf::from(&val);
            if !val.is_empty() && p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn build_blocking_client(timeout_secs: u64) -> reqwest::blocking::Client {
    let mut builder = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs));
    if let Some(bundle) = resolve_requests_verify() {
        if let Ok(bytes) = std::fs::read(&bundle) {
            if let Ok(cert) = reqwest::Certificate::from_pem(&bytes) {
                builder = builder.add_root_certificate(cert);
            }
        }
    }
    builder.build().unwrap_or_else(|_| reqwest::blocking::Client::new())
}

// ---------------------------------------------------------------------------
// Provider-prefix stripping
// ---------------------------------------------------------------------------

/// Strip a recognised provider prefix from a model string.
///
/// `"local:my-model"` -> `"my-model"`; `"qwen3.5:27b"` unchanged (Ollama tag).
pub fn strip_provider_prefix(model: &str) -> String {
    if !model.contains(':') || model.starts_with("http") {
        return model.to_string();
    }
    let (prefix, suffix) = model.split_once(':').unwrap();
    let prefix_lower = prefix.trim().to_lowercase();
    if provider_prefixes().contains(prefix_lower.as_str()) {
        // Don't strip if suffix looks like an Ollama tag.
        if ollama_tag_pattern().is_match(suffix.trim()) {
            return model.to_string();
        }
        return suffix.to_string();
    }
    model.to_string()
}

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

fn normalize_base_url(base_url: &str) -> String {
    base_url.trim().trim_end_matches('/').to_string()
}

fn auth_headers(api_key: &str) -> Vec<(&'static str, String)> {
    let token = api_key.trim();
    if token.is_empty() {
        return Vec::new();
    }
    vec![("Authorization", format!("Bearer {token}"))]
}

/// Extract the hostname from a base URL (lowercased). Mirrors
/// `utils.base_url_hostname`.
pub fn base_url_hostname(base_url: &str) -> String {
    let normalized = normalize_base_url(base_url);
    if normalized.is_empty() {
        return String::new();
    }
    let with_scheme = if normalized.contains("://") {
        normalized.clone()
    } else {
        format!("https://{normalized}")
    };
    match url::Url::parse(&with_scheme) {
        Ok(u) => u.host_str().unwrap_or("").to_lowercase(),
        Err(_) => String::new(),
    }
}

/// Return true if the base URL's host equals or is a subdomain of `domain`.
/// Mirrors `utils.base_url_host_matches`.
pub fn base_url_host_matches(base_url: &str, domain: &str) -> bool {
    let host = base_url_hostname(base_url);
    let domain = domain.to_lowercase();
    host == domain || host.ends_with(&format!(".{domain}"))
}

fn is_openrouter_base_url(base_url: &str) -> bool {
    base_url_host_matches(base_url, "openrouter.ai")
}

fn is_custom_endpoint(base_url: &str) -> bool {
    let normalized = normalize_base_url(base_url);
    !normalized.is_empty() && !is_openrouter_base_url(&normalized)
}

/// Infer the models.dev provider name from a base URL.
pub fn infer_provider_from_url(base_url: &str) -> Option<String> {
    let normalized = normalize_base_url(base_url);
    if normalized.is_empty() {
        return None;
    }
    let with_scheme = if normalized.contains("://") {
        normalized.clone()
    } else {
        format!("https://{normalized}")
    };
    let host = match url::Url::parse(&with_scheme) {
        Ok(u) => {
            let netloc = u.host_str().unwrap_or("").to_lowercase();
            if !netloc.is_empty() {
                netloc
            } else {
                u.path().to_lowercase()
            }
        }
        Err(_) => normalized.to_lowercase(),
    };
    for (url_part, provider) in url_to_provider() {
        if host.contains(url_part) {
            return Some(provider.to_string());
        }
    }
    None
}

fn is_known_provider_base_url(base_url: &str) -> bool {
    infer_provider_from_url(base_url).is_some()
}

/// Return true if `base_url` points to a local machine (loopback, container
/// DNS, RFC-1918 private ranges, link-local, or Tailscale CGNAT 100.64.0.0/10).
pub fn is_local_endpoint(base_url: &str) -> bool {
    let normalized = normalize_base_url(base_url);
    if normalized.is_empty() {
        return false;
    }
    let with_scheme = if normalized.contains("://") {
        normalized.clone()
    } else {
        format!("http://{normalized}")
    };
    let host = match url::Url::parse(&with_scheme) {
        Ok(u) => u.host_str().unwrap_or("").to_string(),
        Err(_) => return false,
    };
    if LOCAL_HOSTS.contains(&host.as_str()) {
        return true;
    }
    if CONTAINER_LOCAL_SUFFIXES
        .iter()
        .any(|suffix| host.ends_with(suffix))
    {
        return true;
    }
    // Parse as IP address and check private / loopback / link-local / CGNAT.
    if let Ok(addr) = host.parse::<std::net::IpAddr>() {
        match addr {
            std::net::IpAddr::V4(v4) => {
                if v4.is_private() || v4.is_loopback() || v4.is_link_local() {
                    return true;
                }
                let oct = v4.octets();
                // Tailscale CGNAT 100.64.0.0/10
                if oct[0] == 100 && (64..=127).contains(&oct[1]) {
                    return true;
                }
            }
            std::net::IpAddr::V6(v6) => {
                if v6.is_loopback() {
                    return true;
                }
                // IPv6 link-local fe80::/10
                let seg0 = v6.segments()[0];
                if (seg0 & 0xffc0) == 0xfe80 {
                    return true;
                }
                // Unique local fc00::/7 (treated like RFC-1918 private)
                if (seg0 & 0xfe00) == 0xfc00 {
                    return true;
                }
            }
        }
    }
    // Bare IP that looks like a private range (string-based fallback).
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() == 4 {
        if let (Ok(first), Ok(second)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
            if first == 10 {
                return true;
            }
            if first == 172 && (16..=31).contains(&second) {
                return true;
            }
            if first == 192 && second == 168 {
                return true;
            }
            if first == 100 && (64..=127).contains(&second) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Local server detection
// ---------------------------------------------------------------------------

/// Detect which local server is running at `base_url`.
///
/// Returns one of `"ollama"`, `"lm-studio"`, `"vllm"`, `"llamacpp"`, or `None`.
pub fn detect_local_server_type(base_url: &str, api_key: &str) -> Option<String> {
    let normalized = normalize_base_url(base_url);
    let mut server_url = normalized.clone();
    if server_url.ends_with("/v1") {
        server_url.truncate(server_url.len() - 3);
    }

    let client = build_blocking_client(2);
    let headers = auth_headers(api_key);

    let get = |url: String| -> Option<reqwest::blocking::Response> {
        let mut req = client.get(&url);
        for (k, v) in &headers {
            req = req.header(*k, v);
        }
        req.send().ok()
    };

    // LM Studio: /api/v1/models
    if let Some(r) = get(format!("{server_url}/api/v1/models")) {
        if r.status().as_u16() == 200 {
            return Some("lm-studio".to_string());
        }
    }
    // Ollama: /api/tags with {"models": [...]}
    if let Some(r) = get(format!("{server_url}/api/tags")) {
        if r.status().as_u16() == 200 {
            if let Ok(data) = r.json::<Value>() {
                if data.get("models").is_some() {
                    return Some("ollama".to_string());
                }
            }
        }
    }
    // llama.cpp: /v1/props (or /props on older builds)
    {
        let mut resp = get(format!("{server_url}/v1/props"));
        if resp.as_ref().map(|r| r.status().as_u16()).unwrap_or(0) != 200 {
            resp = get(format!("{server_url}/props"));
        }
        if let Some(r) = resp {
            if r.status().as_u16() == 200 {
                if let Ok(text) = r.text() {
                    if text.contains("default_generation_settings") {
                        return Some("llamacpp".to_string());
                    }
                }
            }
        }
    }
    // vLLM: /version
    if let Some(r) = get(format!("{server_url}/version")) {
        if r.status().as_u16() == 200 {
            if let Ok(data) = r.json::<Value>() {
                if data.get("version").is_some() {
                    return Some("vllm".to_string());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Payload extraction helpers
// ---------------------------------------------------------------------------

/// Depth-first iterate over every dict (object) nested inside `value`, yielding
/// the object itself first then descending. Mirrors `_iter_nested_dicts`.
fn iter_nested_dicts<'a>(value: &'a Value, out: &mut Vec<&'a serde_json::Map<String, Value>>) {
    match value {
        Value::Object(map) => {
            out.push(map);
            for nested in map.values() {
                iter_nested_dicts(nested, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                iter_nested_dicts(item, out);
            }
        }
        _ => {}
    }
}

/// Coerce a JSON value into a "reasonable" integer within `[minimum, maximum]`.
/// Booleans return None (Python `isinstance(value, bool)` guard).
fn coerce_reasonable_int(value: &Value, minimum: i64, maximum: i64) -> Option<i64> {
    let result: i64 = match value {
        Value::Bool(_) => return None,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                f as i64
            } else {
                return None;
            }
        }
        Value::String(s) => {
            let cleaned = s.trim().replace(',', "");
            match cleaned.parse::<i64>() {
                Ok(v) => v,
                Err(_) => {
                    // Python int() of a float-looking string would raise; emulate
                    // by failing here too.
                    return None;
                }
            }
        }
        _ => return None,
    };
    if minimum <= result && result <= maximum {
        Some(result)
    } else {
        None
    }
}

fn extract_first_int(payload: &Value, keys: &[&str]) -> Option<i64> {
    let keyset: std::collections::HashSet<String> =
        keys.iter().map(|k| k.to_lowercase()).collect();
    let mut dicts = Vec::new();
    iter_nested_dicts(payload, &mut dicts);
    for mapping in dicts {
        for (key, value) in mapping {
            if !keyset.contains(&key.to_lowercase()) {
                continue;
            }
            if let Some(c) = coerce_reasonable_int(value, 1024, 10_000_000) {
                return Some(c);
            }
        }
    }
    None
}

/// Extract a context length from an arbitrary payload.
pub fn extract_context_length(payload: &Value) -> Option<i64> {
    extract_first_int(payload, &CONTEXT_LENGTH_KEYS)
}

/// Extract a max-completion / output token count from an arbitrary payload.
pub fn extract_max_completion_tokens(payload: &Value) -> Option<i64> {
    extract_first_int(payload, &MAX_COMPLETION_KEYS)
}

/// Extract a pricing sub-object, normalised to canonical keys
/// (`prompt`, `completion`, `request`, `cache_read`, `cache_write`).
pub fn extract_pricing(payload: &Value) -> serde_json::Map<String, Value> {
    let alias_map: [(&str, &[&str]); 5] = [
        (
            "prompt",
            &["prompt", "input", "input_cost_per_token", "prompt_token_cost"],
        ),
        (
            "completion",
            &[
                "completion",
                "output",
                "output_cost_per_token",
                "completion_token_cost",
            ],
        ),
        ("request", &["request", "request_cost"]),
        (
            "cache_read",
            &[
                "cache_read",
                "cached_prompt",
                "input_cache_read",
                "cache_read_cost_per_token",
            ],
        ),
        (
            "cache_write",
            &[
                "cache_write",
                "cache_creation",
                "input_cache_write",
                "cache_write_cost_per_token",
            ],
        ),
    ];

    let mut dicts = Vec::new();
    iter_nested_dicts(payload, &mut dicts);
    for mapping in dicts {
        // Build a lowercased-key view.
        let normalized: HashMap<String, &Value> = mapping
            .iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();
        let any_alias = alias_map.iter().any(|(_, aliases)| {
            aliases.iter().any(|alias| normalized.contains_key(*alias))
        });
        if !any_alias {
            continue;
        }
        let mut pricing = serde_json::Map::new();
        for (target, aliases) in &alias_map {
            for alias in *aliases {
                if let Some(v) = normalized.get(*alias) {
                    let is_empty = matches!(v, Value::Null)
                        || matches!(v, Value::String(s) if s.is_empty());
                    if !is_empty {
                        pricing.insert(target.to_string(), (*v).clone());
                        break;
                    }
                }
            }
        }
        if !pricing.is_empty() {
            return pricing;
        }
    }
    serde_json::Map::new()
}

fn add_model_aliases(cache: &mut ModelCache, model_id: &str, entry: &ModelEntry) {
    cache.insert(model_id.to_string(), entry.clone());
    if let Some((_, bare)) = model_id.split_once('/') {
        cache.entry(bare.to_string()).or_insert_with(|| entry.clone());
    }
}

// ---------------------------------------------------------------------------
// OpenRouter metadata
// ---------------------------------------------------------------------------

/// Fetch model metadata from OpenRouter (cached for 1 hour).
pub fn fetch_model_metadata(force_refresh: bool) -> ModelCache {
    {
        let state = global_state().lock().unwrap();
        if !force_refresh
            && !state.model_metadata_cache.is_empty()
            && (now_secs() - state.model_metadata_cache_time) < MODEL_CACHE_TTL
        {
            return state.model_metadata_cache.clone();
        }
    }

    let client = build_blocking_client(10);
    let result = (|| -> Option<ModelCache> {
        let resp = client.get(OPENROUTER_MODELS_URL).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let data: Value = resp.json().ok()?;
        let mut cache: ModelCache = HashMap::new();
        if let Some(models) = data.get("data").and_then(|d| d.as_array()) {
            for model in models {
                let model_id = model
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut entry: ModelEntry = HashMap::new();
                entry.insert(
                    "context_length".to_string(),
                    model
                        .get("context_length")
                        .cloned()
                        .unwrap_or_else(|| Value::from(128000)),
                );
                let max_completion = model
                    .get("top_provider")
                    .and_then(|tp| tp.get("max_completion_tokens"))
                    .cloned()
                    .unwrap_or_else(|| Value::from(4096));
                entry.insert("max_completion_tokens".to_string(), max_completion);
                entry.insert(
                    "name".to_string(),
                    model
                        .get("name")
                        .cloned()
                        .unwrap_or_else(|| Value::from(model_id.clone())),
                );
                entry.insert(
                    "pricing".to_string(),
                    model
                        .get("pricing")
                        .cloned()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
                );
                add_model_aliases(&mut cache, &model_id, &entry);
                if let Some(canonical) = model.get("canonical_slug").and_then(|v| v.as_str()) {
                    if !canonical.is_empty() && canonical != model_id {
                        add_model_aliases(&mut cache, canonical, &entry);
                    }
                }
            }
        }
        Some(cache)
    })();

    match result {
        Some(cache) => {
            let mut state = global_state().lock().unwrap();
            state.model_metadata_cache = cache.clone();
            state.model_metadata_cache_time = now_secs();
            cache
        }
        None => {
            log::warn!("Failed to fetch model metadata from OpenRouter");
            global_state().lock().unwrap().model_metadata_cache.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Endpoint metadata
// ---------------------------------------------------------------------------

/// Fetch model metadata from an OpenAI-compatible `/models` endpoint.
/// Results are cached in memory per base URL.
pub fn fetch_endpoint_model_metadata(
    base_url: &str,
    api_key: &str,
    force_refresh: bool,
) -> ModelCache {
    let normalized = normalize_base_url(base_url);
    if normalized.is_empty() || is_openrouter_base_url(&normalized) {
        return HashMap::new();
    }

    if !force_refresh {
        let state = global_state().lock().unwrap();
        if let Some(cached) = state.endpoint_model_metadata_cache.get(&normalized) {
            let cached_at = state
                .endpoint_model_metadata_cache_time
                .get(&normalized)
                .copied()
                .unwrap_or(0.0);
            if (now_secs() - cached_at) < ENDPOINT_MODEL_CACHE_TTL {
                return cached.clone();
            }
        }
    }

    let mut candidates = vec![normalized.clone()];
    let alternate = if normalized.ends_with("/v1") {
        normalized[..normalized.len() - 3]
            .trim_end_matches('/')
            .to_string()
    } else {
        format!("{normalized}/v1")
    };
    if !alternate.is_empty() && !candidates.contains(&alternate) {
        candidates.push(alternate);
    }

    let headers: Vec<(&str, String)> = if api_key.is_empty() {
        Vec::new()
    } else {
        vec![("Authorization", format!("Bearer {api_key}"))]
    };

    let client = build_blocking_client(10);

    // LM Studio native API branch for local endpoints.
    if is_local_endpoint(&normalized) {
        if detect_local_server_type(&normalized, api_key).as_deref() == Some("lm-studio") {
            let server_url = if normalized.ends_with("/v1") {
                normalized[..normalized.len() - 3]
                    .trim_end_matches('/')
                    .to_string()
            } else {
                normalized.clone()
            };
            let url = format!("{}/api/v1/models", server_url.trim_end_matches('/'));
            let mut req = client.get(&url);
            for (k, v) in &headers {
                req = req.header(*k, v);
            }
            if let Ok(resp) = req.send() {
                if resp.status().is_success() {
                    if let Ok(payload) = resp.json::<Value>() {
                        let mut cache: ModelCache = HashMap::new();
                        if let Some(models) = payload.get("models").and_then(|m| m.as_array()) {
                            for model in models {
                                if !model.is_object() {
                                    continue;
                                }
                                let model_id = model
                                    .get("key")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| model.get("id").and_then(|v| v.as_str()));
                                let model_id = match model_id {
                                    Some(s) if !s.is_empty() => s.to_string(),
                                    _ => continue,
                                };
                                let mut entry: ModelEntry = HashMap::new();
                                entry.insert(
                                    "name".to_string(),
                                    model
                                        .get("name")
                                        .cloned()
                                        .unwrap_or_else(|| Value::from(model_id.clone())),
                                );
                                // loaded_instances -> config.context_length
                                if let Some(insts) =
                                    model.get("loaded_instances").and_then(|v| v.as_array())
                                {
                                    for inst in insts {
                                        if let Some(ctx) = inst
                                            .get("config")
                                            .and_then(|c| c.get("context_length"))
                                            .and_then(|c| c.as_i64())
                                        {
                                            if ctx > 0 {
                                                entry.insert(
                                                    "context_length".to_string(),
                                                    Value::from(ctx),
                                                );
                                                break;
                                            }
                                        }
                                    }
                                }
                                if let Some(mc) = extract_max_completion_tokens(model) {
                                    entry.insert(
                                        "max_completion_tokens".to_string(),
                                        Value::from(mc),
                                    );
                                }
                                let pricing = extract_pricing(model);
                                if !pricing.is_empty() {
                                    entry.insert("pricing".to_string(), Value::Object(pricing));
                                }
                                add_model_aliases(&mut cache, &model_id, &entry);
                                if let Some(alt_id) = model.get("id").and_then(|v| v.as_str()) {
                                    if !alt_id.is_empty() && alt_id != model_id {
                                        add_model_aliases(&mut cache, alt_id, &entry);
                                    }
                                }
                            }
                        }
                        let mut state = global_state().lock().unwrap();
                        state
                            .endpoint_model_metadata_cache
                            .insert(normalized.clone(), cache.clone());
                        state
                            .endpoint_model_metadata_cache_time
                            .insert(normalized.clone(), now_secs());
                        return cache;
                    }
                }
            }
        }
    }

    for candidate in &candidates {
        let url = format!("{}/models", candidate.trim_end_matches('/'));
        let mut req = client.get(&url);
        for (k, v) in &headers {
            req = req.header(*k, v);
        }
        let resp = match req.send() {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };
        let payload: Value = match resp.json() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let mut cache: ModelCache = HashMap::new();
        let empty: Vec<Value> = Vec::new();
        let data_arr = payload
            .get("data")
            .and_then(|d| d.as_array())
            .unwrap_or(&empty);
        for model in data_arr {
            if !model.is_object() {
                continue;
            }
            let model_id = match model.get("id").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            let mut entry: ModelEntry = HashMap::new();
            entry.insert(
                "name".to_string(),
                model
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| Value::from(model_id.clone())),
            );
            if let Some(ctx) = extract_context_length(model) {
                entry.insert("context_length".to_string(), Value::from(ctx));
            }
            if let Some(mc) = extract_max_completion_tokens(model) {
                entry.insert("max_completion_tokens".to_string(), Value::from(mc));
            }
            let pricing = extract_pricing(model);
            if !pricing.is_empty() {
                entry.insert("pricing".to_string(), Value::Object(pricing));
            }
            add_model_aliases(&mut cache, &model_id, &entry);
        }

        // llama.cpp: query /props for the actual allocated context.
        let is_llamacpp = data_arr.iter().any(|m| {
            m.get("owned_by").and_then(|v| v.as_str()) == Some("llamacpp")
        });
        if is_llamacpp {
            let base = candidate.trim_end_matches('/').replace("/v1", "");
            let props_client = build_blocking_client(5);
            let fetch_props = |url: String| -> Option<reqwest::blocking::Response> {
                let mut r = props_client.get(&url);
                for (k, v) in &headers {
                    r = r.header(*k, v);
                }
                r.send().ok()
            };
            let mut props_resp = fetch_props(format!("{base}/v1/props"));
            if !props_resp
                .as_ref()
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                props_resp = fetch_props(format!("{base}/props"));
            }
            if let Some(r) = props_resp {
                if r.status().is_success() {
                    if let Ok(props) = r.json::<Value>() {
                        let n_ctx = props
                            .get("default_generation_settings")
                            .and_then(|g| g.get("n_ctx"))
                            .and_then(|v| v.as_i64());
                        let model_alias = props
                            .get("model_alias")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if let Some(n) = n_ctx {
                            if !model_alias.is_empty() {
                                if let Some(e) = cache.get_mut(model_alias) {
                                    e.insert("context_length".to_string(), Value::from(n));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut state = global_state().lock().unwrap();
        state
            .endpoint_model_metadata_cache
            .insert(normalized.clone(), cache.clone());
        state
            .endpoint_model_metadata_cache_time
            .insert(normalized.clone(), now_secs());
        return cache;
    }

    let mut state = global_state().lock().unwrap();
    state
        .endpoint_model_metadata_cache
        .insert(normalized.clone(), HashMap::new());
    state
        .endpoint_model_metadata_cache_time
        .insert(normalized.clone(), now_secs());
    HashMap::new()
}

fn resolve_endpoint_context_length(model: &str, base_url: &str, api_key: &str) -> Option<i64> {
    let endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key, false);
    let mut matched: Option<&ModelEntry> = endpoint_metadata.get(model);
    if matched.is_none() {
        if endpoint_metadata.len() == 1 {
            matched = endpoint_metadata.values().next();
        } else {
            for (key, entry) in &endpoint_metadata {
                if model.contains(key.as_str()) || key.contains(model) {
                    matched = Some(entry);
                    break;
                }
            }
        }
    }
    if let Some(entry) = matched {
        if let Some(ctx) = entry.get("context_length").and_then(|v| v.as_i64()) {
            return Some(ctx);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Persistent context length cache (YAML on disk)
// ---------------------------------------------------------------------------

fn get_context_cache_path() -> PathBuf {
    crate::agent_file_safety::hermes_home_path().join("context_length_cache.yaml")
}

fn load_context_cache() -> HashMap<String, i64> {
    let path = get_context_cache_path();
    if !path.exists() {
        return HashMap::new();
    }
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let data: Value = match serde_yaml::from_str(&contents) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let mut out = HashMap::new();
    if let Some(map) = data.get("context_lengths").and_then(|v| v.as_object()) {
        for (k, v) in map {
            if let Some(i) = v.as_i64() {
                out.insert(k.clone(), i);
            }
        }
    }
    out
}

fn write_context_cache(cache: &HashMap<String, i64>) -> std::io::Result<()> {
    let path = get_context_cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut map = serde_json::Map::new();
    let mut inner = serde_json::Map::new();
    for (k, v) in cache {
        inner.insert(k.clone(), Value::from(*v));
    }
    map.insert("context_lengths".to_string(), Value::Object(inner));
    let yaml = serde_yaml::to_string(&Value::Object(map))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&path, yaml)
}

/// Persist a discovered context length for a model+provider combo.
/// Cache key is `model@base_url`.
pub fn save_context_length(model: &str, base_url: &str, length: i64) {
    let key = format!("{model}@{base_url}");
    let mut cache = load_context_cache();
    if cache.get(&key) == Some(&length) {
        return;
    }
    cache.insert(key.clone(), length);
    match write_context_cache(&cache) {
        Ok(_) => log::info!("Cached context length {key} -> {length} tokens"),
        Err(e) => log::debug!("Failed to save context length cache: {e}"),
    }
}

/// Look up a previously discovered context length for model+provider.
pub fn get_cached_context_length(model: &str, base_url: &str) -> Option<i64> {
    let key = format!("{model}@{base_url}");
    load_context_cache().get(&key).copied()
}

fn invalidate_cached_context_length(model: &str, base_url: &str) {
    let key = format!("{model}@{base_url}");
    let mut cache = load_context_cache();
    if !cache.contains_key(&key) {
        return;
    }
    cache.remove(&key);
    if let Err(e) = write_context_cache(&cache) {
        log::debug!("Failed to invalidate context length cache entry {key}: {e}");
    }
}

// ---------------------------------------------------------------------------
// Probe tiers and error parsing
// ---------------------------------------------------------------------------

/// Return the next lower probe tier below `current_length`, or None.
pub fn get_next_probe_tier(current_length: i64) -> Option<i64> {
    for tier in CONTEXT_PROBE_TIERS {
        if tier < current_length {
            return Some(tier);
        }
    }
    None
}

/// Try to extract the actual context limit from an API error message.
pub fn parse_context_limit_from_error(error_msg: &str) -> Option<i64> {
    let error_lower = error_msg.to_lowercase();
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        [
            r"(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})",
            r"context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})",
            r"(\d{4,})\s*(?:token)?\s*(?:context|limit)",
            r">\s*(\d{4,})\s*(?:max|limit|token)",
            r"(\d{4,})\s*(?:max(?:imum)?)\b",
        ]
        .iter()
        .map(|p| Regex::new(p).unwrap())
        .collect()
    });
    for re in patterns {
        if let Some(caps) = re.captures(&error_lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(limit) = m.as_str().parse::<i64>() {
                    if (1024..=10_000_000).contains(&limit) {
                        return Some(limit);
                    }
                }
            }
        }
    }
    None
}

/// Detect an "output cap too large" error and return how many output tokens are
/// available, or None if it is not such an error.
pub fn parse_available_output_tokens_from_error(error_msg: &str) -> Option<i64> {
    let error_lower = error_msg.to_lowercase();
    let is_output_cap_error = error_lower.contains("max_tokens")
        && (error_lower.contains("available_tokens") || error_lower.contains("available tokens"));
    if !is_output_cap_error {
        return None;
    }
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        [
            r"available_tokens[:\s]+(\d+)",
            r"available\s+tokens[:\s]+(\d+)",
            r"=\s*(\d+)\s*$",
        ]
        .iter()
        .map(|p| Regex::new(p).unwrap())
        .collect()
    });
    for re in patterns {
        if let Some(caps) = re.captures(&error_lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(tokens) = m.as_str().parse::<i64>() {
                    if tokens >= 1 {
                        return Some(tokens);
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Model-id matching
// ---------------------------------------------------------------------------

/// Return true if `candidate_id` (from server) matches `lookup_model` (configured).
/// Supports exact match and `publisher/slug` basename match.
pub fn model_id_matches(candidate_id: &str, lookup_model: &str) -> bool {
    if candidate_id == lookup_model {
        return true;
    }
    if let Some((_, base)) = candidate_id.rsplit_once('/') {
        if base == lookup_model {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Ollama / local server queries
// ---------------------------------------------------------------------------

fn parse_num_ctx_from_params(params: &str) -> Option<i64> {
    if !params.contains("num_ctx") {
        return None;
    }
    for line in params.split('\n') {
        if line.contains("num_ctx") {
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(v) = parts[parts.len() - 1].parse::<i64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn ctx_from_model_info(model_info: &Value) -> Option<i64> {
    if let Some(map) = model_info.as_object() {
        for (key, value) in map {
            if key.contains("context_length") {
                if let Some(i) = value.as_i64() {
                    return Some(i);
                }
                if let Some(f) = value.as_f64() {
                    return Some(f as i64);
                }
            }
        }
    }
    None
}

/// Query an Ollama server for the model's context length (`num_ctx`).
pub fn query_ollama_num_ctx(model: &str, base_url: &str, api_key: &str) -> Option<i64> {
    let bare_model = strip_provider_prefix(model);
    let mut server_url = base_url.trim_end_matches('/').to_string();
    if server_url.ends_with("/v1") {
        server_url.truncate(server_url.len() - 3);
    }

    if detect_local_server_type(base_url, api_key).as_deref() != Some("ollama") {
        return None;
    }

    let client = build_blocking_client(3);
    let headers = auth_headers(api_key);
    let mut req = client
        .post(format!("{server_url}/api/show"))
        .json(&serde_json::json!({ "name": bare_model }));
    for (k, v) in &headers {
        req = req.header(*k, v);
    }
    let resp = req.send().ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    let data: Value = resp.json().ok()?;
    if let Some(params) = data.get("parameters").and_then(|v| v.as_str()) {
        if let Some(v) = parse_num_ctx_from_params(params) {
            return Some(v);
        }
    }
    if let Some(info) = data.get("model_info") {
        if let Some(v) = ctx_from_model_info(info) {
            return Some(v);
        }
    }
    None
}

fn query_local_context_length(model: &str, base_url: &str, api_key: &str) -> Option<i64> {
    let model = strip_provider_prefix(model);
    let mut server_url = base_url.trim_end_matches('/').to_string();
    if server_url.ends_with("/v1") {
        server_url.truncate(server_url.len() - 3);
    }
    let server_type = detect_local_server_type(base_url, api_key);
    let client = build_blocking_client(3);
    let headers = auth_headers(api_key);

    let get = |url: String| -> Option<reqwest::blocking::Response> {
        let mut r = client.get(&url);
        for (k, v) in &headers {
            r = r.header(*k, v);
        }
        r.send().ok()
    };

    if server_type.as_deref() == Some("ollama") {
        let mut req = client
            .post(format!("{server_url}/api/show"))
            .json(&serde_json::json!({ "name": model }));
        for (k, v) in &headers {
            req = req.header(*k, v);
        }
        if let Ok(resp) = req.send() {
            if resp.status().as_u16() == 200 {
                if let Ok(data) = resp.json::<Value>() {
                    if let Some(params) = data.get("parameters").and_then(|v| v.as_str()) {
                        if let Some(v) = parse_num_ctx_from_params(params) {
                            return Some(v);
                        }
                    }
                    if let Some(info) = data.get("model_info") {
                        if let Some(v) = ctx_from_model_info(info) {
                            return Some(v);
                        }
                    }
                }
            }
        }
    }

    if server_type.as_deref() == Some("lm-studio") {
        if let Some(resp) = get(format!("{server_url}/api/v1/models")) {
            if resp.status().as_u16() == 200 {
                if let Ok(data) = resp.json::<Value>() {
                    if let Some(models) = data.get("models").and_then(|m| m.as_array()) {
                        for m in models {
                            let key = m.get("key").and_then(|v| v.as_str()).unwrap_or("");
                            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            if model_id_matches(key, &model) || model_id_matches(id, &model) {
                                if let Some(insts) =
                                    m.get("loaded_instances").and_then(|v| v.as_array())
                                {
                                    for inst in insts {
                                        if let Some(ctx) = inst
                                            .get("config")
                                            .and_then(|c| c.get("context_length"))
                                        {
                                            if let Some(i) = ctx.as_i64() {
                                                return Some(i);
                                            }
                                            if let Some(f) = ctx.as_f64() {
                                                return Some(f as i64);
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // /v1/models/{model}
    if let Some(resp) = get(format!("{server_url}/v1/models/{model}")) {
        if resp.status().as_u16() == 200 {
            if let Ok(data) = resp.json::<Value>() {
                for k in ["max_model_len", "context_length", "max_tokens"] {
                    if let Some(v) = data.get(k) {
                        if let Some(i) = v.as_i64() {
                            return Some(i);
                        }
                        if let Some(f) = v.as_f64() {
                            return Some(f as i64);
                        }
                    }
                }
            }
        }
    }

    // /v1/models list
    if let Some(resp) = get(format!("{server_url}/v1/models")) {
        if resp.status().as_u16() == 200 {
            if let Ok(data) = resp.json::<Value>() {
                if let Some(models) = data.get("data").and_then(|d| d.as_array()) {
                    for m in models {
                        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        if model_id_matches(id, &model) {
                            for k in ["max_model_len", "context_length", "max_tokens"] {
                                if let Some(v) = m.get(k) {
                                    if let Some(i) = v.as_i64() {
                                        return Some(i);
                                    }
                                    if let Some(f) = v.as_f64() {
                                        return Some(f as i64);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Version normalisation / Nous & Anthropic & Codex resolution
// ---------------------------------------------------------------------------

fn normalize_model_version(model: &str) -> String {
    model.replace('.', "-")
}

fn query_anthropic_context_length(model: &str, base_url: &str, api_key: &str) -> Option<i64> {
    if api_key.is_empty() || api_key.starts_with("sk-ant-oat") {
        return None;
    }
    let mut base = base_url.trim_end_matches('/').to_string();
    if base.ends_with("/v1") {
        base.truncate(base.len() - 3);
    }
    let url = format!("{base}/v1/models?limit=1000");
    let client = build_blocking_client(10);
    let resp = client
        .get(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .send()
        .ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    let data: Value = resp.json().ok()?;
    if let Some(models) = data.get("data").and_then(|d| d.as_array()) {
        for m in models {
            if m.get("id").and_then(|v| v.as_str()) == Some(model) {
                if let Some(ctx) = m.get("max_input_tokens").and_then(|v| v.as_i64()) {
                    if ctx > 0 {
                        return Some(ctx);
                    }
                }
            }
        }
    }
    None
}

fn fetch_codex_oauth_context_lengths(access_token: &str) -> HashMap<String, i64> {
    {
        let state = global_state().lock().unwrap();
        if !state.codex_oauth_context_cache.is_empty()
            && now_secs() - state.codex_oauth_context_cache_time < CODEX_OAUTH_CONTEXT_CACHE_TTL
        {
            return state.codex_oauth_context_cache.clone();
        }
    }

    let client = build_blocking_client(10);
    let resp = match client
        .get("https://chatgpt.com/backend-api/codex/models?client_version=1.0.0")
        .header("Authorization", format!("Bearer {access_token}"))
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            log::debug!("Codex /models probe failed: {e}");
            return HashMap::new();
        }
    };
    if resp.status().as_u16() != 200 {
        log::debug!(
            "Codex /models probe returned HTTP {}; falling back to hardcoded defaults",
            resp.status().as_u16()
        );
        return HashMap::new();
    }
    let data: Value = match resp.json() {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let mut result = HashMap::new();
    if let Some(entries) = data.get("models").and_then(|m| m.as_array()) {
        for item in entries {
            if !item.is_object() {
                continue;
            }
            let slug = item.get("slug").and_then(|v| v.as_str());
            let ctx = item.get("context_window").and_then(|v| v.as_i64());
            if let (Some(slug), Some(ctx)) = (slug, ctx) {
                if ctx > 0 {
                    result.insert(slug.trim().to_string(), ctx);
                }
            }
        }
    }
    if !result.is_empty() {
        let mut state = global_state().lock().unwrap();
        state.codex_oauth_context_cache = result.clone();
        state.codex_oauth_context_cache_time = now_secs();
    }
    result
}

fn resolve_codex_oauth_context_length(model: &str, access_token: &str) -> Option<i64> {
    let model_bare = strip_provider_prefix(model).trim().to_string();
    if model_bare.is_empty() {
        return None;
    }

    if !access_token.is_empty() {
        let live = fetch_codex_oauth_context_lengths(access_token);
        if let Some(c) = live.get(&model_bare) {
            return Some(*c);
        }
        let model_lower = model_bare.to_lowercase();
        for (slug, ctx) in &live {
            if slug.to_lowercase() == model_lower {
                return Some(*ctx);
            }
        }
    }

    let model_lower = model_bare.to_lowercase();
    let mut fallback: Vec<(&str, i64)> = codex_oauth_context_fallback().to_vec();
    fallback.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (slug, ctx) in fallback {
        if model_lower.contains(slug) {
            return Some(ctx);
        }
    }
    None
}

fn resolve_nous_context_length(model: &str) -> Option<i64> {
    let metadata = fetch_model_metadata(false);
    if let Some(entry) = metadata.get(model) {
        return entry.get("context_length").and_then(|v| v.as_i64());
    }

    let normalized = normalize_model_version(model).to_lowercase();

    for (or_id, entry) in &metadata {
        let bare = or_id.split_once('/').map(|(_, b)| b).unwrap_or(or_id);
        if bare.to_lowercase() == model.to_lowercase()
            || normalize_model_version(bare).to_lowercase() == normalized
        {
            return entry.get("context_length").and_then(|v| v.as_i64());
        }
    }

    let model_lower = model.to_lowercase();
    for (or_id, entry) in &metadata {
        let bare = or_id.split_once('/').map(|(_, b)| b).unwrap_or(or_id);
        for (candidate, query) in [
            (bare.to_lowercase(), model_lower.clone()),
            (normalize_model_version(bare).to_lowercase(), normalized.clone()),
        ] {
            if candidate.starts_with(&query) {
                let boundary = candidate.len() == query.len()
                    || candidate[query.len()..]
                        .chars()
                        .next()
                        .map(|c| c == '-' || c == ':' || c == '.')
                        .unwrap_or(false);
                if boundary {
                    return entry.get("context_length").and_then(|v| v.as_i64());
                }
            }
        }
    }
    None
}

fn default_context_lengths_exact(model_lower: &str) -> Option<i64> {
    for (key, length) in default_context_lengths() {
        if key.to_lowercase() == model_lower {
            return Some(*length);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Main resolution entry point
// ---------------------------------------------------------------------------

/// Get the context length for a model. See the module/Python docstring for the
/// full resolution order (config override -> cache -> bedrock -> endpoint ->
/// local -> anthropic -> provider-aware -> OpenRouter -> hardcoded -> fallback).
#[allow(clippy::too_many_arguments)]
pub fn get_model_context_length(
    model: &str,
    base_url: &str,
    api_key: &str,
    config_context_length: Option<i64>,
    provider: &str,
    custom_providers_present: bool,
) -> i64 {
    // 0. Explicit config override.
    if let Some(c) = config_context_length {
        if c > 0 {
            return c;
        }
    }

    // 0b. custom_providers per-model override.
    if custom_providers_present && !base_url.is_empty() && !model.is_empty() {
        if let Some(f) = hooks().lock().unwrap().custom_provider {
            if let Some(cp_ctx) = f(model, base_url) {
                if cp_ctx != 0 {
                    return cp_ctx;
                }
            }
        }
    }

    // Normalise provider-prefixed model names.
    let model = strip_provider_prefix(model);

    // 1. Persistent cache (model+provider). LM Studio excluded.
    if !base_url.is_empty() && provider != "lmstudio" {
        if let Some(cached) = get_cached_context_length(&model, base_url) {
            if provider == "openai-codex" && cached >= 400_000 {
                log::info!(
                    "Dropping stale Codex cache entry {model}@{base_url} -> {cached} (pre-fix value); \
                     re-resolving via live /models probe"
                );
                invalidate_cached_context_length(&model, base_url);
            } else {
                return cached;
            }
        }
    }

    // 1b. AWS Bedrock static table.
    let is_bedrock = provider == "bedrock"
        || (!base_url.is_empty()
            && base_url_hostname(base_url).starts_with("bedrock-runtime.")
            && base_url_host_matches(base_url, "amazonaws.com"));
    if is_bedrock {
        if let Some(f) = hooks().lock().unwrap().bedrock {
            if let Some(ctx) = f(&model) {
                return ctx;
            }
        }
        // If no hook, fall through (Python: ImportError -> generic resolution).
    }

    // 2. Active endpoint metadata for truly custom/unknown endpoints.
    if is_custom_endpoint(base_url) && !is_known_provider_base_url(base_url) {
        if let Some(ctx) = resolve_endpoint_context_length(&model, base_url, api_key) {
            return ctx;
        }
        if !is_known_provider_base_url(base_url) {
            // 3. Try querying local server directly.
            if is_local_endpoint(base_url) {
                if let Some(local_ctx) = query_local_context_length(&model, base_url, api_key) {
                    if local_ctx > 0 {
                        if provider != "lmstudio" {
                            save_context_length(&model, base_url, local_ctx);
                        }
                        return local_ctx;
                    }
                }
            }
            log::info!(
                "Could not detect context length for model {model:?} at {base_url} — \
                 defaulting to {DEFAULT_FALLBACK_CONTEXT} tokens (probe-down). \
                 Set model.context_length in config.yaml to override."
            );
            return DEFAULT_FALLBACK_CONTEXT;
        }
    }

    // 4. Anthropic /v1/models API.
    if provider == "anthropic"
        || (!base_url.is_empty() && base_url_hostname(base_url) == "api.anthropic.com")
    {
        let effective_url = if base_url.is_empty() {
            "https://api.anthropic.com"
        } else {
            base_url
        };
        if let Some(ctx) = query_anthropic_context_length(&model, effective_url, api_key) {
            if ctx != 0 {
                return ctx;
            }
        }
    }

    // 5. Provider-aware lookups.
    let mut effective_provider = provider.to_string();
    if effective_provider.is_empty()
        || effective_provider == "openrouter"
        || effective_provider == "custom"
    {
        if !base_url.is_empty() {
            if let Some(inferred) = infer_provider_from_url(base_url) {
                effective_provider = inferred;
            }
        }
    }

    // 5a. Copilot live /models API.
    if matches!(
        effective_provider.as_str(),
        "copilot" | "copilot-acp" | "github-copilot"
    ) {
        if let Some(f) = hooks().lock().unwrap().copilot {
            if let Some(ctx) = f(&model, api_key) {
                if ctx != 0 {
                    return ctx;
                }
            }
        }
    }

    if effective_provider == "nous" {
        if let Some(ctx) = resolve_nous_context_length(&model) {
            if ctx != 0 {
                return ctx;
            }
        }
    }
    if effective_provider == "openai-codex" {
        if let Some(codex_ctx) = resolve_codex_oauth_context_length(&model, api_key) {
            if codex_ctx != 0 {
                if !base_url.is_empty() {
                    save_context_length(&model, base_url, codex_ctx);
                }
                return codex_ctx;
            }
        }
    }
    if effective_provider == "gmi" && !base_url.is_empty() {
        if let Some(ctx) = resolve_endpoint_context_length(&model, base_url, api_key) {
            return ctx;
        }
    }
    if !effective_provider.is_empty() {
        if let Some(f) = hooks().lock().unwrap().models_dev {
            if let Some(ctx) = f(&effective_provider, &model) {
                if ctx != 0 {
                    return ctx;
                }
            }
        }
    }

    let model_lower = model.to_lowercase();
    if let Some(exact) = default_context_lengths_exact(&model_lower) {
        if exact != 0 {
            return exact;
        }
    }

    // 6. OpenRouter live API metadata.
    let metadata = fetch_model_metadata(false);
    if let Some(entry) = metadata.get(&model) {
        return entry
            .get("context_length")
            .and_then(|v| v.as_i64())
            .unwrap_or(DEFAULT_FALLBACK_CONTEXT);
    }

    // 8. Hardcoded defaults (fuzzy substring; longest key first).
    let mut defaults: Vec<(&str, i64)> = default_context_lengths().to_vec();
    defaults.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (default_model, length) in defaults {
        if model_lower.contains(&default_model.to_lowercase()) {
            return length;
        }
    }

    // 9. Query local server as last resort.
    if !base_url.is_empty() && is_local_endpoint(base_url) {
        if let Some(local_ctx) = query_local_context_length(&model, base_url, api_key) {
            if local_ctx > 0 {
                if provider != "lmstudio" {
                    save_context_length(&model, base_url, local_ctx);
                }
                return local_ctx;
            }
        }
    }

    // 10. Default fallback — 256K.
    DEFAULT_FALLBACK_CONTEXT
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token estimate (~4 chars/token) using ceiling division.
pub fn estimate_tokens_rough(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    ((text.chars().count() as i64) + 3) / 4
}

/// Rough token estimate for a list of messages, where each message is rendered
/// via its `str(...)` form. The caller must pass the already-stringified
/// representation of each message (mirroring Python's `str(msg)`).
pub fn estimate_messages_tokens_rough(message_reprs: &[String]) -> i64 {
    let total_chars: i64 = message_reprs.iter().map(|m| m.chars().count() as i64).sum();
    (total_chars + 3) / 4
}

/// Rough token estimate for a full chat-completions request. `tools_repr`
/// should be the `str(tools)` rendering (or empty if no tools).
pub fn estimate_request_tokens_rough(
    message_reprs: &[String],
    system_prompt: &str,
    tools_repr: &str,
) -> i64 {
    let mut total_chars: i64 = 0;
    if !system_prompt.is_empty() {
        total_chars += system_prompt.chars().count() as i64;
    }
    total_chars += message_reprs
        .iter()
        .map(|m| m.chars().count() as i64)
        .sum::<i64>();
    if !tools_repr.is_empty() {
        total_chars += tools_repr.chars().count() as i64;
    }
    (total_chars + 3) / 4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_provider_prefix_strips_known_prefix() {
        assert_eq!(strip_provider_prefix("local:my-model"), "my-model");
        assert_eq!(strip_provider_prefix("openrouter:gpt-4"), "gpt-4");
    }

    #[test]
    fn strip_provider_prefix_preserves_ollama_tags() {
        assert_eq!(strip_provider_prefix("qwen3.5:27b"), "qwen3.5:27b");
        assert_eq!(strip_provider_prefix("qwen:0.5b"), "qwen:0.5b");
        assert_eq!(strip_provider_prefix("deepseek:latest"), "deepseek:latest");
    }

    #[test]
    fn strip_provider_prefix_no_colon_or_http() {
        assert_eq!(strip_provider_prefix("plain-model"), "plain-model");
        assert_eq!(
            strip_provider_prefix("http://example.com:8080"),
            "http://example.com:8080"
        );
    }

    #[test]
    fn is_local_endpoint_loopback_and_private() {
        assert!(is_local_endpoint("http://localhost:11434"));
        assert!(is_local_endpoint("http://127.0.0.1:8080"));
        assert!(is_local_endpoint("http://192.168.1.10"));
        assert!(is_local_endpoint("http://10.0.0.5"));
        assert!(is_local_endpoint("http://172.16.0.1"));
        // Tailscale CGNAT
        assert!(is_local_endpoint("http://100.77.243.5:11434"));
        assert!(is_local_endpoint("http://host.docker.internal:1234"));
        // Public should be false
        assert!(!is_local_endpoint("https://api.openai.com"));
        assert!(!is_local_endpoint(""));
        // 100.x outside CGNAT range
        assert!(!is_local_endpoint("http://100.200.0.1"));
    }

    #[test]
    fn base_url_helpers() {
        assert_eq!(base_url_hostname("https://api.openai.com/v1"), "api.openai.com");
        assert!(base_url_host_matches("https://openrouter.ai/api/v1", "openrouter.ai"));
        assert!(base_url_host_matches("https://x.openrouter.ai", "openrouter.ai"));
        assert!(!base_url_host_matches("https://openrouter.ai.evil.com", "openrouter.ai"));
    }

    #[test]
    fn infer_provider() {
        assert_eq!(infer_provider_from_url("https://api.openai.com/v1").as_deref(), Some("openai"));
        assert_eq!(
            infer_provider_from_url("https://dashscope.aliyuncs.com/compatible").as_deref(),
            Some("alibaba")
        );
        assert_eq!(infer_provider_from_url("https://example.invalid"), None);
    }

    #[test]
    fn next_probe_tier() {
        assert_eq!(get_next_probe_tier(300_000), Some(256_000));
        assert_eq!(get_next_probe_tier(256_000), Some(128_000));
        assert_eq!(get_next_probe_tier(8_000), None);
        assert_eq!(get_next_probe_tier(1_000), None);
    }

    #[test]
    fn parse_context_limit_variants() {
        assert_eq!(
            parse_context_limit_from_error("maximum context length is 32768 tokens"),
            Some(32768)
        );
        assert_eq!(
            parse_context_limit_from_error("context_length_exceeded: 131072"),
            Some(131072)
        );
        assert_eq!(
            parse_context_limit_from_error("Maximum context size 32768 exceeded"),
            Some(32768)
        );
        assert_eq!(
            parse_context_limit_from_error("250000 tokens > 200000 maximum"),
            // First matching pattern wins; both numbers are >= 1024.
            Some(250000)
        );
        assert_eq!(parse_context_limit_from_error("no numbers here"), None);
        // Too small to be a context length
        assert_eq!(parse_context_limit_from_error("limit is 512"), None);
    }

    #[test]
    fn parse_available_output_tokens() {
        let msg = "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000 = available_tokens: 10000";
        assert_eq!(parse_available_output_tokens_from_error(msg), Some(10000));
        // Not an output-cap error (prompt-too-long)
        assert_eq!(
            parse_available_output_tokens_from_error("prompt is too long: 250000 tokens"),
            None
        );
    }

    #[test]
    fn model_id_matching() {
        assert!(model_id_matches("nemotron-49b", "nemotron-49b"));
        assert!(model_id_matches("nvidia/nemotron-49b", "nemotron-49b"));
        assert!(!model_id_matches("nvidia/nemotron-49b", "nemotron"));
        assert!(!model_id_matches("other", "nemotron-49b"));
    }

    #[test]
    fn extract_context_length_nested() {
        let payload = json!({
            "config": { "max_model_len": 65536 },
            "other": "ignore"
        });
        assert_eq!(extract_context_length(&payload), Some(65536));

        let with_comma = json!({ "context_length": "131,072" });
        assert_eq!(extract_context_length(&with_comma), Some(131072));

        // boolean must be ignored
        let with_bool = json!({ "context_length": true, "n_ctx": 8192 });
        assert_eq!(extract_context_length(&with_bool), Some(8192));

        // below minimum -> skipped
        let too_small = json!({ "context_length": 512 });
        assert_eq!(extract_context_length(&too_small), None);
    }

    #[test]
    fn extract_pricing_aliases() {
        let payload = json!({
            "pricing": {
                "input_cost_per_token": "0.000001",
                "output_cost_per_token": "0.000002"
            }
        });
        let pricing = extract_pricing(&payload);
        assert_eq!(pricing.get("prompt").and_then(|v| v.as_str()), Some("0.000001"));
        assert_eq!(pricing.get("completion").and_then(|v| v.as_str()), Some("0.000002"));

        let none = json!({ "unrelated": 1 });
        assert!(extract_pricing(&none).is_empty());
    }

    #[test]
    fn normalize_version() {
        assert_eq!(normalize_model_version("claude-opus-4.6"), "claude-opus-4-6");
    }

    #[test]
    fn estimate_tokens() {
        assert_eq!(estimate_tokens_rough(""), 0);
        assert_eq!(estimate_tokens_rough("a"), 1); // (1+3)/4 = 1
        assert_eq!(estimate_tokens_rough("abcd"), 1);
        assert_eq!(estimate_tokens_rough("abcde"), 2); // (5+3)/4 = 2
    }

    #[test]
    fn estimate_messages_and_request() {
        let msgs = vec!["hello".to_string(), "world!".to_string()]; // 5 + 6 = 11 chars
        assert_eq!(estimate_messages_tokens_rough(&msgs), (11 + 3) / 4);

        let req = estimate_request_tokens_rough(&msgs, "sys", "[tool]");
        // 11 + 3 + 6 = 20 chars
        assert_eq!(req, (20 + 3) / 4);
    }

    #[test]
    fn default_context_exact_and_fuzzy() {
        // exact
        assert_eq!(default_context_lengths_exact("gpt-4"), Some(128_000));
        assert_eq!(default_context_lengths_exact("unknown-model"), None);
        // longest-first fuzzy ordering: claude-sonnet-4-6 should beat claude
        let mut defaults: Vec<(&str, i64)> = default_context_lengths().to_vec();
        defaults.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let model = "anthropic/claude-sonnet-4-6";
        let mut found = None;
        for (k, v) in defaults {
            if model.contains(&k.to_lowercase()) {
                found = Some(v);
                break;
            }
        }
        assert_eq!(found, Some(1_000_000));
    }

    #[test]
    fn config_override_short_circuits() {
        // config_context_length wins regardless of everything else (no network).
        let ctx = get_model_context_length("anything", "", "", Some(123_456), "", false);
        assert_eq!(ctx, 123_456);
    }

    #[test]
    fn coerce_int_bounds() {
        assert_eq!(coerce_reasonable_int(&json!(2048), 1024, 10_000_000), Some(2048));
        assert_eq!(coerce_reasonable_int(&json!(10), 1024, 10_000_000), None);
        assert_eq!(coerce_reasonable_int(&json!(true), 1024, 10_000_000), None);
        assert_eq!(coerce_reasonable_int(&json!("  4096 "), 1024, 10_000_000), Some(4096));
    }
}
