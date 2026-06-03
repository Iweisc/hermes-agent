//! Models.dev registry integration — primary database for providers and models.
//!
//! Native Rust port of `agent/models_dev.py`.
//!
//! Fetches from <https://models.dev/api.json> — a community-maintained database
//! of 4000+ models across 109+ providers. Provides:
//!
//! - **Provider metadata**: name, base URL, env vars, documentation link
//! - **Model metadata**: context window, max output, cost/M tokens, capabilities
//!   (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
//!   open-weights flag, family grouping, deprecation status
//!
//! Data resolution order (like TypeScript OpenCode):
//!   1. Disk cache (`~/.hermes/models_dev_cache.json`)
//!   2. Network fetch (<https://models.dev/api.json>)
//!
//! The in-memory cache (1 hour TTL) plus disk fallback (short 5-minute effective
//! TTL on load) mirror the Python module.
//!
//! NOTE on parity: `list_provider_models` in Python imports
//! `hermes_cli.models.normalize_provider`; that module is not yet ported to
//! Rust, so this port exposes an optional runtime hook
//! (`set_normalize_provider_hook`). When unset the provider string is used
//! verbatim (matching Python's `normalize_provider(p) or p` when the helper is a
//! no-op).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::{Map, Value};

pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const MODELS_DEV_CACHE_TTL: f64 = 3600.0; // 1 hour in-memory

// ---------------------------------------------------------------------------
// ModelInfo / ProviderInfo / ModelCapabilities structs
// ---------------------------------------------------------------------------

/// Full metadata for a single model from models.dev.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub family: String,
    /// models.dev provider ID (e.g. "anthropic")
    pub provider_id: String,

    // Capabilities
    pub reasoning: bool,
    pub tool_call: bool,
    /// supports image/file attachments (vision)
    pub attachment: bool,
    pub temperature: bool,
    pub structured_output: bool,
    pub open_weights: bool,

    // Modalities ("text", "image", "pdf", ...)
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,

    // Limits
    pub context_window: i64,
    pub max_output: i64,
    pub max_input: Option<i64>,

    // Cost (per million tokens, USD)
    pub cost_input: f64,
    pub cost_output: f64,
    pub cost_cache_read: Option<f64>,
    pub cost_cache_write: Option<f64>,

    // Metadata
    pub knowledge_cutoff: String,
    pub release_date: String,
    /// "alpha", "beta", "deprecated", or ""
    pub status: String,
    /// `true` or `{"field": "reasoning_content"}` — kept as raw JSON.
    pub interleaved: Value,
}

impl Default for ModelInfo {
    fn default() -> Self {
        ModelInfo {
            id: String::new(),
            name: String::new(),
            family: String::new(),
            provider_id: String::new(),
            reasoning: false,
            tool_call: false,
            attachment: false,
            temperature: false,
            structured_output: false,
            open_weights: false,
            input_modalities: Vec::new(),
            output_modalities: Vec::new(),
            context_window: 0,
            max_output: 0,
            max_input: None,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cache_read: None,
            cost_cache_write: None,
            knowledge_cutoff: String::new(),
            release_date: String::new(),
            status: String::new(),
            interleaved: Value::Bool(false),
        }
    }
}

impl ModelInfo {
    pub fn has_cost_data(&self) -> bool {
        self.cost_input > 0.0 || self.cost_output > 0.0
    }

    pub fn supports_vision(&self) -> bool {
        self.attachment || self.input_modalities.iter().any(|m| m == "image")
    }

    pub fn supports_pdf(&self) -> bool {
        self.input_modalities.iter().any(|m| m == "pdf")
    }

    pub fn supports_audio_input(&self) -> bool {
        self.input_modalities.iter().any(|m| m == "audio")
    }

    /// Human-readable cost string, e.g. `"$3.00/M in, $15.00/M out"`.
    pub fn format_cost(&self) -> String {
        if !self.has_cost_data() {
            return "unknown".to_string();
        }
        let mut parts = vec![
            format!("${:.2}/M in", self.cost_input),
            format!("${:.2}/M out", self.cost_output),
        ];
        if let Some(read) = self.cost_cache_read {
            parts.push(format!("cache read ${read:.2}/M"));
        }
        parts.join(", ")
    }

    /// Human-readable capabilities, e.g. `"reasoning, tools, vision, PDF"`.
    pub fn format_capabilities(&self) -> String {
        let mut caps: Vec<&str> = Vec::new();
        if self.reasoning {
            caps.push("reasoning");
        }
        if self.tool_call {
            caps.push("tools");
        }
        if self.supports_vision() {
            caps.push("vision");
        }
        if self.supports_pdf() {
            caps.push("PDF");
        }
        if self.supports_audio_input() {
            caps.push("audio");
        }
        if self.structured_output {
            caps.push("structured output");
        }
        if self.open_weights {
            caps.push("open weights");
        }
        if caps.is_empty() {
            "basic".to_string()
        } else {
            caps.join(", ")
        }
    }
}

/// Full metadata for a provider from models.dev.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProviderInfo {
    /// models.dev provider ID
    pub id: String,
    /// display name
    pub name: String,
    /// env var names for API key
    pub env: Vec<String>,
    /// base URL
    pub api: String,
    /// documentation URL
    pub doc: String,
    pub model_count: usize,
}

/// Structured capability metadata for a model from models.dev.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_reasoning: bool,
    pub context_window: i64,
    pub max_output_tokens: i64,
    pub model_family: String,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        ModelCapabilities {
            supports_tools: true,
            supports_vision: false,
            supports_reasoning: false,
            context_window: 200_000,
            max_output_tokens: 8192,
            model_family: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider ID mapping: Hermes <-> models.dev
// ---------------------------------------------------------------------------

/// Hermes provider names -> models.dev provider IDs.
pub fn provider_to_models_dev() -> &'static [(&'static str, &'static str)] {
    &[
        ("openrouter", "openrouter"),
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("openai-codex", "openai"),
        ("zai", "zai"),
        ("kimi-coding", "kimi-for-coding"),
        ("stepfun", "stepfun"),
        ("kimi-coding-cn", "kimi-for-coding"),
        ("minimax", "minimax"),
        ("minimax-oauth", "minimax"),
        ("minimax-cn", "minimax-cn"),
        ("deepseek", "deepseek"),
        ("alibaba", "alibaba"),
        ("qwen-oauth", "alibaba"),
        ("copilot", "github-copilot"),
        ("ai-gateway", "vercel"),
        ("opencode-zen", "opencode"),
        ("opencode-go", "opencode-go"),
        ("kilocode", "kilo"),
        ("fireworks", "fireworks-ai"),
        ("huggingface", "huggingface"),
        ("gemini", "google"),
        ("google", "google"),
        ("xai", "xai"),
        ("xiaomi", "xiaomi"),
        ("nvidia", "nvidia"),
        ("groq", "groq"),
        ("mistral", "mistral"),
        ("togetherai", "togetherai"),
        ("perplexity", "perplexity"),
        ("cohere", "cohere"),
        ("ollama-cloud", "ollama-cloud"),
    ]
}

/// Look up the models.dev provider ID for a Hermes provider name.
pub fn map_provider_to_models_dev(provider: &str) -> Option<&'static str> {
    provider_to_models_dev()
        .iter()
        .find(|(k, _)| *k == provider)
        .map(|(_, v)| *v)
}

// ---------------------------------------------------------------------------
// Runtime hook for hermes_cli.models.normalize_provider (not yet ported)
// ---------------------------------------------------------------------------

type NormalizeProviderHook = fn(provider: &str) -> Option<String>;

fn normalize_hook() -> &'static Mutex<Option<NormalizeProviderHook>> {
    static H: OnceLock<Mutex<Option<NormalizeProviderHook>>> = OnceLock::new();
    H.get_or_init(|| Mutex::new(None))
}

/// Install the `hermes_cli.models.normalize_provider` hook used by
/// [`list_provider_models`].
pub fn set_normalize_provider_hook(f: NormalizeProviderHook) {
    *normalize_hook().lock().unwrap() = Some(f);
}

fn normalize_provider(provider: &str) -> String {
    if let Some(f) = *normalize_hook().lock().unwrap() {
        if let Some(s) = f(provider) {
            if !s.is_empty() {
                return s;
            }
        }
    }
    provider.to_string()
}

// ---------------------------------------------------------------------------
// In-memory + disk cache
// ---------------------------------------------------------------------------

struct CacheState {
    cache: Map<String, Value>,
    cache_time: f64,
}

fn cache_state() -> &'static Mutex<CacheState> {
    static STATE: OnceLock<Mutex<CacheState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(CacheState {
            cache: Map::new(),
            cache_time: 0.0,
        })
    })
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn get_cache_path() -> PathBuf {
    crate::agent_file_safety::hermes_home_path().join("models_dev_cache.json")
}

fn load_disk_cache() -> Map<String, Value> {
    let path = get_cache_path();
    if !path.exists() {
        return Map::new();
    }
    match std::fs::read_to_string(&path) {
        Ok(contents) => match serde_json::from_str::<Value>(&contents) {
            Ok(Value::Object(map)) => map,
            Ok(_) => Map::new(),
            Err(e) => {
                log::debug!("Failed to parse models.dev disk cache: {e}");
                Map::new()
            }
        },
        Err(e) => {
            log::debug!("Failed to load models.dev disk cache: {e}");
            Map::new()
        }
    }
}

fn save_disk_cache(data: &Map<String, Value>) {
    let path = get_cache_path();
    let result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Match Python's atomic compact write (separators=(",", ":")).
        let serialized = serde_json::to_string(&Value::Object(data.clone()))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serialized.as_bytes())?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    })();
    if let Err(e) = result {
        log::debug!("Failed to save models.dev disk cache: {e}");
    }
}

fn build_blocking_client(timeout_secs: u64) -> reqwest::blocking::Client {
    let mut builder = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs));
    if let Some(bundle) = crate::ag_model_metadata::resolve_requests_verify() {
        if let Ok(bytes) = std::fs::read(&bundle) {
            if let Ok(cert) = reqwest::Certificate::from_pem(&bytes) {
                builder = builder.add_root_certificate(cert);
            }
        }
    }
    builder
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

/// Fetch the models.dev registry. In-memory cache (1hr) + disk fallback.
///
/// Returns the full registry map keyed by provider ID, or an empty map on
/// failure.
pub fn fetch_models_dev(force_refresh: bool) -> Map<String, Value> {
    // Check in-memory cache.
    {
        let state = cache_state().lock().unwrap();
        if !force_refresh
            && !state.cache.is_empty()
            && (now_secs() - state.cache_time) < MODELS_DEV_CACHE_TTL
        {
            return state.cache.clone();
        }
    }

    // Try network fetch.
    let fetched: Option<Map<String, Value>> = (|| {
        let client = build_blocking_client(15);
        let resp = client.get(MODELS_DEV_URL).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let data: Value = resp.json().ok()?;
        match data {
            Value::Object(map) if !map.is_empty() => Some(map),
            _ => None,
        }
    })();

    if let Some(data) = fetched {
        let provider_count = data.len();
        let model_count: usize = data
            .values()
            .filter_map(|p| p.as_object())
            .map(|p| p.get("models").and_then(|m| m.as_object()).map_or(0, |m| m.len()))
            .sum();
        {
            let mut state = cache_state().lock().unwrap();
            state.cache = data.clone();
            state.cache_time = now_secs();
        }
        save_disk_cache(&data);
        log::debug!(
            "Fetched models.dev registry: {provider_count} providers, {model_count} total models"
        );
        return data;
    }

    // Fall back to disk cache — use a short effective TTL (5 min) so we retry the
    // network fetch soon instead of serving stale data for a full hour.
    let mut state = cache_state().lock().unwrap();
    if state.cache.is_empty() {
        let disk = load_disk_cache();
        if !disk.is_empty() {
            let count = disk.len();
            state.cache = disk;
            state.cache_time = now_secs() - MODELS_DEV_CACHE_TTL + 300.0;
            log::debug!("Loaded models.dev from disk cache ({count} providers)");
        }
    }
    state.cache.clone()
}

// ---------------------------------------------------------------------------
// Context lookup
// ---------------------------------------------------------------------------

/// Extract `limit.context` from a models.dev model entry.
///
/// Returns `None` for invalid/zero values (some audio/image models have
/// context=0).
fn extract_context(entry: &Value) -> Option<i64> {
    let entry = entry.as_object()?;
    let limit = entry.get("limit")?.as_object()?;
    let ctx = limit.get("context")?;
    coerce_positive_int(ctx)
}

/// Coerce a JSON number to a positive `i64`. Booleans are rejected (Python
/// `isinstance(ctx, (int, float))` excludes `bool` via the surrounding logic on
/// values that are actually numeric).
fn coerce_positive_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => {
            let v = if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                f as i64
            } else {
                return None;
            };
            if v > 0 {
                Some(v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Look up `limit.context` for a provider+model combo in models.dev.
///
/// Returns the context window in tokens, or `None` if not found. Handles
/// case-insensitive matching and filters out context=0 entries.
pub fn lookup_models_dev_context(provider: &str, model: &str) -> Option<i64> {
    let mdev_provider_id = map_provider_to_models_dev(provider)?;
    let data = fetch_models_dev(false);
    let provider_data = data.get(mdev_provider_id)?.as_object()?;
    let models = provider_data.get("models")?.as_object()?;

    // Exact match.
    if let Some(entry) = models.get(model) {
        if let Some(ctx) = extract_context(entry) {
            return Some(ctx);
        }
    }

    // Case-insensitive match.
    let model_lower = model.to_lowercase();
    for (mid, mdata) in models {
        if mid.to_lowercase() == model_lower {
            if let Some(ctx) = extract_context(mdata) {
                return Some(ctx);
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Provider models helpers
// ---------------------------------------------------------------------------

/// Resolve a Hermes provider ID to its models map from models.dev.
fn get_provider_models(provider: &str) -> Option<Map<String, Value>> {
    let mdev_provider_id = map_provider_to_models_dev(provider)?;
    let data = fetch_models_dev(false);
    let provider_data = data.get(mdev_provider_id)?.as_object()?;
    let models = provider_data.get("models")?.as_object()?;
    Some(models.clone())
}

/// Find a model entry by exact match, then case-insensitive fallback.
fn find_model_entry<'a>(models: &'a Map<String, Value>, model: &str) -> Option<&'a Value> {
    if let Some(entry) = models.get(model) {
        if entry.is_object() {
            return Some(entry);
        }
    }
    let model_lower = model.to_lowercase();
    for (mid, mdata) in models {
        if mid.to_lowercase() == model_lower && mdata.is_object() {
            return Some(mdata);
        }
    }
    None
}

fn entry_bool(entry: &serde_json::Map<String, Value>, key: &str) -> bool {
    entry.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

/// Look up full capability metadata from the models.dev cache.
///
/// Returns `None` if model not found.
pub fn get_model_capabilities(provider: &str, model: &str) -> Option<ModelCapabilities> {
    let models = get_provider_models(provider)?;
    let entry = find_model_entry(&models, model)?;
    let entry = entry.as_object()?;

    let supports_tools = entry_bool(entry, "tool_call");

    // Vision: check both the `attachment` flag and `modalities.input` for "image".
    let input_mods: Vec<String> = entry
        .get("modalities")
        .and_then(|m| m.as_object())
        .and_then(|m| m.get("input"))
        .and_then(|i| i.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let supports_vision = entry_bool(entry, "attachment") || input_mods.iter().any(|m| m == "image");
    let supports_reasoning = entry_bool(entry, "reasoning");

    let limit = entry.get("limit").and_then(|l| l.as_object());
    let context_window = limit
        .and_then(|l| l.get("context"))
        .and_then(coerce_positive_int)
        .unwrap_or(200_000);
    let max_output_tokens = limit
        .and_then(|l| l.get("output"))
        .and_then(coerce_positive_int)
        .unwrap_or(8192);

    let model_family = entry
        .get("family")
        .and_then(|f| f.as_str())
        .unwrap_or("")
        .to_string();

    Some(ModelCapabilities {
        supports_tools,
        supports_vision,
        supports_reasoning,
        context_window,
        max_output_tokens,
        model_family,
    })
}

// ---------------------------------------------------------------------------
// Catalog filtering
// ---------------------------------------------------------------------------

/// Patterns indicating non-agentic or noise models (TTS, embedding, dated
/// preview snapshots, live/streaming-only, image-only).
fn noise_patterns() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|-image\b|-image-preview\b|-customtools\b",
        )
        .unwrap()
    })
}

/// Google models hidden from the Gemini catalogs (low-TPM Gemma + stale slugs).
fn google_hidden_models() -> &'static std::collections::HashSet<&'static str> {
    static SET: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "gemma-4-31b-it",
            "gemma-4-26b-it",
            "gemma-4-26b-a4b-it",
            "gemma-3-1b",
            "gemma-3-1b-it",
            "gemma-3-2b",
            "gemma-3-2b-it",
            "gemma-3-4b",
            "gemma-3-4b-it",
            "gemma-3-12b",
            "gemma-3-12b-it",
            "gemma-3-27b",
            "gemma-3-27b-it",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        .into_iter()
        .collect()
    })
}

fn should_hide_from_provider_catalog(provider: &str, model_id: &str) -> bool {
    let provider_lower = provider.trim().to_lowercase();
    let model_lower = model_id.trim().to_lowercase();
    if (provider_lower == "gemini" || provider_lower == "google")
        && google_hidden_models().contains(model_lower.as_str())
    {
        return true;
    }
    false
}

/// Return all model IDs for a provider from models.dev.
///
/// Returns an empty vec if the provider is unknown or has no data.
pub fn list_provider_models(provider: &str) -> Vec<String> {
    let provider = normalize_provider(provider);
    let models = match get_provider_models(&provider) {
        Some(m) => m,
        None => return Vec::new(),
    };
    models
        .keys()
        .filter(|mid| !should_hide_from_provider_catalog(&provider, mid))
        .cloned()
        .collect()
}

/// Return model IDs suitable for agentic use from models.dev.
///
/// Filters for `tool_call=true` and excludes noise (TTS, embedding, dated
/// preview snapshots, live/streaming, image-only models). Returns an empty vec
/// on any failure.
pub fn list_agentic_models(provider: &str) -> Vec<String> {
    let models = match get_provider_models(provider) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let mut result = Vec::new();
    for (mid, entry) in &models {
        let obj = match entry.as_object() {
            Some(o) => o,
            None => continue,
        };
        if should_hide_from_provider_catalog(provider, mid) {
            continue;
        }
        if !entry_bool(obj, "tool_call") {
            continue;
        }
        if noise_patterns().is_match(mid) {
            continue;
        }
        result.push(mid.clone());
    }
    result
}

// ---------------------------------------------------------------------------
// Rich dataclass constructors
// ---------------------------------------------------------------------------

fn str_or(raw: &serde_json::Map<String, Value>, key: &str, fallback: &str) -> String {
    match raw.get(key).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => fallback.to_string(),
    }
}

/// Coerce a cost JSON value (number or numeric string) to f64, treating
/// null/missing/falsey as 0.0 (mirrors `float(cost.get(k, 0) or 0)`).
fn cost_f64(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Convert a raw models.dev model entry into a [`ModelInfo`].
pub fn parse_model_info(model_id: &str, raw: &Value, provider_id: &str) -> ModelInfo {
    let empty = serde_json::Map::new();
    let raw = raw.as_object().unwrap_or(&empty);

    let limit = raw.get("limit").and_then(|v| v.as_object());
    let cost = raw.get("cost").and_then(|v| v.as_object());
    let modalities = raw.get("modalities").and_then(|v| v.as_object());

    let input_mods: Vec<String> = modalities
        .and_then(|m| m.get("input"))
        .and_then(|i| i.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let output_mods: Vec<String> = modalities
        .and_then(|m| m.get("output"))
        .and_then(|o| o.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let ctx_int = limit
        .and_then(|l| l.get("context"))
        .and_then(coerce_positive_int)
        .unwrap_or(0);
    let out_int = limit
        .and_then(|l| l.get("output"))
        .and_then(coerce_positive_int)
        .unwrap_or(0);
    let inp_int = limit
        .and_then(|l| l.get("input"))
        .and_then(coerce_positive_int);

    // cost_cache_read / cost_cache_write: present-but-null -> None; present non-null -> float.
    let cache_read = match cost.and_then(|c| c.get("cache_read")) {
        Some(Value::Null) | None => None,
        Some(v) => Some(cost_f64(Some(v))),
    };
    let cache_write = match cost.and_then(|c| c.get("cache_write")) {
        Some(Value::Null) | None => None,
        Some(v) => Some(cost_f64(Some(v))),
    };

    ModelInfo {
        id: model_id.to_string(),
        name: str_or(raw, "name", model_id),
        family: str_or(raw, "family", ""),
        provider_id: provider_id.to_string(),
        reasoning: entry_bool(raw, "reasoning"),
        tool_call: entry_bool(raw, "tool_call"),
        attachment: entry_bool(raw, "attachment"),
        temperature: entry_bool(raw, "temperature"),
        structured_output: entry_bool(raw, "structured_output"),
        open_weights: entry_bool(raw, "open_weights"),
        input_modalities: input_mods,
        output_modalities: output_mods,
        context_window: ctx_int,
        max_output: out_int,
        max_input: inp_int,
        cost_input: cost_f64(cost.and_then(|c| c.get("input"))),
        cost_output: cost_f64(cost.and_then(|c| c.get("output"))),
        cost_cache_read: cache_read,
        cost_cache_write: cache_write,
        knowledge_cutoff: str_or(raw, "knowledge", ""),
        release_date: str_or(raw, "release_date", ""),
        status: str_or(raw, "status", ""),
        interleaved: raw.get("interleaved").cloned().unwrap_or(Value::Bool(false)),
    }
}

/// Convert a raw models.dev provider entry into a [`ProviderInfo`].
pub fn parse_provider_info(provider_id: &str, raw: &Value) -> ProviderInfo {
    let empty = serde_json::Map::new();
    let raw = raw.as_object().unwrap_or(&empty);

    let env: Vec<String> = raw
        .get("env")
        .and_then(|e| e.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let model_count = raw
        .get("models")
        .and_then(|m| m.as_object())
        .map_or(0, |m| m.len());

    ProviderInfo {
        id: provider_id.to_string(),
        name: str_or(raw, "name", provider_id),
        env,
        api: str_or(raw, "api", ""),
        doc: str_or(raw, "doc", ""),
        model_count,
    }
}

// ---------------------------------------------------------------------------
// Provider / model level queries
// ---------------------------------------------------------------------------

/// Get full provider metadata from models.dev.
///
/// Accepts either a Hermes provider ID (e.g. "kilocode") or a models.dev ID
/// (e.g. "kilo"). Returns `None` if the provider is not in the catalog.
pub fn get_provider_info(provider_id: &str) -> Option<ProviderInfo> {
    let mdev_id = map_provider_to_models_dev(provider_id).unwrap_or(provider_id);
    let data = fetch_models_dev(false);
    let raw = data.get(mdev_id)?;
    if !raw.is_object() {
        return None;
    }
    Some(parse_provider_info(mdev_id, raw))
}

/// Get full model metadata from models.dev.
///
/// Accepts Hermes or models.dev provider ID. Tries exact match then
/// case-insensitive fallback. Returns `None` if not found.
pub fn get_model_info(provider_id: &str, model_id: &str) -> Option<ModelInfo> {
    let mdev_id = map_provider_to_models_dev(provider_id).unwrap_or(provider_id);
    let data = fetch_models_dev(false);
    let pdata = data.get(mdev_id)?.as_object()?;
    let models = pdata.get("models")?.as_object()?;

    // Exact match.
    if let Some(raw) = models.get(model_id) {
        if raw.is_object() {
            return Some(parse_model_info(model_id, raw, mdev_id));
        }
    }

    // Case-insensitive fallback.
    let model_lower = model_id.to_lowercase();
    for (mid, mdata) in models {
        if mid.to_lowercase() == model_lower && mdata.is_object() {
            return Some(parse_model_info(mid, mdata, mdev_id));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn seed_cache(data: Value) {
        let mut state = cache_state().lock().unwrap();
        state.cache = data.as_object().unwrap().clone();
        state.cache_time = now_secs();
    }

    #[test]
    fn provider_mapping() {
        assert_eq!(map_provider_to_models_dev("kimi-coding"), Some("kimi-for-coding"));
        assert_eq!(map_provider_to_models_dev("gemini"), Some("google"));
        assert_eq!(map_provider_to_models_dev("kilocode"), Some("kilo"));
        assert_eq!(map_provider_to_models_dev("unknown-provider"), None);
    }

    #[test]
    fn model_info_capabilities_and_cost() {
        let mut mi = ModelInfo {
            cost_input: 3.0,
            cost_output: 15.0,
            cost_cache_read: Some(0.3),
            reasoning: true,
            tool_call: true,
            attachment: true,
            structured_output: true,
            open_weights: true,
            input_modalities: vec!["text".into(), "image".into(), "pdf".into(), "audio".into()],
            ..ModelInfo::default()
        };
        assert!(mi.has_cost_data());
        assert!(mi.supports_vision());
        assert!(mi.supports_pdf());
        assert!(mi.supports_audio_input());
        assert_eq!(mi.format_cost(), "$3.00/M in, $15.00/M out, cache read $0.30/M");
        assert_eq!(
            mi.format_capabilities(),
            "reasoning, tools, vision, PDF, audio, structured output, open weights"
        );

        mi.cost_input = 0.0;
        mi.cost_output = 0.0;
        assert!(!mi.has_cost_data());
        assert_eq!(mi.format_cost(), "unknown");

        let basic = ModelInfo::default();
        assert_eq!(basic.format_capabilities(), "basic");
    }

    #[test]
    fn vision_via_modalities_only() {
        let mi = ModelInfo {
            attachment: false,
            input_modalities: vec!["text".into(), "image".into()],
            ..ModelInfo::default()
        };
        assert!(mi.supports_vision());
    }

    #[test]
    fn extract_context_filters_zero_and_missing() {
        assert_eq!(extract_context(&json!({"limit": {"context": 200000}})), Some(200000));
        assert_eq!(extract_context(&json!({"limit": {"context": 0}})), None);
        assert_eq!(extract_context(&json!({"limit": {}})), None);
        assert_eq!(extract_context(&json!({})), None);
        // float
        assert_eq!(extract_context(&json!({"limit": {"context": 128000.0}})), Some(128000));
    }

    #[test]
    fn lookup_context_exact_and_case_insensitive() {
        seed_cache(json!({
            "anthropic": {
                "models": {
                    "claude-opus-4": {"limit": {"context": 200000}},
                    "Claude-Sonnet": {"limit": {"context": 1000000}},
                    "audio-only": {"limit": {"context": 0}}
                }
            }
        }));
        assert_eq!(lookup_models_dev_context("anthropic", "claude-opus-4"), Some(200000));
        // case-insensitive
        assert_eq!(lookup_models_dev_context("anthropic", "claude-sonnet"), Some(1000000));
        // context=0 filtered
        assert_eq!(lookup_models_dev_context("anthropic", "audio-only"), None);
        // unknown model
        assert_eq!(lookup_models_dev_context("anthropic", "nope"), None);
        // unknown provider
        assert_eq!(lookup_models_dev_context("totally-unknown", "x"), None);
    }

    #[test]
    fn capabilities_defaults_and_extraction() {
        seed_cache(json!({
            "anthropic": {
                "models": {
                    "m1": {
                        "tool_call": true,
                        "reasoning": true,
                        "modalities": {"input": ["text", "image"]},
                        "limit": {"context": 500000, "output": 64000},
                        "family": "claude"
                    },
                    "m2": {}
                }
            }
        }));
        let c1 = get_model_capabilities("anthropic", "m1").unwrap();
        assert!(c1.supports_tools);
        assert!(c1.supports_reasoning);
        assert!(c1.supports_vision);
        assert_eq!(c1.context_window, 500000);
        assert_eq!(c1.max_output_tokens, 64000);
        assert_eq!(c1.model_family, "claude");

        // m2: all defaults
        let c2 = get_model_capabilities("anthropic", "m2").unwrap();
        assert!(!c2.supports_tools);
        assert!(!c2.supports_vision);
        assert_eq!(c2.context_window, 200_000);
        assert_eq!(c2.max_output_tokens, 8192);

        assert!(get_model_capabilities("anthropic", "missing").is_none());
    }

    #[test]
    fn noise_pattern_filtering() {
        assert!(noise_patterns().is_match("gemini-tts"));
        assert!(noise_patterns().is_match("text-embedding-3"));
        assert!(noise_patterns().is_match("gemini-live-2.5"));
        assert!(noise_patterns().is_match("gemini-2.0-flash-exp-1206-foo"));
        assert!(noise_patterns().is_match("model-image"));
        assert!(!noise_patterns().is_match("claude-opus-4"));
    }

    #[test]
    fn agentic_models_filtering() {
        seed_cache(json!({
            "google": {
                "models": {
                    "gemini-pro": {"tool_call": true},
                    "gemini-tts": {"tool_call": true},
                    "no-tools": {"tool_call": false},
                    "gemini-1.5-pro": {"tool_call": true}
                }
            }
        }));
        let mut got = list_agentic_models("gemini");
        got.sort();
        // gemini-tts excluded by noise, no-tools excluded, gemini-1.5-pro hidden for google/gemini
        assert_eq!(got, vec!["gemini-pro".to_string()]);
    }

    #[test]
    fn provider_models_hides_google_stale() {
        seed_cache(json!({
            "google": {
                "models": {
                    "gemini-3-pro": {},
                    "gemini-1.5-flash": {},
                    "gemma-3-1b": {}
                }
            }
        }));
        let mut got = list_provider_models("gemini");
        got.sort();
        assert_eq!(got, vec!["gemini-3-pro".to_string()]);
    }

    #[test]
    fn parse_model_info_full() {
        let raw = json!({
            "name": "Claude Opus 4",
            "family": "claude",
            "reasoning": true,
            "tool_call": true,
            "attachment": true,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
            "limit": {"context": 200000, "output": 64000, "input": 190000},
            "cost": {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": null},
            "knowledge": "2025-01",
            "release_date": "2025-05-01",
            "status": "beta",
            "interleaved": true
        });
        let mi = parse_model_info("claude-opus-4", &raw, "anthropic");
        assert_eq!(mi.id, "claude-opus-4");
        assert_eq!(mi.name, "Claude Opus 4");
        assert_eq!(mi.family, "claude");
        assert_eq!(mi.provider_id, "anthropic");
        assert!(mi.reasoning && mi.tool_call && mi.attachment);
        assert_eq!(mi.input_modalities, vec!["text".to_string(), "image".to_string()]);
        assert_eq!(mi.output_modalities, vec!["text".to_string()]);
        assert_eq!(mi.context_window, 200000);
        assert_eq!(mi.max_output, 64000);
        assert_eq!(mi.max_input, Some(190000));
        assert_eq!(mi.cost_input, 3.0);
        assert_eq!(mi.cost_output, 15.0);
        assert_eq!(mi.cost_cache_read, Some(0.3));
        // cache_write present-but-null -> None
        assert_eq!(mi.cost_cache_write, None);
        assert_eq!(mi.knowledge_cutoff, "2025-01");
        assert_eq!(mi.status, "beta");
        assert_eq!(mi.interleaved, Value::Bool(true));
    }

    #[test]
    fn parse_model_info_defaults_name_to_id() {
        let mi = parse_model_info("bare-id", &json!({}), "prov");
        assert_eq!(mi.name, "bare-id");
        assert_eq!(mi.context_window, 0);
        assert_eq!(mi.max_output, 0);
        assert_eq!(mi.max_input, None);
        assert_eq!(mi.cost_input, 0.0);
        assert_eq!(mi.cost_cache_read, None);
        assert_eq!(mi.interleaved, Value::Bool(false));
    }

    #[test]
    fn parse_provider_info_works() {
        let raw = json!({
            "name": "Anthropic",
            "env": ["ANTHROPIC_API_KEY"],
            "api": "https://api.anthropic.com",
            "doc": "https://docs.anthropic.com",
            "models": {"a": {}, "b": {}}
        });
        let pi = parse_provider_info("anthropic", &raw);
        assert_eq!(pi.id, "anthropic");
        assert_eq!(pi.name, "Anthropic");
        assert_eq!(pi.env, vec!["ANTHROPIC_API_KEY".to_string()]);
        assert_eq!(pi.api, "https://api.anthropic.com");
        assert_eq!(pi.model_count, 2);
    }

    #[test]
    fn get_provider_and_model_info_with_mapping() {
        seed_cache(json!({
            "kilo": {
                "name": "Kilo",
                "models": {"foo-model": {"name": "Foo", "tool_call": true}}
            }
        }));
        // Hermes ID "kilocode" -> models.dev "kilo"
        let pi = get_provider_info("kilocode").unwrap();
        assert_eq!(pi.id, "kilo");
        assert_eq!(pi.name, "Kilo");

        let mi = get_model_info("kilocode", "foo-model").unwrap();
        assert_eq!(mi.name, "Foo");
        assert_eq!(mi.provider_id, "kilo");
        // case-insensitive fallback
        let mi2 = get_model_info("kilocode", "FOO-MODEL").unwrap();
        assert_eq!(mi2.id, "foo-model");

        assert!(get_model_info("kilocode", "nope").is_none());
        assert!(get_provider_info("not-a-provider").is_none());
    }
}
