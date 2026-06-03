//! Helpers for Nous subscription managed-tool capabilities.
//!
//! Native Rust port of `hermes_cli/nous_subscription.py`.
//!
//! ## Cross-cutting dependencies
//!
//! The Python original reaches into several subsystems that are either not yet
//! ported with matching signatures, or whose behaviour depends on process /
//! filesystem / network state. To stay faithful *and* testable, those
//! cross-cutting decisions are supplied to the pure logic via a
//! [`SubscriptionEnv`] struct of closures, each with a sensible native default:
//!
//! * `get_env_value` — mirrors `hermes_cli.config.get_env_value`. Default reads
//!   the process environment, returning `None` for missing/empty values
//!   (Python's `get_env_value` returns `None` when unset).
//! * `resolve_toolset` — mirrors `toolsets.resolve_toolset`. Default delegates
//!   to [`crate::mod_toolsets::resolve_toolset`].
//! * `managed_nous_tools_enabled` — mirrors
//!   `tools.tool_backend_helpers.managed_nous_tools_enabled` (not yet ported).
//!   Default: `false`.
//! * `nous_logged_in` — mirrors
//!   `bool(get_nous_auth_status().get("logged_in"))`. Default: reads
//!   [`crate::auth`] is not wired here, so default is `false`.
//! * `is_managed_tool_gateway_ready` — mirrors
//!   `tools.managed_tool_gateway.is_managed_tool_gateway_ready`. Default
//!   delegates to [`crate::tool_managed_tool_gateway::is_managed_tool_gateway_ready`]
//!   with default hooks.
//! * `has_agent_browser` — mirrors `_has_agent_browser`. Default probes `PATH`
//!   and the local `node_modules/.bin/agent-browser`.
//! * `default_platform_toolset` — mirrors `_default_platform_toolset`. Default
//!   uses the `hermes-cli` / `hermes-<platform>` fallback (the platform
//!   registry is consulted by the Python original but the fallback matches when
//!   no registry entry exists).
//!
//! All config is represented as [`serde_json::Value`] objects, matching the
//! Python `Dict[str, object]` shape produced by the YAML loader.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::mod_utils::{is_truthy_value, TruthyInput};

// ---------------------------------------------------------------------------
// Feature-state data structures
// ---------------------------------------------------------------------------

/// Faithful mirror of the frozen Python dataclass `NousFeatureState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NousFeatureState {
    pub key: String,
    pub label: String,
    pub included_by_default: bool,
    pub available: bool,
    pub active: bool,
    pub managed_by_nous: bool,
    pub direct_override: bool,
    pub toolset_enabled: bool,
    pub current_provider: String,
    pub explicit_configured: bool,
}

/// Faithful mirror of the frozen Python dataclass `NousSubscriptionFeatures`.
///
/// The `features` map is keyed by feature key (`"web"`, `"image_gen"`,
/// `"tts"`, `"browser"`, `"modal"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NousSubscriptionFeatures {
    pub subscribed: bool,
    pub nous_auth_present: bool,
    pub provider_is_nous: bool,
    pub features: std::collections::HashMap<String, NousFeatureState>,
}

impl NousSubscriptionFeatures {
    /// Mirrors the `web` property.
    pub fn web(&self) -> &NousFeatureState {
        &self.features["web"]
    }
    /// Mirrors the `image_gen` property.
    pub fn image_gen(&self) -> &NousFeatureState {
        &self.features["image_gen"]
    }
    /// Mirrors the `tts` property.
    pub fn tts(&self) -> &NousFeatureState {
        &self.features["tts"]
    }
    /// Mirrors the `browser` property.
    pub fn browser(&self) -> &NousFeatureState {
        &self.features["browser"]
    }
    /// Mirrors the `modal` property.
    pub fn modal(&self) -> &NousFeatureState {
        &self.features["modal"]
    }

    /// Mirrors `items()`: yields the feature states in the canonical order.
    pub fn items(&self) -> Vec<&NousFeatureState> {
        ["web", "image_gen", "tts", "browser", "modal"]
            .iter()
            .map(|k| &self.features[*k])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Cross-cutting environment hooks
// ---------------------------------------------------------------------------

/// Pluggable cross-cutting decisions used by [`get_nous_subscription_features`]
/// and friends. Each closure stands in for an un-ported / state-dependent
/// Python helper. Use [`SubscriptionEnv::default`] for native behaviour.
pub struct SubscriptionEnv<'a> {
    /// `hermes_cli.config.get_env_value`. Returns `None` for missing/empty.
    pub get_env_value: Box<dyn Fn(&str) -> Option<String> + 'a>,
    /// `toolsets.resolve_toolset`.
    pub resolve_toolset: Box<dyn Fn(&str) -> Vec<String> + 'a>,
    /// `tools.tool_backend_helpers.managed_nous_tools_enabled`.
    pub managed_nous_tools_enabled: Box<dyn Fn() -> bool + 'a>,
    /// `bool(get_nous_auth_status().get("logged_in"))`.
    pub nous_logged_in: Box<dyn Fn() -> bool + 'a>,
    /// `tools.managed_tool_gateway.is_managed_tool_gateway_ready`.
    pub is_managed_tool_gateway_ready: Box<dyn Fn(&str) -> bool + 'a>,
    /// `_has_agent_browser`.
    pub has_agent_browser: Box<dyn Fn() -> bool + 'a>,
    /// `_default_platform_toolset`.
    pub default_platform_toolset: Box<dyn Fn(&str) -> String + 'a>,
}

impl<'a> Default for SubscriptionEnv<'a> {
    fn default() -> Self {
        Self {
            get_env_value: Box::new(default_get_env_value),
            resolve_toolset: Box::new(|name| crate::mod_toolsets::resolve_toolset(name)),
            managed_nous_tools_enabled: Box::new(|| false),
            nous_logged_in: Box::new(|| false),
            is_managed_tool_gateway_ready: Box::new(|vendor| {
                crate::tool_managed_tool_gateway::is_managed_tool_gateway_ready(
                    vendor,
                    &crate::tool_managed_tool_gateway::GatewayResolveHooks::default(),
                )
            }),
            has_agent_browser: Box::new(default_has_agent_browser),
            default_platform_toolset: Box::new(|platform| default_platform_toolset(platform)),
        }
    }
}

/// Default `get_env_value`: read the process env, treating missing or empty as
/// `None` (Python's `get_env_value` returns `None` for unset; an empty string
/// is still present, but downstream truthiness checks treat `""` as falsy, so
/// we surface the empty string faithfully when present).
fn default_get_env_value(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}

/// Default `_default_platform_toolset` fallback (registry not consulted here).
///
/// Mirrors the Python fallback: `"hermes-cli"` for the `cli` platform, else
/// `"hermes-<platform>"`.
pub fn default_platform_toolset(platform: &str) -> String {
    if platform == "cli" {
        "hermes-cli".to_string()
    } else {
        format!("hermes-{platform}")
    }
}

/// Default `_has_agent_browser`: true when `agent-browser` is on `PATH` or a
/// local `node_modules/.bin/agent-browser` exists relative to the executable's
/// project root.
pub fn default_has_agent_browser() -> bool {
    if which_agent_browser() {
        return true;
    }
    // Best-effort local-bin probe relative to the current working directory.
    // The Python original probes a path relative to the source file's
    // grandparent; without that anchor we fall back to CWD.
    if let Ok(cwd) = std::env::current_dir() {
        let local = cwd.join("node_modules").join(".bin").join("agent-browser");
        if local.exists() {
            return true;
        }
    }
    false
}

fn which_agent_browser() -> bool {
    let path = match std::env::var_os("PATH") {
        Some(p) => p,
        None => return false,
    };
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("agent-browser");
        if candidate.exists() {
            return true;
        }
        #[cfg(windows)]
        {
            if dir.join("agent-browser.exe").exists()
                || dir.join("agent-browser.cmd").exists()
            {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Small JSON helpers (mirror Python dict-access idioms)
// ---------------------------------------------------------------------------

/// Return the config as an object map, treating non-objects as empty.
fn as_object(value: &Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m.clone(),
        _ => Map::new(),
    }
}

/// `config.get(key)` returning the sub-object when it is a dict, else `{}`.
/// Matches `config.get(key) if isinstance(config.get(key), dict) else {}`.
fn get_dict<'v>(map: &'v Map<String, Value>, key: &str) -> Map<String, Value> {
    match map.get(key) {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    }
}

/// `str(d.get(key) or "").strip().lower()` — Python coerces None/empty via `or`.
fn str_lower(map: &Map<String, Value>, key: &str) -> String {
    let raw = match map.get(key) {
        Some(Value::String(s)) => s.clone(),
        // Python `str(x)` of non-strings, but the original only ever applies
        // this to string-or-missing config values; a non-string yields its
        // JSON-ish string here. Falsy values (None/empty) fall through.
        Some(Value::Null) | None => String::new(),
        Some(Value::Bool(b)) => {
            // bool is truthy/falsy under `or`; `str(False)` would be "False".
            if *b {
                "true".to_string()
            } else {
                String::new()
            }
        }
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    };
    raw.trim().to_lowercase()
}

/// Convert an optional JSON value into a [`TruthyInput`], matching Python's
/// `dict.get(...)` semantics used by `is_truthy_value`.
fn json_to_truthy(value: Option<&Value>) -> TruthyInput {
    match value {
        None | Some(Value::Null) => TruthyInput::None,
        Some(Value::Bool(b)) => TruthyInput::Bool(*b),
        Some(Value::String(s)) => TruthyInput::Str(s.clone()),
        Some(Value::Number(n)) => {
            let truthy = n.as_f64().map(|f| f != 0.0).unwrap_or(true);
            TruthyInput::Other(truthy)
        }
        Some(Value::Array(a)) => TruthyInput::Other(!a.is_empty()),
        Some(Value::Object(o)) => TruthyInput::Other(!o.is_empty()),
    }
}

/// Return True when a config section explicitly opts into the gateway.
///
/// Faithful port of `_uses_gateway`: non-dicts yield `false`; otherwise
/// `is_truthy_value(section.get("use_gateway"), default=False)`.
pub fn uses_gateway(section: Option<&Value>) -> bool {
    if let Some(Value::Object(map)) = section {
        return is_truthy_value(&json_to_truthy(map.get("use_gateway")), false);
    }
    false
}

/// Truthiness of an env-derived `Option<String>` (Python `bool(value)`):
/// `None` -> false, `Some("")` -> false, otherwise true.
fn env_truthy(value: Option<String>) -> bool {
    matches!(value, Some(v) if !v.is_empty())
}

// ---------------------------------------------------------------------------
// model config & toolset resolution
// ---------------------------------------------------------------------------

/// Faithful port of `_model_config_dict`.
///
/// * dict `model` -> a copy of it.
/// * non-empty str `model` -> `{"default": <stripped>}`.
/// * otherwise -> `{}`.
fn model_config_dict(config: &Map<String, Value>) -> Map<String, Value> {
    match config.get("model") {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) if !s.trim().is_empty() => {
            let mut m = Map::new();
            m.insert("default".to_string(), Value::String(s.trim().to_string()));
            m
        }
        _ => Map::new(),
    }
}

/// Faithful port of `_toolset_enabled`.
fn toolset_enabled(env: &SubscriptionEnv<'_>, config: &Map<String, Value>, toolset_key: &str) -> bool {
    // platform_toolsets defaulting.
    let mut platform_toolsets: Map<String, Value> = match config.get("platform_toolsets") {
        Some(Value::Object(m)) if !m.is_empty() => m.clone(),
        _ => {
            let mut m = Map::new();
            m.insert(
                "cli".to_string(),
                Value::Array(vec![Value::String((env.default_platform_toolset)("cli"))]),
            );
            m
        }
    };
    if platform_toolsets.is_empty() {
        platform_toolsets.insert(
            "cli".to_string(),
            Value::Array(vec![Value::String((env.default_platform_toolset)("cli"))]),
        );
    }

    let target_tools: BTreeSet<String> =
        (env.resolve_toolset)(toolset_key).into_iter().collect();
    if target_tools.is_empty() {
        return false;
    }

    for (platform, raw_toolsets) in platform_toolsets.iter() {
        let mut toolset_names: Vec<String> = match raw_toolsets {
            Value::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    // Non-string entries pass through as-is in Python's list();
                    // they are filtered out below by the `isinstance(str)` check.
                    other => other.to_string(),
                })
                .collect(),
            _ => {
                let default_toolset = (env.default_platform_toolset)(platform);
                if !default_toolset.is_empty() {
                    vec![default_toolset]
                } else {
                    Vec::new()
                }
            }
        };

        // The Python original (re-)computes a default when the names list is
        // empty even though the array branch may have produced an empty list.
        if toolset_names.is_empty() {
            let default_toolset = (env.default_platform_toolset)(platform);
            if !default_toolset.is_empty() {
                toolset_names = vec![default_toolset];
            }
        }

        let mut available_tools: BTreeSet<String> = BTreeSet::new();
        for toolset_name in &toolset_names {
            if toolset_name.is_empty() {
                continue;
            }
            // Only string names that round-trip cleanly are honoured; the
            // resolver is total in Rust, returning an empty Vec for unknowns
            // (equivalent to Python's swallowed exception path).
            for tool in (env.resolve_toolset)(toolset_name) {
                available_tools.insert(tool);
            }
        }

        if !target_tools.is_empty() && target_tools.is_subset(&available_tools) {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Provider labels
// ---------------------------------------------------------------------------

/// Faithful port of `_browser_label`.
pub fn browser_label(current_provider: &str) -> String {
    let key = if current_provider.is_empty() {
        "local"
    } else {
        current_provider
    };
    match key {
        "browserbase" => "Browserbase".to_string(),
        "browser-use" => "Browser Use".to_string(),
        "firecrawl" => "Firecrawl".to_string(),
        "camofox" => "Camofox".to_string(),
        "local" => "Local browser".to_string(),
        other => {
            // mapping.get(provider, provider or "Local browser")
            if other.is_empty() {
                "Local browser".to_string()
            } else {
                other.to_string()
            }
        }
    }
}

/// Faithful port of `_tts_label`.
pub fn tts_label(current_provider: &str) -> String {
    let key = if current_provider.is_empty() {
        "edge"
    } else {
        current_provider
    };
    match key {
        "openai" => "OpenAI TTS".to_string(),
        "elevenlabs" => "ElevenLabs".to_string(),
        "edge" => "Edge TTS".to_string(),
        "xai" => "xAI TTS".to_string(),
        "mistral" => "Mistral Voxtral TTS".to_string(),
        "neutts" => "NeuTTS".to_string(),
        other => {
            if other.is_empty() {
                "Edge TTS".to_string()
            } else {
                other.to_string()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Browser feature-state resolution
// ---------------------------------------------------------------------------

/// Inputs to [`resolve_browser_feature_state`].
#[derive(Debug, Clone)]
pub struct BrowserFeatureInputs {
    pub browser_tool_enabled: bool,
    pub browser_provider: String,
    pub browser_provider_explicit: bool,
    pub browser_local_available: bool,
    pub direct_camofox: bool,
    pub direct_browserbase: bool,
    pub direct_browser_use: bool,
    pub direct_firecrawl: bool,
    pub managed_browser_available: bool,
}

/// Resolved browser feature state: `(current_provider, available, active, managed)`.
pub type BrowserFeatureState = (String, bool, bool, bool);

/// Faithful port of `_resolve_browser_feature_state`.
pub fn resolve_browser_feature_state(i: &BrowserFeatureInputs) -> BrowserFeatureState {
    if i.direct_camofox {
        return ("camofox".to_string(), true, i.browser_tool_enabled, false);
    }

    if i.browser_provider_explicit {
        let current_provider = if i.browser_provider.is_empty() {
            "local".to_string()
        } else {
            i.browser_provider.clone()
        };
        if current_provider == "browserbase" {
            let available = i.browser_local_available && i.direct_browserbase;
            let active = i.browser_tool_enabled && available;
            return (current_provider, available, active, false);
        }
        if current_provider == "browser-use" {
            let provider_available = i.managed_browser_available || i.direct_browser_use;
            let available = i.browser_local_available && provider_available;
            let managed = i.browser_tool_enabled
                && i.browser_local_available
                && i.managed_browser_available
                && !i.direct_browser_use;
            let active = i.browser_tool_enabled && available;
            return (current_provider, available, active, managed);
        }
        if current_provider == "firecrawl" {
            let available = i.browser_local_available && i.direct_firecrawl;
            let active = i.browser_tool_enabled && available;
            return (current_provider, available, active, false);
        }
        if current_provider == "camofox" {
            return (current_provider, false, false, false);
        }

        let current_provider = "local".to_string();
        let available = i.browser_local_available;
        let active = i.browser_tool_enabled && available;
        return (current_provider, available, active, false);
    }

    if i.managed_browser_available || i.direct_browser_use {
        let available = i.browser_local_available;
        let managed = i.browser_tool_enabled
            && i.browser_local_available
            && i.managed_browser_available
            && !i.direct_browser_use;
        let active = i.browser_tool_enabled && available;
        return ("browser-use".to_string(), available, active, managed);
    }

    if i.direct_browserbase {
        let available = i.browser_local_available;
        let active = i.browser_tool_enabled && available;
        return ("browserbase".to_string(), available, active, false);
    }

    let available = i.browser_local_available;
    let active = i.browser_tool_enabled && available;
    ("local".to_string(), available, active, false)
}

// ---------------------------------------------------------------------------
// get_nous_subscription_features
// ---------------------------------------------------------------------------

/// Faithful port of `get_nous_subscription_features`.
///
/// `config` is the loaded config object (Python loads it via `load_config()`
/// when `None`; the caller supplies it here). `env` provides the cross-cutting
/// decisions; use [`SubscriptionEnv::default`] for native behaviour.
pub fn get_nous_subscription_features(
    config: &Value,
    env: &SubscriptionEnv<'_>,
) -> NousSubscriptionFeatures {
    let config = as_object(config);
    let model_cfg = model_config_dict(&config);
    let provider_is_nous = str_lower(&model_cfg, "provider") == "nous";

    let managed_tools_flag = (env.managed_nous_tools_enabled)();
    let nous_auth_present = (env.nous_logged_in)();
    let subscribed = provider_is_nous || nous_auth_present;

    let web_tool_enabled = toolset_enabled(env, &config, "web");
    let image_tool_enabled = toolset_enabled(env, &config, "image_gen");
    let tts_tool_enabled = toolset_enabled(env, &config, "tts");
    let browser_tool_enabled = toolset_enabled(env, &config, "browser");
    let modal_tool_enabled = toolset_enabled(env, &config, "terminal");

    let web_cfg = get_dict(&config, "web");
    let tts_cfg = get_dict(&config, "tts");
    let browser_cfg = get_dict(&config, "browser");
    let terminal_cfg = get_dict(&config, "terminal");

    let web_backend = str_lower(&web_cfg, "backend");
    let web_search_backend = str_lower(&web_cfg, "search_backend");
    let web_extract_backend = str_lower(&web_cfg, "extract_backend");
    let _ = &web_extract_backend; // parsed for parity; not used in active calc
    let tts_provider_raw = match web_or_default(&tts_cfg, "provider", "edge") {
        s => s,
    };
    let tts_provider = tts_provider_raw;
    let browser_provider_explicit = browser_cfg.contains_key("cloud_provider");
    let browser_provider = crate::tool_tool_backend_helpers::normalize_browser_cloud_provider(
        if browser_provider_explicit {
            browser_cfg.get("cloud_provider").and_then(|v| v.as_str())
        } else {
            None
        },
    );
    let terminal_backend = web_or_default(&terminal_cfg, "backend", "local");
    let modal_mode = crate::tool_tool_backend_helpers::normalize_modal_mode(
        terminal_cfg.get("modal_mode").and_then(|v| v.as_str()),
    );

    let web_use_gateway = uses_gateway(config.get("web"));
    let tts_use_gateway = uses_gateway(config.get("tts"));
    let browser_use_gateway = uses_gateway(config.get("browser"));
    let image_use_gateway = uses_gateway(config.get("image_gen"));

    let getenv = &env.get_env_value;

    let mut direct_exa = env_truthy(getenv("EXA_API_KEY"));
    let mut direct_firecrawl =
        env_truthy(getenv("FIRECRAWL_API_KEY")) || env_truthy(getenv("FIRECRAWL_API_URL"));
    let mut direct_parallel = env_truthy(getenv("PARALLEL_API_KEY"));
    let mut direct_tavily = env_truthy(getenv("TAVILY_API_KEY"));
    let direct_searxng = env_truthy(getenv("SEARXNG_URL"));
    let mut direct_fal = crate::tool_tool_backend_helpers::fal_key_is_configured(
        getenv("FAL_KEY").as_deref(),
    );
    let mut direct_openai_tts =
        !crate::tool_tool_backend_helpers::resolve_openai_audio_api_key().is_empty();
    let mut direct_elevenlabs = env_truthy(getenv("ELEVENLABS_API_KEY"));
    let direct_camofox = env_truthy(getenv("CAMOFOX_URL"));
    let mut direct_browserbase =
        env_truthy(getenv("BROWSERBASE_API_KEY")) && env_truthy(getenv("BROWSERBASE_PROJECT_ID"));
    let mut direct_browser_use = env_truthy(getenv("BROWSER_USE_API_KEY"));
    let direct_modal = crate::tool_tool_backend_helpers::has_direct_modal_credentials();

    // When use_gateway is set, suppress direct credentials for managed detection.
    if web_use_gateway {
        direct_firecrawl = false;
        direct_exa = false;
        direct_parallel = false;
        direct_tavily = false;
    }
    if image_use_gateway {
        direct_fal = false;
    }
    if tts_use_gateway {
        direct_openai_tts = false;
        direct_elevenlabs = false;
    }
    if browser_use_gateway {
        direct_browser_use = false;
        direct_browserbase = false;
    }

    let gw_ready = &env.is_managed_tool_gateway_ready;
    let managed_web_available =
        managed_tools_flag && nous_auth_present && gw_ready("firecrawl");
    let managed_image_available =
        managed_tools_flag && nous_auth_present && gw_ready("fal-queue");
    let managed_tts_available =
        managed_tools_flag && nous_auth_present && gw_ready("openai-audio");
    let managed_browser_available =
        managed_tools_flag && nous_auth_present && gw_ready("browser-use");
    let managed_modal_available =
        managed_tools_flag && nous_auth_present && gw_ready("modal");

    let modal_state = crate::tool_tool_backend_helpers::resolve_modal_backend_state(
        Some(modal_mode.as_str()),
        direct_modal,
        managed_modal_available,
        managed_tools_flag,
    );

    let web_managed = web_backend == "firecrawl" && managed_web_available && !direct_firecrawl;
    let web_active = web_tool_enabled
        && (web_managed
            || (web_backend == "exa" && direct_exa)
            || (web_backend == "firecrawl" && direct_firecrawl)
            || (web_backend == "parallel" && direct_parallel)
            || (web_backend == "tavily" && direct_tavily)
            || (web_backend == "searxng" && direct_searxng)
            || (web_search_backend == "searxng" && direct_searxng)
            || (web_search_backend == "exa" && direct_exa)
            || (web_search_backend == "firecrawl" && direct_firecrawl)
            || (web_search_backend == "parallel" && direct_parallel)
            || (web_search_backend == "tavily" && direct_tavily));
    let web_available = managed_web_available
        || direct_exa
        || direct_firecrawl
        || direct_parallel
        || direct_tavily
        || direct_searxng;

    let image_managed = image_tool_enabled && managed_image_available && !direct_fal;
    let image_active = image_tool_enabled && (image_managed || direct_fal);
    let image_available = managed_image_available || direct_fal;

    let tts_current_provider = if tts_provider.is_empty() {
        "edge".to_string()
    } else {
        tts_provider.clone()
    };
    let tts_managed = tts_tool_enabled
        && tts_current_provider == "openai"
        && managed_tts_available
        && !direct_openai_tts;
    let tts_available = tts_current_provider == "edge"
        || tts_current_provider == "neutts"
        || (tts_current_provider == "openai" && (managed_tts_available || direct_openai_tts))
        || (tts_current_provider == "elevenlabs" && direct_elevenlabs)
        || (tts_current_provider == "mistral" && env_truthy(getenv("MISTRAL_API_KEY")));
    let tts_active = tts_tool_enabled && tts_available;

    let browser_local_available = (env.has_agent_browser)();
    let (browser_current_provider, browser_available, browser_active, browser_managed) =
        resolve_browser_feature_state(&BrowserFeatureInputs {
            browser_tool_enabled,
            browser_provider: browser_provider.clone(),
            browser_provider_explicit,
            browser_local_available,
            direct_camofox,
            direct_browserbase,
            direct_browser_use,
            direct_firecrawl,
            managed_browser_available,
        });

    let modal_managed;
    let modal_available;
    let modal_active;
    let modal_direct_override;
    if terminal_backend != "modal" {
        modal_managed = false;
        modal_available = true;
        modal_active = modal_tool_enabled;
        modal_direct_override = false;
    } else if modal_state.selected_backend.as_deref() == Some("managed") {
        modal_managed = modal_tool_enabled;
        modal_available = true;
        modal_active = modal_tool_enabled;
        modal_direct_override = false;
    } else if modal_state.selected_backend.as_deref() == Some("direct") {
        modal_managed = false;
        modal_available = true;
        modal_active = modal_tool_enabled;
        modal_direct_override = modal_tool_enabled;
    } else if modal_mode == "managed" {
        modal_managed = false;
        modal_available = managed_modal_available;
        modal_active = false;
        modal_direct_override = false;
    } else if modal_mode == "direct" {
        modal_managed = false;
        modal_available = direct_modal;
        modal_active = false;
        modal_direct_override = false;
    } else {
        modal_managed = false;
        modal_available = managed_modal_available || direct_modal;
        modal_active = false;
        modal_direct_override = false;
    }

    // tts_explicit_configured: only when the *raw* config has a `provider` key.
    let mut tts_explicit_configured = false;
    if let Some(Value::Object(raw)) = config.get("tts") {
        if raw.contains_key("provider") {
            tts_explicit_configured = tts_provider != "edge" && !tts_provider.is_empty();
        }
    }

    let mut features: std::collections::HashMap<String, NousFeatureState> =
        std::collections::HashMap::new();

    let web_current_provider = if !web_backend.is_empty() {
        web_backend.clone()
    } else if !web_search_backend.is_empty() {
        web_search_backend.clone()
    } else {
        String::new()
    };

    features.insert(
        "web".to_string(),
        NousFeatureState {
            key: "web".to_string(),
            label: "Web tools".to_string(),
            included_by_default: true,
            available: web_available,
            active: web_active,
            managed_by_nous: web_managed,
            direct_override: web_active && !web_managed,
            toolset_enabled: web_tool_enabled,
            current_provider: web_current_provider,
            explicit_configured: !web_backend.is_empty() || !web_search_backend.is_empty(),
        },
    );

    features.insert(
        "image_gen".to_string(),
        NousFeatureState {
            key: "image_gen".to_string(),
            label: "Image generation".to_string(),
            included_by_default: true,
            available: image_available,
            active: image_active,
            managed_by_nous: image_managed,
            direct_override: image_active && !image_managed,
            toolset_enabled: image_tool_enabled,
            current_provider: if direct_fal {
                "FAL".to_string()
            } else if image_managed {
                "Nous Subscription".to_string()
            } else {
                String::new()
            },
            explicit_configured: direct_fal,
        },
    );

    features.insert(
        "tts".to_string(),
        NousFeatureState {
            key: "tts".to_string(),
            label: "OpenAI TTS".to_string(),
            included_by_default: true,
            available: tts_available,
            active: tts_active,
            managed_by_nous: tts_managed,
            direct_override: tts_active && !tts_managed,
            toolset_enabled: tts_tool_enabled,
            current_provider: tts_label(&tts_current_provider),
            explicit_configured: tts_explicit_configured,
        },
    );

    features.insert(
        "browser".to_string(),
        NousFeatureState {
            key: "browser".to_string(),
            label: "Browser automation".to_string(),
            included_by_default: true,
            available: browser_available,
            active: browser_active,
            managed_by_nous: browser_managed,
            direct_override: browser_active && !browser_managed,
            toolset_enabled: browser_tool_enabled,
            current_provider: browser_label(&browser_current_provider),
            explicit_configured: browser_provider_explicit,
        },
    );

    let modal_current_provider = if terminal_backend == "modal" {
        "Modal".to_string()
    } else if !terminal_backend.is_empty() {
        terminal_backend.clone()
    } else {
        "local".to_string()
    };

    features.insert(
        "modal".to_string(),
        NousFeatureState {
            key: "modal".to_string(),
            label: "Modal execution".to_string(),
            included_by_default: false,
            available: modal_available,
            active: modal_active,
            managed_by_nous: modal_managed,
            direct_override: terminal_backend == "modal" && modal_direct_override,
            toolset_enabled: modal_tool_enabled,
            current_provider: modal_current_provider,
            explicit_configured: terminal_backend == "modal",
        },
    );

    NousSubscriptionFeatures {
        subscribed,
        nous_auth_present,
        provider_is_nous,
        features,
    }
}

/// `str(d.get(key) or default).strip().lower()`.
fn web_or_default(map: &Map<String, Value>, key: &str, default: &str) -> String {
    let raw = match map.get(key) {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::Bool(true)) => "true".to_string(),
        Some(Value::Number(n)) if n.as_f64().map(|f| f != 0.0).unwrap_or(true) => n.to_string(),
        // Falsy (None / "" / false / 0) -> default.
        _ => default.to_string(),
    };
    raw.trim().to_lowercase()
}

// ---------------------------------------------------------------------------
// apply_nous_managed_defaults
// ---------------------------------------------------------------------------

/// Faithful port of `apply_nous_managed_defaults`.
///
/// Mutates `config` in place and returns the set of changed tool keys. The
/// Python original early-returns an empty set when managed tools are disabled
/// or the provider is not Nous.
pub fn apply_nous_managed_defaults(
    config: &mut Value,
    enabled_toolsets: &[String],
    env: &SubscriptionEnv<'_>,
) -> BTreeSet<String> {
    if !(env.managed_nous_tools_enabled)() {
        return BTreeSet::new();
    }

    let features = get_nous_subscription_features(config, env);
    if !features.provider_is_nous {
        return BTreeSet::new();
    }

    let selected: BTreeSet<&str> = enabled_toolsets.iter().map(String::as_str).collect();
    let mut changed: BTreeSet<String> = BTreeSet::new();

    let obj = ensure_object(config);
    ensure_section(obj, "web");
    ensure_section(obj, "tts");
    ensure_section(obj, "browser");

    let getenv = &env.get_env_value;

    if selected.contains("web")
        && !features.web().explicit_configured
        && !(env_truthy(getenv("PARALLEL_API_KEY"))
            || env_truthy(getenv("TAVILY_API_KEY"))
            || env_truthy(getenv("FIRECRAWL_API_KEY"))
            || env_truthy(getenv("FIRECRAWL_API_URL")))
    {
        set_in_section(obj, "web", "backend", Value::String("firecrawl".to_string()));
        changed.insert("web".to_string());
    }

    if selected.contains("tts")
        && !features.tts().explicit_configured
        && !(!crate::tool_tool_backend_helpers::resolve_openai_audio_api_key().is_empty()
            || env_truthy(getenv("ELEVENLABS_API_KEY")))
    {
        set_in_section(obj, "tts", "provider", Value::String("openai".to_string()));
        changed.insert("tts".to_string());
    }

    if selected.contains("browser")
        && !features.browser().explicit_configured
        && !(env_truthy(getenv("BROWSER_USE_API_KEY"))
            || env_truthy(getenv("BROWSERBASE_API_KEY")))
    {
        set_in_section(
            obj,
            "browser",
            "cloud_provider",
            Value::String("browser-use".to_string()),
        );
        changed.insert("browser".to_string());
    }

    if selected.contains("image_gen")
        && !crate::tool_tool_backend_helpers::fal_key_is_configured(getenv("FAL_KEY").as_deref())
    {
        changed.insert("image_gen".to_string());
    }

    changed
}

/// Ensure `config` is a JSON object, replacing non-objects with `{}`, and
/// return a mutable reference to its map.
fn ensure_object(config: &mut Value) -> &mut Map<String, Value> {
    if !config.is_object() {
        *config = Value::Object(Map::new());
    }
    config.as_object_mut().expect("just ensured object")
}

/// Ensure `obj[key]` is a dict (matching the Python `if not isinstance(...)`).
fn ensure_section(obj: &mut Map<String, Value>, key: &str) {
    let needs_replace = !matches!(obj.get(key), Some(Value::Object(_)));
    if needs_replace {
        obj.insert(key.to_string(), Value::Object(Map::new()));
    }
}

fn set_in_section(obj: &mut Map<String, Value>, section: &str, key: &str, value: Value) {
    ensure_section(obj, section);
    if let Some(Value::Object(m)) = obj.get_mut(section) {
        m.insert(key.to_string(), value);
    }
}

// ---------------------------------------------------------------------------
// Tool Gateway offer helpers
// ---------------------------------------------------------------------------

/// Faithful port of `_GATEWAY_TOOL_LABELS`.
pub fn gateway_tool_label(key: &str) -> Option<&'static str> {
    match key {
        "web" => Some("Web search & extract (Firecrawl)"),
        "image_gen" => Some("Image generation (FAL)"),
        "tts" => Some("Text-to-speech (OpenAI TTS)"),
        "browser" => Some("Browser automation (Browser Use)"),
        _ => None,
    }
}

/// Faithful port of `_GATEWAY_DIRECT_LABELS`.
pub fn gateway_direct_label(key: &str) -> Option<&'static str> {
    match key {
        "web" => Some("Firecrawl/Exa/Parallel/Tavily key"),
        "image_gen" => Some("FAL key"),
        "tts" => Some("OpenAI/ElevenLabs key"),
        "browser" => Some("Browser Use/Browserbase key"),
        _ => None,
    }
}

/// Faithful port of `_ALL_GATEWAY_KEYS`.
pub const ALL_GATEWAY_KEYS: [&str; 4] = ["web", "image_gen", "tts", "browser"];

/// Faithful port of `_get_gateway_direct_credentials`.
///
/// Returns `(web, image_gen, tts, browser)` direct-credential booleans.
pub fn get_gateway_direct_credentials(
    env: &SubscriptionEnv<'_>,
) -> std::collections::HashMap<String, bool> {
    let getenv = &env.get_env_value;
    let mut out = std::collections::HashMap::new();
    out.insert(
        "web".to_string(),
        env_truthy(getenv("FIRECRAWL_API_KEY"))
            || env_truthy(getenv("FIRECRAWL_API_URL"))
            || env_truthy(getenv("PARALLEL_API_KEY"))
            || env_truthy(getenv("TAVILY_API_KEY"))
            || env_truthy(getenv("EXA_API_KEY")),
    );
    out.insert(
        "image_gen".to_string(),
        crate::tool_tool_backend_helpers::fal_key_is_configured(getenv("FAL_KEY").as_deref()),
    );
    out.insert(
        "tts".to_string(),
        !crate::tool_tool_backend_helpers::resolve_openai_audio_api_key().is_empty()
            || env_truthy(getenv("ELEVENLABS_API_KEY")),
    );
    out.insert(
        "browser".to_string(),
        env_truthy(getenv("BROWSER_USE_API_KEY"))
            || (env_truthy(getenv("BROWSERBASE_API_KEY"))
                && env_truthy(getenv("BROWSERBASE_PROJECT_ID"))),
    );
    out
}

/// `(unconfigured, has_direct, already_managed)` tool-key lists.
pub type GatewayEligibility = (Vec<String>, Vec<String>, Vec<String>);

/// Faithful port of `get_gateway_eligible_tools`.
///
/// Returns three empty lists when managed tools are disabled or the provider
/// is not Nous.
pub fn get_gateway_eligible_tools(
    config: &Value,
    env: &SubscriptionEnv<'_>,
) -> GatewayEligibility {
    if !(env.managed_nous_tools_enabled)() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let config_obj = as_object(config);

    // Quick provider check (avoids the heavy feature computation).
    let provider_is_nous = match config_obj.get("model") {
        Some(Value::Object(m)) => str_lower(m, "provider") == "nous",
        _ => false,
    };
    if !provider_is_nous {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let direct = get_gateway_direct_credentials(env);

    let mut opted_in: std::collections::HashMap<&str, bool> = std::collections::HashMap::new();
    opted_in.insert("web", uses_gateway(config_obj.get("web")));
    opted_in.insert("image_gen", uses_gateway(config_obj.get("image_gen")));
    opted_in.insert("tts", uses_gateway(config_obj.get("tts")));
    opted_in.insert("browser", uses_gateway(config_obj.get("browser")));

    let mut unconfigured: Vec<String> = Vec::new();
    let mut has_direct: Vec<String> = Vec::new();
    let mut already_managed: Vec<String> = Vec::new();
    for key in ALL_GATEWAY_KEYS {
        if *opted_in.get(key).unwrap_or(&false) {
            already_managed.push(key.to_string());
        } else if *direct.get(key).unwrap_or(&false) {
            has_direct.push(key.to_string());
        } else {
            unconfigured.push(key.to_string());
        }
    }
    (unconfigured, has_direct, already_managed)
}

/// Faithful port of `apply_gateway_defaults`.
///
/// Mutates `config` in place, sets `use_gateway: true` (plus backend/provider
/// defaults) for the given tool keys, and returns the changed set.
pub fn apply_gateway_defaults(config: &mut Value, tool_keys: &[String]) -> BTreeSet<String> {
    let mut changed: BTreeSet<String> = BTreeSet::new();
    let obj = ensure_object(config);

    ensure_section(obj, "web");
    ensure_section(obj, "tts");
    ensure_section(obj, "browser");

    let keys: BTreeSet<&str> = tool_keys.iter().map(String::as_str).collect();

    if keys.contains("web") {
        set_in_section(obj, "web", "backend", Value::String("firecrawl".to_string()));
        set_in_section(obj, "web", "use_gateway", Value::Bool(true));
        changed.insert("web".to_string());
    }

    if keys.contains("tts") {
        set_in_section(obj, "tts", "provider", Value::String("openai".to_string()));
        set_in_section(obj, "tts", "use_gateway", Value::Bool(true));
        changed.insert("tts".to_string());
    }

    if keys.contains("browser") {
        set_in_section(
            obj,
            "browser",
            "cloud_provider",
            Value::String("browser-use".to_string()),
        );
        set_in_section(obj, "browser", "use_gateway", Value::Bool(true));
        changed.insert("browser".to_string());
    }

    if keys.contains("image_gen") {
        ensure_section(obj, "image_gen");
        set_in_section(obj, "image_gen", "use_gateway", Value::Bool(true));
        changed.insert("image_gen".to_string());
    }

    changed
}

/// Description lines shown in the gateway prompt. Faithful port of the
/// `desc_parts` builder in `prompt_enable_tool_gateway`.
pub fn build_gateway_prompt_description(
    unconfigured: &[String],
    has_direct: &[String],
    already_managed: &[String],
) -> Vec<String> {
    let mut desc_parts: Vec<String> = vec![
        String::new(),
        "  The Tool Gateway gives you access to web search, image generation,".to_string(),
        "  text-to-speech, and browser automation through your Nous subscription.".to_string(),
        "  No need to sign up for separate API keys — just pick the tools you want.".to_string(),
        String::new(),
    ];
    for k in already_managed {
        if let Some(label) = gateway_tool_label(k) {
            desc_parts.push(format!("  ✓ {label} — using Tool Gateway"));
        }
    }
    for k in unconfigured {
        if let Some(label) = gateway_tool_label(k) {
            desc_parts.push(format!("  ○ {label} — not configured"));
        }
    }
    for k in has_direct {
        if let (Some(label), Some(direct_label)) = (gateway_tool_label(k), gateway_direct_label(k))
        {
            desc_parts.push(format!("  ○ {label} — using {direct_label}"));
        }
    }
    desc_parts
}

/// The choice action returned by [`build_gateway_prompt_choices`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayChoiceAction {
    All,
    Unconfigured,
    Skip,
}

/// The choices/labels/default index for the gateway prompt. Faithful port of
/// the choice-construction block in `prompt_enable_tool_gateway`.
///
/// Returns `(choices, actions, default_idx)`.
pub fn build_gateway_prompt_choices(
    unconfigured: &[String],
    has_direct: &[String],
) -> (Vec<String>, Vec<GatewayChoiceAction>, usize) {
    let mut choices: Vec<String> = Vec::new();
    let mut actions: Vec<GatewayChoiceAction> = Vec::new();

    if !unconfigured.is_empty() && !has_direct.is_empty() {
        choices.push("Enable for all tools (existing keys kept, not used)".to_string());
        actions.push(GatewayChoiceAction::All);
        choices.push("Enable only for tools without existing keys".to_string());
        actions.push(GatewayChoiceAction::Unconfigured);
        choices.push("Skip".to_string());
        actions.push(GatewayChoiceAction::Skip);
    } else if !unconfigured.is_empty() {
        choices.push("Enable Tool Gateway".to_string());
        actions.push(GatewayChoiceAction::Unconfigured);
        choices.push("Skip".to_string());
        actions.push(GatewayChoiceAction::Skip);
    } else {
        choices.push("Enable Tool Gateway (existing keys kept, not used)".to_string());
        actions.push(GatewayChoiceAction::All);
        choices.push("Skip".to_string());
        actions.push(GatewayChoiceAction::Skip);
    }

    // Default to "Enable" (idx 0) when no direct keys; else "Skip" (last).
    let default_idx = if has_direct.is_empty() {
        0
    } else {
        choices.len() - 1
    };
    (choices, actions, default_idx)
}

/// Outcome of resolving a gateway-prompt selection (without performing the
/// interactive prompt or printing). Faithful port of the post-selection logic
/// of `prompt_enable_tool_gateway`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GatewayPromptOutcome {
    /// Tools that actually switched (already-managed excluded).
    pub newly_switched: Vec<String>,
    /// True when all eligible tools were already routed through the gateway.
    pub all_already_managed: bool,
    /// The full changed set returned by [`apply_gateway_defaults`].
    pub changed: BTreeSet<String>,
}

/// Apply the chosen gateway action and compute the reporting outcome.
///
/// This is the side-effect-free core of `prompt_enable_tool_gateway` after the
/// user has chosen an action: it mutates `config`, computes which tools newly
/// switched, and signals when everything was already managed. Persisting the
/// config (`save_config`) and printing the summary lines are left to the
/// caller (the CLI layer), matching the separation used by other ported CLI
/// helpers.
pub fn apply_gateway_choice(
    config: &mut Value,
    action: &GatewayChoiceAction,
    unconfigured: &[String],
    already_managed: &[String],
) -> Option<GatewayPromptOutcome> {
    if matches!(action, GatewayChoiceAction::Skip) {
        return None;
    }

    let to_apply: Vec<String> = match action {
        GatewayChoiceAction::All => ALL_GATEWAY_KEYS.iter().map(|s| s.to_string()).collect(),
        GatewayChoiceAction::Unconfigured => unconfigured.to_vec(),
        GatewayChoiceAction::Skip => unreachable!(),
    };

    let changed = apply_gateway_defaults(config, &to_apply);
    if changed.is_empty() {
        return Some(GatewayPromptOutcome {
            newly_switched: Vec::new(),
            all_already_managed: false,
            changed,
        });
    }

    let already: BTreeSet<&str> = already_managed.iter().map(String::as_str).collect();
    let mut newly_switched: Vec<String> = changed
        .iter()
        .filter(|k| !already.contains(k.as_str()))
        .cloned()
        .collect();
    newly_switched.sort();

    let all_already_managed = !already_managed.is_empty() && newly_switched.is_empty();

    Some(GatewayPromptOutcome {
        newly_switched,
        all_already_managed,
        changed,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Build an env with everything off and no direct credentials, with a
    /// resolver that maps each toolset key to a single sentinel tool.
    fn test_env<'a>(
        managed: bool,
        nous_logged_in: bool,
        gateway_ready: bool,
        enabled_toolsets: Vec<&'a str>,
        envvars: std::collections::HashMap<String, String>,
    ) -> SubscriptionEnv<'a> {
        let enabled: std::collections::HashSet<String> =
            enabled_toolsets.into_iter().map(|s| s.to_string()).collect();
        SubscriptionEnv {
            get_env_value: Box::new(move |k| envvars.get(k).cloned()),
            // Each toolset resolves to a tool named after itself; the cli
            // default toolset resolves to whatever tools are "enabled".
            resolve_toolset: Box::new(move |name| {
                if name == "hermes-cli" {
                    enabled.iter().map(|t| format!("tool::{t}")).collect()
                } else {
                    vec![format!("tool::{name}")]
                }
            }),
            managed_nous_tools_enabled: Box::new(move || managed),
            nous_logged_in: Box::new(move || nous_logged_in),
            is_managed_tool_gateway_ready: Box::new(move |_| gateway_ready),
            has_agent_browser: Box::new(|| true),
            default_platform_toolset: Box::new(|p| default_platform_toolset(p)),
        }
    }

    #[test]
    fn uses_gateway_handles_missing_and_truthy() {
        assert!(!uses_gateway(None));
        assert!(!uses_gateway(Some(&json!("not a dict"))));
        assert!(!uses_gateway(Some(&json!({}))));
        assert!(uses_gateway(Some(&json!({"use_gateway": true}))));
        assert!(uses_gateway(Some(&json!({"use_gateway": "yes"}))));
        assert!(!uses_gateway(Some(&json!({"use_gateway": false}))));
        assert!(!uses_gateway(Some(&json!({"use_gateway": "no"}))));
    }

    #[test]
    fn tts_and_browser_labels() {
        assert_eq!(tts_label("openai"), "OpenAI TTS");
        assert_eq!(tts_label(""), "Edge TTS");
        assert_eq!(tts_label("unknown"), "unknown");
        assert_eq!(browser_label("browser-use"), "Browser Use");
        assert_eq!(browser_label(""), "Local browser");
        assert_eq!(browser_label("camofox"), "Camofox");
    }

    #[test]
    fn model_config_dict_variants() {
        let mut m = Map::new();
        m.insert("model".to_string(), json!({"provider": "nous"}));
        assert_eq!(model_config_dict(&m).get("provider").unwrap(), "nous");

        let mut m2 = Map::new();
        m2.insert("model".to_string(), json!("  gpt-x  "));
        assert_eq!(model_config_dict(&m2).get("default").unwrap(), "gpt-x");

        let mut m3 = Map::new();
        m3.insert("model".to_string(), json!(123));
        assert!(model_config_dict(&m3).is_empty());
    }

    #[test]
    fn default_platform_toolset_fallback() {
        assert_eq!(default_platform_toolset("cli"), "hermes-cli");
        assert_eq!(default_platform_toolset("discord"), "hermes-discord");
    }

    #[test]
    fn browser_feature_state_camofox_precedence() {
        let i = BrowserFeatureInputs {
            browser_tool_enabled: true,
            browser_provider: "browserbase".to_string(),
            browser_provider_explicit: true,
            browser_local_available: false,
            direct_camofox: true,
            direct_browserbase: false,
            direct_browser_use: false,
            direct_firecrawl: false,
            managed_browser_available: false,
        };
        let (provider, available, active, managed) = resolve_browser_feature_state(&i);
        assert_eq!(provider, "camofox");
        assert!(available);
        assert!(active);
        assert!(!managed);
    }

    #[test]
    fn browser_feature_state_managed_browser_use() {
        let i = BrowserFeatureInputs {
            browser_tool_enabled: true,
            browser_provider: String::new(),
            browser_provider_explicit: false,
            browser_local_available: true,
            direct_camofox: false,
            direct_browserbase: false,
            direct_browser_use: false,
            direct_firecrawl: false,
            managed_browser_available: true,
        };
        let (provider, available, active, managed) = resolve_browser_feature_state(&i);
        assert_eq!(provider, "browser-use");
        assert!(available);
        assert!(active);
        assert!(managed);
    }

    #[test]
    fn features_provider_nous_and_managed_web() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = test_env(
            true,
            true,
            true,
            vec!["web"],
            std::collections::HashMap::new(),
        );
        let config = json!({
            "model": {"provider": "nous"},
            "web": {"backend": "firecrawl"},
        });
        let features = get_nous_subscription_features(&config, &env);
        assert!(features.provider_is_nous);
        assert!(features.subscribed);
        assert!(features.nous_auth_present);
        let web = features.web();
        assert!(web.toolset_enabled);
        assert!(web.managed_by_nous);
        assert!(web.active);
        assert!(!web.direct_override);
        assert_eq!(web.current_provider, "firecrawl");
    }

    #[test]
    fn features_direct_firecrawl_overrides_managed() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut vars = std::collections::HashMap::new();
        vars.insert("FIRECRAWL_API_KEY".to_string(), "fc-key".to_string());
        let env = test_env(true, true, true, vec!["web"], vars);
        let config = json!({
            "model": {"provider": "nous"},
            "web": {"backend": "firecrawl"},
        });
        let features = get_nous_subscription_features(&config, &env);
        let web = features.web();
        assert!(web.active);
        assert!(!web.managed_by_nous);
        assert!(web.direct_override);
    }

    #[test]
    fn features_modal_non_modal_backend_available() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = test_env(
            false,
            false,
            false,
            vec!["terminal"],
            std::collections::HashMap::new(),
        );
        let config = json!({"model": {"provider": "openai"}, "terminal": {"backend": "local"}});
        let features = get_nous_subscription_features(&config, &env);
        let modal = features.modal();
        assert!(modal.available);
        assert!(modal.active);
        assert!(!modal.managed_by_nous);
        assert_eq!(modal.current_provider, "local");
        assert!(!modal.explicit_configured);
        assert!(!features.subscribed);
    }

    #[test]
    fn apply_managed_defaults_requires_nous_provider() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = test_env(
            true,
            true,
            false,
            vec!["web", "tts", "browser"],
            std::collections::HashMap::new(),
        );
        let mut config = json!({"model": {"provider": "nous"}});
        let changed = apply_nous_managed_defaults(
            &mut config,
            &["web".to_string(), "tts".to_string(), "browser".to_string()],
            &env,
        );
        assert!(changed.contains("web"));
        assert!(changed.contains("tts"));
        assert!(changed.contains("browser"));
        assert_eq!(config["web"]["backend"], json!("firecrawl"));
        assert_eq!(config["tts"]["provider"], json!("openai"));
        assert_eq!(config["browser"]["cloud_provider"], json!("browser-use"));
    }

    #[test]
    fn apply_managed_defaults_disabled_when_not_nous() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = test_env(true, false, false, vec!["web"], std::collections::HashMap::new());
        let mut config = json!({"model": {"provider": "openai"}});
        let changed = apply_nous_managed_defaults(&mut config, &["web".to_string()], &env);
        assert!(changed.is_empty());
    }

    #[test]
    fn gateway_eligible_partitions_tools() {
        let _g = ENV_LOCK.lock().unwrap();
        let mut vars = std::collections::HashMap::new();
        vars.insert("FAL_KEY".to_string(), "fal".to_string());
        let env = test_env(true, true, false, vec![], vars);
        let config = json!({
            "model": {"provider": "nous"},
            "browser": {"use_gateway": true},
        });
        let (unconfigured, has_direct, already_managed) =
            get_gateway_eligible_tools(&config, &env);
        // image_gen has a direct FAL key; browser opted into gateway; web/tts unconfigured.
        assert_eq!(already_managed, vec!["browser".to_string()]);
        assert_eq!(has_direct, vec!["image_gen".to_string()]);
        assert_eq!(unconfigured, vec!["web".to_string(), "tts".to_string()]);
    }

    #[test]
    fn gateway_eligible_empty_when_disabled() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = test_env(false, true, false, vec![], std::collections::HashMap::new());
        let config = json!({"model": {"provider": "nous"}});
        let (u, h, a) = get_gateway_eligible_tools(&config, &env);
        assert!(u.is_empty() && h.is_empty() && a.is_empty());
    }

    #[test]
    fn apply_gateway_defaults_sets_use_gateway() {
        let mut config = json!({});
        let changed = apply_gateway_defaults(
            &mut config,
            &["web".to_string(), "image_gen".to_string()],
        );
        assert!(changed.contains("web"));
        assert!(changed.contains("image_gen"));
        assert_eq!(config["web"]["backend"], json!("firecrawl"));
        assert_eq!(config["web"]["use_gateway"], json!(true));
        assert_eq!(config["image_gen"]["use_gateway"], json!(true));
    }

    #[test]
    fn gateway_prompt_choices_all_three() {
        let (choices, actions, default_idx) = build_gateway_prompt_choices(
            &["web".to_string()],
            &["image_gen".to_string()],
        );
        assert_eq!(choices.len(), 3);
        assert_eq!(actions[0], GatewayChoiceAction::All);
        assert_eq!(actions[1], GatewayChoiceAction::Unconfigured);
        assert_eq!(actions[2], GatewayChoiceAction::Skip);
        // has_direct non-empty -> default to Skip (last).
        assert_eq!(default_idx, 2);
    }

    #[test]
    fn gateway_prompt_choices_unconfigured_only_defaults_enable() {
        let (choices, actions, default_idx) =
            build_gateway_prompt_choices(&["web".to_string()], &[]);
        assert_eq!(choices.len(), 2);
        assert_eq!(actions[0], GatewayChoiceAction::Unconfigured);
        assert_eq!(default_idx, 0);
    }

    #[test]
    fn apply_gateway_choice_reports_newly_switched() {
        let mut config = json!({});
        let outcome = apply_gateway_choice(
            &mut config,
            &GatewayChoiceAction::All,
            &["web".to_string(), "tts".to_string()],
            &["browser".to_string()],
        )
        .unwrap();
        // All four applied; browser already managed -> excluded from newly_switched.
        assert!(!outcome.newly_switched.contains(&"browser".to_string()));
        assert!(outcome.newly_switched.contains(&"web".to_string()));
        assert!(!outcome.all_already_managed);
    }

    #[test]
    fn apply_gateway_choice_skip_returns_none() {
        let mut config = json!({});
        assert!(apply_gateway_choice(
            &mut config,
            &GatewayChoiceAction::Skip,
            &[],
            &[]
        )
        .is_none());
    }
}
