//! Shared runtime provider resolution for CLI, gateway, cron, and helpers.
//!
//! Native Rust port of `hermes_cli/runtime_provider.py`.
//!
//! The Python original resolves "which provider/base_url/api_key/api_mode" the
//! agent should use, considering: explicit overrides, user-declared custom
//! providers, credential pools, OAuth/portal-minted credentials and env vars.
//!
//! A large part of that logic is pure (URL/api_mode detection, custom-provider
//! config matching, OpenRouter & Azure-Foundry resolution) and is reproduced
//! here faithfully. The credential-fetching tail (Nous portal mint, Codex/Qwen
//! OAuth refresh, Bedrock auth chain, etc.) depends on auth subsystems that are
//! ported in `crate::auth`; those are abstracted behind the [`RuntimeBackend`]
//! trait so this module stays self-contained and testable while remaining wired
//! into the real auth code in production.

use std::collections::BTreeMap;

use serde_yaml::Value as YamlValue;

use crate::mod_hermes_constants::OPENROUTER_BASE_URL;
use crate::mod_utils::{base_url_host_matches, base_url_hostname};

// ---------------------------------------------------------------------------
// Constants mirrored from hermes_cli.auth
// ---------------------------------------------------------------------------

/// `DEFAULT_CODEX_BASE_URL` from `hermes_cli.auth`.
pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
/// `DEFAULT_QWEN_BASE_URL` from `hermes_cli.auth`.
pub const DEFAULT_QWEN_BASE_URL: &str = "https://portal.qwen.ai/v1";
/// `DEFAULT_NOUS_INFERENCE_URL` from `hermes_cli.auth`.
pub const DEFAULT_NOUS_INFERENCE_URL: &str = "https://inference-api.nousresearch.com/v1";

/// The set of api_mode values `_parse_api_mode` accepts.
pub const VALID_API_MODES: &[&str] = &[
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
];

// ---------------------------------------------------------------------------
// Result type — mirrors the Python `Dict[str, Any]` runtime payload.
// ---------------------------------------------------------------------------

/// Resolved runtime credentials for agent execution.
///
/// Mirrors the dict returned by `resolve_runtime_provider` in Python. The
/// always-present fields are dedicated; the optional/provider-specific keys
/// live in [`RuntimeProvider::extra`] so the shape stays exact when serialised.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeProvider {
    pub provider: String,
    pub api_mode: String,
    pub base_url: String,
    pub api_key: String,
    pub source: String,
    /// `requested_provider` is set on most paths; kept optional to mirror the
    /// branches that omit it (custom-pool fast path, etc.).
    pub requested_provider: Option<String>,
    /// Optional model name propagated by custom providers / aliases.
    pub model: Option<String>,
    /// Extra provider-specific keys (expires_at, region, last_refresh, …).
    pub extra: BTreeMap<String, YamlValue>,
}

impl RuntimeProvider {
    fn base(provider: &str, api_mode: &str, base_url: &str, api_key: &str, source: &str) -> Self {
        RuntimeProvider {
            provider: provider.to_string(),
            api_mode: api_mode.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            source: source.to_string(),
            requested_provider: None,
            model: None,
            extra: BTreeMap::new(),
        }
    }

    /// Convenience accessor for an extra string key.
    pub fn extra_str(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(YamlValue::as_str)
    }
}

/// Auth-style error mirroring `hermes_cli.auth.AuthError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthError {
    pub message: String,
    pub code: Option<String>,
}

impl AuthError {
    pub fn new(message: impl Into<String>) -> Self {
        AuthError {
            message: message.into(),
            code: None,
        }
    }

    pub fn with_code(message: impl Into<String>, code: impl Into<String>) -> Self {
        AuthError {
            message: message.into(),
            code: Some(code.into()),
        }
    }
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AuthError {}

/// Errors raised by runtime resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    Auth(AuthError),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Auth(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ResolveError {}

impl From<AuthError> for ResolveError {
    fn from(e: AuthError) -> Self {
        ResolveError::Auth(e)
    }
}

/// Mirror of `format_runtime_provider_error`.
pub fn format_runtime_provider_error(error: &ResolveError) -> String {
    match error {
        ResolveError::Auth(e) => format_auth_error(e),
    }
}

/// Mirror of `hermes_cli.auth.format_auth_error` (best-effort: returns message).
pub fn format_auth_error(error: &AuthError) -> String {
    error.message.clone()
}

// ---------------------------------------------------------------------------
// Env / config abstraction
// ---------------------------------------------------------------------------

/// Abstraction over environment-variable lookups so the resolver is testable
/// without touching the real process environment. The production wiring passes
/// a [`ProcessEnv`].
pub trait EnvSource {
    fn get(&self, key: &str) -> Option<String>;

    /// Trimmed value, empty string if absent — matches `os.getenv(k, "").strip()`.
    fn get_trimmed(&self, key: &str) -> String {
        self.get(key).map(|v| v.trim().to_string()).unwrap_or_default()
    }
}

/// Reads from the real process environment.
pub struct ProcessEnv;

impl EnvSource for ProcessEnv {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

/// Simple map-backed env for tests / deterministic resolution.
#[derive(Debug, Clone, Default)]
pub struct MapEnv {
    pub vars: BTreeMap<String, String>,
}

impl MapEnv {
    pub fn new() -> Self {
        MapEnv::default()
    }
    pub fn set(mut self, key: &str, value: &str) -> Self {
        self.vars.insert(key.to_string(), value.to_string());
        self
    }
}

impl EnvSource for MapEnv {
    fn get(&self, key: &str) -> Option<String> {
        self.vars.get(key).cloned()
    }
}

// ---------------------------------------------------------------------------
// Provider registry abstraction (subset of PROVIDER_REGISTRY needed here).
// ---------------------------------------------------------------------------

/// Minimal provider descriptor mirroring the fields of
/// `hermes_cli.auth.PROVIDER_REGISTRY[provider]` consumed by this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryProvider {
    pub auth_type: String,
    pub inference_base_url: String,
    pub base_url_env_var: Option<String>,
}

/// Look up a provider in the registry. Wraps `crate::providers::get_provider_profile`.
pub fn registry_lookup(provider: &str) -> Option<RegistryProvider> {
    let profile = crate::providers::get_provider_profile(provider)?;
    Some(RegistryProvider {
        auth_type: profile.auth_type.to_string(),
        inference_base_url: profile.base_url.to_string(),
        base_url_env_var: profile.base_url_env_var().map(str::to_string),
    })
}

// ---------------------------------------------------------------------------
// Pure helpers (faithful ports)
// ---------------------------------------------------------------------------

/// `_normalize_custom_provider_name`
pub fn normalize_custom_provider_name(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(' ', "-")
}

/// `_loopback_hostname`
pub fn loopback_hostname(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    let h = h.trim_end_matches('.');
    matches!(h, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

/// `_config_base_url_trustworthy_for_bare_custom`
pub fn config_base_url_trustworthy_for_bare_custom(cfg_base_url: &str, cfg_provider: &str) -> bool {
    let cfg_provider_norm = cfg_provider.trim().to_ascii_lowercase();
    let bu = cfg_base_url.trim();
    if bu.is_empty() {
        return false;
    }
    if cfg_provider_norm == "custom" {
        return true;
    }
    if base_url_host_matches(bu, "openrouter.ai") {
        return false;
    }
    loopback_hostname(&base_url_hostname(bu))
}

/// `_detect_api_mode_for_url`
pub fn detect_api_mode_for_url(base_url: &str) -> Option<&'static str> {
    let normalized = base_url.trim().to_ascii_lowercase();
    let normalized = normalized.trim_end_matches('/');
    let hostname = base_url_hostname(base_url);
    if hostname == "api.x.ai" {
        return Some("codex_responses");
    }
    if hostname == "api.openai.com" {
        return Some("codex_responses");
    }
    if normalized.ends_with("/anthropic") {
        return Some("anthropic_messages");
    }
    if hostname == "api.kimi.com" && normalized.contains("/coding") {
        return Some("anthropic_messages");
    }
    None
}

/// `_parse_api_mode`
pub fn parse_api_mode(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let normalized = raw.trim().to_ascii_lowercase();
    if VALID_API_MODES.contains(&normalized.as_str()) {
        Some(normalized)
    } else {
        None
    }
}

fn parse_api_mode_value(raw: Option<&YamlValue>) -> Option<String> {
    parse_api_mode(raw.and_then(YamlValue::as_str))
}

/// `_provider_supports_explicit_api_mode`
pub fn provider_supports_explicit_api_mode(
    provider: Option<&str>,
    configured_provider: Option<&str>,
) -> bool {
    let normalized_provider = provider.unwrap_or("").trim().to_ascii_lowercase();
    let normalized_configured = configured_provider.unwrap_or("").trim().to_ascii_lowercase();
    if normalized_configured.is_empty() {
        return true;
    }
    if normalized_provider == "custom" {
        return normalized_configured == "custom" || normalized_configured.starts_with("custom:");
    }
    normalized_configured == normalized_provider
}

/// `has_usable_secret` — mirror of `hermes_cli.auth.has_usable_secret`.
///
/// A secret is usable when it is a non-empty, non-placeholder string.
pub fn has_usable_secret(value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() {
        return false;
    }
    let lowered = v.to_ascii_lowercase();
    !matches!(
        lowered.as_str(),
        "no-key-required"
            | "none"
            | "null"
            | "n/a"
            | "na"
            | "placeholder"
            | "your-api-key"
            | "your_api_key"
            | "changeme"
            | "change-me"
            | "xxx"
    )
}

fn rstrip_slash(s: &str) -> String {
    s.trim_end_matches('/').to_string()
}

fn strip_trailing_v1(s: &str) -> String {
    // Equivalent to re.sub(r"/v1/?$", "", base_url)
    if let Some(stripped) = s.strip_suffix("/v1/") {
        return stripped.to_string();
    }
    if let Some(stripped) = s.strip_suffix("/v1") {
        return stripped.to_string();
    }
    s.to_string()
}

// ---------------------------------------------------------------------------
// Model config extraction
// ---------------------------------------------------------------------------

/// Parsed view of the `model:` section of config.yaml, mirroring the dict
/// returned by Python `_get_model_config()`.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Raw map of model.* keys (string-keyed), used for arbitrary lookups.
    pub map: BTreeMap<String, YamlValue>,
}

impl ModelConfig {
    pub fn get_str(&self, key: &str) -> String {
        self.map
            .get(key)
            .and_then(YamlValue::as_str)
            .unwrap_or("")
            .to_string()
    }

    pub fn get_value(&self, key: &str) -> Option<&YamlValue> {
        self.map.get(key)
    }

    pub fn default_model(&self) -> String {
        self.get_str("default")
    }

    pub fn provider(&self) -> String {
        self.get_str("provider")
    }

    pub fn base_url(&self) -> String {
        self.get_str("base_url")
    }
}

/// Build a [`ModelConfig`] from a parsed config document.
///
/// Mirrors `_get_model_config()` minus the local-model auto-detection network
/// call (which is exposed separately via [`auto_detect_local_model`] so callers
/// can opt in). Accepts the top-level config `Value` (a YAML mapping).
pub fn get_model_config(config: &YamlValue) -> ModelConfig {
    let model_cfg = config.get("model");
    match model_cfg {
        Some(YamlValue::Mapping(_)) => {
            let mut map: BTreeMap<String, YamlValue> = BTreeMap::new();
            if let Some(m) = model_cfg.and_then(YamlValue::as_mapping) {
                for (k, v) in m {
                    if let Some(ks) = k.as_str() {
                        map.insert(ks.to_string(), v.clone());
                    }
                }
            }
            // Accept "model" as alias for "default".
            let has_default = map
                .get("default")
                .and_then(YamlValue::as_str)
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if !has_default {
                if let Some(YamlValue::String(model)) = map.get("model").cloned().as_ref() {
                    if !model.is_empty() {
                        map.insert("default".to_string(), YamlValue::String(model.clone()));
                    }
                }
            }
            ModelConfig { map }
        }
        Some(YamlValue::String(s)) if !s.trim().is_empty() => {
            let mut map = BTreeMap::new();
            map.insert(
                "default".to_string(),
                YamlValue::String(s.trim().to_string()),
            );
            ModelConfig { map }
        }
        _ => ModelConfig::default(),
    }
}

/// `_auto_detect_local_model` — query a local OpenAI-compatible server for the
/// single loaded model. Uses a blocking reqwest call. Returns "" on any error.
pub fn auto_detect_local_model(base_url: &str) -> String {
    if base_url.is_empty() {
        return String::new();
    }
    let mut url = rstrip_slash(base_url);
    if !url.ends_with("/v1") {
        url.push_str("/v1");
    }
    let result = (|| -> Option<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()?;
        let resp = client.get(format!("{url}/models")).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body: serde_json::Value = resp.json().ok()?;
        let models = body.get("data")?.as_array()?;
        if models.len() == 1 {
            let model_id = models[0].get("id")?.as_str()?;
            if !model_id.is_empty() {
                return Some(model_id.to_string());
            }
        }
        None
    })();
    result.unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Custom provider lookup from config (pure)
// ---------------------------------------------------------------------------

/// A resolved custom-provider config entry (subset used by the resolver),
/// mirroring the dict returned by `_get_named_custom_provider`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NamedCustomProvider {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: Option<String>,
    pub api_mode: Option<String>,
    pub key_env: Option<String>,
    pub provider_key: Option<String>,
}

fn yaml_str(v: Option<&YamlValue>) -> String {
    v.and_then(YamlValue::as_str).unwrap_or("").trim().to_string()
}

/// `_get_named_custom_provider`.
///
/// `canonical_for` mirrors `auth_mod.resolve_provider(name)` — pass the closure
/// that maps a raw name to its canonical built-in provider (or returns None on
/// AuthError). In production this is wired to `crate::providers`.
pub fn get_named_custom_provider<E: EnvSource, F>(
    requested_provider: &str,
    config: &YamlValue,
    env: &E,
    canonical_for: F,
) -> Option<NamedCustomProvider>
where
    F: Fn(&str) -> Option<String>,
{
    let requested_norm = normalize_custom_provider_name(requested_provider);
    if requested_norm.is_empty() || requested_norm == "custom" {
        return None;
    }
    if requested_norm == "auto" {
        return None;
    }
    if !requested_norm.starts_with("custom:") {
        if let Some(canonical) = canonical_for(&requested_norm) {
            if canonical.trim().to_ascii_lowercase() == requested_norm {
                return None;
            }
        }
    }

    // First: providers: dict (new-style).
    if let Some(providers) = config.get("providers").and_then(YamlValue::as_mapping) {
        for (ep_key, entry) in providers {
            let ep_name = match ep_key.as_str() {
                Some(s) => s,
                None => continue,
            };
            let entry = match entry.as_mapping() {
                Some(_) => entry,
                None => continue,
            };
            let name_norm = normalize_custom_provider_name(ep_name);
            let key_env = yaml_str(entry.get("key_env"));
            let mut resolved_api_key = if !key_env.is_empty() {
                env.get_trimmed(&key_env)
            } else {
                String::new()
            };
            if resolved_api_key.is_empty() {
                resolved_api_key = yaml_str(entry.get("api_key"));
            }

            let custom_key = format!("custom:{name_norm}");
            if requested_norm == ep_name || requested_norm == name_norm || requested_norm == custom_key
            {
                let base_url = first_non_empty(&[
                    yaml_str(entry.get("api")),
                    yaml_str(entry.get("url")),
                    yaml_str(entry.get("base_url")),
                ]);
                if !base_url.is_empty() {
                    let display = yaml_str(entry.get("name"));
                    let mut result = NamedCustomProvider {
                        name: if display.is_empty() {
                            ep_name.to_string()
                        } else {
                            display
                        },
                        base_url,
                        api_key: resolved_api_key.clone(),
                        model: non_empty_opt(yaml_str(entry.get("default_model"))),
                        ..Default::default()
                    };
                    result.api_mode = parse_api_mode_value(entry.get("api_mode"))
                        .or_else(|| parse_api_mode_value(entry.get("transport")));
                    return Some(result);
                }
            }
            // Match by display name.
            let display_name = yaml_str(entry.get("name"));
            if !display_name.is_empty() {
                let display_norm = normalize_custom_provider_name(&display_name);
                let display_custom = format!("custom:{display_norm}");
                if requested_norm == display_name
                    || requested_norm == display_norm
                    || requested_norm == display_custom
                {
                    let base_url = first_non_empty(&[
                        yaml_str(entry.get("api")),
                        yaml_str(entry.get("url")),
                        yaml_str(entry.get("base_url")),
                    ]);
                    if !base_url.is_empty() {
                        let mut result = NamedCustomProvider {
                            name: display_name,
                            base_url,
                            api_key: resolved_api_key.clone(),
                            model: non_empty_opt(yaml_str(entry.get("default_model"))),
                            ..Default::default()
                        };
                        result.api_mode = parse_api_mode_value(entry.get("api_mode"))
                            .or_else(|| parse_api_mode_value(entry.get("transport")));
                        return Some(result);
                    }
                }
            }
        }
    }

    // custom_providers: must be a list. A dict is a misconfiguration → warn/None.
    match config.get("custom_providers") {
        Some(YamlValue::Mapping(_)) => {
            log::warn!(
                "custom_providers in config.yaml is a dict, not a list. \
                 Each entry must be prefixed with '-' in YAML. \
                 Run 'hermes doctor' for details."
            );
            return None;
        }
        _ => {}
    }

    let custom_providers = config
        .get("custom_providers")
        .and_then(YamlValue::as_sequence);
    let custom_providers = match custom_providers {
        Some(seq) if !seq.is_empty() => seq,
        _ => return None,
    };

    for entry in custom_providers {
        let entry = match entry.as_mapping() {
            Some(_) => entry,
            None => continue,
        };
        let name = match entry.get("name").and_then(YamlValue::as_str) {
            Some(s) => s,
            None => continue,
        };
        let base_url = match entry.get("base_url").and_then(YamlValue::as_str) {
            Some(s) => s,
            None => continue,
        };
        let name_norm = normalize_custom_provider_name(name);
        let menu_key = format!("custom:{name_norm}");
        let provider_key = yaml_str(entry.get("provider_key"));
        let provider_key_norm = if provider_key.is_empty() {
            String::new()
        } else {
            normalize_custom_provider_name(&provider_key)
        };
        let provider_menu_key = if provider_key_norm.is_empty() {
            String::new()
        } else {
            format!("custom:{provider_key_norm}")
        };

        let matches = requested_norm == name_norm
            || requested_norm == menu_key
            || (!provider_key_norm.is_empty() && requested_norm == provider_key_norm)
            || (!provider_menu_key.is_empty() && requested_norm == provider_menu_key);
        if !matches {
            continue;
        }

        let mut result = NamedCustomProvider {
            name: name.trim().to_string(),
            base_url: base_url.trim().to_string(),
            api_key: yaml_str(entry.get("api_key")),
            ..Default::default()
        };
        let key_env = yaml_str(entry.get("key_env"));
        if !key_env.is_empty() {
            result.key_env = Some(key_env);
        }
        if !provider_key.is_empty() {
            result.provider_key = Some(provider_key);
        }
        result.api_mode = parse_api_mode_value(entry.get("api_mode"));
        result.model = non_empty_opt(yaml_str(entry.get("model")));
        return Some(result);
    }

    None
}

fn first_non_empty(candidates: &[String]) -> String {
    for c in candidates {
        if !c.is_empty() {
            return c.clone();
        }
    }
    String::new()
}

fn non_empty_opt(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// ---------------------------------------------------------------------------
// resolve_requested_provider
// ---------------------------------------------------------------------------

/// `resolve_requested_provider`.
pub fn resolve_requested_provider<E: EnvSource>(
    requested: Option<&str>,
    config: &YamlValue,
    env: &E,
) -> String {
    if let Some(r) = requested {
        if !r.trim().is_empty() {
            return r.trim().to_ascii_lowercase();
        }
    }
    let model_cfg = get_model_config(config);
    let cfg_provider = model_cfg.provider();
    if !cfg_provider.trim().is_empty() {
        return cfg_provider.trim().to_ascii_lowercase();
    }
    let env_provider = env.get_trimmed("HERMES_INFERENCE_PROVIDER").to_ascii_lowercase();
    if !env_provider.is_empty() {
        return env_provider;
    }
    "auto".to_string()
}

// ---------------------------------------------------------------------------
// OpenRouter / custom runtime resolution (pure, env-driven)
// ---------------------------------------------------------------------------

/// `_resolve_openrouter_runtime`.
///
/// Pure: depends only on config + env. The `pool_lookup` closure mirrors
/// `_try_resolve_from_custom_pool` for custom endpoints (returns None when no
/// pool / not wired).
pub fn resolve_openrouter_runtime<E: EnvSource, P>(
    requested_provider: &str,
    explicit_api_key: Option<&str>,
    explicit_base_url: Option<&str>,
    config: &YamlValue,
    env: &E,
    pool_lookup: P,
) -> RuntimeProvider
where
    P: Fn(&str, &str, Option<&str>) -> Option<RuntimeProvider>,
{
    let model_cfg = get_model_config(config);
    let cfg_base_url = model_cfg.base_url();
    let cfg_provider = model_cfg.provider().trim().to_ascii_lowercase();
    let mut cfg_api_key = String::new();
    for k in ["api_key", "api"] {
        if let Some(v) = model_cfg.get_value(k).and_then(YamlValue::as_str) {
            if !v.trim().is_empty() {
                cfg_api_key = v.trim().to_string();
                break;
            }
        }
    }
    let requested_norm = requested_provider.trim().to_ascii_lowercase();

    let env_openrouter_base_url = env.get_trimmed("OPENROUTER_BASE_URL");
    let env_custom_base_url = env.get_trimmed("CUSTOM_BASE_URL");

    let explicit_base_url_clean = explicit_base_url.unwrap_or("").trim().to_string();

    let mut use_config_base_url = false;
    if !cfg_base_url.trim().is_empty() && explicit_base_url_clean.is_empty() {
        if requested_norm == "auto" {
            if cfg_provider.is_empty() || cfg_provider == "auto" {
                use_config_base_url = true;
            }
        } else if requested_norm == "custom"
            && config_base_url_trustworthy_for_bare_custom(&cfg_base_url, &cfg_provider)
        {
            use_config_base_url = true;
        }
    }

    let base_url = rstrip_slash(&first_non_empty(&[
        explicit_base_url_clean.clone(),
        env_custom_base_url,
        if use_config_base_url {
            cfg_base_url.trim().to_string()
        } else {
            String::new()
        },
        env_openrouter_base_url,
        OPENROUTER_BASE_URL.to_string(),
    ]));

    let is_openrouter_url = base_url_host_matches(&base_url, "openrouter.ai");
    let api_key_candidates: Vec<String> = if is_openrouter_url {
        vec![
            explicit_api_key.unwrap_or("").to_string(),
            env.get_trimmed("OPENROUTER_API_KEY"),
            env.get_trimmed("OPENAI_API_KEY"),
        ]
    } else {
        let is_ollama_url = base_url_host_matches(&base_url, "ollama.com");
        vec![
            explicit_api_key.unwrap_or("").to_string(),
            if use_config_base_url {
                cfg_api_key.clone()
            } else {
                String::new()
            },
            if is_ollama_url {
                env.get_trimmed("OLLAMA_API_KEY")
            } else {
                String::new()
            },
            env.get_trimmed("OPENAI_API_KEY"),
            env.get_trimmed("OPENROUTER_API_KEY"),
        ]
    };
    let mut api_key = String::new();
    for candidate in &api_key_candidates {
        if has_usable_secret(candidate) {
            api_key = candidate.trim().to_string();
            break;
        }
    }

    let source = if explicit_api_key.map(|s| !s.is_empty()).unwrap_or(false)
        || !explicit_base_url_clean.is_empty()
    {
        "explicit"
    } else {
        "env/config"
    };

    let effective_provider = if requested_norm == "custom" {
        "custom"
    } else {
        "openrouter"
    };

    if effective_provider == "custom" && !base_url.is_empty() {
        let mode = parse_api_mode_value(model_cfg.get_value("api_mode"));
        if let Some(pool_result) =
            pool_lookup(&base_url, effective_provider, mode.as_deref())
        {
            return pool_result;
        }
    }

    if effective_provider == "custom" && api_key.is_empty() && !is_openrouter_url {
        api_key = "no-key-required".to_string();
    }

    let api_mode = parse_api_mode_value(model_cfg.get_value("api_mode"))
        .or_else(|| detect_api_mode_for_url(&base_url).map(str::to_string))
        .unwrap_or_else(|| "chat_completions".to_string());

    RuntimeProvider::base(effective_provider, &api_mode, &base_url, &api_key, source)
}

/// `_resolve_azure_foundry_runtime`.
///
/// `api_key_lookup` mirrors `get_env_value("AZURE_FOUNDRY_API_KEY")` (the
/// `.env`-file lookup). `model_api_mode_for` mirrors
/// `azure_foundry_model_api_mode(model)` (None when no inference).
pub fn resolve_azure_foundry_runtime<E: EnvSource, K, M>(
    requested_provider: &str,
    model_cfg: &ModelConfig,
    explicit_api_key: Option<&str>,
    explicit_base_url: Option<&str>,
    target_model: Option<&str>,
    env: &E,
    api_key_lookup: K,
    model_api_mode_for: M,
) -> Result<RuntimeProvider, ResolveError>
where
    K: Fn() -> Option<String>,
    M: Fn(&str) -> Option<String>,
{
    let explicit_api_key = explicit_api_key.unwrap_or("").trim().to_string();
    let explicit_base_url_clean = rstrip_slash(explicit_base_url.unwrap_or("").trim());

    let cfg_provider = model_cfg.provider().trim().to_ascii_lowercase();
    let mut cfg_base_url = String::new();
    let mut cfg_api_mode = "chat_completions".to_string();
    if cfg_provider == "azure-foundry" {
        cfg_base_url = rstrip_slash(model_cfg.base_url().trim());
        cfg_api_mode = parse_api_mode_value(model_cfg.get_value("api_mode"))
            .unwrap_or_else(|| "chat_completions".to_string());
    }

    let effective_model = target_model
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| model_cfg.default_model())
        .trim()
        .to_string();
    if !effective_model.is_empty() && cfg_api_mode != "anthropic_messages" {
        if let Some(inferred) = model_api_mode_for(&effective_model) {
            cfg_api_mode = inferred;
        }
    }

    let env_base_url = rstrip_slash(&env.get_trimmed("AZURE_FOUNDRY_BASE_URL"));
    let mut base_url = first_non_empty(&[explicit_base_url_clean.clone(), cfg_base_url, env_base_url]);
    if base_url.is_empty() {
        return Err(AuthError::new(
            "Azure Foundry requires a base URL. Set it via 'hermes model' or \
             the AZURE_FOUNDRY_BASE_URL environment variable.",
        )
        .into());
    }

    let mut api_key = explicit_api_key.clone();
    if api_key.is_empty() {
        api_key = api_key_lookup().unwrap_or_default();
    }
    if api_key.is_empty() {
        api_key = env.get_trimmed("AZURE_FOUNDRY_API_KEY");
    }
    if api_key.is_empty() {
        return Err(AuthError::new(
            "Azure Foundry requires an API key. Set AZURE_FOUNDRY_API_KEY in \
             ~/.hermes/.env or run 'hermes model' to configure.",
        )
        .into());
    }

    if cfg_api_mode == "anthropic_messages" {
        base_url = strip_trailing_v1(&base_url);
    }

    let source = if !explicit_api_key.is_empty() || explicit_base_url.is_some() {
        "explicit"
    } else {
        "config"
    };
    let mut runtime =
        RuntimeProvider::base("azure-foundry", &cfg_api_mode, &base_url, &api_key, source);
    runtime.requested_provider = Some(requested_provider.to_string());
    Ok(runtime)
}

// ---------------------------------------------------------------------------
// Named custom runtime resolution (pure + pool hook)
// ---------------------------------------------------------------------------

/// `_resolve_named_custom_runtime`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_named_custom_runtime<E: EnvSource, P, F>(
    requested_provider: &str,
    explicit_api_key: Option<&str>,
    explicit_base_url: Option<&str>,
    config: &YamlValue,
    env: &E,
    pool_lookup: P,
    canonical_for: F,
) -> Option<RuntimeProvider>
where
    P: Fn(&str, &str, Option<&str>) -> Option<RuntimeProvider>,
    F: Fn(&str) -> Option<String>,
{
    let requested_norm = requested_provider.trim().to_ascii_lowercase();
    let explicit_base_url_clean = explicit_base_url.unwrap_or("").trim().to_string();

    if requested_norm == "custom" && !explicit_base_url_clean.is_empty() {
        let base_url = rstrip_slash(explicit_base_url_clean.trim());
        let candidates = [
            explicit_api_key.unwrap_or("").trim().to_string(),
            env.get_trimmed("OPENAI_API_KEY"),
            env.get_trimmed("OPENROUTER_API_KEY"),
        ];
        let mut api_key = String::new();
        for c in &candidates {
            if has_usable_secret(c) {
                api_key = c.clone();
                break;
            }
        }
        if api_key.is_empty() {
            api_key = "no-key-required".to_string();
        }
        let api_mode = detect_api_mode_for_url(&base_url)
            .map(str::to_string)
            .unwrap_or_else(|| "chat_completions".to_string());
        let mut rt = RuntimeProvider::base("custom", &api_mode, &base_url, &api_key, "direct-alias");
        rt.requested_provider = Some(requested_provider.to_string());
        return Some(rt);
    }

    let custom_provider = get_named_custom_provider(requested_provider, config, env, canonical_for)?;

    let base_url = rstrip_slash(&first_non_empty(&[
        explicit_base_url_clean.clone(),
        custom_provider.base_url.clone(),
    ]));
    if base_url.is_empty() {
        return None;
    }

    if let Some(mut pool_result) =
        pool_lookup(&base_url, "custom", custom_provider.api_mode.as_deref())
    {
        if let Some(model_name) = &custom_provider.model {
            pool_result.model = Some(model_name.clone());
        }
        return Some(pool_result);
    }

    let key_env_value = custom_provider
        .key_env
        .as_deref()
        .map(|k| env.get_trimmed(k))
        .unwrap_or_default();
    let candidates = [
        explicit_api_key.unwrap_or("").trim().to_string(),
        custom_provider.api_key.clone(),
        key_env_value,
        env.get_trimmed("OPENAI_API_KEY"),
        env.get_trimmed("OPENROUTER_API_KEY"),
    ];
    let mut api_key = String::new();
    for c in &candidates {
        if has_usable_secret(c) {
            api_key = c.clone();
            break;
        }
    }
    if api_key.is_empty() {
        api_key = "no-key-required".to_string();
    }

    let api_mode = custom_provider
        .api_mode
        .clone()
        .or_else(|| detect_api_mode_for_url(&base_url).map(str::to_string))
        .unwrap_or_else(|| "chat_completions".to_string());

    let source = format!("custom_provider:{}", custom_provider.name);
    let mut result = RuntimeProvider::base("custom", &api_mode, &base_url, &api_key, &source);
    if let Some(model) = custom_provider.model {
        result.model = Some(model);
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// Backend trait for credential-fetching / network-bound paths.
// ---------------------------------------------------------------------------

/// Provider-specific copilot api_mode resolution. Mirrors
/// `_copilot_runtime_api_mode`. Implemented by the backend because it may query
/// `copilot_model_api_mode(model, api_key)`.
pub trait RuntimeBackend {
    /// Mirrors `_copilot_runtime_api_mode`.
    fn copilot_runtime_api_mode(&self, model_cfg: &ModelConfig, api_key: &str) -> String {
        let configured_provider = model_cfg.provider().trim().to_ascii_lowercase();
        let configured_mode = parse_api_mode_value(model_cfg.get_value("api_mode"));
        if let Some(mode) = &configured_mode {
            if provider_supports_explicit_api_mode(Some("copilot"), Some(&configured_provider)) {
                return mode.clone();
            }
        }
        let model_name = model_cfg.default_model().trim().to_string();
        if model_name.is_empty() {
            return "chat_completions".to_string();
        }
        self.copilot_model_api_mode(&model_name, api_key)
            .unwrap_or_else(|| "chat_completions".to_string())
    }

    /// `hermes_cli.models.copilot_model_api_mode`.
    fn copilot_model_api_mode(&self, _model: &str, _api_key: &str) -> Option<String> {
        None
    }

    /// `hermes_cli.models.azure_foundry_model_api_mode`.
    fn azure_foundry_model_api_mode(&self, _model: &str) -> Option<String> {
        None
    }

    /// `hermes_cli.models.opencode_model_api_mode`.
    fn opencode_model_api_mode(&self, _provider: &str, _model: &str) -> String {
        "chat_completions".to_string()
    }
}

/// A backend with no model-family inference (defaults only). Useful for tests
/// and for the pure resolution paths.
#[derive(Debug, Clone, Default)]
pub struct NoInferenceBackend;

impl RuntimeBackend for NoInferenceBackend {}

// ---------------------------------------------------------------------------
// Pool-entry runtime resolution (faithful port of _resolve_runtime_from_pool_entry)
// ---------------------------------------------------------------------------

/// Inputs extracted from a `PooledCredential` for runtime resolution. Mirrors
/// the `getattr` reads in `_resolve_runtime_from_pool_entry`.
#[derive(Debug, Clone, Default)]
pub struct PoolEntryView {
    pub runtime_base_url: Option<String>,
    pub base_url: Option<String>,
    pub runtime_api_key: Option<String>,
    pub access_token: String,
    pub source: Option<String>,
}

impl PoolEntryView {
    /// Build a view from a `crate::ag_credential_pool::PooledCredential`.
    pub fn from_pooled(entry: &crate::ag_credential_pool::PooledCredential) -> Self {
        PoolEntryView {
            runtime_base_url: entry.runtime_base_url(),
            base_url: entry.base_url.clone(),
            runtime_api_key: {
                let k = entry.runtime_api_key();
                if k.is_empty() { None } else { Some(k) }
            },
            access_token: entry.access_token.clone(),
            source: Some(entry.source.clone()),
        }
    }
}

/// `_resolve_runtime_from_pool_entry`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_runtime_from_pool_entry<B: RuntimeBackend>(
    provider: &str,
    entry: &PoolEntryView,
    requested_provider: &str,
    model_cfg: &ModelConfig,
    target_model: Option<&str>,
    backend: &B,
) -> RuntimeProvider {
    let effective_model = target_model
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| model_cfg.default_model());

    let mut base_url = rstrip_slash(
        &entry
            .runtime_base_url
            .clone()
            .or_else(|| entry.base_url.clone())
            .unwrap_or_default(),
    );
    let api_key = entry
        .runtime_api_key
        .clone()
        .unwrap_or_else(|| entry.access_token.clone());
    let mut api_mode = "chat_completions".to_string();

    match provider {
        "openai-codex" => {
            api_mode = "codex_responses".to_string();
            if base_url.is_empty() {
                base_url = DEFAULT_CODEX_BASE_URL.to_string();
            }
        }
        "qwen-oauth" => {
            api_mode = "chat_completions".to_string();
            if base_url.is_empty() {
                base_url = DEFAULT_QWEN_BASE_URL.to_string();
            }
        }
        "google-gemini-cli" => {
            api_mode = "chat_completions".to_string();
            if base_url.is_empty() {
                base_url = "cloudcode-pa://google".to_string();
            }
        }
        "anthropic" => {
            api_mode = "anthropic_messages".to_string();
            let cfg_provider = model_cfg.provider().trim().to_ascii_lowercase();
            let mut cfg_base_url = String::new();
            if cfg_provider == "anthropic" {
                cfg_base_url = rstrip_slash(model_cfg.base_url().trim());
            }
            base_url = first_non_empty(&[
                cfg_base_url,
                base_url.clone(),
                "https://api.anthropic.com".to_string(),
            ]);
        }
        "openrouter" => {
            if base_url.is_empty() {
                base_url = OPENROUTER_BASE_URL.to_string();
            }
        }
        "xai" => {
            api_mode = "codex_responses".to_string();
        }
        "nous" => {
            api_mode = "chat_completions".to_string();
        }
        "copilot" => {
            let runtime_key = entry.runtime_api_key.clone().unwrap_or_default();
            api_mode = backend.copilot_runtime_api_mode(model_cfg, &runtime_key);
            if base_url.is_empty() {
                if let Some(pconfig) = registry_lookup("copilot") {
                    base_url = pconfig.inference_base_url;
                }
            }
        }
        "azure-foundry" => {
            let cfg_provider = model_cfg.provider().trim().to_ascii_lowercase();
            if cfg_provider == "azure-foundry" {
                let cfg_base_url = rstrip_slash(model_cfg.base_url().trim());
                if !cfg_base_url.is_empty() {
                    base_url = cfg_base_url;
                }
                if let Some(configured_mode) = parse_api_mode_value(model_cfg.get_value("api_mode")) {
                    api_mode = configured_mode;
                }
            }
            if !effective_model.is_empty() && api_mode != "anthropic_messages" {
                if let Some(inferred) = backend.azure_foundry_model_api_mode(&effective_model) {
                    api_mode = inferred;
                }
            }
            if api_mode == "anthropic_messages" {
                base_url = strip_trailing_v1(&base_url);
            }
        }
        _ => {
            let configured_provider = model_cfg.provider().trim().to_ascii_lowercase();
            let pconfig = registry_lookup(provider);
            let pool_url_is_default = pconfig
                .as_ref()
                .map(|p| rstrip_slash(&base_url) == rstrip_slash(&p.inference_base_url))
                .unwrap_or(false);
            if configured_provider == provider && pool_url_is_default {
                let cfg_base_url = rstrip_slash(model_cfg.base_url().trim());
                if !cfg_base_url.is_empty() {
                    base_url = cfg_base_url;
                }
            }
            let configured_mode = parse_api_mode_value(model_cfg.get_value("api_mode"));
            if provider == "opencode-zen" || provider == "opencode-go" {
                api_mode = backend.opencode_model_api_mode(provider, &effective_model);
            } else if let Some(mode) = &configured_mode {
                if provider_supports_explicit_api_mode(Some(provider), Some(&configured_provider)) {
                    api_mode = mode.clone();
                } else if let Some(detected) = detect_api_mode_for_url(&base_url) {
                    api_mode = detected.to_string();
                }
            } else if let Some(detected) = detect_api_mode_for_url(&base_url) {
                api_mode = detected.to_string();
            }
        }
    }

    if api_mode == "anthropic_messages"
        && (provider == "opencode-zen" || provider == "opencode-go")
    {
        base_url = strip_trailing_v1(&base_url);
    }

    let source = entry.source.clone().unwrap_or_else(|| "pool".to_string());
    let mut rt = RuntimeProvider::base(provider, &api_mode, &base_url, &api_key, &source);
    rt.requested_provider = Some(requested_provider.to_string());
    rt
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value as Y;

    fn yaml(s: &str) -> Y {
        serde_yaml::from_str(s).unwrap()
    }

    fn no_pool(_b: &str, _p: &str, _m: Option<&str>) -> Option<RuntimeProvider> {
        None
    }

    fn no_canonical(_n: &str) -> Option<String> {
        None
    }

    #[test]
    fn test_normalize_custom_provider_name() {
        assert_eq!(normalize_custom_provider_name("  My Provider "), "my-provider");
        assert_eq!(normalize_custom_provider_name("Custom:Local"), "custom:local");
    }

    #[test]
    fn test_loopback_hostname() {
        assert!(loopback_hostname("localhost"));
        assert!(loopback_hostname("127.0.0.1."));
        assert!(loopback_hostname("::1"));
        assert!(!loopback_hostname("openrouter.ai"));
    }

    #[test]
    fn test_detect_api_mode_for_url() {
        assert_eq!(detect_api_mode_for_url("https://api.openai.com/v1"), Some("codex_responses"));
        assert_eq!(detect_api_mode_for_url("https://api.x.ai/v1"), Some("codex_responses"));
        assert_eq!(
            detect_api_mode_for_url("https://api.minimax.io/anthropic"),
            Some("anthropic_messages")
        );
        assert_eq!(
            detect_api_mode_for_url("https://api.kimi.com/coding"),
            Some("anthropic_messages")
        );
        assert_eq!(detect_api_mode_for_url("https://openrouter.ai/api/v1"), None);
    }

    #[test]
    fn test_parse_api_mode() {
        assert_eq!(parse_api_mode(Some(" Chat_Completions ")), Some("chat_completions".into()));
        assert_eq!(parse_api_mode(Some("bedrock_converse")), Some("bedrock_converse".into()));
        assert_eq!(parse_api_mode(Some("garbage")), None);
        assert_eq!(parse_api_mode(None), None);
    }

    #[test]
    fn test_provider_supports_explicit_api_mode() {
        assert!(provider_supports_explicit_api_mode(Some("nous"), None));
        assert!(provider_supports_explicit_api_mode(Some("nous"), Some("")));
        assert!(provider_supports_explicit_api_mode(Some("nous"), Some("nous")));
        assert!(!provider_supports_explicit_api_mode(Some("nous"), Some("openrouter")));
        assert!(provider_supports_explicit_api_mode(Some("custom"), Some("custom:local")));
        assert!(!provider_supports_explicit_api_mode(Some("custom"), Some("openrouter")));
    }

    #[test]
    fn test_has_usable_secret() {
        assert!(has_usable_secret("sk-abc123"));
        assert!(!has_usable_secret(""));
        assert!(!has_usable_secret("   "));
        assert!(!has_usable_secret("no-key-required"));
        assert!(!has_usable_secret("CHANGEME"));
    }

    #[test]
    fn test_config_base_url_trustworthy_for_bare_custom() {
        assert!(config_base_url_trustworthy_for_bare_custom("http://localhost:1234/v1", ""));
        assert!(config_base_url_trustworthy_for_bare_custom("https://anything", "custom"));
        assert!(!config_base_url_trustworthy_for_bare_custom("https://openrouter.ai/api/v1", ""));
        assert!(!config_base_url_trustworthy_for_bare_custom("https://api.z.ai/v1", ""));
        assert!(!config_base_url_trustworthy_for_bare_custom("", ""));
    }

    #[test]
    fn test_strip_trailing_v1() {
        assert_eq!(strip_trailing_v1("https://x/v1"), "https://x");
        assert_eq!(strip_trailing_v1("https://x/v1/"), "https://x");
        assert_eq!(strip_trailing_v1("https://x/v2"), "https://x/v2");
    }

    #[test]
    fn test_get_model_config_dict_with_model_alias() {
        let cfg = yaml("model:\n  model: gpt-4o\n  provider: openai\n");
        let mc = get_model_config(&cfg);
        assert_eq!(mc.default_model(), "gpt-4o");
        assert_eq!(mc.provider(), "openai");
    }

    #[test]
    fn test_get_model_config_string() {
        let cfg = yaml("model: claude-sonnet\n");
        let mc = get_model_config(&cfg);
        assert_eq!(mc.default_model(), "claude-sonnet");
    }

    #[test]
    fn test_resolve_requested_provider_explicit() {
        let cfg = yaml("{}");
        let env = MapEnv::new();
        assert_eq!(resolve_requested_provider(Some(" OpenRouter "), &cfg, &env), "openrouter");
    }

    #[test]
    fn test_resolve_requested_provider_config_then_env_then_auto() {
        let env = MapEnv::new().set("HERMES_INFERENCE_PROVIDER", "nous");
        let cfg_with_provider = yaml("model:\n  provider: anthropic\n");
        assert_eq!(resolve_requested_provider(None, &cfg_with_provider, &env), "anthropic");

        let empty = yaml("{}");
        assert_eq!(resolve_requested_provider(None, &empty, &env), "nous");

        let no_env = MapEnv::new();
        assert_eq!(resolve_requested_provider(None, &empty, &no_env), "auto");
    }

    #[test]
    fn test_named_custom_provider_new_style_dict() {
        let cfg = yaml(
            "providers:\n  mylocal:\n    api: http://localhost:8000/v1\n    api_key: secret\n    default_model: qwen\n    transport: anthropic_messages\n",
        );
        let env = MapEnv::new();
        let result = get_named_custom_provider("mylocal", &cfg, &env, no_canonical).unwrap();
        assert_eq!(result.base_url, "http://localhost:8000/v1");
        assert_eq!(result.api_key, "secret");
        assert_eq!(result.model.as_deref(), Some("qwen"));
        assert_eq!(result.api_mode.as_deref(), Some("anthropic_messages"));
    }

    #[test]
    fn test_named_custom_provider_key_env() {
        let cfg = yaml(
            "providers:\n  mylocal:\n    api: http://localhost:8000/v1\n    key_env: MY_KEY\n",
        );
        let env = MapEnv::new().set("MY_KEY", "from-env");
        let result = get_named_custom_provider("mylocal", &cfg, &env, no_canonical).unwrap();
        assert_eq!(result.api_key, "from-env");
    }

    #[test]
    fn test_named_custom_provider_legacy_list() {
        let cfg = yaml(
            "custom_providers:\n  - name: Z AI\n    base_url: https://api.z.ai/v1\n    api_key: zkey\n    api_mode: chat_completions\n    model: glm-4\n",
        );
        let env = MapEnv::new();
        // matches normalized name "z-ai"
        let result = get_named_custom_provider("z-ai", &cfg, &env, no_canonical).unwrap();
        assert_eq!(result.name, "Z AI");
        assert_eq!(result.base_url, "https://api.z.ai/v1");
        assert_eq!(result.api_key, "zkey");
        assert_eq!(result.model.as_deref(), Some("glm-4"));
    }

    #[test]
    fn test_named_custom_provider_does_not_shadow_builtin() {
        let cfg = yaml(
            "providers:\n  nous:\n    api: http://localhost/v1\n",
        );
        let env = MapEnv::new();
        // canonical_for returns the same canonical name -> built-in wins -> None
        let result = get_named_custom_provider("nous", &cfg, &env, |n| Some(n.to_string()));
        assert!(result.is_none());
    }

    #[test]
    fn test_named_custom_provider_dict_misconfig_warns() {
        let cfg = yaml("custom_providers:\n  foo: bar\n");
        let env = MapEnv::new();
        assert!(get_named_custom_provider("foo", &cfg, &env, no_canonical).is_none());
    }

    #[test]
    fn test_resolve_openrouter_runtime_default() {
        let cfg = yaml("{}");
        let env = MapEnv::new().set("OPENROUTER_API_KEY", "or-key");
        let rt = resolve_openrouter_runtime("auto", None, None, &cfg, &env, no_pool);
        assert_eq!(rt.provider, "openrouter");
        assert_eq!(rt.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(rt.api_key, "or-key");
        assert_eq!(rt.api_mode, "chat_completions");
        assert_eq!(rt.source, "env/config");
    }

    #[test]
    fn test_resolve_openrouter_runtime_custom_no_key() {
        let cfg = yaml("{}");
        let env = MapEnv::new();
        let rt = resolve_openrouter_runtime(
            "custom",
            None,
            Some("http://localhost:8080/v1"),
            &cfg,
            &env,
            no_pool,
        );
        assert_eq!(rt.provider, "custom");
        assert_eq!(rt.base_url, "http://localhost:8080/v1");
        assert_eq!(rt.api_key, "no-key-required");
        assert_eq!(rt.source, "explicit");
    }

    #[test]
    fn test_resolve_openrouter_runtime_custom_endpoint_openai_key() {
        let cfg = yaml("{}");
        let env = MapEnv::new()
            .set("OPENROUTER_API_KEY", "or-key")
            .set("OPENAI_API_KEY", "oa-key");
        // Custom (non-openrouter) endpoint should prefer OPENAI_API_KEY.
        let rt = resolve_openrouter_runtime(
            "custom",
            None,
            Some("https://api.z.ai/v1"),
            &cfg,
            &env,
            no_pool,
        );
        assert_eq!(rt.api_key, "oa-key");
    }

    #[test]
    fn test_resolve_named_custom_runtime_direct_alias() {
        let cfg = yaml("{}");
        let env = MapEnv::new();
        let rt = resolve_named_custom_runtime(
            "custom",
            None,
            Some("https://api.kimi.com/coding"),
            &cfg,
            &env,
            no_pool,
            no_canonical,
        )
        .unwrap();
        assert_eq!(rt.provider, "custom");
        assert_eq!(rt.api_mode, "anthropic_messages");
        assert_eq!(rt.source, "direct-alias");
        assert_eq!(rt.api_key, "no-key-required");
    }

    #[test]
    fn test_resolve_named_custom_runtime_from_config() {
        let cfg = yaml(
            "providers:\n  mylocal:\n    api: http://localhost:8000/v1\n    api_key: secret\n    default_model: qwen\n",
        );
        let env = MapEnv::new();
        let rt = resolve_named_custom_runtime(
            "mylocal", None, None, &cfg, &env, no_pool, no_canonical,
        )
        .unwrap();
        assert_eq!(rt.provider, "custom");
        assert_eq!(rt.base_url, "http://localhost:8000/v1");
        assert_eq!(rt.api_key, "secret");
        assert_eq!(rt.model.as_deref(), Some("qwen"));
        assert_eq!(rt.source, "custom_provider:mylocal");
    }

    #[test]
    fn test_azure_foundry_requires_base_url() {
        let env = MapEnv::new();
        let mc = ModelConfig::default();
        let err = resolve_azure_foundry_runtime(
            "azure-foundry",
            &mc,
            None,
            None,
            None,
            &env,
            || None,
            |_| None,
        )
        .unwrap_err();
        match err {
            ResolveError::Auth(e) => assert!(e.message.contains("base URL")),
        }
    }

    #[test]
    fn test_azure_foundry_resolves_and_strips_v1() {
        let cfg = yaml(
            "model:\n  provider: azure-foundry\n  base_url: https://x.azure.com/v1\n  api_mode: anthropic_messages\n",
        );
        let mc = get_model_config(&cfg);
        let env = MapEnv::new().set("AZURE_FOUNDRY_API_KEY", "az-key");
        let rt = resolve_azure_foundry_runtime(
            "azure-foundry",
            &mc,
            None,
            None,
            None,
            &env,
            || None,
            |_| None,
        )
        .unwrap();
        assert_eq!(rt.provider, "azure-foundry");
        assert_eq!(rt.api_mode, "anthropic_messages");
        assert_eq!(rt.base_url, "https://x.azure.com");
        assert_eq!(rt.api_key, "az-key");
        assert_eq!(rt.source, "config");
    }

    #[test]
    fn test_pool_entry_anthropic() {
        let entry = PoolEntryView {
            runtime_api_key: Some("ak".into()),
            access_token: "ak".into(),
            base_url: Some("https://api.anthropic.com/".into()),
            source: Some("pool".into()),
            ..Default::default()
        };
        let mc = ModelConfig::default();
        let backend = NoInferenceBackend;
        let rt = resolve_runtime_from_pool_entry(
            "anthropic",
            &entry,
            "anthropic",
            &mc,
            None,
            &backend,
        );
        assert_eq!(rt.api_mode, "anthropic_messages");
        assert_eq!(rt.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn test_pool_entry_codex_default_base() {
        let entry = PoolEntryView {
            runtime_api_key: Some("ck".into()),
            access_token: "ck".into(),
            source: Some("pool".into()),
            ..Default::default()
        };
        let mc = ModelConfig::default();
        let backend = NoInferenceBackend;
        let rt = resolve_runtime_from_pool_entry(
            "openai-codex",
            &entry,
            "openai-codex",
            &mc,
            None,
            &backend,
        );
        assert_eq!(rt.api_mode, "codex_responses");
        assert_eq!(rt.base_url, DEFAULT_CODEX_BASE_URL);
    }

    #[test]
    fn test_pool_entry_opencode_strips_v1_for_anthropic() {
        struct B;
        impl RuntimeBackend for B {
            fn opencode_model_api_mode(&self, _p: &str, _m: &str) -> String {
                "anthropic_messages".to_string()
            }
        }
        let entry = PoolEntryView {
            runtime_api_key: Some("k".into()),
            access_token: "k".into(),
            base_url: Some("https://opencode.ai/zen/v1".into()),
            source: Some("pool".into()),
            ..Default::default()
        };
        let mc = ModelConfig::default();
        let rt = resolve_runtime_from_pool_entry(
            "opencode-zen",
            &entry,
            "opencode-zen",
            &mc,
            None,
            &B,
        );
        assert_eq!(rt.api_mode, "anthropic_messages");
        assert_eq!(rt.base_url, "https://opencode.ai/zen");
    }

    #[test]
    fn test_format_runtime_provider_error() {
        let e = ResolveError::Auth(AuthError::new("boom"));
        assert_eq!(format_runtime_provider_error(&e), "boom");
    }
}
