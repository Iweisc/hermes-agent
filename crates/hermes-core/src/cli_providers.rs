//! Single source of truth for provider identity in Hermes Agent.
//!
//! Faithful native Rust port of `hermes_cli/providers.py`.
//!
//! Two/three data sources, merged at runtime:
//!
//! 1. **models.dev catalog** — providers with base URLs, env vars, display
//!    names, and full model metadata. This is the primary database (accessed
//!    via [`crate::ag_models_dev::get_provider_info`]).
//! 2. **Hermes overlays** — transport type, auth patterns, aggregator flags,
//!    and additional env vars that models.dev doesn't track. Small map,
//!    maintained here ([`HERMES_OVERLAYS`]).
//! 3. **User config** (`providers:` / `custom_providers:` sections in
//!    config.yaml) — user-defined endpoints and overrides, merged on top.
//!
//! Other modules import from this file. No parallel registries.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::mod_utils::{base_url_host_matches, base_url_hostname};

// -- Hermes overlay ----------------------------------------------------------
// Hermes-specific metadata that models.dev doesn't provide.

/// Hermes-specific provider metadata layered on top of models.dev.
///
/// Mirrors the frozen `HermesOverlay` dataclass. All fields use static string
/// slices because the overlay table is a compile-time constant set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermesOverlay {
    /// `openai_chat` | `anthropic_messages` | `codex_responses` | ...
    pub transport: &'static str,
    pub is_aggregator: bool,
    /// `api_key` | `oauth_device_code` | `oauth_external` | `external_process` | ...
    pub auth_type: &'static str,
    /// env vars models.dev doesn't list
    pub extra_env_vars: &'static [&'static str],
    /// override if models.dev URL is wrong/missing
    pub base_url_override: &'static str,
    /// env var for user-custom base URL
    pub base_url_env_var: &'static str,
}

impl HermesOverlay {
    /// Matches the Python dataclass defaults.
    const fn new() -> Self {
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: false,
            auth_type: "api_key",
            extra_env_vars: &[],
            base_url_override: "",
            base_url_env_var: "",
        }
    }
}

impl Default for HermesOverlay {
    fn default() -> Self {
        HermesOverlay::new()
    }
}

/// Build the overlay table (id -> overlay). Lazily initialised, cached.
pub fn hermes_overlays() -> &'static HashMap<&'static str, HermesOverlay> {
    static OVERLAYS: OnceLock<HashMap<&'static str, HermesOverlay>> = OnceLock::new();
    OVERLAYS.get_or_init(build_overlays)
}

fn build_overlays() -> HashMap<&'static str, HermesOverlay> {
    let mut m: HashMap<&'static str, HermesOverlay> = HashMap::new();

    m.insert(
        "openrouter",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            extra_env_vars: &["OPENAI_API_KEY"],
            base_url_env_var: "OPENROUTER_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "nous",
        HermesOverlay {
            transport: "openai_chat",
            auth_type: "oauth_device_code",
            base_url_override: "https://inference-api.nousresearch.com/v1",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "openai-codex",
        HermesOverlay {
            transport: "codex_responses",
            auth_type: "oauth_external",
            base_url_override: "https://chatgpt.com/backend-api/codex",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "qwen-oauth",
        HermesOverlay {
            transport: "openai_chat",
            auth_type: "oauth_external",
            base_url_override: "https://portal.qwen.ai/v1",
            base_url_env_var: "HERMES_QWEN_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "google-gemini-cli",
        HermesOverlay {
            transport: "openai_chat",
            auth_type: "oauth_external",
            base_url_override: "cloudcode-pa://google",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "lmstudio",
        HermesOverlay {
            transport: "openai_chat",
            auth_type: "api_key",
            extra_env_vars: &["LM_API_KEY"],
            base_url_override: "http://127.0.0.1:1234/v1",
            base_url_env_var: "LM_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "copilot-acp",
        HermesOverlay {
            transport: "codex_responses",
            auth_type: "external_process",
            base_url_override: "acp://copilot",
            base_url_env_var: "COPILOT_ACP_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "github-copilot",
        HermesOverlay {
            transport: "openai_chat",
            extra_env_vars: &["COPILOT_GITHUB_TOKEN", "GH_TOKEN"],
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "anthropic",
        HermesOverlay {
            transport: "anthropic_messages",
            extra_env_vars: &["ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"],
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "zai",
        HermesOverlay {
            transport: "openai_chat",
            extra_env_vars: &["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"],
            base_url_env_var: "GLM_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "kimi-for-coding",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "KIMI_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "stepfun",
        HermesOverlay {
            transport: "openai_chat",
            extra_env_vars: &["STEPFUN_API_KEY"],
            base_url_override: "https://api.stepfun.ai/step_plan/v1",
            base_url_env_var: "STEPFUN_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "minimax",
        HermesOverlay {
            transport: "anthropic_messages",
            base_url_env_var: "MINIMAX_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "minimax-oauth",
        HermesOverlay {
            transport: "anthropic_messages",
            auth_type: "oauth_external",
            base_url_override: "https://api.minimax.io/anthropic",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "minimax-cn",
        HermesOverlay {
            transport: "anthropic_messages",
            base_url_env_var: "MINIMAX_CN_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "deepseek",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "DEEPSEEK_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "alibaba",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "DASHSCOPE_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "alibaba-coding-plan",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "ALIBABA_CODING_PLAN_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "vercel",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "opencode",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            base_url_env_var: "OPENCODE_ZEN_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "opencode-go",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            base_url_env_var: "OPENCODE_GO_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "kilo",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            base_url_env_var: "KILOCODE_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "huggingface",
        HermesOverlay {
            transport: "openai_chat",
            is_aggregator: true,
            base_url_env_var: "HF_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "xai",
        HermesOverlay {
            transport: "codex_responses",
            base_url_override: "https://api.x.ai/v1",
            base_url_env_var: "XAI_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "nvidia",
        HermesOverlay {
            transport: "openai_chat",
            base_url_override: "https://integrate.api.nvidia.com/v1",
            base_url_env_var: "NVIDIA_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "xiaomi",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "XIAOMI_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "tencent-tokenhub",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "TOKENHUB_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "arcee",
        HermesOverlay {
            transport: "openai_chat",
            base_url_override: "https://api.arcee.ai/api/v1",
            base_url_env_var: "ARCEE_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "gmi",
        HermesOverlay {
            transport: "openai_chat",
            extra_env_vars: &["GMI_API_KEY"],
            base_url_override: "https://api.gmi-serving.com/v1",
            base_url_env_var: "GMI_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "ollama-cloud",
        HermesOverlay {
            transport: "openai_chat",
            base_url_env_var: "OLLAMA_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    // Azure Foundry: supports both OpenAI-style and Anthropic-style endpoints.
    // The transport is determined at runtime from config.yaml model.api_mode.
    m.insert(
        "azure-foundry",
        HermesOverlay {
            transport: "openai_chat", // default; overridden by api_mode in config
            base_url_env_var: "AZURE_FOUNDRY_BASE_URL",
            ..HermesOverlay::new()
        },
    );
    m.insert(
        "bedrock",
        HermesOverlay {
            transport: "bedrock_converse",
            auth_type: "aws_sdk",
            ..HermesOverlay::new()
        },
    );

    m
}

// -- Resolved provider -------------------------------------------------------
// The merged result of models.dev + overlay + user config.

/// Complete provider definition — merged from all sources.
///
/// Mirrors the `ProviderDef` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDef {
    pub id: String,
    pub name: String,
    /// `openai_chat` | `anthropic_messages` | `codex_responses` | ...
    pub transport: String,
    /// all env vars to check for API key
    pub api_key_env_vars: Vec<String>,
    pub base_url: String,
    pub base_url_env_var: String,
    pub is_aggregator: bool,
    pub auth_type: String,
    pub doc: String,
    /// `"models.dev"`, `"hermes"`, `"user-config"`
    pub source: String,
}

impl Default for ProviderDef {
    fn default() -> Self {
        ProviderDef {
            id: String::new(),
            name: String::new(),
            transport: String::new(),
            api_key_env_vars: Vec::new(),
            base_url: String::new(),
            base_url_env_var: String::new(),
            is_aggregator: false,
            auth_type: "api_key".to_string(),
            doc: String::new(),
            source: String::new(),
        }
    }
}

// -- Aliases ------------------------------------------------------------------
// Maps human-friendly / legacy names to canonical provider IDs.
// Uses models.dev IDs where possible.

/// (alias, canonical) pairs. Faithful copy of the Python `ALIASES` dict order.
const ALIAS_PAIRS: &[(&str, &str)] = &[
    // openrouter
    ("openai", "openrouter"), // bare "openai" → route through aggregator
    // zai
    ("glm", "zai"),
    ("z-ai", "zai"),
    ("z.ai", "zai"),
    ("zhipu", "zai"),
    // xai
    ("x-ai", "xai"),
    ("x.ai", "xai"),
    ("grok", "xai"),
    // nvidia
    ("nim", "nvidia"),
    ("nvidia-nim", "nvidia"),
    ("build-nvidia", "nvidia"),
    ("nemotron", "nvidia"),
    // kimi-for-coding (models.dev ID)
    ("kimi", "kimi-for-coding"),
    ("kimi-coding", "kimi-for-coding"),
    ("kimi-coding-cn", "kimi-for-coding"),
    ("moonshot", "kimi-for-coding"),
    // stepfun
    ("step", "stepfun"),
    ("stepfun-coding-plan", "stepfun"),
    // minimax-cn
    ("minimax-china", "minimax-cn"),
    ("minimax_cn", "minimax-cn"),
    // anthropic
    ("claude", "anthropic"),
    ("claude-code", "anthropic"),
    // github-copilot (models.dev ID)
    ("copilot", "github-copilot"),
    ("github", "github-copilot"),
    ("github-copilot-acp", "copilot-acp"),
    // vercel (models.dev ID for AI Gateway)
    ("ai-gateway", "vercel"),
    ("aigateway", "vercel"),
    ("vercel-ai-gateway", "vercel"),
    // opencode (models.dev ID for OpenCode Zen)
    ("opencode-zen", "opencode"),
    ("zen", "opencode"),
    // opencode-go
    ("go", "opencode-go"),
    ("opencode-go-sub", "opencode-go"),
    // kilo (models.dev ID for KiloCode)
    ("kilocode", "kilo"),
    ("kilo-code", "kilo"),
    ("kilo-gateway", "kilo"),
    // deepseek
    ("deep-seek", "deepseek"),
    // alibaba
    ("dashscope", "alibaba"),
    ("aliyun", "alibaba"),
    ("qwen", "alibaba"),
    ("alibaba-cloud", "alibaba"),
    ("alibaba_coding", "alibaba-coding-plan"),
    ("alibaba-coding", "alibaba-coding-plan"),
    ("alibaba_coding_plan", "alibaba-coding-plan"),
    // google-gemini-cli (OAuth + Code Assist)
    ("gemini-cli", "google-gemini-cli"),
    ("gemini-oauth", "google-gemini-cli"),
    // huggingface
    ("hf", "huggingface"),
    ("hugging-face", "huggingface"),
    ("huggingface-hub", "huggingface"),
    // xiaomi
    ("mimo", "xiaomi"),
    ("xiaomi-mimo", "xiaomi"),
    // tencent
    ("tencent", "tencent-tokenhub"),
    ("tokenhub", "tencent-tokenhub"),
    ("tencent-cloud", "tencent-tokenhub"),
    ("tencentmaas", "tencent-tokenhub"),
    // bedrock
    ("aws", "bedrock"),
    ("aws-bedrock", "bedrock"),
    ("amazon-bedrock", "bedrock"),
    ("amazon", "bedrock"),
    // arcee
    ("arcee-ai", "arcee"),
    ("arceeai", "arcee"),
    // gmi
    ("gmi-cloud", "gmi"),
    ("gmicloud", "gmi"),
    // Local server aliases → virtual "local" concept (resolved via user config)
    ("lmstudio", "lmstudio"),
    ("lm-studio", "lmstudio"),
    ("lm_studio", "lmstudio"),
    ("ollama", "custom"), // bare "ollama" = local; use "ollama-cloud" for cloud
    ("vllm", "local"),
    ("llamacpp", "local"),
    ("llama.cpp", "local"),
    ("llama-cpp", "local"),
];

/// Lazily-built alias map.
pub fn aliases() -> &'static HashMap<&'static str, &'static str> {
    static ALIASES: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    ALIASES.get_or_init(|| ALIAS_PAIRS.iter().copied().collect())
}

// -- Display labels -----------------------------------------------------------
// Built dynamically from models.dev + overlays. Fallback for providers
// not in the catalog.

const LABEL_OVERRIDE_PAIRS: &[(&str, &str)] = &[
    ("nous", "Nous Portal"),
    ("openai-codex", "OpenAI Codex"),
    ("copilot-acp", "GitHub Copilot ACP"),
    ("stepfun", "StepFun Step Plan"),
    ("xiaomi", "Xiaomi MiMo"),
    ("gmi", "GMI Cloud"),
    ("tencent-tokenhub", "Tencent TokenHub"),
    ("lmstudio", "LM Studio"),
    ("local", "Local endpoint"),
    ("bedrock", "AWS Bedrock"),
    ("ollama-cloud", "Ollama Cloud"),
];

fn label_overrides() -> &'static HashMap<&'static str, &'static str> {
    static LABELS: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    LABELS.get_or_init(|| LABEL_OVERRIDE_PAIRS.iter().copied().collect())
}

// -- Transport → API mode mapping ---------------------------------------------

/// Map a transport name to its wire-protocol API mode.
///
/// Faithful equivalent of the `TRANSPORT_TO_API_MODE` dict + `.get(t, default)`.
pub fn transport_to_api_mode(transport: &str) -> &'static str {
    match transport {
        "openai_chat" => "chat_completions",
        "anthropic_messages" => "anthropic_messages",
        "codex_responses" => "codex_responses",
        "bedrock_converse" => "bedrock_converse",
        _ => "chat_completions",
    }
}

// -- models.dev hook ----------------------------------------------------------
//
// The Python code does a lazy `from agent.models_dev import get_provider_info`.
// Here we call the ported `crate::ag_models_dev::get_provider_info` directly.
// To keep the module testable without a populated catalog, lookups are routed
// through a small indirection that tests may override.

type MdevLookup = fn(&str) -> Option<crate::ag_models_dev::ProviderInfo>;

#[cfg(not(test))]
fn mdev_lookup(canonical: &str) -> Option<crate::ag_models_dev::ProviderInfo> {
    crate::ag_models_dev::get_provider_info(canonical)
}

#[cfg(test)]
thread_local! {
    static TEST_MDEV: std::cell::RefCell<Option<MdevLookup>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn mdev_lookup(canonical: &str) -> Option<crate::ag_models_dev::ProviderInfo> {
    let overridden =
        TEST_MDEV.with(|cell| cell.borrow().map(|f| f(canonical)));
    match overridden {
        Some(result) => result,
        None => crate::ag_models_dev::get_provider_info(canonical),
    }
}

// -- Helper functions ---------------------------------------------------------

/// Resolve aliases and normalise casing to a canonical provider id.
///
/// Returns the canonical id string. Does *not* validate that the id
/// corresponds to a known provider.
pub fn normalize_provider(name: &str) -> String {
    let key = name.trim().to_lowercase();
    match aliases().get(key.as_str()) {
        Some(canonical) => (*canonical).to_string(),
        None => key,
    }
}

/// Look up a built-in provider by id or alias.
///
/// Resolution order:
///   1. models.dev catalog + Hermes overlay (merged), if present in models.dev.
///   2. Hermes overlay only (for providers not in models.dev).
///
/// Returns a fully-resolved [`ProviderDef`] or `None`.
pub fn get_provider(name: &str) -> Option<ProviderDef> {
    let canonical = normalize_provider(name);

    let mdev_info = mdev_lookup(&canonical);
    let overlay = hermes_overlays().get(canonical.as_str());

    if let Some(mdev) = mdev_info {
        // Merge models.dev + overlay
        let transport = overlay.map(|o| o.transport).unwrap_or("openai_chat");
        let is_agg = overlay.map(|o| o.is_aggregator).unwrap_or(false);
        let auth = overlay.map(|o| o.auth_type).unwrap_or("api_key");
        let base_url_env = overlay.map(|o| o.base_url_env_var).unwrap_or("");
        let base_url_override = overlay.map(|o| o.base_url_override).unwrap_or("");

        // Combine env vars: models.dev env + hermes extra (dedup, preserve order)
        let mut env_vars: Vec<String> = mdev.env.clone();
        if let Some(o) = overlay {
            for ev in o.extra_env_vars {
                if !env_vars.iter().any(|e| e == ev) {
                    env_vars.push((*ev).to_string());
                }
            }
        }

        let base_url = if base_url_override.is_empty() {
            mdev.api.clone()
        } else {
            base_url_override.to_string()
        };

        return Some(ProviderDef {
            id: canonical,
            name: mdev.name,
            transport: transport.to_string(),
            api_key_env_vars: env_vars,
            base_url,
            base_url_env_var: base_url_env.to_string(),
            is_aggregator: is_agg,
            auth_type: auth.to_string(),
            doc: mdev.doc,
            source: "models.dev".to_string(),
        });
    }

    if let Some(o) = overlay {
        // Hermes-only provider (not in models.dev)
        let name = label_overrides()
            .get(canonical.as_str())
            .map(|s| (*s).to_string())
            .unwrap_or_else(|| canonical.clone());
        return Some(ProviderDef {
            id: canonical,
            name,
            transport: o.transport.to_string(),
            api_key_env_vars: o.extra_env_vars.iter().map(|s| s.to_string()).collect(),
            base_url: o.base_url_override.to_string(),
            base_url_env_var: o.base_url_env_var.to_string(),
            is_aggregator: o.is_aggregator,
            auth_type: o.auth_type.to_string(),
            doc: String::new(),
            source: "hermes".to_string(),
        });
    }

    None
}

/// Get a human-readable display name for a provider.
pub fn get_label(provider_id: &str) -> String {
    let canonical = normalize_provider(provider_id);

    // Check label overrides first
    if let Some(label) = label_overrides().get(canonical.as_str()) {
        return (*label).to_string();
    }

    // Try models.dev / overlay
    if let Some(pdef) = get_provider(&canonical) {
        return pdef.name;
    }

    canonical
}

/// Return `true` when the provider is a multi-model aggregator.
pub fn is_aggregator(provider: &str) -> bool {
    get_provider(provider)
        .map(|p| p.is_aggregator)
        .unwrap_or(false)
}

/// Determine the API mode (wire protocol) for a provider/endpoint.
///
/// Resolution order:
///   1. Known provider → transport → [`transport_to_api_mode`]
///      (with URL heuristics for special endpoints applied first).
///   2. URL heuristics for unknown / custom providers.
///   3. Default: `chat_completions`.
pub fn determine_api_mode(provider: &str, base_url: &str) -> String {
    if let Some(pdef) = get_provider(provider) {
        // Even for known providers, check URL heuristics for special endpoints
        // (e.g. kimi /coding endpoint needs anthropic_messages even on 'custom')
        if !base_url.is_empty() {
            let url_lower = base_url.trim_end_matches('/').to_lowercase();
            if url_lower.contains("api.kimi.com/coding") {
                return "anthropic_messages".to_string();
            }
            if url_lower.ends_with("/anthropic") || url_lower.contains("api.anthropic.com") {
                return "anthropic_messages".to_string();
            }
            if url_lower.contains("api.openai.com") {
                return "codex_responses".to_string();
            }
        }
        return transport_to_api_mode(&pdef.transport).to_string();
    }

    // Direct provider checks for providers not in HERMES_OVERLAYS
    if provider == "bedrock" {
        return "bedrock_converse".to_string();
    }

    // URL-based heuristics for custom / unknown providers
    if !base_url.is_empty() {
        let url_lower = base_url.trim_end_matches('/').to_lowercase();
        let hostname = base_url_hostname(base_url);
        if url_lower.ends_with("/anthropic") || hostname == "api.anthropic.com" {
            return "anthropic_messages".to_string();
        }
        if hostname == "api.kimi.com" && url_lower.contains("/coding") {
            return "anthropic_messages".to_string();
        }
        if hostname == "api.openai.com" {
            return "codex_responses".to_string();
        }
        if hostname.starts_with("bedrock-runtime.")
            && base_url_host_matches(base_url, "amazonaws.com")
        {
            return "bedrock_converse".to_string();
        }
    }

    "chat_completions".to_string()
}

// -- Provider from user config ------------------------------------------------

/// Minimal trait for reading user-config entries without binding to a concrete
/// YAML/JSON value type. Implemented for [`serde_json::Value`] and
/// [`serde_yaml::Value`] below.
///
/// `get_str` returns the entry's string value for a key, or `None` if missing
/// or not a string.
pub trait ConfigEntry {
    /// True if this value is a mapping/dict.
    fn is_mapping(&self) -> bool;
    /// String value for a key (only when the value is itself a string).
    fn get_str(&self, key: &str) -> Option<String>;
}

impl ConfigEntry for serde_json::Value {
    fn is_mapping(&self) -> bool {
        self.is_object()
    }
    fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }
}

impl ConfigEntry for serde_yaml::Value {
    fn is_mapping(&self) -> bool {
        self.is_mapping()
    }
    fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }
}

/// Pick the first non-empty value among a sequence of `Option<String>`,
/// falling back to an empty string — mirrors Python's chained `or ""`.
fn first_nonempty(candidates: &[Option<String>]) -> String {
    for c in candidates {
        if let Some(s) = c {
            if !s.is_empty() {
                return s.clone();
            }
        }
    }
    String::new()
}

/// Resolve a provider from the user's config.yaml `providers:` section.
///
/// `user_config` is the `providers:` mapping: provider-name -> entry. Returns a
/// [`ProviderDef`] if a mapping entry named `name` exists, else `None`.
pub fn resolve_user_provider<V: ConfigEntry>(
    name: &str,
    user_config: &HashMap<String, V>,
) -> Option<ProviderDef> {
    if user_config.is_empty() {
        return None;
    }

    let entry = user_config.get(name)?;
    if !entry.is_mapping() {
        return None;
    }

    // Extract fields. `name` field falls back to the lookup key.
    let display_name = {
        let n = entry.get_str("name").unwrap_or_default();
        if n.is_empty() {
            name.to_string()
        } else {
            n
        }
    };
    let api_url = first_nonempty(&[
        entry.get_str("api"),
        entry.get_str("url"),
        entry.get_str("base_url"),
    ]);
    let key_env = entry.get_str("key_env").unwrap_or_default();
    let transport = {
        let t = entry.get_str("transport").unwrap_or_default();
        if t.is_empty() {
            "openai_chat".to_string()
        } else {
            t
        }
    };

    let mut env_vars: Vec<String> = Vec::new();
    if !key_env.is_empty() {
        env_vars.push(key_env);
    }

    Some(ProviderDef {
        id: name.to_string(),
        name: display_name,
        transport,
        api_key_env_vars: env_vars,
        base_url: api_url,
        base_url_env_var: String::new(),
        is_aggregator: false,
        auth_type: "api_key".to_string(),
        doc: String::new(),
        source: "user-config".to_string(),
    })
}

/// Build a canonical slug for a `custom_providers` entry.
///
/// Matches the convention used by runtime_provider and credential_pool
/// (`custom:<normalized-name>`). Centralised here so all call-sites produce
/// identical slugs.
pub fn custom_provider_slug(display_name: &str) -> String {
    format!("custom:{}", display_name.trim().to_lowercase().replace(' ', "-"))
}

/// Resolve a provider from the user's config.yaml `custom_providers` list.
///
/// Each entry is a config mapping. Returns the matched [`ProviderDef`], or the
/// first valid entry when the request is the bare string `"custom"` (self-heal
/// for corrupt state, GH #17478), else `None`.
pub fn resolve_custom_provider<V: ConfigEntry>(
    name: &str,
    custom_providers: Option<&[V]>,
) -> Option<ProviderDef> {
    let custom_providers = custom_providers?;
    if custom_providers.is_empty() {
        return None;
    }

    let requested = name.trim().to_lowercase();
    if requested.is_empty() {
        return None;
    }

    let bare_custom_fallback = requested == "custom";
    let mut first_valid: Option<(String, String)> = None;

    for entry in custom_providers {
        if !entry.is_mapping() {
            continue;
        }

        let display_name = entry.get_str("name").unwrap_or_default().trim().to_string();
        let api_url = first_nonempty(&[
            entry.get_str("base_url"),
            entry.get_str("url"),
            entry.get_str("api"),
        ])
        .trim()
        .to_string();

        if display_name.is_empty() || api_url.is_empty() {
            continue;
        }

        if first_valid.is_none() {
            first_valid = Some((display_name.clone(), api_url.clone()));
        }

        let slug = custom_provider_slug(&display_name);
        if requested != display_name.to_lowercase() && requested != slug {
            continue;
        }

        return Some(ProviderDef {
            id: slug,
            name: display_name,
            transport: "openai_chat".to_string(),
            api_key_env_vars: Vec::new(),
            base_url: api_url,
            base_url_env_var: String::new(),
            is_aggregator: false,
            auth_type: "api_key".to_string(),
            doc: String::new(),
            source: "user-config".to_string(),
        });
    }

    // Self-heal: bare "custom" matched nothing — return first valid entry
    if bare_custom_fallback {
        if let Some((dname, aurl)) = first_valid {
            let slug = custom_provider_slug(&dname);
            return Some(ProviderDef {
                id: slug,
                name: dname,
                transport: "openai_chat".to_string(),
                api_key_env_vars: Vec::new(),
                base_url: aurl,
                base_url_env_var: String::new(),
                is_aggregator: false,
                auth_type: "api_key".to_string(),
                doc: String::new(),
                source: "user-config".to_string(),
            });
        }
    }

    None
}

/// Full resolution chain: built-in → user config → models.dev (direct).
///
/// This is the main entry point for `--provider` flag resolution.
///
/// Arguments mirror the Python signature: optional `providers:` mapping and an
/// optional `custom_providers:` list from config.yaml.
pub fn resolve_provider_full<V: ConfigEntry>(
    name: &str,
    user_providers: Option<&HashMap<String, V>>,
    custom_providers: Option<&[V]>,
) -> Option<ProviderDef> {
    let canonical = normalize_provider(name);

    // 1. Built-in (models.dev + overlays)
    if let Some(pdef) = get_provider(&canonical) {
        return Some(pdef);
    }

    // 2. User-defined providers from config
    if let Some(up) = user_providers {
        if !up.is_empty() {
            // Try canonical name
            if let Some(user_pdef) = resolve_user_provider(&canonical, up) {
                return Some(user_pdef);
            }
            // Try original name (in case alias didn't match)
            let original = name.trim().to_lowercase();
            if let Some(user_pdef) = resolve_user_provider(&original, up) {
                return Some(user_pdef);
            }
        }
    }

    // 2b. Saved custom providers from config
    if let Some(custom_pdef) = resolve_custom_provider(name, custom_providers) {
        return Some(custom_pdef);
    }

    // 3. Try models.dev directly (for providers not in our ALIASES)
    if let Some(mdev) = mdev_lookup(&canonical) {
        return Some(ProviderDef {
            id: canonical,
            name: mdev.name,
            transport: "openai_chat".to_string(),
            api_key_env_vars: mdev.env,
            base_url: mdev.api,
            base_url_env_var: String::new(),
            is_aggregator: false,
            auth_type: "api_key".to_string(),
            doc: String::new(),
            source: "models.dev".to_string(),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ag_models_dev::ProviderInfo;

    /// Install a thread-local models.dev override for the duration of a test.
    fn set_mdev(f: MdevLookup) {
        TEST_MDEV.with(|cell| *cell.borrow_mut() = Some(f));
    }
    fn clear_mdev() {
        TEST_MDEV.with(|cell| *cell.borrow_mut() = None);
    }

    fn no_mdev(_: &str) -> Option<ProviderInfo> {
        None
    }

    fn yaml_map(pairs: &[(&str, &str)]) -> serde_yaml::Value {
        let mut m = serde_yaml::Mapping::new();
        for (k, v) in pairs {
            m.insert(
                serde_yaml::Value::String((*k).to_string()),
                serde_yaml::Value::String((*v).to_string()),
            );
        }
        serde_yaml::Value::Mapping(m)
    }

    #[test]
    fn test_normalize_provider_aliases() {
        assert_eq!(normalize_provider("GLM"), "zai");
        assert_eq!(normalize_provider(" grok "), "xai");
        assert_eq!(normalize_provider("openai"), "openrouter");
        assert_eq!(normalize_provider("claude-code"), "anthropic");
        assert_eq!(normalize_provider("ollama"), "custom");
        // Unknown name passes through lowercased + trimmed.
        assert_eq!(normalize_provider("  MyThing "), "mything");
    }

    #[test]
    fn test_get_provider_overlay_only() {
        set_mdev(no_mdev);
        // "nous" is not in models.dev (per our override) → hermes-only path.
        let p = get_provider("nous").expect("nous overlay");
        assert_eq!(p.id, "nous");
        assert_eq!(p.name, "Nous Portal"); // from label override
        assert_eq!(p.transport, "openai_chat");
        assert_eq!(p.auth_type, "oauth_device_code");
        assert_eq!(p.base_url, "https://inference-api.nousresearch.com/v1");
        assert_eq!(p.source, "hermes");
        clear_mdev();
    }

    #[test]
    fn test_get_provider_overlay_name_fallback() {
        set_mdev(no_mdev);
        // "qwen-oauth" has no label override → name falls back to canonical id.
        let p = get_provider("qwen-oauth").expect("qwen-oauth overlay");
        assert_eq!(p.name, "qwen-oauth");
        assert_eq!(p.base_url_env_var, "HERMES_QWEN_BASE_URL");
        clear_mdev();
    }

    #[test]
    fn test_get_provider_merge_with_mdev() {
        // Provide a fake models.dev entry for "anthropic".
        fn lookup(id: &str) -> Option<ProviderInfo> {
            if id == "anthropic" {
                Some(ProviderInfo {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    env: vec!["ANTHROPIC_API_KEY".to_string()],
                    api: "https://api.anthropic.com".to_string(),
                    doc: "https://docs.anthropic.com".to_string(),
                    model_count: 0,
                })
            } else {
                None
            }
        }
        set_mdev(lookup);
        let p = get_provider("claude").expect("anthropic via alias");
        assert_eq!(p.id, "anthropic");
        assert_eq!(p.name, "Anthropic");
        assert_eq!(p.transport, "anthropic_messages"); // from overlay
        assert_eq!(p.source, "models.dev");
        // models.dev env + overlay extras, deduped & ordered.
        assert_eq!(
            p.api_key_env_vars,
            vec![
                "ANTHROPIC_API_KEY".to_string(),
                "ANTHROPIC_TOKEN".to_string(),
                "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
            ]
        );
        assert_eq!(p.base_url, "https://api.anthropic.com");
        assert_eq!(p.doc, "https://docs.anthropic.com");
        clear_mdev();
    }

    #[test]
    fn test_get_provider_base_url_override_wins() {
        // models.dev gives an api, overlay override should replace it (xai).
        fn lookup(id: &str) -> Option<ProviderInfo> {
            if id == "xai" {
                Some(ProviderInfo {
                    id: "xai".to_string(),
                    name: "xAI".to_string(),
                    env: vec!["XAI_API_KEY".to_string()],
                    api: "https://wrong.example/v1".to_string(),
                    doc: String::new(),
                    model_count: 0,
                })
            } else {
                None
            }
        }
        set_mdev(lookup);
        let p = get_provider("grok").unwrap();
        assert_eq!(p.base_url, "https://api.x.ai/v1");
        assert_eq!(p.transport, "codex_responses");
        clear_mdev();
    }

    #[test]
    fn test_get_provider_unknown() {
        set_mdev(no_mdev);
        assert!(get_provider("totally-unknown-xyz").is_none());
        clear_mdev();
    }

    #[test]
    fn test_get_label() {
        set_mdev(no_mdev);
        assert_eq!(get_label("local"), "Local endpoint");
        assert_eq!(get_label("bedrock"), "AWS Bedrock");
        // Unknown → canonical id returned.
        assert_eq!(get_label("nothinghere"), "nothinghere");
        clear_mdev();
    }

    #[test]
    fn test_is_aggregator() {
        set_mdev(no_mdev);
        assert!(is_aggregator("openrouter"));
        assert!(is_aggregator("vercel"));
        assert!(!is_aggregator("anthropic"));
        assert!(!is_aggregator("unknown-xyz"));
        clear_mdev();
    }

    #[test]
    fn test_determine_api_mode_known() {
        set_mdev(no_mdev);
        // anthropic overlay → anthropic_messages.
        assert_eq!(determine_api_mode("anthropic", ""), "anthropic_messages");
        // openrouter → openai_chat → chat_completions.
        assert_eq!(determine_api_mode("openrouter", ""), "chat_completions");
        // bedrock overlay → bedrock_converse.
        assert_eq!(determine_api_mode("bedrock", ""), "bedrock_converse");
        clear_mdev();
    }

    #[test]
    fn test_determine_api_mode_known_url_heuristics() {
        set_mdev(no_mdev);
        // Known provider but special URL overrides transport mapping.
        assert_eq!(
            determine_api_mode("openrouter", "https://api.kimi.com/coding/v1/"),
            "anthropic_messages"
        );
        assert_eq!(
            determine_api_mode("openrouter", "https://x/anthropic"),
            "anthropic_messages"
        );
        assert_eq!(
            determine_api_mode("openrouter", "https://api.openai.com/v1"),
            "codex_responses"
        );
        clear_mdev();
    }

    #[test]
    fn test_determine_api_mode_unknown_url() {
        set_mdev(no_mdev);
        // Unknown provider, anthropic-style URL.
        assert_eq!(
            determine_api_mode("custom", "https://api.anthropic.com/v1"),
            "anthropic_messages"
        );
        assert_eq!(
            determine_api_mode("custom", "https://proxy.test/anthropic"),
            "anthropic_messages"
        );
        assert_eq!(
            determine_api_mode("custom", "https://api.openai.com"),
            "codex_responses"
        );
        // kimi coding via hostname path.
        assert_eq!(
            determine_api_mode("custom", "https://api.kimi.com/coding"),
            "anthropic_messages"
        );
        // bedrock-runtime host.
        assert_eq!(
            determine_api_mode(
                "custom",
                "https://bedrock-runtime.us-east-1.amazonaws.com"
            ),
            "bedrock_converse"
        );
        // Plain unknown → default.
        assert_eq!(
            determine_api_mode("custom", "https://example.com/v1"),
            "chat_completions"
        );
        clear_mdev();
    }

    #[test]
    fn test_transport_to_api_mode() {
        assert_eq!(transport_to_api_mode("openai_chat"), "chat_completions");
        assert_eq!(
            transport_to_api_mode("anthropic_messages"),
            "anthropic_messages"
        );
        assert_eq!(transport_to_api_mode("codex_responses"), "codex_responses");
        assert_eq!(
            transport_to_api_mode("bedrock_converse"),
            "bedrock_converse"
        );
        assert_eq!(transport_to_api_mode("weird"), "chat_completions");
    }

    #[test]
    fn test_custom_provider_slug() {
        assert_eq!(custom_provider_slug("My Provider"), "custom:my-provider");
        assert_eq!(custom_provider_slug("  Foo Bar  "), "custom:foo-bar");
    }

    #[test]
    fn test_resolve_user_provider() {
        let mut cfg: HashMap<String, serde_yaml::Value> = HashMap::new();
        cfg.insert(
            "mything".to_string(),
            yaml_map(&[
                ("name", "My Thing"),
                ("base_url", "https://api.mything.test/v1"),
                ("key_env", "MYTHING_KEY"),
                ("transport", "anthropic_messages"),
            ]),
        );
        let p = resolve_user_provider("mything", &cfg).unwrap();
        assert_eq!(p.id, "mything");
        assert_eq!(p.name, "My Thing");
        assert_eq!(p.base_url, "https://api.mything.test/v1");
        assert_eq!(p.api_key_env_vars, vec!["MYTHING_KEY".to_string()]);
        assert_eq!(p.transport, "anthropic_messages");
        assert_eq!(p.source, "user-config");
    }

    #[test]
    fn test_resolve_user_provider_defaults() {
        let mut cfg: HashMap<String, serde_yaml::Value> = HashMap::new();
        // No "name", no "transport" → name falls back to key, transport default.
        cfg.insert("plain".to_string(), yaml_map(&[("url", "https://u.test")]));
        let p = resolve_user_provider("plain", &cfg).unwrap();
        assert_eq!(p.name, "plain");
        assert_eq!(p.transport, "openai_chat");
        assert_eq!(p.base_url, "https://u.test");
        assert!(p.api_key_env_vars.is_empty());
    }

    #[test]
    fn test_resolve_user_provider_missing() {
        let cfg: HashMap<String, serde_yaml::Value> = HashMap::new();
        assert!(resolve_user_provider("x", &cfg).is_none());
        let mut cfg2: HashMap<String, serde_yaml::Value> = HashMap::new();
        // entry that isn't a mapping
        cfg2.insert("x".to_string(), serde_yaml::Value::String("nope".into()));
        assert!(resolve_user_provider("x", &cfg2).is_none());
    }

    #[test]
    fn test_resolve_custom_provider_match() {
        let list = vec![
            yaml_map(&[("name", "Alpha"), ("base_url", "https://alpha.test")]),
            yaml_map(&[("name", "Beta"), ("api", "https://beta.test")]),
        ];
        // Match by slug.
        let p = resolve_custom_provider("custom:beta", Some(&list)).unwrap();
        assert_eq!(p.id, "custom:beta");
        assert_eq!(p.name, "Beta");
        assert_eq!(p.base_url, "https://beta.test");
        // Match by lowercased display name.
        let p2 = resolve_custom_provider("alpha", Some(&list)).unwrap();
        assert_eq!(p2.id, "custom:alpha");
    }

    #[test]
    fn test_resolve_custom_provider_bare_fallback() {
        let list = vec![
            // Invalid (no url) — skipped.
            yaml_map(&[("name", "NoUrl")]),
            yaml_map(&[("name", "Gamma"), ("url", "https://gamma.test")]),
        ];
        let p = resolve_custom_provider("custom", Some(&list)).unwrap();
        assert_eq!(p.name, "Gamma");
        assert_eq!(p.id, "custom:gamma");
    }

    #[test]
    fn test_resolve_custom_provider_none() {
        let list: Vec<serde_yaml::Value> = vec![];
        assert!(resolve_custom_provider("x", Some(&list)).is_none());
        assert!(resolve_custom_provider("x", None).is_none());
        // Non-matching, non-bare request.
        let list2 = vec![yaml_map(&[("name", "Alpha"), ("url", "https://a.test")])];
        assert!(resolve_custom_provider("zzz", Some(&list2)).is_none());
    }

    #[test]
    fn test_resolve_provider_full_builtin_first() {
        set_mdev(no_mdev);
        let up: HashMap<String, serde_yaml::Value> = HashMap::new();
        // "nous" resolves as built-in even with empty user config.
        let p = resolve_provider_full("nous", Some(&up), None).unwrap();
        assert_eq!(p.source, "hermes");
        clear_mdev();
    }

    #[test]
    fn test_resolve_provider_full_user_config() {
        set_mdev(no_mdev);
        let mut up: HashMap<String, serde_yaml::Value> = HashMap::new();
        up.insert(
            "weird-local".to_string(),
            yaml_map(&[("base_url", "http://localhost:9999")]),
        );
        let p = resolve_provider_full("weird-local", Some(&up), None).unwrap();
        assert_eq!(p.source, "user-config");
        assert_eq!(p.base_url, "http://localhost:9999");
        clear_mdev();
    }

    #[test]
    fn test_resolve_provider_full_custom_list() {
        set_mdev(no_mdev);
        let list = vec![yaml_map(&[
            ("name", "Delta"),
            ("base_url", "https://delta.test"),
        ])];
        let up: HashMap<String, serde_yaml::Value> = HashMap::new();
        let p = resolve_provider_full("custom:delta", Some(&up), Some(&list)).unwrap();
        assert_eq!(p.name, "Delta");
        assert_eq!(p.source, "user-config");
        clear_mdev();
    }

    #[test]
    fn test_resolve_provider_full_mdev_direct() {
        fn lookup(id: &str) -> Option<ProviderInfo> {
            if id == "some-mdev-only" {
                Some(ProviderInfo {
                    id: "some-mdev-only".to_string(),
                    name: "Some MDev".to_string(),
                    env: vec!["SOME_KEY".to_string()],
                    api: "https://some.test".to_string(),
                    doc: String::new(),
                    model_count: 0,
                })
            } else {
                None
            }
        }
        set_mdev(lookup);
        // Not an alias, not in overlays, but in models.dev → direct fallback.
        // Note: get_provider would already catch it; this confirms the chain.
        let p = resolve_provider_full("some-mdev-only", None, None).unwrap();
        assert_eq!(p.name, "Some MDev");
        assert_eq!(p.source, "models.dev");
        clear_mdev();
    }

    #[test]
    fn test_resolve_provider_full_none() {
        set_mdev(no_mdev);
        assert!(resolve_provider_full("nope-nope", None, None).is_none());
        clear_mdev();
    }
}
