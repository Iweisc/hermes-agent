//! Canonical model catalogs and lightweight validation helpers.
//!
//! Faithful native Rust port of `hermes_cli/models.py`.
//!
//! This module is the single source of truth for:
//!
//! * The static fallback catalogs (`OPENROUTER_MODELS`, `VERCEL_AI_GATEWAY_MODELS`,
//!   the `_PROVIDER_MODELS` map, `CANONICAL_PROVIDERS`, alias tables).
//! * Provider-id normalisation and labelling.
//! * Live catalog / pricing fetchers (OpenRouter, AI Gateway, Anthropic,
//!   Copilot, LM Studio, Ollama Cloud, generic OpenAI-style `/models`).
//! * `validate_requested_model`, the `/model` validation entrypoint.
//!
//! ## Parity notes
//!
//! The Python module performs *many* lazy imports of sibling hermes modules
//! (`hermes_cli.auth`, `hermes_cli.codex_models`, `agent.models_dev`,
//! `agent.anthropic_adapter`, `providers`, `hermes_cli.config`, etc.). Most of
//! those have not been ported yet or live behind credential resolution that has
//! no native equivalent in this crate. Where a cross-module dependency exists:
//!
//! * If a ported sibling exists (`crate::cli_codex_models`,
//!   `crate::tool_fuzzy_match`, `crate::mod_hermes_constants`,
//!   `crate::cli_model_normalize`) it is used directly and recorded in
//!   `cross_refs`.
//! * Otherwise the corresponding `try/except: pass` branch is treated as
//!   *always failing* (the Python fallback path) — exactly what happens when
//!   the import raises. Several functions therefore expose extra parameters
//!   (e.g. a caller-provided access token) so the caller can inject what the
//!   Python code resolved internally.
//!
//! The module-level mutable caches in Python (`_openrouter_catalog_cache`,
//! `_pricing_cache`, `_free_tier_cache`, …) are reproduced with
//! `std::sync::Mutex`-guarded statics so behaviour (and TTLs) match.
//!
//! Network calls use `reqwest::blocking` and keep the exact request shapes
//! (URLs, headers, JSON parsing) from the Python source.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::tool_fuzzy_match::seq_ratio;

/// Hermes version string used in the User-Agent header.
///
/// The Python source imports `hermes_cli.__version__`. There is no equivalent
/// constant available in this crate, so this mirrors the build version.
pub const HERMES_VERSION: &str = env!("CARGO_PKG_VERSION");

/// User-Agent identifying hermes-cli so Cloudflare Browser-Integrity-Check
/// (error 1010) endpoints don't reject a default urllib signature.
pub fn hermes_user_agent() -> String {
    format!("hermes-cli/{}", HERMES_VERSION)
}

pub const COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";
pub fn copilot_models_url() -> String {
    format!("{}/models", COPILOT_BASE_URL)
}
pub const COPILOT_EDITOR_VERSION: &str = "vscode/1.104.1";
pub const COPILOT_REASONING_EFFORTS_GPT5: &[&str] = &["minimal", "low", "medium", "high"];
pub const COPILOT_REASONING_EFFORTS_O_SERIES: &[&str] = &["low", "medium", "high"];

/// Default AI Gateway base URL (mirrors `hermes_constants.AI_GATEWAY_BASE_URL`).
pub const AI_GATEWAY_BASE_URL: &str = crate::mod_hermes_constants::AI_GATEWAY_BASE_URL;

// ---------------------------------------------------------------------------
// Static fallback catalogs
// ---------------------------------------------------------------------------

/// Fallback OpenRouter snapshot used when the live catalog is unavailable.
/// `(model_id, display description shown in menus)`.
pub const OPENROUTER_MODELS: &[(&str, &str)] = &[
    ("moonshotai/kimi-k2.6", "recommended"),
    ("anthropic/claude-opus-4.7", ""),
    ("anthropic/claude-opus-4.6", ""),
    ("anthropic/claude-sonnet-4.6", ""),
    ("qwen/qwen3.6-plus", ""),
    ("anthropic/claude-sonnet-4.5", ""),
    ("anthropic/claude-haiku-4.5", ""),
    ("openrouter/elephant-alpha", "free"),
    ("openrouter/owl-alpha", "free"),
    ("openai/gpt-5.5", ""),
    ("openai/gpt-5.4-mini", ""),
    ("xiaomi/mimo-v2.5-pro", ""),
    ("xiaomi/mimo-v2.5", ""),
    ("tencent/hy3-preview:free", "free"),
    ("openai/gpt-5.3-codex", ""),
    ("google/gemini-3-pro-image-preview", ""),
    ("google/gemini-3-flash-preview", ""),
    ("google/gemini-3.1-pro-preview", ""),
    ("google/gemini-3.1-flash-lite-preview", ""),
    ("qwen/qwen3.5-plus-02-15", ""),
    ("qwen/qwen3.5-35b-a3b", ""),
    ("stepfun/step-3.5-flash", ""),
    ("minimax/minimax-m2.7", ""),
    ("minimax/minimax-m2.5", ""),
    ("minimax/minimax-m2.5:free", "free"),
    ("z-ai/glm-5.1", ""),
    ("z-ai/glm-5v-turbo", ""),
    ("z-ai/glm-5-turbo", ""),
    ("x-ai/grok-4.20", ""),
    ("x-ai/grok-4.3", ""),
    ("nvidia/nemotron-3-super-120b-a12b", ""),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("arcee-ai/trinity-large-preview:free", "free"),
    ("arcee-ai/trinity-large-thinking", ""),
    ("openai/gpt-5.5-pro", ""),
    ("openai/gpt-5.4-nano", ""),
    ("deepseek/deepseek-v4-pro", ""),
];

/// Fallback Vercel AI Gateway snapshot used when the live catalog is
/// unavailable. OSS / open-weight models prioritized first, then closed-source
/// by family.
pub const VERCEL_AI_GATEWAY_MODELS: &[(&str, &str)] = &[
    ("moonshotai/kimi-k2.6", "recommended"),
    ("alibaba/qwen3.6-plus", ""),
    ("zai/glm-5.1", ""),
    ("minimax/minimax-m2.7", ""),
    ("anthropic/claude-sonnet-4.6", ""),
    ("anthropic/claude-opus-4.7", ""),
    ("anthropic/claude-opus-4.6", ""),
    ("anthropic/claude-haiku-4.5", ""),
    ("openai/gpt-5.4", ""),
    ("openai/gpt-5.4-mini", ""),
    ("openai/gpt-5.3-codex", ""),
    ("google/gemini-3.1-pro-preview", ""),
    ("google/gemini-3-flash", ""),
    ("google/gemini-3.1-flash-lite-preview", ""),
    ("xai/grok-4.20-reasoning", ""),
];

// Module-level catalog caches (Python globals).
static OPENROUTER_CATALOG_CACHE: Mutex<Option<Vec<(String, String)>>> = Mutex::new(None);
static AI_GATEWAY_CATALOG_CACHE: Mutex<Option<Vec<(String, String)>>> = Mutex::new(None);

/// Derive the openai-codex curated list from `cli_codex_models` (single source
/// of truth: `DEFAULT_CODEX_MODELS` + forward-compat synthesis).
pub fn codex_curated_models() -> Vec<String> {
    let base: Vec<String> = crate::cli_codex_models::DEFAULT_CODEX_MODELS
        .iter()
        .map(|s| s.to_string())
        .collect();
    crate::cli_codex_models::add_forward_compat_models(base)
}

/// Static fallback for xAI used when the models.dev disk cache is empty.
pub const XAI_STATIC_FALLBACK: &[&str] = &[
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
    "grok-4-1-fast",
    "grok-4-1-fast-non-reasoning",
    "grok-4-fast",
    "grok-4-fast-non-reasoning",
    "grok-4",
    "grok-code-fast-1",
];

/// Derive the xAI-direct curated list.
///
/// The Python version reads `$HERMES_HOME/models_dev_cache.json` directly,
/// returning the sorted xAI model ids when present, else `XAI_STATIC_FALLBACK`.
pub fn xai_curated_models() -> Vec<String> {
    if let Some(ids) = load_xai_ids_from_disk_cache() {
        if !ids.is_empty() {
            let mut sorted = ids;
            sorted.sort();
            return sorted;
        }
    }
    XAI_STATIC_FALLBACK.iter().map(|s| s.to_string()).collect()
}

fn models_dev_cache_path() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_home().join("models_dev_cache.json")
}

fn load_xai_ids_from_disk_cache() -> Option<Vec<String>> {
    let path = models_dev_cache_path();
    let raw = std::fs::read_to_string(path).ok()?;
    let data: Value = serde_json::from_str(&raw).ok()?;
    let xai = data.get("xai")?;
    let models = xai.get("models")?.as_object()?;
    let ids: Vec<String> = models
        .keys()
        .filter(|k| !k.is_empty())
        .cloned()
        .collect();
    if ids.is_empty() {
        None
    } else {
        Some(ids)
    }
}

/// Build the static `_PROVIDER_MODELS` map.
///
/// `openai-codex` and `xai` are computed dynamically (they depend on the codex
/// catalog and the models.dev disk cache); everything else is a static list.
/// `ai-gateway` is derived from `VERCEL_AI_GATEWAY_MODELS`.
pub fn provider_models_map() -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();

    let lit = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };

    m.insert(
        "nous".into(),
        lit(&[
            "moonshotai/kimi-k2.6",
            "xiaomi/mimo-v2.5-pro",
            "xiaomi/mimo-v2.5",
            "tencent/hy3-preview",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-haiku-4.5",
            "openai/gpt-5.5",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.3-codex",
            "google/gemini-3-pro-preview",
            "google/gemini-3-flash-preview",
            "google/gemini-3.1-pro-preview",
            "google/gemini-3.1-flash-lite-preview",
            "qwen/qwen3.5-plus-02-15",
            "qwen/qwen3.5-35b-a3b",
            "stepfun/step-3.5-flash",
            "minimax/minimax-m2.7",
            "minimax/minimax-m2.5",
            "minimax/minimax-m2.5:free",
            "z-ai/glm-5.1",
            "z-ai/glm-5v-turbo",
            "z-ai/glm-5-turbo",
            "x-ai/grok-4.20-beta",
            "x-ai/grok-4.3",
            "nvidia/nemotron-3-super-120b-a12b",
            "arcee-ai/trinity-large-thinking",
            "openai/gpt-5.5-pro",
            "openai/gpt-5.4-nano",
            "deepseek/deepseek-v4-pro",
        ]),
    );
    m.insert(
        "openai".into(),
        lit(&[
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5-mini",
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]),
    );
    m.insert("openai-codex".into(), codex_curated_models());
    m.insert("copilot-acp".into(), lit(&["copilot-acp"]));
    m.insert(
        "copilot".into(),
        lit(&[
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5-mini",
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-sonnet-4.6",
            "claude-sonnet-4",
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "grok-code-fast-1",
        ]),
    );
    m.insert(
        "gemini".into(),
        lit(&[
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
        ]),
    );
    m.insert(
        "google-gemini-cli".into(),
        lit(&[
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ]),
    );
    m.insert(
        "zai".into(),
        lit(&[
            "glm-5.1",
            "glm-5",
            "glm-5v-turbo",
            "glm-5-turbo",
            "glm-4.7",
            "glm-4.5",
            "glm-4.5-flash",
        ]),
    );
    m.insert("xai".into(), xai_curated_models());
    m.insert(
        "nvidia".into(),
        lit(&[
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-nano-30b-a3b",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "qwen/qwen3.5-397b-a17b",
            "deepseek-ai/deepseek-v3.2",
            "moonshotai/kimi-k2.6",
            "minimaxai/minimax-m2.5",
            "z-ai/glm5",
            "openai/gpt-oss-120b",
        ]),
    );
    m.insert(
        "kimi-coding".into(),
        lit(&[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-for-coding",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
        ]),
    );
    m.insert(
        "kimi-coding-cn".into(),
        lit(&[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
        ]),
    );
    m.insert(
        "stepfun".into(),
        lit(&["step-3.5-flash", "step-3.5-flash-2603"]),
    );
    m.insert(
        "moonshot".into(),
        lit(&[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
        ]),
    );
    m.insert(
        "minimax".into(),
        lit(&["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"]),
    );
    m.insert(
        "minimax-oauth".into(),
        lit(&["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]),
    );
    m.insert(
        "minimax-cn".into(),
        lit(&["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"]),
    );
    m.insert(
        "anthropic".into(),
        lit(&[
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ]),
    );
    m.insert(
        "deepseek".into(),
        lit(&[
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-reasoner",
        ]),
    );
    m.insert(
        "xiaomi".into(),
        lit(&[
            "mimo-v2.5-pro",
            "mimo-v2.5",
            "mimo-v2-pro",
            "mimo-v2-omni",
            "mimo-v2-flash",
        ]),
    );
    m.insert("tencent-tokenhub".into(), lit(&["hy3-preview"]));
    m.insert(
        "arcee".into(),
        lit(&[
            "trinity-large-thinking",
            "trinity-large-preview",
            "trinity-mini",
        ]),
    );
    m.insert(
        "gmi".into(),
        lit(&[
            "zai-org/GLM-5.1-FP8",
            "deepseek-ai/DeepSeek-V3.2",
            "moonshotai/Kimi-K2.5",
            "google/gemini-3.1-flash-lite-preview",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
        ]),
    );
    m.insert(
        "opencode-zen".into(),
        lit(&[
            "kimi-k2.5",
            "gpt-5.4-pro",
            "gpt-5.4",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-nano",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-1",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-sonnet-4",
            "claude-haiku-4-5",
            "claude-3-5-haiku",
            "gemini-3.1-pro",
            "gemini-3-pro",
            "gemini-3-flash",
            "minimax-m2.7",
            "minimax-m2.5",
            "minimax-m2.5-free",
            "minimax-m2.1",
            "glm-5",
            "glm-4.7",
            "glm-4.6",
            "kimi-k2-thinking",
            "kimi-k2",
            "qwen3-coder",
            "big-pickle",
        ]),
    );
    m.insert(
        "opencode-go".into(),
        lit(&[
            "kimi-k2.6",
            "kimi-k2.5",
            "glm-5.1",
            "glm-5",
            "mimo-v2.5-pro",
            "mimo-v2.5",
            "mimo-v2-pro",
            "mimo-v2-omni",
            "minimax-m2.7",
            "minimax-m2.5",
            "qwen3.6-plus",
            "qwen3.5-plus",
        ]),
    );
    m.insert(
        "kilocode".into(),
        lit(&[
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
            "google/gemini-3-pro-preview",
            "google/gemini-3-flash-preview",
        ]),
    );
    m.insert(
        "alibaba".into(),
        lit(&[
            "qwen3.6-plus",
            "kimi-k2.5",
            "qwen3.5-plus",
            "qwen3-coder-plus",
            "qwen3-coder-next",
            "glm-5",
            "glm-4.7",
            "MiniMax-M2.5",
        ]),
    );
    m.insert(
        "huggingface".into(),
        lit(&[
            "moonshotai/Kimi-K2.5",
            "Qwen/Qwen3.5-397B-A17B",
            "Qwen/Qwen3.5-35B-A3B",
            "deepseek-ai/DeepSeek-V3.2",
            "MiniMaxAI/MiniMax-M2.5",
            "zai-org/GLM-5",
            "XiaomiMiMo/MiMo-V2-Flash",
            "moonshotai/Kimi-K2-Thinking",
            "moonshotai/Kimi-K2.6",
        ]),
    );
    m.insert(
        "bedrock".into(),
        lit(&[
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-6-v1",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-micro-v1:0",
            "deepseek.v3.2",
            "us.meta.llama4-maverick-17b-instruct-v1:0",
            "us.meta.llama4-scout-17b-instruct-v1:0",
        ]),
    );
    m.insert("azure-foundry".into(), Vec::new());

    // ai-gateway derived from the curated tuple snapshot.
    m.insert(
        "ai-gateway".into(),
        VERCEL_AI_GATEWAY_MODELS
            .iter()
            .map(|(mid, _)| mid.to_string())
            .collect(),
    );

    m
}

// ---------------------------------------------------------------------------
// Canonical provider list
// ---------------------------------------------------------------------------

/// A canonical provider definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderEntry {
    pub slug: String,
    pub label: String,
    pub tui_desc: String,
}

impl ProviderEntry {
    fn new(slug: &str, label: &str, tui_desc: &str) -> Self {
        ProviderEntry {
            slug: slug.to_string(),
            label: label.to_string(),
            tui_desc: tui_desc.to_string(),
        }
    }
}

/// The single source of truth for provider identity.
///
/// The Python source auto-extends this from the `providers/` plugin registry;
/// that registry has no native equivalent here, so the auto-extend `try/except`
/// is treated as the (failing) fallback path — the static list is returned.
pub fn canonical_providers() -> Vec<ProviderEntry> {
    vec![
        ProviderEntry::new("nous", "Nous Portal", "Nous Portal (Nous Research subscription)"),
        ProviderEntry::new("openrouter", "OpenRouter", "OpenRouter (100+ models, pay-per-use)"),
        ProviderEntry::new("lmstudio", "LM Studio", "LM Studio (local desktop app with built-in model server)"),
        ProviderEntry::new("anthropic", "Anthropic", "Anthropic (Claude models — API key or Claude Code)"),
        ProviderEntry::new("openai-codex", "OpenAI Codex", "OpenAI Codex"),
        ProviderEntry::new("xiaomi", "Xiaomi MiMo", "Xiaomi MiMo (MiMo-V2.5 and V2 models — pro, omni, flash)"),
        ProviderEntry::new("tencent-tokenhub", "Tencent TokenHub", "Tencent TokenHub (Hy3 Preview — direct API via tokenhub.tencentmaas.com)"),
        ProviderEntry::new("nvidia", "NVIDIA NIM", "NVIDIA NIM (Nemotron models — build.nvidia.com or local NIM)"),
        ProviderEntry::new("qwen-oauth", "Qwen OAuth (Portal)", "Qwen OAuth (reuses local Qwen CLI login)"),
        ProviderEntry::new("copilot", "GitHub Copilot", "GitHub Copilot (uses GITHUB_TOKEN or gh auth token)"),
        ProviderEntry::new("copilot-acp", "GitHub Copilot ACP", "GitHub Copilot ACP (spawns `copilot --acp --stdio`)"),
        ProviderEntry::new("huggingface", "Hugging Face", "Hugging Face Inference Providers (20+ open models)"),
        ProviderEntry::new("gemini", "Google AI Studio", "Google AI Studio (Gemini models — native Gemini API)"),
        ProviderEntry::new("google-gemini-cli", "Google Gemini (OAuth)", "Google Gemini via OAuth + Code Assist (free tier supported; no API key needed)"),
        ProviderEntry::new("deepseek", "DeepSeek", "DeepSeek (DeepSeek-V3, R1, coder — direct API)"),
        ProviderEntry::new("xai", "xAI", "xAI (Grok models — direct API)"),
        ProviderEntry::new("zai", "Z.AI / GLM", "Z.AI / GLM (Zhipu AI direct API)"),
        ProviderEntry::new("kimi-coding", "Kimi / Kimi Coding Plan", "Kimi Coding Plan (api.kimi.com) & Moonshot API"),
        ProviderEntry::new("kimi-coding-cn", "Kimi / Moonshot (China)", "Kimi / Moonshot China (Moonshot CN direct API)"),
        ProviderEntry::new("stepfun", "StepFun Step Plan", "StepFun Step Plan (agent/coding models via Step Plan API)"),
        ProviderEntry::new("minimax", "MiniMax", "MiniMax (global direct API)"),
        ProviderEntry::new("minimax-oauth", "MiniMax (OAuth)", "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
        ProviderEntry::new("minimax-cn", "MiniMax (China)", "MiniMax China (domestic direct API)"),
        ProviderEntry::new("alibaba", "Alibaba Cloud (DashScope)", "Alibaba Cloud / DashScope Coding (Qwen + multi-provider)"),
        ProviderEntry::new("ollama-cloud", "Ollama Cloud", "Ollama Cloud (cloud-hosted open models — ollama.com)"),
        ProviderEntry::new("arcee", "Arcee AI", "Arcee AI (Trinity models — direct API)"),
        ProviderEntry::new("gmi", "GMI Cloud", "GMI Cloud (multi-model direct API)"),
        ProviderEntry::new("kilocode", "Kilo Code", "Kilo Code (Kilo Gateway API)"),
        ProviderEntry::new("opencode-zen", "OpenCode Zen", "OpenCode Zen (35+ curated models, pay-as-you-go)"),
        ProviderEntry::new("opencode-go", "OpenCode Go", "OpenCode Go (open models, $10/month subscription)"),
        ProviderEntry::new("bedrock", "AWS Bedrock", "AWS Bedrock (Claude, Nova, Llama, DeepSeek — IAM or API key)"),
        ProviderEntry::new("azure-foundry", "Azure Foundry", "Azure Foundry (OpenAI-style or Anthropic-style endpoint — your Azure AI deployment)"),
        ProviderEntry::new("ai-gateway", "Vercel AI Gateway", "Vercel AI Gateway"),
    ]
}

/// `slug -> label` map (with the `custom` special case appended).
pub fn provider_labels() -> HashMap<String, String> {
    let mut m: HashMap<String, String> = HashMap::new();
    for p in canonical_providers() {
        m.insert(p.slug, p.label);
    }
    m.insert("custom".into(), "Custom endpoint".into());
    m
}

/// Provider alias → canonical slug.
pub fn provider_aliases() -> HashMap<&'static str, &'static str> {
    let mut m: HashMap<&'static str, &'static str> = HashMap::new();
    for &(k, v) in &[
        ("glm", "zai"),
        ("z-ai", "zai"),
        ("z.ai", "zai"),
        ("zhipu", "zai"),
        ("github", "copilot"),
        ("github-copilot", "copilot"),
        ("github-models", "copilot"),
        ("github-model", "copilot"),
        ("github-copilot-acp", "copilot-acp"),
        ("copilot-acp-agent", "copilot-acp"),
        ("google", "gemini"),
        ("google-gemini", "gemini"),
        ("google-ai-studio", "gemini"),
        ("kimi", "kimi-coding"),
        ("moonshot", "kimi-coding"),
        ("kimi-cn", "kimi-coding-cn"),
        ("moonshot-cn", "kimi-coding-cn"),
        ("step", "stepfun"),
        ("stepfun-coding-plan", "stepfun"),
        ("arcee-ai", "arcee"),
        ("arceeai", "arcee"),
        ("gmi-cloud", "gmi"),
        ("gmicloud", "gmi"),
        ("minimax-china", "minimax-cn"),
        ("minimax_cn", "minimax-cn"),
        ("minimax-portal", "minimax-oauth"),
        ("minimax-global", "minimax-oauth"),
        ("minimax_oauth", "minimax-oauth"),
        ("claude", "anthropic"),
        ("claude-code", "anthropic"),
        ("deep-seek", "deepseek"),
        ("opencode", "opencode-zen"),
        ("zen", "opencode-zen"),
        ("go", "opencode-go"),
        ("opencode-go-sub", "opencode-go"),
        ("aigateway", "ai-gateway"),
        ("vercel", "ai-gateway"),
        ("vercel-ai-gateway", "ai-gateway"),
        ("kilo", "kilocode"),
        ("kilo-code", "kilocode"),
        ("kilo-gateway", "kilocode"),
        ("dashscope", "alibaba"),
        ("aliyun", "alibaba"),
        ("qwen", "alibaba"),
        ("alibaba-cloud", "alibaba"),
        ("qwen-portal", "qwen-oauth"),
        ("gemini-cli", "google-gemini-cli"),
        ("gemini-oauth", "google-gemini-cli"),
        ("hf", "huggingface"),
        ("hugging-face", "huggingface"),
        ("huggingface-hub", "huggingface"),
        ("mimo", "xiaomi"),
        ("xiaomi-mimo", "xiaomi"),
        ("tencent", "tencent-tokenhub"),
        ("tokenhub", "tencent-tokenhub"),
        ("tencent-cloud", "tencent-tokenhub"),
        ("tencentmaas", "tencent-tokenhub"),
        ("aws", "bedrock"),
        ("aws-bedrock", "bedrock"),
        ("amazon-bedrock", "bedrock"),
        ("amazon", "bedrock"),
        ("grok", "xai"),
        ("x-ai", "xai"),
        ("x.ai", "xai"),
        ("nim", "nvidia"),
        ("nvidia-nim", "nvidia"),
        ("build-nvidia", "nvidia"),
        ("nemotron", "nvidia"),
        ("lmstudio", "lmstudio"),
        ("lm-studio", "lmstudio"),
        ("lm_studio", "lmstudio"),
        ("ollama", "custom"),
        ("ollama_cloud", "ollama-cloud"),
    ] {
        m.insert(k, v);
    }
    m
}

/// Return the default model for a provider, or `""` if unknown.
pub fn get_default_model_for_provider(provider: &str) -> String {
    provider_models_map()
        .get(provider)
        .and_then(|v| v.first().cloned())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Provider normalisation / labelling
// ---------------------------------------------------------------------------

/// Normalize provider aliases to Hermes' canonical provider ids.
///
/// `"auto"` passes through unchanged.
pub fn normalize_provider(provider: Option<&str>) -> String {
    let normalized = provider.unwrap_or("openrouter").trim().to_lowercase();
    match provider_aliases().get(normalized.as_str()) {
        Some(&canon) => canon.to_string(),
        None => normalized,
    }
}

/// Return a human-friendly label for a provider id or alias.
pub fn provider_label(provider: Option<&str>) -> String {
    let original = provider.unwrap_or("openrouter").trim().to_string();
    let normalized_lower = original.to_lowercase();
    if normalized_lower == "auto" {
        return "Auto".to_string();
    }
    let normalized = normalize_provider(Some(&normalized_lower));
    provider_labels()
        .get(&normalized)
        .cloned()
        .unwrap_or_else(|| {
            if original.is_empty() {
                "OpenRouter".to_string()
            } else {
                original
            }
        })
}

/// All provider IDs and aliases valid for the `provider:model` syntax.
pub fn known_provider_names() -> HashSet<String> {
    let mut s: HashSet<String> = HashSet::new();
    for k in provider_labels().keys() {
        s.insert(k.clone());
    }
    for k in provider_aliases().keys() {
        s.insert(k.to_string());
    }
    s.insert("openrouter".into());
    s.insert("custom".into());
    s
}

// ---------------------------------------------------------------------------
// Nous Portal helpers
// ---------------------------------------------------------------------------

fn parse_price_zero(p: &Value, key: &str, default: &str) -> Option<bool> {
    // Mirrors float(p.get(key, default)) == 0 with TypeError/ValueError → None.
    let v = p.get(key);
    let s: String = match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        None => default.to_string(),
        Some(Value::Null) => return None, // float(None) -> TypeError
        _ => return None,
    };
    match s.trim().parse::<f64>() {
        Ok(f) => Some(f == 0.0),
        Err(_) => None,
    }
}

/// Return True if `model_id` has zero-cost prompt AND completion pricing.
pub fn is_model_free(model_id: &str, pricing: &HashMap<String, HashMap<String, String>>) -> bool {
    let p = match pricing.get(model_id) {
        Some(p) if !p.is_empty() => p,
        _ => return false,
    };
    let parse = |key: &str| -> Result<f64, ()> {
        p.get(key)
            .map(|s| s.as_str())
            .unwrap_or("1")
            .trim()
            .parse::<f64>()
            .map_err(|_| ())
    };
    match (parse("prompt"), parse("completion")) {
        (Ok(a), Ok(b)) => a == 0.0 && b == 0.0,
        _ => false,
    }
}

/// Build a blocking reqwest client with the given timeout (seconds, fractional).
fn build_client(timeout_secs: f64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

/// Fetch the user's Nous Portal account/subscription info.
///
/// Returns the parsed JSON value (`Value::Object`) on success, or
/// `Value::Object(empty)` on any failure.
pub fn fetch_nous_account_tier(access_token: &str, portal_base_url: &str) -> Value {
    let base = if portal_base_url.is_empty() {
        "https://portal.nousresearch.com".to_string()
    } else {
        portal_base_url.to_string()
    };
    let base = base.trim_end_matches('/');
    let url = format!("{}/api/oauth/account", base);
    let client = build_client(8.0);
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", access_token))
        .header("Accept", "application/json")
        .send();
    match resp {
        Ok(r) => match r.json::<Value>() {
            Ok(v) => v,
            Err(_) => Value::Object(Default::default()),
        },
        Err(_) => Value::Object(Default::default()),
    }
}

/// Return True if the account info indicates a free (unpaid) tier.
pub fn is_nous_free_tier(account_info: &Value) -> bool {
    let sub = match account_info.get("subscription") {
        Some(v) if v.is_object() => v,
        _ => return false,
    };
    let charge = match sub.get("monthly_charge") {
        Some(Value::Null) | None => return false,
        Some(v) => v,
    };
    match charge {
        Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        Value::String(s) => s.trim().parse::<f64>().map(|f| f == 0.0).unwrap_or(false),
        _ => false,
    }
}

/// Split Nous models into (selectable, unavailable) based on user tier.
pub fn partition_nous_models_by_tier(
    model_ids: &[String],
    pricing: &HashMap<String, HashMap<String, String>>,
    free_tier: bool,
) -> (Vec<String>, Vec<String>) {
    if !free_tier {
        return (model_ids.to_vec(), Vec::new());
    }
    if pricing.is_empty() {
        return (model_ids.to_vec(), Vec::new());
    }
    let mut selectable = Vec::new();
    let mut unavailable = Vec::new();
    for mid in model_ids {
        if is_model_free(mid, pricing) {
            selectable.push(mid.clone());
        } else {
            unavailable.push(mid.clone());
        }
    }
    (selectable, unavailable)
}

// ── Free-tier TTL cache ──
const FREE_TIER_CACHE_TTL: u64 = 180; // seconds
static FREE_TIER_CACHE: Mutex<Option<(bool, Instant)>> = Mutex::new(None);

/// Check whether the current Nous Portal user is on a free tier.
///
/// The Python version resolves the token via `hermes_cli.auth`. That module is
/// not available here, so callers must supply `(access_token, portal_base_url)`.
/// Pass `None` to reproduce the "no credentials → assume paid" fallback (the
/// import-failure path in Python). Results are cached for
/// `FREE_TIER_CACHE_TTL` seconds.
pub fn check_nous_free_tier(creds: Option<(&str, &str)>) -> bool {
    let now = Instant::now();
    {
        let guard = FREE_TIER_CACHE.lock().unwrap();
        if let Some((cached_result, cached_at)) = *guard {
            if now.duration_since(cached_at).as_secs() < FREE_TIER_CACHE_TTL {
                return cached_result;
            }
        }
    }
    let result = match creds {
        Some((token, portal)) if !token.is_empty() => {
            let info = fetch_nous_account_tier(token, portal);
            is_nous_free_tier(&info)
        }
        _ => false,
    };
    *FREE_TIER_CACHE.lock().unwrap() = Some((result, now));
    result
}

// ── Nous recommended models ──
pub const NOUS_RECOMMENDED_MODELS_PATH: &str = "/api/nous/recommended-models";
const NOUS_RECOMMENDED_CACHE_TTL: u64 = 600; // seconds
static NOUS_RECOMMENDED_CACHE: Mutex<Option<HashMap<String, (Value, Instant)>>> = Mutex::new(None);

/// Fetch the Nous Portal's curated recommended-models payload.
///
/// Public endpoint, no auth. Cached per portal URL for
/// `NOUS_RECOMMENDED_CACHE_TTL` seconds; `force_refresh` bypasses the cache.
pub fn fetch_nous_recommended_models(
    portal_base_url: &str,
    timeout: f64,
    force_refresh: bool,
) -> Value {
    let base = if portal_base_url.is_empty() {
        "https://portal.nousresearch.com".to_string()
    } else {
        portal_base_url.to_string()
    };
    let base = base.trim_end_matches('/').to_string();
    let now = Instant::now();

    if !force_refresh {
        let guard = NOUS_RECOMMENDED_CACHE.lock().unwrap();
        if let Some(map) = guard.as_ref() {
            if let Some((payload, cached_at)) = map.get(&base) {
                if now.duration_since(*cached_at).as_secs() < NOUS_RECOMMENDED_CACHE_TTL {
                    return payload.clone();
                }
            }
        }
    }

    let url = format!("{}{}", base, NOUS_RECOMMENDED_MODELS_PATH);
    let client = build_client(timeout);
    let data = match client.get(&url).header("Accept", "application/json").send() {
        Ok(r) => match r.json::<Value>() {
            Ok(v) if v.is_object() => v,
            _ => Value::Object(Default::default()),
        },
        Err(_) => Value::Object(Default::default()),
    };

    let mut guard = NOUS_RECOMMENDED_CACHE.lock().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert(base, (data.clone(), now));
    data
}

/// Pull the `modelName` field from a recommended-model entry.
pub fn extract_model_name(entry: &Value) -> Option<String> {
    let obj = entry.as_object()?;
    let name = obj.get("modelName")?.as_str()?;
    let trimmed = name.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Return the Portal's recommended model name for an auxiliary task.
///
/// When `free_tier` is `None`, it is auto-detected via [`check_nous_free_tier`]
/// using `tier_creds` (mirrors the Python lazy auth resolution). `portal_base_url`
/// empty defaults to the public Portal URL.
pub fn get_nous_recommended_aux_model(
    vision: bool,
    free_tier: Option<bool>,
    portal_base_url: &str,
    force_refresh: bool,
    tier_creds: Option<(&str, &str)>,
) -> Option<String> {
    let base = if portal_base_url.is_empty() {
        "https://portal.nousresearch.com".to_string()
    } else {
        portal_base_url.to_string()
    };
    let payload = fetch_nous_recommended_models(&base, 5.0, force_refresh);
    if !payload.is_object() || payload.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        return None;
    }

    let free_tier = free_tier.unwrap_or_else(|| check_nous_free_tier(tier_creds));

    let (paid_key, free_key) = if vision {
        ("paidRecommendedVisionModel", "freeRecommendedVisionModel")
    } else {
        (
            "paidRecommendedCompactionModel",
            "freeRecommendedCompactionModel",
        )
    };

    let candidates: Vec<&str> = if free_tier {
        vec![free_key]
    } else {
        vec![paid_key, free_key]
    };
    for key in candidates {
        if let Some(entry) = payload.get(key) {
            if let Some(name) = extract_model_name(entry) {
                return Some(name);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// OpenRouter catalog
// ---------------------------------------------------------------------------

fn json_float_zero(v: Option<&Value>, default: &str) -> Option<f64> {
    let s = match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        None => default.to_string(),
        Some(Value::Null) => return None,
        _ => return None,
    };
    s.trim().parse::<f64>().ok()
}

/// Return True when both prompt and completion pricing are zero.
pub fn openrouter_model_is_free(pricing: &Value) -> bool {
    let obj = match pricing {
        Value::Object(_) => pricing,
        _ => return false,
    };
    match (
        json_float_zero(obj.get("prompt"), "0"),
        json_float_zero(obj.get("completion"), "0"),
    ) {
        (Some(a), Some(b)) => a == 0.0 && b == 0.0,
        _ => false,
    }
}

/// Return True when the model advertises tool-calling support, permissively.
pub fn openrouter_model_supports_tools(item: &Value) -> bool {
    let obj = match item {
        Value::Object(_) => item,
        _ => return true,
    };
    match obj.get("supported_parameters") {
        Some(Value::Array(arr)) => arr
            .iter()
            .any(|v| v.as_str().map(|s| s == "tools").unwrap_or(false)),
        _ => true,
    }
}

/// Return the curated OpenRouter picker list, refreshed from the live catalog
/// when possible.
///
/// The Python version optionally pulls a "remote curated manifest" via
/// `hermes_cli.model_catalog`. That module is not ported, so the manifest
/// branch is treated as the failing path and the in-repo `OPENROUTER_MODELS`
/// snapshot is used for the preferred-id ordering.
pub fn fetch_openrouter_models(timeout: f64, force_refresh: bool) -> Vec<(String, String)> {
    {
        let guard = OPENROUTER_CATALOG_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if !force_refresh {
                return cache.clone();
            }
        }
    }

    let fallback: Vec<(String, String)> = OPENROUTER_MODELS
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let preferred_ids: Vec<String> = fallback.iter().map(|(m, _)| m.clone()).collect();

    let cached_or_fallback = || -> Vec<(String, String)> {
        let guard = OPENROUTER_CATALOG_CACHE.lock().unwrap();
        guard.clone().unwrap_or_else(|| fallback.clone())
    };

    let client = build_client(timeout);
    let payload: Value = match client
        .get("https://openrouter.ai/api/v1/models")
        .header("Accept", "application/json")
        .send()
        .and_then(|r| r.json::<Value>())
    {
        Ok(v) => v,
        Err(_) => return cached_or_fallback(),
    };

    let live_items = match payload.get("data") {
        Some(Value::Array(a)) => a.clone(),
        _ => return cached_or_fallback(),
    };

    let mut live_by_id: HashMap<String, Value> = HashMap::new();
    for item in &live_items {
        if !item.is_object() {
            continue;
        }
        let mid = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if mid.is_empty() {
            continue;
        }
        live_by_id.insert(mid, item.clone());
    }

    let mut curated: Vec<(String, String)> = Vec::new();
    for preferred_id in &preferred_ids {
        let live_item = match live_by_id.get(preferred_id) {
            Some(v) => v,
            None => continue,
        };
        if !openrouter_model_supports_tools(live_item) {
            continue;
        }
        let pricing = live_item.get("pricing").cloned().unwrap_or(Value::Null);
        let desc = if openrouter_model_is_free(&pricing) {
            "free"
        } else {
            ""
        };
        curated.push((preferred_id.clone(), desc.to_string()));
    }

    if curated.is_empty() {
        return cached_or_fallback();
    }

    curated[0].1 = "recommended".to_string();
    *OPENROUTER_CATALOG_CACHE.lock().unwrap() = Some(curated.clone());
    curated
}

/// Return just the OpenRouter model-id strings.
pub fn model_ids(force_refresh: bool) -> Vec<String> {
    fetch_openrouter_models(8.0, force_refresh)
        .into_iter()
        .map(|(m, _)| m)
        .collect()
}

/// Return the curated Nous Portal model-id list.
///
/// The Python version prefers a remote manifest (`hermes_cli.model_catalog`),
/// falling back to `_PROVIDER_MODELS["nous"]`. The manifest is not ported, so
/// the static snapshot is returned.
pub fn get_curated_nous_model_ids() -> Vec<String> {
    provider_models_map()
        .get("nous")
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// AI Gateway catalog
// ---------------------------------------------------------------------------

/// Return True if an AI Gateway model has $0 input AND output pricing.
pub fn ai_gateway_model_is_free(pricing: &Value) -> bool {
    let obj = match pricing {
        Value::Object(_) => pricing,
        _ => return false,
    };
    match (
        json_float_zero(obj.get("input"), "0"),
        json_float_zero(obj.get("output"), "0"),
    ) {
        (Some(a), Some(b)) => a == 0.0 && b == 0.0,
        _ => false,
    }
}

/// Return the curated AI Gateway picker list, refreshed from the live catalog.
pub fn fetch_ai_gateway_models(timeout: f64, force_refresh: bool) -> Vec<(String, String)> {
    {
        let guard = AI_GATEWAY_CATALOG_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if !force_refresh {
                return cache.clone();
            }
        }
    }

    let fallback: Vec<(String, String)> = VERCEL_AI_GATEWAY_MODELS
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let preferred_ids: Vec<String> = fallback.iter().map(|(m, _)| m.clone()).collect();

    let cached_or_fallback = || -> Vec<(String, String)> {
        let guard = AI_GATEWAY_CATALOG_CACHE.lock().unwrap();
        guard.clone().unwrap_or_else(|| fallback.clone())
    };

    let url = format!("{}/models", AI_GATEWAY_BASE_URL.trim_end_matches('/'));
    let client = build_client(timeout);
    let payload: Value = match client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .and_then(|r| r.json::<Value>())
    {
        Ok(v) => v,
        Err(_) => return cached_or_fallback(),
    };

    let live_items = match payload.get("data") {
        Some(Value::Array(a)) => a.clone(),
        _ => return cached_or_fallback(),
    };

    // live_by_id preserving insertion order for the moonshot scan.
    let mut order: Vec<String> = Vec::new();
    let mut live_by_id: HashMap<String, Value> = HashMap::new();
    for item in &live_items {
        if !item.is_object() {
            continue;
        }
        let mid = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if mid.is_empty() {
            continue;
        }
        if !live_by_id.contains_key(&mid) {
            order.push(mid.clone());
        }
        live_by_id.insert(mid, item.clone());
    }

    let mut curated: Vec<(String, String)> = Vec::new();
    for preferred_id in &preferred_ids {
        let live_item = match live_by_id.get(preferred_id) {
            Some(v) => v,
            None => continue,
        };
        let pricing = live_item.get("pricing").cloned().unwrap_or(Value::Null);
        let desc = if ai_gateway_model_is_free(&pricing) {
            "free"
        } else {
            ""
        };
        curated.push((preferred_id.clone(), desc.to_string()));
    }

    if curated.is_empty() {
        return cached_or_fallback();
    }

    // Auto-promote a free Moonshot model if the live catalog offers one.
    let free_moonshot = order.iter().find(|mid| {
        if !mid.starts_with("moonshotai/") {
            return false;
        }
        let pricing = live_by_id
            .get(*mid)
            .and_then(|i| i.get("pricing"))
            .cloned()
            .unwrap_or(Value::Null);
        ai_gateway_model_is_free(&pricing)
    });

    if let Some(fm) = free_moonshot.cloned() {
        curated.retain(|(mid, _)| mid != &fm);
        curated.insert(0, (fm, "recommended".to_string()));
    } else {
        curated[0].1 = "recommended".to_string();
    }

    *AI_GATEWAY_CATALOG_CACHE.lock().unwrap() = Some(curated.clone());
    curated
}

/// Return just the AI Gateway model-id strings.
pub fn ai_gateway_model_ids(force_refresh: bool) -> Vec<String> {
    fetch_ai_gateway_models(8.0, force_refresh)
        .into_iter()
        .map(|(m, _)| m)
        .collect()
}

// ---------------------------------------------------------------------------
// Pricing helpers
// ---------------------------------------------------------------------------

// Cache: base_url -> {model_id -> {prompt, completion, ...}}.
type PricingMap = HashMap<String, HashMap<String, String>>;
static PRICING_CACHE: Mutex<Option<HashMap<String, PricingMap>>> = Mutex::new(None);

fn pricing_cache_get(key: &str) -> Option<PricingMap> {
    let guard = PRICING_CACHE.lock().unwrap();
    guard.as_ref().and_then(|m| m.get(key).cloned())
}

fn pricing_cache_set(key: &str, value: PricingMap) {
    let mut guard = PRICING_CACHE.lock().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert(key.to_string(), value);
}

/// Convert a per-token price string to a human-friendly `$/Mtok` string.
pub fn format_price_per_mtok(per_token_str: &str) -> String {
    let val: f64 = match per_token_str.trim().parse::<f64>() {
        Ok(v) => v,
        Err(_) => return "?".to_string(),
    };
    if val == 0.0 {
        return "free".to_string();
    }
    let per_m = val * 1_000_000.0;
    format!("${:.2}", per_m)
}

/// Build a column-aligned model+pricing table for terminal display.
pub fn format_model_pricing_table(
    models: &[(String, String)],
    pricing_map: &HashMap<String, HashMap<String, String>>,
    current_model: &str,
    indent: &str,
) -> Vec<String> {
    if models.is_empty() {
        return Vec::new();
    }

    // (model_id, input, output, cache, is_current)
    let mut rows: Vec<(String, String, String, String, bool)> = Vec::new();
    let mut has_cache = false;
    for (mid, _desc) in models {
        let is_cur = mid == current_model;
        let (inp, out, cache) = match pricing_map.get(mid) {
            Some(p) => {
                let inp = format_price_per_mtok(p.get("prompt").map(|s| s.as_str()).unwrap_or(""));
                let out =
                    format_price_per_mtok(p.get("completion").map(|s| s.as_str()).unwrap_or(""));
                let cache_read = p.get("input_cache_read").map(|s| s.as_str()).unwrap_or("");
                let cache = if !cache_read.is_empty() {
                    format_price_per_mtok(cache_read)
                } else {
                    String::new()
                };
                if !cache.is_empty() {
                    has_cache = true;
                }
                (inp, out, cache)
            }
            None => (String::new(), String::new(), String::new()),
        };
        rows.push((mid.clone(), inp, out, cache, is_cur));
    }

    let name_col = rows.iter().map(|r| r.0.chars().count()).max().unwrap_or(0) + 2;
    let max_or = |f: &dyn Fn(&(String, String, String, String, bool)) -> &str, default: usize| -> usize {
        rows.iter()
            .filter_map(|r| {
                let s = f(r);
                if s.is_empty() {
                    None
                } else {
                    Some(s.chars().count())
                }
            })
            .max()
            .unwrap_or(default)
    };
    let in_w = max_or(&|r| r.1.as_str(), 4);
    let out_w = max_or(&|r| r.2.as_str(), 4);
    let price_col = in_w.max(out_w).max(3);
    let cache_col = if has_cache {
        max_or(&|r| r.3.as_str(), 4).max(5)
    } else {
        0
    };

    let mut lines: Vec<String> = Vec::new();

    if has_cache {
        lines.push(format!(
            "{indent}{:<nc$} {:>pc$}  {:>pc$}  {:>cc$}  /Mtok",
            "Model",
            "In",
            "Out",
            "Cache",
            indent = indent,
            nc = name_col,
            pc = price_col,
            cc = cache_col
        ));
        lines.push(format!(
            "{indent}{} {}  {}  {}",
            "-".repeat(name_col),
            "-".repeat(price_col),
            "-".repeat(price_col),
            "-".repeat(cache_col),
            indent = indent
        ));
    } else {
        lines.push(format!(
            "{indent}{:<nc$} {:>pc$}  {:>pc$}  /Mtok",
            "Model",
            "In",
            "Out",
            indent = indent,
            nc = name_col,
            pc = price_col
        ));
        lines.push(format!(
            "{indent}{} {}  {}",
            "-".repeat(name_col),
            "-".repeat(price_col),
            "-".repeat(price_col),
            indent = indent
        ));
    }

    for (mid, inp, out, cache, is_cur) in &rows {
        let marker = if *is_cur { "  ← current" } else { "" };
        if has_cache {
            lines.push(format!(
                "{indent}{:<nc$} {:>pc$}  {:>pc$}  {:>cc$}{marker}",
                mid,
                inp,
                out,
                cache,
                indent = indent,
                nc = name_col,
                pc = price_col,
                cc = cache_col,
                marker = marker
            ));
        } else {
            lines.push(format!(
                "{indent}{:<nc$} {:>pc$}  {:>pc$}{marker}",
                mid,
                inp,
                out,
                indent = indent,
                nc = name_col,
                pc = price_col,
                marker = marker
            ));
        }
    }

    lines
}

/// Fetch `/v1/models` and return `{model_id: {prompt, completion}}` pricing.
///
/// Cached per `base_url`. Works with any OpenRouter-compatible endpoint.
pub fn fetch_models_with_pricing(
    api_key: Option<&str>,
    base_url: &str,
    timeout: f64,
    force_refresh: bool,
) -> HashMap<String, HashMap<String, String>> {
    let cache_key = base_url.trim_end_matches('/').to_string();
    if !force_refresh {
        if let Some(v) = pricing_cache_get(&cache_key) {
            return v;
        }
    }

    let url = format!("{}/v1/models", cache_key.trim_end_matches('/'));
    let client = build_client(timeout);
    let mut req = client
        .get(&url)
        .header("Accept", "application/json")
        .header("User-Agent", hermes_user_agent());
    if let Some(key) = api_key {
        if !key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
    }

    let payload: Value = match req.send().and_then(|r| r.json::<Value>()) {
        Ok(v) => v,
        Err(_) => {
            pricing_cache_set(&cache_key, HashMap::new());
            return HashMap::new();
        }
    };

    let mut result: HashMap<String, HashMap<String, String>> = HashMap::new();
    if let Some(Value::Array(items)) = payload.get("data") {
        for item in items {
            let mid = item.get("id").and_then(|v| v.as_str());
            let pricing = item.get("pricing");
            if let (Some(mid), Some(pricing)) = (mid, pricing) {
                if !pricing.is_object() {
                    continue;
                }
                let mut entry: HashMap<String, String> = HashMap::new();
                entry.insert(
                    "prompt".into(),
                    value_to_str(pricing.get("prompt")),
                );
                entry.insert(
                    "completion".into(),
                    value_to_str(pricing.get("completion")),
                );
                if let Some(v) = pricing.get("input_cache_read") {
                    if value_is_truthy(v) {
                        entry.insert("input_cache_read".into(), value_to_str(Some(v)));
                    }
                }
                if let Some(v) = pricing.get("input_cache_write") {
                    if value_is_truthy(v) {
                        entry.insert("input_cache_write".into(), value_to_str(Some(v)));
                    }
                }
                result.insert(mid.to_string(), entry);
            }
        }
    }

    pricing_cache_set(&cache_key, result.clone());
    result
}

/// Fetch Vercel AI Gateway `/v1/models` and return hermes-shaped pricing.
///
/// Translates Vercel's `input`/`output` fields to `prompt`/`completion`.
pub fn fetch_ai_gateway_pricing(
    timeout: f64,
    force_refresh: bool,
) -> HashMap<String, HashMap<String, String>> {
    let cache_key = AI_GATEWAY_BASE_URL.trim_end_matches('/').to_string();
    if !force_refresh {
        if let Some(v) = pricing_cache_get(&cache_key) {
            return v;
        }
    }

    let url = format!("{}/models", cache_key);
    let client = build_client(timeout);
    let payload: Value = match client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .and_then(|r| r.json::<Value>())
    {
        Ok(v) => v,
        Err(_) => {
            pricing_cache_set(&cache_key, HashMap::new());
            return HashMap::new();
        }
    };

    let mut result: HashMap<String, HashMap<String, String>> = HashMap::new();
    if let Some(Value::Array(items)) = payload.get("data") {
        for item in items {
            if !item.is_object() {
                continue;
            }
            let mid = item.get("id").and_then(|v| v.as_str());
            let pricing = item.get("pricing");
            let (mid, pricing) = match (mid, pricing) {
                (Some(m), Some(p)) if p.is_object() => (m, p),
                _ => continue,
            };
            let mut entry: HashMap<String, String> = HashMap::new();
            entry.insert("prompt".into(), value_to_str(pricing.get("input")));
            entry.insert("completion".into(), value_to_str(pricing.get("output")));
            if let Some(v) = pricing.get("input_cache_read") {
                if value_is_truthy(v) {
                    entry.insert("input_cache_read".into(), value_to_str(Some(v)));
                }
            }
            if let Some(v) = pricing.get("input_cache_write") {
                if value_is_truthy(v) {
                    entry.insert("input_cache_write".into(), value_to_str(Some(v)));
                }
            }
            result.insert(mid.to_string(), entry);
        }
    }

    pricing_cache_set(&cache_key, result.clone());
    result
}

/// `str(x)` for a JSON value matching Python `str(pricing.get(k, ""))`.
fn value_to_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Some(Value::Null) => "None".into(),
        Some(other) => other.to_string(),
        None => "".into(),
    }
}

/// Python truthiness for the `if pricing.get(...)` guard.
fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Best-effort OpenRouter API key from the environment.
pub fn resolve_openrouter_api_key() -> String {
    std::env::var("OPENROUTER_API_KEY")
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Return live pricing for providers that support it.
///
/// The `nous` branch needs resolved credentials (`api_key`, `base_url`) which
/// the Python version pulls from `hermes_cli.auth`. Supply them via
/// `nous_creds`; pass `None` to reproduce the import-failure path.
pub fn get_pricing_for_provider(
    provider: &str,
    force_refresh: bool,
    nous_creds: Option<(&str, &str)>,
) -> HashMap<String, HashMap<String, String>> {
    let normalized = normalize_provider(Some(provider));
    if normalized == "openrouter" {
        return fetch_models_with_pricing(
            Some(&resolve_openrouter_api_key()),
            "https://openrouter.ai/api",
            8.0,
            force_refresh,
        );
    }
    if normalized == "ai-gateway" {
        return fetch_ai_gateway_pricing(8.0, force_refresh);
    }
    if normalized == "nous" {
        if let Some((api_key, base_url)) = nous_creds {
            if !base_url.is_empty() {
                let mut stripped = base_url.trim_end_matches('/').to_string();
                if stripped.ends_with("/v1") {
                    stripped.truncate(stripped.len() - 3);
                }
                return fetch_models_with_pricing(Some(api_key), &stripped, 8.0, force_refresh);
            }
        }
    }
    HashMap::new()
}

// ---------------------------------------------------------------------------
// /model input parsing
// ---------------------------------------------------------------------------

/// Parse `/model` input into `(provider, model)`.
pub fn parse_model_input(raw: &str, current_provider: &str) -> (String, String) {
    let stripped = raw.trim().to_string();
    if let Some(colon) = stripped.find(':') {
        if colon > 0 {
            let provider_part = stripped[..colon].trim().to_lowercase();
            let model_part = stripped[colon + 1..].trim().to_string();
            if !provider_part.is_empty()
                && !model_part.is_empty()
                && known_provider_names().contains(&provider_part)
            {
                if provider_part == "custom" && model_part.contains(':') {
                    let second_colon = model_part.find(':').unwrap();
                    let custom_name = model_part[..second_colon].trim().to_string();
                    let actual_model = model_part[second_colon + 1..].trim().to_string();
                    if !custom_name.is_empty() && !actual_model.is_empty() {
                        return (format!("custom:{}", custom_name), actual_model);
                    }
                }
                return (normalize_provider(Some(&provider_part)), model_part);
            }
        }
    }
    (current_provider.to_string(), stripped)
}

/// Get the custom endpoint base_url from config.yaml.
///
/// The Python version reads it from `hermes_cli.config.load_config`. The native
/// `crate::config` loader exposes the parsed config; this reads
/// `model.base_url`. Returns `""` on any failure.
pub fn get_custom_base_url() -> String {
    // Best-effort: read config.yaml's model.base_url. We avoid a hard
    // dependency on a specific config API by parsing the YAML directly.
    let path = crate::mod_hermes_constants::get_hermes_home().join("config.yaml");
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let doc: serde_yaml::Value = match serde_yaml::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    doc.get("model")
        .and_then(|m| m.get("base_url"))
        .and_then(|b| b.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Provider catalog detection / aggregation
// ---------------------------------------------------------------------------

const AGGREGATOR_PROVIDERS: &[&str] = &["nous", "openrouter", "ai-gateway", "copilot", "kilocode"];

fn provider_keys(provider: &str) -> HashSet<String> {
    let key = provider.trim().to_lowercase();
    let normalized = normalize_provider(Some(provider));
    let mut s = HashSet::new();
    if !key.is_empty() {
        s.insert(key);
    }
    if !normalized.is_empty() {
        s.insert(normalized);
    }
    s
}

fn model_in_provider_catalog(name_lower: &str, providers: &HashSet<String>) -> bool {
    let pm = provider_models_map();
    providers.iter().any(|provider| {
        pm.get(provider)
            .map(|models| models.iter().any(|m| m.to_lowercase() == name_lower))
            .unwrap_or(false)
    })
}

/// Auto-detect a provider from static catalogs only.
///
/// Returns `(provider_id, model_name)` or `None`.
///
/// The Python source consults `hermes_cli.model_switch.MODEL_ALIASES` for short
/// aliases (sonnet/opus). That table is not ported, so the alias-resolution
/// step is treated as the failing import path (returns `None`) and detection
/// proceeds with the catalog-only logic, preserving the rest of the behaviour.
pub fn detect_static_provider_for_model(
    model_name: &str,
    current_provider: &str,
) -> Option<(String, String)> {
    let name = model_name.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let name_lower = name.to_lowercase();
    let current_keys = provider_keys(current_provider);

    // (alias resolution via MODEL_ALIASES skipped — not ported)

    // Step 0: bare provider name typed as model.
    let aliases = provider_aliases();
    let resolved_provider = aliases
        .get(name_lower.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| name_lower.clone());
    if resolved_provider != "custom" && resolved_provider != "openrouter" {
        let pm = provider_models_map();
        if let Some(default_models) = pm.get(&resolved_provider) {
            if provider_labels().contains_key(&resolved_provider)
                && !default_models.is_empty()
                && !current_keys.contains(&resolved_provider)
            {
                return Some((resolved_provider, default_models[0].clone()));
            }
        }
    }

    if model_in_provider_catalog(&name_lower, &current_keys) {
        return None;
    }

    // Step 1: check static provider catalogs for a direct match.
    let pm = provider_models_map();
    // Iterate deterministically: order doesn't matter for correctness since the
    // condition is exact-match exclusive of current/aggregator providers; the
    // Python dict iteration order is insertion order, but only one provider can
    // match a given exact id in practice.
    for (pid, models) in pm.iter() {
        if current_keys.contains(pid) || AGGREGATOR_PROVIDERS.contains(&pid.as_str()) {
            continue;
        }
        if models.iter().any(|m| m.to_lowercase() == name_lower) {
            return Some((pid.clone(), name.clone()));
        }
    }

    None
}

/// Auto-detect the best provider for a model name.
pub fn detect_provider_for_model(
    model_name: &str,
    current_provider: &str,
) -> Option<(String, String)> {
    let name = model_name.trim().to_string();
    if name.is_empty() {
        return None;
    }

    if let Some(m) = detect_static_provider_for_model(&name, current_provider) {
        return Some(m);
    }
    if model_in_provider_catalog(&name.to_lowercase(), &provider_keys(current_provider)) {
        return None;
    }

    // Step 2: check OpenRouter catalog.
    if let Some(or_slug) = find_openrouter_slug(&name) {
        if current_provider != "openrouter" {
            return Some(("openrouter".to_string(), or_slug));
        }
        if or_slug != name {
            return Some(("openrouter".to_string(), or_slug));
        }
        return None;
    }

    None
}

/// Find the full OpenRouter model slug for a bare or partial model name.
pub fn find_openrouter_slug(model_name: &str) -> Option<String> {
    let name_lower = model_name.trim().to_lowercase();
    if name_lower.is_empty() {
        return None;
    }

    let ids = model_ids(false);

    for mid in &ids {
        if name_lower == mid.to_lowercase() {
            return Some(mid.clone());
        }
    }

    for mid in &ids {
        if let Some((_, model_part)) = mid.split_once('/') {
            if name_lower == model_part.to_lowercase() {
                return Some(mid.clone());
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Provider listing
// ---------------------------------------------------------------------------

/// Info about a single provider for the `provider:model` syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableProvider {
    pub id: String,
    pub label: String,
    pub aliases: Vec<String>,
    pub authenticated: bool,
}

/// Return info about all providers the user could use with `provider:model`.
///
/// The Python version checks live credential status via `hermes_cli.auth`.
/// That module is not ported; the credential check is therefore the failing
/// `try/except` path and `authenticated` is always `false` — except for the
/// `custom`/`openrouter` cases which can be derived locally (custom base URL
/// present / `OPENROUTER_API_KEY` set).
pub fn list_available_providers() -> Vec<AvailableProvider> {
    let mut provider_order: Vec<String> = canonical_providers().into_iter().map(|p| p.slug).collect();
    provider_order.push("custom".to_string());

    // Reverse alias map.
    let mut aliases_for: HashMap<String, Vec<String>> = HashMap::new();
    for (alias, canonical) in provider_aliases() {
        aliases_for
            .entry(canonical.to_string())
            .or_default()
            .push(alias.to_string());
    }

    let labels = provider_labels();
    let mut result = Vec::new();
    for pid in provider_order {
        let label = labels.get(&pid).cloned().unwrap_or_else(|| pid.clone());
        let alias_list = aliases_for.get(&pid).cloned().unwrap_or_default();
        // Credential check: only custom/openrouter resolvable locally.
        let has_creds = if pid == "custom" {
            !get_custom_base_url().trim().is_empty()
        } else if pid == "openrouter" {
            !std::env::var("OPENROUTER_API_KEY")
                .unwrap_or_default()
                .trim()
                .is_empty()
        } else {
            false
        };
        result.push(AvailableProvider {
            id: pid,
            label,
            aliases: alias_list,
            authenticated: has_creds,
        });
    }
    result
}

/// Return `(model_id, description)` tuples for a provider's model list.
///
/// `force_refresh` is forwarded to the OpenRouter path. Falls back to the
/// static catalog when no live list is available.
pub fn curated_models_for_provider(
    provider: Option<&str>,
    force_refresh: bool,
) -> Vec<(String, String)> {
    let normalized = normalize_provider(provider);
    if normalized == "openrouter" {
        return fetch_openrouter_models(8.0, force_refresh);
    }

    let live = provider_model_ids(Some(&normalized), force_refresh);
    if !live.is_empty() {
        return live.into_iter().map(|m| (m, String::new())).collect();
    }

    provider_models_map()
        .get(&normalized)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|m| (m, String::new()))
        .collect()
}

// ---------------------------------------------------------------------------
// Fast / priority mode helpers
// ---------------------------------------------------------------------------

const OPENAI_FAST_MODE_PREFIXES: &[&str] = &["gpt-", "o1", "o3", "o4"];

/// Strip `vendor/` prefix from a model ID, lowercased.
pub fn strip_vendor_prefix(model_id: &str) -> String {
    let raw = model_id.trim().to_lowercase();
    match raw.split_once('/') {
        Some((_, rest)) => rest.to_string(),
        None => raw,
    }
}

fn is_openai_fast_model(model_id: Option<&str>) -> bool {
    let raw = strip_vendor_prefix(model_id.unwrap_or(""));
    let base = raw.split(':').next().unwrap_or("");
    if base.is_empty() {
        return false;
    }
    if base.contains("codex") {
        return false;
    }
    OPENAI_FAST_MODE_PREFIXES.iter().any(|p| base.starts_with(p))
}

fn is_anthropic_fast_model(model_id: Option<&str>) -> bool {
    let raw = strip_vendor_prefix(model_id.unwrap_or(""));
    let base = raw.split(':').next().unwrap_or("");
    if !base.starts_with("claude-") {
        return false;
    }
    base.contains("opus-4-6") || base.contains("opus-4.6")
}

/// Whether Hermes should expose the `/fast` toggle for this model.
pub fn model_supports_fast_mode(model_id: Option<&str>) -> bool {
    is_anthropic_fast_model(model_id) || is_openai_fast_model(model_id)
}

/// Return request_overrides for fast/priority mode, or `None` if unsupported.
pub fn resolve_fast_mode_overrides(model_id: Option<&str>) -> Option<Value> {
    if !model_supports_fast_mode(model_id) {
        return None;
    }
    if is_anthropic_fast_model(model_id) {
        Some(serde_json::json!({"speed": "fast"}))
    } else {
        Some(serde_json::json!({"service_tier": "priority"}))
    }
}

// ---------------------------------------------------------------------------
// models.dev preferred merge
// ---------------------------------------------------------------------------

/// Providers where models.dev is treated as authoritative.
pub const MODELS_DEV_PREFERRED: &[&str] = &[
    "opencode-go",
    "opencode-zen",
    "deepseek",
    "kilocode",
    "fireworks",
    "mistral",
    "togetherai",
    "cohere",
    "perplexity",
    "groq",
    "nvidia",
    "huggingface",
    "zai",
    "gemini",
    "google",
];

/// Merge curated list with fresh models.dev entries for a preferred provider.
///
/// The Python version pulls `agent.models_dev.list_agentic_models`. That live
/// registry is not ported, so the import-failure path (empty models.dev list)
/// is the default — the curated list is returned unchanged. Callers that have
/// models.dev entries can pass them via `mdev`.
pub fn merge_with_models_dev(_provider: &str, curated: &[String], mdev: &[String]) -> Vec<String> {
    if mdev.is_empty() {
        return curated.to_vec();
    }
    let mut seen_lower: HashSet<String> = HashSet::new();
    let mut merged: Vec<String> = Vec::new();
    for mid in mdev {
        let key = mid.to_lowercase();
        if seen_lower.contains(&key) {
            continue;
        }
        seen_lower.insert(key);
        merged.push(mid.clone());
    }
    for mid in curated {
        let key = mid.to_lowercase();
        if seen_lower.contains(&key) {
            continue;
        }
        seen_lower.insert(key);
        merged.push(mid.clone());
    }
    merged
}

/// Return the best known model catalog for a provider.
///
/// Live-endpoint branches that require resolved credentials from
/// `hermes_cli.auth` / `providers` / `agent.*` (Codex token, Nous creds, GMI,
/// Bedrock SDK discovery, etc.) are not reachable without those modules, so
/// this implements the *fallback* behaviour for each branch: it tries the
/// network paths that only need environment variables (openai, custom,
/// anthropic, ai-gateway, ollama-cloud, copilot) and otherwise returns the
/// curated static catalog (with the models.dev merge for preferred providers
/// when models.dev entries are unavailable — i.e. the curated list as-is).
pub fn provider_model_ids(provider: Option<&str>, force_refresh: bool) -> Vec<String> {
    let normalized = normalize_provider(provider);

    if normalized == "openrouter" {
        return model_ids(force_refresh);
    }
    if normalized == "openai-codex" {
        // No OAuth token resolvable here; matches the no-token fallback path.
        return crate::cli_codex_models::get_codex_model_ids(None);
    }
    if normalized == "copilot" || normalized == "copilot-acp" {
        if let Some(live) = fetch_github_models(None, 5.0) {
            if !live.is_empty() {
                return live;
            }
        }
        if normalized == "copilot-acp" {
            return provider_models_map()
                .get("copilot")
                .cloned()
                .unwrap_or_default();
        }
    }
    // nous / stepfun / gmi branches require resolved credentials → fallthrough.
    if normalized == "anthropic" {
        if let Some(live) = fetch_anthropic_models(5.0, None) {
            if !live.is_empty() {
                return live;
            }
        }
    }
    if normalized == "ai-gateway" {
        if let Some(live) = fetch_ai_gateway_models_lang(5.0) {
            if !live.is_empty() {
                return live;
            }
        }
    }
    if normalized == "ollama-cloud" {
        let live = fetch_ollama_cloud_models(None, None, force_refresh, &[]);
        if !live.is_empty() {
            return live;
        }
    }
    if normalized == "openai" {
        let api_key = std::env::var("OPENAI_API_KEY")
            .unwrap_or_default()
            .trim()
            .to_string();
        if !api_key.is_empty() {
            let base_raw = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_default()
                .trim()
                .trim_end_matches('/')
                .to_string();
            let base = if base_raw.is_empty() {
                "https://api.openai.com/v1".to_string()
            } else {
                base_raw
            };
            if let Some(live) = fetch_api_models(Some(&api_key), Some(&base), 5.0, None) {
                if !live.is_empty() {
                    return live;
                }
            }
        }
    }
    if normalized == "custom" {
        let base_url = get_custom_base_url();
        if !base_url.is_empty() {
            let api_key = std::env::var("CUSTOM_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty()))
                .or_else(|| {
                    std::env::var("OPENROUTER_API_KEY")
                        .ok()
                        .filter(|s| !s.is_empty())
                })
                .unwrap_or_default();
            if let Some(live) = fetch_api_models(Some(&api_key), Some(&base_url), 5.0, None) {
                if !live.is_empty() {
                    return live;
                }
            }
        }
    }
    // bedrock SDK discovery requires agent.bedrock_adapter → fallthrough.

    let curated_static = provider_models_map()
        .get(&normalized)
        .cloned()
        .unwrap_or_default();
    if MODELS_DEV_PREFERRED.contains(&normalized.as_str()) {
        return merge_with_models_dev(&normalized, &curated_static, &[]);
    }
    curated_static
}

// ---------------------------------------------------------------------------
// Anthropic /v1/models
// ---------------------------------------------------------------------------

/// Fetch available models from the Anthropic `/v1/models` endpoint.
///
/// The Python version resolves a token + OAuth betas via
/// `agent.anthropic_adapter`. Supply a `(token, is_oauth)` tuple; pass `None`
/// to reproduce the unresolvable-token path (returns `None`). For OAuth the
/// caller is responsible for the beta header set; this sends the minimal
/// required `anthropic-version` plus auth header.
pub fn fetch_anthropic_models(timeout: f64, token: Option<(&str, bool)>) -> Option<Vec<String>> {
    let (token, is_oauth) = token?;
    if token.is_empty() {
        return None;
    }

    let client = build_client(timeout);
    let do_request = |bearer_oauth: bool| -> Result<Value, reqwest::Error> {
        let mut req = client
            .get("https://api.anthropic.com/v1/models")
            .header("anthropic-version", "2023-06-01");
        if bearer_oauth {
            req = req.header("Authorization", format!("Bearer {}", token));
        } else {
            req = req.header("x-api-key", token);
        }
        req.send()?.json::<Value>()
    };

    let data = match do_request(is_oauth) {
        Ok(v) => v,
        Err(_) => return None,
    };

    let mut models: Vec<String> = data
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Sort: opus > sonnet > haiku, alphabetical within tier.
    models.sort_by(|a, b| {
        let key = |m: &str| -> (bool, bool, bool, String) {
            (
                !m.contains("opus"),
                !m.contains("sonnet"),
                !m.contains("haiku"),
                m.to_string(),
            )
        };
        key(a).cmp(&key(b))
    });
    Some(models)
}

// ---------------------------------------------------------------------------
// Copilot catalog
// ---------------------------------------------------------------------------

fn payload_items(payload: &Value) -> Vec<Value> {
    match payload {
        Value::Array(arr) => arr.iter().filter(|i| i.is_object()).cloned().collect(),
        Value::Object(_) => match payload.get("data") {
            Some(Value::Array(arr)) => arr.iter().filter(|i| i.is_object()).cloned().collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Standard headers for Copilot API requests.
///
/// The Python version delegates to `hermes_cli.copilot_auth.copilot_request_headers`
/// when importable. That module is not ported here, so this returns the
/// ImportError fallback header set.
pub fn copilot_default_headers() -> HashMap<String, String> {
    let mut h = HashMap::new();
    h.insert("Editor-Version".into(), COPILOT_EDITOR_VERSION.into());
    h.insert("User-Agent".into(), "HermesAgent/1.0".into());
    h.insert("Openai-Intent".into(), "conversation-edits".into());
    h.insert("x-initiator".into(), "agent".into());
    h
}

fn copilot_catalog_item_is_text_model(item: &Value) -> bool {
    let model_id = item
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if model_id.is_empty() {
        return false;
    }
    if item.get("model_picker_enabled") == Some(&Value::Bool(false)) {
        return false;
    }
    if let Some(caps) = item.get("capabilities") {
        if caps.is_object() {
            let model_type = caps
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            if !model_type.is_empty() && model_type != "chat" {
                return false;
            }
        }
    }
    if let Some(Value::Array(endpoints)) = item.get("supported_endpoints") {
        let normalized: HashSet<String> = endpoints
            .iter()
            .filter_map(|e| e.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let allowed: HashSet<&str> =
            ["/chat/completions", "/responses", "/v1/messages"].into_iter().collect();
        if !normalized.is_empty()
            && !normalized.iter().any(|e| allowed.contains(e.as_str()))
        {
            return false;
        }
    }
    true
}

/// Fetch the live GitHub Copilot model catalog for this account.
pub fn fetch_github_model_catalog(api_key: Option<&str>, timeout: f64) -> Option<Vec<Value>> {
    let mut attempts: Vec<HashMap<String, String>> = Vec::new();
    if let Some(key) = api_key {
        if !key.is_empty() {
            let mut h = copilot_default_headers();
            h.insert("Authorization".into(), format!("Bearer {}", key));
            attempts.push(h);
        } else {
            // api_key passed but empty: Python only appends when truthy.
        }
    }
    attempts.push(copilot_default_headers());

    let client = build_client(timeout);
    for headers in &attempts {
        let mut req = client.get(copilot_models_url());
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let data: Value = match req.send().and_then(|r| r.json::<Value>()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let items = payload_items(&data);
        let mut models: Vec<Value> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();
        for item in items {
            if !copilot_catalog_item_is_text_model(&item) {
                continue;
            }
            let model_id = item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if model_id.is_empty() || seen_ids.contains(&model_id) {
                continue;
            }
            seen_ids.insert(model_id);
            models.push(item);
        }
        if !models.is_empty() {
            return Some(models);
        }
    }
    None
}

// ── Copilot context-window cache ──
static COPILOT_CONTEXT_CACHE: Mutex<Option<(HashMap<String, i64>, Instant)>> = Mutex::new(None);
const COPILOT_CONTEXT_CACHE_TTL: u64 = 3600;

/// Look up `max_prompt_tokens` for a Copilot model from the live `/models` API.
pub fn get_copilot_model_context(model_id: &str, api_key: Option<&str>) -> Option<i64> {
    {
        let guard = COPILOT_CONTEXT_CACHE.lock().unwrap();
        if let Some((cache, cached_at)) = guard.as_ref() {
            if !cache.is_empty()
                && cached_at.elapsed().as_secs() < COPILOT_CONTEXT_CACHE_TTL
            {
                return cache.get(model_id).copied();
            }
        }
    }

    let catalog = fetch_github_model_catalog(api_key, 5.0)?;
    let mut cache: HashMap<String, i64> = HashMap::new();
    for item in &catalog {
        let mid = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if mid.is_empty() {
            continue;
        }
        let max_prompt = item
            .get("capabilities")
            .and_then(|c| c.get("limits"))
            .and_then(|l| l.get("max_prompt_tokens"))
            .and_then(|m| m.as_i64());
        if let Some(mp) = max_prompt {
            if mp > 0 {
                cache.insert(mid, mp);
            }
        }
    }

    let result = cache.get(model_id).copied();
    *COPILOT_CONTEXT_CACHE.lock().unwrap() = Some((cache, Instant::now()));
    result
}

fn is_github_models_base_url(base_url: Option<&str>) -> bool {
    let normalized = base_url
        .unwrap_or("")
        .trim()
        .trim_end_matches('/')
        .to_lowercase();
    normalized.starts_with(&COPILOT_BASE_URL.to_lowercase())
        || normalized.starts_with("https://models.github.ai/inference")
}

// ---------------------------------------------------------------------------
// LM Studio
// ---------------------------------------------------------------------------

/// Error raised when LM Studio rejects auth (HTTP 401/403).
#[derive(Debug, Clone)]
pub struct LmStudioAuthError {
    pub message: String,
    pub code: u16,
}

impl std::fmt::Display for LmStudioAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for LmStudioAuthError {}

/// Strip `/v1` suffix from an LM Studio base URL to get the native API root.
pub fn lmstudio_server_root(base_url: Option<&str>) -> Option<String> {
    let mut root = base_url.unwrap_or("").trim().trim_end_matches('/').to_string();
    if root.ends_with("/v1") {
        root.truncate(root.len() - 3);
        root = root.trim_end_matches('/').to_string();
    }
    if root.is_empty() {
        None
    } else {
        Some(root)
    }
}

fn lmstudio_request_headers(api_key: Option<&str>) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("User-Agent".into(), hermes_user_agent());
    let token = api_key.unwrap_or("").trim().to_string();
    if !token.is_empty() {
        headers.insert("Authorization".into(), format!("Bearer {}", token));
    }
    headers
}

/// Fetch the raw model list from LM Studio's `/api/v1/models`.
///
/// Returns the `models` list of objects on success. Returns
/// `Err(LmStudioAuthError)` on HTTP 401/403, `Ok(None)` on other failures.
pub fn lmstudio_fetch_raw_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
) -> Result<Option<Vec<Value>>, LmStudioAuthError> {
    let server_root = match lmstudio_server_root(base_url) {
        Some(r) => r,
        None => return Ok(None),
    };

    let headers = lmstudio_request_headers(api_key);
    let client = build_client(timeout);
    let mut req = client.get(format!("{}/api/v1/models", server_root));
    for (k, v) in &headers {
        req = req.header(k.as_str(), v.as_str());
    }

    let resp = match req.send() {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let status = resp.status().as_u16();
    if status == 401 || status == 403 {
        return Err(LmStudioAuthError {
            message: format!("LM Studio rejected the request with HTTP {}.", status),
            code: status,
        });
    }
    if !resp.status().is_success() {
        return Ok(None);
    }
    let payload: Value = match resp.json::<Value>() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let raw_models = match payload.get("models") {
        Some(Value::Array(a)) => a.clone(),
        _ => return Ok(None),
    };
    Ok(Some(raw_models))
}

/// Probe LM Studio's model listing for chat-capable model keys.
///
/// Returns `Ok(None)` on network errors / malformed responses / empty base URL,
/// `Ok(Some(vec))` (possibly empty) when reachable. Errors on auth rejection.
pub fn probe_lmstudio_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
) -> Result<Option<Vec<String>>, LmStudioAuthError> {
    let raw_models = match lmstudio_fetch_raw_models(api_key, base_url, timeout)? {
        Some(v) => v,
        None => return Ok(None),
    };

    let mut keys: Vec<String> = Vec::new();
    for raw in &raw_models {
        if !raw.is_object() {
            continue;
        }
        let typ = raw
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if typ == "embedding" {
            continue;
        }
        let key = raw
            .get("key")
            .and_then(|v| v.as_str())
            .or_else(|| raw.get("id").and_then(|v| v.as_str()))
            .unwrap_or("")
            .trim()
            .to_string();
        if !key.is_empty() && !keys.contains(&key) {
            keys.push(key);
        }
    }
    Ok(Some(keys))
}

/// Fetch LM Studio chat-capable model keys. Returns `[]` on errors.
pub fn fetch_lmstudio_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
) -> Vec<String> {
    match probe_lmstudio_models(api_key, base_url, timeout) {
        Ok(Some(v)) => v,
        _ => Vec::new(),
    }
}

/// Ensure LM Studio has `model` loaded with at least `target_context_length`.
pub fn ensure_lmstudio_model_loaded(
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
    mut target_context_length: i64,
    timeout: f64,
) -> Option<i64> {
    let server_root = lmstudio_server_root(base_url)?;
    let headers = lmstudio_request_headers(api_key);

    let raw_models = match lmstudio_fetch_raw_models(api_key, base_url, 10.0) {
        Ok(Some(v)) => v,
        _ => return None,
    };

    let target_entry = raw_models.iter().find(|raw| {
        raw.is_object()
            && (raw.get("key").and_then(|v| v.as_str()) == Some(model)
                || raw.get("id").and_then(|v| v.as_str()) == Some(model))
    })?;

    if let Some(max_ctx) = target_entry.get("max_context_length").and_then(|v| v.as_i64()) {
        if max_ctx > 0 {
            target_context_length = target_context_length.min(max_ctx);
        }
    }

    if let Some(Value::Array(instances)) = target_entry.get("loaded_instances") {
        for inst in instances {
            let loaded_ctx = inst
                .get("config")
                .and_then(|c| c.get("context_length"))
                .and_then(|v| v.as_i64());
            if let Some(lc) = loaded_ctx {
                if lc >= target_context_length {
                    return Some(lc);
                }
            }
        }
    }

    let body = serde_json::json!({
        "model": model,
        "context_length": target_context_length,
    });
    let client = build_client(timeout);
    let mut req = client
        .post(format!("{}/api/v1/models/load", server_root))
        .header("Content-Type", "application/json")
        .body(serde_json::to_vec(&body).ok()?);
    for (k, v) in &headers {
        req = req.header(k.as_str(), v.as_str());
    }
    match req.send() {
        Ok(_) => Some(target_context_length),
        Err(_) => None,
    }
}

/// Return the reasoning `allowed_options` LM Studio publishes for `model`.
pub fn lmstudio_model_reasoning_options(
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
    timeout: f64,
) -> Vec<String> {
    let raw_models = match lmstudio_fetch_raw_models(api_key, base_url, timeout) {
        Ok(Some(v)) => v,
        _ => return Vec::new(),
    };
    if raw_models.is_empty() {
        return Vec::new();
    }

    for raw in &raw_models {
        if !raw.is_object() {
            continue;
        }
        if raw.get("key").and_then(|v| v.as_str()) != Some(model)
            && raw.get("id").and_then(|v| v.as_str()) != Some(model)
        {
            continue;
        }
        let opts = raw
            .get("capabilities")
            .and_then(|c| c.get("reasoning"))
            .and_then(|r| r.get("allowed_options"));
        if let Some(Value::Array(arr)) = opts {
            return arr
                .iter()
                .filter_map(|o| o.as_str().map(|s| s.trim().to_lowercase()))
                .collect();
        }
        return Vec::new();
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// GitHub model id helpers
// ---------------------------------------------------------------------------

fn fetch_github_models(api_key: Option<&str>, timeout: f64) -> Option<Vec<String>> {
    let catalog = fetch_github_model_catalog(api_key, timeout)?;
    Some(
        catalog
            .iter()
            .filter_map(|item| {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    None
                } else {
                    Some(id.to_string())
                }
            })
            .collect(),
    )
}

/// Copilot model aliases (`raw_id` → canonical copilot id).
pub fn copilot_model_aliases() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    for &(k, v) in &[
        ("openai/gpt-5", "gpt-5-mini"),
        ("openai/gpt-5-chat", "gpt-5-mini"),
        ("openai/gpt-5-mini", "gpt-5-mini"),
        ("openai/gpt-5-nano", "gpt-5-mini"),
        ("openai/gpt-4.1", "gpt-4.1"),
        ("openai/gpt-4.1-mini", "gpt-4.1"),
        ("openai/gpt-4.1-nano", "gpt-4.1"),
        ("openai/gpt-4o", "gpt-4o"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("openai/o1", "gpt-5.2"),
        ("openai/o1-mini", "gpt-5-mini"),
        ("openai/o1-preview", "gpt-5.2"),
        ("openai/o3", "gpt-5.3-codex"),
        ("openai/o3-mini", "gpt-5-mini"),
        ("openai/o4-mini", "gpt-5-mini"),
        ("anthropic/claude-opus-4.6", "claude-opus-4.6"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("anthropic/claude-sonnet-4", "claude-sonnet-4"),
        ("anthropic/claude-sonnet-4.5", "claude-sonnet-4.5"),
        ("anthropic/claude-haiku-4.5", "claude-haiku-4.5"),
        ("claude-opus-4-6", "claude-opus-4.6"),
        ("claude-sonnet-4-6", "claude-sonnet-4.6"),
        ("claude-sonnet-4-0", "claude-sonnet-4"),
        ("claude-sonnet-4-5", "claude-sonnet-4.5"),
        ("claude-haiku-4-5", "claude-haiku-4.5"),
        ("anthropic/claude-opus-4-6", "claude-opus-4.6"),
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4.6"),
        ("anthropic/claude-sonnet-4-0", "claude-sonnet-4"),
        ("anthropic/claude-sonnet-4-5", "claude-sonnet-4.5"),
        ("anthropic/claude-haiku-4-5", "claude-haiku-4.5"),
    ] {
        m.insert(k, v);
    }
    m
}

fn copilot_catalog_ids(catalog: Option<&[Value]>, api_key: Option<&str>) -> HashSet<String> {
    let owned;
    let cat: Option<&[Value]> = match catalog {
        Some(c) => Some(c),
        None => {
            if let Some(key) = api_key {
                if !key.is_empty() {
                    owned = fetch_github_model_catalog(Some(key), 5.0);
                    owned.as_deref()
                } else {
                    None
                }
            } else {
                None
            }
        }
    };
    let cat = match cat {
        Some(c) => c,
        None => return HashSet::new(),
    };
    cat.iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").trim();
            if id.is_empty() {
                None
            } else {
                Some(id.to_string())
            }
        })
        .collect()
}

/// Normalize a Copilot model id against aliases and the catalog.
pub fn normalize_copilot_model_id(
    model_id: Option<&str>,
    catalog: Option<&[Value]>,
    api_key: Option<&str>,
) -> String {
    let raw = model_id.unwrap_or("").trim().to_string();
    if raw.is_empty() {
        return String::new();
    }

    let catalog_ids = copilot_catalog_ids(catalog, api_key);
    let aliases = copilot_model_aliases();
    if let Some(alias) = aliases.get(raw.as_str()) {
        return alias.to_string();
    }

    let mut candidates: Vec<String> = vec![raw.clone()];
    if let Some((_, rest)) = raw.split_once('/') {
        candidates.push(rest.trim().to_string());
    }
    if raw.ends_with("-mini") {
        candidates.push(raw[..raw.len() - 5].to_string());
    }
    if raw.ends_with("-nano") {
        candidates.push(raw[..raw.len() - 5].to_string());
    }
    if raw.ends_with("-chat") {
        candidates.push(raw[..raw.len() - 5].to_string());
    }

    let mut seen: HashSet<String> = HashSet::new();
    for candidate in candidates {
        if candidate.is_empty() || seen.contains(&candidate) {
            continue;
        }
        seen.insert(candidate.clone());
        if let Some(alias) = aliases.get(candidate.as_str()) {
            return alias.to_string();
        }
        if catalog_ids.contains(&candidate) {
            return candidate;
        }
    }

    if let Some((_, rest)) = raw.split_once('/') {
        return rest.trim().to_string();
    }
    raw
}

fn github_reasoning_efforts_for_model_id(model_id: &str) -> Vec<String> {
    let raw = model_id.trim().to_lowercase();
    let o_series_prefixes = ["openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4"];
    if o_series_prefixes.iter().any(|p| raw.starts_with(p)) {
        return COPILOT_REASONING_EFFORTS_O_SERIES
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    let normalized = normalize_copilot_model_id(Some(model_id), None, None).to_lowercase();
    if normalized.starts_with("gpt-5") {
        return COPILOT_REASONING_EFFORTS_GPT5
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    Vec::new()
}

/// Decide whether a Copilot model should use the Responses API.
pub fn should_use_copilot_responses_api(model_id: &str) -> bool {
    // ^gpt-(\d+)
    let after = match model_id.strip_prefix("gpt-") {
        Some(s) => s,
        None => return false,
    };
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return false;
    }
    let major: i64 = digits.parse().unwrap_or(0);
    major >= 5 && !model_id.starts_with("gpt-5-mini")
}

/// Determine the API mode for a Copilot model.
pub fn copilot_model_api_mode(
    model_id: Option<&str>,
    catalog: Option<&[Value]>,
    api_key: Option<&str>,
) -> String {
    let fetched: Option<Vec<Value>>;
    let cat: Option<&[Value]> = match catalog {
        Some(c) => Some(c),
        None => {
            if let Some(key) = api_key {
                if !key.is_empty() {
                    fetched = fetch_github_model_catalog(Some(key), 5.0);
                    fetched.as_deref()
                } else {
                    None
                }
            } else {
                None
            }
        }
    };

    let normalized = normalize_copilot_model_id(model_id, cat, api_key);
    if normalized.is_empty() {
        return "chat_completions".to_string();
    }

    if should_use_copilot_responses_api(&normalized) {
        return "codex_responses".to_string();
    }

    if let Some(cat) = cat {
        if let Some(entry) = cat
            .iter()
            .find(|item| item.get("id").and_then(|v| v.as_str()) == Some(normalized.as_str()))
        {
            let supported_endpoints: HashSet<String> = entry
                .get("supported_endpoints")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| e.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if supported_endpoints.contains("/v1/messages")
                && !supported_endpoints.contains("/chat/completions")
            {
                return "anthropic_messages".to_string();
            }
        }
    }

    "chat_completions".to_string()
}

const AZURE_FOUNDRY_RESPONSES_PREFIXES: &[&str] = &["codex", "gpt-5", "o1", "o3", "o4"];

/// Infer Azure Foundry api_mode from a deployment/model name.
pub fn azure_foundry_model_api_mode(model_name: Option<&str>) -> Option<String> {
    let mut raw = model_name.unwrap_or("").trim().to_lowercase();
    if raw.is_empty() {
        return None;
    }
    if let Some(idx) = raw.rfind('/') {
        raw = raw[idx + 1..].to_string();
    }
    for prefix in AZURE_FOUNDRY_RESPONSES_PREFIXES {
        if raw.starts_with(prefix) {
            return Some("codex_responses".to_string());
        }
    }
    None
}

/// Normalize OpenCode config IDs to the bare model slug used in API requests.
pub fn normalize_opencode_model_id(provider_id: Option<&str>, model_id: Option<&str>) -> String {
    let provider = normalize_provider(provider_id);
    let current = model_id.unwrap_or("").trim().to_string();
    if current.is_empty() || (provider != "opencode-zen" && provider != "opencode-go") {
        return current;
    }
    let prefix = format!("{}/", provider);
    if current.to_lowercase().starts_with(&prefix) {
        return current[prefix.len()..].to_string();
    }
    current
}

/// Determine the API mode for an OpenCode Zen / Go model.
pub fn opencode_model_api_mode(provider_id: Option<&str>, model_id: Option<&str>) -> String {
    let provider = normalize_provider(provider_id);
    let normalized = normalize_opencode_model_id(provider_id, model_id).to_lowercase();
    if normalized.is_empty() {
        return "chat_completions".to_string();
    }

    if provider == "opencode-go" {
        if normalized.starts_with("minimax-") {
            return "anthropic_messages".to_string();
        }
        return "chat_completions".to_string();
    }

    if provider == "opencode-zen" {
        if normalized.starts_with("claude-") {
            return "anthropic_messages".to_string();
        }
        if normalized.starts_with("gpt-") {
            return "codex_responses".to_string();
        }
        return "chat_completions".to_string();
    }

    "chat_completions".to_string()
}

/// Return supported reasoning-effort levels for a Copilot-visible model.
pub fn github_model_reasoning_efforts(
    model_id: Option<&str>,
    catalog: Option<&[Value]>,
    api_key: Option<&str>,
) -> Vec<String> {
    let normalized = normalize_copilot_model_id(model_id, catalog, api_key);
    if normalized.is_empty() {
        return Vec::new();
    }

    let fetched: Option<Vec<Value>>;
    let mut catalog_entry: Option<&Value> = None;
    if let Some(cat) = catalog {
        catalog_entry = cat
            .iter()
            .find(|item| item.get("id").and_then(|v| v.as_str()) == Some(normalized.as_str()));
    } else if let Some(key) = api_key {
        if !key.is_empty() {
            fetched = fetch_github_model_catalog(Some(key), 5.0);
            if let Some(cat) = fetched.as_deref() {
                catalog_entry = cat
                    .iter()
                    .find(|item| item.get("id").and_then(|v| v.as_str()) == Some(normalized.as_str()));
            }
        }
    }

    if let Some(entry) = catalog_entry {
        if let Some(caps) = entry.get("capabilities") {
            if caps.is_object() {
                if let Some(Value::Array(efforts)) =
                    caps.get("supports").and_then(|s| s.get("reasoning_effort"))
                {
                    let normalized_efforts: Vec<String> = efforts
                        .iter()
                        .filter_map(|e| e.as_str())
                        .map(|s| s.trim().to_lowercase())
                        .filter(|s| !s.is_empty())
                        .collect();
                    // dict.fromkeys: dedup preserving order.
                    let mut seen = HashSet::new();
                    let mut out = Vec::new();
                    for e in normalized_efforts {
                        if seen.insert(e.clone()) {
                            out.push(e);
                        }
                    }
                    return out;
                }
                return Vec::new();
            }
            // legacy: capabilities is a list of strings
            let legacy: HashSet<String> = caps
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|c| c.as_str())
                        .map(|s| s.trim().to_lowercase())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if !legacy.contains("reasoning") {
                return Vec::new();
            }
        }
    }

    let fallback_id = model_id.filter(|s| !s.is_empty()).unwrap_or(&normalized);
    github_reasoning_efforts_for_model_id(fallback_id)
}

// ---------------------------------------------------------------------------
// Generic /models probe
// ---------------------------------------------------------------------------

/// Result of [`probe_api_models`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeResult {
    pub models: Option<Vec<String>>,
    pub probed_url: Option<String>,
    pub resolved_base_url: String,
    pub suggested_base_url: Option<String>,
    pub used_fallback: bool,
}

/// Probe a `/models` endpoint with light URL heuristics.
pub fn probe_api_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
    api_mode: Option<&str>,
) -> ProbeResult {
    let normalized = base_url.unwrap_or("").trim().trim_end_matches('/').to_string();
    if normalized.is_empty() {
        return ProbeResult {
            models: None,
            probed_url: None,
            resolved_base_url: String::new(),
            suggested_base_url: None,
            used_fallback: false,
        };
    }

    if is_github_models_base_url(Some(&normalized)) {
        let models = fetch_github_models(api_key, timeout);
        return ProbeResult {
            models,
            probed_url: Some(copilot_models_url()),
            resolved_base_url: COPILOT_BASE_URL.to_string(),
            suggested_base_url: None,
            used_fallback: false,
        };
    }

    let alternate_base = if normalized.ends_with("/v1") {
        normalized[..normalized.len() - 3]
            .trim_end_matches('/')
            .to_string()
    } else {
        format!("{}/v1", normalized)
    };

    let mut candidates: Vec<(String, bool)> = vec![(normalized.clone(), false)];
    if !alternate_base.is_empty() && alternate_base != normalized {
        candidates.push((alternate_base.clone(), true));
    }

    let mut tried: Vec<String> = Vec::new();
    let mut headers: HashMap<String, String> = HashMap::new();
    headers.insert("User-Agent".into(), hermes_user_agent());
    if let Some(key) = api_key {
        if !key.is_empty() {
            if api_mode == Some("anthropic_messages") {
                headers.insert("x-api-key".into(), key.to_string());
                headers.insert("anthropic-version".into(), "2023-06-01".into());
            } else {
                headers.insert("Authorization".into(), format!("Bearer {}", key));
            }
        }
    }
    if normalized.starts_with(COPILOT_BASE_URL) {
        for (k, v) in copilot_default_headers() {
            headers.insert(k, v);
        }
    }

    let client = build_client(timeout);
    for (candidate_base, is_fallback) in &candidates {
        let url = format!("{}/models", candidate_base.trim_end_matches('/'));
        tried.push(url.clone());
        let mut req = client.get(&url);
        for (k, v) in &headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let data: Value = match req.send().and_then(|r| r.json::<Value>()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let models: Vec<String> = data
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|m| {
                        m.get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let suggested = if alternate_base != *candidate_base {
            alternate_base.clone()
        } else {
            normalized.clone()
        };
        return ProbeResult {
            models: Some(models),
            probed_url: Some(url),
            resolved_base_url: candidate_base.trim_end_matches('/').to_string(),
            suggested_base_url: Some(suggested),
            used_fallback: *is_fallback,
        };
    }

    let probed = tried
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{}/models", normalized.trim_end_matches('/')));
    let suggested = if alternate_base != normalized {
        Some(alternate_base)
    } else {
        None
    };
    ProbeResult {
        models: None,
        probed_url: Some(probed),
        resolved_base_url: normalized,
        suggested_base_url: suggested,
        used_fallback: false,
    }
}

/// Fetch available language models with tool-use from AI Gateway.
///
/// Reads `AI_GATEWAY_API_KEY` / `AI_GATEWAY_BASE_URL` from the environment.
fn fetch_ai_gateway_models_lang(timeout: f64) -> Option<Vec<String>> {
    let api_key = std::env::var("AI_GATEWAY_API_KEY")
        .unwrap_or_default()
        .trim()
        .to_string();
    if api_key.is_empty() {
        return None;
    }
    let base_env = std::env::var("AI_GATEWAY_BASE_URL")
        .unwrap_or_default()
        .trim()
        .to_string();
    let base_url = if base_env.is_empty() {
        AI_GATEWAY_BASE_URL.to_string()
    } else {
        base_env
    };

    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = build_client(timeout);
    let data: Value = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("User-Agent", hermes_user_agent())
        .send()
        .and_then(|r| r.json::<Value>())
        .ok()?;

    Some(
        data.get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|m| {
                        let has_id = m
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| !s.is_empty())
                            .unwrap_or(false);
                        let is_lang =
                            m.get("type").and_then(|v| v.as_str()) == Some("language");
                        let has_tool = m
                            .get("tags")
                            .and_then(|t| t.as_array())
                            .map(|tags| {
                                tags.iter().any(|t| t.as_str() == Some("tool-use"))
                            })
                            .unwrap_or(false);
                        has_id && is_lang && has_tool
                    })
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

/// Fetch the list of available model IDs from the provider's `/models` endpoint.
pub fn fetch_api_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    timeout: f64,
    api_mode: Option<&str>,
) -> Option<Vec<String>> {
    probe_api_models(api_key, base_url, timeout, api_mode).models
}

// ---------------------------------------------------------------------------
// Ollama Cloud
// ---------------------------------------------------------------------------

const OLLAMA_CLOUD_CACHE_TTL: u64 = 3600;

fn strip_ollama_cloud_suffix(model_id: &str) -> String {
    for suffix in &[":cloud", "-cloud"] {
        if let Some(stripped) = model_id.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    model_id.to_string()
}

fn ollama_cloud_cache_path() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_home().join("ollama_cloud_models_cache.json")
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn load_ollama_cloud_cache(ignore_ttl: bool) -> Option<Vec<String>> {
    let cache_path = ollama_cloud_cache_path();
    if !cache_path.exists() {
        return None;
    }
    let raw = std::fs::read_to_string(&cache_path).ok()?;
    let data: Value = serde_json::from_str(&raw).ok()?;
    if !data.is_object() {
        return None;
    }
    let models = data.get("models")?.as_array()?;
    if models.is_empty() {
        return None;
    }
    if !ignore_ttl {
        let cached_at = data.get("cached_at").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if (now_unix() - cached_at) > OLLAMA_CLOUD_CACHE_TTL as f64 {
            return None;
        }
    }
    Some(
        models
            .iter()
            .filter_map(|m| m.as_str().map(|s| s.to_string()))
            .collect(),
    )
}

fn save_ollama_cloud_cache(models: &[String]) {
    let cache_path = ollama_cloud_cache_path();
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let payload = serde_json::json!({
        "models": models,
        "cached_at": now_unix(),
    });
    if let Ok(bytes) = serde_json::to_vec(&payload) {
        let _ = std::fs::write(&cache_path, bytes);
    }
}

/// Fetch Ollama Cloud models by merging live API + models.dev, with disk cache.
///
/// The Python source pulls models.dev entries via `agent.models_dev`; that
/// registry is not ported, so callers can pass `mdev_models` directly (empty
/// to reproduce the import-failure path). Live `/v1/models` is probed using
/// the provided / env-derived credentials.
pub fn fetch_ollama_cloud_models(
    api_key: Option<&str>,
    base_url: Option<&str>,
    force_refresh: bool,
    mdev_models: &[String],
) -> Vec<String> {
    if !force_refresh {
        if let Some(cached) = load_ollama_cloud_cache(false) {
            return cached;
        }
    }

    let api_key_owned = match api_key {
        Some(k) if !k.is_empty() => k.to_string(),
        _ => std::env::var("OLLAMA_API_KEY").unwrap_or_default(),
    };
    let base_url_owned = match base_url {
        Some(b) if !b.is_empty() => b.to_string(),
        _ => {
            let env = std::env::var("OLLAMA_BASE_URL").unwrap_or_default();
            if env.is_empty() {
                "https://ollama.com/v1".to_string()
            } else {
                env
            }
        }
    };

    let mut live_models: Vec<String> = Vec::new();
    if !api_key_owned.is_empty() {
        if let Some(result) =
            fetch_api_models(Some(&api_key_owned), Some(&base_url_owned), 8.0, None)
        {
            if !result.is_empty() {
                live_models = result;
            }
        }
    }

    if !live_models.is_empty() || !mdev_models.is_empty() {
        let mut seen: HashSet<String> = HashSet::new();
        let mut merged: Vec<String> = Vec::new();
        for m in &live_models {
            if !m.is_empty() && seen.insert(m.clone()) {
                merged.push(m.clone());
            }
        }
        for m in mdev_models {
            let normalized = strip_ollama_cloud_suffix(m);
            if !normalized.is_empty() && seen.insert(normalized.clone()) {
                merged.push(normalized);
            }
        }
        if !merged.is_empty() {
            save_ollama_cloud_cache(&merged);
            return merged;
        }
    }

    if let Some(stale) = load_ollama_cloud_cache(true) {
        return stale;
    }

    Vec::new()
}

// ---------------------------------------------------------------------------
// get_close_matches (difflib parity)
// ---------------------------------------------------------------------------

/// Port of `difflib.get_close_matches`.
///
/// Returns up to `n` best matches with `ratio() >= cutoff`, in descending
/// score order (ties keep input order, matching CPython's `heapq.nlargest`
/// stability for equal scores — CPython is *not* stable here, but for our use
/// the result set sizes are tiny and the exact tie order is not relied upon).
pub fn get_close_matches(word: &str, possibilities: &[String], n: usize, cutoff: f64) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(f64, usize, &String)> = Vec::new();
    for (idx, x) in possibilities.iter().enumerate() {
        let ratio = seq_ratio(word, x);
        if ratio >= cutoff {
            scored.push((ratio, idx, x));
        }
    }
    // Sort by score descending; stable on insertion order for ties.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .take(n)
        .map(|(_, _, s)| s.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// validate_requested_model
// ---------------------------------------------------------------------------

/// Result of [`validate_requested_model`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    pub accepted: bool,
    pub persist: bool,
    pub recognized: bool,
    pub corrected_model: Option<String>,
    pub message: Option<String>,
}

impl ValidationResult {
    fn new(
        accepted: bool,
        persist: bool,
        recognized: bool,
        message: Option<String>,
    ) -> Self {
        ValidationResult {
            accepted,
            persist,
            recognized,
            corrected_model: None,
            message,
        }
    }
}

fn matches_codex_plausible(s: &str) -> bool {
    // ^(?:gpt|o[1-9]|codex)[A-Za-z0-9_.:-]*(?:codex)?  on lowercased input.
    let lower = s.to_lowercase();
    let bytes = lower.as_bytes();
    let starts_gpt = lower.starts_with("gpt");
    let starts_codex = lower.starts_with("codex");
    let starts_o = bytes.len() >= 2 && bytes[0] == b'o' && (b'1'..=b'9').contains(&bytes[1]);
    starts_gpt || starts_codex || starts_o
}

/// Validate a `/model` value for the active provider.
///
/// Performs format checks first, then probes the live API to confirm the model
/// exists. Mirrors the Python decision tree (LM Studio, custom, openai-codex,
/// minimax, anthropic, anthropic_messages, generic probe, bedrock, static
/// fallback).
///
/// Branches that need credentials Python resolved internally
/// (anthropic token, bedrock discovery) reproduce the *fallback* path of the
/// corresponding `try/except` when those resolvers are unavailable here.
pub fn validate_requested_model(
    model_name: &str,
    provider: Option<&str>,
    api_key: Option<&str>,
    base_url: Option<&str>,
    api_mode: Option<&str>,
) -> ValidationResult {
    let requested = model_name.trim().to_string();
    let mut normalized = normalize_provider(provider);
    if normalized == "openrouter" {
        if let Some(b) = base_url {
            if !b.is_empty() && !b.contains("openrouter.ai") {
                normalized = "custom".to_string();
            }
        }
    }
    let mut requested_for_lookup = requested.clone();
    if normalized == "copilot" {
        let n = normalize_copilot_model_id(Some(&requested), None, api_key);
        if !n.is_empty() {
            requested_for_lookup = n;
        }
    }

    if requested.is_empty() {
        return ValidationResult::new(false, false, false, Some("Model name cannot be empty.".into()));
    }
    if requested.chars().any(|c| c.is_whitespace()) {
        return ValidationResult::new(
            false,
            false,
            false,
            Some("Model names cannot contain spaces.".into()),
        );
    }

    // ── LM Studio ──
    if normalized == "lmstudio" {
        let models = match probe_lmstudio_models(api_key, base_url, 5.0) {
            Err(exc) => {
                return ValidationResult::new(
                    false,
                    false,
                    false,
                    Some(format!(
                        "{} Set `LM_API_KEY` (or update it) to match the server's bearer token.",
                        exc
                    )),
                );
            }
            Ok(m) => m,
        };
        match models {
            None => {
                return ValidationResult::new(
                    false,
                    false,
                    false,
                    Some(format!(
                        "Could not reach LM Studio's `/api/v1/models` to validate `{}`.",
                        requested
                    )),
                );
            }
            Some(m) if m.is_empty() => {
                return ValidationResult::new(
                    false,
                    false,
                    false,
                    Some(format!(
                        "LM Studio is reachable but no chat-capable models are loaded. \
                         Load `{}` in LM Studio (Developer tab → Load Model) and try again.",
                        requested
                    )),
                );
            }
            Some(m) => {
                if m.iter().any(|x| x == &requested_for_lookup) {
                    return ValidationResult::new(true, true, true, None);
                }
                return ValidationResult::new(
                    false,
                    false,
                    false,
                    Some(format!(
                        "Model `{}` was not found in LM Studio's model listing.",
                        requested
                    )),
                );
            }
        }
    }

    // ── custom / custom:* ──
    if normalized == "custom" || normalized.starts_with("custom:") {
        let probe = if api_mode == Some("anthropic_messages") {
            probe_api_models(api_key, base_url, 5.0, api_mode)
        } else {
            probe_api_models(api_key, base_url, 5.0, None)
        };
        if let Some(api_models) = probe.models.clone() {
            if api_models.iter().any(|m| m == &requested_for_lookup) {
                return ValidationResult::new(true, true, true, None);
            }
            let auto = get_close_matches(&requested_for_lookup, &api_models, 1, 0.9);
            if let Some(a) = auto.first() {
                let mut r = ValidationResult::new(
                    true,
                    true,
                    true,
                    Some(format!("Auto-corrected `{}` → `{}`", requested, a)),
                );
                r.corrected_model = Some(a.clone());
                return r;
            }
            let suggestions = get_close_matches(&requested, &api_models, 3, 0.5);
            let suggestion_text = suggestions_text(&suggestions);
            let mut message = format!(
                "Note: `{}` was not found in this custom endpoint's model listing ({}). \
                 It may still work if the server supports hidden or aliased models.{}",
                requested,
                probe.probed_url.clone().unwrap_or_default(),
                suggestion_text
            );
            if probe.used_fallback {
                message.push_str(&format!(
                    "\n  Endpoint verification succeeded after trying `{}`. \
                     Consider saving that as your base URL.",
                    probe.resolved_base_url
                ));
            }
            return ValidationResult::new(true, true, false, Some(message));
        }

        let mut message = format!(
            "Note: could not reach this custom endpoint's model listing at `{}`. \
             Hermes will still save `{}`, but the endpoint should expose `/models` for verification.",
            probe.probed_url.clone().unwrap_or_default(),
            requested
        );
        if api_mode == Some("anthropic_messages") {
            message.push_str(
                "\n  Many Anthropic-compatible proxies do not implement the Models API \
                 (GET /v1/models).  The model name has been accepted without verification.",
            );
        }
        if let Some(sug) = probe.suggested_base_url {
            message.push_str(&format!(
                "\n  If this server expects `/v1`, try base URL: `{}`",
                sug
            ));
        }
        return ValidationResult::new(
            api_mode == Some("anthropic_messages"),
            true,
            false,
            Some(message),
        );
    }

    // ── openai-codex ──
    if normalized == "openai-codex" {
        let codex_models = provider_model_ids(Some("openai-codex"), false);
        if !codex_models.is_empty() {
            if codex_models.iter().any(|m| m == &requested_for_lookup) {
                return ValidationResult::new(true, true, true, None);
            }
            let auto = get_close_matches(&requested_for_lookup, &codex_models, 1, 0.9);
            if let Some(a) = auto.first() {
                let mut r = ValidationResult::new(
                    true,
                    true,
                    true,
                    Some(format!("Auto-corrected `{}` → `{}`", requested, a)),
                );
                r.corrected_model = Some(a.clone());
                return r;
            }
            let suggestions = get_close_matches(&requested_for_lookup, &codex_models, 3, 0.5);
            let suggestion_text = suggestions_text(&suggestions);
            let looks_plausible = matches_codex_plausible(&requested_for_lookup);
            let tail = if looks_plausible {
                "It may still work if your ChatGPT/Codex account has access to a newer or hidden model ID."
            } else {
                "Use one of the listed OpenAI Codex models."
            };
            return ValidationResult::new(
                looks_plausible,
                looks_plausible,
                false,
                Some(format!(
                    "Note: `{}` was not found in the OpenAI Codex model listing. {}{}",
                    requested, tail, suggestion_text
                )),
            );
        }
    }

    // ── minimax / minimax-cn ──
    if normalized == "minimax" || normalized == "minimax-cn" {
        let catalog_models = provider_model_ids(Some(&normalized), false);
        if !catalog_models.is_empty() {
            let catalog_lower: HashMap<String, String> = catalog_models
                .iter()
                .map(|m| (m.to_lowercase(), m.clone()))
                .collect();
            if catalog_lower.contains_key(&requested_for_lookup.to_lowercase()) {
                return ValidationResult::new(true, true, true, None);
            }
            let catalog_lower_list: Vec<String> = catalog_lower.keys().cloned().collect();
            let auto =
                get_close_matches(&requested_for_lookup.to_lowercase(), &catalog_lower_list, 1, 0.9);
            if let Some(a) = auto.first() {
                let corrected = catalog_lower.get(a).cloned().unwrap_or_default();
                let mut r = ValidationResult::new(
                    true,
                    true,
                    true,
                    Some(format!("Auto-corrected `{}` → `{}`", requested, corrected)),
                );
                r.corrected_model = Some(corrected);
                return r;
            }
            let suggestions =
                get_close_matches(&requested_for_lookup.to_lowercase(), &catalog_lower_list, 3, 0.5);
            let suggestion_text = suggestions_text_mapped(&suggestions, &catalog_lower);
            return ValidationResult::new(
                true,
                true,
                false,
                Some(format!(
                    "Note: `{}` was not found in the MiniMax catalog.{}\
                     \n  MiniMax does not expose a /models endpoint, so Hermes cannot verify the model name.\
                     \n  The model may still work if it exists on the server.",
                    requested, suggestion_text
                )),
            );
        }
    }

    // ── native anthropic ──
    // The Python version resolves a token via agent.anthropic_adapter. Without
    // that resolver here, _fetch_anthropic_models returns None (no token), so
    // the branch falls through to the generic warning — matching the
    // unresolvable-token path.
    if normalized == "anthropic" {
        if let Some(anthropic_models) = fetch_anthropic_models(5.0, None) {
            if anthropic_models.iter().any(|m| m == &requested_for_lookup) {
                return ValidationResult::new(true, true, true, None);
            }
            let auto = get_close_matches(&requested_for_lookup, &anthropic_models, 1, 0.9);
            if let Some(a) = auto.first() {
                let mut r = ValidationResult::new(
                    true,
                    true,
                    true,
                    Some(format!("Auto-corrected `{}` → `{}`", requested, a)),
                );
                r.corrected_model = Some(a.clone());
                return r;
            }
            let suggestions = get_close_matches(&requested, &anthropic_models, 3, 0.5);
            let suggestion_text = suggestions_text(&suggestions);
            return ValidationResult::new(
                true,
                true,
                false,
                Some(format!(
                    "Note: `{}` was not found in Anthropic's /v1/models listing. \
                     It may still work if you have early-access or snapshot IDs.{}",
                    requested, suggestion_text
                )),
            );
        }
    }

    // ── anthropic_messages transport ──
    if api_mode == Some("anthropic_messages") {
        let api_models = fetch_api_models(api_key, base_url, 5.0, api_mode);
        if let Some(api_models) = api_models {
            if api_models.iter().any(|m| m == &requested_for_lookup) {
                return ValidationResult::new(true, true, true, None);
            }
            let auto = get_close_matches(&requested_for_lookup, &api_models, 1, 0.9);
            if let Some(a) = auto.first() {
                let mut r = ValidationResult::new(
                    true,
                    true,
                    true,
                    Some(format!("Auto-corrected `{}` → `{}`", requested, a)),
                );
                r.corrected_model = Some(a.clone());
                return r;
            }
        }
        return ValidationResult::new(
            true,
            true,
            false,
            Some(format!(
                "Note: could not verify `{}` against this endpoint's model listing.  \
                 Many Anthropic-compatible proxies do not implement GET /v1/models.  \
                 The model name has been accepted without verification.",
                requested
            )),
        );
    }

    // ── generic probe ──
    let mut api_models = fetch_api_models(api_key, base_url, 5.0, None);

    if let Some(ref mut models) = api_models {
        if normalized == "gemini" {
            for m in models.iter_mut() {
                if let Some(stripped) = m.strip_prefix("models/") {
                    *m = stripped.to_string();
                }
            }
        }
        if models.iter().any(|m| m == &requested_for_lookup) {
            return ValidationResult::new(true, true, true, None);
        }
        let auto = get_close_matches(&requested_for_lookup, models, 1, 0.9);
        if let Some(a) = auto.first() {
            let mut r = ValidationResult::new(
                true,
                true,
                true,
                Some(format!("Auto-corrected `{}` → `{}`", requested, a)),
            );
            r.corrected_model = Some(a.clone());
            return r;
        }
        let suggestions = get_close_matches(&requested, models, 3, 0.5);
        let suggestion_text = suggestions_text(&suggestions);
        return ValidationResult::new(
            false,
            false,
            false,
            Some(format!(
                "Model `{}` was not found in this provider's model listing.{}",
                requested, suggestion_text
            )),
        );
    }

    // api_models is None — couldn't reach API.

    // ── bedrock (SDK discovery not ported → fall through to generic) ──
    // The Python version uses agent.bedrock_adapter discovery here. That is not
    // available in this crate, so the try/except falls through, exactly as it
    // would if the import failed.

    // ── static-catalog fallback ──
    let provider_label_str = provider_labels()
        .get(&normalized)
        .cloned()
        .unwrap_or_else(|| normalized.clone());
    let catalog_models = provider_model_ids(Some(&normalized), false);

    if !catalog_models.is_empty() {
        let catalog_lower: HashMap<String, String> = catalog_models
            .iter()
            .map(|m| (m.to_lowercase(), m.clone()))
            .collect();
        if catalog_lower.contains_key(&requested_for_lookup.to_lowercase()) {
            return ValidationResult::new(true, true, true, None);
        }
        let catalog_lower_list: Vec<String> = catalog_lower.keys().cloned().collect();
        let auto =
            get_close_matches(&requested_for_lookup.to_lowercase(), &catalog_lower_list, 1, 0.9);
        if let Some(a) = auto.first() {
            let corrected = catalog_lower.get(a).cloned().unwrap_or_default();
            let mut r = ValidationResult::new(
                true,
                true,
                true,
                Some(format!("Auto-corrected `{}` → `{}`", requested, corrected)),
            );
            r.corrected_model = Some(corrected);
            return r;
        }
        let suggestions =
            get_close_matches(&requested_for_lookup.to_lowercase(), &catalog_lower_list, 3, 0.5);
        let suggestion_text = suggestions_text_mapped(&suggestions, &catalog_lower);
        return ValidationResult::new(
            true,
            true,
            false,
            Some(format!(
                "Note: `{}` was not found in the {} curated catalog \
                 and the /models endpoint was unreachable.{}\
                 \n  The model may still work if it exists on the provider.",
                requested, provider_label_str, suggestion_text
            )),
        );
    }

    ValidationResult::new(
        true,
        true,
        false,
        Some(format!(
            "Note: could not reach the {} API to validate `{}`. \
             If the service isn't down, this model may not be valid.",
            provider_label_str, requested
        )),
    )
}

fn suggestions_text(suggestions: &[String]) -> String {
    if suggestions.is_empty() {
        String::new()
    } else {
        let joined = suggestions
            .iter()
            .map(|s| format!("`{}`", s))
            .collect::<Vec<_>>()
            .join(", ");
        format!("\n  Similar models: {}", joined)
    }
}

fn suggestions_text_mapped(suggestions: &[String], map: &HashMap<String, String>) -> String {
    if suggestions.is_empty() {
        String::new()
    } else {
        let joined = suggestions
            .iter()
            .map(|s| format!("`{}`", map.get(s).cloned().unwrap_or_else(|| s.clone())))
            .collect::<Vec<_>>()
            .join(", ");
        format!("\n  Similar models: {}", joined)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_provider_aliases() {
        assert_eq!(normalize_provider(Some("glm")), "zai");
        assert_eq!(normalize_provider(Some("Z-AI")), "zai");
        assert_eq!(normalize_provider(Some("github")), "copilot");
        assert_eq!(normalize_provider(Some("auto")), "auto");
        assert_eq!(normalize_provider(None), "openrouter");
        assert_eq!(normalize_provider(Some("  Vercel  ")), "ai-gateway");
        // Unknown passes through lowercased.
        assert_eq!(normalize_provider(Some("WeirdProvider")), "weirdprovider");
    }

    #[test]
    fn provider_label_lookup() {
        assert_eq!(provider_label(Some("nous")), "Nous Portal");
        assert_eq!(provider_label(Some("glm")), "Z.AI / GLM");
        assert_eq!(provider_label(Some("auto")), "Auto");
        assert_eq!(provider_label(None), "OpenRouter");
        assert_eq!(provider_label(Some("custom")), "Custom endpoint");
    }

    #[test]
    fn default_model_for_provider() {
        assert_eq!(
            get_default_model_for_provider("nous"),
            "moonshotai/kimi-k2.6"
        );
        assert_eq!(get_default_model_for_provider("azure-foundry"), "");
        assert_eq!(get_default_model_for_provider("does-not-exist"), "");
    }

    #[test]
    fn ai_gateway_derived_catalog() {
        let pm = provider_models_map();
        let ai = pm.get("ai-gateway").unwrap();
        assert_eq!(ai[0], "moonshotai/kimi-k2.6");
        assert_eq!(ai.len(), VERCEL_AI_GATEWAY_MODELS.len());
    }

    #[test]
    fn parse_model_input_variants() {
        assert_eq!(
            parse_model_input("openrouter:anthropic/claude-sonnet-4.5", "nous"),
            ("openrouter".into(), "anthropic/claude-sonnet-4.5".into())
        );
        assert_eq!(
            parse_model_input("nous:hermes-3", "openrouter"),
            ("nous".into(), "hermes-3".into())
        );
        // Colon but left side not a provider -> treated as model.
        assert_eq!(
            parse_model_input("anthropic/claude-3.5-sonnet:beta", "openrouter"),
            ("openrouter".into(), "anthropic/claude-3.5-sonnet:beta".into())
        );
        // Bare name.
        assert_eq!(
            parse_model_input("gpt-5.4", "openai"),
            ("openai".into(), "gpt-5.4".into())
        );
        // custom triple syntax.
        assert_eq!(
            parse_model_input("custom:local:qwen", "openrouter"),
            ("custom:local".into(), "qwen".into())
        );
        // custom single.
        assert_eq!(
            parse_model_input("custom:qwen", "openrouter"),
            ("custom".into(), "qwen".into())
        );
        // alias provider gets normalized.
        assert_eq!(
            parse_model_input("glm:glm-5", "openrouter"),
            ("zai".into(), "glm-5".into())
        );
    }

    #[test]
    fn format_price_examples() {
        assert_eq!(format_price_per_mtok("0.000003"), "$3.00");
        assert_eq!(format_price_per_mtok("0.00003"), "$30.00");
        assert_eq!(format_price_per_mtok("0.00000015"), "$0.15");
        assert_eq!(format_price_per_mtok("0.0000001"), "$0.10");
        assert_eq!(format_price_per_mtok("0.00018"), "$180.00");
        assert_eq!(format_price_per_mtok("0"), "free");
        assert_eq!(format_price_per_mtok("not-a-number"), "?");
    }

    #[test]
    fn fast_mode_detection() {
        // Opus 4.6 supports anthropic fast mode.
        assert!(model_supports_fast_mode(Some("claude-opus-4-6")));
        assert!(model_supports_fast_mode(Some("anthropic/claude-opus-4.6")));
        // Opus 4.7 does not.
        assert!(!model_supports_fast_mode(Some("claude-opus-4-7")));
        // OpenAI flagship eligible.
        assert!(model_supports_fast_mode(Some("gpt-5.4")));
        assert!(model_supports_fast_mode(Some("o3")));
        // Codex excluded.
        assert!(!model_supports_fast_mode(Some("gpt-5.3-codex")));

        let ov = resolve_fast_mode_overrides(Some("claude-opus-4-6")).unwrap();
        assert_eq!(ov.get("speed").unwrap(), "fast");
        let ov2 = resolve_fast_mode_overrides(Some("gpt-5.4")).unwrap();
        assert_eq!(ov2.get("service_tier").unwrap(), "priority");
        assert!(resolve_fast_mode_overrides(Some("gpt-5.3-codex")).is_none());
    }

    #[test]
    fn copilot_responses_api_decision() {
        assert!(should_use_copilot_responses_api("gpt-5"));
        assert!(should_use_copilot_responses_api("gpt-5.3-codex"));
        assert!(!should_use_copilot_responses_api("gpt-5-mini"));
        assert!(!should_use_copilot_responses_api("gpt-4o"));
        assert!(!should_use_copilot_responses_api("claude-sonnet-4.6"));
    }

    #[test]
    fn copilot_model_id_normalization() {
        assert_eq!(
            normalize_copilot_model_id(Some("openai/o3"), None, None),
            "gpt-5.3-codex"
        );
        assert_eq!(
            normalize_copilot_model_id(Some("claude-opus-4-6"), None, None),
            "claude-opus-4.6"
        );
        // unknown vendor/model -> strips vendor prefix
        assert_eq!(
            normalize_copilot_model_id(Some("vendor/some-model"), None, None),
            "some-model"
        );
        assert_eq!(normalize_copilot_model_id(Some(""), None, None), "");
    }

    #[test]
    fn azure_foundry_mode() {
        assert_eq!(
            azure_foundry_model_api_mode(Some("gpt-5.3-codex")),
            Some("codex_responses".into())
        );
        assert_eq!(
            azure_foundry_model_api_mode(Some("openrouter/gpt-5-codex")),
            Some("codex_responses".into())
        );
        assert_eq!(azure_foundry_model_api_mode(Some("gpt-4o")), None);
        assert_eq!(azure_foundry_model_api_mode(Some("")), None);
    }

    #[test]
    fn opencode_modes() {
        assert_eq!(
            opencode_model_api_mode(Some("opencode-zen"), Some("claude-opus-4-6")),
            "anthropic_messages"
        );
        assert_eq!(
            opencode_model_api_mode(Some("opencode-zen"), Some("gpt-5.4")),
            "codex_responses"
        );
        assert_eq!(
            opencode_model_api_mode(Some("opencode-zen"), Some("glm-5")),
            "chat_completions"
        );
        assert_eq!(
            opencode_model_api_mode(Some("opencode-go"), Some("minimax-m2.7")),
            "anthropic_messages"
        );
        assert_eq!(
            opencode_model_api_mode(Some("opencode-go"), Some("glm-5.1")),
            "chat_completions"
        );
        // prefixed id is stripped
        assert_eq!(
            normalize_opencode_model_id(Some("opencode-zen"), Some("opencode-zen/claude-opus-4-6")),
            "claude-opus-4-6"
        );
    }

    #[test]
    fn is_model_free_logic() {
        let mut pricing: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut free = HashMap::new();
        free.insert("prompt".into(), "0".into());
        free.insert("completion".into(), "0".into());
        pricing.insert("free-model".into(), free);
        let mut paid = HashMap::new();
        paid.insert("prompt".into(), "0.001".into());
        paid.insert("completion".into(), "0.002".into());
        pricing.insert("paid-model".into(), paid);

        assert!(is_model_free("free-model", &pricing));
        assert!(!is_model_free("paid-model", &pricing));
        assert!(!is_model_free("unknown", &pricing));
    }

    #[test]
    fn nous_free_tier_parse() {
        let info = serde_json::json!({"subscription": {"monthly_charge": 0}});
        assert!(is_nous_free_tier(&info));
        let info2 = serde_json::json!({"subscription": {"monthly_charge": 20}});
        assert!(!is_nous_free_tier(&info2));
        let info3 = serde_json::json!({"subscription": {}});
        assert!(!is_nous_free_tier(&info3));
        let info4 = serde_json::json!({});
        assert!(!is_nous_free_tier(&info4));
        let info5 = serde_json::json!({"subscription": {"monthly_charge": "0"}});
        assert!(is_nous_free_tier(&info5));
    }

    #[test]
    fn close_matches_basic() {
        let pool: Vec<String> = vec![
            "gpt-5.4".into(),
            "gpt-5.4-mini".into(),
            "gpt-5.3-codex".into(),
        ];
        let m = get_close_matches("gpt-5.4", &pool, 1, 0.9);
        assert_eq!(m, vec!["gpt-5.4".to_string()]);
        let none = get_close_matches("totally-different", &pool, 3, 0.9);
        assert!(none.is_empty());
    }

    #[test]
    fn validate_empty_and_spaces() {
        let r = validate_requested_model("", Some("openrouter"), None, None, None);
        assert!(!r.accepted && !r.persist && !r.recognized);
        assert_eq!(r.message.as_deref(), Some("Model name cannot be empty."));

        let r2 = validate_requested_model("foo bar", Some("openrouter"), None, None, None);
        assert!(!r2.accepted);
        assert_eq!(r2.message.as_deref(), Some("Model names cannot contain spaces."));
    }

    #[test]
    fn validate_openai_codex_plausible() {
        // No api_key/base_url; codex catalog comes from the static curated list.
        let r = validate_requested_model(
            "gpt-5.99-codex-future",
            Some("openai-codex"),
            None,
            None,
            None,
        );
        // plausible codex-looking id -> accepted but not recognized.
        assert!(r.accepted);
        assert!(!r.recognized);
        assert!(r.message.unwrap().contains("not found in the OpenAI Codex"));
    }

    #[test]
    fn validate_minimax_recognized() {
        let r = validate_requested_model("MiniMax-M2.7", Some("minimax"), None, None, None);
        assert!(r.accepted && r.recognized);
        assert!(r.message.is_none());
        // case-insensitive
        let r2 = validate_requested_model("minimax-m2.7", Some("minimax"), None, None, None);
        assert!(r2.accepted && r2.recognized);
    }

    #[test]
    fn known_providers_includes_aliases_and_custom() {
        let names = known_provider_names();
        assert!(names.contains("openrouter"));
        assert!(names.contains("custom"));
        assert!(names.contains("glm")); // alias
        assert!(names.contains("nous"));
    }

    #[test]
    fn strip_vendor_prefix_works() {
        assert_eq!(
            strip_vendor_prefix("anthropic/Claude-Opus-4-6"),
            "claude-opus-4-6"
        );
        assert_eq!(strip_vendor_prefix("gpt-5.4"), "gpt-5.4");
    }

    #[test]
    fn lmstudio_root_strip() {
        assert_eq!(
            lmstudio_server_root(Some("http://localhost:1234/v1")),
            Some("http://localhost:1234".to_string())
        );
        assert_eq!(
            lmstudio_server_root(Some("http://localhost:1234/")),
            Some("http://localhost:1234".to_string())
        );
        assert_eq!(lmstudio_server_root(Some("")), None);
        assert_eq!(lmstudio_server_root(None), None);
    }

    #[test]
    fn pricing_table_renders() {
        let models = vec![
            ("model-a".to_string(), String::new()),
            ("model-b".to_string(), String::new()),
        ];
        let mut pricing: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut a = HashMap::new();
        a.insert("prompt".into(), "0.000003".into());
        a.insert("completion".into(), "0.000015".into());
        pricing.insert("model-a".into(), a);
        let lines = format_model_pricing_table(&models, &pricing, "model-a", "  ");
        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.contains("Model")));
        assert!(lines.iter().any(|l| l.contains("← current")));
        assert!(lines.iter().any(|l| l.contains("$3.00")));
    }

    #[test]
    fn copilot_default_headers_fallback() {
        let h = copilot_default_headers();
        assert_eq!(h.get("Editor-Version").unwrap(), COPILOT_EDITOR_VERSION);
        assert_eq!(h.get("x-initiator").unwrap(), "agent");
    }

    #[test]
    fn github_reasoning_efforts_for_id() {
        assert_eq!(
            github_reasoning_efforts_for_model_id("openai/o3"),
            vec!["low", "medium", "high"]
        );
        assert_eq!(
            github_reasoning_efforts_for_model_id("gpt-5.4"),
            vec!["minimal", "low", "medium", "high"]
        );
        assert!(github_reasoning_efforts_for_model_id("gpt-4o").is_empty());
    }

    #[test]
    fn xai_curated_nonempty() {
        // With no disk cache populated under the test HERMES_HOME, falls back.
        let ids = xai_curated_models();
        assert!(!ids.is_empty());
    }
}
