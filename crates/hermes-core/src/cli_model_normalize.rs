//! Per-provider model name normalization.
//!
//! Native Rust port of `hermes_cli/model_normalize.py`.
//!
//! Different LLM providers expect model identifiers in different formats:
//!
//! - **Aggregators** (OpenRouter, Nous, AI Gateway, Kilo Code) need
//!   `vendor/model` slugs like `anthropic/claude-sonnet-4.6`.
//! - **Anthropic** native API expects bare names with dots replaced by
//!   hyphens: `claude-sonnet-4-6`.
//! - **Copilot** expects bare names *with* dots preserved:
//!   `claude-sonnet-4.6`.
//! - **OpenCode Zen** preserves dots for GPT/GLM/Gemini/Kimi/MiniMax-style
//!   model IDs, but Claude still uses hyphenated native names like
//!   `claude-sonnet-4-6`.
//! - **OpenCode Go** preserves dots in model names: `minimax-m2.7`.
//! - **DeepSeek** accepts `deepseek-chat` (V3), `deepseek-reasoner`
//!   (R1-family), and the first-class V-series IDs (`deepseek-v4-pro`,
//!   `deepseek-v4-flash`, and any future `deepseek-v<N>-*`).
//! - **Custom** and remaining providers pass the name through as-is.
//!
//! This module centralises that translation so callers can simply write:
//!
//! ```ignore
//! let api_model = normalize_model_for_provider(user_input, provider);
//! ```
//!
//! ## Parity notes
//!
//! The Python module performs lazy imports of `hermes_cli.models.normalize_provider`
//! and `hermes_cli.models.normalize_copilot_model_id`, each wrapped in
//! `try/except: pass` fallbacks. Both helpers are ported inline here:
//!
//! - [`normalize_provider`] reproduces the `_PROVIDER_ALIASES` table.
//! - [`normalize_copilot_model_id`] reproduces the alias-table half of the
//!   Python function. The Python version can also consult a *live* Copilot
//!   model catalog (network fetch) when given an `api_key`/`catalog`; this
//!   port models the `catalog=None, api_key=None` call shape used by
//!   `model_normalize.py`, where `_copilot_catalog_ids` returns an empty set
//!   and no catalog membership checks ever match. The pure alias resolution
//!   and slash/suffix-stripping fallbacks are reproduced exactly.

use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Vendor prefix mapping
// ---------------------------------------------------------------------------
// Maps the first hyphen-delimited token of a bare model name to the vendor
// slug used by aggregator APIs (OpenRouter, Nous, etc.).
//
// Example: "claude-sonnet-4.6" -> first token "claude" -> vendor "anthropic"
//          -> aggregator slug: "anthropic/claude-sonnet-4.6"
//
// NOTE: the Python dict has a duplicate "trinity" key; the later definition
// (also "arcee-ai") wins, so a single entry is faithful. Order matters for the
// `startswith` fallback in `detect_vendor`, so this is an ordered Vec.
fn vendor_prefixes() -> &'static [(&'static str, &'static str)] {
    &[
        ("claude", "anthropic"),
        ("gpt", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("gemini", "google"),
        ("gemma", "google"),
        ("deepseek", "deepseek"),
        ("glm", "z-ai"),
        ("kimi", "moonshotai"),
        ("minimax", "minimax"),
        ("grok", "x-ai"),
        ("qwen", "qwen"),
        ("mimo", "xiaomi"),
        ("trinity", "arcee-ai"),
        ("nemotron", "nvidia"),
        ("llama", "meta-llama"),
        ("step", "stepfun"),
    ]
}

/// Providers whose APIs consume `vendor/model` slugs.
fn is_aggregator_provider(p: &str) -> bool {
    matches!(p, "openrouter" | "nous" | "ai-gateway" | "kilocode")
}

/// Providers that want bare names with dots replaced by hyphens.
fn is_dot_to_hyphen_provider(p: &str) -> bool {
    matches!(p, "anthropic")
}

/// Providers that want bare names with dots preserved (vendor-prefix stripped).
fn is_strip_vendor_only_provider(p: &str) -> bool {
    matches!(p, "copilot" | "copilot-acp" | "openai-codex")
}

/// Providers whose native naming is authoritative — pass through unchanged.
fn is_authoritative_native_provider(p: &str) -> bool {
    matches!(p, "gemini" | "huggingface")
}

/// Direct providers that accept bare native names but should repair a matching
/// `provider/` prefix when users copy the aggregator form into config.yaml.
fn is_matching_prefix_strip_provider(p: &str) -> bool {
    matches!(
        p,
        "zai" | "kimi-coding"
            | "kimi-coding-cn"
            | "minimax"
            | "minimax-oauth"
            | "minimax-cn"
            | "alibaba"
            | "qwen-oauth"
            | "xiaomi"
            | "arcee"
            | "ollama-cloud"
            | "custom"
    )
}

/// Providers whose APIs require lowercase model IDs (e.g. Xiaomi's
/// `api.xiaomimimo.com` rejects mixed-case `MiMo-V2.5-Pro`).
fn is_lowercase_model_provider(p: &str) -> bool {
    matches!(p, "xiaomi")
}

// ---------------------------------------------------------------------------
// DeepSeek special handling
// ---------------------------------------------------------------------------

const DEEPSEEK_REASONER_KEYWORDS: &[&str] = &["reasoner", "r1", "think", "reasoning", "cot"];

fn is_deepseek_canonical(model: &str) -> bool {
    matches!(
        model,
        "deepseek-chat" | "deepseek-reasoner" | "deepseek-v4-pro" | "deepseek-v4-flash"
    )
}

/// First-class V-series IDs (`deepseek-v4-pro`, `deepseek-v4-flash`, future
/// `deepseek-v5-*`, dated variants like `deepseek-v4-flash-20260423`).
fn deepseek_v_series_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^deepseek-v\d+([-.].+)?$").unwrap())
}

/// Map a model input to a DeepSeek-accepted identifier.
///
/// Rules:
/// - Already a known canonical -> pass through.
/// - Matches the V-series pattern `deepseek-v<digit>...` -> pass through.
/// - Contains a reasoner keyword (r1, think, reasoning, cot, reasoner)
///   -> `deepseek-reasoner`.
/// - Everything else -> `deepseek-chat`.
fn normalize_for_deepseek(model_name: &str) -> String {
    let bare = strip_vendor_prefix(model_name).to_lowercase();

    if is_deepseek_canonical(&bare) {
        return bare;
    }

    if deepseek_v_series_re().is_match(&bare) {
        return bare;
    }

    for keyword in DEEPSEEK_REASONER_KEYWORDS {
        if bare.contains(keyword) {
            return "deepseek-reasoner".to_string();
        }
    }

    "deepseek-chat".to_string()
}

// ---------------------------------------------------------------------------
// Provider alias table (ported from hermes_cli.models._PROVIDER_ALIASES)
// ---------------------------------------------------------------------------

fn provider_aliases() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let pairs: &[(&str, &str)] = &[
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
        ];
        pairs.iter().copied().collect()
    })
}

/// Normalize provider aliases to Hermes' canonical provider ids.
///
/// Mirrors `hermes_cli.models.normalize_provider`: a `None`/empty provider
/// defaults to `"openrouter"`, then the input is lowercased/trimmed and looked
/// up in the alias table (falling back to the input itself).
///
/// Note: `"auto"` passes through unchanged.
pub fn normalize_provider(provider: Option<&str>) -> String {
    let raw = provider.unwrap_or("openrouter").trim().to_lowercase();
    let raw = if raw.is_empty() {
        "openrouter".to_string()
    } else {
        raw
    };
    provider_aliases()
        .get(raw.as_str())
        .map(|s| s.to_string())
        .unwrap_or(raw)
}

/// Resolve a provider alias to a canonical id, preserving the empty string.
///
/// Mirrors `model_normalize._normalize_provider_alias`: blank input returns
/// blank (the Python helper short-circuits empty before delegating to
/// `normalize_provider`, so the `"openrouter"` default does *not* apply here).
fn normalize_provider_alias(provider_name: &str) -> String {
    let raw = provider_name.trim().to_lowercase();
    if raw.is_empty() {
        return raw;
    }
    provider_aliases()
        .get(raw.as_str())
        .map(|s| s.to_string())
        .unwrap_or(raw)
}

// ---------------------------------------------------------------------------
// Copilot model alias table + normalizer (ported from hermes_cli.models)
// ---------------------------------------------------------------------------

fn copilot_model_aliases() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let pairs: &[(&str, &str)] = &[
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
        ];
        pairs.iter().copied().collect()
    })
}

/// Normalize a Copilot model id using the static alias table and the
/// slash/suffix-stripping fallbacks.
///
/// Ported from `hermes_cli.models.normalize_copilot_model_id` called with
/// `catalog=None, api_key=None` (the call shape used by `model_normalize.py`).
/// In that mode `_copilot_catalog_ids` resolves to an empty set, so the
/// catalog-membership branch never matches and only alias resolution +
/// final slash-stripping apply.
///
/// Returns an empty string for blank input (matching Python).
pub fn normalize_copilot_model_id(model_id: &str) -> String {
    let raw = model_id.trim().to_string();
    if raw.is_empty() {
        return String::new();
    }

    let aliases = copilot_model_aliases();

    if let Some(alias) = aliases.get(raw.as_str()) {
        return alias.to_string();
    }

    let mut candidates: Vec<String> = vec![raw.clone()];
    if let Some((_, after)) = raw.split_once('/') {
        candidates.push(after.trim().to_string());
    }
    if let Some(stripped) = raw.strip_suffix("-mini") {
        candidates.push(stripped.to_string());
    }
    if let Some(stripped) = raw.strip_suffix("-nano") {
        candidates.push(stripped.to_string());
    }
    if let Some(stripped) = raw.strip_suffix("-chat") {
        candidates.push(stripped.to_string());
    }

    let mut seen: Vec<String> = Vec::new();
    for candidate in &candidates {
        if candidate.is_empty() || seen.iter().any(|s| s == candidate) {
            continue;
        }
        seen.push(candidate.clone());
        if let Some(alias) = aliases.get(candidate.as_str()) {
            return alias.to_string();
        }
        // catalog_ids is empty in the catalog=None/api_key=None call shape,
        // so the `candidate in catalog_ids` branch never matches.
    }

    if let Some((_, after)) = raw.split_once('/') {
        return after.trim().to_string();
    }
    raw
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Remove a `vendor/` prefix if present (keeps only the part after the first
/// slash).
fn strip_vendor_prefix(model_name: &str) -> &str {
    match model_name.split_once('/') {
        Some((_, remainder)) => remainder,
        None => model_name,
    }
}

/// Replace dots with hyphens in a model name
/// (`claude-sonnet-4.6` -> `claude-sonnet-4-6`).
fn dots_to_hyphens(model_name: &str) -> String {
    model_name.replace('.', "-")
}

/// Strip `provider/` only when the prefix matches the target provider.
///
/// This prevents arbitrary slash-bearing model IDs from being mangled on
/// native providers while still repairing manual config values like
/// `zai/glm-5.1` for the `zai` provider.
fn strip_matching_provider_prefix(model_name: &str, target_provider: &str) -> String {
    let (prefix, remainder) = match model_name.split_once('/') {
        Some(parts) => parts,
        None => return model_name.to_string(),
    };

    if prefix.trim().is_empty() || remainder.trim().is_empty() {
        return model_name.to_string();
    }

    let normalized_prefix = normalize_provider_alias(prefix);
    let normalized_target = normalize_provider_alias(target_provider);
    if !normalized_prefix.is_empty() && normalized_prefix == normalized_target {
        return remainder.trim().to_string();
    }
    model_name.to_string()
}

/// Detect the vendor slug from a bare model name.
///
/// Uses the first hyphen-delimited token of the model name to look up the
/// corresponding vendor in the vendor-prefix table. Also handles
/// case-insensitive matching and version-suffixed first tokens
/// (e.g. `qwen3.5-plus`).
///
/// Returns `None` if no vendor can be confidently detected. If the name
/// already contains a `vendor/` prefix, the lowercased prefix is returned.
pub fn detect_vendor(model_name: &str) -> Option<String> {
    let name = model_name.trim();
    if name.is_empty() {
        return None;
    }

    // If there's already a vendor/ prefix, extract it.
    if let Some((prefix, _)) = name.split_once('/') {
        let lowered = prefix.to_lowercase();
        if lowered.is_empty() {
            return None;
        }
        return Some(lowered);
    }

    let name_lower = name.to_lowercase();

    // Try first hyphen-delimited token (exact match).
    let first_token = name_lower.split('-').next().unwrap_or("");
    for (prefix, vendor) in vendor_prefixes() {
        if first_token == *prefix {
            return Some((*vendor).to_string());
        }
    }

    // Handle patterns where the first token includes version digits,
    // e.g. "qwen3.5-plus" -> startswith "qwen".
    for (prefix, vendor) in vendor_prefixes() {
        if name_lower.starts_with(prefix) {
            return Some((*vendor).to_string());
        }
    }

    None
}

/// Prepend the detected `vendor/` prefix if missing.
///
/// Used for aggregator providers that require `vendor/model` format. If the
/// name already contains a `/`, it is returned as-is. If no vendor can be
/// detected, the name is returned unchanged.
fn prepend_vendor(model_name: &str) -> String {
    if model_name.contains('/') {
        return model_name.to_string();
    }

    match detect_vendor(model_name) {
        Some(vendor) => format!("{vendor}/{model_name}"),
        None => model_name.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Main normalisation entry point
// ---------------------------------------------------------------------------

/// Translate a model name into the format the target provider's API expects.
///
/// This is the primary entry point for model name normalisation. It accepts
/// any user-facing model identifier and transforms it for the specific
/// provider that will receive the API call. Never panics — always returns a
/// best-effort string.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(normalize_model_for_provider("claude-sonnet-4.6", "openrouter"),
///            "anthropic/claude-sonnet-4.6");
/// assert_eq!(normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic"),
///            "claude-sonnet-4-6");
/// assert_eq!(normalize_model_for_provider("deepseek-r1", "deepseek"),
///            "deepseek-reasoner");
/// ```
pub fn normalize_model_for_provider(model_input: &str, target_provider: &str) -> String {
    let name = model_input.trim().to_string();
    if name.is_empty() {
        return name;
    }

    let provider = normalize_provider_alias(target_provider);

    // --- Aggregators: need vendor/model format ---
    if is_aggregator_provider(&provider) {
        return prepend_vendor(&name);
    }

    // --- OpenCode Zen / OpenCode Go: flat-namespace resellers ---
    if provider == "opencode-zen" || provider == "opencode-go" {
        let mut name = name;
        if let Some((_, bare_after_slash)) = name.split_once('/') {
            let trimmed = bare_after_slash.trim().to_string();
            if !trimmed.is_empty() {
                name = trimmed;
            }
        }
        if provider == "opencode-zen" && name.to_lowercase().starts_with("claude-") {
            return dots_to_hyphens(&name);
        }
        return name;
    }

    // --- Anthropic: strip matching provider prefix, dots -> hyphens ---
    if is_dot_to_hyphen_provider(&provider) {
        let bare = strip_matching_provider_prefix(&name, &provider);
        if bare.contains('/') {
            return bare;
        }
        return dots_to_hyphens(&bare);
    }

    // --- Copilot / Copilot ACP: delegate to the Copilot-specific normalizer ---
    if provider == "copilot" || provider == "copilot-acp" {
        let normalized = normalize_copilot_model_id(&name);
        if !normalized.is_empty() {
            return normalized;
        }
        // Fall through to the generic strip-vendor behaviour below.
    }

    // --- Copilot / Copilot ACP / openai-codex fallback:
    //     strip matching provider prefix, keep dots ---
    if is_strip_vendor_only_provider(&provider) {
        let stripped = strip_matching_provider_prefix(&name, &provider);
        if stripped == name && name.starts_with("openai/") {
            // openai-codex maps openai/gpt-5.4 -> gpt-5.4
            if let Some((_, after)) = name.split_once('/') {
                return after.to_string();
            }
        }
        return stripped;
    }

    // --- DeepSeek: map to one of two canonical names ---
    if provider == "deepseek" {
        let bare = strip_matching_provider_prefix(&name, &provider);
        if bare.contains('/') {
            return bare;
        }
        return normalize_for_deepseek(&bare);
    }

    // --- Direct providers: repair matching provider prefixes only ---
    if is_matching_prefix_strip_provider(&provider) {
        let mut result = strip_matching_provider_prefix(&name, &provider);
        if is_lowercase_model_provider(&provider) {
            result = result.to_lowercase();
        }
        return result;
    }

    // --- Authoritative native providers: preserve user-facing slugs as-is ---
    if is_authoritative_native_provider(&provider) {
        return name;
    }

    // --- Custom & all others: pass through as-is ---
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_prepends_vendor() {
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "openrouter"),
            "anthropic/claude-sonnet-4.6"
        );
        // Already prefixed -> unchanged.
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "openrouter"),
            "anthropic/claude-sonnet-4.6"
        );
        // Unknown vendor -> passthrough.
        assert_eq!(
            normalize_model_for_provider("my-custom-thing", "openrouter"),
            "my-custom-thing"
        );
        // Aliased aggregator (vercel -> ai-gateway).
        assert_eq!(
            normalize_model_for_provider("gpt-5.4", "vercel"),
            "openai/gpt-5.4"
        );
    }

    #[test]
    fn test_anthropic_dots_to_hyphens() {
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "anthropic"),
            "claude-sonnet-4-6"
        );
        // Non-matching slash prefix is preserved verbatim.
        assert_eq!(
            normalize_model_for_provider("openai/gpt-5.4", "anthropic"),
            "openai/gpt-5.4"
        );
    }

    #[test]
    fn test_copilot_keeps_dots() {
        // Copilot-specific normalizer maps anthropic/ dot form to bare dot form.
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "copilot"),
            "claude-sonnet-4.6"
        );
        // Dash-notation Claude repaired to dot-notation via alias table.
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4-6", "copilot"),
            "claude-sonnet-4.6"
        );
        // OpenAI alias mapping.
        assert_eq!(
            normalize_model_for_provider("openai/gpt-5", "copilot"),
            "gpt-5-mini"
        );
    }

    #[test]
    fn test_copilot_fallback_strips_slash() {
        // Unknown model not in alias table / catalog: slash-stripped fallback.
        assert_eq!(
            normalize_model_for_provider("openai/gpt-5.4", "copilot"),
            "gpt-5.4"
        );
    }

    #[test]
    fn test_openai_codex() {
        // openai-codex strips openai/ prefix, keeps dots.
        assert_eq!(
            normalize_model_for_provider("openai/gpt-5.4", "openai-codex"),
            "gpt-5.4"
        );
        // Non-openai slash prefix: not a matching provider, left as-is.
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "openai-codex"),
            "anthropic/claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_opencode_zen() {
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "opencode-zen"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            normalize_model_for_provider("minimax-m2.5-free", "opencode-zen"),
            "minimax-m2.5-free"
        );
        // Strips leading vendor slug.
        assert_eq!(
            normalize_model_for_provider("deepseek/deepseek-v4-flash", "opencode-zen"),
            "deepseek-v4-flash"
        );
        // Claude after slash strip -> dots to hyphens.
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "opencode-zen"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_opencode_go() {
        assert_eq!(
            normalize_model_for_provider("minimax/minimax-m2.7", "opencode-go"),
            "minimax-m2.7"
        );
        // Claude on opencode-go does NOT get dots->hyphens (only zen does).
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "opencode-go"),
            "claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_deepseek() {
        assert_eq!(
            normalize_model_for_provider("deepseek-v3", "deepseek"),
            "deepseek-chat"
        );
        assert_eq!(
            normalize_model_for_provider("deepseek-r1", "deepseek"),
            "deepseek-reasoner"
        );
        assert_eq!(
            normalize_model_for_provider("deepseek-reasoner", "deepseek"),
            "deepseek-reasoner"
        );
        // V-series first-class IDs pass through.
        assert_eq!(
            normalize_model_for_provider("deepseek-v4-pro", "deepseek"),
            "deepseek-v4-pro"
        );
        assert_eq!(
            normalize_model_for_provider("deepseek/deepseek-v4-flash-20260423", "deepseek"),
            "deepseek-v4-flash-20260423"
        );
        // Reasoner keyword.
        assert_eq!(
            normalize_model_for_provider("some-thinking-model", "deepseek"),
            "deepseek-reasoner"
        );
        // Non-matching slash prefix preserved.
        assert_eq!(
            normalize_model_for_provider("openrouter/foo", "deepseek"),
            "openrouter/foo"
        );
    }

    #[test]
    fn test_matching_prefix_strip_providers() {
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "zai"),
            "claude-sonnet-4.6"
        );
        // glm alias -> zai; matching prefix repaired.
        assert_eq!(
            normalize_model_for_provider("glm/glm-5.1", "zai"),
            "glm-5.1"
        );
        // custom passes through (matching strip provider, but no slash).
        assert_eq!(normalize_model_for_provider("my-model", "custom"), "my-model");
    }

    #[test]
    fn test_xiaomi_lowercase() {
        assert_eq!(
            normalize_model_for_provider("MiMo-V2.5-Pro", "xiaomi"),
            "mimo-v2.5-pro"
        );
        // mimo alias -> xiaomi.
        assert_eq!(
            normalize_model_for_provider("MiMo-V2.5-Pro", "mimo"),
            "mimo-v2.5-pro"
        );
    }

    #[test]
    fn test_authoritative_native() {
        assert_eq!(
            normalize_model_for_provider("gemini-3-pro", "gemini"),
            "gemini-3-pro"
        );
        assert_eq!(
            normalize_model_for_provider("org/some-model", "huggingface"),
            "org/some-model"
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(normalize_model_for_provider("", "anthropic"), "");
        assert_eq!(normalize_model_for_provider("   ", "openrouter"), "");
    }

    #[test]
    fn test_detect_vendor() {
        assert_eq!(detect_vendor("claude-sonnet-4.6").as_deref(), Some("anthropic"));
        assert_eq!(detect_vendor("gpt-5.4-mini").as_deref(), Some("openai"));
        assert_eq!(
            detect_vendor("anthropic/claude-sonnet-4.6").as_deref(),
            Some("anthropic")
        );
        assert_eq!(detect_vendor("my-custom-model"), None);
        assert_eq!(detect_vendor(""), None);
        // Version-suffixed first token via startswith fallback.
        assert_eq!(detect_vendor("qwen3.5-plus").as_deref(), Some("qwen"));
        assert_eq!(detect_vendor("llama-4-scout").as_deref(), Some("meta-llama"));
    }

    #[test]
    fn test_normalize_provider() {
        assert_eq!(normalize_provider(Some("glm")), "zai");
        assert_eq!(normalize_provider(Some("GITHUB")), "copilot");
        assert_eq!(normalize_provider(Some("  vercel  ")), "ai-gateway");
        // Default when None/empty.
        assert_eq!(normalize_provider(None), "openrouter");
        assert_eq!(normalize_provider(Some("")), "openrouter");
        // Unknown passes through.
        assert_eq!(normalize_provider(Some("anthropic")), "anthropic");
        // auto passes through.
        assert_eq!(normalize_provider(Some("auto")), "auto");
    }

    #[test]
    fn test_normalize_copilot_model_id() {
        assert_eq!(normalize_copilot_model_id(""), "");
        assert_eq!(normalize_copilot_model_id("openai/o3"), "gpt-5.3-codex");
        assert_eq!(normalize_copilot_model_id("claude-sonnet-4-5"), "claude-sonnet-4.5");
        // Unknown slash-bearing -> strip slash fallback.
        assert_eq!(normalize_copilot_model_id("foo/bar-baz"), "bar-baz");
        // Unknown bare -> passthrough.
        assert_eq!(normalize_copilot_model_id("totally-unknown"), "totally-unknown");
    }

    #[test]
    fn test_deepseek_v_series_regex() {
        assert!(deepseek_v_series_re().is_match("deepseek-v4-flash"));
        assert!(deepseek_v_series_re().is_match("deepseek-v5"));
        assert!(deepseek_v_series_re().is_match("deepseek-v4-flash-20260423"));
        assert!(deepseek_v_series_re().is_match("deepseek-v4.1"));
        assert!(!deepseek_v_series_re().is_match("deepseek-chat"));
        assert!(!deepseek_v_series_re().is_match("deepseek-reasoner"));
    }

    #[test]
    fn test_strip_matching_provider_prefix_blank_parts() {
        // Empty remainder -> unchanged.
        assert_eq!(strip_matching_provider_prefix("zai/", "zai"), "zai/");
        // Empty prefix -> unchanged.
        assert_eq!(strip_matching_provider_prefix("/model", "zai"), "/model");
    }
}
