//! Shared model-switching logic for CLI and gateway `/model` commands.
//!
//! Native Rust port of `hermes_cli/model_switch.py`.
//!
//! Both the CLI and gateway `/model` handlers share the same core pipeline:
//!
//! ```text
//! parse flags -> alias resolution -> provider resolution ->
//! credential resolution -> normalize model name ->
//! metadata lookup -> build result
//! ```
//!
//! This module ties together several foundation layers. Some of those layers
//! (`hermes_cli.providers`, `hermes_cli.runtime_provider`, `hermes_cli.models`)
//! are not yet ported to native Rust. To avoid blocking, this module expresses
//! its dependencies on those layers through small dependency traits
//! ([`ProviderResolver`], [`RuntimeResolver`], [`CatalogProvider`],
//! [`ModelValidator`]). A real integration implements these over
//! `hermes_cli.providers`, `hermes_cli.runtime_provider`, `hermes_cli.models`
//! and `agent.models_dev` (the `ag_models_dev` module in `hermes-core`).
//!
//! Callers that have the real layers wired can supply implementations; a
//! [`DefaultDeps`] is provided that is self-contained (no network) and exercises
//! the algorithmic core (flag parsing, alias resolution, version sorting,
//! pipeline wiring) standalone.
//!
//! Provider switching uses the `--provider` flag exclusively. No colon-based
//! `provider:model` syntax — colons are reserved for OpenRouter variant
//! suffixes (`:free`, `:extended`, `:fast`).

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

// NOTE: The models.dev catalog/metadata layer lives in `hermes-core`
// (`ag_models_dev.rs`) but its items are not re-exported from that crate's
// public API yet. To avoid a hard cross-crate dependency on private internals
// — and to keep this module compilable and testable standalone — catalog and
// metadata access is expressed through the [`CatalogProvider`] trait below.
// Once `ag_models_dev` is exported, an integrator can implement
// [`CatalogProvider`] (and the other dependency traits) over the real
// functions: `list_provider_models`, `get_model_info`,
// `get_model_capabilities`, plus the `ModelInfo`/`ModelCapabilities` types.

/// Subset of models.dev `ModelInfo` consumed by the switch result.
///
/// Mirrors `ag_models_dev::ModelInfo` fields that callers read for the
/// `/model` display (context window, cost, capabilities). Kept local so this
/// module does not hard-depend on the not-yet-exported core type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub family: String,
    pub provider_id: String,
    pub reasoning: bool,
    pub tool_call: bool,
    pub attachment: bool,
    pub context_window: i64,
    pub max_output: i64,
    pub cost_input: f64,
    pub cost_output: f64,
}

/// Subset of models.dev `ModelCapabilities`.
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
// Non-agentic model warning
// ---------------------------------------------------------------------------

/// Warning surfaced when a real Nous Hermes 3/4 chat model is selected.
pub const HERMES_MODEL_WARNING: &str = "Nous Research Hermes 3 & 4 models are NOT agentic and are not designed \
for use with Hermes Agent. They lack the tool-calling capabilities \
required for agent workflows. Consider using an agentic model instead \
(Claude, GPT, Gemini, DeepSeek, etc.).";

/// Match only the real Nous Research Hermes 3 / Hermes 4 chat families.
///
/// The previous substring check (`"hermes" in name.lower()`) false-positived on
/// unrelated local Modelfiles like `hermes-brain:qwen3-14b-ctx16k` that just
/// happen to carry "hermes" in their tag but are fully tool-capable.
///
/// Positive examples the regex must match:
///   `NousResearch/Hermes-3-Llama-3.1-70B`, `hermes-4-405b`, `openrouter/hermes3:70b`
/// Negative examples it must NOT match:
///   `hermes-brain:qwen3-14b-ctx16k`, `qwen3:14b`, `claude-opus-4-6`
fn nous_hermes_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[/:])hermes[-_ ]?[34](?:[-_.:]|$)").expect("valid regex")
    })
}

/// Return `true` if *model_name* is a real Nous Hermes 3/4 chat model.
///
/// Used to decide whether to surface the non-agentic warning at startup.
pub fn is_nous_hermes_non_agentic(model_name: &str) -> bool {
    if model_name.is_empty() {
        return false;
    }
    nous_hermes_re().is_match(model_name)
}

/// Return the warning string if *model_name* is a Nous Hermes 3/4 chat model.
fn check_hermes_model_warning(model_name: &str) -> &'static str {
    if is_nous_hermes_non_agentic(model_name) {
        HERMES_MODEL_WARNING
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Model aliases -- short names -> (vendor, family) with NO version numbers.
// ---------------------------------------------------------------------------

/// Vendor slug and family prefix used for catalog resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub vendor: String,
    pub family: String,
}

impl ModelIdentity {
    fn new(vendor: &str, family: &str) -> Self {
        ModelIdentity {
            vendor: vendor.to_string(),
            family: family.to_string(),
        }
    }
}

/// Static short-name aliases -> (vendor, family).
///
/// Returns an owned map (cheap to build, mirrors `MODEL_ALIASES` ordering is
/// irrelevant since lookup is by key).
pub fn model_aliases() -> HashMap<&'static str, ModelIdentity> {
    let mut m = HashMap::new();
    // Anthropic
    m.insert("sonnet", ModelIdentity::new("anthropic", "claude-sonnet"));
    m.insert("opus", ModelIdentity::new("anthropic", "claude-opus"));
    m.insert("haiku", ModelIdentity::new("anthropic", "claude-haiku"));
    m.insert("claude", ModelIdentity::new("anthropic", "claude"));
    // OpenAI
    m.insert("gpt5", ModelIdentity::new("openai", "gpt-5"));
    m.insert("gpt", ModelIdentity::new("openai", "gpt"));
    m.insert("codex", ModelIdentity::new("openai", "codex"));
    m.insert("o3", ModelIdentity::new("openai", "o3"));
    m.insert("o4", ModelIdentity::new("openai", "o4"));
    // Google
    m.insert("gemini", ModelIdentity::new("google", "gemini"));
    // DeepSeek
    m.insert("deepseek", ModelIdentity::new("deepseek", "deepseek-chat"));
    // X.AI
    m.insert("grok", ModelIdentity::new("x-ai", "grok"));
    // Meta
    m.insert("llama", ModelIdentity::new("meta-llama", "llama"));
    // Qwen / Alibaba
    m.insert("qwen", ModelIdentity::new("qwen", "qwen"));
    // MiniMax
    m.insert("minimax", ModelIdentity::new("minimax", "minimax"));
    // Nvidia
    m.insert("nemotron", ModelIdentity::new("nvidia", "nemotron"));
    // Moonshot / Kimi
    m.insert("kimi", ModelIdentity::new("moonshotai", "kimi"));
    // Z.AI / GLM
    m.insert("glm", ModelIdentity::new("z-ai", "glm"));
    // Step Plan (StepFun)
    m.insert("step", ModelIdentity::new("stepfun", "step"));
    // Xiaomi
    m.insert("mimo", ModelIdentity::new("xiaomi", "mimo"));
    // Arcee
    m.insert("trinity", ModelIdentity::new("arcee-ai", "trinity"));
    m
}

// ---------------------------------------------------------------------------
// Direct aliases — exact model+provider+base_url mappings.
// ---------------------------------------------------------------------------

/// Exact model mapping that bypasses catalog resolution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectAlias {
    pub model: String,
    pub provider: String,
    pub base_url: String,
}

impl DirectAlias {
    pub fn new(model: &str, provider: &str, base_url: &str) -> Self {
        DirectAlias {
            model: model.to_string(),
            provider: provider.to_string(),
            base_url: base_url.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

/// Result of a model switch attempt.
#[derive(Debug, Clone, Default)]
pub struct ModelSwitchResult {
    pub success: bool,
    pub new_model: String,
    pub target_provider: String,
    pub provider_changed: bool,
    pub api_key: String,
    pub base_url: String,
    pub api_mode: String,
    pub error_message: String,
    pub warning_message: String,
    pub provider_label: String,
    pub resolved_via_alias: String,
    pub capabilities: Option<ModelCapabilities>,
    pub model_info: Option<ModelInfo>,
    pub is_global: bool,
}

/// Result of switching to bare `custom` provider with auto-detect.
#[derive(Debug, Clone, Default)]
pub struct CustomAutoResult {
    pub success: bool,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub error_message: String,
}

// ---------------------------------------------------------------------------
// Flag parsing
// ---------------------------------------------------------------------------

/// Parsed `/model` command arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelFlags {
    pub model_input: String,
    pub explicit_provider: String,
    pub is_global: bool,
}

fn unicode_dash_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[\x{2012}\x{2013}\x{2014}\x{2015}](provider|global)").expect("valid regex")
    })
}

/// Parse `--provider` and `--global` flags from `/model` command args.
///
/// Returns `(model_input, explicit_provider, is_global)`.
///
/// Examples:
/// ```text
/// "sonnet"                               -> ("sonnet", "", false)
/// "sonnet --global"                      -> ("sonnet", "", true)
/// "sonnet --provider anthropic"          -> ("sonnet", "anthropic", false)
/// "--provider my-ollama"                 -> ("", "my-ollama", false)
/// "sonnet --provider anthropic --global" -> ("sonnet", "anthropic", true)
/// ```
pub fn parse_model_flags(raw_args: &str) -> ModelFlags {
    let mut is_global = false;
    let mut explicit_provider = String::new();

    // Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash).
    // A single Unicode dash before a flag keyword becomes "--".
    let normalized = unicode_dash_re().replace_all(raw_args, "--$1");
    let mut raw_args = normalized.to_string();

    // Extract --global
    if raw_args.contains("--global") {
        is_global = true;
        raw_args = raw_args.replace("--global", "").trim().to_string();
    }

    // Extract --provider <name>
    let parts: Vec<&str> = raw_args.split_whitespace().collect();
    let mut filtered: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        if parts[i] == "--provider" && i + 1 < parts.len() {
            explicit_provider = parts[i + 1].to_string();
            i += 2;
        } else {
            filtered.push(parts[i]);
            i += 1;
        }
    }

    let model_input = filtered.join(" ").trim().to_string();
    ModelFlags {
        model_input,
        explicit_provider,
        is_global,
    }
}

// ---------------------------------------------------------------------------
// Version-sort key
// ---------------------------------------------------------------------------

/// Sort key for model version preference.
///
/// Extracts version numbers after the family prefix and returns a key that
/// prefers higher versions, with suffix tokens (`pro`, `omni`, ...) as
/// tiebreakers. Lower sort value = preferred.
///
/// The returned tuple is `(version_key, suffix_rank, suffix)` where
/// `version_key` is a vector of negated version components (so higher versions
/// sort first).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSortKey {
    pub version_key: Vec<f64>,
    pub suffix_rank: i32,
    pub suffix: String,
}

impl Eq for ModelSortKey {}

impl PartialOrd for ModelSortKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModelSortKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // Compare version_key element-by-element (Python tuple semantics).
        let n = self.version_key.len().min(other.version_key.len());
        for i in 0..n {
            let a = self.version_key[i];
            let b = other.version_key[i];
            match a.partial_cmp(&b).unwrap_or(Ordering::Equal) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        // Shorter tuple sorts first if it's a prefix (Python: () < (x,)).
        match self.version_key.len().cmp(&other.version_key.len()) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.suffix_rank.cmp(&other.suffix_rank) {
            Ordering::Equal => {}
            ord => return ord,
        }
        self.suffix.cmp(&other.suffix)
    }
}

fn flush_num(num_buf: &str, nums: &mut Vec<f64>) {
    if num_buf.is_empty() {
        return;
    }
    let trimmed = num_buf.trim_end_matches('.');
    if trimmed.is_empty() {
        return;
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        nums.push(v);
    }
}

/// Compute the version/suffix sort key for *model_id* relative to *prefix*.
pub fn model_sort_key(model_id: &str, prefix: &str) -> ModelSortKey {
    // Strip the prefix (and optional "/" separator for aggregator slugs).
    let mut rest: &str = if model_id.len() >= prefix.len() {
        &model_id[prefix.len()..]
    } else {
        ""
    };
    if let Some(stripped) = rest.strip_prefix('/') {
        rest = stripped;
    }
    let rest = rest.trim_start_matches('-').trim();

    let mut nums: Vec<f64> = Vec::new();
    let mut suffix_buf = String::new();
    let mut state = "start";
    let mut num_buf = String::new();

    for ch in rest.chars() {
        match state {
            "start" => {
                if ch == 'v' || ch == 'V' {
                    state = "in_version";
                } else if ch.is_ascii_digit() {
                    state = "in_version";
                    num_buf.push(ch);
                } else if ch == '-' || ch == '_' || ch == '.' {
                    // skip separators before any content
                } else {
                    state = "in_suffix";
                    suffix_buf.push(ch);
                }
            }
            "in_version" => {
                if ch.is_ascii_digit() {
                    num_buf.push(ch);
                } else if ch == '.' {
                    if num_buf.contains('.') {
                        // Second dot — flush current number, start new component.
                        flush_num(&num_buf, &mut nums);
                        num_buf.clear();
                    } else {
                        num_buf.push(ch);
                    }
                } else if ch == '-' || ch == '_' {
                    flush_num(&num_buf, &mut nums);
                    num_buf.clear();
                    state = "between";
                } else {
                    flush_num(&num_buf, &mut nums);
                    num_buf.clear();
                    state = "in_suffix";
                    suffix_buf.push(ch);
                }
            }
            "between" => {
                if ch.is_ascii_digit() {
                    state = "in_version";
                    num_buf.clear();
                    num_buf.push(ch);
                } else if ch == 'v' || ch == 'V' {
                    state = "in_version";
                } else if ch == '-' || ch == '_' || ch == '.' {
                    // skip
                } else {
                    state = "in_suffix";
                    suffix_buf.push(ch);
                }
            }
            "in_suffix" => {
                suffix_buf.push(ch);
            }
            _ => {}
        }
    }

    // Flush remaining buffer (strip trailing dots — "5.4." → "5.4").
    if !num_buf.is_empty() && state == "in_version" {
        flush_num(&num_buf, &mut nums);
    }

    let suffix = suffix_buf
        .to_lowercase()
        .trim_matches(|c| c == '-' || c == '_' || c == '.')
        .trim()
        .to_string();

    // Negate versions so higher → sorts first.
    let version_key: Vec<f64> = nums.iter().map(|n| -n).collect();

    // Suffix quality ranking: pro/max/plus/turbo rank 0, everything else 1.
    let suffix_rank = match suffix.as_str() {
        "pro" | "max" | "plus" | "turbo" => 0,
        _ => 1,
    };

    ModelSortKey {
        version_key,
        suffix_rank,
        suffix,
    }
}

// ---------------------------------------------------------------------------
// Dependency traits (unported Python layers)
// ---------------------------------------------------------------------------

/// Resolved provider definition (subset of Python's `ProviderDef` used here).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderDef {
    pub id: String,
    pub name: String,
    pub base_url: String,
}

/// Resolved runtime credentials (subset of `resolve_runtime_provider()` output).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeProvider {
    pub provider: String,
    pub api_key: String,
    pub base_url: String,
    pub api_mode: String,
}

/// Outcome of validating a requested model against a provider/endpoint.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub accepted: bool,
    pub persist: bool,
    pub recognized: bool,
    pub message: String,
    pub corrected_model: Option<String>,
}

/// Provider identity / overlay layer (Python `hermes_cli.providers`).
pub trait ProviderResolver {
    /// Resolve a provider name via the full built-in -> models.dev -> user chain.
    fn resolve_provider_full(&self, name: &str) -> Option<ProviderDef>;
    /// Human-readable display name for a provider.
    fn get_label(&self, provider_id: &str) -> String;
    /// Whether the provider is a multi-model aggregator.
    fn is_aggregator(&self, provider: &str) -> bool;
    /// Wire-protocol API mode for a provider/endpoint.
    fn determine_api_mode(&self, provider: &str, base_url: &str) -> String;
    /// Canonical slug for a custom_providers display name (`custom:<name>`).
    fn custom_provider_slug(&self, display_name: &str) -> String {
        format!(
            "custom:{}",
            display_name.trim().to_lowercase().replace(' ', "-")
        )
    }
}

/// Runtime credential resolution (Python `hermes_cli.runtime_provider`).
pub trait RuntimeResolver {
    /// Resolve credentials/base_url/api_mode for *requested* + *target_model*.
    fn resolve_runtime_provider(
        &self,
        requested: &str,
        target_model: &str,
    ) -> Result<RuntimeProvider, String>;

    /// Auto-detect a model from a local endpoint's base URL.
    fn auto_detect_local_model(&self, _base_url: &str) -> Option<String> {
        None
    }

    /// Authenticated provider slugs (for fallback alias resolution).
    fn authenticated_provider_slugs(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Model validation + provider detection (Python `hermes_cli.models`).
pub trait ModelValidator {
    fn validate_requested_model(
        &self,
        model: &str,
        provider: &str,
        api_key: &str,
        base_url: &str,
        api_mode: Option<&str>,
    ) -> ValidationResult;

    /// Detect a (provider, model) pair for a bare model name.
    fn detect_provider_for_model(&self, _model: &str, _current_provider: &str) -> Option<(String, String)> {
        None
    }

    /// Copilot api_mode override.
    fn copilot_model_api_mode(&self, _model: &str, _api_key: &str) -> String {
        String::new()
    }

    /// OpenCode api_mode override.
    fn opencode_model_api_mode(&self, _provider: &str, _model: &str) -> String {
        String::new()
    }

    /// Static per-provider curated catalog supplement (Python `_PROVIDER_MODELS`).
    fn static_provider_models(&self, _provider: &str) -> Vec<String> {
        Vec::new()
    }
}

/// models.dev catalog + metadata layer (Python `agent.models_dev`).
///
/// Implement over `crate::ag_models_dev` once those items are exported. The
/// [`DefaultDeps`] implementation returns empty/`None` so the algorithmic core
/// (flag parsing, alias resolution, version sorting, pipeline wiring) is fully
/// exercisable standalone.
pub trait CatalogProvider {
    /// All model IDs for *provider* from the catalog.
    fn list_provider_models(&self, provider: &str) -> Vec<String>;
    /// Legacy capability metadata for *provider*/*model*.
    fn get_model_capabilities(&self, _provider: &str, _model: &str) -> Option<ModelCapabilities> {
        None
    }
    /// Full model metadata for *provider*/*model*.
    fn get_model_info(&self, _provider: &str, _model: &str) -> Option<ModelInfo> {
        None
    }
    /// Normalize a model name for a target provider.
    fn normalize_model_for_provider(&self, model_input: &str, _target_provider: &str) -> String {
        model_input.to_string()
    }
}

// ---------------------------------------------------------------------------
// Direct-alias source
// ---------------------------------------------------------------------------

/// Source of direct aliases (config.yaml `model_aliases:` + `model.aliases`).
///
/// The Python module lazily loads these from config; the Rust port takes the
/// loaded map as data so callers control config access. Empty = no direct
/// aliases (only the catalog/`MODEL_ALIASES` path is used).
pub type DirectAliasMap = HashMap<String, DirectAlias>;

// ---------------------------------------------------------------------------
// Alias resolution
// ---------------------------------------------------------------------------

/// Resolve a short alias against the current provider's catalog.
///
/// Returns `(provider, resolved_model_id, alias_name)` if a match is found, or
/// `None` if the alias doesn't exist or no matching model is available.
pub fn resolve_alias(
    raw_input: &str,
    current_provider: &str,
    direct_aliases: &DirectAliasMap,
    deps: &dyn ProviderResolver,
    catalog_deps: &dyn CatalogProvider,
    validator: &dyn ModelValidator,
) -> Option<(String, String, String)> {
    let key = raw_input.trim().to_lowercase();

    // Check direct aliases first (exact model+provider+base_url mappings).
    if let Some(direct) = direct_aliases.get(&key) {
        return Some((direct.provider.clone(), direct.model.clone(), key));
    }

    // Reverse lookup: match by model ID so full names route through direct
    // aliases instead of falling through to the catalog/OpenRouter.
    for (alias_name, da) in direct_aliases.iter() {
        if da.model.to_lowercase() == key {
            return Some((da.provider.clone(), da.model.clone(), alias_name.clone()));
        }
    }

    let aliases = model_aliases();
    let identity = aliases.get(key.as_str())?;
    let vendor = &identity.vendor;
    let family = &identity.family;

    // Build catalog from models.dev, then merge in static _PROVIDER_MODELS
    // entries that models.dev may be missing.
    let mut catalog = catalog_deps.list_provider_models(current_provider);
    let static_models = validator.static_provider_models(current_provider);
    if !static_models.is_empty() {
        let seen: std::collections::HashSet<String> =
            catalog.iter().map(|m| m.to_lowercase()).collect();
        for m in static_models {
            if !seen.contains(&m.to_lowercase()) {
                catalog.push(m);
            }
        }
    }

    let aggregator = deps.is_aggregator(current_provider);

    let matches: Vec<String> = if aggregator {
        let prefix = format!("{}/{}", vendor, family).to_lowercase();
        catalog
            .into_iter()
            .filter(|mid| mid.to_lowercase().starts_with(&prefix))
            .collect()
    } else {
        let family_lower = family.to_lowercase();
        catalog
            .into_iter()
            .filter(|mid| mid.to_lowercase().starts_with(&family_lower))
            .collect()
    };

    if matches.is_empty() {
        return None;
    }

    let prefix_for_sort = if aggregator {
        format!("{}/{}", vendor, family)
    } else {
        family.clone()
    };

    let mut matches = matches;
    matches.sort_by(|a, b| {
        model_sort_key(a, &prefix_for_sort).cmp(&model_sort_key(b, &prefix_for_sort))
    });

    Some((current_provider.to_string(), matches[0].clone(), key))
}

/// Try to resolve an alias on the user's authenticated providers.
///
/// Falls back to `("openrouter", "nous")` when no authenticated providers are
/// supplied (backwards compat for non-interactive callers).
pub fn resolve_alias_fallback(
    raw_input: &str,
    authenticated_providers: &[String],
    direct_aliases: &DirectAliasMap,
    deps: &dyn ProviderResolver,
    catalog_deps: &dyn CatalogProvider,
    validator: &dyn ModelValidator,
) -> Option<(String, String, String)> {
    let default_list = ["openrouter".to_string(), "nous".to_string()];
    let providers: &[String] = if authenticated_providers.is_empty() {
        &default_list
    } else {
        authenticated_providers
    };
    for provider in providers {
        if let Some(result) =
            resolve_alias(raw_input, provider, direct_aliases, deps, catalog_deps, validator)
        {
            return Some(result);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Core model-switching pipeline
// ---------------------------------------------------------------------------

/// Inputs to [`switch_model`].
#[derive(Debug, Clone, Default)]
pub struct SwitchModelArgs {
    pub raw_input: String,
    pub current_provider: String,
    pub current_model: String,
    pub current_base_url: String,
    pub current_api_key: String,
    pub is_global: bool,
    pub explicit_provider: String,
    /// `providers:` dict from config.yaml: slug -> { "models": <dict|list> }.
    pub user_providers: HashMap<String, UserProviderCfg>,
    /// `custom_providers:` list from config.yaml.
    pub custom_providers: Vec<CustomProviderEntry>,
}

/// Subset of a `providers:` config entry consulted during validation override.
#[derive(Debug, Clone, Default)]
pub struct UserProviderCfg {
    /// Model IDs declared by the user (dict keys or list entries).
    pub models: Vec<String>,
}

/// Subset of a `custom_providers:` entry consulted during validation override.
#[derive(Debug, Clone, Default)]
pub struct CustomProviderEntry {
    pub name: String,
    pub base_url: String,
    pub model: String,
    pub models: Vec<String>,
}

fn v1_strip_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"/v1/?$").expect("valid regex"))
}

/// Core model-switching pipeline shared between CLI and gateway.
///
/// See the module docs for the full resolution chain. Dependencies on unported
/// Python layers are supplied via [`ProviderResolver`], [`RuntimeResolver`],
/// [`ModelValidator`], and the direct-alias map.
pub fn switch_model(
    args: &SwitchModelArgs,
    direct_aliases: &DirectAliasMap,
    provider_deps: &dyn ProviderResolver,
    runtime_deps: &dyn RuntimeResolver,
    catalog_deps: &dyn CatalogProvider,
    validator: &dyn ModelValidator,
) -> ModelSwitchResult {
    let mut resolved_alias = String::new();
    let mut new_model = args.raw_input.trim().to_string();
    let mut target_provider = args.current_provider.clone();
    let explicit_provider = &args.explicit_provider;
    let is_global = args.is_global;

    // =================================================================
    // PATH A: Explicit --provider given
    // =================================================================
    if !explicit_provider.is_empty() {
        let pdef = match provider_deps.resolve_provider_full(explicit_provider) {
            Some(p) => p,
            None => {
                let switch_err = format!(
                    "Unknown provider '{explicit_provider}'. Check 'hermes model' for available \
                     providers, or define it in config.yaml under 'providers:'."
                );
                return ModelSwitchResult {
                    success: false,
                    is_global,
                    error_message: switch_err,
                    ..Default::default()
                };
            }
        };

        target_provider = pdef.id.clone();

        // If no model specified, try auto-detect from endpoint.
        if new_model.is_empty() {
            if !pdef.base_url.is_empty() {
                match runtime_deps.auto_detect_local_model(&pdef.base_url) {
                    Some(detected) => new_model = detected,
                    None => {
                        return ModelSwitchResult {
                            success: false,
                            target_provider,
                            provider_label: pdef.name.clone(),
                            is_global,
                            error_message: format!(
                                "No model detected on {} ({}). Specify the model explicitly: \
                                 /model <model-name> --provider {}",
                                pdef.name, pdef.base_url, explicit_provider
                            ),
                            ..Default::default()
                        };
                    }
                }
            } else {
                return ModelSwitchResult {
                    success: false,
                    target_provider,
                    provider_label: pdef.name.clone(),
                    is_global,
                    error_message: format!(
                        "Provider '{}' has no base URL configured. Specify a model: \
                         /model <model-name> --provider {}",
                        pdef.name, explicit_provider
                    ),
                    ..Default::default()
                };
            }
        }

        // Resolve alias on the TARGET provider.
        if let Some((_, m, alias)) = resolve_alias(
            &new_model,
            &target_provider,
            direct_aliases,
            provider_deps,
            catalog_deps,
            validator,
        ) {
            new_model = m;
            resolved_alias = alias;
        }
    } else {
        // =================================================================
        // PATH B: No explicit provider — resolve from model input
        // =================================================================

        // --- Step a: Try alias resolution on current provider ---
        let alias_result = resolve_alias(
            &args.raw_input,
            &args.current_provider,
            direct_aliases,
            provider_deps,
            catalog_deps,
            validator,
        );

        if let Some((tp, m, alias)) = alias_result {
            target_provider = tp;
            new_model = m;
            resolved_alias = alias;
            log::debug!(
                "Alias '{}' resolved to {} on {}",
                resolved_alias,
                new_model,
                target_provider
            );
        } else {
            // --- Step b: Alias exists but not on current provider -> fallback ---
            let key = args.raw_input.trim().to_lowercase();
            let aliases = model_aliases();
            if aliases.contains_key(key.as_str()) {
                let authed = runtime_deps.authenticated_provider_slugs();
                let fallback = resolve_alias_fallback(
                    &args.raw_input,
                    &authed,
                    direct_aliases,
                    provider_deps,
                    catalog_deps,
                    validator,
                );
                if let Some((tp, m, alias)) = fallback {
                    target_provider = tp;
                    new_model = m;
                    resolved_alias = alias;
                    log::debug!(
                        "Alias '{}' resolved via fallback to {} on {}",
                        resolved_alias,
                        new_model,
                        target_provider
                    );
                } else {
                    let identity = &aliases[key.as_str()];
                    return ModelSwitchResult {
                        success: false,
                        is_global,
                        error_message: format!(
                            "Alias '{}' maps to {}/{} but no matching model was found in any \
                             provider catalog. Try specifying the full model name.",
                            key, identity.vendor, identity.family
                        ),
                        ..Default::default()
                    };
                }
            } else {
                // --- Step c: On aggregator, convert vendor:model to vendor/model ---
                if let Some(colon_pos) = args.raw_input.find(':') {
                    if colon_pos > 0
                        && !args.raw_input.contains('/')
                        && provider_deps.is_aggregator(&args.current_provider)
                    {
                        let left = args.raw_input[..colon_pos].trim().to_lowercase();
                        let right = args.raw_input[colon_pos + 1..].trim().to_string();
                        if !left.is_empty() && !right.is_empty() {
                            new_model = format!("{left}/{right}");
                            log::debug!(
                                "Converted vendor:model '{}' to aggregator slug '{}'",
                                args.raw_input,
                                new_model
                            );
                        }
                    }
                }
            }
        }

        // --- Step d: Aggregator catalog search ---
        let mut resolved_in_current_catalog = false;
        if provider_deps.is_aggregator(&target_provider) && resolved_alias.is_empty() {
            let catalog = catalog_deps.list_provider_models(&target_provider);
            if !catalog.is_empty() {
                let new_model_lower = new_model.to_lowercase();
                let mut matched = false;
                for mid in &catalog {
                    if mid.to_lowercase() == new_model_lower {
                        new_model = mid.clone();
                        resolved_in_current_catalog = true;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    for mid in &catalog {
                        if let Some((_, bare)) = mid.split_once('/') {
                            if bare.to_lowercase() == new_model_lower {
                                new_model = mid.clone();
                                resolved_in_current_catalog = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // --- Step e: detect_provider_for_model() as last resort ---
        let base = &args.current_base_url;
        let is_custom = args.current_provider == "custom"
            || args.current_provider == "local"
            || base.contains("localhost")
            || base.contains("127.0.0.1");

        if target_provider == args.current_provider
            && !is_custom
            && resolved_alias.is_empty()
            && !resolved_in_current_catalog
        {
            if let Some((dp, dm)) =
                validator.detect_provider_for_model(&new_model, &args.current_provider)
            {
                target_provider = dp;
                new_model = dm;
            }
        }
    }

    // =================================================================
    // COMMON PATH: Resolve credentials, normalize, get metadata
    // =================================================================

    let provider_changed = target_provider != args.current_provider;
    let mut provider_label = provider_deps.get_label(&target_provider);
    if target_provider.starts_with("custom:") {
        if let Some(custom_pdef) = provider_deps.resolve_provider_full(&target_provider) {
            provider_label = custom_pdef.name;
        }
    }

    // --- Resolve credentials ---
    let mut api_key = args.current_api_key.clone();
    let mut base_url = args.current_base_url.clone();
    let mut api_mode = String::new();

    if provider_changed || !explicit_provider.is_empty() {
        match runtime_deps.resolve_runtime_provider(&target_provider, &new_model) {
            Ok(runtime) => {
                api_key = runtime.api_key;
                base_url = runtime.base_url;
                api_mode = runtime.api_mode;
            }
            Err(e) => {
                let error_message =
                    format!("Could not resolve credentials for provider '{provider_label}': {e}");
                return ModelSwitchResult {
                    success: false,
                    target_provider,
                    provider_label,
                    is_global,
                    error_message,
                    ..Default::default()
                };
            }
        }
    } else if let Ok(runtime) =
        runtime_deps.resolve_runtime_provider(&args.current_provider, &new_model)
    {
        // If resolution fell through to "custom", keep existing credentials.
        if runtime.provider != "custom" {
            api_key = runtime.api_key;
            base_url = runtime.base_url;
            api_mode = runtime.api_mode;
        }
    }

    // --- Direct alias override: use exact base_url from the alias if set ---
    if !resolved_alias.is_empty() {
        if let Some(da) = direct_aliases.get(&resolved_alias) {
            if !da.base_url.is_empty() {
                base_url = da.base_url.clone();
                api_mode = String::new(); // clear so determine_api_mode re-detects
                if api_key.is_empty() {
                    api_key = "no-key-required".to_string();
                }
            }
        }
    }

    // --- Normalize model name for target provider ---
    new_model = catalog_deps.normalize_model_for_provider(&new_model, &target_provider);

    // --- Validate ---
    let mut validation = validator.validate_requested_model(
        &new_model,
        &target_provider,
        &api_key,
        &base_url,
        if api_mode.is_empty() {
            None
        } else {
            Some(api_mode.as_str())
        },
    );

    // Override rejection if model is in the user's saved provider config.
    if !validation.accepted {
        let mut override_ = false;
        for (slug, cfg) in &args.user_providers {
            if slug == &target_provider {
                if cfg.models.iter().any(|m| m == &new_model) {
                    override_ = true;
                }
                break;
            }
        }
        if !override_ {
            for entry in &args.custom_providers {
                let entry_slug = if !entry.name.is_empty() {
                    format!("custom:{}", entry.name)
                } else {
                    String::new()
                };
                if entry_slug == target_provider || entry.base_url == base_url {
                    if new_model == entry.model {
                        override_ = true;
                        break;
                    }
                    if entry.models.iter().any(|m| m == &new_model) {
                        override_ = true;
                        break;
                    }
                }
            }
        }
        if override_ {
            validation = ValidationResult {
                accepted: true,
                persist: true,
                recognized: false,
                message: validation.message,
                corrected_model: None,
            };
        } else {
            let msg = if validation.message.is_empty() {
                "Invalid model".to_string()
            } else {
                validation.message
            };
            return ModelSwitchResult {
                success: false,
                new_model,
                target_provider,
                provider_label,
                is_global,
                error_message: msg,
                ..Default::default()
            };
        }
    }

    // Apply auto-correction if validation found a closer match.
    if let Some(corrected) = &validation.corrected_model {
        if !corrected.is_empty() {
            new_model = corrected.clone();
        }
    }

    // --- Copilot api_mode override ---
    if target_provider == "copilot" || target_provider == "github-copilot" {
        api_mode = validator.copilot_model_api_mode(&new_model, &api_key);
    }

    // --- OpenCode api_mode override ---
    if matches!(target_provider.as_str(), "opencode-zen" | "opencode-go" | "opencode") {
        api_mode = validator.opencode_model_api_mode(&target_provider, &new_model);
    }

    // --- Determine api_mode if not already set ---
    if api_mode.is_empty() {
        api_mode = provider_deps.determine_api_mode(&target_provider, &base_url);
    }

    // OpenCode base URLs end with /v1 for OpenAI-compatible models, but the
    // Anthropic SDK prepends its own /v1/messages. Strip the trailing /v1.
    if api_mode == "anthropic_messages"
        && matches!(target_provider.as_str(), "opencode-zen" | "opencode-go")
        && !base_url.is_empty()
    {
        base_url = v1_strip_re().replace(&base_url, "").to_string();
    }

    // --- Get capabilities (legacy) ---
    let capabilities = catalog_deps.get_model_capabilities(&target_provider, &new_model);

    // --- Get full model info from models.dev ---
    let model_info = catalog_deps.get_model_info(&target_provider, &new_model);

    // --- Collect warnings ---
    let mut warnings: Vec<String> = Vec::new();
    if !validation.message.is_empty() {
        warnings.push(validation.message.clone());
    }
    let hermes_warn = check_hermes_model_warning(&new_model);
    if !hermes_warn.is_empty() {
        warnings.push(hermes_warn.to_string());
    }

    ModelSwitchResult {
        success: true,
        new_model,
        target_provider,
        provider_changed,
        api_key,
        base_url,
        api_mode,
        warning_message: if warnings.is_empty() {
            String::new()
        } else {
            warnings.join(" | ")
        },
        provider_label,
        resolved_via_alias: resolved_alias,
        capabilities,
        model_info,
        is_global,
        error_message: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Default self-contained dependency implementation
// ---------------------------------------------------------------------------

/// A self-contained [`ProviderResolver`] / [`RuntimeResolver`] /
/// [`CatalogProvider`] / [`ModelValidator`] suitable for tests and standalone
/// use.
///
/// It knows a fixed set of well-known providers, treats `openrouter` and
/// `nous` as aggregators, returns an empty catalog (no models.dev access from
/// this crate), and accepts any non-empty model during validation (mirroring
/// the Python "recognized=false but accepted" path used when no live
/// `/v1/models` check is available). Integrators should provide richer
/// implementations wired to the real `agent.models_dev` /
/// `hermes_cli.providers` / `hermes_cli.runtime_provider` layers.
#[derive(Debug, Default, Clone)]
pub struct DefaultDeps;

/// Minimal built-in provider display labels for [`DefaultDeps`].
fn default_provider_label(canonical: &str) -> String {
    match canonical {
        "anthropic" => "Anthropic",
        "openai" => "OpenAI",
        "openai-codex" => "OpenAI Codex",
        "google" => "Google",
        "deepseek" => "DeepSeek",
        "x-ai" | "xai" => "xAI",
        "openrouter" => "OpenRouter",
        "nous" => "Nous Research",
        "groq" => "Groq",
        "mistral" => "Mistral",
        _ => return canonical.to_string(),
    }
    .to_string()
}

impl ProviderResolver for DefaultDeps {
    fn resolve_provider_full(&self, name: &str) -> Option<ProviderDef> {
        let canonical = name.trim().to_lowercase();
        if canonical.is_empty() {
            return None;
        }
        // Recognize a fixed set of known providers + any custom:* slug.
        let known = matches!(
            canonical.as_str(),
            "anthropic"
                | "openai"
                | "openai-codex"
                | "google"
                | "deepseek"
                | "x-ai"
                | "xai"
                | "openrouter"
                | "nous"
                | "groq"
                | "mistral"
        ) || canonical.starts_with("custom:");
        if known {
            Some(ProviderDef {
                id: canonical.clone(),
                name: default_provider_label(&canonical),
                base_url: String::new(),
            })
        } else {
            None
        }
    }

    fn get_label(&self, provider_id: &str) -> String {
        default_provider_label(&provider_id.trim().to_lowercase())
    }

    fn is_aggregator(&self, provider: &str) -> bool {
        matches!(provider.trim().to_lowercase().as_str(), "openrouter" | "nous")
    }

    fn determine_api_mode(&self, _provider: &str, base_url: &str) -> String {
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
        "chat_completions".to_string()
    }
}

impl RuntimeResolver for DefaultDeps {
    fn resolve_runtime_provider(
        &self,
        requested: &str,
        _target_model: &str,
    ) -> Result<RuntimeProvider, String> {
        // No real credential resolution available standalone; echo provider.
        Ok(RuntimeProvider {
            provider: requested.to_string(),
            api_key: String::new(),
            base_url: String::new(),
            api_mode: String::new(),
        })
    }
}

impl CatalogProvider for DefaultDeps {
    fn list_provider_models(&self, _provider: &str) -> Vec<String> {
        // No models.dev access from this crate standalone.
        Vec::new()
    }
}

impl ModelValidator for DefaultDeps {
    fn validate_requested_model(
        &self,
        model: &str,
        _provider: &str,
        _api_key: &str,
        _base_url: &str,
        _api_mode: Option<&str>,
    ) -> ValidationResult {
        if model.trim().is_empty() {
            return ValidationResult {
                accepted: false,
                persist: false,
                recognized: false,
                message: "Invalid model".to_string(),
                corrected_model: None,
            };
        }
        ValidationResult {
            accepted: true,
            persist: true,
            recognized: false,
            message: String::new(),
            corrected_model: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_nous_hermes_non_agentic() {
        assert!(is_nous_hermes_non_agentic("NousResearch/Hermes-3-Llama-3.1-70B"));
        assert!(is_nous_hermes_non_agentic("hermes-4-405b"));
        assert!(is_nous_hermes_non_agentic("openrouter/hermes3:70b"));
        // Negatives
        assert!(!is_nous_hermes_non_agentic("hermes-brain:qwen3-14b-ctx16k"));
        assert!(!is_nous_hermes_non_agentic("qwen3:14b"));
        assert!(!is_nous_hermes_non_agentic("claude-opus-4-6"));
        assert!(!is_nous_hermes_non_agentic(""));
    }

    #[test]
    fn test_check_hermes_warning() {
        assert_eq!(check_hermes_model_warning("hermes-4-405b"), HERMES_MODEL_WARNING);
        assert_eq!(check_hermes_model_warning("claude-opus-4-6"), "");
    }

    #[test]
    fn test_parse_model_flags_plain() {
        let f = parse_model_flags("sonnet");
        assert_eq!(f.model_input, "sonnet");
        assert_eq!(f.explicit_provider, "");
        assert!(!f.is_global);
    }

    #[test]
    fn test_parse_model_flags_global() {
        let f = parse_model_flags("sonnet --global");
        assert_eq!(f.model_input, "sonnet");
        assert!(f.is_global);
    }

    #[test]
    fn test_parse_model_flags_provider() {
        let f = parse_model_flags("sonnet --provider anthropic");
        assert_eq!(f.model_input, "sonnet");
        assert_eq!(f.explicit_provider, "anthropic");
        assert!(!f.is_global);
    }

    #[test]
    fn test_parse_model_flags_provider_only() {
        let f = parse_model_flags("--provider my-ollama");
        assert_eq!(f.model_input, "");
        assert_eq!(f.explicit_provider, "my-ollama");
    }

    #[test]
    fn test_parse_model_flags_all() {
        let f = parse_model_flags("sonnet --provider anthropic --global");
        assert_eq!(f.model_input, "sonnet");
        assert_eq!(f.explicit_provider, "anthropic");
        assert!(f.is_global);
    }

    #[test]
    fn test_parse_model_flags_unicode_dash() {
        // Em dash before "provider" becomes "--provider".
        let f = parse_model_flags("sonnet \u{2014}provider anthropic");
        assert_eq!(f.model_input, "sonnet");
        assert_eq!(f.explicit_provider, "anthropic");
        // En dash before "global"
        let f2 = parse_model_flags("opus \u{2013}global");
        assert_eq!(f2.model_input, "opus");
        assert!(f2.is_global);
    }

    #[test]
    fn test_model_sort_key_version_pref() {
        // Higher version should sort first (smaller key).
        let k25 = model_sort_key("mimo-v2.5-pro", "mimo");
        let k2 = model_sort_key("mimo-v2-pro", "mimo");
        assert!(k25 < k2, "v2.5-pro should sort before v2-pro");
    }

    #[test]
    fn test_model_sort_key_suffix_pref() {
        // pro suffix ranks above no suffix at same version.
        let pro = model_sort_key("mimo-v2.5-pro", "mimo");
        let none = model_sort_key("mimo-v2.5", "mimo");
        assert!(pro < none, "pro should sort before bare version");
        assert_eq!(pro.suffix, "pro");
        assert_eq!(pro.suffix_rank, 0);
        assert_eq!(none.suffix_rank, 1);
    }

    #[test]
    fn test_model_sort_key_components() {
        // "v2.5" parses as the single float 2.5 (first dot stays in the
        // number buffer), matching the Python state machine.
        let k = model_sort_key("mimo-v2.5-pro", "mimo");
        assert_eq!(k.version_key, vec![-2.5]);
        assert_eq!(k.suffix, "pro");

        let k2 = model_sort_key("mimo-v2-omni", "mimo");
        assert_eq!(k2.version_key, vec![-2.0]);
        assert_eq!(k2.suffix, "omni");
        assert_eq!(k2.suffix_rank, 1);
    }

    #[test]
    fn test_model_sort_key_aggregator_prefix() {
        // With slash-separated aggregator prefix.
        let k = model_sort_key("xiaomi/mimo-v2.5-pro", "xiaomi/mimo");
        assert_eq!(k.version_key, vec![-2.5]);
        assert_eq!(k.suffix, "pro");
    }

    #[test]
    fn test_full_sort_picks_highest() {
        let mut models = vec![
            "mimo-v2-pro".to_string(),
            "mimo-v2.5-pro".to_string(),
            "mimo-v2-omni".to_string(),
            "mimo-v2.5".to_string(),
        ];
        models.sort_by(|a, b| model_sort_key(a, "mimo").cmp(&model_sort_key(b, "mimo")));
        assert_eq!(models[0], "mimo-v2.5-pro");
    }

    #[test]
    fn test_custom_provider_slug() {
        let deps = DefaultDeps;
        assert_eq!(deps.custom_provider_slug("My Ollama"), "custom:my-ollama");
        assert_eq!(deps.custom_provider_slug("  Foo Bar  "), "custom:foo-bar");
    }

    #[test]
    fn test_resolve_alias_direct() {
        let deps = DefaultDeps;
        let mut da: DirectAliasMap = HashMap::new();
        da.insert(
            "qwen".to_string(),
            DirectAlias::new("qwen3.5:397b", "custom", "https://ollama.com/v1"),
        );
        let r = resolve_alias("qwen", "openrouter", &da, &deps, &deps, &deps);
        assert_eq!(
            r,
            Some(("custom".to_string(), "qwen3.5:397b".to_string(), "qwen".to_string()))
        );
    }

    #[test]
    fn test_resolve_alias_reverse_by_model_id() {
        let deps = DefaultDeps;
        let mut da: DirectAliasMap = HashMap::new();
        da.insert(
            "k2".to_string(),
            DirectAlias::new("kimi-k2.5", "moonshotai", ""),
        );
        // Looking up the full model id should reverse-map to the alias.
        let r = resolve_alias("kimi-k2.5", "openrouter", &da, &deps, &deps, &deps);
        assert_eq!(
            r,
            Some(("moonshotai".to_string(), "kimi-k2.5".to_string(), "k2".to_string()))
        );
    }

    #[test]
    fn test_resolve_alias_unknown_returns_none() {
        let deps = DefaultDeps;
        let da: DirectAliasMap = HashMap::new();
        let r = resolve_alias("totally-unknown-xyz", "openrouter", &da, &deps, &deps, &deps);
        assert!(r.is_none());
    }

    #[test]
    fn test_switch_model_explicit_unknown_provider() {
        let deps = DefaultDeps;
        let da: DirectAliasMap = HashMap::new();
        let args = SwitchModelArgs {
            raw_input: "some-model".to_string(),
            current_provider: "anthropic".to_string(),
            explicit_provider: "this-provider-does-not-exist-zzz".to_string(),
            ..Default::default()
        };
        let res = switch_model(&args, &da, &deps, &deps, &deps, &deps);
        assert!(!res.success);
        assert!(res.error_message.contains("Unknown provider"));
    }

    #[test]
    fn test_switch_model_empty_model_rejected_by_validator() {
        let deps = DefaultDeps;
        let da: DirectAliasMap = HashMap::new();
        // No explicit provider, empty model -> validator rejects empty model.
        let args = SwitchModelArgs {
            raw_input: "".to_string(),
            current_provider: "anthropic".to_string(),
            ..Default::default()
        };
        let res = switch_model(&args, &da, &deps, &deps, &deps, &deps);
        assert!(!res.success);
    }

    #[test]
    fn test_switch_model_validation_override_user_provider() {
        // Validator rejects, but user_providers declares the model -> override.
        struct RejectAll;
        impl ProviderResolver for RejectAll {
            fn resolve_provider_full(&self, _name: &str) -> Option<ProviderDef> {
                None
            }
            fn get_label(&self, p: &str) -> String {
                p.to_string()
            }
            fn is_aggregator(&self, _p: &str) -> bool {
                false
            }
            fn determine_api_mode(&self, _p: &str, _b: &str) -> String {
                "chat_completions".to_string()
            }
        }
        impl RuntimeResolver for RejectAll {
            fn resolve_runtime_provider(
                &self,
                requested: &str,
                _m: &str,
            ) -> Result<RuntimeProvider, String> {
                Ok(RuntimeProvider {
                    provider: requested.to_string(),
                    ..Default::default()
                })
            }
        }
        impl CatalogProvider for RejectAll {
            fn list_provider_models(&self, _p: &str) -> Vec<String> {
                Vec::new()
            }
        }
        impl ModelValidator for RejectAll {
            fn validate_requested_model(
                &self,
                _m: &str,
                _p: &str,
                _k: &str,
                _b: &str,
                _am: Option<&str>,
            ) -> ValidationResult {
                ValidationResult {
                    accepted: false,
                    message: "rejected".to_string(),
                    ..Default::default()
                }
            }
        }

        let deps = RejectAll;
        let da: DirectAliasMap = HashMap::new();
        let mut user_providers = HashMap::new();
        user_providers.insert(
            "myprov".to_string(),
            UserProviderCfg {
                models: vec!["secret-model".to_string()],
            },
        );
        let args = SwitchModelArgs {
            raw_input: "secret-model".to_string(),
            current_provider: "myprov".to_string(),
            user_providers,
            ..Default::default()
        };
        let res = switch_model(&args, &da, &deps, &deps, &deps, &deps);
        assert!(res.success, "user-provider override should accept the model");
        assert_eq!(res.new_model, "secret-model");
    }
}
