use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderProfile {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub api_mode: &'static str,
    pub env_vars: &'static [&'static str],
    pub base_url: &'static str,
    pub auth_type: &'static str,
    pub default_headers: &'static [(&'static str, &'static str)],
}

impl ProviderProfile {
    pub fn matches(&self, value: &str) -> bool {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return false;
        }
        self.name == normalized || self.aliases.iter().any(|alias| *alias == normalized)
    }

    pub fn api_key_env_vars(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.env_vars
            .iter()
            .copied()
            .filter(|name| !name.ends_with("_BASE_URL"))
    }

    pub fn base_url_env_var(&self) -> Option<&'static str> {
        self.env_vars
            .iter()
            .copied()
            .find(|name| name.ends_with("_BASE_URL"))
    }
}

const EMPTY_HEADERS: &[(&str, &str)] = &[];
const AI_GATEWAY_HEADERS: &[(&str, &str)] = &[("x-source", "hermes-agent")];
const KIMI_HEADERS: &[(&str, &str)] = &[("User-Agent", "hermes-agent/1.0")];
const AZURE_FOUNDRY_RESPONSES_PREFIXES: &[&str] = &["codex", "gpt-5", "o1", "o3", "o4"];

const PROVIDERS: &[ProviderProfile] = &[
    ProviderProfile {
        name: "openai",
        aliases: &["oa"],
        api_mode: "chat_completions",
        env_vars: &["OPENAI_API_KEY"],
        base_url: "https://api.openai.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "custom",
        aliases: &[
            "ollama",
            "local",
            "vllm",
            "llamacpp",
            "llama.cpp",
            "llama-cpp",
        ],
        api_mode: "chat_completions",
        env_vars: &[],
        base_url: "",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "openrouter",
        aliases: &["or"],
        api_mode: "chat_completions",
        env_vars: &["OPENROUTER_API_KEY"],
        base_url: "https://openrouter.ai/api/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "ai-gateway",
        aliases: &["vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"],
        api_mode: "chat_completions",
        env_vars: &["AI_GATEWAY_API_KEY"],
        base_url: "https://ai-gateway.vercel.sh/v1",
        auth_type: "api_key",
        default_headers: AI_GATEWAY_HEADERS,
    },
    ProviderProfile {
        name: "alibaba-coding-plan",
        aliases: &["alibaba_coding", "alibaba-coding", "dashscope-coding"],
        api_mode: "chat_completions",
        env_vars: &[
            "ALIBABA_CODING_PLAN_API_KEY",
            "DASHSCOPE_API_KEY",
            "ALIBABA_CODING_PLAN_BASE_URL",
        ],
        base_url: "https://coding-intl.dashscope.aliyuncs.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "alibaba",
        aliases: &["dashscope", "alibaba-cloud", "qwen-dashscope"],
        api_mode: "chat_completions",
        env_vars: &["DASHSCOPE_API_KEY"],
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "anthropic",
        aliases: &["claude", "claude-oauth", "claude-code"],
        api_mode: "anthropic_messages",
        env_vars: &[
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ],
        base_url: "https://api.anthropic.com",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "arcee",
        aliases: &["arcee-ai", "arceeai"],
        api_mode: "chat_completions",
        env_vars: &["ARCEEAI_API_KEY"],
        base_url: "https://api.arcee.ai/api/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "azure-foundry",
        aliases: &["azure", "azure-ai-foundry", "azure-ai"],
        api_mode: "chat_completions",
        env_vars: &["AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"],
        base_url: "",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "bedrock",
        aliases: &["aws", "aws-bedrock", "amazon-bedrock", "amazon"],
        api_mode: "bedrock_converse",
        env_vars: &[],
        base_url: "https://bedrock-runtime.us-east-1.amazonaws.com",
        auth_type: "aws_sdk",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "copilot-acp",
        aliases: &["github-copilot-acp", "copilot-acp-agent"],
        api_mode: "chat_completions",
        env_vars: &[],
        base_url: "acp://copilot",
        auth_type: "external_process",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "copilot",
        aliases: &["github-copilot", "github-models", "github-model", "github"],
        api_mode: "chat_completions",
        env_vars: &["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"],
        base_url: "https://api.githubcopilot.com",
        auth_type: "copilot",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "deepseek",
        aliases: &["deepseek-chat"],
        api_mode: "chat_completions",
        env_vars: &["DEEPSEEK_API_KEY"],
        base_url: "https://api.deepseek.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "gemini",
        aliases: &["google", "google-gemini", "google-ai-studio"],
        api_mode: "chat_completions",
        env_vars: &["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "google-gemini-cli",
        aliases: &["gemini-cli", "gemini-oauth"],
        api_mode: "chat_completions",
        env_vars: &[],
        base_url: "cloudcode-pa://google",
        auth_type: "oauth_external",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "gmi",
        aliases: &["gmi-cloud", "gmicloud"],
        api_mode: "chat_completions",
        env_vars: &["GMI_API_KEY", "GMI_BASE_URL"],
        base_url: "https://api.gmi-serving.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "huggingface",
        aliases: &["hf", "hugging-face", "huggingface-hub"],
        api_mode: "chat_completions",
        env_vars: &["HF_TOKEN"],
        base_url: "https://router.huggingface.co/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "kilocode",
        aliases: &["kilo-code", "kilo", "kilo-gateway"],
        api_mode: "chat_completions",
        env_vars: &["KILOCODE_API_KEY"],
        base_url: "https://api.kilo.ai/api/gateway",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "kimi-coding",
        aliases: &["kimi", "moonshot", "kimi-for-coding"],
        api_mode: "chat_completions",
        env_vars: &["KIMI_API_KEY", "KIMI_CODING_API_KEY"],
        base_url: "https://api.moonshot.ai/v1",
        auth_type: "api_key",
        default_headers: KIMI_HEADERS,
    },
    ProviderProfile {
        name: "kimi-coding-cn",
        aliases: &["kimi-cn", "moonshot-cn"],
        api_mode: "chat_completions",
        env_vars: &["KIMI_CN_API_KEY"],
        base_url: "https://api.moonshot.cn/v1",
        auth_type: "api_key",
        default_headers: KIMI_HEADERS,
    },
    ProviderProfile {
        name: "minimax",
        aliases: &["mini-max"],
        api_mode: "anthropic_messages",
        env_vars: &["MINIMAX_API_KEY"],
        base_url: "https://api.minimax.io/anthropic",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "minimax-cn",
        aliases: &["minimax-china", "minimax_cn"],
        api_mode: "anthropic_messages",
        env_vars: &["MINIMAX_CN_API_KEY"],
        base_url: "https://api.minimaxi.com/anthropic",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "minimax-oauth",
        aliases: &["minimax_oauth", "minimax-oauth-io"],
        api_mode: "anthropic_messages",
        env_vars: &[],
        base_url: "https://api.minimax.io/anthropic",
        auth_type: "oauth_external",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "nous",
        aliases: &["nous-portal", "nousresearch"],
        api_mode: "chat_completions",
        env_vars: &["NOUS_API_KEY", "NOUS_INFERENCE_BASE_URL"],
        base_url: "https://inference.nousresearch.com/v1",
        auth_type: "oauth_device_code",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "nvidia",
        aliases: &["nvidia-nim"],
        api_mode: "chat_completions",
        env_vars: &["NVIDIA_API_KEY"],
        base_url: "https://integrate.api.nvidia.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "ollama-cloud",
        aliases: &["ollama_cloud"],
        api_mode: "chat_completions",
        env_vars: &["OLLAMA_API_KEY"],
        base_url: "https://ollama.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "openai-codex",
        aliases: &["codex", "openai_codex"],
        api_mode: "codex_responses",
        env_vars: &[],
        base_url: "https://chatgpt.com/backend-api/codex",
        auth_type: "oauth_external",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "opencode-zen",
        aliases: &["opencode", "opencode_zen", "zen"],
        api_mode: "chat_completions",
        env_vars: &["OPENCODE_ZEN_API_KEY"],
        base_url: "https://opencode.ai/zen/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "opencode-go",
        aliases: &["opencode_go", "go", "opencode-go-sub"],
        api_mode: "chat_completions",
        env_vars: &["OPENCODE_GO_API_KEY"],
        base_url: "https://opencode.ai/zen/go/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "qwen-oauth",
        aliases: &["qwen", "qwen-portal", "qwen-cli"],
        api_mode: "chat_completions",
        env_vars: &["QWEN_API_KEY"],
        base_url: "https://portal.qwen.ai/v1",
        auth_type: "oauth_external",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "stepfun",
        aliases: &["step", "stepfun-coding-plan"],
        api_mode: "chat_completions",
        env_vars: &["STEPFUN_API_KEY"],
        base_url: "https://api.stepfun.ai/step_plan/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "xai",
        aliases: &["grok", "x-ai", "x.ai"],
        api_mode: "codex_responses",
        env_vars: &["XAI_API_KEY"],
        base_url: "https://api.x.ai/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "xiaomi",
        aliases: &["mimo", "xiaomi-mimo"],
        api_mode: "chat_completions",
        env_vars: &["XIAOMI_API_KEY"],
        base_url: "https://api.xiaomimimo.com/v1",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
    ProviderProfile {
        name: "zai",
        aliases: &["glm", "z-ai", "z.ai", "zhipu"],
        api_mode: "chat_completions",
        env_vars: &["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"],
        base_url: "https://api.z.ai/api/paas/v4",
        auth_type: "api_key",
        default_headers: EMPTY_HEADERS,
    },
];

const AUTO_PROVIDER_ORDER: &[&str] = &[
    "openrouter",
    "nous",
    "anthropic",
    "openai",
    "deepseek",
    "gemini",
    "gmi",
    "nvidia",
    "zai",
    "stepfun",
    "alibaba",
    "alibaba-coding-plan",
    "huggingface",
    "ollama-cloud",
    "xiaomi",
    "arcee",
    "ai-gateway",
    "kilocode",
    "kimi-coding",
    "kimi-coding-cn",
    "copilot",
    "minimax",
    "minimax-cn",
    "qwen-oauth",
    "opencode-zen",
    "opencode-go",
];

const VENDOR_PREFIXES: &[(&str, &str)] = &[
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
];

const AGGREGATOR_PROVIDERS: &[&str] = &["openrouter", "nous", "ai-gateway", "kilocode"];
const DOT_TO_HYPHEN_PROVIDERS: &[&str] = &["anthropic"];
const STRIP_VENDOR_ONLY_PROVIDERS: &[&str] = &["copilot", "copilot-acp", "openai-codex"];
const AUTHORITATIVE_NATIVE_PROVIDERS: &[&str] = &["gemini", "huggingface"];
const MATCHING_PREFIX_STRIP_PROVIDERS: &[&str] = &[
    "zai",
    "kimi-coding",
    "kimi-coding-cn",
    "minimax",
    "minimax-oauth",
    "minimax-cn",
    "alibaba",
    "qwen-oauth",
    "xiaomi",
    "arcee",
    "ollama-cloud",
    "custom",
];
const LOWERCASE_MODEL_PROVIDERS: &[&str] = &["xiaomi"];
const DEEPSEEK_CANONICAL_MODELS: &[&str] = &[
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
];
const DEEPSEEK_REASONER_KEYWORDS: &[&str] = &["reasoner", "r1", "think", "reasoning", "cot"];

pub fn list_provider_profiles() -> &'static [ProviderProfile] {
    PROVIDERS
}

pub fn get_provider_profile(name: &str) -> Option<&'static ProviderProfile> {
    let normalized = name.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    PROVIDERS
        .iter()
        .find(|profile| profile.matches(&normalized))
}

pub fn normalize_provider_alias(name: &str) -> String {
    let normalized = name.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return normalized;
    }
    get_provider_profile(&normalized)
        .map(|profile| profile.name.to_string())
        .unwrap_or(normalized)
}

pub fn auto_provider_candidates() -> impl Iterator<Item = &'static ProviderProfile> {
    AUTO_PROVIDER_ORDER
        .iter()
        .filter_map(|name| get_provider_profile(name))
}

pub fn infer_provider_from_base_url(base_url: &str) -> Option<&'static ProviderProfile> {
    let normalized = base_url.trim().trim_end_matches('/').to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    PROVIDERS
        .iter()
        .filter(|profile| !profile.base_url.is_empty())
        .find(|profile| {
            let profile_base = profile.base_url.trim_end_matches('/').to_ascii_lowercase();
            normalized == profile_base || normalized.starts_with(&(profile_base + "/"))
        })
}

pub fn infer_api_mode_from_base_url(base_url: &str) -> Option<&'static str> {
    let normalized = base_url.trim().trim_end_matches('/').to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized.contains("/anthropic") || normalized.contains("api.anthropic.com") {
        return Some("anthropic_messages");
    }
    if normalized.contains("/chat/completions")
        || normalized.ends_with("/v1")
        || normalized.ends_with("/v4")
        || normalized.ends_with("/v1beta")
    {
        return Some("chat_completions");
    }
    None
}

pub fn resolve_provider_api_mode(provider: &str, model_input: &str) -> Option<&'static str> {
    let normalized_provider = normalize_provider_alias(provider);
    match normalized_provider.as_str() {
        "copilot" => Some(copilot_model_api_mode(model_input)),
        "azure-foundry" => azure_foundry_model_api_mode(model_input),
        "opencode-zen" | "opencode-go" => {
            Some(opencode_model_api_mode(&normalized_provider, model_input))
        }
        _ => None,
    }
}

pub fn normalize_model_for_provider(model_input: &str, target_provider: &str) -> String {
    let name = model_input.trim();
    if name.is_empty() {
        return String::new();
    }

    let provider = normalize_provider_alias(target_provider);

    if AGGREGATOR_PROVIDERS.contains(&provider.as_str()) {
        return prepend_vendor(name);
    }

    if matches!(provider.as_str(), "opencode-zen" | "opencode-go") {
        let mut bare = strip_any_vendor_prefix(name).into_owned();
        if provider == "opencode-zen" && bare.to_ascii_lowercase().starts_with("claude-") {
            bare = dots_to_hyphens(&bare).into_owned();
        }
        return bare;
    }

    if DOT_TO_HYPHEN_PROVIDERS.contains(&provider.as_str()) {
        let bare = strip_matching_provider_prefix(name, &provider);
        if bare.contains('/') {
            return bare;
        }
        return dots_to_hyphens(&bare).into_owned();
    }

    if STRIP_VENDOR_ONLY_PROVIDERS.contains(&provider.as_str()) {
        let stripped = strip_matching_provider_prefix(name, &provider);
        if stripped == name && name.starts_with("openai/") {
            return name
                .split_once('/')
                .map(|(_, tail)| tail)
                .unwrap_or(name)
                .to_string();
        }
        return stripped;
    }

    if provider == "deepseek" {
        let bare = strip_matching_provider_prefix(name, &provider);
        if bare.contains('/') {
            return bare;
        }
        return normalize_for_deepseek(&bare);
    }

    if MATCHING_PREFIX_STRIP_PROVIDERS.contains(&provider.as_str()) {
        let mut result = strip_matching_provider_prefix(name, &provider);
        if LOWERCASE_MODEL_PROVIDERS.contains(&provider.as_str()) {
            result = result.to_ascii_lowercase();
        }
        return result;
    }

    if AUTHORITATIVE_NATIVE_PROVIDERS.contains(&provider.as_str()) {
        return name.to_string();
    }

    name.to_string()
}

fn normalize_for_deepseek(model_name: &str) -> String {
    let bare = strip_any_vendor_prefix(model_name).to_ascii_lowercase();
    if DEEPSEEK_CANONICAL_MODELS.contains(&bare.as_str()) {
        return bare;
    }
    if deepseek_v_series(&bare) {
        return bare;
    }
    if DEEPSEEK_REASONER_KEYWORDS
        .iter()
        .any(|keyword| bare.contains(keyword))
    {
        return "deepseek-reasoner".to_string();
    }
    "deepseek-chat".to_string()
}

fn copilot_model_api_mode(model_input: &str) -> &'static str {
    if should_use_copilot_responses_api(model_input) {
        "codex_responses"
    } else {
        "chat_completions"
    }
}

fn should_use_copilot_responses_api(model_input: &str) -> bool {
    let normalized = normalize_model_for_provider(model_input, "copilot").to_ascii_lowercase();
    let Some(rest) = normalized.strip_prefix("gpt-") else {
        return false;
    };
    let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count == 0 {
        return false;
    }
    let Ok(major) = rest[..digit_count].parse::<u32>() else {
        return false;
    };
    major >= 5 && !normalized.starts_with("gpt-5-mini")
}

fn azure_foundry_model_api_mode(model_input: &str) -> Option<&'static str> {
    let mut normalized = model_input.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if let Some((_, tail)) = normalized.rsplit_once('/') {
        normalized = tail.trim().to_string();
    }
    AZURE_FOUNDRY_RESPONSES_PREFIXES
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
        .then_some("codex_responses")
}

fn opencode_model_api_mode(provider: &str, model_input: &str) -> &'static str {
    let normalized_provider = normalize_provider_alias(provider);
    let normalized_model =
        normalize_model_for_provider(model_input, &normalized_provider).to_ascii_lowercase();
    if normalized_model.is_empty() {
        return "chat_completions";
    }

    match normalized_provider.as_str() {
        "opencode-go" => {
            if normalized_model.starts_with("minimax-") {
                "anthropic_messages"
            } else {
                "chat_completions"
            }
        }
        "opencode-zen" => {
            if normalized_model.starts_with("claude-") {
                "anthropic_messages"
            } else if normalized_model.starts_with("gpt-") {
                "codex_responses"
            } else {
                "chat_completions"
            }
        }
        _ => "chat_completions",
    }
}

fn deepseek_v_series(model_name: &str) -> bool {
    let Some(rest) = model_name.strip_prefix("deepseek-v") else {
        return false;
    };
    let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count == 0 {
        return false;
    }
    rest.chars()
        .nth(digit_count)
        .is_none_or(|ch| matches!(ch, '-' | '.'))
}

fn strip_any_vendor_prefix(model_name: &str) -> Cow<'_, str> {
    match model_name.split_once('/') {
        Some((_prefix, remainder)) if !remainder.trim().is_empty() => {
            Cow::Owned(remainder.trim().to_string())
        }
        _ => Cow::Borrowed(model_name),
    }
}

fn strip_matching_provider_prefix(model_name: &str, target_provider: &str) -> String {
    let Some((prefix, remainder)) = model_name.split_once('/') else {
        return model_name.to_string();
    };
    if prefix.trim().is_empty() || remainder.trim().is_empty() {
        return model_name.to_string();
    }
    let normalized_prefix = normalize_provider_alias(prefix);
    let normalized_target = normalize_provider_alias(target_provider);
    if normalized_prefix == normalized_target {
        remainder.trim().to_string()
    } else {
        model_name.to_string()
    }
}

fn dots_to_hyphens(model_name: &str) -> Cow<'_, str> {
    if model_name.contains('.') {
        Cow::Owned(model_name.replace('.', "-"))
    } else {
        Cow::Borrowed(model_name)
    }
}

fn prepend_vendor(model_name: &str) -> String {
    if model_name.contains('/') {
        return model_name.to_string();
    }
    match detect_vendor(model_name) {
        Some(vendor) => format!("{vendor}/{model_name}"),
        None => model_name.to_string(),
    }
}

fn detect_vendor(model_name: &str) -> Option<String> {
    let name = model_name.trim();
    if name.is_empty() {
        return None;
    }
    if let Some((prefix, _)) = name.split_once('/') {
        let normalized = prefix.trim().to_ascii_lowercase();
        return (!normalized.is_empty()).then_some(normalized);
    }

    let lower = name.to_ascii_lowercase();
    let first_token = lower.split('-').next().unwrap_or_default();
    if let Some((_, vendor)) = VENDOR_PREFIXES
        .iter()
        .find(|(prefix, _)| *prefix == first_token)
    {
        return Some((*vendor).to_string());
    }
    VENDOR_PREFIXES
        .iter()
        .find(|(prefix, _)| lower.starts_with(*prefix))
        .map(|(_, vendor)| (*vendor).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_aliases_and_auto_order_include_expected_profiles() {
        assert_eq!(normalize_provider_alias("claude"), "anthropic");
        assert_eq!(normalize_provider_alias("or"), "openrouter");
        let candidates = auto_provider_candidates()
            .map(|profile| profile.name)
            .collect::<Vec<_>>();
        assert!(candidates.starts_with(&["openrouter", "nous", "anthropic"]));
    }

    #[test]
    fn model_normalization_matches_python_rules_for_key_cases() {
        assert_eq!(
            normalize_model_for_provider("claude-sonnet-4.6", "openrouter"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            normalize_model_for_provider("openai/gpt-5.4", "openai-codex"),
            "gpt-5.4"
        );
        assert_eq!(
            normalize_model_for_provider("deepseek-r1", "deepseek"),
            "deepseek-reasoner"
        );
        assert_eq!(
            normalize_model_for_provider("MiMo-V2.5-Pro", "xiaomi"),
            "mimo-v2.5-pro"
        );
    }

    #[test]
    fn infer_provider_and_mode_from_base_url_covers_primary_cases() {
        assert_eq!(
            infer_provider_from_base_url("https://api.minimax.io/anthropic/v1/messages")
                .map(|profile| profile.name),
            Some("minimax")
        );
        assert_eq!(
            infer_api_mode_from_base_url("https://api.anthropic.com"),
            Some("anthropic_messages")
        );
        assert_eq!(
            infer_api_mode_from_base_url("https://api.deepseek.com/v1"),
            Some("chat_completions")
        );
    }

    #[test]
    fn provider_specific_api_mode_routing_matches_python_rules() {
        assert_eq!(
            resolve_provider_api_mode("copilot", "openai/gpt-5.4"),
            Some("codex_responses")
        );
        assert_eq!(
            resolve_provider_api_mode("copilot", "gpt-5-mini"),
            Some("chat_completions")
        );
        assert_eq!(
            resolve_provider_api_mode("azure-foundry", "openai/gpt-5.4"),
            Some("codex_responses")
        );
        assert_eq!(resolve_provider_api_mode("azure-foundry", "gpt-4o"), None);
        assert_eq!(
            resolve_provider_api_mode("opencode-zen", "opencode-zen/gpt-5.4"),
            Some("codex_responses")
        );
        assert_eq!(
            resolve_provider_api_mode("opencode-zen", "claude-sonnet-4.6"),
            Some("anthropic_messages")
        );
        assert_eq!(
            resolve_provider_api_mode("opencode-go", "opencode-go/minimax-m2.5"),
            Some("anthropic_messages")
        );
        assert_eq!(
            resolve_provider_api_mode("opencode-go", "glm-5.1"),
            Some("chat_completions")
        );
    }
}
