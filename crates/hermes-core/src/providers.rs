use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use regex::Regex;

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
const COPILOT_HEADERS: &[(&str, &str)] = &[
    ("Editor-Version", "vscode/1.104.1"),
    ("User-Agent", "HermesAgent/1.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
    ("Openai-Intent", "conversation-edits"),
    ("x-initiator", "agent"),
];
const KIMI_HEADERS: &[(&str, &str)] = &[("User-Agent", "hermes-agent/1.0")];
const AZURE_FOUNDRY_RESPONSES_PREFIXES: &[&str] = &["codex", "gpt-5", "o1", "o3", "o4"];

const BUILTIN_PROVIDERS: &[ProviderProfile] = &[
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
        default_headers: COPILOT_HEADERS,
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

#[derive(Debug)]
struct ProviderRegistryCache {
    profiles: &'static [ProviderProfile],
    aliases: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Default)]
struct ParsedProviderProfile {
    name: String,
    aliases: Option<Vec<String>>,
    api_mode: Option<String>,
    env_vars: Option<Vec<String>>,
    base_url: Option<String>,
    auth_type: Option<String>,
    default_headers: Option<Vec<(String, String)>>,
}

static PROVIDER_REGISTRY_CACHE: OnceLock<Mutex<Option<&'static ProviderRegistryCache>>> =
    OnceLock::new();

pub fn list_provider_profiles() -> &'static [ProviderProfile] {
    provider_registry_cache().profiles
}

pub fn get_provider_profile(name: &str) -> Option<&'static ProviderProfile> {
    let normalized = name.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    let cache = provider_registry_cache();
    cache
        .aliases
        .get(&normalized)
        .and_then(|index| cache.profiles.get(*index))
}

fn provider_registry_cache() -> &'static ProviderRegistryCache {
    let cache = PROVIDER_REGISTRY_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("provider registry cache poisoned");
    if let Some(existing) = *guard {
        return existing;
    }
    let built = Box::leak(Box::new(build_provider_registry_cache()));
    *guard = Some(built);
    built
}

fn build_provider_registry_cache() -> ProviderRegistryCache {
    let mut profiles = BUILTIN_PROVIDERS.to_vec();

    for path in discover_provider_plugin_files(bundled_model_providers_root()) {
        for profile in parse_provider_profiles_from_file(&path) {
            register_parsed_provider_override(&mut profiles, profile);
        }
    }

    for path in discover_provider_plugin_files(user_model_providers_root()) {
        for profile in parse_provider_profiles_from_file(&path) {
            register_parsed_provider_override(&mut profiles, profile);
        }
    }

    for path in discover_legacy_provider_files(legacy_providers_root()) {
        for profile in parse_provider_profiles_from_file(&path) {
            register_parsed_provider_override(&mut profiles, profile);
        }
    }

    let profiles = Box::leak(profiles.into_boxed_slice());
    let mut aliases = BTreeMap::new();
    for (index, profile) in profiles.iter().enumerate() {
        aliases.insert(profile.name.to_string(), index);
        for alias in profile.aliases {
            aliases.insert((*alias).to_string(), index);
        }
    }

    ProviderRegistryCache { profiles, aliases }
}

fn register_provider_override(profiles: &mut Vec<ProviderProfile>, profile: ProviderProfile) {
    if let Some(existing) = profiles.iter_mut().find(|entry| entry.name == profile.name) {
        *existing = profile;
    } else {
        profiles.push(profile);
    }
}

fn register_parsed_provider_override(
    profiles: &mut Vec<ProviderProfile>,
    parsed: ParsedProviderProfile,
) {
    let existing = profiles
        .iter()
        .find(|entry| entry.name == parsed.name)
        .cloned();
    let merged = ProviderProfile {
        name: leak_string(parsed.name),
        aliases: leak_string_slice(
            parsed
                .aliases
                .or_else(|| {
                    existing.as_ref().map(|profile| {
                        profile
                            .aliases
                            .iter()
                            .map(|value| (*value).to_string())
                            .collect::<Vec<_>>()
                    })
                })
                .unwrap_or_default(),
        ),
        api_mode: leak_string(
            parsed
                .api_mode
                .or_else(|| {
                    existing
                        .as_ref()
                        .map(|profile| profile.api_mode.to_string())
                })
                .unwrap_or_else(|| "chat_completions".to_string()),
        ),
        env_vars: leak_string_slice(
            parsed
                .env_vars
                .or_else(|| {
                    existing.as_ref().map(|profile| {
                        profile
                            .env_vars
                            .iter()
                            .map(|value| (*value).to_string())
                            .collect::<Vec<_>>()
                    })
                })
                .unwrap_or_default(),
        ),
        base_url: leak_string(
            parsed
                .base_url
                .or_else(|| {
                    existing
                        .as_ref()
                        .map(|profile| profile.base_url.to_string())
                })
                .unwrap_or_default(),
        ),
        auth_type: leak_string(
            parsed
                .auth_type
                .or_else(|| {
                    existing
                        .as_ref()
                        .map(|profile| profile.auth_type.to_string())
                })
                .unwrap_or_else(|| "api_key".to_string()),
        ),
        default_headers: leak_header_slice(
            parsed
                .default_headers
                .or_else(|| {
                    existing.as_ref().map(|profile| {
                        profile
                            .default_headers
                            .iter()
                            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                            .collect::<Vec<_>>()
                    })
                })
                .unwrap_or_default(),
        ),
    };
    register_provider_override(profiles, merged);
}

fn discover_provider_plugin_files(root: PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !path.is_dir() || name.starts_with(['.', '_']) {
            continue;
        }
        let init = path.join("__init__.py");
        if init.is_file() {
            files.push(init);
        }
    }
    files.sort();
    files
}

fn discover_legacy_provider_files(root: PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !path.is_file()
            || !name.ends_with(".py")
            || matches!(name.as_str(), "__init__.py" | "base.py")
            || name.starts_with('_')
        {
            continue;
        }
        files.push(path);
    }
    files.sort();
    files
}

fn bundled_model_providers_root() -> PathBuf {
    let root = std::env::var_os("HERMES_BUNDLED_PLUGINS")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("plugins"));
    if root
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == "model-providers")
    {
        root
    } else {
        root.join("model-providers")
    }
}

fn user_model_providers_root() -> PathBuf {
    std::env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
        .join("plugins")
        .join("model-providers")
}

fn legacy_providers_root() -> PathBuf {
    repo_root().join("providers")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn parse_provider_profiles_from_file(path: &Path) -> Vec<ParsedProviderProfile> {
    fs::read_to_string(path)
        .ok()
        .map(|source| parse_provider_profiles_from_source(&source))
        .unwrap_or_default()
}

fn parse_provider_profiles_from_source(source: &str) -> Vec<ParsedProviderProfile> {
    let mut assignments = BTreeMap::new();
    for captures in provider_assignment_regex().captures_iter(source) {
        let Some(name) = captures.get(1).map(|value| value.as_str().to_string()) else {
            continue;
        };
        let Some(matched) = captures.get(0) else {
            continue;
        };
        let open_index = matched.end().saturating_sub(1);
        let Some(close_index) = find_matching_delimiter(source, open_index, '(', ')') else {
            continue;
        };
        let body = &source[open_index + 1..close_index];
        let Some(profile) = parse_provider_constructor(body) else {
            continue;
        };
        assignments.insert(name, profile);
    }

    let mut profiles = Vec::new();
    for captures in register_provider_regex().captures_iter(source) {
        let Some(name) = captures.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if let Some(profile) = assignments.get(name) {
            profiles.push(profile.clone());
        }
    }
    profiles
}

fn parse_provider_constructor(body: &str) -> Option<ParsedProviderProfile> {
    let args = parse_keyword_args(body);
    Some(ParsedProviderProfile {
        name: parse_python_string(args.get("name")?)?,
        aliases: args
            .get("aliases")
            .and_then(|value| parse_python_string_list(value)),
        api_mode: args
            .get("api_mode")
            .and_then(|value| parse_python_string(value)),
        env_vars: args
            .get("env_vars")
            .and_then(|value| parse_python_string_list(value)),
        base_url: args
            .get("base_url")
            .and_then(|value| parse_python_string(value)),
        auth_type: args
            .get("auth_type")
            .and_then(|value| parse_python_string(value)),
        default_headers: args
            .get("default_headers")
            .and_then(|value| parse_python_string_dict(value)),
    })
}

fn parse_keyword_args(body: &str) -> BTreeMap<String, String> {
    let mut args = BTreeMap::new();
    for item in split_top_level_items(body, ',') {
        let item = strip_python_comment(&item);
        if item.trim().is_empty() {
            continue;
        }
        let Some((key, value)) = split_top_level_pair(&item, '=') else {
            continue;
        };
        args.insert(key.trim().to_string(), value.trim().to_string());
    }
    args
}

fn find_matching_delimiter(
    source: &str,
    open_index: usize,
    open_char: char,
    close_char: char,
) -> Option<usize> {
    let mut stack = vec![open_char];
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for (offset, ch) in source[open_index + 1..].char_indices() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => comment = true,
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' => {
                if stack.pop() != Some('(') {
                    return None;
                }
                if stack.is_empty() && close_char == ')' {
                    return Some(open_index + 1 + offset);
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return None;
                }
                if stack.is_empty() && close_char == ']' {
                    return Some(open_index + 1 + offset);
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return None;
                }
                if stack.is_empty() && close_char == '}' {
                    return Some(open_index + 1 + offset);
                }
            }
            _ => {}
        }
    }

    None
}

fn split_top_level_items(body: &str, separator: char) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for ch in body.chars() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            current.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => {
                comment = true;
            }
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' | '{' => {
                stack.push(ch);
                current.push(ch);
            }
            ')' | ']' | '}' => {
                stack.pop();
                current.push(ch);
            }
            _ if ch == separator && stack.is_empty() => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    items.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        items.push(trimmed.to_string());
    }
    items
}

fn split_top_level_pair(text: &str, separator: char) -> Option<(String, String)> {
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for (index, ch) in text.char_indices() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => comment = true,
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' | ']' | '}' => {
                stack.pop();
            }
            _ if ch == separator && stack.is_empty() => {
                return Some((
                    text[..index].trim().to_string(),
                    text[index + ch.len_utf8()..].trim().to_string(),
                ));
            }
            _ => {}
        }
    }

    None
}

fn strip_python_comment(text: &str) -> String {
    let mut result = String::new();
    let mut quote = None;
    let mut escaped = false;

    for ch in text.chars() {
        if let Some(active) = quote {
            result.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                result.push(ch);
            }
            '#' => break,
            _ => result.push(ch),
        }
    }

    result.trim().to_string()
}

fn parse_python_string(text: &str) -> Option<String> {
    let trimmed = strip_python_comment(text);
    let bytes = trimmed.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = *bytes.first()?;
    if !matches!(quote, b'\'' | b'"') || bytes.last().copied()? != quote {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    Some(
        inner
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\"),
    )
}

fn parse_python_string_list(text: &str) -> Option<Vec<String>> {
    let trimmed = strip_python_comment(text);
    let bytes = trimmed.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let (open, close) = (bytes[0] as char, *bytes.last()? as char);
    if !matches!((open, close), ('(', ')') | ('[', ']')) {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut values = Vec::new();
    for item in split_top_level_items(inner, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        values.push(parse_python_string(item)?);
    }
    Some(values)
}

fn parse_python_string_dict(text: &str) -> Option<Vec<(String, String)>> {
    let trimmed = strip_python_comment(text);
    let bytes = trimmed.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'{' || *bytes.last()? != b'}' {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut values = Vec::new();
    for item in split_top_level_items(inner, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (key, value) = split_top_level_pair(item, ':')?;
        values.push((parse_python_string(&key)?, parse_python_string(&value)?));
    }
    Some(values)
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn leak_string_slice(values: Vec<String>) -> &'static [&'static str] {
    if values.is_empty() {
        return &[];
    }
    let leaked = values.into_iter().map(leak_string).collect::<Vec<_>>();
    Box::leak(leaked.into_boxed_slice())
}

fn leak_header_slice(values: Vec<(String, String)>) -> &'static [(&'static str, &'static str)] {
    if values.is_empty() {
        return &[];
    }
    let leaked = values
        .into_iter()
        .map(|(key, value)| (leak_string(key), leak_string(value)))
        .collect::<Vec<_>>();
    Box::leak(leaked.into_boxed_slice())
}

fn provider_assignment_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\(")
            .expect("provider assignment regex")
    })
}

fn register_provider_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?m)^register_provider\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
            .expect("provider registration regex")
    })
}

#[cfg(test)]
fn reset_provider_registry_for_tests() {
    if let Some(cache) = PROVIDER_REGISTRY_CACHE.get() {
        let mut guard = cache.lock().expect("provider registry cache poisoned");
        *guard = None;
    }
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
    list_provider_profiles()
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

    if provider == "copilot" {
        return normalize_for_copilot(name);
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
    let normalized = normalize_for_copilot(model_input).to_ascii_lowercase();
    if should_use_copilot_responses_api(&normalized) {
        "codex_responses"
    } else if normalized.starts_with("claude-") {
        "anthropic_messages"
    } else {
        "chat_completions"
    }
}

fn should_use_copilot_responses_api(model_input: &str) -> bool {
    let normalized = normalize_for_copilot(model_input).to_ascii_lowercase();
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

fn normalize_for_copilot(model_name: &str) -> String {
    let raw = model_name.trim();
    if raw.is_empty() {
        return String::new();
    }

    if let Some(alias) = copilot_model_alias(raw) {
        return alias.to_string();
    }

    let mut candidates = Vec::with_capacity(4);
    candidates.push(raw.to_string());
    if let Some((_, tail)) = raw.split_once('/') {
        let trimmed = tail.trim();
        if !trimmed.is_empty() {
            candidates.push(trimmed.to_string());
        }
    }
    for suffix in ["-mini", "-nano", "-chat"] {
        if let Some(stripped) = raw.strip_suffix(suffix)
            && !stripped.trim().is_empty()
        {
            candidates.push(stripped.trim().to_string());
        }
    }

    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.clone()) {
            continue;
        }
        if let Some(alias) = copilot_model_alias(&candidate) {
            return alias.to_string();
        }
    }

    raw.split_once('/')
        .map(|(_, tail)| tail.trim().to_string())
        .filter(|tail| !tail.is_empty())
        .unwrap_or_else(|| raw.to_string())
}

fn copilot_model_alias(model_name: &str) -> Option<&'static str> {
    match model_name {
        "openai/gpt-5" | "openai/gpt-5-chat" | "openai/gpt-5-mini" | "openai/gpt-5-nano" => {
            Some("gpt-5-mini")
        }
        "openai/gpt-4.1" | "openai/gpt-4.1-mini" | "openai/gpt-4.1-nano" => Some("gpt-4.1"),
        "openai/gpt-4o" => Some("gpt-4o"),
        "openai/gpt-4o-mini" => Some("gpt-4o-mini"),
        "openai/o1" | "openai/o1-preview" => Some("gpt-5.2"),
        "openai/o1-mini" | "openai/o3-mini" | "openai/o4-mini" => Some("gpt-5-mini"),
        "openai/o3" => Some("gpt-5.3-codex"),
        "anthropic/claude-opus-4.6" | "anthropic/claude-opus-4-6" | "claude-opus-4-6" => {
            Some("claude-opus-4.6")
        }
        "anthropic/claude-sonnet-4.6" | "anthropic/claude-sonnet-4-6" | "claude-sonnet-4-6" => {
            Some("claude-sonnet-4.6")
        }
        "anthropic/claude-sonnet-4" | "anthropic/claude-sonnet-4-0" | "claude-sonnet-4-0" => {
            Some("claude-sonnet-4")
        }
        "anthropic/claude-sonnet-4.5" | "anthropic/claude-sonnet-4-5" | "claude-sonnet-4-5" => {
            Some("claude-sonnet-4.5")
        }
        "anthropic/claude-haiku-4.5" | "anthropic/claude-haiku-4-5" | "claude-haiku-4-5" => {
            Some("claude-haiku-4.5")
        }
        _ => None,
    }
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
    use std::fs;

    use tempfile::TempDir;

    fn set_optional_env(name: &str, value: Option<&Path>) -> Option<std::ffi::OsString> {
        let previous = std::env::var_os(name);
        match value {
            Some(path) => unsafe { std::env::set_var(name, path) },
            None => unsafe { std::env::remove_var(name) },
        }
        previous
    }

    fn restore_optional_env(name: &str, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }
    }

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
        assert_eq!(
            normalize_model_for_provider("anthropic/claude-sonnet-4-6", "copilot"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_provider("openai/o3", "copilot"),
            "gpt-5.3-codex"
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
            resolve_provider_api_mode("copilot", "anthropic/claude-sonnet-4.6"),
            Some("anthropic_messages")
        );
        assert_eq!(
            resolve_provider_api_mode("copilot", "claude-haiku-4-5"),
            Some("anthropic_messages")
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

    #[test]
    fn lazy_discovery_loads_multiple_profiles_from_plugin_source() {
        let _lock = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp
            .path()
            .join("bundled")
            .join("model-providers")
            .join("demo");
        fs::create_dir_all(&bundled).unwrap();
        fs::write(
            bundled.join("__init__.py"),
            r#"
alpha = ProviderProfile(
    name="alpha-provider",
    aliases=("alpha",),
    env_vars=("ALPHA_API_KEY",),
    base_url="https://alpha.example/v1",
)

beta = CustomProfile(
    name="beta-provider",
    aliases=("beta",),
    api_mode="anthropic_messages",
    env_vars=(),
    base_url="https://beta.example/v1",
    auth_type="oauth_external",
)

register_provider(alpha)
register_provider(beta)
"#,
        )
        .unwrap();

        let old_bundled =
            set_optional_env("HERMES_BUNDLED_PLUGINS", Some(&temp.path().join("bundled")));
        let old_home = set_optional_env("HERMES_HOME", Some(&temp.path().join(".hermes")));
        reset_provider_registry_for_tests();

        let names = list_provider_profiles()
            .iter()
            .map(|profile| profile.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"alpha-provider"));
        assert!(names.contains(&"beta-provider"));
        assert_eq!(
            get_provider_profile("beta").map(|profile| profile.auth_type),
            Some("oauth_external")
        );

        restore_optional_env("HERMES_BUNDLED_PLUGINS", old_bundled);
        restore_optional_env("HERMES_HOME", old_home);
        reset_provider_registry_for_tests();
    }

    #[test]
    fn user_provider_override_wins_over_bundled_and_builtin_profiles() {
        let _lock = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join(".hermes");
        let user_plugin = hermes_home
            .join("plugins")
            .join("model-providers")
            .join("gmi");
        fs::create_dir_all(&user_plugin).unwrap();
        fs::write(
            user_plugin.join("__init__.py"),
            r#"
override = ProviderProfile(
    name="gmi",
    aliases=("gmi-user-override",),
    env_vars=("GMI_API_KEY",),
    base_url="https://override.example.com/v1",
    auth_type="api_key",
)

register_provider(override)
"#,
        )
        .unwrap();

        let old_bundled =
            set_optional_env("HERMES_BUNDLED_PLUGINS", Some(&repo_root().join("plugins")));
        let old_home = set_optional_env("HERMES_HOME", Some(&hermes_home));
        reset_provider_registry_for_tests();

        let profile = get_provider_profile("gmi-user-override").unwrap();
        assert_eq!(profile.name, "gmi");
        assert_eq!(profile.base_url, "https://override.example.com/v1");

        restore_optional_env("HERMES_BUNDLED_PLUGINS", old_bundled);
        restore_optional_env("HERMES_HOME", old_home);
        reset_provider_registry_for_tests();
    }
}
