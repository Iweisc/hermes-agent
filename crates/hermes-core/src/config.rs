use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value};

use crate::{
    HermesContext, HermesError, anthropic_base_url_supports_oauth, anthropic_oauth_default_headers,
    anthropic_token_is_oauth, auto_provider_candidates, codex_cloudflare_headers,
    get_provider_profile, infer_api_mode_from_base_url, infer_provider_from_base_url,
    normalize_model_for_provider, normalize_provider_alias, resolve_anthropic_token,
    resolve_codex_access_token, resolve_copilot_acp_runtime_credentials,
    resolve_copilot_runtime_credentials, resolve_google_gemini_runtime_credentials,
    resolve_minimax_oauth_runtime_credentials, resolve_nous_runtime_credentials,
    resolve_provider_api_mode, resolve_qwen_runtime_credentials,
};

const DEFAULT_SOUL_MD: &str = "You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.";

const BOOTSTRAP_DIRS: [&str; 5] = ["cron", "sessions", "logs", "logs/curator", "memories"];

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub raw: Value,
    pub config: HermesConfig,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelOverrides {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRuntimeConfig {
    pub model: String,
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub api_mode: String,
    pub auth_type: String,
    pub default_headers: Vec<(String, String)>,
}

impl LoadedConfig {
    pub fn cfg_get<'a>(&'a self, keys: &[&str]) -> Option<&'a Value> {
        let mut node = &self.raw;
        for key in keys {
            match node {
                Value::Mapping(mapping) => {
                    node = mapping.get(Value::String((*key).to_string()))?;
                }
                _ => return None,
            }
        }
        Some(node)
    }

    pub fn configured_model_name(&self) -> Option<String> {
        match self.raw.get("model") {
            Some(Value::String(model)) => non_empty_string(model.clone()),
            Some(Value::Mapping(mapping)) => mapping_string(mapping, "default")
                .or_else(|| mapping_string(mapping, "model"))
                .or_else(|| mapping_string(mapping, "name")),
            _ => None,
        }
    }

    pub fn configured_model_provider(&self) -> Option<String> {
        self.raw
            .get("model")
            .and_then(Value::as_mapping)
            .and_then(|mapping| mapping_string(mapping, "provider"))
            .map(|value| value.to_ascii_lowercase())
    }

    pub fn configured_model_base_url(&self) -> Option<String> {
        self.raw
            .get("model")
            .and_then(Value::as_mapping)
            .and_then(|mapping| mapping_string(mapping, "base_url"))
    }

    pub fn configured_model_api_key(&self) -> Option<String> {
        self.raw
            .get("model")
            .and_then(Value::as_mapping)
            .and_then(|mapping| mapping_string(mapping, "api_key"))
    }

    pub fn configured_model_api_mode(&self) -> Option<String> {
        self.raw
            .get("model")
            .and_then(Value::as_mapping)
            .and_then(|mapping| mapping_string(mapping, "api_mode"))
            .map(|value| value.to_ascii_lowercase())
    }

    pub fn configured_model_context_length(&self) -> Option<u64> {
        self.raw
            .get("model")
            .and_then(Value::as_mapping)
            .and_then(|mapping| mapping_u64(mapping, "context_length"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermesConfig {
    #[serde(default = "default_model_value")]
    pub model: Value,
    #[serde(default)]
    pub fallback_providers: Vec<FallbackProviderConfig>,
    #[serde(default = "default_toolsets")]
    pub toolsets: Vec<String>,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub compression: CompressionConfig,
    #[serde(default)]
    pub delegation: DelegationConfig,
    #[serde(default)]
    pub terminal: TerminalConfig,
    #[serde(default)]
    pub display: DisplayConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

impl Default for HermesConfig {
    fn default() -> Self {
        Self {
            model: default_model_value(),
            fallback_providers: Vec::new(),
            toolsets: default_toolsets(),
            agent: AgentConfig::default(),
            compression: CompressionConfig::default(),
            delegation: DelegationConfig::default(),
            terminal: TerminalConfig::default(),
            display: DisplayConfig::default(),
            logging: LoggingConfig::default(),
            memory: MemoryConfig::default(),
            network: NetworkConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_agent_max_turns")]
    pub max_turns: u64,
    #[serde(default = "default_agent_api_max_retries")]
    pub api_max_retries: u64,
    #[serde(default = "default_gateway_timeout")]
    pub gateway_timeout: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: default_agent_max_turns(),
            api_max_retries: default_agent_api_max_retries(),
            gateway_timeout: default_gateway_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct FallbackProviderConfig {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub api_mode: String,
    #[serde(default)]
    pub key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_compression_threshold")]
    pub threshold: f64,
    #[serde(default = "default_compression_target_ratio")]
    pub target_ratio: f64,
    #[serde(default = "default_compression_protect_last_n")]
    pub protect_last_n: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            threshold: default_compression_threshold(),
            target_ratio: default_compression_target_ratio(),
            protect_last_n: default_compression_protect_last_n(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalConfig {
    #[serde(default = "default_terminal_backend")]
    pub backend: String,
    #[serde(default = "default_terminal_cwd")]
    pub cwd: String,
    #[serde(default = "default_terminal_timeout")]
    pub timeout: u64,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            backend: default_terminal_backend(),
            cwd: default_terminal_cwd(),
            timeout: default_terminal_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationConfig {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub api_mode: String,
    #[serde(default = "default_delegation_max_iterations")]
    pub max_iterations: u64,
    #[serde(default = "default_delegation_max_concurrent_children")]
    pub max_concurrent_children: u64,
    #[serde(default = "default_delegation_max_spawn_depth")]
    pub max_spawn_depth: u64,
    #[serde(default = "default_true")]
    pub orchestrator_enabled: bool,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            api_mode: String::new(),
            max_iterations: default_delegation_max_iterations(),
            max_concurrent_children: default_delegation_max_concurrent_children(),
            max_spawn_depth: default_delegation_max_spawn_depth(),
            orchestrator_enabled: default_true(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    #[serde(default = "default_false")]
    pub compact: bool,
    #[serde(default = "default_false")]
    pub streaming: bool,
    #[serde(default = "default_skin")]
    pub skin: String,
    #[serde(default = "default_language")]
    pub language: String,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            compact: default_false(),
            streaming: default_false(),
            skin: default_skin(),
            language: default_language(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_max_size_mb")]
    pub max_size_mb: u64,
    #[serde(default = "default_log_backup_count")]
    pub backup_count: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            max_size_mb: default_log_max_size_mb(),
            backup_count: default_log_backup_count(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_true")]
    pub memory_enabled: bool,
    #[serde(default = "default_true")]
    pub user_profile_enabled: bool,
    #[serde(default = "default_memory_char_limit")]
    pub memory_char_limit: u64,
    #[serde(default = "default_user_char_limit")]
    pub user_char_limit: u64,
    #[serde(default)]
    pub provider: String,
}

impl MemoryConfig {
    pub fn any_enabled(&self) -> bool {
        self.memory_enabled || self.user_profile_enabled
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_enabled: default_true(),
            user_profile_enabled: default_true(),
            memory_char_limit: default_memory_char_limit(),
            user_char_limit: default_user_char_limit(),
            provider: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_false")]
    pub force_ipv4: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            force_ipv4: default_false(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    #[serde(default = "default_false")]
    pub redact_secrets: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            redact_secrets: default_false(),
        }
    }
}

impl HermesContext {
    pub fn ensure_hermes_home(&self) -> Result<(), HermesError> {
        let home = self.hermes_home();
        create_dir_all(&home)?;
        for subdir in BOOTSTRAP_DIRS {
            create_dir_all(&home.join(subdir))?;
        }

        let soul_path = home.join("SOUL.md");
        if !soul_path.exists() {
            fs::write(&soul_path, DEFAULT_SOUL_MD).map_err(|source| HermesError::Io {
                action: "writing",
                path: soul_path,
                source,
            })?;
        }
        Ok(())
    }

    pub fn load_config_document(&self) -> Result<LoadedConfig, HermesError> {
        self.ensure_hermes_home()?;

        let path = self.config_path();
        let mut warnings = Vec::new();
        let mut merged =
            serde_yaml::to_value(HermesConfig::default()).expect("default config serialization");

        if path.exists() {
            match fs::read_to_string(&path) {
                Ok(contents) => match serde_yaml::from_str::<Value>(&contents) {
                    Ok(user_config) => {
                        let normalized = normalize_user_config(user_config);
                        deep_merge_value(&mut merged, normalized);
                    }
                    Err(error) => {
                        warnings.push(format!("Failed to parse {}: {error}", path.display()))
                    }
                },
                Err(error) => warnings.push(format!("Failed to read {}: {error}", path.display())),
            }
        }

        let expanded = expand_env_vars_value(merged);
        let config = match serde_yaml::from_value::<HermesConfig>(expanded.clone()) {
            Ok(config) => config,
            Err(error) => {
                warnings.push(format!(
                    "Failed to decode {} into Rust config: {error}",
                    path.display()
                ));
                HermesConfig::default()
            }
        };

        Ok(LoadedConfig {
            path,
            raw: expanded,
            config,
            warnings,
        })
    }

    pub fn resolve_model_runtime(
        &self,
        loaded: &LoadedConfig,
        overrides: &ModelOverrides,
    ) -> Result<ModelRuntimeConfig, HermesError> {
        let raw_model = overrides
            .model
            .clone()
            .and_then(non_empty_string)
            .or_else(|| {
                env::var("HERMES_INFERENCE_MODEL")
                    .ok()
                    .and_then(non_empty_string)
            })
            .or_else(|| loaded.configured_model_name())
            .ok_or_else(|| HermesError::State {
                action: "resolving model runtime",
                detail: "No model configured. Set model.default in config.yaml or pass --model."
                    .to_string(),
            })?;

        let requested_provider = overrides
            .provider
            .clone()
            .and_then(non_empty_string)
            .or_else(|| {
                env::var("HERMES_INFERENCE_PROVIDER")
                    .ok()
                    .and_then(non_empty_string)
            })
            .or_else(|| loaded.configured_model_provider())
            .unwrap_or_else(|| "auto".to_string())
            .to_ascii_lowercase();

        let config_base_url = loaded.configured_model_base_url();
        let config_api_key = loaded.configured_model_api_key();
        let config_api_mode = loaded.configured_model_api_mode();
        let explicit_base_url = overrides.base_url.clone().and_then(non_empty_string);
        let explicit_api_key = overrides.api_key.clone().and_then(non_empty_string);
        let requested_api_mode = overrides
            .api_mode
            .clone()
            .and_then(non_empty_string)
            .or(config_api_mode.clone())
            .map(|value| value.to_ascii_lowercase());
        let candidate_base_url = explicit_base_url
            .clone()
            .or_else(|| config_base_url.clone());

        let provider = match requested_provider.as_str() {
            "auto" => resolve_auto_provider(candidate_base_url.as_deref())?,
            other => normalize_provider_alias(other),
        };
        let profile = get_provider_profile(&provider).ok_or_else(|| HermesError::State {
            action: "resolving model runtime",
            detail: format!("Provider '{provider}' is not recognized by the Rust runtime."),
        })?;
        let model = normalize_model_for_provider(&raw_model, &provider);
        let copilot_acp = if provider == "copilot-acp" {
            Some(resolve_copilot_acp_runtime_credentials()?)
        } else {
            None
        };
        let minimax_oauth = if provider == "minimax-oauth" {
            Some(resolve_minimax_oauth_runtime_credentials(
                &self.hermes_home(),
            )?)
        } else {
            None
        };
        let google_gemini = if provider == "google-gemini-cli" {
            Some(resolve_google_gemini_runtime_credentials(
                &self.hermes_home(),
            )?)
        } else {
            None
        };
        let qwen_oauth = if provider == "qwen-oauth" {
            Some(resolve_qwen_runtime_credentials()?)
        } else {
            None
        };
        let copilot_runtime = if provider == "copilot" && explicit_api_key.is_none() {
            Some(resolve_copilot_runtime_credentials()?)
        } else {
            None
        };
        let nous_runtime = if provider == "nous"
            && explicit_api_key.is_none()
            && env::var("NOUS_API_KEY")
                .ok()
                .and_then(non_empty_string)
                .is_none()
        {
            Some(resolve_nous_runtime_credentials(
                &self.hermes_home(),
                env::var("HERMES_NOUS_MIN_KEY_TTL_SECONDS")
                    .ok()
                    .and_then(|value| value.trim().parse::<i64>().ok())
                    .unwrap_or(1800),
                env::var("HERMES_NOUS_TIMEOUT_SECONDS")
                    .ok()
                    .and_then(|value| value.trim().parse::<f64>().ok())
                    .unwrap_or(15.0),
            )?)
        } else {
            None
        };
        let anthropic_runtime_token =
            (provider == "anthropic" && explicit_api_key.is_none()).then(resolve_anthropic_token);

        let base_url = explicit_base_url
            .or(config_base_url)
            .or_else(|| copilot_acp.as_ref().map(|creds| creds.base_url.clone()))
            .or_else(|| minimax_oauth.as_ref().map(|creds| creds.base_url.clone()))
            .or_else(|| nous_runtime.as_ref().map(|creds| creds.base_url.clone()))
            .or_else(|| qwen_oauth.as_ref().map(|creds| creds.base_url.clone()))
            .or_else(|| {
                profile
                    .base_url_env_var()
                    .and_then(|name| env::var(name).ok())
                    .and_then(non_empty_string)
            })
            .or_else(|| {
                (provider == "nous")
                    .then(|| env::var("NOUS_INFERENCE_BASE_URL").ok().and_then(non_empty_string))
                    .flatten()
            })
            .or_else(|| non_empty_string(profile.base_url.to_string()))
            .ok_or_else(|| HermesError::State {
                action: "resolving model runtime",
                detail: format!(
                    "Provider '{provider}' requires a base URL. Set model.base_url or pass --base-url."
                ),
            })?;

        let mut api_key = explicit_api_key
            .or(config_api_key)
            .or_else(|| (provider == "copilot-acp").then(|| "copilot-acp".to_string()))
            .or_else(|| copilot_runtime.as_ref().map(|creds| creds.api_key.clone()))
            .or_else(|| {
                minimax_oauth
                    .as_ref()
                    .map(|creds| creds.access_token.clone())
            })
            .or_else(|| nous_runtime.as_ref().map(|creds| creds.api_key.clone()))
            .or_else(|| {
                google_gemini
                    .as_ref()
                    .map(|creds| creds.access_token.clone())
            })
            .or_else(|| anthropic_runtime_token.clone().flatten())
            .or_else(|| qwen_oauth.as_ref().map(|creds| creds.access_token.clone()))
            .or_else(|| {
                profile
                    .api_key_env_vars()
                    .find_map(|name| env::var(name).ok().and_then(non_empty_string))
            })
            .unwrap_or_default();
        if api_key.is_empty() && provider == "openai-codex" {
            api_key = resolve_codex_access_token(&self.hermes_home())?;
        }
        if api_key.is_empty() && provider != "custom" && profile.auth_type != "aws_sdk" {
            let env_hint = profile.api_key_env_vars().collect::<Vec<_>>().join(", ");
            let detail = if provider == "openai-codex" {
                "No Codex OAuth token resolved. Run `hermes auth add openai-codex`, or pass --api-key."
                    .to_string()
            } else {
                format!(
                    "No API key resolved for provider '{provider}'. Set {} or pass --api-key.",
                    if env_hint.is_empty() {
                        "a runtime credential".to_string()
                    } else {
                        env_hint
                    }
                )
            };
            return Err(HermesError::State {
                action: "resolving model runtime",
                detail,
            });
        }

        let inferred_api_mode = infer_api_mode_from_base_url(&base_url).map(ToOwned::to_owned);
        let provider_api_mode = resolve_provider_api_mode(&provider, &model).map(ToOwned::to_owned);
        let api_mode = requested_api_mode
            .or_else(|| {
                matches!(inferred_api_mode.as_deref(), Some("anthropic_messages"))
                    .then(|| "anthropic_messages".to_string())
            })
            .or(provider_api_mode)
            .or(inferred_api_mode)
            .unwrap_or_else(|| profile.api_mode.to_string())
            .to_ascii_lowercase();
        if !matches!(
            api_mode.as_str(),
            "chat_completions" | "anthropic_messages" | "codex_responses" | "bedrock_converse"
        ) {
            return Err(HermesError::State {
                action: "resolving model runtime",
                detail: format!("API mode '{api_mode}' is not ported in the Rust runtime yet."),
            });
        }

        let mut default_headers = profile
            .default_headers
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect::<Vec<_>>();
        let mut auth_type = profile.auth_type.to_string();
        if provider == "anthropic"
            && anthropic_base_url_supports_oauth(&base_url)
            && anthropic_token_is_oauth(&api_key)
        {
            auth_type = "oauth_external".to_string();
            default_headers.extend(anthropic_oauth_default_headers());
        }
        if provider == "openai-codex" {
            default_headers.extend(codex_cloudflare_headers(&api_key));
        }

        Ok(ModelRuntimeConfig {
            model,
            provider,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            api_mode,
            auth_type,
            default_headers,
        })
    }
}

fn resolve_auto_provider(base_url: Option<&str>) -> Result<String, HermesError> {
    if let Some(base_url) = base_url.and_then(non_empty_trimmed_ref) {
        if let Some(profile) = infer_provider_from_base_url(base_url) {
            return Ok(profile.name.to_string());
        }
        return Ok("custom".to_string());
    }

    for profile in auto_provider_candidates() {
        if profile
            .api_key_env_vars()
            .any(|name| env::var(name).ok().and_then(non_empty_string).is_some())
        {
            return Ok(profile.name.to_string());
        }
    }

    Err(HermesError::State {
        action: "resolving model runtime",
        detail: "No provider credentials found. Set a supported provider API key, pass --provider, or pass --base-url/--api-key.".to_string(),
    })
}

fn default_model_value() -> Value {
    Value::String(String::new())
}

fn default_toolsets() -> Vec<String> {
    vec![String::from("hermes-cli")]
}

fn default_delegation_max_iterations() -> u64 {
    50
}

fn default_delegation_max_concurrent_children() -> u64 {
    3
}

fn default_delegation_max_spawn_depth() -> u64 {
    1
}

fn default_agent_max_turns() -> u64 {
    90
}

fn default_agent_api_max_retries() -> u64 {
    3
}

fn default_gateway_timeout() -> u64 {
    1800
}

fn default_compression_threshold() -> f64 {
    0.5
}

fn default_compression_target_ratio() -> f64 {
    0.2
}

fn default_compression_protect_last_n() -> usize {
    20
}

fn default_terminal_backend() -> String {
    String::from("local")
}

fn default_terminal_cwd() -> String {
    String::from(".")
}

fn default_terminal_timeout() -> u64 {
    180
}

fn default_skin() -> String {
    String::from("default")
}

fn default_language() -> String {
    String::from("en")
}

fn default_log_level() -> String {
    String::from("INFO")
}

fn default_log_max_size_mb() -> u64 {
    5
}

fn default_log_backup_count() -> u64 {
    3
}

fn default_false() -> bool {
    false
}

fn default_true() -> bool {
    true
}

fn default_memory_char_limit() -> u64 {
    2200
}

fn default_user_char_limit() -> u64 {
    1375
}

fn non_empty_string(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn non_empty_trimmed_ref(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(Value::String(key.to_string()))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .and_then(non_empty_string)
}

fn mapping_u64(mapping: &Mapping, key: &str) -> Option<u64> {
    let value = mapping.get(Value::String(key.to_string()))?;
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn create_dir_all(path: &Path) -> Result<(), HermesError> {
    fs::create_dir_all(path).map_err(|source| HermesError::Io {
        action: "creating",
        path: path.to_path_buf(),
        source,
    })
}

fn normalize_user_config(value: Value) -> Value {
    let mut mapping = match value {
        Value::Mapping(mapping) => mapping,
        _ => return Value::Mapping(Mapping::new()),
    };

    migrate_max_turns(&mut mapping);
    migrate_root_model_keys(&mut mapping);

    Value::Mapping(mapping)
}

fn migrate_max_turns(root: &mut Mapping) {
    let Some(max_turns) = root.remove(Value::String(String::from("max_turns"))) else {
        return;
    };

    let agent_key = Value::String(String::from("agent"));
    let current = root
        .remove(&agent_key)
        .unwrap_or_else(|| Value::Mapping(Mapping::new()));
    let mut agent_mapping = match current {
        Value::Mapping(mapping) => mapping,
        _ => Mapping::new(),
    };

    let max_turns_key = Value::String(String::from("max_turns"));
    if !agent_mapping.contains_key(&max_turns_key) {
        agent_mapping.insert(max_turns_key, max_turns);
    }
    root.insert(agent_key, Value::Mapping(agent_mapping));
}

fn migrate_root_model_keys(root: &mut Mapping) {
    let provider = root.remove(Value::String(String::from("provider")));
    let base_url = root.remove(Value::String(String::from("base_url")));
    let context_length = root.remove(Value::String(String::from("context_length")));

    if provider.is_none() && base_url.is_none() && context_length.is_none() {
        return;
    }

    let model_key = Value::String(String::from("model"));
    let current = root
        .remove(&model_key)
        .unwrap_or_else(|| Value::Mapping(Mapping::new()));
    let mut model_mapping = match current {
        Value::Mapping(mapping) => mapping,
        Value::String(text) if !text.is_empty() => {
            let mut mapping = Mapping::new();
            mapping.insert(Value::String(String::from("default")), Value::String(text));
            mapping
        }
        _ => Mapping::new(),
    };

    insert_if_missing(&mut model_mapping, "provider", provider);
    insert_if_missing(&mut model_mapping, "base_url", base_url);
    insert_if_missing(&mut model_mapping, "context_length", context_length);
    root.insert(model_key, Value::Mapping(model_mapping));
}

fn insert_if_missing(mapping: &mut Mapping, key: &str, value: Option<Value>) {
    let Some(value) = value else {
        return;
    };
    if value_is_missing(&value) {
        return;
    }
    let key_value = Value::String(key.to_string());
    if !mapping.contains_key(&key_value) {
        mapping.insert(key_value, value);
    }
}

fn value_is_missing(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.trim().is_empty(),
        _ => false,
    }
}

fn deep_merge_value(base: &mut Value, override_value: Value) {
    match (base, override_value) {
        (Value::Mapping(base_map), Value::Mapping(override_map)) => {
            for (key, value) in override_map {
                if let Some(base_value) = base_map.get_mut(&key) {
                    deep_merge_value(base_value, value);
                } else {
                    base_map.insert(key, value);
                }
            }
        }
        (base_value, override_value) => *base_value = override_value,
    }
}

fn expand_env_vars_value(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(expand_env_vars_in_string(&text)),
        Value::Sequence(items) => {
            Value::Sequence(items.into_iter().map(expand_env_vars_value).collect())
        }
        Value::Mapping(mapping) => Value::Mapping(
            mapping
                .into_iter()
                .map(|(key, value)| (key, expand_env_vars_value(value)))
                .collect(),
        ),
        other => other,
    }
}

fn expand_env_vars_in_string(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if chars[index] == '$' && index + 1 < chars.len() && chars[index + 1] == '{' {
            let mut end = index + 2;
            while end < chars.len() && chars[end] != '}' {
                end += 1;
            }
            if end < chars.len() && chars[end] == '}' {
                let key: String = chars[index + 2..end].iter().collect();
                match env::var(&key) {
                    Ok(value) => output.push_str(&value),
                    Err(_) => {
                        output.push('$');
                        output.push('{');
                        output.push_str(&key);
                        output.push('}');
                    }
                }
                index = end + 1;
                continue;
            }
        }
        output.push(chars[index]);
        index += 1;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use tempfile::TempDir;

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().expect("tempdir");
        let home = temp.path().join("home");
        fs::create_dir_all(&home).expect("home dir");
        (temp, HermesContext::new(home))
    }

    fn jwt_with_claims(exp: i64, account_id: &str) -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"none","typ":"JWT"}"#);
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
            json!({
                "exp": exp,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            })
            .to_string(),
        );
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn ensure_hermes_home_creates_bootstrap_dirs_and_soul() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let home = ctx.hermes_home();
        for subdir in BOOTSTRAP_DIRS {
            assert!(home.join(subdir).is_dir(), "missing {subdir}");
        }
        assert_eq!(
            fs::read_to_string(home.join("SOUL.md")).expect("soul"),
            DEFAULT_SOUL_MD
        );
    }

    #[test]
    fn load_config_document_deep_merges_and_expands_env_vars() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let config_yaml = "logging:\n  level: DEBUG\ndisplay:\n  skin: ${HERMES_TEST_SKIN}\n";
        fs::write(ctx.config_path(), config_yaml).expect("write config");
        // SAFETY: test-only process env mutation before spawning threads.
        unsafe { env::set_var("HERMES_TEST_SKIN", "slate") };

        let loaded = ctx.load_config_document().expect("load config");
        assert_eq!(loaded.config.logging.level, "DEBUG");
        assert_eq!(loaded.config.logging.max_size_mb, 5);
        assert_eq!(loaded.config.display.skin, "slate");
        assert_eq!(
            loaded.cfg_get(&["display", "skin"]).expect("skin value"),
            &Value::String(String::from("slate"))
        );
        // SAFETY: test-only cleanup of the process env mutation above.
        unsafe { env::remove_var("HERMES_TEST_SKIN") };
    }

    #[test]
    fn load_config_document_normalizes_legacy_max_turns() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(ctx.config_path(), "max_turns: 42\n").expect("write config");

        let loaded = ctx.load_config_document().expect("load config");
        assert_eq!(loaded.config.agent.max_turns, 42);
        assert_eq!(
            loaded
                .cfg_get(&["agent", "max_turns"])
                .expect("agent.max_turns"),
            &Value::Number(42.into())
        );
    }

    #[test]
    fn resolve_model_runtime_reads_codex_auth_store_and_headers() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let token = jwt_with_claims(i64::MAX / 2, "acct-codex");
        fs::write(
            ctx.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": token,
                            "refresh_token": "refresh-test",
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: gpt-5.4\n  provider: openai-codex\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "openai-codex");
        assert_eq!(runtime.api_mode, "codex_responses");
        assert!(!runtime.api_key.is_empty());
        assert!(
            runtime
                .default_headers
                .iter()
                .any(|(name, value)| name == "originator" && value == "codex_cli_rs")
        );
        assert!(
            runtime
                .default_headers
                .iter()
                .any(|(name, value)| name == "ChatGPT-Account-ID" && value == "acct-codex")
        );
    }

    #[test]
    fn resolve_model_runtime_missing_codex_token_points_to_native_auth_add() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.config_path(),
            "model:\n  default: gpt-5.4\n  provider: openai-codex\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let error = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .unwrap_err();

        match error {
            HermesError::State { detail, .. } => {
                assert!(detail.contains("hermes auth add openai-codex"));
            }
            other => panic!("expected state error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_model_runtime_allows_bedrock_without_api_key() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.config_path(),
            "model:\n  default: anthropic.claude-sonnet-4-6-20250514-v1:0\n  provider: bedrock\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "bedrock");
        assert_eq!(runtime.api_mode, "bedrock_converse");
        assert_eq!(runtime.auth_type, "aws_sdk");
        assert!(runtime.api_key.is_empty());
    }

    #[test]
    fn resolve_model_runtime_applies_provider_specific_api_modes() {
        let cases = [
            (
                "copilot",
                "gpt-5.4",
                "https://api.githubcopilot.com",
                "gpt-5.4",
                "codex_responses",
            ),
            (
                "azure-foundry",
                "gpt-5.4",
                "https://example.openai.azure.com/v1",
                "gpt-5.4",
                "codex_responses",
            ),
            (
                "opencode-zen",
                "claude-sonnet-4.6",
                "https://opencode.ai/zen/v1",
                "claude-sonnet-4-6",
                "anthropic_messages",
            ),
        ];

        for (provider, model_name, base_url, expected_model, expected_api_mode) in cases {
            let (_temp, ctx) = test_context();
            ctx.ensure_hermes_home().expect("ensure home");
            fs::write(
                ctx.config_path(),
                format!(
                    "model:\n  default: {model_name}\n  provider: {provider}\n  base_url: {base_url}\n"
                ),
            )
            .expect("write config");

            let loaded = ctx.load_config_document().expect("load config");
            let runtime = ctx
                .resolve_model_runtime(
                    &loaded,
                    &ModelOverrides {
                        api_key: Some("test-key".to_string()),
                        ..ModelOverrides::default()
                    },
                )
                .expect("resolve runtime");

            assert_eq!(runtime.provider, provider);
            assert_eq!(runtime.model, expected_model);
            assert_eq!(runtime.api_mode, expected_api_mode);
        }
    }

    #[test]
    fn resolve_model_runtime_keeps_anthropic_url_inference_ahead_of_model_routing() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.config_path(),
            "model:\n  default: gpt-5.4\n  provider: azure-foundry\n  base_url: https://example.azure.com/anthropic/v1\n",
        )
        .expect("write config");

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(
                &loaded,
                &ModelOverrides {
                    api_key: Some("test-key".to_string()),
                    ..ModelOverrides::default()
                },
            )
            .expect("resolve runtime");

        assert_eq!(runtime.api_mode, "anthropic_messages");
    }

    #[test]
    fn resolve_model_runtime_reads_minimax_oauth_auth_store() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "minimax-oauth": {
                        "access_token": "mini-runtime-token",
                        "refresh_token": "mini-refresh-token",
                        "portal_base_url": "https://api.minimax.io",
                        "inference_base_url": "https://api.minimaxi.com/anthropic",
                        "client_id": "mini-client",
                        "expires_at": "2999-01-01T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: MiniMax-M2.7-highspeed\n  provider: minimax-oauth\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "minimax-oauth");
        assert_eq!(runtime.api_mode, "anthropic_messages");
        assert_eq!(runtime.api_key, "mini-runtime-token");
        assert_eq!(runtime.base_url, "https://api.minimaxi.com/anthropic");
    }

    #[test]
    fn resolve_model_runtime_reads_google_gemini_oauth_credentials() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::create_dir_all(ctx.hermes_home().join("auth")).expect("auth dir");
        fs::write(
            ctx.hermes_home().join("auth").join("google_oauth.json"),
            json!({
                "refresh": "google-refresh|proj-123|managed-123",
                "access": "google-runtime-token",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: gemini-2.5-pro\n  provider: google-gemini-cli\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "google-gemini-cli");
        assert_eq!(runtime.api_mode, "chat_completions");
        assert_eq!(runtime.api_key, "google-runtime-token");
        assert_eq!(runtime.base_url, "cloudcode-pa://google");
    }

    #[test]
    fn resolve_model_runtime_reads_anthropic_runtime_credentials() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_path = env::var_os("HERMES_ANTHROPIC_CREDENTIALS_PATH");
        let previous_token = env::var_os("ANTHROPIC_TOKEN");
        let previous_cc = env::var_os("CLAUDE_CODE_OAUTH_TOKEN");
        let previous_api_key = env::var_os("ANTHROPIC_API_KEY");

        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let credentials_path = ctx.hermes_home().join("anthropic_credentials.json");
        fs::write(
            &credentials_path,
            json!({
                "claudeAiOauth": {
                    "accessToken": "cc-anthropic-runtime-token",
                    "refreshToken": "anthropic-runtime-refresh",
                    "expiresAt": i64::MAX / 2,
                    "scopes": ["user:inference"]
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: claude-sonnet-4.6\n  provider: anthropic\n",
        )
        .unwrap();

        unsafe {
            env::set_var("HERMES_ANTHROPIC_CREDENTIALS_PATH", &credentials_path);
            env::remove_var("ANTHROPIC_TOKEN");
            env::remove_var("CLAUDE_CODE_OAUTH_TOKEN");
            env::remove_var("ANTHROPIC_API_KEY");
        }

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        match previous_path {
            Some(value) => unsafe { env::set_var("HERMES_ANTHROPIC_CREDENTIALS_PATH", value) },
            None => unsafe { env::remove_var("HERMES_ANTHROPIC_CREDENTIALS_PATH") },
        }
        match previous_token {
            Some(value) => unsafe { env::set_var("ANTHROPIC_TOKEN", value) },
            None => unsafe { env::remove_var("ANTHROPIC_TOKEN") },
        }
        match previous_cc {
            Some(value) => unsafe { env::set_var("CLAUDE_CODE_OAUTH_TOKEN", value) },
            None => unsafe { env::remove_var("CLAUDE_CODE_OAUTH_TOKEN") },
        }
        match previous_api_key {
            Some(value) => unsafe { env::set_var("ANTHROPIC_API_KEY", value) },
            None => unsafe { env::remove_var("ANTHROPIC_API_KEY") },
        }

        assert_eq!(runtime.provider, "anthropic");
        assert_eq!(runtime.api_mode, "anthropic_messages");
        assert_eq!(runtime.api_key, "cc-anthropic-runtime-token");
        assert_eq!(runtime.auth_type, "oauth_external");
        assert!(runtime.default_headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("anthropic-beta") && value.contains("oauth-2025-04-20")
        }));
        assert!(
            runtime
                .default_headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("x-app") && value == "cli")
        );
    }

    #[test]
    fn resolve_model_runtime_reads_nous_runtime_credentials() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.hermes_home().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access",
                        "refresh_token": "nous-refresh",
                        "portal_base_url": "https://portal.nousresearch.com",
                        "inference_base_url": "https://inference-api.nousresearch.com/v1",
                        "client_id": "hermes-cli",
                        "expires_at": "2999-01-01T00:00:00Z",
                        "agent_key": "nous-agent-key",
                        "agent_key_expires_at": "2999-01-02T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: Nous-Hermes-2-Mixtral-8x7B-DPO\n  provider: nous\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "nous");
        assert_eq!(runtime.api_mode, "chat_completions");
        assert_eq!(runtime.api_key, "nous-agent-key");
        assert_eq!(
            runtime.base_url,
            "https://inference-api.nousresearch.com/v1"
        );
    }

    #[test]
    fn resolve_model_runtime_reads_copilot_acp_runtime_credentials() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_command = env::var_os("HERMES_COPILOT_ACP_COMMAND");
        let previous_base = env::var_os("COPILOT_ACP_BASE_URL");

        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let fake = ctx.hermes_home().join("copilot-fake");
        fs::write(&fake, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&fake).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&fake, permissions).unwrap();
        }
        fs::write(
            ctx.config_path(),
            "model:\n  default: claude-sonnet-4.6\n  provider: copilot-acp\n",
        )
        .unwrap();

        unsafe {
            env::set_var("HERMES_COPILOT_ACP_COMMAND", &fake);
            env::set_var("COPILOT_ACP_BASE_URL", "acp://copilot");
        }

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        match previous_command {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_ACP_COMMAND", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_ACP_COMMAND") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var("COPILOT_ACP_BASE_URL", value) },
            None => unsafe { env::remove_var("COPILOT_ACP_BASE_URL") },
        }

        assert_eq!(runtime.provider, "copilot-acp");
        assert_eq!(runtime.api_mode, "chat_completions");
        assert_eq!(runtime.api_key, "copilot-acp");
        assert_eq!(runtime.base_url, "acp://copilot");
    }

    #[test]
    fn resolve_model_runtime_reads_copilot_runtime_credentials() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_copilot = env::var_os("COPILOT_GITHUB_TOKEN");
        let previous_gh = env::var_os("GH_TOKEN");
        let previous_github = env::var_os("GITHUB_TOKEN");
        let previous_exchange = env::var_os("HERMES_COPILOT_TOKEN_EXCHANGE_URL");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request_text = String::from_utf8_lossy(&request);
            assert!(
                request_text
                    .to_ascii_lowercase()
                    .contains("authorization: token gho_runtime_config")
            );
            let body = json!({
                "token": "copilot-runtime-token",
                "expires_at": 4102444800_u64
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.config_path(),
            "model:\n  default: gpt-4.1\n  provider: copilot\n",
        )
        .unwrap();

        unsafe {
            env::set_var("COPILOT_GITHUB_TOKEN", "gho_runtime_config");
            env::remove_var("GH_TOKEN");
            env::remove_var("GITHUB_TOKEN");
            env::set_var(
                "HERMES_COPILOT_TOKEN_EXCHANGE_URL",
                format!("http://{addr}/copilot-token"),
            );
        }

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");
        server.join().unwrap();

        match previous_copilot {
            Some(value) => unsafe { env::set_var("COPILOT_GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("COPILOT_GITHUB_TOKEN") },
        }
        match previous_gh {
            Some(value) => unsafe { env::set_var("GH_TOKEN", value) },
            None => unsafe { env::remove_var("GH_TOKEN") },
        }
        match previous_github {
            Some(value) => unsafe { env::set_var("GITHUB_TOKEN", value) },
            None => unsafe { env::remove_var("GITHUB_TOKEN") },
        }
        match previous_exchange {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_TOKEN_EXCHANGE_URL") },
        }

        assert_eq!(runtime.provider, "copilot");
        assert_eq!(runtime.api_key, "copilot-runtime-token");
        assert_eq!(runtime.base_url, "https://api.githubcopilot.com");
    }

    #[test]
    fn resolve_model_runtime_routes_copilot_claude_to_anthropic_mode() {
        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        fs::write(
            ctx.config_path(),
            "model:\n  default: anthropic/claude-sonnet-4.6\n  provider: copilot\n  api_key: test-key\n",
        )
        .unwrap();

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        assert_eq!(runtime.provider, "copilot");
        assert_eq!(runtime.model, "claude-sonnet-4.6");
        assert_eq!(runtime.api_key, "test-key");
        assert_eq!(runtime.api_mode, "anthropic_messages");
    }

    #[test]
    fn resolve_model_runtime_reads_qwen_oauth_credentials() {
        let _guard = crate::test_env_lock().lock().expect("env lock");
        let previous_home = env::var_os("HOME");
        let previous_qwen_base = env::var_os("HERMES_QWEN_BASE_URL");

        let (_temp, ctx) = test_context();
        ctx.ensure_hermes_home().expect("ensure home");
        let external_home = ctx.hermes_home().join("external-home");
        let qwen_dir = external_home.join(".qwen");
        fs::create_dir_all(&qwen_dir).expect("qwen dir");
        fs::write(
            qwen_dir.join("oauth_creds.json"),
            json!({
                "access_token": "qwen-runtime-token",
                "refresh_token": "qwen-refresh-token",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            ctx.config_path(),
            "model:\n  default: qwen3.5-plus\n  provider: qwen-oauth\n",
        )
        .unwrap();

        unsafe { env::set_var("HOME", &external_home) };
        unsafe { env::set_var("HERMES_QWEN_BASE_URL", "https://portal.qwen.ai/v1") };

        let loaded = ctx.load_config_document().expect("load config");
        let runtime = ctx
            .resolve_model_runtime(&loaded, &ModelOverrides::default())
            .expect("resolve runtime");

        match previous_home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
        match previous_qwen_base {
            Some(value) => unsafe { env::set_var("HERMES_QWEN_BASE_URL", value) },
            None => unsafe { env::remove_var("HERMES_QWEN_BASE_URL") },
        }

        assert_eq!(runtime.provider, "qwen-oauth");
        assert_eq!(runtime.api_mode, "chat_completions");
        assert_eq!(runtime.api_key, "qwen-runtime-token");
        assert_eq!(runtime.base_url, "https://portal.qwen.ai/v1");
    }
}
