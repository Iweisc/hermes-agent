//! Multi-provider authentication system for Hermes Agent (native Rust port of
//! `hermes_cli/auth.py`).
//!
//! Supports OAuth device code flows (Nous Portal, OpenAI Codex), PKCE flows
//! (Spotify, MiniMax), CLI-credential import (Qwen), and traditional API-key
//! providers (OpenRouter, custom endpoints). Auth state is persisted in
//! `~/.hermes/auth.json` with cross-process file locking.
//!
//! Architecture:
//! - [`PROVIDER_REGISTRY`] defines known providers.
//! - The auth store (auth.json) holds per-provider credential state.
//! - [`resolve_provider`] picks the active provider via a priority chain.
//! - `resolve_*_runtime_credentials` handles token refresh and key minting.
//! - [`logout_command`] is the CLI entry point for clearing auth.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

// =============================================================================
// Constants
// =============================================================================

pub const AUTH_STORE_VERSION: i64 = 1;
pub const AUTH_LOCK_TIMEOUT_SECONDS: f64 = 15.0;

pub const DEFAULT_NOUS_PORTAL_URL: &str = "https://portal.nousresearch.com";
pub const DEFAULT_NOUS_INFERENCE_URL: &str = "https://inference-api.nousresearch.com/v1";
pub const DEFAULT_NOUS_CLIENT_ID: &str = "hermes-cli";
pub const DEFAULT_NOUS_SCOPE: &str = "inference:mint_agent_key";
pub const DEFAULT_AGENT_KEY_MIN_TTL_SECONDS: i64 = 30 * 60;
pub const ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
pub const DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS: i64 = 1;
pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub const MINIMAX_OAUTH_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
pub const MINIMAX_OAUTH_SCOPE: &str = "group_id profile model.completion";
pub const MINIMAX_OAUTH_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:user_code";
pub const MINIMAX_OAUTH_GLOBAL_BASE: &str = "https://api.minimax.io";
pub const MINIMAX_OAUTH_CN_BASE: &str = "https://api.minimaxi.com";
pub const MINIMAX_OAUTH_GLOBAL_INFERENCE: &str = "https://api.minimax.io/anthropic";
pub const MINIMAX_OAUTH_CN_INFERENCE: &str = "https://api.minimaxi.com/anthropic";
pub const MINIMAX_OAUTH_REFRESH_SKEW_SECONDS: i64 = 60;
pub const DEFAULT_QWEN_BASE_URL: &str = "https://portal.qwen.ai/v1";
pub const DEFAULT_GITHUB_MODELS_BASE_URL: &str = "https://api.githubcopilot.com";
pub const DEFAULT_COPILOT_ACP_BASE_URL: &str = "acp://copilot";
pub const DEFAULT_OLLAMA_CLOUD_BASE_URL: &str = "https://ollama.com/v1";
pub const STEPFUN_STEP_PLAN_INTL_BASE_URL: &str = "https://api.stepfun.ai/step_plan/v1";
pub const STEPFUN_STEP_PLAN_CN_BASE_URL: &str = "https://api.stepfun.com/step_plan/v1";
pub const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
pub const QWEN_OAUTH_CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
pub const QWEN_OAUTH_TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
pub const QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
pub const DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL: &str = "https://accounts.spotify.com";
pub const DEFAULT_SPOTIFY_API_BASE_URL: &str = "https://api.spotify.com/v1";
pub const DEFAULT_SPOTIFY_REDIRECT_URI: &str = "http://127.0.0.1:43827/spotify/callback";
pub const SPOTIFY_DOCS_URL: &str =
    "https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify";
pub const SPOTIFY_DASHBOARD_URL: &str = "https://developer.spotify.com/dashboard";
pub const SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;

pub const DEFAULT_GEMINI_CLOUDCODE_BASE_URL: &str = "cloudcode-pa://google";
pub const GEMINI_OAUTH_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 60;

/// LM Studio's default no-auth mode still requires *some* non-empty bearer for
/// the API-key code paths to treat the provider as configured. This sentinel is
/// sent only to LM Studio, never to any remote service.
pub const LMSTUDIO_NOAUTH_PLACEHOLDER: &str = "dummy-lm-api-key";

/// Default Spotify scope (space-joined).
pub fn default_spotify_scope() -> String {
    [
        "user-modify-playback-state",
        "user-read-playback-state",
        "user-read-currently-playing",
        "user-read-recently-played",
        "playlist-read-private",
        "playlist-read-collaborative",
        "playlist-modify-public",
        "playlist-modify-private",
        "user-library-read",
        "user-library-modify",
    ]
    .join(" ")
}

/// Display names for non-inference service providers (e.g. Spotify).
pub fn service_provider_names() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("spotify", "Spotify");
    m
}

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

// Kimi Code (kimi.com/code) issues keys prefixed "sk-kimi-" that only work on
// api.kimi.com/coding.
pub const KIMI_CODE_BASE_URL: &str = "https://api.kimi.com/coding";

pub const NOUS_SHARED_STORE_FILENAME: &str = "nous_auth.json";
pub const NOUS_DEVICE_CODE_SOURCE: &str = "device_code";

// =============================================================================
// Provider Registry
// =============================================================================

/// Describes a known inference provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub id: &'static str,
    pub name: &'static str,
    /// "oauth_device_code", "oauth_external", "oauth_minimax", "api_key",
    /// "external_process", or "aws_sdk".
    pub auth_type: &'static str,
    pub portal_base_url: &'static str,
    pub inference_base_url: &'static str,
    pub client_id: &'static str,
    pub scope: &'static str,
    pub extra: &'static [(&'static str, &'static str)],
    /// For API-key providers: env vars to check (in priority order).
    pub api_key_env_vars: &'static [&'static str],
    /// Optional env var for base URL override.
    pub base_url_env_var: &'static str,
}

impl ProviderConfig {
    pub fn extra_get(&self, key: &str) -> Option<&'static str> {
        self.extra.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
}

const fn pc(
    id: &'static str,
    name: &'static str,
    auth_type: &'static str,
    portal_base_url: &'static str,
    inference_base_url: &'static str,
    client_id: &'static str,
    scope: &'static str,
    extra: &'static [(&'static str, &'static str)],
    api_key_env_vars: &'static [&'static str],
    base_url_env_var: &'static str,
) -> ProviderConfig {
    ProviderConfig {
        id,
        name,
        auth_type,
        portal_base_url,
        inference_base_url,
        client_id,
        scope,
        extra,
        api_key_env_vars,
        base_url_env_var,
    }
}

const MINIMAX_OAUTH_EXTRA: &[(&str, &str)] = &[
    ("region", "global"),
    ("cn_portal_base_url", MINIMAX_OAUTH_CN_BASE),
    ("cn_inference_base_url", MINIMAX_OAUTH_CN_INFERENCE),
];

/// The static provider registry, keyed by provider id.
pub fn provider_registry() -> &'static [ProviderConfig] {
    static REGISTRY: &[ProviderConfig] = &[
        pc("nous", "Nous Portal", "oauth_device_code", DEFAULT_NOUS_PORTAL_URL, DEFAULT_NOUS_INFERENCE_URL, DEFAULT_NOUS_CLIENT_ID, DEFAULT_NOUS_SCOPE, &[], &[], ""),
        pc("openai-codex", "OpenAI Codex", "oauth_external", "", DEFAULT_CODEX_BASE_URL, "", "", &[], &[], ""),
        pc("qwen-oauth", "Qwen OAuth", "oauth_external", "", DEFAULT_QWEN_BASE_URL, "", "", &[], &[], ""),
        pc("google-gemini-cli", "Google Gemini (OAuth)", "oauth_external", "", DEFAULT_GEMINI_CLOUDCODE_BASE_URL, "", "", &[], &[], ""),
        pc("lmstudio", "LM Studio", "api_key", "", "http://127.0.0.1:1234/v1", "", "", &[], &["LM_API_KEY"], "LM_BASE_URL"),
        pc("copilot", "GitHub Copilot", "api_key", "", DEFAULT_GITHUB_MODELS_BASE_URL, "", "", &[], &["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"], "COPILOT_API_BASE_URL"),
        pc("copilot-acp", "GitHub Copilot ACP", "external_process", "", DEFAULT_COPILOT_ACP_BASE_URL, "", "", &[], &[], "COPILOT_ACP_BASE_URL"),
        pc("gemini", "Google AI Studio", "api_key", "", "https://generativelanguage.googleapis.com/v1beta", "", "", &[], &["GOOGLE_API_KEY", "GEMINI_API_KEY"], "GEMINI_BASE_URL"),
        pc("zai", "Z.AI / GLM", "api_key", "", "https://api.z.ai/api/paas/v4", "", "", &[], &["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"], "GLM_BASE_URL"),
        pc("kimi-coding", "Kimi / Moonshot", "api_key", "", "https://api.moonshot.ai/v1", "", "", &[], &["KIMI_API_KEY", "KIMI_CODING_API_KEY"], "KIMI_BASE_URL"),
        pc("kimi-coding-cn", "Kimi / Moonshot (China)", "api_key", "", "https://api.moonshot.cn/v1", "", "", &[], &["KIMI_CN_API_KEY"], ""),
        pc("stepfun", "StepFun Step Plan", "api_key", "", STEPFUN_STEP_PLAN_INTL_BASE_URL, "", "", &[], &["STEPFUN_API_KEY"], "STEPFUN_BASE_URL"),
        pc("arcee", "Arcee AI", "api_key", "", "https://api.arcee.ai/api/v1", "", "", &[], &["ARCEEAI_API_KEY"], "ARCEE_BASE_URL"),
        pc("gmi", "GMI Cloud", "api_key", "", "https://api.gmi-serving.com/v1", "", "", &[], &["GMI_API_KEY"], "GMI_BASE_URL"),
        pc("minimax", "MiniMax", "api_key", "", "https://api.minimax.io/anthropic", "", "", &[], &["MINIMAX_API_KEY"], "MINIMAX_BASE_URL"),
        pc("minimax-oauth", "MiniMax (OAuth \u{00b7} minimax.io)", "oauth_minimax", MINIMAX_OAUTH_GLOBAL_BASE, MINIMAX_OAUTH_GLOBAL_INFERENCE, MINIMAX_OAUTH_CLIENT_ID, MINIMAX_OAUTH_SCOPE, MINIMAX_OAUTH_EXTRA, &[], ""),
        pc("anthropic", "Anthropic", "api_key", "", "https://api.anthropic.com", "", "", &[], &["ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"], "ANTHROPIC_BASE_URL"),
        pc("alibaba", "Alibaba Cloud (DashScope)", "api_key", "", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "", "", &[], &["DASHSCOPE_API_KEY"], "DASHSCOPE_BASE_URL"),
        pc("alibaba-coding-plan", "Alibaba Cloud (Coding Plan)", "api_key", "", "https://coding-intl.dashscope.aliyuncs.com/v1", "", "", &[], &["ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"], "ALIBABA_CODING_PLAN_BASE_URL"),
        pc("minimax-cn", "MiniMax (China)", "api_key", "", "https://api.minimaxi.com/anthropic", "", "", &[], &["MINIMAX_CN_API_KEY"], "MINIMAX_CN_BASE_URL"),
        pc("deepseek", "DeepSeek", "api_key", "", "https://api.deepseek.com/v1", "", "", &[], &["DEEPSEEK_API_KEY"], "DEEPSEEK_BASE_URL"),
        pc("xai", "xAI", "api_key", "", "https://api.x.ai/v1", "", "", &[], &["XAI_API_KEY"], "XAI_BASE_URL"),
        pc("nvidia", "NVIDIA NIM", "api_key", "", "https://integrate.api.nvidia.com/v1", "", "", &[], &["NVIDIA_API_KEY"], "NVIDIA_BASE_URL"),
        pc("ai-gateway", "Vercel AI Gateway", "api_key", "", "https://ai-gateway.vercel.sh/v1", "", "", &[], &["AI_GATEWAY_API_KEY"], "AI_GATEWAY_BASE_URL"),
        pc("opencode-zen", "OpenCode Zen", "api_key", "", "https://opencode.ai/zen/v1", "", "", &[], &["OPENCODE_ZEN_API_KEY"], "OPENCODE_ZEN_BASE_URL"),
        pc("opencode-go", "OpenCode Go", "api_key", "", "https://opencode.ai/zen/go/v1", "", "", &[], &["OPENCODE_GO_API_KEY"], "OPENCODE_GO_BASE_URL"),
        pc("kilocode", "Kilo Code", "api_key", "", "https://api.kilo.ai/api/gateway", "", "", &[], &["KILOCODE_API_KEY"], "KILOCODE_BASE_URL"),
        pc("huggingface", "Hugging Face", "api_key", "", "https://router.huggingface.co/v1", "", "", &[], &["HF_TOKEN"], "HF_BASE_URL"),
        pc("xiaomi", "Xiaomi MiMo", "api_key", "", "https://api.xiaomimimo.com/v1", "", "", &[], &["XIAOMI_API_KEY"], "XIAOMI_BASE_URL"),
        pc("tencent-tokenhub", "Tencent TokenHub", "api_key", "", "https://tokenhub.tencentmaas.com/v1", "", "", &[], &["TOKENHUB_API_KEY"], "TOKENHUB_BASE_URL"),
        pc("ollama-cloud", "Ollama Cloud", "api_key", "", DEFAULT_OLLAMA_CLOUD_BASE_URL, "", "", &[], &["OLLAMA_API_KEY"], "OLLAMA_BASE_URL"),
        pc("bedrock", "AWS Bedrock", "aws_sdk", "", "https://bedrock-runtime.us-east-1.amazonaws.com", "", "", &[], &[], "BEDROCK_BASE_URL"),
        pc("azure-foundry", "Azure Foundry", "api_key", "", "", "", "", &[], &["AZURE_FOUNDRY_API_KEY"], "AZURE_FOUNDRY_BASE_URL"),
    ];
    REGISTRY
}

/// Look up a provider config by id.
pub fn get_provider_config(provider_id: &str) -> Option<&'static ProviderConfig> {
    provider_registry().iter().find(|p| p.id == provider_id)
}

// =============================================================================
// Error Types
// =============================================================================

/// Structured auth error with UX mapping hints (mirror of Python `AuthError`).
#[derive(Debug, Clone)]
pub struct AuthError {
    pub message: String,
    pub provider: String,
    pub code: Option<String>,
    pub relogin_required: bool,
}

impl AuthError {
    pub fn new(message: impl Into<String>) -> Self {
        AuthError {
            message: message.into(),
            provider: String::new(),
            code: None,
            relogin_required: false,
        }
    }
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
    pub fn relogin(mut self) -> Self {
        self.relogin_required = true;
        self
    }
    pub fn code_is(&self, code: &str) -> bool {
        self.code.as_deref() == Some(code)
    }
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AuthError {}

/// Map auth failures to concise user-facing guidance.
pub fn format_auth_error(error: &AuthError) -> String {
    if error.relogin_required {
        return format!("{} Run `hermes model` to re-authenticate.", error.message);
    }
    match error.code.as_deref() {
        Some("subscription_required") => {
            "No active paid subscription found on Nous Portal. \
             Please purchase/activate a subscription, then retry."
                .to_string()
        }
        Some("insufficient_credits") => {
            "Subscription credits are exhausted. \
             Top up/renew credits in Nous Portal, then retry."
                .to_string()
        }
        Some("temporarily_unavailable") => {
            format!("{} Please retry in a few seconds.", error.message)
        }
        _ => error.message.clone(),
    }
}

/// Return a short hash fingerprint for telemetry without leaking token bytes.
pub fn token_fingerprint(token: &str) -> Option<String> {
    let cleaned = token.trim();
    if cleaned.is_empty() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(cleaned.as_bytes());
    let digest = hasher.finalize();
    Some(hex_lower(&digest)[..12].to_string())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn oauth_trace_enabled() -> bool {
    let raw = std::env::var("HERMES_OAUTH_TRACE")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    matches!(raw.as_str(), "1" | "true" | "yes" | "on")
}

fn oauth_trace(event: &str, fields: &[(&str, Value)]) {
    if !oauth_trace_enabled() {
        return;
    }
    let mut payload = Map::new();
    payload.insert("event".to_string(), json!(event));
    for (k, v) in fields {
        payload.insert((*k).to_string(), v.clone());
    }
    log::info!(
        "oauth_trace {}",
        serde_json::to_string(&Value::Object(payload)).unwrap_or_default()
    );
}

// =============================================================================
// Placeholder secret detection
// =============================================================================

fn placeholder_secret_values() -> &'static [&'static str] {
    &[
        "*", "**", "***", "changeme", "your_api_key", "your-api-key",
        "placeholder", "example", "dummy", "null", "none",
    ]
}

/// Return true when a configured secret looks usable, not empty/placeholder.
pub fn has_usable_secret(value: &str) -> bool {
    has_usable_secret_min(value, 4)
}

pub fn has_usable_secret_min(value: &str, min_length: usize) -> bool {
    let cleaned = value.trim();
    if cleaned.len() < min_length {
        return false;
    }
    if placeholder_secret_values().contains(&cleaned.to_lowercase().as_str()) {
        return false;
    }
    true
}

// =============================================================================
// Home directory / config helpers
//
// These mirror hermes_cli.config.{get_hermes_home,get_config_path,
// get_env_value,save_env_value}. We re-implement them locally (reading
// HERMES_HOME, then ~/.hermes) so this module is self-contained, but defer to
// crate::mod_hermes_constants when available for parity in the wider binary.
// =============================================================================

/// Resolve the Hermes home directory (HERMES_HOME or ~/.hermes).
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    home_dir().join(".hermes")
}

/// Resolve config.yaml path under HERMES_HOME.
pub fn get_config_path() -> PathBuf {
    get_hermes_home().join("config.yaml")
}

fn env_path() -> PathBuf {
    get_hermes_home().join(".env")
}

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn parse_env_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let rest = trimmed.strip_prefix("export ").unwrap_or(trimmed);
    let (key, raw_val) = rest.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    let mut val = raw_val.trim().to_string();
    // Strip a single layer of surrounding matching quotes.
    if val.len() >= 2 {
        let bytes = val.as_bytes();
        let first = bytes[0];
        let last = bytes[val.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            val = val[1..val.len() - 1].to_string();
        }
    }
    Some((key.to_string(), val))
}

/// Read the `.env` file into a map (key → value).
fn read_env_file() -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Ok(text) = fs::read_to_string(env_path()) {
        for line in text.lines() {
            if let Some((k, v)) = parse_env_line(line) {
                out.insert(k, v);
            }
        }
    }
    out
}

/// Resolve a value from the `.env` file first, then the process environment.
/// Mirrors hermes_cli.config.get_env_value semantics for the callers in this
/// module (which fall back to os.getenv).
pub fn get_env_value(name: &str) -> String {
    let file = read_env_file();
    if let Some(v) = file.get(name) {
        if !v.is_empty() {
            return v.clone();
        }
    }
    std::env::var(name).unwrap_or_default()
}

/// Persist a key/value into the `.env` file (creating it if needed).
pub fn save_env_value(name: &str, value: &str) -> std::io::Result<()> {
    let path = env_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let existing = fs::read_to_string(&path).unwrap_or_default();
    let mut lines: Vec<String> = Vec::new();
    let mut replaced = false;
    for line in existing.lines() {
        if let Some((k, _)) = parse_env_line(line) {
            if k == name {
                lines.push(format!("{}={}", name, value));
                replaced = true;
                continue;
            }
        }
        lines.push(line.to_string());
    }
    if !replaced {
        lines.push(format!("{}={}", name, value));
    }
    let mut body = lines.join("\n");
    body.push('\n');
    fs::write(&path, body)?;
    Ok(())
}

/// Read config.yaml as a JSON Value (object), or null if missing/unparseable.
pub fn read_raw_config() -> Value {
    let path = get_config_path();
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Value::Null,
    };
    match serde_yaml::from_str::<serde_yaml::Value>(&text) {
        Ok(yv) => serde_json::to_value(yv).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

/// Write a JSON Value back to config.yaml (preserving key order via serde_yaml).
fn write_raw_config(config: &Value) -> std::io::Result<()> {
    let path = get_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml::to_string(config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    fs::write(&path, yaml)
}

// =============================================================================
// Timestamp / TTL helpers
// =============================================================================

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn utc_now_iso() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

/// ISO-8601 with trailing Z (mirrors Python's `.isoformat().replace("+00:00","Z")`).
fn utc_now_iso_z() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)
}

fn iso_from_epoch(epoch: f64) -> String {
    let secs = epoch.floor() as i64;
    let nanos = ((epoch - epoch.floor()) * 1_000_000_000.0) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .unwrap_or_else(Utc::now)
        .to_rfc3339_opts(SecondsFormat::Micros, false)
}

/// Parse an ISO timestamp into epoch seconds (handles trailing Z, naive→UTC).
pub fn parse_iso_timestamp(value: &str) -> Option<f64> {
    let text = value.trim();
    if text.is_empty() {
        return None;
    }
    // Try RFC3339 directly first.
    if let Ok(dt) = DateTime::parse_from_rfc3339(text) {
        return Some(dt.timestamp() as f64 + dt.timestamp_subsec_micros() as f64 / 1_000_000.0);
    }
    // Normalize trailing Z.
    let normalized = if let Some(stripped) = text.strip_suffix('Z') {
        format!("{}+00:00", stripped)
    } else {
        text.to_string()
    };
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Some(dt.timestamp() as f64 + dt.timestamp_subsec_micros() as f64 / 1_000_000.0);
    }
    // Naive datetime (no tz) -> assume UTC.
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S%.f") {
        let dt = ndt.and_utc();
        return Some(dt.timestamp() as f64 + dt.timestamp_subsec_micros() as f64 / 1_000_000.0);
    }
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S") {
        let dt = ndt.and_utc();
        return Some(dt.timestamp() as f64);
    }
    None
}

fn parse_iso_opt(value: Option<&Value>) -> Option<f64> {
    value.and_then(|v| v.as_str()).and_then(parse_iso_timestamp)
}

/// True when the timestamp is missing or within `skew_seconds` of expiry.
pub fn is_expiring(expires_at_iso: Option<&Value>, skew_seconds: i64) -> bool {
    match parse_iso_opt(expires_at_iso) {
        None => true,
        Some(epoch) => epoch <= (now_unix() + skew_seconds as f64),
    }
}

fn coerce_ttl_seconds(expires_in: Option<&Value>) -> i64 {
    let ttl = match expires_in {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
        Some(Value::String(s)) => s.trim().parse::<f64>().map(|f| f as i64).unwrap_or(0),
        _ => 0,
    };
    ttl.max(0)
}

fn optional_base_url(value: Option<&Value>) -> Option<String> {
    let s = value?.as_str()?;
    let cleaned = s.trim().trim_end_matches('/');
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn decode_jwt_claims(token: &str) -> Map<String, Value> {
    if token.matches('.').count() != 2 {
        return Map::new();
    }
    let payload = token.split('.').nth(1).unwrap_or("");
    let pad = (4 - payload.len() % 4) % 4;
    let mut padded = payload.to_string();
    for _ in 0..pad {
        padded.push('=');
    }
    let raw = match base64::engine::general_purpose::URL_SAFE.decode(padded.as_bytes()) {
        Ok(r) => r,
        Err(_) => return Map::new(),
    };
    match serde_json::from_slice::<Value>(&raw) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

fn codex_access_token_is_expiring(access_token: &str, skew_seconds: i64) -> bool {
    let claims = decode_jwt_claims(access_token);
    match claims.get("exp") {
        Some(Value::Number(n)) => {
            let exp = n.as_f64().unwrap_or(0.0);
            exp <= (now_unix() + skew_seconds.max(0) as f64)
        }
        _ => false,
    }
}

// =============================================================================
// Auth Store — persistence layer for ~/.hermes/auth.json
// =============================================================================

fn auth_file_path() -> PathBuf {
    get_hermes_home().join("auth.json")
}

fn auth_lock_path() -> PathBuf {
    auth_file_path().with_extension("lock")
}

// Process-local guard standing in for the cross-process advisory file lock.
// The Python code uses fcntl/msvcrt with reentrancy; here we use a global
// reentrant-ish mutex guard. Re-entrancy within one thread is handled by the
// AuthStoreGuard's depth tracking via a thread-local counter.
static AUTH_STORE_MUTEX: Mutex<()> = Mutex::new(());

thread_local! {
    static LOCK_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// RAII guard for the auth store lock. Reentrant per-thread.
pub struct AuthStoreGuard {
    _inner: Option<std::sync::MutexGuard<'static, ()>>,
    held_file: Option<fs::File>,
}

impl Drop for AuthStoreGuard {
    fn drop(&mut self) {
        LOCK_DEPTH.with(|d| {
            let cur = d.get();
            if cur > 0 {
                d.set(cur - 1);
            }
        });
        // Release the advisory file lock if we held one.
        #[cfg(unix)]
        if let Some(f) = self.held_file.take() {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::flock(f.as_raw_fd(), libc::LOCK_UN);
            }
        }
    }
}

/// Acquire the cross-process advisory lock for auth.json. Reentrant within a
/// thread. On non-unix or when locking is unavailable, falls back to the
/// in-process mutex only.
pub fn auth_store_lock() -> AuthStoreGuard {
    auth_store_lock_timeout(AUTH_LOCK_TIMEOUT_SECONDS)
}

pub fn auth_store_lock_timeout(timeout_seconds: f64) -> AuthStoreGuard {
    let depth = LOCK_DEPTH.with(|d| d.get());
    if depth > 0 {
        LOCK_DEPTH.with(|d| d.set(depth + 1));
        return AuthStoreGuard {
            _inner: None,
            held_file: None,
        };
    }

    // Acquire the in-process mutex first (leak the guard's lifetime to 'static
    // via the static mutex).
    let guard = AUTH_STORE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Best-effort cross-process flock on the lock file.
    let held_file = acquire_file_lock(timeout_seconds);

    LOCK_DEPTH.with(|d| d.set(1));
    AuthStoreGuard {
        _inner: Some(guard),
        held_file,
    }
}

#[cfg(unix)]
fn acquire_file_lock(timeout_seconds: f64) -> Option<fs::File> {
    use std::os::unix::io::AsRawFd;
    let lock_path = auth_lock_path();
    if let Some(parent) = lock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&lock_path)
        .ok()?;
    let deadline = now_unix() + timeout_seconds.max(1.0);
    loop {
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Some(file);
        }
        if now_unix() >= deadline {
            // Timed out; return the file unlocked rather than blocking forever —
            // the in-process mutex still serializes same-process callers.
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(not(unix))]
fn acquire_file_lock(_timeout_seconds: f64) -> Option<fs::File> {
    None
}

/// Load the auth store from disk, migrating legacy shapes. Never raises;
/// returns an empty store on parse failure (preserving a `.corrupt` copy).
pub fn load_auth_store() -> Value {
    load_auth_store_from(&auth_file_path())
}

fn empty_store() -> Value {
    json!({"version": AUTH_STORE_VERSION, "providers": {}})
}

fn load_auth_store_from(auth_file: &Path) -> Value {
    if !auth_file.exists() {
        return empty_store();
    }
    let text = match fs::read_to_string(auth_file) {
        Ok(t) => t,
        Err(_) => return empty_store(),
    };
    let raw: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            // Preserve a corrupt copy, best effort.
            let corrupt = auth_file.with_extension("json.corrupt");
            let _ = fs::copy(auth_file, &corrupt);
            log::warn!(
                "auth: failed to parse {} — starting with empty store. Corrupt file preserved at {}",
                auth_file.display(),
                corrupt.display()
            );
            return empty_store();
        }
    };

    if let Value::Object(ref map) = raw {
        let has_providers = map.get("providers").map(|v| v.is_object()).unwrap_or(false);
        let has_pool = map
            .get("credential_pool")
            .map(|v| v.is_object())
            .unwrap_or(false);
        if has_providers || has_pool {
            let mut out = raw.clone();
            if let Value::Object(ref mut m) = out {
                m.entry("providers").or_insert_with(|| json!({}));
            }
            return out;
        }
        // Migrate from "systems" format.
        if let Some(Value::Object(systems)) = map.get("systems") {
            let mut providers = Map::new();
            if let Some(nous) = systems.get("nous_portal") {
                providers.insert("nous".to_string(), nous.clone());
            }
            let active = if providers.is_empty() {
                Value::Null
            } else {
                json!("nous")
            };
            return json!({
                "version": AUTH_STORE_VERSION,
                "providers": providers,
                "active_provider": active,
            });
        }
    }
    empty_store()
}

/// Persist the auth store atomically with owner-only permissions.
pub fn save_auth_store(auth_store: &mut Value) -> std::io::Result<PathBuf> {
    let auth_file = auth_file_path();
    if let Some(parent) = auth_file.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(map) = auth_store.as_object_mut() {
        map.insert("version".to_string(), json!(AUTH_STORE_VERSION));
        map.insert("updated_at".to_string(), json!(utc_now_iso()));
    }
    let payload = format!(
        "{}\n",
        serde_json::to_string_pretty(auth_store).unwrap_or_default()
    );
    let tmp_name = format!(
        "{}.tmp.{}.{}",
        auth_file
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("auth.json"),
        std::process::id(),
        uuid_hex()
    );
    let tmp_path = auth_file.with_file_name(tmp_name);
    {
        let mut handle = fs::File::create(&tmp_path)?;
        handle.write_all(payload.as_bytes())?;
        handle.flush()?;
        let _ = handle.sync_all();
    }
    fs::rename(&tmp_path, &auth_file)?;
    let _ = fs::remove_file(&tmp_path);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&auth_file, fs::Permissions::from_mode(0o600));
    }
    Ok(auth_file)
}

/// Generate a random 32-char lowercase-hex token (uuid4().hex equivalent).
fn uuid_hex() -> String {
    let mut bytes = [0u8; 16];
    if getrandom_fill(&mut bytes).is_err() {
        // Fallback to time-seeded values.
        let n = now_unix().to_bits();
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((n >> (i % 8 * 8)) & 0xff) as u8;
        }
    }
    hex_lower(&bytes)
}

fn getrandom_fill(buf: &mut [u8]) -> Result<(), ()> {
    // Use /dev/urandom on unix; fall back to a weak source otherwise.
    #[cfg(unix)]
    {
        if let Ok(mut f) = fs::File::open("/dev/urandom") {
            use std::io::Read;
            if f.read_exact(buf).is_ok() {
                return Ok(());
            }
        }
    }
    Err(())
}

/// Load a single provider's state (cloned), or None.
pub fn load_provider_state(auth_store: &Value, provider_id: &str) -> Option<Value> {
    let providers = auth_store.get("providers")?.as_object()?;
    match providers.get(provider_id) {
        Some(v) if v.is_object() => Some(v.clone()),
        _ => None,
    }
}

/// Store provider state and set it active.
pub fn save_provider_state(auth_store: &mut Value, provider_id: &str, state: Value) {
    store_provider_state(auth_store, provider_id, state, true);
}

/// Store provider state, optionally marking it active.
pub fn store_provider_state(
    auth_store: &mut Value,
    provider_id: &str,
    state: Value,
    set_active: bool,
) {
    if !auth_store.is_object() {
        *auth_store = empty_store();
    }
    let obj = auth_store.as_object_mut().unwrap();
    let providers = obj
        .entry("providers")
        .or_insert_with(|| json!({}));
    if !providers.is_object() {
        *providers = json!({});
    }
    providers
        .as_object_mut()
        .unwrap()
        .insert(provider_id.to_string(), state);
    if set_active {
        obj.insert("active_provider".to_string(), json!(provider_id));
    }
}

/// True when provider_id is a registry or service provider.
pub fn is_known_auth_provider(provider_id: &str) -> bool {
    let normalized = provider_id.trim().to_lowercase();
    get_provider_config(&normalized).is_some()
        || service_provider_names().contains_key(normalized.as_str())
}

/// Display name for a provider id.
pub fn get_auth_provider_display_name(provider_id: &str) -> String {
    let normalized = provider_id.trim().to_lowercase();
    if let Some(pc) = get_provider_config(&normalized) {
        return pc.name.to_string();
    }
    service_provider_names()
        .get(normalized.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| provider_id.to_string())
}

/// Return the persisted credential pool (full map) or one provider slice.
pub fn read_credential_pool(provider_id: Option<&str>) -> Value {
    let auth_store = load_auth_store();
    let pool = auth_store
        .get("credential_pool")
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    match provider_id {
        None => Value::Object(pool),
        Some(pid) => match pool.get(pid) {
            Some(v) if v.is_array() => v.clone(),
            _ => json!([]),
        },
    }
}

/// Persist one provider's credential pool under auth.json.
pub fn write_credential_pool(provider_id: &str, entries: Vec<Value>) -> std::io::Result<PathBuf> {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    if !auth_store.is_object() {
        auth_store = empty_store();
    }
    let obj = auth_store.as_object_mut().unwrap();
    let pool = obj
        .entry("credential_pool")
        .or_insert_with(|| json!({}));
    if !pool.is_object() {
        *pool = json!({});
    }
    pool.as_object_mut()
        .unwrap()
        .insert(provider_id.to_string(), Value::Array(entries));
    save_auth_store(&mut auth_store)
}

/// Mark a credential source as suppressed so it won't be re-seeded.
pub fn suppress_credential_source(provider_id: &str, source: &str) {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let obj = auth_store.as_object_mut().unwrap();
    let suppressed = obj.entry("suppressed_sources").or_insert_with(|| json!({}));
    if !suppressed.is_object() {
        *suppressed = json!({});
    }
    let list = suppressed
        .as_object_mut()
        .unwrap()
        .entry(provider_id.to_string())
        .or_insert_with(|| json!([]));
    if let Some(arr) = list.as_array_mut() {
        if !arr.iter().any(|v| v.as_str() == Some(source)) {
            arr.push(json!(source));
        }
    }
    let _ = save_auth_store(&mut auth_store);
}

/// Check if a credential source has been suppressed by the user.
pub fn is_source_suppressed(provider_id: &str, source: &str) -> bool {
    let auth_store = load_auth_store();
    auth_store
        .get("suppressed_sources")
        .and_then(|v| v.get(provider_id))
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().any(|v| v.as_str() == Some(source)))
        .unwrap_or(false)
}

/// Clear a suppression marker. Returns true if a marker was cleared.
pub fn unsuppress_credential_source(provider_id: &str, source: &str) -> bool {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let Some(obj) = auth_store.as_object_mut() else {
        return false;
    };
    let Some(suppressed) = obj.get_mut("suppressed_sources").and_then(|v| v.as_object_mut())
    else {
        return false;
    };
    let Some(list) = suppressed.get_mut(provider_id).and_then(|v| v.as_array_mut()) else {
        return false;
    };
    let before = list.len();
    list.retain(|v| v.as_str() != Some(source));
    if list.len() == before {
        return false;
    }
    if list.is_empty() {
        suppressed.remove(provider_id);
    }
    if suppressed.is_empty() {
        obj.remove("suppressed_sources");
    }
    let _ = save_auth_store(&mut auth_store);
    true
}

/// Return persisted auth state for a provider, or None.
pub fn get_provider_auth_state(provider_id: &str) -> Option<Value> {
    let auth_store = load_auth_store();
    load_provider_state(&auth_store, provider_id)
}

/// Return the currently active provider ID from the auth store.
pub fn get_active_provider() -> Option<String> {
    let auth_store = load_auth_store();
    auth_store
        .get("active_provider")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Return True only if the user has explicitly configured this provider.
pub fn is_provider_explicitly_configured(provider_id: &str) -> bool {
    let normalized = provider_id.trim().to_lowercase();

    // 1. auth.json active_provider
    let auth_store = load_auth_store();
    if let Some(active) = auth_store.get("active_provider").and_then(|v| v.as_str()) {
        let active = active.trim().to_lowercase();
        if !active.is_empty() && active == normalized {
            return true;
        }
    }

    // 2. config.yaml model.provider
    let cfg = read_raw_config();
    if let Some(provider) = cfg
        .get("model")
        .and_then(|m| m.get("provider"))
        .and_then(|p| p.as_str())
    {
        if provider.trim().to_lowercase() == normalized {
            return true;
        }
    }

    // 3. Provider-specific env vars (excluding implicit CLAUDE_CODE_OAUTH_TOKEN)
    const IMPLICIT_ENV_VARS: &[&str] = &["CLAUDE_CODE_OAUTH_TOKEN"];
    if let Some(pc) = get_provider_config(&normalized) {
        if pc.auth_type == "api_key" {
            for env_var in pc.api_key_env_vars {
                if IMPLICIT_ENV_VARS.contains(env_var) {
                    continue;
                }
                if has_usable_secret(&std::env::var(env_var).unwrap_or_default()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Clear auth state for a provider (None => active provider).
/// Returns true if something was cleared.
pub fn clear_provider_auth(provider_id: Option<&str>) -> bool {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let target: Option<String> = provider_id
        .map(|s| s.to_string())
        .or_else(|| {
            auth_store
                .get("active_provider")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    let target = match target {
        Some(t) if !t.is_empty() => t,
        _ => return false,
    };

    let obj = auth_store.as_object_mut().unwrap();
    if !obj.get("providers").map(|v| v.is_object()).unwrap_or(false) {
        obj.insert("providers".to_string(), json!({}));
    }
    if !obj
        .get("credential_pool")
        .map(|v| v.is_object())
        .unwrap_or(false)
    {
        obj.insert("credential_pool".to_string(), json!({}));
    }

    let mut cleared = false;
    if obj
        .get_mut("providers")
        .and_then(|v| v.as_object_mut())
        .map(|m| m.remove(&target).is_some())
        .unwrap_or(false)
    {
        cleared = true;
    }
    if obj
        .get_mut("credential_pool")
        .and_then(|v| v.as_object_mut())
        .map(|m| m.remove(&target).is_some())
        .unwrap_or(false)
    {
        cleared = true;
    }
    if obj.get("active_provider").and_then(|v| v.as_str()) == Some(target.as_str()) {
        obj.insert("active_provider".to_string(), Value::Null);
        cleared = true;
    }

    if !cleared {
        return false;
    }
    let _ = save_auth_store(&mut auth_store);
    true
}

/// Clear active_provider without deleting credentials.
pub fn deactivate_provider() {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    if let Some(obj) = auth_store.as_object_mut() {
        obj.insert("active_provider".to_string(), Value::Null);
    }
    let _ = save_auth_store(&mut auth_store);
}

// =============================================================================
// Provider Resolution
// =============================================================================

fn provider_aliases() -> HashMap<&'static str, &'static str> {
    let pairs: &[(&str, &str)] = &[
        ("glm", "zai"), ("z-ai", "zai"), ("z.ai", "zai"), ("zhipu", "zai"),
        ("google", "gemini"), ("google-gemini", "gemini"), ("google-ai-studio", "gemini"),
        ("x-ai", "xai"), ("x.ai", "xai"), ("grok", "xai"),
        ("kimi", "kimi-coding"), ("kimi-for-coding", "kimi-coding"), ("moonshot", "kimi-coding"),
        ("kimi-cn", "kimi-coding-cn"), ("moonshot-cn", "kimi-coding-cn"),
        ("step", "stepfun"), ("stepfun-coding-plan", "stepfun"),
        ("arcee-ai", "arcee"), ("arceeai", "arcee"),
        ("gmi-cloud", "gmi"), ("gmicloud", "gmi"),
        ("minimax-china", "minimax-cn"), ("minimax_cn", "minimax-cn"),
        ("minimax-portal", "minimax-oauth"), ("minimax-global", "minimax-oauth"), ("minimax_oauth", "minimax-oauth"),
        ("alibaba_coding", "alibaba-coding-plan"), ("alibaba-coding", "alibaba-coding-plan"),
        ("alibaba_coding_plan", "alibaba-coding-plan"),
        ("claude", "anthropic"), ("claude-code", "anthropic"),
        ("github", "copilot"), ("github-copilot", "copilot"),
        ("github-models", "copilot"), ("github-model", "copilot"),
        ("github-copilot-acp", "copilot-acp"), ("copilot-acp-agent", "copilot-acp"),
        ("aigateway", "ai-gateway"), ("vercel", "ai-gateway"), ("vercel-ai-gateway", "ai-gateway"),
        ("opencode", "opencode-zen"), ("zen", "opencode-zen"),
        ("qwen-portal", "qwen-oauth"), ("qwen-cli", "qwen-oauth"), ("qwen-oauth", "qwen-oauth"),
        ("google-gemini-cli", "google-gemini-cli"), ("gemini-cli", "google-gemini-cli"), ("gemini-oauth", "google-gemini-cli"),
        ("hf", "huggingface"), ("hugging-face", "huggingface"), ("huggingface-hub", "huggingface"),
        ("mimo", "xiaomi"), ("xiaomi-mimo", "xiaomi"),
        ("tencent", "tencent-tokenhub"), ("tokenhub", "tencent-tokenhub"),
        ("tencent-cloud", "tencent-tokenhub"), ("tencentmaas", "tencent-tokenhub"),
        ("aws", "bedrock"), ("aws-bedrock", "bedrock"), ("amazon-bedrock", "bedrock"), ("amazon", "bedrock"),
        ("go", "opencode-go"), ("opencode-go-sub", "opencode-go"),
        ("kilo", "kilocode"), ("kilo-code", "kilocode"), ("kilo-gateway", "kilocode"),
        ("lmstudio", "lmstudio"), ("lm-studio", "lmstudio"), ("lm_studio", "lmstudio"),
        ("ollama", "custom"), ("ollama_cloud", "ollama-cloud"),
        ("vllm", "custom"), ("llamacpp", "custom"),
        ("llama.cpp", "custom"), ("llama-cpp", "custom"),
    ];
    pairs.iter().copied().collect()
}

/// Determine which inference provider to use.
///
/// Priority (when requested="auto" or None):
/// 1. active_provider in auth.json with valid credentials
/// 2. Explicit CLI api_key/base_url -> "openrouter"
/// 3. OPENAI_API_KEY or OPENROUTER_API_KEY -> "openrouter"
/// 4. Provider-specific API keys -> that provider
/// 5. Fallback raises an AuthError.
pub fn resolve_provider(
    requested: Option<&str>,
    explicit_api_key: Option<&str>,
    explicit_base_url: Option<&str>,
) -> Result<String, AuthError> {
    let normalized_in = requested.unwrap_or("auto").trim().to_lowercase();
    let aliases = provider_aliases();
    let normalized = aliases
        .get(normalized_in.as_str())
        .map(|s| s.to_string())
        .unwrap_or(normalized_in);

    if normalized == "openrouter" {
        return Ok("openrouter".to_string());
    }
    if normalized == "custom" {
        return Ok("custom".to_string());
    }
    if get_provider_config(&normalized).is_some() {
        return Ok(normalized);
    }
    if normalized != "auto" {
        let msg = format!(
            "Unknown provider '{}'. Check 'hermes model' for available providers, \
             or run 'hermes doctor' to diagnose config issues.",
            normalized
        );
        return Err(AuthError::new(msg).with_code("invalid_provider"));
    }

    // Explicit one-off CLI creds always mean openrouter/custom.
    let has_api_key = explicit_api_key.map(|s| !s.is_empty()).unwrap_or(false);
    let has_base_url = explicit_base_url.map(|s| !s.is_empty()).unwrap_or(false);
    if has_api_key || has_base_url {
        return Ok("openrouter".to_string());
    }

    // Active OAuth provider from the auth store.
    let auth_store = load_auth_store();
    if let Some(active) = auth_store.get("active_provider").and_then(|v| v.as_str()) {
        if get_provider_config(active).is_some() {
            let status = get_auth_status(Some(active));
            if status
                .get("logged_in")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return Ok(active.to_string());
            }
        }
    }

    let openai = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let openrouter = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    if has_usable_secret(&openai) || has_usable_secret(&openrouter) {
        return Ok("openrouter".to_string());
    }

    // Auto-detect API-key providers by env vars (skipping copilot/lmstudio).
    for pc in provider_registry() {
        if pc.auth_type != "api_key" {
            continue;
        }
        if pc.id == "copilot" || pc.id == "lmstudio" {
            continue;
        }
        for env_var in pc.api_key_env_vars {
            if has_usable_secret(&std::env::var(env_var).unwrap_or_default()) {
                return Ok(pc.id.to_string());
            }
        }
    }

    Err(AuthError::new(
        "No inference provider configured. Run 'hermes model' to choose a \
         provider and model, or set an API key (OPENROUTER_API_KEY, \
         OPENAI_API_KEY, etc.) in ~/.hermes/.env.",
    )
    .with_code("no_provider_configured"))
}

// =============================================================================
// HTTP helper (reqwest blocking)
// =============================================================================

fn http_client(timeout_seconds: f64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_seconds.max(1.0)))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

// =============================================================================
// Kimi / Z.AI endpoint detection
// =============================================================================

/// Return the correct Kimi base URL based on the API key prefix.
pub fn resolve_kimi_base_url(api_key: &str, default_url: &str, env_override: &str) -> String {
    if !env_override.is_empty() {
        return env_override.to_string();
    }
    if api_key.is_empty() {
        return default_url.to_string();
    }
    if api_key.starts_with("sk-kimi-") {
        return KIMI_CODE_BASE_URL.to_string();
    }
    default_url.to_string()
}

/// Candidate Z.AI endpoints: (id, base_url, probe_models, label).
fn zai_endpoints() -> Vec<(&'static str, &'static str, Vec<&'static str>, &'static str)> {
    vec![
        ("global", "https://api.z.ai/api/paas/v4", vec!["glm-5"], "Global"),
        ("cn", "https://open.bigmodel.cn/api/paas/v4", vec!["glm-5"], "China"),
        ("coding-global", "https://api.z.ai/api/coding/paas/v4", vec!["glm-5.1", "glm-5v-turbo", "glm-4.7"], "Global (Coding Plan)"),
        ("coding-cn", "https://open.bigmodel.cn/api/coding/paas/v4", vec!["glm-5.1", "glm-5v-turbo", "glm-4.7"], "China (Coding Plan)"),
    ]
}

/// Probe z.ai endpoints to find one that accepts this API key.
/// Returns a map {id, base_url, model, label} for the first working endpoint.
pub fn detect_zai_endpoint(api_key: &str, timeout: f64) -> Option<Value> {
    let client = http_client(timeout);
    for (ep_id, base_url, probe_models, label) in zai_endpoints() {
        for model in probe_models {
            let body = json!({
                "model": model,
                "stream": false,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            });
            let resp = client
                .post(format!("{}/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send();
            if let Ok(r) = resp {
                if r.status().as_u16() == 200 {
                    return Some(json!({
                        "id": ep_id,
                        "base_url": base_url,
                        "model": model,
                        "label": label,
                    }));
                }
            }
        }
    }
    None
}

fn sha256_hex16(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex_lower(&hasher.finalize())[..16].to_string()
}

/// Return the correct Z.AI base URL by probing endpoints, caching the result.
pub fn resolve_zai_base_url(api_key: &str, default_url: &str, env_override: &str) -> String {
    if !env_override.is_empty() {
        return env_override.to_string();
    }
    if api_key.is_empty() {
        return default_url.to_string();
    }

    let mut auth_store = load_auth_store();
    let state = load_provider_state(&auth_store, "zai").unwrap_or_else(|| json!({}));
    if let Some(cached) = state.get("detected_endpoint") {
        if cached.is_object() {
            if let Some(base) = cached.get("base_url").and_then(|v| v.as_str()) {
                if !base.is_empty() {
                    let key_hash = cached
                        .get("key_hash")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if key_hash == sha256_hex16(api_key) {
                        return base.to_string();
                    }
                }
            }
        }
    }

    if let Some(detected) = detect_zai_endpoint(api_key, 8.0) {
        if let Some(base) = detected.get("base_url").and_then(|v| v.as_str()) {
            if !base.is_empty() {
                let key_hash = sha256_hex16(api_key);
                let mut new_state = state.clone();
                if let Some(obj) = new_state.as_object_mut() {
                    obj.insert(
                        "detected_endpoint".to_string(),
                        json!({
                            "base_url": base,
                            "endpoint_id": detected.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                            "model": detected.get("model").and_then(|v| v.as_str()).unwrap_or(""),
                            "label": detected.get("label").and_then(|v| v.as_str()).unwrap_or(""),
                            "key_hash": key_hash,
                        }),
                    );
                }
                store_provider_state(&mut auth_store, "zai", new_state, true);
                let _ = save_auth_store(&mut auth_store);
                return base.to_string();
            }
        }
    }
    default_url.to_string()
}

// =============================================================================
// API-key provider secret resolution
// =============================================================================

/// Resolve an API-key provider's token and indicate where it came from.
/// Returns (token, source).
fn resolve_api_key_provider_secret(provider_id: &str, pconfig: &ProviderConfig) -> (String, String) {
    if provider_id == "copilot" {
        // The dedicated copilot auth module performs token validation in the
        // wider binary (crate::auth_cmd / hermes-core auth). Here we surface
        // the raw token from the env chain as a best-effort fallback.
        for env_var in pconfig.api_key_env_vars {
            let val = get_env_value(env_var);
            let val = val.trim();
            if has_usable_secret(val) {
                return (val.to_string(), env_var.to_string());
            }
        }
        return (String::new(), String::new());
    }

    for env_var in pconfig.api_key_env_vars {
        let val = get_env_value(env_var);
        let val = val.trim();
        if has_usable_secret(val) {
            return (val.to_string(), env_var.to_string());
        }
    }

    // Fallback: credential pool stored in auth.json.
    let pool = read_credential_pool(Some(provider_id));
    if let Some(entries) = pool.as_array() {
        if let Some(entry) = entries.first() {
            let key = entry
                .get("access_token")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .or_else(|| entry.get("runtime_api_key").and_then(|v| v.as_str()))
                .unwrap_or("")
                .trim()
                .to_string();
            if has_usable_secret(&key) {
                return (key, format!("credential_pool:{}", provider_id));
            }
        }
    }

    (String::new(), String::new())
}

// =============================================================================
// Qwen CLI OAuth (~/.qwen/oauth_creds.json)
// =============================================================================

fn qwen_cli_auth_path() -> PathBuf {
    home_dir().join(".qwen").join("oauth_creds.json")
}

fn read_qwen_cli_tokens() -> Result<Value, AuthError> {
    let auth_path = qwen_cli_auth_path();
    if !auth_path.exists() {
        return Err(AuthError::new(
            "Qwen CLI credentials not found. Run 'qwen auth qwen-oauth' first.",
        )
        .with_provider("qwen-oauth")
        .with_code("qwen_auth_missing"));
    }
    let text = fs::read_to_string(&auth_path).map_err(|e| {
        AuthError::new(format!(
            "Failed to read Qwen CLI credentials from {}: {}",
            auth_path.display(),
            e
        ))
        .with_provider("qwen-oauth")
        .with_code("qwen_auth_read_failed")
    })?;
    let data: Value = serde_json::from_str(&text).map_err(|e| {
        AuthError::new(format!(
            "Failed to read Qwen CLI credentials from {}: {}",
            auth_path.display(),
            e
        ))
        .with_provider("qwen-oauth")
        .with_code("qwen_auth_read_failed")
    })?;
    if !data.is_object() {
        return Err(AuthError::new(format!(
            "Invalid Qwen CLI credentials in {}.",
            auth_path.display()
        ))
        .with_provider("qwen-oauth")
        .with_code("qwen_auth_invalid"));
    }
    Ok(data)
}

fn save_qwen_cli_tokens(tokens: &Value) -> std::io::Result<PathBuf> {
    let auth_path = qwen_cli_auth_path();
    if let Some(parent) = auth_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = auth_path.with_extension("tmp");
    // sort_keys=True parity: serialize via a BTreeMap.
    let sorted = sort_json_keys(tokens);
    let mut body = serde_json::to_string_pretty(&sorted).unwrap_or_default();
    body.push('\n');
    fs::write(&tmp_path, body)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600));
    }
    fs::rename(&tmp_path, &auth_path)?;
    Ok(auth_path)
}

/// Recursively sort object keys (parity with json.dumps(sort_keys=True)).
fn sort_json_keys(value: &Value) -> Value {
    match value {
        Value::Object(m) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in m {
                sorted.insert(k.clone(), sort_json_keys(v));
            }
            let mut out = Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sort_json_keys).collect()),
        other => other.clone(),
    }
}

fn qwen_access_token_is_expiring(expiry_date_ms: Option<&Value>, skew_seconds: i64) -> bool {
    let expiry_ms = match expiry_date_ms {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
        Some(Value::String(s)) => match s.trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => return true,
        },
        _ => return true,
    };
    ((now_unix() + skew_seconds.max(0) as f64) * 1000.0) as i64 >= expiry_ms
}

fn refresh_qwen_cli_tokens(tokens: &Value, timeout_seconds: f64) -> Result<Value, AuthError> {
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if refresh_token.is_empty() {
        return Err(AuthError::new(
            "Qwen OAuth refresh token missing. Re-run 'qwen auth qwen-oauth'.",
        )
        .with_provider("qwen-oauth")
        .with_code("qwen_refresh_token_missing"));
    }

    let client = http_client(timeout_seconds);
    let resp = client
        .post(QWEN_OAUTH_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.as_str()),
            ("client_id", QWEN_OAUTH_CLIENT_ID),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("Qwen OAuth refresh failed: {}", e))
                .with_provider("qwen-oauth")
                .with_code("qwen_refresh_failed")
        })?;

    let status = resp.status().as_u16();
    let body_text = resp.text().unwrap_or_default();
    if status >= 400 {
        let body = body_text.trim();
        let mut msg = "Qwen OAuth refresh failed. Re-run 'qwen auth qwen-oauth'.".to_string();
        if !body.is_empty() {
            msg.push_str(&format!(" Response: {}", body));
        }
        return Err(AuthError::new(msg)
            .with_provider("qwen-oauth")
            .with_code("qwen_refresh_failed"));
    }

    let payload: Value = serde_json::from_str(&body_text).map_err(|e| {
        AuthError::new(format!("Qwen OAuth refresh returned invalid JSON: {}", e))
            .with_provider("qwen-oauth")
            .with_code("qwen_refresh_invalid_json")
    })?;
    let access_token = payload
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if !payload.is_object() || access_token.is_empty() {
        return Err(AuthError::new("Qwen OAuth refresh response missing access_token.")
            .with_provider("qwen-oauth")
            .with_code("qwen_refresh_invalid_response"));
    }

    let expires_in_seconds = match payload.get("expires_in") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(6 * 60 * 60),
        Some(Value::String(s)) => s.trim().parse::<i64>().unwrap_or(6 * 60 * 60),
        _ => 6 * 60 * 60,
    };

    let prev_token_type = tokens
        .get("token_type")
        .and_then(|v| v.as_str())
        .unwrap_or("Bearer");
    let prev_resource = tokens
        .get("resource_url")
        .and_then(|v| v.as_str())
        .unwrap_or("portal.qwen.ai");

    let next_refresh = payload
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&refresh_token)
        .trim()
        .to_string();
    let token_type = {
        let t = payload
            .get("token_type")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(prev_token_type)
            .trim();
        if t.is_empty() { "Bearer" } else { t }.to_string()
    };
    let resource_url = {
        let r = payload
            .get("resource_url")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(prev_resource)
            .trim();
        if r.is_empty() { "portal.qwen.ai" } else { r }.to_string()
    };
    let refreshed = json!({
        "access_token": access_token,
        "refresh_token": next_refresh,
        "token_type": token_type,
        "resource_url": resource_url,
        "expiry_date": (now_unix() * 1000.0) as i64 + expires_in_seconds.max(1) * 1000,
    });
    let _ = save_qwen_cli_tokens(&refreshed);
    Ok(refreshed)
}

/// Resolve runtime credentials from the Qwen CLI token store.
pub fn resolve_qwen_runtime_credentials(
    force_refresh: bool,
    refresh_if_expiring: bool,
    refresh_skew_seconds: i64,
) -> Result<Value, AuthError> {
    let mut tokens = read_qwen_cli_tokens()?;
    let mut access_token = tokens
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let mut should_refresh = force_refresh;
    if !should_refresh && refresh_if_expiring {
        should_refresh = qwen_access_token_is_expiring(tokens.get("expiry_date"), refresh_skew_seconds);
    }
    if should_refresh {
        tokens = refresh_qwen_cli_tokens(&tokens, 20.0)?;
        access_token = tokens
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
    }
    if access_token.is_empty() {
        return Err(AuthError::new(
            "Qwen OAuth access token missing. Re-run 'qwen auth qwen-oauth'.",
        )
        .with_provider("qwen-oauth")
        .with_code("qwen_access_token_missing"));
    }
    let base_url = {
        let env = std::env::var("HERMES_QWEN_BASE_URL")
            .unwrap_or_default()
            .trim()
            .trim_end_matches('/')
            .to_string();
        if env.is_empty() {
            DEFAULT_QWEN_BASE_URL.to_string()
        } else {
            env
        }
    };
    Ok(json!({
        "provider": "qwen-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "qwen-cli",
        "expires_at_ms": tokens.get("expiry_date").cloned().unwrap_or(Value::Null),
        "auth_file": qwen_cli_auth_path().to_string_lossy(),
    }))
}

pub fn get_qwen_auth_status() -> Value {
    let auth_path = qwen_cli_auth_path();
    match resolve_qwen_runtime_credentials(false, false, QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
        Ok(creds) => json!({
            "logged_in": true,
            "auth_file": auth_path.to_string_lossy(),
            "source": creds.get("source").cloned().unwrap_or(Value::Null),
            "api_key": creds.get("api_key").cloned().unwrap_or(Value::Null),
            "expires_at_ms": creds.get("expires_at_ms").cloned().unwrap_or(Value::Null),
        }),
        Err(exc) => json!({
            "logged_in": false,
            "auth_file": auth_path.to_string_lossy(),
            "error": exc.to_string(),
        }),
    }
}

// =============================================================================
// OpenAI Codex auth — tokens stored in ~/.hermes/auth.json
// =============================================================================

struct CodexTokenData {
    tokens: Value,
    last_refresh: Value,
}

fn read_codex_tokens(do_lock: bool) -> Result<CodexTokenData, AuthError> {
    let auth_store = if do_lock {
        let _guard = auth_store_lock();
        load_auth_store()
    } else {
        load_auth_store()
    };
    let state = load_provider_state(&auth_store, "openai-codex").ok_or_else(|| {
        AuthError::new("No Codex credentials stored. Run `hermes auth` to authenticate.")
            .with_provider("openai-codex")
            .with_code("codex_auth_missing")
            .relogin()
    })?;
    let tokens = state.get("tokens").cloned().filter(|v| v.is_object()).ok_or_else(|| {
        AuthError::new("Codex auth state is missing tokens. Run `hermes auth` to re-authenticate.")
            .with_provider("openai-codex")
            .with_code("codex_auth_invalid_shape")
            .relogin()
    })?;
    let access = tokens.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    if access.trim().is_empty() {
        return Err(AuthError::new(
            "Codex auth is missing access_token. Run `hermes auth` to re-authenticate.",
        )
        .with_provider("openai-codex")
        .with_code("codex_auth_missing_access_token")
        .relogin());
    }
    let refresh = tokens.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("");
    if refresh.trim().is_empty() {
        return Err(AuthError::new(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
        )
        .with_provider("openai-codex")
        .with_code("codex_auth_missing_refresh_token")
        .relogin());
    }
    Ok(CodexTokenData {
        tokens,
        last_refresh: state.get("last_refresh").cloned().unwrap_or(Value::Null),
    })
}

fn save_codex_tokens(tokens: &Value, last_refresh: Option<&str>) {
    let last = last_refresh
        .map(|s| s.to_string())
        .unwrap_or_else(utc_now_iso_z);
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let mut state = load_provider_state(&auth_store, "openai-codex").unwrap_or_else(|| json!({}));
    if let Some(obj) = state.as_object_mut() {
        obj.insert("tokens".to_string(), tokens.clone());
        obj.insert("last_refresh".to_string(), json!(last));
        obj.insert("auth_mode".to_string(), json!("chatgpt"));
    }
    save_provider_state(&mut auth_store, "openai-codex", state);
    let _ = save_auth_store(&mut auth_store);
}

/// Refresh Codex OAuth tokens without mutating Hermes auth state.
/// Returns {access_token, refresh_token, last_refresh}.
pub fn refresh_codex_oauth_pure(
    _access_token: &str,
    refresh_token: &str,
    timeout_seconds: f64,
) -> Result<Value, AuthError> {
    if refresh_token.trim().is_empty() {
        return Err(AuthError::new(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
        )
        .with_provider("openai-codex")
        .with_code("codex_auth_missing_refresh_token")
        .relogin());
    }

    let client = http_client(timeout_seconds.max(5.0));
    let resp = client
        .post(CODEX_OAUTH_TOKEN_URL)
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.trim()),
            ("client_id", CODEX_OAUTH_CLIENT_ID),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("Codex token refresh failed: {}", e))
                .with_provider("openai-codex")
                .with_code("codex_refresh_failed")
        })?;

    let status = resp.status().as_u16();
    let body_text = resp.text().unwrap_or_default();
    if status != 200 {
        let mut code = "codex_refresh_failed".to_string();
        let mut message = format!("Codex token refresh failed with status {}.", status);
        let mut relogin_required = false;
        if let Ok(err) = serde_json::from_str::<Value>(&body_text) {
            if let Some(err_obj) = err.get("error") {
                if err_obj.is_object() {
                    let nested_code = err_obj
                        .get("code")
                        .and_then(|v| v.as_str())
                        .or_else(|| err_obj.get("type").and_then(|v| v.as_str()));
                    if let Some(nc) = nested_code {
                        if !nc.trim().is_empty() {
                            code = nc.trim().to_string();
                        }
                    }
                    if let Some(nm) = err_obj.get("message").and_then(|v| v.as_str()) {
                        if !nm.trim().is_empty() {
                            message = format!("Codex token refresh failed: {}", nm.trim());
                        }
                    }
                } else if let Some(s) = err_obj.as_str() {
                    if !s.trim().is_empty() {
                        code = s.trim().to_string();
                        let desc = err
                            .get("error_description")
                            .and_then(|v| v.as_str())
                            .or_else(|| err.get("message").and_then(|v| v.as_str()));
                        if let Some(d) = desc {
                            if !d.trim().is_empty() {
                                message = format!("Codex token refresh failed: {}", d.trim());
                            }
                        }
                    }
                }
            }
        }
        if matches!(code.as_str(), "invalid_grant" | "invalid_token" | "invalid_request") {
            relogin_required = true;
        }
        if code == "refresh_token_reused" {
            message = "Codex refresh token was already consumed by another client \
                       (e.g. Codex CLI or VS Code extension). \
                       Run `codex` in your terminal to generate fresh tokens, \
                       then run `hermes auth` to re-authenticate."
                .to_string();
            relogin_required = true;
        }
        if (status == 401 || status == 403) && !relogin_required {
            relogin_required = true;
        }
        let mut err = AuthError::new(message)
            .with_provider("openai-codex")
            .with_code(code);
        if relogin_required {
            err = err.relogin();
        }
        return Err(err);
    }

    let payload: Value = serde_json::from_str(&body_text).map_err(|_| {
        AuthError::new("Codex token refresh returned invalid JSON.")
            .with_provider("openai-codex")
            .with_code("codex_refresh_invalid_json")
            .relogin()
    })?;

    let refreshed_access = payload.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    if refreshed_access.trim().is_empty() {
        return Err(AuthError::new("Codex token refresh response was missing access_token.")
            .with_provider("openai-codex")
            .with_code("codex_refresh_missing_access_token")
            .relogin());
    }

    let mut next_refresh = refresh_token.trim().to_string();
    if let Some(nr) = payload.get("refresh_token").and_then(|v| v.as_str()) {
        if !nr.trim().is_empty() {
            next_refresh = nr.trim().to_string();
        }
    }

    Ok(json!({
        "access_token": refreshed_access.trim(),
        "refresh_token": next_refresh,
        "last_refresh": utc_now_iso_z(),
    }))
}

fn refresh_codex_auth_tokens(tokens: &Value, timeout_seconds: f64) -> Result<Value, AuthError> {
    let access = tokens.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    let refresh = tokens.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("");
    let refreshed = refresh_codex_oauth_pure(access, refresh, timeout_seconds)?;
    let mut updated = tokens.clone();
    if let Some(obj) = updated.as_object_mut() {
        obj.insert(
            "access_token".to_string(),
            refreshed.get("access_token").cloned().unwrap_or(Value::Null),
        );
        obj.insert(
            "refresh_token".to_string(),
            refreshed.get("refresh_token").cloned().unwrap_or(Value::Null),
        );
    }
    save_codex_tokens(&updated, None);
    Ok(updated)
}

/// Try to read tokens from ~/.codex/auth.json (Codex CLI shared file).
fn import_codex_cli_tokens() -> Option<Value> {
    let codex_home = std::env::var("CODEX_HOME").unwrap_or_default();
    let codex_home = codex_home.trim();
    let base = if codex_home.is_empty() {
        home_dir().join(".codex")
    } else {
        PathBuf::from(codex_home)
    };
    let auth_path = base.join("auth.json");
    if !auth_path.is_file() {
        return None;
    }
    let payload: Value = serde_json::from_str(&fs::read_to_string(&auth_path).ok()?).ok()?;
    let tokens = payload.get("tokens")?.clone();
    if !tokens.is_object() {
        return None;
    }
    let access = tokens.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    let refresh = tokens.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("");
    if access.is_empty() || refresh.is_empty() {
        return None;
    }
    if codex_access_token_is_expiring(access, 0) {
        return None;
    }
    Some(tokens)
}

/// Resolve runtime credentials from Hermes's own Codex token store.
pub fn resolve_codex_runtime_credentials(
    force_refresh: bool,
    refresh_if_expiring: bool,
    refresh_skew_seconds: i64,
) -> Result<Value, AuthError> {
    let mut data = read_codex_tokens(true)?;
    let mut tokens = data.tokens.clone();
    let mut access_token = tokens
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_timeout_seconds = std::env::var("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(20.0);

    let mut should_refresh = force_refresh;
    if !should_refresh && refresh_if_expiring {
        should_refresh = codex_access_token_is_expiring(&access_token, refresh_skew_seconds);
    }
    if should_refresh {
        let _guard =
            auth_store_lock_timeout(AUTH_LOCK_TIMEOUT_SECONDS.max(refresh_timeout_seconds + 5.0));
        data = read_codex_tokens(false)?;
        tokens = data.tokens.clone();
        access_token = tokens
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let mut should2 = force_refresh;
        if !should2 && refresh_if_expiring {
            should2 = codex_access_token_is_expiring(&access_token, refresh_skew_seconds);
        }
        if should2 {
            tokens = refresh_codex_auth_tokens(&tokens, refresh_timeout_seconds)?;
            access_token = tokens
                .get("access_token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
        }
    }

    let base_url = {
        let env = std::env::var("HERMES_CODEX_BASE_URL")
            .unwrap_or_default()
            .trim()
            .trim_end_matches('/')
            .to_string();
        if env.is_empty() {
            DEFAULT_CODEX_BASE_URL.to_string()
        } else {
            env
        }
    };

    Ok(json!({
        "provider": "openai-codex",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.last_refresh,
        "auth_mode": "chatgpt",
    }))
}

// =============================================================================
// SSH / remote session detection
// =============================================================================

fn is_remote_session() -> bool {
    !std::env::var("SSH_CLIENT").unwrap_or_default().is_empty()
        || !std::env::var("SSH_TTY").unwrap_or_default().is_empty()
}

// =============================================================================
// OAuth Device Code Flow — generic
// =============================================================================

/// POST to the device code endpoint. Returns device_code, user_code, etc.
fn request_device_code(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    client_id: &str,
    scope: Option<&str>,
) -> Result<Value, AuthError> {
    let mut form: Vec<(&str, &str)> = vec![("client_id", client_id)];
    if let Some(s) = scope {
        if !s.is_empty() {
            form.push(("scope", s));
        }
    }
    let resp = client
        .post(format!("{}/api/oauth/device/code", portal_base_url))
        .form(&form)
        .send()
        .map_err(|e| AuthError::new(format!("Device code request failed: {}", e)).with_provider("nous"))?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status >= 400 {
        return Err(AuthError::new(format!(
            "Device code request failed with status {}",
            status
        ))
        .with_provider("nous"));
    }
    let data: Value = serde_json::from_str(&text)
        .map_err(|e| AuthError::new(format!("Device code response invalid JSON: {}", e)).with_provider("nous"))?;
    let required = [
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    ];
    let missing: Vec<&str> = required
        .iter()
        .filter(|f| data.get(*f).is_none())
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(AuthError::new(format!(
            "Device code response missing fields: {}",
            missing.join(", ")
        ))
        .with_provider("nous"));
    }
    Ok(data)
}

/// Poll the token endpoint until the user approves or the code expires.
fn poll_for_token(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    client_id: &str,
    device_code: &str,
    expires_in: i64,
    poll_interval: i64,
) -> Result<Value, AuthError> {
    let deadline = now_unix() + expires_in.max(1) as f64;
    let mut current_interval = poll_interval
        .max(1)
        .min(DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)
        .max(1);

    while now_unix() < deadline {
        let resp = client
            .post(format!("{}/api/oauth/token", portal_base_url))
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", client_id),
                ("device_code", device_code),
            ])
            .send()
            .map_err(|e| AuthError::new(format!("Token poll request failed: {}", e)).with_provider("nous"))?;
        let status = resp.status().as_u16();
        let text = resp.text().unwrap_or_default();
        if status == 200 {
            let payload: Value = serde_json::from_str(&text)
                .map_err(|e| AuthError::new(format!("Token response invalid JSON: {}", e)).with_provider("nous"))?;
            if payload.get("access_token").is_none() {
                return Err(AuthError::new("Token response did not include access_token")
                    .with_provider("nous"));
            }
            return Ok(payload);
        }
        let error_payload: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                return Err(AuthError::new(
                    "Token endpoint returned a non-JSON error response",
                )
                .with_provider("nous"))
            }
        };
        let error_code = error_payload.get("error").and_then(|v| v.as_str()).unwrap_or("");
        if error_code == "authorization_pending" {
            std::thread::sleep(Duration::from_secs(current_interval as u64));
            continue;
        }
        if error_code == "slow_down" {
            current_interval = (current_interval + 1).min(30);
            std::thread::sleep(Duration::from_secs(current_interval as u64));
            continue;
        }
        let description = error_payload
            .get("error_description")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown authentication error");
        return Err(AuthError::new(format!("{}: {}", error_code, description)).with_provider("nous"));
    }
    Err(AuthError::new("Timed out waiting for device authorization").with_provider("nous"))
}

// =============================================================================
// Nous Portal — token refresh, agent key minting, model discovery
// =============================================================================

fn nous_shared_auth_dir() -> PathBuf {
    let override_dir = std::env::var("HERMES_SHARED_AUTH_DIR").unwrap_or_default();
    let override_dir = override_dir.trim();
    if !override_dir.is_empty() {
        return PathBuf::from(override_dir);
    }
    home_dir().join(".hermes").join("shared")
}

fn nous_shared_store_path() -> PathBuf {
    nous_shared_auth_dir().join(NOUS_SHARED_STORE_FILENAME)
}

fn write_shared_nous_state(state: &Value) {
    let refresh_token = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("");
    let access_token = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    if refresh_token.trim().is_empty() || access_token.trim().is_empty() {
        return;
    }
    let shared = json!({
        "_schema": 1,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": state.get("token_type").and_then(|v| v.as_str()).unwrap_or("Bearer"),
        "scope": state.get("scope").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_SCOPE),
        "client_id": state.get("client_id").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_CLIENT_ID),
        "portal_base_url": state.get("portal_base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_PORTAL_URL),
        "inference_base_url": state.get("inference_base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_INFERENCE_URL),
        "obtained_at": state.get("obtained_at").cloned().unwrap_or(Value::Null),
        "expires_at": state.get("expires_at").cloned().unwrap_or(Value::Null),
        "updated_at": utc_now_iso(),
    });
    let path = nous_shared_store_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(&sort_json_keys(&shared)).unwrap_or_default();
    if fs::write(&tmp, body).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
        }
        let _ = fs::rename(&tmp, &path);
        oauth_trace(
            "nous_shared_store_written",
            &[
                ("path", json!(path.to_string_lossy())),
                ("refresh_token_fp", json!(token_fingerprint(refresh_token))),
            ],
        );
    }
}

fn read_shared_nous_state() -> Option<Value> {
    let path = nous_shared_store_path();
    if !path.is_file() {
        return None;
    }
    let payload: Value = serde_json::from_str(&fs::read_to_string(&path).ok()?).ok()?;
    if !payload.is_object() {
        return None;
    }
    let refresh_token = payload.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("");
    let access_token = payload.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    if refresh_token.trim().is_empty() || access_token.trim().is_empty() {
        return None;
    }
    Some(payload)
}

fn try_import_shared_nous_state(timeout_seconds: f64, min_key_ttl_seconds: i64) -> Option<Value> {
    let shared = read_shared_nous_state()?;
    let state = json!({
        "access_token": shared.get("access_token").cloned().unwrap_or(Value::Null),
        "refresh_token": shared.get("refresh_token").cloned().unwrap_or(Value::Null),
        "client_id": shared.get("client_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_NOUS_CLIENT_ID),
        "portal_base_url": shared.get("portal_base_url").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_NOUS_PORTAL_URL),
        "inference_base_url": shared.get("inference_base_url").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_NOUS_INFERENCE_URL),
        "token_type": shared.get("token_type").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or("Bearer"),
        "scope": shared.get("scope").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_NOUS_SCOPE),
        "obtained_at": shared.get("obtained_at").cloned().unwrap_or(Value::Null),
        "expires_at": shared.get("expires_at").cloned().unwrap_or(Value::Null),
        "agent_key": Value::Null,
        "agent_key_expires_at": Value::Null,
        "tls": {"insecure": false, "ca_bundle": Value::Null},
    });
    match refresh_nous_oauth_from_state(&state, min_key_ttl_seconds, timeout_seconds, true, true) {
        Ok(refreshed) => Some(refreshed),
        Err(_) => None,
    }
}

fn refresh_access_token(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    client_id: &str,
    refresh_token: &str,
) -> Result<Value, AuthError> {
    let resp = client
        .post(format!("{}/api/oauth/token", portal_base_url))
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", client_id),
            ("refresh_token", refresh_token),
        ])
        .send()
        .map_err(|_| {
            AuthError::new("Refresh token exchange failed")
                .with_provider("nous")
                .relogin()
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status == 200 {
        let payload: Value = serde_json::from_str(&text).map_err(|_| {
            AuthError::new("Refresh token exchange failed")
                .with_provider("nous")
                .relogin()
        })?;
        if payload.get("access_token").is_none() {
            return Err(AuthError::new("Refresh response missing access_token")
                .with_provider("nous")
                .with_code("invalid_token")
                .relogin());
        }
        return Ok(payload);
    }
    let error_payload: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            return Err(AuthError::new("Refresh token exchange failed")
                .with_provider("nous")
                .relogin())
        }
    };
    let code = error_payload
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("invalid_grant")
        .to_string();
    let mut description = error_payload
        .get("error_description")
        .and_then(|v| v.as_str())
        .unwrap_or("Refresh token exchange failed")
        .to_string();
    let relogin = matches!(code.as_str(), "invalid_grant" | "invalid_token");
    let lowered = description.to_lowercase();
    if lowered.contains("reuse") {
        description = "Nous Portal detected refresh-token reuse and revoked this session.\n\
            This usually means an external process (monitoring script, \
            custom self-heal hook, or another Hermes install sharing \
            ~/.hermes/auth.json) called POST /api/oauth/token with Hermes's \
            refresh token without persisting the rotated token back.\n\
            Nous refresh tokens are single-use — only Hermes may call the \
            refresh endpoint. For health checks, use `hermes auth status` \
            instead.\n\
            Re-authenticate with: hermes auth add nous"
            .to_string();
    }
    let mut err = AuthError::new(description).with_provider("nous").with_code(code);
    if relogin {
        err = err.relogin();
    }
    Err(err)
}

fn mint_agent_key(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    access_token: &str,
    min_ttl_seconds: i64,
) -> Result<Value, AuthError> {
    let resp = client
        .post(format!("{}/api/oauth/agent-key", portal_base_url))
        .header("Authorization", format!("Bearer {}", access_token))
        .json(&json!({"min_ttl_seconds": min_ttl_seconds.max(60)}))
        .send()
        .map_err(|_| {
            AuthError::new("Agent key mint request failed")
                .with_provider("nous")
                .with_code("server_error")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status == 200 {
        let payload: Value = serde_json::from_str(&text).map_err(|_| {
            AuthError::new("Agent key mint request failed")
                .with_provider("nous")
                .with_code("server_error")
        })?;
        if payload.get("api_key").is_none() {
            return Err(AuthError::new("Mint response missing api_key")
                .with_provider("nous")
                .with_code("server_error"));
        }
        return Ok(payload);
    }
    let error_payload: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            return Err(AuthError::new("Agent key mint request failed")
                .with_provider("nous")
                .with_code("server_error"))
        }
    };
    let code = error_payload
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("server_error")
        .to_string();
    let description = error_payload
        .get("error_description")
        .and_then(|v| v.as_str())
        .unwrap_or("Agent key mint request failed")
        .to_string();
    let relogin = matches!(code.as_str(), "invalid_token" | "invalid_grant");
    let mut err = AuthError::new(description).with_provider("nous").with_code(code);
    if relogin {
        err = err.relogin();
    }
    Err(err)
}

/// Fetch available model IDs from the Nous inference API.
pub fn fetch_nous_models(
    inference_base_url: &str,
    api_key: &str,
    timeout_seconds: f64,
) -> Result<Vec<String>, AuthError> {
    let client = http_client(timeout_seconds);
    let resp = client
        .get(format!("{}/models", inference_base_url.trim_end_matches('/')))
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .map_err(|e| {
            AuthError::new(format!("/models request failed: {}", e))
                .with_provider("nous")
                .with_code("models_fetch_failed")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status != 200 {
        let mut description = format!("/models request failed with status {}", status);
        if let Ok(err) = serde_json::from_str::<Value>(&text) {
            if let Some(d) = err
                .get("error_description")
                .and_then(|v| v.as_str())
                .or_else(|| err.get("error").and_then(|v| v.as_str()))
            {
                description = d.to_string();
            }
        }
        return Err(AuthError::new(description)
            .with_provider("nous")
            .with_code("models_fetch_failed"));
    }
    let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let data = match payload.get("data").and_then(|v| v.as_array()) {
        Some(d) => d,
        None => return Ok(vec![]),
    };
    let mut model_ids: Vec<String> = Vec::new();
    for item in data {
        if let Some(mid) = item.get("id").and_then(|v| v.as_str()) {
            let mid = mid.trim();
            if mid.is_empty() {
                continue;
            }
            if mid.to_lowercase().contains("hermes") {
                continue;
            }
            model_ids.push(mid.to_string());
        }
    }
    // Sort: opus > pro > (other) > sonnet
    fn priority(mid: &str) -> i32 {
        let low = mid.to_lowercase();
        if low.contains("opus") {
            return 0;
        }
        if low.contains("pro") && !low.contains("sonnet") {
            return 1;
        }
        if low.contains("sonnet") {
            return 3;
        }
        2
    }
    model_ids.sort_by(|a, b| priority(a).cmp(&priority(b)).then_with(|| a.cmp(b)));
    // Dedup preserving order.
    let mut seen = HashSet::new();
    model_ids.retain(|m| seen.insert(m.clone()));
    Ok(model_ids)
}

fn agent_key_is_usable(state: &Value, min_ttl_seconds: i64) -> bool {
    let key = state.get("agent_key").and_then(|v| v.as_str()).unwrap_or("");
    if key.trim().is_empty() {
        return false;
    }
    !is_expiring(state.get("agent_key_expires_at"), min_ttl_seconds)
}

/// Refresh-aware Nous Portal access token for managed tool gateways.
pub fn resolve_nous_access_token(
    timeout_seconds: f64,
    refresh_skew_seconds: i64,
) -> Result<String, AuthError> {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let mut state = load_provider_state(&auth_store, "nous").ok_or_else(|| {
        AuthError::new("Hermes is not logged into Nous Portal.")
            .with_provider("nous")
            .relogin()
    })?;

    let portal_base_url = optional_base_url(state.get("portal_base_url"))
        .or_else(|| env_nonempty("HERMES_PORTAL_BASE_URL"))
        .or_else(|| env_nonempty("NOUS_PORTAL_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_NOUS_PORTAL_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let client_id = state
        .get("client_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_NOUS_CLIENT_ID)
        .to_string();

    let access_token = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    let refresh_token = state
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if access_token.is_empty() {
        return Err(AuthError::new("No access token found for Nous Portal login.")
            .with_provider("nous")
            .relogin());
    }
    if !is_expiring(state.get("expires_at"), refresh_skew_seconds) {
        return Ok(access_token.to_string());
    }
    if refresh_token.is_empty() {
        return Err(AuthError::new("Session expired and no refresh token is available.")
            .with_provider("nous")
            .relogin());
    }

    let client = http_client(if timeout_seconds > 0.0 { timeout_seconds } else { 15.0 });
    let refreshed = refresh_access_token(&client, &portal_base_url, &client_id, &refresh_token)?;
    let now = now_unix();
    let access_ttl = coerce_ttl_seconds(refreshed.get("expires_in"));
    let new_access = refreshed
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if let Some(obj) = state.as_object_mut() {
        obj.insert("access_token".to_string(), json!(new_access));
        obj.insert(
            "refresh_token".to_string(),
            json!(refreshed
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(&refresh_token)),
        );
        obj.insert(
            "token_type".to_string(),
            refreshed
                .get("token_type")
                .cloned()
                .filter(|v| v.as_str().map(|s| !s.is_empty()).unwrap_or(false))
                .or_else(|| obj.get("token_type").cloned())
                .unwrap_or(json!("Bearer")),
        );
        if let Some(scope) = refreshed.get("scope").cloned() {
            obj.insert("scope".to_string(), scope);
        }
        obj.insert("obtained_at".to_string(), json!(iso_from_epoch(now)));
        obj.insert("expires_in".to_string(), json!(access_ttl));
        obj.insert(
            "expires_at".to_string(),
            json!(iso_from_epoch(now + access_ttl as f64)),
        );
        obj.insert("portal_base_url".to_string(), json!(portal_base_url));
        obj.insert("client_id".to_string(), json!(client_id));
        obj.insert("tls".to_string(), json!({"insecure": false, "ca_bundle": Value::Null}));
    }
    save_provider_state(&mut auth_store, "nous", state.clone());
    let _ = save_auth_store(&mut auth_store);
    Ok(new_access)
}

fn env_nonempty(name: &str) -> Option<String> {
    let v = std::env::var(name).unwrap_or_default();
    let v = v.trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// Refresh Nous OAuth state without mutating auth.json.
#[allow(clippy::too_many_arguments)]
pub fn refresh_nous_oauth_pure(
    access_token: &str,
    refresh_token: &str,
    client_id: &str,
    portal_base_url: &str,
    inference_base_url: &str,
    token_type: &str,
    scope: &str,
    obtained_at: Option<&str>,
    expires_at: Option<&str>,
    agent_key: Option<&str>,
    agent_key_expires_at: Option<&str>,
    min_key_ttl_seconds: i64,
    timeout_seconds: f64,
    force_refresh: bool,
    force_mint: bool,
) -> Result<Value, AuthError> {
    let mut state = json!({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": if client_id.is_empty() { DEFAULT_NOUS_CLIENT_ID } else { client_id },
        "portal_base_url": if portal_base_url.is_empty() { DEFAULT_NOUS_PORTAL_URL } else { portal_base_url }.trim_end_matches('/'),
        "inference_base_url": if inference_base_url.is_empty() { DEFAULT_NOUS_INFERENCE_URL } else { inference_base_url }.trim_end_matches('/'),
        "token_type": if token_type.is_empty() { "Bearer" } else { token_type },
        "scope": if scope.is_empty() { DEFAULT_NOUS_SCOPE } else { scope },
        "obtained_at": obtained_at.map(Value::from).unwrap_or(Value::Null),
        "expires_at": expires_at.map(Value::from).unwrap_or(Value::Null),
        "agent_key": agent_key.map(Value::from).unwrap_or(Value::Null),
        "agent_key_expires_at": agent_key_expires_at.map(Value::from).unwrap_or(Value::Null),
        "tls": {"insecure": false, "ca_bundle": Value::Null},
    });

    let client = http_client(if timeout_seconds > 0.0 { timeout_seconds } else { 15.0 });
    let portal = state.get("portal_base_url").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let cid = state.get("client_id").and_then(|v| v.as_str()).unwrap_or("").to_string();

    if force_refresh || is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
        let rt = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let refreshed = refresh_access_token(&client, &portal, &cid, &rt)?;
        let now = now_unix();
        let access_ttl = coerce_ttl_seconds(refreshed.get("expires_in"));
        apply_refresh_to_state(&mut state, &refreshed, &rt, now, access_ttl);
    }

    if force_mint || !agent_key_is_usable(&state, min_key_ttl_seconds.max(60)) {
        let at = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let mint_payload = mint_agent_key(&client, &portal, &at, min_key_ttl_seconds)?;
        apply_mint_to_state(&mut state, &mint_payload, now_unix());
    }

    Ok(state)
}

fn apply_refresh_to_state(state: &mut Value, refreshed: &Value, prev_rt: &str, now: f64, access_ttl: i64) {
    if let Some(obj) = state.as_object_mut() {
        obj.insert(
            "access_token".to_string(),
            refreshed.get("access_token").cloned().unwrap_or(Value::Null),
        );
        obj.insert(
            "refresh_token".to_string(),
            json!(refreshed
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(prev_rt)),
        );
        let tt = refreshed
            .get("token_type")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| obj.get("token_type").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "Bearer".to_string());
        obj.insert("token_type".to_string(), json!(tt));
        if let Some(scope) = refreshed.get("scope").and_then(|v| v.as_str()) {
            obj.insert("scope".to_string(), json!(scope));
        }
        if let Some(url) = optional_base_url(refreshed.get("inference_base_url")) {
            obj.insert("inference_base_url".to_string(), json!(url));
        }
        obj.insert("obtained_at".to_string(), json!(iso_from_epoch(now)));
        obj.insert("expires_in".to_string(), json!(access_ttl));
        obj.insert("expires_at".to_string(), json!(iso_from_epoch(now + access_ttl as f64)));
    }
}

fn apply_mint_to_state(state: &mut Value, mint_payload: &Value, now: f64) {
    if let Some(obj) = state.as_object_mut() {
        obj.insert("agent_key".to_string(), mint_payload.get("api_key").cloned().unwrap_or(Value::Null));
        obj.insert("agent_key_id".to_string(), mint_payload.get("key_id").cloned().unwrap_or(Value::Null));
        obj.insert("agent_key_expires_at".to_string(), mint_payload.get("expires_at").cloned().unwrap_or(Value::Null));
        obj.insert("agent_key_expires_in".to_string(), mint_payload.get("expires_in").cloned().unwrap_or(Value::Null));
        obj.insert("agent_key_reused".to_string(), json!(mint_payload.get("reused").and_then(|v| v.as_bool()).unwrap_or(false)));
        obj.insert("agent_key_obtained_at".to_string(), json!(iso_from_epoch(now)));
        if let Some(url) = optional_base_url(mint_payload.get("inference_base_url")) {
            obj.insert("inference_base_url".to_string(), json!(url));
        }
    }
}

/// Refresh Nous OAuth from a state dict. Thin wrapper around refresh_nous_oauth_pure.
pub fn refresh_nous_oauth_from_state(
    state: &Value,
    min_key_ttl_seconds: i64,
    timeout_seconds: f64,
    force_refresh: bool,
    force_mint: bool,
) -> Result<Value, AuthError> {
    refresh_nous_oauth_pure(
        state.get("access_token").and_then(|v| v.as_str()).unwrap_or(""),
        state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or(""),
        state.get("client_id").and_then(|v| v.as_str()).unwrap_or("hermes-cli"),
        state.get("portal_base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_INFERENCE_URL),
        state.get("token_type").and_then(|v| v.as_str()).unwrap_or("Bearer"),
        state.get("scope").and_then(|v| v.as_str()).unwrap_or(DEFAULT_NOUS_SCOPE),
        state.get("obtained_at").and_then(|v| v.as_str()),
        state.get("expires_at").and_then(|v| v.as_str()),
        state.get("agent_key").and_then(|v| v.as_str()),
        state.get("agent_key_expires_at").and_then(|v| v.as_str()),
        min_key_ttl_seconds,
        timeout_seconds,
        force_refresh,
        force_mint,
    )
}

/// Persist minted Nous OAuth credentials as the singleton provider state.
pub fn persist_nous_credentials(creds: &Value, label: Option<&str>) {
    let mut state = creds.clone();
    if let Some(l) = label {
        if !l.trim().is_empty() {
            if let Some(obj) = state.as_object_mut() {
                obj.insert("label".to_string(), json!(l.trim()));
            }
        }
    }
    {
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        save_provider_state(&mut auth_store, "nous", state.clone());
        let _ = save_auth_store(&mut auth_store);
    }
    write_shared_nous_state(&state);
}

/// Resolve Nous inference credentials for runtime use. Ensures the access token
/// is valid (refreshing if needed) and a short-lived inference key is present.
pub fn resolve_nous_runtime_credentials(
    min_key_ttl_seconds: i64,
    timeout_seconds: f64,
    force_mint: bool,
) -> Result<Value, AuthError> {
    let min_key_ttl_seconds = min_key_ttl_seconds.max(60);
    let sequence_id = uuid_hex()[..12].to_string();

    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    let mut state = load_provider_state(&auth_store, "nous").ok_or_else(|| {
        AuthError::new("Hermes is not logged into Nous Portal.")
            .with_provider("nous")
            .relogin()
    })?;

    let portal_base_url = optional_base_url(state.get("portal_base_url"))
        .or_else(|| env_nonempty("HERMES_PORTAL_BASE_URL"))
        .or_else(|| env_nonempty("NOUS_PORTAL_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_NOUS_PORTAL_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let mut inference_base_url = optional_base_url(state.get("inference_base_url"))
        .or_else(|| env_nonempty("NOUS_INFERENCE_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_NOUS_INFERENCE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let client_id = state
        .get("client_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_NOUS_CLIENT_ID)
        .to_string();

    let client = http_client(if timeout_seconds > 0.0 { timeout_seconds } else { 15.0 });
    oauth_trace(
        "nous_runtime_credentials_start",
        &[
            ("sequence_id", json!(sequence_id)),
            ("force_mint", json!(force_mint)),
            ("min_key_ttl_seconds", json!(min_key_ttl_seconds)),
        ],
    );

    let access_token = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if access_token.is_empty() {
        return Err(AuthError::new("No access token found for Nous Portal login.")
            .with_provider("nous")
            .relogin());
    }

    // Step 1: refresh access token if expiring.
    if is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
        let refresh_token = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if refresh_token.is_empty() {
            return Err(AuthError::new("Session expired and no refresh token is available.")
                .with_provider("nous")
                .relogin());
        }
        let refreshed = refresh_access_token(&client, &portal_base_url, &client_id, &refresh_token)?;
        let now = now_unix();
        let access_ttl = coerce_ttl_seconds(refreshed.get("expires_in"));
        if let Some(url) = optional_base_url(refreshed.get("inference_base_url")) {
            inference_base_url = url;
        }
        apply_refresh_to_state(&mut state, &refreshed, &refresh_token, now, access_ttl);
        save_provider_state(&mut auth_store, "nous", state.clone());
        let _ = save_auth_store(&mut auth_store);
        write_shared_nous_state(&state);
    }

    // Step 2: mint agent key if missing/expiring.
    let mut used_cached_key = false;
    let mut mint_payload: Option<Value> = None;

    if !force_mint && agent_key_is_usable(&state, min_key_ttl_seconds) {
        used_cached_key = true;
        oauth_trace("agent_key_reuse", &[("sequence_id", json!(sequence_id))]);
    } else {
        let access = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        match mint_agent_key(&client, &portal_base_url, &access, min_key_ttl_seconds) {
            Ok(p) => mint_payload = Some(p),
            Err(exc) => {
                let latest_rt = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let retryable = exc
                    .code
                    .as_deref()
                    .map(|c| c == "invalid_token" || c == "invalid_grant")
                    .unwrap_or(false);
                if retryable && !latest_rt.is_empty() {
                    let refreshed = refresh_access_token(&client, &portal_base_url, &client_id, &latest_rt)?;
                    let now = now_unix();
                    let access_ttl = coerce_ttl_seconds(refreshed.get("expires_in"));
                    if let Some(url) = optional_base_url(refreshed.get("inference_base_url")) {
                        inference_base_url = url;
                    }
                    apply_refresh_to_state(&mut state, &refreshed, &latest_rt, now, access_ttl);
                    save_provider_state(&mut auth_store, "nous", state.clone());
                    let _ = save_auth_store(&mut auth_store);
                    write_shared_nous_state(&state);
                    let access2 = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    mint_payload = Some(mint_agent_key(&client, &portal_base_url, &access2, min_key_ttl_seconds)?);
                } else {
                    return Err(exc);
                }
            }
        }
    }

    if let Some(ref payload) = mint_payload {
        apply_mint_to_state(&mut state, payload, now_unix());
        if let Some(url) = optional_base_url(payload.get("inference_base_url")) {
            inference_base_url = url;
        }
        oauth_trace(
            "mint_success",
            &[
                ("sequence_id", json!(sequence_id)),
                ("reused", json!(payload.get("reused").and_then(|v| v.as_bool()).unwrap_or(false))),
            ],
        );
    }

    if let Some(obj) = state.as_object_mut() {
        obj.insert("portal_base_url".to_string(), json!(portal_base_url));
        obj.insert("inference_base_url".to_string(), json!(inference_base_url.clone()));
        obj.insert("client_id".to_string(), json!(client_id));
        obj.insert("tls".to_string(), json!({"insecure": false, "ca_bundle": Value::Null}));
    }
    save_provider_state(&mut auth_store, "nous", state.clone());
    let _ = save_auth_store(&mut auth_store);
    write_shared_nous_state(&state);

    let api_key = state.get("agent_key").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if api_key.is_empty() {
        return Err(AuthError::new("Failed to resolve a Nous inference API key")
            .with_provider("nous")
            .with_code("server_error"));
    }
    let expires_at = state.get("agent_key_expires_at").cloned().unwrap_or(Value::Null);
    let expires_in = match parse_iso_opt(state.get("agent_key_expires_at")) {
        Some(epoch) => (epoch - now_unix()).max(0.0) as i64,
        None => coerce_ttl_seconds(state.get("agent_key_expires_in")),
    };

    Ok(json!({
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id").cloned().unwrap_or(Value::Null),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": if used_cached_key { "cache" } else { "portal" },
    }))
}

// =============================================================================
// Status helpers
// =============================================================================

fn empty_nous_auth_status() -> Value {
    json!({
        "logged_in": false,
        "portal_base_url": Value::Null,
        "inference_base_url": Value::Null,
        "access_expires_at": Value::Null,
        "agent_key_expires_at": Value::Null,
        "has_refresh_token": false,
    })
}

/// Status snapshot for Nous auth.
pub fn get_nous_auth_status() -> Value {
    let state = match get_provider_auth_state("nous") {
        Some(s) => s,
        None => return empty_nous_auth_status(),
    };
    let mut base_status = json!({
        "logged_in": state.get("access_token").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false),
        "portal_base_url": state.get("portal_base_url").cloned().unwrap_or(Value::Null),
        "inference_base_url": state.get("inference_base_url").cloned().unwrap_or(Value::Null),
        "access_expires_at": state.get("expires_at").cloned().unwrap_or(Value::Null),
        "agent_key_expires_at": state.get("agent_key_expires_at").cloned().unwrap_or(Value::Null),
        "has_refresh_token": state.get("refresh_token").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false),
        "access_token": state.get("access_token").cloned().unwrap_or(Value::Null),
        "source": "auth_store",
    });
    match resolve_nous_runtime_credentials(60, 15.0, false) {
        Ok(creds) => {
            let refreshed_state = get_provider_auth_state("nous").unwrap_or_else(|| state.clone());
            if let Some(obj) = base_status.as_object_mut() {
                obj.insert("logged_in".to_string(), json!(true));
                obj.insert(
                    "portal_base_url".to_string(),
                    refreshed_state.get("portal_base_url").cloned().unwrap_or(Value::Null),
                );
                obj.insert(
                    "inference_base_url".to_string(),
                    creds.get("base_url").cloned().filter(|v| !v.is_null())
                        .or_else(|| refreshed_state.get("inference_base_url").cloned())
                        .unwrap_or(Value::Null),
                );
                obj.insert(
                    "access_expires_at".to_string(),
                    refreshed_state.get("expires_at").cloned().unwrap_or(Value::Null),
                );
                obj.insert(
                    "agent_key_expires_at".to_string(),
                    creds.get("expires_at").cloned().filter(|v| !v.is_null())
                        .or_else(|| refreshed_state.get("agent_key_expires_at").cloned())
                        .unwrap_or(Value::Null),
                );
                obj.insert(
                    "has_refresh_token".to_string(),
                    json!(refreshed_state.get("refresh_token").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false)),
                );
                obj.insert(
                    "source".to_string(),
                    json!(format!("runtime:{}", creds.get("source").and_then(|v| v.as_str()).unwrap_or("portal"))),
                );
                obj.insert("key_id".to_string(), creds.get("key_id").cloned().unwrap_or(Value::Null));
            }
            base_status
        }
        Err(exc) => {
            if let Some(obj) = base_status.as_object_mut() {
                obj.insert("logged_in".to_string(), json!(false));
                obj.insert("error".to_string(), json!(exc.to_string()));
                obj.insert("relogin_required".to_string(), json!(exc.relogin_required));
                obj.insert("error_code".to_string(), json!(exc.code.clone()));
            }
            base_status
        }
    }
}

/// Status snapshot for Codex auth.
pub fn get_codex_auth_status() -> Value {
    match resolve_codex_runtime_credentials(false, true, CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
        Ok(creds) => json!({
            "logged_in": true,
            "auth_store": auth_file_path().to_string_lossy(),
            "last_refresh": creds.get("last_refresh").cloned().unwrap_or(Value::Null),
            "auth_mode": creds.get("auth_mode").cloned().unwrap_or(Value::Null),
            "source": creds.get("source").cloned().unwrap_or(Value::Null),
            "api_key": creds.get("api_key").cloned().unwrap_or(Value::Null),
        }),
        Err(exc) => json!({
            "logged_in": false,
            "auth_store": auth_file_path().to_string_lossy(),
            "error": exc.to_string(),
        }),
    }
}

/// Status snapshot for API-key providers (z.ai, Kimi, MiniMax, etc.).
pub fn get_api_key_provider_status(provider_id: &str) -> Value {
    let pconfig = match get_provider_config(provider_id) {
        Some(pc) if pc.auth_type == "api_key" => pc,
        _ => return json!({"configured": false}),
    };
    let (api_key, key_source) = resolve_api_key_provider_secret(provider_id, pconfig);
    let env_url = if !pconfig.base_url_env_var.is_empty() {
        std::env::var(pconfig.base_url_env_var).unwrap_or_default().trim().to_string()
    } else {
        String::new()
    };
    let base_url = if provider_id == "kimi-coding" || provider_id == "kimi-coding-cn" {
        resolve_kimi_base_url(&api_key, pconfig.inference_base_url, &env_url)
    } else if !env_url.is_empty() {
        env_url
    } else {
        pconfig.inference_base_url.to_string()
    };
    json!({
        "configured": !api_key.is_empty(),
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source,
        "base_url": base_url,
        "logged_in": !api_key.is_empty(),
    })
}

/// Status snapshot for providers that run a local subprocess (copilot-acp).
pub fn get_external_process_provider_status(provider_id: &str) -> Value {
    let pconfig = match get_provider_config(provider_id) {
        Some(pc) if pc.auth_type == "external_process" => pc,
        _ => return json!({"configured": false}),
    };
    let command = env_nonempty("HERMES_COPILOT_ACP_COMMAND")
        .or_else(|| env_nonempty("COPILOT_CLI_PATH"))
        .unwrap_or_else(|| "copilot".to_string());
    let raw_args = std::env::var("HERMES_COPILOT_ACP_ARGS").unwrap_or_default();
    let raw_args = raw_args.trim();
    let args: Vec<String> = if raw_args.is_empty() {
        vec!["--acp".to_string(), "--stdio".to_string()]
    } else {
        shlex_split(raw_args)
    };
    let mut base_url = if !pconfig.base_url_env_var.is_empty() {
        std::env::var(pconfig.base_url_env_var).unwrap_or_default().trim().to_string()
    } else {
        String::new()
    };
    if base_url.is_empty() {
        base_url = pconfig.inference_base_url.to_string();
    }
    let resolved_command = which(&command);
    let configured = resolved_command.is_some() || base_url.starts_with("acp+tcp://");
    json!({
        "configured": configured,
        "provider": provider_id,
        "name": pconfig.name,
        "command": command,
        "args": args,
        "resolved_command": resolved_command,
        "base_url": base_url,
        "logged_in": configured,
    })
}

/// Minimal shlex split (POSIX-ish) for command-arg parsing.
fn shlex_split(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = input.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut has_token = false;
    while let Some(c) = chars.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                cur.push(c);
            }
        } else if in_double {
            if c == '"' {
                in_double = false;
            } else if c == '\\' {
                if let Some(&next) = chars.peek() {
                    if next == '"' || next == '\\' {
                        cur.push(chars.next().unwrap());
                        continue;
                    }
                }
                cur.push(c);
            } else {
                cur.push(c);
            }
        } else {
            match c {
                '\'' => {
                    in_single = true;
                    has_token = true;
                }
                '"' => {
                    in_double = true;
                    has_token = true;
                }
                '\\' => {
                    if let Some(next) = chars.next() {
                        cur.push(next);
                        has_token = true;
                    }
                }
                c if c.is_whitespace() => {
                    if has_token {
                        out.push(std::mem::take(&mut cur));
                        has_token = false;
                    }
                }
                _ => {
                    cur.push(c);
                    has_token = true;
                }
            }
        }
    }
    if has_token {
        out.push(cur);
    }
    out
}

/// Resolve an executable on PATH (shutil.which equivalent).
fn which(command: &str) -> Option<String> {
    if command.is_empty() {
        return None;
    }
    let candidate = Path::new(command);
    if candidate.is_absolute() || command.contains('/') {
        if candidate.is_file() {
            return Some(command.to_string());
        }
        return None;
    }
    let path_var = std::env::var("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        let full = dir.join(command);
        if full.is_file() {
            return Some(full.to_string_lossy().to_string());
        }
    }
    None
}

/// Generic auth status dispatcher.
pub fn get_auth_status(provider_id: Option<&str>) -> Value {
    let target = provider_id
        .map(|s| s.to_string())
        .or_else(get_active_provider)
        .unwrap_or_default();
    match target.as_str() {
        "spotify" => return get_spotify_auth_status(),
        "nous" => return get_nous_auth_status(),
        "openai-codex" => return get_codex_auth_status(),
        "qwen-oauth" => return get_qwen_auth_status(),
        "google-gemini-cli" => return get_gemini_oauth_auth_status(),
        "minimax-oauth" => return get_minimax_oauth_auth_status(),
        "copilot-acp" => return get_external_process_provider_status(&target),
        _ => {}
    }
    if let Some(pc) = get_provider_config(&target) {
        if pc.auth_type == "api_key" {
            return get_api_key_provider_status(&target);
        }
        if pc.auth_type == "aws_sdk" {
            // boto3 credential chain not available natively; report unconfigured.
            return json!({"logged_in": false, "provider": target, "error": "boto3 not installed"});
        }
    }
    json!({"logged_in": false})
}

/// Resolve API key and base URL for an API-key provider.
pub fn resolve_api_key_provider_credentials(provider_id: &str) -> Result<Value, AuthError> {
    let pconfig = match get_provider_config(provider_id) {
        Some(pc) if pc.auth_type == "api_key" => pc,
        _ => {
            return Err(AuthError::new(format!(
                "Provider '{}' is not an API-key provider.",
                provider_id
            ))
            .with_provider(provider_id)
            .with_code("invalid_provider"))
        }
    };

    let (mut api_key, mut key_source) = resolve_api_key_provider_secret(provider_id, pconfig);
    if api_key.is_empty() && provider_id == "lmstudio" {
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER.to_string();
        if key_source.is_empty() {
            key_source = "default".to_string();
        }
    }

    let env_url = if !pconfig.base_url_env_var.is_empty() {
        std::env::var(pconfig.base_url_env_var).unwrap_or_default().trim().to_string()
    } else {
        String::new()
    };

    let base_url = if provider_id == "kimi-coding" || provider_id == "kimi-coding-cn" {
        resolve_kimi_base_url(&api_key, pconfig.inference_base_url, &env_url)
    } else if provider_id == "zai" {
        resolve_zai_base_url(&api_key, pconfig.inference_base_url, &env_url)
    } else if !env_url.is_empty() {
        env_url.trim_end_matches('/').to_string()
    } else {
        pconfig.inference_base_url.to_string()
    };

    Ok(json!({
        "provider": provider_id,
        "api_key": api_key,
        "base_url": base_url.trim_end_matches('/'),
        "source": if key_source.is_empty() { "default".to_string() } else { key_source },
    }))
}

/// Resolve runtime details for local subprocess-backed providers.
pub fn resolve_external_process_provider_credentials(provider_id: &str) -> Result<Value, AuthError> {
    let pconfig = match get_provider_config(provider_id) {
        Some(pc) if pc.auth_type == "external_process" => pc,
        _ => {
            return Err(AuthError::new(format!(
                "Provider '{}' is not an external-process provider.",
                provider_id
            ))
            .with_provider(provider_id)
            .with_code("invalid_provider"))
        }
    };
    let mut base_url = if !pconfig.base_url_env_var.is_empty() {
        std::env::var(pconfig.base_url_env_var).unwrap_or_default().trim().to_string()
    } else {
        String::new()
    };
    if base_url.is_empty() {
        base_url = pconfig.inference_base_url.to_string();
    }
    let command = env_nonempty("HERMES_COPILOT_ACP_COMMAND")
        .or_else(|| env_nonempty("COPILOT_CLI_PATH"))
        .unwrap_or_else(|| "copilot".to_string());
    let raw_args = std::env::var("HERMES_COPILOT_ACP_ARGS").unwrap_or_default();
    let raw_args = raw_args.trim();
    let args: Vec<String> = if raw_args.is_empty() {
        vec!["--acp".to_string(), "--stdio".to_string()]
    } else {
        shlex_split(raw_args)
    };
    let resolved_command = which(&command);
    if resolved_command.is_none() && !base_url.starts_with("acp+tcp://") {
        return Err(AuthError::new(format!(
            "Could not find the Copilot CLI command '{}'. Install GitHub Copilot CLI \
             or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH.",
            command
        ))
        .with_provider(provider_id)
        .with_code("missing_copilot_cli"));
    }
    Ok(json!({
        "provider": provider_id,
        "api_key": "copilot-acp",
        "base_url": base_url.trim_end_matches('/'),
        "command": resolved_command.unwrap_or(command),
        "args": args,
        "source": "process",
    }))
}

/// Status dict for google-gemini-cli (OAuth). Delegates to crate::ag_google_oauth
/// in the wider binary; here we read the credentials file directly.
pub fn get_gemini_oauth_auth_status() -> Value {
    // Credentials live under ~/.hermes/auth/google_oauth.json (managed by
    // agent.google_oauth). Read it best-effort.
    let path = get_hermes_home().join("auth").join("google_oauth.json");
    let creds: Value = match fs::read_to_string(&path) {
        Ok(t) => serde_json::from_str(&t).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let access = creds.get("access_token").and_then(|v| v.as_str()).unwrap_or("");
    if access.is_empty() {
        return json!({
            "logged_in": false,
            "auth_file": path.to_string_lossy(),
            "error": "not logged in",
        });
    }
    json!({
        "logged_in": true,
        "auth_file": path.to_string_lossy(),
        "source": "google-oauth",
        "api_key": access,
        "expires_at_ms": creds.get("expires_ms").cloned().unwrap_or(Value::Null),
        "email": creds.get("email").and_then(|v| v.as_str()).unwrap_or(""),
        "project_id": creds.get("project_id").and_then(|v| v.as_str()).unwrap_or(""),
    })
}

/// Resolve runtime OAuth creds for google-gemini-cli.
pub fn resolve_gemini_oauth_runtime_credentials() -> Result<Value, AuthError> {
    let status = get_gemini_oauth_auth_status();
    if !status.get("logged_in").and_then(|v| v.as_bool()).unwrap_or(false) {
        return Err(AuthError::new("agent.google_oauth is not authenticated")
            .with_provider("google-gemini-cli")
            .with_code("google_oauth_missing"));
    }
    Ok(json!({
        "provider": "google-gemini-cli",
        "base_url": DEFAULT_GEMINI_CLOUDCODE_BASE_URL,
        "api_key": status.get("api_key").cloned().unwrap_or(Value::Null),
        "source": "google-oauth",
        "expires_at_ms": status.get("expires_at_ms").cloned().unwrap_or(Value::Null),
        "auth_file": status.get("auth_file").cloned().unwrap_or(Value::Null),
        "email": status.get("email").cloned().unwrap_or(json!("")),
        "project_id": status.get("project_id").cloned().unwrap_or(json!("")),
    }))
}

// =============================================================================
// MiniMax Portal OAuth (PKCE + user-code device flow)
// =============================================================================

/// Generate (code_verifier, code_challenge_S256, state) for MiniMax OAuth.
fn minimax_pkce_pair() -> (String, String, String) {
    let mut vbytes = [0u8; 48];
    let _ = getrandom_fill(&mut vbytes);
    let verifier: String = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(vbytes)
        .chars()
        .take(96)
        .collect();
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize());
    let mut sbytes = [0u8; 16];
    let _ = getrandom_fill(&mut sbytes);
    let state = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sbytes);
    (verifier, challenge, state)
}

fn minimax_request_user_code(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    client_id: &str,
    code_challenge: &str,
    state: &str,
) -> Result<Value, AuthError> {
    let resp = client
        .post(format!("{}/oauth/code", portal_base_url))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .header("x-request-id", uuid_hex())
        .form(&[
            ("response_type", "code"),
            ("client_id", client_id),
            ("scope", MINIMAX_OAUTH_SCOPE),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("MiniMax OAuth authorization failed: {}", e))
                .with_provider("minimax-oauth")
                .with_code("authorization_failed")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status != 200 {
        return Err(AuthError::new(format!(
            "MiniMax OAuth authorization failed: {}",
            text
        ))
        .with_provider("minimax-oauth")
        .with_code("authorization_failed"));
    }
    let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    for field in ["user_code", "verification_uri", "expired_in"] {
        if payload.get(field).is_none() {
            return Err(AuthError::new(format!(
                "MiniMax OAuth response missing field: {}",
                field
            ))
            .with_provider("minimax-oauth")
            .with_code("authorization_incomplete"));
        }
    }
    if payload.get("state").and_then(|v| v.as_str()) != Some(state) {
        return Err(AuthError::new("MiniMax OAuth state mismatch (possible CSRF).")
            .with_provider("minimax-oauth")
            .with_code("state_mismatch"));
    }
    Ok(payload)
}

fn minimax_poll_token(
    client: &reqwest::blocking::Client,
    portal_base_url: &str,
    client_id: &str,
    user_code: &str,
    code_verifier: &str,
    expired_in: i64,
    interval_ms: Option<i64>,
) -> Result<Value, AuthError> {
    let now_ms = (now_unix() * 1000.0) as i64;
    let deadline = if expired_in > now_ms / 2 {
        expired_in as f64 / 1000.0
    } else {
        now_unix() + expired_in.max(1) as f64
    };
    let interval = (interval_ms.unwrap_or(2000) as f64 / 1000.0).max(2.0);

    while now_unix() < deadline {
        let resp = client
            .post(format!("{}/oauth/token", portal_base_url))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .form(&[
                ("grant_type", MINIMAX_OAUTH_GRANT_TYPE),
                ("client_id", client_id),
                ("user_code", user_code),
                ("code_verifier", code_verifier),
            ])
            .send()
            .map_err(|e| {
                AuthError::new(format!("MiniMax OAuth error: {}", e))
                    .with_provider("minimax-oauth")
                    .with_code("token_exchange_failed")
            })?;
        let status = resp.status().as_u16();
        let text = resp.text().unwrap_or_default();
        let payload: Value = if text.is_empty() {
            json!({})
        } else {
            serde_json::from_str(&text).unwrap_or(json!({}))
        };

        if status != 200 {
            let msg = payload
                .get("base_resp")
                .and_then(|v| v.get("status_msg"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| text.clone());
            return Err(AuthError::new(format!(
                "MiniMax OAuth error: {}",
                if msg.is_empty() { "unknown".to_string() } else { msg }
            ))
            .with_provider("minimax-oauth")
            .with_code("token_exchange_failed"));
        }

        match payload.get("status").and_then(|v| v.as_str()) {
            Some("error") => {
                return Err(AuthError::new(
                    "MiniMax OAuth reported an error. Please try again later.",
                )
                .with_provider("minimax-oauth")
                .with_code("authorization_denied"))
            }
            Some("success") => {
                let ok = ["access_token", "refresh_token", "expired_in"]
                    .iter()
                    .all(|k| payload.get(*k).map(|v| !v.is_null()).unwrap_or(false));
                if !ok {
                    return Err(AuthError::new(
                        "MiniMax OAuth success payload missing required token fields.",
                    )
                    .with_provider("minimax-oauth")
                    .with_code("token_incomplete"));
                }
                return Ok(payload);
            }
            _ => {
                std::thread::sleep(Duration::from_secs_f64(interval));
            }
        }
    }
    Err(AuthError::new("MiniMax OAuth timed out before authorization completed.")
        .with_provider("minimax-oauth")
        .with_code("timeout"))
}

fn minimax_save_auth_state(auth_state: &Value) {
    let _guard = auth_store_lock();
    let mut auth_store = load_auth_store();
    save_provider_state(&mut auth_store, "minimax-oauth", auth_state.clone());
    let _ = save_auth_store(&mut auth_store);
}

/// Run MiniMax OAuth flow, persist tokens, return auth state dict.
pub fn minimax_oauth_login(
    region: &str,
    mut open_browser: bool,
    timeout_seconds: f64,
) -> Result<Value, AuthError> {
    let pconfig = get_provider_config("minimax-oauth").unwrap();
    let (portal_base_url, inference_base_url) = if region == "cn" {
        (
            pconfig.extra_get("cn_portal_base_url").unwrap_or(MINIMAX_OAUTH_CN_BASE),
            pconfig.extra_get("cn_inference_base_url").unwrap_or(MINIMAX_OAUTH_CN_INFERENCE),
        )
    } else {
        (pconfig.portal_base_url, pconfig.inference_base_url)
    };
    let (verifier, challenge, state) = minimax_pkce_pair();
    if is_remote_session() {
        open_browser = false;
    }
    println!("Starting Hermes login via MiniMax ({}) OAuth...", region);
    println!("Portal: {}", portal_base_url);

    let client = http_client(timeout_seconds);
    let code_data = minimax_request_user_code(&client, portal_base_url, pconfig.client_id, &challenge, &state)?;
    let verification_url = code_data.get("verification_uri").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let user_code = code_data.get("user_code").and_then(|v| v.as_str()).unwrap_or("").to_string();

    println!();
    println!("To continue:");
    println!("  1. Open: {}", verification_url);
    println!("  2. If prompted, enter code: {}", user_code);
    if open_browser {
        if open_in_browser(&verification_url) {
            println!("  (Opened browser for verification)");
        } else {
            println!("  Could not open browser automatically -- use the URL above.");
        }
    }

    let interval_ms = code_data
        .get("interval")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())));
    println!("Waiting for approval...");

    let expired_in = code_data
        .get("expired_in")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(0);
    let token_data = minimax_poll_token(
        &client,
        portal_base_url,
        pconfig.client_id,
        &user_code,
        &verifier,
        expired_in,
        interval_ms,
    )?;

    let now = now_unix();
    let expires_in_s = token_data
        .get("expired_in")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(0);
    let expires_at = now + expires_in_s as f64;

    let auth_state = json!({
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal_base_url,
        "inference_base_url": inference_base_url,
        "client_id": pconfig.client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type").and_then(|v| v.as_str()).unwrap_or("Bearer"),
        "access_token": token_data.get("access_token").cloned().unwrap_or(Value::Null),
        "refresh_token": token_data.get("refresh_token").cloned().unwrap_or(Value::Null),
        "resource_url": token_data.get("resource_url").cloned().unwrap_or(Value::Null),
        "obtained_at": iso_from_epoch(now),
        "expires_at": iso_from_epoch(expires_at),
        "expires_in": expires_in_s,
    });
    minimax_save_auth_state(&auth_state);
    println!("\u{2713} MiniMax OAuth login successful.");
    if let Some(msg) = token_data.get("notification_message").and_then(|v| v.as_str()) {
        if !msg.is_empty() {
            println!("Note from MiniMax: {}", msg);
        }
    }
    Ok(auth_state)
}

fn refresh_minimax_oauth_state(state: &Value, timeout_seconds: f64, force: bool) -> Result<Value, AuthError> {
    if state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
        return Err(AuthError::new(
            "MiniMax OAuth state has no refresh_token; please re-login.",
        )
        .with_provider("minimax-oauth")
        .with_code("no_refresh_token")
        .relogin());
    }
    let expires_at = state
        .get("expires_at")
        .and_then(|v| v.as_str())
        .and_then(parse_iso_timestamp)
        .unwrap_or(0.0);
    let now = now_unix();
    if !force && (expires_at - now) > MINIMAX_OAUTH_REFRESH_SKEW_SECONDS as f64 {
        return Ok(state.clone());
    }
    let portal_base_url = state.get("portal_base_url").and_then(|v| v.as_str()).unwrap_or("");
    let client = http_client(timeout_seconds);
    let resp = client
        .post(format!("{}/oauth/token", portal_base_url))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", state.get("client_id").and_then(|v| v.as_str()).unwrap_or("")),
            ("refresh_token", state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("")),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("MiniMax OAuth refresh failed: {}", e))
                .with_provider("minimax-oauth")
                .with_code("refresh_failed")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status != 200 {
        let body = text.to_lowercase();
        let relogin = ["invalid_grant", "refresh_token_reused", "invalid_refresh_token"]
            .iter()
            .any(|m| body.contains(m));
        let mut err = AuthError::new(format!("MiniMax OAuth refresh failed: {}", text))
            .with_provider("minimax-oauth")
            .with_code("refresh_failed");
        if relogin {
            err = err.relogin();
        }
        return Err(err);
    }
    let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if payload.get("status").and_then(|v| v.as_str()) != Some("success") {
        return Err(AuthError::new("MiniMax OAuth refresh did not return success.")
            .with_provider("minimax-oauth")
            .with_code("refresh_failed")
            .relogin());
    }
    let now_dt = now_unix();
    let expires_in_s = payload
        .get("expired_in")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(0);
    let mut new_state = state.clone();
    if let Some(obj) = new_state.as_object_mut() {
        obj.insert("access_token".to_string(), payload.get("access_token").cloned().unwrap_or(Value::Null));
        obj.insert(
            "refresh_token".to_string(),
            payload.get("refresh_token").cloned().unwrap_or_else(|| state.get("refresh_token").cloned().unwrap_or(Value::Null)),
        );
        obj.insert("obtained_at".to_string(), json!(iso_from_epoch(now_dt)));
        obj.insert("expires_at".to_string(), json!(iso_from_epoch(now_dt + expires_in_s as f64)));
        obj.insert("expires_in".to_string(), json!(expires_in_s));
    }
    minimax_save_auth_state(&new_state);
    Ok(new_state)
}

/// Return {provider, api_key, base_url, source} for minimax-oauth.
pub fn resolve_minimax_oauth_runtime_credentials() -> Result<Value, AuthError> {
    let state = get_provider_auth_state("minimax-oauth").filter(|s| {
        s.get("access_token").and_then(|v| v.as_str()).map(|t| !t.is_empty()).unwrap_or(false)
    });
    let state = match state {
        Some(s) => s,
        None => {
            return Err(AuthError::new(
                "Not logged into MiniMax OAuth. Run `hermes model` and select MiniMax (OAuth).",
            )
            .with_provider("minimax-oauth")
            .with_code("not_logged_in")
            .relogin())
        }
    };
    let state = refresh_minimax_oauth_state(&state, 15.0, false)?;
    Ok(json!({
        "provider": "minimax-oauth",
        "api_key": state.get("access_token").cloned().unwrap_or(Value::Null),
        "base_url": state.get("inference_base_url").and_then(|v| v.as_str()).unwrap_or("").trim_end_matches('/'),
        "source": "oauth",
    }))
}

/// Return auth status dict for MiniMax OAuth provider.
pub fn get_minimax_oauth_auth_status() -> Value {
    let state = get_provider_auth_state("minimax-oauth");
    let state = match state {
        Some(s) if s.get("access_token").and_then(|v| v.as_str()).map(|t| !t.is_empty()).unwrap_or(false) => s,
        _ => return json!({"logged_in": false, "provider": "minimax-oauth"}),
    };
    let token_valid = match state.get("expires_at").and_then(|v| v.as_str()).and_then(parse_iso_timestamp) {
        Some(exp) => (exp - now_unix()) > 0.0,
        None => state.get("access_token").and_then(|v| v.as_str()).map(|t| !t.is_empty()).unwrap_or(false),
    };
    json!({
        "logged_in": token_valid,
        "provider": "minimax-oauth",
        "region": state.get("region").and_then(|v| v.as_str()).unwrap_or("global"),
        "expires_at": state.get("expires_at").cloned().unwrap_or(Value::Null),
    })
}

/// Best-effort browser open (xdg-open/open/start). Returns true on success.
fn open_in_browser(url: &str) -> bool {
    if url.is_empty() {
        return false;
    }
    #[cfg(target_os = "macos")]
    let prog = "open";
    #[cfg(target_os = "windows")]
    let prog = "explorer";
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let prog = "xdg-open";
    std::process::Command::new(prog)
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}

// =============================================================================
// CLI config-update helpers
// =============================================================================

/// Update config.yaml and auth.json to reflect the active provider.
pub fn update_config_for_provider(
    provider_id: &str,
    inference_base_url: &str,
    default_model: Option<&str>,
) -> std::io::Result<PathBuf> {
    {
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        if let Some(obj) = auth_store.as_object_mut() {
            obj.insert("active_provider".to_string(), json!(provider_id));
        }
        let _ = save_auth_store(&mut auth_store);
    }

    let config_path = get_config_path();
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut config = read_raw_config();
    if !config.is_object() {
        config = json!({});
    }

    let current_model = config.get("model").cloned();
    let mut model_cfg: Map<String, Value> = match current_model {
        Some(Value::Object(m)) => m,
        Some(Value::String(s)) if !s.trim().is_empty() => {
            let mut m = Map::new();
            m.insert("default".to_string(), json!(s.trim()));
            m
        }
        _ => Map::new(),
    };

    model_cfg.insert("provider".to_string(), json!(provider_id));
    if !inference_base_url.trim().is_empty() {
        model_cfg.insert("base_url".to_string(), json!(inference_base_url.trim_end_matches('/')));
    } else {
        model_cfg.remove("base_url");
    }
    model_cfg.remove("api_key");
    model_cfg.remove("api_mode");

    if let Some(dm) = default_model {
        let cur_default = model_cfg.get("default").and_then(|v| v.as_str()).unwrap_or("");
        if cur_default.is_empty() || cur_default.contains('/') {
            model_cfg.insert("default".to_string(), json!(dm));
        }
    }

    if let Some(obj) = config.as_object_mut() {
        obj.insert("model".to_string(), Value::Object(model_cfg));
    }
    write_raw_config(&config)?;
    Ok(config_path)
}

fn get_config_provider() -> Option<String> {
    let config = read_raw_config();
    let provider = config.get("model")?.get("provider")?.as_str()?;
    let provider = provider.trim().to_lowercase();
    if provider.is_empty() {
        None
    } else {
        Some(provider)
    }
}

fn config_provider_matches(provider_id: &str) -> bool {
    if provider_id.is_empty() {
        return false;
    }
    get_config_provider().as_deref() == Some(provider_id.trim().to_lowercase().as_str())
}

fn logout_default_provider_from_config() -> Option<String> {
    match get_config_provider().as_deref() {
        Some("nous") => Some("nous".to_string()),
        Some("openai-codex") => Some("openai-codex".to_string()),
        _ => None,
    }
}

/// Reset config.yaml provider back to auto after logout.
pub fn reset_config_provider() -> std::io::Result<PathBuf> {
    let config_path = get_config_path();
    if !config_path.exists() {
        return Ok(config_path);
    }
    let mut config = read_raw_config();
    if !config.is_object() {
        return Ok(config_path);
    }
    if let Some(model) = config.get_mut("model").and_then(|v| v.as_object_mut()) {
        model.insert("provider".to_string(), json!("auto"));
        if model.contains_key("base_url") {
            model.insert("base_url".to_string(), json!(OPENROUTER_BASE_URL));
        }
    }
    write_raw_config(&config)?;
    Ok(config_path)
}

/// Save the selected model to config.yaml (single source of truth).
pub fn save_model_choice(model_id: &str) -> std::io::Result<()> {
    let mut config = read_raw_config();
    if !config.is_object() {
        config = json!({});
    }
    let obj = config.as_object_mut().unwrap();
    match obj.get_mut("model") {
        Some(Value::Object(m)) => {
            m.insert("default".to_string(), json!(model_id));
        }
        _ => {
            obj.insert("model".to_string(), json!({"default": model_id}));
        }
    }
    write_raw_config(&config)
}

// =============================================================================
// OpenAI Codex device-code login
// =============================================================================

/// Run the OpenAI device code login flow and return a credentials dict.
pub fn codex_device_code_login() -> Result<Value, AuthError> {
    let issuer = "https://auth.openai.com";
    let client_id = CODEX_OAUTH_CLIENT_ID;

    let client = http_client(15.0);
    let resp = client
        .post(format!("{}/api/accounts/deviceauth/usercode", issuer))
        .header("Content-Type", "application/json")
        .json(&json!({"client_id": client_id}))
        .send()
        .map_err(|e| {
            AuthError::new(format!("Failed to request device code: {}", e))
                .with_provider("openai-codex")
                .with_code("device_code_request_failed")
        })?;
    if resp.status().as_u16() != 200 {
        return Err(AuthError::new(format!(
            "Device code request returned status {}.",
            resp.status().as_u16()
        ))
        .with_provider("openai-codex")
        .with_code("device_code_request_error"));
    }
    let device_data: Value = serde_json::from_str(&resp.text().unwrap_or_default()).unwrap_or(Value::Null);
    let user_code = device_data.get("user_code").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let device_auth_id = device_data.get("device_auth_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let poll_interval = device_data
        .get("interval")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(5)
        .max(3);

    if user_code.is_empty() || device_auth_id.is_empty() {
        return Err(AuthError::new("Device code response missing required fields.")
            .with_provider("openai-codex")
            .with_code("device_code_incomplete"));
    }

    println!("To continue, follow these steps:\n");
    println!("  1. Open this URL in your browser:");
    println!("     \x1b[94m{}/codex/device\x1b[0m\n", issuer);
    println!("  2. Enter this code:");
    println!("     \x1b[94m{}\x1b[0m\n", user_code);
    println!("Waiting for sign-in... (press Ctrl+C to cancel)");

    let max_wait = 15.0 * 60.0;
    let start = std::time::Instant::now();
    let mut code_resp: Option<Value> = None;
    let poll_client = http_client(15.0);
    while start.elapsed().as_secs_f64() < max_wait {
        std::thread::sleep(Duration::from_secs(poll_interval as u64));
        let poll = poll_client
            .post(format!("{}/api/accounts/deviceauth/token", issuer))
            .header("Content-Type", "application/json")
            .json(&json!({"device_auth_id": device_auth_id, "user_code": user_code}))
            .send()
            .map_err(|e| {
                AuthError::new(format!("Device auth polling failed: {}", e))
                    .with_provider("openai-codex")
                    .with_code("device_code_poll_error")
            })?;
        let status = poll.status().as_u16();
        if status == 200 {
            code_resp = serde_json::from_str(&poll.text().unwrap_or_default()).ok();
            break;
        } else if status == 403 || status == 404 {
            continue;
        } else {
            return Err(AuthError::new(format!(
                "Device auth polling returned status {}.",
                status
            ))
            .with_provider("openai-codex")
            .with_code("device_code_poll_error"));
        }
    }

    let code_resp = match code_resp {
        Some(c) => c,
        None => {
            return Err(AuthError::new("Login timed out after 15 minutes.")
                .with_provider("openai-codex")
                .with_code("device_code_timeout"))
        }
    };

    let authorization_code = code_resp.get("authorization_code").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let code_verifier = code_resp.get("code_verifier").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let redirect_uri = format!("{}/deviceauth/callback", issuer);
    if authorization_code.is_empty() || code_verifier.is_empty() {
        return Err(AuthError::new(
            "Device auth response missing authorization_code or code_verifier.",
        )
        .with_provider("openai-codex")
        .with_code("device_code_incomplete_exchange"));
    }

    let token_client = http_client(15.0);
    let token_resp = token_client
        .post(CODEX_OAUTH_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", authorization_code.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("client_id", client_id),
            ("code_verifier", code_verifier.as_str()),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("Token exchange failed: {}", e))
                .with_provider("openai-codex")
                .with_code("token_exchange_failed")
        })?;
    if token_resp.status().as_u16() != 200 {
        return Err(AuthError::new(format!(
            "Token exchange returned status {}.",
            token_resp.status().as_u16()
        ))
        .with_provider("openai-codex")
        .with_code("token_exchange_error"));
    }
    let tokens: Value = serde_json::from_str(&token_resp.text().unwrap_or_default()).unwrap_or(Value::Null);
    let access_token = tokens.get("access_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let refresh_token = tokens.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if access_token.is_empty() {
        return Err(AuthError::new("Token exchange did not return an access_token.")
            .with_provider("openai-codex")
            .with_code("token_exchange_no_access_token"));
    }
    let base_url = {
        let env = std::env::var("HERMES_CODEX_BASE_URL").unwrap_or_default().trim().trim_end_matches('/').to_string();
        if env.is_empty() { DEFAULT_CODEX_BASE_URL.to_string() } else { env }
    };
    Ok(json!({
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
        "base_url": base_url,
        "last_refresh": utc_now_iso_z(),
        "auth_mode": "chatgpt",
        "source": "device-code",
    }))
}

// =============================================================================
// Nous device-code login + CLI persistence wiring
// =============================================================================

/// Run the Nous device-code flow and return full OAuth state without persisting.
#[allow(clippy::too_many_arguments)]
pub fn nous_device_code_login(
    portal_base_url: Option<&str>,
    inference_base_url: Option<&str>,
    client_id: Option<&str>,
    scope: Option<&str>,
    mut open_browser: bool,
    timeout_seconds: f64,
    min_key_ttl_seconds: i64,
) -> Result<Value, AuthError> {
    let pconfig = get_provider_config("nous").unwrap();
    let portal_base_url = portal_base_url
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| env_nonempty("HERMES_PORTAL_BASE_URL"))
        .or_else(|| env_nonempty("NOUS_PORTAL_BASE_URL"))
        .unwrap_or_else(|| pconfig.portal_base_url.to_string())
        .trim_end_matches('/')
        .to_string();
    let requested_inference_url = inference_base_url
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| env_nonempty("NOUS_INFERENCE_BASE_URL"))
        .unwrap_or_else(|| pconfig.inference_base_url.to_string())
        .trim_end_matches('/')
        .to_string();
    let client_id = client_id.filter(|s| !s.is_empty()).unwrap_or(pconfig.client_id).to_string();
    let scope = scope.filter(|s| !s.is_empty()).unwrap_or(pconfig.scope).to_string();

    if is_remote_session() {
        open_browser = false;
    }
    println!("Starting Hermes login via {}...", pconfig.name);
    println!("Portal: {}", portal_base_url);

    let client = http_client(timeout_seconds);
    let device_data = request_device_code(&client, &portal_base_url, &client_id, Some(&scope))?;
    let verification_url = device_data.get("verification_uri_complete").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let user_code = device_data.get("user_code").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let expires_in = device_data
        .get("expires_in")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(0);
    let interval = device_data
        .get("interval")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok())))
        .unwrap_or(5);

    println!();
    println!("To continue:");
    println!("  1. Open: {}", verification_url);
    println!("  2. If prompted, enter code: {}", user_code);
    if open_browser {
        if open_in_browser(&verification_url) {
            println!("  (Opened browser for verification)");
        } else {
            println!("  Could not open browser automatically — use the URL above.");
        }
    }
    let effective_interval = interval.max(1).min(DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS).max(1);
    println!("Waiting for approval (polling every {}s)...", effective_interval);

    let device_code = device_data.get("device_code").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let token_data = poll_for_token(&client, &portal_base_url, &client_id, &device_code, expires_in, interval)?;

    let now = now_unix();
    let token_expires_in = coerce_ttl_seconds(token_data.get("expires_in"));
    let expires_at = now + token_expires_in as f64;
    let resolved_inference_url = optional_base_url(token_data.get("inference_base_url"))
        .unwrap_or_else(|| requested_inference_url.clone());
    if resolved_inference_url != requested_inference_url {
        println!("Using portal-provided inference URL: {}", resolved_inference_url);
    }

    let auth_state = json!({
        "portal_base_url": portal_base_url,
        "inference_base_url": resolved_inference_url,
        "client_id": client_id,
        "scope": token_data.get("scope").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or(&scope),
        "token_type": token_data.get("token_type").and_then(|v| v.as_str()).unwrap_or("Bearer"),
        "access_token": token_data.get("access_token").cloned().unwrap_or(Value::Null),
        "refresh_token": token_data.get("refresh_token").cloned().unwrap_or(Value::Null),
        "obtained_at": iso_from_epoch(now),
        "expires_at": iso_from_epoch(expires_at),
        "expires_in": token_expires_in,
        "tls": {"insecure": false, "ca_bundle": Value::Null},
        "agent_key": Value::Null,
        "agent_key_id": Value::Null,
        "agent_key_expires_at": Value::Null,
        "agent_key_expires_in": Value::Null,
        "agent_key_reused": Value::Null,
        "agent_key_obtained_at": Value::Null,
    });

    refresh_nous_oauth_from_state(&auth_state, min_key_ttl_seconds, timeout_seconds, false, true)
}

// =============================================================================
// Spotify auth — PKCE tokens stored in ~/.hermes/auth.json
// =============================================================================

fn spotify_scope_list(raw_scope: Option<&str>) -> Vec<String> {
    let scope_text = raw_scope.unwrap_or("").trim();
    let default = default_spotify_scope();
    let text = if scope_text.is_empty() { default.as_str() } else { scope_text };
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for part in text.split_whitespace() {
        if seen.insert(part.to_string()) {
            ordered.push(part.to_string());
        }
    }
    ordered
}

fn spotify_scope_string(raw_scope: Option<&str>) -> String {
    spotify_scope_list(raw_scope).join(" ")
}

fn spotify_client_id(explicit: Option<&str>, state: Option<&Value>) -> Result<String, AuthError> {
    let candidates: Vec<Option<String>> = vec![
        explicit.map(|s| s.to_string()),
        env_nonempty("HERMES_SPOTIFY_CLIENT_ID"),
        env_nonempty("SPOTIFY_CLIENT_ID"),
        state.and_then(|s| s.get("client_id")).and_then(|v| v.as_str()).map(|s| s.to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        let cleaned = c.trim();
        if !cleaned.is_empty() {
            return Ok(cleaned.to_string());
        }
    }
    Err(AuthError::new(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.",
    )
    .with_provider("spotify")
    .with_code("spotify_client_id_missing"))
}

fn spotify_redirect_uri(explicit: Option<&str>, state: Option<&Value>) -> String {
    let candidates: Vec<Option<String>> = vec![
        explicit.map(|s| s.to_string()),
        env_nonempty("HERMES_SPOTIFY_REDIRECT_URI"),
        env_nonempty("SPOTIFY_REDIRECT_URI"),
        state.and_then(|s| s.get("redirect_uri")).and_then(|v| v.as_str()).map(|s| s.to_string()),
        Some(DEFAULT_SPOTIFY_REDIRECT_URI.to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        let cleaned = c.trim();
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }
    DEFAULT_SPOTIFY_REDIRECT_URI.to_string()
}

fn spotify_api_base_url(state: Option<&Value>) -> String {
    let candidates: Vec<Option<String>> = vec![
        env_nonempty("HERMES_SPOTIFY_API_BASE_URL"),
        state.and_then(|s| s.get("api_base_url")).and_then(|v| v.as_str()).map(|s| s.to_string()),
        Some(DEFAULT_SPOTIFY_API_BASE_URL.to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        let cleaned = c.trim().trim_end_matches('/');
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }
    DEFAULT_SPOTIFY_API_BASE_URL.to_string()
}

fn spotify_accounts_base_url(state: Option<&Value>) -> String {
    let candidates: Vec<Option<String>> = vec![
        env_nonempty("HERMES_SPOTIFY_ACCOUNTS_BASE_URL"),
        state.and_then(|s| s.get("accounts_base_url")).and_then(|v| v.as_str()).map(|s| s.to_string()),
        Some(DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        let cleaned = c.trim().trim_end_matches('/');
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()
}

fn spotify_code_verifier() -> String {
    let mut bytes = [0u8; 64];
    let _ = getrandom_fill(&mut bytes);
    let raw = base64::engine::general_purpose::URL_SAFE.encode(bytes);
    raw.trim_end_matches('=').chars().take(128).collect()
}

fn spotify_code_challenge(code_verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(code_verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize())
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn spotify_build_authorize_url(
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    state: &str,
    code_challenge: &str,
    accounts_base_url: &str,
) -> String {
    let query = format!(
        "client_id={}&response_type=code&redirect_uri={}&scope={}&state={}&code_challenge_method=S256&code_challenge={}",
        url_encode(client_id),
        url_encode(redirect_uri),
        url_encode(scope),
        url_encode(state),
        url_encode(code_challenge),
    );
    format!("{}/authorize?{}", accounts_base_url, query)
}

/// (host, port, path) parsed from a Spotify PKCE redirect_uri.
fn spotify_validate_redirect_uri(redirect_uri: &str) -> Result<(String, u16, String), AuthError> {
    let parsed = url::Url::parse(redirect_uri).map_err(|_| {
        AuthError::new("Spotify PKCE redirect_uri must use http://localhost or http://127.0.0.1.")
            .with_provider("spotify")
            .with_code("spotify_redirect_invalid")
    })?;
    if parsed.scheme() != "http" {
        return Err(AuthError::new(
            "Spotify PKCE redirect_uri must use http://localhost or http://127.0.0.1.",
        )
        .with_provider("spotify")
        .with_code("spotify_redirect_invalid"));
    }
    let host = parsed.host_str().unwrap_or("").to_string();
    if host != "127.0.0.1" && host != "localhost" {
        return Err(AuthError::new(
            "Spotify PKCE redirect_uri must point to localhost or 127.0.0.1.",
        )
        .with_provider("spotify")
        .with_code("spotify_redirect_invalid"));
    }
    let port = parsed.port().ok_or_else(|| {
        AuthError::new("Spotify PKCE redirect_uri must include an explicit localhost port.")
            .with_provider("spotify")
            .with_code("spotify_redirect_invalid")
    })?;
    let path = if parsed.path().is_empty() {
        "/".to_string()
    } else {
        parsed.path().to_string()
    };
    Ok((host, port, path))
}

/// Spotify callback result captured from the local HTTP server.
#[derive(Default, Debug, Clone)]
pub struct SpotifyCallback {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

fn parse_query_params(query: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        out.insert(url_decode(k), url_decode(v));
    }
    out
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Run a minimal localhost HTTP server until the Spotify callback arrives.
fn spotify_wait_for_callback(redirect_uri: &str, timeout_seconds: f64) -> Result<SpotifyCallback, AuthError> {
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpListener;

    let (host, port, expected_path) = spotify_validate_redirect_uri(redirect_uri)?;
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|e| {
        AuthError::new(format!(
            "Could not bind Spotify callback server on {}:{}: {}",
            host, port, e
        ))
        .with_provider("spotify")
        .with_code("spotify_callback_bind_failed")
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| AuthError::new(format!("Callback server setup failed: {}", e)).with_provider("spotify"))?;

    let deadline = std::time::Instant::now() + Duration::from_secs_f64(timeout_seconds.max(5.0));
    loop {
        if std::time::Instant::now() >= deadline {
            return Err(AuthError::new(
                "Spotify authorization timed out waiting for the local callback.",
            )
            .with_provider("spotify")
            .with_code("spotify_callback_timeout"));
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut reader = BufReader::new(&mut stream);
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    continue;
                }
                // Drain headers (ignore body).
                let mut header_line = String::new();
                while reader.read_line(&mut header_line).map(|n| n > 0).unwrap_or(false) {
                    if header_line == "\r\n" || header_line == "\n" {
                        break;
                    }
                    header_line.clear();
                }
                // request_line: "GET /path?query HTTP/1.1"
                let parts: Vec<&str> = request_line.split_whitespace().collect();
                let target = parts.get(1).copied().unwrap_or("/");
                let (path, query) = match target.split_once('?') {
                    Some((p, q)) => (p, q),
                    None => (target, ""),
                };
                if path != expected_path {
                    let body = b"Not found.";
                    let _ = write_http(&mut stream, 404, "text/plain", body);
                    continue;
                }
                let params = parse_query_params(query);
                let result = SpotifyCallback {
                    code: params.get("code").cloned(),
                    state: params.get("state").cloned(),
                    error: params.get("error").cloned(),
                    error_description: params.get("error_description").cloned(),
                };
                let body = if result.error.is_some() {
                    "<html><body><h1>Spotify authorization failed.</h1>You can close this tab.</body></html>"
                } else {
                    "<html><body><h1>Spotify authorization received.</h1>You can close this tab.</body></html>"
                };
                let _ = write_http(&mut stream, 200, "text/html; charset=utf-8", body.as_bytes());
                // Drain a touch more so the browser receives the body.
                let mut _scratch = [0u8; 64];
                let _ = stream.read(&mut _scratch);
                if result.code.is_some() || result.error.is_some() {
                    return Ok(result);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

fn write_http(stream: &mut std::net::TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        reason,
        content_type,
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn spotify_token_payload_to_state(
    token_payload: &Value,
    client_id: &str,
    redirect_uri: &str,
    requested_scope: &str,
    accounts_base_url: &str,
    api_base_url: &str,
    previous_state: Option<&Value>,
) -> Value {
    let now = now_unix();
    let expires_in = coerce_ttl_seconds(token_payload.get("expires_in"));
    let expires_at = now + expires_in as f64;
    let mut state = previous_state.cloned().unwrap_or_else(|| json!({}));
    let prev_refresh = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let token_type = {
        let t = token_payload
            .get("token_type")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("Bearer")
            .trim();
        if t.is_empty() { "Bearer" } else { t }.to_string()
    };
    if let Some(obj) = state.as_object_mut() {
        obj.insert("client_id".to_string(), json!(client_id));
        obj.insert("redirect_uri".to_string(), json!(redirect_uri));
        obj.insert("accounts_base_url".to_string(), json!(accounts_base_url));
        obj.insert("api_base_url".to_string(), json!(api_base_url));
        obj.insert("scope".to_string(), json!(requested_scope));
        obj.insert(
            "granted_scope".to_string(),
            json!(token_payload.get("scope").and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(requested_scope).trim()),
        );
        obj.insert("token_type".to_string(), json!(token_type));
        obj.insert(
            "access_token".to_string(),
            json!(token_payload.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim()),
        );
        obj.insert(
            "refresh_token".to_string(),
            json!(token_payload
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&prev_refresh)
                .trim()),
        );
        obj.insert("obtained_at".to_string(), json!(iso_from_epoch(now)));
        obj.insert("expires_at".to_string(), json!(iso_from_epoch(expires_at)));
        obj.insert("expires_in".to_string(), json!(expires_in));
        obj.insert("auth_type".to_string(), json!("oauth_pkce"));
    }
    state
}

fn spotify_exchange_code_for_tokens(
    client_id: &str,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
    accounts_base_url: &str,
    timeout_seconds: f64,
) -> Result<Value, AuthError> {
    let client = http_client(timeout_seconds);
    let resp = client
        .post(format!("{}/api/token", accounts_base_url))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("client_id", client_id),
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", code_verifier),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("Spotify token exchange failed: {}", e))
                .with_provider("spotify")
                .with_code("spotify_token_exchange_failed")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status >= 400 {
        let detail = text.trim();
        let mut msg = "Spotify token exchange failed.".to_string();
        if !detail.is_empty() {
            msg.push_str(&format!(" Response: {}", detail));
        }
        return Err(AuthError::new(msg)
            .with_provider("spotify")
            .with_code("spotify_token_exchange_failed"));
    }
    let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if !payload.is_object()
        || payload.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().is_empty()
    {
        return Err(AuthError::new("Spotify token response did not include an access_token.")
            .with_provider("spotify")
            .with_code("spotify_token_exchange_invalid"));
    }
    Ok(payload)
}

fn refresh_spotify_oauth_state(state: &Value, timeout_seconds: f64) -> Result<Value, AuthError> {
    let refresh_token = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if refresh_token.is_empty() {
        return Err(AuthError::new("Spotify refresh token missing. Run `hermes auth spotify` again.")
            .with_provider("spotify")
            .with_code("spotify_refresh_token_missing")
            .relogin());
    }
    let client_id = spotify_client_id(None, Some(state))?;
    let accounts_base_url = spotify_accounts_base_url(Some(state));
    let client = http_client(timeout_seconds);
    let resp = client
        .post(format!("{}/api/token", accounts_base_url))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.as_str()),
            ("client_id", client_id.as_str()),
        ])
        .send()
        .map_err(|e| {
            AuthError::new(format!("Spotify token refresh failed: {}", e))
                .with_provider("spotify")
                .with_code("spotify_refresh_failed")
        })?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status >= 400 {
        let detail = text.trim();
        let mut msg = "Spotify token refresh failed. Run `hermes auth spotify` again.".to_string();
        if !detail.is_empty() {
            msg.push_str(&format!(" Response: {}", detail));
        }
        return Err(AuthError::new(msg)
            .with_provider("spotify")
            .with_code("spotify_refresh_failed")
            .relogin());
    }
    let payload: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if !payload.is_object()
        || payload.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().is_empty()
    {
        return Err(AuthError::new("Spotify refresh response did not include an access_token.")
            .with_provider("spotify")
            .with_code("spotify_refresh_invalid")
            .relogin());
    }
    let scope = state.get("scope").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(default_spotify_scope);
    Ok(spotify_token_payload_to_state(
        &payload,
        &client_id,
        &spotify_redirect_uri(None, Some(state)),
        &scope,
        &accounts_base_url,
        &spotify_api_base_url(Some(state)),
        Some(state),
    ))
}

/// Resolve Spotify runtime credentials (refreshing if expiring).
pub fn resolve_spotify_runtime_credentials(
    force_refresh: bool,
    refresh_if_expiring: bool,
    refresh_skew_seconds: i64,
) -> Result<Value, AuthError> {
    let mut state;
    {
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        state = match load_provider_state(&auth_store, "spotify") {
            Some(s) => s,
            None => {
                return Err(AuthError::new(
                    "Spotify is not authenticated. Run `hermes auth spotify` first.",
                )
                .with_provider("spotify")
                .with_code("spotify_auth_missing")
                .relogin())
            }
        };
        let mut should_refresh = force_refresh;
        if !should_refresh && refresh_if_expiring {
            should_refresh = is_expiring(state.get("expires_at"), refresh_skew_seconds);
        }
        if should_refresh {
            state = refresh_spotify_oauth_state(&state, 20.0)?;
            store_provider_state(&mut auth_store, "spotify", state.clone(), false);
            let _ = save_auth_store(&mut auth_store);
        }
    }

    let access_token = state.get("access_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if access_token.is_empty() {
        return Err(AuthError::new("Spotify access token missing. Run `hermes auth spotify` again.")
            .with_provider("spotify")
            .with_code("spotify_access_token_missing")
            .relogin());
    }
    Ok(json!({
        "provider": "spotify",
        "access_token": access_token,
        "api_key": access_token,
        "token_type": state.get("token_type").and_then(|v| v.as_str()).unwrap_or("Bearer"),
        "base_url": spotify_api_base_url(Some(&state)),
        "scope": state.get("granted_scope").and_then(|v| v.as_str())
            .or_else(|| state.get("scope").and_then(|v| v.as_str())).unwrap_or("").trim(),
        "client_id": spotify_client_id(None, Some(&state)).unwrap_or_default(),
        "redirect_uri": spotify_redirect_uri(None, Some(&state)),
        "expires_at": state.get("expires_at").cloned().unwrap_or(Value::Null),
        "refresh_token": state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").trim(),
    }))
}

/// Status snapshot for Spotify auth.
pub fn get_spotify_auth_status() -> Value {
    let state = match get_provider_auth_state("spotify") {
        Some(s) => s,
        None => return json!({"logged_in": false}),
    };
    let expires_at = state.get("expires_at").cloned().unwrap_or(Value::Null);
    let refresh_token = state.get("refresh_token").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let logged_in = !refresh_token.is_empty() || !is_expiring(Some(&expires_at), 0);
    json!({
        "logged_in": logged_in,
        "auth_type": state.get("auth_type").and_then(|v| v.as_str()).unwrap_or("oauth_pkce"),
        "client_id": state.get("client_id").cloned().unwrap_or(Value::Null),
        "redirect_uri": state.get("redirect_uri").cloned().unwrap_or(Value::Null),
        "scope": state.get("granted_scope").cloned().filter(|v| !v.is_null())
            .or_else(|| state.get("scope").cloned()).unwrap_or(Value::Null),
        "expires_at": expires_at,
        "api_base_url": state.get("api_base_url").cloned().unwrap_or(Value::Null),
        "has_refresh_token": !refresh_token.is_empty(),
    })
}

// =============================================================================
// CLI command argument structs + entry points
// =============================================================================

/// Arguments for the Spotify login command (mirror of argparse fields).
#[derive(Debug, Clone, Default)]
pub struct SpotifyLoginArgs {
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    pub scope: Option<String>,
    pub no_browser: bool,
    pub timeout: Option<f64>,
}

/// Outcome of a CLI command: either completed, or requests a process exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliOutcome {
    Ok,
    Exit(i32),
}

/// `hermes auth spotify` — run the Spotify PKCE login flow.
pub fn login_spotify_command(args: &SpotifyLoginArgs) -> CliOutcome {
    let existing_state = get_provider_auth_state("spotify").unwrap_or_else(|| json!({}));

    let client_id = match spotify_client_id(args.client_id.as_deref(), Some(&existing_state)) {
        Ok(c) => c,
        Err(exc) => {
            if !exc.code_is("spotify_client_id_missing") {
                eprintln!("{}", format_auth_error(&exc));
                return CliOutcome::Exit(1);
            }
            // Non-interactive port: cannot run the interactive wizard here.
            println!("Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.");
            println!("See {} for the full setup guide.", SPOTIFY_DOCS_URL);
            return CliOutcome::Exit(1);
        }
    };

    let redirect_uri = spotify_redirect_uri(args.redirect_uri.as_deref(), Some(&existing_state));
    let scope = spotify_scope_string(
        args.scope
            .as_deref()
            .or_else(|| existing_state.get("scope").and_then(|v| v.as_str())),
    );
    let accounts_base_url = spotify_accounts_base_url(Some(&existing_state));
    let api_base_url = spotify_api_base_url(Some(&existing_state));
    let open_browser = !args.no_browser;

    let code_verifier = spotify_code_verifier();
    let code_challenge = spotify_code_challenge(&code_verifier);
    let state_nonce = uuid_hex();
    let authorize_url = spotify_build_authorize_url(
        &client_id,
        &redirect_uri,
        &scope,
        &state_nonce,
        &code_challenge,
        &accounts_base_url,
    );

    println!("Starting Spotify PKCE login...");
    println!("Client ID: {}", client_id);
    println!("Redirect URI: {}", redirect_uri);
    println!("Make sure this redirect URI is allow-listed in your Spotify app settings.");
    println!();
    println!("Open this URL to authorize Hermes:");
    println!("{}", authorize_url);
    println!();
    println!("Full setup guide: {}", SPOTIFY_DOCS_URL);
    println!();

    if open_browser && !is_remote_session() {
        if open_in_browser(&authorize_url) {
            println!("Browser opened for Spotify authorization.");
        } else {
            println!("Could not open the browser automatically; use the URL above.");
        }
    }

    let callback = match spotify_wait_for_callback(
        &redirect_uri,
        args.timeout.unwrap_or(180.0),
    ) {
        Ok(c) => c,
        Err(exc) => {
            eprintln!("{}", format_auth_error(&exc));
            return CliOutcome::Exit(1);
        }
    };
    if let Some(err) = &callback.error {
        let detail = callback.error_description.clone().unwrap_or_else(|| err.clone());
        println!("Spotify authorization failed: {}", detail);
        return CliOutcome::Exit(1);
    }
    if callback.state.as_deref() != Some(state_nonce.as_str()) {
        println!("Spotify authorization failed: state mismatch.");
        return CliOutcome::Exit(1);
    }

    let token_payload = match spotify_exchange_code_for_tokens(
        &client_id,
        callback.code.as_deref().unwrap_or(""),
        &redirect_uri,
        &code_verifier,
        &accounts_base_url,
        args.timeout.unwrap_or(20.0),
    ) {
        Ok(p) => p,
        Err(exc) => {
            eprintln!("{}", format_auth_error(&exc));
            return CliOutcome::Exit(1);
        }
    };
    let spotify_state = spotify_token_payload_to_state(
        &token_payload,
        &client_id,
        &redirect_uri,
        &scope,
        &accounts_base_url,
        &api_base_url,
        None,
    );

    let saved_to = {
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        store_provider_state(&mut auth_store, "spotify", spotify_state, false);
        save_auth_store(&mut auth_store).unwrap_or_else(|_| auth_file_path())
    };

    println!("Spotify login successful!");
    println!("  Auth state: {}", saved_to.display());
    println!("  Provider state saved under providers.spotify");
    println!("  Docs: {}", SPOTIFY_DOCS_URL);
    CliOutcome::Ok
}

/// Arguments for the MiniMax OAuth login command.
#[derive(Debug, Clone, Default)]
pub struct MinimaxLoginArgs {
    pub region: Option<String>,
    pub no_browser: bool,
    pub timeout: Option<f64>,
}

/// CLI entry for MiniMax OAuth login.
pub fn login_minimax_oauth(args: &MinimaxLoginArgs) -> CliOutcome {
    let region = args.region.clone().unwrap_or_else(|| "global".to_string());
    let open_browser = !args.no_browser;
    let timeout = args.timeout.unwrap_or(15.0);
    match minimax_oauth_login(&region, open_browser, timeout) {
        Ok(_) => CliOutcome::Ok,
        Err(exc) => {
            println!("{}", format_auth_error(&exc));
            CliOutcome::Exit(1)
        }
    }
}

/// Arguments for the logout command.
#[derive(Debug, Clone, Default)]
pub struct LogoutArgs {
    pub provider: Option<String>,
}

/// `hermes logout` — clear auth state for a provider.
pub fn logout_command(args: &LogoutArgs) -> CliOutcome {
    let provider_id = args.provider.clone();

    if let Some(ref pid) = provider_id {
        if !is_known_auth_provider(pid) {
            println!("Unknown provider: {}", pid);
            return CliOutcome::Exit(1);
        }
    }

    let active = get_active_provider();
    let target = provider_id
        .clone()
        .or(active)
        .or_else(logout_default_provider_from_config);

    let target = match target {
        Some(t) if !t.is_empty() => t,
        _ => {
            println!("No provider is currently logged in.");
            return CliOutcome::Ok;
        }
    };

    let config_matches = config_provider_matches(&target);
    let provider_name = get_auth_provider_display_name(&target);

    if clear_provider_auth(Some(&target)) || config_matches {
        let _ = reset_config_provider();
        println!("Logged out of {}.", provider_name);
        if !std::env::var("OPENROUTER_API_KEY").unwrap_or_default().is_empty() {
            println!("Hermes will use OpenRouter for inference.");
        } else {
            println!("Run `hermes model` or configure an API key to use Hermes.");
        }
    } else {
        println!("No auth state found for {}.", provider_name);
    }
    CliOutcome::Ok
}

/// `hermes login` — deprecated; prints guidance and requests exit(0).
pub fn login_command() -> CliOutcome {
    println!("The 'hermes login' command has been removed.");
    println!("Use 'hermes auth' to manage credentials,");
    println!("'hermes model' to select a provider, or 'hermes setup' for full setup.");
    CliOutcome::Exit(0)
}

/// Arguments for the Nous login command.
#[derive(Debug, Clone, Default)]
pub struct NousLoginArgs {
    pub portal_url: Option<String>,
    pub inference_url: Option<String>,
    pub client_id: Option<String>,
    pub scope: Option<String>,
    pub no_browser: bool,
    pub timeout: Option<f64>,
    /// When true, accept the shared-store credentials offer non-interactively.
    pub import_shared: bool,
    /// A model id selected by the caller (the interactive picker lives in the
    /// CLI layer; pass the chosen id here, or None to keep the prior provider).
    pub selected_model: Option<String>,
}

/// `hermes auth add nous` — Nous Portal device authorization flow.
///
/// This mirrors Python `_login_nous`, minus the interactive model-selection
/// prompt (which is provided by the CLI layer via `args.selected_model`). It
/// offers a shared-store one-tap import when `args.import_shared` is set.
pub fn login_nous(args: &NousLoginArgs) -> CliOutcome {
    let timeout_seconds = args.timeout.unwrap_or(15.0);
    let pconfig = get_provider_config("nous").unwrap();

    let mut auth_state: Option<Value> = None;

    // Codex-style auto-import from the shared store.
    if args.import_shared {
        if let Some(_shared) = read_shared_nous_state() {
            println!("Rehydrating Nous session from shared credentials...");
            auth_state = try_import_shared_nous_state(timeout_seconds, 5 * 60);
            if auth_state.is_none() {
                println!("Could not refresh shared credentials — falling back to device-code login.");
            }
        }
    }

    if auth_state.is_none() {
        match nous_device_code_login(
            args.portal_url.as_deref(),
            args.inference_url.as_deref(),
            args.client_id.as_deref().or(Some(pconfig.client_id)),
            args.scope.as_deref().or(Some(pconfig.scope)),
            !args.no_browser,
            timeout_seconds,
            5 * 60,
        ) {
            Ok(s) => auth_state = Some(s),
            Err(exc) => {
                if exc.code_is("subscription_required") {
                    println!();
                    println!("Your Nous Portal account does not have an active subscription.");
                    println!("After subscribing, run `hermes model` again to finish setup.");
                    return CliOutcome::Exit(1);
                }
                println!("Login failed: {}", exc);
                return CliOutcome::Exit(1);
            }
        }
    }

    let auth_state = auth_state.unwrap();
    let inference_base_url = auth_state
        .get("inference_base_url")
        .and_then(|v| v.as_str())
        .unwrap_or(DEFAULT_NOUS_INFERENCE_URL)
        .to_string();

    // Snapshot prior active provider before overwriting to "nous".
    let prior_active_provider = {
        let _guard = auth_store_lock();
        load_auth_store()
            .get("active_provider")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    let saved_to = {
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        save_provider_state(&mut auth_store, "nous", auth_state.clone());
        save_auth_store(&mut auth_store).unwrap_or_else(|_| auth_file_path())
    };
    write_shared_nous_state(&auth_state);

    println!();
    println!("Login successful!");
    println!("  Auth state: {}", saved_to.display());

    // The CLI layer performs model selection and passes the result here.
    let selected_model = args.selected_model.clone().filter(|s| !s.is_empty());

    if selected_model.is_none() {
        // Restore prior active_provider; leave config.yaml model untouched.
        let _guard = auth_store_lock();
        let mut auth_store = load_auth_store();
        if let Some(obj) = auth_store.as_object_mut() {
            match prior_active_provider {
                Some(p) => {
                    obj.insert("active_provider".to_string(), json!(p));
                }
                None => {
                    obj.remove("active_provider");
                }
            }
        }
        let _ = save_auth_store(&mut auth_store);
        println!();
        println!("No provider change. Nous credentials saved for future use.");
        println!("  Run `hermes model` again to switch to Nous Portal.");
        return CliOutcome::Ok;
    }

    let model = selected_model.unwrap();
    match update_config_for_provider("nous", &inference_base_url, Some(&model)) {
        Ok(config_path) => {
            let _ = save_model_choice(&model);
            println!("Default model set to: {}", model);
            println!("  Config updated: {} (model.provider=nous)", config_path.display());
            CliOutcome::Ok
        }
        Err(e) => {
            println!("Login failed: {}", e);
            CliOutcome::Exit(1)
        }
    }
}

/// Arguments for the OpenAI Codex login command.
#[derive(Debug, Clone, Default)]
pub struct CodexLoginArgs {
    pub force_new_login: bool,
    /// When true, accept the "use existing credentials" / "import CLI tokens"
    /// offers non-interactively (the CLI layer handles the prompt).
    pub accept_existing: bool,
    pub import_cli_tokens: bool,
}

/// `hermes auth add openai-codex` — OpenAI Codex login via device-code flow.
/// Mirrors Python `_login_openai_codex`.
pub fn login_openai_codex(args: &CodexLoginArgs) -> CliOutcome {
    // 1. Existing Hermes-owned credentials.
    if !args.force_new_login {
        if let Ok(existing) = resolve_codex_runtime_credentials(false, true, CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) {
            let key = existing.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
            if !key.is_empty() && !codex_access_token_is_expiring(key, 60) {
                println!("Existing Codex credentials found in Hermes auth store.");
                if args.accept_existing {
                    let base = existing.get("base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_CODEX_BASE_URL);
                    match update_config_for_provider("openai-codex", base, None) {
                        Ok(config_path) => {
                            println!();
                            println!("Login successful!");
                            println!("  Config updated: {} (model.provider=openai-codex)", config_path.display());
                            return CliOutcome::Ok;
                        }
                        Err(e) => {
                            println!("Login failed: {}", e);
                            return CliOutcome::Exit(1);
                        }
                    }
                }
            } else {
                println!("Existing Codex credentials are expired. Starting fresh login...");
            }
        }
    }

    // 2. Import Codex CLI tokens (~/.codex/auth.json).
    if !args.force_new_login {
        if let Some(cli_tokens) = import_codex_cli_tokens() {
            println!("Found existing Codex CLI credentials at ~/.codex/auth.json");
            println!("Hermes will create its own session to avoid conflicts with Codex CLI / VS Code.");
            if args.import_cli_tokens {
                save_codex_tokens(&cli_tokens, None);
                let base_url = {
                    let env = std::env::var("HERMES_CODEX_BASE_URL").unwrap_or_default().trim().trim_end_matches('/').to_string();
                    if env.is_empty() { DEFAULT_CODEX_BASE_URL.to_string() } else { env }
                };
                match update_config_for_provider("openai-codex", &base_url, None) {
                    Ok(config_path) => {
                        println!();
                        println!("Credentials imported. Note: if Codex CLI refreshes its token,");
                        println!("Hermes will keep working independently with its own session.");
                        println!("  Config updated: {} (model.provider=openai-codex)", config_path.display());
                        return CliOutcome::Ok;
                    }
                    Err(e) => {
                        println!("Login failed: {}", e);
                        return CliOutcome::Exit(1);
                    }
                }
            }
        }
    }

    // 3. Fresh device-code flow.
    println!();
    println!("Signing in to OpenAI Codex...");
    println!("(Hermes creates its own session — won't affect Codex CLI or VS Code)");
    println!();

    let creds = match codex_device_code_login() {
        Ok(c) => c,
        Err(exc) => {
            println!("{}", format_auth_error(&exc));
            return CliOutcome::Exit(1);
        }
    };
    let tokens = creds.get("tokens").cloned().unwrap_or(Value::Null);
    let last_refresh = creds.get("last_refresh").and_then(|v| v.as_str());
    save_codex_tokens(&tokens, last_refresh);
    let base = creds.get("base_url").and_then(|v| v.as_str()).unwrap_or(DEFAULT_CODEX_BASE_URL);
    match update_config_for_provider("openai-codex", base, None) {
        Ok(config_path) => {
            println!();
            println!("Login successful!");
            println!("  Config updated: {} (model.provider=openai-codex)", config_path.display());
            CliOutcome::Ok
        }
        Err(e) => {
            println!("Login failed: {}", e);
            CliOutcome::Exit(1)
        }
    }
}

/// Return the first usable Anthropic credential, or "".
/// Mirrors PROVIDER_REGISTRY["anthropic"].api_key_env_vars order.
pub fn get_anthropic_key() -> String {
    if let Some(pc) = get_provider_config("anthropic") {
        for var in pc.api_key_env_vars {
            let value = get_env_value(var);
            if !value.is_empty() {
                return value;
            }
        }
    }
    String::new()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Serialize tests that mutate HERMES_HOME / process env.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn with_tmp_home<F: FnOnce(&Path)>(f: F) {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("hermes_cli_auth_test_{}", uuid_hex()));
        fs::create_dir_all(&dir).unwrap();
        let prev = std::env::var("HERMES_HOME").ok();
        unsafe {
            std::env::set_var("HERMES_HOME", &dir);
        }
        f(&dir);
        match prev {
            Some(v) => unsafe { std::env::set_var("HERMES_HOME", v) },
            None => unsafe { std::env::remove_var("HERMES_HOME") },
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn registry_lookup_and_aliases() {
        assert_eq!(get_provider_config("nous").unwrap().name, "Nous Portal");
        assert!(get_provider_config("does-not-exist").is_none());
        let aliases = provider_aliases();
        assert_eq!(aliases.get("glm"), Some(&"zai"));
        assert_eq!(aliases.get("claude"), Some(&"anthropic"));
        assert_eq!(aliases.get("aws"), Some(&"bedrock"));
    }

    #[test]
    fn has_usable_secret_rejects_placeholders() {
        assert!(has_usable_secret("sk-1234567890"));
        assert!(!has_usable_secret("***"));
        assert!(!has_usable_secret("dummy"));
        assert!(!has_usable_secret("abc")); // too short
        assert!(!has_usable_secret("   "));
        assert!(!has_usable_secret("NONE"));
    }

    #[test]
    fn kimi_base_url_routing() {
        assert_eq!(
            resolve_kimi_base_url("sk-kimi-abc", "https://api.moonshot.ai/v1", ""),
            KIMI_CODE_BASE_URL
        );
        assert_eq!(
            resolve_kimi_base_url("sk-other", "https://api.moonshot.ai/v1", ""),
            "https://api.moonshot.ai/v1"
        );
        // explicit env override always wins
        assert_eq!(
            resolve_kimi_base_url("sk-kimi-abc", "https://api.moonshot.ai/v1", "https://x"),
            "https://x"
        );
        // no key -> default
        assert_eq!(
            resolve_kimi_base_url("", "https://api.moonshot.ai/v1", ""),
            "https://api.moonshot.ai/v1"
        );
    }

    #[test]
    fn parse_iso_timestamp_handles_z_and_naive() {
        let with_z = parse_iso_timestamp("2030-01-01T00:00:00Z").unwrap();
        let with_offset = parse_iso_timestamp("2030-01-01T00:00:00+00:00").unwrap();
        assert!((with_z - with_offset).abs() < 1.0);
        // Naive is treated as UTC.
        let naive = parse_iso_timestamp("2030-01-01T00:00:00").unwrap();
        assert!((naive - with_z).abs() < 1.0);
        assert!(parse_iso_timestamp("").is_none());
        assert!(parse_iso_timestamp("not-a-date").is_none());
    }

    #[test]
    fn is_expiring_logic() {
        // Far future -> not expiring.
        assert!(!is_expiring(Some(&json!("2099-01-01T00:00:00Z")), 0));
        // Past -> expiring.
        assert!(is_expiring(Some(&json!("2000-01-01T00:00:00Z")), 0));
        // Missing -> expiring.
        assert!(is_expiring(None, 0));
        assert!(is_expiring(Some(&Value::Null), 0));
    }

    #[test]
    fn coerce_ttl_clamps_negative() {
        assert_eq!(coerce_ttl_seconds(Some(&json!(100))), 100);
        assert_eq!(coerce_ttl_seconds(Some(&json!(-5))), 0);
        assert_eq!(coerce_ttl_seconds(Some(&json!("42"))), 42);
        assert_eq!(coerce_ttl_seconds(Some(&json!("garbage"))), 0);
        assert_eq!(coerce_ttl_seconds(None), 0);
    }

    #[test]
    fn decode_jwt_claims_extracts_payload() {
        // header.payload.signature where payload = {"exp": 9999999999}
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(br#"{"exp":9999999999}"#);
        let token = format!("aaa.{}.bbb", payload);
        let claims = decode_jwt_claims(&token);
        assert_eq!(claims.get("exp").and_then(|v| v.as_i64()), Some(9999999999));
        // Not a JWT -> empty.
        assert!(decode_jwt_claims("not-a-jwt").is_empty());
    }

    #[test]
    fn auth_store_roundtrip_and_active_provider() {
        with_tmp_home(|_| {
            let mut store = load_auth_store();
            assert_eq!(store.get("version").and_then(|v| v.as_i64()), Some(1));
            save_provider_state(&mut store, "nous", json!({"access_token": "abc"}));
            save_auth_store(&mut store).unwrap();

            let loaded = load_auth_store();
            assert_eq!(loaded.get("active_provider").and_then(|v| v.as_str()), Some("nous"));
            let state = load_provider_state(&loaded, "nous").unwrap();
            assert_eq!(state.get("access_token").and_then(|v| v.as_str()), Some("abc"));
            assert_eq!(get_active_provider().as_deref(), Some("nous"));
        });
    }

    #[test]
    fn clear_provider_auth_clears_active() {
        with_tmp_home(|_| {
            let mut store = load_auth_store();
            save_provider_state(&mut store, "nous", json!({"access_token": "abc"}));
            save_auth_store(&mut store).unwrap();

            assert!(clear_provider_auth(Some("nous")));
            let loaded = load_auth_store();
            assert!(load_provider_state(&loaded, "nous").is_none());
            assert!(loaded.get("active_provider").map(|v| v.is_null()).unwrap_or(true));
            // Clearing again finds nothing.
            assert!(!clear_provider_auth(Some("nous")));
        });
    }

    #[test]
    fn suppress_and_unsuppress_sources() {
        with_tmp_home(|_| {
            assert!(!is_source_suppressed("zai", "env"));
            suppress_credential_source("zai", "env");
            assert!(is_source_suppressed("zai", "env"));
            // idempotent
            suppress_credential_source("zai", "env");
            assert!(unsuppress_credential_source("zai", "env"));
            assert!(!is_source_suppressed("zai", "env"));
            assert!(!unsuppress_credential_source("zai", "env"));
        });
    }

    #[test]
    fn resolve_provider_known_and_aliases() {
        assert_eq!(resolve_provider(Some("nous"), None, None).unwrap(), "nous");
        assert_eq!(resolve_provider(Some("glm"), None, None).unwrap(), "zai");
        assert_eq!(resolve_provider(Some("openrouter"), None, None).unwrap(), "openrouter");
        assert_eq!(resolve_provider(Some("custom"), None, None).unwrap(), "custom");
        // Unknown raises.
        let err = resolve_provider(Some("bogus-provider"), None, None).unwrap_err();
        assert!(err.code_is("invalid_provider"));
    }

    #[test]
    fn resolve_provider_explicit_creds_means_openrouter() {
        with_tmp_home(|_| {
            assert_eq!(
                resolve_provider(Some("auto"), Some("sk-xxx"), None).unwrap(),
                "openrouter"
            );
            assert_eq!(
                resolve_provider(None, None, Some("https://x")).unwrap(),
                "openrouter"
            );
        });
    }

    #[test]
    fn migration_from_systems_format() {
        with_tmp_home(|home| {
            let legacy = json!({
                "systems": {"nous_portal": {"access_token": "legacy"}}
            });
            fs::write(home.join("auth.json"), serde_json::to_string(&legacy).unwrap()).unwrap();
            let store = load_auth_store();
            assert_eq!(store.get("active_provider").and_then(|v| v.as_str()), Some("nous"));
            let state = load_provider_state(&store, "nous").unwrap();
            assert_eq!(state.get("access_token").and_then(|v| v.as_str()), Some("legacy"));
        });
    }

    #[test]
    fn credential_pool_read_write() {
        with_tmp_home(|_| {
            write_credential_pool("zai", vec![json!({"access_token": "k1"})]).unwrap();
            let slice = read_credential_pool(Some("zai"));
            assert_eq!(slice.as_array().unwrap().len(), 1);
            let full = read_credential_pool(None);
            assert!(full.get("zai").is_some());
        });
    }

    #[test]
    fn format_auth_error_maps_codes() {
        let relogin = AuthError::new("expired").relogin();
        assert!(format_auth_error(&relogin).contains("re-authenticate"));
        let sub = AuthError::new("x").with_code("subscription_required");
        assert!(format_auth_error(&sub).contains("subscription"));
        let plain = AuthError::new("plain message");
        assert_eq!(format_auth_error(&plain), "plain message");
    }

    #[test]
    fn shlex_split_basic() {
        assert_eq!(shlex_split("--acp --stdio"), vec!["--acp", "--stdio"]);
        assert_eq!(shlex_split("a \"b c\" d"), vec!["a", "b c", "d"]);
        assert_eq!(shlex_split("'single quoted'"), vec!["single quoted"]);
        assert!(shlex_split("").is_empty());
    }

    #[test]
    fn spotify_scope_dedup_preserves_order() {
        let scopes = spotify_scope_list(Some("a b a c b"));
        assert_eq!(scopes, vec!["a", "b", "c"]);
        assert_eq!(spotify_scope_string(Some("x y x")), "x y");
    }

    #[test]
    fn spotify_pkce_challenge_is_url_safe_nopad() {
        let challenge = spotify_code_challenge("verifier");
        assert!(!challenge.contains('='));
        assert!(!challenge.contains('+'));
        assert!(!challenge.contains('/'));
    }

    #[test]
    fn url_encode_decode_roundtrip() {
        let s = "hello world&foo=bar/baz";
        let enc = url_encode(s);
        assert!(!enc.contains(' '));
        assert_eq!(url_decode(&enc), s);
    }

    #[test]
    fn known_provider_and_display_name() {
        assert!(is_known_auth_provider("nous"));
        assert!(is_known_auth_provider("Spotify"));
        assert!(!is_known_auth_provider("nope"));
        assert_eq!(get_auth_provider_display_name("spotify"), "Spotify");
        assert_eq!(get_auth_provider_display_name("zai"), "Z.AI / GLM");
    }

    #[test]
    fn qwen_token_expiry_logic() {
        let future_ms = (now_unix() * 1000.0) as i64 + 10_000_000;
        assert!(!qwen_access_token_is_expiring(Some(&json!(future_ms)), 0));
        let past_ms = (now_unix() * 1000.0) as i64 - 10_000;
        assert!(qwen_access_token_is_expiring(Some(&json!(past_ms)), 0));
        assert!(qwen_access_token_is_expiring(None, 0));
    }
}
