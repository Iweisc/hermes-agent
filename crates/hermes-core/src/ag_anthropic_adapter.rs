//! Anthropic Messages API adapter for Hermes Agent.
//!
//! Native Rust port of `agent/anthropic_adapter.py`. Translates between
//! Hermes's internal OpenAI-style message format and Anthropic's Messages
//! API. All provider-specific logic is isolated here.
//!
//! There is no embeddable "anthropic SDK" in Rust the way the Python module
//! depends on the `anthropic` package, so `build_anthropic_client` is replaced
//! by [`AnthropicClientConfig`], a value type capturing exactly the headers,
//! auth scheme, base URL, query params and timeouts the Python code would have
//! configured on the SDK client. Callers can apply it directly to a
//! `reqwest::blocking::Client` / request builder.
//!
//! Auth supports:
//!   - Regular API keys (sk-ant-api*) → x-api-key header
//!   - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
//!   - Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json) → Bearer auth

use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Mutex;

// ── Effort / thinking tables ──────────────────────────────────────────

/// Manual-thinking budget tokens per Hermes effort level.
pub fn thinking_budget(effort: &str) -> Option<i64> {
    match effort {
        "xhigh" => Some(32000),
        "high" => Some(16000),
        "medium" => Some(8000),
        "low" => Some(4000),
        _ => None,
    }
}

/// Map a Hermes effort level → Anthropic adaptive-thinking effort.
/// "minimal" is a legacy alias for low; unknown → "medium".
pub fn adaptive_effort_map(effort: &str) -> &'static str {
    match effort {
        "max" => "max",
        "xhigh" => "xhigh",
        "high" => "high",
        "medium" => "medium",
        "low" => "low",
        "minimal" => "low",
        _ => "medium",
    }
}

const XHIGH_EFFORT_SUBSTRINGS: &[&str] = &["4-7", "4.7"];
const ADAPTIVE_THINKING_SUBSTRINGS: &[&str] = &["4-6", "4.6", "4-7", "4.7"];
const NO_SAMPLING_PARAMS_SUBSTRINGS: &[&str] = &["4-7", "4.7"];
const FAST_MODE_SUPPORTED_SUBSTRINGS: &[&str] = &["opus-4-6", "opus-4.6"];

/// Max output token limits per Anthropic model. Substring keys.
const ANTHROPIC_OUTPUT_LIMITS: &[(&str, i64)] = &[
    ("claude-opus-4-7", 128_000),
    ("claude-opus-4-6", 128_000),
    ("claude-sonnet-4-6", 64_000),
    ("claude-opus-4-5", 64_000),
    ("claude-sonnet-4-5", 64_000),
    ("claude-haiku-4-5", 64_000),
    ("claude-opus-4", 32_000),
    ("claude-sonnet-4", 64_000),
    ("claude-3-7-sonnet", 128_000),
    ("claude-3-5-sonnet", 8_192),
    ("claude-3-5-haiku", 8_192),
    ("claude-3-opus", 4_096),
    ("claude-3-sonnet", 4_096),
    ("claude-3-haiku", 4_096),
    ("minimax", 131_072),
    ("qwen3", 65_536),
];

const ANTHROPIC_DEFAULT_OUTPUT_LIMIT: i64 = 128_000;

/// Look up the max output token limit for an Anthropic model.
///
/// Longest-prefix (longest matching substring) wins. Dots are normalized to
/// hyphens so `anthropic/claude-opus-4.6` resolves to `claude-opus-4-6`.
pub fn get_anthropic_max_output(model: &str) -> i64 {
    let m = model.to_lowercase().replace('.', "-");
    let mut best_key = "";
    let mut best_val = ANTHROPIC_DEFAULT_OUTPUT_LIMIT;
    for (key, val) in ANTHROPIC_OUTPUT_LIMITS {
        if m.contains(key) && key.len() > best_key.len() {
            best_key = key;
            best_val = *val;
        }
    }
    best_val
}

/// Return `value` floored to a positive int, or `None` if it is not a finite
/// positive number. Booleans are rejected (they aren't numbers here).
///
/// Mirrors `_resolve_positive_anthropic_max_tokens`. JSON has no separate bool
/// /int confusion, so a `Value::Bool` is treated as non-numeric → `None`.
pub fn resolve_positive_anthropic_max_tokens(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(_) => None,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i > 0 {
                    Some(i)
                } else {
                    None
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return None;
                }
                // truncate toward zero like Python int()
                let floored = f.trunc() as i64;
                if floored > 0 {
                    Some(floored)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve the `max_tokens` budget for an Anthropic Messages call.
///
/// Prefers `requested` when positive & finite; otherwise the model's output
/// ceiling. Errors if no positive budget can be resolved.
pub fn resolve_anthropic_messages_max_tokens(
    requested: &Value,
    model: &str,
) -> Result<i64, String> {
    if let Some(resolved) = resolve_positive_anthropic_max_tokens(requested) {
        return Ok(resolved);
    }
    let fallback = get_anthropic_max_output(model);
    if fallback > 0 {
        return Ok(fallback);
    }
    Err(format!(
        "Anthropic Messages adapter requires a positive max_tokens value for model {:?}; \
         got {} and no model default resolved.",
        model, requested
    ))
}

pub fn supports_adaptive_thinking(model: &str) -> bool {
    ADAPTIVE_THINKING_SUBSTRINGS.iter().any(|v| model.contains(v))
}

pub fn supports_xhigh_effort(model: &str) -> bool {
    XHIGH_EFFORT_SUBSTRINGS.iter().any(|v| model.contains(v))
}

pub fn forbids_sampling_params(model: &str) -> bool {
    NO_SAMPLING_PARAMS_SUBSTRINGS.iter().any(|v| model.contains(v))
}

pub fn supports_fast_mode(model: &str) -> bool {
    FAST_MODE_SUPPORTED_SUBSTRINGS
        .iter()
        .any(|v| model.contains(v))
}

// ── Beta headers ───────────────────────────────────────────────────────

pub const COMMON_BETAS: &[&str] = &[
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
    "context-1m-2025-08-07",
];
pub const TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
pub const CONTEXT_1M_BETA: &str = "context-1m-2025-08-07";
pub const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";
pub const OAUTH_ONLY_BETAS: &[&str] = &["claude-code-20250219", "oauth-2025-04-20"];

pub const CLAUDE_CODE_VERSION_FALLBACK: &str = "2.1.74";
pub const CLAUDE_CODE_SYSTEM_PREFIX: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";
pub const MCP_TOOL_PREFIX: &str = "mcp_";

static CLAUDE_CODE_VERSION_CACHE: Mutex<Option<String>> = Mutex::new(None);

/// Detect the installed Claude Code version, falling back to a static constant.
pub fn detect_claude_code_version() -> String {
    for cmd in ["claude", "claude-code"] {
        if let Ok(output) = std::process::Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let trimmed = stdout.trim();
                if !trimmed.is_empty() {
                    if let Some(version) = trimmed.split_whitespace().next() {
                        if version
                            .chars()
                            .next()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or(false)
                        {
                            return version.to_string();
                        }
                    }
                }
            }
        }
    }
    CLAUDE_CODE_VERSION_FALLBACK.to_string()
}

/// Lazily detect & cache the installed Claude Code version.
pub fn get_claude_code_version() -> String {
    let mut guard = CLAUDE_CODE_VERSION_CACHE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(detect_claude_code_version());
    }
    guard.clone().unwrap()
}

/// Check if the key is an Anthropic OAuth/setup token.
pub fn is_oauth_token(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    if key.starts_with("sk-ant-api") {
        return false;
    }
    if key.starts_with("sk-ant-") {
        return true;
    }
    if key.starts_with("eyJ") {
        return true;
    }
    if key.starts_with("cc-") {
        return true;
    }
    false
}

/// Normalize SDK/base transport URL values to a plain trimmed string.
pub fn normalize_base_url_text(base_url: Option<&str>) -> String {
    match base_url {
        Some(s) if !s.is_empty() => s.trim().to_string(),
        _ => String::new(),
    }
}

// ── URL host helpers (ported from utils.base_url_host_matches) ──────────

/// Return the lowercased hostname for a base URL, or "" if absent.
pub fn base_url_hostname(base_url: &str) -> String {
    let raw = base_url.trim();
    if raw.is_empty() {
        return String::new();
    }
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("//{}", raw)
    };
    match url::Url::parse(&candidate) {
        Ok(u) => u
            .host_str()
            .map(|h| h.to_lowercase().trim_end_matches('.').to_string())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

/// Return True when the base URL's hostname is `domain` or a subdomain.
pub fn base_url_host_matches(base_url: &str, domain: &str) -> bool {
    let hostname = base_url_hostname(base_url);
    if hostname.is_empty() {
        return false;
    }
    let domain = domain.trim().to_lowercase();
    let domain = domain.trim_end_matches('.');
    if domain.is_empty() {
        return false;
    }
    hostname == domain || hostname.ends_with(&format!(".{}", domain))
}

/// Return True for non-Anthropic endpoints using the Anthropic Messages API.
pub fn is_third_party_anthropic_endpoint(base_url: Option<&str>) -> bool {
    let normalized = normalize_base_url_text(base_url);
    if normalized.is_empty() {
        return false;
    }
    let normalized = normalized.trim_end_matches('/').to_lowercase();
    if normalized.contains("anthropic.com") {
        return false;
    }
    true
}

/// Return True for Kimi's /coding endpoint that requires claude-code UA.
pub fn is_kimi_coding_endpoint(base_url: Option<&str>) -> bool {
    let normalized = normalize_base_url_text(base_url);
    if normalized.is_empty() {
        return false;
    }
    normalized
        .trim_end_matches('/')
        .to_lowercase()
        .starts_with("https://api.kimi.com/coding")
}

const KIMI_FAMILY_MODEL_PREFIXES: &[&str] = &[
    "kimi-", "kimi_", "moonshot-", "moonshot_", "k1.", "k1-", "k2.", "k2-", "k25", "k2.5",
];

pub fn model_name_is_kimi_family(model: Option<&str>) -> bool {
    let model = match model {
        Some(m) => m,
        None => return false,
    };
    let mut m = model.trim().to_lowercase();
    if m.is_empty() {
        return false;
    }
    if let Some(idx) = m.rfind('/') {
        m = m[idx + 1..].to_string();
    }
    KIMI_FAMILY_MODEL_PREFIXES.iter().any(|p| m.starts_with(p))
}

/// Return True for any Kimi / Moonshot Anthropic-Messages-speaking endpoint.
pub fn is_kimi_family_endpoint(base_url: Option<&str>, model: Option<&str>) -> bool {
    if is_kimi_coding_endpoint(base_url) {
        return true;
    }
    let bu = base_url.unwrap_or("");
    for domain in ["api.kimi.com", "moonshot.ai", "moonshot.cn"] {
        if base_url_host_matches(bu, domain) {
            return true;
        }
    }
    model_name_is_kimi_family(model)
}

/// Return True for DeepSeek's Anthropic-compatible `/anthropic` endpoint.
pub fn is_deepseek_anthropic_endpoint(base_url: Option<&str>) -> bool {
    let bu = base_url.unwrap_or("");
    if !base_url_host_matches(bu, "api.deepseek.com") {
        return false;
    }
    let normalized = normalize_base_url_text(base_url);
    if normalized.is_empty() {
        return false;
    }
    normalized
        .trim_end_matches('/')
        .to_lowercase()
        .contains("/anthropic")
}

/// Return True for Anthropic-compatible providers that require Bearer auth.
pub fn requires_bearer_auth(base_url: Option<&str>) -> bool {
    let normalized = normalize_base_url_text(base_url);
    if normalized.is_empty() {
        return false;
    }
    let normalized = normalized.trim_end_matches('/').to_lowercase();
    normalized.starts_with("https://api.minimax.io/anthropic")
        || normalized.starts_with("https://api.minimaxi.com/anthropic")
}

/// Return the beta headers that are safe for the configured endpoint.
pub fn common_betas_for_base_url(base_url: Option<&str>, drop_context_1m_beta: bool) -> Vec<String> {
    if requires_bearer_auth(base_url) {
        return COMMON_BETAS
            .iter()
            .filter(|b| **b != TOOL_STREAMING_BETA && **b != CONTEXT_1M_BETA)
            .map(|s| s.to_string())
            .collect();
    }
    if drop_context_1m_beta {
        return COMMON_BETAS
            .iter()
            .filter(|b| **b != CONTEXT_1M_BETA)
            .map(|s| s.to_string())
            .collect();
    }
    COMMON_BETAS.iter().map(|s| s.to_string()).collect()
}

// ── Client configuration (replaces build_anthropic_client / SDK) ────────

/// The auth scheme the SDK client would have used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicAuth {
    /// `x-api-key: <value>`
    ApiKey(String),
    /// `Authorization: Bearer <value>`
    Bearer(String),
}

/// Captures everything the Python `build_anthropic_client` would have set on
/// the SDK client: base URL, auth scheme, default headers, query params,
/// connect/read timeouts. Apply this to a reqwest client to make requests.
#[derive(Debug, Clone)]
pub struct AnthropicClientConfig {
    pub base_url: Option<String>,
    pub auth: AnthropicAuth,
    pub default_headers: Vec<(String, String)>,
    pub default_query: Vec<(String, String)>,
    pub read_timeout_secs: f64,
    pub connect_timeout_secs: f64,
}

/// Build an Anthropic client configuration, auto-detecting setup-tokens vs
/// API keys. Faithful port of `build_anthropic_client`.
///
/// `timeout` overrides the default 900s read timeout when positive.
pub fn build_anthropic_client_config(
    api_key: &str,
    base_url: Option<&str>,
    timeout: Option<f64>,
    drop_context_1m_beta: bool,
) -> AnthropicClientConfig {
    let normalized_base_url = normalize_base_url_text(base_url);
    let read_timeout = match timeout {
        Some(t) if t > 0.0 => t,
        _ => 900.0,
    };

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut query: Vec<(String, String)> = Vec::new();
    let mut effective_base_url: Option<String> = None;

    if !normalized_base_url.is_empty() {
        let is_azure = normalized_base_url.to_lowercase().contains("azure.com");
        if is_azure && !normalized_base_url.contains("api-version") {
            effective_base_url = Some(normalized_base_url.trim_end_matches('/').to_string());
            query.push(("api-version".to_string(), "2025-04-15".to_string()));
        } else {
            effective_base_url = Some(normalized_base_url.clone());
        }
    }

    let common_betas =
        common_betas_for_base_url(Some(normalized_base_url.as_str()), drop_context_1m_beta);

    let auth: AnthropicAuth;

    if is_kimi_coding_endpoint(base_url) {
        // Kimi /coding requires claude-code UA. Checked before bearer detection.
        headers.push(("User-Agent".to_string(), "claude-code/0.1.0".to_string()));
        if !common_betas.is_empty() {
            headers.push(("anthropic-beta".to_string(), common_betas.join(",")));
        }
        auth = AnthropicAuth::ApiKey(api_key.to_string());
    } else if requires_bearer_auth(Some(normalized_base_url.as_str())) {
        auth = AnthropicAuth::Bearer(api_key.to_string());
        if !common_betas.is_empty() {
            headers.push(("anthropic-beta".to_string(), common_betas.join(",")));
        }
    } else if is_third_party_anthropic_endpoint(base_url) {
        auth = AnthropicAuth::ApiKey(api_key.to_string());
        if !common_betas.is_empty() {
            headers.push(("anthropic-beta".to_string(), common_betas.join(",")));
        }
    } else if is_oauth_token(api_key) {
        let mut all_betas = common_betas.clone();
        all_betas.extend(OAUTH_ONLY_BETAS.iter().map(|s| s.to_string()));
        auth = AnthropicAuth::Bearer(api_key.to_string());
        headers.push(("anthropic-beta".to_string(), all_betas.join(",")));
        headers.push((
            "user-agent".to_string(),
            format!("claude-cli/{} (external, cli)", get_claude_code_version()),
        ));
        headers.push(("x-app".to_string(), "cli".to_string()));
    } else {
        auth = AnthropicAuth::ApiKey(api_key.to_string());
        if !common_betas.is_empty() {
            headers.push(("anthropic-beta".to_string(), common_betas.join(",")));
        }
    }

    AnthropicClientConfig {
        base_url: effective_base_url,
        auth,
        default_headers: headers,
        default_query: query,
        read_timeout_secs: read_timeout,
        connect_timeout_secs: 10.0,
    }
}

// ── Claude Code credentials ─────────────────────────────────────────────

/// Resolved Claude Code OAuth credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeCodeCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: i64,
    pub source: String,
}

fn parse_claude_oauth(data: &Value, source: &str) -> Option<ClaudeCodeCredentials> {
    let oauth = data.get("claudeAiOauth")?;
    if !oauth.is_object() {
        return None;
    }
    let access_token = oauth
        .get("accessToken")
        .and_then(Value::as_str)
        .unwrap_or("");
    if access_token.is_empty() {
        return None;
    }
    Some(ClaudeCodeCredentials {
        access_token: access_token.to_string(),
        refresh_token: oauth
            .get("refreshToken")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        expires_at: oauth.get("expiresAt").and_then(Value::as_i64).unwrap_or(0),
        source: source.to_string(),
    })
}

/// Read Claude Code OAuth credentials from the macOS Keychain.
pub fn read_claude_code_credentials_from_keychain() -> Option<ClaudeCodeCredentials> {
    if std::env::consts::OS != "macos" {
        return None;
    }
    let output = std::process::Command::new("security")
        .args([
            "find-generic-password",
            "-s",
            "Claude Code-credentials",
            "-w",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let data: Value = serde_json::from_str(raw).ok()?;
    parse_claude_oauth(&data, "macos_keychain")
}

/// Read refreshable Claude Code OAuth credentials (keychain, then JSON file).
pub fn read_claude_code_credentials() -> Option<ClaudeCodeCredentials> {
    if let Some(kc) = read_claude_code_credentials_from_keychain() {
        return Some(kc);
    }
    let cred_path = home_dir()?.join(".claude").join(".credentials.json");
    if cred_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&cred_path) {
            if let Ok(data) = serde_json::from_str::<Value>(&text) {
                if let Some(creds) = parse_claude_oauth(&data, "claude_code_credentials_file") {
                    return Some(creds);
                }
            }
        }
    }
    None
}

/// Read Claude's native managed key from ~/.claude.json for diagnostics only.
pub fn read_claude_managed_key() -> Option<String> {
    let claude_json = home_dir()?.join(".claude.json");
    if claude_json.exists() {
        if let Ok(text) = std::fs::read_to_string(&claude_json) {
            if let Ok(data) = serde_json::from_str::<Value>(&text) {
                if let Some(primary) = data.get("primaryApiKey").and_then(Value::as_str) {
                    let trimmed = primary.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }
    }
    None
}

fn home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Check if Claude Code credentials have a non-expired access token.
pub fn is_claude_code_token_valid(creds: &ClaudeCodeCredentials) -> bool {
    if creds.expires_at == 0 {
        return !creds.access_token.is_empty();
    }
    now_ms() < (creds.expires_at - 60_000)
}

/// Refreshed Anthropic OAuth token state.
#[derive(Debug, Clone)]
pub struct RefreshedOAuth {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at_ms: i64,
}

pub const OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Refresh an Anthropic OAuth token without mutating local credential files.
///
/// Tries platform.claude.com then console.anthropic.com.
pub fn refresh_anthropic_oauth_pure(
    refresh_token: &str,
    use_json: bool,
) -> Result<RefreshedOAuth, String> {
    if refresh_token.is_empty() {
        return Err("refresh_token is required".to_string());
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    let endpoints = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ];
    let ua = format!("claude-cli/{} (external, cli)", get_claude_code_version());

    let mut last_error: Option<String> = None;
    for endpoint in endpoints {
        let mut req = client
            .post(endpoint)
            .header("User-Agent", ua.clone());

        let resp = if use_json {
            req = req.header("Content-Type", "application/json");
            req.body(
                json!({
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": OAUTH_CLIENT_ID,
                })
                .to_string(),
            )
            .send()
        } else {
            req.form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", OAUTH_CLIENT_ID),
            ])
            .send()
        };

        let result: Value = match resp.and_then(|r| r.json::<Value>()) {
            Ok(v) => v,
            Err(e) => {
                last_error = Some(e.to_string());
                continue;
            }
        };

        let access_token = result
            .get("access_token")
            .and_then(Value::as_str)
            .unwrap_or("");
        if access_token.is_empty() {
            return Err("Anthropic refresh response was missing access_token".to_string());
        }
        let next_refresh = result
            .get("refresh_token")
            .and_then(Value::as_str)
            .unwrap_or(refresh_token)
            .to_string();
        let expires_in = result
            .get("expires_in")
            .and_then(Value::as_i64)
            .unwrap_or(3600);
        return Ok(RefreshedOAuth {
            access_token: access_token.to_string(),
            refresh_token: next_refresh,
            expires_at_ms: now_ms() + expires_in * 1000,
        });
    }

    Err(last_error.unwrap_or_else(|| "Anthropic token refresh failed".to_string()))
}

/// Write refreshed credentials back to ~/.claude/.credentials.json.
pub fn write_claude_code_credentials(
    access_token: &str,
    refresh_token: &str,
    expires_at_ms: i64,
    scopes: Option<Vec<String>>,
) {
    let cred_path = match home_dir() {
        Some(h) => h.join(".claude").join(".credentials.json"),
        None => return,
    };

    let mut existing: Value = if cred_path.exists() {
        std::fs::read_to_string(&cred_path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };

    let mut oauth = Map::new();
    oauth.insert("accessToken".to_string(), json!(access_token));
    oauth.insert("refreshToken".to_string(), json!(refresh_token));
    oauth.insert("expiresAt".to_string(), json!(expires_at_ms));
    if let Some(s) = scopes {
        oauth.insert("scopes".to_string(), json!(s));
    } else if let Some(prev_scopes) = existing
        .get("claudeAiOauth")
        .and_then(|o| o.get("scopes"))
        .cloned()
    {
        oauth.insert("scopes".to_string(), prev_scopes);
    }

    if let Some(obj) = existing.as_object_mut() {
        obj.insert("claudeAiOauth".to_string(), Value::Object(oauth));
    }

    if let Some(parent) = cred_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = cred_path.with_extension("tmp");
    if std::fs::write(
        &tmp,
        serde_json::to_string_pretty(&existing).unwrap_or_default(),
    )
    .is_ok()
    {
        let _ = std::fs::rename(&tmp, &cred_path);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&cred_path, std::fs::Permissions::from_mode(0o600));
        }
    }
}

/// Attempt to refresh an expired Claude Code OAuth token.
pub fn refresh_oauth_token(creds: &ClaudeCodeCredentials) -> Option<String> {
    if creds.refresh_token.is_empty() {
        return None;
    }
    match refresh_anthropic_oauth_pure(&creds.refresh_token, false) {
        Ok(refreshed) => {
            write_claude_code_credentials(
                &refreshed.access_token,
                &refreshed.refresh_token,
                refreshed.expires_at_ms,
                None,
            );
            Some(refreshed.access_token)
        }
        Err(_) => None,
    }
}

/// Resolve a token from Claude Code credential files, refreshing if needed.
pub fn resolve_claude_code_token_from_credentials(
    creds: Option<ClaudeCodeCredentials>,
) -> Option<String> {
    let creds = creds.or_else(read_claude_code_credentials)?;
    if is_claude_code_token_valid(&creds) {
        return Some(creds.access_token.clone());
    }
    refresh_oauth_token(&creds)
}

/// Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.
pub fn prefer_refreshable_claude_code_token(
    env_token: &str,
    creds: &Option<ClaudeCodeCredentials>,
) -> Option<String> {
    if env_token.is_empty() || !is_oauth_token(env_token) {
        return None;
    }
    let creds = match creds {
        Some(c) => c,
        None => return None,
    };
    if creds.refresh_token.is_empty() {
        return None;
    }
    let resolved = resolve_claude_code_token_from_credentials(Some(creds.clone()))?;
    if resolved != env_token {
        Some(resolved)
    } else {
        None
    }
}

/// Resolve an Anthropic token from all available sources.
///
/// Priority: ANTHROPIC_TOKEN → CLAUDE_CODE_OAUTH_TOKEN → Claude Code creds
/// (with refresh) → ANTHROPIC_API_KEY.
pub fn resolve_anthropic_token() -> Option<String> {
    let creds = read_claude_code_credentials();

    let token = std::env::var("ANTHROPIC_TOKEN").unwrap_or_default();
    let token = token.trim();
    if !token.is_empty() {
        if let Some(preferred) = prefer_refreshable_claude_code_token(token, &creds) {
            return Some(preferred);
        }
        return Some(token.to_string());
    }

    let cc_token = std::env::var("CLAUDE_CODE_OAUTH_TOKEN").unwrap_or_default();
    let cc_token = cc_token.trim();
    if !cc_token.is_empty() {
        if let Some(preferred) = prefer_refreshable_claude_code_token(cc_token, &creds) {
            return Some(preferred);
        }
        return Some(cc_token.to_string());
    }

    if let Some(resolved) = resolve_claude_code_token_from_credentials(creds) {
        return Some(resolved);
    }

    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    let api_key = api_key.trim();
    if !api_key.is_empty() {
        return Some(api_key.to_string());
    }

    None
}

// ── Hermes-native PKCE OAuth flow ────────────────────────────────────────

pub const OAUTH_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const OAUTH_REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
pub const OAUTH_SCOPES: &str = "org:create_api_key user:profile user:inference";

/// Generate PKCE code_verifier and code_challenge (S256).
pub fn generate_pkce() -> (String, String) {
    use base64::Engine;
    use sha2::{Digest, Sha256};

    let mut token_bytes = [0u8; 32];
    getrandom_bytes(&mut token_bytes);
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token_bytes);
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

fn getrandom_bytes(buf: &mut [u8]) {
    // Fall back to a chrono-seeded fill if getrandom is unavailable at link
    // time; getrandom is in the dependency tree so the happy path is used.
    if getrandom::fill(buf).is_err() {
        let mut seed = now_ms() as u64 ^ 0x9E3779B97F4A7C15;
        for b in buf.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *b = (seed & 0xff) as u8;
        }
    }
}

/// Build the authorization URL for the Hermes-native OAuth PKCE flow.
///
/// Returns `(auth_url, verifier)`. The caller drives the browser/console
/// interaction; this keeps the I/O-free portion testable.
pub fn build_hermes_oauth_authorize_url(challenge: &str, verifier: &str) -> String {
    let params = [
        ("code", "true"),
        ("client_id", OAUTH_CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", OAUTH_REDIRECT_URI),
        ("scope", OAUTH_SCOPES),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", verifier),
    ];
    let query: String = params
        .iter()
        .map(|(k, v)| {
            format!(
                "{}={}",
                urlencode_component(k),
                urlencode_component(v)
            )
        })
        .collect::<Vec<_>>()
        .join("&");
    format!("https://claude.ai/oauth/authorize?{}", query)
}

fn urlencode_component(s: &str) -> String {
    let mut out = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

/// Exchange an authorization code for tokens (the pure HTTP portion of
/// `run_hermes_oauth_login_pure`). `auth_code` may contain a `#state` suffix.
pub fn exchange_hermes_oauth_code(
    auth_code: &str,
    verifier: &str,
) -> Result<RefreshedOAuth, String> {
    let auth_code = auth_code.trim();
    if auth_code.is_empty() {
        return Err("No code entered.".to_string());
    }
    let mut splits = auth_code.splitn(2, '#');
    let code = splits.next().unwrap_or("");
    let state = splits.next().unwrap_or("");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    let body = json!({
        "grant_type": "authorization_code",
        "client_id": OAUTH_CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "code_verifier": verifier,
    })
    .to_string();

    let resp = client
        .post(OAUTH_TOKEN_URL)
        .header("Content-Type", "application/json")
        .header(
            "User-Agent",
            format!("claude-cli/{} (external, cli)", get_claude_code_version()),
        )
        .body(body)
        .send()
        .map_err(|e| format!("Token exchange failed: {}", e))?;
    let result: Value = resp
        .json()
        .map_err(|e| format!("Token exchange failed: {}", e))?;

    let access_token = result
        .get("access_token")
        .and_then(Value::as_str)
        .unwrap_or("");
    if access_token.is_empty() {
        return Err("No access token in response.".to_string());
    }
    let refresh_token = result
        .get("refresh_token")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let expires_in = result
        .get("expires_in")
        .and_then(Value::as_i64)
        .unwrap_or(3600);
    Ok(RefreshedOAuth {
        access_token: access_token.to_string(),
        refresh_token,
        expires_at_ms: now_ms() + expires_in * 1000,
    })
}

/// Read Hermes-managed OAuth credentials from `<hermes_home>/.anthropic_oauth.json`.
pub fn read_hermes_oauth_credentials(hermes_home: &std::path::Path) -> Option<Value> {
    let path = hermes_home.join(".anthropic_oauth.json");
    if path.exists() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(data) = serde_json::from_str::<Value>(&text) {
                if data
                    .get("accessToken")
                    .and_then(Value::as_str)
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
                {
                    return Some(data);
                }
            }
        }
    }
    None
}

// ── Model-name normalization ─────────────────────────────────────────────

/// Detect AWS Bedrock model IDs that use dots as namespace separators.
pub fn is_bedrock_model_id(model: &str) -> bool {
    let lower = model.to_lowercase();
    if ["global.", "us.", "eu.", "ap.", "jp."]
        .iter()
        .any(|p| lower.starts_with(p))
    {
        return true;
    }
    lower.starts_with("anthropic.")
}

/// Normalize a model name for the Anthropic API.
pub fn normalize_model_name(model: &str, preserve_dots: bool) -> String {
    let lower = model.to_lowercase();
    let mut model = model.to_string();
    if lower.starts_with("anthropic/") {
        model = model["anthropic/".len()..].to_string();
    }
    if !preserve_dots {
        if is_bedrock_model_id(&model) {
            return model;
        }
        let l = model.to_lowercase();
        if l.starts_with("claude-") || l.starts_with("anthropic/") {
            model = model.replace('.', "-");
        }
    }
    model
}

// ── Tool conversion ──────────────────────────────────────────────────────

/// Sanitize a tool call ID for the Anthropic API: [a-zA-Z0-9_-], non-empty.
pub fn sanitize_tool_id(tool_id: &str) -> String {
    if tool_id.is_empty() {
        return "tool_0".to_string();
    }
    let sanitized: String = tool_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "tool_0".to_string()
    } else {
        sanitized
    }
}

/// Collapse `anyOf`/`oneOf` nullable unions to the non-null branch.
/// Port of `tools.schema_sanitizer.strip_nullable_unions` (keep_nullable_hint
/// is configurable; the adapter calls it with `false`).
pub fn strip_nullable_unions(schema: &Value, keep_nullable_hint: bool) -> Value {
    if let Value::Array(items) = schema {
        return Value::Array(
            items
                .iter()
                .map(|i| strip_nullable_unions(i, keep_nullable_hint))
                .collect(),
        );
    }
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return schema.clone(),
    };

    let mut stripped = Map::new();
    for (k, v) in obj {
        stripped.insert(k.clone(), strip_nullable_unions(v, keep_nullable_hint));
    }

    for key in ["anyOf", "oneOf"] {
        let variants = match stripped.get(key).and_then(Value::as_array) {
            Some(v) => v.clone(),
            None => continue,
        };
        let non_null: Vec<&Value> = variants
            .iter()
            .filter(|item| {
                !(item.is_object()
                    && item.get("type").and_then(Value::as_str) == Some("null"))
            })
            .collect();
        if non_null.len() == 1 && non_null.len() != variants.len() {
            let mut replacement = non_null[0]
                .as_object()
                .cloned()
                .unwrap_or_else(Map::new);
            if keep_nullable_hint {
                replacement
                    .entry("nullable".to_string())
                    .or_insert(json!(true));
            }
            for meta_key in ["title", "description", "default", "examples"] {
                if stripped.contains_key(meta_key) && !replacement.contains_key(meta_key) {
                    replacement.insert(meta_key.to_string(), stripped[meta_key].clone());
                }
            }
            return strip_nullable_unions(&Value::Object(replacement), keep_nullable_hint);
        }
    }

    Value::Object(stripped)
}

/// Normalize tool schemas before sending them to Anthropic.
pub fn normalize_tool_input_schema(schema: &Value) -> Value {
    let is_empty = match schema {
        Value::Null => true,
        Value::Object(o) => o.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
    };
    if is_empty {
        return json!({"type": "object", "properties": {}});
    }

    let normalized = strip_nullable_unions(schema, false);
    let mut obj = match normalized.as_object() {
        Some(o) => o.clone(),
        None => return json!({"type": "object", "properties": {}}),
    };

    // Strip top-level union keywords that Anthropic's validator rejects.
    let banned = ["oneOf", "allOf", "anyOf"];
    if banned.iter().any(|b| obj.contains_key(*b)) {
        obj.retain(|k, _| !banned.contains(&k.as_str()));
        obj.entry("type".to_string())
            .or_insert(json!("object"));
    }

    if obj.get("type").and_then(Value::as_str) == Some("object")
        && !obj.get("properties").map(Value::is_object).unwrap_or(false)
    {
        obj.insert("properties".to_string(), json!({}));
    }

    Value::Object(obj)
}

/// Convert OpenAI tool definitions to Anthropic format.
pub fn convert_tools_to_anthropic(tools: &[Value]) -> Vec<Value> {
    if tools.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut seen_names: BTreeSet<String> = BTreeSet::new();
    for t in tools {
        let fn_obj = t.get("function").cloned().unwrap_or_else(|| json!({}));
        let name = fn_obj.get("name").and_then(Value::as_str).unwrap_or("");
        if !name.is_empty() && seen_names.contains(name) {
            log::warn!(
                "convert_tools_to_anthropic: duplicate tool name '{}' — dropping second occurrence",
                name
            );
            continue;
        }
        if !name.is_empty() {
            seen_names.insert(name.to_string());
        }
        let params = fn_obj
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        result.push(json!({
            "name": name,
            "description": fn_obj.get("description").and_then(Value::as_str).unwrap_or(""),
            "input_schema": normalize_tool_input_schema(&params),
        }));
    }
    result
}

// ── Content conversion ─────────────────────────────────────────────────

/// Convert an OpenAI-style image URL/data URL into an Anthropic image source.
pub fn image_source_from_openai_url(url: &str) -> Value {
    let url = url.trim();
    if url.is_empty() {
        return json!({"type": "url", "url": ""});
    }
    if let Some(rest) = url.strip_prefix("data:") {
        // header = up to first comma; data = after.
        let (header_part, data) = match rest.split_once(',') {
            Some((h, d)) => (h, d),
            None => (rest, ""),
        };
        let mut media_type = "image/jpeg".to_string();
        // header_part is the part after "data:"; mime is up to first ';'
        let mime_part = header_part.split(';').next().unwrap_or("").trim();
        if mime_part.starts_with("image/") {
            media_type = mime_part.to_string();
        }
        return json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        });
    }
    json!({"type": "url", "url": url})
}

/// Convert a single OpenAI-style content part to an Anthropic block.
pub fn convert_content_part_to_anthropic(part: &Value) -> Option<Value> {
    match part {
        Value::Null => None,
        Value::String(s) => Some(json!({"type": "text", "text": s})),
        Value::Object(_) => {
            let ptype = part.get("type").and_then(Value::as_str);
            let mut block: Value = match ptype {
                Some("input_text") => json!({
                    "type": "text",
                    "text": part.get("text").and_then(Value::as_str).unwrap_or("")
                }),
                Some("image_url") | Some("input_image") => {
                    let image_value = part.get("image_url");
                    let url = match image_value {
                        Some(Value::Object(o)) => o
                            .get("url")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        Some(Value::String(s)) => s.clone(),
                        Some(other) if !other.is_null() => other.to_string(),
                        _ => String::new(),
                    };
                    json!({"type": "image", "source": image_source_from_openai_url(&url)})
                }
                _ => part.clone(),
            };
            if let Some(cc) = part.get("cache_control") {
                if cc.is_object() {
                    if let Some(b) = block.as_object_mut() {
                        b.entry("cache_control".to_string())
                            .or_insert_with(|| cc.clone());
                    }
                }
            }
            Some(block)
        }
        other => Some(json!({"type": "text", "text": other.to_string()})),
    }
}

/// Return Anthropic thinking blocks previously preserved on the message.
pub fn extract_preserved_thinking_blocks(message: &Value) -> Vec<Value> {
    let raw_details = match message.get("reasoning_details").and_then(Value::as_array) {
        Some(d) => d,
        None => return Vec::new(),
    };
    let mut preserved = Vec::new();
    for detail in raw_details {
        if !detail.is_object() {
            continue;
        }
        let block_type = detail
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if block_type != "thinking" && block_type != "redacted_thinking" {
            continue;
        }
        preserved.push(detail.clone());
    }
    preserved
}

/// Convert OpenAI-style multimodal content arrays to Anthropic blocks.
/// Non-array content is returned unchanged.
pub fn convert_content_to_anthropic(content: &Value) -> Value {
    match content {
        Value::Array(parts) => {
            let mut converted = Vec::new();
            for part in parts {
                if let Some(block) = convert_content_part_to_anthropic(part) {
                    converted.push(block);
                }
            }
            Value::Array(converted)
        }
        other => other.clone(),
    }
}

const THINKING_TYPES: [&str; 2] = ["thinking", "redacted_thinking"];

fn is_thinking_block(b: &Value) -> bool {
    b.is_object()
        && b.get("type")
            .and_then(Value::as_str)
            .map(|t| THINKING_TYPES.contains(&t))
            .unwrap_or(false)
}

/// Convert OpenAI-format messages to Anthropic format.
///
/// Returns `(system, anthropic_messages)`. `system` is `Value::Null`, a
/// string, or an array of content blocks.
pub fn convert_messages_to_anthropic(
    messages: &[Value],
    base_url: Option<&str>,
    model: Option<&str>,
) -> (Value, Vec<Value>) {
    let mut system: Value = Value::Null;
    let mut result: Vec<Value> = Vec::new();

    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").cloned().unwrap_or_else(|| json!(""));

        if role == "system" {
            if let Value::Array(parts) = &content {
                let has_cache = parts
                    .iter()
                    .any(|p| p.is_object() && p.get("cache_control").is_some());
                if has_cache {
                    system = Value::Array(
                        parts.iter().filter(|p| p.is_object()).cloned().collect(),
                    );
                } else {
                    let joined: Vec<String> = parts
                        .iter()
                        .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                        .map(|p| {
                            p.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string()
                        })
                        .collect();
                    system = json!(joined.join("\n"));
                }
            } else {
                system = content;
            }
            continue;
        }

        if role == "assistant" {
            let mut blocks = extract_preserved_thinking_blocks(m);
            let content_truthy = is_truthy(&content);
            if content_truthy {
                if let Value::Array(_) = &content {
                    if let Value::Array(converted) = convert_content_to_anthropic(&content) {
                        blocks.extend(converted);
                    }
                } else {
                    blocks.push(json!({"type": "text", "text": value_to_str(&content)}));
                }
            }
            if let Some(tool_calls) = m.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    if !tc.is_object() {
                        continue;
                    }
                    let fn_obj = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                    let args = fn_obj.get("arguments").cloned().unwrap_or_else(|| json!("{}"));
                    let parsed_args = match &args {
                        Value::String(s) => {
                            serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}))
                        }
                        other => other.clone(),
                    };
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": sanitize_tool_id(tc.get("id").and_then(Value::as_str).unwrap_or("")),
                        "name": fn_obj.get("name").and_then(Value::as_str).unwrap_or(""),
                        "input": parsed_args,
                    }));
                }
            }

            // Kimi reasoning_content → thinking block (only if not already present).
            let already_has_thinking = blocks.iter().any(is_thinking_block);
            if let Some(rc) = m.get("reasoning_content").and_then(Value::as_str) {
                if !already_has_thinking {
                    blocks.insert(0, json!({"type": "thinking", "thinking": rc}));
                }
            }

            let effective: Value = if !blocks.is_empty() {
                Value::Array(blocks)
            } else {
                content.clone()
            };
            let effective = if !is_truthy(&effective) {
                json!([{"type": "text", "text": "(empty)"}])
            } else {
                effective
            };
            result.push(json!({"role": "assistant", "content": effective}));
            continue;
        }

        if role == "tool" {
            let result_content = match &content {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            let result_content = if result_content.is_empty() {
                "(no output)".to_string()
            } else {
                result_content
            };
            let mut tool_result = json!({
                "type": "tool_result",
                "tool_use_id": sanitize_tool_id(m.get("tool_call_id").and_then(Value::as_str).unwrap_or("")),
                "content": result_content,
            });
            if let Some(cc) = m.get("cache_control") {
                if cc.is_object() {
                    tool_result
                        .as_object_mut()
                        .unwrap()
                        .insert("cache_control".to_string(), cc.clone());
                }
            }

            // Merge consecutive tool results into the prior user message.
            let merge = result
                .last()
                .map(|last| {
                    last.get("role").and_then(Value::as_str) == Some("user")
                        && last
                            .get("content")
                            .and_then(Value::as_array)
                            .map(|c| {
                                !c.is_empty()
                                    && c[0].get("type").and_then(Value::as_str)
                                        == Some("tool_result")
                            })
                            .unwrap_or(false)
                })
                .unwrap_or(false);
            if merge {
                let last = result.last_mut().unwrap();
                last.get_mut("content")
                    .and_then(Value::as_array_mut)
                    .unwrap()
                    .push(tool_result);
            } else {
                result.push(json!({"role": "user", "content": [tool_result]}));
            }
            continue;
        }

        // Regular user message.
        if let Value::Array(_) = &content {
            let mut converted_blocks = match convert_content_to_anthropic(&content) {
                Value::Array(v) => v,
                _ => Vec::new(),
            };
            let all_text_empty = converted_blocks.is_empty()
                || converted_blocks
                    .iter()
                    .filter(|b| {
                        b.is_object()
                            && b.get("type").and_then(Value::as_str) == Some("text")
                    })
                    .all(|b| {
                        b.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim()
                            .is_empty()
                    });
            if all_text_empty {
                converted_blocks = vec![json!({"type": "text", "text": "(empty message)"})];
            }
            result.push(json!({"role": "user", "content": converted_blocks}));
        } else {
            let content = match &content {
                Value::String(s) if !s.trim().is_empty() => json!(s),
                Value::String(_) => json!("(empty message)"),
                other if is_truthy(other) => other.clone(),
                _ => json!("(empty message)"),
            };
            result.push(json!({"role": "user", "content": content}));
        }
    }

    strip_orphans_and_alternate(&mut result);
    manage_thinking_signatures(&mut result, base_url, model);

    (system, result)
}

fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn strip_orphans_and_alternate(result: &mut Vec<Value>) {
    // Strip orphaned tool_use blocks (no matching tool_result follows).
    let mut tool_result_ids: BTreeSet<String> = BTreeSet::new();
    for m in result.iter() {
        if m.get("role").and_then(Value::as_str) == Some("user") {
            if let Some(content) = m.get("content").and_then(Value::as_array) {
                for block in content {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                            tool_result_ids.insert(id.to_string());
                        }
                    }
                }
            }
        }
    }
    for m in result.iter_mut() {
        if m.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(content) = m.get("content").and_then(Value::as_array).cloned() {
                let filtered: Vec<Value> = content
                    .into_iter()
                    .filter(|b| {
                        b.get("type").and_then(Value::as_str) != Some("tool_use")
                            || b.get("id")
                                .and_then(Value::as_str)
                                .map(|id| tool_result_ids.contains(id))
                                .unwrap_or(false)
                    })
                    .collect();
                let filtered = if filtered.is_empty() {
                    vec![json!({"type": "text", "text": "(tool call removed)"})]
                } else {
                    filtered
                };
                m.as_object_mut()
                    .unwrap()
                    .insert("content".to_string(), Value::Array(filtered));
            }
        }
    }

    // Strip orphaned tool_result blocks (no matching tool_use precedes them).
    let mut tool_use_ids: BTreeSet<String> = BTreeSet::new();
    for m in result.iter() {
        if m.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(content) = m.get("content").and_then(Value::as_array) {
                for block in content {
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        if let Some(id) = block.get("id").and_then(Value::as_str) {
                            tool_use_ids.insert(id.to_string());
                        }
                    }
                }
            }
        }
    }
    for m in result.iter_mut() {
        if m.get("role").and_then(Value::as_str) == Some("user") {
            if let Some(content) = m.get("content").and_then(Value::as_array).cloned() {
                let filtered: Vec<Value> = content
                    .into_iter()
                    .filter(|b| {
                        b.get("type").and_then(Value::as_str) != Some("tool_result")
                            || b.get("tool_use_id")
                                .and_then(Value::as_str)
                                .map(|id| tool_use_ids.contains(id))
                                .unwrap_or(false)
                    })
                    .collect();
                let filtered = if filtered.is_empty() {
                    vec![json!({"type": "text", "text": "(tool result removed)"})]
                } else {
                    filtered
                };
                m.as_object_mut()
                    .unwrap()
                    .insert("content".to_string(), Value::Array(filtered));
            }
        }
    }

    // Enforce strict role alternation by merging consecutive same-role messages.
    let mut fixed: Vec<Value> = Vec::new();
    for mut m in result.drain(..) {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
        let same_role = fixed
            .last()
            .map(|p| p.get("role").and_then(Value::as_str) == Some(role.as_str()))
            .unwrap_or(false);
        if same_role {
            if role == "user" {
                let prev_content = fixed.last().unwrap().get("content").cloned().unwrap();
                let curr_content = m.get("content").cloned().unwrap();
                let merged = merge_contents(prev_content, curr_content, false);
                fixed
                    .last_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("content".to_string(), merged);
            } else {
                // Consecutive assistant: drop thinking blocks from second.
                if let Some(arr) = m.get("content").and_then(Value::as_array).cloned() {
                    let stripped: Vec<Value> =
                        arr.into_iter().filter(|b| !is_thinking_block(b)).collect();
                    m.as_object_mut()
                        .unwrap()
                        .insert("content".to_string(), Value::Array(stripped));
                }
                let prev_content = fixed.last().unwrap().get("content").cloned().unwrap();
                let curr_content = m.get("content").cloned().unwrap();
                let merged = merge_contents(prev_content, curr_content, true);
                fixed
                    .last_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("content".to_string(), merged);
            }
        } else {
            fixed.push(m);
        }
    }
    *result = fixed;
}

fn merge_contents(prev: Value, curr: Value, _assistant: bool) -> Value {
    match (&prev, &curr) {
        (Value::String(p), Value::String(c)) => json!(format!("{}\n{}", p, c)),
        (Value::Array(p), Value::Array(c)) => {
            let mut merged = p.clone();
            merged.extend(c.clone());
            Value::Array(merged)
        }
        _ => {
            let p_list = match prev {
                Value::String(s) => vec![json!({"type": "text", "text": s})],
                Value::Array(a) => a,
                other => vec![other],
            };
            let c_list = match curr {
                Value::String(s) => vec![json!({"type": "text", "text": s})],
                Value::Array(a) => a,
                other => vec![other],
            };
            let mut merged = p_list;
            merged.extend(c_list);
            Value::Array(merged)
        }
    }
}

fn manage_thinking_signatures(result: &mut [Value], base_url: Option<&str>, model: Option<&str>) {
    let is_third_party = is_third_party_anthropic_endpoint(base_url);
    let preserve_unsigned_thinking = is_kimi_family_endpoint(base_url, model)
        || is_deepseek_anthropic_endpoint(base_url);

    let last_assistant_idx = result
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("assistant"));

    for idx in 0..result.len() {
        {
            let m = &result[idx];
            if m.get("role").and_then(Value::as_str) != Some("assistant")
                || !m.get("content").map(Value::is_array).unwrap_or(false)
            {
                continue;
            }
        }
        let content = result[idx]
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let new_content: Vec<Value> = if preserve_unsigned_thinking {
            let mut nc = Vec::new();
            for b in content {
                if !is_thinking_block(&b) {
                    nc.push(b);
                    continue;
                }
                let has_sig = b.get("signature").map(is_truthy).unwrap_or(false)
                    || b.get("data").map(is_truthy).unwrap_or(false);
                if has_sig {
                    continue;
                }
                nc.push(b);
            }
            if nc.is_empty() {
                vec![json!({"type": "text", "text": "(empty)"})]
            } else {
                nc
            }
        } else if is_third_party || Some(idx) != last_assistant_idx {
            let stripped: Vec<Value> =
                content.into_iter().filter(|b| !is_thinking_block(b)).collect();
            if stripped.is_empty() {
                vec![json!({"type": "text", "text": "(thinking elided)"})]
            } else {
                stripped
            }
        } else {
            // Latest assistant on direct Anthropic.
            let mut nc = Vec::new();
            for b in content {
                if !is_thinking_block(&b) {
                    nc.push(b);
                    continue;
                }
                let btype = b.get("type").and_then(Value::as_str).unwrap_or("");
                if btype == "redacted_thinking" {
                    if b.get("data").map(is_truthy).unwrap_or(false) {
                        nc.push(b);
                    }
                } else if b.get("signature").map(is_truthy).unwrap_or(false) {
                    nc.push(b);
                } else {
                    let thinking_text = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                    if !thinking_text.is_empty() {
                        nc.push(json!({"type": "text", "text": thinking_text}));
                    }
                }
            }
            if nc.is_empty() {
                vec![json!({"type": "text", "text": "(empty)"})]
            } else {
                nc
            }
        };

        // Strip cache_control from remaining thinking blocks.
        let new_content: Vec<Value> = new_content
            .into_iter()
            .map(|mut b| {
                if is_thinking_block(&b) {
                    if let Some(o) = b.as_object_mut() {
                        o.remove("cache_control");
                    }
                }
                b
            })
            .collect();

        result[idx]
            .as_object_mut()
            .unwrap()
            .insert("content".to_string(), Value::Array(new_content));
    }
}

// ── build_anthropic_kwargs ───────────────────────────────────────────────

/// Reasoning config: `{enabled?: bool, effort?: str}`.
#[derive(Debug, Clone, Default)]
pub struct ReasoningConfig {
    pub enabled: Option<bool>,
    pub effort: Option<String>,
}

/// OpenAI-style tool_choice.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Tool(String),
}

/// Build kwargs for an Anthropic `messages.create()` call.
///
/// Faithful port of `build_anthropic_kwargs`. Returns a JSON object suitable
/// for serialization into the request body. `extra_headers` (per-request beta
/// override for fast mode) is folded into the returned object under the
/// `extra_headers` key, matching the Python behavior.
#[allow(clippy::too_many_arguments)]
pub fn build_anthropic_kwargs(
    model: &str,
    messages: &[Value],
    tools: Option<&[Value]>,
    max_tokens: &Value,
    reasoning_config: Option<&ReasoningConfig>,
    tool_choice: Option<&ToolChoice>,
    is_oauth: bool,
    preserve_dots: bool,
    context_length: Option<i64>,
    base_url: Option<&str>,
    fast_mode: bool,
    drop_context_1m_beta: bool,
) -> Result<Value, String> {
    let (mut system, mut anthropic_messages) =
        convert_messages_to_anthropic(messages, base_url, Some(model));
    let mut anthropic_tools = match tools {
        Some(t) if !t.is_empty() => convert_tools_to_anthropic(t),
        _ => Vec::new(),
    };

    let model_norm = normalize_model_name(model, preserve_dots);
    let model = model_norm.as_str();

    let mut effective_max_tokens =
        resolve_anthropic_messages_max_tokens(max_tokens, model)?;

    if let Some(ctx) = context_length {
        if ctx > 0 && effective_max_tokens > ctx {
            effective_max_tokens = std::cmp::max(ctx - 1, 1);
        }
    }

    // ── OAuth: Claude Code identity ──
    if is_oauth {
        let cc_block = json!({"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX});
        system = match system {
            Value::Array(arr) => {
                let mut v = vec![cc_block];
                v.extend(arr);
                Value::Array(v)
            }
            Value::String(s) if !s.is_empty() => {
                json!([cc_block, {"type": "text", "text": s}])
            }
            _ => json!([cc_block]),
        };

        // Sanitize product-name references.
        if let Value::Array(blocks) = &mut system {
            for block in blocks.iter_mut() {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                    let text = text
                        .replace("Hermes Agent", "Claude Code")
                        .replace("Hermes agent", "Claude Code")
                        .replace("hermes-agent", "claude-code")
                        .replace("Nous Research", "Anthropic");
                    block
                        .as_object_mut()
                        .unwrap()
                        .insert("text".to_string(), json!(text));
                }
            }
        }

        // Prefix tool names with mcp_.
        for tool in anthropic_tools.iter_mut() {
            if tool.get("name").is_some() {
                let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                tool.as_object_mut().unwrap().insert(
                    "name".to_string(),
                    json!(format!("{}{}", MCP_TOOL_PREFIX, name)),
                );
            }
        }

        // Prefix tool_use names in message history.
        for msg in anthropic_messages.iter_mut() {
            if let Some(content) = msg.get_mut("content").and_then(Value::as_array_mut) {
                for block in content.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        if let Some(name) = block.get("name").and_then(Value::as_str) {
                            if !name.starts_with(MCP_TOOL_PREFIX) {
                                let new_name = format!("{}{}", MCP_TOOL_PREFIX, name);
                                block
                                    .as_object_mut()
                                    .unwrap()
                                    .insert("name".to_string(), json!(new_name));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut kwargs = Map::new();
    kwargs.insert("model".to_string(), json!(model));
    kwargs.insert("messages".to_string(), Value::Array(anthropic_messages));
    kwargs.insert("max_tokens".to_string(), json!(effective_max_tokens));

    if is_truthy(&system) {
        kwargs.insert("system".to_string(), system);
    }

    if !anthropic_tools.is_empty() {
        kwargs.insert("tools".to_string(), Value::Array(anthropic_tools));
        match tool_choice {
            Some(ToolChoice::Auto) | None => {
                kwargs.insert("tool_choice".to_string(), json!({"type": "auto"}));
            }
            Some(ToolChoice::Required) => {
                kwargs.insert("tool_choice".to_string(), json!({"type": "any"}));
            }
            Some(ToolChoice::None) => {
                kwargs.remove("tools");
            }
            Some(ToolChoice::Tool(name)) => {
                kwargs.insert(
                    "tool_choice".to_string(),
                    json!({"type": "tool", "name": name}),
                );
            }
        }
    }

    // ── reasoning_config → thinking ──
    let is_kimi_coding = is_kimi_family_endpoint(base_url, Some(model));
    if let Some(rc) = reasoning_config {
        if !is_kimi_coding && rc.enabled != Some(false) && !model.to_lowercase().contains("haiku") {
            let effort = rc
                .effort
                .clone()
                .unwrap_or_else(|| "medium".to_string())
                .to_lowercase();
            let budget = thinking_budget(&effort).unwrap_or(8000);
            if supports_adaptive_thinking(model) {
                kwargs.insert(
                    "thinking".to_string(),
                    json!({"type": "adaptive", "display": "summarized"}),
                );
                let mut adaptive_effort = adaptive_effort_map(&effort);
                if adaptive_effort == "xhigh" && !supports_xhigh_effort(model) {
                    adaptive_effort = "max";
                }
                kwargs.insert(
                    "output_config".to_string(),
                    json!({"effort": adaptive_effort}),
                );
            } else {
                kwargs.insert(
                    "thinking".to_string(),
                    json!({"type": "enabled", "budget_tokens": budget}),
                );
                kwargs.insert("temperature".to_string(), json!(1));
                let new_max = std::cmp::max(effective_max_tokens, budget + 4096);
                kwargs.insert("max_tokens".to_string(), json!(new_max));
            }
        }
    }

    // ── Strip sampling params on 4.7+ ──
    if forbids_sampling_params(model) {
        for key in ["temperature", "top_p", "top_k"] {
            kwargs.remove(key);
        }
    }

    // ── Fast mode (Opus 4.6 only) ──
    if fast_mode && !is_third_party_anthropic_endpoint(base_url) {
        let mut betas = common_betas_for_base_url(base_url, drop_context_1m_beta);
        if is_oauth {
            betas.extend(OAUTH_ONLY_BETAS.iter().map(|s| s.to_string()));
        }
        if supports_fast_mode(model) {
            let extra_body = kwargs
                .entry("extra_body".to_string())
                .or_insert_with(|| json!({}));
            extra_body
                .as_object_mut()
                .unwrap()
                .insert("speed".to_string(), json!("fast"));
            betas.push(FAST_MODE_BETA.to_string());
        }
        kwargs.insert(
            "extra_headers".to_string(),
            json!({"anthropic-beta": betas.join(",")}),
        );
    }

    Ok(Value::Object(kwargs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_anthropic_max_output_longest_prefix() {
        assert_eq!(get_anthropic_max_output("claude-3-5-sonnet-20241022"), 8_192);
        assert_eq!(get_anthropic_max_output("claude-opus-4-7"), 128_000);
        // dot normalization
        assert_eq!(
            get_anthropic_max_output("anthropic/claude-opus-4.6"),
            128_000
        );
        // unknown → default
        assert_eq!(get_anthropic_max_output("mystery-model"), 128_000);
        assert_eq!(get_anthropic_max_output("claude-sonnet-4-6"), 64_000);
    }

    #[test]
    fn test_resolve_positive_max_tokens() {
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(100)), Some(100));
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(0)), None);
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(-5)), None);
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(0.5)), None);
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(3.9)), Some(3));
        assert_eq!(resolve_positive_anthropic_max_tokens(&json!(true)), None);
        assert_eq!(resolve_positive_anthropic_max_tokens(&Value::Null), None);
    }

    #[test]
    fn test_is_oauth_token() {
        assert!(!is_oauth_token(""));
        assert!(!is_oauth_token("sk-ant-api03-xxx"));
        assert!(is_oauth_token("sk-ant-oat01-xxx"));
        assert!(is_oauth_token("eyJabc"));
        assert!(is_oauth_token("cc-token"));
        assert!(!is_oauth_token("minimax-key"));
    }

    #[test]
    fn test_endpoint_detection() {
        assert!(!is_third_party_anthropic_endpoint(None));
        assert!(!is_third_party_anthropic_endpoint(Some(
            "https://api.anthropic.com/v1"
        )));
        assert!(is_third_party_anthropic_endpoint(Some(
            "https://api.minimax.io/anthropic"
        )));
        assert!(requires_bearer_auth(Some("https://api.minimax.io/anthropic")));
        assert!(requires_bearer_auth(Some(
            "https://api.minimaxi.com/anthropic/"
        )));
        assert!(!requires_bearer_auth(Some("https://api.anthropic.com")));
        assert!(is_kimi_coding_endpoint(Some(
            "https://api.kimi.com/coding/v1"
        )));
    }

    #[test]
    fn test_base_url_host_matches() {
        assert!(base_url_host_matches(
            "https://api.moonshot.ai/v1",
            "moonshot.ai"
        ));
        assert!(base_url_host_matches("https://moonshot.ai", "moonshot.ai"));
        assert!(!base_url_host_matches(
            "https://evil.com/moonshot.ai/v1",
            "moonshot.ai"
        ));
        assert!(!base_url_host_matches(
            "https://moonshot.ai.evil/v1",
            "moonshot.ai"
        ));
    }

    #[test]
    fn test_kimi_family() {
        assert!(model_name_is_kimi_family(Some("kimi-k2.5")));
        assert!(model_name_is_kimi_family(Some("moonshotai/kimi-k2.5")));
        assert!(model_name_is_kimi_family(Some("k2-thinking")));
        assert!(!model_name_is_kimi_family(Some("claude-opus-4-7")));
        assert!(is_kimi_family_endpoint(None, Some("kimi-k2.5")));
    }

    #[test]
    fn test_deepseek_endpoint() {
        assert!(is_deepseek_anthropic_endpoint(Some(
            "https://api.deepseek.com/anthropic"
        )));
        assert!(!is_deepseek_anthropic_endpoint(Some(
            "https://api.deepseek.com/v1"
        )));
    }

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(
            normalize_model_name("anthropic/claude-opus-4.6", false),
            "claude-opus-4-6"
        );
        assert_eq!(
            normalize_model_name("us.anthropic.claude-sonnet-4-5-v1:0", false),
            "us.anthropic.claude-sonnet-4-5-v1:0"
        );
        assert_eq!(normalize_model_name("gpt-5.4", false), "gpt-5.4");
        assert_eq!(
            normalize_model_name("qwen3.5-plus", true),
            "qwen3.5-plus"
        );
    }

    #[test]
    fn test_sanitize_tool_id() {
        assert_eq!(sanitize_tool_id(""), "tool_0");
        assert_eq!(sanitize_tool_id("call_abc-123"), "call_abc-123");
        assert_eq!(sanitize_tool_id("call abc!"), "call_abc_");
    }

    #[test]
    fn test_normalize_tool_input_schema_strips_nullable() {
        let schema = json!({
            "type": "object",
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            }
        });
        let normalized = normalize_tool_input_schema(&schema);
        let x = &normalized["properties"]["x"];
        assert_eq!(x["type"], json!("string"));
        assert!(x.get("anyOf").is_none());
        // nullable hint not kept (keep_nullable_hint=false)
        assert!(x.get("nullable").is_none());
    }

    #[test]
    fn test_normalize_tool_input_schema_top_level_union() {
        let schema = json!({"oneOf": [{"type": "object"}, {"type": "string"}]});
        let normalized = normalize_tool_input_schema(&schema);
        assert!(normalized.get("oneOf").is_none());
        assert_eq!(normalized["type"], json!("object"));
    }

    #[test]
    fn test_normalize_tool_input_schema_empty() {
        let n = normalize_tool_input_schema(&json!({}));
        assert_eq!(n, json!({"type": "object", "properties": {}}));
        let n2 = normalize_tool_input_schema(&Value::Null);
        assert_eq!(n2, json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn test_convert_tools_dedup() {
        let tools = vec![
            json!({"function": {"name": "a", "description": "first", "parameters": {"type": "object"}}}),
            json!({"function": {"name": "a", "description": "dup", "parameters": {"type": "object"}}}),
            json!({"function": {"name": "b", "description": "", "parameters": {}}}),
        ];
        let result = convert_tools_to_anthropic(&tools);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["name"], json!("a"));
        assert_eq!(result[1]["name"], json!("b"));
    }

    #[test]
    fn test_image_source_data_url() {
        let src = image_source_from_openai_url("data:image/png;base64,AAAA");
        assert_eq!(src["type"], json!("base64"));
        assert_eq!(src["media_type"], json!("image/png"));
        assert_eq!(src["data"], json!("AAAA"));

        let src2 = image_source_from_openai_url("https://x.com/a.png");
        assert_eq!(src2["type"], json!("url"));
        assert_eq!(src2["url"], json!("https://x.com/a.png"));

        let src3 = image_source_from_openai_url("");
        assert_eq!(src3, json!({"type": "url", "url": ""}));
    }

    #[test]
    fn test_convert_messages_system_extraction() {
        let messages = vec![
            json!({"role": "system", "content": "you are helpful"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let (system, msgs) = convert_messages_to_anthropic(&messages, None, None);
        assert_eq!(system, json!("you are helpful"));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("user"));
        assert_eq!(msgs[0]["content"], json!("hi"));
    }

    #[test]
    fn test_convert_messages_tool_call_roundtrip() {
        let messages = vec![
            json!({"role": "user", "content": "do it"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "f", "arguments": "{\"a\": 1}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "result"}),
        ];
        let (_system, msgs) = convert_messages_to_anthropic(&messages, None, None);
        // user, assistant (tool_use), user (tool_result)
        assert_eq!(msgs.len(), 3);
        let tu = &msgs[1]["content"][0];
        assert_eq!(tu["type"], json!("tool_use"));
        assert_eq!(tu["id"], json!("call_1"));
        assert_eq!(tu["input"], json!({"a": 1}));
        let tr = &msgs[2]["content"][0];
        assert_eq!(tr["type"], json!("tool_result"));
        assert_eq!(tr["tool_use_id"], json!("call_1"));
    }

    #[test]
    fn test_orphan_tool_use_stripped() {
        // assistant tool_use with no matching tool_result → removed
        let messages = vec![
            json!({"role": "user", "content": "x"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "orphan", "function": {"name": "f", "arguments": "{}"}}]
            }),
        ];
        let (_s, msgs) = convert_messages_to_anthropic(&messages, None, None);
        let asst = msgs.iter().find(|m| m["role"] == json!("assistant")).unwrap();
        assert_eq!(asst["content"][0]["text"], json!("(tool call removed)"));
    }

    #[test]
    fn test_role_alternation_merge() {
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "user", "content": "b"}),
        ];
        let (_s, msgs) = convert_messages_to_anthropic(&messages, None, None);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], json!("a\nb"));
    }

    #[test]
    fn test_thinking_third_party_stripped() {
        let messages = vec![json!({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret", "signature": "sig"},
                {"type": "text", "text": "answer"}
            ]
        })];
        let (_s, msgs) = convert_messages_to_anthropic(
            &messages,
            Some("https://api.minimax.io/anthropic"),
            None,
        );
        let content = msgs[0]["content"].as_array().unwrap();
        assert!(content.iter().all(|b| b["type"] != json!("thinking")));
        assert_eq!(content[0]["text"], json!("answer"));
    }

    #[test]
    fn test_build_kwargs_basic() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            None,
            &json!(1000),
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(kwargs["model"], json!("claude-opus-4-7"));
        assert_eq!(kwargs["max_tokens"], json!(1000));
        assert!(kwargs.get("system").is_none());
    }

    #[test]
    fn test_build_kwargs_adaptive_thinking() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let rc = ReasoningConfig {
            enabled: Some(true),
            effort: Some("xhigh".to_string()),
        };
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            None,
            &Value::Null,
            Some(&rc),
            None,
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(kwargs["thinking"]["type"], json!("adaptive"));
        assert_eq!(kwargs["thinking"]["display"], json!("summarized"));
        assert_eq!(kwargs["output_config"]["effort"], json!("xhigh"));
        // 4.7 forbids sampling params; none should be present anyway
        assert!(kwargs.get("temperature").is_none());
        // max_tokens defaults to model ceiling
        assert_eq!(kwargs["max_tokens"], json!(128_000));
    }

    #[test]
    fn test_build_kwargs_xhigh_downgrade_on_46() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let rc = ReasoningConfig {
            enabled: None,
            effort: Some("xhigh".to_string()),
        };
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-6",
            &messages,
            None,
            &json!(2000),
            Some(&rc),
            None,
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        // 4.6 does not support xhigh → downgrade to max
        assert_eq!(kwargs["output_config"]["effort"], json!("max"));
    }

    #[test]
    fn test_build_kwargs_manual_thinking_old_model() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let rc = ReasoningConfig {
            enabled: Some(true),
            effort: Some("low".to_string()),
        };
        let kwargs = build_anthropic_kwargs(
            "claude-3-5-sonnet",
            &messages,
            None,
            &json!(1000),
            Some(&rc),
            None,
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(kwargs["thinking"]["type"], json!("enabled"));
        assert_eq!(kwargs["thinking"]["budget_tokens"], json!(4000));
        assert_eq!(kwargs["temperature"], json!(1));
        // max(1000, 4000+4096) = 8096
        assert_eq!(kwargs["max_tokens"], json!(8096));
    }

    #[test]
    fn test_build_kwargs_oauth_transforms() {
        let messages = vec![
            json!({"role": "system", "content": "Hermes Agent helps with Nous Research."}),
            json!({"role": "user", "content": "hi"}),
        ];
        let tools = vec![json!({"function": {"name": "search", "description": "", "parameters": {}}})];
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            Some(&tools),
            &json!(1000),
            None,
            None,
            true,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        let system = kwargs["system"].as_array().unwrap();
        assert_eq!(system[0]["text"], json!(CLAUDE_CODE_SYSTEM_PREFIX));
        assert_eq!(
            system[1]["text"],
            json!("Claude Code helps with Anthropic.")
        );
        assert_eq!(kwargs["tools"][0]["name"], json!("mcp_search"));
    }

    #[test]
    fn test_build_kwargs_tool_choice() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"function": {"name": "f", "description": "", "parameters": {}}})];
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            Some(&tools),
            &json!(1000),
            None,
            Some(&ToolChoice::Required),
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(kwargs["tool_choice"], json!({"type": "any"}));

        let kwargs_none = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            Some(&tools),
            &json!(1000),
            None,
            Some(&ToolChoice::None),
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert!(kwargs_none.get("tools").is_none());
        assert!(kwargs_none.get("tool_choice").is_none());
    }

    #[test]
    fn test_context_length_clamp() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let kwargs = build_anthropic_kwargs(
            "claude-opus-4-7",
            &messages,
            None,
            &Value::Null,
            None,
            None,
            false,
            false,
            Some(8000),
            None,
            false,
            false,
        )
        .unwrap();
        // ceiling 128000 > 8000 → clamp to 7999
        assert_eq!(kwargs["max_tokens"], json!(7999));
    }

    #[test]
    fn test_common_betas_bearer_strips() {
        let betas = common_betas_for_base_url(Some("https://api.minimax.io/anthropic"), false);
        assert!(!betas.contains(&TOOL_STREAMING_BETA.to_string()));
        assert!(!betas.contains(&CONTEXT_1M_BETA.to_string()));
        assert!(betas.contains(&"interleaved-thinking-2025-05-14".to_string()));

        let dropped = common_betas_for_base_url(None, true);
        assert!(!dropped.contains(&CONTEXT_1M_BETA.to_string()));
        assert!(dropped.contains(&TOOL_STREAMING_BETA.to_string()));
    }

    #[test]
    fn test_client_config_oauth() {
        let cfg = build_anthropic_client_config("sk-ant-oat01-xyz", None, None, false);
        assert!(matches!(cfg.auth, AnthropicAuth::Bearer(_)));
        let beta = cfg
            .default_headers
            .iter()
            .find(|(k, _)| k == "anthropic-beta")
            .unwrap();
        assert!(beta.1.contains("oauth-2025-04-20"));
        assert!(cfg.default_headers.iter().any(|(k, _)| k == "x-app"));
    }

    #[test]
    fn test_client_config_api_key() {
        let cfg = build_anthropic_client_config("sk-ant-api03-xyz", None, Some(60.0), false);
        assert!(matches!(cfg.auth, AnthropicAuth::ApiKey(_)));
        assert_eq!(cfg.read_timeout_secs, 60.0);
        assert_eq!(cfg.connect_timeout_secs, 10.0);
    }

    #[test]
    fn test_client_config_azure_query() {
        let cfg = build_anthropic_client_config(
            "key",
            Some("https://x.openai.azure.com/anthropic"),
            None,
            false,
        );
        assert!(cfg
            .default_query
            .iter()
            .any(|(k, v)| k == "api-version" && v == "2025-04-15"));
    }

    #[test]
    fn test_client_config_kimi_ua() {
        let cfg = build_anthropic_client_config("key", Some("https://api.kimi.com/coding"), None, false);
        assert!(cfg
            .default_headers
            .iter()
            .any(|(k, v)| k == "User-Agent" && v == "claude-code/0.1.0"));
        assert!(matches!(cfg.auth, AnthropicAuth::ApiKey(_)));
    }

    #[test]
    fn test_pkce_generation() {
        let (verifier, challenge) = generate_pkce();
        assert!(!verifier.is_empty());
        assert!(!challenge.is_empty());
        // base64url no padding
        assert!(!verifier.contains('='));
        assert!(!challenge.contains('='));
    }

    #[test]
    fn test_authorize_url() {
        let url = build_hermes_oauth_authorize_url("chal", "verif");
        assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
        assert!(url.contains("code_challenge=chal"));
        assert!(url.contains("state=verif"));
        assert!(url.contains("code_challenge_method=S256"));
        // scope is url-encoded (spaces → %20)
        assert!(url.contains("scope=org%3Acreate_api_key"));
    }

    #[test]
    fn test_kimi_skips_thinking() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let rc = ReasoningConfig {
            enabled: Some(true),
            effort: Some("high".to_string()),
        };
        let kwargs = build_anthropic_kwargs(
            "kimi-k2.5",
            &messages,
            None,
            &json!(1000),
            Some(&rc),
            None,
            false,
            false,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        // Kimi family → thinking is skipped entirely
        assert!(kwargs.get("thinking").is_none());
        assert!(kwargs.get("output_config").is_none());
    }
}
