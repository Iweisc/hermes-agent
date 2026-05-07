use std::env;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use chrono::Utc;
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::HermesError;

const AUTH_STORE_VERSION: i64 = 1;
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
const DEFAULT_COPILOT_ACP_BASE_URL: &str = "acp://copilot";
const DEFAULT_COPILOT_ACP_COMMAND: &str = "copilot";
const GOOGLE_OAUTH_CLIENT_ID_ENV: &str = "HERMES_GEMINI_CLIENT_ID";
const GOOGLE_OAUTH_CLIENT_SECRET_ENV: &str = "HERMES_GEMINI_CLIENT_SECRET";
const GOOGLE_OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 60;
const GOOGLE_DEFAULT_CLIENT_ID: &str =
    "000000000000-hermes-placeholder.apps.googleusercontent.com";
const GOOGLE_DEFAULT_CLIENT_SECRET: &str = "hermes-placeholder-google-client-secret";
const MINIMAX_OAUTH_REFRESH_SKEW_SECONDS: i64 = 60;
const DEFAULT_QWEN_BASE_URL: &str = "https://portal.qwen.ai/v1";
const QWEN_OAUTH_CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
const QWEN_OAUTH_TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
const QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CodexTokens {
    access_token: String,
    refresh_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimaxOAuthRuntimeCredentials {
    pub access_token: String,
    pub base_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopilotAcpRuntimeCredentials {
    pub base_url: String,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleGeminiRuntimeCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub project_id: String,
    pub managed_project_id: String,
    pub email: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenRuntimeCredentials {
    pub access_token: String,
    pub base_url: String,
}

pub fn resolve_codex_access_token(hermes_home: &Path) -> Result<String, HermesError> {
    resolve_codex_access_token_with_refresh_url(hermes_home, CODEX_OAUTH_TOKEN_URL)
}

pub fn codex_cloudflare_headers(access_token: &str) -> Vec<(String, String)> {
    let mut headers = vec![
        (
            "User-Agent".to_string(),
            "codex_cli_rs/0.0.0 (Hermes Agent)".to_string(),
        ),
        ("originator".to_string(), "codex_cli_rs".to_string()),
    ];
    if let Some(account_id) = chatgpt_account_id_from_token(access_token) {
        headers.push(("ChatGPT-Account-ID".to_string(), account_id));
    }
    headers
}

pub fn resolve_minimax_oauth_runtime_credentials(
    hermes_home: &Path,
) -> Result<MinimaxOAuthRuntimeCredentials, HermesError> {
    resolve_minimax_oauth_runtime_credentials_with_client(hermes_home, &Client::new())
}

pub fn resolve_copilot_acp_runtime_credentials() -> Result<CopilotAcpRuntimeCredentials, HermesError>
{
    let base_url = env::var("COPILOT_ACP_BASE_URL")
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
        .unwrap_or_else(|| DEFAULT_COPILOT_ACP_BASE_URL.to_string());
    let command = env::var("HERMES_COPILOT_ACP_COMMAND")
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
        .or_else(|| {
            env::var("COPILOT_CLI_PATH")
                .ok()
                .as_deref()
                .and_then(non_empty_trimmed)
        })
        .unwrap_or_else(|| DEFAULT_COPILOT_ACP_COMMAND.to_string());
    let resolved = resolve_command_path(&command).ok_or_else(|| HermesError::State {
        action: "resolving Copilot ACP runtime",
        detail: format!(
            "Could not find the Copilot CLI command '{command}'. Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
        ),
    })?;
    Ok(CopilotAcpRuntimeCredentials {
        base_url: base_url.trim_end_matches('/').to_string(),
        command: resolved,
    })
}

pub fn resolve_google_gemini_runtime_credentials(
    hermes_home: &Path,
) -> Result<GoogleGeminiRuntimeCredentials, HermesError> {
    resolve_google_gemini_runtime_credentials_with_client_and_refresh_url(
        hermes_home,
        &Client::new(),
        GOOGLE_OAUTH_TOKEN_URL,
    )
}

pub fn resolve_qwen_runtime_credentials() -> Result<QwenRuntimeCredentials, HermesError> {
    let auth_path = qwen_cli_auth_path()?;
    let base_url_override = std::env::var("HERMES_QWEN_BASE_URL")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty());
    resolve_qwen_runtime_credentials_from_path(&auth_path, &Client::new(), base_url_override)
}

fn resolve_codex_access_token_with_refresh_url(
    hermes_home: &Path,
    refresh_url: &str,
) -> Result<String, HermesError> {
    let auth_path = hermes_home.join("auth.json");
    let mut auth_store = load_auth_store(&auth_path)?;
    let provider_state = auth_store
        .get("providers")
        .and_then(Value::as_object)
        .and_then(|providers| providers.get("openai-codex"))
        .and_then(Value::as_object)
        .ok_or_else(|| HermesError::State {
            action: "resolving Codex auth",
            detail: "No Codex credentials stored. Run `hermes auth codex` to authenticate."
                .to_string(),
        })?;
    let tokens = provider_state
        .get("tokens")
        .and_then(Value::as_object)
        .ok_or_else(|| HermesError::State {
            action: "resolving Codex auth",
            detail:
                "Codex auth state is missing tokens. Run `hermes auth codex` to re-authenticate."
                    .to_string(),
        })?;
    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action: "resolving Codex auth",
            detail:
                "Codex auth is missing access_token. Run `hermes auth codex` to re-authenticate."
                    .to_string(),
        })?;
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action: "resolving Codex auth",
            detail:
                "Codex auth is missing refresh_token. Run `hermes auth codex` to re-authenticate."
                    .to_string(),
        })?;

    if !token_needs_refresh(&access_token) {
        return Ok(access_token);
    }

    let refreshed = refresh_codex_tokens(&refresh_token, refresh_url)?;
    persist_codex_tokens(&auth_path, &mut auth_store, &refreshed)?;
    Ok(refreshed.access_token)
}

fn resolve_minimax_oauth_runtime_credentials_with_client(
    hermes_home: &Path,
    client: &Client,
) -> Result<MinimaxOAuthRuntimeCredentials, HermesError> {
    let auth_path = hermes_home.join("auth.json");
    let mut auth_store = load_auth_store(&auth_path)?;
    let provider_state = auth_store
        .get("providers")
        .and_then(Value::as_object)
        .and_then(|providers| providers.get("minimax-oauth"))
        .and_then(Value::as_object)
        .ok_or_else(|| HermesError::State {
            action: "resolving MiniMax OAuth auth",
            detail:
                "No MiniMax OAuth credentials stored. Authenticate with the Python runtime first."
                    .to_string(),
        })?;

    let access_token = required_object_string(
        provider_state,
        "access_token",
        "resolving MiniMax OAuth auth",
    )?;
    let base_url = required_object_string(
        provider_state,
        "inference_base_url",
        "resolving MiniMax OAuth auth",
    )?;
    let expires_at = provider_state
        .get("expires_at")
        .and_then(Value::as_str)
        .and_then(parse_rfc3339_epoch_seconds);

    if !minimax_token_needs_refresh(expires_at) {
        return Ok(MinimaxOAuthRuntimeCredentials {
            access_token,
            base_url: base_url.trim_end_matches('/').to_string(),
        });
    }

    let refreshed = refresh_minimax_oauth_state(client, provider_state).and_then(|state| {
        persist_minimax_oauth_state(&auth_path, &mut auth_store, &state)?;
        Ok(state)
    })?;

    Ok(MinimaxOAuthRuntimeCredentials {
        access_token: refreshed.access_token,
        base_url: refreshed
            .inference_base_url
            .trim_end_matches('/')
            .to_string(),
    })
}

fn load_auth_store(path: &Path) -> Result<Value, HermesError> {
    if !path.exists() {
        return Ok(json!({
            "version": AUTH_STORE_VERSION,
            "providers": {},
        }));
    }
    let raw = fs::read_to_string(path).map_err(|source| HermesError::Io {
        action: "reading",
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str::<Value>(&raw).map_err(|error| HermesError::State {
        action: "parsing auth store",
        detail: format!("{}: {error}", path.display()),
    })
}

fn qwen_cli_auth_path() -> Result<std::path::PathBuf, HermesError> {
    let home = dirs::home_dir().ok_or_else(|| HermesError::State {
        action: "resolving Qwen OAuth auth",
        detail: "Could not determine the home directory for ~/.qwen/oauth_creds.json.".to_string(),
    })?;
    Ok(home.join(".qwen").join("oauth_creds.json"))
}

fn resolve_command_path(command: &str) -> Option<String> {
    let candidate = command.trim();
    if candidate.is_empty() {
        return None;
    }
    let path = std::path::Path::new(candidate);
    if path.components().count() > 1 || path.is_absolute() {
        return path.is_file().then(|| path.to_string_lossy().to_string());
    }
    let path_var = env::var_os("PATH")?;
    for directory in env::split_paths(&path_var) {
        let full = directory.join(candidate);
        if full.is_file() {
            return Some(full.to_string_lossy().to_string());
        }
    }
    None
}

fn google_oauth_path(hermes_home: &Path) -> std::path::PathBuf {
    hermes_home.join("auth").join("google_oauth.json")
}

fn persist_codex_tokens(
    auth_path: &Path,
    auth_store: &mut Value,
    tokens: &CodexTokens,
) -> Result<(), HermesError> {
    let providers = ensure_object_mut(auth_store, &[])?;
    let providers = ensure_object_mut(
        providers
            .entry("providers".to_string())
            .or_insert_with(|| json!({})),
        &["providers"],
    )?;
    let provider_state = ensure_object_mut(
        providers
            .entry("openai-codex".to_string())
            .or_insert_with(|| json!({})),
        &["providers", "openai-codex"],
    )?;
    provider_state.insert(
        "tokens".to_string(),
        json!({
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
        }),
    );
    provider_state.insert(
        "auth_mode".to_string(),
        Value::String("chatgpt".to_string()),
    );
    provider_state.insert(
        "last_refresh".to_string(),
        Value::String(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    if let Some(root) = auth_store.as_object_mut() {
        root.insert("version".to_string(), Value::from(AUTH_STORE_VERSION));
        root.insert(
            "updated_at".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        );
    }
    let payload = serde_json::to_string_pretty(auth_store).map_err(|error| HermesError::State {
        action: "serializing auth store",
        detail: error.to_string(),
    })?;
    fs::write(auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.to_path_buf(),
        source,
    })
}

fn persist_minimax_oauth_state(
    auth_path: &Path,
    auth_store: &mut Value,
    state: &MinimaxOAuthState,
) -> Result<(), HermesError> {
    let providers = ensure_object_mut(auth_store, &[])?;
    let providers = ensure_object_mut(
        providers
            .entry("providers".to_string())
            .or_insert_with(|| json!({})),
        &["providers"],
    )?;
    let provider_state = ensure_object_mut(
        providers
            .entry("minimax-oauth".to_string())
            .or_insert_with(|| json!({})),
        &["providers", "minimax-oauth"],
    )?;
    provider_state.insert(
        "access_token".to_string(),
        Value::String(state.access_token.clone()),
    );
    provider_state.insert(
        "refresh_token".to_string(),
        Value::String(state.refresh_token.clone()),
    );
    provider_state.insert(
        "portal_base_url".to_string(),
        Value::String(state.portal_base_url.clone()),
    );
    provider_state.insert(
        "inference_base_url".to_string(),
        Value::String(state.inference_base_url.clone()),
    );
    provider_state.insert(
        "client_id".to_string(),
        Value::String(state.client_id.clone()),
    );
    provider_state.insert(
        "obtained_at".to_string(),
        Value::String(state.obtained_at.clone()),
    );
    provider_state.insert(
        "expires_at".to_string(),
        Value::String(state.expires_at.clone()),
    );
    provider_state.insert("expires_in".to_string(), Value::from(state.expires_in));
    if let Some(token_type) = &state.token_type {
        provider_state.insert("token_type".to_string(), Value::String(token_type.clone()));
    }
    if let Some(resource_url) = &state.resource_url {
        provider_state.insert(
            "resource_url".to_string(),
            Value::String(resource_url.clone()),
        );
    }
    if let Some(root) = auth_store.as_object_mut() {
        root.insert("version".to_string(), Value::from(AUTH_STORE_VERSION));
        root.insert(
            "updated_at".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        );
    }
    let payload = serde_json::to_string_pretty(auth_store).map_err(|error| HermesError::State {
        action: "serializing auth store",
        detail: error.to_string(),
    })?;
    fs::write(auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.to_path_buf(),
        source,
    })
}

fn persist_qwen_tokens(auth_path: &Path, tokens: &Value) -> Result<(), HermesError> {
    let payload = serde_json::to_string_pretty(tokens).map_err(|error| HermesError::State {
        action: "serializing Qwen OAuth credentials",
        detail: error.to_string(),
    })?;
    if let Some(parent) = auth_path.parent() {
        fs::create_dir_all(parent).map_err(|source| HermesError::Io {
            action: "creating",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.to_path_buf(),
        source,
    })
}

fn persist_google_state(auth_path: &Path, state: &GoogleOAuthState) -> Result<(), HermesError> {
    let payload =
        serde_json::to_string_pretty(&state.to_json()).map_err(|error| HermesError::State {
            action: "serializing Google OAuth credentials",
            detail: error.to_string(),
        })?;
    if let Some(parent) = auth_path.parent() {
        fs::create_dir_all(parent).map_err(|source| HermesError::Io {
            action: "creating",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(auth_path, format!("{payload}\n")).map_err(|source| HermesError::Io {
        action: "writing",
        path: auth_path.to_path_buf(),
        source,
    })
}

fn ensure_object_mut<'a>(
    value: &'a mut Value,
    path: &[&str],
) -> Result<&'a mut serde_json::Map<String, Value>, HermesError> {
    if !value.is_object() {
        *value = json!({});
    }
    value.as_object_mut().ok_or_else(|| HermesError::State {
        action: "updating auth store",
        detail: format!("Path {} is not an object.", path.join(".")),
    })
}

fn refresh_codex_tokens(
    refresh_token: &str,
    refresh_url: &str,
) -> Result<CodexTokens, HermesError> {
    let client = Client::builder()
        .build()
        .map_err(|error| HermesError::State {
            action: "building Codex auth client",
            detail: error.to_string(),
        })?;
    let response = client
        .post(refresh_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CODEX_OAUTH_CLIENT_ID),
        ])
        .send()
        .map_err(|error| HermesError::State {
            action: "refreshing Codex auth",
            detail: error.to_string(),
        })?;
    let status = response.status();
    let body = response.text().map_err(|error| HermesError::State {
        action: "reading Codex refresh response",
        detail: error.to_string(),
    })?;
    if !status.is_success() {
        return Err(HermesError::State {
            action: "refreshing Codex auth",
            detail: format!(
                "Codex token refresh failed with status {}. Run `hermes auth codex` to re-authenticate.",
                status.as_u16()
            ),
        });
    }
    let payload = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Codex refresh response",
        detail: format!("{error}: {body}"),
    })?;
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action: "refreshing Codex auth",
            detail: "Codex token refresh response was missing access_token.".to_string(),
        })?;
    let next_refresh = payload
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| refresh_token.to_string());
    Ok(CodexTokens {
        access_token,
        refresh_token: next_refresh,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MinimaxOAuthState {
    access_token: String,
    refresh_token: String,
    portal_base_url: String,
    inference_base_url: String,
    client_id: String,
    obtained_at: String,
    expires_at: String,
    expires_in: i64,
    token_type: Option<String>,
    resource_url: Option<String>,
}

fn refresh_minimax_oauth_state(
    client: &Client,
    provider_state: &serde_json::Map<String, Value>,
) -> Result<MinimaxOAuthState, HermesError> {
    let refresh_token = required_object_string(
        provider_state,
        "refresh_token",
        "refreshing MiniMax OAuth auth",
    )?;
    let portal_base_url = required_object_string(
        provider_state,
        "portal_base_url",
        "refreshing MiniMax OAuth auth",
    )?;
    let inference_base_url = required_object_string(
        provider_state,
        "inference_base_url",
        "refreshing MiniMax OAuth auth",
    )?;
    let client_id =
        required_object_string(provider_state, "client_id", "refreshing MiniMax OAuth auth")?;
    let token_type = provider_state
        .get("token_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let resource_url = provider_state
        .get("resource_url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    let refresh_url = format!("{}/oauth/token", portal_base_url.trim_end_matches('/'));
    let response = client
        .post(&refresh_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", client_id.as_str()),
            ("refresh_token", refresh_token.as_str()),
        ])
        .send()
        .map_err(|error| HermesError::State {
            action: "refreshing MiniMax OAuth auth",
            detail: error.to_string(),
        })?;
    let status = response.status();
    let body = response.text().map_err(|error| HermesError::State {
        action: "reading MiniMax OAuth refresh response",
        detail: error.to_string(),
    })?;
    if !status.is_success() {
        return Err(HermesError::State {
            action: "refreshing MiniMax OAuth auth",
            detail: format!(
                "MiniMax OAuth refresh failed with status {}. Re-authenticate in the Python runtime.",
                status.as_u16()
            ),
        });
    }
    let payload = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding MiniMax OAuth refresh response",
        detail: format!("{error}: {body}"),
    })?;
    if payload.get("status").and_then(Value::as_str) != Some("success") {
        return Err(HermesError::State {
            action: "refreshing MiniMax OAuth auth",
            detail: "MiniMax OAuth refresh did not return status=success.".to_string(),
        });
    }
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action: "refreshing MiniMax OAuth auth",
            detail: "MiniMax OAuth refresh response was missing access_token.".to_string(),
        })?;
    let expires_in = payload
        .get("expired_in")
        .and_then(Value::as_i64)
        .or_else(|| {
            payload
                .get("expired_in")
                .and_then(Value::as_u64)
                .map(|value| value as i64)
        })
        .filter(|value| *value > 0)
        .ok_or_else(|| HermesError::State {
            action: "refreshing MiniMax OAuth auth",
            detail: "MiniMax OAuth refresh response was missing expired_in.".to_string(),
        })?;
    let obtained_at = Utc::now();
    let expires_at = (obtained_at + chrono::TimeDelta::seconds(expires_in))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    Ok(MinimaxOAuthState {
        access_token,
        refresh_token: payload
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or(refresh_token),
        portal_base_url: portal_base_url.trim_end_matches('/').to_string(),
        inference_base_url: inference_base_url.trim_end_matches('/').to_string(),
        client_id,
        obtained_at: obtained_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        expires_at,
        expires_in,
        token_type,
        resource_url,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GoogleRefreshParts {
    refresh_token: String,
    project_id: String,
    managed_project_id: String,
}

impl GoogleRefreshParts {
    fn parse(packed: &str) -> Self {
        if packed.trim().is_empty() {
            return Self {
                refresh_token: String::new(),
                project_id: String::new(),
                managed_project_id: String::new(),
            };
        }
        let mut parts = packed.splitn(3, '|');
        Self {
            refresh_token: parts.next().unwrap_or_default().trim().to_string(),
            project_id: parts.next().unwrap_or_default().trim().to_string(),
            managed_project_id: parts.next().unwrap_or_default().trim().to_string(),
        }
    }

    fn format(&self) -> String {
        if self.refresh_token.trim().is_empty() {
            return String::new();
        }
        if self.project_id.trim().is_empty() && self.managed_project_id.trim().is_empty() {
            return self.refresh_token.clone();
        }
        format!(
            "{}|{}|{}",
            self.refresh_token, self.project_id, self.managed_project_id
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GoogleOAuthState {
    access_token: String,
    refresh_token: String,
    expires_ms: i64,
    email: String,
    project_id: String,
    managed_project_id: String,
}

impl GoogleOAuthState {
    fn from_json(value: &Value) -> Result<Self, HermesError> {
        let object = value.as_object().ok_or_else(|| HermesError::State {
            action: "parsing Google OAuth credentials",
            detail: "google_oauth.json does not contain a JSON object.".to_string(),
        })?;
        let refresh = GoogleRefreshParts::parse(
            object
                .get("refresh")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        Ok(Self {
            access_token: object
                .get("access")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string(),
            refresh_token: refresh.refresh_token,
            expires_ms: object
                .get("expires")
                .and_then(value_to_i64)
                .unwrap_or_default(),
            email: object
                .get("email")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string(),
            project_id: refresh.project_id,
            managed_project_id: refresh.managed_project_id,
        })
    }

    fn to_json(&self) -> Value {
        json!({
            "refresh": GoogleRefreshParts {
                refresh_token: self.refresh_token.clone(),
                project_id: self.project_id.clone(),
                managed_project_id: self.managed_project_id.clone(),
            }
            .format(),
            "access": self.access_token,
            "expires": self.expires_ms,
            "email": self.email,
        })
    }
}

fn resolve_google_gemini_runtime_credentials_with_client_and_refresh_url(
    hermes_home: &Path,
    client: &Client,
    refresh_url: &str,
) -> Result<GoogleGeminiRuntimeCredentials, HermesError> {
    let auth_path = google_oauth_path(hermes_home);
    let mut state = load_google_state(&auth_path)?;
    if google_access_token_needs_refresh(&state) {
        state = refresh_google_state(client, &auth_path, &state, refresh_url)?;
    }
    if state.access_token.trim().is_empty() {
        return Err(HermesError::State {
            action: "resolving Google Gemini OAuth auth",
            detail: format!(
                "{} is missing an access token. Run `hermes auth add google-gemini-cli` in the Python runtime first.",
                auth_path.display()
            ),
        });
    }
    Ok(GoogleGeminiRuntimeCredentials {
        access_token: state.access_token,
        refresh_token: state.refresh_token,
        project_id: state.project_id,
        managed_project_id: state.managed_project_id,
        email: state.email,
    })
}

pub(crate) fn persist_google_gemini_project_ids(
    hermes_home: &Path,
    project_id: &str,
    managed_project_id: &str,
) -> Result<(), HermesError> {
    let auth_path = google_oauth_path(hermes_home);
    let mut state = load_google_state(&auth_path)?;
    let project_id = project_id.trim();
    let managed_project_id = managed_project_id.trim();
    if project_id.is_empty() && managed_project_id.is_empty() {
        return Ok(());
    }
    if !project_id.is_empty() {
        state.project_id = project_id.to_string();
    }
    if !managed_project_id.is_empty() {
        state.managed_project_id = managed_project_id.to_string();
    }
    persist_google_state(&auth_path, &state)
}

fn load_google_state(auth_path: &Path) -> Result<GoogleOAuthState, HermesError> {
    if !auth_path.exists() {
        return Err(HermesError::State {
            action: "resolving Google Gemini OAuth auth",
            detail: format!(
                "Google OAuth credentials not found at {}. Run `hermes auth add google-gemini-cli` in the Python runtime first.",
                auth_path.display()
            ),
        });
    }
    let raw = fs::read_to_string(auth_path).map_err(|source| HermesError::Io {
        action: "reading",
        path: auth_path.to_path_buf(),
        source,
    })?;
    let parsed = serde_json::from_str::<Value>(&raw).map_err(|error| HermesError::State {
        action: "parsing Google OAuth credentials",
        detail: format!("{}: {error}", auth_path.display()),
    })?;
    GoogleOAuthState::from_json(&parsed)
}

fn google_access_token_needs_refresh(state: &GoogleOAuthState) -> bool {
    if state.access_token.trim().is_empty() || state.expires_ms <= 0 {
        return true;
    }
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0);
    now_ms >= state.expires_ms - (GOOGLE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS * 1000)
}

fn refresh_google_state(
    client: &Client,
    auth_path: &Path,
    state: &GoogleOAuthState,
    refresh_url: &str,
) -> Result<GoogleOAuthState, HermesError> {
    if state.refresh_token.trim().is_empty() {
        return Err(HermesError::State {
            action: "refreshing Google Gemini OAuth auth",
            detail:
                "Google OAuth refresh token missing. Run `hermes auth add google-gemini-cli` in the Python runtime again."
                    .to_string(),
        });
    }
    let client_id = std::env::var(GOOGLE_OAUTH_CLIENT_ID_ENV)
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
        .unwrap_or_else(|| GOOGLE_DEFAULT_CLIENT_ID.to_string());
    let client_secret = std::env::var(GOOGLE_OAUTH_CLIENT_SECRET_ENV)
        .ok()
        .as_deref()
        .and_then(non_empty_trimmed)
        .unwrap_or_else(|| GOOGLE_DEFAULT_CLIENT_SECRET.to_string());

    let mut form = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", state.refresh_token.as_str()),
        ("client_id", client_id.as_str()),
    ];
    if !client_secret.trim().is_empty() {
        form.push(("client_secret", client_secret.as_str()));
    }

    let response = client
        .post(refresh_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .map_err(|error| HermesError::State {
            action: "refreshing Google Gemini OAuth auth",
            detail: error.to_string(),
        })?;
    let status = response.status();
    let body = response.text().map_err(|error| HermesError::State {
        action: "reading Google Gemini OAuth refresh response",
        detail: error.to_string(),
    })?;
    if !status.is_success() {
        let detail = body.trim();
        return Err(HermesError::State {
            action: "refreshing Google Gemini OAuth auth",
            detail: if detail.is_empty() {
                "Google OAuth refresh failed. Re-run `hermes auth add google-gemini-cli` in the Python runtime."
                    .to_string()
            } else {
                format!(
                    "Google OAuth refresh failed. Re-run `hermes auth add google-gemini-cli` in the Python runtime. Response: {detail}"
                )
            },
        });
    }
    let payload = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Google Gemini OAuth refresh response",
        detail: format!("{error}: {body}"),
    })?;
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| HermesError::State {
            action: "refreshing Google Gemini OAuth auth",
            detail: "Google OAuth refresh response was missing access_token.".to_string(),
        })?;
    let expires_in_seconds = payload
        .get("expires_in")
        .and_then(value_to_i64)
        .filter(|value| *value > 0)
        .unwrap_or(3600);
    let refreshed = GoogleOAuthState {
        access_token: access_token.to_string(),
        refresh_token: payload
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&state.refresh_token)
            .to_string(),
        expires_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0)
            + expires_in_seconds * 1000,
        email: state.email.clone(),
        project_id: state.project_id.clone(),
        managed_project_id: state.managed_project_id.clone(),
    };
    persist_google_state(auth_path, &refreshed)?;
    Ok(refreshed)
}

fn resolve_qwen_runtime_credentials_from_path(
    auth_path: &Path,
    client: &Client,
    base_url_override: Option<String>,
) -> Result<QwenRuntimeCredentials, HermesError> {
    resolve_qwen_runtime_credentials_from_path_with_refresh_url(
        auth_path,
        client,
        base_url_override,
        QWEN_OAUTH_TOKEN_URL,
    )
}

fn resolve_qwen_runtime_credentials_from_path_with_refresh_url(
    auth_path: &Path,
    client: &Client,
    base_url_override: Option<String>,
    refresh_url: &str,
) -> Result<QwenRuntimeCredentials, HermesError> {
    let mut tokens = load_qwen_tokens(auth_path)?;
    let should_refresh = qwen_access_token_needs_refresh(tokens.get("expiry_date"));
    if should_refresh {
        tokens = refresh_qwen_tokens(client, auth_path, &tokens, refresh_url)?;
    }
    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action: "resolving Qwen OAuth auth",
            detail: "Qwen OAuth access_token missing. Re-authenticate with the Python runtime."
                .to_string(),
        })?;
    let base_url = base_url_override.unwrap_or_else(|| DEFAULT_QWEN_BASE_URL.to_string());
    Ok(QwenRuntimeCredentials {
        access_token,
        base_url: base_url.trim_end_matches('/').to_string(),
    })
}

fn load_qwen_tokens(auth_path: &Path) -> Result<Value, HermesError> {
    if !auth_path.exists() {
        return Err(HermesError::State {
            action: "resolving Qwen OAuth auth",
            detail: format!(
                "Qwen CLI credentials not found at {}. Run the Python runtime's Qwen auth flow first.",
                auth_path.display()
            ),
        });
    }
    let raw = fs::read_to_string(auth_path).map_err(|source| HermesError::Io {
        action: "reading",
        path: auth_path.to_path_buf(),
        source,
    })?;
    let parsed = serde_json::from_str::<Value>(&raw).map_err(|error| HermesError::State {
        action: "parsing Qwen OAuth credentials",
        detail: format!("{}: {error}", auth_path.display()),
    })?;
    if !parsed.is_object() {
        return Err(HermesError::State {
            action: "parsing Qwen OAuth credentials",
            detail: format!("{} does not contain a JSON object.", auth_path.display()),
        });
    }
    Ok(parsed)
}

fn qwen_access_token_needs_refresh(expiry_date_ms: Option<&Value>) -> bool {
    let expiry_ms = expiry_date_ms
        .and_then(value_to_i64)
        .filter(|value| *value > 0);
    let Some(expiry_ms) = expiry_ms else {
        return true;
    };
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0);
    now_ms >= expiry_ms - (QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS * 1000)
}

fn refresh_qwen_tokens(
    client: &Client,
    auth_path: &Path,
    tokens: &Value,
    refresh_url: &str,
) -> Result<Value, HermesError> {
    let object = tokens.as_object().ok_or_else(|| HermesError::State {
        action: "refreshing Qwen OAuth auth",
        detail: "Qwen OAuth credentials are not a JSON object.".to_string(),
    })?;
    let refresh_token = object
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| HermesError::State {
            action: "refreshing Qwen OAuth auth",
            detail: "Qwen OAuth refresh_token missing. Re-authenticate with the Python runtime."
                .to_string(),
        })?;

    let response = client
        .post(refresh_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", QWEN_OAUTH_CLIENT_ID),
        ])
        .send()
        .map_err(|error| HermesError::State {
            action: "refreshing Qwen OAuth auth",
            detail: error.to_string(),
        })?;
    let status = response.status();
    let body = response.text().map_err(|error| HermesError::State {
        action: "reading Qwen OAuth refresh response",
        detail: error.to_string(),
    })?;
    if !status.is_success() {
        let detail = body.trim();
        return Err(HermesError::State {
            action: "refreshing Qwen OAuth auth",
            detail: if detail.is_empty() {
                "Qwen OAuth refresh failed. Re-authenticate with the Python runtime.".to_string()
            } else {
                format!(
                    "Qwen OAuth refresh failed. Re-authenticate with the Python runtime. Response: {detail}"
                )
            },
        });
    }
    let payload = serde_json::from_str::<Value>(&body).map_err(|error| HermesError::State {
        action: "decoding Qwen OAuth refresh response",
        detail: format!("{error}: {body}"),
    })?;
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| HermesError::State {
            action: "refreshing Qwen OAuth auth",
            detail: "Qwen OAuth refresh response missing access_token.".to_string(),
        })?;
    let expires_in_seconds = payload
        .get("expires_in")
        .and_then(value_to_i64)
        .filter(|value| *value > 0)
        .unwrap_or(6 * 60 * 60);
    let expiry_date = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
        + expires_in_seconds * 1000;

    let refreshed = json!({
        "access_token": access_token,
        "refresh_token": payload
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(refresh_token),
        "token_type": payload
            .get("token_type")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| object.get("token_type").and_then(Value::as_str))
            .unwrap_or("Bearer"),
        "resource_url": payload
            .get("resource_url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| object.get("resource_url").and_then(Value::as_str))
            .unwrap_or("portal.qwen.ai"),
        "expiry_date": expiry_date,
    });
    persist_qwen_tokens(auth_path, &refreshed)?;
    Ok(refreshed)
}

fn token_needs_refresh(access_token: &str) -> bool {
    let Some(exp) = token_expiry(access_token) else {
        return false;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    now >= exp - CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
}

fn minimax_token_needs_refresh(expires_at: Option<i64>) -> bool {
    let Some(expires_at) = expires_at else {
        return true;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    now >= expires_at - MINIMAX_OAUTH_REFRESH_SKEW_SECONDS
}

fn value_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().map(|raw| raw as i64))
        .or_else(|| {
            value
                .as_str()
                .and_then(|raw| raw.trim().parse::<i64>().ok())
        })
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn token_expiry(access_token: &str) -> Option<i64> {
    decode_jwt_claims(access_token)
        .and_then(|claims| claims.get("exp").and_then(Value::as_i64))
        .filter(|value| *value > 0)
}

fn chatgpt_account_id_from_token(access_token: &str) -> Option<String> {
    decode_jwt_claims(access_token).and_then(|claims| {
        claims
            .get("https://api.openai.com/auth")
            .and_then(Value::as_object)
            .and_then(|auth| auth.get("chatgpt_account_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn decode_jwt_claims(access_token: &str) -> Option<Value> {
    let payload = access_token.split('.').nth(1)?;
    let padded = format!("{payload}{}", "=".repeat((4 - payload.len() % 4) % 4));
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(padded)
        .ok()?;
    serde_json::from_slice::<Value>(&decoded).ok()
}

fn parse_rfc3339_epoch_seconds(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|parsed| parsed.timestamp())
}

fn required_object_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    action: &'static str,
) -> Result<String, HermesError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| HermesError::State {
            action,
            detail: format!(
                "MiniMax OAuth state is missing {key}. Re-authenticate in the Python runtime."
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
    fn codex_headers_include_originator_and_account_id() {
        let token = jwt_with_claims(i64::MAX / 2, "acct-123");
        let headers = codex_cloudflare_headers(&token);
        assert!(
            headers
                .iter()
                .any(|(name, value)| { name == "originator" && value == "codex_cli_rs" })
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| { name == "ChatGPT-Account-ID" && value == "acct-123" })
        );
    }

    #[test]
    fn resolve_codex_access_token_reads_fresh_token_from_auth_store() {
        let temp = TempDir::new().unwrap();
        let token = jwt_with_claims(i64::MAX / 2, "acct-fresh");
        fs::write(
            temp.path().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": token,
                            "refresh_token": "refresh-1",
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let resolved = resolve_codex_access_token(temp.path()).unwrap();
        assert!(resolved.contains("."));
    }

    #[test]
    fn resolve_codex_access_token_refreshes_expired_token() {
        let temp = TempDir::new().unwrap();
        let expired = jwt_with_claims(1, "acct-old");
        fs::write(
            temp.path().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": expired,
                            "refresh_token": "refresh-old",
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let fresh = jwt_with_claims(i64::MAX / 2, "acct-new");
        let fresh_for_server = fresh.clone();
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
            assert!(request_text.starts_with("POST /token "));
            assert!(request_text.contains("grant_type=refresh_token"));
            assert!(request_text.contains("refresh_token=refresh-old"));

            let body = json!({
                "access_token": fresh_for_server,
                "refresh_token": "refresh-new",
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let resolved = resolve_codex_access_token_with_refresh_url(
            temp.path(),
            &format!("http://{addr}/token"),
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(resolved, fresh);
        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(temp.path().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            persisted["providers"]["openai-codex"]["tokens"]["refresh_token"],
            "refresh-new"
        );
    }

    #[test]
    fn resolve_minimax_oauth_runtime_credentials_reads_fresh_state() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "minimax-oauth": {
                        "access_token": "mini-fresh",
                        "refresh_token": "refresh-fresh",
                        "portal_base_url": "https://api.minimax.io",
                        "inference_base_url": "https://api.minimax.io/anthropic",
                        "client_id": "client-mini",
                        "expires_at": "2999-01-01T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let resolved = resolve_minimax_oauth_runtime_credentials(temp.path()).unwrap();
        assert_eq!(resolved.access_token, "mini-fresh");
        assert_eq!(resolved.base_url, "https://api.minimax.io/anthropic");
    }

    #[test]
    fn resolve_copilot_acp_runtime_credentials_reads_env_command_and_base_url() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let previous_command = env::var_os("HERMES_COPILOT_ACP_COMMAND");
        let previous_base = env::var_os("COPILOT_ACP_BASE_URL");

        let temp = TempDir::new().unwrap();
        let fake = temp.path().join("fake-copilot");
        fs::write(&fake, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&fake).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&fake, permissions).unwrap();
        }

        unsafe {
            env::set_var("HERMES_COPILOT_ACP_COMMAND", &fake);
            env::set_var("COPILOT_ACP_BASE_URL", "acp://copilot");
        }

        let resolved = resolve_copilot_acp_runtime_credentials().unwrap();

        match previous_command {
            Some(value) => unsafe { env::set_var("HERMES_COPILOT_ACP_COMMAND", value) },
            None => unsafe { env::remove_var("HERMES_COPILOT_ACP_COMMAND") },
        }
        match previous_base {
            Some(value) => unsafe { env::set_var("COPILOT_ACP_BASE_URL", value) },
            None => unsafe { env::remove_var("COPILOT_ACP_BASE_URL") },
        }

        assert_eq!(resolved.base_url, "acp://copilot");
        assert_eq!(resolved.command, fake.to_string_lossy());
    }

    #[test]
    fn resolve_google_gemini_runtime_credentials_reads_fresh_state() {
        let temp = TempDir::new().unwrap();
        let auth_dir = temp.path().join("auth");
        fs::create_dir_all(&auth_dir).unwrap();
        fs::write(
            auth_dir.join("google_oauth.json"),
            json!({
                "refresh": "google-refresh|proj-123|managed-456",
                "access": "google-fresh",
                "expires": i64::MAX / 2,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();

        let resolved = resolve_google_gemini_runtime_credentials(temp.path()).unwrap();
        assert_eq!(resolved.access_token, "google-fresh");
        assert_eq!(resolved.refresh_token, "google-refresh");
        assert_eq!(resolved.project_id, "proj-123");
        assert_eq!(resolved.managed_project_id, "managed-456");
        assert_eq!(resolved.email, "dev@example.com");
    }

    #[test]
    fn resolve_google_gemini_runtime_credentials_refreshes_expired_state() {
        let temp = TempDir::new().unwrap();
        let auth_dir = temp.path().join("auth");
        fs::create_dir_all(&auth_dir).unwrap();
        fs::write(
            auth_dir.join("google_oauth.json"),
            json!({
                "refresh": "google-refresh-old|proj-old|managed-old",
                "access": "google-old",
                "expires": 1,
                "email": "dev@example.com"
            })
            .to_string(),
        )
        .unwrap();

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
            assert!(request_text.starts_with("POST /token "));
            assert!(request_text.contains("grant_type=refresh_token"));
            assert!(request_text.contains("refresh_token=google-refresh-old"));
            assert!(request_text.contains(&format!("client_id={GOOGLE_DEFAULT_CLIENT_ID}")));
            assert!(request_text.contains(&format!("client_secret={GOOGLE_DEFAULT_CLIENT_SECRET}")));

            let body = json!({
                "access_token": "google-new",
                "refresh_token": "google-refresh-new",
                "expires_in": 7200,
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let client = Client::builder().build().unwrap();
        let resolved = resolve_google_gemini_runtime_credentials_with_client_and_refresh_url(
            temp.path(),
            &client,
            &format!("http://{addr}/token"),
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(resolved.access_token, "google-new");
        assert_eq!(resolved.refresh_token, "google-refresh-new");
        assert_eq!(resolved.project_id, "proj-old");
        assert_eq!(resolved.managed_project_id, "managed-old");
        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(auth_dir.join("google_oauth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(persisted["access"], "google-new");
        assert_eq!(
            persisted["refresh"],
            "google-refresh-new|proj-old|managed-old"
        );
    }

    #[test]
    fn resolve_minimax_oauth_runtime_credentials_refreshes_expired_state() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        fs::write(
            temp.path().join("auth.json"),
            json!({
                "version": 1,
                "providers": {
                    "minimax-oauth": {
                        "access_token": "mini-old",
                        "refresh_token": "refresh-old",
                        "portal_base_url": format!("http://{addr}"),
                        "inference_base_url": "https://api.minimaxi.com/anthropic",
                        "client_id": "client-mini",
                        "expires_at": "1970-01-01T00:00:00Z"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

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
            assert!(request_text.starts_with("POST /oauth/token "));
            assert!(request_text.contains("grant_type=refresh_token"));
            assert!(request_text.contains("refresh_token=refresh-old"));
            assert!(request_text.contains("client_id=client-mini"));

            let body = json!({
                "status": "success",
                "access_token": "mini-new",
                "refresh_token": "refresh-new",
                "expired_in": 3600,
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let resolved = resolve_minimax_oauth_runtime_credentials(temp.path()).unwrap();
        server.join().unwrap();

        assert_eq!(resolved.access_token, "mini-new");
        assert_eq!(resolved.base_url, "https://api.minimaxi.com/anthropic");
        let persisted = serde_json::from_str::<Value>(
            &fs::read_to_string(temp.path().join("auth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            persisted["providers"]["minimax-oauth"]["refresh_token"],
            "refresh-new"
        );
        assert_eq!(
            persisted["providers"]["minimax-oauth"]["access_token"],
            "mini-new"
        );
    }

    #[test]
    fn resolve_qwen_runtime_credentials_reads_fresh_state() {
        let temp = TempDir::new().unwrap();
        let auth_path = temp.path().join("oauth_creds.json");
        fs::write(
            &auth_path,
            json!({
                "access_token": "qwen-fresh",
                "refresh_token": "qwen-refresh",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": i64::MAX / 2,
            })
            .to_string(),
        )
        .unwrap();

        let resolved = resolve_qwen_runtime_credentials_from_path(
            &auth_path,
            &Client::new(),
            Some("https://portal.qwen.ai/v1".to_string()),
        )
        .unwrap();

        assert_eq!(resolved.access_token, "qwen-fresh");
        assert_eq!(resolved.base_url, "https://portal.qwen.ai/v1");
    }

    #[test]
    fn resolve_qwen_runtime_credentials_refreshes_expired_state() {
        let temp = TempDir::new().unwrap();
        let auth_path = temp.path().join("oauth_creds.json");
        fs::write(
            &auth_path,
            json!({
                "access_token": "qwen-old",
                "refresh_token": "qwen-refresh-old",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": 1,
            })
            .to_string(),
        )
        .unwrap();

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
            assert!(request_text.starts_with("POST /oauth2/token "));
            assert!(request_text.contains("grant_type=refresh_token"));
            assert!(request_text.contains("refresh_token=qwen-refresh-old"));
            assert!(request_text.contains(&format!("client_id={QWEN_OAUTH_CLIENT_ID}")));

            let body = json!({
                "access_token": "qwen-new",
                "refresh_token": "qwen-refresh-new",
                "expires_in": 7200,
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let client = Client::builder().build().unwrap();
        let resolved = resolve_qwen_runtime_credentials_from_path_with_refresh_url(
            &auth_path,
            &client,
            Some("http://127.0.0.1:18000/v1".to_string()),
            &format!("http://{addr}/oauth2/token"),
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(resolved.access_token, "qwen-new");
        let persisted =
            serde_json::from_str::<Value>(&fs::read_to_string(&auth_path).unwrap()).unwrap();
        assert_eq!(persisted["access_token"], "qwen-new");
        assert_eq!(persisted["refresh_token"], "qwen-refresh-new");
    }
}
