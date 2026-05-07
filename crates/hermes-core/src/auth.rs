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
const MINIMAX_OAUTH_REFRESH_SKEW_SECONDS: i64 = 60;

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
    use tempfile::TempDir;

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
}
