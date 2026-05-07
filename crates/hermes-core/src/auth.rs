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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CodexTokens {
    access_token: String,
    refresh_token: String,
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
}
