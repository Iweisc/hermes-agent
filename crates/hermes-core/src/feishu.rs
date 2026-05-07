use std::collections::HashMap;
use std::env;
use std::fmt::{self, Display, Formatter};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use reqwest::Method;
use reqwest::Url;
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_FEISHU_OPEN_BASE_URL: &str = "https://open.feishu.cn";
const DEFAULT_LARK_OPEN_BASE_URL: &str = "https://open.larksuite.com";
const REQUEST_TIMEOUT_SECS: u64 = 15;
const TOKEN_CACHE_SKEW_SECS: u64 = 60;
const DEFAULT_PAGE_SIZE: i64 = 100;
const MAX_PAGE_SIZE: i64 = 100;

#[derive(Debug, Clone)]
struct FeishuConfig {
    app_id: String,
    app_secret: String,
    base_url: String,
}

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    expires_at: Instant,
}

#[derive(Debug, Clone)]
struct FeishuApiError {
    status: u16,
    body: String,
}

impl Display for FeishuApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Feishu API error {}: {}", self.status, self.body)
    }
}

#[derive(Debug, Clone)]
enum FeishuError {
    Api(FeishuApiError),
    Message(String),
}

impl Display for FeishuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api(error) => Display::fmt(error, f),
            Self::Message(message) => f.write_str(message),
        }
    }
}

static TOKEN_CACHE: OnceLock<Mutex<HashMap<String, CachedToken>>> = OnceLock::new();

pub fn feishu_available() -> bool {
    resolve_config_from_env().is_ok()
}

pub fn feishu_doc_read_schema() -> Value {
    json!({
        "name": "feishu_doc_read",
        "description": "Read the full content of a Feishu or Lark document as plain text.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_token": {
                    "type": "string",
                    "description": "Document token from the Feishu or Lark document URL."
                }
            },
            "required": ["doc_token"]
        }
    })
}

pub fn feishu_drive_list_comments_schema() -> Value {
    json!({
        "name": "feishu_drive_list_comments",
        "description": "List comments on a Feishu or Lark document.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "Drive file token for the document."
                },
                "file_type": {
                    "type": "string",
                    "description": "Drive file type such as docx.",
                    "default": "docx"
                },
                "is_whole": {
                    "type": "boolean",
                    "description": "When true, list whole-document comments only.",
                    "default": false
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of comments per page. Must be between 1 and 100.",
                    "default": 100
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token returned by a prior list call."
                }
            },
            "required": ["file_token"]
        }
    })
}

pub fn feishu_drive_list_comment_replies_schema() -> Value {
    json!({
        "name": "feishu_drive_list_comment_replies",
        "description": "List replies in a Feishu or Lark document comment thread.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "Drive file token for the document."
                },
                "comment_id": {
                    "type": "string",
                    "description": "Comment thread identifier."
                },
                "file_type": {
                    "type": "string",
                    "description": "Drive file type such as docx.",
                    "default": "docx"
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of replies per page. Must be between 1 and 100.",
                    "default": 100
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token returned by a prior list call."
                }
            },
            "required": ["file_token", "comment_id"]
        }
    })
}

pub fn feishu_drive_reply_comment_schema() -> Value {
    json!({
        "name": "feishu_drive_reply_comment",
        "description": "Reply to a Feishu or Lark document comment thread with plain text.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "Drive file token for the document."
                },
                "comment_id": {
                    "type": "string",
                    "description": "Comment thread identifier."
                },
                "content": {
                    "type": "string",
                    "description": "Reply text. Plain text only."
                },
                "file_type": {
                    "type": "string",
                    "description": "Drive file type such as docx.",
                    "default": "docx"
                }
            },
            "required": ["file_token", "comment_id", "content"]
        }
    })
}

pub fn feishu_drive_add_comment_schema() -> Value {
    json!({
        "name": "feishu_drive_add_comment",
        "description": "Add a whole-document Feishu or Lark comment with plain text content.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "Drive file token for the document."
                },
                "content": {
                    "type": "string",
                    "description": "Comment text. Plain text only."
                },
                "file_type": {
                    "type": "string",
                    "description": "Drive file type such as docx.",
                    "default": "docx"
                }
            },
            "required": ["file_token", "content"]
        }
    })
}

pub fn handle_feishu_doc_read(args: &Value, _runtime: &ToolRuntime) -> String {
    let doc_token = match required_identifier(args, "doc_token") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let config = match resolve_config_from_env() {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match read_document_content(&config, &doc_token) {
        Ok(content) => tool_result(json!({ "success": true, "content": content })),
        Err(error) => tool_error(format!("Failed to read document: {error}")),
    }
}

pub fn handle_feishu_drive_list_comments(args: &Value, _runtime: &ToolRuntime) -> String {
    let file_token = match required_identifier(args, "file_token") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let file_type = match optional_file_type(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let page_size = match optional_page_size(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let page_token = match optional_page_token(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let config = match resolve_config_from_env() {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match list_comments(
        &config,
        &file_token,
        &file_type,
        args.get("is_whole")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        page_size,
        page_token.as_deref(),
    ) {
        Ok(data) => tool_result(data),
        Err(error) => tool_error(format!("List comments failed: {error}")),
    }
}

pub fn handle_feishu_drive_list_comment_replies(args: &Value, _runtime: &ToolRuntime) -> String {
    let file_token = match required_identifier(args, "file_token") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let comment_id = match required_identifier(args, "comment_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let file_type = match optional_file_type(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let page_size = match optional_page_size(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let page_token = match optional_page_token(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let config = match resolve_config_from_env() {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match list_comment_replies(
        &config,
        &file_token,
        &comment_id,
        &file_type,
        page_size,
        page_token.as_deref(),
    ) {
        Ok(data) => tool_result(data),
        Err(error) => tool_error(format!("List replies failed: {error}")),
    }
}

pub fn handle_feishu_drive_reply_comment(args: &Value, _runtime: &ToolRuntime) -> String {
    let file_token = match required_identifier(args, "file_token") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let comment_id = match required_identifier(args, "comment_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let content = match required_content(args, "content") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let file_type = match optional_file_type(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let config = match resolve_config_from_env() {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match reply_comment(&config, &file_token, &comment_id, &file_type, &content) {
        Ok(data) => tool_result(json!({ "success": true, "data": data })),
        Err(error) => tool_error(format!("Reply comment failed: {error}")),
    }
}

pub fn handle_feishu_drive_add_comment(args: &Value, _runtime: &ToolRuntime) -> String {
    let file_token = match required_identifier(args, "file_token") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let content = match required_content(args, "content") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let file_type = match optional_file_type(args) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let config = match resolve_config_from_env() {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match add_comment(&config, &file_token, &file_type, &content) {
        Ok(data) => tool_result(json!({ "success": true, "data": data })),
        Err(error) => tool_error(format!("Add comment failed: {error}")),
    }
}

fn read_document_content(config: &FeishuConfig, doc_token: &str) -> Result<String, FeishuError> {
    let path = format!("/open-apis/docx/v1/documents/{doc_token}/raw_content");
    let body = api_request(config, Method::GET, &path, &[], None)?;
    body.pointer("/data/content")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| FeishuError::Message("Feishu document API returned no content".to_string()))
}

fn list_comments(
    config: &FeishuConfig,
    file_token: &str,
    file_type: &str,
    is_whole: bool,
    page_size: i64,
    page_token: Option<&str>,
) -> Result<Value, FeishuError> {
    let path = format!("/open-apis/drive/v1/files/{file_token}/comments");
    let mut query = vec![
        ("file_type".to_string(), file_type.to_string()),
        ("user_id_type".to_string(), "open_id".to_string()),
        ("page_size".to_string(), page_size.to_string()),
    ];
    if is_whole {
        query.push(("is_whole".to_string(), "true".to_string()));
    }
    if let Some(page_token) = page_token {
        query.push(("page_token".to_string(), page_token.to_string()));
    }
    response_data(api_request(config, Method::GET, &path, &query, None)?)
}

fn list_comment_replies(
    config: &FeishuConfig,
    file_token: &str,
    comment_id: &str,
    file_type: &str,
    page_size: i64,
    page_token: Option<&str>,
) -> Result<Value, FeishuError> {
    let path = format!("/open-apis/drive/v1/files/{file_token}/comments/{comment_id}/replies");
    let mut query = vec![
        ("file_type".to_string(), file_type.to_string()),
        ("user_id_type".to_string(), "open_id".to_string()),
        ("page_size".to_string(), page_size.to_string()),
    ];
    if let Some(page_token) = page_token {
        query.push(("page_token".to_string(), page_token.to_string()));
    }
    response_data(api_request(config, Method::GET, &path, &query, None)?)
}

fn reply_comment(
    config: &FeishuConfig,
    file_token: &str,
    comment_id: &str,
    file_type: &str,
    content: &str,
) -> Result<Value, FeishuError> {
    let path = format!("/open-apis/drive/v1/files/{file_token}/comments/{comment_id}/replies");
    let query = vec![("file_type".to_string(), file_type.to_string())];
    let body = json!({
        "content": {
            "elements": [
                {
                    "type": "text_run",
                    "text_run": {
                        "text": content,
                    }
                }
            ]
        }
    });
    response_data(api_request(
        config,
        Method::POST,
        &path,
        &query,
        Some(body),
    )?)
}

fn add_comment(
    config: &FeishuConfig,
    file_token: &str,
    file_type: &str,
    content: &str,
) -> Result<Value, FeishuError> {
    let path = format!("/open-apis/drive/v1/files/{file_token}/new_comments");
    let body = json!({
        "file_type": file_type,
        "reply_elements": [
            {
                "type": "text",
                "text": content,
            }
        ]
    });
    response_data(api_request(config, Method::POST, &path, &[], Some(body))?)
}

fn response_data(body: Value) -> Result<Value, FeishuError> {
    body.get("data")
        .cloned()
        .ok_or_else(|| FeishuError::Message("Feishu API returned no data payload".to_string()))
}

fn api_request(
    config: &FeishuConfig,
    method: Method,
    path: &str,
    query: &[(String, String)],
    body: Option<Value>,
) -> Result<Value, FeishuError> {
    send_api_request(config, method.clone(), path, query, body.clone(), true)
}

fn send_api_request(
    config: &FeishuConfig,
    method: Method,
    path: &str,
    query: &[(String, String)],
    body: Option<Value>,
    allow_retry: bool,
) -> Result<Value, FeishuError> {
    let token = tenant_access_token(config, allow_retry)?;
    let client = http_client()?;
    let mut url = Url::parse(&format!("{}{}", config.base_url, path))
        .map_err(|error| FeishuError::Message(format!("invalid Feishu URL: {error}")))?;
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (key, value) in query {
            pairs.append_pair(key, value);
        }
    }
    let mut request = client
        .request(method.clone(), url)
        .bearer_auth(token)
        .header("Content-Type", "application/json");
    if let Some(body) = body.as_ref() {
        request = request.json(body);
    }

    let response = request
        .send()
        .map_err(|error| FeishuError::Message(format!("request failed: {error}")))?;
    let status = response.status();
    let raw = response
        .text()
        .map_err(|error| FeishuError::Message(format!("failed to read response body: {error}")))?;

    if status.as_u16() == 401 && allow_retry {
        clear_cached_token(config);
        return send_api_request(config, method, path, query, body, false);
    }
    if !status.is_success() {
        return Err(FeishuError::Api(FeishuApiError {
            status: status.as_u16(),
            body: raw,
        }));
    }

    let payload: Value = serde_json::from_str(&raw)
        .map_err(|error| FeishuError::Message(format!("invalid JSON response: {error}")))?;
    let code = payload.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        let message = payload
            .get("msg")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown error");
        return Err(FeishuError::Message(format!("code={code} msg={message}")));
    }
    Ok(payload)
}

fn tenant_access_token(config: &FeishuConfig, _allow_retry: bool) -> Result<String, FeishuError> {
    let cache_key = token_cache_key(config);
    if let Some(token) = cached_token(&cache_key) {
        return Ok(token);
    }

    let client = http_client()?;
    let url = format!(
        "{}/open-apis/auth/v3/tenant_access_token/internal",
        config.base_url
    );
    let response = client
        .post(&url)
        .json(&json!({
            "app_id": config.app_id,
            "app_secret": config.app_secret,
        }))
        .send()
        .map_err(|error| FeishuError::Message(format!("token request failed: {error}")))?;
    let status = response.status();
    let raw = response
        .text()
        .map_err(|error| FeishuError::Message(format!("failed to read token response: {error}")))?;
    if !status.is_success() {
        return Err(FeishuError::Api(FeishuApiError {
            status: status.as_u16(),
            body: raw,
        }));
    }

    let payload: Value = serde_json::from_str(&raw)
        .map_err(|error| FeishuError::Message(format!("invalid token JSON response: {error}")))?;
    let code = payload.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        let message = payload
            .get("msg")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown error");
        return Err(FeishuError::Message(format!(
            "tenant access token failed: code={code} msg={message}"
        )));
    }

    let token = payload
        .get("tenant_access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            FeishuError::Message("tenant access token response did not contain a token".to_string())
        })?;
    let ttl_seconds = payload
        .get("expire")
        .and_then(value_as_i64)
        .or_else(|| payload.get("expires_in").and_then(value_as_i64))
        .unwrap_or(7200)
        .max(0) as u64;
    store_cached_token(&cache_key, &token, ttl_seconds);
    Ok(token)
}

fn resolve_config_from_env() -> Result<FeishuConfig, String> {
    let app_id = env::var("FEISHU_APP_ID")
        .map_err(|_| "FEISHU_APP_ID is not set".to_string())?
        .trim()
        .to_string();
    let app_secret = env::var("FEISHU_APP_SECRET")
        .map_err(|_| "FEISHU_APP_SECRET is not set".to_string())?
        .trim()
        .to_string();
    if app_id.is_empty() {
        return Err("FEISHU_APP_ID must not be empty".to_string());
    }
    if app_secret.is_empty() {
        return Err("FEISHU_APP_SECRET must not be empty".to_string());
    }

    let domain = env::var("FEISHU_DOMAIN").unwrap_or_else(|_| "feishu".to_string());
    let base_url = base_url_for_domain(&domain)?;
    Ok(FeishuConfig {
        app_id,
        app_secret,
        base_url,
    })
}

fn base_url_for_domain(domain: &str) -> Result<String, String> {
    let trimmed = domain.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("feishu") {
        return Ok(DEFAULT_FEISHU_OPEN_BASE_URL.to_string());
    }
    if trimmed.eq_ignore_ascii_case("lark") {
        return Ok(DEFAULT_LARK_OPEN_BASE_URL.to_string());
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let url =
            Url::parse(trimmed).map_err(|error| format!("invalid FEISHU_DOMAIN URL: {error}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err("FEISHU_DOMAIN URL must include http or https and a host".to_string());
        }
        let mut normalized = url.to_string();
        while normalized.ends_with('/') {
            normalized.pop();
        }
        return Ok(normalized);
    }
    Err(format!(
        "FEISHU_DOMAIN must be 'feishu', 'lark', or a full http(s) base URL, got {trimmed:?}"
    ))
}

fn http_client() -> Result<Client, FeishuError> {
    Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .map_err(|error| FeishuError::Message(format!("failed to build HTTP client: {error}")))
}

fn token_cache() -> &'static Mutex<HashMap<String, CachedToken>> {
    TOKEN_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn token_cache_key(config: &FeishuConfig) -> String {
    format!(
        "{}|{}|{}",
        config.base_url, config.app_id, config.app_secret
    )
}

fn cached_token(cache_key: &str) -> Option<String> {
    let cache = token_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let entry = cache.get(cache_key)?;
    if Instant::now() >= entry.expires_at {
        return None;
    }
    Some(entry.token.clone())
}

fn clear_cached_token(config: &FeishuConfig) {
    let mut cache = token_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache.remove(&token_cache_key(config));
}

fn store_cached_token(cache_key: &str, token: &str, ttl_seconds: u64) {
    let ttl = ttl_seconds.saturating_sub(TOKEN_CACHE_SKEW_SECS).max(1);
    let entry = CachedToken {
        token: token.to_string(),
        expires_at: Instant::now() + Duration::from_secs(ttl),
    };
    let mut cache = token_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache.insert(cache_key.to_string(), entry);
}

fn required_identifier(args: &Value, key: &str) -> Result<String, String> {
    let value = args
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} is required"))?;
    validate_identifier(key, value)?;
    Ok(value.to_string())
}

fn optional_page_token(args: &Value) -> Result<Option<String>, String> {
    let Some(value) = args.get("page_token").and_then(Value::as_str) else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.len() > 512 || trimmed.chars().any(|ch| ch.is_control()) {
        return Err("page_token contains unsupported characters".to_string());
    }
    Ok(Some(trimmed.to_string()))
}

fn validate_identifier(key: &str, value: &str) -> Result<(), String> {
    if value.len() > 256 {
        return Err(format!("{key} is too long"));
    }
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.'))
    {
        return Ok(());
    }
    Err(format!("{key} contains unsupported characters"))
}

fn required_content(args: &Value, key: &str) -> Result<String, String> {
    let value = args
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} is required"))?;
    Ok(value.to_string())
}

fn optional_file_type(args: &Value) -> Result<String, String> {
    let raw = args
        .get("file_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("docx");
    if raw.len() > 64
        || !raw
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        return Err(format!("Invalid file_type format: {raw:?}"));
    }
    Ok(raw.to_string())
}

fn optional_page_size(args: &Value) -> Result<i64, String> {
    let page_size = args
        .get("page_size")
        .and_then(value_as_i64)
        .unwrap_or(DEFAULT_PAGE_SIZE);
    if !(1..=MAX_PAGE_SIZE).contains(&page_size) {
        return Err(format!(
            "page_size must be between 1 and {MAX_PAGE_SIZE}, got {page_size}"
        ));
    }
    Ok(page_size)
}

fn value_as_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_env_lock() -> &'static Mutex<()> {
        TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
        test_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn with_env_var(key: &str, value: Option<&str>) {
        match value {
            Some(value) => unsafe { env::set_var(key, value) },
            None => unsafe { env::remove_var(key) },
        }
    }

    fn clear_token_cache_for_tests() {
        token_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    fn mock_server<F>(request_count: usize, handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(usize, String, String) -> (u16, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            for index in 0..request_count {
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
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|value| value + 4)
                    .unwrap_or(request.len());
                let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let mut body_bytes = request[header_end..].to_vec();
                while body_bytes.len() < content_length {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    body_bytes.extend_from_slice(&buffer[..read]);
                }
                let body = String::from_utf8_lossy(&body_bytes[..content_length]).to_string();
                let (status, response_body) = handler(index, headers, body);
                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });
        (format!("http://{}", addr), join)
    }

    fn config_for(base_url: &str) -> FeishuConfig {
        FeishuConfig {
            app_id: "cli_test".to_string(),
            app_secret: "secret_test".to_string(),
            base_url: base_url.to_string(),
        }
    }

    #[test]
    fn availability_requires_app_credentials() {
        let _guard = acquire_test_lock();
        clear_token_cache_for_tests();
        with_env_var("FEISHU_APP_ID", None);
        with_env_var("FEISHU_APP_SECRET", None);
        with_env_var("FEISHU_DOMAIN", None);
        assert!(!feishu_available());

        with_env_var("FEISHU_APP_ID", Some("cli_test"));
        with_env_var("FEISHU_APP_SECRET", Some("secret_test"));
        assert!(feishu_available());
    }

    #[test]
    fn doc_read_returns_content() {
        clear_token_cache_for_tests();
        let (base_url, join) = mock_server(2, |index, headers, body| match index {
            0 => {
                assert!(
                    headers.starts_with("POST /open-apis/auth/v3/tenant_access_token/internal")
                );
                assert!(body.contains("\"app_id\":\"cli_test\""));
                (
                    200,
                    json!({
                        "code": 0,
                        "tenant_access_token": "tenant-token",
                        "expire": 7200
                    })
                    .to_string(),
                )
            }
            1 => {
                assert!(
                    headers.starts_with("GET /open-apis/docx/v1/documents/doc123/raw_content ")
                );
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer tenant-token")
                );
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "content": "Document body"
                        }
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });

        let content = read_document_content(&config_for(&base_url), "doc123").unwrap();
        assert_eq!(content, "Document body");
        join.join().unwrap();
    }

    #[test]
    fn list_calls_reuse_cached_token() {
        clear_token_cache_for_tests();
        let (base_url, join) = mock_server(3, |index, headers, _body| match index {
            0 => (
                200,
                json!({
                    "code": 0,
                    "tenant_access_token": "tenant-token",
                    "expire": 7200
                })
                .to_string(),
            ),
            1 => {
                assert!(headers.starts_with("GET /open-apis/drive/v1/files/file123/comments?"));
                assert!(headers.contains("page_size=25"));
                assert!(headers.contains("is_whole=true"));
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "items": [{"comment_id": "c1"}],
                            "has_more": false
                        }
                    })
                    .to_string(),
                )
            }
            2 => {
                assert!(headers.starts_with(
                    "GET /open-apis/drive/v1/files/file123/comments/comment1/replies?"
                ));
                assert!(headers.contains("page_size=10"));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer tenant-token")
                );
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "items": [{"reply_id": "r1"}]
                        }
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });

        let config = config_for(&base_url);
        let comments = list_comments(&config, "file123", "docx", true, 25, None).unwrap();
        assert_eq!(comments["items"][0]["comment_id"], json!("c1"));
        let replies =
            list_comment_replies(&config, "file123", "comment1", "docx", 10, None).unwrap();
        assert_eq!(replies["items"][0]["reply_id"], json!("r1"));
        join.join().unwrap();
    }

    #[test]
    fn reply_comment_posts_expected_body() {
        clear_token_cache_for_tests();
        let (base_url, join) = mock_server(2, |index, headers, body| match index {
            0 => (
                200,
                json!({
                    "code": 0,
                    "tenant_access_token": "tenant-token",
                    "expire": 7200
                })
                .to_string(),
            ),
            1 => {
                assert!(headers.starts_with("POST /open-apis/drive/v1/files/file123/comments/comment1/replies?file_type=docx "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(
                    payload["content"]["elements"][0]["text_run"]["text"],
                    json!("Thanks")
                );
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "reply_id": "reply-1"
                        }
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });

        let result = reply_comment(
            &config_for(&base_url),
            "file123",
            "comment1",
            "docx",
            "Thanks",
        )
        .unwrap();
        assert_eq!(result["reply_id"], json!("reply-1"));
        join.join().unwrap();
    }

    #[test]
    fn add_comment_posts_expected_body() {
        clear_token_cache_for_tests();
        let (base_url, join) = mock_server(2, |index, headers, body| match index {
            0 => (
                200,
                json!({
                    "code": 0,
                    "tenant_access_token": "tenant-token",
                    "expire": 7200
                })
                .to_string(),
            ),
            1 => {
                assert!(
                    headers.starts_with("POST /open-apis/drive/v1/files/file123/new_comments ")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["file_type"], json!("docx"));
                assert_eq!(payload["reply_elements"][0]["text"], json!("Whole comment"));
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "comment_id": "comment-1"
                        }
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });

        let result =
            add_comment(&config_for(&base_url), "file123", "docx", "Whole comment").unwrap();
        assert_eq!(result["comment_id"], json!("comment-1"));
        join.join().unwrap();
    }

    #[test]
    fn rejects_invalid_page_size() {
        let error = optional_page_size(&json!({ "page_size": 101 })).unwrap_err();
        assert!(error.contains("page_size"));
    }

    #[test]
    fn lark_domain_uses_larksuite_base() {
        assert_eq!(
            base_url_for_domain("lark").unwrap(),
            DEFAULT_LARK_OPEN_BASE_URL
        );
    }

    #[test]
    fn custom_base_url_is_accepted() {
        assert_eq!(
            base_url_for_domain("http://127.0.0.1:9000/").unwrap(),
            "http://127.0.0.1:9000"
        );
    }
}
