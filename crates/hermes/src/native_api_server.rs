use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hermes_core::{
    DelegateExecutor, HermesContext, LoadedConfig, ModelOverrides, ToolRuntime,
    get_tool_definitions,
};
use serde::Deserialize;
use serde_json::{Value, json};
use serde_yaml::{Mapping, Value as YamlValue};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

use crate::gateway_cmd::GatewayRunArgs;

const DEFAULT_API_SERVER_HOST: &str = "127.0.0.1";
const DEFAULT_API_SERVER_PORT: u16 = 8642;

#[derive(Clone, Debug)]
pub(crate) struct NativeApiServerSettings {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) api_key: String,
    pub(crate) cors_origins: Vec<String>,
    pub(crate) model_name: String,
}

#[derive(Clone)]
pub(crate) struct NativeApiServerState {
    context: HermesContext,
    loaded: LoadedConfig,
    pub(crate) settings: NativeApiServerSettings,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Value>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    input: Value,
    #[serde(default)]
    stream: Option<bool>,
}

pub(crate) fn maybe_run_native_api_server(
    context: &HermesContext,
    args: &GatewayRunArgs,
) -> Result<bool, Box<dyn Error>> {
    let loaded = context.load_config_document()?;
    let Some(settings) = load_native_api_server_settings(context, &loaded)? else {
        return Ok(false);
    };
    if has_other_gateway_platforms_enabled(&loaded) {
        return Ok(false);
    }
    run_native_api_server(context.clone(), loaded, settings, args)?;
    Ok(true)
}

pub(crate) fn load_native_api_server_state(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<Option<NativeApiServerState>, Box<dyn Error>> {
    let Some(settings) = load_native_api_server_settings(context, loaded)? else {
        return Ok(None);
    };
    Ok(Some(NativeApiServerState {
        context: context.clone(),
        loaded: loaded.clone(),
        settings,
    }))
}

fn run_native_api_server(
    context: HermesContext,
    loaded: LoadedConfig,
    settings: NativeApiServerSettings,
    _args: &GatewayRunArgs,
) -> Result<(), Box<dyn Error>> {
    let bind_ip: IpAddr = settings
        .host
        .parse()
        .map_err(|_| "API_SERVER_HOST must be a valid IP address for native Rust runtime")?;
    if is_network_accessible(bind_ip) && settings.api_key.trim().is_empty() {
        return Err(
            "Refusing to start native API server on a non-loopback address without API_SERVER_KEY"
                .into(),
        );
    }
    let state = NativeApiServerState {
        context,
        loaded,
        settings: settings.clone(),
    };
    let runtime = Runtime::new()?;
    runtime.block_on(async move {
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::new(bind_ip, settings.port)).await?;
        println!(
            "Native API server listening on http://{}:{} (model: {})",
            settings.host, settings.port, settings.model_name
        );
        serve_native_api_server(listener, state, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
    })?;
    Ok(())
}

pub(crate) async fn serve_native_api_server<F>(
    listener: tokio::net::TcpListener,
    state: NativeApiServerState,
    shutdown: F,
) -> Result<(), Box<dyn Error>>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/v1/health", get(handle_health))
        .route("/v1/models", get(handle_models).options(handle_options))
        .route(
            "/v1/chat/completions",
            post(handle_chat_completions).options(handle_options),
        )
        .route(
            "/v1/responses",
            post(handle_responses).options(handle_options),
        )
        .with_state(Arc::new(state));
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn handle_health(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
) -> Response {
    json_response(
        StatusCode::OK,
        json!({
            "status": "ok",
            "platform": "api_server",
            "model": state.settings.model_name,
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_models(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    json_response(
        StatusCode::OK,
        json!({
            "object": "list",
            "data": [{
                "id": state.settings.model_name,
                "object": "model",
                "created": unix_ts_secs(),
                "owned_by": "hermes-agent",
            }]
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_chat_completions(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionsRequest>,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    if request.stream.unwrap_or(false) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({ "error": { "message": "stream=true is not supported in the native Rust API server yet." } }),
            &state.settings,
            headers.get(ORIGIN),
        );
    }
    if request.messages.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({ "error": { "message": "messages must not be empty" } }),
            &state.settings,
            headers.get(ORIGIN),
        );
    }
    match run_agent_for_messages_async(Arc::clone(&state), request.model, request.messages).await {
        Ok(result) => {
            let response_id = format!("chatcmpl-rs-{:x}", unix_ts_nanos());
            json_response(
                StatusCode::OK,
                json!({
                    "id": response_id,
                    "object": "chat.completion",
                    "created": unix_ts_secs(),
                    "model": state.settings.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.final_response,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                }),
                &state.settings,
                headers.get(ORIGIN),
            )
        }
        Err(error) => json_response(
            StatusCode::BAD_GATEWAY,
            json!({ "error": { "message": error.to_string() } }),
            &state.settings,
            headers.get(ORIGIN),
        ),
    }
}

async fn handle_responses(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    if request.stream.unwrap_or(false) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({ "error": { "message": "stream=true is not supported in the native Rust API server yet." } }),
            &state.settings,
            headers.get(ORIGIN),
        );
    }
    let messages = match normalize_responses_input(&request.input) {
        Ok(messages) => messages,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({ "error": { "message": error } }),
                &state.settings,
                headers.get(ORIGIN),
            );
        }
    };
    match run_agent_for_messages_async(Arc::clone(&state), request.model, messages).await {
        Ok(result) => {
            let response_id = format!("resp-rs-{:x}", unix_ts_nanos());
            json_response(
                StatusCode::OK,
                json!({
                    "id": response_id,
                    "object": "response",
                    "created_at": unix_ts_secs(),
                    "status": "completed",
                    "model": state.settings.model_name,
                    "output": [{
                        "id": format!("msg-rs-{:x}", unix_ts_nanos()),
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": result.final_response,
                            "annotations": [],
                        }],
                    }],
                }),
                &state.settings,
                headers.get(ORIGIN),
            )
        }
        Err(error) => json_response(
            StatusCode::BAD_GATEWAY,
            json!({ "error": { "message": error.to_string() } }),
            &state.settings,
            headers.get(ORIGIN),
        ),
    }
}

async fn run_agent_for_messages_async(
    state: Arc<NativeApiServerState>,
    model: Option<String>,
    messages: Vec<Value>,
) -> Result<hermes_core::AgentTurnResult, String> {
    tokio::task::spawn_blocking(move || {
        run_agent_for_messages(&state, model, &messages).map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

async fn handle_options(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    apply_cors_headers(response.headers_mut(), &state.settings, headers.get(ORIGIN));
    response.headers_mut().insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET,POST,OPTIONS"),
    );
    response.headers_mut().insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization,content-type"),
    );
    response
}

fn run_agent_for_messages(
    state: &NativeApiServerState,
    model: Option<String>,
    messages: &[Value],
) -> Result<hermes_core::AgentTurnResult, Box<dyn Error>> {
    let enabled_toolsets = state.loaded.config.toolsets.clone();
    let delegate = DelegateExecutor::new(
        state.context.clone(),
        state.loaded.clone(),
        "rust-api-server",
        enabled_toolsets.clone(),
        ModelOverrides::default(),
        api_server_workdir(&state.loaded),
    );
    let tool_names = get_tool_definitions(
        Some(&enabled_toolsets),
        disabled_memory_toolsets(&state.loaded).as_deref(),
    )
    .into_iter()
    .map(|tool| tool.name)
    .collect::<Vec<_>>();
    let mut runtime = ToolRuntime::default()
        .with_hermes_home(state.context.hermes_home())
        .with_available_tool_names(tool_names)
        .with_delegate_callback(move |request| delegate.execute(request));
    let _ = runtime.load_memory_store(&state.loaded.config.memory);
    let overrides = ModelOverrides {
        model,
        ..ModelOverrides::default()
    };
    Ok(state.context.run_chat_turn_with_messages(
        &state.loaded,
        messages,
        &runtime,
        Some(&enabled_toolsets),
        &overrides,
    )?)
}

fn normalize_responses_input(input: &Value) -> Result<Vec<Value>, String> {
    match input {
        Value::String(text) => Ok(vec![json!({
            "role": "user",
            "content": text,
        })]),
        Value::Array(items) => Ok(items.clone()),
        Value::Object(object) => Ok(vec![Value::Object(object.clone())]),
        _ => Err("responses input must be a string, object, or array".to_string()),
    }
}

fn json_response(
    status: StatusCode,
    body: Value,
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
) -> Response {
    let mut response = (status, Json(body)).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    apply_cors_headers(response.headers_mut(), settings, origin);
    response
}

fn apply_cors_headers(
    headers: &mut HeaderMap,
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
) {
    let Some(origin) = origin.and_then(|value| value.to_str().ok()) else {
        return;
    };
    if settings.cors_origins.is_empty() {
        return;
    }
    let allow_any = settings.cors_origins.iter().any(|value| value == "*");
    let allowed = allow_any || settings.cors_origins.iter().any(|value| value == origin);
    if !allowed {
        return;
    }
    let allow_value = if allow_any { "*" } else { origin };
    if let Ok(value) = HeaderValue::from_str(allow_value) {
        headers.insert(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    }
}

fn ensure_authorized(
    settings: &NativeApiServerSettings,
    headers: &HeaderMap,
) -> Result<(), Response> {
    if settings.api_key.trim().is_empty() {
        return Ok(());
    }
    let Some(header) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(unauthorized_response(settings, headers.get(ORIGIN)));
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err(unauthorized_response(settings, headers.get(ORIGIN)));
    };
    if token.trim() != settings.api_key {
        return Err(unauthorized_response(settings, headers.get(ORIGIN)));
    }
    Ok(())
}

fn unauthorized_response(
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
) -> Response {
    json_response(
        StatusCode::UNAUTHORIZED,
        json!({ "error": { "message": "Unauthorized" } }),
        settings,
        origin,
    )
}

fn disabled_memory_toolsets(loaded: &LoadedConfig) -> Option<Vec<String>> {
    (!loaded.config.memory.any_enabled()).then(|| vec![String::from("memory")])
}

fn api_server_workdir(loaded: &LoadedConfig) -> PathBuf {
    let raw = loaded.config.terminal.cwd.trim();
    if raw.is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(raw)
    }
}

pub(crate) fn load_native_api_server_settings(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<Option<NativeApiServerSettings>, Box<dyn Error>> {
    if !api_server_enabled(loaded) {
        return Ok(None);
    }
    let host = env_string("API_SERVER_HOST")
        .or_else(|| {
            platform_api_server_extra(loaded).and_then(|extra| mapping_string(extra, "host"))
        })
        .unwrap_or_else(|| DEFAULT_API_SERVER_HOST.to_string());
    let port = env_string("API_SERVER_PORT")
        .and_then(|raw| raw.parse::<u16>().ok())
        .or_else(|| platform_api_server_extra(loaded).and_then(|extra| mapping_u16(extra, "port")))
        .unwrap_or(DEFAULT_API_SERVER_PORT);
    if port == 0 {
        return Err("API_SERVER_PORT must be between 1 and 65535".into());
    }
    let api_key = env_string("API_SERVER_KEY")
        .or_else(|| {
            platform_api_server_extra(loaded).and_then(|extra| mapping_string(extra, "key"))
        })
        .unwrap_or_default();
    let cors_origins = env_string("API_SERVER_CORS_ORIGINS")
        .map(|raw| split_csv(&raw))
        .or_else(|| {
            platform_api_server_extra(loaded).and_then(|extra| mapping_csv(extra, "cors_origins"))
        })
        .unwrap_or_default();
    let model_name = env_string("API_SERVER_MODEL_NAME")
        .or_else(|| {
            platform_api_server_extra(loaded).and_then(|extra| mapping_string(extra, "model_name"))
        })
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_model_name(context));
    Ok(Some(NativeApiServerSettings {
        host,
        port,
        api_key,
        cors_origins,
        model_name,
    }))
}

fn api_server_enabled(loaded: &LoadedConfig) -> bool {
    env_truthy("API_SERVER_ENABLED")
        || platform_api_server_mapping(loaded)
            .and_then(|mapping| mapping_bool(mapping, "enabled"))
            .unwrap_or(false)
        || env_string("API_SERVER_KEY").is_some()
}

fn has_other_gateway_platforms_enabled(loaded: &LoadedConfig) -> bool {
    let env_enabled = [
        ("TELEGRAM_BOT_TOKEN", false),
        ("DISCORD_BOT_TOKEN", false),
        ("SLACK_BOT_TOKEN", false),
        ("FEISHU_APP_ID", false),
        ("MATRIX_ACCESS_TOKEN", false),
        ("WHATSAPP_ENABLED", true),
        ("DINGTALK_CLIENT_ID", false),
        ("QQ_APP_ID", false),
        ("MATTERMOST_TOKEN", false),
        ("WECOM_BOT_ID", false),
        ("WEIXIN_TOKEN", false),
        ("EMAIL_ADDRESS", false),
        ("TWILIO_ACCOUNT_SID", false),
        ("HASS_TOKEN", false),
        ("BLUEBUBBLES_SERVER_URL", false),
        ("SIGNAL_HTTP_URL", false),
        ("YUANBAO_APP_ID", false),
        ("WEBHOOK_ENABLED", true),
    ]
    .iter()
    .any(|(key, truthy)| {
        if *truthy {
            env_truthy(key)
        } else {
            env_string(key).is_some()
        }
    });
    if env_enabled {
        return true;
    }
    loaded
        .cfg_get(&["platforms"])
        .and_then(YamlValue::as_mapping)
        .is_some_and(|platforms| {
            platforms.iter().any(|(key, value)| {
                let Some(name) = key.as_str() else {
                    return false;
                };
                if name == "api_server" {
                    return false;
                }
                value
                    .as_mapping()
                    .and_then(|mapping| mapping_bool(mapping, "enabled"))
                    .unwrap_or(false)
            })
        })
}

fn default_model_name(context: &HermesContext) -> String {
    let profile = context.current_profile_name();
    if profile == "default" {
        "hermes-agent".to_string()
    } else {
        profile
    }
}

fn env_truthy(key: &str) -> bool {
    env_string(key).is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn platform_api_server_mapping<'a>(loaded: &'a LoadedConfig) -> Option<&'a Mapping> {
    loaded
        .cfg_get(&["platforms", "api_server"])
        .and_then(YamlValue::as_mapping)
}

fn platform_api_server_extra<'a>(loaded: &'a LoadedConfig) -> Option<&'a Mapping> {
    platform_api_server_mapping(loaded)
        .and_then(|mapping| mapping.get(YamlValue::String("extra".to_string())))
        .and_then(YamlValue::as_mapping)
}

fn mapping_bool(mapping: &Mapping, key: &str) -> Option<bool> {
    match mapping.get(YamlValue::String(key.to_string()))? {
        YamlValue::Bool(value) => Some(*value),
        YamlValue::Number(value) => value.as_i64().map(|number| number != 0),
        YamlValue::String(value) => Some(matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )),
        _ => None,
    }
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn mapping_u16(mapping: &Mapping, key: &str) -> Option<u16> {
    match mapping.get(YamlValue::String(key.to_string()))? {
        YamlValue::Number(value) => value.as_u64().and_then(|number| u16::try_from(number).ok()),
        YamlValue::String(value) => value.trim().parse::<u16>().ok(),
        _ => None,
    }
}

fn mapping_csv(mapping: &Mapping, key: &str) -> Option<Vec<String>> {
    let value = mapping.get(YamlValue::String(key.to_string()))?;
    match value {
        YamlValue::Sequence(values) => {
            let out = values
                .iter()
                .filter_map(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            (!out.is_empty()).then_some(out)
        }
        YamlValue::String(value) => {
            let out = split_csv(value);
            (!out.is_empty()).then_some(out)
        }
        _ => None,
    }
}

fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn is_network_accessible(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(value) => value != Ipv4Addr::LOCALHOST,
        IpAddr::V6(value) => value != Ipv6Addr::LOCALHOST,
    }
}

fn unix_ts_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use tempfile::TempDir;

    fn temp_context(config_text: &str) -> (TempDir, HermesContext, LoadedConfig) {
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("config.yaml"), config_text).unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().to_path_buf()));
        let loaded = context.load_config_document().unwrap();
        (temp, context, loaded)
    }

    fn mock_model_server(response_body: String) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
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
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        (format!("http://{}", addr), join)
    }

    #[test]
    fn loads_native_api_server_settings_from_env_and_raw_config() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (_temp, context, loaded) = temp_context(
            "platforms:\n  api_server:\n    enabled: true\n    extra:\n      host: 0.0.0.0\n      port: 9010\n      cors_origins:\n        - https://chat.example.com\n",
        );
        let settings = load_native_api_server_settings(&context, &loaded)
            .unwrap()
            .unwrap();
        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 9010);
        assert_eq!(settings.cors_origins, vec!["https://chat.example.com"]);
        assert_eq!(settings.model_name, "hermes-agent");
    }

    #[test]
    fn native_api_server_chat_completions_returns_agent_output() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "native hello"
                }
            }]
        })
        .to_string();
        let (base_url, join) = mock_model_server(response_body);
        let config_text = format!(
            "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: {base_url}\n  api_key: test-key\nplatforms:\n  api_server:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 0\n"
        );
        let (_temp, context, loaded) = temp_context(&config_text);
        let settings = NativeApiServerSettings {
            host: "127.0.0.1".to_string(),
            port: 0,
            api_key: String::new(),
            cors_origins: Vec::new(),
            model_name: "hermes-agent".to_string(),
        };
        let state = NativeApiServerState {
            context,
            loaded,
            settings: settings.clone(),
        };
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server_thread = thread::spawn(move || {
            let runtime = Runtime::new().unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
                serve_native_api_server(listener, state, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
            });
        });

        let client = reqwest::blocking::Client::new();
        for _ in 0..20 {
            if client.get(format!("http://{addr}/health")).send().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let response: Value = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": "say hi"
                }]
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(
            response["choices"][0]["message"]["content"],
            json!("native hello")
        );

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }
}
