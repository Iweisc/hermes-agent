use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, State};
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
const MAX_STORED_RESPONSES: usize = 100;

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
    response_store: Arc<Mutex<ResponseStore>>,
    run_store: Arc<Mutex<RunStore>>,
}

#[derive(Default)]
struct ResponseStore {
    order: VecDeque<String>,
    responses: BTreeMap<String, StoredResponse>,
}

#[derive(Clone)]
struct StoredResponse {
    response: Value,
    conversation_history: Vec<Value>,
}

#[derive(Default)]
struct RunStore {
    runs: BTreeMap<String, Value>,
}

enum ResponsesMessagesError {
    BadRequest(String),
    NotFound(String),
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
    previous_response_id: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RunsRequest {
    #[serde(default)]
    model: Option<String>,
    input: Value,
    #[serde(default)]
    previous_response_id: Option<String>,
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
        response_store: Arc::new(Mutex::new(ResponseStore::default())),
        run_store: Arc::new(Mutex::new(RunStore::default())),
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
        response_store: Arc::new(Mutex::new(ResponseStore::default())),
        run_store: Arc::new(Mutex::new(RunStore::default())),
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
        .route(
            "/v1/capabilities",
            get(handle_capabilities).options(handle_options),
        )
        .route("/v1/models", get(handle_models).options(handle_options))
        .route(
            "/v1/chat/completions",
            post(handle_chat_completions).options(handle_options),
        )
        .route(
            "/v1/responses",
            post(handle_responses).options(handle_options),
        )
        .route(
            "/v1/responses/{response_id}",
            get(handle_get_response)
                .delete(handle_delete_response)
                .options(handle_options),
        )
        .route("/v1/runs", post(handle_runs).options(handle_options))
        .route(
            "/v1/runs/{run_id}",
            get(handle_get_run).options(handle_options),
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

async fn handle_capabilities(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    json_response(
        StatusCode::OK,
        json!({
            "object": "hermes.api_server.capabilities",
            "model": state.settings.model_name,
            "features": {
                "chat_completions": true,
                "chat_completions_streaming": true,
                "responses_api": true,
                "responses_streaming": true,
                "stored_responses": true,
                "runs_api": true,
                "run_events": false,
                "run_stop": false,
            },
            "endpoints": {
                "models": { "method": "GET", "path": "/v1/models" },
                "capabilities": { "method": "GET", "path": "/v1/capabilities" },
                "chat_completions": { "method": "POST", "path": "/v1/chat/completions" },
                "responses": { "method": "POST", "path": "/v1/responses" },
                "response_get": { "method": "GET", "path": "/v1/responses/{response_id}" },
                "response_delete": { "method": "DELETE", "path": "/v1/responses/{response_id}" },
                "runs": { "method": "POST", "path": "/v1/runs" },
                "run_status": { "method": "GET", "path": "/v1/runs/{run_id}" },
            }
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
        return match run_agent_for_messages_async(
            Arc::clone(&state),
            request.model,
            request.messages,
        )
        .await
        {
            Ok(result) => chat_completions_sse_response(
                &state.settings,
                headers.get(ORIGIN),
                &result.final_response,
            ),
            Err(error) => json_response(
                StatusCode::BAD_GATEWAY,
                json!({ "error": { "message": error.to_string() } }),
                &state.settings,
                headers.get(ORIGIN),
            ),
        };
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
        let messages = match build_responses_messages(
            &state,
            &request.input,
            request.previous_response_id.as_deref(),
        ) {
            Ok(messages) => messages,
            Err(ResponsesMessagesError::BadRequest(error)) => {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    json!({ "error": { "message": error } }),
                    &state.settings,
                    headers.get(ORIGIN),
                );
            }
            Err(ResponsesMessagesError::NotFound(response_id)) => {
                return response_not_found_response(
                    &state.settings,
                    headers.get(ORIGIN),
                    &response_id,
                );
            }
        };
        return match run_agent_for_messages_async(
            Arc::clone(&state),
            request.model,
            messages.clone(),
        )
        .await
        {
            Ok(result) => responses_sse_response(
                &state,
                headers.get(ORIGIN),
                &result.final_response,
                &messages,
            ),
            Err(error) => json_response(
                StatusCode::BAD_GATEWAY,
                json!({ "error": { "message": error.to_string() } }),
                &state.settings,
                headers.get(ORIGIN),
            ),
        };
    }
    let messages = match build_responses_messages(
        &state,
        &request.input,
        request.previous_response_id.as_deref(),
    ) {
        Ok(messages) => messages,
        Err(ResponsesMessagesError::BadRequest(error)) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({ "error": { "message": error } }),
                &state.settings,
                headers.get(ORIGIN),
            );
        }
        Err(ResponsesMessagesError::NotFound(response_id)) => {
            return response_not_found_response(&state.settings, headers.get(ORIGIN), &response_id);
        }
    };
    match run_agent_for_messages_async(Arc::clone(&state), request.model, messages.clone()).await {
        Ok(result) => {
            let response_body = build_completed_responses_payload(
                &state.settings.model_name,
                &result.final_response,
            );
            let conversation_history =
                build_stored_conversation_history(&messages, &result.final_response);
            store_response_snapshot(&state, &response_body, &conversation_history);
            json_response(
                StatusCode::OK,
                response_body,
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

async fn handle_get_response(
    State(state): State<Arc<NativeApiServerState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let Some(response_body) = state.response_store.lock().ok().and_then(|store| {
        store
            .responses
            .get(&response_id)
            .map(|entry| entry.response.clone())
    }) else {
        return response_not_found_response(&state.settings, headers.get(ORIGIN), &response_id);
    };
    json_response(
        StatusCode::OK,
        response_body,
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_delete_response(
    State(state): State<Arc<NativeApiServerState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let deleted = state
        .response_store
        .lock()
        .ok()
        .map(|mut store| {
            store.order.retain(|id| id != &response_id);
            store.responses.remove(&response_id).is_some()
        })
        .unwrap_or(false);
    if !deleted {
        return response_not_found_response(&state.settings, headers.get(ORIGIN), &response_id);
    }
    json_response(
        StatusCode::OK,
        json!({
            "id": response_id,
            "object": "response.deleted",
            "deleted": true,
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_runs(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
    Json(request): Json<RunsRequest>,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let messages = match build_responses_messages(
        &state,
        &request.input,
        request.previous_response_id.as_deref(),
    ) {
        Ok(messages) => messages,
        Err(ResponsesMessagesError::BadRequest(error)) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({ "error": { "message": error } }),
                &state.settings,
                headers.get(ORIGIN),
            );
        }
        Err(ResponsesMessagesError::NotFound(response_id)) => {
            return response_not_found_response(&state.settings, headers.get(ORIGIN), &response_id);
        }
    };
    let run_id = format!("run-rs-{:x}", unix_ts_nanos());
    let created_at = unix_ts_secs();
    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let queued_model_name = model_name.clone();
    update_run_status(
        &state,
        &run_id,
        json!({
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "model": queued_model_name,
        }),
    );

    let state_for_task = Arc::clone(&state);
    let run_id_for_task = run_id.clone();
    let messages_for_task = messages.clone();
    let model_for_task = request.model.clone();
    let running_model_name = model_name.clone();
    tokio::spawn(async move {
        update_run_status(
            &state_for_task,
            &run_id_for_task,
            json!({
                "run_id": run_id_for_task,
                "status": "running",
                "created_at": created_at,
                "model": running_model_name,
            }),
        );
        match run_agent_for_messages_async(
            state_for_task.clone(),
            model_for_task,
            messages_for_task.clone(),
        )
        .await
        {
            Ok(result) => {
                let final_response = result.final_response;
                update_run_status(
                    &state_for_task,
                    &run_id_for_task,
                    json!({
                        "run_id": run_id_for_task,
                        "status": "completed",
                        "created_at": created_at,
                        "model": model_name,
                        "output": final_response,
                    }),
                );
            }
            Err(error) => {
                update_run_status(
                    &state_for_task,
                    &run_id_for_task,
                    json!({
                        "run_id": run_id_for_task,
                        "status": "failed",
                        "created_at": created_at,
                        "model": model_name,
                        "error": error,
                    }),
                );
            }
        }
    });

    json_response(
        StatusCode::ACCEPTED,
        json!({
            "run_id": run_id,
            "status": "started",
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_get_run(
    State(state): State<Arc<NativeApiServerState>>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let Some(status) = state
        .run_store
        .lock()
        .ok()
        .and_then(|store| store.runs.get(&run_id).cloned())
    else {
        return run_not_found_response(&state.settings, headers.get(ORIGIN), &run_id);
    };
    json_response(StatusCode::OK, status, &state.settings, headers.get(ORIGIN))
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
        .with_delegate_callback(move |request, parent_runtime| {
            delegate.execute(request, parent_runtime)
        });
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

fn build_responses_messages(
    state: &NativeApiServerState,
    input: &Value,
    previous_response_id: Option<&str>,
) -> Result<Vec<Value>, ResponsesMessagesError> {
    let mut messages =
        normalize_responses_input(input).map_err(ResponsesMessagesError::BadRequest)?;
    let Some(previous_response_id) = previous_response_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(messages);
    };
    let Some(previous_history) = state.response_store.lock().ok().and_then(|store| {
        store
            .responses
            .get(previous_response_id)
            .map(|entry| entry.conversation_history.clone())
    }) else {
        return Err(ResponsesMessagesError::NotFound(
            previous_response_id.to_string(),
        ));
    };
    let mut combined = previous_history;
    combined.append(&mut messages);
    Ok(combined)
}

fn build_stored_conversation_history(messages: &[Value], final_response: &str) -> Vec<Value> {
    let mut conversation_history = messages.to_vec();
    conversation_history.push(json!({
        "role": "assistant",
        "content": final_response,
    }));
    conversation_history
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

fn sse_response(
    body: String,
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache"),
    );
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    apply_cors_headers(response.headers_mut(), settings, origin);
    response
}

fn chat_completions_sse_response(
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
    final_response: &str,
) -> Response {
    let completion_id = format!("chatcmpl-rs-{:x}", unix_ts_nanos());
    let created = unix_ts_secs();
    let body = [
        format!(
            "data: {}\n\n",
            json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_name,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant" },
                    "finish_reason": Value::Null,
                }],
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_name,
                "choices": [{
                    "index": 0,
                    "delta": { "content": final_response },
                    "finish_reason": Value::Null,
                }],
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            })
        ),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();
    sse_response(body, settings, origin)
}

fn responses_sse_response(
    state: &NativeApiServerState,
    origin: Option<&HeaderValue>,
    final_response: &str,
    messages: &[Value],
) -> Response {
    let completed_response =
        build_completed_responses_payload(&state.settings.model_name, final_response);
    let conversation_history = build_stored_conversation_history(messages, final_response);
    store_response_snapshot(state, &completed_response, &conversation_history);
    let response_id = completed_response["id"].as_str().unwrap_or_default();
    let message_id = completed_response["output"][0]["id"]
        .as_str()
        .unwrap_or_default();
    let created_at = completed_response["created_at"]
        .as_u64()
        .unwrap_or_else(unix_ts_secs);
    let body = [
        format!(
            "event: response.created\ndata: {}\n\n",
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": state.settings.model_name,
                    "output": [],
                }
            })
        ),
        format!(
            "event: response.output_item.added\ndata: {}\n\n",
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
            })
        ),
        format!(
            "event: response.output_text.delta\ndata: {}\n\n",
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": final_response,
                "logprobs": [],
            })
        ),
        format!(
            "event: response.output_text.done\ndata: {}\n\n",
            json!({
                "type": "response.output_text.done",
                "sequence_number": 3,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "text": final_response,
                "logprobs": [],
            })
        ),
        format!(
            "event: response.output_item.done\ndata: {}\n\n",
            json!({
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 0,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": final_response,
                        "annotations": [],
                    }],
                }
            })
        ),
        format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type": "response.completed",
                "sequence_number": 5,
                "response": completed_response,
            })
        ),
    ]
    .concat();
    sse_response(body, &state.settings, origin)
}

fn build_completed_responses_payload(model_name: &str, final_response: &str) -> Value {
    json!({
        "id": format!("resp-rs-{:x}", unix_ts_nanos()),
        "object": "response",
        "created_at": unix_ts_secs(),
        "status": "completed",
        "model": model_name,
        "output": [{
            "id": format!("msg-rs-{:x}", unix_ts_nanos()),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": final_response,
                "annotations": [],
            }],
        }],
    })
}

fn store_response_snapshot(
    state: &NativeApiServerState,
    response_body: &Value,
    conversation_history: &[Value],
) {
    let Some(response_id) = response_body.get("id").and_then(Value::as_str) else {
        return;
    };
    let Ok(mut store) = state.response_store.lock() else {
        return;
    };
    store.order.retain(|id| id != response_id);
    store.order.push_back(response_id.to_string());
    store.responses.insert(
        response_id.to_string(),
        StoredResponse {
            response: response_body.clone(),
            conversation_history: conversation_history.to_vec(),
        },
    );
    while store.order.len() > MAX_STORED_RESPONSES {
        if let Some(oldest) = store.order.pop_front() {
            store.responses.remove(&oldest);
        }
    }
}

fn update_run_status(state: &NativeApiServerState, run_id: &str, status: Value) {
    let Ok(mut store) = state.run_store.lock() else {
        return;
    };
    store.runs.insert(run_id.to_string(), status);
}

fn response_not_found_response(
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
    response_id: &str,
) -> Response {
    json_response(
        StatusCode::NOT_FOUND,
        json!({ "error": { "message": format!("Response not found: {response_id}") } }),
        settings,
        origin,
    )
}

fn run_not_found_response(
    settings: &NativeApiServerSettings,
    origin: Option<&HeaderValue>,
    run_id: &str,
) -> Response {
    json_response(
        StatusCode::NOT_FOUND,
        json!({ "error": { "message": format!("Run not found: {run_id}") } }),
        settings,
        origin,
    )
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

    fn mock_model_server_sequence<F>(
        responses: Vec<String>,
        handler: F,
    ) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(Vec<String>) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let mut requests = Vec::new();
            for response_body in responses {
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
                    .unwrap();
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
                while request.len() < header_end + content_length {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                }
                requests.push(String::from_utf8_lossy(&request).to_string());
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
            handler(requests);
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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

    #[test]
    fn native_api_server_chat_completions_streams_sse_output() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "streamed hello"
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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
        let response = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": "say hi"
                }],
                "stream": true,
            }))
            .send()
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "text/event-stream"
        );
        let body = response.text().unwrap();
        assert!(body.contains("\"chat.completion.chunk\""));
        assert!(body.contains("\"role\":\"assistant\""));
        assert!(body.contains("\"content\":\"streamed hello\""));
        assert!(body.contains("data: [DONE]"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_responses_streams_sse_output() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "responses hello"
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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
        let response = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
                "stream": true,
            }))
            .send()
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "text/event-stream"
        );
        let body = response.text().unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("\"delta\":\"responses hello\""));
        assert!(body.contains("event: response.completed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_capabilities_and_response_lookup_work() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "stored hello"
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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

        let capabilities: Value = client
            .get(format!("http://{addr}/v1/capabilities"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(capabilities["features"]["stored_responses"], json!(true));
        assert_eq!(capabilities["features"]["runs_api"], json!(true));
        assert_eq!(
            capabilities["endpoints"]["response_get"]["path"],
            json!("/v1/responses/{response_id}")
        );
        assert_eq!(
            capabilities["endpoints"]["run_status"]["path"],
            json!("/v1/runs/{run_id}")
        );

        let response_body: Value = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        let response_id = response_body["id"].as_str().unwrap().to_string();

        let fetched: Value = client
            .get(format!("http://{addr}/v1/responses/{response_id}"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(fetched, response_body);

        let deleted: Value = client
            .delete(format!("http://{addr}/v1/responses/{response_id}"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(deleted["deleted"], json!(true));

        let missing = client
            .get(format!("http://{addr}/v1/responses/{response_id}"))
            .send()
            .unwrap();
        assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_runs_can_be_started_and_polled() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-run",
            "choices": [{
                "message": {
                    "content": "run hello"
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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

        let started: Value = client
            .post(format!("http://{addr}/v1/runs"))
            .json(&json!({
                "input": "say hi",
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(started["status"], json!("started"));
        let run_id = started["run_id"].as_str().unwrap().to_string();

        let mut status = Value::Null;
        for _ in 0..40 {
            status = client
                .get(format!("http://{addr}/v1/runs/{run_id}"))
                .send()
                .unwrap()
                .json()
                .unwrap();
            if status["status"] == json!("completed") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert_eq!(status["status"], json!("completed"));
        assert_eq!(status["output"], json!("run hello"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_responses_support_previous_response_id_chaining() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "id": "chatcmpl-first",
                    "choices": [{
                        "message": {
                            "content": "first reply"
                        }
                    }]
                })
                .to_string(),
                json!({
                    "id": "chatcmpl-second",
                    "choices": [{
                        "message": {
                            "content": "second reply"
                        }
                    }]
                })
                .to_string(),
            ],
            |requests| {
                assert_eq!(requests.len(), 2);
                let second_request = requests[1].split("\r\n\r\n").nth(1).unwrap_or_default();
                let payload: Value = serde_json::from_str(second_request).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("first")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("assistant")
                        && message["content"] == json!("first reply")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("second")
                }));
            },
        );
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
            response_store: Arc::new(Mutex::new(ResponseStore::default())),
            run_store: Arc::new(Mutex::new(RunStore::default())),
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
        let first_response: Value = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "first",
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        let previous_response_id = first_response["id"].as_str().unwrap().to_string();

        let second_response: Value = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "second",
                "previous_response_id": previous_response_id,
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(
            second_response["output"][0]["content"][0]["text"],
            json!("second reply")
        );

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }
}
