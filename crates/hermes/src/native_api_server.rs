use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fs;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hermes_core::{
    AgentProgressCallback, AgentProgressEvent, DelegateExecutor, HermesContext, LoadedConfig,
    ModelOverrides, SessionCreate, ToolRuntime, get_tool_definitions,
};
use serde::Deserialize;
use serde_json::{Value, json};
use serde_yaml::{Mapping, Value as YamlValue};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::gateway_cmd::GatewayRunArgs;

const DEFAULT_API_SERVER_HOST: &str = "127.0.0.1";
const DEFAULT_API_SERVER_PORT: u16 = 8642;
const MAX_STORED_RESPONSES: usize = 100;
const MAX_SESSION_HEADER_LEN: usize = 256;
#[cfg(test)]
const SSE_KEEPALIVE_INTERVAL: Duration = Duration::from_millis(50);
#[cfg(not(test))]
const SSE_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);

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
    conversations: BTreeMap<String, String>,
}

#[derive(Clone)]
struct StoredResponse {
    response: Value,
    conversation_history: Vec<Value>,
    instructions: Option<String>,
    session_id: Option<String>,
}

#[derive(Default)]
struct RunStore {
    runs: BTreeMap<String, StoredRun>,
}

struct StoredRun {
    status: Value,
    events: Vec<Value>,
    broadcaster: broadcast::Sender<Value>,
    interrupt_requested: Arc<AtomicBool>,
}

#[derive(Clone)]
struct PendingResponseToolCall {
    item_id: String,
    output_index: usize,
    call_id: String,
    name: String,
    arguments: String,
}

enum LiveResponseEvent {
    Progress(AgentProgressEvent),
    Completed(hermes_core::AgentTurnResult),
    Failed(String),
}

enum ResponsesMessagesError {
    BadRequest(String),
    NotFound(String),
}

struct ResolvedResponsesMessages {
    messages: Vec<Value>,
    conversation: Option<String>,
    instructions: Option<String>,
    session_id: Option<String>,
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
    conversation: Option<String>,
    #[serde(default)]
    conversation_history: Option<Vec<Value>>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    store: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RunsRequest {
    #[serde(default)]
    model: Option<String>,
    input: Value,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<String>,
    #[serde(default)]
    conversation_history: Option<Vec<Value>>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
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
        .route("/health/detailed", get(handle_health_detailed))
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
        .route(
            "/v1/runs/{run_id}/events",
            get(handle_run_events).options(handle_options),
        )
        .route(
            "/v1/runs/{run_id}/stop",
            post(handle_stop_run).options(handle_options),
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
            "platform": "hermes-agent",
            "model": state.settings.model_name,
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn handle_health_detailed(
    State(state): State<Arc<NativeApiServerState>>,
    headers: HeaderMap,
) -> Response {
    let runtime = read_gateway_runtime_status(&state.context);
    json_response(
        StatusCode::OK,
        json!({
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": runtime.get("gateway_state").cloned().unwrap_or(Value::Null),
            "platforms": runtime.get("platforms").cloned().unwrap_or_else(|| json!({})),
            "active_agents": runtime.get("active_agents").cloned().unwrap_or_else(|| json!(0)),
            "exit_reason": runtime.get("exit_reason").cloned().unwrap_or(Value::Null),
            "updated_at": runtime.get("updated_at").cloned().unwrap_or(Value::Null),
            "pid": std::process::id(),
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
            "platform": "hermes-agent",
            "model": state.settings.model_name,
            "auth": {
                "type": "bearer",
                "required": !state.settings.api_key.trim().is_empty(),
            },
            "features": {
                "chat_completions": true,
                "chat_completions_streaming": true,
                "responses_api": true,
                "responses_streaming": true,
                "stored_responses": true,
                "responses_conversation_aliases": true,
                "responses_conversation_history": true,
                "responses_instructions": true,
                "runs_api": true,
                "run_submission": true,
                "run_status": true,
                "run_events": true,
                "run_events_sse": true,
                "run_stop": true,
                "runs_conversation_history": true,
                "runs_instructions": true,
                "tool_progress_events": true,
                "session_continuity_header": "X-Hermes-Session-Id",
                "cors": !state.settings.cors_origins.is_empty(),
            },
            "endpoints": {
                "health": { "method": "GET", "path": "/health" },
                "health_detailed": { "method": "GET", "path": "/health/detailed" },
                "health_v1": { "method": "GET", "path": "/v1/health" },
                "models": { "method": "GET", "path": "/v1/models" },
                "capabilities": { "method": "GET", "path": "/v1/capabilities" },
                "chat_completions": { "method": "POST", "path": "/v1/chat/completions" },
                "responses": { "method": "POST", "path": "/v1/responses" },
                "response_get": { "method": "GET", "path": "/v1/responses/{response_id}" },
                "response_delete": { "method": "DELETE", "path": "/v1/responses/{response_id}" },
                "runs": { "method": "POST", "path": "/v1/runs" },
                "run_status": { "method": "GET", "path": "/v1/runs/{run_id}" },
                "run_events": { "method": "GET", "path": "/v1/runs/{run_id}/events" },
                "run_stop": { "method": "POST", "path": "/v1/runs/{run_id}/stop" },
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
    let continued_session_id =
        match resolve_chat_session_id(&state.settings, &headers, &request.messages) {
            Ok(session_id) => session_id,
            Err(response) => return response,
        };
    if request.stream.unwrap_or(false) {
        return chat_completions_streaming_response(
            Arc::clone(&state),
            headers.get(ORIGIN),
            request.model,
            request.messages,
            continued_session_id,
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
    let result = if let Some(session_id) = continued_session_id.clone() {
        run_chat_session_messages_async(
            Arc::clone(&state),
            request.model,
            request.messages,
            session_id,
            None,
            None,
        )
        .await
    } else {
        run_agent_for_messages_async(
            Arc::clone(&state),
            request.model,
            request.messages,
            None,
            None,
        )
        .await
    };
    match result {
        Ok(result) => {
            let response_id = format!("chatcmpl-rs-{:x}", unix_ts_nanos());
            let mut response = json_response(
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
            );
            if let Some(session_id) = result.session_id.as_deref() {
                if let Ok(value) = HeaderValue::from_str(session_id) {
                    response.headers_mut().insert("X-Hermes-Session-Id", value);
                }
            }
            response
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
        let resolved = match build_responses_messages(
            &state,
            &request.input,
            request.previous_response_id.as_deref(),
            request.conversation.as_deref(),
            request.conversation_history.as_deref(),
            request.instructions.as_deref(),
            request.session_id.as_deref(),
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
        return responses_streaming_response(
            Arc::clone(&state),
            headers.get(ORIGIN),
            request.model,
            resolved,
            request.store.unwrap_or(true),
        );
    }
    let resolved = match build_responses_messages(
        &state,
        &request.input,
        request.previous_response_id.as_deref(),
        request.conversation.as_deref(),
        request.conversation_history.as_deref(),
        request.instructions.as_deref(),
        request.session_id.as_deref(),
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
    let should_store = request.store.unwrap_or(true);
    let result = if let Some(session_id) = resolved.session_id.clone() {
        run_chat_session_messages_async(
            Arc::clone(&state),
            request.model,
            resolved.messages.clone(),
            session_id,
            None,
            None,
        )
        .await
    } else {
        run_agent_for_messages_async(
            Arc::clone(&state),
            request.model,
            resolved.messages.clone(),
            None,
            None,
        )
        .await
    };
    match result {
        Ok(result) => {
            let response_body = build_completed_responses_payload(
                &state.settings.model_name,
                &result.final_response,
            );
            if should_store {
                let conversation_history = build_stored_conversation_history(
                    &resolved.messages,
                    &result.final_response,
                    resolved.instructions.as_deref(),
                );
                store_response_snapshot(
                    &state,
                    &response_body,
                    &conversation_history,
                    resolved.instructions.as_deref(),
                    resolved.conversation.as_deref(),
                    result
                        .session_id
                        .as_deref()
                        .or(resolved.session_id.as_deref()),
                );
            }
            let mut response = json_response(
                StatusCode::OK,
                response_body,
                &state.settings,
                headers.get(ORIGIN),
            );
            if let Some(session_id) = result
                .session_id
                .as_deref()
                .or(resolved.session_id.as_deref())
                && let Ok(value) = HeaderValue::from_str(session_id)
            {
                response.headers_mut().insert("X-Hermes-Session-Id", value);
            }
            response
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
        .map(|mut store| remove_stored_response(&mut store, &response_id))
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
    let resolved = match build_responses_messages(
        &state,
        &request.input,
        request.previous_response_id.as_deref(),
        request.conversation.as_deref(),
        request.conversation_history.as_deref(),
        request.instructions.as_deref(),
        request.session_id.as_deref(),
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
    let interrupt_requested = Arc::new(AtomicBool::new(false));
    initialize_run(
        &state,
        &run_id,
        "run.queued",
        json!({
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "session_id": resolved.session_id.clone().unwrap_or_else(|| run_id.clone()),
            "model": queued_model_name,
        }),
        Arc::clone(&interrupt_requested),
    );

    let state_for_task = Arc::clone(&state);
    let run_id_for_task = run_id.clone();
    let messages_for_task = resolved.messages.clone();
    let session_id_for_task = resolved.session_id.clone();
    let model_for_task = request.model.clone();
    let running_model_name = model_name.clone();
    let interrupt_requested_for_task = Arc::clone(&interrupt_requested);
    let progress_callback: AgentProgressCallback = {
        let state = Arc::clone(&state);
        let run_id = run_id.clone();
        Arc::new(move |event| append_progress_run_event(&state, &run_id, event))
    };
    tokio::spawn(async move {
        if interrupt_requested_for_task.load(Ordering::SeqCst) {
            record_run_status(
                &state_for_task,
                &run_id_for_task,
                "run.cancelled",
                json!({
                    "run_id": run_id_for_task,
                    "status": "cancelled",
                    "created_at": created_at,
                    "session_id": session_id_for_task.clone().unwrap_or_else(|| run_id_for_task.clone()),
                    "model": running_model_name,
                }),
            );
            return;
        }
        record_run_status(
            &state_for_task,
            &run_id_for_task,
            "run.running",
            json!({
                "run_id": run_id_for_task,
                "status": "running",
                "created_at": created_at,
                "session_id": session_id_for_task.clone().unwrap_or_else(|| run_id_for_task.clone()),
                "model": running_model_name,
            }),
        );
        let result = if let Some(session_id) = session_id_for_task.clone() {
            run_chat_session_messages_async(
                state_for_task.clone(),
                model_for_task,
                messages_for_task.clone(),
                session_id,
                None,
                None,
            )
            .await
        } else {
            run_agent_for_messages_async(
                state_for_task.clone(),
                model_for_task,
                messages_for_task.clone(),
                Some(Arc::clone(&interrupt_requested_for_task)),
                Some(progress_callback),
            )
            .await
        };
        match result {
            Ok(result) => {
                let final_response = result.final_response;
                record_run_status(
                    &state_for_task,
                    &run_id_for_task,
                    "run.completed",
                    json!({
                        "run_id": run_id_for_task,
                        "status": "completed",
                        "created_at": created_at,
                        "session_id": result.session_id.clone().or(session_id_for_task.clone()).unwrap_or_else(|| run_id_for_task.clone()),
                        "model": model_name,
                        "output": final_response,
                    }),
                );
            }
            Err(error) => {
                if interrupt_requested_for_task.load(Ordering::SeqCst)
                    || error.trim() == "running agent turn: Run interrupted."
                {
                    record_run_status(
                        &state_for_task,
                        &run_id_for_task,
                        "run.cancelled",
                        json!({
                            "run_id": run_id_for_task,
                            "status": "cancelled",
                            "created_at": created_at,
                            "session_id": session_id_for_task.clone().unwrap_or_else(|| run_id_for_task.clone()),
                            "model": model_name,
                        }),
                    );
                    return;
                }
                record_run_status(
                    &state_for_task,
                    &run_id_for_task,
                    "run.failed",
                    json!({
                        "run_id": run_id_for_task,
                        "status": "failed",
                        "created_at": created_at,
                        "session_id": session_id_for_task.clone().unwrap_or_else(|| run_id_for_task.clone()),
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
        .and_then(|store| store.runs.get(&run_id).map(|run| run.status.clone()))
    else {
        return run_not_found_response(&state.settings, headers.get(ORIGIN), &run_id);
    };
    json_response(StatusCode::OK, status, &state.settings, headers.get(ORIGIN))
}

async fn handle_run_events(
    State(state): State<Arc<NativeApiServerState>>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let Some((history, mut subscriber)) = state.run_store.lock().ok().and_then(|store| {
        store
            .runs
            .get(&run_id)
            .map(|run| (run.events.clone(), run.broadcaster.subscribe()))
    }) else {
        return run_not_found_response(&state.settings, headers.get(ORIGIN), &run_id);
    };

    let (tx, rx) = mpsc::channel::<String>(16);
    tokio::spawn(async move {
        for event in history {
            if tx.send(format_run_event_chunk(&event)).await.is_err() {
                return;
            }
            if run_event_is_terminal(&event) {
                let _ = tx.send(": stream closed\n\n".to_string()).await;
                return;
            }
        }

        loop {
            match tokio::time::timeout(SSE_KEEPALIVE_INTERVAL, subscriber.recv()).await {
                Ok(Ok(event)) => {
                    let terminal = run_event_is_terminal(&event);
                    if tx.send(format_run_event_chunk(&event)).await.is_err() {
                        return;
                    }
                    if terminal {
                        let _ = tx.send(": stream closed\n\n".to_string()).await;
                        return;
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(broadcast::error::RecvError::Closed)) => {
                    let _ = tx.send(": stream closed\n\n".to_string()).await;
                    return;
                }
                Err(_) => {
                    if tx.send(": keepalive\n\n".to_string()).await.is_err() {
                        return;
                    }
                }
            }
        }
    });

    let mut response = Response::new(Body::from_stream(
        ReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>),
    ));
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
    apply_cors_headers(response.headers_mut(), &state.settings, headers.get(ORIGIN));
    response
}

async fn handle_stop_run(
    State(state): State<Arc<NativeApiServerState>>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = ensure_authorized(&state.settings, &headers) {
        return response;
    }
    let Some((status, interrupt_requested)) = state.run_store.lock().ok().and_then(|store| {
        store
            .runs
            .get(&run_id)
            .map(|run| (run.status.clone(), Arc::clone(&run.interrupt_requested)))
    }) else {
        return run_not_found_response(&state.settings, headers.get(ORIGIN), &run_id);
    };

    if run_status_is_terminal(&status) {
        return json_response(
            StatusCode::CONFLICT,
            json!({ "error": { "message": format!("Run is already terminal: {run_id}") } }),
            &state.settings,
            headers.get(ORIGIN),
        );
    }

    interrupt_requested.store(true, Ordering::SeqCst);
    record_run_status(
        &state,
        &run_id,
        "run.stopping",
        json!({
            "run_id": run_id,
            "status": "stopping",
            "created_at": status.get("created_at").cloned().unwrap_or_else(|| json!(unix_ts_secs())),
            "model": status
                .get("model")
                .cloned()
                .unwrap_or_else(|| json!(state.settings.model_name.clone())),
        }),
    );
    json_response(
        StatusCode::OK,
        json!({
            "run_id": run_id,
            "status": "stopping",
        }),
        &state.settings,
        headers.get(ORIGIN),
    )
}

async fn run_agent_for_messages_async(
    state: Arc<NativeApiServerState>,
    model: Option<String>,
    messages: Vec<Value>,
    interrupt_requested: Option<Arc<AtomicBool>>,
    progress_callback: Option<AgentProgressCallback>,
) -> Result<hermes_core::AgentTurnResult, String> {
    tokio::task::spawn_blocking(move || {
        run_agent_for_messages(
            &state,
            model,
            &messages,
            interrupt_requested,
            progress_callback,
        )
        .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| error.to_string())?
}

async fn run_chat_session_messages_async(
    state: Arc<NativeApiServerState>,
    model: Option<String>,
    messages: Vec<Value>,
    session_id: String,
    interrupt_requested: Option<Arc<AtomicBool>>,
    progress_callback: Option<AgentProgressCallback>,
) -> Result<hermes_core::AgentTurnResult, String> {
    tokio::task::spawn_blocking(move || {
        run_chat_session_messages(
            &state,
            model,
            &messages,
            &session_id,
            interrupt_requested.as_ref(),
            progress_callback.as_ref(),
        )
        .map_err(|error| error.to_string())
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
    interrupt_requested: Option<Arc<AtomicBool>>,
    progress_callback: Option<AgentProgressCallback>,
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
    Ok(state.context.run_chat_turn_with_messages_interruptible(
        &state.loaded,
        messages,
        &runtime,
        Some(&enabled_toolsets),
        &overrides,
        interrupt_requested.as_ref(),
        progress_callback.as_ref(),
    )?)
}

fn run_chat_session_messages(
    state: &NativeApiServerState,
    model: Option<String>,
    messages: &[Value],
    session_id: &str,
    interrupt_requested: Option<&Arc<AtomicBool>>,
    progress_callback: Option<&AgentProgressCallback>,
) -> Result<hermes_core::AgentTurnResult, Box<dyn Error>> {
    let user_content =
        extract_last_user_content(messages).map_err(|error| io::Error::other(error))?;
    let system_prompt = extract_system_prompt(messages);
    let enabled_toolsets = state.loaded.config.toolsets.clone();
    let delegate = DelegateExecutor::new(
        state.context.clone(),
        state.loaded.clone(),
        "rust-api-server",
        enabled_toolsets.clone(),
        ModelOverrides::default(),
        state.context.hermes_home(),
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
    let session_store = state.context.open_session_store()?;
    if session_store.get_session(session_id)?.is_none() {
        session_store.create_session(&SessionCreate {
            id: session_id.to_string(),
            source: "rust-api-server".to_string(),
            user_id: None,
            model: Some(
                overrides
                    .model
                    .clone()
                    .unwrap_or_else(|| state.settings.model_name.clone()),
            ),
            model_config: None,
            system_prompt: system_prompt.clone(),
            parent_session_id: None,
        })?;
    }
    if session_store.get_messages(session_id)?.is_empty() {
        seed_chat_completion_session_history(&session_store, session_id, messages)?;
    }
    Ok(state
        .context
        .run_chat_turn_with_user_content_interruptible(
            &state.loaded,
            user_content,
            &runtime,
            Some(&enabled_toolsets),
            &overrides,
            Some(session_id),
            Some(&session_store),
            interrupt_requested,
            progress_callback,
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

fn parse_session_continuation_header(
    settings: &NativeApiServerSettings,
    headers: &HeaderMap,
) -> Result<Option<String>, Response> {
    let Some(raw) = headers
        .get("X-Hermes-Session-Id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    if settings.api_key.trim().is_empty() {
        return Err(json_response(
            StatusCode::FORBIDDEN,
            json!({ "error": { "message": "Session continuation requires API key authentication. Configure API_SERVER_KEY to enable this feature." } }),
            settings,
            headers.get(ORIGIN),
        ));
    }
    if raw.len() > MAX_SESSION_HEADER_LEN {
        return Err(json_response(
            StatusCode::BAD_REQUEST,
            json!({ "error": { "message": "Session ID too long" } }),
            settings,
            headers.get(ORIGIN),
        ));
    }
    if raw.chars().any(|ch| matches!(ch, '\r' | '\n' | '\0')) {
        return Err(json_response(
            StatusCode::BAD_REQUEST,
            json!({ "error": { "message": "Invalid session ID" } }),
            settings,
            headers.get(ORIGIN),
        ));
    }
    Ok(Some(raw.to_string()))
}

fn resolve_chat_session_id(
    settings: &NativeApiServerSettings,
    headers: &HeaderMap,
    messages: &[Value],
) -> Result<Option<String>, Response> {
    if let Some(explicit) = parse_session_continuation_header(settings, headers)? {
        return Ok(Some(explicit));
    }
    let system_prompt = extract_system_prompt(messages);
    let first_user_message = messages.iter().find_map(|message| {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            return None;
        }
        normalize_chat_text(message.get("content")?)
    });
    Ok(first_user_message
        .map(|first_user| derive_chat_session_id(system_prompt.as_deref(), &first_user)))
}

fn derive_chat_session_id(system_prompt: Option<&str>, first_user_message: &str) -> String {
    let seed = format!(
        "{}\n{}",
        system_prompt.unwrap_or_default(),
        first_user_message
    );
    let digest = Sha256::digest(seed.as_bytes());
    let hex = digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("api-{hex}")
}

fn read_gateway_runtime_status(context: &HermesContext) -> serde_json::Map<String, Value> {
    let path = context.hermes_home().join("gateway_state.json");
    let Ok(raw) = fs::read_to_string(path) else {
        return serde_json::Map::new();
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&raw) else {
        return serde_json::Map::new();
    };
    parsed.as_object().cloned().unwrap_or_default()
}

fn extract_last_user_content(messages: &[Value]) -> Result<Value, String> {
    messages
        .iter()
        .rev()
        .find(|message| {
            message
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| role == "user")
        })
        .and_then(|message| message.get("content").cloned())
        .filter(chat_content_is_meaningful)
        .ok_or_else(|| "messages must include a non-empty user message".to_string())
}

fn seed_chat_completion_session_history(
    session_store: &hermes_core::SessionStore,
    session_id: &str,
    messages: &[Value],
) -> Result<(), Box<dyn Error>> {
    let last_user_index = messages
        .iter()
        .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"));
    let Some(last_user_index) = last_user_index else {
        return Ok(());
    };
    for message in &messages[..last_user_index] {
        let Some(role) = message
            .get("role")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        if role == "system" {
            continue;
        }
        let tool_calls = message.get("tool_calls").cloned();
        let tool_name = tool_calls
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|calls| calls.first())
            .and_then(|call| call.get("function"))
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        session_store.append_message(
            session_id,
            &hermes_core::MessageAppend {
                role: role.to_string(),
                content: message.get("content").cloned(),
                tool_call_id: message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                tool_calls,
                tool_name,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        )?;
    }
    Ok(())
}

fn extract_system_prompt(messages: &[Value]) -> Option<String> {
    messages.iter().find_map(|message| {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            return None;
        }
        normalize_chat_text(message.get("content")?)
    })
}

fn normalize_chat_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Object(object) => object
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .map(ToOwned::to_owned),
                    _ => None,
                })
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn chat_content_is_meaningful(content: &Value) -> bool {
    match content {
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(parts) => parts.iter().any(|part| match part {
            Value::String(text) => !text.trim().is_empty(),
            Value::Object(object) => object
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.trim().is_empty()),
            _ => false,
        }),
        Value::Object(object) => object
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.trim().is_empty()),
        _ => false,
    }
}

fn build_responses_messages(
    state: &NativeApiServerState,
    input: &Value,
    previous_response_id: Option<&str>,
    conversation: Option<&str>,
    conversation_history: Option<&[Value]>,
    instructions: Option<&str>,
    session_id: Option<&str>,
) -> Result<ResolvedResponsesMessages, ResponsesMessagesError> {
    let normalized_previous_response_id = previous_response_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let normalized_conversation = conversation
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    if normalized_conversation.is_some() && normalized_previous_response_id.is_some() {
        return Err(ResponsesMessagesError::BadRequest(
            "cannot use both conversation and previous_response_id".to_string(),
        ));
    }

    let resolved_previous_response_id =
        if let Some(conversation_name) = normalized_conversation.as_deref() {
            state.response_store.lock().ok().and_then(|store| {
                store
                    .conversations
                    .get(conversation_name)
                    .cloned()
                    .filter(|value| !value.trim().is_empty())
            })
        } else {
            normalized_previous_response_id
        };

    let stored_response = resolved_previous_response_id
        .as_deref()
        .map(|response_id| load_stored_response(state, response_id))
        .transpose()?
        .flatten();

    let mut messages = if let Some(history) = conversation_history {
        normalize_message_history(history)?
    } else if let Some(previous_response) = stored_response.as_ref() {
        previous_response.conversation_history.clone()
    } else {
        Vec::new()
    };

    let carried_instructions = if conversation_history.is_none() {
        stored_response
            .as_ref()
            .and_then(|entry| entry.instructions.clone())
    } else {
        None
    };
    let effective_session_id = session_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            stored_response
                .as_ref()
                .and_then(|entry| entry.session_id.clone())
        });
    let effective_instructions = instructions
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or(carried_instructions);
    if let Some(instructions) = effective_instructions.as_deref() {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": instructions,
            }),
        );
    }

    let mut input_messages =
        normalize_responses_input(input).map_err(ResponsesMessagesError::BadRequest)?;
    messages.append(&mut input_messages);
    Ok(ResolvedResponsesMessages {
        messages,
        conversation: normalized_conversation,
        instructions: effective_instructions,
        session_id: effective_session_id,
    })
}

fn build_stored_conversation_history_with_assistant(
    messages: &[Value],
    assistant_response: Option<&str>,
    instructions: Option<&str>,
) -> Vec<Value> {
    let mut conversation_history = messages.to_vec();
    if let Some(expected) = instructions
        .map(str::trim)
        .filter(|value| !value.is_empty())
        && conversation_history.first().is_some_and(|message| {
            message.get("role").and_then(Value::as_str) == Some("system")
                && message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.trim() == expected)
        })
    {
        conversation_history.remove(0);
    }
    if let Some(response) = assistant_response
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        conversation_history.push(json!({
            "role": "assistant",
            "content": response,
        }));
    }
    conversation_history
}

fn build_stored_conversation_history(
    messages: &[Value],
    final_response: &str,
    instructions: Option<&str>,
) -> Vec<Value> {
    build_stored_conversation_history_with_assistant(messages, Some(final_response), instructions)
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

fn chat_completions_chunk(
    completion_id: &str,
    created: u64,
    model_name: &str,
    delta: Value,
    finish_reason: Value,
    include_usage: bool,
) -> String {
    let mut body = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    if include_usage {
        body["usage"] = json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        });
    }
    format!("data: {body}\n\n")
}

fn chat_completions_tool_progress_event(
    tool_name: &str,
    tool_call_id: &str,
    arguments: Option<&str>,
    status: &str,
) -> Option<String> {
    if tool_name.trim().is_empty() || tool_name.starts_with('_') || tool_call_id.trim().is_empty() {
        return None;
    }
    let mut payload = json!({
        "tool": tool_name,
        "toolCallId": tool_call_id,
        "status": status,
    });
    if status == "running" {
        payload["label"] = json!(tool_name);
    }
    if let Some(arguments) = arguments.map(str::trim).filter(|value| !value.is_empty()) {
        payload["arguments"] = json!(arguments);
    }
    Some(format!("event: hermes.tool.progress\ndata: {payload}\n\n"))
}

fn chat_completions_streaming_response(
    state: Arc<NativeApiServerState>,
    origin: Option<&HeaderValue>,
    model: Option<String>,
    messages: Vec<Value>,
    session_id: Option<String>,
) -> Response {
    let completion_id = format!("chatcmpl-rs-{:x}", unix_ts_nanos());
    let created = unix_ts_secs();
    let model_name = model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let origin = origin.cloned();
    let interrupt_requested = Arc::new(AtomicBool::new(false));

    let (body_tx, body_rx) = mpsc::channel::<String>(32);
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<LiveResponseEvent>();
    let progress_callback: AgentProgressCallback = {
        let event_tx = event_tx.clone();
        Arc::new(move |event| {
            let _ = event_tx.send(LiveResponseEvent::Progress(event));
        })
    };
    let state_for_run = Arc::clone(&state);
    let messages_for_run = messages.clone();
    let session_id_for_run = session_id.clone();
    let interrupt_for_run = Arc::clone(&interrupt_requested);
    let model_name_for_run = model_name.clone();
    tokio::spawn(async move {
        let result = if let Some(session_id) = session_id_for_run {
            run_chat_session_messages_async(
                state_for_run,
                Some(model_name_for_run.clone()),
                messages_for_run,
                session_id,
                Some(interrupt_for_run),
                Some(progress_callback),
            )
            .await
        } else {
            run_agent_for_messages_async(
                state_for_run,
                Some(model_name_for_run.clone()),
                messages_for_run,
                Some(interrupt_for_run),
                Some(progress_callback),
            )
            .await
        };
        let terminal = match result {
            Ok(result) => LiveResponseEvent::Completed(result),
            Err(error) => LiveResponseEvent::Failed(error),
        };
        let _ = event_tx.send(terminal);
    });

    let model_name_for_stream = model_name.clone();
    tokio::spawn(async move {
        let role_chunk = chat_completions_chunk(
            &completion_id,
            created,
            &model_name_for_stream,
            json!({ "role": "assistant" }),
            Value::Null,
            false,
        );
        if !send_sse_chunk(&body_tx, &interrupt_requested, role_chunk).await {
            return;
        }

        let mut emitted_text = false;
        loop {
            let event = match tokio::time::timeout(SSE_KEEPALIVE_INTERVAL, event_rx.recv()).await {
                Ok(Some(event)) => event,
                Ok(None) => return,
                Err(_) => {
                    if !send_sse_chunk(
                        &body_tx,
                        &interrupt_requested,
                        ": keepalive\n\n".to_string(),
                    )
                    .await
                    {
                        return;
                    }
                    continue;
                }
            };
            match event {
                LiveResponseEvent::Progress(AgentProgressEvent::MessageDelta { delta }) => {
                    emitted_text = true;
                    let chunk = chat_completions_chunk(
                        &completion_id,
                        created,
                        &model_name_for_stream,
                        json!({ "content": delta }),
                        Value::Null,
                        false,
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                        return;
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::ToolStarted {
                    tool_call_id,
                    tool_name,
                    arguments,
                }) => {
                    if let Some(chunk) = chat_completions_tool_progress_event(
                        &tool_name,
                        &tool_call_id,
                        Some(&arguments),
                        "running",
                    ) {
                        if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                            return;
                        }
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::ToolCompleted {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                }) => {
                    if let Some(chunk) = chat_completions_tool_progress_event(
                        &tool_name,
                        &tool_call_id,
                        Some(&arguments),
                        "completed",
                    ) {
                        if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                            return;
                        }
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::ReasoningAvailable { .. }) => {}
                LiveResponseEvent::Completed(result) => {
                    if !emitted_text && !result.final_response.trim().is_empty() {
                        let chunk = chat_completions_chunk(
                            &completion_id,
                            created,
                            &model_name_for_stream,
                            json!({ "content": result.final_response }),
                            Value::Null,
                            false,
                        );
                        if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                            return;
                        }
                    }
                    let finish_chunk = chat_completions_chunk(
                        &completion_id,
                        created,
                        &model_name_for_stream,
                        json!({}),
                        json!("stop"),
                        true,
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, finish_chunk).await {
                        return;
                    }
                    let _ = send_sse_chunk(
                        &body_tx,
                        &interrupt_requested,
                        "data: [DONE]\n\n".to_string(),
                    )
                    .await;
                    return;
                }
                LiveResponseEvent::Failed(_) => {
                    let error_chunk = chat_completions_chunk(
                        &completion_id,
                        created,
                        &model_name_for_stream,
                        json!({}),
                        json!("error"),
                        false,
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, error_chunk).await {
                        return;
                    }
                    let _ = send_sse_chunk(
                        &body_tx,
                        &interrupt_requested,
                        "data: [DONE]\n\n".to_string(),
                    )
                    .await;
                    return;
                }
            }
        }
    });

    let mut response = Response::new(Body::from_stream(
        ReceiverStream::new(body_rx).map(Ok::<_, io::Error>),
    ));
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
    if let Some(session_id) = session_id.as_deref()
        && let Ok(value) = HeaderValue::from_str(session_id)
    {
        response.headers_mut().insert("X-Hermes-Session-Id", value);
    }
    apply_cors_headers(response.headers_mut(), &state.settings, origin.as_ref());
    response
}

async fn send_sse_chunk(
    sender: &mpsc::Sender<String>,
    interrupt_requested: &Arc<AtomicBool>,
    chunk: String,
) -> bool {
    if sender.send(chunk).await.is_err() {
        interrupt_requested.store(true, Ordering::SeqCst);
        return false;
    }
    true
}

fn format_sse_event(event_name: &str, sequence_number: &mut u64, payload: Value) -> String {
    let mut object = payload.as_object().cloned().unwrap_or_default();
    object.insert("sequence_number".to_string(), json!(*sequence_number));
    *sequence_number += 1;
    format!("event: {event_name}\ndata: {}\n\n", Value::Object(object))
}

fn responses_streaming_response(
    state: Arc<NativeApiServerState>,
    origin: Option<&HeaderValue>,
    model: Option<String>,
    resolved: ResolvedResponsesMessages,
    should_store: bool,
) -> Response {
    let response_id = format!("resp-rs-{:x}", unix_ts_nanos());
    let created_at = unix_ts_secs();
    let model_name = model.unwrap_or_else(|| state.settings.model_name.clone());
    let origin = origin.cloned();
    let interrupt_requested = Arc::new(AtomicBool::new(false));

    let (body_tx, body_rx) = mpsc::channel::<String>(32);
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<LiveResponseEvent>();

    let progress_callback: AgentProgressCallback = {
        let event_tx = event_tx.clone();
        Arc::new(move |event| {
            let _ = event_tx.send(LiveResponseEvent::Progress(event));
        })
    };
    let state_for_run = Arc::clone(&state);
    let messages_for_run = resolved.messages.clone();
    let session_id_for_run = resolved.session_id.clone();
    let interrupt_for_run = Arc::clone(&interrupt_requested);
    tokio::spawn(async move {
        let result = if let Some(session_id) = session_id_for_run {
            run_chat_session_messages_async(
                state_for_run,
                Some(model_name.clone()),
                messages_for_run,
                session_id,
                Some(interrupt_for_run),
                Some(progress_callback),
            )
            .await
        } else {
            run_agent_for_messages_async(
                state_for_run,
                Some(model_name.clone()),
                messages_for_run,
                Some(interrupt_for_run),
                Some(progress_callback),
            )
            .await
        };
        let terminal = match result {
            Ok(result) => LiveResponseEvent::Completed(result),
            Err(error) => LiveResponseEvent::Failed(error),
        };
        let _ = event_tx.send(terminal);
    });

    let state_for_stream = Arc::clone(&state);
    let settings = state.settings.clone();
    let instructions = resolved.instructions.clone();
    let conversation = resolved.conversation.clone();
    let session_id = resolved.session_id.clone();
    let messages = resolved.messages;
    let response_id_for_stream = response_id.clone();
    tokio::spawn(async move {
        let mut sequence_number = 0_u64;
        let mut output_index = 0_usize;
        let mut pending_tool_calls = Vec::<PendingResponseToolCall>::new();
        let mut emitted_items = Vec::<Value>::new();
        let message_item_id = format!("msg-rs-{:x}", unix_ts_nanos());
        let mut message_opened = false;
        let mut message_output_index = None::<usize>;
        let mut final_text = String::new();

        let created_chunk = format_sse_event(
            "response.created",
            &mut sequence_number,
            json!({
                "type": "response.created",
                "response": {
                    "id": response_id_for_stream,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": settings.model_name,
                    "output": [],
                }
            }),
        );
        if should_store {
            persist_stream_response_snapshot(
                &state_for_stream,
                &settings.model_name,
                &response_id_for_stream,
                created_at,
                "in_progress",
                Vec::new(),
                None,
                None,
                &messages,
                instructions.as_deref(),
                conversation.as_deref(),
                session_id.as_deref(),
            );
        }
        if !send_sse_chunk(&body_tx, &interrupt_requested, created_chunk).await {
            if should_store {
                persist_incomplete_stream_response_snapshot(
                    &state_for_stream,
                    &settings.model_name,
                    &response_id_for_stream,
                    created_at,
                    &emitted_items,
                    &final_text,
                    &messages,
                    instructions.as_deref(),
                    conversation.as_deref(),
                    session_id.as_deref(),
                );
            }
            return;
        }

        loop {
            let event = match tokio::time::timeout(SSE_KEEPALIVE_INTERVAL, event_rx.recv()).await {
                Ok(Some(event)) => event,
                Ok(None) => return,
                Err(_) => {
                    if !send_sse_chunk(
                        &body_tx,
                        &interrupt_requested,
                        ": keepalive\n\n".to_string(),
                    )
                    .await
                    {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                    continue;
                }
            };
            match event {
                LiveResponseEvent::Progress(AgentProgressEvent::ToolStarted {
                    tool_call_id,
                    tool_name,
                    arguments,
                }) => {
                    let item_id = format!("fc-rs-{:x}", unix_ts_nanos());
                    let item = json!({
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": tool_name,
                        "call_id": tool_call_id,
                        "arguments": arguments,
                    });
                    pending_tool_calls.push(PendingResponseToolCall {
                        item_id,
                        output_index,
                        call_id: tool_call_id,
                        name: item["name"].as_str().unwrap_or_default().to_string(),
                        arguments: item["arguments"].as_str().unwrap_or_default().to_string(),
                    });
                    emitted_items.push(json!({
                        "type": "function_call",
                        "name": item["name"],
                        "call_id": item["call_id"],
                        "arguments": item["arguments"],
                    }));
                    let chunk = format_sse_event(
                        "response.output_item.added",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": item,
                        }),
                    );
                    output_index += 1;
                    if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::ToolCompleted {
                    tool_call_id,
                    tool_name: _,
                    arguments: _,
                    result,
                    duration_secs: _,
                    is_error: _,
                }) => {
                    let Some(pending_index) = pending_tool_calls
                        .iter()
                        .position(|call| call.call_id == tool_call_id)
                    else {
                        continue;
                    };
                    let pending = pending_tool_calls.remove(pending_index);
                    let done_chunk = format_sse_event(
                        "response.output_item.done",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_item.done",
                            "output_index": pending.output_index,
                            "item": {
                                "id": pending.item_id,
                                "type": "function_call",
                                "status": "completed",
                                "name": pending.name,
                                "call_id": pending.call_id,
                                "arguments": pending.arguments,
                            }
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, done_chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                    let output_item = json!({
                        "id": format!("fco-rs-{:x}", unix_ts_nanos()),
                        "type": "function_call_output",
                        "call_id": pending.call_id,
                        "output": [{
                            "type": "input_text",
                            "text": result,
                        }],
                        "status": "completed",
                    });
                    emitted_items.push(json!({
                        "type": "function_call_output",
                        "call_id": output_item["call_id"],
                        "output": output_item["output"],
                    }));
                    let added_chunk = format_sse_event(
                        "response.output_item.added",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": output_item,
                        }),
                    );
                    output_index += 1;
                    if !send_sse_chunk(&body_tx, &interrupt_requested, added_chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                    let done_chunk = format_sse_event(
                        "response.output_item.done",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_item.done",
                            "output_index": output_index - 1,
                            "item": output_item,
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, done_chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::MessageDelta { delta }) => {
                    if !message_opened {
                        let chunk = format_sse_event(
                            "response.output_item.added",
                            &mut sequence_number,
                            json!({
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": {
                                    "id": message_item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                }
                            }),
                        );
                        message_output_index = Some(output_index);
                        output_index += 1;
                        message_opened = true;
                        if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                            if should_store {
                                persist_incomplete_stream_response_snapshot(
                                    &state_for_stream,
                                    &settings.model_name,
                                    &response_id_for_stream,
                                    created_at,
                                    &emitted_items,
                                    &final_text,
                                    &messages,
                                    instructions.as_deref(),
                                    conversation.as_deref(),
                                    session_id.as_deref(),
                                );
                            }
                            return;
                        }
                    }
                    final_text.push_str(&delta);
                    let chunk = format_sse_event(
                        "response.output_text.delta",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_text.delta",
                            "item_id": message_item_id,
                            "output_index": message_output_index.unwrap_or(0),
                            "content_index": 0,
                            "delta": delta,
                            "logprobs": [],
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                }
                LiveResponseEvent::Progress(AgentProgressEvent::ReasoningAvailable { text }) => {
                    let chunk = format_sse_event(
                        "response.reasoning_summary_part.added",
                        &mut sequence_number,
                        json!({
                            "type": "response.reasoning_summary_part.added",
                            "text": text,
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                }
                LiveResponseEvent::Completed(result) => {
                    if final_text.is_empty() && !result.final_response.trim().is_empty() {
                        if !message_opened {
                            let chunk = format_sse_event(
                                "response.output_item.added",
                                &mut sequence_number,
                                json!({
                                    "type": "response.output_item.added",
                                    "output_index": output_index,
                                    "item": {
                                        "id": message_item_id,
                                        "type": "message",
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": [],
                                    }
                                }),
                            );
                            message_output_index = Some(output_index);
                            if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                                if should_store {
                                    persist_incomplete_stream_response_snapshot(
                                        &state_for_stream,
                                        &settings.model_name,
                                        &response_id_for_stream,
                                        created_at,
                                        &emitted_items,
                                        &final_text,
                                        &messages,
                                        instructions.as_deref(),
                                        conversation.as_deref(),
                                        session_id.as_deref(),
                                    );
                                }
                                return;
                            }
                        }
                        final_text = result.final_response.clone();
                        let chunk = format_sse_event(
                            "response.output_text.delta",
                            &mut sequence_number,
                            json!({
                                "type": "response.output_text.delta",
                                "item_id": message_item_id,
                                "output_index": message_output_index.unwrap_or(0),
                                "content_index": 0,
                                "delta": result.final_response,
                                "logprobs": [],
                            }),
                        );
                        if !send_sse_chunk(&body_tx, &interrupt_requested, chunk).await {
                            if should_store {
                                persist_incomplete_stream_response_snapshot(
                                    &state_for_stream,
                                    &settings.model_name,
                                    &response_id_for_stream,
                                    created_at,
                                    &emitted_items,
                                    &final_text,
                                    &messages,
                                    instructions.as_deref(),
                                    conversation.as_deref(),
                                    session_id.as_deref(),
                                );
                            }
                            return;
                        }
                    }
                    let text_done = format_sse_event(
                        "response.output_text.done",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_text.done",
                            "item_id": message_item_id,
                            "output_index": message_output_index.unwrap_or(0),
                            "content_index": 0,
                            "text": final_text,
                            "logprobs": [],
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, text_done).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                    let message_item = json!({
                        "id": message_item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                        }],
                    });
                    let item_done = format_sse_event(
                        "response.output_item.done",
                        &mut sequence_number,
                        json!({
                            "type": "response.output_item.done",
                            "output_index": message_output_index.unwrap_or(0),
                            "item": message_item.clone(),
                        }),
                    );
                    if !send_sse_chunk(&body_tx, &interrupt_requested, item_done).await {
                        if should_store {
                            persist_incomplete_stream_response_snapshot(
                                &state_for_stream,
                                &settings.model_name,
                                &response_id_for_stream,
                                created_at,
                                &emitted_items,
                                &final_text,
                                &messages,
                                instructions.as_deref(),
                                conversation.as_deref(),
                                session_id.as_deref(),
                            );
                        }
                        return;
                    }
                    emitted_items.push(message_item.clone());
                    let completed_response = build_completed_responses_payload_with_output(
                        &settings.model_name,
                        &response_id_for_stream,
                        created_at,
                        emitted_items.clone(),
                    );
                    if should_store {
                        let conversation_history = build_stored_conversation_history(
                            &messages,
                            &final_text,
                            instructions.as_deref(),
                        );
                        store_response_snapshot(
                            &state_for_stream,
                            &completed_response,
                            &conversation_history,
                            instructions.as_deref(),
                            conversation.as_deref(),
                            session_id.as_deref(),
                        );
                    }
                    let chunk = format_sse_event(
                        "response.completed",
                        &mut sequence_number,
                        json!({
                            "type": "response.completed",
                            "response": completed_response,
                        }),
                    );
                    let _ = send_sse_chunk(&body_tx, &interrupt_requested, chunk).await;
                    return;
                }
                LiveResponseEvent::Failed(error) => {
                    if should_store {
                        persist_stream_response_snapshot(
                            &state_for_stream,
                            &settings.model_name,
                            &response_id_for_stream,
                            created_at,
                            "failed",
                            emitted_items.clone(),
                            Some(final_text.as_str()),
                            Some(error.as_str()),
                            &messages,
                            instructions.as_deref(),
                            conversation.as_deref(),
                            session_id.as_deref(),
                        );
                    }
                    let chunk = format_sse_event(
                        "response.failed",
                        &mut sequence_number,
                        json!({
                            "type": "response.failed",
                            "response": {
                                "id": response_id_for_stream,
                                "object": "response",
                                "created_at": created_at,
                                "status": "failed",
                                "model": settings.model_name,
                                "output": emitted_items,
                                "error": { "message": error, "type": "server_error" },
                            }
                        }),
                    );
                    let _ = send_sse_chunk(&body_tx, &interrupt_requested, chunk).await;
                    return;
                }
            }
        }
    });

    let mut response = Response::new(Body::from_stream(
        ReceiverStream::new(body_rx).map(Ok::<_, io::Error>),
    ));
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
    if let Some(session_id) = resolved.session_id.as_deref()
        && let Ok(value) = HeaderValue::from_str(session_id)
    {
        response.headers_mut().insert("X-Hermes-Session-Id", value);
    }
    apply_cors_headers(response.headers_mut(), &state.settings, origin.as_ref());
    response
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

fn build_completed_responses_payload_with_output(
    model_name: &str,
    response_id: &str,
    created_at: u64,
    output: Vec<Value>,
) -> Value {
    build_responses_payload_with_status_and_output(
        model_name,
        response_id,
        created_at,
        "completed",
        output,
        None,
    )
}

fn build_responses_payload_with_status_and_output(
    model_name: &str,
    response_id: &str,
    created_at: u64,
    status: &str,
    output: Vec<Value>,
    error: Option<Value>,
) -> Value {
    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model_name,
        "output": output,
        "error": error,
    })
}

fn append_stream_assistant_message(output: &mut Vec<Value>, assistant_text: Option<&str>) {
    if let Some(text) = assistant_text
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        output.push(json!({
            "id": format!("msg-rs-{:x}", unix_ts_nanos()),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        }));
    }
}

fn persist_stream_response_snapshot(
    state: &NativeApiServerState,
    model_name: &str,
    response_id: &str,
    created_at: u64,
    status: &str,
    output: Vec<Value>,
    assistant_text: Option<&str>,
    error_message: Option<&str>,
    messages: &[Value],
    instructions: Option<&str>,
    conversation: Option<&str>,
    session_id: Option<&str>,
) {
    let mut snapshot_output = output;
    append_stream_assistant_message(&mut snapshot_output, assistant_text);
    let response_body = build_responses_payload_with_status_and_output(
        model_name,
        response_id,
        created_at,
        status,
        snapshot_output,
        error_message.map(|message| {
            json!({
                "message": message,
                "type": "server_error",
            })
        }),
    );
    let conversation_history =
        build_stored_conversation_history_with_assistant(messages, assistant_text, instructions);
    store_response_snapshot(
        state,
        &response_body,
        &conversation_history,
        instructions,
        conversation,
        session_id,
    );
}

fn persist_incomplete_stream_response_snapshot(
    state: &NativeApiServerState,
    model_name: &str,
    response_id: &str,
    created_at: u64,
    output: &[Value],
    assistant_text: &str,
    messages: &[Value],
    instructions: Option<&str>,
    conversation: Option<&str>,
    session_id: Option<&str>,
) {
    persist_stream_response_snapshot(
        state,
        model_name,
        response_id,
        created_at,
        "incomplete",
        output.to_vec(),
        Some(assistant_text),
        None,
        messages,
        instructions,
        conversation,
        session_id,
    );
}

fn store_response_snapshot(
    state: &NativeApiServerState,
    response_body: &Value,
    conversation_history: &[Value],
    instructions: Option<&str>,
    conversation: Option<&str>,
    session_id: Option<&str>,
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
            instructions: instructions
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned),
            session_id: session_id
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned),
        },
    );
    if let Some(conversation) = conversation
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        store
            .conversations
            .insert(conversation.to_string(), response_id.to_string());
    }
    while store.order.len() > MAX_STORED_RESPONSES {
        if let Some(oldest) = store.order.pop_front() {
            remove_stored_response(&mut store, &oldest);
        }
    }
}

fn remove_stored_response(store: &mut ResponseStore, response_id: &str) -> bool {
    store.order.retain(|id| id != response_id);
    let removed = store.responses.remove(response_id).is_some();
    if removed {
        store
            .conversations
            .retain(|_, current| current != response_id);
    }
    removed
}

fn load_stored_response(
    state: &NativeApiServerState,
    response_id: &str,
) -> Result<Option<StoredResponse>, ResponsesMessagesError> {
    let normalized_response_id = response_id.trim();
    if normalized_response_id.is_empty() {
        return Ok(None);
    }
    let stored = state
        .response_store
        .lock()
        .ok()
        .and_then(|store| store.responses.get(normalized_response_id).cloned());
    match stored {
        Some(entry) => Ok(Some(entry)),
        None => Err(ResponsesMessagesError::NotFound(
            normalized_response_id.to_string(),
        )),
    }
}

fn normalize_message_history(history: &[Value]) -> Result<Vec<Value>, ResponsesMessagesError> {
    let mut normalized = Vec::with_capacity(history.len());
    for (index, entry) in history.iter().enumerate() {
        let Some(object) = entry.as_object() else {
            return Err(ResponsesMessagesError::BadRequest(format!(
                "conversation_history[{index}] must be an object"
            )));
        };
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ResponsesMessagesError::BadRequest(format!(
                    "conversation_history[{index}] must include a non-empty role"
                ))
            })?;
        if !object.contains_key("content") {
            return Err(ResponsesMessagesError::BadRequest(format!(
                "conversation_history[{index}] must include content"
            )));
        }
        normalized.push(json!({
            "role": role,
            "content": object.get("content").cloned().unwrap_or(Value::Null),
        }));
    }
    Ok(normalized)
}

fn initialize_run(
    state: &NativeApiServerState,
    run_id: &str,
    event_name: &str,
    mut status: Value,
    interrupt_requested: Arc<AtomicBool>,
) {
    let Some(status_object) = status.as_object_mut() else {
        return;
    };
    status_object.insert("last_event".to_string(), json!(event_name));
    status_object.insert("updated_at".to_string(), json!(unix_ts_secs()));

    let mut event = status.clone();
    let Some(event_object) = event.as_object_mut() else {
        return;
    };
    event_object.insert("event".to_string(), json!(event_name));
    event_object.insert("timestamp".to_string(), json!(unix_ts_secs()));

    let (broadcaster, _) = broadcast::channel(32);
    let Ok(mut store) = state.run_store.lock() else {
        return;
    };
    store.runs.insert(
        run_id.to_string(),
        StoredRun {
            status,
            events: vec![event],
            broadcaster,
            interrupt_requested,
        },
    );
}

fn record_run_status(
    state: &NativeApiServerState,
    run_id: &str,
    event_name: &str,
    mut status: Value,
) {
    let Some(status_object) = status.as_object_mut() else {
        return;
    };
    status_object.insert("last_event".to_string(), json!(event_name));
    status_object.insert("updated_at".to_string(), json!(unix_ts_secs()));

    let mut event = status.clone();
    let Some(event_object) = event.as_object_mut() else {
        return;
    };
    event_object.insert("event".to_string(), json!(event_name));
    event_object.insert("timestamp".to_string(), json!(unix_ts_secs()));

    let Ok(mut store) = state.run_store.lock() else {
        return;
    };
    if let Some(run) = store.runs.get_mut(run_id) {
        run.status = status;
        run.events.push(event.clone());
        let _ = run.broadcaster.send(event);
        return;
    }

    let (broadcaster, _) = broadcast::channel(32);
    store.runs.insert(
        run_id.to_string(),
        StoredRun {
            status,
            events: vec![event],
            broadcaster,
            interrupt_requested: Arc::new(AtomicBool::new(false)),
        },
    );
}

fn append_progress_run_event(
    state: &NativeApiServerState,
    run_id: &str,
    event: AgentProgressEvent,
) {
    let (event_name, payload) = match event {
        AgentProgressEvent::MessageDelta { delta } => (
            "message.delta",
            json!({
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": unix_ts_secs(),
                "delta": delta,
            }),
        ),
        AgentProgressEvent::ToolStarted {
            tool_call_id,
            tool_name,
            arguments,
        } => (
            "tool.started",
            json!({
                "event": "tool.started",
                "run_id": run_id,
                "timestamp": unix_ts_secs(),
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "preview": arguments,
            }),
        ),
        AgentProgressEvent::ToolCompleted {
            tool_call_id,
            tool_name,
            arguments: _,
            result: _,
            duration_secs,
            is_error,
        } => (
            "tool.completed",
            json!({
                "event": "tool.completed",
                "run_id": run_id,
                "timestamp": unix_ts_secs(),
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "duration": (duration_secs * 1000.0).round() / 1000.0,
                "error": is_error,
            }),
        ),
        AgentProgressEvent::ReasoningAvailable { text } => (
            "reasoning.available",
            json!({
                "event": "reasoning.available",
                "run_id": run_id,
                "timestamp": unix_ts_secs(),
                "text": text,
            }),
        ),
    };

    let Ok(mut store) = state.run_store.lock() else {
        return;
    };
    let Some(run) = store.runs.get_mut(run_id) else {
        return;
    };
    if let Some(status_object) = run.status.as_object_mut() {
        status_object.insert("last_event".to_string(), json!(event_name));
        status_object.insert("updated_at".to_string(), json!(unix_ts_secs()));
    }
    run.events.push(payload.clone());
    let _ = run.broadcaster.send(payload);
}

fn format_run_event_chunk(event: &Value) -> String {
    match event.get("event").and_then(Value::as_str) {
        Some(event_name) => format!("event: {event_name}\ndata: {event}\n\n"),
        None => format!("data: {event}\n\n"),
    }
}

fn run_event_is_terminal(event: &Value) -> bool {
    event
        .get("event")
        .and_then(Value::as_str)
        .is_some_and(|event_name| {
            matches!(event_name, "run.completed" | "run.failed" | "run.cancelled")
        })
}

fn run_status_is_terminal(status: &Value) -> bool {
    status
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "completed" | "failed" | "cancelled"))
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

    fn mock_model_server_with_delay(
        response_body: String,
        delay: std::time::Duration,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let start = std::time::Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(stream) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if start.elapsed() > std::time::Duration::from_secs(2) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                }
            };
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
            std::thread::sleep(delay);
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

    fn extract_sse_event_payload(body: &str, event_name: &str) -> Value {
        let expected_event = format!("event: {event_name}");
        for block in body.split("\n\n") {
            let mut saw_event = false;
            for line in block.lines() {
                if line == expected_event {
                    saw_event = true;
                    continue;
                }
                if saw_event && let Some(payload) = line.strip_prefix("data: ") {
                    return serde_json::from_str(payload).unwrap();
                }
            }
        }
        panic!("missing SSE event: {event_name}");
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
    fn native_api_server_health_detailed_reads_gateway_state() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (_temp, context, loaded) =
            temp_context("platforms:\n  api_server:\n    enabled: true\n");
        fs::write(
            context.hermes_home().join("gateway_state.json"),
            r#"{"gateway_state":"draining","exit_reason":"restart","active_agents":2,"platforms":{"telegram":{"state":"fatal","error_message":"boom"}},"updated_at":"2026-05-25T12:00:00Z"}"#,
        )
        .unwrap();
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
            .get(format!("http://{addr}/health/detailed"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(response["status"], json!("ok"));
        assert_eq!(response["platform"], json!("hermes-agent"));
        assert_eq!(response["gateway_state"], json!("draining"));
        assert_eq!(response["exit_reason"], json!("restart"));
        assert_eq!(response["active_agents"], json!(2));
        assert_eq!(
            response["platforms"]["telegram"]["error_message"],
            json!("boom")
        );
        assert_eq!(response["updated_at"], json!("2026-05-25T12:00:00Z"));
        assert!(response["pid"].as_u64().is_some());

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
    }

    #[test]
    fn native_api_server_chat_completions_can_resume_explicit_session() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "First turn."
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Second turn."
                        }
                    }]
                })
                .to_string(),
            ],
            |requests| {
                assert_eq!(requests.len(), 2);
                let second_body = requests[1].split("\r\n\r\n").nth(1).unwrap_or_default();
                let payload: Value = serde_json::from_str(second_body).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("hello")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("assistant")
                        && message["content"] == json!("First turn.")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("continue")
                }));
            },
        );
        let config_text = format!(
            "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: {base_url}\n  api_key: test-key\nplatforms:\n  api_server:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 0\n      key: native-key\n"
        );
        let (_temp, context, loaded) = temp_context(&config_text);
        let settings = NativeApiServerSettings {
            host: "127.0.0.1".to_string(),
            port: 0,
            api_key: "native-key".to_string(),
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

        let first = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .header(reqwest::header::AUTHORIZATION, "Bearer native-key")
            .header("X-Hermes-Session-Id", "sess-native-1")
            .json(&json!({
                "messages": [
                    { "role": "system", "content": "Be concise." },
                    { "role": "user", "content": "hello" }
                ]
            }))
            .send()
            .unwrap();
        assert_eq!(
            first.headers().get("X-Hermes-Session-Id").unwrap(),
            "sess-native-1"
        );
        let first_body: Value = first.json().unwrap();
        assert_eq!(
            first_body["choices"][0]["message"]["content"],
            json!("First turn.")
        );

        let second = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .header(reqwest::header::AUTHORIZATION, "Bearer native-key")
            .header("X-Hermes-Session-Id", "sess-native-1")
            .json(&json!({
                "messages": [
                    { "role": "user", "content": "continue" }
                ]
            }))
            .send()
            .unwrap();
        assert_eq!(
            second.headers().get("X-Hermes-Session-Id").unwrap(),
            "sess-native-1"
        );
        let second_body: Value = second.json().unwrap();
        assert_eq!(
            second_body["choices"][0]["message"]["content"],
            json!("Second turn.")
        );

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_chat_completions_derive_stable_session_id() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "First derived turn."
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Second derived turn."
                        }
                    }]
                })
                .to_string(),
            ],
            |requests| {
                assert_eq!(requests.len(), 2);
                let second_body = requests[1].split("\r\n\r\n").nth(1).unwrap_or_default();
                let payload: Value = serde_json::from_str(second_body).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("hello")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("assistant")
                        && message["content"] == json!("First derived turn.")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("continue")
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

        let first = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [
                    { "role": "system", "content": "Be concise." },
                    { "role": "user", "content": "hello" }
                ]
            }))
            .send()
            .unwrap();
        let first_session = first
            .headers()
            .get("X-Hermes-Session-Id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(first_session.starts_with("api-"));
        let first_body: Value = first.json().unwrap();
        assert_eq!(
            first_body["choices"][0]["message"]["content"],
            json!("First derived turn.")
        );

        let second = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [
                    { "role": "system", "content": "Be concise." },
                    { "role": "user", "content": "hello" },
                    { "role": "assistant", "content": "First derived turn." },
                    { "role": "user", "content": "continue" }
                ]
            }))
            .send()
            .unwrap();
        let second_session = second
            .headers()
            .get("X-Hermes-Session-Id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(second_session, first_session);
        let second_body: Value = second.json().unwrap();
        assert_eq!(
            second_body["choices"][0]["message"]["content"],
            json!("Second derived turn.")
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
    fn native_api_server_chat_completions_streams_tool_progress() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_todo_chat_stream",
                                "type": "function",
                                "function": {
                                    "name": "todo",
                                    "arguments": "{}"
                                }
                            }]
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "content": "chat tool stream hello"
                        }
                    }]
                })
                .to_string(),
            ],
            |_| {},
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
        let body = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": "say hi"
                }],
                "stream": true,
            }))
            .send()
            .unwrap()
            .text()
            .unwrap();
        assert!(body.contains("event: hermes.tool.progress"));
        assert!(body.contains("\"toolCallId\":\"call_todo_chat_stream\""));
        assert!(body.contains("\"status\":\"running\""));
        assert!(body.contains("\"status\":\"completed\""));
        assert!(body.contains("\"content\":\"chat tool stream hello\""));
        assert!(body.contains("data: [DONE]"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_chat_completions_streams_keepalive() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "keepalive hello"
                }
            }]
        })
        .to_string();
        let (base_url, join) =
            mock_model_server_with_delay(response_body, std::time::Duration::from_millis(120));
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
        let body = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": "say hi"
                }],
                "stream": true,
            }))
            .send()
            .unwrap()
            .text()
            .unwrap();
        assert!(body.contains(": keepalive"));
        assert!(body.contains("\"content\":\"keepalive hello\""));
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
    fn native_api_server_responses_streams_keepalive() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": "responses keepalive hello"
                }
            }]
        })
        .to_string();
        let (base_url, join) =
            mock_model_server_with_delay(response_body, std::time::Duration::from_millis(120));
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
        let body = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
                "stream": true,
            }))
            .send()
            .unwrap()
            .text()
            .unwrap();
        assert!(body.contains(": keepalive"));
        assert!(body.contains("\"delta\":\"responses keepalive hello\""));
        assert!(body.contains("event: response.completed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_responses_stream_tool_events() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_todo_stream",
                                "type": "function",
                                "function": {
                                    "name": "todo",
                                    "arguments": "{}"
                                }
                            }]
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "content": "tool stream hello"
                        }
                    }]
                })
                .to_string(),
            ],
            |_| {},
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
        let response = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
                "stream": true,
            }))
            .send()
            .unwrap();
        let body = response.text().unwrap();
        assert!(body.contains("\"type\":\"function_call\""));
        assert!(body.contains("\"call_id\":\"call_todo_stream\""));
        assert!(body.contains("\"type\":\"function_call_output\""));
        assert!(body.contains("event: response.output_item.done"));
        assert!(body.contains("\"text\":\"tool stream hello\""));
        assert!(body.contains("event: response.completed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_responses_stream_tool_events_with_session_id() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_todo_stream_session",
                                "type": "function",
                                "function": {
                                    "name": "todo",
                                    "arguments": "{}"
                                }
                            }]
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "content": "session tool stream hello"
                        }
                    }]
                })
                .to_string(),
            ],
            |_| {},
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
        let response = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
                "stream": true,
                "session_id": "sess-response-stream-1",
            }))
            .send()
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get("X-Hermes-Session-Id")
                .unwrap()
                .to_str()
                .unwrap(),
            "sess-response-stream-1"
        );
        let body = response.text().unwrap();
        assert!(body.contains("\"type\":\"function_call\""));
        assert!(body.contains("\"call_id\":\"call_todo_stream_session\""));
        assert!(body.contains("\"type\":\"function_call_output\""));
        assert!(body.contains("event: response.output_item.done"));
        assert!(body.contains("\"text\":\"session tool stream hello\""));
        assert!(body.contains("event: response.completed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_failed_stream_persists_response_snapshot() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "choices": [{
                "message": {
                    "role": "assistant"
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
        let body = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "say hi",
                "stream": true,
            }))
            .send()
            .unwrap()
            .text()
            .unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.failed"));
        let created = extract_sse_event_payload(&body, "response.created");
        let response_id = created["response"]["id"].as_str().unwrap();

        let stored: Value = client
            .get(format!("http://{addr}/v1/responses/{response_id}"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(stored["status"], json!("failed"));
        assert_eq!(stored["id"], json!(response_id));

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
        assert_eq!(capabilities["platform"], json!("hermes-agent"));
        assert_eq!(capabilities["auth"]["type"], json!("bearer"));
        assert_eq!(capabilities["auth"]["required"], json!(false));
        assert_eq!(capabilities["features"]["stored_responses"], json!(true));
        assert_eq!(
            capabilities["features"]["responses_conversation_aliases"],
            json!(true)
        );
        assert_eq!(
            capabilities["features"]["responses_conversation_history"],
            json!(true)
        );
        assert_eq!(
            capabilities["features"]["responses_instructions"],
            json!(true)
        );
        assert_eq!(capabilities["features"]["runs_api"], json!(true));
        assert_eq!(capabilities["features"]["run_submission"], json!(true));
        assert_eq!(capabilities["features"]["run_status"], json!(true));
        assert_eq!(capabilities["features"]["run_events"], json!(true));
        assert_eq!(capabilities["features"]["run_events_sse"], json!(true));
        assert_eq!(capabilities["features"]["run_stop"], json!(true));
        assert_eq!(
            capabilities["features"]["runs_conversation_history"],
            json!(true)
        );
        assert_eq!(capabilities["features"]["runs_instructions"], json!(true));
        assert_eq!(
            capabilities["features"]["tool_progress_events"],
            json!(true)
        );
        assert_eq!(capabilities["features"]["cors"], json!(false));
        assert_eq!(
            capabilities["endpoints"]["health"]["path"],
            json!("/health")
        );
        assert_eq!(
            capabilities["endpoints"]["response_get"]["path"],
            json!("/v1/responses/{response_id}")
        );
        assert_eq!(
            capabilities["endpoints"]["run_status"]["path"],
            json!("/v1/runs/{run_id}")
        );
        assert_eq!(
            capabilities["endpoints"]["run_events"]["path"],
            json!("/v1/runs/{run_id}/events")
        );
        assert_eq!(
            capabilities["endpoints"]["run_stop"]["path"],
            json!("/v1/runs/{run_id}/stop")
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
    fn native_api_server_runs_accept_conversation_history_and_instructions() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "id": "chatcmpl-run-history",
                    "choices": [{
                        "message": {
                            "content": "history hello"
                        }
                    }]
                })
                .to_string(),
            ],
            |requests| {
                assert_eq!(requests.len(), 1);
                let body = requests[0].split("\r\n\r\n").nth(1).unwrap_or_default();
                let payload: Value = serde_json::from_str(body).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("system")
                        && message["content"] == json!("Follow policy.")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("earlier user")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("assistant")
                        && message["content"] == json!("earlier assistant")
                }));
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("user") && message["content"] == json!("current")
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

        let started: Value = client
            .post(format!("http://{addr}/v1/runs"))
            .json(&json!({
                "input": "current",
                "instructions": "Follow policy.",
                "conversation_history": [
                    { "role": "user", "content": "earlier user" },
                    { "role": "assistant", "content": "earlier assistant" }
                ],
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
        assert_eq!(status["output"], json!("history hello"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_run_events_stream_lifecycle_updates() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-run-events",
            "choices": [{
                "message": {
                    "content": "run stream hello"
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
        let run_id = started["run_id"].as_str().unwrap();

        let response = client
            .get(format!("http://{addr}/v1/runs/{run_id}/events"))
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
        assert!(body.contains("event: run.queued"));
        assert!(body.contains("event: run.running"));
        assert!(body.contains("event: run.completed"));
        assert!(body.contains("\"output\":\"run stream hello\""));
        assert!(body.contains(": stream closed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_run_events_include_tool_and_message_updates() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_todo_1",
                                "type": "function",
                                "function": {
                                    "name": "todo",
                                    "arguments": "{}"
                                }
                            }]
                        }
                    }]
                })
                .to_string(),
                json!({
                    "choices": [{
                        "message": {
                            "content": "tool-backed hello"
                        }
                    }]
                })
                .to_string(),
            ],
            |_| {},
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

        let started: Value = client
            .post(format!("http://{addr}/v1/runs"))
            .json(&json!({
                "input": "say hi",
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        let run_id = started["run_id"].as_str().unwrap();

        let response = client
            .get(format!("http://{addr}/v1/runs/{run_id}/events"))
            .send()
            .unwrap();
        let body = response.text().unwrap();
        assert!(body.contains("event: tool.started"));
        assert!(body.contains("\"tool\":\"todo\""));
        assert!(body.contains("event: tool.completed"));
        assert!(body.contains("\"error\":false"));
        assert!(body.contains("event: message.delta"));
        assert!(body.contains("\"delta\":\"tool-backed hello\""));
        assert!(body.contains("event: run.completed"));

        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        join.join().unwrap();
    }

    #[test]
    fn native_api_server_runs_can_be_stopped() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let response_body = json!({
            "id": "chatcmpl-run-stop",
            "choices": [{
                "message": {
                    "content": "run stop hello"
                }
            }]
        })
        .to_string();
        let (base_url, join) =
            mock_model_server_with_delay(response_body, std::time::Duration::from_millis(250));
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
        let run_id = started["run_id"].as_str().unwrap().to_string();

        let mut running_status = Value::Null;
        for _ in 0..40 {
            running_status = client
                .get(format!("http://{addr}/v1/runs/{run_id}"))
                .send()
                .unwrap()
                .json()
                .unwrap();
            if running_status["status"] == json!("running") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(running_status["status"], json!("running"));

        let stopping: Value = client
            .post(format!("http://{addr}/v1/runs/{run_id}/stop"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(stopping["status"], json!("stopping"));

        let mut status = Value::Null;
        for _ in 0..60 {
            status = client
                .get(format!("http://{addr}/v1/runs/{run_id}"))
                .send()
                .unwrap()
                .json()
                .unwrap();
            if status["status"] == json!("cancelled") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert_eq!(status["status"], json!("cancelled"));
        assert_eq!(status["last_event"], json!("run.cancelled"));

        let events = client
            .get(format!("http://{addr}/v1/runs/{run_id}/events"))
            .send()
            .unwrap()
            .text()
            .unwrap();
        assert!(events.contains("event: run.stopping"));
        assert!(events.contains("event: run.cancelled"));

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

    #[test]
    fn native_api_server_responses_support_conversation_alias_and_instruction_carryover() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (base_url, join) = mock_model_server_sequence(
            vec![
                json!({
                    "id": "chatcmpl-conversation-first",
                    "choices": [{
                        "message": {
                            "content": "first reply"
                        }
                    }]
                })
                .to_string(),
                json!({
                    "id": "chatcmpl-conversation-second",
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
                let second_body = requests[1].split("\r\n\r\n").nth(1).unwrap_or_default();
                let payload: Value = serde_json::from_str(second_body).unwrap();
                let messages = payload["messages"].as_array().unwrap();
                assert!(messages.iter().any(|message| {
                    message["role"] == json!("system") && message["content"] == json!("Be terse.")
                }));
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
                "conversation": "demo",
                "instructions": "Be terse.",
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(
            first_response["output"][0]["content"][0]["text"],
            json!("first reply")
        );

        let second_response: Value = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "input": "second",
                "conversation": "demo",
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
