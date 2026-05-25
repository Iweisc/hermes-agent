use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hermes_core::{
    DelegateExecutor, HermesContext, LoadedConfig, ModelOverrides, ToolRuntime, dispatch_tool,
    get_tool_definitions,
};
use hmac::{Hmac, Mac};
use regex::Regex;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};
use sha2::Sha256;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use crate::gateway_cmd::GatewayRunArgs;

const DEFAULT_WEBHOOK_HOST: &str = "0.0.0.0";
const DEFAULT_WEBHOOK_PORT: u16 = 8644;
const DEFAULT_RATE_LIMIT: usize = 30;
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;
const INSECURE_NO_AUTH: &str = "INSECURE_NO_AUTH";
const DYNAMIC_ROUTES_FILENAME: &str = "webhook_subscriptions.json";

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug)]
pub(crate) struct NativeWebhookSettings {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) global_secret: String,
    pub(crate) rate_limit: usize,
    pub(crate) max_body_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeWebhookRoute {
    pub(crate) events: Vec<String>,
    pub(crate) secret: String,
    pub(crate) prompt: String,
    pub(crate) deliver: String,
    pub(crate) deliver_only: bool,
    pub(crate) deliver_extra: JsonMap<String, JsonValue>,
    pub(crate) skills: Vec<String>,
}

#[derive(Default)]
struct WebhookRuntimeState {
    rate_counts: BTreeMap<String, Vec<f64>>,
    seen_deliveries: BTreeMap<String, f64>,
}

#[derive(Clone)]
pub(crate) struct NativeWebhookState {
    context: HermesContext,
    loaded: LoadedConfig,
    pub(crate) settings: NativeWebhookSettings,
    static_routes: BTreeMap<String, NativeWebhookRoute>,
    dynamic_routes_path: PathBuf,
    runtime_state: Arc<Mutex<WebhookRuntimeState>>,
}

pub(crate) fn maybe_run_native_webhook_server(
    context: &HermesContext,
    args: &GatewayRunArgs,
) -> Result<bool, Box<dyn Error>> {
    let loaded = context.load_config_document()?;
    if !webhook_enabled(&loaded) || webhook_has_other_platforms_enabled(&loaded) {
        return Ok(false);
    }
    let settings = load_webhook_settings(&loaded)?;
    let static_routes = load_static_routes(&loaded, &settings.global_secret)?;
    let dynamic_routes_path = context.hermes_home().join(DYNAMIC_ROUTES_FILENAME);
    let dynamic_routes = load_dynamic_routes(&dynamic_routes_path, &settings.global_secret)?;
    if !routes_support_native(static_routes.values().chain(dynamic_routes.values())) {
        return Ok(false);
    }
    run_native_webhook_server(
        context.clone(),
        loaded,
        settings,
        static_routes,
        dynamic_routes_path,
        args,
    )?;
    Ok(true)
}

pub(crate) fn load_native_webhook_state(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<Option<NativeWebhookState>, Box<dyn Error>> {
    if !webhook_enabled(loaded) {
        return Ok(None);
    }
    let settings = load_webhook_settings(loaded)?;
    let static_routes = load_static_routes(loaded, &settings.global_secret)?;
    let dynamic_routes_path = context.hermes_home().join(DYNAMIC_ROUTES_FILENAME);
    let dynamic_routes = load_dynamic_routes(&dynamic_routes_path, &settings.global_secret)?;
    if !routes_support_native(static_routes.values().chain(dynamic_routes.values())) {
        return Ok(None);
    }
    Ok(Some(NativeWebhookState {
        context: context.clone(),
        loaded: loaded.clone(),
        settings,
        static_routes,
        dynamic_routes_path,
        runtime_state: Arc::new(Mutex::new(WebhookRuntimeState::default())),
    }))
}

fn run_native_webhook_server(
    context: HermesContext,
    loaded: LoadedConfig,
    settings: NativeWebhookSettings,
    static_routes: BTreeMap<String, NativeWebhookRoute>,
    dynamic_routes_path: PathBuf,
    _args: &GatewayRunArgs,
) -> Result<(), Box<dyn Error>> {
    let bind_ip: IpAddr = settings
        .host
        .parse()
        .map_err(|_| "WEBHOOK_HOST must be a valid IP address for native Rust runtime")?;
    let state = NativeWebhookState {
        context,
        loaded,
        settings: settings.clone(),
        static_routes,
        dynamic_routes_path,
        runtime_state: Arc::new(Mutex::new(WebhookRuntimeState::default())),
    };
    let runtime = Runtime::new()?;
    runtime.block_on(async move {
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::new(bind_ip, settings.port)).await?;
        println!(
            "Native webhook server listening on http://{}:{}",
            settings.host, settings.port
        );
        serve_native_webhook_server(listener, state, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
        Ok::<(), Box<dyn Error>>(())
    })?;
    Ok(())
}

pub(crate) async fn serve_native_webhook_server<F>(
    listener: tokio::net::TcpListener,
    state: NativeWebhookState,
    shutdown: F,
) -> Result<(), Box<dyn Error>>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/webhooks/{route_name}", post(handle_webhook))
        .with_state(Arc::new(state));
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn handle_health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "platform": "webhook",
    }))
}

async fn handle_webhook(
    State(state): State<Arc<NativeWebhookState>>,
    Path(route_name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let routes = match merged_routes(&state) {
        Ok(routes) => routes,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": error.to_string() })),
            )
                .into_response();
        }
    };
    let Some(route) = routes.get(&route_name).cloned() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Unknown route: {route_name}") })),
        )
            .into_response();
    };

    if body.len() > state.settings.max_body_bytes {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": "Payload too large" })),
        )
            .into_response();
    }

    if route.secret != INSECURE_NO_AUTH && !validate_signature(&headers, &body, &route.secret) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "Invalid signature" })),
        )
            .into_response();
    }

    let payload = match parse_webhook_body(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": error }))).into_response();
        }
    };

    let event_type = webhook_event_type(&headers, &payload);
    if !route.events.is_empty() && !route.events.iter().any(|event| event == &event_type) {
        return Json(json!({
            "status": "ignored",
            "event": event_type,
        }))
        .into_response();
    }

    let delivery_id = headers
        .get("X-GitHub-Delivery")
        .and_then(|value| value.to_str().ok())
        .or_else(|| {
            headers
                .get("X-Request-ID")
                .and_then(|value| value.to_str().ok())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "delivery")
        .to_string()
        + &format!("-{:x}", unix_ts_nanos());

    if let Some(response) =
        enforce_rate_limit_and_idempotency(&state, &route_name, &delivery_id).await
    {
        return response;
    }

    let prompt = render_prompt(&route.prompt, &payload, &event_type, &route_name);
    let rendered_extra = render_delivery_extra(&route.deliver_extra, &payload);

    if route.deliver_only {
        match deliver_content_async(
            Arc::clone(&state),
            route.deliver.clone(),
            rendered_extra.clone(),
            prompt.clone(),
        )
        .await
        {
            Ok(()) => {
                return Json(json!({
                    "status": "delivered",
                    "route": route_name,
                    "target": route.deliver,
                    "delivery_id": delivery_id,
                }))
                .into_response();
            }
            Err(error) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "status": "error", "error": error, "delivery_id": delivery_id })),
                )
                    .into_response();
            }
        }
    }

    let state_clone = Arc::clone(&state);
    let route_name_for_task = route_name.clone();
    tokio::spawn(async move {
        let result = run_webhook_agent_and_deliver(
            state_clone,
            route_name_for_task.clone(),
            route.deliver.clone(),
            rendered_extra,
            prompt,
        )
        .await;
        if let Err(error) = result {
            eprintln!(
                "[webhook] route={} delivery failed: {}",
                route_name_for_task, error
            );
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "accepted",
            "route": route_name,
            "event": event_type,
            "delivery_id": delivery_id,
        })),
    )
        .into_response()
}

async fn run_webhook_agent_and_deliver(
    state: Arc<NativeWebhookState>,
    route_name: String,
    deliver: String,
    deliver_extra: JsonMap<String, JsonValue>,
    prompt: String,
) -> Result<(), String> {
    let state_for_agent = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_agent_for_prompt(&state_for_agent, &prompt))
            .await
            .map_err(|error| error.to_string())??;
    let response = result.final_response;
    deliver_content_async(state, deliver, deliver_extra, response)
        .await
        .map_err(|error| format!("route={route_name}: {error}"))
}

fn run_agent_for_prompt(
    state: &NativeWebhookState,
    prompt: &str,
) -> Result<hermes_core::AgentTurnResult, String> {
    let enabled_toolsets = state.loaded.config.toolsets.clone();
    let delegate = DelegateExecutor::new(
        state.context.clone(),
        state.loaded.clone(),
        "rust-webhook",
        enabled_toolsets.clone(),
        ModelOverrides::default(),
        webhook_workdir(&state.loaded),
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
    state
        .context
        .run_chat_completions_turn(
            &state.loaded,
            prompt,
            &runtime,
            Some(&enabled_toolsets),
            &ModelOverrides::default(),
            None,
            None,
        )
        .map_err(|error| error.to_string())
}

async fn deliver_content_async(
    state: Arc<NativeWebhookState>,
    deliver: String,
    deliver_extra: JsonMap<String, JsonValue>,
    content: String,
) -> Result<(), String> {
    tokio::task::spawn_blocking(move || {
        deliver_content_blocking(&state, &deliver, &deliver_extra, &content)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn deliver_content_blocking(
    state: &NativeWebhookState,
    deliver: &str,
    deliver_extra: &JsonMap<String, JsonValue>,
    content: &str,
) -> Result<(), String> {
    if deliver == "log" {
        println!("[webhook] {content}");
        return Ok(());
    }
    if deliver == "github_comment" {
        return deliver_github_comment(deliver_extra, content);
    }
    if !send_message_supported_platform(deliver) {
        return Err(format!("unsupported native deliver target: {deliver}"));
    }
    let target = if let Some(chat_id) = deliver_extra.get("chat_id").and_then(JsonValue::as_str) {
        format!("{deliver}:{}", chat_id.trim())
    } else {
        deliver.to_string()
    };
    let runtime = ToolRuntime::default().with_hermes_home(state.context.hermes_home());
    let output = dispatch_tool(
        "send_message",
        json!({
            "target": target,
            "message": content,
        }),
        &runtime,
    );
    let parsed: JsonValue = serde_json::from_str(&output)
        .map_err(|error| format!("send_message decode failed: {error}"))?;
    if parsed.get("success").and_then(JsonValue::as_bool) == Some(true) {
        return Ok(());
    }
    if let Some(error) = parsed.get("error").and_then(JsonValue::as_str) {
        return Err(error.to_string());
    }
    Err(output)
}

fn deliver_github_comment(
    deliver_extra: &JsonMap<String, JsonValue>,
    content: &str,
) -> Result<(), String> {
    let repo = deliver_extra
        .get("repo")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing repo".to_string())?;
    let pr_number = deliver_extra
        .get("pr_number")
        .and_then(JsonValue::as_i64)
        .map(|value| value.to_string())
        .or_else(|| {
            deliver_extra
                .get("pr_number")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
        .ok_or_else(|| "Missing pr_number".to_string())?;
    let output = Command::new("gh")
        .args([
            "pr", "comment", &pr_number, "--repo", repo, "--body", content,
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

async fn enforce_rate_limit_and_idempotency(
    state: &NativeWebhookState,
    route_name: &str,
    delivery_id: &str,
) -> Option<Response> {
    let mut guard = state.runtime_state.lock().await;
    let now = unix_ts_secs_f64();
    guard.seen_deliveries.retain(|_, ts| now - *ts < 3600.0);
    if guard.seen_deliveries.contains_key(delivery_id) {
        return Some(
            Json(json!({
                "status": "duplicate",
                "delivery_id": delivery_id,
            }))
            .into_response(),
        );
    }
    let window = guard.rate_counts.entry(route_name.to_string()).or_default();
    window.retain(|ts| now - *ts < 60.0);
    if window.len() >= state.settings.rate_limit {
        return Some(
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({ "error": "Rate limit exceeded" })),
            )
                .into_response(),
        );
    }
    window.push(now);
    guard.seen_deliveries.insert(delivery_id.to_string(), now);
    None
}

fn merged_routes(
    state: &NativeWebhookState,
) -> Result<BTreeMap<String, NativeWebhookRoute>, Box<dyn Error>> {
    let dynamic = load_dynamic_routes(&state.dynamic_routes_path, &state.settings.global_secret)?;
    let mut merged = dynamic;
    for (name, route) in &state.static_routes {
        merged.insert(name.clone(), route.clone());
    }
    Ok(merged)
}

pub(crate) fn load_webhook_settings(
    loaded: &LoadedConfig,
) -> Result<NativeWebhookSettings, Box<dyn Error>> {
    let extra = loaded
        .cfg_get(&["platforms", "webhook", "extra"])
        .and_then(YamlValue::as_mapping);
    let host = env_string("WEBHOOK_HOST")
        .or_else(|| extra.and_then(|mapping| mapping_string(mapping, "host")))
        .unwrap_or_else(|| DEFAULT_WEBHOOK_HOST.to_string());
    let port = env_string("WEBHOOK_PORT")
        .and_then(|value| value.parse::<u16>().ok())
        .or_else(|| extra.and_then(|mapping| mapping_u16(mapping, "port")))
        .unwrap_or(DEFAULT_WEBHOOK_PORT);
    if port == 0 {
        return Err("WEBHOOK_PORT must be between 1 and 65535".into());
    }
    Ok(NativeWebhookSettings {
        host,
        port,
        global_secret: env_string("WEBHOOK_SECRET")
            .or_else(|| extra.and_then(|mapping| mapping_string(mapping, "secret")))
            .unwrap_or_default(),
        rate_limit: extra
            .and_then(|mapping| mapping_usize(mapping, "rate_limit"))
            .unwrap_or(DEFAULT_RATE_LIMIT),
        max_body_bytes: extra
            .and_then(|mapping| mapping_usize(mapping, "max_body_bytes"))
            .unwrap_or(DEFAULT_MAX_BODY_BYTES),
    })
}

pub(crate) fn load_static_routes(
    loaded: &LoadedConfig,
    global_secret: &str,
) -> Result<BTreeMap<String, NativeWebhookRoute>, Box<dyn Error>> {
    let routes = loaded
        .cfg_get(&["platforms", "webhook", "extra", "routes"])
        .cloned()
        .unwrap_or(YamlValue::Null);
    parse_routes_jsonish(
        &serde_json::to_value(routes).unwrap_or(JsonValue::Null),
        global_secret,
    )
}

pub(crate) fn load_dynamic_routes(
    path: &PathBuf,
    global_secret: &str,
) -> Result<BTreeMap<String, NativeWebhookRoute>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path)?;
    let parsed: JsonValue = serde_json::from_str(&raw).unwrap_or(JsonValue::Null);
    parse_routes_jsonish(&parsed, global_secret)
}

fn parse_routes_jsonish(
    value: &JsonValue,
    global_secret: &str,
) -> Result<BTreeMap<String, NativeWebhookRoute>, Box<dyn Error>> {
    let Some(object) = value.as_object() else {
        return Ok(BTreeMap::new());
    };
    let mut routes = BTreeMap::new();
    for (name, route) in object {
        let normalized = normalize_route_name(name)?;
        let Some(route_object) = route.as_object() else {
            continue;
        };
        let secret = route_object
            .get("secret")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(global_secret)
            .to_string();
        if secret.is_empty() {
            return Err(format!("webhook route '{normalized}' has no HMAC secret").into());
        }
        let deliver = route_object
            .get("deliver")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("log")
            .to_string();
        let deliver_only = route_object
            .get("deliver_only")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        if deliver_only && deliver == "log" {
            return Err(format!(
                "webhook route '{normalized}' has deliver_only=true but deliver=log"
            )
            .into());
        }
        routes.insert(
            normalized.clone(),
            NativeWebhookRoute {
                events: route_object
                    .get("events")
                    .and_then(JsonValue::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(JsonValue::as_str)
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(ToOwned::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                secret,
                prompt: route_object
                    .get("prompt")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_string(),
                deliver,
                deliver_only,
                deliver_extra: route_object
                    .get("deliver_extra")
                    .and_then(JsonValue::as_object)
                    .cloned()
                    .unwrap_or_default(),
                skills: route_object
                    .get("skills")
                    .and_then(JsonValue::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(JsonValue::as_str)
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(ToOwned::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            },
        );
    }
    Ok(routes)
}

pub(crate) fn routes_support_native<'a>(
    mut routes: impl Iterator<Item = &'a NativeWebhookRoute>,
) -> bool {
    routes.all(|route| {
        route.skills.is_empty()
            && (route.deliver == "log"
                || route.deliver == "github_comment"
                || send_message_supported_platform(&route.deliver))
    })
}

fn validate_signature(headers: &HeaderMap, body: &[u8], secret: &str) -> bool {
    if let Some(signature) = headers
        .get("X-Hub-Signature-256")
        .and_then(|value| value.to_str().ok())
    {
        return signature == compute_signature(secret, body);
    }
    if let Some(token) = headers
        .get("X-Gitlab-Token")
        .and_then(|value| value.to_str().ok())
    {
        return token == secret;
    }
    if let Some(signature) = headers
        .get("X-Webhook-Signature")
        .and_then(|value| value.to_str().ok())
    {
        return signature == compute_generic_signature(secret, body);
    }
    false
}

fn parse_webhook_body(body: &[u8]) -> Result<JsonValue, String> {
    if let Ok(value) = serde_json::from_slice::<JsonValue>(body) {
        return Ok(value);
    }
    let mut object = JsonMap::new();
    for (key, value) in url::form_urlencoded::parse(body) {
        object.insert(key.to_string(), JsonValue::String(value.to_string()));
    }
    if object.is_empty() {
        return Err("Cannot parse body".to_string());
    }
    Ok(JsonValue::Object(object))
}

fn webhook_event_type(headers: &HeaderMap, payload: &JsonValue) -> String {
    headers
        .get("X-GitHub-Event")
        .and_then(|value| value.to_str().ok())
        .or_else(|| {
            headers
                .get("X-GitLab-Event")
                .and_then(|value| value.to_str().ok())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            payload
                .get("event_type")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn render_prompt(
    template: &str,
    payload: &JsonValue,
    event_type: &str,
    route_name: &str,
) -> String {
    if template.trim().is_empty() {
        return format!(
            "Webhook event '{}' on route '{}':\n\n```json\n{}\n```",
            event_type,
            route_name,
            truncate_chars(
                &serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string()),
                4000
            )
        );
    }
    let regex = Regex::new(r"\{([a-zA-Z0-9_.]+)\}").unwrap();
    regex
        .replace_all(template, |captures: &regex::Captures<'_>| {
            let key = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
            if key == "__raw__" {
                return truncate_chars(
                    &serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string()),
                    4000,
                );
            }
            resolve_payload_path(payload, key).unwrap_or_else(|| format!("{{{key}}}"))
        })
        .into_owned()
}

fn render_delivery_extra(
    extra: &JsonMap<String, JsonValue>,
    payload: &JsonValue,
) -> JsonMap<String, JsonValue> {
    extra
        .iter()
        .map(|(key, value)| {
            let rendered = if let Some(text) = value.as_str() {
                JsonValue::String(render_prompt(text, payload, "", ""))
            } else {
                value.clone()
            };
            (key.clone(), rendered)
        })
        .collect()
}

fn resolve_payload_path(payload: &JsonValue, key: &str) -> Option<String> {
    let mut current = payload;
    for part in key.split('.') {
        current = current.get(part)?;
    }
    Some(match current {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(_) | JsonValue::Object(_) => truncate_chars(
            &serde_json::to_string_pretty(current).unwrap_or_else(|_| "null".to_string()),
            2000,
        ),
        other => other.to_string(),
    })
}

fn send_message_supported_platform(platform: &str) -> bool {
    matches!(
        platform,
        "telegram"
            | "discord"
            | "slack"
            | "signal"
            | "sms"
            | "whatsapp"
            | "matrix"
            | "mattermost"
            | "homeassistant"
            | "email"
            | "dingtalk"
            | "feishu"
            | "wecom"
            | "weixin"
            | "bluebubbles"
            | "qqbot"
            | "yuanbao"
    )
}

fn webhook_enabled(loaded: &LoadedConfig) -> bool {
    env_truthy("WEBHOOK_ENABLED")
        || loaded
            .cfg_get(&["platforms", "webhook", "enabled"])
            .and_then(yaml_bool)
            .unwrap_or(false)
}

fn webhook_has_other_platforms_enabled(loaded: &LoadedConfig) -> bool {
    let env_enabled = [
        ("API_SERVER_ENABLED", true),
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
                if name == "webhook" {
                    return false;
                }
                value
                    .as_mapping()
                    .and_then(|mapping| mapping_bool(mapping, "enabled"))
                    .unwrap_or(false)
            })
        })
}

fn disabled_memory_toolsets(loaded: &LoadedConfig) -> Option<Vec<String>> {
    (!loaded.config.memory.any_enabled()).then(|| vec![String::from("memory")])
}

fn webhook_workdir(loaded: &LoadedConfig) -> PathBuf {
    let cwd = loaded.config.terminal.cwd.trim();
    if cwd.is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(cwd)
    }
}

fn normalize_route_name(value: &str) -> Result<String, Box<dyn Error>> {
    let normalized = value.trim().to_ascii_lowercase().replace(' ', "-");
    let valid = !normalized.is_empty()
        && normalized.chars().enumerate().all(|(index, ch)| match ch {
            'a'..='z' | '0'..='9' => true,
            '-' | '_' => index > 0,
            _ => false,
        });
    if !valid {
        return Err(format!("Invalid webhook route name: {value}").into());
    }
    Ok(normalized)
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

fn yaml_bool(value: &YamlValue) -> Option<bool> {
    match value {
        YamlValue::Bool(boolean) => Some(*boolean),
        YamlValue::String(text) => Some(matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )),
        _ => None,
    }
}

fn mapping_bool(mapping: &Mapping, key: &str) -> Option<bool> {
    mapping
        .get(YamlValue::String(key.to_string()))
        .and_then(yaml_bool)
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
        YamlValue::Number(number) => number.as_u64().and_then(|value| u16::try_from(value).ok()),
        YamlValue::String(text) => text.trim().parse::<u16>().ok(),
        _ => None,
    }
}

fn mapping_usize(mapping: &Mapping, key: &str) -> Option<usize> {
    match mapping.get(YamlValue::String(key.to_string()))? {
        YamlValue::Number(number) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok()),
        YamlValue::String(text) => text.trim().parse::<usize>().ok(),
        _ => None,
    }
}

fn compute_signature(secret: &str, body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(body);
    let bytes = mac.finalize().into_bytes();
    format!("sha256={}", hex_bytes(&bytes))
}

fn compute_generic_signature(secret: &str, body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(body);
    let bytes = mac.finalize().into_bytes();
    hex_bytes(&bytes)
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        value.chars().take(max).collect()
    }
}

fn unix_ts_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_ts_secs_f64() -> f64 {
    unix_ts_secs() as f64
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::oneshot;

    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn remove_env_var(key: &str) {
        unsafe {
            std::env::remove_var(key);
        }
    }

    fn temp_context(config_text: &str) -> (TempDir, HermesContext, LoadedConfig) {
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("config.yaml"), config_text).unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().to_path_buf()));
        let loaded = context.load_config_document().unwrap();
        (temp, context, loaded)
    }

    fn mock_http_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnOnce(String, String) + Send + 'static,
    {
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
            let mut body = request[header_end..].to_vec();
            while body.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body.extend_from_slice(&buffer[..read]);
            }
            handler(
                headers,
                String::from_utf8_lossy(&body[..content_length]).to_string(),
            );
            let response_body = "{\"errcode\":0,\"errmsg\":\"ok\"}";
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

    fn mock_model_server<F>(response_body: String, handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnOnce(String) + Send + 'static,
    {
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
            handler(String::from_utf8_lossy(&request).to_string());
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
    fn native_webhook_deliver_only_routes_to_dingtalk() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (dingtalk_base_url, dingtalk_join) = mock_http_server(|headers, body| {
            assert!(headers.starts_with("POST /robot/send?access_token=test-token "));
            let payload: JsonValue = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["text"]["content"], json!("Alert: ping"));
        });
        let webhook_url = format!("{dingtalk_base_url}/robot/send?access_token=test-token");
        let config_text = "platforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 8644\n      routes:\n        alerts:\n          secret: topsecret\n          prompt: 'Alert: {message}'\n          deliver: dingtalk\n          deliver_only: true\n          deliver_extra:\n            chat_id: cidding==\n";
        let (_temp, context, loaded) = temp_context(config_text);
        let settings = load_webhook_settings(&loaded).unwrap();
        let static_routes = load_static_routes(&loaded, &settings.global_secret).unwrap();
        let state = NativeWebhookState {
            context: context.clone(),
            loaded,
            settings: settings.clone(),
            static_routes,
            dynamic_routes_path: context.hermes_home().join(DYNAMIC_ROUTES_FILENAME),
            runtime_state: Arc::new(Mutex::new(WebhookRuntimeState::default())),
        };
        set_env_var("DINGTALK_WEBHOOK_URL", &webhook_url);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server_thread = thread::spawn(move || {
            let runtime = Runtime::new().unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
                let app = Router::new()
                    .route("/health", get(handle_health))
                    .route("/webhooks/{route_name}", post(handle_webhook))
                    .with_state(Arc::new(state));
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
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
        let payload = br#"{"message":"ping"}"#;
        let response: JsonValue = client
            .post(format!("http://{addr}/webhooks/alerts"))
            .header("Content-Type", "application/json")
            .header(
                "X-Hub-Signature-256",
                compute_signature("topsecret", payload),
            )
            .header("X-GitHub-Event", "test")
            .body(payload.to_vec())
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(response["status"], json!("delivered"));
        drop(client);
        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        dingtalk_join.join().unwrap();
        remove_env_var("DINGTALK_WEBHOOK_URL");
    }

    #[test]
    fn native_webhook_agent_mode_runs_model_and_delivers() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (dingtalk_tx, dingtalk_rx) = mpsc::channel();
        let (dingtalk_base_url, dingtalk_join) = mock_http_server(move |_headers, body| {
            let payload: JsonValue = serde_json::from_str(&body).unwrap();
            assert_eq!(
                payload["text"]["content"],
                json!("handled by native webhook")
            );
            dingtalk_tx.send(()).unwrap();
        });
        let (model_tx, model_rx) = mpsc::channel();
        let (model_base_url, model_join) = mock_model_server(
            json!({
                "id": "chatcmpl-test",
                "choices": [{
                    "message": {
                        "content": "handled by native webhook"
                    }
                }]
            })
            .to_string(),
            move |request| {
                assert!(request.starts_with("POST /chat/completions "));
                model_tx.send(()).unwrap();
            },
        );
        let webhook_url = format!("{dingtalk_base_url}/robot/send?access_token=test-token");
        let config_text = format!(
            "model:\n  default: test-model\n  provider: custom\n  base_url: {model_base_url}\n  api_key: test-key\n  api_mode: chat_completions\nplatforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 8644\n      routes:\n        review:\n          secret: topsecret\n          prompt: 'Summarize: {{message}}'\n          deliver: dingtalk\n          deliver_extra:\n            chat_id: cidding==\n"
        );
        let (_temp, context, loaded) = temp_context(&config_text);
        let settings = load_webhook_settings(&loaded).unwrap();
        let static_routes = load_static_routes(&loaded, &settings.global_secret).unwrap();
        let state = NativeWebhookState {
            context: context.clone(),
            loaded,
            settings: settings.clone(),
            static_routes,
            dynamic_routes_path: context.hermes_home().join(DYNAMIC_ROUTES_FILENAME),
            runtime_state: Arc::new(Mutex::new(WebhookRuntimeState::default())),
        };
        set_env_var("DINGTALK_WEBHOOK_URL", &webhook_url);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server_thread = thread::spawn(move || {
            let runtime = Runtime::new().unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
                let app = Router::new()
                    .route("/health", get(handle_health))
                    .route("/webhooks/{route_name}", post(handle_webhook))
                    .with_state(Arc::new(state));
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
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
        let payload = br#"{"message":"ping"}"#;
        let response: JsonValue = client
            .post(format!("http://{addr}/webhooks/review"))
            .header("Content-Type", "application/json")
            .header(
                "X-Hub-Signature-256",
                compute_signature("topsecret", payload),
            )
            .header("X-GitHub-Event", "test")
            .body(payload.to_vec())
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(response["status"], json!("accepted"));
        model_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        dingtalk_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        drop(client);
        let _ = shutdown_tx.send(());
        server_thread.join().unwrap();
        dingtalk_join.join().unwrap();
        model_join.join().unwrap();
        remove_env_var("DINGTALK_WEBHOOK_URL");
    }
}
