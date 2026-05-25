use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command as StdCommand, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, HeaderValue, Request, Response, StatusCode};
use axum::middleware::{self, Next};
use axum::routing::{delete, get, patch, post, put};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use regex::Regex;
use reqwest::blocking::Client as BlockingClient;
use ring::rand::{SecureRandom, SystemRandom};
use rusqlite::{Connection, params};
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::process::Command as TokioCommand;
use tokio::sync::{broadcast, mpsc};

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::gateway_cmd::{
    GatewayArgs, GatewayCommand, GatewayServiceArgs, GatewaySystemArgs, print_gateway,
};
use crate::plugins_cmd::{
    dashboard_context_engine_options, dashboard_current_context_engine,
    dashboard_current_memory_provider, dashboard_install_plugin, dashboard_list_plugins,
    dashboard_memory_provider_options, dashboard_plugin_sets, dashboard_remove_user_plugin,
    dashboard_save_context_engine, dashboard_save_memory_provider,
    dashboard_set_agent_plugin_enabled, dashboard_update_user_plugin,
};
use crate::python_bridge::resolve_repo_python;
use hermes_core::{
    HermesContext, SessionStore, ToolRuntime, clear_provider_auth_state, get_all_toolsets,
    handle_cronjob, list_provider_profiles, resolve_codex_access_token,
    resolve_minimax_oauth_runtime_credentials, resolve_nous_runtime_credentials,
};

const SESSION_HEADER_NAME: &str = "X-Hermes-Session-Token";
const BUILTIN_DASHBOARD_THEMES: &[(&str, &str, &str)] = &[
    (
        "default",
        "Hermes Teal",
        "Classic dark teal - the canonical Hermes look",
    ),
    (
        "default-large",
        "Hermes Teal (Large)",
        "Hermes Teal with bigger fonts and roomier spacing",
    ),
    ("midnight", "Midnight", "Deep blue-violet with cool accents"),
    ("ember", "Ember", "Warm crimson and bronze - forge vibes"),
    ("mono", "Mono", "Clean grayscale - minimal and focused"),
    (
        "cyberpunk",
        "Cyberpunk",
        "Neon green on black - matrix terminal",
    ),
    (
        "rose",
        "Rose",
        "Soft pink and warm ivory - easy on the eyes",
    ),
];
const LOOPBACK_HOSTS: &[&str] = &["127.0.0.1", "::1", "localhost", "testclient"];
const VALID_CHANNEL_PATTERN: &str = r"^[A-Za-z0-9._-]{1,128}$";
const DEFAULT_CONFIG_YAML: &str = include_str!("default_config.yaml");
const RESERVED_ALIAS_NAMES: &[&str] = &["hermes", "default", "test", "tmp", "root", "sudo"];
const HERMES_SUBCOMMANDS: &[&str] = &[
    "chat",
    "model",
    "gateway",
    "setup",
    "whatsapp",
    "login",
    "logout",
    "status",
    "cron",
    "doctor",
    "dump",
    "config",
    "pairing",
    "skills",
    "tools",
    "mcp",
    "sessions",
    "insights",
    "version",
    "update",
    "uninstall",
    "profile",
    "plugins",
    "honcho",
    "acp",
];
const DASHBOARD_ENV_KEYS: &[(&str, &str)] = &[
    ("OPENROUTER_API_KEY", "OpenRouter API key"),
    ("OPENAI_API_KEY", "OpenAI API key"),
    ("ANTHROPIC_API_KEY", "Anthropic API key"),
    ("ANTHROPIC_TOKEN", "Anthropic token"),
    ("NOUS_API_KEY", "Nous API key"),
    ("GOOGLE_API_KEY", "Google API key"),
    ("GEMINI_API_KEY", "Gemini API key"),
    ("GLM_API_KEY", "GLM API key"),
    ("ZAI_API_KEY", "ZAI API key"),
    ("KIMI_API_KEY", "Kimi API key"),
    ("MINIMAX_API_KEY", "MiniMax API key"),
    ("DEEPSEEK_API_KEY", "DeepSeek API key"),
    ("DASHSCOPE_API_KEY", "DashScope API key"),
    ("HF_TOKEN", "Hugging Face token"),
    ("AI_GATEWAY_API_KEY", "AI Gateway API key"),
    ("FIRECRAWL_API_KEY", "Firecrawl API key"),
    ("TAVILY_API_KEY", "Tavily API key"),
    ("BROWSERBASE_API_KEY", "Browserbase API key"),
    ("FAL_KEY", "FAL API key"),
    ("ELEVENLABS_API_KEY", "ElevenLabs API key"),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
    ("DISCORD_BOT_TOKEN", "Discord bot token"),
    ("SLACK_BOT_TOKEN", "Slack bot token"),
];
const LOG_COMPONENT_PREFIXES: &[(&str, &[&str])] = &[
    ("gateway", &["gateway"]),
    (
        "agent",
        &["agent", "run_agent", "model_tools", "batch_runner"],
    ),
    ("tools", &["tools"]),
    ("cli", &["hermes_cli", "cli"]),
    ("cron", &["cron"]),
];
const OAUTH_PROVIDER_CATALOG: &[(&str, &str, &str, &str, &str)] = &[
    (
        "anthropic",
        "Anthropic (Claude API)",
        "pkce",
        "hermes auth add anthropic",
        "https://docs.claude.com/en/api/getting-started",
    ),
    (
        "claude-code",
        "Claude Code (subscription)",
        "external",
        "claude setup-token",
        "https://docs.claude.com/en/docs/claude-code",
    ),
    (
        "nous",
        "Nous Portal",
        "device_code",
        "hermes auth add nous",
        "https://portal.nousresearch.com",
    ),
    (
        "openai-codex",
        "OpenAI Codex (ChatGPT)",
        "device_code",
        "hermes auth add openai-codex",
        "https://platform.openai.com/docs",
    ),
    (
        "qwen-oauth",
        "Qwen (via Qwen CLI)",
        "external",
        "hermes auth add qwen-oauth",
        "https://github.com/QwenLM/qwen-code",
    ),
    (
        "minimax-oauth",
        "MiniMax (OAuth)",
        "device_code",
        "hermes auth add minimax-oauth",
        "https://www.minimax.io",
    ),
];
const OAUTH_SESSION_TTL_SECONDS: i64 = 15 * 60;
const ANTHROPIC_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const DEFAULT_ANTHROPIC_OAUTH_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const DEFAULT_ANTHROPIC_OAUTH_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const ANTHROPIC_OAUTH_REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const ANTHROPIC_OAUTH_SCOPES: &str = "org:create_api_key user:profile user:inference";
const ANTHROPIC_OAUTH_USER_AGENT: &str = "claude-cli/0.0.0 (external, cli)";
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_CODEX_OAUTH_ISSUER: &str = "https://auth.openai.com";
const DEFAULT_CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const DEFAULT_NOUS_PORTAL_URL: &str = "https://portal.nousresearch.com";
const DEFAULT_NOUS_INFERENCE_URL: &str = "https://inference-api.nousresearch.com/v1";
const DEFAULT_NOUS_CLIENT_ID: &str = "hermes-cli";
const DEFAULT_NOUS_SCOPE: &str = "inference:mint_agent_key";
const MINIMAX_OAUTH_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
const MINIMAX_OAUTH_SCOPE: &str = "group_id profile model.completion";
const MINIMAX_OAUTH_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:user_code";
const DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL: &str = "https://api.minimax.io";
const DEFAULT_MINIMAX_OAUTH_INFERENCE_BASE_URL: &str = "https://api.minimax.io/anthropic";
const DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL: &str = "https://api.minimaxi.com";
const DEFAULT_MINIMAX_OAUTH_CN_INFERENCE_BASE_URL: &str = "https://api.minimaxi.com/anthropic";
const AUXILIARY_TASK_SLOTS: &[&str] = &[
    "vision",
    "web_extract",
    "compression",
    "session_search",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "curator",
];

#[derive(Clone)]
pub(crate) struct DashboardLaunchConfig {
    pub host: String,
    pub port: u16,
    pub open_browser: bool,
    pub allow_public: bool,
    pub embedded_chat: bool,
    pub project_root: PathBuf,
}

#[derive(Clone)]
struct DashboardState {
    context: HermesContext,
    project_root: PathBuf,
    web_dist: PathBuf,
    token: String,
    embedded_chat: bool,
    bound_host: String,
    bound_port: u16,
    defaults: JsonValue,
    schema: JsonValue,
    plugins: Arc<RwLock<PluginCache>>,
    oauth_sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    channel_re: Regex,
}

#[derive(Default)]
struct PluginCache {
    manifests: Vec<JsonValue>,
    asset_dirs: HashMap<String, PathBuf>,
}

#[derive(Deserialize)]
struct SessionsQuery {
    limit: Option<i64>,
    offset: Option<i64>,
}

#[derive(Deserialize)]
struct SessionSearchQuery {
    q: Option<String>,
    limit: Option<i64>,
}

#[derive(Deserialize)]
struct PtyQuery {
    token: Option<String>,
    resume: Option<String>,
    channel: Option<String>,
}

#[derive(Deserialize)]
struct TokenChannelQuery {
    token: Option<String>,
    channel: Option<String>,
}

#[derive(Deserialize)]
struct DaysQuery {
    days: Option<i64>,
}

#[derive(Deserialize)]
struct LogsQuery {
    file: Option<String>,
    lines: Option<i64>,
    level: Option<String>,
    component: Option<String>,
    search: Option<String>,
}

#[derive(Deserialize)]
struct ActionStatusQuery {
    lines: Option<i64>,
}

#[derive(Deserialize)]
struct ConfigUpdateBody {
    config: JsonValue,
}

#[derive(Deserialize)]
struct RawConfigUpdateBody {
    yaml_text: String,
}

#[derive(Deserialize)]
struct EnvVarBody {
    key: String,
    value: Option<String>,
}

#[derive(Deserialize)]
struct EnvRevealBody {
    key: String,
}

#[derive(Deserialize)]
struct ThemeSetBody {
    name: String,
}

#[derive(Deserialize)]
struct OAuthSubmitBody {
    session_id: String,
    code: String,
}

#[derive(Deserialize)]
struct SoulUpdateBody {
    content: String,
}

#[derive(Deserialize)]
struct ModelAssignmentBody {
    scope: String,
    provider: String,
    model: String,
    task: Option<String>,
}

#[derive(Deserialize)]
struct CronJobCreateBody {
    prompt: String,
    schedule: String,
    name: Option<String>,
    deliver: Option<String>,
}

#[derive(Deserialize)]
struct CronJobUpdateBody {
    updates: JsonValue,
}

#[derive(Deserialize)]
struct AgentPluginInstallBody {
    identifier: String,
    force: Option<bool>,
    enable: Option<bool>,
}

#[derive(Deserialize)]
struct PluginProvidersBody {
    memory_provider: Option<String>,
    context_engine: Option<String>,
}

#[derive(Deserialize)]
struct PluginVisibilityBody {
    hidden: bool,
}

#[derive(Deserialize)]
struct ProfileCreateBody {
    name: String,
    clone_from_default: bool,
}

#[derive(Deserialize)]
struct ProfileRenameBody {
    new_name: String,
}

#[derive(Deserialize)]
struct SkillToggleBody {
    name: String,
    enabled: bool,
}

#[derive(Debug, Clone)]
struct ProfileSummary {
    name: String,
    path: PathBuf,
    is_default: bool,
    model: Option<String>,
    provider: Option<String>,
    has_env: bool,
    skill_count: usize,
}

#[derive(Debug)]
enum ProfileLookupError {
    Invalid(String),
    Missing(String),
}

#[derive(Debug, Clone)]
struct SkillSummary {
    name: String,
    description: String,
    category: Option<String>,
}

struct ActionProcess {
    child: Option<Child>,
    pid: u32,
    exit_code: Option<i32>,
}

#[derive(Debug, Clone)]
struct OAuthSession {
    session_id: String,
    provider: String,
    flow: String,
    created_at: i64,
    status: String,
    error_message: Option<String>,
    expires_at: Option<i64>,
    verifier: Option<String>,
    state_nonce: Option<String>,
    verification_url: Option<String>,
    user_code: Option<String>,
    poll_interval: Option<i64>,
    device_code: Option<String>,
    device_auth_id: Option<String>,
    portal_base_url: Option<String>,
    inference_base_url: Option<String>,
    client_id: Option<String>,
    scope: Option<String>,
}

pub(crate) fn run_dashboard_server(
    config: DashboardLaunchConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_bind_host(&config.host, config.allow_public)?;

    if !config.allow_public && !is_loopback_bind(&config.host) {
        return Err(format!(
            "Refusing to bind to {} - use --insecure to override on trusted networks only.",
            config.host
        )
        .into());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let context = HermesContext::detect();
        let defaults = serde_yaml::from_str::<serde_yaml::Value>(DEFAULT_CONFIG_YAML)
            .ok()
            .and_then(yaml_to_json)
            .unwrap_or_else(|| json!({}));
        let schema = build_config_schema(&defaults);
        let state = Arc::new(DashboardState {
            context: context.clone(),
            project_root: config.project_root.clone(),
            web_dist: dashboard_web_dist(&config.project_root),
            token: generate_session_token()?,
            embedded_chat: config.embedded_chat,
            bound_host: config.host.clone(),
            bound_port: config.port,
            defaults,
            schema,
            plugins: Arc::new(RwLock::new(discover_dashboard_plugins(
                &config.project_root,
                &context.hermes_home(),
            ))),
            oauth_sessions: Arc::new(Mutex::new(HashMap::new())),
            channel_re: Regex::new(VALID_CHANNEL_PATTERN).expect("valid channel regex"),
        });

        let app = app_router(state.clone());
        let listener = TcpListener::bind((config.host.as_str(), config.port)).await?;

        if config.open_browser {
            let url = format!("http://{}:{}", config.host, config.port);
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(1));
                let _ = try_open_browser(&url);
            });
        }

        println!("  Hermes Web UI -> http://{}:{}", config.host, config.port);
        if std::env::var("HERMES_DASHBOARD_TEST_EXIT_AFTER_BIND")
            .ok()
            .is_some_and(|value| value == "1")
        {
            return Ok::<(), Box<dyn std::error::Error>>(());
        }
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

fn app_router(state: Arc<DashboardState>) -> Router {
    Router::new()
        .route("/api/status", get(get_status))
        .route("/api/config", get(get_config).put(update_config))
        .route("/api/config/defaults", get(get_config_defaults))
        .route("/api/config/schema", get(get_config_schema))
        .route(
            "/api/config/raw",
            get(get_config_raw).put(update_config_raw),
        )
        .route(
            "/api/env",
            get(get_env_vars).put(set_env_var).delete(delete_env_var),
        )
        .route("/api/env/reveal", post(reveal_env_var))
        .route("/api/model/info", get(get_model_info))
        .route("/api/model/options", get(get_model_options))
        .route("/api/model/auxiliary", get(get_auxiliary_models))
        .route("/api/model/set", post(set_model_assignment))
        .route("/api/providers/oauth", get(list_oauth_providers))
        .route(
            "/api/providers/oauth/:provider_id/start",
            post(start_oauth_login),
        )
        .route(
            "/api/providers/oauth/:provider_id/submit",
            post(submit_oauth_code),
        )
        .route(
            "/api/providers/oauth/:provider_id/poll/:session_id",
            get(poll_oauth_session),
        )
        .route(
            "/api/providers/oauth/sessions/:session_id",
            delete(cancel_oauth_session),
        )
        .route(
            "/api/providers/oauth/:provider_id",
            delete(disconnect_oauth_provider),
        )
        .route("/api/profiles", get(get_profiles).post(create_profile))
        .route(
            "/api/profiles/:name",
            patch(rename_profile).delete(delete_profile),
        )
        .route(
            "/api/profiles/:name/setup-command",
            get(get_profile_setup_command),
        )
        .route(
            "/api/profiles/:name/open-terminal",
            post(open_profile_terminal),
        )
        .route(
            "/api/profiles/:name/soul",
            get(get_profile_soul).put(update_profile_soul),
        )
        .route("/api/skills", get(get_skills))
        .route("/api/skills/toggle", put(toggle_skill))
        .route("/api/tools/toolsets", get(get_toolsets))
        .route("/api/analytics/usage", get(get_usage_analytics))
        .route("/api/analytics/models", get(get_models_analytics))
        .route("/api/logs", get(get_logs))
        .route("/api/cron/jobs", get(list_cron_jobs).post(create_cron_job))
        .route("/api/cron/jobs/:job_id/pause", post(pause_cron_job))
        .route("/api/cron/jobs/:job_id/resume", post(resume_cron_job))
        .route("/api/cron/jobs/:job_id/trigger", post(trigger_cron_job))
        .route(
            "/api/cron/jobs/:job_id",
            get(get_cron_job)
                .put(update_cron_job)
                .delete(delete_cron_job),
        )
        .route("/api/gateway/restart", post(restart_gateway))
        .route("/api/hermes/update", post(update_hermes))
        .route("/api/actions/:name/status", get(get_action_status))
        .route("/api/sessions", get(get_sessions))
        .route("/api/sessions/search", get(search_sessions))
        .route(
            "/api/sessions/:session_id",
            get(get_session).delete(delete_session),
        )
        .route(
            "/api/sessions/:session_id/messages",
            get(get_session_messages),
        )
        .route("/api/dashboard/themes", get(get_dashboard_themes))
        .route("/api/dashboard/theme", put(set_dashboard_theme))
        .route("/api/dashboard/plugins", get(get_dashboard_plugins))
        .route(
            "/api/dashboard/plugins/rescan",
            get(rescan_dashboard_plugins),
        )
        .route("/api/dashboard/plugins/hub", get(get_plugins_hub))
        .route(
            "/api/dashboard/agent-plugins/install",
            post(post_agent_plugin_install),
        )
        .route(
            "/api/dashboard/agent-plugins/:name/enable",
            post(post_agent_plugin_enable),
        )
        .route(
            "/api/dashboard/agent-plugins/:name/disable",
            post(post_agent_plugin_disable),
        )
        .route(
            "/api/dashboard/agent-plugins/:name/update",
            post(post_agent_plugin_update),
        )
        .route(
            "/api/dashboard/agent-plugins/:name",
            delete(delete_agent_plugin),
        )
        .route("/api/dashboard/plugin-providers", put(put_plugin_providers))
        .route(
            "/api/dashboard/plugins/:name/visibility",
            post(post_plugin_visibility),
        )
        .route("/api/ws", get(gateway_ws))
        .route("/api/pty", get(pty_ws))
        .route("/api/pub", get(pub_ws))
        .route("/api/events", get(events_ws))
        .route("/dashboard-plugins/:name/*path", get(serve_plugin_asset))
        .route("/assets/*path", get(serve_asset))
        .route("/*path", get(serve_spa))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            host_header_middleware,
        ))
        .with_state(state)
}

async fn host_header_middleware(
    State(state): State<Arc<DashboardState>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let host_header = request
        .headers()
        .get(HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !is_accepted_host(host_header, &state.bound_host) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({
                "detail": "Invalid Host header. Dashboard requests must use the hostname the server was bound to."
            }),
        );
    }
    next.run(request).await
}

async fn auth_middleware(
    State(state): State<Arc<DashboardState>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let path = request.uri().path();
    if path.starts_with("/api/")
        && !is_public_api_path(path)
        && !is_websocket_api_path(path)
        && !has_valid_session_token(request.headers(), &state.token)
    {
        return json_response(StatusCode::UNAUTHORIZED, json!({"detail": "Unauthorized"}));
    }
    next.run(request).await
}

async fn get_status(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let loaded = state.context.load_config_document().ok();
    let current_ver = loaded
        .as_ref()
        .and_then(|config| config_version_from_raw(&config.raw))
        .unwrap_or(0);
    let latest_ver = config_version_from_json(&state.defaults).unwrap_or(current_ver);
    let runtime = read_runtime_status(&state.context);
    let gateway_pid = get_running_gateway_pid(&state.context);
    let gateway_running = gateway_pid.is_some();
    let gateway_state = runtime
        .as_ref()
        .and_then(|value| value.get("gateway_state"))
        .cloned()
        .unwrap_or_else(|| {
            if gateway_running {
                JsonValue::String(String::from("running"))
            } else {
                JsonValue::String(String::from("stopped"))
            }
        });
    let gateway_platforms = if gateway_running {
        runtime
            .as_ref()
            .and_then(|value| value.get("platforms"))
            .cloned()
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };
    let gateway_exit_reason = runtime
        .as_ref()
        .and_then(|value| value.get("exit_reason"))
        .cloned()
        .unwrap_or(JsonValue::Null);
    let gateway_updated_at = runtime
        .as_ref()
        .and_then(|value| value.get("updated_at"))
        .cloned()
        .unwrap_or(JsonValue::Null);
    let active_sessions = count_active_sessions(&state.context).unwrap_or(0);

    json_response(
        StatusCode::OK,
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "release_date": null,
            "hermes_home": state.context.hermes_home(),
            "config_path": state.context.config_path(),
            "env_path": state.context.env_path(),
            "config_version": current_ver,
            "latest_config_version": latest_ver,
            "gateway_running": gateway_running,
            "gateway_pid": gateway_pid,
            "gateway_health_url": null,
            "gateway_state": gateway_state,
            "gateway_platforms": gateway_platforms,
            "gateway_exit_reason": gateway_exit_reason,
            "gateway_updated_at": gateway_updated_at,
            "active_sessions": active_sessions,
        }),
    )
}

async fn get_config(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    match state.context.load_config_document() {
        Ok(loaded) => {
            let payload = yaml_to_json(loaded.raw).unwrap_or_else(|| json!({}));
            json_response(StatusCode::OK, payload)
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn update_config(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<ConfigUpdateBody>,
) -> Response<Body> {
    let Some(config_value) = json_to_yaml(body.config) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "config must be a JSON object"}),
        );
    };
    let serde_yaml::Value::Mapping(mapping) = config_value else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "config must be a mapping"}),
        );
    };
    match write_yaml_mapping(&state.context.config_path(), &mapping) {
        Ok(()) => json_response(StatusCode::OK, json!({"ok": true})),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_config_defaults(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    json_response(StatusCode::OK, state.defaults.clone())
}

async fn get_config_schema(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    json_response(StatusCode::OK, state.schema.clone())
}

async fn get_config_raw(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    match fs::read_to_string(state.context.config_path()) {
        Ok(yaml) => json_response(StatusCode::OK, json!({"yaml": yaml})),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            json_response(StatusCode::OK, json!({"yaml": ""}))
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn update_config_raw(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<RawConfigUpdateBody>,
) -> Response<Body> {
    let parsed = match serde_yaml::from_str::<serde_yaml::Value>(&body.yaml_text) {
        Ok(value) => value,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"detail": format!("Invalid YAML: {error}")}),
            );
        }
    };
    match parsed {
        serde_yaml::Value::Mapping(mapping) => {
            match write_yaml_mapping(&state.context.config_path(), &mapping) {
                Ok(()) => json_response(StatusCode::OK, json!({"ok": true})),
                Err(error) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": error.to_string()}),
                ),
            }
        }
        _ => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "YAML must be a mapping"}),
        ),
    }
}

async fn get_env_vars(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let env_file = load_simple_env(&state.context.env_path());
    let mut keys = env_file.keys().cloned().collect::<Vec<_>>();
    for (key, _) in DASHBOARD_ENV_KEYS {
        if !keys.iter().any(|candidate| candidate == key) {
            keys.push((*key).to_string());
        }
    }
    for profile in list_provider_profiles().iter() {
        for key in profile.env_vars {
            if !keys.iter().any(|candidate| candidate == key) {
                keys.push((*key).to_string());
            }
        }
    }
    keys.sort();
    keys.dedup();

    let payload = keys
        .into_iter()
        .map(|key| {
            let raw_value: Option<String> = env_file
                .get(&key)
                .cloned()
                .or_else(|| std::env::var(&key).ok());
            let is_password = looks_like_secret_key(&key);
            let (description, url, tools, advanced) = env_metadata_for_key(&key);
            let category =
                if key.ends_with("_API_KEY") || key.ends_with("_TOKEN") || key.ends_with("_SECRET")
                {
                    "provider"
                } else {
                    "setting"
                };
            (
                key,
                json!({
                    "is_set": raw_value.as_ref().is_some_and(|value| !value.trim().is_empty()),
                    "redacted_value": raw_value.as_deref().map(redact_secret),
                    "description": description,
                    "url": url,
                    "category": category,
                    "is_password": is_password,
                    "tools": tools,
                    "advanced": advanced,
                }),
            )
        })
        .collect::<JsonMap<_, _>>();

    json_response(StatusCode::OK, JsonValue::Object(payload))
}

async fn set_env_var(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<EnvVarBody>,
) -> Response<Body> {
    let key = body.key.trim();
    if !is_valid_env_key(key) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "invalid environment variable name"}),
        );
    }
    let Some(value) = body.value.as_deref() else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "value is required"}),
        );
    };
    match save_env_value(state.context.env_path(), key, value) {
        Ok(()) => json_response(StatusCode::OK, json!({"ok": true, "key": key})),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn delete_env_var(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<EnvVarBody>,
) -> Response<Body> {
    let key = body.key.trim();
    if !is_valid_env_key(key) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "invalid environment variable name"}),
        );
    }
    match remove_env_key(&state.context.env_path(), key) {
        Ok(true) => json_response(StatusCode::OK, json!({"ok": true, "key": key})),
        Ok(false) => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": format!("{key} not found in .env")}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn reveal_env_var(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<EnvRevealBody>,
) -> Response<Body> {
    let key = body.key.trim();
    if !is_valid_env_key(key) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "invalid environment variable name"}),
        );
    }
    let env_file = load_simple_env(&state.context.env_path());
    match env_file.get(key) {
        Some(value) => json_response(StatusCode::OK, json!({"key": key, "value": value})),
        None => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": format!("{key} not found in .env")}),
        ),
    }
}

async fn get_model_info(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let loaded = state.context.load_config_document().ok();
    let model = loaded
        .as_ref()
        .and_then(|config| config.configured_model_name())
        .unwrap_or_default();
    let provider = loaded
        .as_ref()
        .and_then(|config| config.configured_model_provider())
        .unwrap_or_default();

    json_response(
        StatusCode::OK,
        json!({
            "model": model,
            "provider": provider,
            "auto_context_length": 0,
            "config_context_length": 0,
            "effective_context_length": 0,
            "capabilities": {},
        }),
    )
}

async fn get_model_options(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let loaded = state.context.load_config_document().ok();
    let provider = loaded
        .as_ref()
        .and_then(|config| config.configured_model_provider())
        .unwrap_or_default();
    let model = loaded
        .as_ref()
        .and_then(|config| config.configured_model_name())
        .unwrap_or_default();
    let providers = list_provider_profiles()
        .iter()
        .filter(|profile| {
            profile.name == provider
                || profile.auth_type != "api_key"
                || profile.api_key_env_vars().any(|key| {
                    std::env::var(key)
                        .ok()
                        .is_some_and(|value| !value.trim().is_empty())
                })
        })
        .map(|profile| {
            json!({
                "name": profile.name,
                "slug": profile.name,
                "models": [],
                "total_models": 0,
                "is_current": profile.name == provider,
                "source": "builtin",
            })
        })
        .collect::<Vec<_>>();
    json_response(
        StatusCode::OK,
        json!({
            "providers": providers,
            "provider": provider,
            "model": model,
        }),
    )
}

async fn get_auxiliary_models(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let loaded = match state.context.load_config_document() {
        Ok(loaded) => loaded,
        Err(error) => {
            return json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": error.to_string()}),
            );
        }
    };
    let tasks = AUXILIARY_TASK_SLOTS
        .into_iter()
        .map(|task| {
            let provider = loaded
                .cfg_get(&["auxiliary", task, "provider"])
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or("auto");
            let model = loaded
                .cfg_get(&["auxiliary", task, "model"])
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or("");
            let base_url = loaded
                .cfg_get(&["auxiliary", task, "base_url"])
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or("");
            json!({
                "task": task,
                "provider": provider,
                "model": model,
                "base_url": base_url,
            })
        })
        .collect::<Vec<_>>();
    let main_provider = loaded.configured_model_provider().unwrap_or_default();
    let main_model = loaded.configured_model_name().unwrap_or_default();
    json_response(
        StatusCode::OK,
        json!({
            "tasks": tasks,
            "main": {
                "provider": main_provider,
                "model": main_model,
            }
        }),
    )
}

async fn set_model_assignment(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<ModelAssignmentBody>,
) -> Response<Body> {
    let scope = body.scope.trim().to_ascii_lowercase();
    let provider = body.provider.trim();
    let model = body.model.trim();
    let task = body.task.as_deref().map(str::trim).unwrap_or("");

    if scope != "main" && scope != "auxiliary" {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "scope must be 'main' or 'auxiliary'"}),
        );
    }

    if scope == "main" {
        if provider.is_empty() || model.is_empty() {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"detail": "provider and model required for main"}),
            );
        }
        match set_main_model_assignment(&state.context, provider, model) {
            Ok(()) => {
                return json_response(
                    StatusCode::OK,
                    json!({
                        "ok": true,
                        "scope": "main",
                        "provider": provider,
                        "model": model,
                    }),
                );
            }
            Err(error) => {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": error.to_string()}),
                );
            }
        }
    }

    match set_auxiliary_model_assignment(&state.context, provider, model, task) {
        Ok(targets) => json_response(
            StatusCode::OK,
            if task == "__reset__" {
                json!({"ok": true, "scope": "auxiliary", "reset": true})
            } else {
                json!({
                    "ok": true,
                    "scope": "auxiliary",
                    "tasks": targets,
                    "provider": provider,
                    "model": model,
                })
            },
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn list_oauth_providers(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let providers = OAUTH_PROVIDER_CATALOG
        .iter()
        .map(|(id, name, flow, cli_command, docs_url)| {
            let status = oauth_provider_status(&state.context, id);
            json!({
                "id": id,
                "name": name,
                "flow": flow,
                "cli_command": cli_command,
                "docs_url": docs_url,
                "status": status,
            })
        })
        .collect::<Vec<_>>();
    json_response(StatusCode::OK, json!({"providers": providers}))
}

async fn disconnect_oauth_provider(
    State(state): State<Arc<DashboardState>>,
    AxumPath(provider_id): AxumPath<String>,
) -> Response<Body> {
    let provider = provider_id.trim();
    if provider.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "provider_id is required"}),
        );
    }
    if !OAUTH_PROVIDER_CATALOG
        .iter()
        .any(|(id, _, _, _, _)| *id == provider)
    {
        let available = OAUTH_PROVIDER_CATALOG
            .iter()
            .map(|(id, _, _, _, _)| *id)
            .collect::<Vec<_>>()
            .join(", ");
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": format!("Unknown provider: {provider}. Available: {available}")}),
        );
    }

    let result = if matches!(provider, "anthropic" | "claude-code") {
        let oauth_file = state.context.hermes_home().join(".anthropic_oauth.json");
        if oauth_file.exists() {
            let _ = fs::remove_file(&oauth_file);
        }
        let _ = clear_provider_auth_state(state.context.hermes_home().as_path(), "anthropic");
        true
    } else {
        match clear_provider_auth_state(state.context.hermes_home().as_path(), provider) {
            Ok(changed) => {
                if provider == "qwen-oauth" {
                    let qwen_path = qwen_oauth_creds_path(state.context.home_dir());
                    if qwen_path.exists() {
                        let _ = fs::remove_file(qwen_path);
                    }
                }
                changed
            }
            Err(error) => {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": error.to_string()}),
                );
            }
        }
    };

    json_response(
        StatusCode::OK,
        json!({
            "ok": if matches!(provider, "anthropic" | "claude-code") {
                true
            } else {
                result
            },
            "provider": provider,
        }),
    )
}

async fn start_oauth_login(
    State(state): State<Arc<DashboardState>>,
    AxumPath(provider_id): AxumPath<String>,
) -> Response<Body> {
    let provider = provider_id.trim();
    gc_oauth_sessions(&state);
    let Some((_, _, flow, cli_command, _)) = OAUTH_PROVIDER_CATALOG
        .iter()
        .find(|(id, _, _, _, _)| *id == provider)
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": format!("Unknown provider {provider}")}),
        );
    };
    if *flow == "external" {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": format!("{provider} uses an external CLI; run `{cli_command}` manually")}),
        );
    }

    match *flow {
        "pkce" => match start_anthropic_pkce(&state, provider) {
            Ok(payload) => json_response(StatusCode::OK, payload),
            Err(error) => {
                json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error}))
            }
        },
        "device_code" => {
            let state = state.clone();
            let provider = provider.to_string();
            match tokio::task::spawn_blocking(move || start_device_code_flow(&state, &provider))
                .await
            {
                Ok(Ok(payload)) => json_response(StatusCode::OK, payload),
                Ok(Err(error)) => {
                    json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error}))
                }
                Err(error) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": error.to_string()}),
                ),
            }
        }
        _ => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "Unsupported flow"}),
        ),
    }
}

async fn submit_oauth_code(
    State(state): State<Arc<DashboardState>>,
    AxumPath(provider_id): AxumPath<String>,
    axum::extract::Json(body): axum::extract::Json<OAuthSubmitBody>,
) -> Response<Body> {
    let provider = provider_id.trim();
    if provider != "anthropic" {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": format!("submit not supported for {provider}")}),
        );
    }
    let session_id = body.session_id.trim();
    let code = body.code.trim();
    if session_id.is_empty() || code.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "session_id and code are required"}),
        );
    }
    let state = state.clone();
    let session_id = session_id.to_string();
    let code = code.to_string();
    match tokio::task::spawn_blocking(move || submit_anthropic_pkce(&state, &session_id, &code))
        .await
    {
        Ok(Ok(payload)) => json_response(StatusCode::OK, payload),
        Ok(Err(error)) => {
            json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error}))
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn poll_oauth_session(
    State(state): State<Arc<DashboardState>>,
    AxumPath((provider_id, session_id)): AxumPath<(String, String)>,
) -> Response<Body> {
    let provider = provider_id.trim();
    let session = {
        let sessions = state.oauth_sessions.lock().expect("oauth session lock");
        sessions.get(session_id.trim()).cloned()
    };
    let Some(session) = session else {
        return json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": "Session not found or expired"}),
        );
    };
    if session.provider != provider {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "Provider mismatch for session"}),
        );
    }
    json_response(
        StatusCode::OK,
        json!({
            "session_id": session.session_id,
            "status": session.status,
            "error_message": session.error_message,
            "expires_at": session.expires_at,
        }),
    )
}

async fn cancel_oauth_session(
    State(state): State<Arc<DashboardState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Response<Body> {
    let removed = state
        .oauth_sessions
        .lock()
        .expect("oauth session lock")
        .remove(session_id.trim());
    if removed.is_none() {
        return json_response(
            StatusCode::OK,
            json!({"ok": false, "message": "session not found"}),
        );
    }
    json_response(
        StatusCode::OK,
        json!({"ok": true, "session_id": session_id.trim()}),
    )
}

async fn get_profiles(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    match list_profiles(&state.context) {
        Ok(profiles) => {
            let payload = profiles
                .into_iter()
                .map(|profile| {
                    json!({
                        "name": profile.name,
                        "path": profile.path,
                        "is_default": profile.is_default,
                        "model": profile.model,
                        "provider": profile.provider,
                        "has_env": profile.has_env,
                        "skill_count": profile.skill_count,
                    })
                })
                .collect::<Vec<_>>();
            json_response(StatusCode::OK, json!({"profiles": payload}))
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn create_profile(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<ProfileCreateBody>,
) -> Response<Body> {
    let name = body.name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "profile name is required"}),
        );
    }
    match create_profile_dir(
        &state.context,
        &state.project_root,
        name,
        body.clone_from_default,
    ) {
        Ok(path) => json_response(
            StatusCode::OK,
            json!({"ok": true, "name": name, "path": path.display().to_string()}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn rename_profile(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
    axum::extract::Json(body): axum::extract::Json<ProfileRenameBody>,
) -> Response<Body> {
    match rename_profile_dir(&state.context, &name, body.new_name.trim()) {
        Ok(path) => json_response(
            StatusCode::OK,
            json!({
                "ok": true,
                "name": body.new_name.trim(),
                "path": path.display().to_string(),
            }),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn delete_profile(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    match delete_profile_dir(&state.context, &name) {
        Ok(path) => json_response(
            StatusCode::OK,
            json!({"ok": true, "path": path.display().to_string()}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_profile_setup_command(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    match resolve_profile_dir(&state.context, &name) {
        Ok(_) => {
            let command = if name.trim().eq_ignore_ascii_case("default") {
                "hermes setup".to_string()
            } else {
                format!("{} setup", name.trim())
            };
            json_response(StatusCode::OK, json!({"command": command}))
        }
        Err(ProfileLookupError::Invalid(detail)) => {
            json_response(StatusCode::BAD_REQUEST, json!({"detail": detail}))
        }
        Err(ProfileLookupError::Missing(detail)) => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": detail}))
        }
    }
}

async fn open_profile_terminal(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    match resolve_profile_dir(&state.context, &name) {
        Ok(_) => {
            let command = if name.trim().eq_ignore_ascii_case("default") {
                "hermes setup".to_string()
            } else {
                format!("{} setup", name.trim())
            };
            match launch_profile_terminal(&command) {
                Ok(()) => json_response(StatusCode::OK, json!({"ok": true, "command": command})),
                Err(error) => json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"detail": error.to_string()}),
                ),
            }
        }
        Err(ProfileLookupError::Invalid(detail)) => {
            json_response(StatusCode::BAD_REQUEST, json!({"detail": detail}))
        }
        Err(ProfileLookupError::Missing(detail)) => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": detail}))
        }
    }
}

async fn get_profile_soul(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    match resolve_profile_dir(&state.context, &name) {
        Ok(profile_dir) => {
            let soul_path = profile_dir.join("SOUL.md");
            let content = fs::read_to_string(&soul_path).unwrap_or_default();
            json_response(
                StatusCode::OK,
                json!({
                    "content": content,
                    "exists": soul_path.exists(),
                }),
            )
        }
        Err(ProfileLookupError::Invalid(detail)) => {
            json_response(StatusCode::BAD_REQUEST, json!({"detail": detail}))
        }
        Err(ProfileLookupError::Missing(detail)) => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": detail}))
        }
    }
}

async fn update_profile_soul(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
    axum::extract::Json(body): axum::extract::Json<SoulUpdateBody>,
) -> Response<Body> {
    match resolve_profile_dir(&state.context, &name) {
        Ok(profile_dir) => match fs::write(profile_dir.join("SOUL.md"), body.content) {
            Ok(()) => json_response(StatusCode::OK, json!({"ok": true})),
            Err(error) => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": error.to_string()}),
            ),
        },
        Err(ProfileLookupError::Invalid(detail)) => {
            json_response(StatusCode::BAD_REQUEST, json!({"detail": detail}))
        }
        Err(ProfileLookupError::Missing(detail)) => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": detail}))
        }
    }
}

async fn get_skills(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let raw_config = state
        .context
        .load_config_document()
        .ok()
        .map(|loaded| loaded.raw)
        .unwrap_or(YamlValue::Mapping(Mapping::new()));
    match discover_dashboard_skills(&state.context, &raw_config) {
        Ok(skills) => {
            let disabled = resolve_dashboard_disabled_skills(&raw_config);
            let payload = skills
                .into_iter()
                .map(|skill| {
                    json!({
                        "name": skill.name,
                        "description": skill.description,
                        "category": skill.category,
                        "enabled": !disabled.contains(&skill.name),
                    })
                })
                .collect::<Vec<_>>();
            json_response(StatusCode::OK, JsonValue::Array(payload))
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn toggle_skill(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<SkillToggleBody>,
) -> Response<Body> {
    let name = body.name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "skill name is required"}),
        );
    }
    match set_skill_enabled(&state.context, name, body.enabled) {
        Ok(()) => json_response(
            StatusCode::OK,
            json!({"ok": true, "name": name, "enabled": body.enabled}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_toolsets(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let configured = state
        .context
        .load_config_document()
        .ok()
        .map(|loaded| loaded.config.toolsets)
        .unwrap_or_default();
    let payload = get_all_toolsets()
        .into_values()
        .map(|toolset| {
            let enabled = configured.iter().any(|item| item == &toolset.name);
            let mut tools = toolset.resolved_tools;
            tools.sort();
            tools.dedup();
            json!({
                "name": toolset.name,
                "label": toolset.name,
                "description": toolset.description,
                "enabled": enabled,
                "configured": enabled,
                "available": toolset.available,
                "tools": tools,
            })
        })
        .collect::<Vec<_>>();
    json_response(StatusCode::OK, JsonValue::Array(payload))
}

async fn get_logs(
    State(state): State<Arc<DashboardState>>,
    Query(query): Query<LogsQuery>,
) -> Response<Body> {
    let file = query.file.as_deref().unwrap_or("agent").trim();
    let log_file = match log_file_name(file) {
        Some(name) => name,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"detail": format!("Unknown log file: {file}")}),
            );
        }
    };
    let level = normalize_log_level(query.level.as_deref());
    let component = match normalize_log_component(query.component.as_deref()) {
        Ok(value) => value,
        Err(error) => {
            return json_response(StatusCode::BAD_REQUEST, json!({"detail": error}));
        }
    };
    let search = query
        .search
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase);
    let requested_lines = query.lines.unwrap_or(100).clamp(1, 500) as usize;
    let read_limit = if search.is_some() {
        2_000
    } else {
        requested_lines
    };
    let log_path = state.context.hermes_home().join("logs").join(log_file);
    if !log_path.exists() {
        return json_response(StatusCode::OK, json!({"file": file, "lines": []}));
    }
    match read_log_lines(&log_path, read_limit, level, component, search.as_deref()) {
        Ok(lines) => {
            let lines = if lines.len() > requested_lines {
                lines[lines.len() - requested_lines..].to_vec()
            } else {
                lines
            };
            json_response(StatusCode::OK, json!({"file": file, "lines": lines}))
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_usage_analytics(
    State(state): State<Arc<DashboardState>>,
    Query(query): Query<DaysQuery>,
) -> Response<Body> {
    let days = normalize_analytics_days(query.days);
    match read_usage_analytics(&state.context, days) {
        Ok(payload) => json_response(StatusCode::OK, payload),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_models_analytics(
    State(state): State<Arc<DashboardState>>,
    Query(query): Query<DaysQuery>,
) -> Response<Body> {
    let days = normalize_analytics_days(query.days);
    match read_models_analytics(&state.context, days) {
        Ok(payload) => json_response(StatusCode::OK, payload),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn list_cron_jobs(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    match read_cron_jobs(&state.context) {
        Ok(jobs) => json_response(StatusCode::OK, JsonValue::Array(jobs)),
        Err(error) => json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error})),
    }
}

async fn create_cron_job(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<CronJobCreateBody>,
) -> Response<Body> {
    if body.prompt.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "prompt is required"}),
        );
    }
    if body.schedule.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "schedule is required"}),
        );
    }
    match cron_action_result(
        &state.context,
        json!({
            "action": "create",
            "prompt": body.prompt.trim(),
            "schedule": body.schedule.trim(),
            "name": body.name.as_deref().unwrap_or("").trim(),
            "deliver": body.deliver.as_deref().unwrap_or("local").trim(),
        }),
    ) {
        Ok(value) => match value
            .get("job")
            .and_then(JsonValue::as_object)
            .and_then(|job| job.get("job_id"))
            .and_then(JsonValue::as_str)
        {
            Some(job_id) => match read_cron_job(&state.context, job_id) {
                Ok(Some(job)) => json_response(StatusCode::OK, job),
                Ok(None) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": "created cron job could not be reloaded"}),
                ),
                Err(error) => {
                    json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error}))
                }
            },
            None => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": "invalid cron create response"}),
            ),
        },
        Err(error) => json_response(StatusCode::BAD_REQUEST, json!({"detail": error})),
    }
}

async fn get_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response<Body> {
    match read_cron_job(&state.context, job_id.trim()) {
        Ok(Some(job)) => json_response(StatusCode::OK, job),
        Ok(None) => json_response(StatusCode::NOT_FOUND, json!({"detail": "Job not found"})),
        Err(error) => json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error})),
    }
}

async fn update_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
    axum::extract::Json(body): axum::extract::Json<CronJobUpdateBody>,
) -> Response<Body> {
    let Some(updates) = body.updates.as_object() else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "updates must be a JSON object"}),
        );
    };
    let mut payload = JsonMap::new();
    payload.insert(
        String::from("action"),
        JsonValue::String(String::from("update")),
    );
    payload.insert(
        String::from("job_id"),
        JsonValue::String(job_id.trim().to_string()),
    );
    for (key, value) in updates {
        payload.insert(key.clone(), value.clone());
    }
    match cron_action_result(&state.context, JsonValue::Object(payload)) {
        Ok(value) => match value.get("job").cloned() {
            Some(job) => json_response(StatusCode::OK, job),
            None => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": "invalid cron update response"}),
            ),
        },
        Err(error) if error.contains("not found") => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": error}))
        }
        Err(error) => json_response(StatusCode::BAD_REQUEST, json!({"detail": error})),
    }
}

async fn pause_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response<Body> {
    cron_job_action_response(&state.context, "pause", &job_id)
}

async fn resume_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response<Body> {
    cron_job_action_response(&state.context, "resume", &job_id)
}

async fn trigger_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response<Body> {
    cron_job_action_response(&state.context, "run", &job_id)
}

async fn delete_cron_job(
    State(state): State<Arc<DashboardState>>,
    AxumPath(job_id): AxumPath<String>,
) -> Response<Body> {
    match cron_action_result(
        &state.context,
        json!({"action": "remove", "job_id": job_id}),
    ) {
        Ok(_) => json_response(StatusCode::OK, json!({"ok": true})),
        Err(error) if error.contains("not found") => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": error}))
        }
        Err(error) => json_response(StatusCode::BAD_REQUEST, json!({"detail": error})),
    }
}

async fn restart_gateway(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    action_spawn_response(&state, &["gateway", "restart"], "gateway-restart")
}

async fn update_hermes(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    action_spawn_response(&state, &["update"], "hermes-update")
}

async fn get_action_status(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
    Query(query): Query<ActionStatusQuery>,
) -> Response<Body> {
    let Some(log_file) = action_log_file(&name) else {
        return json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": format!("Unknown action: {name}")}),
        );
    };
    let line_count = query.lines.unwrap_or(200).clamp(1, 2000) as usize;
    let log_path = state.context.hermes_home().join("logs").join(log_file);
    let lines = tail_lines(&log_path, line_count);
    let (running, exit_code, pid) = action_process_status(&name);
    json_response(
        StatusCode::OK,
        json!({
            "name": name,
            "running": running,
            "exit_code": exit_code,
            "pid": pid,
            "lines": lines,
        }),
    )
}

async fn get_sessions(
    State(state): State<Arc<DashboardState>>,
    Query(query): Query<SessionsQuery>,
) -> Response<Body> {
    let limit = query.limit.unwrap_or(20).clamp(1, 500);
    let offset = query.offset.unwrap_or(0).max(0);
    match open_session_store(&state.context).and_then(|store| {
        let total = store.session_count()?;
        let sessions = store.search_sessions(None, limit, offset)?;
        Ok((total, sessions))
    }) {
        Ok((total, sessions)) => {
            let now = now_ts();
            let payload = sessions
                .into_iter()
                .map(|session| {
                    json!({
                        "id": session.id,
                        "source": session.source,
                        "model": session.model,
                        "title": session.title,
                        "started_at": session.started_at,
                        "ended_at": session.ended_at,
                        "end_reason": session.end_reason,
                        "message_count": session.message_count,
                        "tool_call_count": session.tool_call_count,
                        "last_active": session.last_active,
                        "preview": session.preview,
                        "is_active": session.ended_at.is_none() && (now - session.last_active) < 300.0,
                    })
                })
                .collect::<Vec<_>>();
            json_response(
                StatusCode::OK,
                json!({
                    "sessions": payload,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }),
            )
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn search_sessions(
    State(state): State<Arc<DashboardState>>,
    Query(query): Query<SessionSearchQuery>,
) -> Response<Body> {
    let Some(raw_query) = query
        .q
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return json_response(StatusCode::OK, json!({"results": []}));
    };
    let limit = query.limit.unwrap_or(20).clamp(1, 200);
    match open_session_store(&state.context).and_then(|store| {
        let matches = store.search_messages(raw_query, None, None, None, limit, 0)?;
        Ok(matches)
    }) {
        Ok(matches) => {
            let mut seen = JsonMap::new();
            for entry in matches {
                if seen.contains_key(&entry.session_id) {
                    continue;
                }
                seen.insert(
                    entry.session_id.clone(),
                    json!({
                        "session_id": entry.session_id,
                        "snippet": entry.snippet,
                        "role": entry.role,
                        "source": entry.source,
                        "model": entry.model,
                        "session_started": entry.session_started,
                    }),
                );
            }
            json_response(
                StatusCode::OK,
                json!({"results": seen.into_values().collect::<Vec<_>>() }),
            )
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_session(
    State(state): State<Arc<DashboardState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Response<Body> {
    match open_session_store(&state.context).and_then(|store| {
        let resolved = resolve_session_id(&store, &session_id)?;
        Ok((store, resolved))
    }) {
        Ok((store, Some(resolved))) => match store.get_session(&resolved) {
            Ok(Some(session)) => json_response(
                StatusCode::OK,
                serde_json::to_value(session).unwrap_or_else(|_| json!({})),
            ),
            Ok(None) => json_response(
                StatusCode::NOT_FOUND,
                json!({"detail": "Session not found"}),
            ),
            Err(error) => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": error.to_string()}),
            ),
        },
        Ok((_, None)) => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": "Session not found"}),
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_session_messages(
    State(state): State<Arc<DashboardState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Response<Body> {
    match open_session_store(&state.context).and_then(|store| {
        let resolved = resolve_session_id(&store, &session_id)?;
        Ok((store, resolved))
    }) {
        Ok((store, Some(resolved))) => match store.get_messages(&resolved) {
            Ok(messages) => json_response(
                StatusCode::OK,
                json!({
                    "session_id": resolved,
                    "messages": serde_json::to_value(messages).unwrap_or_else(|_| json!([])),
                }),
            ),
            Err(error) => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": error.to_string()}),
            ),
        },
        Ok((_, None)) => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": "Session not found"}),
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn delete_session(
    State(state): State<Arc<DashboardState>>,
    AxumPath(session_id): AxumPath<String>,
) -> Response<Body> {
    match open_session_store(&state.context).and_then(|store| {
        let resolved = resolve_session_id(&store, &session_id)?;
        Ok((store, resolved))
    }) {
        Ok((store, Some(resolved))) => match store.delete_session(&resolved) {
            Ok(deleted) => {
                if deleted {
                    remove_session_files(&state.context.hermes_home().join("sessions"), &resolved);
                }
                json_response(StatusCode::OK, json!({"ok": deleted}))
            }
            Err(error) => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": error.to_string()}),
            ),
        },
        Ok((_, None)) => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": "Session not found"}),
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_dashboard_themes(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let active = state
        .context
        .load_config_document()
        .ok()
        .and_then(|loaded| {
            loaded
                .cfg_get(&["dashboard", "theme"])
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| String::from("default"));
    let themes = BUILTIN_DASHBOARD_THEMES
        .iter()
        .map(|(name, label, description)| {
            json!({
                "name": name,
                "label": label,
                "description": description,
            })
        })
        .collect::<Vec<_>>();
    json_response(StatusCode::OK, json!({"active": active, "themes": themes}))
}

async fn set_dashboard_theme(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<ThemeSetBody>,
) -> Response<Body> {
    let name = body.name.trim();
    if !is_valid_theme_name(name) {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "invalid theme name"}),
        );
    }
    match update_yaml_setting(&state.context.config_path(), &["dashboard", "theme"], name) {
        Ok(()) => json_response(
            StatusCode::OK,
            json!({"ok": true, "theme": body.name.trim()}),
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn get_dashboard_plugins(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    let manifests = visible_dashboard_manifests(&state);
    json_response(StatusCode::OK, JsonValue::Array(manifests))
}

async fn rescan_dashboard_plugins(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    refresh_dashboard_plugin_cache(&state);
    let count = state
        .plugins
        .read()
        .map(|cache| cache.manifests.len())
        .unwrap_or(0);
    json_response(StatusCode::OK, json!({"ok": true, "count": count}))
}

async fn get_plugins_hub(State(state): State<Arc<DashboardState>>) -> Response<Body> {
    match build_plugins_hub(&state) {
        Ok(payload) => json_response(StatusCode::OK, payload),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn post_agent_plugin_install(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<AgentPluginInstallBody>,
) -> Response<Body> {
    let identifier = body.identifier.trim();
    if identifier.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "plugin identifier is required"}),
        );
    }
    match dashboard_install_plugin(
        &state.context,
        identifier,
        body.force.unwrap_or(false),
        body.enable.unwrap_or(true),
    ) {
        Ok(result) => {
            refresh_dashboard_plugin_cache(&state);
            json_response(
                StatusCode::OK,
                json!({
                    "ok": true,
                    "plugin_name": result.plugin_name,
                    "warnings": result.warnings,
                    "missing_env": result.missing_env,
                    "enabled": result.enabled,
                }),
            )
        }
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn post_agent_plugin_enable(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    plugin_enable_disable_response(&state, &name, true)
}

async fn post_agent_plugin_disable(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    plugin_enable_disable_response(&state, &name, false)
}

async fn post_agent_plugin_update(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    let name = name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "plugin name is required"}),
        );
    }
    match dashboard_update_user_plugin(&state.context, name) {
        Ok(result) => {
            refresh_dashboard_plugin_cache(&state);
            json_response(
                StatusCode::OK,
                json!({
                    "ok": true,
                    "name": result.name,
                    "output": result.output,
                    "unchanged": result.unchanged,
                }),
            )
        }
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn delete_agent_plugin(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
) -> Response<Body> {
    let name = name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "plugin name is required"}),
        );
    }
    match dashboard_remove_user_plugin(&state.context, name) {
        Ok(removed_name) => {
            refresh_dashboard_plugin_cache(&state);
            json_response(StatusCode::OK, json!({"ok": true, "name": removed_name}))
        }
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn put_plugin_providers(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Json(body): axum::extract::Json<PluginProvidersBody>,
) -> Response<Body> {
    if let Some(memory_provider) = body.memory_provider.as_deref() {
        if let Err(error) = dashboard_save_memory_provider(&state.context, memory_provider.trim()) {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"detail": error.to_string()}),
            );
        }
    }
    if let Some(context_engine) = body.context_engine.as_deref() {
        if let Err(error) = dashboard_save_context_engine(&state.context, context_engine.trim()) {
            return json_response(
                StatusCode::BAD_REQUEST,
                json!({"detail": error.to_string()}),
            );
        }
    }
    json_response(StatusCode::OK, json!({"ok": true}))
}

async fn post_plugin_visibility(
    State(state): State<Arc<DashboardState>>,
    AxumPath(name): AxumPath<String>,
    axum::extract::Json(body): axum::extract::Json<PluginVisibilityBody>,
) -> Response<Body> {
    let name = name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "plugin name is required"}),
        );
    }
    match set_dashboard_plugin_hidden(&state.context, name, body.hidden) {
        Ok(()) => json_response(
            StatusCode::OK,
            json!({"ok": true, "name": name, "hidden": body.hidden}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn serve_asset(
    State(state): State<Arc<DashboardState>>,
    AxumPath(path): AxumPath<String>,
) -> Response<Body> {
    serve_file_from_root(&state.web_dist.join("assets"), &path).unwrap_or_else(|| {
        json_response(
            StatusCode::NOT_FOUND,
            json!({"error": "Frontend asset not found"}),
        )
    })
}

async fn serve_plugin_asset(
    State(state): State<Arc<DashboardState>>,
    AxumPath((name, path)): AxumPath<(String, String)>,
) -> Response<Body> {
    let asset_root = state
        .plugins
        .read()
        .ok()
        .and_then(|cache| cache.asset_dirs.get(&name).cloned());
    match asset_root.and_then(|root| serve_file_from_root(&root, &path)) {
        Some(response) => response,
        None => json_response(
            StatusCode::NOT_FOUND,
            json!({"detail": "Plugin asset not found"}),
        ),
    }
}

async fn serve_spa(
    State(state): State<Arc<DashboardState>>,
    AxumPath(path): AxumPath<String>,
) -> Response<Body> {
    if !state.web_dist.exists() {
        return json_response(
            StatusCode::NOT_FOUND,
            json!({"error": "Frontend not built. Run: cd web && npm run build"}),
        );
    }

    if !path.is_empty() {
        if let Some(response) = serve_file_from_root(&state.web_dist, &path) {
            return response;
        }
    }

    let index_path = state.web_dist.join("index.html");
    match fs::read_to_string(&index_path) {
        Ok(html) => {
            let chat_js = if state.embedded_chat { "true" } else { "false" };
            let token_script = format!(
                "<script>window.__HERMES_SESSION_TOKEN__=\"{}\";window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={};</script>",
                state.token, chat_js
            );
            let body = html.replacen("</head>", &format!("{token_script}</head>"), 1);
            let mut response = Response::new(Body::from(body));
            response.headers_mut().insert(
                CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            );
            response.headers_mut().insert(
                "Cache-Control",
                HeaderValue::from_static("no-store, no-cache, must-revalidate"),
            );
            response
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

async fn pty_ws(
    State(state): State<Arc<DashboardState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(query): Query<PtyQuery>,
    ws: WebSocketUpgrade,
) -> Response<Body> {
    if !state.embedded_chat {
        return ws
            .on_failed_upgrade(|_| {})
            .on_upgrade(|socket| async move {
                let _ = socket.close().await;
            });
    }
    let Some(token) = query.token.as_deref() else {
        return websocket_close(ws, StatusCode::UNAUTHORIZED);
    };
    if token != state.token || !ws_client_allowed(&state, addr.ip()) {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }

    let resume = query.resume.clone();
    let channel = query
        .channel
        .as_deref()
        .filter(|value| state.channel_re.is_match(value))
        .map(str::to_string);
    ws.on_upgrade(move |socket| handle_pty_ws(state, socket, resume, channel))
}

async fn gateway_ws(
    State(state): State<Arc<DashboardState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(query): Query<PtyQuery>,
    ws: WebSocketUpgrade,
) -> Response<Body> {
    if !state.embedded_chat {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    let Some(token) = query.token.as_deref() else {
        return websocket_close(ws, StatusCode::UNAUTHORIZED);
    };
    if token != state.token || !ws_client_allowed(&state, addr.ip()) {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    ws.on_upgrade(move |socket| handle_gateway_ws(state, socket))
}

async fn pub_ws(
    State(state): State<Arc<DashboardState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(query): Query<TokenChannelQuery>,
    ws: WebSocketUpgrade,
) -> Response<Body> {
    if !state.embedded_chat {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    let Some(token) = query.token.as_deref() else {
        return websocket_close(ws, StatusCode::UNAUTHORIZED);
    };
    let Some(channel) = query
        .channel
        .clone()
        .filter(|value| state.channel_re.is_match(value))
    else {
        return websocket_close(ws, StatusCode::BAD_REQUEST);
    };
    if token != state.token || !ws_client_allowed(&state, addr.ip()) {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    ws.on_upgrade(move |socket| handle_pub_ws(state, socket, channel))
}

async fn events_ws(
    State(state): State<Arc<DashboardState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(query): Query<TokenChannelQuery>,
    ws: WebSocketUpgrade,
) -> Response<Body> {
    if !state.embedded_chat {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    let Some(token) = query.token.as_deref() else {
        return websocket_close(ws, StatusCode::UNAUTHORIZED);
    };
    let Some(channel) = query
        .channel
        .clone()
        .filter(|value| state.channel_re.is_match(value))
    else {
        return websocket_close(ws, StatusCode::BAD_REQUEST);
    };
    if token != state.token || !ws_client_allowed(&state, addr.ip()) {
        return websocket_close(ws, StatusCode::FORBIDDEN);
    }
    ws.on_upgrade(move |socket| handle_events_ws(state, socket, channel))
}

async fn handle_pty_ws(
    state: Arc<DashboardState>,
    socket: WebSocket,
    resume: Option<String>,
    channel: Option<String>,
) {
    let Ok((argv, cwd, env_overrides)) =
        resolve_chat_command(&state, resume.as_deref(), channel.as_deref())
    else {
        let _ = send_ws_banner(socket, "Chat unavailable: failed to resolve TUI command").await;
        return;
    };
    let Ok(mut session) = PtySession::spawn(argv, cwd, env_overrides) else {
        let _ = send_ws_banner(socket, "Chat unavailable: failed to start PTY child").await;
        return;
    };

    let resize_re = Regex::new(r"^\x1b\[RESIZE:(\d+);(\d+)\]$").expect("valid resize regex");
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    session.start_reader(tx);

    let send_task = tokio::spawn(async move {
        while let Some(chunk) = rx.recv().await {
            if sender.send(Message::Binary(chunk)).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(message)) = receiver.next().await {
        match message {
            Message::Binary(bytes) => {
                if let Some((cols, rows)) =
                    parse_resize(&resize_re, std::str::from_utf8(&bytes).ok())
                {
                    session.resize(cols, rows);
                } else {
                    session.write(&bytes);
                }
            }
            Message::Text(text) => {
                if let Some((cols, rows)) = parse_resize(&resize_re, Some(&text)) {
                    session.resize(cols, rows);
                } else {
                    session.write(text.as_bytes());
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    send_task.abort();
    session.close();
}

async fn handle_gateway_ws(state: Arc<DashboardState>, socket: WebSocket) {
    let Some(python) = resolve_repo_python(&state.project_root, Some("HERMES_DASHBOARD_PYTHON"))
    else {
        let _ = send_ws_banner(socket, "Gateway unavailable: Python interpreter not found").await;
        return;
    };

    let mut child = match TokioCommand::new(python)
        .current_dir(&state.project_root)
        .env("PYTHONPATH", state.project_root.display().to_string())
        .env(
            "HERMES_PYTHON_SRC_ROOT",
            state.project_root.display().to_string(),
        )
        .arg("-m")
        .arg("tui_gateway.entry")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            let _ =
                send_ws_banner(socket, "Gateway unavailable: failed to spawn tui_gateway").await;
            return;
        }
    };

    let Some(stdout) = child.stdout.take() else {
        let _ = send_ws_banner(socket, "Gateway unavailable: missing stdout").await;
        let _ = child.kill().await;
        return;
    };
    let Some(mut stdin) = child.stdin.take() else {
        let _ = send_ws_banner(socket, "Gateway unavailable: missing stdin").await;
        let _ = child.kill().await;
        return;
    };

    let (mut sender, mut receiver) = socket.split();
    let read_task = tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if sender.send(Message::Text(line)).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(message)) = receiver.next().await {
        match message {
            Message::Text(text) => {
                if stdin.write_all(text.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
            Message::Binary(bytes) => {
                if stdin.write_all(&bytes).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    read_task.abort();
    let _ = child.kill().await;
    let _ = child.wait().await;
}

async fn handle_pub_ws(state: Arc<DashboardState>, socket: WebSocket, channel: String) {
    let sender = event_sender(&state, &channel);
    let (_, mut receiver) = socket.split();
    while let Some(Ok(message)) = receiver.next().await {
        if let Message::Text(text) = message {
            let _ = sender.send(text);
        }
    }
}

async fn handle_events_ws(state: Arc<DashboardState>, socket: WebSocket, channel: String) {
    let sender = event_sender(&state, &channel);
    let mut receiver = sender.subscribe();
    let (mut ws_sender, mut ws_receiver) = socket.split();

    let send_task = tokio::spawn(async move {
        while let Ok(payload) = receiver.recv().await {
            if ws_sender.send(Message::Text(payload)).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(message)) = ws_receiver.next().await {
        if matches!(message, Message::Close(_)) {
            break;
        }
    }

    send_task.abort();
}

fn event_sender(state: &DashboardState, channel: &str) -> broadcast::Sender<String> {
    let _ = state;
    let mut channels = EVENT_CHANNELS.lock().expect("event channel lock");
    channels
        .entry(channel.to_string())
        .or_insert_with(|| broadcast::channel(64).0)
        .clone()
}

static EVENT_CHANNELS: std::sync::LazyLock<Mutex<HashMap<String, broadcast::Sender<String>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

struct PtySession {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    master: Arc<Mutex<Box<dyn portable_pty::MasterPty + Send>>>,
    child: Arc<Mutex<Box<dyn portable_pty::Child + Send + Sync>>>,
    reader: Option<Box<dyn Read + Send>>,
}

impl PtySession {
    fn spawn(
        argv: Vec<String>,
        cwd: Option<PathBuf>,
        env_overrides: HashMap<String, String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        let mut command = CommandBuilder::new(argv.first().ok_or("missing argv")?);
        for arg in argv.iter().skip(1) {
            command.arg(arg);
        }
        if let Some(cwd) = cwd {
            command.cwd(cwd);
        }
        for (key, value) in env_overrides {
            command.env(key, value);
        }
        let child = pair.slave.spawn_command(command)?;
        let writer = pair.master.take_writer()?;
        let reader = pair.master.try_clone_reader()?;
        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            master: Arc::new(Mutex::new(pair.master)),
            child: Arc::new(Mutex::new(child)),
            reader: Some(reader),
        })
    }

    fn start_reader(&mut self, sender: mpsc::UnboundedSender<Vec<u8>>) {
        let Some(mut reader) = self.reader.take() else {
            return;
        };
        thread::spawn(move || {
            let mut buffer = [0_u8; 65536];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(count) => {
                        if sender.send(buffer[..count].to_vec()).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
    }

    fn write(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.write_all(bytes);
            let _ = writer.flush();
        }
    }

    fn resize(&self, cols: u16, rows: u16) {
        if let Ok(master) = self.master.lock() {
            let _ = master.resize(PtySize {
                rows: rows.max(1),
                cols: cols.max(1),
                pixel_width: 0,
                pixel_height: 0,
            });
        }
    }

    fn close(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn generate_session_token() -> Result<String, Box<dyn std::error::Error>> {
    let rng = SystemRandom::new();
    let mut bytes = [0_u8; 32];
    rng.fill(&mut bytes)?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn dashboard_web_dist(project_root: &Path) -> PathBuf {
    std::env::var_os("HERMES_WEB_DIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("hermes_cli").join("web_dist"))
}

fn validate_bind_host(host: &str, allow_public: bool) -> Result<(), Box<dyn std::error::Error>> {
    if host.trim().is_empty() {
        return Err("--host must not be empty".into());
    }
    if host.chars().any(char::is_whitespace) {
        return Err("--host must not contain whitespace".into());
    }
    if !allow_public && !is_loopback_bind(host) {
        return Err(format!("Refusing to bind to {host} without --insecure").into());
    }
    Ok(())
}

fn is_loopback_bind(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn is_accepted_host(host_header: &str, bound_host: &str) -> bool {
    if bound_host == "0.0.0.0" || bound_host == "::" {
        return true;
    }
    let Some(mut host) = non_empty_trimmed(host_header) else {
        return false;
    };
    if host.starts_with('[') {
        host = host
            .split(']')
            .next()
            .unwrap_or_default()
            .trim_start_matches('[')
            .to_ascii_lowercase();
    } else {
        host = host
            .rsplit_once(':')
            .map(|(value, _)| value.to_string())
            .unwrap_or(host)
            .to_ascii_lowercase();
    }
    let bound = bound_host.to_ascii_lowercase();
    if LOOPBACK_HOSTS.contains(&bound.as_str()) {
        return LOOPBACK_HOSTS.contains(&host.as_str());
    }
    host == bound
}

fn is_public_api_path(path: &str) -> bool {
    matches!(
        path,
        "/api/status"
            | "/api/config/defaults"
            | "/api/config/schema"
            | "/api/model/info"
            | "/api/dashboard/themes"
            | "/api/dashboard/plugins"
            | "/api/dashboard/plugins/rescan"
    ) || (path.starts_with("/api/providers/oauth/") && path.contains("/poll/"))
}

fn is_websocket_api_path(path: &str) -> bool {
    matches!(path, "/api/pty" | "/api/ws" | "/api/pub" | "/api/events")
}

fn has_valid_session_token(headers: &HeaderMap, token: &str) -> bool {
    if headers
        .get(SESSION_HEADER_NAME)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == token)
    {
        return true;
    }
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("Bearer {token}"))
}

fn open_session_store(context: &HermesContext) -> Result<SessionStore, hermes_core::HermesError> {
    context.open_session_store()
}

fn resolve_session_id(
    store: &SessionStore,
    session_id: &str,
) -> Result<Option<String>, hermes_core::HermesError> {
    store.resolve_session_id(session_id)
}

fn remove_session_files(sessions_dir: &Path, session_id: &str) {
    for suffix in [".json", ".jsonl"] {
        let path = sessions_dir.join(format!("{session_id}{suffix}"));
        let _ = fs::remove_file(path);
    }
    if let Ok(entries) = fs::read_dir(sessions_dir) {
        let prefix = format!("request_dump_{session_id}_");
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with(&prefix) && name.ends_with(".json") {
                let _ = fs::remove_file(path);
            }
        }
    }
}

fn count_active_sessions(context: &HermesContext) -> Result<usize, hermes_core::HermesError> {
    let store = context.open_session_store()?;
    let sessions = store.search_sessions(None, 50, 0)?;
    let now = now_ts();
    Ok(sessions
        .into_iter()
        .filter(|session| session.ended_at.is_none() && (now - session.last_active) < 300.0)
        .count())
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

fn config_version_from_raw(value: &serde_yaml::Value) -> Option<i64> {
    value
        .get("_config_version")
        .and_then(serde_yaml::Value::as_i64)
}

fn config_version_from_json(value: &JsonValue) -> Option<i64> {
    value.get("_config_version").and_then(JsonValue::as_i64)
}

fn yaml_to_json(value: serde_yaml::Value) -> Option<JsonValue> {
    serde_json::to_value(value).ok()
}

fn json_to_yaml(value: JsonValue) -> Option<YamlValue> {
    serde_yaml::to_value(value).ok()
}

fn build_config_schema(defaults: &JsonValue) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut category_order = Vec::new();
    let Some(root) = defaults.as_object() else {
        return json!({"fields": {}, "category_order": []});
    };
    for key in root.keys() {
        category_order.push(JsonValue::String(key.clone()));
    }
    flatten_schema_fields("", defaults, &mut fields);
    json!({
        "fields": fields,
        "category_order": category_order,
    })
}

fn flatten_schema_fields(prefix: &str, value: &JsonValue, fields: &mut JsonMap<String, JsonValue>) {
    match value {
        JsonValue::Object(map) if !map.is_empty() => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_schema_fields(&path, child, fields);
            }
        }
        _ => {
            let category = prefix.split('.').next().unwrap_or("general");
            fields.insert(
                prefix.to_string(),
                json!({
                    "type": schema_type_for(value),
                    "category": category,
                    "description": "",
                }),
            );
        }
    }
}

fn schema_type_for(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "object",
        _ => "string",
    }
}

fn load_simple_env(path: &Path) -> HashMap<String, String> {
    let Ok(raw) = fs::read_to_string(path) else {
        return HashMap::new();
    };
    raw.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                return None;
            }
            let trimmed = trimmed.strip_prefix("export ").unwrap_or(trimmed);
            let (key, value) = trimmed.split_once('=')?;
            let key = key.trim();
            if !is_valid_env_key(key) {
                return None;
            }
            Some((key.to_string(), value.trim().to_string()))
        })
        .collect()
}

fn gc_oauth_sessions(state: &DashboardState) {
    let cutoff = now_unix_seconds().saturating_sub(OAUTH_SESSION_TTL_SECONDS);
    let mut sessions = state.oauth_sessions.lock().expect("oauth session lock");
    sessions.retain(|_, session| {
        session.created_at >= cutoff
            && session
                .expires_at
                .is_none_or(|expires_at| expires_at >= now_unix_seconds())
    });
}

fn new_oauth_session(provider: &str, flow: &str, expires_in: i64) -> Result<OAuthSession, String> {
    Ok(OAuthSession {
        session_id: generate_session_token().map_err(|error| error.to_string())?,
        provider: provider.to_string(),
        flow: flow.to_string(),
        created_at: now_unix_seconds(),
        status: String::from("pending"),
        error_message: None,
        expires_at: Some(now_unix_seconds().saturating_add(expires_in.max(1))),
        verifier: None,
        state_nonce: None,
        verification_url: None,
        user_code: None,
        poll_interval: None,
        device_code: None,
        device_auth_id: None,
        portal_base_url: None,
        inference_base_url: None,
        client_id: None,
        scope: None,
    })
}

fn start_anthropic_pkce(state: &DashboardState, provider: &str) -> Result<JsonValue, String> {
    let verifier = oauth_random_token(32)?;
    let challenge = oauth_code_challenge(&verifier);
    let auth_url = anthropic_authorize_url(&verifier, &challenge)?;
    let mut session = new_oauth_session(provider, "pkce", OAUTH_SESSION_TTL_SECONDS)?;
    session.verifier = Some(verifier.clone());
    session.state_nonce = Some(verifier);
    let session_id = session.session_id.clone();
    state
        .oauth_sessions
        .lock()
        .expect("oauth session lock")
        .insert(session_id.clone(), session);
    Ok(json!({
        "session_id": session_id,
        "flow": "pkce",
        "auth_url": auth_url,
        "expires_in": OAUTH_SESSION_TTL_SECONDS,
    }))
}

fn submit_anthropic_pkce(
    state: &DashboardState,
    session_id: &str,
    code_input: &str,
) -> Result<JsonValue, String> {
    let session = {
        let sessions = state.oauth_sessions.lock().expect("oauth session lock");
        sessions.get(session_id).cloned()
    }
    .ok_or_else(|| String::from("Unknown or expired session"))?;
    if session.provider != "anthropic" || session.flow != "pkce" {
        return Err(String::from("Unknown or expired session"));
    }
    if session.status != "pending" {
        return Ok(json!({
            "ok": false,
            "status": session.status,
            "message": session.error_message,
        }));
    }

    let (code, returned_state) = split_pkce_callback_code(code_input);
    if code.is_empty() {
        return Ok(json!({"ok": false, "status": "error", "message": "No code provided"}));
    }
    let token_payload = anthropic_exchange_code(
        code,
        if returned_state.is_empty() {
            session.state_nonce.as_deref().unwrap_or_default()
        } else {
            returned_state
        },
        session.verifier.as_deref().unwrap_or_default(),
    )?;
    save_anthropic_dashboard_auth(
        &state.context,
        &token_payload.access_token,
        &token_payload.refresh_token,
        token_payload.expires_at_ms,
    )?;
    mark_oauth_session(state, session_id, "approved", None);
    Ok(json!({"ok": true, "status": "approved"}))
}

fn start_device_code_flow(state: &DashboardState, provider: &str) -> Result<JsonValue, String> {
    match provider {
        "openai-codex" => start_codex_device_flow(state),
        "nous" => start_nous_device_flow(state),
        "minimax-oauth" => start_minimax_device_flow(state),
        _ => Err(format!(
            "Provider {provider} does not support device-code flow"
        )),
    }
}

fn start_codex_device_flow(state: &DashboardState) -> Result<JsonValue, String> {
    let issuer = codex_oauth_issuer();
    let response = BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?
        .post(format!(
            "{}/api/accounts/deviceauth/usercode",
            issuer.trim_end_matches('/')
        ))
        .header(CONTENT_TYPE.as_str(), "application/json")
        .json(&json!({"client_id": CODEX_OAUTH_CLIENT_ID}))
        .send()
        .map_err(|error| format!("Failed to request Codex device code: {error}"))?;
    if response.status().as_u16() != 200 {
        return Err(format!(
            "Codex device code request returned status {}.",
            response.status().as_u16()
        ));
    }
    let payload = response
        .json::<JsonValue>()
        .map_err(|error| error.to_string())?;
    let payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| String::from("Codex device code response was not a JSON object."))?;
    let user_code = json_map_string(&payload, "user_code")
        .ok_or_else(|| String::from("Codex device code response missing user_code."))?
        .to_string();
    let device_auth_id = json_map_string(&payload, "device_auth_id")
        .ok_or_else(|| String::from("Codex device code response missing device_auth_id."))?
        .to_string();
    let poll_interval = json_map_i64(&payload, "interval").unwrap_or(5).max(1);
    let expires_in = 15 * 60;
    let mut session = new_oauth_session("openai-codex", "device_code", expires_in)?;
    session.user_code = Some(user_code.clone());
    session.verification_url = Some(format!("{}/codex/device", issuer.trim_end_matches('/')));
    session.poll_interval = Some(poll_interval);
    session.device_auth_id = Some(device_auth_id);
    let session_id = session.session_id.clone();
    state
        .oauth_sessions
        .lock()
        .expect("oauth session lock")
        .insert(session_id.clone(), session);
    spawn_codex_worker(
        state.context.clone(),
        state.oauth_sessions.clone(),
        session_id.clone(),
    );
    Ok(json!({
        "session_id": session_id,
        "flow": "device_code",
        "user_code": user_code,
        "verification_url": format!("{}/codex/device", issuer.trim_end_matches('/')),
        "expires_in": expires_in,
        "poll_interval": poll_interval,
    }))
}

fn start_nous_device_flow(state: &DashboardState) -> Result<JsonValue, String> {
    let portal_base_url = nous_portal_base_url()?;
    let inference_base_url = nous_inference_base_url(&portal_base_url)?;
    let client_id = nous_client_id();
    let scope = nous_scope();
    let response = BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?
        .post(format!(
            "{}/api/oauth/device/code",
            portal_base_url.trim_end_matches('/')
        ))
        .header("Accept", "application/json")
        .form(&[("client_id", client_id.as_str()), ("scope", scope.as_str())])
        .send()
        .map_err(|error| format!("Failed to request Nous device code: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Nous device code request returned status {}.",
            response.status().as_u16()
        ));
    }
    let payload = response
        .json::<JsonValue>()
        .map_err(|error| error.to_string())?;
    let payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| String::from("Nous device code response was not a JSON object."))?;
    let device_code = json_map_string(&payload, "device_code")
        .ok_or_else(|| String::from("Nous device code response missing device_code."))?
        .to_string();
    let user_code = json_map_string(&payload, "user_code")
        .ok_or_else(|| String::from("Nous device code response missing user_code."))?
        .to_string();
    let verification_url = json_map_string(&payload, "verification_uri_complete")
        .or_else(|| json_map_string(&payload, "verification_uri"))
        .ok_or_else(|| String::from("Nous device code response missing verification_uri."))?
        .to_string();
    let expires_in = json_map_i64(&payload, "expires_in").unwrap_or(900).max(1);
    let poll_interval = json_map_i64(&payload, "interval").unwrap_or(5).clamp(1, 30);
    let mut session = new_oauth_session("nous", "device_code", expires_in)?;
    session.user_code = Some(user_code.clone());
    session.verification_url = Some(verification_url.clone());
    session.poll_interval = Some(poll_interval);
    session.device_code = Some(device_code);
    session.portal_base_url = Some(portal_base_url.clone());
    session.inference_base_url = Some(inference_base_url);
    session.client_id = Some(client_id.clone());
    session.scope = Some(scope);
    let session_id = session.session_id.clone();
    state
        .oauth_sessions
        .lock()
        .expect("oauth session lock")
        .insert(session_id.clone(), session);
    spawn_nous_worker(
        state.context.clone(),
        state.oauth_sessions.clone(),
        session_id.clone(),
    );
    Ok(json!({
        "session_id": session_id,
        "flow": "device_code",
        "user_code": user_code,
        "verification_url": verification_url,
        "expires_in": expires_in,
        "poll_interval": poll_interval,
    }))
}

fn start_minimax_device_flow(state: &DashboardState) -> Result<JsonValue, String> {
    let portal_base_url = minimax_portal_base_url()?;
    let inference_base_url = minimax_inference_base_url(&portal_base_url)?;
    let client_id = minimax_client_id();
    let scope = minimax_scope();
    let verifier = oauth_random_token(32)?;
    let challenge = oauth_code_challenge(&verifier);
    let state_nonce = oauth_random_token(16)?;
    let payload = minimax_request_user_code(
        &portal_base_url,
        &client_id,
        &scope,
        &challenge,
        &state_nonce,
    )?;
    let user_code = json_map_string(&payload, "user_code")
        .ok_or_else(|| String::from("MiniMax OAuth response missing user_code."))?
        .to_string();
    let verification_url = json_map_string(&payload, "verification_uri")
        .ok_or_else(|| String::from("MiniMax OAuth response missing verification_uri."))?
        .to_string();
    let expires_in = json_map_i64(&payload, "expired_in").unwrap_or(900).max(1);
    let poll_interval = json_map_i64(&payload, "interval").unwrap_or(2000).max(2000);
    let mut session = new_oauth_session("minimax-oauth", "device_code", expires_in)?;
    session.user_code = Some(user_code.clone());
    session.verification_url = Some(verification_url.clone());
    session.poll_interval = Some((poll_interval / 1000).max(2));
    session.verifier = Some(verifier);
    session.state_nonce = Some(state_nonce);
    session.portal_base_url = Some(portal_base_url.clone());
    session.inference_base_url = Some(inference_base_url);
    session.client_id = Some(client_id.clone());
    session.scope = Some(scope);
    let session_id = session.session_id.clone();
    state
        .oauth_sessions
        .lock()
        .expect("oauth session lock")
        .insert(session_id.clone(), session);
    spawn_minimax_worker(
        state.context.clone(),
        state.oauth_sessions.clone(),
        session_id.clone(),
    );
    Ok(json!({
        "session_id": session_id,
        "flow": "device_code",
        "user_code": user_code,
        "verification_url": verification_url,
        "expires_in": expires_in,
        "poll_interval": (poll_interval / 1000).max(2),
    }))
}

fn spawn_codex_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    thread::spawn(move || run_codex_worker(context, sessions, session_id));
}

fn run_codex_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    let Some(session) = lookup_oauth_session(&sessions, &session_id) else {
        return;
    };
    let issuer = codex_oauth_issuer();
    let poll_interval = session.poll_interval.unwrap_or(5).max(1) as u64;
    let Some(device_auth_id) = session.device_auth_id else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("missing device_auth_id"),
        );
        return;
    };
    let Some(user_code) = session.user_code else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing user_code"));
        return;
    };
    let client = match BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
            return;
        }
    };
    let deadline = session
        .expires_at
        .unwrap_or_else(|| now_unix_seconds().saturating_add(OAUTH_SESSION_TTL_SECONDS));
    let (authorization_code, code_verifier) = loop {
        if lookup_oauth_session(&sessions, &session_id).is_none() {
            return;
        }
        if now_unix_seconds() >= deadline {
            mark_oauth_session_shared(
                &sessions,
                &session_id,
                "expired",
                Some("Device code expired before approval"),
            );
            return;
        }
        let response = match client
            .post(format!(
                "{}/api/accounts/deviceauth/token",
                issuer.trim_end_matches('/')
            ))
            .header(CONTENT_TYPE.as_str(), "application/json")
            .json(&json!({
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            }))
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                mark_oauth_session_shared(
                    &sessions,
                    &session_id,
                    "error",
                    Some(&format!("Codex device auth polling failed: {error}")),
                );
                return;
            }
        };
        match response.status().as_u16() {
            200 => {
                let payload = match response.json::<JsonValue>() {
                    Ok(payload) => payload,
                    Err(error) => {
                        mark_oauth_session_shared(
                            &sessions,
                            &session_id,
                            "error",
                            Some(&error.to_string()),
                        );
                        return;
                    }
                };
                let payload = match payload.as_object() {
                    Some(payload) => payload,
                    None => {
                        mark_oauth_session_shared(
                            &sessions,
                            &session_id,
                            "error",
                            Some("Codex device auth poll response was not a JSON object."),
                        );
                        return;
                    }
                };
                let Some(authorization_code) =
                    json_map_string(payload, "authorization_code").map(ToOwned::to_owned)
                else {
                    mark_oauth_session_shared(
                        &sessions,
                        &session_id,
                        "error",
                        Some("Codex device auth response missing authorization_code."),
                    );
                    return;
                };
                let Some(code_verifier) =
                    json_map_string(payload, "code_verifier").map(ToOwned::to_owned)
                else {
                    mark_oauth_session_shared(
                        &sessions,
                        &session_id,
                        "error",
                        Some("Codex device auth response missing code_verifier."),
                    );
                    return;
                };
                break (authorization_code, code_verifier);
            }
            403 | 404 => thread::sleep(Duration::from_secs(poll_interval)),
            status => {
                mark_oauth_session_shared(
                    &sessions,
                    &session_id,
                    "error",
                    Some(&format!(
                        "Codex device auth polling returned status {status}."
                    )),
                );
                return;
            }
        }
    };
    let response = match client
        .post(codex_oauth_token_url())
        .header(CONTENT_TYPE.as_str(), "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", authorization_code.as_str()),
            (
                "redirect_uri",
                format!("{}/deviceauth/callback", issuer.trim_end_matches('/')).as_str(),
            ),
            ("client_id", CODEX_OAUTH_CLIENT_ID),
            ("code_verifier", code_verifier.as_str()),
        ])
        .send()
    {
        Ok(response) => response,
        Err(error) => {
            mark_oauth_session_shared(
                &sessions,
                &session_id,
                "error",
                Some(&format!("Codex token exchange failed: {error}")),
            );
            return;
        }
    };
    if response.status().as_u16() != 200 {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some(&format!(
                "Codex token exchange returned status {}.",
                response.status().as_u16()
            )),
        );
        return;
    }
    let payload = match response.json::<JsonValue>() {
        Ok(payload) => payload,
        Err(error) => {
            mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
            return;
        }
    };
    let payload = match payload.as_object() {
        Some(payload) => payload,
        None => {
            mark_oauth_session_shared(
                &sessions,
                &session_id,
                "error",
                Some("Codex token exchange response was not a JSON object."),
            );
            return;
        }
    };
    let Some(access_token) = json_map_string(payload, "access_token") else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("Codex token exchange did not return an access_token."),
        );
        return;
    };
    let refresh_token = json_map_string(payload, "refresh_token");
    if let Err(error) =
        save_codex_dashboard_auth(&context, access_token, refresh_token.map(str::to_string))
    {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
        return;
    }
    let _ = resolve_codex_access_token(context.hermes_home().as_path());
    mark_oauth_session_shared(&sessions, &session_id, "approved", None);
}

fn spawn_nous_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    thread::spawn(move || run_nous_worker(context, sessions, session_id));
}

fn run_nous_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    let Some(session) = lookup_oauth_session(&sessions, &session_id) else {
        return;
    };
    let Some(portal_base_url) = session.portal_base_url else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("missing portal_base_url"),
        );
        return;
    };
    let Some(inference_base_url) = session.inference_base_url else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("missing inference_base_url"),
        );
        return;
    };
    let Some(client_id) = session.client_id else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing client_id"));
        return;
    };
    let scope = session.scope.unwrap_or_else(nous_scope);
    let Some(device_code) = session.device_code else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing device_code"));
        return;
    };
    let mut poll_interval = session.poll_interval.unwrap_or(5).clamp(1, 30);
    let client = match BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
            return;
        }
    };
    let deadline = session
        .expires_at
        .unwrap_or_else(|| now_unix_seconds().saturating_add(OAUTH_SESSION_TTL_SECONDS));
    let token_payload = loop {
        if lookup_oauth_session(&sessions, &session_id).is_none() {
            return;
        }
        if now_unix_seconds() >= deadline {
            mark_oauth_session_shared(
                &sessions,
                &session_id,
                "expired",
                Some("Timed out waiting for device authorization."),
            );
            return;
        }
        let response = match client
            .post(format!(
                "{}/api/oauth/token",
                portal_base_url.trim_end_matches('/')
            ))
            .header("Accept", "application/json")
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", client_id.as_str()),
                ("device_code", device_code.as_str()),
            ])
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                mark_oauth_session_shared(
                    &sessions,
                    &session_id,
                    "error",
                    Some(&format!("Nous device auth polling failed: {error}")),
                );
                return;
            }
        };
        let status = response.status();
        let payload = response.json::<JsonValue>().unwrap_or(JsonValue::Null);
        if status.is_success() {
            let Some(payload) = payload.as_object().cloned() else {
                mark_oauth_session_shared(
                    &sessions,
                    &session_id,
                    "error",
                    Some("Nous token response was not a JSON object."),
                );
                return;
            };
            break payload;
        }
        let Some(error_payload) = payload.as_object() else {
            mark_oauth_session_shared(
                &sessions,
                &session_id,
                "error",
                Some(&format!(
                    "Nous token exchange returned status {}.",
                    status.as_u16()
                )),
            );
            return;
        };
        let error_code = json_map_string(error_payload, "error").unwrap_or_default();
        if error_code == "authorization_pending" {
            thread::sleep(Duration::from_secs(poll_interval as u64));
            continue;
        }
        if error_code == "slow_down" {
            poll_interval = (poll_interval + 1).min(30);
            thread::sleep(Duration::from_secs(poll_interval as u64));
            continue;
        }
        let description = json_map_string(error_payload, "error_description")
            .unwrap_or("Unknown authentication error");
        let message = if error_code.is_empty() {
            description.to_string()
        } else {
            format!("{error_code}: {description}")
        };
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&message));
        return;
    };
    if let Err(error) = save_nous_dashboard_auth(
        &context,
        &portal_base_url,
        &inference_base_url,
        &client_id,
        &scope,
        &token_payload,
    ) {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
        return;
    }
    if let Err(error) = resolve_nous_runtime_credentials(context.hermes_home().as_path(), 300, 15.0)
    {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
        return;
    }
    if let Err(error) = finalize_nous_dashboard_pool(&context) {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
        return;
    }
    mark_oauth_session_shared(&sessions, &session_id, "approved", None);
}

fn spawn_minimax_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    thread::spawn(move || run_minimax_worker(context, sessions, session_id));
}

fn run_minimax_worker(
    context: HermesContext,
    sessions: Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: String,
) {
    let Some(session) = lookup_oauth_session(&sessions, &session_id) else {
        return;
    };
    let Some(portal_base_url) = session.portal_base_url else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("missing portal_base_url"),
        );
        return;
    };
    let Some(inference_base_url) = session.inference_base_url else {
        mark_oauth_session_shared(
            &sessions,
            &session_id,
            "error",
            Some("missing inference_base_url"),
        );
        return;
    };
    let Some(client_id) = session.client_id else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing client_id"));
        return;
    };
    let scope = session.scope.unwrap_or_else(minimax_scope);
    let Some(user_code) = session.user_code else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing user_code"));
        return;
    };
    let Some(code_verifier) = session.verifier else {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some("missing verifier"));
        return;
    };
    let expires_in = session
        .expires_at
        .unwrap_or_else(|| now_unix_seconds().saturating_add(OAUTH_SESSION_TTL_SECONDS));
    let poll_interval_ms = session.poll_interval.unwrap_or(2).max(2) * 1000;
    let client = match BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
            return;
        }
    };
    let token_payload = match minimax_poll_token(
        &client,
        &portal_base_url,
        &client_id,
        &user_code,
        &code_verifier,
        expires_in,
        Some(poll_interval_ms),
        &sessions,
        &session_id,
    ) {
        Ok(payload) => payload,
        Err(error) => {
            mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
            return;
        }
    };
    if let Err(error) = save_minimax_dashboard_auth(
        &context,
        &portal_base_url,
        &inference_base_url,
        &client_id,
        &scope,
        &token_payload,
    ) {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
        return;
    }
    if let Err(error) = resolve_minimax_oauth_runtime_credentials(context.hermes_home().as_path()) {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error.to_string()));
        return;
    }
    if let Err(error) = finalize_minimax_dashboard_pool(&context) {
        mark_oauth_session_shared(&sessions, &session_id, "error", Some(&error));
        return;
    }
    mark_oauth_session_shared(&sessions, &session_id, "approved", None);
}

fn lookup_oauth_session(
    sessions: &Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: &str,
) -> Option<OAuthSession> {
    sessions
        .lock()
        .expect("oauth session lock")
        .get(session_id)
        .cloned()
}

fn mark_oauth_session(state: &DashboardState, session_id: &str, status: &str, error: Option<&str>) {
    mark_oauth_session_shared(&state.oauth_sessions, session_id, status, error);
}

fn mark_oauth_session_shared(
    sessions: &Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: &str,
    status: &str,
    error: Option<&str>,
) {
    if let Some(session) = sessions
        .lock()
        .expect("oauth session lock")
        .get_mut(session_id)
    {
        session.status = status.to_string();
        session.error_message = error.map(ToOwned::to_owned);
    }
}

fn oauth_random_token(byte_len: usize) -> Result<String, String> {
    let rng = SystemRandom::new();
    let mut bytes = vec![0_u8; byte_len.max(16)];
    rng.fill(&mut bytes).map_err(|error| error.to_string())?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn oauth_code_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

fn anthropic_authorize_base_url() -> String {
    env_trimmed("HERMES_AUTH_ANTHROPIC_AUTHORIZE_URL")
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_OAUTH_AUTHORIZE_URL.to_string())
}

fn anthropic_token_url() -> String {
    env_trimmed("HERMES_AUTH_ANTHROPIC_TOKEN_URL")
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_OAUTH_TOKEN_URL.to_string())
}

fn anthropic_authorize_url(state: &str, challenge: &str) -> Result<String, String> {
    let mut url =
        reqwest::Url::parse(&anthropic_authorize_base_url()).map_err(|error| error.to_string())?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("code", "true");
        query.append_pair("client_id", ANTHROPIC_OAUTH_CLIENT_ID);
        query.append_pair("response_type", "code");
        query.append_pair("redirect_uri", ANTHROPIC_OAUTH_REDIRECT_URI);
        query.append_pair("scope", ANTHROPIC_OAUTH_SCOPES);
        query.append_pair("code_challenge", challenge);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("state", state);
    }
    Ok(url.to_string())
}

fn split_pkce_callback_code(raw: &str) -> (&str, &str) {
    if let Some((code, state)) = raw.split_once('#') {
        (code.trim(), state.trim())
    } else {
        (raw.trim(), "")
    }
}

struct AnthropicOauthTokens {
    access_token: String,
    refresh_token: String,
    expires_at_ms: i64,
}

fn anthropic_exchange_code(
    code: &str,
    state: &str,
    verifier: &str,
) -> Result<AnthropicOauthTokens, String> {
    if code.trim().is_empty() {
        return Err(String::from(
            "Anthropic authorization failed: missing authorization code.",
        ));
    }
    let response = BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?
        .post(anthropic_token_url())
        .header(CONTENT_TYPE.as_str(), "application/json")
        .header("User-Agent", ANTHROPIC_OAUTH_USER_AGENT)
        .json(&json!({
            "grant_type": "authorization_code",
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": ANTHROPIC_OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }))
        .send()
        .map_err(|error| format!("Anthropic token exchange failed: {error}"))?;
    if response.status().as_u16() >= 400 {
        let detail = response.text().unwrap_or_default();
        let suffix = if detail.trim().is_empty() {
            String::new()
        } else {
            format!(" Response: {}", detail.trim())
        };
        return Err(format!("Anthropic token exchange failed.{suffix}"));
    }
    let payload = response
        .json::<JsonValue>()
        .map_err(|error| error.to_string())?;
    let payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| String::from("Anthropic token response was not a JSON object."))?;
    let access_token = json_map_string(&payload, "access_token")
        .ok_or_else(|| String::from("Anthropic token response did not include an access_token."))?
        .to_string();
    let refresh_token = json_map_string(&payload, "refresh_token")
        .unwrap_or("")
        .to_string();
    let expires_in = json_map_i64(&payload, "expires_in").unwrap_or(3600).max(0);
    Ok(AnthropicOauthTokens {
        access_token,
        refresh_token,
        expires_at_ms: now_unix_millis().saturating_add(expires_in.saturating_mul(1000)),
    })
}

fn codex_oauth_issuer() -> String {
    env_trimmed("HERMES_AUTH_CODEX_ISSUER")
        .unwrap_or_else(|| DEFAULT_CODEX_OAUTH_ISSUER.to_string())
}

fn codex_oauth_token_url() -> String {
    env_trimmed("HERMES_AUTH_CODEX_TOKEN_URL")
        .unwrap_or_else(|| DEFAULT_CODEX_OAUTH_TOKEN_URL.to_string())
}

fn codex_base_url() -> String {
    env_trimmed("HERMES_CODEX_BASE_URL")
        .unwrap_or_else(|| String::from("https://chatgpt.com/backend-api/codex"))
}

fn codex_now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339().replace("+00:00", "Z")
}

fn env_trimmed(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_http_url(value: &str, label: &str) -> Result<String, String> {
    let parsed =
        reqwest::Url::parse(value.trim()).map_err(|error| format!("invalid {label}: {error}"))?;
    match parsed.scheme() {
        "http" | "https" => Ok(parsed.to_string().trim_end_matches('/').to_string()),
        _ => Err(format!("{label} must use http or https")),
    }
}

fn nous_portal_base_url() -> Result<String, String> {
    normalize_http_url(
        &env_trimmed("HERMES_PORTAL_BASE_URL")
            .or_else(|| env_trimmed("NOUS_PORTAL_BASE_URL"))
            .unwrap_or_else(|| DEFAULT_NOUS_PORTAL_URL.to_string()),
        "portal URL",
    )
}

fn nous_inference_base_url(portal_base_url: &str) -> Result<String, String> {
    normalize_http_url(
        &env_trimmed("HERMES_AUTH_NOUS_INFERENCE_URL").unwrap_or_else(|| match portal_base_url {
            DEFAULT_NOUS_PORTAL_URL => DEFAULT_NOUS_INFERENCE_URL.to_string(),
            other => format!("{}/v1", other.trim_end_matches('/')),
        }),
        "inference URL",
    )
}

fn nous_client_id() -> String {
    env_trimmed("HERMES_AUTH_NOUS_CLIENT_ID").unwrap_or_else(|| DEFAULT_NOUS_CLIENT_ID.to_string())
}

fn nous_scope() -> String {
    env_trimmed("HERMES_AUTH_NOUS_SCOPE").unwrap_or_else(|| DEFAULT_NOUS_SCOPE.to_string())
}

fn minimax_portal_base_url() -> Result<String, String> {
    normalize_http_url(
        &env_trimmed("HERMES_AUTH_MINIMAX_PORTAL_URL")
            .unwrap_or_else(|| DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL.to_string()),
        "portal URL",
    )
}

fn minimax_inference_base_url(portal_base_url: &str) -> Result<String, String> {
    let value =
        env_trimmed("HERMES_AUTH_MINIMAX_INFERENCE_URL").unwrap_or_else(|| match portal_base_url {
            DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL => {
                DEFAULT_MINIMAX_OAUTH_CN_INFERENCE_BASE_URL.to_string()
            }
            DEFAULT_MINIMAX_OAUTH_PORTAL_BASE_URL => {
                DEFAULT_MINIMAX_OAUTH_INFERENCE_BASE_URL.to_string()
            }
            other => format!("{}/anthropic", other.trim_end_matches('/')),
        });
    normalize_http_url(&value, "inference URL")
}

fn minimax_client_id() -> String {
    env_trimmed("HERMES_AUTH_MINIMAX_CLIENT_ID")
        .unwrap_or_else(|| MINIMAX_OAUTH_CLIENT_ID.to_string())
}

fn minimax_scope() -> String {
    env_trimmed("HERMES_AUTH_MINIMAX_SCOPE").unwrap_or_else(|| MINIMAX_OAUTH_SCOPE.to_string())
}

fn minimax_request_user_code(
    portal_base_url: &str,
    client_id: &str,
    scope: &str,
    challenge: &str,
    state: &str,
) -> Result<JsonMap<String, JsonValue>, String> {
    let response = BlockingClient::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?
        .post(format!(
            "{}/oauth/code",
            portal_base_url.trim_end_matches('/')
        ))
        .header(CONTENT_TYPE.as_str(), "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .header(
            "x-request-id",
            oauth_random_token(12).unwrap_or_else(|_| String::from("hermes-minimax")),
        )
        .form(&[
            ("response_type", "code"),
            ("client_id", client_id),
            ("scope", scope),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
        ])
        .send()
        .map_err(|error| format!("MiniMax OAuth authorization failed: {error}"))?;
    let status = response.status();
    let body = response.text().unwrap_or_default();
    if !status.is_success() {
        let detail = if body.trim().is_empty() {
            format!("status {}", status.as_u16())
        } else {
            body.trim().to_string()
        };
        return Err(format!("MiniMax OAuth authorization failed: {detail}"));
    }
    let payload = serde_json::from_str::<JsonValue>(&body).map_err(|error| {
        format!("MiniMax OAuth authorization response was invalid JSON: {error}")
    })?;
    let payload = payload.as_object().cloned().ok_or_else(|| {
        String::from("MiniMax OAuth authorization response was not a JSON object.")
    })?;
    for field in ["user_code", "verification_uri", "expired_in"] {
        if !payload.contains_key(field) {
            return Err(format!("MiniMax OAuth response missing field: {field}"));
        }
    }
    if json_map_string(&payload, "state") != Some(state) {
        return Err(String::from(
            "MiniMax OAuth state mismatch (possible CSRF).",
        ));
    }
    Ok(payload)
}

fn minimax_poll_token(
    client: &BlockingClient,
    portal_base_url: &str,
    client_id: &str,
    user_code: &str,
    code_verifier: &str,
    expires_at: i64,
    interval_ms: Option<i64>,
    sessions: &Arc<Mutex<HashMap<String, OAuthSession>>>,
    session_id: &str,
) -> Result<JsonMap<String, JsonValue>, String> {
    let interval = Duration::from_millis(interval_ms.unwrap_or(2000).max(2000) as u64);
    while now_unix_seconds() < expires_at {
        if lookup_oauth_session(sessions, session_id).is_none() {
            return Err(String::from("cancelled"));
        }
        let response = client
            .post(format!(
                "{}/oauth/token",
                portal_base_url.trim_end_matches('/')
            ))
            .header(CONTENT_TYPE.as_str(), "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .form(&[
                ("grant_type", MINIMAX_OAUTH_GRANT_TYPE),
                ("client_id", client_id),
                ("user_code", user_code),
                ("code_verifier", code_verifier),
            ])
            .send()
            .map_err(|error| format!("MiniMax OAuth token exchange failed: {error}"))?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        let payload = serde_json::from_str::<JsonValue>(&body).unwrap_or(JsonValue::Null);
        let payload = payload.as_object().cloned().unwrap_or_default();
        if status.as_u16() != 200 {
            let detail = payload
                .get("base_resp")
                .and_then(JsonValue::as_object)
                .and_then(|base_resp| json_map_string(base_resp, "status_msg"))
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| body.trim().to_string());
            return Err(format!(
                "MiniMax OAuth error: {}",
                if detail.is_empty() {
                    "unknown"
                } else {
                    detail.as_str()
                }
            ));
        }
        match json_map_string(&payload, "status") {
            Some("success") => return Ok(payload),
            Some("error") => {
                return Err(String::from(
                    "MiniMax OAuth reported an error. Please try again later.",
                ));
            }
            _ => thread::sleep(interval),
        }
    }
    Err(String::from(
        "MiniMax OAuth timed out before authorization completed.",
    ))
}

fn save_anthropic_dashboard_auth(
    context: &HermesContext,
    access_token: &str,
    refresh_token: &str,
    expires_at_ms: i64,
) -> Result<(), String> {
    let payload = json!({
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    });
    let path = context.hermes_home().join(".anthropic_oauth.json");
    write_json_file(&path, &payload)?;
    upsert_auth_pool_entry(
        context.hermes_home().as_path(),
        "anthropic",
        "manual:hermes_pkce",
        label_from_token(access_token, "anthropic"),
        access_token.to_string(),
        Some(refresh_token.to_string()),
        None,
        Some(expires_at_ms),
        None,
    )?;
    Ok(())
}

fn save_codex_dashboard_auth(
    context: &HermesContext,
    access_token: &str,
    refresh_token: Option<String>,
) -> Result<(), String> {
    let mut auth_store = load_auth_store_json_map(context.hermes_home().as_path())?;
    let mut state = JsonMap::new();
    let mut tokens = JsonMap::new();
    tokens.insert(
        String::from("access_token"),
        JsonValue::String(access_token.to_string()),
    );
    if let Some(refresh_token) = refresh_token
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        tokens.insert(
            String::from("refresh_token"),
            JsonValue::String(refresh_token.to_string()),
        );
    }
    state.insert(String::from("tokens"), JsonValue::Object(tokens));
    state.insert(
        String::from("auth_mode"),
        JsonValue::String(String::from("chatgpt")),
    );
    state.insert(
        String::from("last_refresh"),
        JsonValue::String(codex_now_rfc3339()),
    );
    store_provider_state_json(&mut auth_store, "openai-codex", state)?;
    save_auth_store_json_map(context.hermes_home().as_path(), &auth_store)?;
    upsert_auth_pool_entry(
        context.hermes_home().as_path(),
        "openai-codex",
        "manual:device_code",
        label_from_token(access_token, "openai-codex"),
        access_token.to_string(),
        refresh_token,
        Some(codex_base_url()),
        None,
        Some(codex_now_rfc3339()),
    )?;
    Ok(())
}

fn save_nous_dashboard_auth(
    context: &HermesContext,
    portal_base_url: &str,
    requested_inference_url: &str,
    client_id: &str,
    scope: &str,
    token_payload: &JsonMap<String, JsonValue>,
) -> Result<(), String> {
    let obtained_at = chrono::Utc::now();
    let token_expires_in = json_map_i64(token_payload, "expires_in")
        .unwrap_or(0)
        .max(0);
    let resolved_inference_url = json_map_string(token_payload, "inference_base_url")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| requested_inference_url.to_string());
    let mut state = JsonMap::new();
    state.insert(
        String::from("portal_base_url"),
        JsonValue::String(portal_base_url.to_string()),
    );
    state.insert(
        String::from("inference_base_url"),
        JsonValue::String(resolved_inference_url),
    );
    state.insert(
        String::from("client_id"),
        JsonValue::String(client_id.to_string()),
    );
    state.insert(
        String::from("scope"),
        JsonValue::String(
            json_map_string(token_payload, "scope")
                .unwrap_or(scope)
                .to_string(),
        ),
    );
    state.insert(
        String::from("token_type"),
        JsonValue::String(
            json_map_string(token_payload, "token_type")
                .unwrap_or("Bearer")
                .to_string(),
        ),
    );
    state.insert(
        String::from("access_token"),
        JsonValue::String(
            json_map_string(token_payload, "access_token")
                .ok_or_else(|| String::from("Nous token response missing access_token."))?
                .to_string(),
        ),
    );
    if let Some(refresh_token) = json_map_string(token_payload, "refresh_token") {
        state.insert(
            String::from("refresh_token"),
            JsonValue::String(refresh_token.to_string()),
        );
    }
    state.insert(
        String::from("obtained_at"),
        JsonValue::String(obtained_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    state.insert(
        String::from("expires_at"),
        JsonValue::String(
            (obtained_at + chrono::Duration::seconds(token_expires_in))
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        ),
    );
    state.insert(
        String::from("expires_in"),
        JsonValue::from(token_expires_in),
    );
    state.insert(String::from("agent_key"), JsonValue::Null);
    state.insert(String::from("agent_key_id"), JsonValue::Null);
    state.insert(String::from("agent_key_expires_at"), JsonValue::Null);
    state.insert(String::from("agent_key_expires_in"), JsonValue::Null);
    state.insert(String::from("agent_key_reused"), JsonValue::Null);
    state.insert(String::from("agent_key_obtained_at"), JsonValue::Null);
    let mut auth_store = load_auth_store_json_map(context.hermes_home().as_path())?;
    store_provider_state_json(&mut auth_store, "nous", state)?;
    save_auth_store_json_map(context.hermes_home().as_path(), &auth_store)
}

fn finalize_nous_dashboard_pool(context: &HermesContext) -> Result<(), String> {
    let state = auth_provider_state(context, "nous")?
        .ok_or_else(|| String::from("Nous auth state is missing after runtime resolution."))?;
    write_shared_nous_state(&state, context.home_dir())?;
    let access_token = json_map_string(&state, "access_token").ok_or_else(|| {
        String::from("Nous auth state is missing access_token after runtime resolution.")
    })?;
    upsert_auth_pool_entry(
        context.hermes_home().as_path(),
        "nous",
        "device_code",
        state
            .get("label")
            .and_then(JsonValue::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| label_from_token(access_token, "nous")),
        access_token.to_string(),
        json_map_string(&state, "refresh_token").map(ToOwned::to_owned),
        json_map_string(&state, "inference_base_url").map(ToOwned::to_owned),
        None,
        None,
    )?;
    Ok(())
}

fn write_shared_nous_state(
    state: &JsonMap<String, JsonValue>,
    home_dir: &Path,
) -> Result<(), String> {
    let Some(access_token) = json_map_string(state, "access_token") else {
        return Ok(());
    };
    let Some(refresh_token) = json_map_string(state, "refresh_token") else {
        return Ok(());
    };
    let payload = json!({
        "_schema": 1,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": json_map_string(state, "token_type").unwrap_or("Bearer"),
        "scope": json_map_string(state, "scope").unwrap_or(DEFAULT_NOUS_SCOPE),
        "client_id": json_map_string(state, "client_id").unwrap_or(DEFAULT_NOUS_CLIENT_ID),
        "portal_base_url": json_map_string(state, "portal_base_url").unwrap_or(DEFAULT_NOUS_PORTAL_URL),
        "inference_base_url": json_map_string(state, "inference_base_url").unwrap_or(DEFAULT_NOUS_INFERENCE_URL),
        "obtained_at": state.get("obtained_at").cloned().unwrap_or(JsonValue::Null),
        "expires_at": state.get("expires_at").cloned().unwrap_or(JsonValue::Null),
        "updated_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
    write_json_file(&nous_shared_auth_path(home_dir), &payload)
}

fn save_minimax_dashboard_auth(
    context: &HermesContext,
    portal_base_url: &str,
    inference_base_url: &str,
    client_id: &str,
    scope: &str,
    token_payload: &JsonMap<String, JsonValue>,
) -> Result<(), String> {
    let access_token = json_map_string(token_payload, "access_token")
        .ok_or_else(|| String::from("MiniMax OAuth token payload missing access_token."))?;
    let refresh_token = json_map_string(token_payload, "refresh_token")
        .ok_or_else(|| String::from("MiniMax OAuth token payload missing refresh_token."))?;
    let expires_in = json_map_i64(token_payload, "expired_in")
        .ok_or_else(|| String::from("MiniMax OAuth token payload missing expired_in."))?
        .max(1);
    let obtained_at = chrono::Utc::now();
    let expires_at = (obtained_at + chrono::Duration::seconds(expires_in))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut state = JsonMap::new();
    state.insert(
        String::from("provider"),
        JsonValue::String(String::from("minimax-oauth")),
    );
    state.insert(
        String::from("region"),
        JsonValue::String(
            if portal_base_url == DEFAULT_MINIMAX_OAUTH_CN_PORTAL_BASE_URL {
                "cn"
            } else {
                "global"
            }
            .to_string(),
        ),
    );
    state.insert(
        String::from("portal_base_url"),
        JsonValue::String(portal_base_url.to_string()),
    );
    state.insert(
        String::from("inference_base_url"),
        JsonValue::String(inference_base_url.to_string()),
    );
    state.insert(
        String::from("client_id"),
        JsonValue::String(client_id.to_string()),
    );
    state.insert(String::from("scope"), JsonValue::String(scope.to_string()));
    state.insert(
        String::from("access_token"),
        JsonValue::String(access_token.to_string()),
    );
    state.insert(
        String::from("refresh_token"),
        JsonValue::String(refresh_token.to_string()),
    );
    state.insert(
        String::from("obtained_at"),
        JsonValue::String(obtained_at.to_rfc3339()),
    );
    state.insert(String::from("expires_at"), JsonValue::String(expires_at));
    state.insert(String::from("expires_in"), JsonValue::from(expires_in));
    if let Some(token_type) = json_map_string(token_payload, "token_type") {
        state.insert(
            String::from("token_type"),
            JsonValue::String(token_type.to_string()),
        );
    }
    if let Some(resource_url) = json_map_string(token_payload, "resource_url") {
        state.insert(
            String::from("resource_url"),
            JsonValue::String(resource_url.to_string()),
        );
    }
    let mut auth_store = load_auth_store_json_map(context.hermes_home().as_path())?;
    store_provider_state_json(&mut auth_store, "minimax-oauth", state)?;
    save_auth_store_json_map(context.hermes_home().as_path(), &auth_store)
}

fn finalize_minimax_dashboard_pool(context: &HermesContext) -> Result<(), String> {
    let state = auth_provider_state(context, "minimax-oauth")?
        .ok_or_else(|| String::from("MiniMax auth state is missing after runtime resolution."))?;
    let access_token = json_map_string(&state, "access_token")
        .ok_or_else(|| String::from("MiniMax auth state is missing access_token."))?;
    upsert_auth_pool_entry(
        context.hermes_home().as_path(),
        "minimax-oauth",
        "manual:minimax_oauth",
        label_from_token(access_token, "minimax-oauth"),
        access_token.to_string(),
        json_map_string(&state, "refresh_token").map(ToOwned::to_owned),
        json_map_string(&state, "inference_base_url").map(ToOwned::to_owned),
        state
            .get("expires_at")
            .and_then(JsonValue::as_str)
            .and_then(parse_rfc3339_to_unix_ms),
        None,
    )?;
    Ok(())
}

fn load_auth_store_json_map(hermes_home: &Path) -> Result<JsonMap<String, JsonValue>, String> {
    let path = hermes_home.join("auth.json");
    if !path.exists() {
        let mut root = JsonMap::new();
        root.insert(String::from("version"), JsonValue::from(1));
        return Ok(root);
    }
    read_json_object_file(&path)
}

fn save_auth_store_json_map(
    hermes_home: &Path,
    root: &JsonMap<String, JsonValue>,
) -> Result<(), String> {
    write_json_file(
        &hermes_home.join("auth.json"),
        &JsonValue::Object(root.clone()),
    )
}

fn store_provider_state_json(
    root: &mut JsonMap<String, JsonValue>,
    provider: &str,
    state: JsonMap<String, JsonValue>,
) -> Result<(), String> {
    if !root.contains_key("providers") {
        root.insert(String::from("providers"), JsonValue::Object(JsonMap::new()));
    }
    let providers = root
        .get_mut("providers")
        .and_then(JsonValue::as_object_mut)
        .ok_or_else(|| String::from("providers is not a JSON object"))?;
    providers.insert(provider.to_string(), JsonValue::Object(state));
    Ok(())
}

fn upsert_auth_pool_entry(
    hermes_home: &Path,
    provider: &str,
    source: &str,
    label: String,
    access_token: String,
    refresh_token: Option<String>,
    base_url: Option<String>,
    expires_at_ms: Option<i64>,
    last_refresh: Option<String>,
) -> Result<usize, String> {
    let mut auth_store = load_auth_store_json_map(hermes_home)?;
    clear_provider_suppressions_json(&mut auth_store, provider);
    if !auth_store.contains_key("credential_pool") {
        auth_store.insert(
            String::from("credential_pool"),
            JsonValue::Object(JsonMap::new()),
        );
    }
    let pool = auth_store
        .get_mut("credential_pool")
        .and_then(JsonValue::as_object_mut)
        .ok_or_else(|| String::from("credential_pool is not a JSON object"))?;
    let entries = pool
        .entry(provider.to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| format!("credential_pool.{provider} is not an array"))?;
    entries.retain(|entry| {
        entry
            .get("source")
            .and_then(JsonValue::as_str)
            .is_none_or(|existing| existing != source)
    });
    let mut entry = JsonMap::new();
    entry.insert(String::from("label"), JsonValue::String(label));
    entry.insert(
        String::from("auth_type"),
        JsonValue::String(String::from("oauth")),
    );
    entry.insert(
        String::from("source"),
        JsonValue::String(source.to_string()),
    );
    entry.insert(
        String::from("access_token"),
        JsonValue::String(access_token),
    );
    if let Some(refresh_token) = refresh_token.filter(|value| !value.trim().is_empty()) {
        entry.insert(
            String::from("refresh_token"),
            JsonValue::String(refresh_token),
        );
    }
    if let Some(base_url) = base_url.filter(|value| !value.trim().is_empty()) {
        entry.insert(String::from("base_url"), JsonValue::String(base_url));
    }
    if let Some(expires_at_ms) = expires_at_ms.filter(|value| *value > 0) {
        entry.insert(
            String::from("expires_at_ms"),
            JsonValue::from(expires_at_ms),
        );
    }
    if let Some(last_refresh) = last_refresh.filter(|value| !value.trim().is_empty()) {
        entry.insert(
            String::from("last_refresh"),
            JsonValue::String(last_refresh),
        );
    }
    entries.push(JsonValue::Object(entry));
    let count = entries.len();
    save_auth_store_json_map(hermes_home, &auth_store)?;
    Ok(count)
}

fn clear_provider_suppressions_json(root: &mut JsonMap<String, JsonValue>, provider: &str) {
    let Some(suppressed) = root
        .get_mut("suppressed_sources")
        .and_then(JsonValue::as_object_mut)
    else {
        return;
    };
    suppressed.remove(provider);
    if suppressed.is_empty() {
        root.remove("suppressed_sources");
    }
}

fn write_json_file(path: &Path, value: &JsonValue) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    fs::write(path, format!("{payload}\n")).map_err(|error| format!("{}: {error}", path.display()))
}

fn label_from_token(token: &str, fallback: &str) -> String {
    let Some(payload) = token.split('.').nth(1) else {
        return fallback.to_string();
    };
    let Ok(decoded) = URL_SAFE_NO_PAD.decode(payload.as_bytes()) else {
        return fallback.to_string();
    };
    let Ok(JsonValue::Object(claims)) = serde_json::from_slice::<JsonValue>(&decoded) else {
        return fallback.to_string();
    };
    for key in ["email", "preferred_username", "upn"] {
        if let Some(value) = claims.get(key).and_then(JsonValue::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    fallback.to_string()
}

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs() as i64)
        .unwrap_or(0)
}

fn now_unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or(0)
}

fn parse_rfc3339_to_unix_ms(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.timestamp_millis())
}

fn oauth_provider_status(context: &HermesContext, provider: &str) -> JsonValue {
    let result = match provider {
        "anthropic" => oauth_status_anthropic(context),
        "claude-code" => oauth_status_claude_code(context),
        "nous" => oauth_status_nous(context),
        "openai-codex" => oauth_status_openai_codex(context),
        "qwen-oauth" => oauth_status_qwen(context),
        "minimax-oauth" => oauth_status_minimax(context),
        _ => Ok(json!({"logged_in": false})),
    };
    match result {
        Ok(status) => status,
        Err(error) => json!({"logged_in": false, "error": error}),
    }
}

fn oauth_status_anthropic(context: &HermesContext) -> Result<JsonValue, String> {
    let hermes_oauth_path = context.hermes_home().join(".anthropic_oauth.json");
    if let Ok(payload) = read_json_object_file(&hermes_oauth_path)
        && let Some(access_token) = json_map_string(&payload, "accessToken")
    {
        return Ok(json!({
            "logged_in": true,
            "source": "hermes_pkce",
            "source_label": format!("Hermes PKCE ({})", hermes_oauth_path.display()),
            "token_preview": truncate_token_preview(access_token),
            "expires_at": json_map_iso8601(&payload, "expiresAt"),
            "has_refresh_token": json_map_string(&payload, "refreshToken").is_some(),
        }));
    }

    let claude_code_path = claude_code_credentials_path(context.home_dir());
    if let Some(payload) = read_claude_code_payload(&claude_code_path)?
        && let Some(access_token) = json_map_string(&payload, "accessToken")
    {
        return Ok(json!({
            "logged_in": true,
            "source": "claude_code",
            "source_label": "Claude Code (~/.claude/.credentials.json)",
            "token_preview": truncate_token_preview(access_token),
            "expires_at": json_map_iso8601(&payload, "expiresAt"),
            "has_refresh_token": json_map_string(&payload, "refreshToken").is_some(),
        }));
    }

    let env_file = load_simple_env(&context.env_path());
    let env_token = env_file
        .get("ANTHROPIC_TOKEN")
        .cloned()
        .or_else(|| env_file.get("CLAUDE_CODE_OAUTH_TOKEN").cloned())
        .or_else(|| std::env::var("ANTHROPIC_TOKEN").ok())
        .or_else(|| std::env::var("CLAUDE_CODE_OAUTH_TOKEN").ok());
    if let Some(access_token) = env_token.filter(|value| !value.trim().is_empty()) {
        return Ok(json!({
            "logged_in": true,
            "source": "env_var",
            "source_label": "ANTHROPIC_TOKEN environment variable",
            "token_preview": truncate_token_preview(&access_token),
            "expires_at": JsonValue::Null,
            "has_refresh_token": false,
        }));
    }

    Ok(json!({"logged_in": false, "source": JsonValue::Null}))
}

fn oauth_status_claude_code(context: &HermesContext) -> Result<JsonValue, String> {
    let claude_code_path = claude_code_credentials_path(context.home_dir());
    if let Some(payload) = read_claude_code_payload(&claude_code_path)?
        && let Some(access_token) = json_map_string(&payload, "accessToken")
    {
        return Ok(json!({
            "logged_in": true,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": truncate_token_preview(access_token),
            "expires_at": json_map_iso8601(&payload, "expiresAt"),
            "has_refresh_token": json_map_string(&payload, "refreshToken").is_some(),
        }));
    }
    Ok(json!({"logged_in": false, "source": JsonValue::Null}))
}

fn oauth_status_nous(context: &HermesContext) -> Result<JsonValue, String> {
    if let Some(payload) = auth_provider_state(context, "nous")? {
        let access_token = json_map_string(&payload, "access_token");
        let refresh_token = json_map_string(&payload, "refresh_token");
        let agent_key = json_map_string(&payload, "agent_key");
        return Ok(json!({
            "logged_in": access_token.is_some() || refresh_token.is_some() || agent_key.is_some(),
            "source": "nous_portal",
            "source_label": json_map_string(&payload, "portal_base_url").unwrap_or("Nous Portal"),
            "token_preview": access_token.map(truncate_token_preview),
            "expires_at": payload.get("expires_at").cloned().unwrap_or(JsonValue::Null),
            "has_refresh_token": refresh_token.is_some(),
        }));
    }

    let shared_path = nous_shared_auth_path(context.home_dir());
    if let Ok(payload) = read_json_object_file(&shared_path) {
        let access_token = json_map_string(&payload, "access_token");
        let refresh_token = json_map_string(&payload, "refresh_token");
        return Ok(json!({
            "logged_in": access_token.is_some() || refresh_token.is_some(),
            "source": "nous_portal",
            "source_label": json_map_string(&payload, "portal_base_url").unwrap_or("Nous Portal"),
            "token_preview": access_token.map(truncate_token_preview),
            "expires_at": payload.get("expires_at").cloned().unwrap_or(JsonValue::Null),
            "has_refresh_token": refresh_token.is_some(),
        }));
    }

    Ok(json!({"logged_in": false}))
}

fn oauth_status_openai_codex(context: &HermesContext) -> Result<JsonValue, String> {
    let Some(payload) = auth_provider_state(context, "openai-codex")? else {
        return Ok(json!({"logged_in": false}));
    };
    let tokens = payload
        .get("tokens")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    let access_token = json_map_string(&tokens, "access_token");
    let refresh_token = json_map_string(&tokens, "refresh_token");
    Ok(json!({
        "logged_in": access_token.is_some() || refresh_token.is_some(),
        "source": json_map_string(&payload, "auth_mode").unwrap_or("openai_codex"),
        "source_label": json_map_string(&payload, "auth_mode").unwrap_or("OpenAI Codex"),
        "token_preview": access_token.map(truncate_token_preview),
        "expires_at": JsonValue::Null,
        "has_refresh_token": false,
        "last_refresh": payload.get("last_refresh").cloned().unwrap_or(JsonValue::Null),
    }))
}

fn oauth_status_qwen(context: &HermesContext) -> Result<JsonValue, String> {
    let path = qwen_oauth_creds_path(context.home_dir());
    if !path.exists() {
        return Ok(json!({"logged_in": false}));
    }
    let payload = read_json_object_file(&path)?;
    let access_token = json_map_string(&payload, "access_token");
    let refresh_token = json_map_string(&payload, "refresh_token");
    Ok(json!({
        "logged_in": access_token.is_some() || refresh_token.is_some(),
        "source": "qwen_cli",
        "source_label": path.display().to_string(),
        "token_preview": access_token.map(truncate_token_preview),
        "expires_at": json_map_iso8601(&payload, "expiry_date"),
        "has_refresh_token": refresh_token.is_some(),
    }))
}

fn oauth_status_minimax(context: &HermesContext) -> Result<JsonValue, String> {
    let Some(payload) = auth_provider_state(context, "minimax-oauth")? else {
        return Ok(json!({"logged_in": false}));
    };
    let access_token = json_map_string(&payload, "access_token");
    let refresh_token = json_map_string(&payload, "refresh_token");
    let portal = json_map_string(&payload, "portal_base_url").unwrap_or_default();
    let region = if portal.contains("minimaxi.com") {
        "cn"
    } else {
        "global"
    };
    Ok(json!({
        "logged_in": access_token.is_some() || refresh_token.is_some(),
        "source": "minimax_oauth",
        "source_label": format!("MiniMax ({region})"),
        "token_preview": JsonValue::Null,
        "expires_at": payload.get("expires_at").cloned().unwrap_or(JsonValue::Null),
        "has_refresh_token": refresh_token.is_some(),
    }))
}

fn auth_provider_state(
    context: &HermesContext,
    provider: &str,
) -> Result<Option<JsonMap<String, JsonValue>>, String> {
    let auth_path = context.hermes_home().join("auth.json");
    if !auth_path.exists() {
        return Ok(None);
    }
    let payload = read_json_object_file(&auth_path)?;
    Ok(payload
        .get("providers")
        .and_then(JsonValue::as_object)
        .and_then(|providers| providers.get(provider))
        .and_then(JsonValue::as_object)
        .cloned())
}

fn read_claude_code_payload(path: &Path) -> Result<Option<JsonMap<String, JsonValue>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let payload = read_json_object_file(path)?;
    Ok(payload
        .get("claudeAiOauth")
        .and_then(JsonValue::as_object)
        .cloned())
}

fn read_json_object_file(path: &Path) -> Result<JsonMap<String, JsonValue>, String> {
    let raw = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value = serde_json::from_str::<JsonValue>(&raw)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{} does not contain a JSON object", path.display()))
}

fn json_map_string<'a>(map: &'a JsonMap<String, JsonValue>, key: &str) -> Option<&'a str> {
    map.get(key)
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn json_map_i64(map: &JsonMap<String, JsonValue>, key: &str) -> Option<i64> {
    match map.get(key) {
        Some(JsonValue::Number(number)) => number.as_i64(),
        Some(JsonValue::String(value)) => value.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn json_map_iso8601(map: &JsonMap<String, JsonValue>, key: &str) -> JsonValue {
    if let Some(value) = json_map_string(map, key) {
        return JsonValue::String(value.to_string());
    }
    if let Some(value) = json_map_i64(map, key)
        && let Some(rendered) = unix_ms_to_rfc3339(value)
    {
        return JsonValue::String(rendered);
    }
    JsonValue::Null
}

fn unix_ms_to_rfc3339(value: i64) -> Option<String> {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(value)
        .map(|timestamp| timestamp.to_rfc3339())
}

fn truncate_token_preview(value: &str) -> String {
    let mut trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains('.') && trimmed.matches('.').count() >= 2 {
        trimmed = trimmed.rsplit('.').next().unwrap_or(trimmed);
    }
    if trimmed.len() <= 6 {
        trimmed.to_string()
    } else {
        format!("...{}", &trimmed[trimmed.len() - 6..])
    }
}

fn claude_code_credentials_path(home_dir: &Path) -> PathBuf {
    home_dir.join(".claude").join(".credentials.json")
}

fn qwen_oauth_creds_path(home_dir: &Path) -> PathBuf {
    home_dir.join(".qwen").join("oauth_creds.json")
}

fn nous_shared_auth_path(home_dir: &Path) -> PathBuf {
    std::env::var("HERMES_SHARED_AUTH_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir.join(".hermes").join("shared"))
        .join("nous_auth.json")
}

fn log_file_name(file: &str) -> Option<&'static str> {
    match file {
        "agent" => Some("agent.log"),
        "errors" => Some("errors.log"),
        "gateway" => Some("gateway.log"),
        _ => None,
    }
}

fn normalize_log_level(value: Option<&str>) -> Option<&'static str> {
    let normalized = value?.trim().to_ascii_uppercase();
    match normalized.as_str() {
        "" | "ALL" => None,
        "DEBUG" => Some("DEBUG"),
        "INFO" => Some("INFO"),
        "WARNING" => Some("WARNING"),
        "ERROR" => Some("ERROR"),
        "CRITICAL" => Some("CRITICAL"),
        _ => None,
    }
}

fn normalize_log_component(value: Option<&str>) -> Result<Option<&'static [&'static str]>, String> {
    let Some(component) = value.map(str::trim) else {
        return Ok(None);
    };
    if component.is_empty() || component.eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    let Some((_, prefixes)) = LOG_COMPONENT_PREFIXES
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(component))
    else {
        let available = LOG_COMPONENT_PREFIXES
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unknown component: {component}. Available: {available}"
        ));
    };
    Ok(Some(*prefixes))
}

fn launch_profile_terminal(command: &str) -> Result<(), Box<dyn std::error::Error>> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("command is required".into());
    }

    if let Some(override_bin) = std::env::var_os("HERMES_DASHBOARD_TERMINAL_BIN") {
        let status = StdCommand::new(override_bin).arg(trimmed).spawn()?;
        let _ = status.id();
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        StdCommand::new("cmd.exe")
            .args(["/c", "start", "", trimmed])
            .spawn()?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        let escaped = trimmed.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(
            "tell application \"Terminal\"\nactivate\ndo script \"{}\"\nend tell",
            escaped
        );
        StdCommand::new("osascript").arg("-e").arg(script).spawn()?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let candidates: &[(&str, &[&str])] = &[
            ("x-terminal-emulator", &["-e", "sh", "-lc"]),
            ("gnome-terminal", &["--", "sh", "-lc"]),
            ("konsole", &["-e", "sh", "-lc"]),
            ("xfce4-terminal", &["-e", "sh", "-lc"]),
            ("mate-terminal", &["-e", "sh", "-lc"]),
            ("lxterminal", &["-e", "sh", "-lc"]),
            ("tilix", &["-e", "sh", "-lc"]),
            ("alacritty", &["-e", "sh", "-lc"]),
            ("kitty", &["sh", "-lc"]),
            ("xterm", &["-e", "sh", "-lc"]),
        ];
        for (program, args) in candidates {
            if command_exists(program) {
                let mut process = StdCommand::new(program);
                process.args(*args).arg(trimmed).spawn()?;
                return Ok(());
            }
        }
        return Err("No supported terminal emulator found".into());
    }

    #[allow(unreachable_code)]
    Err("Opening a terminal is not supported on this platform".into())
}

fn command_exists(program: &str) -> bool {
    let path = Path::new(program);
    if path.components().count() > 1 || path.is_absolute() {
        return path.is_file();
    }
    std::env::var_os("PATH").is_some_and(|path_var| {
        std::env::split_paths(&path_var).any(|directory| directory.join(program).is_file())
    })
}

fn read_log_lines(
    path: &Path,
    line_count: usize,
    min_level: Option<&str>,
    component_prefixes: Option<&[&str]>,
    search: Option<&str>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw_lines = tail_lines(path, line_count.saturating_mul(20).max(2_000));
    let mut filtered = raw_lines
        .into_iter()
        .filter(|line| {
            line_matches_level(line, min_level)
                && line_matches_component(line, component_prefixes)
                && search.is_none_or(|needle| line.to_ascii_lowercase().contains(needle))
        })
        .collect::<Vec<_>>();
    if filtered.len() > line_count {
        filtered = filtered[filtered.len() - line_count..].to_vec();
    }
    Ok(filtered)
}

fn line_matches_level(line: &str, min_level: Option<&str>) -> bool {
    let Some(min_level) = min_level else {
        return true;
    };
    let Some(line_level) = extract_log_level(line) else {
        return false;
    };
    log_level_rank(line_level) >= log_level_rank(min_level)
}

fn log_level_rank(level: &str) -> i32 {
    match level {
        "DEBUG" => 0,
        "INFO" => 1,
        "WARNING" => 2,
        "ERROR" => 3,
        "CRITICAL" => 4,
        _ => -1,
    }
}

fn extract_log_level(line: &str) -> Option<&'static str> {
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        .into_iter()
        .find(|level| line.contains(&format!(" {level} ")))
}

fn line_matches_component(line: &str, prefixes: Option<&[&str]>) -> bool {
    let Some(prefixes) = prefixes else {
        return true;
    };
    let Some(logger) = extract_logger_name(line) else {
        return false;
    };
    prefixes.iter().any(|prefix| logger.starts_with(prefix))
}

fn extract_logger_name(line: &str) -> Option<&str> {
    let level = extract_log_level(line)?;
    let marker = format!(" {level} ");
    let index = line.find(&marker)? + marker.len();
    let mut rest = line.get(index..)?.trim_start();
    if rest.starts_with('[') {
        rest = rest.split_once(']')?.1.trim_start();
    }
    rest.split_whitespace().next()?.strip_suffix(':')
}

fn load_hidden_plugin_set(
    context: &HermesContext,
) -> Result<std::collections::BTreeSet<String>, Box<dyn std::error::Error>> {
    let raw = read_raw_yaml_mapping(&context.config_path())?;
    let hidden = raw
        .get(YamlValue::String(String::from("dashboard")))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(YamlValue::String(String::from("hidden_plugins"))))
        .and_then(YamlValue::as_sequence)
        .cloned()
        .unwrap_or_default();
    Ok(hidden
        .into_iter()
        .filter_map(|value| value.as_str().map(str::trim).map(ToOwned::to_owned))
        .filter(|value| !value.is_empty())
        .collect())
}

fn set_dashboard_plugin_hidden(
    context: &HermesContext,
    name: &str,
    hidden: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_dashboard_plugin_name(name)?;
    let mut hidden_set = load_hidden_plugin_set(context)?;
    if hidden {
        hidden_set.insert(name.to_string());
    } else {
        hidden_set.remove(name);
    }
    let mut mapping = read_raw_yaml_mapping(&context.config_path())?;
    set_yaml_mapping_path(
        &mut mapping,
        &["dashboard", "hidden_plugins"],
        YamlValue::Sequence(
            hidden_set
                .into_iter()
                .map(YamlValue::String)
                .collect::<Vec<_>>(),
        ),
    )?;
    write_yaml_mapping(&context.config_path(), &mapping)
}

fn validate_dashboard_plugin_name(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let trimmed = name.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
    {
        return Err("invalid plugin name".into());
    }
    Ok(())
}

fn is_valid_env_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_uppercase()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_uppercase() || ch.is_ascii_digit())
}

fn looks_like_secret_key(key: &str) -> bool {
    key.ends_with("_API_KEY")
        || key.ends_with("_TOKEN")
        || key.ends_with("_SECRET")
        || key.ends_with("_PASSWORD")
        || key.ends_with("_KEY")
}

fn redact_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    if chars.len() <= 8 {
        return "*".repeat(chars.len());
    }
    let prefix = chars[..4].iter().collect::<String>();
    let suffix = chars[chars.len() - 4..].iter().collect::<String>();
    format!("{prefix}...{suffix}")
}

fn env_metadata_for_key(
    key: &str,
) -> (&'static str, Option<&'static str>, Vec<&'static str>, bool) {
    match key {
        "OPENROUTER_API_KEY" => (
            "OpenRouter API key",
            Some("https://openrouter.ai/keys"),
            vec!["vision_analyze", "mixture_of_agents"],
            true,
        ),
        "OPENAI_API_KEY" => (
            "OpenAI API key",
            Some("https://platform.openai.com/api-keys"),
            vec![],
            true,
        ),
        "ANTHROPIC_API_KEY" => (
            "Anthropic API key",
            Some("https://console.anthropic.com/settings/keys"),
            vec![],
            true,
        ),
        "GOOGLE_API_KEY" | "GEMINI_API_KEY" => (
            "Google AI Studio API key",
            Some("https://aistudio.google.com/app/apikey"),
            vec![],
            true,
        ),
        "FIRECRAWL_API_KEY" => (
            "Firecrawl API key",
            Some("https://www.firecrawl.dev"),
            vec![],
            true,
        ),
        _ => {
            let description = DASHBOARD_ENV_KEYS
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, label)| *label)
                .unwrap_or("");
            (description, None, vec![], key.ends_with("_BASE_URL"))
        }
    }
}

fn remove_env_key(path: &Path, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
    if !is_valid_env_key(key) {
        return Err("invalid environment variable name".into());
    }
    if !path.exists() {
        return Ok(false);
    }
    let original = fs::read_to_string(path)?;
    let mut removed = false;
    let mut lines = Vec::new();
    for line in original.lines() {
        let trimmed = line.trim_start();
        let without_export = trimmed.strip_prefix("export ").unwrap_or(trimmed);
        if without_export
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
        {
            removed = true;
            continue;
        }
        lines.push(line);
    }
    if !removed {
        return Ok(false);
    }
    let mut payload = lines.join("\n");
    if !payload.is_empty() {
        payload.push('\n');
    }
    fs::write(path, payload)?;
    Ok(true)
}

fn list_profiles(
    context: &HermesContext,
) -> Result<Vec<ProfileSummary>, Box<dyn std::error::Error>> {
    let mut profiles = Vec::new();
    let default_home = context.default_hermes_root();
    if default_home.is_dir() {
        let (model, provider) = read_profile_model(&default_home);
        profiles.push(ProfileSummary {
            name: String::from("default"),
            path: default_home.clone(),
            is_default: true,
            model,
            provider,
            has_env: default_home.join(".env").exists(),
            skill_count: count_profile_skills(&default_home),
        });
    }

    let profiles_root = context.profiles_root();
    if profiles_root.is_dir() {
        let mut entries = fs::read_dir(&profiles_root)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if hermes_core::validate_profile_name(&name).is_err() {
                continue;
            }
            let (model, provider) = read_profile_model(&path);
            profiles.push(ProfileSummary {
                name,
                path: path.clone(),
                is_default: false,
                model,
                provider,
                has_env: path.join(".env").exists(),
                skill_count: count_profile_skills(&path),
            });
        }
    }

    profiles.sort_by(|left, right| {
        left.is_default
            .cmp(&right.is_default)
            .reverse()
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(profiles)
}

fn read_profile_model(profile_dir: &Path) -> (Option<String>, Option<String>) {
    let config_path = profile_dir.join("config.yaml");
    let Ok(raw) = fs::read_to_string(config_path) else {
        return (None, None);
    };
    let Ok(parsed) = serde_yaml::from_str::<YamlValue>(&raw) else {
        return (None, None);
    };
    let Some(model_cfg) = parsed.get("model") else {
        return (None, None);
    };
    if let Some(text) = model_cfg.as_str() {
        return (Some(text.to_string()), None);
    }
    let Some(map) = model_cfg.as_mapping() else {
        return (None, None);
    };
    let model = map
        .get(YamlValue::String(String::from("default")))
        .or_else(|| map.get(YamlValue::String(String::from("model"))))
        .and_then(YamlValue::as_str)
        .map(str::to_string);
    let provider = map
        .get(YamlValue::String(String::from("provider")))
        .and_then(YamlValue::as_str)
        .map(str::to_string);
    (model, provider)
}

fn count_profile_skills(profile_dir: &Path) -> usize {
    let skills_dir = profile_dir.join("skills");
    if !skills_dir.is_dir() {
        return 0;
    }
    walk_skill_files(&skills_dir)
}

fn walk_skill_files(root: &Path) -> usize {
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            !path.components().any(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .is_some_and(|name| name == ".git" || name == ".hub")
            })
        })
        .map(|path| {
            if path.is_dir() {
                walk_skill_files(&path)
            } else if path.file_name().and_then(|value| value.to_str()) == Some("SKILL.md") {
                1
            } else {
                0
            }
        })
        .sum()
}

fn resolve_profile_dir(context: &HermesContext, name: &str) -> Result<PathBuf, ProfileLookupError> {
    let canon = hermes_core::normalize_profile_name(name)
        .map_err(|error| ProfileLookupError::Invalid(error.to_string()))?;
    hermes_core::validate_profile_name(&canon)
        .map_err(|error| ProfileLookupError::Invalid(error.to_string()))?;
    let path = context
        .profile_dir(&canon)
        .map_err(|error| ProfileLookupError::Invalid(error.to_string()))?;
    if canon != "default" && !path.is_dir() {
        return Err(ProfileLookupError::Missing(format!(
            "Profile '{canon}' does not exist."
        )));
    }
    Ok(path)
}

fn update_yaml_setting(
    path: &Path,
    dotted_path: &[&str],
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mapping = read_raw_yaml_mapping(path)?;
    set_yaml_mapping_path(
        &mut mapping,
        dotted_path,
        YamlValue::String(value.to_string()),
    )?;
    write_yaml_mapping(path, &mapping)
}

fn set_yaml_mapping_path(
    mapping: &mut Mapping,
    path: &[&str],
    value: YamlValue,
) -> Result<(), Box<dyn std::error::Error>> {
    if path.is_empty() {
        return Err("config path must not be empty".into());
    }
    if path.len() == 1 {
        mapping.insert(YamlValue::String(path[0].to_string()), value);
        return Ok(());
    }

    let key = YamlValue::String(path[0].to_string());
    if !mapping
        .get(&key)
        .is_some_and(|existing| existing.is_mapping())
    {
        mapping.insert(key.clone(), YamlValue::Mapping(Mapping::new()));
    }
    let Some(child) = mapping.get_mut(&key).and_then(YamlValue::as_mapping_mut) else {
        return Err("config path points through a non-mapping value".into());
    };
    set_yaml_mapping_path(child, &path[1..], value)
}

fn is_valid_theme_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

fn create_profile_dir(
    context: &HermesContext,
    project_root: &Path,
    raw_name: &str,
    clone_from_default: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let canon = hermes_core::normalize_profile_name(raw_name)?;
    hermes_core::validate_profile_name(&canon)?;
    if canon == "default" {
        return Err("Cannot create the default profile.".into());
    }

    let profile_dir = context.create_profile(&canon)?;
    let profile_context = context
        .clone()
        .with_hermes_home_env(Some(profile_dir.clone()));
    profile_context.ensure_hermes_home()?;

    if clone_from_default {
        clone_profile_config(&context.default_hermes_root(), &profile_dir)?;
    } else {
        seed_dashboard_bundled_skills(project_root, &profile_dir)?;
    }

    if check_alias_collision(context.home_dir(), &canon)?.is_none() {
        let _ = create_wrapper_script(context.home_dir(), &canon, &canon);
    }
    Ok(profile_dir)
}

fn rename_profile_dir(
    context: &HermesContext,
    raw_old: &str,
    raw_new: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let old_name = hermes_core::normalize_profile_name(raw_old)?;
    let new_name = hermes_core::normalize_profile_name(raw_new)?;
    hermes_core::validate_profile_name(&old_name)?;
    hermes_core::validate_profile_name(&new_name)?;
    if old_name == "default" {
        return Err("Cannot rename the default profile.".into());
    }
    if new_name == "default" {
        return Err("Cannot rename to 'default' — it is reserved.".into());
    }
    let old_dir = context.profile_dir(&old_name)?;
    let new_dir = context.profile_dir(&new_name)?;
    if !old_dir.is_dir() {
        return Err(format!("Profile '{old_name}' does not exist.").into());
    }
    if new_dir.exists() {
        return Err(format!("Profile '{new_name}' already exists.").into());
    }

    cleanup_gateway_service(&old_dir);
    fs::rename(&old_dir, &new_dir)?;
    let _ = remove_wrapper_script(context.home_dir(), &old_name);
    if check_alias_collision(context.home_dir(), &new_name)?.is_none() {
        let _ = create_wrapper_script(context.home_dir(), &new_name, &new_name);
    }
    if context.active_profile() == old_name {
        context.set_active_profile(&new_name)?;
    }
    Ok(new_dir)
}

fn delete_profile_dir(
    context: &HermesContext,
    raw_name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let canon = hermes_core::normalize_profile_name(raw_name)?;
    hermes_core::validate_profile_name(&canon)?;
    if canon == "default" {
        return Err("Cannot delete the default profile (~/.hermes).".into());
    }
    let profile_dir = context.profile_dir(&canon)?;
    if !profile_dir.is_dir() {
        return Err(format!("Profile '{canon}' does not exist.").into());
    }
    cleanup_gateway_service(&profile_dir);
    let _ = remove_wrapper_script(context.home_dir(), &canon);
    fs::remove_dir_all(&profile_dir)?;
    if context.active_profile() == canon {
        context.set_active_profile("default")?;
    }
    Ok(profile_dir)
}

fn clone_profile_config(
    source: &Path,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for name in ["config.yaml", ".env", "SOUL.md"] {
        let src = source.join(name);
        if src.exists() {
            fs::copy(&src, destination.join(name))?;
        }
    }
    let source_skills = source.join("skills");
    if source_skills.is_dir() {
        copy_dir_recursive(&source_skills, &destination.join("skills"), &|path, _| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == ".git" || name == ".github" || name == ".hub")
        })?;
    }
    for relative in ["memories/MEMORY.md", "memories/USER.md"] {
        let src = source.join(relative);
        if src.exists() {
            let dest = destination.join(relative);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(src, dest)?;
        }
    }
    Ok(())
}

fn seed_dashboard_bundled_skills(
    project_root: &Path,
    profile_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let bundled_dir = std::env::var_os("HERMES_BUNDLED_SKILLS")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| project_root.join("skills"));
    if !bundled_dir.is_dir() {
        return Ok(());
    }
    copy_dir_recursive(&bundled_dir, &profile_dir.join("skills"), &|path, _| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == ".git" || name == ".github" || name == ".hub")
    })
}

fn copy_dir_recursive(
    source: &Path,
    destination: &Path,
    skip: &dyn Fn(&Path, usize) -> bool,
) -> Result<(), Box<dyn std::error::Error>> {
    fn walk(
        source: &Path,
        destination: &Path,
        depth: usize,
        skip: &dyn Fn(&Path, usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if skip(source, depth) {
            return Ok(());
        }
        fs::create_dir_all(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let src = entry.path();
            let dest = destination.join(entry.file_name());
            if skip(&src, depth + 1) {
                continue;
            }
            if src.is_dir() {
                walk(&src, &dest, depth + 1, skip)?;
            } else {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&src, &dest)?;
            }
        }
        Ok(())
    }

    walk(source, destination, 0, skip)
}

fn cleanup_gateway_service(profile_dir: &Path) {
    let profile_context =
        HermesContext::new("/tmp").with_hermes_home_env(Some(profile_dir.to_path_buf()));
    for command in [
        GatewayCommand::Stop(GatewayServiceArgs {
            system: false,
            all: false,
        }),
        GatewayCommand::Stop(GatewayServiceArgs {
            system: true,
            all: false,
        }),
        GatewayCommand::Uninstall(GatewaySystemArgs { system: false }),
        GatewayCommand::Uninstall(GatewaySystemArgs { system: true }),
    ] {
        let _ = print_gateway(
            &profile_context,
            GatewayArgs {
                accept_hooks: false,
                command: Some(command),
            },
        );
    }
}

fn check_alias_collision(
    home_dir: &Path,
    raw_name: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let canon = hermes_core::normalize_profile_name(raw_name)?;
    if RESERVED_ALIAS_NAMES.contains(&canon.as_str()) {
        return Ok(Some(format!("'{canon}' is a reserved name")));
    }
    if HERMES_SUBCOMMANDS.contains(&canon.as_str()) {
        return Ok(Some(format!(
            "'{canon}' conflicts with a hermes subcommand"
        )));
    }
    if let Some(existing) = which_on_path(&canon) {
        let wrapper = wrapper_dir(home_dir).join(&canon);
        if existing == wrapper
            && fs::read_to_string(&wrapper)
                .ok()
                .is_some_and(|content| content.contains("hermes -p"))
        {
            return Ok(None);
        }
        return Ok(Some(format!(
            "'{canon}' conflicts with an existing command ({})",
            existing.display()
        )));
    }
    Ok(None)
}

fn create_wrapper_script(
    home_dir: &Path,
    raw_alias: &str,
    raw_profile: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let alias = hermes_core::normalize_profile_name(raw_alias)?;
    let profile = hermes_core::normalize_profile_name(raw_profile)?;
    let wrapper_root = wrapper_dir(home_dir);
    fs::create_dir_all(&wrapper_root)?;
    let wrapper_path = wrapper_root.join(alias);
    fs::write(
        &wrapper_path,
        format!("#!/bin/sh\nexec hermes -p {profile} \"$@\"\n"),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&wrapper_path)?.permissions();
        perms.set_mode(perms.mode() | 0o111);
        fs::set_permissions(&wrapper_path, perms)?;
    }
    Ok(wrapper_path)
}

fn remove_wrapper_script(home_dir: &Path, raw_alias: &str) -> bool {
    let Ok(alias) = hermes_core::normalize_profile_name(raw_alias) else {
        return false;
    };
    let wrapper_path = wrapper_dir(home_dir).join(alias);
    if !wrapper_path.exists() {
        return false;
    }
    let Ok(content) = fs::read_to_string(&wrapper_path) else {
        return false;
    };
    if !content.contains("hermes -p") {
        return false;
    }
    fs::remove_file(&wrapper_path).is_ok()
}

fn wrapper_dir(home_dir: &Path) -> PathBuf {
    home_dir.join(".local").join("bin")
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if cfg!(windows) {
            let candidate = candidate.with_extension("exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn discover_dashboard_skills(
    context: &HermesContext,
    raw_config: &YamlValue,
) -> Result<Vec<SkillSummary>, Box<dyn std::error::Error>> {
    let mut dirs = vec![context.hermes_home().join("skills")];
    dirs.extend(external_skills_dirs(context, raw_config));
    let mut seen = std::collections::HashSet::new();
    let mut skills = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let mut skill_files = Vec::new();
        collect_dashboard_skill_files(&dir, &mut skill_files)?;
        for skill_md in skill_files {
            let Ok(content) = fs::read_to_string(&skill_md) else {
                continue;
            };
            let (frontmatter, _body) = parse_skill_frontmatter(&content);
            if !skill_frontmatter_matches_platform(&frontmatter) {
                continue;
            }
            let Some(skill_dir) = skill_md.parent() else {
                continue;
            };
            let fallback_name = skill_dir
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("skill");
            let name = frontmatter
                .get(YamlValue::String(String::from("name")))
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(fallback_name)
                .to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            let description = frontmatter
                .get(YamlValue::String(String::from("description")))
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            skills.push(SkillSummary {
                name,
                description,
                category: skill_category_from_path(&dir, &skill_md),
            });
        }
    }
    skills.sort_by(|left, right| {
        let left_key = (
            left.category.as_deref().unwrap_or_default(),
            left.name.as_str(),
        );
        let right_key = (
            right.category.as_deref().unwrap_or_default(),
            right.name.as_str(),
        );
        left_key.cmp(&right_key)
    });
    Ok(skills)
}

fn collect_dashboard_skill_files(
    root: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if matches!(name.as_ref(), ".git" | ".github" | ".hub" | ".archive") {
                continue;
            }
            collect_dashboard_skill_files(&path, output)?;
        } else if name == "SKILL.md" {
            output.push(path);
        }
    }
    Ok(())
}

fn parse_skill_frontmatter(content: &str) -> (Mapping, String) {
    if !content.starts_with("---") {
        return (Mapping::new(), content.to_string());
    }
    let tail = &content[3..];
    let Some(end_offset) = tail.find("\n---\n").or_else(|| tail.find("\n---\r\n")) else {
        return (Mapping::new(), content.to_string());
    };
    let yaml_content = &tail[..end_offset];
    let body = tail[end_offset + 5..].to_string();
    match serde_yaml::from_str::<YamlValue>(yaml_content) {
        Ok(YamlValue::Mapping(mapping)) => (mapping, body),
        _ => (Mapping::new(), body),
    }
}

fn skill_frontmatter_matches_platform(frontmatter: &Mapping) -> bool {
    let Some(platforms) = frontmatter.get(YamlValue::String(String::from("platforms"))) else {
        return true;
    };
    let values = match platforms {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>(),
        YamlValue::String(value) => vec![value.trim().to_string()],
        _ => Vec::new(),
    };
    if values.is_empty() {
        return true;
    }
    let current = std::env::consts::OS;
    values.into_iter().any(|platform| {
        let normalized = platform.to_ascii_lowercase();
        match normalized.as_str() {
            "macos" => current == "macos",
            "linux" => current == "linux",
            "windows" => current == "windows",
            _ => normalized == current,
        }
    })
}

fn skill_category_from_path(skills_root: &Path, skill_md: &Path) -> Option<String> {
    let rel = skill_md.strip_prefix(skills_root).ok()?;
    let mut parts = rel.components();
    let first = parts.next()?;
    let second = parts.next()?;
    if second.as_os_str() == "SKILL.md" {
        return None;
    }
    match first {
        Component::Normal(value) => Some(value.to_string_lossy().to_string()),
        _ => None,
    }
}

fn external_skills_dirs(context: &HermesContext, raw_config: &YamlValue) -> Vec<PathBuf> {
    let Some(root) = raw_config.as_mapping() else {
        return Vec::new();
    };
    let Some(skills) = root
        .get(YamlValue::String(String::from("skills")))
        .and_then(YamlValue::as_mapping)
    else {
        return Vec::new();
    };
    let Some(raw_dirs) = skills.get(YamlValue::String(String::from("external_dirs"))) else {
        return Vec::new();
    };
    let values = match raw_dirs {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>(),
        YamlValue::String(value) => vec![value.clone()],
        _ => return Vec::new(),
    };
    let local_skills = context.hermes_home().join("skills");
    let local_resolved = local_skills
        .canonicalize()
        .unwrap_or_else(|_| local_skills.clone());
    let mut seen = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for raw in values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let expanded = expand_path_like(trimmed);
        let candidate = if Path::new(&expanded).is_absolute() {
            PathBuf::from(expanded)
        } else {
            context.hermes_home().join(expanded)
        };
        let resolved = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if resolved == local_resolved || !resolved.is_dir() || !seen.insert(resolved.clone()) {
            continue;
        }
        result.push(resolved);
    }
    result
}

fn expand_path_like(raw: &str) -> String {
    if raw == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| raw.to_string());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return std::env::var("HOME")
            .map(|home| format!("{home}/{rest}"))
            .unwrap_or_else(|_| raw.to_string());
    }
    raw.to_string()
}

fn resolve_dashboard_disabled_skills(raw_config: &YamlValue) -> std::collections::HashSet<String> {
    let platform = std::env::var("HERMES_PLATFORM")
        .ok()
        .or_else(|| std::env::var("HERMES_SESSION_PLATFORM").ok());
    dashboard_disabled_skills_for_platform(raw_config, platform.as_deref())
}

fn dashboard_disabled_skills_for_platform(
    raw_config: &YamlValue,
    platform: Option<&str>,
) -> std::collections::HashSet<String> {
    let Some(root) = raw_config.as_mapping() else {
        return std::collections::HashSet::new();
    };
    let Some(skills) = root
        .get(YamlValue::String(String::from("skills")))
        .and_then(YamlValue::as_mapping)
    else {
        return std::collections::HashSet::new();
    };
    let global_disabled =
        normalize_yaml_string_set(skills.get(YamlValue::String(String::from("disabled"))));
    let Some(platform) = platform.map(str::trim).filter(|value| !value.is_empty()) else {
        return global_disabled;
    };
    skills
        .get(YamlValue::String(String::from("platform_disabled")))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(YamlValue::String(platform.to_string())))
        .map(|value| normalize_yaml_string_set(Some(value)))
        .unwrap_or(global_disabled)
}

fn normalize_yaml_string_set(value: Option<&YamlValue>) -> std::collections::HashSet<String> {
    match value {
        Some(YamlValue::Sequence(items)) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        Some(YamlValue::String(value)) => value
            .trim()
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        _ => std::collections::HashSet::new(),
    }
}

fn set_skill_enabled(
    context: &HermesContext,
    skill_name: &str,
    enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let loaded = context.load_config_document().ok();
    let raw = loaded
        .as_ref()
        .map(|loaded| loaded.raw.clone())
        .unwrap_or(YamlValue::Mapping(Mapping::new()));
    let mut disabled = resolve_dashboard_disabled_skills(&raw);
    if enabled {
        disabled.remove(skill_name);
    } else {
        disabled.insert(skill_name.to_string());
    }
    let mut mapping = read_raw_yaml_mapping(&context.config_path())?;
    set_yaml_mapping_path(
        &mut mapping,
        &["skills", "disabled"],
        YamlValue::Sequence(
            disabled
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(YamlValue::String)
                .collect(),
        ),
    )?;
    write_yaml_mapping(&context.config_path(), &mapping)
}

fn cron_action_result(context: &HermesContext, args: JsonValue) -> Result<JsonValue, String> {
    let runtime = ToolRuntime::default().with_hermes_home(context.hermes_home());
    let raw = handle_cronjob(&args, &runtime);
    let value: JsonValue =
        serde_json::from_str(&raw).map_err(|error| format!("invalid cron response: {error}"))?;
    if let Some(error) = value.get("error").and_then(JsonValue::as_str) {
        return Err(error.to_string());
    }
    if !value
        .get("success")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        return Err(String::from("cron action failed"));
    }
    Ok(value)
}

fn cron_jobs_path(context: &HermesContext) -> PathBuf {
    context.hermes_home().join("cron").join("jobs.json")
}

fn read_cron_jobs(context: &HermesContext) -> Result<Vec<JsonValue>, String> {
    let path = cron_jobs_path(context);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    let value: JsonValue = serde_json::from_str(&text)
        .map_err(|error| format!("parsing {} failed: {error}", path.display()))?;
    if let Some(jobs) = value.as_array() {
        return Ok(jobs.clone());
    }
    Ok(value
        .get("jobs")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default())
}

fn read_cron_job(context: &HermesContext, job_id: &str) -> Result<Option<JsonValue>, String> {
    let job_id = job_id.trim();
    if job_id.is_empty() {
        return Ok(None);
    }
    Ok(read_cron_jobs(context)?
        .into_iter()
        .find(|job| job.get("id").and_then(JsonValue::as_str) == Some(job_id)))
}

fn cron_job_action_response(context: &HermesContext, action: &str, job_id: &str) -> Response<Body> {
    if job_id.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "job id is required"}),
        );
    }
    match cron_action_result(context, json!({"action": action, "job_id": job_id.trim()})) {
        Ok(_) => match read_cron_job(context, job_id.trim()) {
            Ok(Some(job)) => json_response(StatusCode::OK, job),
            Ok(None) if action == "run" => json_response(StatusCode::OK, json!({"ok": true})),
            Ok(None) => json_response(
                StatusCode::NOT_FOUND,
                json!({"detail": format!("cron job not found: {}", job_id.trim())}),
            ),
            Err(error) => {
                json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": error}))
            }
        },
        Err(error) if error.contains("not found") => {
            json_response(StatusCode::NOT_FOUND, json!({"detail": error}))
        }
        Err(error) => json_response(StatusCode::BAD_REQUEST, json!({"detail": error})),
    }
}

fn set_main_model_assignment(
    context: &HermesContext,
    provider: &str,
    model: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mapping = read_raw_yaml_mapping(&context.config_path())?;
    let model_mapping = ensure_yaml_mapping_path(&mut mapping, &["model"])?;
    model_mapping.insert(
        YamlValue::String(String::from("provider")),
        YamlValue::String(provider.to_string()),
    );
    model_mapping.insert(
        YamlValue::String(String::from("default")),
        YamlValue::String(model.to_string()),
    );
    model_mapping.remove(YamlValue::String(String::from("base_url")));
    model_mapping.remove(YamlValue::String(String::from("context_length")));
    write_yaml_mapping(&context.config_path(), &mapping)
}

fn set_auxiliary_model_assignment(
    context: &HermesContext,
    provider: &str,
    model: &str,
    task: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut mapping = read_raw_yaml_mapping(&context.config_path())?;
    let aux_mapping = ensure_yaml_mapping_path(&mut mapping, &["auxiliary"])?;

    if task == "__reset__" {
        for slot in AUXILIARY_TASK_SLOTS {
            let slot_mapping = ensure_yaml_mapping_entry(aux_mapping, slot)?;
            slot_mapping.insert(
                YamlValue::String(String::from("provider")),
                YamlValue::String(String::from("auto")),
            );
            slot_mapping.insert(
                YamlValue::String(String::from("model")),
                YamlValue::String(String::new()),
            );
        }
        write_yaml_mapping(&context.config_path(), &mapping)?;
        return Ok(Vec::new());
    }

    if provider.is_empty() {
        return Err("provider required for auxiliary".into());
    }

    let targets = if task.is_empty() {
        AUXILIARY_TASK_SLOTS
            .iter()
            .map(|slot| (*slot).to_string())
            .collect::<Vec<_>>()
    } else {
        if !AUXILIARY_TASK_SLOTS.iter().any(|slot| *slot == task) {
            return Err(format!("unknown auxiliary task: {task}").into());
        }
        vec![task.to_string()]
    };

    for slot in &targets {
        let slot_mapping = ensure_yaml_mapping_entry(aux_mapping, slot)?;
        slot_mapping.insert(
            YamlValue::String(String::from("provider")),
            YamlValue::String(provider.to_string()),
        );
        slot_mapping.insert(
            YamlValue::String(String::from("model")),
            YamlValue::String(model.to_string()),
        );
    }

    write_yaml_mapping(&context.config_path(), &mapping)?;
    Ok(targets)
}

fn ensure_yaml_mapping_path<'a>(
    mapping: &'a mut Mapping,
    path: &[&str],
) -> Result<&'a mut Mapping, Box<dyn std::error::Error>> {
    if path.is_empty() {
        return Err("config path must not be empty".into());
    }
    let key = YamlValue::String(path[0].to_string());
    if path.len() == 1 {
        if !mapping
            .get(&key)
            .is_some_and(|existing| existing.is_mapping())
        {
            mapping.insert(key.clone(), YamlValue::Mapping(Mapping::new()));
        }
        return mapping
            .get_mut(&key)
            .and_then(YamlValue::as_mapping_mut)
            .ok_or_else(|| "config path points through a non-mapping value".into());
    }
    let child = ensure_yaml_mapping_entry(mapping, path[0])?;
    ensure_yaml_mapping_path(child, &path[1..])
}

fn ensure_yaml_mapping_entry<'a>(
    mapping: &'a mut Mapping,
    key: &str,
) -> Result<&'a mut Mapping, Box<dyn std::error::Error>> {
    let yaml_key = YamlValue::String(key.to_string());
    if !mapping
        .get(&yaml_key)
        .is_some_and(|existing| existing.is_mapping())
    {
        mapping.insert(yaml_key.clone(), YamlValue::Mapping(Mapping::new()));
    }
    mapping
        .get_mut(&yaml_key)
        .and_then(YamlValue::as_mapping_mut)
        .ok_or_else(|| "config path points through a non-mapping value".into())
}

fn normalize_analytics_days(days: Option<i64>) -> i64 {
    days.unwrap_or(30).clamp(1, 3650)
}

fn open_state_connection(
    context: &HermesContext,
) -> Result<Option<Connection>, Box<dyn std::error::Error>> {
    let path = context.state_db_path();
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(Connection::open(path)?))
}

fn read_usage_analytics(
    context: &HermesContext,
    days: i64,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let cutoff = now_ts() - (days as f64 * 86_400.0);
    let skills = read_skill_usage_summary(context);
    let Some(connection) = open_state_connection(context)? else {
        return Ok(json!({
            "daily": [],
            "by_model": [],
            "totals": {
                "total_input": 0,
                "total_output": 0,
                "total_cache_read": 0,
                "total_reasoning": 0,
                "total_estimated_cost": 0.0,
                "total_actual_cost": 0.0,
                "total_sessions": 0,
                "total_api_calls": 0,
            },
            "period_days": days,
            "skills": skills,
        }));
    };

    let daily = query_json_rows(
        &connection,
        "SELECT date(started_at, 'unixepoch') as day,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) as cache_read_tokens,
                COALESCE(SUM(reasoning_tokens), 0) as reasoning_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                COUNT(*) as sessions,
                COALESCE(SUM(api_call_count), 0) as api_calls
         FROM sessions WHERE started_at > ?
         GROUP BY day ORDER BY day",
        &[cutoff],
    )?;
    let by_model = query_json_rows(
        &connection,
        "SELECT model,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                COUNT(*) as sessions,
                COALESCE(SUM(api_call_count), 0) as api_calls
         FROM sessions
         WHERE started_at > ? AND model IS NOT NULL AND model != ''
         GROUP BY model
         ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC",
        &[cutoff],
    )?;
    let totals = query_json_row(
        &connection,
        "SELECT COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                COALESCE(SUM(reasoning_tokens), 0) as total_reasoning,
                COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                COUNT(*) as total_sessions,
                COALESCE(SUM(api_call_count), 0) as total_api_calls
         FROM sessions WHERE started_at > ?",
        &[cutoff],
    )?
    .unwrap_or_else(|| {
        json!({
            "total_input": 0,
            "total_output": 0,
            "total_cache_read": 0,
            "total_reasoning": 0,
            "total_estimated_cost": 0.0,
            "total_actual_cost": 0.0,
            "total_sessions": 0,
            "total_api_calls": 0,
        })
    });

    Ok(json!({
        "daily": daily,
        "by_model": by_model,
        "totals": totals,
        "period_days": days,
        "skills": skills,
    }))
}

fn read_models_analytics(
    context: &HermesContext,
    days: i64,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let cutoff = now_ts() - (days as f64 * 86_400.0);
    let Some(connection) = open_state_connection(context)? else {
        return Ok(json!({
            "models": [],
            "totals": {
                "distinct_models": 0,
                "total_input": 0,
                "total_output": 0,
                "total_cache_read": 0,
                "total_reasoning": 0,
                "total_estimated_cost": 0.0,
                "total_actual_cost": 0.0,
                "total_sessions": 0,
                "total_api_calls": 0,
            },
            "period_days": days,
        }));
    };

    let mut models = query_json_rows(
        &connection,
        "SELECT model,
                billing_provider,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) as cache_read_tokens,
                COALESCE(SUM(reasoning_tokens), 0) as reasoning_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                COUNT(*) as sessions,
                COALESCE(SUM(api_call_count), 0) as api_calls,
                COALESCE(SUM(tool_call_count), 0) as tool_calls,
                COALESCE(MAX(started_at), 0) as last_used_at,
                COALESCE(AVG(input_tokens + output_tokens), 0) as avg_tokens_per_session
         FROM sessions
         WHERE started_at > ? AND model IS NOT NULL AND model != ''
         GROUP BY model, billing_provider
         ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC",
        &[cutoff],
    )?;
    for model in &mut models {
        if let Some(object) = model.as_object_mut() {
            let provider = object
                .get("billing_provider")
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            let provider = if provider.is_empty() {
                object
                    .get("model")
                    .and_then(JsonValue::as_str)
                    .and_then(derive_provider_from_model)
                    .unwrap_or("")
            } else {
                provider
            };
            object.insert(
                String::from("provider"),
                JsonValue::String(provider.to_string()),
            );
            object.remove("billing_provider");
            object.insert(String::from("capabilities"), json!({}));
        }
    }
    let totals = query_json_row(
        &connection,
        "SELECT COUNT(DISTINCT model) as distinct_models,
                COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                COALESCE(SUM(reasoning_tokens), 0) as total_reasoning,
                COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                COUNT(*) as total_sessions,
                COALESCE(SUM(api_call_count), 0) as total_api_calls
         FROM sessions
         WHERE started_at > ? AND model IS NOT NULL AND model != ''",
        &[cutoff],
    )?
    .unwrap_or_else(|| {
        json!({
            "distinct_models": 0,
            "total_input": 0,
            "total_output": 0,
            "total_cache_read": 0,
            "total_reasoning": 0,
            "total_estimated_cost": 0.0,
            "total_actual_cost": 0.0,
            "total_sessions": 0,
            "total_api_calls": 0,
        })
    });

    Ok(json!({
        "models": models,
        "totals": totals,
        "period_days": days,
    }))
}

fn query_json_rows(
    connection: &Connection,
    sql: &str,
    params_values: &[f64],
) -> Result<Vec<JsonValue>, Box<dyn std::error::Error>> {
    let mut statement = connection.prepare(sql)?;
    let mut rows = statement.query(params![params_values[0]])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        result.push(sqlite_row_to_json(row)?);
    }
    Ok(result)
}

fn query_json_row(
    connection: &Connection,
    sql: &str,
    params_values: &[f64],
) -> Result<Option<JsonValue>, Box<dyn std::error::Error>> {
    let mut statement = connection.prepare(sql)?;
    let mut rows = statement.query(params![params_values[0]])?;
    match rows.next()? {
        Some(row) => Ok(Some(sqlite_row_to_json(row)?)),
        None => Ok(None),
    }
}

fn sqlite_row_to_json(row: &rusqlite::Row<'_>) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let mut object = JsonMap::new();
    let row_ref = row.as_ref();
    for index in 0..row_ref.column_count() {
        let name = row_ref.column_name(index)?.to_string();
        let value = row.get_ref(index)?;
        let json_value = match value {
            rusqlite::types::ValueRef::Null => JsonValue::Null,
            rusqlite::types::ValueRef::Integer(value) => JsonValue::from(value),
            rusqlite::types::ValueRef::Real(value) => JsonValue::from(value),
            rusqlite::types::ValueRef::Text(value) => {
                JsonValue::String(String::from_utf8_lossy(value).to_string())
            }
            rusqlite::types::ValueRef::Blob(_) => JsonValue::Null,
        };
        object.insert(name, json_value);
    }
    Ok(JsonValue::Object(object))
}

fn derive_provider_from_model(model: &str) -> Option<&str> {
    let (provider, rest) = model.split_once('/')?;
    (!provider.trim().is_empty() && !rest.trim().is_empty()).then_some(provider)
}

fn read_skill_usage_summary(context: &HermesContext) -> JsonValue {
    let path = context.hermes_home().join("skills").join(".usage.json");
    let Ok(raw) = fs::read_to_string(path) else {
        return json!({
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        });
    };
    let Ok(value) = serde_json::from_str::<JsonValue>(&raw) else {
        return json!({
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        });
    };
    let Some(map) = value.as_object() else {
        return json!({
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        });
    };

    let mut total_skill_loads = 0_i64;
    let mut total_skill_edits = 0_i64;
    let mut top_skills = Vec::new();
    for (skill, record) in map {
        let use_count = record
            .get("use_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        let view_count = record
            .get("view_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        let patch_count = record
            .get("patch_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        let loads = use_count + view_count;
        let total_count = loads + patch_count;
        if total_count <= 0 {
            continue;
        }
        total_skill_loads += loads;
        total_skill_edits += patch_count;
        let last_used_at = record
            .get("last_used_at")
            .or_else(|| record.get("last_viewed_at"))
            .or_else(|| record.get("last_patched_at"))
            .and_then(JsonValue::as_str)
            .and_then(iso_to_timestamp);
        top_skills.push(json!({
            "skill": skill,
            "view_count": loads,
            "manage_count": patch_count,
            "total_count": total_count,
            "percentage": 0.0,
            "last_used_at": last_used_at,
        }));
    }
    let total_skill_actions = total_skill_loads + total_skill_edits;
    top_skills.sort_by(|left, right| {
        let left_total = left
            .get("total_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        let right_total = right
            .get("total_count")
            .and_then(JsonValue::as_i64)
            .unwrap_or(0);
        right_total.cmp(&left_total)
    });
    for entry in &mut top_skills {
        if let Some(object) = entry.as_object_mut() {
            let total_count = object
                .get("total_count")
                .and_then(JsonValue::as_i64)
                .unwrap_or(0);
            let percentage = if total_skill_actions > 0 {
                total_count as f64 / total_skill_actions as f64 * 100.0
            } else {
                0.0
            };
            object.insert(String::from("percentage"), JsonValue::from(percentage));
        }
    }
    json!({
        "summary": {
            "total_skill_loads": total_skill_loads,
            "total_skill_edits": total_skill_edits,
            "total_skill_actions": total_skill_actions,
            "distinct_skills_used": top_skills.len(),
        },
        "top_skills": top_skills,
    })
}

fn iso_to_timestamp(value: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|parsed| parsed.timestamp_millis() as f64 / 1000.0)
}

fn action_spawn_response(
    state: &DashboardState,
    subcommand: &[&str],
    name: &str,
) -> Response<Body> {
    match spawn_dashboard_action(state, subcommand, name) {
        Ok(pid) => json_response(
            StatusCode::OK,
            json!({"ok": true, "pid": pid, "name": name}),
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": error.to_string()}),
        ),
    }
}

fn spawn_dashboard_action(
    state: &DashboardState,
    subcommand: &[&str],
    name: &str,
) -> Result<u32, Box<dyn std::error::Error>> {
    let log_file_name = action_log_file(name).ok_or("unknown action")?;
    let logs_dir = state.context.hermes_home().join("logs");
    fs::create_dir_all(&logs_dir)?;
    let log_path = logs_dir.join(log_file_name);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    writeln!(
        file,
        "\n=== {name} started {} ===",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    )?;
    let stdout = file.try_clone()?;
    let stderr = file;

    let program = std::env::var_os("HERMES_DASHBOARD_ACTION_BIN")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_exe()?);
    let mut command = StdCommand::new(program);
    command
        .current_dir(&state.project_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env("HERMES_NONINTERACTIVE", "1")
        .args(subcommand);
    let child = command.spawn()?;
    let pid = child.id();
    let mut actions = ACTION_PROCESSES.lock().expect("action process lock");
    actions.insert(
        name.to_string(),
        ActionProcess {
            child: Some(child),
            pid,
            exit_code: None,
        },
    );
    Ok(pid)
}

fn action_log_file(name: &str) -> Option<&'static str> {
    match name {
        "gateway-restart" => Some("gateway-restart.log"),
        "hermes-update" => Some("hermes-update.log"),
        _ => None,
    }
}

fn tail_lines(path: &Path, line_count: usize) -> Vec<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines = text.lines().map(str::to_string).collect::<Vec<_>>();
    if line_count == 0 || lines.len() <= line_count {
        return lines;
    }
    lines[lines.len() - line_count..].to_vec()
}

fn action_process_status(name: &str) -> (bool, Option<i32>, Option<u32>) {
    let mut actions = ACTION_PROCESSES.lock().expect("action process lock");
    let Some(entry) = actions.get_mut(name) else {
        return (false, None, None);
    };
    if entry.exit_code.is_none()
        && let Some(child) = entry.child.as_mut()
        && let Ok(Some(status)) = child.try_wait()
    {
        entry.exit_code = status.code();
        entry.child = None;
    }
    (entry.child.is_some(), entry.exit_code, Some(entry.pid))
}

static ACTION_PROCESSES: std::sync::LazyLock<Mutex<HashMap<String, ActionProcess>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

fn plugin_enable_disable_response(
    state: &DashboardState,
    name: &str,
    enabled: bool,
) -> Response<Body> {
    let name = name.trim();
    if name.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": "plugin name is required"}),
        );
    }
    match dashboard_set_agent_plugin_enabled(&state.context, name, enabled) {
        Ok(result) => json_response(
            StatusCode::OK,
            json!({"ok": true, "name": result.name, "unchanged": result.unchanged}),
        ),
        Err(error) => json_response(
            StatusCode::BAD_REQUEST,
            json!({"detail": error.to_string()}),
        ),
    }
}

fn refresh_dashboard_plugin_cache(state: &DashboardState) {
    let discovered = discover_dashboard_plugins(&state.project_root, &state.context.hermes_home());
    if let Ok(mut cache) = state.plugins.write() {
        *cache = discovered;
    }
}

fn visible_dashboard_manifests(state: &DashboardState) -> Vec<JsonValue> {
    let hidden = load_hidden_plugin_set(&state.context).unwrap_or_default();
    state
        .plugins
        .read()
        .map(|cache| {
            cache
                .manifests
                .iter()
                .filter(|manifest| {
                    manifest
                        .get("name")
                        .and_then(JsonValue::as_str)
                        .is_none_or(|name| !hidden.contains(name))
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn build_plugins_hub(state: &DashboardState) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let dashboard_manifests = visible_dashboard_manifests(state);
    let manifest_by_name = dashboard_manifests
        .iter()
        .filter_map(|manifest| {
            manifest
                .get("name")
                .and_then(JsonValue::as_str)
                .map(|name| (name.to_string(), manifest.clone()))
        })
        .collect::<HashMap<_, _>>();
    let hidden_plugins = load_hidden_plugin_set(&state.context)?;
    let (enabled_plugins, disabled_plugins) = dashboard_plugin_sets(&state.context)?;
    let plugins_root = state.context.hermes_home().join("plugins");
    let plugins_root = plugins_root.canonicalize().unwrap_or(plugins_root);

    let rows = dashboard_list_plugins(&state.context)?
        .into_iter()
        .map(|entry| {
            let runtime_status = if disabled_plugins.contains(&entry.name) {
                "disabled"
            } else if enabled_plugins.contains(&entry.name) {
                "enabled"
            } else {
                "inactive"
            };
            let dashboard_manifest = manifest_by_name.get(&entry.name).cloned();
            let has_dashboard_manifest = dashboard_manifest.is_some()
                || entry.path.join("dashboard").join("manifest.json").exists();
            let under_user_tree = entry
                .path
                .canonicalize()
                .ok()
                .and_then(|resolved| resolved.strip_prefix(&plugins_root).ok().map(|_| true))
                .unwrap_or(false);
            let can_remove = matches!(entry.source.as_str(), "user" | "git")
                && under_user_tree
                && entry.path.is_dir();
            let can_update_git = can_remove && entry.path.join(".git").exists();
            let missing_env = plugin_missing_env_names(&state.context, &entry.path);
            let auth_required = !missing_env.is_empty();
            let auth_command = missing_env
                .first()
                .map(|key| format!("hermes config env {} <value>", key))
                .unwrap_or_default();
            json!({
                "name": entry.name,
                "version": entry.version,
                "description": entry.description,
                "source": entry.source,
                "runtime_status": runtime_status,
                "has_dashboard_manifest": has_dashboard_manifest,
                "dashboard_manifest": dashboard_manifest,
                "path": entry.path,
                "can_remove": can_remove,
                "can_update_git": can_update_git,
                "auth_required": auth_required,
                "auth_command": auth_command,
                "user_hidden": hidden_plugins.contains(&entry.name),
            })
        })
        .collect::<Vec<_>>();

    let plugin_names = rows
        .iter()
        .filter_map(|row| row.get("name").and_then(JsonValue::as_str))
        .collect::<std::collections::BTreeSet<_>>();
    let orphan_dashboard_plugins = dashboard_manifests
        .into_iter()
        .filter(|manifest| {
            manifest
                .get("name")
                .and_then(JsonValue::as_str)
                .is_some_and(|name| !plugin_names.contains(name))
        })
        .collect::<Vec<_>>();

    let memory_options = dashboard_memory_provider_options(&state.context)
        .into_iter()
        .map(|provider| json!({"name": provider.name, "description": provider.description}))
        .collect::<Vec<_>>();
    let context_options = dashboard_context_engine_options(&state.context)?
        .into_iter()
        .map(|provider| json!({"name": provider.name, "description": provider.description}))
        .collect::<Vec<_>>();

    Ok(json!({
        "plugins": rows,
        "orphan_dashboard_plugins": orphan_dashboard_plugins,
        "providers": {
            "memory_provider": dashboard_current_memory_provider(&state.context)?,
            "memory_options": memory_options,
            "context_engine": dashboard_current_context_engine(&state.context)?,
            "context_options": context_options,
        }
    }))
}

fn plugin_missing_env_names(context: &HermesContext, plugin_dir: &Path) -> Vec<String> {
    let env_values = load_simple_env(&context.env_path());
    let manifest_path = plugin_dir.join("plugin.yaml");
    let alt_manifest_path = plugin_dir.join("plugin.yml");
    let manifest_path = if manifest_path.exists() {
        manifest_path
    } else if alt_manifest_path.exists() {
        alt_manifest_path
    } else {
        return Vec::new();
    };
    let Ok(raw) = fs::read_to_string(manifest_path) else {
        return Vec::new();
    };
    let Ok(parsed) = serde_yaml::from_str::<YamlValue>(&raw) else {
        return Vec::new();
    };
    let Some(mapping) = parsed.as_mapping() else {
        return Vec::new();
    };
    let Some(items) = mapping
        .get(YamlValue::String(String::from("requires_env")))
        .and_then(YamlValue::as_sequence)
    else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| match item {
            YamlValue::String(name) => Some(name.trim().to_string()),
            YamlValue::Mapping(mapping) => mapping
                .get(YamlValue::String(String::from("name")))
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .map(str::to_string),
            _ => None,
        })
        .filter(|name| !name.is_empty())
        .filter(|name| {
            std::env::var(name)
                .ok()
                .is_none_or(|value| value.trim().is_empty())
                && env_values
                    .get(name)
                    .is_none_or(|value| value.trim().is_empty())
        })
        .collect()
}

fn discover_dashboard_plugins(project_root: &Path, hermes_home: &Path) -> PluginCache {
    let mut manifests = Vec::new();
    let mut asset_dirs = HashMap::new();
    let mut seen_names = std::collections::BTreeSet::new();
    for (root, source) in [
        (hermes_home.join("plugins"), "user"),
        (project_root.join("plugins"), "bundled"),
    ] {
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let plugin_root = entry.path();
            let manifest_path = plugin_root.join("dashboard").join("manifest.json");
            if !manifest_path.exists() {
                continue;
            }
            let Ok(raw) = fs::read_to_string(&manifest_path) else {
                continue;
            };
            let Ok(mut manifest) = serde_json::from_str::<JsonValue>(&raw) else {
                continue;
            };
            let Some(object) = manifest.as_object_mut() else {
                continue;
            };
            let name = object
                .get("name")
                .and_then(JsonValue::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| entry.file_name().to_string_lossy().to_string());
            if seen_names.contains(&name) {
                continue;
            }
            object
                .entry(String::from("name"))
                .or_insert(JsonValue::String(name.clone()));
            object
                .entry(String::from("source"))
                .or_insert(JsonValue::String(source.to_string()));
            object
                .entry(String::from("has_api"))
                .or_insert(JsonValue::Bool(false));
            seen_names.insert(name.clone());
            asset_dirs.insert(name, plugin_root.join("dashboard"));
            manifests.push(manifest);
        }
    }
    PluginCache {
        manifests,
        asset_dirs,
    }
}

fn serve_file_from_root(root: &Path, requested_path: &str) -> Option<Response<Body>> {
    let relative = sanitize_relative_path(requested_path)?;
    let full_path = root.join(relative);
    if !full_path.is_file() {
        return None;
    }
    let body = fs::read(&full_path).ok()?;
    let mime = mime_guess::from_path(&full_path).first_or_octet_stream();
    let mut response = Response::new(Body::from(body));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_str(mime.as_ref()).ok()?);
    Some(response)
}

fn sanitize_relative_path(path: &str) -> Option<PathBuf> {
    let candidate = Path::new(path);
    let mut clean = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(value) => clean.push(value),
            Component::CurDir => {}
            _ => return None,
        }
    }
    Some(clean)
}

fn get_running_gateway_pid(context: &HermesContext) -> Option<i64> {
    let path = context.hermes_home().join("gateway.pid");
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let pid = serde_json::from_str::<JsonValue>(trimmed)
        .ok()
        .and_then(|value| value.get("pid").and_then(JsonValue::as_i64))
        .or_else(|| trimmed.parse::<i64>().ok())?;
    process_is_alive(pid).then_some(pid)
}

fn read_runtime_status(context: &HermesContext) -> Option<JsonValue> {
    let path = context.hermes_home().join("gateway_state.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<JsonValue>(&raw).ok()
}

fn process_is_alive(pid: i64) -> bool {
    #[cfg(unix)]
    {
        let rc = unsafe { libc::kill(pid as i32, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn parse_resize(regex: &Regex, text: Option<&str>) -> Option<(u16, u16)> {
    let captures = regex.captures(text?)?;
    let cols = captures.get(1)?.as_str().parse::<u16>().ok()?;
    let rows = captures.get(2)?.as_str().parse::<u16>().ok()?;
    Some((cols, rows))
}

fn ws_client_allowed(state: &DashboardState, ip: IpAddr) -> bool {
    if state.bound_host == "0.0.0.0" || state.bound_host == "::" {
        return true;
    }
    ip.is_loopback()
}

fn resolve_chat_command(
    state: &DashboardState,
    resume: Option<&str>,
    channel: Option<&str>,
) -> Result<(Vec<String>, Option<PathBuf>, HashMap<String, String>), Box<dyn std::error::Error>> {
    let python = resolve_repo_python(&state.project_root, Some("HERMES_DASHBOARD_PYTHON"))
        .ok_or("python interpreter not found")?;
    let script = concat!(
        "import json, os\n",
        "from hermes_cli.main import PROJECT_ROOT, _make_tui_argv\n",
        "argv, cwd = _make_tui_argv(PROJECT_ROOT / 'ui-tui', tui_dev=False)\n",
        "payload = {'argv': list(argv), 'cwd': str(cwd) if cwd else None}\n",
        "print(json.dumps(payload))\n",
    );
    let output = std::process::Command::new(python)
        .current_dir(&state.project_root)
        .env("PYTHONPATH", state.project_root.display().to_string())
        .arg("-c")
        .arg(script)
        .output()?;
    if !output.status.success() {
        return Err("failed to resolve TUI argv".into());
    }
    let payload: JsonValue = serde_json::from_slice(&output.stdout)?;
    let argv = payload
        .get("argv")
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .ok_or("missing argv")?;
    let cwd = payload
        .get("cwd")
        .and_then(JsonValue::as_str)
        .map(PathBuf::from);
    let mut env_overrides = HashMap::new();
    env_overrides.insert(String::from("NODE_ENV"), String::from("production"));
    if let Some(resume) = resume.filter(|value| !value.trim().is_empty()) {
        env_overrides.insert(String::from("HERMES_TUI_RESUME"), resume.to_string());
    }
    if let Some(channel) = channel {
        env_overrides.insert(
            String::from("HERMES_TUI_SIDECAR_URL"),
            format!(
                "ws://{}:{}/api/pub?token={}&channel={}",
                state.bound_host, state.bound_port, state.token, channel
            ),
        );
    }
    Ok((argv, cwd, env_overrides))
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn try_open_browser(url: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let commands: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("open", &[])]
    } else if cfg!(target_os = "windows") {
        &[("cmd", &["/C", "start", ""])]
    } else {
        &[("xdg-open", &[])]
    };
    for (program, args) in commands {
        let status = std::process::Command::new(program)
            .args(*args)
            .arg(url)
            .status();
        match status {
            Ok(status) if status.success() => return Ok(true),
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(false)
}

fn json_response(status: StatusCode, value: JsonValue) -> Response<Body> {
    let mut response = Response::new(Body::from(value.to_string()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn websocket_close(ws: WebSocketUpgrade, _status: StatusCode) -> Response<Body> {
    ws.on_upgrade(|socket| async move {
        let _ = socket.close().await;
    })
}

async fn send_ws_banner(socket: WebSocket, message: &str) -> Result<(), ()> {
    let mut socket = socket;
    socket
        .send(Message::Text(format!("\r\n\x1b[31m{message}\x1b[0m\r\n")))
        .await
        .map_err(|_| ())?;
    let _ = socket.close().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use futures_util::StreamExt;
    use std::io::{Read, Write};
    use tokio_tungstenite::connect_async;
    use tower::ServiceExt;

    fn test_state(tmp_home: &Path) -> Arc<DashboardState> {
        test_state_with_options(tmp_home, false, 9119)
    }

    fn test_state_with_options(
        tmp_home: &Path,
        embedded_chat: bool,
        bound_port: u16,
    ) -> Arc<DashboardState> {
        let defaults = serde_yaml::from_str::<serde_yaml::Value>(DEFAULT_CONFIG_YAML)
            .ok()
            .and_then(yaml_to_json)
            .unwrap_or_else(|| json!({}));
        Arc::new(DashboardState {
            context: HermesContext::new(tmp_home.to_path_buf())
                .with_hermes_home_env(Some(tmp_home.to_path_buf())),
            project_root: tmp_home.to_path_buf(),
            web_dist: tmp_home.join("web_dist"),
            token: String::from("test-token"),
            embedded_chat,
            bound_host: String::from("127.0.0.1"),
            bound_port,
            defaults: defaults.clone(),
            schema: build_config_schema(&defaults),
            plugins: Arc::new(RwLock::new(discover_dashboard_plugins(tmp_home, tmp_home))),
            oauth_sessions: Arc::new(Mutex::new(HashMap::new())),
            channel_re: Regex::new(VALID_CHANNEL_PATTERN).unwrap(),
        })
    }

    fn authed_request(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header(HOST, "localhost:9119")
            .header(SESSION_HEADER_NAME, "test-token")
            .body(Body::empty())
            .unwrap()
    }

    fn json_request(method: &str, uri: &str, payload: JsonValue) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(HOST, "localhost:9119")
            .header(SESSION_HEADER_NAME, "test-token")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    }

    async fn response_json(response: Response<Body>) -> JsonValue {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn run_git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Hermes")
            .env("GIT_AUTHOR_EMAIL", "hermes@example.com")
            .env("GIT_COMMITTER_NAME", "Hermes")
            .env("GIT_COMMITTER_EMAIL", "hermes@example.com")
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {:?} failed", args);
    }

    fn init_plugin_repo(repo: &Path, manifest: &str) {
        fs::create_dir_all(repo).unwrap();
        run_git(repo, &["init"]);
        fs::write(repo.join("plugin.yaml"), manifest).unwrap();
        run_git(repo, &["add", "."]);
        run_git(repo, &["commit", "-m", "init"]);
    }

    fn http_json_response(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 4096];
        let mut header_end = None;
        let mut content_length = 0_usize;
        loop {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if header_end.is_none()
                && let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
            {
                header_end = Some(index + 4);
                let headers = String::from_utf8_lossy(&buffer[..index + 4]);
                content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("Content-Length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
            }
            if let Some(header_end) = header_end
                && buffer.len() >= header_end + content_length
            {
                break;
            }
        }
        String::from_utf8(buffer).unwrap()
    }

    fn spawn_anthropic_token_server() -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = requests.clone();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            requests_clone.lock().unwrap().push(request);
            let body = json!({
                "access_token": "header.eyJlbWFpbCI6ImFudGhAc2VydmVyLnRlc3QifQ.sig",
                "refresh_token": "anthropic-refresh",
                "expires_in": 3600
            })
            .to_string();
            stream
                .write_all(http_json_response("200 OK", &body).as_bytes())
                .unwrap();
        });
        (format!("http://{addr}/oauth/token"), requests, handle)
    }

    fn spawn_codex_oauth_server() -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = requests.clone();
        let handle = thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                let first_line = request.lines().next().unwrap_or_default().to_string();
                requests_clone.lock().unwrap().push(first_line.clone());
                let body = if first_line.starts_with("POST /api/accounts/deviceauth/usercode ") {
                    json!({
                        "user_code": "USER-CODE",
                        "device_auth_id": "device-auth-123",
                        "interval": 1
                    })
                    .to_string()
                } else if first_line.starts_with("POST /api/accounts/deviceauth/token ") {
                    json!({
                        "authorization_code": "auth-code-123",
                        "code_verifier": "verifier-xyz"
                    })
                    .to_string()
                } else {
                    json!({
                        "access_token": "header.eyJlbWFpbCI6ImNvZGV4LWRhc2hib2FyZEB0ZXN0In0.sig",
                        "refresh_token": "codex-refresh"
                    })
                    .to_string()
                };
                stream
                    .write_all(http_json_response("200 OK", &body).as_bytes())
                    .unwrap();
            }
        });
        (format!("http://{addr}"), requests, handle)
    }

    fn spawn_nous_oauth_server() -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = requests.clone();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                requests_clone.lock().unwrap().push(request);
                let body = match idx {
                    0 => json!({
                        "device_code": "nous-device-code",
                        "user_code": "NOUS-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "verification_uri_complete": format!("{base_for_thread}/verify?code=NOUS-CODE"),
                        "expires_in": 60,
                        "interval": 1
                    }),
                    1 => json!({
                        "access_token": "header.eyJlbWFpbCI6Im5vdXMtZGFzaGJvYXJkQHRlc3QifQ.sig",
                        "refresh_token": "nous-refresh",
                        "token_type": "Bearer",
                        "scope": "inference:mint_agent_key offline_access",
                        "expires_in": 3600,
                        "inference_base_url": format!("{base_for_thread}/portal-inference/v1")
                    }),
                    _ => json!({
                        "api_key": "nous-agent-key",
                        "key_id": "agent-key-123",
                        "expires_at": "2999-01-02T00:00:00Z",
                        "expires_in": 86400,
                        "inference_base_url": format!("{base_for_thread}/runtime-inference/v1"),
                        "reused": false
                    }),
                }
                .to_string();
                stream
                    .write_all(http_json_response("200 OK", &body).as_bytes())
                    .unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_minimax_oauth_server() -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = requests.clone();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                requests_clone.lock().unwrap().push(request.clone());
                let state = request
                    .split("\r\n\r\n")
                    .nth(1)
                    .and_then(|body| {
                        body.split('&').find_map(|part| {
                            let (key, value) = part.split_once('=')?;
                            (key == "state").then(|| value.to_string())
                        })
                    })
                    .unwrap_or_else(|| "dashboard-minimax-state".to_string());
                let body = match idx {
                    0 => json!({
                        "user_code": "MINIMAX-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "expired_in": 60,
                        "interval": 100,
                        "state": state
                    }),
                    _ => json!({
                        "status": "success",
                        "access_token": "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pLWRhc2hib2FyZEB0ZXN0In0.sig",
                        "refresh_token": "minimax-refresh",
                        "expired_in": 3600,
                        "token_type": "Bearer",
                        "resource_url": "group-123",
                        "notification_message": "quota synced"
                    }),
                }
                .to_string();
                stream
                    .write_all(http_json_response("200 OK", &body).as_bytes())
                    .unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn seed_session_row(
        state: &DashboardState,
        session_id: &str,
        model: &str,
        billing_provider: &str,
    ) {
        let _store = state.context.open_session_store().unwrap();
        let connection = Connection::open(state.context.state_db_path()).unwrap();
        connection
            .execute(
                "INSERT INTO sessions (
                    id, source, model, started_at, message_count, tool_call_count,
                    input_tokens, output_tokens, cache_read_tokens, reasoning_tokens,
                    billing_provider, estimated_cost_usd, actual_cost_usd, api_call_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    session_id,
                    "cli",
                    model,
                    now_ts(),
                    3_i64,
                    2_i64,
                    120_i64,
                    45_i64,
                    11_i64,
                    7_i64,
                    billing_provider,
                    0.123_f64,
                    0.234_f64,
                    4_i64
                ],
            )
            .unwrap();
    }

    #[test]
    fn accepted_host_matches_loopback_rules() {
        assert!(is_accepted_host("localhost:9119", "127.0.0.1"));
        assert!(is_accepted_host("[::1]:9119", "::1"));
        assert!(!is_accepted_host("evil.example", "127.0.0.1"));
        assert!(is_accepted_host("evil.example", "0.0.0.0"));
    }

    #[test]
    fn schema_flattens_default_config() {
        let defaults = serde_yaml::from_str::<serde_yaml::Value>(DEFAULT_CONFIG_YAML)
            .ok()
            .and_then(yaml_to_json)
            .unwrap();
        let schema = build_config_schema(&defaults);
        let fields = schema.get("fields").and_then(JsonValue::as_object).unwrap();
        assert!(fields.len() > 100);
        assert!(fields.contains_key("model"));
        assert!(fields.contains_key("terminal.backend"));
    }

    #[tokio::test]
    async fn status_route_stays_public() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header(HOST, "localhost:9119")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn protected_route_requires_session_token() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/config")
                    .header(HOST, "localhost:9119")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn pty_websocket_streams_child_output() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let fake_python = temp.path().join("fake-python.sh");
        fs::write(
            &fake_python,
            "#!/bin/sh\nprintf '{\"argv\":[\"/bin/sh\",\"-lc\",\"printf pty-ok; sleep 1\"],\"cwd\":null}\\n'\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = test_state_with_options(temp.path(), true, port);
        let app = app_router(state);
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });

        unsafe {
            std::env::set_var("HERMES_DASHBOARD_PYTHON", &fake_python);
        }
        let (mut socket, _) =
            connect_async(format!("ws://127.0.0.1:{port}/api/pty?token=test-token"))
                .await
                .unwrap();
        let message = socket.next().await.unwrap().unwrap();
        unsafe {
            std::env::remove_var("HERMES_DASHBOARD_PYTHON");
        }
        server.abort();

        let payload = match message {
            tokio_tungstenite::tungstenite::Message::Binary(bytes) => {
                String::from_utf8_lossy(&bytes).to_string()
            }
            tokio_tungstenite::tungstenite::Message::Text(text) => text.to_string(),
            other => panic!("unexpected websocket message: {other:?}"),
        };
        assert!(payload.contains("pty-ok"), "{payload}");
    }

    #[tokio::test]
    async fn config_raw_rejects_non_mapping_yaml() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));
        let response = app
            .oneshot(json_request(
                "PUT",
                "/api/config/raw",
                json!({"yaml_text": "- item"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn env_var_round_trip_supports_reveal() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));

        let set_response = app
            .clone()
            .oneshot(json_request(
                "PUT",
                "/api/env",
                json!({"key": "TEST_REVEAL_KEY", "value": "super-secret-value-12345"}),
            ))
            .await
            .unwrap();
        assert_eq!(set_response.status(), StatusCode::OK);

        let reveal_response = app
            .oneshot(json_request(
                "POST",
                "/api/env/reveal",
                json!({"key": "TEST_REVEAL_KEY"}),
            ))
            .await
            .unwrap();
        assert_eq!(reveal_response.status(), StatusCode::OK);
        let data = response_json(reveal_response).await;
        assert_eq!(
            data.get("key").and_then(JsonValue::as_str),
            Some("TEST_REVEAL_KEY")
        );
        assert_eq!(
            data.get("value").and_then(JsonValue::as_str),
            Some("super-secret-value-12345")
        );
    }

    #[tokio::test]
    async fn profile_setup_command_uses_named_wrapper() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("profiles").join("coder")).unwrap();
        let app = app_router(test_state(temp.path()));
        let response = app
            .oneshot(authed_request("/api/profiles/coder/setup-command"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let data = response_json(response).await;
        assert_eq!(
            data.get("command").and_then(JsonValue::as_str),
            Some("coder setup")
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn profile_open_terminal_uses_override_binary() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("profiles").join("coder")).unwrap();
        let log_path = temp.path().join("terminal.log");
        let fake_terminal = temp.path().join("fake-terminal.sh");
        fs::write(
            &fake_terminal,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$1\" > \"{}\"\n",
                log_path.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_terminal).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_terminal, perms).unwrap();

        unsafe {
            std::env::set_var("HERMES_DASHBOARD_TERMINAL_BIN", &fake_terminal);
        }
        let app = app_router(test_state(temp.path()));
        let response = app
            .oneshot(json_request(
                "POST",
                "/api/profiles/coder/open-terminal",
                json!({}),
            ))
            .await
            .unwrap();
        unsafe {
            std::env::remove_var("HERMES_DASHBOARD_TERMINAL_BIN");
        }
        assert_eq!(response.status(), StatusCode::OK);
        for _ in 0..20 {
            if log_path.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(25));
        }
        assert_eq!(fs::read_to_string(log_path).unwrap().trim(), "coder setup");
    }

    #[tokio::test]
    async fn dashboard_theme_update_persists_in_config() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));
        let response = app
            .clone()
            .oneshot(json_request(
                "PUT",
                "/api/dashboard/theme",
                json!({"name": "ember"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let config_response = app.oneshot(authed_request("/api/config")).await.unwrap();
        let config = response_json(config_response).await;
        assert_eq!(
            config
                .get("dashboard")
                .and_then(JsonValue::as_object)
                .and_then(|dashboard| dashboard.get("theme"))
                .and_then(JsonValue::as_str),
            Some("ember")
        );
    }

    #[tokio::test]
    async fn profile_create_rename_delete_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));

        let create_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/profiles",
                json!({"name": "writer", "clone_from_default": false}),
            ))
            .await
            .unwrap();
        assert_eq!(create_response.status(), StatusCode::OK);
        assert!(temp.path().join("profiles").join("writer").is_dir());

        let rename_response = app
            .clone()
            .oneshot(json_request(
                "PATCH",
                "/api/profiles/writer",
                json!({"new_name": "writer-2"}),
            ))
            .await
            .unwrap();
        assert_eq!(rename_response.status(), StatusCode::OK);
        assert!(temp.path().join("profiles").join("writer-2").is_dir());
        assert!(!temp.path().join("profiles").join("writer").exists());

        let delete_response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/profiles/writer-2")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete_response.status(), StatusCode::OK);
        assert!(!temp.path().join("profiles").join("writer-2").exists());
    }

    #[tokio::test]
    async fn skill_toggle_round_trip_updates_config() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("demo");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: Demo skill\n---\ncontent\n",
        )
        .unwrap();
        let app = app_router(test_state(temp.path()));

        let list_response = app
            .clone()
            .oneshot(authed_request("/api/skills"))
            .await
            .unwrap();
        assert_eq!(list_response.status(), StatusCode::OK);
        let data = response_json(list_response).await;
        let skills = data.as_array().unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(
            skills[0].get("name").and_then(JsonValue::as_str),
            Some("demo")
        );
        assert_eq!(
            skills[0].get("enabled").and_then(JsonValue::as_bool),
            Some(true)
        );

        let toggle_response = app
            .clone()
            .oneshot(json_request(
                "PUT",
                "/api/skills/toggle",
                json!({"name": "demo", "enabled": false}),
            ))
            .await
            .unwrap();
        assert_eq!(toggle_response.status(), StatusCode::OK);

        let list_response = app.oneshot(authed_request("/api/skills")).await.unwrap();
        let data = response_json(list_response).await;
        let skills = data.as_array().unwrap();
        assert_eq!(
            skills[0].get("enabled").and_then(JsonValue::as_bool),
            Some(false)
        );
    }

    #[tokio::test]
    async fn model_assignment_updates_main_and_auxiliary_config() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));

        let main_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/model/set",
                json!({
                    "scope": "main",
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet-4.6"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(main_response.status(), StatusCode::OK);

        let aux_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/model/set",
                json!({
                    "scope": "auxiliary",
                    "task": "vision",
                    "provider": "openai",
                    "model": "gpt-5"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(aux_response.status(), StatusCode::OK);

        let config_response = app.oneshot(authed_request("/api/config")).await.unwrap();
        let config = response_json(config_response).await;
        assert_eq!(
            config
                .get("model")
                .and_then(JsonValue::as_object)
                .and_then(|model| model.get("provider"))
                .and_then(JsonValue::as_str),
            Some("openrouter")
        );
        assert_eq!(
            config
                .get("auxiliary")
                .and_then(JsonValue::as_object)
                .and_then(|aux| aux.get("vision"))
                .and_then(JsonValue::as_object)
                .and_then(|vision| vision.get("provider"))
                .and_then(JsonValue::as_str),
            Some("openai")
        );
    }

    #[tokio::test]
    async fn analytics_routes_report_seeded_state_data() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(temp.path());
        seed_session_row(
            &state,
            "analytics-1",
            "anthropic/claude-sonnet-4.6",
            "openrouter",
        );
        fs::create_dir_all(temp.path().join("skills")).unwrap();
        fs::write(
            temp.path().join("skills").join(".usage.json"),
            serde_json::to_string(&json!({
                "demo-skill": {
                    "use_count": 2,
                    "view_count": 1,
                    "patch_count": 3,
                    "last_used_at": "2026-01-01T00:00:00+00:00"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let app = app_router(state);

        let usage_response = app
            .clone()
            .oneshot(authed_request("/api/analytics/usage?days=7"))
            .await
            .unwrap();
        assert_eq!(usage_response.status(), StatusCode::OK);
        let usage = response_json(usage_response).await;
        assert_eq!(
            usage
                .get("totals")
                .and_then(JsonValue::as_object)
                .and_then(|totals| totals.get("total_sessions"))
                .and_then(JsonValue::as_i64),
            Some(1)
        );
        assert_eq!(
            usage
                .get("skills")
                .and_then(JsonValue::as_object)
                .and_then(|skills| skills.get("summary"))
                .and_then(JsonValue::as_object)
                .and_then(|summary| summary.get("total_skill_actions"))
                .and_then(JsonValue::as_i64),
            Some(6)
        );

        let models_response = app
            .oneshot(authed_request("/api/analytics/models?days=7"))
            .await
            .unwrap();
        assert_eq!(models_response.status(), StatusCode::OK);
        let models = response_json(models_response).await;
        assert_eq!(
            models
                .get("models")
                .and_then(JsonValue::as_array)
                .map(|items| items.len()),
            Some(1)
        );
        assert_eq!(
            models
                .get("totals")
                .and_then(JsonValue::as_object)
                .and_then(|totals| totals.get("distinct_models"))
                .and_then(JsonValue::as_i64),
            Some(1)
        );
    }

    #[tokio::test]
    async fn oauth_provider_routes_list_status_and_disconnect() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join(".claude")).unwrap();
        fs::create_dir_all(temp.path().join(".qwen")).unwrap();
        fs::write(
            temp.path().join(".anthropic_oauth.json"),
            serde_json::to_string(&json!({
                "accessToken": "header.payload.hermespkce",
                "refreshToken": "anthropic-refresh",
                "expiresAt": 4_102_444_800_000_i64
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            temp.path().join(".claude").join(".credentials.json"),
            serde_json::to_string(&json!({
                "claudeAiOauth": {
                    "accessToken": "header.payload.claudecode",
                    "refreshToken": "claude-refresh",
                    "expiresAt": 4_102_444_800_000_i64
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            temp.path().join(".qwen").join("oauth_creds.json"),
            serde_json::to_string(&json!({
                "access_token": "header.payload.qwenaccess",
                "refresh_token": "qwen-refresh",
                "expiry_date": 4_102_444_800_000_i64
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            temp.path().join("auth.json"),
            serde_json::to_string_pretty(&json!({
                "version": 1,
                "providers": {
                    "anthropic": {
                        "tokens": {
                            "access_token": "anthropic-store"
                        }
                    },
                    "openai-codex": {
                        "auth_mode": "chatgpt",
                        "tokens": {
                            "access_token": "header.eyJlbWFpbCI6ImNvZGV4QGV4YW1wbGUuY29tIn0.codexsig",
                            "refresh_token": "codex-refresh"
                        },
                        "last_refresh": "2026-05-08T00:00:00Z"
                    },
                    "nous": {
                        "access_token": "header.payload.nousaccess",
                        "refresh_token": "nous-refresh",
                        "portal_base_url": "https://portal.nous.test",
                        "expires_at": "2999-01-01T00:00:00Z"
                    },
                    "qwen-oauth": {
                        "access_token": "stored-qwen"
                    },
                    "minimax-oauth": {
                        "access_token": "header.payload.minimaxaccess",
                        "refresh_token": "minimax-refresh",
                        "portal_base_url": "https://api.minimax.io",
                        "expires_at": "2999-01-01T00:00:00Z"
                    }
                },
                "credential_pool": {
                    "anthropic": [{"source": "hermes_pkce"}],
                    "qwen-oauth": [{"source": "manual:qwen_cli"}]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let app = app_router(test_state(temp.path()));

        let list_response = app
            .clone()
            .oneshot(authed_request("/api/providers/oauth"))
            .await
            .unwrap();
        assert_eq!(list_response.status(), StatusCode::OK);
        let payload = response_json(list_response).await;
        let providers = payload
            .get("providers")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap();
        assert_eq!(providers.len(), 6);

        let anthropic = providers
            .iter()
            .find(|row| row.get("id").and_then(JsonValue::as_str) == Some("anthropic"))
            .unwrap();
        assert_eq!(
            anthropic
                .get("status")
                .and_then(JsonValue::as_object)
                .and_then(|status| status.get("source"))
                .and_then(JsonValue::as_str),
            Some("hermes_pkce")
        );

        let claude_code = providers
            .iter()
            .find(|row| row.get("id").and_then(JsonValue::as_str) == Some("claude-code"))
            .unwrap();
        assert_eq!(
            claude_code
                .get("status")
                .and_then(JsonValue::as_object)
                .and_then(|status| status.get("source"))
                .and_then(JsonValue::as_str),
            Some("claude_code_cli")
        );

        let codex = providers
            .iter()
            .find(|row| row.get("id").and_then(JsonValue::as_str) == Some("openai-codex"))
            .unwrap();
        assert_eq!(
            codex
                .get("status")
                .and_then(JsonValue::as_object)
                .and_then(|status| status.get("last_refresh"))
                .and_then(JsonValue::as_str),
            Some("2026-05-08T00:00:00Z")
        );

        let qwen = providers
            .iter()
            .find(|row| row.get("id").and_then(JsonValue::as_str) == Some("qwen-oauth"))
            .unwrap();
        assert_eq!(
            qwen.get("docs_url").and_then(JsonValue::as_str),
            Some("https://github.com/QwenLM/qwen-code")
        );
        assert_eq!(
            qwen.get("status")
                .and_then(JsonValue::as_object)
                .and_then(|status| status.get("logged_in"))
                .and_then(JsonValue::as_bool),
            Some(true)
        );

        let disconnect_anthropic = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/providers/oauth/anthropic")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disconnect_anthropic.status(), StatusCode::OK);
        assert!(!temp.path().join(".anthropic_oauth.json").exists());

        let disconnect_qwen = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/providers/oauth/qwen-oauth")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disconnect_qwen.status(), StatusCode::OK);
        assert!(!temp.path().join(".qwen").join("oauth_creds.json").exists());

        let auth_store: JsonValue =
            serde_json::from_str(&fs::read_to_string(temp.path().join("auth.json")).unwrap())
                .unwrap();
        assert!(
            auth_store["providers"]
                .as_object()
                .is_some_and(|providers| !providers.contains_key("anthropic"))
        );
        assert!(
            auth_store["providers"]
                .as_object()
                .is_some_and(|providers| !providers.contains_key("qwen-oauth"))
        );
    }

    #[tokio::test]
    async fn oauth_anthropic_pkce_routes_round_trip_and_cancel() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let (token_url, requests, server) = spawn_anthropic_token_server();
        unsafe {
            std::env::set_var("HERMES_AUTH_ANTHROPIC_TOKEN_URL", &token_url);
            std::env::set_var("HERMES_AUTH_ANTHROPIC_TEST_VERIFIER", "dashboard-verifier");
        }

        let app = app_router(test_state(temp.path()));
        let start_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/providers/oauth/anthropic/start")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(start_response.status(), StatusCode::OK);
        let start = response_json(start_response).await;
        let session_id = start
            .get("session_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();
        assert_eq!(start.get("flow").and_then(JsonValue::as_str), Some("pkce"));

        let submit_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/providers/oauth/anthropic/submit",
                json!({"session_id": session_id, "code": "code-123"}),
            ))
            .await
            .unwrap();
        assert_eq!(submit_response.status(), StatusCode::OK);
        let submit = response_json(submit_response).await;
        assert_eq!(submit.get("ok").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(
            submit.get("status").and_then(JsonValue::as_str),
            Some("approved")
        );

        let oauth_file = temp.path().join(".anthropic_oauth.json");
        assert!(oauth_file.exists());
        let auth_store: JsonValue =
            serde_json::from_str(&fs::read_to_string(temp.path().join("auth.json")).unwrap())
                .unwrap();
        assert_eq!(
            auth_store["credential_pool"]["anthropic"][0]["source"],
            "manual:hermes_pkce"
        );
        assert!(
            requests
                .lock()
                .unwrap()
                .first()
                .is_some_and(|request| request.contains("code-123"))
        );

        let cancel_start = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/providers/oauth/anthropic/start")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cancel_start = response_json(cancel_start).await;
        let cancel_id = cancel_start
            .get("session_id")
            .and_then(JsonValue::as_str)
            .unwrap();
        let cancel_response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/providers/oauth/sessions/{cancel_id}"))
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancel_response.status(), StatusCode::OK);
        let cancel = response_json(cancel_response).await;
        assert_eq!(cancel.get("ok").and_then(JsonValue::as_bool), Some(true));

        unsafe {
            std::env::remove_var("HERMES_AUTH_ANTHROPIC_TOKEN_URL");
            std::env::remove_var("HERMES_AUTH_ANTHROPIC_TEST_VERIFIER");
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oauth_codex_device_code_routes_round_trip() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let (issuer, requests, server) = spawn_codex_oauth_server();
        unsafe {
            std::env::set_var("HERMES_AUTH_CODEX_ISSUER", &issuer);
            std::env::set_var(
                "HERMES_AUTH_CODEX_TOKEN_URL",
                format!("{issuer}/oauth/token"),
            );
        }

        let app = app_router(test_state(temp.path()));
        let start_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/providers/oauth/openai-codex/start")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(start_response.status(), StatusCode::OK);
        let start = response_json(start_response).await;
        assert_eq!(
            start.get("flow").and_then(JsonValue::as_str),
            Some("device_code")
        );
        let session_id = start
            .get("session_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();

        let mut approved = false;
        for _ in 0..20 {
            let poll_response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!(
                            "/api/providers/oauth/openai-codex/poll/{session_id}"
                        ))
                        .header(HOST, "localhost:9119")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(poll_response.status(), StatusCode::OK);
            let poll = response_json(poll_response).await;
            let status = poll.get("status").and_then(JsonValue::as_str).unwrap();
            if status == "approved" {
                approved = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(approved, "codex session never reached approved");

        let auth_store: JsonValue =
            serde_json::from_str(&fs::read_to_string(temp.path().join("auth.json")).unwrap())
                .unwrap();
        assert_eq!(
            auth_store["providers"]["openai-codex"]["auth_mode"],
            "chatgpt"
        );
        assert_eq!(
            auth_store["credential_pool"]["openai-codex"][0]["source"],
            "manual:device_code"
        );
        let captured = requests.lock().unwrap().clone();
        assert!(
            captured
                .iter()
                .any(|line| line.contains("/deviceauth/usercode"))
        );
        assert!(
            captured
                .iter()
                .any(|line| line.contains("/deviceauth/token"))
        );
        assert!(captured.iter().any(|line| line.contains("/oauth/token")));

        unsafe {
            std::env::remove_var("HERMES_AUTH_CODEX_ISSUER");
            std::env::remove_var("HERMES_AUTH_CODEX_TOKEN_URL");
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oauth_nous_device_code_routes_round_trip() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let (portal_base_url, requests, server) = spawn_nous_oauth_server();
        unsafe {
            std::env::set_var("HERMES_PORTAL_BASE_URL", &portal_base_url);
        }

        let app = app_router(test_state(temp.path()));
        let start_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/providers/oauth/nous/start")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(start_response.status(), StatusCode::OK);
        let start = response_json(start_response).await;
        let session_id = start
            .get("session_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();

        let mut approved = false;
        for _ in 0..30 {
            let poll_response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/api/providers/oauth/nous/poll/{session_id}"))
                        .header(HOST, "localhost:9119")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let poll = response_json(poll_response).await;
            if poll.get("status").and_then(JsonValue::as_str) == Some("approved") {
                approved = true;
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(approved, "nous session never reached approved");

        let auth_store: JsonValue =
            serde_json::from_str(&fs::read_to_string(temp.path().join("auth.json")).unwrap())
                .unwrap();
        assert_eq!(
            auth_store["credential_pool"]["nous"][0]["source"],
            "device_code"
        );
        assert_eq!(
            auth_store["providers"]["nous"]["agent_key"],
            "nous-agent-key"
        );
        assert!(
            temp.path()
                .join(".hermes")
                .join("shared")
                .join("nous_auth.json")
                .exists()
        );
        let captured = requests.lock().unwrap().clone();
        assert!(
            captured
                .iter()
                .any(|request| request.contains("/api/oauth/device/code"))
        );
        assert!(
            captured
                .iter()
                .any(|request| request.contains("/api/oauth/token"))
        );
        assert!(
            captured
                .iter()
                .any(|request| request.contains("/api/oauth/agent-key"))
        );

        unsafe {
            std::env::remove_var("HERMES_PORTAL_BASE_URL");
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oauth_minimax_device_code_routes_round_trip() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let (portal_base_url, requests, server) = spawn_minimax_oauth_server();
        unsafe {
            std::env::set_var("HERMES_AUTH_MINIMAX_PORTAL_URL", &portal_base_url);
        }

        let app = app_router(test_state(temp.path()));
        let start_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/providers/oauth/minimax-oauth/start")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(start_response.status(), StatusCode::OK);
        let start = response_json(start_response).await;
        let session_id = start
            .get("session_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();

        let mut approved = false;
        for _ in 0..30 {
            let poll_response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!(
                            "/api/providers/oauth/minimax-oauth/poll/{session_id}"
                        ))
                        .header(HOST, "localhost:9119")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let poll = response_json(poll_response).await;
            if poll.get("status").and_then(JsonValue::as_str) == Some("approved") {
                approved = true;
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(approved, "minimax session never reached approved");

        let auth_store: JsonValue =
            serde_json::from_str(&fs::read_to_string(temp.path().join("auth.json")).unwrap())
                .unwrap();
        assert_eq!(
            auth_store["credential_pool"]["minimax-oauth"][0]["source"],
            "manual:minimax_oauth"
        );
        assert_eq!(
            auth_store["providers"]["minimax-oauth"]["portal_base_url"],
            portal_base_url
        );
        let captured = requests.lock().unwrap().clone();
        assert!(
            captured
                .iter()
                .any(|request| request.contains("/oauth/code"))
        );
        assert!(
            captured
                .iter()
                .any(|request| request.contains("/oauth/token"))
        );

        unsafe {
            std::env::remove_var("HERMES_AUTH_MINIMAX_PORTAL_URL");
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn logs_route_filters_by_level_and_component() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("logs")).unwrap();
        fs::write(
            temp.path().join("logs").join("agent.log"),
            concat!(
                "2026-01-01 00:00:00 INFO agent: startup ok\n",
                "2026-01-01 00:00:01 WARNING tools.runner: tool warning\n",
                "2026-01-01 00:00:02 ERROR gateway.run: gateway failed\n",
            ),
        )
        .unwrap();
        let app = app_router(test_state(temp.path()));

        let response = app
            .oneshot(authed_request(
                "/api/logs?file=agent&lines=50&level=WARNING&component=tools",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let payload = response_json(response).await;
        let lines = payload.get("lines").and_then(JsonValue::as_array).unwrap();
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0]
                .as_str()
                .is_some_and(|line| line.contains("tool warning"))
        );
    }

    #[tokio::test]
    async fn plugins_hub_routes_support_visibility_and_provider_persistence() {
        let temp = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("HERMES_BUNDLED_PLUGINS", temp.path().join("plugins"));
        }
        let bundled_plugin = temp.path().join("plugins").join("demo");
        fs::create_dir_all(bundled_plugin.join("dashboard")).unwrap();
        fs::write(
            bundled_plugin.join("plugin.yaml"),
            concat!(
                "name: demo\n",
                "version: 1.0.0\n",
                "description: Demo plugin\n",
                "requires_env:\n",
                "  - DEMO_TOKEN\n",
            ),
        )
        .unwrap();
        fs::write(
            bundled_plugin.join("dashboard").join("manifest.json"),
            r#"{"name":"demo","label":"Demo","description":"Demo tab","icon":"Puzzle","version":"1.0.0","tab":{"path":"/demo"},"entry":"dist/index.js","has_api":false}"#,
        )
        .unwrap();
        fs::create_dir_all(temp.path().join("plugins").join("memx")).unwrap();
        fs::write(
            temp.path().join("plugins").join("memx").join("__init__.py"),
            "def register_memory_provider():\n    pass\n",
        )
        .unwrap();
        fs::write(
            temp.path().join("plugins").join("memx").join("plugin.yaml"),
            "description: Demo memory provider\n",
        )
        .unwrap();
        fs::create_dir_all(temp.path().join("plugins").join("ctxx")).unwrap();
        fs::write(
            temp.path().join("plugins").join("ctxx").join("__init__.py"),
            "class ContextEngine:\n    pass\n",
        )
        .unwrap();
        fs::write(
            temp.path().join("plugins").join("ctxx").join("plugin.yaml"),
            "description: Demo context engine\n",
        )
        .unwrap();

        let app = app_router(test_state(temp.path()));

        let hub_response = app
            .clone()
            .oneshot(authed_request("/api/dashboard/plugins/hub"))
            .await
            .unwrap();
        assert_eq!(hub_response.status(), StatusCode::OK);
        let hub = response_json(hub_response).await;
        let demo_row = hub
            .get("plugins")
            .and_then(JsonValue::as_array)
            .and_then(|rows| {
                rows.iter()
                    .find(|row| row.get("name").and_then(JsonValue::as_str) == Some("demo"))
            })
            .cloned()
            .unwrap();
        assert_eq!(
            demo_row.get("name").and_then(JsonValue::as_str),
            Some("demo")
        );
        assert_eq!(
            demo_row.get("auth_required").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(
            hub.get("providers")
                .and_then(JsonValue::as_object)
                .and_then(|providers| providers.get("memory_options"))
                .and_then(JsonValue::as_array)
                .is_some_and(|items| items
                    .iter()
                    .any(|item| item.get("name").and_then(JsonValue::as_str) == Some("memx")))
        );

        let visibility_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/dashboard/plugins/demo/visibility",
                json!({"hidden": true}),
            ))
            .await
            .unwrap();
        assert_eq!(visibility_response.status(), StatusCode::OK);

        let list_response = app
            .clone()
            .oneshot(authed_request("/api/dashboard/plugins"))
            .await
            .unwrap();
        assert_eq!(list_response.status(), StatusCode::OK);
        let manifests = response_json(list_response).await;
        assert_eq!(manifests.as_array().map(|items| items.len()), Some(0));

        let provider_response = app
            .oneshot(json_request(
                "PUT",
                "/api/dashboard/plugin-providers",
                json!({"memory_provider": "memx", "context_engine": "ctxx"}),
            ))
            .await
            .unwrap();
        assert_eq!(provider_response.status(), StatusCode::OK);
        let config = fs::read_to_string(temp.path().join("config.yaml")).unwrap();
        assert!(config.contains("provider: memx"));
        assert!(config.contains("engine: ctxx"));
        unsafe {
            std::env::remove_var("HERMES_BUNDLED_PLUGINS");
        }
    }

    #[tokio::test]
    async fn agent_plugin_routes_install_update_disable_and_remove() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("demo-plugin");
        init_plugin_repo(
            &repo,
            concat!(
                "name: demo-user\n",
                "version: 1.0.0\n",
                "description: demo plugin\n",
                "requires_env:\n",
                "  - DEMO_TOKEN\n",
            ),
        );

        let app = app_router(test_state(temp.path()));

        let install_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/dashboard/agent-plugins/install",
                json!({
                    "identifier": format!("file://{}", repo.display()),
                    "force": false,
                    "enable": true
                }),
            ))
            .await
            .unwrap();
        assert_eq!(install_response.status(), StatusCode::OK);
        let install = response_json(install_response).await;
        assert_eq!(
            install.get("plugin_name").and_then(JsonValue::as_str),
            Some("demo-user")
        );
        assert!(
            install
                .get("missing_env")
                .and_then(JsonValue::as_array)
                .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("DEMO_TOKEN")))
        );

        fs::write(
            repo.join("plugin.yaml"),
            "name: demo-user\nversion: 2.0.0\ndescription: updated plugin\n",
        )
        .unwrap();
        run_git(&repo, &["add", "."]);
        run_git(&repo, &["commit", "-m", "update plugin"]);

        let update_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/dashboard/agent-plugins/demo-user/update",
                json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(update_response.status(), StatusCode::OK);
        let update = response_json(update_response).await;
        assert_eq!(
            update.get("unchanged").and_then(JsonValue::as_bool),
            Some(false)
        );

        let disable_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/dashboard/agent-plugins/demo-user/disable",
                json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(disable_response.status(), StatusCode::OK);

        let hub_response = app
            .clone()
            .oneshot(authed_request("/api/dashboard/plugins/hub"))
            .await
            .unwrap();
        let hub = response_json(hub_response).await;
        assert_eq!(
            hub.get("plugins")
                .and_then(JsonValue::as_array)
                .and_then(|rows| rows
                    .iter()
                    .find(|row| row.get("name").and_then(JsonValue::as_str) == Some("demo-user")))
                .and_then(|row| row.get("runtime_status"))
                .and_then(JsonValue::as_str),
            Some("disabled")
        );

        let remove_response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/dashboard/agent-plugins/demo-user")
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(remove_response.status(), StatusCode::OK);
        assert!(!temp.path().join("plugins").join("demo-user").exists());
    }

    #[tokio::test]
    async fn cron_routes_create_list_and_mutate_jobs() {
        let temp = tempfile::tempdir().unwrap();
        let app = app_router(test_state(temp.path()));

        let create_response = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/api/cron/jobs",
                json!({
                    "prompt": "Ping",
                    "schedule": "30m",
                    "name": "demo-job",
                    "deliver": "local"
                }),
            ))
            .await
            .unwrap();
        let create_status = create_response.status();
        let created = response_json(create_response).await;
        assert_eq!(create_status, StatusCode::OK, "{created}");
        assert_eq!(
            created.get("prompt").and_then(JsonValue::as_str),
            Some("Ping")
        );
        let job_id = created
            .get("id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .to_string();

        let list_response = app
            .clone()
            .oneshot(authed_request("/api/cron/jobs"))
            .await
            .unwrap();
        assert_eq!(list_response.status(), StatusCode::OK);
        let jobs = response_json(list_response).await;
        assert_eq!(jobs.as_array().map(|items| items.len()), Some(1));

        let get_response = app
            .clone()
            .oneshot(authed_request(&format!("/api/cron/jobs/{job_id}")))
            .await
            .unwrap();
        assert_eq!(get_response.status(), StatusCode::OK);
        let job = response_json(get_response).await;
        assert_eq!(
            job.get("id").and_then(JsonValue::as_str),
            Some(job_id.as_str())
        );

        let update_response = app
            .clone()
            .oneshot(json_request(
                "PUT",
                &format!("/api/cron/jobs/{job_id}"),
                json!({"updates": {"name": "demo-job-2"}}),
            ))
            .await
            .unwrap();
        assert_eq!(update_response.status(), StatusCode::OK);
        let updated = response_json(update_response).await;
        assert_eq!(
            updated.get("name").and_then(JsonValue::as_str),
            Some("demo-job-2")
        );

        for path in [
            format!("/api/cron/jobs/{job_id}/pause"),
            format!("/api/cron/jobs/{job_id}/resume"),
            format!("/api/cron/jobs/{job_id}/trigger"),
        ] {
            let response = app
                .clone()
                .oneshot(json_request("POST", &path, json!({})))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let delete_response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/cron/jobs/{job_id}"))
                    .header(HOST, "localhost:9119")
                    .header(SESSION_HEADER_NAME, "test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete_response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn action_routes_spawn_and_report_status() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let bin = temp.path().join("fake-action.sh");
        fs::write(&bin, "#!/bin/sh\necho action-ok\n").unwrap();
        let mut perms = fs::metadata(&bin).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).unwrap();

        unsafe {
            std::env::set_var("HERMES_DASHBOARD_ACTION_BIN", &bin);
        }
        ACTION_PROCESSES.lock().unwrap().clear();
        let app = app_router(test_state(temp.path()));

        let spawn_response = app
            .clone()
            .oneshot(json_request("POST", "/api/gateway/restart", json!({})))
            .await
            .unwrap();
        assert_eq!(spawn_response.status(), StatusCode::OK);
        thread::sleep(Duration::from_millis(50));

        let status_response = app
            .oneshot(authed_request(
                "/api/actions/gateway-restart/status?lines=20",
            ))
            .await
            .unwrap();
        assert_eq!(status_response.status(), StatusCode::OK);
        let data = response_json(status_response).await;
        assert_eq!(
            data.get("name").and_then(JsonValue::as_str),
            Some("gateway-restart")
        );
        assert!(data.get("pid").and_then(JsonValue::as_u64).is_some());
        assert!(
            data.get("lines")
                .and_then(JsonValue::as_array)
                .is_some_and(|lines| lines
                    .iter()
                    .any(|line| line.as_str().is_some_and(|text| text.contains("action-ok"))))
        );

        unsafe {
            std::env::remove_var("HERMES_DASHBOARD_ACTION_BIN");
        }
    }
}
